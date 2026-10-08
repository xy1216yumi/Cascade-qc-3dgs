"""核心检查器：单张渲染图 -> 结构化 JSON（SceneCheck）。

这是"VLM 质量闭环"的最小 agent 骨架，包含两层：

1) 结构化输出：Kimi K3 通过 langchain 的 with_structured_output 直接输出
   SceneCheck（Pydantic schema），失败时自动降级为 JSON mode 手动解析。

2) 反思闭环（REFLECT=True 时）：第一轮输出经过自洽性检查
   （如 "clean=True 但 split/sticky=True" 这类矛盾、低置信度、字段缺失），
   把矛盾点反馈给模型做第二轮复查，取第二轮结果。
   这就是论文里"检测器发现失败 -> 局部重跑"机制的最小原型。

运行方式（在 vlm_checker 目录下）：
    python checker.py sample/image_01.png
"""
import base64
import json
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import config
from schema import SceneCheck

# Windows 控制台中文输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SYSTEM_PROMPT = """你是一位严格的 3D 实例分割质检员。你会收到一张"渲染图"：深色实心的物体上
叠加了半透明的彩色 mask（不同实例不同颜色，中心有数字编号）。mask 是半透明的，
你可以透过它看到下面的物体。你的任务是对图上每一个编号实例做三项质量判断，并输出结构化结果。

判断三步法（务必按顺序执行）：
第 1 步：透过每个编号的 mask，数一数它覆盖了几个独立物体。
   - 只数"被色块半透明区域覆盖住"的物体；色块外部边缘的游离小块（同色小碎片）是杂块，不是物体。
   - 色块覆盖了 2 个或更多物体（比如两个杯子叠在一起、两个物体挨在一起被同一色块盖住）
     -> is_sticky=True，is_clean=False。
第 2 步：如果只覆盖 1 个物体，看这个 mask 的主体是否连通：
   - mask 主体被切成互不连通的几块（同一色块分成上下/左右两块、中间有缺口）-> is_split=True，is_clean=False。
   - 主体连通但旁边有游离的小碎片/小块 -> 这是"杂块"：is_clean=False，is_split=False，is_sticky=False。
第 3 步：如果 mask 主体连通且无杂块，看边界：
   - mask 边界完全贴合物体轮廓 -> is_clean=True。
   - mask 比物体大一圈、明显偏移、有大块空洞 -> is_clean=False（不贴合，但不算分裂/粘连）。

规则：
1. is_clean=True 时，is_split 和 is_sticky 必须都为 False。
2. instance_id 必须与图上的编号一一对应，一个实例一行。
3. confidence 是你对自己判断的把握（0~1）。
4. reason 用中文，不超过 30 字，必须写出判断依据（如"色块覆盖两个物体"、"主体切成两块"、"主体旁有游离碎片"）。
"""

USER_PROMPT_TEMPLATE = """请检查这张渲染图中每个编号实例的 mask 质量，按规则输出 JSON。{extra}"""


def _img_to_data_uri(path: str) -> str:
    """本地图片 -> base64 data URI（避免依赖公网 URL）。"""
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "image/png")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def build_llm() -> ChatOpenAI:
    err = config.check_config()
    if err:
        raise RuntimeError(err)
    return ChatOpenAI(
        model=config.KIMI_MODEL,
        api_key=config.KIMI_API_KEY,
        base_url=config.KIMI_BASE_URL,
        temperature=config.KIMI_TEMPERATURE,
        timeout=config.get_int("KIMI_TIMEOUT", 300),
    )


def _call_structured(llm: ChatOpenAI, image_path: str, extra: str = "") -> SceneCheck:
    """第一优先：function_calling 结构化输出。失败抛异常由上层降级。"""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": USER_PROMPT_TEMPLATE.format(extra=extra)},
                {"type": "image_url", "image_url": {"url": _img_to_data_uri(image_path)}},
            ]
        ),
    ]
    structured = llm.with_structured_output(SceneCheck, method="function_calling")
    return structured.invoke(messages)


def _call_json_mode(llm: ChatOpenAI, image_path: str, extra: str = "") -> SceneCheck:
    """降级路径：JSON mode + 手动解析校验。"""
    messages = [
        SystemMessage(content=SYSTEM_PROMPT + "\n你必须只输出合法 JSON，不要输出任何其他文字或 markdown 代码块。"),
        HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": USER_PROMPT_TEMPLATE.format(extra=extra)
                    + '\n输出格式示例：{"instances": [{"instance_id": 1, "is_clean": true, "is_split": false, "is_sticky": false, "confidence": 0.9, "reason": "边界贴合"}]}',
                },
                {"type": "image_url", "image_url": {"url": _img_to_data_uri(image_path)}},
            ]
        ),
    ]
    resp = llm.invoke(messages)
    text = resp.content if isinstance(resp.content, str) else json.dumps(resp.content, ensure_ascii=False)
    # 去掉可能的 ```json 围栏
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    data = json.loads(text)
    return SceneCheck(**data)


def _call(llm: ChatOpenAI, image_path: str, extra: str = "", verbose: bool = True):
    """按配置选择调用路径。

    kimi-k3 不支持 function_calling（每次必 400），默认直接走 JSON mode；
    若在 .env 设 KIMI_PREFER_FC=1，则优先 function_calling、失败自动降级。
    返回 (SceneCheck, used_json_mode)。
    """
    if config.KIMI_PREFER_FC:
        try:
            return _call_structured(llm, image_path, extra=extra), False
        except Exception as e:
            if verbose:
                print(f"  [降级] function_calling 失败({type(e).__name__})，改用 JSON mode")
            return _call_json_mode(llm, image_path, extra=extra), True
    return _call_json_mode(llm, image_path, extra=extra), True


def _has_conflict(check: SceneCheck) -> str:
    """自洽性检查，返回冲突描述；无冲突返回空串。

    对应论文中"检测器发现质量问题"的最小规则集：
    1) clean=True 却 split/sticky=True（自相矛盾）
    2) 低置信度（<0.5）
    3) 没有输出任何实例（疑似漏检）
    """
    problems = []
    for inst in check.instances:
        if inst.is_clean and (inst.is_split or inst.is_sticky):
            problems.append(f"实例{inst.instance_id}: 判定为干净但又标记了分裂/粘连，自相矛盾")
        if inst.confidence < 0.5:
            problems.append(f"实例{inst.instance_id}: 置信度过低({inst.confidence:.2f})")
    if not check.instances:
        problems.append("未输出任何实例，疑似漏检")
    return "；".join(problems)


def check_image(image_path: str, reflect: bool = True, verbose: bool = True) -> dict:
    """对一张渲染图执行检查（含反思闭环）。

    返回 dict：{image, instances(list[dict]), reflect_used, conflict, raw1, raw2}
    """
    llm = build_llm()

    # 第一轮：按配置选择调用路径（默认 JSON mode，避免无谓的失败调用）
    first, used_json_mode = _call(llm, image_path, verbose=verbose)

    conflict = _has_conflict(first)
    reflect_used = False

    if reflect and conflict:
        if verbose:
            print(f"  [反思] 第一轮发现冲突: {conflict}，请求模型复查")
        extra = (
            f"上一轮你的判断存在以下问题：{conflict}。"
            "请重新仔细看图，修正矛盾之处，只输出最终 JSON。"
        )
        second, _ = _call(llm, image_path, extra=extra, verbose=verbose)
        final = second
        reflect_used = True
        raw2 = second.model_dump()
    else:
        final = first
        raw2 = None

    return {
        "image": image_path,
        "instances": [i.model_dump() for i in final.instances],
        "reflect_used": reflect_used,
        "conflict": conflict,
        "json_mode": used_json_mode,
        "raw2": raw2,
    }


if __name__ == "__main__":
    import os

    if len(sys.argv) < 2:
        print("用法: python checker.py <图片路径>")
        sys.exit(1)
    img = sys.argv[1]
    if not os.path.exists(img):
        print(f"图片不存在: {img}")
        sys.exit(1)
    result = check_image(img)
    print(json.dumps(result, ensure_ascii=False, indent=2))