"""VLM 语义层接入级联：对升级到 Stage2 的疑难实例，让 VLM 看两个视角做语义仲裁。

设计动机（README 定调"规则管几何、VLM 管语义"）：
  Stage2 几何给出一致性分数但给不出"为什么"；VLM 有语义理解但单帧有盲区。
  本实验把 VLM 放在级联的语义层：输入 = 视角 A 的参照 mask 叠加图 + 视角 B 的
  待检 mask 叠加图（两图同实例同编号），输出缺陷类型 + 自然语言解释。
  评估：VLM 语义判定 vs 几何判定 vs 真值，以及"几何 OR VLM"组合的准确率。

数据：复用 cascade_eval_all 的注入缺陷实例（仅取升级到 Stage2 的子集，
即 results/cascade_all_results.csv 中 s2_score 非空的行；缺陷注入是确定性的，
按同参数重新生成，与级联评测的 mask 一致）。

用法：python vlm_semantic_layer.py [--limit N] [--out-tag ""]
输出：results/vlm_semantic_layer{tag}.csv + results/vlm_semantic_report{tag}.txt
"""
import argparse, csv, json, os, sys, time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import checker  # noqa: E402
from mv_checker import load_gray  # noqa: E402
from cascade_checker import inject_defect  # noqa: E402
from prepare_real_scene import draw_overlay  # noqa: E402

SCENE_DIR = "data/real_data/test"

PROMPT = """你是实例分割质检员。给你同一物体在两个相机视角下的分割结果：
- 图1（视角 A，参照）：绿色半透明区域是该物体的 mask，中心数字是实例编号；
- 图2（视角 B，待检）：红色半透明区域是同一物体在另一帧的 mask。
请判断图2 的 mask 属于哪一类：
  clean  = 完整贴合该物体；
  split  = 同一物体被切成多块/缺了一块；
  dirty  = mask 里混入了不属于该物体的游离碎块；
  sticky = mask 把旁边另一个物体也圈了进来（两个物体连成一块）。
拿不准就答 uncertain。
只输出合法 JSON：{"defect": "clean|split|dirty|sticky|uncertain", "confidence": 0.0-1.0, "reason": "一句话中文解释"}"""


def vlm_judge(cfg, imgA_path, imgB_path):
    """直连 Kimi API（OpenAI 兼容 /chat/completions），绕开 langchain 的 httpx 兼容问题。"""
    import urllib.request
    body = {
        "model": cfg.KIMI_MODEL,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": "请判断图2 的 mask 缺陷类型。"},
                {"type": "image_url", "image_url": {"url": checker._img_to_data_uri(imgA_path)}},
                {"type": "image_url", "image_url": {"url": checker._img_to_data_uri(imgB_path)}},
            ]},
        ],
        "temperature": 1,
    }
    req = urllib.request.Request(
        cfg.KIMI_BASE_URL.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {cfg.KIMI_API_KEY}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=cfg.get_int("KIMI_TIMEOUT", 300)) as r:
        d = json.loads(r.read())
    text = d["choices"][0]["message"]["content"].strip().strip("`")
    if text.startswith("json"):
        text = text[4:].strip()
    try:
        d = json.loads(text)
        return d.get("defect", "uncertain"), float(d.get("confidence", 0.5)), d.get("reason", "")
    except Exception:
        return "uncertain", 0.0, f"解析失败: {text[:80]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-tag", default="")
    args = ap.parse_args()

    casc = list(csv.DictReader(open("results/cascade_all_results.csv", encoding="utf-8-sig")))
    rows = [r for r in casc if r["s2_score"] != ""]  # 升级到 Stage2 的疑难实例
    if args.limit:
        rows = rows[:args.limit]
    print(f"Stage2 疑难实例 {len(rows)} 个，逐例调用 VLM 语义仲裁...", flush=True)

    cfg_err = checker.config.check_config()
    if cfg_err:
        raise RuntimeError(cfg_err)
    cfg = checker.config
    os.makedirs("results", exist_ok=True)
    tmp_dir = "results/_vlm_tmp"
    os.makedirs(tmp_dir, exist_ok=True)

    out_rows = []
    for n, r in enumerate(rows):
        scene, fa, fb, idx = r["scene"], r["frame_a"], r["frame_b"], int(r["inst_idx"])
        kind = r["defect"]
        sd = os.path.join(SCENE_DIR, scene)
        mdir = os.path.join(sd, "mask_visib")
        gt = json.load(open(os.path.join(sd, "scene_gt.json")))
        n_inst = len(gt[fa])

        maskA = load_gray(os.path.join(mdir, f"{int(fa):06d}_{idx:06d}.png"))
        maskB_gt = load_gray(os.path.join(mdir, f"{int(fb):06d}_{idx:06d}.png"))
        maskB = inject_defect(maskB_gt, kind, mdir, fb, idx, n_inst, *maskB_gt.shape)

        rgbA = Image.open(os.path.join(sd, "rgb", f"{int(fa):06d}.png")).convert("RGB")
        rgbB = Image.open(os.path.join(sd, "rgb", f"{int(fb):06d}.png")).convert("RGB")
        ovA = draw_overlay(rgbA, [(idx, maskA.astype(np.uint8) * 255, (30, 200, 60))])
        ovB = draw_overlay(rgbB, [(idx, maskB.astype(np.uint8) * 255, (230, 40, 40))])
        pA = os.path.join(tmp_dir, f"{n}_A.png")
        pB = os.path.join(tmp_dir, f"{n}_B.png")
        ovA.save(pA); ovB.save(pB)

        try:
            defect_v, conf, reason = vlm_judge(cfg, pA, pB)
        except Exception as e:
            time.sleep(5)
            try:
                defect_v, conf, reason = vlm_judge(cfg, pA, pB)
            except Exception as e2:
                defect_v, conf, reason = "uncertain", 0.0, f"API 错误: {e2}"

        geo_conflict = float(r["s2_score"]) < 0.70
        vlm_conflict = defect_v in ("split", "dirty", "sticky")
        is_conflict = kind != "clean"
        out_rows.append({
            "scene": scene, "fa": fa, "fb": fb, "idx": idx, "gt_kind": kind,
            "s2_score": r["s2_score"], "geo_conflict": geo_conflict,
            "vlm_defect": defect_v, "vlm_conf": conf, "vlm_conflict": vlm_conflict,
            "geo_correct": geo_conflict == is_conflict,
            "vlm_correct": vlm_conflict == is_conflict,
            "vlm_kind_exact": defect_v == kind,
            "reason": reason,
        })
        print(f"[{n+1}/{len(rows)}] {scene} #{idx} 真值={kind} | 几何={'冲突' if geo_conflict else '一致'}"
              f" | VLM={defect_v}({conf:.2f}) | {reason[:40]}", flush=True)

    n = len(out_rows)
    geo_acc = np.mean([r["geo_correct"] for r in out_rows])
    vlm_acc = np.mean([r["vlm_correct"] for r in out_rows])
    or_acc = np.mean([((r["geo_conflict"] or r["vlm_conflict"]) == (r["gt_kind"] != "clean"))
                      for r in out_rows])
    and_acc = np.mean([((r["geo_conflict"] and r["vlm_conflict"]) == (r["gt_kind"] != "clean"))
                       for r in out_rows])
    kind_acc = np.mean([r["vlm_kind_exact"] for r in out_rows])

    print("\n" + "=" * 66)
    print(f"VLM 语义层仲裁（Stage2 疑难子集，{n} 实例）")
    print("=" * 66)
    print(f"几何判定准确率          : {geo_acc:.1%}")
    print(f"VLM 语义判定准确率      : {vlm_acc:.1%}（缺陷类型完全一致 {kind_acc:.1%}）")
    print(f"几何 OR VLM（高召回）   : {or_acc:.1%}")
    print(f"几何 AND VLM（高精确）  : {and_acc:.1%}")

    tag = args.out_tag
    with open(f"results/vlm_semantic_layer{tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    with open(f"results/vlm_semantic_report{tag}.txt", "w", encoding="utf-8") as f:
        f.write(f"VLM 语义层仲裁（Stage2 疑难子集，{n} 实例）\n")
        f.write(f"几何 {geo_acc:.1%} | VLM {vlm_acc:.1%}（类型一致 {kind_acc:.1%}）"
                f" | OR {or_acc:.1%} | AND {and_acc:.1%}\n")
        for r in out_rows:
            f.write(f"{r['scene']} {r['fa']}->{r['fb']} #{r['idx']} 真值={r['gt_kind']} "
                    f"VLM={r['vlm_defect']} 几何={'冲突' if r['geo_conflict'] else '一致'} "
                    f"理由={r['reason']}\n")
    print(f"\n明细已存 results/vlm_semantic_layer{tag}.csv / vlm_semantic_report{tag}.txt")


if __name__ == "__main__":
    main()
