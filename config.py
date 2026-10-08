"""统一配置入口：读取项目根目录 .env 中的环境变量。

用法：在 vlm_checker 目录下创建 .env 文件（参考 .env.example），
所有脚本都从本模块读取配置，不要在其他文件里硬编码 key。
"""
import os
from dotenv import load_dotenv

# 始终从项目根目录加载 .env（无论从哪个子目录运行）
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_BASE_DIR, ".env"))

# Kimi（月之暗面）OpenAI 兼容接口配置
KIMI_API_KEY = os.getenv("KIMI_API_KEY", "").strip()
KIMI_BASE_URL = os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1").strip()
KIMI_MODEL = os.getenv("KIMI_MODEL", "kimi-k3").strip()
# 注意：当前 KIMI 模型只接受 temperature=1（传其他值会 400 报错），
# 换模型时如需调整，在 .env 里加 KIMI_TEMPERATURE=xxx。
KIMI_TEMPERATURE = float(os.getenv("KIMI_TEMPERATURE", "1"))
# 是否优先尝试 function_calling 结构化输出（kimi-k3 不支持，默认 False 直接走 JSON mode）
KIMI_PREFER_FC = os.getenv("KIMI_PREFER_FC", "0") == "1"

# 图片输入目录 / 输出文件（可被命令行参数覆盖）
DEFAULT_INPUT_DIR = os.path.join(_BASE_DIR, "data", "sample")
DEFAULT_RESULTS = os.path.join(_BASE_DIR, "results.jsonl")
DEFAULT_TEMPLATE = os.path.join(_BASE_DIR, "annotation_template.csv")
DEFAULT_ANSWERS = os.path.join(_BASE_DIR, "annotation_answers.csv")
DEFAULT_REPORT = os.path.join(_BASE_DIR, "eval_report.txt")


def get_int(name: str, default: int) -> int:
    """从 .env 读整数配置，解析失败或缺失时返回默认值。"""
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def check_config() -> str:
    """检查关键配置，返回错误说明；无错误返回空串。"""
    if not KIMI_API_KEY:
        return (
            "未找到 KIMI_API_KEY。请在 vlm_checker 目录下创建 .env 文件，"
            "内容格式见 .env.example，然后填入你的 Kimi API Key。"
        )
    return ""
