"""列出你的 Kimi API Key 可用的模型名（用于确认 KIMI_MODEL 该填什么）。

用法：python list_models.py
输出后，把正确的模型名（如 kimi-k3 或其他）填入 .env 的 KIMI_MODEL。
"""
import sys

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

err = config.check_config()
if err:
    print(err)
    sys.exit(1)

from openai import OpenAI  # noqa: E402

client = OpenAI(api_key=config.KIMI_API_KEY, base_url=config.KIMI_BASE_URL)
models = client.models.list()
print("可用模型：")
for m in models.data:
    print(" -", m.id)
print(f"\n当前 KIMI_MODEL = {config.KIMI_MODEL}")
print("如果上面列表里有更准确的模型名，请更新 .env 中的 KIMI_MODEL。")
