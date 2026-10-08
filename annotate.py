"""生成人工标注模板：从 results.jsonl 抽出所有 (图片, 实例编号) 对，输出 CSV。

用法：
    python annotate.py --input results.jsonl --template annotation_template.csv

生成后请用 Excel/WPS 打开 annotation_template.csv（已用 UTF-8-BOM 编码，不会乱码），
按图逐行人工判断并填写：
    is_clean / is_split / is_sticky 三列填：是 或 否
    无法判断就留空（评估时自动跳过该行）
    note 列可写备注

重要：如果渲染图上存在模型漏掉的实例（图上编号在 CSV 里没有），
请手动在 CSV 末尾追加一行：图片名, 漏掉的编号, 是/否, 是/否, 是/否, 备注
评估时会自动把这些行计为"漏检实例"。
填完后另存为 annotation_answers.csv（保持表头不变）。
"""
import argparse
import csv
import json
import sys

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

HEADERS = ["image_id", "instance_id", "is_clean", "is_split", "is_sticky", "note"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=config.DEFAULT_RESULTS)
    ap.add_argument("--template", default=config.DEFAULT_TEMPLATE)
    args = ap.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "instances" not in rec or not rec["instances"]:
                print(f"[跳过] {rec.get('image', '?')}: 无实例输出（可能调用失败）")
                continue
            img = rec["image"]
            for inst in rec["instances"]:
                rows.append([img, inst["instance_id"], "", "", "", ""])

    if not rows:
        print("没有可标注的实例。请先运行 batch_check.py 生成 results.jsonl。")
        sys.exit(1)

    with open(args.template, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(HEADERS)
        w.writerows(rows)

    imgs = len({r[0] for r in rows})
    print(f"已生成 {args.template}：{imgs} 张图、{len(rows)} 个实例待标注。")
    print("人工填完后另存为 annotation_answers.csv，再运行 evaluate.py。")


if __name__ == "__main__":
    main()
