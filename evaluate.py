"""评估 VLM 检查器准确率：对比人工标注与模型预测。

用法：
    python evaluate.py --answers annotation_answers.csv --predictions results.jsonl

输出（屏幕 + eval_report.txt）：
- 实例级三布尔完全一致率（核心指标）
- 每个属性的单独一致率（clean / split / sticky）
- 漏检统计（模型没输出、人工补列的实例）
- 逐实例错误明细（便于分析哪类判断最容易错）

口径说明：
- 人工答案中留空（无法判断）的行不参与统计。
- 只有模型也输出了该 (图, 实例) 的行才计入准确率；人工补列的漏检实例
  单独统计数量，不计入一致率分母。
"""
import argparse
import csv
import json
import sys

import config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ATTRS = ["is_clean", "is_split", "is_sticky"]


def parse_answer(v: str):
    """'是' -> True, '否' -> False, 其他 -> None（无法判断）"""
    v = (v or "").strip()
    if v == "是":
        return True
    if v == "否":
        return False
    return None


def load_predictions(path: str) -> dict[tuple[str, int], dict]:
    pred: dict[tuple[str, int], dict] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            for inst in rec.get("instances", []):
                pred[(rec["image"], inst["instance_id"])] = inst
    return pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default=config.DEFAULT_ANSWERS)
    ap.add_argument("--predictions", default=config.DEFAULT_RESULTS)
    ap.add_argument("--report", default=config.DEFAULT_REPORT)
    args = ap.parse_args()

    pred = load_predictions(args.predictions)

    rows = []
    with open(args.answers, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # 人工标注自洽性检查（发现矛盾只提醒，不阻断统计）
    for i, r in enumerate(rows, start=2):
        h = {a: parse_answer(r.get(a)) for a in ATTRS}
        if h["is_clean"] is True and (h["is_split"] is True or h["is_sticky"] is True):
            print(
                f"  [标注警告] 第{i}行 ({r.get('image_id')} 实例{r.get('instance_id')}): "
                f"clean=是 但同时 split/sticky=是，自相矛盾，请核对"
            )

    total = 0          # 参与统计的实例数（有模型输出 + 人工可判断）
    exact = 0          # 三布尔全部一致
    attr_hit = {a: 0 for a in ATTRS}
    attr_n = {a: 0 for a in ATTRS}
    missed = []        # 漏检：人工补列而模型没有
    details = []       # 错误明细

    for r in rows:
        img = r["image_id"].strip()
        try:
            iid = int(r["instance_id"])
        except ValueError:
            continue
        key = (img, iid)
        human = {a: parse_answer(r.get(a)) for a in ATTRS}

        if key not in pred:
            # 人工补列 / 模型漏检：只要人工能判断就算漏检
            if any(v is not None for v in human.values()):
                missed.append((img, iid, human))
            continue

        p = pred[key]
        if not any(v is not None for v in human.values()):
            continue  # 人工无法判断，跳过

        total += 1
        agree = True
        for a in ATTRS:
            h = human[a]
            if h is None:
                continue
            attr_n[a] += 1
            if (p.get(a) is True) == h:
                attr_hit[a] += 1
            else:
                agree = False
        if agree:
            exact += 1
        else:
            details.append(
                (img, iid, {a: human[a] for a in ATTRS}, {a: bool(p.get(a)) for a in ATTRS}, p.get("reason", ""))
            )

    lines = []
    def out(s: str = ""):
        print(s)
        lines.append(s)

    out("=" * 60)
    out("VLM 检查器准确率评估报告")
    out("=" * 60)
    if total == 0:
        out("没有可统计的样本（请检查标注文件与结果文件是否对应）")
    else:
        out(f"参与统计实例数: {total}")
        out(f"实例级完全一致率 (三布尔全对): {exact}/{total} = {exact / total:.1%}")
        out("")
        out("各属性一致率:")
        for a in ATTRS:
            n = attr_n[a]
            h = attr_hit[a]
            rate = f"{h / n:.1%}" if n else "N/A"
            out(f"  {a:>10}: {h}/{n} = {rate}")
    if missed:
        out("")
        out(f"漏检实例数: {len(missed)}（模型未输出、人工补列的实例）")
        for img, iid, h in missed[:20]:
            out(f"  - {img} 实例{iid} (人工: {h})")
    if details:
        out("")
        out(f"错误明细（共 {len(details)} 条，最多显示 30 条）:")
        for img, iid, h, p, reason in details[:30]:
            out(f"  - {img} 实例{iid}: 人工={h} 预测={p} | 模型理由: {reason}")

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    out("")
    out(f"报告已保存到 {args.report}")


if __name__ == "__main__":
    main()
