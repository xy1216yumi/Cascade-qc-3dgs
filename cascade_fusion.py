"""置信度信号接入级联（增强项 5）：融合规则 = 级联判定 OR SAM2 自我置信度低。

阈值不靠标签调 oracle：用全部 SAM2 mask 的分数分布（median − 2×MAD，稳健统计）
自动定阈值，然后在 53 实例级联评估集上报告融合前后对比。
标签两种口径：IoU<0.8（通用质量）与 重建完整度<0.88（下游受损，更贴近论文目标）。

用法：python cascade_fusion.py
输出：results/cascade_fusion_report.txt
"""
import csv, json
import numpy as np

LOW_IOU = 0.8
COV_THR = 0.88


def prf(y, p):
    tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum()); tn = int((~p & ~y).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return P, R, 2 * P * R / max(P + R, 1e-9), (tp + tn) / len(y)


def main():
    sam2 = [json.loads(l) for l in open("results/sam2_real_batch.jsonl", encoding="utf-8")]
    all_scores = np.array([r["score"] for r in sam2])
    med = np.median(all_scores)
    mad = np.median(np.abs(all_scores - med))
    tau = med - 2 * mad  # 稳健低分阈值（不看标签）

    score_map = {(r["scene"], r["frame"], r["inst_idx"]): r["score"] for r in sam2}
    casc = list(csv.DictReader(open("results/sam2_cascade_eval.csv", encoding="utf-8-sig")))
    e2e = {(r["scene"], r["pair"], int(r["idx"])): float(r["cov_sam"])
           for r in csv.DictReader(open("results/e2e_remedy.csv", encoding="utf-8-sig"))}

    y_iou, y_cov, p_casc, p_fuse = [], [], [], []
    for r in casc:
        fb = int(r["pair"].split("_")[1])
        key = (r["scene"], fb, int(r["idx"]))
        sc = score_map.get(key)
        cov = e2e.get((r["scene"], r["pair"], int(r["idx"])))
        if sc is None or cov is None:
            continue
        y_iou.append(float(r["iou_b"]) < LOW_IOU)
        y_cov.append(cov < COV_THR)
        p_casc.append(r["action"] != "PASS")
        p_fuse.append((r["action"] != "PASS") or (sc < tau))

    y_iou, y_cov = np.array(y_iou), np.array(y_cov)
    p_casc, p_fuse = np.array(p_casc), np.array(p_fuse)

    lines = [
        f"置信度融合阈值 τ = {tau:.4f}（全部 {len(all_scores)} 个 SAM2 mask 分数的 median−2×MAD，无标签参与）",
        f"评估实例 {len(y_iou)} 个",
        "",
        f"{'规则':<22s}{'标签口径':<18s}{'P':>6s}{'R':>6s}{'F1':>6s}{'准确率':>8s}",
    ]
    for name, p in [("级联单独", p_casc), ("融合（级联∨低置信）", p_fuse)]:
        for lname, y in [(f"IoU<{LOW_IOU}", y_iou), (f"完整度<{COV_THR}", y_cov)]:
            P, R, F1, acc = prf(y, p)
            lines.append(f"{name:<22s}{lname:<18s}{P:6.2f}{R:6.2f}{F1:6.2f}{acc:8.1%}")

    text = "\n".join(lines)
    print(text)
    with open("results/cascade_fusion_report.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n已存 results/cascade_fusion_report.txt")


if __name__ == "__main__":
    main()
