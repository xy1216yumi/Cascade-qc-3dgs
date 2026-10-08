"""Baseline 对比实验：证明级联调度优于常见的"无几何/无学习"质检基线。

设置 1（缺陷检出，与 cascade_eval_all 完全同数据：12 场景 × 3 帧对，注入缺陷）：
  - area_consistency ：跨视图面积一致性（面积比应 ≈ 深度比平方），朴素几何基线
  - boundary_gradient：mask 边界处的图像梯度强度（分割质量经典启发式：好 mask 边界贴着图像边缘）
  - solidity         ：mask 面积 / 凸包面积（形状紧凑度启发式）
  每个基线用"oracle 阈值"（扫阈值取最高准确率，对基线最宽容）+ AUC 报告。

设置 2（真实 SAM2 质量预测，53 实例，低质定义 = IoU<0.8）：
  - sam2_self_score：SAM2 自己输出的预测 IoU 分数（score 字段）
  - iou_gap        ：两帧 IoU 之差 |iou_a - iou_b|（朴素跨视图稳定性）
  - 对照：级联动作（RESEGMENT ↔ 低质），来自 results/sam2_cascade_eval.csv

输出：results/baselines_report.txt + results/baselines_detail.csv

用法：python eval_baselines.py
"""
import csv, json, os, sys

import numpy as np
from PIL import Image
from scipy import ndimage, spatial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_checker import load_gray  # noqa: E402
from cascade_checker import inject_defect  # noqa: E402
from mv_eval_all import pick_frame_pairs  # noqa: E402

SCENE_DIR = "data/real_data/test"
LOW_IOU = 0.8  # 设置 2 的低质阈值


def auc(scores, labels):
    """Mann-Whitney AUC：scores 越大越倾向判"有缺陷/低质"。"""
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    pos = np.array(labels, dtype=bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def best_acc(scores, labels):
    """oracle 阈值扫描：返回 (最佳准确率, 阈值)。"""
    best = (0.0, None)
    for t in np.unique(scores):
        pred = scores >= t
        acc = (pred == np.array(labels, dtype=bool)).mean()
        if acc > best[0]:
            best = (float(acc), float(t))
    return best


def boundary_gradient(mask, rgb):
    """mask 边界带上的平均图像梯度（经整图梯度标准化；越低说明边界越没贴着图像边缘）。"""
    gray = np.array(rgb.convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    band = mask.astype(bool) ^ ndimage.binary_erosion(mask, iterations=2)
    band |= ndimage.binary_dilation(mask, iterations=2) & ~mask.astype(bool)
    if band.sum() < 10 or mag.mean() == 0:
        return np.nan
    return float(mag[band].mean() / mag.mean())


def solidity_defect(mask):
    """1 - 面积/凸包面积（越大越不紧凑）。凸包退化时返回 nan。"""
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return np.nan
    pts = np.stack([xs, ys], axis=1)
    step = max(1, len(pts) // 2000)  # 抽稀加速
    try:
        hull = spatial.ConvexHull(pts[::step])
        hull_area = hull.volume  # 2D 时 volume 即面积
    except Exception:
        return np.nan
    if hull_area <= 0:
        return np.nan
    return float(1 - mask.sum() / hull_area)


def main():
    os.makedirs("results", exist_ok=True)
    rows = []

    # ===== 设置 1：注入缺陷检出 =====
    scenes = sorted(d for d in os.listdir(SCENE_DIR)
                    if os.path.isdir(os.path.join(SCENE_DIR, d)))
    for scene in scenes:
        sd = os.path.join(SCENE_DIR, scene)
        cam = json.load(open(os.path.join(sd, "scene_camera.json")))
        gt = json.load(open(os.path.join(sd, "scene_gt.json")))
        mdir = os.path.join(sd, "mask_visib")
        ddir = os.path.join(sd, "depth")

        for fa, fb in pick_frame_pairs(cam, gt, max_pairs=3):
            n_inst = len(gt[fa])
            idxs = [i for i in range(n_inst)
                    if os.path.exists(os.path.join(mdir, f"{int(fa):06d}_{i:06d}.png"))
                    and os.path.exists(os.path.join(mdir, f"{int(fb):06d}_{i:06d}.png"))]
            if not idxs:
                continue
            depthA = np.array(Image.open(os.path.join(ddir, f"{int(fa):06d}.png"))).astype(np.float32) * 0.1
            depthB = np.array(Image.open(os.path.join(ddir, f"{int(fb):06d}.png"))).astype(np.float32) * 0.1
            rgbB = Image.open(os.path.join(sd, "rgb", f"{int(fb):06d}.png")).convert("RGB")

            for idx in idxs:
                maskA = load_gray(os.path.join(mdir, f"{int(fa):06d}_{idx:06d}.png"))
                maskB_gt = load_gray(os.path.join(mdir, f"{int(fb):06d}_{idx:06d}.png"))
                kind = ["clean", "split", "dirty", "sticky"][idx % 4]
                maskB = inject_defect(maskB_gt, kind, mdir, fb, idx, n_inst, *maskB_gt.shape)

                # 特征 1：跨视图面积一致性（面积 ∝ 1/z²）
                medA = np.median(depthA[maskA]) if maskA.any() else np.nan
                medB = np.median(depthB[maskB]) if maskB.any() else np.nan
                if medA > 0 and medB > 0 and maskA.sum() > 0:
                    expected = (medA / medB) ** 2
                    f_area = abs(np.log((maskB.sum() / maskA.sum()) / expected + 1e-9))
                else:
                    f_area = np.nan

                f_grad = boundary_gradient(maskB, rgbB)
                f_solid = solidity_defect(maskB)

                rows.append({
                    "scene": scene, "fa": fa, "fb": fb, "idx": idx,
                    "defect": kind, "is_conflict": kind != "clean",
                    "f_area": round(f_area, 4) if np.isfinite(f_area) else "",
                    "f_grad": round(f_grad, 4) if np.isfinite(f_grad) else "",
                    "f_solid": round(f_solid, 4) if np.isfinite(f_solid) else "",
                })
        print(f"[{scene}] 完成")

    labels = np.array([r["is_conflict"] for r in rows])
    n = len(rows)
    report = []
    report.append(f"设置 1：注入缺陷检出（{n} 实例，12 场景 × ≤3 帧对）")
    report.append(f"{'基线':<20s} {'AUC':>7s} {'oracle 阈值最佳准确率':>12s}")
    for name, key, higher_is_defect in [
        ("area_consistency", "f_area", True),
        ("boundary_gradient", "f_grad", False),
        ("solidity", "f_solid", True),
    ]:
        vals = np.array([r[key] if r[key] != "" else np.nan for r in rows], dtype=float)
        ok = np.isfinite(vals)
        s = vals[ok] * (1 if higher_is_defect else -1)  # 统一成"越大越像有缺陷"
        a = auc(s, labels[ok])
        acc, thr = best_acc(s, labels[ok])
        report.append(f"{name:<20s} {a:7.3f} {acc:12.1%}（n={ok.sum()}）")
    report.append("")
    report.append("对照（同数据，引自 results/mv_all_report.txt / cascade_all_report.txt）：")
    report.append("  跨视图-only 75.0% | 级联 90.0%（跨视图调用率 35.6%）")

    # ===== 设置 2：真实 SAM2 质量预测 =====
    sam2 = [json.loads(l) for l in open("results/sam2_real_batch.jsonl", encoding="utf-8")]
    casc = list(csv.DictReader(open("results/sam2_cascade_eval.csv", encoding="utf-8-sig")))
    score_map = {(r["scene"], r["frame"], r["inst_idx"]): r["score"] for r in sam2}

    y, s_self, s_gap, pred_casc = [], [], [], []
    for r in casc:
        fa, fb = map(int, r["pair"].split("_"))
        low = float(r["iou_b"]) < LOW_IOU
        sc = score_map.get((r["scene"], fb, int(r["idx"])))
        if sc is None:
            continue
        y.append(low)
        s_self.append(-float(sc))                      # score 低 = 低质
        s_gap.append(abs(float(r["iou_a"]) - float(r["iou_b"])))
        pred_casc.append(r["action"] != "PASS")

    y = np.array(y)
    report.append("")
    report.append(f"设置 2：真实 SAM2 mask 质量预测（{len(y)} 实例，低质=IoU<{LOW_IOU}，"
                  f"低质占比 {y.mean():.1%}）")
    report.append(f"{'方法':<20s} {'AUC':>7s} {'准确率':>9s}")
    for name, s in [("sam2_self_score", np.array(s_self)), ("iou_gap", np.array(s_gap))]:
        a = auc(s, y)
        acc, _ = best_acc(s, y)
        report.append(f"{name:<20s} {a:7.3f} {acc:9.1%}（oracle 阈值）")
    acc_casc = (np.array(pred_casc) == y).mean()
    report.append(f"{'级联调度（无阈值）':<20s} {'—':>7s} {acc_casc:9.1%}")

    text = "\n".join(report)
    print("\n" + text)
    with open("results/baselines_report.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    with open("results/baselines_detail.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n报告已存 results/baselines_report.txt，明细 results/baselines_detail.csv")


if __name__ == "__main__":
    main()
