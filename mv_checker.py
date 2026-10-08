"""跨视图一致性检查器：同一实例在两帧的 mask 是否吻合（重投影验证）。

核心数学：
- 帧A mask 像素 + 深度图 → 相机3D点（毫米）：x_c=(u-cx)*z/fx, y_c=(v-cy)*z/fy, z=depth*scale
- 相机A坐标 → 世界坐标：Pw = R_A^T (Pc_A - t_A)   （cam_R_w2c / cam_t_w2c 为世界→相机）
- 世界坐标 → 相机B坐标：Pc_B = R_B Pw + t_B
- 投影到帧B：u'=fx*x/z+cx, v'=fy*y/z+cy
- 命中：投影点在帧B图像内，且落在帧B的该实例 mask 内
- 可见性过滤：帧B深度图上该点深度与投影深度相差 > vis_tol(mm) 视为被遮挡 → 跳过
  （跨视图检查只对"两帧都可见"的表面做判断，避免视角差导致的合理差异误判）

分数：双向命中率的均值（或最小值，见 --score-mode），越低越可能冲突。
判定：score < threshold → 冲突（两帧 mask 不一致）。

输入：BOP 场景目录 + mv_answers.csv（prepare_mv_data.py 生成）+ 缺陷 mask 目录
输出：报告（stdout + mv_eval_report.txt）

用法：
    python mv_checker.py --scene-dir real_data/test/000048 --answers mv_answers.csv --masks mv_masks
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_gray(p):
    return np.array(Image.open(p).convert("L")) > 0


def make_pose_resolver(cam, gt):
    """返回 pose(f) -> (K, R_w2c, t_w2c)。

    YCB-V 等数据集直接在 scene_camera.json 里给 cam_R_w2c/cam_t_w2c；
    LMO 等只给物体的 cam_R_m2c/cam_t_m2c（模型系=世界系，跨帧一致），
    此时用"所有帧都出现的参照物体"的 m2c 位姿代替（相对位姿不变，重投影不受影响）。
    """
    if all("cam_R_w2c" in cam[f] for f in cam):
        return lambda f: (np.array(cam[f]["cam_K"]).reshape(3, 3),
                          np.array(cam[f]["cam_R_w2c"]).reshape(3, 3),
                          np.array(cam[f]["cam_t_w2c"]).reshape(3))
    from collections import Counter
    cnt = Counter(inst["obj_id"] for f in gt for inst in gt[f])
    ref = cnt.most_common(1)[0][0]

    def resolve(f):
        for inst in gt[f]:
            if inst["obj_id"] == ref:
                return (np.array(cam[f]["cam_K"]).reshape(3, 3),
                        np.array(inst["cam_R_m2c"]).reshape(3, 3),
                        np.array(inst["cam_t_m2c"]).reshape(3))
        return None
    return resolve


def unproject(mask, depth, K, scale):
    """mask 内像素 + 深度 → 相机3D点（mm）。返回 Nx3。"""
    ys, xs = np.where(mask)
    z = depth[ys, xs].astype(np.float32) * scale
    ok = z > 1.0
    xs, ys, z = xs[ok], ys[ok], z[ok]
    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return np.stack([(xs - cx) * z / fx, (ys - cy) * z / fy, z], axis=1).astype(np.float32)


def project(Pc, K, H, W):
    """相机3D点 → 图像坐标，返回有效点下标（图像内且 z>0）。"""
    z = Pc[:, 2]
    ok = z > 1.0
    if not ok.any():
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), ok
    u = Pc[ok, 0] * K[0, 0] / z[ok] + K[0, 2]
    v = Pc[ok, 1] * K[1, 1] / z[ok] + K[1, 2]
    in_img = (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    sub = np.where(ok)[0][in_img]
    return u[in_img].astype(int), v[in_img].astype(int), sub


def one_direction(Pc_src, R_src, t_src, R_dst, t_dst, K, H, W, mask_dst, depth_dst, vis_tol,
                  scale_dst=0.1):
    """把源相机系点 Pc_src 变换到目标系并投影、做可见性过滤和命中判断。
    scale_dst 为目标深度图换算到 mm 的比例（YCB-V=0.1，LMO=1.0）。
    返回 (命中数, 有效数)。"""
    Pw = (R_src.T @ (Pc_src - t_src).T).T
    Pc_dst = (R_dst @ Pw.T + t_dst[:, None]).T
    u, v, sub = project(Pc_dst, K, H, W)
    if len(sub) == 0:
        return 0, 0
    z_proj = Pc_dst[sub, 2]
    z_act = depth_dst[v, u].astype(np.float32) * scale_dst
    vis = np.abs(z_proj - z_act) < vis_tol
    if not vis.any():
        return 0, 0
    hit = int(mask_dst[v[vis], u[vis]].sum())
    return hit, int(vis.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", default="data/real_data/test/000048")
    ap.add_argument("--answers", default="results/mv_answers.csv")
    ap.add_argument("--masks", default="data/mv_masks", help="缺陷 mask 目录（帧B 待检查 mask）")
    ap.add_argument("--threshold", type=float, default=0.65,
                    help="一致性分数低于该值判定为冲突")
    ap.add_argument("--vis-tol", type=float, default=30.0,
                    help="可见性深度差容忍（mm），两帧深度差小于该值视为同一表面")
    ap.add_argument("--score-mode", default="min", choices=["mean", "min"])
    ap.add_argument("--report", default="mv_eval_report.txt")
    args = ap.parse_args()

    cam = json.load(open(os.path.join(args.scene_dir, "scene_camera.json")))
    gt = json.load(open(os.path.join(args.scene_dir, "scene_gt.json")))
    mdir = os.path.join(args.scene_dir, "mask_visib")
    ddir = os.path.join(args.scene_dir, "depth")

    rows = list(csv.DictReader(open(args.answers, encoding="utf-8-sig")))
    print("=" * 60)
    print("跨视图一致性检查器评估报告")
    print("=" * 60)
    print(f"场景: {args.scene_dir} | 阈值: {args.threshold} | 可见性容差: {args.vis_tol}mm | 分数: {args.score_mode}")
    print()

    details = []
    stats = {"total": 0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for r in rows:
        scene = r["scene"]
        fa, fb, idx = r["frame_a"], r["frame_b"], int(r["inst_idx"])
        is_conflict = r["is_conflict"] == "是"

        maskA = load_gray(os.path.join(mdir, f"{int(fa):06d}_{idx:06d}.png"))
        maskB = load_gray(os.path.join(args.masks, f"{scene}_{fa}_{fb}_idx{idx}.png"))
        depthA = np.array(Image.open(os.path.join(ddir, f"{int(fa):06d}.png"))).astype(np.float32)
        depthB = np.array(Image.open(os.path.join(ddir, f"{int(fb):06d}.png"))).astype(np.float32)
        H, W = depthA.shape
        K = np.array(cam[fa]["cam_K"]).reshape(3, 3)
        RA, tA = np.array(cam[fa]["cam_R_w2c"]).reshape(3, 3), np.array(cam[fa]["cam_t_w2c"]).reshape(3)
        RB, tB = np.array(cam[fb]["cam_R_w2c"]).reshape(3, 3), np.array(cam[fb]["cam_t_w2c"]).reshape(3)

        PcA = unproject(maskA, depthA, K, 0.1)
        hit_AB, n_AB = one_direction(PcA, RA, tA, RB, tB, K, H, W, maskB, depthB, args.vis_tol)
        PcB = unproject(maskB, depthB, K, 0.1)
        hit_BA, n_BA = one_direction(PcB, RB, tB, RA, tA, K, H, W, maskA, depthA, args.vis_tol)

        rate_AB = hit_AB / n_AB if n_AB else np.nan
        rate_BA = hit_BA / n_BA if n_BA else np.nan
        if args.score_mode == "min":
            score = min(rate_AB, rate_BA)
        else:
            score = 0.5 * rate_AB + 0.5 * rate_BA
        pred = "冲突" if score < args.threshold else "一致"
        correct = (pred == ("冲突" if is_conflict else "一致"))
        if correct:
            stats["tp" if is_conflict else "tn"] += 1
        else:
            stats["fp" if not is_conflict else "fn"] += 1
        stats["total"] += 1

        details.append((r, rate_AB, rate_BA, score, pred, correct))
        mark = "OK" if correct else "MISS"
        print(f"[{mark}] 帧{fa}→{fb} 实例{idx} [{r['defect']}] 命中率 A→B={rate_AB:.3f} B→A={rate_BA:.3f} "
              f"一致性={score:.3f} 判定={pred} 人工={'冲突' if is_conflict else '一致'}")

    n = stats["total"]
    acc = (stats["tp"] + stats["tn"]) / n if n else 0.0
    print("\n" + "=" * 60)
    print(f"参与统计实例数: {n}")
    print(f"准确率: {stats['tp'] + stats['tn']}/{n} = {acc:.1%}")
    print(f"  冲突检出(TP): {stats['tp']} | 一致保留(TN): {stats['tn']} | "
          f"误报(FP): {stats['fp']} | 漏检(FN): {stats['fn']}")

    # 分缺陷类型检出率
    by_kind = {}
    for r, a, b, s, p, c in details:
        k = r["defect"]
        by_kind.setdefault(k, [0, 0])
        by_kind[k][0] += 1
        if p == "冲突":
            by_kind[k][1] += 1
    print("\n分缺陷检出率（判定为冲突的比例）:")
    for k, (tot, con) in by_kind.items():
        print(f"  {k:6s}: {con}/{tot} = {con / tot:.1%}" if tot else f"  {k}: -")

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("跨视图一致性检查器评估报告\n")
        f.write(f"场景: {args.scene_dir} | 阈值: {args.threshold} | 可见性容差: {args.vis_tol}mm\n")
        f.write(f"准确率: {stats['tp'] + stats['tn']}/{n} = {acc:.1%}\n")
        for r, a, b, s, p, c in details:
            f.write(f"{r['frame_a']}->{r['frame_b']} 实例{r['inst_idx']} [{r['defect']}] "
                    f"score={s:.3f} pred={p} gt={'冲突' if r['is_conflict'] == '是' else '一致'}\n")
    print(f"\n报告已保存到 {args.report}")


if __name__ == "__main__":
    main()
