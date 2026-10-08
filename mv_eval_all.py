"""多场景批量评估：跨视图一致性检查器在 BOP YCB-V 全部 12 场景的泛化验证。

对每个场景：
1. 自动选择"最大视角差"帧对（帧间旋转角最大，其次平移距离大；要求两帧实例 mask 齐全）
2. 注入缺陷（clean/split/dirty/sticky 按实例索引轮换）到帧B mask
3. 深度反投影 + 位姿重投影双向命中率 → 一致性分数（min）
4. 阈值 0.70 判定 → 与 ground truth 比对

输出：
- stdout 汇总表
- mv_all_results.csv  每实例明细（论文实验表素材）
- mv_all_report.txt   文本报告

用法：python mv_eval_all.py [--pairs 3]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prepare_mv_data import split_mask, dilate, add_blob  # noqa: E402
from mv_checker import load_gray, unproject, one_direction, make_pose_resolver  # noqa: E402

SCENE_DIR = "data/real_data/test"
THRESHOLD = 0.70
VIS_TOL = 30.0


def pick_frame_pair(cam, gt, ang_min=25, ang_max=55):
    """选适中视角差帧对：旋转角落在 [ang_min, ang_max]，优先平移基线大者。

    视角差太小（<25°）两视图信息几乎相同，验证无意义；
    视角差太大（>55°）遮挡变化剧烈、反投影误差放大，可见交集过小。
    真实场景的跨视图检查（视频相邻帧/小基线多视图）对应适中视角差。
    返回 (fa, fb) 或 None。
    """
    frames = list(cam.keys())
    best = None
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            fa, fb = frames[i], frames[j]
            Ra = np.array(cam[fa]["cam_R_w2c"]).reshape(3, 3)
            Rb = np.array(cam[fb]["cam_R_w2c"]).reshape(3, 3)
            ta = np.array(cam[fa]["cam_t_w2c"]).reshape(3)
            tb = np.array(cam[fb]["cam_t_w2c"]).reshape(3)
            Rrel = Ra @ Rb.T
            ang = float(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))
            if not (ang_min <= ang <= ang_max):
                continue
            tdist = float(np.linalg.norm(ta - tb))
            if best is None or tdist > best[0]:
                best = (tdist, fa, fb)
    if best is None:
        return None
    return best[1], best[2]


def pick_frame_pairs(cam, gt, max_pairs=3, ang_min=25, ang_max=55):
    """选多个视角差适中且角度分桶多样化的帧对（扩大评测规模用）。

    视角差落在 [ang_min, ang_max] 的候选帧对按角度均分为 max_pairs 个桶，
    每桶取平移基线最大者，保证视角多样性。返回 [(fa, fb), ...]。
    """
    frames = list(cam.keys())
    pose = make_pose_resolver(cam, gt)
    cands = []
    for i in range(len(frames)):
        for j in range(i + 1, len(frames)):
            fa, fb = frames[i], frames[j]
            pa, pb = pose(fa), pose(fb)
            if pa is None or pb is None:
                continue
            Ra, Rb = pa[1], pb[1]
            ta, tb = pa[2], pb[2]
            Rrel = Ra @ Rb.T
            ang = float(np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1))))
            if ang_min <= ang <= ang_max:
                cands.append((ang, float(np.linalg.norm(ta - tb)), fa, fb))
    if not cands:
        return []
    width = (ang_max - ang_min) / max_pairs
    bins = {}
    for ang, tdist, fa, fb in cands:
        b = min(int((ang - ang_min) / width), max_pairs - 1)
        if b not in bins or tdist > bins[b][0]:
            bins[b] = (tdist, fa, fb)
    return [(fa, fb) for _, fa, fb in sorted(bins.values(), reverse=True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=3, help="每场景最多帧对数（角度分桶多样化）")
    ap.add_argument("--scene-dir", default=SCENE_DIR, help="BOP test 目录")
    ap.add_argument("--out-tag", default="", help="输出文件名后缀，如 _lmo")
    ap.add_argument("--min-vis", type=int, default=300, help="跨视图有效采样点下限（小物体数据集调低）")
    ap.add_argument("--depth-bias", type=float, default=1.0,
                    help="深度系统性偏差校正系数（T-LESS PrimeSense 约需 1.05）")
    args = ap.parse_args()
    scene_dir = args.scene_dir
    scenes = sorted(d for d in os.listdir(scene_dir)
                    if os.path.isdir(os.path.join(scene_dir, d)))
    all_rows = []
    summary = []

    for scene in scenes:
        sd = os.path.join(scene_dir, scene)
        cam = json.load(open(os.path.join(sd, "scene_camera.json")))
        gt = json.load(open(os.path.join(sd, "scene_gt.json")))
        pose = make_pose_resolver(cam, gt)
        mdir = os.path.join(sd, "mask_visib")
        ddir = os.path.join(sd, "depth")

        pairs = pick_frame_pairs(cam, gt, max_pairs=args.pairs)
        if not pairs:
            print(f"[跳过] {scene}: 无帧对")
            continue

        for fa, fb in pairs:
            n_inst = len(gt[fa])
            # 该帧对可用的实例索引（两帧 mask 都存在）
            idxs = []
            for idx in range(n_inst):
                pa = os.path.join(mdir, f"{int(fa):06d}_{idx:06d}.png")
                pb = os.path.join(mdir, f"{int(fb):06d}_{idx:06d}.png")
                if os.path.exists(pa) and os.path.exists(pb):
                    idxs.append(idx)
            if len(idxs) == 0:
                print(f"[跳过] {scene}: 帧对 {fa},{fb} 无共同实例")
                continue

            K, RA, tA = pose(fa)
            _, RB, tB = pose(fb)
            depthA = np.array(Image.open(os.path.join(ddir, f"{int(fa):06d}.png"))).astype(np.float32) * args.depth_bias
            depthB = np.array(Image.open(os.path.join(ddir, f"{int(fb):06d}.png"))).astype(np.float32) * args.depth_bias
            ds = cam[fa].get("depth_scale", 0.1)
            H, W = depthA.shape

            sc_ok = sc_tot = 0
            skipped = 0
            for idx in idxs:
                maskA = load_gray(os.path.join(mdir, f"{int(fa):06d}_{idx:06d}.png"))
                maskB_gt = load_gray(os.path.join(mdir, f"{int(fb):06d}_{idx:06d}.png"))
                kind = ["clean", "split", "dirty", "sticky"][idx % 4]
                if kind == "split":
                    maskB = split_mask(maskB_gt)
                elif kind == "dirty":
                    maskB = add_blob(dilate(maskB_gt, 3), H, W)
                elif kind == "sticky":
                    nb = os.path.join(mdir, f"{int(fb):06d}_{(idx + 1) % n_inst:06d}.png")
                    maskB = maskB_gt | load_gray(nb) if os.path.exists(nb) else maskB_gt
                    if not os.path.exists(nb):
                        kind = "clean"
                else:
                    maskB = maskB_gt

                PcA = unproject(maskA, depthA, K, ds)
                hit_AB, n_AB = one_direction(PcA, RA, tA, RB, tB, K, H, W, maskB, depthB, VIS_TOL,
                                             scale_dst=ds)
                PcB = unproject(maskB, depthB, K, ds)
                hit_BA, n_BA = one_direction(PcB, RB, tB, RA, tA, K, H, W, maskA, depthA, VIS_TOL,
                                             scale_dst=ds)
                # 有效采样点太少（两帧可见交集过小）→ 帧对对该实例不具判别力，跳过
                if min(n_AB, n_BA) < args.min_vis:
                    skipped += 1
                    continue
                r_AB = hit_AB / n_AB if n_AB else np.nan
                r_BA = hit_BA / n_BA if n_BA else np.nan
                # 重投影 IoU：对面积缺失（split）更敏感
                iou_AB = hit_AB / (n_AB + n_BA - hit_AB) if (n_AB + n_BA - hit_AB) > 0 else np.nan
                # 注：IoU 用同一双向命中计数估算，仅作指标对比
                score = min(r_AB, r_BA)
                score_iou = min(r_AB, iou_AB)
                is_conflict = kind != "clean"
                pred = score < THRESHOLD
                correct = pred == is_conflict
                sc_ok += int(correct)
                sc_tot += 1
                all_rows.append({
                    "scene": scene, "frame_a": fa, "frame_b": fb, "inst_idx": idx,
                    "defect": kind, "is_conflict": is_conflict,
                    "hit_AB": round(r_AB, 4), "hit_BA": round(r_BA, 4),
                    "score": round(score, 4), "score_iou": round(score_iou, 4),
                    "pred": pred, "correct": correct,
                })
            acc = sc_ok / sc_tot if sc_tot else 0.0
            summary.append((scene, fa, fb, sc_ok, sc_tot, acc))
            skip_note = f"（跳过低采样点 {skipped}）" if skipped else ""
            print(f"[{scene}] 帧对 {fa}→{fb} | 实例 {sc_tot} | 准确率 {sc_ok}/{sc_tot} = {acc:.1%} {skip_note}")

    # ===== 汇总 =====
    n = len(all_rows)
    tp = sum(1 for r in all_rows if r["pred"] and r["is_conflict"])
    tn = sum(1 for r in all_rows if not r["pred"] and not r["is_conflict"])
    fp = sum(1 for r in all_rows if r["pred"] and not r["is_conflict"])
    fn = sum(1 for r in all_rows if not r["pred"] and r["is_conflict"])
    acc = (tp + tn) / n if n else 0.0

    print("\n" + "=" * 66)
    print("跨视图一致性检查器 · 12 场景汇总")
    print("=" * 66)
    print(f"参与统计实例数: {n}（{len(scenes)} 场景）")
    print(f"总体准确率: {tp + tn}/{n} = {acc:.1%}")
    print(f"  冲突检出(TP): {tp} | 一致保留(TN): {tn} | 误报(FP): {fp} | 漏检(FN): {fn}")
    by_kind = {}
    for r in all_rows:
        by_kind.setdefault(r["defect"], [0, 0])
        by_kind[r["defect"]][0] += 1
        if r["pred"]:
            by_kind[r["defect"]][1] += 1
    print("\n分缺陷检出率（判定为冲突的比例）:")
    for k in ["clean", "split", "dirty", "sticky"]:
        tot, con = by_kind.get(k, [0, 0])
        rate = f"{con}/{tot} = {con / tot:.1%}" if tot else "-"
        print(f"  {k:6s}: {rate}")

    os.makedirs("results", exist_ok=True)
    with open(f"results/mv_all_results{args.out_tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with open(f"results/mv_all_report{args.out_tag}.txt", "w", encoding="utf-8") as f:
        f.write("跨视图一致性检查器 · 12 场景汇总\n")
        f.write(f"阈值: {THRESHOLD} | 可见性容差: {VIS_TOL}mm | 分数: min\n")
        for scene, fa, fb, ok, tot, a in summary:
            f.write(f"{scene}: 帧对 {fa}->{fb} | {ok}/{tot} = {a:.1%}\n")
        f.write(f"\n总体: {tp + tn}/{n} = {acc:.1%} | TP={tp} TN={tn} FP={fp} FN={fn}\n")
        for k in ["clean", "split", "dirty", "sticky"]:
            tot, con = by_kind.get(k, [0, 0])
            f.write(f"{k}: {con}/{tot}\n")
    print(f"\n明细已保存到 results/mv_all_results{args.out_tag}.csv / results/mv_all_report{args.out_tag}.txt")


if __name__ == "__main__":
    main()
