"""级联质量校验调度器 · 12 场景全量评估。

与 mv_eval_all.py 完全同源（同帧对选择、同缺陷注入、同跨视图实现），
叠加 Stage 1 单帧规则前置路由，用于对比：
  - 全量跨视图（mv_eval_all）：每个实例都跑跨视图
  - 级联调度（本脚本）：单帧能判的（split/dirty）不花跨视图算力

用法：python cascade_eval_all.py [--route aggressive|conservative]
"""
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_checker import load_gray, unproject, one_direction, make_pose_resolver  # noqa: E402
from cascade_checker import (  # noqa: E402
    stage1_single_frame, stage2_cross_view, inject_defect, THRESHOLD, VIS_TOL,
)
from mv_eval_all import pick_frame_pairs  # noqa: E402
import argparse

SCENE_DIR = "data/real_data/test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", choices=["aggressive", "conservative"],
                    default="aggressive",
                    help="aggressive=Stage1拦截split+dirty；conservative=仅拦截微碎块dirty(<2%%)")
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
    stage2_calls = 0

    for scene in scenes:
        sd = os.path.join(scene_dir, scene)
        cam = json.load(open(os.path.join(sd, "scene_camera.json")))
        gt = json.load(open(os.path.join(sd, "scene_gt.json")))
        pose = make_pose_resolver(cam, gt)
        mdir = os.path.join(sd, "mask_visib")
        ddir = os.path.join(sd, "depth")

        pairs = pick_frame_pairs(cam, gt, max_pairs=args.pairs)
        if not pairs:
            continue

        for fa, fb in pairs:
            n_inst = len(gt[fa])
            idxs = []
            for idx in range(n_inst):
                pa = os.path.join(mdir, f"{int(fa):06d}_{idx:06d}.png")
                pb = os.path.join(mdir, f"{int(fb):06d}_{idx:06d}.png")
                if os.path.exists(pa) and os.path.exists(pb):
                    idxs.append(idx)
            if not idxs:
                continue

            K, RA, tA = pose(fa)
            _, RB, tB = pose(fb)
            depthA = np.array(Image.open(os.path.join(ddir, f"{int(fa):06d}.png"))).astype(np.float32) * args.depth_bias
            depthB = np.array(Image.open(os.path.join(ddir, f"{int(fb):06d}.png"))).astype(np.float32) * args.depth_bias
            ds = cam[fa].get("depth_scale", 0.1)
            H, W = depthA.shape

            sc_ok = sc_tot = 0
            for idx in idxs:
                maskA = load_gray(os.path.join(mdir, f"{int(fa):06d}_{idx:06d}.png"))
                maskB_gt = load_gray(os.path.join(mdir, f"{int(fb):06d}_{idx:06d}.png"))
                kind = ["clean", "split", "dirty", "sticky"][idx % 4]
                maskB = inject_defect(maskB_gt, kind, mdir, fb, idx, n_inst, H, W)
                is_conflict = kind != "clean"

                s1, s1_reason, dust_ratio = stage1_single_frame(maskB)
                if args.route == "aggressive":
                    intercept = s1 in ("split", "dirty")
                else:
                    intercept = (s1 == "dirty" and dust_ratio < 0.02)
                if intercept:
                    pred, action, s2 = True, "RESEGMENT", None
                else:
                    stage2_calls += 1
                    s2, n_vis = stage2_cross_view(maskA, maskB, depthA, depthB,
                                                  K, RA, tA, RB, tB, depth_scale=ds,
                                                  min_vis=args.min_vis)
                    if s2 is None:
                        continue
                    pred = s2 < THRESHOLD
                    action = "REGENERATE" if pred else "PASS"

                correct = pred == is_conflict
                sc_ok += int(correct)
                sc_tot += 1
                all_rows.append({
                    "scene": scene, "frame_a": fa, "frame_b": fb, "inst_idx": idx,
                    "defect": kind, "s1": s1, "s1_reason": s1_reason,
                    "s2_score": round(s2, 4) if s2 is not None else "",
                    "action": action, "pred_conflict": pred, "correct": correct,
                })
            acc = sc_ok / sc_tot if sc_tot else 0.0
            summary.append((scene, fa, fb, sc_ok, sc_tot, acc))
            print(f"[{scene}] 帧对 {fa}→{fb} | 实例 {sc_tot} | 准确率 {sc_ok}/{sc_tot} = {acc:.1%}")

    n = len(all_rows)
    tp = sum(1 for r in all_rows if r["pred_conflict"] and r["defect"] != "clean")
    tn = sum(1 for r in all_rows if not r["pred_conflict"] and r["defect"] == "clean")
    fp = sum(1 for r in all_rows if r["pred_conflict"] and r["defect"] == "clean")
    fn = sum(1 for r in all_rows if not r["pred_conflict"] and r["defect"] != "clean")
    acc = (tp + tn) / n if n else 0.0

    print("\n" + "=" * 66)
    print("级联质量校验调度器 · 12 场景汇总")
    print("=" * 66)
    print(f"参与统计实例数: {n}")
    print(f"级联准确率: {tp + tn}/{n} = {acc:.1%}")
    print(f"  冲突检出(TP): {tp} | 一致保留(TN): {tn} | 误报(FP): {fp} | 漏检(FN): {fn}")
    print(f"跨视图调用率: {stage2_calls}/{n} = {stage2_calls / n:.1%} "
          f"（Stage1 单帧拦截 {(n - stage2_calls) / n:.1%}）")

    by_kind = {}
    for r in all_rows:
        by_kind.setdefault(r["defect"], [0, 0])
        by_kind[r["defect"]][0] += 1
        if r["pred_conflict"]:
            by_kind[r["defect"]][1] += 1
    print("\n分缺陷检出率:")
    for k in ["clean", "split", "dirty", "sticky"]:
        tot, con = by_kind.get(k, [0, 0])
        rate = f"{con}/{tot} = {con / tot:.1%}" if tot else "-"
        print(f"  {k:6s}: {rate}")

    os.makedirs("results", exist_ok=True)
    with open(f"results/cascade_all_results{args.out_tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with open(f"results/cascade_all_report{args.out_tag}.txt", "w", encoding="utf-8") as f:
        f.write("级联质量校验调度器 · 12 场景汇总\n")
        f.write(f"准确率: {acc:.1%} | 跨视图调用率: {stage2_calls}/{n} = {stage2_calls / n:.1%}\n")
        for scene, fa, fb, ok, tot, a in summary:
            f.write(f"{scene}: 帧对 {fa}->{fb} | {ok}/{tot} = {a:.1%}\n")
        for k in ["clean", "split", "dirty", "sticky"]:
            tot, con = by_kind.get(k, [0, 0])
            f.write(f"{k}: {con}/{tot}\n")
    print(f"\n明细已保存到 results/cascade_all_results{args.out_tag}.csv / results/cascade_all_report{args.out_tag}.txt")


if __name__ == "__main__":
    main()
