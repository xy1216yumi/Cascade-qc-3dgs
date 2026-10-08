"""级联在真实缺陷集（mine_real_defects 产出）上的评测。

与注入实验的唯一差别：mask 来自 SAM2 真实输出（data/sam2_mined{tag}/），
标签来自 GT 对比规则（results/mined_labels{tag}.csv），全程无人工注入。
指标：缺陷二分类（clean vs 缺陷）准确率 / TP/TN/FP/FN + 分类型检出率。

用法：python eval_mined_cascade.py [--scene-dir ...] [--pairs 3] [--out-tag _ycbv] [--min-vis 300] [--depth-bias 1.0]
输出：results/mined_cascade{tag}.csv + 汇总打印
"""
import argparse, csv, json, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_checker import load_gray, make_pose_resolver  # noqa: E402
from cascade_checker import stage1_single_frame, stage2_cross_view, THRESHOLD  # noqa: E402
from mv_eval_all import pick_frame_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", default="data/real_data/test")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--out-tag", default="_ycbv")
    ap.add_argument("--min-vis", type=int, default=300)
    ap.add_argument("--depth-bias", type=float, default=1.0)
    args = ap.parse_args()

    labels = {}
    for r in csv.DictReader(open(f"results/mined_labels{args.out_tag}.csv", encoding="utf-8-sig")):
        labels[(r["scene"], int(r["frame"]), int(r["idx"]))] = r["label"]

    sam_dir = f"data/sam2_mined{args.out_tag}"
    scenes = sorted(d for d in os.listdir(args.scene_dir)
                    if os.path.isdir(os.path.join(args.scene_dir, d)))
    rows = []
    skipped = 0

    for scene in scenes:
        sd = os.path.join(args.scene_dir, scene)
        cam = json.load(open(os.path.join(sd, "scene_camera.json")))
        gt = json.load(open(os.path.join(sd, "scene_gt.json")))
        pose = make_pose_resolver(cam, gt)
        ddir = os.path.join(sd, "depth")

        for fa, fb in pick_frame_pairs(cam, gt, max_pairs=args.pairs):
            n_inst = len(gt[fa])
            K, RA, tA = pose(fa)
            _, RB, tB = pose(fb)
            ds = cam[fa].get("depth_scale", 0.1)
            dA = np.array(Image.open(os.path.join(ddir, f"{int(fa):06d}.png"))).astype(np.float32) * args.depth_bias
            dB = np.array(Image.open(os.path.join(ddir, f"{int(fb):06d}.png"))).astype(np.float32) * args.depth_bias

            for idx in range(n_inst):
                lab_b = labels.get((scene, int(fb), idx))
                if lab_b is None or lab_b in ("empty", "low_other"):
                    continue
                pa = os.path.join(sam_dir, f"{scene}_{int(fa):06d}_{idx:06d}.png")
                pb = os.path.join(sam_dir, f"{scene}_{int(fb):06d}_{idx:06d}.png")
                if not os.path.exists(pa) or not os.path.exists(pb):
                    continue
                maskA = load_gray(pa)
                maskB = load_gray(pb)

                s1, _, _ = stage1_single_frame(maskB)
                if s1 in ("split", "dirty"):
                    pred, action, s2 = True, "RESEGMENT", None
                else:
                    s2, _ = stage2_cross_view(maskA, maskB, dA, dB, K, RA, tA, RB, tB,
                                              depth_scale=ds, min_vis=args.min_vis)
                    if s2 is None:
                        skipped += 1
                        continue
                    pred = s2 < THRESHOLD
                    action = "REGENERATE" if pred else "PASS"

                is_conflict = lab_b != "clean"
                rows.append({"scene": scene, "fa": fa, "fb": fb, "idx": idx,
                             "label": lab_b, "s1": s1,
                             "s2": round(s2, 4) if s2 is not None else "",
                             "action": action, "pred": pred,
                             "correct": pred == is_conflict})

    n = len(rows)
    tp = sum(1 for r in rows if r["pred"] and r["label"] != "clean")
    tn = sum(1 for r in rows if not r["pred"] and r["label"] == "clean")
    fp = sum(1 for r in rows if r["pred"] and r["label"] == "clean")
    fn = sum(1 for r in rows if not r["pred"] and r["label"] != "clean")
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9)

    print("=" * 66)
    print(f"级联 · 真实缺陷集评测（{n} 实例，跳过低采样 {skipped}）")
    print("=" * 66)
    print(f"准确率 {(tp+tn)/n:.1%} | TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"精确率 {P:.2f} | 召回率 {R:.2f} | F1 {F1:.2f}")
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["label"], [0, 0])
        by_kind[r["label"]][0] += 1
        if r["pred"]:
            by_kind[r["label"]][1] += 1
    for k in ["clean", "split", "dirty", "sticky"]:
        if k in by_kind:
            tot, con = by_kind[k]
            print(f"  {k:6s}: {con}/{tot} = {con/tot:.1%}")

    with open(f"results/mined_cascade{args.out_tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n明细已存 results/mined_cascade{args.out_tag}.csv")


if __name__ == "__main__":
    main()
