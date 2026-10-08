"""真实缺陷检测 v2：多帧跨视图投票 + 留出集阈值校准。

动机：单参照帧的跨视图命中率受深度噪声影响大；真实缺陷在多视角下持续存在，
噪声则随机。对目标帧的 SAM2 mask，用其余所有已挖掘帧（同场景、同实例、每场景 6 帧）
逐一做跨视图一致性打分，取中位数聚合——显著缩小 clean/severe 分布重叠。

校准合法性：按场景分半，calib 场景选阈值，test 场景报告指标（不按答案调参）。

用法：python eval_mined_multiframe.py [--out-tag _ycbv] [--min-vis 50]
输出：results/mined_multiframe{tag}.csv + 汇总打印
"""
import argparse, csv, json, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_checker import load_gray, make_pose_resolver  # noqa: E402
from cascade_checker import stage1_single_frame, stage2_cross_view  # noqa: E402
from mv_eval_all import pick_frame_pairs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", default="data/real_data/test")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--out-tag", default="_ycbv")
    ap.add_argument("--min-vis", type=int, default=50)
    args = ap.parse_args()

    labels = {}
    for r in csv.DictReader(open(f"results/mined_labels{args.out_tag}.csv", encoding="utf-8-sig")):
        labels[(r["scene"], int(r["frame"]), int(r["idx"]))] = (r["label"], float(r["iou"]), float(r["score"]))

    sam_dir = f"data/sam2_mined{args.out_tag}"
    scenes = sorted(d for d in os.listdir(args.scene_dir)
                    if os.path.isdir(os.path.join(args.scene_dir, d)))
    rows = []

    for scene in scenes:
        sd = os.path.join(args.scene_dir, scene)
        cam = json.load(open(os.path.join(sd, "scene_camera.json")))
        gt = json.load(open(os.path.join(sd, "scene_gt.json")))
        pose = make_pose_resolver(cam, gt)
        ddir = os.path.join(sd, "depth")
        ds = list(cam.values())[0].get("depth_scale", 0.1)

        pairs = pick_frame_pairs(cam, gt, max_pairs=args.pairs)
        frames = sorted({f for p in pairs for f in p}, key=int)  # cam 的字符串键
        inst_ids = sorted({idx for (s, f, idx) in labels if s == scene})

        cache = {}
        def load_frame(f):
            if f not in cache:
                d = np.array(Image.open(os.path.join(ddir, f"{int(f):06d}.png"))).astype(np.float32)
                cache[f] = (pose(f), d)
            return cache[f]

        for fb in frames:
            (Kb, RB, tB), dB = load_frame(fb)
            for idx in inst_ids:
                key = (scene, int(fb), idx)
                if key not in labels:
                    continue
                pb = os.path.join(sam_dir, f"{scene}_{int(fb):06d}_{idx:06d}.png")
                if not os.path.exists(pb):
                    continue
                maskB = load_gray(pb)
                s1, _, _ = stage1_single_frame(maskB)

                scores = []
                for fr in frames:
                    if fr == fb:
                        continue
                    pr = os.path.join(sam_dir, f"{scene}_{int(fr):06d}_{idx:06d}.png")
                    if not os.path.exists(pr):
                        continue
                    (Kr, RR, tR), dR = load_frame(fr)
                    maskR = load_gray(pr)
                    s2, _ = stage2_cross_view(maskR, maskB, dR, dB, Kb, RR, tR, RB, tB,
                                              depth_scale=ds, min_vis=args.min_vis)
                    if s2 is not None:
                        scores.append(s2)
                if not scores:
                    continue
                lab, iou, conf = labels[key]
                rows.append({
                    "scene": scene, "fb": fb, "idx": idx, "label": lab, "iou": iou,
                    "conf": conf, "s1": s1, "n_ref": len(scores),
                    "vote_med": round(float(np.median(scores)), 4),
                    "vote_min": round(float(np.min(scores)), 4),
                    "vote_p25": round(float(np.percentile(scores, 25)), 4),
                })
        print(f"[{scene}] 完成（累计 {len(rows)}）", flush=True)

    # 留出集校准：按场景分半
    calib_scenes = set(scenes[: len(scenes) // 2])
    for r in rows:
        r["severe"] = (r["label"] != "clean") and (r["iou"] < 0.8)

    def eval_rule(pred_fn, calib_rows, test_rows, thr_choices):
        yc = np.array([r["severe"] for r in calib_rows])
        best_t, best_f1 = None, -1
        for t in thr_choices:
            p = pred_fn(t, calib_rows)
            tp = (p & yc).sum(); fp = (p & ~yc).sum(); fn = (~p & yc).sum()
            P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
            F1 = 2 * P * R / max(P + R, 1e-9)
            if F1 > best_f1:
                best_f1, best_t = F1, t
        yt = np.array([r["severe"] for r in test_rows])
        p = pred_fn(best_t, test_rows)
        tp = int((p & yt).sum()); fp = int((p & ~yt).sum()); fn = int((~p & yt).sum()); tn = int((~p & ~yt).sum())
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
        F1 = 2 * P * R / max(P + R, 1e-9)
        return best_t, P, R, F1, (tp + tn) / max(len(test_rows), 1), (tp, tn, fp, fn)

    calib = [r for r in rows if r["scene"] in calib_scenes]
    test = [r for r in rows if r["scene"] not in calib_scenes]

    print("\n" + "=" * 66)
    print(f"多帧投票 · 真实缺陷检测（calib {len(calib)} / test {len(test)} 实例）")
    print("=" * 66)
    rules = [
        ("vote_med", lambda t, rs: np.array([r["vote_med"] < t for r in rs]),
         np.arange(0.5, 1.0, 0.05)),
        ("vote_p25", lambda t, rs: np.array([r["vote_p25"] < t for r in rs]),
         np.arange(0.4, 1.0, 0.05)),
        ("vote_med + stage1", lambda t, rs: np.array([(r["vote_med"] < t) or (r["s1"] in ("split", "dirty")) for r in rs]),
         np.arange(0.5, 1.0, 0.05)),
        ("vote_med + stage1 + 置信度", lambda t, rs: np.array([(r["vote_med"] < t) or (r["s1"] in ("split", "dirty")) or (r["conf"] < 0.9675) for r in rs]),
         np.arange(0.5, 1.0, 0.05)),
    ]
    for name, fn, ts in rules:
        t, P, R, F1, acc, (tp, tn, fp, fn_) = eval_rule(fn, calib, test, ts)
        print(f"{name:<26s} τ={t:.2f} | P={P:.2f} R={R:.2f} F1={F1:.2f} acc={acc:.1%}")

    with open(f"results/mined_multiframe{args.out_tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n明细已存 results/mined_multiframe{args.out_tag}.csv")


if __name__ == "__main__":
    main()
