"""真实缺陷检测 v3：prompt 抖动集成分歧信号。

原理：对"难区域"，分割器对微小扰动敏感。同一实例用 5 个抖动框（±8% 宽高扰动）
跑 SAM2，两两 mask 的平均 IoU 作为稳定性分数——分歧大 = 该区域难分割 = 可能有缺陷。
信号无需 GT，可与几何/置信度互补。

运行（SAM2 在 pytorch-env）：
    D:\\Anaconda_envs\\envs\\pytorch-env\\python.exe mine_jitter_ensemble.py [--out-tag _ycbv]

输出：results/jitter_ensemble{tag}.csv + 融合评估打印
"""
import argparse, csv, json, os, sys, time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_baselines import auc  # noqa: E402

CKPT = r"D:\AAAApython\3Dpipelines_pytorch-env\sam2_hiera_large.pt"
CFG = "configs/sam2/sam2_hiera_l.yaml"
N_JITTER = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tag", default="_ycbv")
    args = ap.parse_args()

    labels = {}
    for r in csv.DictReader(open(f"results/mined_labels{args.out_tag}.csv", encoding="utf-8-sig")):
        labels[(r["scene"], int(r["frame"]), int(r["idx"]))] = (r["label"], float(r["iou"]))
    rows_in = list(csv.DictReader(open(f"results/mined_cascade{args.out_tag}.csv", encoding="utf-8-sig")))

    print(f"抖动集成：{len(rows_in)} 实例 × {N_JITTER} 次推理。加载 SAM2...", flush=True)
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(build_sam2(CFG, CKPT, device="cpu"))

    rng = np.random.default_rng(42)
    rows = []
    img_cache = {}
    for n, r in enumerate(rows_in):
        scene, fb, idx = r["scene"], int(r["fb"]), int(r["idx"])
        lab, iou = labels[(scene, fb, idx)]
        sd = os.path.join("data/real_data/test", scene)
        mdir = os.path.join(sd, "mask_visib")
        gt_mask = np.array(Image.open(os.path.join(mdir, f"{fb:06d}_{idx:06d}.png"))) > 0
        ys, xs = np.where(gt_mask)
        box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)

        if fb not in img_cache:
            rgb = np.array(Image.open(os.path.join(sd, "rgb", f"{fb:06d}.png")).convert("RGB"))
            predictor.set_image(rgb)
            img_cache.clear(); img_cache[fb] = True

        w, h = box[2] - box[0], box[3] - box[1]
        masks = []
        for j in range(N_JITTER):
            if j == 0:
                jb = box
            else:
                cxy = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
                nw, nh = w * rng.uniform(0.92, 1.08), h * rng.uniform(0.92, 1.08)
                ox, oy = rng.uniform(-0.04, 0.04) * w, rng.uniform(-0.04, 0.04) * h
                jb = np.array([cxy[0] - nw / 2 + ox, cxy[1] - nh / 2 + oy,
                               cxy[0] + nw / 2 + ox, cxy[1] + nh / 2 + oy])
            pm, _, _ = predictor.predict(box=jb[None, :], multimask_output=False)
            masks.append(pm[0] > 0)

        ious = []
        for a in range(N_JITTER):
            for b in range(a + 1, N_JITTER):
                inter = (masks[a] & masks[b]).sum()
                union = (masks[a] | masks[b]).sum()
                ious.append(inter / max(union, 1))
        stability = float(np.mean(ious))
        rows.append({"scene": scene, "fb": fb, "idx": idx, "label": lab, "iou": iou,
                     "stability": round(stability, 4)})
        if (n + 1) % 20 == 0:
            print(f"[{n+1}/{len(rows_in)}] ...", flush=True)

    with open(f"results/jitter_ensemble{args.out_tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows)

    y = np.array([(r["label"] != "clean") and (r["iou"] < 0.8) for r in rows])
    s = -np.array([r["stability"] for r in rows])  # 越不稳定越可能有缺陷
    print(f"\n抖动分歧 AUC（严重缺陷，{y.sum()}/{len(y)}）: {auc(s, y):.3f}")
    print(f"明细已存 results/jitter_ensemble{args.out_tag}.csv")


if __name__ == "__main__":
    main()
