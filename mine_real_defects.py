"""真实缺陷挖掘：SAM2 在 BOP 场景全实例跑分割，用 GT 对比自动标注天然缺陷。

与注入实验的关键区别：缺陷来自 SAM2 真实失误（漏边/粘连/碎块），
标签完全由 GT 对比规则产生（不用我们的检测器，避免循环论证）：

  sticky：SAM mask 与"另一个 GT 实例"的交集 > 其面积 10%
  split ：GT 被 SAM 覆盖不足 75%，且未覆盖部分含 ≥2 个连通域（各 ≥2% GT 面积）
  dirty ：SAM 超出 GT 的部分（且不属任何 GT 实例）含 ≥1 个 ≥2% SAM 面积的连通域
  clean ：以上都不满足且 IoU ≥ 0.85
  其余（IoU 低但不属于三类）标为 low_other

运行（SAM2 在 pytorch-env）：
    D:\\Anaconda_envs\\envs\\pytorch-env\\python.exe mine_real_defects.py [--scene-dir ...] [--pairs 3] [--out-tag _ycbv]

输出：data/sam2_mined{tag}/{scene}_{frame}_{idx}.png + results/mined_labels{tag}.csv
"""
import argparse, json, os, sys, time

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_eval_all import pick_frame_pairs  # noqa: E402

CKPT = r"D:\AAAApython\3Dpipelines_pytorch-env\sam2_hiera_large.pt"
CFG = "configs/sam2/sam2_hiera_l.yaml"
# 备选：sam2.1 tiny（--ckpt sam2.1_hiera_tiny.pt --cfg configs/sam2.1/sam2.1_hiera_t.yaml）


def bbox_of(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def components(mask):
    lab, n = ndimage.label(mask)
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return n, sizes


def auto_label(sam, gt, other_gts):
    """GT 对比规则标注（不借助检测器）。"""
    a = sam.sum()
    if a == 0:
        return "empty", 0.0
    inter = (sam & gt).sum()
    union = (sam | gt).sum()
    iou = inter / max(union, 1)
    cover = inter / max(gt.sum(), 1)

    # sticky：盖住别的 GT 实例 >10% 自身面积
    for j, og in other_gts:
        if (sam & og).sum() > 0.1 * a:
            return "sticky", iou
    # split：GT 覆盖不足且残余 ≥2 个连通块
    if cover < 0.75:
        residual = gt & ~sam
        n, sizes = components(residual)
        if (sizes >= 0.02 * gt.sum()).sum() >= 2:
            return "split", iou
    # dirty：超出 GT 的游离块（不属任何 GT 实例）≥2%
    excess = sam & ~gt
    for _, og in other_gts:
        excess = excess & ~og
    n, sizes = components(excess)
    if n >= 1 and sizes.max() >= 0.02 * a:
        return "dirty", iou
    if iou >= 0.85:
        return "clean", iou
    return "low_other", iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", default="data/real_data/test")
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--out-tag", default="_ycbv")
    ap.add_argument("--ckpt", default=CKPT, help="SAM2 权重路径")
    ap.add_argument("--cfg", default=CFG, help="SAM2 配置（与权重匹配）")
    args = ap.parse_args()

    print(f"加载 SAM2（CPU）: {os.path.basename(args.ckpt)}", flush=True)
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(build_sam2(args.cfg, args.ckpt, device="cpu"))

    out_dir = f"data/sam2_mined{args.out_tag}"
    os.makedirs(out_dir, exist_ok=True)
    scenes = sorted(d for d in os.listdir(args.scene_dir)
                    if os.path.isdir(os.path.join(args.scene_dir, d)))
    rows = []
    t_start = time.time()

    for scene in scenes:
        sd = os.path.join(args.scene_dir, scene)
        cam = json.load(open(os.path.join(sd, "scene_camera.json")))
        gt = json.load(open(os.path.join(sd, "scene_gt.json")))
        mdir = os.path.join(sd, "mask_visib")
        pairs = pick_frame_pairs(cam, gt, max_pairs=args.pairs)
        frames = sorted({int(f) for p in pairs for f in p})

        for frame in frames:
            rgb_path = os.path.join(sd, "rgb", f"{frame:06d}.png")
            if not os.path.exists(rgb_path):
                continue
            predictor.set_image(np.array(Image.open(rgb_path).convert("RGB")))
            insts = gt.get(str(frame), [])
            masks_gt = []
            for idx in range(len(insts)):
                p = os.path.join(mdir, f"{frame:06d}_{idx:06d}.png")
                m = np.array(Image.open(p)) > 0 if os.path.exists(p) else None
                masks_gt.append(m)

            for idx, gm in enumerate(masks_gt):
                if gm is None or gm.sum() < 50:
                    continue
                box = bbox_of(gm)
                pred_masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=False)
                sam = pred_masks[0] > 0
                others = [(j, m) for j, m in enumerate(masks_gt)
                          if j != idx and m is not None]
                label, iou = auto_label(sam, gm, others)
                Image.fromarray(sam.astype(np.uint8) * 255).save(
                    os.path.join(out_dir, f"{scene}_{frame:06d}_{idx:06d}.png"))
                rows.append({"scene": scene, "frame": frame, "idx": idx,
                             "label": label, "iou": round(iou, 4),
                             "score": round(float(scores[0]), 4)})
            print(f"[{scene}] f{frame} 完成（累计 {len(rows)}，{(time.time()-t_start)/60:.1f}min）",
                  flush=True)

    import csv
    os.makedirs("results", exist_ok=True)
    with open(f"results/mined_labels{args.out_tag}.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    cnt = Counter(r["label"] for r in rows)
    print(f"\n共 {len(rows)} 实例 | 标签分布: {dict(cnt)}")
    print(f"标签已存 results/mined_labels{args.out_tag}.csv")


if __name__ == "__main__":
    main()
