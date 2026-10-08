"""物体区域 PSNR：渲染图 vs GT，只在 GT 物体 mask 内计算。

用法：python eval_psnr_obj.py --renders results/render_output/000048_obj0_gt \
      --data_dir data/cloud_data/000048 --mask_dir data/cloud_data/000048/obj0/gt
"""
import argparse, os
import numpy as np
from PIL import Image

from eval_psnr import read_image_names, psnr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--mask_dir", required=True)
    args = ap.parse_args()

    names = read_image_names(f"{args.data_dir}/sparse/0/images.txt")
    vals = []
    for i, name in enumerate(names):
        rp = os.path.join(args.renders, f"render_{i:03d}.png")
        mp = os.path.join(args.mask_dir, os.path.splitext(name)[0] + ".png")
        if not os.path.exists(rp) or not os.path.exists(mp):
            continue
        m = np.array(Image.open(mp).convert("L")) > 127
        if m.sum() < 50:
            continue
        r = np.array(Image.open(rp).convert("RGB")).astype(np.float64)
        g = np.array(Image.open(os.path.join(args.data_dir, "images", name)).convert("RGB")).astype(np.float64)
        mse = (((r - g) ** 2)[m]).mean()
        vals.append(20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float("inf"))

    print(f"{args.renders}: 物体区域 PSNR 平均 {np.mean(vals):.2f} dB "
          f"（{len(vals)} 帧，中位 {np.median(vals):.2f}）")


if __name__ == "__main__":
    main()
