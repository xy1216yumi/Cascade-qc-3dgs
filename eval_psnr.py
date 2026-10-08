"""渲染图 vs GT 的 PSNR 评估。

用法：python eval_psnr.py --renders output/000048 --data_dir cloud_data/000048
按 sparse/0/images.txt 的顺序配对 render_{i:03d}.png ↔ images/{name}。
"""
import argparse, os
import numpy as np
from PIL import Image


def read_image_names(path):
    names = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 10 and not line.startswith("#"):
                names.append(p[9])
    return names


def psnr(a, b):
    mse = ((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean()
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    names = read_image_names(f"{args.data_dir}/sparse/0/images.txt")
    rows = []
    for i, name in enumerate(names):
        rp = os.path.join(args.renders, f"render_{i:03d}.png")
        gp = os.path.join(args.data_dir, "images", name)
        if not os.path.exists(rp):
            continue
        r = np.array(Image.open(rp).convert("RGB"))
        g = np.array(Image.open(gp).convert("RGB"))
        if r.shape != g.shape:
            r = np.array(Image.open(rp).convert("RGB").resize((g.shape[1], g.shape[0])))
        rows.append((i, name, psnr(r, g)))

    vals = [v for _, _, v in rows]
    for i, name, v in rows:
        print(f"frame {i:03d} ({name}): PSNR = {v:.2f} dB")
    print(f"\n共 {len(rows)} 帧，平均 PSNR = {np.mean(vals):.2f} dB，"
          f"中位 {np.median(vals):.2f}，最低 {np.min(vals):.2f}")

    if args.out_csv:
        with open(args.out_csv, "w") as f:
            f.write("frame,image,psnr\n")
            for i, name, v in rows:
                f.write(f"{i},{name},{v:.4f}\n")
        print(f"明细已保存 {args.out_csv}")


if __name__ == "__main__":
    main()
