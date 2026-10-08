"""Fig.1 motivation 图：GT vs 系统性 split 缺陷训练的渲染对比（000048 obj0）。

自动选取 split100 与 gt 渲染差异最大的帧，裁剪物体区域，三栏并排输出。
用法：python make_fig1.py [--scene 000048] [--out docs/figures/fig1_motivation.png]
"""
import argparse, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eval_psnr import read_image_names, psnr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="000048")
    ap.add_argument("--out", default="docs/figures/fig1_motivation.png")
    args = ap.parse_args()

    data_dir = f"data/cloud_data/{args.scene}"
    names = read_image_names(f"{data_dir}/sparse/0/images.txt")
    gt_mask_dir = f"{data_dir}/obj0/gt"

    # 选 gt 渲染明显好于 split 渲染的帧（物体区域内 PSNR 差最大）
    def mpsnr(render, gt_np, m):
        mse = (((render.astype(np.float64) - gt_np.astype(np.float64)) ** 2)[m]).mean()
        return 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else 99.0

    best = (None, -99)
    for i, name in enumerate(names):
        mp = os.path.join(gt_mask_dir, os.path.splitext(name)[0] + ".png")
        if not os.path.exists(mp):
            continue
        m = np.array(Image.open(mp).convert("L")) > 127
        if m.sum() < 50:
            continue
        gt_np = np.array(Image.open(f"{data_dir}/images/{name}").convert("RGB"))
        a = np.array(Image.open(f"results/render_output/{args.scene}_obj0_gt/render_{i:03d}.png").convert("RGB"))
        b = np.array(Image.open(f"results/render_output/{args.scene}_obj0_split100/render_{i:03d}.png").convert("RGB"))
        diff = mpsnr(a, gt_np, m) - mpsnr(b, gt_np, m)
        if diff > best[1]:
            best = (i, diff)
    i = best[0]
    name = names[i]
    print(f"选择帧 {i}（{name}），物体区 PSNR 差={best[1]:.2f}dB")

    mp = os.path.join(gt_mask_dir, os.path.splitext(name)[0] + ".png")
    m = np.array(Image.open(mp).convert("L")) > 127
    ys, xs = np.where(m)
    pad = 40
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, m.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, m.shape[1])

    gt_img = Image.open(f"{data_dir}/images/{name}").convert("RGB")
    r_gt = Image.open(f"results/render_output/{args.scene}_obj0_gt/render_{i:03d}.png").convert("RGB")
    r_sp = Image.open(f"results/render_output/{args.scene}_obj0_split100/render_{i:03d}.png").convert("RGB")

    g_psnr = mpsnr(np.array(r_gt), np.array(gt_img), m)
    s_psnr = mpsnr(np.array(r_sp), np.array(gt_img), m)

    crops = [img.crop((x0, y0, x1, y1)) for img in (gt_img, r_gt, r_sp)]
    w, h = crops[0].size
    label_h = 70
    canvas = Image.new("RGB", (w * 3 + 40, h + label_h + 10), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
        font_big = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = font_big = ImageFont.load_default()

    titles = ["Input photo (GT)", "Trained w/ GT masks", "Trained w/ split masks"]
    subs = ["", f"PSNR {g_psnr:.1f} dB", f"PSNR {s_psnr:.1f} dB"]
    for k, (c, t, s) in enumerate(zip(crops, titles, subs)):
        x = k * (w + 20)
        canvas.paste(c, (x, 0))
        draw.text((x + w // 2, h + 12), t, font=font_big, fill="black", anchor="mm")
        if s:
            draw.text((x + w // 2, h + 42), s, font=font,
                      fill="darkred" if k == 2 else "darkgreen", anchor="mm")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    canvas.save(args.out)
    print(f"已保存 {args.out}（gt {g_psnr:.2f}dB vs split {s_psnr:.2f}dB）")


if __name__ == "__main__":
    main()
