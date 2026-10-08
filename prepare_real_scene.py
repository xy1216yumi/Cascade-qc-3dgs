"""真实场景验证数据制作：BOP 格式 (RGB + mask_visib) → 缺陷注入 → 检查器三件套 + ground truth。

为什么需要缺陷注入：真实数据集的实例 mask 标注是正确的（全 clean），检查器无法直接测。
本脚本把 YCB-V 等真实场景的 GT mask 人为注入四类缺陷（与合成实验同构）：
- clean  : GT mask 原样（真实物体轮廓）
- split  : GT mask 沿水平中线切两半、留缺口
- dirty  : GT mask 膨胀 12% + 右上角游离杂块
- sticky : 合并两个相邻实例的 GT mask 为一个编号（一个 mask 盖住两个真实物体）

输出（与 make_test_images.py 完全同构，可直接跑 geo_checker / batch_check / evaluate）：
- real_images/   {scene}_{frame}.png      叠加图（真实照片 + 半透明 mask + 编号）
- real_objects/  {scene}_{frame}.png      无 mask 原图
- real_masks/    {scene}_{frame}_instance{i}.png  注入缺陷后的真 mask（模拟 SAM2 输出）
- real_answers.csv                         注入缺陷的 ground truth

用法：
    python prepare_real_scene.py --test-dir real_data/ycbv/test \
        --scenes 000001,000002 --frames 000000 --max-instances 5
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MASK_COLORS = [(255, 107, 129), (255, 169, 77), (77, 171, 247),
               (252, 196, 25), (81, 207, 102), (177, 151, 252)]
ALPHA = 165  # 真实照片纹理复杂，mask 要比合成图更实才看得清

FONT_CANDIDATES = [
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeuib.ttf",
    "C:/Windows/Fonts/msyhbd.ttc",
]


def get_font(size: int):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def dilate(mask, radius):
    """简单膨胀：3x3 max filter 迭代 radius 次。"""
    m = mask
    for _ in range(radius):
        padded = np.pad(m, 1)
        m = np.maximum.reduce([
            padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:],
            padded[1:-1, :-2], padded[1:-1, 1:-1], padded[1:-1, 2:],
            padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:],
        ])
    return m


def split_mask(mask):
    """沿 bbox 水平中线切两半、留缺口。返回新 mask。"""
    ys, xs = np.where(mask > 0)
    y0, y1 = ys.min(), ys.max()
    mid = (y0 + y1) // 2
    gap = max(4, int((y1 - y0) * 0.1))
    out = mask.copy()
    out[max(0, mid - gap // 2):mid + gap // 2 + 1, :] = 0
    return out


def add_blob(mask, img_shape):
    """在 mask 右上角外加一个小杂块（模拟分割毛刺）。"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return mask
    x1, y0 = xs.max(), ys.min()
    h = ys.max() - y0
    out = mask.copy()
    rx0, ry0 = min(x1 + 8, img_shape[1] - 20), max(0, y0 - int(h * 0.35))
    rx1, ry1 = min(x1 + 30, img_shape[1]), max(1, y0 - int(h * 0.15))
    for y in range(ry0, ry1):
        for x in range(rx0, rx1):
            if (x - rx0) ** 2 + (y - ry0) ** 2 < 64:
                out[y, x] = 255
    return out


def instance_center(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0, 0
    return int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2)


def draw_overlay(rgb, masks_with_color_and_id):
    """在真实照片上叠加半透明 mask + 2px 边框 + 编号。

    注意：alpha_composite 返回新对象，编号必须等全部 mask 合成完成后再统一画，
    否则会画在被丢弃的旧 Image 对象上（已踩坑）。
    """
    img = rgb.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    for iid, mask, color in masks_with_color_and_id:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        # 半透明 mask 填充
        mask_rgba = np.zeros((img.size[1], img.size[0], 4), dtype=np.uint8)
        mask_rgba[ys, xs] = (*color, ALPHA)
        overlay = Image.alpha_composite(overlay, Image.fromarray(mask_rgba, "RGBA"))
        # 2px 外扩边框（沿 mask 轮廓外圈，比像素内圈更醒目）
        ring = dilate((mask > 0).astype(np.uint8), 2) & ~(mask > 0)
        ry, rx = np.where(ring)
        if len(rx) > 0:
            edge_rgba = np.zeros((img.size[1], img.size[0], 4), dtype=np.uint8)
            edge_rgba[ry, rx] = (*color, 255)
            overlay = Image.alpha_composite(overlay, Image.fromarray(edge_rgba, "RGBA"))
    # 全部 mask/边框合成完后，统一画编号（白字黑描边，加大字号）
    draw = ImageDraw.Draw(overlay)
    font = get_font(52)
    for iid, mask, color in masks_with_color_and_id:
        cx, cy = instance_center(mask)
        draw.text((cx, cy), str(iid), font=font, fill=(255, 255, 255, 255),
                  anchor="mm", stroke_width=4, stroke_fill=(20, 20, 20, 255))
    return Image.alpha_composite(img, overlay).convert("RGB")


def load_mask(p):
    return np.array(Image.open(p).convert("L"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", default="data/real_data/test", help="BOP test 目录（含场景子目录）")
    ap.add_argument("--scenes", default="000001,000002", help="场景列表，逗号分隔")
    ap.add_argument("--frames", default="000000", help="每个场景取的帧，逗号分隔")
    ap.add_argument("--max-instances", type=int, default=5, help="每帧最多取几个实例")
    ap.add_argument("--out-dir", default="data/real_images")
    ap.add_argument("--objects-dir", default="data/real_objects")
    ap.add_argument("--masks-dir", default="data/real_masks")
    ap.add_argument("--answers", default="results/real_answers.csv")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.objects_dir, exist_ok=True)
    os.makedirs(args.masks_dir, exist_ok=True)

    rows = []
    counter = {"clean": 0, "dirty": 0, "split": 0, "sticky": 0}
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    auto_frames = args.frames.strip().lower() in ("", "auto")
    frames = [] if auto_frames else [f.strip() for f in args.frames.split(",") if f.strip()]

    for scene in scenes:
        sdir = os.path.join(args.test_dir, scene)
        if not os.path.isdir(sdir):
            print(f"[跳过] 场景 {scene} 不存在: {sdir}")
            continue
        if auto_frames:
            # 自动取该场景 rgb 目录下按字典序前 2 个存在的帧
            rgb_files = sorted(glob.glob(os.path.join(sdir, "rgb", "*.png")))
            frames = [os.path.splitext(os.path.basename(p))[0] for p in rgb_files[:2]]
        for frame in frames:
            rgb_path = os.path.join(sdir, "rgb", f"{frame}.png")
            if not os.path.exists(rgb_path):
                print(f"[跳过] 帧 {scene}/{frame} 不存在 rgb")
                continue
            # 读取该帧全部实例 mask，按面积排序取最大的 N 个
            mdir = os.path.join(sdir, "mask_visib")
            cands = []
            for mp in glob.glob(os.path.join(mdir, f"{frame}_*.png")):
                objid = int(re.search(r"_(\d+)\.png$", mp).group(1))
                m = load_mask(mp)
                if m.sum() < 500:  # 过滤极小 mask
                    continue
                cands.append((objid, m))
            cands.sort(key=lambda t: -int(t[1].sum()))
            cands = cands[: args.max_instances]
            if len(cands) < 3:
                print(f"[跳过] {scene}/{frame} 有效实例不足 ({len(cands)})")
                continue

            rgb = Image.open(rgb_path).convert("RGB")
            rgb.save(os.path.join(args.objects_dir, f"{scene}_{frame}.png"))

            # 固定缺陷分配（保证每帧四类尽量齐全）：
            #   1=clean, 2=split, 3=dirty, 4&5=sticky(两实例合并)，其余 clean
            n = len(cands)
            if n >= 5:
                kinds = {0: "clean", 1: "split", 2: "dirty", 3: "sticky", 4: "sticky"}
                for i in range(5, n):
                    kinds[i] = "clean"
            elif n == 4:
                kinds = {0: "clean", 1: "split", 2: "sticky", 3: "sticky"}
            else:
                kinds = {i: ["clean", "split", "dirty"][i] for i in range(n)}
            sticky_idx = [i for i, k in kinds.items() if k == "sticky"]

            masks = {}   # iid -> (缺陷mask, note)
            for i, (objid, gt) in enumerate(cands, start=1):
                kind = kinds[i - 1]
                if kind == "sticky":
                    other = sticky_idx[1] if (i - 1) == sticky_idx[0] else sticky_idx[0]
                    gt_other = cands[other][1]
                    m = np.maximum(gt, gt_other)
                    masks[i] = (m, "sticky")
                    counter["sticky"] += 1
                elif kind == "split":
                    masks[i] = (split_mask(gt), "split")
                    counter["split"] += 1
                elif kind == "dirty":
                    m = dilate(gt, 2)
                    m = add_blob(m, (rgb.size[1], rgb.size[0]))
                    masks[i] = (m, "dirty")
                    counter["dirty"] += 1
                else:
                    masks[i] = (gt, "clean")
                    counter["clean"] += 1

            # 输出真 mask + overlay + ground truth
            color_idx = 0
            for i in sorted(masks):
                m, note = masks[i]
                Image.fromarray(np.where(m > 0, 255, 0).astype(np.uint8)).save(
                    os.path.join(args.masks_dir, f"{scene}_{frame}_instance{i}.png"))
                # overlay 颜色按实例轮流（同图不重复）
                color = MASK_COLORS[color_idx % len(MASK_COLORS)]
                color_idx += 1
                # 记录 overlay 用的颜色（供后续？仅打印）
            overlay = draw_overlay(rgb, [(i, m, MASK_COLORS[(k) % len(MASK_COLORS)])
                                         for k, (i, (m, _)) in enumerate(sorted(masks.items()))])
            overlay.save(os.path.join(args.out_dir, f"{scene}_{frame}.png"))

            for i in sorted(masks):
                _, note = masks[i]
                rows.append({
                    "image_id": os.path.join(args.out_dir, f"{scene}_{frame}.png"),
                    "instance_id": i,
                    "is_clean": "是" if note == "clean" else "否",
                    "is_split": "是" if note == "split" else "否",
                    "is_sticky": "是" if note == "sticky" else "否",
                    "note": note,
                })
            print(f"[完成] {scene}/{frame}：{len(masks)} 个实例 -> {masks.values() and [n for _, n in masks.values()]}")

    with open(args.answers, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "instance_id", "is_clean", "is_split", "is_sticky", "note"])
        w.writeheader()
        w.writerows(rows)
    total = sum(counter.values())
    print(f"\n真实场景数据制作完成：{total} 个实例 -> {args.out_dir}/  {args.masks_dir}/  {args.objects_dir}/")
    print("实例分布:", counter)
    print("\n接下来运行：")
    print(f"  python batch_check.py --input {args.out_dir} --output real_results.jsonl")
    print(f"  python evaluate.py --answers {args.answers} --predictions real_results.jsonl")


if __name__ == "__main__":
    main()
