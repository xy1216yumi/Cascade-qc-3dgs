"""跨视图一致性测试集制作：BOP 场景帧对 + 实例 → 帧B mask 注入缺陷 → ground truth。

背景：单帧检查器无法区分"sticky（两物体合并成一块）"与"split（一物体断开成两块）"。
跨视图一致性检查的思路：同一 3D 物体在多视图的 mask 应彼此吻合（重投影一致）。
本脚本模拟"SAM2 在同一实例的不同帧输出不稳定的 mask"：帧A 用 GT（正确），
帧B 注入缺陷（clean/split/sticky/dirty），生成跨视图测试集。

输出：
- mv_masks/{scene}_{fA}_{fB}_idx{i}.png   帧B 的"待检查 mask"（可能含缺陷）
- mv_answers.csv                           ground truth：is_conflict = defect != clean
- （帧A 的 GT mask 直接用 BOP 原数据 mask_visib，不复制）

用法：
    python prepare_mv_data.py --scene-dir real_data/test/000048 --frames 1,1024 --max-instances 5
"""
import argparse
import csv
import os
import sys

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def load_gray(p):
    return np.array(Image.open(p).convert("L")) > 0


def split_mask(mask):
    """真实 SAM2 断裂：实例中间出现一块缺失，面积损失约 1/3。

    保留上下两端、切掉中部 1/3 高度并留 10% 宽缝（缺失区域）。
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask
    y0, y1 = ys.min(), ys.max()
    h = y1 - y0
    top_h = max(2, int(h * 0.33))
    bot_h = max(2, int(h * 0.33))
    gap = max(4, int(h * 0.12))
    out = mask.copy()
    cut_lo = y0 + top_h
    cut_hi = y1 - bot_h
    out[cut_lo:cut_hi, :] = 0
    return out


def dilate(mask, radius):
    m = mask.astype(np.uint8)
    for _ in range(radius):
        p = np.pad(m, 1)
        m = np.maximum.reduce([
            p[:-2, :-2], p[:-2, 1:-1], p[:-2, 2:],
            p[1:-1, :-2], p[1:-1, 1:-1], p[1:-1, 2:],
            p[2:, :-2], p[2:, 1:-1], p[2:, 2:],
        ])
    return m > 0


def add_blob(mask, H, W):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return mask
    x1, y0 = xs.max(), ys.min()
    h = ys.max() - y0
    out = mask.copy()
    rng = np.random.RandomState(7)
    for k in range(3):
        rx0 = min(x1 + 5 + k * 12, W - 24)
        ry0 = max(1, y0 - int(h * 0.45) + k * int(h * 0.12))
        rr = 6 + k * 3
        for y in range(ry0, min(ry0 + 2 * rr, H)):
            for x in range(rx0, min(rx0 + 2 * rr, W)):
                if (x - rx0 - rr) ** 2 + (y - ry0 - rr) ** 2 < rr * rr:
                    out[y, x] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", default="data/real_data/test/000048")
    ap.add_argument("--frames", default="1,1024", help="帧对列表：帧对间用|分隔，如 '1,1024|1,1059|36,1024'")
    ap.add_argument("--max-instances", type=int, default=5)
    ap.add_argument("--out-dir", default="data/mv_masks")
    ap.add_argument("--answers", default="results/mv_answers.csv")
    args = ap.parse_args()

    scene = os.path.basename(args.scene_dir.rstrip("/\\"))
    frame_pairs = [tuple(f.strip() for f in p.split(",")) for p in args.frames.split("|")]
    mdir = os.path.join(args.scene_dir, "mask_visib")

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    counter = {}
    for fa, fb in frame_pairs:
        for idx in range(args.max_instances):
            ma_path = os.path.join(mdir, f"{int(fa):06d}_{idx:06d}.png")
            mb_path = os.path.join(mdir, f"{int(fb):06d}_{idx:06d}.png")
            if not (os.path.exists(ma_path) and os.path.exists(mb_path)):
                print(f"[跳过] 实例 {idx} 在帧 {fa}/{fb} 不同时存在")
                continue
            ma = load_gray(ma_path)
            mb = load_gray(mb_path)
            H, W = mb.shape

            # 缺陷轮换：clean / split / dirty / sticky（sticky 与下一实例并集）
            kind = ["clean", "split", "dirty", "sticky"][idx % 4]
            if kind == "split":
                mb_defect = split_mask(mb)
            elif kind == "dirty":
                mb_defect = dilate(mb, 3)
                mb_defect = add_blob(mb_defect, H, W)
            elif kind == "sticky":
                nb_path = os.path.join(mdir, f"{int(fb):06d}_{(idx + 1) % args.max_instances:06d}.png")
                if os.path.exists(nb_path):
                    mb_defect = mb | load_gray(nb_path)
                else:
                    mb_defect = mb
                    kind = "clean"
            else:
                mb_defect = mb

            Image.fromarray((mb_defect * 255).astype(np.uint8)).save(
                os.path.join(args.out_dir, f"{scene}_{fa}_{fb}_idx{idx}.png"))
            rows.append({
                "scene": scene, "frame_a": fa, "frame_b": fb, "inst_idx": idx,
                "defect": kind, "is_conflict": "是" if kind != "clean" else "否",
            })
            counter[kind] = counter.get(kind, 0) + 1
            print(f"[完成] 帧{fa}→{fb} 实例{idx}: {kind}")

    with open(args.answers, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "frame_a", "frame_b", "inst_idx", "defect", "is_conflict"])
        w.writeheader()
        w.writerows(rows)
    print(f"\n跨视图测试集完成：{len(rows)} 个实例 -> {args.out_dir}/ + {args.answers}")
    print("缺陷分布:", counter)
    print("\n接下来运行：")
    print(f"  python mv_checker.py --scene-dir {args.scene_dir} --answers {args.answers} --masks {args.out_dir}")


if __name__ == "__main__":
    main()
