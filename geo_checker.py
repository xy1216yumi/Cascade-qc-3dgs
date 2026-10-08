"""确定性几何检查器：基于"真 mask + 无 mask 渲染图"用纯像素规则做三属性判断。

背景：纯 VLM 检查器在合成图上 sticky（一个 mask 盖住多个物体）检出率为 0/10——
模型没有图层感知，无法透过半透明 mask 数物体。本工具用 OpenCV 风格的确定性规则
补上这块，输出与 VLM 检查器完全相同的 schema（image / instances[]），
可直接用 evaluate.py 对比，验证"VLM + 规则混合闭环"的必要性。

规则（按优先级）：
1. sticky：mask 覆盖区域内，物体像素按颜色相似聚类的连通域数 >= 2
   （一个 mask 盖住了多个物体。物体图无 mask 叠加，杂块不会混入）
2. split：mask 的连通域中 >=2 个大块（各 >=25% 面积）
3. dirty：mask 有游离小碎块（<25%），或 mask 面积明显大于物体（>1.18 倍）
4. clean：其余

用法：
    python geo_checker.py --input test_images --masks test_masks \
        --objects test_objects --output geo_results.jsonl
    python evaluate.py --answers test_answers.csv --predictions geo_results.jsonl
"""
import argparse
import json
import os
import sys
from collections import deque

import numpy as np
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BG = np.array([245, 243, 238], dtype=int)   # 合成图背景色
COLOR_EPS = 45                               # 颜色聚类阈值（曼哈顿距离）
BG_EPS = 25                                  # 判定"是背景"的颜色距离阈值


def load_gray(p):
    return np.array(Image.open(p).convert("L"))


def load_rgb(p):
    return np.array(Image.open(p).convert("RGB"))


def connected_components(mask_bin):
    """8 邻域连通域分析，返回 (连通域数, 各面积列表)。"""
    h, w = mask_bin.shape
    label = np.zeros((h, w), dtype=np.int32)
    areas = []
    n = 0
    for y in range(h):
        for x in range(w):
            if mask_bin[y, x] and label[y, x] == 0:
                n += 1
                q = deque([(y, x)])
                label[y, x] = n
                area = 0
                while q:
                    cy, cx = q.popleft()
                    area += 1
                    for dy in (-1, 0, 1):
                        for dx in (-1, 0, 1):
                            ny, nx = cy + dy, cx + dx
                            if (0 <= ny < h and 0 <= nx < w
                                    and mask_bin[ny, nx] and label[ny, nx] == 0):
                                label[ny, nx] = n
                                q.append((ny, nx))
                areas.append(area)
    return n, areas


def count_object_regions(mask_bin, obj_img):
    """mask 区域内，物体像素（非背景色）按颜色相似（距离<COLOR_EPS）聚类的连通域数。

    输入 obj_img 必须是无 mask 的物体图：mask 层（含 dirty 杂块）不会混入计数。
    """
    h, w = mask_bin.shape
    visited = np.zeros((h, w), dtype=bool)
    regions = 0
    for y in range(h):
        for x in range(w):
            if (not mask_bin[y, x]) or visited[y, x]:
                continue
            if np.abs(obj_img[y, x].astype(int) - BG).sum() < BG_EPS:
                continue  # 背景像素不计数
            regions += 1
            q = deque([(y, x)])
            visited[y, x] = True
            c0 = obj_img[y, x].astype(int)
            while q:
                cy, cx = q.popleft()
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = cy + dy, cx + dx
                        if (0 <= ny < h and 0 <= nx < w and mask_bin[ny, nx]
                                and not visited[ny, nx]
                                and np.abs(obj_img[ny, nx].astype(int) - BG).sum() >= BG_EPS
                                and np.abs(obj_img[ny, nx].astype(int) - c0).sum() < COLOR_EPS):
                            visited[ny, nx] = True
                            q.append((ny, nx))
    return regions


def check_instance(mask, obj_img=None, mode="synthetic"):
    """返回与 VLM 检查器相同的单实例判断。

    优先级：split（mask 层面分块）> sticky（单块 mask 内多个物体）> dirty > clean。

    mode:
      synthetic —— 物体图为纯色背景 + 纯色物体，可用背景色阈值 + 颜色聚类数物体
                  （sticky 可判、面积比可判）
      real     —— 物体图为真实照片，无平坦背景、纹理复杂：
                  * split 仍可用（mask 二值连通域，与背景无关）
                  * dirty 只用小碎块判据（面积比/颜色聚类不可用）
                  * sticky 单帧不可判（需物体先验/多视图几何信号），降级为 clean
    """
    mask_area = int((mask > 0).sum())
    if mask_area == 0:
        return {"is_clean": False, "is_split": False, "is_sticky": False,
                "confidence": 1.0, "reason": "空mask"}

    n_cc, areas = connected_components(mask > 0)
    big = [a for a in areas if a >= 0.25 * mask_area]
    small_cc = any(0 < a < 0.25 * mask_area for a in areas)

    # 1) split：mask 主体在 mask 层面被切成 >=2 个大块（两种模式都可靠）
    if len(big) >= 2:
        return {"is_clean": False, "is_split": True, "is_sticky": False,
                "confidence": 1.0, "reason": f"mask分为{len(big)}块"}

    # 2) sticky：仅合成模式能用颜色聚类数物体；真实照片单帧无法规则化数物体
    if mode == "synthetic" and obj_img is not None:
        n_obj = count_object_regions(mask > 0, obj_img)
        if n_obj >= 2:
            return {"is_clean": False, "is_split": False, "is_sticky": True,
                    "confidence": 1.0, "reason": f"mask覆盖{n_obj}个物体"}

    # 3) dirty：小碎块（两种模式都可靠）；面积比仅合成模式可用
    ratio = 1.0
    if mode == "synthetic" and obj_img is not None:
        obj_pixels = int(((np.abs(obj_img.astype(int) - BG).sum(axis=2) >= BG_EPS)
                          & (mask > 0)).sum())
        ratio = mask_area / max(obj_pixels, 1)
    if small_cc or (mode == "synthetic" and ratio > 1.18):
        reason = "小碎块" if small_cc else f"面积比{ratio:.2f}"
        return {"is_clean": False, "is_split": False, "is_sticky": False,
                "confidence": 1.0, "reason": reason}

    # 4) clean
    return {"is_clean": True, "is_split": False, "is_sticky": False,
            "confidence": 1.0, "reason": "规则判定干净"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/test_images", help="叠加图目录（仅用于定位图名）")
    ap.add_argument("--masks", default="data/test_masks", help="每实例真 mask 目录")
    ap.add_argument("--objects", default="data/test_objects", help="无 mask 物体图目录")
    ap.add_argument("--output", default="geo_results.jsonl")
    ap.add_argument("--mode", choices=["synthetic", "real"], default="synthetic",
                    help="synthetic=纯色背景合成图；real=真实照片（sticky 单帧不可判）")
    args = ap.parse_args()

    results = []
    for fname in sorted(os.listdir(args.input)):
        if not fname.endswith(".png"):
            continue
        stem = fname[:-4]
        obj_img = load_rgb(os.path.join(args.objects, fname))
        instances = []
        for mname in sorted(os.listdir(args.masks)):
            if mname.startswith(stem + "_instance") and mname.endswith(".png"):
                iid = int(mname[len(stem) + len("_instance"):-4])
                mask = load_gray(os.path.join(args.masks, mname))
                inst = check_instance(mask, obj_img, mode=args.mode)
                inst["instance_id"] = iid
                instances.append(inst)
        instances.sort(key=lambda x: x["instance_id"])
        results.append({"image": os.path.join(args.input, fname), "instances": instances})

    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    total = sum(len(r["instances"]) for r in results)
    print(f"几何检查完成：{len(results)} 张图、{total} 个实例 -> {args.output}")
    print("接下来运行：")
    print(f"  python evaluate.py --answers test_answers.csv --predictions {args.output}")


if __name__ == "__main__":
    main()
