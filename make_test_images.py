"""合成标准测试图 + 自动 ground truth（无人工标注也能验证 VLM 检查器能力）。

为什么要这个工具：直接拿真实 3D 渲染图喂给 VLM 时，如果图上没有
"mask 叠加层 + 实例编号"，VLM 只能猜，准确率没有意义（之前的实验
就踩了这个坑——模型每条都报告"未见叠加掩码"）。本工具合成
"物体 + mask 叠加 + 编号"的标准图，注入已知缺陷，并输出已知答案的
ground truth CSV，三步就能跑出 VLM 检查器的真实准确率。

缺陷类型（与 schema.py 三个判断一一对应）：
  clean  : mask 完全贴合物体          -> is_clean=是, split=否, sticky=否
  dirty  : mask 比物体大一圈/带杂块   -> is_clean=否, split=否, sticky=否
  split  : 同一编号的 mask 切成两块   -> is_clean=否, split=是, sticky=否
  sticky : 一个 mask 盖住两个物体     -> is_clean=否, split=否, sticky=是

用法（在项目根目录，即本文件所在目录）：
    python make_test_images.py --num 10 --seed 42
    python batch_check.py --input test_images --output test_results.jsonl
    python evaluate.py --answers test_answers.csv --predictions test_results.jsonl
"""
import argparse
import csv
import os
import random
import sys

from PIL import Image, ImageDraw, ImageFont

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

W, H = 1024, 1024
BG = (245, 243, 238, 255)

# 物体：深色不透明（真实轮廓）；mask：高饱和半透明（叠加层），两者必须可区分
OBJ_COLORS = [(92, 92, 92, 255), (107, 79, 58, 255), (63, 93, 122, 255),
              (74, 107, 74, 255), (122, 74, 74, 255)]
MASK_COLORS = [(255, 107, 129), (255, 169, 77), (77, 171, 247),
               (252, 196, 25), (81, 207, 102), (177, 151, 252)]
ALPHA = 135  # mask 半透明度

# 6 个网格锚点，每格放一个实例（sticky 在一格内放两个小物体）
ANCHORS = [(256, 256), (768, 256), (256, 512), (768, 512), (256, 768), (768, 768)]
CELL = 360  # 锚点周围可用空间半径

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


def draw_shape(draw, kind: str, bbox, fill, outline=None, width=1):
    """在 bbox 内画椭圆或圆角矩形。kind: 'ellipse' | 'rect'"""
    if kind == "ellipse":
        draw.ellipse(bbox, fill=fill, outline=outline, width=width)
    else:
        draw.rounded_rectangle(bbox, radius=18, fill=fill, outline=outline, width=width)


def draw_object(draw, kind: str, bbox, color):
    """不透明深色实心物体（真实轮廓）。"""
    draw_shape(draw, kind, bbox, color)


def draw_mask(draw, kind: str, bbox, color, num_id, font, mode: str):
    """画半透明 mask + 边框 + 中心编号。

    mode: 'clean'/'dirty'/'sticky' 画整块；'split' 沿水平中线切成两块留缺口。
    """
    c = color + (ALPHA,)
    outline = color + (255,)
    if mode == "split":
        x0, y0, x1, y1 = bbox
        mid = (y0 + y1) // 2
        gap = max(6, (y1 - y0) // 8)
        draw_shape(draw, kind, (x0, y0, x1, mid - gap // 2), c, outline, 3)
        draw_shape(draw, kind, (x0, mid + gap // 2, x1, y1), c, outline, 3)
        cx, cy = (x0 + x1) // 2, mid
    else:
        draw_shape(draw, kind, bbox, c, outline, 3)
        cx, cy = (bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2
    draw.text((cx, cy), str(num_id), font=font, fill=(255, 255, 255, 255),
              anchor="mm", stroke_width=3, stroke_fill=(20, 20, 20, 255))


def make_bbox(anchor, w, h):
    ax, ay = anchor
    return (int(ax - w / 2), int(ay - h / 2), int(ax + w / 2), int(ay + h / 2))


def make_instance(iid, kind, anchor, rng, mask_color):
    shape = rng.choice(["ellipse", "rect"])
    w = int(CELL * rng.uniform(0.35, 0.55))
    h = int(CELL * rng.uniform(0.3, 0.48))
    bbox = make_bbox(anchor, w, h)
    return {
        "instance_id": iid,
        "kind": kind,
        "shape": shape,
        "bbox": bbox,
        "obj_color": rng.choice(OBJ_COLORS),
        "mask_color": mask_color,
    }


def render_image(specs, rng, objects_only: bool = False) -> Image.Image:
    """按实例规格画一张测试图。objects_only=True 时只画物体（无 mask 叠加）。"""
    img = Image.new("RGBA", (W, H), BG)
    draw = ImageDraw.Draw(img)
    font = get_font(52)

    for sp in specs:
        x0, y0, x1, y1 = sp["bbox"]
        kind = sp["kind"]
        if kind == "sticky":
            # 两个"颜色差异大、形状不同、中间留可见缝隙"的物体左右并排，
            # 一个连通 mask 盖住两者。异色+缝隙是"一个mask覆盖多个物体"的关键视觉线索。
            # 第一个物体用深色，第二个用亮色（如白色盘子、亮色杯子），对比强才数得出来。
            mid_x = (x0 + x1) // 2
            gap = max(10, int((x1 - x0) * 0.15))
            b1 = (x0, y0, mid_x - gap // 2, y1)
            b2 = (mid_x + gap // 2, y0, x1, y1)
            shape2 = "rect" if sp["shape"] == "ellipse" else "ellipse"
            c2 = sp.get("c2", sp["obj_color"])
            draw_object(draw, sp["shape"], b1, sp["obj_color"])
            draw_object(draw, shape2, b2, c2)
            if not objects_only:
                # mask 用矩形：完整盖住两个物体，避免椭圆 mask 露出矩形角造成"偏移"歧义
                mb = (x0 - 12, y0 - 12, x1 + 12, y1 + 12)
                draw_mask(draw, "rect", mb, sp["mask_color"], sp["instance_id"], font, "sticky")
        elif kind == "split":
            draw_object(draw, sp["shape"], sp["bbox"], sp["obj_color"])
            if not objects_only:
                draw_mask(draw, sp["shape"], sp["bbox"], sp["mask_color"], sp["instance_id"], font, "split")
        elif kind == "dirty":
            draw_object(draw, sp["shape"], sp["bbox"], sp["obj_color"])
            if not objects_only:
                dw, dh = x1 - x0, y1 - y0
                mb = (int(x0 - dw * 0.15), int(y0 - dh * 0.15), int(x1 + dw * 0.15), int(y1 + dh * 0.15))
                draw_mask(draw, sp["shape"], mb, sp["mask_color"], sp["instance_id"], font, "dirty")
                # 额外杂块（游离色块）
                blob = (int(x1 + dw * 0.02), int(y0 - dh * 0.38), int(x1 + dw * 0.24), int(y0 - dh * 0.06))
                draw.ellipse(blob, fill=sp["mask_color"] + (ALPHA,),
                             outline=sp["mask_color"] + (255,), width=3)
        else:  # clean
            draw_object(draw, sp["shape"], sp["bbox"], sp["obj_color"])
            if not objects_only:
                draw_mask(draw, sp["shape"], sp["bbox"], sp["mask_color"], sp["instance_id"], font, "clean")
    return img.convert("RGB")


def render_mask_only(sp) -> Image.Image:
    """输出该实例的"真 mask"二值图（白=实例 mask 区域，黑=背景）。

    模拟 SAM2 的输出：含缺陷（split 两块 / sticky 一块盖两物体 / dirty 大一圈+杂块）。
    与图上叠加的 mask 完全一致，供几何检查器（geo_checker.py）使用。
    """
    img = Image.new("RGB", (W, H), (0, 0, 0))
    draw = ImageDraw.Draw(img)
    white = (255, 255, 255)
    x0, y0, x1, y1 = sp["bbox"]
    kind = sp["kind"]
    if kind == "sticky":
        mb = (x0 - 12, y0 - 12, x1 + 12, y1 + 12)
        draw_shape(draw, "rect", mb, white)
    elif kind == "split":
        mid = (y0 + y1) // 2
        gap = max(6, (y1 - y0) // 8)
        draw_shape(draw, sp["shape"], (x0, y0, x1, mid - gap // 2), white)
        draw_shape(draw, sp["shape"], (x0, mid + gap // 2, x1, y1), white)
    elif kind == "dirty":
        dw, dh = x1 - x0, y1 - y0
        mb = (int(x0 - dw * 0.15), int(y0 - dh * 0.15), int(x1 + dw * 0.15), int(y1 + dh * 0.15))
        draw_shape(draw, sp["shape"], mb, white)
        blob = (int(x1 + dw * 0.02), int(y0 - dh * 0.38), int(x1 + dw * 0.24), int(y0 - dh * 0.06))
        draw.ellipse(blob, fill=white)
    else:  # clean
        draw_shape(draw, sp["shape"], sp["bbox"], white)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=10, help="生成几张测试图")
    ap.add_argument("--out-dir", default="data/test_images")
    ap.add_argument("--objects-dir", default="data/test_objects")
    ap.add_argument("--masks-dir", default="data/test_masks")
    ap.add_argument("--answers", default="test_answers.csv")
    ap.add_argument("--seed", type=int, default=42, help="随机种子，保证可复现")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.objects_dir, exist_ok=True)
    os.makedirs(args.masks_dir, exist_ok=True)
    rng = random.Random(args.seed)
    rows = []
    counter = {"clean": 0, "dirty": 0, "split": 0, "sticky": 0}

    for idx in range(1, args.num + 1):
        n_inst = rng.randint(3, 5)
        anchors = rng.sample(ANCHORS, n_inst)
        # 每张图保证至少 1 clean + 1 split + 1 sticky，其余随机补 dirty/clean
        kinds = ["clean", "split", "sticky"]
        kinds += rng.choices(["dirty", "clean"], k=max(0, n_inst - 3))
        rng.shuffle(kinds)

        specs = []
        # 同一张图内 mask 颜色不重复，避免 VLM 把同色误当同一实例
        colors = rng.sample(MASK_COLORS, n_inst)
        bright = [(205, 178, 112), (176, 96, 58), (98, 148, 192), (222, 222, 206)]
        for iid, (kind, anchor) in enumerate(zip(kinds, anchors), start=1):
            sp = make_instance(iid, kind, anchor, rng, colors[iid - 1])
            if kind == "sticky":
                # sticky 的第二个物体颜色预先生成（渲染与真 mask 共用，保证一致）
                sp["c2"] = rng.choice([c for c in OBJ_COLORS if c != sp["obj_color"]] + bright)
            specs.append(sp)

        fname = f"test_{idx:02d}.png"
        img = render_image(specs, rng)
        img.save(os.path.join(args.out_dir, fname))
        # 配套输出：无 mask 物体图（几何检查器用）+ 每实例真 mask（模拟 SAM2 输出）
        render_image(specs, rng, objects_only=True).save(os.path.join(args.objects_dir, fname))
        for sp in specs:
            render_mask_only(sp).save(
                os.path.join(args.masks_dir, f"test_{idx:02d}_instance{sp['instance_id']}.png"))

        for sp in specs:
            kind = sp["kind"]
            counter[kind] += 1
            rows.append({
                "image_id": os.path.join(args.out_dir, fname),
                "instance_id": sp["instance_id"],
                "is_clean": "是" if kind == "clean" else "否",
                "is_split": "是" if kind == "split" else "否",
                "is_sticky": "是" if kind == "sticky" else "否",
                "note": kind,
            })

    with open(args.answers, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["image_id", "instance_id", "is_clean", "is_split", "is_sticky", "note"])
        w.writeheader()
        w.writerows(rows)

    total = sum(counter.values())
    print(f"已生成 {args.num} 张测试图 -> {args.out_dir}/")
    print(f"ground truth -> {args.answers}（共 {total} 个实例）")
    print("实例分布:", {k: counter[k] for k in counter})
    print("\n接下来运行：")
    print(f"  python batch_check.py --input {args.out_dir} --output test_results.jsonl")
    print(f"  python evaluate.py --answers {args.answers} --predictions test_results.jsonl")


if __name__ == "__main__":
    main()
