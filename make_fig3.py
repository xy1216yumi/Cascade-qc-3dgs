"""Fig.3 级联调度架构图（PIL 绘制，投稿前可用 PPT/AI 重绘美化）。

用法：python make_fig3.py [--out docs/figures/fig3_architecture.png]
"""
import argparse, os
from PIL import Image, ImageDraw, ImageFont

W, H = 1560, 760
C_BOX = (232, 242, 255)
C_GEO = (232, 255, 238)
C_VLM = (255, 243, 224)
C_ACT = (255, 232, 232)
C_EDGE = (90, 90, 90)


def font(sz):
    try:
        return ImageFont.truetype("arialbd.ttf", sz)
    except Exception:
        return ImageFont.load_default()


def box(d, x, y, w, h, fill, title, lines, tfont, lfont):
    d.rounded_rectangle([x, y, x + w, y + h], 14, fill=fill, outline=C_EDGE, width=2)
    d.text((x + w / 2, y + 26), title, font=tfont, fill="black", anchor="mm")
    for i, ln in enumerate(lines):
        d.text((x + w / 2, y + 62 + i * 26), ln, font=lfont, fill=(40, 40, 40), anchor="mm")


def arrow(d, x1, y1, x2, y2, label="", lfont=None, above=True):
    d.line([x1, y1, x2, y2], fill=C_EDGE, width=3)
    # 箭头头部
    import math
    ang = math.atan2(y2 - y1, x2 - x1)
    for da in (2.6, -2.6):
        d.line([x2, y2, x2 + 14 * math.cos(ang + da), y2 + 14 * math.sin(ang + da)], fill=C_EDGE, width=3)
    if label and lfont:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + (-22 if above else 22)
        d.text((mx, my), label, font=lfont, fill=(160, 40, 40), anchor="mm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figures/fig3_architecture.png")
    args = ap.parse_args()

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    tf, lf, sf = font(30), font(22), font(20)

    d.text((W / 2, 40), "Cascaded Quality-Checking Scheduler", font=font(36), fill="black", anchor="mm")

    # 输入
    box(d, 30, 150, 240, 200, C_BOX, "Input", ["multi-view RGB-D", "+ instance masks", "+ depth & poses", "(any segmenter)"], tf, lf)

    # Stage1
    box(d, 350, 150, 300, 200, C_GEO, "Stage 1", ["single-frame rules", "components / area / dust", "split & dirty intercepted"], tf, lf)
    d.text((500, 380), "64.4% resolved here", font=sf, fill=(30, 130, 60), anchor="mm")

    # Stage2
    box(d, 730, 150, 320, 200, C_GEO, "Stage 2", ["cross-view arbitration", "back-project → transform", "→ re-project → hit-rate", "sticky vs split resolved"], tf, lf)
    d.text((890, 380), "35.6% escalated", font=sf, fill=(160, 40, 40), anchor="mm")

    # VLM 语义层
    box(d, 1130, 150, 300, 200, C_VLM, "VLM layer", ["two-view arbitration", "defect type + explanation", "AND w/ geo: 93.0% prec."], tf, lf)

    # 动作
    box(d, 730, 480, 320, 180, C_ACT, "Action", ["PASS", "RESEGMENT", "REGENERATE"], tf, lf)

    # 修复
    box(d, 1130, 480, 380, 180, C_ACT, "Safe repair", ["cross-view propagation", "SAM2 re-prompt (dense mask)", "fuse: adopt iff conf. not drop"], tf, lf)

    # 输出
    box(d, 350, 480, 300, 180, C_BOX, "3D reconstruction", ["3DGS training", "verified masks only"], tf, lf)

    arrow(d, 270, 250, 350, 250, "", sf)
    arrow(d, 650, 250, 730, 250, "ambiguous", sf)
    arrow(d, 1050, 250, 1130, 250, "conflict", sf)
    arrow(d, 1280, 350, 1280, 480, "", sf)
    arrow(d, 1130, 570, 1050, 570, "", sf)          # Safe repair → Action
    arrow(d, 730, 570, 650, 570, "verified masks", sf)  # Action → 3D recon
    arrow(d, 890, 350, 890, 480, "", sf)

    d.text((W / 2, 720), "Injected defects: rules-first cascade  |  Natural defects: high-recall screen (conf / jitter) first",
           font=sf, fill=(80, 80, 80), anchor="mm")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    img.save(args.out)
    print(f"已保存 {args.out}")


if __name__ == "__main__":
    main()
