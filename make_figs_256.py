"""Fig.2（缺陷分类四宫格）+ Fig.5（PSNR 闭环柱状图）+ Fig.6（检测 ROC 曲线）。

用法：python make_figs_256.py
输出：docs/figures/fig2_taxonomy.png / fig5_psnr_bar.png / fig6_roc.png
"""
import csv, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("docs/figures", exist_ok=True)


def fig2():
    answers = list(csv.DictReader(open("results/test_answers.csv", encoding="utf-8-sig")))
    picked = {}
    for r in answers:
        img = os.path.basename(r["image_id"])
        iid = int(r["instance_id"])
        if r["is_clean"] == "是" and "clean" not in picked:
            picked["clean"] = (img, iid)
        elif r["is_split"] == "是" and "split" not in picked:
            picked["split"] = (img, iid)
        elif r["is_sticky"] == "是" and "sticky" not in picked:
            picked["sticky"] = (img, iid)
        elif r["note"] == "dirty" and "dirty" not in picked:
            picked["dirty"] = (img, iid)
    print("选中实例:", picked)

    try:
        font = ImageFont.truetype("arialbd.ttf", 28)
    except Exception:
        font = ImageFont.load_default()
    cells = []
    for k in ["clean", "dirty", "split", "sticky"]:
        img, iid = picked[k]
        ov = Image.open(f"data/test_images/{img}").convert("RGB")
        m = np.array(Image.open(f"data/test_masks/{img[:-4]}_instance{iid}.png").convert("L")) > 127
        ys, xs = np.where(m)
        pad = 25
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, ov.size[1])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, ov.size[0])
        c = ov.crop((x0, y0, x1, y1)).resize((320, 240))
        cells.append((k, c))

    canvas = Image.new("RGB", (660, 540), "white")
    d = ImageDraw.Draw(canvas)
    for i, (k, c) in enumerate(cells):
        x, y = (i % 2) * 330 + 5, (i // 2) * 270 + 5
        canvas.paste(c, (x, y))
        d.text((x + 160, y + 245), k, font=font, fill="black", anchor="mm")
    canvas.save("docs/figures/fig2_taxonomy.png")
    print("fig2 已保存")


def fig5():
    rows = []
    for line in open("results/psnr_obj_loop.csv", encoding="utf-8-sig"):
        p = line.strip().split(",")
        if len(p) == 4:
            rows.append({"scene": p[0], "mode": p[1], "variant": p[2], "psnr": p[3]})
    modes = {"partial": ("random split\n(40% views)", {"gt": [], "defect": [], "fixed": []}),
             "split100": ("systematic split\n(all views)", {"gt": [], "defect": [], "fixed": []}),
             "sticky100": ("systematic sticky\n(all views)", {"gt": [], "defect": [], "fixed": []})}
    for r in rows:
        if r["mode"] in modes and r["variant"] == "gt":
            modes[r["mode"]][1]["gt"].append(float(r["psnr"]))
    for r in rows:
        if r["mode"] in modes:
            modes[r["mode"]][1][r["variant"]].append(float(r["psnr"]))

    labels, gt_v, def_v, fix_v = [], [], [], []
    for m in ["partial", "split100", "sticky100"]:
        lab, d = modes[m]
        labels.append(lab)
        gt_v.append(np.mean(d["gt"]))
        def_v.append(np.mean(d["defect"]))
        fix_v.append(np.mean(d["fixed"]) if d["fixed"] else np.nan)

    x = np.arange(3); w = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=600)
    ax.bar(x - w, gt_v, w, label="GT masks (upper bound)", color="#4C9F70")
    ax.bar(x, def_v, w, label="defective masks", color="#D05050")
    ax.bar(x + w, [0 if np.isnan(v) else v for v in fix_v], w,
           label="repaired (propagation)", color="#5B8DD9")
    for i, v in enumerate(fix_v):
        if np.isnan(v) or v == 0:
            ax.text(x[i] + w, def_v[i] * 0.5, "no clean\nsource", ha="center",
                    fontsize=8, color="#555", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("object-region PSNR (dB)")
    ax.set_ylim(0, 22)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Defect mode vs. object-level reconstruction quality (3 scenes)")
    fig.tight_layout()
    fig.savefig("docs/figures/fig5_psnr_bar.png")
    fig.savefig("docs/figures/fig5_psnr_bar.tiff")
    print("fig5 已保存")


def fig6():
    jt = list(csv.DictReader(open("results/jitter_ensemble_ycbv.csv", encoding="utf-8-sig")))
    mf = {(r["scene"], int(r["fb"]), int(r["idx"])): r
          for r in csv.DictReader(open("results/mined_multiframe_ycbv.csv", encoding="utf-8-sig"))}
    y, s_conf, s_jit = [], [], []
    for r in jt:
        key = (r["scene"], int(r["fb"]), int(r["idx"]))
        if key not in mf:
            continue
        y.append((r["label"] != "clean") and (float(r["iou"]) < 0.8))
        s_conf.append(-float(mf[key]["conf"]))
        s_jit.append(-float(r["stability"]))
    y = np.array(y); s_conf = np.array(s_conf); s_jit = np.array(s_jit)

    def roc(s):
        order = np.argsort(-s)  # 分数越高越倾向"有缺陷"，从严格到宽松扫描
        ys = y[order]
        tps = np.cumsum(ys) / max(ys.sum(), 1)
        fps = np.cumsum(~ys) / max((~ys).sum(), 1)
        return np.concatenate([[0], fps]), np.concatenate([[0], tps])

    fig, ax = plt.subplots(figsize=(5, 4.6), dpi=600)
    for name, s, c in [("SAM2 self-confidence", s_conf, "#5B8DD9"),
                       ("prompt-jitter disagreement", s_jit, "#D05050")]:
        fpr, tpr = roc(s)
        ax.plot(fpr, tpr, label=name, color=c, lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR (severe-defect recall)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_title("Natural defect detection (held-out, YCB-V)")
    fig.tight_layout()
    fig.savefig("docs/figures/fig6_roc.png")
    fig.savefig("docs/figures/fig6_roc.tiff")
    print("fig6 已保存")


fig2(); fig5(); fig6()
