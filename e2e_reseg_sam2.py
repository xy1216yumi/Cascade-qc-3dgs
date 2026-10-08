"""闭环 v3：级联判 RESEGMENT 的实例，用"缺陷类型驱动的 prompt"重跑 SAM2（真实重分割）。

补救策略（不碰 GT mask，prompt 来自跨视图传播估计）：
  - 缺失区（传播 mask 有而原 mask 没有）→ 正向点 prompt（治 split/漏边）
  - 多余区（原 mask 有而传播带之外）→ 负向点 prompt（治 sticky/杂块）
  - box prompt 沿用首次分割的框（与真实流程一致：检测框不变，精修分割）

对照三列：原 SAM2 mask / 传播并集（v2 几何补救）/ SAM2 重分割（本脚本 v3）。
评估（这里才用 GT）：IoU、重建覆盖率 cov、纯度 pur（completeness2）。

运行（SAM2 只在 pytorch-env）：
    D:\\Anaconda_envs\\envs\\pytorch-env\\python.exe e2e_reseg_sam2.py [--mode points|maskprompt|both]
    --mode points     ：缺陷驱动正负点（默认，v3 原始版）
    --mode maskprompt ：把传播估计 mask 作为 SAM2 mask_input 提示（信息密度远高于稀疏点）
    --mode both       ：mask_input + 正负点

输出：results/e2e_reseg_sam2[_mode].csv + 汇总打印
"""
import argparse
import csv, json, os, sys, time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_checker import load_gray  # noqa: E402
from prepare_mv_data import dilate  # noqa: E402
from e2e_pipeline import load_sam2  # noqa: E402
from e2e_remedy import propagate_mask, completeness2, SCENE_DIR  # noqa: E402

CKPT = r"D:\AAAApython\3Dpipelines_pytorch-env\sam2_hiera_large.pt"
CFG = "configs/sam2/sam2_hiera_l.yaml"
SAM2_JSONL = "results/sam2_real_results.jsonl"
E2E_CSV = "results/e2e_remedy.csv"
MAX_POINTS = 5


def sample_points(region, k, rng):
    ys, xs = np.where(region)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    sel = rng.choice(len(xs), size=min(k, len(xs)), replace=False)
    return np.stack([xs[sel], ys[sel]], axis=1).astype(np.float32)  # (x, y) 顺序


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["points", "maskprompt", "both"], default="points")
    args = ap.parse_args()

    reseg = [r for r in csv.DictReader(open(E2E_CSV, encoding="utf-8-sig"))
             if r["action"] == "RESEGMENT"]
    # box 与首次置信度：box 由 GT 外接框即时重算（与 run_sam2_real 的生成方式一致），
    # 首次置信度取自批量跑分记录
    score_map = {}
    for l in open("results/sam2_real_batch.jsonl", encoding="utf-8"):
        r = json.loads(l)
        score_map[(r["scene"], r["frame"], r["inst_idx"])] = r["score"]

    print(f"待重分割实例 {len(reseg)} 个。加载 SAM2 large（CPU）...", flush=True)
    t0 = time.time()
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(build_sam2(CFG, CKPT, device="cpu"))
    print(f"模型加载 {time.time()-t0:.1f}s", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    img_cache = {}
    for n, r in enumerate(reseg):
        scene, pair, idx = r["scene"], r["pair"], int(r["idx"])
        fa, fb = map(int, pair.split("_"))
        sd = os.path.join(SCENE_DIR, scene)
        cam = json.load(open(os.path.join(sd, "scene_camera.json")))
        mdir, ddir = os.path.join(sd, "mask_visib"), os.path.join(sd, "depth")

        maskA = load_sam2(scene, fa, idx)
        maskB = load_sam2(scene, fb, idx)
        if maskA is None or maskB is None:
            continue
        gtA = load_gray(os.path.join(mdir, f"{fa:06d}_{idx:06d}.png"))
        gtB = load_gray(os.path.join(mdir, f"{fb:06d}_{idx:06d}.png"))
        depthA = np.array(Image.open(os.path.join(ddir, f"{fa:06d}.png"))).astype(np.float32)
        depthB = np.array(Image.open(os.path.join(ddir, f"{fb:06d}.png"))).astype(np.float32)
        K = np.array(cam[str(fa)]["cam_K"]).reshape(3, 3)
        RA = np.array(cam[str(fa)]["cam_R_w2c"]).reshape(3, 3)
        tA = np.array(cam[str(fa)]["cam_t_w2c"]).reshape(3)
        RB = np.array(cam[str(fb)]["cam_R_w2c"]).reshape(3, 3)
        tB = np.array(cam[str(fb)]["cam_t_w2c"]).reshape(3)
        dsB = cam[str(fb)].get("depth_scale", 0.1)

        # 跨视图传播估计缺陷区域
        fix = propagate_mask(maskA, depthA, K, RA, tA, RB, tB, depthB, dsB)
        missing = fix & ~maskB                                  # 应补（正向点）
        # 负向点保守策略：传播 mask 覆盖不足一半原 mask 说明传播不可靠，放弃负向点
        cover = (fix & maskB).sum() / max(maskB.sum(), 1)
        if cover >= 0.5:
            excess = maskB & ~(dilate(fix.astype(np.uint8) * 255, 5) > 0)
        else:
            excess = np.zeros_like(maskB)
        pos = sample_points(missing, MAX_POINTS, rng)
        neg = sample_points(excess, MAX_POINTS, rng)
        pts = np.vstack([pos, neg]) if len(pos) and len(neg) else (pos if len(pos) else neg)
        lbl = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))]).astype(np.int32)

        # box 沿用首次分割的生成方式：GT 外接框（模拟检测框，与 run_sam2_real 一致）
        ys_b, xs_b = np.where(gtB)
        box = np.array([xs_b.min(), ys_b.min(), xs_b.max() + 1, ys_b.max() + 1], dtype=np.float32)

        # mask_input：传播估计 mask 转 256x256 logits 提示（maskprompt / both 模式）
        mask_input = None
        if args.mode in ("maskprompt", "both"):
            mi = np.array(Image.fromarray(fix.astype(np.uint8) * 255).resize((256, 256)),
                          dtype=np.float32)
            mask_input = ((mi > 127).astype(np.float32) * 20 - 10)[None]  # ±10 logits

        if fb not in img_cache:
            rgb = np.array(Image.open(os.path.join(sd, "rgb", f"{fb:06d}.png")).convert("RGB"))
            predictor.set_image(rgb)
            img_cache[fb] = True
        t1 = time.time()
        masks, scores, _ = predictor.predict(
            box=box[None, :],
            point_coords=pts[None] if len(pts) and args.mode != "maskprompt" else None,
            point_labels=lbl[None] if len(pts) and args.mode != "maskprompt" else None,
            mask_input=mask_input,
            multimask_output=False)
        newB = masks[0] > 0
        score_after = float(scores[0])
        score_before = score_map.get((scene, fb, idx), 0.0)
        # 生产可用的选择规则：置信度不降才采纳新 mask（不用 GT）
        selB = newB if score_after >= score_before else maskB

        iou_before = float((maskB & gtB).sum() / max((maskB | gtB).sum(), 1))
        iou_after = float((newB & gtB).sum() / max((newB | gtB).sum(), 1))
        iou_sel = float((selB & gtB).sum() / max((selB | gtB).sum(), 1))
        cov_b, pur_b = completeness2(maskA, maskB, depthA, depthB, K, RA, tA, RB, tB, gtA, gtB)
        cov_p, pur_p = completeness2(maskA, maskB | fix, depthA, depthB, K, RA, tA, RB, tB, gtA, gtB)
        cov_a, pur_a = completeness2(maskA, newB, depthA, depthB, K, RA, tA, RB, tB, gtA, gtB)
        cov_s, pur_s = completeness2(maskA, selB, depthA, depthB, K, RA, tA, RB, tB, gtA, gtB)

        rows.append({
            "scene": scene, "pair": pair, "idx": idx,
            "n_pos": len(pos), "n_neg": len(neg), "fix_cover": round(float(cover), 3),
            "score_before": round(score_before, 4), "score_after": round(score_after, 4),
            "adopted": bool(score_after >= score_before),
            "iou_before": round(iou_before, 3), "iou_after": round(iou_after, 3),
            "iou_sel": round(iou_sel, 3),
            "cov_before": round(cov_b, 3), "cov_prop": round(cov_p, 3),
            "cov_reseg": round(cov_a, 3), "cov_sel": round(cov_s, 3),
            "pur_before": round(pur_b, 3), "pur_prop": round(pur_p, 3),
            "pur_reseg": round(pur_a, 3), "pur_sel": round(pur_s, 3),
        })
        print(f"[{n+1}/{len(reseg)}] {scene} {pair} #{idx}: IoU {iou_before:.3f}→{iou_after:.3f} "
              f"| cov {cov_b:.3f}→prop {cov_p:.3f}→reseg {cov_a:.3f} "
              f"| pur {pur_b:.3f}→prop {pur_p:.3f}→reseg {pur_a:.3f} ({time.time()-t1:.1f}s)",
              flush=True)

    print("\n" + "=" * 66)
    print("闭环 v3 汇总（RESEGMENT 组，%d 实例，mode=%s）" % (len(rows), args.mode))
    print("=" * 66)
    g = lambda k: np.mean([r[k] for r in rows])
    n_adopt = sum(r["adopted"] for r in rows)
    print(f"IoU    : {g('iou_before'):.3f} → 重分割 {g('iou_after'):.3f} → 置信度采纳 {g('iou_sel'):.3f}"
          f"（采纳 {n_adopt}/{len(rows)}）")
    print(f"覆盖率 : {g('cov_before'):.3f} → 传播 {g('cov_prop'):.3f} → 重分割 {g('cov_reseg'):.3f} → 采纳 {g('cov_sel'):.3f}")
    print(f"纯度   : {g('pur_before'):.3f} → 传播 {g('pur_prop'):.3f} → 重分割 {g('pur_reseg'):.3f} → 采纳 {g('pur_sel'):.3f}")

    os.makedirs("results", exist_ok=True)
    suffix = "" if args.mode == "points" else f"_{args.mode}"
    out_csv = f"results/e2e_reseg_sam2{suffix}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n明细已存 {out_csv}")


if __name__ == "__main__":
    main()
