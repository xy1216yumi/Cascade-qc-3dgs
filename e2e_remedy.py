"""端到端闭环 v2：补救从"GT 上限"换成真实几何重分割（跨视图传播修复）。

与 e2e_pipeline.py 的差异：
  - 补救手段 = 跨视图传播：把另一帧的 mask 经 深度反投影→位姿变换→重投影
     splat 成本帧 mask（z-buffer 取最近 + 深度一致性过滤 + 膨胀补洞），
    对 split 补回缺失块、对 sticky 切掉粘连邻居——全程不用 GT mask；
  - 仍同时记录 GT 上限 cov_gt 作参照；
  - 增益 = cov_fixed - cov_sam（实测），不再是 cov_gt - cov_sam（上限）。

用法：python e2e_remedy.py [--remedy auto|union|banded]
  auto   = 并集/带状两候选，级联自查选优（默认）
  union  = 仅并集（补 split，覆盖率优先）
  banded = 仅带状交集（切 sticky，纯度优先）
"""
import argparse
import csv, json, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_checker import load_gray, unproject  # noqa: E402
from cascade_checker import stage1_single_frame, stage2_cross_view  # noqa: E402
from prepare_mv_data import dilate  # noqa: E402
from e2e_pipeline import load_sam2  # noqa: E402

SCENE_DIR = "data/real_data/test"
JSONL = "results/sam2_real_batch.jsonl"
THRESHOLD = 0.70
VIS_TOL = 30.0  # mm，与 mv_checker 一致


def propagate_mask(mask_src, depth_src, K, R_s, t_s, R_d, t_d, depth_dst,
                   depth_scale_dst=0.1, splat_dilate=1):
    """把 src 帧 mask 传播到 dst 帧：反投影→变换→splat（z-buffer + 深度一致性 + 膨胀补洞）。

    depth_dst 为 BOP 原始深度值，需乘 depth_scale_dst 换算成 mm 后再做一致性比较。
    """
    H, W = mask_src.shape
    depth_dst = depth_dst * depth_scale_dst
    Pc = unproject(mask_src, depth_src, K, 0.1)       # Nx3 相机系(src)
    if len(Pc) == 0:
        return np.zeros_like(mask_src)
    Pw = (R_s.T @ (Pc - t_s).T).T                      # 世界系
    Pd = (R_d @ Pw.T).T + t_d                          # 相机系(dst)
    z = Pd[:, 2]
    ok = z > 1.0
    Pd, z = Pd[ok], z[ok]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u = np.round(fx * Pd[:, 0] / z + cx).astype(np.int32)
    v = np.round(fy * Pd[:, 1] / z + cy).astype(np.int32)
    inb = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[inb], v[inb], z[inb]

    # z-buffer：同一像素只留最近点
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    np.minimum.at(zbuf, (v, u), z)
    hit = np.zeros((H, W), dtype=bool)
    keep = z <= zbuf[v, u] + 1e-6
    hit[v[keep], u[keep]] = True

    # 深度一致性：投影点与 dst 实际深度差太大 → 被遮挡，剔除
    dz = np.abs(zbuf - depth_dst)
    hit &= np.isfinite(zbuf) & (dz < VIS_TOL * 3)

    # 膨胀补 splat 空洞（点云离散采样导致）；越大越易把背景溢出并进并集
    if splat_dilate > 0:
        return dilate(hit.astype(np.uint8) * 255, splat_dilate) > 0
    return hit


def completeness2(maskA, maskB, depthA, depthB, K,
                  RA, tA, RB, tB, gt_maskA, gt_maskB):
    """(覆盖率 cov, 纯度 pur)：cov 衡量 split 修复，pur 惩罚 sticky/dirty 多余体素。"""
    from e2e_pipeline import world_points, VOX

    def vox(P):
        v = np.round(P / VOX).astype(np.int32)
        return set(map(tuple, v))

    Pw = np.vstack([world_points(maskA, depthA, K, RA, tA),
                    world_points(maskB, depthB, K, RB, tB)])
    Gw = np.vstack([world_points(gt_maskA, depthA, K, RA, tA),
                    world_points(gt_maskB, depthB, K, RB, tB)])
    vi, vg = vox(Pw), vox(Gw)
    if not vg or not vi:
        return None, None
    cov = len(vi & vg) / len(vg)
    pur = len(vi & vg) / len(vi)
    return cov, pur


def remedy_candidates(maskB, fixB):
    """两个补救候选：并集（补 split 缺块）/ 并集后限于传播带（再切 sticky）。"""
    band = dilate(fixB.astype(np.uint8) * 255, 5) > 0
    return [(maskB | fixB), (maskB | fixB) & band]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remedy", choices=["auto", "union", "banded"], default="auto")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")]
    by = {}
    for r in rows:
        by.setdefault((r["scene"], r["pair"], r["inst_idx"]), {})[r["frame"]] = r

    evals = []
    for (scene, pair, idx), d in sorted(by.items()):
        fa, fb = map(int, pair.split("_"))
        if fa not in d or fb not in d:
            continue
        sd = os.path.join(SCENE_DIR, scene)
        cam = json.load(open(os.path.join(sd, "scene_camera.json")))
        mdir = os.path.join(sd, "mask_visib")
        ddir = os.path.join(sd, "depth")

        maskA, maskB = load_sam2(scene, fa, idx), load_sam2(scene, fb, idx)
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

        # 级联判定（与 e2e_pipeline 相同）
        s1, _, _ = stage1_single_frame(maskB)
        if s1 in ("split", "dirty"):
            action, s2 = "RESEGMENT", None
        else:
            s2, nvis = stage2_cross_view(maskA, maskB, depthA, depthB, K, RA, tA, RB, tB)
            if s2 is None:
                continue
            action = "RESEGMENT" if s2 < THRESHOLD else "PASS"

        cov_sam, pur_sam = completeness2(maskA, maskB, depthA, depthB, K,
                                         RA, tA, RB, tB, gtA, gtB)
        cov_gt, pur_gt = completeness2(gtA, gtB, depthA, depthB, K,
                                       RA, tA, RB, tB, gtA, gtB)
        if cov_sam is None:
            continue

        if action == "RESEGMENT":
            # 真实补救：双向跨视图传播 + 候选生成 + 级联自查选优（不用 GT）
            dsA = cam[str(fa)].get("depth_scale", 0.1)
            dsB = cam[str(fb)].get("depth_scale", 0.1)
            fixB = propagate_mask(maskA, depthA, K, RA, tA, RB, tB, depthB, dsB)
            fixA = propagate_mask(maskB, depthB, K, RB, tB, RA, tA, depthA, dsA)
            if args.remedy == "union":
                candsA, candsB = remedy_candidates(maskA, fixA)[:1], remedy_candidates(maskB, fixB)[:1]
            elif args.remedy == "banded":
                candsA, candsB = remedy_candidates(maskA, fixA)[1:], remedy_candidates(maskB, fixB)[1:]
            else:
                candsA = remedy_candidates(maskA, fixA)
                candsB = remedy_candidates(maskB, fixB)
            best, best_score = None, -1.0
            for ca in candsA:
                for cb in candsB:
                    s, _ = stage2_cross_view(ca, cb, depthA, depthB, K, RA, tA, RB, tB)
                    if s is not None and s > best_score:
                        best, best_score = (ca, cb), s
            if best is None:
                best = (maskA, maskB)
            maskA_fix, maskB_fix = best
            cov_fix, pur_fix = completeness2(maskA_fix, maskB_fix, depthA, depthB, K,
                                             RA, tA, RB, tB, gtA, gtB)
        else:
            cov_fix, pur_fix = cov_sam, pur_sam

        evals.append({
            "scene": scene, "pair": pair, "idx": idx,
            "iou_b": d[fb]["iou_vs_gt"], "action": action,
            "s2": round(s2, 3) if s2 else None,
            "cov_sam": round(cov_sam, 3), "cov_fix": round(cov_fix, 3),
            "cov_gt": round(cov_gt, 3),
            "pur_sam": round(pur_sam, 3), "pur_fix": round(pur_fix, 3),
            "pur_gt": round(pur_gt, 3),
            "gain_cov": round(cov_fix - cov_sam, 3),
            "gain_pur": round(pur_fix - pur_sam, 3),
        })

    n = len(evals)
    pas = [e for e in evals if e["action"] == "PASS"]
    seg = [e for e in evals if e["action"] == "RESEGMENT"]

    print("=" * 66)
    print("端到端闭环 v2：级联调度 → 真实跨视图重分割补救 → 重建完整度")
    print("=" * 66)
    print(f"实例 {n} 个 | PASS {len(pas)} | RESEGMENT {len(seg)}\n")

    print("【PASS 组】好 mask，无需干预：")
    print(f"  平均输入完整度 = {np.mean([e['cov_sam'] for e in pas]):.3f} | "
          f"纯度 = {np.mean([e['pur_sam'] for e in pas]):.3f}")

    print("\n【RESEGMENT 组】差 mask，触发真实补救（跨视图传播 + 级联自查选优）：")
    print(f"  完整度: {np.mean([e['cov_sam'] for e in seg]):.3f} → "
          f"{np.mean([e['cov_fix'] for e in seg]):.3f}  "
          f"(实测 {np.mean([e['gain_cov'] for e in seg]):+.3f} | GT 上限 {np.mean([e['cov_gt'] for e in seg]):.3f})")
    print(f"  纯度  : {np.mean([e['pur_sam'] for e in seg]):.3f} → "
          f"{np.mean([e['pur_fix'] for e in seg]):.3f}  "
          f"(实测 {np.mean([e['gain_pur'] for e in seg]):+.3f} | GT 上限 {np.mean([e['pur_gt'] for e in seg]):.3f})")

    os.makedirs("results", exist_ok=True)
    out_csv = f"results/e2e_remedy_{args.remedy}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(evals[0].keys()))
        w.writeheader()
        w.writerows(evals)
    print(f"\n明细已存 {out_csv}")


if __name__ == "__main__":
    main()
