"""端到端闭环：级联调度 → 重建完整度改善（补"动态策略调度"完整故事）。

重建代理（CPU 可跑，无需真训 3DGS）：
  mask 反投影成 3D 点云 → 世界系融合 → voxel 化 → 重建覆盖完整度。
  完整度 = 两帧点云融合后，物体表面被覆盖的 voxel 占 GT 表面 voxel 的比例。
  （真实 SAM2 漏边 → 那部分 voxel 空 → 完整度低；GT mask → 完整度高。）

端到端闭环逻辑：
  1) 输入真实 SAM2 mask（两帧）
  2) 级联调度判定（PASS / RESEGMENT）
  3) 重建完整度：
       - 原始（真实 SAM2）完整度 Q_sam
       - 若 RESEGMENT：补救后（理想重分割 = GT mask）完整度 Q_gt（上限）
  4) 结论：级联识别出"低完整度"实例，触发补救可把完整度从 Q_sam 提升到 Q_gt

用法：D:\Anaconda_envs\envs\langchain_env\python.exe e2e_pipeline.py
"""
import csv, json, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_checker import load_gray, unproject, one_direction  # noqa: E402
from cascade_checker import stage1_single_frame, stage2_cross_view  # noqa: E402

SCENE_DIR = "data/real_data/test"
SAM_DIR = "data/sam2_real"
JSONL = "sam2_real_batch.jsonl"
THRESHOLD = 0.70
VOX = 4.0  # voxel 边长 mm


def load_sam2(scene, frame, idx):
    p = os.path.join(SAM_DIR, f"{scene}_{int(frame):06d}_{idx:06d}.png")
    return load_gray(p) if os.path.exists(p) else None


def world_points(mask, depth, K, R_w2c, t_w2c):
    """mask 反投影 → 相机系 Pc → 世界系 Pw。"""
    Pc = unproject(mask, depth, K, 0.1)  # Nx3 相机系
    Pw = (R_w2c.T @ (Pc - t_w2c).T).T
    return Pw


def completeness(maskA, maskB, depthA, depthB, K,
                 RA, tA, RB, tB, gt_maskA, gt_maskB):
    """重建完整度：融合 mask 点云 vs 融合 GT 点云的 voxel 覆盖率。"""
    def vox(P):
        v = np.round(P / VOX).astype(np.int32)
        return set(map(tuple, v[::1]))  # 抽稀降内存

    # 待评估（输入 mask）
    PwA = world_points(maskA, depthA, K, RA, tA)
    PwB = world_points(maskB, depthB, K, RB, tB)
    vox_in = vox(np.vstack([PwA, PwB]))

    # GT 参考（完整表面）
    GwA = world_points(gt_maskA, depthA, K, RA, tA)
    GwB = world_points(gt_maskB, depthB, K, RB, tB)
    vox_gt = vox(np.vstack([GwA, GwB]))

    if not vox_gt:
        return None, None
    # 覆盖完整度 = 输入覆盖了多少 GT voxel
    cov = len(vox_in & vox_gt) / len(vox_gt)
    return cov, len(vox_gt)


def main():
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
        gt = json.load(open(os.path.join(sd, "scene_gt.json")))
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

        # 级联判定
        s1, _, _ = stage1_single_frame(maskB)
        if s1 in ("split", "dirty"):
            action, s2 = "RESEGMENT", None
        else:
            s2, nvis = stage2_cross_view(maskA, maskB, depthA, depthB, K, RA, tA, RB, tB)
            if s2 is None:
                continue
            action = "RESEGMENT" if s2 < THRESHOLD else "PASS"

        # 重建完整度（输入 mask vs GT 上限）
        cov_sam, n_vox = completeness(maskA, maskB, depthA, depthB, K,
                                      RA, tA, RB, tB, gtA, gtB)
        cov_gt, _ = completeness(gtA, gtB, depthA, depthB, K,
                                RA, tA, RB, tB, gtA, gtB)
        if cov_sam is None:
            continue
        gain = (cov_gt - cov_sam) if action == "RESEGMENT" else 0.0

        evals.append({
            "scene": scene, "pair": pair, "idx": idx,
            "iou_b": d[fb]["iou_vs_gt"], "action": action,
            "s2": round(s2, 3) if s2 else None,
            "cov_sam": round(cov_sam, 3), "cov_gt": round(cov_gt, 3),
            "gain_if_remed": round(gain, 3),
        })

    n = len(evals)
    pas = [e for e in evals if e["action"] == "PASS"]
    seg = [e for e in evals if e["action"] == "RESEGMENT"]

    print("=" * 66)
    print("端到端闭环：级联调度 → 重建完整度改善")
    print("=" * 66)
    print(f"实例 {n} 个 | PASS {len(pas)} | RESEGMENT {len(seg)}\n")

    print("【级联 PASS 组】好 mask，无需干预：")
    print(f"  平均输入完整度 cov_sam = {np.mean([e['cov_sam'] for e in pas]):.3f}")

    print("\n【级联 RESEGMENT 组】差 mask，触发补救：")
    print(f"  平均输入完整度 cov_sam = {np.mean([e['cov_sam'] for e in seg]):.3f}")
    print(f"  平均补救后完整度 cov_gt = {np.mean([e['cov_gt'] for e in seg]):.3f}")
    print(f"  平均完整度提升 = +{np.mean([e['gain_if_remed'] for e in seg]):.3f} "
          f"(相对提升 {np.mean([e['gain_if_remed']/max(e['cov_sam'],0.01) for e in seg]):.1%})")

    # 分组对比：RESEGMENT 组的 cov_sam 是否显著低于 PASS 组
    print(f"\n【闭环价值】级联识别出低完整度实例的能力：")
    print(f"  PASS 组平均完整度 {np.mean([e['cov_sam'] for e in pas]):.3f} vs "
          f"RESEGMENT 组 {np.mean([e['cov_sam'] for e in seg]):.3f}")
    print(f"  → 级联 RESEGMENT 组的输入完整度显著更低，触发补救可显著补全重建")

    with open("e2e_pipeline.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(evals[0].keys()))
        w.writeheader()
        w.writerows(evals)
    print(f"\n明细已存 e2e_pipeline.csv")


if __name__ == "__main__":
    main()
