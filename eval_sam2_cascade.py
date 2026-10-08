"""真实 SAM2 mask 上的级联调度评测（补局限 3）。

与合成实验的区别：
  合成口径：GT mask + 人工注入缺陷，真值="是否注入"
  真实口径：SAM2 真实分割 mask，真值="SAM2 mask vs GT 的 IoU"（真实质量）

流程：
  1) 读 sam2_real_batch.jsonl，按场景/帧对/实例分组
  2) 对每个实例：帧A/帧B 都是真实 SAM2 mask
     - Stage1 单帧规则（对帧B 真实 mask）
     - Stage2 跨视图一致性（真实 mask 互查，深度+位姿同前）
     - 级联动作 PASS / RESEGMENT
  3) 用帧B IoU 分档作真实质量参考：
       IoU 低（< iou_bad）  = 真实差 mask（期望 RESEGMENT）
       IoU 高（>= iou_good）= 好 mask（期望 PASS）
     先打印 IoU 分布，再按分档算吻合率
  4) 输出真实口径下：级联判定 vs 真实质量的吻合率

用法：D:\Anaconda_envs\envs\langchain_env\python.exe eval_sam2_cascade.py
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


def load_sam2_mask(scene, frame, idx):
    p = os.path.join(SAM_DIR, f"{scene}_{int(frame):06d}_{idx:06d}.png")
    if not os.path.exists(p):
        return None
    return load_gray(p)


def main():
    rows = [json.loads(l) for l in open(JSONL, encoding="utf-8")]
    # 按 scene+pair+idx 聚合：fa 行与 fb 行
    by = {}
    for r in rows:
        key = (r["scene"], r["pair"], r["inst_idx"])
        by.setdefault(key, {})[r["frame"]] = r

    evals = []
    for (scene, pair, idx), d in sorted(by.items()):
        fa, fb = map(int, pair.split("_"))
        if fa not in d or fb not in d:
            continue
        sd = os.path.join(SCENE_DIR, scene)
        cam = json.load(open(os.path.join(sd, "scene_camera.json")))
        ddir = os.path.join(sd, "depth")

        maskA = load_sam2_mask(scene, fa, idx)
        maskB = load_sam2_mask(scene, fb, idx)
        if maskA is None or maskB is None:
            continue

        # Stage1
        s1, s1_reason, dust = stage1_single_frame(maskB)
        intercept = s1 in ("split", "dirty")
        if intercept:
            action, s2 = "RESEGMENT", None
        else:
            depthA = np.array(Image.open(os.path.join(ddir, f"{fa:06d}.png"))).astype(np.float32)
            depthB = np.array(Image.open(os.path.join(ddir, f"{fb:06d}.png"))).astype(np.float32)
            K = np.array(cam[str(fa)]["cam_K"]).reshape(3, 3)
            RA = np.array(cam[str(fa)]["cam_R_w2c"]).reshape(3, 3)
            tA = np.array(cam[str(fa)]["cam_t_w2c"]).reshape(3)
            RB = np.array(cam[str(fb)]["cam_R_w2c"]).reshape(3, 3)
            tB = np.array(cam[str(fb)]["cam_t_w2c"]).reshape(3)
            s2, nvis = stage2_cross_view(maskA, maskB, depthA, depthB, K, RA, tA, RB, tB)
            if s2 is None:
                continue
            action = "RESEGMENT" if s2 < THRESHOLD else "PASS"

        evals.append({
            "scene": scene, "pair": pair, "idx": idx,
            "iou_b": d[fb]["iou_vs_gt"], "iou_a": d[fa]["iou_vs_gt"],
            "s1": s1, "s2": round(s2, 3) if s2 else None, "action": action,
        })

    # ===== IoU 分布 =====
    ious = sorted(e["iou_b"] for e in evals)
    print("=" * 66)
    print(f"真实 SAM2 级联评测 · 实例 {len(evals)} 个")
    print("=" * 66)
    print("帧B IoU 分布（SAM2 真实质量）:")
    print(f"  min={ious[0]:.3f}  P25={ious[len(ious)//4]:.3f}  "
          f"median={ious[len(ious)//2]:.3f}  "
          f"P75={ious[3*len(ious)//4]:.3f}  max={ious[-1]:.3f}")

    # ===== 分档吻合率 =====
    # 低质量档：IoU < 0.82（漏边/分裂/粘连）→ 期望 RESEGMENT
    # 高质量档：IoU >= 0.90 → 期望 PASS
    IOU_BAD, IOU_GOOD = 0.82, 0.90
    bad = [e for e in evals if e["iou_b"] < IOU_BAD]
    good = [e for e in evals if e["iou_b"] >= IOU_GOOD]
    mid = [e for e in evals if IOU_BAD <= e["iou_b"] < IOU_GOOD]

    def seg_rate(grp):
        if not grp:
            return 0, 0
        n_seg = sum(1 for e in grp if e["action"] == "RESEGMENT")
        return n_seg, len(grp)

    print(f"\n分档（IoU 阈值 bad<{IOU_BAD} / good>={IOU_GOOD}）:")
    nb, nt = seg_rate(bad)
    ng, gt_ = seg_rate(good)
    print(f"  低质量 IoU<{IOU_BAD}: {len(bad)} 个 → 级联判 RESEGMENT {nb}/{nt} = "
          f"{nb/nt:.1%}" if nt else f"  低质量: 0")
    print(f"  中间档 {IOU_BAD}~{IOU_GOOD}: {len(mid)} 个")
    print(f"  高质量 IoU>={IOU_GOOD}: {len(good)} 个 → 级联判 PASS {gt_-ng}/{gt_} = "
          f"{(gt_-ng)/gt_:.1%}" if gt_ else f"  高质量: 0")

    # 总体方向相关性：低质量的平均判 RESEGMENT 率应远高于高质量
    print(f"\n判定 vs IoU 关系:")
    print(f"  低质量组平均 IoU = {np.mean([e['iou_b'] for e in bad]):.3f}, 组内 RESEGMENT 率 = "
          f"{nb/nt:.1%}" if nt else "  低质量组: 0")
    print(f"  高质量组平均 IoU = {np.mean([e['iou_b'] for e in good]):.3f}, 组内 RESEGMENT 率 = "
          f"{ng/gt_:.1%}" if gt_ else "  高质量组: 0")

    with open("sam2_cascade_eval.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(evals[0].keys()))
        w.writeheader()
        w.writerows(evals)
    print(f"\n明细已存 sam2_cascade_eval.csv")


if __name__ == "__main__":
    main()
