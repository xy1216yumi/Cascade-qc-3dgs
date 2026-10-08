"""级联质量校验调度器：单帧规则（Stage 1）→ 跨视图复核（Stage 2）→ 触发动作。

论文定位：质量校验闭环（VLM 智能体调度器的核心决策模块）——不是所有实例都跑
最贵的检查，而是按"单帧能否消解"动态路由，仅在单帧无解时升级到跨视图。

调度逻辑：
  Stage 1 单帧规则（便宜，零外部依赖）——帧B 待查 mask 的连通域分析：
    * split（>=2 个 >=25% 面积的大连通域）→ 冲突 → 动作 RESEGMENT，不进 Stage 2
    * dirty（存在 <25% 面积的游离碎块）→ 冲突 → 动作 RESEGMENT，不进 Stage 2
    * 单连通域且无碎块 → 单帧无解（clean 与 sticky 在此不可区分）→ 进 Stage 2
  Stage 2 跨视图一致性（昂贵，需深度+位姿）——帧A 参考 mask vs 帧B 待查 mask：
    * score < 阈值 → 冲突 → 动作 REGENERATE（sticky 需重生成多视图以分离）
    * score >= 阈值 → 一致 → 动作 PASS（质量合格，进入重建管线）

指标：
  - 级联判定准确率（冲突=需处理 vs 一致=放行）
  - 跨视图调用率（省算力证据：Stage 1 直接拦截的比例）

用法：
    python cascade_checker.py --scene-dir real_data/test/000048 --answers mv_answers.csv
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_checker import load_gray, unproject, one_direction  # noqa: E402
from geo_checker import connected_components  # noqa: E402

THRESHOLD = 0.70
VIS_TOL = 30.0


def stage1_single_frame(mask, min_dust_px=50):
    """单帧规则检查。返回 (判定, 理由, 最大小连通域占比)。

    判定: 'split' | 'dirty' | 'unsure'
    min_dust_px：小于该像素数的连通块视为渲染毛刺忽略（BOP GT mask 边缘
    常有 1~30px 孤立伪影；缺陷注入的杂块半径 6-12px，面积 >100px，不受影响）。
    """
    mask_area = int((mask > 0).sum())
    if mask_area == 0:
        return "unsure", "空mask", 0.0
    n_cc, areas = connected_components(mask > 0)
    big = [a for a in areas if a >= 0.25 * mask_area]
    if len(big) >= 2:
        return "split", f"mask分为{len(big)}个大块", 0.0
    small = [a for a in areas if min_dust_px <= a < 0.25 * mask_area]
    if small:
        mx = max(small) / mask_area
        return "dirty", "存在游离小碎块", mx
    return "unsure", "单连通域无碎块", 0.0


def stage2_cross_view(maskA, maskB, depthA, depthB, K, RA, tA, RB, tB, vis_tol=VIS_TOL,
                      depth_scale=0.1, min_vis=300):
    """跨视图一致性。返回 (score, n_vis)。score 为 min(A→B, B→A) 命中率。
    depth_scale 为深度图换算到 mm 的比例（YCB-V=0.1，LMO=1.0）。"""
    H, W = depthA.shape
    PcA = unproject(maskA, depthA, K, depth_scale)
    hit_AB, n_AB = one_direction(PcA, RA, tA, RB, tB, K, H, W, maskB, depthB, vis_tol,
                                 scale_dst=depth_scale)
    PcB = unproject(maskB, depthB, K, depth_scale)
    hit_BA, n_BA = one_direction(PcB, RB, tB, RA, tA, K, H, W, maskA, depthA, vis_tol,
                                 scale_dst=depth_scale)
    if min(n_AB, n_BA) < min_vis:
        return None, min(n_AB, n_BA)
    r_AB = hit_AB / n_AB
    r_BA = hit_BA / n_BA
    return min(r_AB, r_BA), min(n_AB, n_BA)


def inject_defect(maskB_gt, kind, mdir, fb, idx, n_inst, H, W):
    """与 mv_eval_all 相同的缺陷注入，保证可对比。"""
    from prepare_mv_data import split_mask, dilate, add_blob
    if kind == "split":
        return split_mask(maskB_gt)
    if kind == "dirty":
        return add_blob(dilate(maskB_gt, 3), H, W)
    if kind == "sticky":
        nb = os.path.join(mdir, f"{int(fb):06d}_{(idx + 1) % n_inst:06d}.png")
        return maskB_gt | load_gray(nb) if os.path.exists(nb) else maskB_gt
    return maskB_gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", default="data/real_data/test/000048")
    ap.add_argument("--answers", default="mv_answers.csv")
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--route", choices=["aggressive", "conservative"], default="aggressive",
                    help="aggressive=Stage1拦截split+dirty；conservative=仅拦截微碎块dirty(<2%%)，"
                         "split/大碎块升级跨视图复核（区分遮挡断裂与缺陷）")
    ap.add_argument("--report", default="cascade_report.txt")
    args = ap.parse_args()

    scene_dir = args.scene_dir
    cam = json.load(open(os.path.join(scene_dir, "scene_camera.json")))
    gt = json.load(open(os.path.join(scene_dir, "scene_gt.json")))
    mdir = os.path.join(scene_dir, "mask_visib")
    ddir = os.path.join(scene_dir, "depth")

    rows = list(csv.DictReader(open(args.answers, encoding="utf-8-sig")))
    rows = [r for r in rows if r["scene"] == os.path.basename(scene_dir.rstrip("/\\"))]
    if not rows:
        print("answers 中无该场景条目；如需全场景级联请用 cascade_eval_all.py")
        return

    print("=" * 66)
    print("级联质量校验调度器（Stage1 单帧规则 → Stage2 跨视图）")
    print("=" * 66)
    stage2_calls = 0
    n_skip = 0
    details = []
    stats = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}

    for r in rows:
        fa, fb = r["frame_a"], r["frame_b"]
        idx = int(r["inst_idx"])
        is_conflict = r["is_conflict"] == "是"
        n_inst = len(gt[fa])

        maskA = load_gray(os.path.join(mdir, f"{int(fa):06d}_{idx:06d}.png"))
        maskB_gt = load_gray(os.path.join(mdir, f"{int(fb):06d}_{idx:06d}.png"))
        H, W = maskB_gt.shape
        maskB = inject_defect(maskB_gt, r["defect"], mdir, fb, idx, n_inst, H, W)

        # ---- Stage 1：单帧规则 ----
        s1, s1_reason, dust_ratio = stage1_single_frame(maskB)
        if args.route == "aggressive":
            # 拦截一切 split/dirty。代价：真实遮挡断裂被误判（多一次重分割，不坏管线）；
            # 收益：dirty 100% 检出（跨视图对 dirty 盲，漏检会放行坏 mask 进重建，代价更高）
            intercept = s1 in ("split", "dirty")
        else:  # conservative：仅拦截微碎块 dirty（<2% 面积），split/大碎块升级复核
            intercept = (s1 == "dirty" and dust_ratio < 0.02)
        if intercept:
            pred_conflict = True
            action = "RESEGMENT"
            stage2_score = None
        else:
            # ---- Stage 2：跨视图复核 ----
            stage2_calls += 1
            depthA = np.array(Image.open(os.path.join(ddir, f"{int(fa):06d}.png"))).astype(np.float32)
            depthB = np.array(Image.open(os.path.join(ddir, f"{int(fb):06d}.png"))).astype(np.float32)
            K = np.array(cam[fa]["cam_K"]).reshape(3, 3)
            RA = np.array(cam[fa]["cam_R_w2c"]).reshape(3, 3)
            tA = np.array(cam[fa]["cam_t_w2c"]).reshape(3)
            RB = np.array(cam[fb]["cam_R_w2c"]).reshape(3, 3)
            tB = np.array(cam[fb]["cam_t_w2c"]).reshape(3)
            stage2_score, n_vis = stage2_cross_view(maskA, maskB, depthA, depthB,
                                                     K, RA, tA, RB, tB)
            if stage2_score is None:
                n_skip += 1
                continue
            pred_conflict = stage2_score < args.threshold
            action = "REGENERATE" if pred_conflict else "PASS"

        correct = pred_conflict == is_conflict
        if correct:
            stats["tp" if is_conflict else "tn"] += 1
        else:
            stats["fp" if not is_conflict else "fn"] += 1

        details.append((r, s1, s1_reason, stage2_score, pred_conflict, action, correct))
        s2_txt = f"{stage2_score:.3f}" if stage2_score is not None else "--"
        mark = "OK" if correct else "MISS"
        print(f"[{mark}] 帧{fa}→{fb} 实例{idx} [{r['defect']}] "
              f"S1={s1}({s1_reason}) S2={s2_txt} 判定={'冲突' if pred_conflict else '一致'} "
              f"动作={action} GT={'冲突' if is_conflict else '一致'}")

    n = len(details)
    acc = (stats["tp"] + stats["tn"]) / n if n else 0.0
    print("\n" + "=" * 66)
    print(f"级联调度结果：参与 {n} 实例（跳过低采样点 {n_skip}）")
    print(f"准确率: {stats['tp'] + stats['tn']}/{n} = {acc:.1%}")
    print(f"  冲突检出(TP): {stats['tp']} | 一致保留(TN): {stats['tn']} | "
          f"误报(FP): {stats['fp']} | 漏检(FN): {stats['fn']}")
    print(f"跨视图调用率: {stage2_calls}/{n} = {stage2_calls / n:.1%} "
          f"（Stage1 单帧拦截 {(n - stage2_calls) / n:.1%}，省算力证据）")

    by_kind = {}
    for r, s1, _, sc, pc, act, c in details:
        by_kind.setdefault(r["defect"], [0, 0, 0])
        by_kind[r["defect"]][0] += 1
        if pc:
            by_kind[r["defect"]][1] += 1
        by_kind[r["defect"]][2] += int(act == "PASS")
    print("\n分缺陷（检出冲突数 / PASS 数 / 总数）:")
    for k in ["clean", "split", "dirty", "sticky"]:
        if k in by_kind:
            con, tot = by_kind[k][1], by_kind[k][0]
            print(f"  {k:6s}: 冲突判定 {con}/{tot} | PASS {by_kind[k][2]}")

    with open(args.report, "w", encoding="utf-8") as f:
        f.write("级联质量校验调度器报告\n")
        f.write(f"场景: {scene_dir} | 阈值: {args.threshold}\n")
        f.write(f"准确率: {acc:.1%} | 跨视图调用率: {stage2_calls}/{n} = {stage2_calls / n:.1%}\n")
        for r, s1, s1r, sc, pc, act, c in details:
            f.write(f"{r['frame_a']}->{r['frame_b']} 实例{r['inst_idx']} [{r['defect']}] "
                    f"S1={s1} S2={sc if sc is None else round(sc, 3)} 动作={act} "
                    f"GT={'冲突' if r['is_conflict'] == '是' else '一致'}\n")
    print(f"\n报告已保存到 {args.report}")


if __name__ == "__main__":
    main()
