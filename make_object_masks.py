"""PSNR 闭环实验：生成物体级三组 mask（gt / defect / fixed）。

- gt     ：mask_visib 二值化（模拟"完美分割"，上限）
- defect ：对随机 defect_ratio 的可见帧注入 split 缺陷（模拟 SAM2 在部分视角失效）
- fixed  ：对缺陷帧做真实补救——从时间上最近的干净帧做跨视图传播（并集补全），
           全程不用缺陷帧的 GT

输出到 data/cloud_data/<scene>/obj<idx>/{gt,defect,fixed}/{frame:06d}.png

用法：python make_object_masks.py --scene 000048 --obj 0
"""
import argparse, json, os, sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_checker import load_gray  # noqa: E402
from prepare_mv_data import split_mask  # noqa: E402
from e2e_remedy import propagate_mask  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--obj", type=int, default=0)
    ap.add_argument("--defect-ratio", type=float, default=0.4)
    ap.add_argument("--mode", choices=["partial", "split100", "sticky100"], default="partial",
                    help="partial=随机部分帧注入split（默认）；split100=全部帧split；sticky100=全部帧与邻居实例粘连")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-root", default="data/cloud_data")
    args = ap.parse_args()

    sd = os.path.join("data/real_data/test", args.scene)
    cam = json.load(open(os.path.join(sd, "scene_camera.json")))
    mdir = os.path.join(sd, "mask_visib")
    ddir = os.path.join(sd, "depth")
    frames = sorted(cam.keys(), key=int)

    out = {k: os.path.join(args.out_root, args.scene, f"obj{args.obj}_{args.mode}", k)
           for k in ("gt", "defect", "fixed")}
    for d in out.values():
        os.makedirs(d, exist_ok=True)

    # 先收集 GT mask 与可见帧
    gt_masks, visible = {}, []
    for f in frames:
        p = os.path.join(mdir, f"{int(f):06d}_{args.obj:06d}.png")
        m = load_gray(p) if os.path.exists(p) else None
        if m is None:
            H, W = 480, 640
            m = np.zeros((H, W), dtype=bool)
        gt_masks[f] = m
        if m.sum() >= 50:
            visible.append(f)
    print(f"{args.scene} obj{args.obj}: 可见 {len(visible)}/{len(frames)} 帧")
    if len(visible) < 30:
        print("!! 可见帧太少，换一个 obj idx")
        return

    # 选缺陷帧（ seeded ）；split100/sticky100 为全部可见帧
    rng = np.random.default_rng(args.seed)
    if args.mode == "partial":
        n_def = int(len(visible) * args.defect_ratio)
        defect_frames = set(rng.choice(visible, size=n_def, replace=False).tolist())
    else:
        defect_frames = set(visible)

    def load_depth(f):
        return np.array(Image.open(os.path.join(ddir, f"{int(f):06d}.png"))).astype(np.float32)

    def pose(f):
        return (np.array(cam[f]["cam_K"]).reshape(3, 3),
                np.array(cam[f]["cam_R_w2c"]).reshape(3, 3),
                np.array(cam[f]["cam_t_w2c"]).reshape(3))

    fidx = {f: i for i, f in enumerate(frames)}
    for f in frames:
        gt = gt_masks[f]
        if f in defect_frames:
            if args.mode == "sticky100":
                nb = os.path.join(mdir, f"{int(f):06d}_{args.obj + 1:06d}.png")
                defect = gt | load_gray(nb) if os.path.exists(nb) else gt.copy()
            else:
                defect = split_mask(gt)
        else:
            defect = gt.copy()

        if f in defect_frames and args.mode == "partial":
            # 补救：找时间上最近的干净可见帧，传播其 mask 过来，与缺陷 mask 取并集
            # （100% 缺陷模式无干净源帧，正好演示"系统性缺陷必须重分割"）
            clean = [c for c in visible if c not in defect_frames]
            src = min(clean, key=lambda c: abs(fidx[c] - fidx[f]))
            Ks, Rs, ts = pose(src)
            Kd, Rd, td = pose(f)
            fix = propagate_mask(gt_masks[src], load_depth(src), Ks, Rs, ts,
                                 Rd, td, load_depth(f),
                                 cam[f].get("depth_scale", 0.1))
            fixed = defect | fix
        else:
            fixed = defect.copy()

        Image.fromarray(gt.astype(np.uint8) * 255).save(os.path.join(out["gt"], f"{int(f):06d}.png"))
        Image.fromarray(defect.astype(np.uint8) * 255).save(os.path.join(out["defect"], f"{int(f):06d}.png"))
        Image.fromarray(fixed.astype(np.uint8) * 255).save(os.path.join(out["fixed"], f"{int(f):06d}.png"))

    #  sanity：缺陷帧的 IoU(defect, gt) 与 IoU(fixed, gt)
    ious_d, ious_f = [], []
    for f in defect_frames:
        gt = gt_masks[f]
        if gt.sum() == 0:
            continue
        d = gt & (load_gray(os.path.join(out["defect"], f"{int(f):06d}.png")))
        u = gt | (load_gray(os.path.join(out["defect"], f"{int(f):06d}.png")))
        ious_d.append(d.sum() / max(u.sum(), 1))
        fx = load_gray(os.path.join(out["fixed"], f"{int(f):06d}.png"))
        ious_f.append((gt & fx).sum() / max((gt | fx).sum(), 1))
    print(f"缺陷帧 {len(defect_frames)} | IoU defect={np.mean(ious_d):.3f} → fixed={np.mean(ious_f):.3f}")
    print(f"输出：{out['gt']} 等三组 × {len(frames)} 帧")


if __name__ == "__main__":
    main()
