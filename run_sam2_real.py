"""真实 SAM2 mask 评测（补局限 3）。

之前所有验证 = GT mask + 人工注入缺陷（合成口径）。本脚本真跑 SAM2 推理，
产出"真实分割器 mask"，验证级联调度在真实噪声下是否仍有效。

流程：
  对每帧每个实例：
    1) 取 GT mask_visib 的外接矩形作为 box prompt（模拟真实流程的粗检测框）
    2) SAM2ImagePredictor 推理 → 真实分割 mask
    3) 算 SAM2 mask vs GT mask 的 IoU（= 真实质量参考）
  输出：
    sam2_real/{scene}_{frame}_{idx}.png   SAM2 输出 mask
    sam2_real_results.jsonl                scene/frame/idx/iou_vs_gt/box 等

评测口径（与合成实验对应）：
    - IoU 高（≈1.0）= SAM2 分割干净 → 级联应判 PASS
    - IoU 低（<0.8） = 真实缺陷（漏边/分裂/粘连）→ 级联应判 RESEGMENT
    验证：级联判定与"真实 IoU 质量"的吻合度。

运行（用 pytorch-env，CPU）：
    D:\Anaconda_envs\envs\pytorch-env\python.exe run_sam2_real.py --scene-dir real_data/test/000048 --frames 1133 2160
"""
import argparse, io, json, os, sys, time

import numpy as np
from PIL import Image

CKPT = r"D:\AAAApython\3Dpipelines_pytorch-env\sam2_hiera_large.pt"
CFG = "configs/sam2/sam2_hiera_l.yaml"


def bbox_of(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def iou(a, b):
    a = a > 0
    b = b > 0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--frames", nargs="+", type=int, default=None,
                    help="指定帧；不给则自动取有 mask 的前 N 帧")
    ap.add_argument("--out-dir", default="data/sam2_real")
    ap.add_argument("--out-jsonl", default="sam2_real_results.jsonl")
    ap.add_argument("--max-instances", type=int, default=6)
    args = ap.parse_args()

    scene_dir = args.scene_dir
    cam = json.load(open(os.path.join(scene_dir, "scene_camera.json")))
    gt = json.load(open(os.path.join(scene_dir, "scene_gt.json")))
    mdir = os.path.join(scene_dir, "mask_visib")
    rdir = os.path.join(scene_dir, "rgb")
    os.makedirs(args.out_dir, exist_ok=True)
    scene = os.path.basename(scene_dir.rstrip("/\\"))

    frames = args.frames if args.frames else sorted(
        int(f.split(".")[0]) for f in os.listdir(rdir))[:6]

    # 延迟导入 torch/sam2（仅本脚本用 pytorch-env 跑）
    print("加载 SAM2 large（CPU，较慢）...", flush=True)
    t0 = time.time()
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    model = build_sam2(CFG, CKPT, device="cpu")
    predictor = SAM2ImagePredictor(model)
    print(f"模型加载 {time.time()-t0:.1f}s", flush=True)

    results = []
    for fi, frame in enumerate(frames):
        rgb_path = os.path.join(rdir, f"{frame:06d}.png")
        if not os.path.exists(rgb_path):
            continue
        rgb = np.array(Image.open(rgb_path).convert("RGB"))
        n_inst = len(gt.get(str(frame), []))
        predictor.set_image(rgb)
        for idx in range(min(n_inst, args.max_instances)):
            gt_mask = np.array(Image.open(os.path.join(
                mdir, f"{frame:06d}_{idx:06d}.png")))
            box = bbox_of(gt_mask)
            if box is None:
                continue
            t1 = time.time()
            masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=False)
            sam_mask = (masks[0] > 0).astype(np.uint8) * 255
            iou_val = iou(sam_mask, gt_mask)
            Image.fromarray(sam_mask).save(os.path.join(
                args.out_dir, f"{scene}_{frame:06d}_{idx:06d}.png"))
            results.append({
                "scene": scene, "frame": frame, "inst_idx": idx,
                "iou_vs_gt": round(iou_val, 4), "best_score": round(float(scores[0]), 4),
                "box": [float(v) for v in box],
            })
            print(f"[{fi+1}/{len(frames)}] {scene} f{frame} 实例{idx}: "
                  f"IoU={iou_val:.3f} score={float(scores[0]):.3f} ({time.time()-t1:.1f}s)",
                  flush=True)

    with io.open(args.out_jsonl, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n完成 {len(results)} 个真实 SAM2 mask，已追加到 {args.out_jsonl}")


if __name__ == "__main__":
    main()
