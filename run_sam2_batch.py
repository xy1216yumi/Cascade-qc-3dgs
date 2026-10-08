"""批量真实 SAM2 推理：12 场景各取适中视角差帧对，产出真实分割 mask + IoU。

输出：
    sam2_real/{scene}_{frame:06d}_{idx:06d}.png   SAM2 真实 mask
    sam2_real_batch.jsonl  每行: scene/frame_a/frame_b/idx/iou_a/iou_b/box
"""
import io, json, os, sys, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mv_eval_all import pick_frame_pair  # noqa: E402

CKPT = r"D:\AAAApython\3Dpipelines_pytorch-env\sam2_hiera_large.pt"
CFG = "configs/sam2/sam2_hiera_l.yaml"
SCENE_DIR = "data/real_data/test"
OUT_DIR = "data/sam2_real"
OUT_JSONL = "sam2_real_batch.jsonl"


def bbox_of(mask):
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def iou(a, b):
    a, b = a > 0, b > 0
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    scenes = sorted(d for d in os.listdir(SCENE_DIR)
                    if os.path.isdir(os.path.join(SCENE_DIR, d)))

    print("加载 SAM2 large（CPU）...", flush=True)
    t0 = time.time()
    import torch
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor = SAM2ImagePredictor(build_sam2(CFG, CKPT, device="cpu"))
    print(f"模型加载 {time.time()-t0:.1f}s\n", flush=True)

    rows = []
    for scene in scenes:
        sd = os.path.join(SCENE_DIR, scene)
        cam = json.load(open(os.path.join(sd, "scene_camera.json")))
        gt = json.load(open(os.path.join(sd, "scene_gt.json")))
        mdir = os.path.join(sd, "mask_visib")
        rdir = os.path.join(sd, "rgb")
        pair = pick_frame_pair(cam, gt)
        if pair is None:
            print(f"[跳过] {scene}: 无帧对")
            continue
        fa, fb = pair
        for frame in (int(fa), int(fb)):
            rgb = np.array(Image.open(os.path.join(rdir, f"{frame:06d}.png")).convert("RGB"))
            predictor.set_image(rgb)
            n_inst = len(gt.get(str(frame), []))
            for idx in range(min(n_inst, 6)):
                gt_mask = np.array(Image.open(os.path.join(mdir, f"{frame:06d}_{idx:06d}.png")))
                box = bbox_of(gt_mask)
                if box is None:
                    continue
                masks, scores, _ = predictor.predict(box=box[None, :], multimask_output=False)
                sam = (masks[0] > 0).astype(np.uint8) * 255
                Image.fromarray(sam).save(os.path.join(OUT_DIR, f"{scene}_{frame:06d}_{idx:06d}.png"))
                rows.append({"scene": scene, "frame": frame, "inst_idx": idx,
                             "iou_vs_gt": round(iou(sam, gt_mask), 4),
                             "score": round(float(scores[0]), 4),
                             "pair": f"{fa}_{fb}"})
        print(f"[{scene}] 帧对 {fa}->{fb} 完成", flush=True)

    with io.open(OUT_JSONL, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n完成 {len(rows)} 个真实 SAM2 mask → {OUT_JSONL}")


if __name__ == "__main__":
    main()
