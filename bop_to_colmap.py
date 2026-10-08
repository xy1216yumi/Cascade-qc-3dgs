"""BOP YCB-V → COLMAP 格式转换（gsplat simple_trainer 可读）。

gsplat 官方 simple_trainer 读 COLMAP 目录：
  sparse/0/cameras.txt
  sparse/0/images.txt
  sparse/0/points3D.txt
  images/*.jpg

输入 BOP：
  scene_camera.json: key=帧号, {cam_K(3x3), cam_R_w2c(9), cam_t_w2c(3), depth_scale}
  rgb/{frame:06d}.png
  depth/{frame:06d}.png（用于反投影生成 points3D，作为 3DGS 初始化点云）

输出 data/{scene}/:
  images/{frame:06d}.jpg
  sparse/0/cameras.txt / images.txt / points3D.txt

用法：python bop_to_colmap.py --src real_data/test/000048 --dst cloud_data/000048
"""
import argparse, json, os, shutil

import numpy as np
from PIL import Image


def write_cameras(f, w, h, fx, fy, cx, cy):
    # COLMAP PINHOLE: CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy
    f.write(f"1 PINHOLE {w} {h} {fx:.4f} {fy:.4f} {cx:.4f} {cy:.4f}\n")


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]])


def rotmat2qvec(R):
    # 旋转矩阵 → 四元数（COLMAP 约定 w,x,y,z）
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        i = np.argmax(np.diag(R))
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
    return np.array([w, x, y, z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    args = ap.parse_args()

    cam = json.load(open(os.path.join(args.src, "scene_camera.json")))
    rdir = os.path.join(args.src, "rgb")
    frames = sorted(cam.keys(), key=lambda intx: int(intx))

    os.makedirs(os.path.join(args.dst, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.dst, "sparse", "0"), exist_ok=True)

    # 假设所有帧内参相同（取第一帧）
    K0 = np.array(cam[frames[0]]["cam_K"]).reshape(3, 3)
    tmp = np.array(Image.open(os.path.join(rdir, f"{int(frames[0]):06d}.png")))
    H, W = tmp.shape[:2]
    fx, fy = K0[0, 0], K0[1, 1]
    cx, cy = K0[0, 2], K0[1, 2]

    with open(os.path.join(args.dst, "sparse", "0", "cameras.txt"), "w") as f:
        write_cameras(f, W, H, fx, fy, cx, cy)

    with open(os.path.join(args.dst, "sparse", "0", "images.txt"), "w") as f:
        for i, frame in enumerate(frames):
            R = np.array(cam[frame]["cam_R_w2c"]).reshape(3, 3)  # w2c
            t = np.array(cam[frame]["cam_t_w2c"]).reshape(3)
            qvec = rotmat2qvec(R)
            img_name = f"{int(frame):06d}.jpg"
            # 复制并转 jpg
            src_img = os.path.join(rdir, f"{int(frame):06d}.png")
            Image.open(src_img).convert("RGB").save(os.path.join(args.dst, "images", img_name))
            # COLMAP images.txt: IMAGE_ID QVEC(4) TVEC(3) CAMERA_ID NAME
            f.write(f"{i+1} {qvec[0]:.8f} {qvec[1]:.8f} {qvec[2]:.8f} {qvec[3]:.8f} "
                    f"{t[0]:.4f} {t[1]:.4f} {t[2]:.4f} 1 {img_name}\n")
            f.write("\n")  # 每个 images 条目占两行，第二行为空（无点观测）

    # 从深度图反投影生成初始化点云（COLMAP points3D 格式，track 留空）
    ddir = os.path.join(args.src, "depth")
    rng = np.random.default_rng(0)
    pts_per_frame = 400
    with open(os.path.join(args.dst, "sparse", "0", "points3D.txt"), "w") as f:
        pid = 1
        for frame in frames:
            depth_path = os.path.join(ddir, f"{int(frame):06d}.png")
            if not os.path.exists(depth_path):
                continue
            depth_scale = cam[frame].get("depth_scale", 1.0)
            D = np.array(Image.open(depth_path), dtype=np.float64) * depth_scale
            RGB = np.array(Image.open(os.path.join(rdir, f"{int(frame):06d}.png")).convert("RGB"))
            R = np.array(cam[frame]["cam_R_w2c"]).reshape(3, 3)
            t = np.array(cam[frame]["cam_t_w2c"]).reshape(3)
            ys, xs = np.where(D > 0)
            if len(xs) == 0:
                continue
            sel = rng.choice(len(xs), size=min(pts_per_frame, len(xs)), replace=False)
            ys, xs = ys[sel], xs[sel]
            z = D[ys, xs]
            X = (xs - cx) * z / fx
            Y = (ys - cy) * z / fy
            pc = np.stack([X, Y, z], axis=1)          # 相机系
            pw = (pc - t) @ R                          # w2c 的逆：pw = R^T (pc - t)，行向量即 (pc - t) @ R
            colors = RGB[ys, xs]
            for i in range(len(pw)):
                f.write(f"{pid} {pw[i,0]:.4f} {pw[i,1]:.4f} {pw[i,2]:.4f} "
                        f"{colors[i,0]} {colors[i,1]} {colors[i,2]} 1.0\n")
                pid += 1
        n_pts = pid - 1

    print(f"转换完成：{len(frames)} 帧 → {args.dst}")
    print(f"  images/ 共 {len(frames)} 张 jpg")
    print(f"  sparse/0/ (cameras.txt, images.txt, points3D.txt 共 {n_pts} 点)")


if __name__ == "__main__":
    main()
