"""纯 PyTorch 3DGS 渲染（先出图验证管线，再训练）。

用法：
  1. 先只渲染（不训练）：python train_torchgs.py --data_dir cloud_data/000048 --render_only
  2. 训练：python train_torchgs.py --data_dir cloud_data/000048 --max_steps 2000
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def read_cameras(path):
    cams = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            p = line.split()
            cams[int(p[0])] = (int(p[2]), int(p[3]), float(p[4]), float(p[5]), float(p[6]), float(p[7]))
    return cams


def read_images(path):
    out = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    for line in lines:
        p = line.split()
        q = np.array([float(p[1]), float(p[2]), float(p[3]), float(p[4])])
        t = np.array([float(p[5]), float(p[6]), float(p[7])])
        out.append((q, t, int(p[8]), p[9]))
    return out


def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w, 2*x*z+2*y*w],
        [2*x*y+2*z*w, 1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w, 2*y*z+2*x*w, 1-2*x*x-2*y*y]])


def read_points3D(path):
    """读 COLMAP points3D.txt → (N,3) 坐标 + (N,3) RGB(0-1)。空文件返回 None。"""
    pts, cols = [], []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            pts.append([float(p[1]), float(p[2]), float(p[3])])
            cols.append([float(p[4]) / 255, float(p[5]) / 255, float(p[6]) / 255])
    if not pts:
        return None
    return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.float32)


def render(means, scales, opacities, colors, viewmat, K, W, H, device):
    """简化 3DGS 渲染（向量化，无 inplace）。"""
    R = viewmat[:3, :3]; t = viewmat[:3, 3]
    cam_pts = (R @ means.T).T + t
    valid = cam_pts[:, 2] > 0.1
    if valid.sum() < 10:
        return torch.ones(H, W, 3, device=device)

    cam_pts = cam_pts[valid]
    s = scales[valid].mean(dim=-1)
    o = opacities[valid]
    c = colors[valid]

    fx, fy = K[0, 0], K[1, 1]; cx, cy = K[0, 2], K[1, 2]
    x2d = fx * cam_pts[:, 0] / cam_pts[:, 2] + cx
    y2d = fy * cam_pts[:, 1] / cam_pts[:, 2] + cy
    sig = (s * fx / cam_pts[:, 2]).clamp(0.5, 15.0)

    # 按深度排序（远→近）
    order = torch.argsort(cam_pts[:, 2], descending=True)
    x2d = x2d[order]; y2d = y2d[order]; sig = sig[order]; o = o[order]; c = c[order]

    # 低分辨率渲染：投影坐标与网格统一换算到半分辨率
    W2, H2 = W // 2, H // 2
    x2d = x2d / 2; y2d = y2d / 2; sig = sig / 2
    img = torch.ones(H2, W2, 3, device=device)
    yy, xx = torch.meshgrid(torch.arange(H2, device=device).float(),
                           torch.arange(W2, device=device).float(), indexing="ij")

    for i in range(x2d.shape[0]):
        if o[i] < 0.01: continue
        r = sig[i] * 3
        x0 = max(0, int(x2d[i].item() - r)); x1 = min(W2, int(x2d[i].item() + r) + 1)
        y0 = max(0, int(y2d[i].item() - r)); y1 = min(H2, int(y2d[i].item() + r) + 1)
        if x0 >= x1 or y0 >= y1: continue
        dx = xx[y0:y1, x0:x1] - x2d[i]
        dy = yy[y0:y1, x0:x1] - y2d[i]
        g = torch.exp(-(dx**2 + dy**2) / (2 * sig[i]**2 + 1e-6))
        a = o[i] * g
        region = img[y0:y1, x0:x1]
        new_region = region * (1 - a.unsqueeze(-1)) + c[i].view(1, 1, 3) * a.unsqueeze(-1)
        # out-of-place 更新
        img_top = img[:y0]
        img_mid = torch.cat([img[y0:y1, :x0], new_region, img[y0:y1, x1:]], dim=1)
        img_bot = img[y1:]
        img = torch.cat([img_top, img_mid, img_bot], dim=0)

    return F.interpolate(img.permute(2,0,1).unsqueeze(0), size=(H,W), mode="bilinear")[0].permute(1,2,0).clamp(0,1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--n_gaussians", type=int, default=10000)
    ap.add_argument("--render_only", action="store_true")
    ap.add_argument("--out", default="results/render_output/000048")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = args.device
    cams = read_cameras(f"{args.data_dir}/sparse/0/cameras.txt")
    imgs_meta = read_images(f"{args.data_dir}/sparse/0/images.txt")

    viewmats, Ks, images = [], [], []
    for q, t, cid, name in imgs_meta:
        w, h, fx, fy, cx, cy = cams[cid]
        R = qvec2rotmat(q)
        vm = np.eye(4); vm[:3,:3] = R; vm[:3,3] = t
        K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
        viewmats.append(vm); Ks.append(K)
        images.append(np.array(Image.open(f"{args.data_dir}/images/{name}")) / 255.0)

    viewmats = torch.tensor(np.stack(viewmats), dtype=torch.float32, device=device)
    Ks = torch.tensor(np.stack(Ks), dtype=torch.float32, device=device)
    images = torch.tensor(np.stack(images), dtype=torch.float32, device=device)
    C, H, W = images.shape[0], images.shape[1], images.shape[2]
    print(f"加载 {C} 帧，{W}x{H}")

    # 初始化：优先用 points3D（深度反投影点云），否则围绕相机中心随机
    pc = read_points3D(f"{args.data_dir}/sparse/0/points3D.txt")
    if pc is not None:
        pts, cols = pc
        means = torch.tensor(pts, dtype=torch.float32, device=device)
        N = means.shape[0]
        colors = torch.tensor(cols, dtype=torch.float32, device=device)
        sub = means[torch.randint(0, N, (min(2000, N),))]
        d = torch.cdist(sub, means).kthvalue(4, dim=1).values.mean().clamp(min=1.0)
        scales = torch.ones(N, 3, device=device) * d
        opacities = 0.5 * torch.ones(N, device=device)
        print(f"从 points3D 初始化 {N} 个高斯，初始尺度 {d.item():.2f}mm")
    else:
        N = args.n_gaussians
        cam_centers = -torch.bmm(viewmats[:,:3,:3].transpose(1,2), viewmats[:,:3,3:4]).squeeze(-1)
        center = cam_centers.mean(0)
        radius = (cam_centers - center).norm(dim=-1).mean() * 1.5
        means = (torch.rand(N, 3, device=device) - 0.5) * 0.6 * radius + center
        scales = torch.ones(N, 3, device=device) * radius / 8
        opacities = 0.5 * torch.ones(N, device=device)
        colors = torch.rand(N, 3, device=device)
    for p in (means, scales, opacities, colors):
        p.requires_grad_(True)

    os.makedirs(args.out, exist_ok=True)

    # 先渲染一帧看看
    print("渲染第 0 帧（随机初始化）...")
    with torch.no_grad():
        img = render(means, scales, opacities, colors, viewmats[0], Ks[0], W, H, device)
    arr = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(arr).save(f"{args.out}/preview_frame0.png")
    print(f">> 预览已保存: {args.out}/preview_frame0.png")

    if args.render_only:
        return

    optimizer = torch.optim.Adam([
        {"params": [means], "lr": 1e-4},
        {"params": [scales], "lr": 1e-3},
        {"params": [opacities], "lr": 1e-2},
        {"params": [colors], "lr": 1e-3},
    ])

    for step in range(args.max_steps):
        idx = torch.randint(0, C, (1,)).item()
        pred = render(means, scales, opacities, colors, viewmats[idx], Ks[idx], W, H, device)
        loss = ((pred - images[idx]) ** 2).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if step % 50 == 0:
            print(f"step {step}/{args.max_steps}  loss={loss.item():.5f}")
        if step % 500 == 0 or step == args.max_steps - 1:
            with torch.no_grad():
                for i in range(C):
                    img = render(means, scales, opacities, colors, viewmats[i], Ks[i], W, H, device)
                    arr = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(arr).save(f"{args.out}/render_{i:03d}.png")
            print(f">> 保存 {C} 张渲染图")


if __name__ == "__main__":
    main()
