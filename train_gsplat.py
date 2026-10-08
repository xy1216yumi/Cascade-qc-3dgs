"""自包含 3DGS 训练脚本（gsplat 1.5.3，COLMAP 格式）。

用法：
  python train_gsplat.py --data_dir cloud_data/000048 --max_steps 7000 --out output/000048
  # 增强版（致密化 + SSIM）：
  python train_gsplat.py --data_dir cloud_data/000048 --max_steps 15000 --densify --ssim_lambda 0.2
"""
import argparse, os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat.rendering import rasterization


def read_cameras(path):
    cams = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            cid = int(p[0])
            w, h = int(p[2]), int(p[3])
            fx, fy, cx, cy = float(p[4]), float(p[5]), float(p[6]), float(p[7])
            cams[cid] = (w, h, fx, fy, cx, cy)
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


def ssim_loss(pred, gt):
    """标准 SSIM（11x11 高斯窗，sigma=1.5）。pred/gt: (H,W,3) in [0,1]。返回 1-SSIM。"""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    g = torch.exp(-((torch.arange(11, device=pred.device).float() - 5) ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    kernel = (g[:, None] @ g[None, :]).expand(3, 1, 11, 11).contiguous()
    p = pred.permute(2, 0, 1).unsqueeze(0)
    t = gt.permute(2, 0, 1).unsqueeze(0)
    mu_p = F.conv2d(p, kernel, padding=5, groups=3)
    mu_t = F.conv2d(t, kernel, padding=5, groups=3)
    mu_p2, mu_t2, mu_pt = mu_p ** 2, mu_t ** 2, mu_p * mu_t
    sp = F.conv2d(p * p, kernel, padding=5, groups=3) - mu_p2
    st = F.conv2d(t * t, kernel, padding=5, groups=3) - mu_t2
    spt = F.conv2d(p * t, kernel, padding=5, groups=3) - mu_pt
    ssim_map = ((2 * mu_pt + C1) * (2 * spt + C2)) / ((mu_p2 + mu_t2 + C1) * (sp + st + C2))
    return 1 - ssim_map.mean()


def make_optimizer(params):
    return torch.optim.Adam([
        {"params": [params["means"]], "lr": 1.6e-4},
        {"params": [params["quats"]], "lr": 1e-3},
        {"params": [params["scales"]], "lr": 5e-3},
        {"params": [params["opacities"]], "lr": 5e-2},
        {"params": [params["sh0"]], "lr": 2.5e-3},
        {"params": [params["shN"]], "lr": 2.5e-3 / 20},
    ])


def densify(params, grad_acc, scene_scale):
    """简化致密化（原版 3DGS 思路）：克隆高梯度小球、分裂高梯度大球、剪除透明球。"""
    means, scales, opacities = params["means"], params["scales"], params["opacities"]
    thr = torch.quantile(grad_acc, 0.98)
    sel = grad_acc >= thr
    big = torch.exp(scales).max(dim=1).values > (scene_scale / 20)
    clone_idx = torch.where(sel & ~big)[0]
    split_idx = torch.where(sel & big)[0]

    if len(split_idx):  # 大球：原球缩小 + 复制一个偏移小球
        n = len(split_idx)
        noise = torch.randn(n, 3, device=means.device) * torch.exp(scales[split_idx]).mean(dim=1, keepdim=True)
        for k in params:
            extra = params[k][split_idx].clone()
            params[k] = torch.cat([params[k], extra], dim=0)
        params["means"][-n:] += noise
        params["scales"][-n:] -= np.log(1.6)
        params["scales"][split_idx] -= np.log(1.6)
    if len(clone_idx):  # 小球：原位克隆
        for k in params:
            params[k] = torch.cat([params[k], params[k][clone_idx].clone()], dim=0)

    # 剪除几乎透明的球
    keep = torch.sigmoid(params["opacities"]) > 0.005
    for k in params:
        params[k] = params[k][keep]
    for k in params:
        params[k] = params[k].detach().requires_grad_(True)
    return params, make_optimizer(params)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--max_steps", type=int, default=7000)
    ap.add_argument("--n_gaussians", type=int, default=80000)
    ap.add_argument("--out", default="results/render_output/000048")
    ap.add_argument("--mask_dir", default=None,
                    help="可选：物体 mask 目录（{帧号:06d}.png），损失只在 mask 区域计算")
    ap.add_argument("--densify", action="store_true", help="开启简化致密化（克隆/分裂/剪枝）")
    ap.add_argument("--densify_interval", type=int, default=500)
    ap.add_argument("--ssim_lambda", type=float, default=0.0, help="SSIM 损失权重（建议 0.2）")
    args = ap.parse_args()

    device = "cuda"
    cams = read_cameras(f"{args.data_dir}/sparse/0/cameras.txt")
    imgs_meta = read_images(f"{args.data_dir}/sparse/0/images.txt")

    viewmats, Ks, images = [], [], []
    for q, t, cid, name in imgs_meta:
        w, h, fx, fy, cx, cy = cams[cid]
        R = qvec2rotmat(q)
        vm = np.eye(4); vm[:3, :3] = R; vm[:3, 3] = t
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        viewmats.append(vm); Ks.append(K)
        images.append(np.array(Image.open(f"{args.data_dir}/images/{name}")) / 255.0)

    viewmats = torch.tensor(np.stack(viewmats), dtype=torch.float32, device=device)
    Ks = torch.tensor(np.stack(Ks), dtype=torch.float32, device=device)
    images = torch.tensor(np.stack(images), dtype=torch.float32, device=device)
    C, H, W = images.shape[0], images.shape[1], images.shape[2]
    print(f"加载 {C} 帧，分辨率 {W}x{H}")

    # 可选：物体掩码（损失只在 mask 区域计算；物体不可见的帧不参与训练采样）
    masks = None
    valid_frames = list(range(C))
    if args.mask_dir:
        ms = []
        valid_frames = []
        for i, (_, _, _, name) in enumerate(imgs_meta):
            mp = os.path.join(args.mask_dir, os.path.splitext(name)[0] + ".png")
            m = np.array(Image.open(mp).convert("L")) > 127 if os.path.exists(mp) \
                else np.zeros((H, W), dtype=bool)
            ms.append(m)
            if m.sum() >= 50:
                valid_frames.append(i)
        masks = torch.tensor(np.stack(ms), dtype=torch.float32, device=device).unsqueeze(-1)
        print(f"掩码模式：{args.mask_dir}，物体可见帧 {len(valid_frames)}/{C}")

    # 初始化高斯：优先用 points3D（深度反投影点云），否则围绕场景中心随机
    pc = read_points3D(f"{args.data_dir}/sparse/0/points3D.txt")
    if pc is not None:
        pts, cols = pc
        means = torch.tensor(pts, dtype=torch.float32, device=device)
        N = means.shape[0]
        rgb = torch.tensor(cols, dtype=torch.float32, device=device)
        # 用采样近邻距离估计初始尺度
        sub = means[torch.randint(0, N, (min(2000, N),))]
        d = torch.cdist(sub, means).kthvalue(4, dim=1).values.mean().clamp(min=1.0)
        sh0 = ((rgb - 0.5) / 0.28209).unsqueeze(1)
        print(f"从 points3D 初始化 {N} 个高斯，初始尺度 {d.item():.2f}mm")
    else:
        N = args.n_gaussians
        cam_centers = -torch.bmm(viewmats[:, :3, :3].transpose(1, 2),
                                 viewmats[:, :3, 3:4]).squeeze(-1)
        center = cam_centers.mean(0)
        d = (cam_centers - center).norm(dim=-1).mean() * 1.5
        means = (torch.rand(N, 3, device=device) - 0.5) * 2 * d + center
        sh0 = torch.randn(N, 1, 3, device=device) * 0.1
        print(f"随机初始化 {N} 个高斯")

    scene_scale = float(d) * 10
    params = {
        "means": means,
        "quats": torch.nn.functional.normalize(torch.randn(N, 4, device=device), dim=-1),
        "scales": torch.log(torch.ones(N, 3, device=device) * d),
        "opacities": torch.logit(0.1 * torch.ones(N, device=device)),
        "sh0": sh0,
        "shN": torch.zeros(N, 15, 3, device=device),
    }
    for p in params.values():
        p.requires_grad_(True)
    optimizer = make_optimizer(params)

    os.makedirs(args.out, exist_ok=True)
    sh_deg = 3
    grad_acc = torch.zeros(N, device=device)
    densify_until = int(args.max_steps * 0.7)

    for step in range(args.max_steps):
        idx = valid_frames[torch.randint(0, len(valid_frames), (1,)).item()]
        vm = viewmats[idx:idx+1]; K = Ks[idx:idx+1]; gt = images[idx]

        colors_all = torch.cat([params["sh0"], params["shN"]], dim=1)
        rc, ra, _ = rasterization(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=colors_all,
            viewmats=vm, Ks=K, width=W, height=H,
            packed=False, render_mode="RGB", sh_degree=sh_deg,
        )
        if masks is not None:
            m = masks[idx]
            pixel_loss = (((rc[0] - gt) * m) ** 2).sum() / (m.sum() * 3 + 1e-8)
        else:
            pixel_loss = ((rc[0] - gt) ** 2).mean()
        loss = pixel_loss
        if args.ssim_lambda > 0:
            loss = (1 - args.ssim_lambda) * pixel_loss + args.ssim_lambda * ssim_loss(rc[0], gt)
        optimizer.zero_grad()
        loss.backward()
        if args.densify and params["means"].grad is not None:
            grad_acc += params["means"].grad.norm(dim=-1).detach()
            if len(grad_acc) != len(params["means"]):
                grad_acc = torch.zeros(len(params["means"]), device=device)
        optimizer.step()

        if args.densify and 500 <= step < densify_until and step % args.densify_interval == 0:
            params, optimizer = densify(params, grad_acc, scene_scale)
            grad_acc = torch.zeros(len(params["means"]), device=device)
            print(f"  >> 致密化 @step {step}: 高斯数 {len(params['means'])}")

        if step % 100 == 0:
            print(f"step {step}/{args.max_steps}  loss={loss.item():.5f}")

        if step % 2000 == 0 or step == args.max_steps - 1:
            with torch.no_grad():
                rc, _, _ = rasterization(
                    means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
                    opacities=torch.sigmoid(params["opacities"]),
                    colors=torch.cat([params["sh0"], params["shN"]], dim=1),
                    viewmats=viewmats, Ks=Ks, width=W, height=H,
                    packed=False, render_mode="RGB", sh_degree=sh_deg,
                )
            for i in range(C):
                img = (rc[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                Image.fromarray(img).save(f"{args.out}/render_{i:03d}.png")
            print(f">> 已保存 {C} 张渲染图到 {args.out}/")


if __name__ == "__main__":
    main()
