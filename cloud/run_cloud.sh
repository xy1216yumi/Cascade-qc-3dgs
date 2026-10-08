#!/bin/bash
# ============================================================
# 上云一键运行脚本（SmoothCloud 24GB GPU 实例）
# 用法：把本文件 + bop_to_colmap.py + BOP 数据传到 /root/workspace/
#      然后：bash run_cloud.sh
# ============================================================
set -e

echo "========== 1. 检查 GPU =========="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"

echo "========== 2. 安装 gsplat =========="
pip install gsplat pycolmap -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -5

echo "========== 3. 克隆 gsplat 官方训练示例 =========="
if [ ! -d "gsplat" ]; then
  git clone --depth 1 https://github.com/nerfstudio-project/gsplat.git
fi
cd gsplat

echo "========== 4. BOP → COLMAP 格式转换 =========="
python ../bop_to_colmap.py --src ../real_data/test/000048 --dst ../cloud_data/000048

echo "========== 5. 启动 3DGS 训练（单场景，~20-30 分钟）=========="
python examples/simple_trainer/train.py ../cloud_data/000048 \
    --data_factor 2 \
    --max_steps 7000 \
    --eval_interval 1000 \
    --export_path ../output/000048

echo "========== 完成！渲染视频在 output/000048 =========="
ls -lh ../output/000048/
