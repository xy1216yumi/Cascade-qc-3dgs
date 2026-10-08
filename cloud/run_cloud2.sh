#!/bin/bash
# ============================================================
# 上云一键运行脚本 v2（自包含训练，不依赖 gsplat 官方 examples）
# 用法：把本文件 + train_gsplat.py + cloud_data/000048/ 传到 /root/workspace/
#      然后：bash run_cloud2.sh
# ============================================================
set -e

echo "========== 1. 检查 GPU =========="
python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| cuda', torch.version.cuda)"

echo "========== 2. 安装 gsplat（优先预编译 wheel，失败则源码编译）=========="
PT=$(python -c "import torch; v=torch.__version__.split('+')[0].split('.'); print(f'pt{v[0]}{v[1]}')")
CU=$(python -c "import torch; print('cu'+torch.version.cuda.replace('.',''))")
echo "检测到的 wheel 标签: ${PT}${CU}"
if ! pip install gsplat --index-url "https://docs.gsplat.studio/whl/${PT}${CU}" 2>&1 | tail -2; then
  echo "预编译 wheel 不可用，改用源码编译（首次 import 会 JIT 编译，约 10 分钟）..."
  pip install gsplat -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -2
fi

echo "========== 3. 启动 3DGS 训练（75 帧，3 万初始化点）=========="
python train_gsplat.py --data_dir cloud_data/000048 \
    --max_steps 7000 \
    --out output/000048

echo "========== 完成！渲染图在 output/000048/ =========="
ls -lh output/000048/
