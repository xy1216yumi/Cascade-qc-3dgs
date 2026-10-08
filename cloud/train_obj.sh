#!/bin/bash
# PSNR 闭环：3 场景 × 3 组 mask（gt/defect/fixed）物体级训练
cd /root/workspace
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=/root/workspace/venv/bin:/usr/local/cuda-12.8/bin:/usr/local/bin:/usr/bin:/bin
export TORCH_CUDA_ARCH_LIST=12.0+PTX
for s in 000048 000051 000056; do
  for v in gt defect fixed; do
    echo "=== $s obj0 $v START $(date +%H:%M:%S) ==="
    venv/bin/python train_gsplat.py --data_dir cloud_data/$s \
      --mask_dir cloud_data/$s/obj0/$v \
      --max_steps 7000 --out output/${s}_obj0_${v} 2>&1 | grep -v HAMI | tail -3
    echo "=== $s obj0 $v DONE ==="
  done
done
echo "=== OBJ LOOP ALL DONE ==="
