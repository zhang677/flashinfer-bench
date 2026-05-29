#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/accrl_no_cu12
printf 'blocked intentionally for cu13 launch\n' > /tmp/accrl_no_cu12/libcudart.so.12

export CUDA_VISIBLE_DEVICES=4,5
export TVM_FFI_CUDA_ARCH_LIST=10.0a
export LD_LIBRARY_PATH=/tmp/accrl_no_cu12:/raid/user_data/yixind/miniforge3/envs/acc/lib/python3.12/site-packages/nvidia/cu13/lib:/usr/local/cuda/targets/x86_64-linux/lib

exec flashinfer-bench serve --local /var/tmp/accrl-training/ --port 10000 --timeout 30 --config acc_config.yaml
