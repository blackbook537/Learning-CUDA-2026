#!/usr/bin/env bash
# 本机 WSL 开发环境：用 Makefile + 变量覆盖完成构建（等价 scripts/build_wsl.sh，
# 但走标准 make 流程，逐文件编译、增量构建）。
set -euo pipefail
cd "$(dirname "$0")/.."

make clean >/dev/null 2>&1 || true
make build \
  NVCC=/home/zhoufei/cuda/cuda_nvcc-linux-x86_64-12.9.86-archive/bin/nvcc \
  CXX_HOST=/home/zhoufei/micromamba/envs/cxx/bin/x86_64-conda-linux-gnu-g++ \
  CUDA_RT=/home/zhoufei/.local/lib/python3.14/site-packages/nvidia/cuda_runtime \
  ARCH=sm_86 \
  -j"$(nproc)"

./build/nlm_denoise units
