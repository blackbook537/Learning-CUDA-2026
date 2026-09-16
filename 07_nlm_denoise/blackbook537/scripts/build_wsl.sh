#!/usr/bin/env bash
# 本地开发机构建脚本（WSL 无系统级 CUDA 时使用）
# 使用 NVIDIA redist 版 nvcc + pip 版 cudart 编译项目。
# 训练营服务器有完整 CUDA Toolkit 时直接用根目录 Makefile 即可，无需本脚本。
set -euo pipefail
cd "$(dirname "$0")/.."

NVCC="${NVCC:-/home/zhoufei/cuda/cuda_nvcc-linux-x86_64-12.9.86-archive/bin/nvcc}"
RT="${CUDA_RT:-/home/zhoufei/.local/lib/python3.14/site-packages/nvidia/cuda_runtime}"
ARCH="${ARCH:-sm_86}"
# 系统 gcc 15 与 nvcc 12.9 前端不兼容，使用 conda 用户空间的 gcc 13
CXX="${CXX:-/home/zhoufei/micromamba/envs/cxx/bin/x86_64-conda-linux-gnu-g++}"

OUT="${OUTPUT:-build/nlm_denoise}"
mkdir -p "$(dirname "${OUT}")"

"$NVCC" -std=c++17 -O3 -arch="$ARCH" -ccbin "$CXX" \
    -DPLATFORM_NVIDIA ${EXTRA_DEFS:-} -Iinclude -Ithird_party/stb -I"$RT/include" \
    src/main.cpp src/params.cpp src/image_io.cpp src/pipeline.cpp \
    src/nlm_cpu_ref.cpp tester/validate.cpp tester/benchmark.cpp \
    tester/metrics.cpp tester/test_units.cpp kernels/nvidia/kernels.cu \
    -o "${OUT}" -L"$RT/lib" -lcudart_static -lpthread -ldl -lrt

echo "构建完成: ${OUT}"
