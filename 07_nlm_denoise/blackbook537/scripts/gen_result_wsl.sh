#!/usr/bin/env bash
# 重构后全量验证 + 生成 nvidia_result.txt（真实运行输出）：
#   1) make run VERBOSE=true   —— 验证新 Makefile 目标语义（units + GPU 端到端校验）
#   2) validate 512x512 RGB    —— 三版本 vs CPU 参考 MAE/PSNR
#   3) bench 快速抽样          —— 确认基准链路正常（完整数据见 bench.csv）
set -uo pipefail
cd "$(dirname "$0")/.."

OUT=nvidia_result.txt
{
echo "==============================================================="
echo " nlm_denoise —— NVIDIA 平台实测结果（nvidia_result.txt）"
echo " 生成方式: bash scripts/gen_result_wsl.sh（可复现）"
echo "==============================================================="
echo
echo "[环境]"
echo "GPU     : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"
echo "CUDA    : $($HOME/cuda/cuda_nvcc-linux-x86_64-12.9.86-archive/bin/nvcc --version 2>/dev/null | grep -o 'release [0-9.]*' | head -1)"
echo "OS      : $(uname -sr)"
echo "参数基线: pr=3 sr=10 h=10.0 sigma=25.0"
echo
echo "==============================================================="
echo "[1] make run VERBOSE=true（单元测试 + GPU 端到端校验 256x256）"
echo "==============================================================="
make run VERBOSE=true \
  NVCC=$HOME/cuda/cuda_nvcc-linux-x86_64-12.9.86-archive/bin/nvcc \
  CXX_HOST=$HOME/micromamba/envs/cxx/bin/x86_64-conda-linux-gnu-g++ \
  CUDA_RT=$HOME/.local/lib/python3.14/site-packages/nvidia/cuda_runtime \
  ARCH=sm_86 2>&1
echo
echo "==============================================================="
echo "[2] 正确性校验 validate（512x512 RGB，三版本 vs CPU 参考）"
echo "==============================================================="
./build/nlm_denoise gen -o /tmp/nlm_res_512.png --size 512x512 --channels 3 --sigma 25
./build/nlm_denoise validate -i /tmp/nlm_res_512.png -p params/default.txt
echo
echo "==============================================================="
echo "[3] 性能基准抽样（640x360，完整 48 组数据见 bench.csv）"
echo "==============================================================="
./build/nlm_denoise bench --sizes 640x360 --channels 1,3 --warmup 1 --repeat 3 --log /tmp/nlm_res_bench.csv
echo
echo "----- bench.csv 摘要（RTX 3060 Laptop 实测，warmup=3 repeat=10）-----"
head -1 bench.csv
grep -E '^1920x1080,(1|3),3,10,10,25,' bench.csv
grep -E '^3840x2160,(1|3),3,10,10,25,' bench.csv
} | tee "$OUT"

echo "已生成 $OUT"
