#!/usr/bin/env bash
# 一键准入检查：构建 -> 单元测试 -> 正确性校验 -> 性能基准
# 用法: PLATFORM=nvidia bash scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PLATFORM="${PLATFORM:-nvidia}"
echo "===== [1/4] 构建 (PLATFORM=${PLATFORM}) ====="
make build PLATFORM="${PLATFORM}" -j"$(nproc)"

echo "===== [2/4] 单元测试 ====="
./build/nlm_denoise units

echo "===== [3/4] 正确性校验（1080p RGB 合成含噪图，任务基线参数）====="
./build/nlm_denoise gen -o /tmp/nlm_val_1080.png --size 1920x1080 --channels 3 --sigma 25
./build/nlm_denoise validate -i /tmp/nlm_val_1080.png -o /tmp/nlm_out_1080.png -p params/default.txt

echo "===== [4/4] 性能基准（1080p + 4K，灰度 + RGB）====="
./build/nlm_denoise bench --sizes 1920x1080,3840x2160 --channels 1,3 \
    --warmup 3 --repeat 10 --log bench.csv

echo "===== 全部通过，性能日志: bench.csv ====="
