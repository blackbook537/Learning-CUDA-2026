#!/usr/bin/env bash
# FAST_EXP（__expf 近似）误差来源分析：
#   构建 -DNLM_FAST_EXP 变体，与精确版本在相同输入上对比 MAE/PSNR。
set -euo pipefail
cd "$(dirname "$0")/.."

EXTRA_DEFS="-DNLM_FAST_EXP" OUTPUT=build/nlm_denoise_fastexp bash scripts/build_wsl.sh

# /tmp 为 tmpfs（WSL 重启即清空），测试图一律现场生成
./build/nlm_denoise gen -o /tmp/val512.png --size 512x512 --channels 3 --sigma 25
./build/nlm_denoise gen -o /tmp/val_1080.png --size 1920x1080 --channels 3 --sigma 25

echo "===== FAST_EXP 变体正确性（512x512 RGB，vs CPU 精确参考）====="
./build/nlm_denoise_fastexp validate -i /tmp/val512.png -p params/default.txt

echo "===== FAST_EXP vs 精确版 kernel 耗时对比（1080p RGB 基线）====="
./build/nlm_denoise run -i /tmp/val_1080.png -o /tmp/fe_out.png -p params/default.txt --kernel 2 --log /dev/null
./build/nlm_denoise_fastexp run -i /tmp/val_1080.png -o /tmp/fe_out2.png -p params/default.txt --kernel 2 --log /dev/null
