#!/usr/bin/env bash
# 数据集中管理验证：gen 自动命名 / run 输出与日志入 data / validate 数据流
set -euo pipefail
cd "$(dirname "$0")/.."

echo "===== [1] gen（缺省 -o：应自动命名 data/noisy/noisy_256x256_3ch_sigma25.png）====="
./build/nlm_denoise gen --size 256x256 --channels 3 --sigma 25 --seed 7
ls -la data/noisy/ | grep noisy_256 || { echo "FAIL: 自动命名文件不存在"; exit 1; }

echo
echo "===== [2] run（-o data/output/，--log 缺省 data/logs/nlm_perf.log）====="
./build/nlm_denoise run -i data/noisy/noisy_256x256_3ch_sigma25.png \
                         -o data/output/denoised_noisy_256x256_3ch_v2.png \
                         -p params/default.txt --kernel 2
ls -la data/output/ || { echo "FAIL: 输出目录为空"; exit 1; }
tail -1 data/logs/nlm_perf.log

echo
echo "===== [3] validate（新数据流端到端）====="
./build/nlm_denoise validate -i data/noisy/noisy_256x256_3ch_sigma25.png \
                             -o data/output/validated_256.png \
                             -p params/default.txt

echo
echo "===== [4] 确认根目录无新增数据文件 ====="
ls *.png *.log 2>/dev/null && { echo "FAIL: 根目录出现数据文件"; exit 1; } || echo "OK: 根目录干净"

echo
echo "===== 数据流验证全部通过 ====="
