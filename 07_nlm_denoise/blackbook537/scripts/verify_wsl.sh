#!/usr/bin/env bash
# 修正 V2 分派后的回归验证：
#   1) large 参数(pr=4,sr=14)下 V2 性能应与 V1 持平（不再回退到展开路径）
#   2) 基线参数正确性（512x512 快速 validate，三版本 MAE 应仍为 0）
set -euo pipefail
cd "$(dirname "$0")/.."

cat > /tmp/large.txt <<'EOF'
patch_radius = 4
search_radius = 14
h = 12.0
sigma = 35.0
EOF

./build/nlm_denoise gen -o /tmp/chk.png --size 1920x1080 --channels 3 --sigma 35

echo "===== large 参数 pr=4 sr=14 ====="
./build/nlm_denoise run -i /tmp/chk.png -o /tmp/chk_o1.png -p /tmp/large.txt --kernel 1 --log /dev/null
./build/nlm_denoise run -i /tmp/chk.png -o /tmp/chk_o2.png -p /tmp/large.txt --kernel 2 --log /dev/null

echo "===== 基线参数正确性（512x512 RGB）====="
./build/nlm_denoise gen -o /tmp/val512.png --size 512x512 --channels 3 --sigma 25
./build/nlm_denoise validate -i /tmp/val512.png -p params/default.txt

echo "===== 单元测试 ====="
./build/nlm_denoise units
