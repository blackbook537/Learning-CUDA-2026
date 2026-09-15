#!/usr/bin/env bash
# 重构后 Makefile 目标图验证（dry-run，不实际编译）
set -u
cd "$(dirname "$0")/.."

echo "===== make -n（默认 all = build + run）====="
make -n 2>&1 | grep -vE '^make(\[|:)' | head -6
echo
echo "===== make -n run VERBOSE=true（verbose 分支）====="
make -n run VERBOSE=true 2>&1 | grep -E 'nlm_denoise (units|gen|validate)|Verbose mode'
echo
echo "===== make -n clean ====="
make -n clean 2>&1 | tail -1
echo
echo "===== make -n build PLATFORM=moore（平台后缀切换）====="
make -n build PLATFORM=moore 2>&1 | grep -o 'kernels/moore/kernels.mu' | head -1
