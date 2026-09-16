#!/usr/bin/env bash
# Reproduce the Moore Threads MUSA acceptance run and keep raw evidence together.
set -euo pipefail
cd "$(dirname "$0")/.."

MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
RESULT_DIR="${RESULT_DIR:-experiments/results/current/moore_s4000}"
BASE_WARMUP="${BASE_WARMUP:-3}"
BASE_REPEAT="${BASE_REPEAT:-10}"
MATRIX_WARMUP="${MATRIX_WARMUP:-1}"
MATRIX_REPEAT="${MATRIX_REPEAT:-3}"

export PATH="${MUSA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${MUSA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${RESULT_DIR}"

echo "===== [1/7] Environment ====="
PLATFORM=moore MUSA_HOME="${MUSA_HOME}" \
    bash scripts/collect_environment.sh "${RESULT_DIR}/environment.txt"

echo "===== [2/7] Clean build ====="
make clean
make build PLATFORM=moore MUSA_HOME="${MUSA_HOME}" -j"$(nproc)" \
    2>&1 | tee "${RESULT_DIR}/build.log"

echo "===== [3/7] make test ====="
make test PLATFORM=moore MUSA_HOME="${MUSA_HOME}" \
    2>&1 | tee "${RESULT_DIR}/make_test.log"

echo "===== [4/7] 1080p RGB correctness ====="
./build/nlm_denoise validate \
    -i data/noisy/noisy_1920x1080_3ch_sigma25.png \
    -o "${RESULT_DIR}/denoised_1920x1080_base_v2.png" \
    -p params/default.txt 2>&1 | tee "${RESULT_DIR}/validate_1920x1080_rgb.log"

echo "===== [5/7] Base 1080p/4K statistics (with 1080p CPU baseline) ====="
./build/nlm_denoise bench --sizes 1920x1080,3840x2160 --channels 3 \
    --param-sets base --warmup "${BASE_WARMUP}" --repeat "${BASE_REPEAT}" \
    --with-cpu --cpu-repeat 1 --log "${RESULT_DIR}/benchmark_base.csv" \
    2>&1 | tee "${RESULT_DIR}/benchmark_base.log"

echo "===== [6/7] Remaining full-matrix coverage ====="
./build/nlm_denoise bench --sizes 1920x1080,3840x2160 --channels 1,3 \
    --param-sets small,strong-h,large --warmup "${MATRIX_WARMUP}" \
    --repeat "${MATRIX_REPEAT}" --log "${RESULT_DIR}/benchmark_matrix.csv" \
    2>&1 | tee "${RESULT_DIR}/benchmark_matrix.log"

echo "===== [7/7] Quality sweeps ====="
PLATFORM=moore MUSA_HOME="${MUSA_HOME}" RESULT_DIR="${RESULT_DIR}" \
    WORK_DIR="${RESULT_DIR}/quality_images" \
    bash scripts/run_quality_sweep.sh 2>&1 | tee "${RESULT_DIR}/quality_sweep.log"

echo "===== Moore Threads acceptance PASS ====="
echo "Evidence: ${RESULT_DIR}"
