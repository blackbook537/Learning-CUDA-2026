#!/usr/bin/env bash
# Reproduce the Iluvatar CoreX acceptance run and keep raw evidence together.
set -euo pipefail
cd "$(dirname "$0")/.."

COREX_HOME="${COREX_HOME:-/usr/local/corex}"
COREX_CXX="${COREX_CXX:-${COREX_HOME}/bin/clang++}"
RESULT_DIR="${RESULT_DIR:-experiments/results/current/iluvatar_mrv100}"
BASE_WARMUP="${BASE_WARMUP:-3}"
BASE_REPEAT="${BASE_REPEAT:-10}"
CPU_REPEAT="${CPU_REPEAT:-3}"
MATRIX_WARMUP="${MATRIX_WARMUP:-1}"
MATRIX_REPEAT="${MATRIX_REPEAT:-3}"

export PATH="${COREX_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${COREX_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${RESULT_DIR}/logs"

echo "===== [1/8] Environment ====="
PLATFORM=iluvatar COREX_HOME="${COREX_HOME}" COREX_CXX="${COREX_CXX}" \
    bash scripts/collect_environment.sh "${RESULT_DIR}/environment.txt"

echo "===== [2/8] Clean build ====="
make clean
make build PLATFORM=iluvatar COREX_HOME="${COREX_HOME}" COREX_CXX="${COREX_CXX}" \
    -j"$(nproc)" 2>&1 | tee "${RESULT_DIR}/logs/build.log"

echo "===== [3/8] Unit tests and verbose GPU validation ====="
make test PLATFORM=iluvatar COREX_HOME="${COREX_HOME}" COREX_CXX="${COREX_CXX}" \
    2>&1 | tee "${RESULT_DIR}/logs/make_test.log"
make test PLATFORM=iluvatar COREX_HOME="${COREX_HOME}" COREX_CXX="${COREX_CXX}" \
    VERBOSE=true 2>&1 | tee "${RESULT_DIR}/logs/make_test_verbose.log"

echo "===== [4/8] 1080p RGB correctness ====="
./build/nlm_denoise validate \
    -i data/noisy/noisy_1920x1080_3ch_sigma25.png \
    -o "${RESULT_DIR}/denoised_1920x1080_base_v2.png" \
    -p params/default.txt 2>&1 | tee "${RESULT_DIR}/logs/validate_1920x1080_rgb.log"

echo "===== [5/8] Base 1080p/4K statistics (with 1080p CPU baseline) ====="
./build/nlm_denoise bench --sizes 1920x1080,3840x2160 --channels 3 \
    --param-sets base --warmup "${BASE_WARMUP}" --repeat "${BASE_REPEAT}" \
    --with-cpu --cpu-repeat "${CPU_REPEAT}" --log "${RESULT_DIR}/benchmark_base.csv" \
    2>&1 | tee "${RESULT_DIR}/logs/benchmark_base.log"

echo "===== [6/8] Full 48-configuration matrix ====="
./build/nlm_denoise bench --sizes 1920x1080,3840x2160 --channels 1,3 \
    --param-sets small,base,strong-h,large --warmup "${MATRIX_WARMUP}" \
    --repeat "${MATRIX_REPEAT}" --log "${RESULT_DIR}/benchmark_matrix.csv" \
    2>&1 | tee "${RESULT_DIR}/logs/benchmark_matrix.log"

echo "===== [7/8] Quality sweeps ====="
PLATFORM=iluvatar COREX_HOME="${COREX_HOME}" COREX_CXX="${COREX_CXX}" \
    RESULT_DIR="${RESULT_DIR}" WORK_DIR="${RESULT_DIR}/quality_images" \
    bash scripts/run_quality_sweep.sh 2>&1 | tee "${RESULT_DIR}/logs/quality_sweep.log"

echo "===== [8/8] Native profiler availability ====="
{
    for profiler in ixprof nsys ncu; do
        if command -v "${profiler}" >/dev/null 2>&1; then
            printf '%s=%s\n' "${profiler}" "$(command -v "${profiler}")"
            "${profiler}" --version 2>&1 || true
        else
            printf '%s=not-found\n' "${profiler}"
        fi
    done
} | tee "${RESULT_DIR}/logs/profiler_availability.log"

echo "===== Iluvatar acceptance PASS ====="
echo "Evidence: ${RESULT_DIR}"
