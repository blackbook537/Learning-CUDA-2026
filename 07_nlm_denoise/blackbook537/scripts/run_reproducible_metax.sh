#!/usr/bin/env bash
# Reproduce the MetaX MACA acceptance run and keep raw evidence together.
set -euo pipefail
cd "$(dirname "$0")/.."

MACA_HOME="${MACA_HOME:-/opt/maca}"
MACA_CUDA="${MACA_CUDA:-${MACA_HOME}/tools/cu-bridge}"
MXCC="${MXCC:-${MACA_HOME}/mxgpu_llvm/bin/mxcc}"
RESULT_DIR="${RESULT_DIR:-experiments/results/current/metax_c500}"
BASE_WARMUP="${BASE_WARMUP:-3}"
BASE_REPEAT="${BASE_REPEAT:-10}"
CPU_REPEAT="${CPU_REPEAT:-3}"
MATRIX_WARMUP="${MATRIX_WARMUP:-1}"
MATRIX_REPEAT="${MATRIX_REPEAT:-3}"

export PATH="$(dirname "${MXCC}"):${PATH}"
export LD_LIBRARY_PATH="${MACA_HOME}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
mkdir -p "${RESULT_DIR}"

echo "===== [1/8] Environment ====="
PLATFORM=metax MACA_HOME="${MACA_HOME}" MACA_CUDA="${MACA_CUDA}" MXCC="${MXCC}" \
    bash scripts/collect_environment.sh "${RESULT_DIR}/environment.txt"

echo "===== [2/8] Clean build ====="
make clean
make build PLATFORM=metax MACA_HOME="${MACA_HOME}" MACA_CUDA="${MACA_CUDA}" \
    MXCC="${MXCC}" -j"$(nproc)" 2>&1 | tee "${RESULT_DIR}/build.log"

echo "===== [3/8] make test ====="
make test PLATFORM=metax MACA_HOME="${MACA_HOME}" MACA_CUDA="${MACA_CUDA}" \
    MXCC="${MXCC}" 2>&1 | tee "${RESULT_DIR}/make_test.log"

echo "===== [4/8] 1080p RGB correctness ====="
./build/nlm_denoise validate \
    -i data/noisy/noisy_1920x1080_3ch_sigma25.png \
    -o "${RESULT_DIR}/denoised_1920x1080_base_v2.png" \
    -p params/default.txt 2>&1 | tee "${RESULT_DIR}/validate_1920x1080_rgb.log"

echo "===== [5/8] Base 1080p/4K statistics (with 1080p CPU baseline) ====="
./build/nlm_denoise bench --sizes 1920x1080,3840x2160 --channels 3 \
    --param-sets base --warmup "${BASE_WARMUP}" --repeat "${BASE_REPEAT}" \
    --with-cpu --cpu-repeat "${CPU_REPEAT}" --log "${RESULT_DIR}/benchmark_base.csv" \
    2>&1 | tee "${RESULT_DIR}/benchmark_base.log"

echo "===== [6/8] Remaining matrix coverage ====="
./build/nlm_denoise bench --sizes 1920x1080,3840x2160 --channels 1,3 \
    --param-sets small,base,strong-h,large --warmup "${MATRIX_WARMUP}" \
    --repeat "${MATRIX_REPEAT}" --log "${RESULT_DIR}/benchmark_matrix.csv" \
    2>&1 | tee "${RESULT_DIR}/benchmark_matrix.log"

echo "===== [7/8] Quality sweeps ====="
PLATFORM=metax MACA_HOME="${MACA_HOME}" MACA_CUDA="${MACA_CUDA}" MXCC="${MXCC}" \
    RESULT_DIR="${RESULT_DIR}" \
    WORK_DIR="${RESULT_DIR}/quality_images" \
    bash scripts/run_quality_sweep.sh 2>&1 | tee "${RESULT_DIR}/quality_sweep.log"

echo "===== [8/8] MACA native timeline ====="
if test -x "${MACA_HOME}/bin/mcTracer"; then
    mkdir -p "${RESULT_DIR}/mctracer"
    "${MACA_HOME}/bin/mcTracer" --mctx --odname "${RESULT_DIR}/mctracer" \
        --name nlm_1080_v1 ./build/nlm_denoise run \
        -i data/noisy/noisy_1920x1080_3ch_sigma25.png \
        -o /tmp/nlm_mctracer_1080.png -p params/default.txt \
        --kernel 1 --log /dev/null 2>&1 | tee "${RESULT_DIR}/mctracer_1080_v1.log"
else
    echo "mcTracer not found; timeline skipped" | tee "${RESULT_DIR}/mctracer_1080_v1.log"
fi

echo "===== MetaX acceptance PASS ====="
echo "Evidence: ${RESULT_DIR}"
