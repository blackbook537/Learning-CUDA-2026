#!/usr/bin/env bash
# Complete RTX 4090 acceptance run. No result is considered verified until this exits 0.
set -euo pipefail
cd "$(dirname "$0")/.."

export PLATFORM="${PLATFORM:-nvidia}"
export ARCH="${ARCH:-sm_89}"
export RESULT_DIR="${RESULT_DIR:-experiments/results/current}"
export WORK_DIR="${WORK_DIR:-experiments/work}"
if ! command -v nvcc >/dev/null 2>&1 && test -x /usr/local/cuda/bin/nvcc; then
    export PATH="/usr/local/cuda/bin:${PATH}"
fi
if test -d /usr/local/cuda/lib64; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi
mkdir -p "${RESULT_DIR}" "${WORK_DIR}"

echo "[1/8] Environment"
bash scripts/collect_environment.sh "${RESULT_DIR}/environment.txt"
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "${GPU_NAME}" != *"4090"* ]] && [[ "${ALLOW_NON_4090:-0}" != "1" ]]; then
    echo "ERROR: expected an RTX 4090-class GPU, found: ${GPU_NAME}"
    echo "Set ALLOW_NON_4090=1 only for script smoke tests, never for official 4090 results."
    exit 2
fi

echo "[2/8] Clean build"
make clean
make build PLATFORM="${PLATFORM}" ARCH="${ARCH}" -j"$(nproc)" \
    2>&1 | tee "${RESULT_DIR}/build.log"

echo "[3/8] Unit tests"
./build/nlm_denoise units 2>&1 | tee "${RESULT_DIR}/units.log"

echo "[4/8] 1080p RGB correctness against CPU"
./build/nlm_denoise gen -o "${WORK_DIR}/validate_1080p_rgb_sigma25.png" \
    --size 1920x1080 --channels 3 --sigma 25 --seed 7
./build/nlm_denoise validate \
    -i "${WORK_DIR}/validate_1080p_rgb_sigma25.png" \
    -o "${WORK_DIR}/validate_1080p_rgb_v2.png" \
    -p experiments/configs/base.txt \
    2>&1 | tee "${RESULT_DIR}/validate_1080p_rgb.log"

echo "[5/8] GPU matrix: 1080p/4K x gray/RGB x V0/V1/V2"
./build/nlm_denoise bench \
    --sizes 1920x1080,3840x2160 --channels 1,3 \
    --param-sets small,base,strong-h,large \
    --warmup 3 --repeat 10 \
    --log "${RESULT_DIR}/benchmark.csv" \
    2>&1 | tee "${RESULT_DIR}/benchmark.log"

echo "[6/8] CPU vs V0/V1/V2: 1080p RGB base, identical input/parameters"
./build/nlm_denoise bench \
    --sizes 1920x1080 --channels 3 --param-sets base \
    --warmup 3 --repeat 10 --with-cpu --cpu-repeat 3 \
    --log "${RESULT_DIR}/cpu_baseline.csv" \
    2>&1 | tee "${RESULT_DIR}/cpu_baseline.log"

echo "[7/8] Quality trade-off and sigma sweep"
bash scripts/run_quality_sweep.sh 2>&1 | tee "${RESULT_DIR}/quality.log"

echo "[8/8] Nsight Systems summary"
if command -v nsys >/dev/null 2>&1; then
    nsys profile -f true -o "${WORK_DIR}/nsys_v2_1080p" \
        ./build/nlm_denoise run \
        -i "${WORK_DIR}/validate_1080p_rgb_sigma25.png" \
        -o "${WORK_DIR}/nsys_v2_output.png" \
        -p experiments/configs/base.txt --kernel 2 --log /dev/null
    nsys stats "${WORK_DIR}/nsys_v2_1080p.nsys-rep" \
        --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
        | tee "${RESULT_DIR}/nsys_stats.txt"
else
    echo "ERROR: nsys not found; complete acceptance requires Nsight Systems" \
        | tee "${RESULT_DIR}/nsys_stats.txt"
    exit 2
fi

echo "PASS: complete RTX 4090 acceptance run"
echo "Results: ${RESULT_DIR}"
echo "Use scripts/generate_report_assets.py after copying verified results into rtx4090d/."
