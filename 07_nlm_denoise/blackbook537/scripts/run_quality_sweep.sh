#!/usr/bin/env bash
# Reproduce quality experiments on paired deterministic synthetic images.
# The clean/noisy pair always uses the same seed; only noise amplitude changes.
set -euo pipefail
cd "$(dirname "$0")/.."

RESULT_DIR="${RESULT_DIR:-experiments/results/current}"
WORK_DIR="${WORK_DIR:-experiments/work}"
PLATFORM="${PLATFORM:-nvidia}"
ARCH="${ARCH:-sm_89}"
SEED="${SEED:-7}"
SIZE="${SIZE:-1920x1080}"
CHANNELS="${CHANNELS:-3}"

mkdir -p "${RESULT_DIR}" "${WORK_DIR}"
rm -f "${RESULT_DIR}/quality_tradeoff.csv" "${RESULT_DIR}/quality_sigma.csv"

if [[ "${PLATFORM}" == "nvidia" ]]; then
    make build PLATFORM="${PLATFORM}" ARCH="${ARCH}" -j"$(nproc)"
else
    make build PLATFORM="${PLATFORM}" -j"$(nproc)"
fi

CLEAN="${WORK_DIR}/clean_${SIZE}_${CHANNELS}ch.png"
./build/nlm_denoise gen -o "${CLEAN}" --size "${SIZE}" \
    --channels "${CHANNELS}" --sigma 0 --seed "${SEED}"

# A fair small/base/large comparison: same sigma=25 input, h=10, only pr/sr change.
NOISY25="${WORK_DIR}/noisy_${SIZE}_${CHANNELS}ch_sigma25.png"
./build/nlm_denoise gen -o "${NOISY25}" --size "${SIZE}" \
    --channels "${CHANNELS}" --sigma 25 --seed "${SEED}"
./build/nlm_denoise metrics --reference "${CLEAN}" --test "${NOISY25}" \
    --label noisy-sigma25 --log "${RESULT_DIR}/quality_tradeoff.csv"

for NAME in small base large; do
    OUTPUT="${WORK_DIR}/denoised_${NAME}_${SIZE}_${CHANNELS}ch_sigma25.png"
    ./build/nlm_denoise run -i "${NOISY25}" -o "${OUTPUT}" \
        -p "experiments/configs/${NAME}.txt" --kernel 2 --log /dev/null
    ./build/nlm_denoise metrics --reference "${CLEAN}" --test "${OUTPUT}" \
        --label "${NAME}" --log "${RESULT_DIR}/quality_tradeoff.csv"
done

# Noise-strength sweep: parameter sigma matches the generated input amplitude.
for SIGMA in 10 25 50; do
    NOISY="${WORK_DIR}/noisy_${SIZE}_${CHANNELS}ch_sigma${SIGMA}.png"
    OUTPUT="${WORK_DIR}/denoised_base_${SIZE}_${CHANNELS}ch_sigma${SIGMA}.png"
    ./build/nlm_denoise gen -o "${NOISY}" --size "${SIZE}" \
        --channels "${CHANNELS}" --sigma "${SIGMA}" --seed "${SEED}"
    ./build/nlm_denoise metrics --reference "${CLEAN}" --test "${NOISY}" \
        --label "noisy-sigma${SIGMA}" --log "${RESULT_DIR}/quality_sigma.csv"
    ./build/nlm_denoise run -i "${NOISY}" -o "${OUTPUT}" \
        -p "experiments/configs/sigma${SIGMA}.txt" --kernel 2 --log /dev/null
    ./build/nlm_denoise metrics --reference "${CLEAN}" --test "${OUTPUT}" \
        --label "denoised-sigma${SIGMA}" --log "${RESULT_DIR}/quality_sigma.csv"
done

echo "Quality results: ${RESULT_DIR}/quality_tradeoff.csv, ${RESULT_DIR}/quality_sigma.csv"
echo "Generated images: ${WORK_DIR}/"
