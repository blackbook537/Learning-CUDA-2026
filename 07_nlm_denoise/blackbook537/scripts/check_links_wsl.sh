#!/usr/bin/env bash
# README 引用路径有效性检查（开发辅助）
set -u
cd "$(dirname "$0")/.."
fail=0
for f in \
    README.md \
    LICENSE \
    experiments/README.md \
    experiments/results/rtx3060_laptop/run_all_legacy.txt \
    experiments/results/rtx4090d/run_all_legacy.txt \
    experiments/results/rtx4090d/benchmark_legacy.csv \
    experiments/results/moore_s4000_musa5.1/README.md \
    experiments/results/moore_s4000_musa5.1/environment_collector_verified.txt \
    experiments/results/moore_s4000_musa5.1/benchmark_1080p_rgb_base.csv \
    experiments/results/moore_s4000_musa5.1/benchmark_4k_rgb_base.csv \
    experiments/results/moore_s4000_musa5.1/benchmark_full_matrix_repeat3.csv \
    experiments/results/metax_c500_maca3.0/README.md \
    experiments/results/metax_c500_maca3.0/environment.txt \
    experiments/results/metax_c500_maca3.0/benchmark_base.csv \
    experiments/results/metax_c500_maca3.0/cpu_baseline_repeat3.csv \
    experiments/results/metax_c500_maca3.0/benchmark_base_gray.csv \
    experiments/results/metax_c500_maca3.0/benchmark_matrix.csv \
    experiments/results/iluvatar_mrv100_corex4.4.0/README.md \
    experiments/results/iluvatar_mrv100_corex4.4.0/environment.txt \
    experiments/results/iluvatar_mrv100_corex4.4.0/benchmark_base.csv \
    experiments/results/iluvatar_mrv100_corex4.4.0/cpu_baseline_repeat3.csv \
    experiments/results/iluvatar_mrv100_corex4.4.0/benchmark_matrix.csv \
    experiments/results/iluvatar_mrv100_corex4.4.0/quality_tradeoff.csv \
    docs/summary_report.md \
    docs/architecture_design.md \
    docs/user_guide.md \
    docs/refactor_log.md \
    docs/experiment_protocol.md \
    params/default.txt \
    experiments/configs/base.txt \
    experiments/configs/small.txt \
    experiments/configs/large.txt \
    data/noisy/noisy_1920x1080_3ch_sigma25.png \
    data/clean/clean_1920x1080_3ch.png \
    scripts/check_dataflow_wsl.sh \
    scripts/collect_environment.sh \
    scripts/run_quality_sweep.sh \
    scripts/run_reproducible_4090.sh \
    scripts/run_reproducible_moore.sh \
    scripts/run_reproducible_metax.sh \
    scripts/run_reproducible_iluvatar.sh \
    scripts/generate_report_assets.py \
    docs/assets/performance_moore_s4000.png \
    docs/assets/performance_metax_c500.png \
    docs/assets/quality_latency_metax_c500.png \
    docs/assets/quality_triptych_metax_c500.png \
    docs/assets/mctracer_timeline_metax_c500.png \
    docs/assets/performance_iluvatar_mrv100.png \
    docs/assets/quality_latency_iluvatar_mrv100.png \
    docs/assets/quality_triptych_iluvatar_mrv100.png \
    scripts/run_all.sh \
    scripts/ncu_profile.sh \
    scripts/nsys_profile.sh \
    scripts/build_wsl.sh \
    Makefile \
    include/nlm/pipeline.h \
    include/nlm/params.h \
    include/kernels/kernels.h \
    include/pal/platform_api.h \
    include/core/image_io.h \
    include/core/nlm_cpu_ref.h \
    include/tester/utils.h \
    kernels/common/kernels_impl.inl \
    kernels/nvidia/kernels.cu \
    kernels/iluvatar/kernels.cu \
    kernels/metax/kernels.maca \
    kernels/moore/kernels.mu \
    src/main.cpp \
    src/pipeline.cpp \
    tester/validate.cpp \
    tester/benchmark.cpp \
    tester/metrics.cpp \
    tester/test_units.cpp \
    third_party/stb/stb_image.h \
    third_party/stb/stb_image_write.h
do
    if [ -f "$f" ]; then
        echo "OK   $f"
    else
        echo "MISS $f"
        fail=1
    fi
done
exit $fail
