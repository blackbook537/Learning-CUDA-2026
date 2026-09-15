#!/usr/bin/env bash
# README 引用路径有效性检查（开发辅助）
set -u
cd "$(dirname "$0")/.."
fail=0
for f in \
    README.md \
    LICENSE \
    nvidia_result.txt \
    nvidia_4090_result.txt \
    metaX_result.txt \
    moore_result.txt \
    docs/summary_report.md \
    docs/architecture_design.md \
    docs/user_guide.md \
    docs/refactor_log.md \
    bench.csv \
    bench_4090.csv \
    params/default.txt \
    data/noisy/noisy.png \
    data/clean/clean_1920x1080_3ch.png \
    data/logs/nlm_perf.log \
    scripts/check_dataflow_wsl.sh \
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
