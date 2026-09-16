# NVIDIA / Iluvatar / Moore Threads / MetaX 实验复现与 PR 证据手册

本文档定义环境、工具、参数、命令、结果文件和截图口径。目标是让评审者从一个干净
checkout 出发，能够复现构建、正确性、性能、质量与 profiling 结论。

## 1. 证据状态

| 证据 | 当前状态 | 文件 |
|---|---|---|
| 4090 构建、12/12 单测、1080p validate | 已实测 | `experiments/results/rtx4090d/run_all_legacy.txt` |
| 4090 的 48 组性能 | 已实测，旧 schema | `experiments/results/rtx4090d/benchmark_legacy.csv` |
| 4090 CPU/V0/V1/V2 基线 | 已实测一次 | `validation_legacy.csv` |
| 4090 mean/min/stddev 与 CPU 三次统计 | 待用新脚本重采 | `run_reproducible_4090.sh` 将生成 |
| small/base/large 质量权衡 | 3060/S4000/C500/MR-V100 已实测；同一输入公平比较 | `run_quality_sweep.sh` |
| σ=10/25/50 质量扫描 | 3060/S4000/C500/MR-V100 已实测 | `run_quality_sweep.sh` |
| nsys kernel/API 摘要 | 已实测 | `nsys_summary_legacy.csv`、旧全程日志 |
| nsys GUI 时间线截图 | 待从新 `.nsys-rep` 导出 | 见 §8 |
| ncu 计数器 | 宿主限制，未伪造 | 见 §9 |
| MTT S4000 构建、单测、256p/1080p validate | 已实测，全部 PASS | `experiments/results/moore_s4000_musa5.1/` |
| S4000 RGB/base 十次统计与同机 CPU 基线 | 已实测 | `benchmark_1080p_rgb_base.csv`、`benchmark_4k_rgb_base.csv` |
| S4000 完整 48 组合矩阵 | 已实测，warmup=1/repeat=3 | `benchmark_full_matrix_repeat3.csv` |
| S4000 profiler | 镜像没有 profiler CLI，未伪造 | S4000 结果目录 `README.md` |
| C500 构建、单测、256p/1080p validate | 已实测，全部 PASS | `experiments/results/metax_c500_maca3.0/` |
| C500 RGB/base 十次统计与 CPU 三次基线 | 已实测 | `cpu_baseline_repeat3.csv`、`benchmark_base.csv` |
| C500 完整 48 唯一配置 | 已实测 | `benchmark_base_gray.csv`、`benchmark_base.csv`、`benchmark_matrix.csv` |
| C500 MACA 原生时间线 | 已用 mcTracer 实测 | `mctracer/`、`mctracer_1080_v1_events.txt` |
| MR-V100 构建、单测、256p/1080p validate | 已实测，全部 PASS | `experiments/results/iluvatar_mrv100_corex4.4.0/` |
| MR-V100 RGB/base 十次统计与 CPU 三次基线 | 已实测 | `benchmark_base.csv`、`cpu_baseline_repeat3.csv` |
| MR-V100 完整 48 唯一配置 | 已实测，warmup=1/repeat=3 | `benchmark_matrix.csv`、`matrix_integrity.txt` |
| MR-V100 profiler | 镜像没有 ixprof/nsys/ncu CLI，未伪造 | `logs/profiler_availability.log` |

旧数据保留是为了可追溯；新脚本不能凭旧的 min/mean 汇总反推标准差。只有重新采样后，
才将 `experiments/results/current/` 的文件转入 `rtx4090d/`。

## 2. 已验证的 RTX 4090 D 环境

| 项目 | 值 |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 D，compute capability 8.9，24564 MiB |
| 驱动 | 570.124.06 |
| CUDA Toolkit | 12.8.61，`/usr/local/cuda -> cuda-12.8` |
| 编译架构 | `ARCH=sm_89`，必须显式传入 |
| 运行环境 | Linux Docker 容器，128 vCPU |
| Nsight Systems | 2024.6.2.225 |
| Nsight Compute | 2025.1.0.0；宿主设置导致 `ERR_NVGPUCTRPERM` |

Makefile 的 `ARCH` 默认值为空，不要把“nvcc 默认架构”写成“默认 sm_89”。每一份正式
结果都必须由 `collect_environment.sh` 记录当次实际环境和 Git commit。

### 2.1 已验证的 MTT S4000 环境

| 项目 | 值 |
|---|---|
| GPU | MTT S4000，47.91 GB，compute capability 2.2，warp size 128 |
| 驱动 / Runtime | 5.1 / 5.1 |
| MUSA Toolkit / mcc | 5.1.0 / 5.1.0 |
| Host | Ubuntu 22.04.5，g++ 11.4.0，GNU Make 4.3 |
| 构建参数 | `PLATFORM=moore MUSA_HOME=/usr/local/musa` |

### 2.2 已验证的 MetaX C500 环境

| 项目 | 值 |
|---|---|
| GPU | MetaX C500，sGPU id 2，25% compute quota，16,000 MiB |
| 驱动 / MACA | 3.8.30 / 3.0.0.8 |
| MACA SDK / mxcc | `/opt/maca -> /opt/maca-3.0.0` / mxcc 1.0.0 |
| Host | Ubuntu 22.04.3，g++ 11.4.0，GNU Make 4.3 |
| 构建参数 | `PLATFORM=metax MACA_HOME=/opt/maca` |
| Profiler | `/opt/maca/bin/mcTracer` 3.0.0.8 |

### 2.3 已验证的 Iluvatar MR-V100 环境

| 项目 | 值 |
|---|---|
| GPU | Iluvatar MR-V100，32,768 MiB |
| IX-ML / 驱动 | 4.4.0 / 4.4.0 |
| CoreX SDK | `/usr/local/corex -> /usr/local/corex-4.4.0` |
| CoreX clang++ | 18.1.8（CoreX 4.4.0） |
| CUDA 兼容层 | 10.2；nvcc 10.2.89 |
| Host | Ubuntu 24.04.4，g++ 13.3.0，GNU Make 4.3 |
| 构建参数 | `PLATFORM=iluvatar COREX_HOME=/usr/local/corex` |
| Profiler | ixprof/nsys/ncu 均未提供；使用程序内 CUDA Event 分项统计 |

## 3. 工具与依赖

必须：

- NVIDIA 驱动、CUDA Toolkit 11.0+（正式 4090 结果用 CUDA 12.8）；
- Iluvatar 路径需要 CoreX SDK 与对应驱动（正式结果用 MR-V100 / CoreX 4.4.0）；
- Moore 路径需要 MUSA Toolkit 5.1 与对应驱动（正式结果用 MTT S4000）；
- MetaX 路径需要 MACA SDK 3.0 与对应驱动（正式结果用 C500 25% sGPU）；
- GNU Make、支持 C++17 的主机编译器；
- Bash，以及目标平台的设备查询工具（`nvidia-smi` / `ixsmi` / `mx-smi` / `musaInfo`）。

实验/报告工具：

- Nsight Systems：时间线和 CUDA kernel/memory API 汇总；
- Nsight Compute：SOL、Memory Workload、Occupancy（需要宿主授权计数器）；
- MACA mcTracer：MetaX kernel/memcpy 原生时间线 JSON；
- Python 3 + Pillow：只负责把已有 CSV/图像渲染为 PNG，不参与 NLM 运算。

安装报告依赖：

```bash
python3 -m pip install -r requirements-report.txt
```

## 4. 从干净 checkout 开始

在 `07_nlm_denoise/blackbook537/` 目录执行：

```bash
git status --short
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

ARCH=sm_89 bash scripts/collect_environment.sh \
  experiments/results/current/environment.txt

make clean
make build PLATFORM=nvidia ARCH=sm_89 -j"$(nproc)"
./build/nlm_denoise units
```

验收条件：构建零 error；正式提交前处理所有新增 warning；单测必须显示 `12/12 PASS`。

Moore Threads 从干净 checkout 执行：

```bash
MUSA_HOME=/usr/local/musa
PLATFORM=moore MUSA_HOME="$MUSA_HOME" \
  bash scripts/collect_environment.sh \
  experiments/results/current/moore/environment.txt

make clean
make build PLATFORM=moore MUSA_HOME="$MUSA_HOME" -j"$(nproc)"
make test PLATFORM=moore MUSA_HOME="$MUSA_HOME"
readelf -d build/nlm_denoise | grep -E 'RPATH|RUNPATH'
```

默认 Makefile 直接调用 `${MUSA_HOME}/bin/mcc` 并写入 `${MUSA_HOME}/lib` RUNPATH，
因此不要求用户预先导出 PATH/LD_LIBRARY_PATH。正式验收继续执行 §5–7 的同一 CLI，
或直接用 §10 的一键脚本。

MetaX 从干净 checkout 执行：

```bash
MACA_HOME=/opt/maca
PLATFORM=metax MACA_HOME="$MACA_HOME" \
  bash scripts/collect_environment.sh \
  experiments/results/current/metax/environment.txt

make clean
make build PLATFORM=metax MACA_HOME="$MACA_HOME" -j"$(nproc)"
make test PLATFORM=metax MACA_HOME="$MACA_HOME"
readelf -d build/nlm_denoise | grep -E 'RPATH|RUNPATH'
```

默认布局为 `MACA_CUDA=${MACA_HOME}/tools/cu-bridge`、
`MXCC=${MACA_HOME}/mxgpu_llvm/bin/mxcc`；Makefile 显式链接 `libruntime_cu` 与
`libsymbol_cu`，并把 `${MACA_HOME}/lib` 写入 RUNPATH。

Iluvatar 从干净 checkout 执行：

```bash
COREX_HOME=/usr/local/corex
PLATFORM=iluvatar COREX_HOME="$COREX_HOME" \
  bash scripts/collect_environment.sh \
  experiments/results/current/iluvatar/environment.txt

make clean
make build PLATFORM=iluvatar COREX_HOME="$COREX_HOME" -j"$(nproc)"
make test PLATFORM=iluvatar COREX_HOME="$COREX_HOME" VERBOSE=true
readelf -d build/nlm_denoise | grep -E 'RPATH|RUNPATH'
```

默认使用 `${COREX_HOME}/bin/clang++`，设备编译增加
`-x ivcore --cuda-path=${COREX_HOME}`，并把 `${COREX_HOME}/lib64` 写入 RUNPATH。

## 5. 1080p RGB 正确性与统一 CPU 基线

正确性校验会对同一输入依次运行 CPU、V0、V1、V2：

```bash
mkdir -p experiments/work experiments/results/current

./build/nlm_denoise gen \
  -o experiments/work/validate_1080p_rgb_sigma25.png \
  --size 1920x1080 --channels 3 --sigma 25 --seed 7

./build/nlm_denoise validate \
  -i experiments/work/validate_1080p_rgb_sigma25.png \
  -o experiments/work/validate_1080p_rgb_v2.png \
  -p experiments/configs/base.txt \
  | tee experiments/results/current/validate_1080p_rgb.log
```

判定：V0/V1/V2 对 CPU 参考均须 PASS。CPU 加速比表用新的统计 benchmark 重新采集：

```bash
./build/nlm_denoise bench \
  --sizes 1920x1080 --channels 3 --param-sets base \
  --warmup 3 --repeat 10 --with-cpu --cpu-repeat 3 \
  --log experiments/results/current/cpu_baseline.csv \
  | tee experiments/results/current/cpu_baseline.log
```

`cpu_baseline.csv` 对 CPU 与三个 GPU 版本使用相同 seed、输入、pr/sr/h/σ。输出含采样次数、
mean、min、标准差和 `CPU mean / GPU e2e mean` 加速比，不再出现旧 CSV 中的 `-1`。
CPU 单线程每次约需 175 秒，三次采样预计约 9 分钟。

## 6. 1080p/4K 性能矩阵

```bash
./build/nlm_denoise bench \
  --sizes 1920x1080,3840x2160 --channels 1,3 \
  --param-sets small,base,strong-h,large \
  --warmup 3 --repeat 10 \
  --log experiments/results/current/benchmark.csv \
  | tee experiments/results/current/benchmark.log
```

参数定义：

| 名称 | pr | sr | h | σ | 用途 |
|---|---:|---:|---:|---:|---|
| small | 2 | 7 | 10 | 25 | 速度优先 |
| base | 3 | 10 | 10 | 25 | 任务基线 |
| strong-h | 3 | 10 | 15 | 25 | 同计算量增强平滑 |
| large | 4 | 14 | 10 | 25 | 大搜索窗压力测试 |

small/base/large 固定 h=10、σ=25，只改变 pr/sr，保证质量—延迟对比没有混入噪声强度
变化。正式表至少报告 `kernel/e2e mean、min、stddev、repeat、Mpx/s`。

## 7. 图像质量实验

```bash
ARCH=sm_89 RESULT_DIR=experiments/results/current \
  bash scripts/run_quality_sweep.sh

# Moore Threads
PLATFORM=moore MUSA_HOME=/usr/local/musa \
  RESULT_DIR=experiments/results/current/moore \
  bash scripts/run_quality_sweep.sh

# MetaX
PLATFORM=metax MACA_HOME=/opt/maca \
  RESULT_DIR=experiments/results/current/metax \
  bash scripts/run_quality_sweep.sh

# Iluvatar
PLATFORM=iluvatar COREX_HOME=/usr/local/corex \
  RESULT_DIR=experiments/results/current/iluvatar \
  bash scripts/run_quality_sweep.sh
```

该脚本生成两类配对实验：

1. `quality_tradeoff.csv`：同一张 σ=25、seed=7 输入，比较 noisy/small/base/large；
2. `quality_sigma.csv`：σ=10/25/50，输入噪声幅度与参数文件中的 sigma 一致。

每行通过 `metrics --reference ... --test ...` 计算 MAE/PSNR。合成图才有 ground truth；
真实照片只能提交视觉对比并注明“无 GT”，不得给出虚构 PSNR。

## 8. Nsight Systems 与截图

完整脚本会采集 V2 1080p：

```bash
nsys profile -f true -o experiments/work/nsys_v2_1080p \
  ./build/nlm_denoise run \
  -i experiments/work/validate_1080p_rgb_sigma25.png \
  -o experiments/work/nsys_v2_output.png \
  -p experiments/configs/base.txt --kernel 2 --log /dev/null

nsys stats experiments/work/nsys_v2_1080p.nsys-rep \
  --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
  | tee experiments/results/current/nsys_stats.txt
```

GUI 截图步骤：用 Nsight Systems 打开 `nsys_v2_1080p.nsys-rep`，展开 CUDA HW / CUDA API，
缩放到一次 `NlmSmemUnrollKernel<3>`，截图同时包含时间标尺、H2D、NLM、D2H，并在图注中
写明设备、输入尺寸、参数与 kernel 版本。`.nsys-rep` 较大且已忽略；提交压缩 PNG 和
`nsys_stats.txt` 即可。

MetaX 不能使用 Nsight 替代原生工具。在已验证的 MACA 3.0 镜像中执行：

```bash
mkdir -p experiments/results/current/metax/mctracer
/opt/maca/bin/mcTracer --mctx \
  --odname experiments/results/current/metax/mctracer \
  --name nlm_1080_v1 \
  ./build/nlm_denoise run \
  -i data/noisy/noisy_1920x1080_3ch_sigma25.png \
  -o /tmp/nlm_mctracer_1080.png -p params/default.txt \
  --kernel 1 --log /dev/null
```

mcTracer 输出 Chrome trace JSON。正式性能仍用未插桩 benchmark；trace 只用于分解
H2D、转换 kernel、NLM 和 D2H，占比图由 `generate_report_assets.py --trace ...`
生成。本次 V1 NLM 占求和设备事件 99.617%。

## 9. Nsight Compute

```bash
bash scripts/ncu_profile.sh
```

若出现 `ERR_NVGPUCTRPERM`，保留完整错误并注明宿主
`RmProfilingAdminOnly=1`。只有管理员在宿主设置
`NVreg_RestrictProfilingToAdminUsers=0`，或提供带 `SYS_ADMIN` 权限的容器后才能采集。
不能用静态估算冒充 ncu 计数器实测。

## 10. 一键正式验收

```bash
ARCH=sm_89 RESULT_DIR=experiments/results/current \
  bash scripts/run_reproducible_4090.sh
```

脚本退出码为 0 且末行出现 `PASS: complete RTX 4090 acceptance run` 后，检查环境、CSV、
日志中的设备和 Git commit，再把 `current/` 中的正式结果复制到 `rtx4090d/`。不要直接
覆盖历史结果；建议在文件名中加入日期或同时记录 commit。

Moore Threads 使用独立入口：

```bash
MUSA_HOME=/usr/local/musa \
RESULT_DIR=experiments/results/current/moore_s4000 \
bash scripts/run_reproducible_moore.sh
```

该脚本依次采集环境、干净构建、`make test`、1080p CPU 对照、1080p/4K base 十次
统计、其余参数矩阵和质量扫描。默认运行时间较长，可仅在冒烟阶段通过
`BASE_REPEAT`/`MATRIX_REPEAT` 调低重复数；正式表保持默认值。

MetaX 使用独立入口：

```bash
MACA_HOME=/opt/maca \
RESULT_DIR=experiments/results/current/metax_c500 \
bash scripts/run_reproducible_metax.sh
```

该脚本按同样顺序采集环境、干净构建、单测、1080p CPU 对照、base 正式统计、
其余矩阵和质量扫描。C500 上推荐业务运行 `--kernel 1`，但验收仍必须测试三版本。

Iluvatar 使用独立入口：

```bash
COREX_HOME=/usr/local/corex \
RESULT_DIR=experiments/results/current/iluvatar_mrv100 \
bash scripts/run_reproducible_iluvatar.sh
```

脚本执行同一验收口径，并额外保存 ixprof/nsys/ncu 可用性探测。正式 MR-V100
结果中三者均不存在，因此不要求也不伪造天数平台时间线。

## 11. 生成 PR 图表

完成性能、质量实验后：

```bash
python3 scripts/generate_report_assets.py \
  --benchmark experiments/results/rtx4090d/benchmark.csv \
  --quality experiments/results/rtx4090d/quality_tradeoff.csv \
  --quality-benchmark experiments/results/rtx4090d/benchmark.csv \
  --quality-device "RTX 4090 D"

# 只生成 S4000 的 1080p/4K base 图
python3 scripts/generate_report_assets.py --performance-only \
  --benchmark experiments/results/moore_s4000_musa5.1/benchmark_1080p_rgb_base.csv \
  --benchmark experiments/results/moore_s4000_musa5.1/benchmark_4k_rgb_base.csv \
  --performance-device "MTT S4000 / MUSA 5.1" \
  --performance-output performance_moore_s4000.png

# 生成 C500 的性能、质量和 mcTracer 时间线图
python3 scripts/generate_report_assets.py --experiment-only \
  --benchmark experiments/results/metax_c500_maca3.0/benchmark_base.csv \
  --benchmark experiments/results/metax_c500_maca3.0/cpu_baseline_repeat3.csv \
  --performance-device "MetaX C500 (25% sGPU)" \
  --performance-output performance_metax_c500.png \
  --quality experiments/results/metax_c500_maca3.0/quality_tradeoff.csv \
  --quality-benchmark experiments/results/metax_c500_maca3.0/benchmark_base.csv \
  --quality-benchmark experiments/results/metax_c500_maca3.0/benchmark_matrix.csv \
  --quality-benchmark experiments/results/metax_c500_maca3.0/cpu_baseline_repeat3.csv \
  --quality-device "MetaX C500 (25% sGPU)" \
  --clean experiments/results/metax_c500_maca3.0/quality_images/clean_1920x1080_3ch.png \
  --noisy experiments/results/metax_c500_maca3.0/quality_images/noisy_1920x1080_3ch_sigma25.png \
  --denoised experiments/results/metax_c500_maca3.0/quality_images/denoised_base_1920x1080_3ch_sigma25.png \
  --triptych-output quality_triptych_metax_c500.png \
  --quality-output quality_latency_metax_c500.png \
  --trace experiments/results/metax_c500_maca3.0/mctracer/nlm_1080_v1-3213.json \
  --trace-output mctracer_timeline_metax_c500.png

# 生成 MR-V100 的性能、质量和三联图（该镜像无原生 trace）
python3 scripts/generate_report_assets.py --experiment-only \
  --benchmark experiments/results/iluvatar_mrv100_corex4.4.0/benchmark_base.csv \
  --performance-device "Iluvatar MR-V100 / CoreX 4.4.0" \
  --performance-output performance_iluvatar_mrv100.png \
  --quality experiments/results/iluvatar_mrv100_corex4.4.0/quality_tradeoff.csv \
  --quality-benchmark experiments/results/iluvatar_mrv100_corex4.4.0/benchmark_matrix.csv \
  --quality-benchmark experiments/results/iluvatar_mrv100_corex4.4.0/benchmark_base.csv \
  --quality-device "Iluvatar MR-V100" \
  --clean experiments/results/iluvatar_mrv100_corex4.4.0/quality_images/clean_1920x1080_3ch.png \
  --noisy experiments/results/iluvatar_mrv100_corex4.4.0/quality_images/noisy_1920x1080_3ch_sigma25.png \
  --denoised experiments/results/iluvatar_mrv100_corex4.4.0/quality_images/denoised_base_1920x1080_3ch_sigma25.png \
  --triptych-output quality_triptych_iluvatar_mrv100.png \
  --quality-output quality_latency_iluvatar_mrv100.png
```

生成至 `docs/assets/`：

- `quality_triptych.png`：clean/noisy/denoised + 4× ROI；
- `performance_4090.png`：1080p/4K V0/V1/V2 kernel 与 e2e；
- `quality_latency_tradeoff.png`：small/base/large 的 PSNR—延迟；
- `nsys_kernel_summary.png`：nsys kernel 时间摘要；
- `validation_summary.png`：12/12、CPU/V0/V1/V2 与 PASS 状态。
- `performance_metax_c500.png` / `quality_latency_metax_c500.png`：C500 性能与质量权衡；
- `mctracer_timeline_metax_c500.png`：C500 原生设备时间线。

图表脚本遇到缺失数据会明确跳过并返回非零，不会用占位数生成图片。

## 12. PR 前检查

```bash
git diff --check
git status --short
git diff --stat upstream/2026-summer-project...HEAD
```

确认没有 `build/`、`experiments/work/`、`.nsys-rep`、`.ncu-rep` 和临时日志；文档中的
图片与链接能从 GitHub 页面直接打开。PR 中将“已实测”和“脚本就绪/待授权”分开陈述。
