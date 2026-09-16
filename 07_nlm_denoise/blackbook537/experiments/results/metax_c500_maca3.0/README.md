# MetaX C500 / MACA 3.0 实测结果

本目录保存 2026-09-15 在曦云实例上生成的原始 CSV、文本日志、输出图像与
`mcTracer` JSON。实例实际硬件为 **MetaX C500 的 25% sGPU 切片**，不是
Iluvatar/CoreX；以下结论仅适用于本次 16 GB、25% compute quota 的切片配置。

## 环境

| 项目 | 实测值 |
|---|---|
| GPU | MetaX C500，sGPU id 2，25% compute quota，16,000 MiB 配额 |
| 驱动 / MACA | Kernel Mode Driver 3.8.30 / MACA 3.0.0.8 |
| MACA SDK | `/opt/maca -> /opt/maca-3.0.0` |
| 编译器 | mxcc 1.0.0 (`df29922f9c`) |
| OS | Ubuntu 22.04.3 LTS，Linux 5.15 |
| Host 编译器 | g++ 11.4.0；GNU Make 4.3 |
| 构建参数 | `PLATFORM=metax FAST_EXP=0 WITH_OPENCV=0` |

完整设备状态与工具链路径见 `environment.txt`。测试使用当前未提交工作树的上传快照，
因此环境清单如实标为 `git_commit=unavailable`；归档哈希及包内文件哈希均已核验。

## 验收结果

| 检查 | 结果 | 证据 |
|---|---|---|
| 原始 Makefile 干净构建 | FAIL：设备编译缺 `cuda_runtime.h` | `build_original_failure.log` |
| 增加 cu-bridge include 后链接 | FAIL：缺 `wcuda*` 实现 | `build_with_sdk_include.log` |
| 修正版干净构建 | PASS | `build.log`、`build_with_sdk_link.log` |
| 动态库与 RUNPATH | PASS：所有 MACA 库可解析 | `binary_runtime.txt` |
| 默认 / verbose `make test PLATFORM=metax` | PASS；12/12 单测，verbose 附加 GPU 校验 | `make_test.log`、`make_test_verbose.log` |
| 256×256 RGB，V0/V1/V2 对 CPU | PASS，三版本 MAE=0、PSNR=∞ | `make_test_verbose.log` |
| 1920×1080 RGB，V0/V1/V2 对 CPU | PASS，三版本 MAE=0、PSNR=∞ | `validate_1920x1080_rgb.log` |
| RGB/base 1080p/4K，warmup=3/repeat=10 | PASS | `benchmark_base.csv` |
| CPU 三次独立基线 + GPU 十次统计 | PASS | `cpu_baseline_repeat3.csv` |
| 完整 48 唯一配置 | PASS；灰度/base 6 行 + RGB/base 6 行 + 其余 36 行 | `benchmark_base_gray.csv`、`benchmark_base.csv`、`benchmark_matrix.csv` |
| small/base/large 与 σ=10/25/50 质量扫描 | PASS | `quality_*.csv`、`quality_sweep.log` |
| MACA 原生时间线 | PASS，mcTracer 3.0.0.8 JSON | `mctracer/`、`mctracer_1080_v1_events.txt` |

1080p 正确性校验的 CPU 参考耗时为 179,382.8 ms；V0/V1/V2 单次 kernel
耗时为 798.987 / 418.401 / 1487.083 ms，量化输出均与 CPU 逐像素一致。
V1 与 V2 输出 PNG 的 SHA-256 也完全相同，见 `binary_runtime.txt`。

## 正式基准：RGB/base

以下采用 warmup=3、repeat=10；`mean ± stddev (min)` 单位均为 ms。1080p CPU
基线独立测量三次，为 179,598.341 ± 100.088 (179,467.824) ms；4K 按安全约定
不运行 CPU 参考。1080p 采用 `cpu_baseline_repeat3.csv`，4K 采用 `benchmark_base.csv`。

| 尺寸 | 版本 | kernel mean ± stddev (min) | e2e mean ± stddev (min) | Mpx/s | CPU/e2e |
|---|---|---:|---:|---:|---:|
| 1920×1080 | V0 | 795.540 ± 1.129 (793.667) | 797.894 ± 1.162 (795.943) | 2.607 | 225.09× |
| 1920×1080 | V1 | **418.397 ± 0.014 (418.380)** | **420.733 ± 0.049 (420.654)** | **4.956** | **426.87×** |
| 1920×1080 | V2 | 1488.151 ± 1.161 (1486.245) | 1490.697 ± 1.160 (1489.131) | 1.393 | 120.48× |
| 3840×2160 | V0 | 3114.038 ± 0.895 (3112.780) | 3123.608 ± 0.814 (3122.292) | 2.664 | n.a. |
| 3840×2160 | V1 | **1645.636 ± 0.018 (1645.616)** | **1655.068 ± 0.107 (1654.861)** | **5.040** | n.a. |
| 3840×2160 | V2 | 5894.479 ± 0.500 (5893.784) | 5904.009 ± 0.596 (5903.218) | 1.407 | n.a. |

MetaX 上 V1 相对 V0 的 kernel 加速为 1080p **1.90×**、4K **1.89×**；
V2 则比 V1 慢 3.55–3.58×。原因不是结果错误，而是 `<3>` 模板全展开在该
MACA 编译器/架构上产生了不利代码。应用在 MetaX 上应明确选择 `--kernel 1`；
不能把 NVIDIA 或 Moore 上 V2 更快的结论外推到 C500。

## 质量与延迟

同一 σ=25、seed=7 的 1080p RGB 合成输入，质量由确定性输出计算。延迟为 V2；
small/large 来自 warmup=1/repeat=3，base 来自 warmup=3/repeat=10。

| 配置 | pr/sr | MAE vs clean | PSNR | V2 e2e mean ± stddev |
|---|---|---:|---:|---:|
| noisy | — | 12.485684 | 24.947823 dB | — |
| small | 2/7 | 0.803424 | 47.358490 dB | 107.777 ± 0.055 ms |
| base | 3/10 | 0.597227 | 49.414889 dB | 1489.555 ± 0.650 ms |
| large | 4/14 | 0.463451 | 50.948416 dB | 2359.575 ± 0.990 ms |

| σ | noisy PSNR | denoised PSNR | 改善 |
|---:|---:|---:|---:|
| 10 | 32.883942 dB | 53.265898 dB | +20.381956 dB |
| 25 | 24.947823 dB | 49.414889 dB | +24.467066 dB |
| 50 | 18.960290 dB | 43.964257 dB | +25.003967 dB |

## mcTracer 时间线

镜像未提供 NVIDIA `nsys`/`ncu`，但提供 MACA 原生 `mcTracer 3.0.0.8`。对
1080p RGB/base/V1 的原生 trace 显示设备事件跨度 427.839 ms，求和 423.851 ms：

| 设备事件 | 时间 | 占设备事件总时长 |
|---|---:|---:|
| H2D | 0.006656 ms | 0.002% |
| U8→F32 | 0.235520 ms | 0.056% |
| NLM V1 | 422.226432 ms | **99.617%** |
| F32→U8 | 0.549376 ms | 0.130% |
| D2H | 0.833024 ms | 0.197% |

这证明该配置的设备侧瓶颈仍在 NLM 主 kernel。`mcTracer` 会显著放大 Host/首次分配
开销，所以性能表使用未插桩 benchmark，不能用 trace 下的程序 e2e 代替正式延迟。

## 复现

```bash
cd 07_nlm_denoise/blackbook537
MACA_HOME=/opt/maca \
RESULT_DIR=experiments/results/current/metax_c500 \
bash scripts/run_reproducible_metax.sh
```

本次服务器工作区保留在
`/data/nlm_metax_blackbook537_20260915_I8ilux`。`checksums.sha256` 覆盖首批 35 个
原始证据文件，`addendum_checksums.sha256` 覆盖 CPU 三次基线和灰度/base 的 4 个
新增文件；两组下载均在本地复核通过。
