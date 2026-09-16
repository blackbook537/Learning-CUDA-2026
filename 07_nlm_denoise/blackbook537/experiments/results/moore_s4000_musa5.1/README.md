# MTT S4000 / MUSA 5.1 实测结果

本目录保存 2026-09-15 在 Moore Threads MTT S4000 上生成的原始 CSV 与文本日志。
结论：`PLATFORM=moore` 的干净构建、单元测试、256×256 与 1080p 正确性、1080p/4K
性能矩阵及质量扫描全部通过。

## 环境

| 项目 | 实测值 |
|---|---|
| GPU | MTT S4000，47.91 GB，64 个计算单元，warp size 128 |
| 驱动 / Runtime | 5.1 / 5.1 |
| MUSA Toolkit / mcc | 5.1.0 / 5.1.0（clang 14.0.0 前端） |
| OS | Ubuntu 22.04.5 LTS，Linux 5.15 |
| CPU | 2 × Intel Xeon Platinum 8358P，128 逻辑线程 |
| Host 编译器 | g++ 11.4.0；GNU Make 4.3 |
| 构建参数 | `PLATFORM=moore FAST_EXP=0 WITH_OPENCV=0` |

完整设备属性和工具链 commit 见 `environment_collector_verified.txt`。

## 验收结果

| 检查 | 结果 | 证据 |
|---|---|---|
| 干净构建 | PASS | `build.log` |
| 不设置 MUSA PATH/LD_LIBRARY_PATH 的修正版构建与运行 | PASS | `build_rpath.log`、`smoke/patched_rpath_run.log` |
| 平台对象/可执行文件隔离、公共 CLI 选择与增量构建 | PASS | `build_platform_target.log`、`build_platform_target_incremental.log`、`smoke/platform_target_run.log` |
| 默认 / verbose `make test PLATFORM=moore` | PASS；默认只跑 12/12 单测，verbose 附加 GPU 校验 | `make_test_default.log`、`make_test_verbose.log` |
| 256×256 RGB，V0/V1/V2 对 CPU | PASS，三版本 MAE=0、PSNR=∞ | `validate_256x256.log` |
| 1920×1080 RGB，V0/V1/V2 对 CPU | PASS，三版本 MAE=0、PSNR=∞ | `validate_1920x1080_rgb.log` |
| 完整 48 组合矩阵 | PASS，48 个唯一配置，零无效耗时 | `benchmark_full_matrix_repeat3.csv` |
| σ=10/25/50 与 small/base/large 质量扫描 | PASS | `quality_*.csv` |

1080p 正确性测试的 CPU 参考耗时为 187,922.0 ms；V0/V1/V2 的单次 kernel
耗时为 1030.048 / 569.469 / 468.039 ms，量化输出均与 CPU 逐像素一致。
`make_test.log` 保留修复前“默认模式误入 verbose”的现场；最终验收应以
`make_test_default.log` 和 `make_test_verbose.log` 为准。

## 正式基准：RGB/base

以下两组采用 warmup=3、repeat=10。表中的 `mean ± stddev (min)` 单位均为 ms。
1080p CPU 基线独立测量一次，为 186,736.456 ms；4K 按安全约定不运行 CPU 参考。

| 尺寸 | 版本 | kernel mean ± stddev (min) | e2e mean ± stddev (min) | Mpx/s | CPU/e2e |
|---|---|---:|---:|---:|---:|
| 1920×1080 | V0 | 1030.027 ± 0.045 (1029.961) | 1033.800 ± 0.908 (1032.837) | 2.013 | 180.63× |
| 1920×1080 | V1 | 567.906 ± 1.576 (566.320) | 571.939 ± 1.891 (569.703) | 3.651 | 326.50× |
| 1920×1080 | V2 | **467.985 ± 0.066 (467.900)** | **471.324 ± 0.598 (470.701)** | **4.431** | **396.20×** |
| 3840×2160 | V0 | 4101.874 ± 0.130 (4101.651) | 4130.489 ± 7.841 (4117.538) | 2.022 | n.a. |
| 3840×2160 | V1 | 2237.455 ± 0.037 (2237.411) | 2261.090 ± 10.077 (2250.584) | 3.707 | n.a. |
| 3840×2160 | V2 | **1842.982 ± 1.425 (1842.171)** | **1873.327 ± 8.184 (1860.180)** | **4.501** | n.a. |

V2 相对 V0 的 kernel 加速比为 1080p 2.20×、4K 2.23×。base 配置在 S4000
上的端到端帧率约为 1080p 2.12 fps、4K 0.53 fps，未达到交互式门槛。

## 质量与延迟

同一 σ=25、seed=7 的 1080p RGB 合成输入，V2，延迟来自同一次
warmup=3/repeat=10 扫描：

| 配置 | pr/sr | MAE vs clean | PSNR | e2e mean ± stddev |
|---|---|---:|---:|---:|
| noisy | — | 12.485684 | 24.947823 dB | — |
| small | 2/7 | 0.803424 | 47.358490 dB | 109.352 ± 0.008 ms |
| base | 3/10 | 0.597227 | 49.414889 dB | 471.184 ± 0.250 ms |
| large | 4/14 | 0.463451 | 50.948416 dB | 2426.481 ± 12.716 ms |

small 相对 base 快 4.31×，PSNR 下降 2.06 dB；large 相对 base 慢 5.15×，
PSNR 只提高 1.53 dB。质量数值与 NVIDIA 实测逐位一致，说明设备平台没有改变
量化后的算法输出。

| σ | noisy PSNR | denoised PSNR | 改善 |
|---:|---:|---:|---:|
| 10 | 32.883942 dB | 53.265898 dB | +20.381956 dB |
| 25 | 24.947823 dB | 49.414889 dB | +24.467066 dB |
| 50 | 18.960290 dB | 43.964257 dB | +25.003967 dB |

## 复现

```bash
cd 07_nlm_denoise/blackbook537
MUSA_HOME=/usr/local/musa \
RESULT_DIR=experiments/results/current/moore_s4000 \
bash scripts/run_reproducible_moore.sh
```

`benchmark_1080p_rgb_base.csv`、`benchmark_4k_rgb_base.csv` 是十次正式统计；
`benchmark_1080p_rgb_all.csv` 是质量—延迟表的统一采样；
`benchmark_full_matrix_repeat3.csv` 以 warmup=1、repeat=3 覆盖完整 48 组合。

本镜像包含 MUPTI 组件，但未提供 `muprof`、`nsys`、`ncu` 或 `perf` 可执行程序，
因此本次没有声称获得 Moore 平台时间线或硬件计数器数据。服务器生成的 PNG 保留在
`/data/nlm_moore_blackbook537_20260915_rbEVW2`；仓库仅保存 CSV/日志，避免重复提交
大图。`SHA256SUMS.txt` 对下载回本地的 36 个文本证据校验全部通过。
