# Iluvatar MR-V100 / CoreX 4.4.0 实测结果

本目录保存 2026-09-15 至 2026-09-16 在 Gitee AI 天数智芯实例上生成的原始
CSV、终端日志、输出图像与校验和。本次实例实际硬件为 **Iluvatar MR-V100 32 GB**，
以下结论只对应本目录记录的 CoreX 4.4.0 环境。

## 环境

| 项目 | 实测值 |
|---|---|
| GPU | Iluvatar MR-V100，32,768 MiB |
| IX-ML / 驱动 | 4.4.0 / 4.4.0 |
| CoreX | `/usr/local/corex -> /usr/local/corex-4.4.0` |
| CUDA 兼容层 | 10.2；nvcc 10.2.89 |
| 设备编译器 | CoreX clang++ 18.1.8（CoreX 4.4.0） |
| OS | Ubuntu 24.04.4 LTS，Linux 5.4 |
| CPU | 2 × Xeon Platinum 8358P，112 逻辑 CPU；CPU 参考实现为串行口径 |
| Host 工具 | g++ 13.3.0；GNU Make 4.3 |
| 构建参数 | `PLATFORM=iluvatar COREX_HOME=/usr/local/corex FAST_EXP=0 WITH_OPENCV=0` |

完整设备状态、工具版本与 CPU 拓扑见 `environment.txt`。测试使用当前未提交工作树的
上传快照，Git 元数据未随归档上传，因此环境清单如实记录为 `git_commit=unavailable`；
`source_snapshot.txt` 保存源快照与两次最小补丁归档的 SHA-256。

`benchmark_base.csv` 在原始 Makefile 成功构建后采集；随后补丁只把同一个 CoreX
编译器/SDK 路径参数化并增加 runtime RUNPATH，没有修改算法、编译优化级别或 kernel。
补丁版经干净重建后用于 48 配置矩阵和质量扫描，并重新通过单测与 256p GPU 校验。

## 验收结果

| 检查 | 结果 | 证据 |
|---|---|---|
| 原始 Iluvatar Makefile 干净构建 | PASS | `logs/build_original.log` |
| CoreX 路径参数化并写入 RUNPATH 后干净构建 | PASS | `logs/build.log`、`binary_runtime.txt` |
| 默认 / verbose `make test PLATFORM=iluvatar` | PASS；12/12 单测，verbose 附加 GPU 校验 | `logs/make_test*.log` |
| 256×256 RGB，V0/V1/V2 对 CPU | PASS，三版本 MAE=0、PSNR=∞ | `logs/make_test_verbose_patched.log` |
| 1920×1080 RGB，V0/V1/V2 对 CPU | PASS，三版本 MAE=0、PSNR=∞ | `logs/validate_1920x1080_rgb.log` |
| RGB/base 1080p/4K，warmup=3/repeat=10 | PASS | `benchmark_base.csv` |
| CPU 三次独立基线 | PASS | `cpu_baseline_repeat3.csv` |
| 完整 48 个唯一配置 | PASS，warmup=1/repeat=3 | `benchmark_matrix.csv`、`matrix_integrity.txt` |
| small/base/large 与 σ=10/25/50 质量扫描 | PASS | `quality_*.csv`、`logs/quality_sweep.log` |
| 三版本输出文件一致性 | PASS，三个 PNG 的 SHA-256 相同 | `output_equivalence.txt` |
| 原生 profiler | 镜像未提供 ixprof/nsys/ncu CLI；未伪造 trace | `logs/profiler_availability.log` |

1080p 正确性校验的 CPU 参考耗时为 232,556.7 ms；V0/V1/V2 单次 kernel
耗时为 1561.551 / 792.049 / 712.314 ms，量化输出均与 CPU 逐像素一致。

## 正式基准：RGB/base

以下采用 warmup=3、repeat=10；`mean ± stddev (min)` 单位均为 ms。1080p CPU
基线独立测量三次，为 232,814.573 ± 63.255 (232,737.430) ms；4K 按安全约定
不运行 CPU 参考。

| 尺寸 | 版本 | kernel mean ± stddev (min) | e2e mean ± stddev (min) | Mpx/s | CPU/e2e |
|---|---|---:|---:|---:|---:|
| 1920×1080 | V0 | 1561.032 ± 1.772 (1556.603) | 1565.984 ± 1.812 (1561.638) | 1.328 | 148.67× |
| 1920×1080 | V1 | 779.246 ± 7.910 (767.522) | 783.766 ± 8.029 (771.946) | 2.661 | 297.05× |
| 1920×1080 | V2 | **716.595 ± 9.802 (705.615)** | **721.346 ± 9.947 (710.146)** | **2.894** | **322.75×** |
| 3840×2160 | V0 | 6157.025 ± 6.053 (6149.317) | 6176.088 ± 6.140 (6167.897) | 1.347 | n.a. |
| 3840×2160 | V1 | 3100.127 ± 20.517 (3062.086) | 3118.679 ± 20.577 (3080.607) | 2.676 | n.a. |
| 3840×2160 | V2 | **2869.991 ± 42.442 (2823.746)** | **2888.719 ± 42.348 (2843.283)** | **2.890** | n.a. |

V2 相对 V0 的 kernel 加速为 1080p **2.18×**、4K **2.15×**；V2 相对 V1
为 1.09×/1.08×。base 的 2.89 Mpx/s 远未达到 1080p 实时帧率，因此本结果只证明
完整流程和优化收益，不能声称 MR-V100 上实现了 base 参数实时处理。

## 质量与延迟

同一 σ=25、seed=7 的 1080p RGB 合成输入，质量由确定性输出计算。small/large
延迟来自 warmup=1/repeat=3，base 来自 warmup=3/repeat=10；均采用 V2。

| 配置 | pr/sr | MAE vs clean | PSNR | V2 e2e mean ± stddev |
|---|---|---:|---:|---:|
| noisy | — | 12.485684 | 24.947823 dB | — |
| small | 2/7 | 0.803424 | 47.358490 dB | 149.895 ± 0.097 ms |
| base | 3/10 | 0.597227 | 49.414889 dB | 721.346 ± 9.947 ms |
| large | 4/14 | 0.463451 | 50.948416 dB | 3762.661 ± 1.089 ms |

| σ | noisy PSNR | denoised PSNR | 改善 |
|---:|---:|---:|---:|
| 10 | 32.883942 dB | 53.265898 dB | +20.381956 dB |
| 25 | 24.947823 dB | 49.414889 dB | +24.467066 dB |
| 50 | 18.960290 dB | 43.964257 dB | +25.003967 dB |

## 复现

```bash
cd 07_nlm_denoise/blackbook537
COREX_HOME=/usr/local/corex \
RESULT_DIR=experiments/results/current/iluvatar_mrv100 \
bash scripts/run_reproducible_iluvatar.sh
```

本次服务器工作区保留在
`/data/nlm_iluvatar_blackbook537_20260915_zLziaW`。下载归档 SHA-256 为
`12a9605c0bd13820097098a757bd883db1b079aa02104dfe75bcdffe25caa66e`；
`checksums.sha256` 覆盖 39 个原始证据文件，本地复核 39/39 通过。
