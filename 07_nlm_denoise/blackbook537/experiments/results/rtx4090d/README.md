# RTX 4090 D 结果说明

`benchmark_legacy.csv` 与 `run_all_legacy.txt` 是 2026-09-07 在 RTX 4090 D
服务器采集的已验证结果。旧 benchmark 只保存 `kernel min` 与 `e2e mean`，没有保留
10 次原始样本，因此不能从该文件诚实反推标准差。

升级后的 `bench` 会输出 mean/min/stddev、warmup/repeat、参数组名，以及使用同一输入和
同一参数采集的 CPU 统计。执行 `scripts/run_reproducible_4090.sh` 后，将生成：

- `environment.txt`：硬件、驱动、CUDA、编译器及 profiler 版本；
- `units.log`、`validate_1080p_rgb.log`：测试和正确性日志；
- `benchmark.csv`：1080p/4K、灰度/RGB、四参数组的 GPU 统计；
- `cpu_baseline.csv`：1080p RGB base 的 CPU/V0/V1/V2 统一口径数据；
- `quality_tradeoff.csv`、`quality_sigma.csv`：质量实验；
- `nsys_stats.txt`：可文本审阅的 Nsight Systems 摘要。

仓库中的 `run_all_legacy.txt` 曾写有“Makefile 默认 ARCH=sm_89”，实际 Makefile 的
`ARCH` 默认留空。本次 4090 实验应始终显式使用 `ARCH=sm_89`。
