# RTX 3060 Laptop 历史结果

这里保存开发机 WSL2 上的历史实测基线。文件名带 `legacy` 表示其 CSV schema 是旧版：
仅含 kernel 最小值和端到端均值，不含逐次样本或标准差。后续正式 PR 的主要性能结论
以 `experiments/results/rtx4090d/` 为准。

2026-09-15 使用当前未提交工作树补跑了实验链路验证：

- `environment_quality.txt`：RTX 3060 Laptop、驱动 610.88、CUDA 12.9.86、sm_86；
- `quality_tradeoff.csv` / `quality_sigma.csv`：配对图像质量指标；
- `benchmark_quality.csv`：small/base/large、V0/V1/V2、warmup=3/repeat=5 的
  mean/min/stddev；
- `quality_run.log` / `benchmark_quality.log`：完整终端文本。

环境文件同时记录 `git_worktree=dirty`，表示这些结果用于验证尚未提交的重构代码。
完成提交后，4090 正式复跑必须记录新的 commit 和 `git_worktree=clean`。
