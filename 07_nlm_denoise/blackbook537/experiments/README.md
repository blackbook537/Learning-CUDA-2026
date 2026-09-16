# 实验数据组织

本目录只保存可复现实验的定义与证据：

- `configs/`：受版本控制的参数组；质量权衡实验固定同一 σ=25 输入，只改变 pr/sr/h；
- `results/rtx4090d/`：正式 NVIDIA 服务器结果；
- `results/rtx3060_laptop/`：开发机结果；
- `results/moore_s4000_musa5.1/`：MTT S4000 / MUSA 5.1 正式实测结果；
- `results/metax_c500_maca3.0/`：MetaX C500 25% sGPU / MACA 3.0 正式实测结果；
- `results/iluvatar_mrv100_corex4.4.0/`：Iluvatar MR-V100 / CoreX 4.4.0 正式实测结果；
- `results/unverified/`：保留状态说明；当前三个国产平台入口均已完成硬件实测；
- `results/current/`：脚本默认输出，已忽略；完整验收通过后再复制到对应设备目录；
- `work/`：生成的 clean/noisy/denoised 图片及 profiler 二进制，已忽略。

正式 CSV 通常应能追溯到 `environment.txt` 中的 Git commit；若为上传的未提交工作树
快照，必须像 C500/MR-V100 结果一样写明 `git_commit=unavailable`，同时保留归档及文件校验和。
旧结果统一带 `_legacy`，表示它们来自升级统计 schema 之前，不能用来声称已测
mean/stddev。

完整规程见 [实验复现手册](../docs/experiment_protocol.md)。
