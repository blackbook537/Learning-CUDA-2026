# nlm_denoise —— 实时图像非局部均值降噪（CUDA）

> "巡天"深空探测影像机载降噪模块：在 GPU 上实现可配置、可验证、可分析性能瓶颈的
> Non-Local Means（NLM）图像降噪，支持灰度/RGB、1080p 基准与 4K 进阶处理，
> NVIDIA 路径已在 RTX 3060 Laptop 与 RTX 4090 D 实测；国产平台已在 Iluvatar
> MR-V100 / CoreX 4.4.0、MTT S4000 / MUSA 5.1、MetaX C500 25% sGPU /
> MACA 3.0 完成全流程实测。

![平台](https://img.shields.io/badge/platform-NVIDIA%20%7C%20Iluvatar%20%7C%20MetaX%20%7C%20Moore-blue)
![标准](https://img.shields.io/badge/C%2B%2B-17%20%2F%2011(MUSA)-green)
![测试](https://img.shields.io/badge/units-12%2F12%20PASS-brightgreen)
![正确性](https://img.shields.io/badge/MAE%20vs%20CPU-0.0000-brightgreen)

---

## 目录

- [项目概述](#项目概述)
- [核心功能](#核心功能)
- [性能速览](#性能速览)
- [项目结构](#项目结构)
- [安装步骤](#安装步骤)
- [配置指南](#配置指南)
- [使用方法](#使用方法)
- [数据管理](#数据管理)
- [API 文档](#api-文档)
- [测试与验证](#测试与验证)
- [贡献指南](#贡献指南)
- [常见问题解答](#常见问题解答)
- [联系方式](#联系方式)
- [文档导航](#文档导航)

---

## 项目概述

探测器在远日小行星背光面低空飞行，机载相机在极低照度、短曝光下拍摄的画面充满随机噪声与传感器热噪声，直接用于地形重建、着陆点筛选与异常目标识别。普通均值滤波会抹掉陨石坑边缘与细小裂隙，过强局部滤波又保留大量颗粒噪声。

本项目实现 **NLM 降噪**：以 patch 相似性加权平均，在保留纹理边缘的同时抑制噪声。权重公式（本项目的统一实现口径，CPU 参考与全部 GPU kernel 严格一致）：

```text
w(p,q) = exp( -max(dist(P_p,P_q) - 2·σ²·N_patch, 0) / h² )
  dist    : 两 patch 的 L2 距离之和（RGB 跨通道求和、三通道共享权重）
  N_patch : (2·patch_radius+1)² × channels
  边界    : 每次访问坐标独立 clamp（等价 OpenCV BORDER_REPLICATE）
```

算法计算量极大（1080p RGB 基线单帧约 4.5×10¹⁰ 次乘加），本项目通过三级渐进式 GPU 优化（naive → shared memory → 模板展开）；RTX 4090 D 上相对单线程 CPU 参考实测约 **2838×**。

## 核心功能

| 功能 | 入口 | 说明 |
|---|---|---|
| 单图降噪 | `run` | GPU NLM 降噪，输出同格式图像 + 性能日志（ms / Mpx/s / 加速比） |
| 正确性校验 | `validate` | 三个 kernel 版本逐一对比 CPU 参考，MAE/PSNR 判定，退出码可接 CI |
| 性能基准 | `bench` | 分辨率 × 通道 × 参数组合 × kernel 版本扫描，输出 CSV |
| 图像指标 | `metrics` | 对参考图与测试图计算 MAE/PSNR，结果可追加到 CSV |
| 测试图生成 | `gen` | 合成"渐变纹理 + 可控噪声"测试图，零素材端到端验证 |
| 单元测试 | `units` | 12 项 Host 侧测试（无需 GPU） |

**三级优化 kernel**（`--kernel 0/1/2`）：

| 版本 | 技术手段 | RTX 4090 D、1080p RGB kernel 实测 |
|---|---|---|
| V0 naive | 1 线程 = 1 像素，全全局内存 | 125.91 ms |
| V1 smem | halo tile 协同加载 shared memory，patch 距离全命中 smem | 58.90 ms |
| V2 unroll | V1 + patch 半径模板化全展开（pr≤3 自适应分派） | **57.73 ms** |

工程特性：smem 需求超设备上限自动回退 naive；输出 bit 级确定（无原子加）；
pr=4 展开回归已按实测修正；`__expf` 近似为可关断编译开关。

## 性能速览

> 主验证机：RTX 4090 D（sm_89），CUDA 12.8；以下是 2026-09-07 历史实测的
> kernel 最小值。原始数据与环境见
> [experiments/results/rtx4090d](experiments/results/rtx4090d/)。

| 配置 | V0 | V1 | V2 | V2 吞吐 | V2 vs V0 |
|---|---|---|---|---|---|
| 1080p 灰度 base | 72.02 ms | 31.40 ms | **26.99 ms** | 76.8 Mpx/s | 2.67× |
| 1080p RGB base | 125.91 ms | 58.90 ms | **57.73 ms** | 35.9 Mpx/s | 2.18× |
| 4K 灰度 base | 286.95 ms | 124.66 ms | **106.37 ms** | 78.0 Mpx/s | 2.70× |
| 4K RGB base | 502.12 ms | 235.93 ms | **229.22 ms** | 36.2 Mpx/s | 2.19× |

正确性（1080p RGB 基线 validate）：三版本对 CPU 参考 **MAE = 0.0000、PSNR = inf**；
CPU 参考 174.535 s，V2 GPU 端到端 61.489 ms，对应约 **2838×**。

Moore Threads MTT S4000 / MUSA 5.1 也已完成全流程实测。1080p RGB/base 的 V2
kernel/e2e 均值为 **467.985/471.324 ms**，对同机 CPU 参考加速 **396.20×**；
4K RGB/base 为 **1842.982/1873.327 ms**。三版本对 CPU 均 MAE=0，完整 48 组合
矩阵无失败。原始证据见
[experiments/results/moore_s4000_musa5.1](experiments/results/moore_s4000_musa5.1/)。

MetaX C500 25% sGPU / MACA 3.0 也已完成同一流程。1080p RGB/base 的最优版本
是 **V1**：kernel/e2e 均值 **418.397/420.733 ms**，对同机三次 CPU 参考加速
**426.87×**；4K 为 **1645.636/1655.068 ms**。V2 在该平台反而比 V1 慢
3.55–3.58×，详见
[C500 结果目录](experiments/results/metax_c500_maca3.0/)。

Iluvatar MR-V100 / CoreX 4.4.0 的同口径正式基准中，1080p RGB/base 的 V2
kernel/e2e 均值为 **716.595/721.346 ms**，同机 CPU 三次均值 232.815 s，
加速 **322.75×**；4K V2 为 **2869.991/2888.719 ms**。干净构建、12/12 单测、
1080p 三版本 CPU 对照、48 个唯一配置和质量扫描全部通过，见
[MR-V100 结果目录](experiments/results/iluvatar_mrv100_corex4.4.0/)。

![RTX 4090 performance](docs/assets/performance_4090.png)

![MTT S4000 performance](docs/assets/performance_moore_s4000.png)

![MetaX C500 performance](docs/assets/performance_metax_c500.png)

![Iluvatar MR-V100 performance](docs/assets/performance_iluvatar_mrv100.png)

![Visual quality](docs/assets/quality_triptych.png)

## 项目结构

```text
blackbook537/
├── Makefile / README.md / LICENSE
├── include/ src/ kernels/ tester/ third_party/   # 产品代码与测试
├── params/default.txt                             # 默认运行参数
├── data/{clean,noisy}/                            # 最小可复现输入数据
├── experiments/
│   ├── configs/                                   # small/base/large/sigma10/25/50
│   ├── results/{rtx4090d,rtx3060_laptop,moore_s4000_musa5.1,metax_c500_maca3.0}/
│   │                                               # 已验证原始结果
│   ├── results/unverified/                        # 未经硬件验证的平台说明
│   └── work/                                      # 运行期图像，不提交
├── scripts/                                       # 环境采集、验收、质量/性能、profiling
└── docs/
    ├── experiment_protocol.md                     # 完整复现实验手册
    └── assets/                                    # PR 可直接显示的压缩图表
```

### 文件存放规范

| 规则 | 说明 |
|---|---|
| 头文件只进 `include/` | 按层次归类：对外 API → `nlm/`，算子接口 → `kernels/`，平台适配 → `pal/`，支撑模块 → `core/`，测试辅助 → `tester/`；源码引用统一带层次前缀（如 `#include "nlm/pipeline.h"`），编译统一 `-Iinclude` |
| kernel 按架构归位 | 新增平台：`kernels/<平台>/kernels.<后缀>` 薄包装（`#include "../common/kernels_impl.inl"`），Makefile 按 `PLATFORM` 自动选取，无需改构建脚本 |
| 算法改动只碰共享实现 | 新增 kernel 版本（如 V3）只改 `kernels/common/kernels_impl.inl`，四个平台同步生效 |
| 平台差异只进 PAL | 设备 API 差异只允许出现在 `include/pal/platform_api.h`，算法代码禁止直接调用平台原生 API |
| 产物不落源码目录 | 编译产物（可执行文件、.o）只允许出现在 `build/`，`make clean` 一键清理 |
| 数据与证据分离 | 输入样例进 `data/`；临时输出进 `experiments/work/`；可复核 CSV/日志进 `experiments/results/<device>/`；PR 图片进 `docs/assets/` |

## 安装步骤

### 环境要求

| 组件 | 要求 |
|---|---|
| OS | Linux x86_64 |
| 编译 | NVIDIA：CUDA Toolkit ≥ 11.0；Iluvatar：CoreX 4.4；MetaX：MACA 3.0；Moore：MUSA Toolkit 5.1；GNU Make |
| GPU | NVIDIA sm_86/sm_89、Iluvatar MR-V100、MetaX C500 25% sGPU、MTT S4000 MUSA cc 2.2 均已验证 |
| 可选 | OpenCV 4.x（`validate` 交叉验证） |

### 构建

```bash
git clone <Learning-CUDA 仓库>
cd Learning-CUDA/07_nlm_denoise/blackbook537

make                          # 构建并运行测试（同参考仓库约定：all = build + run）
make build                    # 仅编译
make run                      # 运行测试（单元测试）
make run VERBOSE=true         # verbose 模式：附加 GPU 端到端正确性校验
make clean                    # 清理产物
make test                     # 同 run（兼容别名）

make PLATFORM=metax           # 沐曦 MACA（编译 kernels/metax/kernels.maca）
make PLATFORM=moore           # 摩尔 MUSA（默认 MUSA_HOME=/usr/local/musa）
make PLATFORM=iluvatar        # 天数 CoreX
make WITH_OPENCV=1            # 附加 OpenCV 交叉验证
make FAST_EXP=1               # 权重 __expf 近似开关（默认关，见 FAQ）
```

构建产物为单可执行文件 `nlm_denoise`；图像编解码由 stb 静态编入，运行时仍需目标
GPU 驱动和对应的 CUDA/MUSA runtime。

> **本地开发机（无系统级 CUDA 的 WSL 等）**：nvcc 不在 PATH 或系统 gcc 版本过新时，
> 用变量覆盖而非改 Makefile：
> ```bash
> make NVCC=/path/to/nvcc \
>      CXX_HOST=/path/to/g++ \         # nvcc 12.x 需 gcc ≤ 14
>      CUDA_RT=/path/to/cuda_runtime   # 含 include/ 与 lib/ 的 cudart 目录
> ```
> 或直接执行 `bash scripts/make_local_wsl.sh`（已按本机环境预置路径）。

> **Moore Threads / MUSA**：Makefile 默认使用 `/usr/local/musa/bin/mcc`，并把
> `/usr/local/musa/lib` 写入二进制 RUNPATH，不要求预先配置 PATH/LD_LIBRARY_PATH。
> SDK 位于其他目录时执行
> `make build PLATFORM=moore MUSA_HOME=/path/to/musa`。

> **MetaX / MACA**：Makefile 默认使用 `/opt/maca`，从
> `/opt/maca/tools/cu-bridge/include` 取 CUDA 兼容头，并把 `/opt/maca/lib` 写入
> RUNPATH。SDK 位于其他目录时执行
> `make build PLATFORM=metax MACA_HOME=/path/to/maca`；必要时同时覆盖
> `MACA_CUDA=/path/to/cu-bridge MXCC=/path/to/mxcc`。

> **Iluvatar / CoreX**：Makefile 默认使用 `/usr/local/corex/bin/clang++`，设备侧以
> `-x ivcore --cuda-path=/usr/local/corex` 编译，并把 CoreX `lib64` 写入 RUNPATH。
> SDK 位于其他目录时执行
> `make build PLATFORM=iluvatar COREX_HOME=/path/to/corex`；必要时覆盖
> `COREX_CXX=/path/to/clang++`。

## 配置指南

参数文件（默认 [params/default.txt](params/default.txt)），`key = value` 文本格式，`#` 注释：

```text
patch_radius = 3          # patch 半径，实际 patch 大小为 (2*patch_radius+1)^2
search_radius = 10        # 搜索窗口半径
h = 10.0                  # 滤波强度
sigma = 25.0              # 噪声标准差估计，对于8位图像范围0-255
```

- 取值范围：pr∈[1,8]，sr∈[1,32]，h∈(0,1000]，σ∈[0,1000]；任务基线 pr=3 / sr=10；
- 未知 key、缺 `=`、越界取值均报明确错误；
- 调参建议：`sigma` 按真实噪声估计；保纹理先减小 `h`；计算量 ∝ sr²，先调 `h` 再调 `sr`。

## 使用方法

```bash
# 1) 生成测试图（缺省 -o 自动按规范命名到 data/noisy/）
./build/nlm_denoise gen --size 1920x1080 --channels 3 --sigma 25
#   -> data/noisy/noisy_1920x1080_3ch_sigma25.png

# 2) 单图降噪（kernel：0=naive / 1=smem / 2=smem+unroll；输出目录自动创建）
./build/nlm_denoise run -i data/noisy/noisy_1920x1080_3ch_sigma25.png \
                         -o data/output/denoised_noisy_1920x1080_3ch_v2.png \
                         -p params/default.txt --kernel 2 --with-cpu
#   [run] kernel=347.9 ms  e2e=354.0 ms (H2D 1.8 / D2H 3.8)  5.96 Mpx/s
#   [run] CPU=141210.5 ms  加速比=399.0x   （性能日志追加至 data/logs/nlm_perf.log）

# 3) 正确性校验（退出码 0=PASS / 1=FAIL / 2=错误）
./build/nlm_denoise validate -i data/noisy/noisy_1920x1080_3ch_sigma25.png \
                             -o data/output/validated_v2.png -p params/default.txt
#   ver  kernel(ms)  e2e(ms)   MAE(vsCPU)  PSNR(dB)  判定
#   2    347.898     354.021   0.0000      inf       PASS

# 4) 性能基准（mean/min/stddev + CPU 统一口径）
./build/nlm_denoise bench --sizes 1920x1080,3840x2160 --channels 1,3 \
                    --param-sets all --warmup 3 --repeat 10 \
                    --log experiments/results/current/benchmark.csv

# 5) RTX 4090 完整复现实验
ARCH=sm_89 bash scripts/run_reproducible_4090.sh

# 6) MTT S4000 / MUSA 完整复现实验
MUSA_HOME=/usr/local/musa bash scripts/run_reproducible_moore.sh

# 7) MetaX C500 / MACA 完整复现实验
MACA_HOME=/opt/maca bash scripts/run_reproducible_metax.sh

# 8) Iluvatar MR-V100 / CoreX 完整复现实验
COREX_HOME=/usr/local/corex bash scripts/run_reproducible_iluvatar.sh
```

各子命令完整选项表见 [docs/user_guide.md](docs/user_guide.md) 第 5 章。

## 数据管理

全部运行期数据（系统生成 + 手动准备）集中存放于 `data/`，按数据流阶段分类：

| 目录 | 用途 | 写入方 |
|---|---|---|
| `data/noisy/` | 含噪输入图：`gen` 合成图或手动放置的实拍噪声图 | `gen`（缺省自动命名）/ 手动 |
| `data/clean/` | 干净参考图：无噪原图，供 MAE/PSNR 质量评估对照 | 手动 |
| `data/output/` | 降噪结果图 | `run -o` / `validate -o` |
| `data/logs/` | 运行日志：`nlm_perf.log`（CSV 行追加写） | `run --log`（缺省值） |

**命名规范**（保证文件可追溯：看名知尺寸/通道/噪声强度/kernel 版本）：

| 数据类型 | 命名模板 | 示例 |
|---|---|---|
| 含噪输入 | `noisy_<宽x高>_<通道>ch_sigma<σ>[_seed<s>].png` | `noisy_1920x1080_3ch_sigma25.png` |
| 干净参考 | `clean_<对应输入基名>.png` | `clean_1920x1080_3ch.png` |
| 降噪结果 | `denoised_<输入基名>_v<kernel版本>.png` | `denoised_noisy_1920x1080_3ch_v2.png` |
| 运行日志 | `nlm_perf.log`（固定名，追加写） | — |

**自动化保障**（忘记路径也不会污染仓库根目录）：

- `gen` 缺省 `-o` 时按命名规范自动生成 `data/noisy/` 下的文件名；
- `run --log` 缺省 `data/logs/nlm_perf.log`；
- `SaveImage` 与日志写入前自动逐级创建父目录（`EnsureParentDir`），`-o data/output/xx.png` 无需手工建目录；
- 数据流可一键验证：`bash scripts/check_dataflow_wsl.sh`（gen → run → validate 全链路 + 根目录清洁检查）。

> 正式实验输出统一写入 `experiments/results/<device>/`；临时生成图像写入
> `experiments/work/` 并由 `.gitignore` 排除。

## API 文档

核心 C++ 接口（供二次开发/集成；完整注释见各头文件）：

```cpp
// include/nlm/pipeline.h —— GPU 降噪全流程（无状态，外层可包帧循环）
bool NlmDenoiseGpu(const ImageU8& src, ImageU8* dst,
                   const NlmParams& params, int kernel_version,
                   NlmPerfReport* perf, std::string* err);

// include/core/nlm_cpu_ref.h —— CPU 参考（正确性 ground truth）
void NlmDenoiseCpuRef(const ImageU8& src, ImageU8* dst, const NlmParams& params);

// include/nlm/params.h —— 参数文件解析（pr/sr/h/sigma + 范围校验）
bool ParseParams(const char* path, NlmParams* out, std::string* err);

// include/core/image_io.h —— PNG/JPG 编解码（灰度/RGB，自动剥离 alpha）
bool LoadImage(const char* path, ImageU8* out, std::string* err);
bool SaveImage(const char* path, const ImageU8& img, std::string* err);

// include/kernels/kernels.h —— 设备资源与 kernel 调度（pipeline 专用，屏蔽 GPU 运行时）
bool GpuNlm(const float* d_src, float* d_dst, int w, int h, int c,
            const NlmParamsDev& p, int version, float* kernel_ms, std::string* err);

// include/tester/utils.h —— 质量指标
double ComputeMAE(const ImageU8& a, const ImageU8& b);
double ComputePSNR(const ImageU8& a, const ImageU8& b);
```

约定：图像 Host 侧 interleaved u8、设备侧 planar f32；`kernel_version ∈ {0,1,2}`（3 为预留进阶版本）；所有接口通过 `err` 返回中文诊断，不打印、不抛异常。

## 测试与验证

| 层级 | 命令 | 覆盖 |
|---|---|---|
| 单元测试 | `./build/nlm_denoise units` | 参数解析、MAE/PSNR、CPU 参考不变式（常量恒等/确定性/极小图/降噪有效性），12 项全过 |
| 集成校验 | `./build/nlm_denoise validate ...` | 三版本 GPU vs CPU 参考 MAE/PSNR（1080p 实测 MAE=0） |
| 性能回归 | `./build/nlm_denoise bench ...` | 统计 CSV，对比 `experiments/results/` 中同硬件基线 |
| 数据流检查 | `bash scripts/check_dataflow_wsl.sh` | gen 自动命名 → run 入 data/output+logs → validate → 根目录清洁 |
| 一键准入 | `bash scripts/run_all.sh` | 以上全部串联，任何一步失败即中断 |

## 贡献指南

1. **分支约定**：在 Learning-CUDA 仓库 `2026-summer-project` 分支上开发，提交前同步最新主干；
2. **提交准入**（必须全部满足）：
   - `make test` 单元测试 12/12 通过；
   - `./build/nlm_denoise validate` 三版本判定 PASS（V0/V1 MAE≤0.5，V2≤1.0，PSNR≥30dB）；
   - 新增 kernel 版本须附带与 V0 的 MAE 对比与 bench 数据（写入提交说明）；
3. **编码规范**：
   - NLM 主体及关键计算**禁止调用库函数直接实现**；主要计算须在 GPU 完成；
   - 算法核心代码集中于 `kernels/common/kernels_impl.inl`（多平台共享），平台差异只允许进 `include/pal/platform_api.h`；
   - 保持 C++11 子集兼容（摩尔线程平台）；风格与现有代码一致，重要逻辑配中文注释；
   - 所有 GPU API 调用经 `GPU_CHECK` 宏，错误经 `std::string* err` 上传，不裸 printf；
4. **学术诚信**：禁止抄袭其他学员与开源实现（可讨论思路，禁止看/抄代码），一经发现成绩作废；
5. **提交内容**：代码、测试、环境清单、原始 CSV/文本日志、复现实验手册及压缩图表一并提交。

## 常见问题解答

**Q：validate 为什么不直接以 OpenCV 为验收基线？**
A：任务公式 `dist` 为 L2 距离之和（不做均值）、减项为 `2σ²·N_patch`，与 OpenCV 的 `h` 标定及边界实现存在差异，直接对比会产生虚假 MAE。故以语义严格一致的自研 CPU 参考为门槛（三版本实测 MAE=0），OpenCV 仅作信息性交叉验证（`make WITH_OPENCV=1`）。

**Q：支持哪些图像？大图/特殊参数会怎样？**
A：8 位灰度或 RGB 的 PNG/JPG（RGBA 自动剥离 alpha）。pr≤8、sr≤32；smem 需求超设备上限时 V1/V2 自动回退 naive，功能不会失败（性能下降）。

**Q：V2 一定比 V1 快吗？**
A：不一定。NVIDIA/Moore/Iluvatar 的 pr≤3 实测通常更快，但 MetaX C500 上 base/pr=3 的
V2 比 V1 慢约 3.55×；这是编译器/架构相关的模板展开回退。pr≥4 已在代码中自动
退回 V1 路径；MetaX 当前建议显式使用 `--kernel 1`。

**Q：`FAST_EXP=1` 值得开吗？**
A：实测无性能收益（瓶颈在 patch FMA 而非 SFU 指数），近似误差低于 u8 量化阈值；默认关闭保留精确 `expf`，该开关主要用于报告的误差来源分析对照。

**Q：如何接入 CI / 脚本？**
A：所有子命令退出码规范化（0 成功 / 1 质量不达标 / 2 运行错误），`validate` 与 `units` 可直接作为流水线步骤。

更多问答与故障排除对照表见 [docs/user_guide.md](docs/user_guide.md) 第 8–9 章。

## 联系方式

- 项目答疑：InfiniTensor 训练营社群（@助教）
- 问题反馈：Learning-CUDA 仓库 Issues（请注明平台、GPU 型号、复现命令与完整报错）
- 提交地址：Learning-CUDA 本季度项目分支 `2026-summer-project`

## 文档导航

| 文档 | 内容 |
|---|---|
| [README.md](README.md) | 本文档：概览、安装、使用、贡献 |
| [docs/summary_report.md](docs/summary_report.md) | 总结报告：质量/性能指标、质量-延迟权衡分析、瓶颈定位、问题修正记录 |
| [docs/architecture_design.md](docs/architecture_design.md) | 架构设计：需求基线、分层架构、优化路线、风险权衡 |
| [docs/user_guide.md](docs/user_guide.md) | 详细讲解：运行实录、性能解读、FAQ、故障排除 |
| [docs/refactor_log.md](docs/refactor_log.md) | 结构重构变更记录（对齐 Learning-CUDA 仓库约定） |
| [docs/experiment_protocol.md](docs/experiment_protocol.md) | 实验环境、工具、命令、结果验收和截图生成流程 |
| [experiments/results/rtx4090d](experiments/results/rtx4090d/) | **RTX 4090 D** 环境、测试、基准、质量及 nsys 原始证据 |
| [experiments/results/rtx3060_laptop](experiments/results/rtx3060_laptop/) | RTX 3060 Laptop 开发机历史结果与补充实验 |
| [experiments/results/metax_c500_maca3.0](experiments/results/metax_c500_maca3.0/) | MetaX C500 / MACA 3.0 完整实测、原始 CSV、日志与 mcTracer JSON |
| [experiments/results/moore_s4000_musa5.1](experiments/results/moore_s4000_musa5.1/) | **MTT S4000 / MUSA 5.1** 环境、测试、基准和质量原始证据 |
| [experiments/results/iluvatar_mrv100_corex4.4.0](experiments/results/iluvatar_mrv100_corex4.4.0/) | **Iluvatar MR-V100 / CoreX 4.4.0** 环境、测试、基准、质量和校验和 |
