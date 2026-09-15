# nlm_denoise 项目讲解与使用文档

**实时图像非局部均值降噪（CUDA）—— "巡天"深空探测影像机载降噪模块**

| 文档版本 | v1.0 |
|---|---|
| 适用代码 | nlm_denoise_project（2026-summer-project） |
| 默认平台 | NVIDIA（兼容 天数 CoreX / 沐曦 MACA / 摩尔 MUSA） |
| 配套文档 | [架构设计文档](architecture_design.md) |

---

## 目录

1. [项目概述](#1-项目概述)
2. [核心功能](#2-核心功能)
3. [技术架构说明](#3-技术架构说明)
4. [环境配置指南](#4-环境配置指南)
5. [详细使用步骤](#5-详细使用步骤)
6. [运行实录与结果解读](#6-运行实录与结果解读)
7. [性能数据总览](#7-性能数据总览)
8. [常见问题解答（FAQ）](#8-常见问题解答faq)
9. [故障排除](#9-故障排除)
10. [附录](#10-附录)

---

## 1. 项目概述

### 1.1 项目背景

探测器在远日小行星背光面低空飞行，机载相机只能在极低照度、短曝光下拍摄，传回画面充满随机噪声与传感器热噪声，而这些画面直接用于地形重建、着陆点筛选和异常目标识别。普通均值滤波会抹掉陨石坑边缘和细小裂隙，过强局部滤波又保留大量颗粒噪声。

本项目在机载 GPU 上实现 **非局部均值（Non-Local Means, NLM）实时降噪**：通过比较图像中小块 patch 的相似性进行加权平均，在最大限度保留纹理与边缘的同时抑制噪声。

### 1.2 NLM 算法一分钟速览

对中心像素 `p` 与搜索窗内的候选像素 `q`，权重由两个 patch 的 L2 距离决定：

```text
w(p, q) = exp( -max(dist(P_p, P_q) - 2·σ²·N_patch, 0) / h² )

  P_p, P_q : 以 p、q 为中心的 (2·pr+1)² patch
  dist     : 两 patch 的 L2 距离之和（RGB 跨通道求和、三通道共享权重）
  N_patch  : (2·pr+1)² × 通道数
  σ        : 噪声标准差估计（8 位图像域 0–255）
  h        : 滤波强度
  边界     : 每次访问坐标独立 clamp（等价 OpenCV BORDER_REPLICATE）
```

输出为搜索窗内所有候选像素的加权平均。**算法难点**：计算量极大（1080p RGB 基线参数单帧约 4.5×10¹⁰ 次乘加）、访存高度冗余、边界处理复杂——这正是 GPU 并行化与 shared memory 数据复用的价值所在。

### 1.3 项目定位

| 维度 | 说明 |
|---|---|
| 输入 | PNG/JPG（灰度单通道或 RGB 三通道，0–255 整数） |
| 参数 | 文本文件配置 `patch_radius` / `search_radius` / `h` / `sigma` |
| 输出 | 同格式降噪图像 + 性能日志（ms、Mpx/s、加速比） |
| 正确性 | 与 CPU 参考逐版本对比 MAE/PSNR（可选 OpenCV 交叉验证） |
| 性能 | 1080p 基准 + 4K 进阶，多参数组合扫描（CSV） |
| 平台 | NVIDIA 默认；天数/沐曦/摩尔经 PAL 层一套代码多平台编译 |

---

## 2. 核心功能

### 2.1 功能清单

| 功能 | 子命令 | 说明 |
|---|---|---|
| 单图降噪 | `run` | GPU NLM 降噪，输出图像 + 性能日志，可选 CPU 加速比 |
| 正确性校验 | `validate` | 三个 kernel 版本逐一对比 CPU 参考（MAE/PSNR 判定） |
| 性能基准 | `bench` | 分辨率 × 通道 × 参数组合 × kernel 版本扫描，输出 CSV |
| 测试图生成 | `gen` | 合成"渐变纹理 + 可控噪声"测试图，无外部素材即可端到端验证 |
| 单元测试 | `units` | 12 项 Host 侧测试（无需 GPU），覆盖参数解析/指标/算法不变式 |

### 2.2 三级渐进式优化 kernel

| 版本 | 名称 | 技术手段 | 1080p RGB 基线实测* |
|---|---|---|---|
| V0 | naive | 1 线程 = 1 输出像素，全部走全局内存 | 798 ms |
| V1 | smem | block 协同加载 halo tile 至 shared memory，patch 距离全命中 smem | 369 ms（2.2×） |
| V2 | smem+unroll | V1 + patch 半径模板化 `#pragma unroll` 全展开（pr≤3） | 348 ms（2.3×） |

\* 测试环境：RTX 3060 Laptop（sm_86），CUDA 12.9，pr=3/sr=10/h=10/σ=25。对单线程 CPU 参考（141.2 s）加速比约 **400×**。

**关键设计**：smem kernel 的 halo 采用"复制边界"加载，使 `smem[局部坐标+偏移] ≡ src[clamp(全局坐标+偏移)]`，与 CPU 参考的 clamp 语义**逐位一致**——因此内层循环零分支、零 clamp，且三版本输出与 CPU 参考 MAE=0（bit 级一致）。

### 2.3 工程健壮性设计

- **显存自适应**：V1/V2 启动前计算动态 smem 需求（`c × tileH × (tileW+1) × 4B`），超出设备上限自动回退 naive 路径，功能永不失败；
- **数值安全**：权重和 `Σw ≥ 1`（自身权重恒为 1），归一化无除零；`expf` 输入恒 ≤ 0 无上溢；量化做 round + clamp[0,255]；
- **确定性**：同输入同参数同版本，输出 bit 级一致（无原子加、归约全在线程私有寄存器）；
- **实测修正**：基准数据发现 pr=4 时模板全展开导致寄存器溢出、性能反降 70%，已将 V2 展开分派限制为 pr≤3（代码内有数据注释）。

---

## 3. 技术架构说明

### 3.1 五层架构

```text
┌─────────────────────────────────────────────────────────┐
│ L5 应用层      nlm_denoise CLI（run/validate/bench/gen/units）│
├─────────────────────────────────────────────────────────┤
│ L4 流程编排层   pipeline：decode→cvt→NLM→量化→encode + 计时日志 │
├─────────────────────────────────────────────────────────┤
│ L3 算子层       转换 kernel + NLM V0/V1/V2（kernels_impl.inl）  │
├─────────────────────────────────────────────────────────┤
│ L2 平台抽象层   PAL 宏：GPU_MALLOC/EVENT/CHECK…               │
│                NVIDIA │ Iluvatar │ MetaX │ Moore           │
├─────────────────────────────────────────────────────────┤
│ L1 基础设施层   stb 图像编解码 │ 参数解析 │ CPU 参考 │ tester   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 模块依赖关系

```text
main.cpp ──┬──► pipeline ──► kernels.h（Gpu* 接口）──► kernels_impl.inl
           │        │                                        │
           │        └──► image_io / params                   ├──► platform_api.h（PAL）
           ├──► nlm_cpu_ref（CPU 基线）                       │
           └──► tester/utils.h（MAE/PSNR/合成图）
                  ├──► tester/validate.cpp
                  ├──► tester/benchmark.cpp
                  └──► tester/test_units.cpp
```

依赖规则：上层可依赖下层，**算子层不依赖任何 Host I/O**；pipeline 通过 `kernels.h` 的纯 C 接口持有设备资源，自身不包含任何 GPU 运行时头文件。

### 3.3 数据流程（run 子命令）

```text
输入PNG/JPG ─► [Host] stb 解码 (u8, H×W×C, interleaved)
                ─► [Host] 参数文件解析 + 校验
                ─► [GPU]  H2D ─► CvtU8ToF32Planar kernel（planar f32）
                ─► [GPU]  NLM kernel（V0/V1/V2，event 计时）
                          每像素：搜索窗×patch 循环 → dist → w=expf(...) → Σw·I, Σw
                ─► [GPU]  CvtF32ToU8Interleaved kernel（round+clamp）─► D2H
                ─► [Host] stb 编码输出（同输入格式）
                ─► [Host] 性能日志：kernel/e2e ms、Mpx/s、CPU 加速比
```

### 3.4 内存布局与映射策略

- **Planar 通道分离**：`d[c][y][x]`，各通道连续，保证每通道访存 coalesced；RGB 联合 patch 距离在一个线程内累加三通道、共享权重（与 OpenCV colored 语义一致）；
- **线程映射**：1 thread = 1 output pixel，`block=(16,16)`，`grid=(⌈W/16⌉, ⌈H/16⌉)`，天然支持任意分辨率与灰度/RGB 统一路径；
- **smem tile**：`tile = (16+2(sr+pr))²`，基线参数 42²×4B≈7KB/通道；行 stride `+1` padding 消除 bank conflict。

### 3.5 多平台适配机制

算法实现 100% 集中于 `kernels/common/kernels_impl.inl`，四个平台编译单元仅一行 include：

```cpp
// kernels/{nvidia,iluvatar}/kernels.cu、kernels/metax/kernels.maca、kernels/moore/kernels.mu
#include "../common/kernels_impl.inl"
```

平台差异全部收敛到 `platform_api.h` 的 PAL 宏（NVIDIA/MACA/CoreX 为 CUDA 兼容运行时；摩尔 MUSA 用 `musa` 前缀，代码限定 C++11 子集）。Makefile 通过 `PLATFORM=` 切换编译器与源文件后缀，与 Learning-CUDA 仓库约定一致。

---

## 4. 环境配置指南

### 4.1 环境要求

| 组件 | 要求 |
|---|---|
| 操作系统 | Linux x86_64（训练营服务器或本地） |
| 编译器 | CUDA Toolkit ≥ 11.0（nvcc）；GNU Make |
| GPU | NVIDIA 计算能力 ≥ 7.0（开发验证机：RTX 3060 Laptop, sm_86） |
| C++ 标准 | C++17（摩尔线程平台为 C++11 子集，代码已兼容） |
| 可选依赖 | OpenCV 4.x（仅 `validate` 交叉验证用，`pkg-config opencv4` 可见） |

国产平台：天数 BI-150 CoreX 环境（clang++）、沐曦标准 MACA 环境（mxcc）、摩尔 MUSA 环境（mcc）。

### 4.2 获取与构建

```bash
cd nlm_denoise_project

# 目标语义与 Learning-CUDA 参考仓库一致
make                          # 构建并运行测试（all = build + run）
make build                    # 仅编译
make run                      # 运行单元测试（无需 GPU）
make run VERBOSE=true         # verbose：附加 GPU 端到端校验（256x256）
make clean                    # 清理产物
make test                     # 同 run（兼容别名）

# 国产平台
make build PLATFORM=metax     # 沐曦（编译 kernels/metax/kernels.maca）
make build PLATFORM=moore     # 摩尔（C++11）
make build PLATFORM=iluvatar  # 天数

# 可选开关
make WITH_OPENCV=1            # validate 增加 OpenCV 交叉验证
make FAST_EXP=1               # 权重换 __expf 近似（见 §8 FAQ-6）
```

### 4.3  smoke test（30 秒验证环境）

```bash
make run                      # 构建 + 运行 12 项单元测试（无需 GPU）
./build/nlm_denoise gen -o /tmp/t.png --size 512x512 --channels 3 --sigma 25
./build/nlm_denoise run -i /tmp/t.png -o /tmp/t_out.png -p params/default.txt --kernel 2
```

看到 `[run] kernel=... ms ... PASS 风格输出` 即环境就绪。

---

## 5. 详细使用步骤

### 5.1 参数文件（params/default.txt）

```text
patch_radius = 3          # patch 半径，实际 patch 大小为 (2*patch_radius+1)^2
search_radius = 10        # 搜索窗口半径
h = 10.0                  # 滤波强度
sigma = 25.0              # 噪声标准差估计，对于8位图像范围0-255
```

规则：`key = value`，`#` 后为注释，缺项用默认值，未知 key / 越界取值报错。
范围约束：pr∈[1,8]，sr∈[1,32]，h∈(0,1000]，σ∈[0,1000]（任务基线 pr=3、sr=10）。

**调参经验**：噪声越大 → `sigma` 如实估计、`h` 适当增大；纹理保真优先 → 减小 `h`；计算预算与 `sr²` 成正比，先动 `h` 再动 `sr`。

### 5.2 子命令详解

#### 5.2.1 run —— 单图降噪

```bash
./build/nlm_denoise run -i data/noisy/noisy.png -o data/output/denoised_noisy_v2.png \
                  -p params/default.txt --kernel 2 --with-cpu
```

| 选项 | 默认 | 说明 |
|---|---|---|
| `-i` / `-o` | （必填） | 输入/输出图像（PNG/JPG/BMP/TGA，按扩展名编码） |
| `-p` | params/default.txt | 参数文件 |
| `--kernel` | 2 | 0=naive，1=smem，2=smem+unroll |
| `--log` | data/logs/nlm_perf.log | 性能日志（CSV 行，追加写） |
| `--with-cpu` | 关 | 同时跑 CPU 参考，输出 MAE/PSNR 与加速比（大图较慢） |

#### 5.2.2 validate —— 正确性校验

```bash
./build/nlm_denoise validate -i data/noisy/noisy.png -o data/output/validated.png -p params/default.txt
```

流程：CPU 参考 → V0/V1/V2 逐一 GPU 运行 → 对比 MAE/PSNR → 判定 PASS/FAIL。
判定阈值：V0/V1 MAE ≤ 0.5，V2 MAE ≤ 1.0，PSNR ≥ 30 dB；`-DWITH_OPENCV` 构建时附 OpenCV 信息性对比（不作门槛，原因见 §8 FAQ-4）。退出码：0=PASS，1=FAIL，2=运行错误——可直接接入 CI。

#### 5.2.3 bench —— 性能基准

```bash
./build/nlm_denoise bench --sizes 1920x1080,3840x2160 --channels 1,3 \
                    --warmup 3 --repeat 10 --log bench.csv [--with-cpu]
```

对每组（分辨率 × 通道 × 4 组参数 × 3 个 kernel 版本）预热 warmup 次、测量 repeat 次，
`kernel_ms` 取最小值、`e2e_ms` 取均值。CSV 列：

```text
size,channels,pr,sr,h,sigma,kernel_ver,kernel_ms,e2e_ms,mpx_s,cpu_ms,speedup
```

> 提示：`--with-cpu` 仅对 ≤1080p 灰度量级的配置测量 CPU 耗时（单线程 CPU 极慢，1080p RGB 约 141 s）。

#### 5.2.4 gen —— 测试图生成

```bash
./build/nlm_denoise gen -o data/noisy/noisy_3840x2160_3ch_sigma25.png --size 3840x2160 --channels 3 --sigma 25 --seed 1
```

生成"双向渐变纹理 + LCG 均匀噪声"合成图，相同 seed 输出完全一致，便于复现。

#### 5.2.5 units —— 单元测试

```bash
./build/nlm_denoise units     # 12 项，无需 GPU；全过返回 0
```

### 5.3 一键流程与性能分析

```bash
bash scripts/run_all.sh        # 构建 → units → 1080p validate → 1080p+4K bench
bash scripts/ncu_profile.sh    # Nsight Compute：V0/V1/V2 三组指标采集（需 ncu）
bash scripts/nsys_profile.sh   # Nsight Systems：4K 全链路时间线（需 nsys）
```

---

## 6. 运行实录与结果解读

以下为开发验证机（RTX 3060 Laptop + CUDA 12.9）上的真实终端输出。

### 6.1 单元测试

```text
========== NLM 单元测试 ==========
[units] 参数解析
  [PASS] 正常解析（含注释/缺省）
  [PASS] 未知 key 拒绝
  [PASS] 越界取值拒绝
[units] 质量指标
  [PASS] MAE(相同图像)=0      [PASS] PSNR(相同图像)=inf
  [PASS] MAE(恒差10)=10       [PASS] PSNR(恒差10)≈28.13dB
[units] CPU 参考实现
  [PASS] 常量灰度图恒等        [PASS] 常量 RGB 图恒等
  [PASS] 确定性（两次运行一致） [PASS] 极小图（5x7, sr=10）不崩溃且尺寸正确
    降噪前后对干净图 MAE: 12.48 -> 1.57
  [PASS] 降噪有效性（MAE 下降）
========== 全部通过（失败 0 项）==========
```

### 6.2 正确性校验（1080p RGB，任务基线参数）

```text
[validate] 图像 1920x1080x3，参数 pr=3 sr=10 h=10.0 sigma=25.0
[validate] CPU 参考耗时: 141210.5 ms
ver    kernel(ms)   e2e(ms)      MAE(vsCPU)   PSNR(dB)   判定
0      797.658      997.716      0.0000       inf        PASS
1      368.950      374.857      0.0000       inf        PASS
2      347.898      354.021      0.0000       inf        PASS
[validate] 总体判定: PASS
```

**解读**：三个优化版本与 CPU 参考 **MAE=0、PSNR=inf**（bit 级一致），证明 smem 边界语义等价性；对 CPU 加速比 ≈ 141210/354 ≈ **399×**。

### 6.3 单图降噪（run）

```text
[run] 1920x1080x3  kernel v2  pr=3 sr=10 h=10.0 sigma=25.0
[run] kernel=347.898 ms  e2e=354.021 ms (H2D 1.826 / D2H 3.782)  5.96 Mpx/s
```

**解读**：kernel 纯计算 348 ms；H2D/D2H 合计 < 6 ms，说明计算是绝对瓶颈（compute-bound），进一步优化应指向算法级（积分图近似等）而非传输。

---

## 7. 性能数据总览

完整 48 组数据见项目根目录 `bench.csv`（warmup=3，repeat=10，kernel_ms 取最小值）。

### 7.1 kernel 耗时（ms）与吞吐量

| 配置 | V0 naive | V1 smem | V2 unroll | V2 吞吐 | V2 vs V0 |
|---|---|---|---|---|---|
| 1080p 灰度 base | 430.6 | 194.1 | **155.0** | 13.4 Mpx/s | 2.8× |
| 1080p RGB base | 806.3 | 379.3 | **366.5** | 5.7 Mpx/s | 2.2× |
| 1080p RGB small(pr2,sr7) | 229.1 | 106.5 | **99.0** | 21.0 Mpx/s | 2.3× |
| 1080p RGB large(pr4,sr14) | 2668.6 | **1465.3** | 1447.3* | 1.4 Mpx/s | 1.8× |
| 4K 灰度 base | 2002.1 | 879.3 | **700.1** | 11.8 Mpx/s | 2.9× |
| 4K RGB base | 3305.3 | 1514.6 | **1440.8** | 5.8 Mpx/s | 2.3× |

\* V2 检测到 pr=4 展开有害后自动回退 V1 路径（见 §2.3）。

### 7.2 关键结论

1. **smem 是最有效优化**：V0→V1 稳定 2.2–2.9×，与理论（全局访存复用率提升约一个数量级）一致；
2. **模板展开收益有边界**：pr≤3 收益 5–30%，pr=4 寄存器溢出反降 70% → 已按实测自适应分派；
3. **`__expf` 无收益**：近似指数误差低于 u8 量化阈值（MAE<5e-5）但耗时不变——瓶颈在 patch FMA 而非 SFU，默认保留精确 `expf`；
4. **4K 交互式目标**：当前 4K RGB base 约 1.66 s，距离交互式（≤100 ms）需算法级近似（积分图/搜索窗裁剪，见架构文档 V3 路线）。

---

## 8. 常见问题解答（FAQ）

**Q1：为什么 validate 必须以自研 CPU 参考为主基线，而不是直接对 OpenCV？**
任务公式中 `dist` 为 patch L2 距离之和（不做均值），减项相应为 `2σ²·N_patch`；OpenCV 的 `h` 标定与边界/归一化实现存在差异，直接对比会出现"虚假 MAE"。因此 CPU 参考（语义与任务公式严格一致）为 PASS/FAIL 门槛，OpenCV 仅作信息性交叉验证。

**Q2：灰度图和 RGB 图处理路径有何不同？**
代码路径完全统一：RGB 采用"三通道 patch 距离求和、共享权重"（OpenCV colored 语义），灰度是 C=1 的退化情形。图像解码时 RGBA/灰度+alpha 会自动丢弃 alpha。

**Q3：patch_radius/search_radius 能设多大？**
解析器允许 pr≤8、sr≤32。smem 需求 = `c×(16+2(sr+pr))×(17+2(sr+pr))×4B`，超出设备动态 smem 上限时自动回退 naive（功能正确、性能下降）。基线 pr=3/sr=10 仅需约 21 KB（RGB）。

**Q4：输出图像为什么和 OpenCV fastNlMeansDenoising 略有差异？**
见 Q1。此外 OpenCV 彩色版对亮度和色度使用不同 h，本实现对三通道使用联合距离与共享权重。

**Q5：为什么多次运行结果完全一样？**
设计使然：无原子加、无归约竞争，累加顺序固定，输出 bit 级确定，便于回归验证。

**Q6：`FAST_EXP=1` 什么时候该用？**
实测无性能收益（瓶颈非 exp），默认不建议开启；保留该开关是为报告中"查表/快速数学近似"误差来源分析提供对照。若开启，validate 的 MAE 仍在阈值内（<5e-5），但最终以你们目标平台实测为准。

**Q7：V3（预留版本）报"未实现"？**
`--kernel 3` 为架构预留的进阶版本（积分图近似/多像素复用/warp 协作），当前返回明确错误提示，属预期行为。

**Q8：能处理 16 位图或视频流吗？**
当前仅支持 8 位 PNG/JPG（任务要求）。pipeline 接口为无状态纯函数，外层包帧循环即可扩展为视频/交互式处理；16 位需扩展 image_io 与转换 kernel。

---

## 9. 故障排除

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `make: nvcc: command not found` | CUDA Toolkit 未安装/未入 PATH | 训练营服务器按算力文档配置；本地安装 CUDA ≥ 11.0 |
| `图像解码失败 (can't open file)` | 路径错误 / /tmp 被清空 | 检查路径；WSL 的 /tmp 是 tmpfs，重启后需重新 `gen` |
| `仅支持灰度或 RGB 图像` | 输入为 CMYK/16 位 PNG 等 | 转换为 8 位灰度或 RGB |
| `未知参数 key` / `超出允许范围` | 参数文件拼写或取值错误 | 对照 params/default.txt 修正 |
| `kernel version=3 未实现` | 使用了预留版本 | 用 `--kernel 0/1/2` |
| GPU 报错 `[GPU] ... @ kernels_impl.inl` | 驱动/显存异常 | 错误信息含 CUDA 原始描述与行号；显存不足时换更小分辨率验证 |
| V2 比 V1 慢 | pr≥4 展开回归（旧代码） | 当前代码已自动回退；确认源码为修正后版本 |
| 国产平台编译失败 | 未切 PLATFORM / 环境未配 | 按算力文档配置后 `make PLATFORM=metax|moore|iluvatar` |
| Windows 上 nvcc 报 `Host compiler targets unsupported OS` | Windows nvcc 仅支持 MSVC，不支持 MinGW g++ | 安装 MSVC，或用 WSL + `scripts/build_wsl.sh`（开发机用法，服务器无需） |

**诊断工具**：
- `./build/nlm_denoise units` —— 先排除 Host 逻辑问题（12 项全过则参数/指标/算法不变式正常）；
- `validate` 退出码 —— 0/1/2 区分"通过/质量不达标/运行错误"；
- `--log /dev/null` —— 性能测试时关闭日志落盘干扰；
- `ncu`/`nsys` 脚本 —— 瓶颈定位（占用率、bank conflict、时间线气泡）。

---

## 10. 附录

### 10.1 目录结构

```text
nlm_denoise_project/
├── Makefile                    # 多平台构建（目标语义对齐参考仓库）
├── LICENSE                     # MIT
├── README.md                   # 快速上手
├── nvidia_result.txt           # NVIDIA 平台实测结果（真实输出）
├── metaX_result.txt            # 沐曦结果（待实测占位）
├── moore_result.txt            # 摩尔结果（待实测占位）
├── bench.csv                   # 48 组实测性能数据
├── include/                    # 全部头文件（nlm/kernels/pal/core/tester 五层）
├── src/                        # Host 实现：main / pipeline / params / image_io / CPU 参考
├── kernels/                    # 设备算子：common 共享实现 + 四平台架构子目录
├── tester/                     # validate / benchmark / test_units
├── build/                      # 构建产物（nlm_denoise + obj/，make clean 清空）
├── params/default.txt          # 任务基线参数
├── data/                       # 数据集中管理（noisy/clean/output/logs 四类）
├── scripts/                    # run_all / gen_result / ncu / nsys / WSL 开发辅助
├── docs/                       # architecture_design / user_guide / refactor_log
└── third_party/stb/            # stb_image / stb_image_write
```

### 10.2 退出码约定

| 码 | 含义 |
|---|---|
| 0 | 成功（validate 判定 PASS） |
| 1 | validate 判定 FAIL（质量不达标） |
| 2 | 运行错误（输入非法 / GPU 错误 / 用法错误） |

### 10.3 典型工作流示例

```bash
# 场景：拿到一张新的 4K 含噪图，需要评估质量与延迟
make
./build/nlm_denoise validate -i scene_4k.png -o data/output/scene_4k_denoised.png -p params/default.txt
#   → 看三版本 MAE/PSNR 是否 PASS（正确性）
./build/nlm_denoise run -i scene_4k.png -o /dev/null -p params/default.txt --kernel 2 --log /dev/null
#   → 看 kernel/e2e ms（延迟）；再换 --kernel 1 / small 参数对比权衡
./build/nlm_denoise bench --sizes 3840x2160 --channels 3 --log mybench.csv
#   → 得到该机型 4K 全参数组合延迟表，用于质量-延迟权衡分析
```
