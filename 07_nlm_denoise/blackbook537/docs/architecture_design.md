# 实时图像非局部均值降噪（CUDA）—— 项目开发架构设计文档

| 项目代号 | SkyEye-NLM（"巡天"深空探测影像机载降噪模块） |
|---|---|
| 文档版本 | v1.0 |
| 适用阶段 | 2026-summer-project 项目阶段 |
| 目标平台 | NVIDIA（默认）/ 天数智芯 Iluvatar / 沐曦 MetaX / 摩尔线程 Moore |
| 依据文档 | 《七.实时图像非局部均值降噪（CUDA）》 |

---

## 1. 任务需求基线（Requirements Baseline）

本节严格摘录并量化任务文档中的全部强制性指标，作为架构设计的验收基线，后续所有设计决策均不得偏离。

### 1.1 功能性需求

| 编号 | 需求 | 量化指标 |
|---|---|---|
| FR-1 | 输入图像 | 标准 PNG/JPG，灰度单通道或 RGB 三通道，像素值 [0,255] 整数，处理时转浮点 |
| FR-2 | 参数文件 | 文本格式，含 `patch_radius`、`search_radius`、`h`、`sigma` 四项 |
| FR-3 | 参数范围 | `patch_radius ≥ 3`（patch 边长 ≥ 7），`search_radius ≥ 10`（搜索窗边长 ≥ 21）；`h`、`sigma` 可配置 |
| FR-4 | 核心算法 | NLM 权重 `w(p,q) = exp(-max(dist(P_p,P_q) - 2σ², 0) / h²)` |
| FR-5 | 输出 | 与输入同格式的降噪图像 + 性能日志（耗时 ms、吞吐量 Mpx/s、对 CPU/OpenCV 参考的加速比） |
| FR-6 | 正确性验证 | 与 CPU 参考或 OpenCV `fastNlMeansDenoising(Colored)` 对比，报告 MAE / PSNR；若用近似算法须说明误差来源 |

### 1.2 性能与平台需求

| 编号 | 需求 | 量化指标 |
|---|---|---|
| NFR-1 | 基准性能测试 | 至少 1920×1080，给出不同参数组合下的运行时间 |
| NFR-2 | 进阶目标 | 4K（3840×2160）图像的交互式处理（目标 ≤ 100 ms/帧量级，争取接近实时） |
| NFR-3 | 平台适配 | NVIDIA 必须支持；每多适配一款国产平台（天数/沐曦/摩尔）额外加分 |
| NFR-4 | 可分析性 | 程序需支持性能瓶颈定位（提供 ncu / nsys 分析流程与结论，属加分项） |

### 1.3 计算复杂度评估（设计前提）

以基线参数 `patch_radius=3`（patch 49 px）、`search_radius=10`（搜索窗 441 候选）估算：

- 每像素权重计算量 ≈ 441 × 49 ≈ 2.16 万次乘加 + 441 次 `expf`；
- 1080p（2.07 Mpx）单帧 ≈ 4.5×10¹⁰ FLOPs，4K（8.29 Mpx）≈ 1.8×10¹¹ FLOPs；
- **结论：算法为典型 compute-bound + 访存高度冗余型负载，GPU 并行化与数据复用（shared memory 缓存）是设计核心。** 朴素 CPU 单线程实现 1080p 预计在秒级~十秒级，GPU 目标加速比 ≥ 100×。

---

## 2. 系统总体架构

### 2.1 分层架构视图

系统采用经典五层分层架构，自底向上为：

```text
┌───────────────────────────────────────────────────────────────┐
│ L5 应用层 (Application)                                        │
│   nlm_denoise CLI  │  benchmark 工具  │  validate 验证工具      │
├───────────────────────────────────────────────────────────────┤
│ L4 流程编排层 (Pipeline / Orchestration)                       │
│   DenoisePipeline：decode→preprocess→NLM→postprocess→encode    │
│   + Timer/Logger：耗时统计、吞吐量计算、性能日志落盘              │
├───────────────────────────────────────────────────────────────┤
│ L3 算子层 (Kernel / Operator)                                  │
│   NLM Kernel（多版本：naive → smem → vectorized → 优化版）       │
│   辅助 kernel：u8→f32 转换、f32→u8 量化、（彩色）通道规划          │
├───────────────────────────────────────────────────────────────┤
│ L2 平台抽象层 (PAL, Platform Abstraction Layer)                │
│   统一设备 API：malloc/memcpy/launch/event/error 宏封装          │
│   NVIDIA CUDA │ Iluvatar CoreX │ MetaX MACA │ Moore MUSA      │
├───────────────────────────────────────────────────────────────┤
│ L1 基础设施层 (Infrastructure)                                 │
│   图像 I/O（stb_image / OpenCV）│ 参数解析 │ 参考CPU实现 │ 测试    │
└───────────────────────────────────────────────────────────────┘
```

### 2.2 设计原则

1. **正确性优先**：每个优化版本 kernel 必须通过与 naive 版本及 OpenCV 参考的 MAE/PSNR 校验后方可合入（与评分规则一致）。
2. **单一代码基线，多平台编译**：算法核心以 CUDA 语法编写，通过 PAL 层宏与文件后缀（`.cu/.maca/.mu`）适配国产平台，与 Learning-CUDA 仓库的多平台组织方式保持一致。
3. **关键计算全部在 GPU**：u8↔f32 转换、NLM 主体、后处理量化均在设备侧完成；Host 仅做 I/O、参数解析与调度。
4. **可配置、可复现**：所有参数走参数文件 + CLI 覆盖；benchmark 固定 warmup/repeat 次数，输出机器可读日志。
5. **零外部 kernel 库依赖**：NLM 主体不使用任何现成库函数实现；OpenCV 仅用于参考对照与图像读写（可选降级为 stb_image）。

---

## 3. 模块划分

### 3.1 源码目录结构

```text
nlm_denoise_project/
├── Makefile                      # 多平台构建（目标语义对齐 Learning-CUDA 仓库）
├── LICENSE                       # MIT（与目标仓库一致）
├── README.md
├── nvidia_result.txt             # NVIDIA 平台实测结果（真实运行输出）
├── metaX_result.txt              # 沐曦结果（待实测占位）
├── moore_result.txt              # 摩尔结果（待实测占位）
├── bench.csv                     # 48 组实测性能数据
├── docs/
│   ├── architecture_design.md    # 本文档
│   ├── user_guide.md             # 讲解与使用文档
│   └── refactor_log.md           # 结构重构变更记录
├── include/                      # 全部头文件（按接口层次分类，统一 -Iinclude）
│   ├── nlm/                      #   对外接口层：pipeline.h / params.h
│   ├── kernels/                  #   算子接口层：kernels.h
│   ├── pal/                      #   平台抽象层：platform_api.h（差异唯一收敛点）
│   ├── core/                     #   公共支撑层：image_io.h / nlm_cpu_ref.h
│   └── tester/                   #   测试支撑层：utils.h
├── src/                          # Host 侧实现（与 include/ 层次一一对应）
│   ├── main.cpp                  #   CLI 入口：子命令 dispatch
│   ├── image_io.cpp              #   图像读写（stb，可选 OpenCV）
│   ├── params.cpp                #   参数文件解析与校验
│   ├── pipeline.cpp              #   DenoisePipeline 流程编排 + 计时 + 日志
│   └── nlm_cpu_ref.cpp           #   CPU 参考实现（正确性基线 + 加速比基准）
├── kernels/                      # 设备侧算子（按 GPU/NPU 架构分类）
│   ├── common/kernels_impl.inl   #   全部设备实现（多平台 100% 共享）
│   ├── nvidia/kernels.cu         #   NVIDIA CUDA 编译单元
│   ├── iluvatar/kernels.cu       #   天数 CoreX 编译单元（CUDA 兼容语法）
│   ├── metax/kernels.maca        #   沐曦 MACA 适配（可选加分项）
│   └── moore/kernels.mu          #   摩尔线程适配（可选加分项，C++11 子集）
├── tester/
│   ├── validate.cpp              # MAE/PSNR 校验工具（对比 CPU 参考，可选 OpenCV）
│   ├── benchmark.cpp             # 参数扫描 × 分辨率扫描性能测试
│   └── test_units.cpp            # Host 侧单元测试
├── params/
│   └── default.txt               # 任务文档示例参数
├── scripts/
│   ├── run_all.sh                # 一键：构建→校验→基准→生成日志
│   ├── gen_result_wsl.sh         # {platform}_result.txt 生成（可复现）
│   ├── ncu_profile.sh            # ncu 指标采集脚本
│   ├── nsys_profile.sh           # nsys 时间线采集脚本
│   └── *_wsl.sh                  # 本地开发辅助（无系统 CUDA 环境）
├── build/                        # 全部构建产物（可执行文件 + obj/，clean 整体删除）
├── data/                         # 数据集中管理（noisy 输入 / clean 参考 / output 结果 / logs 日志）
└── third_party/stb/              # stb_image / stb_image_write（单头文件）
```

> 测试图像不落盘存储：`gen` 子命令现场合成（seed 固定、完全可复现），
> 避免仓库携带大体积二进制素材。

### 3.2 模块职责与依赖

| 模块 | 职责 | 依赖 | 关键约束 |
|---|---|---|---|
| `main` | 解析 CLI（`run / validate / bench` 子命令），组装参数 | pipeline, params | 不做任何计算 |
| `image_io` | PNG/JPG 解码为 `uint8` 缓冲；编码输出同格式 | stb_image（单头文件） | 保留原始通道数；灰度/RGB 均支持 |
| `params` | 解析 `key = value` 文本，校验范围 | 无 | 缺省值与任务文档示例一致 |
| `pipeline` | 设备内存管理、H2D/D2H、kernel 调度、CUDA event 计时、性能日志 | platform_api, kernels | 唯一允许持有设备资源的模块 |
| `kernels.*` | NLM 各优化版本 kernel + 数据转换 kernel | platform_api | 纯设备代码，无 Host I/O |
| `nlm_cpu_ref` | 与 GPU 语义严格一致的 CPU 实现 | 无 | 作为正确性 ground truth 与加速比基线 |
| `validate` | 三方对比：GPU vs CPU-ref vs OpenCV，输出 MAE/PSNR | OpenCV(imgcodecs/photo) | 通过阈值：MAE ≤ 2.0，PSNR ≥ 30 dB（建议） |
| `benchmark` | 分辨率 × 参数矩阵扫描，CSV/文本日志 | pipeline | 固定 warmup=3，repeat=10，取均值 |

---

## 4. 技术栈选型

| 领域 | 选型 | 备选 | 选型理由 |
|---|---|---|---|
| 主语言 | CUDA C++17（NVIDIA，nvcc） | — | 任务默认平台；与 Learning-CUDA 环境一致（CUDA Toolkit ≥ 11.0） |
| 国产平台编译 | clang++（Iluvatar CoreX，`-x ivcore`）、mxcc（MetaX MACA）、mcc（Moore MUSA，C++11 子集） | — | 与训练营算力文档及 Makefile 的 `PLATFORM` 切换机制完全一致 |
| 图像编解码 | stb_image / stb_image_write（单头文件） | OpenCV `cv::imread/imwrite` | 零依赖、易交叉编译到国产平台；OpenCV 仅在 validate/bench 主机工具中使用 |
| 参考实现 | 自研 CPU NLM + OpenCV `fastNlMeansDenoising(Colored)` 双基线 | — | 任务要求允许二者其一；双基线可交叉验证 OpenCV 自身近似带来的偏差 |
| 构建系统 | GNU Make（沿用 `PLATFORM=` 切换约定） | CMake | 与训练营提交要求及既有仓库一致，降低评审环境风险 |
| 性能分析 | Nsight Compute (ncu) + Nsight Systems (nsys) | nvprof（已弃用） | 任务加分项要求 |
| 版本控制 | Git，提交至 Learning-CUDA `2026-summer-project` 分支 | — | 任务提交要求 |
| 精度 | 计算全程 FP32；存储/IO 为 UINT8 | FP16 加速（见 §8 权衡） | FP32 保证与 CPU/OpenCV 参考的 PSNR 可比性 |

---

## 5. 数据流程设计

### 5.1 端到端数据流（`run` 子命令）

```text
输入PNG/JPG ──► [Host] stb_image 解码 (uint8, H×W×C)
                    │
                    ▼
[Host] 参数文件解析 ──► Params{pr, sr, h, sigma}（校验 pr≥3, sr≥10）
                    │
                    ▼
[Device] cudaMalloc + H2D 拷贝 d_src_u8
                    │
                    ▼
[Device] cvt_u8_to_f32 kernel ──► d_src_f32 (planar: 每通道 H×W)
                    │
                    ▼
[Device] NLM kernel（主计算，逐通道或联合）
   对每像素 p：遍历搜索窗内 q ──► patch L2 距离 dist
   ──► w = expf(-max(dist - 2σ²·Npatch, 0) / h²')  ──► Σw·I(q), Σw
                    │
                    ▼
[Device] normalize + cvt_f32_to_u8 kernel ──► d_dst_u8
                    │
                    ▼
[Host] D2H 拷贝 ──► stb_image_write 编码输出（同输入格式）
                    │
                    ▼
[Host] 性能日志：kernel 耗时(cudaEvent)、端到端耗时、Mpx/s、
       CPU 参考耗时、加速比 ──► stdout + data/logs/nlm_perf.log
```

### 5.2 内存布局决策

- **Planar（通道分离）存储**：`d_src_f32[c][y][x]`，各通道独立连续。
  - 理由：NLM 对 RGB 的标准做法之一是将 patch 距离按通道求和（联合 patch 距离），planar 布局下每通道访问均为连续 coalesced；interleaved (HWC) 会导致每线程跨 3 倍步长访存。
  - 灰度图等价于 C=1 的退化情形，代码路径统一。
- **边界处理**：采用 **clamp-to-edge（镜像亦可，但须与 CPU 参考保持一致并写进报告）**。所有边界逻辑收敛到一个内联函数 `idx = clamp(i, 0, H-1)`，避免 kernel 内分支扩散。
- **浮点转换**：`I_f32 = (float)I_u8`（不做 0–1 归一化），保证 `h`、`sigma` 参数与任务文档（0–255 域）语义一致。

### 5.3 计时口径（写入性能日志）

| 指标 | 测量方式 |
|---|---|
| Kernel 纯计算时间 | `cudaEventRecord` 包裹 NLM kernel，多版本分别报告 |
| 端到端 GPU 时间 | 含 H2D/D2H 与转换 kernel |
| 吞吐量 | `H×W / kernel_time`（Mpx/s），另报含 IO 的端到端吞吐 |
| 加速比 | `t_cpu_ref / t_gpu_e2e`（同时给出对 OpenCV 参考耗时的对比） |

---

## 6. 接口设计规范

### 6.1 参数文件格式（严格遵循任务文档）

```text
patch_radius = 3          # patch 半径，实际 patch 大小为 (2*patch_radius+1)^2
search_radius = 10        # 搜索窗口半径
h = 10.0                  # 滤波强度
sigma = 25.0              # 噪声标准差估计，8位图像范围 0-255
```

解析规则：`key = value`，`#` 后注释忽略，未知 key 报错；缺项用默认值（即上表数值）。

### 6.2 CLI 接口

```bash
nlm_denoise run      -i input.png -o output.png -p params/default.txt [--kernel 0|1|2] [--log data/logs/nlm_perf.log]
nlm_denoise validate -i input.png -o output.png -p params/default.txt
nlm_denoise bench    --sizes 1920x1080,3840x2160 --channels 1,3 --log bench.csv
```

### 6.3 核心 C/C++ API

```cpp
// params.h —— 参数结构与解析
struct NlmParams {
    int   patch_radius  = 3;    // ≥ 3（强制校验）
    int   search_radius = 10;   // ≥ 10（强制校验）
    float h             = 10.0f;
    float sigma         = 25.0f;
};
bool ParseParams(const char* path, NlmParams* out, std::string* err);

// image_io.h —— 图像对象（Host 侧）
struct ImageU8 {
    int width = 0, height = 0, channels = 0;   // channels ∈ {1, 3}
    std::vector<uint8_t> data;                 // interleaved，与文件格式一致
};
bool LoadImage(const char* path, ImageU8* out, std::string* err);
bool SaveImage(const char* path, const ImageU8& img, std::string* err);

// pipeline.h —— GPU 降噪流程（对调用者隐藏全部设备细节）
struct NlmPerfReport {
    float kernel_ms, e2e_ms, h2d_ms, d2h_ms;
    float throughput_mpx_s;
    float cpu_ref_ms, speedup_vs_cpu;
};
bool NlmDenoiseGpu(const ImageU8& src, ImageU8* dst,
                   const NlmParams& params,
                   int kernel_version,            // 0=naive,1=smem,2=vec,3=final
                   NlmPerfReport* perf, std::string* err);

// nlm_cpu_ref.h —— CPU 参考（ground truth）
void NlmDenoiseCpuRef(const ImageU8& src, ImageU8* dst, const NlmParams& params);

// include/tester/utils.h —— 质量指标
float ComputeMAE (const ImageU8& a, const ImageU8& b);
float ComputePSNR(const ImageU8& a, const ImageU8& b);
```

### 6.4 Kernel 接口（`include/kernels/kernels.h`，实现在 `kernels/common/kernels_impl.inl`）

```cpp
// 数据转换
__global__ void CvtU8ToF32Planar(const uint8_t* __restrict__ in,
                                 float* __restrict__ out,   // planar C×H×W
                                 int w, int h, int c);
__global__ void CvtF32ToU8Interleaved(const float* __restrict__ in,
                                      uint8_t* __restrict__ out,
                                      int w, int h, int c);

// NLM 主体：输入 planar f32，输出 planar f32（未做 u8 量化，便于校验）
__global__ void NlmNaive(const float* __restrict__ src, float* __restrict__ dst,
                         int w, int h, int channels, NlmParamsDev p);
__global__ void NlmSmem (const float* __restrict__ src, float* __restrict__ dst,
                         int w, int h, int channels, NlmParamsDev p);
__global__ void NlmFinal(const float* __restrict__ src, float* __restrict__ dst,
                         int w, int h, int channels, NlmParamsDev p);
```

约定：
- 所有 kernel 参数含 `__restrict__`；设备侧参数打包为 `NlmParamsDev`（含预计算的 `1/h²`、`2σ²` 等，避免 kernel 内重复计算）。
- 网格映射统一为 **1 thread = 1 output pixel**，`blockDim = (16,16)`，`gridDim = (⌈W/16⌉, ⌈H/16⌉)`；通道在线程内循环（planar 布局，灰度/RGB 统一路径）。
- 所有设备 API 调用经 PAL 层 `GPU_CHECK()` 宏封装（含 `cudaGetLastError()` 后置检查），错误以 `std::string` 向上传递。

### 6.5 PAL 平台抽象层（`platform_api.h`）

```cpp
#if defined(PLATFORM_NVIDIA)
  #include <cuda_runtime.h>
  #define GPU_API                       cuda
  using gpuError_t = cudaError_t;
  // ... 类型别名与内联封装
#elif defined(PLATFORM_METAX)   // MACA：API 与 CUDA 基本同构
  // mcMalloc / mcMemcpy / ... 经宏映射
#elif defined(PLATFORM_MOORE)   // MUSA
  // musaMalloc / ...（注意 C++11 子集：禁用 if constexpr、结构化绑定）
#elif defined(PLATFORM_ILUVATAR)
  // CoreX：经 clang++ -x ivcore 编译，运行时 API 映射
#endif

GPU_MALLOC(ptr, bytes)  GPU_MEMCPY(dst, src, bytes, kind)
GPU_EVENT_CREATE(e)     GPU_EVENT_RECORD(e, stream)  GPU_EVENT_ELAPSED(ms, e0, e1)
GPU_LAUNCH(kernel, grid, block, args...)   GPU_CHECK_LAST()
```

适配策略：**算法代码 100% 共享**，仅各平台编译单元做编译期包装（`kernels/{nvidia,iluvatar,metax,moore}/kernels.*` 一行 `#include "../common/kernels_impl.inl"`），差异全部收敛于 `include/pal/platform_api.h`。摩尔线程注意其 C++11 限制，核心实现只使用 C++11 子集特性。

---

## 7. 安全与健壮性策略

本项目为离线/机载批处理程序，无网络暴露面，"安全"重点为**内存安全、输入健壮性与结果可信性**：

1. **输入校验**：图像解码失败、通道数 ∉ {1,3}、尺寸越界（>16384×16384）、参数负值/超范围一律拒绝并给出诊断信息；`patch_radius<3`、`search_radius<10` 视为配置错误（任务基线要求）。
2. **显存安全**：所有 `cudaMalloc` 前估算需求（1080p RGB f32 planar ≈ 25 MB，4K ≈ 100 MB，多版本缓冲 ≤ 2 倍），失败时回滚已分配资源；进程退出前统一 `cudaFree` + `cudaDeviceSynchronize`。
3. **数值健壮性**：权重和 `Σw` 恒 ≥ 1（自身权重 w(p,p)=exp(0)=1），归一化无除零风险；`expf` 输入恒 ≤ 0，无上溢；最终量化做 `round + clamp[0,255]`。
4. **确定性**：同一输入 + 同一参数 + 同一 kernel 版本，输出 bit 级一致（单 kernel 内无原子加；归约仅在线程私有寄存器内进行），保证验证可复现。
5. **防越界**：搜索窗/patch 的边界 clamp 统一经 `__device__ __forceinline__ int clampi(...)`，CPU 参考与 GPU 使用完全相同的边界语义，杜绝"验证通过但语义不同"的隐性 bug。
6. **学术诚信**：NLM 主体不调用任何现成库实现，符合训练营注意事项。

---

## 8. 性能优化方案（渐进式 Roadmap）

遵循"先正确、后优化、每步可验证"原则，共 4 个 kernel 版本，每版给出预期收益与验证门槛。

### V0 —— Naive 基线（正确性锚点）

- 1 thread/pixel，三层循环（搜索窗 dy,dx × patch dy,dx），全部走全局内存。
- 用途：正确性 ground truth（与 CPU 参考逐像素对比，要求 MAE = 0 或 < 0.5）。
- 预期性能：1080p RGB 数百 ms 级，作为加速比基线。

### V1 —— Shared Memory 块缓存（核心优化，预期 4–10×）

- **思路**：block 计算 `(16,16)` 输出 tile，其搜索窗+p patch 覆盖的输入区域为
  `(16 + 2·(sr+pr))² = (16+26)² = 42²`（sr=10, pr=3）。将该 halo 区域协同加载进 shared memory，patch 距离全部命中 smem。
- **要点**：
  - halo 加载采用线性化索引 + 边界 clamp，保证 coalesced；
  - smem 需求：42² × 4B ≈ 7 KB/通道/block，远低于 48 KB 静态上限，多 block 常驻无压力；
  - 对 `search_radius` 动态参数，smem 尺寸按运行期参数 `dynamic shared memory` 分配（`extern __shared__`），支持 sr>10。
- 验证：与 V0 对比 MAE < 0.5。

### V2 —— 访存与指令级优化（预期再 1.5–3×）

- 搜索窗内层按行扫描，利用 smem bank 分布避免冲突（tile 行 stride 加 padding `+1`）。
- 预计算并常量缓存：将 `NlmParamsDev` 放 `__constant__` 或以 `__grid_constant__` 传参。
- `expf` 替换为 `__expf`（快速数学）——**作为独立开关**：快速数学带来的 MAE 增量写入报告误差来源分析；若 MAE 超阈值则保留 `expf`。
- `#pragma unroll` 展开 patch 内层循环（pr≤3 时 7×7 完全展开），寄存器压力通过 `__launch_bounds__` 约束验证。
- 占用率调优：用 `cudaOccupancyMaxPotentialBlockSize` 选 block 形状，ncu 复核 achieved occupancy ≥ 50%。

### V3 —— 进阶优化（按 4K 交互式目标选做，预期再 1.5–4×）

按投入产出排序，逐项接入并独立验证：

1. **向量化与多像素/thread**：1 thread 计算 1×2/2×1 相邻像素，共享搜索窗内大部分 patch 数据，寄存器内复用；`float4` 加载 smem 行。
2. **积分图/盒式滤波近似（可选近似路径）**：patch L2 距离可用积分图 O(1) 求得（误差来源：clamp 边界与浮点累加），计算量从 O(sr²·pr²) 降为 O(sr²)；作为 `--kernel v3 --approx` 可选路径，报告中对比 MAE/PSNR 退化。
3. **Warp 级协作**：搜索窗按 warp 内 lane 切分，warp shuffle 归约权重和，减少 smem 往返。
4. **查表近似 exp**：权重函数关于 dist 单调，可对 dist 量化后查表（shared memory LUT），权衡精度报告。
5. **4K 专用调度**：大分辨率下显存带宽充裕、计算为主，采用 stream 分片处理 ROI， overlap H2D/D2H（为交互式场景铺路）。

### 优化总表

| 版本 | 主要手段 | 预期累计加速（vs CPU 单线程） | 验证门槛 |
|---|---|---|---|
| V0 | 直接移植 | 20–60× | 与 CPU ref MAE < 0.5 |
| V1 | smem halo 缓存 | 100–400× | vs V0 MAE < 0.5 |
| V2 | unroll/常量/快速数学/占用率 | 200–800× | vs V0 MAE < 1.0（含 __expf 误差说明） |
| V3 | 多像素复用/积分图近似/LUT | 400× 以上，1080p 趋近 <30 ms，4K <150 ms | 近似路径单独报 MAE/PSNR 与误差分析 |

### ncu / nsys 分析计划（加分项）

- **ncu**：对 V0→V2 各采 `SpeedOfLight`、`MemoryWorkloadAnalysis`、`Occupancy` 三组指标，给出 compute/memory throughput 占比、L2 命中率、smem bank conflict 计数、achieved occupancy，形成"瓶颈—优化—指标变化"闭环表格写入报告。
- **nsys**：对 `bench` 全链路采时间线，确认 kernel 占端到端时间比 > 90%，H2D/D2H 无异常气泡；4K 分片版本验证 stream overlap 效果。

---

## 9. 扩展性设计

1. **参数扩展**：`Params` 结构预留版本号字段；新增参数（如 `boundary_mode`、`approx_mode`）向后兼容，解析器对未知 key 的策略可切换（strict/warn）。
2. **kernel 版本注册表**：`pipeline` 内以 `{version_id → launch 函数指针}` 注册表调度，新增优化版本只需注册一行，CLI `--kernel vN` 即生效，旧版本保留用于回归对比。
3. **平台扩展**：新增国产平台仅需实现 `platform_api.h` 中一组宏 + Makefile 增加一个 `PLATFORM` 分支，算法文件零改动（已通过 C++11 子集约束保证摩尔线程兼容）。
4. **分辨率/通道扩展**：grid 三维映射天然支持任意分辨率与 C∈{1,3}；若未来支持 RGBA，仅需在 `image_io` 与转换 kernel 增加 C=4 分支。
5. **数据类型扩展**：转换 kernel 与 NLM kernel 以模板参数 `T` 预留 FP16 路径（精度验证流程同 FP32），为后续 tensor core/半精度加速留口。
6. **交互式/流式扩展**：pipeline 的 `NlmDenoiseGpu` 已是无状态纯函数接口，外层可直接包装为帧循环；stream + 固定内存（pinned memory）接口在 PAL 预留宏位。

---

## 10. 测试与验证方案

### 10.1 正确性测试矩阵

| 维度 | 取值 |
|---|---|
| 分辨率 | 64×64（边界密集）、512×512、1920×1080、3840×2160 |
| 通道 | 灰度 ×1，RGB ×3 |
| 参数 | (pr=1,sr=3) 快速用例、(pr=3,sr=10) 基线、(pr=4,sr=21) 压力用例 |
| 噪声 | 干净图 + 合成高斯噪声（σ=10/25/50）|
| 对比基线 | CPU 自研参考（主）、OpenCV `fastNlMeansDenoising(Colored)`（辅） |

通过阈值（写入 tester）：**vs CPU 参考 MAE ≤ 0.5（同语义实现应接近 0）；vs OpenCV MAE ≤ 2.0 且 PSNR ≥ 30 dB**。近似路径单独建档说明误差来源。

### 10.2 性能测试规程

- warmup 3 次 + repeat 10 次取均值，报告 min/mean；
- 固定 GPU  clocks 说明（若环境允许 `nvidia-smi -lgc`，在报告中注明）；
- 输出 CSV：`size, channels, pr, sr, h, sigma, kernel_ver, kernel_ms, e2e_ms, mpx_s, cpu_ms, speedup`；
- 参数组合覆盖任务基线 (3,10,10,25) 及至少 4 组变体 × 1080p/4K × 灰度/RGB。

### 10.3 回归机制

`scripts/run_all.sh` 一键执行：构建 → validate（失败即退出非零）→ bench → 汇总日志，作为每次提交前的准入检查。

---

## 11. 开发与部署流程

### 11.1 开发流程（里程碑制）

```text
M1  工程骨架：Makefile 多平台框架、params/image_io/CLI 打通、stb 编解码验证
M2  正确性基线：nlm_cpu_ref + V0 naive kernel + validate 工具，1080p 跑通 MAE/PSNR
M3  核心优化：V1 smem + V2 指令级优化，benchmark 矩阵跑通，ncu 第一轮分析
M4  进阶冲刺：V3 选做项 + 4K 调优，nsys 全链路分析，性能日志定型
M5  平台扩展：按优先级 MetaX → Moore → Iluvatar 适配验证（Makefile PLATFORM 分支）
M6  交付整理：报告（思路/优化历程/质量与性能指标/ncu-nsys 分析/未来工作）、
    代码清理、提交至 Learning-CUDA 2026-summer-project 分支
```

### 11.2 构建与部署

```bash
# NVIDIA（默认）；目标语义同参考仓库：make = build + run tests
make                            # 构建并运行单元测试
make run VERBOSE=true           # 附加 GPU 端到端正确性校验
./build/nlm_denoise run -i data/noisy/noisy.png -o data/output/denoised.png -p params/default.txt

# 国产平台（与 Learning-CUDA 约定一致）
make build PLATFORM=metax       # mxcc，编译 kernels/metax/kernels.maca
make build PLATFORM=moore       # mcc -std=c++11，编译 kernels/moore/kernels.mu
make build PLATFORM=iluvatar    # clang++ -x ivcore

# 验证与基准
./build/nlm_denoise validate ... && ./build/nlm_denoise bench ...
```

部署形态：单可执行文件 + 参数文件 + 测试图像，无外部运行时依赖（stb 全静态编译；OpenCV 仅 validate/bench 需要，可条件编译 `-DWITH_OPENCV` 剥离）。

### 11.3 环境要求

- NVIDIA：CUDA Toolkit ≥ 11.0，GPU 计算能力 ≥ 7.0（V100/T4 及以上优先；Ampere/Hopper 直接可用）；
- 天数：BI-150 CoreX 环境；沐曦：标准 MACA 环境；摩尔：MUSA 环境（C++11）；
- 主机：Linux x86_64，GNU Make，C++17 编译器。

---

## 12. 风险与权衡分析

| 风险/权衡 | 影响 | 对策 |
|---|---|---|
| 搜索窗 sr 增大导致 smem 溢出（如 sr=21 → halo 64²×4B=16 KB，仍可；sr 更大时） | V1 无法编译/启动 | 动态 smem + 启动前检查 `cudaFuncAttributeMaxDynamicSharedMemorySize`，超限自动回退全局内存路径 |
| `__expf`/LUT 近似降低 PSNR | 验证不达标 | 近似全部做成独立开关，默认精确路径；报告中量化误差来源 |
| OpenCV 参考与任务公式语义差异（OpenCV 使用预计算模板与不同归一化） | MAE 虚高 | 以自研 CPU 参考为主基线；报告中说明两者公式差异，仅将 OpenCV 作为交叉验证 |
| RGB 联合 patch 距离 vs 逐通道独立 NLM 的语义选择 | 与参考对比口径 | 默认实现"三通道 patch 距离求和、共享权重"（与 OpenCV colored 一致），并在报告中声明 |
| 摩尔线程 C++11 限制 | 编译失败 | 核心代码只用 C++11 子集；CI 中 `PLATFORM=moore` 做编译检查 |
| 寄存器压力导致 occupancy 塌陷（V2/V3） | 性能反降 | `__launch_bounds__` + ncu 复核，每项优化以实测数据决定去留 |

---

## 13. 交付物清单（对照任务"需提交内容"）

1. **完整程序**：本结构全部源码（含 tester 与 scripts），提交至 Learning-CUDA `2026-summer-project` 分支；
2. **测试**：validate 工具 + 测试矩阵结果（MAE/PSNR 表）；
3. **性能日志**：benchmark CSV/文本，含 1080p/4K、多参数组合、CPU/OpenCV 加速比；
4. **总结报告**：实现思路与优化历程、质量指标（MAE/PSNR）、性能指标（ms、Mpx/s、加速比）、ncu/nsys 分析、未来可提升方向（积分图近似调优、FP16/张量核探索、流式管线、更多国产平台）。

---

## 附录 A：权重公式实现口径（防歧义）

任务公式：`w = exp(-max(dist - 2σ², 0)/h²)`。实现口径约定：

- `dist` 为两 patch 的 **L2 距离之和**（Σ(ΔI)²，不做均值），则减项为 `2σ²·N_patch`（N_patch = (2pr+1)²，RGB 时 ×3），与 OpenCV 及经典文献（Buades et al.）的 `h²` 标定保持一致；
- 该口径同时应用于 CPU 参考与全部 GPU kernel，并在报告中以公式形式明确写出，避免验证口径争议。
