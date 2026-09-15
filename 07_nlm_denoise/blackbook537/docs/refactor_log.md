# 结构重构变更记录（refactor_log）

| 项 | 内容 |
|---|---|
| 重构目标 | 将项目目录结构对齐参考仓库 Learning-CUDA-2026 的组织约定 |
| 重构原则 | 骨架完全一致；功能性扩展目录保留为增量；零源码改动（仅构建/脚本/文档层） |
| 验证结论 | clean 重建成功；units 12/12；`make run VERBOSE=true` GPU 端到端校验 PASS（256×256 与 512×512 三版本 MAE=0）；bench 链路正常 |

## 1. 参考结构与映射结果

参考仓库 Learning-CUDA-2026 的结构约定：

```text
├── LICENSE / Makefile / README.md          # 根级三件套
├── {platform}_result.txt                   # 各平台运行结果（nvidia/metaX/moore）
├── src/kernels.{cu,maca,mu}                # 实现：平台后缀切换
└── tester/utils.h + tester_{platform}.o    # 测试：工具头 + 平台对象
```

映射对照（参考项 → 本项目）：

| 参考约定 | 本项目现状 | 处理 |
|---|---|---|
| 根级 LICENSE | 缺失 | **新增**：复制参考仓库 MIT LICENSE（提交目标仓库一致） |
| 根级 Makefile 目标语义（all=build+run、VERBOSE=true、clean） | 目标为 all/run(演示)/test/clean | **重写 Makefile** 对齐（见 §2） |
| 根级 nvidia_result.txt | 无（仅 bench.csv） | **新增**：真实运行输出生成（环境/测试/校验/基准） |
| 根级 metaX_result.txt / moore_result.txt | 无 | **新增**：诚实占位（"待实测"，附复现步骤） |
| src/kernels.{cu,maca,mu} 平台后缀 | 已一致 | 无改动 |
| tester/ 目录组织（utils.h + 测试逻辑） | 已一致（utils.h + 三个 cpp） | 无改动 |
| 参考的 tester_{platform}.o 为官方预编译测试壳 | 本项目 tester 为自研源码 | **不迁移**：语义不同（参考的 .o 是题目的预编译测试器，本项目自带完整测试体系，编译产物落位 tester/*.o） |

## 2. Makefile 重写明细

对齐参考的目标语义，同时保留本项目扩展能力：

| 目标 | 参考语义 | 本项目实现 |
|---|---|---|
| `make`（all） | 构建 + 运行测试 | `all: build run` |
| `make build` | 仅编译 | 编译 kernels + 全部 host/tester 模块并链接 |
| `make run` | 运行测试 | 运行 `./nlm_denoise units`（12 项单元测试） |
| `make run VERBOSE=true` | verbose 测试 | 追加 GPU 端到端校验（gen 256×256 + validate） |
| `make clean` | 删除临时文件 | 删除可执行文件与全部 .o |
| —（参考无） | — | `make test`（run 兼容别名）、`PLATFORM=` 四平台、`WITH_OPENCV`/`FAST_EXP`、本地覆盖变量 `NVCC/CXX_HOST/CUDA_RT/ARCH` |

保留原因：四平台切换与本地开发覆盖是功能必需；`test` 别名保证既有文档/CI 兼容。
参考的 `VERBOSE → TEST_VERBOSE_FLAG` 翻译机制（大小写不敏感、默认关闭）逐行沿用。

## 3. 变更清单（文件级）

| 文件 | 变更 | 原因 |
|---|---|---|
| `LICENSE` | 新增（复制自参考仓库） | 根级三件套完整性；提交目标仓库许可一致 |
| `Makefile` | 重写（目标语义 + 保留扩展） | 对齐参考构建约定（见 §2） |
| `nvidia_result.txt` | 新增（脚本真实输出） | 对齐参考的平台结果文件约定；数据来自本次重构后验证运行 |
| `metaX_result.txt` / `moore_result.txt` | 新增（占位说明） | 结构一致性；未实测平台如实标注 PENDING 并附复现步骤，不造假数据 |
| `scripts/gen_result_wsl.sh` | 新增 | nvidia_result.txt 的可复现生成入口（环境+测试+校验+基准抽样） |
| `scripts/run_all.sh` | `make` → `make build` | 新 Makefile 默认目标含运行测试，避免与脚本显式步骤重复 |
| `scripts/make_local_wsl.sh` | `make` → `make build` | 同上 |
| `scripts/check_links_wsl.sh` | 文件清单扩充 | 覆盖新增的 LICENSE/result 文件 |
| `README.md` | 结构树/构建命令/文档导航更新 | 反映新结构与 Makefile 语义 |
| `docs/user_guide.md` | 构建章节/目录树更新 | 同上 |
| `docs/architecture_design.md` | 目录树/构建流程更新 | 同上 |
| `src/**`、`tester/**` 源码 | **零改动** | 结构对齐不需要动代码；重构后重新编译验证 |

## 4. 功能性扩展目录的保留决策

参考仓库是"作业骨架"（题面 + 预编译测试器），本项目是完整应用，以下目录为功能必需，
作为参考骨架之上的**增量**保留（不与参考结构冲突，均为参考结构中不存在的层级）：

- `params/`——任务要求的参数文件（CLI 运行时读取，不可并入其他位置）；
- `third_party/stb/`——图像编解码依赖（静态编译入二进制，保证零运行时依赖）；
- `docs/`——任务要求的架构设计文档、使用文档与本变更记录；
- `scripts/`——一键准入、性能分析（ncu/nsys）、本地开发辅助。

## 5. 重构后验证记录（RTX 3060 Laptop, CUDA 12.9, WSL）

执行 `scripts/make_local_wsl.sh`（make clean → make build → units）与
`scripts/gen_result_wsl.sh`（make run VERBOSE=true → validate 512×512 → bench 抽样）：

- clean 重建：成功，零警告（ARCH=sm_86）；
- 单元测试：12/12 PASS；
- `make run VERBOSE=true`：新目标语义正确（verbose 附加 GPU 校验 256×256，三版本 MAE=0 PASS）；
- validate 512×512 RGB：三版本 MAE=0.0000 / PSNR=inf，总体 PASS；
- bench 抽样（640×360 × 灰度/RGB × 4 参数 × 3 版本）：正常出数，趋势与 bench.csv 一致；
- 产物 `nvidia_result.txt`：由上述真实输出落盘。

## 6. 回滚与兼容性说明

- 全部变更位于构建/脚本/文档层，`git checkout -- Makefile scripts/ README.md docs/` +
  删除新增文件即可回滚；
- `make test` 兼容别名保留，既有文档与脚本中的调用不受影响；
- 训练营服务器（系统级 CUDA）上 `make` / `make build` / `make run` 直接可用，
  本地 WSL 环境使用 `scripts/make_local_wsl.sh` 或变量覆盖（README 安装步骤）。

---

# 第二轮重构：目录结构系统性重构（提升可读性与 kernel 开发规范性）

| 项 | 内容 |
|---|---|
| 重构目标 | 头文件集中分类、kernel 按平台架构分目录、接口层次化、产物隔离、文档同步 |
| 重构原则 | 纯位置迁移 + include 路径规范化 + Makefile 适配，**算法代码零逻辑改动** |
| 验证结论 | clean 重建成功（零警告）；units 12/12；validate 256×256 与 512×512 三版本 MAE=0；bench 链路正常；nvidia_result.txt 重新生成 |

## R1. 目录映射（旧 → 新）

| 旧位置 | 新位置 | 说明 |
|---|---|---|
| `src/pipeline.h` `src/params.h` | `include/nlm/` | 对外接口层（上层应用只需 include 此层） |
| `src/kernels.h` | `include/kernels/` | 算子接口层 |
| `src/platform_api.h` | `include/pal/` | 平台抽象层（四平台差异唯一收敛点） |
| `src/image_io.h` `src/nlm_cpu_ref.h` | `include/core/` | 公共支撑层 |
| `tester/utils.h` | `include/tester/` | 测试支撑层 |
| `src/kernels_impl.inl` | `kernels/common/` | 全平台共享算法实现 |
| `src/kernels.cu` | `kernels/nvidia/kernels.cu` | NVIDIA 编译单元（原含天数注释，天数现已独立目录） |
| （无） | `kernels/iluvatar/kernels.cu` | **新增**：天数独立编译单元（CUDA 兼容语法，与 nvidia 同为薄包装） |
| `src/kernels.maca` | `kernels/metax/kernels.maca` | 沐曦编译单元 |
| `src/kernels.mu` | `kernels/moore/kernels.mu` | 摩尔编译单元 |
| 根目录 `nlm_denoise`、`src/*.o`、`tester/*.o` | `build/` | **产物隔离**：可执行文件 + `build/obj/` 分层对象，`make clean` 整体删除 |

`src/` 仅保留 Host 侧 `.cpp` 实现，与 `include/` 层次一一对应。

## R2. include 路径规范化

- 全部源码统一 `-Iinclude` + 层次前缀引用（如 `#include "nlm/pipeline.h"`、
  `#include "pal/platform_api.h"`），消除 `../src/` 相对路径；
- tester 三文件由 `../src/*.h` 改为规范前缀；
- 平台编译单元由 `#include "kernels_impl.inl"` 改为 `#include "../common/kernels_impl.inl"`；
- 共 17 个文件的 include 区域更新，算法逻辑零改动。

## R3. Makefile 适配

- `INCLUDES := -Iinclude -Ithird_party/stb`（原 `-Isrc -Itester`）；
- `KER_SRC := kernels/$(PLATFORM)/kernels.$(SUFFIX)`——平台名即目录名，新增平台零改构建脚本；
- 产物全部入 `build/`：`TARGET := build/nlm_denoise`，pattern rule
  `build/obj/%.o: %.cpp`（自动 mkdir），kernel 对象规则显式列出 impl + 三个头文件依赖；
- `clean` 由逐个删除改为 `rm -rf build`；
- 目标语义（all/build/run/VERBOSE/clean）与变量（PLATFORM/FAST_EXP/WITH_OPENCV/
  NVCC/CXX_HOST/CUDA_RT/ARCH）全部保持不变。

## R4. 脚本与文档同步

- 全部脚本可执行文件引用 `./nlm_denoise` → `./build/nlm_denoise`；
  `build_wsl.sh` 源列表与 `-Iinclude` 同步；`check_links_wsl.sh` 清单扩至 34 项；
- README 重写"项目结构"章节：新目录树 + **文件存放规范**五条
  （头文件只进 include、kernel 按架构归位、算法改动只碰共享实现、
  平台差异只进 PAL、产物不落源码目录）；
- user_guide / architecture_design 的目录树、构建命令、接口路径说明同步；
- `nvidia_result.txt` 以新结构重新生成（真实输出）。

## R5. 重构后验证记录（RTX 3060 Laptop, CUDA 12.9, WSL）

- `scripts/make_local_wsl.sh`（make clean → make build → units）：成功，零警告，12/12 PASS；
- `scripts/gen_result_wsl.sh`（make run VERBOSE=true → validate 512×512 → bench 抽样）：
  256×256 与 512×512 三版本 MAE=0.0000 / PSNR=inf，总体 PASS；bench 出数正常；
- Makefile 目标图 dry-run（check_makegraph_wsl.sh）：all/run/clean/平台切换符合设计；
- 文档链接检查：34/34 OK。

## R6. 回滚说明

本轮为纯位置重构：源码内容除 include 行与注释外零改动，回滚只需按 R1 表反向移动文件
并还原 Makefile/scripts/文档。两轮重构（仓库约定对齐 + 目录层次化）已叠加记录于本文档。

---

# 第三轮重构：数据集中管理（data/ 入口与分类机制）

| 项 | 内容 |
|---|---|
| 重构目标 | 系统生成与手动准备的测试数据统一整合至 `data/`，建立分类结构与命名规范 |
| 涉及代码 | `image_io`（EnsureParentDir）、`main.cpp`（gen/run 默认路径与自动命名），算法零改动 |
| 验证结论 | units 12/12；数据流脚本全过（gen 自动命名 / run 入 data/output+logs / validate 三版本 MAE=0 / 根目录清洁） |

## D1. data/ 分类结构

| 目录 | 用途 | 写入方 |
|---|---|---|
| `data/noisy/` | 含噪输入图（gen 合成或手动放置） | `gen`（缺省自动命名）/ 手动 |
| `data/clean/` | 干净参考图（质量评估基准） | 手动 |
| `data/output/` | 降噪结果图 | `run -o` / `validate -o` |
| `data/logs/` | 运行日志（nlm_perf.log 追加写） | `run --log`（缺省） |

迁移：根目录 `nlm_perf.log` → `data/logs/`；既有 `data/clean/clean.png`、
`data/noisy/noisy.png`（手动数据）保持原名归位。`bench.csv` 为交付报告数据保留根级。

## D2. 命名规范与自动化落实

- 模板：输入 `noisy_<宽x高>_<通道>ch_sigma<σ>[_seed<s>].png`、
  结果 `denoised_<输入基名>_v<kernel版本>.png`、参考 `clean_<基名>.png`；
- **gen 缺省 `-o` 自动按规范命名**（尺寸/通道/σ 已知，运行期拼接文件名）；
- **`run --log` 缺省 `data/logs/nlm_perf.log`**；
- `image_io` 新增导出 `EnsureParentDir()`（POSIX mkdir 逐级创建，C++11 兼容）：
  `SaveImage` 内部自动建父目录，日志写入前亦调用——`-o data/output/xx.png` 无需手工建目录。

## D3. 变更清单

| 文件 | 变更 |
|---|---|
| `include/core/image_io.h` / `src/image_io.cpp` | 新增 `EnsureParentDir`；`SaveImage` 保存前自动建目录 |
| `src/main.cpp` | `--log` 缺省改 `data/logs/nlm_perf.log`；`gen -o` 缺省自动命名；帮助文本同步 |
| `scripts/check_dataflow_wsl.sh` | 新增：gen→run→validate 数据流 + 根目录清洁检查 |
| `scripts/check_links_wsl.sh` | 清单扩至 37 项（含 data/ 文件） |
| `README.md` | 新增"数据管理"章节（分类表 + 命名规范表 + 自动化保障）；结构树/使用示例/存放规范同步 |
| `docs/user_guide.md` / `architecture_design.md` | 示例命令与目录树同步 data/ 路径 |
| 根目录 | `nlm_perf.log` 迁出，运行期不再产生根级数据文件 |

## D4. 验证记录（RTX 3060 Laptop, CUDA 12.9, WSL）

- 重建 + units：12/12 PASS；
- `scripts/check_dataflow_wsl.sh`：gen 自动命名 `data/noisy/noisy_256x256_3ch_sigma25.png`
  生成正确；run 输出 `data/output/denoised_..._v2.png` 且日志行追加至
  `data/logs/nlm_perf.log`；validate 三版本 MAE=0.0000 / PSNR=inf PASS；
  根目录无 png/log 新增（清洁检查 OK）。
