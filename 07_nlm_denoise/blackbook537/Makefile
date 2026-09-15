# *********************************************************************
# nlm_denoise Makefile —— 实时图像非局部均值降噪（CUDA）
# 目标语义与 Learning-CUDA 仓库约定一致：
#   make                 : Build + run tests（默认，非 verbose）
#   make build           : Only compile (no run)
#   make run             : Run tests (after build, non-verbose)
#   make run VERBOSE=true: Run tests with verbose output（附加 GPU 端到端校验）
#   make test            : 同 run（兼容别名）
#   make clean           : Delete temporary files
# 变量：
#   PLATFORM=nvidia|iluvatar|metax|moore   目标平台（默认 nvidia）
#   WITH_OPENCV=1                          validate 增加 OpenCV 交叉验证
#   FAST_EXP=1                             NLM 权重使用 __expf 快速指数（近似，报告需注明）
#   NVCC/CXX_HOST/CUDA_RT/ARCH             本地开发机覆盖（见 README 安装步骤）
#
# 目录约定（详见 README 目录结构章节）：
#   include/   全部头文件（nlm=对外接口 kernels=算子接口 pal=平台抽象
#              core=公共支撑 tester=测试支撑）
#   src/       Host 侧实现（.cpp）
#   kernels/   设备侧算子（common=共享实现，{nvidia,iluvatar,metax,moore}/=
#              各平台架构编译单元）
#   tester/    测试与基准实现
#   build/     全部构建产物（可执行文件 + 对象文件，clean 时整体删除）
# *********************************************************************

# -------------------------------
# Configuration
# -------------------------------
PLATFORM    ?= nvidia
WITH_OPENCV ?= 0
FAST_EXP    ?= 0

# 可选覆盖变量（默认即训练营服务器环境，无需设置）：
#   NVCC     nvcc 路径（默认取 PATH 中的 nvcc）
#   CXX_HOST nvcc 的主机编译器（本地 gcc 版本过新时指定，如 conda gcc13）
#   CUDA_RT  非系统位置的 cudart 运行库目录（需含 include/ 与 lib/）
#   ARCH     目标 GPU 架构（如 sm_80/sm_86/sm_90；留空用 nvcc 默认全架构，
#            会产生 pre-sm_75 弃用警告并增加编译时间）
NVCC        ?= nvcc
CXX_HOST    ?=
CUDA_RT     ?=
ARCH        ?=

SUFFIX          := cu
PLATFORM_DEFINE := -DPLATFORM_NVIDIA
CFLAGS          := -std=c++17 -O3
EXTRA_LIBS      :=
COMP_ONLY_FLAGS :=

ifeq ($(PLATFORM),nvidia)
    CC              := $(NVCC)
    PLATFORM_DEFINE := -DPLATFORM_NVIDIA
    ifneq ($(CXX_HOST),)
        CFLAGS += -ccbin $(CXX_HOST)
    endif
    ifneq ($(CUDA_RT),)
        CFLAGS     += -I$(CUDA_RT)/include
        EXTRA_LIBS += -L$(CUDA_RT)/lib -lcudart_static -lpthread -ldl -lrt
    endif
    ifneq ($(ARCH),)
        CFLAGS += -arch=$(ARCH)
    endif
else ifeq ($(PLATFORM),iluvatar)
    CC              := clang++
    CFLAGS          := -std=c++17 -O3 -I/usr/local/corex/include -fPIC
    COMP_ONLY_FLAGS := -x ivcore --cuda-path=/usr/local/corex
    PLATFORM_DEFINE := -DPLATFORM_ILUVATAR
    EXTRA_LIBS      := -lcudart -L/usr/local/corex/lib64
else ifeq ($(PLATFORM),metax)
    CC              := mxcc
    SUFFIX          := maca
    PLATFORM_DEFINE := -DPLATFORM_METAX
else ifeq ($(PLATFORM),moore)
    CC              := mcc
    CFLAGS          := -std=c++11 -O3
    SUFFIX          := mu
    PLATFORM_DEFINE := -DPLATFORM_MOORE
    EXTRA_LIBS      := -I/usr/local/musa/include -L/usr/lib/gcc/x86_64-linux-gnu/11/ -L/usr/local/musa/lib -lmusart
else
    $(error Unsupported PLATFORM '$(PLATFORM)' (expected: nvidia, iluvatar, metax, moore))
endif

ifeq ($(FAST_EXP),1)
    PLATFORM_DEFINE += -DNLM_FAST_EXP
endif

ifeq ($(WITH_OPENCV),1)
    PLATFORM_DEFINE += -DWITH_OPENCV
    CFLAGS    += $(shell pkg-config --cflags opencv4)
    EXTRA_LIBS += $(shell pkg-config --libs opencv4)
endif

# -------------------------------
# Sources & Objects
# -------------------------------
TARGET      := build/nlm_denoise
OBJ_DIR     := build/obj
INCLUDES    := -Iinclude -Ithird_party/stb

HOST_SRCS   := src/main.cpp src/params.cpp src/image_io.cpp src/pipeline.cpp \
               src/nlm_cpu_ref.cpp tester/validate.cpp tester/benchmark.cpp \
               tester/test_units.cpp
KER_SRC     := kernels/$(PLATFORM)/kernels.$(SUFFIX)
HOST_OBJS   := $(addprefix $(OBJ_DIR)/,$(HOST_SRCS:.cpp=.o))
# kernel 对象按平台目录隔离：切换 PLATFORM 自动触发重编，避免跨平台复用旧对象
KER_OBJ     := $(OBJ_DIR)/kernels/$(PLATFORM)/kernels.o
OBJS        := $(HOST_OBJS) $(KER_OBJ)

# Tester's actual verbose argument (e.g., --verbose, -v)
TEST_VERBOSE_FLAG := --verbose
# User-provided verbose mode (true/false; default: false)
VERBOSE          :=

# -------------------------------
# Process User Input (VERBOSE → Tester Flag)
# -------------------------------
# Translates `VERBOSE=true` (case-insensitive) to the tester's verbose flag.
# If VERBOSE is not "true" (or empty), no flag is passed.
VERBOSE_ARG := $(if $(filter true True TRUE, $(VERBOSE)), $(TEST_VERBOSE_FLAG), )

# -------------------------------
# Phony Targets
# -------------------------------
.PHONY: all build run test clean

# Default target: Build + run tests (non-verbose)
all: build run

# Build target: Compile application (kernels + host modules)
build: $(TARGET)

# Run target: Execute test suite (supports `VERBOSE=true` for verbose output)
run: $(TARGET)
	@echo "=== Running tests (output from $(TARGET)) ==="
	@# Show verbose mode status (friendly for users)
	@if [ -n "$(VERBOSE_ARG)" ]; then \
	    echo "=== Verbose mode: Enabled (using '$(TEST_VERBOSE_FLAG)') ==="; \
	else \
	    echo "=== Verbose mode: Disabled ==="; \
	fi
	$(TARGET) units
	@if [ -n "$(VERBOSE_ARG)" ]; then \
	    echo "=== Verbose extra: GPU 端到端正确性校验（256x256 RGB）==="; \
	    $(TARGET) gen -o /tmp/nlm_make_run.png --size 256x256 --channels 3 --sigma 25; \
	    $(TARGET) validate -i /tmp/nlm_make_run.png -p params/default.txt; \
	fi

# Backward-compatible alias of `run`
test: run

# Clean target: Delete the whole build directory (executable + objects)
clean:
	@echo "=== Cleaning temporary files (build/) ==="
	rm -rf build

# -------------------------------
# Dependency Rules (Core Logic)
# -------------------------------
# Generate executable: Link kernels + host modules + tester modules
$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	@echo "=== Linking executable ($(TARGET), $(PLATFORM)) ==="
	$(CC) $(CFLAGS) $(PLATFORM_DEFINE) -o $@ $(OBJS) $(EXTRA_LIBS)

# Host 侧对象：src/**.cpp 与 tester/**.cpp → build/obj/{src,tester}/**.o
$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(PLATFORM_DEFINE) $(INCLUDES) -c $< -o $@

# 设备侧对象：kernels/$(PLATFORM)/kernels.$(SUFFIX) → build/obj/kernels/kernels.o
# 头文件依赖显式列出（impl + 接口头 + PAL + 参数头），保证改动即触发重编
$(KER_OBJ): $(KER_SRC) kernels/common/kernels_impl.inl \
            include/kernels/kernels.h include/pal/platform_api.h include/nlm/params.h
	@mkdir -p $(dir $@)
	@echo "=== Compiling kernels ($(KER_SRC)) ==="
	$(CC) $(CFLAGS) $(COMP_ONLY_FLAGS) $(PLATFORM_DEFINE) $(INCLUDES) -c $(KER_SRC) -o $@
