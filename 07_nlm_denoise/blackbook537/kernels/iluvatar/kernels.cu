// 天数智芯 Iluvatar CoreX 平台编译单元（clang++ -x ivcore 编译）
// CoreX 提供 CUDA 兼容运行时（API 与 NVIDIA 同名），算法实现全部位于
// common/kernels_impl.inl（多平台共享），平台差异由 include/pal/platform_api.h
// 收敛，本文件仅做编译期包装。
#include "../common/kernels_impl.inl"
