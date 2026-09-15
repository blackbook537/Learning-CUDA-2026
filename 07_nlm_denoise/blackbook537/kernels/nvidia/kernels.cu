// NVIDIA CUDA 平台编译单元（nvcc 编译）
// 算法实现全部位于 common/kernels_impl.inl（多平台共享），
// 平台差异由 include/pal/platform_api.h 收敛，本文件仅做编译期包装。
#include "../common/kernels_impl.inl"
