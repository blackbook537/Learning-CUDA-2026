#include <vector>
#include <cmath>
#include <musa_fp16.h>

#include "../tester/utils.h"

// =====================================================================
// 运行时 API 抽象层：使同一份源码可在 NVIDIA / Iluvatar / MetaX / Moore
// 四个平台上编译。NVIDIA 与 Iluvatar (CoreX) 使用 cuda 运行时；
// MetaX (MACA) 使用 mc 运行时；Moore (MUSA) 使用 musa 运行时。
// 设备端代码（kernel 内的 __shfl_xor_sync / expf / rsqrtf 等）均为类 CUDA
// 语法，各平台通用。
// =====================================================================
#if defined(PLATFORM_NVIDIA) || defined(PLATFORM_ILUVATAR)
  #define RT_MALLOC        cudaMalloc
  #define RT_MEMCPY        cudaMemcpy
  #define RT_FREE          cudaFree
  #define RT_GET_LAST_ERR  cudaGetLastError
  #define RT_DEVICE_SYNC   cudaDeviceSynchronize
  #define RT_MEM_H2D       cudaMemcpyHostToDevice
  #define RT_MEM_D2H       cudaMemcpyDeviceToHost
#elif defined(PLATFORM_METAX)
  #define RT_MALLOC        mcMalloc
  #define RT_MEMCPY        mcMemcpy
  #define RT_FREE          mcFree
  #define RT_GET_LAST_ERR  mcGetLastError
  #define RT_DEVICE_SYNC   mcDeviceSynchronize
  #define RT_MEM_H2D       mcMemcpyHostToDevice
  #define RT_MEM_D2H       mcMemcpyDeviceToHost
#elif defined(PLATFORM_MOORE)
  #define RT_MALLOC        musaMalloc
  #define RT_MEMCPY        musaMemcpy
  #define RT_FREE          musaFree
  #define RT_GET_LAST_ERR  musaGetLastError
  #define RT_DEVICE_SYNC   musaDeviceSynchronize
  #define RT_MEM_H2D       musaMemcpyHostToDevice
  #define RT_MEM_D2H       musaMemcpyDeviceToHost
#endif

// =====================================================================
// 通用工具：在 device 上进行 T <-> float 的转换。
// 使用 static_cast 以保证在各平台的 half 类型上均可移植编译。
// 核心计算统一在 float 精度下进行，兼顾数值精度与 half 类型支持。
// =====================================================================
template <typename T>
__device__ __forceinline__ float toFloatVal(T x) {
  return static_cast<float>(x);
}
template <typename T>
__device__ __forceinline__ T fromFloatVal(float x) {
  return static_cast<T>(x);
}

// =====================================================================
//                             rmsNorm
// =====================================================================
// 对输入矩阵 [rows, hidden_dim] 的每一行独立做 RMSNorm：
//   mean_square = sum_j input[i,j]^2 / hidden_dim
//   output[i,j] = input[i,j] * rsqrt(mean_square + eps) * weight[j]
//
// 实现策略：一个 thread block 负责一行，block 内做树形归约得到该行的
// 平方和。线程数取 hidden_dim 向上取到的 2 的幂（上限 1024），保证树形
// 归约的正确性；当 hidden_dim 大于线程数时，每个线程以跨步方式处理多个
// 元素。所有累加在 float 下进行，保证 half 输入的精度。
// =====================================================================
template <typename T>
__global__ void rmsNormKernel(const T* __restrict__ input,
                              const T* __restrict__ weight,
                              T* __restrict__ output,
                              size_t rows, size_t hidden_dim, float eps) {
  extern __shared__ float sdata[];  // 用于块内归约
  const size_t row = blockIdx.x;
  if (row >= rows) return;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const T* in_row = input + row * hidden_dim;
  T* out_row = output + row * hidden_dim;

  // 1) 每个线程累加本行中跨步元素的平方和（float 精度）
  float local_sum = 0.0f;
  for (size_t j = tid; j < hidden_dim; j += nthreads) {
    float v = toFloatVal(in_row[j]);
    local_sum += v * v;
  }
  sdata[tid] = local_sum;
  __syncthreads();

  // 2) 树形归约（blockDim.x 已保证为 2 的幂）
  for (int s = nthreads / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // 3) 计算 inv_rms 并写出结果
  const float mean_square = sdata[0] / static_cast<float>(hidden_dim);
  const float inv_rms = rsqrtf(mean_square + eps);

  for (size_t j = tid; j < hidden_dim; j += nthreads) {
    float v = toFloatVal(in_row[j]);
    float w = toFloatVal(weight[j]);
    out_row[j] = fromFloatVal<T>(v * inv_rms * w);
  }
}

template <typename T>
void rmsNorm(const std::vector<T>& h_input, const std::vector<T>& h_weight,
             std::vector<T>& h_output, size_t rows, size_t hidden_dim,
             float eps) {
  const size_t n = rows * hidden_dim;
  if (n == 0) return;

  T* d_input = nullptr;
  T* d_weight = nullptr;
  T* d_output = nullptr;
  RUNTIME_CHECK(RT_MALLOC(&d_input, n * sizeof(T)));
  RUNTIME_CHECK(RT_MALLOC(&d_weight, hidden_dim * sizeof(T)));
  RUNTIME_CHECK(RT_MALLOC(&d_output, n * sizeof(T)));
  RUNTIME_CHECK(RT_MEMCPY(d_input, h_input.data(), n * sizeof(T), RT_MEM_H2D));
  RUNTIME_CHECK(RT_MEMCPY(d_weight, h_weight.data(), hidden_dim * sizeof(T),
                          RT_MEM_H2D));

  // 线程数取 >= hidden_dim 的最小 2 的幂，上限 1024，保证树形归约正确
  int threads = 1;
  while (threads < static_cast<int>(hidden_dim) && threads < 1024) {
    threads <<= 1;
  }
  if (threads < 1) threads = 1;
  const size_t smem = static_cast<size_t>(threads) * sizeof(float);

  rmsNormKernel<T><<<static_cast<unsigned int>(rows), threads, smem>>>(
      d_input, d_weight, d_output, rows, hidden_dim, eps);
  RUNTIME_CHECK(RT_GET_LAST_ERR());
  RUNTIME_CHECK(RT_DEVICE_SYNC());
  RUNTIME_CHECK(RT_MEMCPY(h_output.data(), d_output, n * sizeof(T), RT_MEM_D2H));

  RT_FREE(d_input);
  RT_FREE(d_weight);
  RT_FREE(d_output);
}

// =====================================================================
//                         flashAttention
// =====================================================================
// 实现 Flash Attention 前向，行为与
// torch.nn.functional.scaled_dot_product_attention 一致：
//   scale = 1 / sqrt(head_dim)
//   S = Q @ K^T * scale         (支持 GQA：每 group_size 个 q head 共享 1 个 kv head)
//   causal: 屏蔽 key j > query i 的位置（左对齐下三角，与 PyTorch 一致）
//   P = softmax(S, dim=-1)
//   O = P @ V
//
// 数值策略：采用「精确 softmax」（两遍扫描），与参考实现一致：
//   Pass 1 —— 遍历所有 key 分块，计算 score S[i,j] = (Q·K) * scale，
//             warp 级归约得到每行全局最大值 row_max。
//   Pass 2 —— 再次遍历所有 key 分块，计算 P[i,j] = exp(S[i,j] - row_max)，
//             累加 sum_p 和输出 O = Σ P[j] * V[j]。
//   最后 O /= sum_p。
//
// 相比 online softmax，精确 softmax 消除了分块间 rescaling 因子累积带来的
// 舍入误差，在长序列（src_seq_len 大、分块数多）下 float 精度显著更优。
//
// 核函数设计（warp-per-row）：
//   - 每个 block 处理 BM 个 query 行（同一个 (batch, query_head)）。
//   - block 含 WARPS 个 warp，每个 warp 负责一行 query。
//   - 沿 src_seq 方向以 BN(=32) 为粒度分块，逐块加载 K/V 到 shared memory，
//     所有 warp 共享同一块 K/V，从而复用 K/V（BM 倍 HBM 流量节省）。
//   - 每个 warp 内，lane t 计算一个 key（j=n0+t）对应的 score，使用 warp 级
//     shuffle 归约得到 row-max 与 row-sum。
//   - 全程在 float 下计算，half 输入/输出仅在边界处转换，保证精度。
//   - K/V 在 shared memory 中行间 stride 取 Dp = D+1，以消除 32-way bank conflict。
// =====================================================================
template <typename T>
__global__ void flashAttentionKernel(const T* __restrict__ Q,
                                     const T* __restrict__ K,
                                     const T* __restrict__ V,
                                     T* __restrict__ O,
                                     int B, int M, int N, int Hq, int Hkv,
                                     int D, int group_size, bool is_causal) {
  constexpr int BM = 8;        // 每个 block 处理的 query 行数
  constexpr int BN = 32;       // key 分块大小（等于 warp size，lane<->key 一一对应）
  constexpr int WARPS = BM;    // warp 数 = 行数
  constexpr int THREADS = WARPS * 32;

  const int Dp = D + 1;        // K/V 行的 shared memory stride，打破 bank conflict

  extern __shared__ float smem[];
  float* q_sh = smem;                       // [BM, D]
  float* k_sh = q_sh + BM * D;              // [BN, Dp]
  float* v_sh = k_sh + BN * Dp;             // [BN, Dp]
  float* s_sh = v_sh + BN * Dp;             // [BM, BN]  每行各 key 的 P 值
  float* o_sh = s_sh + BM * BN;             // [BM, D]   输出累加器

  const int q_block = blockIdx.x;
  const int h = blockIdx.y;       // query head
  const int b = blockIdx.z;       // batch
  const int q_base = q_block * BM;
  const int kvh = h / group_size; // GQA：该 q head 对应的 kv head

  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int qi = q_base + warp;   // 该 warp 负责的 query 行（warp 一致）

  const float scale = 1.0f / sqrtf(static_cast<float>(D));

  // ---- 加载 Q tile [BM, D] 到 shared（协作加载，越界行填 0）----
  for (int i = tid; i < BM * D; i += THREADS) {
    int row = i / D;
    int col = i % D;
    int qidx = q_base + row;
    float val = 0.0f;
    if (qidx < M) {
      val = toFloatVal(Q[((static_cast<size_t>(b) * M + qidx) * Hq + h) * D + col]);
    }
    q_sh[row * D + col] = val;
  }

  // ---- 初始化输出累加器 o_sh = 0 ----
  for (int i = tid; i < BM * D; i += THREADS) {
    o_sh[i] = 0.0f;
  }
  __syncthreads();

  // ===== Pass 1: 遍历所有 key 分块，找到每行全局最大值 row_max =====
  // 精确 softmax 的第一步：先扫描所有 score 得到全局 max，避免 online softmax
  // 的分块间 rescaling 累积误差。
  float row_max = -INFINITY;

  for (int n0 = 0; n0 < N; n0 += BN) {
    // 协作加载 K tile [BN, D]（越界 key 填 0），stride = Dp
    for (int i = tid; i < BN * D; i += THREADS) {
      int row = i / D;
      int col = i % D;
      int kj = n0 + row;
      float kv = 0.0f;
      if (kj < N) {
        size_t idx = ((static_cast<size_t>(b) * N + kj) * Hkv + kvh) * D + col;
        kv = toFloatVal(K[idx]);
      }
      k_sh[row * Dp + col] = kv;
    }
    __syncthreads();

    if (qi < M) {
      // lane t 计算 key j = n0 + t 的 score
      float s = -INFINITY;
      int kj = n0 + lane;
      if (kj < N) {
        const float* qrow = q_sh + warp * D;
        const float* krow = k_sh + lane * Dp;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
          dot += qrow[d] * krow[d];
        }
        s = dot * scale;
        // causal 屏蔽：key j > query i 置 -inf（与 PyTorch SDPA 一致）
        if (is_causal && kj > qi) {
          s = -INFINITY;
        }
      }

      // warp 级归约：本块 row-max
      float m_block = s;
      for (int off = 16; off > 0; off >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, m_block, off);
        m_block = (other > m_block) ? other : m_block;
      }
      // 更新全局 row_max（warp 内各 lane 归约后一致）
      row_max = (m_block > row_max) ? m_block : row_max;
    }
    __syncthreads();
  }

  // ===== Pass 2: 再次遍历所有 key 分块，计算 exp/sum 并累加 O =====
  // P[i,j] = exp(S[i,j] - row_max)，sum_p = Σ P[i,j]，O[i,:] = Σ P[i,j] * V[j,:]
  float sum_p = 0.0f;

  for (int n0 = 0; n0 < N; n0 += BN) {
    // 协作加载 K/V tile [BN, D]（越界 key 填 0），stride = Dp
    for (int i = tid; i < BN * D; i += THREADS) {
      int row = i / D;
      int col = i % D;
      int kj = n0 + row;
      float kv = 0.0f, vv = 0.0f;
      if (kj < N) {
        size_t idx = ((static_cast<size_t>(b) * N + kj) * Hkv + kvh) * D + col;
        kv = toFloatVal(K[idx]);
        vv = toFloatVal(V[idx]);
      }
      k_sh[row * Dp + col] = kv;
      v_sh[row * Dp + col] = vv;
    }
    __syncthreads();

    if (qi < M) {
      // 重新计算 score 并求 P = exp(s - row_max)（被屏蔽位置 P = 0）
      int kj = n0 + lane;
      float p = 0.0f;
      if (kj < N && (!is_causal || kj <= qi)) {
        const float* qrow = q_sh + warp * D;
        const float* krow = k_sh + lane * Dp;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
          dot += qrow[d] * krow[d];
        }
        p = expf(dot * scale - row_max);
      }
      s_sh[warp * BN + lane] = p;

      // warp 级归约：本块 P 之和，累加到 sum_p（warp 内各 lane 一致）
      float sum_block = p;
      for (int off = 16; off > 0; off >>= 1) {
        sum_block += __shfl_xor_sync(0xffffffff, sum_block, off);
      }
      sum_p += sum_block;

      // 累加输出：o[d] += Σ_j P[j] * V[j][d]
      // 每个 lane 以跨步方式负责 d = lane, lane+32, ...
      __syncwarp();
      for (int dd = 0; dd * 32 + lane < D; ++dd) {
        int d = lane + dd * 32;
        float acc = 0.0f;
        const float* vrow = v_sh + d;  // v_sh[j*Dp + d]
        for (int j = 0; j < BN; ++j) {
          acc += s_sh[warp * BN + j] * vrow[j * Dp];
        }
        o_sh[warp * D + d] += acc;
      }
    }
    __syncthreads();
  }

  // ---- 写出 O = o / sum_p ----
  if (qi < M) {
    float inv_sum = (sum_p > 0.0f) ? (1.0f / sum_p) : 0.0f;
    for (int dd = 0; dd * 32 + lane < D; ++dd) {
      int d = lane + dd * 32;
      float val = o_sh[warp * D + d] * inv_sum;
      O[((static_cast<size_t>(b) * M + qi) * Hq + h) * D + d] = fromFloatVal<T>(val);
    }
  }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len,
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  // GQA：query_heads 必须是 kv_heads 的整数倍（由 tester 保证）
  const int group_size = query_heads / kv_heads;

  const size_t q_sz = static_cast<size_t>(batch_size) * target_seq_len * query_heads * head_dim;
  const size_t kv_sz = static_cast<size_t>(batch_size) * src_seq_len * kv_heads * head_dim;
  if (q_sz == 0) return;

  T *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_o = nullptr;
  RUNTIME_CHECK(RT_MALLOC(&d_q, q_sz * sizeof(T)));
  RUNTIME_CHECK(RT_MALLOC(&d_k, kv_sz * sizeof(T)));
  RUNTIME_CHECK(RT_MALLOC(&d_v, kv_sz * sizeof(T)));
  RUNTIME_CHECK(RT_MALLOC(&d_o, q_sz * sizeof(T)));
  RUNTIME_CHECK(RT_MEMCPY(d_q, h_q.data(), q_sz * sizeof(T), RT_MEM_H2D));
  RUNTIME_CHECK(RT_MEMCPY(d_k, h_k.data(), kv_sz * sizeof(T), RT_MEM_H2D));
  RUNTIME_CHECK(RT_MEMCPY(d_v, h_v.data(), kv_sz * sizeof(T), RT_MEM_H2D));

  constexpr int BM = 8;
  const int Dp = head_dim + 1;
  const size_t smem =
      (static_cast<size_t>(2 * BM * head_dim) +  // q_sh + o_sh
       static_cast<size_t>(2 * 32 * Dp) +        // k_sh + v_sh
       static_cast<size_t>(BM * 32)) *           // s_sh
      sizeof(float);

  // 当 shared memory 超过 48KB 时，申请 opt-in 上限以支持较大 head_dim。
  // 仅在 NVIDIA / Iluvatar（cuda 运行时）上启用；其余平台依赖默认 48KB
  // （足以支持常见的 head_dim <= 128）。
#if defined(PLATFORM_NVIDIA) || defined(PLATFORM_ILUVATAR)
  if (smem > 48 * 1024) {
    int max_shared = 0;
    cudaDeviceGetAttribute(&max_shared,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    if (static_cast<int>(smem) <= max_shared) {
      cudaFuncSetAttribute(
          flashAttentionKernel<T>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem));
    }
  }
#endif

  const int q_blocks = (target_seq_len + BM - 1) / BM;
  dim3 grid(q_blocks, query_heads, batch_size);
  dim3 block(BM * 32);

  flashAttentionKernel<T><<<grid, block, smem>>>(
      d_q, d_k, d_v, d_o, batch_size, target_seq_len, src_seq_len,
      query_heads, kv_heads, head_dim, group_size, is_causal);
  RUNTIME_CHECK(RT_GET_LAST_ERR());
  RUNTIME_CHECK(RT_DEVICE_SYNC());
  RUNTIME_CHECK(RT_MEMCPY(h_o.data(), d_o, q_sz * sizeof(T), RT_MEM_D2H));

  RT_FREE(d_q);
  RT_FREE(d_k);
  RT_FREE(d_v);
  RT_FREE(d_o);
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template void rmsNorm<float>(const std::vector<float>&, const std::vector<float>&,
  std::vector<float>&, size_t, size_t, float);
template void rmsNorm<half>(const std::vector<half>&, const std::vector<half>&,
  std::vector<half>&, size_t, size_t, float);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
