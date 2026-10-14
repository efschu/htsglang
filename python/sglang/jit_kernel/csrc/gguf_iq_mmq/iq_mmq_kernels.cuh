// GGUF-NF G6 (2026-10-09): IQ-type MMQ / MoE-MMQ kernels for GGUF I-quants, vendored as NEW JIT sources.
// Source: sgl-project/sglang PR #36122 "[CUDA] Add dense and MoE GGUF MMQ kernels for eight I-quant types"
//   (open, unreviewed; head 0a39ec3a8754e3c4426950a5a93046c048a03f07, base 6e7beace143964386ce2e4418401b7369f0dabd0;
//   diff sha256 cf93506bd6b55eca441794007b1d5e92f6a66bf5f0524f288f460f1418bcee38),
//   itself adapted from https://github.com/aimbit-ni/vllm/commit/037e1d547c15313aa7e2bb5fc83390ab0c857314 and
//   ggml-org/llama.cpp PR #8495 (MIT, (c) 2023-2024 The ggml authors).
// Existing sgl-kernel/csrc/quantization/gguf/* is NOT touched (sha256-pinned by test_gguf_iq_mmq_1009.py).
//
// Generic MMQ / MoE-MMQ tile loops (iq_mul_mat_q, iq_moe_q) and the eight I-quant instantiations of PR #36122.
// iq_mul_mat_q is sgl-kernel/csrc/quantization/gguf/mmq.cuh mul_mat_q (base c651892375) with one edit: the tile loader
// gets `blocks_per_row_x - ib0` (blocks_left). iq_moe_q is the fork's moe.cuh moe_q (int64 expert stride, local
// expert-count guard) with the PR #36122 changes: early return for padding blocks, zero-fill of invalid-expert blocks,
// per-column q8 scale load; plus the same blocks_left edit.
#pragma once

#include "iq_mmq_tiles.cuh"

template <
    typename scalar_t,
    int qk,
    int qr,
    int qi,
    bool need_sum,
    typename block_q_t,
    int mmq_x,
    int mmq_y,
    int nwarps,
    iq_allocate_tiles_cuda_t allocate_tiles,
    iq_load_tiles_cuda_t load_tiles,
    int vdr,
    iq_vec_dot_q_mul_mat_cuda_t vec_dot>
static __device__ __forceinline__ void iq_mul_mat_q(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst) {
  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;

  const int blocks_per_row_x = ncols_x / qk;
  const int blocks_per_col_y = nrows_y / QK8_1;
  const int blocks_per_warp = WARP_SIZE_GGUF / qi;

  const int& ncols_dst = ncols_y;

  const auto row_dst_0 = blockIdx.x * mmq_y;
  const int& row_x_0 = row_dst_0;

  const auto col_dst_0 = blockIdx.y * mmq_x;
  const int& col_y_0 = col_dst_0;

  int* tile_x_ql = nullptr;
  half2* tile_x_dm = nullptr;
  int* tile_x_qh = nullptr;
  int* tile_x_sc = nullptr;

  allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

  __shared__ int tile_y_qs[mmq_x * WARP_SIZE_GGUF];
  __shared__ half2 tile_y_ds[mmq_x * WARP_SIZE_GGUF / QI8_1];

  float sum[mmq_y / WARP_SIZE_GGUF][mmq_x / nwarps] = {{0.0f}};

  for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {
    load_tiles(
        x + row_x_0 * blocks_per_row_x + ib0,
        tile_x_ql,
        tile_x_dm,
        tile_x_qh,
        tile_x_sc,
        threadIdx.y,
        nrows_x - row_x_0 - 1,
        threadIdx.x,
        blocks_per_row_x,
        blocks_per_row_x - ib0);

#pragma unroll
    for (int ir = 0; ir < qr && ib0 + ir * blocks_per_warp / qr < blocks_per_row_x; ++ir) {
      const auto kqs = ir * WARP_SIZE_GGUF + threadIdx.x;
      const int kbxd = kqs / QI8_1;

#pragma unroll
      for (int i = 0; i < mmq_x; i += nwarps) {
        const int col_y_eff = min(col_y_0 + threadIdx.y + i, ncols_y - 1);  // to prevent out-of-bounds memory accesses
        const block_q8_1* by0 = &y[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) + kbxd];
        const int index_y = (threadIdx.y + i) * WARP_SIZE_GGUF + kqs % WARP_SIZE_GGUF;
        tile_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
      }

#pragma unroll
      for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
        const int ids = (ids0 + threadIdx.y * QI8_1 + threadIdx.x / (WARP_SIZE_GGUF / QI8_1)) % mmq_x;
        const auto kby = threadIdx.x % (WARP_SIZE_GGUF / QI8_1);
        const int col_y_eff = min(col_y_0 + ids, ncols_y - 1);

        // if the sum is not needed it's faster to transform the scale to f32 ahead of time
        const half2* dsi_src =
            &y[col_y_eff * blocks_per_col_y + ib0 * (qk / QK8_1) + ir * (WARP_SIZE_GGUF / QI8_1) + kby].ds;
        half2* dsi_dst = &tile_y_ds[ids * (WARP_SIZE_GGUF / QI8_1) + kby];
        if (need_sum) {
          *dsi_dst = *dsi_src;
        } else {
          float* dfi_dst = (float*)dsi_dst;
          *dfi_dst = __low2float(*dsi_src);
        }
      }

      __syncthreads();

      // #pragma unroll // unrolling this loop causes too much register pressure
      for (int k = ir * WARP_SIZE_GGUF / qr; k < (ir + 1) * WARP_SIZE_GGUF / qr; k += vdr) {
#pragma unroll
        for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
          for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
            sum[i / WARP_SIZE_GGUF][j / nwarps] += vec_dot(
                tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs, tile_y_ds, threadIdx.x + i, threadIdx.y + j, k);
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int j = 0; j < mmq_x; j += nwarps) {
    const auto col_dst = col_dst_0 + j + threadIdx.y;
    if (col_dst >= ncols_dst) {
      return;
    }

#pragma unroll
    for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
      const auto row_dst = row_dst_0 + threadIdx.x + i;
      if (row_dst >= nrows_dst) {
        continue;
      }
      dst[col_dst * nrows_dst + row_dst] = sum[i / WARP_SIZE_GGUF][j / nwarps];
    }
  }
}


#include <cstdint>

/* Adapted from ./csrc/quantization/gguf/mmq.cuh
 */
template <
    typename scalar_t,
    int qk,
    int qr,
    int qi,
    bool need_sum,
    typename block_q_t,
    int mmq_x,
    int mmq_y,
    int nwarps,
    iq_allocate_tiles_cuda_t allocate_tiles,
    iq_load_tiles_cuda_t load_tiles,
    int vdr,
    iq_vec_dot_q_mul_mat_cuda_t vec_dot>
static __device__ __forceinline__ void iq_moe_q(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int* __restrict__ sorted_token_ids,
    const int* __restrict__ expert_ids,
    const int* __restrict__ num_tokens_post_padded,
    const int64_t exp_stride,
    const int num_experts,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    const int top_k) {
  const int blocks_per_row_x = ncols_x / qk;
  const int blocks_per_col_y = nrows_y / QK8_1;
  const int blocks_per_warp = WARP_SIZE_GGUF / qi;

  const int ncols_dst = ncols_y * top_k;

  const auto row_dst_0 = blockIdx.x * mmq_y;
  const int& row_x_0 = row_dst_0;

  const auto col_dst_0 = blockIdx.y * mmq_x;

  // PR #36122: padding blocks beyond num_tokens_post_padded write nothing.
  if (col_dst_0 >= num_tokens_post_padded[0]) return;

  int token_offs[mmq_x / nwarps];
  for (int i = 0; i < mmq_x; i += nwarps) {
    token_offs[i / nwarps] = sorted_token_ids[col_dst_0 + threadIdx.y + i];
  }

  const int exp_idx = expert_ids[blockIdx.y];
  // Skip padding/filler blocks and any out-of-range expert id. num_experts is
  // the number of LOCAL experts held by this rank (vx.size(0) == W.sizes()[0]);
  // under the fork's expert-dim TP sharding this is < 256, so the previous
  // hardcoded `exp_idx > 255` guard let a padding sentinel or stale id in
  // [num_experts, 255] through and read vx + exp_idx*exp_stride out of bounds.
  // PR #36122: an invalid / filler block is ZERO-filled instead of skipped, so no destination row keeps an
  // uninitialised value; the guard is bound to the LOCAL expert count (fork fix #109/#112) instead of the PR's literal 255.
  if (exp_idx >= num_experts || exp_idx < 0) {
#pragma unroll
    for (int j = 0; j < mmq_x; j += nwarps) {
      const int col_dst = token_offs[j / nwarps];
      if (col_dst >= ncols_dst) {
        continue;
      }

#pragma unroll
      for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
        const int row_dst = row_dst_0 + threadIdx.x + i;
        if (row_dst < nrows_dst) {
          dst[col_dst * nrows_dst + row_dst] = scalar_t(0);
        }
      }
    }
    return;
  }

  // #512: exp_stride is a BYTE stride (W.stride(0) of the uint8 expert
  // tensor), so exp_idx * exp_stride is the byte offset of the LAST
  // expert, i.e. approximately the size of the whole local tensor. It was
  // computed in 32-bit, which wraps to a negative offset -- an
  // out-of-bounds read far BELOW the tensor -- once one rank's per-layer
  // expert weights exceed 2 GiB. DeepSeek-V4-Flash reaches that at 256
  // experts of Q4_K w13 (2*2048*4096 elements * 0.5625 B = 9.44 MB each,
  // 2.416e9 B in total, past the 2^31 = 2.147e9 ceiling from local expert
  // 227 upward); on this rig TP=3 expert sharding keeps it under, TP=1
  // does not. The call site already passes int64_t (W.stride(0)), so this
  // was a narrowing at the boundary and nothing else. Same failure mode
  // as #109/#112 -- an out-of-range read through exp_idx -- from the
  // other operand.
  const block_q_t* x = (const block_q_t*)((const char*)vx + exp_idx * exp_stride);
  const block_q8_1* y = (const block_q8_1*)(vy);

  int* tile_x_ql = nullptr;
  half2* tile_x_dm = nullptr;
  int* tile_x_qh = nullptr;
  int* tile_x_sc = nullptr;

  allocate_tiles(&tile_x_ql, &tile_x_dm, &tile_x_qh, &tile_x_sc);

  __shared__ int tile_y_qs[mmq_x * WARP_SIZE_GGUF];
  __shared__ half2 tile_y_ds[mmq_x * WARP_SIZE_GGUF / QI8_1];

  float sum[mmq_y / WARP_SIZE_GGUF][mmq_x / nwarps] = {{0.0f}};

  for (int ib0 = 0; ib0 < blocks_per_row_x; ib0 += blocks_per_warp) {
    load_tiles(
        x + row_x_0 * blocks_per_row_x + ib0,
        tile_x_ql,
        tile_x_dm,
        tile_x_qh,
        tile_x_sc,
        threadIdx.y,
        nrows_x - row_x_0 - 1,
        threadIdx.x,
        blocks_per_row_x,
        blocks_per_row_x - ib0);

    const int n_per_r = ((qk * blocks_per_warp) / qr);
#pragma unroll
    for (int ir = 0; ir < qr && ib0 * qk + ir * n_per_r < ncols_x; ++ir) {
      const auto kqs = ir * WARP_SIZE_GGUF + threadIdx.x;
      const int kbxd = kqs / QI8_1;

#pragma unroll
      for (int i = 0; i < mmq_x; i += nwarps) {
        const int col_y_eff = token_offs[i / nwarps] / top_k;
        const int block_x = ib0 * (qk / QK8_1) + kbxd;
        if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
          const block_q8_1* by0 = &y[col_y_eff * blocks_per_col_y + block_x];
          const int index_y = (threadIdx.y + i) * WARP_SIZE_GGUF + kqs % WARP_SIZE_GGUF;
          tile_y_qs[index_y] = get_int_from_int8_aligned(by0->qs, threadIdx.x % QI8_1);
        }
      }

      // PR #36122: load every destination column's q8 scale. token_offs is local to one warp, so indexing it by a
      // global tile column is invalid when mmq_x > nwarps and out of bounds when mmq_x == nwarps.
#pragma unroll
      for (int ids0 = 0; ids0 < mmq_x; ids0 += nwarps * QI8_1) {
        const int ids = (ids0 + threadIdx.y * QI8_1 + threadIdx.x / (WARP_SIZE_GGUF / QI8_1)) % mmq_x;
        const auto kby = threadIdx.x % (WARP_SIZE_GGUF / QI8_1);
        const int col_y_eff = sorted_token_ids[col_dst_0 + ids] / top_k;
        const int block_x = ib0 * (qk / QK8_1) + ir * (WARP_SIZE_GGUF / QI8_1) + kby;

        if (col_y_eff < ncols_y && block_x < blocks_per_col_y) {
          const half2* dsi_src = &y[col_y_eff * blocks_per_col_y + block_x].ds;
          half2* dsi_dst = &tile_y_ds[ids * (WARP_SIZE_GGUF / QI8_1) + kby];

          if (need_sum) {
            *dsi_dst = *dsi_src;
          } else {
            float* dfi_dst = (float*)dsi_dst;
            *dfi_dst = __low2float(*dsi_src);
          }
        }
      }
      __syncthreads();

      // #pragma unroll // unrolling this loop causes too much register pressure
      for (int k = ir * WARP_SIZE_GGUF / qr; k < (ir + 1) * WARP_SIZE_GGUF / qr; k += vdr) {
#pragma unroll
        for (int j = 0; j < mmq_x; j += nwarps) {
#pragma unroll
          for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
            sum[i / WARP_SIZE_GGUF][j / nwarps] += vec_dot(
                tile_x_ql, tile_x_dm, tile_x_qh, tile_x_sc, tile_y_qs, tile_y_ds, threadIdx.x + i, threadIdx.y + j, k);
          }
        }
      }
      __syncthreads();
    }
  }

#pragma unroll
  for (int j = 0; j < mmq_x; j += nwarps) {
    const int col_dst = token_offs[j / nwarps];
    if (col_dst >= ncols_dst) {
      return;
    }

#pragma unroll
    for (int i = 0; i < mmq_y; i += WARP_SIZE_GGUF) {
      const auto row_dst = row_dst_0 + threadIdx.x + i;
      if (row_dst >= nrows_dst) {
        continue;
      }
      dst[col_dst * nrows_dst + row_dst] = sum[i / WARP_SIZE_GGUF][j / nwarps];
    }
  }
}



// Adapted from
// https://github.com/aimbit-ni/vllm/commit/037e1d547c15313aa7e2bb5fc83390ab0c857314.
#if !defined(USE_MUSA)

#if defined(USE_ROCM)
#define MMQ_X_IQ4_NL 64
#define MMQ_Y_IQ4_NL 128
#define NWARPS_IQ4_NL 8
#else
#define MMQ_X_IQ4_NL 4
#define MMQ_Y_IQ4_NL 32
#define NWARPS_IQ4_NL 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ4_NL, 2)
#endif
    mul_mat_iq4_nl(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst) {
  const int mmq_x = MMQ_X_IQ4_NL;
  const int mmq_y = MMQ_Y_IQ4_NL;
  const int nwarps = NWARPS_IQ4_NL;

  iq_mul_mat_q<
      scalar_t,
      QK4_NL,
      QR4_NL,
      QI4_NL,
      true,
      block_iq4_nl,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq4_nl<mmq_y>,
      load_tiles_iq4_nl<mmq_y, nwarps, need_check>,
      VDR_IQ4_NL_Q8_1_MMQ,
      vec_dot_iq4_nl_q8_1_mul_mat>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template <typename scalar_t>
static void ggml_mul_mat_iq4_nl_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    cudaStream_t stream) {
  const int mmq_x = MMQ_X_IQ4_NL;
  const int mmq_y = MMQ_Y_IQ4_NL;
  const int nwarps = NWARPS_IQ4_NL;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    const bool need_check = false;
    mul_mat_iq4_nl<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  } else {
    const bool need_check = true;
    mul_mat_iq4_nl<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  }
}

#if defined(USE_ROCM)
#define MMQ_X_IQ4_XS 64
#define MMQ_Y_IQ4_XS 128
#define NWARPS_IQ4_XS 8
#else
#define MMQ_X_IQ4_XS 4
#define MMQ_Y_IQ4_XS 32
#define NWARPS_IQ4_XS 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ4_XS, 2)
#endif
    mul_mat_iq4_xs(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst) {
  const int mmq_x = MMQ_X_IQ4_XS;
  const int mmq_y = MMQ_Y_IQ4_XS;
  const int nwarps = NWARPS_IQ4_XS;

  iq_mul_mat_q<
      scalar_t,
      QK_K,
      QR_IQ4_XS_MMQ,
      QI_IQ4_XS_MMQ,
      true,
      block_iq4_xs,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq4_xs<mmq_y>,
      load_tiles_iq4_xs<mmq_y, nwarps, need_check>,
      VDR_IQ4_XS_Q8_1_MMQ,
      vec_dot_iq4_xs_q8_1_mul_mat>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template <typename scalar_t>
static void ggml_mul_mat_iq4_xs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    cudaStream_t stream) {
  const int mmq_x = MMQ_X_IQ4_XS;
  const int mmq_y = MMQ_Y_IQ4_XS;
  const int nwarps = NWARPS_IQ4_XS;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    const bool need_check = false;
    mul_mat_iq4_xs<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  } else {
    const bool need_check = true;
    mul_mat_iq4_xs<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  }
}

#if defined(USE_ROCM)
#define MMQ_X_IQ3_S 64
#define MMQ_Y_IQ3_S 128
#define NWARPS_IQ3_S 8
#else
#define MMQ_X_IQ3_S 4
#define MMQ_Y_IQ3_S 32
#define NWARPS_IQ3_S 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ3_S, 2)
#endif
    mul_mat_iq3_s(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst) {
  const int mmq_x = MMQ_X_IQ3_S;
  const int mmq_y = MMQ_Y_IQ3_S;
  const int nwarps = NWARPS_IQ3_S;

  iq_mul_mat_q<
      scalar_t,
      QK_K,
      QR_IQ3_S_MMQ,
      QI_IQ3_S_MMQ,
      true,
      block_iq3_s,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq3_s<mmq_y>,
      load_tiles_iq3_s<mmq_y, nwarps, need_check>,
      VDR_IQ3_S_Q8_1_MMQ,
      vec_dot_iq3_s_q8_1_mul_mat>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template <typename scalar_t>
static void ggml_mul_mat_iq3_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    cudaStream_t stream) {
  const int mmq_x = MMQ_X_IQ3_S;
  const int mmq_y = MMQ_Y_IQ3_S;
  const int nwarps = NWARPS_IQ3_S;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    const bool need_check = false;
    mul_mat_iq3_s<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  } else {
    const bool need_check = true;
    mul_mat_iq3_s<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  }
}

#if defined(USE_ROCM)
#define MMQ_X_IQ3_XXS 64
#define MMQ_Y_IQ3_XXS 128
#define NWARPS_IQ3_XXS 8
#else
#define MMQ_X_IQ3_XXS 4
#define MMQ_Y_IQ3_XXS 32
#define NWARPS_IQ3_XXS 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ3_XXS, 2)
#endif
    mul_mat_iq3_xxs(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst) {
  const int mmq_x = MMQ_X_IQ3_XXS;
  const int mmq_y = MMQ_Y_IQ3_XXS;
  const int nwarps = NWARPS_IQ3_XXS;

  iq_mul_mat_q<
      scalar_t,
      QK_K,
      QR_IQ3_XXS_MMQ,
      QI_IQ3_XXS_MMQ,
      true,
      block_iq3_xxs,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq3_xxs<mmq_y>,
      load_tiles_iq3_xxs<mmq_y, nwarps, need_check>,
      VDR_IQ3_XXS_Q8_1_MMQ,
      vec_dot_iq3_xxs_q8_1_mul_mat>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template <typename scalar_t>
static void ggml_mul_mat_iq3_xxs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    cudaStream_t stream) {
  const int mmq_x = MMQ_X_IQ3_XXS;
  const int mmq_y = MMQ_Y_IQ3_XXS;
  const int nwarps = NWARPS_IQ3_XXS;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    const bool need_check = false;
    mul_mat_iq3_xxs<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  } else {
    const bool need_check = true;
    mul_mat_iq3_xxs<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  }
}

#if defined(USE_ROCM)
#define MMQ_X_IQ2_XXS 64
#define MMQ_Y_IQ2_XXS 128
#define NWARPS_IQ2_XXS 8
#else
#define MMQ_X_IQ2_XXS 4
#define MMQ_Y_IQ2_XXS 32
#define NWARPS_IQ2_XXS 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ2_XXS, 2)
#endif
    mul_mat_iq2_xxs(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst) {
  const int mmq_x = MMQ_X_IQ2_XXS;
  const int mmq_y = MMQ_Y_IQ2_XXS;
  const int nwarps = NWARPS_IQ2_XXS;

  iq_mul_mat_q<
      scalar_t,
      QK_K,
      QR_IQ2_XXS_MMQ,
      QI_IQ2_XXS_MMQ,
      true,
      block_iq2_xxs,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq2_xxs<mmq_y>,
      load_tiles_iq2_xxs<mmq_y, nwarps, need_check>,
      VDR_IQ2_XXS_Q8_1_MMQ,
      vec_dot_iq2_xxs_q8_1_mul_mat>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template <typename scalar_t>
static void ggml_mul_mat_iq2_xxs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    cudaStream_t stream) {
  const int mmq_x = MMQ_X_IQ2_XXS;
  const int mmq_y = MMQ_Y_IQ2_XXS;
  const int nwarps = NWARPS_IQ2_XXS;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    const bool need_check = false;
    mul_mat_iq2_xxs<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  } else {
    const bool need_check = true;
    mul_mat_iq2_xxs<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  }
}

#if defined(USE_ROCM)
#define MMQ_X_IQ2_XS 64
#define MMQ_Y_IQ2_XS 128
#define NWARPS_IQ2_XS 8
#else
#define MMQ_X_IQ2_XS 4
#define MMQ_Y_IQ2_XS 32
#define NWARPS_IQ2_XS 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ2_XS, 2)
#endif
    mul_mat_iq2_xs(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst) {
  const int mmq_x = MMQ_X_IQ2_XS;
  const int mmq_y = MMQ_Y_IQ2_XS;
  const int nwarps = NWARPS_IQ2_XS;

  iq_mul_mat_q<
      scalar_t,
      QK_K,
      QR_IQ2_XS_MMQ,
      QI_IQ2_XS_MMQ,
      true,
      block_iq2_xs,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq2_xs<mmq_y>,
      load_tiles_iq2_xs<mmq_y, nwarps, need_check>,
      VDR_IQ2_XS_Q8_1_MMQ,
      vec_dot_iq2_xs_q8_1_mul_mat>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template <typename scalar_t>
static void ggml_mul_mat_iq2_xs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    cudaStream_t stream) {
  const int mmq_x = MMQ_X_IQ2_XS;
  const int mmq_y = MMQ_Y_IQ2_XS;
  const int nwarps = NWARPS_IQ2_XS;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    const bool need_check = false;
    mul_mat_iq2_xs<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  } else {
    const bool need_check = true;
    mul_mat_iq2_xs<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  }
}

#if defined(USE_ROCM)
#define MMQ_X_IQ2_S 64
#define MMQ_Y_IQ2_S 128
#define NWARPS_IQ2_S 8
#else
#define MMQ_X_IQ2_S 4
#define MMQ_Y_IQ2_S 32
#define NWARPS_IQ2_S 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ2_S, 2)
#endif
    mul_mat_iq2_s(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst) {
  const int mmq_x = MMQ_X_IQ2_S;
  const int mmq_y = MMQ_Y_IQ2_S;
  const int nwarps = NWARPS_IQ2_S;

  iq_mul_mat_q<
      scalar_t,
      QK_K,
      QR_IQ2_S_MMQ,
      QI_IQ2_S_MMQ,
      true,
      block_iq2_s,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq2_s<mmq_y>,
      load_tiles_iq2_s<mmq_y, nwarps, need_check>,
      VDR_IQ2_S_Q8_1_MMQ,
      vec_dot_iq2_s_q8_1_mul_mat>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template <typename scalar_t>
static void ggml_mul_mat_iq2_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    cudaStream_t stream) {
  const int mmq_x = MMQ_X_IQ2_S;
  const int mmq_y = MMQ_Y_IQ2_S;
  const int nwarps = NWARPS_IQ2_S;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    const bool need_check = false;
    mul_mat_iq2_s<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  } else {
    const bool need_check = true;
    mul_mat_iq2_s<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  }
}

#if defined(USE_ROCM)
#define MMQ_X_IQ1_S 64
#define MMQ_Y_IQ1_S 128
#define NWARPS_IQ1_S 8
#else
#define MMQ_X_IQ1_S 4
#define MMQ_Y_IQ1_S 32
#define NWARPS_IQ1_S 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ1_S, 2)
#endif
    mul_mat_iq1_s(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst) {
  const int mmq_x = MMQ_X_IQ1_S;
  const int mmq_y = MMQ_Y_IQ1_S;
  const int nwarps = NWARPS_IQ1_S;

  iq_mul_mat_q<
      scalar_t,
      QK_K,
      QR_IQ1_S_MMQ,
      QI_IQ1_S_MMQ,
      true,
      block_iq1_s,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq1_s<mmq_y>,
      load_tiles_iq1_s<mmq_y, nwarps, need_check>,
      VDR_IQ1_S_Q8_1_MMQ,
      vec_dot_iq1_s_q8_1_mul_mat>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
}

template <typename scalar_t>
static void ggml_mul_mat_iq1_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    cudaStream_t stream) {
  const int mmq_x = MMQ_X_IQ1_S;
  const int mmq_y = MMQ_Y_IQ1_S;
  const int nwarps = NWARPS_IQ1_S;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = (ncols_y + mmq_x - 1) / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    const bool need_check = false;
    mul_mat_iq1_s<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  } else {
    const bool need_check = true;
    mul_mat_iq1_s<scalar_t, need_check>
        <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols_x, nrows_x, ncols_y, nrows_y, nrows_dst);
  }
}

#endif  // !defined(USE_MUSA)


// Adapted from
// https://github.com/aimbit-ni/vllm/commit/037e1d547c15313aa7e2bb5fc83390ab0c857314.
#if !defined(USE_MUSA)

#if defined(USE_ROCM)
#define MOE_X_IQ4_NL 64
#define MOE_Y_IQ4_NL 128
#define NWARPS_IQ4_NL 8
#else
#define MOE_X_IQ4_NL 4
#define MOE_Y_IQ4_NL 32
#define NWARPS_IQ4_NL 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ4_NL, 2)
#endif
    moe_iq4_nl(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const int64_t exp_stride,
        const int num_experts,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst,
        const int top_k) {
  const int mmq_x = MOE_X_IQ4_NL;
  const int mmq_y = MOE_Y_IQ4_NL;
  const int nwarps = NWARPS_IQ4_NL;

  iq_moe_q<
      scalar_t,
      QK4_NL,
      QR4_NL,
      QI4_NL,
      true,
      block_iq4_nl,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq4_nl<mmq_y>,
      load_tiles_iq4_nl<mmq_y, nwarps, need_check>,
      VDR_IQ4_NL_Q8_1_MMQ,
      vec_dot_iq4_nl_q8_1_mul_mat>(
      vx,
      vy,
      dst,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_padded,
      exp_stride,
      num_experts,
      ncols_x,
      nrows_x,
      ncols_y,
      nrows_y,
      nrows_dst,
      top_k);
}

template <typename scalar_t>
static void ggml_moe_iq4_nl_q8_1_cuda(
    const void* inp,
    const void* w,
    scalar_t* dst,
    const int* sorted_token_ids,
    const int* expert_ids,
    const int* num_tokens_post_padded,
    const int64_t exp_stride,
    const int num_experts,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    const int top_k,
    const int tokens_post_padded,
    cudaStream_t stream) {
  const int mmq_x = MOE_X_IQ4_NL;
  const int mmq_y = MOE_Y_IQ4_NL;
  const int nwarps = NWARPS_IQ4_NL;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = tokens_post_padded / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_iq4_nl<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  } else {
    constexpr bool need_check = true;
    moe_iq4_nl<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  }
}

#if defined(USE_ROCM)
#define MOE_X_IQ4_XS 64
#define MOE_Y_IQ4_XS 128
#define NWARPS_IQ4_XS 8
#else
#define MOE_X_IQ4_XS 4
#define MOE_Y_IQ4_XS 32
#define NWARPS_IQ4_XS 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ4_XS, 2)
#endif
    moe_iq4_xs(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const int64_t exp_stride,
        const int num_experts,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst,
        const int top_k) {
  const int mmq_x = MOE_X_IQ4_XS;
  const int mmq_y = MOE_Y_IQ4_XS;
  const int nwarps = NWARPS_IQ4_XS;

  iq_moe_q<
      scalar_t,
      QK_K,
      QR_IQ4_XS_MMQ,
      QI_IQ4_XS_MMQ,
      true,
      block_iq4_xs,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq4_xs<mmq_y>,
      load_tiles_iq4_xs<mmq_y, nwarps, need_check>,
      VDR_IQ4_XS_Q8_1_MMQ,
      vec_dot_iq4_xs_q8_1_mul_mat>(
      vx,
      vy,
      dst,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_padded,
      exp_stride,
      num_experts,
      ncols_x,
      nrows_x,
      ncols_y,
      nrows_y,
      nrows_dst,
      top_k);
}

template <typename scalar_t>
static void ggml_moe_iq4_xs_q8_1_cuda(
    const void* inp,
    const void* w,
    scalar_t* dst,
    const int* sorted_token_ids,
    const int* expert_ids,
    const int* num_tokens_post_padded,
    const int64_t exp_stride,
    const int num_experts,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    const int top_k,
    const int tokens_post_padded,
    cudaStream_t stream) {
  const int mmq_x = MOE_X_IQ4_XS;
  const int mmq_y = MOE_Y_IQ4_XS;
  const int nwarps = NWARPS_IQ4_XS;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = tokens_post_padded / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_iq4_xs<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  } else {
    constexpr bool need_check = true;
    moe_iq4_xs<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  }
}

#if defined(USE_ROCM)
#define MOE_X_IQ3_S 64
#define MOE_Y_IQ3_S 128
#define NWARPS_IQ3_S_MOE 8
#else
#define MOE_X_IQ3_S 4
#define MOE_Y_IQ3_S 32
#define NWARPS_IQ3_S_MOE 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ3_S_MOE, 2)
#endif
    moe_iq3_s(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const int64_t exp_stride,
        const int num_experts,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst,
        const int top_k) {
  const int mmq_x = MOE_X_IQ3_S;
  const int mmq_y = MOE_Y_IQ3_S;
  const int nwarps = NWARPS_IQ3_S_MOE;

  iq_moe_q<
      scalar_t,
      QK_K,
      QR_IQ3_S_MMQ,
      QI_IQ3_S_MMQ,
      true,
      block_iq3_s,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq3_s<mmq_y>,
      load_tiles_iq3_s<mmq_y, nwarps, need_check>,
      VDR_IQ3_S_Q8_1_MMQ,
      vec_dot_iq3_s_q8_1_mul_mat>(
      vx,
      vy,
      dst,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_padded,
      exp_stride,
      num_experts,
      ncols_x,
      nrows_x,
      ncols_y,
      nrows_y,
      nrows_dst,
      top_k);
}

template <typename scalar_t>
static void ggml_moe_iq3_s_q8_1_cuda(
    const void* inp,
    const void* w,
    scalar_t* dst,
    const int* sorted_token_ids,
    const int* expert_ids,
    const int* num_tokens_post_padded,
    const int64_t exp_stride,
    const int num_experts,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    const int top_k,
    const int tokens_post_padded,
    cudaStream_t stream) {
  const int mmq_x = MOE_X_IQ3_S;
  const int mmq_y = MOE_Y_IQ3_S;
  const int nwarps = NWARPS_IQ3_S_MOE;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = tokens_post_padded / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_iq3_s<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  } else {
    constexpr bool need_check = true;
    moe_iq3_s<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  }
}

#if defined(USE_ROCM)
#define MOE_X_IQ3_XXS 64
#define MOE_Y_IQ3_XXS 128
#define NWARPS_IQ3_XXS_MOE 8
#else
#define MOE_X_IQ3_XXS 4
#define MOE_Y_IQ3_XXS 32
#define NWARPS_IQ3_XXS_MOE 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ3_XXS_MOE, 2)
#endif
    moe_iq3_xxs(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const int64_t exp_stride,
        const int num_experts,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst,
        const int top_k) {
  const int mmq_x = MOE_X_IQ3_XXS;
  const int mmq_y = MOE_Y_IQ3_XXS;
  const int nwarps = NWARPS_IQ3_XXS_MOE;

  iq_moe_q<
      scalar_t,
      QK_K,
      QR_IQ3_XXS_MMQ,
      QI_IQ3_XXS_MMQ,
      true,
      block_iq3_xxs,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq3_xxs<mmq_y>,
      load_tiles_iq3_xxs<mmq_y, nwarps, need_check>,
      VDR_IQ3_XXS_Q8_1_MMQ,
      vec_dot_iq3_xxs_q8_1_mul_mat>(
      vx,
      vy,
      dst,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_padded,
      exp_stride,
      num_experts,
      ncols_x,
      nrows_x,
      ncols_y,
      nrows_y,
      nrows_dst,
      top_k);
}

template <typename scalar_t>
static void ggml_moe_iq3_xxs_q8_1_cuda(
    const void* inp,
    const void* w,
    scalar_t* dst,
    const int* sorted_token_ids,
    const int* expert_ids,
    const int* num_tokens_post_padded,
    const int64_t exp_stride,
    const int num_experts,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    const int top_k,
    const int tokens_post_padded,
    cudaStream_t stream) {
  const int mmq_x = MOE_X_IQ3_XXS;
  const int mmq_y = MOE_Y_IQ3_XXS;
  const int nwarps = NWARPS_IQ3_XXS_MOE;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = tokens_post_padded / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_iq3_xxs<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  } else {
    constexpr bool need_check = true;
    moe_iq3_xxs<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  }
}

#if defined(USE_ROCM)
#define MOE_X_IQ2_XXS 64
#define MOE_Y_IQ2_XXS 128
#define NWARPS_IQ2_XXS_MOE 8
#else
#define MOE_X_IQ2_XXS 4
#define MOE_Y_IQ2_XXS 32
#define NWARPS_IQ2_XXS_MOE 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ2_XXS_MOE, 2)
#endif
    moe_iq2_xxs(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const int64_t exp_stride,
        const int num_experts,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst,
        const int top_k) {
  const int mmq_x = MOE_X_IQ2_XXS;
  const int mmq_y = MOE_Y_IQ2_XXS;
  const int nwarps = NWARPS_IQ2_XXS_MOE;

  iq_moe_q<
      scalar_t,
      QK_K,
      QR_IQ2_XXS_MMQ,
      QI_IQ2_XXS_MMQ,
      true,
      block_iq2_xxs,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq2_xxs<mmq_y>,
      load_tiles_iq2_xxs<mmq_y, nwarps, need_check>,
      VDR_IQ2_XXS_Q8_1_MMQ,
      vec_dot_iq2_xxs_q8_1_mul_mat>(
      vx,
      vy,
      dst,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_padded,
      exp_stride,
      num_experts,
      ncols_x,
      nrows_x,
      ncols_y,
      nrows_y,
      nrows_dst,
      top_k);
}

template <typename scalar_t>
static void ggml_moe_iq2_xxs_q8_1_cuda(
    const void* inp,
    const void* w,
    scalar_t* dst,
    const int* sorted_token_ids,
    const int* expert_ids,
    const int* num_tokens_post_padded,
    const int64_t exp_stride,
    const int num_experts,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    const int top_k,
    const int tokens_post_padded,
    cudaStream_t stream) {
  const int mmq_x = MOE_X_IQ2_XXS;
  const int mmq_y = MOE_Y_IQ2_XXS;
  const int nwarps = NWARPS_IQ2_XXS_MOE;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = tokens_post_padded / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_iq2_xxs<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  } else {
    constexpr bool need_check = true;
    moe_iq2_xxs<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  }
}

#if defined(USE_ROCM)
#define MOE_X_IQ2_XS 64
#define MOE_Y_IQ2_XS 128
#define NWARPS_IQ2_XS_MOE 8
#else
#define MOE_X_IQ2_XS 4
#define MOE_Y_IQ2_XS 32
#define NWARPS_IQ2_XS_MOE 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ2_XS_MOE, 2)
#endif
    moe_iq2_xs(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const int64_t exp_stride,
        const int num_experts,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst,
        const int top_k) {
  const int mmq_x = MOE_X_IQ2_XS;
  const int mmq_y = MOE_Y_IQ2_XS;
  const int nwarps = NWARPS_IQ2_XS_MOE;

  iq_moe_q<
      scalar_t,
      QK_K,
      QR_IQ2_XS_MMQ,
      QI_IQ2_XS_MMQ,
      true,
      block_iq2_xs,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq2_xs<mmq_y>,
      load_tiles_iq2_xs<mmq_y, nwarps, need_check>,
      VDR_IQ2_XS_Q8_1_MMQ,
      vec_dot_iq2_xs_q8_1_mul_mat>(
      vx,
      vy,
      dst,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_padded,
      exp_stride,
      num_experts,
      ncols_x,
      nrows_x,
      ncols_y,
      nrows_y,
      nrows_dst,
      top_k);
}

template <typename scalar_t>
static void ggml_moe_iq2_xs_q8_1_cuda(
    const void* inp,
    const void* w,
    scalar_t* dst,
    const int* sorted_token_ids,
    const int* expert_ids,
    const int* num_tokens_post_padded,
    const int64_t exp_stride,
    const int num_experts,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    const int top_k,
    const int tokens_post_padded,
    cudaStream_t stream) {
  const int mmq_x = MOE_X_IQ2_XS;
  const int mmq_y = MOE_Y_IQ2_XS;
  const int nwarps = NWARPS_IQ2_XS_MOE;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = tokens_post_padded / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_iq2_xs<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  } else {
    constexpr bool need_check = true;
    moe_iq2_xs<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  }
}

#if defined(USE_ROCM)
#define MOE_X_IQ2_S 64
#define MOE_Y_IQ2_S 128
#define NWARPS_IQ2_S_MOE 8
#else
#define MOE_X_IQ2_S 4
#define MOE_Y_IQ2_S 32
#define NWARPS_IQ2_S_MOE 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ2_S_MOE, 2)
#endif
    moe_iq2_s(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const int64_t exp_stride,
        const int num_experts,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst,
        const int top_k) {
  const int mmq_x = MOE_X_IQ2_S;
  const int mmq_y = MOE_Y_IQ2_S;
  const int nwarps = NWARPS_IQ2_S_MOE;

  iq_moe_q<
      scalar_t,
      QK_K,
      QR_IQ2_S_MMQ,
      QI_IQ2_S_MMQ,
      true,
      block_iq2_s,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq2_s<mmq_y>,
      load_tiles_iq2_s<mmq_y, nwarps, need_check>,
      VDR_IQ2_S_Q8_1_MMQ,
      vec_dot_iq2_s_q8_1_mul_mat>(
      vx,
      vy,
      dst,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_padded,
      exp_stride,
      num_experts,
      ncols_x,
      nrows_x,
      ncols_y,
      nrows_y,
      nrows_dst,
      top_k);
}

template <typename scalar_t>
static void ggml_moe_iq2_s_q8_1_cuda(
    const void* inp,
    const void* w,
    scalar_t* dst,
    const int* sorted_token_ids,
    const int* expert_ids,
    const int* num_tokens_post_padded,
    const int64_t exp_stride,
    const int num_experts,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    const int top_k,
    const int tokens_post_padded,
    cudaStream_t stream) {
  const int mmq_x = MOE_X_IQ2_S;
  const int mmq_y = MOE_Y_IQ2_S;
  const int nwarps = NWARPS_IQ2_S_MOE;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = tokens_post_padded / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_iq2_s<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  } else {
    constexpr bool need_check = true;
    moe_iq2_s<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  }
}

#if defined(USE_ROCM)
#define MOE_X_IQ1_S 64
#define MOE_Y_IQ1_S 128
#define NWARPS_IQ1_S_MOE 8
#else
#define MOE_X_IQ1_S 4
#define MOE_Y_IQ1_S 32
#define NWARPS_IQ1_S_MOE 4
#endif

template <typename scalar_t, bool need_check>
static __global__ void
#if defined(USE_ROCM)
__launch_bounds__(WARP_SIZE_GGUF* NWARPS_IQ1_S_MOE, 2)
#endif
    moe_iq1_s(
        const void* __restrict__ vx,
        const void* __restrict__ vy,
        scalar_t* __restrict__ dst,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const int64_t exp_stride,
        const int num_experts,
        const int ncols_x,
        const int nrows_x,
        const int ncols_y,
        const int nrows_y,
        const int nrows_dst,
        const int top_k) {
  const int mmq_x = MOE_X_IQ1_S;
  const int mmq_y = MOE_Y_IQ1_S;
  const int nwarps = NWARPS_IQ1_S_MOE;

  iq_moe_q<
      scalar_t,
      QK_K,
      QR_IQ1_S_MMQ,
      QI_IQ1_S_MMQ,
      true,
      block_iq1_s,
      mmq_x,
      mmq_y,
      nwarps,
      allocate_tiles_iq1_s<mmq_y>,
      load_tiles_iq1_s<mmq_y, nwarps, need_check>,
      VDR_IQ1_S_Q8_1_MMQ,
      vec_dot_iq1_s_q8_1_mul_mat>(
      vx,
      vy,
      dst,
      sorted_token_ids,
      expert_ids,
      num_tokens_post_padded,
      exp_stride,
      num_experts,
      ncols_x,
      nrows_x,
      ncols_y,
      nrows_y,
      nrows_dst,
      top_k);
}

template <typename scalar_t>
static void ggml_moe_iq1_s_q8_1_cuda(
    const void* inp,
    const void* w,
    scalar_t* dst,
    const int* sorted_token_ids,
    const int* expert_ids,
    const int* num_tokens_post_padded,
    const int64_t exp_stride,
    const int num_experts,
    const int ncols_x,
    const int nrows_x,
    const int ncols_y,
    const int nrows_y,
    const int nrows_dst,
    const int top_k,
    const int tokens_post_padded,
    cudaStream_t stream) {
  const int mmq_x = MOE_X_IQ1_S;
  const int mmq_y = MOE_Y_IQ1_S;
  const int nwarps = NWARPS_IQ1_S_MOE;

  const int block_num_x = (nrows_x + mmq_y - 1) / mmq_y;
  const int block_num_y = tokens_post_padded / mmq_x;
  const dim3 block_nums(block_num_x, block_num_y, 1);
  const dim3 block_dims(WARP_SIZE_GGUF, nwarps, 1);

  if (nrows_x % mmq_y == 0) {
    constexpr bool need_check = false;
    moe_iq1_s<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  } else {
    constexpr bool need_check = true;
    moe_iq1_s<scalar_t, need_check><<<block_nums, block_dims, 0, stream>>>(
        w,
        inp,
        dst,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        exp_stride,
        num_experts,
        ncols_x,
        nrows_x,
        ncols_y,
        nrows_y,
        nrows_dst,
        top_k);
  }
}

#endif  // !defined(USE_MUSA)
