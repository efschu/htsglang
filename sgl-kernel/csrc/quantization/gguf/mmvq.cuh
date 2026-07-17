// copied from
// https://github.com/vllm-project/vllm/blob/4492e3a55428e161ca8db381edc28263e5da4c8d/csrc/quantization/gguf/mmvq.cuh
// copied and adapted from https://github.com/ggerganov/llama.cpp/blob/b2899/ggml-cuda/mmvq.cu
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void mul_mat_vec_q(
    const void* __restrict__ vx,
    const void* __restrict__ vy,
    scalar_t* __restrict__ dst,
    const int ncols,
    const int nrows,
    const int nvecs) {
  const auto row = blockIdx.x * blockDim.y + threadIdx.y;
  const auto vec = blockIdx.y;

  if (row >= nrows || vec >= nvecs) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;
  const int nrows_y = (ncols + 512 - 1) / 512 * 512;

  // partial sum for each thread
  float tmp = 0.0f;

  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;

  for (auto i = threadIdx.x / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i;  // x block index

    const int iby = vec * (nrows_y / QK8_1) + i * (qk / QK8_1);  // y block index that aligns with ibx

    const int iqs = vdr * (threadIdx.x % (qi / vdr));  // x block quant index when casting the quants to int

    tmp += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
  }

  // sum up partial sums and write back result
#pragma unroll
  for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
    tmp += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), tmp, mask);
  }

  if (threadIdx.x == 0) {
    dst[vec * nrows + row] = tmp;
  }
}

// Batched MMVQ, adapted from llama.cpp's modern mul_mat_vec_q (ncols_dst
// template, ggml-cuda/mmvq.cu): ONE weight-block load is dotted against up
// to 8 quantized activation columns. The legacy kernel above instead puts
// each activation column into its own blockIdx.y slice, re-streaming the
// full weight matrix per column -- cost grows ~linearly with the token
// count M, which is exactly the spec-decode verify shape (M =
// num_draft_tokens, typically 4). With the column loop unrolled inside the
// block loop the compiler keeps the x-block loads in registers across the j
// iterations (same address, no intervening writes), so M<=8 costs close to
// M=1.
//
// Numerics: for a given column j the accumulation order is IDENTICAL to the
// legacy kernel (same block traversal, same warp reduction), so results are
// bit-identical -- including ncols_dst=1 vs. the old M=1 path.
template <
    typename scalar_t,
    int ncols_dst,
    int qk,
    int qi,
    typename block_q_t,
    int vdr,
    vec_dot_q_cuda_t vec_dot_q_cuda>
static __global__ void mul_mat_vec_q_nc(
    const void* __restrict__ vx, const void* __restrict__ vy, scalar_t* __restrict__ dst, const int ncols, const int nrows) {
  const auto row = blockIdx.x * blockDim.y + threadIdx.y;

  if (row >= nrows) {
    return;
  }

  const int blocks_per_row = ncols / qk;
  const int blocks_per_warp = vdr * WARP_SIZE / qi;
  const int nrows_y = (ncols + 512 - 1) / 512 * 512;

  // partial sums for each thread, one per activation column
  float tmp[ncols_dst];
#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
    tmp[j] = 0.0f;
  }

  const block_q_t* x = (const block_q_t*)vx;
  const block_q8_1* y = (const block_q8_1*)vy;

  for (auto i = threadIdx.x / (qi / vdr); i < blocks_per_row; i += blocks_per_warp) {
    const int ibx = row * blocks_per_row + i;  // x block index

    const int iqs = vdr * (threadIdx.x % (qi / vdr));  // x block quant index when casting the quants to int

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
      const int iby = j * (nrows_y / QK8_1) + i * (qk / QK8_1);  // y block index that aligns with ibx
      tmp[j] += vec_dot_q_cuda(&x[ibx], &y[iby], iqs);
    }
  }

  // sum up partial sums and write back results
#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
      tmp[j] += SGLANG_SHFL_XOR_SYNC(uint32_t(-1), tmp[j], mask);
    }
    if (threadIdx.x == 0) {
      dst[j * nrows + row] = tmp[j];
    }
  }
}

// Launch helper: nvecs<=8 takes the batched single-pass kernel (grid.y == 1),
// larger nvecs falls back to the legacy per-column kernel. All per-type
// launchers below route through this dispatcher.
template <typename scalar_t, int qk, int qi, typename block_q_t, int vdr, vec_dot_q_cuda_t vec_dot_q_cuda>
static void mul_mat_vec_q_dispatch(
    const void* vx, const void* vy, scalar_t* dst, const int ncols, const int nrows, const int nvecs, cudaStream_t stream) {
  const int block_num_y = (nrows + GGML_CUDA_MMV_Y - 1) / GGML_CUDA_MMV_Y;
  const dim3 block_dims(WARP_SIZE, GGML_CUDA_MMV_Y, 1);
  const dim3 block_nums_nc(block_num_y, 1, 1);
  switch (nvecs) {
#define SGLANG_MMVQ_NC_CASE(NC)                                                 \
  case NC:                                                                      \
    mul_mat_vec_q_nc<scalar_t, NC, qk, qi, block_q_t, vdr, vec_dot_q_cuda>      \
        <<<block_nums_nc, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows);  \
    break;
    SGLANG_MMVQ_NC_CASE(1)
    SGLANG_MMVQ_NC_CASE(2)
    SGLANG_MMVQ_NC_CASE(3)
    SGLANG_MMVQ_NC_CASE(4)
    SGLANG_MMVQ_NC_CASE(5)
    SGLANG_MMVQ_NC_CASE(6)
    SGLANG_MMVQ_NC_CASE(7)
    SGLANG_MMVQ_NC_CASE(8)
#undef SGLANG_MMVQ_NC_CASE
    default: {
      const dim3 block_nums(block_num_y, nvecs, 1);
      mul_mat_vec_q<scalar_t, qk, qi, block_q_t, vdr, vec_dot_q_cuda>
          <<<block_nums, block_dims, 0, stream>>>(vx, vy, dst, ncols, nrows, nvecs);
    } break;
  }
}

template <typename scalar_t>
static void mul_mat_vec_q4_0_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK4_0, QI4_0, block_q4_0, VDR_Q4_0_Q8_1_MMVQ, vec_dot_q4_0_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q4_1_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK4_0, QI4_1, block_q4_1, VDR_Q4_1_Q8_1_MMVQ, vec_dot_q4_1_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q5_0_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK5_0, QI5_0, block_q5_0, VDR_Q5_0_Q8_1_MMVQ, vec_dot_q5_0_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q5_1_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK5_1, QI5_1, block_q5_1, VDR_Q5_1_Q8_1_MMVQ, vec_dot_q5_1_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q8_0_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK8_0, QI8_0, block_q8_0, VDR_Q8_0_Q8_1_MMVQ, vec_dot_q8_0_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q2_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK_K, QI2_K, block_q2_K, VDR_Q2_K_Q8_1_MMVQ, vec_dot_q2_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q3_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK_K, QI3_K, block_q3_K, VDR_Q3_K_Q8_1_MMVQ, vec_dot_q3_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q4_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK_K, QI4_K, block_q4_K, VDR_Q4_K_Q8_1_MMVQ, vec_dot_q4_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q5_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK_K, QI5_K, block_q5_K, VDR_Q5_K_Q8_1_MMVQ, vec_dot_q5_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_q6_K_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK_K, QI6_K, block_q6_K, VDR_Q6_K_Q8_1_MMVQ, vec_dot_q6_K_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq2_xxs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK_K, QI2_XXS, block_iq2_xxs, 1, vec_dot_iq2_xxs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq2_xs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK_K, QI2_XS, block_iq2_xs, 1, vec_dot_iq2_xs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq2_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK_K, QI2_S, block_iq2_s, 1, vec_dot_iq2_s_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq3_xxs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK_K, QI3_XXS, block_iq3_xxs, 1, vec_dot_iq3_xxs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq1_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK_K, QI1_S, block_iq1_s, 1, vec_dot_iq1_s_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq1_m_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK_K, QI1_M, block_iq1_m, 1, vec_dot_iq1_m_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq4_nl_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK4_NL, QI4_NL, block_iq4_nl, VDR_Q4_0_Q8_1_MMVQ, vec_dot_iq4_nl_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq4_xs_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK_K, QI4_XS, block_iq4_xs, 1, vec_dot_iq4_xs_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}

template <typename scalar_t>
static void mul_mat_vec_iq3_s_q8_1_cuda(
    const void* vx,
    const void* vy,
    scalar_t* dst,
    const int ncols,
    const int nrows,
    const int nvecs,
    cudaStream_t stream) {
  mul_mat_vec_q_dispatch<scalar_t, QK_K, QI3_XS, block_iq3_s, 1, vec_dot_iq3_s_q8_1>(vx, vy, dst, ncols, nrows, nvecs, stream);
}
