// GGUF-NF G6 (2026-10-09): host side (tvm-ffi, torch-free) of the IQ-type MMQ / MoE-MMQ JIT module.
// Kernels: PR sgl-project/sglang#36122 (see iq_mmq_kernels.cuh). The host code below is NEW (the PR's host code is torch AOT
// code in gguf_kernel.cu); it keeps the PR's semantics: q8_1 activation quantisation with K padded to 512, one launch
// per ggml type, dense (ggml_mul_mat_a8 analogue) and MoE (ggml_moe_a8 analogue) entry points.
// Output and q8_1 scratch are allocated by the Python caller (jit_kernel/gguf_iq_mmq.py).
//
// GGUF_IQ_MMQ_TYPE_MASK: bit (type - 16) set = the kernels of that ggml type are compiled into THIS module (one module
// per type keeps the nvcc time of a cold JIT cache at ~1/8; a type not in the mask is refused at run time).
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/utils.h>

#include <algorithm>
#include <cstdint>

#include "iq_mmq_kernels.cuh"

#ifndef GGUF_IQ_MMQ_TYPE_MASK
#define GGUF_IQ_MMQ_TYPE_MASK 0xFF
#endif

namespace gguf_iq_mmq {

template <int kType>
inline constexpr bool kWanted = kType >= 16 && kType <= 23 && ((GGUF_IQ_MMQ_TYPE_MASK >> (kType - 16)) & 1);

// Required input size (K) multiple: 256 = one super block for every QK_K type; IQ4_NL (32-element blocks) takes
// K = 128 * n because its tile loader zero-fills the blocks past the row end (see iq_mmq_tiles.cuh).
inline int k_alignment(int64_t type) {
  switch (type) {
    case 16: return 256;  // IQ2_XXS
    case 17: return 256;  // IQ2_XS
    case 18: return 256;  // IQ3_XXS
    case 19: return 256;  // IQ1_S
    case 20: return 128;  // IQ4_NL
    case 21: return 256;  // IQ3_S
    case 22: return 256;  // IQ2_S
    case 23: return 256;  // IQ4_XS
    default: return 0;
  }
}

// q8_1 activation quantiser: verbatim copy of quantize_q8_1 / quantize_row_q8_1_cuda of
// sgl-kernel/csrc/quantization/gguf/gguf_kernel.cu (base c651892375); SGLANG_SHFL_XOR_SYNC_WIDTH -> __shfl_xor_sync.
template <typename scalar_t>
static __global__ void quantize_q8_1(const scalar_t* __restrict__ x, void* __restrict__ vy, const int kx, const int kx_padded) {
  const auto ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const auto iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? static_cast<float>(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, mask, 32));
    sum += __shfl_xor_sync(0xffffffffu, sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

template <typename scalar_t>
static void quantize_row_q8_1_cuda(const scalar_t* x, void* vy, const int kx, const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x = (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  constexpr int MAX_BLOCK_SIZE = 65535;
  for (int off = 0; off < ky; off += MAX_BLOCK_SIZE) {
    const int num_blocks_y = std::min(ky, off + MAX_BLOCK_SIZE) - off;
    const dim3 num_blocks(block_num_x, num_blocks_y, 1);
    const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
    quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(
        &x[(int64_t)off * kx], (int32_t*)vy + (int64_t)off * (kx_padded / 32 * 9), kx, kx_padded);
  }
}

template <int kType, typename scalar_t>
inline void dense_one(
    const void* w, const void* qx, scalar_t* y, int col, int row, int batch, int padded, cudaStream_t stream) {
  if constexpr (kType == 16 && kWanted<16>) {
    ggml_mul_mat_iq2_xxs_q8_1_cuda<scalar_t>(w, qx, y, col, row, batch, padded, row, stream);
  } else   if constexpr (kType == 17 && kWanted<17>) {
    ggml_mul_mat_iq2_xs_q8_1_cuda<scalar_t>(w, qx, y, col, row, batch, padded, row, stream);
  } else   if constexpr (kType == 18 && kWanted<18>) {
    ggml_mul_mat_iq3_xxs_q8_1_cuda<scalar_t>(w, qx, y, col, row, batch, padded, row, stream);
  } else   if constexpr (kType == 19 && kWanted<19>) {
    ggml_mul_mat_iq1_s_q8_1_cuda<scalar_t>(w, qx, y, col, row, batch, padded, row, stream);
  } else   if constexpr (kType == 20 && kWanted<20>) {
    ggml_mul_mat_iq4_nl_q8_1_cuda<scalar_t>(w, qx, y, col, row, batch, padded, row, stream);
  } else   if constexpr (kType == 21 && kWanted<21>) {
    ggml_mul_mat_iq3_s_q8_1_cuda<scalar_t>(w, qx, y, col, row, batch, padded, row, stream);
  } else   if constexpr (kType == 22 && kWanted<22>) {
    ggml_mul_mat_iq2_s_q8_1_cuda<scalar_t>(w, qx, y, col, row, batch, padded, row, stream);
  } else   if constexpr (kType == 23 && kWanted<23>) {
    ggml_mul_mat_iq4_xs_q8_1_cuda<scalar_t>(w, qx, y, col, row, batch, padded, row, stream);
  } else {
    host::Panic("IQ MMQ: ggml type ", kType, " is not compiled into this module (GGUF_IQ_MMQ_TYPE_MASK)");
  }
}

template <typename scalar_t>
inline void dense_dispatch(
    int64_t type, const void* w, const void* qx, scalar_t* y, int col, int row, int batch, int padded, cudaStream_t stream) {
  switch (type) {
    case 16: dense_one<16, scalar_t>(w, qx, y, col, row, batch, padded, stream); break;  // IQ2_XXS
    case 17: dense_one<17, scalar_t>(w, qx, y, col, row, batch, padded, stream); break;  // IQ2_XS
    case 18: dense_one<18, scalar_t>(w, qx, y, col, row, batch, padded, stream); break;  // IQ3_XXS
    case 19: dense_one<19, scalar_t>(w, qx, y, col, row, batch, padded, stream); break;  // IQ1_S
    case 20: dense_one<20, scalar_t>(w, qx, y, col, row, batch, padded, stream); break;  // IQ4_NL
    case 21: dense_one<21, scalar_t>(w, qx, y, col, row, batch, padded, stream); break;  // IQ3_S
    case 22: dense_one<22, scalar_t>(w, qx, y, col, row, batch, padded, stream); break;  // IQ2_S
    case 23: dense_one<23, scalar_t>(w, qx, y, col, row, batch, padded, stream); break;  // IQ4_XS
    default: host::Panic("IQ MMQ: unsupported ggml type ", type);
  }
}

template <int kType, typename scalar_t>
inline void moe_one(
    const void* qx,
    const void* w,
    scalar_t* y,
    const int* sorted_token_ids,
    const int* expert_ids,
    const int* num_tokens_post_padded,
    int64_t exp_stride,
    int num_experts,
    int col,
    int row,
    int tokens,
    int padded,
    int top_k,
    int tokens_post_padded,
    cudaStream_t stream) {
  if constexpr (kType == 16 && kWanted<16>) {
    ggml_moe_iq2_xxs_q8_1_cuda<scalar_t>(
        qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, row, top_k, tokens_post_padded, stream);
  } else   if constexpr (kType == 17 && kWanted<17>) {
    ggml_moe_iq2_xs_q8_1_cuda<scalar_t>(
        qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, row, top_k, tokens_post_padded, stream);
  } else   if constexpr (kType == 18 && kWanted<18>) {
    ggml_moe_iq3_xxs_q8_1_cuda<scalar_t>(
        qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, row, top_k, tokens_post_padded, stream);
  } else   if constexpr (kType == 19 && kWanted<19>) {
    ggml_moe_iq1_s_q8_1_cuda<scalar_t>(
        qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, row, top_k, tokens_post_padded, stream);
  } else   if constexpr (kType == 20 && kWanted<20>) {
    ggml_moe_iq4_nl_q8_1_cuda<scalar_t>(
        qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, row, top_k, tokens_post_padded, stream);
  } else   if constexpr (kType == 21 && kWanted<21>) {
    ggml_moe_iq3_s_q8_1_cuda<scalar_t>(
        qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, row, top_k, tokens_post_padded, stream);
  } else   if constexpr (kType == 22 && kWanted<22>) {
    ggml_moe_iq2_s_q8_1_cuda<scalar_t>(
        qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, row, top_k, tokens_post_padded, stream);
  } else   if constexpr (kType == 23 && kWanted<23>) {
    ggml_moe_iq4_xs_q8_1_cuda<scalar_t>(
        qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, row, top_k, tokens_post_padded, stream);
  } else {
    host::Panic("IQ MoE MMQ: ggml type ", kType, " is not compiled into this module (GGUF_IQ_MMQ_TYPE_MASK)");
  }
}

template <typename scalar_t>
inline void moe_dispatch(
    int64_t type,
    const void* qx,
    const void* w,
    scalar_t* y,
    const int* sorted_token_ids,
    const int* expert_ids,
    const int* num_tokens_post_padded,
    int64_t exp_stride,
    int num_experts,
    int col,
    int row,
    int tokens,
    int padded,
    int top_k,
    int tokens_post_padded,
    cudaStream_t stream) {
  switch (type) {
    case 16: moe_one<16, scalar_t>(qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, top_k, tokens_post_padded, stream); break;  // IQ2_XXS
    case 17: moe_one<17, scalar_t>(qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, top_k, tokens_post_padded, stream); break;  // IQ2_XS
    case 18: moe_one<18, scalar_t>(qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, top_k, tokens_post_padded, stream); break;  // IQ3_XXS
    case 19: moe_one<19, scalar_t>(qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, top_k, tokens_post_padded, stream); break;  // IQ1_S
    case 20: moe_one<20, scalar_t>(qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, top_k, tokens_post_padded, stream); break;  // IQ4_NL
    case 21: moe_one<21, scalar_t>(qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, top_k, tokens_post_padded, stream); break;  // IQ3_S
    case 22: moe_one<22, scalar_t>(qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, top_k, tokens_post_padded, stream); break;  // IQ2_S
    case 23: moe_one<23, scalar_t>(qx, w, y, sorted_token_ids, expert_ids, num_tokens_post_padded, exp_stride, num_experts, col, row, tokens, padded, top_k, tokens_post_padded, stream); break;  // IQ4_XS
    default: host::Panic("IQ MoE MMQ: unsupported ggml type ", type);
  }
}

inline int64_t padded_k(int64_t col) {
  return (col + 512 - 1) / 512 * 512;
}

// W [row, bytes_per_row] uint8 (rows contiguous), X [batch, col] fp16/bf16 (contiguous), Y [batch, row] like X,
// quant_X int32 [batch, padded(col) / 32 * 9] scratch.
inline void mul_mat_a8(
    tvm::ffi::TensorView W,
    tvm::ffi::TensorView X,
    tvm::ffi::TensorView Y,
    tvm::ffi::TensorView quant_X,
    int64_t type,
    int64_t row) {
  using namespace host;
  RuntimeCheck(X.device().device_type == kDLCUDA, "X must be a CUDA tensor");
  RuntimeCheck(W.device() == X.device() && Y.device() == X.device() && quant_X.device() == X.device(), "tensors on different devices");
  RuntimeCheck(W.dim() == 2 && X.dim() == 2 && Y.dim() == 2 && quant_X.dim() == 2, "W, X, Y, quant_X must be 2D");
  RuntimeCheck(is_type<uint8_t>(W.dtype()), "W must be uint8 (raw ggml blocks)");
  RuntimeCheck(is_type<fp16_t>(X.dtype()) || is_type<bf16_t>(X.dtype()), "X must be fp16 or bf16");
  RuntimeCheck(X.dtype() == Y.dtype(), "Y dtype must equal X dtype");
  RuntimeCheck(is_type<int32_t>(quant_X.dtype()), "quant_X must be int32");
  RuntimeCheck(X.stride(1) == 1 && X.stride(0) == X.size(1), "X must be contiguous");
  RuntimeCheck(Y.stride(1) == 1 && Y.stride(0) == Y.size(1), "Y must be contiguous");
  RuntimeCheck(W.stride(1) == 1 && W.stride(0) == W.size(1), "W rows must be contiguous");
  const int64_t col = X.size(1);
  const int64_t batch = X.size(0);
  const int align = k_alignment(type);
  RuntimeCheck(align > 0, "IQ MMQ: unsupported ggml type ", type);
  RuntimeCheck(col % align == 0, "IQ MMQ type ", type, " requires an input size divisible by ", align, ", got ", col);
  RuntimeCheck(W.size(0) == row, "W rows (", W.size(0), ") != row (", row, ")");
  RuntimeCheck(Y.size(0) == batch && Y.size(1) == row, "Y must be [batch, row]");
  const int64_t padded = padded_k(col);
  RuntimeCheck(quant_X.size(0) >= batch && quant_X.size(1) >= padded / 32 * 9, "quant_X scratch too small");
  RuntimeCheck(batch < (1LL << 30) && row < (1LL << 30) && padded < (1LL << 30), "dimensions exceed int32");
  if (batch == 0) {
    return;
  }
  const cudaStream_t stream = LaunchKernel::resolve_device(X.device());
  const auto launch = [&](auto tag) {
    using scalar_t = decltype(tag);
    quantize_row_q8_1_cuda<scalar_t>(
        static_cast<const scalar_t*>(X.data_ptr()), quant_X.data_ptr(), static_cast<int>(col), static_cast<int>(batch), stream);
    dense_dispatch<scalar_t>(
        type,
        W.data_ptr(),
        quant_X.data_ptr(),
        static_cast<scalar_t*>(Y.data_ptr()),
        static_cast<int>(col),
        static_cast<int>(row),
        static_cast<int>(batch),
        static_cast<int>(padded),
        stream);
  };
  if (is_type<fp16_t>(X.dtype())) {
    launch(fp16_t{});
  } else {
    launch(bf16_t{});
  }
  RuntimeDeviceCheck(cudaGetLastError());
}

// X [tokens, col] fp16/bf16 (contiguous); W [E, row, bytes_per_row] uint8 (experts stride W.stride(0) bytes);
// sorted_token_ids / expert_ids / num_tokens_post_padded: output of moe_align_block_size (block size 4);
// Y [tokens * top_k, row] like X; quant_X int32 [tokens, padded(col) / 32 * 9] scratch.
inline void moe_a8(
    tvm::ffi::TensorView X,
    tvm::ffi::TensorView W,
    tvm::ffi::TensorView sorted_token_ids,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_tokens_post_padded,
    tvm::ffi::TensorView Y,
    tvm::ffi::TensorView quant_X,
    int64_t type,
    int64_t row,
    int64_t top_k,
    int64_t tokens) {
  using namespace host;
  RuntimeCheck(X.device().device_type == kDLCUDA, "X must be a CUDA tensor");
  RuntimeCheck(
      W.device() == X.device() && Y.device() == X.device() && quant_X.device() == X.device() &&
          sorted_token_ids.device() == X.device() && expert_ids.device() == X.device() &&
          num_tokens_post_padded.device() == X.device(),
      "tensors on different devices");
  RuntimeCheck(X.dim() == 2 && W.dim() == 3 && Y.dim() == 2 && quant_X.dim() == 2, "bad tensor ranks");
  RuntimeCheck(sorted_token_ids.dim() == 1 && expert_ids.dim() == 1, "sorted_token_ids / expert_ids must be 1D");
  RuntimeCheck(is_type<uint8_t>(W.dtype()), "W must be uint8 (raw ggml blocks)");
  RuntimeCheck(is_type<fp16_t>(X.dtype()) || is_type<bf16_t>(X.dtype()), "X must be fp16 or bf16");
  RuntimeCheck(X.dtype() == Y.dtype(), "Y dtype must equal X dtype");
  RuntimeCheck(is_type<int32_t>(quant_X.dtype()), "quant_X must be int32");
  RuntimeCheck(
      is_type<int32_t>(sorted_token_ids.dtype()) && is_type<int32_t>(expert_ids.dtype()) &&
          is_type<int32_t>(num_tokens_post_padded.dtype()),
      "routing tensors must be int32");
  RuntimeCheck(X.stride(1) == 1 && X.stride(0) == X.size(1), "X must be contiguous");
  RuntimeCheck(Y.stride(1) == 1 && Y.stride(0) == Y.size(1), "Y must be contiguous");
  RuntimeCheck(W.stride(2) == 1 && W.stride(1) == W.size(2), "W rows must be contiguous");
  RuntimeCheck(sorted_token_ids.stride(0) == 1 && expert_ids.stride(0) == 1, "routing tensors must be contiguous");
  const int64_t col = X.size(1);
  const int align = k_alignment(type);
  RuntimeCheck(align > 0, "IQ MoE MMQ: unsupported ggml type ", type);
  RuntimeCheck(col % align == 0, "IQ MoE MMQ type ", type, " requires an input size divisible by ", align, ", got ", col);
  RuntimeCheck(X.size(0) == tokens, "X rows (", X.size(0), ") != tokens (", tokens, ")");
  RuntimeCheck(W.size(1) == row, "W rows (", W.size(1), ") != row (", row, ")");
  RuntimeCheck(Y.size(0) == tokens * top_k && Y.size(1) == row, "Y must be [tokens * top_k, row]");
  RuntimeCheck(top_k >= 1 && top_k < (1 << 20), "bad top_k");
  const int64_t padded = padded_k(col);
  RuntimeCheck(quant_X.size(0) >= tokens && quant_X.size(1) >= padded / 32 * 9, "quant_X scratch too small");
  RuntimeCheck(tokens < (1LL << 30) && row < (1LL << 30) && padded < (1LL << 30), "dimensions exceed int32");
  RuntimeCheck(W.size(0) < (1LL << 30), "too many experts");
  // The MoE tile width is 4 (MOE_X_IQ*) on CUDA; the Python caller aligns with block size 4 and the grid.y limit is 65535.
  static_assert(MOE_X_IQ4_NL == 4 && MOE_X_IQ3_S == 4 && MOE_X_IQ4_XS == 4, "Python side assumes moe block size 4");
  const int64_t tokens_post_padded = sorted_token_ids.size(0);
  RuntimeCheck(tokens_post_padded % 4 == 0, "sorted_token_ids length must be a multiple of the MoE block size 4");
  RuntimeCheck(tokens_post_padded / 4 <= 65535, "MoE MMQ grid.y limit: ", tokens_post_padded / 4, " > 65535 blocks");
  RuntimeCheck(expert_ids.size(0) >= tokens_post_padded / 4, "expert_ids shorter than the number of blocks");
  if (tokens == 0) {
    return;
  }
  const cudaStream_t stream = LaunchKernel::resolve_device(X.device());
  const auto launch = [&](auto tag) {
    using scalar_t = decltype(tag);
    quantize_row_q8_1_cuda<scalar_t>(
        static_cast<const scalar_t*>(X.data_ptr()), quant_X.data_ptr(), static_cast<int>(col), static_cast<int>(tokens), stream);
    moe_dispatch<scalar_t>(
        type,
        quant_X.data_ptr(),
        W.data_ptr(),
        static_cast<scalar_t*>(Y.data_ptr()),
        static_cast<const int*>(sorted_token_ids.data_ptr()),
        static_cast<const int*>(expert_ids.data_ptr()),
        static_cast<const int*>(num_tokens_post_padded.data_ptr()),
        W.stride(0),
        static_cast<int>(W.size(0)),
        static_cast<int>(col),
        static_cast<int>(row),
        static_cast<int>(tokens),
        static_cast<int>(padded),
        static_cast<int>(top_k),
        static_cast<int>(tokens_post_padded),
        stream);
  };
  if (is_type<fp16_t>(X.dtype())) {
    launch(fp16_t{});
  } else {
    launch(bf16_t{});
  }
  RuntimeDeviceCheck(cudaGetLastError());
}

}  // namespace gguf_iq_mmq
