/*
 * Marlin W4A8 (int4 weights, int8 activations) MoE GEMM -- JIT host side.
 *
 * Provenance / licence (H88-A, 2026-10-07): the device code (kernel.h,
 * marlin_template.h in this directory, plus the shared marlin.cuh,
 * marlin_dtypes.cuh, dequant.h, marlin_mma.h in ../marlin_a8/) is vendored from
 * vLLM (local fork /spinning/shvllm, csrc/libtorch_stable/moe/marlin_moe_wna16/
 * and csrc/libtorch_stable/quantization/marlin/, HEAD 25faf817d5; the 8-bit
 * activation path is vLLM PR #24722, jinzhen-lin). Mechanical edits only:
 * `vllm::` -> `host::`, scalar-type include, default namespace
 * (marlin_moe_wna16 -> marlin_a8_moe), shared-header paths.
 * Marlin: Copyright (C) Marlin.2024 Elias Frantar (IST-DASLab,
 * https://github.com/IST-DASLab/marlin), modified by Neural Magic and the vLLM
 * project; Apache License, Version 2.0
 * (http://www.apache.org/licenses/LICENSE-2.0).
 *
 * This host file is the tvm-ffi counterpart of vLLM's marlin_moe_wna16/ops.cu.
 * It instantiates ONLY a_type = kS8 with b_type in {kU4B8 (sym), kU4 (asym+zp)},
 * group_blocks in {-1, 2, 4, 8} (channelwise / g32 / g64 / g128), thread-m
 * blocks 1..4 (moe_block_size 16..64; moe_block_size 8 has no int8 kernel),
 * no act-order, for ONE output dtype per module.
 *
 * Differences against the A16 module ../marlin_moe/moe_wna16_marlin.cuh that the
 * caller must know:
 *   - expert_ids must NOT contain -1 (vLLM semantics: moe_align_block_size with
 *     ignore_invalid_experts=True drops the blocks of experts that are not
 *     local; there is no is_ep switch in this kernel).
 *   - a is int8 [size_m, size_k] (top_k rows are expanded in-kernel through
 *     sorted_token_ids exactly as for A16) and a_scales is float32 [size_m].
 *   - The A16 local patches (SGLANG_MARLIN_NO_K_SPLIT, sms override, arch
 *     override of task #49) are NOT part of this module.
 */

#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/scalar_type.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "kernel.h"
#include "marlin_template.h"

namespace marlin_a8_moe {

__global__ void MarlinDefault(MARLIN_KERNEL_PARAMS){};

using MarlinFuncPtr = void (*)(MARLIN_KERNEL_PARAMS);

typedef struct {
  int thread_k;
  int thread_n;
  int num_threads;
} thread_config_t;

static thread_config_t small_batch_thread_configs[] = {
    // Ordered by priority
    // thread_k, thread_n, num_threads
    {128, 128, 256},
    {64, 128, 128},
    {128, 64, 128}};

static thread_config_t large_batch_thread_configs[] = {
    // Ordered by priority
    // thread_k, thread_n, num_threads
    {64, 256, 256},
    {64, 128, 128},
    {128, 64, 128}};

typedef struct {
  int blocks_per_sm;
  thread_config_t tb_cfg;
} exec_config_t;

inline int get_scales_cache_size(thread_config_t const& th_config, int group_size, int stages) {
  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;
  int tb_groups = group_size == -1 ? 1 : div_ceil(tb_k, group_size);
  int tb_scales = tb_groups * tb_n * 2;
  return tb_scales * stages;
}

// vLLM ops.cu get_kernel_cache_size with the terms that cannot occur for the
// instantiated kernels (act-order, float zero point) removed; is_a_8bit == true.
inline int get_kernel_cache_size(
    thread_config_t const& th_config, int thread_m_blocks, int num_bits, int group_size, int has_zp, int stages) {
  constexpr bool is_a_8bit = true;
  int pack_factor = 32 / num_bits;

  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;

  // shm size for block_sorted_ids/rd_block_sorted_ids/block_topk_weights
  // both of them requires tb_m * 4 bytes (tb_m * int32 or tb_m * float32)
  int sh_block_meta_size = tb_m * 16;
  int sh_a_size = stages * (tb_m * tb_k) * (is_a_8bit ? 1 : 2);
  int sh_b_size = stages * (tb_k * tb_n / pack_factor) * 4;
  int sh_red_size = tb_m * (tb_n + 8) * 2;
  int sh_bias_size = tb_n * 2;
  int tmp_size = (sh_b_size > sh_red_size ? sh_red_size : sh_b_size) + sh_bias_size;
  tmp_size = max(max(sh_b_size, sh_red_size), tmp_size);

  int sh_s_size = get_scales_cache_size(th_config, group_size, stages);
  int sh_zp_size = 0;
  if (has_zp) {
    if (num_bits == 4)
      sh_zp_size = sh_s_size / 4;
    else if (num_bits == 8)
      sh_zp_size = sh_s_size / 2;
  }

  return tmp_size + sh_a_size + sh_s_size + sh_zp_size + sh_block_meta_size;
}

inline bool is_valid_config(
    thread_config_t const& th_config,
    int thread_m_blocks,
    int prob_n,
    int prob_k,
    int num_bits,
    int group_size,
    int has_zp,
    int stages,
    int max_shared_mem) {
  if (th_config.thread_k == -1 || th_config.thread_n == -1 || th_config.num_threads == -1) {
    return false;
  }
  if (prob_k % th_config.thread_k != 0 || prob_n % th_config.thread_n != 0) {
    return false;
  }
  if (th_config.thread_n < min_thread_n || th_config.thread_k < min_thread_k) {
    return false;
  }
  if (th_config.num_threads < 128) {
    return false;
  }
  int cache_size = get_kernel_cache_size(th_config, thread_m_blocks, num_bits, group_size, has_zp, stages);
  return cache_size <= max_shared_mem;
}

#define MARLIN_A8_MOE_GET_IF(B_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS, NUM_THREADS) \
  else if (                                                                                                        \
      b_type == B_TYPE && thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS &&              \
      thread_k_blocks == THREAD_K_BLOCKS && group_blocks == GROUP_BLOCKS && threads == NUM_THREADS) {              \
    kernel = Marlin<                                                                                               \
        host::kS8.id(),                                                                                            \
        B_TYPE.id(),                                                                                               \
        c_type_id,                                                                                                 \
        c_type_id,                                                                                                 \
        NUM_THREADS,                                                                                               \
        THREAD_M_BLOCKS,                                                                                           \
        THREAD_N_BLOCKS,                                                                                           \
        THREAD_K_BLOCKS,                                                                                           \
        false,                                                                                                     \
        4,                                                                                                         \
        GROUP_BLOCKS,                                                                                              \
        false>;                                                                                                    \
  }

#define MARLIN_A8_MOE_GET_IF_GROUPS(B_TYPE, M, N, K, T) \
  MARLIN_A8_MOE_GET_IF(B_TYPE, M, N, K, -1, T)          \
  MARLIN_A8_MOE_GET_IF(B_TYPE, M, N, K, 2, T)           \
  MARLIN_A8_MOE_GET_IF(B_TYPE, M, N, K, 4, T)           \
  MARLIN_A8_MOE_GET_IF(B_TYPE, M, N, K, 8, T)

#define MARLIN_A8_MOE_GET_IF_M1(B_TYPE)             \
  MARLIN_A8_MOE_GET_IF_GROUPS(B_TYPE, 1, 8, 8, 256) \
  MARLIN_A8_MOE_GET_IF_GROUPS(B_TYPE, 1, 8, 4, 128) \
  MARLIN_A8_MOE_GET_IF_GROUPS(B_TYPE, 1, 4, 8, 128)

#define MARLIN_A8_MOE_GET_IF_MBIG(B_TYPE, M)         \
  MARLIN_A8_MOE_GET_IF_GROUPS(B_TYPE, M, 16, 4, 256) \
  MARLIN_A8_MOE_GET_IF_GROUPS(B_TYPE, M, 8, 4, 128)  \
  MARLIN_A8_MOE_GET_IF_GROUPS(B_TYPE, M, 4, 8, 128)

#define MARLIN_A8_MOE_GET_IF_ALL(B_TYPE) \
  MARLIN_A8_MOE_GET_IF_M1(B_TYPE)        \
  MARLIN_A8_MOE_GET_IF_MBIG(B_TYPE, 2)   \
  MARLIN_A8_MOE_GET_IF_MBIG(B_TYPE, 3)   \
  MARLIN_A8_MOE_GET_IF_MBIG(B_TYPE, 4)

template <host::ScalarTypeId c_type_id>
MarlinFuncPtr get_marlin_kernel(
    const host::ScalarType b_type,
    int thread_m_blocks,
    int thread_n_blocks,
    int thread_k_blocks,
    int group_blocks,
    int threads) {
  auto kernel = MarlinDefault;
  if (false) {
  }
  MARLIN_A8_MOE_GET_IF_ALL(host::kU4B8)
  MARLIN_A8_MOE_GET_IF_ALL(host::kU4)
  return kernel;
}

template <host::ScalarTypeId c_type_id>
exec_config_t determine_exec_config(
    const host::ScalarType& b_type,
    int prob_m,
    int prob_n,
    int prob_k,
    int top_k,
    int thread_m_blocks,
    int num_bits,
    int group_size,
    bool has_zp,
    int stages,
    int max_shared_mem,
    int sms) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1 ? large_batch_thread_configs : small_batch_thread_configs;
  int thread_configs_size = thread_m_blocks > 1 ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
                                                : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  int count = 0;
  constexpr int device_max_reg_size = 255 * 1024;
  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(
            th_config, thread_m_blocks, prob_n, prob_k, num_bits, group_size, has_zp, stages, max_shared_mem - 512)) {
      continue;
    }

    int cache_size = get_kernel_cache_size(th_config, thread_m_blocks, num_bits, group_size, has_zp, stages);

    int group_blocks = group_size == -1 ? -1 : (group_size / 16);

    auto kernel = get_marlin_kernel<c_type_id>(
        b_type, thread_m_blocks, th_config.thread_n / 16, th_config.thread_k / 16, group_blocks, th_config.num_threads);

    if (kernel == MarlinDefault) continue;

    cudaFuncAttributes attr;
    host::RuntimeDeviceCheck(cudaFuncGetAttributes(&attr, kernel));
    int reg_size = max(attr.numRegs, 1) * th_config.num_threads * 4;
    int allow_count = min(device_max_reg_size / reg_size, max_shared_mem / (cache_size + 1536));
    if (thread_m_blocks == 1)
      allow_count = max(min(allow_count, 4), 1);
    else
      allow_count = max(min(allow_count, 2), 1);

    if (prob_n / th_config.thread_n * prob_m * top_k * 4 < sms * allow_count) {
      allow_count = max(prob_n / th_config.thread_n * prob_m * top_k * 4 / sms, 1);
    }

    if (allow_count > count) {
      count = allow_count;
      exec_cfg = {count, th_config};
    };
  }

  return exec_cfg;
}

template <host::ScalarTypeId c_type_id>
void marlin_mm(
    const void* A,
    const void* B,
    void* C,
    void* C_tmp,
    void* b_bias,
    void* a_s,
    void* b_s,
    void* zp,
    void* sorted_token_ids,
    void* expert_ids,
    void* num_tokens_past_padded,
    void* topk_weights,
    int moe_block_size,
    int top_k,
    bool mul_topk_weights,
    int prob_m,
    int prob_n,
    int prob_k,
    void* workspace,
    host::ScalarType const& b_type,
    bool has_bias,
    bool has_zp,
    int num_groups,
    int group_size,
    int dev,
    cudaStream_t stream,
    int sms,
    bool use_atomic_add,
    bool use_fp32_reduce) {
  using host::RuntimeCheck;
  int thread_m_blocks = div_ceil(moe_block_size, 16);

  RuntimeCheck(prob_m > 0 && prob_n > 0 && prob_k > 0, "Invalid MNK = [", prob_m, ", ", prob_n, ", ", prob_k, "]");

  int group_blocks;
  if (group_size == -1) {
    group_blocks = -1;
  } else {
    group_blocks = group_size / 16;
    RuntimeCheck(
        prob_k % group_blocks == 0, "prob_k = ", prob_k, " is not divisible by group_blocks = ", group_blocks);
  }

  int num_bits = b_type.size_bits();
  const int4* A_ptr = (const int4*)A;
  const int4* B_ptr = (const int4*)B;
  int4* C_ptr = (int4*)C;
  int4* C_tmp_ptr = (int4*)C_tmp;
  const int4* bias_ptr = (const int4*)b_bias;
  const float* a_s_ptr = (const float*)a_s;
  const int4* b_s_ptr = (const int4*)b_s;
  const float* g_s_ptr = nullptr;  // global scale: nvfp4 only
  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = nullptr;  // act-order is not instantiated for a8
  const int32_t* sorted_token_ids_ptr = (const int32_t*)sorted_token_ids;
  const int32_t* expert_ids_ptr = (const int32_t*)expert_ids;
  const int32_t* num_tokens_past_padded_ptr = (const int32_t*)num_tokens_past_padded;
  const float* topk_weights_ptr = (const float*)topk_weights;
  int* locks = (int*)workspace;

  int max_shared_mem = 0;
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  RuntimeCheck(max_shared_mem > 0);

  int major_capability, minor_capability;
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor, dev));
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor, dev));
  RuntimeCheck(
      major_capability * 10 + minor_capability >= 80, "marlin w4a8 (int8 activation) needs compute capability >= 8.0");
  const int stages = 4;

  exec_config_t exec_cfg = determine_exec_config<c_type_id>(
      b_type, prob_m, prob_n, prob_k, top_k, thread_m_blocks, num_bits, group_size, has_zp, stages, max_shared_mem, sms);
  thread_config_t thread_tfg = exec_cfg.tb_cfg;

  int num_threads = thread_tfg.num_threads;
  int thread_k = thread_tfg.thread_k;
  int thread_n = thread_tfg.thread_n;
  int blocks = sms * exec_cfg.blocks_per_sm;
  if (exec_cfg.blocks_per_sm > 1) max_shared_mem = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;

  RuntimeCheck(
      is_valid_config(thread_tfg, thread_m_blocks, prob_n, prob_k, num_bits, group_size, has_zp, stages, max_shared_mem),
      "Invalid thread config: thread_m_blocks = ",
      thread_m_blocks,
      ", thread_k = ",
      thread_tfg.thread_k,
      ", thread_n = ",
      thread_tfg.thread_n,
      ", num_threads = ",
      thread_tfg.num_threads,
      " for MKN = [",
      prob_m,
      ", ",
      prob_k,
      ", ",
      prob_n,
      "] and num_bits = ",
      num_bits,
      ", group_size = ",
      group_size,
      ", has_zp = ",
      has_zp,
      ", max_shared_mem = ",
      max_shared_mem);

  auto kernel = get_marlin_kernel<c_type_id>(
      b_type, thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks, num_threads);

  if (kernel == MarlinDefault) {
    host::Panic(
        "Unsupported shapes for marlin w4a8 moe: MNK = [",
        prob_m,
        ", ",
        prob_n,
        ", ",
        prob_k,
        "]",
        ", num_groups = ",
        num_groups,
        ", group_size = ",
        group_size,
        ", thread_m_blocks = ",
        thread_m_blocks,
        ", thread_n_blocks = ",
        thread_n_blocks,
        ", thread_k_blocks = ",
        thread_k_blocks,
        ", num_bits = ",
        num_bits);
  }

  host::RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem));

  host::LaunchKernel(blocks, num_threads, stream, max_shared_mem)(
      kernel,
      A_ptr,
      B_ptr,
      C_ptr,
      C_tmp_ptr,
      bias_ptr,
      a_s_ptr,
      b_s_ptr,
      g_s_ptr,
      zp_ptr,
      g_idx_ptr,
      sorted_token_ids_ptr,
      expert_ids_ptr,
      num_tokens_past_padded_ptr,
      topk_weights_ptr,
      top_k,
      mul_topk_weights,
      num_groups,
      prob_m,
      prob_n,
      prob_k,
      locks,
      has_bias,
      use_atomic_add,
      use_fp32_reduce);
}

}  // namespace marlin_a8_moe

// c_scalar_t: fp16_t or bf16_t -- dtype of C, of b_scales and of the bias.
template <typename c_scalar_t>
void moe_wna16_marlin_gemm_a8(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView a_scales,
    tvm::ffi::TensorView c,
    tvm::ffi::TensorView b_q_weight,
    tvm::ffi::TensorView b_bias,
    tvm::ffi::TensorView b_scales,
    tvm::ffi::TensorView b_zeros,
    tvm::ffi::TensorView workspace,
    tvm::ffi::TensorView sorted_token_ids,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_tokens_post_padded,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView c_tmp,
    int64_t moe_block_size,
    int64_t top_k,
    bool mul_topk_weights,
    int64_t b_q_type_id,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    bool use_atomic_add,
    bool use_fp32_reduce) {
  using namespace host;

  static_assert(
      std::is_same<c_scalar_t, fp16_t>::value || std::is_same<c_scalar_t, bf16_t>::value,
      "marlin w4a8 moe output dtype must be float16 or bfloat16");
  constexpr host::ScalarTypeId c_type_id =
      std::is_same<c_scalar_t, fp16_t>::value ? host::kFloat16.id() : host::kBFloat16.id();

  ScalarType const b_q_type = ScalarType::from_id(b_q_type_id);
  RuntimeCheck(b_q_type.size_bits() == 4, "marlin w4a8 moe: only 4-bit weights are instantiated. Got = ", b_q_type.str());
  int pack_factor = 32 / b_q_type.size_bits();

  // int8 activation kernels exist for thread_m_blocks 1..4 only (no m_block_size_8)
  RuntimeCheck(
      moe_block_size == 16 || moe_block_size == 32 || moe_block_size == 48 || moe_block_size == 64,
      "marlin w4a8 moe: moe_block_size must be 16, 32, 48 or 64. Got = ",
      moe_block_size);

  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  // a: int8 [size_m, size_k], contiguous
  RuntimeCheck(a.dim() == 2, "a rank = ", a.dim(), " is not 2");
  RuntimeCheck(a.size(0) == size_m, "Shape mismatch: a.size(0) = ", a.size(0), ", size_m = ", size_m);
  RuntimeCheck(a.size(1) == size_k, "Shape mismatch: a.size(1) = ", a.size(1), ", size_k = ", size_k);
  TensorMatcher({-1, -1}).with_dtype<int8_t>().with_device(device).verify(a);
  RuntimeCheck(a.is_contiguous(), "A is not contiguous");
  RuntimeCheck(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0, "a must be aligned to 16 bytes");

  auto am = SymbolicSize{"am"};
  am.set_value(size_m);
  TensorMatcher({am}).with_dtype<float>().with_device(device).verify(a_scales);

  // b_q_weight: int32 [E, K/16, N*16/pack_factor]
  RuntimeCheck(b_q_weight.dim() == 3, "b_q_weight rank = ", b_q_weight.dim(), " is not 3");
  RuntimeCheck(
      size_k % marlin_a8_moe::tile_size == 0,
      "size_k = ",
      size_k,
      " is not divisible by tile_size = ",
      marlin_a8_moe::tile_size);
  RuntimeCheck(
      (size_k / marlin_a8_moe::tile_size) == b_q_weight.size(1),
      "Shape mismatch: b_q_weight.size(1) = ",
      b_q_weight.size(1),
      ", size_k = ",
      size_k,
      ", tile_size = ",
      marlin_a8_moe::tile_size);
  RuntimeCheck(
      b_q_weight.size(2) % marlin_a8_moe::tile_size == 0,
      "b_q_weight.size(2) = ",
      b_q_weight.size(2),
      " is not divisible by tile_size = ",
      marlin_a8_moe::tile_size);
  int64_t actual_size_n = (b_q_weight.size(2) / marlin_a8_moe::tile_size) * pack_factor;
  RuntimeCheck(size_n == actual_size_n, "size_n = ", size_n, ", actual_size_n = ", actual_size_n);
  TensorMatcher({-1, -1, -1}).with_dtype<int32_t>().with_device(device).verify(b_q_weight);
  RuntimeCheck(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  // b_scales: c dtype [E, G, N]
  TensorMatcher({-1, -1, -1}).with_dtype<c_scalar_t>().with_device(device).verify(b_scales);
  RuntimeCheck(b_scales.is_contiguous(), "b_scales is not contiguous");
  RuntimeCheck(b_scales.size(2) == size_n, "b_scales dim 2 = ", b_scales.size(2), " is not size_n = ", size_n);
  RuntimeCheck(b_scales.size(0) == b_q_weight.size(0), "b_scales dim 0 != number of experts");
  int num_groups = static_cast<int>(b_scales.size(1));

  // c: c dtype [size_m * top_k, size_n]
  TensorMatcher({-1, -1}).with_dtype<c_scalar_t>().with_device(device).verify(c);
  RuntimeCheck(c.is_contiguous(), "c is not contiguous");
  RuntimeCheck(
      c.size(0) == size_m * top_k, "Shape mismatch: c.size(0) = ", c.size(0), ", size_m * topk = ", size_m * top_k);
  RuntimeCheck(c.size(1) == size_n, "Shape mismatch: c.size(1) = ", c.size(1), ", size_n = ", size_n);

  int group_size = -1;
  if (num_groups > 1) {
    RuntimeCheck(size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by b_scales.size(1) = ", num_groups);
    group_size = static_cast<int>(size_k / num_groups);
    RuntimeCheck(
        group_size == 32 || group_size == 64 || group_size == 128,
        "marlin w4a8 moe: group_size must be -1 (channelwise), 32, 64 or 128. Got = ",
        group_size);
  }

  bool has_zp = b_zeros.size(0) > 0;
  if (has_zp) {
    RuntimeCheck(b_q_type == kU4, "b_q_type must be u4 when b_zeros is given (has_zp = True). Got = ", b_q_type.str());
    RuntimeCheck(b_zeros.dim() == 3, "b_zeros rank = ", b_zeros.dim(), " is not 3");
    RuntimeCheck(b_zeros.size(1) == num_groups, "b_zeros dim 1 = ", b_zeros.size(1), " is not num_groups = ", num_groups);
    RuntimeCheck(
        b_zeros.size(2) == size_n / pack_factor,
        "b_zeros dim 2 = ",
        b_zeros.size(2),
        " is not size_n / pack_factor = ",
        size_n / pack_factor);
    RuntimeCheck(b_zeros.is_contiguous(), "b_zeros is not contiguous");
    device.verify(b_zeros.device());
  } else {
    RuntimeCheck(
        b_q_type == kU4B8, "b_q_type must be uint4b8 when b_zeros is empty (has_zp = False). Got = ", b_q_type.str());
  }

  bool has_bias = b_bias.size(0) > 0;
  if (has_bias) {
    RuntimeCheck(b_bias.dim() == 2, "b_bias rank = ", b_bias.dim(), " is not 2");
    RuntimeCheck(b_bias.size(1) == size_n, "b_bias.size(1) != size_n");
    RuntimeCheck(b_bias.stride(1) == 1, "b_bias.stride(1) != 1");
    RuntimeCheck(b_bias.is_contiguous(), "b_bias is not contiguous");
    device.verify(b_bias.device());
  }

  RuntimeCheck(
      size_n % marlin_a8_moe::min_thread_n == 0,
      "size_n = ",
      size_n,
      ", is not divisible by min_thread_n = ",
      marlin_a8_moe::min_thread_n);

  DLDevice dl_device = device.unwrap();
  int dev = dl_device.device_id;
  cudaStream_t stream = LaunchKernel::resolve_device(dl_device);
  int sms = -1;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));

  int64_t max_n_tiles = size_n / marlin_a8_moe::min_thread_n;
  int64_t min_workspace_size =
      std::min(max_n_tiles * (sorted_token_ids.size(0) / moe_block_size), static_cast<int64_t>(sms) * 4);
  RuntimeCheck(
      workspace.size(0) >= min_workspace_size,
      "workspace.numel = ",
      workspace.size(0),
      " is below min_workspace_size = ",
      min_workspace_size);

  if (size_m == 0) return;

  marlin_a8_moe::marlin_mm<c_type_id>(
      a.data_ptr(),
      b_q_weight.data_ptr(),
      c.data_ptr(),
      c_tmp.data_ptr(),
      b_bias.data_ptr(),
      a_scales.data_ptr(),
      b_scales.data_ptr(),
      b_zeros.data_ptr(),
      sorted_token_ids.data_ptr(),
      expert_ids.data_ptr(),
      num_tokens_post_padded.data_ptr(),
      topk_weights.data_ptr(),
      static_cast<int>(moe_block_size),
      static_cast<int>(top_k),
      mul_topk_weights,
      static_cast<int>(size_m),
      static_cast<int>(size_n),
      static_cast<int>(size_k),
      workspace.data_ptr(),
      b_q_type,
      has_bias,
      has_zp,
      num_groups,
      group_size,
      dev,
      stream,
      sms,
      use_atomic_add,
      use_fp32_reduce);
}
