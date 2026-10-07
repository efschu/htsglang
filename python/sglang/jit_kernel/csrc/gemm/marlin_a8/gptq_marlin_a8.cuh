/*
 * Marlin W4A8 (int4 weights, int8 activations) dense GEMM -- JIT host side.
 *
 * Provenance / licence (H88-A, 2026-10-07):
 *   The device code in this directory (marlin.cuh, marlin_dtypes.cuh,
 *   dequant.h, marlin_mma.h, marlin_template.h, kernel.h) is a vendored copy of
 *   the vLLM Marlin kernel with the 8-bit-activation path from vLLM PR #24722
 *   (jinzhen-lin, "[Kernel][Quantization] add w4a8 support for marlin kernel"),
 *   taken from the local vLLM fork /spinning/shvllm at
 *   csrc/libtorch_stable/quantization/marlin/ (HEAD 25faf817d5). The only edits
 *   against that source are mechanical: `vllm::` -> `host::`, the scalar-type
 *   include (core/scalar_type.hpp -> sgl_kernel/scalar_type.hpp), the default
 *   namespace (marlin -> marlin_a8, marlin_moe_wna16 -> marlin_a8_moe) and the
 *   header guards.
 *   Original: Copyright (C) Marlin.2024 Elias Frantar (IST-DASLab,
 *   https://github.com/IST-DASLab/marlin), modified by Neural Magic and the vLLM
 *   project. Licensed under the Apache License, Version 2.0
 *   (http://www.apache.org/licenses/LICENSE-2.0).
 *
 *   This host file is the tvm-ffi JIT counterpart of vLLM's marlin.cu. It
 *   replaces the generated kernel_selector.h by a macro table that
 *   instantiates ONLY the int8-activation kernels (a_type = kS8; b_type = kU4B8
 *   [GPTQ/sym] or kU4 [AWQ/asym+zp]; group_blocks in {-1, 2, 4, 8}, i.e. group
 *   sizes channelwise/32/64/128; no act-order) for ONE output dtype per module.
 *
 *   The A16 Marlin in ../marlin/ is NOT touched and shares no symbol with this
 *   file (separate namespace marlin_a8, separate JIT module).
 *
 * Contract (see jit_kernel/gptq_marlin_w4a8.py):
 *   a          int8    [M, K]   per-token quantised activations, lda % 16 == 0
 *   a_scales   float32 [M]      per-token scale (already multiplied by the
 *                               global factor from marlin_act_int8_process_scales
 *                               when group scales are int16-encoded)
 *   b_q_weight int32   [K/16, N*16/8]  Marlin W4A8 layout (is_a_8bit repack)
 *   b_scales   fp16/bf16 [G, N]  Marlin-permuted; for group sizes != -1 the
 *                               values are int16 x4096-encoded (bit pattern
 *                               viewed as fp16/bf16)
 *   b_zeros    int32   [G, N/8] (asym only) or empty
 *   bias       fp16/bf16 [N] or empty
 */

#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/scalar_type.hpp>

#include "kernel.h"
#include "marlin_template.h"

namespace marlin_a8 {

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

// The a8 kernels are never act-order (group_blocks == 0 is not instantiated),
// so the act-order terms of the vLLM cache-size formulas drop out; everything
// else is the vLLM formula unchanged (marlin.cu get_kernel_cache_size).
inline int get_scales_cache_size(thread_config_t const& th_config, int group_size, int stages) {
  int tb_n = th_config.thread_n;
  int tb_k = th_config.thread_k;

  int tb_groups;
  if (group_size == -1) {
    tb_groups = 1;
  } else {
    tb_groups = div_ceil(tb_k, group_size);
  }

  int tb_scales = tb_groups * tb_n * 2;
  return tb_scales * stages;
}

inline int get_kernel_cache_size(
    thread_config_t const& th_config, int thread_m_blocks, int num_bits, int group_size, int has_zp, int stages) {
  constexpr bool is_a_8bit = true;
  int pack_factor = 32 / num_bits;

  int tb_k = th_config.thread_k;
  int tb_n = th_config.thread_n;
  int tb_m = thread_m_blocks * 16;
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

  return tmp_size + sh_a_size + sh_s_size + sh_zp_size;
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

// ---------------------------------------------------------------------------
// Kernel table. One row per instantiated kernel:
//   a = kS8, b in {kU4B8, kU4}, c = s = module output dtype, stages = 4,
//   m_block_size_8 = false, is_zp_float = false.
// Thread configs follow vLLM generate_kernels.py (THREAD_CONFIGS pruned the same
// way: 256 threads only as (128,128) for m == 1 and (64,256) for m > 1).
// ---------------------------------------------------------------------------
#define MARLIN_A8_GET_IF(B_TYPE, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, GROUP_BLOCKS, NUM_THREADS) \
  else if (                                                                                                    \
      b_type == B_TYPE && thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS &&          \
      thread_k_blocks == THREAD_K_BLOCKS && group_blocks == GROUP_BLOCKS && threads == NUM_THREADS) {          \
    kernel = Marlin<                                                                                           \
        host::kS8.id(),                                                                                        \
        B_TYPE.id(),                                                                                           \
        c_type_id,                                                                                             \
        c_type_id,                                                                                             \
        NUM_THREADS,                                                                                           \
        THREAD_M_BLOCKS,                                                                                       \
        THREAD_N_BLOCKS,                                                                                       \
        THREAD_K_BLOCKS,                                                                                       \
        false,                                                                                                 \
        4,                                                                                                     \
        GROUP_BLOCKS,                                                                                          \
        false>;                                                                                                \
  }

#define MARLIN_A8_GET_IF_GROUPS(B_TYPE, M, N, K, T) \
  MARLIN_A8_GET_IF(B_TYPE, M, N, K, -1, T)          \
  MARLIN_A8_GET_IF(B_TYPE, M, N, K, 2, T)           \
  MARLIN_A8_GET_IF(B_TYPE, M, N, K, 4, T)           \
  MARLIN_A8_GET_IF(B_TYPE, M, N, K, 8, T)

#define MARLIN_A8_GET_IF_M1(B_TYPE)             \
  MARLIN_A8_GET_IF_GROUPS(B_TYPE, 1, 8, 8, 256) \
  MARLIN_A8_GET_IF_GROUPS(B_TYPE, 1, 8, 4, 128) \
  MARLIN_A8_GET_IF_GROUPS(B_TYPE, 1, 4, 8, 128)

#define MARLIN_A8_GET_IF_MBIG(B_TYPE, M)         \
  MARLIN_A8_GET_IF_GROUPS(B_TYPE, M, 16, 4, 256) \
  MARLIN_A8_GET_IF_GROUPS(B_TYPE, M, 8, 4, 128)  \
  MARLIN_A8_GET_IF_GROUPS(B_TYPE, M, 4, 8, 128)

#define MARLIN_A8_GET_IF_ALL(B_TYPE) \
  MARLIN_A8_GET_IF_M1(B_TYPE)        \
  MARLIN_A8_GET_IF_MBIG(B_TYPE, 2)   \
  MARLIN_A8_GET_IF_MBIG(B_TYPE, 3)   \
  MARLIN_A8_GET_IF_MBIG(B_TYPE, 4)

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
  MARLIN_A8_GET_IF_ALL(host::kU4B8)
  MARLIN_A8_GET_IF_ALL(host::kU4)
  return kernel;
}

template <host::ScalarTypeId c_type_id>
exec_config_t determine_exec_config(
    const host::ScalarType& b_type,
    int prob_n,
    int prob_k,
    int thread_m_blocks,
    int num_bits,
    int group_size,
    bool has_zp,
    int stages,
    int max_shared_mem) {
  exec_config_t exec_cfg = exec_config_t{1, thread_config_t{-1, -1, -1}};
  thread_config_t* thread_configs = thread_m_blocks > 1 ? large_batch_thread_configs : small_batch_thread_configs;
  int thread_configs_size = thread_m_blocks > 1 ? sizeof(large_batch_thread_configs) / sizeof(thread_config_t)
                                                : sizeof(small_batch_thread_configs) / sizeof(thread_config_t);

  for (int i = 0; i < thread_configs_size; i++) {
    thread_config_t th_config = thread_configs[i];

    if (!is_valid_config(
            th_config, thread_m_blocks, prob_n, prob_k, num_bits, group_size, has_zp, stages, max_shared_mem - 512)) {
      continue;
    }

    int group_blocks = group_size == -1 ? -1 : group_size / 16;

    auto kernel = get_marlin_kernel<c_type_id>(
        b_type, thread_m_blocks, th_config.thread_n / 16, th_config.thread_k / 16, group_blocks, th_config.num_threads);

    if (kernel == MarlinDefault) continue;

    return {1, th_config};
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
    int prob_m,
    int prob_n,
    int prob_k,
    int lda,
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
  const float* g_s_ptr = nullptr;  // global scale: nvfp4 only, never used with int8 A
  const int4* zp_ptr = (const int4*)zp;
  const int* g_idx_ptr = nullptr;  // act-order is not instantiated for a8
  int* locks = (int*)workspace;

  int max_shared_mem = 0;
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
  RuntimeCheck(max_shared_mem > 0);

  int major_capability, minor_capability;
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&major_capability, cudaDevAttrComputeCapabilityMajor, dev));
  host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&minor_capability, cudaDevAttrComputeCapabilityMinor, dev));
  // The int8 kernels are built from sm80 SASS (runs on sm86) and from the
  // sm120 target; vLLM's own floor is sm75, this module is built for >= sm80.
  RuntimeCheck(
      major_capability * 10 + minor_capability >= 80, "marlin w4a8 (int8 activation) needs compute capability >= 8.0");
  const int stages = 4;

  int max_par = 16;
  if (prob_n <= 4096) max_par = 16 * 8;
  int max_shared_mem_new = max_shared_mem;
  int rest_m = prob_m;
  int max_thread_m_blocks = 4;
  while (rest_m) {
    int par_count = rest_m / (max_thread_m_blocks * 16);
    if (par_count > max_par) par_count = max_par;
    int prob_m_split = par_count > 0 ? (par_count * (max_thread_m_blocks * 16)) : rest_m;

    int thread_m_blocks = min(div_ceil(prob_m_split, 16), max_thread_m_blocks);

    exec_config_t exec_cfg = determine_exec_config<c_type_id>(
        b_type, prob_n, prob_k, thread_m_blocks, num_bits, group_size, has_zp, stages, max_shared_mem);
    thread_config_t thread_tfg = exec_cfg.tb_cfg;
    if (thread_tfg.thread_n != -1) {
      if (prob_n / thread_tfg.thread_n * div_ceil(prob_m_split, thread_m_blocks * 16) * 4 <= sms) {
        if (is_valid_config(
                {128, 64, 128},
                thread_m_blocks,
                prob_n,
                prob_k,
                num_bits,
                group_size,
                has_zp,
                stages,
                max_shared_mem_new)) {
          thread_tfg = {128, 64, 128};
          exec_cfg = {1, thread_tfg};
        }
      }
    }

    if (thread_tfg.thread_k == -1 && max_thread_m_blocks > 1) {
      max_thread_m_blocks--;
      continue;
    }

    int num_threads = thread_tfg.num_threads;
    int thread_k = thread_tfg.thread_k;
    int thread_n = thread_tfg.thread_n;
    int blocks = sms * exec_cfg.blocks_per_sm;
    if (exec_cfg.blocks_per_sm > 1) max_shared_mem_new = max_shared_mem / exec_cfg.blocks_per_sm - 1024;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    RuntimeCheck(
        is_valid_config(
            thread_tfg, thread_m_blocks, prob_n, prob_k, num_bits, group_size, has_zp, stages, max_shared_mem_new),
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
        ", prob_m_split = ",
        prob_m_split,
        ", group_size = ",
        group_size,
        ", has_zp = ",
        has_zp,
        ", max_shared_mem_new = ",
        max_shared_mem_new);

    auto kernel = get_marlin_kernel<c_type_id>(
        b_type, thread_m_blocks, thread_n_blocks, thread_k_blocks, group_blocks, num_threads);

    if (kernel == MarlinDefault) {
      host::Panic(
          "Unsupported shapes for marlin w4a8: MNK = [",
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
          ", prob_m_split = ",
          prob_m_split,
          ", thread_m_blocks = ",
          thread_m_blocks,
          ", thread_n_blocks = ",
          thread_n_blocks,
          ", thread_k_blocks = ",
          thread_k_blocks,
          ", num_threads = ",
          num_threads,
          ", num_bits = ",
          num_bits);
    }

    host::RuntimeDeviceCheck(
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem_new));

    bool part_use_atomic_add = use_atomic_add && div_ceil(prob_m_split, 64) * prob_n <= 2048;

    host::LaunchKernel(blocks, num_threads, stream, max_shared_mem_new)(
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
        num_groups,
        prob_m_split,
        prob_n,
        prob_k,
        lda,
        locks,
        has_bias,
        part_use_atomic_add,
        use_fp32_reduce,
        max_shared_mem_new);

    // int8 A: A_ptr is an int4 (16 byte) pointer, lda counts int8 elements.
    A_ptr += prob_m_split * (lda / 16);
    a_s_ptr += prob_m_split;
    C_ptr += prob_m_split * (prob_n / 8);
    rest_m -= prob_m_split;
  }
}

}  // namespace marlin_a8

// c_scalar_t: fp16_t or bf16_t -- the dtype of C, of b_scales and of the bias.
template <typename c_scalar_t>
void gptq_marlin_gemm_a8(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView a_scales,
    tvm::ffi::TensorView b_q_weight,
    tvm::ffi::TensorView b_scales,
    tvm::ffi::TensorView b_zeros,
    tvm::ffi::TensorView b_bias,
    tvm::ffi::TensorView c,
    tvm::ffi::TensorView c_tmp,
    tvm::ffi::TensorView workspace,
    int64_t b_q_type_id,
    bool use_atomic_add,
    bool use_fp32_reduce) {
  using namespace host;

  static_assert(
      std::is_same<c_scalar_t, fp16_t>::value || std::is_same<c_scalar_t, bf16_t>::value,
      "marlin w4a8 output dtype must be float16 or bfloat16");
  constexpr host::ScalarTypeId c_type_id =
      std::is_same<c_scalar_t, fp16_t>::value ? host::kFloat16.id() : host::kBFloat16.id();

  ScalarType const b_q_type = ScalarType::from_id(b_q_type_id);
  RuntimeCheck(b_q_type.size_bits() == 4, "marlin w4a8: only 4-bit weights are instantiated. Got = ", b_q_type.str());
  int pack_factor = 32 / b_q_type.size_bits();

  auto M = SymbolicSize{"M"};
  auto K = SymbolicSize{"K"};
  auto N = SymbolicSize{"N"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();

  // a: int8 [M, K], rows 16-byte aligned (A is read as int4 vectors)
  auto lda = SymbolicSize{"lda"};
  TensorMatcher({M, K}).with_strides({lda, 1}).with_dtype<int8_t>().with_device(device).verify(a);
  int64_t size_m = M.unwrap();
  int64_t size_k = K.unwrap();
  int64_t a_stride0 = a.stride(0);
  RuntimeCheck(a_stride0 % 16 == 0, "a.stride(0) must be divisible by 16 for int8 activations");
  RuntimeCheck(reinterpret_cast<uintptr_t>(a.data_ptr()) % 16 == 0, "a must be aligned to 16 bytes");

  // a_scales: float32 [M]
  auto am = SymbolicSize{"am"};
  am.set_value(size_m);
  TensorMatcher({am}).with_dtype<float>().with_device(device).verify(a_scales);

  // b_q_weight: int32 [K/16, N*16/pack_factor]
  RuntimeCheck(
      size_k % marlin_a8::tile_size == 0, "size_k = ", size_k, " is not divisible by tile_size = ", marlin_a8::tile_size);
  auto bqw0 = SymbolicSize{"bqw0"};
  auto bqw1 = SymbolicSize{"bqw1"};
  bqw0.set_value(size_k / marlin_a8::tile_size);
  TensorMatcher({bqw0, bqw1}).with_dtype<int32_t>().with_device(device).verify(b_q_weight);
  RuntimeCheck(
      b_q_weight.size(1) % marlin_a8::tile_size == 0,
      "b_q_weight.size(1) = ",
      b_q_weight.size(1),
      " is not divisible by tile_size = ",
      marlin_a8::tile_size);
  int64_t size_n = (b_q_weight.size(1) / marlin_a8::tile_size) * pack_factor;
  N.set_value(size_n);
  RuntimeCheck(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");

  // b_scales: c dtype [G, N]
  auto G = SymbolicSize{"G"};
  TensorMatcher({G, N}).with_dtype<c_scalar_t>().with_device(device).verify(b_scales);
  RuntimeCheck(b_scales.is_contiguous(), "b_scales is not contiguous");
  int num_groups = static_cast<int>(G.unwrap());

  // c: c dtype [M, N]
  TensorMatcher({M, N}).with_dtype<c_scalar_t>().with_device(device).verify(c);
  RuntimeCheck(c.is_contiguous(), "c is not contiguous");

  int group_size = -1;
  if (num_groups > 1) {
    RuntimeCheck(size_k % num_groups == 0, "size_k = ", size_k, ", is not divisible by num_groups = ", num_groups);
    group_size = static_cast<int>(size_k / num_groups);
    RuntimeCheck(
        group_size == 32 || group_size == 64 || group_size == 128,
        "marlin w4a8: group_size must be -1 (channelwise), 32, 64 or 128. Got = ",
        group_size);
  }

  // zero points (asym, AWQ-style) -- u4 with zero points, otherwise u4b8 symmetric
  bool has_zp = b_zeros.size(0) > 0;
  if (has_zp) {
    RuntimeCheck(b_q_type == kU4, "b_q_type must be u4 when b_zeros is given (has_zp = True). Got = ", b_q_type.str());
    RuntimeCheck(b_zeros.dim() == 2, "b_zeros rank = ", b_zeros.dim(), " is not 2");
    RuntimeCheck(b_zeros.size(0) == num_groups, "b_zeros dim 0 = ", b_zeros.size(0), " is not num_groups = ", num_groups);
    RuntimeCheck(
        b_zeros.size(1) == size_n / pack_factor,
        "b_zeros dim 1 = ",
        b_zeros.size(1),
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
    RuntimeCheck(b_bias.size(0) == size_n, "b_bias.size(0) != size_n");
    RuntimeCheck(b_bias.is_contiguous(), "b_bias is not contiguous");
    device.verify(b_bias.device());
  }

  RuntimeCheck(
      size_n % marlin_a8::min_thread_n == 0,
      "size_n = ",
      size_n,
      ", is not divisible by min_thread_n = ",
      marlin_a8::min_thread_n);

  if (size_m == 0) return;

  DLDevice dl_device = device.unwrap();
  int dev = dl_device.device_id;
  cudaStream_t stream = LaunchKernel::resolve_device(dl_device);

  int sms = -1;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  RuntimeCheck(
      workspace.size(0) >= sms, "workspace.size(0) = ", workspace.size(0), " is below min_workspace_size = ", sms);

  marlin_a8::marlin_mm<c_type_id>(
      a.data_ptr(),
      b_q_weight.data_ptr(),
      c.data_ptr(),
      c_tmp.data_ptr(),
      b_bias.data_ptr(),
      a_scales.data_ptr(),
      b_scales.data_ptr(),
      b_zeros.data_ptr(),
      static_cast<int>(size_m),
      static_cast<int>(size_n),
      static_cast<int>(size_k),
      static_cast<int>(a_stride0),
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
