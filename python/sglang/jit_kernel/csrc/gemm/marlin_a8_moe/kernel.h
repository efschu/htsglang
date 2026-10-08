// H88-A-PROVENANCE-BEGIN
// Vendored device code (H88-A, 2026-10-07). Source: vLLM, local fork
// /spinning/shvllm, csrc/libtorch_stable/moe/marlin_moe_wna16/kernel.h (HEAD 25faf817d5).
// The int8-activation (W4A8) path is vLLM PR #24722 (jinzhen-lin).
// Original: Marlin, Copyright (C) 2024 Elias Frantar (IST-DASLab,
// https://github.com/IST-DASLab/marlin), modified by Neural Magic and the vLLM
// project. Licensed under the Apache License, Version 2.0
// (http://www.apache.org/licenses/LICENSE-2.0).
// Edits against the original are mechanical only: `vllm::` -> `host::`, the
// scalar-type include (core/scalar_type.hpp -> sgl_kernel/scalar_type.hpp),
// the default namespace (marlin -> marlin_a8, marlin_moe_wna16 ->
// marlin_a8_moe), shared-header paths, header-guard names. The A16 Marlin in
// ../marlin/ and ../marlin_moe/ is a different, untouched code base.
// H88-A-PROVENANCE-END

#ifndef MARLIN_NAMESPACE_NAME
  #define MARLIN_NAMESPACE_NAME marlin_a8_moe
#endif

#include "../marlin_a8/marlin.cuh"
#include "../marlin_a8/marlin_dtypes.cuh"
#include <sgl_kernel/scalar_type.hpp>

#define MARLIN_KERNEL_PARAMS                                          \
  const int4 *__restrict__ A, const int4 *__restrict__ B,             \
      int4 *__restrict__ C, int4 *__restrict__ C_tmp,                 \
      const int4 *__restrict__ b_bias_ptr,                            \
      const float *__restrict__ a_scales_ptr,                         \
      const int4 *__restrict__ scales_ptr,                            \
      const float *__restrict__ global_scale_ptr,                     \
      const int4 *__restrict__ zp_ptr, const int *__restrict__ g_idx, \
      const int32_t *__restrict__ sorted_token_ids_ptr,               \
      const int32_t *__restrict__ expert_ids_ptr,                     \
      const int32_t *__restrict__ num_tokens_past_padded_ptr,         \
      const float *__restrict__ topk_weights_ptr, int top_k,          \
      bool mul_topk_weights, int num_groups, int prob_m, int prob_n,  \
      int prob_k, int *locks, bool has_bias, bool use_atomic_add,     \
      bool use_fp32_reduce

namespace MARLIN_NAMESPACE_NAME {
template <const host::ScalarTypeId a_type_id,  // A ScalarType id
          const host::ScalarTypeId b_type_id,  // B ScalarType id
          const host::ScalarTypeId c_type_id,  // C ScalarType id
          const host::ScalarTypeId s_type_id,  // B_SCALE ScalarType id
          const int threads,          // number of threads in a threadblock
          const int thread_m_blocks,  // number of 16x16 blocks in the m
                                      // dimension (batchsize) of the
                                      // threadblock
          const int thread_n_blocks,  // same for n dimension (output)
          const int thread_k_blocks,  // same for k dimension (reduction)
          const bool m_block_size_8,  // whether m_block_size == 8
                                      // only works when thread_m_blocks == 1
          const int stages,  // number of stages for the async global->shared
                             // fetch pipeline
          const int group_blocks,  // number of consecutive 16x16 blocks
                                   // with a separate quantization scale
          const bool is_zp_float   // is zero point of float16 type?
          >
__global__ void Marlin(MARLIN_KERNEL_PARAMS);

}
