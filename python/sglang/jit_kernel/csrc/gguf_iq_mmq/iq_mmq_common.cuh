// GGUF-NF G6 (2026-10-09): IQ-type MMQ / MoE-MMQ kernels for GGUF I-quants, vendored as NEW JIT sources.
// Source: sgl-project/sglang PR #36122 "[CUDA] Add dense and MoE GGUF MMQ kernels for eight I-quant types"
//   (open, unreviewed; head 0a39ec3a8754e3c4426950a5a93046c048a03f07, base 6e7beace143964386ce2e4418401b7369f0dabd0;
//   diff sha256 cf93506bd6b55eca441794007b1d5e92f6a66bf5f0524f288f460f1418bcee38),
//   itself adapted from https://github.com/aimbit-ni/vllm/commit/037e1d547c15313aa7e2bb5fc83390ab0c857314 and
//   ggml-org/llama.cpp PR #8495 (MIT, (c) 2023-2024 The ggml authors).
// Existing sgl-kernel/csrc/quantization/gguf/* is NOT touched (sha256-pinned by test_gguf_iq_mmq_1009.py).
//
// Helpers copied from sgl-kernel/csrc/quantization/gguf/vecdotq.cuh (lines 1-60 of the base), with the PR #36122
// change to get_int_b2 / get_int_from_int8 / get_int_from_uint8 (unsigned shifts: `int(x16) << 16` is signed-overflow UB).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "iq_mmq_ggml_common.h"

static __device__ __forceinline__ int get_int_b2(const void* x, const int& i32) {
  const uint16_t* x16 = (const uint16_t*)x;  // assume at least 2 byte alignment

  uint32_t x32 = uint32_t(x16[2 * i32 + 0]);
  x32 |= uint32_t(x16[2 * i32 + 1]) << 16;

  return static_cast<int>(x32);
}

static __device__ __forceinline__ int get_int_from_int8(const int8_t* x8, const int& i32) {
  const uint16_t* x16 = (const uint16_t*)(x8 + sizeof(int) * i32);  // assume at least 2 byte alignment
  uint32_t x32 = uint32_t(x16[0]);
  x32 |= uint32_t(x16[1]) << 16;
  return static_cast<int>(x32);
}

static __device__ __forceinline__ int get_int_from_uint8(const uint8_t* x8, const int& i32) {
  const uint16_t* x16 = (const uint16_t*)(x8 + sizeof(int) * i32);  // assume at least 2 byte alignment
  uint32_t x32 = uint32_t(x16[0]);
  x32 |= uint32_t(x16[1]) << 16;
  return static_cast<int>(x32);
}

static __device__ __forceinline__ int get_int_from_int8_aligned(const int8_t* x8, const int& i32) {
  return *((const int*)(x8 + sizeof(int) * i32));  // assume at least 4 byte alignment
}

static __device__ __forceinline__ int get_int_from_uint8_aligned(const uint8_t* x8, const int& i32) {
  return *((const int*)(x8 + sizeof(int) * i32));  // assume at least 4 byte alignment
}

// Copied from vecdotq.cuh (base c651892375) with the PR #36122 unsigned-shift change. IQ4_NL / IQ4_XS are NOT affected
// by the nvcc 13.2 byte-mask miscompile (llama.cpp #28581), so the byte extraction through `aux32` stays as in ggml.
static __device__ __forceinline__ void
get_int_from_table_16(const uint32_t& q4, const uint8_t* values, int& val1, int& val2) {
  uint32_t aux32;
  const uint8_t* q8 = (const uint8_t*)&aux32;
  aux32 = q4 & 0x0f0f0f0f;
  uint16_t v1 = values[q8[0]] | (values[q8[1]] << 8);
  uint16_t v2 = values[q8[2]] | (values[q8[3]] << 8);
  val1 = static_cast<int>(uint32_t(v1) | (uint32_t(v2) << 16));
  aux32 = (q4 >> 4) & 0x0f0f0f0f;
  v1 = values[q8[0]] | (values[q8[1]] << 8);
  v2 = values[q8[2]] | (values[q8[3]] << 8);
  val2 = static_cast<int>(uint32_t(v1) | (uint32_t(v2) << 16));
}

// Tile-loader / dot-product function pointer types. NOT the typedefs of ggml-common.h: the IQ loaders carry one extra
// parameter, `blocks_left` (number of valid blocks of the row from the current K window on), so that the 32-element
// IQ4_NL type can run on K = 128 * n (e.g. the NF ffn_down, K = 640 = 20 blocks) without reading past the row end.
typedef void (*iq_allocate_tiles_cuda_t)(int** x_ql, half2** x_dm, int** x_qh, int** x_sc);
typedef void (*iq_load_tiles_cuda_t)(
    const void* __restrict__ vx,
    int* __restrict__ x_ql,
    half2* __restrict__ x_dm,
    int* __restrict__ x_qh,
    int* __restrict__ x_sc,
    const int& i_offset,
    const int& i_max,
    const int& k,
    const int& blocks_per_row,
    const int& blocks_left);
typedef float (*iq_vec_dot_q_mul_mat_cuda_t)(
    const int* __restrict__ x_ql,
    const half2* __restrict__ x_dm,
    const int* __restrict__ x_qh,
    const int* __restrict__ x_sc,
    const int* __restrict__ y_qs,
    const half2* __restrict__ y_ms,
    const int& i,
    const int& j,
    const int& k);
