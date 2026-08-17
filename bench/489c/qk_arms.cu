// SPDX-License-Identifier: Apache-2.0
//
// #489 (c) / #726 -- QK microbench arms for the int8-KV question.
//
// THE QUESTION, restated so the kernel cannot drift from it: does QK with the
// K-cache held in int8 and multiplied by NATIVE IMMA beat the deployed fp8-KV
// path at decode depths, INCLUDING the deep end? The published -72% @58K
// inversion came from a dequant-to-bf16 Triton lane. That lane is not what is
// measured here -- arm A never materialises a bf16 K.
//
// THREE ARMS, one file, so the tile shape and the memory traffic are the same
// question asked three ways:
//
//   A  int8 K + IMMA, dequant-free  -- mma.sync m16n8k32.s8 on the s32
//      accumulator; the per-token group scale is applied ONCE to the s32
//      result, never to K before the multiply.
//   B  fp8 KV, the deployed path's shape.
//   C  bf16 reference, for the correctness bound only. Never a speed claim.
//
// PER-CARD FACT THAT DECIDES THE SHAPE OF ARM B, and the reason #489 forbids
// averaging across this rig: fp8 tensor-core MMA requires sm_89+. On sm_86
// (both 3080s) arm B MUST dequantise fp8 -> half and run HMMA, because there
// is no fp8 MMA to run; on sm_120 (the 5090) it can use the native path. Arm A
// is native on both (IMMA has existed since sm_75). So the two card families
// are not running the same comparison, and a mean over them would be a
// number about nothing. ARM_B_IS_NATIVE is compiled per target and reported.

#include <cuda_fp16.h>
#include <cstdint>

#if (__CUDA_ARCH__ >= 890)
#define ARM_B_IS_NATIVE 1
#else
#define ARM_B_IS_NATIVE 0
#endif

// One m16n8k32 IMMA tile. a = 4x b32 (16 int8 lanes), b = 2x b32, c/d = 4x s32.
__device__ __forceinline__ void imma_m16n8k32(
    const uint32_t a[4], const uint32_t b[2], int32_t d[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// ---------------------------------------------------------------- arm A ----
// int8 Q x int8 K -> s32, scaled once at the end. K is NEVER widened.
extern "C" __global__ void qk_int8_imma(
    const uint32_t* __restrict__ q,     // packed int8, [heads][d/4]
    const uint32_t* __restrict__ k,     // packed int8, [tokens][d/4]
    const __half* __restrict__ k_scale, // fp16 scale, per token per 64-group
    float* __restrict__ out,            // [heads][tokens]
    int tokens, int d_words, int groups_per_token) {
  const int tok = blockIdx.x * 16 + (threadIdx.x >> 2);
  if (tok >= tokens) return;

  int32_t acc[4] = {0, 0, 0, 0};
  uint32_t af[4], bf[2];

  // Walk the head dimension in k=32 (8 words) steps.
  for (int w = 0; w < d_words; w += 8) {
#pragma unroll
    for (int i = 0; i < 4; ++i) af[i] = q[w + i];
#pragma unroll
    for (int i = 0; i < 2; ++i) bf[i] = k[(size_t)tok * d_words + w + i];
    imma_m16n8k32(af, bf, acc);
  }

  // ONE dequant, on the accumulator. This is the whole point of arm A: the
  // published inversion's lane widened K to bf16 before the multiply and paid
  // for it in bandwidth and in a second pass; here the scale is a scalar
  // multiply on an s32 that is already in a register.
  const float s = __half2float(k_scale[(size_t)tok * groups_per_token]);
  out[(size_t)blockIdx.y * tokens + tok] =
      (float)(acc[0] + acc[1] + acc[2] + acc[3]) * s;
}

// ---------------------------------------------------------------- arm B ----
// fp8 KV in the deployed path's shape. On sm_86 there is no fp8 MMA, so the
// honest shape is dequantise-then-HMMA; the widening is the cost being
// measured, not an artefact of the harness.
extern "C" __global__ void qk_fp8_deployed(
    const __half* __restrict__ q, const uint8_t* __restrict__ k_fp8,
    const __half* __restrict__ k_scale, float* __restrict__ out, int tokens,
    int d, int groups_per_token) {
  const int tok = blockIdx.x * blockDim.x + threadIdx.x;
  if (tok >= tokens) return;
  float acc = 0.f;
  const float s = __half2float(k_scale[(size_t)tok * groups_per_token]);
  for (int i = 0; i < d; ++i) {
    // e4m3 -> float, the widening arm A does not perform.
    const uint8_t byte = k_fp8[(size_t)tok * d + i];
    const int sgn = (byte >> 7) & 1;
    const int exp = (byte >> 3) & 0xF;
    const int man = byte & 0x7;
    float v = exp == 0 ? ldexpf((float)man, -9)
                       : ldexpf(1.f + (float)man * 0.125f, exp - 7);
    if (sgn) v = -v;
    acc += __half2float(q[i]) * v * s;
  }
  out[(size_t)blockIdx.y * tokens + tok] = acc;
}

// ---------------------------------------------------------------- arm C ----
// bf16 reference. Correctness bound only -- never quoted as a speed result.
extern "C" __global__ void qk_bf16_reference(
    const float* __restrict__ q, const float* __restrict__ k,
    float* __restrict__ out, int tokens, int d) {
  const int tok = blockIdx.x * blockDim.x + threadIdx.x;
  if (tok >= tokens) return;
  float acc = 0.f;
  for (int i = 0; i < d; ++i) acc += q[i] * k[(size_t)tok * d + i];
  out[(size_t)blockIdx.y * tokens + tok] = acc;
}

// Reported by the runner so the log says which shape arm B actually took on
// this card, rather than the runner inferring it from the arch string.
extern "C" __global__ void report_arm_b_native(int* out) { *out = ARM_B_IS_NATIVE; }

// ---------------------------------------------------------------------------
// Host launchers. Kept HERE and not in the binding because `<<<>>>` is CUDA
// syntax the host compiler cannot parse -- the binding stays plain C++ so the
// .cu keeps compiling with `nvcc -cubin` for both targets, which is the pin
// the ticket rests on.
extern "C" {

void c_launch_int8_imma(const void* q, const void* k, const void* ks, float* out,
                        int tokens, int heads, int d_words, int groups) {
  dim3 grid((tokens + 15) / 16, heads);
  qk_int8_imma<<<grid, 64>>>((const uint32_t*)q, (const uint32_t*)k,
                             (const __half*)ks, out, tokens, d_words, groups);
}

void c_launch_fp8_deployed(const void* q, const void* k, const void* ks,
                           float* out, int tokens, int heads, int d, int groups) {
  dim3 grid((tokens + 255) / 256, heads);
  qk_fp8_deployed<<<grid, 256>>>((const __half*)q, (const uint8_t*)k,
                                 (const __half*)ks, out, tokens, d, groups);
}

void c_launch_bf16_reference(const float* q, const float* k, float* out,
                             int tokens, int heads, int d) {
  dim3 grid((tokens + 255) / 256, heads);
  qk_bf16_reference<<<grid, 256>>>(q, k, out, tokens, d);
}

int c_arm_b_native(void) {
  int* d = nullptr;
  cudaMalloc(&d, sizeof(int));
  report_arm_b_native<<<1, 1>>>(d);
  int h = -1;
  cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d);
  return h;
}

}  // extern "C"
