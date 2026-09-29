// NVFP4 W4A8-g16 DECODE GEMV for sm_86 (RTX 3080): small M (1..48) on the INT8 tensor cores, reading the
// NATIVE NVFP4 byte layout straight from global memory (no shared-memory staging of weights, no repack).
//
// Agent N4D (user order 2026-09-25). Same bytes and same arithmetic as N4A's GEMM (nvfp4_w4a8_sm86.cuh):
//   weight        uint8 [Nw, Kp/2] E2M1, element 2j in the LOW nibble of byte j (row stride ldw >= Kp/2)
//   weight_scale  e4m3 128x4-swizzled: byte (n, kb) at (n/128)*tile_stride + (kb/4)*512 + (n%32)*16
//                 + ((n%128)/32)*4 + kb%4
//   per 16-block b: exact INT32 block sum S of (2*e2m1) x q8(x) on mma.s8, INT32->FP32 by the magic accumulator
//   init (bits of 1.5*2^23 + S), t = fma(f, s, -1.5*2^23*s) == s*S exactly, acc += t in FP32, final factor
//   s_x[m] * weight_scale_2 * 2^119 (raw E4M3 bits = value * 2^-120, /2 for the x2 table). Output bf16.
//
// Why a separate kernel for small M: N4A's GEMM stages 32 B per weight row per k-tile through shared memory,
// i.e. every DRAM access is one 32-B sector of a different row (rows are Kp/2 bytes apart); measured 215-233 GB/s
// in decode. Here every warp streams its rows with 128-bit loads straight into registers (L1 no-allocate, L2
// evict-first), u consecutive 128-element segments per step with the next step already in flight. Launch shape
// (runtime): kw warps split K of one 16-row tile (chunks c = w, w+kw, ...), rw row tiles per CTA; the kw partial
// sums are reduced in shared memory in a FIXED order (deterministic, no second kernel).
//
// Row tile (16 rows, one mma M tile): inside a 128-row scale tile, row tile t (0..7) holds physical rows
//   mma row i -> 64*(t/4) + 8*(t%4) + (i%8) + 32*(i/8)
// so lane (g, q) owns rows rA = ...+g and rB = rA + 32, whose scales share one 16-B swizzle granule: ONE
// 8-byte load gives both rows' 4 block scales of a 64-element K group.
//
// Two modes (both exact per block, both FP32 accumulation):
//  MODE 0 "diag"  (M <= 8): m16n8k32, the 8 mma COLUMNS are the 8 blocks of a 128-element segment. Lane q feeds
//     its own 16 loaded bytes (blocks 2q, 2q+1) as A without any exchange; B is the token's activation masked to
//     "slot block == column", so column n of C accumulates exactly block n. 4 mma per token per segment.
//  MODE 1 "xpose" (M <= 48): m16n8k16 per block, the 8 columns are 8 tokens. The quad transposes its 16-B row
//     segments in registers (4 PRMT + 4 SHFL per row) so lane p holds elements 4p..4p+3 of every block; the
//     activation comes in a permuted layout from the quantiser (lane-contiguous). 8 mma per n-tile per segment.
//
// FUSED activation quantisation (row 25 of the fLLiper release table, opt-in SGLANG_W4A8_DECODE_FUSED_QUANT=1, only
// compiled with -DSGL_W4A8_DEC_FUSED, i.e. in the separate JIT module nvfp4_w4a8_decode_fused_sm86): the GEMV
// quantises the bf16 activation itself instead of a separate quant launch before it (measured 25.09. in CUDA
// graphs: "full" - "gemm" = 2.0-3.5 us per call on the 27B D shapes, 10-15 % of D.down). Protocol, per launch:
//   (1) every CTA issues its first weight/scale loads (they never touch the activation);
//   (2) rows are CLAIMED through sem[0] (atomicAdd) by whichever CTAs arrive first -- normally one row each -- and
//       quantised with arithmetic identical to w4a8_dec_quant_kernel (bit for bit), then published (red.release on
//       sem[1]). Claiming instead of a static owner is what makes the hand-over deadlock free: a CTA only waits for
//       rows held by running CTAs that do not wait themselves (a static owner may never get a slot);
//   (3) thread 0 polls sem[1] (relaxed) until it reaches M, one fence.acq_rel, bar.sync -- the activation is read
//       with COHERENT loads from here on (ld.global, never .nc: it was written inside this kernel);
//   (4) the CTA that finishes last (atomicAdd sem[2]) resets sem[0..2] to 0 for the next launch in stream order.
// Row tiles, stagger and accumulation order are those of the unfused kernel (blockIdx), so the output is
// bit-identical to quant + GEMM. sem must not be shared by two launches that may run concurrently:
// the Python side keys it by (device, weight pointer, stream).

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace device::nvfp4_w4a8_dec {

constexpr uint32_t kMagicBits = 0x4B400000u;  // float 1.5 * 2^23
constexpr float kMagic = 12582912.0f;
constexpr float kFinal = 6.6461399789245794e35f;  // 2^119

struct DecParams {
  const uint8_t* weight;
  const uint8_t* wscale;
  const int8_t* xq;  // MODE 0: natural [M, ldx]; MODE 1: permuted per 128-element segment
  const float* xs;
  const float* gscale;
  __nv_bfloat16* out;
  int M, N, Nw;
  int ldw, ldx, ldo;
  int nseg;  // Kp / 128
  long long tile_stride;
  int kw;       // warps splitting K for one row tile (1..8)
  int rw;       // row tiles per CTA (their warps read the same segments -> activation hits in L1)
  int n_tiles;  // row tiles in total (ceil128(Nw) / 16)
};

// FUSED kernel parameters (see the header): bf16 activation [M, ldxb] (K == Kp columns), quantised into xq_w / xs_w
// (the same buffers as xq / xs) by the CTAs that claim its rows; sem[0] rows claimed, sem[1] rows published, sem[2]
// CTAs done. A separate type, so the unfused kernels keep their parameter block (and their code) unchanged.
struct FusedDecParams : DecParams {
  const __nv_bfloat16* xb16;
  int ldxb;
  int8_t* xq_w;
  float* xs_w;
  unsigned int* sem;
};

template <bool FUSED>
using DecParamsT = typename std::conditional<FUSED, FusedDecParams, DecParams>::type;

// Weights are streamed exactly once: L1 no-allocate and L2 evict-first, so the stream does not push the
// activation (read by every CTA, M*K bytes) and the shared scale granules out of the 5 MB L2 (measured without
// the hint: lm_head M=16 764..1089 us in a GEMM-only loop, i.e. the activation re-fetched from DRAM).
__device__ __forceinline__ uint64_t policy_evict_first() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ uint64_t policy_evict_last() {
  uint64_t pol;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
  return pol;
}
__device__ __forceinline__ uint4 ldg_stream16(const void* p, uint64_t pol) {
  uint4 r;
  asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
      : "l"(p), "l"(pol));
  return r;
}
__device__ __forceinline__ uint4 ldg_keep16(const void* p, uint64_t pol) {
  uint4 r;
  asm("ld.global.nc.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n"
      : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
      : "l"(p), "l"(pol));
  return r;
}
__device__ __forceinline__ uint2 ldg_nc8(const void* p) {
  uint2 r;
  asm("ld.global.nc.v2.u32 {%0,%1}, [%2];\n" : "=r"(r.x), "=r"(r.y) : "l"(p));
  return r;
}
__device__ __forceinline__ uint4 ldg_nc16(const void* p) {
  uint4 r;
  asm("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];\n" : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// Activation load: read-only (.nc, L2 evict-last) in the unfused kernel. In the FUSED kernel the activation is written
// inside the same launch, so it must be a COHERENT load (no .nc) and must not be hoisted above the semaphore wait
// (volatile + memory clobber).
template <bool FUSED>
__device__ __forceinline__ uint4 ldg_act16(const void* p, uint64_t pol) {
  if constexpr (FUSED) {
    uint4 r;
    asm volatile("ld.global.L2::cache_hint.v4.u32 {%0,%1,%2,%3}, [%4], %5;\n"
                 : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
                 : "l"(p), "l"(pol)
                 : "memory");
    return r;
  } else {
    return ldg_keep16(p, pol);
  }
}

// 4 packed E2M1 codes (low 16 bits, nibble i = element i) -> 4 int8 = 2 * e2m1 (identical to N4A's helper).
__device__ __forceinline__ uint32_t e2m1x4_to_s8x4(uint32_t x) {
  const uint32_t pos = __byte_perm(0x03020100u, 0x0C080604u, x);
  const uint32_t neg = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, x);
  const uint32_t sel = ((x & 0x8888u) >> 1) | 0x3210u;
  return __byte_perm(pos, neg, sel);
}

// byte kb of w (E4M3 code) -> FP32 with value e4m3 * 2^-120 (identical to N4A's helper; FTZ must stay off).
__device__ __forceinline__ float e4m3_scaled_bits(uint32_t w, int kb) {
  const uint32_t y = __byte_perm(w, 0u, 0x0444u | (static_cast<uint32_t>(kb) << 12));
  const int z = static_cast<int>(y) >> 4;
  return __int_as_float(z & static_cast<int>(0x87F00000u));
}

__device__ __forceinline__ void mma_s8_16832(
    int (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
  asm(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8_16816_magic(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a0), "r"(a1), "r"(b0), "r"(kMagicBits));
}

// Segment sweep order of a CTA: rotated by a CTA dependent offset, so the resident CTAs do not all read the same K
// offset of their rows at the same time (27B rows are 2048 / 2560 / 8704 B apart). Not shown to matter by itself
// (sweep 25.09.); kept because it costs nothing. Per CTA, not per row tile: the rw row tiles of one CTA must read
// the same activation segments at the same time so that those loads hit in L1.
__device__ __forceinline__ int stagger_of(int cta, int nseg) {
  return static_cast<int>((static_cast<unsigned>(cta) * 2654435761u) >> 8) % nseg;
}
__device__ __forceinline__ int seg_at(int s0, int it, int nseg) {
  const int v = s0 + it;
  return v >= nseg ? v - nseg : v;
}

// physical row of mma row g (0..7) of row tile T; the lane's second row is +32
__device__ __forceinline__ int row_a_of(int T, int g) {
  const int t = T & 7;
  return (T >> 3) * 128 + 64 * (t >> 2) + 8 * (t & 3) + g;
}

// Deterministic cross-warp reduction + final scaling. red: [rw][kw][MT][16] floats (MT tokens per CTA tile).
template <int MT>
__device__ __forceinline__ void reduce_store(const DecParams& p, const float* red, int T0, int m0) {
  __syncthreads();
  const float gs = *p.gscale;
  for (int i = threadIdx.x; i < p.rw * MT * 16; i += blockDim.x) {
    const int rt = i / (MT * 16), mm = (i >> 4) % MT, r = i & 15;
    const int T = T0 + rt;
    const int m = m0 + mm;
    float v = 0.f;
    for (int w = 0; w < p.kw; ++w) v += red[((rt * p.kw + w) * MT + mm) * 16 + r];
    const int n = row_a_of(T, 0) + (r & 7) + 32 * (r >> 3);
    if (T < p.n_tiles && m < p.M && n < p.N) {
      p.out[static_cast<long long>(m) * p.ldo + n] = __float2bfloat16_rn(v * (p.xs[m] * gs * kFinal));
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// FUSED activation quantisation (protocol in the header)
// ---------------------------------------------------------------------------------------------------------------
// One activation row, the whole CTA (blockDim.x a multiple of 32, <= 256). Arithmetic identical to
// w4a8_dec_quant_kernel (= N4A's quantiser = per_token_quant_int8): the max is exact in any order, the rest is the
// same per-element expression, so int8 values and the scale are bit-identical to the separate quant launch.
template <bool PERM>
__device__ __forceinline__ void fused_quant_row(const DecParamsT<true>& p, int row) {
  __shared__ float s_red[8];
  const int K = p.nseg * 128;
  const __nv_bfloat16* xr = p.xb16 + static_cast<long long>(row) * p.ldxb;
  int8_t* qr = p.xq_w + static_cast<long long>(row) * p.ldx;
  const int nthr = blockDim.x;
  float amax = 0.f;
  for (int k = threadIdx.x * 8; k < K; k += nthr * 8) {
    const uint4 v = *reinterpret_cast<const uint4*>(xr + k);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(h[j]);
      amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  if ((threadIdx.x & 31) == 0) s_red[threadIdx.x >> 5] = amax;
  __syncthreads();
  float v = 0.f;
  for (int w = 0; w < (nthr >> 5); ++w) v = fmaxf(v, s_red[w]);
  amax = fmaxf(v, 1e-10f);
  const float inv = __fdiv_rn(127.f, amax);
  if (threadIdx.x == 0) p.xs_w[row] = __fdiv_rn(amax, 127.f);
  for (int k = threadIdx.x * 8; k < K; k += nthr * 8) {
    const uint4 xv = *reinterpret_cast<const uint4*>(xr + k);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&xv);
    uint32_t w[2];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(h[j]);
      const int q0 = static_cast<int>(roundf(__fmul_rn(f.x, inv)));
      const int q1 = static_cast<int>(roundf(__fmul_rn(f.y, inv)));
      const uint32_t pair = (static_cast<uint32_t>(q0) & 0xFFu) | ((static_cast<uint32_t>(q1) & 0xFFu) << 8);
      if (j & 1)
        w[j >> 1] |= pair << 16;
      else
        w[j >> 1] = pair;
    }
    if constexpr (PERM) {
      const int seg = k >> 7, kk = k & 127, b = kk >> 4, pp = (kk & 15) >> 2;
      int8_t* base = qr + seg * 128 + 4 * b;
      *reinterpret_cast<uint32_t*>(base + 32 * pp) = w[0];
      *reinterpret_cast<uint32_t*>(base + 32 * (pp + 1)) = w[1];
    } else {
      *reinterpret_cast<uint2*>(qr + k) = make_uint2(w[0], w[1]);
    }
  }
  __syncthreads();  // s_red is reused by the next row of this CTA
}

// Rows are CLAIMED (atomicAdd sem[0]), not assigned: a CTA only ever waits for rows that a running CTA has claimed
// and is quantising without waiting itself, so the hand-over cannot deadlock however few CTAs are resident (a static
// assignment -- by blockIdx or by arrival ticket -- can: the owner of a missing row may never get a slot). Normally
// the first M CTAs to arrive take one row each; every later CTA sees sem[0] >= M with one relaxed load and skips.
template <bool PERM>
__device__ __forceinline__ void fused_quant_claimed_rows(const DecParamsT<true>& p) {
  __shared__ int s_row;
  unsigned int rows = 0;
  for (;;) {
    if (threadIdx.x == 0) {
      int r = p.M;
      unsigned int seen;
      asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];\n" : "=r"(seen) : "l"(p.sem) : "memory");
      if (seen < static_cast<unsigned int>(p.M)) r = static_cast<int>(atomicAdd(p.sem, 1u));
      s_row = r;
    }
    __syncthreads();
    const int r = s_row;
    __syncthreads();  // s_row is rewritten by the next claim
    if (r >= p.M) break;
    fused_quant_row<PERM>(p, r);
    ++rows;
  }
  if (rows != 0 && threadIdx.x == 0) {
    // fused_quant_row ended on bar.sync, so every store of the CTA is ordered before this release (cumulativity)
    asm volatile("red.release.gpu.global.add.u32 [%0], %1;\n" : : "l"(p.sem + 1), "r"(rows) : "memory");
  }
}

__device__ __forceinline__ void fused_wait(const DecParamsT<true>& p) {
  if (threadIdx.x == 0) {
    const unsigned int want = static_cast<unsigned int>(p.M);
    unsigned int v;
    for (;;) {  // relaxed polls (an acquire per poll would invalidate the SM's L1 every time)
      asm volatile("ld.relaxed.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p.sem + 1) : "memory");
      if (v >= want) break;
      __nanosleep(32);
    }
    asm volatile("fence.acq_rel.gpu;\n" : : : "memory");  // acquire side of the publishers' release
  }
  __syncthreads();
}

// The last CTA to finish resets the semaphore for the next launch: when the done count reaches nctas, every claim
// is made and every wait has returned (thread 0's poll completed before its own done increment, program order after
// the acquire fence). No fence here: nothing this CTA wrote is read through the semaphore, and the next launch is
// ordered by the kernel boundary.
__device__ __forceinline__ void fused_done(const DecParamsT<true>& p, int nctas) {
  if (threadIdx.x == 0) {
    const unsigned int d = atomicAdd(p.sem + 2, 1u);
    if (d == static_cast<unsigned int>(nctas - 1)) {
      atomicExch(p.sem, 0u);
      atomicExch(p.sem + 1, 0u);
      atomicExch(p.sem + 2, 0u);
    }
  }
}

// ---------------------------------------------------------------------------------------------------------------
// MODE 0: M <= MT (MT <= 8), block-diagonal B on m16n8k32
// ---------------------------------------------------------------------------------------------------------------
template <int MT, int U, bool FUSED>
__global__ void __launch_bounds__(256) w4a8_dec_diag_kernel(const DecParamsT<FUSED> p) {
  extern __shared__ float red[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int g = lane >> 2, q = lane & 3;
  const int rt = wid / p.kw, warp = wid - rt * p.kw;  // row tile within the CTA, K slice
  const int T0 = blockIdx.x * p.rw;
  const int T = min(T0 + rt, p.n_tiles - 1);  // a CTA past the last tile recomputes it and stores nothing
  const int rA = row_a_of(T, g), rB = rA + 32;
  const bool okA = rA < p.Nw, okB = rB < p.Nw;
  const uint8_t* wA = p.weight + static_cast<long long>(okA ? rA : 0) * p.ldw + 16 * q;
  const uint8_t* wB = p.weight + static_cast<long long>(okB ? rB : 0) * p.ldw + 16 * q;
  const int t = T & 7;
  const uint8_t* sc = p.wscale + static_cast<long long>(T >> 3) * p.tile_stride + (8 * (t & 3) + g) * 16 +
                      8 * (t >> 2) + (q >> 1) * 512;
  const int kbi = 2 * (q & 1);  // this lane's two blocks are kb%4 = kbi, kbi+1 of its granule
  const bool act = (q == (g >> 1));
  // lanes with act supply block g of every segment; mma j (0..3) carries block 2q + (j>>1) of lane q
  const uint32_t mlo = (act && (g & 1) == 0) ? 0xFFFFFFFFu : 0u;  // j = 0, 1
  const uint32_t mhi = (act && (g & 1) == 1) ? 0xFFFFFFFFu : 0u;  // j = 2, 3
  const int8_t* xb = p.xq + 16 * g;

  float acc[MT][4];
#pragma unroll
  for (int m = 0; m < MT; ++m)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[m][e] = 0.f;

  const int s0 = stagger_of(blockIdx.x, p.nseg);  // per CTA: its rw row tiles sweep K in lockstep (L1 reuse)
  const uint64_t pol_w = policy_evict_first(), pol_x = policy_evict_last();
  struct Buf {
    uint4 a, b;
    uint2 s;
  };
  auto load_seg = [&](int l, Buf& f) {
    f.a = make_uint4(0, 0, 0, 0);
    f.b = f.a;
    f.s = make_uint2(0, 0);
    if (l < p.nseg) {
      const int sg = seg_at(s0, l, p.nseg);
      if (okA) f.a = ldg_stream16(wA + sg * 64, pol_w);
      if (okB) f.b = ldg_stream16(wB + sg * 64, pol_w);
      f.s = ldg_nc8(sc + static_cast<long long>(sg) * 1024);
    }
  };
  auto compute_seg = [&](int l, const Buf& f) {
    if (l >= p.nseg) return;
    const int sg = seg_at(s0, l, p.nseg);
    uint4 xv[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      xv[m] = make_uint4(0, 0, 0, 0);
      if (act && m < p.M) xv[m] = ldg_act16<FUSED>(xb + static_cast<long long>(m) * p.ldx + sg * 128, pol_x);
    }
    // A fragments: j-th 32-bit word of each row = 8 elements; low 16 bits -> k-lo slots, high -> k-hi slots
    const uint32_t wa[4] = {f.a.x, f.a.y, f.a.z, f.a.w};
    const uint32_t wb[4] = {f.b.x, f.b.y, f.b.z, f.b.w};
    uint32_t A[4][4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      A[j][0] = e2m1x4_to_s8x4(wa[j]);
      A[j][1] = e2m1x4_to_s8x4(wb[j]);
      A[j][2] = e2m1x4_to_s8x4(wa[j] >> 16);
      A[j][3] = e2m1x4_to_s8x4(wb[j] >> 16);
    }
    const float sA0 = e4m3_scaled_bits(f.s.x, kbi), sA1 = e4m3_scaled_bits(f.s.x, kbi + 1);
    const float sB0 = e4m3_scaled_bits(f.s.y, kbi), sB1 = e4m3_scaled_bits(f.s.y, kbi + 1);
    const float nA0 = sA0 * -kMagic, nA1 = sA1 * -kMagic, nB0 = sB0 * -kMagic, nB1 = sB1 * -kMagic;
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      int d[4] = {static_cast<int>(kMagicBits), static_cast<int>(kMagicBits), static_cast<int>(kMagicBits),
                  static_cast<int>(kMagicBits)};
      const uint32_t xw[4] = {xv[m].x, xv[m].y, xv[m].z, xv[m].w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint32_t msk = (j >> 1) ? mhi : mlo;
        mma_s8_16832(d, A[j][0], A[j][1], A[j][2], A[j][3], xw[2 * (j & 1)] & msk, xw[2 * (j & 1) + 1] & msk);
      }
      acc[m][0] += __fmaf_rn(__int_as_float(d[0]), sA0, nA0);
      acc[m][1] += __fmaf_rn(__int_as_float(d[1]), sA1, nA1);
      acc[m][2] += __fmaf_rn(__int_as_float(d[2]), sB0, nB0);
      acc[m][3] += __fmaf_rn(__int_as_float(d[3]), sB1, nB1);
    }
  };
  // warp w sweeps chunks w, w+kw, ... of U consecutive segments; the next chunk is in flight during the current one
  Buf cur[U], nxt[U];
#pragma unroll
  for (int u = 0; u < U; ++u) load_seg(warp * U + u, cur[u]);
  if constexpr (FUSED) {  // the first weight loads are in flight while the activation rows are quantised
    fused_quant_claimed_rows<false>(p);
    fused_wait(p);
  }
  for (int c = warp; c * U < p.nseg; c += p.kw) {
#pragma unroll
    for (int u = 0; u < U; ++u) load_seg((c + p.kw) * U + u, nxt[u]);
#pragma unroll
    for (int u = 0; u < U; ++u) compute_seg(c * U + u, cur[u]);
#pragma unroll
    for (int u = 0; u < U; ++u) cur[u] = nxt[u];
  }

  // quad reduction (blocks 2q, 2q+1 of every segment -> whole row), identical result in all 4 lanes
#pragma unroll
  for (int m = 0; m < MT; ++m) {
    float vA = acc[m][0] + acc[m][1];
    float vB = acc[m][2] + acc[m][3];
    vA += __shfl_xor_sync(0xffffffffu, vA, 1);
    vB += __shfl_xor_sync(0xffffffffu, vB, 1);
    vA += __shfl_xor_sync(0xffffffffu, vA, 2);
    vB += __shfl_xor_sync(0xffffffffu, vB, 2);
    if (q == 0) {
      red[(wid * MT + m) * 16 + g] = vA;
      red[(wid * MT + m) * 16 + g + 8] = vB;
    }
  }
  reduce_store<MT>(p, red, T0, 0);
  if constexpr (FUSED) fused_done(p, static_cast<int>(gridDim.x));
}

// ---------------------------------------------------------------------------------------------------------------
// MODE 1: NT n-tiles of 8 tokens, quad register transpose, m16n8k16 per block
// ---------------------------------------------------------------------------------------------------------------
// Quad transpose: lane q holds Y[t] (t = target lane); returns Z[src] = Y_src[q].
__device__ __forceinline__ void quad_transpose(const uint32_t (&Y)[4], uint32_t (&Z)[4], int q) {
  const bool b0 = q & 1, b1 = (q >> 1) & 1;
  uint32_t R[4];
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const uint32_t own = b0 ? Y[2 * j + 1] : Y[2 * j];
    const uint32_t snd = b0 ? Y[2 * j] : Y[2 * j + 1];
    const uint32_t rcv = __shfl_xor_sync(0xffffffffu, snd, 1);
    R[2 * j + 0] = b0 ? rcv : own;
    R[2 * j + 1] = b0 ? own : rcv;
  }
#pragma unroll
  for (int c = 0; c < 2; ++c) {
    const uint32_t snd = b1 ? R[c] : R[2 + c];
    const uint32_t rcv = __shfl_xor_sync(0xffffffffu, snd, 2);
    Z[c] = b1 ? rcv : R[c];
    Z[2 + c] = b1 ? R[2 + c] : rcv;
  }
}

template <int NT, int U, bool FUSED>
__global__ void __launch_bounds__(256) w4a8_dec_xpose_kernel(const DecParamsT<FUSED> p) {
  constexpr int MT = NT * 8;
  extern __shared__ float red[];
  const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
  const int g = lane >> 2, q = lane & 3;
  const int rt = wid / p.kw, warp = wid - rt * p.kw;  // row tile within the CTA, K slice
  const int T0 = blockIdx.x * p.rw;
  const int T = min(T0 + rt, p.n_tiles - 1);  // a CTA past the last tile recomputes it and stores nothing
  const int m0 = blockIdx.y * MT;
  const int rA = row_a_of(T, g), rB = rA + 32;
  const bool okA = rA < p.Nw, okB = rB < p.Nw;
  const uint8_t* wA = p.weight + static_cast<long long>(okA ? rA : 0) * p.ldw + 16 * q;
  const uint8_t* wB = p.weight + static_cast<long long>(okB ? rB : 0) * p.ldw + 16 * q;
  const int t = T & 7;
  const uint8_t* sc =
      p.wscale + static_cast<long long>(T >> 3) * p.tile_stride + (8 * (t & 3) + g) * 16 + 8 * (t >> 2);
  // permuted activation: token m, segment s: 128 bytes = [lane-quad position q][block b][4 bytes]
  const int8_t* xb = p.xq + 32 * q;

  float acc[NT][4];
#pragma unroll
  for (int n = 0; n < NT; ++n)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[n][e] = 0.f;

  const int s0 = stagger_of(blockIdx.x, p.nseg);  // per CTA: its rw row tiles sweep K in lockstep (L1 reuse)
  const uint64_t pol_w = policy_evict_first(), pol_x = policy_evict_last();
  struct Buf {
    uint4 a, b;
    uint2 s0, s1;
  };
  auto load_seg = [&](int l, Buf& f) {
    f.a = make_uint4(0, 0, 0, 0);
    f.b = f.a;
    f.s0 = make_uint2(0, 0);
    f.s1 = f.s0;
    if (l < p.nseg) {
      const int sg = seg_at(s0, l, p.nseg);
      if (okA) f.a = ldg_stream16(wA + sg * 64, pol_w);
      if (okB) f.b = ldg_stream16(wB + sg * 64, pol_w);
      f.s0 = ldg_nc8(sc + static_cast<long long>(sg) * 1024);
      f.s1 = ldg_nc8(sc + static_cast<long long>(sg) * 1024 + 512);
    }
  };
  auto compute_seg = [&](int l, const Buf& f) {
    if (l >= p.nseg) return;
    const int sg = seg_at(s0, l, p.nseg);
    uint32_t xw[NT][8];
#pragma unroll
    for (int n = 0; n < NT; ++n) {
      const int m = m0 + n * 8 + g;
      uint4 lo = make_uint4(0, 0, 0, 0), hi = lo;
      if (m < p.M) {
        const int8_t* src = xb + static_cast<long long>(m) * p.ldx + sg * 128;
        lo = ldg_act16<FUSED>(src, pol_x);
        hi = ldg_act16<FUSED>(src + 16, pol_x);
      }
      xw[n][0] = lo.x; xw[n][1] = lo.y; xw[n][2] = lo.z; xw[n][3] = lo.w;
      xw[n][4] = hi.x; xw[n][5] = hi.y; xw[n][6] = hi.z; xw[n][7] = hi.w;
    }
    // rows -> per-target words: Y[t] = (half t of block 2q, half t of block 2q+1)
    uint32_t YA[4], YB[4], ZA[4], ZB[4];
    {
      const uint32_t wa[4] = {f.a.x, f.a.y, f.a.z, f.a.w};
      const uint32_t wb[4] = {f.b.x, f.b.y, f.b.z, f.b.w};
#pragma unroll
      for (int tt = 0; tt < 4; ++tt) {
        const uint32_t sel = (tt & 1) ? 0x7632u : 0x5410u;
        YA[tt] = __byte_perm(wa[tt >> 1], wa[2 + (tt >> 1)], sel);
        YB[tt] = __byte_perm(wb[tt >> 1], wb[2 + (tt >> 1)], sel);
      }
    }
    quad_transpose(YA, ZA, q);
    quad_transpose(YB, ZB, q);
#pragma unroll
    for (int b = 0; b < 8; ++b) {
      const uint32_t hA = (b & 1) ? (ZA[b >> 1] >> 16) : ZA[b >> 1];
      const uint32_t hB = (b & 1) ? (ZB[b >> 1] >> 16) : ZB[b >> 1];
      const uint32_t a0 = e2m1x4_to_s8x4(hA), a1 = e2m1x4_to_s8x4(hB);
      const uint2 sw = (b < 4) ? f.s0 : f.s1;
      const float sA = e4m3_scaled_bits(sw.x, b & 3), sB = e4m3_scaled_bits(sw.y, b & 3);
      const float nA = sA * -kMagic, nB = sB * -kMagic;
#pragma unroll
      for (int n = 0; n < NT; ++n) {
        int d[4];
        mma_s8_16816_magic(d, a0, a1, xw[n][b]);
        acc[n][0] += __fmaf_rn(__int_as_float(d[0]), sA, nA);
        acc[n][1] += __fmaf_rn(__int_as_float(d[1]), sA, nA);
        acc[n][2] += __fmaf_rn(__int_as_float(d[2]), sB, nB);
        acc[n][3] += __fmaf_rn(__int_as_float(d[3]), sB, nB);
      }
    }
  };
  Buf cur[U], nxt[U];
#pragma unroll
  for (int u = 0; u < U; ++u) load_seg(warp * U + u, cur[u]);
  if constexpr (FUSED) {  // the first weight loads are in flight while the activation rows are quantised
    fused_quant_claimed_rows<true>(p);
    fused_wait(p);
  }
  for (int c = warp; c * U < p.nseg; c += p.kw) {
#pragma unroll
    for (int u = 0; u < U; ++u) load_seg((c + p.kw) * U + u, nxt[u]);
#pragma unroll
    for (int u = 0; u < U; ++u) compute_seg(c * U + u, cur[u]);
#pragma unroll
    for (int u = 0; u < U; ++u) cur[u] = nxt[u];
  }
  // C fragment: d0,d1 -> row g (rA), tokens 2q, 2q+1 of the n-tile; d2,d3 -> row g+8 (rB)
#pragma unroll
  for (int n = 0; n < NT; ++n) {
    const int mm = n * 8 + 2 * q;
    red[(wid * MT + mm) * 16 + g] = acc[n][0];
    red[(wid * MT + mm + 1) * 16 + g] = acc[n][1];
    red[(wid * MT + mm) * 16 + g + 8] = acc[n][2];
    red[(wid * MT + mm + 1) * 16 + g + 8] = acc[n][3];
  }
  reduce_store<MT>(p, red, T0, m0);
  if constexpr (FUSED) fused_done(p, static_cast<int>(gridDim.x * gridDim.y));
}

// bf16 [M, K] -> int8 (natural [M, ldq] or segment-permuted) + fp32 scale [M]; arithmetic identical to N4A's
// w4a8_quant_act_kernel (= per_token_quant_int8): amax = max(max|x|, 1e-10), s = amax/127, q = round(x*127/amax).
// Permuted: element k of segment S (k = 16b + 4p + j) goes to byte S*128 + 32p + 4b + j. K == Kp here.
template <int kThreads, bool PERM>
__global__ void __launch_bounds__(kThreads) w4a8_dec_quant_kernel(
    const __nv_bfloat16* __restrict__ x, int ldx, int K, int8_t* __restrict__ xq, int ldq, float* __restrict__ xs) {
  const int row = blockIdx.x;
  const __nv_bfloat16* xr = x + static_cast<long long>(row) * ldx;
  int8_t* qr = xq + static_cast<long long>(row) * ldq;
  float amax = 0.f;
  for (int k = threadIdx.x * 8; k < K; k += kThreads * 8) {
    const uint4 v = *reinterpret_cast<const uint4*>(xr + k);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(h[j]);
      amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  __shared__ float red[kThreads / 32];
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < kThreads / 32 ? red[threadIdx.x] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if (threadIdx.x == 0) red[0] = v;
  }
  __syncthreads();
  amax = fmaxf(red[0], 1e-10f);
  const float inv = __fdiv_rn(127.f, amax);
  if (threadIdx.x == 0) xs[row] = __fdiv_rn(amax, 127.f);
  for (int k = threadIdx.x * 8; k < K; k += kThreads * 8) {
    const uint4 v = *reinterpret_cast<const uint4*>(xr + k);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
    uint32_t w[2];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 f = __bfloat1622float2(h[j]);
      const int q0 = static_cast<int>(roundf(__fmul_rn(f.x, inv)));
      const int q1 = static_cast<int>(roundf(__fmul_rn(f.y, inv)));
      const uint32_t pair = (static_cast<uint32_t>(q0) & 0xFFu) | ((static_cast<uint32_t>(q1) & 0xFFu) << 8);
      if (j & 1)
        w[j >> 1] |= pair << 16;
      else
        w[j >> 1] = pair;
    }
    if constexpr (PERM) {
      // k = 8i: elements k..k+3 -> (b = (k%128)/16, p = (k%16)/4), k+4..k+7 -> p+1
      const int seg = k >> 7, kk = k & 127, b = kk >> 4, pp = (kk & 15) >> 2;
      int8_t* base = qr + seg * 128 + 4 * b;
      *reinterpret_cast<uint32_t*>(base + 32 * pp) = w[0];
      *reinterpret_cast<uint32_t*>(base + 32 * (pp + 1)) = w[1];
    } else {
      *reinterpret_cast<uint2*>(qr + k) = make_uint2(w[0], w[1]);
    }
  }
}

}  // namespace device::nvfp4_w4a8_dec

namespace host::nvfp4_w4a8_dec {

using namespace device::nvfp4_w4a8_dec;

template <int MT, int U, bool FUSED>
inline void launch_diag(const DecParamsT<FUSED>& p, DLDevice device) {
  const int smem = p.rw * p.kw * MT * 16 * 4;
  host::LaunchKernel(dim3((p.n_tiles + p.rw - 1) / p.rw), dim3(p.rw * p.kw * 32), device, smem)(
      w4a8_dec_diag_kernel<MT, U, FUSED>, p);
}
template <int NT, int U, bool FUSED>
inline void launch_xpose(const DecParamsT<FUSED>& p, DLDevice device) {
  const int smem = p.rw * p.kw * NT * 8 * 16 * 4;
  host::LaunchKernel(dim3((p.n_tiles + p.rw - 1) / p.rw, (p.M + NT * 8 - 1) / (NT * 8)), dim3(p.rw * p.kw * 32),
                     device, smem)(w4a8_dec_xpose_kernel<NT, U, FUSED>, p);
}

template <int U, bool FUSED>
inline void dispatch_u(int mode, const DecParamsT<FUSED>& p, DLDevice device) {
  if (mode == 0) {
    if (p.M <= 1) return launch_diag<1, U, FUSED>(p, device);
    if (p.M <= 2) return launch_diag<2, U, FUSED>(p, device);
    if (p.M <= 4) return launch_diag<4, U, FUSED>(p, device);
    return launch_diag<8, U, FUSED>(p, device);
  }
  switch ((p.M + 7) / 8) {
    case 1: return launch_xpose<1, U, FUSED>(p, device);
    case 2: return launch_xpose<2, U, FUSED>(p, device);
    case 3: return launch_xpose<3, U, FUSED>(p, device);
    case 4: return launch_xpose<4, U, FUSED>(p, device);
    case 5: return launch_xpose<5, U, FUSED>(p, device);
    default: return launch_xpose<6, U, FUSED>(p, device);
  }
}

template <bool FUSED>
inline void dispatch(int64_t u, int mode, const DecParamsT<FUSED>& p, DLDevice device) {
  switch (u) {
    case 1: return dispatch_u<1, FUSED>(mode, p, device);
    case 2: return dispatch_u<2, FUSED>(mode, p, device);
    default: return dispatch_u<4, FUSED>(mode, p, device);
  }
}

}  // namespace host::nvfp4_w4a8_dec

// out bf16 [M, >= n_out]; xq int8 [M, Kp] (mode 0 natural, mode 1 permuted); xs fp32 [M]; weight u8 [Nw, Kp/2];
// wscale e4m3/u8 [ceil128(Nw), Ks] contiguous or tile view [ceil128(Nw)/128, Ks*128]; gscale fp32 [1].
namespace host::nvfp4_w4a8_dec {

// Validation + launch parameters shared by the unfused and the FUSED entry point.
inline DecParams make_dec_params(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView xq,
    tvm::ffi::TensorView xs,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView wscale,
    tvm::ffi::TensorView gscale,
    int64_t n_out,
    int64_t mode,
    int64_t kw,
    int64_t rw,
    int64_t u,
    DLDevice* device_out) {
  using namespace host;

  SymbolicDevice dev;
  dev.set_options<kDLCUDA>();
  SymbolicSize M{"M"}, Kp{"Kp"}, Nw{"Nw"}, Kh{"Kp/2"}, NO{"out_width"};
  SymbolicSize ldx{"ldx"}, ldw{"ldw"}, ldo{"ldo"};
  TensorMatcher({M, Kp}).with_strides({ldx, 1}).with_dtype<int8_t>().with_device(dev).verify(xq);
  TensorMatcher({M}).with_dtype<float>().with_device(dev).verify(xs);
  TensorMatcher({Nw, Kh}).with_strides({ldw, 1}).with_dtype<uint8_t>().with_device(dev).verify(weight);
  TensorMatcher({M, NO}).with_strides({ldo, 1}).with_dtype<bf16_t>().with_device(dev).verify(out);
  TensorMatcher({1}).with_dtype<float>().with_device(dev).verify(gscale);

  const int64_t m = M.unwrap(), kp = Kp.unwrap(), nw = Nw.unwrap();
  RuntimeCheck(Kh.unwrap() * 2 == kp, "xq has ", kp, " columns but the packed weight holds ", Kh.unwrap() * 2);
  RuntimeCheck(kp % 128 == 0, "decode GEMV needs Kp % 128 == 0, got ", kp);
  RuntimeCheck(mode == 0 || mode == 1, "mode must be 0 (diag) or 1 (xpose)");
  RuntimeCheck(m >= 1 && m <= (mode == 0 ? 8 : 48), "decode GEMV: M out of range for mode ", mode, ": ", m);
  RuntimeCheck(kw >= 1 && rw >= 1 && kw * rw <= 8, "kw * rw must be in [1, 8]");
  RuntimeCheck(u == 1 || u == 2 || u == 4, "u must be 1, 2 or 4");
  RuntimeCheck(n_out > 0 && n_out <= nw && n_out <= NO.unwrap(), "n_out out of range: ", n_out);
  RuntimeCheck(ldx.unwrap() % 16 == 0 && ldw.unwrap() % 16 == 0, "row strides must be 16-byte aligned");
  RuntimeCheck(reinterpret_cast<uintptr_t>(xq.data_ptr()) % 16 == 0, "xq must be 16-byte aligned");
  RuntimeCheck(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0, "weight must be 16-byte aligned");
  RuntimeCheck(wscale.ndim() == 2 && wscale.stride(1) == 1, "weight_scale must be 2-D with unit inner stride");
  RuntimeCheck(wscale.dtype().bits == 8, "weight_scale must be an 8-bit (e4m3) tensor");
  RuntimeCheck(reinterpret_cast<uintptr_t>(wscale.data_ptr()) % 16 == 0, "weight_scale must be 16-byte aligned");

  const int64_t nseg = kp / 128;
  const int64_t tiles128 = (nw + 127) / 128;
  int64_t tile_stride = 0;
  if (wscale.size(0) == tiles128 * 128) {
    RuntimeCheck(wscale.stride(0) == wscale.size(1), "a [N_pad, Ks] weight_scale must be contiguous");
    RuntimeCheck(wscale.size(1) % 4 == 0 && wscale.size(1) >= nseg * 8, "weight_scale has ", wscale.size(1),
                 " columns, needs >= ", nseg * 8);
    tile_stride = wscale.size(1) * 128;
  } else {
    RuntimeCheck(wscale.size(0) == tiles128, "weight_scale rows: expected ", tiles128 * 128, " or ", tiles128,
                 " got ", wscale.size(0));
    RuntimeCheck(wscale.size(1) % 512 == 0 && wscale.size(1) / 512 >= nseg * 2, "weight_scale tile view has ",
                 wscale.size(1), " columns, needs ", nseg * 1024);
    RuntimeCheck(wscale.stride(0) % 16 == 0, "weight_scale tile-row stride must be 16-byte aligned");
    tile_stride = wscale.stride(0);
  }

  DecParams p{};
  p.weight = static_cast<const uint8_t*>(weight.data_ptr());
  p.wscale = static_cast<const uint8_t*>(wscale.data_ptr());
  p.xq = static_cast<const int8_t*>(xq.data_ptr());
  p.xs = static_cast<const float*>(xs.data_ptr());
  p.gscale = static_cast<const float*>(gscale.data_ptr());
  p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  p.M = static_cast<int>(m);
  p.N = static_cast<int>(n_out);
  p.Nw = static_cast<int>(nw);
  p.ldw = static_cast<int>(ldw.unwrap());
  p.ldx = static_cast<int>(ldx.unwrap());
  p.ldo = static_cast<int>(ldo.unwrap());
  p.nseg = static_cast<int>(nseg);
  p.tile_stride = tile_stride;

  p.n_tiles = static_cast<int>(tiles128 * 8);
  p.kw = static_cast<int>(kw);
  p.rw = static_cast<int>(rw);
  *device_out = dev.unwrap();
  return p;
}

}  // namespace host::nvfp4_w4a8_dec

#ifndef SGL_W4A8_DEC_FUSED
inline void nvfp4_w4a8_decode_gemm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView xq,
    tvm::ffi::TensorView xs,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView wscale,
    tvm::ffi::TensorView gscale,
    int64_t n_out,
    int64_t mode,
    int64_t kw,
    int64_t rw,
    int64_t u) {
  using namespace host::nvfp4_w4a8_dec;
  DLDevice device;
  const DecParams p = make_dec_params(out, xq, xs, weight, wscale, gscale, n_out, mode, kw, rw, u, &device);
  dispatch<false>(u, static_cast<int>(mode), p, device);
}
#endif  // !SGL_W4A8_DEC_FUSED

#ifdef SGL_W4A8_DEC_FUSED
// FUSED entry point (module nvfp4_w4a8_decode_fused_sm86 only): x bf16 [M, Kp] is quantised inside the GEMV into the
// workspace xq int8 [M, Kp] (layout of `mode`) / xs fp32 [M]; sem int32 [>= 3] is zero between launches (the kernel
// resets it) and must not be shared by launches that can run concurrently.
inline void nvfp4_w4a8_decode_fused_gemm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView x,
    tvm::ffi::TensorView xq,
    tvm::ffi::TensorView xs,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView wscale,
    tvm::ffi::TensorView gscale,
    tvm::ffi::TensorView sem,
    int64_t n_out,
    int64_t mode,
    int64_t kw,
    int64_t rw,
    int64_t u) {
  using namespace host;
  using namespace host::nvfp4_w4a8_dec;
  DLDevice device;
  FusedDecParams p{};
  static_cast<DecParams&>(p) = make_dec_params(out, xq, xs, weight, wscale, gscale, n_out, mode, kw, rw, u, &device);
  SymbolicDevice dev;
  dev.set_options<kDLCUDA>();
  SymbolicSize M{"M"}, K{"K"}, ldxb{"ldxb"}, S{"sem"};
  TensorMatcher({M, K}).with_strides({ldxb, 1}).with_dtype<bf16_t>().with_device(dev).verify(x);
  TensorMatcher({S}).with_dtype<int32_t>().with_device(dev).verify(sem);
  RuntimeCheck(M.unwrap() == p.M, "x has ", M.unwrap(), " rows, xq ", p.M);
  RuntimeCheck(K.unwrap() == static_cast<int64_t>(p.nseg) * 128, "x has ", K.unwrap(), " columns, needs Kp = ",
               static_cast<int64_t>(p.nseg) * 128, " (no K padding in the fused kernel)");
  RuntimeCheck(ldxb.unwrap() % 8 == 0, "x row stride must be a multiple of 8 elements");
  RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");
  RuntimeCheck(S.unwrap() >= 3, "sem needs >= 3 int32 counters");
  RuntimeCheck(reinterpret_cast<uintptr_t>(sem.data_ptr()) % 4 == 0, "sem must be 4-byte aligned");
  p.xb16 = static_cast<const __nv_bfloat16*>(x.data_ptr());
  p.ldxb = static_cast<int>(ldxb.unwrap());
  p.xq_w = static_cast<int8_t*>(xq.data_ptr());
  p.xs_w = static_cast<float*>(xs.data_ptr());
  p.sem = static_cast<unsigned int*>(sem.data_ptr());
  dispatch<true>(u, static_cast<int>(mode), p, device);
}
#endif  // SGL_W4A8_DEC_FUSED



#ifndef SGL_W4A8_DEC_FUSED
inline void nvfp4_w4a8_decode_quant(
    tvm::ffi::TensorView x, tvm::ffi::TensorView xq, tvm::ffi::TensorView xs, int64_t permuted) {
  using namespace host;
  SymbolicDevice dev;
  dev.set_options<kDLCUDA>();
  SymbolicSize M{"M"}, K{"K"}, ldx{"ldx"}, ldq{"ldq"};
  TensorMatcher({M, K}).with_strides({ldx, 1}).with_dtype<bf16_t>().with_device(dev).verify(x);
  TensorMatcher({M, K}).with_strides({ldq, 1}).with_dtype<int8_t>().with_device(dev).verify(xq);
  TensorMatcher({M}).with_dtype<float>().with_device(dev).verify(xs);
  RuntimeCheck(K.unwrap() % 128 == 0, "decode quantiser needs K % 128 == 0");
  RuntimeCheck(ldx.unwrap() % 8 == 0 && ldq.unwrap() % 16 == 0, "row strides must be vector aligned");
  RuntimeCheck(reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0, "x must be 16-byte aligned");
  RuntimeCheck(reinterpret_cast<uintptr_t>(xq.data_ptr()) % 16 == 0, "xq must be 16-byte aligned");
  if (M.unwrap() == 0) return;
  constexpr int kThreads = 256;
  const dim3 grid(static_cast<unsigned>(M.unwrap()));
  if (permuted) {
    LaunchKernel(grid, dim3(kThreads), dev.unwrap())(
        device::nvfp4_w4a8_dec::w4a8_dec_quant_kernel<kThreads, true>, static_cast<const __nv_bfloat16*>(x.data_ptr()),
        static_cast<int>(ldx.unwrap()), static_cast<int>(K.unwrap()), static_cast<int8_t*>(xq.data_ptr()),
        static_cast<int>(ldq.unwrap()), static_cast<float*>(xs.data_ptr()));
  } else {
    LaunchKernel(grid, dim3(kThreads), dev.unwrap())(
        device::nvfp4_w4a8_dec::w4a8_dec_quant_kernel<kThreads, false>,
        static_cast<const __nv_bfloat16*>(x.data_ptr()), static_cast<int>(ldx.unwrap()),
        static_cast<int>(K.unwrap()), static_cast<int8_t*>(xq.data_ptr()), static_cast<int>(ldq.unwrap()),
        static_cast<float*>(xs.data_ptr()));
  }
}
#endif  // !SGL_W4A8_DEC_FUSED
