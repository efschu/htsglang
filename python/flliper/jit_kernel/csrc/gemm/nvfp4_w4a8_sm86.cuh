// NVFP4 W4A8-g16 GEMM for sm_86 (RTX 3080): INT8 tensor cores reading the NATIVE NVFP4 byte layout.
//
// Backlog #38 (user order 2026-09-25): "die 5090 soll nativ nvfp4 nehmen und die 3080er die int8 kerne".
// The 5090 computes W4A4 on its FP4 tensor cores from the layout ModelOptFp4LinearMethod builds for sm_120
// (modelopt_quant.py process_weights_after_loading, native branch). This kernel reads EXACTLY those bytes,
// so a weight exchange at the flip moves bytes and never reshapes them:
//
//   weight          uint8 [Nw, Kp/2]   E2M1, element 2j in the LOW nibble of byte j, rows = output features.
//                                      Row stride may exceed Kp/2 (a K-shard view).
//   weight_scale    e4m3  128x4-swizzled (modelopt_quant.py: reshape(B, M/128, 4, 32, K/4, 4).permute(0,1,4,3,2,5)).
//                                      Byte address of (row n, scale column kb = k/16):
//                                        (n/128)*tile_row_stride + (kb/4)*512 + (n%32)*16 + ((n%128)/32)*4 + kb%4
//                                      tile_row_stride = Ks*128 for the contiguous [ceil128(N), Ks] tensor, larger
//                                      for a K-shard taken as a column slice of the tile view [N/128, Ks*128].
//   weight_scale_2  fp32 scalar (max over the fused partitions; exact for Qwen3.8 gate/up, see contract §0).
//
// Arithmetic (the user's formula, binding):
//   * E2M1 x 2 is exactly {0, +-1, +-2, +-3, +-4, +-6, +-8, +-12} -> INT8 by table, lossless (3 PRMT).
//   * activations: INT8 per token, symmetric, scale s_x[m] = amax/127 (same rounding as per_token_quant_int8).
//   * one mma.sync.m16n8k16.s32.s8.s8.s32 per 16-element K block -> exact INT32 block sum S, |S| <= 16*12*127 = 24384.
//   * INT32 -> FP32 for free: the mma accumulator input C is the magic 0x4B400000 (= 1.5*2^23), so the int32
//     result's bit pattern IS the float 1.5*2^23 + S (exact because |S| < 2^22).
//   * block scale: t = fma(f, s, -1.5*2^23*s) == s*S EXACTLY (s has <= 4 significant bits, S <= 15 bits, the
//     product and -1.5*2^23*s are both exactly representable, one rounding of an exact value); acc += t (FP32).
//   * the E4M3 scale is used as its raw bit pattern shifted into an FP32 (value * 2^-120, subnormals included,
//     FTZ must stay OFF); 2^120 / 2 (the x2 of the table) / s_x[m] / weight_scale_2 are folded into ONE factor
//     applied at the very end. Output bf16.
//
// Operand roles: the WEIGHT tile is the mma A operand (16 weight rows), the TOKENS are the B operand (8 tokens),
// so the decode path (M = 1..48) wastes nothing on a 16-row token tile.

#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.cuh>

#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace device::nvfp4_w4a8 {

constexpr int kBK = 64;             // K elements per pipeline stage = 4 NVFP4 blocks = one 128x4 scale tile column
constexpr int kBN = 128;            // weight rows per CTA = one 128-row scale tile
constexpr int kScaleTileBytes = 512;
constexpr uint32_t kMagicBits = 0x4B400000u;  // float 1.5 * 2^23
constexpr float kMagic = 12582912.0f;          // 1.5 * 2^23

struct GemmParams {
  const uint8_t* weight;   // [Nw, ldw bytes]
  const uint8_t* wscale;   // swizzled e4m3 bytes
  const int8_t* xq;        // [M, ldx]
  const float* xs;         // [M] per-token scale
  const float* gscale;     // [1] weight_scale_2
  __nv_bfloat16* out;      // [M, ldo]
  float* ws;               // split-K partials [splits, M, N] (raw accumulators), may be null when splits == 1
  int M, N, Nw;
  int ldw, ldx, ldo;
  int Kp;                  // padded K (elements) = 2 * weight.size(1)
  long long scale_tile_row_stride;  // bytes between two 128-row scale tiles
  int num_kt;              // ceil(Kp / 64)
  int kt_per_split;
  int splits;
};

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  const int bytes = pred ? 16 : 0;  // src-size 0 -> the 16 destination bytes are zero-filled
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// 4 packed E2M1 codes (low 16 bits, nibble i = element i) -> 4 int8 = 2 * e2m1 value (byte i = element i).
__device__ __forceinline__ uint32_t e2m1x4_to_s8x4(uint32_t x) {
  const uint32_t pos = __byte_perm(0x03020100u, 0x0C080604u, x);  // codes 0..7  -> 0,1,2,3,4,6,8,12
  const uint32_t neg = __byte_perm(0xFDFEFF00u, 0xF4F8FAFCu, x);  // codes 8..15 -> 0,-1,-2,-3,-4,-6,-8,-12
  const uint32_t sel = ((x & 0x8888u) >> 1) | 0x3210u;            // sign bit -> pick from `neg`
  return __byte_perm(pos, neg, sel);
}

// Byte kb of w (an E4M3 code) -> FP32 whose value is e4m3 * 2^-120 (exact for normals AND subnormals).
__device__ __forceinline__ float e4m3_scaled_bits(uint32_t w, int kb) {
  const uint32_t y = __byte_perm(w, 0u, 0x0444u | (static_cast<uint32_t>(kb) << 12));  // byte kb at bits 31..24
  const int z = static_cast<int>(y) >> 4;  // sign smeared into 31..27, exponent 26..23, mantissa 22..20
  return __int_as_float(z & static_cast<int>(0x87F00000u));
}

__device__ __forceinline__ void mma_s8_16816_magic(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a0), "r"(a1), "r"(b0), "r"(kMagicBits));
}

// C = {magic, magic, magic, 0}: d0..d2 come out as FP32 bit patterns, d3 as the plain INT32 block sum
__device__ __forceinline__ void mma_s8_16816_magic3(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%8};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a0), "r"(a1), "r"(b0), "r"(kMagicBits), "r"(0));
}

__device__ __forceinline__ void mma_s8_16816_acc(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0, const int (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a0), "r"(a1), "r"(b0), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

__device__ __forceinline__ void mma_s8_16816_zero(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
      : "r"(a0), "r"(a1), "r"(b0), "r"(0));
}

// acc += (int64) a * b, one IMAD.WIDE
__device__ __forceinline__ void mad_wide_s32(long long& acc, int a, int b) {
  asm("mad.wide.s32 %0, %1, %2, %0;\n" : "+l"(acc) : "r"(a), "r"(b));
}

// E4M3 code -> the exact integer e4m3 * 2^9 (|c| <= 448 * 512 = 229376). Normal: (8 + m) << (e - 1);
// subnormal (e = 0): m. The two NaN codes map to 0 (never produced by modelopt's quantiser).
__device__ __forceinline__ int e4m3_times_512(uint32_t b) {
  const int mag = static_cast<int>(b & 0x7Fu);
  const int e = mag >> 3, m = mag & 7;
  int c = (mag == 0x7F) ? 0 : (e ? ((8 | m) << (e - 1)) : m);
  return (b & 0x80u) ? -c : c;
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t& r0, uint32_t& r1, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x1(uint32_t& r0, uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n" : "=r"(r0) : "r"(addr));
}

// Shared-memory layout of one stage (all XOR swizzles are bank-conflict free for the access patterns below):
//   weights : kBN rows x 32 B; 16-B chunk c of row r at r*32 + ((c ^ ((r>>2)&1)) << 4)
//   tokens  : BM rows x 64 B;  16-B chunk c of row r at r*64 + ((c ^ ((r>>1)&3)) << 4)
//   scales  : the 512-B swizzled tile, verbatim
template <int WARPS_N, int WARPS_M, int MT, int NT, int STAGES>
struct Cfg {
  static constexpr int kThreads = WARPS_N * WARPS_M * 32;
  static constexpr int BM = WARPS_M * NT * 8;
  static constexpr int kWBytes = kBN * 32;
  static constexpr int kABytes = BM * 64;
  static constexpr int kStageBytes = kWBytes + kABytes + kScaleTileBytes;
  static constexpr int kSmemBytes = kStageBytes * STAGES + 256 * 4;  // + E4M3 -> int LUT (exact epilogue)
  static_assert(WARPS_N * MT * 16 == kBN, "a CTA covers exactly one 128-row scale tile");
};

// EPI 0: FP32 epilogue per block (magic-number INT32->FP32, t = s*S exact, FP32 accumulation) -- the order's recipe.
// EPI 2: as EPI 0, but one of the four accumulators per mma converts with I2F (conversion pipe) and needs a single
//        FFMA (acc = fma(float(S), s, acc): exact product, one rounding -- the same accuracy as EPI 0).
// EPI 1: EXACT integer epilogue: acc64 += S * (e4m3 * 2^9) with one IMAD.WIDE per element, no rounding at all
//        until the final int64 -> fp32 conversion (measured variant, see the probe in the report).
template <int WARPS_N, int WARPS_M, int MT, int NT, int STAGES, int MIN_BLOCKS = 1, int EPI = 0>
__global__ void __launch_bounds__(WARPS_N* WARPS_M * 32, MIN_BLOCKS) w4a8_nvfp4_gemm_kernel(const GemmParams p) {
  using C = Cfg<WARPS_N, WARPS_M, MT, NT, STAGES>;
  constexpr int BM = C::BM;
  extern __shared__ __align__(128) uint8_t smem[];

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wn = warp % WARPS_N;
  const int wm = warp / WARPS_N;
  const int g = lane >> 2;
  const int q = lane & 3;

  const int n_tile = blockIdx.x;
  const int n0 = n_tile * kBN;
  const int m0 = blockIdx.y * BM;
  const int split = blockIdx.z;
  const int kt_begin = split * p.kt_per_split;
  const int kt_end = min(p.num_kt, kt_begin + p.kt_per_split);

  const uint8_t* __restrict__ wsc_tile_row = p.wscale + static_cast<long long>(n_tile) * p.scale_tile_row_stride;

  auto load_stage = [&](int slot, int kt) {
    uint8_t* base = smem + slot * C::kStageBytes;
    // weights: kBN rows x 2 chunks of 16 B (32 K elements each)
    constexpr int kWChunks = kBN * 2;
#pragma unroll
    for (int i = 0; i < (kWChunks + C::kThreads - 1) / C::kThreads; ++i) {
      const int c = tid + i * C::kThreads;
      if (kWChunks % C::kThreads != 0 && c >= kWChunks) break;
      const int r = c >> 1, ch = c & 1;
      const int n = n0 + r;
      const int kbyte = kt * 32 + ch * 16;
      const bool ok = (n < p.Nw) && (kbyte < (p.Kp >> 1));
      const uint8_t* src = ok ? p.weight + static_cast<long long>(n) * p.ldw + kbyte : p.weight;
      cp_async16(smem_addr(base + r * 32 + ((ch ^ ((r >> 2) & 1)) << 4)), src, ok);
    }
    // tokens: BM rows x 4 chunks of 16 B (one NVFP4 block each)
    uint8_t* sa = base + C::kWBytes;
    constexpr int kAChunks = BM * 4;
#pragma unroll
    for (int i = 0; i < (kAChunks + C::kThreads - 1) / C::kThreads; ++i) {
      const int c = tid + i * C::kThreads;
      if (kAChunks % C::kThreads != 0 && c >= kAChunks) break;
      const int r = c >> 2, ch = c & 3;
      const int m = m0 + r;
      const int k = kt * kBK + ch * 16;
      const bool ok = (m < p.M) && (k < p.Kp);
      const int8_t* src = ok ? p.xq + static_cast<long long>(m) * p.ldx + k : p.xq;
      cp_async16(smem_addr(sa + r * 64 + ((ch ^ ((r >> 1) & 3)) << 4)), src, ok);
    }
    // scales: one 512-B tile (128 rows x 4 blocks), contiguous by construction of the swizzle
    uint8_t* ss = sa + C::kABytes;
    if (tid < kScaleTileBytes / 16) {
      cp_async16(smem_addr(ss + tid * 16), wsc_tile_row + kt * kScaleTileBytes + tid * 16, true);
    }
  };

  using Acc = std::conditional_t<EPI == 1, long long, float>;
  Acc acc[MT][NT][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][j][e] = Acc(0);

  int* lut = reinterpret_cast<int*>(smem + C::kStageBytes * STAGES);
  [[maybe_unused]] int dsave[EPI == 3 ? MT : 1][EPI == 3 ? NT : 1][4];
  if constexpr (EPI == 1) {
    for (int i = tid; i < 256; i += C::kThreads) lut[i] = e4m3_times_512(static_cast<uint32_t>(i));
    // made visible by the __syncthreads at the top of the first main-loop iteration
  }

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (kt_begin + s < kt_end) load_stage(s, kt_begin + s);
    cp_async_commit();
  }

  const int rb = wn * (MT * 16);  // first weight row of this warp inside the CTA tile
  const int tb = wm * (NT * 8);   // first token row of this warp inside the CTA tile
  const int wsw = (g >> 2) & 1;   // weight-row chunk swizzle bit, identical for rows g and g+8

  for (int kt = kt_begin, it = 0; kt < kt_end; ++kt, ++it) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    {
      const int kn = kt + STAGES - 1;
      if (kn < kt_end) load_stage((it + STAGES - 1) % STAGES, kn);
      cp_async_commit();
    }

    const uint8_t* sw = smem + (it % STAGES) * C::kStageBytes;
    const uint8_t* sa = sw + C::kWBytes;
    const uint8_t* ss = sa + C::kABytes;

    // the 4 block scales of each of this thread's 2*MT weight rows: one 32-bit word per row
    uint32_t swd[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = rb + mt * 16 + g + 8 * h;
        swd[mt][h] = *reinterpret_cast<const uint32_t*>(ss + (r & 31) * 16 + (r >> 5) * 4);
      }

#pragma unroll
    for (int kb = 0; kb < 4; ++kb) {
      uint32_t a[MT][2];
      float sc[MT][2], nc[MT][2];
      int ci[MT][2];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int h = 0; h < 2; ++h) {
          const int r = rb + mt * 16 + g + 8 * h;
          const int off = r * 32 + (((kb >> 1) ^ wsw) << 4) + (kb & 1) * 8 + 2 * q;
          a[mt][h] = e2m1x4_to_s8x4(*reinterpret_cast<const uint16_t*>(sw + off));
          if constexpr (EPI == 1) {
            ci[mt][h] = lut[(swd[mt][h] >> (8 * kb)) & 0xFFu];
          } else {
            sc[mt][h] = e4m3_scaled_bits(swd[mt][h], kb);
            nc[mt][h] = sc[mt][h] * -kMagic;  // exact: 3 * 2^22 * (<= 4 significant bits)
          }
        }

      uint32_t b[NT];
      {
        int nt = 0;
#pragma unroll
        for (; nt + 4 <= NT; nt += 4) {
          const int tr = tb + (nt + (lane >> 3)) * 8 + (lane & 7);
          ldmatrix_x4(b[nt], b[nt + 1], b[nt + 2], b[nt + 3], smem_addr(sa + tr * 64 + ((kb ^ ((tr >> 1) & 3)) << 4)));
        }
        if constexpr (NT % 4 >= 2) {
          const int tr = tb + (nt + ((lane >> 3) & 1)) * 8 + (lane & 7);
          ldmatrix_x2(b[nt], b[nt + 1], smem_addr(sa + tr * 64 + ((kb ^ ((tr >> 1) & 3)) << 4)));
          nt += 2;
        }
        if constexpr (NT % 2 == 1) {
          const int tr = tb + nt * 8 + (lane & 7);
          ldmatrix_x1(b[nt], smem_addr(sa + tr * 64 + ((kb ^ ((tr >> 1) & 3)) << 4)));
        }
      }

#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
          int d[4];
          if constexpr (EPI == 1) {
            mma_s8_16816_zero(d, a[mt][0], a[mt][1], b[nt]);
#pragma unroll
            for (int e = 0; e < 4; ++e) mad_wide_s32(acc[mt][nt][e], d[e], ci[mt][e >> 1]);
          } else if constexpr (EPI == 3) {
            // RATE PROBE for a group-32 format (e.g. INT4 g32): the INT32 sum runs over two k16 blocks, then ONE
            // epilogue per 32 K. NOT numerically valid for NVFP4 (g16 scales differ per block) -- speed only.
            if ((kb & 1) == 0) {
              mma_s8_16816_magic(dsave[mt][nt], a[mt][0], a[mt][1], b[nt]);
            } else {
              mma_s8_16816_acc(d, a[mt][0], a[mt][1], b[nt], dsave[mt][nt]);
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                acc[mt][nt][e] += __fmaf_rn(__int_as_float(d[e]), sc[mt][e >> 1], nc[mt][e >> 1]);
              }
            }
          } else if constexpr (EPI == 2) {
            mma_s8_16816_magic3(d, a[mt][0], a[mt][1], b[nt]);
#pragma unroll
            for (int e = 0; e < 3; ++e) {
              acc[mt][nt][e] += __fmaf_rn(__int_as_float(d[e]), sc[mt][e >> 1], nc[mt][e >> 1]);
            }
            acc[mt][nt][3] = __fmaf_rn(__int2float_rn(d[3]), sc[mt][1], acc[mt][nt][3]);
          } else {
            mma_s8_16816_magic(d, a[mt][0], a[mt][1], b[nt]);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              acc[mt][nt][e] += __fmaf_rn(__int_as_float(d[e]), sc[mt][e >> 1], nc[mt][e >> 1]);
            }
          }
        }
    }
  }
  cp_async_wait<0>();

  // epilogue. C fragment: d0,d1 -> weight row g, tokens 2q,2q+1; d2,d3 -> weight row g+8.
  const float gs = *p.gscale;
#pragma unroll
  for (int nt = 0; nt < NT; ++nt)
#pragma unroll
    for (int e1 = 0; e1 < 2; ++e1) {
      const int m = m0 + tb + nt * 8 + 2 * q + e1;
      if (m >= p.M) continue;
      if (p.splits == 1) {
        // EPI 0: x 2^120 (raw E4M3 bits) x 1/2 (the x2 of the INT8 table) = 2^119
        // EPI 1: x 2^-9 (integer scale) x 1/2 = 2^-10
        const float factor = p.xs[m] * gs * (EPI == 1 ? 9.765625e-4f : 6.6461399789245794e35f);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int n = n0 + rb + mt * 16 + g + 8 * h;
            if (n < p.N) {
              float v;
              if constexpr (EPI == 1) v = __ll2float_rn(acc[mt][nt][2 * h + e1]);
              else v = acc[mt][nt][2 * h + e1];
              p.out[static_cast<long long>(m) * p.ldo + n] = __float2bfloat16_rn(v * factor);
            }
          }
      } else {
        Acc* dst = reinterpret_cast<Acc*>(p.ws) + (static_cast<long long>(split) * p.M + m) * p.N;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int h = 0; h < 2; ++h) {
            const int n = n0 + rb + mt * 16 + g + 8 * h;
            if (n < p.N) dst[n] = acc[mt][nt][2 * h + e1];
          }
      }
    }
}

// split-K finish: fixed summation order over the splits (deterministic), then the same final factor.
template <int EPI>
__global__ void w4a8_nvfp4_splitk_reduce_kernel(
    const void* __restrict__ ws_raw,
    const float* __restrict__ xs,
    const float* __restrict__ gscale,
    __nv_bfloat16* __restrict__ out,
    int M,
    int N,
    int ldo,
    int splits) {
  const long long total = static_cast<long long>(M) * N;
  const float gs = *gscale;
  for (long long i = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<long long>(gridDim.x) * blockDim.x) {
    const int m = static_cast<int>(i / N);
    const int n = static_cast<int>(i - static_cast<long long>(m) * N);
    float v;
    if constexpr (EPI == 1) {
      const long long* ws = static_cast<const long long*>(ws_raw);
      long long s = 0;
      for (int z = 0; z < splits; ++z) s += ws[static_cast<long long>(z) * total + i];
      v = __ll2float_rn(s) * (xs[m] * gs * 9.765625e-4f);
    } else {
      const float* ws = static_cast<const float*>(ws_raw);
      float s = 0.f;
      for (int z = 0; z < splits; ++z) s += ws[static_cast<long long>(z) * total + i];
      v = s * (xs[m] * gs * 6.6461399789245794e35f);
    }
    out[static_cast<long long>(m) * ldo + n] = __float2bfloat16_rn(v);
  }
}

// bf16 [M, K] -> int8 [M, Kp] (zero padded K..Kp) + fp32 scale [M].
// Same arithmetic as sglang per_token_quant_int8 (int8_kernel.py): absmax = max(max|x|, 1e-10),
// scale = absmax / 127, q = round_half_away(x * (127 / absmax)).
template <int kThreads>
__global__ void __launch_bounds__(kThreads) w4a8_quant_act_kernel(
    const __nv_bfloat16* __restrict__ x, int ldx, int K, int8_t* __restrict__ xq, int ldq, int Kp, float* __restrict__ xs) {
  const int row = blockIdx.x;
  const __nv_bfloat16* xr = x + static_cast<long long>(row) * ldx;
  int8_t* qr = xq + static_cast<long long>(row) * ldq;
  const bool vec = ((K & 7) == 0) && ((ldx & 7) == 0) && ((reinterpret_cast<uintptr_t>(x) & 15) == 0);

  float amax = 0.f;
  if (vec) {
    for (int k = threadIdx.x * 8; k < K; k += kThreads * 8) {
      const uint4 v = *reinterpret_cast<const uint4*>(xr + k);
      const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h[j]);
        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
    }
  } else {
    for (int k = threadIdx.x; k < K; k += kThreads) amax = fmaxf(amax, fabsf(__bfloat162float(xr[k])));
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

  const bool qvec = vec && ((ldq & 7) == 0) && ((reinterpret_cast<uintptr_t>(xq) & 7) == 0);
  if (qvec) {
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
      *reinterpret_cast<uint2*>(qr + k) = make_uint2(w[0], w[1]);
    }
  } else {
    for (int k = threadIdx.x; k < K; k += kThreads)
      qr[k] = static_cast<int8_t>(static_cast<int>(roundf(__fmul_rn(__bfloat162float(xr[k]), inv))));
  }
  for (int k = K + threadIdx.x; k < Kp; k += kThreads) qr[k] = 0;
}

}  // namespace device::nvfp4_w4a8

namespace host::nvfp4_w4a8 {

using namespace device::nvfp4_w4a8;

template <int WARPS_N, int WARPS_M, int MT, int NT, int STAGES, int MIN_BLOCKS = 1, int EPI = 0>
inline void launch_cfg(const GemmParams& p, DLDevice device) {
  using C = Cfg<WARPS_N, WARPS_M, MT, NT, STAGES>;
  auto kernel = w4a8_nvfp4_gemm_kernel<WARPS_N, WARPS_M, MT, NT, STAGES, MIN_BLOCKS, EPI>;
  static bool attr_set = false;  // per instantiation; the attribute is per function, idempotent
  if (!attr_set) {
    RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmemBytes));
    attr_set = true;
  }
  const dim3 grid((p.Nw + kBN - 1) / kBN, (p.M + C::BM - 1) / C::BM, p.splits);
  host::LaunchKernel(grid, dim3(C::kThreads), device, C::kSmemBytes)(kernel, p);
}

// Tile variants, chosen by M (token count). id: 0 -> M<=8, 1 -> <=16, 2 -> <=32, 3 -> <=48, 4 -> <=64, 5 -> large.
inline int config_for_m(int M) {
  if (M <= 8) return 0;
  if (M <= 16) return 1;
  if (M <= 32) return 2;
  if (M <= 48) return 3;
  if (M <= 64) return 4;
  return 5;
}
inline int config_bm(int cfg) {
  constexpr int bm[23] = {8, 16, 32, 48, 64, 128, 128, 64, 8, 16, 32, 48, 64, 128, 64, 8, 16, 128, 64, 256, 128, 128, 256};
  return bm[cfg];
}

inline void dispatch(int cfg, const GemmParams& p, DLDevice device) {
  switch (cfg) {
    case 0: return launch_cfg<4, 1, 2, 1, 6>(p, device);
    case 1: return launch_cfg<4, 1, 2, 2, 6>(p, device);
    case 2: return launch_cfg<4, 1, 2, 4, 5>(p, device);
    case 3: return launch_cfg<4, 1, 2, 6, 4>(p, device);
    case 4: return launch_cfg<4, 1, 2, 8, 4>(p, device);
    case 5: return launch_cfg<4, 2, 2, 8, 4>(p, device);
    case 6: return launch_cfg<4, 2, 2, 8, 3, 2>(p, device);  // 128 regs, 2 CTAs / SM
    case 7: return launch_cfg<4, 1, 2, 8, 4, 2>(p, device);  // BM=64, 2 CTAs / SM
    // exact INT64 epilogue (EPI 1): 2 registers per accumulator -> smaller token tiles
    case 8: return launch_cfg<4, 1, 2, 1, 6, 1, 1>(p, device);   // M <= 8
    case 9: return launch_cfg<4, 1, 2, 2, 6, 1, 1>(p, device);   // M <= 16
    case 10: return launch_cfg<4, 1, 2, 4, 5, 1, 1>(p, device);  // M <= 32
    case 11: return launch_cfg<4, 1, 2, 6, 4, 1, 1>(p, device);  // M <= 48
    case 12: return launch_cfg<4, 2, 2, 4, 4, 1, 1>(p, device);  // BM=64, 256 threads
    case 13: return launch_cfg<4, 4, 2, 4, 3, 1, 1>(p, device);  // BM=128, 512 threads
    case 14: return launch_cfg<4, 2, 2, 4, 3, 2, 1>(p, device);  // BM=64, 2 CTAs / SM
    case 15: return launch_cfg<4, 1, 2, 1, 4>(p, device);        // M <= 8, 4 stages (4 CTAs / SM)
    case 16: return launch_cfg<4, 1, 2, 2, 4>(p, device);        // M <= 16, 4 stages
    // EPI 2 (one I2F per mma) on the large tiles
    case 17: return launch_cfg<4, 2, 2, 8, 4, 1, 2>(p, device);  // BM=128
    case 18: return launch_cfg<4, 1, 2, 8, 4, 1, 2>(p, device);  // BM=64
    // larger warp tiles (64 weight rows x 64 tokens per warp, 128 FP32 accumulators), EPI 0
    case 19: return launch_cfg<2, 4, 4, 8, 3>(p, device);        // BM=256, 8 warps
    case 20: return launch_cfg<2, 2, 4, 8, 3>(p, device);        // BM=128, 4 warps
    // g32 RATE PROBE (EPI 3, numerically invalid for NVFP4)
    case 21: return launch_cfg<4, 2, 2, 8, 4, 1, 3>(p, device);  // BM=128
    default: return launch_cfg<2, 4, 4, 8, 3, 1, 3>(p, device);  // 22: BM=256
  }
}

inline bool config_is_exact(int cfg) {
  return cfg >= 8 && cfg <= 14;
}

}  // namespace host::nvfp4_w4a8

// ---------------------------------------------------------------------------------------------------------------
// FFI entry points
// ---------------------------------------------------------------------------------------------------------------

// out bf16 [M, >=N] (row stride ldo), xq int8 [M, Kp] (row stride), xs fp32 [M], weight u8 [Nw, Kp/2] (row stride),
// wscale e4m3 (or u8) 2-D: either [ceil128(Nw), Ks] contiguous, or the tile view [ceil128(Nw)/128, Ks*128] with
// stride(1) == 1, gscale fp32 [1] (= weight_scale_2), workspace fp32 [>= splits*M*N] (only read when splits > 1).
// n_out: the unpadded output width N. splits: >= 1. config: -1 = choose by M.
inline void nvfp4_w4a8_gemm(
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView xq,
    tvm::ffi::TensorView xs,
    tvm::ffi::TensorView weight,
    tvm::ffi::TensorView wscale,
    tvm::ffi::TensorView gscale,
    tvm::ffi::TensorView workspace,
    int64_t n_out,
    int64_t splits,
    int64_t config) {
  using namespace host;
  using namespace host::nvfp4_w4a8;

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
  RuntimeCheck(kp % 32 == 0, "Kp must be a multiple of 32 (pad_nvfp4_weight), got ", kp);
  RuntimeCheck(n_out > 0 && n_out <= nw && n_out <= NO.unwrap(), "n_out out of range: ", n_out);
  RuntimeCheck(ldx.unwrap() % 16 == 0 && ldw.unwrap() % 16 == 0, "row strides must be 16-byte aligned");
  RuntimeCheck(reinterpret_cast<uintptr_t>(xq.data_ptr()) % 16 == 0, "xq must be 16-byte aligned");
  RuntimeCheck(reinterpret_cast<uintptr_t>(weight.data_ptr()) % 16 == 0, "weight must be 16-byte aligned");
  RuntimeCheck(wscale.ndim() == 2 && wscale.stride(1) == 1, "weight_scale must be 2-D with unit inner stride");
  RuntimeCheck(wscale.dtype().bits == 8, "weight_scale must be an 8-bit (e4m3) tensor");
  RuntimeCheck(reinterpret_cast<uintptr_t>(wscale.data_ptr()) % 16 == 0, "weight_scale must be 16-byte aligned");

  const int64_t num_kt = (kp + kBK - 1) / kBK;
  const int64_t n_tiles = (nw + kBN - 1) / kBN;
  int64_t tile_row_stride = 0;
  if (wscale.size(0) == n_tiles * kBN) {  // [ceil128(N), Ks] swizzled tensor (contiguous rows of Ks bytes)
    RuntimeCheck(wscale.stride(0) == wscale.size(1), "a [N_pad, Ks] weight_scale must be contiguous");
    RuntimeCheck(wscale.size(1) % 4 == 0 && wscale.size(1) / 4 >= num_kt, "weight_scale has ", wscale.size(1),
                 " columns, needs a multiple of 4 covering ", num_kt * 4);
    tile_row_stride = wscale.size(1) * kBN;
  } else {  // tile view [N_pad/128, Ks*128], e.g. a K-shard column slice
    RuntimeCheck(wscale.size(0) == n_tiles, "weight_scale rows: expected ", n_tiles * kBN, " or ", n_tiles, " got ",
                 wscale.size(0));
    RuntimeCheck(wscale.size(1) % kScaleTileBytes == 0 && wscale.size(1) / kScaleTileBytes >= num_kt,
                 "weight_scale tile view has ", wscale.size(1), " columns, needs ", num_kt * kScaleTileBytes);
    RuntimeCheck(wscale.stride(0) % 16 == 0, "weight_scale tile-row stride must be 16-byte aligned");
    tile_row_stride = wscale.stride(0);
  }

  RuntimeCheck(splits >= 1 && splits <= num_kt, "splits must be in [1, ", num_kt, "], got ", splits);
  const int cfg = config < 0 ? config_for_m(static_cast<int>(m)) : static_cast<int>(config);
  RuntimeCheck(cfg >= 0 && cfg <= 22, "config id out of range");
  const int64_t kt_per_split = (num_kt + splits - 1) / splits;
  const int64_t eff_splits = (num_kt + kt_per_split - 1) / kt_per_split;  // no empty split

  GemmParams p{};
  p.weight = static_cast<const uint8_t*>(weight.data_ptr());
  p.wscale = static_cast<const uint8_t*>(wscale.data_ptr());
  p.xq = static_cast<const int8_t*>(xq.data_ptr());
  p.xs = static_cast<const float*>(xs.data_ptr());
  p.gscale = static_cast<const float*>(gscale.data_ptr());
  p.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  p.ws = nullptr;
  p.M = static_cast<int>(m);
  p.N = static_cast<int>(n_out);
  p.Nw = static_cast<int>(nw);
  p.ldw = static_cast<int>(ldw.unwrap());
  p.ldx = static_cast<int>(ldx.unwrap());
  p.ldo = static_cast<int>(ldo.unwrap());
  p.Kp = static_cast<int>(kp);
  p.scale_tile_row_stride = tile_row_stride;
  p.num_kt = static_cast<int>(num_kt);
  p.kt_per_split = static_cast<int>(kt_per_split);
  p.splits = static_cast<int>(eff_splits);
  if (m == 0) return;

  const DLDevice device = dev.unwrap();
  const bool exact = config_is_exact(cfg);
  if (eff_splits > 1) {
    const int64_t need_bytes = eff_splits * m * n_out * (exact ? 8 : 4);
    RuntimeCheck(workspace.numel() * (workspace.dtype().bits / 8) >= need_bytes, "workspace too small: need ",
                 need_bytes, " bytes");
    RuntimeCheck(reinterpret_cast<uintptr_t>(workspace.data_ptr()) % 8 == 0, "workspace must be 8-byte aligned");
    p.ws = static_cast<float*>(workspace.data_ptr());
  }
  dispatch(cfg, p, device);
  if (eff_splits > 1) {
    const int64_t total = m * n_out;
    const int threads = 256;
    const int64_t blocks = std::min<int64_t>((total + threads - 1) / threads, 68 * 16);
    if (exact) {
      LaunchKernel(dim3(static_cast<unsigned>(blocks)), dim3(threads), device)(
          w4a8_nvfp4_splitk_reduce_kernel<1>, static_cast<const void*>(p.ws), p.xs, p.gscale, p.out, p.M, p.N,
          p.ldo, p.splits);
    } else {
      LaunchKernel(dim3(static_cast<unsigned>(blocks)), dim3(threads), device)(
          w4a8_nvfp4_splitk_reduce_kernel<0>, static_cast<const void*>(p.ws), p.xs, p.gscale, p.out, p.M, p.N,
          p.ldo, p.splits);
    }
  }
}

inline void nvfp4_w4a8_quant_act(tvm::ffi::TensorView x, tvm::ffi::TensorView xq, tvm::ffi::TensorView xs) {
  using namespace host;
  SymbolicDevice dev;
  dev.set_options<kDLCUDA>();
  SymbolicSize M{"M"}, K{"K"}, Kp{"Kp"}, ldx{"ldx"}, ldq{"ldq"};
  TensorMatcher({M, K}).with_strides({ldx, 1}).with_dtype<bf16_t>().with_device(dev).verify(x);
  TensorMatcher({M, Kp}).with_strides({ldq, 1}).with_dtype<int8_t>().with_device(dev).verify(xq);
  TensorMatcher({M}).with_dtype<float>().with_device(dev).verify(xs);
  RuntimeCheck(Kp.unwrap() >= K.unwrap(), "xq narrower than x");
  if (M.unwrap() == 0) return;
  constexpr int kThreads = 256;
  LaunchKernel(dim3(static_cast<unsigned>(M.unwrap())), dim3(kThreads), dev.unwrap())(
      device::nvfp4_w4a8::w4a8_quant_act_kernel<kThreads>,
      static_cast<const __nv_bfloat16*>(x.data_ptr()),
      static_cast<int>(ldx.unwrap()),
      static_cast<int>(K.unwrap()),
      static_cast<int8_t*>(xq.data_ptr()),
      static_cast<int>(ldq.unwrap()),
      static_cast<int>(Kp.unwrap()),
      static_cast<float*>(xs.data_ptr()));
}
