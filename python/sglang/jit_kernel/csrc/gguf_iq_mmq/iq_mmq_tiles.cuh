// GGUF-NF G6 (2026-10-09): IQ-type MMQ / MoE-MMQ kernels for GGUF I-quants, vendored as NEW JIT sources.
// Source: sgl-project/sglang PR #36122 "[CUDA] Add dense and MoE GGUF MMQ kernels for eight I-quant types"
//   (open, unreviewed; head 0a39ec3a8754e3c4426950a5a93046c048a03f07, base 6e7beace143964386ce2e4418401b7369f0dabd0;
//   diff sha256 cf93506bd6b55eca441794007b1d5e92f6a66bf5f0524f288f460f1418bcee38),
//   itself adapted from https://github.com/aimbit-ni/vllm/commit/037e1d547c15313aa7e2bb5fc83390ab0c857314 and
//   ggml-org/llama.cpp PR #8495 (MIT, (c) 2023-2024 The ggml authors).
// Existing sgl-kernel/csrc/quantization/gguf/* is NOT touched (sha256-pinned by test_gguf_iq_mmq_1009.py).
//
// I-quant tile loaders and dot products: the PR #36122 addition to vecdotq.cuh, with three documented edits:
//   (1) every load_tiles_iq* takes `blocks_left` (see iq_mmq_common.cuh); only IQ4_NL uses it (K = 128 * n);
//   (2) the #28784 (ggml-org/llama.cpp, nvcc 13.2.0/13.2.1 sm_120 miscompile, issue #28581) byte extraction through
//       __byte_perm for the grid indices of IQ1_S / IQ2_S / IQ3_S;
//   (3) none of the arithmetic is changed.
#pragma once

#include "iq_mmq_common.cuh"

// I-quant MMQ tile operations adapted from
// https://github.com/aimbit-ni/vllm/commit/037e1d547c15313aa7e2bb5fc83390ab0c857314
// and https://github.com/ggml-org/llama.cpp/pull/8495.
#if !defined(USE_MUSA)
static __device__ __forceinline__ int iq_mmq_dp4a(const int a, const int b, int c) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610) || defined(USE_ROCM)
  return __dp4a(a, b, c);
#else
  const int8_t* va = reinterpret_cast<const int8_t*>(&a);
  const int8_t* vb = reinterpret_cast<const int8_t*>(&b);
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    c += va[i] * vb[i];
  }
  return c;
#endif
}

// ========================= IQ4_NL MMQ =========================
// IQ4_NL is structurally identical to Q4_0 (QK=32, QR=2, QI=4) but uses
// a 16-entry lookup table (kvalues_iq4nl) to convert 4-bit indices to signed int8.
// Tiles store raw packed nibble data (same as Q4_0); the lookup is applied in vec_dot.
// need_sum=true keeps the q8_1 (d, sum) tile layout; signed lookup values use only d.

#define VDR_IQ4_NL_Q8_1_MMQ 4

template <int mmq_y>
static __device__ __forceinline__ void allocate_tiles_iq4_nl(int** x_ql, half2** x_dm, int** x_qh, int** x_sc) {
  __shared__ int tile_x_qs[mmq_y * (WARP_SIZE_GGUF) + mmq_y];
  __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF / QI4_NL) + mmq_y / QI4_NL];

  *x_ql = tile_x_qs;
  *x_dm = (half2*)tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles_iq4_nl(
    const void* __restrict__ vx,
    int* __restrict__ x_ql,
    half2* __restrict__ x_dm,
    int* __restrict__ x_qh,
    int* __restrict__ x_sc,
    const int& i_offset,
    const int& i_max,
    const int& k,
    const int& blocks_per_row,
    const int& blocks_left) {
  const int kbx = k / QI4_NL;
  const int kqsx = k % QI4_NL;

  const block_iq4_nl* bx0 = (const block_iq4_nl*)vx;
  float* x_dmf = (float*)x_dm;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
    int i = i0 + i_offset;
    if (need_check) {
      i = min(i, i_max);
    }
    // G6 edit: the tile holds 8 blocks per K window; for K = 128 * n the last window has only 4 valid blocks. Blocks past
    // the row end are loaded as ZERO (never read from memory) and their scale is 0, so they contribute exactly 0.
    if (kbx < blocks_left) {
      const block_iq4_nl* bxi = bx0 + i * blocks_per_row + kbx;
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    } else {
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = 0;
    }
  }

  const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI4_NL;
  const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI4_NL) {
    int i = i0 + i_offset * QI4_NL + k / blocks_per_tile_x_row;
    if (need_check) {
      i = min(i, i_max);
    }
    // G6 edit: see above (zero scale for blocks past the row end)
    float d_val = 0.0f;
    if (kbxd < blocks_left) {
      const block_iq4_nl* bxi = bx0 + i * blocks_per_row + kbxd;
      d_val = __half2float(bxi->d);
    }
    x_dmf[i * (WARP_SIZE_GGUF / QI4_NL) + i / QI4_NL + kbxd] = d_val;
  }
}

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1_mul_mat(
    const int* __restrict__ x_ql,
    const half2* __restrict__ x_dm,
    const int* __restrict__ x_qh,
    const int* __restrict__ x_sc,
    const int* __restrict__ y_qs,
    const half2* __restrict__ y_ds,
    const int& i,
    const int& j,
    const int& k) {
  (void)x_qh;
  (void)x_sc;

  const int kyqs = k % (QI8_1 / 2) + QI8_1 * (k / (QI8_1 / 2));
  const float* x_dmf = (const float*)x_dm;

  int u[2 * VDR_IQ4_NL_Q8_1_MMQ];

#pragma unroll
  for (int l = 0; l < VDR_IQ4_NL_Q8_1_MMQ; ++l) {
    u[2 * l + 0] = y_qs[j * WARP_SIZE_GGUF + (kyqs + l) % WARP_SIZE_GGUF];
    u[2 * l + 1] = y_qs[j * WARP_SIZE_GGUF + (kyqs + l + QI4_NL) % WARP_SIZE_GGUF];
  }

  int sumi = 0;

  // IQ4_NL lookup table values (same as kvalues_iq4nl in ggml-common.h).
  const int8_t iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

#pragma unroll
  for (int l = 0; l < VDR_IQ4_NL_Q8_1_MMQ; ++l) {
    int v1, v2;
    get_int_from_table_16(x_ql[i * (WARP_SIZE_GGUF + 1) + k + l], (const uint8_t*)iq4nl, v1, v2);
    sumi = iq_mmq_dp4a(v1, u[2 * l + 0], sumi);
    sumi = iq_mmq_dp4a(v2, u[2 * l + 1], sumi);
  }

  // need_sum=true: y_ds is stored as half2 (d, sum). We only need the d component
  // since IQ4_NL values are already signed (no unsigned-to-signed bias correction).
  const float2 ds8 = __half22float2(y_ds[j * (WARP_SIZE_GGUF / QI8_1) + (2 * k / QI8_1) % (WARP_SIZE_GGUF / QI8_1)]);

  return x_dmf[i * (WARP_SIZE_GGUF / QI4_NL) + i / QI4_NL + k / QI4_NL] * ds8.x * sumi;
}

// ========================= IQ4_XS MMQ =========================
// IQ4_XS is a 256-element super-block (QK_K=256) with 8 sub-blocks of 32 elements.
// Each sub-block has a 6-bit scale: 4 low bits from scales_l + 2 high bits from scales_h.
// For MMQ, we use Q4_K-like tiling: QR=2, QI=32 (blocks_per_warp=1).
// This ensures all 128 bytes of qs are loaded into shared memory per step.
// The MMVQ constants (QR4_XS=8, QI4_XS=8) are NOT used for MMQ.

#define QR_IQ4_XS_MMQ 2
#define QI_IQ4_XS_MMQ (QK_K / (4 * QR_IQ4_XS_MMQ))
#define VDR_IQ4_XS_Q8_1_MMQ 4

template <int mmq_y>
static __device__ __forceinline__ void allocate_tiles_iq4_xs(int** x_ql, half2** x_dm, int** x_qh, int** x_sc) {
  __shared__ int tile_x_ql[mmq_y * (WARP_SIZE_GGUF) + mmq_y];
  __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF / QI_IQ4_XS_MMQ) + mmq_y / QI_IQ4_XS_MMQ];
  __shared__ int tile_x_sc[mmq_y * (WARP_SIZE_GGUF / 8) + mmq_y / 8];

  *x_ql = tile_x_ql;
  *x_dm = (half2*)tile_x_d;
  *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles_iq4_xs(
    const void* __restrict__ vx,
    int* __restrict__ x_ql,
    half2* __restrict__ x_dm,
    int* __restrict__ x_qh,
    int* __restrict__ x_sc,
    const int& i_offset,
    const int& i_max,
    const int& k,
    const int& blocks_per_row,
    const int& blocks_left) {
  const int kbx = k / QI_IQ4_XS_MMQ;
  const int kqsx = k % QI_IQ4_XS_MMQ;

  const block_iq4_xs* bx0 = (const block_iq4_xs*)vx;
  float* x_dmf = (float*)x_dm;

  // Load qs data (packed nibbles, 128 bytes per block)
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
    int i = i0 + i_offset;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq4_xs* bxi = bx0 + i * blocks_per_row + kbx;
    x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8_aligned(bxi->qs, kqsx);
  }

  // Load global d scale (1 float per block)
  const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ4_XS_MMQ;
  const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ4_XS_MMQ) {
    int i = (i0 + i_offset * QI_IQ4_XS_MMQ + k / blocks_per_tile_x_row) % mmq_y;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq4_xs* bxi = bx0 + i * blocks_per_row + kbxd;
    x_dmf[i * (WARP_SIZE_GGUF / QI_IQ4_XS_MMQ) + i / QI_IQ4_XS_MMQ + kbxd] = __half2float(bxi->d);
  }

  // Load sub-block scales (8 per block, decoded to int8, packed 4 per int32)
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
    int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF / 8)) % mmq_y;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq4_xs* bxi = bx0 + i * blocks_per_row + (k % (WARP_SIZE_GGUF / 8)) / (QI_IQ4_XS_MMQ / 8);

    const int ksc = k % (WARP_SIZE_GGUF / 8);

    // Pack 4 decoded sub-block scales per int32 (only need ksc=0,1 for 8 scales)
    if (ksc < 2) {
      int scales_packed = 0;
#pragma unroll
      for (int sb = 0; sb < 4; sb++) {
        const int ib32 = ksc * 4 + sb;
        const int8_t ls =
            ((bxi->scales_l[ib32 / 2] >> (4 * (ib32 % 2))) & 0xf) | (((bxi->scales_h >> (2 * ib32)) & 3) << 4);
        ((int8_t*)&scales_packed)[sb] = ls - 32;
      }
      x_sc[i * (WARP_SIZE_GGUF / 8) + i / 8 + ksc] = scales_packed;
    }
  }
}

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1_mul_mat(
    const int* __restrict__ x_ql,
    const half2* __restrict__ x_dm,
    const int* __restrict__ x_qh,
    const int* __restrict__ x_sc,
    const int* __restrict__ y_qs,
    const half2* __restrict__ y_ds,
    const int& i,
    const int& j,
    const int& k) {
  (void)x_qh;

  const float* x_dmf = (const float*)x_dm;

  // IQ4_NL lookup table values (same as kvalues_iq4nl in ggml-common.h).
  const int8_t iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

  // Sub-block index: each sub-block = 4 int32 of packed nibble data = 32 elements
  const int ib32 = k / 4;
  const int8_t* sc_bytes = (const int8_t*)&x_sc[i * (WARP_SIZE_GGUF / 8) + i / 8 + ib32 / 4];
  const float sub_scale = (float)sc_bytes[ib32 % 4];

  const int index_y = j * WARP_SIZE_GGUF + (QR_IQ4_XS_MMQ * k) % WARP_SIZE_GGUF;

  int sumi = 0;

#pragma unroll
  for (int l = 0; l < VDR_IQ4_XS_Q8_1_MMQ; ++l) {
    int v1, v2;
    get_int_from_table_16(x_ql[i * (WARP_SIZE_GGUF + 1) + k + l], (const uint8_t*)iq4nl, v1, v2);
    sumi = iq_mmq_dp4a(v1, y_qs[index_y + l], sumi);
    sumi = iq_mmq_dp4a(v2, y_qs[index_y + l + 4], sumi);
  }

  // need_sum=true: y_ds is stored as half2 (d, sum). We only need the d component
  // since IQ4_NL values are already signed (no unsigned-to-signed bias correction).
  const float2 ds8 =
      __half22float2(y_ds[j * (WARP_SIZE_GGUF / QI8_1) + (QR_IQ4_XS_MMQ * k / QI8_1) % (WARP_SIZE_GGUF / QI8_1)]);

  return x_dmf[i * (WARP_SIZE_GGUF / QI_IQ4_XS_MMQ) + i / QI_IQ4_XS_MMQ] * sub_scale * ds8.x * sumi;
}

// ========================= IQ3_S MMQ =========================
// IQ3_S: QK_K=256, qs[64], qh[8], signs[32], scales[4], d(half).
// Custom QR=2, QI=32 for MMQ (1 block/warp, like IQ4_XS_MMQ).
// tile_x_ql[0..15]: qs (64 bytes), tile_x_ql[16..23]: signs (32 bytes).
// tile_x_qh: qh (8 bytes). tile_x_sc: scales (4 bytes). tile_x_dm: d.
// Grid lookup (iq3xs_grid[512]) in vec_dot.

#define QR_IQ3_S_MMQ 2
#define QI_IQ3_S_MMQ (QK_K / (4 * QR_IQ3_S_MMQ))
#define VDR_IQ3_S_Q8_1_MMQ 4

template <int mmq_y>
static __device__ __forceinline__ void allocate_tiles_iq3_s(int** x_ql, half2** x_dm, int** x_qh, int** x_sc) {
  __shared__ int tile_x_ql[mmq_y * (WARP_SIZE_GGUF) + mmq_y];
  __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF / QI_IQ3_S_MMQ) + mmq_y / QI_IQ3_S_MMQ];
  __shared__ int tile_x_qh[mmq_y * (WARP_SIZE_GGUF / 4) + mmq_y / 4];
  __shared__ int tile_x_sc[mmq_y * (WARP_SIZE_GGUF / 8) + mmq_y / 8];

  *x_ql = tile_x_ql;
  *x_dm = (half2*)tile_x_d;
  *x_qh = tile_x_qh;
  *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles_iq3_s(
    const void* __restrict__ vx,
    int* __restrict__ x_ql,
    half2* __restrict__ x_dm,
    int* __restrict__ x_qh,
    int* __restrict__ x_sc,
    const int& i_offset,
    const int& i_max,
    const int& k,
    const int& blocks_per_row,
    const int& blocks_left) {
  const int kbx = k / QI_IQ3_S_MMQ;   // 0 (1 block per warp)
  const int kqsx = k % QI_IQ3_S_MMQ;  // 0..31

  const block_iq3_s* bx0 = (const block_iq3_s*)vx;
  float* x_dmf = (float*)x_dm;

  // Load qs (64 bytes = 16 int32, slots 0..15) and signs (32 bytes = 8 int32, slots 16..23)
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
    int i = i0 + i_offset;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq3_s* bxi = bx0 + i * blocks_per_row + kbx;
    if (kqsx < 16) {
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    } else if (kqsx < 24) {
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->signs, kqsx - 16);
    } else {
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = 0;
    }
  }

  // Load d scale
  const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ3_S_MMQ;
  const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ3_S_MMQ) {
    int i = (i0 + i_offset * QI_IQ3_S_MMQ + k / blocks_per_tile_x_row) % mmq_y;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq3_s* bxi = bx0 + i * blocks_per_row + kbxd;
    x_dmf[i * (WARP_SIZE_GGUF / QI_IQ3_S_MMQ) + i / QI_IQ3_S_MMQ + kbxd] = __half2float(bxi->d);
  }

  // Load qh (8 bytes = 2 int32)
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
    int i = (i0 + i_offset * 4 + k / (WARP_SIZE_GGUF / 4)) % mmq_y;
    if (need_check) {
      i = min(i, i_max);
    }
    const int kqh = k % (WARP_SIZE_GGUF / 4);
    if (kqh < 2) {
      const block_iq3_s* bxi = bx0 + i * blocks_per_row + kbx;
      x_qh[i * (WARP_SIZE_GGUF / 4) + i / 4 + kqh] = get_int_from_uint8(bxi->qh, kqh);
    }
  }

  // Load scales (4 bytes = 1 int32)
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
    int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF / 8)) % mmq_y;
    if (need_check) {
      i = min(i, i_max);
    }
    const int ksc = k % (WARP_SIZE_GGUF / 8);
    if (ksc < 1) {
      const block_iq3_s* bxi = bx0 + i * blocks_per_row + kbx;
      x_sc[i * (WARP_SIZE_GGUF / 8) + i / 8 + ksc] = get_int_from_uint8(bxi->scales, ksc);
    }
  }
}

static __device__ __forceinline__ float vec_dot_iq3_s_q8_1_mul_mat(
    const int* __restrict__ x_ql,
    const half2* __restrict__ x_dm,
    const int* __restrict__ x_qh,
    const int* __restrict__ x_sc,
    const int* __restrict__ y_qs,
    const half2* __restrict__ y_ds,
    const int& i,
    const int& j,
    const int& k) {
  const float* x_dmf = (const float*)x_dm;
  const int ib32 = k / 4;  // sub-block index (same for all VDR iterations since k is multiple of 4)

  // Read qs base for this ib32 (8 bytes of grid indices)
  const int qs_w0 = x_ql[i * (WARP_SIZE_GGUF + 1) + ib32 * 2 + 0];
  const int qs_w1 = x_ql[i * (WARP_SIZE_GGUF + 1) + ib32 * 2 + 1];
  // Read qh byte for this ib32
  const uint8_t* qh_all = (const uint8_t*)&x_qh[i * (WARP_SIZE_GGUF / 4) + i / 4];
  const uint8_t qh_val = qh_all[ib32];
  // Signs base
  const uint8_t* all_signs = (const uint8_t*)&x_ql[i * (WARP_SIZE_GGUF + 1) + 16];
  // Per-sub-block scale
  const uint8_t* sc_bytes = (const uint8_t*)&x_sc[i * (WARP_SIZE_GGUF / 8) + i / 8];
  const float sub_scale = 0.5f + ((sc_bytes[ib32 / 2] >> (4 * (ib32 % 2))) & 0xf);

  int sumi = 0;
#pragma unroll
  for (int l = 0; l < VDR_IQ3_S_Q8_1_MMQ; ++l) {
    const int sub = l;  // k%4=0 always, so (k+l)%4 = l

    // Grid lookup: 9-bit index
    // #28784 analogue: bytes via __byte_perm (see IQ1_S above); qs bytes 2*sub, 2*sub+1 of the 8 bytes of this ib32
    const int qs_word = (sub < 2) ? qs_w0 : qs_w1;
    const int grid1_val = iq3xs_grid[__byte_perm(qs_word, 0, 0x4440 | ((2 * sub) & 3)) | ((qh_val << (8 - 2 * sub)) & 256)];
    const int grid2_val = iq3xs_grid[__byte_perm(qs_word, 0, 0x4440 | ((2 * sub + 1) & 3)) | ((qh_val << (7 - 2 * sub)) & 256)];

    // Sign extraction
    const uint8_t sign_byte = all_signs[4 * ib32 + sub];
    const uint32_t signs0 = __vcmpeq4(((sign_byte & 0xf) * 0x01010101) & 0x08040201, 0x08040201);
    const uint32_t signs1 = __vcmpeq4(((sign_byte >> 4) * 0x01010101) & 0x08040201, 0x08040201);
    const int grid_l = __vsub4(grid1_val ^ signs0, signs0);
    const int grid_h = __vsub4(grid2_val ^ signs1, signs1);

    const int index_y = j * WARP_SIZE_GGUF + (QR_IQ3_S_MMQ * (k + l)) % WARP_SIZE_GGUF;
    sumi = iq_mmq_dp4a(grid_l, y_qs[index_y + 0], sumi);
    sumi = iq_mmq_dp4a(grid_h, y_qs[index_y + 1], sumi);
  }

  const float2 ds8 =
      __half22float2(y_ds[j * (WARP_SIZE_GGUF / QI8_1) + (QR_IQ3_S_MMQ * k / QI8_1) % (WARP_SIZE_GGUF / QI8_1)]);

  return x_dmf[i * (WARP_SIZE_GGUF / QI_IQ3_S_MMQ) + i / QI_IQ3_S_MMQ] * sub_scale * 0.5f * ds8.x * sumi;
}

// ========================= IQ3_XXS MMQ =========================
// IQ3_XXS: QK_K=256, qs[3*(QK_K/8)=96 bytes]: first 64 = grid indices, last 32 = gas (signs+scales).
// Grid: iq3xxs_grid[256] (uint32). Sign via ksigns64. Scale from gas aux bits.
// Same custom QR=2, QI=32 approach.

#define QR_IQ3_XXS_MMQ 2
#define QI_IQ3_XXS_MMQ (QK_K / (4 * QR_IQ3_XXS_MMQ))
#define VDR_IQ3_XXS_Q8_1_MMQ 4

template <int mmq_y>
static __device__ __forceinline__ void allocate_tiles_iq3_xxs(int** x_ql, half2** x_dm, int** x_qh, int** x_sc) {
  // tile_x_ql: slots 0..15 = grid indices (64 bytes), slots 16..23 = gas data (32 bytes)
  __shared__ int tile_x_ql[mmq_y * (WARP_SIZE_GGUF) + mmq_y];
  __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF / QI_IQ3_XXS_MMQ) + mmq_y / QI_IQ3_XXS_MMQ];

  *x_ql = tile_x_ql;
  *x_dm = (half2*)tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles_iq3_xxs(
    const void* __restrict__ vx,
    int* __restrict__ x_ql,
    half2* __restrict__ x_dm,
    int* __restrict__ x_qh,
    int* __restrict__ x_sc,
    const int& i_offset,
    const int& i_max,
    const int& k,
    const int& blocks_per_row,
    const int& blocks_left) {
  (void)x_qh;
  (void)x_sc;

  const int kbx = k / QI_IQ3_XXS_MMQ;
  const int kqsx = k % QI_IQ3_XXS_MMQ;

  const block_iq3_xxs* bx0 = (const block_iq3_xxs*)vx;
  float* x_dmf = (float*)x_dm;

  // Load grid indices (64 bytes, slots 0..15) and gas data (32 bytes, slots 16..23)
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
    int i = i0 + i_offset;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq3_xxs* bxi = bx0 + i * blocks_per_row + kbx;
    if (kqsx < 16) {
      // Grid indices: first 64 bytes of qs
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    } else if (kqsx < 24) {
      // Gas data (signs+scales): bytes 64..95 of qs
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs + QK_K / 4, kqsx - 16);
    } else {
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = 0;
    }
  }

  // Load d scale
  const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ3_XXS_MMQ;
  const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ3_XXS_MMQ) {
    int i = (i0 + i_offset * QI_IQ3_XXS_MMQ + k / blocks_per_tile_x_row) % mmq_y;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq3_xxs* bxi = bx0 + i * blocks_per_row + kbxd;
    x_dmf[i * (WARP_SIZE_GGUF / QI_IQ3_XXS_MMQ) + i / QI_IQ3_XXS_MMQ + kbxd] = __half2float(bxi->d);
  }
}

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1_mul_mat(
    const int* __restrict__ x_ql,
    const half2* __restrict__ x_dm,
    const int* __restrict__ x_qh,
    const int* __restrict__ x_sc,
    const int* __restrict__ y_qs,
    const half2* __restrict__ y_ds,
    const int& i,
    const int& j,
    const int& k) {
  (void)x_qh;
  (void)x_sc;

  const float* x_dmf = (const float*)x_dm;

  const int ib32 = k / 4;

  // Grid indices for this ib32
  const uint8_t* q3 = (const uint8_t*)&x_ql[i * (WARP_SIZE_GGUF + 1) + ib32 * 2];

  // Gas data for this ib32 (signs + scale packed in 4 bytes per ib32)
  const uint16_t* gas = (const uint16_t*)&x_ql[i * (WARP_SIZE_GGUF + 1) + 16 + ib32];
  uint32_t aux32 = uint32_t(gas[0]) | (uint32_t(gas[1]) << 16);
  const float sub_scale = 0.5f + (aux32 >> 28);

  int sumi = 0;
#pragma unroll
  for (int l = 0; l < VDR_IQ3_XXS_Q8_1_MMQ; ++l) {
    const int sub = l;

    // Extract signs for this sub-group
    uint32_t signs_val = aux32 >> (7 * sub);
    const uint32_t* signs = (const uint32_t*)(ksigns64 + (signs_val & 127));

    // Grid lookup
    const uint32_t* grid1 = iq3xxs_grid + q3[2 * sub + 0];
    const uint32_t* grid2 = iq3xxs_grid + q3[2 * sub + 1];

    // Apply signs
    const int grid_l = __vsub4(grid1[0] ^ signs[0], signs[0]);
    const int grid_h = __vsub4(grid2[0] ^ signs[1], signs[1]);

    // Q8 data
    const int index_y = j * WARP_SIZE_GGUF + (QR_IQ3_XXS_MMQ * (k + l)) % WARP_SIZE_GGUF;
    sumi = iq_mmq_dp4a(grid_l, y_qs[index_y + 0], sumi);
    sumi = iq_mmq_dp4a(grid_h, y_qs[index_y + 1], sumi);
  }

  const float2 ds8 =
      __half22float2(y_ds[j * (WARP_SIZE_GGUF / QI8_1) + (QR_IQ3_XXS_MMQ * k / QI8_1) % (WARP_SIZE_GGUF / QI8_1)]);

  return x_dmf[i * (WARP_SIZE_GGUF / QI_IQ3_XXS_MMQ) + i / QI_IQ3_XXS_MMQ] * sub_scale * 0.5f * ds8.x * sumi;
}

// ========================= IQ2_XXS MMQ =========================
// IQ2_XXS: QK_K=256, qs[QK_K/8=32 uint16]. Grid: iq2xxs_grid[256] (uint64).
// Per ib32: 4 uint16 from qs. First 2 = grid indices (8-bit each). Last 2 = aux (signs+scale).

#define QR_IQ2_XXS_MMQ 2
#define QI_IQ2_XXS_MMQ (QK_K / (4 * QR_IQ2_XXS_MMQ))
#define VDR_IQ2_XXS_Q8_1_MMQ 4

template <int mmq_y>
static __device__ __forceinline__ void allocate_tiles_iq2_xxs(int** x_ql, half2** x_dm, int** x_qh, int** x_sc) {
  // qs data: 32 uint16 = 64 bytes = 16 int32 in slots 0..15
  __shared__ int tile_x_ql[mmq_y * (WARP_SIZE_GGUF) + mmq_y];
  __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF / QI_IQ2_XXS_MMQ) + mmq_y / QI_IQ2_XXS_MMQ];

  *x_ql = tile_x_ql;
  *x_dm = (half2*)tile_x_d;
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles_iq2_xxs(
    const void* __restrict__ vx,
    int* __restrict__ x_ql,
    half2* __restrict__ x_dm,
    int* __restrict__ x_qh,
    int* __restrict__ x_sc,
    const int& i_offset,
    const int& i_max,
    const int& k,
    const int& blocks_per_row,
    const int& blocks_left) {
  (void)x_qh;
  (void)x_sc;

  const int kbx = k / QI_IQ2_XXS_MMQ;
  const int kqsx = k % QI_IQ2_XXS_MMQ;

  const block_iq2_xxs* bx0 = (const block_iq2_xxs*)vx;
  float* x_dmf = (float*)x_dm;

  // Load qs data (32 uint16 = 64 bytes = 16 int32)
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
    int i = i0 + i_offset;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq2_xxs* bxi = bx0 + i * blocks_per_row + kbx;
    if (kqsx < 16) {
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_b2(bxi->qs, kqsx);
    } else {
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = 0;
    }
  }

  // Load d scale
  const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ2_XXS_MMQ;
  const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ2_XXS_MMQ) {
    int i = (i0 + i_offset * QI_IQ2_XXS_MMQ + k / blocks_per_tile_x_row) % mmq_y;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq2_xxs* bxi = bx0 + i * blocks_per_row + kbxd;
    x_dmf[i * (WARP_SIZE_GGUF / QI_IQ2_XXS_MMQ) + i / QI_IQ2_XXS_MMQ + kbxd] = __half2float(bxi->d);
  }
}

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1_mul_mat(
    const int* __restrict__ x_ql,
    const half2* __restrict__ x_dm,
    const int* __restrict__ x_qh,
    const int* __restrict__ x_sc,
    const int* __restrict__ y_qs,
    const half2* __restrict__ y_ds,
    const int& i,
    const int& j,
    const int& k) {
  (void)x_qh;
  (void)x_sc;

  const float* x_dmf = (const float*)x_dm;

  const int ib32 = k / 4;

  // Read the 4 uint16 for this ib32
  const uint16_t* q2 = (const uint16_t*)&x_ql[i * (WARP_SIZE_GGUF + 1) + ib32 * 2];
  const uint8_t* aux8 = (const uint8_t*)q2;
  uint32_t aux32 = uint32_t(q2[2]) | (uint32_t(q2[3]) << 16);
  const float sub_scale = 0.5f + (aux32 >> 28);

  int sumi = 0;
#pragma unroll
  for (int l = 0; l < VDR_IQ2_XXS_Q8_1_MMQ; ++l) {
    const int sub = l;
    const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[sub]);
    const uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * sub)) & 127];

    const int index_y = j * WARP_SIZE_GGUF + (QR_IQ2_XXS_MMQ * (k + l)) % WARP_SIZE_GGUF;
    const int8_t* q8 = (const int8_t*)&y_qs[index_y];

    for (int jj = 0; jj < 8; ++jj) {
      sumi += q8[jj] * grid[jj] * (signs & kmask_iq2xs[jj] ? -1 : 1);
    }
  }

  const float2 ds8 =
      __half22float2(y_ds[j * (WARP_SIZE_GGUF / QI8_1) + (QR_IQ2_XXS_MMQ * k / QI8_1) % (WARP_SIZE_GGUF / QI8_1)]);

  return x_dmf[i * (WARP_SIZE_GGUF / QI_IQ2_XXS_MMQ) + i / QI_IQ2_XXS_MMQ] * sub_scale * 0.25f * ds8.x * sumi;
}

// ========================= IQ2_XS MMQ =========================
// IQ2_XS: QK_K=256, qs[QK_K/8=32 uint16], scales[QK_K/32=8 uint8]. Grid: iq2xs_grid[512] (uint64).
// Per ib32: 4 uint16 from qs. Each uint16: low 9 bits = grid index, upper 7 = signs.

#define QR_IQ2_XS_MMQ 2
#define QI_IQ2_XS_MMQ (QK_K / (4 * QR_IQ2_XS_MMQ))
#define VDR_IQ2_XS_Q8_1_MMQ 4

template <int mmq_y>
static __device__ __forceinline__ void allocate_tiles_iq2_xs(int** x_ql, half2** x_dm, int** x_qh, int** x_sc) {
  __shared__ int tile_x_ql[mmq_y * (WARP_SIZE_GGUF) + mmq_y];
  __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF / QI_IQ2_XS_MMQ) + mmq_y / QI_IQ2_XS_MMQ];
  __shared__ int tile_x_sc[mmq_y * (WARP_SIZE_GGUF / 8) + mmq_y / 8];

  *x_ql = tile_x_ql;
  *x_dm = (half2*)tile_x_d;
  *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles_iq2_xs(
    const void* __restrict__ vx,
    int* __restrict__ x_ql,
    half2* __restrict__ x_dm,
    int* __restrict__ x_qh,
    int* __restrict__ x_sc,
    const int& i_offset,
    const int& i_max,
    const int& k,
    const int& blocks_per_row,
    const int& blocks_left) {
  (void)x_qh;

  const int kbx = k / QI_IQ2_XS_MMQ;
  const int kqsx = k % QI_IQ2_XS_MMQ;

  const block_iq2_xs* bx0 = (const block_iq2_xs*)vx;
  float* x_dmf = (float*)x_dm;

  // Load qs data (32 uint16 = 64 bytes = 16 int32)
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
    int i = i0 + i_offset;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq2_xs* bxi = bx0 + i * blocks_per_row + kbx;
    if (kqsx < 16) {
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_b2(bxi->qs, kqsx);
    } else {
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = 0;
    }
  }

  // Load d scale
  const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ2_XS_MMQ;
  const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ2_XS_MMQ) {
    int i = (i0 + i_offset * QI_IQ2_XS_MMQ + k / blocks_per_tile_x_row) % mmq_y;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq2_xs* bxi = bx0 + i * blocks_per_row + kbxd;
    x_dmf[i * (WARP_SIZE_GGUF / QI_IQ2_XS_MMQ) + i / QI_IQ2_XS_MMQ + kbxd] = __half2float(bxi->d);
  }

  // Load scales (8 bytes = 2 int32)
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
    int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF / 8)) % mmq_y;
    if (need_check) {
      i = min(i, i_max);
    }
    const int ksc = k % (WARP_SIZE_GGUF / 8);
    if (ksc < 2) {
      const block_iq2_xs* bxi = bx0 + i * blocks_per_row + kbx;
      x_sc[i * (WARP_SIZE_GGUF / 8) + i / 8 + ksc] = get_int_from_uint8(bxi->scales, ksc);
    }
  }
}

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1_mul_mat(
    const int* __restrict__ x_ql,
    const half2* __restrict__ x_dm,
    const int* __restrict__ x_qh,
    const int* __restrict__ x_sc,
    const int* __restrict__ y_qs,
    const half2* __restrict__ y_ds,
    const int& i,
    const int& j,
    const int& k) {
  (void)x_qh;

  const float* x_dmf = (const float*)x_dm;

  const int ib32 = k / 4;

  // Read the 4 uint16 for this ib32
  const uint16_t* q2 = (const uint16_t*)&x_ql[i * (WARP_SIZE_GGUF + 1) + ib32 * 2];

  // Per-sub-block scales
  const uint8_t* sc_bytes = (const uint8_t*)&x_sc[i * (WARP_SIZE_GGUF / 8) + i / 8];

  float sumf = 0;
#pragma unroll
  for (int l = 0; l < VDR_IQ2_XS_Q8_1_MMQ; ++l) {
    const int sub = l;

    // Grid lookup: 9-bit index, 7-bit signs
    const uint8_t* grid = (const uint8_t*)(iq2xs_grid + (q2[sub] & 511));
    const uint8_t signs = ksigns_iq2xs[q2[sub] >> 9];

    const int index_y = j * WARP_SIZE_GGUF + (QR_IQ2_XS_MMQ * (k + l)) % WARP_SIZE_GGUF;
    const int8_t* q8 = (const int8_t*)&y_qs[index_y];

    int sumi = 0;
    for (int jj = 0; jj < 8; ++jj) {
      sumi += q8[jj] * grid[jj] * (signs & kmask_iq2xs[jj] ? -1 : 1);
    }

    const uint8_t ls = (sub < 2) ? (sc_bytes[ib32] & 0xf) : (sc_bytes[ib32] >> 4);
    sumf += (0.5f + ls) * sumi;
  }

  const float2 ds8 =
      __half22float2(y_ds[j * (WARP_SIZE_GGUF / QI8_1) + (QR_IQ2_XS_MMQ * k / QI8_1) % (WARP_SIZE_GGUF / QI8_1)]);

  return x_dmf[i * (WARP_SIZE_GGUF / QI_IQ2_XS_MMQ) + i / QI_IQ2_XS_MMQ] * 0.25f * ds8.x * sumf;
}

// ========================= IQ2_S MMQ =========================
// IQ2_S: QK_K=256, qs[QK_K/4=64], qh[QK_K/32=8], scales[QK_K/32=8]. Grid: iq2s_grid[1024] (uint64).
// Per ib32: qs[4] + qh[1] + signs[4] + scales[1].
// 10-bit index: 8 from qs + 2 from qh. Signs from dedicated array at qs+64.

#define QR_IQ2_S_MMQ 2
#define QI_IQ2_S_MMQ (QK_K / (4 * QR_IQ2_S_MMQ))
#define VDR_IQ2_S_Q8_1_MMQ 4

template <int mmq_y>
static __device__ __forceinline__ void allocate_tiles_iq2_s(int** x_ql, half2** x_dm, int** x_qh, int** x_sc) {
  // slots 0..15: qs (64 bytes), slots 16..23: signs (32 bytes)
  __shared__ int tile_x_ql[mmq_y * (WARP_SIZE_GGUF) + mmq_y];
  __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF / QI_IQ2_S_MMQ) + mmq_y / QI_IQ2_S_MMQ];
  __shared__ int tile_x_qh[mmq_y * (WARP_SIZE_GGUF / 4) + mmq_y / 4];
  __shared__ int tile_x_sc[mmq_y * (WARP_SIZE_GGUF / 8) + mmq_y / 8];

  *x_ql = tile_x_ql;
  *x_dm = (half2*)tile_x_d;
  *x_qh = tile_x_qh;
  *x_sc = tile_x_sc;
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles_iq2_s(
    const void* __restrict__ vx,
    int* __restrict__ x_ql,
    half2* __restrict__ x_dm,
    int* __restrict__ x_qh,
    int* __restrict__ x_sc,
    const int& i_offset,
    const int& i_max,
    const int& k,
    const int& blocks_per_row,
    const int& blocks_left) {
  const int kbx = k / QI_IQ2_S_MMQ;
  const int kqsx = k % QI_IQ2_S_MMQ;

  const block_iq2_s* bx0 = (const block_iq2_s*)vx;
  float* x_dmf = (float*)x_dm;

  // Load qs (64 bytes, slots 0..15) and signs (32 bytes at qs+64, slots 16..23)
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
    int i = i0 + i_offset;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq2_s* bxi = bx0 + i * blocks_per_row + kbx;
    if (kqsx < 16) {
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs, kqsx);
    } else if (kqsx < 24) {
      // MMVQ stores grid indices in qs[0..31] and signs in qs[32..63].
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_from_uint8(bxi->qs + QK_K / 8, kqsx - 16);
    } else {
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = 0;
    }
  }

  // Load d
  const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ2_S_MMQ;
  const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ2_S_MMQ) {
    int i = (i0 + i_offset * QI_IQ2_S_MMQ + k / blocks_per_tile_x_row) % mmq_y;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq2_s* bxi = bx0 + i * blocks_per_row + kbxd;
    x_dmf[i * (WARP_SIZE_GGUF / QI_IQ2_S_MMQ) + i / QI_IQ2_S_MMQ + kbxd] = __half2float(bxi->d);
  }

  // Load qh (8 bytes = 2 int32)
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
    int i = (i0 + i_offset * 4 + k / (WARP_SIZE_GGUF / 4)) % mmq_y;
    if (need_check) {
      i = min(i, i_max);
    }
    const int kqh = k % (WARP_SIZE_GGUF / 4);
    if (kqh < 2) {
      const block_iq2_s* bxi = bx0 + i * blocks_per_row + kbx;
      x_qh[i * (WARP_SIZE_GGUF / 4) + i / 4 + kqh] = get_int_from_uint8(bxi->qh, kqh);
    }
  }

  // Load scales (8 bytes = 2 int32)
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 8) {
    int i = (i0 + i_offset * 8 + k / (WARP_SIZE_GGUF / 8)) % mmq_y;
    if (need_check) {
      i = min(i, i_max);
    }
    const int ksc = k % (WARP_SIZE_GGUF / 8);
    if (ksc < 2) {
      const block_iq2_s* bxi = bx0 + i * blocks_per_row + kbx;
      x_sc[i * (WARP_SIZE_GGUF / 8) + i / 8 + ksc] = get_int_from_uint8(bxi->scales, ksc);
    }
  }
}

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1_mul_mat(
    const int* __restrict__ x_ql,
    const half2* __restrict__ x_dm,
    const int* __restrict__ x_qh,
    const int* __restrict__ x_sc,
    const int* __restrict__ y_qs,
    const half2* __restrict__ y_ds,
    const int& i,
    const int& j,
    const int& k) {
  const float* x_dmf = (const float*)x_dm;

  const int ib32 = k / 4;

  // Grid indices from qs[0..31] (stored in tile slots 0..7 as int32)
  const int qs_word = x_ql[i * (WARP_SIZE_GGUF + 1) + ib32];  // qs[4*ib32 .. 4*ib32+3]

  // qh for the 10-bit index
  const uint8_t* qh_all = (const uint8_t*)&x_qh[i * (WARP_SIZE_GGUF / 4) + i / 4];
  const uint8_t qh_val = qh_all[ib32];

  // Signs from qs[32..63] (stored in tile slots 16..23)
  const uint8_t* sign_bytes = (const uint8_t*)&x_ql[i * (WARP_SIZE_GGUF + 1) + 16];

  // Per-sub-block scales
  const uint8_t* sc_bytes = (const uint8_t*)&x_sc[i * (WARP_SIZE_GGUF / 8) + i / 8];

  float sumf = 0;
#pragma unroll
  for (int l = 0; l < VDR_IQ2_S_Q8_1_MMQ; ++l) {
    const int sub = l;

    // 10-bit grid index
    // #28784 analogue: byte via __byte_perm (see IQ1_S above)
    const uint32_t* grid = (const uint32_t*)(iq2s_grid + (__byte_perm(qs_word, 0, 0x4440 | sub) | ((qh_val << (8 - 2 * sub)) & 0x300)));

    const uint8_t sign_byte = sign_bytes[4 * ib32 + sub];
    uint32_t signs0 = __vcmpeq4(((sign_byte & 0xf) * 0x01010101) & 0x08040201, 0x08040201);
    uint32_t signs1 = __vcmpeq4(((sign_byte >> 4) * 0x01010101) & 0x08040201, 0x08040201);
    const int grid_l = __vsub4(grid[0] ^ signs0, signs0);
    const int grid_h = __vsub4(grid[1] ^ signs1, signs1);

    const int index_y = j * WARP_SIZE_GGUF + (QR_IQ2_S_MMQ * (k + l)) % WARP_SIZE_GGUF;

    int sumi = 0;
    sumi = iq_mmq_dp4a(grid_l, y_qs[index_y + 0], sumi);
    sumi = iq_mmq_dp4a(grid_h, y_qs[index_y + 1], sumi);

    const uint8_t ls = (sub < 2) ? (sc_bytes[ib32] & 0xf) : (sc_bytes[ib32] >> 4);
    sumf += (0.5f + ls) * sumi;
  }

  const float2 ds8 =
      __half22float2(y_ds[j * (WARP_SIZE_GGUF / QI8_1) + (QR_IQ2_S_MMQ * k / QI8_1) % (WARP_SIZE_GGUF / QI8_1)]);

  return x_dmf[i * (WARP_SIZE_GGUF / QI_IQ2_S_MMQ) + i / QI_IQ2_S_MMQ] * 0.25f * ds8.x * sumf;
}

// ========================= IQ1_S MMQ =========================
// IQ1_S: QK_K=256, qs[QK_K/8=32], qh[QK_K/32=8 uint16]. Grid: iq1s_grid_gpu[2048] (uint64).
// 11-bit grid index (8 from qs + 3 from qh). need_sum=true (delta correction).
// Grid values are packed nibbles: (grid>>0)&0x0F0F0F0F, (grid>>4)&0x0F0F0F0F.

#define QR_IQ1_S_MMQ 2
#define QI_IQ1_S_MMQ (QK_K / (4 * QR_IQ1_S_MMQ))
#define VDR_IQ1_S_Q8_1_MMQ 4

template <int mmq_y>
static __device__ __forceinline__ void allocate_tiles_iq1_s(int** x_ql, half2** x_dm, int** x_qh, int** x_sc) {
  // slots 0..7: qs (32 bytes = 8 int32)
  __shared__ int tile_x_ql[mmq_y * (WARP_SIZE_GGUF) + mmq_y];
  __shared__ float tile_x_d[mmq_y * (WARP_SIZE_GGUF / QI_IQ1_S_MMQ) + mmq_y / QI_IQ1_S_MMQ];
  __shared__ int tile_x_qh[mmq_y * (WARP_SIZE_GGUF / 4) + mmq_y / 4];  // qh: 16 bytes = 4 int32

  *x_ql = tile_x_ql;
  *x_dm = (half2*)tile_x_d;
  *x_qh = tile_x_qh;
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void load_tiles_iq1_s(
    const void* __restrict__ vx,
    int* __restrict__ x_ql,
    half2* __restrict__ x_dm,
    int* __restrict__ x_qh,
    int* __restrict__ x_sc,
    const int& i_offset,
    const int& i_max,
    const int& k,
    const int& blocks_per_row,
    const int& blocks_left) {
  (void)x_sc;

  const int kbx = k / QI_IQ1_S_MMQ;
  const int kqsx = k % QI_IQ1_S_MMQ;

  const block_iq1_s* bx0 = (const block_iq1_s*)vx;
  float* x_dmf = (float*)x_dm;

  // Load qs (32 bytes = 8 int32, slots 0..7)
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps) {
    int i = i0 + i_offset;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq1_s* bxi = bx0 + i * blocks_per_row + kbx;
    if (kqsx < 8) {
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = get_int_b2(bxi->qs, kqsx);
    } else {
      x_ql[i * (WARP_SIZE_GGUF + 1) + k] = 0;
    }
  }

  // Load d scale
  const int blocks_per_tile_x_row = WARP_SIZE_GGUF / QI_IQ1_S_MMQ;
  const int kbxd = k % blocks_per_tile_x_row;

#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * QI_IQ1_S_MMQ) {
    int i = (i0 + i_offset * QI_IQ1_S_MMQ + k / blocks_per_tile_x_row) % mmq_y;
    if (need_check) {
      i = min(i, i_max);
    }
    const block_iq1_s* bxi = bx0 + i * blocks_per_row + kbxd;
    x_dmf[i * (WARP_SIZE_GGUF / QI_IQ1_S_MMQ) + i / QI_IQ1_S_MMQ + kbxd] = __half2float(bxi->d);
  }

  // Load qh (8 uint16 = 16 bytes = 4 int32)
#pragma unroll
  for (int i0 = 0; i0 < mmq_y; i0 += nwarps * 4) {
    int i = (i0 + i_offset * 4 + k / (WARP_SIZE_GGUF / 4)) % mmq_y;
    if (need_check) {
      i = min(i, i_max);
    }
    const int kqh = k % (WARP_SIZE_GGUF / 4);
    if (kqh < 4) {
      const block_iq1_s* bxi = bx0 + i * blocks_per_row + kbx;
      x_qh[i * (WARP_SIZE_GGUF / 4) + i / 4 + kqh] = get_int_b2(bxi->qh, kqh);
    }
  }
}

static __device__ __forceinline__ float vec_dot_iq1_s_q8_1_mul_mat(
    const int* __restrict__ x_ql,
    const half2* __restrict__ x_dm,
    const int* __restrict__ x_qh,
    const int* __restrict__ x_sc,
    const int* __restrict__ y_qs,
    const half2* __restrict__ y_ds,
    const int& i,
    const int& j,
    const int& k) {
  (void)x_sc;

  const float* x_dmf = (const float*)x_dm;

  const int iqs = k / 4;

  // Read qs (4 bytes per iqs group)
  const int qs_packed = x_ql[i * (WARP_SIZE_GGUF + 1) + iqs];

  // Read qh for this iqs group
  const uint16_t* qh_all = (const uint16_t*)&x_qh[i * (WARP_SIZE_GGUF / 4) + i / 4];
  const int qh = qh_all[iqs];

  int sumi = 0;
#pragma unroll
  for (int l = 0; l < VDR_IQ1_S_Q8_1_MMQ; ++l) {
    const int sub = l;

    // 11-bit grid index: qs[sub] | (3 bits from qh << 8)
    // #28784 analogue: byte via __byte_perm (nvcc 13.2.0/13.2.1 drop a plain byte mask on sm_120 and the grid index runs out of range)
    const int grid = iq1s_grid_gpu[__byte_perm(qs_packed, 0, 0x4440 | sub) | (((qh >> 3 * sub) & 0x07) << 8)];
    const int grid0 = (grid >> 0) & 0x0F0F0F0F;
    const int grid1 = (grid >> 4) & 0x0F0F0F0F;

    const int index_y = j * WARP_SIZE_GGUF + (QR_IQ1_S_MMQ * (k + l)) % WARP_SIZE_GGUF;
    sumi = iq_mmq_dp4a(grid0, y_qs[index_y + 0], sumi);
    sumi = iq_mmq_dp4a(grid1, y_qs[index_y + 1], sumi);
  }

  const float d1q = x_dmf[i * (WARP_SIZE_GGUF / QI_IQ1_S_MMQ) + i / QI_IQ1_S_MMQ] * (((qh >> 11) & 0x0E) + 1);
  const float delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f * IQ1S_DELTA / 0x8000);

  // need_sum=true: y_ds stores half2(d, sum)
  const float2 ds =
      __half22float2(y_ds[j * (WARP_SIZE_GGUF / QI8_1) + (QR_IQ1_S_MMQ * k / QI8_1) % (WARP_SIZE_GGUF / QI8_1)]);

  return d1q * (ds.x * sumi + ds.y * delta);
}
#endif  // !defined(USE_MUSA)
