# SPDX-License-Identifier: Apache-2.0
"""Native pieces of the BAR1 direct transport.

Two separate extensions, because they have different requirements:

``barlink_bar1_ext``  (CUDA)
    The collective kernels. ``mesh`` and ``ring`` for ``all_reduce``, ported
    from ``/spinning/nvidia-open-595/bar1_kollektiv.cu``; plus ``a2a`` for
    ``all_to_all_single``, which has no counterpart in the probe and is
    therefore deliberately NOT ported but newly written -- unmeasured until
    the benchmark run exists. Needs only nvcc and torch, builds anywhere
    ``barlink_device`` also builds.

``barlink_bar1_dmabuf_ext``  (C++ only)
    The dma-buf export via the RM ioctls. Needs the **headers of the
    open-source NVIDIA kernel modules**; without them it isn't built and
    the transport declines. This path can't be reconstructed from Python:
    ``NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD`` and
    ``NV_ESC_EXPORT_TO_DMABUF_FD`` require the UAPI structures
    (``nvos.h``, ``nv-ioctl.h``, ``class/cl0000.h``), and their sizes and
    field offsets are version-bound.

What was taken from the probe, and what changed
------------------------------------------------------------
TAKEN OVER, as literally as possible: ``writeV4`` (``st.global.wt``),
``readV4`` (``ld.global.cv``), ``readFlag`` (``ld.mmio.relaxed.sys`` or
``ld.global.cv``), ``readFlush``, ``barrier<GRID>``, the step sequence and
the barrier count of both kernels, the chunk geometry (remainder onto the
leading chunks), the 256-byte-per-rank flag lines (topology, step, sender),
and the grid-size computation.

CHANGED, each with its reason:

* **Arbitrary R instead of a fixed three.** ``RANGE_N`` was a compile-time
  constant; here ``R`` is a field of the argument struct. The loops run
  over ``R``, and the arrays are sized to
  ``BARLINK_BAR1_MAX_RANKS``.
* **Data types.** The probe computed exclusively in ``float``. Here the
  reduction is templated over ``float``/``half``/``bfloat16``. The
  **access width stays 128 bits** (``uint4``) -- that is the property the
  measurement depends on, not the element type. The ``float`` path is
  bit-identical to the probe.
* **No self-check.** ``sollWert``/``basisWert``/``fehlerZahl`` were
  measurement tooling of the probe (it knew its own input). A transport
  does not know its input; the check was dropped without replacement
  rather than faked. The byte-level verification at setup time remains.
* **No round term ``rt``.** Also measurement tooling (it made the payload
  differ per round so a stale buffer would stand out).
* **Round number in device memory.** The probe passed ``round`` in as a
  value on every launch. That doesn't survive CUDA graph capture -- a
  captured launch would replay the same number forever and every flag
  would appear satisfied immediately. The round therefore lives in a
  device word, is read by the kernel, and is advanced at the end, exactly
  the way ``barlink_device`` does it with its sequence number.
* **No phase clock.** ``%globaltimer`` timestamps and the end-of-kernel
  logging write belong in the probe, not on the hot path.
"""

from __future__ import annotations

import logging
import os
import pathlib
import time
from typing import Optional

from sglang.srt.distributed.device_communicators import (
    barlink_env_guard,  # noqa: F401  (rejects retired SGLANG_HTCCL* vars)
)

logger = logging.getLogger(__name__)

_ext = None
_dmabuf_ext = None
_dmabuf_reason = ""

#: Largest rank group for which the argument arrays have room. Eight,
#: because that keeps the struct around 1 KiB and safely within the 4 KiB
#: region for kernel parameters.
MAX_RANGE = 8

#: Where the headers of the open-source kernel modules live. Rank-uniform
#: like every other SGLANG_BARLINK* variable.
NV_SOURCE_DEFAULT = "/spinning/nvidia-open-595"

#: Subdirectories relative to that root -- exactly the three that
#: sonden/dmabuf_p2p_probe.cpp is compiled against.
NV_INCLUDES = (
    "kernel-open/common/inc",
    "src/common/sdk/nvidia/inc",
    "src/nvidia/arch/nvalloc/unix/include",
)


# ===========================================================================
# Kernel source
# ===========================================================================

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <vector>

namespace cg = cooperative_groups;

#define BARLINK_BAR1_MAX_RANKS 8
#define BARLINK_BAR1_MAX_STEPS (2 * (BARLINK_BAR1_MAX_RANKS - 1))

// Rate limit of the host abort probe inside the spin loops. The word lives in
// pinned, device-mapped host memory, so reading it is a PCIe round trip -- one
// per 1024 spin iterations keeps it far below the R volatile BAR1 flag loads
// the same loop already issues per iteration, while still reacting within
// microseconds of the host store. A collective that completes never reaches
// the probe: the loop leaves through its "all peers arrived" break first.
#define BARLINK_BAR1_HOST_MASK 1023u

// Kernel variants, unchanged from the probe.
#define K_1BLK   0
#define K_GRID 1
// Flag load mode. LA_MMIO is the only genuine cache bypass and the probe's
// default (run field uncachedLeseart = LA_MMIO).
#define LA_CV    0
#define LA_MMIO  2

using u64 = unsigned long long;

// ---------------------------------------------------------------------------
// Accesses that must not let a cache stage swallow them. Taken literally
// from bar1_kollektiv.cu: the bytes arrive via DMA from a FOREIGN card into
// the framebuffer, and a reused cache line would show the stale state.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint4 readV4(const void *p)
{
    uint4 v;
    asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p) : "memory");
    return v;
}

__device__ __forceinline__ void writeV4(void *p, uint4 v)
{
    asm volatile("st.global.wt.v4.u32 [%0], {%1,%2,%3,%4};"
                 :: "l"(p), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

template<int LA>
__device__ __forceinline__ u64 readFlag(const u64 *p)
{
    u64 v;
    if (LA == LA_MMIO) {
        asm volatile("ld.mmio.relaxed.sys.global.u64 %0, [%1];"
                     : "=l"(v) : "l"(p) : "memory");
    } else {
        asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
    }
    return v;
}

__device__ __forceinline__ void writeU64(void *p, u64 v)
{
    asm volatile("st.global.wt.u64 [%0], %1;" :: "l"(p), "l"(v) : "memory");
}

// Single byte, cache-bypassing. Needed ONLY in the remainder path of
// all_to_all: there a block can end at a boundary that isn't a multiple of
// 16, and the last bytes come -- like the rest -- via DMA from a foreign
// card. Without .cv, a reused cache line would sit in front of them.
__device__ __forceinline__ unsigned char readB(const void *p)
{
    unsigned int v;
    asm volatile("ld.global.cv.u8 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return (unsigned char)v;
}

// Read flush for the case where payload and flag sit on DIFFERENT PCIe
// targets. Kept for continuity, but not used by this transport (both sit on
// the same target card, and per-requester/completer-pair ordering holds);
// the switch exists so the acknowledgement style can be tested later
// without a code change.
__device__ __forceinline__ void readFlush(const uint4 *peerRecv, int n4)
{
    if (n4 <= 0) return;
    unsigned int d = 0;
    asm volatile("ld.mmio.relaxed.sys.global.u32 %0, [%1];"
                 : "+r"(d) : "l"(peerRecv + (n4 - 1)) : "memory");
}

// ---------------------------------------------------------------------------
// Element-wise addition WITHIN a 128-bit packet.
//
// The probe only knew float4. The access width stays the same here -- what
// changed is only how the 16 bytes are interpreted. For float, the
// generated code is identical to the probe.
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int addF(unsigned int a, unsigned int b)
{
    return __float_as_uint(__uint_as_float(a) + __uint_as_float(b));
}

__device__ __forceinline__ unsigned int addH2(unsigned int a, unsigned int b)
{
    __half2 x = *(const __half2 *)&a;
    __half2 y = *(const __half2 *)&b;
    __half2 z = __hadd2(x, y);
    return *(const unsigned int *)&z;
}

__device__ __forceinline__ unsigned int addBF2(unsigned int a, unsigned int b)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    __nv_bfloat162 x = *(const __nv_bfloat162 *)&a;
    __nv_bfloat162 y = *(const __nv_bfloat162 *)&b;
    __nv_bfloat162 z = __hadd2(x, y);
    return *(const unsigned int *)&z;
#else
    // Before sm_80 there are no bf16 arithmetic units. Routed through float
    // so the same source still compiles -- bf16 was only measured on sm_86
    // and sm_120.
    const __nv_bfloat16 *x = (const __nv_bfloat16 *)&a;
    const __nv_bfloat16 *y = (const __nv_bfloat16 *)&b;
    __nv_bfloat16 r[2];
    r[0] = __float2bfloat16(__bfloat162float(x[0]) + __bfloat162float(y[0]));
    r[1] = __float2bfloat16(__bfloat162float(x[1]) + __bfloat162float(y[1]));
    return *(const unsigned int *)r;
#endif
}

template<typename T>
__device__ __forceinline__ uint4 addV4(uint4 a, uint4 b);

template<>
__device__ __forceinline__ uint4 addV4<float>(uint4 a, uint4 b)
{
    uint4 r;
    r.x = addF(a.x, b.x); r.y = addF(a.y, b.y);
    r.z = addF(a.z, b.z); r.w = addF(a.w, b.w);
    return r;
}

template<>
__device__ __forceinline__ uint4 addV4<__half>(uint4 a, uint4 b)
{
    uint4 r;
    r.x = addH2(a.x, b.x); r.y = addH2(a.y, b.y);
    r.z = addH2(a.z, b.z); r.w = addH2(a.w, b.w);
    return r;
}

template<>
__device__ __forceinline__ uint4 addV4<__nv_bfloat16>(uint4 a, uint4 b)
{
    uint4 r;
    r.x = addBF2(a.x, b.x); r.y = addBF2(a.y, b.y);
    r.z = addBF2(a.z, b.z); r.w = addBF2(a.w, b.w);
    return r;
}

// ---------------------------------------------------------------------------
// Chunk geometry. Chunk j out of n4 packets over R ranks; the remainder
// n4 % R goes onto the LEADING chunks. Unchanged from the probe, just R
// instead of RANGE_N. The same computation appears once more host-side in
// the seam check (bar1_all_reduce): a seam checked on both sides with THE
// SAME wrong formula would never surface.
// ---------------------------------------------------------------------------
__device__ __host__ __forceinline__ void chunkBounds(int n4, int j, int R,
                                                      int *off, int *len)
{
    const int base = n4 / R;
    const int rest  = n4 - base * R;
    *len = base + (j < rest ? 1 : 0);
    *off = j * base + (j < rest ? j : rest);
}

// ---------------------------------------------------------------------------
// Arguments. One struct, so the cooperative launch and the normal <<< >>>
// launch provably pass the same thing.
// ---------------------------------------------------------------------------
struct Bar1Args {
    const uint4 *in;          // local, VRAM
    uint4       *out;         // local, VRAM
    u64         *roundDev;    // one word, local: the running round number
    unsigned int *ctlStatus;  // 1 = time limit exceeded
    unsigned int *abortDev; // K_GRID only: grid-wide abort bit
    // Host-set abort word (pinned, device-mapped). nullptr = absent, and then
    // the spin loops keep only their cycle deadline. Set to 1 by the peer
    // watchdog when a peer process is provably gone, which is the one fact the
    // rank-local deadline cannot see.
    const unsigned int *abortHost;
    int          n4;
    int          R;
    int          rank;
    u64          capCycles;

    // mesh: indexed by RANK NUMBER, own entry stays empty.
    uint4       *nzSendRS[BARLINK_BAR1_MAX_RANKS];
    uint4       *nzSendAG[BARLINK_BAR1_MAX_RANKS];
    const uint4 *nzRecvRS[BARLINK_BAR1_MAX_RANKS];
    const uint4 *nzRecvAG[BARLINK_BAR1_MAX_RANKS];
    u64         *nzFlagTo [2][BARLINK_BAR1_MAX_RANKS];
    const u64   *nzFlagFrom[2][BARLINK_BAR1_MAX_RANKS];

    // ring: indexed by STEP, 0 .. 2*(R-1)-1.
    uint4       *rgSend   [BARLINK_BAR1_MAX_STEPS];
    const uint4 *rgRecv   [BARLINK_BAR1_MAX_STEPS];
    u64         *rgFlagTo [BARLINK_BAR1_MAX_STEPS];
    const u64   *rgFlagFrom[BARLINK_BAR1_MAX_STEPS];
};

// ---------------------------------------------------------------------------
// Barrier -- the only difference between the kernel variants.
// ---------------------------------------------------------------------------
template<int GRID>
__device__ __forceinline__ void barrier(void)
{
    if (GRID == K_GRID) cg::this_grid().sync();
    else                  __syncthreads();
}

// ---------------------------------------------------------------------------
// Phases. All kernels use the same ones.
// ---------------------------------------------------------------------------

// Write own contribution via DMA into the target card's receive slot.
__device__ __forceinline__ void sendPhase(const uint4 *__restrict__ in,
                                           uint4 *peerRecv,
                                           int n4, int tid, int nth)
{
    for (int j = tid; j < n4; j += nth) writeV4(peerRecv + j, in[j]);
}

// One piece to ALL peers, in ONE loop -- so the result is read from local
// VRAM only once (the probe's distribute phase, generalized from two
// targets to R-1).
__device__ __forceinline__ void scatterPhase(const uint4 *__restrict__ res,
                                              uint4 *const *dst,
                                              int R, int rank,
                                              int n4, int tid, int nth)
{
    for (int j = tid; j < n4; j += nth) {
        const uint4 v = res[j];
        for (int z = 0; z < R; ++z) {
            if (z == rank) continue;
            writeV4(dst[z] + j, v);
        }
    }
}

// Reduce locally over the own contribution and ALL R-1 received ones.
// (the probe's reduce3 phase, generalized.)
template<typename T>
__device__ __forceinline__ void reduceNPhase(const uint4 *__restrict__ in,
                                                uint4 *out,
                                                const uint4 *const *recv,
                                                int R, int rank,
                                                int n4, int tid, int nth)
{
    for (int j = tid; j < n4; j += nth) {
        uint4 s = in[j];
        for (int q = 0; q < R; ++q) {
            if (q == rank) continue;
            s = addV4<T>(s, readV4(recv[q] + j));
        }
        out[j] = s;
    }
}

// Adopt what was received (the probe's adopt phase, without the check).
__device__ __forceinline__ void takeOverPhase(uint4 *out, const uint4 *recv,
                                                int n4, int tid, int nth)
{
    for (int j = tid; j < n4; j += nth) out[j] = readV4(recv + j);
}

// Ring reduce-scatter, receive step: own contribution plus partial sum.
template<typename T>
__device__ __forceinline__ void ringAdd(const uint4 *__restrict__ in,
                                            uint4 *out, const uint4 *recv,
                                            int n4, int tid, int nth)
{
    for (int j = tid; j < n4; j += nth) out[j] = addV4<T>(in[j], readV4(recv + j));
}

// ---------------------------------------------------------------------------
// Round number.
//
// Read by EVERY thread at kernel start (a uniform value, one L2 hit),
// advanced by the globally first thread at kernel end -- including on the
// abort path. If it weren't advanced on abort, the ranks would drift apart
// permanently: one would count the call, the other wouldn't, and every
// subsequent flag would land at the wrong slot.
// ---------------------------------------------------------------------------
// Templated because the a2a kernel has its own argument struct and the
// round advance must be identical in EVERY kernel -- a second version would
// be exactly the place where the ranks drift apart.
template<typename ARGS>
__device__ __forceinline__ void writeRound(const ARGS &A, u64 round)
{
    *(volatile u64 *)A.roundDev = round;
    __threadfence_system();
}

// ---------------------------------------------------------------------------
// TOPOLOGY 'mesh' -- reduce-scatter + allgather over ALL pairs, two
// barriers. Structure unchanged from ar3_netz_kernel; the fixed ranks a and
// b are replaced by loops over all peers.
//
// SEPARATE SLOTS FOR RS AND AG: there is no ordering between "I read my RS
// slots" and "the other side writes its AG chunk". With a shared slot set
// this would need a third barrier.
// ---------------------------------------------------------------------------
template<typename T, int LA, int FLUSH, int GRID>
__global__ void bar1_mesh_kernel(Bar1Args A)
{
    const int tid = (GRID == K_GRID)
                        ? (int)(blockIdx.x * blockDim.x + threadIdx.x)
                        : (int)threadIdx.x;
    const int nth = (GRID == K_GRID)
                        ? (int)(gridDim.x * blockDim.x)
                        : (int)blockDim.x;
    const bool isFirst = (tid == 0);
    const int  n4 = A.n4, R = A.R, r = A.rank;
    const u64  round = *(const volatile u64 *)A.roundDev + 1ull;

    __shared__ int abortS;
    if (GRID == K_1BLK) {
        if (threadIdx.x == 0) abortS = 0;
        __syncthreads();
    } else if (isFirst) {
        *(volatile unsigned int *)A.abortDev = 0u;
        __threadfence();
    }

    // -----------------------------------------------------------------------
    // The pointer tables go into shared memory first.
    //
    // `A` lives in parameter space, which doesn't support dynamic indexing.
    // A single `A.nzSendRS[z]` with a running z therefore forces nvcc to
    // copy the WHOLE parameter block into local memory per thread --
    // measured on exactly this kernel: STACK 64 bytes, versus 0 for
    // `bar1_ring_kernel` (fixed indices). At 256 threads that's 16 KiB of
    // shuffling per block before a single payload byte moves.
    //
    // Same fix as in the a2a kernel and the pipelined kernel: one thread per
    // block writes the tables into __shared__ once, after which everyone
    // indexes dynamically from there. Per block, not per grid --
    // `__syncthreads()` suffices and holds in BOTH kernel variants;
    // `barrier<GRID>()` would be the wrong barrier here, because shared
    // memory is block-local.
    //
    // The flag pointers are included here even though only `isFirst` touches
    // them: what determines whether the parameter block must go to local
    // memory is THAT something is indexed dynamically anywhere, not by how
    // many threads.
    //
    // Entries with z == r are 0 host-side and are never dereferenced; they
    // are still copied along so that the index stays the rank number in
    // every table.
    __shared__ uint4       *sSendRS[BARLINK_BAR1_MAX_RANKS];
    __shared__ uint4       *sSendAG[BARLINK_BAR1_MAX_RANKS];
    __shared__ const uint4 *sRecvRS[BARLINK_BAR1_MAX_RANKS];
    __shared__ const uint4 *sRecvAG[BARLINK_BAR1_MAX_RANKS];
    __shared__ u64         *sFlagTo [2][BARLINK_BAR1_MAX_RANKS];
    __shared__ const u64   *sFlagFrom[2][BARLINK_BAR1_MAX_RANKS];
    if (threadIdx.x == 0) {
        for (int z = 0; z < R; ++z) {
            sSendRS[z] = A.nzSendRS[z];
            sSendAG[z] = A.nzSendAG[z];
            sRecvRS[z] = A.nzRecvRS[z];
            sRecvAG[z] = A.nzRecvAG[z];
            sFlagTo [0][z] = A.nzFlagTo [0][z];
            sFlagTo [1][z] = A.nzFlagTo [1][z];
            sFlagFrom[0][z] = A.nzFlagFrom[0][z];
            sFlagFrom[1][z] = A.nzFlagFrom[1][z];
        }
    }
    __syncthreads();

    int offR, lenR;
    chunkBounds(n4, r, R, &offR, &lenR);

    // --- 1. Reduce-scatter: chunk z to rank z ------------------------------
    for (int z = 0; z < R; ++z) {
        if (z == r) continue;
        int off, len;
        chunkBounds(n4, z, R, &off, &len);
        sendPhase(A.in + off, sSendRS[z], len, tid, nth);
    }
    __threadfence_system();
    barrier<GRID>();

    if (isFirst) {
        for (int z = 0; z < R; ++z) {
            if (z == r) continue;
            if (FLUSH) {
                int off, len;
                chunkBounds(n4, z, R, &off, &len);
                readFlush(sSendRS[z], len);
            }
            writeU64(sFlagTo[0][z], round);
        }
        __threadfence_system();

        bool ab = false;
        long long t0 = clock64();
        unsigned int probeCounter = 0u;
        for (;;) {
            bool allArrived = true;
            for (int s = 0; s < R; ++s) {
                if (s == r) continue;
                if (readFlag<LA>(sFlagFrom[0][s]) != round) { allArrived = false; break; }
            }
            if (allArrived) break;
            if ((u64)(clock64() - t0) > A.capCycles) { ab = true; break; }
            if (A.abortHost != nullptr && ((++probeCounter & BARLINK_BAR1_HOST_MASK) == 0u)
                && *(const volatile unsigned int *)A.abortHost != 0u) { ab = true; break; }
        }
        if (ab) {
            if (GRID == K_1BLK) abortS = 1;
            else { *(volatile unsigned int *)A.abortDev = 1u; __threadfence(); }
        }
    }
    barrier<GRID>();
    {
        const int aborted = (GRID == K_1BLK)
                                ? abortS
                                : (int)*(volatile unsigned int *)A.abortDev;
        if (aborted) {
            if (isFirst) {
                *A.ctlStatus = 1u;
                writeRound(A, round);
            }
            return;
        }
    }
    __threadfence_system();

    // --- 2. reduce own chunk ------------------------------------------------
    {
        // The receive slot carries ONLY the own chunk, so it starts at 0 --
        // the chunk offset offR applies to the LOCAL buffer, not to the
        // slot.
        //
        // `sRecvRS` instead of a local copy: the copy was a per-THREAD array
        // in local memory, and it was also the reason the whole parameter
        // block had to go there.
        reduceNPhase<T>(A.in + offR, A.out + offR, sRecvRS, R, r, lenR,
                           tid, nth);
    }
    barrier<GRID>();

    // --- 3. Allgather: the finished own chunk to everyone else --------------
    scatterPhase(A.out + offR, sSendAG, R, r, lenR, tid, nth);
    __threadfence_system();
    barrier<GRID>();

    if (isFirst) {
        for (int z = 0; z < R; ++z) {
            if (z == r) continue;
            if (FLUSH) readFlush(sSendAG[z], lenR);
            writeU64(sFlagTo[1][z], round);
        }
        __threadfence_system();

        bool ab = false;
        long long t0 = clock64();
        unsigned int probeCounter = 0u;
        for (;;) {
            bool allArrived = true;
            for (int s = 0; s < R; ++s) {
                if (s == r) continue;
                if (readFlag<LA>(sFlagFrom[1][s]) != round) { allArrived = false; break; }
            }
            if (allArrived) break;
            if ((u64)(clock64() - t0) > A.capCycles) { ab = true; break; }
            if (A.abortHost != nullptr && ((++probeCounter & BARLINK_BAR1_HOST_MASK) == 0u)
                && *(const volatile unsigned int *)A.abortHost != 0u) { ab = true; break; }
        }
        if (ab) {
            if (GRID == K_1BLK) abortS = 1;
            else { *(volatile unsigned int *)A.abortDev = 1u; __threadfence(); }
        }
    }
    barrier<GRID>();
    {
        const int aborted = (GRID == K_1BLK)
                                ? abortS
                                : (int)*(volatile unsigned int *)A.abortDev;
        if (aborted) {
            if (isFirst) {
                *A.ctlStatus = 1u;
                writeRound(A, round);
            }
            return;
        }
    }
    __threadfence_system();

    for (int s = 0; s < R; ++s) {
        if (s == r) continue;
        int off, len;
        chunkBounds(n4, s, R, &off, &len);
        takeOverPhase(A.out + off, sRecvAG[s], len, tid, nth);
    }
    barrier<GRID>();
    if (isFirst) writeRound(A, round);
}

// ---------------------------------------------------------------------------
// TOPOLOGY 'ring' -- ring reduce-scatter + ring allgather, 2*(R-1) barriers.
// Always sends to (r+1)%R, always receives from (r-1+R)%R.
// Unchanged from ar3_ring_kernel, only RANGE_N -> A.R and without the check.
// ---------------------------------------------------------------------------
template<typename T, int LA, int FLUSH, int GRID>
__global__ void bar1_ring_kernel(Bar1Args A)
{
    const int tid = (GRID == K_GRID)
                        ? (int)(blockIdx.x * blockDim.x + threadIdx.x)
                        : (int)threadIdx.x;
    const int nth = (GRID == K_GRID)
                        ? (int)(gridDim.x * blockDim.x)
                        : (int)blockDim.x;
    const bool isFirst = (tid == 0);
    const int  n4 = A.n4, R = A.R, r = A.rank;
    const u64  round = *(const volatile u64 *)A.roundDev + 1ull;

    __shared__ int abortS;
    if (GRID == K_1BLK) {
        if (threadIdx.x == 0) abortS = 0;
        __syncthreads();
    } else if (isFirst) {
        *(volatile unsigned int *)A.abortDev = 0u;
        __threadfence();
    }

    // ------------------------- Reduce-scatter -------------------------------
    for (int s = 0; s < R - 1; ++s) {
        const int cs = (r - s + 2 * R) % R;         // chunk sent
        const int cr = (r - s - 1 + 2 * R) % R;     // chunk received
        int offS, lenS, offE, lenE;
        chunkBounds(n4, cs, R, &offS, &lenS);
        chunkBounds(n4, cr, R, &offE, &lenE);

        // Step 0 sends the own contribution, every further step the partial
        // sum formed in the previous step.
        const uint4 *source = (s == 0) ? (A.in + offS)
                                       : (const uint4 *)(A.out + offS);
        sendPhase(source, A.rgSend[s], lenS, tid, nth);
        __threadfence_system();
        barrier<GRID>();

        if (isFirst) {
            if (FLUSH) readFlush(A.rgSend[s], lenS);
            writeU64(A.rgFlagTo[s], round);
            __threadfence_system();

            bool ab = false;
            long long t0 = clock64();
            unsigned int probeCounter = 0u;
            while (readFlag<LA>(A.rgFlagFrom[s]) != round) {
                if ((u64)(clock64() - t0) > A.capCycles) { ab = true; break; }
                if (A.abortHost != nullptr && ((++probeCounter & BARLINK_BAR1_HOST_MASK) == 0u)
                    && *(const volatile unsigned int *)A.abortHost != 0u) { ab = true; break; }
            }
            if (ab) {
                if (GRID == K_1BLK) abortS = 1;
                else { *(volatile unsigned int *)A.abortDev = 1u; __threadfence(); }
            }
        }
        barrier<GRID>();
        {
            const int aborted = (GRID == K_1BLK)
                                    ? abortS
                                    : (int)*(volatile unsigned int *)A.abortDev;
            if (aborted) {
                if (isFirst) { *A.ctlStatus = 1u; writeRound(A, round); }
                return;
            }
        }
        __threadfence_system();

        ringAdd<T>(A.in + offE, A.out + offE, A.rgRecv[s], lenE, tid, nth);
        barrier<GRID>();
    }

    // --------------------------- Allgather -----------------------------------
    for (int s = 0; s < R - 1; ++s) {
        const int sl = (R - 1) + s;                 // slot + flag line
        const int cs = (r + 1 - s + 2 * R) % R;     // chunk sent
        const int cr = (r - s + 2 * R) % R;         // chunk adopted
        int offS, lenS, offE, lenE;
        chunkBounds(n4, cs, R, &offS, &lenS);
        chunkBounds(n4, cr, R, &offE, &lenE);

        sendPhase((const uint4 *)(A.out + offS), A.rgSend[sl], lenS, tid, nth);
        __threadfence_system();
        barrier<GRID>();

        if (isFirst) {
            if (FLUSH) readFlush(A.rgSend[sl], lenS);
            writeU64(A.rgFlagTo[sl], round);
            __threadfence_system();

            bool ab = false;
            long long t0 = clock64();
            unsigned int probeCounter = 0u;
            while (readFlag<LA>(A.rgFlagFrom[sl]) != round) {
                if ((u64)(clock64() - t0) > A.capCycles) { ab = true; break; }
                if (A.abortHost != nullptr && ((++probeCounter & BARLINK_BAR1_HOST_MASK) == 0u)
                    && *(const volatile unsigned int *)A.abortHost != 0u) { ab = true; break; }
            }
            if (ab) {
                if (GRID == K_1BLK) abortS = 1;
                else { *(volatile unsigned int *)A.abortDev = 1u; __threadfence(); }
            }
        }
        barrier<GRID>();
        {
            const int aborted = (GRID == K_1BLK)
                                    ? abortS
                                    : (int)*(volatile unsigned int *)A.abortDev;
            if (aborted) {
                if (isFirst) { *A.ctlStatus = 1u; writeRound(A, round); }
                return;
            }
        }
        __threadfence_system();

        takeOverPhase(A.out + offE, A.rgRecv[sl], lenE, tid, nth);
        barrier<GRID>();
    }

    if (isFirst) writeRound(A, round);
}

// ---------------------------------------------------------------------------
// TOPOLOGY 'a2a' -- all_to_all_single. ONE step, ONE barrier.
//
// Rank r writes its block for rank z directly into that rank's receive
// slot, all targets in the SAME flat index space (so different warps write
// to different cards at the same time), then a barrier, then everyone reads
// its R-1 slots into the output buffer. There is no reduction and hence no
// data type: the kernel moves bytes. fp8, bf16, int32 -- the same path.
//
// WHY A DOUBLE BUFFER INSTEAD OF A SECOND BARRIER
// ------------------------------------------------
// With only (R-1) slots, the sender would not be allowed to start the next
// round before the receiver had read the previous one -- but the flag only
// says "written", not "read". Instead of a second barrier (which at MoE
// sizes would be half the latency), there are 2(R-1) slots and the round
// picks the half: par = round & 1.
//
// The proof that two halves suffice: A writes into half N%2 in round N. It
// was last used in round N-2. A's kernel for round N only starts once A's
// kernel for round N-1 is finished (one stream, in order). That one only
// finished after A had seen B's flag for round N-1. B sets that flag
// INSIDE its kernel for round N-1, which in turn only starts once B's
// kernel for round N-2 is finished -- i.e. after B has read out half
// (N-2)%2 = N%2. So the slot is free before A touches it again.
//
// The round number lives in device memory (as with mesh/ring), so the
// KERNEL decides the half, not the host -- otherwise the host would have to
// know the round and synchronize for it.
//
// Sender and receiver must pick the same half, i.e. count the same round.
// That is not an extra assumption: the flag CARRIES the round number, and
// the wait is for equality. If two ranks count differently, the flag never
// arrives and the timeout cap fires -- a wrong half cannot occur without a
// wrong flag. The failure is therefore a reported abort (ctlStatus = 1),
// never silent corruption.
//
// ALIGNMENT
// ---------
// The access width is 128 bits. Over PCIe, writes are therefore ALWAYS done
// in 16-byte packets, even at the end of a block: the last, incomplete
// packet is assembled from the available bytes in a register and issued as
// one packet. That means there are neither partial-width writes into a
// write-combining aperture, nor a read past the end of the input tensor.
// The slot begins on a page boundary and is a multiple of 16, so the
// rounded-up packet never hits the neighbor.
//
// VEC=1 means: all block offsets AND both buffer base addresses are
// 16-byte-aligned, so the bulk of the traffic runs with 128-bit accesses.
// VEC=0 is the remainder path (row width not a multiple of 16): then every
// packet is assembled byte by byte. Correct, slow, and honestly named --
// it has not been measured.
// ---------------------------------------------------------------------------
struct A2aArgs {
    const unsigned char *in;
    unsigned char       *out;
    u64          *roundDev;
    unsigned int *ctlStatus;
    unsigned int *abortDev;
    // Host-set abort word (pinned, device-mapped). nullptr = absent.
    const unsigned int *abortHost;
    int           R;
    int           rank;
    u64           capCycles;
    long long     slot;                 // slot size in bytes
    long long     sendOff[BARLINK_BAR1_MAX_RANKS];   // offset in `in`
    long long     sendLen[BARLINK_BAR1_MAX_RANKS];
    long long     recvOff[BARLINK_BAR1_MAX_RANKS];   // offset in `out`
    long long     recvLen[BARLINK_BAR1_MAX_RANKS];
    unsigned char *dstBase[BARLINK_BAR1_MAX_RANKS];  // peer's a2a region
    const unsigned char *ownBase;                 // own a2a region
    u64          *flagTo [BARLINK_BAR1_MAX_RANKS];
    const u64    *flagFrom[BARLINK_BAR1_MAX_RANKS];
};

// Assemble a 16-byte packet from at most 16 bytes. The remainder stays 0;
// it lands in the slot's overhang and is never read by the receiver.
__device__ __forceinline__ uint4 packBytes(const unsigned char *q, int n)
{
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    unsigned char *b = (unsigned char *)&v;
#pragma unroll
    for (int i = 0; i < 16; ++i) if (i < n) b[i] = q[i];
    return v;
}

// Where I (rank r) write for rank z: my position in that rank's ascending
// peer list, within half `par`. The same position formula as in mesh --
// NOT (r < z).
__device__ __forceinline__ unsigned char *a2aDst(const A2aArgs &A, int z, int par)
{
    const int p = A.rank - (A.rank > z ? 1 : 0);
    return A.dstBase[z] + (long long)(par * (A.R - 1) + p) * A.slot;
}

// Where rank s's block for me sits: its position in MY ascending peer
// list.
__device__ __forceinline__ const unsigned char *a2aSource(const A2aArgs &A,
                                                          int s, int par)
{
    const int p = s - (s > A.rank ? 1 : 0);
    return A.ownBase + (long long)(par * (A.R - 1) + p) * A.slot;
}

template<int VEC, int LA, int GRID>
__global__ void bar1_a2a_kernel(A2aArgs A)
{
    const long long tid = (GRID == K_GRID)
                              ? (long long)blockIdx.x * blockDim.x + threadIdx.x
                              : (long long)threadIdx.x;
    const long long nth = (GRID == K_GRID)
                              ? (long long)gridDim.x * blockDim.x
                              : (long long)blockDim.x;
    const bool isFirst = (tid == 0);
    const int  R = A.R, r = A.rank;
    const u64  round = *(const volatile u64 *)A.roundDev + 1ull;
    const int  par = (int)(round & 1ull);

    __shared__ int abortS;
    if (GRID == K_1BLK) {
        if (threadIdx.x == 0) abortS = 0;
        __syncthreads();
    } else if (isFirst) {
        *(volatile unsigned int *)A.abortDev = 0u;
        __threadfence();
    }

    // Everything the inner loops need with a RUNNING index goes into shared
    // memory. A dynamically indexed field in the argument struct forces
    // nvcc to copy the whole parameter block into local memory per thread --
    // at 256 threads and a half-kilobyte-sized struct that's more traffic
    // than the payload itself. The slot addresses are computed right away:
    // they only depend on (z, par), not on the packet.
    __shared__ long long sPre[BARLINK_BAR1_MAX_RANKS + 1];
    __shared__ long long ePre[BARLINK_BAR1_MAX_RANKS + 1];
    __shared__ long long sLenS[BARLINK_BAR1_MAX_RANKS];
    __shared__ long long eLenS[BARLINK_BAR1_MAX_RANKS];
    __shared__ const unsigned char *sSource[BARLINK_BAR1_MAX_RANKS];
    __shared__ unsigned char       *sDst  [BARLINK_BAR1_MAX_RANKS];
    __shared__ const unsigned char *eSource[BARLINK_BAR1_MAX_RANKS];
    __shared__ unsigned char       *eDst  [BARLINK_BAR1_MAX_RANKS];
    if (threadIdx.x == 0) {
        sPre[0] = 0; ePre[0] = 0;
        for (int z = 0; z < R; ++z) {
            sLenS[z] = A.sendLen[z];
            eLenS[z] = A.recvLen[z];
            sSource[z] = A.in + A.sendOff[z];
            eDst[z]   = A.out + A.recvOff[z];
            sDst[z]   = (z == r) ? nullptr : a2aDst(A, z, par);
            eSource[z] = (z == r) ? nullptr : a2aSource(A, z, par);
            sPre[z + 1] = sPre[z] + ((z == r) ? 0LL : (sLenS[z] + 15LL) / 16LL);
            ePre[z + 1] = ePre[z] + ((z == r) ? 0LL : (eLenS[z] + 15LL) / 16LL);
        }
    }
    __syncthreads();

    // --- 1. Send phase: all targets in the same flat index space -----------
    {
        const long long total = sPre[R];
        for (long long j = tid; j < total; j += nth) {
            int z = 0;
            while (sPre[z + 1] <= j) ++z;          // R <= 8, so a short scan
            const long long b = (j - sPre[z]) * 16LL;
            const int rest = (int)((sLenS[z] - b) < 16LL
                                       ? (sLenS[z] - b) : 16LL);
            const unsigned char *q = sSource[z] + b;
            uint4 v;
            if (VEC && rest == 16) v = *(const uint4 *)q;
            else                   v = packBytes(q, rest);
            writeV4(sDst[z] + b, v);
        }
    }

    // --- 1b. own block: purely local, no detour through the aperture -------
    {
        const long long n = sLenS[r];
        const unsigned char *q = sSource[r];
        unsigned char *z = eDst[r];
        if (VEC) {
            const long long p = n / 16LL;
            for (long long k = tid; k < p; k += nth)
                *(uint4 *)(z + k * 16LL) = *(const uint4 *)(q + k * 16LL);
            for (long long k = p * 16LL + tid; k < n; k += nth) z[k] = q[k];
        } else {
            for (long long k = tid; k < n; k += nth) z[k] = q[k];
        }
    }

    __threadfence_system();
    barrier<GRID>();

    // --- 2. The one barrier --------------------------------------------------
    if (isFirst) {
        for (int z = 0; z < R; ++z) {
            if (z == r) continue;
            writeU64(A.flagTo[z], round);
        }
        __threadfence_system();

        bool ab = false;
        long long t0 = clock64();
        unsigned int probeCounter = 0u;
        for (;;) {
            bool allArrived = true;
            for (int s = 0; s < R; ++s) {
                if (s == r) continue;
                if (readFlag<LA>(A.flagFrom[s]) != round) { allArrived = false; break; }
            }
            if (allArrived) break;
            if ((u64)(clock64() - t0) > A.capCycles) { ab = true; break; }
            if (A.abortHost != nullptr && ((++probeCounter & BARLINK_BAR1_HOST_MASK) == 0u)
                && *(const volatile unsigned int *)A.abortHost != 0u) { ab = true; break; }
        }
        if (ab) {
            if (GRID == K_1BLK) abortS = 1;
            else { *(volatile unsigned int *)A.abortDev = 1u; __threadfence(); }
        }
    }
    barrier<GRID>();
    {
        const int aborted = (GRID == K_1BLK)
                                ? abortS
                                : (int)*(volatile unsigned int *)A.abortDev;
        if (aborted) {
            if (isFirst) {
                *A.ctlStatus = 1u;
                writeRound(A, round);
            }
            return;
        }
    }
    __threadfence_system();

    // --- 3. Receive phase: own slots into the output buffer -----------------
    {
        const long long total = ePre[R];
        for (long long j = tid; j < total; j += nth) {
            int s = 0;
            while (ePre[s + 1] <= j) ++s;
            const long long b = (j - ePre[s]) * 16LL;
            const int rest = (int)((eLenS[s] - b) < 16LL
                                       ? (eLenS[s] - b) : 16LL);
            const unsigned char *q = eSource[s] + b;
            unsigned char *z = eDst[s] + b;
            if (VEC && rest == 16) {
                *(uint4 *)z = readV4(q);
            } else {
                for (int i = 0; i < rest; ++i) z[i] = readB(q + i);
            }
        }
    }
    barrier<GRID>();
    if (isFirst) writeRound(A, round);
}

// ===========================================================================
// Host side
// ===========================================================================

// Grid size for the 'grid' variant. Two caps, both necessary (unchanged
// from the probe):
//   1. Never more blocks than can be resident SIMULTANEOUSLY -- otherwise
//      grid.sync() waits on a block that isn't even running.
//   2. More blocks than there is work is pointless: ceil(n4/threads).
static int gridSize(const void *fn, int threads, int n4)
{
    int perSM = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSM, fn, threads, 0)
            != cudaSuccess || perSM < 1)
        return 1;
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return 1;
    int sms = 0;
    if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev)
            != cudaSuccess || sms < 1)
        return 1;
    int g = perSM * sms;
    int needed = (n4 + threads - 1) / threads;
    if (needed < 1) needed = 1;
    if (needed < g) g = needed;
    if (g < 1) g = 1;
    return g;
}

// Launches the chosen kernel. Flag load mode, read flush, and barrier kind
// are template arguments -- no branch is left over in the wait loops.
template<typename T>
static void start(int algo, int kernel_variant, int la, int read_flush, Bar1Args &A,
                   int threads, cudaStream_t stream)
{
#define BARLINK_BAR1_LAUNCH(KERNELFN, LA, FL)                                      \
    do {                                                                       \
        if (kernel_variant == K_GRID) {                                                \
            const void *fn = (const void *)KERNELFN<T, LA, FL, K_GRID>;        \
            int g = gridSize(fn, threads, A.n4);                          \
            dim3 gd((unsigned)g), bd((unsigned)threads);                       \
            void *args[1] = { &A };                                            \
            cudaError_t e = cudaLaunchCooperativeKernel(fn, gd, bd, args, 0,   \
                                                        stream);                \
            TORCH_CHECK(e == cudaSuccess,                                      \
                        "barlink-bar1: cudaLaunchCooperativeKernel -> ",         \
                        cudaGetErrorString(e));                                \
            return;                                                            \
        }                                                                      \
        KERNELFN<T, LA, FL, K_1BLK><<<1, threads, 0, stream>>>(A);                \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,                         \
                    "barlink-bar1: kernel launch failed");                       \
        return;                                                                \
    } while (0)

#define BARLINK_BAR1_SELECT(KERNELFN)                                              \
    do {                                                                       \
        if (la == LA_MMIO) {                                                   \
            if (read_flush) BARLINK_BAR1_LAUNCH(KERNELFN, LA_MMIO, 1);                  \
            else       BARLINK_BAR1_LAUNCH(KERNELFN, LA_MMIO, 0);                  \
        } else {                                                               \
            if (read_flush) BARLINK_BAR1_LAUNCH(KERNELFN, LA_CV, 1);                    \
            else       BARLINK_BAR1_LAUNCH(KERNELFN, LA_CV, 0);                    \
        }                                                                      \
    } while (0)

    if (algo == 0) BARLINK_BAR1_SELECT(bar1_mesh_kernel);
    else           BARLINK_BAR1_SELECT(bar1_ring_kernel);
#undef BARLINK_BAR1_SELECT
#undef BARLINK_BAR1_LAUNCH
}

// ---------------------------------------------------------------------------
// bar1_all_reduce
//
// `peer_payload` / `peer_flag` carry, per rank, THIS card's device pointer into
// the target's BAR1 region; the own entry is the local pointer to the own
// region. The slot offsets are computed here, not in Python: the same code
// that writes them into the kernel arguments also checks them.
// ---------------------------------------------------------------------------
void bar1_all_reduce(at::Tensor inp, at::Tensor out,
                     int64_t rank, int64_t world, int64_t algo,
                     std::vector<int64_t> peer_payload,
                     std::vector<int64_t> peer_flag,
                     int64_t own_payload, int64_t own_flag,
                     int64_t chunk_max, int64_t off_mesh, int64_t off_ring,
                     at::Tensor round_dev, at::Tensor ctl_dev,
                     int64_t cap_cycles, int64_t threads, int64_t kernel_variant,
                     int64_t load_shape, int64_t read_flush, int64_t abort_host)
{
    const int R = (int)world, r = (int)rank;
    TORCH_CHECK(R >= 2 && R <= BARLINK_BAR1_MAX_RANKS,
                "barlink-bar1: world ", R, " outside 2..", BARLINK_BAR1_MAX_RANKS);
    TORCH_CHECK(r >= 0 && r < R, "barlink-bar1: rank out of range");
    TORCH_CHECK(algo == 0 || algo == 1, "barlink-bar1: algo 0=mesh 1=ring");
    TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(),
                "barlink-bar1: only contiguous tensors");
    TORCH_CHECK(inp.numel() == out.numel() && inp.scalar_type() == out.scalar_type(),
                "barlink-bar1: in and out do not match");
    TORCH_CHECK(inp.data_ptr() != out.data_ptr(),
                "barlink-bar1: in and out must not be the same -- the ring "
                "still reads 'in' while it writes 'out'");
    TORCH_CHECK((int64_t)peer_payload.size() == world &&
                (int64_t)peer_flag.size() == world,
                "barlink-bar1: peer table has the wrong length");

    const size_t nbytes = (size_t)inp.numel() * (size_t)inp.element_size();
    TORCH_CHECK(nbytes % 16 == 0,
                "barlink-bar1: payload ", nbytes, " is not a multiple of 16 "
                "bytes -- the access width is 128 bits");
    TORCH_CHECK(((uintptr_t)inp.data_ptr() % 16) == 0 &&
                ((uintptr_t)out.data_ptr() % 16) == 0,
                "barlink-bar1: buffer not aligned to 16 bytes");
    const int n4 = (int)(nbytes / 16);
    TORCH_CHECK(n4 >= R, "barlink-bar1: fewer than one 128-bit packet per rank");

    // Seam check: the LARGEST chunk must fit in a slot. Deliberately here
    // and not only in Python -- computed with chunkBounds itself, not with
    // a second version of the formula.
    {
        int maxLen = 0;
        for (int j = 0; j < R; ++j) {
            int off, len;
            chunkBounds(n4, j, R, &off, &len);
            if (len > maxLen) maxLen = len;
        }
        TORCH_CHECK((int64_t)maxLen * 16 <= chunk_max,
                    "barlink-bar1: largest chunk ", (int64_t)maxLen * 16,
                    " bytes does not fit in the slot of ", chunk_max,
                    " bytes. The caller should have asked handles().");
    }

    Bar1Args A;
    std::memset(&A, 0, sizeof(A));
    A.in           = (const uint4 *)inp.data_ptr();
    A.out          = (uint4 *)out.data_ptr();
    A.roundDev     = (u64 *)round_dev.data_ptr();
    A.ctlStatus    = (unsigned int *)ctl_dev.data_ptr();
    A.abortDev   = ((unsigned int *)ctl_dev.data_ptr()) + 1;
    // 0 means the host could not map the abort word; the kernels then see
    // nullptr and skip the probe entirely.
    A.abortHost  = (const unsigned int *)(uintptr_t)abort_host;
    A.n4           = n4;
    A.R            = R;
    A.rank         = r;
    A.capCycles = (u64)cap_cycles;

    const int steps_mesh = 2;
    const int steps_ring = 2 * (R - 1);
    // FSLOT(topo, step, sender) = FBASE[topo] + (step*R + sender)*256,
    // FBASE = { 0, 2*R*256 }. Unchanged from the probe, just without the
    // two additional topologies it also measured.
    const size_t fbase_mesh = 0;
    const size_t fbase_ring = (size_t)steps_mesh * (size_t)R * 256u;
#define FSLOT(BASE, STEP, SENDER) \
    ((BASE) + (size_t)((STEP) * R + (SENDER)) * 256u)

    if (algo == 0) {
        for (int z = 0; z < R; ++z) {
            if (z == r) continue;
            // My position in the ascending peer list of receiver z.
            // NOT (r < z) -- see the derivation in the probe.
            const size_t p = (size_t)(r - (r > z ? 1 : 0));
            char *zb = (char *)(uintptr_t)peer_payload[z];
            A.nzSendRS[z] = (uint4 *)(zb + off_mesh +
                                      (size_t)(0 * (R - 1) + (int)p) * chunk_max);
            A.nzSendAG[z] = (uint4 *)(zb + off_mesh +
                                      (size_t)(1 * (R - 1) + (int)p) * chunk_max);
        }
        {
            char *mb = (char *)(uintptr_t)own_payload;
            for (int s = 0; s < R; ++s) {
                if (s == r) continue;
                const size_t p = (size_t)(s - (s > r ? 1 : 0));
                A.nzRecvRS[s] = (const uint4 *)(mb + off_mesh +
                                  (size_t)(0 * (R - 1) + (int)p) * chunk_max);
                A.nzRecvAG[s] = (const uint4 *)(mb + off_mesh +
                                  (size_t)(1 * (R - 1) + (int)p) * chunk_max);
            }
        }
        for (int ph = 0; ph < 2; ++ph) {
            for (int s = 0; s < R; ++s) {
                if (s == r) continue;
                A.nzFlagTo[ph][s] = (u64 *)((char *)(uintptr_t)peer_flag[s] +
                                            FSLOT(fbase_mesh, ph, r));
                A.nzFlagFrom[ph][s] = (const u64 *)((char *)(uintptr_t)own_flag +
                                                   FSLOT(fbase_mesh, ph, s));
            }
        }
    } else {
        const int next_rank = (r + 1) % R;
        const int prev_rank  = (r + R - 1) % R;
        char *zb = (char *)(uintptr_t)peer_payload[next_rank];
        char *mb = (char *)(uintptr_t)own_payload;
        for (int s = 0; s < steps_ring; ++s) {
            A.rgSend[s]    = (uint4 *)(zb + off_ring + (size_t)s * chunk_max);
            A.rgRecv[s]    = (const uint4 *)(mb + off_ring + (size_t)s * chunk_max);
            A.rgFlagTo[s]  = (u64 *)((char *)(uintptr_t)peer_flag[next_rank] +
                                     FSLOT(fbase_ring, s, r));
            A.rgFlagFrom[s] = (const u64 *)((char *)(uintptr_t)own_flag +
                                           FSLOT(fbase_ring, s, prev_rank));
        }
    }
#undef FSLOT

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    const int kernelI = (int)kernel_variant, laI = (int)load_shape, flushI = (int)read_flush;
    switch (inp.scalar_type()) {
        case at::kFloat:
            start<float>((int)algo, kernelI, laI, flushI, A, (int)threads, stream);
            break;
        case at::kHalf:
            start<__half>((int)algo, kernelI, laI, flushI, A, (int)threads, stream);
            break;
        case at::kBFloat16:
            start<__nv_bfloat16>((int)algo, kernelI, laI, flushI, A, (int)threads,
                                  stream);
            break;
        default:
            TORCH_CHECK(false, "barlink-bar1: data type ", inp.scalar_type(),
                        " is not supported (float32/float16/bfloat16)");
    }
}

// ---------------------------------------------------------------------------
// bar1_all_to_all
//
// No data type, no reduction, no templating over element types: the kernel
// moves bytes. What comes in are byte offsets and byte lengths per rank --
// so the same function carries both the evenly-split and the unevenly-split
// form (input_split_sizes/output_split_sizes), and fp8 is simply a byte.
//
// `peer_payload` is the SAME peer pointer table as in bar1_all_reduce; the a2a
// region sits as a third section in the same receive region (offset
// `off_a2a`). Nothing is mapped here -- that happened during setup, and
// only there.
// ---------------------------------------------------------------------------
static void startA2a(int vec, int kernel_variant, int la, A2aArgs &A, int threads,
                      long long packets, cudaStream_t stream)
{
#define BARLINK_A2A_LAUNCH(VEC, LA)                                              \
    do {                                                                       \
        if (kernel_variant == K_GRID) {                                                \
            const void *fn = (const void *)bar1_a2a_kernel<VEC, LA, K_GRID>; \
            int n4 = (int)(packets > 2147483647LL ? 2147483647LL : packets);     \
            int g = gridSize(fn, threads, n4);                            \
            dim3 gd((unsigned)g), bd((unsigned)threads);                       \
            void *args[1] = { &A };                                            \
            cudaError_t e = cudaLaunchCooperativeKernel(fn, gd, bd, args, 0,   \
                                                        stream);                \
            TORCH_CHECK(e == cudaSuccess,                                      \
                        "barlink-bar1 a2a: cudaLaunchCooperativeKernel -> ",     \
                        cudaGetErrorString(e));                                \
            return;                                                            \
        }                                                                      \
        bar1_a2a_kernel<VEC, LA, K_1BLK><<<1, threads, 0, stream>>>(A);         \
        TORCH_CHECK(cudaGetLastError() == cudaSuccess,                         \
                    "barlink-bar1 a2a: kernel launch failed");                   \
        return;                                                                \
    } while (0)

    if (vec) {
        if (la == LA_MMIO) BARLINK_A2A_LAUNCH(1, LA_MMIO);
        else               BARLINK_A2A_LAUNCH(1, LA_CV);
    } else {
        if (la == LA_MMIO) BARLINK_A2A_LAUNCH(0, LA_MMIO);
        else               BARLINK_A2A_LAUNCH(0, LA_CV);
    }
#undef BARLINK_A2A_LAUNCH
}

void bar1_all_to_all(at::Tensor inp, at::Tensor out,
                     int64_t rank, int64_t world,
                     std::vector<int64_t> send_off,
                     std::vector<int64_t> send_len,
                     std::vector<int64_t> recv_off,
                     std::vector<int64_t> recv_len,
                     std::vector<int64_t> peer_payload,
                     std::vector<int64_t> peer_flag,
                     int64_t own_payload, int64_t own_flag,
                     int64_t slot, int64_t off_a2a, int64_t fbase_a2a,
                     at::Tensor round_dev, at::Tensor ctl_dev,
                     int64_t cap_cycles, int64_t threads, int64_t kernel_variant,
                     int64_t load_shape, int64_t abort_host)
{
    const int R = (int)world, r = (int)rank;
    TORCH_CHECK(R >= 2 && R <= BARLINK_BAR1_MAX_RANKS,
                "barlink-bar1 a2a: world ", R, " outside 2..",
                BARLINK_BAR1_MAX_RANKS);
    TORCH_CHECK(r >= 0 && r < R, "barlink-bar1 a2a: rank out of range");
    TORCH_CHECK(inp.is_contiguous() && out.is_contiguous(),
                "barlink-bar1 a2a: only contiguous tensors");
    TORCH_CHECK(inp.scalar_type() == out.scalar_type(),
                "barlink-bar1 a2a: in and out have different data types -- "
                "all_to_all does not convert anything");
    TORCH_CHECK(inp.data_ptr() != out.data_ptr(),
                "barlink-bar1 a2a: in and out must not be the same");
    TORCH_CHECK((int64_t)send_off.size() == world &&
                (int64_t)send_len.size() == world &&
                (int64_t)recv_off.size() == world &&
                (int64_t)recv_len.size() == world &&
                (int64_t)peer_payload.size() == world &&
                (int64_t)peer_flag.size() == world,
                "barlink-bar1 a2a: one of the tables has the wrong length");
    TORCH_CHECK(slot > 0 && (slot % 16) == 0,
                "barlink-bar1 a2a: slot size ", slot,
                " is not a positive multiple of 16");

    const int64_t in_bytes  = (int64_t)inp.numel()  * (int64_t)inp.element_size();
    const int64_t out_bytes = (int64_t)out.numel() * (int64_t)out.element_size();

    // The seam check. Deliberately HERE and not only in Python: the slot
    // boundary is the condition the mapping actually depends on, and it is
    // checked with the same numbers used for the computation -- not with a
    // second version of the same formula.
    int64_t max_send = 0;
    for (int z = 0; z < R; ++z) {
        TORCH_CHECK(send_len[z] >= 0 && recv_len[z] >= 0 &&
                    send_off[z] >= 0 && recv_off[z] >= 0,
                    "barlink-bar1 a2a: negative split size at rank ", z);
        TORCH_CHECK(send_off[z] + send_len[z] <= in_bytes,
                    "barlink-bar1 a2a: send block ", z, " (", send_off[z], "+",
                    send_len[z], ") lies past the end of the input tensor (",
                    in_bytes, " bytes)");
        TORCH_CHECK(recv_off[z] + recv_len[z] <= out_bytes,
                    "barlink-bar1 a2a: receive block ", z, " (", recv_off[z], "+",
                    recv_len[z], ") lies past the end of the output tensor (",
                    out_bytes, " bytes)");
        if (z == r) continue;
        if (send_len[z] > max_send) max_send = send_len[z];
        TORCH_CHECK(recv_len[z] <= slot,
                    "barlink-bar1 a2a: receive block from rank ", z, " is ",
                    recv_len[z], " bytes and does not fit in the slot of ",
                    slot, " bytes. The caller should have asked handles() "
                    "or supports_a2a().");
    }
    TORCH_CHECK(max_send <= slot,
                "barlink-bar1 a2a: largest send block ", max_send,
                " bytes does not fit in the slot of ", slot, " bytes.");
    TORCH_CHECK(send_len[r] == recv_len[r],
                "barlink-bar1 a2a: the own block is ",
                send_len[r], " bytes when sending and ", recv_len[r],
                " bytes when receiving -- the split sizes do not match");

    A2aArgs A;
    std::memset(&A, 0, sizeof(A));
    A.in           = (const unsigned char *)inp.data_ptr();
    A.out          = (unsigned char *)out.data_ptr();
    A.roundDev     = (u64 *)round_dev.data_ptr();
    A.ctlStatus    = (unsigned int *)ctl_dev.data_ptr();
    A.abortDev   = ((unsigned int *)ctl_dev.data_ptr()) + 1;
    // 0 means the host could not map the abort word; nullptr disables the
    // probe and leaves only the cycle deadline.
    A.abortHost  = (const unsigned int *)(uintptr_t)abort_host;
    A.R            = R;
    A.rank         = r;
    A.capCycles = (u64)cap_cycles;
    A.slot      = (long long)slot;
    A.ownBase   = (const unsigned char *)((char *)(uintptr_t)own_payload
                                             + off_a2a);

    // VEC only when EVERYTHING is aligned: both base addresses and every
    // block offset. A single misaligned offset makes the 128-bit load
    // instruction invalid, and there is no such thing as "mostly aligned".
    int vec = (((uintptr_t)inp.data_ptr() % 16) == 0 &&
               ((uintptr_t)out.data_ptr() % 16) == 0) ? 1 : 0;
    long long packets = 0;
    for (int z = 0; z < R; ++z) {
        A.sendOff[z] = (long long)send_off[z];
        A.sendLen[z] = (long long)send_len[z];
        A.recvOff[z] = (long long)recv_off[z];
        A.recvLen[z] = (long long)recv_len[z];
        if ((send_off[z] % 16) || (recv_off[z] % 16)) vec = 0;
        if (z != r) packets += (A.sendLen[z] + 15LL) / 16LL;
    }
    for (int z = 0; z < R; ++z) {
        if (z == r) continue;
        A.dstBase[z] = (unsigned char *)((char *)(uintptr_t)peer_payload[z]
                                           + off_a2a);
        // One flag line per (step=0, sender). I write into MY line at the
        // receiver and read THEIR line locally.
        A.flagTo[z]  = (u64 *)((char *)(uintptr_t)peer_flag[z] +
                               fbase_a2a + (size_t)r * 256u);
        A.flagFrom[z] = (const u64 *)((char *)(uintptr_t)own_flag +
                                     fbase_a2a + (size_t)z * 256u);
    }

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    startA2a(vec, (int)kernel_variant, (int)load_shape, A, (int)threads, packets, stream);
}
"""

_CPP_SRC = """
void bar1_all_reduce(at::Tensor inp, at::Tensor out,
                     int64_t rank, int64_t world, int64_t algo,
                     std::vector<int64_t> peer_payload,
                     std::vector<int64_t> peer_flag,
                     int64_t own_payload, int64_t own_flag,
                     int64_t chunk_max, int64_t off_mesh, int64_t off_ring,
                     at::Tensor round_dev, at::Tensor ctl_dev,
                     int64_t cap_cycles, int64_t threads, int64_t kernel_variant,
                     int64_t load_shape, int64_t read_flush, int64_t abort_host);

void bar1_all_to_all(at::Tensor inp, at::Tensor out,
                     int64_t rank, int64_t world,
                     std::vector<int64_t> send_off,
                     std::vector<int64_t> send_len,
                     std::vector<int64_t> recv_off,
                     std::vector<int64_t> recv_len,
                     std::vector<int64_t> peer_payload,
                     std::vector<int64_t> peer_flag,
                     int64_t own_payload, int64_t own_flag,
                     int64_t slot, int64_t off_a2a, int64_t fbase_a2a,
                     at::Tensor round_dev, at::Tensor ctl_dev,
                     int64_t cap_cycles, int64_t threads, int64_t kernel_variant,
                     int64_t load_shape, int64_t abort_host);
"""


# ===========================================================================
# dma-buf export via the RM ioctls
# ===========================================================================

_DMABUF_SRC = r"""
#include <torch/extension.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

// UAPI of the open-source NVIDIA kernel modules. The include paths are
// passed in at compile time; if they're missing, this extension isn't
// built in the first place and the transport declines.
#include <nvtypes.h>
#include <nvos.h>
#include <nv-ioctl.h>
#include <nv_escape.h>
#include <class/cl0000.h>
#include <class/cl0080.h>
#include <ctrl/ctrl0000/ctrl0000gpu.h>
#include <ctrl/ctrl0000/ctrl0000unix.h>

static int nvIoctl(int fd, int nr, void *p, size_t size)
{
    return ioctl(fd, _IOC(_IOC_READ | _IOC_WRITE, NV_IOCTL_MAGIC, nr,
                          (unsigned)size), p);
}

// Ported from sonden/dmabuf_p2p_probe.cpp::nvExportToDmabuf().
//
// Return value: { dmabuf_fd, ctl_fd, dev_fd }. The last two MUST stay open
// as long as the dma-buf is in use: ctl_fd is where the RM client that owns
// the imported memory object hangs off of. If it's closed, RM frees the
// object and the dma-buf points into nothing. That's why this function
// hands them out instead of silently leaking them -- the caller holds them
// and closes them on teardown.
//
// objfd = fd from cuMemExportToShareableHandle(POSIX_FILE_DESCRIPTOR)
// pci_bus = CU_DEVICE_ATTRIBUTE_PCI_BUS_ID of the EXPORTING card
std::vector<int64_t> bar1_export_dmabuf(int64_t objfd, int64_t pci_bus,
                                        int64_t size)
{
    const NvHandle hDevice = 0xbee00001;
    const NvHandle hMemory = 0xbee00010;
    NvHandle hClient = 0;
    int ctlFd = -1, devFd = -1;
    char errmsg[256] = {0};

#define NVX_FAIL(...)                                                          \
    do {                                                                       \
        std::snprintf(errmsg, sizeof(errmsg), __VA_ARGS__);                    \
        if (devFd >= 0) close(devFd);                                          \
        if (ctlFd >= 0) close(ctlFd);                                          \
        TORCH_CHECK(false, "barlink-bar1 dma-buf export: ", errmsg);             \
    } while (0)

    ctlFd = open("/dev/nvidiactl", O_RDWR);
    if (ctlFd < 0) NVX_FAIL("open(/dev/nvidiactl): %s", std::strerror(errno));

    {   // Version handshake
        nv_ioctl_rm_api_version_t v;
        char buf[256] = {0}, ver[64] = {0};
        std::memset(&v, 0, sizeof(v));
        v.cmd = NV_RM_API_VERSION_CMD_RELAXED;
        FILE *f = std::fopen("/proc/driver/nvidia/version", "r");
        if (f) { if (!std::fgets(buf, sizeof(buf), f)) buf[0] = 0; std::fclose(f); }
        char *p = std::strstr(buf, "for x86_64");
        if (p && std::sscanf(p, "for x86_64 %63s", ver) == 1)
            std::strncpy(v.versionString, ver, sizeof(v.versionString) - 1);
        if (nvIoctl(ctlFd, NV_ESC_CHECK_VERSION_STR, &v, sizeof(v)) < 0)
            NVX_FAIL("NV_ESC_CHECK_VERSION_STR: %s", std::strerror(errno));
    }

    NvU32 gpuId = 0, minor = 0;
    {
        nv_ioctl_card_info_t ci[32];
        std::memset(ci, 0, sizeof(ci));
        if (nvIoctl(ctlFd, NV_ESC_CARD_INFO, ci, sizeof(ci)) < 0)
            NVX_FAIL("NV_ESC_CARD_INFO: %s", std::strerror(errno));
        bool found = false;
        for (int i = 0; i < 32; ++i) {
            if (!ci[i].valid) continue;
            if ((int64_t)ci[i].pci_info.bus == pci_bus) {
                gpuId = ci[i].gpu_id;
                minor = ci[i].minor_number;
                found = true;
            }
        }
        if (!found)
            NVX_FAIL("PCI bus 0x%02x not found in NV_ESC_CARD_INFO",
                     (unsigned)pci_bus);
    }

    {   // own RM client
        NVOS21_PARAMETERS a;
        std::memset(&a, 0, sizeof(a));
        a.hClass = NV01_ROOT;
        if (nvIoctl(ctlFd, NV_ESC_RM_ALLOC, &a, sizeof(a)) < 0 || a.status != 0)
            NVX_FAIL("RM_ALLOC(root) status=0x%x", (unsigned)a.status);
        hClient = a.hObjectNew;
    }

    NvU32 devInst = 0;
    {
        NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS p;
        NVOS54_PARAMETERS c;
        std::memset(&p, 0, sizeof(p)); std::memset(&c, 0, sizeof(c));
        p.gpuId   = gpuId;
        c.hClient = hClient; c.hObject = hClient;
        c.cmd     = NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2;
        c.params  = (NvP64)(uintptr_t)&p; c.paramsSize = sizeof(p);
        if (nvIoctl(ctlFd, NV_ESC_RM_CONTROL, &c, sizeof(c)) < 0 || c.status != 0)
            NVX_FAIL("GPU_GET_ID_INFO_V2 status=0x%x", (unsigned)c.status);
        devInst = p.deviceInstance;
    }

    {
        NV0080_ALLOC_PARAMETERS dp;
        NVOS21_PARAMETERS a;
        std::memset(&dp, 0, sizeof(dp)); std::memset(&a, 0, sizeof(a));
        dp.deviceId = devInst; dp.hClientShare = hClient;
        a.hRoot = hClient; a.hObjectParent = hClient; a.hObjectNew = hDevice;
        a.hClass = NV01_DEVICE_0;
        a.pAllocParms = (NvP64)(uintptr_t)&dp;
        if (nvIoctl(ctlFd, NV_ESC_RM_ALLOC, &a, sizeof(a)) < 0 || a.status != 0)
            NVX_FAIL("RM_ALLOC(device) status=0x%x", (unsigned)a.status);
    }

    {   // import the object fd into our own client
        NV0000_CTRL_OS_UNIX_IMPORT_OBJECT_FROM_FD_PARAMS p;
        NVOS54_PARAMETERS c;
        std::memset(&p, 0, sizeof(p)); std::memset(&c, 0, sizeof(c));
        p.fd = (NvS32)objfd;
        p.object.type = NV0000_CTRL_OS_UNIX_EXPORT_OBJECT_TYPE_RM;
        p.object.data.rmObject.hDevice = hDevice;
        p.object.data.rmObject.hParent = hDevice;
        p.object.data.rmObject.hObject = hMemory;
        c.hClient = hClient; c.hObject = hClient;
        c.cmd     = NV0000_CTRL_CMD_OS_UNIX_IMPORT_OBJECT_FROM_FD;
        c.params  = (NvP64)(uintptr_t)&p; c.paramsSize = sizeof(p);
        if (nvIoctl(ctlFd, NV_ESC_RM_CONTROL, &c, sizeof(c)) < 0 || c.status != 0)
            NVX_FAIL("IMPORT_OBJECT_FROM_FD status=0x%x", (unsigned)c.status);
    }

    int dmabufFd = -1;
    {
        char devpath[64];
        std::snprintf(devpath, sizeof(devpath), "/dev/nvidia%u", (unsigned)minor);
        devFd = open(devpath, O_RDWR);
        if (devFd < 0) NVX_FAIL("open(%s): %s", devpath, std::strerror(errno));

        nv_ioctl_export_to_dma_buf_fd_t p;
        std::memset(&p, 0, sizeof(p));
        p.fd           = -1;               // -1 = create a new dma-buf
        p.hClient      = hClient;
        p.totalObjects = 1;
        p.numObjects   = 1;
        p.index        = 0;
        p.totalSize    = (NvU64)size;
        p.mappingType  = NV_DMABUF_EXPORT_MAPPING_TYPE_DEFAULT;
        p.bAllowMmap   = NV_FALSE;
        p.handles[0]   = hMemory;
        p.offsets[0]   = 0;
        p.sizes[0]     = (NvU64)size;

        if (nvIoctl(devFd, NV_ESC_EXPORT_TO_DMABUF_FD, &p, sizeof(p)) < 0)
            NVX_FAIL("NV_ESC_EXPORT_TO_DMABUF_FD: %s", std::strerror(errno));
        if (p.status != 0)
            NVX_FAIL("NV_ESC_EXPORT_TO_DMABUF_FD status=0x%08x",
                     (unsigned)p.status);
        dmabufFd = p.fd;
    }
#undef NVX_FAIL

    std::vector<int64_t> out;
    out.push_back((int64_t)dmabufFd);
    out.push_back((int64_t)ctlFd);
    out.push_back((int64_t)devFd);
    return out;
}
"""

# Deliberately NO separate declaration block like for the CUDA extension:
# `load_inline` concatenates cpp_sources in the order passed in, and a
# declaration with std::vector BEFORE the #includes won't compile. The
# definition alone is enough, pybind needs nothing more.


# ===========================================================================
# Compiling
# ===========================================================================


def nv_include_paths() -> Optional[list]:
    """The three include directories of the driver tree, or ``None``.

    ``None`` means: the tree isn't present here. No guessing and no bundled
    copy of the UAPI structures -- their field offsets are version-bound,
    and a wrongly guessed struct sends garbage into a kernel ioctl.
    """
    root = os.environ.get("SGLANG_BARLINK_BAR1_NV_SOURCE", NV_SOURCE_DEFAULT)
    p = pathlib.Path(root)
    paths = [p / t for t in NV_INCLUDES]
    missing = [str(x) for x in paths if not x.is_dir()]
    if missing:
        return None
    # A header spot-check: the directory can exist and still be the wrong
    # tree.
    if not (paths[1] / "nvos.h").is_file():
        return None
    return [str(x) for x in paths]


def load_collective_ext(cpu_group):
    """The CUDA extension with the ``mesh`` and ``ring`` kernels.

    Cache hygiene, arch resolution, and the build tag come from
    ``barlink_device``: same mechanism, own name. The name carries vendor and
    arches, because torch keys its build directory only by the name --
    without that, a rank could be handed a .so its card cannot run.
    """
    global _ext
    if _ext is not None:
        return _ext

    from torch.utils.cpp_extension import load_inline

    from sglang.srt.distributed.device_communicators.barlink_device import (
        _build_flags,
        _ext_cache_guarded,
        _local_vendor,
        _resolve_build_arches,
    )

    vendor = _local_vendor()
    if vendor != "cuda":
        raise RuntimeError(
            f"barlink-BAR1: the direct path is NVIDIA-specific (BAR1 aperture, "
            f"dma-buf export via the RM ioctls, PTX inline asm such as "
            f"ld.mmio.relaxed.sys). This rank reports {vendor!r}. There is "
            f"no hipify translation of these lines, and faking one would be "
            f"worse than declining."
        )
    by_vendor = _resolve_build_arches(cpu_group)
    arches = by_vendor.get(vendor, [])
    flags = _build_flags(vendor, arches)
    # NO -rdc=true and no -lcudadevrt: the probe compiles the same
    # cooperative-groups calls without either (build line in the header of
    # bar1_kollektiv.cu). Separate compilation would need device linking
    # here, which load_inline doesn't do -- and it isn't needed.
    name = "barlink_bar1_ext_" + vendor
    if arches:
        name += "_" + "_".join(a.replace(".", "") for a in arches)
    t0 = time.time()
    with _ext_cache_guarded(name) as build_dir:
        _ext = load_inline(
            name=name,
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=["bar1_all_reduce", "bar1_all_to_all"],
            extra_cuda_cflags=flags or None,
            verbose=False,
            build_directory=str(build_dir) if build_dir is not None else None,
        )
    logger.info(
        "barlink-BAR1: collective extension %r built for arches %s in %.1f s.",
        name, ",".join(arches) or "<torch default>", time.time() - t0,
    )
    return _ext


def load_dmabuf_ext():
    """The C++ extension for the dma-buf export, or ``None``.

    ``None`` with a logged reason if the driver tree is missing. The caller
    tries ``cuMemGetHandleForAddressRange`` first; only if that fails -- on
    GeForce the driver reports
    ``CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 0`` -- is this path needed.
    """
    global _dmabuf_ext, _dmabuf_reason
    if _dmabuf_ext is not None:
        return _dmabuf_ext
    if _dmabuf_reason:
        return None

    paths = nv_include_paths()
    if paths is None:
        _dmabuf_reason = (
            f"The headers of the open-source NVIDIA kernel modules are not "
            f"under "
            f"{os.environ.get('SGLANG_BARLINK_BAR1_NV_SOURCE', NV_SOURCE_DEFAULT)!r} "
            f"(expected: {', '.join(NV_INCLUDES)}). Without them, "
            f"NV_ESC_EXPORT_TO_DMABUF_FD cannot be called. Set the path via "
            f"SGLANG_BARLINK_BAR1_NV_SOURCE."
        )
        logger.info("barlink-BAR1: %s", _dmabuf_reason)
        return None

    from torch.utils.cpp_extension import load_inline

    from sglang.srt.distributed.device_communicators.barlink_device import (
        _ext_cache_guarded,
    )

    name = "barlink_bar1_dmabuf_ext"
    t0 = time.time()
    try:
        with _ext_cache_guarded(name) as build_dir:
            _dmabuf_ext = load_inline(
                name=name,
                cpp_sources=_DMABUF_SRC,
                functions=["bar1_export_dmabuf"],
                extra_include_paths=paths,
                verbose=False,
                build_directory=str(build_dir) if build_dir is not None else None,
            )
    except Exception as e:                     # a compile error is itself a reason
        _dmabuf_reason = f"Compilation failed: {e}"
        logger.info("barlink-BAR1: dma-buf extension not built -- %s", e)
        return None
    logger.info(
        "barlink-BAR1: dma-buf extension built in %.1f s (driver tree %s).",
        time.time() - t0, paths[0],
    )
    return _dmabuf_ext


def dmabuf_reason() -> str:
    return _dmabuf_reason
