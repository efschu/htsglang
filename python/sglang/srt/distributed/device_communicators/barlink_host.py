# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""barlink host transport — pinned, portable host memory, driven by the GPU.

WHY THIS EXISTS
===============
Measured on this rig, same hardware, same link, 20 KiB ping-pong:

    plain pinned-host path      7.30 us
    NCCL                       37.41 us

Factor 5.1, and there is no other transport in between — the difference is
NCCL's protocol and launch overhead, nothing else. The gain needs no driver
patch, holds for every size and every link, and is therefore worth more than
any direct-path work. This module is that plain path, made into a transport.

WHAT IT IS
==========
One POSIX shared-memory segment per group, mapped by every rank and
page-locked with each rank's OWN runtime as PORTABLE + MAPPED
(``cudaHostRegister`` / ``hipHostRegister``, flags 1|2). Portable means every
CUDA context of the process treats it as pinned; mapped means the GPU gets a
device address for it, so kernels load and store the segment DIRECTLY over
PCIe. Each process registering with its own runtime is what keeps the
transport vendor-neutral (an NVIDIA and an AMD rank can share one segment).

The whole collective is two kernels, both stream-ordered, neither touching
the host:

    K1 "put"      store the payload into this rank's mapped slot,
                  __threadfence_system(), publish "slot holds op #seq"
    K2 "get"      spin on the peers' publish flags, then read their slots
                  straight out of host memory and combine into the result,
                  publish "op #seq consumed", bump the device-side seq

That is the entire data plane. No ``cudaMemcpyAsync``, no second stream, no
device scratch, no chunk pipeline, no startup calibration — the leanness IS
the feature, because the 7.3 us number is a latency number and every extra
launch shows up in it whole.

FIVE PROPERTIES THAT ARE LOAD-BEARING
=====================================
1. **Buffers are allocated once.** The segment, the sequence counters and the
   block counter are built in ``__init__`` and never grow. The hot path
   allocates exactly one thing: the out-of-place result tensor, from torch's
   caching allocator (no ``cudaMalloc``). It must be a FRESH tensor —
   returning a shape-keyed cache once corrupted the model forward outright,
   see ``BarlinkCommunicator._get_out_buf``.
2. **No host synchronization anywhere.** Completion is reported through u64
   flags in the pinned segment, read by the peers' kernels. There is no
   ``cudaStreamSynchronize`` in the hot path, so the ops are stream-ordered
   like NCCL's and are CUDA-graph capturable: the per-op sequence number
   lives in DEVICE memory and is incremented by the kernel itself, so a
   replay advances it exactly as the first run did.
3. **Slots are double-buffered by ``seq & 1``.** That is what removes the
   third kernel (``barlink_device`` needs a separate "begin" kernel to wait for
   the peers to release the single slot). The put kernel still checks the
   consumption flags — but for op ``seq-2``, which in steady state is long
   done, so the check is a few loads and never a spin.
4. **Cycle deadline instead of a hang.** Every spin is bounded by
   ``clock64()``; on expiry the kernel writes an abort code into ``seq_dev[1]``
   and RETURNS, and :meth:`BarlinkHostTransport.check_aborted` turns that word
   into a structured ``HostCollectiveAborted`` on the host — exactly as in
   ``barlink_device`` (#583) and the BAR1 transport (#431 fix 2). It must
   never ``__trap()``: see the comment on ``wait_ge``. Deadlock safety
   otherwise rests on the same invariant NCCL kernels use: all ranks issue the
   same sequence of collectives (SPMD).
5. **Point-to-point too.** ``send``/``recv`` use per-ordered-pair buffers and
   per-pair sequence counters, so they do not have to be group-uniform. The
   communicator in ``barlink.py`` does not dispatch p2p today; the methods are
   part of this transport's own API and are what the ping-pong number above
   actually measures.

The CUDA below sticks to constructs hipify translates 1:1
(``__threadfence_system``, ``clock64``, volatile loads, ``__shared__``,
``__syncthreads``), so the same file serves a ROCm rank. There is deliberately
no device-side trap of any kind left to translate (#653).

NOT A "HOST-STAGED" TRANSPORT
=============================
The name says where the BYTES live, not who drives. ``shm``/``gloo``/``ucx``
are host-staged in the sense ``parallel_state._enforce_cpu_transport_needs_
eager`` cares about — they block the calling thread and are illegal inside a
capture. This one never does, which is why it is listed in
``CAPTURABLE_BARLINK_TRANSPORTS`` next to ``device``.
"""

import ctypes
import logging
import os
import time
from multiprocessing import shared_memory

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators import (
    barlink_abort_gate,
    barlink_liveness,
)

logger = logging.getLogger(__name__)

_ONCE_SEEN: set = set()


def _info_once(msg: str, *args) -> None:
    """sglang's logger has no `info_once`."""
    if msg in _ONCE_SEEN:
        return
    _ONCE_SEEN.add(msg)
    logger.info(msg, *args)


# ---------------------------------------------------------------------------
# Tunables. RANK-UNIFORM, like every other SGLANG_BARLINK* knob: the ranks agree
# on a flag protocol and on the slot geometry, so a rank that reads a
# different value does not return a wrong answer, it hangs until the cycle
# deadline fires and the abort word is raised on the host.
# ---------------------------------------------------------------------------

#: Per-rank staging slot (MiB). Unset -> inherit SGLANG_BARLINK_SLOT_MIB, which
#: the factory in barlink.py passes in, so there is one knob for the common case
#: and a second only when the host transport should differ. TWO slots per rank
#: are allocated (double buffering), so the segment is 2 x world x this.
_SLOT_MIB_ENV = "SGLANG_BARLINK_HOST_SLOT_MIB"
#: Per-ordered-pair send/recv buffer (MiB), also double-buffered. 0 disables
#: point-to-point entirely (`handles("send", ...)` then answers False instead
#: of the transport quietly not having a buffer).
_P2P_BYTES = int(os.environ.get("SGLANG_BARLINK_HOST_P2P_MIB", "4")) * 1024 * 1024
#: Grid width of the data kernels. The payloads this transport is for are
#: latency-bound; more blocks buy nothing below ~1 MiB and cost tail latency.
_BLOCKS = int(os.environ.get("SGLANG_BARLINK_HOST_BLOCKS", "32"))

_MAX_RANKS = 8
#: u64 flag columns per rank. Layout of one row (rank r):
#:   0                      collective publish   ("my slot holds op #seq")
#:   1                      collective consume   ("I read every slot of #seq")
#:   2 + d                  p2p publish, r -> d
#:   2 + _MAX_RANKS + s     p2p consume, s -> r
#: 32 columns = 256 B per row, so two ranks never share a cache line.
_COLS = 32
#: Flags live at offset 0; slots start after this. 4096 >= 8 * 32 * 8 B.
_HEADER_BYTES = 4096


_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define BARLINK_HOST_MAX_RANKS 8
#define BARLINK_HOST_COLS 32

using u64 = unsigned long long;

// Device-visible addresses of the mapped host slots, rank-major and
// parity-minor: p[r * 2 + (seq & 1)]. Passed BY VALUE because the parity is
// only known on the device -- the host cannot pick the pointer, so it hands
// over both and the kernel chooses. That is exactly what keeps the schedule
// CUDA-graph capturable.
struct HostSlots { char* p[2 * BARLINK_HOST_MAX_RANKS]; };

__device__ __forceinline__ u64 vload(const volatile u64* p) { return *p; }

// Sum accumulator per element type: fp32 for the half types (NCCL numerics),
// native for the integer types (a float accumulator would round int64).
template <typename T> struct AccOf { using type = float; };
template <> struct AccOf<int> { using type = int; };
template <> struct AccOf<long long> { using type = long long; };

__device__ __forceinline__ float to_acc(float v) { return v; }
__device__ __forceinline__ float to_acc(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_acc(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
__device__ __forceinline__ int to_acc(int v) { return v; }
__device__ __forceinline__ long long to_acc(long long v) { return v; }

template <typename T>
__device__ __forceinline__ T from_acc(typename AccOf<T>::type v);
template <>
__device__ __forceinline__ float from_acc<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half from_acc<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_acc<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}
template <>
__device__ __forceinline__ int from_acc<int>(int v) { return v; }
template <>
__device__ __forceinline__ long long from_acc<long long>(long long v) {
  return v;
}

// 16-byte access unit. Zero-copy traffic over PCIe needs wide loads and many
// of them in flight; a scalar half-by-half loop reaches a fraction of the
// link. The host side passes nvec = 0 whenever any pointer is not 16-byte
// aligned, and the scalar tail below covers the remainder.
template <typename T>
struct alignas(16) Vec16 { T x[16 / sizeof(T)]; };

// Bounded spin on one flag column. Returns TRUE when the deadline expired,
// false when the awaited value arrived.
//
// #583/#653: on expiry this writes an abort code into seq_dev[1] and RETURNS
// -- it must never __trap(). A device trap destroys the CUDA context, and
// from that moment every later CUDA call in the process returns a sticky
// cudaErrorLaunchFailure ("unspecified launch failure") at whatever unrelated
// call site happens to be next: #583 is a production boot in which a barlink
// expiry surfaced inside Triton's load_binary and named a kernel that was
// only the victim. The device transport was cured on 2026-08-05
// (barlink_device.py::barlink_begin_kernel); this transport kept the trap
// until #653. The host reads the word in
// BarlinkHostTransport.check_aborted and raises a structured
// HostCollectiveAborted instead. Same contract as the BAR1 transport's
// ctlStatus word (#431 fix 2).
//
// The caller must consume the return value -- see the shared-abort pattern at
// the three call sites, which is what turns "thread 0 gave up" into "the
// whole block leaves without touching the payload".
__device__ __forceinline__ bool wait_ge(const volatile u64* flags, int idx,
                                        u64 target, u64 timeout, u64* seq_dev,
                                        u64 code) {
  u64 start = clock64();
  while (vload(&flags[idx]) < target) {
    if ((u64)(clock64() - start) > timeout) { seq_dev[1] = code; return true; }
  }
  return false;
}

// "Am I the block that finished last?" -- the __threadfence + atomicAdd
// pattern from the CUDA programming guide. It is what lets ONE kernel both
// move the payload with the whole grid and publish a single flag afterwards,
// which is how this transport gets away with two launches instead of four.
// Callers must have executed __threadfence_system() before calling.
__device__ __forceinline__ bool last_block(unsigned* ctr) {
  __shared__ bool amlast;
  __syncthreads();  // this block's stores are complete and fenced
  if (threadIdx.x == 0) {
    unsigned t = atomicAdd(ctr, 1u);
    amlast = (t == gridDim.x - 1);
  }
  __syncthreads();
  return amlast;
}

// ---------------------------------------------------------------------------
// K1: publish this rank's payload into its mapped slot.
// ---------------------------------------------------------------------------
__global__ void barlink_host_put_kernel(
    const char* __restrict__ src, HostSlots slots, int slot_pair,
    size_t nbytes, size_t nvec, volatile u64* flags, u64* seq_dev,
    unsigned* blk_ctr, int pub_idx, int cons_idx0, int cons_stride,
    int cons_count, int lag, u64 timeout, int bump_seq) {
  const u64 seq = *seq_dev;
  // THE SHARED-ABORT PATTERN (#653), used identically at all three spin
  // sites of this file and taken from bar1_mesh_kernel's K_1BLK arm.
  //
  // Only thread 0 spins -- but a spin that gives up must take the WHOLE
  // block with it, because thread 0 returning alone would leave its peers at
  // a __syncthreads() only it was going to reach. So the outcome is
  // published in shared memory and every thread acts on it after the
  // barrier, before any payload work and before last_block()'s own
  // __syncthreads().
  //
  // ONE barrier, not two: `abortS` is written by thread 0 only, and every
  // other thread's first read of it is after the __syncthreads() below, so
  // the initialization needs no barrier of its own. That keeps the barrier
  // count of this kernel exactly what it was before the fix.
  //
  // TERMINAL-ABORT CONTRACT. After an abort the host raises and the process
  // is done -- HostCollectiveAborted is not recoverable, the output buffers
  // are partially written by construction. Cross-BLOCK skew is therefore
  // acceptable residue: some blocks may abort while others found the flag in
  // time, moved their share of the payload and even won last_block() and
  // published, and blk_ctr is left stale because the aborting blocks never
  // counted themselves in. None of that is repaired here. The abort word is
  // set in every one of those interleavings, which is the only thing the
  // host's raise needs; adding cross-block coordination would buy a tidier
  // post-mortem state for a process that is about to die anyway.
  __shared__ int abortS;
  // Reuse guard: the buffer we are about to overwrite was last READ during
  // op seq-lag (lag = 2, because the slot is double-buffered by parity). In
  // steady state every peer is long past it and this is a handful of loads.
  if (threadIdx.x == 0) {
    abortS = 0;
    if (seq > (u64)lag) {
      for (int k = 0; k < cons_count; ++k) {
        if (wait_ge(flags, cons_idx0 + k * cons_stride, seq - (u64)lag,
                    timeout, seq_dev, 1ull)) { abortS = 1; break; }
      }
    }
  }
  __syncthreads();
  if (abortS) return;

  char* dst = slots.p[slot_pair * 2 + (int)(seq & 1ULL)];
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (nvec) {
    const uint4* s4 = (const uint4*)src;
    uint4* d4 = (uint4*)dst;
    for (size_t i = tid; i < nvec; i += stride) d4[i] = s4[i];
  }
  for (size_t i = nvec * 16 + tid; i < nbytes; i += stride) dst[i] = src[i];

  // Order the payload ahead of the flag for a reader on ANOTHER device.
  __threadfence_system();
  if (last_block(blk_ctr) && threadIdx.x == 0) {
    *blk_ctr = 0;
    flags[pub_idx] = seq;
    // Collectives bump the sequence in K2; a send has no K2 of its own.
    if (bump_seq) *seq_dev = seq + 1;
  }
}

// ---------------------------------------------------------------------------
// K2a: wait for every peer, then reduce straight out of their mapped slots.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void barlink_host_reduce_kernel(
    const T* __restrict__ inp, T* __restrict__ out, size_t n, size_t nvec,
    HostSlots slots, int world, int rank, volatile u64* flags, u64* seq_dev,
    unsigned* blk_ctr, u64 timeout) {
  const u64 seq = *seq_dev;
  // Shared-abort pattern (#653) -- see barlink_host_put_kernel for the rule
  // and for the terminal-abort contract. Code 2 = "waiting for a peer to
  // publish its slot", distinct from the put kernel's reuse-guard wait
  // because the two point at different culprits.
  __shared__ int abortS;
  if (threadIdx.x == 0) {
    abortS = 0;
    for (int r = 0; r < world; ++r) {
      if (r == rank) continue;
      if (wait_ge(flags, r * BARLINK_HOST_COLS + 0, seq, timeout, seq_dev,
                  2ull)) { abortS = 1; break; }
    }
  }
  __syncthreads();
  if (abortS) return;
  // Acquire side of the release above. Mapped host memory is not cached on
  // the device, so ordering is all that is needed here -- no invalidation.
  __threadfence_system();

  const int par = (int)(seq & 1ULL);
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  constexpr int K = 16 / sizeof(T);
  using A = typename AccOf<T>::type;

  if (nvec) {
    const Vec16<T>* vin = (const Vec16<T>*)inp;
    Vec16<T>* vout = (Vec16<T>*)out;
    for (size_t i = tid; i < nvec; i += stride) {
      Vec16<T> a = vin[i];
      A acc[K];
#pragma unroll
      for (int k = 0; k < K; ++k) acc[k] = to_acc(a.x[k]);
      for (int r = 0; r < world; ++r) {
        if (r == rank) continue;
        const Vec16<T>* vp = (const Vec16<T>*)slots.p[r * 2 + par];
        Vec16<T> b = vp[i];
#pragma unroll
        for (int k = 0; k < K; ++k) acc[k] += to_acc(b.x[k]);
      }
      Vec16<T> o;
#pragma unroll
      for (int k = 0; k < K; ++k) o.x[k] = from_acc<T>(acc[k]);
      vout[i] = o;
    }
  }
  for (size_t i = nvec * K + tid; i < n; i += stride) {
    A acc = to_acc(inp[i]);
    for (int r = 0; r < world; ++r) {
      if (r == rank) continue;
      acc += to_acc(((const T*)slots.p[r * 2 + par])[i]);
    }
    out[i] = from_acc<T>(acc);
  }

  __threadfence_system();
  if (last_block(blk_ctr) && threadIdx.x == 0) {
    *blk_ctr = 0;
    flags[rank * BARLINK_HOST_COLS + 1] = seq;  // consumed
    *seq_dev = seq + 1;
  }
}

// ---------------------------------------------------------------------------
// K2b: wait, then copy source regions out. Serves all_gather (rn = world),
// broadcast (rn = 1, or rn = 0 on the source rank, which only has to close
// the op) and recv (rn = 1, a pair buffer). Pure byte movement, so it is
// exact for EVERY dtype -- including the int64 index tensors the speculative
// draft-pick sync carries, which no reduce kernel even dispatches.
// ---------------------------------------------------------------------------
__global__ void barlink_host_copyout_kernel(
    char* __restrict__ out, const char* __restrict__ self_src, HostSlots slots,
    int r0, int rn, int slot_step, int self_rank, size_t nbytes, size_t nvec,
    volatile u64* flags, u64* seq_dev, unsigned* blk_ctr, int pub_base,
    int pub_stride, int cons_idx, u64 timeout) {
  const u64 seq = *seq_dev;
  // Shared-abort pattern (#653) -- see barlink_host_put_kernel. Code 3 =
  // "waiting for a source rank to publish", which is the all_gather /
  // broadcast / recv side of the same protocol.
  __shared__ int abortS;
  if (threadIdx.x == 0) {
    abortS = 0;
    for (int k = 0; k < rn; ++k) {
      const int r = r0 + k;
      if (r == self_rank) continue;
      if (wait_ge(flags, pub_base + r * pub_stride, seq, timeout, seq_dev,
                  3ull)) { abortS = 1; break; }
    }
  }
  __syncthreads();
  if (abortS) return;
  __threadfence_system();

  const int par = (int)(seq & 1ULL);
  const size_t stride = (size_t)gridDim.x * blockDim.x;
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  for (int k = 0; k < rn; ++k) {
    const int r = r0 + k;
    const char* s = (r == self_rank)
                        ? self_src
                        : (const char*)slots.p[(slot_step ? r * 2 : 0) + par];
    char* d = out + (size_t)k * nbytes;
    if (nvec) {
      const uint4* s4 = (const uint4*)s;
      uint4* d4 = (uint4*)d;
      for (size_t i = tid; i < nvec; i += stride) d4[i] = s4[i];
    }
    for (size_t i = nvec * 16 + tid; i < nbytes; i += stride) d[i] = s[i];
  }

  __threadfence_system();
  if (last_block(blk_ctr) && threadIdx.x == 0) {
    *blk_ctr = 0;
    flags[cons_idx] = seq;
    *seq_dev = seq + 1;
  }
}

// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------
static inline int barlink_grid(size_t items, int cap) {
  size_t b = (items + 255) / 256;
  if (b < 1) b = 1;
  if (b > (size_t)cap) b = (size_t)cap;
  return (int)b;
}

// Addresses arrive as an int64 CPU TENSOR, not as a python list. pybind
// materializes a std::vector<int64_t> element by element on every call, and
// this is called once per collective in a path whose whole budget is single
// digit microseconds -- a tensor argument is a handle, not a conversion.
static inline HostSlots barlink_slots(const at::Tensor& addrs) {
  TORCH_CHECK(addrs.device().is_cpu() && addrs.scalar_type() == at::kLong &&
                  addrs.is_contiguous(),
              "barlink host: addresses must be a contiguous int64 CPU tensor");
  TORCH_CHECK(addrs.numel() <= 2 * BARLINK_HOST_MAX_RANKS,
              "barlink host: too many slot addresses");
  HostSlots s;
  const int64_t* a = addrs.data_ptr<int64_t>();
  for (int i = 0; i < 2 * BARLINK_HOST_MAX_RANKS; ++i) s.p[i] = nullptr;
  for (int64_t i = 0; i < addrs.numel(); ++i) s.p[i] = (char*)a[i];
  return s;
}

// nvec = number of 16-byte units, or 0 when anything involved is misaligned.
// Alignment is checked by OR-ing the pointers, not by walking a container:
// this runs per launch, and a heap allocation per launch is exactly the kind
// of hot-path cost this transport exists to not have.
static inline uintptr_t barlink_or_slots(const at::Tensor& addrs) {
  uintptr_t bits = 0;
  const int64_t* a = addrs.data_ptr<int64_t>();
  for (int64_t i = 0; i < addrs.numel(); ++i) bits |= (uintptr_t)a[i];
  return bits;
}

static inline size_t barlink_nvec(size_t nbytes, uintptr_t or_bits) {
  if (nbytes % 16) return 0;
  if (or_bits % 16) return 0;
  return nbytes / 16;
}

static void barlink_launch_put(const void* src, const at::Tensor& addrs,
                             int slot_pair, size_t nbytes, int64_t flags_addr,
                             at::Tensor& seq_dev, at::Tensor& blk_ctr,
                             int pub_idx, int cons_idx0, int cons_stride,
                             int cons_count, int lag, int64_t timeout,
                             int64_t cap, int bump_seq, cudaStream_t stream) {
  HostSlots slots = barlink_slots(addrs);
  size_t nvec = barlink_nvec(nbytes, (uintptr_t)src | barlink_or_slots(addrs));
  int blocks = barlink_grid(nvec ? nvec : nbytes, (int)cap);
  barlink_host_put_kernel<<<blocks, 256, 0, stream>>>(
      (const char*)src, slots, slot_pair, nbytes, nvec,
      (volatile u64*)flags_addr, (u64*)seq_dev.data_ptr(),
      (unsigned*)blk_ctr.data_ptr(), pub_idx, cons_idx0, cons_stride,
      cons_count, lag, (u64)timeout, bump_seq);
}

static void barlink_launch_copyout(void* out, const void* self_src,
                                 const at::Tensor& addrs, int r0,
                                 int rn, int slot_step, int self_rank,
                                 size_t nbytes, int64_t flags_addr,
                                 at::Tensor& seq_dev, at::Tensor& blk_ctr,
                                 int pub_base, int pub_stride, int cons_idx,
                                 int64_t timeout, int64_t cap,
                                 cudaStream_t stream) {
  HostSlots slots = barlink_slots(addrs);
  size_t nvec = barlink_nvec(nbytes, (uintptr_t)out | (uintptr_t)self_src |
                                       barlink_or_slots(addrs));
  int blocks = barlink_grid(nvec ? nvec : nbytes, (int)cap);
  barlink_host_copyout_kernel<<<blocks, 256, 0, stream>>>(
      (char*)out, (const char*)self_src, slots, r0, rn, slot_step, self_rank,
      nbytes, nvec, (volatile u64*)flags_addr, (u64*)seq_dev.data_ptr(),
      (unsigned*)blk_ctr.data_ptr(), pub_base, pub_stride, cons_idx,
      (u64)timeout);
}

template <typename T>
static void barlink_launch_reduce(const at::Tensor& inp, at::Tensor& out,
                                const at::Tensor& addrs, int rank,
                                int world, int64_t flags_addr,
                                at::Tensor& seq_dev, at::Tensor& blk_ctr,
                                int64_t timeout, int64_t cap,
                                cudaStream_t stream) {
  HostSlots slots = barlink_slots(addrs);
  size_t n = inp.numel();
  size_t nvec = barlink_nvec(n * sizeof(T), (uintptr_t)inp.data_ptr() |
                                              (uintptr_t)out.data_ptr() |
                                              barlink_or_slots(addrs));
  int blocks = barlink_grid(nvec ? nvec : n, (int)cap);
  barlink_host_reduce_kernel<T><<<blocks, 256, 0, stream>>>(
      (const T*)inp.data_ptr(), (T*)out.data_ptr(), n, nvec, slots, world,
      rank, (volatile u64*)flags_addr, (u64*)seq_dev.data_ptr(),
      (unsigned*)blk_ctr.data_ptr(), (u64)timeout);
}

void barlink_host_all_reduce(at::Tensor inp, at::Tensor out,
                           at::Tensor slot_addrs, int64_t flags_addr,
                           at::Tensor seq_dev, at::Tensor blk_ctr,
                           int64_t rank, int64_t world, int64_t timeout,
                           int64_t blocks_cap) {
  TORCH_CHECK(world <= BARLINK_HOST_MAX_RANKS, "barlink host: too many ranks");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(inp.numel() == out.numel(), "barlink host: shape mismatch");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  size_t nbytes = inp.numel() * inp.element_size();
  barlink_launch_put(inp.data_ptr(), slot_addrs, (int)rank, nbytes, flags_addr,
                   seq_dev, blk_ctr, (int)rank * BARLINK_HOST_COLS + 0,
                   /*cons_idx0=*/1, /*cons_stride=*/BARLINK_HOST_COLS,
                   /*cons_count=*/(int)world, /*lag=*/2, timeout, blocks_cap,
                   /*bump_seq=*/0, stream);
  switch (inp.scalar_type()) {
    case at::kBFloat16:
      barlink_launch_reduce<__nv_bfloat16>(inp, out, slot_addrs, (int)rank,
                                         (int)world, flags_addr, seq_dev,
                                         blk_ctr, timeout, blocks_cap, stream);
      break;
    case at::kHalf:
      barlink_launch_reduce<__half>(inp, out, slot_addrs, (int)rank, (int)world,
                                  flags_addr, seq_dev, blk_ctr, timeout,
                                  blocks_cap, stream);
      break;
    case at::kFloat:
      barlink_launch_reduce<float>(inp, out, slot_addrs, (int)rank, (int)world,
                                 flags_addr, seq_dev, blk_ctr, timeout,
                                 blocks_cap, stream);
      break;
    case at::kInt:
      barlink_launch_reduce<int>(inp, out, slot_addrs, (int)rank, (int)world,
                               flags_addr, seq_dev, blk_ctr, timeout,
                               blocks_cap, stream);
      break;
    case at::kLong:
      barlink_launch_reduce<long long>(inp, out, slot_addrs, (int)rank,
                                     (int)world, flags_addr, seq_dev, blk_ctr,
                                     timeout, blocks_cap, stream);
      break;
    default:
      TORCH_CHECK(false, "barlink host: unsupported all_reduce dtype");
  }
}

void barlink_host_all_gather(at::Tensor inp, at::Tensor out,
                           at::Tensor slot_addrs, int64_t flags_addr,
                           at::Tensor seq_dev, at::Tensor blk_ctr,
                           int64_t rank, int64_t world, int64_t timeout,
                           int64_t blocks_cap) {
  TORCH_CHECK(world <= BARLINK_HOST_MAX_RANKS, "barlink host: too many ranks");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(out.numel() == inp.numel() * world, "barlink host: bad gather out");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  size_t nbytes = inp.numel() * inp.element_size();
  barlink_launch_put(inp.data_ptr(), slot_addrs, (int)rank, nbytes, flags_addr,
                   seq_dev, blk_ctr, (int)rank * BARLINK_HOST_COLS + 0, 1,
                   BARLINK_HOST_COLS, (int)world, 2, timeout, blocks_cap, 0,
                   stream);
  barlink_launch_copyout(out.data_ptr(), inp.data_ptr(), slot_addrs, /*r0=*/0,
                       /*rn=*/(int)world, /*slot_step=*/1,
                       /*self_rank=*/(int)rank, nbytes, flags_addr, seq_dev,
                       blk_ctr, /*pub_base=*/0, /*pub_stride=*/BARLINK_HOST_COLS,
                       /*cons_idx=*/(int)rank * BARLINK_HOST_COLS + 1, timeout,
                       blocks_cap, stream);
}

void barlink_host_broadcast(at::Tensor t, at::Tensor slot_addrs,
                          int64_t flags_addr, at::Tensor seq_dev,
                          at::Tensor blk_ctr, int64_t rank, int64_t world,
                          int64_t src, int64_t timeout, int64_t blocks_cap) {
  TORCH_CHECK(world <= BARLINK_HOST_MAX_RANKS, "barlink host: too many ranks");
  TORCH_CHECK(t.is_contiguous());
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  size_t nbytes = t.numel() * t.element_size();
  if (rank == src) {
    barlink_launch_put(t.data_ptr(), slot_addrs, (int)rank, nbytes, flags_addr,
                     seq_dev, blk_ctr, (int)rank * BARLINK_HOST_COLS + 0, 1,
                     BARLINK_HOST_COLS, (int)world, 2, timeout, blocks_cap, 0,
                     stream);
    // rn = 0: nothing to copy, but the op must still be closed -- every rank
    // publishes a consumption flag for every op, which is what lets a run of
    // back-to-back broadcasts from the same source reuse its slot safely.
    barlink_launch_copyout(t.data_ptr(), nullptr, slot_addrs, 0, 0, 1, -1,
                         nbytes, flags_addr, seq_dev, blk_ctr, 0,
                         BARLINK_HOST_COLS,
                         (int)rank * BARLINK_HOST_COLS + 1, timeout, blocks_cap,
                         stream);
  } else {
    barlink_launch_copyout(t.data_ptr(), nullptr, slot_addrs, (int)src, 1, 1, -1,
                         nbytes, flags_addr, seq_dev, blk_ctr, 0,
                         BARLINK_HOST_COLS,
                         (int)rank * BARLINK_HOST_COLS + 1, timeout, blocks_cap,
                         stream);
  }
}

void barlink_host_send(at::Tensor inp, at::Tensor pair_addrs,
                     int64_t flags_addr, at::Tensor seq_dev,
                     at::Tensor blk_ctr, int64_t rank, int64_t dst,
                     int64_t timeout, int64_t blocks_cap) {
  TORCH_CHECK(inp.is_contiguous());
  TORCH_CHECK(pair_addrs.numel() == 2, "barlink host: need both pair buffers");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  size_t nbytes = inp.numel() * inp.element_size();
  barlink_launch_put(inp.data_ptr(), pair_addrs, /*slot_pair=*/0, nbytes,
                   flags_addr, seq_dev, blk_ctr,
                   /*pub_idx=*/(int)rank * BARLINK_HOST_COLS + 2 + (int)dst,
                   /*cons_idx0=*/(int)dst * BARLINK_HOST_COLS + 2 +
                       BARLINK_HOST_MAX_RANKS + (int)rank,
                   /*cons_stride=*/0, /*cons_count=*/1, /*lag=*/2, timeout,
                   blocks_cap, /*bump_seq=*/1, stream);
}

void barlink_host_recv(at::Tensor out, at::Tensor pair_addrs,
                     int64_t flags_addr, at::Tensor seq_dev,
                     at::Tensor blk_ctr, int64_t rank, int64_t src,
                     int64_t timeout, int64_t blocks_cap) {
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(pair_addrs.numel() == 2, "barlink host: need both pair buffers");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  size_t nbytes = out.numel() * out.element_size();
  barlink_launch_copyout(out.data_ptr(), nullptr, pair_addrs, /*r0=*/(int)src,
                       /*rn=*/1, /*slot_step=*/0, /*self_rank=*/-1, nbytes,
                       flags_addr, seq_dev, blk_ctr,
                       /*pub_base=*/2 + (int)rank,
                       /*pub_stride=*/BARLINK_HOST_COLS,
                       /*cons_idx=*/(int)rank * BARLINK_HOST_COLS + 2 +
                           BARLINK_HOST_MAX_RANKS + (int)src,
                       timeout, blocks_cap, stream);
}
"""

_CPP_SRC = """
void barlink_host_all_reduce(at::Tensor inp, at::Tensor out,
                           at::Tensor slot_addrs, int64_t flags_addr,
                           at::Tensor seq_dev, at::Tensor blk_ctr,
                           int64_t rank, int64_t world, int64_t timeout,
                           int64_t blocks_cap);
void barlink_host_all_gather(at::Tensor inp, at::Tensor out,
                           at::Tensor slot_addrs, int64_t flags_addr,
                           at::Tensor seq_dev, at::Tensor blk_ctr,
                           int64_t rank, int64_t world, int64_t timeout,
                           int64_t blocks_cap);
void barlink_host_broadcast(at::Tensor t, at::Tensor slot_addrs,
                          int64_t flags_addr, at::Tensor seq_dev,
                          at::Tensor blk_ctr, int64_t rank, int64_t world,
                          int64_t src, int64_t timeout, int64_t blocks_cap);
void barlink_host_send(at::Tensor inp, at::Tensor pair_addrs,
                     int64_t flags_addr, at::Tensor seq_dev,
                     at::Tensor blk_ctr, int64_t rank, int64_t dst,
                     int64_t timeout, int64_t blocks_cap);
void barlink_host_recv(at::Tensor out, at::Tensor pair_addrs,
                     int64_t flags_addr, at::Tensor seq_dev,
                     at::Tensor blk_ctr, int64_t rank, int64_t src,
                     int64_t timeout, int64_t blocks_cap);
"""

_ext = None


# ---------------------------------------------------------------------------
# The build/arch/cache machinery is DELIBERATELY not reimplemented here.
# barlink_device.py already resolves the group's architectures per vendor, keeps
# the two namespaces apart, and guards torch's shared extension cache against
# a killed build. A second copy of that would drift; a shared import will not.
#
# Every sglang import stays INSIDE a function: this module must remain
# importable without creating a CUDA context (pinned by
# test_barlink_port.py::test_import_does_not_initialize_cuda), and importing the
# `sglang.srt.utils` package initializes CUDA in its __init__.
# ---------------------------------------------------------------------------


_DEV_MOD = None


def _device_mod():
    # Cached: this is reached once per collective through the timeout
    # resolver, and re-executing an import statement there is ~1 us of Python
    # in a path whose whole budget is single-digit microseconds.
    global _DEV_MOD
    if _DEV_MOD is None:
        from sglang.srt.distributed.device_communicators import barlink_device

        _DEV_MOD = barlink_device
    return _DEV_MOD


def resolve_timeout_cycles(base_cycles: int) -> int:
    """Deadline for a collective right now -- see utils/jit_cold_build.

    Not the raw constant: during a cold start a peer may still be inside nvcc
    building a first-call JIT kernel, and the cycle deadline would trap this
    rank's wait long before the peer arrives. Outside that window the
    resolver is the identity, so the value baked into a CAPTURED graph is
    unchanged.
    """
    return _device_mod().resolve_timeout_cycles(base_cycles)


def _load_ext(cpu_group):
    global _ext
    if _ext is not None:
        return _ext
    from torch.utils.cpp_extension import load_inline

    dev = _device_mod()
    by_vendor = dev._resolve_build_arches(cpu_group)
    vendor = dev._local_vendor()
    arches = by_vendor.get(vendor, [])
    flags = dev._build_flags(vendor, arches)
    # (vendor, that vendor's arches) in the CACHE KEY, not just in the flags:
    # torch rebuilds only when the SOURCES change, so a name that ignores the
    # architecture hands a stale single-arch .so to a rank that cannot run it.
    name = "barlink_host_ext_" + vendor
    if arches:
        name += "_" + "_".join(a.replace(".", "") for a in arches)
    t0 = time.time()
    with dev._ext_cache_guarded(name) as build_dir:
        _ext = load_inline(
            name=name,
            cpp_sources=_CPP_SRC,
            cuda_sources=_CUDA_SRC,
            functions=[
                "barlink_host_all_reduce",
                "barlink_host_all_gather",
                "barlink_host_broadcast",
                "barlink_host_send",
                "barlink_host_recv",
            ],
            extra_cuda_cflags=flags or None,
            verbose=False,
            build_directory=str(build_dir) if build_dir is not None else None,
        )
    _info_once(
        "barlink host extension %r built for %s arch(es) %s in %.1f s",
        name, vendor, ",".join(arches) or "<torch default>", time.time() - t0,
    )
    return _ext


def _register_portable_mapped(ptr: int, nbytes: int) -> int:
    """Page-lock a host mapping as PORTABLE + MAPPED, return its device address.

    CUDA and ROCm expose the identical pair of calls (cudaHostRegister /
    hipHostRegister, cudaHostGetDevicePointer / hipHostGetDevicePointer) with
    identical flag values, and each process registers only with its OWN
    runtime -- that is what lets a CUDA and a ROCm rank share one segment.

    Portable (1): pinned for every context of this process, not just the one
    that registered. Mapped (2): the segment gets a device address, which is
    what makes the kernels able to load and store it directly. Under UVA the
    device address equals the host one; the call is made anyway rather than
    assumed, because "assumed" is not something a data plane should be.

    Raises instead of degrading: an unpinned or unmapped segment would not be
    a slower version of this transport, it would be a kernel dereferencing a
    host pointer the GPU has no translation for.
    """
    hip = torch.version.hip is not None
    try:
        lib = ctypes.CDLL("libamdhip64.so" if hip else "libcudart.so")
    except OSError as e:
        raise RuntimeError(
            f"barlink host transport: {'hip' if hip else 'cuda'} runtime not "
            f"loadable ({e}); the segment cannot be page-locked."
        ) from e
    register = lib.hipHostRegister if hip else lib.cudaHostRegister
    get_dev = lib.hipHostGetDevicePointer if hip else lib.cudaHostGetDevicePointer
    register.restype = ctypes.c_int
    get_dev.restype = ctypes.c_int
    # 1 = Portable, 2 = Mapped (same values in both runtimes).
    rc = register(ctypes.c_void_p(ptr), ctypes.c_size_t(nbytes), ctypes.c_uint(3))
    if rc != 0:
        raise RuntimeError(
            f"barlink host transport: host-register of the {nbytes} B segment "
            f"failed (rc={rc}). Common cause is a locked-memory ulimit below "
            f"the segment size -- check `ulimit -l`."
        )
    dptr = ctypes.c_void_p()
    rc = get_dev(ctypes.byref(dptr), ctypes.c_void_p(ptr), ctypes.c_uint(0))
    if rc != 0:
        raise RuntimeError(
            f"barlink host transport: no device pointer for the mapped segment "
            f"(rc={rc}); this GPU cannot map host memory."
        )
    if int(dptr.value or 0) != ptr:
        logger.info(
            "barlink host transport: device mapping is not identity "
            "(host 0x%x -> device 0x%x); using the device address.",
            ptr, int(dptr.value or 0),
        )
    return int(dptr.value or 0)


def _unregister(ptr: int) -> None:
    hip = torch.version.hip is not None
    lib = ctypes.CDLL("libamdhip64.so" if hip else "libcudart.so")
    fn = lib.hipHostUnregister if hip else lib.cudaHostUnregister
    fn.restype = ctypes.c_int
    fn(ctypes.c_void_p(ptr))


class HostCollectiveAborted(RuntimeError):
    """A host-transport spin kernel hit its deadline and gave up.

    Carries the same structured context as the device transport's
    ``DeviceCollectiveAborted`` (#583) and the BAR1 transport's
    ``Bar1CollectiveAborted`` (#431 fix 2): which rank, which world, which
    wait. Raised from :meth:`BarlinkHostTransport.check_aborted`.
    """

    def __init__(self, message: str, *, rank: int, world: int, code: int,
                 where: str):
        super().__init__(message)
        self.rank = rank
        self.world = world
        self.code = code
        self.where = where


#: seq_dev[1] codes written by the three spin sites on deadline expiry.
#: Distinct on purpose: the three point at different culprits.
_ABORT_WAITS = {
    1: (
        "barlink_host_put_kernel (reuse guard: waiting for the peers to "
        "consume the op whose slot is about to be overwritten)"
    ),
    2: (
        "barlink_host_reduce_kernel (waiting for a peer to publish its slot "
        "for this op)"
    ),
    3: (
        "barlink_host_copyout_kernel (waiting for a source rank to publish "
        "its slot -- the all_gather / broadcast / recv side)"
    ),
}


class BarlinkHostTransport:
    """Collectives and point-to-point over one pinned, portable host segment.

    Two kernels per op, no host synchronization, no per-op allocation. See the
    module docstring for the protocol; the interface below is the transport
    seam documented in barlink.py.
    """

    #: ~30 s at 2 GHz. On expiry a spin kernel writes the abort word into
    #: `seq_dev[1]` and RETURNS, instead of hanging the GPU forever -- and
    #: instead of trapping, which is what #653 removed: a device trap
    #: destroys the CUDA context and every later CUDA call in the process
    #: then fails with a sticky "unspecified launch failure" at an unrelated
    #: site (#583). `check_aborted` turns the word into a raise.
    _TIMEOUT_CYCLES = 60_000_000_000

    #: At CLASS level on purpose, like the device transport's abort-poll
    #: state: `check_aborted` and `close` must answer for every construction
    #: path, including the ones that never run `__init__` (the #653 suite
    #: builds transports through `__new__` to exercise the guard without a
    #: card). A guard whose first statement can raise AttributeError is not a
    #: guard, and "not registered / no counters" is the correct default.
    _registered_in_gate = False
    _seq_all = None

    # -- pluggable-transport interface (see barlink.py "transport seam") ------
    #
    # The communicator ASKS instead of knowing, so every size/op limit of this
    # transport is stated here and nowhere else.
    #
    # all_reduce / broadcast / reduce_scatter chunk over the slot in Python
    # (the chunk count depends only on the shape, so this stays graph-safe)
    # and therefore have NO payload ceiling. all_gather does not chunk: its
    # output is [world, n] and a chunked variant would need a strided
    # write-out, which is a separate, testable change -- so it declares the
    # ceiling instead of asserting it later.
    BARLINK_OPS = frozenset(
        {"all_reduce", "all_gather", "reduce_scatter", "broadcast"}
    )
    P2P_OPS = frozenset({"send", "recv"})

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        slot_bytes: int,
        p2p_bytes: int | None = None,
    ):
        self.cpu_group = cpu_group
        self.device = device
        self.world_size = dist.get_world_size(cpu_group)
        self.rank = dist.get_rank(cpu_group)
        if self.world_size > _MAX_RANKS:
            raise RuntimeError(
                f"barlink host transport supports up to {_MAX_RANKS} ranks, "
                f"got {self.world_size}."
            )
        # Own knob wins over the shared SGLANG_BARLINK_SLOT_MIB the factory
        # passes in; unset means "same slot as the other transports".
        own = os.environ.get(_SLOT_MIB_ENV)
        self.slot_bytes = int(own) * 1024 * 1024 if own else int(slot_bytes)
        self.p2p_bytes = _P2P_BYTES if p2p_bytes is None else int(p2p_bytes)
        if self.slot_bytes % 16 or self.slot_bytes <= 0:
            raise RuntimeError("barlink host: slot size must be a positive x16")
        self._blocks = max(1, _BLOCKS)

        npairs = self.world_size * (self.world_size - 1)
        self._p2p_base_off = _HEADER_BYTES + self.world_size * 2 * self.slot_bytes
        total = self._p2p_base_off + npairs * 2 * self.p2p_bytes

        # Peer identities first, BEFORE the rendezvous: the exchange is
        # itself a collective, so a rank that never arrives is caught by
        # the exchange's own deadline instead of by the unbounded
        # broadcast below.
        self._peer_table = barlink_liveness.install(cpu_group)

        # Rendezvous over the (vendor-neutral) gloo group: rank 0 creates the
        # segment and broadcasts its name.
        if self.rank == 0:
            self._shm = shared_memory.SharedMemory(create=True, size=total)
            name = [self._shm.name]
        else:
            name = [None]
        # Inline in torch, so not pollable: refusing to enter it when a peer
        # is already gone is the bound.
        barlink_liveness.check_peers(
            "barlink host rendezvous (broadcast_object_list)",
            table=self._peer_table,
        )
        dist.broadcast_object_list(
            name, src=dist.get_global_rank(cpu_group, 0), group=cpu_group
        )
        if self.rank != 0:
            self._shm = shared_memory.SharedMemory(name=name[0])

        buf = self._shm.buf
        #: Host view of the flag block. NOT used in the hot path -- the hot
        #: path is entirely device-side. It exists so a caller can observe
        #: completion from the host by reading the pinned flag instead of
        #: calling cudaStreamSynchronize (see `completed_seq`).
        self._flags_np = np.frombuffer(
            buf, dtype=np.uint64, count=_MAX_RANKS * _COLS, offset=0
        ).reshape(_MAX_RANKS, _COLS)
        if self.rank == 0:
            self._flags_np[:] = 0

        host_base = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        # Device-side state first: allocating it creates this rank's CUDA
        # context, which cudaHostRegister below needs.
        with torch.cuda.device(device):
            # ONE backing tensor, TWO words per counter (#653).
            #
            # Word 0 is the sequence number; word 1 is the abort word a
            # tripped spin kernel writes instead of trapping. Every kernel
            # already receives its counter as `seq_dev`, so the abort path
            # needed no new parameter on any launcher or entry point and no
            # captured graph changes shape -- the same argument the device
            # transport's two-element `_seq_dev` carries (#583).
            #
            # ONE tensor rather than three is what makes `check_aborted` a
            # single read: every abort word this transport can produce --
            # collective, per-destination send, per-source recv -- lives in
            # column 1 of this tensor. The rows are handed to the extension as
            # 2-element CONTIGUOUS views (`self._seq_all[i]`), which is
            # exactly the [seq, abort] pair the kernels index.
            #
            # Row 0 = collectives, rows 1.._MAX_RANKS = send-to-dst,
            # the remainder = recv-from-src.
            self._seq_all = torch.zeros(
                1 + 2 * _MAX_RANKS, 2, dtype=torch.int64, device=device
            )
            # seq starts at 1, so the put kernel's reuse guard (which only
            # applies from seq > 2) is trivially satisfied for the first two
            # ops -- the buffers were never used then. The abort column stays
            # 0: nothing has aborted yet, and the word is one-way.
            self._seq_all[:, 0] = 1
            self._seq = self._seq_all[0]
            self._send_seq = self._seq_all[1 : 1 + _MAX_RANKS]
            self._recv_seq = self._seq_all[1 + _MAX_RANKS :]
            # Last-block election counter, reset by the block that wins it.
            # ONE counter is enough because the two kernels of an op run in
            # stream order and therefore never overlap -- which also states
            # the limit: this transport serves ONE stream at a time. Two
            # collectives issued concurrently on different streams would
            # share this counter and mis-elect. The communicator issues on
            # the current stream only, so that case does not arise today.
            self._blk = torch.zeros(1, dtype=torch.int32, device=device)
            self._host_base = host_base
            dev_base = _register_portable_mapped(host_base, total)
        self._dev_base = dev_base

        # Precomputed once and handed to the extension as int64 CPU TENSORS,
        # not as python lists: a list argument makes pybind materialize a
        # std::vector element by element on EVERY collective, which is real
        # microseconds in a path whose whole budget is single-digit
        # microseconds. Nothing in the hot path builds a list or computes an
        # address.
        self._slot_addrs = torch.tensor(
            [
                dev_base + _HEADER_BYTES + (r * 2 + p) * self.slot_bytes
                for r in range(self.world_size)
                for p in (0, 1)
            ],
            dtype=torch.int64,
        )
        self._pair_addrs = {
            (s, d): torch.tensor(
                [
                    dev_base
                    + self._p2p_base_off
                    + (self._pair_index(s, d) * 2 + p) * self.p2p_bytes
                    for p in (0, 1)
                ],
                dtype=torch.int64,
            )
            for s in range(self.world_size)
            for d in range(self.world_size)
            if s != d and self.p2p_bytes > 0
        }
        self._flags_addr = dev_base

        self._ext = _load_ext(cpu_group)
        # Bound once: the hot path calls this per collective.
        self._resolve_timeout = resolve_timeout_cycles
        # Everyone attached, registered and zeroed before the first op.
        barlink_liveness.bounded_barrier(
            cpu_group,
            "barlink host bring-up barrier",
            table=self._peer_table,
        )
        # #653: join the abort gate, exactly as the device transport does
        # (barlink_device.py, "#583: join the abort gate"). `_after_transport`
        # in barlink.py already reaches `check_aborted` after every dispatched
        # collective; the gate covers the case that dispatch site cannot -- a
        # collective that only ever runs inside a REPLAYED CUDA graph, where
        # no host code runs between the kernels. This transport is in
        # CAPTURABLE_BARLINK_TRANSPORTS, so that case is not hypothetical.
        barlink_abort_gate.register(self)
        self._registered_in_gate = True
        _info_once(
            "barlink host transport up: %d ranks, %d MiB slots (x2, "
            "double-buffered), %d MiB p2p pairs, pinned portable+mapped "
            "segment of %.1f MiB, %d blocks/op.",
            self.world_size, self.slot_bytes // 2**20, self.p2p_bytes // 2**20,
            total / 2**20, self._blocks,
        )

    def _pair_index(self, src: int, dst: int) -> int:
        """Dense index of the ordered pair (src -> dst), diagonal excluded."""
        return src * (self.world_size - 1) + (dst - (1 if dst > src else 0))

    # ------------------------------------------------------------------
    # transport seam
    # ------------------------------------------------------------------

    def handles(self, op: str, nbytes: int) -> bool:
        """What this transport serves, at this size. Rank-uniform: the answer
        depends only on the op, the payload size (identical on every rank for
        every op here) and group-uniform configuration.

        DTYPE IS NOT PART OF THE QUESTION, and cannot be -- the seam does not
        carry it. all_reduce dispatches bf16/fp16/fp32/int32/int64 and raises
        from the extension on anything else; the byte-moving ops (all_gather,
        broadcast, send/recv) are exact for every dtype. Same limitation the
        device transport has, stated rather than discovered.
        """
        if op == "all_gather":
            return nbytes <= self.slot_bytes
        if op in self.BARLINK_OPS:
            return True
        if op in self.P2P_OPS:
            return self.p2p_bytes > 0
        return False

    def barlink_all_reduce(self, comm, inp: torch.Tensor) -> torch.Tensor:
        return self.all_reduce(inp)

    def barlink_all_gather(self, comm, inp: torch.Tensor, dim: int) -> torch.Tensor:
        return self.all_gather(inp, dim)

    def barlink_reduce_scatter(self, comm, inp: torch.Tensor, dim: int) -> torch.Tensor:
        return self.reduce_scatter(inp, dim)

    def barlink_broadcast(self, comm, tensor: torch.Tensor, src: int):
        return self.broadcast(tensor, src)

    # ------------------------------------------------------------------
    # collectives
    # ------------------------------------------------------------------

    def _timeout(self) -> int:
        return self._resolve_timeout(self._TIMEOUT_CYCLES)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """Sum-all-reduce, out-of-place.

        The result is a FRESH tensor per call and must stay that way: a
        shape-keyed cache here made two same-shape results the SAME tensor and
        corrupted the model forward silently (see
        BarlinkCommunicator._get_out_buf). `empty_like` comes from torch's
        caching allocator -- no cudaMalloc in the hot path -- and inside a
        CUDA-graph capture it comes from the graph's private pool, so replays
        keep a stable address.
        """
        inp = input_.contiguous()
        out = torch.empty_like(inp)
        flat_in = inp.view(-1)
        flat_out = out.view(-1)
        n = flat_in.numel()
        slot_elems = self.slot_bytes // inp.element_size()
        timeout = self._timeout()
        for start in range(0, n, slot_elems):
            end = min(start + slot_elems, n)
            self._ext.barlink_host_all_reduce(
                flat_in[start:end], flat_out[start:end], self._slot_addrs,
                self._flags_addr, self._seq, self._blk, self.rank,
                self.world_size, timeout, self._blocks,
            )
        return out

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += input_.dim()
        inp = input_.contiguous()
        nbytes = inp.numel() * inp.element_size()
        if nbytes > self.slot_bytes:
            # Unreachable through the communicator (`handles` declines first);
            # a direct caller gets the reason, not a corrupt gather.
            raise RuntimeError(
                f"barlink host all_gather: {nbytes} B exceeds the {self.slot_bytes} B "
                f"slot; raise {_SLOT_MIB_ENV} or SGLANG_BARLINK_SLOT_MIB."
            )
        input_size = inp.size()
        output = torch.empty(
            (self.world_size,) + tuple(input_size),
            dtype=inp.dtype,
            device=inp.device,
        )
        self._ext.barlink_host_all_gather(
            inp.view(-1), output.view(-1), self._slot_addrs, self._flags_addr,
            self._seq, self._blk, self.rank, self.world_size, self._timeout(),
            self._blocks,
        )
        output = output.movedim(0, dim)
        return output.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += input_.dim()
        reduced = self.all_reduce(input_)
        # movedim(dim, 0) -- NOT movedim(0, dim). The two agree only for dim
        # in {0, 1}; from dim >= 2 the wrong axis is sliced while every shape
        # check still passes (see the same note in barlink.py).
        moved = reduced.movedim(dim, 0).contiguous()
        assert moved.shape[0] % self.world_size == 0
        chunk = moved.shape[0] // self.world_size
        shard = moved[self.rank * chunk : (self.rank + 1) * chunk]
        return shard.movedim(0, dim).contiguous()

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """In-place broadcast from group-local rank `src`.

        Pure byte movement, so it is exact for every dtype -- unlike the
        device transport, which builds broadcast out of all_gather because its
        reduce kernel does not dispatch int64.
        """
        inp = tensor.contiguous()
        flat = inp.view(-1)
        n = flat.numel()
        slot_elems = self.slot_bytes // inp.element_size()
        timeout = self._timeout()
        for start in range(0, n, slot_elems):
            end = min(start + slot_elems, n)
            self._ext.barlink_host_broadcast(
                flat[start:end], self._slot_addrs, self._flags_addr, self._seq,
                self._blk, self.rank, self.world_size, src, timeout,
                self._blocks,
            )
        if inp.data_ptr() != tensor.data_ptr():
            tensor.copy_(inp.view(tensor.shape))
        return tensor

    # ------------------------------------------------------------------
    # point-to-point
    #
    # NOT dispatched from barlink.py (the communicator exposes collectives
    # only). Kept here because the 7.3 us number this transport exists for is
    # a ping-pong number, and because a p2p pair is what a future PD-KV or
    # pipeline seam would need. Per-pair buffers and per-pair sequence
    # counters, so send/recv do NOT have to be group-uniform -- only matched
    # pairwise and in order between the two ranks involved.
    # ------------------------------------------------------------------

    def send(self, tensor: torch.Tensor, dst: int) -> None:
        if self.p2p_bytes <= 0:
            raise RuntimeError(
                "barlink host: point-to-point disabled "
                "(SGLANG_BARLINK_HOST_P2P_MIB=0)."
            )
        inp = tensor.contiguous()
        flat = inp.view(-1)
        n = flat.numel()
        chunk = self.p2p_bytes // inp.element_size()
        addrs = self._pair_addrs[(self.rank, dst)]
        timeout = self._timeout()
        # A ROW, not a 1-element slice: `seq_dev[1]` is this pair's abort word
        # (#653), so the view handed to the kernel must be the [seq, abort]
        # pair. A `[dst:dst+1]` slice of a flat counter array would make the
        # kernel's abort store land on the NEXT destination's sequence number.
        seq = self._send_seq[dst]
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            self._ext.barlink_host_send(
                flat[start:end], addrs, self._flags_addr, seq, self._blk,
                self.rank, dst, timeout, self._blocks,
            )

    def recv(self, tensor: torch.Tensor, src: int) -> torch.Tensor:
        if self.p2p_bytes <= 0:
            raise RuntimeError(
                "barlink host: point-to-point disabled "
                "(SGLANG_BARLINK_HOST_P2P_MIB=0)."
            )
        out = tensor if tensor.is_contiguous() else tensor.contiguous()
        flat = out.view(-1)
        n = flat.numel()
        chunk = self.p2p_bytes // out.element_size()
        addrs = self._pair_addrs[(src, self.rank)]
        timeout = self._timeout()
        # A ROW, not a 1-element slice -- see `send` for why.
        seq = self._recv_seq[src]
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            self._ext.barlink_host_recv(
                flat[start:end], addrs, self._flags_addr, seq, self._blk,
                self.rank, src, timeout, self._blocks,
            )
        if out.data_ptr() != tensor.data_ptr():
            tensor.copy_(out.view(tensor.shape))
        return tensor

    # ------------------------------------------------------------------
    # host-side observation (NOT the hot path)
    # ------------------------------------------------------------------

    def completed_seq(self, rank: int | None = None) -> int:
        """Last collective this rank finished, read from the pinned flag.

        A caller that needs to know on the HOST that an op is done can poll
        this instead of calling cudaStreamSynchronize: the flag is written by
        the closing kernel into page-locked memory, so the read is a load from
        RAM, not a driver round trip. Used for diagnostics and by the
        benchmark; the transport itself never waits on the host.
        """
        r = self.rank if rank is None else rank
        return int(self._flags_np[r, 1])

    # ------------------------------------------------------------------
    # abort guard (#653)
    # ------------------------------------------------------------------

    def check_aborted(self, where: str) -> None:
        """Raise if a spin kernel took its abort path. Called after collectives.

        WHAT IT READS. Column 1 of ``_seq_all`` -- the abort word of the
        collective counter and of every p2p counter in one strided read. A
        kernel that exceeds its cycle deadline writes its code there and
        returns, leaving the output buffer partially written; nothing else in
        the process would ever notice, which is exactly the state #583
        described and the reason a trap felt like the alternative.

        IMPLEMENTATION CHOICE: A PLAIN IN-LINE READ, deliberately, and NOT
        the device transport's watchdog-staged read. That machinery (#517)
        exists because ``BarlinkDeviceTransport.check_aborted`` sits on the
        production serving path, where an in-line device read synchronizes
        the compute stream and costs the overlap scheduler its run-ahead --
        task #600 measured it at ~7 ms of a 46.5 ms bs=1 round. This
        transport is not that path: it is an explicitly-selected measurement
        vehicle (``SGLANG_BARLINK_TRANSPORT=host``, see the no-fallback note
        in barlink.py), reached only when somebody chose it on purpose. A
        staged read here would add a private stream, a pinned destination and
        a watchdog dependency to a guard nobody would ever exercise in that
        shape, and the first bug in it would be found by the incident it was
        supposed to report. One read that is obviously correct is worth more.

        WHAT IT SKIPS, mirroring the device transport exactly:
        * ``graph_capture_running()`` -- reading a device word inside a stream
          capture is illegal, not merely slow. The CUDA-graph replay boundary
          picks those kernels up instead, through ``barlink_abort_gate``,
          which this transport registers with at bring-up.
        * ``barlink_abort_gate.abort_check_enabled()`` -- one switch for every
          transport's guard.
        """
        seq_all = self._seq_all
        if seq_all is None:
            return
        from sglang.srt.distributed.device_communicators.barlink import (
            graph_capture_running,
        )

        if graph_capture_running():
            return
        if not barlink_abort_gate.abort_check_enabled():
            return
        code = int(seq_all[:, 1].max())
        if code == 0:
            return
        self._raise_aborted(where, code)

    def _raise_aborted(self, where: str, code: int) -> None:
        """The report. Named separately from ``check_aborted`` so the message
        has ONE definition, the way the device transport's does."""
        wait = _ABORT_WAITS.get(code, f"unknown abort code {code}")
        # #650: the PEER STATEMENT. Everything above is rank-local; the
        # lockstep sentinel's sidecar keeps exchanging while a peer's main
        # thread hangs, so its last gather states every peer's ring position
        # -- the one line that lets a survivor's report say WHERE the wedged
        # rank last was. Warn-never-raise, and imported here rather than at
        # module level because this module must stay importable without
        # creating a CUDA context.
        try:
            from sglang.srt.distributed.device_communicators import (
                lockstep_sentinel,
            )

            peer_stmt = lockstep_sentinel.peer_statement()
        except Exception:  # noqa: BLE001 - an instrument must not mask the abort
            peer_stmt = "peer statement: <unavailable>"
        raise HostCollectiveAborted(
            f"barlink host collective aborted at {where}: rank "
            f"{self.rank}/{self.world_size} gave up in {wait} after "
            f"{self._TIMEOUT_CYCLES} device cycles (~30 s at 2 GHz). "
            f"PEER POSITIONS (#650): {peer_stmt}. "
            f"The output buffer of that collective is partially written -- "
            f"the kernel returned mid-op -- so every result computed from it "
            f"is garbage, which is why this raises instead of logging. A peer "
            f"either died, diverged from the collective sequence, or was "
            f"starved of PCIe bandwidth long enough to miss the deadline. Set "
            f"{barlink_abort_gate.ENV_ENABLE}=0 to silence this check and "
            f"restore the behaviour of continuing over corrupt buffers; it "
            f"does not make the collective succeed.",
            rank=int(self.rank),
            world=int(self.world_size),
            code=int(code),
            where=where,
        )

    # ------------------------------------------------------------------
    # teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Unpin and release the segment (see BarlinkCommunicator.close)."""
        # Withdraw from the gate BEFORE the tensors go, or a replay-boundary
        # check could reach a transport whose counters are being torn down.
        if self._registered_in_gate:
            barlink_abort_gate.unregister(self)
            self._registered_in_gate = False
        try:
            del self._flags_np
        except Exception:
            pass
        try:
            _unregister(self._host_base)
        except Exception as e:
            logger.warning("barlink host: host-unregister failed (%s).", e)
        try:
            self._shm.close()
            if self.rank == 0:
                self._shm.unlink()
        except Exception:
            pass
