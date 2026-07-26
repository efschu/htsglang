# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the shvllm fork (branch feature/htccl),
# vllm/distributed/device_communicators/htccl_device.py
"""HTCCL device-driven transport.

The CPU-orchestrated shm transport costs ~150 us per collective (stream
syncs, spin on the CPU, Python dispatch) and, worse, forces piecewise
CUDA graphs because the op synchronizes with the host. This transport
moves the WHOLE collective onto the GPU, NCCL-style:

- The shared-memory segment is host-registered (cudaHostRegister /
  hipHostRegister), so under UVA every GPU can DMA-read and -write it
  directly from kernels (zero-copy mapped memory).
- Sequence flags live in the same mapped segment; a kernel spin-waits
  on the peers' flags with volatile loads — no CPU involvement, no
  stream synchronization.
- The op therefore consists of four tiny stream-ordered kernels
  (begin / stage / reduce / done) and is fully CUDA-graph-capturable:
  the per-op sequence number lives in DEVICE memory and is incremented
  by the kernel itself, so graph replays keep working.

Deadlock safety relies on the same invariant NCCL kernels use: all TP
ranks issue the identical sequence of collectives (SPMD). A cycle-based
timeout traps the kernel if a peer diverges.

The CUDA source below sticks to constructs that hipify translates 1:1
(__threadfence_system, clock64, volatile loads), so the identical file
serves the ROCm build once the AMD card is installed.
"""

import logging
import os
import pathlib
import time
from contextlib import contextmanager

import torch
from torch.distributed import ProcessGroup

# NOT a module-level `from sglang.srt.utils.jit_cold_build import ...`:
# importing the `sglang.srt.utils` PACKAGE runs its __init__, which creates a
# CUDA context, and this file must stay importable on the ROCm/CPU-only rank
# of a cross-vendor group (pinned by
# test_htccl_port.py::test_import_does_not_initialize_cuda). By the time a
# collective is launched the module is long since imported, so the lazy
# lookup below is a cached global read.
_resolve_timeout_cycles = None


def resolve_timeout_cycles(base_cycles: int) -> int:
    """Deadline for a device collective right now -- see utils/jit_cold_build."""
    global _resolve_timeout_cycles
    if _resolve_timeout_cycles is None:
        from sglang.srt.utils.jit_cold_build import (
            resolve_timeout_cycles as _resolver,
        )

        _resolve_timeout_cycles = _resolver
    return _resolve_timeout_cycles(base_cycles)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Calibration-persistence stub.
#
# In the vLLM fork these came from `vllm.shvllm_tune`, a fork-local cache that
# lets a second start skip the startup calibrations by replaying a bundle
# string. That subsystem is unrelated to collective correctness, so it is NOT
# ported; the live self-calibration below still runs, it just does not persist.
# The env overrides (SGLANG_HTCCL_RSAG_SHARES / SGLANG_HTCCL_PIPE_CHUNK_MIB)
# remain the supported way to pin the calibrated values, and they are what
# matters for reproducibility.
#
# RANK-UNIFORMITY: these must return the SAME value on every rank. Returning a
# constant None is trivially uniform. Do not make this read per-rank state.
# ---------------------------------------------------------------------------


def _tune_get(key: str):
    return None


def _tune_report(key: str, value: str) -> None:
    return None


_ONCE_SEEN: set = set()


def _info_once(msg: str, *args) -> None:
    """sglang's logger has no `info_once`; keep the vLLM call sites intact."""
    if msg in _ONCE_SEEN:
        return
    _ONCE_SEEN.add(msg)
    logger.info(msg, *args)

_MAX_RANKS = 8

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define HTCCL_MAX_RANKS 8

// __trap() is CUDA-only and hipify does NOT translate it: the AMD build fails
// with "use of undeclared identifier '__trap'". The note at the top of this
// module lists __threadfence_system / clock64 / volatile loads as the
// constructs that survive hipify 1:1 -- __trap was simply not among them and
// had never been compiled for HIP. abort() is HIP's device-side equivalent and
// lowers to s_trap on amdgcn.
#if defined(__HIP_PLATFORM_AMD__) || defined(USE_ROCM)
#define HTCCL_TRAP() abort()
#else
#define HTCCL_TRAP() __trap()
#endif

using u64 = unsigned long long;

struct SlotPtrs {
  const void* p[HTCCL_MAX_RANKS];
};

__device__ __forceinline__ u64 vload(const volatile u64* p) { return *p; }

// Explicit converters — the implicit __half/__nv_bfloat16 <-> float
// operators are not available in every compilation context.
__device__ __forceinline__ float to_f(float v) { return v; }
__device__ __forceinline__ float to_f(__half v) { return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v) {
  return __bfloat162float(v);
}
template <typename T>
__device__ __forceinline__ T from_f(float v);
template <>
__device__ __forceinline__ float from_f<float>(float v) { return v; }
template <>
__device__ __forceinline__ __half from_f<__half>(float v) {
  return __float2half(v);
}
template <>
__device__ __forceinline__ __nv_bfloat16 from_f<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

// K0: bump the device-side sequence number and wait until every rank
// has CONSUMED the previous op (our slot may be overwritten again).
__global__ void htccl_begin_kernel(u64* seq_dev,
                                   const volatile u64* cons_flags,
                                   int world, u64 timeout) {
  u64 seq = *seq_dev + 1;
  *seq_dev = seq;
  u64 start = clock64();
  for (int r = 0; r < world; ++r) {
    while (vload(&cons_flags[r * 8]) < seq - 1) {
      if (clock64() - start > timeout) { HTCCL_TRAP(); }
    }
  }
}

#define HTCCL_MAX_CHUNKS 64

// K1: publish "my slot holds chunk #c of op #seq". Runs after the D2H
// DMA copy in stream order; the system fence orders the flag store
// behind it.
__global__ void htccl_publish_kernel(volatile u64* pub_flags,
                                     const u64* seq_dev, int rank,
                                     int chunk) {
  __threadfence_system();
  pub_flags[rank * HTCCL_MAX_CHUNKS + chunk] = *seq_dev;
}

// K2: spin until every peer published chunk #c of op #seq.
__global__ void htccl_wait_kernel(const volatile u64* pub_flags,
                                  const u64* seq_dev, int rank, int world,
                                  int chunk, u64 timeout) {
  u64 seq = *seq_dev;
  u64 start = clock64();
  for (int r = 0; r < world; ++r) {
    if (r == rank) continue;
    while (vload(&pub_flags[r * HTCCL_MAX_CHUNKS + chunk]) < seq) {
      if (clock64() - start > timeout) { HTCCL_TRAP(); }
    }
  }
  __threadfence_system();
}

// K2b: spin until ONE specific rank published flag column #idx.
__global__ void htccl_wait_one_kernel(const volatile u64* pub_flags,
                                      const u64* seq_dev, int owner,
                                      int idx, u64 timeout) {
  u64 seq = *seq_dev;
  u64 start = clock64();
  while (vload(&pub_flags[owner * HTCCL_MAX_CHUNKS + idx]) < seq) {
    if (clock64() - start > timeout) { HTCCL_TRAP(); }
  }
  __threadfence_system();
}

// K3: out = inp + sum(peer payloads) — everything in device memory
// (the peer data was H2D-DMA'd into scratch), fp32 accumulation.
template <typename T>
__global__ void htccl_add_kernel(const T* __restrict__ inp,
                                 T* __restrict__ out,
                                 const T* __restrict__ scratch,
                                 int npeers, size_t n) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    float acc = to_f(inp[i]);
    for (int p = 0; p < npeers; ++p) {
      acc += to_f(scratch[(size_t)p * n + i]);
    }
    out[i] = from_f<T>(acc);
  }
}

// K4: publish "I consumed every slot of this op".
__global__ void htccl_done_kernel(volatile u64* cons_flags,
                                  const u64* seq_dev, int rank) {
  cons_flags[rank * 8] = *seq_dev;
}

template <typename T>
void launch(const at::Tensor& inp, at::Tensor& out, at::Tensor& scratch,
            int rank, int world, at::Tensor& seq_dev,
            const std::vector<int64_t>& slot_addrs, int64_t pub_addr,
            int64_t cons_addr, int64_t timeout, cudaStream_t stream) {
  size_t n = inp.numel();
  size_t nbytes = n * sizeof(T);
  u64* seq = (u64*)seq_dev.data_ptr();
  volatile u64* pub = (volatile u64*)pub_addr;
  volatile u64* cons = (volatile u64*)cons_addr;
  int threads = 256;
  int blocks = (int)std::min<size_t>((n + threads - 1) / threads, 128);

  htccl_begin_kernel<<<1, 1, 0, stream>>>(seq, (const volatile u64*)cons,
                                          world, (u64)timeout);
  // D2H: DMA into the own mapped slot at full PCIe bandwidth.
  cudaMemcpyAsync((void*)slot_addrs[rank], inp.data_ptr(), nbytes,
                  cudaMemcpyDefault, stream);
  htccl_publish_kernel<<<1, 1, 0, stream>>>(pub, (const u64*)seq, rank, 0);
  htccl_wait_kernel<<<1, 1, 0, stream>>>((const volatile u64*)pub,
                                         (const u64*)seq, rank, world, 0,
                                         (u64)timeout);
  // H2D: DMA every peer slot into device scratch.
  T* scr = (T*)scratch.data_ptr();
  int p = 0;
  for (int r = 0; r < world; ++r) {
    if (r == rank) continue;
    cudaMemcpyAsync((void*)(scr + (size_t)p * n), (const void*)slot_addrs[r],
                    nbytes, cudaMemcpyDefault, stream);
    ++p;
  }
  htccl_add_kernel<T><<<blocks, threads, 0, stream>>>(
      (const T*)inp.data_ptr(), (T*)out.data_ptr(), (const T*)scr,
      world - 1, n);
  htccl_done_kernel<<<1, 1, 0, stream>>>(cons, (const u64*)seq, rank);
}

// Pipelined variant for large payloads: stream S1 uploads chunk after
// chunk (D2H copy engine) and publishes per-chunk flags; stream S2
// pulls each peer chunk (H2D copy engine) as soon as its flags arrive
// and accumulates. Upload and download overlap — PCIe is full duplex.
// Events are persistent (created once); the whole schedule is CUDA-
// graph-capturable.
static cudaEvent_t htccl_fork_ev = nullptr;
static cudaEvent_t htccl_join_ev = nullptr;

template <typename T>
void launch_pipelined(const at::Tensor& inp, at::Tensor& out,
                      at::Tensor& scratch, int rank, int world,
                      at::Tensor& seq_dev,
                      const std::vector<int64_t>& slot_addrs,
                      int64_t pub_addr, int64_t cons_addr, int64_t timeout,
                      size_t chunk_elems, cudaStream_t s1,
                      cudaStream_t s2) {
  if (htccl_fork_ev == nullptr) {
    cudaEventCreateWithFlags(&htccl_fork_ev, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&htccl_join_ev, cudaEventDisableTiming);
  }
  size_t n = inp.numel();
  int nchunks = (int)((n + chunk_elems - 1) / chunk_elems);
  TORCH_CHECK(nchunks <= HTCCL_MAX_CHUNKS, "htccl: too many chunks");
  u64* seq = (u64*)seq_dev.data_ptr();
  volatile u64* pub = (volatile u64*)pub_addr;
  volatile u64* cons = (volatile u64*)cons_addr;
  const T* in_ptr = (const T*)inp.data_ptr();
  T* out_ptr = (T*)out.data_ptr();
  T* scr = (T*)scratch.data_ptr();
  int threads = 256;

  htccl_begin_kernel<<<1, 1, 0, s1>>>(seq, (const volatile u64*)cons, world,
                                      (u64)timeout);
  cudaEventRecord(htccl_fork_ev, s1);
  cudaStreamWaitEvent(s2, htccl_fork_ev, 0);

  for (int c = 0; c < nchunks; ++c) {
    size_t off = (size_t)c * chunk_elems;
    size_t cn = std::min(chunk_elems, n - off);
    cudaMemcpyAsync((void*)((T*)slot_addrs[rank] + off), in_ptr + off,
                    cn * sizeof(T), cudaMemcpyDefault, s1);
    htccl_publish_kernel<<<1, 1, 0, s1>>>(pub, (const u64*)seq, rank, c);
  }
  for (int c = 0; c < nchunks; ++c) {
    size_t off = (size_t)c * chunk_elems;
    size_t cn = std::min(chunk_elems, n - off);
    int blocks = (int)std::min<size_t>((cn + threads - 1) / threads, 128);
    htccl_wait_kernel<<<1, 1, 0, s2>>>((const volatile u64*)pub,
                                       (const u64*)seq, rank, world, c,
                                       (u64)timeout);
    int p = 0;
    for (int r = 0; r < world; ++r) {
      if (r == rank) continue;
      cudaMemcpyAsync((void*)(scr + (size_t)p * cn),
                      (const void*)((const T*)slot_addrs[r] + off),
                      cn * sizeof(T), cudaMemcpyDefault, s2);
      ++p;
    }
    htccl_add_kernel<T><<<blocks, threads, 0, s2>>>(
        in_ptr + off, out_ptr + off, (const T*)scr, world - 1, cn);
  }
  cudaEventRecord(htccl_join_ev, s2);
  cudaStreamWaitEvent(s1, htccl_join_ev, 0);
  htccl_done_kernel<<<1, 1, 0, s1>>>(cons, (const u64*)seq, rank);
}

// Reduce-scatter + all-gather schedule for world >= 3: every rank
// reduces only its OWN contiguous block of chunks (phase 1) and pulls
// the other blocks already reduced from their owners (phase 2, staged
// in the second half of the owner's slot). Per-rank traffic drops from
// (W-1) x payload down to 2(W-1)/W x payload per direction — the same
// volume NCCL's ring moves. Phase-2 flags live in flag columns 32+c.
template <typename T>
void launch_rs_ag(const at::Tensor& inp, at::Tensor& out,
                  at::Tensor& scratch, int rank, int world,
                  at::Tensor& seq_dev,
                  const std::vector<int64_t>& slot_addrs, int64_t pub_addr,
                  int64_t cons_addr, int64_t timeout, size_t chunk_elems,
                  size_t region2_off_elems,
                  const std::vector<int64_t>& owner_weights, cudaStream_t s1,
                  cudaStream_t s2) {
  if (htccl_fork_ev == nullptr) {
    cudaEventCreateWithFlags(&htccl_fork_ev, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&htccl_join_ev, cudaEventDisableTiming);
  }
  size_t n = inp.numel();
  int nchunks = (int)((n + chunk_elems - 1) / chunk_elems);
  TORCH_CHECK(nchunks <= 32, "htccl rs_ag: too many chunks");
  // Contiguous chunk ownership, weighted by measured per-rank link
  // bandwidth: a rank on a slower PCIe link owns fewer chunks, so the
  // per-rank transfer times equalize on heterogeneous links.
  int counts[HTCCL_MAX_RANKS];
  {
    int64_t wsum = 0;
    for (int r = 0; r < world; ++r) wsum += owner_weights[r];
    int assigned = 0;
    for (int r = 0; r < world; ++r) {
      counts[r] = (int)((nchunks * owner_weights[r]) / wsum);
      assigned += counts[r];
    }
    // Distribute the rounding remainder to the highest-weight ranks.
    for (int r = 0; assigned < nchunks; r = (r + 1) % world) {
      ++counts[r];
      ++assigned;
    }
  }
  auto owner_of = [&](int c) {
    int acc = 0;
    for (int r = 0; r < world; ++r) {
      if (c < acc + counts[r]) return r;
      acc += counts[r];
    }
    return world - 1;
  };
  auto first_chunk_of = [&](int r) {
    int acc = 0;
    for (int q = 0; q < r; ++q) acc += counts[q];
    return acc;
  };
  u64* seq = (u64*)seq_dev.data_ptr();
  volatile u64* pub = (volatile u64*)pub_addr;
  volatile u64* cons = (volatile u64*)cons_addr;
  const T* in_ptr = (const T*)inp.data_ptr();
  T* out_ptr = (T*)out.data_ptr();
  T* scr = (T*)scratch.data_ptr();
  int threads = 256;

  htccl_begin_kernel<<<1, 1, 0, s1>>>(seq, (const volatile u64*)cons, world,
                                      (u64)timeout);
  cudaEventRecord(htccl_fork_ev, s1);
  cudaStreamWaitEvent(s2, htccl_fork_ev, 0);

  // Phase 0 (S1): upload the full payload chunk-wise, publish per chunk.
  for (int c = 0; c < nchunks; ++c) {
    size_t off = (size_t)c * chunk_elems;
    size_t cn = std::min(chunk_elems, n - off);
    cudaMemcpyAsync((void*)((T*)slot_addrs[rank] + off), in_ptr + off,
                    cn * sizeof(T), cudaMemcpyDefault, s1);
    htccl_publish_kernel<<<1, 1, 0, s1>>>(pub, (const u64*)seq, rank, c);
  }
  // Phase 1 (S2): reduce the own chunks, stage the reduced result into
  // the second slot region, publish flag column 32+c.
  int my_first = first_chunk_of(rank);
  for (int c = 0; c < nchunks; ++c) {
    if (owner_of(c) != rank) continue;
    size_t off = (size_t)c * chunk_elems;
    size_t cn = std::min(chunk_elems, n - off);
    int blocks = (int)std::min<size_t>((cn + threads - 1) / threads, 128);
    htccl_wait_kernel<<<1, 1, 0, s2>>>((const volatile u64*)pub,
                                       (const u64*)seq, rank, world, c,
                                       (u64)timeout);
    int p = 0;
    for (int r = 0; r < world; ++r) {
      if (r == rank) continue;
      cudaMemcpyAsync((void*)(scr + (size_t)p * cn),
                      (const void*)((const T*)slot_addrs[r] + off),
                      cn * sizeof(T), cudaMemcpyDefault, s2);
      ++p;
    }
    htccl_add_kernel<T><<<blocks, threads, 0, s2>>>(
        in_ptr + off, out_ptr + off, (const T*)scr, world - 1, cn);
    size_t r2off = region2_off_elems + ((size_t)(c - my_first)) * chunk_elems;
    cudaMemcpyAsync((void*)((T*)slot_addrs[rank] + r2off), out_ptr + off,
                    cn * sizeof(T), cudaMemcpyDefault, s2);
    htccl_publish_kernel<<<1, 1, 0, s2>>>(pub, (const u64*)seq, rank,
                                          32 + c);
  }
  // Phase 2 (S2): pull every foreign chunk already reduced by its owner.
  for (int c = 0; c < nchunks; ++c) {
    int owner = owner_of(c);
    if (owner == rank) continue;
    size_t off = (size_t)c * chunk_elems;
    size_t cn = std::min(chunk_elems, n - off);
    htccl_wait_one_kernel<<<1, 1, 0, s2>>>((const volatile u64*)pub,
                                           (const u64*)seq, owner, 32 + c,
                                           (u64)timeout);
    size_t r2off = region2_off_elems +
                   ((size_t)(c - first_chunk_of(owner))) * chunk_elems;
    cudaMemcpyAsync((void*)(out_ptr + off),
                    (const void*)((const T*)slot_addrs[owner] + r2off),
                    cn * sizeof(T), cudaMemcpyDefault, s2);
  }
  cudaEventRecord(htccl_join_ev, s2);
  cudaStreamWaitEvent(s1, htccl_join_ev, 0);
  htccl_done_kernel<<<1, 1, 0, s1>>>(cons, (const u64*)seq, rank);
}

// All-gather over the same slots: stage own payload, wait for peers,
// then DMA every rank's slot into the right slice of `out`
// (out layout: [world, n] flattened).
void htccl_allgather(at::Tensor inp, at::Tensor out, int64_t rank,
                     int64_t world, at::Tensor seq_dev,
                     std::vector<int64_t> slot_addrs, int64_t pub_addr,
                     int64_t cons_addr, int64_t timeout) {
  TORCH_CHECK(world <= HTCCL_MAX_RANKS, "htccl: too many ranks");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous());
  TORCH_CHECK(out.numel() == inp.numel() * world, "htccl: bad gather out");
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  size_t nbytes = inp.numel() * inp.element_size();
  u64* seq = (u64*)seq_dev.data_ptr();
  volatile u64* pub = (volatile u64*)pub_addr;
  volatile u64* cons = (volatile u64*)cons_addr;
  char* out_base = (char*)out.data_ptr();

  htccl_begin_kernel<<<1, 1, 0, stream>>>(seq, (const volatile u64*)cons,
                                          (int)world, (u64)timeout);
  cudaMemcpyAsync((void*)slot_addrs[rank], inp.data_ptr(), nbytes,
                  cudaMemcpyDefault, stream);
  htccl_publish_kernel<<<1, 1, 0, stream>>>(pub, (const u64*)seq, (int)rank,
                                            0);
  htccl_wait_kernel<<<1, 1, 0, stream>>>((const volatile u64*)pub,
                                         (const u64*)seq, (int)rank,
                                         (int)world, 0, (u64)timeout);
  for (int r = 0; r < world; ++r) {
    const void* src = r == rank ? inp.data_ptr() : (const void*)slot_addrs[r];
    cudaMemcpyAsync((void*)(out_base + (size_t)r * nbytes), src, nbytes,
                    cudaMemcpyDefault, stream);
  }
  htccl_done_kernel<<<1, 1, 0, stream>>>(cons, (const u64*)seq, (int)rank);
}

void htccl_allreduce(at::Tensor inp, at::Tensor out, at::Tensor scratch,
                     int64_t rank, int64_t world, at::Tensor seq_dev,
                     std::vector<int64_t> slot_addrs, int64_t pub_addr,
                     int64_t cons_addr, int64_t timeout,
                     int64_t chunk_elems, int64_t stream2,
                     int64_t region2_off_elems,
                     std::vector<int64_t> owner_weights) {
  TORCH_CHECK(world <= HTCCL_MAX_RANKS, "htccl: too many ranks");
  TORCH_CHECK((int64_t)owner_weights.size() >= world,
              "htccl: owner_weights too short");
  TORCH_CHECK(inp.is_contiguous() && out.is_contiguous());
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  bool pipe = chunk_elems > 0 && (size_t)inp.numel() > (size_t)chunk_elems;
  // world >= 3 and at least one chunk per rank: use the RS+AG schedule.
  bool rs_ag = pipe && world >= 3 && region2_off_elems > 0 &&
               (size_t)inp.numel() > (size_t)chunk_elems * (world - 1);
  TORCH_CHECK(scratch.numel() >=
                  (pipe ? (size_t)chunk_elems : inp.numel()) * (world - 1),
              "htccl: scratch too small");
  switch (inp.scalar_type()) {
    case at::kBFloat16:
      if (rs_ag)
        launch_rs_ag<__nv_bfloat16>(inp, out, scratch, (int)rank, (int)world,
                           seq_dev, slot_addrs, pub_addr, cons_addr,
                           timeout, (size_t)chunk_elems,
                           (size_t)region2_off_elems, owner_weights, stream,
                           (cudaStream_t)stream2);
      else if (pipe)
        launch_pipelined<__nv_bfloat16>(inp, out, scratch, (int)rank,
                                        (int)world, seq_dev, slot_addrs,
                                        pub_addr, cons_addr, timeout,
                                        (size_t)chunk_elems, stream,
                                        (cudaStream_t)stream2);
      else
        launch<__nv_bfloat16>(inp, out, scratch, (int)rank, (int)world,
                              seq_dev, slot_addrs, pub_addr, cons_addr,
                              timeout, stream);
      break;
    case at::kHalf:
      if (rs_ag)
        launch_rs_ag<__half>(inp, out, scratch, (int)rank, (int)world,
                           seq_dev, slot_addrs, pub_addr, cons_addr,
                           timeout, (size_t)chunk_elems,
                           (size_t)region2_off_elems, owner_weights, stream,
                           (cudaStream_t)stream2);
      else if (pipe)
        launch_pipelined<__half>(inp, out, scratch, (int)rank, (int)world,
                                 seq_dev, slot_addrs, pub_addr, cons_addr,
                                 timeout, (size_t)chunk_elems, stream,
                                 (cudaStream_t)stream2);
      else
        launch<__half>(inp, out, scratch, (int)rank, (int)world, seq_dev,
                       slot_addrs, pub_addr, cons_addr, timeout, stream);
      break;
    case at::kFloat:
      if (rs_ag)
        launch_rs_ag<float>(inp, out, scratch, (int)rank, (int)world,
                           seq_dev, slot_addrs, pub_addr, cons_addr,
                           timeout, (size_t)chunk_elems,
                           (size_t)region2_off_elems, owner_weights, stream,
                           (cudaStream_t)stream2);
      else if (pipe)
        launch_pipelined<float>(inp, out, scratch, (int)rank, (int)world,
                                seq_dev, slot_addrs, pub_addr, cons_addr,
                                timeout, (size_t)chunk_elems, stream,
                                (cudaStream_t)stream2);
      else
        launch<float>(inp, out, scratch, (int)rank, (int)world, seq_dev,
                      slot_addrs, pub_addr, cons_addr, timeout, stream);
      break;
    default:
      TORCH_CHECK(false, "htccl: unsupported dtype");
  }
}
"""

_CPP_SRC = """
void htccl_allreduce(at::Tensor inp, at::Tensor out, at::Tensor scratch,
                     int64_t rank, int64_t world, at::Tensor seq_dev,
                     std::vector<int64_t> slot_addrs, int64_t pub_addr,
                     int64_t cons_addr, int64_t timeout,
                     int64_t chunk_elems, int64_t stream2,
                     int64_t region2_off_elems,
                     std::vector<int64_t> owner_weights);
void htccl_allgather(at::Tensor inp, at::Tensor out, int64_t rank,
                     int64_t world, at::Tensor seq_dev,
                     std::vector<int64_t> slot_addrs, int64_t pub_addr,
                     int64_t cons_addr, int64_t timeout);
"""

_ext = None


def _local_vendor() -> str:
    """``"hip"`` or ``"cuda"``, taken from the RUNTIME build.

    `torch.version.hip` is set only on a ROCm build and is None on a CUDA one.
    Deliberately NOT inferred from an architecture string: conflating the two
    namespaces is exactly the defect this module had. `torch.cuda.*` is the
    same API on both builds (ROCm maps it onto HIP), so the API used says
    nothing about the vendor -- only `torch.version` does.
    """
    return "hip" if torch.version.hip else "cuda"


def _local_arches(vendor: str) -> list[str]:
    """This rank's architectures, in its OWN vendor's namespace.

    CUDA yields capability strings ("7.5"); HIP yields gfx names ("gfx900").
    The two namespaces overlap numerically and must never be mixed: gfx900
    reports `get_device_capability() == (9, 0)`, i.e. the string "9.0", which
    is ALSO NVIDIA Hopper. A group containing a Vega therefore used to make
    every CUDA rank compile an sm_90 cubin. Returning gfx names on HIP removes
    the collision at the source rather than trying to disambiguate later.
    """
    import re

    if vendor == "hip":
        env = os.environ.get("PYTORCH_ROCM_ARCH")
        if env and env.strip():
            return sorted({s.strip().split(":")[0]
                           for s in re.split(r"[;,\s]+", env) if s.strip()})
        # Clamp to what this torch build actually contains. On ROCm
        # get_arch_list() returns gfx names, so the CUDA-shaped `"sm_" in a`
        # test matched nothing and the clamp silently did nothing on exactly
        # the platform whose namespace it could not parse. Parse gfx instead.
        supported = {a.split(":")[0] for a in torch.cuda.get_arch_list()
                     if a.startswith("gfx")}
        out = []
        for i in range(torch.cuda.device_count()):
            # gcnArchName carries feature suffixes ("gfx900:xnack-"); the base
            # name is what --offload-arch and get_arch_list() agree on.
            name = torch.cuda.get_device_properties(i).gcnArchName.split(":")[0]
            if supported and name not in supported:
                logger.warning(
                    "HTCCL device transport: this torch build lists %s but the "
                    "GPU reports %s; building for it anyway, which will fail if "
                    "the compiler cannot target it.",
                    ",".join(sorted(supported)), name,
                )
            out.append(name)
        return out

    env = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if env and env.strip().lower() != "native":
        # An explicit list wins (and is assumed rank-uniform, like every
        # other SGLANG_HTCCL* / build env in this transport).
        specs = [s.strip() for s in re.split(r"[;,\s]+", env) if s.strip()]
        return sorted({s.replace("+PTX", "").strip() for s in specs})
    # Clamp to what the installed nvcc/torch can actually emit, exactly
    # as torch.utils.cpp_extension._get_cuda_arch_flags does.
    supported_sm = [
        int("".join(re.findall(r"\d+", a.split("_")[1])))
        for a in torch.cuda.get_arch_list()
        if a.startswith("sm_")
    ]
    ceiling = max((sm // 10, sm % 10) for sm in supported_sm) if supported_sm else None
    out = []
    for i in range(torch.cuda.device_count()):
        cap = torch.cuda.get_device_capability(i)
        if ceiling is not None:
            cap = min(ceiling, cap)
        out.append(f"{cap[0]}.{cap[1]}")
    return out


def _arch_sort_key(vendor: str, arch: str):
    if vendor == "hip":
        return arch
    return tuple(int(x) for x in arch.split("."))


def _resolve_build_arches(cpu_group) -> dict:
    """Vendor -> sorted architectures present in the group, as (vendor, arch).

    Each rank can only see its OWN card: the fork isolates a rank to one
    physical GPU via ``CUDA_VISIBLE_DEVICES``, so ``torch.cuda.device_count()``
    is 1 inside a worker and torch's own arch inference (which walks the
    *visible* devices) yields a single architecture. On a mixed-arch rig that
    single architecture is whichever rank happened to compile first, and every
    other rank then loads a cubin its GPU cannot run. Union-ing over the group
    is therefore not an optimisation but the only way a rank can learn the
    node's real mix.

    What is gathered is (vendor, arch) PAIRS, never bare capability strings.
    A cross-vendor group holds architectures from two disjoint namespaces that
    can collide numerically, so a flat set of strings cannot represent it: the
    union is taken WITHIN each vendor and the vendors stay separate all the way
    through to the compiler flags and the cache key.
    """
    import torch.distributed as dist

    vendor = _local_vendor()
    local = [(vendor, a) for a in _local_arches(vendor)]
    gathered: list = [None] * dist.get_world_size(cpu_group)
    dist.all_gather_object(gathered, local, group=cpu_group)
    by_vendor: dict = {}
    for part in gathered:
        for v, a in part or ():
            by_vendor.setdefault(v, set()).add(a)
    return {
        v: sorted(arches, key=lambda a, _v=v: _arch_sort_key(_v, a))
        for v, arches in by_vendor.items()
    }


def _build_flags(vendor: str, arches: list) -> list:
    """Compiler flags for ONE vendor. No flag of one namespace ever reaches
    the compiler of the other: `-gencode` is nvcc-only and made hipcc fail
    outright, and `--offload-arch` is meaningless to nvcc."""
    if vendor == "hip":
        return [f"--offload-arch={a}" for a in arches]
    tags = [a.replace(".", "") for a in arches]
    flags = [f"-gencode=arch=compute_{t},code=sm_{t}" for t in tags]
    if tags:
        # PTX of the newest arch, so a card added later still runs.
        flags.append(f"-gencode=arch=compute_{tags[-1]},code=compute_{tags[-1]}")
    return flags


# ---------------------------------------------------------------------------
# torch_extensions cache hygiene (#181).
#
# This extension is built by torch's load_inline into
# $TORCH_EXTENSIONS_DIR/<name>/ and that takes ~150 s of nvcc. A boot killed
# inside that window leaves the directory holding main.cpp, cuda.cu,
# build.ninja and *.o -- and NO .so. torch hands the wreck to every later boot
# and ninja's mtime check will happily call it up to date, so one interrupted
# build becomes a permanent failure. Same poisoning class as the tvm-ffi JIT
# cache (#172b), different writer, so the MECHANICS are reused from
# jit_kernel.cache_health rather than copied: completeness by artifact, a
# host+pid+time build marker whose liveness check keeps a co-located rank's
# in-flight build from looking like residue, rename-before-delete purge, and
# never failing a boot on cache hygiene.
#
# THE ONE THING THAT IS GENUINELY DIFFERENT HERE: torch's extensions root is
# SHARED with every other cpp_extension on the machine, while the tvm-ffi root
# belongs to sglang alone. A half-built entry there may be another extension's
# live business, so this sweep is scoped BY NAME to our own entries. An
# unscoped sweep of that root would be a new bug, not a fix.
# ---------------------------------------------------------------------------

#: Every cache entry this module owns starts with this. The name is extended
#: with vendor + arches below; the prefix is what makes the sweep scopeable.
_EXT_NAME_PREFIX = "htccl_device_ext"

#: A finished cpp_extension build. `.pyd` is torch's Windows name for it.
_EXT_ARTIFACT_SUFFIXES = (".so", ".pyd")

_EXT_CACHE_LABEL = "torch extension cache"


def _ext_owned(name: str) -> bool:
    return name.startswith(_EXT_NAME_PREFIX)


def _ext_selfheal_enabled() -> bool:
    # Read per call, not at import: the switch exists so an operator can turn
    # the sweep off in place if it ever misjudges an entry.
    return os.environ.get("SGLANG_EXT_CACHE_SELFHEAL", "1") not in (
        "0",
        "false",
        "False",
    )


def _ext_build_dir(name: str) -> pathlib.Path:
    """The directory torch will build ``name`` into.

    Asked of torch itself rather than reconstructed: the layout depends on
    TORCH_EXTENSIONS_DIR, the python tag and the accelerator tag, and a
    reimplementation that drifted would heal one directory while torch built
    into another.
    """
    from torch.utils.cpp_extension import _get_build_directory

    return pathlib.Path(_get_build_directory(name, False))


@contextmanager
def _ext_cache_guarded(name: str):
    """Sweep our own poisoned entries, then hold the build marker for `name`.

    Cache hygiene must never be the reason a server fails to start, so every
    step is best-effort: on any error the build simply proceeds as it did
    before this existed.
    """
    build_dir = None
    marker = None
    try:
        from sglang.jit_kernel import cache_health

        build_dir = _ext_build_dir(name)
        if _ext_selfheal_enabled():
            # The whole root first (a wreck under a DIFFERENT arch name still
            # breaks the boot that wants that arch), then the specific entry
            # we are about to build into, so ninja starts clean.
            cache_health.sweep_cache_root(
                build_dir.parent,
                artifact_suffixes=_EXT_ARTIFACT_SUFFIXES,
                name_filter=_ext_owned,
                label=_EXT_CACHE_LABEL,
            )
            cache_health.heal_entry(
                build_dir,
                artifact_suffixes=_EXT_ARTIFACT_SUFFIXES,
                label=_EXT_CACHE_LABEL,
            )
        marker = cache_health.building_marker
    except Exception as exc:  # never fail a boot on cache hygiene
        logger.warning("HTCCL extension cache self-heal skipped: %r", exc)
        build_dir = None
        marker = None

    if marker is None:
        # No marker and no build_directory override: byte-identical to the
        # pre-#181 call, which is the right thing to degrade to.
        yield None
        return
    # The marker is what lets the NEXT process tell "a co-located rank is
    # compiling this right now" from "somebody was killed compiling this",
    # without a lock and without deleting live work. An orderly failure inside
    # the block clears it, so the entry is immediately recognisable as poison.
    with marker(build_dir):
        yield build_dir


def _load_ext(cpu_group):
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load_inline

        by_vendor = _resolve_build_arches(cpu_group)
        vendor = _local_vendor()
        # Build for THIS rank's vendor only. A cubin for gfx900 is meaningless
        # to nvcc and vice versa, so the cross-vendor union is a union of
        # artifacts, not of flags. Within a vendor the union is preserved,
        # which is what the mixed-NVIDIA case needs.
        arches = by_vendor.get(vendor, [])
        flags = _build_flags(vendor, arches)
        # The arch list belongs in the CACHE KEY, not just in the flags:
        # torch keys its JIT build directory by extension name and rebuilds
        # only when the SOURCES change. A name that ignores the architecture
        # lets a stale single-arch .so from an earlier run be handed back to
        # a rank it cannot run on -- which is the failure this guards, and it
        # survives the flags alone.
        #
        # The VENDOR belongs in it too. The name used to be built from the
        # group-wide arch union, which is identical on every rank by
        # construction -- and that is precisely what made it wrong across
        # vendors: one key naming two incompatible artifacts. Keying on
        # (vendor, that vendor's arches) keeps it deterministic per rank while
        # letting the two vendors' builds coexist.
        name = "htccl_device_ext_" + vendor
        if arches:
            name += "_" + "_".join(a.replace(".", "") for a in arches)
        t0 = time.time()
        # Discard a killed build's residue under our own names before trusting
        # this cache, and hold the build marker while nvcc runs -- see the
        # cache-hygiene block above. Passing build_directory explicitly ties
        # the directory that was healed and marked to the one torch builds
        # into, so the two can never drift apart.
        with _ext_cache_guarded(name) as build_dir:
            _ext = load_inline(
                name=name,
                cpp_sources=_CPP_SRC,
                cuda_sources=_CUDA_SRC,
                functions=["htccl_allreduce", "htccl_allgather"],
                extra_cuda_cflags=flags or None,
                verbose=False,
                build_directory=str(build_dir) if build_dir is not None else None,
            )
        _info_once(
            "HTCCL device extension %r built for %s arch(es) %s in %.1f s "
            "(group: %s)",
            name, vendor, ",".join(arches) or "<torch default>",
            time.time() - t0,
            "; ".join(f"{v}:{','.join(a)}" for v, a in sorted(by_vendor.items())),
        )
    return _ext


class HTCCLDeviceTransport:
    """GPU-driven all-reduce over the mapped shm segment.

    Reuses HTCCLShmTransport's segment/registration/rendezvous; only
    the data path differs: kernels spin on the mapped flags instead of
    the CPU, so ops are CUDA-graph-capturable and never touch the host.
    """

    # ~30 s at 2 GHz; a diverged peer traps the kernel instead of
    # hanging the GPU forever.
    _TIMEOUT_CYCLES = 60_000_000_000

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        slot_bytes: int,
    ):
        from sglang.srt.distributed.device_communicators.htccl_shm import (
            HTCCLShmTransport,
        )

        self._shm = HTCCLShmTransport(
            cpu_group=cpu_group, device=device, slot_bytes=slot_bytes
        )
        if not self._shm._pinned:
            raise RuntimeError(
                "HTCCL device transport needs a host-registered shm "
                "segment (cudaHostRegister failed)."
            )
        self.device = device
        self.cpu_group = cpu_group
        self.rank = self._shm.rank
        self.world_size = self._shm.world_size
        self.slot_bytes = slot_bytes
        self._ext = _load_ext(cpu_group)

        import ctypes

        base = ctypes.addressof(ctypes.c_char.from_buffer(self._shm._shm.buf))
        # Header layout (see htccl_shm): CPU-path counters at offset 0;
        # device-path per-chunk publish flags at 4096
        # (rank * MAX_CHUNKS + chunk, u64), consumption counters at 8192.
        self._pub_addr = base + 4096
        self._cons_addr = base + 8192
        from sglang.srt.distributed.device_communicators.htccl_shm import (
            _HEADER_BYTES,
        )

        self._slot_addrs = [
            base + _HEADER_BYTES + r * slot_bytes
            for r in range(self.world_size)
        ]
        # Device-side sequence number — incremented by the kernel, so
        # captured graphs stay valid across replays.
        self._seq_dev = torch.zeros(1, dtype=torch.int64, device=device)
        # Per-rank RS+AG ownership weights, proportional to the measured
        # PCIe link bandwidth of each rank (heterogeneous 8x/4x links!).
        # Override/skip the measurement with SGLANG_HTCCL_RSAG_SHARES.
        self._owner_weights = self._resolve_owner_weights()
        # Persistent device scratch for the peers' H2D-DMA'd payloads.
        self._scratch: dict[torch.dtype, torch.Tensor] = {}
        # Zero the device-path flag rows before first use.
        if self.rank == 0:
            import numpy as np

            np.frombuffer(
                self._shm._shm.buf, dtype=np.uint64, count=1024, offset=4096
            )[:] = 0
        # Second stream for the pipelined path: S1 uploads chunks (D2H
        # engine) while S2 pulls peer chunks (H2D engine) and reduces.
        self._stream2 = torch.cuda.Stream(device=device)
        import os

        self._pipe_chunk_bytes = (
            int(os.environ.get("SGLANG_HTCCL_PIPE_CHUNK_MIB", "4")) * 1024 * 1024
        )
        import torch.distributed as dist

        dist.barrier(group=cpu_group)
        # After the barrier the transport is fully usable — calibrate
        # the pipeline chunk size on the real all_reduce path (skipped
        # when SGLANG_HTCCL_PIPE_CHUNK_MIB or the tune bundle provides it).
        self._resolve_pipe_chunk()
        _info_once(
            "HTCCL device transport up: %d ranks, GPU-driven collectives "
            "(CUDA-graph capturable).",
            self.world_size,
        )

    def _resolve_owner_weights(self) -> list[int]:
        import os

        import torch.distributed as dist

        # Specific env wins over the consolidated bundle, the bundle
        # wins over calibration.
        override = os.environ.get("SGLANG_HTCCL_RSAG_SHARES") or _tune_get(
            "rsag_shares"
        )
        if override:
            w = [int(x) for x in override.split(",")]
            if len(w) != self.world_size or any(v <= 0 for v in w):
                raise ValueError(
                    f"SGLANG_HTCCL_RSAG_SHARES={override!r} must name "
                    f"{self.world_size} positive integers."
                )
            return w
        if self.world_size < 3:
            return [1] * self.world_size
        # One-shot calibration: time a round-trip DMA through the own
        # mapped slot (D2H + H2D, 16 MiB) — that is exactly the transfer
        # RS+AG performs — and share the bandwidths via gloo.
        nbytes = min(16 * 1024 * 1024, self.slot_bytes)
        buf = torch.empty(nbytes, dtype=torch.uint8, device=self.device)
        slot = self._shm._slot_tensors[self.rank][:nbytes]
        for _ in range(3):  # warmup
            slot.copy_(buf, non_blocking=True)
            buf.copy_(slot, non_blocking=True)
        torch.cuda.synchronize(self.device)
        import time

        t0 = time.perf_counter()
        for _ in range(8):
            slot.copy_(buf, non_blocking=True)
            buf.copy_(slot, non_blocking=True)
        torch.cuda.synchronize(self.device)
        gbps = 16 * nbytes / (time.perf_counter() - t0) / 1e9
        gathered = [None] * self.world_size
        dist.all_gather_object(gathered, gbps, group=self.cpu_group)
        # Integerize to small weights (per-mille of total, gcd-reduced).
        import math

        total = sum(gathered)
        w = [max(1, round(20 * g / total)) for g in gathered]
        g = math.gcd(*w)
        w = [v // g for v in w]
        _info_once(
            "HTCCL RS+AG link calibration: %s GB/s -> ownership weights "
            "%s (skip with SGLANG_HTCCL_RSAG_SHARES=%s)",
            "/".join(f"{x:.1f}" for x in gathered),
            str(w),
            ",".join(str(v) for v in w),
        )
        if self.rank == 0:
            _tune_report("rsag_shares", ",".join(str(v) for v in w))
        return w

    def _resolve_pipe_chunk(self) -> None:
        import os

        if "SGLANG_HTCCL_PIPE_CHUNK_MIB" in os.environ:
            return  # explicit env, already applied in __init__
        bundled = _tune_get("pipe_mib")
        if bundled:
            self._pipe_chunk_bytes = int(bundled) * 1024 * 1024
            return
        import time

        import torch.distributed as dist

        # Collective sweep over the REAL all_reduce path (16 MiB bf16
        # payload). All ranks iterate the candidates in the same order
        # and decide on the summed times, so every rank picks the same
        # value — the per-chunk flag protocol requires agreement, hence
        # pipe_mib is a scalar, not per-rank.
        payload = torch.randn(
            8 * 1024 * 1024, dtype=torch.bfloat16, device=self.device
        )
        candidates = [1, 2, 4, 8]
        my_times = []
        for mib in candidates:
            self._pipe_chunk_bytes = mib * 1024 * 1024
            dist.barrier(group=self.cpu_group)
            for _ in range(2):  # warmup
                self.all_reduce(payload)
            torch.cuda.synchronize(self.device)
            dist.barrier(group=self.cpu_group)
            t0 = time.perf_counter()
            for _ in range(5):
                self.all_reduce(payload)
            torch.cuda.synchronize(self.device)
            my_times.append(time.perf_counter() - t0)
        gathered: list[list[float] | None] = [None] * self.world_size
        dist.all_gather_object(gathered, my_times, group=self.cpu_group)
        totals = [
            sum(times[i] for times in gathered)  # type: ignore[index]
            for i in range(len(candidates))
        ]
        best = candidates[totals.index(min(totals))]
        self._pipe_chunk_bytes = best * 1024 * 1024
        if self.rank == 0:
            logger.info(
                "HTCCL pipeline-chunk calibration: %s -> %d MiB (skip with "
                "SGLANG_HTCCL_PIPE_CHUNK_MIB=%d)",
                ", ".join(
                    f"{c} MiB {t * 1000 / 5 / self.world_size:.2f} ms"
                    for c, t in zip(candidates, totals)
                ),
                best,
                best,
            )
            _tune_report("pipe_mib", str(best))

    # -- pluggable-transport interface (see htccl.py "transport seam") --
    #
    # The device transport chunks internally, so it has no payload ceiling:
    # size is irrelevant to `handles`. It serves all three collectives, which
    # is exactly what the previous per-op `if self.device_transport is not
    # None` checks did -- this is a restatement of today's behaviour, not a
    # change. Nothing in the GPU-validated data path below is touched.
    HTCCL_OPS = frozenset(
        {"all_reduce", "all_gather", "reduce_scatter", "broadcast"}
    )

    def handles(self, op: str, nbytes: int) -> bool:
        return op in self.HTCCL_OPS

    def htccl_all_reduce(self, comm, inp: torch.Tensor) -> torch.Tensor:
        return self.all_reduce(inp)

    def htccl_broadcast(self, comm, tensor: torch.Tensor, src: int):
        """CUDA-GRAPH-CAPTURABLE broadcast: all_gather, then take src's slice.

        The communicator's generic broadcast is host-staged (pinned buffer +
        gloo), which is illegal inside a capture -- `cudaErrorStreamCapture
        Unsupported`. That never surfaced while a PyNccl communicator existed
        alongside HTCCL, because the speculative path preferred pynccl for its
        capture-time syncs. Suppressing PyNccl under HTCCL (required: NCCL
        cannot span vendors and ncclCommInitRank segfaults on a mixed group)
        removed that fallback and routed those syncs here, inside the EAGLE
        draft capture.

        all_gather is the right primitive to build this on: its kernel is pure
        byte movement (nbytes + cudaMemcpyAsync, no dtype dispatch and no
        arithmetic), so unlike a zero+all_reduce emulation it is exact for
        EVERY dtype -- including the int64 `topk_index` the draft-pick sync
        carries, which the reduce kernel does not even dispatch.

        Cost is world_size x payload instead of 1x. The payloads on this path
        are per-step draft picks (a few KiB), so this is not a hot path.
        """
        n = tensor.numel()
        gathered = self.all_gather(tensor.reshape(-1), dim=0)
        tensor.copy_(gathered[src * n : (src + 1) * n].view(tensor.shape))
        return tensor

    def htccl_all_gather(self, comm, inp: torch.Tensor, dim: int) -> torch.Tensor:
        return self.all_gather(inp, dim)

    def htccl_reduce_scatter(self, comm, inp: torch.Tensor, dim: int) -> torch.Tensor:
        return self.reduce_scatter(inp, dim)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        inp = input_.contiguous()
        out = torch.empty_like(inp)
        flat_in = inp.view(-1)
        flat_out = out.view(-1)
        # Payloads beyond the slot run as a fixed number of chunked
        # collectives — the chunk count depends only on the shape, so
        # this stays CUDA-graph-safe.
        esize = flat_in.element_size()
        if self.world_size >= 3:
            # RS+AG stages the reduced own-share in the slot's second
            # half — cap the outer slice at half the slot.
            slot_elems = (self.slot_bytes // 2) // esize
            region2_off_elems = (self.slot_bytes // 2) // esize
        else:
            slot_elems = self.slot_bytes // esize
            region2_off_elems = 0
        pipe_elems = self._pipe_chunk_bytes // esize
        scratch = self._scratch.get(inp.dtype)
        need = min(pipe_elems, max(flat_in.numel(), 1)) * (self.world_size - 1)
        if scratch is None or scratch.numel() < need:
            scratch = torch.empty(
                pipe_elems * (self.world_size - 1),
                dtype=inp.dtype,
                device=self.device,
            )
            self._scratch[inp.dtype] = scratch
        for start in range(0, flat_in.numel(), slot_elems):
            end = min(start + slot_elems, flat_in.numel())
            self._ext.htccl_allreduce(
                flat_in[start:end],
                flat_out[start:end],
                scratch,
                self.rank,
                self.world_size,
                self._seq_dev,
                self._slot_addrs,
                self._pub_addr,
                self._cons_addr,
                # Not the raw constant: during the pre-capture warmup a peer
                # may still be in nvcc building a first-call JIT kernel, and
                # the cycle deadline would trap this rank's wait kernel long
                # before it arrives. resolve_timeout_cycles() is the identity
                # outside that window, so the value baked into the CAPTURED
                # graph -- and every replay after it -- is unchanged.
                resolve_timeout_cycles(self._TIMEOUT_CYCLES),
                pipe_elems,
                self._stream2.cuda_stream,
                region2_off_elems,
                self._owner_weights,
            )
        return out

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += input_.dim()
        inp = input_.contiguous()
        nbytes = inp.numel() * inp.element_size()
        assert nbytes <= self.slot_bytes, (
            f"HTCCL device all_gather: payload {nbytes} B exceeds the shm "
            "slot; raise SGLANG_HTCCL_SLOT_MIB."
        )
        input_size = inp.size()
        output = torch.empty(
            (self.world_size,) + tuple(input_size),
            dtype=inp.dtype,
            device=inp.device,
        )
        self._ext.htccl_allgather(
            inp.view(-1),
            output.view(-1),
            self.rank,
            self.world_size,
            self._seq_dev,
            self._slot_addrs,
            self._pub_addr,
            self._cons_addr,
            # See all_reduce: relaxed only inside the cold-build window.
            resolve_timeout_cycles(self._TIMEOUT_CYCLES),
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
        # Full all-reduce + slice: bounded extra traffic at TP sizes 2-4,
        # trivially correct, stays graph-capturable.
        reduced = self.all_reduce(input_)
        # movedim(dim, 0) -- NOT movedim(0, dim). The two agree only for dim
        # in {0, 1}; from dim >= 2 they differ, and the old form left the
        # ORIGINAL axis 1 in front, so the scatter sliced the wrong axis while
        # every shape check still passed. Measured: shape (4,6,2) dim=2 sliced
        # 6 instead of 2; (2,4,6,2) dim=3 sliced 4 instead of 2. The signature
        # defaults to dim=-1, so a bare reduce_scatter(x) on ndim >= 3
        # scattered the wrong axis SILENTLY. 2-D happened to be correct, which
        # is why it survived.
        moved = reduced.movedim(dim, 0).contiguous()
        assert moved.shape[0] % self.world_size == 0
        chunk = moved.shape[0] // self.world_size
        shard = moved[self.rank * chunk : (self.rank + 1) * chunk]
        return shard.movedim(0, dim).contiguous()

    def close(self) -> None:
        """Release the underlying shm segment (see HTCCLCommunicator.close)."""
        self._shm.close()
