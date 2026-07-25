# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from the shvllm fork (branch feature/htccl),
# vllm/distributed/device_communicators/htccl.py
"""HTCCL — heterogeneous collective communication layer.

Vendor-neutral collectives for TP groups that span GPUs which share no
common device-native collective library (e.g. NVIDIA/NCCL + AMD/RCCL).
Every collective is executed over the host-staging path that NCCL itself
falls back to when P2P is unavailable:

    GPU (D2H, async) -> pinned host buffer -> gloo collective on the
    group's CPU process group -> (H2D, async) -> GPU

gloo runs entirely CPU-side, so the two endpoints of the collective may
be CUDA and ROCm processes — the device only ever performs plain
``memcpy`` to/from its own pinned staging buffer, which both vendors
implement identically.

Large tensors are processed in chunks and pipelined: while gloo reduces
chunk *i* on the CPU, the D2H copy of chunk *i+1* is already in flight
on the device's copy stream. On systems without P2P between the GPUs
this is functionally the same transport NCCL would use, so forcing
HTCCL on an all-NVIDIA group (``SGLANG_HTCCL=1``) is a faithful test bed
for the mixed-vendor case.

Limitations (v1):
- Collectives synchronize with the CPU, so they cannot be captured in
  CUDA graphs — run with ``--enforce-eager``.
- Reduction happens on the CPU inside gloo (fp32 accumulation for
  half/bfloat16 inputs via upcast).
"""

import logging
import os

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

# Chunk size for the D2H -> gloo -> H2D pipeline. Small enough to
# overlap copy and CPU reduction, large enough to amortize per-op
# latency. Tunable via SGLANG_HTCCL_CHUNK_MIB.
_CHUNK_BYTES = int(os.environ.get("SGLANG_HTCCL_CHUNK_MIB", "8")) * 1024 * 1024

# gloo reduces half/bfloat16 with fp32 accumulation only when the
# tensor is upcast explicitly; reducing bf16 directly through gloo
# accumulates in bf16 and loses precision vs NCCL. Upcast by default,
# disable with SGLANG_HTCCL_FP32_REDUCE=0 to trade accuracy for speed.
_FP32_REDUCE = bool(int(os.environ.get("SGLANG_HTCCL_FP32_REDUCE", "1")))


# Preferred data plane: "shm" (pinned shared-memory slots + GPU-side
# reduction, single-node — matches NCCL's no-P2P SHM path) or "gloo"
# (TCP data plane, also works multi-node; slower).
# "device" = GPU-driven kernels over the mapped segment (fastest,
# CUDA-graph-capturable), "shm" = CPU-orchestrated pinned staging,
# "gloo" = TCP data plane (also multi-node).
_TRANSPORT = os.environ.get("SGLANG_HTCCL_TRANSPORT", "device")
# Per-rank shm slot size; all_reduce payloads above this fall back to
# the gloo path. 64 MiB covers a 4096-token x 5120-hidden bf16 chunk.
_SLOT_BYTES = int(os.environ.get("SGLANG_HTCCL_SLOT_MIB", "64")) * 1024 * 1024


# ----------------------------------------------------------------------
# Transport seam
#
# A transport is anything exposing:
#     handles(op: str, nbytes: int) -> bool
#     htccl_<op>(comm, ...)            for each op it declares
# The communicator ASKS `handles` rather than knowing a transport's limits, so
# no transport-specific condition (e.g. the shm slot-size ceiling) lives at a
# call site. Adding a transport -- `ucx` for RDMA is the expected next one --
# is one registry entry plus its module; no dispatch site changes.
#
# `None` means "no transport object": the inline gloo data plane below, which
# is always available and is the universal fallback.
# ----------------------------------------------------------------------


def _make_device_transport(cpu_group, device):
    from sglang.srt.distributed.device_communicators.htccl_device import (
        HTCCLDeviceTransport,
    )

    return HTCCLDeviceTransport(
        cpu_group=cpu_group, device=device, slot_bytes=_SLOT_BYTES
    )


def _make_shm_transport(cpu_group, device):
    from sglang.srt.distributed.device_communicators.htccl_shm import (
        HTCCLShmTransport,
    )

    return HTCCLShmTransport(
        cpu_group=cpu_group, device=device, slot_bytes=_SLOT_BYTES
    )


# name -> factory. "gloo" is intentionally absent: it is the inline plane.
TRANSPORT_REGISTRY = {
    "device": _make_device_transport,
    "shm": _make_shm_transport,
}

# Transports that must NOT silently fall back to the gloo plane on failure.
# The device transport is the one case: the compilation config allowed CUDA
# graphs on the strength of it, so a CPU-orchestrated replacement would be
# captured and crash later. Preserved verbatim from the pre-refactor code.
_NO_FALLBACK = frozenset({"device"})


def _build_transport(name: str, cpu_group, device, disabled: bool):
    if disabled:
        return None
    factory = TRANSPORT_REGISTRY.get(name)
    if factory is None:
        return None  # "gloo" or an unknown name -> inline data plane
    if name in _NO_FALLBACK:
        return factory(cpu_group, device)
    try:
        return factory(cpu_group, device)
    except Exception as e:
        logger.warning(
            "HTCCL: %s transport unavailable (%s); using the gloo data plane.",
            name,
            e,
        )
        return None


class HTCCLCommunicator:
    """Host-staged collectives over the group's gloo CPU process group."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
    ):
        self.cpu_group = cpu_group
        self.device = device
        self.world_size = dist.get_world_size(cpu_group)
        self.rank = dist.get_rank(cpu_group)
        self.disabled = self.world_size == 1
        self.transport = _build_transport(
            _TRANSPORT, cpu_group, device, disabled=self.disabled
        )
        # Dedicated copy stream: D2H of the next chunk overlaps with the
        # CPU-side gloo reduction of the current one.
        self._stream = torch.cuda.Stream(device=device)
        # Pinned staging buffers, grown on demand and reused. Two
        # buffers per direction so chunk i+1 can stage while chunk i is
        # still being reduced/written back.
        self._host_bufs: list[torch.Tensor] = []
        self._host_buf_bytes = 0

    def _select(self, op: str, nbytes: int):
        """The transport for ``op`` at this size, or None for the gloo plane.

        One attribute test plus the transport's own `handles` -- the same shape
        at every dispatch site, so op coverage can no longer differ silently
        between ops the way it did when each site hard-coded its own condition.
        """
        t = self.transport
        if t is not None and t.handles(op, nbytes):
            return t
        return None

    def _get_out_buf(self, ref: torch.Tensor) -> torch.Tensor:
        """One FRESH output tensor per call — never a shape-keyed cache.

        This used to hand out a persistent per-(shape, dtype) buffer, on the
        theory that a piecewise CUDA graph captured downstream of the
        collective needs the result at a stable address. That reasoning does
        not survive contact with either path that exists:

        * the CPU transports (shm/gloo) synchronize with the host inside the
          collective, so they can never be inside a captured region at all;
        * the one graph-capturable transport (`htccl_device`) does NOT use
          this helper — it allocates with `torch.empty_like` per call, and is
          correct precisely because capture-time allocations come from the
          graph's private pool and therefore already replay at a stable
          address.

        Meanwhile the cache actively BREAKS the documented contract of
        `all_reduce` ("returns a new tensor, out-of-place"): two results of
        the same shape and dtype were the SAME tensor, so the second call
        silently overwrote the first while the model still held it. That is
        not a hypothetical — it corrupted the forward outright (garbage
        tokens, no crash, no hang) on every non-device transport.
        """
        return torch.empty_like(ref)

    def _get_host_bufs(self, nbytes: int, count: int = 2) -> list[torch.Tensor]:
        if self._host_buf_bytes < nbytes or len(self._host_bufs) < count:
            self._host_bufs = [
                torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
                for _ in range(count)
            ]
            self._host_buf_bytes = nbytes
        return self._host_bufs

    # ------------------------------------------------------------------
    # all_reduce
    # ------------------------------------------------------------------

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        """Sum-all-reduce ``input_`` across the group, host-staged.

        Returns a new tensor (out-of-place), matching the contract of
        the other vLLM all-reduce backends.
        """
        if self.disabled:
            return input_.clone()
        inp = input_.contiguous()
        nbytes = inp.numel() * inp.element_size()
        t = self._select("all_reduce", nbytes)
        if t is not None:
            return t.htccl_all_reduce(self, inp)
        out = self._get_out_buf(inp)

        reduce_dtype = (
            torch.float32
            if _FP32_REDUCE and inp.dtype in (torch.float16, torch.bfloat16)
            else inp.dtype
        )
        elem_bytes = torch.tensor([], dtype=reduce_dtype).element_size()
        chunk_elems = max(_CHUNK_BYTES // elem_bytes, 1)

        flat_in = inp.view(-1)
        flat_out = out.view(-1)
        n = flat_in.numel()
        n_chunks = (n + chunk_elems - 1) // chunk_elems

        bufs = self._get_host_bufs(min(n, chunk_elems) * elem_bytes)
        staged: list[tuple[int, int, torch.Tensor, torch.cuda.Event]] = []

        current = torch.cuda.current_stream(self.device)
        self._stream.wait_stream(current)

        def _stage(ci: int) -> None:
            start = ci * chunk_elems
            end = min(start + chunk_elems, n)
            host = (
                bufs[ci % len(bufs)][: (end - start) * elem_bytes]
                .view(reduce_dtype)[: end - start]
            )
            with torch.cuda.stream(self._stream):
                src = flat_in[start:end]
                if src.dtype != reduce_dtype:
                    src = src.to(reduce_dtype)
                host.copy_(src, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(self._stream)
            staged.append((start, end, host, ev))

        _stage(0)
        for ci in range(n_chunks):
            if ci + 1 < n_chunks:
                _stage(ci + 1)  # D2H of next chunk overlaps gloo below
            start, end, host, ev = staged[ci]
            ev.synchronize()
            dist.all_reduce(host, group=self.cpu_group)
            with torch.cuda.stream(self._stream):
                dst = flat_out[start:end]
                if host.dtype != dst.dtype:
                    dst.copy_(host.to(dst.dtype), non_blocking=False)
                else:
                    dst.copy_(host, non_blocking=True)

        current.wait_stream(self._stream)
        return out

    # ------------------------------------------------------------------
    # all_gather / reduce_scatter / broadcast
    # ------------------------------------------------------------------

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.disabled:
            return input_
        t = self._select(
            "all_gather", input_.numel() * input_.element_size()
        )
        if t is not None:
            return t.htccl_all_gather(self, input_, dim)
        if dim < 0:
            dim += input_.dim()
        inp = input_.contiguous()
        input_size = inp.size()

        host_in = torch.empty(
            inp.shape, dtype=inp.dtype, pin_memory=True
        )
        host_in.copy_(inp, non_blocking=False)
        host_out = [torch.empty_like(host_in) for _ in range(self.world_size)]
        dist.all_gather(host_out, host_in, group=self.cpu_group)

        output = torch.empty(
            (self.world_size,) + tuple(input_size),
            dtype=inp.dtype,
            device=inp.device,
        )
        for i, h in enumerate(host_out):
            output[i].copy_(h, non_blocking=True)
        torch.cuda.current_stream(self.device).synchronize()

        output = output.movedim(0, dim)
        return output.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.disabled:
            return input_
        t = self._select(
            "reduce_scatter", input_.numel() * input_.element_size()
        )
        if t is not None:
            return t.htccl_reduce_scatter(self, input_, dim)
        if dim < 0:
            dim += input_.dim()
        # Host-staged: full all-reduce, then slice this rank's shard.
        # For the small TP world sizes HTCCL targets (2-4 ranks) the
        # extra traffic vs a true reduce-scatter is bounded and the
        # code stays trivially correct. Axis handling mirrors the base
        # communicator's reduce_scatter exactly.
        reduced = self.all_reduce(input_)
        moved = reduced.movedim(0, dim).contiguous()
        assert moved.shape[0] % self.world_size == 0
        chunk = moved.shape[0] // self.world_size
        shard = moved[self.rank * chunk : (self.rank + 1) * chunk]
        return shard.movedim(0, dim).contiguous()

    # ------------------------------------------------------------------
    # out-parameter forms
    #
    # sglang calls these directly (they are NOT reachable from the dim-based
    # variants above), so leaving them out would silently route part of the
    # traffic back to NCCL -- which on a mixed-vendor group is not a slow
    # path but a hang. Both are pure compositions of the collectives above:
    # they introduce NO new collective, which keeps the rank-uniformity
    # argument unchanged.
    # ------------------------------------------------------------------

    def all_gather_into_tensor(
        self, output: torch.Tensor, input_: torch.Tensor
    ) -> None:
        """`output[i*n:(i+1)*n] = input_` of rank i, matching
        torch.distributed.all_gather_into_tensor."""
        if self.disabled:
            output.copy_(input_)
            return
        # dim=0 concatenation IS the [world, n]-flattened layout this API
        # specifies, so no extra transposition is needed.
        gathered = self.all_gather(input_, dim=0)
        output.copy_(gathered.reshape(output.shape))

    def reduce_scatter_tensor(
        self, output: torch.Tensor, input_: torch.Tensor
    ) -> None:
        """Sum-reduce `input_` and scatter along dim 0 into `output`,
        matching torch.distributed.reduce_scatter_tensor."""
        if self.disabled:
            output.copy_(input_)
            return
        shard = self.reduce_scatter(input_, dim=0)
        output.copy_(shard.reshape(output.shape))

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.disabled:
            return tensor
        host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
        if self.rank == src:
            host.copy_(tensor, non_blocking=False)
        dist.broadcast(host, src=dist.get_global_rank(self.cpu_group, src),
                       group=self.cpu_group)
        if self.rank != src:
            tensor.copy_(host, non_blocking=False)
        return tensor

    # ------------------------------------------------------------------
    # teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the POSIX shm segment backing the shm/device transports.

        Without this the segment survives the process and leaks a /dev/shm
        entry per run (rank 0 owns the unlink). Called from
        GroupCoordinator.destroy().
        """
        if self.transport is None:
            return
        try:
            self.transport.close()
        except Exception as e:  # teardown must never mask the real error
            logger.warning("HTCCL: transport close failed (%s).", e)
