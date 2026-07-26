# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""HTCCL UCX transport -- RDMA data plane for cross-node TP groups.

Same collective semantics as the gloo data plane, RDMA instead of TCP. This is
the inter-rig leg of the Nordstern ladder (L1): the intra-rig transports (shm,
device) move bytes over PCIe inside one box; this one moves them between boxes
over RoCE.

Data path, unchanged from the rest of HTCCL::

    GPU --(D2H)--> pinned host --UCX/RDMA--> pinned host --(H2D)--> GPU

There is deliberately no GPUDirect: this hardware has no P2P between the NIC
and the GPUs (see the interconnect survey -- everything is PHB, no NVLink), so
staging through the host is not a simplification, it is the only path that
exists. That also keeps the transport vendor-neutral, which is the entire
premise of HTCCL: the device only ever memcpys to and from its own host
buffer, and both CUDA and ROCm implement that identically.

Algorithms
----------
Collectives are latency-shaped, not bandwidth-shaped, because the workload
that matters is bs=1 decode: a TP all-reduce there is a few KiB, and the link
is ~1.5 us / ~26.8 Gbit/s. So the default is a *single-step* full exchange --
every rank posts all its receives and all its sends at once and progresses
them together, costing one round trip regardless of world size. Only above
``SGLANG_HTCCL_UCX_RING_MIB`` does all_reduce switch to a ring
(reduce-scatter + all-gather), where the O(N) bandwidth of the flat exchange
would start to cost more than the extra 2(N-1) steps.

Version parity
--------------
UCX peers must run the *same* release. A 1.18.1 <-> 1.16.0 pair does not
degrade, it fails at endpoint creation with ``invalid bandwidth 0.00``, which
tells the operator nothing. The rendezvous therefore compares versions across
the group *before* creating any endpoint and refuses with an actionable
message. See ``_check_version_parity``.
"""

import logging
import os
import threading

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.distributed.device_communicators.htccl_ucx_bindings import (
    UcpLibrary,
    UcpWorker,
    UcxError,
    UcxVersionMismatch,
)

logger = logging.getLogger(__name__)

# Largest single UCX transfer. Mirrors SGLANG_HTCCL_CHUNK_MIB on the gloo
# plane and the device transport's slot sizing: it bounds how much of one
# collective is in flight as a single registered transfer. Chunks of one step
# are posted together and progressed together, so this costs no extra round
# trips -- it only caps per-request footprint.
_CHUNK_BYTES = int(os.environ.get("SGLANG_HTCCL_UCX_CHUNK_MIB", "4")) * 1024 * 1024

# all_reduce switches from the one-step flat exchange to the ring above this.
_RING_BYTES = int(os.environ.get("SGLANG_HTCCL_UCX_RING_MIB", "1")) * 1024 * 1024

# Matches SGLANG_HTCCL_FP32_REDUCE on the gloo plane: half/bfloat16 are summed
# in fp32 so the accumulated error matches what NCCL would produce.
_FP32_REDUCE = bool(int(os.environ.get("SGLANG_HTCCL_FP32_REDUCE", "1")))

_TIMEOUT_S = float(os.environ.get("SGLANG_HTCCL_UCX_TIMEOUT_S", "300"))


def _tag(seq: int, phase: int, src: int, chunk: int) -> int:
    """Pack a collective step into a 64-bit UCX tag.

    ``(seq, phase, src, chunk)`` uniquely identifies one transfer: ``seq``
    counts collectives, ``phase`` the step within one, ``src`` the sender, and
    ``chunk`` the piece of an oversized payload. Receives match exactly (full
    tag mask), so two concurrently posted transfers can never be crossed.
    """
    return (
        ((seq & 0xFFFFFF) << 40)
        | ((phase & 0xFF) << 32)
        | ((src & 0xFF) << 24)
        | (chunk & 0xFFFFFF)
    )


class HTCCLUcxTransport:
    """Persistent-endpoint UCX collectives over the group's CPU process group.

    Rendezvous rides on the gloo ``cpu_group`` that HTCCL already builds -- the
    same channel the shm transport uses to publish its segment name. No side
    channel, no extra port to open, and it works before any UCX endpoint
    exists (which is exactly when the version check has to run).
    """

    # No size condition, unlike the shm transport's `nbytes <= slot_bytes`.
    # That is deliberate. The communicator dispatches per rank, so a
    # size-dependent answer would let ranks disagree about whether a given
    # collective goes over UCX or over gloo -- half the group waiting on a
    # tag that the other half never sends. Op identity is group-uniform by
    # construction; payload size is only uniform for the collectives that
    # require identical shapes anyway. Keeping the routing decision purely
    # op-based makes the divergence structurally impossible instead of
    # merely unlikely.
    HTCCL_OPS = frozenset({"all_reduce", "all_gather", "broadcast", "reduce_scatter"})

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        chunk_bytes: int = _CHUNK_BYTES,
        ring_bytes: int = _RING_BYTES,
    ):
        self.cpu_group = cpu_group
        self.device = device
        self.world_size = dist.get_world_size(cpu_group)
        self.rank = dist.get_rank(cpu_group)
        self.chunk_bytes = max(int(chunk_bytes), 4096)
        self.ring_bytes = int(ring_bytes)
        self._lock = threading.Lock()
        self._seq = 0
        self._staging_bufs: dict[str, torch.Tensor] = {}
        self._view_cache: dict[tuple, torch.Tensor] = {}
        self._closed = False

        if self.world_size > 256:
            raise UcxError(
                f"htccl ucx: world size {self.world_size} exceeds the 256 the "
                f"tag layout encodes."
            )

        self.lib = UcpLibrary.instance()
        self.worker = UcpWorker(self.lib, timeout_s=_TIMEOUT_S)

        # One collective carries both halves of the rendezvous: the version
        # (checked first) and the worker address (used only if the check
        # passes). Gathering them together keeps the group in lock step -- a
        # rank must not create endpoints while a peer is still aborting.
        info = {
            "rank": self.rank,
            "version": self.lib.version(),
            "version_string": self.lib.version_string(),
            "lib_path": self.lib.path,
            "address": self.worker.address(),
        }
        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, info, group=cpu_group)
        self._check_version_parity(gathered)

        for peer in range(self.world_size):
            if peer == self.rank:
                continue
            self.worker.connect(peer, gathered[peer]["address"])

        # UCX wires an endpoint up lazily, on first use. Pay that here so the
        # first decode step does not: a wireup handshake is orders of
        # magnitude more expensive than the ~1.5 us the link is capable of,
        # and it would otherwise land inside the first forward pass.
        self.barrier()
        logger.info(
            "HTCCL: ucx transport ready (rank %d/%d, UCX %s, chunk %d MiB, "
            "ring threshold %d MiB).",
            self.rank, self.world_size, info["version_string"],
            self.chunk_bytes // (1024 * 1024), self.ring_bytes // (1024 * 1024),
        )

    # ------------------------------------------------------------------
    # rendezvous
    # ------------------------------------------------------------------

    def _check_version_parity(self, gathered: list) -> None:
        """Refuse a group whose ranks run different UCX releases.

        UCX encodes transport capabilities in the UCP worker address, and that
        encoding changed within the 1.x line. Mixing releases does not
        negotiate down -- ``ucp_ep_create`` fails, or worse the link comes up
        and reports ``invalid bandwidth 0.00``. Neither message points at the
        cause, so the check happens here, before any endpoint exists, and says
        what to do about it.
        """
        versions = {tuple(g["version"]) for g in gathered}
        if len(versions) == 1:
            return
        lines = [
            f"  rank {g['rank']}: UCX {g['version_string']} "
            f"(loaded from {g['lib_path']})"
            for g in sorted(gathered, key=lambda g: g["rank"])
        ]
        newest = max(versions)
        raise UcxVersionMismatch(
            "HTCCL ucx: the ranks of this group run different UCX releases, "
            "which the UCP wire address format does not support. This does "
            "NOT degrade gracefully -- endpoint creation fails, typically "
            "with the unhelpful 'invalid bandwidth 0.00'.\n"
            + "\n".join(lines)
            + "\n\nFix it by giving every rank the SAME UCX release. Either "
            "install matching packages on all hosts, or point the odd ones "
            "out at a side-by-side build via\n"
            "    SGLANG_HTCCL_UCX_LIB=/opt/ucx<ver>/lib/libucp.so.0\n"
            "(this transport also pre-loads libucs/libuct from that same "
            "directory, so LD_LIBRARY_PATH is not additionally required).\n"
            f"The highest version present in the group is "
            f"{'.'.join(str(v) for v in newest)}; the lowest is the one every "
            "rank must be able to load."
        )

    # ------------------------------------------------------------------
    # transport seam
    # ------------------------------------------------------------------

    def handles(self, op: str, nbytes: int) -> bool:
        return op in self.HTCCL_OPS

    # ------------------------------------------------------------------
    # staging buffers
    #
    # INTERNAL ONLY. Nothing in this file may return one of these, or a view
    # of one, to a caller: two collectives of the same shape would alias, and
    # the second would silently overwrite a result the model still holds. That
    # exact bug has already been paid for once here -- see
    # HTCCLCommunicator._get_out_buf. Every public method below allocates its
    # result fresh and copies out of the staging buffer into it.
    # ------------------------------------------------------------------

    def _staging(self, key: str, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """A reusable host buffer view of ``numel`` elements of ``dtype``.

        Views are memoised per (key, numel, dtype). Re-deriving them costs a
        slice plus a dtype `view()` on every call, which is invisible at 32 MiB
        and dominant at 8 KiB -- and 8 KiB is the size that matters, because
        that is what a bs=1 decode all-reduce looks like. Steady-state decode
        repeats the same shapes forever, so the cache hits essentially always.
        """
        cache_key = (key, numel, dtype)
        view = self._view_cache.get(cache_key)
        if view is not None:
            return view
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        buf = self._staging_bufs.get(key)
        if buf is None or buf.numel() < nbytes:
            pin = self.device.type == "cuda" and torch.cuda.is_available()
            buf = torch.empty(max(nbytes, 4096), dtype=torch.uint8, pin_memory=pin)
            self._staging_bufs[key] = buf
            # Views of the previous, smaller buffer must not be handed out
            # again -- they would silently address a buffer nothing else uses.
            for stale in [k for k in self._view_cache if k[0] == key]:
                del self._view_cache[stale]
        view = buf[:nbytes].view(dtype)
        self._view_cache[cache_key] = view
        return view

    # ------------------------------------------------------------------
    # wire primitives
    # ------------------------------------------------------------------

    def _post_send(self, peer: int, view: torch.Tensor, seq, phase, chunk0=0) -> list:
        """Post ``view`` to ``peer``, split into chunk_bytes pieces."""
        reqs = []
        ptr = view.data_ptr()
        total = view.numel() * view.element_size()
        off, idx = 0, chunk0
        while off < total:
            n = min(self.chunk_bytes, total - off)
            reqs.append(
                self.worker.post_send(peer, ptr + off, n, _tag(seq, phase, self.rank, idx))
            )
            off += n
            idx += 1
        return reqs

    def _post_recv(self, src: int, view: torch.Tensor, seq, phase, chunk0=0) -> list:
        reqs = []
        ptr = view.data_ptr()
        total = view.numel() * view.element_size()
        off, idx = 0, chunk0
        while off < total:
            n = min(self.chunk_bytes, total - off)
            reqs.append(self.worker.post_recv(ptr + off, n, _tag(seq, phase, src, idx)))
            off += n
            idx += 1
        return reqs

    def _next_seq(self) -> int:
        """Collective counter.

        Rank-uniform because HTCCL's contract is that every rank issues the
        same collectives in the same order, and `handles()` above routes purely
        on op identity, so no rank can skip a collective the others take.
        """
        self._seq = (self._seq + 1) & 0xFFFFFF
        return self._seq

    def _exchange_all(self, send: torch.Tensor, recvs: dict, seq: int, phase: int = 0):
        """Send ``send`` to every peer and receive each peer's into ``recvs``.

        Receives are posted before sends and everything is progressed as one
        set, so the whole exchange costs one round trip rather than N-1 of
        them -- and cannot deadlock on posting order.
        """
        reqs = []
        for peer, view in recvs.items():
            reqs += self._post_recv(peer, view, seq, phase)
        for peer in recvs:
            reqs += self._post_send(peer, send, seq, phase)
        self.worker.wait(reqs)

    def barrier(self) -> None:
        """One-byte all-to-all exchange. Also warms up every endpoint."""
        with self._lock:
            seq = self._next_seq()
            send = self._staging("bar_s", 1, torch.uint8)
            send.fill_(0)
            recvs = {
                p: self._staging(f"bar_r{p}", 1, torch.uint8)
                for p in range(self.world_size)
                if p != self.rank
            }
            self._exchange_all(send, recvs, seq)

    # ------------------------------------------------------------------
    # all_reduce
    # ------------------------------------------------------------------

    def _wire_all_reduce_flat(self, host: torch.Tensor, seq: int) -> None:
        """One-step exchange + local sum. Latency-optimal for small payloads."""
        n = host.numel()
        recvs = {
            p: self._staging(f"ar_r{p}", n, host.dtype)
            for p in range(self.world_size)
            if p != self.rank
        }
        # `host` is both the send source and the accumulator, so it must not
        # be summed into until every send has completed reading it.
        self._exchange_all(host, recvs, seq)
        for view in recvs.values():
            host.add_(view)

    def _wire_all_reduce_ring(self, host: torch.Tensor, seq: int) -> None:
        """Ring reduce-scatter + all-gather. Bandwidth-optimal for large ones.

        ``host`` is padded to a multiple of world_size by the caller, so every
        block is the same length and the schedule is symmetric across ranks.
        """
        N = self.world_size
        block = host.numel() // N
        blocks = [host[i * block : (i + 1) * block] for i in range(N)]
        right = (self.rank + 1) % N
        left = (self.rank - 1) % N
        tmp = self._staging("ar_ring", block, host.dtype)

        for step in range(N - 1):
            send_idx = (self.rank - step) % N
            recv_idx = (self.rank - step - 1) % N
            reqs = self._post_recv(left, tmp, seq, step)
            reqs += self._post_send(right, blocks[send_idx], seq, step)
            self.worker.wait(reqs)
            blocks[recv_idx].add_(tmp)

        for step in range(N - 1):
            send_idx = (self.rank - step + 1) % N
            recv_idx = (self.rank - step) % N
            phase = (N - 1) + step
            reqs = self._post_recv(left, blocks[recv_idx], seq, phase)
            reqs += self._post_send(right, blocks[send_idx], seq, phase)
            self.worker.wait(reqs)

    def htccl_all_reduce(self, comm, inp: torch.Tensor) -> torch.Tensor:
        with self._lock:
            seq = self._next_seq()
            reduce_dtype = (
                torch.float32
                if _FP32_REDUCE and inp.dtype in (torch.float16, torch.bfloat16)
                else inp.dtype
            )
            n = inp.numel()
            nbytes = n * inp.element_size()
            use_ring = nbytes >= self.ring_bytes and self.world_size > 2
            # The ring needs equal blocks; pad up and zero the tail so every
            # rank contributes zeroes there and the sum is unaffected.
            padded = (
                ((n + self.world_size - 1) // self.world_size) * self.world_size
                if use_ring
                else n
            )
            host = self._staging("ar", padded, reduce_dtype)
            host[:n].copy_(inp.reshape(-1))  # D2H (+ upcast)
            if padded > n:
                host[n:].zero_()

            if use_ring:
                self._wire_all_reduce_ring(host, seq)
            else:
                self._wire_all_reduce_flat(host, seq)

            # Fresh output, never a staging buffer: all_reduce is documented
            # and dispatched as out-of-place.
            out = comm._get_out_buf(inp)
            out.reshape(-1).copy_(host[:n])  # H2D (+ downcast)
            return out

    # ------------------------------------------------------------------
    # all_gather
    # ------------------------------------------------------------------

    def htccl_all_gather(self, comm, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """Concatenate every rank's ``inp`` along ``dim``.

        Shapes must be identical across ranks -- the same contract the gloo
        plane has always had. Axis handling is copied from it verbatim so the
        two transports cannot drift apart.
        """
        with self._lock:
            seq = self._next_seq()
            if dim < 0:
                dim += inp.dim()
            inp = inp.contiguous()
            input_size = inp.size()
            n = inp.numel()

            send = self._staging("ag_s", n, inp.dtype)
            send.copy_(inp.reshape(-1))
            recvs = {
                p: self._staging(f"ag_r{p}", n, inp.dtype)
                for p in range(self.world_size)
                if p != self.rank
            }
            self._exchange_all(send, recvs, seq)

            output = torch.empty(
                (self.world_size,) + tuple(input_size),
                dtype=inp.dtype,
                device=inp.device,
            )
            for p in range(self.world_size):
                src = send if p == self.rank else recvs[p]
                output[p].reshape(-1).copy_(src)
            output = output.movedim(0, dim)
            return output.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )

    # ------------------------------------------------------------------
    # reduce_scatter
    # ------------------------------------------------------------------

    def htccl_reduce_scatter(self, comm, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """Sum across ranks, then keep this rank's slice along ``dim``.

        Composed from all_reduce exactly as the gloo plane composes it,
        including the ``movedim(dim, 0)`` orientation -- writing it the other
        way round silently scatters the wrong axis from ndim >= 3 while every
        shape check still passes. A native ring reduce-scatter would move
        1/N of the bytes; that is a separate, testable change.
        """
        if dim < 0:
            dim += inp.dim()
        reduced = self.htccl_all_reduce(comm, inp.contiguous())
        moved = reduced.movedim(dim, 0).contiguous()
        assert moved.shape[0] % self.world_size == 0
        chunk = moved.shape[0] // self.world_size
        shard = moved[self.rank * chunk : (self.rank + 1) * chunk]
        return shard.movedim(0, dim).contiguous()

    # ------------------------------------------------------------------
    # broadcast
    # ------------------------------------------------------------------

    def htccl_broadcast(self, comm, tensor: torch.Tensor, src: int) -> torch.Tensor:
        """In-place broadcast from ``src``; returns ``tensor`` itself.

        Host-staged, so this synchronises with the CPU and -- like the gloo
        plane and unlike the device transport -- must not be called from
        inside a CUDA-graph capture.
        """
        with self._lock:
            seq = self._next_seq()
            n = tensor.numel()
            host = self._staging("bc", n, tensor.dtype)
            if self.rank == src:
                host.copy_(tensor.reshape(-1))
                reqs = []
                for peer in range(self.world_size):
                    if peer != self.rank:
                        reqs += self._post_send(peer, host, seq, 0)
                self.worker.wait(reqs)
            else:
                self.worker.wait(self._post_recv(src, host, seq, 0))
                tensor.reshape(-1).copy_(host)
            return tensor

    # ------------------------------------------------------------------
    # teardown
    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        # Agree on the end of the stream before destroying endpoints. Without
        # this the first rank to exit tears its endpoints down while peers are
        # still progressing, and UCX reports 'Connection reset by remote peer'
        # on an otherwise clean run. Best-effort and short-fused: a rank that
        # is shutting down because a PEER already died must not block here.
        try:
            self.worker.timeout_s = 5.0
            self.barrier()
        except Exception:
            pass
        self._staging_bufs.clear()
        self._view_cache.clear()
        try:
            self.worker.close()
        except Exception as e:  # teardown must never mask the real error
            logger.warning("HTCCL: ucx worker close failed (%s).", e)
