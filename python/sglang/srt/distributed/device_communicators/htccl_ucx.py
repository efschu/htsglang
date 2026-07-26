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

# Sub-block size for the pipelined staging copies, in KiB. See
# ``_staged_copy``: the CPU passes over a chunk are interrupted this often to
# progress the UCX worker, so the rendezvous handshake for the NEXT chunk can
# start while this one's bytes are still being touched. 0 disables the
# interleaving (the copies then run in one shot, as before L2).
#
# 256 KiB is ~7 us of memcpy at the measured in-cache bandwidth -- short
# enough that a handshake is never delayed noticeably, long enough that the
# per-sub-block Python slicing stays under a few percent of the copy.
_PROGRESS_BYTES = int(os.environ.get("SGLANG_HTCCL_UCX_PROGRESS_KIB", "256")) * 1024

# Set to 0 to fall back to the pre-L2 unpipelined all_reduce (stage the whole
# payload, one exchange, then accumulate and copy out). Kept as an escape
# hatch and as the A/B control for the pipelining measurement.
_PIPELINE = bool(int(os.environ.get("SGLANG_HTCCL_UCX_PIPELINE", "1")))


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
        progress_bytes: int = _PROGRESS_BYTES,
        pipeline: bool = _PIPELINE,
    ):
        self.cpu_group = cpu_group
        self.device = device
        self.world_size = dist.get_world_size(cpu_group)
        self.rank = dist.get_rank(cpu_group)
        self.chunk_bytes = max(int(chunk_bytes), 4096)
        self.ring_bytes = int(ring_bytes)
        self.progress_bytes = int(progress_bytes)
        self.pipeline = bool(pipeline)
        self._lock = threading.Lock()
        self._seq = 0
        self._staging_bufs: dict[str, torch.Tensor] = {}
        self._view_cache: dict[tuple, tuple] = {}
        self._closed = False

        # Everything below is per-collective bookkeeping that does not depend
        # on the payload, so it is built once. At 8 KiB -- the size a bs=1
        # decode all-reduce actually is -- rebuilding a peer list and
        # formatting its staging-buffer keys per call was ~0.4 us of a ~12 us
        # collective.
        self._peers = tuple(p for p in range(self.world_size) if p != self.rank)
        # Two slot sets, one per pipeline parity. Chunk k uses parity k & 1.
        self._ar_keys = tuple(
            (f"ar_s{par}", tuple(f"ar_r{par}_{p}" for p in self._peers))
            for par in (0, 1)
        )
        self._bar_slots = None

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

    def _slot(self, key: str, numel: int, dtype: torch.dtype) -> tuple:
        """``(view, ptr, nbytes)`` for a reusable host buffer of ``numel``.

        Memoised per (key, numel, dtype). Re-deriving the view costs a slice
        plus a dtype `view()`; re-deriving the address and length costs three
        more tensor method calls per post. All of that is invisible at 32 MiB
        and dominant at 8 KiB -- and 8 KiB is the size that matters, because
        that is what a bs=1 decode all-reduce looks like. Steady-state decode
        repeats the same shapes forever, so the cache hits essentially always.

        ``ptr`` is safe to cache alongside the view: the view keeps the
        backing tensor alive, and the entry is dropped the moment the backing
        buffer is replaced (below).

        Slots are addressed by *key*, and distinct keys are distinct backing
        tensors -- that is what makes the pipeline's two parities, and the
        send and receive sides within one parity, provably non-overlapping.
        """
        cache_key = (key, numel, dtype)
        rec = self._view_cache.get(cache_key)
        if rec is not None:
            return rec
        nbytes = numel * torch.empty((), dtype=dtype).element_size()
        buf = self._staging_bufs.get(key)
        if buf is None or buf.numel() < nbytes:
            pin = self.device.type == "cuda" and torch.cuda.is_available()
            buf = torch.empty(max(nbytes, 4096), dtype=torch.uint8, pin_memory=pin)
            self._staging_bufs[key] = buf
            # Views of the previous, smaller buffer must not be handed out
            # again -- they would silently address a buffer nothing else uses,
            # and the cached ptr would be a dangling address once the old
            # tensor is collected.
            for stale in [k for k in self._view_cache if k[0] == key]:
                del self._view_cache[stale]
        view = buf[:nbytes].view(dtype)
        rec = (view, view.data_ptr(), nbytes)
        self._view_cache[cache_key] = rec
        return rec

    def _staging(self, key: str, numel: int, dtype: torch.dtype) -> torch.Tensor:
        """The view half of :meth:`_slot`."""
        return self._slot(key, numel, dtype)[0]

    # ------------------------------------------------------------------
    # progress-interleaved host work
    #
    # A staged copy or accumulate over a 4 MiB chunk is a few hundred
    # microseconds of CPU. If the UCX worker is not progressed during it, the
    # rendezvous handshake for the NEXT chunk cannot even start -- and then
    # the pipeline overlaps nothing, because the wire sits idle for exactly
    # as long as the CPU is busy. Progressing every ``progress_bytes`` keeps
    # the handshake latency at sub-block granularity.
    # ------------------------------------------------------------------

    def _staged_copy(self, dst: torch.Tensor, src: torch.Tensor, prog) -> None:
        if prog is None:
            dst.copy_(src)
            return
        n = dst.numel()
        step = max(self.progress_bytes // dst.element_size(), 1)
        if n <= step:
            dst.copy_(src)
            prog()
            return
        off = 0
        while off < n:
            end = off + step
            if end > n:
                end = n
            dst[off:end].copy_(src[off:end])
            prog()
            off = end

    def _staged_add(self, dst: torch.Tensor, src: torch.Tensor, prog) -> None:
        if prog is None:
            dst.add_(src)
            return
        n = dst.numel()
        step = max(self.progress_bytes // dst.element_size(), 1)
        if n <= step:
            dst.add_(src)
            prog()
            return
        off = 0
        while off < n:
            end = off + step
            if end > n:
                end = n
            dst[off:end].add_(src[off:end])
            prog()
            off = end

    # ------------------------------------------------------------------
    # wire primitives
    # ------------------------------------------------------------------

    def _post_send_bytes(
        self, peer: int, ptr: int, total: int, seq, phase, chunk0=0
    ) -> list:
        """Post ``total`` bytes at ``ptr`` to ``peer``, split into chunks.

        The single-chunk case is spelled out separately because it is what
        every small collective and every step of the pipeline below hits: the
        loop's `while`, `min` and index bookkeeping are pure overhead there,
        and at 8 KiB the whole collective is only ~12 us.
        """
        if total <= self.chunk_bytes:
            return [
                self.worker.post_send(
                    peer, ptr, total, _tag(seq, phase, self.rank, chunk0)
                )
            ]
        reqs = []
        off, idx = 0, chunk0
        while off < total:
            n = min(self.chunk_bytes, total - off)
            reqs.append(
                self.worker.post_send(peer, ptr + off, n, _tag(seq, phase, self.rank, idx))
            )
            off += n
            idx += 1
        return reqs

    def _post_recv_bytes(
        self, src: int, ptr: int, total: int, seq, phase, chunk0=0
    ) -> list:
        if total <= self.chunk_bytes:
            return [
                self.worker.post_recv(ptr, total, _tag(seq, phase, src, chunk0))
            ]
        reqs = []
        off, idx = 0, chunk0
        while off < total:
            n = min(self.chunk_bytes, total - off)
            reqs.append(self.worker.post_recv(ptr + off, n, _tag(seq, phase, src, idx)))
            off += n
            idx += 1
        return reqs

    def _post_send(self, peer: int, view: torch.Tensor, seq, phase, chunk0=0) -> list:
        """Post ``view`` to ``peer``, split into chunk_bytes pieces."""
        return self._post_send_bytes(
            peer,
            view.data_ptr(),
            view.numel() * view.element_size(),
            seq,
            phase,
            chunk0,
        )

    def _post_recv(self, src: int, view: torch.Tensor, seq, phase, chunk0=0) -> list:
        return self._post_recv_bytes(
            src,
            view.data_ptr(),
            view.numel() * view.element_size(),
            seq,
            phase,
            chunk0,
        )

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
        """One-byte all-to-all exchange. Also warms up every endpoint.

        Nothing here depends on the payload, so the slots and their addresses
        are resolved once and the per-call path is two posts and a wait. The
        token byte is zeroed at construction rather than per call: its VALUE
        is never read by anyone -- the barrier is the arrival of the message,
        not its content -- so filling it every time was 0.55 us of a 12 us
        collective spent writing a byte no one looks at. It is still zeroed
        once so we never put uninitialised host memory on the wire.
        """
        with self._lock:
            slots = self._bar_slots
            if slots is None:
                send_view, send_ptr, _ = self._slot("bar_s", 1, torch.uint8)
                send_view.zero_()
                slots = (
                    send_ptr,
                    tuple(
                        (p, self._slot(f"bar_r{p}", 1, torch.uint8)[1])
                        for p in self._peers
                    ),
                )
                self._bar_slots = slots
            send_ptr, recvs = slots
            seq = self._next_seq()
            reqs = []
            for peer, rptr in recvs:
                reqs += self._post_recv_bytes(peer, rptr, 1, seq, 0)
            for peer, _ in recvs:
                reqs += self._post_send_bytes(peer, send_ptr, 1, seq, 0)
            self.worker.wait(reqs)

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

    def _pipelined_all_reduce(
        self, inp: torch.Tensor, out: torch.Tensor, n: int, dtype, seq: int
    ) -> None:
        """Chunked flat all_reduce that overlaps the host passes with the wire.

        The problem this exists for
        ---------------------------
        The unpipelined version makes four passes over the whole payload --
        stage in, wire, accumulate, stage out -- and none of the three CPU
        passes overlap the one that is on the wire. Measured on this link,
        that is the entire 32 MiB regression: the four passes over 32 MiB cost
        ~9.4 ms single-threaded against ~13 ms of wire time, and 24.3 ms was
        observed for a transfer whose wire budget is 13 ms. Worse, at 32 MiB
        the buffers no longer fit in L3, so each pass runs at ~13 GiB/s
        instead of the ~34 GiB/s a 4 MiB buffer sustains.

        The schedule
        ------------
        Chunk k occupies slot parity ``k & 1``. Per iteration::

            stage-in  k+1   <- overlaps chunk k on the wire
            wait      k
            post      k+1   <- next chunk hits the wire BEFORE this one's
                               accumulate, not after
            finish    k     <- accumulate + copy out, overlaps chunk k+1

        so both the CPU passes that follow the wire (accumulate, stage out)
        and the one that precedes it (stage in) sit underneath a transfer.
        Chunking also drags each pass back into cache, which is worth as much
        again as the overlap.

        Buffer ownership -- read this before changing anything
        -----------------------------------------------------
        Two slot sets, indexed by parity. Nothing is shared between them and
        nothing is handed to a caller. Slot parity p is, strictly in this
        program order on this one thread:

            written by stage-in(k)  ->  read by the UCX send of chunk k
            ->  waited on           ->  read+written by finish(k)
            ->  reused by stage-in(k+2)

        The rotation is what makes that safe, and it is safe only because
        ``k+1`` never has the same parity as ``k``: stage-in(k+1) cannot touch
        the slot chunk k is still flying out of, and stage-in(k+2) is issued
        at the top of the next iteration, i.e. strictly after finish(k) has
        finished reading parity ``k & 1``. This is not a comment-only
        invariant -- ``_slot`` gives each parity a distinct backing tensor
        keyed by name, and the pipelining tests exercise chunk boundaries,
        ragged tails and multi-chunk accumulation order against a reference.
        """
        # `dtype.itemsize`, not `torch.empty((), dtype=dtype).element_size()`:
        # the latter allocates a tensor on every collective, which is real
        # money at 8 KiB and invisible at 32 MiB. Same reason for the
        # single-chunk fast path below.
        esz = dtype.itemsize
        chunk = max(self.chunk_bytes // esz, 1)
        peers = self._peers
        src = inp.reshape(-1)
        # `view`, not `reshape`: the result is written chunk by chunk, and a
        # reshape of a non-contiguous `out` would silently return a COPY --
        # every chunk would land in a temporary and the caller would get an
        # untouched tensor back. `_get_out_buf` always returns an
        # `empty_like` of an already-contiguous input, so this never fires;
        # if it ever does, it fails loudly instead of returning garbage.
        dst = out.view(-1)

        if n <= chunk:
            # One chunk: there is no next transfer to overlap with, so the
            # pipeline would be pure overhead. This branch is what a bs=1
            # decode all-reduce takes -- ~8 KiB, ~33 us end to end -- and at
            # that size the closures, the staging dict and the extra call
            # layers of the general path cost ~7 us, a fifth of the whole
            # collective. Deliberately flat, deliberately duplicated.
            skey, rkeys = self._ar_keys[0]
            send_view, send_ptr, send_nbytes = self._slot(skey, n, dtype)
            send_view.copy_(src)  # (+ upcast)
            reqs = []
            recv_views = []
            for i, peer in enumerate(peers):
                rv, rptr, rnbytes = self._slot(rkeys[i], n, dtype)
                recv_views.append(rv)
                reqs.append(
                    self.worker.post_recv(rptr, rnbytes, _tag(seq, 0, peer, 0))
                )
            for peer in peers:
                reqs.append(
                    self.worker.post_send(
                        peer, send_ptr, send_nbytes, _tag(seq, 0, self.rank, 0)
                    )
                )
            self.worker.wait(reqs)
            for rv in recv_views:
                send_view.add_(rv)
            dst.copy_(send_view)  # (+ downcast)
            return

        n_chunks = (n + chunk - 1) // chunk
        if n_chunks > 0xFFFFFF:
            raise UcxError(
                f"htccl ucx: all_reduce of {n * esz} bytes needs {n_chunks} "
                f"chunks, more than the 2^24 the tag layout encodes. Raise "
                f"SGLANG_HTCCL_UCX_CHUNK_MIB."
            )
        prog = self.worker.progress if self.progress_bytes > 0 else None
        staged: dict = {}

        def stage(k: int) -> None:
            par = k & 1
            off = k * chunk
            count = n - off if off + chunk > n else chunk
            skey, rkeys = self._ar_keys[par]
            send_view, send_ptr, send_nbytes = self._slot(skey, count, dtype)
            self._staged_copy(send_view, src[off : off + count], prog)  # (+ upcast)
            recvs = [self._slot(rkeys[i], count, dtype) for i in range(len(peers))]
            staged[par] = (off, count, send_view, send_ptr, send_nbytes, recvs)

        def post(k: int) -> list:
            _, _, _, send_ptr, send_nbytes, recvs = staged[k & 1]
            reqs = []
            # Receives first, as everywhere else in this file: posting sends
            # first can wedge two peers that post in the same order.
            for i, peer in enumerate(peers):
                reqs += self._post_recv_bytes(
                    peer, recvs[i][1], recvs[i][2], seq, 0, k
                )
            for peer in peers:
                reqs += self._post_send_bytes(peer, send_ptr, send_nbytes, seq, 0, k)
            return reqs

        stage(0)
        reqs = post(0)
        for k in range(n_chunks):
            nxt = k + 1
            has_next = nxt < n_chunks
            if has_next:
                stage(nxt)
            self.worker.wait(reqs)
            if has_next:
                reqs = post(nxt)
            off, count, send_view, _, _, recvs = staged[k & 1]
            for rec in recvs:
                self._staged_add(send_view, rec[0], prog)
            self._staged_copy(dst[off : off + count], send_view, prog)  # (+ downcast)

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

            # Fresh output, never a staging buffer: all_reduce is documented
            # and dispatched as out-of-place.
            out = comm._get_out_buf(inp)

            if self.pipeline and not use_ring:
                self._pipelined_all_reduce(inp, out, n, reduce_dtype, seq)
                return out

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
        # Holds raw addresses into the buffers just dropped.
        self._bar_slots = None
        try:
            self.worker.close()
        except Exception as e:  # teardown must never mask the real error
            logger.warning("HTCCL: ucx worker close failed (%s).", e)
