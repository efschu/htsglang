# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""barlink UCX transport -- RDMA data plane for cross-node TP groups.

Same collective semantics as the gloo data plane, RDMA instead of TCP. This is
the inter-rig leg of the Nordstern ladder (L1): the intra-rig transports (shm,
device) move bytes over PCIe inside one box; this one moves them between boxes
over RoCE.

Data path, unchanged from the rest of barlink::

    GPU --(D2H)--> pinned host --UCX/RDMA--> pinned host --(H2D)--> GPU

There is deliberately no GPUDirect: this hardware has no P2P between the NIC
and the GPUs (see the interconnect survey -- everything is PHB, no NVLink), so
staging through the host is not a simplification, it is the only path that
exists. That also keeps the transport vendor-neutral, which is the entire
premise of barlink: the device only ever memcpys to and from its own host
buffer, and both CUDA and ROCm implement that identically.

Algorithms
----------
Small collectives are latency-shaped: a bs=1 decode all-reduce is a few KiB
and the link is ~1.5 us / ~26.8 Gbit/s, so below ``SGLANG_BARLINK_UCX_RING_KIB``
all_reduce is a *single-step* full exchange -- every rank posts all its
receives and all its sends at once and progresses them together, one round
trip regardless of world size.

Above that threshold the flat exchange's O(N) bandwidth dominates and
all_reduce switches to a ring (reduce-scatter + all-gather), which moves half
the bytes at world 4 for 2(N-1) steps. The threshold is 24 KiB, measured, not
assumed -- see ``_RING_BYTES``. A speculative verify all-reduce is well past
it, so on a cross-rig group the ring is the ordinary decode path, not the
large-payload exception.

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

from sglang.srt.distributed.device_communicators.barlink_ucx_bindings import (
    UcpLibrary,
    UcpWorker,
    UcxError,
    UcxVersionMismatch,
    wait_multi,
)

logger = logging.getLogger(__name__)

# Largest single UCX transfer. Mirrors SGLANG_BARLINK_CHUNK_MIB on the gloo
# plane and the device transport's slot sizing: it bounds how much of one
# collective is in flight as a single registered transfer. Chunks of one step
# are posted together and progressed together, so this costs no extra round
# trips -- it only caps per-request footprint.
_CHUNK_BYTES = int(os.environ.get("SGLANG_BARLINK_UCX_CHUNK_MIB", "4")) * 1024 * 1024

# all_reduce switches from the one-step flat exchange to the ring above this.
#
# 24 KiB, not the 1 MiB this was set to originally. The 1 MiB came from the
# reasoning that the flat exchange costs one round trip and the ring costs
# 2(N-1), so the ring only pays once bandwidth beats latency -- correct in
# shape, wrong in scale. The flat exchange puts (W-1) payloads on the wire in
# each direction per collective; the ring puts 2(W-1)/W, i.e. half the bytes
# at W=4. Measured cross-rig at world 4 (task #244, all_reduce, us, median of
# 100 after 20 warmup, ranks 0-2 on rig 1 and rank 3 on rig 2):
#
#     KiB     10    16    20    24    32    40    48    64    80   128
#     flat  57.9  75.6 101.1 117.5 125.8 207.9 225.3 319.0 385.9 646.3
#     ring  93.3  99.6 106.6 112.3 120.9 138.8 145.2 168.5 195.0 270.4
#
# The two cross at ~22 KiB and the ring wins by 2x from 64 KiB up. A 4-token
# MTP verify all-reduce on a 5120-hidden model is 80 KiB in the default fp32
# staging -- the far side of the crossover, and the single most frequent
# collective in a cross-rig decode.
#
# _KIB is the knob; _MIB stays accepted so an existing launch line keeps
# working, and wins when both are set.
_RING_BYTES = int(os.environ.get("SGLANG_BARLINK_UCX_RING_KIB", "24")) * 1024
if os.environ.get("SGLANG_BARLINK_UCX_RING_MIB"):
    _RING_BYTES = int(os.environ["SGLANG_BARLINK_UCX_RING_MIB"]) * 1024 * 1024

# Same switch for all_gather. 32 KiB, measured, not assumed -- and it is worth
# saying why the number exists at all, because the byte count predicts that it
# should not: an all_gather has to deliver (W-1)*n bytes into every rank and
# the flat exchange already moves exactly that, once, in one round trip. The
# ring moves the same bytes in W-1 lock-step round trips, so unlike the
# all_reduce ring (which halves the bytes at W=4) it cannot win on volume.
#
# It wins anyway, because the cost here is not the wire but the one
# single-threaded UCX worker per rank. The flat exchange hands it 2(W-1)
# simultaneous requests; the ring hands it two at a time. Measured cross-rig
# at world 4 (task #263, us, median of 100 after 20 warmup, ranks 0-2 on rig 1
# and rank 3 on rig 2, with the host passes of task #263 already chunked):
#
#     KiB      8     20     40     80    128     256
#     flat  54.0   97.2  195.8  397.6  642.3  1321.9
#     ring  73.3  110.2  180.2  298.6  452.0   966.9
#
# They cross at ~32 KiB and the ring holds ~25-30 % from 80 KiB up. A bs=1
# decode gather (20 KiB) stays flat; a 4-token verify gather (80 KiB) rings.
# 0 disables the ring entirely and is the A/B control.
_AG_RING_BYTES = int(os.environ.get("SGLANG_BARLINK_UCX_AG_RING_KIB", "32")) * 1024

# Matches SGLANG_BARLINK_FP32_REDUCE on the gloo plane: half/bfloat16 are summed
# in fp32 so the accumulated error matches what NCCL would produce.
_FP32_REDUCE = bool(int(os.environ.get("SGLANG_BARLINK_FP32_REDUCE", "1")))

_TIMEOUT_S = float(os.environ.get("SGLANG_BARLINK_UCX_TIMEOUT_S", "300"))

# How many independent UCX contexts/workers this rank builds for the
# collective plane. 1 is the shape every measurement up to task #263 was taken
# on: one context, one worker, one endpoint per peer, everything on it.
#
# 2 exists because the single worker is the thing the ring was already winning
# against. Task #263 established that an all_gather ring beats the flat
# exchange while moving the SAME bytes, purely because it hands the worker two
# requests at a time instead of 2(W-1); and task #244 established that the
# same world-4 group with no wire at all is SLOWER than across the RDMA link.
# Both point at one rank's serialised worker, not at the link.
#
# With 2 workers the two structures that were serialised on one worker are
# split:
#
#   * the flat exchange assigns peer pair (r, p) to worker (r + p) % ways --
#     symmetric in r and p, so both ends of a pair pick the same worker
#     without exchanging anything;
#   * the ring runs BIDIRECTIONALLY, half the payload clockwise on worker 0
#     and half counter-clockwise on worker 1, so a step moves half the bytes
#     per hop in the same number of steps.
#
# RANK-UNIFORM, like every other SGLANG_BARLINK* knob and more strictly than
# most: a rank that disagrees about the worker count posts one half of a step
# where its peer is not listening, and the collective hangs rather than
# returning a wrong answer. wait_multi's timeout message names this.
#
# The default is 1 because 2 did not pay on this hardware -- see the
# measurement in FEATURES_VS_UPSTREAM section 21 (task #266). It is kept as a
# knob because the mechanism is real and the ceiling it runs into is this
# rig's (one core per rank progressing both workers), not the design's.
_WORKERS = max(int(os.environ.get("SGLANG_BARLINK_UCX_WORKERS", "1")), 1)

# Above this world size the bidirectional ring falls back to the
# unidirectional one: the direction is encoded into the 8-bit tag phase as
# 2*step + direction, which the all-gather half of an all_reduce ring runs up
# to 4W-5. The unidirectional ring's own phase ceiling is 2W-3.
_BIDIR_MAX_WORLD = 64

# Whether the second worker also splits the RING, half the payload each way
# round. Off, because it measured negative and the two halves of the knob turn
# out to be independent mechanisms with opposite signs (task #266, cross-rig
# world 4, median of 100 after 20 warmup, interleaved A/B):
#
#                  all_reduce                     all_gather
#     KiB      20      80     128     256      20      80     128     256
#     1 wkr  96.6   203.1   277.7   573.3    99.0   304.4   463.4  1007.6
#     2 wkr  89.3   236.8   301.4   599.1    92.5   327.6   487.7  1047.8
#              -8%    +17%     +9%     +5%     -7%     +8%     +5%     +4%
#
# The gain is the FLAT exchange: (W-1) simultaneous requests per direction on
# one worker is the cost task #263 already caught the all_gather ring beating,
# and splitting the peers over two workers takes another 7 % off it. The loss
# is the ring: a ring step is two requests in lock step, so there is no
# concurrency for a second worker to expose -- halving the bytes per hop buys
# nothing on a link where task #244 already showed the bytes are not the cost,
# and every spin pass now progresses two engines instead of one.
#
# So the two are shipped separately: ..._WORKERS=2 splits the flat exchange
# (the bs=1 decode collectives), ..._RING_BIDIR=1 additionally splits the ring
# (the 4-token verify collectives), and only the first one pays here.
_RING_BIDIR = bool(int(os.environ.get("SGLANG_BARLINK_UCX_RING_BIDIR", "0")))


def _collective_net_device() -> str | None:
    """The link this instance's UCX collective plane is pinned to.

    ``SGLANG_COLLECTIVE_NET_SMALL`` (``--collective-net-small``) names it.
    Unset -> ``None``, and the context is built from the unmodified
    environment config, i.e. whatever process-wide ``UCX_NET_DEVICES`` says,
    exactly as before this knob existed.

    Read at construction rather than at import: the flag is exported into the
    environment during server-args resolution, which happens after this module
    may already have been imported.

    SCOPE -- this pins the WHOLE collective plane, not just the small
    messages. The small/large split (classes (a) and (b) in the runbook) lives
    inside one UCX context with one endpoint per peer, so there is no seam
    here to route a 512 KiB prefill chunk differently from an 8 KiB decode
    all-reduce. ``--collective-net-bulk`` therefore reaches the bulk transfers
    that DO have their own transport (PD-KV / HiCache, via
    ``--disaggregation-ib-device``) and not this one. See
    ``docs/rig-runbook.md``.
    """
    dev = os.environ.get("SGLANG_COLLECTIVE_NET_SMALL", "").strip()
    return dev or None


# Sub-block size for the pipelined staging copies, in KiB. See
# ``_staged_copy``: the CPU passes over a chunk are interrupted this often to
# progress the UCX worker, so the rendezvous handshake for the NEXT chunk can
# start while this one's bytes are still being touched. 0 disables the
# interleaving (the copies then run in one shot, as before L2).
#
# 256 KiB is ~7 us of memcpy at the measured in-cache bandwidth -- short
# enough that a handshake is never delayed noticeably, long enough that the
# per-sub-block Python slicing stays under a few percent of the copy.
_PROGRESS_BYTES = int(os.environ.get("SGLANG_BARLINK_UCX_PROGRESS_KIB", "256")) * 1024

# Set to 0 to fall back to the pre-L2 unpipelined all_reduce (stage the whole
# payload, one exchange, then accumulate and copy out). Kept as an escape
# hatch and as the A/B control for the pipelining measurement.
_PIPELINE = bool(int(os.environ.get("SGLANG_BARLINK_UCX_PIPELINE", "1")))


# Largest host-side pass, in elements, that torch keeps on the calling thread.
# `at::internal::GRAIN_SIZE`; above it a CPU->CPU `copy_`/`add_` is dispatched
# through `at::parallel_for`, which opens an OpenMP region for a pass that
# takes single-digit microseconds.
#
# That is a cliff, not a tax, and it is why the 128 -> 256 KiB step cost
# milliseconds (task #263). Co-located TP ranks leave the barrier together and
# hit their host passes at the same instant, so N ranks x N threads land on
# one core count and the region's join waits on a descheduled thread. Measured
# on the rig-1 host, 3 concurrent processes, 320 KiB fp32:
#
#     whole buffer     median  7.91 us   max 6000.80 us
#     grain-chunked    median 22.41 us   max   46.23 us
#
# Single-threaded is 2.8x slower in the median and ~130x better in the tail,
# and under the real harness the tail IS the median. Set to 0 to restore the
# unchunked passes as an A/B control.
#
# Only the passes that are not already chunked go through this: the pipelined
# multi-chunk path splits by SGLANG_BARLINK_UCX_PROGRESS_KIB and its large-payload
# throughput does want the parallel copy, so `_staged_copy`/`_staged_add` are
# deliberately left alone.
_GRAIN_ELEMS = int(os.environ.get("SGLANG_BARLINK_UCX_GRAIN_ELEMS", "32768"))


def _grain_step(dst: torch.Tensor, src: torch.Tensor) -> int:
    """Elements per sub-pass for ``dst``, or 0 meaning "do it in one".

    Device copies are memcpys, not tensor iterators, so they are never split:
    chunking a D2H or H2D would only add per-call overhead.
    """
    if (
        not _GRAIN_ELEMS
        or dst.numel() <= _GRAIN_ELEMS
        or dst.device.type != "cpu"
        or src.device.type != "cpu"
        or not dst.is_contiguous()
        or not src.is_contiguous()
    ):
        return 0
    return _GRAIN_ELEMS


def _host_copy_(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``dst.copy_(src)``, kept single-threaded when both sides are host.

    Byte-neutral by construction: a copy is elementwise, so splitting it into
    contiguous sub-ranges cannot change a single output byte.
    """
    step = _grain_step(dst, src)
    if not step:
        dst.copy_(src)
        return
    d, s, n = dst.view(-1), src.view(-1), dst.numel()
    off = 0
    while off < n:
        end = off + step if off + step < n else n
        d[off:end].copy_(s[off:end])
        off = end


def _host_add_(dst: torch.Tensor, src: torch.Tensor) -> None:
    """``dst.add_(src)``, same split. Also byte-neutral: the accumulation is
    elementwise, so no sub-range sees a different addend or a different order.
    """
    step = _grain_step(dst, src)
    if not step:
        dst.add_(src)
        return
    d, s, n = dst.view(-1), src.view(-1), dst.numel()
    off = 0
    while off < n:
        end = off + step if off + step < n else n
        d[off:end].add_(s[off:end])
        off = end


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


class _UcxAsyncHandle:
    """An in-flight collective issued by one of the ``*_async`` methods.

    Owns its staging slots (``send``, ``recvs`` -- pool records) until
    ``wait_async`` releases them, and carries its own ``seq`` so handles can
    be awaited out of issue order. See the async section of the transport
    for the full ownership / progress / order contracts.
    """

    __slots__ = (
        "op",
        "seq",
        "reqs",
        "send",
        "recvs",
        "shape",
        "dtype",
        "device",
        "n",
        "dim",
        "done",
    )

    def __init__(self, op, seq, reqs, send, recvs, shape, dtype, device, n, dim):
        self.op = op
        self.seq = seq
        self.reqs = reqs
        self.send = send
        self.recvs = recvs
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.n = n
        self.dim = dim
        self.done = False


class BarlinkUcxTransport:
    """Persistent-endpoint UCX collectives over the group's CPU process group.

    Rendezvous rides on the gloo ``cpu_group`` that barlink already builds -- the
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
    BARLINK_OPS = frozenset({"all_reduce", "all_gather", "broadcast", "reduce_scatter"})

    # Class-level default so _init_worker_routing reads the same attribute
    # whether it is called from __init__ or from a test fixture that built the
    # transport with __new__ and never went through it.
    ring_bidir = False

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        chunk_bytes: int = _CHUNK_BYTES,
        ring_bytes: int = _RING_BYTES,
        progress_bytes: int = _PROGRESS_BYTES,
        pipeline: bool = _PIPELINE,
        ag_ring_bytes: int = _AG_RING_BYTES,
        net_device: str | None = None,
        workers: int = _WORKERS,
        ring_bidir: bool = _RING_BIDIR,
    ):
        self.cpu_group = cpu_group
        self.device = device
        self.world_size = dist.get_world_size(cpu_group)
        self.rank = dist.get_rank(cpu_group)
        self.chunk_bytes = max(int(chunk_bytes), 4096)
        self.ring_bytes = int(ring_bytes)
        self.ag_ring_bytes = int(ag_ring_bytes)
        self.progress_bytes = int(progress_bytes)
        self.pipeline = bool(pipeline)
        self._lock = threading.Lock()
        self._seq = 0
        self._staging_bufs: dict[str, torch.Tensor] = {}
        self._view_cache: dict[tuple, tuple] = {}
        # Whether copy-OUT of a staging slot may be enqueued non_blocking.
        # See the copy-out section below; the CPU-only tests run with this
        # False and exercise the event bookkeeping through _async_h2d_ok.
        self._use_cuda = self.device.type == "cuda" and torch.cuda.is_available()
        # Completion event of the last in-flight non_blocking device read OUT
        # of a staging slot, keyed like _staging_bufs. One persistent event
        # per key, re-recorded per read -- mirrors the event use of the gloo
        # plane (barlink.py, _stage/ev.synchronize()).
        self._h2d_events: dict[str, torch.cuda.Event] = {}
        # Free list for the async collectives' staging slots, keyed by
        # power-of-two size class: (buffer, completion event of its last
        # non_blocking read, or None). See the async section below.
        self._async_free: dict[int, list[tuple]] = {}
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
        # Same two-parity layout for all_gather. Distinct keys from the
        # all_reduce set on purpose: sharing them would be functionally safe
        # today (collectives are strictly serialised under self._lock) but
        # would make slot lifetime depend on that serialisation -- the exact
        # comment-only invariant the async follow-up must not inherit.
        self._ag_keys = tuple(
            (f"ag_s{par}", tuple(f"ag_r{par}_{p}" for p in self._peers))
            for par in (0, 1)
        )
        self._bar_slots = None

        if self.world_size > 256:
            raise UcxError(
                f"barlink ucx: world size {self.world_size} exceeds the 256 the "
                f"tag layout encodes."
            )

        self.lib = UcpLibrary.instance()
        self.net_device = _collective_net_device() if net_device is None else net_device
        self.ring_bidir = bool(ring_bidir)
        self.workers = tuple(
            UcpWorker(self.lib, timeout_s=_TIMEOUT_S, net_devices=self.net_device)
            for _ in range(max(int(workers), 1))
        )
        self._init_worker_routing()

        # One collective carries both halves of the rendezvous: the version
        # (checked first) and the worker addresses (used only if the check
        # passes). Gathering them together keeps the group in lock step -- a
        # rank must not create endpoints while a peer is still aborting.
        info = {
            "rank": self.rank,
            "version": self.lib.version(),
            "version_string": self.lib.version_string(),
            "lib_path": self.lib.path,
            # List, one per worker. `address` (singular) is kept alongside it
            # so a rank running an older tree still finds what it reads --
            # this rendezvous is the only thing two sides of a cross-rig group
            # exchange before any endpoint exists.
            "address": self.workers[0].address(),
            "addresses": [w.address() for w in self.workers],
        }
        gathered: list = [None] * self.world_size
        dist.all_gather_object(gathered, info, group=cpu_group)
        self._check_version_parity(gathered)
        self._check_worker_parity(gathered)

        for peer in range(self.world_size):
            if peer == self.rank:
                continue
            for i, w in enumerate(self.workers):
                w.connect(peer, gathered[peer]["addresses"][i])

        # UCX wires an endpoint up lazily, on first use. Pay that here so the
        # first decode step does not: a wireup handshake is orders of
        # magnitude more expensive than the ~1.5 us the link is capable of,
        # and it would otherwise land inside the first forward pass. Every
        # worker's endpoints are warmed, not just the ones the flat exchange
        # happens to route through at this world size.
        self._wireup()
        logger.info(
            "barlink: ucx transport ready (rank %d/%d, UCX %s, chunk %d MiB, "
            "ring threshold %d KiB, workers %d%s, net device %s).",
            self.rank,
            self.world_size,
            info["version_string"],
            self.chunk_bytes // (1024 * 1024),
            self.ring_bytes // 1024,
            self.ways,
            " (bidirectional ring)" if self._ring_ways == 2 else "",
            # Deliberately NOT rank-uniform, unlike every SGLANG_BARLINK* knob:
            # the device NAME is per host (rocep4s0f1 here, rocep1s0f1 on the
            # peer). Only the link the names resolve to has to be the same,
            # and that is a wiring fact UCX itself reports at wireup.
            self.net_device or "<from environment>",
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
            "barlink ucx: the ranks of this group run different UCX releases, "
            "which the UCP wire address format does not support. This does "
            "NOT degrade gracefully -- endpoint creation fails, typically "
            "with the unhelpful 'invalid bandwidth 0.00'.\n"
            + "\n".join(lines)
            + "\n\nFix it by giving every rank the SAME UCX release. Either "
            "install matching packages on all hosts, or point the odd ones "
            "out at a side-by-side build via\n"
            "    SGLANG_BARLINK_UCX_LIB=/opt/ucx<ver>/lib/libucp.so.0\n"
            "(this transport also pre-loads libucs/libuct from that same "
            "directory, so LD_LIBRARY_PATH is not additionally required).\n"
            f"The highest version present in the group is "
            f"{'.'.join(str(v) for v in newest)}; the lowest is the one every "
            "rank must be able to load."
        )

    def _check_worker_parity(self, gathered: list) -> None:
        """Refuse a group whose ranks built different numbers of workers.

        ``SGLANG_BARLINK_UCX_WORKERS`` decides which worker carries which half
        of a collective. A rank that disagrees does not compute a wrong
        answer, it posts where nobody is listening and the group hangs at the
        first collective -- the failure mode this whole file is written to
        avoid. Checked at rendezvous, before any endpoint exists, like the
        version check.
        """
        counts = {len(g["addresses"]) for g in gathered}
        if len(counts) == 1:
            return
        lines = [
            f"  rank {g['rank']}: {len(g['addresses'])} worker(s)"
            for g in sorted(gathered, key=lambda g: g["rank"])
        ]
        raise UcxError(
            "barlink ucx: the ranks of this group built different numbers of "
            "UCX workers. The worker a collective's half rides is derived "
            "from that count on each side independently, so a mismatch hangs "
            "the group instead of failing.\n"
            + "\n".join(lines)
            + "\n\nSet SGLANG_BARLINK_UCX_WORKERS to the same value on every "
            "rank (it is rank-uniform, like every other SGLANG_BARLINK* knob)."
        )

    # ------------------------------------------------------------------
    # multi-worker plumbing
    #
    # All of this is inert at ways == 1: _progress_all is the single worker's
    # ctypes partial, _wait_split calls UcpWorker.wait, and every routing
    # table below collapses to "worker 0".
    # ------------------------------------------------------------------

    def _init_worker_routing(self) -> None:
        """Derive every worker-routing table from ``self.workers``.

        A method rather than inline ``__init__`` code because the unit tests
        build a transport with ``__new__`` and a fake worker, and a routing
        table that only exists inside ``__init__`` is one the fixtures have to
        re-derive by hand -- which is how a fixture silently stops mirroring
        the thing it is standing in for.
        """
        self.ways = len(self.workers)
        # Every existing call site, and the r3val harness, address worker 0 by
        # this name. Kept as the single-worker configuration's only worker so
        # nothing below has to branch on `ways` to reach it.
        self.worker = self.workers[0]
        # Worker carrying the flat exchange with each peer. Symmetric in the
        # pair, so the two ends agree without exchanging anything; a rule that
        # was not symmetric (say `peer % ways`) would have rank 0 posting on
        # worker 1 while rank 1 listened on worker 0.
        self._peer_worker = tuple(
            (self.rank + p) % self.ways for p in range(self.world_size)
        )
        # Same table, resolved to the worker object. The flat fast paths index
        # this instead of looking `self.worker` up, so the single-worker
        # configuration pays a tuple index where it used to pay an attribute
        # load -- not a new branch on the decode path.
        self._peer_w = tuple(self.workers[i] for i in self._peer_worker)
        # Bidirectional only when asked for AND with a second worker to put the
        # second direction on: one worker running both directions would
        # serialise them again AND double its outstanding requests, which is
        # the opposite of what the ring is for.
        self._ring_ways = (
            2
            if self.ring_bidir
            and self.ways >= 2
            and self.world_size <= _BIDIR_MAX_WORLD
            else 1
        )
        # Progress callback handed to the interleaved host passes. Bound to
        # the single worker's ctypes partial when there is only one, so the
        # default configuration keeps exactly the call it had before.
        self._progress_all = (
            self.worker.progress if self.ways == 1 else self._progress_every
        )

    def _progress_every(self) -> None:
        for w in self.workers:
            w.progress()

    def _wait_split(self, req_lists) -> None:
        """Wait on ``req_lists[i]`` on ``self.workers[i]``."""
        if self.ways == 1:
            self.worker.wait(req_lists[0])
            return
        wait_multi(self.workers, req_lists, _TIMEOUT_S)

    def _wireup(self) -> None:
        """One-byte exchange with every peer on every worker.

        The barrier alone is not enough once ways > 1: at world 2 the flat
        exchange's ``(r + p) % ways`` rule routes the only peer pair to ONE
        worker, so the other worker's endpoints would still be cold and the
        first ring collective would pay a wireup handshake inside a forward
        pass -- exactly what wiring up at construction exists to prevent.
        """
        for i, w in enumerate(self.workers):
            send_view, send_ptr, _ = self._slot(f"wu_s{i}", 1, torch.uint8)
            send_view.zero_()
            seq = self._next_seq()
            reqs = []
            for p in self._peers:
                rptr = self._slot(f"wu_r{i}_{p}", 1, torch.uint8)[1]
                reqs.append(w.post_recv(rptr, 1, _tag(seq, 0, p, 0)))
            for p in self._peers:
                reqs.append(w.post_send(p, send_ptr, 1, _tag(seq, 0, self.rank, 0)))
            w.wait(reqs)

    # ------------------------------------------------------------------
    # transport seam
    # ------------------------------------------------------------------

    def handles(self, op: str, nbytes: int) -> bool:
        return op in self.BARLINK_OPS

    # ------------------------------------------------------------------
    # staging buffers
    #
    # INTERNAL ONLY. Nothing in this file may return one of these, or a view
    # of one, to a caller: two collectives of the same shape would alias, and
    # the second would silently overwrite a result the model still holds. That
    # exact bug has already been paid for once here -- see
    # BarlinkCommunicator._get_out_buf. Every public method below allocates its
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
            # A still-running async read OUT of the old, smaller buffer must
            # complete before the tensor is dropped -- freeing pinned memory
            # under an in-flight cudaMemcpyAsync is a use-after-free.
            self._slot_guard(key)
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
    # non_blocking copy-out (task #246)
    #
    # Two kinds of device-boundary copies live in this file, and only one of
    # them must synchronize:
    #
    # * Copies INTO a staging slot before a post (the D2H stage-in) must be
    #   BLOCKING: UCX reads and writes the pinned buffer from the CPU/NIC
    #   side, outside any stream order, so the bytes must be resident before
    #   the post is issued. That one drain per collective is irreducible on
    #   this transport -- and it is why this transport can never be inside a
    #   CUDA-graph capture (see CAPTURABLE_BARLINK_TRANSPORTS).
    #
    # * Copies OUT of a staging slot into the device-side result have only
    #   one consumer: the compute stream itself, which orders them against
    #   every later kernel. Blocking them (`Tensor.copy_` without
    #   non_blocking goes through c10 memcpy_and_sync ->
    #   cudaStreamSynchronize) drained the launch pipeline once per copy --
    #   3 of the 4 drains of every all_gather, 1 of the 2 of every
    #   all_reduce, 416 drains per cross-rig verify forward. These are the
    #   copies made non_blocking here.
    #
    # Same pattern as the gloo plane (barlink.py: events + non_blocking); the
    # dedicated copy stream it also uses buys nothing here, because at
    # copy-out time the current stream is empty (the stage-in already
    # drained it), so there is no compute to overlap the copy WITH -- a
    # separate stream would only add a wait_stream edge.
    #
    # The lifetime hazard this creates -- a slot must not be rewritten (D2H
    # stage, CPU downcast, NIC recv) while the device may still be reading
    # it -- is NOT left as a comment-only stream-ordering argument (that is
    # the shared-buffer bug family this repo keeps paying for): every
    # rewrite site calls _slot_guard(key) first, and _h2d_async records the
    # event it waits on.
    # ------------------------------------------------------------------

    def _slot_guard(self, key: str) -> None:
        """Wait for the last async device read OUT of slot ``key``, if any.

        Sub-microsecond when the copy has long completed (the steady-state
        case: a whole layer of compute runs between two uses of one slot);
        never a pipeline drain -- it waits on the recorded copy only, not on
        the stream.
        """
        ev = self._h2d_events.get(key)
        if ev is not None:
            ev.synchronize()

    def _async_h2d_ok(self, dst: torch.Tensor) -> bool:
        """Whether the copy-out into ``dst`` may be enqueued non_blocking.

        Separated out so the CPU-only unit tests can force it on and drive
        the event bookkeeping without a GPU.
        """
        return self._use_cuda and dst.is_cuda

    def _h2d_async(self, dst: torch.Tensor, host: torch.Tensor, key: str) -> None:
        """Copy staging slot ``key`` (view ``host``) into ``dst`` without
        draining the compute stream. ``dst`` must be contiguous (every caller
        passes a flat view of a freshly allocated output)."""
        if not self._async_h2d_ok(dst):
            _host_copy_(dst, host)
            return
        if host.dtype != dst.dtype:
            # Downcast on the CPU into a pinned slot, then a same-dtype
            # non_blocking H2D. A direct cross-dtype `copy_` also converts
            # host-side (aten Copy.cu materialises a converted temporary on
            # the source device), but that temporary is PAGEABLE -- and a
            # pageable-source cudaMemcpyAsync synchronizes the stream first,
            # which is exactly the drain being removed. Same CPU convert
            # kernel either way, so the result bytes are identical; only the
            # staging of the memcpy changes.
            dn_key = key + ":dn"
            self._slot_guard(dn_key)
            down = self._staging(dn_key, dst.numel(), dst.dtype)
            _host_copy_(down, host)
            host, key = down, dn_key
        dst.copy_(host, non_blocking=True)
        ev = self._h2d_events.get(key)
        if ev is None:
            ev = torch.cuda.Event()
            self._h2d_events[key] = ev
        ev.record()

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
        self, peer: int, ptr: int, total: int, seq, phase, chunk0=0, w=None
    ) -> list:
        """Post ``total`` bytes at ``ptr`` to ``peer``, split into chunks.

        ``w`` is the worker to post on; ``None`` means worker 0, which is the
        only one that exists in the default configuration.

        The single-chunk case is spelled out separately because it is what
        every small collective and every step of the pipeline below hits: the
        loop's `while`, `min` and index bookkeeping are pure overhead there,
        and at 8 KiB the whole collective is only ~12 us.
        """
        if w is None:
            w = self.worker
        if total <= self.chunk_bytes:
            return [w.post_send(peer, ptr, total, _tag(seq, phase, self.rank, chunk0))]
        reqs = []
        off, idx = 0, chunk0
        while off < total:
            n = min(self.chunk_bytes, total - off)
            reqs.append(
                w.post_send(peer, ptr + off, n, _tag(seq, phase, self.rank, idx))
            )
            off += n
            idx += 1
        return reqs

    def _post_recv_bytes(
        self, src: int, ptr: int, total: int, seq, phase, chunk0=0, w=None
    ) -> list:
        if w is None:
            w = self.worker
        if total <= self.chunk_bytes:
            return [w.post_recv(ptr, total, _tag(seq, phase, src, chunk0))]
        reqs = []
        off, idx = 0, chunk0
        while off < total:
            n = min(self.chunk_bytes, total - off)
            reqs.append(w.post_recv(ptr + off, n, _tag(seq, phase, src, idx)))
            off += n
            idx += 1
        return reqs

    def _post_send(
        self, peer: int, view: torch.Tensor, seq, phase, chunk0=0, w=None
    ) -> list:
        """Post ``view`` to ``peer``, split into chunk_bytes pieces."""
        return self._post_send_bytes(
            peer,
            view.data_ptr(),
            view.numel() * view.element_size(),
            seq,
            phase,
            chunk0,
            w,
        )

    def _post_recv(
        self, src: int, view: torch.Tensor, seq, phase, chunk0=0, w=None
    ) -> list:
        return self._post_recv_bytes(
            src,
            view.data_ptr(),
            view.numel() * view.element_size(),
            seq,
            phase,
            chunk0,
            w,
        )

    def _next_seq(self) -> int:
        """Collective counter.

        Rank-uniform because barlink's contract is that every rank issues the
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
        pw = self._peer_worker
        reqs = [[] for _ in self.workers]
        for peer, view in recvs.items():
            reqs[pw[peer]] += self._post_recv(
                peer, view, seq, phase, 0, self.workers[pw[peer]]
            )
        for peer in recvs:
            reqs[pw[peer]] += self._post_send(
                peer, send, seq, phase, 0, self.workers[pw[peer]]
            )
        self._wait_split(reqs)

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
            pw = self._peer_worker
            reqs = [[] for _ in self.workers]
            for peer, rptr in recvs:
                reqs[pw[peer]] += self._post_recv_bytes(
                    peer, rptr, 1, seq, 0, 0, self.workers[pw[peer]]
                )
            for peer, _ in recvs:
                reqs[pw[peer]] += self._post_send_bytes(
                    peer, send_ptr, 1, seq, 0, 0, self.workers[pw[peer]]
                )
            self._wait_split(reqs)

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
            _host_add_(host, view)

    def _wire_all_reduce_ring_bidir(self, host: torch.Tensor, seq: int) -> None:
        """Two rings, opposite directions, one worker each.

        The unidirectional ring below moves ``n/W`` bytes per hop and takes
        ``2(W-1)`` hops, all of them through one UCX worker. Splitting the
        payload in half and running the second half the other way round the
        ring keeps the hop COUNT (the halves are independent, so their steps
        proceed together) and halves the bytes per hop -- and the two halves
        ride separate contexts, so neither waits on the other's progress.

        ``host`` is padded to a multiple of ``2 * world_size`` by the caller,
        so both halves split into equal blocks.

        REDUCTION ORDER. Each block is summed along the ring it travels, so
        the counter-clockwise half accumulates its ranks in the opposite order
        from the clockwise half -- and from the unidirectional ring, and from
        the flat exchange, which sums in peer order. All three are valid
        orders and none of them is the others' bit-for-bit equal in general
        (fp addition is not associative; under the default fp32 reduce the
        inputs are upcast first, which bounds the difference but does not
        remove it). What IS guaranteed, and is what the group needs, is that
        every rank sees the SAME bytes: a block is reduced exactly once, on
        one path, and then all-gathered verbatim. Switching worker count
        therefore changes the last bits of a large all_reduce exactly as
        switching the ring threshold already does.
        """
        N = self.world_size
        half = host.numel() // 2
        block = half // N
        parts = (host[:half], host[half:])
        blocks = tuple(
            tuple(parts[d][i * block : (i + 1) * block] for i in range(N))
            for d in (0, 1)
        )
        tmps = (
            self._staging("ar_ring_cw", block, host.dtype),
            self._staging("ar_ring_ccw", block, host.dtype),
        )
        # d = 0 clockwise (send right, receive left), d = 1 counter-clockwise.
        # `sign` turns both into one schedule: the block a rank sends at step
        # s is `rank + sign * s`, the one it receives is `rank + sign * (s+1)`.
        sign = (-1, 1)
        dst = ((self.rank + 1) % N, (self.rank - 1) % N)
        src = ((self.rank - 1) % N, (self.rank + 1) % N)
        ws = self.workers

        for step in range(N - 1):
            reqs = [[], []]
            recv_idx = [0, 0]
            for d in (0, 1):
                si = (self.rank + sign[d] * step) % N
                recv_idx[d] = (self.rank + sign[d] * (step + 1)) % N
                phase = 2 * step + d
                reqs[d] = self._post_recv(src[d], tmps[d], seq, phase, 0, ws[d])
                reqs[d] += self._post_send(dst[d], blocks[d][si], seq, phase, 0, ws[d])
            self._wait_split(reqs)
            for d in (0, 1):
                _host_add_(blocks[d][recv_idx[d]], tmps[d])

        base = 2 * (N - 1)
        for step in range(N - 1):
            reqs = [[], []]
            for d in (0, 1):
                si = (self.rank + sign[d] * (step - 1)) % N
                ri = (self.rank + sign[d] * step) % N
                phase = base + 2 * step + d
                reqs[d] = self._post_recv(src[d], blocks[d][ri], seq, phase, 0, ws[d])
                reqs[d] += self._post_send(dst[d], blocks[d][si], seq, phase, 0, ws[d])
            self._wait_split(reqs)

    def _wire_all_reduce_ring(self, host: torch.Tensor, seq: int) -> None:
        """Ring reduce-scatter + all-gather. Bandwidth-optimal for large ones.

        ``host`` is padded to a multiple of world_size by the caller, so every
        block is the same length and the schedule is symmetric across ranks.
        """
        if self._ring_ways == 2:
            self._wire_all_reduce_ring_bidir(host, seq)
            return
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
            _host_add_(blocks[recv_idx], tmp)

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
            # The previous all_reduce's copy-out may still be reading this
            # slot (the same-dtype case reads send_view itself).
            self._slot_guard(skey)
            # BLOCKING on purpose -- do not add non_blocking here: the posts
            # below hand this buffer to UCX, which reads it host-side outside
            # any stream order, so the bytes must be resident first. This is
            # the transport's one irreducible drain per all_reduce.
            _host_copy_(send_view, src)  # (+ upcast)
            # One request list per worker. At ways == 1 this is a single list
            # and _wait_split hands it straight to UcpWorker.wait, so the
            # hand-specialised 1-/2-request spins still apply; above that the
            # (W-1) peers are spread over the workers by the symmetric
            # `(rank + peer) % ways` rule, which is the point -- the flat
            # exchange's 2(W-1) simultaneous requests on ONE worker is the
            # cost the all_gather ring was already beating (task #263).
            pw, pws = self._peer_worker, self._peer_w
            reqs = [[] for _ in self.workers]
            recv_views = []
            for i, peer in enumerate(peers):
                rv, rptr, rnbytes = self._slot(rkeys[i], n, dtype)
                recv_views.append(rv)
                reqs[pw[peer]].append(
                    pws[peer].post_recv(rptr, rnbytes, _tag(seq, 0, peer, 0))
                )
            for peer in peers:
                reqs[pw[peer]].append(
                    pws[peer].post_send(
                        peer, send_ptr, send_nbytes, _tag(seq, 0, self.rank, 0)
                    )
                )
            self._wait_split(reqs)
            for rv in recv_views:
                _host_add_(send_view, rv)
            # Copy-out is non_blocking: its only consumer is the compute
            # stream. This removes the second of the two drains every
            # decode all_reduce used to pay.
            self._h2d_async(dst, send_view, skey)  # (+ downcast)
            return

        n_chunks = (n + chunk - 1) // chunk
        if n_chunks > 0xFFFFFF:
            raise UcxError(
                f"barlink ucx: all_reduce of {n * esz} bytes needs {n_chunks} "
                f"chunks, more than the 2^24 the tag layout encodes. Raise "
                f"SGLANG_BARLINK_UCX_CHUNK_MIB."
            )
        prog = self._progress_all if self.progress_bytes > 0 else None
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
                reqs += self._post_recv_bytes(peer, recvs[i][1], recvs[i][2], seq, 0, k)
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
            # Blocking copy-out on purpose: finish(k) must have finished
            # READING parity k & 1 before stage-in(k+2) rewrites it -- the
            # ownership contract above relies on program order, and a
            # non_blocking read here would need a per-parity event guard.
            # This branch runs only above chunk_bytes (>= 4 MiB), never in
            # decode, and its host passes already overlap the wire.
            self._staged_copy(dst[off : off + count], send_view, prog)  # (+ downcast)

    def barlink_all_reduce(self, comm, inp: torch.Tensor) -> torch.Tensor:
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
            # rank contributes zeroes there and the sum is unaffected. The
            # bidirectional ring splits the payload in two first, so its
            # divisor is 2 * world_size -- at most 2W-1 elements of padding
            # either way, i.e. bytes the wire would not have noticed.
            grain = self.world_size * self._ring_ways
            padded = ((n + grain - 1) // grain) * grain if use_ring else n
            host = self._staging("ar", padded, reduce_dtype)
            # The previous all_reduce's copy-out may still be reading this
            # slot, and both the stage-in below and the wire (the NIC writes
            # straight into `host`) rewrite it.
            self._slot_guard("ar")
            _host_copy_(host[:n], inp.reshape(-1))  # D2H (+ upcast)
            if padded > n:
                host[n:].zero_()

            if use_ring:
                self._wire_all_reduce_ring(host, seq)
            else:
                self._wire_all_reduce_flat(host, seq)

            # Copy-out is non_blocking, like the single-chunk fast path: its
            # only consumer is the compute stream. This branch stopped being
            # "decode never reaches it" when the ring threshold dropped to
            # 24 KiB (see _RING_BYTES) -- every cross-rig verify all-reduce
            # comes through here now, and leaving the blocking copy in would
            # have handed back one pipeline drain per collective in exchange
            # for the bytes the ring saves.
            self._h2d_async(out.reshape(-1), host[:n], "ar")  # (+ downcast)
            return out

    # ------------------------------------------------------------------
    # all_gather
    # ------------------------------------------------------------------

    def _pipelined_all_gather(
        self, inp: torch.Tensor, out_rows: torch.Tensor, n: int, seq: int
    ) -> None:
        """Chunked flat all_gather that overlaps the host passes with the wire.

        Same schedule and the same slot-ownership discipline as
        :meth:`_pipelined_all_reduce` -- read the ownership section there
        before changing anything here. The differences are exactly two:

        * No arithmetic. finish(k) only scatters each peer's received chunk
          into that peer's row of ``out_rows``, so pipelined and unpipelined
          agree bit for bit by construction (the selftest pins that down).
        * This rank's own slice never crosses the wire: it is copied
          device-locally from ``inp`` into its output row inside finish(k) --
          on a GPU that is a D2D copy that skips the host round trip
          entirely, and doing it per chunk keeps it underneath chunk k+1's
          transfer like every other finish pass.

        ``out_rows`` is the caller's freshly allocated ``(world, *shape)``
        output -- contiguous by construction, so the per-rank flat views
        below are views, never copies.
        """
        dtype = inp.dtype
        esz = dtype.itemsize
        chunk = max(self.chunk_bytes // esz, 1)
        peers = self._peers
        src = inp.reshape(-1)
        dsts = tuple(out_rows[p].reshape(-1) for p in range(self.world_size))
        own = dsts[self.rank]

        n_chunks = (n + chunk - 1) // chunk
        if n_chunks > 0xFFFFFF:
            raise UcxError(
                f"barlink ucx: all_gather of {n * esz} bytes needs {n_chunks} "
                f"chunks, more than the 2^24 the tag layout encodes. Raise "
                f"SGLANG_BARLINK_UCX_CHUNK_MIB."
            )
        prog = self._progress_all if self.progress_bytes > 0 else None
        staged: dict = {}

        def stage(k: int) -> None:
            par = k & 1
            off = k * chunk
            count = n - off if off + chunk > n else chunk
            skey, rkeys = self._ag_keys[par]
            send_view, send_ptr, send_nbytes = self._slot(skey, count, dtype)
            self._staged_copy(send_view, src[off : off + count], prog)
            recvs = [self._slot(rkeys[i], count, dtype) for i in range(len(peers))]
            staged[par] = (off, count, send_ptr, send_nbytes, recvs)

        def post(k: int) -> list:
            _, _, send_ptr, send_nbytes, recvs = staged[k & 1]
            reqs = []
            # Receives first, as everywhere else in this file.
            for i, peer in enumerate(peers):
                reqs += self._post_recv_bytes(peer, recvs[i][1], recvs[i][2], seq, 0, k)
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
            off, count, _, _, recvs = staged[k & 1]
            # Blocking copy-outs on purpose -- same parity-ownership reason
            # as the finish pass of _pipelined_all_reduce; never decode-hot.
            for i, peer in enumerate(peers):
                self._staged_copy(dsts[peer][off : off + count], recvs[i][0], prog)
            self._staged_copy(own[off : off + count], src[off : off + count], prog)

    def _ag_view(self, out_flat: torch.Tensor, input_size, dim: int) -> torch.Tensor:
        """View a flat ``world * numel`` gather result as the caller's shape.

        Shared by every branch that builds the result FLAT. For ``dim == 0``
        -- the overwhelmingly common gather axis -- this is a single ``view``:
        no movedim, no reshape.
        """
        if dim == 0:
            return out_flat.view(
                (self.world_size * input_size[0],) + tuple(input_size[1:])
                if len(input_size)
                else (self.world_size,)
            )
        return (
            out_flat.view((self.world_size,) + tuple(input_size))
            .movedim(0, dim)
            .reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
        )

    def _ring_all_gather(self, inp: torch.Tensor, n: int, seq: int) -> torch.Tensor:
        """Ring all_gather over one host-side ``(world, n)`` staging block.

        Taken above ``SGLANG_BARLINK_UCX_AG_RING_KIB`` (32 KiB, measured -- see
        the table there). Note that this ring saves no bytes: an all_gather
        must deliver ``(W-1) * n`` into every rank and the flat exchange
        already moves exactly that in one round trip, where
        ``_wire_all_reduce_ring`` halves its volume. What it saves is
        concurrent work for the one single-threaded UCX worker per rank --
        two outstanding requests per step instead of ``2(W-1)`` at once.
        """
        N = self.world_size
        dtype = inp.dtype
        # One contiguous (world, n) host block: rows are the wire units and
        # the whole thing is one H2D copy-out at the end.
        host = self._staging("ag_ring", N * n, dtype)
        self._slot_guard("ag_ring")
        rows = host.view(N, n)
        # BLOCKING on purpose, like every other stage-in here: UCX reads this
        # buffer host-side, outside any stream order.
        _host_copy_(rows[self.rank], inp.reshape(-1))

        if self._ring_ways == 2:
            # Bidirectional: the left half of every row travels clockwise on
            # worker 0, the right half counter-clockwise on worker 1. Same
            # W-1 steps, half the bytes per hop, and the two halves do not
            # queue behind each other on one worker -- which is the whole
            # reason this ring beat the flat exchange in the first place.
            #
            # Bit-identical to the unidirectional ring by construction, not by
            # measurement: an all_gather does no arithmetic, so every output
            # byte is a copy of the same input byte whichever way round it
            # came. (The all_reduce ring, which does add, is the one where
            # direction is visible in the last bits -- see
            # _wire_all_reduce_ring_bidir.)
            cut = n // 2
            spans = ((0, cut), (cut, n))
            sign = (-1, 1)
            dst = ((self.rank + 1) % N, (self.rank - 1) % N)
            src = ((self.rank - 1) % N, (self.rank + 1) % N)
            ws = self.workers
            for step in range(N - 1):
                reqs = [[], []]
                for d in (0, 1):
                    a, b = spans[d]
                    if a == b:  # odd tiny payload: one direction carries all
                        continue
                    si = (self.rank + sign[d] * step) % N
                    ri = (self.rank + sign[d] * (step + 1)) % N
                    phase = 2 * step + d
                    reqs[d] = self._post_recv(
                        src[d], rows[ri][a:b], seq, phase, 0, ws[d]
                    )
                    reqs[d] += self._post_send(
                        dst[d], rows[si][a:b], seq, phase, 0, ws[d]
                    )
                self._wait_split(reqs)
        else:
            right = (self.rank + 1) % N
            left = (self.rank - 1) % N
            for step in range(N - 1):
                # Forward the row received in the previous step; after W-1
                # steps every row has travelled the whole ring exactly once.
                send_idx = (self.rank - step) % N
                recv_idx = (self.rank - step - 1) % N
                reqs = self._post_recv(left, rows[recv_idx], seq, step)
                reqs += self._post_send(right, rows[send_idx], seq, step)
                self.worker.wait(reqs)

        out_flat = torch.empty(N * n, dtype=dtype, device=inp.device)
        self._h2d_async(out_flat, host, "ag_ring")
        return out_flat

    def barlink_all_gather(self, comm, inp: torch.Tensor, dim: int) -> torch.Tensor:
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

            if (
                self.ag_ring_bytes
                and self.world_size > 2
                and (n * inp.dtype.itemsize >= self.ag_ring_bytes)
            ):
                return self._ag_view(
                    self._ring_all_gather(inp, n, seq), input_size, dim
                )

            if self.pipeline and n <= max(self.chunk_bytes // inp.dtype.itemsize, 1):
                # Single-chunk fast path: the same one-step flat exchange as
                # the generic branch below, with the per-call staging dict,
                # its eagerly formatted keys and the extra dispatch layers
                # stripped -- the all_gather twin of the single-chunk branch
                # in _pipelined_all_reduce. This is what a decode-sized
                # all_gather (logits gather, DCP merges) hits, and at that
                # size the bookkeeping was a third of the whole collective.
                #
                # The output is built FLAT and viewed as (world, n) rows:
                # per-rank `select + copy` is two dispatches instead of the
                # `select + reshape + copy` of the generic path, and for
                # dim == 0 (the overwhelmingly common gather axis) the result
                # is a single `view` -- no movedim, no reshape.
                src = inp.reshape(-1)
                skey, rkeys = self._ag_keys[0]
                send_view, send_ptr, send_nbytes = self._slot(skey, n, inp.dtype)
                # BLOCKING on purpose -- do not add non_blocking here: the
                # posts below hand this buffer to UCX, which reads it
                # host-side outside any stream order. The transport's one
                # irreducible drain per all_gather (was one of four).
                _host_copy_(send_view, src)
                # Per-worker request lists, same symmetric peer split as the
                # all_reduce fast path above.
                pw, pws = self._peer_worker, self._peer_w
                reqs = [[] for _ in self.workers]
                recv_views = []
                for i, peer in enumerate(self._peers):
                    rv, rptr, rnbytes = self._slot(rkeys[i], n, inp.dtype)
                    recv_views.append(rv)
                    # The NIC writes into this slot from the moment the recv
                    # is posted; the previous all_gather's non_blocking
                    # copy-out of it must have completed first.
                    self._slot_guard(rkeys[i])
                    reqs[pw[peer]].append(
                        pws[peer].post_recv(rptr, rnbytes, _tag(seq, 0, peer, 0))
                    )
                for peer in self._peers:
                    reqs[pw[peer]].append(
                        pws[peer].post_send(
                            peer, send_ptr, send_nbytes, _tag(seq, 0, self.rank, 0)
                        )
                    )
                out_flat = torch.empty(
                    self.world_size * n, dtype=inp.dtype, device=inp.device
                )
                rows = out_flat.view(self.world_size, n)
                # Own slice straight from the input (on a GPU a D2D copy that
                # never touches the host staging buffer), placed BETWEEN the
                # posts and the wait so it runs while the wire is in flight.
                # Above one progress block the copy is interleaved with
                # worker progress -- a multi-hundred-us memcpy with no
                # progress call would starve the RNDV handshake of the very
                # transfer it is trying to hide under.
                if 0 < self.progress_bytes < n * inp.dtype.itemsize:
                    self._staged_copy(rows[self.rank], src, self._progress_all)
                else:
                    _host_copy_(rows[self.rank], src)
                self._wait_split(reqs)
                # Peer-row copy-outs are non_blocking: their only consumer
                # is the compute stream. These three (W-1) copies were 3 of
                # the 4 drains every decode all_gather used to pay.
                for i, peer in enumerate(self._peers):
                    self._h2d_async(rows[peer], recv_views[i], rkeys[i])
                return self._ag_view(out_flat, input_size, dim)

            output = torch.empty(
                (self.world_size,) + tuple(input_size),
                dtype=inp.dtype,
                device=inp.device,
            )

            if self.pipeline:
                self._pipelined_all_gather(inp, output, n, seq)
            else:
                # Pre-pipelining path, kept verbatim as the A/B control
                # (SGLANG_BARLINK_UCX_PIPELINE=0).
                send = self._staging("ag_s", n, inp.dtype)
                _host_copy_(send, inp.reshape(-1))
                recvs = {
                    p: self._staging(f"ag_r{p}", n, inp.dtype)
                    for p in range(self.world_size)
                    if p != self.rank
                }
                self._exchange_all(send, recvs, seq)
                for p in range(self.world_size):
                    src = send if p == self.rank else recvs[p]
                    _host_copy_(output[p].reshape(-1), src)

            output = output.movedim(0, dim)
            return output.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )

    # ------------------------------------------------------------------
    # reduce_scatter
    # ------------------------------------------------------------------

    def barlink_reduce_scatter(self, comm, inp: torch.Tensor, dim: int) -> torch.Tensor:
        """Sum across ranks, then keep this rank's slice along ``dim``.

        Composed from all_reduce exactly as the gloo plane composes it,
        including the ``movedim(dim, 0)`` orientation -- writing it the other
        way round silently scatters the wrong axis from ndim >= 3 while every
        shape check still passes. A native ring reduce-scatter would move
        1/N of the bytes; that is a separate, testable change.
        """
        if dim < 0:
            dim += inp.dim()
        reduced = self.barlink_all_reduce(comm, inp.contiguous())
        moved = reduced.movedim(dim, 0).contiguous()
        assert moved.shape[0] % self.world_size == 0
        chunk = moved.shape[0] // self.world_size
        shard = moved[self.rank * chunk : (self.rank + 1) * chunk]
        return shard.movedim(0, dim).contiguous()

    # ------------------------------------------------------------------
    # broadcast
    # ------------------------------------------------------------------

    def barlink_broadcast(self, comm, tensor: torch.Tensor, src: int) -> torch.Tensor:
        """In-place broadcast from ``src``; returns ``tensor`` itself.

        Host-staged, so this synchronises with the CPU and -- like the gloo
        plane and unlike the device transport -- must not be called from
        inside a CUDA-graph capture.
        """
        with self._lock:
            seq = self._next_seq()
            n = tensor.numel()
            host = self._staging("bc", n, tensor.dtype)
            # Both roles rewrite the slot (source: D2H stage; receiver: NIC
            # write on recv); a previous broadcast's non_blocking copy-out
            # must have completed first.
            self._slot_guard("bc")
            if self.rank == src:
                # BLOCKING on purpose -- do not add non_blocking here: the
                # sends below hand this buffer to UCX, which reads it
                # host-side outside any stream order.
                host.copy_(tensor.reshape(-1))
                reqs = []
                for peer in range(self.world_size):
                    if peer != self.rank:
                        reqs += self._post_send(peer, host, seq, 0)
                self.worker.wait(reqs)
            else:
                self.worker.wait(self._post_recv(src, host, seq, 0))
                # Receiver copy-out is non_blocking: the caller's tensor is
                # consumed on the compute stream (or is a CPU tensor, in
                # which case _h2d_async copies synchronously).
                self._h2d_async(tensor.reshape(-1), host, "bc")
            return tensor

    # ------------------------------------------------------------------
    # async collectives -- issue now, wait later (task #198, block 4)
    #
    # The three contracts, in the order they have historically been broken:
    #
    # OWNERSHIP. An async collective's staging buffers are acquired from a
    # free-list pool at issue and released only inside wait_async, after the
    # last byte has been read out of them. They are owned by the HANDLE, not
    # by the transport's per-call slot sets -- the sync paths' two-parity
    # rotation is safe precisely because a sync collective cannot outlive its
    # call, and an async one can. Nothing here may hand a pool buffer (or a
    # view of one) to the caller; results are freshly allocated in wait.
    # The caller's INPUT is free the moment issue returns: it is staged
    # before the first post, and -- for all_gather -- the own-rank slice is
    # later copied out of the STAGING slot, never re-read from the input.
    #
    # PROGRESS. UCX makes no progress unattended. Issue ends with a single
    # progress pass to push eager sends onto the wire; from then on the
    # transfer advances in hardware for eager-sized payloads (every
    # decode-sized collective) and completes under the progress loop inside
    # wait_async. Rendezvous-sized payloads additionally need both peers to
    # progress, so their overlap window degrades toward the sync cost --
    # still correct, just not faster. There is deliberately no progress
    # thread: the worker is created THREAD_MODE_SINGLE (the fast mode), and
    # every entry point here runs under self._lock on the caller's thread.
    #
    # ORDER. _next_seq counts ISSUES, under the lock, so the group-uniform
    # contract ("every rank issues the same collectives in the same order")
    # covers async issues exactly like sync calls. Each handle carries its
    # seq, and every transfer's tag carries (seq, src, chunk) with exact
    # matching -- so wait_async order is free: handles may be awaited out of
    # issue order without any risk of crossed payloads.
    # ------------------------------------------------------------------

    def _pool_acquire(self, numel: int, dtype: torch.dtype) -> tuple:
        """``(view, ptr, nbytes, buf, size_class)`` from the async free list.

        Size classes are powers of two >= 4096, so steady-state decode
        (same shapes forever) recycles the same few buffers and never grows
        the pool.
        """
        nbytes = max(numel * dtype.itemsize, 1)
        cls = 1 << max(nbytes - 1, 4095).bit_length()
        free = self._async_free.get(cls)
        if free:
            buf, ev = free.pop()
            if ev is not None:
                # The buffer goes straight back into use (CPU stage-in write
                # or NIC recv); the non_blocking read of its previous life,
                # recorded at release, must have completed first. Same
                # slot-lifetime rule as _slot_guard, pool-shaped.
                ev.synchronize()
        else:
            pin = self.device.type == "cuda" and torch.cuda.is_available()
            buf = torch.empty(cls, dtype=torch.uint8, pin_memory=pin)
        view = buf[:nbytes].view(dtype)
        return (view, view.data_ptr(), nbytes, buf, cls)

    def _pool_release(self, rec: tuple, ev=None) -> None:
        """Return a pool record to the free list.

        ``ev`` is the completion event of the last non_blocking device read
        out of the buffer, or None when every read of it was synchronous.
        Stored with the buffer and waited on in _pool_acquire, so a handle's
        result copy may be in flight while the slots are already reusable.
        """
        self._async_free.setdefault(rec[4], []).append((rec[3], ev))

    def all_reduce_async(self, comm, inp: torch.Tensor) -> "_UcxAsyncHandle":
        """Stage and post an all_reduce; complete it later with wait_async.

        Always the flat exchange, never the ring: the async path exists for
        the decode-sized collectives the consumer wants to hide behind
        compute, and the ring's 2(N-1) lock-step phases cannot run
        unattended anyway (each phase needs the previous one's arrival).
        Oversized payloads stay correct -- all chunks are posted here and
        completed in wait -- they just lean on the RNDV note above.
        """
        with self._lock:
            seq = self._next_seq()
            reduce_dtype = (
                torch.float32
                if _FP32_REDUCE and inp.dtype in (torch.float16, torch.bfloat16)
                else inp.dtype
            )
            n = inp.numel()
            send = self._pool_acquire(n, reduce_dtype)
            # BLOCKING on purpose, and the reason the overlap switch measured
            # neutral: the posts below hand this buffer to UCX, which reads
            # it host-side outside any stream order, so the drain happens at
            # ISSUE -- exactly where the sync path pays it. Making the issue
            # non-draining means a non_blocking D2H plus an event, with the
            # posts deferred into wait_async behind that event -- a rework of
            # the PROGRESS and ORDER contracts above (the wire would no
            # longer start at issue), not a copy-flag change. Named as
            # follow-up in task #246.
            send[0].copy_(inp.reshape(-1))  # (+ upcast); input is free after this
            recvs = [self._pool_acquire(n, reduce_dtype) for _ in self._peers]
            reqs = []
            for i, peer in enumerate(self._peers):
                reqs += self._post_recv_bytes(peer, recvs[i][1], recvs[i][2], seq, 0)
            for peer in self._peers:
                reqs += self._post_send_bytes(peer, send[1], send[2], seq, 0)
            self.worker.progress()
            return _UcxAsyncHandle(
                "all_reduce",
                seq,
                reqs,
                send,
                recvs,
                tuple(inp.shape),
                inp.dtype,
                inp.device,
                n,
                None,
            )

    def all_gather_async(self, comm, inp: torch.Tensor, dim: int) -> "_UcxAsyncHandle":
        """Stage and post an all_gather; complete it later with wait_async."""
        with self._lock:
            seq = self._next_seq()
            if dim < 0:
                dim += inp.dim()
            inp = inp.contiguous()
            n = inp.numel()
            send = self._pool_acquire(n, inp.dtype)
            # BLOCKING on purpose; same issue-time drain as all_reduce_async
            # above, same follow-up.
            send[0].copy_(inp.reshape(-1))  # input is free after this
            recvs = [self._pool_acquire(n, inp.dtype) for _ in self._peers]
            reqs = []
            for i, peer in enumerate(self._peers):
                reqs += self._post_recv_bytes(peer, recvs[i][1], recvs[i][2], seq, 0)
            for peer in self._peers:
                reqs += self._post_send_bytes(peer, send[1], send[2], seq, 0)
            self.worker.progress()
            return _UcxAsyncHandle(
                "all_gather",
                seq,
                reqs,
                send,
                recvs,
                tuple(inp.shape),
                inp.dtype,
                inp.device,
                n,
                dim,
            )

    def wait_async(self, handle: "_UcxAsyncHandle") -> torch.Tensor:
        """Progress the worker until ``handle`` completes; return its result.

        The result is freshly allocated -- never a pool buffer, never a view
        of one. The handle's slots go back to the free list before this
        returns, so a handle must be awaited exactly once: a second wait
        would read slots that a later collective may already own, and it
        raises instead.
        """
        with self._lock:
            if handle.done:
                raise UcxError(
                    "barlink ucx: wait_async called twice on the same handle "
                    f"(op={handle.op}, seq={handle.seq})."
                )
            self.worker.wait(handle.reqs)
            handle.done = True
            # Result copies below are non_blocking where the consumer is the
            # compute stream (same rationale as the sync paths' _h2d_async);
            # `extra` is the pinned downcast slot, released with the same
            # event as the handle's own slots.
            extra = None
            async_read = False
            try:
                if handle.op == "all_reduce":
                    acc = handle.send[0]
                    for rec in handle.recvs:
                        acc.add_(rec[0])
                    out = torch.empty(
                        handle.shape, dtype=handle.dtype, device=handle.device
                    )
                    flat = out.reshape(-1)
                    if not self._async_h2d_ok(flat):
                        flat.copy_(acc)  # (+ downcast)
                    elif acc.dtype != flat.dtype:
                        # Pinned CPU downcast + same-dtype non_blocking H2D;
                        # byte-identical to the blocking converting copy_
                        # (which also converts host-side) minus its
                        # pageable-temporary stream drain. See _h2d_async.
                        extra = self._pool_acquire(handle.n, flat.dtype)
                        extra[0].copy_(acc)
                        flat.copy_(extra[0], non_blocking=True)
                        async_read = True
                    else:
                        flat.copy_(acc, non_blocking=True)
                        async_read = True
                    return out

                # all_gather -- same flat-output construction as the sync
                # fast path; own slice from the STAGED copy (see OWNERSHIP).
                W = self.world_size
                n = handle.n
                out_flat = torch.empty(W * n, dtype=handle.dtype, device=handle.device)
                rows = out_flat.view(W, n)
                if self._async_h2d_ok(out_flat):
                    rows[self.rank].copy_(handle.send[0], non_blocking=True)
                    for i, peer in enumerate(self._peers):
                        rows[peer].copy_(handle.recvs[i][0], non_blocking=True)
                    async_read = True
                else:
                    rows[self.rank].copy_(handle.send[0])
                    for i, peer in enumerate(self._peers):
                        rows[peer].copy_(handle.recvs[i][0])
                shape = handle.shape
                dim = handle.dim
                if dim == 0:
                    return out_flat.view((W * shape[0],) + shape[1:])
                return (
                    out_flat.view((W,) + shape)
                    .movedim(0, dim)
                    .reshape(shape[:dim] + (W * shape[dim],) + shape[dim + 1 :])
                )
            finally:
                ev = None
                if async_read:
                    # One event after the last read covers every slot of this
                    # handle: the reads are enqueued on one stream and
                    # therefore complete in order.
                    ev = torch.cuda.Event()
                    ev.record()
                self._pool_release(handle.send, ev)
                for rec in handle.recvs:
                    self._pool_release(rec, ev)
                if extra is not None:
                    self._pool_release(extra, ev)

    def poke_async(self) -> None:
        """One unlocked progress pass.

        Optional: a consumer with an outstanding handle may call this
        between compute steps to nudge RNDV handshakes along. Never
        required for correctness -- wait_async completes everything.
        """
        self.worker.progress()

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
            for w in self.workers:
                w.timeout_s = 5.0
            self.barrier()
        except Exception:
            pass
        # In-flight non_blocking reads out of the staging/pool buffers must
        # complete before the pinned memory is dropped.
        for ev in self._h2d_events.values():
            try:
                ev.synchronize()
            except Exception:
                pass
        self._h2d_events.clear()
        for free in self._async_free.values():
            for _, ev in free:
                if ev is not None:
                    try:
                        ev.synchronize()
                    except Exception:
                        pass
        self._async_free.clear()
        self._staging_bufs.clear()
        self._view_cache.clear()
        # Holds raw addresses into the buffers just dropped.
        self._bar_slots = None
        for i, w in enumerate(self.workers):
            try:
                w.close()
            except Exception as e:  # teardown must never mask the real error
                logger.warning("barlink: ucx worker %d close failed (%s).", i, e)
