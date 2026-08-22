# SPDX-License-Identifier: Apache-2.0
"""Phase-boundary KV resharding runtime (#297).

The physical actuator behind the ``dcp_ratio`` rung of the #287 KV pressure
ladder: moves EXISTING KV-cache bytes between DCP ranks when the weighted
token-ratio vector flips (the #320-proven redistribution, e.g.
``7,3,3 -> 2,11,10``), at a TRUE phase boundary -- the scheduler is fully
idle -- rank-uniformly and byte-identically. DESIGN_297_kv_resharding.md is
the design record; the short form:

* NO metadata rewrite. Physical pool rows are a pure function of the global
  slot id and the vector (``layers/dcp/owner.py``); ``req_to_token``, the
  radix tree and the allocator store global ids only. A reshard is a byte
  move plus a refresh of the few caches that snapshot the vector.
* CONSENSUS FIRST, BYTES SECOND (the #287 discipline). Every
  ``consensus_interval``-th round -- gated by the replicated round counter,
  never by local state -- every rank enters ONE bounded MIN-reduction with
  ``(armed, ready, epoch, vector)``. ``epoch`` and (once all ranks are
  armed) the target vector are EQUALITY-checked: a mismatch raises the same
  loud :class:`KvReshardError` on every rank. ``armed`` and ``ready`` are
  MIN-semantics: disagreement is LEGAL and uniformly resolves to "wait" --
  arming arrives via a broadcast RPC or a ladder flip and may skew by an
  iteration, and idleness may skew while async queues drain. The
  byte-moving exchange is entered only from a group-agreed state
  ([[rank-lokaler-test-vor-kollektiv]]).
* READS BEFORE WRITES. Old and new rows overlap inside the same physical
  buffer, so the executor packs every outgoing row and (per layer) gathers
  retained rows into a temp before the first scatter touches that layer's
  buffer. The aliasing falsifier in test_kv_reshard.py pins this.
* ATOMIC AT THE BOUNDARY. The move runs synchronously inside the scheduler
  round while the server is fully idle: nothing allocates, reads or writes
  KV between the ready check and the cutover. The pool is UNTOUCHED through
  pack, exchange and checksum verification -- a failure up to there aborts
  the attempt cleanly (a later boundary may retry). Only the write phase is
  the no-return region: an error there is FATAL and loud on every rank --
  the server never serves from a mixed layout because the failure raises
  before the round continues.
* THE EXCHANGE IS DEVICE-NATIVE NCCL, deliberately. torch's gloo p2p works
  (SendWork/RecvWork) do not implement ``is_completed``, so a bounded poll
  over them can never observe completion -- measured on the first card run
  as a clean 120 s ``CollectiveTimeoutError`` on every rank (the #312
  family catching exactly the hang class it exists for). NCCL p2p works
  poll truthfully, and ``batch_isend_irecv`` gives the group-uniform op
  batch NCCL p2p requires.

Capacity comes from the fitted-ceiling reservation
(``reshard_ceiling_rows``): pools are sized at boot for every declared
vector, so no growth, no address change, no CUDA-graph recapture (graph
metadata is rebuilt host-side per replay from the refreshed backend bounds).
"""

from __future__ import annotations

import dataclasses
import logging
import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.dcp.reshard_plan import KvReshardError, build_transition
from sglang.srt.model_executor.weights_arena import uint8_checksum

logger = logging.getLogger(__name__)

LOG_PREFIX = "KV-RESHARD"

#: #363 last mile. How long an arm may sit on the same hold reason before it is
#: re-reported, and the age past which the report escalates to WARNING. An arm
#: that cannot execute is not an error -- the hold reasons are all legitimate
#: waits -- but one that has waited THIS long under load is the shape the #656
#: audit predicted (``_execute`` is gated on ``is_fully_idle()``, which a loaded
#: server may never satisfy), and it means the controller believes it acted.
HOLD_REPORT_INTERVAL_S = 60.0
HOLD_ESCALATE_AFTER_S = 300.0

#: The greppable marker for a stuck arm, so a watchdog keys on one string.
HOLD_STUCK_MARKER = f"{LOG_PREFIX} arm still pending"

#: Consensus payload fields, packed as (value, -value) pairs so ONE
#: MIN-reduction yields (min, -max) per field. ``armed``, ``ready`` and
#: ``headroom`` are MIN-checked (divergence legal -> wait or refuse);
#: ``epoch`` is equality-checked always; the vector elements are
#: equality-checked once min(armed) == 1.
#:
#: ``headroom`` is #363's addition and it is in the MIN family for one
#: reason: a rank that discovers it cannot afford the move must not act on
#: that alone. See :meth:`KvReshardRuntime._headroom_check`.
_MIN_FIELDS = ("armed", "ready", "headroom")
_EQ_FIELDS = ("epoch",)

_CHECKSUM_BYTES = 8

#: The corridor law's floor, in MiB: no card on this rig may be shown with
#: less than this much FREE memory, sampled continuously (100 ms) against the
#: NVML FREE column -- never against total-minus-used, which hides the
#: ~424/518 MiB carve-out. The reshard's transient buffers are charged
#: against free memory ABOVE this floor, so a move that would eat into the
#: user's reserve is refused rather than performed.
CORRIDOR_FLOOR_MIB = 1024

#: Rows gathered/scattered per step inside :class:`KvPoolView` (#631). The
#: indexed read ``k[rows]`` and the strided ``.contiguous()`` on the write
#: side both materialise their result before it is placed, so an unblocked
#: call holds a transient proportional to the resident sequence. Blocking
#: bounds it; the bytes produced are identical either way. Large enough
#: that the per-block launch overhead is noise against a 2 KiB row.
DEFAULT_GATHER_BLOCK_ROWS = 16384


def _gather_block_rows() -> int:
    raw = os.environ.get("SGLANG_FLIP_GATHER_ROWS", "").strip()
    if not raw:
        return DEFAULT_GATHER_BLOCK_ROWS
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_GATHER_BLOCK_ROWS


def _fmt_bytes(n: int) -> str:
    """MiB where MiB is the honest unit, bytes where it is not.

    A refusal that reports "0 MiB needed, margin -0 MiB" is arithmetically
    true and useless to read; the guard's own tests run at kilobyte scale."""
    n = int(n)
    if abs(n) >= 1024 * 1024:
        return f"{n / (1024.0 * 1024.0):.1f} MiB"
    return f"{n:,d} B"


@dataclasses.dataclass
class _MovePlan:
    """A priced move: what ``_execute`` will do, and what it will cost.

    Built at the consensus boundary so the headroom guard and the allocation
    it guards are talking about the same move (#363)."""

    t0: float
    slots: "torch.Tensor"
    tr: object
    terms: dict


def _encode(values: Sequence[int]) -> List[int]:
    out: List[int] = []
    for v in values:
        out.append(int(v))
        out.append(-int(v))
    return out


class KvPoolView:
    """Byte-level row access to the token-sharded full-attention KV pool.

    One instance wraps the per-layer K and V buffers (row-major: dim 0 is
    the compact physical row). ``read_rows`` returns the rows of ONE layer
    as a ``[n, row_nbytes]`` uint8 tensor on the pool's device, K bytes
    first and V bytes appended -- the packed wire format; ``write_rows`` is
    its exact inverse. Pure adapter: no owner arithmetic in here.
    """

    def __init__(
        self,
        k_buffers: Sequence[torch.Tensor],
        v_buffers: Sequence[torch.Tensor],
        valid_rows_fn: Optional[Callable[[], int]] = None,
    ):
        if len(k_buffers) != len(v_buffers):
            raise KvReshardError(
                f"K/V layer count mismatch: {len(k_buffers)} vs {len(v_buffers)}"
            )
        if not k_buffers:
            raise KvReshardError("pool view needs at least one layer")
        self._k = list(k_buffers)
        self._v = list(v_buffers)
        rows = {int(t.shape[0]) for t in self._k} | {int(t.shape[0]) for t in self._v}
        if len(rows) != 1:
            raise KvReshardError(f"per-layer row counts differ: {sorted(rows)}")
        self._tensor_rows = rows.pop()
        # #330 dial lane: the tensors span the reserved VA upper bound while
        # only a prefix is physically backed; the bounds check must hold
        # against the BACKED prefix, not the tensor shape. None keeps the
        # stock tensor-shape bound (fully-backed pools).
        self._valid_rows_fn = valid_rows_fn
        self.num_layers = len(self._k)

    @property
    def num_rows(self) -> int:
        if self._valid_rows_fn is not None:
            return min(self._tensor_rows, int(self._valid_rows_fn()))
        return self._tensor_rows

    def _row_nbytes(self, buf: torch.Tensor) -> int:
        return int(buf[0].numel()) * buf.element_size()

    def row_nbytes(self, layer: int) -> int:
        return self._row_nbytes(self._k[layer]) + self._row_nbytes(self._v[layer])

    @property
    def total_row_nbytes(self) -> int:
        return sum(self.row_nbytes(layer) for layer in range(self.num_layers))

    @staticmethod
    def _as_bytes(rows_data: torch.Tensor, row_nbytes: int) -> torch.Tensor:
        """Flatten ``[n, ...]`` rows to a ``[n, row_nbytes]`` uint8 view.

        The width is passed IN rather than inferred with ``view(n, -1)``:
        at n == 0 there is nothing to infer it from, and torch refuses the
        ambiguous -1 ("cannot reshape tensor of 0 elements into shape
        [0, -1]").

        An empty move is not an error. Flipping a server whose live set is
        empty -- idle, or with the prefix cache just flushed -- is the most
        ordinary case there is, and it is the one a caller reaches when it
        makes room for the flip first. Every rank died here on exactly
        that (#631 boot 20, 2026-08-08).
        """
        n = rows_data.shape[0]
        if n == 0:
            return torch.empty(
                (0, row_nbytes), dtype=torch.uint8, device=rows_data.device
            )
        return rows_data.contiguous().view(n, -1).view(torch.uint8)

    @property
    def device(self) -> torch.device:
        return self._k[0].device

    def storage_ranges(self) -> List[Tuple[object, int, int]]:
        """``(device, first byte, last byte + 1)`` of every buffer.

        Exists so a caller can ask whether two views OVERLAY THE SAME
        BYTES. The #631 waved seam interleaves reads of one layout with
        writes of the other, which is only sound while the two occupy
        disjoint storage; if they alias, every source row must be read
        before any destination row is written and the seam cannot be
        waved at all. Answering that from the actual pointers beats
        answering it from a comment.
        """
        out: List[Tuple[object, int, int]] = []
        for buf in list(self._k) + list(self._v):
            lo = int(buf.data_ptr())
            out.append((buf.device, lo, lo + buf.numel() * buf.element_size()))
        return out

    def overlaps(self, other: "KvPoolView") -> bool:
        """True when any of this view's bytes are also ``other``'s bytes."""
        mine = self.storage_ranges()
        theirs = other.storage_ranges()
        for dev_a, lo_a, hi_a in mine:
            for dev_b, lo_b, hi_b in theirs:
                if dev_a == dev_b and lo_a < hi_b and lo_b < hi_a:
                    return True
        return False

    def read_rows_into(self, layer: int, rows: torch.Tensor, out: torch.Tensor) -> None:
        """``read_rows`` without materialising the concatenation.

        Writes exactly what ``read_rows`` returns into a caller-owned
        ``[n, row_nbytes]`` uint8 destination, which is what lets a packer
        fill ONE exact-size staging buffer instead of building a tensor
        per layer and concatenating them. That distinction is the whole
        point: the concatenating form held a peer's payload two to three
        times at its peak, and that transient is the term that breaches
        the VRAM corridor and starves the flip's own affordability gate
        (#631, HANDOFF_664 section 13).

        The per-layer gather is still materialised. It is one layer's rows
        -- 1/(2L) of a payload -- and it dies before the next layer's, so
        it is a bounded staging window rather than a term that grows with
        the sequence.
        """
        k, v = self._k[layer], self._v[layer]
        n = int(rows.numel())
        k_bytes = self._row_nbytes(k)
        expect = k_bytes + self._row_nbytes(v)
        if tuple(out.shape) != (n, expect):
            raise KvReshardError(
                f"read_rows_into destination shape {tuple(out.shape)} != "
                f"({n}, {expect}) for layer {layer}"
            )
        if out.dtype != torch.uint8:
            raise KvReshardError(
                f"read_rows_into destination must be uint8, got {out.dtype}"
            )
        if n == 0:
            return
        idx = rows.to(k.device)
        # THE GATHER IS BLOCKED OVER ROWS. ``k[idx]`` materialises a fresh
        # tensor of the indexed rows before it is copied into ``out``, so
        # an unblocked gather holds one layer's K (then V) for the WHOLE
        # row list -- a transient proportional to the resident sequence,
        # which is the same shape of term the waved seam exists to remove
        # (#631). Blocking bounds it to ``FLIP_GATHER_ROWS`` rows without
        # changing a single output byte: the destination slices are
        # disjoint and written in ascending order either way.
        block = _gather_block_rows()
        for lo in range(0, n, block):
            hi = min(lo + block, n)
            sub = idx[lo:hi]
            out[lo:hi, :k_bytes] = self._as_bytes(k[sub], k_bytes)
            out[lo:hi, k_bytes:] = self._as_bytes(v[sub], expect - k_bytes)

    def read_rows(self, layer: int, rows: torch.Tensor) -> torch.Tensor:
        """``[n, row_nbytes]`` uint8 tensor of layer ``layer``'s rows, on the
        pool's device (the exchange stays device-native; only the injected
        test channels ever see host tensors).

        Implemented ON TOP of ``read_rows_into`` rather than beside it, so
        the streamed packing path and this one cannot drift into producing
        different bytes for the same rows -- a divergence the payload
        checksum could not catch, because sender and receiver would both
        compute it over the same wrong order.
        """
        n = int(rows.numel())
        out = torch.empty(
            (n, self.row_nbytes(layer)),
            dtype=torch.uint8,
            device=self._k[layer].device,
        )
        self.read_rows_into(layer, rows, out)
        return out

    def write_rows(self, layer: int, rows: torch.Tensor, data: torch.Tensor) -> None:
        """Inverse of ``read_rows``: scatter packed bytes into layer rows."""
        k, v = self._k[layer], self._v[layer]
        n = int(rows.numel())
        k_bytes = self._row_nbytes(k)
        expect = k_bytes + self._row_nbytes(v)
        if data.shape != (n, expect):
            raise KvReshardError(
                f"payload shape {tuple(data.shape)} != ({n}, {expect}) for "
                f"layer {layer}"
            )
        idx = rows.to(k.device)
        # Blocked for the same reason as the gather: ``.contiguous()`` on a
        # strided half of the payload materialises that half in full, so an
        # unblocked scatter holds a live-set-sized transient at the widest
        # point of the move.
        block = _gather_block_rows()
        for lo in range(0, n, block):
            hi = min(lo + block, n)
            m = hi - lo
            sub = idx[lo:hi]
            k_part = data[lo:hi, :k_bytes].contiguous().to(k.device)
            v_part = data[lo:hi, k_bytes:].contiguous().to(v.device)
            k[sub] = k_part.view(k.dtype).view((m,) + tuple(k.shape[1:]))
            v[sub] = v_part.view(v.dtype).view((m,) + tuple(v.shape[1:]))


def _checksum(payload: torch.Tensor) -> torch.Tensor:
    """8-byte trailer: int64 sum of the uint8 payload (packing-order pin)."""
    # Bounded-transient checksum (the weights_arena host-OOM family,
    # 2026-08-08): KV payloads are GB-scale, a converted copy is 8x.
    total = uint8_checksum(payload)
    return torch.tensor([total], dtype=torch.int64).view(torch.uint8).to(payload.device)


class KvReshardRuntime:
    """Drives one group's phase-boundary KV resharding (#297).

    Injectables mirror the #287 runtime so the hermetic tests can drive REAL
    threads through mock channels: ``collective_min`` is the consensus
    channel (packed int payload -> element-wise MIN across the group);
    ``exchange`` is the pairwise byte channel
    (``outgoing: {peer: uint8 1-D}``, ``incoming_nbytes: {peer: int}`` ->
    ``{peer: uint8 1-D}``); ``pool_view`` is a :class:`KvPoolView`;
    ``live_slots_fn`` enumerates the replicated live global slot ids;
    ``ready_fn`` is the fully-idle predicate; ``cutover_fn`` installs the
    new vector into every snapshot cache (DESIGN_297 section 5).
    """

    def __init__(
        self,
        *,
        dcp_size: int,
        dcp_rank: int,
        allowed_vectors: Sequence[Sequence[int]],
        current_vector: Sequence[int],
        consensus_interval: int = 8,
        collective_min: Optional[Callable[[List[int]], List[int]]] = None,
        exchange: Optional[
            Callable[[Dict[int, torch.Tensor], Dict[int, int]], Dict[int, torch.Tensor]]
        ] = None,
        pool_view: Optional[KvPoolView] = None,
        live_slots_fn: Optional[Callable[[], torch.Tensor]] = None,
        ready_fn: Optional[Callable[[], bool]] = None,
        cutover_fn: Optional[Callable[[Tuple[int, ...]], None]] = None,
        guards: Sequence[str] = (),
        clock: Callable[[], float] = time.perf_counter,
        fit_check: Optional[Callable[[Tuple[int, ...]], Tuple[bool, str]]] = None,
        free_bytes_fn: Optional[Callable[[], int]] = None,
        corridor_floor_bytes: int = CORRIDOR_FLOOR_MIB * 1024 * 1024,
    ):
        if dcp_size < 2:
            raise KvReshardError(
                f"KV resharding needs a multi-rank DCP group, got dcp_size={dcp_size}"
            )
        if consensus_interval < 1:
            raise ValueError(
                f"consensus_interval must be >= 1, got {consensus_interval}"
            )
        if collective_min is None or exchange is None:
            raise KvReshardError(
                "a multi-rank reshard needs both a consensus channel "
                "(collective_min) and a pairwise byte channel (exchange); "
                "running without them would turn the first honest divergence "
                "into a hang instead of a loud error."
            )
        if (
            pool_view is None
            or live_slots_fn is None
            or ready_fn is None
            or cutover_fn is None
        ):
            missing = [
                name
                for fn, name in (
                    (pool_view, "pool_view"),
                    (live_slots_fn, "live_slots_fn"),
                    (ready_fn, "ready_fn"),
                    (cutover_fn, "cutover_fn"),
                )
                if fn is None
            ]
            raise KvReshardError(f"KvReshardRuntime needs {', '.join(missing)}")
        self._size = int(dcp_size)
        self._rank = int(dcp_rank)
        self._allowed = [tuple(int(x) for x in v) for v in allowed_vectors]
        for vec in self._allowed:
            if len(vec) != self._size:
                raise KvReshardError(
                    f"allowed vector {vec} has {len(vec)} entries but "
                    f"dcp_size is {self._size}"
                )
        self._current = tuple(int(x) for x in current_vector)
        if self._current not in self._allowed:
            raise KvReshardError(
                f"boot vector {self._current} is not in the declared ceiling "
                f"set {self._allowed} -- the pool would be undersized for a "
                f"reshard back to boot"
            )
        self._interval = int(consensus_interval)
        self._collective_min = collective_min
        self._exchange = exchange
        self._pool = pool_view
        self._live_slots_fn = live_slots_fn
        self._ready_fn = ready_fn
        self._cutover_fn = cutover_fn
        # #330: after a capacity grow, the boot fitted-ceiling invariant no
        # longer covers every declared vector -- a reshard target that would
        # need more rows than are physically backed must HOLD (uniformly,
        # via ready=0) instead of failing the bounds check mid-move. The
        # callback is a pure function of replicated capacity state, so every
        # rank computes the same verdict. None = stock behavior.
        self._fit_check = fit_check
        # #363: `fit_check` answers "does the TARGET VECTOR fit the backed
        # pool" and it passed on both runs that then died. What nothing
        # checked was the TRANSIENT the move allocates on top of the resident
        # pool -- the staged reads, the packed payloads and the receive
        # buffers. On metal that was a 616 MiB allocation against 550 MiB free
        # (under load) and 758 against 256 (drained), and it broke the
        # corridor SEVEN SECONDS BEFORE the OOM in both runs: the allocation
        # drove free memory under the floor, the floor did not drift into the
        # allocation. `free_bytes_fn` reports this rank's NVML FREE bytes;
        # None leaves the guard inert, which only a test fleet should do.
        self._free_bytes_fn = free_bytes_fn
        self._corridor_floor_bytes = int(corridor_floor_bytes)
        #: names of features that block Stage-A resharding for this process
        #: (evaluated once at construction; arming refuses while non-empty).
        self.blocking_guards = tuple(guards)
        self._clock = clock

        self._round = 0
        self._epoch = 0
        self._pending: Optional[Tuple[int, ...]] = None
        self._last_hold_reason: Optional[str] = None
        #: #363 last mile: when the pending arm was taken, and when its hold was
        #: last reported. A hold is logged once per CHANGE of reason, which is
        #: right for a reason that changes and wrong for one that does not: the
        #: #656 audit established that ``_execute`` needs a fully-idle round and
        #: that "under continuous load a fully idle round may never arrive", so
        #: the one hold reason that can persist forever was also the one that
        #: printed exactly once and then went quiet. These two carry the AGE, so
        #: a decision that never became a move is visible as such.
        self._pending_since: Optional[float] = None
        self._last_hold_report_at: Optional[float] = None
        self.desync_checks = 0
        self.completed = 0
        self.last_stats: Optional[dict] = None
        #: #363 observability: how many armed moves the group refused for
        #: headroom, and the arithmetic behind the last local verdict. The
        #: window records both; a refusal with no numbers is not a finding.
        self.refused_headroom = 0
        self.last_headroom: Optional[dict] = None

        logger.info(
            "%s armed at boot: rank %d/%d, vector %s, ceiling set %s, "
            "consensus every %d rounds%s",
            LOG_PREFIX,
            self._rank,
            self._size,
            self._current,
            self._allowed,
            self._interval,
            (
                "; Stage-A guards BLOCKING arming: " + ", ".join(self.blocking_guards)
                if self.blocking_guards
                else ""
            ),
        )

    # -- state ---------------------------------------------------------------
    @property
    def allowed_vectors(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(self._allowed)

    @property
    def current_vector(self) -> Tuple[int, ...]:
        return self._current

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def pending(self) -> Optional[Tuple[int, ...]]:
        return self._pending

    # -- arming (replicated callers: ladder flip or broadcast RPC) -----------
    def arm(self, target: Sequence[int], source: str) -> Tuple[bool, str]:
        """Arm a reshard to ``target``. Replicated call; the consensus round
        commits it once every rank is armed AND ready. Returns (ok, msg)."""
        vec = tuple(int(x) for x in target)
        if self.blocking_guards:
            msg = (
                f"KV reshard refused (Stage A guards): "
                f"{', '.join(self.blocking_guards)} -- these features cache "
                f"or encode owner state the Stage-A move does not migrate"
            )
            logger.warning("%s %s", LOG_PREFIX, msg)
            return False, msg
        if len(vec) != self._size or any(v < 1 for v in vec):
            return False, (
                f"invalid target vector {vec}: needs {self._size} entries >= 1"
            )
        if vec == self._current:
            return False, f"target {vec} is already the current vector"
        if vec not in self._allowed:
            return False, (
                f"target {vec} is not in the declared ceiling set "
                f"{self._allowed}; the pool has no reserved rows for it "
                f"(add it to --kv-reshard-vectors and reboot)"
            )
        if self._pending is not None and self._pending != vec:
            logger.warning(
                "%s re-arming %s -> %s (source %s) replaces the pending target",
                LOG_PREFIX,
                self._pending,
                vec,
                source,
            )
        # The age belongs to the TARGET, not to the call: re-arming the same
        # vector must not restart the clock, or a controller that re-proposes
        # every boundary would keep a permanently stuck arm looking fresh.
        if self._pending != vec or self._pending_since is None:
            self._pending_since = self._clock()
            self._last_hold_report_at = None
        self._pending = vec
        msg = (
            f"reshard armed: {self._current} -> {vec} (source {source}); "
            f"commits at the next consensus boundary where every rank is "
            f"idle"
        )
        logger.warning("%s %s", LOG_PREFIX, msg)
        return True, msg

    # -- the per-round hook ---------------------------------------------------
    def on_round(self) -> Optional[dict]:
        """One scheduler round. Cadence-gated by the REPLICATED round counter
        (never by local state); at a boundary every rank enters ONE bounded
        MIN-reduction, armed or not. Returns move stats when a reshard
        executed this round, else ``None``."""
        self._round += 1
        if self._round % self._interval != 0:
            return None
        armed = 1 if self._pending is not None else 0
        fit_ok, fit_msg = True, ""
        if armed and self._fit_check is not None:
            fit_ok, fit_msg = self._fit_check(self._pending)
        local_ready = bool(armed and fit_ok and self._ready_fn())
        ready = 1 if local_ready else 0
        # #363 headroom. Evaluated ONLY by a rank that would otherwise move,
        # because it costs a live-slot read and a transition build, and
        # because a rank that is not ready has nothing to price yet. A rank
        # that did not evaluate reports the NEUTRAL 1: it cannot veto on a
        # number it never computed. That is safe because the refusal below is
        # only ever consulted once min(ready) == 1 -- i.e. once EVERY rank was
        # ready, and therefore every rank did evaluate.
        headroom = 1
        plan: Optional["_MovePlan"] = None
        hr_local: Optional[dict] = None
        if local_ready:
            plan = self._plan_move()
            headroom, hr_local = self._headroom_check(plan)
        vec = self._pending if self._pending is not None else (0,) * self._size
        payload = _encode([armed, ready, headroom, self._epoch, *vec])
        self.desync_checks += 1
        reduced = self._collective_min(payload)
        if len(reduced) != len(payload):
            raise KvReshardError(
                f"consensus channel returned {len(reduced)} values for a "
                f"{len(payload)}-value payload; the channel contract is "
                f"element-wise MIN of the packed proposal."
            )
        fields = (
            list(_MIN_FIELDS)
            + list(_EQ_FIELDS)
            + [f"vector[{i}]" for i in range(self._size)]
        )
        lo = {f: reduced[2 * i] for i, f in enumerate(fields)}
        hi = {f: -reduced[2 * i + 1] for i, f in enumerate(fields)}

        # Equality family: epoch always; vector once every rank is armed.
        eq_checked = list(_EQ_FIELDS)
        if lo["armed"] == 1:
            eq_checked += [f"vector[{i}]" for i in range(self._size)]
        mismatches = [
            f"{f}: min={lo[f]} max={hi[f]}" for f in eq_checked if lo[f] != hi[f]
        ]
        if mismatches:
            raise KvReshardError(
                f"{LOG_PREFIX} DESYNC at round {self._round}: the ranks "
                f"disagree on the reshard state ({'; '.join(mismatches)}; "
                f"this rank: armed={armed} target={vec} epoch={self._epoch}). "
                f"A reshard that disagrees across ranks must fail loudly "
                f"HERE, before any rank moves a byte under the wrong vector "
                f"-- continuing would be the #94/#194/#259 hang, or worse, a "
                f"silent mixed-ownership pool."
            )
        # MIN family: wait uniformly (every rank sees the same reduction).
        if lo["armed"] == 0:
            if hi["armed"] == 1:
                self._hold("waiting for every rank to arm (delivery skew)")
            return None
        if lo["ready"] == 0:
            if not fit_ok:
                self._hold(
                    f"armed, but the target does not fit the backed pool: {fit_msg}"
                )
            else:
                self._hold(
                    "armed, waiting for a group-wide idle boundary "
                    f"(this rank ready={ready})"
                )
            return None
        # Every rank was ready, so every rank priced the move. If ANY of them
        # cannot afford it, ALL of them refuse -- here, before a single rank
        # has allocated a byte. A guard at the allocation site instead would
        # let the affording ranks proceed into the exchange while the short
        # one aborted, which is the #94/#194/#259 hang the desync check above
        # exists to prevent: the same reduction has to carry it.
        if lo["headroom"] == 0:
            self._refuse_for_headroom(hr_local)
            return None
        self._last_hold_reason = None
        return self._execute(plan)

    def _hold(self, reason: str) -> None:
        """Report why the armed move did not run this boundary.

        Deduplicated by reason, as before -- a boundary-rate log would bury
        itself. What is new is that dedup no longer means SILENCE: an arm whose
        reason does not change is exactly the case the #656 audit named, so the
        hold is re-reported on an interval with the arm's AGE, and escalates to
        WARNING once the wait stops being plausibly transient.

        The age is the only number here that is not already in the reason
        string, and it is the one that separates "waiting for the next idle
        round" from "this reshard is never going to run".
        """
        now = self._clock()
        changed = reason != self._last_hold_reason
        last = self._last_hold_report_at
        due = last is None or (now - last) >= HOLD_REPORT_INTERVAL_S
        if not changed and not due:
            return

        self._last_hold_reason = reason
        self._last_hold_report_at = now
        if self._pending_since is None:
            # No arm to age (a hold without a pending vector); keep the old
            # shape rather than inventing an age of zero.
            logger.info("%s hold: %s", LOG_PREFIX, reason)
            return

        age_s = now - self._pending_since
        if age_s >= HOLD_ESCALATE_AFTER_S:
            logger.warning(
                "%s (%.0fs) -> %s, held by: %s. The move is ARMED and has not "
                "run: _execute needs a fully-idle round, and under continuous "
                "load one may never arrive (#656). The controller counts this "
                "stage as acted on. Disarm it or accept that the vector will "
                "not change.",
                HOLD_STUCK_MARKER,
                age_s,
                list(self._pending),
                reason,
            )
        else:
            logger.info(
                "%s hold (%.0fs, target %s): %s",
                LOG_PREFIX,
                age_s,
                list(self._pending),
                reason,
            )

    # -- #363 headroom -------------------------------------------------------
    def _plan_move(self) -> "_MovePlan":
        """The move's transition and its transient byte terms. READS ONLY.

        Built once per boundary and handed to :meth:`_execute`, so the bytes
        the guard prices are the bytes the move then allocates. Recomputing
        them in ``_execute`` would reopen a window in which the live set moved
        between the verdict and the allocation -- a guard that prices a
        different move than the one it admits is not a guard.
        """
        target = self._pending
        assert target is not None
        t0 = self._clock()
        slots = self._live_slots_fn()
        slots = torch.unique(slots.detach().to("cpu", torch.int64))
        tr = build_transition(slots, self._current, target, self._rank)
        return _MovePlan(t0=t0, slots=slots, tr=tr, terms=self._transient_terms(tr))

    def _transient_terms(self, tr) -> dict:
        """Every device allocation ``_execute`` makes ON TOP of the pool.

        Read straight off the PACK and EXCHANGE phases below, in their order,
        and named individually so a refusal can show its work:

        ``staged``  ``out_parts`` -- one ``read_rows`` result per (layer,
                    peer). Held for the whole pack loop, never released early.
        ``packed``  ``outgoing_payloads`` -- the ``torch.cat`` copy of the
                    above plus one checksum per peer. Alive at the same time
                    as ``staged``, which is why both terms count.
        ``largest`` ``flat``, the biggest single per-peer ``cat`` transient
                    inside the loop.
        ``recv``    the ``torch.empty`` receive buffers -- the allocation that
                    actually raised on metal.

        The pool itself is NOT counted: it is already resident.
        """
        row_bytes = self._pool.total_row_nbytes
        out_per_peer = {
            p: int(rows.numel()) * row_bytes for p, rows in tr.outgoing_rows.items()
        }
        in_per_peer = {
            p: int(rows.numel()) * row_bytes for p, rows in tr.incoming_rows.items()
        }
        staged = sum(out_per_peer.values())
        packed = staged + _CHECKSUM_BYTES * len(out_per_peer)
        largest = max(out_per_peer.values(), default=0)
        recv = sum(in_per_peer.values()) + _CHECKSUM_BYTES * len(in_per_peer)
        return {
            "staged": staged,
            "packed": packed,
            "largest_peer_pack": largest,
            "recv": recv,
            "peak": staged + packed + largest + recv,
        }

    def _headroom_check(self, plan: "_MovePlan") -> Tuple[int, Optional[dict]]:
        """Can THIS rank afford the move without breaking the corridor?

        ``free - corridor_floor - peak >= 0``. The floor is subtracted before
        the move is priced, not after: the corridor is the user's reserve, so
        it was never memory the reshard could spend. Returns ``(0|1, detail)``
        and NEVER acts -- the caller folds the verdict into the group-wide MIN
        and only the group refuses.
        """
        if self._free_bytes_fn is None:
            return 1, None
        try:
            free = int(self._free_bytes_fn())
        except Exception as e:  # noqa: BLE001
            # A guard that cannot read free memory must not pretend the move
            # is affordable. Refusing costs a flip; guessing cost the server.
            detail = {
                "rank": self._rank,
                "error": f"{type(e).__name__}: {e}",
                "peak_bytes": plan.terms["peak"],
                "corridor_floor_bytes": self._corridor_floor_bytes,
            }
            self.last_headroom = detail
            return 0, detail
        peak = int(plan.terms["peak"])
        margin = free - self._corridor_floor_bytes - peak
        detail = {
            "rank": self._rank,
            "free_bytes": free,
            "corridor_floor_bytes": self._corridor_floor_bytes,
            "peak_bytes": peak,
            "margin_bytes": margin,
            "terms": dict(plan.terms),
            "ok": margin >= 0,
        }
        self.last_headroom = detail
        return (1 if margin >= 0 else 0), detail

    def _refuse_for_headroom(self, hr_local: Optional[dict]) -> None:
        """The group refused. Disarm, name it, stay on the incumbent layout.

        Disarm rather than hold: the metal reproduction was WORSE on a drained
        pool (758 MiB needed against 256 free) than under load (616 against
        550), so this is not a transient a silent retry loop grows out of. The
        controller may arm again when something has actually changed; an
        unbounded hold would just be the same refusal with no record.
        """
        self._pending = None
        self._pending_since = None
        self._last_hold_report_at = None
        self.refused_headroom += 1
        self._last_hold_reason = None
        d = hr_local or {}
        if "error" in d:
            detail = f"this rank could not read free memory ({d['error']})"
        elif d:
            t = d.get("terms") or {}
            detail = (
                f"this rank: {_fmt_bytes(d['free_bytes'])} free, "
                f"{_fmt_bytes(d['corridor_floor_bytes'])} corridor floor, "
                f"{_fmt_bytes(d['peak_bytes'])} transient needed "
                f"(staged {_fmt_bytes(t.get('staged', 0))} + packed "
                f"{_fmt_bytes(t.get('packed', 0))} + pack-peak "
                f"{_fmt_bytes(t.get('largest_peer_pack', 0))} + recv "
                f"{_fmt_bytes(t.get('recv', 0))}) -> margin "
                f"{_fmt_bytes(d['margin_bytes'])}"
            )
        else:
            detail = "this rank had headroom; another rank in the group did not"
        logger.warning(
            "%s REFUSED for headroom at round %d: the move stays on the "
            "incumbent vector %s. At least one rank cannot allocate the "
            "move's transient buffers without taking free memory below the "
            "%d MiB corridor floor, so NO rank allocates. %s",
            LOG_PREFIX,
            self._round,
            self._current,
            self._corridor_floor_bytes // (1024 * 1024),
            detail,
        )

    # -- the move -------------------------------------------------------------
    def _execute(self, plan: Optional["_MovePlan"] = None) -> dict:
        """Copy -> exchange -> scatter -> cutover, synchronously, on every
        rank of the group in the same round. See module docstring for the
        aliasing and atomicity arguments.

        ``plan`` is the transition the headroom guard priced this round
        (#363). It is reused rather than rebuilt so the admitted move is the
        priced move; ``None`` rebuilds it, which is the unguarded path."""
        target = self._pending
        assert target is not None
        if plan is None:
            plan = self._plan_move()
        t0, tr = plan.t0, plan.tr

        # Bounds check BEFORE any write: the fitted ceiling must hold.
        max_row = tr.max_new_row()
        if max_row >= self._pool.num_rows:
            raise KvReshardError(
                f"{LOG_PREFIX} target {target} needs row {max_row} but the "
                f"pool holds {self._pool.num_rows} rows -- the fitted "
                f"ceiling does not cover this vector (sizing bug: the "
                f"ceiling set and the allowed set must be the same)."
            )

        layers = range(self._pool.num_layers)
        row_bytes = self._pool.total_row_nbytes

        # PACK phase (reads only, pool untouched): stage every outgoing row
        # into per-peer payloads -- [n, total_row_bytes] flattened plus an
        # 8-byte checksum trailer pinning the packing-order convention (both
        # ends derive the order from the same sorted slot set; the checksum
        # keeps that contract falsifiable at runtime).
        t_read0 = self._clock()
        out_parts: Dict[int, List[torch.Tensor]] = {p: [] for p in tr.outgoing_rows}
        for layer in layers:
            for peer, rows in tr.outgoing_rows.items():
                out_parts[peer].append(self._pool.read_rows(layer, rows))
        outgoing_payloads: Dict[int, torch.Tensor] = {}
        for peer, parts in out_parts.items():
            flat = torch.cat(parts, dim=1).reshape(-1)
            outgoing_payloads[peer] = torch.cat([flat, _checksum(flat)])
        read_ms = (self._clock() - t_read0) * 1000.0

        # EXCHANGE (pool still untouched): a failure up to and including this
        # point aborts the attempt with the pool byte-identical -- a later
        # boundary may retry. Only the WRITE phase below is the
        # no-return-on-error region.
        incoming_nbytes = {
            peer: int(rows.numel()) * row_bytes + _CHECKSUM_BYTES
            for peer, rows in tr.incoming_rows.items()
        }
        t_xfer0 = self._clock()
        received = self._exchange(outgoing_payloads, incoming_nbytes)
        xfer_ms = (self._clock() - t_xfer0) * 1000.0
        incoming_data: Dict[int, torch.Tensor] = {}
        for peer, rows in tr.incoming_rows.items():
            payload = received.get(peer)
            if payload is None or payload.numel() != incoming_nbytes[peer]:
                got = 0 if payload is None else payload.numel()
                raise KvReshardError(
                    f"{LOG_PREFIX} exchange returned {got} bytes from peer "
                    f"{peer}, expected {incoming_nbytes[peer]}"
                )
            data = payload[:-_CHECKSUM_BYTES]
            # clone(): a tail slice's storage offset is not 8-byte aligned in
            # general, and dtype-view requires alignment.
            want = int(payload[-_CHECKSUM_BYTES:].clone().view(torch.int64).item())
            have = uint8_checksum(data)
            if want != have:
                raise KvReshardError(
                    f"{LOG_PREFIX} payload checksum mismatch from peer "
                    f"{peer}: sender {want}, receiver {have} -- the packing "
                    f"order contract broke; refusing to scatter."
                )
            incoming_data[peer] = data.view(int(rows.numel()), row_bytes)

        # WRITE phase, layer by layer: gather this layer's retained rows
        # into a temp FIRST (read), then scatter retained and received rows
        # into their new rows. Reads of a layer strictly precede its writes,
        # which makes the old/new row overlap inside one buffer harmless;
        # retained and incoming target rows are disjoint (the row map is
        # injective). Every read of OTHER layers happened in the pack phase
        # or lands in other buffers.
        t_write0 = self._clock()
        for layer in layers:
            if tr.retained_old_rows.numel():
                tmp = self._pool.read_rows(layer, tr.retained_old_rows)
                self._pool.write_rows(layer, tr.retained_new_rows, tmp)
            col_lo = sum(self._pool.row_nbytes(lay) for lay in range(layer))
            col_hi = col_lo + self._pool.row_nbytes(layer)
            for peer, rows in tr.incoming_rows.items():
                self._pool.write_rows(
                    layer, rows, incoming_data[peer][:, col_lo:col_hi]
                )
        write_ms = (self._clock() - t_write0) * 1000.0

        # CUTOVER: install the new vector into the global and every snapshot
        # cache (DESIGN_297 section 5), then account.
        old = self._current
        self._cutover_fn(target)
        self._current = target
        self._pending = None
        self._pending_since = None
        self._last_hold_report_at = None
        self._epoch += 1
        self.completed += 1
        total_ms = (self._clock() - t0) * 1000.0
        moved_out = sum(int(t.numel()) for t in outgoing_payloads.values())
        moved_in = sum(incoming_nbytes.values())
        stats = {
            "old_vector": old,
            "new_vector": target,
            "epoch": self._epoch,
            "live_slots": tr.total_slots,
            "retained_moved_rows": tr.retained_moving,
            "sent_rows": tr.outgoing_slots,
            "received_rows": tr.incoming_slots,
            "sent_bytes": moved_out,
            "received_bytes": moved_in,
            "read_ms": read_ms,
            "exchange_ms": xfer_ms,
            "write_ms": write_ms,
            "total_ms": total_ms,
        }
        self.last_stats = stats
        logger.warning(
            "%s DONE %s -> %s (epoch %d) in %.1f ms: %d live slots, "
            "%d local row moves, sent %d rows / %.2f MiB, received %d rows "
            "/ %.2f MiB (read %.1f ms, exchange %.1f ms, write %.1f ms)",
            LOG_PREFIX,
            old,
            target,
            self._epoch,
            total_ms,
            tr.total_slots,
            tr.retained_moving,
            tr.outgoing_slots,
            moved_out / 1048576.0,
            tr.incoming_slots,
            moved_in / 1048576.0,
            read_ms,
            xfer_ms,
            write_ms,
        )
        return stats


# ---------------------------------------------------------------------------
# Production wiring (scheduler side).
# ---------------------------------------------------------------------------


def _dist_exchange(device_group, device):
    """Pairwise byte channel over the TP DEVICE (NCCL) group, bounded via
    #312.

    One ``batch_isend_irecv`` batch (recvs in ascending peer order, then
    sends -- the same deterministic order on every rank), then every work is
    polled through ``bounded_collective``: a dead peer is a loud
    ``PeerLostError``/``CollectiveTimeoutError`` within the probe interval,
    never a hang. Device-native on purpose: NCCL p2p works implement
    ``is_completed`` truthfully (gloo's SendWork/RecvWork do not -- module
    docstring), and the payloads never leave the GPUs.
    """
    import torch.distributed as dist

    from sglang.srt.distributed.device_communicators.barlink_liveness import (
        bounded_collective,
    )

    def _exchange(
        outgoing: Dict[int, torch.Tensor], incoming_nbytes: Dict[int, int]
    ) -> Dict[int, torch.Tensor]:
        # #802: AGREE THE SIZES BEFORE MOVING A SINGLE BYTE.
        #
        # THE HOLE THIS CLOSES, measured 2026-08-22 17:22:44. `incoming_nbytes`
        # is a PREDICTION: the caller derives it locally (gdn_flip_mover.move's
        # `_pair_nbytes(..., len(slots), ...)`) from its OWN rank-local slot
        # enumeration, while the sender packs its payload from ITS OWN. For the
        # GDN leg that enumeration is `flip_mamba_slots` -- "resident requests'
        # mamba slots UNION the radix tree's checkpoints" -- which has no
        # cross-rank agreement step at all, unlike the KV leg's
        # `build_flip_live_slots_fn`. The three ranks demonstrably disagreed
        # that day (150695 / 159848 / 151656 enumerated rows).
        #
        # Nothing downstream could catch it. `torch.empty` below is never
        # zeroed; NCCL p2p with mismatched counts does not raise; the buffer
        # keeps its ALLOCATED numel, so the receiver's length check passes; and
        # `bounded_collective` polls `is_completed()`, never a byte count. The
        # short receive therefore leaves the original allocator garbage exactly
        # where the payload's checksum trailer is read from, and the flip died
        # reporting a data corruption that had not happened.
        #
        # A zero-filled matrix summed across the group is a full all-to-all of
        # the advertised sizes in one symmetric collective: every rank writes
        # only its own row, so the sum is the truth and no rank can shadow
        # another's entry. It runs UNCONDITIONALLY and BEFORE the `not ops`
        # early return -- a rank that skipped it while its peers did not would
        # be the very desynchronisation this is here to prevent.
        # THE REFUSAL MUST BE COLLECTIVE, and the first cut of this guard was
        # not -- a mistake its own three-process test caught within a minute.
        # Checking only "what MY peers advertised versus what I expected" makes
        # the verdict rank-local: in a three-rank group where rank 1 packs a
        # short payload, ranks 0 and 2 see the disagreement and raise while
        # rank 1 sees nothing wrong, walks into `batch_isend_irecv` and blocks
        # for ever on peers that have already left. A guard that strands the
        # group is worse than the corruption it prevents.
        #
        # So BOTH halves are gathered -- what each rank will SEND to each peer,
        # and what each rank has SIZED ITS RECEIVE for -- and every rank then
        # checks every ordered pair against the identical matrix and reaches
        # the identical verdict. Zero-filled and summed: each rank writes only
        # its own row, so the sum is the truth and no rank can shadow another.
        n_ranks = dist.get_world_size(device_group)
        me = dist.get_rank(device_group)
        agreed = torch.zeros((2, n_ranks, n_ranks), dtype=torch.int64, device=device)
        for peer, buf in outgoing.items():
            agreed[0][me][int(peer)] = int(buf.numel())
        for peer, want in incoming_nbytes.items():
            agreed[1][me][int(peer)] = int(want)
        dist.all_reduce(agreed, group=device_group)
        for sender in range(n_ranks):
            for receiver in range(n_ranks):
                if sender == receiver:
                    continue
                sent = int(agreed[0][sender][receiver].item())
                want = int(agreed[1][receiver][sender].item())
                if sent == want:
                    continue
                raise KvReshardError(
                    f"kv_reshard p2p SIZE DISAGREEMENT on the {sender}->"
                    f"{receiver} pair: rank {sender} is sending {sent} bytes, "
                    f"rank {receiver} sized its receive for {want}. The two "
                    "ends derived the byte count independently from their own "
                    "state and never agreed it, so one of the enumerations "
                    "they are built on has diverged (for the GDN leg that is "
                    "flip_mamba_slots, which is rank-local and has no union "
                    "step, unlike the KV leg's build_flip_live_slots_fn). "
                    "REFUSED BEFORE ANY TRANSFER: nothing was received, so no "
                    "buffer holds a partly-written payload and no checksum "
                    "trailer is read out of allocator garbage. Raised on EVERY "
                    "rank off one shared matrix, so no peer is left in the "
                    "exchange. This is a state divergence, NOT a data "
                    "corruption (#802)."
                )
        received: Dict[int, torch.Tensor] = {}
        ops = []
        for peer in sorted(incoming_nbytes):
            buf = torch.empty(
                int(incoming_nbytes[peer]), dtype=torch.uint8, device=device
            )
            received[peer] = buf
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    buf,
                    peer=dist.get_global_rank(device_group, peer),
                    group=device_group,
                )
            )
        for peer in sorted(outgoing):
            ops.append(
                dist.P2POp(
                    dist.isend,
                    outgoing[peer].contiguous().to(device),
                    peer=dist.get_global_rank(device_group, peer),
                    group=device_group,
                )
            )
        if not ops:
            return received
        works = dist.batch_isend_irecv(ops)
        for i, work in enumerate(works):
            bounded_collective(lambda w=work: w, f"kv_reshard.p2p[{i}]")
        # The works signal kernel completion on the p2p streams; the write
        # phase consumes the buffers on the default stream, so order the two.
        if torch.cuda.is_available() and received:
            torch.cuda.synchronize(device)
        return received

    return _exchange


def _free_bytes_fn_for(device) -> Callable[[], int]:
    """Free bytes on THIS rank's card, read the way the corridor is judged.

    NVML FREE, not total-minus-used: the ~424/518 MiB carve-out is invisible
    to the subtraction and the corridor law is written against the FREE
    column, so pricing a move against the wrong quantity would let the guard
    pass a move the corridor sampler then fails.

    The card is resolved by UUID, never by index. A scheduler rank runs under
    a narrowed ``CUDA_VISIBLE_DEVICES``, so its torch device 0 is some other
    NVML index entirely -- the same divergence that made the card probe
    unreachable (#363 defect 3). A UUID survives the narrowing unchanged.

    Falls back to ``torch.cuda.mem_get_info`` when NVML is unavailable, and
    raises otherwise: :meth:`KvReshardRuntime._headroom_check` treats a raise
    as a refusal, which is the safe direction.
    """

    def _free_bytes() -> int:
        import pynvml  # noqa: PLC0415  (optional dependency)

        want = str(torch.cuda.get_device_properties(device).uuid)
        pynvml.nvmlInit()
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                uuid = pynvml.nvmlDeviceGetUUID(h)
                if isinstance(uuid, bytes):
                    uuid = uuid.decode()
                if str(uuid).replace("GPU-", "") == want.replace("GPU-", ""):
                    return int(pynvml.nvmlDeviceGetMemoryInfo(h).free)
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:  # noqa: BLE001
                pass
        raise RuntimeError(
            f"no NVML device carries this rank's card UUID {want}; free "
            f"memory cannot be attributed to the card the move allocates on"
        )

    def _free_bytes_with_fallback() -> int:
        try:
            return _free_bytes()
        except ImportError:
            return int(torch.cuda.mem_get_info(device)[0])

    return _free_bytes_with_fallback


def _stage_a_guards(scheduler) -> List[str]:
    """Features whose owner-state encodings the Stage-A move does not
    migrate (DESIGN_297 section 2). Arming refuses while any is active."""
    from sglang.srt.disaggregation.utils import DisaggregationMode

    guards: List[str] = []
    if scheduler.disaggregation_mode != DisaggregationMode.NULL:
        guards.append("PD disaggregation")
    if getattr(scheduler, "enable_hierarchical_cache", False):
        guards.append("hierarchical cache")
    if getattr(scheduler, "kv_session_offload", None) is not None:
        guards.append("kv-session-offload (host pool rows sized from the boot vector)")
    try:
        from sglang.srt.distributed.utils import weightless_kv_active

        if weightless_kv_active():
            guards.append("weightless-KV ranks")
    except ImportError:
        pass
    if getattr(scheduler, "is_dual_group_lane", False) or getattr(
        scheduler.server_args, "dual_group_lane", None
    ):
        guards.append("dual-group lane")
    if not hasattr(scheduler.tree_cache, "all_values_flatten"):
        guards.append(
            f"tree cache {type(scheduler.tree_cache).__name__} (no "
            f"all_values_flatten enumeration)"
        )
    return guards


def _cutover_fn_for(scheduler) -> Callable[[Tuple[int, ...]], None]:
    def _cutover(new_vector: Tuple[int, ...]) -> None:
        from sglang.srt.distributed.utils import set_cp_token_ratios
        from sglang.srt.layers.dcp.owner import refresh_all_owner_bounds

        set_cp_token_ratios(list(new_vector))
        refreshed = refresh_all_owner_bounds()
        # HiCache's controller memoizes the owner ctx; Stage A refuses to arm
        # with hicache active, but a stale memo must still not survive.
        controller = getattr(scheduler, "cache_controller", None)
        if controller is not None and hasattr(controller, "_dcp_owner_ctx_cache"):
            delattr(controller, "_dcp_owner_ctx_cache")
        logger.info(
            "%s cutover: vector %s installed, %d owner-bounds consumers refreshed",
            LOG_PREFIX,
            list(new_vector),
            refreshed,
        )

    return _cutover


def build_kv_reshard_runtime(scheduler) -> KvReshardRuntime:
    """Construct the runtime for one scheduler rank, or raise loudly.

    Called lazily on the first scheduler iteration when
    ``--kv-reshard-vectors`` is set (mirrors the #287 lazy build: every
    dependency exists by then, no request has been admitted). Flag unset =
    never called = today's behavior, byte-identical.
    """
    from sglang.srt.distributed.utils import get_cp_token_ratios, uneven_dcp_active
    from sglang.srt.layers.dcp.reshard_plan import reshard_vector_set
    from sglang.srt.runtime_context import get_parallel
    from sglang.srt.managers.kv_pressure_runtime import default_collective_min
    from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool

    server_args = scheduler.server_args
    dcp_size = int(get_parallel().attn_dcp_size)
    if not uneven_dcp_active(dcp_size):
        raise KvReshardError(
            "--kv-reshard-vectors requires WEIGHTED uneven DCP (a non-uniform "
            "token vector installed via --rank-kv-ratio with "
            "SGLANG_UNEVEN_DCP=1 SGLANG_UNEVEN_DCP_WEIGHTED=1); without it "
            "there is no owner vector to reshard."
        )
    boot_vector = tuple(get_cp_token_ratios())
    vectors = reshard_vector_set(server_args.kv_reshard_vectors, dcp_size, boot_vector)
    pool = scheduler.token_to_kv_pool_allocator.get_kvcache()
    if not isinstance(pool, HybridLinearKVPool):
        raise KvReshardError(
            f"--kv-reshard-vectors Stage A supports the hybrid-linear KV "
            f"pool family only (token-sharded full-attention layers); got "
            f"{type(pool).__name__}. The SWA-hybrid and dense-DCP lanes are "
            f"named follow-ups."
        )
    full = pool.full_kv_pool
    # #330 dial lane: the VMM-backed pool's tensors span the reserved VA
    # upper bound; bound row checks by the physically-backed prefix instead.
    valid_rows_fn = (
        (lambda: int(full.size) + int(full.page_size))
        if getattr(pool, "post_capture_active", False)
        else None
    )
    view = KvPoolView(full.k_buffer, full.v_buffer, valid_rows_fn=valid_rows_fn)
    dcp_rank = int(get_parallel().attn_dcp_rank)
    tree_cache = scheduler.tree_cache

    def _live_slots() -> torch.Tensor:
        values = tree_cache.all_values_flatten()
        if values is None or values.numel() == 0:
            return torch.empty(0, dtype=torch.int64)
        return values

    def _fit_check(target_vec):
        # #330: consult the capacity runtime (when armed) about whether the
        # target vector's rows fit every rank's backed pool at the current
        # ceiling. Pure function of replicated state -> uniform verdict.
        cap_rt = getattr(scheduler, "kv_capacity_runtime", None)
        if cap_rt is None:
            return True, ""
        return cap_rt.reshard_fit_check(target_vec)

    return KvReshardRuntime(
        dcp_size=dcp_size,
        dcp_rank=dcp_rank,
        allowed_vectors=vectors,
        current_vector=boot_vector,
        consensus_interval=int(
            getattr(server_args, "kv_reshard_consensus_interval", 8)
        ),
        collective_min=default_collective_min(scheduler.tp_cpu_group),
        exchange=_dist_exchange(scheduler.tp_group.device_group, view.device),
        pool_view=view,
        live_slots_fn=_live_slots,
        ready_fn=lambda: scheduler.is_fully_idle(),
        cutover_fn=_cutover_fn_for(scheduler),
        guards=_stage_a_guards(scheduler),
        fit_check=_fit_check,
        # #363: without this the cutover allocates its transient buffers
        # blind, which is how it OOM'd twice on metal and took the TP group
        # down with it. Wired here, unconditionally, so the guard is a
        # property of the production path rather than of a launch flag.
        free_bytes_fn=_free_bytes_fn_for(view.device),
    )
