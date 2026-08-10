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

import logging
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from sglang.srt.layers.dcp.reshard_plan import KvReshardError, build_transition
from sglang.srt.model_executor.weights_arena import uint8_checksum

logger = logging.getLogger(__name__)

LOG_PREFIX = "KV-RESHARD"

#: Consensus payload fields, packed as (value, -value) pairs so ONE
#: MIN-reduction yields (min, -max) per field. ``armed`` and ``ready`` are
#: MIN-checked (divergence legal -> wait); ``epoch`` is equality-checked
#: always; the vector elements are equality-checked once min(armed) == 1.
_MIN_FIELDS = ("armed", "ready")
_EQ_FIELDS = ("epoch",)

_CHECKSUM_BYTES = 8


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

    def read_rows_into(
        self, layer: int, rows: torch.Tensor, out: torch.Tensor
    ) -> None:
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
        out[:, :k_bytes] = self._as_bytes(k[idx], k_bytes)
        out[:, k_bytes:] = self._as_bytes(v[idx], expect - k_bytes)

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
        k_part = data[:, :k_bytes].contiguous().to(k.device)
        v_part = data[:, k_bytes:].contiguous().to(v.device)
        k[idx] = k_part.view(k.dtype).view((n,) + tuple(k.shape[1:]))
        v[idx] = v_part.view(v.dtype).view((n,) + tuple(v.shape[1:]))


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
        fit_check: Optional[
            Callable[[Tuple[int, ...]], Tuple[bool, str]]
        ] = None,
    ):
        if dcp_size < 2:
            raise KvReshardError(
                f"KV resharding needs a multi-rank DCP group, got dcp_size="
                f"{dcp_size}"
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
        #: names of features that block Stage-A resharding for this process
        #: (evaluated once at construction; arming refuses while non-empty).
        self.blocking_guards = tuple(guards)
        self._clock = clock

        self._round = 0
        self._epoch = 0
        self._pending: Optional[Tuple[int, ...]] = None
        self._last_hold_reason: Optional[str] = None
        self.desync_checks = 0
        self.completed = 0
        self.last_stats: Optional[dict] = None

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
                f"invalid target vector {vec}: needs {self._size} entries " f">= 1"
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
                "%s re-arming %s -> %s (source %s) replaces the pending " "target",
                LOG_PREFIX,
                self._pending,
                vec,
                source,
            )
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
        ready = 1 if (armed and fit_ok and self._ready_fn()) else 0
        vec = self._pending if self._pending is not None else (0,) * self._size
        payload = _encode([armed, ready, self._epoch, *vec])
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
                    f"armed, but the target does not fit the backed pool: "
                    f"{fit_msg}"
                )
            else:
                self._hold(
                    "armed, waiting for a group-wide idle boundary "
                    f"(this rank ready={ready})"
                )
            return None
        self._last_hold_reason = None
        return self._execute()

    def _hold(self, reason: str) -> None:
        if reason != self._last_hold_reason:
            logger.info("%s hold: %s", LOG_PREFIX, reason)
            self._last_hold_reason = reason

    # -- the move -------------------------------------------------------------
    def _execute(self) -> dict:
        """Copy -> exchange -> scatter -> cutover, synchronously, on every
        rank of the group in the same round. See module docstring for the
        aliasing and atomicity arguments."""
        target = self._pending
        assert target is not None
        t0 = self._clock()
        slots = self._live_slots_fn()
        slots = torch.unique(slots.detach().to("cpu", torch.int64))
        tr = build_transition(slots, self._current, target, self._rank)

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
            "%s cutover: vector %s installed, %d owner-bounds consumers " "refreshed",
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
    )
