# SPDX-License-Identifier: Apache-2.0
"""#631 phase-flip KV mover: the LIVE SET is the corridor bound.

HANDOFF_664 section 13 traced the length-scaling VRAM transient that both
breaches the 1024 MiB corridor and produces the staging livelock to ONE
place: the mover in ``PhaseFlipRuntime._execute``. Not the prefill path --
every prefill-path candidate was excluded in code, the largest at 0.43 MiB.

The mover's payload is irreducible: rows that live in the source layout
must be carried to the destination layout, and the backing swap between
the last read and the first write means they cannot be streamed straight
through. What is NOT irreducible is holding each byte two or three times:

* ``parts`` (one tensor per layer) is still referenced when
  ``flat = torch.cat(parts)`` exists, and ``flat`` is still referenced
  while the checksum-appended copy is built -- three copies of one peer's
  payload at the peak;
* the outgoing payloads stay referenced for the whole rest of the move,
  long after the sends completed, so they are still resident while the
  local leg is read and the writes run.

This file measures the peak, it does not inspect the source. The probe
counts bytes of live tensor STORAGE inside the flipping thread via
``TorchDispatchMode``, so it sees exactly what the caching allocator would
see and it is blind to how the code is written. A rewrite that moves the
copies around without removing them still fails.

The bound is computed from the PLAN, never hardcoded: the mover has to
hold at most ``incoming + max(outgoing, local)`` because the outgoing
payloads are dead once the exchange returns and the local leg is not read
until then. A tolerance covers the per-layer gather (one layer's rows,
1/2L of a payload) and the harness itself.
"""

from __future__ import annotations

import threading
import unittest
import weakref

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves

from sglang.srt.layers.dcp.phase_flip_plan import (
    PP_TO_TP,
    TP_TO_PP,
    build_phase_flip_transition,
)
from sglang.srt.layers.dcp.reshard_plan import owner_of, rows_of
from sglang.srt.managers.kv_reshard import _CHECKSUM_BYTES, KvPoolView
from sglang.srt.managers.phase_flip_runtime import PHASE_PP, PhaseFlipRuntime
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=60, suite="base-a-test-cpu")

# Production-shaped row geometry: 8 heads x 64 dim x bf16 = 1024 B for K
# and the same for V, so row_nbytes == 2048 B per token per full-attention
# layer -- the figure HANDOFF_664 section 13 models the transient with.
MAP_625 = ((0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))
N_LAYERS = 16
VEC = (3, 2, 2)
HEADS, DIM = 8, 64
ROW_NBYTES = HEADS * DIM * 2 * 2
NUM_SLOTS = 1536
MIB = 1024 * 1024


class LiveStorageProbe(TorchDispatchMode):
    """High-water mark of live tensor storage bytes in THIS thread.

    Keyed on ``data_ptr`` with a weakref to the first tensor seen at that
    address, so a view is not counted twice (a view keeps its base alive
    through ``_base``) and an address the allocator recycles is counted
    again. Dead entries are purged BEFORE each op runs, which is what
    makes the high-water honest: a payload released just before the next
    allocation must not inflate the peak that allocation sets.

    The probe is deliberately not a torch memory-stats reader: those exist
    for CUDA only and this family's suite runs CPU-only
    (``CUDA_VISIBLE_DEVICES=99`` in ``scripts/run_631_flip_family.sh``).
    """

    def __init__(self, exclude=(), floor_nbytes: int = 64 * 1024):
        super().__init__()
        # Ignore small allocations: index tensors, scalars and barrier
        # bookkeeping are noise against MiB-scale payloads, and tracking
        # them costs more than it measures.
        self._floor = int(floor_nbytes)
        # The KV POOLS are not staging. An in-place op returns the tensor
        # it mutated, so ``write_rows``' ``k[idx] = ...`` hands the probe
        # the destination pool itself -- 32 buffers, 20.6 MiB, a fifth of
        # the reading in the first version of this test. They are resident
        # for the process lifetime and the corridor accounts for them at
        # boot; what this probe exists to measure is what the MOVE adds on
        # top. Excluded by storage address, captured before the window.
        self._excluded = set()
        for t in exclude:
            try:
                self._excluded.add(t.untyped_storage().data_ptr())
            except Exception:  # pragma: no cover
                pass
        self._entries: dict[int, tuple[weakref.ref, int]] = {}
        self.live = 0
        self.peak = 0
        self.peak_breakdown: list[int] = []

    def _purge(self) -> None:
        dead = [p for p, (ref, _) in self._entries.items() if ref() is None]
        for p in dead:
            self.live -= self._entries.pop(p)[1]

    def _note(self, t) -> None:
        if not isinstance(t, torch.Tensor):
            return
        try:
            storage = t.untyped_storage()
            nbytes = storage.nbytes()
            ptr = storage.data_ptr()
        except Exception:  # pragma: no cover - meta/sparse/fake tensors
            return
        if ptr == 0 or nbytes < self._floor or ptr in self._excluded:
            return
        seen = self._entries.get(ptr)
        if seen is not None:
            if seen[0]() is not None:
                return
            self.live -= seen[1]
        self._entries[ptr] = (weakref.ref(t), nbytes)
        self.live += nbytes
        if self.live > self.peak:
            self.peak = self.live
            # WHAT was resident at the high-water, largest first. A bare
            # peak number tells a successor that the mover holds too much
            # and nothing about which copy to remove; this chain has lost
            # whole sessions to that gap.
            self.peak_breakdown = sorted(
                (nb for _, nb in self._entries.values()), reverse=True
            )

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self._purge()
        out = func(*args, **(kwargs or {}))
        for leaf in tree_leaves(out):
            self._note(leaf)
        return out

    def reset(self) -> None:
        self._entries.clear()
        self.live = 0
        self.peak = 0
        self.peak_breakdown = []


class _BarrierMinChannel:
    def __init__(self, n, timeout=60.0):
        self.n = n
        self._barrier = threading.Barrier(n, timeout=timeout)
        self._slots = [None] * n
        self._result = None

    def channel_for(self, rank):
        def _reduce(vals):
            self._slots[rank] = list(vals)
            index = self._barrier.wait()
            if index == 0:
                self._result = [min(col) for col in zip(*self._slots)]
            self._barrier.wait()
            return list(self._result)

        return _reduce


class _TransportMailbox:
    """Byte channel with the LIFETIMES of ``kv_reshard._dist_exchange``.

    The stock test mailbox clones every outgoing payload into a dict that
    it never clears, which pins one extra copy of the whole send side for
    the rest of the test -- fatal to a measurement of exactly that
    quantity. This one mirrors the real transport instead: the receiver
    allocates its own buffer before the transfer (``torch.empty`` then
    ``irecv``), and the sender's payload is releasable the moment the
    batch completes.
    """

    def __init__(self, n, timeout=60.0):
        self.n = n
        self._posted = threading.Barrier(n, timeout=timeout)
        self._drained = threading.Barrier(n, timeout=timeout)
        self._mail = {}

    def exchange_for(self, rank):
        def _exchange(outgoing, incoming_nbytes):
            for peer, payload in outgoing.items():
                self._mail[(rank, peer)] = payload
            self._posted.wait()
            received = {}
            for peer, nbytes in incoming_nbytes.items():
                src = self._mail.get((peer, rank))
                buf = torch.empty(int(nbytes), dtype=torch.uint8)
                if src is not None:
                    buf.copy_(src)
                received[peer] = buf
            self._drained.wait()
            for peer in list(outgoing):
                self._mail.pop((rank, peer), None)
            return received

        return _exchange


def _make_layout_pools(layer_map, vec, num_slots, seed=7):
    torch.manual_seed(seed)
    n_ranks = len(vec)
    ref_k = [
        torch.randn(num_slots, HEADS, DIM, dtype=torch.bfloat16)
        for _ in range(N_LAYERS)
    ]
    ref_v = [
        torch.randn(num_slots, HEADS, DIM, dtype=torch.bfloat16)
        for _ in range(N_LAYERS)
    ]
    live = (
        torch.arange(num_slots, dtype=torch.int64)[
            torch.randperm(num_slots)[: int(num_slots * 0.8)]
        ]
        .sort()
        .values
    )
    owner = owner_of(live, vec)
    pp_pools, pp_views, tp_pools, tp_views = [], [], [], []
    tp_rows_needed = 1
    for r in range(n_ranks):
        rr = rows_of(live[owner == r], vec, r)
        if rr.numel():
            tp_rows_needed = max(tp_rows_needed, int(rr.max().item()) + 1)
    for r in range(n_ranks):
        k_bufs = [
            torch.zeros(num_slots, HEADS, DIM, dtype=torch.bfloat16)
            for _ in layer_map[r]
        ]
        v_bufs = [
            torch.zeros(num_slots, HEADS, DIM, dtype=torch.bfloat16)
            for _ in layer_map[r]
        ]
        for i, f in enumerate(layer_map[r]):
            k_bufs[i][live] = ref_k[f][live]
            v_bufs[i][live] = ref_v[f][live]
        pp_pools.append((k_bufs, v_bufs))
        pp_views.append(KvPoolView(k_bufs, v_bufs))
        tk = [
            torch.zeros(tp_rows_needed, HEADS, DIM, dtype=torch.bfloat16)
            for _ in range(N_LAYERS)
        ]
        tv = [
            torch.zeros(tp_rows_needed, HEADS, DIM, dtype=torch.bfloat16)
            for _ in range(N_LAYERS)
        ]
        tp_pools.append((tk, tv))
        tp_views.append(KvPoolView(tk, tv))
    return (ref_k, ref_v), live, pp_pools, pp_views, tp_pools, tp_views


def _build_runtimes(pp_views, tp_views, live, mailbox=None, channel=None):
    n = len(VEC)
    channel = channel or _BarrierMinChannel(n)
    mailbox = mailbox or _TransportMailbox(n)
    runtimes = []
    for r in range(n):
        runtimes.append(
            PhaseFlipRuntime(
                n_ranks=n,
                rank=r,
                layer_map=MAP_625,
                n_layers=N_LAYERS,
                tp_vector=VEC,
                boot_phase=PHASE_PP,
                consensus_interval=2,
                collective_min=channel.channel_for(r),
                exchange=mailbox.exchange_for(r),
                pp_pool_view=pp_views[r],
                tp_pool_view=tp_views[r],
                live_slots_fn=lambda: live,
                ready_fn=lambda: True,
                cutover_fn=lambda d: None,
            )
        )
    return runtimes


def _pool_tensors(*pool_lists):
    """Every persistent pool buffer, for the probe's exclusion set."""
    out = []
    for pools in pool_lists:
        for ks, vs in pools:
            out.extend(ks)
            out.extend(vs)
    return out


def _run_ranks_probing(runtimes, directions, probe_rank, probe, rounds=8):
    """Drive one runtime per rank on a real thread, with ``probe`` active
    inside ``probe_rank``'s thread only. ``TorchDispatchMode`` is pushed on
    the per-thread dispatch stack, so entering it in the main thread would
    measure nothing."""
    n = len(runtimes)
    exceptions = [None] * n

    def _body(r):
        if directions[r] is not None:
            runtimes[r].arm(directions[r], source=f"probe-rank{r}")
        for _ in range(rounds):
            runtimes[r].on_round()

    def _worker(r):
        try:
            if r == probe_rank:
                with probe:
                    _body(r)
            else:
                _body(r)
        except BaseException as e:  # noqa: BLE001 -- the assertion target
            exceptions[r] = e

    threads = [threading.Thread(target=_worker, args=(r,)) for r in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120.0)
    alive = [t for t in threads if t.is_alive()]
    if alive:
        raise AssertionError(
            f"{len(alive)} rank threads still alive after 120s -- a flip hang"
        )
    return exceptions


def _plan_legs(live, rank, direction, src, dst, layer_map=MAP_625):
    """Bytes each leg of the move owes, straight from the plan."""
    slots = torch.unique(live.detach().to("cpu", torch.int64))
    tr = build_phase_flip_transition(slots, layer_map, N_LAYERS, VEC, rank, direction)

    def _src_idx(f):
        return layer_map[rank].index(f) if direction == PP_TO_TP else f

    def _dst_idx(f):
        return f if direction == PP_TO_TP else layer_map[rank].index(f)

    outgoing = 0
    for peer, layers in tr.send_layers.items():
        n = int(tr.send_rows[peer].numel())
        outgoing += (
            sum(src.row_nbytes(_src_idx(f)) * n for f in layers) + _CHECKSUM_BYTES
        )
    incoming = 0
    for peer, layers in tr.recv_layers.items():
        n = int(tr.recv_rows[peer].numel())
        incoming += (
            sum(dst.row_nbytes(_dst_idx(f)) * n for f in layers) + _CHECKSUM_BYTES
        )
    local_rows = tr.local_pp_rows if direction == PP_TO_TP else tr.local_tp_rows
    n_local = int(local_rows.numel())
    local = sum(src.row_nbytes(_src_idx(f)) * n_local for f in tr.local_layers)
    return tr, int(outgoing), int(incoming), int(local)


class TestMoverLiveSetIsBounded(CustomTestCase):
    """The mover must hold its payload ONCE, not two or three times."""

    def _measure(self, direction, probe_rank=0):
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, NUM_SLOTS
        )
        runtimes = _build_runtimes(pp_views, tp_views, live)
        persistent = _pool_tensors(pp_pools, tp_pools)
        if direction == TP_TO_PP:
            # Reach the TP phase honestly first, unprobed.
            quiet = LiveStorageProbe(exclude=persistent)
            errs = _run_ranks_probing(runtimes, [PP_TO_TP] * 3, probe_rank, quiet)
            self.assertEqual([e for e in errs if e], [])
        src = pp_views[probe_rank] if direction == PP_TO_TP else tp_views[probe_rank]
        dst = tp_views[probe_rank] if direction == PP_TO_TP else pp_views[probe_rank]
        tr, outgoing, incoming, local = _plan_legs(
            live, probe_rank, direction, src, dst
        )
        probe = LiveStorageProbe(exclude=persistent)
        errs = _run_ranks_probing(runtimes, [direction] * 3, probe_rank, probe)
        self.assertEqual([e for e in errs if e], [])
        self.assertEqual(
            runtimes[probe_rank].completed, 1 if direction == PP_TO_TP else 2
        )
        return probe.peak, outgoing, incoming, local, probe.peak_breakdown

    def _assert_bounded(self, direction):
        peak, outgoing, incoming, local, breakdown = self._measure(direction)
        # The mover CANNOT do better than this: at the exchange the send
        # and receive buffers are both live; after it the receive buffers
        # and the local leg are. Everything else is a copy it chose to
        # keep.
        irreducible = max(outgoing + incoming, incoming + local)
        # Slack for the per-layer gather inside read_rows (one layer's
        # rows out of L, plus its K/V halves) and harness bookkeeping.
        per_layer = outgoing / max(1, len(MAP_625[0]))
        bound = irreducible + 2 * per_layer + 8 * MIB
        self.assertLessEqual(
            peak,
            bound,
            f"{direction}: mover live set {peak / MIB:.1f} MiB exceeds the "
            f"bound {bound / MIB:.1f} MiB (outgoing {outgoing / MIB:.1f}, "
            f"incoming {incoming / MIB:.1f}, local {local / MIB:.1f}, "
            f"irreducible {irreducible / MIB:.1f}). The payload is being "
            f"held more than once -- see HANDOFF_664 section 13. "
            f"Resident at the high-water, MiB: {[round(b / MIB, 2) for b in breakdown[:12]]}",
        )

    def test_pp_to_tp_live_set_is_bounded_by_the_plan(self):
        self._assert_bounded(PP_TO_TP)

    def test_tp_to_pp_live_set_is_bounded_by_the_plan(self):
        self._assert_bounded(TP_TO_PP)

    def test_outgoing_is_released_before_the_local_leg_is_read(self):
        """Peak must not include outgoing AND local at the same time.

        This is the half of the win that a smarter packing alone does not
        buy: the send buffers are dead once the exchange returns, but they
        stayed referenced through the local read, the backing swap and
        every write.
        """
        peak, outgoing, incoming, local, breakdown = self._measure(PP_TO_TP)
        both = outgoing + incoming + local
        self.assertLess(
            peak,
            both,
            f"live set {peak / MIB:.1f} MiB is at least the sum of ALL "
            f"three legs ({both / MIB:.1f} MiB), so the outgoing payloads "
            f"were still resident while the local leg was read. Resident "
            f"at the high-water, MiB: {[round(b / MIB, 2) for b in breakdown[:12]]}",
        )


class TestStagingFormulaMatchesReality(CustomTestCase):
    """``_staging_bytes`` is what the affordability gate spends. It has to
    be the quantity the mover actually holds, or every refusal -- including
    the one that produced the HANDOFF_664 section 9 livelock -- is computed
    from a number that does not exist."""

    def test_staging_bytes_predicts_the_measured_peak(self):
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, NUM_SLOTS
        )
        runtimes = _build_runtimes(pp_views, tp_views, live)
        rt = runtimes[0]
        src, dst = pp_views[0], tp_views[0]
        tr, outgoing, incoming, local = _plan_legs(live, 0, PP_TO_TP, src, dst)
        # PRICE THE SEAM THAT IS ABOUT TO RUN, NOT A DIFFERENT ONE. #656
        # successor 38: this test compared an UNWAVED prediction with a WAVED
        # measurement and had been red since the seam learned to wave. The
        # production gate passes ``_flip_waves(direction)`` into the same call
        # (``phase_flip_runtime._execute``), so a single-wave estimate is a
        # quantity no caller ever spends -- on this fixture it is 20.2 MiB
        # against a 3.8 MiB measured peak, and reading that gap as a 5.4x
        # over-reservation in production is exactly the wrong conclusion.
        single_wave = rt._staging_bytes(tr, PP_TO_TP, src, dst)
        predicted = rt._staging_bytes(tr, PP_TO_TP, src, dst, rt._flip_waves(PP_TO_TP))

        probe = LiveStorageProbe(exclude=_pool_tensors(pp_pools, tp_pools))
        errs = _run_ranks_probing(runtimes, [PP_TO_TP] * 3, 0, probe)
        self.assertEqual([e for e in errs if e], [])

        # The gate must not UNDER-reserve: that is the accounting hole in
        # HANDOFF_664 section 13a, which omitted the local leg entirely. Stated
        # on the single-wave estimate, because the provable floor below is
        # summed over the WHOLE plan and a waved seam never holds it at once.
        self.assertGreaterEqual(
            single_wave,
            peak_floor := max(outgoing + incoming, incoming + local),
            f"_staging_bytes {single_wave / MIB:.1f} MiB is below the bytes "
            f"the move provably holds ({peak_floor / MIB:.1f} MiB): "
            f"outgoing {outgoing / MIB:.1f}, incoming {incoming / MIB:.1f}, "
            f"local {local / MIB:.1f} -- the gate under-reserves",
        )
        # ...and must not GROSSLY over-reserve either: the gate's only
        # action is to refuse, and a refusal does not drain the condition
        # it tested (HANDOFF_664 section 13c), so slack here is livelock
        # brought forward.
        self.assertLessEqual(
            predicted,
            1.25 * probe.peak,
            f"_staging_bytes {predicted / MIB:.1f} MiB over-reserves "
            f"against the measured live set {probe.peak / MIB:.1f} MiB; "
            f"the gate refuses flips that would have fit, and a refusal "
            f"does not drain the resident set it refused on",
        )

    def test_the_waved_price_is_short_of_the_measured_live_set(self):
        """REGISTER C21, as a RATCHET rather than an approval.

        Measured by #656 successor 38 while reframing the test above: the
        price the gate actually spends -- ``_staging_bytes`` with the run's own
        wave plan -- is **2.398 MiB against a measured live set of 3.769 MiB**
        on this three-rank fixture, a ratio of 0.64. The formula models the
        seam's peak as ``incoming + max(outgoing, local)`` on the strength of
        the send buffers being dead before the retained leg is read; on this
        fixture the high-water holds an outgoing leg (0.689 + 0.684 MiB)
        alongside a 1.025 MiB local read and its 1.025 MiB gather window.

        NOT REPRODUCED ON METAL and not fixed here. The s37 acceptance window
        priced the binding card's pp->tp seams at 1177 MiB p50 while the
        deepest NVML drawdown across any of its 115 cutovers was 504 MiB, so
        torch's allocator cache absorbs this transient rather than the driver
        being asked for it -- which is why 65 minutes and 348 flips saw 0
        corridor breaches. Widening the reservation on that evidence would buy
        an earlier wedge (HANDOFF_664 section 13c), so the gap is BOOKED and
        pinned instead of papered over.

        The assertion is one-sided on purpose: closing the gap passes, and only
        WIDENING it fails.
        """
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, NUM_SLOTS
        )
        runtimes = _build_runtimes(pp_views, tp_views, live)
        rt = runtimes[0]
        src, dst = pp_views[0], tp_views[0]
        tr, _outgoing, _incoming, _local = _plan_legs(live, 0, PP_TO_TP, src, dst)
        predicted = rt._staging_bytes(tr, PP_TO_TP, src, dst, rt._flip_waves(PP_TO_TP))
        probe = LiveStorageProbe(exclude=_pool_tensors(pp_pools, tp_pools))
        errs = _run_ranks_probing(runtimes, [PP_TO_TP] * 3, 0, probe)
        self.assertEqual([e for e in errs if e], [])
        self.assertGreaterEqual(
            predicted,
            0.60 * probe.peak,
            f"the waved staging price is {predicted / probe.peak:.2f}x the "
            f"measured live set ({predicted / MIB:.2f} vs {probe.peak / MIB:.2f} "
            f"MiB). C21 recorded 0.64x; a SMALLER ratio means the gate's "
            f"shortfall grew, and the corridor's only defence against it today "
            f"is the allocator cache",
        )

    def test_the_local_leg_is_counted_when_it_dominates(self):
        """The geometry the old formula was blind to.

        ``2 x outgoing + incoming`` happens to exceed the true peak while
        the outgoing leg is the big one, which is why the hole survived
        five successors and a livelock: on the usual token vector the
        wrong formula looked conservative. Give one rank most of the
        tokens and the retained leg overtakes twice the outgoing one, and
        the old expression under-reserves by the whole difference -- the
        state in which the gate says a flip fits and the allocator
        disagrees.
        """
        global VEC
        original = VEC
        # rank 0 owns 5/7 of the slots: it keeps far more than it sends.
        VEC = (5, 1, 1)
        try:
            ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
                MAP_625, VEC, NUM_SLOTS, seed=13
            )
            runtimes = _build_runtimes(pp_views, tp_views, live)
            src, dst = pp_views[0], tp_views[0]
            tr, outgoing, incoming, local = _plan_legs(live, 0, PP_TO_TP, src, dst)
            self.assertGreater(
                local,
                2 * outgoing,
                "geometry does not exercise the hole: the local leg must "
                "exceed twice the outgoing one for the old formula to "
                "under-reserve",
            )
            superseded = 2 * outgoing + incoming
            predicted = runtimes[0]._staging_bytes(tr, PP_TO_TP, src, dst)
            floor = max(outgoing + incoming, incoming + local)
            self.assertLess(
                superseded,
                floor,
                "the superseded formula is supposed to be short here",
            )
            self.assertGreaterEqual(
                predicted,
                floor,
                f"_staging_bytes {predicted / MIB:.1f} MiB still under-"
                f"reserves against {floor / MIB:.1f} MiB (outgoing "
                f"{outgoing / MIB:.1f}, incoming {incoming / MIB:.1f}, "
                f"local {local / MIB:.1f})",
            )
            probe = LiveStorageProbe(exclude=_pool_tensors(pp_pools, tp_pools))
            errs = _run_ranks_probing(runtimes, [PP_TO_TP] * 3, 0, probe)
            self.assertEqual([e for e in errs if e], [])
            self.assertGreaterEqual(
                predicted,
                probe.peak,
                f"_staging_bytes {predicted / MIB:.1f} MiB is below the "
                f"MEASURED live set {probe.peak / MIB:.1f} MiB",
            )
        finally:
            VEC = original


class TestWireFormatUnchanged(CustomTestCase):
    """Streaming the pack must not change a single byte on the wire.

    The mover carries correctness-critical state; a packing-order change
    would be silently wrong for the rows it mis-places and is exactly what
    the payload checksum cannot catch (sender and receiver would agree on
    a wrong order).
    """

    def test_streamed_pack_equals_the_concatenation_reference(self):
        ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, NUM_SLOTS, seed=11
        )
        runtimes = _build_runtimes(pp_views, tp_views, live)
        rt = runtimes[0]
        src = pp_views[0]
        tr, *_ = _plan_legs(live, 0, PP_TO_TP, src, tp_views[0])

        from sglang.srt.managers.kv_reshard import _checksum

        for peer in tr.send_layers:
            parts = [
                src.read_rows(MAP_625[0].index(f), tr.send_rows[peer]).reshape(-1)
                for f in tr.send_layers[peer]
            ]
            flat = torch.cat(parts)
            reference = torch.cat([flat, _checksum(flat)])
            produced = rt._pack_outgoing(tr, PP_TO_TP, src, peer)
            self.assertEqual(produced.dtype, torch.uint8)
            self.assertEqual(produced.numel(), reference.numel())
            self.assertTrue(
                torch.equal(produced, reference),
                f"streamed pack for peer {peer} differs from the "
                f"concatenation reference -- the wire format moved",
            )


if __name__ == "__main__":
    unittest.main()
