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
import types
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


class _SchedStub:
    """The minimum the #856 seam requires of a scheduler.

    The flip now RETRACTS its residents and DROPS the prefix tree before the
    cutover (the flip carries no KV; see `release_residents_for_cutover`), so
    a runtime built by hand -- rather than by `build_phase_flip_runtime`,
    which binds the real scheduler -- has to supply a resettable tree. There
    are no resident requests in this harness, so the retraction is a no-op and
    none of the pools are touched.
    """

    def __init__(self):
        self.resets = 0
        self.tree_cache = types.SimpleNamespace(reset=self._reset)

    def _reset(self):
        self.resets += 1


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
        runtimes[-1]._census_scheduler = _SchedStub()
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


class TestTheFlipCarriesNoKv(CustomTestCase):
    """#856 -- REPLACES ``TestStagingFormulaMatchesReality``.

    THE RETIRED CONTRACT was "``_staging_bytes`` must cover the bytes the
    mover holds", asserted three ways: not under-reserving against the plan's
    legs, not over-reserving against the measured peak, and counting the local
    leg when it dominates. All three priced A MOVE.

    The mover no longer runs inside a flip. The residents are retracted and
    the transfer plan is rebuilt EMPTY before the wave loop
    (``_release_residents_for_cutover``), so the seam holds nothing, and a
    reservation sized to a move reserves for something that cannot happen.
    That reservation is what refused 33 arms in W25 -- 25 on the staging rate
    limit, 17 FLIP ABANDONED -- against a 2339 MiB ``tp_to_pp`` ask on PP0.

    So the assertions INVERT: a flip that still moves KV must FAIL here. The
    fixture is unchanged and still builds a real, non-empty live set, which is
    what makes "it moved nothing" a result rather than a tautology.
    """

    def _flip_and_probe(self, direction):
        _ref, live, pp_pools, pp_views, tp_pools, tp_views = _make_layout_pools(
            MAP_625, VEC, NUM_SLOTS
        )
        runtimes = _build_runtimes(pp_views, tp_views, live)
        probe = LiveStorageProbe(exclude=_pool_tensors(pp_pools, tp_pools))
        errs = _run_ranks_probing(runtimes, [direction] * 3, 0, probe)
        self.assertEqual([e for e in errs if e], [])
        return runtimes, probe, pp_views, tp_views, live

    def test_the_seam_moves_no_kv_at_all(self):
        # THE HEADLINE ASSERTION of the whole ticket. `probe.peak` is the
        # transient storage the seam allocated outside the persistent pools;
        # under the old contract it was 3.8-27.7 MiB on this fixture.
        _rt, probe, _s, _d, _live = self._flip_and_probe(PP_TO_TP)
        self.assertEqual(
            probe.peak,
            0,
            f"the flip allocated {probe.peak / MIB:.1f} MiB of transient "
            f"storage; under #856 it carries NO KV and must hold nothing",
        )

    def test_the_other_direction_moves_no_kv_either(self):
        _rt, probe, _s, _d, _live = self._flip_and_probe(TP_TO_PP)
        self.assertEqual(probe.peak, 0)

    def test_the_prefix_tree_is_dropped_once_per_flip_on_every_rank(self):
        # The other half of the contract: nothing is carried, so every rank
        # must invalidate its device tier or the next phase reads prefixes
        # naming rows that hold no KV.
        runtimes, _p, _s, _d, _l = self._flip_and_probe(PP_TO_TP)
        for rt in runtimes:
            self.assertEqual(rt._census_scheduler.resets, 1)

    def test_the_staging_ask_is_independent_of_the_live_set(self):
        # `wave_peak` is retired from the ask, so the SAME number must come
        # back for a full plan and an empty one. This is the funding claim in
        # unit form: the gate can no longer refuse a flip because of KV the
        # flip will not move.
        _ref, live, _ppp, pp_views, _tpp, tp_views = _make_layout_pools(
            MAP_625, VEC, NUM_SLOTS
        )
        rt = _build_runtimes(pp_views, tp_views, live)[0]
        src, dst = pp_views[0], tp_views[0]
        tr, outgoing, incoming, local = _plan_legs(live, 0, PP_TO_TP, src, dst)
        # The fixture must really have a live set, or this proves nothing.
        self.assertGreater(outgoing + incoming + local, 0)
        empty = build_phase_flip_transition(
            torch.empty(0, dtype=torch.int64), MAP_625, N_LAYERS, VEC, 0, PP_TO_TP
        )
        waves = rt._flip_waves(PP_TO_TP)
        self.assertEqual(
            rt._seam_reserve_bytes(tr, PP_TO_TP, src, dst, waves),
            rt._seam_reserve_bytes(empty, PP_TO_TP, src, dst, waves),
            "the staging ask still tracks the live set; wave_peak is not "
            "retired and the gate can still refuse a flip over KV it will "
            "not move",
        )

    def test_the_retired_term_is_still_reported(self):
        # A term that vanishes silently cannot be shown to have been retired,
        # and the proof window's funding claim is exactly the difference
        # between what this used to reserve and what it now reserves.
        _ref, live, _ppp, pp_views, _tpp, tp_views = _make_layout_pools(
            MAP_625, VEC, NUM_SLOTS
        )
        rt = _build_runtimes(pp_views, tp_views, live)[0]
        src, dst = pp_views[0], tp_views[0]
        tr, _o, _i, _l = _plan_legs(live, 0, PP_TO_TP, src, dst)
        rt._seam_reserve_bytes(tr, PP_TO_TP, src, dst, rt._flip_waves(PP_TO_TP))
        self.assertGreater(int(rt._retired_wave_peak_bytes), 0)


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
