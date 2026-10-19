"""S77-B: a live KV-stage grow maps only into bytes the card HAS -- never rc=2.

THE METAL (NF rc17 boot_..._7d93470b5e_1019_171636, D.log, 18:07:42Z). D-MEM-SCHED
grew S8 -> S9 in the middle of a D prefill (``from=S8 to=S9 used=379286
incoming=135152``). TP1/TP2 (3080) paid the KV with expert rows (``LIVE-SPANS
rows_on 8->1 ... order=bank-first``) and applied. TP0 (5090) books the SAME 6 rows
at S8 and S9 (its slack pays the KV in the plan), so ``apply_stage`` mapped its KV
first (``order=kv-first``) -- but the last chunk's transient still sat in the
caching allocator (``PDFLIP-VRAM-PEAK ... reserved_mib=32980 allocated_mib=30126
card_free_mib=109``; PDFLIP-EXTEND-CACHE-TRIM runs only BEFORE the next chunk). The
draft k0 (~68 MiB) mapped, the draft v0 (~68 MiB) did not:
``PdFlipDSeatVramNoMemory: tms_set_spans(v0, now=True) rc=2`` -> TP0 dead, TP1/TP2
dead in the next collective (HiCacheCollectiveError, drain_retired_prefetch).

WHAT MUST HOLD.
(1) Before a live KV grow maps, the rank MEASURES the fresh bytes against the
    card; a short card trims the allocator cache first (the trim's own move).
(2) Still short: this rank's own expert rows go OFF first (bank-first), the
    largest k whose released cells pay the KV. Rows are elastic, the KV is demand.
(3) A real wall (all rows OFF do not pay) stops by name BEFORE anything moved.
(4) The runtime tick asks the GROUP before any rank moves: one rank short -> no
    rank applies the grow, the stage and the cap stay, the admission waits.

The card: ``CardTms`` (the RB test's saver on a card of fixed bytes) plus a
``FakeCard`` whose ``trim`` hands the allocator's cached bytes back to the card.
RED on 7d93470b5e (rc=2 in (1)); GREEN with S77-B.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_s77_rb", os.path.join(os.path.dirname(__file__), "test_pdflip_d_stage_grow_release_first_rb.py"))
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
h = rb.h

KV_GROW = 2 * 64 * 1024  # S0 -> S1: two KV tensors x 64 rows x 1 KiB
ROW = 2 * 4096           # one expert row ON: two banks x one 4 KiB cell


class FakeCard:
    """cudaMemGetInfo of the card: what the saver's budget leaves; ``cache`` = the
    allocator's cached segments (the chunk transient) that only a trim gives back."""

    def __init__(self, tms, cache=0):
        self.tms, self.cache, self.trims = tms, int(cache), 0

    def free(self):
        return int(self.tms.budget - self.tms.total())

    def trim(self):
        self.trims += 1
        got, self.cache = self.cache, 0
        self.tms.budget += got
        return got


def _kv_first_state(bite):
    """The metal's TP0 form: rows_on at or below the S1 cell's rows, so the next
    S0 -> S1 is kv-first. S0 -> S1 (32 -> 16 rows) on a full card, then back to S0:
    the KV prefix goes first (RB) and ``bite`` of its freed bytes are taken
    before the bank regrows (the chunk transient the caching allocator keeps), so
    the elastic regrow maps only what is left."""
    tms = rb.CardTms()
    ctx = h._armed(tms)
    ctx.__enter__()
    r, ctl = rb._live_s0(tms)
    tms.budget = tms.total()
    ctl.apply_stage(2, 1)            # S1, rows 32 -> 16 (bank-first)
    real = ctl._grow_bank_elastic

    def _bitten(plans, k, census):
        tms.budget -= bite           # the transient lands in the KV's freed bytes
        return real(plans, k, census)

    with mock.patch.object(ctl, "_grow_bank_elastic", _bitten):
        ctl.apply_stage(2, 0)        # S0: the KV goes, the bank regrows only what fits
    return tms, ctx, r, ctl


def _snapshot(tms):
    return {p: list(a["ext"]) for p, a in tms.allocs.items()}


def test_rc17_tp0_the_kv_first_grow_trims_the_cache_and_never_runs_into_rc2(caplog):
    """RED on 7d93470b5e: PdFlipDSeatVramNoMemory (rc=2) -- the metal's TP0.
    GREEN: the card is measured short, the cache trim returns the transient, the
    KV maps; no expert row had to go."""
    caplog.set_level(logging.INFO)
    tms, ctx, r, ctl = _kv_first_state(bite=KV_GROW)
    try:
        assert ctl.rows_on == 16 and ctl.cells[(2, 1)].extra_rows == 16  # kv-first
        card = FakeCard(tms, cache=KV_GROW)
        with mock.patch.object(dsv, "_CARD", card, create=True):
            applied = ctl.apply_stage(2, 1)
        assert (applied.stage, applied.extra_rows) == (1, 16)
        assert card.trims == 1 and tms.total() <= tms.budget
        for t in r.kv:
            assert tms.mapped(t.data_ptr()) == (128 + h.PAGE) * 1024
        line = [m for m in caplog.messages if dsv.GROW_ROOM_MARK in m][-1]
        assert "verdict=ok" in line and "rows_to=-" in line
    finally:
        ctx.__exit__(None, None, None)


def test_an_untrimmable_short_card_sends_this_ranks_own_rows_off_first(caplog):
    """Nothing in the cache: the rows pay. 16 rows ON, 128 KiB to map, 0 free ->
    all 16 rows OFF (bank-first), the KV maps, the rank lives."""
    caplog.set_level(logging.INFO)
    tms, ctx, r, ctl = _kv_first_state(bite=KV_GROW)
    try:
        card = FakeCard(tms, cache=0)
        with mock.patch.object(dsv, "_CARD", card, create=True):
            applied = ctl.apply_stage(2, 1)
        assert applied.stage == 1 and applied.extra_rows == 0 and ctl.rows_on == 0
        assert tms.total() <= tms.budget
        live = [m for m in caplog.messages if dsv.LIVE_MARK in m][-1]
        assert "rows_on 16->0" in live and "order=bank-first" in live
    finally:
        ctx.__exit__(None, None, None)


def test_bank_first_that_still_falls_short_takes_more_rows_than_the_cell():
    """RB alone is not enough when the card is short: the cell's k (16) releases 8
    rows of 24, 64 KiB for 128 KiB of KV -- rc=2 on 7d93470b5e. GREEN: the largest
    k ON THE ROWS' LATTICE whose released cells pay the KV (no straddled extent)."""
    tms, ctx, r, ctl = _kv_first_state(bite=KV_GROW // 2)
    try:
        assert ctl.rows_on == 24  # the elastic regrow took half of what it gave
        want = max(c for c in ctl.row_cut_ks if c < 16 and (24 - c) * ROW >= KV_GROW)
        card = FakeCard(tms, cache=0)
        with mock.patch.object(dsv, "_CARD", card, create=True):
            applied = ctl.apply_stage(2, 1)
        assert applied.stage == 1 and applied.extra_rows == want < 16
        assert tms.total() <= tms.budget
    finally:
        ctx.__exit__(None, None, None)


def test_a_real_wall_stops_by_name_before_anything_moved():
    """No reserve invented, no partial form: all rows OFF still leave the KV 4 KiB
    short -> the named refusal, every extent and the rows exactly as before."""
    tms, ctx, r, ctl = _kv_first_state(bite=KV_GROW)
    try:
        tms.budget -= 4096
        before, rows = _snapshot(tms), ctl.rows_on
        card = FakeCard(tms, cache=0)
        with mock.patch.object(dsv, "_CARD", card, create=True):
            with pytest.raises(dsv.PdFlipDSeatVramNoMemory, match="GROW-ROOM"):
                ctl.apply_stage(2, 1)
        assert _snapshot(tms) == before and ctl.rows_on == rows
    finally:
        ctx.__exit__(None, None, None)


def test_without_a_measured_card_the_plan_is_applied_as_before():
    """CPU / no CUDA context: free() is None -> nothing measured, nothing trimmed
    (the RB test's own card-too-small refusal stays a saver rc=2)."""
    tms, ctx, r, ctl = _kv_first_state(bite=0)
    try:
        card = types.SimpleNamespace(free=lambda: None, trim=lambda: pytest.fail("trimmed"))
        with mock.patch.object(dsv, "_CARD", card, create=True):
            assert ctl.grow_room(2, 1) is None
            applied = ctl.apply_stage(2, 1)
        assert applied.stage == 1
    finally:
        ctx.__exit__(None, None, None)


# ---- (4) the group verdict in the runtime tick ---------------------------------

from test_pdflip_d_mem_sched_0929 import _req, tick_env  # noqa: E402,F401


class _Ctl:
    """A rank's controller as the tick sees it: cells, a measured room, applies."""

    def __init__(self, ok):
        self.cells = {(6, 0): None}
        self.ok = ok
        self.applied = []
        self.evicted_rows = self.refilled_rows = 0
        self.sync_ms = self.copy_ms = 0.0
        self.bytes_copied = 0

    def grow_room(self, n, stage, **_kw):
        return dsv.GrowRoom(ok=self.ok, need=232 << 20, free_before=109 << 20,
                            free_after=109 << 20, trimmed=0, rows_from=6, rows_cell=6,
                            rows_to=None, release_all=0)

    def apply_stage(self, n, stage):
        self.applied.append((n, stage))
        return types.SimpleNamespace(extra_rows=6)

    def warm_rows(self):
        return 0


def test_one_rank_short_moves_no_rank_and_keeps_the_cap(tick_env, caplog):
    """The rank here has room; a PEER has none (group MIN 0): no apply, the stage
    and the cap stay where they were, the refusal is counted and named."""
    caplog.set_level(logging.WARNING)
    dsv_, sched, caps, _floor, votes, grid = tick_env
    ctl = _Ctl(ok=True)
    setattr(sched, dsv_.CTL_ATTR, ctl)
    sched._pdflip_group_min_ints = lambda vals: (votes.append(list(vals)) or [0] * len(vals))
    sched.running_batch.reqs = [_req("a", 300_000)]
    st = dsv_.runtime_tick(sched)
    ms = getattr(sched, dsv_.MEM_SCHED_ATTR)
    assert not st.changed and ms.stage == 0 and getattr(sched, dsv_.PHASE_ATTR).stage == 0
    assert ctl.applied == [] and caps[-1] == grid[0]
    assert [1] in votes and ms.counters["stage_up_refused_room"] == 1
    assert any(dsv_.GROW_ROOM_MARK in m and "a peer rank" in m for m in caplog.messages)


def test_this_rank_short_names_its_own_numbers(tick_env, caplog):
    caplog.set_level(logging.WARNING)
    dsv_, sched, caps, _floor, votes, grid = tick_env
    ctl = _Ctl(ok=False)
    setattr(sched, dsv_.CTL_ATTR, ctl)
    sched._pdflip_group_min_ints = lambda vals: (votes.append(list(vals)) or [min(vals)])
    sched.running_batch.reqs = [_req("a", 300_000)]
    st = dsv_.runtime_tick(sched)
    assert not st.changed and ctl.applied == [] and [0] in votes
    assert any("maps 232.0 MiB fresh" in m for m in caplog.messages)


def test_every_rank_with_room_grows_as_before(tick_env):
    dsv_, sched, caps, _floor, votes, grid = tick_env
    ctl = _Ctl(ok=True)
    setattr(sched, dsv_.CTL_ATTR, ctl)
    sched.running_batch.reqs = [_req("a", 300_000)]
    st = dsv_.runtime_tick(sched)
    ms = getattr(sched, dsv_.MEM_SCHED_ATTR)
    assert st.changed and ms.stage > 0 and ctl.applied == [(6, ms.stage)]
    assert [1] in votes and caps[-1] == grid[ms.stage]


def test_a_rank_without_pages_still_enters_the_vote(tick_env):
    """A 3080 worker with no controller (CTL False) votes yes -- it must still
    enter the collective, or the ranks with pages wait for a partner forever."""
    dsv_, sched, caps, _floor, votes, grid = tick_env
    sched.running_batch.reqs = [_req("a", 300_000)]
    st = dsv_.runtime_tick(sched)
    assert st.changed and [1] in votes
