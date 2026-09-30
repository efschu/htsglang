"""D-SEAT-REWAKE (Nutzer 30.09.): D's phase seat count n moves live, both ways.

Wörtlich: "oder D sleeped (ohne wirklich runterzufaren) und waket sofort wieder
mit mehr sitzen, macht den decode selbst (weil kurz) und decoded dann alle
sitze weiter"; "dann kann der D teil auch wenn sitze nicht mehr belegt sein
intern flippen zu einem layout mit weniger sitzen, dann gibts mehr platz für
experten - selbe funktion nur auch wieder in die andere richtung".

Metal (NF y4s 09301432, front): ``why=seat taken=3 n=3`` with 441613 KV tokens
free -- weg2-10-16 waited 6.43 s, weg2-4-9 6.4 s: the wake fixed n (H95c,
handoff_n + parked_n) and nothing moved it until a seat ended.

Pinned: (1) a live GROW keeps every GDN byte of the running seats and pays the
new seats' cells with expert rows released FIRST (a card without slack holds
it); (2) a live SHRINK keeps the GDN cells of the seats that stay -- which
needs the lattice cut at the wake (without it: the named wipe refusal) -- and
gives the freed pages back to expert rows; (3) the tick grows at once when
waiting requests find every seat taken, shrinks only past the MEASURED price
of a re-plan round trip (ski rental: no flapping on a short gap), never with
nothing measured, never over a slot still in use, and every verdict goes
through the group MIN (rank-uniform); (4) the front counts D's cap as its seats.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.weg2 import d_seat_rewake as R  # noqa: E402
from sglang.srt.weg2 import d_seat_vram as dsv  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_rb_rewake", os.path.join(os.path.dirname(__file__), "test_weg2_d_stage_grow_release_first_rb.py"))
rb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rb)
h = rb.h
G = h.G
SLOTS = 12          # the desk rank's GDN tensor: 3 layers x 12 slots x 4 KiB (pool 11)


def _gdn_ids(tms, t, keep):
    """The extents holding slots [0, keep) of every layer."""
    return [tms.ids(t.data_ptr(), layer * SLOTS * G, (layer * SLOTS + keep) * G) for layer in range(3)]


def _live_phase(tms, n, rewake):
    r = h._rank(tms)
    h._pause_all(tms)
    with envs.SGLANG_WEG2_D_SEAT_REWAKE.override(rewake):
        dsv.on_wake(r.sched, h._kv_leg("e1", n, 40), h._seats(n))
    for p in list(tms.allocs):
        tms.resume(p)
    return r, r.sched._weg2_d_seat_vram, r.sched._weg2_d_seat_phase


# ---------------------------------------------------------------- (1)/(2) pages

def test_a_live_grow_keeps_the_running_seats_gdn_bytes_and_releases_rows_first(caplog):
    caplog.set_level(logging.INFO)
    tms = rb.CardTms()
    with h._armed(tms), envs.SGLANG_WEG2_D_SEAT_REWAKE.override(True):
        r, ctl, st = _live_phase(tms, 1, True)
        keep1 = int(ctl.mamba_keep)
        assert keep1 < SLOTS
        before = _gdn_ids(tms, r.temporal, keep1)
        rows1 = ctl.rows_on
        # the card holds the boot form: n=1's pages plus the granule slack the
        # n=2 cell maps above them (the cell table never books more), no more
        j = int(st.stage)
        tms.budget = tms.total() + max(0, ctl.cells[(2, j)].mapped - ctl.cells[(1, j)].mapped)
        applied = ctl.reseat_live(2, st.stage)
        assert _gdn_ids(tms, r.temporal, keep1) == before, "no GDN byte of a running seat moved"
        assert ctl.mamba_keep is None or ctl.mamba_keep > keep1
        assert applied.extra_rows <= rows1 and ctl.rows_on == applied.extra_rows
        assert tms.total() <= tms.budget              # released before mapped
        line = [m for m in caplog.messages if dsv.LIVE_MARK in m][-1]
        assert "order=bank-first" in line and "wiped=0" in line


def test_a_live_shrink_keeps_the_staying_seats_and_gives_the_pages_to_expert_rows():
    tms = rb.CardTms()
    with h._armed(tms), envs.SGLANG_WEG2_D_SEAT_REWAKE.override(True):
        r, ctl, st = _live_phase(tms, 2, True)
        keep1 = ctl.reseat_target(1, st.stage)[1]
        before = _gdn_ids(tms, r.temporal, keep1)
        rows2, mapped2 = ctl.rows_on, tms.mapped(r.temporal.data_ptr())
        tms.budget = tms.total()
        applied = ctl.reseat_live(1, st.stage)
        assert _gdn_ids(tms, r.temporal, keep1) == before
        assert tms.mapped(r.temporal.data_ptr()) < mapped2
        assert applied.extra_rows >= rows2              # the freed pages became expert rows
        assert tms.total() <= tms.budget


def test_a_shrink_without_the_wakes_lattice_stops_by_name():
    """Without the cut GDN plan (the wake before this switch) a live shrink
    would release the one extent of each layer: the named wipe, nothing moved."""
    tms = rb.CardTms()
    with h._armed(tms):
        r, ctl, st = _live_phase(tms, 2, False)
        with envs.SGLANG_WEG2_D_SEAT_REWAKE.override(True):
            ctl.spans_by_ptr[r.temporal.data_ptr()] = tuple(
                (e[0], e[1]) for e in tms.allocs[r.temporal.data_ptr()]["ext"])
            with pytest.raises(dsv.Weg2DSeatVramWipe):
                ctl.reseat_live(1, st.stage)


# ---------------------------------------------------------------- pure policy

def test_grow_at_once_shrink_only_past_the_measured_price():
    assert R.grow_target(3, 6, 3, 2) == 5
    assert R.grow_target(3, 6, 3, 9) == 6              # at most the cap
    assert R.grow_target(3, 6, 2, 1) is None            # a seat is free: no grow
    assert R.grow_target(6, 6, 6, 1) is None
    assert R.shrink_target(4, 2, 0) == 2
    assert R.shrink_target(4, 2, 1) is None             # somebody waits
    assert R.shrink_target(1, 0, 0) is None
    rs = R.RewakeState()
    assert R.price_ms(rs, None) is None                 # nothing measured: no price
    assert R.price_ms(rs, 40.0) == 80.0                 # the wake's page apply, both ways
    rs.grow_ms = 55.0
    assert R.price_ms(rs, 40.0) == 110.0
    rs.shrink_ms = 30.0
    assert R.price_ms(rs, 40.0) == 85.0
    assert not R.shrink_due(0.05, 85.0) and R.shrink_due(0.09, 85.0)
    assert not R.shrink_due(10.0, None)


# ---------------------------------------------------------------- the tick

class _Alloc:
    def __init__(self, size=38, used=()):
        self.size = size
        self.slot_used = torch.zeros(size + 1, dtype=torch.bool)
        for u in used:
            self.slot_used[u] = True
        self.limits = []

    def set_phase_limit(self, limit, seats=None):
        if limit is not None and bool(self.slot_used[int(limit) + 1:].any()):
            return False
        self.limits.append((limit, seats))
        return True


class _Ctl:
    def __init__(self, fail=False):
        self.calls, self.fail, self.rows_on, self.mamba_keep = [], fail, 20, 7

    def reseat_live(self, n, stage):
        self.calls.append(n)
        if self.fail:
            raise dsv.Weg2DSeatVramNoMemory("rc=2")
        return None


def _sched(n, running, waiting, alloc=None, gm_calls=None, gm=None):
    s = types.SimpleNamespace(
        running_batch=types.SimpleNamespace(reqs=[types.SimpleNamespace(rid="r%d" % i) for i in range(running)],
                                            batch_is_full=True),
        waiting_queue=[types.SimpleNamespace(rid="w%d" % i) for i in range(waiting)],
        chunked_req=None, weg2_dormant=False)
    s._weg2_d_seat_phase = dsv.PhaseState(epoch="e1", n=n, cap=6, has_n=True, done=True, apply_ms=40.0)
    rtp = types.SimpleNamespace(mamba_allocator=alloc or _Alloc())
    s.tp_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(req_to_token_pool=rtp))
    calls = gm_calls if gm_calls is not None else []

    def group_min(flags):
        calls.append(list(flags))
        return gm(flags) if gm else [1 if f else 0 for f in flags]
    s._weg2_group_min_flags = group_min
    return s, calls


def _tick_env(ctl):
    return [envs.SGLANG_WEG2_D_SEAT_REWAKE.override(True),
            mock.patch.object(dsv, "armed", lambda env=None: True),
            mock.patch.object(dsv, "controller", lambda sched: ctl),
            mock.patch.object(dsv, "stage_form", lambda env=None: None),
            mock.patch.object(dsv, "_unmerged_extend", lambda sched, running: [])]


def _run(ctxs, fn):
    for c in ctxs:
        c.__enter__()
    try:
        return fn()
    finally:
        for c in reversed(ctxs):
            c.__exit__(None, None, None)


def test_y4s_a_waiting_request_grows_the_seats_at_the_next_round(caplog):
    """y4s: taken=3 n=3, a request waits with its KV free -> n=4 now, the
    allocator's limit rises, admission re-opens; group MIN asked once."""
    caplog.set_level(logging.WARNING)
    ctl = _Ctl()
    s, calls = _sched(3, 3, 1)
    got = _run(_tick_env(ctl), lambda: R.tick(s))
    assert got == "grow" and ctl.calls == [4]
    assert s._weg2_d_seat_phase.n == 4
    assert s.tp_worker.model_runner.req_to_token_pool.mamba_allocator.limits[-1][1] == 4
    assert s.running_batch.batch_is_full is False
    assert calls == [[True]]
    assert any("D-SEAT-REWAKE GROW n=3->4" in m and "pause_ms=" in m for m in caplog.messages)


def test_a_grow_one_rank_cannot_map_leaves_n_on_every_rank():
    ctl = _Ctl()
    s, calls = _sched(3, 3, 1, gm=lambda flags: [0])     # another rank refused
    assert _run(_tick_env(ctl), lambda: R.tick(s)) is None
    assert s._weg2_d_seat_phase.n == 3
    assert s.tp_worker.model_runner.req_to_token_pool.mamba_allocator.limits == []
    assert _run(_tick_env(ctl), lambda: R.tick(s)) is None and ctl.calls == [4], "not asked again"


def test_free_seats_shrink_only_after_the_price_no_flapping(caplog):
    caplog.set_level(logging.WARNING)
    ctl = _Ctl()
    s, calls = _sched(4, 2, 0)
    clock = [1000.0]

    def ctxs():
        return _tick_env(ctl) + [mock.patch.object(R.time, "monotonic", lambda: clock[0])]
    # a short gap: 32 rounds in 50 ms < the 80 ms price (2 x the wake's 40 ms)
    for _ in range(R.SHRINK_ASK_ROUNDS):
        clock[0] += 0.05 / R.SHRINK_ASK_ROUNDS
        assert _run(ctxs(), lambda: R.tick(s)) is None
    assert ctl.calls == [] and calls == [[False]]
    # the gap goes on past the price -> n=2
    got = None
    for _ in range(R.SHRINK_ASK_ROUNDS):
        clock[0] += 0.05 / R.SHRINK_ASK_ROUNDS
        got = _run(ctxs(), lambda: R.tick(s))
    assert got == "shrink" and ctl.calls == [2] and s._weg2_d_seat_phase.n == 2
    assert any("D-SEAT-REWAKE SHRINK n=4->2" in m and "price_ms=80.0" in m for m in caplog.messages)


def test_a_waiter_resets_the_free_seat_clock():
    ctl = _Ctl()
    s, calls = _sched(4, 2, 0)
    for _ in range(R.SHRINK_ASK_ROUNDS - 1):
        _run(_tick_env(ctl), lambda: R.tick(s))
    s.waiting_queue = [types.SimpleNamespace(rid="w")]
    _run(_tick_env(ctl), lambda: R.tick(s))                         # a seat is free: the waiter is admitted, no grow
    assert getattr(s, R.ATTR).idle_rounds == 0 and ctl.calls == []


def test_no_shrink_over_a_slot_still_in_use_and_none_without_a_price():
    ctl = _Ctl()
    s, calls = _sched(4, 2, 0, alloc=_Alloc(used=(30,)))
    for _ in range(R.SHRINK_ASK_ROUNDS * 3):
        _run(_tick_env(ctl), lambda: R.tick(s))
    assert ctl.calls == [] and calls == []
    s2, calls2 = _sched(4, 2, 0)
    s2._weg2_d_seat_phase.apply_ms = None                 # nothing measured
    for _ in range(R.SHRINK_ASK_ROUNDS * 3):
        _run(_tick_env(ctl), lambda: R.tick(s2))
    assert ctl.calls == [] and all(f == [False] for f in calls2)


def test_off_nothing_moves():
    ctl = _Ctl()
    s, calls = _sched(3, 3, 1)
    ctxs = _tick_env(ctl)
    ctxs[0] = envs.SGLANG_WEG2_D_SEAT_REWAKE.override(False)
    assert _run(ctxs, lambda: R.tick(s)) is None and ctl.calls == [] and calls == []


def test_the_scheduler_runs_the_tick_after_the_stage_tick():
    import inspect

    from sglang.srt.managers import scheduler as SC

    src = inspect.getsource(SC)
    i = src.index("_weg2_d_seat_vram.runtime_tick(self)")
    assert "_weg2_d_seat_rewake.tick(self)" in src[i:i + 400]


# ---------------------------------------------------------------- (4) the front

def test_the_front_counts_ds_cap_as_its_seats(monkeypatch):
    import sys

    sys.path.insert(0, os.path.dirname(__file__))
    from test_weg2_arrival_seat_rule_0929 import _front  # noqa: E402

    f = _front(running=["a", "b", "c"], n=3)
    assert f._arrival_seat_taken() == (3, 3)
    with envs.SGLANG_WEG2_D_SEAT_REWAKE.override(True):
        assert f._arrival_seat_taken() == (3, 6)          # d_bs: D grows its seats live
