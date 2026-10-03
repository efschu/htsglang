"""Q-711 SHORT-KEPT-BOUND (INT8 y8vb 03.10., desk analysis 870): a SHORT kept for D had no upper bound.

METAL (boot_weg2_dkr27browauthoritybar1fs10031930_76f5bbde19_1003_193056.front.log): weg2-14-79
SHORT-KEPT 19:40:51, P-BATCH-TAKES 19:43:08 (137.4 s); 16-84/85/86 the same cluster; 18-95/96/97
152.3 s. SHORT-KEPT p50 4.4 s, p90 137.4 s (y8va p90 4.1 s / 1.0 s). A kept SHORT sits in the
front's ``_ready_for_d``; DECODE-COLLECT release, ``_immediate_park_due``, the fairness bound and
the wait bound all read only ``self.queue``, so nothing flipped D to P for it; it left only when
a seat freed or an unrelated LONG flipped D (P-BATCH-ALL took it along).

Fix: a D-direct entry that waited ``SGLANG_WEG2_SHORT_KEPT_MAX_WAIT_S`` in D's admission line
with no free seat moves to P's queue; the existing triggers see it. Profile row: qwen27b 30 s,
nextflash 0 (off); the dual layout never runs it.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import form as FORM  # noqa: E402
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.srt.weg2.front import Front, Pending  # noqa: E402

ENV = "SGLANG_WEG2_SHORT_KEPT_MAX_WAIT_S"


def _front(seats_free=0, dual=False):
    f = Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, tp_prefill_max_tokens=4096)
    f.state = "serving"
    f.dual_layout = dual
    f.seats_free = lambda: seats_free
    return f


def _pending(rid, age_s, **kw):
    loop = asyncio.new_event_loop()
    return Pending(rid=rid, path="/generate", payload={}, text="x", t_arrive=time.time() - age_s,
                   fut=loop.create_future(), est_prompt=1000, est_uncached=200, **kw)


def _kept(rid, age_s):
    return _pending(rid, age_s, skip_leg1=True, short_kept=True, d_direct=True)


def test_a_kept_short_without_a_seat_moves_to_p_after_the_bound(monkeypatch, caplog):
    """RED on 1de15e82ca: no _short_kept_bound, the entry stayed 137 s."""
    caplog.set_level(logging.INFO)
    monkeypatch.setenv(ENV, "30")
    f = _front(seats_free=0)
    k = _kept("weg2-14-79", 64.0)
    old_q = _pending("queued", 20.0)
    f._ready_for_d.append(k)
    f.queue.append(old_q)
    t0 = 1000.0
    assert f._short_kept_bound(t0) == 0                     # first sight stamps the entry
    assert f._short_kept_bound(t0 + 29.9) == 0
    assert list(f._ready_for_d) == [k]
    assert f._short_kept_bound(t0 + 30.0) == 1
    assert list(f._ready_for_d) == []
    assert [p.rid for p in f.queue] == ["weg2-14-79", "queued"], "arrival order (t_arrive)"
    assert not k.skip_leg1 and not k.short_kept and not k.d_direct and not k.leg1_done
    assert f.counters["short_kept_bound"] == 1
    assert any("WEG2 SHORT-KEPT-BOUND rid=weg2-14-79 waited_s=30.0 bound_s=30.0" in m for m in caplog.messages)
    assert f._batch_gate.is_set(), "the gate follows the emptied deque"


def test_a_free_seat_means_the_admitter_takes_it_nothing_moves(monkeypatch):
    monkeypatch.setenv(ENV, "30")
    f = _front(seats_free=1)
    f._ready_for_d.append(_kept("k", 90.0))
    assert f._short_kept_bound(0.0) == 0
    assert f._short_kept_bound(500.0) == 0
    assert len(f._ready_for_d) == 1


def test_off_is_byte_identical(monkeypatch):
    monkeypatch.setenv(ENV, "0")
    f = _front()
    f._ready_for_d.append(_kept("k", 90.0))
    assert f._short_kept_bound(0.0) == 0 and f._short_kept_bound(500.0) == 0
    assert len(f._ready_for_d) == 1 and not hasattr(f._ready_for_d[0], "t_ready_seen")


def test_dual_layout_never_runs_it(monkeypatch):
    monkeypatch.setenv(ENV, "30")
    f = _front(dual=True)
    f._ready_for_d.append(_kept("k", 90.0))
    assert f._short_kept_bound(0.0) == 0 and f._short_kept_bound(500.0) == 0


def test_a_request_p_already_prefilled_and_the_resume_via_p_stay(monkeypatch):
    monkeypatch.setenv(ENV, "30")
    f = _front()
    handoff = _pending("handoff", 90.0, leg1_done=True)                 # P did leg 1: waits for D
    rvp = _pending("rvp", 90.0, d_direct=True, resume_via_p=True)
    only_p = _pending("only_p", 90.0, d_direct=True, p_only=True)
    f._ready_for_d.extend([handoff, rvp, only_p])
    assert f._short_kept_bound(0.0) == 0 and f._short_kept_bound(500.0) == 0
    assert [p.rid for p in f._ready_for_d] == ["handoff", "rvp", "only_p"]


def test_only_the_expired_entry_moves_the_younger_one_keeps_waiting(monkeypatch):
    monkeypatch.setenv(ENV, "30")
    f = _front()
    a, b = _kept("a", 80.0), _kept("b", 70.0)
    f._ready_for_d.append(a)
    f._short_kept_bound(0.0)
    f._ready_for_d.append(b)
    f._short_kept_bound(20.0)                                            # b stamped at 20
    assert f._short_kept_bound(31.0) == 1
    assert [p.rid for p in f._ready_for_d] == ["b"] and [p.rid for p in f.queue] == ["a"]


def test_the_controller_calls_it_before_it_reads_the_queue():
    import inspect

    src = inspect.getsource(F.Front)
    call = src.index("self._short_kept_bound(_now)")
    assert call < src.index("oldest = self.queue[0].t_arrive if self.queue else None", call - 400)
    assert src.index("self._drop_lapsed_parks()", call - 400) < call


def test_the_profile_rows_nextflash_stays_off_and_the_27b_row_bounds_at_30s():
    assert FORM.PROFILES["qwen27b"].short_kept_max_wait_s == 30.0
    assert FORM.PROFILES["nextflash"].short_kept_max_wait_s == 0.0
    assert FORM.PROFILES["qwen27b"].switch_defaults()[ENV] == 30.0
    assert FORM.PROFILES["nextflash"].switch_defaults()[ENV] == 0.0
    assert FORM.PROFILE_SWITCH_DEFAULTS["qwen27b"][ENV] == 30.0


def test_without_a_form_the_default_is_off(monkeypatch):
    from sglang.srt.environ import envs

    monkeypatch.delenv(ENV, raising=False)
    assert float(envs.SGLANG_WEG2_SHORT_KEPT_MAX_WAIT_S.get()) == 0.0
