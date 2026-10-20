# SPDX-License-Identifier: Apache-2.0
"""#1210 (04.10., NF y9nf4 e23f7dff30, boot 1004_031945): a 69-token short
waited 280 s (driver) behind two parked requests whose store read stayed short.

DER BEFUND (D-, P- und Front-Log):
* D's sleep 03:33:18 found the Mamba anchor arena full (``#1427 ARENA-CLAIM
  REFUSED statuses=[4]`` x8, ``#1421 BACKUP-REFUSED why=mamba_claim`` /
  ``parent_unbacked``) and dropped the park anchors of pdflip-8-79 (49408) and
  pdflip-8-82 (76032): ``PDFLIP-ANCHOR-LOST at=flush``. Every later read stopped
  at the deepest host anchor (47104 / 67840) -- for good; P never saw the two
  before their reroute.
* The #1471 settle polled them (``writer=p-handoff``); every park_running
  folded them into the flip park (``settle-folded``), every wake re-stamped
  the 20-s clock, no D phase (15-19 s) let it lapse: 10 wakes, 5.5 min.
* The park barrier counted them as ``parked_outside`` and held the newcomer
  pdflip-12-99 (69 tokens, no prefix) in 43 passes over 7 D phases of 6 seats
  with 1-3 running; the pass in which the two left the settle (W50-REROUTE
  03:39:04) admitted it.

SC: the settle clock of a folded request runs over the wake (as #287 NEED0
for a budget-refused one); the wake's own read is awaited once.
SB: while the barrier stands only on parked requests OUTSIDE the queue, a
newcomer takes a phase seat beyond those held for every waiting parked one
(the user's backfill rule, 30.09.).
"""
from __future__ import annotations

import inspect
import os
import types

import pytest

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import d_park_runtime as rt  # noqa: E402
from flliper.srt.pdflip import d_seats as ds  # noqa: E402
from flliper.srt.pdflip import settle_writer as sw  # noqa: E402

_SRT = os.path.dirname(os.path.dirname(ds.__file__))
SETTLE_S = 20.0


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    for name in (ds.PARK_ENV, ds.RESUME_MARGIN_ENV, "FLLIPER_PDFLIP_SEAT_ROTATE",
                 "FLLIPER_PDFLIP_D_PARK_BARRIER_BACKFILL", "FLLIPER_PDFLIP_SETTLE_CLOCK_CARRY"):
        monkeypatch.delenv(name, raising=False)


def _req(rid, seq, site=None):
    r = types.SimpleNamespace(rid=rid, kv_arrival_seq=seq, origin_input_ids=[0] * 10, output_ids=[])
    if site is not None:
        ds.mark_parked(r, site)
    return r


def _incident_gate(*, seats=6, running_n=1, waiting=None):
    """D phase 16 (03:35:15): 6 seats, pdflip-14-100 running, the two settle
    requests flip-parked outside the queue, the newcomer pdflip-12-99 waiting."""
    running = [_req(f"pdflip-14-10{i}", 100 + i) for i in range(running_n)]
    outside = [_req("pdflip-8-79", 79, ds.SITE_FLIP), _req("pdflip-8-82", 82, ds.SITE_FLIP)]
    short = _req("pdflip-12-99", 99)
    waiting = [short] if waiting is None else waiting
    gate = ds.admission_gate(waiting, running=running, pending_outside=outside, seats=seats)
    return gate, short


# ---- SB: the barrier ---------------------------------------------------------------

@pytest.mark.parametrize("rotate", ["0", "1"])
def test_y9nf4_short_backfills_a_free_seat_beside_the_settle_held_parked(monkeypatch, rotate):
    monkeypatch.setenv("FLLIPER_PDFLIP_SEAT_ROTATE", rotate)
    gate, short = _incident_gate()
    assert gate.barrier and gate.parked_outside == 2
    assert gate.seat_room == 6 - 1 - 2
    assert gate.skip(short, admitted=[]) is None


def test_switch_off_keeps_the_static_barrier(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_D_PARK_BARRIER_BACKFILL", "0")
    gate, short = _incident_gate()
    assert gate.skip(short, admitted=[]) == "pdflip_d_park_first"


def test_no_seat_beyond_the_held_ones_keeps_the_newcomer_back():
    gate, short = _incident_gate(seats=3, running_n=1)  # 3 - 1 running - 2 held = 0
    assert gate.skip(short, admitted=[]) == "pdflip_d_park_first"


def test_the_backfill_takes_only_as_many_newcomers_as_seats_are_free():
    gate, short = _incident_gate(seats=4, running_n=1)  # room 1
    other = _req("pdflip-13-101", 101)
    assert gate.skip(short, admitted=[]) is None
    assert gate.skip(other, admitted=["pdflip-12-99"]) == "pdflip_d_park_first"


def test_a_parked_request_still_waiting_in_the_queue_goes_first():
    parked_q = _req("pdflip-10-91", 91, ds.SITE_FLIP)
    short = _req("pdflip-12-99", 99)
    gate, _ = _incident_gate(waiting=[parked_q, short])
    # the parked one in the queue was not admitted this pass: no backfill
    assert gate.skip(short, admitted=[]) == "pdflip_d_park_first"
    # admitted in this pass: the newcomer takes a remaining seat
    assert gate.skip(short, admitted=["pdflip-10-91"]) is None


def test_without_a_phase_seat_count_the_barrier_is_unchanged():
    gate, short = _incident_gate(seats=None)
    assert gate.seat_room is None
    assert gate.skip(short, admitted=[]) == "pdflip_d_park_first"


def test_old_static_call_without_admitted_list_is_unchanged():
    gate, short = _incident_gate()
    assert gate.skip(short) == "pdflip_d_park_first"


def test_admission_passes_the_replicated_phase_seats():
    src = inspect.getsource(rt.admission)
    assert "seats=_phase_seat_n(sched)" in src
    sched = types.SimpleNamespace(pdflip_d_phase_seats=ds.phase_seats(2, 4, cap=6))
    assert rt._phase_seat_n(sched) == 6
    assert rt._phase_seat_n(types.SimpleNamespace()) is None


# ---- SC: the settle clock ----------------------------------------------------------

def _wake(req, now):
    """The wake's parked branch (scheduler._pdflip_release_dormant_hold)."""
    req._1471_since = sw.settle_since_for_wake(req, now)
    sw.reset_for_wake(req)


def _tick_lapsed(req, now, state):
    """The settle tick's lapse (scheduler._pdflip_post_wake_settle_tick)."""
    lapsed = now - float(req._1471_since) >= SETTLE_S
    if sw.carry_holds_lapse(req, state):
        lapsed = False
    return lapsed


def _phases_until_lapse(d_phase_s=17.0, p_phase_s=17.0, wakes=12):
    """pdflip-8-82: short at every wake, folded at every D->P park."""
    req = types.SimpleNamespace(rid="pdflip-8-82", _1471_short=True, _pdflip_settled_wake=3)
    t = 0.0
    for k in range(1, wakes + 1):
        _wake(req, t)
        # this wake's own read is in flight first, then answers short
        if _tick_lapsed(req, t + 0.5, "reading"):
            return k
        for dt in (2.5, d_phase_s - 0.1):
            if _tick_lapsed(req, t + dt, "wait"):
                return k
        sw.note_fold(req)  # park_running: settle-folded
        t += d_phase_s + p_phase_s
    return None


def test_y9nf4_folded_short_read_lapses_at_the_second_wake():
    assert _phases_until_lapse() == 2


def test_switch_off_restamps_every_wake_as_before(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_SETTLE_CLOCK_CARRY", "0")
    assert _phases_until_lapse() is None  # 15-19 s phases never reach 20 s


def test_the_wakes_own_read_is_awaited_once():
    req = types.SimpleNamespace(rid="r", _1471_short=True, _1471_since=0.0, _pdflip_settled_wake=1)
    sw.note_fold(req)
    _wake(req, 34.0)
    assert req._1471_since == 0.0
    assert not _tick_lapsed(req, 34.5, "reading")  # the wake read is in flight
    assert _tick_lapsed(req, 36.0, "wait")         # answered short: lapsed
    assert _tick_lapsed(req, 38.0, "reading")      # a later 2-s re-read never holds it again


def test_a_carry_is_stale_once_the_request_was_released():
    req = types.SimpleNamespace(rid="r", _1471_short=True, _1471_since=0.0, _pdflip_settled_wake=1)
    sw.note_fold(req)
    req._pdflip_settled_wake = 2  # released at the next wake (complete), ran, parked again
    assert sw.settle_since_for_wake(req, 100.0) == 100.0
    assert not getattr(req, sw.CARRIED_ATTR)


def test_an_unfolded_request_starts_the_bound_at_each_wake_as_before():
    req = types.SimpleNamespace(rid="x", _1471_since=3.0)
    assert sw.settle_since_for_wake(req, 50.0) == 50.0


def test_wiring_fold_and_tick():
    psrc = inspect.getsource(rt.park_running) if hasattr(rt, "park_running") else ""
    rsrc = open(os.path.join(_SRT, "pdflip", "d_park_runtime.py")).read()
    assert "_sw_sc.note_fold(req)" in (psrc or rsrc)
    ssrc = open(os.path.join(_SRT, "managers", "scheduler.py")).read()
    assert "if _sw_nw.carry_holds_lapse(req, state):" in ssrc
    assert "_r._1471_since = _sw_nw.settle_since_for_wake(_r, _now)" in ssrc
