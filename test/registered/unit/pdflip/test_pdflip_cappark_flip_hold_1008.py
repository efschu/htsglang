"""CAPPARK-FLIP-HOLD (08.10., NF dauer10081045, int17 49873f9af8, D->P flip
epoch 4->5): the #248h capacity re-queue ran parked requests on D while the
front's D->P flip waited for D to go idle.

Metal: pdflip-0-9 / -0-10 / -0-12 were #248h capacity-parked on D (their store
reads ended short on D's own arena) and sat in the #1471 post-wake settle at
the P->D wake of epoch 4. ``PARK-IMMEDIATE FIRED`` (10:53:16.621, cause
over-x) sent ``park_running``; D folded the settle into its park list
(``settle-folded=['pdflip-0-9','pdflip-0-10','pdflip-0-12','pdflip-2-28']``) and the
front, told they were parked, began the D->P flip (10:53:17.170). In the very
next pass ``park_tick`` -- the awake re-queue not due (30 s) -- fell through to
the #248h capacity re-queue: their capacity stamp was older than the 2-s
re-read cadence and the arena had room, so ``#248h PDFLIP-D-PARK requeue
(capacity): 3 parked request(s) at the queue head`` (10:53:17). D prefilled
them (#cached 43612 / 128448 / 82048) and decoded them to their end
(10:53:45), the quiesce read D busy for 28180 ms (PDFLIP-QUIESCE-FAST polls=447)
and the waiting P-only request (est_uncached 88703) waited for it. The same
class as NF y9n 10032307 (front.py park_lapse_race: "quiesce then waited 28 s
for pdflip-10-42 (D's #248h capacity requeue ran it to its end)").

The rule: while a flip park is open (from ``park_running`` to the sleep's
``hold_parked`` or the awake re-queue), the capacity re-queue does not run --
the parked requests ride the flip like every other parked request, and the
wake's hold read (or the awake re-queue) takes them up again. Outside a flip
park the capacity re-queue runs as before. ``FLLIPER_PDFLIP_ENABLE_CAPPARK_FLIP_HOLD=0``
restores the old behaviour.
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import d_park_runtime as rt  # noqa: E402
from flliper.srt.pdflip import d_seats as ds  # noqa: E402
from flliper.srt.pdflip import resume_via_p as rvp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_pdflip_park_wait_h91c2 import FakeClock, _req, _Sched  # noqa: E402


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.delenv(ds.PARK_ENV, raising=False)
    monkeypatch.delenv(ds.RESUME_MARGIN_ENV, raising=False)
    monkeypatch.setenv(ds.AWAKE_REQUEUE_ENV, "30")
    # no arena bound on this fake scheduler: "the arena has room" (metal: it had)


@pytest.fixture
def clock(monkeypatch):
    c = FakeClock()
    monkeypatch.setattr(rt, "time", c)
    return c


def _cappark(req):
    """A #248h capacity park older than the 2-s re-read cadence (rvp's own
    monotonic clock, as on metal: stamped seconds before the wake)."""
    now = time.monotonic()
    setattr(req, rvp.CAPPARK_SINCE_ATTR, now - 30.0)
    setattr(req, rvp.CAPPARK_AT_ATTR, now - 10.0)
    return req


def _park_immediate(s, epoch=4):
    from flliper.srt.managers.io_struct import PdFlipParkRunningReqInput

    return rt.park_running(s, PdFlipParkRunningReqInput(epoch=epoch, reason="immediate-over-x"),
                           late_hold_armed=True)


def _epoch4_constellation():
    """Two running decodes, three capacity-parked requests in the #1471 settle."""
    run = [_req("pdflip-0-6", 1), _req("pdflip-0-8", 2)]
    s = _Sched(running=run)
    s.pdflip_post_wake_settle = [_cappark(_req("pdflip-0-9", 3)), _cappark(_req("pdflip-0-10", 4)),
                               _cappark(_req("pdflip-0-12", 5))]
    return s


def test_a_flip_park_holds_the_capacity_parked_requests_until_the_sleep(clock):
    """Red on 49873f9af8: park_tick re-queued pdflip-0-9/-0-10/-0-12 right
    after the park (#248h requeue (capacity)), D ran them to their end while
    the front's quiesce waited (28 s on metal)."""
    s = _epoch4_constellation()
    out = _park_immediate(s)
    assert sorted(out.parked) == ["pdflip-0-6", "pdflip-0-8"]
    assert sorted(out.held) == ["pdflip-0-10", "pdflip-0-12", "pdflip-0-9"]
    for _ in range(3):                      # the passes during the D->P quiesce
        clock.t += 0.5
        assert rt.park_tick(s) == 0
    assert s.waiting_queue == [], "nothing of the park runs on D during the flip"
    # the sleep holds all five, the capacity park rides along to the wake's read
    s.pdflip_dormant = True
    assert rt.hold_parked(s, hold_armed=True) == 5
    assert sorted(r.rid for r in s.pdflip_dormant_hold) == [
        "pdflip-0-10", "pdflip-0-12", "pdflip-0-6", "pdflip-0-8", "pdflip-0-9"]
    assert all(getattr(r, rvp.CAPPARK_AT_ATTR) is not None
               for r in s.pdflip_dormant_hold if r.rid in ("pdflip-0-9", "pdflip-0-10", "pdflip-0-12"))


def test_epoch51_a_capacity_park_already_in_the_park_list_and_a_late_hand_off(clock):
    """Same boot, D->P flip epoch 50->51 (11:24:12.540 PARK-IMMEDIATE over-x,
    pdflip-50-253): pdflip-50-247 sat in D's park list as a #248h capacity park
    BEFORE the park (not a settle fold), four decodes ran, the hand-off
    pdflip-50-254 reached D after the park (late hold, front: in_flight_held).
    Red on 49873f9af8: 11:24:14 ``#248h requeue (capacity): ['pdflip-50-247']``,
    its re-read held D not idle (hicache_prefetch) until 11:24:23 and the
    quiesce took 11175 ms. With the rule the late hand-off and the capacity
    park both stay behind the park and ride the sleep."""
    from test_pdflip_park_wait_h91c3 import _IntakeSched

    run = [_req("pdflip-50-240", 1), _req("pdflip-50-245", 2), _req("pdflip-50-249", 3), _req("pdflip-50-251", 4)]
    s = _IntakeSched(running=run)
    cap = _cappark(_req("pdflip-50-247", 5))
    rt.parked_list(s).append(cap)           # resume_via_p.park_for_capacity, before the park
    ds.mark_parked(cap, ds.SITE_FLIP, epoch=None, now=clock.t)
    out = _park_immediate(s, epoch=50)
    assert "pdflip-50-247" in out.parked and out.late_hold is True
    s._add_request_to_queue(_req("pdflip-50-254", 6))   # the hand-off lands after the park
    for _ in range(3):
        clock.t += 0.5
        assert rt.park_tick(s) == 0
    assert s.waiting_queue == []
    s.pdflip_dormant = True
    assert rt.hold_parked(s, hold_armed=True) == 6
    assert {"pdflip-50-247", "pdflip-50-254"} <= {r.rid for r in s.pdflip_dormant_hold}


def test_after_the_flip_the_capacity_requeue_runs_again(clock):
    """The rule only covers the flip-park window: after the sleep (and wake),
    a request the next D phase capacity-parks is re-queued as before."""
    s = _epoch4_constellation()
    _park_immediate(s)
    s.pdflip_dormant = True
    rt.hold_parked(s, hold_armed=True)
    s.pdflip_dormant = False                  # the wake; the hold read took them up
    s.pdflip_dormant_hold = []
    later = _cappark(_req("pdflip-0-9", 3))
    rt.parked_list(s).append(later)         # resume_via_p.park_for_capacity's list
    ds.mark_parked(later, ds.SITE_FLIP, epoch=None, now=clock.t)
    assert rt.park_tick(s) == 1
    assert s.waiting_queue == [later] and s.pdflip_d_parked == []


def test_a_park_whose_sleep_never_comes_releases_everything_by_the_awake_bound(clock):
    """Nothing is lost when the flip does not follow the park: the awake
    re-queue (30 s) takes the whole park, the folded settle requests go back
    to the settle, and the flip-park window closes (capacity rule live again)."""
    s = _epoch4_constellation()
    _park_immediate(s)
    clock.t += 29.9
    assert rt.park_tick(s) == 0
    clock.t += 0.1
    assert rt.park_tick(s) == 5
    assert sorted(r.rid for r in s.waiting_queue) == ["pdflip-0-6", "pdflip-0-8"]
    assert sorted(r.rid for r in s.pdflip_post_wake_settle) == ["pdflip-0-10", "pdflip-0-12", "pdflip-0-9"]
    later = _cappark(_req("pdflip-0-99", 9))
    rt.parked_list(s).append(later)
    ds.mark_parked(later, ds.SITE_FLIP, epoch=None, now=clock.t)
    assert rt.park_tick(s) == 1


def test_the_switch_off_restores_the_capacity_requeue_inside_the_park(clock):
    s = _epoch4_constellation()
    _park_immediate(s)
    with envs.FLLIPER_PDFLIP_ENABLE_CAPPARK_FLIP_HOLD.override(False):
        clock.t += 0.5
        assert rt.park_tick(s) == 3
    assert sorted(r.rid for r in s.waiting_queue) == ["pdflip-0-10", "pdflip-0-12", "pdflip-0-9"]
