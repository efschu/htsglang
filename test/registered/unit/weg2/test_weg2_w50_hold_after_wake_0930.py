"""DRAIN-W50 (z30y12, 30.09. 23:35:46-23:36:11, D->P flip epoch 51):
``drain+quiesce=19567 ms`` -- the quiesce polled /flush_cache 576 times over
19.2 s while D decoded ONE request at bs=1 that the front counted as held.

The chain on D (TP0, all three ranks alike):

* 23:35:30 ``WEG2-D-PARK hold: 6 parked request(s) at the head of the dormant
  hold`` -- ``hold_parked`` set ``_weg2_d_park_slept = True`` and emptied
  ``weg2_d_parked`` into the #1443 hold.
* 23:35:44 wake; the hold releases the six. ``park_tick`` returns early on the
  empty parked list, so the mark is never cleared.
* 23:35:46 ``WEG2 W50-REROUTE rid=weg2-35-112 ... hold=new`` (keep_on_d parks
  it for P) and in the SAME pass ``WEG2-D-PARK requeue (awake): 1 parked
  request(s) at the queue head ['weg2-35-112']`` -- the stale mark made the
  awake re-queue due at once.
* 23:35:50 the re-queued request is admitted (X-GATE uncached=150), extends
  and decodes to its natural end; front ``WEG2-SERVED ... rid=weg2-35-112``
  23:36:09.640, ``WEG2-QUIESCE-FAST group=D polls=576 ms=19168`` 23:36:09.809.

Every first W50 midstream hold after a wake in that boot was re-queued at once
(weg2-26-73, weg2-34-110, weg2-35-112, weg2-52-159, weg2-62-181,
weg2-64-186); the second one of a phase (weg2-62-182) held, because the first
re-queue had cleared the mark. Red on df1f81516c, green with the fix
(hold_parked marks "slept" only when the list stays parked).
"""
from __future__ import annotations

import os
import sys
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import d_park_runtime as rt  # noqa: E402
from sglang.srt.weg2 import d_seats as ds  # noqa: E402
from sglang.srt.weg2 import resume_via_p as rvp  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_weg2_park_wait_h91c2 import FakeClock, _park, _Sched  # noqa: E402


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.delenv(ds.PARK_ENV, raising=False)
    monkeypatch.delenv(rvp.ENV, raising=False)
    monkeypatch.setenv(rvp.ENV_OPEN_STREAM, "0")
    monkeypatch.setenv(ds.AWAKE_REQUEUE_ENV, "30")
    c = FakeClock()
    monkeypatch.setattr(rt, "time", c)
    monkeypatch.setattr(rvp, "time", c)
    return c


def _req(rid, seq, out=300):
    r = types.SimpleNamespace(
        rid=rid, kv_arrival_seq=seq, stream=True, origin_input_ids=list(range(200)),
        output_ids=list(range(1000, 1000 + out)), is_fast_lane=False, spill_class=None,
    )
    r.full_untruncated_fill_ids = r.origin_input_ids + r.output_ids
    return r


def _sleep_and_wake(s, *, hold_armed):
    """D->P park + sleep, then the P->D wake with the hold released."""
    _park(s)
    s.weg2_dormant = True
    n = rt.hold_parked(s, hold_armed=hold_armed)
    s.weg2_dormant = False
    if hold_armed:
        s.waiting_queue = list(getattr(s, "weg2_dormant_hold", []) or [])
        s.weg2_dormant_hold = []
    return n


def _w50_hold(s, req):
    """The scheduler's W31 answer for a streamed request: it leaves the queue
    and keep_on_d parks it for P (resume_via_p.keep_on_d)."""
    s.waiting_queue = [q for q in s.waiting_queue if q is not req]
    rvp.keep_on_d(s, req, 53653, 12288)


def test_first_w50_hold_after_an_armed_wake_stays_parked(env):
    """Red on df1f81516c: park_tick re-queued weg2-35-112 in the very pass of
    its W50 hold (the sleep's stale mark) -- D decoded it for 19.2 s while the
    front, told it was held, waited in quiesce."""
    a, b = _req("weg2-35-112", 1), _req("weg2-40-127", 2)
    s = _Sched(running=[a, b])
    assert _sleep_and_wake(s, hold_armed=True) == 2
    assert s.weg2_d_parked == [] and [r.rid for r in s.waiting_queue] == ["weg2-35-112", "weg2-40-127"]
    env.t += 2.2                                          # the X gate refuses the resumed one
    _w50_hold(s, a)
    assert s.weg2_d_parked == [a] and ds.park_site(a) == ds.SITE_FLIP
    env.t += 0.02                                         # the next scheduler pass
    assert rt.park_tick(s) == 0, "the W50 hold lapsed in the pass after it was taken"
    assert s.weg2_d_parked == [a] and a not in s.waiting_queue
    env.t += 29.0
    assert rt.park_tick(s) == 0
    env.t += 1.0                                          # the 30 s net still holds
    assert rt.park_tick(s) == 1 and s.waiting_queue[0] is a


def test_second_w50_hold_of_the_phase_behaves_like_the_first(env):
    """Metal: weg2-62-181 was re-queued at once, weg2-62-182 (same phase) held.
    With the fix both hold."""
    a, b = _req("weg2-62-181", 1), _req("weg2-62-182", 2)
    s = _Sched(running=[a, b])
    _sleep_and_wake(s, hold_armed=True)
    for r in (a, b):
        env.t += 1.0
        _w50_hold(s, r)
        env.t += 0.02
        assert rt.park_tick(s) == 0
    assert s.weg2_d_parked == [a, b] and s.waiting_queue == []


def test_unarmed_sleep_still_requeues_the_park_on_the_first_awake_pass(env):
    """The mark's purpose (park_tick docstring): a sleep whose hold was NOT
    armed leaves the list parked, and the first awake pass re-queues it at
    once, not after the 30 s net."""
    a, b = _req("r1", 1), _req("r2", 2)
    s = _Sched(running=[a, b])
    assert _sleep_and_wake(s, hold_armed=False) == 0
    assert [r.rid for r in s.weg2_d_parked] == ["r1", "r2"]
    env.t += 0.02
    assert rt.park_tick(s) == 2 and s.weg2_d_parked == []
    # and the requeue closed the mark: a later W50 hold in this phase holds
    env.t += 1.0
    _w50_hold(s, a)
    env.t += 0.02
    assert rt.park_tick(s) == 0 and s.weg2_d_parked == [a]
