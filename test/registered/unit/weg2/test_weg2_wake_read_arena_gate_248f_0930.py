"""#248f (30.09., NF y4b ep18, D TP0 03:51:37-42): the #248 hold reads of a
wake were issued all at once -- weg2-0-4 (1412 pages), weg2-14-27 (1728) and
weg2-16-29 (3841) = 6981 pages against 6485 KV arena slots. 16-29's L3 fills
found the arena full, ``#248e ORDERED-EVICT site=l3fill`` took 449 of its OWN
kept pages, 33 fills were refused, 4 re-reads, held 4.5 s after the wake.

Now the reads are issued in arrival order only while they fit in the arena
TOGETHER (slot count, no reserve); the first one that would overrun it and
every younger one wait by name in the #1471 settle and are issued as the
older reads leave it."""

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.environ import envs
from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.weg2 import park_l3

S = sched_mod.Scheduler
PAGE = 64
SLOTS = 6485


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    with envs.SGLANG_WEG2_ENABLE_PARK_L3.override(True):
        yield


def _req(rid, pages, tokens=None):
    n = tokens if tokens is not None else pages * PAGE
    r = types.SimpleNamespace(rid=rid, origin_input_ids=[0] * n, output_ids=[])
    setattr(r, park_l3.DEFER_ATTR, True)
    return r


def _sched(read_states=None):
    h = types.SimpleNamespace()
    h.issued = []
    h.waiting_queue = []
    h.weg2_dormant_hold = []
    h.weg2_dormant = False
    h.page_size = 1        # the D argv's --page-size 1: the arena page is the controller's
    h.WEG2_POST_WAKE_SETTLE_S = S.WEG2_POST_WAKE_SETTLE_S
    h.ps = types.SimpleNamespace(tp_size=1)
    h.tree_cache = types.SimpleNamespace(cache_controller=types.SimpleNamespace(
        weg2_hold_rids=set(), page_size=PAGE,
        mem_pool_host=types.SimpleNamespace(arena=types.SimpleNamespace(slots=SLOTS))))

    def _prefetch(req):
        h.issued.append(req.rid)
        return "issued"

    h._prefetch_kvcache = _prefetch
    states = read_states if read_states is not None else {}
    h._weg2_refetch_one = lambda req, now, allow_reissue=True: states.get(req.rid, "reading")
    for n in ("_weg2_release_dormant_hold", "_weg2_post_wake_settle_tick", "_weg2_group_min_flags"):
        setattr(h, n, types.MethodType(getattr(S, n), h))
    return h


def _ep18():
    # hold (arrival) order as in the metal's read_order
    return [_req("weg2-0-4", 1412), _req("weg2-14-27", 1728), _req("weg2-16-29", 3841)]


def test_y4b_ep18_the_youngest_read_waits_for_arena_room(caplog):
    h = _sched()
    hold = _ep18()
    with caplog.at_level(logging.INFO, logger=park_l3.__name__):
        out = park_l3.issue_deferred_reads(h, hold)
    assert [r.rid for r in out] == ["weg2-0-4", "weg2-14-27"]      # base: all three
    assert h.issued == ["weg2-0-4", "weg2-14-27"]
    assert park_l3.capacity_waiting(hold[2]) and park_l3.deferred(hold[2])
    assert any(park_l3.CAPWAIT_MARK in r.getMessage() and "weg2-16-29" in r.getMessage()
               for r in caplog.records)


def test_arrival_order_no_younger_read_overtakes():
    h = _sched()
    hold = [_req("old", 3000), _req("big", 4000), _req("small", 10)]
    park_l3.issue_deferred_reads(h, hold)
    assert h.issued == ["old"]                     # "small" would fit, but "big" is older
    assert park_l3.capacity_waiting(hold[1]) and park_l3.capacity_waiting(hold[2])


def test_a_lone_read_larger_than_the_arena_reads_as_before():
    h = _sched()
    hold = [_req("huge", 9000), _req("next", 5)]
    park_l3.issue_deferred_reads(h, hold)
    assert h.issued == ["huge"]                    # the first read always goes
    assert park_l3.capacity_waiting(hold[1])


def test_the_switch_off_issues_every_read_as_before():
    h = _sched()
    with envs.SGLANG_WEG2_ENABLE_WAKE_READ_ARENA_GATE.override(False):
        park_l3.issue_deferred_reads(h, _ep18())
    assert h.issued == ["weg2-0-4", "weg2-14-27", "weg2-16-29"]


def test_the_wake_parks_the_waiter_and_the_settle_issues_it_when_room_frees(caplog):
    """The scheduler wiring: the wake parks the waiting read in the settle
    (never queued unread); a released read keeps the arena until its admission
    loads it (queued = still holding), and the settle tick issues the waiter
    once that room is free, in arrival order."""
    states = {"weg2-0-4": "complete", "weg2-14-27": "reading", "weg2-16-29": "reading"}
    h = _sched(states)
    h.weg2_dormant_hold = _ep18()
    h.tree_cache.cache_controller.weg2_hold_rids = {"weg2-0-4", "weg2-14-27", "weg2-16-29"}
    n = h._weg2_release_dormant_hold()
    assert n == 1 and [r.rid for r in h.waiting_queue] == ["weg2-0-4"]
    assert [r.rid for r in h.weg2_post_wake_settle] == ["weg2-14-27", "weg2-16-29"]
    assert h.issued == ["weg2-0-4", "weg2-14-27"]
    # 0-4 released but not admitted (its arena references stand until the
    # load-back): 1412 + 1728 + 3841 = 6981 > 6485 -> 16-29 still waits
    h._weg2_post_wake_settle_tick()
    assert h.issued == ["weg2-0-4", "weg2-14-27"]
    # 0-4 admitted (left the queue): 1728 + 3841 = 5569 <= 6485 -> 16-29 reads now
    h.waiting_queue.clear()
    with caplog.at_level(logging.INFO, logger=park_l3.__name__):
        h._weg2_post_wake_settle_tick()
    assert h.issued == ["weg2-0-4", "weg2-14-27", "weg2-16-29"]
    assert any(park_l3.CAPISSUE_MARK in r.getMessage() for r in caplog.records)
    assert [r.rid for r in h.weg2_post_wake_settle] == ["weg2-14-27", "weg2-16-29"]


def _y4b_0358():
    # y4b D 03:58:53, the wake whose reads ended in the client W50 of
    # weg2-32-72 / 33-73: hold order and token lengths from the metal
    return [_req("weg2-30-68", 0, 116860), _req("weg2-32-70", 0, 131100),
            _req("weg2-32-71", 0, 131137), _req("weg2-32-72", 0, 95137),
            _req("weg2-33-73", 0, 95294)]


def test_y4b_0358_the_two_youngest_wait_instead_of_reading_short(caplog):
    """Base: all five issued at once (5927 + 2976 pages > 6485): 32-72 read
    33728 of 95104 (FETCH CAP lost=85, #248e victims=weg2-32-72), W88 -> W31
    -> W50 twice -> the client's Weg2TpPrefillExceeded."""
    h = _sched()
    hold = _y4b_0358()
    with caplog.at_level(logging.INFO, logger=park_l3.__name__):
        park_l3.issue_deferred_reads(h, hold)
    assert h.issued == ["weg2-30-68", "weg2-32-70", "weg2-32-71"]
    assert [r.rid for r in hold if park_l3.capacity_waiting(r)] == ["weg2-32-72", "weg2-33-73"]
    msg = [r.getMessage() for r in caplog.records if park_l3.CAPWAIT_MARK in r.getMessage()]
    assert msg and "weg2-32-72" in msg[0] and "arena_slots=6485" in msg[0]
    # the three older ones released, still queued (not loaded): nothing reads
    h.waiting_queue = hold[:3]
    for r in hold[:3]:
        setattr(r, park_l3.ISSUED_ATTR, False)          # after_release
    assert park_l3.issue_capacity_waiters(h, hold[3:]) == []
    # admitted: both read in full, oldest first
    h.waiting_queue = []
    out = park_l3.issue_capacity_waiters(h, hold[3:])
    assert [r.rid for r in out] == ["weg2-32-72", "weg2-33-73"]


def test_a_queued_request_of_an_earlier_wake_does_not_count():
    h = _sched()
    old = _req("old", 6000)
    setattr(old, park_l3.PAGES_ATTR, 6000)
    setattr(old, park_l3.WAKE_ATTR, 1)
    h._weg2_wake_seq = 2
    h.waiting_queue = [old]
    hold = [_req("a", 3000), _req("b", 3000)]
    park_l3.issue_deferred_reads(h, hold)
    assert h.issued == ["a", "b"]


def test_the_settle_keeps_a_waiter_unread_and_unreleased_while_no_room():
    states = {"a": "reading", "w": "complete"}   # "w" never ran: its verdict must not release it
    h = _sched(states)
    a, w = _req("a", 5000), _req("w", 3000)
    park_l3.issue_deferred_reads(h, [a, w])
    assert h.issued == ["a"] and park_l3.capacity_waiting(w)
    import time
    a._1471_since = w._1471_since = time.monotonic()
    h.weg2_post_wake_settle = [a, w]
    assert h._weg2_post_wake_settle_tick() == 0
    assert h.issued == ["a"]
    assert [r.rid for r in h.weg2_post_wake_settle] == ["a", "w"]
    assert h.waiting_queue == []
