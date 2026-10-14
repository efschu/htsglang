"""NF arena fix 2 (14.10.2026, base f4afb10f28): the wake's hold reads pin the
arena only for the requests that get a seat next.

THE MEASUREMENT (nf-arena-voll-analyse-1014 section 3.4, D.log 03:06-03:07,
``park_l3.py:189-235``): ``#248f WAKE-READ CAPACITY-WAIT in_flight_pages=6100 /
5809 / 5746 ...`` of 6485 (231 such lines in the boot, maximum 6466 at
D.log:268166). The #248f gate holds hold reads up to the WHOLE slot count -- every
waiting request's pages are pinned by its reader reference although only
``max_running_requests`` of them can run. D's own retain/park backup of the flip
(1750 pages, ``ARENA-DROP need=1750 freed=0``) found no slot, pages were
PARK-LOST, L3 fills ended at the first slot without a claim.

THE FIX: beside the page gate, a hold read is issued only for as many requests as
D has free seats (``max_running_requests`` minus the running requests minus the
flip-parked requests waiting in the queue), in hold (arrival) order; the rest
wait named in the #1471 settle and read when a seat frees. No reserve.
"""

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.environ import envs
from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.weg2 import d_seats, park_l3

S = sched_mod.Scheduler
PAGE = 64
SLOTS = 6485
SEATS = 6


@pytest.fixture(autouse=True)
def _group_d(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    with envs.SGLANG_WEG2_ENABLE_PARK_L3.override(True):
        yield


def _req(rid, pages):
    r = types.SimpleNamespace(rid=rid, origin_input_ids=[0] * (pages * PAGE), output_ids=[])
    setattr(r, park_l3.DEFER_ATTR, True)
    return r


def _running(n):
    return types.SimpleNamespace(reqs=[types.SimpleNamespace(rid="run-%d" % i) for i in range(n)])


def _sched(seats=SEATS, running=0, read_states=None):
    h = types.SimpleNamespace()
    h.issued = []
    h.waiting_queue = []
    h.weg2_dormant_hold = []
    h.weg2_dormant = False
    h.page_size = 1
    h.max_running_requests = seats
    h.running_batch = _running(running)
    h.cur_batch = None
    h.last_batch = None
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


def _hold(n, pages=600):
    """n held requests in arrival order; 10 x 600 pages = 6000 of 6485 slots:
    the page gate alone lets ALL of them read (the 92 % pin of the metal)."""
    return [_req("weg2-%d-%d" % (i, i + 1), pages) for i in range(n)]


def test_the_page_gate_alone_pins_the_arena_for_every_waiter():
    """RED on f4afb10f28 (= the switch off): 10 held requests, 6 seats -- all 10
    reads are issued, 6000 of 6485 pages are pinned (92.5 %)."""
    h = _sched()
    with envs.SGLANG_WEG2_WAKE_READ_SEAT_GATE.override(False):
        out = park_l3.issue_deferred_reads(h, _hold(10))
    assert len(out) == 10
    assert sum(park_l3.read_pages(h, r) for r in out) == 6000


def test_only_the_next_seats_read():
    """GREEN: the six oldest read (3600 pages), the four younger wait named."""
    h = _sched()
    hold = _hold(10)
    out = park_l3.issue_deferred_reads(h, hold)
    assert [r.rid for r in out] == [r.rid for r in hold[:6]], "the seats go in arrival order"
    assert h.issued == [r.rid for r in hold[:6]]
    assert all(park_l3.capacity_waiting(r) and park_l3.deferred(r) for r in hold[6:])
    assert sum(park_l3.read_pages(h, r) for r in out) == 3600


def test_running_requests_hold_seats():
    h = _sched(running=4)
    out = park_l3.issue_deferred_reads(h, _hold(5))
    assert len(out) == 2, "6 seats - 4 running = 2 free"


def test_flip_parked_requests_in_the_queue_hold_their_seats():
    h = _sched()
    parked = [types.SimpleNamespace(rid="park-%d" % i) for i in range(2)]
    for p in parked:
        d_seats.mark_parked(p, d_seats.SITE_FLIP, epoch=None, now=0.0)
    h.waiting_queue = list(parked)
    out = park_l3.issue_deferred_reads(h, _hold(8))
    assert len(out) == 4, "6 seats - 2 flip-parked resumes (they go first) = 4"


def test_the_current_and_last_batch_count_until_the_extend_merged():
    """A request admitted this pass sits in cur_batch, not yet in running_batch."""
    h = _sched(running=3)
    h.cur_batch = types.SimpleNamespace(reqs=[types.SimpleNamespace(rid="new-1"), h.running_batch.reqs[0]])
    assert park_l3.seat_room(h) == SEATS - 3 - 1, "the merged one counts once, the extending one once"


def test_the_first_read_always_goes_when_a_seat_is_free_and_a_huge_read_reads_alone():
    h = _sched()
    hold = [_req("huge", 9000), _req("next", 5)]
    out = park_l3.issue_deferred_reads(h, hold)
    assert [r.rid for r in out] == ["huge"], "page gate as before: a lone read larger than the arena"
    assert park_l3.capacity_waiting(hold[1])


def test_no_free_seat_no_read():
    h = _sched(running=SEATS)
    hold = _hold(3)
    out = park_l3.issue_deferred_reads(h, hold)
    assert out == [] and h.issued == []
    assert all(park_l3.capacity_waiting(r) for r in hold)


def test_no_seat_count_known_is_the_page_gate_alone():
    """A fixture / stock resume without max_running_requests: nothing changes."""
    h = _sched(seats=0)
    del h.max_running_requests
    assert park_l3.seat_room(h) is None
    out = park_l3.issue_deferred_reads(h, _hold(10))
    assert len(out) == 10


def test_not_group_d_no_gate(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    assert park_l3.seat_room(_sched()) is None


def test_the_capacity_wait_line_names_the_seats(caplog):
    h = _sched()
    with caplog.at_level(logging.INFO, logger=park_l3.__name__):
        park_l3.issue_deferred_reads(h, _hold(8))
    line = [r.getMessage() for r in caplog.records if park_l3.CAPWAIT_MARK in r.getMessage()][0]
    assert "seat_room=6 reading=6" in line


def test_the_settle_issues_a_waiter_when_a_seat_frees_not_when_a_read_is_admitted(caplog):
    """The scheduler wiring (the settle tick, ``issue_capacity_waiters``). 8
    held, 6 seats: the wake reads 6 and parks 2 as waiters; the six reads
    completed and joined the waiting queue (the post-wake state; the wake's own
    release vote is the cohort's business, not this test's). Admitted, a read
    leaves the queue but runs (seat taken): the waiters stay. A running request
    that finishes frees the seat: ONE waiter reads, in arrival order, the
    younger one still waits."""
    hold = _hold(8)
    h = _sched()
    park_l3.issue_deferred_reads(h, hold)
    assert h.issued == [r.rid for r in hold[:6]]
    assert [r.rid for r in hold if park_l3.capacity_waiting(r)] == [hold[6].rid, hold[7].rid]
    h.waiting_queue = list(hold[:6])           # the six reads completed and were released to the queue
    h.weg2_post_wake_settle = list(hold[6:])   # the two waiters are parked in the settle
    h._weg2_post_wake_settle_tick()
    assert h.issued == [r.rid for r in hold[:6]], "6 seats claimed by the 6 queued reads: no waiter reads"
    first = h.waiting_queue.pop(0)             # admitted: it leaves the queue ...
    h.running_batch.reqs.append(first)         # ... and runs, its seat is taken
    h._weg2_post_wake_settle_tick()
    assert h.issued == [r.rid for r in hold[:6]], "admitted = still a seat taken: the waiters wait"
    h.running_batch.reqs.remove(first)         # a running request finishes -> a seat frees
    with caplog.at_level(logging.INFO, logger=park_l3.__name__):
        h._weg2_post_wake_settle_tick()
    assert h.issued == [r.rid for r in hold[:6]] + [hold[6].rid], "the OLDEST waiter reads, not both"
    assert park_l3.capacity_waiting(hold[7])
    assert any(park_l3.CAPISSUE_MARK in r.getMessage() and "seat_room=" in r.getMessage()
               for r in caplog.records)


def test_the_decision_is_the_same_on_every_rank():
    """Three TP ranks, the same schedule: the same reads are issued."""
    ranks = [_sched(running=1) for _ in range(3)]
    holds = [_hold(9) for _ in range(3)]
    for h, hold in zip(ranks, holds):
        park_l3.issue_deferred_reads(h, hold)
    assert ranks[0].issued == ranks[1].issued == ranks[2].issued
    assert len(ranks[0].issued) == SEATS - 1


def test_the_seat_room_reads_no_rank_identity_and_no_pool_state():
    import ast
    import inspect
    import textwrap

    seen = set()
    for fn in (park_l3.seat_room, park_l3._no_seat_yet, park_l3._in_flight):
        for node in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(fn)))):
            if isinstance(node, ast.Attribute):
                seen.add(node.attr)
    banned = {"tp_rank", "rank", "attn_tp_rank", "dp_rank", "local_rank", "free_pages", "available_size",
              "evictable_size"}
    assert not (seen & banned), sorted(seen & banned)
