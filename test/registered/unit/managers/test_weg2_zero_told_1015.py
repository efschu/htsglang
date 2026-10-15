"""T2 ZERO-TOLD (NF h16p 226aad7105, boots ..._1015_115748 / _122948 / _124245, P.log): the PF group told=0 fallback
re-prefilled what the store held in two classes B5 (anchor acks) does not touch.

(a) TOLD-ACK NO-ROOM -- 12:46:20 weg2-0-3 (PP1 and PP2): the store was read in full on every stage (``loaded=184768``),
    then ``PF TOLD-ACK NO-ROOM loadback_rows=184768 room=162176`` -> ack 0 -> ``TOLD-FALLBACK told=184768 -> 0
    reason=mismatch acks={1: 0}``: 185k re-prefilled. 12:33:29 weg2-0-10: ``188928 / room=129472``, same. The room was
    short because a RUNNING prefill held ~100k rows ("full token usage: 0.38" on that line, 0.06 one batch later): a
    shortage that lasts as long as that request. T2 ROOM-DEFER: the follower holds the ack back while room + the rows
    others hold would suffice; PP0 reads the missing ack as lag.

(b) ``reason=frist`` -- 12:17:37 weg2-33-81 ``told=121344 acks={1: 121344}`` (PP2's ack surfaced at 12:17:43 as ``TOLD-ACK
    LATE own=121344``), 12:52:58 weg2-4-25 ``told=43392 acks={1: 43392}`` after 0.16 s (PP2's ``TOLD-ACK LATE
    own=43392`` at 12:53:02), 12:17:31 weg2-33-82 ``told=24704 acks={1: 24704}``, 12:05:49 weg2-3-28 ``told=119360
    acks={}``. Every late ack named EXACTLY told; PP0 had thrown the prefix away for an ack that was on its way (the
    followers are chain hops behind a PP0 whose passes are 3-7 s while it prefills). T2 WAIT-PROGRESS: the Frist
    stays the bound of an IDLE box; while the box serves (the #699 wedge detector, at its threshold less a margin,
    would not call it wedged) PP0 waits; a hard cap bounds a dead stream.

PP0 decides alone; a follower only holds back or sends what it can load. Rank equality is asserted on the ring.
"""

from __future__ import annotations

import logging
import os
import sys
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _told_ring_pf as R  # noqa: E402

from sglang.srt.managers import weg2_store_told as m  # noqa: E402
from sglang.srt.managers.scheduler_components import invariant_checker  # noqa: E402,F401  (the scheduler imports it)
from sglang.srt.managers import weg2_told_fallback as fb  # noqa: E402

T2_SWITCHES = (fb.ENV_WAIT_PROGRESS, fb.ENV_ROOM_DEFER, fb.ENV_WAIT_HARD_S, fb.ENV_WAIT_MARGIN_S)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in list(os.environ):
        if k.startswith("SGLANG_WEG2_TOLD"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv(fb.ENV_MIN_PASSES, raising=False)


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


# ---------------------------------------------------------------------------
# (b) PP0: the Frist is up, the box serves
# ---------------------------------------------------------------------------


def _pp0(clock, *, serving=True, prefill_age=2.0, first_age=120.0, running=0):
    """PP0 double: the wedge detector's inputs (queue, running set, the two progress clocks)."""
    s = types.SimpleNamespace(
        ps=types.SimpleNamespace(pp_size=3, pp_rank=0),
        waiting_queue=[object()],
        running_batch=types.SimpleNamespace(reqs=[object()] * running),
        last_first_token_progress_time=clock() - first_age,
        last_prefill_progress_time=clock() - (prefill_age if serving else 600.0),
    )
    return s


def _open(s, rid, told, acks, *, published_at=0.0, own_read_s=0.0, window_s=1.0, anchors=None):
    fb.pp0_open(s, rid, told, published_at=published_at, own_read_s=own_read_s, window_s=window_s, published_pass=0)
    o = fb._pp0_open_map(s)[rid]
    o.acks.update(acks)
    o.anchors.update(anchors or {})
    return o


# the four frist cases of the log: (rid, told, acks at the Frist, seconds after the read-ahead, own read seconds)
FRIST_CASES = [
    ("weg2-33-81", 121344, {1: 121344}, 9.77, 0.0),
    ("weg2-4-25", 43392, {1: 43392}, 0.16, 25.0),  # PP0's own read took 25 s: deadline = the read-ahead itself
    ("weg2-33-82", 24704, {1: 24704}, 4.29, 0.0),
    ("weg2-3-28", 119360, {}, 4.69, 0.0),
]


@pytest.mark.parametrize("rid,told,acks,after,own_read", FRIST_CASES)
def test_frist_cases_red_without_the_wait(monkeypatch, rid, told, acks, after, own_read):
    """The h16p code (T2 WAIT-PROGRESS off): told=0 at the Frist, whatever the box does."""
    monkeypatch.setenv(fb.ENV_WAIT_PROGRESS, "0")
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    s = _pp0(clk)
    _open(s, rid, told, acks, own_read_s=own_read)
    assert fb.pp0_decide(s, rid, now=after + 6.0, pass_n=10) == (0, fb.REASON_FRIST)


@pytest.mark.parametrize("rid,told,acks,after,own_read", FRIST_CASES)
def test_frist_cases_green_pp0_waits_and_the_late_ack_names_told(monkeypatch, rid, told, acks, after, own_read):
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    s = _pp0(clk)
    o = _open(s, rid, told, acks, own_read_s=own_read)
    # the Frist is up (and more): PP0 keeps the rid, the box serves
    assert fb.pp0_decide(s, rid, now=after + 6.0, pass_n=10) is None
    assert o.waited_since is not None
    # the late ack of the log arrives, equal to told: admitted in full on every rank
    o.acks.update({1: told, 2: told})
    assert fb.pp0_decide(s, rid, now=after + 10.0, pass_n=11) == (told, fb.REASON_ACKS)


def test_an_idle_box_keeps_the_frist(monkeypatch):
    """Nothing serves: the #699 wedge clock runs, the Frist is the bound as before."""
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    s = _pp0(clk, serving=False)
    _open(s, "r", 50000, {1: 50000})
    assert fb.pp0_decide(s, "r", now=100.0, pass_n=10) == (0, fb.REASON_FRIST)


def test_a_box_without_clocks_keeps_the_frist(monkeypatch):
    """No progress clock to read = no proof of progress: the old Frist (the H64/B5 tests' schedulers)."""
    s = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=3, pp_rank=0))
    _open(s, "r", 50000, {})
    assert fb.pp0_decide(s, "r", now=1e9, pass_n=10_000) == (0, fb.REASON_FRIST)


def test_no_wedge_guard_loaded_keeps_the_frist_and_never_imports_in_the_pass(monkeypatch):
    """The first import of the detector (torch & co., seconds) must not happen inside PP0's pass: not loaded = no
    proof of progress = the old Frist (the gloo ring harness measures publish time)."""
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    monkeypatch.delitem(sys.modules, "sglang.srt.managers.scheduler_components.invariant_checker")
    ok, detail = fb.box_serving(_pp0(clk))
    assert ok is False and "not loaded" in detail
    assert "sglang.srt.managers.scheduler_components.invariant_checker" not in sys.modules


def test_the_wedge_margin_is_the_detectors_threshold_less_five_seconds(monkeypatch):
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    # alarm threshold 20 s - margin 5 s = 15 s
    assert fb.box_serving(_pp0(clk, prefill_age=14.0))[0] is True
    assert fb.box_serving(_pp0(clk, prefill_age=16.0))[0] is False
    # a decode round running is serving whatever the clocks say
    assert fb.box_serving(_pp0(clk, serving=False, running=1))[0] is True
    monkeypatch.setenv(fb.ENV_WAIT_MARGIN_S, "0")
    assert fb.box_serving(_pp0(clk, prefill_age=16.0))[0] is True


def test_a_dead_stream_is_bounded_by_the_hard_cap(monkeypatch):
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    s = _pp0(clk)
    _open(s, "r", 50000, {1: 50000}, published_at=100.0)
    assert fb.pp0_decide(s, "r", now=100.0 + 59.0, pass_n=10) is None
    assert fb.pp0_decide(s, "r", now=100.0 + 61.0, pass_n=10) == (0, fb.REASON_FRIST_HARD)
    monkeypatch.setenv(fb.ENV_WAIT_HARD_S, "5")
    assert fb.pp0_decide(s, "r", now=100.0 + 6.0, pass_n=10) == (0, fb.REASON_FRIST_HARD)


def test_a_floor_still_possible_waits_for_the_missing_ack_while_serving(monkeypatch):
    """B5's partial acks: a common anchor exists, the third stage has not acked -- wait, do not answer 0."""
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    s = _pp0(clk)
    o = _open(s, "r", 91712, {1: 91712}, anchors={1: (87296,)})
    assert fb.pp0_decide(s, "r", now=50.0, pass_n=10) is None
    o.acks[2] = 87296
    o.anchors[2] = ()
    assert fb.pp0_decide(s, "r", now=51.0, pass_n=11) == (87296, fb.REASON_FLOOR)


def test_equal_acks_never_wait(monkeypatch):
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    s = _pp0(clk)
    _open(s, "r", 50000, {1: 50000, 2: 50000})
    assert fb.pp0_decide(s, "r", now=0.0, pass_n=0) == (50000, fb.REASON_ACKS)


def test_the_verdict_after_a_wait_is_named(monkeypatch, caplog):
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    s = _pp0(clk)
    o = _open(s, "weg2-4-25", 43392, {1: 43392}, own_read_s=25.0)
    with caplog.at_level(logging.INFO, logger=fb.logger.name):
        assert fb.pp0_decide(s, "weg2-4-25", now=6.0, pass_n=10) is None
        o.acks[2] = 43392
        verdict = fb.pp0_decide(s, "weg2-4-25", now=10.0, pass_n=11)
        fb.pp0_note_verdict(s, "weg2-4-25", 43392, verdict[0], verdict[1], 10.0, 0.0)
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "PF TOLD-WAIT rid=weg2-4-25" in text
    assert "PF TOLD-WAITED rid=weg2-4-25 told=43392 -> 43392 reason=acks" in text
    assert "PF TOLD-FALLBACK" not in text


# ---------------------------------------------------------------------------
# (b) the whole ring: PP2's read is late, PP0 prefills other work
# ---------------------------------------------------------------------------

RID_L = "weg2-4-25"
TOLD_L = 43392


class _ServingStage(R.Stage):
    """PP0 whose box serves: the wedge detector's inputs read healthy."""

    running_batch = types.SimpleNamespace(reqs=[])

    @property
    def last_first_token_progress_time(self):
        return self.tree_cache.clock() - 120.0

    @property
    def last_prefill_progress_time(self):
        return self.tree_cache.clock() - 1.5  # a chunk retired a second ago, every pass


def _late_ring(monkeypatch, serving=True, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv(fb.ENV_FALLBACK, "1")
    monkeypatch.setenv("SGLANG_WEG2_TOLD_ABSOLUTE", "1")
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    monkeypatch.setattr(R, "DT", 0.5)
    # PP0 read for 25 s (PP0 intake -> read-ahead), PP1 1 s, PP2 29 s: PP2's ack comes 4 s after the read-ahead
    reads = {0: {RID_L: 25.0}, 1: {RID_L: 1.0}, 2: {RID_L: 29.0}}
    ring = R.Ring(m, monkeypatch, {RID_L: TOLD_L}, reads)
    monkeypatch.setattr(fb, "_perf", ring.clock)
    if serving:
        ring.stages[0].__class__ = _ServingStage
    ring.arrive(RID_L)
    ring.run(120)
    return ring


def _admits(ring):
    return [
        (o.told, getattr(o, fb.WIRE_FALLBACK, None), getattr(o, fb.WIRE_FLOOR, None))
        for o in ring.wire_objs()
        if type(o).__name__ == "Weg2StoreAdmit"
    ]


def _uniform(ring, rid):
    plans = ring.plans(rid)
    assert all(len(p) == 1 for p in plans), plans
    assert plans[0] == plans[1] == plans[2], plans  # RANK EQUALITY: one plan, every stage
    return plans[0][0][2]


def test_ring_weg2_4_25_red_without_the_wait(monkeypatch):
    ring = _late_ring(monkeypatch, **{fb.ENV_WAIT_PROGRESS: "0"})
    assert _admits(ring) == [(0, 1, None)]
    assert _uniform(ring, RID_L) == 0
    assert ring.stages[0]._pf_fallback_n == 1


def test_ring_weg2_4_25_green_the_late_ack_admits_told_on_every_rank(monkeypatch):
    ring = _late_ring(monkeypatch)
    assert _admits(ring) == [(TOLD_L, None, None)]
    assert _uniform(ring, RID_L) == TOLD_L
    assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0
    assert ring.stages[0]._pf_waited_n == 1
    assert all(not s.tree_cache.released for s in ring.stages)


def test_ring_idle_box_keeps_the_fallback(monkeypatch):
    """The same lag on an idle box: the Frist bounds it, as before (told=0, uniform)."""
    ring = _late_ring(monkeypatch, serving=False)
    assert _admits(ring) == [(0, 1, None)]
    assert _uniform(ring, RID_L) == 0


def test_ring_a_follower_that_never_acks_ends_in_frist_hard(monkeypatch):
    for k, v in {fb.ENV_FALLBACK: "1", "SGLANG_WEG2_TOLD_ABSOLUTE": "1", fb.ENV_WAIT_HARD_S: "20"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    monkeypatch.setattr(R, "DT", 0.5)
    reads = {0: {RID_L: 1.0}, 1: {RID_L: 1.0}, 2: {RID_L: 10**6}}
    ring = R.Ring(m, monkeypatch, {RID_L: TOLD_L}, reads)
    monkeypatch.setattr(fb, "_perf", ring.clock)
    ring.stages[0].__class__ = _ServingStage
    ring.arrive(RID_L)
    ring.run(120)
    assert _admits(ring) == [(0, 1, None)]
    assert _uniform(ring, RID_L) == 0
    assert ring.clock() >= 20.0


# ---------------------------------------------------------------------------
# (a) the follower: NO-ROOM that lasts as long as a running request
# ---------------------------------------------------------------------------

POOL = 262144


class _Follower:
    """A follower stage at the NO-ROOM ack: ``avail`` free rows, ``evict`` reported-evictable rows; the rest of the
    pool (``max_total_num_tokens - avail - evict``) is held by running requests."""

    def __init__(self, monkeypatch, rid, told, avail, evict, pool=POOL, pp_rank=1):
        self.rid, self.told = rid, told
        self.avail, self.evict = avail, evict
        self.clock = FakeClock()
        tree = types.SimpleNamespace(
            check_prefetch_progress=lambda r: True,
            token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: self.avail),
            evictable_size=lambda: self.evict,
        )
        self.sched = types.SimpleNamespace(
            ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=3), tree_cache=tree, max_total_num_tokens=pool,
        )
        self.sent = []
        self.rows = told
        st = fb._fstate(self.sched, create=True)
        st.expect[rid] = told
        st.registered[rid] = types.SimpleNamespace(rid=rid, _weg2_early_told=None)
        monkeypatch.setattr(fb, "_perf", self.clock)
        monkeypatch.setattr(fb, "_loadback_rows", lambda sched, req, told_: self.rows)
        monkeypatch.setattr(fb, "ack_anchors", lambda *a, **k: [])
        monkeypatch.setattr(fb, "own_prefix",
                            lambda sched, req, r, told_: fb._room_own(sched, req, r, int(told_)))
        monkeypatch.setattr(fb, "_channel", lambda s: types.SimpleNamespace(
            send_nowait=lambda ack: self.sent.append(ack) or True, pump=lambda: None))

    def pump(self, dt=0.0):
        self.clock.t += dt
        fb.follower_pump(self.sched)
        return [e for ack in self.sent for e in ack.reads]


def test_noroom_120446_red_the_ack_says_zero(monkeypatch):
    """weg2-0-3: rows 184768, room 162176 -- the h16p code acks 0 and PP0 answers told=0 for every rank."""
    monkeypatch.setenv(fb.ENV_ROOM_DEFER, "0")
    f1 = _Follower(monkeypatch, "weg2-0-3", 184768, avail=60000, evict=102176)
    assert f1.pump() == [("weg2-0-3", 0, ())]
    f2 = _Follower(monkeypatch, "weg2-0-3", 184768, avail=60000, evict=102176, pp_rank=2)
    assert f2.pump() == [("weg2-0-3", 0, ())]
    s = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=3, pp_rank=0))
    _open(s, "weg2-0-3", 184768, {1: 0, 2: 0})
    assert fb.pp0_decide(s, "weg2-0-3", now=0.0, pass_n=10) == (0, fb.REASON_MISMATCH)


def test_noroom_120446_green_the_ack_waits_for_the_room_and_names_the_full_reach(monkeypatch, caplog):
    f = _Follower(monkeypatch, "weg2-0-3", 184768, avail=60000, evict=102176)
    with caplog.at_level(logging.INFO, logger=fb.logger.name):
        assert f.pump() == []  # nothing sent: the 99968 rows a running prefill holds are not this rank's room yet
        assert f.pump(dt=1.0) == []
        assert "weg2-0-3" in fb._fstate(f.sched).registered  # the read stays registered
        # the running prefill ends: "full token usage" 0.38 -> 0.06
        f.avail, f.evict = POOL - 16000, 10000
        assert f.pump(dt=1.0) == [("weg2-0-3", 184768, ())]
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert text.count("PF TOLD-ACK ROOM-DEFER rank pp=1 rid=weg2-0-3") == 1  # named once, not every pass
    assert "PF TOLD-ACK ROOM-DEFER RELEASED rank pp=1 rid=weg2-0-3" in text
    assert "PF TOLD-ACK NO-ROOM" not in text
    assert "weg2-0-3" not in fb._fstate(f.sched).registered
    # PP0: both followers' acks are told: admitted in full
    s = types.SimpleNamespace(ps=types.SimpleNamespace(pp_size=3, pp_rank=0))
    _open(s, "weg2-0-3", 184768, {1: 184768, 2: 184768})
    assert fb.pp0_decide(s, "weg2-0-3", now=0.0, pass_n=10) == (184768, fb.REASON_ACKS)


def test_noroom_123329_the_second_case_is_transient_too(monkeypatch):
    """weg2-0-10 (boot 122948): rows 188928, room 129472, usage 0.51 -> 132672 rows held by others."""
    f = _Follower(monkeypatch, "weg2-0-10", 188928, avail=30000, evict=99472)
    assert f.pump() == []
    f.avail, f.evict = POOL, 0
    assert f.pump(dt=1.0) == [("weg2-0-10", 188928, ())]


def test_noroom_when_it_cannot_fit_even_unlocked_still_acks_zero(monkeypatch):
    """The told's load-back exceeds the whole pool: no wait helps, the ack says 0 (B5's anchors below still apply)."""
    f = _Follower(monkeypatch, "r", 250000, avail=60000, evict=0, pool=200000)
    assert f.pump() == [("r", 0, ())]


def test_noroom_on_an_idle_pool_acks_zero(monkeypatch):
    """Nothing held by others (locked 0) and still short: permanent, the ack says 0 as before."""
    f = _Follower(monkeypatch, "r", 200000, avail=100000, evict=0, pool=100000)
    assert f.pump() == [("r", 0, ())]


def test_a_deferred_rid_rechecks_at_most_every_50_ms(monkeypatch):
    f = _Follower(monkeypatch, "weg2-0-3", 184768, avail=60000, evict=102176)
    calls = []
    real = fb.own_prefix
    monkeypatch.setattr(fb, "own_prefix", lambda *a: calls.append(1) or real(*a))
    f.pump()
    f.pump(dt=0.01)
    f.pump(dt=0.01)
    assert len(calls) == 1
    f.pump(dt=0.05)
    assert len(calls) == 2


def test_the_fallback_admit_drops_a_deferred_rid(monkeypatch):
    """PP0's Frist (idle box) or hard cap ended in told=0: the follower forgets the deferral with its read."""
    f = _Follower(monkeypatch, "weg2-0-3", 184768, avail=60000, evict=102176)
    f.pump()
    st = fb._fstate(f.sched)
    assert "weg2-0-3" in st.deferred
    fb.follower_forget(f.sched, "weg2-0-3")
    assert "weg2-0-3" not in st.deferred and "weg2-0-3" not in st.registered
    assert fb.room_deferred(f.sched, "weg2-0-3") is None


def test_room_own_marks_only_the_own_depth_not_the_anchors(monkeypatch):
    f = _Follower(monkeypatch, "r", 184768, avail=60000, evict=102176)
    assert fb._room_own(f.sched, object(), "r", 184768, mark=False) == 0
    assert fb.room_deferred(f.sched, "r") is None
    assert fb._room_own(f.sched, object(), "r", 184768) == 0
    assert fb.room_deferred(f.sched, "r") == (184768, 162176, 99968)


def test_room_own_with_enough_room_is_unchanged(monkeypatch):
    f = _Follower(monkeypatch, "r", 100000, avail=60000, evict=102176)
    assert fb._room_own(f.sched, object(), "r", 100000) == 100000
    assert fb.room_deferred(f.sched, "r") is None


# ---------------------------------------------------------------------------
# (a) end to end on the ring: the follower's room is short while PP0 serves
# ---------------------------------------------------------------------------


def test_ring_no_room_then_room_admits_told_on_every_rank(monkeypatch):
    """PP1/PP2 hold the read in full but their room is short for 3 s (a running prefill); PP0 waits, the acks come."""
    for k, v in {fb.ENV_FALLBACK: "1", "SGLANG_WEG2_TOLD_ABSOLUTE": "1"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    monkeypatch.setattr(R, "DT", 0.5)
    reads = {0: {RID_L: 1.0}, 1: {RID_L: 0.5}, 2: {RID_L: 0.5}}
    ring = R.Ring(m, monkeypatch, {RID_L: TOLD_L}, reads)
    monkeypatch.setattr(fb, "_perf", ring.clock)
    ring.stages[0].__class__ = _ServingStage
    pool = {"free": False}

    def fake_room(sched, req, rid, own, mark=True):
        if mark:
            fb._room_defer_map(sched).pop(str(rid), None)
        if pool["free"] or ring.clock() >= 6.0:
            return own
        if mark:
            fb._room_defer_map(sched)[str(rid)] = (int(own), 162176, 99968)
        return 0

    monkeypatch.setattr(fb, "_room_own", fake_room)
    ring.arrive(RID_L)
    ring.run(120)
    assert _admits(ring) == [(TOLD_L, None, None)]
    assert _uniform(ring, RID_L) == TOLD_L
    assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0
    assert ring.stages[0]._pf_waited_n == 1
    assert all(not s.tree_cache.released for s in ring.stages)


def test_ring_no_room_red_without_the_defer(monkeypatch):
    """The h16p code on the same pool: the first ack says 0 -> told=0 for every rank."""
    for k, v in {fb.ENV_FALLBACK: "1", "SGLANG_WEG2_TOLD_ABSOLUTE": "1", fb.ENV_ROOM_DEFER: "0"}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    monkeypatch.setattr(R, "DT", 0.5)
    reads = {0: {RID_L: 1.0}, 1: {RID_L: 0.5}, 2: {RID_L: 0.5}}
    ring = R.Ring(m, monkeypatch, {RID_L: TOLD_L}, reads)
    monkeypatch.setattr(fb, "_perf", ring.clock)
    ring.stages[0].__class__ = _ServingStage

    def room_zero(sched, req, rid, own, mark=True):
        return 0 if ring.clock() < 6.0 else own

    monkeypatch.setattr(fb, "_room_own", room_zero)
    ring.arrive(RID_L)
    ring.run(120)
    assert _admits(ring) == [(0, 1, None)]
    assert _uniform(ring, RID_L) == 0
