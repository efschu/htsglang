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


def _is_beat(entry):
    return len(entry) > 3 and bool(entry[3]) and entry[3][0] == fb.DEFER_MARK


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
        return [e for ack in self.sent for e in ack.reads if not _is_beat(e)]

    def beats(self):
        """The defer entries (DEFER-LIFESIGN) this stage has put on the wire so far."""
        return [e for ack in self.sent for e in ack.reads if _is_beat(e)]


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
    # the release leaves no deferral state behind (a later read of the same rid starts clean)
    assert "weg2-0-3" not in fb._fstate(f.sched).deferred
    assert "weg2-0-3" not in (getattr(f.sched, "_weg2_fb_room_defer_seen", None) or set())
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
    assert fb.room_deferred(f.sched, "r") == (184768, 162176, 99968, 0)


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
            fb._room_defer_map(sched)[str(rid)] = (int(own), 162176, 99968, 0)
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


# ===========================================================================
# review A1 (B1): the hard bound hangs on LIFESIGNS, not on 60 s of wall clock from the read-ahead
# ===========================================================================

TOLD_D = 184768  # weg2-0-3


def _beat_ack(rank, rid, rows=TOLD_D, room=162176, locked=99968, unpayable=0, seq=0):
    return fb.Weg2ToldReadAck(
        rank=rank, seq=seq, reads=[(rid, 0, (), (fb.DEFER_MARK, rows, room, locked, unpayable))]
    )


def _pp0c(clk, **kw):
    """A PP0 double with an ack channel: ``s.box`` is what the followers 'sent', harvested by ``pp0_harvest``."""
    s = _pp0(clk, **kw)
    s.box = []

    def _harvest():
        out, s.box[:] = list(s.box), []
        return out

    s._weg2_fb_channel = types.SimpleNamespace(harvest=_harvest)
    return s


def _beat(s, rank, t, rid="r", **kw):
    s.box.append(_beat_ack(rank, rid, **kw))
    fb.pp0_harvest(s, now=t)


def _ack(s, rank, t, own, rid="r"):
    s.box.append(fb.Weg2ToldReadAck(rank=rank, seq=0, reads=[(rid, own)]))
    fb.pp0_harvest(s, now=t)


def test_a_defer_entry_is_a_lifesign_never_an_ack(monkeypatch, caplog):
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    s = _pp0c(clk)
    o = _open(s, "r", TOLD_D, {})
    with caplog.at_level(logging.INFO, logger=fb.logger.name):
        _beat(s, 1, 3.0)
        _beat(s, 1, 4.0, room=170000)
        # a beat for a rid PP0 decided already (or never opened) is not a late ack
        _beat(s, 1, 5.0, rid="gone")
    assert o.acks == {}
    assert o.beats[1].n == 2 and o.beats[1].at == 4.0 and o.beats[1].progress_at == 4.0  # the room grew
    assert "PF TOLD-ACK LATE" not in "\n".join(r.getMessage() for r in caplog.records)
    # the ack of that rank ends its deferral
    _ack(s, 1, 6.0, TOLD_D)
    assert o.acks == {1: TOLD_D} and 1 not in o.beats
    # a beat of a rank that acked is ignored
    _beat(s, 1, 7.0)
    assert 1 not in o.beats


def test_a_beat_without_progress_keeps_its_progress_clock(monkeypatch):
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    s = _pp0c(clk)
    o = _open(s, "r", TOLD_D, {})
    _beat(s, 1, 1.0)
    _beat(s, 1, 2.0)  # same room, same locked
    _beat(s, 1, 3.0, room=100000)  # a wobble DOWN is no progress
    assert (o.beats[1].at, o.beats[1].progress_at) == (3.0, 1.0)
    _beat(s, 1, 3.5)  # ... and the way back UP to the best seen is none either
    assert (o.beats[1].at, o.beats[1].progress_at) == (3.5, 1.0)
    _beat(s, 1, 4.0, locked=90000)  # the rows others hold shrank
    assert o.beats[1].progress_at == 4.0
    _beat(s, 1, 5.0, locked=95000)  # back up: no progress against the best seen
    assert o.beats[1].progress_at == 4.0
    _beat(s, 1, 6.0, unpayable=0)  # nothing shrank: no progress
    assert o.beats[1].progress_at == 4.0


def test_a_malformed_beat_is_no_lifesign(monkeypatch):
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    s = _pp0c(clk)
    o = _open(s, "r", TOLD_D, {})
    s.box.append(fb.Weg2ToldReadAck(rank=1, seq=0, reads=[("r", 0, (), (fb.DEFER_MARK, "x"))]))
    fb.pp0_harvest(s, now=1.0)
    assert o.beats == {} and o.acks == {}


def _deferring_pp0(monkeypatch, *, until, **kw):
    """PP0 (box serving) of weg2-0-3: PP2 acked told, PP1 defers (room short, a prefill holds the rows) and beats every
    second until ``until``. Returns (scheduler, open entry, the verdicts at each whole second)."""
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    s = _pp0c(clk, **kw)
    o = _open(s, "r", TOLD_D, {2: TOLD_D})
    return s, o


def test_a_deferring_follower_is_waited_for_beyond_the_old_hard_bound_room_comes_at_75s(monkeypatch):
    """B1: the room comes at t=75 s, after the old 60 s wall bound -- the told is read, not recomputed."""
    s, o = _deferring_pp0(monkeypatch, until=75)
    for t in range(1, 75):
        _beat(s, 1, float(t))
        assert fb.pp0_decide(s, "r", now=float(t), pass_n=10 + t) is None, t
    _ack(s, 1, 75.0, TOLD_D)
    assert fb.pp0_decide(s, "r", now=75.0, pass_n=90) == (TOLD_D, fb.REASON_ACKS)


def test_red_without_the_lifesign_the_same_wait_ends_in_frist_hard_at_60s(monkeypatch):
    monkeypatch.setenv(fb.ENV_DEFER_LIFESIGN, "0")
    s, o = _deferring_pp0(monkeypatch, until=75)
    verdicts = {}
    for t in range(1, 75):
        _beat(s, 1, float(t))
        verdicts[t] = fb.pp0_decide(s, "r", now=float(t), pass_n=10 + t)
    assert verdicts[59] is None
    assert verdicts[61] == (0, fb.REASON_FRIST_HARD)


def test_a_dead_follower_still_ends_in_frist_hard_60s_after_its_last_beat(monkeypatch):
    s, o = _deferring_pp0(monkeypatch, until=10)
    for t in range(1, 11):
        _beat(s, 1, float(t))
    assert fb.pp0_decide(s, "r", now=69.0, pass_n=99) is None  # 59 s after the last beat
    assert fb.pp0_decide(s, "r", now=71.0, pass_n=99) == (0, fb.REASON_FRIST_HARD)


def test_a_silent_missing_rank_is_counted_from_the_read_ahead_whatever_the_other_beats(monkeypatch):
    """PP1 beats forever, PP2 neither acked nor beats: PP2 is dead for all PP0 can tell -- 60 s from the read-ahead."""
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    s = _pp0c(clk)
    o = _open(s, "r", TOLD_D, {})
    for t in range(1, 70):
        _beat(s, 1, float(t))
        v = fb.pp0_decide(s, "r", now=float(t), pass_n=10 + t)
        if t < 60:
            assert v is None, t
    assert fb.pp0_decide(s, "r", now=69.0, pass_n=99) == (0, fb.REASON_FRIST_HARD)


def test_a_follower_that_beats_forever_meets_the_absolute_ceiling(monkeypatch):
    monkeypatch.setenv(fb.ENV_WAIT_ABS_S, "100")
    s, o = _deferring_pp0(monkeypatch, until=10_000)
    last = None
    for t in range(1, 102):
        _beat(s, 1, float(t), room=100000 + t)  # the room grows: a perfect lifesign
        last = fb.pp0_decide(s, "r", now=float(t), pass_n=10 + t)
        if t < 100:
            assert last is None, t
    assert last == (0, fb.REASON_FRIST_ABS)


def test_unpayable_evictable_is_bounded_by_progress_even_while_the_box_serves(monkeypatch):
    """room + locked < rows (the full-arena state): only a growing room helps; beats alone do not extend the wait."""
    s, o = _deferring_pp0(monkeypatch, until=120)
    for t in range(1, 62):
        _beat(s, 1, float(t), rows=200000, room=100000, locked=10000, unpayable=150000)
    assert fb.pp0_decide(s, "r", now=61.5, pass_n=99) == (0, fb.REASON_FRIST_HARD)  # 60.5 s since the first beat
    # the same wait with a room that grows at t=50 lives on
    s2, o2 = _deferring_pp0(monkeypatch, until=120)
    for t in range(1, 62):
        _beat(s2, 1, float(t), rows=200000, room=100000 if t < 50 else 150000, locked=10000, unpayable=150000)
    assert o2.beats[1].progress_at == 50.0
    assert fb.pp0_decide(s2, "r", now=61.5, pass_n=99) is None
    assert fb.pp0_decide(s2, "r", now=111.0, pass_n=99) == (0, fb.REASON_FRIST_HARD)


def test_an_idle_box_waits_only_for_a_follower_that_makes_progress(monkeypatch):
    """The box idles (the Frist would fire) but a follower beats WITH progress: wait. Without progress: 60 s."""
    s, o = _deferring_pp0(monkeypatch, until=70, serving=False)
    for t in range(1, 31):
        _beat(s, 1, float(t), room=100000 + 1000 * t)
        assert fb.pp0_decide(s, "r", now=float(t), pass_n=10 + t) is None, t
    s2, o2 = _deferring_pp0(monkeypatch, until=70, serving=False)
    for t in range(1, 62):
        _beat(s2, 1, float(t))  # a live follower, nothing changes
    assert fb.pp0_decide(s2, "r", now=30.0, pass_n=99) is None
    assert fb.pp0_decide(s2, "r", now=61.5, pass_n=99) == (0, fb.REASON_FRIST_HARD)


def test_an_idle_box_with_a_stale_beat_keeps_the_frist(monkeypatch):
    """A beat older than four beat periods is no 'defers now': an idle box falls back to the Frist."""
    s, o = _deferring_pp0(monkeypatch, until=1, serving=False)
    _beat(s, 1, 3.5)  # the Frist (3 s) is up; a beat from 0.5 s ago
    assert fb.pp0_decide(s, "r", now=4.0, pass_n=99) is None
    assert fb.pp0_decide(s, "r", now=10.0, pass_n=99) == (0, fb.REASON_FRIST)  # 6.5 s old: stale


def test_a_sleep_wake_during_the_wait_does_not_age_it(monkeypatch):
    """Review point 4: published_at runs on through a sleep; the wake is a floor for the hard bound."""
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)

    def at(t, wake_at=None):
        # PP0's clock for ``now`` and the perf clock run together; the wake happened at 'now' = wake_at
        clk.t = 1000.0 + t
        s = _pp0(clk)
        if wake_at is not None:
            s._weg2_last_wake_t = 1000.0 + wake_at
        _open(s, "r", TOLD_D, {2: TOLD_D}, published_at=0.0)
        return s

    assert fb.pp0_decide(at(100.0), "r", now=100.0, pass_n=10) == (0, fb.REASON_FRIST_HARD)  # no wake: 100 s old
    assert fb.pp0_decide(at(100.0, wake_at=95.0), "r", now=100.0, pass_n=10) is None
    assert fb.pp0_decide(at(154.0, wake_at=95.0), "r", now=154.0, pass_n=10) is None  # 59 s after the wake
    assert fb.pp0_decide(at(156.0, wake_at=95.0), "r", now=156.0, pass_n=10) == (0, fb.REASON_FRIST_HARD)


# ---------------------------------------------------------------------------
# the follower side of the beat
# ---------------------------------------------------------------------------


def test_the_deferring_follower_beats_at_most_every_second_and_the_real_ack_replaces_the_beat(monkeypatch):
    f = _Follower(monkeypatch, "weg2-0-3", TOLD_D, avail=60000, evict=102176)
    assert f.pump() == []
    assert len(f.beats()) == 1  # at once: PP0 learns the follower lives
    f.pump(dt=0.5)
    assert len(f.beats()) == 1
    f.pump(dt=0.6)
    assert len(f.beats()) == 2
    assert f.beats()[-1][0] == "weg2-0-3" and f.beats()[-1][1] == 0  # an own of 0 -- but a beat is no ack
    assert f.beats()[-1][3] == (fb.DEFER_MARK, TOLD_D, 162176, 99968, 0)
    f.avail, f.evict = POOL - 16000, 10000
    assert f.pump(dt=0.5) == [("weg2-0-3", TOLD_D, ())]
    n = len(f.beats())
    f.pump(dt=5.0)
    f.pump(dt=5.0)
    assert len(f.beats()) == n  # nothing after the ack


def test_the_real_ack_replaces_a_beat_still_waiting_in_the_outbox(monkeypatch):
    f = _Follower(monkeypatch, "weg2-0-3", TOLD_D, avail=60000, evict=102176)
    st = fb._fstate(f.sched)
    # the channel is busy: nothing goes out, the outbox keeps the beat
    monkeypatch.setattr(fb, "_channel", lambda s: types.SimpleNamespace(send_nowait=lambda ack: False, pump=lambda: None))
    f.pump()
    assert [e[0] for e in st.outbox] == ["weg2-0-3"] and _is_beat(st.outbox[0])
    f.pump(dt=1.5)
    assert len(st.outbox) == 1  # one entry per rid, the newest beat
    f.avail, f.evict = POOL - 16000, 10000
    f.pump(dt=0.5)
    assert st.outbox == [("weg2-0-3", TOLD_D, ())]


def test_no_beats_with_the_lifesign_switch_off(monkeypatch):
    monkeypatch.setenv(fb.ENV_DEFER_LIFESIGN, "0")
    f = _Follower(monkeypatch, "weg2-0-3", TOLD_D, avail=60000, evict=102176)
    f.pump()
    f.pump(dt=2.0)
    assert f.beats() == [] and f.pump() == []


def test_a_beat_goes_through_the_pickled_ack_message_unchanged():
    import pickle

    ack = _beat_ack(1, "weg2-0-3")
    back = pickle.loads(pickle.dumps(ack))
    assert back.reads == ack.reads and _is_beat(back.reads[0])


# ===========================================================================
# review A2 (B2): the full-arena NO-ROOM is transient too
# ===========================================================================


def test_noroom_with_unpayable_evictable_the_full_arena_state_red_acks_zero(monkeypatch):
    """h16n numbers (ACK-ROOM-PAYABLE): reported 255040, payable 48192 -- the room is short because the peel has no
    arena slot for the backups. The h16p code (and the first T2 stand) acked 0."""
    monkeypatch.setenv(fb.ENV_ROOM_DEFER_UNPAYABLE, "0")
    f = _Follower(monkeypatch, "weg2-16-53", 150000, avail=3000, evict=255040)
    monkeypatch.setattr(fb, "_ack_evictable", lambda tree: 48192)
    assert f.pump() == [("weg2-16-53", 0, ())]


def test_noroom_with_unpayable_evictable_green_defers_and_reads_when_the_arena_clears(monkeypatch, caplog):
    f = _Follower(monkeypatch, "weg2-16-53", 150000, avail=3000, evict=255040)
    payable = {"v": 48192}
    monkeypatch.setattr(fb, "_ack_evictable", lambda tree: payable["v"])
    with caplog.at_level(logging.INFO, logger=fb.logger.name):
        assert f.pump() == []
        # unpayable = reported 255040 - payable 48192; locked = 262144 - 3000 - 255040
        assert f.beats()[-1][3] == (fb.DEFER_MARK, 150000, 51192, 4104, 206848)
        assert f.pump(dt=1.0) == []
        payable["v"] = 255040  # the backups found their arena slots: the peel pays everything
        assert f.pump(dt=1.0) == [("weg2-16-53", 150000, ())]
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert "class=unpayable" in text and "full-arena state" in text
    assert "PF TOLD-ACK NO-ROOM" not in text


def test_noroom_a_told_beyond_the_whole_pool_is_permanent_even_with_unpayable_evictable(monkeypatch):
    f = _Follower(monkeypatch, "r", 270000, avail=3000, evict=255040)
    monkeypatch.setattr(fb, "_ack_evictable", lambda tree: 48192)
    assert f.pump() == [("r", 0, ())]
    assert f.beats() == []


def test_noroom_with_an_unreadable_pool_is_permanent(monkeypatch):
    f = _Follower(monkeypatch, "r", 150000, avail=3000, evict=255040, pool=0)
    monkeypatch.setattr(fb, "_ack_evictable", lambda tree: 48192)
    assert f.pump() == [("r", 0, ())]


def test_the_ring_full_arena_noroom_admits_told_on_every_rank_after_the_beats_outlive_the_old_bound(monkeypatch):
    """End to end: both followers defer for 45 s (the hard bound is 20 s): their beats keep PP0 waiting, the acks
    come, told is admitted on every rank."""
    ring = _deferring_ring(monkeypatch, room_at=45.0, **{fb.ENV_WAIT_HARD_S: "20"})
    assert _admits(ring) == [(TOLD_L, None, None)]
    assert _uniform(ring, RID_L) == TOLD_L
    assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0
    assert ring.clock() >= 45.0
    assert all(not s.tree_cache.released for s in ring.stages)


def test_the_ring_red_without_the_lifesign_the_same_wait_ends_in_frist_hard(monkeypatch):
    ring = _deferring_ring(monkeypatch, room_at=45.0, **{fb.ENV_WAIT_HARD_S: "20", fb.ENV_DEFER_LIFESIGN: "0"})
    assert _admits(ring) == [(0, 1, None)]
    assert _uniform(ring, RID_L) == 0


def test_the_ring_a_follower_that_stops_beating_is_cut_off_20s_after_its_last_beat(monkeypatch):
    """The stamps of a beat are on PP0's pass clock (pp0_publish hands ``now`` to pp0_harvest): PP1 beats until
    t=10 and dies, PP2 beats on -- hard bound 20 s: told=0 for every rank at ~30 s, not at the absolute ceiling."""
    real_pump = fb.follower_pump

    def pump(sched):
        if sched.ps.pp_rank == 1 and ring_box["ring"].clock() >= 10.0:
            return  # PP1 is dead
        return real_pump(sched)

    ring_box = {}
    monkeypatch.setattr(fb, "follower_pump", pump)
    ring_box["ring"] = ring = _deferring_ring(
        monkeypatch, room_at=1e9, run=300, box=ring_box, **{fb.ENV_WAIT_HARD_S: "20"}
    )
    assert _admits(ring) == [(0, 1, None)]
    assert _uniform(ring, RID_L) == 0
    verdict_at = ring.stages[0].admitted[0][0] * 0.5  # PP0's pass index x DT
    assert 28.0 <= verdict_at <= 36.0, verdict_at


def _deferring_ring(monkeypatch, room_at, run=400, box=None, **env):
    for k, v in {fb.ENV_FALLBACK: "1", "SGLANG_WEG2_TOLD_ABSOLUTE": "1", **env}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    monkeypatch.setattr(R, "DT", 0.5)
    reads = {0: {RID_L: 1.0}, 1: {RID_L: 0.5}, 2: {RID_L: 0.5}}
    ring = R.Ring(m, monkeypatch, {RID_L: TOLD_L}, reads)
    monkeypatch.setattr(fb, "_perf", ring.clock)
    ring.stages[0].__class__ = _ServingStage

    def fake_room(sched, req, rid, own, mark=True):
        if mark:
            fb._room_defer_map(sched).pop(str(rid), None)
        if ring.clock() >= room_at:
            return own
        if mark:
            fb._room_defer_map(sched)[str(rid)] = (int(own), 162176, 99968, 0)
        return 0

    monkeypatch.setattr(fb, "_room_own", fake_room)
    if box is not None:
        box["ring"] = ring
    ring.arrive(RID_L)
    ring.run(run)
    return ring


# ===========================================================================
# review A6 (B7, N5, N7): the B5 floor branch waits too; own_prefix has one caller
# ===========================================================================


def test_floor_branch_a_missing_ack_waits_while_serving_and_falls_to_the_frist_when_idle(monkeypatch):
    """acks={1: 87296} differs from told 91712, a common anchor exists, PP2 has not acked: the Floor branch of
    pp0_decide. Serving -> keep waiting; idle -> the Frist. Kills 'the first _frist_verdict call = the old Frist'."""
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    s = _pp0(clk)
    o = _open(s, "weg2-3-27", 91712, {1: 87296}, anchors={1: (87296,)})
    assert fb._floor_candidate(o) == 87296 and not all(r in o.acks for r in (1, 2))
    assert fb.pp0_decide(s, "weg2-3-27", now=50.0, pass_n=10) is None
    assert o.waited_since is not None
    s_idle = _pp0(clk, serving=False)
    _open(s_idle, "weg2-3-27", 91712, {1: 87296}, anchors={1: (87296,)})
    assert fb.pp0_decide(s_idle, "weg2-3-27", now=50.0, pass_n=10) == (0, fb.REASON_FRIST)
    # and the ack that completes the floor ends the wait with the floor
    o.acks[2] = 87296
    assert fb.pp0_decide(s, "weg2-3-27", now=51.0, pass_n=11) == (87296, fb.REASON_FLOOR)


def test_floor_branch_a_deferring_follower_keeps_the_floor_wait_alive(monkeypatch):
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    s = _pp0c(clk)
    o = _open(s, "weg2-3-27", 91712, {1: 87296}, anchors={1: (87296,)})
    for t in range(1, 80):
        _beat(s, 2, float(t), rid="weg2-3-27", rows=91712)
    assert fb.pp0_decide(s, "weg2-3-27", now=79.0, pass_n=10) is None


def test_room_own_returns_zero_on_a_deferred_probe_every_time(monkeypatch):
    """N5: the defer is carried by the side table AND by the return value 0 -- a repeated probe of a named defer
    must not hand back ``own`` (a caller outside follower_pump would ack a reach that does not fit)."""
    f = _Follower(monkeypatch, "r", TOLD_D, avail=60000, evict=102176)
    assert fb._room_own(f.sched, object(), "r", TOLD_D) == 0
    assert fb.room_deferred(f.sched, "r") is not None
    assert fb._room_own(f.sched, object(), "r", TOLD_D) == 0  # the named (seen) path
    assert fb.room_deferred(f.sched, "r") is not None


def test_only_follower_pump_calls_own_prefix():
    """N5 (the fragile coupling): a deferred ack is held back by ``follower_pump`` reading the side table. Any other
    caller of ``own_prefix`` would get 0 for a transient shortage and ack it. Pin the callers."""
    import inspect

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python", "sglang"))
    pump_src = inspect.getsource(fb.follower_pump)
    hits = []
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, encoding="utf-8", errors="replace") as fh:
                for i, line in enumerate(fh, 1):
                    if "own_prefix(" in line and "def own_prefix" not in line:
                        hits.append((os.path.relpath(path, root), line.strip()))
    assert len(hits) == 1, hits
    assert hits[0][0].endswith("weg2_told_fallback.py") and hits[0][1] in pump_src, hits


# ===========================================================================
# review A3: the metal instruments
# ===========================================================================


def test_the_wait_and_the_verdict_name_the_pinned_l2_slots_and_the_idle_blockers(monkeypatch, caplog):
    clk = FakeClock()
    monkeypatch.setattr(fb, "_perf", clk)
    monkeypatch.setattr(fb, "_pinned_slots", lambda sched, rid: 2887)
    s = _pp0c(clk)
    s.idle_blockers = lambda: ["waiting_queue", "hicache_prefetch(1: weg2-0-3)"]
    o = _open(s, "weg2-0-3", TOLD_D, {2: TOLD_D})
    with caplog.at_level(logging.INFO, logger=fb.logger.name):
        _beat(s, 1, 2.0, rid="weg2-0-3")
        assert fb.pp0_decide(s, "weg2-0-3", now=3.0, pass_n=10) is None
        _ack(s, 1, 9.0, TOLD_D, rid="weg2-0-3")
        verdict = fb.pp0_decide(s, "weg2-0-3", now=10.0, pass_n=11)
        fb.pp0_note_verdict(s, "weg2-0-3", TOLD_D, verdict[0], verdict[1], 10.0, 0.0)
    text = "\n".join(r.getMessage() for r in caplog.records)
    wait = [ln for ln in text.splitlines() if "PF TOLD-WAIT rid=" in ln][0]
    waited = [ln for ln in text.splitlines() if "PF TOLD-WAITED" in ln][0]
    for ln in (wait, waited):
        assert "pinned_l2_slots=2887" in ln and "hicache_prefetch(1: weg2-0-3)" in ln
    assert "defer[pp1:beats=1 rows=184768 room=162176 locked=99968" in wait
    assert "reason=acks" in waited and "waited=" in waited


def test_the_released_defer_names_duration_beats_and_pinned_slots(monkeypatch, caplog):
    monkeypatch.setattr(fb, "_pinned_slots", lambda sched, rid: 2887)
    f = _Follower(monkeypatch, "weg2-0-3", TOLD_D, avail=60000, evict=102176)
    with caplog.at_level(logging.INFO, logger=fb.logger.name):
        f.pump()
        f.pump(dt=1.5)
        f.avail, f.evict = POOL - 16000, 10000
        f.pump(dt=1.0)
    line = [r.getMessage() for r in caplog.records if "ROOM-DEFER RELEASED" in r.getMessage()][0]
    assert "after=2.50s" in line and "beats=2" in line and "pinned_l2_slots=2887" in line


def test_pinned_slots_sums_the_slots_of_the_rids_pinned_span(monkeypatch):
    from sglang.srt.weg2 import park_hold_yield as phy

    monkeypatch.setattr(phy, "_kv_pool", lambda tree: object())
    monkeypatch.setattr(phy, "_span_nodes", lambda tree, rid: ["a", "b"] if rid == "r" else [])
    monkeypatch.setattr(phy, "_node_slots", lambda pool, node: list(range(2000 if node == "a" else 887)))
    sched = types.SimpleNamespace(tree_cache=object())
    assert fb._pinned_slots(sched, "r") == 2887
    assert fb._pinned_slots(sched, "other") == 0
    monkeypatch.setattr(phy, "_kv_pool", lambda tree: None)
    assert fb._pinned_slots(sched, "r") == -1


def test_the_hourly_census_line_gives_the_delta_of_arena_full_and_claim_refused(monkeypatch, caplog):
    clk = FakeClock(0.0)
    monkeypatch.setattr(fb, "_mono", clk)
    cen = {"arena_full": 5}
    monkeypatch.setitem(sys.modules, "sglang.srt.weg2.read_depth", types.SimpleNamespace(census=lambda: dict(cen)))
    pool_cls = types.SimpleNamespace(**{"_1427_full_n": 3})
    monkeypatch.setitem(
        sys.modules, "sglang.srt.mem_cache.pool_host.arena_pool", types.SimpleNamespace(ArenaMHAHostPool=pool_cls)
    )
    s = types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=1, pp_size=3))
    with caplog.at_level(logging.INFO, logger=fb.logger.name):
        fb.census_tick(s)  # baseline
        clk.t = 1800.0
        fb.census_tick(s)  # not yet an hour
        cen["arena_full"] = 12
        setattr(pool_cls, "_1427_full_n", 10)
        s._pf_waited_n = 4
        clk.t = 3601.0
        fb.census_tick(s)
        fb.census_tick(s)  # the window restarted: no second line
    lines = [r.getMessage() for r in caplog.records if "PF TOLD-CENSUS" in r.getMessage()]
    assert len(lines) == 1
    assert "'arena_full_reads': 7" in lines[0] and "'arena_claim_refused': 7" in lines[0] and "'waited': 4" in lines[0]


def test_the_census_never_raises_without_the_modules(monkeypatch):
    clk = FakeClock(0.0)
    monkeypatch.setattr(fb, "_mono", clk)
    monkeypatch.delitem(sys.modules, "sglang.srt.weg2.read_depth", raising=False)
    monkeypatch.delitem(sys.modules, "sglang.srt.mem_cache.pool_host.arena_pool", raising=False)
    s = types.SimpleNamespace()
    fb.census_tick(s)
    clk.t = 4000.0
    fb.census_tick(s)
