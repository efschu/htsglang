"""H64 (NF h16n 17c2563647, boot ..._1015_042125, P.log): the PF group told=0
fallback fired on reads the store HELD -- 130k tokens recomputed.

Case A, weg2-6-44 (04:30:43-45): absolute twin told 96640 (head == told,
window 0, Frist = 2 s grace). PP1/PP2 absorbed the read-ahead in the same
second: the early read's record said 93440 (read at intake, before the
sibling published), the settle said "short" and registered the told-limited
read -- which found the span LOCALLY ("FOLLOWER SATISFIED LOCALLY
local_prefix=96640") and issued nothing. The ack was promised "when the read
ends"; the next follower pass came only with PP0's next chain message
(CHAIN-RECV blocked 2131 ms), and PP0's next pass was past the Frist:
``PF TOLD-FALLBACK told=96640 -> 0 reason=frist after=2.16s acks={}``.

Case B, weg2-8-47 (04:31:04-14): every stage read 33792 alike. Passes were
3-7 s long (the case-A recompute), PP2 is two chain hops down and its ack
needs two PP0 passes (size frame, then the payload irecv PP0 posts only on
seeing the size). The Frist (9.88 s) expired after three PP0 passes with
``acks={1: 33792}``; PP0 then had no paced rid, stopped harvesting, and PP2's
ack surfaced 3.5 min later as ``PF TOLD-ACK LATE``.

Fixed: SETTLE-ACK-NOW (A), FRIST-PASSES (A, B), ACK-DRAIN (B tail), and
GROUP-FLOOR -- equal follower acks below told admit that depth for every rank
instead of 0. Rank equality is asserted on every scenario (one plan, all
stages, same cap): PP0 decides, the followers follow the wire.

The gloo ack stream is modelled one step per PP0 harvest per follower: an ack
is posted (size on the wire), its size is taken on PP0's next harvest (the
payload receive is posted then), its payload on the harvest after; the
follower's send stays in flight until PP0 posted that payload receive.
"""

from __future__ import annotations

import os
import sys

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _told_ring_pf as R  # noqa: E402

from sglang.srt.managers import weg2_store_told as m  # noqa: E402
from sglang.srt.managers import weg2_told_fallback as fb  # noqa: E402

RID_A = "weg2-6-44"
TOLD_A = 96640
STALE_A = 93440
RID_B = "weg2-8-47"
TOLD_B = 33792
RID_NEXT = "weg2-9-50"

FIX_SWITCHES = (
    fb.ENV_MIN_PASSES,
    fb.ENV_ACK_DRAIN,
    fb.ENV_FLOOR,
    m.ENV_SETTLE_ACK_NOW,
)


# ---------------------------------------------------------------------------
# the two-step gloo stream
# ---------------------------------------------------------------------------


class GlooLike:
    def __init__(self):
        self.q = {1: [], 2: []}  # src -> [[ack, step]]; step 0 = size on the wire, 1 = payload posted
        self.harvests = 0


class FollowerEnd:
    def __init__(self, g, rank):
        self.g, self.rank = g, rank

    def _inflight(self):
        q = self.g.q[self.rank]
        return bool(q) and q[-1][1] == 0

    def send_nowait(self, ack):
        if self._inflight():
            return False  # the previous payload isend waits for PP0's receive
        self.g.q[self.rank].append([ack, 0])
        return True

    def pump(self):
        return not self._inflight()


class Pp0End:
    def __init__(self, g):
        self.g = g

    def harvest(self):
        self.g.harvests += 1
        out = []
        for _r, q in self.g.q.items():
            if not q:
                continue
            if q[0][1] == 0:
                q[0][1] = 1  # size taken, payload receive posted -- not complete yet
            else:
                out.append(q.pop(0)[0])  # payload taken; the re-armed size waits a pass
        return out


def _clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith("SGLANG_WEG2_TOLD") or k in (
            "SGLANG_WEG2_P_TWIN_DEFER",
            m.ENV_FOLLOWER_EARLY_READ,
            m.ENV_SETTLE_ACK_NOW,
            "SGLANG_WEG2_DUAL_SHARE",
        ):
            monkeypatch.delenv(k, raising=False)


def _ring(monkeypatch, prompts, reads, dt, fixed=True, **env):
    _clean(monkeypatch)
    monkeypatch.setenv(fb.ENV_FALLBACK, "1")
    monkeypatch.setenv("SGLANG_WEG2_TOLD_ABSOLUTE", "1")
    monkeypatch.setenv(m.ENV_FOLLOWER_EARLY_READ, "1")
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    if not fixed:
        monkeypatch.setenv(fb.ENV_MIN_PASSES, "0")
        monkeypatch.setenv(fb.ENV_ACK_DRAIN, "0")
        monkeypatch.setenv(fb.ENV_FLOOR, "0")
        monkeypatch.setenv(m.ENV_SETTLE_ACK_NOW, "0")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(R, "DT", dt)
    ring = R.Ring(m, monkeypatch, prompts, reads)
    g = GlooLike()
    ring.stages[0]._weg2_fb_channel = Pp0End(g)
    for s in ring.stages[1:]:
        s._weg2_fb_channel = FollowerEnd(g, s.ps.pp_rank)
    ring.gloo = g
    return ring


def _uniform(ring, rid):
    plans = ring.plans(rid)
    assert all(len(p) == 1 for p in plans), plans
    assert plans[0] == plans[1] == plans[2], plans  # RANK EQUALITY: one plan, every stage
    return plans[0][0][2]


def _admits(ring):
    return [
        (o.told, getattr(o, fb.WIRE_FALLBACK, None), getattr(o, fb.WIRE_FLOOR, None))
        for o in ring.wire_objs()
        if type(o).__name__ == "Weg2StoreAdmit"
    ]


# ---------------------------------------------------------------------------
# case A: the stale early record, satisfied locally, long passes
# ---------------------------------------------------------------------------


def _arm_case_a(ring):
    pp0 = ring.stages[0]

    def pp0_read(req, rematch=True, limit_tokens=None):
        # TWIN-REGISTER head=96640 verdict=declined:too_short: the span is local
        req._prefetch_registered_prefix_len = TOLD_A
        return "declined:too_short"

    pp0._prefetch_kvcache = pp0_read
    for s in ring.stages[1:]:
        tree = s.tree_cache

        def fol_read(req, rematch=True, limit_tokens=None, tree=tree):
            if limit_tokens is None:  # the early read at intake: stale 93440
                tree.start_read(req.rid, STALE_A, 0.0)  # done long before the told
                return "issued"
            return "declined:too_short"  # #915 PREFETCH REFUSED need=0

        s._prefetch_kvcache = fol_read
    ring.arrive(RID_A)
    for s in ring.stages[1:]:
        for r in s.waiting_queue:
            r.host_hit_length = TOLD_A  # MATCH-CENSUS-DEEP reached=96640 on PP1/PP2


def _case_a(monkeypatch, fixed=True, dt=2.2, **env):
    reads = {r: {RID_A: 0.1} for r in range(3)}
    ring = _ring(monkeypatch, {RID_A: 0}, reads, dt, fixed=fixed, **env)
    _arm_case_a(ring)
    ring.run(12)
    return ring


def test_case_a_reproduced_without_the_fix(monkeypatch):
    """RED side (all four H64 switches 0 = the h16n code): acks={} at the
    first PP0 pass past the 2 s Frist, told=0 for every rank."""
    ring = _case_a(monkeypatch, fixed=False)
    assert _uniform(ring, RID_A) == 0
    assert _admits(ring) == [(0, 1, None)]
    assert ring.stages[0]._pf_fallback_n == 1
    assert all(RID_A in s.tree_cache.released for s in ring.stages[1:])


def test_case_a_admits_told_on_every_rank(monkeypatch):
    ring = _case_a(monkeypatch)
    assert _uniform(ring, RID_A) == TOLD_A
    assert _admits(ring) == [(TOLD_A, None, None)]
    assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0
    assert ring.stages[0]._pf_admit_acks_n == 1
    assert ring.sleeps == {0: 0.0, 1: 0.0, 2: 0.0}
    assert all(not s.tree_cache.released for s in ring.stages)


def test_case_a_settle_acks_in_the_absorbing_pass(monkeypatch):
    """SETTLE-ACK-NOW alone: the satisfied follower's ack leaves in the pass
    that absorbed the read-ahead (it used to wait for a read never issued,
    i.e. for the next follower pass -- on metal the one carrying PP0's
    fallback)."""
    for on, lag in ((True, 1), (False, 2)):
        sent = {}
        orig = FollowerEnd.send_nowait

        def spy(self, ack, orig=orig, sent=sent):
            ok = orig(self, ack)
            if ok:
                for rid, _own in ack.reads:
                    sent.setdefault((self.rank, rid), ring_box[0].pass_n)
            return ok

        monkeypatch.setattr(FollowerEnd, "send_nowait", spy)
        ring_box = [None]
        reads = {r: {RID_A: 0.1} for r in range(3)}
        env = {} if on else {m.ENV_SETTLE_ACK_NOW: "0"}
        ring = _ring(monkeypatch, {RID_A: 0}, reads, 0.05, **env)
        ring_box[0] = ring
        _arm_case_a(ring)
        ring.run(12)
        told_k = next(k for k in sorted(ring.wire) if ring.wire[k])
        assert sent[(1, RID_A)] - told_k == lag, (on, sent, told_k)
        assert sent[(2, RID_A)] - told_k == lag + 1, (on, sent, told_k)
        assert _uniform(ring, RID_A) == TOLD_A  # short passes: told either way


def test_settle_unit_short_satisfied_returns_ack_now(monkeypatch):
    """follower_early_settle: a registration that issued nothing is not
    "short"; a registration that issued a read still is."""
    from types import SimpleNamespace

    _clean(monkeypatch)

    class T:
        is_eagle = False

        def pop_prefetch_loaded_tokens(self, rid):
            return 0

    def stage(verdict):
        s = SimpleNamespace(ps=SimpleNamespace(pp_rank=1, pp_size=3), tree_cache=T())
        s._prefetch_kvcache = lambda req, rematch=True, limit_tokens=None: verdict
        return s

    req = SimpleNamespace(rid=RID_A, host_hit_length=TOLD_A, prefix_indices=None)
    assert m.follower_early_settle(stage("declined:too_short"), req, TOLD_A, STALE_A, absolute=True) == m.SHORT_SATISFIED
    req2 = SimpleNamespace(rid=RID_A, host_hit_length=0, prefix_indices=None)
    assert m.follower_early_settle(stage("declined:store_absent"), req2, TOLD_A, STALE_A, absolute=True) == m.SHORT_DECLINED
    assert m.follower_early_settle(stage("issued"), req2, TOLD_A, STALE_A, absolute=True) == "short"
    monkeypatch.setenv(m.ENV_SETTLE_ACK_NOW, "0")
    assert m.follower_early_settle(stage("declined:too_short"), req, TOLD_A, STALE_A, absolute=True) == "short"


# ---------------------------------------------------------------------------
# case B: equal reads, long passes, PP2 two hops down
# ---------------------------------------------------------------------------


def _case_b(monkeypatch, fixed=True, dt=3.3, then_next=False, **env):
    reads = {r: {RID_B: 0.1, RID_NEXT: 0.1} for r in range(3)}
    prompts = {RID_B: TOLD_B, RID_NEXT: 4096}
    # metal: Frist 9.88 s = three PP0 passes of 3-7 s. Here: told at pass 1
    # (own read 3.3 s -> window 4.1 s), grace 5.5 -> Frist 9.6 s -> PP0 decides
    # at pass 4 = read-ahead + 3, when PP2's ack has only its size taken.
    env.setdefault(fb.ENV_GRACE_S, "5.5")
    ring = _ring(monkeypatch, prompts, reads, dt, fixed=fixed, **env)
    ring.arrive(RID_B)
    ring.run(10)
    if then_next:
        ring.run(20)  # quiet: no paced rid open
        ring.arrive(RID_NEXT)
        ring.run(10)
    return ring


def test_case_b_reproduced_without_the_fix(monkeypatch):
    ring = _case_b(monkeypatch, fixed=False, then_next=True)
    assert _uniform(ring, RID_B) == 0
    assert _admits(ring)[0] == (0, 1, None)
    assert ring.stages[0]._pf_fallback_n >= 1
    # PP2's ack surfaced only with the next paced rid (PF TOLD-ACK LATE)
    assert getattr(ring.stages[0], "_pf_ack_late_n", 0) >= 1


def test_case_b_admits_told_on_every_rank(monkeypatch):
    ring = _case_b(monkeypatch)
    assert _uniform(ring, RID_B) == TOLD_B
    assert _admits(ring) == [(TOLD_B, None, None)]
    assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0
    assert getattr(ring.stages[0], "_pf_frist_held_n", 0) == 1  # the wall Frist was up, the passes not


def test_frist_passes_only_with_the_floor_of_passes(monkeypatch):
    """The pass floor alone (no other H64 switch) carries case B."""
    ring = _case_b(
        monkeypatch, fixed=False,
        **{fb.ENV_MIN_PASSES: "4"},
    )
    assert _uniform(ring, RID_B) == TOLD_B


def test_frist_still_fires_for_a_follower_that_never_acks(monkeypatch):
    """The pass floor delays the Frist, never removes it: a read that never
    ends still gets told=0 for every rank."""
    reads = {0: {RID_B: 0.1}, 1: {RID_B: 0.1}, 2: {RID_B: 10**6}}
    ring = _ring(monkeypatch, {RID_B: TOLD_B}, reads, 3.3)
    ring.arrive(RID_B)
    ring.run(14)
    assert _uniform(ring, RID_B) == 0
    assert _admits(ring) == [(0, 1, None)]
    # and not before min_passes PP0 passes after the read-ahead
    told_k = next(k for k in sorted(ring.wire) if ring.wire[k])
    admit_k = max(k for k in ring.wire if ring.wire[k])
    assert admit_k - told_k >= fb.min_passes(3)


def test_ack_drain_takes_a_stranded_ack_without_a_paced_rid(monkeypatch):
    """After a Frist with PP2's ack half taken, PP0 keeps polling: the ack is
    taken (LATE, named) within passes, not with the next paced rid."""
    for drain, expect in (("1", True), ("0", False)):
        ring = _case_b(
            monkeypatch, fixed=False,
            **{fb.ENV_ACK_DRAIN: drain},
        )
        ring.run(6)  # quiet passes, no paced rid
        late = getattr(ring.stages[0], "_pf_ack_late_n", 0) >= 1
        assert late is expect, (drain, ring.gloo.q)
        assert (not ring.gloo.q[2]) is expect


def test_min_passes_default_and_override(monkeypatch):
    monkeypatch.delenv(fb.ENV_MIN_PASSES, raising=False)
    assert fb.min_passes(3) == 4
    monkeypatch.setenv(fb.ENV_MIN_PASSES, "0")
    assert fb.min_passes(3) == 0
    monkeypatch.setenv(fb.ENV_MIN_PASSES, "junk")
    assert fb.min_passes(3) == 4


# ---------------------------------------------------------------------------
# GROUP-FLOOR (target (a)): the group-agreed minimum every stage proves
# ---------------------------------------------------------------------------

RID_F = "weg2-0-33"
TOLD_F = 40960
FLOOR_F = 40832


def _floor_ring(monkeypatch, acks=(FLOOR_F, FLOOR_F), dt=0.05, **env):
    reads = {r: {RID_F: 0.1} for r in range(3)}
    ring = _ring(monkeypatch, {RID_F: TOLD_F}, reads, dt, **env)
    for s, own in zip(ring.stages[1:], acks):
        tree = s.tree_cache

        def short_read(req, rematch=True, limit_tokens=None, tree=tree, own=own):
            tree.start_read(req.rid, own, 0.1)  # this stage's store/aux reach
            return "issued"

        s._prefetch_kvcache = short_read
    ring.arrive(RID_F)
    ring.run(40)
    return ring


def test_floor_equal_acks_below_told_admit_that_depth_everywhere(monkeypatch):
    ring = _floor_ring(monkeypatch)
    assert _uniform(ring, RID_F) == FLOOR_F
    assert _admits(ring) == [(FLOOR_F, None, 1)]
    assert ring.stages[0]._pf_floor_n == 1
    assert all(getattr(s, "_pf_floor_applied_n", 0) == 1 for s in ring.stages)
    assert getattr(ring.stages[0], "_pf_fallback_n", 0) == 0
    assert all(not s.tree_cache.released for s in ring.stages)  # nothing released
    assert ring.sleeps == {0: 0.0, 1: 0.0, 2: 0.0}


def test_floor_with_kv_past_the_followers_anchor(monkeypatch):
    """weg2-0-33 shape on the follower side: each follower's read completed
    KV past its recurrent anchor (record FLOOR+64), its RESUMABLE reach is the
    anchor FLOOR -- the ack says FLOOR (ACK-RESUMABLE). Every rank admits at
    FLOOR; a follower comparing its record with the floor would stop the group
    in STORE-TOLD MISMATCH, so the floor marker satisfies it at FLOOR."""
    from sglang.srt.managers import weg2_told_fidelity as tf

    def resumable(scheduler, req, told):
        if int(scheduler.ps.pp_rank) == 0:
            return None
        return min(int(told), FLOOR_F)

    monkeypatch.setattr(tf, "rank_resumable", resumable)
    ring = _floor_ring(monkeypatch, acks=(FLOOR_F + 64, FLOOR_F + 64))
    assert _uniform(ring, RID_F) == FLOOR_F
    assert _admits(ring) == [(FLOOR_F, None, 1)]
    assert all(not s.tree_cache.released for s in ring.stages)


def test_floor_off_is_told_zero(monkeypatch):
    ring = _floor_ring(monkeypatch, **{fb.ENV_FLOOR: "0"})
    assert _uniform(ring, RID_F) == 0
    assert _admits(ring) == [(0, 1, None)]


def test_floor_differing_acks_are_told_zero(monkeypatch):
    """A follower that acked MORE than the other proved nothing about the
    smaller depth: no floor, told=0 for every rank."""
    ring = _floor_ring(monkeypatch, acks=(FLOOR_F, FLOOR_F - 64))
    assert _uniform(ring, RID_F) == 0
    assert _admits(ring) == [(0, 1, None)]


def test_floor_never_from_an_ack_over_told(monkeypatch):
    ring = _floor_ring(monkeypatch, acks=(TOLD_F + 128, TOLD_F + 128))
    # an over-read acks told (satisfied) or its own reach; neither is a floor below told
    assert _uniform(ring, RID_F) in (0, TOLD_F)
    assert all(a[2] is None for a in _admits(ring))


def test_floor_refused_by_pp0_tf_is_told_zero(monkeypatch):
    from sglang.srt.managers import weg2_told_fidelity as tf

    real = tf.pp0_verdict

    def refuse_floor(scheduler, req, told, absolute=True):
        if int(told) == FLOOR_F:
            return 0, FLOOR_F - 1000
        return real(scheduler, req, told, absolute)

    monkeypatch.setattr(m._tf, "pp0_verdict", refuse_floor)
    ring = _floor_ring(monkeypatch)
    assert _uniform(ring, RID_F) == 0
    assert _admits(ring) == [(0, 1, None)]


def test_floor_waits_for_the_missing_ack_then_frist_is_zero(monkeypatch):
    """One ack below told, the other follower never acks: no floor on a
    partial picture -- the Frist answers told=0."""
    reads = {0: {RID_F: 0.1}, 1: {RID_F: 0.1}, 2: {RID_F: 10**6}}
    ring = _ring(monkeypatch, {RID_F: TOLD_F}, reads, 0.05)
    tree1 = ring.stages[1].tree_cache

    def short_read(req, rematch=True, limit_tokens=None):
        tree1.start_read(req.rid, FLOOR_F, 0.1)
        return "issued"

    ring.stages[1]._prefetch_kvcache = short_read
    ring.arrive(RID_F)
    ring.run(80)
    assert _uniform(ring, RID_F) == 0
    assert _admits(ring) == [(0, 1, None)]


def test_floor_decide_unit(monkeypatch):
    from types import SimpleNamespace

    _clean(monkeypatch)
    s = SimpleNamespace(ps=SimpleNamespace(pp_rank=0, pp_size=3))
    fb.pp0_open(s, "r", 1000, published_at=0.0, own_read_s=0.1, window_s=0.1, published_pass=1)
    o = fb._pp0_open_map(s)["r"]
    o.acks[1] = 900
    assert fb.pp0_decide(s, "r", now=0.5, pass_n=2) is None  # floor still possible: wait
    o.acks[2] = 900
    assert fb.pp0_decide(s, "r", now=0.5, pass_n=2) == (900, fb.REASON_FLOOR)
    o.acks[2] = 0
    assert fb.pp0_decide(s, "r", now=0.5, pass_n=2) == (0, fb.REASON_MISMATCH)
    o.acks[2] = 1000
    assert fb.pp0_decide(s, "r", now=0.5, pass_n=2) == (0, fb.REASON_MISMATCH)
    # equal acks OVER told (OVER-UNRESUMABLE acks its own read): never a floor
    o.acks[1] = o.acks[2] = 1100
    assert fb.pp0_decide(s, "r", now=0.5, pass_n=2) == (0, fb.REASON_MISMATCH)
    o.acks[1] = 900
    monkeypatch.setenv(fb.ENV_FLOOR, "0")
    o.acks[2] = 900
    assert fb.pp0_decide(s, "r", now=0.5, pass_n=2) == (0, fb.REASON_MISMATCH)


def test_followers_follow_the_floor_marker_only(monkeypatch):
    """A follower applies the floor because the WIRE says so (never its own
    env): with its own switch off it still follows PP0's marker."""
    ring = _floor_ring(monkeypatch)
    assert _uniform(ring, RID_F) == FLOOR_F
    # the follower side has no floor decision at all: apply_floor runs only on the marker
    import inspect

    src = inspect.getsource(m._follower_admit)
    assert "WIRE_FLOOR" in src and "floor_on" not in src


# ---------------------------------------------------------------------------
# the switches: all four 0 = the h16n behaviour on the PF ring
# ---------------------------------------------------------------------------


def test_switches_default_on(monkeypatch):
    for k in FIX_SWITCHES:
        monkeypatch.delenv(k, raising=False)
    assert fb.ack_drain_on() and fb.floor_on() and m.settle_ack_now_on()
    assert fb.min_passes(3) == 4


@pytest.mark.parametrize("dt", [0.05, 2.2])
def test_fast_ring_unchanged_by_the_fix(monkeypatch, dt):
    """A plain equal read with short passes admits told on its acks with and
    without the fix (the fix only acts where the wall Frist would pre-empt
    the passes)."""
    out = []
    for fixed in (True, False):
        reads = {r: {RID_B: 0.1} for r in range(3)}
        ring = _ring(monkeypatch, {RID_B: TOLD_B}, reads, 0.05, fixed=fixed)
        ring.arrive(RID_B)
        ring.run(30)
        out.append((_uniform(ring, RID_B), _admits(ring)))
    assert out[0] == out[1] == (TOLD_B, [(TOLD_B, None, None)])
    del dt
