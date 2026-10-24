"""REVIEW SIBLING-U-1024 (#229) acceptance tests: what happens to a sibling RIDER on a D verdict -- MUSS-1 / SOLL-1 to SOLL-4.

The review's two probes (P1, P2) asserted the DANGER on the reviewed head (green there = the danger was real). Here they are
turned round into rule tests of the planner decision 24.10. (variant (a), SOLL R1/R2/R7, R1a, R11): a rider whose kurz class
comes only from its R1u rest behind an anchor is kurz only AFTER the anchor is computed; before that its u is its own figure.
So a D verdict of its set does NOT release it in the same tick: it waits behind its D anchor until the anchor's d_pf_book row
is closed, then the S63 re-probe reads its real rest (kurz -> R2 D at once; else R3/R4). Anchor gone -> classed on its own
figure at once; the wait is bounded by T_halb (FLLIPER_PDFLIP_SIBLING_RIDER_WAIT, default ON).

Danger directions and their mutants (test/registered/unit/pdflip, each must turn at least one test here red):
  (i)  the rider is released with its anchor in the same tick -> D computes its own figure uncounted (over X, R1/R7);
  (ii) the wait has no bound / ignores the anchor's loss -> the rider hangs."""
from __future__ import annotations

import logging
import os
import time

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from test_pdflip_set_scheduler_1017 import _d_began, _fx, _on, _verdicts  # noqa: E402
from test_pdflip_arrival_seat_rule_0929 import _pending  # noqa: E402

from flliper.srt.pdflip import front as front_mod  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402

A912, R913 = "pdflip-80-912", "pdflip-80-913"
A179, R180 = "pdflip-40-179", "pdflip-40-180"


def _session(f, rid, sess, ids, depth):
    f.__dict__.setdefault("_sess_by_rid", {})[rid] = sess
    f.__dict__.setdefault("_set_ids", {})[rid] = np.asarray(ids, dtype=np.int32)
    f.__dict__.setdefault("_set_depth", {})[rid] = depth


def _waits(caplog):
    return [r.getMessage() for r in caplog.records if "PDFLIP SIBLING-RIDER-WAIT" in r.getMessage()]


def _pair_1536(monkeypatch, t0):
    """The 15:36 pair (X 6185, halbX 3092): 912 halb 5396 + 913 own 5534 halb / rest 1502 kurz behind 912 (both SHORTs)."""
    _on(monkeypatch)
    f = _fx()
    f.tp_prefill_max_tokens = 6185
    _session(f, A912, "a4c8b097ac", list(range(25428)), 20032)
    _session(f, R913, "a4c8b097ac", list(range(24064)) + [-i - 1 for i in range(25566 - 24064)], 20032)
    st = f._set_st()
    st["shorts"][A912] = (t0, 5396, 6185)
    st["shorts"][R913] = (t0 + 1.5, 5534, 6185)
    return f, st


def _pair_1033(monkeypatch, t0, running=()):
    """The 10:33:42 shape (X 4096): anchor halb SHORT 3950, rider = queue entry own 4108 / rest 162."""
    _on(monkeypatch)
    f = _fx(running=running)
    f.tp_prefill_max_tokens = 4096
    _session(f, A179, "s1033", list(range(60000 + 3950)), 60000)
    _session(f, R180, "s1033", list(range(60000 + 3950 - 4)) + [-i - 1 for i in range(166)], 60000)
    st = f._set_st()
    st["shorts"][A179] = (t0 - 0.6, 3950, 4096)
    f.queue = [_pending(R180, 4108, t0 - 0.1, est_prompt=60000 + 4108)]
    return f, st


def _anchor_done(f, st, anchor):
    """D computed the anchor: its seat path picked the release up, its d_pf_book row closed at the first content, D holds it."""
    st["released"].pop(anchor, None)
    f._d_pf_book().done(rid=anchor)
    f.groups["D"].outstanding[anchor] = 1.0


def _anchor_gone(f, st, anchor):
    """The anchor left D (abort / client gone / park / reroute): no row, D does not hold it."""
    st["released"].pop(anchor, None)
    f._d_pf_book().done(rid=anchor)
    f.groups["D"].outstanding.pop(anchor, None)


def _reprobe_stub(monkeypatch, apply=None):
    """S63 re-probe at the rider's release: records the call; ``apply(rid)`` plays the new exact number D gives."""
    calls = []

    def gate(self, rids, site, verdict):
        calls.append((tuple(rids), site, verdict))
        if apply is not None:
            for r in rids:
                apply(r)
        return True
    monkeypatch.setattr(front_mod.Front, "_set_reprobe_gate", gate)
    return calls


# ------------------------------------------------------------------------------------------ MUSS-1 (P1 turned round)

def test_muss1_r1_r2_r7_halb_short_rider_is_not_released_to_d_with_its_anchor(monkeypatch, caplog):
    """P1 turned round (SOLL R1/R2/R7, Nutzergesetz "D prefillt nie ueber X"): at T_halb the 15:36 set goes to D
    (S=5396 < X, R3) -- the ANCHOR 912 is released, the rider 913 is NOT (its D figure before 912 is computed is its own
    5534; 5396 + 1502 = 6898 > X would be on D uncounted). It waits behind 912; after 912's D prefill the re-probe reads
    its rest 1502 -> kurz -> R2 D at once."""
    t0 = time.time()
    f, st = _pair_1536(monkeypatch, t0)
    calls = _reprobe_stub(monkeypatch, apply=lambda r: st["shorts"].__setitem__(r, (t0 + 1.5, 1502, 6185)))
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 1.5) == "hold"
        route = f._decode_collect(t0 + 4.7)
        lines = _verdicts(caplog)
        assert route == "D", (route, lines)
        assert "verdict=D_AFTER_TIMER" in lines[-1] and "S=5396 " in lines[-1] and "sibling-rider=pdflip-80-913:1502" in lines[-1]
        assert st["released"].get(A912) == "D"
        assert R913 not in st["released"], st["released"]                       # (i): not in the anchor's tick
        assert st["rider_hold"][R913]["anchor"] == A912 and R913 in st["shorts"]
        (begin,) = _waits(caplog)
        assert "begin rid=pdflip-80-913 anchor=pdflip-80-912 own=5534 rest=1502 where=short" in begin, begin
        # the anchor still prefills on D: the rider keeps waiting, it is no member (no set, no verdict on it)
        assert f._decode_collect(t0 + 4.8) is None
        assert R913 not in st["released"] and R913 in st["rider_hold"] and not calls
        _anchor_done(f, st, A912)
        assert f._decode_collect(t0 + 5.2) == "D"                                 # re-probed: kurz -> R2 D at once
    assert calls == [((R913,), "sibling-rider", "RIDER_RELEASE")], calls
    assert st["released"].get(R913) == "D" and R913 not in st.get("rider_hold", {})
    assert "verdict=D_AFTER_TIMER" in _verdicts(caplog)[-1] and "R2 kurz only" in _verdicts(caplog)[-1]
    assert any("outcome=release_rest" in w for w in _waits(caplog)), _waits(caplog)
    assert f.counters["sibling_rider_wait"] == 1 and f.counters["sibling_rider_wait_release_rest"] == 1


def test_muss1_rider_with_a_halb_rest_after_the_reprobe_is_classed_r3(monkeypatch, caplog):
    """After the anchor: a re-probe that reads a rest that is NOT kurz (D evicted part of the prefix) -> the ordinary class
    (R3 halb): the rider is a set member again on that number, never released on the old rest."""
    t0 = time.time()
    f, st = _pair_1536(monkeypatch, t0)
    _reprobe_stub(monkeypatch, apply=lambda r: st["shorts"].__setitem__(r, (t0 + 5.0, 4000, 6185)))
    assert f._decode_collect(t0 + 4.7) == "D"
    _anchor_done(f, st, A912)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 5.1) == "hold"                               # R3: the halb rider collects again
    (line,) = _verdicts(caplog)
    assert "members=pdflip-80-913:4000" in line and "classes=halb" in line and "verdict=COLLECT" in line, line
    assert R913 not in st["released"]


# ------------------------------------------------------------------------------------------ SOLL-1 (P2 turned round)

def test_soll1_r8_queue_rider_does_not_flip_alone_after_its_anchors_d_verdict(monkeypatch, caplog):
    """P2 turned round (SOLL R8 + R2/R7): nothing decodes on D; tick 1 decides D at once (S=3950 < X, R8) -- the anchor goes,
    the queue rider waits behind it. Tick 2 does NOT flip it alone on its own 4108 (no set member, no P trigger). Once the
    anchor's D prefill is over the re-probe reads 162 -> kurz, D's (d_eligible), no flip at all."""
    t0 = time.time()
    f, st = _pair_1033(monkeypatch, t0)
    (q,) = f.queue

    def _num(r):
        q.est_uncached, q.d_eligible = 162, True          # what _set_reprobe_apply does for a queue entry crossing X downwards
    calls = _reprobe_stub(monkeypatch, apply=_num)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        r1 = f._decode_collect(t0)
        lines1 = list(_verdicts(caplog))
        r2 = f._decode_collect(t0 + 0.05)
    assert r1 == "D", (r1, lines1)
    assert "sibling-rider=pdflip-40-180" in lines1[-1] and "S=3950 " in lines1[-1] and "R8" in lines1[-1], lines1
    assert st["released"].get(A179) == "D"
    assert r2 is None, (r2, _verdicts(caplog))                                   # SOLL-1: no lone R8 flip
    assert R180 not in st["go_p"] and q.sibling_hold == A179
    assert pp.immediate_park_trigger([q], 4096) is None                          # no P trigger while it waits
    assert f._set_flip_justified(t0 + 0.06) is False
    _anchor_done(f, st, A179)
    assert f._decode_collect(t0 + 1.0) is None
    assert calls == [((R180,), "sibling-rider", "RIDER_RELEASE")]
    assert q.sibling_hold == "" and q.d_eligible and R180 not in st["go_p"] and not st.get("rider_hold")


# ------------------------------------------------------------------------------------------ danger (ii): bounded, no hang

def test_danger_ii_anchor_gone_rider_is_classed_on_its_own_figure_at_once(monkeypatch, caplog):
    """The anchor leaves D (abort / park / death / reroute): the rider is classed on its own figure AT ONCE, no re-probe wait
    (planner: "Schranke, kein Haenger"). Here R8: own 4108 >= X -> it flips alone, which is then right."""
    t0 = time.time()
    f, st = _pair_1033(monkeypatch, t0)
    (q,) = f.queue
    calls = _reprobe_stub(monkeypatch)
    assert f._decode_collect(t0) == "D"
    assert f._decode_collect(t0 + 0.05) is None
    _anchor_gone(f, st, A179)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.1) == "P"
    assert R180 in st["go_p"] and q.sibling_hold == ""
    assert not [c for c in calls if c[1] == "sibling-rider"], calls
    assert any("outcome=release_own" in w and "anchor gone" in w for w in _waits(caplog)), _waits(caplog)
    assert "classes=gross" in _verdicts(caplog)[-1] and "members=pdflip-40-180:4108" in _verdicts(caplog)[-1]


def test_danger_ii_the_wait_is_bounded_by_t_half(monkeypatch, caplog):
    """The anchor stays on D (its row stays open): the rider waits at most T_halb (4.5 s, no number of its own), then its own
    figure classes it -- never a hang."""
    t0 = time.time()
    f, st = _pair_1033(monkeypatch, t0)
    (q,) = f.queue
    _reprobe_stub(monkeypatch)
    assert f._decode_collect(t0) == "D"
    assert f._decode_collect(t0 + 4.4) is None and q.sibling_hold == A179           # anchor pending, inside the bound
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 4.6) == "P"                                      # bound over: own 4108, R8
    assert q.sibling_hold == "" and R180 in st["go_p"]
    assert any("outcome=release_own" in w and "bound 4.5 s over" in w for w in _waits(caplog)), _waits(caplog)


def test_rider_hold_step_table():
    """The pure step (R11: the front calls it, the ranks never decide)."""
    assert pp.rider_hold_step(pp.ANCHOR_PENDING, 0.1, 4.5)[0] == pp.RIDER_HOLD
    assert pp.rider_hold_step(pp.ANCHOR_PENDING, 4.5, 4.5)[0] == pp.RIDER_RELEASE_OWN
    assert pp.rider_hold_step(pp.ANCHOR_DONE, 0.1, 4.5)[0] == pp.RIDER_RELEASE_REST
    assert pp.rider_hold_step(pp.ANCHOR_DONE, 9.0, 4.5)[0] == pp.RIDER_RELEASE_REST
    assert pp.rider_hold_step(pp.ANCHOR_GONE, 0.0, 4.5)[0] == pp.RIDER_RELEASE_OWN
    assert pp.rider_hold_step("", 0.0, 4.5)[0] == pp.RIDER_RELEASE_OWN


# ------------------------------------------------------------------------------------------ R1a, switch, epoch

def test_r1a_a_decided_flip_takes_the_waiting_riders(monkeypatch):
    """R1a: a flip decided while riders wait behind their D anchors takes them -- the SHORT rider is released to P, the queue
    rider committed (go_p); the waits end."""
    t0 = time.time()
    f, st = _pair_1536(monkeypatch, t0)
    assert f._decode_collect(t0 + 4.7) == "D" and R913 in st["rider_hold"]
    f.queue = [_pending("big-1", 9000, t0 + 4.8)]
    st["go_p"].add("big-1")                                  # a committed flip (an earlier verdict) is carried out
    assert f._decode_collect(t0 + 4.9) == "P"
    assert st["released"].get(R913) == "P" and R913 in st["go_p"] and not st.get("rider_hold")
    t1 = time.time()
    g, gst = _pair_1033(monkeypatch, t1, running=("dec",))
    assert g._decode_collect(t1 + 4.6) == "D"
    (q,) = g.queue
    assert q.sibling_hold == A179
    g.queue.append(_pending("big-2", 9000, t1 + 4.7))
    g.queue.append(_pending("big-3", 9000, t1 + 4.7))         # R4: a big member and company -> FLIP_P
    assert g._decode_collect(t1 + 4.8) == "P"
    assert R180 in gst["go_p"] and q.sibling_hold == "" and not gst.get("rider_hold")


def test_switch_off_is_the_pre_review_release_and_named(monkeypatch, caplog):
    t0 = time.time()
    f, st = _pair_1536(monkeypatch, t0)
    monkeypatch.setenv("FLLIPER_PDFLIP_SIBLING_RIDER_WAIT", "0")
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 4.7) == "D"
    assert st["released"].get(A912) == "D" and st["released"].get(R913) == "D"
    assert any("SIBLING-RIDER-WAIT off rids=pdflip-80-913" in w for w in _waits(caplog)), _waits(caplog)


def test_switch_default_on():
    from flliper.srt.environ import envs
    assert envs.FLLIPER_PDFLIP_SIBLING_RIDER_WAIT.default is True


def test_a_new_epoch_ends_every_wait(monkeypatch):
    t0 = time.time()
    f, st = _pair_1033(monkeypatch, t0)
    (q,) = f.queue
    assert f._decode_collect(t0) == "D" and q.sibling_hold == A179
    f.epoch += 1
    st2 = f._set_st()
    assert q.sibling_hold == "" and not st2.get("rider_hold")
    assert pp.immediate_park_trigger([q], 4096) is q                              # P's again in the new phase


# ------------------------------------------------------------------------------------------ SOLL-2: one presence set

def test_soll2_the_line_classes_against_the_same_members_as_decide(monkeypatch, caplog):
    """A SHORT anchor whose exact u is kurz (1500 < halbX 3092) is no member (R2, ``decide``'s own-figure filter): the
    sibling behind it is NOT anchored in the set -- classed on its own 4000 (halb), its share counted in S. The line must say
    the same (classes=kurz,halb, no sibling-rider), not ``kurz,kurz [... not in S]`` next to an S that counts it."""
    _on(monkeypatch)
    f = _fx()
    f.tp_prefill_max_tokens = 6185
    _session(f, "a", "s2", list(range(21500)), 20000)
    _session(f, "b", "s2", list(range(23000)) + [-i - 1 for i in range(1000)], 20000)
    t0 = time.time()
    st = f._set_st()
    st["shorts"]["a"] = (t0, 1500, 6185)
    st["shorts"]["b"] = (t0 + 0.2, 4000, 6185)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0 + 0.3) == "hold"
    (line,) = _verdicts(caplog)
    # b is no rider (its anchor is no member): classed on its own 4000 (halb), its R1u share 2500 counts in S
    assert "S=2500 " in line and "classes=kurz,halb" in line and "sibling-rider" not in line, line
    members = f._set_members(t0 + 0.3, [], st)
    keep, out = pp.set_keep(members, 6185, 0.5)
    assert out == ["a"] and [m.rid for m in keep] == ["b"]
    d = pp.decide("set", members=members, x_tokens=6185, half_frac=0.5, t_half_s=4.5, now=0.3, t_open=0.0,
                  d_decoding=True, seats_free=4, new_since_last=0)
    assert d.classes == ("halb",) and d.in_set == ("b",) and d.verdict.s_tokens == 2500


# ------------------------------------------------------------------------------------------ SOLL-3: presence at the front

def test_soll3_r1u_inflight_share_without_anchor_keeps_the_own_class(monkeypatch, caplog):
    """``_r1u_members`` (review A3: a member's share of a chunk D launched for a running prefill) sets ``own_uncached`` WITHOUT
    an anchor. The member's class stays its own figure (gross 4500 at X 4096), never its reduced rest (1500): R7 / D over X.
    Mutant M2 (class on the rest whenever own_uncached > 0) turns this red."""
    _on(monkeypatch)
    f = _fx()
    f.tp_prefill_max_tokens = 4096
    t0 = time.time()
    f.queue = [_pending("g", 4500, t0 - 1.0, est_prompt=30000)]

    def _run(self, running, members):
        self.__dict__["_r1u_member_off"] = 3000
        return running
    monkeypatch.setattr(front_mod.Front, "_r1u_running", _run)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0) == "hold"
    (line,) = _verdicts(caplog)
    assert "classes=gross" in line and "R4 big member waits" in line and "S=1500 " in line, line
    ev = f._set_eval(t0, list(f.queue), f._set_st())
    (m,) = ev[0]
    assert (m.uncached, m.own_uncached, m.anchor) == (1500, 4500, "")
    assert pp.member_class(m, 4096, 0.5, {"g"}) == pp.CLASS_GROSS
    assert pp.member_class(m, 4096, 0.5, ()) == pp.CLASS_GROSS
    # R8 (nothing decodes): the own class decides at once -> P, never a D verdict on the rest
    g = _fx(running=())
    g.tp_prefill_max_tokens = 4096
    g.queue = [_pending("g", 4500, t0 - 1.0, est_prompt=30000)]
    assert g._decode_collect(t0) == "P"


def test_soll3_pure_anchor_name_without_presence_classes_own():
    m = pp.SetMember(rid="b", uncached=163, t_arrive=1.0, own_uncached=4677, anchor="a")
    assert pp.member_class(m, 4096, 0.5, {"b"}) == pp.CLASS_GROSS               # anchor named, not present
    assert pp.member_class(m, 4096, 0.5, {"a", "b"}) == pp.CLASS_KURZ
    n = pp.SetMember(rid="b", uncached=163, t_arrive=1.0, own_uncached=4677, anchor="")
    assert pp.member_class(n, 4096, 0.5, {"a", "b"}) == pp.CLASS_GROSS          # reduced without an anchor


# ------------------------------------------------------------------------------------------ SOLL-4: R1a for R5 and R6

def test_soll4_r5_park_no_seat_counts_and_takes_the_rider():
    """R5: the seats are checked against ALL members, the rider too -- 1 seat < 2 members -> PARK_NO_SEAT, the rider in the
    set (rides the park, R1a); 2 seats -> FLIP_P."""
    m111 = pp.SetMember(rid="pdflip-16-111", uncached=4518, t_arrive=0.0)
    m112 = pp.SetMember(rid="pdflip-16-112", uncached=163, t_arrive=1.23, own_uncached=4677, anchor="pdflip-16-111")
    kw = dict(x_tokens=4096, half_frac=0.5, t_half_s=4.5, now=4.6, t_open=0.0, d_decoding=True, new_since_last=0)
    d1 = pp.decide("set", members=[m111, m112], seats_free=1, **kw)
    assert d1.verdict.verdict == "PARK_NO_SEAT" and "1 seats < 2 members" in d1.verdict.why, d1
    assert d1.in_set == ("pdflip-16-111", "pdflip-16-112") and d1.verdict.s_tokens == 4518
    assert pp.decide("set", members=[m111, m112], seats_free=2, **kw).verdict.verdict == "FLIP_P"


def test_soll4_r5_park_no_seat_at_the_front_commits_the_rider(monkeypatch):
    _on(monkeypatch)
    f = _fx(running=("d1", "d2", "d3", "d4", "d5"))          # 1 free seat of 6
    f.tp_prefill_max_tokens = 4096
    ids111 = list(range(69286))
    ids112 = list(range(69282)) + [-i - 1 for i in range(69445 - 69282)]
    for rid, ids in (("pdflip-16-111", ids111), ("pdflip-16-112", ids112)):
        _session(f, rid, "09c6f64aa8", ids, 64768)
    t0 = time.time()
    f.queue = [_pending("pdflip-16-111", 4518, t0 - 4.6, est_prompt=69286),
               _pending("pdflip-16-112", 4677, t0 - 4.6 + 1.23, est_prompt=69445)]
    assert f._decode_collect(t0) == "P"
    st = f._set_st()
    assert {"pdflip-16-111", "pdflip-16-112"} <= set(st["go_p"])
    assert {"pdflip-16-111", "pdflip-16-112"} <= set(st.get("park_no_seat", ()))


def test_soll4_r6_reroute_p_takes_the_rider(monkeypatch, caplog):
    """R6: a running D prefill (300, really begun) + anchor 3950 = 4250 > X 4096 -> REROUTE_P; the rider's kurz rest is not in
    S (R2) but it rides the reroute (R1a): released to P / committed."""
    t0 = time.time()
    f, st = _pair_1033(monkeypatch, t0, running=("dec",))
    f._d_pf_book().grant(rid="pdflip-40-170", tokens=300, now=time.time())
    f._d_pf_book().enter_leg2(rid="pdflip-40-170")
    f.groups["D"].outstanding["pdflip-40-170"] = 1.0
    _d_began(f, "pdflip-40-170")
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        assert f._decode_collect(t0) == "P"
    line = _verdicts(caplog)[-1]
    assert "verdict=REROUTE_P" in line and "S=4250 " in line and "sibling-rider=pdflip-40-180:162" in line, line
    assert R180 in st["go_p"] and st["released"].get(A179) == "P"
    d = pp.decide("set", members=[pp.SetMember(rid=A179, uncached=3950, t_arrive=0.0),
                                  pp.SetMember(rid=R180, uncached=162, t_arrive=0.5, own_uncached=4108, anchor=A179)],
                  x_tokens=4096, half_frac=0.5, t_half_s=4.5, now=0.6, t_open=0.0, d_decoding=True, seats_free=4,
                  d_prefill_running_uncached=300, new_since_last=0)
    assert d.verdict.verdict == "REROUTE_P" and R180 in d.in_set and d.verdict.s_tokens == 4250
