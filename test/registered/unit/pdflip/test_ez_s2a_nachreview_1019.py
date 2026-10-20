"""EZ-S2a review Nachbesserung (nf-review-ez-s2a-1019.md, Auflagen A1, A2; SOLL R7 R2 R2a R3 R1a).

A1 (S2a-1, R7): with ``FLLIPER_PDFLIP_DECIDER`` ON an entry D refused under its X (``x_requeues`` with a named extent and no
   ``measure_class``; ``x_routed`` / ``x_deferred`` under ``x_set``) has EXACTLY ONE owner (plan B2/E11 WAIT_SEAT):
   kurz -> ``d_eligible``, handed by ``_asr_queued_short_to_d`` at the decision X >= u; halb -> set member, D at
   ``D_AFTER_TIMER`` (``d_eligible``, handed), or rides a decided flip (R1a). A D verdict that is not carried out is not decided
   and logged again per tick (no SET-VERDICT loop). Real path ``_requeue_after_x_refusal`` -> ``_decode_collect`` ->
   ``_asr_queued_short_to_d``, switch ON; one counter-test per file that was pinned OFF (#246b, flipqueue, 27b_park2,
   27b_park_immediate), each holding the danger direction "every entry has an owner".
A2: R2a in ``needs_p_class`` / ``immediate_park_trigger`` -- ``x_set`` != the static X, both directions; ``_set_x_for`` = the
   SMALLEST own X (RV9).
"""
import logging
import os
import sys
import time
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flliper.srt.environ import envs  # noqa: E402
from flliper.srt.pdflip import front as F  # noqa: E402
from flliper.srt.pdflip import phase_policy as pp  # noqa: E402
from test_ez_s1_review_1019 import _fresh, _pend, _refuse, X, HALF_X  # noqa: E402


@pytest.fixture(autouse=True)
def decider_on():
    with envs.FLLIPER_PDFLIP_DECIDER.override(True):
        yield


def _d_awake(f):
    f.awake, f.state, f.admit_d = "D", "serving", True
    f._dc_decoding = lambda: True
    return f


def _queue(f, rid, u, **kw):
    p = _pend(rid)
    p.est_uncached = u
    p.d_direct = False
    for k, v in kw.items():
        setattr(p, k, v)
    f.queue.append(p)
    return p


def _live(f):
    return [q for q in f.queue if not q.fut.done()]


def _settle(f, ticks=5, step=30.0, hand=True):
    """T_halb long over; ``ticks`` controller decisions; then the step that hands queued SHORTs to D."""
    now = time.time() + step
    dcs = [f._decode_collect(now + i) for i in range(ticks)]
    handed = [h.rid for h in F.Front._asr_queued_short_to_d(f, _live(f), now + ticks)] if hand else []
    return dcs, handed


def _set_verdict_lines(caplog):
    return sum(1 for m in caplog.messages if "SET-VERDICT" in m)


def _with_owner_record(f, rid):
    return any(r.get("rid") == rid and r.get("action") == "WAIT_SEAT" for r in getattr(f._dec(), "_seen", []) or [])


# ------------------------------------------------------------------------------------- A1 (the reviewer's orphan, switch ON)

@pytest.mark.parametrize("u", [1000, HALF_X - 1])
def test_a1_kurz_refused_under_x_has_one_owner_d(u, caplog):
    f = _d_awake(_fresh())
    p = _refuse(f, "pdflip-0-1", u)
    assert p.x_requeues == 1 and not p.measure_class and not p.d_eligible
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        dcs, handed = _settle(f)
    assert dcs == [None] * 5                                   # no set, no flip for a kurz entry (R7/R2)
    assert handed == ["pdflip-0-1"]                            # the ONE owner: D's admission line
    assert p in f._ready_for_d and p not in f.queue
    assert p.rid not in f._set_st()["go_p"]                    # not P
    assert _set_verdict_lines(caplog) == 0
    assert f.counters["wait_seat_kurz_to_d"] == 1


@pytest.mark.parametrize("u", [HALF_X, X - 1])
def test_a1_halb_refused_under_x_one_set_verdict_d_after_timer_then_d(u, caplog):
    f = _d_awake(_fresh())
    p = _refuse(f, "pdflip-0-1", u)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        dcs, handed = _settle(f, ticks=8)
    assert dcs == ["D"] + [None] * 7                           # decided ONCE (R3: the timer ended, S < X)
    assert handed == ["pdflip-0-1"] and p in f._ready_for_d
    assert _set_verdict_lines(caplog) == 1                     # no SET-VERDICT per tick
    assert f.counters["wait_seat_halb_to_d"] == 1 and not f._set_st()["go_p"]


def test_a1_a_d_verdict_nobody_can_carry_out_is_not_decided_every_tick(caplog):
    """The reviewer's loop: D_AFTER_TIMER on every call with a fresh set_id and one SET-VERDICT line each (5/s). The verdict is
    D's; if D cannot take the entry now it waits (WAIT_SEAT) -- no new verdict, no new line."""
    f = _d_awake(_fresh())
    _refuse(f, "pdflip-0-1", 3500)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        dcs, handed = _settle(f, ticks=40, hand=False)      # the hand-over does not run (D busy / not admitting)
    assert dcs.count("D") == 1 and handed == []
    assert _set_verdict_lines(caplog) == 1
    assert f._set_st()["set_id"] == 1                          # one set was decided, not forty


def test_a1_halb_state_entry_rides_a_decided_flip_r1a():
    """halb + halb: S = 3500 + 3500 >= X -> FLIP_P; the entry D refused is a member and committed (R1a)."""
    f = _d_awake(_fresh())
    a = _refuse(f, "pdflip-0-1", 3500)
    b = _refuse(f, "pdflip-0-2", 3500)
    dcs, handed = _settle(f)
    assert dcs[0] == "P" and handed == []
    assert {a.rid, b.rid} <= f._set_st()["go_p"]


def test_a1_kurz_state_entry_rides_a_decided_flip_r1a():
    """A decided flip takes the kurz entry D refused along (R1a) -- not left behind in the queue without an owner."""
    f = _d_awake(_fresh())
    k = _refuse(f, "pdflip-0-1", 1000)
    big = _queue(f, "pdflip-0-2", X + 500)                     # gross: P candidate (R4), D decodes -> waits T_halb
    dcs, handed = _settle(f)
    assert "P" in dcs
    st = f._set_st()
    assert big.rid in st["go_p"] and k.rid in st["go_p"]
    assert f.counters["wait_seat_rides_flip"] >= 1


def test_a1_x_deferred_and_x_routed_under_x_set_are_owned_too():
    f = _d_awake(_fresh())
    a = _queue(f, "pdflip-1-1", 1000, x_deferred=True)
    b = _queue(f, "pdflip-1-2", 1000, x_routed=500)            # routed LONG on a smaller X_busy, u under x_set
    dcs, handed = _settle(f)
    assert dcs == [None] * 5 and sorted(handed) == ["pdflip-1-1", "pdflip-1-2"]
    f2 = _d_awake(_fresh())
    c = _queue(f2, "pdflip-1-3", 3500, x_deferred=True)        # halb: a member, D at D_AFTER_TIMER
    dcs2, handed2 = _settle(f2)
    assert dcs2[0] == "D" and handed2 == ["pdflip-1-3"] and c in f2._ready_for_d


@pytest.mark.parametrize("kw", [dict(p_only=True), dict(reroutes=1), dict(x_ref_nonum=True), dict(leg1_done=True),
                                dict(skip_leg1=True)])
def test_a1_what_is_not_owned_here_stays_out(kw):
    """R9/N1 (p_only, leg-2 reroute), D's word without a number (P), a leg done / skipped: no WAIT_SEAT adoption."""
    f = _d_awake(_fresh())
    p = _queue(f, "pdflip-2-1", 1000, x_requeues=1, **kw)
    assert not F.Front._wait_seat_owned(f, p)
    _dcs, handed = _settle(f)
    assert handed == []


def test_a1_a_measured_entry_keeps_its_b1_owner():
    f = _d_awake(_fresh())
    p = _queue(f, "pdflip-2-2", 1000, x_requeues=1, measure_class="kurz")
    assert not F.Front._wait_seat_owned(f, p)                  # B1 executes it by D's class


def test_a1_switch_off_is_the_s1_code():
    with envs.FLLIPER_PDFLIP_DECIDER.override(False):
        f = _d_awake(_fresh())
        p = _refuse(f, "pdflip-0-1", 1000)
        assert not F.Front._wait_seat_owned(f, p)
        assert pp.immediate_park_trigger([p], X) is p          # S1: P by state (the pinned tests describe this)


# ----------------------------------------- A1: one counter-test per file that is pinned OFF (switch ON, danger direction)

def test_a1_pin_246b_x_routed_band_entry_has_an_owner_on():
    """test_pdflip_arrival_seat_x_routed_246b_0930 (X_live 4964, routed on 4096): u=4447 and 4191 are halb on x_set -> a
    member, D at the timer; before: P by state. No entry is left in the queue unowned."""
    f = _d_awake(_fresh())
    f.tp_prefill_max_tokens = 4964
    a = _queue(f, "pdflip-22-37", 4447, x_routed=4096, x_set=4964)
    b = _queue(f, "pdflip-22-38", 300, x_routed=250, x_set=4964)      # the 1216 / pdflip-28-76 shape: kurz (sum fits X)
    assert pp.immediate_park_trigger([a, b], 4964) is None            # R7: no flip under X
    dcs, handed = _settle(f)
    assert sorted(handed) == ["pdflip-22-37", "pdflip-22-38"] and not _live(f)


def test_a1_pin_flipqueue_refused_under_x_does_not_need_p_but_has_an_owner():
    """test_pdflip_flipqueue_1015::test_p_only_and_refused_over_x_need_p -- ON: a W31 refusal at u=10 is no P work; D owns it."""
    f = _d_awake(_fresh())
    p = _refuse(f, "w31", 10)
    assert not F.Front._p_phase_needs_p(f, p)                  # the P phase may end without it (no double flip)
    assert F.Front._p_phase_needs_p(f, _queue(f, "v", 10, p_only=True))   # R9 stays P
    _dcs, handed = _settle(f)
    # EZ-S2a H8 (one owner): the p_only "v" decides a flip (R9/R4) -- w31 is then its R1a rider (go_p), no longer ALSO handed to D
    assert "w31" in f._set_st()["go_p"] and "w31" not in handed


def test_a1_pin_flipqueue_refused_under_x_without_a_flip_is_handed_to_d():
    f = _d_awake(_fresh())
    _refuse(f, "w31", 10)
    _dcs, handed = _settle(f)
    assert "w31" in handed and "w31" not in f._set_st()["go_p"]


def test_a1_pin_27b_park2_band_deferred_has_an_owner_on():
    """test_27b_park2_metal_0927 (pdflip-28-62: u 4172, live X 7290, x_deferred): ON no immediate park for it, D takes it;
    prefilled already (leg1_done) waits for D, not P (as before)."""
    f = _d_awake(_fresh())
    f.tp_prefill_max_tokens = 7290
    band = _queue(f, "pdflip-28-62", 4172, x_deferred=True, x_set=7290)
    assert pp.immediate_park_trigger([band], 7290) is None
    _dcs, handed = _settle(f)
    assert handed == ["pdflip-28-62"]
    done = SimpleNamespace(rid="d", est_uncached=4172, skip_leg1=False, leg1_done=True, p_only=False, x_requeues=0,
                           x_deferred=True)
    assert pp.immediate_park_trigger([done], 7290) is None


def test_a1_pin_27b_park_immediate_trigger_is_over_x_or_p_only_on():
    """test_27b_park_immediate::test_needs_p_is_exactly_over_x_or_p_only -- ON: AT X is gross (R4), p_only is P (R9),
    a refusal under X is none of these (R7) and has its owner, a leg done / skipped is not P."""
    Xs = 12288
    assert pp.needs_p_class(Xs, Xs) and not pp.needs_p_class(Xs - 1, Xs)
    assert pp.needs_p_class(10, Xs, p_only=True)
    assert not pp.needs_p_class(10, Xs)                                   # W31 under X: not P by state
    assert not pp.needs_p_class(10 * Xs, Xs, leg1_done=True) and not pp.needs_p_class(10 * Xs, Xs, skip_leg1=True)
    f = _d_awake(_fresh())
    q = _refuse(f, "refused", 10)
    assert pp.immediate_park_trigger([q], X) is None and F.Front._wait_seat_owned(f, q)


# ------------------------------------------------------------------------------------------------- A2: R2a on x_set

def _p(u, **kw):
    base = dict(rid="r", est_uncached=u, skip_leg1=False, leg1_done=False, p_only=False, x_requeues=0, x_set=0)
    base.update(kw)
    return SimpleNamespace(**base)


def test_a2_x_set_above_the_static_x_a_request_between_is_not_gross():
    """x_set 6000 > static X 4096: u 5000 is under ITS X -> no P candidate, no flip under its own X (RV2 flips)."""
    assert pp.immediate_park_trigger([_p(5000, x_set=6000)], 4096) is None
    assert pp.immediate_park_trigger([_p(6000, x_set=6000)], 4096) is not None      # AT its own X: gross


def test_a2_x_set_below_the_static_x_a_request_between_is_gross():
    """x_set 3000 < static X 4096: u 3500 is over ITS X -> P candidate (RV2 misses it)."""
    q = _p(3500, x_set=3000)
    assert pp.immediate_park_trigger([q], 4096) is q
    assert pp.immediate_park_trigger([_p(2999, x_set=3000)], 4096) is None


def test_a2_no_x_set_falls_back_to_the_static_x():
    assert pp.immediate_park_trigger([_p(4095)], 4096) is None
    assert pp.immediate_park_trigger([_p(4096)], 4096) is not None


def test_a2_wait_seat_owner_reads_x_set_not_the_static_x():
    f = _d_awake(_fresh())
    over_static = _queue(f, "o", 5000, x_requeues=1, x_set=6000)      # under its own X: owned
    under_static = _queue(f, "w", 3500, x_requeues=1, x_set=3000)     # over its own X: P work, not owned
    assert F.Front._wait_seat_owned(f, over_static) and not F.Front._wait_seat_owned(f, under_static)


def test_a2_rv9_s_is_decided_against_the_smallest_member_x():
    """RV9 (max instead of min): A (x_set 2000, u 1100) + B (x_set 3500, u 1800): S = 2900. The smallest X (2000) is
    reached -> FLIP_P; against the largest (3500) it would collect."""
    f = _d_awake(_fresh())
    assert F.Front._set_x_for(f, [("a", 0, 2000), ("b", 0, 3500)]) == 2000
    a = _queue(f, "a", 1100, x_set=2000, x_requeues=1)    # halb members on their own X (state entries classed by u)
    b = _queue(f, "b", 1800, x_set=3500, x_requeues=1)
    members, sv, kw = f._set_eval(time.time() + 1.0, _live(f), f._set_st())
    assert kw["x_tok"] == 2000 and sv.s_tokens == 2900
    assert sv.verdict in (pp.V_FLIP_P, pp.V_PARK_NO_SEAT, pp.V_REROUTE_P)
