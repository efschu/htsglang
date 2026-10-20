"""EZ S2a review B1 (S2a-1b, SOLL R7/R3) + hints H7/H8: a halb entry that D refuses a SECOND time after the D_AFTER_TIMER hand-over
has exactly one owner again (a set member), one SET-VERDICT line per round, never an orphan (#246b / S28-A2).
Switch FLLIPER_PDFLIP_DECIDER ON, real path ``_requeue_after_x_refusal``."""
import logging
import os
import sys
import time

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


def _live(f):
    return [q for q in f.queue if not q.fut.done()]


def _round(f, t0, ticks=5):
    dcs = [f._decode_collect(t0 + i) for i in range(ticks)]
    handed = [h.rid for h in F.Front._asr_queued_short_to_d(f, _live(f), t0 + ticks)]
    return dcs, handed


def _sv(caplog):
    return sum(1 for m in caplog.messages if "SET-VERDICT" in m)


def _owners(f, p):
    st = f._set_st()
    return dict(set_member=p in F.Front._set_cands(f, _live(f), st), go_p=p.rid in st["go_p"],
                d_eligible=bool(p.d_eligible), needs_p=pp.immediate_park_trigger([p], int(f.tp_prefill_max_tokens)) is not None)


def test_b1_halb_second_refusal_is_a_member_again_one_verdict_per_round(caplog):
    f = _d_awake(_fresh())
    u = HALF_X + 500
    p = _refuse(f, "pdflip-0-1", u)
    with caplog.at_level(logging.INFO, logger="pdflip.front"):
        dcs1, handed1 = _round(f, time.time() + 30.0)
        assert dcs1[0] == "D" and handed1 == ["pdflip-0-1"] and _sv(caplog) == 1
        assert p.rid in f._set_st()["d_after"]
        f._ready_for_d.remove(p)                       # D takes it and refuses again (n=2, no leg 1 ran -> requeue, not W35)
        p = _refuse(f, "pdflip-0-1", u, p=p)
        assert p.x_requeues == 2 and p.rid not in f._set_st()["d_after"]     # B1: the spent verdict is gone
        assert _owners(f, p) == dict(set_member=True, go_p=False, d_eligible=False, needs_p=False)   # exactly one owner
        dcs2, handed2 = _round(f, time.time() + 60.0)
    assert dcs2.count("D") == 1 and handed2 == ["pdflip-0-1"] and p in f._ready_for_d
    assert _sv(caplog) == 2                            # one SET-VERDICT line per round, none per tick
    assert not f._set_st()["go_p"]


def test_b1_kurz_second_refusal_still_adopted():
    f = _d_awake(_fresh())
    p = _refuse(f, "pdflip-0-1", 1000)
    _round(f, time.time() + 30.0)
    f._ready_for_d.remove(p)
    p = _refuse(f, "pdflip-0-1", 1000, p=p)
    dcs, handed = _round(f, time.time() + 60.0)
    assert dcs == [None] * 5 and handed == ["pdflip-0-1"]


def test_h8_wait_seat_adopt_skips_an_entry_committed_to_a_flip():
    f = _d_awake(_fresh())
    p = _refuse(f, "pdflip-0-1", 1000)
    f._set_st()["go_p"].add(p.rid)                     # R1a rider of a decided flip
    assert F.Front._wait_seat_adopt(f, _live(f)) == 0 and not p.d_eligible
    f._set_st()["go_p"].discard(p.rid)
    assert F.Front._wait_seat_adopt(f, _live(f)) == 1 and p.d_eligible


def test_h7_pbound_flip_now_names_the_oldest_member_not_a_kurz_rider(caplog):
    f = _d_awake(_fresh())
    rider = _pend("pdflip-0-1")
    rider.est_uncached = 1000
    rider.t_arrive = time.time() - 50.0                # older than the member
    member = _pend("pdflip-0-2")
    member.est_uncached = HALF_X + 100
    member.t_arrive = time.time() - 10.0
    with caplog.at_level(logging.WARNING, logger="pdflip.front"):
        ok, _fair, trig = F.Front._asr_pbound_flip_now(f, [rider, member], X, time.time(), trigger_pool=[member])
    assert ok and trig is member
    assert any("PBOUND-FLIP-NOW" in m and "rid=pdflip-0-2" in m for m in caplog.messages)
    ok, _fair, trig = F.Front._asr_pbound_flip_now(_d_awake(_fresh()), [rider, member], X, time.time())   # default: oldest of cands
    assert trig is rider
