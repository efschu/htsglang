"""LOCKSTEP flip order (NF y7o, 02.10. 14:06:18-14:07:54, D->P ep3/9/11/13).

Front 14:06:18.987 `WEG2-FLIP-ORDER epoch=2 src=D driver_free={0: 1455, 1: 2150,
2: 1213}` sent the LEAST-DEFICIT order 0,9,1,14,10,2,... PP2's first claim
`weights_14` (4014 MiB) stood at position 3. D TP2 (card 2, tags 1048 MiB,
weights_1 1148) had paused weights_0/9/1/14 by 14:06:19.575 -- published
4292 MiB, 406 MiB on-card staging booked -> balance 3885, OVERDRAWN by 128
(P log 14:06:20 `WEG2-CREDIT-EARLY tag=weights_14 ... floor=858 MiB ...
balance 3885 MiB`). The card path needs free >= 4014 + 858. TP2's NEXT pause
(weights_10, deposit to PP1 on card 0) waited 753 ms (CYCLE-SPILL), TP1's
weights_14 deposit to PP2 926 ms, TP0's 964 ms: PP2 1298 ms, PP1 weights_10
1084 ms, PP0 weights_2 904 ms -- the three-rank convoy. Fast flips (ep1/5/7)
max 481-567 ms.

The fix moves the ORDER only (wake_credit.lockstep_claims): every waker's
first claim stands behind the co-located sleeper pauses that fund it. No
reserve, no expert cap, no floor change.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.weg2_memory_saver import (  # noqa: E402
    VramCredit,
    Weg2VramCreditRefused,
)
from sglang.srt.weg2 import wake_credit as wc  # noqa: E402

MIB = 1 << 20


def _uniform(v, tags, **over):
    d = {t: float(v) for t in tags}
    d.update({k: float(x) for k, x in over.items()})
    return d


_ALL = ["weights"] + ["weights_%d" % i for i in range(16)]

#: the front's --wake-credit-plan of y7o, "D->P" (front.log WEG2-LAUNCH argv,
#: rounded to MiB -- the planner's table holds D's boot-start tag sizes)
Y7O_TABLE = [
    {"card": 1, "sleeper": "D TP0", "waker": "P PP0",
     "release": dict(_uniform(1208, _ALL, weights=1258, weights_0=1250, weights_1=1276,
                              weights_3=1212, weights_4=1218, weights_7=1212, weights_8=1218,
                              weights_11=1212, weights_12=1218, weights_15=1212),
                     weights_draft=1556.0),
     "demand": {"weights": 630.0, "weights_0": 1754.0, "weights_1": 1762.0, "weights_2": 1702.0,
                "weights_3": 1702.0, "weights_4": 1708.0, "weights_5": 1702.0, "weights_6": 1702.0,
                "weights_7": 1702.0, "weights_8": 1708.0, "weights_9": 1137.0},
     "oncard": {"weights": 616.0, "weights_0": 351.0, "weights_1": 311.0, "weights_2": 312.0,
                "weights_3": 312.0, "weights_4": 319.0, "weights_5": 312.0, "weights_6": 312.0,
                "weights_7": 312.0, "weights_8": 319.0, "weights_9": 206.0}},
    {"card": 0, "sleeper": "D TP1", "waker": "P PP1",
     "release": _uniform(877, _ALL, weights=2, weights_1=977),
     "demand": {"weights_10": 3149.0, "weights_11": 3085.0, "weights_12": 3091.0,
                "weights_13": 1031.0, "weights_9": 1037.0},
     "oncard": {"weights_10": 507.0, "weights_11": 507.0, "weights_12": 507.0,
                "weights_13": 169.0, "weights_9": 169.0}},
    {"card": 2, "sleeper": "D TP2", "waker": "P PP2",
     "release": _uniform(935, _ALL, weights=2, weights_1=1035),
     "demand": {"weights": 650.0, "weights_13": 2655.0, "weights_14": 4016.0,
                "weights_15": 3956.0},
     "oncard": {"weights": 1.0, "weights_13": 384.0, "weights_14": 576.0, "weights_15": 576.0}},
]
FLOORS = {0: 1095.0, 1: 1055.0, 2: 858.0}
#: the interleave the credit order starts from (front epoch 0, uncycled)
GIVEN = ["weights_0", "weights_9", "weights_14", "weights_1", "weights_10", "weights_15",
         "weights_2", "weights_11", "weights_3", "weights_12", "weights_4", "weights_13",
         "weights_5", "weights_6", "weights_7", "weights_8", "weights"]
#: driver_free at the slow D->P flips (front epochs 2/8/10/12) and a fast one (4)
SLOW = {"ep3": {0: 1455, 1: 2150, 2: 1213}, "ep9": {0: 1495, 1: 2192, 2: 1251},
        "ep11": {0: 1489, 1: 2188, 2: 1251}, "ep13": {0: 1505, 1: 2194, 2: 1207}}
#: the order the front SENT at ep3 (LEAST-DEFICIT, +512 MiB)
EP3_SENT = ["weights_0", "weights_9", "weights_1", "weights_14", "weights_10", "weights_2",
            "weights_3", "weights_4", "weights_15", "weights_11", "weights_5", "weights_12",
            "weights_6", "weights_13", "weights_7", "weights_8", "weights"]

#: card 2 at the metal (D log 14:06:18 WEG2-DC-BREAKDOWN TP2 tms_resident;
#: P log WEG2-VRAM-CREDIT / WEG2-CREDIT-EARLY tag=weights_14)
TP2_TAG_MIB = {"weights_1": 1148}
TP2_TAG_DEFAULT_MIB = 1048
PP2_FIRST_NEED_MIB = 4014
FLOOR2_MIB = 858
STAGED_MIB = 406           # TP2's booked on-card staging of weights_14
PP2_FREE_AT_BEGIN_MIB = 829  # P log `WEG2-RESUME begin tag=weights_14 ... free_mib=829`
RING_LIVE_MIB = 2 * 426    # TP2's on-card ring, two slots of slot_bytes=426040320


def _front(free, lockstep):
    return wc.front_order(GIVEN, Y7O_TABLE, free_mib=free, floor_mib=FLOORS,
                          double_staging=False, least_deficit=True, lockstep_first=lockstep)


def _first(order, card):
    c = [x for x in Y7O_TABLE if x["card"] == card][0]
    for k, t in enumerate(order):
        if float(c["demand"].get(t, 0.0)) > 0.0:
            return k, t
    raise AssertionError("no claim on card %d" % card)


def test_without_lockstep_the_front_sends_the_y7o_order():
    """The base: the order of the slow flip, PP2's first claim at position 3."""
    order, why = _front(SLOW["ep3"], lockstep=False)
    assert order == EP3_SENT
    assert _first(order, 2) == (3, "weights_14")


@pytest.mark.parametrize("ep", sorted(SLOW))
def test_lockstep_puts_every_first_claim_behind_the_pauses_that_fund_it(ep):
    order, why = _front(SLOW[ep], lockstep=True)
    assert sorted(order) == sorted(GIVEN) and order[-1] == "weights"
    assert "LOCKSTEP (y7o)" in why
    cards, _ = wc.front_cards(order, Y7O_TABLE, free_mib=SLOW[ep], floor_mib=FLOORS)
    for line in wc.leg_order_lines(order, cards):
        assert "lockstep=funded" in line, line
    # PP2's first claim on card 2 comes AFTER at least one more of TP2's
    # cross-card pauses than in the sent order -- the user's ask: D pauses
    # PP0's tags (card 1) first, PP2's 13/14 later.
    k_new, t_new = _first(order, 2)
    k_old, t_old = _first(wc.front_order(GIVEN, Y7O_TABLE, free_mib=SLOW[ep], floor_mib=FLOORS,
                                         double_staging=False, least_deficit=True)[0], 2)
    assert k_new > k_old


def test_ep3_order_is_the_smallest_move_weights_2_before_weights_14():
    order, why = _front(SLOW["ep3"], lockstep=True)
    assert order[:6] == ["weights_0", "weights_9", "weights_1", "weights_2", "weights_14",
                         "weights_10"]
    assert order[6:] == EP3_SENT[6:]


def test_a_funded_order_stays_byte_identical():
    """ep1 (driver_free {0: 6493, 1: 6348, 2: 4755}): every first claim is
    funded, the lockstep pass changes nothing -- the 27B line and every
    fast form keep their order."""
    free = {0: 6493, 1: 6348, 2: 4755}
    assert _front(free, lockstep=True) == _front(free, lockstep=False)


def test_the_leg_order_marker_names_card_first_claim_and_paused_first():
    order, _ = _front(SLOW["ep3"], lockstep=True)
    lines = wc.front_leg_order_lines(order, Y7O_TABLE, free_mib=SLOW["ep3"], floor_mib=FLOORS,
                                     double_staging=False)
    card2 = [ln for ln in lines if ln.startswith("WEG2-LEG-ORDER card=2 ")]
    assert card2 and "first_claim=weights_14 pos=4" in card2[0]
    assert "paused_first=weights_0,weights_9,weights_1,weights_2,weights_14" in card2[0]


def _tp2_counter(tmp_path, name, order):
    """Card 2's book when PP2 asks for its first tag: TP2 has paused every tag
    before PP2's first claim (each a cross-card deposit, collected by PP0/PP1)
    and the claim tag itself (on-card only), and staged 406 MiB for PP2."""
    k, first = _first(order, 2)
    c = VramCredit(name, credit_dir=str(tmp_path))
    c.begin_leg("y7o-ep3")
    published = 0
    for t in order[:k] + [first]:
        mib = TP2_TAG_MIB.get(t, TP2_TAG_DEFAULT_MIB)
        c.publish(t, mib * MIB)
        published += mib
    assert c.debit("ipc-stage-" + first, STAGED_MIB * MIB)
    c.mark_live(STAGED_MIB * MIB)
    free = (PP2_FREE_AT_BEGIN_MIB + published - RING_LIVE_MIB) * MIB
    return c, published, free


def test_the_y7o_convoy_on_card_2_balance_3885_against_4014(tmp_path):
    """RED half: the sent order. Balance 4292 - 406 = 3885 < 4014 and the card
    is short of 4014 + 858 -- the grant needs a LATER TP2 pause, i.e. the
    convoy. Within a short budget it is refused (on the metal: 1298 ms)."""
    c, published, free = _tp2_counter(tmp_path, "GPU-y7o-old", EP3_SENT)
    assert published == 4292
    assert c.read()["credit_bytes"] - c.read()["consumed_bytes"] == 3886 * MIB
    with pytest.raises(Weg2VramCreditRefused):
        c.wait_for(PP2_FIRST_NEED_MIB * MIB, budget_s=0.3, tag="weights_14",
                   free_bytes_now=free, free_reader=lambda: free,
                   floor_bytes=FLOOR2_MIB * MIB, epoch="y7o-ep3", poll_s=0.02)


def test_the_lockstep_order_funds_pp2s_first_tag_from_the_published_balance(tmp_path):
    """GREEN half: the lockstep order. TP2 has published one more cross-card
    pause before PP2's first claim; the balance covers 4014 at once, no card
    overdraw, no wait on a later pause."""
    order, _ = _front(SLOW["ep3"], lockstep=True)
    c, published, free = _tp2_counter(tmp_path, "GPU-y7o-new", order)
    assert published - STAGED_MIB >= PP2_FIRST_NEED_MIB
    rec = c.wait_for(PP2_FIRST_NEED_MIB * MIB, budget_s=0.3, tag="weights_14",
                     free_bytes_now=free, free_reader=lambda: free,
                     floor_bytes=FLOOR2_MIB * MIB, epoch="y7o-ep3", poll_s=0.02)
    assert rec["waited_s"] == 0.0
    assert rec["claimed_bytes"] == PP2_FIRST_NEED_MIB * MIB
    assert rec["available_bytes"] >= 0          # the counter covered it, no overdraw


def test_front_applies_it_on_d_to_p_only_and_the_switch_turns_it_off():
    from sglang.srt.weg2 import front

    class _Card:
        def __init__(self, i, free):
            self.nvml_index, self.free_mib, self.uuid = i, free, "GPU-%d" % i

    cards = [_Card(i, f) for i, f in SLOW["ep3"].items()]
    plan = {"D->P": Y7O_TABLE, "P->D": Y7O_TABLE}
    floor_of = lambda uuid: int(FLOORS[int(uuid.split("-")[1])])  # noqa: E731
    on, why = front.credit_pause_order(list(GIVEN), "rr", plan, "D", "P", cards, floor_of=floor_of)
    assert "LOCKSTEP (y7o)" in why and on != EP3_SENT
    pd, why_pd = front.credit_pause_order(list(GIVEN), "rr", plan, "P", "D", cards,
                                          floor_of=floor_of)
    assert "LOCKSTEP" not in why_pd
    with envs.SGLANG_WEG2_ENABLE_FLIP_ORDER_LOCKSTEP.override(False):
        off, why_off = front.credit_pause_order(list(GIVEN), "rr", plan, "D", "P", cards,
                                                floor_of=floor_of)
    assert off == EP3_SENT and "LOCKSTEP" not in why_off
