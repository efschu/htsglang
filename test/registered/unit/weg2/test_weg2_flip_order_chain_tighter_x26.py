"""23.09. (fnFL2x26): the chain interleave never overtakes a tighter card.

The destination is TP, so every resume costs EVERY card, while a card is
refunded only by the pauses of its own bands (``interleave_pause_order``,
tightest card first). The xsn367 chain interleave pulled the 5090's nine
bands ahead of the 3080s' on the Next-Flash geometry: card 0 paid seven
resumes before PP1's first pause, D TP1 overdrew its credit (balance 870 <
982 at weights_9), PP1's depth-1 drain wait for its next deposit then waited
for a collector that could not run -- the P<->D cycle, 180 s, W68, W17.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front  # noqa: E402

#: The Next-Flash P layout of fnFL2x26 (front.log WEG2-FLIP-ORDER MAP:
#: layer split [29, 11, 8] over 48 layers, 3 layers per band, nvml [1, 0, 2]
#: in stage order): nine bands on the 5090, four on card 0, three on card 2.
NF_CARDS = {**{f"weights_{i}": (1,) for i in range(9)},
            **{f"weights_{i}": (0,) for i in range(9, 13)},
            **{f"weights_{i}": (2,) for i in range(13, 16)}}
#: interleave_pause_order's answer on that map (tightest card first: card 2,
#: then card 0, then the 5090) -- x22's order, which flipped in 14.6 s.
NF_TIGHTEST = ([f"weights_{i}" for i in (13, 14, 15, 9, 10, 11, 12)]
               + [f"weights_{i}" for i in range(9)] + ["weights_draft", "weights"])
#: driver_free at the epoch-1 order of x26 (front.log), MiB.
NF_FREE = {0: 7211, 1: 7804, 2: 5567}
#: D TP1's cost per resume on card 0 (D.log RESUME begin need_mib=982) and
#: the corridor floor the credit wait keeps (WEG2-VRAM-CREDIT
#: corridor_floor_mib=700). Only the prefix BEFORE card 0's first refund is
#: modelled: what one PP1 pause gives back there is a per-band size the
#: front does not know (x22 measured +5.4 GiB at one step), and the walk
#: after the first refund is the credit ledger's business, not the order's.
NF_RESUME_MIB, NF_FLOOR_MIB = 982, 700

#: xsn367's own geometry (chain card 0 carries six of eight bands).
CARDS_27B = {"weights_0": (0,), "weights_1": (0,), "weights_2": (0,), "weights_3": (0,),
             "weights_4": (0,), "weights_5": (0, 1), "weights_6": (1, 2), "weights_7": (2,)}
TIGHTEST_27B = ["weights_6", "weights_7", "weights_0", "weights_1", "weights_2", "weights_3",
                "weights_4", "weights_5", "weights_draft", "weights"]


def _card0_drawdown_before_first_refund(order):
    """MiB card 0 pays before PP1's first pause refunds it: every chunk step
    resumes one of D's (TP) shards on every card; only a band of card 0's
    own stage gives anything back there."""
    n = 0
    for t in order:
        if t not in NF_CARDS:
            continue
        if NF_CARDS[t] == (0,):
            break
        n += 1
    return n * NF_RESUME_MIB


def test_the_xsn367_interleave_ran_card_0_under_its_floor_on_the_x26_geometry():
    """The red pin: without a free sample the chain card's bands are pulled
    ahead of the 3080s' and card 0 pays seven resumes before its first
    refund -- the overdraw that turned PP1's drain wait into the cycle."""
    order, _why = front.interleave_chain_card(NF_TIGHTEST, "tightest-card-first", NF_CARDS, env={})
    assert order[:8] == ["weights_0", "weights_13", "weights_1", "weights_14",
                         "weights_2", "weights_15", "weights_3", "weights_9"]
    # seven resumes (6874 MiB) against 7211 - 700 of room: under the floor
    assert _card0_drawdown_before_first_refund(order) > NF_FREE[0] - NF_FLOOR_MIB


def test_with_the_free_sample_the_chain_card_stays_behind_the_tighter_cards():
    order, why = front.interleave_chain_card(
        NF_TIGHTEST, "tightest-card-first", NF_CARDS, env={}, free_mib=NF_FREE)
    # both 3080s are tighter than the 5090 and no roomier card is left to
    # interleave with: the tightest-card-first order is kept as given
    assert order == NF_TIGHTEST
    assert "NOT interleaved" in why and "[0, 2]" in why
    # three resumes (2946 MiB) before PP1's first pause: inside the room
    assert _card0_drawdown_before_first_refund(order) <= NF_FREE[0] - NF_FLOOR_MIB


def test_a_roomier_card_is_still_interleaved_behind_the_tighter_ones():
    # chain = card 1; card 2 tighter, card 0 roomier than the chain card
    free = {0: 12000, 1: 7804, 2: 5567}
    order, why = front.interleave_chain_card(
        NF_TIGHTEST, "tightest-card-first", NF_CARDS, env={}, free_mib=free)
    assert order[:3] == ["weights_13", "weights_14", "weights_15"]
    assert order[3:11] == ["weights_0", "weights_9", "weights_1", "weights_10",
                           "weights_2", "weights_11", "weights_3", "weights_12"]
    assert order[-2:] == ["weights_draft", "weights"]
    assert sorted(order) == sorted(NF_TIGHTEST)
    assert "behind tighter cards [2]" in why


def test_the_27b_form_is_unchanged_when_the_chain_card_is_the_tightest():
    order, why = front.interleave_chain_card(
        TIGHTEST_27B, "tightest-card-first", CARDS_27B, env={},
        free_mib={0: 4000, 1: 9000, 2: 9000})
    assert order == ["weights_0", "weights_6", "weights_1", "weights_7", "weights_2", "weights_3",
                     "weights_4", "weights_5", "weights_draft", "weights"]
    assert why.endswith("chain card 0 interleaved")


def test_the_front_hands_the_chain_interleave_its_free_sample():
    """Bookkeeping: the fix lives in the keyword; a call site that drops it
    silently restores the x26 order."""
    import inspect
    import re
    src = inspect.getsource(front.Front)
    assert re.search(r"interleave_chain_card\((?:[^()]|\([^()]*\))*free_mib=free_mib", src), (
        "Front.flip calls interleave_chain_card without free_mib=free_mib")
