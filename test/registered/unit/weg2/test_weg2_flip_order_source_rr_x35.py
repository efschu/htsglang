"""23.09. (fnFL2x35): the P->D pause order round-robins over the SOURCE cards.

x35 (draft collect fixed) stalled in P->D on the 5090: the order was
tightest-card-first over the source (P) cards [13,14,15,draft, 9..12, 0..8]
with driver_free {0: 7385, 1: 8170, 2: 5741}. The destination D is TP, so
EVERY resume costs EVERY card (~800 MiB per chunk shard on the 5090, 3984
MiB for the draft) while a card is refunded only by the pauses of its OWN
bands -- and the 5090's first own band was ninth. D TP0 ran dry at
weights_0 (need 822, balance 260); P PP0's next deposit sat in the depth-1
diagonal drain behind exactly that resume: the P<->D credit cycle, 70 s
FLIP STALL, W35, W17.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front  # noqa: E402

NF_CARDS = {**{f"weights_{i}": (1,) for i in range(9)},
            **{f"weights_{i}": (0,) for i in range(9, 13)},
            **{f"weights_{i}": (2,) for i in range(13, 16)},
            "weights_draft": (2,)}
NF_TAGS = [f"weights_{i}" for i in range(16)] + ["weights_draft", "weights"]
X35_FREE = {0: 7385, 1: 8170, 2: 5741}
X35_ORDER = ["weights_13", "weights_14", "weights_15", "weights_draft",
             "weights_9", "weights_10", "weights_11", "weights_12",
             "weights_0", "weights_1", "weights_2", "weights_3", "weights_4",
             "weights_5", "weights_6", "weights_7", "weights_8", "weights"]


def _foreign_before_first_and_between_own(order, card):
    """(foreign resumes before this card's first refund, max foreign resumes
    between two of its refunds) -- what the card PAYS on a TP destination."""
    own = [i for i, t in enumerate(order) if t != "weights" and NF_CARDS[t][0] == card]
    before = own[0]
    gaps = [b - a - 1 for a, b in zip(own, own[1:])]
    return before, (max(gaps) if gaps else 0)


def test_x35s_order_made_the_5090_pay_eight_resumes_and_the_draft_first():
    """The measured mutant: the roomiest card's bands came last."""
    before, _ = _foreign_before_first_and_between_own(X35_ORDER, 1)
    assert before == 8 and X35_ORDER.index("weights_draft") < X35_ORDER.index("weights_0")


def test_the_uniform_destination_gets_one_band_per_source_card_per_round():
    order, why = front.interleave_pause_order(NF_TAGS, NF_CARDS, X35_FREE)
    assert order == ["weights_13", "weights_9", "weights_0",
                     "weights_14", "weights_10", "weights_1",
                     "weights_15", "weights_11", "weights_2",
                     "weights_draft", "weights_12", "weights_3",
                     "weights_4", "weights_5", "weights_6", "weights_7", "weights_8",
                     "weights"]
    assert why.startswith("tightest-card-first") and "SOURCE cards [2, 0, 1]" in why
    # every card's exposure is bounded by construction: at most its rank in
    # the tightness order before the first refund, at most (cards - 1) between
    for card, rank in ((2, 0), (0, 1), (1, 2)):
        before, between = _foreign_before_first_and_between_own(order, card)
        assert before == rank, (card, before)
        assert between <= 2, (card, between)


def test_the_chain_interleave_does_not_undo_the_round_robin():
    rr, why = front.interleave_pause_order(NF_TAGS, NF_CARDS, X35_FREE)
    order, why2 = front.interleave_chain_card(rr, why, NF_CARDS, env={}, free_mib=X35_FREE)
    assert order == rr
    assert why2.endswith("chain card: subsumed by the source round-robin")


def test_a_pp_destination_keeps_the_destination_round_robin():
    """D->P (dst_cards given: every tag lands on ONE card) is the 15-s leg that
    works; its order is untouched by the source round-robin."""
    dst = {t: (int(c[0]),) for t, c in NF_CARDS.items()}
    order, why = front.interleave_pause_order(NF_TAGS, {}, X35_FREE, dst_cards=dst)
    assert "SOURCE cards" not in why and "round-robin over destination cards" in why
    assert order[-1] == "weights" and sorted(order) == sorted(NF_TAGS)


def test_one_source_card_is_identity():
    cards = {t: (1,) for t in NF_CARDS}
    order, why = front.interleave_pause_order(NF_TAGS, cards, X35_FREE)
    assert order == [t for t in NF_TAGS if t != "weights"] + ["weights"]
    assert why == "tightest-card-first"
