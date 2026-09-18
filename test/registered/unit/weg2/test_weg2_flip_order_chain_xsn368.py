"""18.09. (xsn367/368): the pause order interleaves the chain card's tags
(the source card with the most bands, PP0) with the other cards' bands, so
a destination that collects two tags at once never idles PP0's chain."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front  # noqa: E402

CARDS = {"weights_0": (0,), "weights_1": (0,), "weights_2": (0,), "weights_3": (0,),
         "weights_4": (0,), "weights_5": (0, 1), "weights_6": (1, 2), "weights_7": (2,)}
TIGHTEST = ["weights_6", "weights_7", "weights_0", "weights_1", "weights_2", "weights_3",
            "weights_4", "weights_5", "weights_draft", "weights"]


def test_chain_card_tags_alternate_with_the_others_and_the_base_tag_closes():
    order, why = front.interleave_chain_card(TIGHTEST, "tightest-card-first", CARDS, env={})
    assert order == ["weights_0", "weights_6", "weights_1", "weights_7", "weights_2", "weights_3",
                     "weights_4", "weights_5", "weights_draft", "weights"]
    assert "chain card 0" in why
    assert sorted(order) == sorted(TIGHTEST)


def test_identity_when_disabled_one_card_or_no_map():
    assert front.interleave_chain_card(TIGHTEST, "w", CARDS, env={"SGLANG_WEG2_FLIP_ORDER_CHAIN": "0"}) == (TIGHTEST, "w")
    assert front.interleave_chain_card(TIGHTEST, "w", {}, env={}) == (TIGHTEST, "w")
    one = {t: (1,) for t in CARDS}
    assert front.interleave_chain_card(TIGHTEST, "w", one, env={}) == (TIGHTEST, "w")
