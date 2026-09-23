"""23.09. (fnFL2x30): the draft tag is paused as a band of its card.

x30 (chain fix c360844661 in place) stalled on card 2: P PP2's draft tag
(2.9 GB, the MTP head on the last stage) went over lane p4 as a BAR1 ring
(4 x 32 MiB), so its deposit completes only as the collector (D TP0)
consumes it -- and the collector reaches the draft LAST in the pause
order. D TP2 needed the draft's 2770 MiB refund on card 2 at weights_4
(need 994, balance 356) and waited 120 s (W35), PP2 sat in the ring's recv,
W17. In x22 the same tag went over a SEQ host buffer, completed in 1.9 s
and was credited before weights_4 -- the order was never the problem
until the transport stopped buffering it.
"""
from __future__ import annotations

import os
import re

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import front  # noqa: E402

NF_CARDS = {**{f"weights_{i}": (1,) for i in range(9)},
            **{f"weights_{i}": (0,) for i in range(9, 13)},
            **{f"weights_{i}": (2,) for i in range(13, 16)}}
NF_CARDS_WITH_DRAFT = {**NF_CARDS, "weights_draft": (2,)}
NF_TAGS = [f"weights_{i}" for i in range(16)] + ["weights_draft", "weights"]
NF_FREE = {0: 7211, 1: 7804, 2: 5567}


def test_the_draft_follows_its_cards_bands_and_the_base_tag_still_closes():
    order, why = front.interleave_pause_order(NF_TAGS, NF_CARDS_WITH_DRAFT, NF_FREE)
    assert order[:4] == ["weights_13", "weights_14", "weights_15", "weights_draft"]
    assert order[-1] == "weights"
    assert sorted(order) == sorted(NF_TAGS)
    assert why == "tightest-card-first"


def test_without_a_card_the_draft_keeps_its_old_place_before_the_base_tag():
    """The 27B arm publishes no card for the draft: nothing moves there."""
    order, _ = front.interleave_pause_order(NF_TAGS, NF_CARDS, NF_FREE)
    assert order[-2:] == ["weights_draft", "weights"]
    assert order[:3] == ["weights_13", "weights_14", "weights_15"]


def test_the_chain_interleave_keeps_the_draft_with_its_card():
    tight, _ = front.interleave_pause_order(NF_TAGS, NF_CARDS_WITH_DRAFT, NF_FREE)
    order, why = front.interleave_chain_card(
        tight, "tightest-card-first", NF_CARDS_WITH_DRAFT, env={}, free_mib=NF_FREE)
    assert order == tight
    assert "NOT interleaved" in why


def test_the_draft_is_ahead_of_the_roomier_cards_when_the_chain_is_interleaved():
    free = {0: 12000, 1: 7804, 2: 5567}
    tight, _ = front.interleave_pause_order(NF_TAGS, NF_CARDS_WITH_DRAFT, free)
    order, why = front.interleave_chain_card(
        tight, "tightest-card-first", NF_CARDS_WITH_DRAFT, env={}, free_mib=free)
    assert order[:4] == ["weights_13", "weights_14", "weights_15", "weights_draft"]
    assert order[-1] == "weights"
    assert "behind tighter cards [2]" in why


def test_the_launcher_publishes_the_drafts_card_next_to_the_chunk_map():
    """Bookkeeping: the front can only order what the map names. A map
    without the draft restores the x30 order silently."""
    import inspect

    from sglang.srt.weg2 import launcher

    src = inspect.getsource(launcher.main)
    assert re.search(
        r'src_chunk_cards\["P"\]\["weights_draft"\]\s*=\s*\[int\(cards\[-1\]\.nvml_index\)\]',
        src), "launcher.main no longer maps weights_draft to the last stage's card"
