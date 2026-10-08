"""y6l: the #580 group vote prices a read with the same minimum as the local
gate.

The local gate honours ``min_tokens`` (``_min_len = max(1, min_tokens)``,
xsn437), but the participation vote declined on ``group_len <
self.prefetch_threshold``. At D TP=3 every hand-back read below 256 tokens
(1160d65e1d hands back with min_tokens=1) therefore ended as ``#915 PREFETCH
REFUSED reason=vote_negative`` although every rank voted present -- the
min_tokens=1 hand-off had no effect. 27B saw the same on its metal
(need=24/40).

Driven through the REAL ``UnifiedRadixCache.prefetch_from_storage`` on three
simulated ranks (the H99 harness). RED on e411286a79: group_len 24,
min_tokens=1, all present -- the vote declines on every rank.
"""

from __future__ import annotations

import types
import unittest

import test_nf_form_a_prefetch_span_h99 as h99

from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

RID = "pdflip-hb-24"
BASE = 4096
NEED = 24


def _vote(*, min_tokens, need=NEED, span_base=False):
    group = h99.MockGlooGroup()
    caches = {}
    prompt = list(range(BASE + need))

    def _rank(r):
        caches[r] = c = h99._carrier(r, group)
        c._hp1_note_end_vote = types.MethodType(UnifiedRadixCache._hp1_note_end_vote, c)
        kw = {"min_tokens": min_tokens}
        if span_base:
            kw["span_base"] = BASE
        c.prefetch_from_storage(
            RID, h99._host_node(), prompt[BASE:], last_hash=None, prefix_keys=None, **kw
        )
        return h99._registered_len(c, RID)

    with h99._Env():
        results, errors = h99.run_ranks(_rank)
    return results, errors, group


class TestHandbackReadPassesTheVote(unittest.TestCase):
    def test_group_len_24_min_tokens_1_all_present_is_registered(self):
        results, errors, group = _vote(min_tokens=1)
        self.assertEqual(errors, {}, errors)
        self.assertEqual(group.errors, [])
        self.assertEqual(
            results, {r: (NEED, NEED) for r in range(3)},
            "all three ranks present, group_len=24, min_tokens=1: the vote "
            "declined (vote_negative) -- it ignores the read's own minimum",
        )

    def test_form_a_end_vote_same_read_is_registered(self):
        results, errors, _g = _vote(min_tokens=1, span_base=True)
        self.assertEqual(errors, {}, errors)
        self.assertEqual(results, {r: (NEED, NEED) for r in range(3)})


class TestDefaultUnchanged(unittest.TestCase):
    def test_without_min_tokens_the_threshold_still_declines(self):
        results, errors, _g = _vote(min_tokens=None)
        self.assertEqual(errors, {}, errors)
        self.assertEqual(set(results.values()), {None}, results)

    def test_min_tokens_above_group_len_declines_uniformly(self):
        results, errors, _g = _vote(min_tokens=40)
        self.assertEqual(errors, {}, errors)
        self.assertEqual(set(results.values()), {None}, results)


if __name__ == "__main__":
    unittest.main()
