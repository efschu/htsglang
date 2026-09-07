# SPDX-License-Identifier: Apache-2.0
"""Weg 2 draft KV across the flip (#1233, spec T14): the D-side claim.

The presence probe answers KV pages ``k`` and draft pages ``d``. The claim
is ``d`` (trim, at most one HiCache chunk of re-prefill) when ``k - d`` fits
one chunk, else ``k`` with the request marked draft-cold BY NAME over
``[d, k)`` -- the #993 zero fill is the deterministic fill behind the name,
never silent. ``_draft_page_get`` stops discarding the hit count, and a rank
whose claim differs from the group's is a STOP, never a compensation.
"""

import types
import unittest

import pytest

from sglang.srt.managers.cache_controller import (
    HiCacheController,
    Weg2DraftDisagree,
    assert_draft_claims_agree,
    resolve_draft_claim,
)
from sglang.srt.managers.phase_flip_draft_bootstrap import draft_cold_reason
from sglang.test.test_utils import CustomTestCase

CHUNK = 4096


class Stub:
    def __init__(self, **kw):
        self.draft_page_get_func = None
        self._draft_l3_hits = 0
        self._draft_l3_misses = 0
        self._draft_l3_logged_at = -1
        self._draft_get_warned_at = 0
        self.__dict__.update(kw)

    _draft_page_get = HiCacheController._draft_page_get
    _draft_page_get_flags = HiCacheController._draft_page_get_flags
    _log_draft_l3_progress = HiCacheController._log_draft_l3_progress


class TestClaimPolicy(CustomTestCase):
    def test_t14_draft_page_get_returns_the_hit_count(self):
        self.assertEqual(Stub()._draft_page_get(["a", "b"], None), -1)
        s = Stub(draft_page_get_func=lambda h, i: [True, False, True])
        self.assertEqual(s._draft_page_get(["a", "b", "c"], None), 2)

        def boom(h, i):
            raise OSError("store gone")

        self.assertEqual(Stub(draft_page_get_func=boom)._draft_page_get(["a"], None), 0)

    def test_t14_claim_full_trim_cold(self):
        no_reprobe = lambda d: d  # noqa: E731 - the anchor sits at d
        self.assertEqual(resolve_draft_claim(10, 10, CHUNK, no_reprobe), (10, 10, "full", None))
        self.assertEqual(resolve_draft_claim(10, 12, CHUNK, no_reprobe), (10, 10, "full", None))
        self.assertEqual(resolve_draft_claim(5000, 4000, CHUNK, no_reprobe), (4000, 4000, "trim", None))
        self.assertEqual(
            resolve_draft_claim(9000, 4000, CHUNK, no_reprobe), (9000, 4000, "cold", (4000, 9000))
        )
        # the trim lands on the nearest anchor at or below d; if that pushes the
        # re-prefill past one chunk the request goes cold by name instead
        self.assertEqual(resolve_draft_claim(5000, 4000, CHUNK, lambda d: 3500), (3500, 3500, "trim", None))
        self.assertEqual(
            resolve_draft_claim(5000, 4000, CHUNK, lambda d: 800), (5000, 4000, "cold", (4000, 5000))
        )
        self.assertEqual(resolve_draft_claim(10, 0, CHUNK, no_reprobe), (0, 0, "trim", None))
        self.assertEqual(resolve_draft_claim(0, 0, CHUNK, no_reprobe), (0, 0, "full", None))

    def test_t14_cold_reason_third_trigger(self):
        controller = types.SimpleNamespace(draft_cold_spans={"r1": (4000, 9000)})
        sched = types.SimpleNamespace(
            tree_cache=types.SimpleNamespace(cache_controller=controller)
        )
        req = types.SimpleNamespace(rid="r1", prefix_indices=list(range(9000)))
        reason = draft_cold_reason(sched, req, True)
        self.assertIsNotNone(reason)
        self.assertIn("5000 of 9000", reason)
        self.assertIn("[4000, 9000)", reason)
        self.assertNotIn("r1", controller.draft_cold_spans)  # consumed once
        warm = types.SimpleNamespace(rid="r2", prefix_indices=list(range(9000)))
        self.assertIsNone(draft_cold_reason(sched, warm, True))

    def test_t14_rank_disagreement_is_a_stop(self):
        assert_draft_claims_agree(7, 7, "rid")
        with pytest.raises(Weg2DraftDisagree, match="WEG2 DRAFT-DISAGREE STOP"):
            assert_draft_claims_agree(7, 9, "rid")


if __name__ == "__main__":
    unittest.main()
