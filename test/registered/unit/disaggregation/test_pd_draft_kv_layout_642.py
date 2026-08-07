# SPDX-License-Identifier: Apache-2.0
"""A token-sharded PD arm must run ``--draft-kv-layout dcp`` (#642).

The draft KV pool rides the main transfer as extra layers addressed by the
SAME index array as the target pool (``prefill.py:186-195``,
``decode.py:439-448``, "The indices are always shared with a target model"),
and ``mooncake/conn.py`` contains no occurrence of "draft" at all, so the
transport could not treat them differently even if it wanted to.

On a DCP decode arm those indices are compact owner-rule rows. Whether that is
the RIGHT address for the draft pool depends on ``--draft-kv-layout``:

* ``dcp`` -> draft pool takes ``dcp_compact_pool_rows``, the target's own
  coordinate system. Correct by construction.
* ``replicated`` (default) -> draft pool keeps "the full global context per
  rank" (``model_runner_kv_cache_mixin.py:2819-2823``). Compact rows are then
  addresses in a different coordinate system, and every draft row lands where
  another token lives.

``HazardTest`` below is the point of this file. A guard whose test only checks
that the guard raises proves nothing about the hazard it claims to prevent, so
the corruption is DEMONSTRATED first, from the two code sites' own arithmetic,
independently of the guard. ``GuardTest`` then shows the configuration cannot
be reached.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.arg_groups.pd_disaggregation_hook import validate_pd_draft_kv_layout
from sglang.srt.disaggregation.draft_kv_canonical import (
    CANONICAL_LAYOUT_VERSION,
    DraftKvCanonicalLayout,
    DraftKvLayoutMismatch,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _compact(global_slot: int, cp_S: int, cp_lo: int, cp_hi: int) -> int:
    """The rewrite decode.py:1119-1125 applies, stated once here.

    ``(L // S) * (hi - lo) + (L % S - lo)`` -- the same expression as
    ``dcp_weighted_write_slots``.
    """
    return (global_slot // cp_S) * (cp_hi - cp_lo) + (global_slot % cp_S - cp_lo)


def _owns(global_slot: int, cp_S: int, cp_lo: int, cp_hi: int) -> bool:
    return cp_lo <= global_slot % cp_S < cp_hi


class HazardTest(CustomTestCase):
    """The corruption, demonstrated rather than asserted.

    Models only the ADDRESSING -- which is where the defect is. Rank 0 of a
    3-way token split (S=3, [lo,hi) = [0,1)) over a 12-token global context.
    """

    CP_S, CP_LO, CP_HI, CONTEXT = 3, 0, 1, 12

    def _owned_slots(self):
        return [
            L
            for L in range(self.CONTEXT)
            if _owns(L, self.CP_S, self.CP_LO, self.CP_HI)
        ]

    def test_replicated_pool_receives_other_tokens_rows(self):
        """The default layout: a full-context pool written at compact rows.

        The pool has one row per GLOBAL slot, so row i belongs to token i.
        The transfer writes token L at row compact(L). Any L where those
        differ silently overwrites another token's row.
        """
        owned = self._owned_slots()
        self.assertEqual(owned, [0, 3, 6, 9])

        # 'replicated': full global context per rank (mixin :2821-2823).
        pool = [None] * self.CONTEXT
        for L in owned:
            pool[_compact(L, self.CP_S, self.CP_LO, self.CP_HI)] = f"token{L}"

        # Row i of a full-context pool is token i. Read it back that way.
        self.assertEqual(pool[0], "token0", "slot 0 happens to be a fixed point")
        self.assertEqual(
            pool[1],
            "token3",
            "token 3's draft KV was written into token 1's row -- this is the "
            "corruption, and nothing raises",
        )
        self.assertEqual(pool[2], "token6")
        self.assertEqual(pool[3], "token9")
        # And the tokens that really own rows 6 and 9 got nothing at all.
        self.assertIsNone(pool[6])
        self.assertIsNone(pool[9])

        misplaced = [L for L in owned if _compact(L, self.CP_S, self.CP_LO, self.CP_HI) != L]
        self.assertEqual(
            misplaced, [3, 6, 9], "3 of 4 owned tokens land at a wrong address"
        )

    def test_dcp_pool_is_addressed_correctly_by_the_same_indices(self):
        """The 'dcp' layout: compact rows, so the shared indices are right.

        Same write, a pool sized by dcp_compact_pool_rows instead. Every owned
        token is readable at its own compact row and nothing is overwritten.
        """
        owned = self._owned_slots()
        compact_rows = (self.CONTEXT // self.CP_S + 1) * (self.CP_HI - self.CP_LO)
        pool = [None] * compact_rows

        for L in owned:
            pool[_compact(L, self.CP_S, self.CP_LO, self.CP_HI)] = f"token{L}"

        for L in owned:
            self.assertEqual(
                pool[_compact(L, self.CP_S, self.CP_LO, self.CP_HI)], f"token{L}"
            )
        self.assertEqual(len([x for x in pool if x is not None]), len(owned))

    def test_the_two_pools_are_not_even_the_same_size(self):
        """Size alone shows they are different coordinate systems."""
        compact_rows = (self.CONTEXT // self.CP_S + 1) * (self.CP_HI - self.CP_LO)
        self.assertNotEqual(compact_rows, self.CONTEXT)
        self.assertLess(compact_rows, self.CONTEXT)


def _args(**over):
    base = dict(
        disaggregation_mode="decode",
        speculative_algorithm="NEXTN",
        dcp_size=3,
        tp_size=3,
        draft_kv_layout="replicated",
    )
    base.update(over)
    return SimpleNamespace(**base)


class GuardTest(CustomTestCase):
    """L11: the hazardous configuration cannot be reached."""

    def test_dcp_arm_on_replicated_draft_layout_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            validate_pd_draft_kv_layout(_args())
        msg = str(ctx.exception)
        self.assertIn("--draft-kv-layout", msg)
        self.assertIn("decode", msg, "refusal must name the arm")
        self.assertIn("different token", msg, "refusal must name the hazard")

    def test_prefill_arm_is_refused_too(self):
        with self.assertRaises(ValueError):
            validate_pd_draft_kv_layout(_args(disaggregation_mode="prefill"))

    def test_dcp_layout_is_accepted(self):
        validate_pd_draft_kv_layout(_args(draft_kv_layout="dcp"))

    def test_inert_without_speculation(self):
        """No draft worker, no draft pool, no registration, no hazard."""
        validate_pd_draft_kv_layout(_args(speculative_algorithm=None))

    def test_inert_off_the_token_sharded_layout(self):
        for over in ({"dcp_size": 1}, {"dcp_size": 2, "tp_size": 3}):
            with self.subTest(over=over):
                validate_pd_draft_kv_layout(_args(**over))

    def test_inert_for_a_monolithic_server(self):
        """The standing production boot is monolithic NEXTN on 'replicated'."""
        validate_pd_draft_kv_layout(_args(disaggregation_mode="null"))


class LayoutAgreementTest(CustomTestCase):
    """L10: the two arms must agree on draft_kv_layout."""

    def _layout(self, **over):
        base = dict(
            version=CANONICAL_LAYOUT_VERSION,
            num_kv_heads=4,
            head_dim=256,
            element_size=1,
            num_draft_layers=1,
            draft_kv_layout="dcp",
        )
        base.update(over)
        return DraftKvCanonicalLayout(**base)

    def test_matching_layouts_pair(self):
        self._layout().assert_compatible(self._layout(), peer="decode")

    def test_disagreeing_arms_are_refused(self):
        with self.assertRaises(DraftKvLayoutMismatch) as ctx:
            self._layout().assert_compatible(
                self._layout(draft_kv_layout="replicated"), peer="decode-arm"
            )
        self.assertIn("draft_kv_layout", str(ctx.exception))

    def test_guard_one_deliberately_cannot_cover_this(self):
        """Pin WHY this lives here and not in the identity hash.

        compute_model_identity_hash is taken with
        include_parallel_vectors=False so a PP prefill and a TP+DCP decode on
        identical weights still pair (#631a guard 1). draft_kv_layout is a
        parallelism decision, so adding it there would reintroduce exactly the
        false refusal that flag prevents.
        """
        import inspect

        from sglang.srt.mem_cache.hicache_storage import compute_model_identity_hash

        self.assertNotIn(
            "draft_kv_layout",
            inspect.getsource(compute_model_identity_hash),
            "draft_kv_layout must NOT enter the weights-identity hash",
        )


if __name__ == "__main__":
    unittest.main()
