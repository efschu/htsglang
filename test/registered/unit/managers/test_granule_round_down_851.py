"""#851 F3: a pressed post must deliver the granules that FIT, not zero.

W22 (boot_w22_0824_0656.log line 4864, 07:04:52, PP0 -- the rank UNDER
pressure, not the veto rank) printed::

    kv-slack 0 MiB (ask of 2776 MiB rounds up to one granule of 256 MiB,
    which exceeds the 2776 MiB this post holds)

on the same line as ``slack=88837 rows`` and "KV capacity is the funder". The
post held 2776 MiB of priced slack and the PRICING refused all of it.

THE CLOSED FORM, ``Post.creditable`` (funding_authority.py:260-274)::

    draw    = min(want, have)                  # 2776 MiB, have 2776 MiB
    rounded = ceil(draw / gran) * gran         # 2816 MiB  (11 granules)
    if rounded > have: return 0                # refuses ENTIRELY

Law 3 is right that a SUB-granule draw delivers zero -- a partial granule is
not deliverable. It does not follow that a draw of ten-and-a-bit granules
delivers zero. The correct answer when the rounded ask overshoots holdings is
the largest whole number of granules that fits, ``(have // gran) * gran`` =
2560 MiB. The bug triggers exactly in the pressed case: ask >= holdings, and
holdings not granule-aligned -- i.e. whenever the funder is needed most.

WHY THIS MUST LAND BEFORE F4 (the plan's ordering, restated so a later reader
cannot reorder it by accident): F4 wires ``can_fund`` into the corridor
refusal, which makes this post VISIBLE to the gate for the first time. Wiring
it up first would only change the refusal's text from "[nothing]" to a new
false reason -- a funder that reports zero while holding 2776 MiB.

CLASSIFICATION TODAY: an INDICATOR defect. ``can_fund`` feeds the census and
the refusal line, not the actuator (FEATURE_CATALOG: "not yet for the refusal
itself"). It becomes an ACTUATOR defect the moment F4 lands.

PRIOR-ART DISTINCTION, mandatory. This is NOT the retracted "granule axis"
(Strang 12, 2026-08-22, kapazitaets-hebel-inventar): that claim was
*granule > pool*, built on a test-fixture number (229376 rows), and was
correctly retracted. This is a different mechanism -- a round-UP refusal
inside ``Post.creditable`` with the real 256 MiB granule -- verified against
the shipped code and against the live log line it printed.

Fills the gap at ``test/srt/test_funding_authority_770.py:274-296``
(TestGranularityForm), which covers sub-granule asks only.
"""

import unittest

from sglang.srt.managers.funding_authority import MIB, FundingAuthority, Post

#: The shipped KV granule on this rig: 8192 rows x 32 KiB = 256 MiB.
KV_GRANULE_BYTES = 256 * MIB

#: W22's numbers, verbatim from log line 4864.
W22_HOLDS = 2776 * MIB
W22_ASK = 2776 * MIB
#: (2776 // 256) * 256 -- ten whole granules.
W22_FITS = 2560 * MIB


class TestThePressedPostDeliversWhatFits(unittest.TestCase):
    def test_the_w22_specimen_credits_ten_granules_not_zero(self):
        """RED before F3. The whole point of the fix."""
        post = Post("kv-slack", W22_HOLDS, granule_bytes=KV_GRANULE_BYTES)
        drawn, reason = post.creditable(W22_ASK)
        self.assertEqual(drawn, W22_FITS)
        self.assertEqual(reason, "")

    def test_the_credit_never_exceeds_holdings(self):
        """The direction that must NOT break: a post may not over-promise.

        Rounding DOWN is safe precisely because it cannot exceed what the post
        holds; this pins that, so a later "just round up and let the caller
        sort it out" cannot pass.
        """
        for holds_mib, ask_mib in ((2776, 2776), (2776, 99999), (512, 300), (256, 256)):
            post = Post("kv-slack", holds_mib * MIB, granule_bytes=KV_GRANULE_BYTES)
            drawn, _ = post.creditable(ask_mib * MIB)
            self.assertLessEqual(drawn, holds_mib * MIB, f"{holds_mib}/{ask_mib}")

    def test_the_credit_is_always_a_whole_number_of_granules(self):
        """Law 3 survives the fix: a partial granule is still not deliverable."""
        for holds_mib, ask_mib in ((2776, 2776), (2776, 1000), (900, 900)):
            post = Post("kv-slack", holds_mib * MIB, granule_bytes=KV_GRANULE_BYTES)
            drawn, _ = post.creditable(ask_mib * MIB)
            self.assertEqual(drawn % KV_GRANULE_BYTES, 0, f"{holds_mib}/{ask_mib}")


class TestTheNamedRefusalSurvives(unittest.TestCase):
    """CAN-FAIL TWINS. F3 must not turn the real refusals into silent zeros."""

    def test_holdings_below_one_granule_are_still_a_NAMED_refusal(self):
        post = Post("kv-slack", 100 * MIB, granule_bytes=KV_GRANULE_BYTES)
        drawn, reason = post.creditable(50 * MIB)
        self.assertEqual(drawn, 0)
        self.assertIn("granule", reason)

    def test_a_sub_granule_ask_still_rounds_UP_when_affordable(self):
        """The existing law-3 behaviour, unchanged by the round-down branch.

        This is `test_funding_authority_770`'s first case, restated here
        because F3 edits the exact lines it depends on.
        """
        post = Post("kv-slack", 2776 * MIB, granule_bytes=KV_GRANULE_BYTES)
        drawn, reason = post.creditable(10 * MIB)
        self.assertEqual(reason, "")
        self.assertEqual(drawn, KV_GRANULE_BYTES)

    def test_an_unavailable_post_still_names_its_reason(self):
        post = Post(
            "kv-slack",
            0,
            granule_bytes=KV_GRANULE_BYTES,
            unavailable_reason="pool is at or below its rung floor",
        )
        drawn, reason = post.creditable(W22_ASK)
        self.assertEqual(drawn, 0)
        self.assertIn("rung floor", reason)

    def test_a_post_without_a_granule_is_untouched(self):
        post = Post("allocator-cache", 1234 * MIB)
        drawn, reason = post.creditable(999 * MIB)
        self.assertEqual(drawn, 999 * MIB)
        self.assertEqual(reason, "")


class TestTheAuthorityReportsTheFundedFigure(unittest.TestCase):
    """The census/refusal line must carry the new number, not just the post."""

    def test_can_fund_sees_the_granules_that_fit(self):
        auth = FundingAuthority()
        auth.declare_post(
            Post("kv-slack", W22_HOLDS, granule_bytes=KV_GRANULE_BYTES)
        )
        verdict = auth.can_fund(W22_ASK)
        self.assertEqual(verdict.covered_bytes, W22_FITS)


if __name__ == "__main__":
    unittest.main()
