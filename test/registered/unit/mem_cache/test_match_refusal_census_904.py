"""#904 (g)/(h): the read-side discriminator for a zero prefix hit.

WHAT IT SEPARATES. ``#cached-token: 0`` is one number covering three worlds:
nothing was ever stored (NOT_PRESENT), the node survived but its bytes did
not (DEAD), or the bytes were reachable and a component validator declined
them (REFUSED). Only the last two are the load-then-invalidate half of #904.

WHY IT IS NEEDED EVEN THOUGH #869b IS ANSWERED.
``/spinning/gpu-arb/ANALYSE_869b_pp_tier_zero_hits.md`` closed W40/W41
correctly on the WRITE side -- 1959 of 1959 flip-writeback fences carried
``staged=0 acked=0``, so the store was empty and the read path was never
reached. It names its own residue in as many words: the read path was
INFERRED from an empty store, not shown. On a boot where the store is not
empty that inference is unavailable, and #873's "``cached_tokens == 0``
means recomputed" has no evidence either way. This is the instrument that
makes the next such boot decidable instead of arguable.

The desk half is here. The boot half is a window ticket: arm
``SGLANG_MATCH_REFUSAL_CENSUS_EVERY`` on a run whose store is demonstrably
non-empty (``staged>0 acked>0`` on the fence line) and read the verdict
field.
"""

import unittest

from sglang.srt.mem_cache.match_refusal_census import (
    MatchOutcome,
    MatchRefusalCensus,
    classify,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestTheThreeWorlds(CustomTestCase):
    def test_untouched_census_says_it_did_not_measure(self):
        """INDIKATOR-GESETZ. An instrument that cannot say 'I did not
        measure' is indistinguishable from one that measured nothing, and a
        zero from the wrong cause is not distinguishable from green."""
        self.assertEqual(MatchRefusalCensus().verdict(), MatchOutcome.NO_OBSERVATION)
        self.assertEqual(classify(None), MatchOutcome.NO_OBSERVATION)
        self.assertFalse(MatchRefusalCensus().is_resident_but_unusable())

    def test_walk_that_found_nothing_is_not_present(self):
        c = MatchRefusalCensus()
        c.note_reached(0)
        self.assertEqual(c.verdict(), MatchOutcome.NOT_PRESENT)
        self.assertFalse(c.is_resident_but_unusable())

    def test_node_present_but_bytes_gone_is_dead(self):
        c = MatchRefusalCensus()
        c.note_dead_stop(64)
        self.assertEqual(c.verdict(), MatchOutcome.DEAD)
        self.assertTrue(c.is_resident_but_unusable())

    def test_reachable_bytes_declined_is_refused(self):
        c = MatchRefusalCensus()
        c.note_reached(128)
        c.note_refused("MambaComponent", 128)
        self.assertEqual(c.verdict(), MatchOutcome.REFUSED)
        self.assertTrue(c.is_resident_but_unusable())
        self.assertEqual(c.top_refusers(), (("MambaComponent", 128),))

    def test_any_accepted_token_is_a_hit(self):
        c = MatchRefusalCensus()
        c.note_reached(64)
        c.note_accepted(64)
        c.note_reached(64)
        c.note_refused("FullComponent", 64)
        self.assertEqual(c.verdict(), MatchOutcome.HIT)
        self.assertFalse(c.is_resident_but_unusable())


class TestThePartitionIsExhaustive(CustomTestCase):
    """#873's defect, one level down: a narrowed candidate set reads as a
    decomposition. The parts must sum or the object refuses to speak."""

    def test_parts_sum_to_reached(self):
        c = MatchRefusalCensus()
        c.note_reached(100)
        c.note_accepted(40)
        c.note_refused("FullComponent", 60)
        c.check_partition()
        self.assertEqual(c.refused_tokens, 60)

    def test_double_attributed_refusal_does_not_break_the_partition(self):
        """Two components may decline the SAME node. The blame dict then
        exceeds the partition term, which is why the term is DERIVED from
        reached-minus-accepted and never summed from the dict."""
        c = MatchRefusalCensus()
        c.note_reached(50)
        c.note_refused("MambaComponent", 50)
        c.note_refused("FullComponent", 50)
        c.check_partition()
        self.assertEqual(c.refused_tokens, 50)
        self.assertEqual(sum(c.refused_tokens_by_component.values()), 100)

    def test_a_miscounting_census_raises_instead_of_reporting(self):
        c = MatchRefusalCensus()
        c.note_reached(10)
        c.accepted_tokens = 99
        with self.assertRaises(ValueError):
            c.log_fields()

    def test_log_line_carries_every_field_needed_to_classify(self):
        c = MatchRefusalCensus()
        c.note_reached(32)
        c.note_refused("MambaComponent", 32)
        line = c.format_line()
        for token in ("verdict=refused", "reached=32", "accepted=0", "refused=32"):
            self.assertIn(token, line)


class TestTheGateIsOffByDefault(CustomTestCase):
    def test_disarmed_builds_no_census(self):
        from sglang.srt.mem_cache.match_refusal_census import (
            census_every,
            new_match_census,
        )

        self.assertEqual(census_every(), 0)
        self.assertIsNone(new_match_census())

    def test_armed_builds_one(self):
        import os

        from sglang.srt.mem_cache import match_refusal_census as m

        os.environ["SGLANG_MATCH_REFUSAL_CENSUS_EVERY"] = "5"
        try:
            self.assertEqual(m.census_every(), 5)
            self.assertIsNotNone(m.new_match_census())
        finally:
            os.environ.pop("SGLANG_MATCH_REFUSAL_CENSUS_EVERY", None)

    def test_a_refusal_is_never_sampled_away(self):
        import logging
        import os

        from sglang.srt.mem_cache import match_refusal_census as m

        class _Sink(logging.Logger):
            def __init__(self):
                super().__init__("sink")
                self.lines = []

            def info(self, fmt, *args):
                self.lines.append(fmt % args if args else fmt)

        os.environ["SGLANG_MATCH_REFUSAL_CENSUS_EVERY"] = "1000"
        try:
            sink = _Sink()
            c = m.MatchRefusalCensus()
            c.note_reached(8)
            c.note_refused("MambaComponent", 8)
            m.emit(c, sink)
            self.assertEqual(len(sink.lines), 1, "a refusal must always be emitted")

            sink2 = _Sink()
            hit = m.MatchRefusalCensus()
            hit.note_reached(8)
            hit.note_accepted(8)
            m.emit(hit, sink2)
            self.assertEqual(len(sink2.lines), 0, "hits are sampled, not streamed")
        finally:
            os.environ.pop("SGLANG_MATCH_REFUSAL_CENSUS_EVERY", None)


class TestTheWalkIsWired(CustomTestCase):
    """PRESENT-BUT-UNWIRED is the expensive middle state. The census must be
    fed from the LIVE match walk (``UnifiedRadixCache._match_prefix_helper``,
    the path registry.py routes hybrid SSM + hicache to), not merely exist."""

    def test_the_live_match_helper_feeds_it(self):
        import inspect

        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache._match_prefix_helper)
        self.assertIn("new_match_census()", src)
        self.assertIn("note_dead_stop", src)
        self.assertIn("note_refused", src)
        self.assertIn("census_emit", src)

    def test_refusals_are_attributed_per_component(self):
        """Blame without a name is not a discriminator: 'something refused'
        does not tell a reader whether the mamba half or the full half went
        missing, which is the first branch of any follow-up."""
        import inspect

        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        src = inspect.getsource(UnifiedRadixCache._match_prefix_helper)
        self.assertIn("component_names", src)


if __name__ == "__main__":
    unittest.main()
