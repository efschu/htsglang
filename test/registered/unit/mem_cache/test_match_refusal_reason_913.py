"""#913/W42: the refusing component's NAME is not the defect.

WHAT THE 0826 ACCEPTANCE WINDOW MEASURED, and could not act on. With
``SGLANG_MATCH_REFUSAL_CENSUS_EVERY=50`` armed over 93 phase flips, the #904
census emitted exactly five distinct lines
(``/spinning/evidence-665-f1/boot_accept0826r7fix_0826_1817.log``):

    361  verdict=refused reached=45   accepted=0    refused=45 dead=0 refusers=MambaComponent:45
    301  verdict=refused reached=49   accepted=0    refused=49 dead=0 refusers=MambaComponent:49
      9  verdict=refused reached=47   accepted=0    refused=47 dead=0 refusers=MambaComponent:47
      9  verdict=not_present reached=0 accepted=0   refused=0  dead=0 refusers=-
      5  verdict=hit      reached=4096 accepted=4047 refused=49 dead=0 refusers=MambaComponent:49

671 of 675 walks refused, MambaComponent every time. The line names a
component and a token count and stops there -- and behind that one bit sit
TWO conditions whose fixes are in different files pointing in opposite
directions:

  * the node carries no recurrent state at all (the WRITE-side tombstone left
    by ``commit_insert_component_data``), or
  * it carries one at a position that is not a multiple of
    ``--mamba-checkpoint-interval`` (a READ-side determinism policy declining
    a usable anchor on purpose, #747).

Acting on the wrong one is not merely wasted work. Relaxing the grid rule
when the true cause was an absent state is the #767 corruption direction
verbatim: a match reaching a depth its recurrent state never did.

So this is an INSTRUMENT ticket, not a policy one. Nothing here changes which
nodes are accepted -- ``is_resume_candidate`` is now *defined* in terms of the
explanation, so the predicate cannot drift from it, and the test below pins
exactly that. What changes is that the next armed boot answers the question
instead of re-posing it.
"""

import unittest

from sglang.srt.mem_cache.mamba_ckpt_utils import (
    RESUME_REFUSAL_ABSENT,
    RESUME_REFUSAL_OFF_GRID,
    is_resume_candidate,
    resume_refusal_reason,
)
from sglang.srt.mem_cache.match_refusal_census import MatchOutcome, MatchRefusalCensus
from sglang.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

INTERVAL = 4096  # the value the 0826 window actually ran, from its own log


class TestTheTwoReasonsAreSeparated(CustomTestCase):
    """The whole point: one bit became two named causes."""

    def test_a_node_with_no_state_is_absent_not_off_grid(self):
        # Depth 45 is off the 4096 grid AND the node holds nothing. Presence is
        # decided first on purpose: there is no checkpoint to be on or off a
        # grid, and reporting the grid here sends a reader to the read-side
        # policy for a write-side hole.
        self.assertEqual(
            resume_refusal_reason(45, INTERVAL, has_device_value=False),
            RESUME_REFUSAL_ABSENT,
        )

    def test_a_node_with_a_state_off_the_grid_says_off_grid(self):
        self.assertEqual(
            resume_refusal_reason(45, INTERVAL, has_device_value=True),
            RESUME_REFUSAL_OFF_GRID,
        )

    def test_an_admissible_anchor_has_no_reason(self):
        self.assertIsNone(resume_refusal_reason(4096, INTERVAL, has_device_value=True))

    def test_host_backed_state_counts_only_when_the_host_tier_is_consulted(self):
        # device_only=True is the non-HiCache walk: a host copy is not a state
        # it may resume from, so the node reads ABSENT.
        self.assertEqual(
            resume_refusal_reason(
                4096, INTERVAL, has_device_value=False, has_host_value=True
            ),
            RESUME_REFUSAL_ABSENT,
        )
        # device_only=False is the HiCache walk, where the same node is a hit.
        self.assertIsNone(
            resume_refusal_reason(
                4096,
                INTERVAL,
                has_device_value=False,
                has_host_value=True,
                device_only=False,
            )
        )

    def test_no_interval_can_never_produce_an_off_grid_refusal(self):
        # interval=None degenerates to the pure presence test (#747), so the
        # grid term must be unreachable rather than merely usually false.
        for depth in (0, 1, 45, 4095, 4096, 999983):
            self.assertIsNone(resume_refusal_reason(depth, None, has_device_value=True))
            self.assertEqual(
                resume_refusal_reason(depth, None, has_device_value=False),
                RESUME_REFUSAL_ABSENT,
            )


class TestThePredicateCannotDriftFromTheExplanation(CustomTestCase):
    """#747 records these two match lineages drifting apart once already.

    ``is_resume_candidate`` is now DEFINED as "no reason", so the two agree by
    construction rather than by review. This pins that they still do, over the
    whole input space rather than at the three points a reviewer would pick.
    """

    def test_agreement_over_the_full_cross_product(self):
        checked = 0
        for interval in (None, 1, 45, 4096):
            for depth in (0, 1, 44, 45, 49, 4095, 4096, 8192):
                for has_device in (False, True):
                    for has_host in (False, True):
                        for device_only in (False, True):
                            reason = resume_refusal_reason(
                                depth,
                                interval,
                                has_device_value=has_device,
                                has_host_value=has_host,
                                device_only=device_only,
                            )
                            ok = is_resume_candidate(
                                depth,
                                interval,
                                has_device_value=has_device,
                                has_host_value=has_host,
                                device_only=device_only,
                            )
                            self.assertEqual(
                                ok,
                                reason is None,
                                f"drift at depth={depth} interval={interval} "
                                f"dev={has_device} host={has_host} "
                                f"device_only={device_only}: "
                                f"candidate={ok} reason={reason}",
                            )
                            checked += 1
        # A loop that silently ran zero times is a green from the wrong cause.
        self.assertEqual(checked, 4 * 8 * 2 * 2 * 2)


class _StubComponentData:
    def __init__(self, value=None, host_value=None):
        self.value = value
        self.host_value = host_value


class _StubNode:
    def __init__(self, ct, data):
        self.component_data = {ct: data}


class _StubCache:
    def __init__(self, is_eagle=False, cache_controller=None):
        self.is_eagle = is_eagle
        self.cache_controller = cache_controller


class _StubMambaComponent:
    """Carries only what ``explain_match_refusal`` reads.

    The REAL method body runs against this -- it is bound off the class, not
    reimplemented -- so a change to the method is caught here. Constructing a
    true ``MambaComponent`` needs a device pool; a stand-in that re-stated the
    logic would be a test of the test.
    """

    component_type = "mamba"

    def __init__(self, interval, is_eagle=False, cache_controller=None):
        self.mamba_checkpoint_interval = interval
        self.cache = _StubCache(is_eagle, cache_controller)

    _raw_token_pos = MambaComponent._raw_token_pos
    explain_match_refusal = MambaComponent.explain_match_refusal


class TestTheComponentExplainsItself(CustomTestCase):
    def _node(self, **kw):
        return _StubNode("mamba", _StubComponentData(**kw))

    def test_tombstone_node_reports_absent(self):
        """A node the commit path declined: KV cached, mamba value None."""
        comp = _StubMambaComponent(INTERVAL)
        self.assertEqual(
            comp.explain_match_refusal(self._node(value=None), 45),
            RESUME_REFUSAL_ABSENT,
        )

    def test_off_grid_node_holding_a_state_reports_off_grid(self):
        comp = _StubMambaComponent(INTERVAL)
        self.assertEqual(
            comp.explain_match_refusal(self._node(value=object()), 45),
            RESUME_REFUSAL_OFF_GRID,
        )

    def test_an_admissible_node_reports_no_reason(self):
        """On a multi-component walk the refuser may be someone else. 'No
        reason' is then the truthful answer, not a missing one."""
        comp = _StubMambaComponent(INTERVAL)
        self.assertIsNone(
            comp.explain_match_refusal(self._node(value=object()), INTERVAL)
        )

    def test_the_grid_is_read_in_raw_tokens_under_eagle(self):
        """#783: k bigrams span k+1 raw tokens. The explanation must use the
        same unit system as the predicate or it would name a term the
        predicate never evaluated."""
        comp = _StubMambaComponent(INTERVAL, is_eagle=True)
        # 4095 KEY units == 4096 RAW tokens == on grid.
        self.assertIsNone(comp.explain_match_refusal(self._node(value=object()), 4095))
        # 4096 KEY units == 4097 RAW tokens == off grid.
        self.assertEqual(
            comp.explain_match_refusal(self._node(value=object()), 4096),
            RESUME_REFUSAL_OFF_GRID,
        )


class TestTheCensusCarriesTheReason(CustomTestCase):
    def test_reason_is_recorded_beside_the_component(self):
        c = MatchRefusalCensus()
        c.note_reached(45)
        c.note_refused("MambaComponent", 45, RESUME_REFUSAL_ABSENT)
        self.assertEqual(c.verdict(), MatchOutcome.REFUSED)
        self.assertEqual(c.refused_tokens_by_component, {"MambaComponent": 45})
        self.assertEqual(c.refused_tokens_by_reason, {"MambaComponent:absent": 45})

    def test_a_component_that_cannot_say_is_visible_as_unexplained(self):
        """#829: an instrument that silently omits what it did not measure is
        indistinguishable from one that measured nothing."""
        c = MatchRefusalCensus()
        c.note_reached(45)
        c.note_refused("SWAComponent", 45)
        self.assertEqual(c.refused_tokens_by_reason, {"SWAComponent:unexplained": 45})

    def test_the_reason_reaches_the_log_line(self):
        c = MatchRefusalCensus()
        c.note_reached(45)
        c.note_refused("MambaComponent", 45, RESUME_REFUSAL_OFF_GRID)
        fields = c.log_fields()
        self.assertEqual(fields["why"], "MambaComponent:off_grid=45")
        self.assertIn("why=MambaComponent:off_grid=45", c.format_line())

    def test_the_reason_axis_does_not_disturb_the_partition(self):
        """The #904 partition is computed from reached and accepted, never by
        summing an attribution dict. Adding a second attribution axis must not
        change that."""
        c = MatchRefusalCensus()
        c.note_reached(4096)
        c.note_accepted(4047)
        c.note_refused("MambaComponent", 49, RESUME_REFUSAL_ABSENT)
        c.note_refused("SWAComponent", 49)
        c.check_partition()  # raises if the parts stopped summing
        self.assertEqual(c.verdict(), MatchOutcome.HIT)
        self.assertEqual(c.refused_tokens, 49)
        # Two components blamed the same node; the reason axis double-counts
        # in step with the component axis, exactly as documented.
        self.assertEqual(sum(c.refused_tokens_by_reason.values()), 98)


class TestTheMeasuredWindowLineBecomesActionable(CustomTestCase):
    """Reconstruct the two shapes the 0826 window could not tell apart.

    Both produce byte-identical ``refusers=MambaComponent:45``. Before this
    ticket that was the end of the line; the assertion below is that they are
    now distinguishable on the same record.
    """

    def _census_for(self, has_state):
        comp = _StubMambaComponent(INTERVAL)
        node = _StubNode(
            "mamba", _StubComponentData(value=object() if has_state else None)
        )
        c = MatchRefusalCensus()
        c.note_reached(45)
        c.note_refused("MambaComponent", 45, comp.explain_match_refusal(node, 45))
        return c

    def test_both_shapes_still_read_identically_on_the_component_axis(self):
        tombstone = self._census_for(has_state=False)
        off_grid = self._census_for(has_state=True)
        self.assertEqual(
            tombstone.log_fields()["refusers"], off_grid.log_fields()["refusers"]
        )
        self.assertEqual(tombstone.log_fields()["refusers"], "MambaComponent:45")

    def test_and_are_now_separated_on_the_reason_axis(self):
        tombstone = self._census_for(has_state=False)
        off_grid = self._census_for(has_state=True)
        self.assertEqual(tombstone.log_fields()["why"], "MambaComponent:absent=45")
        self.assertEqual(off_grid.log_fields()["why"], "MambaComponent:off_grid=45")
        self.assertNotEqual(tombstone.log_fields()["why"], off_grid.log_fields()["why"])


class TestTheWalkAsksForTheReason(CustomTestCase):
    """PRESENT-AND-VERDRAHTET, not merely present.

    A helper nothing calls is the middle of the three delivery states and the
    most expensive to mistake for either end.
    """

    def test_the_match_walk_calls_explain_match_refusal(self):
        import inspect

        from sglang.srt.mem_cache import unified_radix_cache

        src = inspect.getsource(
            unified_radix_cache.UnifiedRadixCache._match_prefix_helper
        )
        self.assertIn("explain_match_refusal", src)
        self.assertIn("census.note_refused(name, tokens, reason)", src)

    def test_every_component_answers_the_hook(self):
        """It is declared on the base so no component can be silently inert."""
        from sglang.srt.mem_cache.unified_cache_components import tree_component

        self.assertTrue(hasattr(tree_component.TreeComponent, "explain_match_refusal"))


if __name__ == "__main__":
    unittest.main()
