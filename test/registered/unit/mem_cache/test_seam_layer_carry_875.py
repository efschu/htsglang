"""#875: a PP copy restorable in TP and back -- the layer axis, both directions.

THE USER'S ORDER IS "fix #875, do not merely refuse it". `restore_seam_state`
today refuses a cross-layout restore, and that refusal is correct: it is the only
thing between the tree and the W40 IndexError in one direction and a silent
wrong-layer write in the other. But its own counter comment states the cost --
"a layout refusal says the seam carry is structurally impossible in that
direction, i.e. every flip loses its prefixes". The refusal is the honest
non-answer; this is the first axis of the answer.

THE REFUSAL'S OWN JUSTIFICATION IS WHAT HAD TO GO ON TRIAL FIRST, and one leg of
it did not survive. `check_cpu_copy_layers` claimed "the KV head sharding also
differs between the phases (PP holds all heads of its stage, TP a head shard of
every layer), so even the overlapping entries are not interchangeable". FALSE on
this rig: under #345 uneven-DCP replication the full-attention cache is
TOKEN-sharded and every rank stores the FULL kv-heads
(model_runner_kv_cache_mixin.py:3155-3160, and the boot line at :3228 prints
`get_total_num_kv_heads()`); the PP pool takes `get_num_kv_heads(attn_tp_size)`
with `attn_tp_size == 1`, the same total. Identical widths, interchangeable
entries. The other leg -- "rank-locally under PP it cannot cover the layers" --
is TRUE but only rank-locally, and the union of the three PP stages IS every
layer. So the correct statement was never "a remap is impossible", it was "a
RANK-LOCAL remap is impossible, and nobody asked about a collective one".

WHAT THIS FILE COVERS, AND WHAT IT DELIBERATELY DOES NOT. The layer axis only.
Two other axes stand between here and a shipped carry:
  HEAD  -- replicated in both phases, no remap needed. Settled, above.
  TOKEN -- PP holds every token at allocator slots; TP holds an owner-rule
           SUBSET at compacted rows (layers/dcp/owner.py:159). A second
           remap, also unavailable rank-locally, NOT solved.
So nothing here is wired into `restore_seam_state`. A layer-correct,
token-wrong carry is precisely the "matching row ids, mismatched widths"
corruption #719 already walked into once, and the refusal stays until axis 3 is
answered.

BOTH DIRECTIONS ARE TESTED SEPARATELY, WITH DIFFERENT FALSIFIERS, because they
fail differently and a single test cannot see both:
  PP -> TP  the loud arm. Too few entries for the destination; the old code
            walked off the end. Asserted as a REFUSAL that names the missing
            global layers -- an exception assert is the right shape here.
  TP -> PP  the SILENT arm. Enough entries, wrong ones. No exception is
            available to assert on, so this is asserted on CONTENT: every
            destination slot must hold the global layer it is supposed to hold,
            compared value by value. An exception-only test passes this arm by
            doing nothing.

Geometry throughout is this rig's, from the boot log's own server args:
`pp_attn_stage_ratio=[8, 4, 4]` over 16 attention layers.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest

from sglang.srt.mem_cache.seam_layer_carry import (
    SeamCarryError,
    assemble_for,
    carry_across,
    global_layer_ids,
    label_contributions,
)
from sglang.test.test_utils import CustomTestCase

# The rig: 16 attention layers, PP stages 8/4/4.
TP_LAYERS = 16
PP_STAGES = ((0, 8), (8, 4), (12, 4))  # (start_layer, layer_num)


def _entries(start, n):
    """Payload that NAMES the global layer it belongs to, so a wrong-layer
    placement is visible as a value and not merely as an absence."""
    return [f"L{start + i}" for i in range(n)]


class TestTheLabellingIsExact(CustomTestCase):
    """CONTROL. If a copy's entries cannot be labelled with global ids, every
    assertion below is about something else."""

    def test_local_slots_map_to_global_layers(self):
        self.assertEqual(list(range(8, 12)), list(global_layer_ids(4, 8)))

    def test_a_copy_labels_its_entries_by_global_layer(self):
        got = label_contributions(4, 8, _entries(8, 4))
        self.assertEqual({8: "L8", 9: "L9", 10: "L10", 11: "L11"}, got)

    def test_a_payload_that_contradicts_its_label_is_refused(self):
        """The label is the only thing that makes a global id recoverable, so a
        disagreement between label and payload must stop here rather than
        produce confident nonsense."""
        with self.assertRaises(SeamCarryError):
            label_contributions(4, 8, _entries(8, 3))


class TestPPtoTPTheLoudDirection(CustomTestCase):
    """The W40 arm: a PP stage's copy against a destination needing all 16."""

    def test_one_pp_stage_alone_cannot_fill_a_tp_pool(self):
        """RED before the carry existed -- and it must REFUSE, not index. This
        is the case that crashed three ranks 24 s after health 200."""
        start, n = PP_STAGES[1]
        with self.assertRaises(SeamCarryError) as caught:
            carry_across(n, start, _entries(start, n), TP_LAYERS, 0)
        msg = str(caught.exception)
        self.assertIn("0", msg)
        self.assertIn("12", msg, "the refusal must name the missing global layers")

    def test_the_three_pp_stages_TOGETHER_fill_a_tp_pool_exactly(self):
        """THE POINT OF THE WHOLE TICKET. Rank-locally impossible, collectively
        exact -- which is the sentence the old refusal did not contain."""
        peers = [(n, s, _entries(s, n)) for s, n in PP_STAGES]
        got = carry_across(
            PP_STAGES[0][1], PP_STAGES[0][0], _entries(0, 8), TP_LAYERS, 0, peers
        )
        self.assertEqual([f"L{i}" for i in range(TP_LAYERS)], got)

    def test_a_single_missing_stage_is_refused_by_name(self):
        """Two of three stages is the partial restore that must never happen:
        12 of 16 layers right and 4 stale reads as restored."""
        peers = [(n, s, _entries(s, n)) for s, n in PP_STAGES[:2]]
        with self.assertRaises(SeamCarryError) as caught:
            carry_across(8, 0, _entries(0, 8), TP_LAYERS, 0, peers)
        self.assertIn("12", str(caught.exception))


class TestTPtoPPTheSilentDirection(CustomTestCase):
    """The mirror. Before any of this, a TP copy restored into a PP pool ran to
    completion and wrote global layers 0..3 into global layers 8..11 with no
    exception at all. There is no throw to assert on, so these compare CONTENT."""

    def test_a_tp_copy_lands_on_the_stage_s_OWN_global_layers(self):
        start, n = PP_STAGES[1]  # global 8..11
        got = carry_across(TP_LAYERS, 0, _entries(0, TP_LAYERS), n, start)
        self.assertEqual(
            ["L8", "L9", "L10", "L11"],
            got,
            "the destination's local slots must hold ITS global layers",
        )

    def test_the_naive_positional_restore_is_what_this_replaces(self):
        """CONTROL, pinned as a fact about the OLD behaviour so it keeps saying
        what it says: walking the destination's layer count positionally over a
        longer copy takes global layers 0..3 -- the wrong ones."""
        naive = _entries(0, TP_LAYERS)[: PP_STAGES[1][1]]
        self.assertEqual(["L0", "L1", "L2", "L3"], naive)
        start, n = PP_STAGES[1]
        self.assertNotEqual(
            naive,
            carry_across(TP_LAYERS, 0, _entries(0, TP_LAYERS), n, start),
            "the carry must differ from the naive positional restore, or it is "
            "reproducing the silent defect it exists to remove",
        )

    def test_every_stage_selects_its_own_slice_and_they_tile_the_model(self):
        """All three destinations together must reconstruct the model exactly
        once -- no gap, no overlap. A per-stage test alone cannot see a scheme
        that double-counts a layer."""
        seen = []
        for start, n in PP_STAGES:
            seen.extend(carry_across(TP_LAYERS, 0, _entries(0, TP_LAYERS), n, start))
        self.assertEqual([f"L{i}" for i in range(TP_LAYERS)], seen)


class TestTheSameLayoutCaseIsNotSpecialCased(CustomTestCase):
    """The path that already works must go through the same code, or it drifts
    away from the cross-layout path it is supposed to anchor."""

    def test_a_same_layout_carry_is_positionally_identical(self):
        start, n = PP_STAGES[2]
        src = _entries(start, n)
        self.assertEqual(src, carry_across(n, start, src, n, start))

    def test_a_tp_to_tp_carry_is_positionally_identical(self):
        src = _entries(0, TP_LAYERS)
        self.assertEqual(src, carry_across(TP_LAYERS, 0, src, TP_LAYERS, 0))


class TestAssemblyRefusesRatherThanFills(CustomTestCase):
    def test_a_hole_in_the_middle_is_refused(self):
        contributions = {i: f"L{i}" for i in range(TP_LAYERS) if i != 7}
        with self.assertRaises(SeamCarryError) as caught:
            assemble_for(TP_LAYERS, 0, contributions)
        self.assertIn("7", str(caught.exception))

    def test_a_complete_set_assembles_in_destination_order(self):
        contributions = {i: f"L{i}" for i in reversed(range(TP_LAYERS))}
        self.assertEqual(
            [f"L{i}" for i in range(TP_LAYERS)],
            assemble_for(TP_LAYERS, 0, contributions),
            "assembly must order by the DESTINATION's layout, not by insertion",
        )

    def test_a_surplus_contribution_is_ignored_not_appended(self):
        """A TP-wide gather offered to a 4-layer destination must yield 4
        entries. Length is what the pool loop trusts."""
        contributions = {i: f"L{i}" for i in range(TP_LAYERS)}
        got = assemble_for(4, 8, contributions)
        self.assertEqual(["L8", "L9", "L10", "L11"], got)


if __name__ == "__main__":
    unittest.main()
