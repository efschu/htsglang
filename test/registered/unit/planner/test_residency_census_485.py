"""Hermetic tests for the #485 residency census owner grouping.

No GPU and no model: the walk is fed synthetic ``(name, tensor)`` pairs, so
the grouping rule is pinned independently of any checkpoint.

The regression that motivates this file: the first version of the classifier
identified attention layers by matching ``self_attn`` -- the name the
CHECKPOINT uses. The loaded sglang module calls it ``attn``
(``RadixAttention``), so on the first real boot every attention layer was
reported as unclassified and the census's own family split was silently
wrong. It was caught because the numbers had to add up against the KV arena,
not because anything failed. The rule is now positive on the linear family
and exclusive on the other, and this file holds it there.
"""

import unittest

from sglang.srt.planner import residency_census
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _T:
    """The smallest thing the walk needs: bytes and a device kind."""

    class _Dev:
        def __init__(self, kind):
            self.type = kind

    def __init__(self, nbytes, kind="cuda"):
        self.nbytes = nbytes
        self.device = self._Dev(kind)


MIB = 1 << 20


class TestOwnerGrouping(CustomTestCase):
    def test_loaded_module_names_are_classified_not_dropped(self):
        # The exact shapes the loaded model produces: attention layers expose
        # `attn`, never `self_attn`.
        named = [
            ("model.language_model.layers.0.linear_attn.in_proj_a.weight", _T(MIB)),
            ("model.language_model.layers.0.mlp.up_proj.weight", _T(2 * MIB)),
            ("model.language_model.layers.3.attn.qkv_proj.weight", _T(4 * MIB)),
            ("model.language_model.layers.3.mlp.up_proj.weight", _T(2 * MIB)),
        ]
        g = residency_census.group_parameter_bytes(named)
        self.assertEqual(g["n_layers_linear"], 1)
        self.assertEqual(g["n_layers_attention"], 1)
        self.assertEqual(g["layers_linear"], 3 * MIB)
        self.assertEqual(g["layers_attention"], 6 * MIB)
        self.assertNotIn("layers_unclassified", g)

    def test_checkpoint_spelling_also_classifies(self):
        # Whichever spelling shows up, the answer must be the same -- the rule
        # must not depend on the attention module's name at all.
        named = [
            ("model.language_model.layers.3.self_attn.q_proj.weight", _T(4 * MIB)),
            ("model.language_model.layers.0.linear_attn.in_proj_b.weight", _T(MIB)),
        ]
        g = residency_census.group_parameter_bytes(named)
        self.assertEqual(g["n_layers_attention"], 1)
        self.assertEqual(g["n_layers_linear"], 1)

    def test_non_layer_owners_are_separated_by_role(self):
        named = [
            ("model.language_model.embed_tokens.weight", _T(10 * MIB)),
            ("lm_head.weight", _T(10 * MIB)),
            ("model.visual.blocks.0.attn.qkv.weight", _T(3 * MIB)),
            ("model.language_model.norm.weight", _T(1)),
        ]
        g = residency_census.group_parameter_bytes(named)
        self.assertEqual(g["embed_tokens"], 10 * MIB)
        self.assertEqual(g["lm_head"], 10 * MIB)
        self.assertEqual(g["visual"], 3 * MIB)
        self.assertEqual(g["other"], 1)
        # The vision tower's own `layers`-free naming must not leak into the
        # transformer census.
        self.assertNotIn("layers_attention", g)

    def test_draft_weights_do_not_pollute_the_layer_census(self):
        # The MTP/NEXTN head has `layers.N.` names too. Counting them as
        # stage layers would inflate the per-layer figure the gate divides by.
        named = [
            ("model.mtp.layers.0.attn.qkv_proj.weight", _T(7 * MIB)),
            ("model.language_model.layers.3.attn.qkv_proj.weight", _T(4 * MIB)),
        ]
        g = residency_census.group_parameter_bytes(named)
        self.assertEqual(g["draft_mtp"], 7 * MIB)
        self.assertEqual(g["layers_attention"], 4 * MIB)
        self.assertEqual(g["n_layers_attention"], 1)

    def test_host_tensors_are_not_counted(self):
        named = [
            ("model.language_model.layers.3.attn.qkv_proj.weight", _T(4 * MIB)),
            ("model.language_model.layers.3.attn.k_proj.weight", _T(99 * MIB, "cpu")),
        ]
        g = residency_census.group_parameter_bytes(named)
        self.assertEqual(g["layers_attention"], 4 * MIB)

    def test_the_census_is_off_by_default(self):
        # The whole point of the instrument is that it can ride along on a
        # corridor-measuring boot. If it were ever on by default it would be
        # perturbing every window on this line.
        self.assertFalse(residency_census.census_enabled())



class TestTheTransientCensus(CustomTestCase):
    """#485/law 31: the per-load-state transient instrument."""

    def _census(self, baseline_mib=2038.0):
        from sglang.srt.planner.transient_census import TransientCensus

        return TransientCensus(0, "RTX 5090", int(baseline_mib * 1024 * 1024))

    @staticmethod
    def _mib(n):
        return int(n * 1024 * 1024)

    def test_it_is_off_by_default(self):
        from sglang.srt.planner import transient_census

        self.assertFalse(transient_census.census_enabled())
        self.assertFalse(transient_census.ARMED)
        # And the scheduler's per-batch call must be a no-op in that state.
        self.assertIsNone(transient_census.note("DECODE"))

    def test_the_worst_state_is_the_one_reported(self):
        c = self._census()
        c.note("DECODE", self._mib(1800))
        c.note("EXTEND", self._mib(900))
        self.assertEqual(c.worst(), "EXTEND")
        self.assertAlmostEqual(c.draw_mib()["EXTEND"], 1138.0, delta=0.01)

    def test_the_draw_is_referenced_to_the_post_capture_level(self):
        # A CORRECTION I MADE TO MYSELF MID-SHIFT. On a phase-flip boot the
        # boot-time backing swap releases the non-resident layout AFTER
        # capture, so free at rest ends up GiB above the armed baseline, and
        # referencing draws to the highest observed free is tempting. It is
        # wrong: fixed_overhead_mib is calibrated at the post-capture point
        # and already counts the layout that is later released, so the higher
        # reference charges it twice on the rank where the constraint binds.
        c = self._census(baseline_mib=2038.0)
        c.note("DECODE", self._mib(7000))  # a flip phase released memory
        c.note("EXTEND", self._mib(1355))
        self.assertAlmostEqual(c.draw_mib()["EXTEND"], 683.0, delta=0.01)
        self.assertEqual(c.draw_mib()["DECODE"], 0.0)
        # The higher level is still RECORDED, because a reader must know that
        # "at rest" is not one number under a running flip.
        self.assertAlmostEqual(
            c.payload()["max_free_observed_mib"], 7000.0, delta=0.01
        )

    def test_the_raw_minima_are_kept_so_a_reader_can_re_reference(self):
        c = self._census()
        c.note("EXTEND", self._mib(1400))
        payload = c.payload()
        self.assertAlmostEqual(
            payload["min_free_mib_by_load_state"]["EXTEND"], 1400.0, delta=0.01
        )
        self.assertIn("max_free_observed_mib", payload)
        self.assertIn("baseline_free_mib", payload)

    def test_a_state_never_seen_is_absent_rather_than_zero(self):
        c = self._census()
        c.note("DECODE", self._mib(1800))
        self.assertNotIn("EXTEND", c.draw_mib())

    def test_nothing_is_written_when_nothing_was_measured(self):
        import tempfile

        c = self._census()
        self.assertIsNone(c.write(tempfile.mkdtemp()))


if __name__ == "__main__":
    unittest.main()
