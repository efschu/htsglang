"""#753: a gapped layer set must be REFUSED until the mid-loop wire exists.

``parse_pp_layer_sets`` validates that every layer is owned exactly once and
lies in range. It does NOT validate that a stage's layers are CONTIGUOUS, and
nothing downstream does either.

That gap is not cosmetic. ``qwen3_5.py:1466-1518`` exchanges ``pp_proxy_tensors``
ONCE per rank, at the stage boundary. A rank that owns a gapped set therefore
runs its layers back to back -- layer 2 straight into layer 4 -- with the peer
layers in between simply skipped. No exception, no warning: the model produces
fluent, confidently wrong output, which is the worst failure shape there is.

So until the wire lands, the honest answer to a gapped set is a refusal that
names why. ``allow_gapped=True`` is how the wire, once wired, says it can carry
one -- an explicit opt-in rather than a default, because the default has to be
the safe reading.
"""

import unittest

from sglang.srt.distributed.utils import PPLayerSetError, parse_pp_layer_sets
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")

N = 64

#: The user's target layout: GDN on stage 0, the 16 full-attention layers split
#: across the two 3080s. Stages 1 and 2 are GAPPED by construction -- FA sits at
#: every 4th layer -- which is exactly why this feature needs the wire.
FA = [i for i in range(N) if i % 4 == 3]
GDN = [i for i in range(N) if i % 4 != 3]
TARGET = (
    ",".join(map(str, GDN))
    + ";"
    + ",".join(map(str, FA[:8]))
    + ";"
    + ",".join(map(str, FA[8:]))
)


class TestGappedSetsAreRefused(CustomTestCase):
    def test_the_contiguous_split_is_still_accepted(self):
        """The #735 step-1 layout must keep working, unchanged."""
        owned = parse_pp_layer_sets("0-30;31-47;48-63", N, 3)
        self.assertEqual(len(owned), 3)
        self.assertEqual(sorted(l for s in owned for l in s), list(range(N)))

    def test_a_gapped_stage_is_refused_by_default(self):
        with self.assertRaises(PPLayerSetError) as ctx:
            parse_pp_layer_sets("0-30,32-47;31;48-63", N, 3)
        msg = str(ctx.exception)
        self.assertIn("#753", msg)
        self.assertIn("contiguous", msg.lower())

    def test_the_refusal_names_the_stage_and_the_gap(self):
        with self.assertRaises(PPLayerSetError) as ctx:
            parse_pp_layer_sets("0-30,32-47;31;48-63", N, 3)
        msg = str(ctx.exception)
        self.assertIn("stage 0", msg)
        self.assertIn("31", msg, "the missing interior layer must be named")

    def test_the_users_target_layout_is_refused_without_the_wire(self):
        """The whole point: the fast layout is gapped and must not run silently."""
        with self.assertRaises(PPLayerSetError) as ctx:
            parse_pp_layer_sets(TARGET, N, 3)
        self.assertIn("#753", str(ctx.exception))

    def test_allow_gapped_admits_it(self):
        """The wire's opt-in. Explicit, never the default."""
        owned = parse_pp_layer_sets(TARGET, N, 3, allow_gapped=True)
        self.assertEqual(len(owned), 3)
        self.assertEqual(sorted(l for s in owned for l in s), list(range(N)))
        self.assertEqual(sorted(owned[1]), FA[:8])

    def test_allow_gapped_does_not_weaken_the_other_validations(self):
        """CAN-FAIL GUARD: the opt-in must not become a bypass.

        A gapped set is admissible with the wire; a set that loses or
        duplicates a layer never is, wire or not.
        """
        with self.assertRaises(PPLayerSetError):
            parse_pp_layer_sets("0-29;31-47;48-63", N, 3, allow_gapped=True)
        with self.assertRaises(PPLayerSetError):
            parse_pp_layer_sets("0-31;31-47;48-63", N, 3, allow_gapped=True)
        with self.assertRaises(PPLayerSetError):
            parse_pp_layer_sets("0-30;31-47;48-64", N, 3, allow_gapped=True)

    def test_a_single_layer_stage_is_contiguous(self):
        """Degenerate case: one layer has no interior to be gapped."""
        parse_pp_layer_sets("0-30;31;32-63", N, 3)


if __name__ == "__main__":
    unittest.main()


class TestPpSizeOneIsNotAnError754(CustomTestCase):
    """#754, folded into #753: it is the same resolution seam.

    ``SGLANG_PP_LAYER_SET`` is process-wide, but ``get_pp_layer_set`` is called
    again by the TP stack during a phase flip -- with ``pp_size=1``. A 3-stage
    string is then not merely inapplicable but INVALID, and the parser refused
    it by stage count ("3 stage(s) given but pp_size is 1"), taking down a flip
    that had nothing to do with layer sets.

    A single stage owns every layer, so the set form has nothing to express.
    ``None`` hands the caller back to ``get_pp_indices`` -- the correct
    contiguous answer, not a suppressed error.
    """

    RAW = "0-30;31-47;48-63"

    def _get(self, pp_rank, pp_size):
        import os
        from unittest.mock import patch

        from sglang.srt.distributed.utils import get_pp_layer_set

        with patch.dict(os.environ, {"SGLANG_PP_LAYER_SET": self.RAW}):
            return get_pp_layer_set(N, pp_rank, pp_size)

    def test_pp_size_one_answers_none_instead_of_raising(self):
        self.assertIsNone(self._get(0, 1))

    def test_the_real_pp_size_still_resolves(self):
        """CAN-FAIL GUARD: the pp_size=1 exit must not swallow the real path."""
        owned = self._get(1, 3)
        self.assertIsNotNone(owned)
        self.assertEqual(sorted(owned)[0], 31)

    def test_a_genuinely_wrong_stage_count_is_still_refused(self):
        """The exit is for pp_size=1 only, not a blanket amnesty."""
        import os
        from unittest.mock import patch

        from sglang.srt.distributed.utils import get_pp_layer_set

        with patch.dict(os.environ, {"SGLANG_PP_LAYER_SET": self.RAW}):
            with self.assertRaises(PPLayerSetError):
                get_pp_layer_set(N, 0, 2)
