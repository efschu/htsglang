"""Per-rank MoE resident-expert fraction.

The fraction is the GPU-resident / host-pinned split within one rank's own
expert shard. It became a vector because on a heterogeneous group the right
split differs per card: measured at a uniform 0.45, the 5090 rank of this rig
held ~4.0 GiB of VRAM idle while both 3080 ranks were 32 MiB short.

Two properties carry the change and are pinned here:

* the scalar path -- which is every launch that existed before -- must be
  byte-identical, including the plain-float type the dozen existing readers
  expect;
* a vector must actually produce different resident counts per rank, or the
  flag is decoration.

GPU-free: the planner is pure arithmetic over expert counts.
"""

import os
import unittest
from unittest import mock

from sglang.srt.environ import envs
from sglang.srt.layers.moe.expert_offload import (
    plan_load_time_staging,
    resident_slot_count,
)
from sglang.srt.layers.moe.resident_fraction import (
    describe,
    offload_active,
    resident_fraction_for_rank,
    resident_fraction_vector,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

_ENV = "SGLANG_MOE_RESIDENT_EXPERT_FRACTION"


class _EnvMixin:
    def setUp(self):
        super().setUp()
        self._saved = os.environ.get(_ENV)
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = self._saved

    @staticmethod
    def _set(value):
        if value is None:
            os.environ.pop(_ENV, None)
        else:
            os.environ[_ENV] = value


class TestScalarPathUnchanged(_EnvMixin, CustomTestCase):
    """Backward compatibility. This is the assertion the change must not cost."""

    def test_default_is_a_plain_float_one(self):
        """A dozen readers do `< 1.0` on this. It must stay a float, not
        become a 1-tuple, or every one of them changes meaning."""
        self._set(None)
        value = envs.SGLANG_MOE_RESIDENT_EXPERT_FRACTION.get()
        self.assertIsInstance(value, float)
        self.assertEqual(value, 1.0)
        self.assertFalse(offload_active())

    def test_scalar_still_reads_as_a_plain_float(self):
        self._set("0.45")
        value = envs.SGLANG_MOE_RESIDENT_EXPERT_FRACTION.get()
        self.assertIsInstance(value, float)
        self.assertEqual(value, 0.45)
        self.assertTrue(offload_active())

    def test_scalar_broadcasts_to_every_rank(self):
        self._set("0.45")
        self.assertEqual(resident_fraction_vector(3), (0.45, 0.45, 0.45))
        self.assertEqual(
            [resident_fraction_for_rank(i) for i in range(3)], [0.45, 0.45, 0.45]
        )

    def test_scalar_plan_is_identical_to_passing_the_float_directly(self):
        """The accessor must introduce no drift: routing the same number
        through the new resolver has to produce the same plan the old direct
        env read produced."""
        self._set("0.45")
        for experts in (8, 64, 256, 257):
            with self.subTest(experts=experts):
                via_env = plan_load_time_staging(experts)
                direct = plan_load_time_staging(experts, fraction=0.45)
                self.assertEqual(via_env.resident_count, direct.resident_count)
                self.assertEqual(via_env.buffer_slots, direct.buffer_slots)
                self.assertEqual(via_env.is_static_layout, direct.is_static_layout)

    def test_fraction_one_still_disables_offload_entirely(self):
        self._set("1.0")
        self.assertFalse(offload_active())
        self.assertIsNone(plan_load_time_staging(64))


class TestVectorPath(_EnvMixin, CustomTestCase):
    def test_vector_gives_each_rank_its_own_fraction(self):
        self._set("0.485,0.42,0.42")
        self.assertEqual(resident_fraction_vector(3), (0.485, 0.42, 0.42))
        self.assertEqual(resident_fraction_for_rank(0), 0.485)
        self.assertEqual(resident_fraction_for_rank(1), 0.42)
        self.assertEqual(resident_fraction_for_rank(2), 0.42)

    def test_vector_produces_different_resident_counts_per_rank(self):
        """The point of the whole change. If these counts were equal the flag
        would be decoration."""
        self._set("0.485,0.42,0.42")
        experts = 256
        counts = [
            resident_slot_count(experts, resident_fraction_for_rank(r))
            for r in range(3)
        ]
        self.assertGreater(
            counts[0], counts[1], "the big card must keep MORE experts resident"
        )
        self.assertEqual(counts[1], counts[2], "the two identical cards must agree")
        # And the planner must carry it through, not just the arithmetic.
        plans = [
            plan_load_time_staging(experts, fraction=resident_fraction_for_rank(r))
            for r in range(3)
        ]
        self.assertEqual(plans[0].resident_count, counts[0])
        self.assertEqual(plans[1].resident_count, counts[1])
        self.assertGreater(plans[0].resident_count, plans[1].resident_count)

    def test_offload_active_is_group_wide(self):
        """A rank whose own fraction is 1.0 must still build the offload
        machinery if any peer offloads -- otherwise the ranks diverge
        structurally and the group cannot run."""
        self._set("1.0,0.42,0.42")
        self.assertTrue(offload_active())
        self.assertEqual(resident_fraction_for_rank(0), 1.0)

    def test_describe_names_the_split(self):
        self._set("0.485,0.42,0.42")
        text = describe()
        self.assertIn("rank0=0.485", text)
        self.assertIn("rank1=0.42", text)
        self._set("0.45")
        self.assertIn("uniform", describe())


class TestRefusals(_EnvMixin, CustomTestCase):
    def test_length_mismatch_is_refused(self):
        self._set("0.485,0.42,0.42")
        with self.assertRaises(ValueError) as ctx:
            resident_fraction_vector(2)
        self.assertIn("3 entries", str(ctx.exception))
        self.assertIn("tensor parallelism is 2", str(ctx.exception))

    def test_out_of_range_is_refused(self):
        for bad in ("1.5", "0.0", "-0.2", "0.5,1.5,0.5"):
            with self.subTest(value=bad):
                self._set(bad)
                with self.assertRaises(ValueError):
                    resident_fraction_vector(3)

    def test_scalar_getter_refuses_loudly_once_a_vector_is_set(self):
        """A reader that has not been taught about ranks must fail with a
        message naming the accessor, never silently size a buffer from a
        tuple or from the wrong element."""
        self._set("0.485,0.42,0.42")
        with self.assertRaises(RuntimeError) as ctx:
            envs.SGLANG_MOE_RESIDENT_EXPERT_FRACTION.get()
        message = str(ctx.exception)
        self.assertIn("per rank", message)
        self.assertIn("resident_fraction_for_rank", message)
        self.assertIn("offload_active", message)

    def test_env_and_flag_must_agree(self):
        import sglang.srt.layers.moe.resident_fraction as mod

        self._set("0.485,0.42,0.42")
        with mock.patch.object(
            mod, "_from_flag", lambda server_args=None: (0.5, 0.5, 0.5)
        ):
            with self.assertRaises(ValueError) as ctx:
                resident_fraction_vector(3)
            self.assertIn("disagree", str(ctx.exception))
        # Identical values from both sources are fine.
        with mock.patch.object(
            mod, "_from_flag", lambda server_args=None: (0.485, 0.42, 0.42)
        ):
            self.assertEqual(resident_fraction_vector(3), (0.485, 0.42, 0.42))

    def test_flag_wins_when_env_is_absent(self):
        import sglang.srt.layers.moe.resident_fraction as mod

        self._set(None)
        with mock.patch.object(
            mod, "_from_flag", lambda server_args=None: (0.6, 0.3, 0.3)
        ):
            self.assertEqual(resident_fraction_vector(3), (0.6, 0.3, 0.3))


class TestCanFail(_EnvMixin, CustomTestCase):
    """Falsifiers: each one shows the assertions above are not vacuous."""

    def test_equal_fractions_would_fail_the_per_rank_assertion(self):
        self._set("0.42,0.42,0.42")
        counts = [
            resident_slot_count(256, resident_fraction_for_rank(r)) for r in range(3)
        ]
        with self.assertRaises(AssertionError):
            self.assertGreater(counts[0], counts[1])

    def test_a_broken_broadcast_would_be_caught(self):
        """If the scalar path stopped broadcasting, the vector would be too
        short and every rank past 0 would silently read rank 0's value."""
        self._set("0.45")
        self.assertEqual(len(resident_fraction_vector(3)), 3)
        with self.assertRaises(AssertionError):
            self.assertEqual(len(resident_fraction_vector(3)), 1)


class TestMoeGroupRefusal(_EnvMixin, CustomTestCase):
    """A vector is indexed by the MoE rank but specified per TP rank. When the
    two groups differ that is ambiguous, and ambiguity is refused by name."""

    def test_vector_refused_when_moe_group_differs_from_tp_group(self):
        import sglang.srt.layers.moe.resident_fraction as mod

        self._set("0.485,0.42,0.42")
        with mock.patch.object(mod, "_moe_tp_size", lambda: 6):
            with self.assertRaises(ValueError) as ctx:
                resident_fraction_vector(3)
            message = str(ctx.exception)
            self.assertIn("moe_tp_size=6", message)
            self.assertIn("tp_size=3", message)

    def test_scalar_is_still_fine_when_the_groups_differ(self):
        """A single value is unambiguous no matter how the groups are split."""
        import sglang.srt.layers.moe.resident_fraction as mod

        self._set("0.45")
        with mock.patch.object(mod, "_moe_tp_size", lambda: 6):
            self.assertEqual(resident_fraction_vector(3), (0.45, 0.45, 0.45))


class TestEagerCliValidation(CustomTestCase):
    """Length is checked at CLI-parse time like every sibling rank-vector flag,
    not lazily in a worker six minutes into a 98 GiB checkpoint stream."""

    def _validate(self, vec, tp_size=3):
        """`model_path="dummy"` short-circuits __post_init__ (the pattern the
        other server_args tests use), so the validator is invoked directly --
        which is also what pins that it lives in the always-run part of
        _handle_uneven_tp rather than behind the uneven-plan early return."""
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs(model_path="dummy")
        args.tp_size = tp_size
        args.rank_moe_resident_fraction = vec
        args._handle_uneven_tp()

    def test_wrong_length_is_refused_at_parse_time(self):
        with self.assertRaises(ValueError) as ctx:
            self._validate([0.485, 0.42])
        self.assertIn("--rank-moe-resident-fraction", str(ctx.exception))

    def test_out_of_range_is_refused_at_parse_time(self):
        with self.assertRaises(ValueError) as ctx:
            self._validate([0.485, 0.42, 1.5])
        self.assertIn("(0.0, 1.0]", str(ctx.exception))

    def test_correct_length_and_scalar_both_accepted(self):
        self._validate([0.485, 0.42, 0.42])
        self._validate([0.45])


if __name__ == "__main__":
    unittest.main()
