"""Unit tests for --rank-kv-ratio (task #88): the DECOUPLED uneven-DCP
KV-token ownership knob. CPU only, NVML mocked.

Covers: CLI parsing ('coupled'/'capacity'/'auto' alias/explicit vector),
fail-fast validation in _handle_uneven_tp, the graceful 'capacity'
degeneration on a collapsed-to-even auto plan, the env-free weighted-DCP
implication (dcp_size auto-set + uneven_weighted_dcp_enabled), and the
resolve_cp_token_ratios precedence (env > flag vector > derivations).
"""

import argparse
import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import sglang.srt.server_args as server_args_module
from sglang.srt.distributed.utils import resolve_cp_token_ratios
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# GPU 0: 32 GiB card, GPUs 1/2: 20 GiB cards (CUDA device indices).
FAKE_GPU_MEMORY = {
    0: (32768, 30000),
    1: (20480, 19000),
    2: (20480, 19000),
}

_UNEVEN_ENVS = (
    "SGLANG_UNEVEN_DCP",
    "SGLANG_UNEVEN_DCP_WEIGHTED",
    "SGLANG_UNEVEN_TOKEN_VECTOR",
)


def _fake_query(gpu_ids):
    result = {}
    for gpu_id in sorted(set(gpu_ids)):
        result[gpu_id] = FAKE_GPU_MEMORY[gpu_id]
    return result


def make_args(**kwargs):
    return ServerArgs(model_path="dummy", **kwargs)


def run_handler(args):
    with patch.object(
        server_args_module, "_query_rank_gpu_memory_mib", _fake_query
    ):
        args._handle_uneven_tp()
    return args


class KvRatioTestCase(CustomTestCase):
    def setUp(self):
        # The flag must work without the env pair; make sure the test env
        # is clean of them.
        self._env = patch.dict(
            os.environ, {}, clear=False
        )
        self._env.start()
        for name in _UNEVEN_ENVS:
            os.environ.pop(name, None)

    def tearDown(self):
        self._env.stop()


class TestCliParsing(KvRatioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def parse(self, *extra):
        return self.parser.parse_args(["--model-path", "m", *extra])

    def test_default_is_coupled(self):
        self.assertEqual(self.parse().rank_kv_ratio, "coupled")

    def test_modes_and_auto_alias(self):
        self.assertEqual(
            self.parse("--rank-kv-ratio", "coupled").rank_kv_ratio, "coupled"
        )
        self.assertEqual(
            self.parse("--rank-kv-ratio", "capacity").rank_kv_ratio, "capacity"
        )
        self.assertEqual(
            self.parse("--rank-kv-ratio", "auto").rank_kv_ratio, "capacity"
        )

    def test_vector(self):
        self.assertEqual(
            self.parse("--rank-kv-ratio", "5,4,4").rank_kv_ratio, [5, 4, 4]
        )

    def test_invalid_string_rejected(self):
        with self.assertRaises(SystemExit):
            self.parse("--rank-kv-ratio", "bogus")


class TestValidation(KvRatioTestCase):
    def test_default_path_untouched(self):
        args = run_handler(make_args())
        self.assertEqual(args.rank_kv_ratio, "coupled")
        self.assertFalse(args.uneven_kv_flag_active())
        self.assertFalse(args.uneven_weighted_dcp_enabled())

    def test_flag_requires_uneven_plan(self):
        with self.assertRaisesRegex(ValueError, "rank-kv-ratio"):
            run_handler(make_args(rank_kv_ratio="capacity"))
        with self.assertRaisesRegex(ValueError, "rank-kv-ratio"):
            run_handler(make_args(rank_kv_ratio=[2, 1, 1]))

    def _uneven(self, **kwargs):
        return make_args(
            tp_size=3,
            rank_gpu_id=[0, 1, 2],
            rank_tp_ratio=[2, 1, 1],
            rank_gpu_memory_mib=[26000, 17000, 17000],
            **kwargs,
        )

    def test_vector_gcd_reduced(self):
        args = run_handler(self._uneven(rank_kv_ratio=[4, 2, 2]))
        self.assertEqual(args.rank_kv_ratio, [2, 1, 1])
        self.assertTrue(args.uneven_kv_flag_active())
        self.assertTrue(args.uneven_weighted_dcp_enabled())

    def test_all_equal_vector_is_legal(self):
        # Uniform token ownership under uneven weights (even-modulo rule).
        args = run_handler(self._uneven(rank_kv_ratio=[3, 3, 3]))
        self.assertEqual(args.rank_kv_ratio, [1, 1, 1])

    def test_vector_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "length"):
            run_handler(self._uneven(rank_kv_ratio=[2, 1]))

    def test_vector_nonpositive(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            run_handler(self._uneven(rank_kv_ratio=[2, 0, 1]))

    def test_dcp_auto_set_without_env(self):
        # Non-'coupled' implies the weighted-DCP path: dcp_size == tp_size
        # without SGLANG_UNEVEN_DCP/_WEIGHTED.
        args = run_handler(self._uneven(rank_kv_ratio="capacity"))
        self.assertEqual(args.dcp_size, 3)
        self.assertTrue(args.uneven_kv_capacity_mode())

    def test_coupled_does_not_auto_set_dcp(self):
        args = run_handler(self._uneven())
        self.assertEqual(args.dcp_size, 1)

    def test_capacity_degrades_on_even_auto_plan(self):
        # GPUs 1 and 2 have identical budgets -> the VRAM-auto plan
        # collapses to the even split; 'capacity' falls back to 'coupled'
        # with a warning instead of erroring.
        args = run_handler(
            make_args(
                tp_size=2,
                rank_gpu_id=[1, 2],
                rank_tp_ratio="auto",
                rank_auto_reserve_mib=2048,
                rank_kv_ratio="capacity",
            )
        )
        self.assertIsNone(args.rank_tp_ratio)
        self.assertEqual(args.rank_kv_ratio, "coupled")

    def test_explicit_vector_errors_on_even_auto_plan(self):
        with self.assertRaisesRegex(ValueError, "even split"):
            run_handler(
                make_args(
                    tp_size=2,
                    rank_gpu_id=[1, 2],
                    rank_tp_ratio="auto",
                    rank_auto_reserve_mib=2048,
                    rank_kv_ratio=[3, 2],
                )
            )


class TestResolvePrecedence(KvRatioTestCase):
    """resolve_cp_token_ratios: env > --rank-kv-ratio vector > derivations."""

    def _args(self, **kwargs):
        base = dict(
            rank_tp_ratio=[2, 1, 1],
            dcp_size=3,
            rank_gpu_memory_mib=None,
            model_path=None,
            rank_kv_ratio="coupled",
        )
        base.update(kwargs)
        return SimpleNamespace(**base)

    def test_flag_vector_wins_over_weights_fallback(self):
        args = self._args(rank_kv_ratio=[6, 4, 4])
        self.assertEqual(
            resolve_cp_token_ratios(args, checkpoint_size_mib=0), [3, 2, 2]
        )

    def test_env_wins_over_flag_vector(self):
        args = self._args(rank_kv_ratio=[6, 4, 4])
        with patch.dict(
            os.environ, {"SGLANG_UNEVEN_TOKEN_VECTOR": "5,3,3"}
        ):
            self.assertEqual(
                resolve_cp_token_ratios(args, checkpoint_size_mib=0),
                [5, 3, 3],
            )

    def test_all_equal_flag_vector_means_even_modulo(self):
        args = self._args(rank_kv_ratio=[1, 1, 1])
        self.assertIsNone(resolve_cp_token_ratios(args, checkpoint_size_mib=0))

    def test_coupled_falls_back_to_weights(self):
        args = self._args()
        self.assertEqual(
            resolve_cp_token_ratios(args, checkpoint_size_mib=0), [2, 1, 1]
        )

    def test_capacity_mode_keeps_estimate_chain(self):
        # 'capacity' is not a vector: phase 1 uses the same chain as
        # 'coupled' (here: the weights fallback); the measured install
        # happens later in the model runner.
        args = self._args(rank_kv_ratio="capacity")
        self.assertEqual(
            resolve_cp_token_ratios(args, checkpoint_size_mib=0), [2, 1, 1]
        )

    def test_capacity_seed_beats_estimate(self):
        # Draft-solo phase-1 seed: used INSTEAD of the budget/weights
        # estimate, but the mode string stays 'capacity' so the measured
        # phase-2 install still runs.
        args = self._args(rank_kv_ratio="capacity", rank_kv_capacity_seed=[16, 23, 25])
        self.assertEqual(
            resolve_cp_token_ratios(args, checkpoint_size_mib=0), [16, 23, 25]
        )

    def test_explicit_vector_beats_capacity_seed(self):
        args = self._args(
            rank_kv_ratio=[6, 4, 4], rank_kv_capacity_seed=[16, 23, 25]
        )
        self.assertEqual(
            resolve_cp_token_ratios(args, checkpoint_size_mib=0), [3, 2, 2]
        )

    def test_env_beats_capacity_seed(self):
        args = self._args(
            rank_kv_ratio="capacity", rank_kv_capacity_seed=[16, 23, 25]
        )
        with patch.dict(os.environ, {"SGLANG_UNEVEN_TOKEN_VECTOR": "5,3,3"}):
            self.assertEqual(
                resolve_cp_token_ratios(args, checkpoint_size_mib=0), [5, 3, 3]
            )

    def test_capacity_seed_is_gcd_reduced(self):
        args = self._args(rank_kv_ratio="capacity", rank_kv_capacity_seed=[8, 12, 16])
        self.assertEqual(
            resolve_cp_token_ratios(args, checkpoint_size_mib=0), [2, 3, 4]
        )


class TestSoloPlannerSeed(KvRatioTestCase):
    """The draft-solo planner seed must not downgrade 'capacity' to a pin.

    Writing the predicted vector into ``rank_kv_ratio`` itself made
    ``uneven_kv_capacity_mode()`` False, which cancelled the post-profiling
    measured install (the boot then only logged the un-actioned
    'restart with SGLANG_UNEVEN_TOKEN_VECTOR=...' hint)."""

    def test_capacity_mode_survives_the_seed(self):
        args = make_args(rank_kv_ratio="capacity")
        # What the planner does on the solo path.
        if args.uneven_kv_capacity_mode():
            args.rank_kv_capacity_seed = [16, 23, 25]
        else:  # pragma: no cover - guarded below
            args.rank_kv_ratio = [16, 23, 25]
        self.assertEqual(args.rank_kv_ratio, "capacity")
        self.assertTrue(args.uneven_kv_capacity_mode())
        self.assertTrue(args.uneven_kv_flag_active())
        self.assertEqual(args.rank_kv_capacity_seed, [16, 23, 25])

    def test_coupled_mode_seed_path_unchanged(self):
        args = make_args(rank_kv_ratio="coupled")
        self.assertFalse(args.uneven_kv_capacity_mode())
        args.rank_kv_ratio = [16, 23, 25]
        self.assertIsNone(args.rank_kv_capacity_seed)
        self.assertEqual(args.rank_kv_ratio, [16, 23, 25])

    def test_seed_defaults_to_none(self):
        self.assertIsNone(make_args().rank_kv_capacity_seed)


class TestRankKvRatioIsNeverSilentlyInert(CustomTestCase):
    """#500-B2: accepted-and-then-inert is the #421 wiring failure mode.

    ``--rank-kv-ratio``'s validation was deliberately hoisted ABOVE the
    ``if self.rank_gpu_id is None: ... return`` early return
    (``server_args.py:9644``) because it describes the PARTITION, not the
    placement. The uneven-DCP auto-engage that MAKES it act --
    ``self.dcp_size = self.tp_size`` (``server_args.py:9845``) -- sits BELOW
    that return. So without ``--rank-gpu-id`` the flag passed every check,
    ``uneven_weighted_dcp_enabled()`` answered True, and ``dcp_size`` stayed 1:

        --rank-kv-ratio 'speed' without --rank-gpu-id ->
            rank_kv_ratio='speed' dcp_size=1
            uneven_weighted_dcp_enabled=True kv_replicated=False

    ``uneven_dcp_kv_replicated`` is False there, ``configure_scheduler_process``
    installs no token vector (its gate is ``_dcp_size > 1``,
    ``managers/scheduler.py:5952``), and the existing honesty guard
    ``reject_silently_inert_dcp`` cannot help because it is itself gated on
    ``dcp_size``. The operator asked for the #210 decode lever and was served
    the coupled layout with no line in the log.

    The refusal is chosen over auto-engaging DCP on the no-placement path for
    two reasons. (1) Nothing that works today changes: the combination is inert
    by construction, so no boot can be relying on it. (2) The no-placement path
    exists for the cross-vendor two-launcher arm (``server_args.py:9540-9548``),
    which has no coverage on this rig -- silently switching its KV layout to
    token-sharded DCP is the opposite of what the decoupling was for. The
    refusal names both ways out instead.
    """

    def _refusal(self, **overrides):
        kw = dict(tp_size=3, rank_tp_ratio=[2, 1, 1], rank_kv_ratio="speed")
        kw.update(overrides)
        args = make_args(**kw)
        with self.assertRaises(ValueError) as cm:
            args._handle_uneven_tp()
        return str(cm.exception)

    def test_every_active_mode_is_refused_without_a_placement(self):
        for mode in ("speed", "capacity", "auto", [2, 1, 1]):
            with self.subTest(mode=mode):
                msg = self._refusal(rank_kv_ratio=mode)
                self.assertIn("--rank-kv-ratio", msg)
                # both ways out must be named, not just the problem
                self.assertIn("--rank-gpu-id", msg)
                self.assertIn("--dcp-size", msg)

    def test_the_default_no_placement_path_is_untouched(self):
        """'coupled' is the default and is not a request for anything, so the
        cross-vendor two-launcher arm keeps booting exactly as before."""
        args = make_args(tp_size=3, rank_tp_ratio=[2, 1, 1])
        args._handle_uneven_tp()
        self.assertEqual(args.rank_kv_ratio, "coupled")
        self.assertEqual(args.dcp_size, 1)

    def test_an_explicit_dcp_size_is_the_documented_way_out(self):
        """With DCP engaged by hand the flag is NOT inert, so it must pass."""
        args = make_args(tp_size=3, rank_tp_ratio=[2, 1, 1], rank_kv_ratio="speed")
        args.dcp_size = 3
        args._handle_uneven_tp()
        self.assertEqual(args.rank_kv_ratio, "speed")
        self.assertEqual(args.dcp_size, 3)
        self.assertTrue(args.uneven_weighted_dcp_enabled())

    def test_the_pre_existing_bare_flag_guard_is_unchanged(self):
        """With no uneven-TP flag at all the refusal was ALREADY there
        (``server_args.py:9424``, inside the all-three-None early return) --
        which is why the hole was easy to miss: the bare form refused, and only
        the form WITH a base plan walked through. The new guard completes that
        edge; it must not replace or shadow this one."""
        args = make_args(tp_size=3, rank_kv_ratio="speed")
        with self.assertRaises(ValueError) as cm:
            args._handle_uneven_tp()
        self.assertIn("requires --rank-gpu-id", str(cm.exception))

    def test_the_perf_tune_auto_selection_cannot_trip_this_guard(self):
        """The one false-refusal risk, checked rather than assumed.

        ``--rank-perf-tune dec`` SETS ``rank_kv_ratio = 'speed'`` itself
        (``server_args.py:8981``), so a boot that never typed ``--rank-kv-ratio``
        could in principle land in the new refusal. It cannot: ``_check_perf_flags``
        refuses any ``--rank-perf-tune`` outside ``--rank-tp-ratio
        auto-performance`` (``:8950-8955``), and ``auto-performance`` itself
        requires ``--rank-gpu-id`` (``:8971``). The auto-selection is therefore
        unreachable without a placement."""
        with self.assertRaises(ValueError) as cm:
            make_args(
                tp_size=3, rank_tp_ratio=[2, 1, 1], rank_perf_tune="dec"
            )._handle_uneven_tp()
        self.assertIn("auto-performance", str(cm.exception))

    def test_the_reachable_shape_is_exactly_the_explicit_vector_one(self):
        """Every other no-placement shape is already refused upstream, which
        is the reason this hole was a single narrow one: a per-rank budget
        needs a placement (``--rank-gpu-memory-mib requires --rank-gpu-id``)
        and ``--rank-tp-ratio auto`` needs one too (``:8971``). An EXPLICIT
        base vector needs neither -- and that is the shape that fell through."""
        with self.assertRaises(ValueError) as cm:
            make_args(
                tp_size=3, rank_kv_ratio="speed", rank_gpu_memory_mib=15000
            )._handle_uneven_tp()
        self.assertIn("--rank-gpu-memory-mib requires --rank-gpu-id", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            make_args(
                tp_size=3, rank_kv_ratio="speed", rank_tp_ratio="auto"
            )._handle_uneven_tp()
        self.assertIn("--rank-tp-ratio auto requires --rank-gpu-id", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
