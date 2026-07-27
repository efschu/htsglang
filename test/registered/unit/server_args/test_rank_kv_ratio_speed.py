"""Unit tests for --rank-kv-ratio speed and --rank-perf-tune dec|maxkv (#210).

'capacity' maximises max_total_num_tokens and, on a heterogeneous rig, hands
the biggest token share to whichever ranks have the most free VRAM after
weights -- typically the WEAK cards. Under DCP each rank runs attention over
the tokens it owns and at bs=1 the group waits on the slowest rank, so that
choice costs deep-context decode. 'speed' is the derived counterpart: the
ownership vector shifted toward the per-rank memory-bandwidth proportion, as
far as --rank-perf-loose-ctx-percent allows.

CPU only, NVML mocked. Covers the flag surface (parsing, the tune->kv-mode
promotion and its precedence rules) and the derivation itself
(cp_token_speed_vector: direction, the loose-ctx floor, the hybrid-cap free
window, and determinism).
"""

import argparse
import os
import unittest
from unittest.mock import patch

import sglang.srt.server_args as server_args_module
from sglang.srt.distributed.utils import (
    cp_token_context_budget,
    cp_token_speed_vector,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# GPU 0: 32 GiB card, GPUs 1/2: 20 GiB cards (CUDA device indices).
FAKE_GPU_MEMORY = {0: (32768, 30000), 1: (20480, 19000), 2: (20480, 19000)}

_UNEVEN_ENVS = (
    "SGLANG_UNEVEN_DCP",
    "SGLANG_UNEVEN_DCP_WEIGHTED",
    "SGLANG_UNEVEN_TOKEN_VECTOR",
)

# The measured shape of the #210 vehicle: the FAST card (rank 0, a 5090) has
# the SMALLEST post-weight token capacity, because the MTP draft and its pools
# live there. That is exactly the case where capacity and speed disagree.
CAPS_5090_POOR = [211241, 316072, 316072]
# Measured bandwidth proportion on the same rig (1558 / 723 / 723 GB/s).
BW = [13, 6, 6]


def make_args(**kwargs):
    return ServerArgs(model_path="dummy", **kwargs)


def run_handler(args):
    with patch.object(
        server_args_module, "_query_rank_gpu_memory_mib", lambda ids: {
            g: FAKE_GPU_MEMORY[g] for g in sorted(set(ids))
        }
    ):
        args._handle_uneven_tp()
    return args


class SpeedTestCase(CustomTestCase):
    def setUp(self):
        self._env = patch.dict(os.environ, {}, clear=False)
        self._env.start()
        for name in _UNEVEN_ENVS:
            os.environ.pop(name, None)

    def tearDown(self):
        self._env.stop()


class TestCliSurface(SpeedTestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(cls.parser)

    def parse(self, *extra):
        return self.parser.parse_args(["--model-path", "m", *extra])

    def test_speed_parses_as_a_mode_string(self):
        args = self.parse("--rank-kv-ratio", "speed")
        self.assertEqual(args.rank_kv_ratio, "speed")

    def test_speed_predicates(self):
        args = make_args(rank_kv_ratio="speed")
        self.assertTrue(args.uneven_kv_speed_mode())
        self.assertTrue(args.uneven_kv_derived_mode())
        self.assertTrue(args.uneven_kv_flag_active())
        self.assertFalse(args.uneven_kv_capacity_mode())

    def test_capacity_predicates_unchanged(self):
        args = make_args(rank_kv_ratio="capacity")
        self.assertTrue(args.uneven_kv_capacity_mode())
        self.assertTrue(args.uneven_kv_derived_mode())
        self.assertFalse(args.uneven_kv_speed_mode())

    def test_coupled_is_neither(self):
        args = make_args()
        self.assertFalse(args.uneven_kv_derived_mode())
        self.assertFalse(args.uneven_kv_speed_mode())

    def test_maxkv_is_an_accepted_tune_target(self):
        self.assertEqual(
            self.parse("--rank-perf-tune", "maxkv").rank_perf_tune, "maxkv"
        )

    def test_speed_weights_default_none(self):
        self.assertIsNone(make_args().rank_kv_speed_weights)


class TestTunePromotion(SpeedTestCase):
    """--rank-perf-tune picks the KV mode only when --rank-kv-ratio was left
    at its default, and only under auto-performance."""

    def _perf_args(self, **kw):
        """Exercise the promotion in isolation.

        The promotion lives in _check_perf_flags, which _handle_uneven_tp calls
        first. Calling it directly keeps the test CPU-only: the full handler
        would go on to resolve 'auto-performance' through NVML device queries
        that are not mockable from here, and none of that is under test."""
        args = make_args(tp_size=3, rank_gpu_id=[0, 1, 2],
                         rank_tp_ratio="auto-performance", **kw)
        args._check_perf_flags(ratio_was_perf=True)
        return args

    def test_dec_selects_speed(self):
        self.assertEqual(self._perf_args(rank_perf_tune="dec").rank_kv_ratio, "speed")

    def test_maxkv_selects_capacity(self):
        self.assertEqual(
            self._perf_args(rank_perf_tune="maxkv").rank_kv_ratio, "capacity"
        )

    def test_both_and_enc_leave_the_default_alone(self):
        for tune in ("both", "enc"):
            self.assertEqual(
                self._perf_args(rank_perf_tune=tune).rank_kv_ratio,
                "coupled",
                msg=tune,
            )

    def test_explicit_kv_ratio_wins_over_the_tune_target(self):
        args = self._perf_args(rank_perf_tune="dec", rank_kv_ratio="capacity")
        self.assertEqual(args.rank_kv_ratio, "capacity")
        args = self._perf_args(rank_perf_tune="maxkv", rank_kv_ratio="speed")
        self.assertEqual(args.rank_kv_ratio, "speed")

    def test_pinned_vector_wins_over_the_tune_target(self):
        args = self._perf_args(rank_perf_tune="dec", rank_kv_ratio=[2, 1, 1])
        self.assertEqual(args.rank_kv_ratio, [2, 1, 1])

    def test_tune_target_rejected_outside_auto_performance(self):
        args = make_args(rank_perf_tune="dec")
        with self.assertRaises(ValueError):
            args._check_perf_flags(ratio_was_perf=False)
        # and it never reaches the KV mode
        self.assertEqual(args.rank_kv_ratio, "coupled")

    def test_unknown_tune_target_rejected(self):
        args = make_args(tp_size=3, rank_gpu_id=[0, 1, 2],
                         rank_tp_ratio="auto-performance",
                         rank_perf_tune="bogus")
        with self.assertRaises(ValueError):
            args._check_perf_flags(ratio_was_perf=True)
        self.assertEqual(args.rank_kv_ratio, "coupled")


class TestSpeedVector(SpeedTestCase):
    def test_capacity_vector_is_the_zero_loose_baseline(self):
        """With no loose-ctx budget and no other ceiling, 'speed' may not give
        up a single token of context, so it must return the capacity vector."""
        cap_vec = [2, 3, 3]  # what 'capacity' installs on this vehicle
        cap_budget = cp_token_context_budget(cap_vec, CAPS_5090_POOR)
        vec, budget, t = cp_token_speed_vector(CAPS_5090_POOR, BW, 0.0)
        # No context may be lost. A tiny shift IS allowed at loose=0 when the
        # integer owner grid happens to fund it -- that is a free gain, which
        # is exactly what a 0 % budget is defined to permit.
        self.assertGreaterEqual(budget, cap_budget)
        self.assertLess(t, 0.1)

    def test_loose_budget_moves_tokens_toward_the_fast_rank(self):
        """The whole point: given room to give up context, ownership moves to
        the high-bandwidth rank."""
        base, _, _ = cp_token_speed_vector(CAPS_5090_POOR, BW, 0.0)
        vec, _, t = cp_token_speed_vector(CAPS_5090_POOR, BW, 50.0)
        self.assertGreater(t, 0.0)
        share_base = base[0] / sum(base)
        share_speed = vec[0] / sum(vec)
        self.assertGreater(share_speed, share_base)

    def test_more_loose_budget_never_moves_back(self):
        shares = []
        for loose in (0.0, 10.0, 25.0, 50.0, 75.0):
            vec, _, _ = cp_token_speed_vector(CAPS_5090_POOR, BW, loose)
            shares.append(vec[0] / sum(vec))
        for a, b in zip(shares, shares[1:]):
            self.assertLessEqual(a, b + 1e-12)

    def test_loose_floor_is_respected(self):
        for loose in (0.0, 10.0, 30.0, 60.0):
            vec, budget, _ = cp_token_speed_vector(CAPS_5090_POOR, BW, loose)
            cap_vec, cap_budget, _ = cp_token_speed_vector(
                CAPS_5090_POOR, BW, 0.0
            )
            self.assertGreaterEqual(
                budget, int(cap_budget * (1 - loose / 100.0)), msg=str(loose)
            )

    def test_hybrid_cap_makes_the_shift_free(self):
        """While the hybrid mamba/SWA cap binds, max_total_num_tokens does not
        change with the vector, so the bandwidth shift costs nothing and must
        be taken in full even at the default loose_ctx_percent=0. This is the
        measured #210 case (both arms reported 393228)."""
        cap_only, _, t_cap = cp_token_speed_vector(CAPS_5090_POOR, BW, 0.0)
        vec, _, t = cp_token_speed_vector(
            CAPS_5090_POOR, BW, 0.0, hard_cap=98328
        )
        self.assertLess(t_cap, 0.1)  # uncapped: essentially no room at 0 %
        self.assertEqual(t, 1.0)  # capped: the whole shift is free
        self.assertGreater(vec[0] / sum(vec), cap_only[0] / sum(cap_only))
        # and the full-shift vector is the bandwidth proportion
        self.assertAlmostEqual(vec[0] / sum(vec), BW[0] / sum(BW), places=2)

    def test_uniform_capacities_and_bandwidth_give_a_flat_vector(self):
        """Equal cards must not be pulled apart. Exact equality is not
        expressible on the 64-unit owner grid for 3 ranks (the capacity path
        has the same property), so the invariant is flatness to one unit."""
        vec, _, _ = cp_token_speed_vector([1000, 1000, 1000], [1, 1, 1], 50.0)
        self.assertLessEqual(max(vec) - min(vec), 1)

    def test_deterministic(self):
        """Every rank must derive the identical vector from the identical
        all-gathered capacities -- the phase-2 install invariant."""
        for loose in (0.0, 20.0, 55.0):
            a = cp_token_speed_vector(CAPS_5090_POOR, BW, loose, hard_cap=98328)
            b = cp_token_speed_vector(CAPS_5090_POOR, BW, loose, hard_cap=98328)
            self.assertEqual(a, b)

    def test_vector_is_gcd_reduced_and_positive(self):
        for loose in (0.0, 33.0, 90.0):
            vec, _, _ = cp_token_speed_vector(CAPS_5090_POOR, BW, loose)
            self.assertTrue(all(v > 0 for v in vec))
            import math

            self.assertEqual(math.gcd(*vec), 1)

    def test_context_budget_matches_the_owner_rule(self):
        self.assertEqual(
            cp_token_context_budget([2, 3, 3], [211241, 316072, 316072]),
            min(211241 // 2, 316072 // 3, 316072 // 3) * 8,
        )


if __name__ == "__main__":
    unittest.main()
