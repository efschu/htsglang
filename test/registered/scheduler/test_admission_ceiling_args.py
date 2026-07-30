"""Hermetic tests for the --max-running-requests-ceiling argument wiring
(#287): the ceiling takes over the dimensioning field, the user's
--max-running-requests becomes the float's start, and nothing changes when
the flag is absent.
"""

import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRT = _REPO_ROOT / "python" / "sglang" / "srt"


def _args(**kw) -> ServerArgs:
    return ServerArgs(model_path="dummy", **kw)


class TestBackwardCompatibility(unittest.TestCase):
    def test_no_ceiling_leaves_max_running_requests_alone(self):
        sa = _args(max_running_requests=17)
        self.assertEqual(sa.max_running_requests, 17)
        self.assertIsNone(sa.max_running_requests_start)
        self.assertIsNone(sa.max_running_requests_ceiling)
        self.assertTrue(sa.max_running_requests_user_set)

    def test_no_ceiling_and_no_start_stays_unset(self):
        sa = _args()
        self.assertIsNone(sa.max_running_requests)
        self.assertIsNone(sa.max_running_requests_start)
        self.assertFalse(sa.max_running_requests_user_set)

    def test_tuning_knobs_are_inert_without_a_ceiling(self):
        # Nonsense marks must not fail a boot that never arms the controller.
        sa = _args(admission_throttle_high=0.1, admission_release_low=0.9)
        self.assertIsNone(sa.max_running_requests_ceiling)
        self.assertIsNone(sa.max_running_requests_start)


class TestCeilingRewrite(unittest.TestCase):
    def test_ceiling_becomes_the_dimensioning_value(self):
        sa = _args(max_running_requests=8, max_running_requests_ceiling=64)
        self.assertEqual(sa.max_running_requests, 64)
        self.assertEqual(sa.max_running_requests_start, 8)
        self.assertTrue(sa.max_running_requests_user_set)

    def test_ceiling_without_a_start_floats_from_the_top(self):
        sa = _args(max_running_requests_ceiling=64)
        self.assertEqual(sa.max_running_requests, 64)
        self.assertIsNone(sa.max_running_requests_start)
        self.assertTrue(sa.max_running_requests_user_set)

    def test_start_above_ceiling_is_a_hard_error(self):
        with self.assertRaises(ValueError) as ctx:
            _args(max_running_requests=128, max_running_requests_ceiling=64)
        self.assertIn("exceeds", str(ctx.exception))

    def test_non_positive_ceiling_is_a_hard_error(self):
        with self.assertRaises(ValueError):
            _args(max_running_requests_ceiling=0)

    def test_floor_above_ceiling_is_a_hard_error(self):
        with self.assertRaises(ValueError):
            _args(max_running_requests_ceiling=4, admission_floor=8)

    def test_release_mark_must_sit_below_the_throttle_mark(self):
        with self.assertRaises(ValueError):
            _args(
                max_running_requests_ceiling=64,
                admission_throttle_high=0.8,
                admission_release_low=0.8,
            )

    def test_throttle_mark_must_be_a_fraction(self):
        with self.assertRaises(ValueError):
            _args(max_running_requests_ceiling=64, admission_throttle_high=1.5)

    def test_hysteresis_must_be_positive(self):
        with self.assertRaises(ValueError):
            _args(max_running_requests_ceiling=64, admission_release_hysteresis=0)

    def test_cli_round_trip(self):
        import argparse

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        parsed = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--max-running-requests",
                "8",
                "--max-running-requests-ceiling",
                "48",
                "--admission-floor",
                "2",
            ]
        )
        sa = ServerArgs.from_cli_args(parsed)
        self.assertEqual(sa.max_running_requests, 48)
        self.assertEqual(sa.max_running_requests_start, 8)
        self.assertEqual(sa.admission_floor, 2)


class TestPoolDimensioning(unittest.TestCase):
    """The rewrite must actually reach the pool resolver -- that is the whole
    reason it is done in ServerArgs rather than at each sizing site."""

    @staticmethod
    def _resolve(sa, token_capacity=100_000, dp_size=1):
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            ModelRunnerKVCacheMixin,
        )

        stub = SimpleNamespace(
            server_args=sa,
            dp_size=dp_size,
            model_config=SimpleNamespace(context_len=4096),
            mambaish_config=None,
        )
        return ModelRunnerKVCacheMixin._resolve_max_num_reqs(stub, token_capacity)

    def test_pools_are_sized_for_the_ceiling_not_the_start(self):
        sa = _args(max_running_requests=8, max_running_requests_ceiling=64)
        self.assertEqual(self._resolve(sa), 64)

    def test_without_a_ceiling_pools_are_sized_for_the_start(self):
        sa = _args(max_running_requests=8)
        self.assertEqual(self._resolve(sa), 8)

    def test_dp_division_applies_to_the_ceiling(self):
        sa = _args(max_running_requests=8, max_running_requests_ceiling=64)
        self.assertEqual(self._resolve(sa, dp_size=4), 16)

    def test_capture_set_is_clamped_to_the_request_pool(self):
        # req_to_token_pool.size comes from the resolver above, so binding the
        # capture list to the pool size binds it to the ceiling. Checked in
        # source because the runner needs an initialized parallel context.
        src = (
            _SRT / "model_executor" / "runner" / "base_cuda_graph_runner.py"
        ).read_text()
        block = src[src.index("def get_batch_sizes_to_capture(") :]
        self.assertIn("num_max_requests = model_runner.req_to_token_pool.size", block)
        self.assertIn(
            "capture_bs = [bs for bs in capture_bs if bs <= num_max_requests]", block
        )

    def test_capture_bound_is_widened_to_the_ceiling(self):
        cfg = SimpleNamespace(max_bs=64, bs=None)
        sa = _args(max_running_requests=8, max_running_requests_ceiling=256)
        sa._widen_decode_capture_to_session_ceiling(cfg)
        self.assertEqual(cfg.max_bs, 256)

    def test_capture_bound_widening_respects_dp(self):
        cfg = SimpleNamespace(max_bs=8, bs=None)
        sa = _args(max_running_requests_ceiling=256, dp_size=4)
        sa.enable_dp_attention = True
        sa._widen_decode_capture_to_session_ceiling(cfg)
        self.assertEqual(cfg.max_bs, 64)

    def test_capture_bound_untouched_without_a_ceiling(self):
        cfg = SimpleNamespace(max_bs=64, bs=None)
        sa = _args(max_running_requests=512)
        sa._widen_decode_capture_to_session_ceiling(cfg)
        self.assertEqual(cfg.max_bs, 64)

    def test_capture_bound_untouched_when_pinned(self):
        cfg = SimpleNamespace(max_bs=64, bs=None)
        sa = _args(max_running_requests_ceiling=256)
        sa.cuda_graph_max_bs_decode = 64
        sa._widen_decode_capture_to_session_ceiling(cfg)
        self.assertEqual(cfg.max_bs, 64)

    def test_capture_bound_never_lowered(self):
        cfg = SimpleNamespace(max_bs=512, bs=None)
        sa = _args(max_running_requests_ceiling=256)
        sa._widen_decode_capture_to_session_ceiling(cfg)
        self.assertEqual(cfg.max_bs, 512)

    def test_pool_constructors_read_the_resolved_value(self):
        src = (_SRT / "model_executor" / "model_runner_kv_cache_mixin.py").read_text()
        self.assertIn("max_num_reqs = self.max_running_requests", src)
        self.assertIn("self.max_running_requests = config.max_running_requests", src)


class TestSchedulerWiring(unittest.TestCase):
    """Source-level checks on the admission call sites. Constructing a real
    Scheduler needs a model and a device; these keep the wiring honest
    without one."""

    def setUp(self):
        self.src = (_SRT / "managers" / "scheduler.py").read_text()

    def test_allocatable_reqs_consults_the_limiter(self):
        block = self.src[self.src.index("    def get_num_allocatable_reqs(") :]
        block = block[: block.index("\n    def ", 1)]
        self.assertIn("self.admission_limiter.current", block)
        self.assertIn("pp_max_micro_batch_size", block)

    def test_prefill_adder_gets_the_floating_limit(self):
        self.assertIn("max_running_requests=self.admission_limiter.current,", self.src)

    def test_internal_state_reports_the_floating_limit(self):
        self.assertIn(
            'ret["effective_max_running_requests_per_dp"] = '
            "self.admission_limiter.current",
            self.src,
        )
        self.assertIn('ret["admission_limiter"]', self.src)

    def test_load_snapshot_reports_the_effective_limit(self):
        from sglang.srt.managers.admission_limiter import (
            AdmissionLimiter,
            admission_limiter_scope,
        )
        from sglang.srt.managers.scheduler_components.load_inquirer import (
            SchedulerLoadInquirer,
        )

        stub = SimpleNamespace(max_running_requests=64)
        fn = SchedulerLoadInquirer.effective_max_running_requests
        # No limiter published -> the stock figure.
        self.assertEqual(fn(stub), 64)
        with admission_limiter_scope(AdmissionLimiter(64, 12, auto=True)):
            self.assertEqual(fn(stub), 12)
        # A lane's larger limiter can never widen this worker's report.
        with admission_limiter_scope(AdmissionLimiter(128, 128, auto=True)):
            self.assertEqual(fn(stub), 64)

    def test_runtime_setter_is_whitelisted(self):
        block = self.src[self.src.index("    def set_internal_state(") :]
        block = block[: block.index("\n    def ", 1)]
        self.assertIn('"effective_max_running_requests",', block)
        self.assertIn("self.admission_limiter.set_limit(v)", block)
        # And it must not leak into the generic server-args override.
        self.assertIn('remaining.pop("effective_max_running_requests", None)', block)


if __name__ == "__main__":
    unittest.main()
