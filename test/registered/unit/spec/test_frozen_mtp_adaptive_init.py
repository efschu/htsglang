"""Init-time guard tests for adaptive speculative decoding on frozen-KV MTP.

Regression for the DoA-crash class: the generic ``DEFAULT_ADAPTIVE_CONFIG``
contains step 0 (nospec) in its bs>=8 slots, but frozen MTP hard-rejects any
candidate step < 1 (the frozen seed / draft-extend path has no no-draft
branch). Enabling ``--speculative-adaptive`` on frozen MTP without an explicit
config must therefore resolve to the frozen-MTP default (all steps >= 1) and
construct cleanly, instead of raising ValueError at worker init.
"""

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.speculative.adaptive_runtime_state import AdaptiveController
from sglang.srt.speculative.adaptive_spec_params import (
    DEFAULT_ADAPTIVE_CONFIG,
    FROZEN_MTP_DEFAULT_ADAPTIVE_CONFIG,
    default_adaptive_config_for,
    resolve_candidate_steps_from_config,
)
from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import FrozenKVMTPWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _frozen_server_args(cfg_path=None):
    return SimpleNamespace(
        speculative_algorithm="FROZEN_KV_MTP",
        speculative_adaptive=True,
        speculative_adaptive_config=cfg_path,
    )


def _assert_supported(server_args):
    """Run the real init-time guard on a detached worker instance."""
    worker = object.__new__(FrozenKVMTPWorkerV2)
    with patch(
        "sglang.srt.speculative.frozen_kv_mtp_worker_v2.adaptive_unsupported_reason",
        return_value=None,
    ):
        FrozenKVMTPWorkerV2._assert_adaptive_supported(worker, server_args)


class TestFrozenMtpDefaultConfig(CustomTestCase):
    def test_default_config_selection_by_algorithm(self):
        self.assertIs(
            default_adaptive_config_for("FROZEN_KV_MTP"),
            FROZEN_MTP_DEFAULT_ADAPTIVE_CONFIG,
        )
        self.assertIs(default_adaptive_config_for("EAGLE"), DEFAULT_ADAPTIVE_CONFIG)
        self.assertIs(default_adaptive_config_for(None), DEFAULT_ADAPTIVE_CONFIG)

    def test_frozen_default_has_no_step_below_one(self):
        steps = resolve_candidate_steps_from_config(algorithm="FROZEN_KV_MTP")
        self.assertEqual(steps, [1, 2, 3])
        self.assertTrue(all(s >= 1 for s in steps))

    def test_generic_default_still_contains_step_zero(self):
        # The frozen-specific default exists BECAUSE the generic one has step 0;
        # if this ever changes, revisit FROZEN_MTP_DEFAULT_ADAPTIVE_CONFIG.
        self.assertIn(0, resolve_candidate_steps_from_config(algorithm="EAGLE"))


class TestFrozenMtpAdaptiveInitGuard(CustomTestCase):
    def test_default_config_passes_init_guard(self):
        # DoA regression: default config (no --speculative-adaptive-config)
        # must NOT raise for frozen MTP.
        _assert_supported(_frozen_server_args(cfg_path=None))

    def test_explicit_step_zero_config_still_rejected(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump({"1": {"candidate_steps": [0, 1, 3]}}, f)
            f.flush()
            with self.assertRaisesRegex(ValueError, "candidate steps >= 1"):
                _assert_supported(_frozen_server_args(cfg_path=f.name))

    def test_generic_default_would_be_rejected(self):
        # Sanity: the guard is load-bearing — feeding the generic default
        # (union {0,1,3,...}) through the frozen guard must raise.
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(DEFAULT_ADAPTIVE_CONFIG, f)
            f.flush()
            with self.assertRaisesRegex(ValueError, "candidate steps >= 1"):
                _assert_supported(_frozen_server_args(cfg_path=f.name))


class TestFrozenMtpControllerInit(CustomTestCase):
    def test_controller_constructs_with_frozen_default(self):
        # The controller must build its BS slots from the frozen default when
        # no config path is given (init-time construction, no GPU needed).
        worker = SimpleNamespace(speculative_num_steps=3)
        controller = AdaptiveController(
            worker, config_path=None, algorithm="FROZEN_KV_MTP"
        )
        self.assertEqual(controller.candidate_steps, [1, 2, 3])
        self.assertTrue(all(s >= 1 for s in controller.candidate_steps))
        # The launch value (3) is a member of the bs1 slot, so it is kept.
        self.assertEqual(controller.params.get_steps_for_batch(1), 3)


if __name__ == "__main__":
    unittest.main()
