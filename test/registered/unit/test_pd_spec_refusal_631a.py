# SPDX-License-Identifier: Apache-2.0
"""PD disaggregation refuses speculation instead of dropping it (#631a).

Before #631a, launching either PD arm with ``--speculative-algorithm NEXTN``
logged a warning and set ``speculative_algorithm = None``. The server then
came up, answered correctly, and merely decoded slower. That is the worst
shape a defect can take on this rig: the decode optimum the whole PD split
exists to protect was gone, and nothing downstream could distinguish it from
a slow card. No smoke test catches "correct but 3x slower".

So the default is now a refusal that names the arm and the reason, and the
old behaviour is reachable only by asking for it explicitly through
``SGLANG_PD_AUTO_DISABLE_SPEC=1``.

The tests drive ``handle_pd_disaggregation`` with a stub rather than a real
``ServerArgs``: constructing ``ServerArgs`` requires an accelerator (it raises
"No accelerator ... is available" under ``CUDA_VISIBLE_DEVICES=99``), and a
hermetic desk test must not need a GPU to check an argument-validation rule.
The stub carries exactly the attributes the hook touches.
"""

import unittest
from dataclasses import dataclass
from typing import Optional
from unittest import mock

from sglang.srt.arg_groups.pd_disaggregation_hook import handle_pd_disaggregation


@dataclass
class _StubArgs:
    """Only the fields ``handle_pd_disaggregation`` reads or writes."""

    disaggregation_mode: str = "null"
    speculative_algorithm: Optional[str] = None
    speculative_draft_model_path: Optional[str] = None
    disaggregation_transfer_backend: str = "mooncake"
    disaggregation_ib_device: Optional[str] = None
    disaggregation_decode_enable_radix_cache: bool = False
    disaggregation_decode_extra_slots: Optional[int] = None
    disaggregation_topology: Optional[str] = None
    disable_radix_cache: bool = False
    enable_hisparse: bool = False
    max_running_requests: Optional[int] = 4
    dp_size: int = 1


class PdSpecRefusalTest(unittest.TestCase):
    def test_decode_arm_with_spec_is_refused(self):
        args = _StubArgs(disaggregation_mode="decode", speculative_algorithm="NEXTN")
        with self.assertRaises(ValueError) as ctx:
            handle_pd_disaggregation(args)
        msg = str(ctx.exception)
        # The refusal has to be actionable, so pin what it must name.
        self.assertIn("decode", msg, "refusal does not name which arm")
        self.assertIn("NEXTN", msg, "refusal does not name the algorithm asked for")
        self.assertIn("uneven-head-sharded", msg, "refusal does not give the reason")
        self.assertIn(
            "SGLANG_PD_AUTO_DISABLE_SPEC",
            msg,
            "refusal does not name the escape hatch",
        )

    def test_prefill_arm_with_spec_is_refused(self):
        args = _StubArgs(disaggregation_mode="prefill", speculative_algorithm="EAGLE")
        with self.assertRaises(ValueError) as ctx:
            handle_pd_disaggregation(args)
        self.assertIn("prefill", str(ctx.exception))

    def test_spec_survives_the_hook_when_not_disaggregated(self):
        """The refusal must not leak into monolithic servers.

        This is the regression that matters for production: the standing
        serving boot is a monolithic NEXTN server and must be untouched.
        """
        args = _StubArgs(disaggregation_mode="null", speculative_algorithm="NEXTN")
        handle_pd_disaggregation(args)
        self.assertEqual(args.speculative_algorithm, "NEXTN")

    def test_pd_arm_without_spec_is_unaffected(self):
        args = _StubArgs(disaggregation_mode="decode", speculative_algorithm=None)
        handle_pd_disaggregation(args)
        self.assertIsNone(args.speculative_algorithm)

    def test_escape_hatch_restores_the_auto_disable(self):
        """Opt-in returns the OLD behaviour exactly: disabled, not refused."""
        args = _StubArgs(
            disaggregation_mode="decode",
            speculative_algorithm="NEXTN",
            speculative_draft_model_path="/some/draft",
        )
        with mock.patch(
            "sglang.srt.environ.envs.SGLANG_PD_AUTO_DISABLE_SPEC.get",
            return_value=True,
        ):
            handle_pd_disaggregation(args)
        self.assertIsNone(args.speculative_algorithm)
        self.assertIsNone(args.speculative_draft_model_path)

    def test_escape_hatch_is_off_by_default(self):
        """The knob's default decides whether the fix is real. Pin it."""
        from sglang.srt.environ import envs

        self.assertFalse(envs.SGLANG_PD_AUTO_DISABLE_SPEC.get())


if __name__ == "__main__":
    unittest.main()
