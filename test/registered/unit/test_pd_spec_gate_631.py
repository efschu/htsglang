# SPDX-License-Identifier: Apache-2.0
"""Speculation on a PD arm: refused where it cannot work, admitted where it can.

Two rulings live here, and the second one narrows the first.

#631a made the old silent auto-disable a REFUSAL. That part stands and is
still pinned below. Launching a PD arm with ``--speculative-algorithm NEXTN``
used to log a warning, set ``speculative_algorithm = None``, come up, answer
correctly and merely decode slower -- the worst shape a defect can take on
this rig, because the decode optimum the PD split exists to protect was gone
and nothing downstream could tell it apart from a slow card.

#631b keeps that refusal but stops it being a BLANKET. The reason #631a gave
was specific -- "the MTP/EAGLE draft KV pool is uneven-head-sharded (not DCP
token-sharded)" -- and that is a statement about one layout, not about all of
them. The draft pool rides the main transfer as extra layers addressed by the
SAME index array as the target pool, so the only question is whether draft
rows and target rows share a coordinate system on both arms. Two shapes say
yes and are now admitted: ``tp_size == 1`` (nothing is sharded) and
``dcp_size == tp_size > 1`` (the token-sharded path, where with
``--draft-kv-layout dcp`` the draft pool takes the target's own compact
owner-rule rows).

The decision also MOVED, and the move is the substance rather than a tidy-up:
it now runs after ``_handle_uneven_tp`` resolves ``dcp_size``, because in
``handle_pd_disaggregation`` ``dcp_size`` is still 1 for exactly the
token-sharded configuration that is admitted. A gate that reads an unresolved
field can only ever answer with a blanket. ``test_the_hook_no_longer_decides``
pins that the early hook has stopped deciding, so the two cannot drift back
into both being able to turn speculation off.

The tests drive the gates with a stub carrying exactly the attributes they
touch, for the reason the #631a file already recorded: the rule is a pure
function of a few fields, and a real model path resolves and dies on
"No accelerator ... is available" at a desk.
"""

import unittest
from dataclasses import dataclass
from typing import Optional
from unittest import mock

from sglang.srt.arg_groups.pd_disaggregation_hook import (
    handle_pd_disaggregation,
    validate_pd_speculation,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@dataclass
class _StubArgs:
    """Only the fields the two gates read or write."""

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
    # Resolved by _handle_uneven_tp BEFORE this gate runs.
    tp_size: int = 2
    dcp_size: int = 1
    draft_kv_layout: str = "replicated"


class PdSpecRefusalTest(CustomTestCase):
    """What #631a established and #631b did NOT loosen."""

    def test_head_sharded_decode_arm_is_refused(self):
        args = _StubArgs(
            disaggregation_mode="decode", speculative_algorithm="NEXTN", tp_size=2
        )
        with self.assertRaises(ValueError) as ctx:
            validate_pd_speculation(args)
        msg = str(ctx.exception)
        # The refusal has to be actionable, so pin what it must name.
        self.assertIn("decode", msg, "refusal does not name which arm")
        self.assertIn("NEXTN", msg, "refusal does not name the algorithm asked for")
        self.assertIn("head-sharded", msg, "refusal does not give the reason")
        self.assertIn(
            "SGLANG_PD_AUTO_DISABLE_SPEC", msg, "refusal does not name the escape hatch"
        )

    def test_head_sharded_prefill_arm_is_refused(self):
        args = _StubArgs(
            disaggregation_mode="prefill", speculative_algorithm="EAGLE", tp_size=2
        )
        with self.assertRaises(ValueError) as ctx:
            validate_pd_speculation(args)
        self.assertIn("prefill", str(ctx.exception))

    def test_refusal_names_both_supported_shapes(self):
        """A refusal that does not say what WOULD work sends the operator
        back to the source. Both admitted shapes must be in the text."""
        args = _StubArgs(
            disaggregation_mode="decode", speculative_algorithm="NEXTN", tp_size=4
        )
        with self.assertRaises(ValueError) as ctx:
            validate_pd_speculation(args)
        msg = str(ctx.exception)
        self.assertIn("tp_size == 1", msg)
        self.assertIn("--draft-kv-layout dcp", msg)

    def test_spec_survives_when_not_disaggregated(self):
        """The refusal must not leak into monolithic servers.

        This is the regression that matters for production: the standing
        serving boot is a monolithic NEXTN server and must be untouched.
        """
        args = _StubArgs(disaggregation_mode="null", speculative_algorithm="NEXTN")
        validate_pd_speculation(args)
        self.assertEqual(args.speculative_algorithm, "NEXTN")

    def test_pd_arm_without_spec_is_unaffected(self):
        args = _StubArgs(disaggregation_mode="decode", speculative_algorithm=None)
        validate_pd_speculation(args)
        self.assertIsNone(args.speculative_algorithm)

    def test_escape_hatch_restores_the_auto_disable(self):
        """Opt-in returns the OLD behaviour exactly: disabled, not refused."""
        args = _StubArgs(
            disaggregation_mode="decode",
            speculative_algorithm="NEXTN",
            speculative_draft_model_path="/some/draft",
            tp_size=2,
        )
        with mock.patch(
            "sglang.srt.environ.envs.SGLANG_PD_AUTO_DISABLE_SPEC.get",
            return_value=True,
        ):
            validate_pd_speculation(args)
        self.assertIsNone(args.speculative_algorithm)
        self.assertIsNone(args.speculative_draft_model_path)

    def test_escape_hatch_is_off_by_default(self):
        """The knob's default decides whether the fix is real. Pin it."""
        from sglang.srt.environ import envs

        self.assertFalse(envs.SGLANG_PD_AUTO_DISABLE_SPEC.get())


class PdSpecAdmissionTest(CustomTestCase):
    """What #631b admits, and why each shape is congruent."""

    def test_unsharded_arm_is_admitted(self):
        """tp_size == 1: draft rows and target rows are both global token
        ids, so the shared index array needs no reslicing. This shape was
        swept up by the blanket purely because the blanket did not look."""
        for mode in ("prefill", "decode"):
            with self.subTest(mode=mode):
                args = _StubArgs(
                    disaggregation_mode=mode,
                    speculative_algorithm="NEXTN",
                    tp_size=1,
                    dcp_size=1,
                )
                validate_pd_speculation(args)
                self.assertEqual(args.speculative_algorithm, "NEXTN")

    def test_token_sharded_arm_is_admitted(self):
        """dcp_size == tp_size > 1: the draft pool takes the target's own
        compact owner-rule rows, so the shared indices are correct by
        construction."""
        args = _StubArgs(
            disaggregation_mode="decode",
            speculative_algorithm="NEXTN",
            tp_size=2,
            dcp_size=2,
            draft_kv_layout="dcp",
        )
        validate_pd_speculation(args)
        self.assertEqual(args.speculative_algorithm, "NEXTN")

    def test_token_sharded_layout_question_belongs_to_the_642_gate(self):
        """A token-sharded arm with the WRONG draft layout must not be
        refused here. #642 refuses it by name with the addressing argument
        spelled out; duplicating that decision would mean one hazard with
        two texts that can drift apart."""
        args = _StubArgs(
            disaggregation_mode="decode",
            speculative_algorithm="NEXTN",
            tp_size=2,
            dcp_size=2,
            draft_kv_layout="replicated",
        )
        validate_pd_speculation(args)  # admitted here...
        from sglang.srt.arg_groups.pd_disaggregation_hook import (
            validate_pd_draft_kv_layout,
        )

        with self.assertRaises(ValueError) as ctx:  # ...and refused there
            validate_pd_draft_kv_layout(args)
        self.assertIn("--draft-kv-layout", str(ctx.exception))

    def test_partial_dcp_is_not_the_token_sharded_shape(self):
        """dcp_size > 1 but != tp_size is neither congruent shape. The
        equality is what makes every rank's draft rows compact; a partial
        split leaves some ranks head-sharded."""
        args = _StubArgs(
            disaggregation_mode="decode",
            speculative_algorithm="NEXTN",
            tp_size=4,
            dcp_size=2,
            draft_kv_layout="dcp",
        )
        with self.assertRaises(ValueError):
            validate_pd_speculation(args)


class PdSpecGatePlacementTest(CustomTestCase):
    def test_the_hook_no_longer_decides(self):
        """The early hook must neither refuse nor auto-disable.

        It runs before dcp_size is resolved, so anything it decided about
        speculation could only be a blanket. Pinning this keeps the refusal
        and the escape hatch from drifting back into two places -- which is
        exactly how a hardened escape hatch can end up quietly holding a
        corruption path open on the other branch.
        """
        args = _StubArgs(
            disaggregation_mode="decode",
            speculative_algorithm="NEXTN",
            speculative_draft_model_path="/some/draft",
            tp_size=2,
        )
        handle_pd_disaggregation(args)
        self.assertEqual(args.speculative_algorithm, "NEXTN")
        self.assertEqual(args.speculative_draft_model_path, "/some/draft")

    def test_hook_still_decides_the_escape_hatch_nowhere(self):
        """Even with the hatch on, the HOOK stays out of it."""
        args = _StubArgs(
            disaggregation_mode="decode",
            speculative_algorithm="NEXTN",
            tp_size=2,
        )
        with mock.patch(
            "sglang.srt.environ.envs.SGLANG_PD_AUTO_DISABLE_SPEC.get",
            return_value=True,
        ):
            handle_pd_disaggregation(args)
        self.assertEqual(args.speculative_algorithm, "NEXTN")


if __name__ == "__main__":
    unittest.main()
