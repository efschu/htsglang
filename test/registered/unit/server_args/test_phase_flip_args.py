# SPDX-License-Identifier: Apache-2.0
"""#631 phase-flip server args (--enable-phase-flip,
--phase-flip-tp-vector): CPU-only argument-time validation tests.

Every refusal DESIGN_631 section 5.1 promises at argument time is proven
refusable here; the green shape is the Route A boot (pp_size=3, tp=1,
vector length 3)."""

import unittest

from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def make_args(**kwargs):
    """model_path='dummy' short-circuits __post_init__ so the handler can
    be exercised in isolation (the uneven-TP args test pattern)."""
    return ServerArgs(model_path="dummy", **kwargs)


class TestPhaseFlipArgs(CustomTestCase):
    def test_default_off_is_silent(self):
        args = make_args()
        self.assertFalse(args.enable_phase_flip)
        args._handle_phase_flip()  # no raise

    def test_green_route_a_shape(self):
        args = make_args(
            enable_phase_flip=True,
            phase_flip_tp_vector="30,17,17",
            pp_size=3,
        )
        args._handle_phase_flip()  # no raise

    def test_vector_without_enable_refused(self):
        args = make_args(phase_flip_tp_vector="30,17,17", pp_size=3)
        with self.assertRaisesRegex(ValueError, "requires --enable-phase-flip"):
            args._handle_phase_flip()

    def test_enable_without_vector_refused(self):
        args = make_args(enable_phase_flip=True, pp_size=3)
        with self.assertRaisesRegex(ValueError, "requires --phase-flip-tp-vector"):
            args._handle_phase_flip()

    def test_malformed_and_nonpositive_vectors_refused(self):
        for vec, pat in (("30,x,17", "integer vector"), ("30,0,17", ">= 1")):
            args = make_args(
                enable_phase_flip=True, phase_flip_tp_vector=vec, pp_size=3
            )
            with self.assertRaisesRegex(ValueError, pat):
                args._handle_phase_flip()

    def test_length_must_match_pp_size(self):
        args = make_args(
            enable_phase_flip=True, phase_flip_tp_vector="30,17", pp_size=3
        )
        with self.assertRaisesRegex(ValueError, "counts must match"):
            args._handle_phase_flip()

    def test_pp1_and_tp_gt1_refused(self):
        args = make_args(
            enable_phase_flip=True, phase_flip_tp_vector="30", pp_size=1
        )
        with self.assertRaisesRegex(ValueError, "pp_size > 1"):
            args._handle_phase_flip()
        args = make_args(
            enable_phase_flip=True,
            phase_flip_tp_vector="30,17,17",
            pp_size=3,
            tp_size=2,
        )
        with self.assertRaisesRegex(ValueError, "tp_size"):
            args._handle_phase_flip()

    def test_v1_blockers_named(self):
        base = dict(
            enable_phase_flip=True, phase_flip_tp_vector="30,17,17", pp_size=3
        )
        for extra, pat in (
            ({"enable_hierarchical_cache": True}, "#630"),
            ({"speculative_algorithm": "NEXTN"}, "follow-up"),
            ({"dp_size": 2}, "dp-size"),
            ({"disaggregation_mode": "prefill"}, "disaggregation"),
        ):
            args = make_args(**base, **extra)
            with self.assertRaisesRegex(ValueError, pat):
                args._handle_phase_flip()


if __name__ == "__main__":
    unittest.main()
