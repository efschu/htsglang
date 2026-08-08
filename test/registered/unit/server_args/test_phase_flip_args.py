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
        # NEXTN used to be on this list, refused as "the named phase-flip
        # follow-up". That follow-up shipped: speculation now runs in the
        # TP decode phase and is allowed here. See
        # TestFlipV1SpeculationBlockers for what stays refused and why.
        for extra, pat in (
            ({"enable_hierarchical_cache": True}, "#630"),
            ({"dp_size": 2}, "dp-size"),
            ({"disaggregation_mode": "prefill"}, "disaggregation"),
        ):
            args = make_args(**base, **extra)
            with self.assertRaisesRegex(ValueError, pat):
                args._handle_phase_flip()


class TestSpecInTpDecodePhaseGate(CustomTestCase):
    """#631 speculation slice: PP + spec is refused UNLESS the instance is
    a phase flip, where speculation runs in the TP decode phase only.

    The underlying incompatibility is not waived. There is no PP-shaped
    draft worker -- the constructors take no pp_rank -- so none is built
    in the PP phase; the flip builds one on its TP stack and arms it at
    cutover. What changes here is only WHERE the rule is enforced."""

    def _refusal(self, needle, **kwargs):
        """The refusal message matching `needle`, or None.

        check_server_args validates far more than the one rule under test
        and cannot run to completion against model_path='dummy', so the
        assertion is isolated by its message instead of by reaching the
        end of the method. Asserting "no refusal" therefore means "not
        THIS refusal", which is exactly the claim being made.
        """
        args = make_args(pp_size=3, disable_overlap_schedule=True, **kwargs)
        try:
            args.check_server_args()
        except AssertionError as e:
            return str(e) if needle in str(e) else None
        except Exception:
            return None
        return None

    def test_pp_plus_spec_without_the_flip_is_still_refused(self):
        msg = self._refusal("speculative", speculative_algorithm="NEXTN")
        self.assertIsNotNone(msg, "PP + spec without the flip must be refused")
        self.assertIn("phase-flip", msg.replace("phase flip", "phase-flip"))

    def test_pp_plus_spec_with_the_flip_is_allowed(self):
        self.assertIsNone(
            self._refusal(
                "speculative",
                speculative_algorithm="NEXTN",
                enable_phase_flip=True,
                phase_flip_tp_vector="30,17,17",
            ),
            "a phase-flip instance must be allowed to carry speculation",
        )

    def test_pp_without_spec_is_unchanged(self):
        self.assertIsNone(self._refusal("speculative"))

    def test_pp_plus_overlap_schedule_is_still_refused(self):
        args = make_args(pp_size=3, disable_overlap_schedule=False)
        try:
            args.check_server_args()
            hit = None
        except AssertionError as e:
            hit = str(e) if "overlap schedule" in str(e) else None
        except Exception:
            hit = None
        self.assertIsNotNone(hit, "PP + overlap schedule must stay refused")


class TestFlipV1SpeculationBlockers(CustomTestCase):
    """The flip's own V1 blocker list. Speculation used to be refused here
    outright, naming "the TP+NEXTN decode arm" as a follow-up; that arm
    exists now, so the blanket refusal is replaced by the two shapes for
    which "armed in the TP phase only" is not a complete answer."""

    def _flip(self, **kwargs):
        args = make_args(
            enable_phase_flip=True,
            phase_flip_tp_vector="30,17,17",
            pp_size=3,
            **kwargs,
        )
        args._handle_phase_flip()

    def test_nextn_is_allowed(self):
        self._flip(speculative_algorithm="NEXTN")

    def test_eagle_is_allowed(self):
        self._flip(speculative_algorithm="EAGLE")

    def test_ngram_is_refused_naming_the_corpus_manager(self):
        with self.assertRaisesRegex(ValueError, "ngram"):
            self._flip(speculative_algorithm="NGRAM")

    def test_solo_draft_placement_is_refused(self):
        with self.assertRaisesRegex(ValueError, "solo"):
            self._flip(
                speculative_algorithm="NEXTN",
                speculative_draft_placement="solo",
            )

    def test_other_v1_blockers_are_untouched(self):
        with self.assertRaisesRegex(ValueError, "dp-size"):
            self._flip(dp_size=2)

    # -- #631 automatic phase policy ------------------------------------

    def test_policy_defaults_to_manual(self):
        """The default must not change behaviour: a flip boot still flips
        only when a human says so."""
        args = make_args(
            enable_phase_flip=True,
            phase_flip_tp_vector="30,17,17",
            pp_size=3,
        )
        args._handle_phase_flip()
        self.assertEqual(args.phase_flip_policy, "manual")

    def test_policy_auto_is_accepted_on_a_flip_boot(self):
        args = make_args(
            enable_phase_flip=True,
            phase_flip_tp_vector="30,17,17",
            pp_size=3,
            phase_flip_policy="auto",
        )
        args._handle_phase_flip()
        self.assertEqual(args.phase_flip_policy, "auto")

    def test_policy_auto_without_the_flip_flag_is_refused(self):
        """Refused rather than ignored: silently accepting it would read
        as 'the policy is running' when no secondary stack exists."""
        args = make_args(phase_flip_policy="auto")
        with self.assertRaisesRegex(ValueError, "requires --enable-phase-flip"):
            args._handle_phase_flip()

    def test_policy_manual_without_the_flip_flag_is_silent(self):
        args = make_args(phase_flip_policy="manual")
        args._handle_phase_flip()  # no raise

    def test_unknown_policy_mode_is_named(self):
        args = make_args(
            enable_phase_flip=True,
            phase_flip_tp_vector="30,17,17",
            pp_size=3,
            phase_flip_policy="sometimes",
        )
        with self.assertRaisesRegex(ValueError, "not a known mode"):
            args._handle_phase_flip()


if __name__ == "__main__":
    unittest.main()
