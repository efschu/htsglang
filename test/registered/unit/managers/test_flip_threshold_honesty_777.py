"""#777 -- N says one thing and does another, in two different ways.

THE COMPLAINT: ~1.8k-token prompts route into PP prefill while the policy
itself logs a break-even of 7004 tokens. Both halves of that sentence are true,
and neither is a wiring bug -- `cfg.flip_tokens` really is the number printed
AND the number compared. The defect is the #708 shape twice over: a diagnostic
that claims something the code does not do.

  1. SCOPE. The module docstring said "Below N the prompt never wanted PP
     anyway and now prefills in TP without any flip at all" -- a per-prompt
     promise. N is compared against `pending_prefill_tokens`, an AGGREGATE over
     the waiting queue, the chunked remainder and in-flight arrivals. A 1.8k
     prompt reaches PP whenever the SUM crosses N, and once the layout rests in
     PP every prompt admitted prefills there at any size. Nothing in the module
     compares one request's length to N.

  2. PROVENANCE. The pricing comment said "from the first measurement on, the
     estimate wins". The ESTIMATOR does follow measurements; N does not.
     `config_from_env` has one caller (Scheduler.__init__), `flip_tokens` is
     assigned in one place, and the only `dataclasses.replace` on the live
     config touches `decode_contention`. So N is frozen at boot, priced off a
     seam seed that no flip has yet contradicted, for the life of the process.

WHAT IS AND IS NOT FIXED HERE. Not fixed: the value of N. Repricing it from
7004 to a measured ~49,250 moves when the server flips at all -- a policy
decision with a 7x blast radius that belongs to the planner, and one this task
was explicitly told not to take from the gut. Fixed: the silence. The first
flip that proves N stale now says so, with both numbers and their ratio, and
the two overstated comments now describe what the code actually does.
"""

import unittest

from sglang.srt.managers import phase_policy as pp
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_LOGGER = "sglang.srt.managers.phase_policy"

# The live-boot numbers this task is about, from the module's own comment and
# from test_flip_cost_calibration_677.py: a 3.2 s seed against 22.5 s flips.
_SEED_S = 3.2
_MEASURED_S = 22.5
_TP_TOK_S = 1681.0
_PP_TOK_S = 7245.5


class _EstimatorState(CustomTestCase):
    """The estimator and the pricing record are process-global."""

    def setUp(self):
        super().setUp()
        self._saved = (
            pp._FLIP_COST_ESTIMATOR,
            pp._FLIP_TOKENS_AT_BOOT,
            pp._FLIP_TOKENS_PRICING,
            pp._FLIP_TOKENS_STALE_SAID,
        )
        self.addCleanup(self._restore)

    def _restore(self):
        (
            pp._FLIP_COST_ESTIMATOR,
            pp._FLIP_TOKENS_AT_BOOT,
            pp._FLIP_TOKENS_PRICING,
            pp._FLIP_TOKENS_STALE_SAID,
        ) = self._saved

    def _arm(self, seed_s=_SEED_S, explicit=False):
        pp._FLIP_COST_ESTIMATOR = pp.FlipCostEstimator(seed_s=seed_s)
        n = pp.break_even_tokens(seed_s, _TP_TOK_S, _PP_TOK_S)
        pp.note_flip_tokens_pricing(n, _TP_TOK_S, _PP_TOK_S, explicit)
        return n


class TestTheNumbersThisIsAbout(CustomTestCase):
    """The premise, reproduced from the formula rather than asserted."""

    def test_the_seed_prices_the_logged_break_even(self):
        self.assertEqual(pp.break_even_tokens(_SEED_S, _TP_TOK_S, _PP_TOK_S), 7004)

    def test_the_measurement_prices_a_far_larger_one(self):
        n = pp.break_even_tokens(_MEASURED_S, _TP_TOK_S, _PP_TOK_S)
        self.assertGreater(n, 40000)
        # The gap the operator sees: a 1.8k prompt is far below BOTH, which is
        # why the complaint is about scope and not about the value of N.
        self.assertLess(1800, 7004)


class TestStalenessIsNamed(_EstimatorState):
    """Direction 1: once a measurement contradicts N, the gap is spoken."""

    def test_the_first_contradicting_flip_says_so(self):
        n_boot = self._arm()
        with self.assertLogs(_LOGGER, level="WARNING") as cm:
            pp.observe_flip_cost(_MEASURED_S)
        body = "\n".join(cm.output)
        self.assertIn("#777 N IS STALE", body)
        self.assertIn(str(n_boot), body)
        self.assertIn(
            str(pp.break_even_tokens(_MEASURED_S, _TP_TOK_S, _PP_TOK_S)), body
        )

    def test_it_says_so_once(self):
        self._arm()
        with self.assertLogs(_LOGGER, level="WARNING"):
            pp.observe_flip_cost(_MEASURED_S)
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            for _ in range(5):
                pp.observe_flip_cost(_MEASURED_S)

    def test_the_threshold_itself_is_not_touched(self):
        # THE LINE THIS TASK MUST NOT CROSS: naming the gap is not closing it.
        n_boot = self._arm()
        pp.observe_flip_cost(_MEASURED_S)
        self.assertEqual(pp._FLIP_TOKENS_AT_BOOT, n_boot)
        repriced = pp.repriced_flip_tokens()
        self.assertEqual(repriced[0], n_boot)
        self.assertNotEqual(repriced[1], n_boot)


class TestSilenceWhereSilenceIsRight(_EstimatorState):
    """Direction 2: the reasons to say nothing are kept apart, not collapsed."""

    def test_nothing_to_say_before_a_measurement(self):
        self._arm()
        self.assertIsNone(pp.repriced_flip_tokens())

    def test_nothing_to_say_when_the_measurement_agrees(self):
        self._arm()
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            pp.observe_flip_cost(_SEED_S)
        repriced = pp.repriced_flip_tokens()
        self.assertEqual(repriced[0], repriced[1])

    def test_an_explicitly_pinned_n_is_an_assertion_not_a_derivation(self):
        self._arm(explicit=True)
        with self.assertNoLogs(_LOGGER, level="WARNING"):
            pp.observe_flip_cost(_MEASURED_S)
        self.assertIsNone(pp.repriced_flip_tokens())

    def test_nothing_to_say_without_a_config(self):
        pp._FLIP_COST_ESTIMATOR = pp.FlipCostEstimator(seed_s=_SEED_S)
        pp._FLIP_TOKENS_AT_BOOT = None
        pp._FLIP_TOKENS_PRICING = None
        pp._FLIP_COST_ESTIMATOR.observe(_MEASURED_S)
        self.assertIsNone(pp.repriced_flip_tokens())

    def test_observing_still_feeds_the_estimator(self):
        # The warning must not have become the point of the function.
        self._arm()
        pp.observe_flip_cost(_MEASURED_S)
        self.assertTrue(pp._FLIP_COST_ESTIMATOR.calibrated)
        self.assertAlmostEqual(pp._FLIP_COST_ESTIMATOR.value(), _MEASURED_S, places=3)


class TestTheRecordingIsWired(_EstimatorState):
    """The edge itself, driven through the REAL config builder.

    Written after a mutant survived: every test above armed the pricing record
    by calling `note_flip_tokens_pricing` directly, so deleting its call site
    in `config_from_env` changed nothing they could see. That is the same
    counter-without-an-actuator shape this task is about, reproduced inside its
    own test file -- a seam exercised only through its own front door proves
    the front door, not the wiring.
    """

    def setUp(self):
        super().setUp()
        pp._FLIP_COST_ESTIMATOR = None
        pp._FLIP_TOKENS_AT_BOOT = None
        pp._FLIP_TOKENS_PRICING = None

    def test_building_a_config_records_how_n_was_priced(self):
        cfg = pp.config_from_env(enabled=True)
        self.assertEqual(pp._FLIP_TOKENS_AT_BOOT, cfg.flip_tokens)
        self.assertIsNotNone(pp._FLIP_TOKENS_PRICING)
        tp_tok_s, pp_tok_s, explicit = pp._FLIP_TOKENS_PRICING
        self.assertGreater(tp_tok_s, 0)
        self.assertGreater(pp_tok_s, tp_tok_s)
        self.assertFalse(explicit)

    def test_a_real_config_then_a_real_flip_reaches_the_warning(self):
        # End to end through both edges: build the config the way the scheduler
        # does, then feed a flip the way the seam does.
        pp.config_from_env(enabled=True)
        with self.assertLogs(_LOGGER, level="WARNING") as cm:
            pp.observe_flip_cost(_MEASURED_S)
        self.assertIn("#777 N IS STALE", "\n".join(cm.output))


class TestTheDocumentedClaimsMatchTheCode(CustomTestCase):
    """The #708 half: a diagnostic may not claim what the code does not do."""

    def test_the_per_prompt_promise_stands_only_as_a_retracted_quote(self):
        # The old sentence is kept ON PURPOSE, quoted and attributed, so a
        # reader who remembers it learns it was withdrawn rather than finding
        # it silently gone. What must not survive is the claim STANDING on its
        # own, so the quote has to carry its retraction in the same breath.
        doc = pp.__doc__ or ""
        claim = "Below N the prompt never wanted PP anyway"
        self.assertEqual(doc.count(claim), 1, "the claim appears more than once")
        head = doc.split(claim)[0]
        self.assertIn("used\nto end", head[-120:])
        self.assertIn("which is a claim about ONE prompt that the code does", doc)
        self.assertIn("aggregate", doc.lower())

    def test_the_module_names_the_aggregate_it_actually_compares(self):
        doc = pp.__doc__ or ""
        self.assertIn("pending_prefill", doc)

    def test_no_per_request_comparison_against_n_exists(self):
        # The absence the docstring now asserts, checked rather than believed.
        import inspect

        src = inspect.getsource(pp.decide)
        self.assertIn("pending_prefill_tokens", src)
        for per_request in ("origin_input_ids", "len(req.", "req.input_ids"):
            self.assertNotIn(per_request, src)


if __name__ == "__main__":
    unittest.main()
