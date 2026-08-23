"""#817 -- the #677 hold may veto one arm, and it must not guess which.

THE DEFECT. `decide()` wrapped every PP_TO_TP arm in the #677 layout hold
unless the reason STRING started with IDLE_LOCKED or contained "DRAINED" or
"decode starvation cap". A denylist of substrings cannot tell an UNRECOGNISED
arm from an exempt one, and it fails toward swallowing. The blocked-admission
exit -- added by 3f49f51c51 one day before the wrapper, in the same ticket --
was unrecognised, so a legitimate exit became a hold capped at 8 rounds.

WHY THAT PARTICULAR SWALLOW IS THE WEDGE FORM. 3f49f51c51 exists because of a
live wedge: 403779 tokens of prefill frozen, every slot held by a carried
decode, PP unable to decode under strict purity, so no slot could free and no
chunk could land. Its commit spends a paragraph establishing that the missing
exit is NOT another timer -- that is the whole point of the rule. The hold it
was converted into then announced, over the specimen's own frozen token count,
"the timer does not get to take it away". The lever vetoed an exit by calling
it something its author had explicitly ruled out.

THE INVERSION, as the wrapper's own comment prescribed ("If a fourth exemption
ever appears, invert this into an allowlist rather than adding it"):
eligibility is a property of the ARM (`PhasePolicyDecision.hold_eligible`), set
where the arm is built, defaulting to False. An arm nobody marked is an EXIT.
The swallow becomes structurally impossible rather than string-dependent.

STILL OPEN, NOT DECIDED HERE: whether the LEGACY STOPWATCH arm should be
hold-eligible at all. It is the only member of the allowlist today, and three
pre-existing tests demand it exit. See the #817 COORD -- this file pins the
part that was decided, and pins the stopwatch's CURRENT membership so that
changing it is a deliberate act with a failing test, not a silent drift.
"""

import unittest

from sglang.srt.managers import phase_policy as pp
from sglang.srt.managers.phase_policy import PP_TO_TP, PhasePolicyDecision
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestTheDefaultIsExit(CustomTestCase):
    """The structural half: forgetting to mark an arm cannot swallow it."""

    def test_an_unmarked_arm_is_not_hold_eligible(self):
        d = PhasePolicyDecision(PP_TO_TP, "some future arm nobody classified")
        self.assertFalse(d.hold_eligible)

    def test_eligibility_must_be_claimed_explicitly(self):
        d = PhasePolicyDecision(PP_TO_TP, "a timer arm", hold_eligible=True)
        self.assertTrue(d.hold_eligible)

    def test_the_wrapper_reads_the_flag_and_not_the_reason(self):
        # THE DANGER DIRECTION. The reason string carries every substring the
        # old denylist keyed on, and none of them may matter any more. If the
        # wrapper ever goes back to sniffing text, this fails.
        import inspect

        src = inspect.getsource(pp.decide)
        self.assertIn("d.hold_eligible", src)
        for sniffed in ('"DRAINED" not in', '"decode starvation cap" not in'):
            self.assertNotIn(sniffed, src)


class TestTheBlockedExitIsNeverHeld(CustomTestCase):
    """The decided half, and the falsification the escalation asked for."""

    def _blocked_arm(self):
        """The arm as `_decide_from_load` builds it, reason text and all."""
        return PhasePolicyDecision(
            PP_TO_TP,
            "blocked admission: pending frozen at 403779 tok for 30.0s with "
            "bs 4 carried (no chunk admitted in 12.0s) -- exit condition: "
            "admission blocked",
        )

    def test_the_blocked_arm_is_not_eligible(self):
        self.assertFalse(self._blocked_arm().hold_eligible)

    def test_the_old_denylist_would_have_swallowed_it(self):
        # CONFIRMATION, not assumption: the escalation called the link to the
        # wedge family SUSPECT and asked for it to be settled either way. This
        # replays the old membership test against the real specimen reason and
        # shows it lands in the hold -- the state (pending > 0, carried
        # decodes, admission blocked) that #788/#768/#698/#748 all share.
        r = self._blocked_arm().reason
        old_denylist_would_hold = (
            not r.startswith(pp.IDLE_LOCKED)
            and "DRAINED" not in r
            and "decode starvation cap" not in r
        )
        self.assertTrue(
            old_denylist_would_hold,
            "the specimen no longer reproduces the swallow; re-derive the "
            "premise before trusting this file",
        )

    def test_the_three_old_exemptions_still_pass_through(self):
        # They pass for a better reason now -- not being named -- but they must
        # still not be held, or the inversion would have traded one swallow for
        # three.
        for reason in (
            f"{pp.IDLE_LOCKED} the layout can build nothing",
            "DRAINED: 12 tok remaining -- exit condition: drained",
            "decode stall cap: ... -- exit condition: decode starvation cap",
        ):
            self.assertFalse(PhasePolicyDecision(PP_TO_TP, reason).hold_eligible)


def _cfg(**kw):
    """The pure-drain shape from the specimen's own suite: no stopwatch, no
    decode-stall cap, so the blocked-admission exit is the ONLY way out."""
    base = dict(
        enabled=True,
        flip_tokens=1,
        flip_cost_s=3.0,
        decode_stall_slo_s=0.0,
        pp_window_s=0.0,
        pp_exit_tokens=512,
        pp_prefill_tok_s=1000.0,
    )
    base.update(kw)
    return pp.PhasePolicyConfig(**base)


def _drive(cfg, state, pending, bs, now):
    inp = pp.PhasePolicyInputs(
        phase=pp.PHASE_PP, pending_prefill_tokens=pending, running_bs=bs, now=now
    )
    pp.observe_idle(state, inp)
    return pp.decide(cfg, state, inp)


class TestTheWrapperBehavesBothWays(CustomTestCase):
    """Through the real `decide()`, in both directions.

    Written after a mutant survived: every test above was a source pin or a
    dataclass default, so flipping the wrapper's condition to `not
    d.hold_eligible` -- the exact inversion of the fix -- passed all of them.
    A membership rule that is only ever asserted about itself proves nothing
    about the machine that consumes it.
    """

    def test_the_blocked_exit_leaves_pp(self):
        # The specimen the escalation named: pending frozen, decodes carried,
        # admission blocked. Under the old wrapper this became a hold.
        cfg = _cfg()
        state = pp.PhasePolicyState()
        window = pp.pp_progress_stall_window_s(cfg)
        self.assertGreater(window, 0)
        frozen = 403779
        self.assertIsNone(_drive(cfg, state, frozen, 4, 0.0).direction)
        out = _drive(cfg, state, frozen, 4, window + 1.0)
        self.assertEqual(
            pp.PP_TO_TP, out.direction, f"the blocked exit was swallowed: {out.reason}"
        )
        self.assertIn("blocked admission", out.reason)

    def test_the_eligible_stopwatch_arm_is_still_held(self):
        # The other direction, and the reason the allowlist is not empty: the
        # one arm that IS eligible must really reach the hold, or the fix would
        # have deleted the lever instead of aiming it.
        cfg = _cfg(pp_window_s=15.0, flip_tokens=7004)
        state = pp.PhasePolicyState()
        # Prefill must be MOVING here, or the blocked-admission exit fires
        # first and this would test the wrong arm -- the two are distinguished
        # by progress, which is exactly the distinction the old denylist lost.
        self.assertIsNone(_drive(cfg, state, 302757, 2, 0.0).direction)
        self.assertIsNone(_drive(cfg, state, 302000, 2, 7.0).direction)
        out = _drive(cfg, state, 301000, 2, 15.0)
        self.assertIsNone(out.direction, f"the eligible arm was not held: {out.reason}")
        self.assertIn("HOLD", out.reason)


class TestTheAllowlistMembership(CustomTestCase):
    """Exactly one arm is eligible, and which one is pinned deliberately."""

    def test_only_the_legacy_stopwatch_claims_eligibility(self):
        # Enumerated from the source rather than asserted from memory: every
        # PhasePolicyDecision built with hold_eligible=True inside the rules.
        import inspect

        src = inspect.getsource(pp._decide_from_load)
        self.assertEqual(
            src.count("hold_eligible=True"),
            1,
            "the hold allowlist changed size -- that is a policy decision and "
            "needs its own reasoning, not a passing test",
        )
        # And it is the stopwatch: the marker sits above the pp-window arm.
        marker = src.index("hold_eligible=True")
        self.assertIn("HAND-SET STOPWATCH", src[marker : marker + 1200])

    def test_the_stopwatch_arm_is_the_plain_timer(self):
        # The wrapper's stated rule is that it may veto the plain timer and
        # nothing else. This pins that the eligible arm really is guarded by a
        # clock and by the ABSENCE of the starvation cap -- the fact that makes
        # its membership arguable, and the reason the COORD leaves it open.
        import inspect

        src = inspect.getsource(pp._decide_from_load)
        marker = src.index("hold_eligible=True")
        head = src[max(0, marker - 2000) : marker]
        self.assertIn("cap <= 0", head)
        self.assertIn("cfg.pp_window_s > 0", head)


if __name__ == "__main__":
    unittest.main()
