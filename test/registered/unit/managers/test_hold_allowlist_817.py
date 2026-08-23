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

THE ADMISSION CONDITION, which is what the allowlist actually encodes:

    An arm may carry hold_eligible=True only if, in EVERY state where that arm
    fires, a SECOND INDEPENDENT anti-starvation bound is armed.

The hold's licence to veto "the plain timer/economics exit" was never about the
arm being a timer -- it rested on the unstated assumption that something else
would still stop the layout pinning. The legacy pp_window stopwatch destroys
that assumption: it is guarded by `cap <= 0`, so it fires ONLY when the
decode-starvation cap is absent, making it the LAST bound in every state it
fires in. Vetoing the last bound is an unbounded hold -- verbatim the condition
"THE STARVATION REGRESSION" exists to prevent. So it fails the condition by
construction and is an exit too.

THE ALLOWLIST IS THEREFORE EMPTY, and that is the honest state rather than an
oversight. The seam stays: it is the socket for a future arm that really is
backstopped, and with no member no arm can be held at all. #677's economics is
not lost with it -- it lives in the window-length machinery and in the
threshold repricing (#819), on the flip-DECISION side where a flip can be
weighed before it is chosen, rather than as a veto on an exit the rules have
already decided.
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
        """The REAL arm, taken from the rules rather than hand-written.

        An earlier draft typed the reason string out by hand and got its tail
        backwards ("admission blocked" for "blocked admission"). The denylist
        replay below would not have noticed -- it keys on three OTHER
        substrings -- which is exactly how a test comes to assert something
        about a string that no code produces. So the specimen is built by
        driving the rules and is therefore always the string being shipped.
        """
        cfg = _cfg()
        state = pp.PhasePolicyState()
        window = pp.pp_progress_stall_window_s(cfg)
        _drive(cfg, state, 403779, 4, 0.0)
        d = _drive(cfg, state, 403779, 4, window + 1.0)
        self.assertEqual(PP_TO_TP, d.direction, d.reason)
        self.assertIn("blocked admission", d.reason)
        return d

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

    def test_the_stopwatch_arm_also_leaves_pp(self):
        # The other direction, decided in #817: the stopwatch is the LAST
        # anti-pinning bound in every state it fires in (`cap <= 0`), so
        # vetoing it is an unbounded hold. It exits.
        cfg = _cfg(pp_window_s=15.0, flip_tokens=7004)
        state = pp.PhasePolicyState()
        # Prefill must be MOVING here, or the blocked-admission exit fires
        # first and this would test the wrong arm -- the two are distinguished
        # by progress, which is exactly the distinction the old denylist lost.
        self.assertIsNone(_drive(cfg, state, 302757, 2, 0.0).direction)
        self.assertIsNone(_drive(cfg, state, 302000, 2, 7.0).direction)
        out = _drive(cfg, state, 301000, 2, 15.0)
        self.assertEqual(
            PP_TO_TP, out.direction, f"the stopwatch was swallowed: {out.reason}"
        )
        self.assertIn("pp window", out.reason)

    def test_no_arm_can_be_held_while_the_list_is_empty(self):
        # The consequence of an empty allowlist, stated as behaviour rather
        # than inferred from the count: no reachable PP_TO_TP arm becomes a
        # HOLD. If a member is ever added, this fails and the author has to
        # come argue the admission condition.
        for cfg, series in (
            (_cfg(), [(403779, 4, 0.0), (403779, 4, 40.0)]),
            (_cfg(pp_window_s=15.0), [(9000, 2, 0.0), (8000, 2, 15.0)]),
            (_cfg(flip_tokens=99999), [(10, 2, 0.0), (10, 2, 5.0)]),
        ):
            state = pp.PhasePolicyState()
            for pending, bs, now in series:
                out = _drive(cfg, state, pending, bs, now)
                self.assertNotIn("HOLD", out.reason, f"held: {out.reason}")


class TestTheAllowlistMembership(CustomTestCase):
    """Exactly one arm is eligible, and which one is pinned deliberately."""

    def test_the_allowlist_is_empty(self):
        # Enumerated from the source rather than asserted from memory. Empty is
        # the DECIDED state, not an oversight: no arm in the rules today meets
        # the admission condition, so none claims eligibility.
        import inspect

        src = inspect.getsource(pp._decide_from_load)
        self.assertEqual(
            src.count("hold_eligible=True"),
            0,
            "an arm claims hold eligibility -- that is admissible only if a "
            "SECOND independent anti-starvation bound is armed in every state "
            "where that arm fires. Argue that here, do not just add it.",
        )

    def test_the_admission_condition_is_written_down_where_it_binds(self):
        # THE RULE ITSELF, pinned. A condition that lives only in a commit
        # message is one refactor away from being lost, and the next author to
        # add an arm reads the dataclass, not the history.
        doc = pp.PhasePolicyDecision.__doc__ or ""
        import inspect

        src = inspect.getsource(pp.PhasePolicyDecision)
        text = doc + src
        self.assertIn("SECOND INDEPENDENT anti-starvation bound", text)
        self.assertIn("EMPTY TODAY", text)

    def test_the_stopwatch_is_the_arm_that_fails_the_condition(self):
        # Why it fails, checked against the code rather than recited: the arm
        # is guarded by the ABSENCE of the decode-starvation cap, so it is the
        # last bound in every state it fires in.
        import inspect

        src = inspect.getsource(pp._decide_from_load)
        marker = src.index("HAND-SET STOPWATCH")
        head = src[max(0, marker - 2500) : marker]
        self.assertIn("cap <= 0", head)
        self.assertIn("cfg.pp_window_s > 0", head)


if __name__ == "__main__":
    unittest.main()
