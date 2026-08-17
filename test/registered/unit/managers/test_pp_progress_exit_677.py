"""#677(a): PP must leave when admission is BLOCKED, not only when it drains.

THE WEDGE, LIVE, 2026-08-16 06:04. Four minutes into a pure-drain PP boot the
instance stopped serving and the user saw a dead server:

    last prefill batch admitted   06:04:21
    pending prefill               403779 tok, FROZEN from 06:04:47 onward
    running bs                    4  (= max_running_requests, all carried decodes)
    policy line                   "holding in pp: prefilling in pp (403779 tok
                                   pending) (pending prefill 403779 tok,
                                   running bs 4)"

Every slot was held by a carried decode, PP may not decode under strict purity,
so no slot could free; admission needs a slot, so no prefill could be admitted;
and the drain rule waits for pending to fall below one chunk, which it never
would. The policy was not wrong about any single fact -- it was waiting for an
event that could no longer happen.

WHY THE EXISTING EXITS DID NOT COVER IT.
  * DRAINED fires on ``pending <= pp_exit_tokens``. Pending was 403779 and
    frozen: the condition is unreachable, not merely distant.
  * The DECODE-STARVATION CAP is solved from ``decode_stall_slo_s``, which the
    user's pure-drain decision sets to no stopwatch -- deliberately, because a
    real drain must never be cut short by a clock. With the cap off there was
    nothing left to bound the wedge.

So the missing exit is not another timer. It is the observation that PREFILL
ITSELF STOPPED MOVING while decodes were carried -- which distinguishes a wedge
from a slow drain without appealing to a deadline at all. A drain that is
progressing keeps resetting the clock however long it takes; a wedge cannot.

PROGRESS IS MEASURED, NOT INFERRED. ``observe_idle`` records the last tick at
which ``pending_prefill_tokens`` DECREASED. ``decide`` stays pure and reads that
stamp -- the same split the idle clock already uses, for the same reason.

THE WINDOW IS SOLVED. A wedge is "no chunk landed in the time several chunks
should have taken": ``PROGRESS_STALL_CHUNKS * (pp_exit_tokens /
pp_prefill_tok_s)``, floored at ``2 * flip_cost_s`` so that seam time alone --
the one interval in which prefill legitimately makes no progress -- can never
trigger it.
"""

import unittest

from sglang.srt.managers import phase_policy as pp
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

FROZEN_TOK = 403779
CARRIED_BS = 4


def _cfg(**kw):
    """The pure-drain shape: no stopwatch, no decode-stall cap."""
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


def _inp(now, *, pending=FROZEN_TOK, bs=CARRIED_BS, phase="pp"):
    return pp.PhasePolicyInputs(
        phase=phase, pending_prefill_tokens=pending, running_bs=bs, now=now
    )


def _run(cfg, series):
    """Drive observe->decide over a series of (now, pending) as the loop does."""
    state = pp.PhasePolicyState()
    last = None
    for now, pending in series:
        inp = _inp(now, pending=pending)
        pp.observe_idle(state, inp)
        last = pp.decide(cfg, state, inp)
    return last, state


class TheWedgeIsLeft(unittest.TestCase):
    def test_frozen_pending_with_carried_decodes_exits_pp(self):
        """RED before #677(a): the policy holds in PP forever."""
        cfg = _cfg()
        window = pp.pp_progress_stall_window_s(cfg)
        self.assertGreater(window, 0)
        # One observation, then the clock runs with pending frozen.
        series = [(0.0, FROZEN_TOK), (window + 1.0, FROZEN_TOK)]
        decision, _ = _run(cfg, series)
        self.assertEqual(pp.PP_TO_TP, decision.direction, decision.reason)

    def test_the_receipt_names_the_frozen_value_the_stall_and_the_carry(self):
        """A distinct receipt, so the next audit can count wedge exits against
        drain exits instead of inferring which one happened."""
        cfg = _cfg()
        window = pp.pp_progress_stall_window_s(cfg)
        decision, _ = _run(cfg, [(0.0, FROZEN_TOK), (window + 1.0, FROZEN_TOK)])
        self.assertIn("blocked admission", decision.reason)
        self.assertIn(str(FROZEN_TOK), decision.reason)
        self.assertIn(str(CARRIED_BS), decision.reason)

    def test_it_does_not_fire_before_the_window(self):
        cfg = _cfg()
        window = pp.pp_progress_stall_window_s(cfg)
        decision, _ = _run(cfg, [(0.0, FROZEN_TOK), (window * 0.5, FROZEN_TOK)])
        self.assertIsNone(decision.direction, decision.reason)


class ARealDrainIsNeverCutShort(unittest.TestCase):
    """The property that makes this safe under the user's pure-drain decision:
    a drain that is MOVING is never interrupted, however long it takes."""

    def test_a_strictly_decreasing_backlog_never_triggers_the_exit(self):
        cfg = _cfg()
        window = pp.pp_progress_stall_window_s(cfg)
        pending = 400_000
        series = []
        now = 0.0
        # Far longer than the window, but progressing on every tick.
        while now < window * 10:
            series.append((now, pending))
            pending -= 512
            now += window / 4
        decision, _ = _run(cfg, series)
        self.assertIsNone(
            decision.direction,
            f"a progressing drain was cut short: {decision.reason}",
        )

    def test_a_slow_but_moving_drain_survives_many_windows(self):
        """One chunk per window is still progress. The rule is 'moving', not
        'fast' -- a speed threshold would be the stopwatch again."""
        cfg = _cfg()
        window = pp.pp_progress_stall_window_s(cfg)
        pending, now, series = 400_000, 0.0, []
        for _ in range(8):
            series.append((now, pending))
            now += window * 0.9
            pending -= 1
        decision, _ = _run(cfg, series)
        self.assertIsNone(decision.direction, decision.reason)

    def test_progress_resets_the_clock_so_a_stall_must_be_fresh(self):
        cfg = _cfg()
        window = pp.pp_progress_stall_window_s(cfg)
        decision, _ = _run(
            cfg,
            [
                (0.0, FROZEN_TOK),
                (window * 0.9, FROZEN_TOK),
                (window * 0.95, FROZEN_TOK - 512),  # a chunk lands
                (window * 1.5, FROZEN_TOK - 512),  # stalled again, but briefly
            ],
        )
        self.assertIsNone(decision.direction, decision.reason)

    def test_an_empty_backlog_is_the_DRAINED_exit_not_this_one(self):
        cfg = _cfg()
        window = pp.pp_progress_stall_window_s(cfg)
        decision, _ = _run(cfg, [(0.0, 100), (window + 1.0, 100)])
        self.assertEqual(pp.PP_TO_TP, decision.direction)
        self.assertIn("DRAINED", decision.reason)

    def test_no_carried_decode_means_no_wedge_to_break(self):
        """With bs 0 nothing is being starved; PP may keep working."""
        cfg = _cfg()
        window = pp.pp_progress_stall_window_s(cfg)
        state = pp.PhasePolicyState()
        last = None
        for now in (0.0, window + 1.0):
            inp = _inp(now, bs=0)
            pp.observe_idle(state, inp)
            last = pp.decide(cfg, state, inp)
        self.assertNotIn("blocked admission", last.reason)


class TheWindowIsSolvedNotSet(unittest.TestCase):
    def test_it_is_a_multiple_of_the_chunk_cadence(self):
        cfg = _cfg(pp_exit_tokens=512, pp_prefill_tok_s=1000.0, flip_cost_s=0.0)
        cadence = 512 / 1000.0
        self.assertAlmostEqual(
            pp.PROGRESS_STALL_CHUNKS * cadence, pp.pp_progress_stall_window_s(cfg)
        )

    def test_seam_time_alone_can_never_trigger_it(self):
        """THE FLOOR, and the reason it exists. During a cutover prefill makes
        no progress by construction; without this floor a fast rig could solve
        a window shorter than its own seam and exit on the seam it just paid
        for."""
        cfg = _cfg(pp_exit_tokens=1, pp_prefill_tok_s=1e9, flip_cost_s=3.0)
        self.assertGreaterEqual(pp.pp_progress_stall_window_s(cfg), 2 * 3.0)

    def test_an_unusable_rate_disables_the_rule_rather_than_dividing_by_zero(self):
        cfg = _cfg(pp_prefill_tok_s=0.0, flip_cost_s=0.0)
        self.assertEqual(0.0, pp.pp_progress_stall_window_s(cfg))
        decision, _ = _run(cfg, [(0.0, FROZEN_TOK), (10_000.0, FROZEN_TOK)])
        self.assertIsNone(decision.direction, decision.reason)


class TheOuterBackstopStillExists(unittest.TestCase):
    """The 180s SGLANG_PHASE_POLICY_DECODE_STALL_SLO_S mitigation stays; this
    exit must fire FIRST so the wedge is bounded by progress, not by a clock."""

    def test_the_progress_exit_beats_the_decode_stall_cap(self):
        cfg = _cfg(decode_stall_slo_s=180.0)
        cap = pp.pp_residency_cap_s(cfg)
        window = pp.pp_progress_stall_window_s(cfg)
        self.assertGreater(cap, 0.0, "the backstop must still be configured")
        self.assertLess(
            window, cap, "the progress exit must be reachable before the timer"
        )
        state = pp.PhasePolicyState()
        last = None
        for now in (0.0, window + 1.0):
            inp = _inp(now)
            pp.observe_idle(state, inp)
            last = pp.decide(cfg, state, inp)
        self.assertEqual(pp.PP_TO_TP, last.direction)
        self.assertIn("blocked admission", last.reason)


if __name__ == "__main__":
    unittest.main()
