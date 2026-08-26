"""#889 -- a shipped flag that a COMBINATION turns into a silent no-op.

THE DEFECT, operator-verified at pin 0cd27d957d
-----------------------------------------------
``--phase-policy-pp-window-s`` is inert whenever a decode-stall SLO is also
declared. ``decide`` reaches the hand-set stopwatch only through

    cap <= 0 and cfg.pp_window_s > 0 and in_pp >= cfg.pp_window_s

and ``cap`` comes from ``pp_residency_cap_s`` = ``slo - 2 * flip_cost_s``. So
for every ``slo > 2 * flip_cost_s`` the stopwatch is UNREACHABLE. On the
w38b -> w39 -> w40 line BOTH flags were set (window 15 s, SLO 180 s, seam
3.2 s), which makes the effective PP residency bound

    173.6 s, not the 15 s the boot line printed -- a factor of 11.573,

and in the dangerous direction: much LONGER than every reader believed. The
supersession itself is intended (``phase_purity.validate_purity_policy_pair``
accepts a declared SLO as the substitute bound, on purpose). What was a defect
is that it happened in SILENCE, and that the one artifact an operator reads --
the ``PHASE-POLICY armed:`` line -- reported the inert number as if it governed.

WHAT THIS SUITE PINS
--------------------
1. ``effective_pp_exit_term`` names the bound ``decide`` will actually use, and
   agrees with ``decide`` ON BOTH SIDES of the boundary. The two must not be
   able to drift: a guard changed in one place and not the other is exactly how
   this defect was born.
2. ``superseded_pp_bound_warning`` names BOTH numbers and the direction, in
   both directions of supersession (SLO silences the window; an SLO tighter
   than the round trip is itself silenced by the window).
3. The boot line prints the EFFECTIVE term, and the warning is emitted at parse
   time.
4. Nothing fires when only one bound is declared -- the fix must not add noise
   to the configurations that were never ambiguous.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest import mock

from sglang.srt.managers.phase_policy import (
    PP_EXIT_BY_DRAIN,
    PP_EXIT_BY_SLO_CAP,
    PP_EXIT_BY_STOPWATCH,
    PP_TO_TP,
    PhasePolicyConfig,
    PhasePolicyInputs,
    PhasePolicyState,
    config_from_env,
    decide,
    effective_pp_exit_term,
    pp_residency_cap_s,
    superseded_pp_bound_warning,
)
from sglang.test.test_utils import CustomTestCase

# The booted values on the w38b/w39/w40 line, so the numbers below are live.
N = 7004
SEAM = 3.2
WINDOW = 15.0
SLO = 180.0
CAP = SLO - 2 * SEAM  # 173.6


def _cfg(**kw):
    base = dict(
        enabled=True,
        flip_tokens=N,
        min_dwell_s=0.0,
        tp_decode_floor_s=0.0,
        prefill_runs_in_tp=True,
        flip_cost_s=SEAM,
        pp_window_s=WINDOW,
        decode_stall_slo_s=SLO,
        pp_prefill_tok_s=7245.5,
        tp_prefill_tok_s=1681.0,
    )
    base.update(kw)
    return PhasePolicyConfig(**base)


def _pp(cfg, in_pp, pending=200_000, bs=3):
    state = PhasePolicyState()
    state.phase_since = 1000.0
    return decide(
        cfg,
        state,
        PhasePolicyInputs(
            phase="pp",
            pending_prefill_tokens=pending,
            running_bs=bs,
            now=1000.0 + in_pp,
        ),
    )


class TestTheEffectiveTermIsTheOneDecideUses(CustomTestCase):
    """The helper is only worth having if it cannot disagree with `decide`."""

    def test_the_live_pair_reports_the_cap_not_the_requested_window(self):
        name, term = effective_pp_exit_term(_cfg())
        self.assertEqual(name, PP_EXIT_BY_SLO_CAP)
        self.assertAlmostEqual(term, CAP)

    def test_the_requested_window_governs_when_no_slo_is_declared(self):
        name, term = effective_pp_exit_term(_cfg(decode_stall_slo_s=0.0))
        self.assertEqual(name, PP_EXIT_BY_STOPWATCH)
        self.assertAlmostEqual(term, WINDOW)

    def test_neither_declared_leaves_the_phase_to_drain(self):
        name, term = effective_pp_exit_term(
            _cfg(decode_stall_slo_s=0.0, pp_window_s=0.0)
        )
        self.assertEqual(name, PP_EXIT_BY_DRAIN)
        self.assertEqual(term, 0.0)

    def test_an_slo_tighter_than_the_round_trip_hands_it_back(self):
        """cap collapses to 0, so the stopwatch becomes reachable again."""
        cfg = _cfg(decode_stall_slo_s=4.0)
        self.assertEqual(pp_residency_cap_s(cfg), 0.0)
        name, term = effective_pp_exit_term(cfg)
        self.assertEqual(name, PP_EXIT_BY_STOPWATCH)
        self.assertAlmostEqual(term, WINDOW)

    def test_decide_agrees_with_the_reported_term_on_both_sides(self):
        """THE ANTI-DRIFT PIN. A guard moved in one place and not the other is
        how #889 was born; this fails on either half alone."""
        for cfg in (
            _cfg(),  # cap governs
            _cfg(decode_stall_slo_s=0.0),  # stopwatch governs
            _cfg(decode_stall_slo_s=4.0),  # cap collapsed, stopwatch back
        ):
            name, term = effective_pp_exit_term(cfg)
            with self.subTest(name=name, term=term):
                self.assertIsNone(_pp(cfg, term - 0.1).direction)
                self.assertEqual(_pp(cfg, term + 0.1).direction, PP_TO_TP)


class TestTheSupersessionIsAnnounced(CustomTestCase):
    def test_it_names_both_numbers_and_the_dangerous_direction(self):
        msg = superseded_pp_bound_warning(_cfg())
        self.assertIsNotNone(msg)
        self.assertIn("15", msg)
        self.assertIn("173.6", msg)
        self.assertIn("LONGER", msg)
        self.assertIn("phase-policy-pp-window-s", msg)
        self.assertIn("SGLANG_PHASE_POLICY_DECODE_STALL_SLO_S", msg)

    def test_a_cap_shorter_than_the_window_is_announced_as_shorter(self):
        msg = superseded_pp_bound_warning(_cfg(decode_stall_slo_s=20.0))
        self.assertIsNotNone(msg)
        self.assertIn("SHORTER", msg)
        self.assertIn("13.6", msg)

    def test_the_mirror_direction_is_announced_too(self):
        """An SLO below the round trip is the one that goes silently inert."""
        msg = superseded_pp_bound_warning(_cfg(decode_stall_slo_s=4.0))
        self.assertIsNotNone(msg)
        self.assertIn("4", msg)
        self.assertIn("15", msg)
        self.assertIn("6.4", msg)  # 2 x seam, the bar the SLO failed to clear

    def test_one_bound_alone_is_never_ambiguous_and_stays_quiet(self):
        self.assertIsNone(superseded_pp_bound_warning(_cfg(decode_stall_slo_s=0.0)))
        self.assertIsNone(superseded_pp_bound_warning(_cfg(pp_window_s=0.0)))
        self.assertIsNone(
            superseded_pp_bound_warning(_cfg(pp_window_s=0.0, decode_stall_slo_s=0.0))
        )


class TestTheBootLineStopsLying(CustomTestCase):
    """The `PHASE-POLICY armed:` line is the artifact operators read. At the pin
    it printed `pp window 15s` while the phase actually ran to 173.6 s."""

    def _boot(self, **env):
        base = {
            "SGLANG_PHASE_POLICY_PP_WINDOW_S": "15",
            "SGLANG_PHASE_POLICY_DECODE_STALL_SLO_S": "180",
            "SGLANG_PHASE_POLICY_FLIP_COST_S": "3.2",
        }
        base.update(env)
        with mock.patch.dict("os.environ", base, clear=False):
            with self.assertLogs(
                "sglang.srt.managers.phase_policy", level="WARNING"
            ) as cm:
                cfg = config_from_env(enabled=True)
        return cfg, "\n".join(cm.output)

    def _armed_line(self, log):
        """The ONE record an operator reads. Asserting against the joined log
        instead let mutant M6 survive: the supersession warning carried the
        effective number, so a boot line that had reverted to printing only the
        inert window still looked green."""
        lines = [ln for ln in log.splitlines() if "armed:" in ln]
        self.assertEqual(len(lines), 1, log)
        return lines[0]

    def test_the_armed_line_carries_the_effective_term(self):
        _, log = self._boot()
        armed = self._armed_line(log)
        self.assertIn("effective pp exit", armed)
        self.assertIn(PP_EXIT_BY_SLO_CAP, armed)
        self.assertIn("173.6", armed)
        # and the requested knob is still shown, so the pair is legible in one
        # line rather than only in the warning below it.
        self.assertIn("pp window 15s", armed)

    def test_a_slo_free_boot_names_the_stopwatch_in_the_armed_line(self):
        _, log = self._boot(SGLANG_PHASE_POLICY_DECODE_STALL_SLO_S="0")
        armed = self._armed_line(log)
        self.assertIn(PP_EXIT_BY_STOPWATCH, armed)
        self.assertIn("15s", armed)

    def test_the_supersession_warning_is_emitted_at_parse_time(self):
        _, log = self._boot()
        self.assertIn("LONGER", log)
        self.assertIn("173.6", log)
        self.assertIn("15", log)

    def test_a_window_only_boot_says_the_window_governs_and_does_not_warn(self):
        _, log = self._boot(SGLANG_PHASE_POLICY_DECODE_STALL_SLO_S="0")
        self.assertIn("armed:", log)
        self.assertNotIn("LONGER", log)
        self.assertNotIn("SHORTER", log)


if __name__ == "__main__":
    unittest.main()
