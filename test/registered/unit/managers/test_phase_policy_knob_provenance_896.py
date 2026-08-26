"""#896: an effective phase-policy knob PRINTS where its value came from.

The ticket, concretely. Boot ``boot_w38rerun_0826_1304.log`` armed a cutover on
``decode_stall_slo_s=180``: line 31147 says "residency 173.6s >= 173.6s solved
as slo - 2x3.2s seam", which is 180 - 2*3.2. The arming line one screen earlier
prints min dwell, idle dwell, pp window, tp decode floor, seam, strand weight
and decode contention -- and NOT the SLO. So the only trace of the governing
180 in that boot was the 27 KB ``ServerArgs`` repr on line 71. Two follow-up
tickets hung their semantics on that number while its origin was, in practice,
unfindable.

The value itself was never lost: it came from ``--phase-policy-decode-stall-slo-s
180`` in the launcher, promoted to a flag by #781. What was missing is the
PRINTED provenance -- and "the value is recoverable if you read the right 27 KB
line" is the same failure as not printing it.

The class is "effective value with no printed provenance", so the guard is not
about the SLO alone: ``test_every_core_knob_prints_a_source`` fails if ANY core
knob is dropped from the line. A per-knob guard would let the next knob go
silent exactly the way this one did.

Hermetic: ``config_from_env`` is pure argument/env plumbing -- no CPU device, no
model, no server.
"""

import os
import unittest
from unittest import mock

from sglang.srt.managers.phase_policy import (
    ENV_DECODE_STALL_SLO,
    LOG_PREFIX,
    config_from_env,
)

#: Every SGLANG_PHASE_POLICY_* name config_from_env reads. Cleared for every
#: case so a stray var in the developer's shell cannot turn a "default" run
#: into an "env" run -- which is precisely the confusion #781 was filed for.
_POLICY_ENV = (
    "SGLANG_PHASE_POLICY_MIN_DWELL_S",
    "SGLANG_PHASE_POLICY_IDLE_DWELL_S",
    "SGLANG_PHASE_POLICY_PP_WINDOW_S",
    "SGLANG_PHASE_POLICY_TP_DECODE_FLOOR_S",
    "SGLANG_PHASE_POLICY_FLIP_COST_S",
    "SGLANG_PHASE_POLICY_FLIP_TOKENS",
    "SGLANG_PHASE_POLICY_DECODE_STRAND_WEIGHT",
    "SGLANG_PHASE_POLICY_DECODE_CONTENTION",
    ENV_DECODE_STALL_SLO,
    "SGLANG_PHASE_POLICY_DRAIN_MODE",
    "SGLANG_PHASE_POLICY_DRAIN_MODE_STRICT",
    "SGLANG_PHASE_POLICY_TP_TOK_S",
    "SGLANG_PHASE_POLICY_PP_TOK_S",
    "SGLANG_PHASE_POLICY_PP_EXIT_TOKENS",
    "SGLANG_PHASE_POLICY_REFUSAL_BACKOFF_CAP_S",
    "SGLANG_PHASE_POLICY_REFUSAL_DEGRADE_AFTER",
    "HTSGLANG_PHASE_IDLE_STATE",
)

#: The knobs whose value governs a cutover decision. Name as printed.
CORE_KNOBS = (
    "min_dwell_s",
    "idle_dwell_s",
    "pp_window_s",
    "tp_decode_floor_s",
    "flip_cost_s",
    "decode_strand_weight",
    "decode_contention",
    "decode_stall_slo_s",
    "tp_prefill_tok_s",
    "pp_prefill_tok_s",
    "pp_exit_tokens",
    "refusal_backoff_cap_s",
    "refusal_degrade_after",
    "flip_tokens",
    "rest_state",
)


class _Args:
    """Stand-in for ServerArgs: config_from_env only ever getattr()s it."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __getattr__(self, name):  # unset flags read as None, like the dataclass
        return None


class TestPhasePolicyKnobProvenance896(unittest.TestCase):
    def _provenance_line(self, args=None, env=None):
        """Arm the policy and return the single 'knob provenance' log line."""
        # The baseline is REMOVED from the environment; what the caller asks
        # for is SET, empty string included. Collapsing the two -- popping
        # every empty value -- is how the first version of this helper made
        # `test_an_empty_env_var_is_not_a_source` unfalsifiable: it removed the
        # very variable whose empty-but-present state was under test, and a
        # mutant that treated `FOO=` as a source survived it.
        env = dict(env or {})
        with mock.patch.dict(os.environ, env, clear=False):
            for k in _POLICY_ENV:
                if k not in env:
                    os.environ.pop(k, None)
            with self.assertLogs(
                "sglang.srt.managers.phase_policy", level="WARNING"
            ) as caught:
                config_from_env(enabled=True, server_args=args or _Args())
        lines = [m for m in caught.output if "knob provenance" in m]
        self.assertEqual(
            len(lines),
            1,
            f"expected exactly one provenance line, got {len(lines)}: {caught.output}",
        )
        self.assertIn(LOG_PREFIX, lines[0])
        return lines[0]

    def test_slo_from_a_flag_names_the_flag(self):
        """The #896 boot, reproduced: 180 arrives by flag and the log says so."""
        line = self._provenance_line(_Args(phase_policy_decode_stall_slo_s=180.0))
        self.assertIn(
            "decode_stall_slo_s=180 from flag --phase-policy-decode-stall-slo-s",
            line,
            f"the governing SLO did not name its flag: {line}",
        )

    def test_slo_from_the_env_names_the_env_var(self):
        """The deprecated bridge is still a real source and must be named as one."""
        line = self._provenance_line(env={ENV_DECODE_STALL_SLO: "90"})
        self.assertIn(f"decode_stall_slo_s=90 from env {ENV_DECODE_STALL_SLO}", line)

    def test_slo_from_neither_says_default(self):
        """0 = cap disabled. Reading that as 'someone set it to 0' is the same
        blindness in the other direction, so the default is named too."""
        line = self._provenance_line()
        self.assertIn("decode_stall_slo_s=0 from default", line)

    def test_an_empty_env_var_is_not_a_source(self):
        """`FOO=` falls through to the default in _env_float, so claiming 'env'
        there would name a source that did not supply the value."""
        line = self._provenance_line(env={ENV_DECODE_STALL_SLO: ""})
        self.assertIn("decode_stall_slo_s=0 from default", line)

    def test_an_empty_env_var_is_not_a_source_on_the_flagless_path_either(self):
        """The same rule, the other resolver. Knobs with no flag yet
        (idle dwell, seam seed, strand weight) run through ``_env_source``
        rather than ``_flag_or_env``, so the empty-string case has to be held
        in both places or half the line can still name a phantom source."""
        line = self._provenance_line(env={"SGLANG_PHASE_POLICY_IDLE_DWELL_S": ""})
        self.assertIn("idle_dwell_s=3 from default", line)
        self.assertNotIn("idle_dwell_s=3 from env", line)

    def test_flag_wins_over_env_in_the_printed_source_too(self):
        """#781 made the flag authoritative for the VALUE. If the provenance
        disagreed with the resolution, the line would be a second opinion about
        the first one -- worse than no line."""
        line = self._provenance_line(
            _Args(phase_policy_decode_stall_slo_s=180.0),
            env={ENV_DECODE_STALL_SLO: "12"},
        )
        self.assertIn("decode_stall_slo_s=180 from flag", line)
        self.assertNotIn(
            f"decode_stall_slo_s=180 from env {ENV_DECODE_STALL_SLO}", line
        )

    def test_every_core_knob_prints_a_source(self):
        """THE CLASS GUARD. Not the SLO alone -- any core knob dropped from the
        line reintroduces exactly the silence this ticket is about."""
        line = self._provenance_line()
        missing = [k for k in CORE_KNOBS if f"{k}=" not in line]
        self.assertEqual(missing, [], f"knobs with no printed provenance: {missing}")
        for knob in CORE_KNOBS:
            with self.subTest(knob=knob):
                after = line.split(f"{knob}=", 1)[1]
                self.assertRegex(
                    after.split("|")[0],
                    r"\S+ from \S+",
                    f"{knob} printed a value but no 'from <source>': {line}",
                )

    def test_the_seam_says_seed_or_measured_not_just_where_the_number_came_from(self):
        """flip_cost_s has two provenances and both govern: where the SEED came
        from, and whether the estimator still sits on it. Printing only the
        first reads as 'measured' when nothing has been measured."""
        line = self._provenance_line(env={"SGLANG_PHASE_POLICY_FLIP_COST_S": "5.918"})
        self.assertIn(
            "flip_cost_s=5.918 from env SGLANG_PHASE_POLICY_FLIP_COST_S", line
        )
        seam = line.split("flip_cost_s=", 1)[1]
        self.assertIn(
            "seed", seam, f"the seam did not report its estimator state: {line}"
        )

    def test_the_line_is_silent_when_the_policy_is_off(self):
        """A disabled policy governs nothing; provenance for knobs that never
        take effect is noise, and the arming line is already gated the same way."""
        with mock.patch.dict(os.environ, {}, clear=False):
            with self.assertLogs(
                "sglang.srt.managers.phase_policy", level="WARNING"
            ) as caught:
                import logging

                logging.getLogger("sglang.srt.managers.phase_policy").warning("probe")
                config_from_env(enabled=False, server_args=_Args())
        self.assertEqual([m for m in caught.output if "knob provenance" in m], [])


if __name__ == "__main__":
    unittest.main()
