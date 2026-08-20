"""#781: the phase-policy tuning knobs are FLAGS; the env is a deprecated bridge.

These knobs used to come from the environment only, and the boot env was
assembled by concatenating a captured shell environment, a heredoc and
EXTRA_ENV. A key written twice resolved silently as "last one wins":
SGLANG_PHASE_POLICY_TP_DECODE_FLOOR_S was 10 in one half and 8 in the other, and
the 10 had been dead the whole time with nobody aware of it.

What has to hold now:
  1. a flag reaches the config,
  2. the flag WINS over a conflicting env var -- this is the test that would
     have caught the original defect,
  3. an unset flag still resolves the env, so existing deployments are
     unchanged,
  4. neither set leaves the default in place by accident.

Hermetic: config_from_env is pure argument/env plumbing, so this runs on CPU
with no model, no device and no server.
"""

import os
import unittest
from unittest import mock

from sglang.srt.managers.phase_policy import config_from_env


class _Args:
    """Stand-in for ServerArgs: config_from_env only ever getattr()s it."""

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def __getattr__(self, name):  # unset flags read as None, like the dataclass
        return None


# (ServerArgs field, env var, flag value, conflicting env value, config attr)
CASES = [
    ("phase_policy_min_dwell_s", "SGLANG_PHASE_POLICY_MIN_DWELL_S",
     7.0, "99.0", "min_dwell_s"),
    ("phase_policy_tp_decode_floor_s", "SGLANG_PHASE_POLICY_TP_DECODE_FLOOR_S",
     10.0, "8", "tp_decode_floor_s"),
    ("phase_policy_pp_window_s", "SGLANG_PHASE_POLICY_PP_WINDOW_S",
     4.0, "15", "pp_window_s"),
    ("phase_policy_decode_stall_slo_s", "SGLANG_PHASE_POLICY_DECODE_STALL_SLO_S",
     180.0, "12", "decode_stall_slo_s"),
    ("phase_policy_decode_contention", "SGLANG_PHASE_POLICY_DECODE_CONTENTION",
     1.0, "0.25", "decode_contention"),
]


class TestPhasePolicyFlagPromotion781(unittest.TestCase):
    def _cfg(self, args=None):
        return config_from_env(enabled=False, server_args=args)

    def test_flag_reaches_the_config(self):
        for field, _env, flag_value, _bad, attr in CASES:
            with self.subTest(field=field):
                cfg = self._cfg(_Args(**{field: flag_value}))
                self.assertEqual(
                    getattr(cfg, attr), flag_value,
                    f"--{field.replace('_', '-')} did not reach cfg.{attr}",
                )

    def test_flag_beats_a_conflicting_env_var(self):
        """The defect this whole promotion exists for.

        The env said 8 while the other half of the config said 10. With the
        flag authoritative, the flag's value is the one that survives.
        """
        for field, env, flag_value, bad_env, attr in CASES:
            with self.subTest(field=field):
                with mock.patch.dict(os.environ, {env: bad_env}, clear=False):
                    cfg = self._cfg(_Args(**{field: flag_value}))
                self.assertEqual(
                    getattr(cfg, attr), flag_value,
                    f"env {env}={bad_env} overrode the flag -- last-wins is back",
                )

    def test_env_still_resolves_when_the_flag_is_unset(self):
        """The deprecation bridge: nobody's running deployment changes today."""
        for field, env, _flag_value, env_value, attr in CASES:
            with self.subTest(field=field):
                with mock.patch.dict(os.environ, {env: env_value}, clear=False):
                    cfg = self._cfg(_Args())
                self.assertEqual(
                    float(getattr(cfg, attr)), float(env_value),
                    f"{env} stopped being honoured with the flag unset",
                )

    def test_drain_mode_flag_and_env(self):
        cfg = self._cfg(_Args(phase_policy_drain_mode=True))
        self.assertTrue(cfg.drain_mode)
        # flag False must beat env "1" -- a bool flag that cannot turn a
        # feature OFF is the inverted-polarity trap that bit
        # SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK.
        with mock.patch.dict(
            os.environ, {"SGLANG_PHASE_POLICY_DRAIN_MODE": "1"}, clear=False
        ):
            cfg = self._cfg(_Args(phase_policy_drain_mode=False))
            self.assertFalse(
                cfg.drain_mode, "flag False could not override env-on"
            )
            cfg_env_only = self._cfg(_Args())
            self.assertTrue(
                cfg_env_only.drain_mode, "env fallback stopped working"
            )

    def test_no_server_args_is_unchanged_behaviour(self):
        """Every existing caller passes no ServerArgs; none of them may move."""
        with mock.patch.dict(
            os.environ, {"SGLANG_PHASE_POLICY_PP_WINDOW_S": "6"}, clear=False
        ):
            self.assertEqual(config_from_env(enabled=False).pp_window_s, 6.0)

    def test_fields_exist_on_server_args(self):
        """The flags must be real ServerArgs fields, not just accepted kwargs."""
        import dataclasses

        from sglang.srt.server_args import ServerArgs

        names = {f.name for f in dataclasses.fields(ServerArgs)}
        for field, _env, _v, _b, _a in CASES:
            self.assertIn(field, names, f"{field} is not a ServerArgs field")
        for extra in (
            "phase_policy_drain_mode",
            "seam_entry_margin_mib",
            "seam_entry_delay_budget_s",
            "flip_seam_chunk_mib",
            "collective_census_interval",
            "phase_flip_corridor_floor_mib",
        ):
            self.assertIn(extra, names, f"{extra} is not a ServerArgs field")


if __name__ == "__main__":
    unittest.main()
