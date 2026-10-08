# SPDX-License-Identifier: Apache-2.0
"""LEISTUNGSSCHALTER NF (user rule 29.09. ~10:15Z: a switch proven on metal is
default ON in the code; "Profilzeile ist kein Ersatz").

The NF class-(a) switches of /spinning/gpu-arb/docs/LEISTUNGSSCHALTER-INVENTAR-0929.md
default ON on the registry row ``nextflash`` with the value the running profile
(nf-h91-dpr-sa-vis-adopt-st-cut-vsync-odx-2b-swr-e2cut-z30y2-arr-wre-ta-dh-ml-ef-
rwf-pfo-pw-hc-tse-srw-tsw-dres.env) gave each GROUP; the launcher writes them into
--env-p/--env-d (a stated or exported value wins) and takes --d-kv-token-cut owned
as the row's default. The qwen27b row stays byte-identical. Class (d): the NaN
guards' wave/fetch checks are diagnostics and default OFF.
"""

import argparse
import os
import types
import unittest.mock as mock

from flliper.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import form as F  # noqa: E402
from flliper.srt.pdflip import launcher as L  # noqa: E402

#: the running NF profile, NF_ENV_P_FORM / NF_ENV_D_FORM / NF_ENV_D / _form
PROFILE_P = {
    "FLLIPER_PDFLIP_ENABLE_P_TAIL_FOLD": "1",
    "FLLIPER_PDFLIP_TAIL_KEEP_MIB": "512",
    "FLLIPER_PDFLIP_PLE_STATE_HANDOFF": "1",
    "FLLIPER_FORCE_QSA_ROWS_CONFIG": "inf=64/8/2",
    "FLLIPER_PDFLIP_QSA_FP8_DECODE": "ptx",
    "FLLIPER_PDFLIP_QSA_ROWS_FUSED_EAGER": "1",
    "FLLIPER_UNEVEN_MOE_EXPERT_SHARD": "1",
    "FLLIPER_HC_MIXER_INT8": "1",
    "FLLIPER_WEIGHT_LOADER_COALESCE_MIB": "32",
    # SWITCH-DEFAULTS 1004: nf-int4-h6-abl.env NF_ENV_P (P only)
    "FLLIPER_PDFLIP_ENABLE_PREFILL_FETCH_OVERLAP": "1",
    "FLLIPER_PDFLIP_ENABLE_TARGETED_PREWARM": "1",
}
PROFILE_D = {
    "FLLIPER_PDFLIP_TAIL_KEEP_MIB": "512",
    "FLLIPER_PDFLIP_PLE_STATE_HANDOFF": "1",
    "FLLIPER_PDFLIP_FORM_A_PLE_FULL_VOCAB": "1",
    "FLLIPER_PDFLIP_PLE_STAGE_BEHIND_REPLAY": "1",
    "FLLIPER_PDFLIP_PLE_STAGE_AUTONOMOUS": "1",
    "FLLIPER_PDFLIP_PLE_STAGE_BONUS_EARLY": "1",
    "FLLIPER_QWEN4_PLE_DECODE_PREAD_PROCS": "8",
    "FLLIPER_QWEN4_PLE_DECODE_PREAD_THREADS": "8",
    "FLLIPER_UNEVEN_MOE_EXPERT_SHARD": "1",
    "FLLIPER_PDFLIP_ENABLE_CUT_WORKER_END": "1",
    "FLLIPER_PDFLIP_ENABLE_D_PARK_END": "1",
    "FLLIPER_HC_MIXER_INT8": "1",
    "FLLIPER_WEIGHT_LOADER_COALESCE_MIB": "32",
    # SCHALTER-HALBPORT 1002: nf-int4.env NF_ENV_D_FORM (D only)
    "FLLIPER_PDFLIP_STORE_SHORT_TAIL": "1",
    # SWITCH-DEFAULTS 1004: nf-int4-h6-abl.env NF_ENV_D (D only)
    "FLLIPER_PDFLIP_ENABLE_TAIL_STAGE_EARLY": "1",
    "FLLIPER_PDFLIP_RESUME_WARM_FINISH": "1",
    "FLLIPER_PDFLIP_TAIL_STAGE_WORKER": "1",
}
#: a per-group value that overrides the row's form-wide default on ONE group
#: (the row's global value is what the other group runs): STORE_SHORT_TAIL, row
#: off (= P), D on (SCHALTER-HALBPORT 1002).
GROUP_OVER_GLOBAL = {"FLLIPER_PDFLIP_STORE_SHORT_TAIL": ("D", False)}
FORM_A_D ="--max-total-tokens 262144 --rank-role host,worker,worker --rank-tp-ratio 1,0,0"


def _ns(profile, env_p="", env_d="", extra_d=FORM_A_D, **kw):
    base = dict(profile=profile, env_p=env_p, env_d=env_d, extra_d=extra_d, teardown=False,
                d_kv_token_cut="off", form_kv=None, d_only=False, pdflip_disable_hicache=False)
    base.update(kw)
    return argparse.Namespace(**base)


def _clean_environ():
    keys = set(PROFILE_P) | set(PROFILE_D)
    return {k: v for k, v in os.environ.items() if k not in keys}


class TestRegistryRows(CustomTestCase):
    def test_nextflash_carries_the_metal_values_per_group(self):
        row = F.PROFILES[F.PROFILE_NEXTFLASH]
        self.assertEqual(dict(row.group_switch_defaults["P"]), PROFILE_P)
        self.assertEqual(dict(row.group_switch_defaults["D"]), PROFILE_D)
        self.assertEqual(row.d_kv_token_cut, "owned")

    def test_qwen27b_row_is_unchanged(self):
        row = F.PROFILES[F.PROFILE_QWEN27B]
        self.assertEqual(dict(row.group_switch_defaults), {})
        self.assertEqual(row.d_kv_token_cut, F.KV_TOKEN_CUT_OFF)

    def test_switch_defaults_do_not_carry_them(self):
        """Global (form-wide) defaults would reach both groups and the front:
        none of the per-group switches is in PROFILE_SWITCH_DEFAULTS."""
        for pid in F.PROFILES:
            got = set(F.PROFILE_SWITCH_DEFAULTS[pid])
            self.assertFalse(got & (set(PROFILE_P) | set(PROFILE_D)) - set(GROUP_OVER_GLOBAL), pid)
        # the one deliberate override: the row's global value is what the
        # group that does not state it runs
        groups = F.PROFILES[F.PROFILE_NEXTFLASH].group_switch_defaults
        for name, (group, global_value) in GROUP_OVER_GLOBAL.items():
            self.assertIs(F.PROFILE_SWITCH_DEFAULTS[F.PROFILE_NEXTFLASH][name], global_value)
            self.assertEqual([g for g in groups if name in groups[g]], [group])


class TestLauncherGroupDefaults(CustomTestCase):
    def test_nf_empty_envs_get_the_metal_values(self):
        ns = _ns(F.PROFILE_NEXTFLASH)
        line = L.apply_profile_group_switch_defaults(ns, environ=_clean_environ())
        self.assertIn(L.GROUP_SWITCH_DEFAULTS_MARKER, line)
        self.assertEqual(L.parse_group_env(ns.env_p), PROFILE_P)
        self.assertEqual(L.parse_group_env(ns.env_d), PROFILE_D)

    def test_stated_and_exported_values_win(self):
        ns = _ns(F.PROFILE_NEXTFLASH, env_p="FLLIPER_PDFLIP_ENABLE_P_TAIL_FOLD=0",
                 env_d="FLLIPER_MOE_SCRATCH_SLOTS=74,48,48;FLLIPER_PDFLIP_ENABLE_D_PARK_END=0")
        env = dict(_clean_environ(), FLLIPER_HC_MIXER_INT8="0")
        L.apply_profile_group_switch_defaults(ns, environ=env)
        p, d = L.parse_group_env(ns.env_p), L.parse_group_env(ns.env_d)
        self.assertEqual(p["FLLIPER_PDFLIP_ENABLE_P_TAIL_FOLD"], "0")
        self.assertEqual(d["FLLIPER_PDFLIP_ENABLE_D_PARK_END"], "0")
        self.assertEqual(d["FLLIPER_MOE_SCRATCH_SLOTS"], "74,48,48")
        self.assertTrue(ns.env_d.startswith("FLLIPER_MOE_SCRATCH_SLOTS=74,48,48;FLLIPER_PDFLIP_ENABLE_D_PARK_END=0"))
        self.assertNotIn("FLLIPER_HC_MIXER_INT8", p)
        self.assertNotIn("FLLIPER_HC_MIXER_INT8", d)

    def test_running_profile_envs_are_already_complete(self):
        """The running profile states every one: nothing is added (the boot's
        argv stays what the metal ran)."""
        env_p = ";".join(f"{k}={v}" for k, v in PROFILE_P.items())
        env_d = ";".join(f"{k}={v}" for k, v in PROFILE_D.items())
        ns = _ns(F.PROFILE_NEXTFLASH, env_p=env_p, env_d=env_d)
        self.assertIsNone(L.apply_profile_group_switch_defaults(ns, environ=_clean_environ()))
        self.assertEqual((ns.env_p, ns.env_d), (env_p, env_d))

    def test_qwen27b_envs_byte_identical(self):
        ns = _ns(F.PROFILE_QWEN27B, env_p="A=1", env_d="B=2;C=3")
        self.assertIsNone(L.apply_profile_group_switch_defaults(ns, environ=_clean_environ()))
        self.assertEqual((ns.env_p, ns.env_d), ("A=1", "B=2;C=3"))


class TestLauncherTokenCutDefault(CustomTestCase):
    ARGV = ["--profile", "nextflash"]

    def test_nf_form_a_takes_owned(self):
        ns = _ns(F.PROFILE_NEXTFLASH)
        line = L.apply_profile_d_kv_token_cut_default(ns, self.ARGV)
        self.assertIn("--d-kv-token-cut owned", line)
        self.assertEqual(ns.d_kv_token_cut, "owned")
        kv, _ = F.derive_kv(FORM_A_D.split(), ns.d_kv_token_cut)
        self.assertEqual(kv, "qsa_forma_dcp")

    def test_given_flag_wins(self):
        ns = _ns(F.PROFILE_NEXTFLASH)
        self.assertIsNone(L.apply_profile_d_kv_token_cut_default(ns, self.ARGV + ["--d-kv-token-cut", "off"]))
        self.assertEqual(ns.d_kv_token_cut, "off")
        ns = _ns(F.PROFILE_NEXTFLASH, d_kv_token_cut="maxmin")
        self.assertIsNone(L.apply_profile_d_kv_token_cut_default(ns, self.ARGV + ["--d-kv-token-cut=maxmin"]))
        self.assertEqual(ns.d_kv_token_cut, "maxmin")

    def test_not_where_the_cut_cannot_run(self):
        for kw in (dict(extra_d="--max-total-tokens 262144"),            # no Form A
                   dict(form_kv="qsa_forma"),                            # stated uncut form
                   dict(pdflip_disable_hicache=True),                      # flip boot w/o host tier
                   dict(env_d="FLLIPER_PDFLIP_D_KV_STAGE_TOKENS=131072,262144"),  # two named stages
                   dict(teardown=True)):
            ns = _ns(F.PROFILE_NEXTFLASH, **kw)
            self.assertIsNone(L.apply_profile_d_kv_token_cut_default(ns, self.ARGV), kw)
            self.assertEqual(ns.d_kv_token_cut, "off", kw)
        ns = _ns(F.PROFILE_NEXTFLASH, d_only=True, pdflip_disable_hicache=True)
        L.apply_profile_d_kv_token_cut_default(ns, self.ARGV)
        self.assertEqual(ns.d_kv_token_cut, "owned")

    def test_qwen27b_stays_off(self):
        ns = _ns(F.PROFILE_QWEN27B)
        self.assertIsNone(L.apply_profile_d_kv_token_cut_default(ns, []))
        self.assertEqual(ns.d_kv_token_cut, "off")


class TestNanGuardDiagnosticsDefaultOff(CustomTestCase):
    """Class (d): FLLIPER_NAN_GUARD_WAVE / _FETCH default 0 -- with the guard
    armed and the switch unset, neither check touches its input; =1 arms it.
    (Both checks swallow every exception, so the probe RECORDS instead of
    raising.)"""

    @staticmethod
    def _probe(seen, tag):
        class _Probe:
            def __getattr__(self, name):
                seen.append(tag)
                raise AttributeError(name)

            def __iter__(self):
                seen.append(tag)
                return iter(())

        return _Probe()

    def _run(self, env):
        from flliper.srt.layers.moe.expert_offload import MoEExpertOffloadCache as C

        seen = []
        base = {k: v for k, v in os.environ.items()
                if k not in ("FLLIPER_NAN_GUARD_WAVE", "FLLIPER_NAN_GUARD_FETCH")}
        base.update(env)
        with mock.patch("flliper.srt.layers.nan_guard.nan_guard_on", return_value=True), \
                mock.patch.dict(os.environ, base, clear=True):
            fake = types.SimpleNamespace(layer=None, _resident={})
            C._nan_guard_wave(fake, self._probe(seen, "wave"), 0, [], 0)
            C._nan_guard_fetched(fake, self._probe(seen, "fetch"))
        return seen

    def test_unset_is_off(self):
        self.assertEqual(self._run({}), [])

    def test_one_arms_each(self):
        self.assertEqual(self._run({"FLLIPER_NAN_GUARD_WAVE": "1", "FLLIPER_NAN_GUARD_FETCH": "1"}),
                         ["wave", "fetch"])


if __name__ == "__main__":
    import unittest

    unittest.main()
