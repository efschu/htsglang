# SPDX-License-Identifier: Apache-2.0
"""rc12e: the D form books the gather ring as a fixed post and the MEASURED activation.

rc12d (227ae1becd) D-TP0 extend transients (WEG2-VRAM-PEAK transient_mib):
chunk rows=4059 1101 MiB, rounds with the draft extend 994 -- above the 1024
MiB the fnFL2 reference books; rc12c/rc12b rounds 1006/934. Every window >= 800
MiB carried an allocator retry. And the gather ring 227ae1becd made persistent
(16 rows x 2.417 MiB = 38.7 MiB per rank) came after the posts line
D_FIXED_MIB was read from, so no post held it.

The solve (plan_d_residency, the real NF checkpoint's geometry) now adds the
ring to the record's fixed post and takes the activation from
``D_ACTIVATION_MIB``; qwen27b has neither record and stays byte-identical.
"""

import importlib.util
import os
import unittest
from pathlib import Path

import pytest

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.planner import expert_residency as ER  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402

NF_MODEL = (
    "/spinning/llm_stuff/club-3090/models-cache/"
    "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"
)
ENV = {
    "SGLANG_UNEVEN_MOE_EXPERT_SHARD": "1", "SGLANG_MOE_OFFLOAD_GRAPH_MODE": "pool",
    "SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL": "1", "SGLANG_WEG2_DRAFT_SHARE_EMBED": "1",
}
#: the launcher's own NF default (H95, apply_profile_d_pool_waves_default)
WAVES = 2


def _rc12c_terms():
    p = Path(__file__).with_name("test_weg2_d_awake_rest_rc12c.py")
    spec = importlib.util.spec_from_file_location("_rc12c_terms_act", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _solve(scratch, *, activation=True, waves=WAVES, derive=True):
    T = _rc12c_terms()
    terms = []
    _c, budgets = T._d_pass("nextflash", [], terms=terms)
    led = L.d_card_ledger(terms, budgets, "D")
    fx, fs = L.d_fixed_record("nextflash")
    ax, as_ = L.d_activation_record("nextflash") if activation else (None, "")
    env = dict(ENV, SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES=str(waves),
               SGLANG_MOE_SCRATCH_SLOTS=",".join(str(s) for s in scratch))
    plan = ER.plan_d_residency(
        model_path=NF_MODEL, budgets_mib=[float(b) for b in budgets], ratios=[183, 137, 168],
        fractions=[0.06, 0.51, 0.48], scratch_rows=list(scratch), rank_tp_ratio="1,0,0",
        env_d=env, reference_logs="", kv_tokens=262144, label="D", marker="M",
        replayssm_spec=ER.ReplaySSMSpecForm(ring_len=16, draft_tokens=4, max_running=6,
                                            ssm_dtype="bfloat16"),
        seats=6, fixed_record_mib=fx, fixed_record_source=fs, card_ledger=led,
        activation_record_mib=ax, activation_record_source=as_, derive_waves=derive)
    return plan, led


needs_nf = pytest.mark.skipif(not os.path.isdir(NF_MODEL), reason="NF checkpoint not on this desk")


class TheRecordsAreMeasured(CustomTestCase):
    def test_activation_record_is_the_measured_maximum(self):
        vals, boots = L.d_activation_record("nextflash")
        self.assertEqual(vals, [1104.0, None, None])      # 1101 rounded up to 8 MiB
        self.assertIn("dkrnfh91bar1dauer09270212", boots)
        self.assertEqual(L.d_activation_record("qwen27b"), (None, ""))

    def test_the_planner_ring_is_the_runtime_ring(self):
        from sglang.srt.layers.moe import expert_offload as EO

        self.assertEqual(ER.GATHER_RING_ROWS, EO.GATHER_RING_ROWS)


@needs_nf
class TheSolveBooksRingAndActivation(CustomTestCase):
    def test_ring_is_in_the_fixed_post(self):
        plan, led = _solve((100, 48, 48), activation=False)
        ring = [l for l in plan.lines if "GATHER-RING (rc12e)" in l]
        self.assertEqual(len(ring), 1)
        self.assertIn("['38.7', '38.7', '38.7'] MiB je Rang (16 Zeilen x 2.417 MiB", ring[0])
        self.assertAlmostEqual(plan.fits[0].fixed_mib, 7442.0 + 38.7, delta=0.1)
        # the ring alone keeps the rc12d form: scratch 91, two waves, trim 1791
        capped = L.d_scratch_cap(plan.fits, [100, 48, 48])[0]
        self.assertEqual(capped, [91, 48, 48])
        plan2, _ = _solve(tuple(capped), activation=False)
        self.assertIsNone(plan2.refusal)
        self.assertEqual(plan2.overflow_waves, 2)
        self.assertIn("OVERFLOW-WAVES rang0: rows 181 / scratch 91 -> 2",
                      "\n".join(plan2.lines))
        self.assertEqual(L.d_extend_trim_env(led, plan2.fits), "1791,1724,1725")

    def test_activation_moves_tp0_scratch_91_to_90(self):
        plan, led = _solve((100, 48, 48))
        act = [l for l in plan.lines if "AKTIVIERUNG (rc12e)" in l][0]
        self.assertIn("['1104', '1024', '1024'] statt ['1024', '1024', '1024']", act)
        capped, lines = L.d_scratch_cap(plan.fits, [100, 48, 48])
        self.assertEqual(capped, [90, 48, 48], lines)
        plan2, _ = _solve(tuple(capped))
        self.assertIsNone(plan2.refusal)
        for f in plan2.fits:
            self.assertEqual(f.verdict, "PASST", f)
            self.assertGreaterEqual(f.kv_tokens, 262144)
        self.assertEqual(plan2.fits[0].activation_mib, 1104.0)
        # the waves follow the solved form: 181 rows / scratch 90 -> 3
        self.assertEqual(plan2.overflow_waves, 3)
        self.assertIn("OVERFLOW-WAVES rang0: rows 181 / scratch 90 -> 3", "\n".join(plan2.lines))
        # the trim threshold = floor + booked activation; the ring is persistent,
        # never free, so it is not in the threshold
        self.assertEqual(L.d_extend_trim_env(led, plan2.fits), "1871,1724,1725")

    def test_told_waves_are_kept_and_still_refused(self):
        # a value told in --env-d is not derived: 181 rows > 2 x 90 -> W-SITZE
        plan, _ = _solve((90, 48, 48), derive=False)
        self.assertIsNone(plan.overflow_waves)
        self.assertIn("W-SITZE", plan.refusal)
        self.assertIn("SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES", plan.refusal)
        self.assertNotIn("OVERFLOW-WAVES rang", "\n".join(plan.lines))


class TheTwentySevenBStaysByteIdentical(CustomTestCase):
    def test_no_records_no_kwargs(self):
        self.assertEqual(L.d_fixed_record("qwen27b"), (None, ""))
        self.assertEqual(L.d_activation_record("qwen27b"), (None, ""))

    def test_only_the_launchers_own_default_is_derived(self):
        import types

        told = types.SimpleNamespace(profile="nextflash", env_d="SGLANG_OPT_MOE_POOL_OVERFLOW_WAVES=2")
        from unittest import mock

        with mock.patch.object(L, "profile_standard_form", lambda ns: True):
            self.assertIsNone(L.apply_profile_d_pool_waves_default(told))
            self.assertFalse(getattr(told, "d_pool_waves_derived", False))
            dflt = types.SimpleNamespace(profile="nextflash", env_d="")
            self.assertIn("D-POOL-WELLEN", L.apply_profile_d_pool_waves_default(dflt))
            self.assertTrue(dflt.d_pool_waves_derived)
        q = types.SimpleNamespace(profile="qwen27b", env_d="")
        with mock.patch.object(L, "profile_standard_form", lambda ns: False):
            self.assertIsNone(L.apply_profile_d_pool_waves_default(q))
        self.assertFalse(getattr(q, "d_pool_waves_derived", False))


if __name__ == "__main__":
    unittest.main()
