"""VRAM loop 28.09. (user: "VRAM liegt brach ODER OOM -- messen statt raten").

P1b: the D row planner's overhead calibration names an expert-offload D
(NF Form A) as NOT applicable instead of refusing it as a "cell mismatch"
-- that planner books every expert resident, and NF's D budget is the
MEASURED-D budget line + FRACTION-SOLVE D, which read NF's own records.
"""

import json
import os
import tempfile
import unittest

from flliper.srt.pdflip import launcher as L


class _Line:
    def accepts_log(self, path):
        return True


def _write_boot(ev, argv_tail, kv_lines):
    stem = os.path.join(ev, "boot_weg2_nft_abcdef0123_0928_175259")
    with open(stem + ".front.log", "w") as f:
        f.write("[t] PDFLIP-LAUNCH group D argv: /v/python -m flliper.launch_server "
                "--model-path /m " + argv_tail + "\n")
    with open(stem + ".D.log", "w") as f:
        for line in kv_lines:
            f.write(line + "\n")


# NF rc12z26 17:52 D.log: TP0 cell 14143, the two Form A workers cell 0
_NF_KV = [
    "[2026-09-28 17:56:51 TP1] KV pool sizing: available_bytes=1587544064 (1.479 GiB), cell_size=0, page_size=64 -> max_total_num_tokens=1048576",
    "[2026-09-28 17:56:51 TP2] KV pool sizing: available_bytes=595591168 (0.555 GiB), cell_size=0, page_size=64 -> max_total_num_tokens=1048576",
    "[2026-09-28 17:56:51 TP0] KV pool sizing: available_bytes=8013045760 (7.463 GiB), cell_size=14143, page_size=64 -> max_total_num_tokens=566528",
]


class TestExpertOffloadCalibration(unittest.TestCase):
    def test_expert_offload_d_is_named_not_applicable(self):
        with tempfile.TemporaryDirectory() as ev:
            _write_boot(ev, "--rank-gpu-memory-mib 26400,17728,17936 --rank-tp-ratio 1,0,0 "
                            "--max-running-requests 6 --rank-moe-resident-fraction 0.06,0.51,0.48",
                        _NF_KV)
            ovh, why = L.d_overhead_calibration("/m", _Line(), evidence_dir=ev)
        self.assertIsNone(ovh)
        self.assertIn("expert-offload D", why)
        self.assertIn("FRACTION-SOLVE D", why)
        self.assertNotIn("would not transfer", why)



import types as _types


def _plan(ceil, solved=(), owner=()):
    fits = tuple(_types.SimpleNamespace(ceiling_fraction=c) for c in ceil)
    return _types.SimpleNamespace(fits=fits, solved_fractions=tuple(solved),
                                  solved_owner_ratio=tuple(owner))


_GIVEN = [0.06, 0.51, 0.48]
_CEIL = [0.062, 0.586, 0.491]   # NF rc12z26 17:52 front.log FRACTION-SOLVE D
_P0 = {L.TORCH_CACHE_CAP_ENV: "1"}


class TestFrCeilingAdopt(unittest.TestCase):
    def test_edge_wins_with_p0(self):
        new, lines = L.d_fr_ceiling_adopt(_plan(_CEIL), _GIVEN, _P0, owned=False)
        self.assertEqual(new, _CEIL)
        self.assertEqual(len([ln for ln in lines if "rang" in ln]), 3)

    def test_without_p0_the_stated_fr_stays_and_is_named(self):
        new, lines = L.d_fr_ceiling_adopt(_plan(_CEIL), _GIVEN, {}, owned=False)
        self.assertIsNone(new)
        self.assertTrue(any("P0 aus" in ln for ln in lines))

    def test_owned_cut_is_untouched(self):
        new, lines = L.d_fr_ceiling_adopt(_plan(_CEIL, owner=(215, 113, 160)), _GIVEN, _P0,
                                          owned=True)
        self.assertIsNone(new)
        self.assertEqual(lines, [])

    def test_s2b_solved_fr_is_untouched(self):
        new, lines = L.d_fr_ceiling_adopt(_plan(_CEIL, solved=(0.1, 0.5, 0.5)), _GIVEN, _P0,
                                          owned=False)
        self.assertIsNone(new)
        self.assertEqual(lines, [])

    def test_named_cap_binds(self):
        env = dict(_P0, **{L.FR_D_CAP_ENV: "0.07,0.55,0.60"})
        new, _ = L.d_fr_ceiling_adopt(_plan(_CEIL), _GIVEN, env, owned=False)
        self.assertEqual(new, [0.062, 0.55, 0.491])

    def test_raise_only_and_switch_off(self):
        new, lines = L.d_fr_ceiling_adopt(_plan([0.05, 0.586, 0.491]), _GIVEN, _P0, owned=False)
        self.assertEqual(new, [0.06, 0.586, 0.491])
        self.assertTrue(any("Decke unter gegeben" in ln for ln in lines))
        off = dict(_P0, **{L.FR_D_CEILING_ENV: "0"})
        self.assertIsNone(L.d_fr_ceiling_adopt(_plan(_CEIL), _GIVEN, off, owned=False)[0])

    def test_no_edge_on_a_rank_keeps_the_stated_fr(self):
        new, _ = L.d_fr_ceiling_adopt(_plan([0.062, None, 0.491]), _GIVEN, _P0, owned=False)
        self.assertIsNone(new)

    def test_never_after_the_platztausch_map(self):
        # rc12z30b: the map read FR_D before P started; a raised FR_D after it
        # is a D form the map does not describe (rc12z29b died of that class)
        new, lines = L.d_fr_ceiling_adopt(_plan(_CEIL), _GIVEN, _P0, owned=False,
                                          map_built=True)
        self.assertIsNone(new)
        self.assertTrue(any("Platztausch-Karte" in ln for ln in lines))

    def test_launcher_passes_the_map_state(self):
        import inspect
        src = inspect.getsource(L.log_d_rank_vram_solve)
        self.assertIn("map_built=bool(_pinned or getattr(ns, \"_expert_map_path\", \"\"))", src)
        self.assertIn("ns._expert_map_path = _emap", inspect.getsource(L.main))


from flliper.srt.pdflip import torch_cache_cap as TCC


class _FakeCuda:
    def __init__(self):
        self.calls = []

    def mem_get_info(self, dev):
        return (1 << 30, 32607 * (1 << 20))

    def set_per_process_memory_fraction(self, frac, dev):
        self.calls.append((frac, dev))


class _FakeTorch:
    def __init__(self):
        self.cuda = _FakeCuda()


class TestTorchCacheCap(unittest.TestCase):
    def setUp(self):
        self._env = {k: os.environ.get(k) for k in (TCC.ENV, TCC.MIB_ENV, "FLLIPER_PDFLIP_GROUP")}

    def tearDown(self):
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_off_by_default_does_nothing(self):
        os.environ.pop(TCC.ENV, None)
        t = _FakeTorch()
        self.assertIsNone(TCC.arm(0, 0, 0, torch_mod=t))
        self.assertEqual(t.cuda.calls, [])

    def test_d_rank_arms_its_own_tp_entry(self):
        os.environ.update({TCC.ENV: "1", TCC.MIB_ENV: "28413,18237,18289", "FLLIPER_PDFLIP_GROUP": "D"})
        t = _FakeTorch()
        frac = TCC.arm(0, 0, 0, torch_mod=t)
        self.assertAlmostEqual(frac, 28413 / 32607, places=4)
        self.assertEqual(len(t.cuda.calls), 1)

    def test_p_rank_uses_its_pp_ordinal(self):
        os.environ.update({TCC.ENV: "1", TCC.MIB_ENV: "100,200,300", "FLLIPER_PDFLIP_GROUP": "P"})
        self.assertEqual(TCC.rank_index(0, 2), 2)

    def test_missing_entry_is_named_and_unarmed(self):
        os.environ.update({TCC.ENV: "1", TCC.MIB_ENV: "100", "FLLIPER_PDFLIP_GROUP": "D"})
        t = _FakeTorch()
        self.assertIsNone(TCC.arm(2, 0, 0, torch_mod=t))
        self.assertEqual(t.cuda.calls, [])

    def test_launcher_caps_are_verdict_minus_floor(self):
        vs = [_types.SimpleNamespace(available_mib=29180.0),
              _types.SimpleNamespace(available_mib=19056.0),
              _types.SimpleNamespace(available_mib=19062.0)]
        self.assertEqual(TCC.launcher_caps(vs, 767.0), "28413,18289,18295")


from flliper.srt.pdflip import profile_records as PR
from flliper.srt.pdflip import vram_peak_record as VPR

_D_LOG = [
    "[2026-09-28 18:04:21 TP0] #251 WAKE-RESHARD n=6 stage=S1 tokens=393216 demand=- over=no kv_mib=- rows_on=- epoch=1",
    "[2026-09-28 18:04:22 TP0] Decode batch, #running-req: 3, #full token: 1, gen throughput (token/s): 100.0",
    "[2026-09-28 18:04:23 TP0] PDFLIP-VRAM-PEAK rank=0 phase=round n=1 peak_allocated_mib=31000 allocated_mib=30000 reserved_mib=31500 card_free_mib=900",
    "[2026-09-28 18:04:23 TP2] PDFLIP-VRAM-PEAK rank=2 phase=round n=1 peak_allocated_mib=16000 allocated_mib=15800 reserved_mib=17000 card_free_mib=76",
    "[2026-09-28 18:04:24 TP2] PDFLIP-VRAM-PEAK rank=2 phase=chunk rows=25 n=1 peak_allocated_mib=16700 allocated_mib=16000 reserved_mib=17000 card_free_mib=300",
]


class TestVramPeakRecord(unittest.TestCase):
    def test_state_joins_seats_and_stage(self):
        rows = VPR.samples("D", _D_LOG)
        states = {(r["rank"], r["phase"]): r["state"] for r in rows}
        self.assertEqual(states[(0, "round")], "bs3/S1")
        self.assertEqual(states[(2, "round")], "bs3/S0")   # TP2 reported no stage -> S0
        self.assertEqual(states[(2, "chunk")], "chunk")

    def test_aggregate_names_cache_and_slack(self):
        agg = VPR.aggregate(VPR.samples("D", _D_LOG))
        x = agg["D"]["2"]["round"]["bs3/S0"]
        self.assertEqual(x["cache_unused_max"], 1200)       # 17000 - 15800
        self.assertEqual(x["slack_at_peak_min"], 1076)      # 76 + 17000 - 16000
        self.assertEqual(x["card_free_min"], 76)

    def test_written_record_loads_through_the_registry(self):
        with tempfile.TemporaryDirectory() as d:
            stem = os.path.join(d, "boot_weg2_t_abc_0928_180000")
            with open(stem + ".D.log", "w") as f:
                f.write("\n".join(_D_LOG) + "\n")
            rec = VPR.record_for(stem, "nextflash")
            path = os.path.join(d, "nextflash.json")
            with open(path, "w") as f:
                json.dump({"profile": "nextflash", "records": []}, f)
            VPR.merge_into(path, rec)
            with open(path) as f:
                data = json.load(f)
            r = PR.parse_record("nextflash", data["records"][0])
        self.assertEqual(r.name, VPR.RECORD_NAME)
        self.assertEqual(r.value["D"]["0"]["round"]["bs3/S1"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
