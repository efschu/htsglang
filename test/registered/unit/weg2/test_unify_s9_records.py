"""UNIFY_PLAN Schritt 9: measured values come from RECORDS, a record holds only
for its power limit (row 27, Agent PL) and -- a per-stage table -- only for
its P cut (Agent PG); docker/profiles are checked against the registry.

Pins:
  * the registry constants from records are the literals they replaced
    (byte-identical argv on both profiles);
  * the NF row borrows exactly the 17 rows it borrowed before, by name;
  * power/cut filters (select, CalibrationIdentity power term);
  * the D speed constants from records reproduce the old literal fits (INT8,
    NVFP4); FP8 now has its own one-point record (dkr27bfp8bar1final09260250)
    and names what it still borrows;
  * the builtin P chunk model is a record with limit and cut; the chunk
    model's cut check (match / unknown / JSON mismatch refused / builtin warn);
  * profile_docker renders and diffs.
"""

import json
import os
import tempfile
import types
import unittest

from sglang.srt.weg2 import d_reshard as D
from sglang.srt.weg2 import form as F
from sglang.srt.weg2 import p_stage_model as M
from sglang.srt.weg2 import profile_docker as PD
from sglang.srt.weg2 import profile_records as R

try:
    from sglang.srt.weg2 import launcher as L
except Exception:  # noqa: BLE001
    L = None

RIG = {"RTX5090": (400.0,), "RTX3080": (230.0, 230.0)}

#: the literals of weg2/form.py _QWEN27B_CONSTANTS before Schritt 9
OLD_27B = {
    "DC_MEASURED_D_5090_MIB": 2228, "DC_MEASURED_D_3080_MIB": 1922,
    "DC_MEASURED_D_XCHG_MIB": (2588, 3084, 2588), "P_OVERSHOOT_MIB": (920, 0, 512),
    "D_OVERSHOOT_MIB": (489, 0, 0), "P_DRAFT_RESIDENT_BUDGET_MIB": 405.2 + 1213.0,
    "CALIBRATION_LAYERS": 64, "MEASURED_MS_PER_LAYER": "8.10,35.16,33.59",
    "P_PP_STAGE_FIXED_MIB": "2342.0,1105.5,3518.0", "P_MAMBA_MIB_PER_LINEAR_LAYER_PER_SLOT": 1.5588,
    "X_RECORDED_R_D_TOKS": 690.0, "X_RECORDED_R_P_TOKS": 3640.0, "X_RECORDED_FLIP_S": 13.247,
    "STORE_CENSUS_PROVENANCE": "boot weg2sb5g W9 store census 2026-09-09T07:12:38Z",
    "STORE_CENSUS_KV_PAGES": 50651, "STORE_CENSUS_MAMBA_BLOBS": 42, "STORE_CENSUS_DRAFT_PAGES": 26040,
    "STORE_CENSUS_KV_PAGE_BYTES": 32768,
}


class TestRecordsAreTheOldLiterals(unittest.TestCase):
    def test_qwen27b_constants_unchanged(self):
        row = F.PROFILES["qwen27b"]
        self.assertEqual(set(row.constants), set(OLD_27B))
        for n, v in OLD_27B.items():
            got = row.constant(n)
            self.assertEqual(got, v, n)
            self.assertEqual(type(got), type(v), n)
            self.assertEqual(row.constants[n].measured_on, "qwen27b")

    def test_nextflash_own_and_borrowed(self):
        row = F.PROFILES["nextflash"]
        self.assertEqual(row.constant("P_DRAFT_RESIDENT_BUDGET_MIB"), 615.7 + 1522.7)
        self.assertEqual(row.constants["P_DRAFT_RESIDENT_BUDGET_MIB"].measured_on, "nextflash")
        got = dict(F.borrowed_constants("nextflash"))
        self.assertEqual(set(got), set(OLD_27B) - {"P_DRAFT_RESIDENT_BUDGET_MIB"})
        self.assertEqual(set(got.values()), {"qwen27b"})
        for n in got:
            self.assertEqual(row.constant(n), OLD_27B[n], n)
        self.assertIn("17 constant row(s) BORROWED", F.borrowed_constants_line("nextflash"))

    def test_launcher_aliases(self):
        if L is None:
            self.skipTest("launcher unavailable")
        self.assertEqual(L.P_OVERSHOOT_MIB, [920, 0, 512])
        self.assertEqual(L.P_DRAFT_RESIDENT_BUDGET_MIB, 1618.2)
        self.assertEqual(L.P_CHUNK_BUILTIN_INT8, (
            {"a_ms": 43.94, "b_ms_per_1k": 1.022, "fwd_overhead_ms": 3.73, "eager_floor_ms": 63.0},
            {"a_ms": 36.87, "b_ms_per_1k": 0.922, "fwd_overhead_ms": 0.0, "eager_floor_ms": 42.0},
            {"a_ms": 40.41, "b_ms_per_1k": 0.995, "fwd_overhead_ms": 0.0, "eager_floor_ms": 74.0},
        ))


class TestFilters(unittest.TestCase):
    def test_power_verdict(self):
        rec = {"RTX5090": 400.0, "RTX3080": 230.0}
        self.assertEqual(R.power_verdict(rec, RIG), R.POWER_MATCH)
        self.assertEqual(R.power_verdict(rec, {"RTX5090": (600.0,), "RTX3080": (230.0,)}), R.POWER_MISMATCH)
        self.assertEqual(R.power_verdict(rec, {"RTX5090": 410.0}), R.POWER_MATCH)  # inside 5 %
        self.assertEqual(R.power_verdict(None, RIG), R.POWER_UNKNOWN)
        self.assertEqual(R.power_verdict(rec, None), R.POWER_UNKNOWN)
        self.assertEqual(R.power_verdict(rec, {"A100": (300.0,)}), R.POWER_UNKNOWN)

    def _dir(self, recs):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "m.json"), "w") as fh:
            json.dump({"profile": "m", "records": recs}, fh)
        return d

    def test_select_prefers_the_running_limit_and_drops_a_foreign_cut(self):
        d = self._dir([
            {"name": "T", "value": 1, "provenance": "a", "kind": "timing",
             "power_limit_w": {"RTX5090": 600, "RTX3080": 320}},
            {"name": "T", "value": 2, "provenance": "b", "kind": "timing",
             "power_limit_w": {"RTX5090": 400, "RTX3080": 230}},
            {"name": "C", "value": 3, "provenance": "c", "pp_layer_ratio": "42,11,11"},
        ])
        s = R.select("m", "T", current_power=RIG, records_dir=d)
        self.assertEqual((s.record.value, s.power), (2, R.POWER_MATCH))
        hi = {"RTX5090": (600.0,), "RTX3080": (320.0, 320.0)}
        self.assertEqual(R.select("m", "T", current_power=hi, records_dir=d).record.value, 1)
        only_foreign = self._dir([{"name": "T", "value": 1, "provenance": "a",
                                   "power_limit_w": {"RTX5090": 600}}])
        loose = R.select("m", "T", current_power=RIG, records_dir=only_foreign)
        self.assertEqual(loose.power, R.POWER_MISMATCH)  # returned, and says so
        self.assertIsNone(R.select("m", "T", current_power=RIG, strict_power=True,
                                   records_dir=only_foreign))
        self.assertEqual(R.select("m", "C", cut=(42, 11, 11), records_dir=d).cut, R.CUT_MATCH)
        self.assertIsNone(R.select("m", "C", cut=(41, 12, 11), records_dir=d))
        self.assertEqual(R.select("m", "C", records_dir=d).cut, R.CUT_UNKNOWN)

    def test_a_row_is_never_measured_and_borrowed(self):
        d = tempfile.mkdtemp()
        json.dump({"profile": "a", "records": [{"name": "X", "value": 1, "provenance": "p"}]},
                  open(os.path.join(d, "a.json"), "w"))
        json.dump({"profile": "b", "records": [{"name": "X", "value": 2, "provenance": "q"}],
                   "borrow": {"from": "a", "names": ["X"]}}, open(os.path.join(d, "b.json"), "w"))
        with self.assertRaises(R.RecordError):
            R.records("b", d)


class TestCalibrationIdentityPowerTerm(unittest.TestCase):
    def _log(self, d, name, lines):
        p = os.path.join(d, name)
        with open(p, "w") as fh:
            fh.write("\n".join(lines) + "\n")
        return p

    def test_foreign_limit_log_does_not_count(self):
        d = tempfile.mkdtemp()
        pl = lambda w5, w3: [  # noqa: E731
            f"[2026-09-26 12:00:00 PP0] POWER-LIMIT rank tp_rank=0 pp_rank=0 gpu_id=0 nvml_index=1 "
            f"card=RTX5090 power_limit_w={w5} power_limit_default_w=575",
            f"[2026-09-26 12:00:00 PP1] POWER-LIMIT rank tp_rank=0 pp_rank=1 gpu_id=0 nvml_index=0 "
            f"card=RTX3080 power_limit_w={w3} power_limit_default_w=320"]
        same = self._log(d, "same.P.log", pl(400, 230))
        foreign = self._log(d, "foreign.P.log", pl(575, 320))
        old = self._log(d, "old.P.log", ["no power lines here"])
        ident = F.CalibrationIdentity("/m/x", d, None, ("checkpoint", "power_limit"),
                                      power_limits=tuple(sorted(RIG.items())))
        self.assertTrue(ident.uses_power_limit)
        self.assertTrue(ident.power_ok(same))
        self.assertFalse(ident.power_ok(foreign))
        self.assertTrue(ident.power_ok(old))  # predates row 27: not filtered
        self.assertIn("power limit within 5 %", ident.describe())
        blind = F.CalibrationIdentity("/m/x", d, None, ("checkpoint", "power_limit"))
        self.assertFalse(blind.uses_power_limit)
        self.assertTrue(blind.power_ok(foreign))
        self.assertIn("declared, not filtered: power_limit", blind.describe())

    def test_both_profiles_key_the_power_limit(self):
        for p in ("qwen27b", "nextflash"):
            self.assertIn("power_limit", F.PROFILES[p].records.fields)


def _old_int8():
    g = D.rc9_geometry("int8")
    a = D.rank_bytes(g, D.Vector((58, 25, 25)))
    b = D.rank_bytes(g, D.Vector((3465, 2154, 2128)))
    e0, f0 = D.fit_two_point(a[0], 18.3, b[0], 16.3)
    e1, f1 = D.fit_two_point((a[1] + a[2]) / 2, (21.9 + 21.8) / 2, (b[1] + b[2]) / 2, (23.8 + 23.0) / 2)
    s = [x / sum(a) for x in a]
    pc = (256.8, 365.2, 368.4)
    return (e0, e1, e1), (f0, f1, f1), tuple(0.35 * c for c in pc), tuple(0.65 * c / s[r] for r, c in enumerate(pc))


class TestDSpeedRecords(unittest.TestCase):
    def test_int8_is_the_old_fit(self):
        c = D.rc9_calib("int8")
        e, f, pff, pfk = _old_int8()
        self.assertEqual((c.e_gbs, c.f_ms, c.pf_fixed_ms, c.pf_k_ms), (e, f, pff, pfk))
        self.assertEqual((c.beta, c.attn_ms_per_ktok, c.w1_ms, c.w_slope_ms, c.pf_wait_ms),
                         ((0.05, 0.07, 0.07), (0.0264, 0.0468, 0.0468), 9.1, 3.8, 1584.7))
        self.assertEqual(c.power_limit_w, (400.0, 230.0, 230.0))

    def test_nvfp4_is_the_old_one_point(self):
        i8, c = D.rc9_calib("int8"), D.rc9_calib("nvfp4")
        g = D.rc9_geometry("nvfp4")
        a = D.rank_bytes(g, D.Vector((58, 25, 25)))
        s = [x / sum(a) for x in a]
        pt = (14.6, 17.9, 17.6)
        self.assertEqual(c.f_ms, tuple(pt[r] - (a[r] / 1e9) * 1000.0 / i8.e_gbs[r] for r in range(3)))
        pc = (294.9, 807.4, 802.3)
        self.assertEqual(c.pf_fixed_ms, tuple(0.35 * x for x in pc))
        self.assertEqual(c.pf_k_ms, tuple(0.65 * x / s[r] for r, x in enumerate(pc)))
        self.assertEqual((c.e_gbs, c.w1_ms, c.pf_wait_ms), (i8.e_gbs, 9.2, 1585.6))

    def test_fp8_has_its_own_point_and_names_the_rest(self):
        i8, c = D.rc9_calib("int8"), D.rc9_calib("fp8")
        g = D.rc9_geometry("fp8")
        a = D.rank_bytes(g, D.Vector((58, 25, 25)))
        pt = (20.6, 26.2, 25.6)
        self.assertEqual(c.f_ms, tuple(pt[r] - (a[r] / 1e9) * 1000.0 / i8.e_gbs[r] for r in range(3)))
        self.assertEqual(c.w1_ms, 9.8)
        self.assertEqual((c.pf_fixed_ms, c.pf_k_ms, c.pf_wait_ms), (i8.pf_fixed_ms, i8.pf_k_ms, i8.pf_wait_ms))
        self.assertIn("dkr27bfp8bar1final09260250", c.provenance)
        self.assertIn("prefill BORROWED from INT8", c.provenance)
        self.assertNotEqual(c.f_ms, i8.f_ms)  # no longer INT8's constants
        self.assertEqual(D.advisory_calib("fp8").f_ms, c.f_ms)

    def test_advisory_names_borrowed_parts(self):
        cfg = dict(D._QWEN38_27B)
        lines = D.speed_advisory_lines(cfg, "fp8", (58, 25, 25), (87, 26, 23), 136, (26, 19, 19),
                                       current_power_w=(400.0, 230.0, 230.0))
        self.assertIn("BORROWED(INT8 E/beta/attn + D prefill)", lines[0])
        self.assertFalse(any("MISMATCH" in ln for ln in lines))

    def test_foreign_limit_is_named(self):
        lines = D.speed_advisory_lines(dict(D._QWEN38_27B), "compressed-tensors", (58, 25, 25),
                                       (707, 191, 190), 1088, (26, 19, 19),
                                       current_power_w=(600.0, 320.0, 320.0))
        self.assertTrue(any("MISMATCH" in ln for ln in lines))


@unittest.skipIf(L is None, "launcher unavailable")
class TestChunkModelCut(unittest.TestCase):
    def setUp(self):
        self.logged = []
        L._P_CHUNK["spec"] = object()  # a dynamic policy is installed

    def tearDown(self):
        L._P_CHUNK["spec"] = None

    def _ns(self, src):
        return types.SimpleNamespace(p_chunk_model=src)

    def _cut(self, counts):
        return types.SimpleNamespace(layer_counts=counts, gapped=False, layer_set="")

    def test_builtin_record_carries_limit_and_cut(self):
        rec = L._P_CHUNK_BUILTIN_RECORD
        self.assertEqual(rec.pp_layer_ratio, (42, 11, 11))
        self.assertEqual(rec.power_limits(), {"RTX5090": 400.0, "RTX3080": 230.0})
        vals, _ = L.p_chunk_model_power_limits("builtin-int8", 3)
        self.assertEqual(vals, [400.0, 230.0, 230.0])

    def test_builtin(self):
        self.assertEqual(L.p_chunk_model_cut_check(self._ns("builtin-int8"), self._cut((42, 11, 11)),
                                                   self.logged.append), "match")
        self.assertEqual(L.p_chunk_model_cut_check(self._ns("builtin-int8"), self._cut((42, 10, 12)),
                                                   self.logged.append), "mismatch")
        self.assertIn("CUT-MISMATCH WARNING", self.logged[-1])

    def test_json_mismatch_is_refused_unknown_warns(self):
        d = tempfile.mkdtemp()
        with_cut = os.path.join(d, "a.pchunk.json")
        json.dump({"stages": [], "pp_layer_ratio": "49,8,7"}, open(with_cut, "w"))
        old = os.path.join(d, "b.pchunk.json")
        json.dump({"stages": []}, open(old, "w"))
        self.assertEqual(L.p_chunk_model_cut_check(self._ns(with_cut), self._cut((49, 8, 7)),
                                                   self.logged.append), "match")
        with self.assertRaises(SystemExit):
            L.p_chunk_model_cut_check(self._ns(with_cut), self._cut((42, 11, 11)), self.logged.append)
        self.assertEqual(L.p_chunk_model_cut_check(self._ns(old), self._cut((42, 11, 11)),
                                                   self.logged.append), "unknown")
        self.assertIn("CUT-UNKNOWN WARNING", self.logged[-1])

    def test_fixed_policy_checks_nothing(self):
        L._P_CHUNK["spec"] = None
        self.assertEqual(L.p_chunk_model_cut_check(self._ns("builtin-int8"), self._cut((1, 2, 3)),
                                                   self.logged.append), "")
        self.assertEqual(self.logged, [])


class TestPchunkTablesNameTheirCut(unittest.TestCase):
    DATA = os.path.join(os.path.dirname(M.__file__), "p_stage_model_data")

    def test_committed_tables(self):
        for fn in os.listdir(self.DATA):
            if not fn.endswith(".pchunk.json"):
                continue
            cut = fn.split("_cut")[1].split(".")[0].replace("-", ",")
            with open(os.path.join(self.DATA, fn)) as fh:
                self.assertEqual(json.load(fh)["pp_layer_ratio"], cut, fn)

    def test_export_writes_the_cut(self):
        m = M.load_model(os.path.join(self.DATA, "27b_int8_rc9j.json"))
        self.assertEqual(M.pchunk_json(m, (41, 12, 11))["pp_layer_ratio"], "41,12,11")


class TestDockerProfiles(unittest.TestCase):
    def test_render_carries_the_registry(self):
        txt = PD.render("27b-nvfp4", "qwen27b", "nvfp4")
        self.assertIn("PROFILE_MODEL=/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-NVFP4-RadixArk", txt)
        self.assertIn("--pp-stage-ratio 49,8,7", txt)
        self.assertIn("--fp4-native-mixed", txt)
        self.assertIn("_form SGLANG_WEG2_MAMBA_ANCHOR_INTERVAL 4096", txt)
        self.assertIn("GENERATED", txt)

    def _runner(self, form):
        def run(_path):
            reg = {f.key: f.value for f in PD.registry_facts("qwen27b", "int8")}
            lines = [f"VAR\tPROFILE_LINE\t27b", f"VAR\tPROFILE_FORMAT\tint8",
                     f"VAR\tPROFILE_MODEL\t{reg['PROFILE_MODEL']}", f"VAR\tPROFILE_DRAFT\t{reg['PROFILE_DRAFT']}"]
            for a in ("--spec-form", "DFLASH", "--tp-prefill-max-tokens", "4096", "--x-ceiling-tokens", "12288",
                      "--p-chunk-policy", "dynamic"):
                lines.append(f"ARG\t{a}")
            for k, v in form.items():
                lines.append(f"FORM\t{k}\t{v}")
            return "\n".join(lines)
        return run

    def test_check_counts_a_diff(self):
        d = tempfile.mkdtemp()
        open(os.path.join(d, "27b.env"), "w").close()
        lines, n = PD.check_dir(d, runner=self._runner({"SGLANG_WEG2_MAMBA_ANCHOR_INTERVAL": "4096"}))
        self.assertEqual(n, 0, lines)
        lines, n = PD.check_dir(d, runner=self._runner({"SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL": "1"}))
        self.assertEqual(n, 1, lines)
        self.assertTrue(any("DIFF" in ln and "DENSE_REPACK_OUTSIDE_POOL" in ln for ln in lines))

    def test_generate_never_writes_a_profile(self):
        d = tempfile.mkdtemp()
        out = PD.generate(d)
        self.assertTrue(out and all(p.endswith(".registry.env") for p in out))


if __name__ == "__main__":
    unittest.main()
