# SPDX-License-Identifier: Apache-2.0
"""L15-POOL S1b: the per-card record P_AWAKE_PEAK_MIB (hermetic, stdlib only).

Cases: the record is the MAXIMUM of the measured windows (never a mean, over
boots and lines, with its origin); no record = no pool share on that card; the
switch off = the planner input byte for byte the legacy one (nothing read,
nothing printed); the CLI builds the same record from a log directory.
"""

import importlib.util
import json
import os
import re
import tempfile
from types import SimpleNamespace

from flliper.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import l15_plan as LP  # noqa: E402
from flliper.srt.pdflip import l15_pool_peak as PK  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *([os.pardir] * 4)))


def _ln(rank, phase, res, alloc=None, total=32088, free=3000, ts="2026-10-04 08:11:49", na=False):
    pa = "na" if na else (alloc if alloc is not None else res - 500)
    pr = "na" if na else res
    return (f"[{ts} PP{rank}] PDFLIP-VRAM-PEAK rank={rank} phase={phase} rows=1024 n=1 "
            f"t0_unix_ms=1 t_unix_ms=2 window_ms=3 peak_allocated_mib={pa} peak_reserved_mib={pr} "
            f"start_allocated_mib=1 transient_mib=2 allocated_mib=3 reserved_mib=4 "
            f"card_free_start_mib=5 card_free_mib={free} card_total_mib={total} "
            f"alloc_retries=0 ooms=0 alloc_retries_total=0")


CARDS = [SimpleNamespace(total_mib=32607, reserved_mib=519),
         SimpleNamespace(total_mib=20480, reserved_mib=425),
         SimpleNamespace(total_mib=20480, reserved_mib=425)]
SRC = [("bootA", [_ln(0, "chunk", 100), _ln(0, "chunk", 300, ts="2026-10-04 08:12:00"),
                  _ln(0, "chunk", 200), _ln(1, "chunk", 50, total=20055)]),
       ("bootB", [_ln(0, "round", 250, ts="2026-10-04 09:00:00"),
                  _ln(2, "idle", 70, total=20055)])]


class MaximumNotMean(CustomTestCase):
    def test_line_parse(self):
        r = PK.parse_peak_line(_ln(1, "chunk", 8138, total=20055))
        self.assertEqual((r["rank"], r["phase"], r["peak_reserved_mib"], r["card_total_mib"]),
                         (1, "chunk", 8138, 20055))
        self.assertIsNone(PK.parse_peak_line("[x] something else"))

    def test_record_is_the_max_with_origin(self):
        rec = PK.build_record(SRC)
        c0 = rec["cards"]["0"]
        # max of 100,300,200,250 = 300; the mean would be 212.5
        self.assertEqual(c0["peak_mib"], 300)
        self.assertEqual(c0["source"], {"boot": "bootA", "t": "2026-10-04 08:12:00", "line": 2,
                                        "phase": "chunk"})
        self.assertEqual(c0["n"], 4)
        self.assertEqual(c0["per_boot"], {"bootA": {"peak_mib": 300, "n": 3},
                                          "bootB": {"peak_mib": 250, "n": 1}})
        self.assertEqual(rec["cards"]["1"]["peak_mib"], 50)
        self.assertEqual(rec["cards"]["2"]["peak_mib"], 70)
        self.assertEqual(rec["aggregate"], "max")

    def test_max_across_boots_not_last_boot(self):
        rec = PK.build_record([("a", [_ln(0, "chunk", 900)]), ("b", [_ln(0, "chunk", 400)])])
        self.assertEqual(rec["cards"]["0"]["peak_mib"], 900)
        self.assertEqual(rec["cards"]["0"]["source"]["boot"], "a")

    def test_flip_and_na_lines_do_not_count(self):
        rec = PK.build_record([("a", [_ln(0, "flip", 99999), _ln(0, "chunk", 0, na=True),
                                      _ln(0, "chunk", 400)])])
        c = rec["cards"]["0"]
        self.assertEqual((c["peak_mib"], c["n"], c["n_skipped"]), (400, 1, 2))

    def test_no_measured_line_means_no_entry(self):
        rec = PK.build_record([("a", [_ln(0, "flip", 500), "noise"])])
        self.assertEqual(rec["cards"], {})

    def test_basis_allocated(self):
        rec = PK.build_record([("a", [_ln(0, "chunk", 900, alloc=800)])], basis="peak_allocated_mib")
        self.assertEqual(rec["cards"]["0"]["peak_mib"], 800)
        with self.assertRaises(ValueError):
            PK.build_record([], basis="mean")


class NoRecordNoShare(CustomTestCase):
    def _totals(self):
        return [(c.total_mib, c.total_mib - c.reserved_mib) for c in CARDS]

    def test_no_record_zero_share_everywhere(self):
        peaks, lines = PK.resolve_peaks(None, "qwen27b", self._totals())
        self.assertEqual(peaks, [None, None, None])
        self.assertEqual(lines[0], "L15-POOL-PEAK card=0 peak_mib=none source=NO-RECORD n=0")
        posts = LP.resolve_posts("qwen27b", [20000, 15000, 15000], peaks, {"FLLIPER_PDFLIP_L15": "1"})
        self.assertEqual([(p.mib, p.src) for p in posts], [(0, "UNMEASURED")] * 3)

    def test_card_without_entry_gets_zero_others_priced(self):
        rec = PK.build_record([("a", [_ln(0, "chunk", 26000), _ln(2, "chunk", 12000, total=20055)])])
        peaks, lines = PK.resolve_peaks(rec, "qwen27b", self._totals())
        self.assertEqual(peaks, [26000, None, 12000])
        self.assertIn("card=1 peak_mib=none source=NO-RECORD n=0", lines[1])
        self.assertRegex(lines[0], r"^L15-POOL-PEAK card=0 peak_mib=26000 source=RECORD\(\S+\) n=1$")
        posts = LP.resolve_posts("qwen27b", [30000, 15000, 15000], peaks, {"FLLIPER_PDFLIP_L15": "1"})
        self.assertEqual([p.mib for p in posts], [4000, 0, 3000])
        self.assertEqual([p.src for p in posts],
                         ["RECORD(P_AWAKE_PEAK_MIB)", "UNMEASURED", "RECORD(P_AWAKE_PEAK_MIB)"])

    def test_foreign_card_or_profile_is_never_priced(self):
        rec = PK.build_record([("a", [_ln(0, "chunk", 26000, total=20055)])])  # record: 20055 card on ordinal 0
        peaks, lines = PK.resolve_peaks(rec, "qwen27b", self._totals())
        self.assertIsNone(peaks[0])
        self.assertIn("CARD-MISMATCH", lines[0])
        rec2 = PK.build_record([("a", [_ln(0, "chunk", 26000)])], profile="nextflash")
        peaks2, lines2 = PK.resolve_peaks(rec2, "qwen27b", self._totals())
        self.assertEqual(peaks2, [None] * 3)
        self.assertIn("PROFILE-MISMATCH", lines2[0])

    def test_nvml_total_also_matches(self):
        rec = PK.build_record([("a", [_ln(0, "chunk", 26000, total=32607)])])
        self.assertEqual(PK.resolve_peaks(rec, "qwen27b", self._totals())[0][0], 26000)


class SwitchOffIsByteForByte(CustomTestCase):
    def _boom(self, path):
        raise AssertionError("switch off must not read any record")

    def test_default_off_and_spellings(self):
        self.assertFalse(PK.switch_on({}))
        self.assertFalse(PK.switch_on({PK.SWITCH_ENV: "0"}))
        for v in ("1", "true", "ON"):
            self.assertTrue(PK.switch_on({PK.SWITCH_ENV: v}))

    def test_off_returns_legacy_peaks_untouched_and_prints_nothing(self):
        for legacy in ([None, None, None], [26000, 8000, 12000]):
            peaks, lines = PK.planner_peaks("qwen27b", CARDS, legacy, {"FLLIPER_PDFLIP_L15": "1"},
                                            load=self._boom)
            self.assertEqual((peaks, lines), (legacy, []))
            budgets = [30000, 15000, 15000]
            env = {"FLLIPER_PDFLIP_L15": "1"}
            self.assertEqual(LP.resolve_posts("qwen27b", budgets, peaks, env),
                             LP.resolve_posts("qwen27b", budgets, legacy, env))

    def test_on_ignores_legacy_and_uses_only_the_pool_record(self):
        rec = PK.build_record([("a", [_ln(0, "chunk", 26000)])])
        env = {PK.SWITCH_ENV: "1"}
        peaks, lines = PK.planner_peaks("qwen27b", CARDS, [1, 2, 3], env, load=lambda p: rec)
        self.assertEqual(peaks, [26000, None, None])
        self.assertEqual(len(lines), 3)

    def test_on_without_a_file_is_no_share_not_a_crash(self):
        env = {PK.SWITCH_ENV: "1", PK.FILE_ENV: "/nonexistent/l15_pool_peak.json"}
        peaks, lines = PK.planner_peaks("qwen27b", CARDS, [9, 9, 9], env)
        self.assertEqual(peaks, [None] * 3)
        self.assertTrue(all("source=NO-RECORD" in ln for ln in lines))

    def test_launcher_wiring_is_the_gated_call_before_resolve_posts(self):
        src = open(os.path.join(ROOT, "python/flliper/srt/pdflip/launcher.py"), encoding="utf-8").read()
        i = src.index("l15_pool_peak.planner_peaks(")
        j = src.index("l15_plan.resolve_posts(", i)
        self.assertLess(i, j)
        self.assertLess(j - i, 900)
        self.assertEqual(src.count("l15_pool_peak.planner_peaks("), 1)

    def test_environ_declares_both_with_default_off(self):
        from flliper.srt.environ import envs
        self.assertIs(envs.FLLIPER_PDFLIP_L15_POOL_PEAK_RECORD.get(), False)
        self.assertEqual(envs.FLLIPER_PDFLIP_L15_POOL_PEAK_RECORD_FILE.get(), "")


class FileAndCli(CustomTestCase):
    def test_json_roundtrip_through_load_record(self):
        rec = PK.build_record(SRC)
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "r.json")
            with open(p, "w") as fh:
                json.dump(rec, fh)
            self.assertEqual(PK.load_record(p), rec)
            self.assertIsNone(PK.load_record(os.path.join(d, "missing.json")))
            with open(p, "w") as fh:
                fh.write("{not json")
            self.assertIsNone(PK.load_record(p))
            with open(p, "w") as fh:
                json.dump({"record": "OTHER", "schema": 1}, fh)
            self.assertIsNone(PK.load_record(p))

    def test_cli_builds_the_same_record_from_a_directory(self):
        spec = importlib.util.spec_from_file_location(
            "cli", os.path.join(ROOT, "scripts", "l15_pool_peak_record.py"))
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        with tempfile.TemporaryDirectory() as d:
            for boot, lines in SRC:
                with open(os.path.join(d, f"boot_{boot}.P.log"), "w") as fh:
                    fh.write("\n".join(lines) + "\n")
            out = os.path.join(d, "rec.json")
            self.assertEqual(cli.main([d, "--write", out]), 0)
            got = json.load(open(out))
        self.assertEqual({k: v["peak_mib"] for k, v in got["cards"].items()},
                         {"0": 300, "1": 50, "2": 70})
        self.assertEqual(got["cards"]["0"]["source"]["boot"], "boot_bootA")
