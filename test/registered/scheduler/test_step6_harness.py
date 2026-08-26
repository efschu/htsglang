# SPDX-License-Identifier: Apache-2.0
"""#631 step-6 harness mock-smoke (the desk-written-never-executed rule).

The card window must be execution-only, so every window script is smoked
here against stubs and -- for the log parser -- against log lines CAPTURED
FROM THE REAL RUNTIME during a real mock flip, so the parser cannot drift
from the format it will meet on cards.
"""

import importlib.util
import json
import logging
import os
import tempfile
import unittest
from pathlib import Path


from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name.replace(".py", ""), _SCRIPTS / name
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestFlipStatsParserAgainstRealFormat(CustomTestCase):
    def test_parser_reads_lines_the_real_runtime_emits(self):
        """Run a REAL 3-rank mock flip with log capture; the step-6 parser
        must extract the kv record from exactly those lines, plus a GDN
        line built by the real mover's logger call signature."""
        from test_phase_flip_runtime import (
            _build_runtimes,
            _make_layout_pools,
            _run_ranks,
        )
        import test_phase_flip_runtime as harness

        bench = _load("route_a_631_step6_bench.py")

        records = []
        handler = logging.Handler()
        handler.emit = lambda rec: records.append(rec.getMessage())
        logger = logging.getLogger("sglang.srt.managers.phase_flip_runtime")
        logger.addHandler(handler)
        old_level = logger.level
        logger.setLevel(logging.INFO)
        try:
            _, live, _, pp_views, _, tp_views = _make_layout_pools(
                harness.MAP_625, list(harness.VEC), num_slots=32
            )
            runtimes, _ = _build_runtimes(pp_views, tp_views, live)
            errors = _run_ranks(
                len(harness.VEC),
                runtimes=runtimes,
                directions=["pp_to_tp"] * len(harness.VEC),
            )
            self.assertEqual([e for e in errors if e], [])
        finally:
            logger.removeHandler(handler)
            logger.setLevel(old_level)

        done_lines = [m for m in records if "DONE" in m]
        self.assertTrue(done_lines, "the real runtime logged no DONE line")
        parsed = bench.parse_flip_stats(done_lines)
        kv = [r for r in parsed if r["kind"] == "kv"]
        self.assertEqual(len(kv), len(done_lines))
        for r in kv:
            self.assertEqual(r["direction"], "pp_to_tp")
            self.assertGreaterEqual(r["total_ms"], 0.0)
            # #905 CONTRACT CHANGE. This line read `>= 1` -- a flip that
            # reported zero slots was a flip whose live-set enumeration had
            # gone blind, and that reading was correct until #856. The seam
            # now retracts every resident and rebuilds the plan on an EMPTY
            # slot tensor before it emits this line, so the honest expectation
            # inverted: EXACTLY ZERO. A DONE line reporting slots is a flip
            # that carried KV, which the no-carry seam forbids -- so the
            # assertion still fails on a blind enumeration (it would have to
            # report a nonzero count to do so) and now also fails on a carry.
            self.assertEqual(
                r["live_slots"],
                0,
                "the runtime reported a non-empty transfer plan at the "
                "cutover; #856 rebuilds it empty and the flip carries no KV",
            )

        # GDN line: emitted through the real mover logger call signature.
        gdn_logger_line = (
            "PHASE-FLIP-GDN moved %d slot(s) %s: sent %.2f MiB, received "
            "%.2f MiB" % (3, "pp_to_tp", 1.23, 4.56)
        )
        parsed = bench.parse_flip_stats([gdn_logger_line])
        self.assertEqual(parsed[0]["kind"], "gdn")
        self.assertEqual(parsed[0]["slots"], 3)
        self.assertEqual(parsed[0]["sent_mib"], 1.23)

    def test_flip_stats_cli_writes_json(self):
        bench = _load("route_a_631_step6_bench.py")
        with tempfile.TemporaryDirectory() as d:
            log = os.path.join(d, "s.log")
            out = os.path.join(d, "o.json")
            open(log, "w").write(
                "noise\nPHASE-FLIP DONE tp_to_pp (epoch 2) in 900.1 ms: 7 "
                "live slots, sent 10 cells / 1.00 MiB, received 12 cells / "
                "2.00 MiB (read 1.0 ms, exchange 2.0 ms, write 3.0 ms)\n"
            )
            rc = bench.main(["flip-stats", "--log", log, "--out", out])
            self.assertEqual(rc, 0)
            data = json.load(open(out))
            self.assertEqual(data[0]["epoch"], 2)
            self.assertEqual(data[0]["write_ms"], 3.0)
            # Pre-wave format: reported as absent, not as 1.
            self.assertIsNone(data[0]["seam_waves"])

    def test_the_waved_done_line_still_parses(self):
        """The waved seam (#631) added a clause mid-line.

        Both formats must parse: the archived pre-wave logs are the
        baseline every flip-time row is compared against.
        """
        bench = _load("route_a_631_step6_bench.py")
        line = (
            "PHASE-FLIP DONE pp_to_tp (epoch 5) in 812.0 ms over 16 seam "
            "wave(s): 270031 live slots, sent 120 cells / 3.00 MiB, "
            "received 140 cells / 4.00 MiB, local 1.00 MiB, staging "
            "reserved 574.00 MiB (read 10.0 ms, exchange 20.0 ms, write "
            "30.0 ms)"
        )
        parsed = bench.parse_flip_stats([line])
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["seam_waves"], 16)
        self.assertEqual(parsed[0]["live_slots"], 270031)
        self.assertEqual(parsed[0]["write_ms"], 30.0)


class TestBenchDriverSmoke(CustomTestCase):
    def test_prefill_ladder_smoke_with_stub_post(self):
        bench = _load("route_a_631_step6_bench.py")
        calls = []

        def _stub_post(url, path, payload, timeout=0):
            calls.append((path, len(payload.get("input_ids", []))))
            return {"meta_info": {"completion_tokens": 1}}

        bench._post = _stub_post
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "ladder.json")
            rc = bench.main(
                [
                    "prefill-ladder",
                    "--lengths",
                    "64",
                    "128",
                    "--draws",
                    "2",
                    "--out",
                    out,
                ]
            )
            self.assertEqual(rc, 0)
            data = json.load(open(out))
            self.assertEqual(set(data["lengths"]), {"64", "128"})
            self.assertIn("floor_pct", data)
            # warm-up + 2xA + 2xB per length
            self.assertEqual(len(calls), 2 * 5)
            # uncached: every draw sends input_ids of the ladder length
            self.assertEqual({n for _, n in calls}, {64, 128})

    def test_decode_smoke_with_stub_post(self):
        bench = _load("route_a_631_step6_bench.py")
        bench._post = lambda *a, **k: {"meta_info": {"completion_tokens": 32}}
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "decode.json")
            rc = bench.main(
                ["decode", "--max-new", "32", "--prompt-tokens", "16", "--out", out]
            )
            self.assertEqual(rc, 0)
            self.assertEqual(json.load(open(out))["completion_tokens"], 32)


class TestTokenExactCompare(CustomTestCase):
    def _write(self, d, name, ids):
        p = os.path.join(d, name)
        json.dump({"token_ids": ids, "text": "", "flip": False}, open(p, "w"))
        return p

    def test_identical_runs_report_token_exact(self):
        te = _load("route_a_631_token_exact.py")
        with tempfile.TemporaryDirectory() as d:
            a = self._write(d, "a.json", [1, 2, 3, 4])
            b = self._write(d, "b.json", [1, 2, 3, 4])
            self.assertEqual(te.compare(a, b), 0)

    def test_divergence_is_reported_with_position(self):
        te = _load("route_a_631_token_exact.py")
        with tempfile.TemporaryDirectory() as d:
            a = self._write(d, "a.json", [1, 2, 3, 4])
            b = self._write(d, "b.json", [1, 2, 9, 4])
            self.assertEqual(te.compare(a, b), 1)

    def test_length_mismatch_and_empty_are_failures(self):
        te = _load("route_a_631_token_exact.py")
        with tempfile.TemporaryDirectory() as d:
            a = self._write(d, "a.json", [1, 2, 3])
            b = self._write(d, "b.json", [1, 2, 3, 4])
            self.assertEqual(te.compare(a, b), 1)
            c = self._write(d, "c.json", [])
            self.assertEqual(te.compare(a, c), 2)


if __name__ == "__main__":
    unittest.main()
