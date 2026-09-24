"""FORM-MATRIX-GATE (24.09.): the desk gate before every 27B boot.

Hermetic: fake front logs in a temp dir, no dry run, no evidence directory.
Pinned: argv/env classification (SOLVED / PER-BOOT / FORM / UNEXPLAINED /
CALIBRATION), the reference choice (newest boot OF THIS FORM that reached
front READY), and the calibration-source extraction that turns a moved
source into an explanation instead of a failure. The live runs (27B against
weg2xsn418/xsn411, NF against fnFL2x144 and the base tree) are recorded in
the commit message.
"""

from __future__ import annotations

import os
import tempfile
import time
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import form as F
from sglang.srt.weg2 import form_matrix_gate as G

Q27 = "Qwen3.8-27B-INT8-gdncov-vocabembed"
NF = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"


def _front(ev, tag, stamp, model, algo, ready=True, env_p="SGLANG_A=1;SGLANG_WEG2_BOOT_TOKEN=x",
           extra_p="", form_line=None):
    path = os.path.join(ev, f"boot_weg2_{tag}_abcdef1234_{stamp}.front.log")
    with open(path, "w") as f:
        f.write(f"[t] WEG2-LAUNCH === WEG2 BOOT tag={tag}\n")
        if form_line:
            f.write(f"[t] WEG2-LAUNCH {form_line}\n")
        f.write(f"[t] WEG2-LAUNCH group P argv: /v/python -m sglang.launch_server --model-path /m/{model} "
                f"--speculative-algorithm {algo} --rank-gpu-memory-mib 1,2,3 --page-size 1{extra_p}\n")
        f.write(f"[t] WEG2-LAUNCH WEG2-GROUP-ENV P: {env_p}\n")
        f.write(f"[t] WEG2-LAUNCH group D argv: /v/python -m sglang.launch_server --model-path /m/{model} "
                f"--speculative-algorithm {algo} --tp-size 3\n")
        f.write("[t] WEG2-LAUNCH WEG2-GROUP-ENV D: (leer)\n")
        if ready:
            f.write("[t] WEG2-LAUNCH R1 READY group=front port=30030 after 1.0 s\n")
    return path


class TestFlagMap(unittest.TestCase):
    def test_forms(self):
        fm = G.flag_map(["--a", "1", "--b=2", "--c", "--d", "1", "2", "--e"])
        self.assertEqual(fm, {"--a": ("1",), "--b": ("2",), "--c": ("",), "--d": ("1 2",), "--e": ("",)})

    def test_repeated_flag_keeps_both(self):
        self.assertEqual(G.flag_map(["--m", "1", "--m", "2"])["--m"], ("1", "2"))


class TestClassify(unittest.TestCase):
    def test_argv_classes(self):
        rows = {r.key: r for r in G.classify_argv(
            "argv P",
            ["--model-path", "/a", "--rank-gpu-memory-mib", "1", "--page-size", "1", "--admin-api-key=k1"],
            ["--model-path", "/b", "--rank-gpu-memory-mib", "2", "--page-size", "64", "--admin-api-key=k2"],
            {})}
        self.assertNotIn("--model-path", rows)
        self.assertEqual(rows["--rank-gpu-memory-mib"].klass, "SOLVED")
        self.assertEqual(rows["--page-size"].klass, "UNEXPLAINED")
        self.assertEqual(rows["--admin-api-key"].klass, "PER-BOOT")
        rows = G.classify_argv("argv P", ["--page-size", "1"], ["--page-size", "64"],
                               {"--page-size": "QSA"})
        self.assertEqual(rows[0].klass, "EXPLAINED")

    def test_env_classes(self):
        form = F.Weg2Form(arch="dense", experts="none", draft="dflash", p_draft="cold",
                          kv="paged_dcp", flip="family", vision="transient",
                          profile="qwen27b", model=Q27)
        rows = {r.key: r for r in G.classify_env(
            "env P",
            {"SGLANG_DIR": "/x/tagA/y", "SGLANG_WEG2_BOOT_TOKEN": "a", G.P_PREFILL_TRANSIENT_ENV: "918",
             "SGLANG_KNOB": "1"},
            {"SGLANG_DIR": "/x/tagB/y", "SGLANG_WEG2_BOOT_TOKEN": "b", F.FORM_ENV: form.env_value(),
             "SGLANG_KNOB": "2"},
            ref_tag="tagA", new_tag="tagB", form=form, explain={})}
        self.assertEqual(rows["SGLANG_DIR"].klass, "PER-BOOT")
        self.assertEqual(rows["SGLANG_WEG2_BOOT_TOKEN"].klass, "PER-BOOT")
        self.assertEqual(rows[G.P_PREFILL_TRANSIENT_ENV].klass, "FORM")
        self.assertEqual(rows[F.FORM_ENV].klass, "FORM")
        self.assertEqual(rows["SGLANG_KNOB"].klass, "UNEXPLAINED")

    def test_env_alias_for_work_dirs(self):
        rows = G.classify_env("env P", {"K": "/w/base/farm/x"}, {"K": "/w/farm/x"}, ref_tag="", new_tag="",
                              form=None, explain={}, aliases=[("/w/base", "/w")])
        self.assertEqual(rows[0].klass, "PER-BOOT")


class TestReferencesAndSources(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.ev = self._td.name
        now = time.time()
        self.old27 = _front(self.ev, "q27a", "0920_000000", Q27, "DFLASH")
        self.new27 = _front(self.ev, "q27b", "0924_000000", Q27, "DFLASH")
        self.dead27 = _front(self.ev, "q27c", "0924_100000", Q27, "DFLASH", ready=False)
        self.mtp27 = _front(self.ev, "q27d", "0924_110000", Q27, "NEXTN")
        self.nf = _front(self.ev, "nf1", "0924_120000", NF, "NEXTN")
        for i, p in enumerate((self.old27, self.new27, self.dead27, self.mtp27, self.nf)):
            os.utime(p, (now + i, now + i))

    def tearDown(self):
        self._td.cleanup()

    def test_newest_reference_is_newest_ready_boot_of_the_form(self):
        got = G.newest_reference(G.CASE_27B.expect, "/m/" + Q27, self.ev)
        self.assertEqual(got, self.new27)  # q27c never READY, q27d is NEXTN, nf1 other model
        self.assertEqual(G.newest_reference(G.CASE_NF.expect, "/m/" + NF, self.ev), self.nf)
        self.assertEqual(G.tag_front_log("q27a", self.ev), self.old27)
        self.assertIsNone(G.tag_front_log("nosuch", self.ev))

    def test_parse_boot_lines(self):
        b = G.parse_boot_lines(self.new27)
        self.assertEqual(b.tag, "q27b")
        self.assertIn("--speculative-algorithm", b.argv["P"])
        self.assertEqual(b.env["P"], {"SGLANG_A": "1", "SGLANG_WEG2_BOOT_TOKEN": "x"})
        self.assertEqual(b.env["D"], {})

    def test_form_line_reference_needs_same_residue_axes(self):
        mtp = F.Weg2Form(arch="dense", experts="none", draft="mtp", p_draft="compute", kv="paged_dcp",
                         flip="family", vision="transient", profile="qwen27b", model=Q27)
        p = _front(self.ev, "q27e", "0924_130000", Q27, "DFLASH", form_line=mtp.line())
        os.utime(p, (time.time() + 99, time.time() + 99))
        # the form line (draft=mtp) wins over the argv's drafter
        self.assertEqual(G.newest_reference(G.CASE_27B.expect, "/m/" + Q27, self.ev), self.new27)

    def test_calibration_sources(self):
        p = os.path.join(self.ev, "dry.log")
        with open(p, "w") as f:
            f.write("[t] WEG2-LAUNCH #1444 DC-RESIDUE group=D source=RECORD: boot weg2xsn418 at 2026 form\n")
            f.write("[t] WEG2-LAUNCH PP-CUT depth axis: design_prefix=42971 tokens (MEASURED mean prefix over "
                    "110 prefill-batch chunks of /e/boot_weg2_weg2xsn411_x_0920_211224.P.log (rank PP0\n")
            f.write("[t] ... MEASURED run-moment residual of boot weg2xsn418 @ d37b\n")
        src = G.calibration_sources(p)
        self.assertEqual(src["#1444 D residue record"], "RECORD weg2xsn418")
        self.assertEqual(src["P-cut design prefix log"], "/e/boot_weg2_weg2xsn411_x_0920_211224.P.log")
        self.assertEqual(src["host-ledger run sample"], "weg2xsn418")
        with open(p, "w") as f:
            f.write("[t] WEG2-LAUNCH #1444 DC-RESIDUE group=D source=CONSTANT: no group-D record\n")
        self.assertEqual(G.calibration_sources(p)["#1444 D residue record"], "CONSTANT")


class TestCases(unittest.TestCase):
    def test_cases_expect_valid_axes(self):
        for case in G.CASES.values():
            self.assertEqual(set(case.expect), set(F.AXES))
            for a, v in case.expect.items():
                self.assertIn(v, F.AXIS_VALUES[a])
        self.assertTrue(G.CASE_27B.strict_ref)
        self.assertFalse(G.CASE_NF.strict_ref)
        self.assertEqual(G.CASE_27B.good_ref, "weg2xsn411")


if __name__ == "__main__":
    unittest.main()
