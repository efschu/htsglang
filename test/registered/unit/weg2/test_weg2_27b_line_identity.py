"""27B line (24.09.): every reader of a measured source takes only samples of
THIS checkpoint AND this line (weg2/line_identity.py).

User order 11:4xZ: "die nf werte passen nicht fuers 27b ... und umgekehrt".
Pinned hermetically (temp git repo, temp evidence dir, temp record):
  * a boot counts only with the same checkpoint AND a commit that is an
    ancestor of the running tree; an NF-line commit never is;
  * read_measured_record / flip_ratchet_candidates / the P-log readers apply
    the filter, and without it answer exactly as before;
  * a run origin whose every sample was REFUSED by the identity is the launch
    reading (#1350e rule), an empty sidecar still the dk7 stand-in.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger
from sglang.srt.weg2 import launcher
from sglang.srt.weg2 import line_identity as LI

Q27 = "Qwen3.8-27B-INT8-gdncov-vocabembed"
NF = "Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist"


def _git(repo, *args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _front(ev, tag, tip, stamp, model):
    path = os.path.join(ev, f"boot_weg2_{tag}_{tip}_{stamp}.front.log")
    with open(path, "w") as f:
        f.write(f"[t] WEG2-LAUNCH === WEG2 BOOT tag={tag}\n")
        f.write(f"[t] WEG2-LAUNCH group P argv: /v/python -m sglang.launch_server --model-path /m/{model} "
                f"--speculative-draft-model-path /m/draft\n")
    return path


def _plog(ev, tag, tip, stamp):
    path = os.path.join(ev, f"boot_weg2_{tag}_{tip}_{stamp}.P.log")
    with open(path, "w") as f:
        for _ in range(3):
            f.write("[2026-09-20 21:13:06 PP0] Prefill batch, #new-seq: 1, #new-token: 4096, "
                    "#cached-token: 0, token usage: 0.01\n")
    return path


class _Line(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        root = self._td.name
        self.repo = os.path.join(root, "repo")
        os.makedirs(self.repo)
        _git(self.repo, "init", "-q")
        _git(self.repo, "config", "user.email", "t@t")
        _git(self.repo, "config", "user.name", "t")
        for msg in ("base", "line"):
            with open(os.path.join(self.repo, "f"), "a") as f:
                f.write(msg)
            _git(self.repo, "commit", "-qam", msg) if msg != "base" else (
                _git(self.repo, "add", "f"), _git(self.repo, "commit", "-qm", msg))
        self.line_tip = _git(self.repo, "rev-parse", "--short=10", "HEAD")
        self.base_tip = _git(self.repo, "rev-parse", "--short=10", "HEAD~1")
        _git(self.repo, "checkout", "-q", "-b", "nf", "HEAD~1")
        with open(os.path.join(self.repo, "g"), "w") as f:
            f.write("nf")
        _git(self.repo, "add", "g")
        _git(self.repo, "commit", "-qm", "nf line")
        self.nf_tip = _git(self.repo, "rev-parse", "--short=10", "HEAD")
        _git(self.repo, "checkout", "-q", "-")
        self.ev = os.path.join(root, "ev")
        os.makedirs(self.ev)
        self.f_good = _front(self.ev, "xsn411", self.base_tip, "0920_211224", Q27)
        self.f_nfline = _front(self.ev, "xsn418", self.nf_tip, "0924_103618", Q27)
        self.f_nf = _front(self.ev, "fnFL2x144", self.base_tip, "0924_102507", NF)
        self.line = LI.LineIdentity(model="/m/" + Q27, repo=self.repo, head=self.line_tip,
                                    evidence_dir=self.ev)

    def tearDown(self):
        self._td.cleanup()


class TestLineIdentity(_Line):
    def test_boot_needs_checkpoint_and_ancestry(self):
        self.assertTrue(self.line.accepts_boot(self.base_tip, self.f_good))
        self.assertFalse(self.line.accepts_boot(self.nf_tip, self.f_nfline))    # NF line
        self.assertFalse(self.line.accepts_boot(self.base_tip, self.f_nf))      # other model
        self.assertFalse(self.line.accepts_boot(self.base_tip, None))           # unproven
        self.assertFalse(self.line.accepts_boot("deadbeef00", self.f_good))     # unknown commit

    def test_sample_and_log(self):
        self.assertTrue(self.line.accepts_sample({"boot_tag": "xsn411"}))
        self.assertFalse(self.line.accepts_sample({"boot_tag": "xsn418"}))
        self.assertFalse(self.line.accepts_sample({"boot_tag": "fnFL2x144"}))
        self.assertFalse(self.line.accepts_sample({"boot_tag": "nosuch"}))
        self.assertFalse(self.line.accepts_sample({}))
        p_good = _plog(self.ev, "xsn411", self.base_tip, "0920_211224")
        p_bad = _plog(self.ev, "xsn418", self.nf_tip, "0924_103618")
        self.assertTrue(self.line.accepts_log(p_good))
        self.assertFalse(self.line.accepts_log(p_bad))

    def test_spec_round_trip(self):
        back = LI.parse_spec(self.line.spec())
        self.assertEqual(back, self.line)
        self.assertIsNone(LI.parse_spec(""))
        self.assertIsNone(LI.parse_spec("model=x"))


class TestReaders(_Line):
    def _record(self):
        rec = os.path.join(self.ev, "rec.json")
        with open(rec, "w") as f:
            json.dump({"samples": [
                {"group": "D", "boot_tag": "xsn411", "at": "2026-09-20T21:14:45Z", "rss_shmem_gib": 1.0,
                 "vram_residue_mib": {"u": 1496}, "run_residual_gib": 5.0, "sampled_at_flip_epoch": 0},
                {"group": "D", "boot_tag": "xsn418", "at": "2026-09-24T10:39:28Z", "rss_shmem_gib": 1.0,
                 "vram_residue_mib": {"u": 1404}},
                {"group": "D", "boot_tag": "fnFL2x144", "at": "2026-09-24T10:30:32Z", "rss_shmem_gib": 1.0,
                 "vram_residue_mib": {"u": 768}, "run_residual_gib": 9.0, "sampled_at_flip_epoch": 0},
                {"group": "FLIP", "boot_tag": "fnFL2x144", "at": "2026-09-24T10:31:00Z",
                 "form_key": "wtags=10", "cushion_min_gib": 0.1, "xchg_bounce_gib": 1.0,
                 "flip_ratchet_gib": 8.0},
                {"group": "FLIP", "boot_tag": "xsn411", "at": "2026-09-20T21:15:03Z",
                 "form_key": "wtags=10", "cushion_min_gib": 0.9, "xchg_bounce_gib": 1.0,
                 "flip_ratchet_gib": 4.0},
            ]}, f)
        return rec

    def test_read_measured_record(self):
        rec = self._record()
        self.assertEqual(host_ledger.read_measured_record(rec)["D"]["boot_tag"], "xsn418")
        got = host_ledger.read_measured_record(rec, accept=self.line.accepts_sample)
        self.assertEqual(got["D"]["boot_tag"], "xsn411")
        self.assertEqual(got["FLIP"]["boot_tag"], "xsn411")

    def test_flip_ratchet_candidates(self):
        rec = self._record()
        self.assertEqual(len(host_ledger.flip_ratchet_candidates(rec, "wtags=10")), 2)
        got = host_ledger.flip_ratchet_candidates(rec, "wtags=10", accept=self.line.accepts_sample)
        self.assertEqual([e["boot_tag"] for e in got], ["xsn411"])
        c, b, prov = host_ledger.resolve_prior_cushion(rec, "wtags=10", accept=self.line.accepts_sample)
        self.assertEqual(c, 0.9)

    def test_p_log_readers(self):
        now = time.time()
        p_good = _plog(self.ev, "xsn411", self.base_tip, "0920_211224")
        p_bad = _plog(self.ev, "xsn418", self.nf_tip, "0924_103618")
        os.utime(p_good, (now - 100, now - 100))
        os.utime(p_bad, (now, now))
        self.assertEqual(launcher.newest_prefill_census_log(self.ev), p_bad)
        self.assertEqual(launcher.newest_prefill_census_log(self.ev, accept=self.line.accepts_log), p_good)
        self.assertIsNone(launcher.newest_bubble_log(self.ev, accept=self.line.accepts_log))


class TestRunOrigin(unittest.TestCase):
    def test_identity_refused_is_the_launch_reading(self):
        origin, src = host_ledger.run_origin_gib(7.93, {}, identity_refused=("sample of boot fnFL2x148",))
        self.assertEqual(origin, 7.93)
        self.assertIn("refused by the calibration identity", src)

    def test_empty_sidecar_still_dk7(self):
        origin, src = host_ledger.run_origin_gib(0.5, {})
        self.assertAlmostEqual(origin, host_ledger.dk7_run_residual_gib(), places=6)
        self.assertNotIn("calibration identity", src)

    def test_surviving_sample_unchanged_by_refusals(self):
        rec = {"D": {"boot_tag": "b", "run_residual_gib": 12.0, "sampled_at_flip_epoch": 0,
                     "at": "2026-09-20T21:14:45Z"}}
        a = host_ledger.run_origin_gib(3.0, rec)
        b = host_ledger.run_origin_gib(3.0, rec, identity_refused=("x",))
        self.assertEqual(a[0], b[0])


if __name__ == "__main__":
    unittest.main()
