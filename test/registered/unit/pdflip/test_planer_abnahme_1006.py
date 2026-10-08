"""AP-J acceptance (plan PLAN-PROFIL-PLANER-1006 section 1.3 A1, section 4c Nachtrag 18:00Z item 4): the 27B Dual reference against the
RE-SNAPSHOTTED live release profile.

The AP0 Dual golden (``plan_27b_dual_n3.txt``) is of the profile snapshot of 2026-10-06 14:01Z (sha 1cc8890c).  The 27B seat moved the live
profile afterwards (P-KV 266240, cut 31,17,16 / attn 7,5,4, P budgets 6610,5050,5200, ``--env-d FLLIPER_PDFLIP_DUAL_D_WANT_LOCKED=1``); its
snapshot ``profiles/27b-nvfp4-dual-live1521.env`` (sha ce347c79, AP-E) is the live file byte for byte on 2026-10-07.  AP-J generated the
golden of that snapshot (``golden/plan_27b_dual_live1521_n3.txt`` + ``.provenance.json`` + ``launch_27b-nvfp4-dual-live1521.json``) with
``propose_oracle golden`` on the acceptance tree and pins here:

* the snapshot and the golden are tied by their provenance (sha256 of profile, base, pchunk);
* ``propose()`` for the reference rig (RTX 5090 + 2x RTX 3080) in the form Dual with that profile as basis gives the profile's argv/env
  (0 differences) and its launcher dry run is the new golden with 0 diff lines.

On this box the dry run of that profile is refused by the launcher's own W64 (no measured dual-D log; see the provenance note): the golden
pins that named verdict, it is not a runnable plan.  ``test_live_file_is_the_snapshot`` REPORTS (skip with the reason) when the live
release profile moves again.

GPU-free, NVML-free, Docker-free.
"""

from __future__ import annotations

import hashlib
import json
import os
import unittest

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    import test_planer_ape_dual_1006 as APE
    from flliper.srt.pdflip import propose_oracle as O
except Exception as exc:  # pragma: no cover - no pdflip launcher in this build
    pytest.skip(f"pdflip launcher unavailable: {exc}", allow_module_level=True)

LIVE = "27b-nvfp4-dual-live1521"
LIVE_RELEASE = "/spinning/gpu-arb/docker/profiles_release/27b-nvfp4-dual.env"
GOLDEN_TXT = os.path.join(APE.GOLDEN, "plan_27b_dual_live1521_n3.txt")
GOLDEN_PROV = os.path.join(APE.GOLDEN, "plan_27b_dual_live1521_n3.provenance.json")
GOLDEN_LAUNCH = os.path.join(APE.GOLDEN, "launch_27b-nvfp4-dual-live1521.json")


def setUpModule():
    APE.setUpModule()


def tearDownModule():
    APE.tearDownModule()


def _sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _live_sha_as_snapshot(path: str) -> list:
    """sha256 of a live profile in the spelling(s) the committed snapshot may be in (F0-G): the file as it is (a live dir already converted
    by ``tools/release/profconv.py --convert-live``) and the kit's conversion of it (a live dir still carrying the old names -- the snapshots
    are of the renamed spelling).  The snapshot is current when EITHER equals its recorded sha."""
    import hashlib as _h
    import importlib.util as _u
    import sys as _s

    kit = os.path.join(str(__import__("pathlib").Path(__file__).resolve().parents[4]), "tools", "release")
    with open(path, "rb") as fh:
        raw = fh.read()
    out = [_h.sha256(raw).hexdigest()]
    try:
        spec = _u.spec_from_file_location("profconv_snap", os.path.join(kit, "profconv.py"))
        mod = _u.module_from_spec(spec)
        _s.path.insert(0, kit)
        try:
            spec.loader.exec_module(mod)
        finally:
            _s.path.remove(kit)
        imap = mod.R._load_imap(os.path.join(kit, "data", "merged_0928.json"))
        out.append(_h.sha256(mod.convert(raw.decode("utf-8"), imap).encode("utf-8")).hexdigest())
    except Exception:
        pass
    return out


def _prov() -> dict:
    with open(GOLDEN_PROV, encoding="utf-8") as fh:
        return json.load(fh)


class TestProvenance(unittest.TestCase):
    def test_golden_provenance_ties_snapshot_and_inputs(self):
        p = _prov()
        self.assertEqual(p["golden"], os.path.basename(GOLDEN_TXT))
        fix = os.path.dirname(APE.GOLDEN)
        self.assertEqual(_sha(os.path.join(fix, p["profile"]["file"])), p["profile"]["sha256"])
        self.assertEqual(p["profile"]["sha256"], APE.PROVENANCE[LIVE + ".env"])
        for k in ("base", "pchunk"):
            self.assertEqual(_sha(os.path.join(fix, p["profile_inputs"][k]["file"])), p["profile_inputs"][k]["sha256"], k)

    def test_live_file_is_the_snapshot(self):
        if not os.path.isfile(LIVE_RELEASE):
            self.skipTest("live release profile not on this box: %s" % LIVE_RELEASE)
        live = _live_sha_as_snapshot(LIVE_RELEASE)
        if _prov()["profile"]["sha256"] not in live:
            self.skipTest("LIVE PROFILE MOVED under the AP-J snapshot (re-snapshot, regenerate golden + launch json + provenance): "
                          "%s sha %s" % (LIVE_RELEASE, live[0]))

    def test_launch_json_is_the_snapshot(self):
        # the ``launch`` sub-command leaves image paths as they are (asset_dirs=()), so does this comparison; the pchunk token is an
        # absolute path inside the worktree that made the file: compared with the worktree root cut off
        li = O.profile_launch_input(os.path.join(APE.PROFILES, LIVE + ".env"), asset_dirs=())
        with open(GOLDEN_LAUNCH, encoding="utf-8") as fh:
            want = json.load(fh)

        def norm(doc):
            return json.loads(json.dumps(doc).replace(os.path.join(APE.TREE, ""), "<TREE>/").replace(
                "/spinning/htsglang/.claude/worktrees/planer-abnahme-1006/", "<TREE>/"))

        self.assertEqual(norm(O.launch_input_doc(li)), norm(want))


@APE._NEEDS_DUAL
class TestLiveDualReference(unittest.TestCase):
    """A1 Dual against the new golden: the proposal is the profile, and its dry run is the golden with 0 diff lines."""

    def test_proposal_is_the_live_profile(self):
        li = APE._dual_profile(LIVE)
        v = APE._propose("ref3", profile=LIVE)
        self.assertEqual(v["argv"], list(li.argv))
        self.assertEqual(v["env"], dict(li.env))
        self.assertTrue(v["vectors_ok"], v["vectors_wrong"])

    def test_proposal_dry_run_equals_the_new_golden(self):
        v = APE._propose("ref3", profile=LIVE)
        b = APE._dual_profile(LIVE)
        li = O.LaunchInput(v["argv"], v["env"], b.vars, [], "propose:" + v["basis"], b.instruments)
        res = O.run_profile("", APE._ref_rows(), tree=APE.TREE, force=False, launch_input=li).result
        with open(GOLDEN_TXT, encoding="utf-8") as fh:
            want = fh.read()
        d = O.diff_lines(want, res.dump())
        self.assertEqual(d, [], "plan diff vs %s: %d lines\n%s" % (os.path.basename(GOLDEN_TXT), len(d), "\n".join(x[:240] for x in d[:12])))
        # the golden is the launcher's own named verdict on this box (provenance note), not a runnable plan
        self.assertEqual(res.exc_type, "PdFlipLaunchRefused", res.exc_msg[:300])
        self.assertIn("W64", res.exc_msg[:200])
        self.assertEqual(res.forced, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
