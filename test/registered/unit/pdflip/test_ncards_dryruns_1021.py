"""NCARDS 1021: the launcher DRY RUN of every N x RTX 3090 release profile (27B INT8, 27B NVFP4, 27B NVFP4 dual, NF) on 2, 3 and 4 cards.

One dry run = one subprocess of ``fixtures/ncards_1021/dryrun_driver.py`` (~1.8 GB resident each: serial, the parent imports nothing heavy,
so the whole file stays under the 2 GB scope cap).  Each run is compared with its golden (``fixtures/ncards_1021/golden``): pass/refuse,
the forced codes, the P cut (stage count, layers, pool, floor), the rank budgets.  The first boot on new cards IS the measurement, so
every run is ``--force`` (HW-BORROWED / HW-UNCALIBRATED / HW-COUNT named); without ``--force`` the launcher refuses BY NAME
(``test_unforced_is_a_named_refusal``) and the reference rig is refused by nothing.

Box-bound (the censuses and checkpoints named by the profiles live on the rig box); skipped elsewhere.
REGENERATE: ``NCARDS_REGEN=1 pytest test_ncards_dryruns_1021.py`` (writes, then fails once so a regeneration is never silent).
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
TREE = HERE.parents[3]
FIX = HERE / "fixtures" / "ncards_1021"
GOLDEN = FIX / "golden"
PROFILES = TREE / "docker" / "flliper" / "profiles_release"
CENSUS_27B = "/spinning/gpu-arb/weg2/census/xchg_census_weg2xsn246_27198a2711.json"
REGEN = os.environ.get("NCARDS_REGEN") == "1"
NEEDS_BOX = unittest.skipUnless(os.path.exists(CENSUS_27B), "27B census not on this box (dry-run goldens are box-bound)")

#: (profile, n): the matrix of the round.  27B INT8 2x carries the per-request ceiling 65536 (2 x 24 GB cannot hold 27B INT8 with a
#: 262144-token request: the cut solver names 81623 tokens as the best servable P pool).
MATRIX = (
    [("27b-%dx3090" % n, n) for n in (2, 3, 4, 5, 6, 7)]
    + [("27b-nvfp4-%dx3090" % n, n) for n in (2, 3, 4, 5, 6, 7)]
    + [("27b-nvfp4-dual-%dx3090" % n, n) for n in (2, 3, 4, 5, 6, 7)]
    + [("nf-%dx3090" % n, n) for n in (2, 3, 4, 5, 6, 7)]
)


def _run(profile, n, *extra):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH=str(TREE / "python"), NCARDS_TREE=str(TREE), PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, str(FIX / "dryrun_driver.py"), str(PROFILES / (profile + ".env")), "rtx3090-24", str(n), *extra],
                       capture_output=True, text=True, env=env, timeout=3000)
    for line in p.stdout.splitlines():
        if line.startswith("NCARDS-JSON "):
            return json.loads(line[len("NCARDS-JSON "):])
    raise AssertionError("no NCARDS-JSON from the driver (rc=%s): %s" % (p.returncode, (p.stderr or p.stdout)[-1500:]))


def _mask(o):
    return json.loads(re.sub(r"/spinning/wt-[A-Za-z0-9_.-]+", "<TREE>", json.dumps(o, sort_keys=True)))


def _check(self, profile, n):
    if not (PROFILES / (profile + ".env")).is_file():
        self.fail("profile file missing: %s" % profile)
    doc = _run(profile, n)
    gp = GOLDEN / ("dryrun_%s.json" % profile)
    if REGEN:
        gp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        self.fail("golden regenerated: %s (run again without NCARDS_REGEN)" % profile)
    self.assertTrue(gp.is_file(), "no golden for %s (NCARDS_REGEN=1)" % profile)
    self.assertEqual(_mask(doc), _mask(json.loads(gp.read_text(encoding="utf-8"))))
    # the properties that matter, beside the golden: it runs, forced past named codes only, N stages
    self.assertIsNone(doc["exc_type"], doc)
    self.assertEqual(set(doc["forced"]) - {"HW-UNCALIBRATED", "HW-BORROWED", "HW-COUNT"}, set(), doc["forced"])
    self.assertEqual("HW-COUNT" in doc["forced"], n != 3, doc["forced"])      # N = 3 is the proven count: no HW-COUNT
    # review 1021 A1: the launcher's D residency check (W122/W130) passes the planner's NF distribution on every N and it RUNS on every N
    # (N != 3: the 3-rank reference taken to N ranks by rank role, named "BORROWED BY RANK ROLE")
    # nfblock 1022: the launcher's own ownership / KV token-cut solve (D-EIGENTUM, #239 S3f) decides on EVERY N -- a cut form (N = 2, 3) or Form A kept (N = 4..7:
    # every worker expert resident in Form A, no cut keeps x1) -- and nothing is forced past as D-BUDGET any more (was: N = 2, 4, 5, 6, 7)
    self.assertNotIn("D-BUDGET", doc["forced"])
    # a refusal the real boot would raise is printed by the dry run as "(dry run: would refuse)" without a forced code (NF 2 / 4 x 3090 carried the S3h flip riegel
    # that way before nfblock 1022): none on any profile
    self.assertNotIn("would_refuse", doc, doc.get("would_refuse"))
    if profile.startswith("nf-"):
        self.assertEqual(doc.get("d_owner"), "cut" if n <= 3 else "form_a", doc.get("d_owner"))
        chk = doc.get("d_check")
        self.assertIsNotNone(chk, "the D FRACTION-SOLVE did not run: " + profile)
        self.assertFalse(chk["entfaellt"], chk)
        self.assertEqual(chk["borrowed_by_role"], n != 3, chk)
        self.assertEqual(len(chk["ranks"]), n, chk)
        self.assertEqual({r["verdict"] for r in chk["ranks"]}, {"PASST"}, chk)
    if doc.get("pp_cut", {}).get("layers"):
        self.assertEqual(doc["stages"], n, doc["pp_cut"])


@NEEDS_BOX
class TestForcedDryRuns(unittest.TestCase):
    """One test per profile (``test_<profile>``), so one family can be run alone: ``-k nf_``."""


for _profile, _n in MATRIX:
    setattr(TestForcedDryRuns, "test_" + _profile.replace("-", "_"), (lambda p, n: lambda self: _check(self, p, n))(_profile, _n))


@NEEDS_BOX
class TestUnforced(unittest.TestCase):
    def test_unforced_is_a_named_refusal(self):
        for profile, n in (("27b-2x3090", 2), ("27b-4x3090", 4)):
            with self.subTest(profile=profile):
                doc = _run(profile, n, "--noforce")
                self.assertIsNotNone(doc["exc_type"], "an unforced launch on foreign cards must be refused")
                # review 1021: W40 (the cut solver's "not one cut ... priceable") is exactly the regression this test must catch -- the unforced
                # launch is refused by the HW check (HW-COUNT / HW-UNCALIBRATED / HW-BORROWED), never by a cut that could not be priced
                self.assertRegex(doc["exc_head"], r"HW-(UNCALIBRATED|BORROWED|COUNT)", doc["exc_head"])
                self.assertNotRegex(doc["exc_head"], r"W40|not one cut", doc["exc_head"])

    def test_27b_int8_on_2x3090_with_the_full_context_ceiling_is_a_named_physical_refusal(self):
        """2 x 24 GB hold 27B INT8 only with a per-request ceiling of ~80k tokens: with 262144 even ``--force`` is refused by name (the cut solver
        names the best servable P pool), not crashed and not silently clamped.  The profile's own ceiling (65536) is the answer."""
        doc = _run("27b-2x3090", 2, "--arg=--max-kv-per-request=262144")
        self.assertIsNotNone(doc["exc_type"], doc)
        self.assertRegex(doc["exc_head"] + doc["exc_type"], r"W40|PPCutRefused|SERVABLE", doc)


if __name__ == "__main__":
    unittest.main()
