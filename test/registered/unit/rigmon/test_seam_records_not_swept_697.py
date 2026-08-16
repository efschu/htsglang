"""#697: the planner's budget sweep was deleting the seam records too.

THE SYMPTOM. ``kv_budget-<digest>-seam-rank<N>.json`` kept vanishing from
``~/.cache/sglang`` -- three times on 2026-08-16, once mid-soak. The cost is
not the file: a boot that cannot find its seam record sizes COLD, and cold
sizing is what took the oversized 550000 pin into an OOM while loading the
NEXTN weights at 12:04.

THE DELETER, and it leaves a signature that named it. The cache directory had
no live seam records but a pile of ``.bak-YYYYMMDD-HHMMSS`` copies, timestamps
matching the disappearances. Only one place writes that name::

    rigmon/kvbudget.py  reset_budget()   copy2(path, path + ".bak-<ts>")
                                         os.remove(path)

and it is reached automatically from::

    planner/runner.py   neutralise_kv_budget()  "Run before each [boot]"
                          for path in lister(cache_dir): resetter(path)

THE BUG IS THE LISTER'S REACH, not the reset. ``list_budget_files`` selected
``startswith("kv_budget-") and endswith(".json")``, and the seam records are
named ``kv_budget-<digest>-seam-rank<N>.json`` -- so they matched a sweep
intended for one file. ``RunPolicy`` states the intent exactly and the
implementation exceeded it:

    #: #188: clear ``kv_budget-<hash>.json`` before every point.

The two artifacts have different lifecycles. The token-vector budget is
per-boot and MEANT to be cleared so an A/B arm re-measures. The seam records
are the two-boot protocol's memory: deleting them does not make the next boot
re-measure, it makes it size without a measurement at all.

THE FIX IS THE SHAPE OF THE NAME, not a charset. A budget file is
``kv_budget-<digest>.json`` with nothing between the digest and the suffix; a
sub-artifact keyed off the same digest carries another ``-``. Selecting on that
excludes the seam records AND any future ``kv_budget-<digest>-<something>.json``
without needing to know its name in advance -- which is the property that keeps
this from recurring with the next artifact.
"""

import json
import os
import shutil
import tempfile
import unittest

from sglang.srt.rigmon import kvbudget
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)

DIGEST = "23a739c1b253"
BUDGET = f"kv_budget-{DIGEST}.json"
SEAM = [f"kv_budget-{DIGEST}-seam-rank{i}.json" for i in range(3)]


def _cache_dir():
    d = tempfile.mkdtemp(prefix="seam697-")
    with open(os.path.join(d, BUDGET), "w") as fh:
        json.dump({"components": [], "safety_mib": 0}, fh)
    for name in SEAM:
        with open(os.path.join(d, name), "w") as fh:
            json.dump({"fixed_bytes": 1, "per_row_bytes": 2.0, "id_space": 3}, fh)
    # Unrelated neighbours that must never be touched either.
    for other in ("card_library.json", "hw_profile-abc.json"):
        with open(os.path.join(d, other), "w") as fh:
            fh.write("{}")
    return d


class TheListerClaimsOnlyBudgetFiles(unittest.TestCase):
    def setUp(self):
        self.dir = _cache_dir()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_the_token_vector_budget_is_listed(self):
        got = [os.path.basename(p) for p in kvbudget.list_budget_files(self.dir)]
        self.assertIn(BUDGET, got)

    def test_the_seam_records_are_not_listed(self):
        got = [os.path.basename(p) for p in kvbudget.list_budget_files(self.dir)]
        for name in SEAM:
            self.assertNotIn(
                name,
                got,
                "the budget lister still claims the seam records, so the "
                "planner's pre-boot sweep will delete the two-boot protocol's "
                "measurements",
            )

    def test_nothing_else_in_the_cache_is_claimed(self):
        got = [os.path.basename(p) for p in kvbudget.list_budget_files(self.dir)]
        self.assertEqual(got, [BUDGET])

    def test_a_future_sub_artifact_is_excluded_by_shape(self):
        """The property that stops this recurring: anything keyed off the same
        digest with a further ``-`` is a sub-artifact, whatever it is called."""
        extra = f"kv_budget-{DIGEST}-something-new.json"
        with open(os.path.join(self.dir, extra), "w") as fh:
            fh.write("{}")
        got = [os.path.basename(p) for p in kvbudget.list_budget_files(self.dir)]
        self.assertNotIn(extra, got)


class TheResetRefusesWhatItDoesNotOwn(unittest.TestCase):
    """Defence in depth: even handed a seam path directly, the budget resetter
    must not delete it. The lister is the root fix; this is the second lock."""

    def setUp(self):
        self.dir = _cache_dir()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_a_seam_record_is_refused_and_survives(self):
        path = os.path.join(self.dir, SEAM[0])
        res = kvbudget.reset_budget(path)
        self.assertFalse(res.get("removed"))
        self.assertTrue(os.path.isfile(path), "the seam record was deleted")

    def test_the_refusal_says_why(self):
        res = kvbudget.reset_budget(os.path.join(self.dir, SEAM[0]))
        self.assertIn("seam", str(res.get("reason", "")).lower() + str(res))

    def test_a_real_budget_file_is_still_reset(self):
        path = os.path.join(self.dir, BUDGET)
        res = kvbudget.reset_budget(path)
        self.assertTrue(res["removed"])
        self.assertFalse(os.path.isfile(path))
        self.assertTrue(os.path.isfile(res["backup"]))


class ThePlannerSweepLeavesTheSeamRecords(unittest.TestCase):
    """End-to-end on the real callsite, with the real lister and resetter.

    This is the case that would have caught it: the unit above tests the
    lister, this tests the thing that was actually deleting files.
    """

    def setUp(self):
        self.dir = _cache_dir()
        self.addCleanup(shutil.rmtree, self.dir, ignore_errors=True)

    def test_the_default_policy_clears_the_budget_but_not_the_seam(self):
        from sglang.srt.planner.runner import RunPolicy, neutralise_kv_budget

        policy = RunPolicy()
        self.assertTrue(policy.reset_kv_budget)
        self.assertIsNone(policy.pin_token_vector)

        out = neutralise_kv_budget(policy, cache_dir=self.dir)
        self.assertEqual(out["strategy"], "reset")

        self.assertFalse(
            os.path.isfile(os.path.join(self.dir, BUDGET)),
            "the token-vector budget was NOT cleared; the A/B arm would "
            "inherit the previous measurement",
        )
        for name in SEAM:
            self.assertTrue(
                os.path.isfile(os.path.join(self.dir, name)),
                f"{name} was swept by the planner's pre-boot budget reset -- "
                "the next boot sizes COLD, which is the 12:04 OOM",
            )

    def test_a_pinned_policy_still_touches_nothing(self):
        from sglang.srt.planner.runner import RunPolicy, neutralise_kv_budget

        policy = RunPolicy(pin_token_vector="32,16,16")
        out = neutralise_kv_budget(policy, cache_dir=self.dir)
        self.assertEqual(out["strategy"], "pinned")
        self.assertTrue(os.path.isfile(os.path.join(self.dir, BUDGET)))
        for name in SEAM:
            self.assertTrue(os.path.isfile(os.path.join(self.dir, name)))


if __name__ == "__main__":
    unittest.main()
