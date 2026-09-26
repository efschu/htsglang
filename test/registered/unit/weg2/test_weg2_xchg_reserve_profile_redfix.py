"""The exchange form's group-D dormant reserve is chosen PER MODEL PROFILE.

NF cb1575e94e (user 22.09.: "diesen wert nicht doppelt oder gar nicht nehmen")
made the census of THIS model the reserve under the exchange form and let it
replace the constant in ``dc_expect_d``. On the NF line that is the model's own
measurement. In the unified tree it ran for EVERY profile, so a 27B exchange
boot without its own #1444 record priced group D at its census (weg2xsn246:
1622 / 1250 / 1250 MiB) instead of the 27B line's constant (weg2xsn14:
3084 / 2588 / 2588 MiB) -- below every 27B D residue the records hold (RC9
records: 2074-2150 MiB on the 5090) -> P's budget ~1.4 GiB/card too large and
W19 at D's first sleep. With a record the record won (max), which is why the
27B dry runs were byte-identical and the defect stayed latent.

The 27B pins of the constant (test_weg2_xchg_reserve_1273 x3,
test_weg2_w19_form_residue_1273) have been red since the unified base for this
reason; this file pins the profile split that makes both lines' semantics true
in one tree.
"""

import inspect
import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import form as weg2_form
from sglang.srt.weg2 import launcher

BIG = "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d"
SM1 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
SM2 = "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4"
CARDS = [
    launcher.Card(1, BIG, "NVIDIA GeForce RTX 5090", 32607),
    launcher.Card(0, SM1, "NVIDIA GeForce RTX 3080", 20480),
    launcher.Card(2, SM2, "NVIDIA GeForce RTX 3080", 20480),
]
#: the 27B census's own readings (weg2xsn246, the file 27b.env boots with)
CENSUS_27B = {BIG: 1622, SM1: 1250, SM2: 1250}
CONSTANT_27B = {BIG: 3084, SM1: 2588, SM2: 2588}
FAMILY = [f"weights_{k}" for k in range(8)] + ["weights"]


def _census(dormant):
    blob = {"cards": {u: {"tags": {g: {t: 100 for t in FAMILY} for g in ("P", "D")},
                          "dormant_proc_used_mib": dormant[u],
                          "dormant_source": "READING: boot weg2xsn246 front WEG2-DC"}
                      for u in dormant},
            "waves": [FAMILY], "provenance": "test"}
    path = os.path.join(tempfile.mkdtemp(), "census.json")
    with open(path, "w") as fh:
        json.dump(blob, fh)
    return path


class TheRegistryNamesWhoseCensusIsTheReserve(unittest.TestCase):
    def test_only_nextflash_takes_its_census(self):
        rows = weg2_form.PROFILES
        self.assertFalse(rows[weg2_form.PROFILE_QWEN27B].d_residue_census)
        self.assertTrue(rows[weg2_form.PROFILE_NEXTFLASH].d_residue_census)


class TheReserveFollowsTheProfile(unittest.TestCase):
    def test_27b_keeps_the_constant_even_where_its_census_is_lower(self):
        out, lines = launcher.xchg_form_dormant_reserve(
            CARDS, _census(CENSUS_27B), profile=weg2_form.PROFILE_QWEN27B)
        self.assertEqual(out, CONSTANT_27B)
        big = next(ln for ln in lines if "5090" in ln)
        self.assertIn("reserved_mib=3084 source=measured:weg2xsn14", big)
        self.assertIn("measured_mib=1622", big)
        self.assertIn("delta_mib=1462", big)

    def test_no_profile_is_the_launcher_default_27b(self):
        out, _ = launcher.xchg_form_dormant_reserve(CARDS, _census(CENSUS_27B))
        self.assertEqual(out, CONSTANT_27B)

    def test_nextflash_takes_its_own_census(self):
        nf = {BIG: 1896, SM1: 1076, SM2: 1474}
        out, lines = launcher.xchg_form_dormant_reserve(
            CARDS, _census(nf), profile=weg2_form.PROFILE_NEXTFLASH)
        self.assertEqual(out, nf)
        big = next(ln for ln in lines if "5090" in ln)
        self.assertIn("reserved_mib=1896 source=census:", big)
        self.assertIn("constant_27b_mib=3084", big)


class MainGatesTheBudgetOverrideByTheSameProfile(unittest.TestCase):
    """The census replaces ``dc_expect_d`` (P's budget, the W19 riegel) only on
    the profile whose reserve it is -- the report and the budget read ONE
    decision, never two."""

    def test_main_passes_the_profile_and_gates_the_override(self):
        src = inspect.getsource(launcher.main)
        self.assertEqual(src.count("xchg_form_dormant_reserve("), 1)
        self.assertIn("xchg_form_dormant_reserve(cards, ns.weg2_xchg_census, log=log, "
                      "profile=ns.profile)", src)
        k = src.index("xchg_form_dormant_reserve(")
        gate = src.index("if xchg_census_is_reserve(ns.profile):", k)
        self.assertLess(gate, src.index("census-korrigiert", k))

    def test_the_gate_reads_the_registry(self):
        self.assertFalse(launcher.xchg_census_is_reserve(weg2_form.PROFILE_QWEN27B))
        self.assertFalse(launcher.xchg_census_is_reserve(None))
        self.assertTrue(launcher.xchg_census_is_reserve(weg2_form.PROFILE_NEXTFLASH))


if __name__ == "__main__":
    unittest.main()
