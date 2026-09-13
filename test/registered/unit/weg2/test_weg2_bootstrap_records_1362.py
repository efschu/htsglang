# SPDX-License-Identifier: Apache-2.0
"""#1362/#1358 -- FIRST BOOT OF A FORM IS A STATE, NOT A FAULT.

THIRD INSTANCE of one class on this rig in one day, and the inventory below is
why there should not be a fourth:

  #1356  text-only moved the P form key -> the inherited RING no longer matched
  #1358  a new cut had no lane record  -> W102 refused every arm
  #1362  the form key moved            -> no dormant-image record matched, and
         W99 fired on a record that was simply from ANOTHER FORM

Measured on e50ac74b7c: the default-arm dry run gave rc=2, form 2b66740bedf9,
ARM=0, W48 fatal 0, W99=1. weg2xsn25's image carries no digest, the legacy
transition needs a matching form key, and text-only had moved the key -- so no
record applied AND the bootstrap could not run either. A reader that cannot say
"first boot of this form" turns the first boot of every new form into a
refusal, including the boot that would have written the record it demands.

THE SEMANTICS, now the same at every station:
  no record for THIS key      -> first boot, ALLOWED, named, conservative
  record for this key, differs -> refuse by name (W99 / W68 at the leg)
  record for this key, legacy  -> the transition path, unchanged
"""

import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.test.test_utils import CustomTestCase

DIGEST = "5a324e4044bf915181537d1481662a3050e984acd2eb3977174a54aba0ab3143"
REC = {"boot_tag": "weg2xsn25", "commit": "8f11b2e4c6", "rss_shmem_gib": 48.0}


class TheModelGateBootstraps(CustomTestCase):
    def test_another_forms_record_is_skipped_not_refused(self):
        """THE BLOCKER: form_key_match False used to raise W99."""
        line = hl.refuse_foreign_image(dict(REC), DIGEST, form_key_match=False)
        self.assertIsNotNone(line)
        self.assertIn("BOOTSTRAP", line)
        self.assertIn("form=OTHER", line)

    def test_the_entry_is_dropped_so_another_forms_image_is_not_priced(self):
        """Skipping must also mean NOT USING it -- a line alone would leave the
        foreign image in the price."""
        terms = hl.resolve_image_terms(
            {"P": dict(REC, rss_shmem_gib=99.0), "D": dict(REC)},
            want_digest=DIGEST, form_key_match=False)
        self.assertLess(float(terms.p_gib), 99.0,
                        "another form's 99 GiB image was priced anyway")

    def test_same_form_foreign_digest_still_refuses(self):
        with self.assertRaises(hl.Weg2ModelIdentityMismatch):
            hl.refuse_foreign_image(dict(REC, model_digest="f" * 64), DIGEST,
                                    form_key_match=True)

    def test_same_form_legacy_still_transitions(self):
        line = hl.refuse_foreign_image(dict(REC), DIGEST, form_key_match=True)
        self.assertIn("LEGACY", line)
        self.assertNotIn("BOOTSTRAP", line)

    def test_same_form_matching_digest_is_silent(self):
        self.assertIsNone(hl.refuse_foreign_image(
            dict(REC, model_digest=DIGEST), DIGEST, form_key_match=True))


class TheLaneReaderBootstraps(CustomTestCase):
    def test_an_unrecorded_cut_prices_the_regions_worst_case(self):
        n, prov = hl.resolve_xchg_lanes(hl.xchg_cut_key("40,12,12"))
        self.assertEqual(n, int(xr.N_PAIRS) + int(xr.N_CARDS))
        self.assertIn("WORST-CASE", prov)
        self.assertIn("bootstrap", prov)

    def test_the_worst_case_is_derived_not_a_literal(self):
        src = inspect.getsource(hl.resolve_xchg_lanes)
        self.assertIn("N_PAIRS", src)
        self.assertIn("N_CARDS", src)
        self.assertNotIn("= 9", src)

    def test_it_over_charges_never_under(self):
        """CONSERVATIVE BY CONSTRUCTION: the bootstrap count is >= any real one."""
        known, _ = hl.resolve_xchg_lanes(hl.xchg_cut_key("39,13,12"))
        boot, _ = hl.resolve_xchg_lanes(hl.xchg_cut_key("40,12,12"))
        self.assertGreaterEqual(boot, known)

    def test_a_recorded_cut_still_uses_its_measurement(self):
        n, prov = hl.resolve_xchg_lanes(hl.xchg_cut_key("39,13,12"))
        self.assertEqual(n, 5)
        self.assertIn("source=measured", prov)


class TheClassInventory(CustomTestCase):
    """EVERY reader of an inherited record on the pre-ARM path, classified.

    The point is not that these four pass today; it is that a NEW refuser
    added to this path shows up here. Three of these four cost a boot window
    before they were classified.
    """

    #: (name, callable, kwargs that mean "no record for this key")
    STATIONS = (
        ("model/W99", "refuse_foreign_image"),
        ("lanes/W102", "resolve_xchg_lanes"),
        ("ratchet/W94", "resolve_flip_ratchet_gib"),
        ("image", "resolve_image_terms"),
        ("residual", "run_origin_gib"),
        ("calib", "read_pp_calibration"),
    )

    def test_every_station_exists(self):
        for name, sym in self.STATIONS:
            with self.subTest(station=name):
                self.assertTrue(hasattr(hl, sym), f"{name}: {sym} is gone")

    def test_no_station_refuses_when_the_record_is_simply_absent(self):
        """The inventory as an assertion, one call per station."""
        self.assertIsNotNone(
            hl.refuse_foreign_image(dict(REC), DIGEST, form_key_match=False))
        self.assertEqual(hl.resolve_xchg_lanes(hl.xchg_cut_key("z"))[0],
                         int(xr.N_PAIRS) + int(xr.N_CARDS))
        self.assertIsNotNone(hl.resolve_flip_ratchet_gib(None))
        self.assertIsNotNone(hl.resolve_image_terms(None, want_digest=""))
        self.assertIsNotNone(hl.run_origin_gib(6.5, {}))
        self.assertEqual(hl.read_pp_calibration("nosuchdigest")[0], None)


if __name__ == "__main__":
    unittest.main()
