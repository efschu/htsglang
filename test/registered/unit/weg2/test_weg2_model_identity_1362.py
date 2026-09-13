# SPDX-License-Identifier: Apache-2.0
"""#1362 -- three 27B fossils, found by running the 4B-FP8 transition vehicle.

The desk run of RedHatAI/Qwen3.5-4B-FP8-dynamic (7.48 GiB, 32 layers, LLLA x8)
exposed constants that were measured on the 27B and say so nowhere:

  (3) Sigma H is solved from the PREVIOUS BOOT's census. The 4B would have been
      handed `Sigma H 48240 MiB = per card the LARGER of [tag census 47512,
      measured dormant image 48672]` -- digit for digit weg2xsn25's -- i.e. a
      47 GiB ring for a 7.48 GiB model, dying in W48 + 3x W51 without ever
      naming the cause. Hen-and-egg: the record can only come from a boot, and
      the boot cannot start because it is priced from the old model.
  (1) P_PP_STAGE_RATIO_SCORES=(32,18,14) and MEASURED_MS_PER_LAYER are a
      64-layer measurement with no model reference. On 32 layers the chain ends
      in a BARE ValueError from the cut solver -- the only unnamed refusal in
      the whole launch path.
  (4) `--served-model-name "Qwen3.8-27B"` was a literal on EVERY boot: the name
      the front routes on and every probe asserts against, so a 4B boot answers
      under the 27B's name. A silent identity swap in the one field whose job
      is identity.
"""

import json
import os
import tempfile
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import launcher as lc
from sglang.test.test_utils import CustomTestCase

#: The two models this ticket has to keep apart, as the launcher sees them.
M27 = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8"
M4B = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.5-4B-FP8-dynamic"


def _calib(d, digest, **over):
    rec = {
        "schema": hl.CALIB_SCHEMA, "model_digest": digest,
        "measured_ms_per_layer": [8.10, 35.16, 33.59],
        "measured_counts": [11, 11, 10], "measured_attn_counts": [2, 3, 3],
        "calibration_prefix_tokens": 4096, "provenance": "boot weg2q4b @ test",
    }
    rec.update(over)
    with open(os.path.join(d, f"{digest}.json"), "w") as f:
        json.dump(rec, f)
    return rec


class TheDigestSeparatesTheModels1362(CustomTestCase):
    def test_two_models_never_share_a_digest(self):
        self.assertNotEqual(hl.model_digest(M27), hl.model_digest(M4B))

    def test_the_digest_is_stable_and_names_the_model(self):
        self.assertEqual(hl.model_digest(M4B), hl.model_digest(M4B + "/"))
        self.assertTrue(hl.model_digest(M4B).startswith("Qwen3.5-4B-FP8-dynamic@"))


class TheCalibrationIsNeverBorrowed1362(CustomTestCase):
    """(1): a 64-layer measurement may not be spent on a 32-layer model."""

    def test_a_missing_calibration_refuses_by_name_not_by_ValueError(self):
        with tempfile.TemporaryDirectory() as d:
            rec, why = hl.read_pp_calibration(hl.model_digest(M4B), d)
            self.assertIsNone(rec)
            self.assertIn("no PP calibration", why)
            exc = hl.refuse_foreign_calibration(hl.model_digest(M4B), why)
            self.assertIsInstance(exc, hl.Weg2ModelIdentityMismatch)
            msg = str(exc)
            self.assertIn("W99 Weg2ModelIdentityMismatch", msg)
            self.assertIn("MEASURED_MS_PER_LAYER", msg)
            self.assertIn("bare ValueError", msg)
            self.assertIn("measuring run", msg)

    def test_a_file_holding_another_models_numbers_is_refused(self):
        """The exact transfer this check exists to stop."""
        with tempfile.TemporaryDirectory() as d:
            _calib(d, hl.model_digest(M4B), model_digest=hl.model_digest(M27))
            rec, why = hl.read_pp_calibration(hl.model_digest(M4B), d)
            self.assertIsNone(rec)
            self.assertIn("carries model_digest", why)

    def test_ms_and_counts_are_validated_as_a_PAIR(self):
        with tempfile.TemporaryDirectory() as d:
            dg = hl.model_digest(M4B)
            _calib(d, dg, measured_counts=None, stage_layer_counts=None)
            rec, why = hl.read_pp_calibration(dg, d)
            self.assertIsNone(rec)
            self.assertIn("as a PAIR", why)
            _calib(d, dg, measured_attn_counts=[2, 3])      # one short
            rec, why = hl.read_pp_calibration(dg, d)
            self.assertIsNone(rec)
            self.assertIn("disagrees with itself", why)

    def test_a_refused_measurement_is_an_absence_WITH_a_reason(self):
        """'refused at measuring time' and 'nobody measured' are different
        states, and only one is fixed by booking another window."""
        with tempfile.TemporaryDirectory() as d:
            dg = hl.model_digest(M4B)
            _calib(d, dg, refused="P died in load, no ms figures")
            rec, why = hl.read_pp_calibration(dg, d)
            self.assertIsNone(rec)
            self.assertIn("REFUSED at measuring time", why)
            self.assertIn("P died in load", why)

    def test_the_matching_record_is_returned_with_its_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            dg = hl.model_digest(M4B)
            _calib(d, dg)
            rec, why = hl.read_pp_calibration(dg, d)
            self.assertIsNotNone(rec)
            self.assertEqual(rec["measured_counts"], [11, 11, 10])
            self.assertIn("MEASURED for model", why)
            self.assertIn("weg2q4b", why)


class TheServedNameFollowsTheModel1362(CustomTestCase):
    """(4): the literal is gone; the name is derived."""

    def test_the_27B_keeps_the_name_it_always_had(self):
        self.assertEqual(lc.served_model_name(M27), "Qwen3.8-27B-INT8")

    def test_the_4B_does_not_answer_under_the_27Bs_name(self):
        self.assertEqual(lc.served_model_name(M4B), "Qwen3.5-4B-FP8-dynamic")
        self.assertNotIn("27B", lc.served_model_name(M4B))

    def test_a_snapshot_path_resolves_to_the_repo_not_the_commit(self):
        p = ("/cache/models--RedHatAI--Qwen3.5-4B-FP8-dynamic/snapshots/"
             "397b7ba47a99b3221ebfc0cfc5a279118cb733ad")
        self.assertEqual(lc.served_model_name(p), "RedHatAI/Qwen3.5-4B-FP8-dynamic")

    def test_an_override_wins(self):
        self.assertEqual(lc.served_model_name(M4B, "my-name"), "my-name")

    def test_the_literal_is_gone_from_the_argv_builder(self):
        """RATCHET: red before this commit, where the string was hard-coded."""
        import inspect
        src = inspect.getsource(lc)
        i = src.index('"--served-model-name"')
        self.assertIn("served_model_name(model)", src[i:i + 120])


class TheSurchargeIsPrintedNotAdopted1362(CustomTestCase):
    """(3): the manifest anchor's 40 % shortfall, named and borrowed openly."""

    def test_the_ratio_is_the_measured_one(self):
        self.assertAlmostEqual(hl.MANIFEST_CENSUS_SURCHARGE, 47512 / 32783, places=3)

    def test_it_says_it_is_borrowed_and_conservative(self):
        s = hl.MANIFEST_CENSUS_SURCHARGE_SOURCE
        self.assertIn("BORROWED", s)
        self.assertIn("conservative", s)
        self.assertIn("32783", s)
        self.assertIn("47512", s)

    def test_it_makes_the_ring_bigger_never_smaller(self):
        """DANGER DIRECTION: a foreign number may only over-size here."""
        self.assertGreater(hl.MANIFEST_CENSUS_SURCHARGE, 1.0)


if __name__ == "__main__":
    unittest.main()


class TheDormantImageIsNeverSpentOnAnotherModel1362(CustomTestCase):
    """(3) WIRED, not merely built -- the [21b] lesson applied in advance.

    A gate that exists and is never called is the state this line has paid for
    four times. So: the launcher STAMPS the digest into the record, and the
    consumer REFUSES a foreign one.
    """

    def test_the_launcher_stamps_the_digest_into_the_record(self):
        """RATCHET on reachability: red before this commit (0 occurrences)."""
        import inspect
        src = inspect.getsource(lc)
        i = src.index("host_ledger.dormant_image_sample(")
        call = src[i:src.index("log(host_ledger.format_dormant_image", i)]
        self.assertIn("model_digest_=host_ledger.model_digest(ns.model)", call)

    def test_the_record_carries_it(self):
        rec = hl.dormant_image_sample(
            group="P", shmem_before_bytes=0, shmem_after_bytes=0, pids=[],
            weight_tags_gib=1.0, interleaved=False, boot_tag="t", commit="c",
            model_digest_=hl.model_digest(M4B))
        self.assertEqual(rec["model_digest"], hl.model_digest(M4B))

    def test_a_foreign_image_refuses_by_name(self):
        entry = {"boot_tag": "weg2xsn25", "commit": "b568f9afd5",
                 "model_digest": hl.model_digest(M27)}
        with self.assertRaises(hl.Weg2ModelIdentityMismatch) as cm:
            hl.refuse_foreign_image(entry, hl.model_digest(M4B))
        msg = str(cm.exception)
        self.assertIn("W99 Weg2ModelIdentityMismatch", msg)
        self.assertIn("47 GiB ring for a 7.48 GiB model", msg)
        self.assertIn("W48", msg)
        self.assertIn("1.449", msg)          # the named way forward

    def test_an_empty_digest_is_UNKNOWN_and_not_matching(self):
        """DANGER DIRECTION: a pre-#1362 record must not pass as 'same model'."""
        with self.assertRaises(hl.Weg2ModelIdentityMismatch) as cm:
            hl.refuse_foreign_image({"boot_tag": "old"}, hl.model_digest(M4B))
        self.assertIn("NO model digest", str(cm.exception))

    def test_the_matching_model_passes(self):
        dg = hl.model_digest(M27)
        hl.refuse_foreign_image({"model_digest": dg}, dg)   # must not raise
