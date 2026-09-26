"""UNIFY S7: group P's producer flag family under DFLASH is the external draft's
(27B RC9 argv_p: ``spec_flags(producer=True)``), never the NEXTN constants read
off D's MTP depth (``p_draft_kv_flags``). NEXTN (the NF form) stays byte-identical.
Hermetic: no GPU, no boot.
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as L
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestPDraftKvFlags(CustomTestCase):
    def setUp(self):
        self._saved = dict(L._SPEC_FORM)

    def tearDown(self):
        L._SPEC_FORM.clear()
        L._SPEC_FORM.update(self._saved)

    def test_nextn_reads_d_depth_and_keeps_the_constants_otherwise(self):
        L._SPEC_FORM["form"] = "NEXTN"
        self.assertEqual(L.p_draft_kv_flags(()), L.P_DRAFT_KV_FLAGS)
        got = L.p_draft_kv_flags(["--speculative-num-steps", "3",
                                  "--speculative-num-draft-tokens", "4"])
        self.assertIn("3", got)
        self.assertEqual(got[-1], "--speculative-draft-kv-only")
        self.assertEqual(got[1], L.SPEC_ALGORITHM)

    def test_dflash_is_the_producer_family_of_the_external_draft(self):
        L._SPEC_FORM["form"] = "DFLASH"
        got = list(L.p_draft_kv_flags(["--speculative-num-steps", "3"]))
        self.assertEqual(got, L.spec_flags(producer=True))
        self.assertEqual(got[:2], ["--speculative-algorithm", "DFLASH"])
        self.assertNotIn("NEXTN", got)
        self.assertNotIn("--speculative-num-steps", got)
        self.assertEqual(got[-1], "--speculative-draft-kv-only")


if __name__ == "__main__":
    unittest.main()
