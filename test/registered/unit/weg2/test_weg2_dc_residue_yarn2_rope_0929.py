# SPDX-License-Identifier: Apache-2.0
"""D's dormant reserve takes D's RoPE context (YaRN x2 27B, zw4skn 29.09.).

METAL (dkr27browauthorityyarn2bar1fs09291825, z30y3g, 524288): at the first P wake
  "WEG2 STOP W19 DormantResidueRefused -- measured D_c(D) exceeds the reserve P's budget assumed:
   {3080: (1718, 1678), 5090: (2146, 2106), 3080: (1718, 1678)} (measured, reserved) MiB";
the reserve came from the x1 record 1786 / 1358 / 1358 (+256 record margin +64 slack) -- measured at 262144.
D is TP: every rank holds the target's cos/sin cache (64 cols) and the DFlash draft's (128 cols); both stay
resident while D sleeps. The fix adds D's share of the SAME RoPE reading (d_rope_context_delta_mib) to the
reserve, relative to the context the residue was measured at (a record of a longer context names it).
"""
import os
import types
import unittest

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover

    def register_cpu_ci(*args, **kwargs):
        return None


from sglang.srt.weg2 import host_ledger as H
from sglang.srt.weg2 import launcher as L
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

TARGET = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov-vocabembed"
DRAFT = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-DFlash2-W8-lued"
MEASURED = {"5090": 2146, "3080a": 1718, "3080b": 1718}       # D_c(D) at the first P wake, x2
RESERVED_X1 = {"5090": 2106, "3080a": 1678, "3080b": 1678}    # what 4b95474c0f priced (x1 record + margins)


def _ns(ctx):
    extra = f"--context-length {ctx}" if ctx else ""
    return types.SimpleNamespace(extra_p=extra, extra_d=extra, model=TARGET, spec_form="DFLASH",
                                 dflash_draft_path=DRAFT)


@unittest.skipUnless(os.path.isdir(TARGET) and os.path.isdir(DRAFT), "rig checkpoints not mounted")
class DcResidueYarn2RopeTest(CustomTestCase):
    def test_metal_w19_closed_by_the_d_rope_share(self):
        delta, line = L.d_rope_context_delta_mib(_ns(524288))
        self.assertAlmostEqual(delta, 192.0, places=3)             # (64 + 128) cols x 262144 rows x 4 B
        self.assertIn("524288", line)
        add = int(__import__("math").ceil(delta))
        for card, measured in MEASURED.items():
            self.assertGreater(measured, RESERVED_X1[card])         # the W19 of the metal, reproduced
            self.assertLessEqual(measured, RESERVED_X1[card] + add)  # closed with the D share

    def test_same_reading_as_the_p_pool_floor(self):
        """One reading: D's per-rank share == target + draft of the P function (last stage carries both)."""
        d, _ = L.d_rope_context_delta_mib(_ns(524288))
        p, _ = L.p_rope_context_delta_mib(_ns(524288), 3)
        self.assertAlmostEqual(d, p[-1], places=6)

    def test_x1_byte_identical_and_record_context(self):
        self.assertEqual(L.d_rope_context_delta_mib(_ns(None)), (0.0, None))
        self.assertEqual(L.d_rope_context_delta_mib(_ns(262144)), (0.0, None))
        # a record measured at 524288 prices 0 for a 524288 boot and a negative delta for an x1 boot
        self.assertEqual(L.d_rope_context_delta_mib(_ns(524288), 524288), (0.0, None))
        neg, _ = L.d_rope_context_delta_mib(_ns(262144), 524288)
        self.assertAlmostEqual(neg, -192.0, places=3)

    def test_record_stamps_its_context_only_when_longer(self):
        """The dormant sample names a longer context (so the next boot prices the difference); x1: no key."""
        self.assertIn("vram_residue_context_tokens", H.dormant_image_sample.__code__.co_varnames)
        src = open(H.__file__).read()
        self.assertIn('if vram_residue_context_tokens else {}', src)

if __name__ == "__main__":
    unittest.main()
