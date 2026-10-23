# SPDX-License-Identifier: Apache-2.0
"""NF rc36 (23.10.): the B10 LSE conversion (cherry-picked 34b28ec3a8) names
its first use per process with ONE INFO line ``DCP-FI-LSE-LN2 armed``.

NF runs its full attention under Form-A DCP through QSA (natural LSE), so
whether a FlashInfer LSE ever reaches an uneven-DCP merge on NF is not
visible from the boot log without this line (27b-logprob-1023.md, NF section).
Pinned: exactly one line for many calls, the value is still multiplied by
ln 2, and with FLLIPER_DCP_FI_LSE_LN2=0 no armed line and no conversion.
"""

import math
import os
import unittest
from unittest import mock

import torch

from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

from flliper.srt.layers.attention import flashinfer_backend as fb  # noqa: E402


class TestArmedLine(CustomTestCase):
    def setUp(self):
        fb._FI_LSE_LN2_CACHE.clear()

    def tearDown(self):
        fb._FI_LSE_LN2_CACHE.clear()

    def test_one_info_line_on_first_conversion(self):
        lse = torch.tensor([[1.0, 2.0]])
        with mock.patch.dict(os.environ, {"FLLIPER_DCP_FI_LSE_LN2": "1"}):
            with self.assertLogs(fb.logger, level="INFO") as cm:
                outs = [fb._fi_lse_to_merge_domain(lse) for _ in range(5)]
        armed = [r for r in cm.output if "DCP-FI-LSE-LN2 armed" in r]
        self.assertEqual(len(armed), 1, cm.output)
        self.assertIn("shape=(1, 2)", armed[0])
        for o in outs:
            torch.testing.assert_close(o, lse * math.log(2.0))

    def test_off_no_line_no_conversion(self):
        lse = torch.tensor([3.0])
        with mock.patch.dict(os.environ, {"FLLIPER_DCP_FI_LSE_LN2": "0"}):
            with self.assertLogs(fb.logger, level="INFO") as cm:
                out = fb._fi_lse_to_merge_domain(lse)
        self.assertFalse(any("armed" in r for r in cm.output), cm.output)
        self.assertTrue(any("FLLIPER_DCP_FI_LSE_LN2=0" in r for r in cm.output), cm.output)
        torch.testing.assert_close(out, lse)

    def test_non_tensor_passes_through_silently(self):
        with mock.patch.dict(os.environ, {"FLLIPER_DCP_FI_LSE_LN2": "1"}):
            self.assertIsNone(fb._fi_lse_to_merge_domain(None))
        self.assertNotIn("noted", fb._FI_LSE_LN2_CACHE)


if __name__ == "__main__":
    unittest.main()
