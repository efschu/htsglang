"""Regression for the architecture boundary of CuTe GDN/KDA prefill.

The CuTe chunk prefill kernels are validated for the SM100/SM103 family only.
A ``major >= 10`` floor also matches consumer Blackwell (SM12x), which
silently routes an RTX 50-series card into a tcgen05/TMA path built for the
datacenter line; SM120 must keep using the Triton prefill fallback.

The gate is asked through :func:`get_cuda_capability`, so that is what these
cases patch -- the range arithmetic under test stays real.
"""

import unittest
from unittest.mock import patch

from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import CuteDSLGDNKernel
from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import CuteDSLKDAKernel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCuteDSLPrefillArchitectureGate(unittest.TestCase):
    def test_only_sm10x_advertises_cutedsl_prefill(self):
        cases = (
            ((8, 6), False),  # Ampere
            ((9, 0), False),  # Hopper
            ((10, 0), True),  # B200
            ((10, 3), True),  # B300
            ((12, 0), False),  # consumer Blackwell (RTX 50-series)
            ((12, 1), False),
        )
        for capability, expected in cases:
            with (
                self.subTest(capability=capability),
                patch(
                    "sglang.srt.utils.common.get_cuda_capability",
                    return_value=capability,
                ),
            ):
                self.assertEqual(CuteDSLGDNKernel().supports_prefill, expected)
                self.assertEqual(CuteDSLKDAKernel().supports_prefill, expected)

    def test_non_nvidia_never_advertises_prefill(self):
        """gfx1030 reports (10, 3) and gfx1100 reports (11, 0) through torch.

        get_cuda_capability returns None off NVIDIA precisely so those
        integers never reach a comparison; the gate must treat None as "no".
        """
        with patch(
            "sglang.srt.utils.common.get_cuda_capability",
            return_value=None,
        ):
            self.assertFalse(CuteDSLGDNKernel().supports_prefill)
            self.assertFalse(CuteDSLKDAKernel().supports_prefill)


if __name__ == "__main__":
    unittest.main()
