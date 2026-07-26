"""The bfloat16 fallback must be decided vendor-first (#171).

`get_device_capability()[0] < 8` reads a number whose namespace depends on the
torch build, while the threshold 8 is an NVIDIA compute capability. The two
COLLIDE: gfx900 reports (9, 0) -- the same integer as Hopper -- so `9 < 8` is
False and the float16 fallback never fires on a Vega 64, a card with no
bfloat16 at all.

That is the dangerous direction: not a loud refusal but a SILENTLY WRONG dtype,
on the load path of every rank. (It is why the second host passes
`--dtype float16` by hand.)

Pure CPU tests: vendor and capability are mocked, no GPU is touched.
"""

import types
import unittest
from unittest import mock

from sglang.srt.model_executor import model_runner as mr


class TestBf16FallbackIsVendorFirst(unittest.TestCase):
    def _run(self, *, hip, capability=(9, 0), bf16=None, arch="gfx900"):
        cuda_ns = types.SimpleNamespace(
            get_device_capability=lambda *a, **k: capability,
            is_bf16_supported=(
                (lambda: bf16) if bf16 is not None
                else mock.Mock(side_effect=RuntimeError("probe blew up"))
            ),
            get_device_properties=(
                (lambda *a, **k: types.SimpleNamespace(gcnArchName=arch))
                if arch is not None
                else mock.Mock(side_effect=RuntimeError("no properties"))
            ),
        )
        with mock.patch.object(mr.torch, "version",
                               types.SimpleNamespace(hip=hip)), \
             mock.patch.object(mr.torch, "cuda", cuda_ns):
            return mr._needs_float16_fallback()

    # ---- the defect this exists to prevent ------------------------------
    def test_gfx900_falls_back_even_though_rocm_CLAIMS_bf16(self):
        """MEASURED on a Vega 64: torch.cuda.is_bf16_supported() returns True
        while bf16 matmul is 62% SLOWER than fp16 (2.885 vs 1.785 ms) -- ROCm
        reports the dtype as usable (emulated) and says nothing about hardware.
        Trusting it was the original bug; the arch family is the honest signal."""
        self.assertTrue(
            self._run(hip="6.3.0", capability=(9, 0), bf16=True, arch="gfx900")
        )

    def test_arch_suffix_is_stripped(self):
        """gcnArchName carries feature suffixes, e.g. 'gfx900:xnack-'."""
        self.assertTrue(
            self._run(hip="6.3.0", bf16=True, arch="gfx900:xnack-")
        )

    def test_amd_with_bf16_keeps_bfloat16(self):
        """MI300-class: has bf16, so no fallback -- must not over-trigger and
        silently downcast a capable card."""
        self.assertFalse(
            self._run(hip="6.3.0", capability=(9, 4), bf16=True, arch="gfx942")
        )

    def test_unknown_arch_is_left_alone(self):
        """Only families measured to lack bf16 are listed; an unknown or newer
        arch must not be downgraded on a guess."""
        self.assertFalse(self._run(hip="6.3.0", bf16=True, arch="gfx1100"))

    def test_amd_probe_failure_is_not_fatal(self):
        """A probe that raises must not take the server down; it warns and
        leaves the dtype alone."""
        with self.assertLogs(mr.logger, level="WARNING") as logs:
            self.assertFalse(self._run(hip="6.3.0", capability=(9, 0), arch=None))
        self.assertIn("--dtype float16", " ".join(logs.output))

    # ---- NVIDIA behaviour must not move (regression criterion) ----------
    def test_cuda_sm75_still_falls_back(self):
        self.assertTrue(self._run(hip=None, capability=(7, 5)))

    def test_cuda_sm86_keeps_bfloat16(self):
        self.assertFalse(self._run(hip=None, capability=(8, 6)))

    def test_cuda_sm120_keeps_bfloat16(self):
        self.assertFalse(self._run(hip=None, capability=(12, 0)))

    def test_cuda_never_consults_the_amd_arch_path(self):
        """On CUDA the numeric comparison alone decides, exactly as before --
        so a bf16 probe that would answer differently must be ignored."""
        # bf16=True would say "no fallback", but sm70 must still fall back
        self.assertTrue(self._run(hip=None, capability=(7, 0), bf16=True))


if __name__ == "__main__":
    unittest.main()
