"""#269: GGUF on an unsupported-capability card must hard-fail loudly, early.

sgl-kernel's GGUF cubins are gencode-restricted to sm_80+ (no PTX fallback),
so Turing (sm75; 2080/2080 Ti/T4) holds no code they can execute (#212). The
``_has_sgl_gguf_kernels`` flag (set at process start, when the cubin import
either succeeds or raises) already recorded this, but nothing ever consulted
it, and ``GGUFConfig.get_min_capability()`` returned 60 (Pascal) -- low enough
that a real sm75 card (capability 75) sailed straight past
``_enforce_capability_floor``'s NVIDIA-namespace numeric floor unnoticed. GGUF
was then only ever caught later, deep inside a kernel launch mid-forward,
after the checkpoint had already been loaded.

This module tests, hermetically (no GPU, no real sgl_kernel import):
  * ``GGUFConfig.get_min_capability()`` is now 80 (not 60) on CUDA, and
    unchanged (21) on MUSA;
  * ``GGUFConfig.supports_current_device()`` -- previously absent, so the
    generic vendor-first gate (#171) always fell through to the numeric
    floor -- now answers functionally by reusing ``_has_sgl_gguf_kernels``,
    and is None off CUDA (so MUSA/CPU/other vendors keep their own path);
  * wired end-to-end through the *actual*, already-existing
    ``_enforce_capability_floor`` gate in ``model_loader/loader.py`` (#171):
    a GGUF boot on a card where the cubin import failed raises ValueError
    before any weight is read, and a GGUF boot where it succeeded is admitted.

Run:  PYTHONPATH=<repo>/python python -m pytest \
        test/registered/unit/quantization/test_gguf_capability_floor.py -q
"""

import types
import unittest
from unittest import mock

from sglang.srt.layers.quantization import gguf as gguf_mod
from sglang.srt.model_loader import loader as loader_mod


def _model_config(name="gguf"):
    return types.SimpleNamespace(quantization=name)


class TestGGUFMinCapability(unittest.TestCase):
    """get_min_capability() must reflect the real sm_80 cubin floor (#212)."""

    def test_min_capability_is_sm80_on_cuda(self):
        with mock.patch.object(gguf_mod, "_is_musa", False):
            self.assertEqual(gguf_mod.GGUFConfig.get_min_capability(), 80)

    def test_min_capability_unchanged_on_musa(self):
        with mock.patch.object(gguf_mod, "_is_musa", True):
            self.assertEqual(gguf_mod.GGUFConfig.get_min_capability(), 21)

    def test_min_capability_is_no_longer_the_stale_sm60_floor(self):
        # The regression this guard exists for: 75 (sm75) >= 60 admitted a
        # Turing card silently. Pin the new value away from the old one so a
        # future edit cannot quietly reintroduce it.
        with mock.patch.object(gguf_mod, "_is_musa", False):
            self.assertNotEqual(gguf_mod.GGUFConfig.get_min_capability(), 60)


class TestGGUFSupportsCurrentDevice(unittest.TestCase):
    """supports_current_device() must wire in _has_sgl_gguf_kernels (#269)."""

    def test_reflects_kernel_flag_true_on_cuda(self):
        cfg = gguf_mod.GGUFConfig()
        with mock.patch.object(gguf_mod, "_is_cuda", True), mock.patch.object(
            gguf_mod, "_has_sgl_gguf_kernels", True
        ):
            self.assertIs(cfg.supports_current_device(), True)

    def test_reflects_kernel_flag_false_on_cuda(self):
        """The sm75 case: cubin import failed, so this must answer False --
        not None, which would fall through and re-expose the numeric floor
        as the only line of defense."""
        cfg = gguf_mod.GGUFConfig()
        with mock.patch.object(gguf_mod, "_is_cuda", True), mock.patch.object(
            gguf_mod, "_has_sgl_gguf_kernels", False
        ):
            self.assertIs(cfg.supports_current_device(), False)

    def test_none_off_cuda_regardless_of_stale_kernel_flag(self):
        """_has_sgl_gguf_kernels is only ever set True inside `if _is_cuda`,
        so it is always False off-CUDA -- must not be reported as a real
        negative there (MUSA/NPU/CPU take their own path)."""
        cfg = gguf_mod.GGUFConfig()
        with mock.patch.object(gguf_mod, "_is_cuda", False), mock.patch.object(
            gguf_mod, "_has_sgl_gguf_kernels", False
        ):
            self.assertIsNone(cfg.supports_current_device())


class TestGGUFCapabilityFloorEndToEnd(unittest.TestCase):
    """Wired through the real, already-existing _enforce_capability_floor
    gate (#171, model_loader/loader.py) -- no new call site, no new
    comparison, just a previously-blind config now answering correctly."""

    def _run(self, *, is_cuda, has_kernels, hip, capability):
        cfg = gguf_mod.GGUFConfig()
        with mock.patch.object(gguf_mod, "_is_cuda", is_cuda), mock.patch.object(
            gguf_mod, "_has_sgl_gguf_kernels", has_kernels
        ), mock.patch.object(
            loader_mod.torch, "version", types.SimpleNamespace(hip=hip)
        ), mock.patch.object(
            loader_mod, "get_device_capability", return_value=capability
        ):
            loader_mod._enforce_capability_floor(cfg, _model_config())

    def test_sm75_turing_is_rejected_before_any_weight_load(self):
        """The concrete #212/#269 scenario: a 2080 Ti (sm75), cubin import
        already failed at process start."""
        with self.assertRaises(ValueError) as ctx:
            self._run(
                is_cuda=True, has_kernels=False, hip=None, capability=(7, 5)
            )
        msg = str(ctx.exception)
        self.assertIn("gguf", msg)
        self.assertIn("does not exist here", msg)

    def test_sm80_ampere_is_admitted(self):
        """Cubin import succeeded (real sm80+ card) -- must not raise."""
        self._run(is_cuda=True, has_kernels=True, hip=None, capability=(8, 6))

    def test_numeric_fallback_still_catches_sm75_even_without_the_hook(self):
        """Defense in depth: even if supports_current_device() were somehow
        bypassed (None), the corrected 80 floor alone still rejects sm75
        through the pre-existing NVIDIA-namespace numeric path."""
        with mock.patch.object(gguf_mod, "_is_cuda", False):
            # _is_cuda False -> supports_current_device() returns None ->
            # _enforce_capability_floor falls through to the numeric compare.
            cfg = gguf_mod.GGUFConfig()
            with mock.patch.object(
                loader_mod.torch, "version", types.SimpleNamespace(hip=None)
            ), mock.patch.object(
                loader_mod, "get_device_capability", return_value=(7, 5)
            ):
                with self.assertRaises(ValueError) as ctx:
                    loader_mod._enforce_capability_floor(cfg, _model_config())
        self.assertIn("Minimum capability: 80", str(ctx.exception))
        self.assertIn("Current capability: 75", str(ctx.exception))

    def test_rocm_gguf_warns_instead_of_silently_admitting(self):
        """GGUF is not documented as ROCm-general-purpose here; the vendor
        gate must at least warn rather than pretend the floor was checked,
        matching the existing #171 ROCm contract for an unteached config."""
        with self.assertLogs(loader_mod.logger, level="WARNING"):
            self._run(
                is_cuda=False, has_kernels=False, hip="6.3.0", capability=(9, 0)
            )


if __name__ == "__main__":
    unittest.main()
