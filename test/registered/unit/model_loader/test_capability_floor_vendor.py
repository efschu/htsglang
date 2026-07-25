"""The capability floor must be applied VENDOR-FIRST (#171).

`get_min_capability()` returns NVIDIA compute capabilities, so it may only be
compared against a capability read in the NVIDIA namespace.
`get_device_capability()` answers in whichever namespace the torch build belongs
to, and the two COLLIDE: gfx900 reports (9, 0) -- the same integer as Hopper.

The old gate compared them unconditionally and so failed in the DANGEROUS
direction: an 8 GB Vega 64 passed an sm89 fp8 floor and died later inside a
kernel that does not exist there, instead of being refused at startup.

These are pure CPU tests: vendor and capability are mocked, no GPU is touched.
"""

import types
import unittest
from unittest import mock

from sglang.srt.model_loader import loader as loader_mod


class _CfgNoHook:
    """A config that predates supports_current_device() entirely."""

    def __init__(self, min_cap=89):
        self._min_cap = min_cap

    def get_min_capability(self):
        return self._min_cap


class _Cfg(_CfgNoHook):
    """Minimal stand-in for a QuantizationConfig, with the vendor hook."""

    def __init__(self, min_cap=89, supports=None):
        super().__init__(min_cap)
        self._supports = supports

    def supports_current_device(self):
        return self._supports


def _model_config(name="fp8"):
    return types.SimpleNamespace(quantization=name)


class TestCapabilityFloorIsVendorFirst(unittest.TestCase):
    def _run(self, cfg, *, hip, capability):
        """Invoke the gate with a mocked vendor and capability."""
        with mock.patch.object(loader_mod.torch, "version",
                               types.SimpleNamespace(hip=hip)), \
             mock.patch.object(loader_mod, "get_device_capability",
                               return_value=capability):
            loader_mod._enforce_capability_floor(cfg, _model_config())

    # ---- the defect this exists to prevent ------------------------------
    def test_amd_arch_does_not_sail_through_an_nvidia_floor(self):
        """gfx900 reports (9, 0); 90 >= 89 must NOT be read as 'supported'."""
        cfg = _Cfg(min_cap=89, supports=False)  # functional probe says no
        with self.assertRaises(ValueError) as ctx:
            self._run(cfg, hip="6.3.0", capability=(9, 0))
        msg = str(ctx.exception)
        self.assertIn("not comparable", msg.lower().replace("-", " ")
                      if "not comparable" in msg.lower() else msg.lower())
        self.assertIn("does not exist here", msg)

    def test_amd_with_working_kernel_is_admitted(self):
        """MI300-class: the functional probe says yes, so it must pass."""
        cfg = _Cfg(min_cap=89, supports=True)
        self._run(cfg, hip="6.3.0", capability=(9, 4))  # must not raise

    def test_amd_unknown_support_warns_and_does_not_pretend(self):
        """No hook answer on ROCm: the floor is NOT enforceable -- say so.

        Deliberately does not raise: rejecting every unteached config on ROCm
        would break working setups. But it must not silently look like a passed
        check either, so a warning naming the namespace problem is required.
        """
        cfg = _Cfg(min_cap=89, supports=None)
        with self.assertLogs(loader_mod.logger, level="WARNING") as logs:
            self._run(cfg, hip="6.3.0", capability=(9, 0))
        joined = " ".join(logs.output)
        self.assertIn("not comparable", joined)
        self.assertIn("CANNOT be enforced", joined)

    # ---- NVIDIA behaviour must not move (the regression criterion) -------
    def test_cuda_below_floor_still_rejected(self):
        cfg = _Cfg(min_cap=89, supports=None)
        with self.assertRaises(ValueError) as ctx:
            self._run(cfg, hip=None, capability=(7, 5))  # sm75 -> 75 < 89
        self.assertIn("Minimum capability: 89", str(ctx.exception))
        self.assertIn("Current capability: 75", str(ctx.exception))

    def test_cuda_at_or_above_floor_still_admitted(self):
        cfg = _Cfg(min_cap=89, supports=None)
        self._run(cfg, hip=None, capability=(8, 9))   # exactly at the floor
        self._run(cfg, hip=None, capability=(12, 0))  # well above

    def test_cuda_hook_is_not_consulted_for_the_number(self):
        """On CUDA a config returning None must fall through to the numbers,
        i.e. the numeric path is still the one that decides."""
        cfg = _Cfg(min_cap=89, supports=None)
        with self.assertRaises(ValueError):
            self._run(cfg, hip=None, capability=(8, 0))  # 80 < 89

    def test_config_without_the_hook_still_works(self):
        """Configs that never heard of supports_current_device must not break."""
        cfg = _CfgNoHook(min_cap=80)
        self.assertFalse(hasattr(cfg, "supports_current_device"))
        self._run(cfg, hip=None, capability=(8, 6))  # must not raise

    # ---- the hook's own contract ----------------------------------------
    def test_base_config_default_is_unknown_not_supported(self):
        """A config that cannot answer for its vendor must return None, never
        True -- claiming support by default is the dangerous direction."""
        from sglang.srt.layers.quantization.base_config import QuantizationConfig

        self.assertIsNone(
            QuantizationConfig.supports_current_device(object())
        )


if __name__ == "__main__":
    unittest.main()
