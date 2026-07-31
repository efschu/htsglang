"""Capability gates must name their vendor (#171).

`get_device_capability()` answers in whichever namespace the torch build
belongs to, and the namespaces COLLIDE: gfx900 reports (9, 0) -- the same
integer as NVIDIA Hopper -- gfx1030 reports (10, 3) like a Blackwell B300, and
gfx1100 reports (11, 0). Every threshold written in this codebase ("sm80",
"SM100", "compute capability >= 9") is an NVIDIA statement, so applying it to a
number read in another vendor's namespace fails in the DANGEROUS direction: the
AMD card sails through the gate and dies later inside a kernel that does not
exist there, instead of being refused at startup.

The root fix is the helper family in `sglang.srt.utils.common`: helpers that
carry the vendor IN THEIR NAME and answer False/None off that vendor, so the
blind comparison is not expressible. These tests pin both halves --

  * on a mocked NVIDIA context every helper answers exactly what the bare
    comparison it replaced answered (the regression criterion), and
  * on a mocked ROCm context (is_hip, gfx arch) it answers "not applicable"
    rather than a colliding number.

Pure CPU tests: vendor and capability are mocked throughout, no GPU is touched.
"""

import contextlib
import inspect
import types
import unittest
from unittest import mock

from sglang.srt.utils import common


@contextlib.contextmanager
def _mocked_capability():
    """Per-device capability caches must not leak between mocked contexts.

    The gates are cached per device id (#343); a mocked context changes what
    the underlying probe answers for the SAME id, which no production caller
    can do. Cleared on both edges so the order of the tests cannot matter.
    """
    common.clear_per_device_gate_caches()
    try:
        yield
    finally:
        common.clear_per_device_gate_caches()


@contextlib.contextmanager
def nvidia(capability=(9, 0)):
    """A mocked NVIDIA context reporting ``capability``."""
    with _mocked_capability(), mock.patch.object(
        common, "is_cuda", lambda: True
    ), mock.patch.object(
        common, "get_device_capability_no_init", lambda device=None: capability
    ), mock.patch.object(
        common.torch, "version", types.SimpleNamespace(hip=None, cuda="12.8")
    ):
        yield


@contextlib.contextmanager
def rocm(arch="gfx942", capability=(9, 4)):
    """A mocked ROCm context: torch.version.hip set, gfx arch reported, and the
    capability tuple torch derives from it -- the colliding number."""
    props = types.SimpleNamespace(gcnArchName=f"{arch}:sramecc+:xnack-")
    with _mocked_capability(), mock.patch.object(
        common, "is_cuda", lambda: False
    ), mock.patch.object(
        common, "get_device_capability_no_init", lambda device=None: capability
    ), mock.patch.object(
        common.torch, "version", types.SimpleNamespace(hip="6.3.0", cuda=None)
    ), mock.patch.object(
        common.torch.cuda, "get_device_properties", lambda device=0: props
    ):
        yield


class TestNvidiaNamespaceHelpers(unittest.TestCase):
    """On NVIDIA the new helpers must read exactly what the old bare
    comparisons read -- otherwise the fix moves CUDA behaviour."""

    def test_capability_and_sm_are_the_reported_numbers(self):
        with nvidia((8, 6)):
            self.assertEqual(common.get_cuda_capability(), (8, 6))
            self.assertEqual(common.get_cuda_sm(), 86)

    def test_at_least_matches_the_comparison_it_replaces(self):
        with nvidia((9, 0)):
            self.assertTrue(common.cuda_sm_at_least(9))
            self.assertTrue(common.cuda_sm_at_least(8))
            self.assertFalse(common.cuda_sm_at_least(10))

    def test_below_matches_the_comparison_it_replaces(self):
        with nvidia((7, 5)):
            self.assertTrue(common.cuda_sm_below(8))
            self.assertFalse(common.cuda_sm_below(7))

    def test_range_is_half_open(self):
        with nvidia((8, 9)):
            self.assertTrue(common.cuda_sm_in_range((8, 0), (10, 0)))
        with nvidia((10, 0)):
            self.assertFalse(common.cuda_sm_in_range((8, 0), (10, 0)))

    def test_major_in(self):
        with nvidia((12, 0)):
            self.assertTrue(common.cuda_sm_major_in([10, 11, 12]))
            self.assertFalse(common.cuda_sm_major_in([9]))

    def test_the_amd_reader_is_silent_on_nvidia(self):
        with nvidia((9, 0)):
            self.assertIsNone(common.get_hip_arch())
            self.assertFalse(common.hip_arch_in(["gfx9"]))


class TestRocmDoesNotAnswerNvidiaQuestions(unittest.TestCase):
    """The defect these helpers exist to prevent: an AMD arch clearing an
    NVIDIA threshold because the two namespaces share integers."""

    def test_gfx942_does_not_pass_an_sm90_gate(self):
        """gfx942 reports (9, 4). `major >= 9` said yes; the vendor says no."""
        with rocm("gfx942", (9, 4)):
            self.assertIsNone(common.get_cuda_capability())
            self.assertIsNone(common.get_cuda_sm())
            self.assertFalse(common.cuda_sm_at_least(9))

    def test_gfx1030_is_not_a_blackwell_b300(self):
        """gfx1030 reports (10, 3), the same tuple as a B300."""
        with rocm("gfx1030", (10, 3)):
            self.assertFalse(common.cuda_sm_at_least(10))
            self.assertFalse(common.cuda_sm_major_in([10, 11, 12]))

    def test_gfx1100_is_not_beyond_blackwell(self):
        with rocm("gfx1100", (11, 0)):
            self.assertFalse(common.cuda_sm_at_least(10))

    def test_gfx900_is_not_hopper_and_not_a_small_nvidia_card(self):
        """Both directions: (9, 0) must neither clear an sm90 gate nor be
        described as 'below sm80' -- an AMD card is not a small NVIDIA one."""
        with rocm("gfx900", (9, 0)):
            self.assertFalse(common.cuda_sm_at_least(9))
            self.assertFalse(common.cuda_sm_below(8))

    def test_the_amd_reader_answers_in_the_amd_namespace(self):
        with rocm("gfx942", (9, 4)):
            self.assertEqual(common.get_hip_arch(), "gfx942")
            self.assertTrue(common.hip_arch_in(["gfx942"]))
            self.assertTrue(common.hip_arch_in(["gfx94"]))
            self.assertFalse(common.hip_arch_in(["gfx95"]))

    def test_gfx_family_predicates_ride_on_the_amd_reader(self):
        for fn, arch, expected in (
            (common.is_gfx942_supported, "gfx942", True),
            (common.is_gfx942_supported, "gfx950", False),
            (common.is_gfx95_supported, "gfx950", True),
            (common.is_gfx95_supported, "gfx942", False),
        ):
            with self.subTest(fn=fn.__name__, arch=arch):
                fn.cache_clear()
                with rocm(arch):
                    self.assertEqual(fn(), expected)
                fn.cache_clear()

    def test_gfx_family_predicates_are_false_on_nvidia(self):
        for fn in (common.is_gfx942_supported, common.is_gfx95_supported):
            with self.subTest(fn=fn.__name__):
                fn.cache_clear()
                with nvidia():
                    self.assertFalse(fn())
                fn.cache_clear()


class TestNoDeviceDegradesQuietly(unittest.TestCase):
    """A CPU-only process must get "not applicable", not an exception: these
    helpers sit on import paths that run in GPU-passive processes."""

    def test_cpu_build(self):
        with mock.patch.object(common, "is_cuda", lambda: False), mock.patch.object(
            common.torch, "version", types.SimpleNamespace(hip=None, cuda=None)
        ):
            self.assertIsNone(common.get_cuda_capability())
            self.assertIsNone(common.get_cuda_sm())
            self.assertFalse(common.cuda_sm_at_least(8))
            self.assertFalse(common.cuda_sm_below(8))
            self.assertIsNone(common.get_hip_arch())

    def test_a_failing_probe_is_not_fatal(self):
        def boom(device=None):
            raise RuntimeError("no CUDA driver")

        with mock.patch.object(common, "is_cuda", lambda: True), mock.patch.object(
            common, "get_device_capability_no_init", boom
        ):
            self.assertIsNone(common.get_cuda_capability())
            self.assertFalse(common.cuda_sm_at_least(8))


class TestFixedSitesAskInTheNvidiaNamespace(unittest.TestCase):
    """The gates the sweep found. Each one asserted in both directions: the
    NVIDIA answer is unchanged, the ROCm answer is no longer the collision."""

    def test_programmatic_dependent_launch(self):
        from sglang.srt.layers import fused_qk_rmsnorm_rope_gate as mod

        with nvidia((9, 0)):
            self.assertTrue(mod._pdl_supported())
        with nvidia((8, 6)):
            self.assertFalse(mod._pdl_supported())
        # PDL is a CUDA feature; gfx942's (9, 4) used to enable it on ROCm.
        with rocm("gfx942", (9, 4)):
            self.assertFalse(mod._pdl_supported())

    def test_cutedsl_blackwell_gates(self):
        from sglang.srt.layers.attention.linear.kernels import (
            gdn_cutedsl,
            kda_cutedsl,
        )

        for mod in (gdn_cutedsl, kda_cutedsl):
            with self.subTest(module=mod.__name__):
                with nvidia((10, 0)):
                    self.assertTrue(mod._is_blackwell())
                with nvidia((9, 0)):
                    self.assertFalse(mod._is_blackwell())
                # (10, 3) and (11, 0) are AMD arches, not Blackwell parts.
                with rocm("gfx1030", (10, 3)):
                    self.assertFalse(mod._is_blackwell())
                with rocm("gfx1100", (11, 0)):
                    self.assertFalse(mod._is_blackwell())

    def test_marlin_fp8_auto_enable_range(self):
        from sglang.srt.layers.quantization import fp8_utils

        with nvidia((8, 6)):
            self.assertTrue(fp8_utils.can_auto_enable_marlin_fp8())
        with nvidia((8, 9)):
            self.assertFalse(fp8_utils.can_auto_enable_marlin_fp8())
        # gfx803 reports (8, 0) and would land inside the sm80..88 range.
        with rocm("gfx803", (8, 0)):
            self.assertFalse(fp8_utils.can_auto_enable_marlin_fp8())

    def test_marlin_supported_types_floor(self):
        from sglang.srt.layers.quantization import marlin_utils

        with nvidia((8, 0)):
            self.assertNotEqual(marlin_utils.query_marlin_supported_quant_types(), [])
        with nvidia((7, 5)):
            self.assertEqual(marlin_utils.query_marlin_supported_quant_types(), [])
        # Marlin has no ROCm kernel at all; gfx900's (9, 0) cleared the floor.
        with rocm("gfx900", (9, 0)):
            self.assertEqual(marlin_utils.query_marlin_supported_quant_types(), [])

    def test_moe_wna16_awq_floor_does_not_apply_on_rocm(self):
        from sglang.srt.layers.quantization import moe_wna16

        cfg = {"quant_method": "awq", "bits": 4, "desc_act": False}

        def compatible(*, hip, capability):
            with mock.patch.object(
                moe_wna16, "get_device_capability", lambda: capability
            ), mock.patch.object(moe_wna16, "is_hip", lambda: hip):
                return moe_wna16.MoeWNA16Config.is_moe_wna16_compatible(cfg)

        self.assertTrue(compatible(hip=False, capability=(7, 5)))
        self.assertFalse(compatible(hip=False, capability=(7, 0)))
        # On ROCm the number is an AMD arch: the NVIDIA floor is not applied,
        # in either direction. A gfx part reporting (7, 0) was refused for a
        # threshold that says nothing about it.
        self.assertTrue(compatible(hip=True, capability=(7, 0)))

    def test_flashattention_v3_refuses_rocm_before_reading_the_number(self):
        """Source order matters: after the number the vendor gate is useless,
        because gfx942's (9, 4) already satisfies `major == 9`."""
        from sglang.srt.layers.attention import attention_registry

        src = inspect.getsource(attention_registry.create_flashattention_v3_backend)
        vendor_at = src.find("assert not _is_hip")
        cap_at = src.find("or major == 9")
        self.assertNotEqual(vendor_at, -1, "vendor gate is missing entirely")
        self.assertNotEqual(cap_at, -1, "capability gate disappeared")
        self.assertLess(vendor_at, cap_at)
        self.assertIn("no ROCm kernel", src)

    def test_dsa_backends_default_in_the_nvidia_namespace(self):
        from sglang.srt.arg_groups import overrides

        src = inspect.getsource(overrides._dsa_split_backend_resolution)
        self.assertIn("cuda_sm_at_least(10)", src)
        self.assertNotIn("major >= 10", src)
        src = inspect.getsource(overrides._dsa_kv_cache_dtype_default)
        self.assertIn("cuda_sm_at_least(10)", src)
        self.assertNotIn("major >= 10", src)


if __name__ == "__main__":
    unittest.main()
