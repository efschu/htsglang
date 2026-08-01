"""#417 -- which FlashMLA implementation each device gets.

Boot 11 died on TP0 (5090, sm120) with::

    RuntimeError: Sparse Attention Forward Kernel is only supported on
    SM90a and SM100f architectures.

reached because the sparse-prefill branch was entered on a card that has no
sparse-prefill kernel, and on TP1/TP2 (3080, sm86) one step later because the
dense branch fell through to the FlashMLA CUDA kernel, which Ampere does not
have either.

Both are dispatch questions, and on a heterogeneous group they have different
answers per rank in the same run. This file pins the gates and the dispatch
they drive, GPU-free: capability is mocked rather than sniffed, so the whole
matrix runs on CPU-only CI.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.attention import flash_mla_arch
from sglang.srt.layers.attention.flash_mla_arch import (
    flash_mla_cuda_kernel_supported,
    flash_mla_sparse_fwd_supported,
    resolve_flashmla_fallback_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

# (capability, has a FlashMLA CUDA kernel). The domain is stated by the
# kernels themselves: SM90a and SM100f.
_CAPABILITY_MATRIX = [
    ((8, 0), False),  # A100
    ((8, 6), False),  # RTX 3080 -- TP1/TP2 of this rig
    ((8, 9), False),  # L20 / Ada
    ((9, 0), True),  # H100
    ((10, 0), True),  # B200
    ((10, 3), True),  # GB300
    ((12, 0), False),  # RTX 5090 -- TP0 of this rig
    ((12, 1), False),  # DGX Spark GB10
]


class _CapabilityMixin:
    def setUp(self):
        super().setUp()
        self._clear()
        self.addCleanup(self._clear)

    @staticmethod
    def _clear():
        flash_mla_cuda_kernel_supported.cache_clear()
        flash_mla_sparse_fwd_supported.cache_clear()
        flash_mla_arch._is_sm12x.cache_clear()

    def _with_capability(self, capability, cuda=True):
        self._clear()
        return mock.patch.multiple(
            flash_mla_arch,
            is_cuda=lambda: cuda,
            get_device_capability_no_init=lambda device_id: capability,
        )

    def _with_capabilities(self, per_device, cuda=True):
        self._clear()
        return mock.patch.multiple(
            flash_mla_arch,
            is_cuda=lambda: cuda,
            get_device_capability_no_init=lambda device_id: per_device[device_id],
        )


class TestFlashMlaCapabilityGates(_CapabilityMixin, CustomTestCase):
    def test_cuda_kernel_domain(self):
        for capability, expected in _CAPABILITY_MATRIX:
            with self.subTest(sm=capability), self._with_capability(capability):
                self.assertEqual(flash_mla_cuda_kernel_supported(0), expected)

    def test_sparse_fwd_domain_is_the_same(self):
        """`flash_mla_sparse_fwd` and `flash_mla_with_kvcache` come from the
        same CUDA extension and refuse on the same architectures."""
        for capability, expected in _CAPABILITY_MATRIX:
            with self.subTest(sm=capability), self._with_capability(capability):
                self.assertEqual(flash_mla_sparse_fwd_supported(0), expected)

    def test_hopper_and_datacenter_blackwell_are_unchanged(self):
        """The architectures that work today must keep taking the CUDA kernel.

        This is the backward-compatibility assertion for the whole task: every
        gate is phrased so that only devices which currently *crash* change
        branch.
        """
        for capability in ((9, 0), (10, 0), (10, 3)):
            with self.subTest(sm=capability), self._with_capability(capability):
                self.assertTrue(flash_mla_cuda_kernel_supported(0))
                self.assertTrue(flash_mla_sparse_fwd_supported(0))

    def test_the_two_cards_of_this_rig_disagree_in_one_process(self):
        """#343: the 5090 and a 3080 must get different answers from the same
        process. A gate cached in a single slot would hand the second card the
        first one's answer -- which is the bug class #340 was."""
        per_device = {0: (12, 0), 1: (8, 6), 2: (8, 6)}
        with self._with_capabilities(per_device):
            self.assertFalse(flash_mla_cuda_kernel_supported(0))
            self.assertFalse(flash_mla_cuda_kernel_supported(1))
            # Ask again in the other order; nothing may have collapsed.
            self.assertFalse(flash_mla_cuda_kernel_supported(2))
            self.assertFalse(flash_mla_cuda_kernel_supported(0))

        # Same shape with a group that genuinely differs in the answer.
        per_device = {0: (9, 0), 1: (8, 6)}
        with self._with_capabilities(per_device):
            self.assertTrue(flash_mla_cuda_kernel_supported(0))
            self.assertFalse(flash_mla_cuda_kernel_supported(1))
            self.assertTrue(flash_mla_cuda_kernel_supported(0))

    def test_non_cuda_vendors_keep_their_own_path(self):
        for capability in ((9, 4), (11, 0)):
            with self.subTest(sm=capability), self._with_capability(
                capability, cuda=False
            ):
                self.assertTrue(flash_mla_cuda_kernel_supported(0))
                self.assertTrue(flash_mla_sparse_fwd_supported(0))


class TestFallbackBackendResolution(_CapabilityMixin, CustomTestCase):
    def _with_env(self, value, is_set):
        return mock.patch.object(
            flash_mla_arch.envs,
            "SGLANG_SM120_FLASHMLA_BACKEND",
            SimpleNamespace(get=lambda: value, is_set=lambda: is_set),
        )

    def test_default_flashinfer_survives_on_sm12x(self):
        with self._with_capability((12, 0)), self._with_env("flashinfer", False):
            self.assertEqual(resolve_flashmla_fallback_backend(0), "flashinfer")

    def test_default_flashinfer_is_not_selectable_on_ampere(self):
        """flashinfer's sparse-MLA kernels carry
        @supported_compute_capability([120, 121]). Handing an Ampere rank the
        env default would be handing it a kernel that refuses to run."""
        for capability in ((8, 0), (8, 6), (8, 9)):
            with self.subTest(sm=capability), self._with_capability(
                capability
            ), self._with_env("flashinfer", False):
                self.assertEqual(resolve_flashmla_fallback_backend(0), "triton")

    def test_explicit_triton_and_torch_apply_everywhere(self):
        """An explicit launch flag is a statement about the launch, not a probe
        (#343's rule for --fp8-gemm-backend)."""
        for requested in ("triton", "torch"):
            for capability in ((8, 6), (12, 0)):
                with self.subTest(
                    requested=requested, sm=capability
                ), self._with_capability(capability), self._with_env(requested, True):
                    self.assertEqual(resolve_flashmla_fallback_backend(0), requested)

    def test_explicit_flashinfer_off_sm12x_is_downgraded_loudly(self):
        with self._with_capability((8, 6)), self._with_env("flashinfer", True):
            with self.assertLogs(flash_mla_arch.logger, level="WARNING") as logs:
                self.assertEqual(resolve_flashmla_fallback_backend(0), "triton")
            self.assertTrue(
                any("flashinfer" in line for line in logs.output),
                "the downgrade must name the backend it replaced",
            )

    def test_resolution_is_per_device_not_per_process(self):
        """The regression this replaces: `SGLANG_SM120_FLASHMLA_BACKEND` used to
        be read once into a module global at import, so every rank of a mixed
        group got whichever card imported first."""
        per_device = {0: (12, 0), 1: (8, 6)}
        with self._with_capabilities(per_device), self._with_env("flashinfer", False):
            self.assertEqual(resolve_flashmla_fallback_backend(0), "flashinfer")
            self.assertEqual(resolve_flashmla_fallback_backend(1), "triton")
            self.assertEqual(resolve_flashmla_fallback_backend(0), "flashinfer")


class TestPortableEntryPointDispatch(CustomTestCase):
    """`flash_mla_with_kvcache_sm120` must consult the resolver per call."""

    def _call(self, backend):
        from sglang.srt.layers.attention import flash_mla_sm120

        kwargs = dict(
            q=torch.zeros(1, 1, 1, 8, dtype=torch.bfloat16),
            k_cache=torch.zeros(1, 4, 1, 8, dtype=torch.uint8),
            indices=torch.zeros(1, 4, dtype=torch.int32),
            head_dim_v=4,
            softmax_scale=1.0,
        )
        with mock.patch.object(
            flash_mla_sm120,
            "resolve_flashmla_fallback_backend",
            lambda device_id=None: backend,
        ):
            return flash_mla_sm120.flash_mla_with_kvcache_sm120(**kwargs)

    def test_flashinfer_branch(self):
        from sglang.srt.layers.attention import flash_mla_sm120

        sentinel = object()
        with mock.patch.object(
            flash_mla_sm120, "_flash_mla_flashinfer", lambda *a, **k: sentinel
        ):
            self.assertIs(self._call("flashinfer"), sentinel)

    def test_torch_branch(self):
        from sglang.srt.layers.attention import flash_mla_sm120

        with mock.patch.object(
            flash_mla_sm120,
            "_sm120_sparse_decode_fwd",
            lambda *a, **k: ("out", "lse"),
        ):
            self.assertEqual(self._call("torch"), ("out", "lse"))

    def test_triton_branch(self):
        import sglang.srt.layers.attention.flash_mla_sm120_triton as triton_mod

        with mock.patch.object(
            triton_mod,
            "flash_mla_sparse_decode_triton",
            lambda *a, **k: ("t_out", "t_lse"),
        ):
            self.assertEqual(self._call("triton"), ("t_out", "t_lse"))

    def test_no_module_level_backend_global_remains(self):
        """The frozen global is the bug; its absence is the fix."""
        from sglang.srt.layers.attention import flash_mla_sm120

        self.assertFalse(
            hasattr(flash_mla_sm120, "_sm120_default_backend"),
            "a module-level backend choice cannot be right for two cards at "
            "once (#343)",
        )


class TestBackendImportSmoke(CustomTestCase):
    """The DSV4 backend must still import with no CUDA device present.

    Cheap, but it is what catches an import-time capability probe sneaking
    back in -- a probe at import answers device 0 and creates a CUDA context
    in GPU-passive processes (#237, #343).
    """

    def test_backend_imports_without_cuda(self):
        import sglang.srt.layers.attention.deepseek_v4_backend as backend

        self.assertTrue(hasattr(backend, "DeepseekV4AttnBackend"))
        self.assertIs(
            backend.flash_mla_sparse_fwd_supported, flash_mla_sparse_fwd_supported
        )
        self.assertIs(
            backend.flash_mla_cuda_kernel_supported, flash_mla_cuda_kernel_supported
        )

    def test_dsv4_kv_kernels_import_without_cuda(self):
        from sglang.srt.layers.attention.dsv4 import (  # noqa: F401
            dequant_k_cache,
            index_buf_accessor,
            quant_k_cache,
        )

        self.assertTrue(callable(dequant_k_cache.dequantize_k_cache_paged))
        self.assertTrue(callable(quant_k_cache.quant_to_nope_fp8_rope_bf16_pack_triton))


if __name__ == "__main__":
    unittest.main()
