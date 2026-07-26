"""`weak_ref_tensor` must not hard-require sgl-kernel (#164).

`sglang/srt/compilation/weak_ref_tensor.py` did `from sgl_kernel import
weak_ref_tensor` at module scope for every CUDA/HIP/MUSA/XPU build. sgl-kernel's
wheel is cubin-only with a gencode floor of sm_80 and has no ROCm build below
gfx942, so on a Turing card or a gfx900 card the package is simply not there --
and the import raises.

Where it raises is what made it hard to read: `_weak_ref_if_tensor` is called
from inside a breakable-CUDA-graph break point, in the window where no segment
is open, so the ModuleNotFoundError used to surface as
`breakable_cuda_graph.py:380 assert graph is not None`.

The replacement is a real fallback, not a guarded import: a non-owning tensor
built through the CUDA array interface -- the pure-Python equivalent of
sgl-kernel's `at::from_blob(data_ptr, sizes, strides, options)`. Keeping the
tensor itself instead (a strong reference) is only the last resort, because it
pins every per-layer intermediate in the graph mempool for the process's
lifetime.

Pure CPU tests: the interface description is checked directly, without ever
handing it to torch, so no device is needed.
"""

import unittest
from unittest import mock

import torch

from sglang.srt.compilation import weak_ref_tensor as wrt
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestImplSelection(unittest.TestCase):
    """Which implementation a rank gets is a capability question.

    The backend is mocked so the answer does not depend on the machine the
    test runs on.
    """

    def _patch_backend(self, **flags):
        for name in ("is_cuda", "is_hip", "is_musa", "is_xpu", "is_npu"):
            enabled = flags.get(name, False)
            patcher = mock.patch.object(wrt, name, return_value=enabled)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_sgl_kernel_present_is_used_verbatim(self):
        sentinel = object()
        self._patch_backend(is_cuda=True)
        patcher = mock.patch.object(wrt, "sgl_kernel_importable", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        fake = mock.Mock(weak_ref_tensor=sentinel)
        with mock.patch.dict("sys.modules", {"sgl_kernel": fake}):
            self.assertIs(wrt._select_weak_ref_tensor(), sentinel)

    def test_sgl_kernel_absent_falls_back_natively(self):
        """A Turing / gfx900 rank must get a working implementation, not an
        ImportError from inside a graph capture."""
        self._patch_backend(is_cuda=True)
        with mock.patch.object(wrt, "sgl_kernel_importable", return_value=False):
            self.assertIs(wrt._select_weak_ref_tensor(), wrt._native_weak_ref_tensor)

    def test_module_imports_without_any_accelerator(self):
        """Importing must not depend on a backend -- only USING it does."""
        self.assertTrue(callable(wrt.weak_ref_tensor))

    def test_unsupported_backend_still_refuses_clearly(self):
        self._patch_backend()
        with self.assertRaises(NotImplementedError):
            wrt._select_weak_ref_tensor()


class TestCudaArrayInterface(unittest.TestCase):
    """The description handed to torch must match the source tensor exactly --
    a wrong stride or itemsize here is silent memory corruption, so it is
    checked field by field."""

    def test_contiguous_2d_fp16(self):
        t = torch.zeros(4, 8, dtype=torch.float16, device="cpu")
        cai = wrt._cuda_array_interface(t)
        self.assertEqual(cai["shape"], (4, 8))
        self.assertEqual(cai["typestr"], "<f2")
        self.assertEqual(cai["data"], (t.data_ptr(), False))
        self.assertIsNone(cai["strides"])
        self.assertEqual(cai["version"], 2)

    def test_non_contiguous_strides_are_in_bytes(self):
        t = torch.zeros(4, 8, dtype=torch.float32, device="cpu")[:, ::2]
        self.assertFalse(t.is_contiguous())
        cai = wrt._cuda_array_interface(t)
        self.assertEqual(cai["shape"], (4, 4))
        self.assertEqual(cai["strides"], tuple(s * 4 for s in t.stride()))

    def test_dtypes_the_interface_cannot_express_are_refused(self):
        """bfloat16 has no array-interface type code. Refusing is required --
        guessing a code would reinterpret the bytes."""
        for dtype in (torch.bfloat16, torch.float8_e4m3fn):
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeError):
                    wrt._cuda_array_interface(torch.zeros(2, dtype=dtype, device="cpu"))

    def test_itemsize_matches_every_supported_dtype(self):
        for dtype, typestr in wrt._CAI_TYPESTR.items():
            with self.subTest(dtype=dtype):
                probe = torch.zeros(1, dtype=dtype, device="cpu")
                self.assertEqual(int(typestr[2:]), probe.element_size())


class TestNativeFallbackBehaviour(unittest.TestCase):
    def setUp(self):
        wrt._identity_warned.clear()

    def _pretend_device_tensor(self):
        patcher = mock.patch.object(wrt, "_is_device_tensor", return_value=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cpu_and_empty_tensors_pass_through(self):
        """Nothing to reclaim, and no device interface to describe."""
        t = torch.zeros(4, 4, device="cpu")
        self.assertIs(wrt._native_weak_ref_tensor(t), t)
        empty = torch.zeros(0, device="cpu")
        self.assertIs(wrt._native_weak_ref_tensor(empty), empty)

    def test_unexpressible_dtype_degrades_to_a_strong_ref_and_says_so(self):
        """Last resort: correct but memory-hungry, so it must be visible."""
        t = torch.zeros(4, dtype=torch.bfloat16, device="cpu")
        self._pretend_device_tensor()
        with self.assertLogs(wrt.logger, level="WARNING") as logs:
            self.assertIs(wrt._native_weak_ref_tensor(t), t)
        joined = " ".join(logs.output)
        self.assertIn("bfloat16", joined)
        self.assertIn("memory", joined.lower())

    def test_the_warning_is_emitted_once_per_dtype(self):
        t = torch.zeros(4, dtype=torch.bfloat16, device="cpu")
        self._pretend_device_tensor()
        with self.assertLogs(wrt.logger, level="WARNING") as logs:
            wrt._native_weak_ref_tensor(t)
            wrt._native_weak_ref_tensor(t)
        self.assertEqual(len(logs.output), 1)

    def test_a_failing_interface_import_is_not_fatal(self):
        """If torch on this platform will not take the interface, the capture
        must still run -- degraded, not dead."""
        t = torch.zeros(4, dtype=torch.float16, device="cpu")
        self._pretend_device_tensor()
        blew_up = RuntimeError("no array-interface ingestion here")
        with mock.patch.object(wrt.torch, "as_tensor", side_effect=blew_up):
            with self.assertLogs(wrt.logger, level="WARNING"):
                self.assertIs(wrt._native_weak_ref_tensor(t), t)


class TestWeakRefTensorsWrapper(unittest.TestCase):
    """The plural helper's structure handling must not move."""

    def test_single_list_and_tuple(self):
        with mock.patch.object(wrt, "weak_ref_tensor", side_effect=lambda t: t):
            t = torch.zeros(2, device="cpu")
            self.assertIs(wrt.weak_ref_tensors(t), t)
            self.assertEqual(wrt.weak_ref_tensors([t, t]), [t, t])
            self.assertEqual(wrt.weak_ref_tensors((t, t)), (t, t))
            with self.assertRaises(ValueError):
                wrt.weak_ref_tensors(7)


if __name__ == "__main__":
    unittest.main()
