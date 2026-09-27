# SPDX-License-Identifier: Apache-2.0
"""WEG2-GRAPH-UPLOAD (27B b1 death, 27.09.2026 06:57:17Z, dkr27breleasedraftbar1w109270652).

D TP0 (5090) OOMed inside ``CUDAGraph.replay`` of the FIRST bs=3 TARGET_VERIFY replay of that process,
at card_free 5 MiB with ~1.5 GiB free in the torch cache: the missing bytes were the driver's first-launch
upload, which torch never pays at capture. These tests pin the helper that pays it at capture
(weg2/graph_upload.py), the capture hook, and the launcher default for group D -- all with mocks, no GPU.
"""

import os
import unittest
from unittest import mock

from sglang.test.test_utils import CustomTestCase

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import graph_upload as GU  # noqa: E402

MIB = 1 << 20


class FakeStream:
    cuda_stream = 0x5150


class FakeCuda:
    """torch.cuda stand-in: mem_get_info walks a list of free values."""

    def __init__(self, frees, capturing=False):
        self.frees = list(frees)
        self.capturing = capturing
        self.syncs = 0

    def is_current_stream_capturing(self):
        return self.capturing

    def current_stream(self):
        return FakeStream()

    def synchronize(self):
        self.syncs += 1

    def mem_get_info(self):
        return self.frees.pop(0), 32088 * MIB


class FakeLib:
    def __init__(self, rc=0):
        self.rc = rc
        self.calls = []

    def cuGraphUpload(self, h, s):
        self.calls.append((h.value, s.value))
        return self.rc


class FakeGraph:
    def __init__(self, handle=0xABCD):
        self.handle = handle

    def raw_cuda_graph_exec(self):
        return self.handle


class LazyGraph:
    """torch 2.11 with keep_graph=True: no exec until instantiate() (rc12i)."""

    def __init__(self, handle=0xBEEF):
        self.handle = handle
        self.instantiated = 0

    def raw_cuda_graph_exec(self):
        if not self.instantiated:
            raise RuntimeError("You cannot access the raw cudaGraphExec_t instance until instantiate() "
                               "has been called")
        return self.handle

    def instantiate(self):
        self.instantiated += 1


class TheLazyExecIsInstantiatedNotSkipped(CustomTestCase):
    """rc12i metal (27.09., 2d680cbe66): all 36 shapes were 'skipped: RuntimeError
    ... until instantiate() has been called'. The helper must instantiate."""

    def test_a_lazy_graph_is_instantiated_once_and_uploaded(self):
        g, lib = LazyGraph(), FakeLib()
        with self.assertLogs(GU.logger, level="INFO") as cm:
            got = GU.upload_after_capture(g, "ShapeKey(size=3)", FakeStream(), lib=lib,
                                          cuda=FakeCuda([900 * MIB, 890 * MIB]))
        self.assertEqual(got, (0, 10.0))
        self.assertEqual(g.instantiated, 1)
        self.assertEqual(lib.calls, [(0xBEEF, 0x5150)])
        self.assertIn("instantiated_here=1", cm.output[-1])
        self.assertNotIn("skipped", cm.output[-1])

    def test_an_already_instantiated_graph_is_not_instantiated_again(self):
        g = LazyGraph(); g.instantiated = 1
        with self.assertLogs(GU.logger, level="INFO") as cm:
            GU.upload_after_capture(g, "k", FakeStream(), lib=FakeLib(), cuda=FakeCuda([5 * MIB, 5 * MIB]))
        self.assertEqual(g.instantiated, 1)
        self.assertIn("instantiated_here=0", cm.output[-1])

    def test_a_lazy_graph_without_instantiate_is_a_skipped_line(self):
        class NoInst(LazyGraph):
            instantiate = None

        lib = FakeLib()
        with self.assertLogs(GU.logger, level="WARNING") as cm:
            self.assertIsNone(GU.upload_after_capture(NoInst(), "k", FakeStream(), lib=lib, cuda=FakeCuda([1, 1])))
        self.assertEqual(lib.calls, [])
        self.assertIn("skipped: RuntimeError", cm.output[-1])


class TheUploadIsPaidAtCaptureAndPrinted(CustomTestCase):
    def test_upload_calls_the_driver_with_exec_and_stream_and_prints_mib(self):
        lib, cuda = FakeLib(), FakeCuda([1000 * MIB, 988 * MIB])
        with self.assertLogs(GU.logger, level="INFO") as cm:
            got = GU.upload_after_capture(FakeGraph(), "bs3/verify", FakeStream(), lib=lib, cuda=cuda)
        self.assertEqual(got, (0, 12.0))
        self.assertEqual(lib.calls, [(0xABCD, 0x5150)])
        self.assertEqual(cuda.syncs, 2)
        line = cm.output[-1]
        self.assertIn("WEG2-GRAPH-UPLOAD shape=bs3/verify mib=12.0", line)
        self.assertIn("rc=0", line)

    def test_default_stream_is_the_current_one(self):
        lib = FakeLib()
        GU.upload_after_capture(FakeGraph(), "k", None, lib=lib, cuda=FakeCuda([5 * MIB, 5 * MIB]))
        self.assertEqual(lib.calls[0][1], 0x5150)

    def test_a_driver_error_is_a_warning_never_an_exception(self):
        lib = FakeLib(rc=2)  # CUDA_ERROR_OUT_OF_MEMORY
        with self.assertLogs(GU.logger, level="WARNING") as cm:
            got = GU.upload_after_capture(FakeGraph(), "k", FakeStream(), lib=lib, cuda=FakeCuda([5 * MIB, 5 * MIB]))
        self.assertEqual(got[0], 2)
        self.assertIn("FAILED rc=2", cm.output[-1])
        self.assertIn("uploads lazily at its first replay", cm.output[-1])

    def test_never_inside_a_capture(self):
        lib = FakeLib()
        self.assertIsNone(GU.upload_after_capture(FakeGraph(), "k", FakeStream(), lib=lib,
                                                  cuda=FakeCuda([1, 1], capturing=True)))
        self.assertEqual(lib.calls, [])

    def test_old_torch_without_exec_handle_is_skipped(self):
        class Old:
            pass

        lib = FakeLib()
        with self.assertLogs(GU.logger, level="INFO") as cm:
            self.assertIsNone(GU.upload_after_capture(Old(), "k", FakeStream(), lib=lib, cuda=FakeCuda([1, 1])))
        self.assertEqual(lib.calls, [])
        self.assertIn("no graph exec handle", cm.output[-1])

    def test_a_null_exec_is_skipped(self):
        lib = FakeLib()
        self.assertIsNone(GU.upload_after_capture(FakeGraph(0), "k", FakeStream(), lib=lib, cuda=FakeCuda([1, 1])))
        self.assertEqual(lib.calls, [])

    def test_anything_raising_is_swallowed(self):
        class Boom(FakeCuda):
            def mem_get_info(self):
                raise RuntimeError("driver gone")

        with self.assertLogs(GU.logger, level="WARNING") as cm:
            self.assertIsNone(GU.upload_after_capture(FakeGraph(), "k", FakeStream(), lib=FakeLib(), cuda=Boom([])))
        self.assertIn("skipped: RuntimeError", cm.output[-1])

    def test_missing_libcuda_is_a_line(self):
        GU.reset_for_tests()
        with mock.patch.object(GU.ctypes, "CDLL", side_effect=OSError("no driver")):
            with self.assertLogs(GU.logger, level="INFO") as cm:
                self.assertIsNone(GU.upload_after_capture(FakeGraph(), "k", FakeStream(), cuda=FakeCuda([1, 1])))
        self.assertIn("libcuda.so.1 not loadable", cm.output[-1])
        GU.reset_for_tests()


class TheSwitch(CustomTestCase):
    def test_code_default_is_off(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_WEG2_GRAPH_UPLOAD_AT_CAPTURE", None)
            self.assertFalse(GU.enabled())

    def test_env_turns_it_on(self):
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_GRAPH_UPLOAD_AT_CAPTURE": "1"}):
            self.assertTrue(GU.enabled())


class TheCaptureHookCallsTheUpload(CustomTestCase):
    """capture_one uploads the graph it just stored, with the capture stream,
    only when the switch is on -- driven with a fully mocked backend."""

    def _backend(self):
        from sglang.srt.model_executor.runner_backend import full_cuda_graph_backend as FB

        b = FB.FullCudaGraphBackend.__new__(FB.FullCudaGraphBackend)
        b._device_module = mock.MagicMock()
        b._tp_group = None
        b._skip_warmup_barrier = True
        b._memory_saver_adapter = None
        b._pool = None
        b._capture_stream = FakeStream()
        b._graphs, b._outputs = {}, {}
        return FB, b

    def _run(self, on):
        FB, b = self._backend()
        g = FakeGraph()
        with mock.patch.object(FB, "run_capture_warmups"), \
                mock.patch.object(FB.torch.cuda, "CUDAGraph", return_value=g), \
                mock.patch.object(FB.adaptive_graph_memory, "capture_graph_ctx", return_value=mock.MagicMock()), \
                mock.patch.object(FB.barlink_capture_census, "segment", return_value=mock.MagicMock()), \
                mock.patch.object(GU, "enabled", return_value=on), \
                mock.patch.object(GU, "upload_after_capture") as up:
            b.capture_one("bs3", lambda: "out")
        return b, g, up

    def test_on_uploads_the_stored_graph_with_the_capture_stream(self):
        b, g, up = self._run(True)
        self.assertIs(b._graphs["bs3"], g)
        up.assert_called_once()
        args = up.call_args[0]
        self.assertIs(args[0], g)
        self.assertEqual(args[1], "bs3")
        self.assertIsInstance(args[2], FakeStream)

    def test_off_does_nothing(self):
        _b, _g, up = self._run(False)
        up.assert_not_called()


class TheLauncherTurnsItOnForDOnly(CustomTestCase):
    def _env(self, group, extra=None):
        from sglang.srt.weg2 import launcher as L

        return L.build_env("/tmp", "/tmp/venv", "0,1,2", "/tmp/store", False, "t", group=group,
                           group_env_extra=extra)

    def test_d_gets_it_p_does_not(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_WEG2_GRAPH_UPLOAD_AT_CAPTURE", None)
            self.assertEqual(self._env("D").get("SGLANG_WEG2_GRAPH_UPLOAD_AT_CAPTURE"), "1")
            self.assertNotIn("SGLANG_WEG2_GRAPH_UPLOAD_AT_CAPTURE", self._env("P"))

    def test_an_explicit_value_wins(self):
        with mock.patch.dict(os.environ, {"SGLANG_WEG2_GRAPH_UPLOAD_AT_CAPTURE": "0"}):
            self.assertEqual(self._env("D").get("SGLANG_WEG2_GRAPH_UPLOAD_AT_CAPTURE"), "0")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_WEG2_GRAPH_UPLOAD_AT_CAPTURE", None)
            env = self._env("D", {"SGLANG_WEG2_GRAPH_UPLOAD_AT_CAPTURE": "0"})
            self.assertEqual(env.get("SGLANG_WEG2_GRAPH_UPLOAD_AT_CAPTURE"), "0")


if __name__ == "__main__":
    unittest.main()
