# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#356: PrefillCudaGraphRunner.capture() must enter model_capture_mode() for
the raw-CUDA-graph backends (Full, Breakable) and must NOT enter it for
tc_piecewise.

Before #356 the prefill runner never entered the context, so every capture-time
branch that reads get_is_capture_mode() / the disable_dispose_tensor flag saw a
frozen, wrong value under the Full backend (Breakable happened to be covered by
is_in_breakable_cuda_graph()). This is the capture-time-frozen-state bug family:
a value that is wrong at capture time and then replayed forever.

Two contracts are pinned here so a future refactor that flips one silently fails:
  1. WIRING -- capture() enters model_capture_mode() iff the backend records a
     raw CUDA graph. The observable is disable_dispose_tensor, which ONLY
     model_capture_mode() sets.
  2. READERS -- inside the real per-backend flag stack, each named reader
     (get_is_capture_mode, dispose_tensor suppression, is_in_tc_piecewise)
     sees its intended value.

CPU only -- these are properties of the capture-mode wiring, not of any kernel.
"""

from __future__ import annotations

import contextlib
import unittest
from unittest import mock

import torch

from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
    FullCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend import (
    TcPiecewiseCudaGraphBackend,
)
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context import (
    enable_breakable_cuda_graph,
)
from sglang.srt.model_executor.runner_backend_utils.tc_piecewise_cuda_graph import (
    enable_tc_piecewise_cuda_graph,
    is_in_tc_piecewise_cuda_graph,
)
from sglang.srt.model_executor.runner_utils.capture_mode import (
    get_is_capture_mode,
    model_capture_mode,
)
from sglang.srt.runtime_context import get_flags
from sglang.srt.utils.common import dispose_tensor

import sglang.srt.model_executor.runner.prefill_cuda_graph_runner as pcg_module

PrefillCudaGraphRunner = pcg_module.PrefillCudaGraphRunner


@contextlib.contextmanager
def _fake_graph_capture(*args, **kwargs):
    yield mock.Mock(stream=mock.Mock())


def _dispose_suppressed() -> bool:
    return get_flags().capture.disable_dispose_tensor


class TestPrefillCaptureModeWiring(unittest.TestCase):
    """capture() enters model_capture_mode() exactly for the raw-capture
    backends. Observed via disable_dispose_tensor, which ONLY that context sets,
    so the assertion cannot be satisfied by a backend session flag."""

    def _run_capture(self, backend_cls):
        runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
        runner.backend = mock.Mock(spec=backend_cls)
        runner.backend.capture_session.return_value = contextlib.nullcontext()
        runner.model_runner = mock.Mock()
        runner.model_runner.server_args.enable_cudagraph_gc = False
        runner.warmup = mock.Mock()

        observed = {}

        def record():
            observed["capture_mode"] = get_is_capture_mode()
            observed["dispose_suppressed"] = _dispose_suppressed()

        runner._capture_one_stream = record

        with (
            mock.patch.object(
                pcg_module, "freeze_gc", lambda *a, **k: contextlib.nullcontext()
            ),
            mock.patch.object(pcg_module, "graph_capture", _fake_graph_capture),
        ):
            runner.capture()

        # The context is exited cleanly: nothing leaks past capture().
        self.assertFalse(_dispose_suppressed())
        self.assertFalse(get_is_capture_mode())
        return observed

    def test_full_backend_enters_capture_mode(self):
        observed = self._run_capture(FullCudaGraphBackend)
        # Was the whole point of #356: Full set no flag, so both were False.
        self.assertTrue(observed["dispose_suppressed"])
        self.assertTrue(observed["capture_mode"])

    def test_breakable_backend_enters_capture_mode(self):
        observed = self._run_capture(BreakableCudaGraphBackend)
        # get_is_capture_mode() was already True via the breakable flag, but
        # disable_dispose_tensor was silently False before #356.
        self.assertTrue(observed["dispose_suppressed"])
        self.assertTrue(observed["capture_mode"])

    def test_tc_piecewise_backend_does_not_enter_capture_mode(self):
        observed = self._run_capture(TcPiecewiseCudaGraphBackend)
        # tc_piecewise owns dispose suppression via is_in_tc_piecewise_cuda_graph
        # and must not let get_is_capture_mode() flip its FX-traced branches.
        self.assertFalse(observed["dispose_suppressed"])
        self.assertFalse(observed["capture_mode"])

    def test_helper_classifies_backends(self):
        for cls, expected in (
            (FullCudaGraphBackend, True),
            (BreakableCudaGraphBackend, True),
            (TcPiecewiseCudaGraphBackend, False),
        ):
            runner = PrefillCudaGraphRunner.__new__(PrefillCudaGraphRunner)
            runner.backend = mock.Mock(spec=cls)
            self.assertEqual(
                runner._uses_raw_cuda_graph_capture(), expected, msg=cls.__name__
            )


class TestPrefillCaptureReaderContract(unittest.TestCase):
    """Inside the real per-backend flag stack a prefill capture produces, each
    named reader sees its intended value. Whole-family verdict per backend:

      Full        -> model_capture_mode() only
      Breakable   -> model_capture_mode() + enable_breakable_cuda_graph()
      tc_piecewise-> enable_tc_piecewise_cuda_graph() only (NO capture mode)
    """

    @contextlib.contextmanager
    def _full_capture_env(self):
        with model_capture_mode():
            yield

    @contextlib.contextmanager
    def _breakable_capture_env(self):
        with model_capture_mode():
            with enable_breakable_cuda_graph():
                yield

    @contextlib.contextmanager
    def _tc_piecewise_capture_env(self):
        with enable_tc_piecewise_cuda_graph():
            yield

    def test_get_is_capture_mode_per_backend(self):
        with self._full_capture_env():
            self.assertTrue(get_is_capture_mode())
        with self._breakable_capture_env():
            self.assertTrue(get_is_capture_mode())
        with self._tc_piecewise_capture_env():
            # Legitimately False: the FX trace must not take capture-time
            # multi-stream / nested-compile branches.
            self.assertFalse(get_is_capture_mode())

    def test_dispose_tensor_suppressed_under_every_prefill_backend(self):
        # Correctness pin: freeing a tensor mid-capture records data_ptr()==0
        # into the graph. dispose_tensor() must be a no-op under all three
        # prefill capture environments.
        for name, env in (
            ("full", self._full_capture_env),
            ("breakable", self._breakable_capture_env),
            ("tc_piecewise", self._tc_piecewise_capture_env),
        ):
            with self.subTest(backend=name):
                x = torch.empty(4)
                with env():
                    dispose_tensor(x)
                self.assertEqual(x.numel(), 4, msg=f"{name}: tensor was freed")

    def test_dispose_tensor_frees_outside_capture(self):
        # The suppression is capture-scoped: eager serving still frees.
        x = torch.empty(4)
        dispose_tensor(x)
        self.assertEqual(x.numel(), 0)

    def test_tc_piecewise_flag_only_under_tc_piecewise(self):
        with self._full_capture_env():
            self.assertFalse(is_in_tc_piecewise_cuda_graph())
        with self._breakable_capture_env():
            self.assertFalse(is_in_tc_piecewise_cuda_graph())
        with self._tc_piecewise_capture_env():
            self.assertTrue(is_in_tc_piecewise_cuda_graph())


if __name__ == "__main__":
    unittest.main()
