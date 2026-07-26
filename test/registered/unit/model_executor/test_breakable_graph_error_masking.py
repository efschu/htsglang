"""A failure inside a graph break must not be replaced by the state-machine
assertion (#164).

`BreakableCUDAGraphCapture` splits a capture at every `@eager_on_graph` break
point. The split is two steps: `_end_current_segment()` closes the segment that
ran up to the break, then the eager function runs, then `_begin_new_segment()`
opens the next one. Between those two steps there is NO open segment, so
`self._current_graph is None` is the normal, intended state of that window.

If anything raises inside that window -- the eager function itself, or the
weak-ref bookkeeping that follows it -- the exception unwinds out of the `with`
block and Python calls `__exit__`, which called `_end_current_segment()`
unconditionally and hit `assert graph is not None`. The AssertionError then
became the reported failure and the real cause was demoted to a chained
context. That is exactly the shape of the sm75 prefill report
(`breakable_cuda_graph.py:380 assert graph is not None`, raised from
`__exit__` at line 341): the assert is the messenger, not the defect.

These are pure CPU tests. Every CUDA object is a stub; no device is touched.
"""

import types
import unittest
from unittest import mock

import torch

from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    breakable_cuda_graph as bcg,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeGraph:
    """Stand-in for torch.cuda.CUDAGraph with the three calls used here."""

    def __init__(self, keep_graph: bool = False):
        self.keep_graph = keep_graph
        self.begun = False
        self.ended = False
        self.end_raises: BaseException | None = None

    def capture_begin(self, pool=None, capture_error_mode="global"):
        self.begun = True

    def capture_end(self):
        if self.end_raises is not None:
            raise self.end_raises
        self.ended = True

    def instantiate(self):
        pass

    def replay(self):
        pass


class _FakeStream:
    cuda_stream = 1


class _RecordingStreamCtx:
    """Stand-in for the object torch.cuda.stream() returns.

    Records the triple __exit__ is called with, which is the whole point: the
    capture context manager must forward its own exception triple here.
    """

    def __init__(self, stream):
        self.stream = stream
        self.entered = False
        self.exit_args: tuple | None = None

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.exit_args = (exc_type, exc_val, exc_tb)
        return False


class BreakableCaptureTestBase(unittest.TestCase):
    def setUp(self):
        self.graphs: list[_FakeGraph] = []

        def make_graph(*args, **kwargs):
            graph = _FakeGraph(*args, **kwargs)
            self.graphs.append(graph)
            return graph

        cuda_ns = types.SimpleNamespace(
            CUDAGraph=make_graph,
            current_stream=lambda device=None: _FakeStream(),
            Stream=torch.cuda.Stream,
            stream=torch.cuda.stream,
        )
        patcher = mock.patch.object(bcg.torch, "cuda", cuda_ns)
        patcher.start()
        self.addCleanup(patcher.stop)

    def assertCaptureStateClean(self):
        """No hook, no context leaked, whatever the outcome was."""
        self.assertEqual(bcg._hook_refcount, 0)
        self.assertIsNone(bcg._original_wait_stream)
        self.assertIsNone(bcg._current_capture_var.get())
        self.assertIsNone(bcg._current_stream_var.get())


class TestBreakFailureIsReported(BreakableCaptureTestBase):
    # ---- the defect this exists to prevent ------------------------------
    def test_eager_break_exception_survives_exit(self):
        """The eager function's own error is what the user must see."""
        boom = ModuleNotFoundError("No module named 'sgl_kernel'")

        @bcg.eager_on_graph(True)
        def eager_step():
            raise boom

        graph = bcg.BreakableCUDAGraph()
        with self.assertRaises(ModuleNotFoundError) as caught:
            with bcg.BreakableCUDAGraphCapture(cuda_graph=graph):
                eager_step()
        self.assertIs(caught.exception, boom)
        self.assertCaptureStateClean()

    def test_weak_ref_bookkeeping_exception_survives_exit(self):
        """The window also covers the weak-ref step AFTER the eager call --
        that is where an absent sgl_kernel actually raises."""
        boom = ModuleNotFoundError("No module named 'sgl_kernel'")

        @bcg.eager_on_graph(True)
        def eager_step(x):
            return x

        graph = bcg.BreakableCUDAGraph()
        with mock.patch.object(bcg, "_weak_ref_if_tensor", side_effect=boom):
            with self.assertRaises(ModuleNotFoundError) as caught:
                with bcg.BreakableCUDAGraphCapture(cuda_graph=graph):
                    eager_step(torch.zeros(2, device="cpu"))
        self.assertIs(caught.exception, boom)
        self.assertCaptureStateClean()

    def test_error_inside_a_segment_survives_exit(self):
        """Same requirement when the failure happens mid-segment, i.e. with a
        segment still open: closing it must not overwrite the diagnosis."""
        boom = RuntimeError("kernel launch failed")
        graph = bcg.BreakableCUDAGraph()
        with self.assertRaises(RuntimeError) as caught:
            with bcg.BreakableCUDAGraphCapture(cuda_graph=graph):
                raise boom
        self.assertIs(caught.exception, boom)
        self.assertCaptureStateClean()

    def test_capture_end_failure_does_not_replace_the_cause(self):
        """A capture that is already unwinding is usually invalid, so
        capture_end() may fail too. The original error still wins."""
        boom = RuntimeError("illegal memory access")
        graph = bcg.BreakableCUDAGraph()
        with self.assertRaises(RuntimeError) as caught:
            with bcg.BreakableCUDAGraphCapture(cuda_graph=graph):
                self.graphs[-1].end_raises = RuntimeError("capture_end blew up")
                raise boom
        self.assertIs(caught.exception, boom)
        self.assertCaptureStateClean()

    def test_nested_break_inside_a_break_is_inert(self):
        """A break point reached from inside another break's eager region has
        no segment to split -- it must simply run, not blow up the capture."""
        seen = []

        @bcg.eager_on_graph(True)
        def inner_step():
            seen.append("inner")

        @bcg.eager_on_graph(True)
        def outer_step():
            inner_step()
            seen.append("outer")

        graph = bcg.BreakableCUDAGraph()
        with bcg.BreakableCUDAGraphCapture(cuda_graph=graph):
            outer_step()
        self.assertEqual(seen, ["inner", "outer"])
        # one break -> two segments; the nested call must not add a third
        self.assertEqual(len(graph._segments), 2)
        self.assertEqual(len(graph._break_fns), 1)
        self.assertCaptureStateClean()


class TestCaptureOnADedicatedStream(BreakableCaptureTestBase):
    """Every real capture runs on a stream -- so __exit__ must survive one.

    `capture_session()` always hands the runner's capture stream to
    `BreakableCUDAGraphCapture(stream=...)`, so `self._stream_ctx` is NEVER
    None in production; the only captures that leave it None are the ones this
    file used to construct. That gap let the #164 signature change
    (`__exit__(self, *args)` -> `__exit__(self, exc_type=None, exc_val=None,
    exc_tb=None)`) ship with `self._stream_ctx.__exit__(*args)` still in the
    body: a NameError on the closing line of EVERY breakable capture, on every
    GPU, that no test could see. Constructing the manager with a stream is the
    entire falsifier.
    """

    def setUp(self):
        super().setUp()
        # A NameError raised INSIDE __exit__'s finally block aborts it before
        # _uninstall_wait_stream_hook(), so the module-global hook state leaks
        # into whatever runs next -- that leak is part of the bug's blast
        # radius, but it must not turn one red test into a red file. Restore
        # the globals after each test here so the red set stays exactly the
        # tests that assert on the defect.
        self.addCleanup(self._restore_hook_globals)

    def _restore_hook_globals(self):
        if bcg._original_wait_stream is not None:
            torch.cuda.Stream.wait_stream = bcg._original_wait_stream
        bcg._original_wait_stream = None
        bcg._hook_refcount = 0

    def _capture(self, cuda_graph):
        """A capture whose stream context is a recorder, as in production."""
        self.stream_ctxs: list[_RecordingStreamCtx] = getattr(self, "stream_ctxs", [])

        def make_stream_ctx(stream):
            ctx = _RecordingStreamCtx(stream)
            self.stream_ctxs.append(ctx)
            return ctx

        patcher = mock.patch.object(bcg.torch.cuda, "stream", make_stream_ctx)
        patcher.start()
        self.addCleanup(patcher.stop)
        return bcg.BreakableCUDAGraphCapture(
            cuda_graph=cuda_graph, stream=_FakeStream()
        )

    # ---- the defect this exists to prevent ------------------------------
    def test_successful_capture_on_a_stream_closes_the_stream_context(self):
        """The plain path: no exception, and the stream context is exited with
        the empty triple. Before the fix this raised NameError('args')."""
        graph = bcg.BreakableCUDAGraph()
        with self._capture(graph):
            pass
        self.assertEqual(len(graph._segments), 1)
        self.assertEqual(len(self.stream_ctxs), 1)
        self.assertTrue(self.stream_ctxs[0].entered)
        self.assertEqual(self.stream_ctxs[0].exit_args, (None, None, None))
        self.assertCaptureStateClean()

    def test_failing_capture_on_a_stream_still_reports_ITS_error(self):
        """The unwinding path with a stream: the body's exception must reach
        the caller, not a NameError raised while closing the stream context --
        and the stream context must see the real triple."""
        boom = RuntimeError("kernel launch failed")
        graph = bcg.BreakableCUDAGraph()
        with self.assertRaises(RuntimeError) as caught:
            with self._capture(graph):
                raise boom
        self.assertIs(caught.exception, boom)
        ctx = self.stream_ctxs[0]
        self.assertIs(ctx.exit_args[0], RuntimeError)
        self.assertIs(ctx.exit_args[1], boom)
        self.assertCaptureStateClean()

    def test_break_points_on_a_stream_capture_normally(self):
        """A stream capture with graph breaks -- the shape real prefill capture
        has -- produces the same segments as the streamless one."""

        @bcg.eager_on_graph(True)
        def eager_step():
            return torch.zeros(1, device="cpu")

        graph = bcg.BreakableCUDAGraph()
        with self._capture(graph):
            eager_step()
        self.assertEqual(len(graph._segments), 2)
        self.assertEqual(self.stream_ctxs[0].exit_args, (None, None, None))
        self.assertCaptureStateClean()


class TestNormalCaptureUnchanged(BreakableCaptureTestBase):
    """Regression criterion: the working path must not move."""

    def test_no_break_is_one_segment(self):
        graph = bcg.BreakableCUDAGraph()
        with bcg.BreakableCUDAGraphCapture(cuda_graph=graph):
            pass
        self.assertEqual(len(graph._segments), 1)
        self.assertEqual(len(graph._break_fns), 0)
        self.assertTrue(graph._segments[0].begun)
        self.assertTrue(graph._segments[0].ended)
        self.assertCaptureStateClean()

    def test_two_breaks_are_three_segments_and_replay_runs_them_in_order(self):
        order = []

        @bcg.eager_on_graph(True)
        def eager_step(tag):
            order.append(("capture", tag))
            return torch.zeros(1, device="cpu")

        graph = bcg.BreakableCUDAGraph()
        with bcg.BreakableCUDAGraphCapture(cuda_graph=graph):
            eager_step("a")
            eager_step("b")
        self.assertEqual(len(graph._segments), 3)
        self.assertEqual(len(graph._break_fns), 2)
        self.assertTrue(all(g.ended for g in graph._segments))

        order.clear()
        graph.replay()
        self.assertEqual(order, [("capture", "a"), ("capture", "b")])
        self.assertCaptureStateClean()

    def test_break_outside_a_capture_just_calls_through(self):
        calls = []

        @bcg.eager_on_graph(True)
        def eager_step():
            calls.append(1)
            return 7

        self.assertEqual(eager_step(), 7)
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    unittest.main()
