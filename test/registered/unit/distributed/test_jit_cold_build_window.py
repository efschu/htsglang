"""The cold-JIT-build window must lift the device-collective deadline for the
pre-capture warmup forwards -- and for nothing else.

Falsifier for the r3 finding: on a cold kernel cache the merge tip went RED in
6/6 boots and GREEN in 1/1 once the cache was warm. The stall was 23-30 s,
which is HTCCLDeviceTransport._TIMEOUT_CYCLES = 60e9 at the 5090's ~2.6 GHz.
Two 3080 ranks were inside a multi-minute nvcc build of gptq_marlin -- a
kernel that builds on FIRST CALL, and whose first call is the warmup forward
that precedes graph capture -- while rank 0 waited in a device collective.

What is pinned here:

 1. Outside the window, resolving a deadline is the IDENTITY function. That is
    the whole backward-compatibility claim: steady state and the default path
    are untouched.
 2. Inside the window, the deadline is multiplied.
 3. The RECORDED pass is outside the window, so the deadline baked into the
    captured graph -- the one that governs every replay afterwards -- is the
    unrelaxed one. A fix that simply raised the constant would fail this.
 4. The window is opened RANK-UNIFORMLY and unconditionally: no rank-local
    predicate in front of the barrier. (pynccl and CustomAllreduce were both
    hangs of exactly that shape.)
 5. The HTCCL device collectives actually route their deadline through the
    resolver, rather than reading the constant directly.
 6. A failure inside the window is re-raised WITH the cold-build hypothesis,
    not silently forwarded as some unrelated kernel's launch failure.

CPU only: this tests the deadline POLICY and the loop that carries it, not
any CUDA kernel.
"""

import inspect
import os
import unittest
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# THE REAL MODULE, imported -- never re-implemented here. A private copy in
# the test would keep these assertions green across a revert of the fix.
from sglang.srt.utils.jit_cold_build import (  # noqa: E402
    ColdBuildWindowError,
    cold_build_window,
    in_cold_build_window,
    resolve_timeout_cycles,
    run_capture_warmups,
)

_BASE = 60_000_000_000  # HTCCLDeviceTransport._TIMEOUT_CYCLES


class TestColdBuildWindow(CustomTestCase):
    def test_outside_the_window_the_deadline_is_unchanged(self):
        self.assertFalse(in_cold_build_window())
        self.assertEqual(resolve_timeout_cycles(_BASE), _BASE)
        self.assertEqual(resolve_timeout_cycles(1), 1)

    def test_inside_the_window_the_deadline_is_raised(self):
        with cold_build_window("unit test"):
            self.assertTrue(in_cold_build_window())
            relaxed = resolve_timeout_cycles(_BASE)
        self.assertGreater(
            relaxed,
            _BASE,
            "the cold-build window did not lift the collective deadline; a "
            "peer stuck in nvcc still traps the waiting rank",
        )
        # ...and it closes again.
        self.assertFalse(in_cold_build_window())
        self.assertEqual(resolve_timeout_cycles(_BASE), _BASE)

    def test_multiplier_is_read_from_the_environment_at_call_time(self):
        with mock.patch.dict(
            os.environ, {"SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT": "7"}, clear=False
        ):
            with cold_build_window("unit test"):
                self.assertEqual(resolve_timeout_cycles(_BASE), _BASE * 7)
        # An explicit 1 is the documented escape hatch back to the old
        # behaviour -- it must be honoured even inside the window.
        with mock.patch.dict(
            os.environ, {"SGLANG_JIT_COLD_BUILD_TIMEOUT_MULT": "1"}, clear=False
        ):
            with cold_build_window("unit test"):
                self.assertEqual(resolve_timeout_cycles(_BASE), _BASE)

    def test_window_is_reentrant_and_exception_safe(self):
        with cold_build_window("outer"):
            with cold_build_window("inner"):
                self.assertTrue(in_cold_build_window())
            # The inner exit must NOT close the outer window.
            self.assertTrue(in_cold_build_window())
        self.assertFalse(in_cold_build_window())

        try:
            with cold_build_window("raises"):
                raise ValueError("boom")
        except ValueError:
            pass
        self.assertFalse(
            in_cold_build_window(),
            "an exception left the cold-build window open; every later "
            "collective would run with the relaxed deadline",
        )


class TestWarmupLoopCarriesTheWindow(CustomTestCase):
    """The loop the graph backends run, driven with fakes."""

    def _fakes(self):
        calls = []

        class FakeDeviceModule:
            def synchronize(self_inner):
                calls.append("sync")

        class FakeGroup:
            def barrier(self_inner):
                calls.append("barrier")

        return calls, FakeDeviceModule(), FakeGroup()

    def test_warmups_run_inside_the_window_and_keep_the_original_order(self):
        calls, dev, group = self._fakes()
        seen = []

        def forward_fn():
            calls.append("forward")
            seen.append(resolve_timeout_cycles(_BASE))

        run_capture_warmups(forward_fn, device_module=dev, tp_group=group)

        self.assertEqual(calls, ["sync", "barrier", "forward"] * 2)
        self.assertTrue(
            seen and all(v > _BASE for v in seen),
            f"warmup forwards saw deadlines {seen}, expected the relaxed one",
        )

    def test_the_recorded_pass_is_outside_the_window(self):
        """The capture itself must see the UNRELAXED deadline.

        Otherwise the relaxed value is baked into the graph and every replay
        for the rest of the process inherits it -- turning a diverged peer
        from a 23 s trap into a 15 min wedge.
        """
        _, dev, group = self._fakes()
        run_capture_warmups(lambda: None, device_module=dev, tp_group=group)
        self.assertEqual(
            resolve_timeout_cycles(_BASE),
            _BASE,
            "the window was still open after the warmups; the recorded pass "
            "would bake a relaxed deadline into the captured graph",
        )

    def test_the_last_forwards_result_is_returned(self):
        """The breakable backend sizes its shared output buffer from it."""
        _, dev, group = self._fakes()
        seq = iter(["first", "last"])
        out = run_capture_warmups(
            lambda: next(seq), device_module=dev, tp_group=group
        )
        self.assertEqual(out, "last")

    def test_skip_barrier_and_hook_are_honoured(self):
        calls, dev, group = self._fakes()
        run_capture_warmups(
            lambda: calls.append("forward"),
            device_module=dev,
            tp_group=group,
            skip_barrier=True,
            post_warmup_hook=lambda: calls.append("hook"),
        )
        self.assertEqual(calls, ["sync", "forward", "hook"] * 2)

    def test_failure_inside_the_window_names_the_cold_build(self):
        _, dev, group = self._fakes()

        def boom():
            raise RuntimeError("CUDA error: unspecified launch failure")

        with self.assertRaises(ColdBuildWindowError) as ctx:
            run_capture_warmups(boom, device_module=dev, tp_group=group)
        msg = str(ctx.exception)
        self.assertIn("cold-build", msg.lower())
        self.assertIsInstance(ctx.exception.__cause__, RuntimeError)
        self.assertFalse(in_cold_build_window())


class TestCallSites(CustomTestCase):
    """The wiring, checked against the real sources.

    These are source assertions on purpose: the behaviour they protect needs
    three GPUs and a cold cache to observe, and the failure mode of getting
    them wrong is a hang, not a wrong number.
    """

    def test_htccl_device_collectives_resolve_their_deadline(self):
        from sglang.srt.distributed.device_communicators import htccl_device

        src = inspect.getsource(htccl_device)
        self.assertIn("resolve_timeout_cycles", src)
        # No launch site may still pass the raw constant: that is the one
        # that trapped rank 0 while its peers were in nvcc.
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("self._TIMEOUT_CYCLES,"):
                self.fail(
                    "an HTCCL device collective still launches with the raw "
                    f"_TIMEOUT_CYCLES: {stripped!r}"
                )

    def test_graph_backends_use_the_shared_warmup_loop(self):
        from sglang.srt.model_executor.runner_backend import (
            breakable_cuda_graph_backend,
            full_cuda_graph_backend,
            tc_piecewise_cuda_graph_backend,
        )

        for mod in (
            full_cuda_graph_backend,
            tc_piecewise_cuda_graph_backend,
            # PD prefill is graph-covered by default via the breakable
            # backend -- NOT eager -- so it pays the cold builds too.
            breakable_cuda_graph_backend,
        ):
            src = inspect.getsource(mod)
            self.assertIn(
                "run_capture_warmups",
                src,
                f"{mod.__name__} does not route its pre-capture warmups "
                "through the cold-build window",
            )

    def test_the_window_is_opened_rank_uniformly(self):
        """No rank-local predicate may gate the window.

        `run_capture_warmups` opens it unconditionally for every caller; the
        signature carries no rank, no device and no cache state, so there is
        nothing a single rank could decide differently.
        """
        params = set(inspect.signature(run_capture_warmups).parameters)
        for forbidden in ("rank", "tp_rank", "is_cold", "cache_hit"):
            self.assertNotIn(forbidden, params)
        src = inspect.getsource(run_capture_warmups)
        window_line = [l for l in src.splitlines() if "with cold_build_window" in l]
        self.assertEqual(len(window_line), 1)
        self.assertNotIn("if ", window_line[0])


if __name__ == "__main__":
    unittest.main()
