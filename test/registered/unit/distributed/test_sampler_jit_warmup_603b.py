# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#603b: the sampling JIT must be built at boot, not inside a serving forward.

FALSIFIER, stated plainly. The production defect was a 60-90 s nvcc build
reached lazily from ``Sampler.forward``, which left peers spinning on a
deadline-bearing collective until they aborted with "a peer did not arrive".
The fix is PLACEMENT: build at boot, on every rank, inside the cold-build
window, then barrier. So the tests here pin the three things that make the
placement work, not merely that a function exists:

  * the public sampling entry points are actually reached (a warmup that
    builds nothing warms nothing),
  * the build happens INSIDE the cold-build window,
  * the barrier runs AFTER it, and runs even when the build FAILS -- because a
    barrier some ranks skip is itself a desync, and the error path is exactly
    where that is easy to get wrong.

Plus the wiring test: the module is inert unless ``init_model_worker`` calls
it, so a revert of the call site must fail here rather than pass quietly.

Hermetic: no CUDA, no real process group, no flashinfer. ``flashinfer.sampling``
is stubbed into ``sys.modules``, so this collects and runs on any host.
"""

import sys
import types

import pytest

from sglang.srt.layers.sampler_warmup import (
    sampling_backend_needs_jit_warmup,
    warm_sampling_backend_kernels,
)
from sglang.srt.utils import jit_cold_build


class _RecordingGroup:
    """Stands in for the TP GroupCoordinator; records barrier calls."""

    def __init__(self, log):
        self._log = log

    def barrier(self):
        self._log.append("barrier")


def _install_flashinfer_stub(log, *, fail=False):
    """Stub ``flashinfer.sampling`` and record calls + window state.

    Recording ``in_cold_build_window()`` AT CALL TIME is the point: it is the
    only way to assert the build is inside the window rather than merely
    preceded by one that already closed.
    """

    def _top_k_top_p(probs, top_ks, top_ps, filter_apply_order=None):
        log.append(("top_k_top_p", jit_cold_build.in_cold_build_window()))
        if fail:
            raise RuntimeError("ninja: build stopped")
        return probs

    def _min_p(probs, min_ps):
        log.append(("min_p", jit_cold_build.in_cold_build_window()))
        return probs

    mod = types.ModuleType("flashinfer.sampling")
    mod.top_k_top_p_sampling_from_probs = _top_k_top_p
    mod.min_p_sampling_from_probs = _min_p
    pkg = types.ModuleType("flashinfer")
    pkg.sampling = mod
    return {"flashinfer": pkg, "flashinfer.sampling": mod}


@pytest.fixture
def stub_flashinfer(monkeypatch):
    def _install(log, *, fail=False):
        for name, mod in _install_flashinfer_stub(log, fail=fail).items():
            monkeypatch.setitem(sys.modules, name, mod)

    return _install


def test_warmup_reaches_both_public_sampling_entrypoints(stub_flashinfer):
    """A warmup that builds nothing warms nothing."""
    log = []
    stub_flashinfer(log)
    status = warm_sampling_backend_kernels("flashinfer", device="cpu")
    assert status == "ok"
    called = [name for name, _ in log if isinstance(name, str) and name != "barrier"]
    assert called == ["top_k_top_p", "min_p"]


def test_build_runs_inside_the_cold_build_window(stub_flashinfer):
    """The placement half of the fix.

    Falsifiable: the recorded flag is read inside the stubbed kernel call, so
    if the build were moved outside the window this reads False.
    """
    log = []
    stub_flashinfer(log)
    warm_sampling_backend_kernels("flashinfer", device="cpu")
    windows = [inside for name, inside in log if name in ("top_k_top_p", "min_p")]
    assert windows == [True, True]
    # And it must not leak: a window left open would relax every later deadline.
    assert not jit_cold_build.in_cold_build_window()


def test_barrier_runs_after_the_build(stub_flashinfer):
    """Order matters: barrier before the build would rendezvous too early."""
    log = []
    stub_flashinfer(log)
    group = _RecordingGroup(log)
    warm_sampling_backend_kernels("flashinfer", device="cpu", tp_group=group)
    assert log[-1] == "barrier"
    assert sum(1 for entry in log if entry == "barrier") == 1
    # The build must be recorded before it, not merely somewhere in the log.
    assert log.index("barrier") > max(
        i for i, e in enumerate(log) if isinstance(e, tuple)
    )


def test_barrier_still_runs_when_the_build_fails(stub_flashinfer):
    """A barrier that some ranks skip IS a desync.

    The build failing on ONE rank must not remove that rank from the
    rendezvous -- that would convert a build error into the very hang this
    change exists to remove.
    """
    log = []
    stub_flashinfer(log, fail=True)
    group = _RecordingGroup(log)
    status = warm_sampling_backend_kernels("flashinfer", device="cpu", tp_group=group)
    assert status.startswith("failed: RuntimeError")
    assert log[-1] == "barrier"
    assert not jit_cold_build.in_cold_build_window()


def test_non_flashinfer_backend_takes_no_collective_and_no_build(stub_flashinfer):
    """Default-path guard: byte-identical behaviour when the backend is torch."""
    log = []
    stub_flashinfer(log)
    group = _RecordingGroup(log)
    status = warm_sampling_backend_kernels("pytorch", device="cpu", tp_group=group)
    assert "skipped" in status
    assert log == []


def test_warmup_gate_keys_only_on_the_replicated_arg():
    """The gate decides who enters a COLLECTIVE, so it must be rank-uniform.

    Pinning this explicitly because the tempting refinement -- "only warm when
    this card actually has the kernels" -- is a per-rank capability probe, and
    would put some ranks in the barrier and leave others out.
    """
    assert sampling_backend_needs_jit_warmup("flashinfer") is True
    assert sampling_backend_needs_jit_warmup("pytorch") is False
    assert sampling_backend_needs_jit_warmup("ascend") is False


def test_scheduler_boot_actually_calls_the_warmup():
    """Wiring. The module is inert unless the boot path calls it.

    Asserted against the REAL Scheduler source, so deleting the call site fails
    here instead of leaving a green suite around a reverted fix.
    """
    import inspect

    from sglang.srt.managers.scheduler import Scheduler

    src = inspect.getsource(Scheduler.init_model_worker)
    assert "warm_sampling_backend()" in src, (
        "Scheduler.init_model_worker no longer calls warm_sampling_backend(); "
        "the #603b lazy-JIT wedge is reachable again."
    )
    # And it must come after graph capture: capture itself is a long cold-build
    # stretch, and warming before it would put the barrier in the wrong place.
    assert src.index("init_all_cuda_graphs()") < src.index("warm_sampling_backend()")
    assert hasattr(Scheduler, "warm_sampling_backend")
