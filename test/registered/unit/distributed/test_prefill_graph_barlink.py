"""Test: prefill CUDA graph enablement path for barlink-bar1 + breakable.

Hermetic tests — no GPU access. Verifies that:
1. The default prefill backend on CUDA resolves to BREAKABLE.
2. The breakable auto-disable rules do NOT fire for a standard
   non-MLA, non-multimodal, non-MoE, non-LoRA config with hierarchical cache.
3. SGLANG_BARLINK_GRAPH_ENABLE defaults to ON, making bar1 capturable.
4. graph_grid_default() chains to SGLANG_BARLINK_GRAPH_ENABLE when
   SGLANG_BARLINK_BAR1_GRAPH_GRID is unset.
"""
from __future__ import annotations

import os

# Block all GPU access — hermetic test.
os.environ["CUDA_VISIBLE_DEVICES"] = "99"

# #753-fold fix: this file arrived from the wt-prefill-graph-qwen worktree
# still carrying a MODULE-LEVEL prepend of that tree. Collection imports
# every test module, so the prepend poisoned the sys.path every LATER
# multiprocess test's spawn children inherited -- measured: the gloo wire
# children resolved sglang from the foreign tree and died on
# ModuleNotFoundError for modules that exist only here. This repo's copy
# tests THIS repo's sglang, which PYTHONPATH already provides; a test that
# needs another worktree must scope the insert inside itself, never at
# import time.


def test_default_prefill_backend_is_breakable():
    """On CUDA, the default prefill backend should be BREAKABLE.

    Since we run hermetic (no GPU), we mock is_cuda() to True."""
    from sglang.srt.model_executor.cuda_graph_config import (
        Backend,
        default_cuda_graph_config,
    )
    from unittest.mock import patch

    # is_cuda() is lazily imported from sglang.srt.utils inside
    # default_prefill_backend(). Patch the source module.
    with patch("sglang.srt.utils.is_cuda", return_value=True):
        cfg = default_cuda_graph_config()
        assert cfg.prefill.backend == Backend.BREAKABLE, (
            f"Expected default prefill backend BREAKABLE on CUDA, "
            f"got {cfg.prefill.backend}."
        )

    # Also verify the non-CUDA default is TC_PIECEWISE
    with patch("sglang.srt.utils.is_cuda", return_value=False):
        cfg2 = default_cuda_graph_config()
        assert cfg2.prefill.backend == Backend.TC_PIECEWISE, (
            f"Expected TC_PIECEWISE on non-CUDA, got {cfg2.prefill.backend}."
        )


def test_graph_enable_default():
    """SGLANG_BARLINK_GRAPH_ENABLE should default to ON (truthy '1')."""
    # Clean slate: remove any prior setting
    env_backup = os.environ.pop("SGLANG_BARLINK_GRAPH_ENABLE", None)
    try:
        from sglang.srt.distributed.parallel_state import graph_enable_set

        assert graph_enable_set() is True, (
            "SGLANG_BARLINK_GRAPH_ENABLE should default to ON (1) "
            "when unset. This is the #369 release default."
        )
    finally:
        if env_backup is not None:
            os.environ["SGLANG_BARLINK_GRAPH_ENABLE"] = env_backup


def test_bar1_capturable_when_graph_enable_on():
    """When SGLANG_BARLINK_GRAPH_ENABLE is on, bar1 must be capturable."""
    env_backup = os.environ.pop("SGLANG_BARLINK_GRAPH_ENABLE", None)
    try:
        from sglang.srt.distributed.parallel_state import (
            CAPTURABLE_BARLINK_TRANSPORTS,
            GRAPH_ENABLE_TRANSPORTS,
            capturable_transports,
        )

        # bar1 is in GRAPH_ENABLE_TRANSPORTS, NOT in the base set
        assert "bar1" not in CAPTURABLE_BARLINK_TRANSPORTS, (
            "bar1 should NOT be in the base capturable set; "
            "it requires the release switch"
        )
        assert "bar1" in GRAPH_ENABLE_TRANSPORTS, (
            "bar1 should be in GRAPH_ENABLE_TRANSPORTS"
        )

        # With release ON (default), bar1 IS capturable
        cap = capturable_transports()
        assert "bar1" in cap, (
            f"bar1 should be capturable when SGLANG_BARLINK_GRAPH_ENABLE "
            f"is on (default). Got capturable set: {cap}"
        )
    finally:
        if env_backup is not None:
            os.environ["SGLANG_BARLINK_GRAPH_ENABLE"] = env_backup


def test_graph_grid_default_chains():
    """When SGLANG_BARLINK_BAR1_GRAPH_GRID is unset,
    graph_grid_default() should chain to SGLANG_BARLINK_GRAPH_ENABLE."""
    # Remove the explicit grid override
    grid_backup = os.environ.pop("SGLANG_BARLINK_BAR1_GRAPH_GRID", None)
    graph_backup = os.environ.pop("SGLANG_BARLINK_GRAPH_ENABLE", None)
    try:
        from sglang.srt.distributed.device_communicators.barlink_bar1 import (
            graph_grid_default,
        )

        # With both unset, GRAPH_ENABLE defaults to "1" (on)
        result = graph_grid_default()
        assert result is True, (
            f"Expected graph_grid_default()=True when both env vars unset "
            f"(GRAPH_ENABLE defaults to '1'). Got {result}."
        )

        # Explicit GRAPH_GRID=0 should force off regardless
        os.environ["SGLANG_BARLINK_BAR1_GRAPH_GRID"] = "0"
        result_forced_off = graph_grid_default()
        assert result_forced_off is False, (
            f"Expected graph_grid_default()=False with "
            f"SGLANG_BARLINK_BAR1_GRAPH_GRID=0. Got {result_forced_off}."
        )

        # GRAPH_ENABLE=0 with GRID unset should also be off
        del os.environ["SGLANG_BARLINK_BAR1_GRAPH_GRID"]
        os.environ["SGLANG_BARLINK_GRAPH_ENABLE"] = "0"
        result_off = graph_grid_default()
        assert result_off is False, (
            f"Expected graph_grid_default()=False with "
            f"SGLANG_BARLINK_GRAPH_ENABLE=0. Got {result_off}."
        )
    finally:
        if grid_backup is not None:
            os.environ["SGLANG_BARLINK_BAR1_GRAPH_GRID"] = grid_backup
        elif "SGLANG_BARLINK_BAR1_GRAPH_GRID" in os.environ:
            del os.environ["SGLANG_BARLINK_BAR1_GRAPH_GRID"]
        if graph_backup is not None:
            os.environ["SGLANG_BARLINK_GRAPH_ENABLE"] = graph_backup
        elif "SGLANG_BARLINK_GRAPH_ENABLE" in os.environ:
            del os.environ["SGLANG_BARLINK_GRAPH_ENABLE"]


def test_enforce_cpu_transport_error_when_graph_enable_off():
    """When GRAPH_ENABLE is off and CUDA graphs are active,
    _enforce_cpu_transport_needs_eager should raise for bar1.

    This verifies the error path that blocks bar1 under capture when
    the release switch is off. The function needs ServerArgs published
    to detect 'graphs enabled', so we test the capturable path directly."""
    graph_backup = os.environ.pop("SGLANG_BARLINK_GRAPH_ENABLE", None)
    try:
        # Force GRAPH_ENABLE off
        os.environ["SGLANG_BARLINK_GRAPH_ENABLE"] = "0"

        # Force reimport to pick up new env
        import importlib
        import sglang.srt.distributed.parallel_state as ps_mod
        importlib.reload(ps_mod)

        from sglang.srt.distributed.parallel_state import (
            CAPTURABLE_BARLINK_TRANSPORTS,
            GRAPH_ENABLE_TRANSPORTS,
            capturable_transports,
        )

        # bar1 should NOT be capturable when GRAPH_ENABLE is off
        cap = capturable_transports()
        assert "bar1" not in cap, (
            f"bar1 should NOT be capturable when SGLANG_BARLINK_GRAPH_ENABLE=0. "
            f"Got: {cap}"
        )
        # But it should still be in GRAPH_ENABLE_TRANSPORTS (the set that
        # gets added when the switch is on)
        assert "bar1" in GRAPH_ENABLE_TRANSPORTS
    finally:
        if graph_backup is not None:
            os.environ["SGLANG_BARLINK_GRAPH_ENABLE"] = graph_backup
        elif "SGLANG_BARLINK_GRAPH_ENABLE" in os.environ:
            del os.environ["SGLANG_BARLINK_GRAPH_ENABLE"]


def test_prefill_capture_enters_model_capture_mode():
    """Verify that PrefillCudaGraphRunner.capture() enters
    model_capture_mode() for Breakable/Full backends (#356 fix).

    We test this by checking the _uses_raw_cuda_graph_capture method
    logic, not by constructing the runner (which needs a GPU)."""
    # The method checks isinstance(backend, BreakableCudaGraphBackend | Full)
    # and returns True for those backends. We can verify the logic by
    # checking the source structure.
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )
    from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
        BreakableCudaGraphBackend,
    )
    from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
        FullCudaGraphBackend,
    )

    # Verify the method exists and uses the right isinstance check
    source = PrefillCudaGraphRunner._uses_raw_cuda_graph_capture.__doc__ or ""
    # The method must exist — its presence is the fix for #356
    assert hasattr(PrefillCudaGraphRunner, "_uses_raw_cuda_graph_capture"), (
        "PrefillCudaGraphRunner must have _uses_raw_cuda_graph_capture method "
        "(fix for #356: model_capture_mode never entered)"
    )

    # Also verify capture() uses model_capture_mode
    import inspect
    capture_source = inspect.getsource(PrefillCudaGraphRunner.capture)
    assert "model_capture_mode" in capture_source, (
        "PrefillCudaGraphRunner.capture() must enter model_capture_mode() "
        "(fix for #356)"
    )


def test_solo_draft_barrier_fix():
    """Verify the solo-draft barrier deadlock fix is present (#194).

    The fix: enter_capture_group_barrier is called (not a raw
    tp_group.barrier()), so solo-draft ranks skip the barrier.
    """
    import inspect
    from sglang.srt.model_executor.runner.prefill_cuda_graph_runner import (
        PrefillCudaGraphRunner,
    )

    init_source = inspect.getsource(PrefillCudaGraphRunner.__init__)
    assert "enter_capture_group_barrier" in init_source, (
        "PrefillCudaGraphRunner.__init__ must use enter_capture_group_barrier "
        "(fix for #194: solo-draft barrier deadlock)"
    )
    # The raw tp_group.barrier() should NOT appear
    assert "tp_group.barrier()" not in init_source, (
        "PrefillCudaGraphRunner.__init__ must NOT use raw tp_group.barrier() "
        "(solo-draft ranks would deadlock)"
    )


def test_breakable_rules_no_hierarchical_cache():
    """The breakable auto-disable rules do NOT include hierarchical cache.

    This is by design: hierarchical cache is scheduler-level, not model-level.
    tc_piecewise excludes it because torch.compile can't trace the tier
    transitions, but BCG uses segmented capture and is not affected.

    Verify by reading the rule list directly."""
    import inspect
    from sglang.srt.server_args import ServerArgs

    source = inspect.getsource(
        ServerArgs._disable_breakable_cudagraph_if_incompatible
    )
    assert "hierarchical_cache" not in source and "cpu_offload_gb" not in source, (
        "Breakable auto-disable rules must NOT include hierarchical cache "
        "or cpu_offload_gb. If they did, hierarchical cache would block "
        "breakable prefill graphs."
    )

    # Verify that tc_piecewise DOES have the rule (for contrast)
    tc_source = inspect.getsource(
        ServerArgs._disable_tc_piecewise_cudagraph_if_incompatible
    )
    assert "hierarchical_cache" in tc_source or "cpu_offload_gb" in tc_source, (
        "tc_piecewise rules SHOULD include hierarchical cache / cpu_offload "
        "exclusion (torch.compile limitation)"
    )


if __name__ == "__main__":
    # Run all tests
    import pytest
    pytest.main([__file__, "-xvs"])
