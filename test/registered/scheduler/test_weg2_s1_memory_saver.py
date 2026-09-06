"""Weg-2 slice S1 -- memory saver on the line.

Red-first tests for the six things S1 owns:

1. the CAMPAIGN (a) MUST_FIX release order (``flush_cache()`` BEFORE
   ``pause(kv_cache)``) -- boot-proven fatal on a hybrid GDN/mamba model,
   measured 2026-09-06 (CAMPAIGN_a_0906.md); the AST assertions below are that
   campaign's matched can-it-fail check, ported into the suite;
2. refusal W12 ``Weg2MemorySaverInactive`` on the no-op adapter, at the launch
   check the launcher calls AND at the first sleep;
3. ``empty_cache()`` in the sleep RPC (spec (S) 2.4 step 10) -- without it the
   freed pages sit in torch's reserve and NVML free does not move;
4. the sleep-acceptance census (spec (S) 2.4 step 11) -- an instrument on the
   flip path, with its denominator, never a registry;
5. the per-physical-GPU PCIe ``flock`` (spec (S) 2.7, lifted from #89
   hibernate) -- a different lock from the L2 ring's;
6. BOTH wake paths behind the per-group launcher flag
   ``--enable-weights-cpu-backup`` (record (S) 1d / (S) 1f B3).

Hermetic: no CUDA, no NVML required.  Run with ``CUDA_VISIBLE_DEVICES=""`` and
``PYTHONPATH=<worktree>/python`` -- without the latter these read FALSE-RED
against a different tree, so :func:`test_tree_under_test_is_this_worktree`
prints the resolved source file and is the first thing to read on a surprise.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import textwrap
import threading
import time
from typing import Any, List, Optional

import pytest

from sglang.srt.constants import (
    GPU_MEMORY_TYPE_CUDA_GRAPH,
    GPU_MEMORY_TYPE_KV_CACHE,
    GPU_MEMORY_TYPE_WEIGHTS,
)
from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

# ---------------------------------------------------------------------------
# AST helpers -- the campaign (a) matched check.  py_compile is structurally
# blind to "a device-touching call sits AFTER pause() in the same tag block",
# which is exactly the error class of the MUST_FIX.
# ---------------------------------------------------------------------------


def _func_ast(func_name: str) -> ast.FunctionDef:
    source_file = inspect.getsourcefile(wu)
    assert source_file is not None
    tree = ast.parse(open(source_file, "r", encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return node
    raise AssertionError(f"{func_name} not found in {source_file}")


def _tag_block(func: ast.FunctionDef, tag_name: str) -> List[ast.stmt]:
    """The body of ``if <tag_name> in tags:`` inside ``func``."""
    for node in func.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == tag_name
            and any(isinstance(op, ast.In) for op in test.ops)
        ):
            return node.body
    raise AssertionError(f"no `if {tag_name} in tags:` block in {func.name}")


def _call_index(body: List[ast.stmt], needle: str) -> int:
    """Index of the first top-level statement in ``body`` calling ``needle``."""
    for i, stmt in enumerate(body):
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call):
                if needle in ast.unparse(sub):
                    return i
    return -1


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAdapter:
    """Stands in for ``TorchMemorySaverAdapter``; records the tag order."""

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self.paused: List[str] = []
        self.resumed: List[str] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def pause(self, tag: str) -> None:
        self.paused.append(tag)

    def resume(self, tag: str) -> None:
        self.resumed.append(tag)


class FakeDeviceModule:
    def __init__(self):
        self.calls: List[str] = []

    def synchronize(self) -> None:
        self.calls.append("synchronize")

    def empty_cache(self) -> None:
        self.calls.append("empty_cache")


class FakeServerArgs:
    def __init__(self, *, enable_weights_cpu_backup: bool):
        self.enable_weights_cpu_backup = enable_weights_cpu_backup
        self.model_path = "/models/fake-27b"
        self.load_format = "auto"
        self.enable_memory_saver = True


class FakeScheduler:
    def __init__(self, server_args):
        self.server_args = server_args
        self.disaggregation_mode = None


def _make_manager(
    *,
    adapter: FakeAdapter,
    server_args: Optional[FakeServerArgs] = None,
    idle: bool = True,
) -> Any:
    scheduler = FakeScheduler(server_args) if server_args is not None else None
    return wu.SchedulerWeightUpdaterManager(
        tp_worker=None,
        draft_worker=None,
        tp_cpu_group=None,
        memory_saver_adapter=adapter,
        flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: idle,
        scheduler=scheduler,
    )


def _record_disk_reload(monkeypatch, outcome=None) -> List[Any]:
    """Record every ``update_weights_from_disk`` request.

    Patched on the CLASS: ``SchedulerWeightUpdaterManager`` is a
    ``slots=True`` dataclass, so an instance attribute cannot be planted.
    """
    calls: List[Any] = []

    def _stub(self, req):
        calls.append(req)
        return outcome if outcome is not None else _ok()

    monkeypatch.setattr(
        wu.SchedulerWeightUpdaterManager, "update_weights_from_disk", _stub
    )
    return calls


@pytest.fixture()
def fake_device(monkeypatch):
    dev = FakeDeviceModule()
    monkeypatch.setattr(wu.torch, "get_device_module", lambda *a, **k: dev)
    return dev


# ---------------------------------------------------------------------------
# 0. worktree sanity -- the PYTHONPATH trap
# ---------------------------------------------------------------------------


def test_tree_under_test_is_this_worktree():
    source_file = inspect.getsourcefile(wu)
    print(f"weight_updater under test: {source_file}")
    print(f"sys.path[0]: {sys.path[0]}")
    assert source_file is not None and os.path.exists(source_file)


# ---------------------------------------------------------------------------
# 1. MUST_FIX -- release order (campaign (a) can-it-fail check)
# ---------------------------------------------------------------------------


def test_kv_block_flushes_before_pause():
    """RED on the base tree: upstream pauses first and flushes second.

    ``flush_cache()`` -> ``HybridReqToTokenPool.clear()`` ->
    ``MambaPool.reset_state()`` zeroes tensors allocated under
    ``region(GPU_MEMORY_TYPE_KV_CACHE)``; after the pause those pages are
    unmapped and the zero_() is an illegal memory access.
    """
    body = _tag_block(
        _func_ast("release_memory_occupation"), "GPU_MEMORY_TYPE_KV_CACHE"
    )
    flush_at = _call_index(body, "self.flush_cache")
    pause_at = _call_index(body, "pause(GPU_MEMORY_TYPE_KV_CACHE)")
    print(
        f"kv_cache block statements: {len(body)} / "
        f"pause at [{pause_at}] / flush at [{flush_at}]"
    )
    assert flush_at >= 0, "flush_cache() not called in the kv_cache block"
    assert pause_at >= 0, "pause(kv_cache) not called in the kv_cache block"
    assert flush_at < pause_at, "flush_cache still runs AFTER pause(kv_cache)"


def test_kv_block_pause_is_the_last_statement():
    """Nothing device-touching may follow the pause inside the same block."""
    body = _tag_block(
        _func_ast("release_memory_occupation"), "GPU_MEMORY_TYPE_KV_CACHE"
    )
    pause_at = _call_index(body, "pause(GPU_MEMORY_TYPE_KV_CACHE)")
    assert pause_at == len(body) - 1, (
        f"pause(kv_cache) is statement [{pause_at}] of {len(body)}; "
        "every later statement in this block touches released pages"
    )


# ---------------------------------------------------------------------------
# 2. W12 -- Weg2MemorySaverInactive
# ---------------------------------------------------------------------------


def test_noop_adapter_pause_is_a_no_op():
    """GREEN PIN: the hazard W12 exists for.  Upstream behaviour, unchanged."""
    adapter = TorchMemorySaverAdapter.create(enable=False)
    assert adapter.enabled is False
    assert adapter.pause(GPU_MEMORY_TYPE_KV_CACHE) is None
    assert adapter.resume(GPU_MEMORY_TYPE_KV_CACHE) is None


def test_assert_memory_saver_active_refuses_the_noop_adapter():
    from sglang.srt.managers.weg2_memory_saver import (
        Weg2MemorySaverInactive,
        assert_memory_saver_active,
    )

    adapter = TorchMemorySaverAdapter.create(enable=False)
    with pytest.raises(Weg2MemorySaverInactive) as exc:
        assert_memory_saver_active(adapter, context="launcher")
    assert "launcher" in str(exc.value)
    assert "--enable-memory-saver" in str(exc.value)


def test_assert_memory_saver_active_accepts_a_live_adapter():
    from sglang.srt.managers.weg2_memory_saver import assert_memory_saver_active

    assert_memory_saver_active(FakeAdapter(enabled=True), context="first sleep")


def test_launch_arms_w12_when_the_memory_saver_was_requested():
    """The launch half of W12: ``--enable-memory-saver`` set, adapter dead.

    ``_TorchMemorySaverAdapterReal.enabled`` is ``_memory_saver.enabled``, so a
    present-but-disabled library reports False while the flag says True -- the
    silent case the launcher must refuse before the first request.
    """
    import ast as _ast
    import inspect as _inspect

    from sglang.srt.managers import scheduler as sched

    source_file = _inspect.getsourcefile(sched)
    assert source_file is not None
    src = open(source_file, "r", encoding="utf-8").read()
    tree = _ast.parse(src)
    # Look for the CALL, not the name: an `import assert_memory_saver_active`
    # inside the same block makes a name-substring check blind to a deleted
    # call (caught by mutant M1b during this slice's own mutation run).
    guarded = False
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.If):
            continue
        if "enable_memory_saver" not in _ast.unparse(node.test):
            continue
        for sub in _ast.walk(node):
            if isinstance(sub, _ast.Call) and _ast.unparse(sub.func).endswith(
                "assert_memory_saver_active"
            ):
                guarded = True
    assert guarded, (
        "scheduler init never CALLS assert_memory_saver_active under "
        "`if enable_memory_saver:`; a flag-on/adapter-dead boot would serve "
        "with every sleep a silent no-op"
    )


def test_first_sleep_refuses_when_the_adapter_is_inactive():
    from sglang.srt.managers.weg2_memory_saver import Weg2MemorySaverInactive

    manager = _make_manager(adapter=FakeAdapter(enabled=False))
    with pytest.raises(Weg2MemorySaverInactive):
        manager.release_memory_occupation(
            wu.ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
        )


def test_first_sleep_does_not_pause_anything_when_it_refuses():
    """A refusal that already mutated VRAM is not a refusal."""
    from sglang.srt.managers.weg2_memory_saver import Weg2MemorySaverInactive

    adapter = FakeAdapter(enabled=False)
    manager = _make_manager(adapter=adapter)
    with pytest.raises(Weg2MemorySaverInactive):
        manager.release_memory_occupation(
            wu.ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
        )
    assert adapter.paused == []


# ---------------------------------------------------------------------------
# 3. empty_cache() in the sleep RPC
# ---------------------------------------------------------------------------


def test_release_rpc_empties_the_allocator_cache(fake_device):
    adapter = FakeAdapter(enabled=True)
    manager = _make_manager(adapter=adapter)
    manager.release_memory_occupation(
        wu.ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
    )
    assert "empty_cache" in fake_device.calls, (
        "release_memory_occupation did not empty the allocator cache; freed "
        "pages stay in torch's reserve and NVML free does not move"
    )
    assert fake_device.calls.index("synchronize") < fake_device.calls.index(
        "empty_cache"
    ), "empty_cache() must follow the device synchronize"


# ---------------------------------------------------------------------------
# 4. sleep-acceptance census (spec (S) 2.4 step 11)
# ---------------------------------------------------------------------------


def test_census_reports_per_process_bytes_and_its_denominator():
    from sglang.srt.managers.weg2_memory_saver import sleep_acceptance_census

    census = sleep_acceptance_census(
        nvml_uuid="GPU-fake",
        _process_bytes={4242: 1294 * 1024 * 1024},
        _memory_info=(32_088 * 1024 * 1024, 30_730 * 1024 * 1024),
        _arena_census={},
        _pid=4242,
    )
    assert census.proc_used_bytes == 1294 * 1024 * 1024
    assert census.accepted is True
    assert census.denominator, "an instrument without a denominator is not one"
    line = census.format_line()
    assert "GPU-fake" in line and "denominator=" in line


def test_census_refuses_acceptance_when_handles_are_retained():
    """M3: ``retain_handles=True`` keeps the ADDRESS SPACE charged to us."""
    from sglang.srt.managers.weg2_memory_saver import sleep_acceptance_census

    census = sleep_acceptance_census(
        nvml_uuid="GPU-fake",
        _process_bytes={4242: 1294 * 1024 * 1024},
        _memory_info=(32_088 * 1024 * 1024, 30_730 * 1024 * 1024),
        _arena_census={
            0: {"reserved": 8 << 30, "backed": 0, "retained": 4 << 30, "arenas": 1}
        },
        _pid=4242,
    )
    assert census.retain_handles_asserted is False
    assert census.accepted is False
    assert census.arena_retained_bytes == 4 << 30
    assert "retain" in (census.refusal_reason or "").lower()


def test_census_refuses_acceptance_when_the_instrument_is_blind():
    """No NVML reading is not a passing reading."""
    from sglang.srt.managers.weg2_memory_saver import sleep_acceptance_census

    census = sleep_acceptance_census(
        nvml_uuid=None,
        _process_bytes=None,
        _memory_info=None,
        _arena_census={},
        _pid=4242,
    )
    assert census.accepted is False
    assert census.proc_used_bytes is None
    assert census.refusal_reason


def test_census_never_raises_on_a_broken_instrument():
    from sglang.srt.managers.weg2_memory_saver import sleep_acceptance_census

    def _boom(*a, **k):
        raise RuntimeError("nvml is on fire")

    census = sleep_acceptance_census(nvml_uuid="GPU-fake", _process_bytes=_boom)
    assert census.accepted is False
    assert census.refusal_reason


# ---------------------------------------------------------------------------
# 5. the per-physical-GPU PCIe flock (spec (S) 2.7)
# ---------------------------------------------------------------------------


def test_pcie_lock_serialises_the_same_card(tmp_path):
    from sglang.srt.managers.weg2_memory_saver import pcie_transfer_lock

    order: List[str] = []
    started = threading.Event()

    def _second():
        started.wait()
        with pcie_transfer_lock(
            nvml_uuid="GPU-a", lock_dir=str(tmp_path), timeout_s=5.0
        ):
            order.append("second-in")

    thread = threading.Thread(target=_second)
    with pcie_transfer_lock(nvml_uuid="GPU-a", lock_dir=str(tmp_path), timeout_s=5.0):
        thread.start()
        started.set()
        time.sleep(0.3)
        order.append("first-out")
    thread.join(timeout=10)
    assert order == ["first-out", "second-in"], order


def test_pcie_lock_does_not_serialise_distinct_cards(tmp_path):
    from sglang.srt.managers.weg2_memory_saver import pcie_transfer_lock

    with pcie_transfer_lock(nvml_uuid="GPU-a", lock_dir=str(tmp_path), timeout_s=5.0):
        with pcie_transfer_lock(
            nvml_uuid="GPU-b", lock_dir=str(tmp_path), timeout_s=1.0
        ):
            pass


def test_pcie_lock_wait_is_bounded(tmp_path):
    """Bounded waits: the deadline's expiry is a refusal, never a longer wait."""
    from sglang.srt.managers.weg2_memory_saver import (
        Weg2PcieLockTimeout,
        pcie_transfer_lock,
    )

    hold = threading.Event()
    released = threading.Event()

    def _holder():
        with pcie_transfer_lock(
            nvml_uuid="GPU-a", lock_dir=str(tmp_path), timeout_s=5.0
        ):
            hold.set()
            released.wait(timeout=10)

    thread = threading.Thread(target=_holder)
    thread.start()
    try:
        hold.wait(timeout=10)
        t0 = time.perf_counter()
        with pytest.raises(Weg2PcieLockTimeout):
            with pcie_transfer_lock(
                nvml_uuid="GPU-a", lock_dir=str(tmp_path), timeout_s=0.5
            ):
                pass
        assert time.perf_counter() - t0 < 5.0
    finally:
        released.set()
        thread.join(timeout=10)


def _with_items_around(body: List[ast.stmt], needle: str) -> List[str]:
    """``with`` context expressions whose body contains a call to ``needle``."""
    found: List[str] = []
    for stmt in body:
        if not isinstance(stmt, ast.With):
            continue
        for sub in ast.walk(stmt):
            if isinstance(sub, ast.Call) and needle in ast.unparse(sub):
                found.extend(ast.unparse(item.context_expr) for item in stmt.items)
                break
    return found


def test_sleep_d2h_is_serialised_on_the_card():
    body = _tag_block(_func_ast("release_memory_occupation"), "GPU_MEMORY_TYPE_WEIGHTS")
    guards = _with_items_around(body, "pause(GPU_MEMORY_TYPE_WEIGHTS)")
    assert any("_weg2_pcie_lock" in g for g in guards), (
        "the weights D2H is not under the per-physical-GPU PCIe lock; a "
        "co-located rank's wake-H2D would halve both legs"
    )


def test_wake_h2d_is_serialised_on_the_card():
    body = _tag_block(_func_ast("resume_memory_occupation"), "GPU_MEMORY_TYPE_WEIGHTS")
    guards = _with_items_around(body, "resume(GPU_MEMORY_TYPE_WEIGHTS)")
    assert any("_weg2_pcie_lock" in g for g in guards)


def test_no_collective_is_held_under_the_pcie_lock():
    """A lock held across ``torch.distributed.barrier`` is a deadlock shape."""
    for func_name in ("release_memory_occupation", "resume_memory_occupation"):
        body = _tag_block(_func_ast(func_name), "GPU_MEMORY_TYPE_WEIGHTS")
        for stmt in body:
            if not isinstance(stmt, ast.With):
                continue
            if not any(
                "_weg2_pcie_lock" in ast.unparse(item.context_expr)
                for item in stmt.items
            ):
                continue
            inner = ast.unparse(stmt)
            assert (
                "torch.distributed.barrier" not in inner
            ), f"{func_name} holds the PCIe lock across a collective"


def test_pcie_lock_path_is_not_the_l2_ring_path(tmp_path):
    """(S) 5: 'It is a different lock from F6's ring flock and must be named
    separately in the code.'"""
    from sglang.srt.managers.weg2_memory_saver import pcie_lock_path

    path = pcie_lock_path("GPU-a", lock_dir=str(tmp_path))
    assert "weg2-l2-" not in path, "collides with the F6 L2 ring backing file"
    assert "pcie" in os.path.basename(path)
    assert "GPU-a" in os.path.basename(path)


# ---------------------------------------------------------------------------
# 6. both wake paths (record (S) 1d / (S) 1f B3)
# ---------------------------------------------------------------------------


def test_wake_reloads_from_disk_when_cpu_backup_is_off(monkeypatch):
    manager = _make_manager(
        adapter=FakeAdapter(enabled=True),
        server_args=FakeServerArgs(enable_weights_cpu_backup=False),
    )
    calls = _record_disk_reload(monkeypatch)
    manager._weg2_wake_reload_weights()
    assert len(calls) == 1, "resume() recommits pages whose CONTENT IS UNDEFINED"
    req = calls[0]
    assert req.model_path == "/models/fake-27b"
    assert req.load_format == "auto"


def test_wake_reload_never_flushes_the_still_paused_kv_pool(monkeypatch):
    """Danger direction: at this point ``pause(kv_cache)`` is still in force."""
    manager = _make_manager(
        adapter=FakeAdapter(enabled=True),
        server_args=FakeServerArgs(enable_weights_cpu_backup=False),
    )
    calls = _record_disk_reload(monkeypatch)
    manager._weg2_wake_reload_weights()
    assert calls[0].flush_cache is False, (
        "flush_cache() during the wake would zero the mamba pool while its "
        "pages are still unmapped -- the CAMPAIGN (a) fault, mirrored"
    )
    assert calls[0].torch_empty_cache is False


def test_wake_does_not_reload_when_cpu_backup_is_on(monkeypatch):
    manager = _make_manager(
        adapter=FakeAdapter(enabled=True),
        server_args=FakeServerArgs(enable_weights_cpu_backup=True),
    )
    calls = _record_disk_reload(monkeypatch)
    manager._weg2_wake_reload_weights()
    assert calls == [], "the TMS cpu-backup buffer already carried the bytes"


def test_wake_refuses_when_the_reload_fails(monkeypatch):
    from sglang.srt.managers.weg2_memory_saver import Weg2WakeRefused

    manager = _make_manager(
        adapter=FakeAdapter(enabled=True),
        server_args=FakeServerArgs(enable_weights_cpu_backup=False),
    )
    _record_disk_reload(monkeypatch, outcome=_fail("no such checkpoint"))
    with pytest.raises(Weg2WakeRefused):
        manager._weg2_wake_reload_weights()


def test_wake_refuses_when_the_path_is_undecidable():
    from sglang.srt.managers.weg2_memory_saver import Weg2WakeRefused

    manager = _make_manager(adapter=FakeAdapter(enabled=True), server_args=None)
    with pytest.raises(Weg2WakeRefused):
        manager._weg2_wake_reload_weights()


def test_resume_rpc_calls_the_wake_reload_inside_the_weights_block():
    body = _tag_block(_func_ast("resume_memory_occupation"), "GPU_MEMORY_TYPE_WEIGHTS")
    assert (
        _call_index(body, "_weg2_wake_reload_weights") >= 0
    ), "the backup-OFF wake path is unreachable from the resume RPC"


def test_wake_reload_precedes_the_static_state_import():
    """The stash was exported from the live model; it is the last writer."""
    body = _tag_block(_func_ast("resume_memory_occupation"), "GPU_MEMORY_TYPE_WEIGHTS")
    reload_at = _call_index(body, "_weg2_wake_reload_weights")
    import_at = _call_index(body, "_import_static_state")
    assert reload_at >= 0 and import_at >= 0
    assert reload_at < import_at


# ---------------------------------------------------------------------------
# 7. GREEN PINS -- surviving upstream invariants
# ---------------------------------------------------------------------------


def test_release_exports_static_state_and_barriers_before_pausing_weights():
    body = _tag_block(_func_ast("release_memory_occupation"), "GPU_MEMORY_TYPE_WEIGHTS")
    export_at = _call_index(body, "_export_static_state")
    barrier_at = _call_index(body, "torch.distributed.barrier")
    pause_at = _call_index(body, "pause(GPU_MEMORY_TYPE_WEIGHTS)")
    assert 0 <= export_at < barrier_at < pause_at


def test_resume_order_is_graph_then_weights_then_kv():
    func = _func_ast("resume_memory_occupation")
    seen: List[str] = []
    for node in func.body:
        if not isinstance(node, ast.If):
            continue
        text = ast.unparse(node.test)
        for tag in (
            "GPU_MEMORY_TYPE_CUDA_GRAPH",
            "GPU_MEMORY_TYPE_WEIGHTS",
            "GPU_MEMORY_TYPE_KV_CACHE",
        ):
            if tag in text:
                seen.append(tag)
    assert seen == [
        "GPU_MEMORY_TYPE_CUDA_GRAPH",
        "GPU_MEMORY_TYPE_WEIGHTS",
        "GPU_MEMORY_TYPE_KV_CACHE",
    ], seen


def test_resume_barriers_between_weights_resume_and_static_import():
    body = _tag_block(_func_ast("resume_memory_occupation"), "GPU_MEMORY_TYPE_WEIGHTS")
    resume_at = _call_index(body, "resume(GPU_MEMORY_TYPE_WEIGHTS)")
    barrier_at = _call_index(body, "torch.distributed.barrier")
    import_at = _call_index(body, "_import_static_state")
    assert 0 <= resume_at < barrier_at < import_at


def test_offload_tags_are_still_tracked_both_ways():
    release_src = ast.unparse(_func_ast("release_memory_occupation"))
    resume_src = ast.unparse(_func_ast("resume_memory_occupation"))
    assert "self.offload_tags.add(tag)" in release_src
    assert "self.offload_tags.remove(tag)" in resume_src


def test_all_three_upstream_tags_are_still_handled_on_release():
    func = _func_ast("release_memory_occupation")
    src = ast.unparse(func)
    for tag in (
        "GPU_MEMORY_TYPE_KV_CACHE",
        "GPU_MEMORY_TYPE_WEIGHTS",
        "GPU_MEMORY_TYPE_CUDA_GRAPH",
    ):
        assert tag in src


# ---------------------------------------------------------------------------
# small helpers for the wake tests
# ---------------------------------------------------------------------------


class _Out:
    def __init__(self, success: bool, message: str = ""):
        self.success = success
        self.message = message


def _ok():
    return _Out(True)


def _fail(message: str):
    return _Out(False, message)


# keep the linters honest about the imports the AST tests reference by name
_ = (GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_WEIGHTS, textwrap)
