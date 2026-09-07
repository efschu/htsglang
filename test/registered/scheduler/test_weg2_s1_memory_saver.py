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
from contextlib import contextmanager
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
    def __init__(
        self,
        *,
        enable_weights_cpu_backup: bool,
        enable_memory_saver: bool = True,
        enable_draft_weights_cpu_backup: bool = False,
        speculative_draft_model_path: Optional[str] = None,
    ):
        self.enable_weights_cpu_backup = enable_weights_cpu_backup
        # model_runner.py:2342-2344 builds the WEIGHTS region with
        # `enable_weights_cpu_backup or (is_draft_worker and
        # enable_draft_weights_cpu_backup)`, so the draft shard's backup verdict
        # is a DIFFERENT expression from the main shard's.
        self.enable_draft_weights_cpu_backup = enable_draft_weights_cpu_backup
        self.speculative_draft_model_path = speculative_draft_model_path
        self.model_path = "/models/fake-27b"
        self.load_format = "auto"
        self.enable_memory_saver = enable_memory_saver


class FakeScheduler:
    def __init__(self, server_args):
        self.server_args = server_args
        self.disaggregation_mode = None


class FakeTpWorker:
    """Just enough of ``self.tp_worker.model_runner.model`` for the sleep RPC."""

    class _Runner:
        model = object()

    model_runner = _Runner()


def _make_manager(
    *,
    adapter: FakeAdapter,
    server_args: Optional[FakeServerArgs] = None,
    idle: bool = True,
    draft_worker: Any = None,
    tp_worker: Any = None,
) -> Any:
    scheduler = FakeScheduler(server_args) if server_args is not None else None
    return wu.SchedulerWeightUpdaterManager(
        tp_worker=tp_worker,
        draft_worker=draft_worker,
        tp_cpu_group=None,
        memory_saver_adapter=adapter,
        flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: idle,
        scheduler=scheduler,
    )


@pytest.fixture()
def noop_pcie_lock(monkeypatch):
    """Hermetic stand-in for the per-card ``flock``.

    ``_weg2_wake_reload_weights`` now takes the PCIe lock around the refill, so
    a test that calls it directly would otherwise reach NVML and ``/dev/shm``.
    The lock's own two behaviours are pinned separately by
    ``test_pcie_lock_wrapper_propagates_the_named_refusal`` and
    ``test_pcie_lock_wrapper_degrades_when_the_card_key_is_unresolvable``.
    """
    from contextlib import contextmanager as _cm

    monkeypatch.setattr(wu, "resolve_pcie_lock_key", lambda: "GPU-hermetic")

    @_cm
    def _noop(**kwargs):
        yield kwargs.get("label", "transfer")

    monkeypatch.setattr(wu, "pcie_transfer_lock", _noop)


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
    # ZERO live arenas is the PERMANENT state under Weg 2 (--enable-vram-dial
    # is refused by W13 and the phase flip is deleted by S0/S7), so the arena
    # half must read n/a here.  Printing `arena_backed=0.0 MiB
    # retain_handles_asserted=True` would be a fabricated measurement, and the
    # boot postmortem would quote a pass the instrument never took.
    assert census.retain_handles_asserted is None
    assert "arena_backed=n/a" in line
    assert "retain_handles=n/a" in line
    assert "arena_rows=0" in census.denominator


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
    """BOTH wake legs, because only one of them transfers in each arm.

    With ``--enable-weights-cpu-backup`` the H2D is inside
    ``resume(GPU_MEMORY_TYPE_WEIGHTS)``.  With it OFF -- the V1 arm of record
    (record 1b round-2 Q2, option (ii)) -- that resume is a pure VMM recommit
    that moves no bytes, and the whole 12-17 s refill is the
    ``update_weights_from_disk`` leg inside ``_weg2_wake_reload_weights``.
    A gate that pins only the first arm leaves the configured arm unserialised.
    """
    body = _tag_block(_func_ast("resume_memory_occupation"), "GPU_MEMORY_TYPE_WEIGHTS")
    guards = _with_items_around(body, "resume(GPU_MEMORY_TYPE_WEIGHTS)")
    assert any(
        "_weg2_pcie_lock" in g for g in guards
    ), "the backup-ON wake H2D is not under the per-physical-GPU PCIe lock"

    reload_body = _func_ast("_weg2_wake_reload_weights").body
    reload_guards = _with_items_around(reload_body, "update_weights_from_disk")
    assert any("_weg2_pcie_lock" in g for g in reload_guards), (
        "the backup-OFF wake refill (update_weights_from_disk, the only leg "
        "that moves bytes in the V1 arm) runs OUTSIDE the PCIe lock; a "
        "co-located sibling's sleep-D2H halves both"
    )


def test_wake_reload_holds_no_collective_under_the_pcie_lock():
    """The second lock take must keep the invariant the first one keeps."""
    func = _func_ast("_weg2_wake_reload_weights")
    for stmt in ast.walk(func):
        if not isinstance(stmt, ast.With):
            continue
        if not any(
            "_weg2_pcie_lock" in ast.unparse(item.context_expr) for item in stmt.items
        ):
            continue
        assert "torch.distributed.barrier" not in ast.unparse(
            stmt
        ), "_weg2_wake_reload_weights holds the PCIe lock across a collective"


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


def test_wake_reloads_from_disk_when_cpu_backup_is_off(monkeypatch, noop_pcie_lock):
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


def test_wake_reload_never_flushes_the_still_paused_kv_pool(
    monkeypatch, noop_pcie_lock
):
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


def test_wake_refuses_when_the_reload_fails(monkeypatch, noop_pcie_lock):
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
# 8. FIX ROUND 1 -- the stock boot, the wired census, and the wrapper itself
# ---------------------------------------------------------------------------


@pytest.fixture()
def stock_boot_stubs(monkeypatch):
    """Everything the weights sleep touches that a hermetic test has no copy of.

    Deliberately does NOT stub ``_hibernate_park_weights``: each test that
    needs it records it itself, so a test cannot pass on a stub that swallowed
    the very call it claims to prove.
    """
    monkeypatch.setattr(wu, "_export_static_state", lambda model: {"stub": True})
    monkeypatch.setattr(wu.torch.distributed, "barrier", lambda *a, **k: None)
    monkeypatch.setattr(
        wu.SchedulerWeightUpdaterManager,
        "_weg2_log_sleep_acceptance",
        lambda self: None,
    )
    monkeypatch.setattr(wu, "resolve_pcie_lock_key", lambda: "GPU-hermetic")

    @contextmanager
    def _noop_lock(**kwargs):
        yield kwargs.get("label", "transfer")

    monkeypatch.setattr(wu, "pcie_transfer_lock", _noop_lock)


def _record_hibernate_park(monkeypatch) -> List[Any]:
    calls: List[Any] = []
    monkeypatch.setattr(
        wu.SchedulerWeightUpdaterManager,
        "_hibernate_park_weights",
        lambda self, req: calls.append(req),
    )
    return calls


def test_stock_boot_sleep_is_not_refused_and_still_parks_to_disk(
    monkeypatch, fake_device, stock_boot_stubs
):
    """W12 is a WEG-2 refusal; ungated it deletes the fork's #89 hibernate.

    ``/hibernate`` (http_server.py:2034-2036) sets ``destination="disk"``,
    ``tags=["weights"]`` and posts THIS RPC, and hibernate is not gated on the
    memory saver (server_args.py:18214-18222 requires only --hibernate-dir).
    On a stock boot the adapter is the no-op one, so an ungated
    ``assert_memory_saver_active`` raises before ``_hibernate_park_weights``
    ever runs -- and the raise is not caught by the dispatcher
    (scheduler.py:2963), so it reaches ``parent_process.send_signal(SIGQUIT)``:
    a POST /hibernate kills the server instead of parking weights.  The
    registry's Class-1 adapter then swallows it
    (registry/adapters/class1_srt.py:401-408, `except Exception -> warning,
    stopping anyway`), so the feature dies silently.
    """
    adapter = FakeAdapter(enabled=False)
    manager = _make_manager(
        adapter=adapter,
        server_args=FakeServerArgs(
            enable_weights_cpu_backup=False, enable_memory_saver=False
        ),
        tp_worker=FakeTpWorker(),
    )
    parked = _record_hibernate_park(monkeypatch)
    req = wu.ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
    req.destination = "disk"

    out = manager.release_memory_occupation(req)

    assert isinstance(out, wu.ReleaseMemoryOccupationReqOutput)
    assert len(parked) == 1, (
        "the #89 hibernate disk park never ran on a stock boot; W12's "
        "first-sleep arm is ungated and deletes the feature"
    )


def test_weg2_boot_sleep_still_refuses_a_dead_adapter(fake_device, stock_boot_stubs):
    """The gate must narrow W12, never delete it: flag ON + adapter dead."""
    from sglang.srt.managers.weg2_memory_saver import Weg2MemorySaverInactive

    adapter = FakeAdapter(enabled=False)
    manager = _make_manager(
        adapter=adapter,
        server_args=FakeServerArgs(
            enable_weights_cpu_backup=False, enable_memory_saver=True
        ),
    )
    with pytest.raises(Weg2MemorySaverInactive):
        manager.release_memory_occupation(
            wu.ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
        )
    assert adapter.paused == []


def test_wake_does_not_reload_from_disk_on_a_stock_boot(monkeypatch):
    """A non-Weg-2 resume must be byte-for-byte the upstream path.

    ``--enable-memory-saver`` without ``--enable-weights-cpu-backup`` is the
    ORDINARY upstream RL configuration (both default False, server_args.py:6636
    / :6640).  Ungated, every such wake paid a full
    ``update_weights_from_disk`` -- measured 12.073/14.143/16.749 s on this rig
    (record (S) 2.6) -- and it is not side-effect-free: model_runner.py:2866-2872
    rewrites ``self.load_config`` from a bare ``LoadConfig(load_format=...)``
    (:2825), discarding download_dir / model_loader_extra_config /
    ignore_patterns, and records a ``model_runner.update_weights`` override
    event that no operator asked for.
    """
    manager = _make_manager(
        adapter=FakeAdapter(enabled=True),
        server_args=FakeServerArgs(
            enable_weights_cpu_backup=False, enable_memory_saver=False
        ),
    )
    calls = _record_disk_reload(monkeypatch)
    manager._weg2_wake_reload_weights()
    assert calls == [], (
        "a stock memory-saver wake performed a full checkpoint reload and "
        "rewrote the model runner's load_config"
    )


def test_wake_refuses_when_the_draft_backup_verdict_differs(monkeypatch):
    """``enable_weights_cpu_backup`` alone is not the authority.

    model_runner.py:2342-2344: the draft runner's region is cpu-backed by
    ``enable_weights_cpu_backup or (is_draft_worker and
    enable_draft_weights_cpu_backup)``.  Under the draft flag ALONE the two
    shards in this process disagree about whether their bytes were carried,
    and the one upstream primitive available here
    (``update_weights_from_disk``) serves both shards with one request -- so
    there is no arrangement that refills the main shard without also pushing
    the MAIN model path through the draft runner
    (eagle_worker_v2.py:3104-3107).  Ranks/shards never disagree: STOP.
    """
    from sglang.srt.managers.weg2_memory_saver import Weg2WakeRefused

    manager = _make_manager(
        adapter=FakeAdapter(enabled=True),
        server_args=FakeServerArgs(
            enable_weights_cpu_backup=False,
            enable_draft_weights_cpu_backup=True,
        ),
        draft_worker=object(),
    )
    calls = _record_disk_reload(monkeypatch)
    with pytest.raises(Weg2WakeRefused) as exc:
        manager._weg2_wake_reload_weights()
    assert "--enable-draft-weights-cpu-backup" in str(exc.value)
    assert calls == [], "the refusal must precede the reload, not follow it"


def test_wake_refuses_when_the_draft_checkpoint_is_a_different_one(monkeypatch):
    """One request, one ``model_path`` -- and the draft runner gets it too."""
    from sglang.srt.managers.weg2_memory_saver import Weg2WakeRefused

    manager = _make_manager(
        adapter=FakeAdapter(enabled=True),
        server_args=FakeServerArgs(
            enable_weights_cpu_backup=False,
            speculative_draft_model_path="/models/other-draft",
        ),
        draft_worker=object(),
    )
    calls = _record_disk_reload(monkeypatch)
    with pytest.raises(Weg2WakeRefused) as exc:
        manager._weg2_wake_reload_weights()
    assert "/models/other-draft" in str(exc.value)
    assert calls == []


def test_wake_reloads_when_the_draft_shares_the_main_checkpoint(
    monkeypatch, noop_pcie_lock
):
    """The V1 arm: MTP, no separate draft path, both backup flags off."""
    manager = _make_manager(
        adapter=FakeAdapter(enabled=True),
        server_args=FakeServerArgs(enable_weights_cpu_backup=False),
        draft_worker=object(),
    )
    calls = _record_disk_reload(monkeypatch)
    manager._weg2_wake_reload_weights()
    assert len(calls) == 1
    assert calls[0].model_path == "/models/fake-27b"


def test_census_refuses_acceptance_when_the_arena_still_backs_device_memory():
    """B-M3: ``backed`` is the field that guards RESIDENT device memory.

    ``arena_census()``'s own docstring (kv_vmm_backing.py:307) calls ``backed``
    mapped physical memory, while ``retained`` is only unmapped-but-owned
    address space -- so this is the MORE load-bearing of the two, and it was
    the untested one.
    """
    from sglang.srt.managers.weg2_memory_saver import sleep_acceptance_census

    census = sleep_acceptance_census(
        nvml_uuid="GPU-fake",
        _process_bytes={4242: 1294 * 1024 * 1024},
        _memory_info=(32_088 * 1024 * 1024, 30_730 * 1024 * 1024),
        _arena_census={
            0: {"reserved": 8 << 30, "backed": 2 << 30, "retained": 0, "arenas": 1}
        },
        _pid=4242,
    )
    assert census.arena_backed_bytes == 2 << 30
    assert census.accepted is False
    assert "backs" in (census.refusal_reason or "")


def test_census_asserts_retain_handles_only_on_a_row_bearing_read():
    """The genuinely-earned branch: rows exist and they say retained == 0."""
    from sglang.srt.managers.weg2_memory_saver import sleep_acceptance_census

    census = sleep_acceptance_census(
        nvml_uuid="GPU-fake",
        _process_bytes={4242: 1294 * 1024 * 1024},
        _memory_info=(32_088 * 1024 * 1024, 30_730 * 1024 * 1024),
        _arena_census={
            0: {"reserved": 8 << 30, "backed": 0, "retained": 0, "arenas": 1}
        },
        _pid=4242,
    )
    assert census.retain_handles_asserted is True
    assert census.accepted is True
    assert "retain_handles=True" in census.format_line()


def test_sleep_rpc_runs_the_acceptance_census():
    """B-M6: the one instrument whose wiring was asserted only in prose.

    Building an instrument means wiring it in the SAME step (standing law);
    every sibling wiring claim in this slice is pinned by an AST gate.
    """
    func = _func_ast("release_memory_occupation")
    idx = _call_index(func.body, "_weg2_log_sleep_acceptance")
    empty_at = _call_index(func.body, "empty_cache")
    assert idx >= 0, "the sleep-acceptance census is never executed on the sleep path"
    assert empty_at >= 0
    assert empty_at < idx, "the census must read AFTER the allocator cache is emptied"


def _bind_real_pcie_lock(monkeypatch, tmp_path, *, timeout_s: float = 0.5):
    """Point the wrapper at a real ``flock`` in ``tmp_path`` with a short deadline."""
    from sglang.srt.managers import weg2_memory_saver as wms

    real = wms.pcie_transfer_lock
    monkeypatch.setattr(wu, "resolve_pcie_lock_key", lambda: "GPU-a")

    def _bound(**kwargs):
        return real(
            nvml_uuid=kwargs.get("nvml_uuid") or "GPU-a",
            lock_dir=str(tmp_path),
            timeout_s=timeout_s,
            label=kwargs.get("label", "transfer"),
        )

    monkeypatch.setattr(wu, "pcie_transfer_lock", _bound)


def test_pcie_lock_wrapper_propagates_the_named_refusal(monkeypatch, tmp_path):
    """B-M2: the wrapper, not the module-level lock.

    ``test_pcie_lock_wait_is_bounded`` exercises ``pcie_transfer_lock``; the
    WRAPPER ``_weg2_pcie_lock`` -- the thing the two RPCs actually call -- had
    no direct test, and its refusal survived only because an ``except
    Weg2PcieLockTimeout: raise`` clause happened to sit above a broad
    ``except Exception -> log + yield``.  Deleting those four lines turned
    every expiry into a silent unserialised overlap with the whole suite green.
    """
    from sglang.srt.managers import weg2_memory_saver as wms
    from sglang.srt.managers.weg2_memory_saver import Weg2PcieLockTimeout

    _bind_real_pcie_lock(monkeypatch, tmp_path)
    manager = _make_manager(adapter=FakeAdapter(enabled=True))

    hold = threading.Event()
    released = threading.Event()

    def _holder():
        with wms.pcie_transfer_lock(
            nvml_uuid="GPU-a", lock_dir=str(tmp_path), timeout_s=5.0
        ):
            hold.set()
            released.wait(timeout=10)

    thread = threading.Thread(target=_holder)
    thread.start()
    try:
        hold.wait(timeout=10)
        entered = False
        with pytest.raises(Weg2PcieLockTimeout):
            with manager._weg2_pcie_lock("sleep-D2H weights"):
                entered = True
        assert not entered, "the wrapper ran the transfer body after the deadline"
    finally:
        released.set()
        thread.join(timeout=10)


def test_pcie_lock_wrapper_degrades_when_the_card_key_is_unresolvable(monkeypatch):
    """The OTHER failure mode, kept apart on purpose: no key to serialise on.

    Intentional degrade -- this lock is a throughput guard; the correctness
    guards on this path are W12, W4 and the barriers.
    """

    def _boom():
        raise RuntimeError("no NVML on this box")

    monkeypatch.setattr(wu, "resolve_pcie_lock_key", _boom)
    manager = _make_manager(adapter=FakeAdapter(enabled=True))
    ran = False
    with manager._weg2_pcie_lock("wake-H2D weights"):
        ran = True
    assert ran, "an unresolvable card key must not skip the transfer"


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
