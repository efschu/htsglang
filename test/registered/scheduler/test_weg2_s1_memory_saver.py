"""Weg-2 slice S1 -- memory saver on the line.

Red-first tests for the six things S1 owns:

1. the CAMPAIGN (a) MUST_FIX release order (``flush_cache()`` BEFORE
   ``pause(kv_cache)``) -- boot-proven fatal on a hybrid GDN/mamba model,
   measured 2026-09-06 (CAMPAIGN_a_0906.md); the AST assertions below are that
   campaign's matched can-it-fail check, ported into the suite;
2. refusal W12 ``Weg2MemorySaverInactive`` on the no-op adapter, at the launch
   check the launcher calls AND at the first sleep;
3. ``empty_cache()`` in the sleep RPC, kept for the UNTAGGED remainder only:
   spec (S) 2.4 step 10's premise ("without it NVML free does not move") is
   MEASURED FALSE for the tagged regions -- campaign (a) measured NVML free
   1870.8 -> 30730.8 MiB and per-process 30,154 -> 1,294 MiB WITHOUT the call,
   9/9 cycles over two cold boots (CAMPAIGN_a_0906.md section 2 arm 1), because
   TMS unmaps the tagged segments' physical pages directly.  Gated with the
   census so a stock POST /hibernate keeps upstream's tail;
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
from typing import Any, List, Optional, Tuple

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


def _child_bodies(stmt: ast.stmt) -> List[Tuple[str, List[ast.stmt]]]:
    """Every ``list[ast.stmt]`` field ``stmt`` owns, named by its field."""
    out: List[Tuple[str, List[ast.stmt]]] = []
    for field in ("body", "orelse", "finalbody"):
        value = getattr(stmt, field, None)
        if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
            out.append((field, value))
    for i, handler in enumerate(getattr(stmt, "handlers", []) or []):
        out.append((f"handler[{i}]", handler.body))
    return out


def _stmt_path_to_call(
    body: List[ast.stmt], needle: str
) -> Optional[List[Tuple[List[ast.stmt], int, str]]]:
    """Path from ``body`` down to the statement whose OWN expression calls ``needle``.

    Each hop is ``(enclosing_body, index_in_it, field_name_of_that_body)``; the
    last hop is the innermost statement holding the call.  ``_call_index``
    returns only the TOP-LEVEL index, which is why the order gate it fed could
    not fail once the call was nested (own mutant RB2-M2, which the whole suite
    survived: a second ``_export_static_state`` appended INSIDE the PCIe-lock
    ``with``, after the pause).
    """
    for i, stmt in enumerate(body):
        if not any(
            isinstance(sub, ast.Call) and needle in ast.unparse(sub)
            for sub in ast.walk(stmt)
        ):
            continue
        for field, child in _child_bodies(stmt):
            deeper = _stmt_path_to_call(child, needle)
            if deeper is not None:
                return [(body, i, field)] + deeper
        return [(body, i, "")]
    return None


def _bodies_running_after(stmt: ast.stmt, taken_field: str) -> List[str]:
    """Sibling bodies of ``stmt`` that still execute after ``taken_field`` finishes.

    ``if``/``with`` have none (``orelse`` is mutually exclusive, a ``with`` owns
    one body).  ``try`` runs ``else``/``finally`` afterwards, and a loop both
    repeats its body and runs its ``else`` afterwards -- so "last statement of
    the block" would stop meaning "nothing runs after the pause" if the code
    ever grew one of those shapes around it.
    """
    after: List[str] = []
    if isinstance(stmt, ast.Try):
        if taken_field == "body":
            after += [f for f in ("orelse", "finalbody") if getattr(stmt, f, None)]
        elif taken_field.startswith("handler") or taken_field == "orelse":
            after += ["finalbody"] if stmt.finalbody else []
    elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
        after.append("the loop repeats its own body")
        if stmt.orelse:
            after.append("orelse")
    return after


def _assert_nothing_runs_after_the_call(
    body: List[ast.stmt], needle: str, block_name: str
) -> None:
    """``needle`` is the LAST thing this block does, at EVERY nesting level."""
    path = _stmt_path_to_call(body, needle)
    assert path is not None, f"{needle} is never called in the {block_name} block"
    depth_names = " -> ".join(
        f"{type(enclosing[index]).__name__}[{index}/{len(enclosing) - 1}]"
        for enclosing, index, _ in path
    )
    print(f"{block_name}: {needle} nesting path {depth_names}")
    for level, (enclosing, index, field) in enumerate(path):
        assert index == len(enclosing) - 1, (
            f"{needle} sits at nesting level {level} as statement [{index}] of "
            f"{len(enclosing)}; {len(enclosing) - 1 - index} statement(s) run "
            f"after it in the {block_name} block (path {depth_names}). Every "
            "one of them touches pages the pause has already unmapped."
        )
        if level + 1 < len(path):
            stmt = enclosing[index]
            running_after = _bodies_running_after(stmt, path[level + 1][2])
            assert not running_after, (
                f"{needle} is nested inside a "
                f"{type(stmt).__name__} whose {', '.join(running_after)} still "
                f"runs after it in the {block_name} block; 'last statement' no "
                "longer means 'nothing touches the released pages afterwards'"
            )


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeAdapter:
    """Stands in for ``TorchMemorySaverAdapter``; records the tag order.

    ``events`` is ONE ordered list shared with the manager's injected
    ``flush_cache`` (see :func:`_make_manager`).  Two separate lists cannot pin
    an ORDER between a call recorded in one and a call recorded in the other,
    which is how own mutant RB-M6 (``flush_cache()`` wrapped in an
    always-false guard) survived the whole suite: the AST gates read a
    STATEMENT INDEX and no test recorded that the call ever happened.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self.paused: List[str] = []
        self.resumed: List[str] = []
        self.regions: List[dict] = []
        self.events: List[str] = []

    @property
    def enabled(self) -> bool:
        return self._enabled

    def pause(self, tag: str) -> None:
        self.paused.append(tag)
        self.events.append(f"pause:{tag}")

    def resume(self, tag: str) -> None:
        self.resumed.append(tag)
        self.events.append(f"resume:{tag}")

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):
        """Records which TMS tag is ACTIVE while the body allocates."""
        entry = {"tag": tag, "enable_cpu_backup": enable_cpu_backup, "closed": False}
        self.regions.append(entry)
        self.events.append(f"region-enter:{tag}")
        try:
            yield tag
        finally:
            entry["closed"] = True
            self.events.append(f"region-exit:{tag}")

    def active_region_tag(self) -> Optional[str]:
        for entry in reversed(self.regions):
            if not entry["closed"]:
                return entry["tag"]
        return None


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


class FakeModelConfig:
    def __init__(self, quantization: Optional[str] = None):
        self.quantization = quantization


class FakeTpWorker:
    """Just enough of ``self.tp_worker.model_runner.model`` for the sleep RPC."""

    class _Runner:
        model = object()
        model_config = FakeModelConfig()

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

    def _flush_cache(*a, **k):
        # Recorded into the ADAPTER's list, so `flush` and `pause:<tag>` are
        # entries in ONE ordered sequence and their order is assertable.
        adapter.events.append("flush")
        return True

    return wu.SchedulerWeightUpdaterManager(
        tp_worker=tp_worker,
        draft_worker=draft_worker,
        tp_cpu_group=None,
        memory_saver_adapter=adapter,
        flush_cache=_flush_cache,
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
    # `_call_index` matches a Call nested ANYWHERE inside a top-level
    # statement, so an index comparison alone is satisfied by a flush wrapped
    # in a guard that is never true (own mutant RB-M6, which the whole suite
    # survived).  Require the UNCONDITIONAL form: a bare expression statement
    # of this block, not a call inside an If/Try/comprehension.  The matching
    # behavioural pin is test_kv_block_flush_actually_runs_before_the_pause.
    flush_stmt = body[flush_at]
    assert isinstance(flush_stmt, ast.Expr) and isinstance(
        flush_stmt.value, ast.Call
    ), (
        f"flush_cache() is not an unconditional statement of the kv_cache "
        f"block; it sits inside a {type(flush_stmt).__name__}, so the pool may "
        "never be quiesced before the pause unmaps its pages"
    )


def test_kv_block_pause_is_the_last_statement():
    """Nothing device-touching may follow the pause inside the same block.

    NESTING-AWARE (own mutant RB2-M2 on the sibling tag): a top-level statement
    index is satisfied by anything appended INSIDE a ``with``/``if`` that ends
    the block, and the weights leg already has exactly such a ``with`` (the
    PCIe lock).  The kv leg has no lock today, so this assertion carries the
    same shape in advance -- the gate must stay correct if one is ever added
    here, not go quietly blind the day it is.
    """
    body = _tag_block(
        _func_ast("release_memory_occupation"), "GPU_MEMORY_TYPE_KV_CACHE"
    )
    _assert_nothing_runs_after_the_call(
        body, "pause(GPU_MEMORY_TYPE_KV_CACHE)", "kv_cache"
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
        "release_memory_occupation did not empty the allocator cache; the "
        "TAGGED regions release without it (campaign (a), measured), but the "
        "untagged remainder stays in torch's reserve"
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
        before_bytes=30_154 * 1024 * 1024,
        min_released_fraction=0.5,
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
    """Context expressions of EVERY enclosing ``with`` around a ``needle`` call.

    Nested ``with`` statements are collected too -- the wake refill now sits
    inside ``region(GPU_MEMORY_TYPE_WEIGHTS)`` AND inside ``_weg2_pcie_lock``,
    and a helper that reads only the outermost one would report the inner guard
    as absent.
    """
    found: List[str] = []
    for stmt in body:
        for sub in ast.walk(stmt):
            if not isinstance(sub, ast.With):
                continue
            if any(
                isinstance(call, ast.Call) and needle in ast.unparse(call)
                for call in ast.walk(sub)
            ):
                found.extend(ast.unparse(item.context_expr) for item in sub.items)
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
        lambda self, before=None, tags=None: None,
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
        before_bytes=30_154 * 1024 * 1024,
        min_released_fraction=0.5,
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
    # Both calls now sit inside the SAME `if weg2_memory_saver_on:` block (the
    # stock hibernate path must stay byte-for-byte upstream), so the ordering
    # has to be read inside that block -- at the function's top level the two
    # would resolve to one and the same `if` statement and the assertion below
    # would compare an index with itself.
    block: List[ast.stmt] = []
    for stmt in func.body:
        if isinstance(stmt, ast.If) and "_weg2_log_sleep_acceptance" in ast.unparse(
            stmt
        ):
            block = stmt.body
            break
    assert block, (
        "the sleep-acceptance census is not inside a guarded block of "
        "release_memory_occupation"
    )
    idx = _call_index(block, "_weg2_log_sleep_acceptance")
    empty_at = _call_index(block, "empty_cache")
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


# ---------------------------------------------------------------------------
# 9. FIX ROUND 2 -- the census gets a criterion, W12 gets its hazard back,
#    the card-pin guard gets one definition, and the weights block gets the
#    kv block's prohibition.
# ---------------------------------------------------------------------------

MIB_ = 1024 * 1024
#: The two ends of campaign (a)'s own swing, per-process, in MiB
#: (CAMPAIGN_a_0906.md section 2 arm 1, n=9 over two cold boots).
AWAKE_MIB = 30_154
ASLEEP_MIB = 1_294


def _census(**kwargs):
    from sglang.srt.managers.weg2_memory_saver import sleep_acceptance_census

    base = dict(
        nvml_uuid="GPU-fake",
        _memory_info=(32_088 * MIB_, 30_730 * MIB_),
        _arena_census={},
        _pid=4242,
    )
    base.update(kwargs)
    return sleep_acceptance_census(**base)


def test_census_refuses_a_sleep_that_released_nothing():
    """The condition the instrument exists to catch, at the number it happens at.

    A no-op ``pause()`` returns success and leaves the whole shard resident.
    Before this round the census reported ``accepted=True`` for BOTH ends of
    campaign (a)'s 28.9 GiB swing -- the same verdict at 30,154 MiB and at
    1,294 MiB -- so the number it printed was decorative.
    """
    census = _census(
        before_bytes=AWAKE_MIB * MIB_,
        min_released_fraction=0.5,
        _process_bytes={4242: AWAKE_MIB * MIB_},
    )
    assert census.proc_used_bytes == AWAKE_MIB * MIB_
    assert census.released_bytes == 0
    assert census.accepted is False, (
        "the census accepted a rank that released nothing -- a gate that "
        "cannot fail is not a gate"
    )
    assert "released" in (census.refusal_reason or "")


def test_census_accepts_the_measured_asleep_reading():
    """The other end of the same swing must still pass, or the gate is a wall."""
    census = _census(
        before_bytes=AWAKE_MIB * MIB_,
        min_released_fraction=0.5,
        _process_bytes={4242: ASLEEP_MIB * MIB_},
    )
    assert census.accepted is True
    assert census.released_bytes == (AWAKE_MIB - ASLEEP_MIB) * MIB_
    line = census.format_line()
    assert "released=" in line and "min_released_fraction=" in line


def test_census_refuses_when_no_residency_criterion_is_supplied():
    """No criterion is not a pass; it is a reading."""
    census = _census(_process_bytes={4242: ASLEEP_MIB * MIB_})
    assert census.accepted is False
    assert "criterion" in (census.refusal_reason or "")
    assert "no criterion" in census.denominator


def test_census_refuses_residency_above_the_declared_ceiling():
    """The S3 form: a measured D_c ceiling, graded on the same instrument."""
    census = _census(
        expected_max_resident_bytes=2_048 * MIB_,
        _process_bytes={4242: AWAKE_MIB * MIB_},
    )
    assert census.accepted is False
    assert "ceiling" in (census.refusal_reason or "")

    ok = _census(
        expected_max_resident_bytes=2_048 * MIB_,
        _process_bytes={4242: ASLEEP_MIB * MIB_},
    )
    assert ok.accepted is True


def test_census_refuses_when_the_pre_pause_reading_was_blind():
    """A ``before`` whose own NVML read failed cannot grade an ``after``."""
    census = _census(
        before_bytes=None,
        min_released_fraction=0.5,
        _process_bytes={4242: ASLEEP_MIB * MIB_},
    )
    assert census.accepted is False
    assert "criterion" in (census.refusal_reason or "")


def test_sleep_rpc_grades_the_census_against_a_pre_pause_reading(
    monkeypatch, fake_device, stock_boot_stubs
):
    """The wiring: the RPC must take a BEFORE reading and hand it to the census.

    Recorded on the real entry point, not on prose: the sleep path takes two
    readings of the same instrument, and the second one is graded against the
    first.  Without the ``before`` the census has no criterion and refuses.
    """
    from sglang.srt.managers import weg2_memory_saver as wms

    seen: List[dict] = []
    answers = [AWAKE_MIB * MIB_, ASLEEP_MIB * MIB_]

    def _fake_census(**kwargs):
        seen.append(dict(kwargs))
        return wms.sleep_acceptance_census(
            nvml_uuid="GPU-fake",
            _memory_info=(32_088 * MIB_, 30_730 * MIB_),
            _arena_census={},
            _pid=4242,
            _process_bytes={4242: answers[min(len(seen) - 1, 1)]},
            **kwargs,
        )

    monkeypatch.undo()  # drop the stubbed-out _weg2_log_sleep_acceptance
    monkeypatch.setattr(wu, "_export_static_state", lambda model: {"stub": True})
    monkeypatch.setattr(wu.torch.distributed, "barrier", lambda *a, **k: None)
    monkeypatch.setattr(wu.torch, "get_device_module", lambda *a, **k: fake_device)
    monkeypatch.setattr(wu, "sleep_acceptance_census", _fake_census)

    manager = _make_manager(
        adapter=FakeAdapter(enabled=True),
        server_args=FakeServerArgs(
            enable_weights_cpu_backup=False, enable_memory_saver=True
        ),
    )
    manager.release_memory_occupation(
        wu.ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
    )

    assert len(seen) == 2, (
        f"expected a pre-pause reading and a post-pause verdict, saw {len(seen)} "
        "census call(s); an ungraded reading is a number, not a verdict"
    )
    assert (
        seen[0].get("before_bytes") is None
        and seen[0].get("min_released_fraction") is None
    ), "the pre-pause reading must not be graded against itself"
    assert seen[1].get("before_bytes") == AWAKE_MIB * MIB_
    # LITERAL, not `== wms.WEG2_SLEEP_MIN_RELEASED_FRACTION`: that comparison
    # is a tautology -- it holds for whatever the constant is detuned to, and
    # own mutant RB-M1 (0.5 -> 0.005) survived the whole suite because of it.
    # The shipped value is pinned here and again in
    # test_shipped_min_released_fraction_is_the_recorded_number.
    assert seen[1].get("min_released_fraction") == 0.5
    # Both calls must declare the POPULATION they read, or the printed verdict
    # cannot be attributed to the request that produced it.
    assert seen[0].get("tags") == [GPU_MEMORY_TYPE_KV_CACHE]
    assert seen[1].get("tags") == [GPU_MEMORY_TYPE_KV_CACHE]


def test_first_sleep_refuses_a_genuine_sleep_on_a_flagless_boot(fake_device):
    """Spec section 10 S1 mutant M1: drop --enable-memory-saver -> W12 fires.

    The hazard section 2.1 names is a LAUNCHER EDIT that drops the flag.  A
    gate keyed on that same flag can never see it -- the launch arm is gated on
    it too, so both arms go silent together.  The request object carries the
    discriminator that actually separates the two cases
    (io_struct.py:1964 ``destination``), and the #89 hibernate park is the only
    caller that sets it.
    """
    from sglang.srt.managers.weg2_memory_saver import Weg2MemorySaverInactive

    adapter = FakeAdapter(enabled=False)
    manager = _make_manager(
        adapter=adapter,
        server_args=FakeServerArgs(
            enable_weights_cpu_backup=False, enable_memory_saver=False
        ),
        tp_worker=FakeTpWorker(),
    )
    req = wu.ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
    assert getattr(req, "destination", None) is None

    with pytest.raises(Weg2MemorySaverInactive):
        manager.release_memory_occupation(req)
    assert adapter.paused == [], "a refusal that already paused a tag is not a refusal"


def test_stock_hibernate_touches_no_device_call_after_the_pauses(
    monkeypatch, fake_device, stock_boot_stubs
):
    """The #89 park must stay byte-for-byte upstream, on the tail too.

    ``empty_cache()`` and the acceptance census are Weg-2 additions.  Left
    ungated they added a post-pause device call and two NVML reads to a stock
    ``POST /hibernate`` -- a path that has never been executed on metal in that
    shape, and the one thing the round-1 gating exercise was for.
    """
    manager = _make_manager(
        adapter=FakeAdapter(enabled=False),
        server_args=FakeServerArgs(
            enable_weights_cpu_backup=False, enable_memory_saver=False
        ),
        tp_worker=FakeTpWorker(),
    )
    _record_hibernate_park(monkeypatch)
    req = wu.ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_WEIGHTS])
    req.destination = "disk"

    manager.release_memory_occupation(req)

    assert "empty_cache" not in fake_device.calls, (
        "a stock POST /hibernate ran the Weg-2 empty_cache(); that path is "
        "upstream's and has never been executed on metal with this call in it"
    )
    assert fake_device.calls == [
        "synchronize"
    ], f"the stock tail is not upstream's: {fake_device.calls}"


def test_weg2_sleep_still_empties_the_cache_and_censuses(fake_device, stock_boot_stubs):
    """The gate must narrow the addition, never delete it."""
    manager = _make_manager(
        adapter=FakeAdapter(enabled=True),
        server_args=FakeServerArgs(
            enable_weights_cpu_backup=False, enable_memory_saver=True
        ),
        tp_worker=FakeTpWorker(),
    )
    manager.release_memory_occupation(
        wu.ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
    )
    assert "empty_cache" in fake_device.calls


def test_weights_block_pause_is_the_last_statement():
    """The kv block's gate, mirrored onto its sibling tag (own mutant RB-M1).

    ``_export_static_state`` clones every named buffer of a model allocated
    inside ``region(GPU_MEMORY_TYPE_WEIGHTS)`` (model_runner.py:2344-2348), so
    a second export appended after the pause reads unmapped pages -- the
    campaign (a) fault on the weights tag.  The existing order gate uses the
    FIRST matching statement and stays green under exactly that duplicate.

    NESTING-AWARE since fix 4.  The previous form compared TOP-LEVEL statement
    indices, and fix 1 had since nested the pause inside
    ``with self._weg2_pcie_lock("sleep-D2H weights"):`` -- which IS the last
    top-level statement of the block no matter what is appended after the pause
    inside it.  Own mutant RB2-M2 (a second ``_export_static_state`` in exactly
    that position, i.e. the shape this docstring names) survived the whole
    suite 80/80; only the control one level further out was killed.  A gate that
    cannot fail on the hazard its own docstring names is not a gate, so the
    assertion now walks the nesting path and requires "nothing after" at EVERY
    level.
    """
    body = _tag_block(_func_ast("release_memory_occupation"), "GPU_MEMORY_TYPE_WEIGHTS")
    _assert_nothing_runs_after_the_call(
        body, "pause(GPU_MEMORY_TYPE_WEIGHTS)", "weights"
    )


def test_census_reads_this_pid_not_the_colocated_sibling():
    """Weg 2 puts TWO ranks on every card, so the pid key is load-bearing.

    ``process_bytes_on_uuid`` returns ``{pid: bytes}`` for EVERY compute
    process on the card (registry/nvml.py:402-425).  Reading the first value
    instead of this pid's would print the awake sibling's residency as the
    dormant rank's verdict -- own mutant RB-M4, which the whole suite survived
    green because every fixture passed a single-entry table.
    """
    census = _census(
        before_bytes=AWAKE_MIB * MIB_,
        min_released_fraction=0.5,
        _process_bytes={9999: 27 << 30, 4242: ASLEEP_MIB * MIB_},
    )
    assert census.proc_used_bytes == ASLEEP_MIB * MIB_
    assert census.accepted is True
    assert "over 2 compute process(es)" in census.denominator


def test_census_refuses_when_this_pid_is_absent_from_nvml():
    """The pid-miss branch, which no test reached."""
    census = _census(
        before_bytes=AWAKE_MIB * MIB_,
        min_released_fraction=0.5,
        _process_bytes={9999: 27 << 30},
    )
    assert census.proc_used_bytes is None
    assert census.accepted is False
    assert "not among" in (census.refusal_reason or "")


def test_census_declines_to_create_a_cuda_context_to_name_the_card(monkeypatch):
    """The canonical instrument's guard, which this one had dropped.

    ``sleep_acceptance_census()`` is public and the S3 launcher calls it
    PRE-LAUNCH per rank -- the exact moment where ``current_device_uuid()``
    would fall back to torch and buy a CUDA context, corrupting the very
    residency number the census reports.
    """
    from sglang.srt.mem_ledger import flight_recorder as fr
    from sglang.srt.registry import nvml as registry_nvml

    monkeypatch.setattr(fr, "cuda_initialized", lambda: False)
    monkeypatch.setattr(registry_nvml, "pin_resolvable_without_cuda", lambda: False)

    def _must_not_be_called():
        raise AssertionError(
            "the census resolved the card through torch and bought a CUDA "
            "context to describe the state before it"
        )

    monkeypatch.setattr(registry_nvml, "current_device_uuid", _must_not_be_called)

    from sglang.srt.managers.weg2_memory_saver import sleep_acceptance_census

    census = sleep_acceptance_census()
    assert census.nvml_uuid is None
    assert census.accepted is False
    assert "no CUDA context yet" in (census.refusal_reason or "")


def test_the_card_pin_guard_has_exactly_one_definition():
    """One job, one mover: the guard's reason string lives in one place.

    ``flight_recorder`` is the canonical holder of this payload
    (``_nvml_view`` / ``_kv_arena_view``); the Weg-2 census is a verdict
    wrapper over it.  A second copy of the guard drifts from the first, and a
    second copy of its reason string makes two boot logs say the same absence
    in two ways.
    """
    import inspect as _inspect

    from sglang.srt.managers import weg2_memory_saver as wms
    from sglang.srt.mem_ledger import flight_recorder as fr

    assert callable(fr.card_pin_unresolvable_without_cuda)
    weg2_src = open(_inspect.getsourcefile(wms), "r", encoding="utf-8").read()
    assert (
        "card_pin_unresolvable_without_cuda" in weg2_src
    ), "the Weg-2 census does not use the canonical card-pin guard"
    assert "refusing to create a context" not in weg2_src, (
        "the guard's reason string was COPIED into weg2_memory_saver.py "
        "instead of being used from its canonical holder"
    )
    # and the recorder still uses it too, so there is one definition and two
    # users rather than two definitions.
    fr_src = open(_inspect.getsourcefile(fr), "r", encoding="utf-8").read()
    assert fr_src.count("refusing to create a context") == 1


# ---------------------------------------------------------------------------
# 7. ROUND-3 FINDINGS
# ---------------------------------------------------------------------------

# --- finding 4: the MUST_FIX release order had no BEHAVIOURAL pin -----------


def test_kv_block_flush_actually_runs_before_the_pause(fake_device):
    """The campaign (a) MUST_FIX, pinned by what RUNS, not by a statement index.

    Own mutant RB-M6 wrapped ``self.flush_cache()`` in ``if not
    self.offload_tags:`` -- always false, because ``offload_tags.add(tag)`` runs
    for every tag before this block is entered -- and the whole 59-test suite
    stayed green: both existing gates read an AST index, ``_call_index``
    matches a Call nested anywhere inside a statement, and the injected
    ``flush_cache`` was never recorded.  On metal that mutant leaves the radix
    tree / HybridReqToTokenPool / MambaPool unreset while the pause unmaps
    their pages, and (S) 2.4 says the same call is what clears the L2 ring at
    every sleep -- the premise (S) 4.2.2 rests on.
    """
    adapter = FakeAdapter(enabled=True)
    manager = _make_manager(adapter=adapter)
    manager.release_memory_occupation(
        wu.ReleaseMemoryOccupationReqInput(tags=[GPU_MEMORY_TYPE_KV_CACHE])
    )
    assert adapter.events == ["flush", f"pause:{GPU_MEMORY_TYPE_KV_CACHE}"], (
        "expected exactly one flush_cache() and then pause(kv_cache); saw "
        f"{adapter.events}. A flush that does not run leaves the mamba pool "
        "live while the pause unmaps its pages (CAMPAIGN (a), boot-fatal)."
    )


# --- finding 5: the criterion constant shipped untested ---------------------


def test_shipped_min_released_fraction_is_the_recorded_number():
    """A constant nothing compares to a LITERAL can be detuned freely.

    Own mutant RB-M1 (0.5 -> 0.005) survived the whole suite: every census test
    passed its own local ``0.5`` and the only reference to the constant was
    ``seen[1][...] == wms.WEG2_SLEEP_MIN_RELEASED_FRACTION``, which holds for
    any value.  Provenance of the number is in the module docstring: campaign
    (a) measured 95.7 % released / 4.3 % retained (30,154 -> 1,294 MiB, n=9,
    two cold boots); a no-op releases 0 %.
    """
    from sglang.srt.managers import weg2_memory_saver as wms

    assert wms.WEG2_SLEEP_MIN_RELEASED_FRACTION == 0.5


def test_shipped_criterion_refuses_a_one_percent_release():
    """The SHIPPED constant, not a local literal, must refuse a no-op sleep."""
    from sglang.srt.managers import weg2_memory_saver as wms

    census = _census(
        before_bytes=AWAKE_MIB * MIB_,
        min_released_fraction=wms.WEG2_SLEEP_MIN_RELEASED_FRACTION,
        _process_bytes={4242: int(AWAKE_MIB * MIB_ * 0.99)},
    )
    assert census.accepted is False, (
        "a sleep that released 1 % of the shard is a no-op sleep; the shipped "
        f"criterion {wms.WEG2_SLEEP_MIN_RELEASED_FRACTION} accepted it"
    )
    assert "below the required" in (census.refusal_reason or "")


def test_shipped_criterion_still_accepts_the_measured_sleep():
    """And it is not a wall: campaign (a)'s own genuine sleep must pass."""
    from sglang.srt.managers import weg2_memory_saver as wms

    census = _census(
        before_bytes=AWAKE_MIB * MIB_,
        min_released_fraction=wms.WEG2_SLEEP_MIN_RELEASED_FRACTION,
        _process_bytes={4242: ASLEEP_MIB * MIB_},
    )
    assert census.accepted is True, (
        "the measured 95.7 % release must pass, or the criterion is a gate "
        f"that refuses the working case: {census.refusal_reason}"
    )


# --- finding 2: the census graded a population it never named ---------------


def test_census_prints_its_tag_population():
    census = _census(
        tags=[GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS],
        before_bytes=AWAKE_MIB * MIB_,
        min_released_fraction=0.5,
        _process_bytes={4242: ASLEEP_MIB * MIB_},
    )
    line = census.format_line()
    assert "tags=['kv_cache', 'weights']" in line, line
    assert "declared tags" in census.denominator
    assert census.accepted is True


def test_census_line_says_when_no_tag_set_was_declared():
    census = _census(
        before_bytes=AWAKE_MIB * MIB_,
        min_released_fraction=0.5,
        _process_bytes={4242: ASLEEP_MIB * MIB_},
    )
    assert "tags=n/a (whole-process residency)" in census.format_line()


def test_weights_only_release_is_not_graded_against_the_whole_process_floor():
    """The #89 park's request shape, and the false FAIL it used to produce.

    ``/hibernate`` posts ``destination="disk", tags=["weights"]``
    (http_server.py:2034-2035).  On a Weg-2 boot the census runs for it too,
    and the delta floor is 50 % of everything this process holds -- an awake
    residency that also carries KV, graphs, activations and the CUDA context.
    A weights-only release need not clear it (per-rank weight images 13,724.7 /
    7,422.5 / 8,382.4 MiB, spec section 2.6), so a correct park printed
    ``accepted=False``.
    """
    census = _census(
        tags=[GPU_MEMORY_TYPE_WEIGHTS],
        before_bytes=AWAKE_MIB * MIB_,
        min_released_fraction=0.5,
        # A genuine weights-only park: the shard is gone, the rest stays.
        _process_bytes={4242: int(AWAKE_MIB * MIB_ * 0.55)},
    )
    assert census.delta_form_in_force is False
    assert "below the required" not in (census.refusal_reason or ""), (
        "a weights-only release was graded against a fraction of the WHOLE "
        f"process residency: {census.refusal_reason}"
    )
    assert "partial tag set" in (census.refusal_reason or "")
    line = census.format_line()
    assert "min_released_fraction=n/a (partial tag set: ['weights']" in line, line


def test_kv_only_release_never_passes_with_the_weights_shard_still_resident():
    """The mirror error: a false PASS on the request the suite itself drives.

    A ``tags=["kv_cache"]`` release on a rank whose KV exceeds half its
    residency clears the whole-process floor with the entire weights image
    still on the card.  That is not a hypothetical shape -- it is exactly what
    ``test_release_rpc_wires_the_census_between_two_readings`` posts.
    """
    weights_image_bytes = 13_725 * MIB_
    census = _census(
        tags=[GPU_MEMORY_TYPE_KV_CACHE],
        before_bytes=AWAKE_MIB * MIB_,
        min_released_fraction=0.5,
        # 47 % of the awake residency left: 53 % released, so the 50 % floor
        # is CLEARED -- and what stays is still more than the whole weights
        # image, which a kv-only release does not touch at all.
        _process_bytes={4242: int(AWAKE_MIB * MIB_ * 0.47)},
    )
    assert census.proc_used_bytes > weights_image_bytes
    assert census.accepted is False, (
        "the census accepted a kv-only release while this process still holds "
        f"{census.proc_used_mib:.1f} MiB, more than the whole weights image"
    )
    assert "partial tag set" in (census.refusal_reason or "")


def test_partial_tag_set_still_grades_against_a_supplied_ceiling():
    """Suppressing the delta form must not disarm the S3 ceiling form."""
    census = _census(
        tags=[GPU_MEMORY_TYPE_WEIGHTS],
        before_bytes=AWAKE_MIB * MIB_,
        min_released_fraction=0.5,
        expected_max_resident_bytes=2_000 * MIB_,
        _process_bytes={4242: 1_500 * MIB_},
    )
    assert census.delta_form_in_force is False
    assert census.accepted is True, census.refusal_reason
    assert "ceiling form" in census.denominator


# --- finding 1: the backup-OFF wake refilled OUTSIDE the weights region -----


def test_wake_reload_is_lexically_inside_the_weights_region():
    """AST gate, same shape as test_sleep_d2h_is_serialised_on_the_card."""
    reload_body = _func_ast("_weg2_wake_reload_weights").body
    guards = _with_items_around(reload_body, "update_weights_from_disk")
    assert any("region" in g and "GPU_MEMORY_TYPE_WEIGHTS" in g for g in guards), (
        "the backup-OFF refill runs OUTSIDE region(GPU_MEMORY_TYPE_WEIGHTS); "
        "the repacked parameters it allocates are then untagged and the next "
        f"pause(weights) releases a region the model no longer points at. "
        f"guards seen: {guards}"
    )


def test_wake_reload_allocates_under_the_weights_tag(monkeypatch, noop_pcie_lock):
    """BEHAVIOURAL: which TMS tag is ACTIVE while the loader allocates.

    ``load_weights_and_postprocess`` ends in
    ``quant_method.process_weights_after_loading(module)``, which for every
    repacking scheme REPLACES the parameter with a fresh device allocation
    (loader.py:921-931).  An allocation made with no region active is not under
    the weights tag, so after the first such wake ``pause(weights)`` releases a
    region the live weights are no longer in -- the silent-no-op class one level
    down, invisible until the SECOND sleep.
    """
    adapter = FakeAdapter(enabled=True)
    manager = _make_manager(
        adapter=adapter,
        server_args=FakeServerArgs(enable_weights_cpu_backup=False),
    )
    seen: List[Optional[str]] = []

    def _stub(self, req):
        seen.append(adapter.active_region_tag())
        return _ok()

    monkeypatch.setattr(
        wu.SchedulerWeightUpdaterManager, "update_weights_from_disk", _stub
    )
    manager._weg2_wake_reload_weights()

    assert seen == [GPU_MEMORY_TYPE_WEIGHTS], (
        f"the refill ran under region tag {seen!r}; the boot load runs under "
        "GPU_MEMORY_TYPE_WEIGHTS (model_runner.py:2345-2348) and the wake must "
        "put the repacked parameters in the same place"
    )
    assert adapter.regions[0]["enable_cpu_backup"] is False, (
        "model_runner.py:2342-2344's expression is False for both shards on "
        "this path (main_carried False, draft_carried == main_carried)"
    )
    assert adapter.regions[0]["closed"] is True, "the region was left open"


def test_wake_reload_raise_becomes_a_named_refusal(monkeypatch, noop_pcie_lock):
    """model_runner re-runs the failing load as its rollback OUTSIDE any try.

    ``model_runner.py:2857-2865`` catches the first ``model_load_weights``
    failure and then calls it again, unguarded, so a raw exception escapes
    instead of the ``(False, message)`` tuple.  A wake leg that ends in an
    unnamed RuntimeError is the same undefined state as one that returns
    ``success=False`` and must carry the same name.
    """
    from sglang.srt.managers.weg2_memory_saver import Weg2WakeRefused

    manager = _make_manager(
        adapter=FakeAdapter(enabled=True),
        server_args=FakeServerArgs(enable_weights_cpu_backup=False),
    )

    def _boom(self, req):
        raise RuntimeError("Attempted to load weight into parameter")

    monkeypatch.setattr(
        wu.SchedulerWeightUpdaterManager, "update_weights_from_disk", _boom
    )
    with pytest.raises(Weg2WakeRefused) as excinfo:
        manager._weg2_wake_reload_weights()
    assert "RuntimeError" in str(excinfo.value)


def test_quantized_post_load_replaces_the_weight_parameter():
    """DESK PROOF of the hazard the W4 refusal below is built on.

    The reference line's own scheme.  ``process_weights_after_loading`` does
    ``layer.weight = Parameter(weight.t(), requires_grad=False)``
    (compressed_tensors_w8a8_int8.py:159), which drops the ``weight_loader``
    attribute and transposes the shape.  A LATER ``model.load_weights(iter)``
    therefore resolves the DEFAULT loader, whose
    ``assert param.size() == loaded_weight.size()`` (weight_utils.py:1709)
    cannot hold against a transposed parameter -- so the backup-OFF wake's
    refill raises on the FIRST wake, not the second.
    """
    torch = pytest.importorskip("torch")
    quantization = pytest.importorskip("compressed_tensors.quantization")
    from sglang.srt.layers.parameter import ModelWeightParameter
    from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8 import (  # noqa: E501
        CompressedTensorsW8A8Int8,
    )

    class _Layer(torch.nn.Module):
        pass

    layer = _Layer()
    layer.logical_widths = [4]
    layer.register_parameter(
        "weight",
        ModelWeightParameter(
            data=torch.zeros(4, 8, dtype=torch.int8),
            input_dim=1,
            output_dim=0,
            weight_loader=lambda *a, **k: None,
        ),
    )
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(torch.zeros(4, 1), requires_grad=False)
    )

    scheme = CompressedTensorsW8A8Int8.__new__(CompressedTensorsW8A8Int8)
    scheme.strategy = quantization.QuantizationStrategy.CHANNEL
    scheme.is_static_input_scheme = False
    scheme.input_symmetric = True

    assert hasattr(layer.weight, "weight_loader")
    before_shape = tuple(layer.weight.shape)
    scheme.process_weights_after_loading(layer)

    assert not hasattr(layer.weight, "weight_loader"), (
        "the repack kept the weight_loader, so a second load_weights would be "
        "defined and the W4 refusal below is over-broad"
    )
    assert tuple(layer.weight.shape) == before_shape[::-1], (
        f"expected a transpose, got {tuple(layer.weight.shape)} from " f"{before_shape}"
    )


def test_wake_refuses_the_backup_off_arm_on_a_quantized_checkpoint(
    monkeypatch, noop_pcie_lock
):
    from sglang.srt.managers.weg2_memory_saver import Weg2WakeRefused

    class _Worker:
        class _Runner:
            model = object()
            model_config = FakeModelConfig(quantization="compressed-tensors")

        model_runner = _Runner()

    manager = _make_manager(
        adapter=FakeAdapter(enabled=True),
        server_args=FakeServerArgs(enable_weights_cpu_backup=False),
        tp_worker=_Worker(),
    )
    calls = _record_disk_reload(monkeypatch)
    with pytest.raises(Weg2WakeRefused) as excinfo:
        manager._weg2_wake_reload_weights()
    assert calls == [], "the refusal must fire BEFORE anything is refilled"
    assert "compressed-tensors" in str(excinfo.value)
    assert "--enable-weights-cpu-backup" in str(excinfo.value)


def test_wake_still_reloads_an_unquantized_checkpoint(monkeypatch, noop_pcie_lock):
    """Can-it-pass: the refusal is not a wall on the arm that IS defined."""
    manager = _make_manager(
        adapter=FakeAdapter(enabled=True),
        server_args=FakeServerArgs(enable_weights_cpu_backup=False),
        tp_worker=FakeTpWorker(),
    )
    calls = _record_disk_reload(monkeypatch)
    manager._weg2_wake_reload_weights()
    assert len(calls) == 1


def test_backup_off_refusal_reads_the_config_when_the_flag_is_unset():
    """An auto-detected quant config without --quantization is covered too."""
    from sglang.srt.managers.weg2_memory_saver import (
        Weg2WakeRefused,
        assert_backup_off_wake_refill_is_defined,
        checkpoint_quantization,
    )

    class _ServerArgs:
        quantization = None

    assert (
        checkpoint_quantization(FakeModelConfig("awq_marlin"), _ServerArgs())
        == "awq_marlin"
    )
    assert checkpoint_quantization(FakeModelConfig(None), _ServerArgs()) is None
    assert (
        assert_backup_off_wake_refill_is_defined(quantization=None, context="t") is None
    )
    with pytest.raises(Weg2WakeRefused):
        assert_backup_off_wake_refill_is_defined(quantization="fp8", context="t")


def test_launch_arm_refuses_the_backup_off_quantized_combination():
    """The refusal is at LAUNCH too, where nothing is committed yet.

    Source gate: ``scheduler.py``'s memory-saver block must call the SAME
    function, and it must call it under the NEGATED
    ``enable_weights_cpu_backup``.

    AST, not a substring (own mutant RB2-M5, same class as F3-M2 and as the
    W12 launch gate's own lesson two functions up -- "look for the CALL, not
    the name"): flipping ``if not self.server_args.enable_weights_cpu_backup:``
    to ``if self.server_args.enable_weights_cpu_backup:`` leaves all three
    substrings in place and the suite green 80/80, while the one arm the
    refusal exists for -- backup OFF on a quantized checkpoint -- boots
    unrefused and is then caught only at the first wake, after
    ``resume(GPU_MEMORY_TYPE_WEIGHTS)`` has recommitted the VMM pages, which
    the block's own comment calls fatal for the group.
    """
    from sglang.srt.managers import scheduler as sched

    source = textwrap.dedent(
        inspect.getsource(sched.Scheduler.init_watch_dog_memory_saver_input_blocker)
    )
    tree = ast.parse(source)

    def _calls_the_refusal(node: ast.AST) -> bool:
        return any(
            isinstance(sub, ast.Call)
            and ast.unparse(sub.func).endswith(
                "assert_backup_off_wake_refill_is_defined"
            )
            for sub in ast.walk(node)
        )

    assert _calls_the_refusal(tree), (
        "the launch arm does not refuse the undefined backup-OFF/quantized "
        "combination; the first wake would then commit the VMM pages and fail "
        "inside the loader"
    )

    negated_guards = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.UnaryOp)
        and isinstance(node.test.op, ast.Not)
        and "enable_weights_cpu_backup" in ast.unparse(node.test.operand)
    ]
    print(f"negated enable_weights_cpu_backup guards found: {len(negated_guards)}")
    assert negated_guards, (
        "no `if not <...>enable_weights_cpu_backup:` guard in the launch block; "
        "the refusal is either ungated or gated on the WRONG polarity, in which "
        "case it fires on the safe arm and never on the undefined one. (This "
        "gate accepts the `not` spelling only, so a rewrite fails closed.)"
    )
    assert any(_calls_the_refusal(node) for node in negated_guards), (
        "assert_backup_off_wake_refill_is_defined is called, but not inside the "
        "`if not ...enable_weights_cpu_backup:` guard"
    )
    assert "checkpoint_quantization" in source


def _drive_launch_memory_saver_block(
    monkeypatch,
    *,
    enable_weights_cpu_backup: bool,
    quantization: Optional[str],
    enable_memory_saver: bool = True,
) -> Any:
    """Run ``init_watch_dog_memory_saver_input_blocker`` on a fake scheduler.

    Behavioural pin for the launch refusal.  The source gate above gets the
    polarity right structurally; this one gets it right by CONSEQUENCE, which
    is what "a refusal that cannot be shown to fire is not a refusal" asks for.
    Everything the method touches besides the refusal is stubbed: the two
    watchdog factories (they start threads), the adapter class (the real
    ``create(enable=True)`` needs the library), and ``get_bool_env_var`` (the
    input blocker reads ``self.ps``, which a fake scheduler has not got).
    ``SchedulerRecvSkipper.maybe_create`` returns ``None`` on
    ``scheduler_recv_interval <= 1`` (scheduler_recv_skipper.py:9-10), so it
    needs no stub.
    """
    from sglang.srt.managers import scheduler as sched

    class _FakeTMS:
        @staticmethod
        def create(enable):
            return FakeAdapter(enabled=bool(enable))

    monkeypatch.setattr(sched, "create_scheduler_watchdog", lambda *a, **k: "watchdog")
    monkeypatch.setattr(
        sched, "create_admission_wedge_watchdog", lambda *a, **k: "wedge"
    )
    monkeypatch.setattr(sched, "TorchMemorySaverAdapter", _FakeTMS)
    monkeypatch.setattr(sched, "get_bool_env_var", lambda *a, **k: False)

    server_args = FakeServerArgs(
        enable_weights_cpu_backup=enable_weights_cpu_backup,
        enable_memory_saver=enable_memory_saver,
    )
    server_args.watchdog_timeout = 300.0
    server_args.scheduler_recv_interval = 1
    server_args.quantization = None

    class _FakeScheduler:
        pass

    fake = _FakeScheduler()
    fake.server_args = server_args
    fake.model_config = FakeModelConfig(quantization)
    sched.Scheduler.init_watch_dog_memory_saver_input_blocker(fake)
    return fake


def test_launch_refuses_backup_off_on_a_quantized_checkpoint(monkeypatch):
    """Polarity, by consequence: the arm the refusal exists for MUST refuse."""
    from sglang.srt.managers.weg2_memory_saver import Weg2WakeRefused

    with pytest.raises(Weg2WakeRefused) as exc:
        _drive_launch_memory_saver_block(
            monkeypatch,
            enable_weights_cpu_backup=False,
            quantization="compressed-tensors",
        )
    assert "launch" in str(exc.value)


def test_launch_accepts_backup_on_with_the_same_quantized_checkpoint(monkeypatch):
    """The other polarity: with the backup funded the wake is defined.

    Without this half a refusal that fires on EVERY boot would also pass the
    test above.
    """
    fake = _drive_launch_memory_saver_block(
        monkeypatch,
        enable_weights_cpu_backup=True,
        quantization="compressed-tensors",
    )
    assert fake.memory_saver_adapter.enabled is True


def test_launch_accepts_backup_off_on_an_unquantized_checkpoint(monkeypatch):
    """The second defined lane named in the refusal message."""
    fake = _drive_launch_memory_saver_block(
        monkeypatch, enable_weights_cpu_backup=False, quantization=None
    )
    assert fake.memory_saver_adapter.enabled is True


def test_launch_leaves_a_stock_boot_untouched(monkeypatch):
    """No ``--enable-memory-saver`` -> neither W12 nor W4 is reachable.

    The denominator of the two tests above: they say what happens INSIDE the
    memory-saver block, not that the block is entered on a stock boot.
    """
    fake = _drive_launch_memory_saver_block(
        monkeypatch,
        enable_weights_cpu_backup=False,
        quantization="compressed-tensors",
        enable_memory_saver=False,
    )
    assert fake.memory_saver_adapter.enabled is False


# --- finding 3: two flocks on one physical link -----------------------------


def test_hibernate_park_takes_the_one_pcie_lock():
    """(S) 2.7 'adopt the lock, not the module' -- one lock per physical link.

    #89's park used to open its own ``<hibernate_dir>/.park_lock_<uuid>``.  An
    flock on that file and one on ``/dev/shm/.weg2-pcie-serialize-<uuid>`` are
    independent, so a park's D2H and a co-located sibling's wake-H2D overlapped
    freely -- the exact halving the S1 lock exists to prevent.
    """
    from sglang.srt.model_loader import hibernate

    source = inspect.getsource(hibernate.park_weights_to_disk)
    assert "pcie_transfer_lock" in source, "the park does not take the S1 lock"
    # AST, not a substring: own mutant F3-M2 replaced the CALL
    # (`lock_uuid = resolve_pcie_lock_key()`) with `lock_uuid = nvml_uuid` and
    # the substring gate stayed green on the import line and this comment.
    called = {
        node.func.id
        for node in ast.walk(ast.parse(textwrap.dedent(source)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "resolve_pcie_lock_key" in called, (
        "the park keys the lock with a second resolver; two co-located ranks "
        "that disagree about the card's name hold two files and exclude "
        f"nothing. calls seen: {sorted(called)}"
    )
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert "park_lock" not in code, "the second lock file is still opened"
    assert "fcntl" not in code, "the park still takes an flock of its own"


def test_no_second_flock_keys_the_same_physical_gpu():
    """Tree-wide sibling gate: exactly ONE flock is keyed on a GPU uuid.

    Grep-shaped on purpose -- the defect is not in any one file but in the
    EXISTENCE of a second call site keying the same physical link.  The other
    ``fcntl.flock`` users in the tree key on a ledger path, an IPC handle, a
    store shard or a snapshot, never on a card.
    """
    import subprocess

    root = os.path.dirname(inspect.getsourcefile(wu))
    srt = os.path.abspath(os.path.join(root, "..", ".."))
    out = subprocess.run(
        ["grep", "-rln", "fcntl.flock", srt],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    gpu_keyed = []
    for path in out:
        text = open(path, "r", encoding="utf-8").read()
        if "nvml_uuid" in text or "current_device_uuid" in text:
            gpu_keyed.append(os.path.relpath(path, srt))
    assert gpu_keyed == ["managers/weg2_memory_saver.py"], (
        "more than one flock call site keys a physical GPU; two files on one "
        f"link exclude nothing: {gpu_keyed}"
    )


def test_pcie_lock_is_not_reentrant_so_the_park_is_locked_exactly_once(tmp_path):
    """Why the park leg is NOT ALSO wrapped at the weight_updater call site.

    ``flock`` is held per OPEN FILE DESCRIPTION, so the same process taking the
    same lock through two ``open()`` calls does not recurse -- it contends with
    itself.  Wrapping ``_hibernate_park_weights`` in ``_weg2_pcie_lock`` on top
    of the park's own take (the round-3 finding's second edit) would therefore
    burn the whole deadline and end in ``Weg2PcieLockTimeout`` on every park.
    One take, inside ``park_weights_to_disk``, is the covered form.
    """
    from sglang.srt.managers.weg2_memory_saver import (
        Weg2PcieLockTimeout,
        pcie_transfer_lock,
    )

    with pcie_transfer_lock(
        nvml_uuid="GPU-reentrancy", lock_dir=str(tmp_path), timeout_s=0.2
    ):
        with pytest.raises(Weg2PcieLockTimeout):
            with pcie_transfer_lock(
                nvml_uuid="GPU-reentrancy", lock_dir=str(tmp_path), timeout_s=0.2
            ):
                pass


def test_release_rpc_leaves_the_hibernate_park_to_the_park_s_own_lock():
    """The park call must NOT be double-wrapped (see the reentrancy test)."""
    body = _tag_block(_func_ast("release_memory_occupation"), "GPU_MEMORY_TYPE_WEIGHTS")
    guards = _with_items_around(body, "_hibernate_park_weights")
    assert not any("_weg2_pcie_lock" in g for g in guards), (
        "the #89 park is wrapped in the PCIe lock AND takes it inside "
        "park_weights_to_disk; flock does not recurse, so every park would "
        "burn the deadline and refuse"
    )
