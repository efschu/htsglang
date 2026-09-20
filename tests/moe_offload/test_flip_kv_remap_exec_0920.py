# SPDX-License-Identifier: Apache-2.0
"""R3: the VMM remap executor, against stub ops."""
from __future__ import annotations

import pytest

from sglang.srt.flip_kv_remap import DISPOSITION_REMAP, plan_kv_remap
from sglang.srt.flip_kv_remap_exec import (
    CudaVmmOps,
    FakeVmmOps,
    LayerHandle,
    execute_kv_remap,
)
from sglang.srt.flip_nextflash_plan import Weg2FlipKvRelayInfeasible


def _handles(plan, only_remap=True, nbytes=1 << 20):
    out = []
    for m in plan.moves:
        if only_remap and m.disposition != DISPOSITION_REMAP:
            continue
        out.append(
            LayerHandle(
                layer_index=m.layer_index,
                handle=object(),
                src_va=0x1000 + m.layer_index * 0x100,
                dst_va=0x9000 + m.layer_index * 0x100,
                nbytes=nbytes,
            )
        )
    return out


def _mapped_src(ops, handles):
    """Pre-map the sources, as the live PP0 pool would have them."""
    for h in handles:
        ops.map(h.src_va, h.nbytes, h.handle)
        ops.set_access(h.src_va, h.nbytes, 0)
    ops.calls.clear()


# --------------------------------------------------------------------------
# The happy path
# --------------------------------------------------------------------------
def test_the_seven_device_local_layers_are_remapped():
    plan = plan_kv_remap()
    hs = _handles(plan)
    ops = FakeVmmOps()
    _mapped_src(ops, hs)
    res = execute_kv_remap(plan, hs, ops)
    assert len(res.remapped_layers) == 7
    assert len(res.legged_layers) == 5
    assert res.in_flight_layers == ()
    # every destination is mapped AND accessible
    for h in hs:
        assert ops.mapped.get(h.dst_va) is h.handle
        assert h.dst_va in ops.accessible
        assert h.src_va not in ops.mapped


def test_the_order_is_unmap_then_map_then_set_access():
    """Map-first looks safer and is the version that doubles residency --
    the exact OOM the remap exists to avoid."""
    plan = plan_kv_remap()
    hs = _handles(plan)
    ops = FakeVmmOps()
    _mapped_src(ops, hs)
    execute_kv_remap(plan, hs, ops)
    verbs = [c for c, _ in ops.calls]
    assert verbs[:3] == ["unmap", "map", "set_access"]
    # and never a map before its own unmap
    for i in range(0, len(ops.calls), 3):
        u, m, s = ops.calls[i : i + 3]
        assert u[0] == "unmap" and m[0] == "map" and s[0] == "set_access"


def test_the_stub_enforces_the_driver_invariant_a_handle_maps_once():
    """A handle mapped at two addresses at once is the inversion; the stub
    must catch it, or the order test above proves nothing."""
    ops = FakeVmmOps()
    h = object()
    ops.map(0x100, 4096, h)
    with pytest.raises(RuntimeError) as exc:
        ops.map(0x200, 4096, h)
    assert "cannot back two virtual ranges" in str(exc.value)


def test_set_access_is_not_optional():
    """A freshly mapped range is not accessible until cuMemSetAccess --
    skipping it faults at first touch."""
    plan = plan_kv_remap()
    hs = _handles(plan)
    ops = FakeVmmOps()
    _mapped_src(ops, hs)
    execute_kv_remap(plan, hs, ops)
    assert len(ops.accessible) == 7


def test_the_result_line_names_remapped_and_legged_separately():
    plan = plan_kv_remap()
    hs = _handles(plan)
    ops = FakeVmmOps()
    _mapped_src(ops, hs)
    line = execute_kv_remap(plan, hs, ops).line()
    assert "remapped=7" in line and "legged=5" in line


# --------------------------------------------------------------------------
# The refusals
# --------------------------------------------------------------------------
def test_a_missing_handle_is_a_hole_not_a_smaller_remap():
    plan = plan_kv_remap()
    hs = _handles(plan)[:-1]
    ops = FakeVmmOps()
    _mapped_src(ops, hs)
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        execute_kv_remap(plan, hs, ops)
    msg = str(exc.value)
    assert "no physical handle" in msg
    assert "reads as zeros" in msg


def test_a_handle_for_an_unscheduled_layer_is_refused():
    plan = plan_kv_remap()
    hs = _handles(plan) + [LayerHandle(99, object(), 0x1, 0x2, 4096)]
    ops = FakeVmmOps()
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        execute_kv_remap(plan, hs, ops)
    assert "does not move" in str(exc.value)


def test_src_equal_dst_is_a_noop_wearing_a_remaps_name():
    plan = plan_kv_remap()
    hs = _handles(plan)
    hs[0] = LayerHandle(hs[0].layer_index, hs[0].handle, 0x500, 0x500, 4096)
    ops = FakeVmmOps()
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        execute_kv_remap(plan, hs, ops)
    assert "no-op wearing a remap" in str(exc.value)


def test_a_failure_names_the_layers_in_the_window():
    """Between unmap and map the handle is owned by neither pool. A fault
    there holes the KV pool SILENTLY -- unmapped KV reads as zeros."""
    plan = plan_kv_remap()
    hs = _handles(plan)
    ops = FakeVmmOps(fail_on_map=[hs[2].dst_va])
    _mapped_src(ops, hs)
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        execute_kv_remap(plan, hs, ops)
    msg = str(exc.value)
    assert f"[{hs[2].layer_index}] are IN THE WINDOW" in msg
    assert "do NOT roll back blindly" in msg
    assert "second bug on top" in msg
    # the two that completed are named too
    assert "are complete" in msg


def test_an_unavailable_device_layer_refuses_instead_of_copying():
    class Dead(FakeVmmOps):
        def available(self):
            return False

    plan = plan_kv_remap()
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        execute_kv_remap(plan, _handles(plan), Dead())
    assert "S6-REMAP-STATT-ALLOKATION" in str(exc.value)


# --------------------------------------------------------------------------
# The real ops: refuse by name, never a silent copy
# --------------------------------------------------------------------------
def test_the_cuda_ops_refuse_by_name_without_bindings():
    ops = CudaVmmOps()
    if ops.available():
        pytest.skip("driver bindings present in this process")
    with pytest.raises(Weg2FlipKvRelayInfeasible) as exc:
        ops.unmap(0x1000, 4096)
    msg = str(exc.value)
    assert "no CUDA driver bindings" in msg
    assert "0.24 GiB free under extend" in msg


def test_the_cuda_ops_never_degrade_to_a_copy():
    """A remap that quietly became a copy is the 1.99 GB transient this seam
    exists to avoid, and would surface as an OOM with no line saying why."""
    import inspect

    from sglang.srt import flip_kv_remap_exec as mod

    src = inspect.getsource(mod.CudaVmmOps)
    assert "copy" in src  # it TALKS about the copy...
    assert "fallback" not in src.replace("silent fallback", "")  # ...but never does one
