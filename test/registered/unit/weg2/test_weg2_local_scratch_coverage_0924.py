"""27B line: Marlin's lock workspace under the weg2 flip -- the NF local-scratch
mechanism, adopted (NF 54266526b8 + 1a87bef818 + ba5847a7c0; test shape after
NF's test_weg2_local_scratch_coverage_0921.py). Desk, no GPU.

Every Marlin linear carries ``layer.workspace`` (``marlin_make_workspace``:
sms x int32, zero at rest, 170 on the 5090 / 68 on a 3080): FP8 through
``prepare_fp8_layer_for_marlin`` (Fp8LinearMethod and, since the #31340 port,
ModelOptFp8LinearMethod), NVFP4 through ``prepare_nvfp4_layer_for_marlin``.
It is in no checkpoint and therefore in no exchange plan, and it lives under
the weights tag, so

* the W84 coverage books it as ``local_scratch`` (counted and named on the
  COVER line, not refused) -- whether the walk meets it as a Parameter or as a
  plain ATTRIBUTE (the kind is not what it IS);
* the waking side re-zeroes it once the weights family is complete, on BOTH
  runners of the rank (target + draft), because the resume maps recycled pages.

This replaces the FP8-only private registry of fdade8572f
(SGLANG_FP8_MARLIN_PRIVATE_WORKSPACE): that one moved the workspace OFF the
module, and ``should_apply_lm_head_quant_method`` requires ``workspace`` ON a
Marlin NVFP4 lm_head -- pinned below.
"""

from __future__ import annotations

import inspect
import os
import types

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as WU  # noqa: E402
from sglang.srt.weg2 import weight_exchange as WX  # noqa: E402


def _marlin_like_model(as_parameter: bool):
    """layers.0.mlp: a weight the plan names + Marlin's workspace (not planned)."""
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    lin = torch.nn.Linear(8, 8, bias=False)
    model.layers[0].mlp = lin
    ws = torch.full((68,), 7, dtype=torch.int32)
    if as_parameter:
        lin.workspace = torch.nn.Parameter(ws, requires_grad=False)
    else:
        lin.workspace = ws  # a PLAIN attribute, like prepare_*_layer_for_marlin's
    return model


def _coverage(model, extra_plan=None):
    live = WX.walk_live_tensors(model)
    w = next(t for t in live if t.name == "layers.0.mlp.weight")
    plan = {w.tag: {"layers.0.mlp.weight": int(w.nbytes)}}
    for name, nbytes in (extra_plan or {}).items():
        plan[w.tag][name] = nbytes
    rows = WX.build_coverage(model, rank=0, planned_bytes_by_tag=plan,
                             tag_bytes=lambda tag: 0, mode="exchange")
    return rows[w.tag]


# -- the predicate and the memset -------------------------------------------------


def test_the_leaf_name_decides_not_a_substring():
    assert WX.is_local_scratch("model.layers.3.mlp.gate_up_proj.workspace")
    assert WX.is_local_scratch("lm_head.workspace")
    assert WX.is_local_scratch("workspace")
    assert not WX.is_local_scratch("model.workspace.weight")
    assert not WX.is_local_scratch("model.layers.0.mlp.gate_up_proj.weight_scale")


def test_it_zeroes_every_workspace_and_nothing_else():
    for as_parameter in (False, True):
        m = _marlin_like_model(as_parameter)
        m.layers[0].mlp.weight.data.fill_(3.0)
        done = WX.zero_local_scratch(m)
        assert done == ["layers.0.mlp.workspace"], (as_parameter, done)
        assert int(m.layers[0].mlp.workspace.abs().sum()) == 0
        assert float(m.layers[0].mlp.weight.sum()) == 3.0 * 64


def test_an_empty_or_absent_workspace_is_not_reported():
    m = torch.nn.Module()
    m.workspace = torch.zeros((0,), dtype=torch.int32)
    assert WX.zero_local_scratch(m) == []
    assert WX.zero_local_scratch(torch.nn.Linear(2, 2)) == []


def test_one_storage_reached_by_both_walks_is_zeroed_once():
    m = torch.nn.Module()
    m.a = torch.nn.Module()
    m.a.workspace = torch.nn.Parameter(torch.ones(4, dtype=torch.int32), requires_grad=False)
    m.b = torch.nn.Module()
    m.b.workspace = m.a.workspace.data  # same storage as a plain attribute elsewhere
    assert WX.zero_local_scratch(m) == ["a.workspace"]


# -- W84: booked, not refused --------------------------------------------------------


def test_the_coverage_books_the_workspace_as_local_scratch_attribute_and_parameter():
    for as_parameter in (False, True):
        row = _coverage(_marlin_like_model(as_parameter))
        assert row.local_scratch == ("layers.0.mlp.workspace",), as_parameter
        assert row.uncovered == () and row.ok, as_parameter
        line = row.cover_line()
        assert "local_scratch=1 " in line and "scratch_reason=" in line
        # appended after the exempt fields, before the saver answer
        assert line.index("exempt=") < line.index("local_scratch=") < line.index("tms_answered=")


def test_a_real_unplanned_tensor_still_refuses():
    model = _marlin_like_model(False)
    model.layers[0].mlp.stray = torch.ones(16)  # bytes, no plan, not scratch
    row = _coverage(model)
    assert [t.name for t in row.uncovered] == ["layers.0.mlp.stray"]
    assert not row.ok
    assert row.local_scratch == ("layers.0.mlp.workspace",)


def test_a_line_without_scratch_prints_a_zero_and_no_reason():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    model.layers[0].mlp = torch.nn.Linear(8, 8, bias=False)
    row = _coverage(model)
    line = row.cover_line()
    assert "local_scratch=0 " in line and "scratch_reason=" not in line
    assert row.ok


# -- the waking side ----------------------------------------------------------------


def test_the_family_complete_wake_zeroes_both_runners_after_the_static_import():
    src = inspect.getsource(WU.SchedulerWeightUpdaterManager.resume_memory_occupation)
    assert "from sglang.srt.weg2.weight_exchange import zero_local_scratch" in src
    assert "WEG2-RESUME local-scratch zeroed=%d" in src
    assert "for _m in self._weg2_wake_models():" in src
    # 27B placement: inside the family_complete block, after the static-state
    # import (the whole family is mapped), never per partial wake chunk
    fc = src.index("if family_complete:")
    assert fc < src.index("_import_static_state(") < src.index("zero_local_scratch(_m)")
    assert src.index('_weg2_ph("leg_collects")') < fc
    # the FP8-only registry of fdade8572f is gone
    full = inspect.getsource(WU)
    assert "_weg2_zero_fp8_marlin_workspaces" not in full
    assert "WEG2-WAKE FP8-MARLIN" not in full


def test_wake_models_are_target_and_distinct_draft():
    target, draft = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    fake = types.SimpleNamespace(
        tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(model=target)))
    fake._weg2_model_for_group = lambda g: draft if g == "D" else target
    got = WU.SchedulerWeightUpdaterManager._weg2_wake_models(fake)
    assert len(got) == 2 and got[0] is target and got[1] is draft
    fake._weg2_model_for_group = lambda g: target  # no distinct draft
    assert WU.SchedulerWeightUpdaterManager._weg2_wake_models(fake) == [target]
    fake._weg2_model_for_group = lambda g: None
    assert WU.SchedulerWeightUpdaterManager._weg2_wake_models(fake) == [target]


def test_the_fp8_private_registry_is_gone():
    from sglang.srt import environ
    from sglang.srt.layers.quantization import fp8, marlin_utils_fp8

    assert not hasattr(environ.Envs, "SGLANG_FP8_MARLIN_PRIVATE_WORKSPACE")
    assert not hasattr(marlin_utils_fp8, "fp8_marlin_workspace_off_module")
    assert not hasattr(marlin_utils_fp8, "zero_fp8_marlin_workspaces")
    assert "private_ws" not in inspect.getsource(fp8.Fp8LinearMethod.apply)
