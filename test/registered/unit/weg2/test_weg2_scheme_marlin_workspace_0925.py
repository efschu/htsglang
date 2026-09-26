"""The Marlin lock workspace that lives on a compressed-tensors SCHEME object
(27B line, 2026-09-25; the production draft Qwen3.8-27B-DFlash2-W8-lued).

``compressed_tensors_wNa16.CompressedTensorsWNA16._repack_to_marlin`` keeps its
lock array on the scheme -- ``self.workspace = marlin_make_workspace(device)`` --
born in the runner's weights tag pool (``weights_draft`` for the draft). That tag
is released at every sleep and resumed with fresh, recycled pages and no CPU
backup, the exchange writes only planned parameters and buffers, and the kernel
needs every lock at zero when a multi-block slice starts (``barrier_acquire`` in
``jit_kernel/csrc/gemm/marlin/marlin_template.h`` spins until ``lock ==
slice_idx``). ``zero_local_scratch`` -- the resume-side memset adopted from the NF
line -- reaches ``layer.workspace`` and parameters only; a scheme's workspace is
one object deeper and kept the pages' residue.

RED on RC3 5917ceba34: the scheme's workspace is neither walked nor zeroed.
Desk, CPU tensors, the real scheme class (constructed without its config).
"""

from __future__ import annotations

import inspect
import os

import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.layers.quantization.compressed_tensors.schemes import (  # noqa: E402
    compressed_tensors_wNa16 as W,
)
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402

RESIDUE = 0x5A5A5A5A


def _scheme_with_residue(n=8):
    scheme = W.CompressedTensorsWNA16.__new__(W.CompressedTensorsWNA16)
    scheme.workspace = torch.full((n,), RESIDUE, dtype=torch.int32)
    return scheme


def _draft_like_model():
    """layers.0.{qkv_proj, down_proj}: two Marlin linears whose workspace sits on
    the scheme, one plain layer.workspace (the FP8/NVFP4 form) beside them."""
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    layer = model.layers[0]
    for name in ("qkv_proj", "down_proj"):
        lin = torch.nn.Module()
        lin.weight_packed = torch.nn.Parameter(
            torch.ones(2, 2, dtype=torch.int32), requires_grad=False
        )
        lin.scheme = _scheme_with_residue()
        setattr(layer, name, lin)
    fp8 = torch.nn.Module()
    fp8.workspace = torch.full((4,), RESIDUE, dtype=torch.int32)
    layer.o_proj = fp8
    return model


def test_the_scheme_workspace_is_zeroed_at_wake():
    """RED on RC3: the scheme's lock array keeps the residue."""
    model = _draft_like_model()
    done = wx.zero_local_scratch(model)
    for name in ("qkv_proj", "down_proj"):
        ws = getattr(model.layers[0], name).scheme.workspace
        assert int(torch.count_nonzero(ws)) == 0, name
        assert f"layers.0.{name}.scheme.workspace" in done
    assert "layers.0.o_proj.workspace" in done  # the module-level form, as before


def test_the_residue_found_is_counted_before_it_is_zeroed():
    model = _draft_like_model()
    residue = []
    wx.zero_local_scratch(model, residue=residue)
    assert residue == [2 * 8 + 4]
    residue = []
    wx.zero_local_scratch(model, residue=residue)
    assert residue == [0]


def test_the_coverage_walk_books_it_as_local_scratch():
    """The walk sees it (ATTRIBUTE, leaf ``workspace``), so the cover line books
    it as local_scratch instead of leaving its bytes unattributed."""
    names = {t.name: t for t in wx.walk_live_tensors(_draft_like_model())}
    t = names["layers.0.qkv_proj.scheme.workspace"]
    assert t.kind == wx.ATTRIBUTE and wx.is_local_scratch(t.name)


def test_only_a_leaf_named_workspace_is_touched_one_level_deep():
    model = torch.nn.Module()
    holder = type("Holder", (), {})()
    holder.cache = torch.full((3,), 7, dtype=torch.int32)  # not scratch: kept
    holder.inner = type("Inner", (), {})()
    holder.inner.workspace = torch.full((3,), 7, dtype=torch.int32)  # two deep: kept
    model.holder = holder
    model.names = ["workspace"]  # a container is not descended into
    assert wx.zero_local_scratch(model) == []
    assert int(torch.count_nonzero(holder.cache)) == 3
    assert int(torch.count_nonzero(holder.inner.workspace)) == 3
    assert not [t for t in wx.walk_live_tensors(model) if "holder" in t.name]


def test_a_workspace_shared_by_two_schemes_is_zeroed_once():
    model = _draft_like_model()
    shared = model.layers[0].qkv_proj.scheme.workspace
    model.layers[0].down_proj.scheme.workspace = shared
    done = wx.zero_local_scratch(model)
    assert sum(1 for n in done if n.endswith("scheme.workspace")) == 1


def test_the_scheme_is_where_the_wna16_repack_puts_the_lock_array():
    """The premise, pinned: if the repack ever moves it onto the layer, the
    module-level walk already covers it and this test says so."""
    src = inspect.getsource(W.CompressedTensorsWNA16._repack_to_marlin)
    assert "self.workspace = marlin_make_workspace(device)" in src


def test_the_resume_hook_reports_the_residue():
    from sglang.srt.managers.scheduler_components import weight_updater as WU

    src = inspect.getsource(WU)
    # UNIFY S2: the helper forwards the counter; the target's count always,
    # the draft's only when no H25 unpark is in flight (host sync)
    assert "zero_local_scratch(_m, residue=residue)" in src
    assert "self._weg2_scratch_residue = [0, \"all\"]" in src
    assert "if getattr(self, \"_weg2_draft_unpark_ph0\", None) is None:" in src
    assert "residue_nonzero=" in src
