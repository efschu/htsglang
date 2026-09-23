# SPDX-License-Identifier: Apache-2.0
"""fnFL2x10 (23.09.): the first D->P flip ran through all sixteen expert tags
and died on ``weights_draft``:

  D TP0  W68 lane src=0 dst=2 tag='weights_draft': 1 of 33 descs have no
         address on the side this rank owns (model.embed_tokens.weight)
  D TP1/TP2  W106 ... zero descriptors ... disk-reload undefined on
         'compressed-tensors'

Both draft manifests were written at LOAD and named a BF16
``model.embed_tokens.weight`` (1212.5 MiB). After load the drafter changes:
D's ``set_embed_and_head`` hands it the target's PACKED embed and head, P's
last stage rebuilds the vocab packed (``load_resident_embedding``). At the
flip that BF16 table existed on neither side. D TP1/TP2 are Form A's meta
shadows: zero bytes of the draft tag, nothing to release, nothing to reload.
"""

from __future__ import annotations

import os
import types

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.managers.weg2_memory_saver import (
    GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
)
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import xchg_manifest as xm

Manager = wu.SchedulerWeightUpdaterManager
DRAFT = GPU_MEMORY_TYPE_WEIGHTS_DRAFT


def _piece(name, rows, cols, item=2):
    return xm.ManifestPiece(param_name=name, tensor_class="rows",
                            rows_full=rows, cols_full=cols, itemsize=item,
                            tag=DRAFT, nbytes=rows * cols * item)


def _man(pieces):
    return xm.RankManifest(group="D", rank=0, card=0, region_tag=DRAFT,
                           boot_token="x10", tp_rank=0, pp_rank=0,
                           pieces=tuple(pieces))


LOAD_TIME = [_piece("fc_embedding.weight", 16, 16),
             _piece("model.embed_tokens.weight", 248, 32)]
LIVE = [_piece("fc_embedding.weight", 16, 16),
        _piece("model.embed_tokens.weight_packed", 248, 8, item=4),
        _piece("model.embed_tokens.weight_scale", 248, 2),
        _piece("model.embed_tokens.weight_shape", 2, 1, item=8)]


def test_x10_the_live_rewrite_replaces_the_stale_load_time_manifest(tmp_path):
    """The same writer publishes its post-share drafter over its own
    load-time file; the plain write keeps weg2xsn22's collision ratchet."""
    xm.write_rank_manifest(_man(LOAD_TIME), str(tmp_path))
    with pytest.raises(wx.Weg2XchgPlanDisagree):
        xm.write_rank_manifest(_man(LIVE), str(tmp_path))
    xm.write_rank_manifest(_man(LIVE), str(tmp_path), supersede=True)
    (back,) = xm.load_manifests(str(tmp_path))
    names = {p.param_name for p in back.pieces}
    assert "model.embed_tokens.weight" not in names
    assert "model.embed_tokens.weight_packed" in names


def test_the_rewrite_touches_only_the_armed_draft_region(monkeypatch):
    calls = []
    monkeypatch.setattr(wx, "_write_placement_manifest",
                        lambda model, **kw: calls.append(kw), raising=True)
    ident = dict(rank=0, tp_rank=0, pp_rank=0, tp_size=3)
    monkeypatch.delenv(wx.WEIGHT_SOURCE_ENV, raising=False)
    wx.rewrite_draft_manifest_live(object(), region_tag=DRAFT, **ident)
    assert calls == [], "no exchange armed: nothing is published at all"
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    wx.rewrite_draft_manifest_live(object(), region_tag="weights", **ident)
    assert calls == [], "the main region's parameters are final at load"
    wx.rewrite_draft_manifest_live(object(), region_tag=DRAFT, **ident)
    assert len(calls) == 1 and calls[0]["supersede"] is True


# ---------------------------------------------------------------------------
# The quantized head: shared PARAMETER BY PARAMETER, skipped whole or not at
# all.
# ---------------------------------------------------------------------------


def _head(**params):
    mod = torch.nn.Module()
    for name, tensor in params.items():
        mod.register_parameter(name, torch.nn.Parameter(tensor,
                                                        requires_grad=False))
    return mod


def _manager_with_heads(monkeypatch, draft_head, target_head):
    target_model = types.SimpleNamespace(lm_head=target_head)
    draft_model = types.SimpleNamespace(lm_head=draft_head)
    runner = types.SimpleNamespace(model=draft_model)
    monkeypatch.setattr(wu, "_get_draft_model_runner", lambda dw: runner,
                        raising=True)
    return Manager(
        tp_worker=types.SimpleNamespace(
            model_runner=types.SimpleNamespace(model=target_model)),
        draft_worker=object(), tp_cpu_group=None, memory_saver_adapter=None,
        flush_cache=lambda *a, **k: True, is_fully_idle=lambda *a, **k: True,
    )


def test_a_packed_head_shared_param_by_param_is_skipped_whole(monkeypatch):
    """x10's head: weight_packed/weight_scale and no `weight`, so the
    single-tensor proof answered no-weight and the head stayed in the join."""
    target = _head(weight_packed=torch.zeros(4, 8, dtype=torch.int32),
                   weight_scale=torch.zeros(4, 2))
    m = _manager_with_heads(monkeypatch, draft_head=target, target_head=target)
    assert m._weg2_draft_lm_head_is_target_share().startswith("no-weight")
    names, proof = m._weg2_draft_lm_head_target_shares()
    assert names == {"lm_head.weight_packed", "lm_head.weight_scale"}
    assert proof.startswith("MEASURED-SHARED")


def test_a_partly_own_head_is_never_skipped(monkeypatch):
    """The danger direction: one own parameter means the head needs a source
    -- dropping it would leave that parameter's bytes undefined at wake."""
    target = _head(weight_packed=torch.zeros(4, 8, dtype=torch.int32),
                   weight_scale=torch.zeros(4, 2))
    draft = _head(weight_packed=target.weight_packed,
                  weight_scale=torch.zeros(4, 2))
    m = _manager_with_heads(monkeypatch, draft_head=draft, target_head=target)
    names, proof = m._weg2_draft_lm_head_target_shares()
    assert names == frozenset()
    assert "NOT-SHARED" in proof and "weight_scale" in proof


# ---------------------------------------------------------------------------
# The meta shadow: zero bytes of the draft tag on this rank.
# ---------------------------------------------------------------------------


class _Adapter:
    def __init__(self, nbytes):
        self._nbytes = nbytes

    def tag_bytes(self, tag):
        return self._nbytes


class _DraftWorker:
    def __init__(self):
        self.calls = []

    def update_weights_from_disk(self, req):
        self.calls.append(req)
        return True, ""


def _shadow_manager(monkeypatch, *, tag_bytes):
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_AUTHORITATIVE)
    monkeypatch.setenv(wx.WEIGHTS_CPU_BACKUP_ENV, wx.WEIGHTS_CPU_BACKUP_AUTO)
    monkeypatch.setattr(Manager, "_weg2_draft_checkpoint_quantization",
                        lambda self: "compressed-tensors", raising=True)
    return Manager(
        tp_worker=types.SimpleNamespace(model_runner=None),
        draft_worker=_DraftWorker(), tp_cpu_group=None,
        memory_saver_adapter=(None if tag_bytes is None
                              else _Adapter(tag_bytes)),
        flush_cache=lambda *a, **k: True, is_fully_idle=lambda *a, **k: True,
    )


def test_x10_a_meta_shadow_is_not_a_w106_gap(monkeypatch):
    m = _shadow_manager(monkeypatch, tag_bytes=0)
    assert m._weg2_xchg_wake_source_gap(
        DRAFT, cdescs_present=False, resident_bytes=0) is None


@pytest.mark.parametrize("resident", [None, 1382 << 20])
def test_a_real_or_unmeasurable_holder_still_refuses(monkeypatch, resident):
    """The exemption keys on a MEASURED zero; bytes held, or a count nobody
    could read, keep the quantized-checkpoint refusal."""
    m = _shadow_manager(monkeypatch, tag_bytes=resident)
    gap = m._weg2_xchg_wake_source_gap(
        DRAFT, cdescs_present=False, resident_bytes=resident)
    assert gap is not None and "compressed-tensors" in gap


def test_x10_the_shadow_never_reaches_the_disk_reload(monkeypatch):
    """At P->D the shadow's collect is empty; without the zero-byte check
    the reload path raised W4 on the quantized checkpoint."""
    m = _shadow_manager(monkeypatch, tag_bytes=0)
    assert m._weg2_xchg_draft_reload_from_disk() is False
    assert m.draft_worker.calls == []
