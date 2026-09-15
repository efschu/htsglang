# SPDX-License-Identifier: Apache-2.0
"""#1378 xsn34 root, second half: the draft's ONE-SIDED lm_head leaves the
join BY NAME, on a MEASURED target-share, and nothing else does.

MEASURED on weg2xsn34 (boot log, D rank 0, hook=source):
  W74 Weg2XchgSourceMissing: 1 of 20 tensors held by D have no counterpart
  in P's manifest (first: 'lm_head.weight')
P's drafter has no lm_head parameter AT ALL (P census: 19 draft tensors, no
lm_head; P rank 1: "Parameter lm_head.weight not found in params_dict")
because #1259(b) builds P's producer head under lm_head_from_target() and
shares the co-located TARGET's table in (set_lm_head_from_target --
qwen3_5_mtp.py:351 ``self.lm_head = target_lm_head``; eagle_worker_v2:648/
659 on D's NEXTN path). D's own 808.75 MiB x 3 table is the orphaned
build-time allocation; the bytes the draft's logits actually read are the
TARGET's lm_head -- which the MAIN region's leg already moves (tag=weights,
both sides hold it).

THE FIX IS ENG: the exclusion fires ONLY for ``lm_head.weight`` in the draft
region, ONLY when the rank MEASURES the share (data_ptr identity of draft
head vs target head), logged with both pointers. Any OTHER missing
counterpart still refuses W74; a share that is NOT proven refuses W74 too
(form (ii): the named refusal is infinitely better than undefined logits).

RED-FIRST: against the pre-fix tree the measured-share case raised W74 (the
wall that ended weg2xsn34's draft leg before its wake).
"""

from __future__ import annotations

import os
import types

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.managers.weg2_memory_saver import GPU_MEMORY_TYPE_WEIGHTS_DRAFT
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import xchg_manifest as xm
from sglang.srt.weg2.weight_exchange import Weg2XchgSourceMissing

Manager = wu.SchedulerWeightUpdaterManager
CARDS = (0, 1, 2)
DRAFT = GPU_MEMORY_TYPE_WEIGHTS_DRAFT


class _FakeServerArgs:
    def __init__(self):
        self.enable_memory_saver = True
        self.enable_weights_cpu_backup = True
        self.enable_draft_weights_cpu_backup = False
        self.speculative_draft_model_path = None
        self.model_path = "/models/main"
        self.speculative_algorithm = "NEXTN"


class _FakeModel:
    def __init__(self, names_and_shapes):
        self._params = {
            name: torch.zeros(shape, dtype=torch.float32)
            for name, shape in names_and_shapes
        }

    def named_parameters(self):
        return list(self._params.items())


class _FakeRunner:
    def __init__(self, *, model=None, is_draft_worker=False):
        self.model = model
        self.model_config = None
        self.is_draft_worker = is_draft_worker
        self.is_phase_flip_tp_stack = False
        self.is_dual_group_lane = False
        self.server_args = _FakeServerArgs()


class _FakeWorker:
    def __init__(self, model_runner):
        self.model_runner = model_runner


class _FakeDraftWorker:
    def __init__(self, draft_model_runner):
        self.draft_model_runner = draft_model_runner


def _manager(monkeypatch, *, group, rank, main_model, draft_model):
    main_runner = _FakeRunner(model=main_model, is_draft_worker=False)
    draft_runner = _FakeRunner(model=draft_model, is_draft_worker=True)
    monkeypatch.setattr(Manager, "_weg2_server_args",
                        lambda self: _FakeServerArgs(), raising=True)
    monkeypatch.setattr(Manager, "_weg2_group_name", lambda self: group,
                        raising=True)
    monkeypatch.setattr(Manager, "_weg2_rank", lambda self: rank,
                        raising=True)
    monkeypatch.setattr(Manager, "_weg2_device_index",
                        lambda self: CARDS[rank], raising=True)
    return Manager(
        tp_worker=_FakeWorker(main_runner),
        draft_worker=_FakeDraftWorker(draft_runner),
        tp_cpu_group=None, memory_saver_adapter=None,
        flush_cache=lambda *a, **k: True, is_fully_idle=lambda *a, **k: True,
    )


def _write_xsn34_shape_boot(monkeypatch, *, extra_missing=()):
    """The xsn34 shape on the REAL group geometry (3 P + 3 D): D's draft
    carries fc + lm_head, P's draft carries fc only -- lm_head has no
    counterpart on P (P's drafter shares the target's head)."""
    manifests = []
    for rank in CARDS:
        manifests.append(xm.RankManifest(
            group="P", rank=rank, card=CARDS[rank], region_tag=DRAFT,
            boot_token="xsn34", tp_rank=0, pp_rank=rank,
            pieces=(_piece("model.layers.0.mtp.fc.weight", 15, 8,
                           tag=DRAFT),)))
    for rank in CARDS:
        manifests.append(xm.RankManifest(
            group="D", rank=rank, card=CARDS[rank], region_tag=DRAFT,
            boot_token="xsn34", tp_rank=rank, pp_rank=0,
            pieces=tuple(
                [_piece("model.layers.0.mtp.fc.weight",
                        15 // len(CARDS), 8, tag=DRAFT),
                 _piece("lm_head.weight", 8, 32, tag=DRAFT)]
                + [_piece(n, 4, 4, tag=DRAFT) for n in extra_missing])))
    monkeypatch.setattr(xm, "load_manifests", lambda *_a, **_k: manifests,
                        raising=True)


def _piece(name, rows, cols, *, tag, item=4):
    return xm.ManifestPiece(param_name=name, tensor_class="rows",
                            rows_full=rows, cols_full=cols, itemsize=item,
                            tag=tag, nbytes=rows * cols * item)


def _models_with_shared_head(*, shared: bool):
    """The main model carries an lm_head; the draft model's head IS the
    target's module when ``shared`` (the measured share), else its own
    distinct tensor (the NOT-SHARED danger shape)."""
    main_model = _FakeModel([
        ("model.layers.0.self_attn.q.weight", (63, 32)),
        ("lm_head.weight", (8, 32)),
    ])
    main_model.lm_head = types.SimpleNamespace(
        weight=main_model._params["lm_head.weight"])
    draft_model = _FakeModel([
        ("model.layers.0.mtp.fc.weight", (15 // len(CARDS), 8)),
        ("model.layers.0.mtp.fc.weight_scale", (4, 4)),
        ("lm_head.weight", (8, 32)),
    ])
    if shared:
        draft_model.lm_head = main_model.lm_head
    else:
        draft_model.lm_head = types.SimpleNamespace(
            weight=torch.zeros(8, 32, dtype=torch.float32))
    return main_model, draft_model


@pytest.fixture()
def armed(monkeypatch):
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_AUTHORITATIVE)


def _plan(monkeypatch, *, shared, extra_missing=()):
    _write_xsn34_shape_boot(monkeypatch, extra_missing=extra_missing)
    main_model, draft_model = _models_with_shared_head(shared=shared)
    m = _manager(monkeypatch, group="D", rank=0,
                 main_model=main_model, draft_model=draft_model)
    return m._weg2_xchg_draft_plan_or_none(
        hook="source", group="D", rank=0, mans=xm.load_manifests())


def test_measured_share_moves_the_other_nineteen(armed, monkeypatch):
    """THE xsn34 WALL, desk-reproduced and fixed: lm_head has no counterpart,
    the share is MEASURED (data_ptr identity), so the join proceeds for the
    tensors that DO have counterparts instead of discarding the whole
    region."""
    plan, reason = _plan(monkeypatch, shared=True)
    assert plan is not None, f"the join must proceed: {reason}"
    names = {str(getattr(d, "param_name", "")) or str(getattr(d, "name", ""))
             for d in plan.descs}
    assert "lm_head.weight" not in names, \
        "the one-sided head must not be planned"
    assert any("fc" in n for n in names), \
        f"the joinable tensors must move: {names}"


def _plan_with_log(monkeypatch, caplog, **kw):
    import logging
    with caplog.at_level(logging.INFO, logger="sglang.srt.weg2.xchg_manifest"):
        return _plan(monkeypatch, **kw)


def test_M_another_missing_tensor_is_excluded_by_name_never_planned(
        armed, monkeypatch, caplog):
    """THE CONTRACT SINCE xsn53 (join_manifests, 'DIE RICHTUNG'): a name the
    destination does not hold is NOT a W74 -- the plan covers the
    INTERSECTION and the join names the exclusion with its byte count
    (``WEG2-XCHG-PLAN ... destination-only-names=``). Measured on the metal
    (weg2xsn87-89): D's own lm_head is exactly such a name every leg, and
    the flip is MATCH x12 with GEN-SMOKE MATCH. What must never happen is
    the danger direction this mutant was written for: the tensor planned
    anyway (a desc with no destination) or dropped SILENTLY."""
    plan, reason = _plan_with_log(
        monkeypatch, caplog, shared=True,
        extra_missing=("model.layers.0.mtp.fc.weight_scale",))
    assert plan is not None, reason
    assert all(d.param_name != "model.layers.0.mtp.fc.weight_scale"
               for d in plan.descs), "excluded, never planned"
    assert any("destination-only-names" in r.getMessage()
               and "fc.weight_scale" in r.getMessage()
               for r in caplog.records), \
        f"the exclusion must be NAMED: {[r.getMessage()[:120] for r in caplog.records]}"


def test_unproven_share_leaves_the_head_out_by_name(armed, monkeypatch, caplog):
    """Form (ii) under the xsn53 contract: a draft head that is NOT the
    target's module is not skipped by the share proof, so it reaches the
    join -- and the join leaves it out BY NAME (no counterpart on the peer),
    it is never planned against a missing destination. The wake side's
    W106 gap check / disk-reload path (#1394) is where an unsourced draft
    tensor becomes a refusal."""
    plan, reason = _plan_with_log(monkeypatch, caplog, shared=False)
    assert plan is not None, reason
    assert all(d.param_name != "lm_head.weight" for d in plan.descs)
    assert any("destination-only-names" in r.getMessage()
               and "lm_head.weight" in r.getMessage()
               for r in caplog.records), \
        f"the exclusion must be NAMED: {[r.getMessage()[:120] for r in caplog.records]}"


if __name__ == "__main__":
    import unittest
    unittest.main()
