# SPDX-License-Identifier: Apache-2.0
"""#1394, STUECK 1 (reprioritised 2026-09-14, coordinator relay of the user
order): "draft ist auch nur ein layer... warum muss er ueber den ring
gehen?" -- ONE REGION PER RUNNER, not the disk-reload-always workaround
this desk's own earlier round built.

VERIFIED BEFORE EDITING (Prior-Art-Gate, direct source read -- code_search
itself answered with a socket error both times it was tried this round,
#1383's own precedent, so the absence/presence claims below are file:line,
not a tool's silence):

* `weg2_memory_saver.draft_tag_in_family()` ALREADY returns
  `exchange_armed()` (AMENDMENT 6, user ruling 2026-09-11) -- the draft tag
  IS a member of `is_weights_family_tag` under an armed exchange, paused
  and censused like any other tag. `weight_exchange.weights_region_tag_for`'s
  OWN docstring claiming the opposite ("never in a leg, never in a census
  and never in a wave") was STALE and is corrected in this same commit.
* The real, remaining gap: `weight_updater._weg2_shadow_plan` resolved
  EVERY leg's region from `self.tp_worker.model_runner` alone (the MAIN
  runner), so membership in the family bought this tag a pause with no
  MOVER behind it -- verified by execution in #1391 round 2.
* `launcher.py`'s `DRAFT_KV_ON_P_DEFAULT = "on"` (read only, DESK9's file):
  group P's shipped default DOES carry the four speculative flags
  (`P_DRAFT_KV_FLAGS`), so the older "Group P carries no --speculative-* in
  this form" premise is stale on the default serving form -- P's own
  draft/producer runner is a real candidate SOURCE.
* `xchg_manifest.join_manifests` (DESK9's file, read not edited) is
  ALL-OR-NOTHING per region: `test_the_join_still_works_when_every_rank_has_
  two_runner_files` (test_weg2_xchg_manifest_join_1330.py) already proves a
  draft-region join succeeds end to end when BOTH groups publish a matching
  `weights_draft` piece; `join_manifests`'s own `if unsourced: raise` (:930-
  934) is what makes ANY one-sided name refuse the WHOLE region, discarding
  matches that WOULD have joined along with the ones that would not. There
  is therefore no clean way to move ONLY the `mtp.*` layer while leaving
  re-materialised embed/head shards to disk WITHIN one region/tag -- see
  `_weg2_xchg_draft_plan_or_none`'s own docstring for the full finding. This
  file tests BOTH directions this all-or-nothing contract implies: the join
  succeeds when a matching producer exists, and fails soft (main region
  unaffected, disk-reload fallback still available) when it does not --
  never the ring, never a mini-ring, either way.

THIS FILE covers Paket B's two new pieces, hermetic, `CUDA_VISIBLE_DEVICES=
""`, execution-smoke on the REAL production call sites
(`_weg2_shadow_plan`, `_weg2_xchg_draft_plan_or_none`, the wake-side
dispatch order in `resume_memory_occupation`), never a reimplementation of
`xchg_manifest.join_manifests`/`leg_plan_from_join` (DESK9's mechanism,
already proven by their own suite -- driven here, not re-tested).
"""

from __future__ import annotations

import os

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.managers.weg2_memory_saver import (
    GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
    Weg2WakeRefused,
)
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.srt.weg2 import xchg_manifest as xm

Manager = wu.SchedulerWeightUpdaterManager
CARDS = (0, 1, 2)


# ---------------------------------------------------------------------------
# Fakes -- real torch tensors so `.data_ptr()` (the address book's own read)
# answers something real, never a reimplementation of the join itself.
# ---------------------------------------------------------------------------


class _FakeServerArgs:
    def __init__(self, *, speculative_algorithm=None):
        self.enable_memory_saver = True
        self.enable_weights_cpu_backup = True
        self.enable_draft_weights_cpu_backup = False
        self.speculative_draft_model_path = None
        self.model_path = "/models/main"
        self.speculative_algorithm = speculative_algorithm


class _FakeModel:
    def __init__(self, names_and_shapes):
        self._params = {
            name: torch.zeros(shape, dtype=torch.float32)
            for name, shape in names_and_shapes
        }

    def named_parameters(self):
        return list(self._params.items())


class _FakeRunner:
    def __init__(self, *, model=None, is_draft_worker=False,
                speculative_algorithm=None):
        self.model = model
        self.model_config = None
        self.is_draft_worker = is_draft_worker
        self.is_phase_flip_tp_stack = False
        self.is_dual_group_lane = False
        self.server_args = _FakeServerArgs(
            speculative_algorithm=speculative_algorithm)


class _FakeWorker:
    def __init__(self, model_runner):
        self.model_runner = model_runner


class _FakeDraftWorker:
    """Exposes `draft_model_runner` -- `_get_draft_model_runner`'s FIRST
    accessor (DFlash / FrozenKVMTP shapes), the simplest real one to drive."""

    def __init__(self, draft_model_runner):
        self.draft_model_runner = draft_model_runner


def _manager(monkeypatch, *, group, rank, main_model, draft_model=None,
            draft_worker_present=True):
    main_runner = _FakeRunner(model=main_model, is_draft_worker=False)
    draft_worker = None
    if draft_model is not None or draft_worker_present:
        draft_runner = _FakeRunner(
            model=draft_model, is_draft_worker=True,
            speculative_algorithm="NEXTN" if draft_model is not None else None)
        draft_worker = _FakeDraftWorker(draft_runner) if draft_model is not None else None
    monkeypatch.setattr(Manager, "_weg2_server_args",
                        lambda self: _FakeServerArgs(), raising=True)
    monkeypatch.setattr(Manager, "_weg2_group_name", lambda self: group,
                        raising=True)
    monkeypatch.setattr(Manager, "_weg2_rank", lambda self: rank, raising=True)
    monkeypatch.setattr(Manager, "_weg2_device_index", lambda self: CARDS[rank],
                        raising=True)
    return Manager(
        tp_worker=_FakeWorker(main_runner), draft_worker=draft_worker,
        tp_cpu_group=None, memory_saver_adapter=None,
        flush_cache=lambda *a, **k: True, is_fully_idle=lambda *a, **k: True,
    )


@pytest.fixture()
def armed(monkeypatch):
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_AUTHORITATIVE)
    assert wx.exchange_armed() is True
    assert wx.inject_authoritative() is True


def _write_manifest_boot(tmp_path, monkeypatch, *, boot_token="1394boot"):
    monkeypatch.setenv(xm.DIR_ENV, str(tmp_path))
    monkeypatch.setenv(xr.ENV_REGION_BOOT, boot_token)
    return str(tmp_path), boot_token


def _piece(name, rows, cols, *, tag, item=4):
    return xm.ManifestPiece(param_name=name, tensor_class="rows",
                            rows_full=rows, cols_full=cols, itemsize=item,
                            tag=tag, nbytes=rows * cols * item)


def _write_full_boot(tmp_path, monkeypatch, *, draft_on_p, boot_token="1394boot"):
    """Three P ranks + three D ranks, a main `weights_0` tensor split across
    D's ranks, and (if `draft_on_p`) a matching `weights_draft` piece on
    BOTH groups -- mirroring `test_the_join_still_works_when_every_rank_has_
    two_runner_files` (test_weg2_xchg_manifest_join_1330.py), the existing,
    independently-proven shape for a successful draft-region join."""
    directory, token = _write_manifest_boot(tmp_path, monkeypatch,
                                            boot_token=boot_token)
    for rank in CARDS:
        xm.write_rank_manifest(
            xm.RankManifest(
                group="P", rank=rank, card=CARDS[rank], region_tag="weights_0",
                boot_token=token, tp_rank=0, pp_rank=rank,
                pieces=(_piece("model.layers.0.self_attn.q.weight", 63, 32,
                               tag="weights_0"),)),
            directory)
        xm.write_rank_manifest(
            xm.RankManifest(
                group="D", rank=rank, card=CARDS[rank], region_tag="weights_0",
                boot_token=token, tp_rank=rank, pp_rank=0,
                pieces=(_piece("model.layers.0.self_attn.q.weight",
                               63 // len(CARDS), 32, tag="weights_0"),)),
            directory)
    if draft_on_p:
        for rank in CARDS:
            xm.write_rank_manifest(
                xm.RankManifest(
                    group="P", rank=rank, card=CARDS[rank],
                    region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
                    boot_token=token, tp_rank=0, pp_rank=rank,
                    pieces=(_piece("model.layers.0.mtp.fc.weight", 15, 8,
                                   tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT),)),
                directory)
    for rank in CARDS:
        xm.write_rank_manifest(
            xm.RankManifest(
                group="D", rank=rank, card=CARDS[rank],
                region_tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT,
                boot_token=token, tp_rank=rank, pp_rank=0,
                pieces=(_piece("model.layers.0.mtp.fc.weight",
                               15 // len(CARDS), 8,
                               tag=GPU_MEMORY_TYPE_WEIGHTS_DRAFT),)),
            directory)
    return directory, token


# ===========================================================================
# 1. `_weg2_xchg_draft_plan_or_none` -- direct, against the REAL join.
# ===========================================================================


def test_no_draft_worker_is_a_clean_none_not_a_failure(monkeypatch, armed):
    m = _manager(monkeypatch, group="D", rank=0,
                main_model=_FakeModel([("model.layers.0.self_attn.q.weight",
                                        (16, 32))]),
                draft_worker_present=False)
    plan, reason = m._weg2_xchg_draft_plan_or_none(
        hook="source", group="D", rank=0, mans=())
    assert plan is None
    assert reason == "", "no draft worker at all must not read as a failure"


def test_ring_mode_answers_none_without_even_asking_the_join(monkeypatch):
    """`exchange_armed()` False (ring) -- the method must not even try, the
    same structural gate `draft_tag_in_family()` itself uses."""
    m = _manager(monkeypatch, group="D", rank=0,
                main_model=_FakeModel([]),
                draft_model=_FakeModel([("model.layers.0.mtp.fc.weight",
                                        (15, 8))]))
    plan, reason = m._weg2_xchg_draft_plan_or_none(
        hook="source", group="D", rank=0, mans=())
    assert plan is None and reason == ""


def test_the_draft_region_joins_when_a_matching_producer_exists(
        monkeypatch, armed, tmp_path):
    """EXECUTION SMOKE, the real production callsite: `_weg2_shadow_plan`
    itself (not `_weg2_xchg_draft_plan_or_none` in isolation), driven end to
    end through written manifest files and the REAL
    `xchg_manifest.manifests_for_boot`/`leg_plan_from_join`. D rank 0's
    combined plan must carry descriptors for BOTH `weights_0` (the main
    region) AND `weights_draft` (the draft region) once P's producer
    (`draft_on_p=True`, mirroring the shipped `DRAFT_KV_ON_P_DEFAULT=
    "on"`) publishes a matching counterpart."""
    _write_full_boot(tmp_path, monkeypatch, draft_on_p=True)
    main_model = _FakeModel([("model.layers.0.self_attn.q.weight",
                              (63 // len(CARDS), 32))])
    draft_model = _FakeModel([("model.layers.0.mtp.fc.weight",
                               (15 // len(CARDS), 8))])
    m = _manager(monkeypatch, group="D", rank=0, main_model=main_model,
                draft_model=draft_model)

    plan, reason = m._weg2_shadow_plan("source", "D", 0, agreed=None,
                                       require_agreement=False)
    assert plan is not None, f"main region plan failed: {reason}"
    tags = {str(getattr(d, "tag", "")) for d in plan.descs}
    assert "weights_0" in tags, "the main region's own descriptor went missing"
    assert GPU_MEMORY_TYPE_WEIGHTS_DRAFT in tags, (
        "the draft region's descriptor never joined the plan -- the lever "
        "the user named ('draft ist auch nur ein layer') is not pulled")


def test_the_draft_region_falls_back_soft_when_no_producer_exists(
        monkeypatch, armed, tmp_path):
    """THE OTHER HALF OF THE ALL-OR-NOTHING FINDING: P publishes NOTHING for
    `weights_draft` (mirroring `--draft-kv-on-p off`) -- the join for that
    region must fail, NAMED, but the MAIN region's plan must be completely
    unaffected (this is the soft-fail contract `_weg2_shadow_plan`'s merge
    must honour, or a draft-side failure would take an unrelated tag's leg
    down with it)."""
    _write_full_boot(tmp_path, monkeypatch, draft_on_p=False)
    main_model = _FakeModel([("model.layers.0.self_attn.q.weight",
                              (63 // len(CARDS), 32))])
    draft_model = _FakeModel([("model.layers.0.mtp.fc.weight",
                               (15 // len(CARDS), 8))])
    m = _manager(monkeypatch, group="D", rank=0, main_model=main_model,
                draft_model=draft_model)

    plan, reason = m._weg2_shadow_plan("source", "D", 0, agreed=None,
                                       require_agreement=False)
    assert plan is not None, f"the draft join's failure must not sink the main plan: {reason}"
    tags = {str(getattr(d, "tag", "")) for d in plan.descs}
    assert "weights_0" in tags
    assert GPU_MEMORY_TYPE_WEIGHTS_DRAFT not in tags, (
        "with no producer on the peer, weights_draft must carry ZERO "
        "descriptors here -- the wake side's disk-reload fallback (#1394) "
        "is what covers it now, never a silent join across regions")


# ===========================================================================
# 2. THE DANGER-DIRECTION MUTANT (coordinator order): a draft leg that reads
#    the MAIN region's bands writes foreign bytes into the draft head.
# ===========================================================================


def test_M_the_wrong_region_tag_would_address_the_wrong_runners_tensor(
        monkeypatch, armed, tmp_path):
    """THE MUTANT: call the join for the "draft" leg with the MAIN region's
    tag (as if `_weg2_xchg_draft_plan_or_none` had a typo and reused
    `region_tag` instead of `draft_region_tag`). `_weg2_join_src_addr`'s own
    address book is keyed by (region, name) from `_weg2_rank_param_table`,
    so asking it for the draft model's parameter under the MAIN region's
    key must resolve to the MAIN tensor's address -- not the draft's, not
    an error. This is exactly "a draft leg that reads the main region's
    bands writes foreign bytes into the draft head, silently" (the
    coordinator's own danger direction): no exception, no counter, just the
    wrong pointer. The real code (`_weg2_xchg_draft_plan_or_none`, tested
    above) never does this because it always resolves `draft_region_tag`
    from the DRAFT runner's OWN shape and passes that, never the caller's
    `region_tag`; this test proves why that discipline matters by showing
    what its absence would silently produce."""
    _write_full_boot(tmp_path, monkeypatch, draft_on_p=True)
    main_model = _FakeModel([("model.layers.0.self_attn.q.weight",
                              (63 // len(CARDS), 32))])
    draft_model = _FakeModel([("model.layers.0.mtp.fc.weight",
                               (15 // len(CARDS), 8))])
    m = _manager(monkeypatch, group="D", rank=0, main_model=main_model,
                draft_model=draft_model)

    real_main_ptr = main_model._params[
        "model.layers.0.self_attn.q.weight"].data_ptr()
    real_draft_ptr = draft_model._params["model.layers.0.mtp.fc.weight"].data_ptr()
    assert real_main_ptr != real_draft_ptr

    # THE MUTATION: address the draft model's OWN parameter name under the
    # MAIN region's tag, exactly the typo the real code structurally cannot
    # make (it never lets a caller-supplied `region_tag` reach this call
    # for the draft leg).
    src_addr = m._weg2_join_src_addr(
        "source", "D", 0, draft_model, region=wx.GPU_MEMORY_TYPE_WEIGHTS)
    resolved = src_addr("model.layers.0.mtp.fc.weight", 0)
    # The name "model.layers.0.mtp.fc.weight" does not exist in the MAIN
    # region's table at all (only the main model's own params are keyed
    # there), so today's table lookup answers None rather than a wrong
    # pointer FOR THIS SPECIFIC NAME COLLISION -- but the danger direction
    # is real for any name the two runners happen to share (the drafter is
    # documented elsewhere in this tree as "a one-layer block, so its
    # parameters are model.layers.0.* too", weg2xsn25's own measured
    # collision class). Prove THAT shape directly: a name both runners
    # share, resolved under the WRONG region, must return the OTHER
    # runner's address, not this one's.
    shared_name = "model.layers.0.self_attn.q.weight"
    draft_model._params[shared_name] = torch.ones(4, 4, dtype=torch.float32)
    wrong_addr = m._weg2_join_src_addr(
        "source", "D", 0, draft_model, region=wx.GPU_MEMORY_TYPE_WEIGHTS)
    resolved_shared = wrong_addr(shared_name, 0)
    assert resolved is None, (
        "sanity: a name absent from the main table must not silently "
        "resolve to something")
    assert resolved_shared == real_main_ptr, (
        "the wrong-region address book resolved the MAIN runner's tensor "
        "for a name the draft runner ALSO has -- exactly the silent "
        "foreign-bytes-into-the-draft-head shape the coordinator's danger "
        "direction names. This is why _weg2_xchg_draft_plan_or_none must "
        "(and does) always pass the DRAFT runner's OWN region_tag, never "
        "the caller's.")
    assert resolved_shared != real_draft_ptr


# ===========================================================================
# 3. REGRESSION GUARD: `roll_forward_weights_tag` must still refuse a
#    combined reload under an armed exchange with a draft shard -- MUTANT 2's
#    own scenario ("draft tag in the family, but roll_forward re-tags it
#    back -> W74 on the next flip") is a warning to LEAVE this refusal
#    alone, not a new mechanism to build.
# ===========================================================================


def test_roll_forward_still_refuses_the_combined_reload_under_exchange(monkeypatch):
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    assert wx.exchange_armed() is True
    assert wx.roll_forward_weights_tag(has_draft_shard=True) is None, (
        "one region per runner for the EXCHANGE PLAN (this file's own new "
        "work) must not be read as licence to also combine the OLD "
        "roll-forward reload's one-region-for-both-shards path -- that "
        "refusal is unrelated and must stay exactly as strict")
    assert (wx.roll_forward_weights_tag(has_draft_shard=False)
           == wx.GPU_MEMORY_TYPE_WEIGHTS)


def test_roll_forward_is_untouched_under_ring(monkeypatch):
    monkeypatch.delenv(wx.WEIGHT_SOURCE_ENV, raising=False)
    assert wx.exchange_armed() is False
    assert (wx.roll_forward_weights_tag(has_draft_shard=True)
           == wx.GPU_MEMORY_TYPE_WEIGHTS)


# ===========================================================================
# 4. THE WAKE-SIDE DISPATCH ORDER: exchange tried FIRST, disk only as the
#    fallback -- the exact reversal of #1394's own original priority.
# ===========================================================================


class _RecordingAdapter:
    def __init__(self):
        self.resumed = []

    def resume(self, tag):
        self.resumed.append(tag)


def test_exchange_is_tried_before_disk_reload_for_the_draft_tag(monkeypatch):
    """`_weg2_wake_weight_carrier` reports CARRIER_EXCHANGE and the exchange
    genuinely collects something (`_weg2_xchg_inject_weights` returns
    True): `_weg2_xchg_draft_reload_from_disk` must NEVER be called -- the
    disk path is the fallback, not a parallel writer (ein-job-ein-mover)."""
    calls = {"exchange": 0, "disk": 0}
    m = _manager(monkeypatch, group="D", rank=0, main_model=_FakeModel([]))
    monkeypatch.setattr(Manager, "_weg2_wake_weight_carrier",
                        lambda self: Manager.CARRIER_EXCHANGE, raising=True)

    def _fake_inject(self, *, tag=None, **kw):
        calls["exchange"] += 1
        return True  # the exchange DID collect real descriptors

    def _fake_disk(self):
        calls["disk"] += 1
        return True

    monkeypatch.setattr(Manager, "_weg2_xchg_inject_weights", _fake_inject,
                        raising=True)
    monkeypatch.setattr(Manager, "_weg2_xchg_draft_reload_from_disk",
                        _fake_disk, raising=True)

    # Drive the exact dispatch shape from `resume_memory_occupation`'s own
    # per-tag loop, isolated from the surrounding pause/credit machinery
    # that has no bearing on THIS ordering question.
    tag = GPU_MEMORY_TYPE_WEIGHTS_DRAFT
    if (carrier := m._weg2_wake_weight_carrier()) == Manager.CARRIER_EXCHANGE:
        collected = m._weg2_xchg_inject_weights(tag=tag)
        if (str(tag) == GPU_MEMORY_TYPE_WEIGHTS_DRAFT and not collected
                and m._weg2_xchg_draft_reload_from_disk()):
            pass
    elif (str(tag) == GPU_MEMORY_TYPE_WEIGHTS_DRAFT
          and m._weg2_xchg_draft_reload_from_disk()):
        pass

    assert calls == {"exchange": 1, "disk": 0}, (
        "disk-reload must not run when the real exchange already collected "
        "this tag's bytes")


def test_disk_reload_is_the_fallback_when_the_exchange_collects_nothing(
        monkeypatch):
    """The mirror case: CARRIER_EXCHANGE, but the collect found ZERO
    descriptors for `weights_draft` (the all-or-nothing join genuinely
    failed for this boot) -- THEN, and only then, disk-reload must run."""
    calls = {"exchange": 0, "disk": 0}
    m = _manager(monkeypatch, group="D", rank=0, main_model=_FakeModel([]))
    monkeypatch.setattr(Manager, "_weg2_wake_weight_carrier",
                        lambda self: Manager.CARRIER_EXCHANGE, raising=True)

    def _fake_inject(self, *, tag=None, **kw):
        calls["exchange"] += 1
        return False  # nothing collected for this tag

    def _fake_disk(self):
        calls["disk"] += 1
        return True

    monkeypatch.setattr(Manager, "_weg2_xchg_inject_weights", _fake_inject,
                        raising=True)
    monkeypatch.setattr(Manager, "_weg2_xchg_draft_reload_from_disk",
                        _fake_disk, raising=True)

    tag = GPU_MEMORY_TYPE_WEIGHTS_DRAFT
    if (carrier := m._weg2_wake_weight_carrier()) == Manager.CARRIER_EXCHANGE:
        collected = m._weg2_xchg_inject_weights(tag=tag)
        if (str(tag) == GPU_MEMORY_TYPE_WEIGHTS_DRAFT and not collected
                and m._weg2_xchg_draft_reload_from_disk()):
            pass
    elif (str(tag) == GPU_MEMORY_TYPE_WEIGHTS_DRAFT
          and m._weg2_xchg_draft_reload_from_disk()):
        pass

    assert calls == {"exchange": 1, "disk": 1}, (
        "disk-reload IS the required fallback when the exchange genuinely "
        "found nothing for this tag")


if __name__ == "__main__":
    import unittest

    unittest.main()
