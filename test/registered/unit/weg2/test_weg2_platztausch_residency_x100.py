"""fnFL2x100 (23.09. 22:12Z): the first NextFlash form with more experts on the
cards than the proven 10 % form died in its first D->P sleep leg.

Form: FR_P 0.45/0.95/1.0, FR_D 0.006/0.62/0.52 (x87, same code, FR_P
0.26/0.45/0.39 + FR_D 0.06/0.44/0.365, flipped cleanly). The chain:

1. P stage 2 (layers 40-47) held all 512 experts. ``plan_load_time_staging``
   builds no Platztausch buffer at ``R >= E``, so the stage kept the plain
   ``mlp.experts.w13_weight_packed`` [512, ...] stack and published it under
   that name, while every D rank published ``...experts.weg2_experts_*``.
2. The join dropped D's 32 buffers (3.69 GB) as "destination-only names ...
   they come back from the disk-reload fallback" -- a fallback this INT4
   checkpoint does not have -- and never looked at P's plain stacks.
3. D TP1's source plan lost weights_14/weights_15 whole: W106 on weights_14.
   D TP2 passed W106 only through the replicated ``mlp.gate.weight``.
4. Every other rank waited out its own 120 s budget (P PP1 VRAM credit for
   weights_10, D TP0 BAR1 'free' on p0, P PP0 lane mode on p2) for a leg that
   had died at 22:12:34 -- W29 at 22:14:36.

Each case below turns red on the pre-fix tree (5a96de48be).
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.layers.moe import expert_map as em
from sglang.srt.layers.moe.expert_offload import (
    resident_slot_count,
    scratch_slot_count,
)
from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.managers.weg2_memory_saver import (
    VramCredit,
    Weg2XchgWakeSourceGapRefused,
)
from sglang.srt.weg2 import leg_abort as la
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.srt.weg2 import xchg_manifest as xm
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="stage-a-weg2-unit")

Manager = wu.SchedulerWeightUpdaterManager

RATIOS = "183,137,168"
X100 = ("0.45,0.95,1.0", "0.006,0.62,0.52")
X87 = ("0.26,0.45,0.39", "0.06,0.44,0.365")
P_STAGE_LAYERS = [29, 11, 8]


# ---------------------------------------------------------------------------
# 1. The dry run refuses the form whose Karte a rank will not build (W120).
# ---------------------------------------------------------------------------


def _publish(monkeypatch, tmp_path, form):
    from sglang.srt.planner import pp_cut
    from sglang.srt.weg2 import launcher

    monkeypatch.setattr(pp_cut, "checkpoint_weight_terms",
                        lambda model: SimpleNamespace(num_experts=512))
    fr_p, fr_d = form
    ns = SimpleNamespace(
        extra_d=f"--rank-moe-ratio {RATIOS} --rank-moe-resident-fraction {fr_d}",
        extra_p=f"--rank-moe-resident-fraction {fr_p}",
        pp_cut_expert_device_fraction=fr_p, tag="t", model="m")
    lines = []
    path = launcher.publish_expert_map(
        ns, "m", str(tmp_path), lines.append,
        p_stage_layers=P_STAGE_LAYERS, chunk_layers=3)
    return path, lines


def test_x100_form_is_refused_by_the_dry_run_with_stage_layers_and_tags(
        monkeypatch, tmp_path):
    """Pre-fix: the Karte was written and the boot loaded for 5 minutes."""
    from sglang.srt.weg2.launcher import Weg2LaunchRefused

    with pytest.raises(Weg2LaunchRefused) as exc:
        _publish(monkeypatch, tmp_path, X100)
    msg = str(exc.value)
    assert "W120" in msg
    assert "P-Stufe 2: 512 von 512" in msg
    assert "Layer 40-47" in msg
    assert "weights_13,weights_14,weights_15" in msg
    assert "0.996" in msg
    assert "D-Rang" not in msg, "FR_D 0.62/0.52 leave every D rank a buffer"


def test_the_proven_10pct_form_still_publishes(monkeypatch, tmp_path):
    """Negative branch: the predicate must not degrade to always-refuse."""
    path, _lines = _publish(monkeypatch, tmp_path, X87)
    assert path and os.path.exists(path)


def test_the_largest_buffered_fraction_agrees_with_the_ranks_own_scratch_rule():
    """Derived boundary: a rank builds its buffer iff E - R >= 2, and the
    ``max_fraction`` the refusal recommends must be one the rank's own
    ``scratch_slot_count`` accepts while the next resident row is refused --
    for P stages (E=512) and for a D rank (E = span + pad = 145)."""
    base = dict(total=512, ratios=[183, 137, 168],
                p_layer_stage=[0] * 29 + [1] * 11 + [2] * 8, pad_tp=1)
    ok = em.build_nested(fr_pp=[0.2, 0.2, 0.996], fr_tp=[0.1, 0.1, 0.1], **base)
    assert em.unbuilt_platztausch_buffers(ok) == ()
    bad = em.build_nested(fr_pp=[0.2, 0.2, 0.998], fr_tp=[0.1, 0.986, 0.1], **base)
    unbuilt = {(u.phase, u.index): u for u in em.unbuilt_platztausch_buffers(bad)}
    assert set(unbuilt) == {("P", 2)}, "D rank 1 at 0.986 keeps 2 scratch rows"
    bad_d = em.build_nested(fr_pp=[0.2, 0.2, 0.2], fr_tp=[0.1, 0.99, 0.1], **base)
    (u,) = em.unbuilt_platztausch_buffers(bad_d)
    assert (u.phase, u.index, u.experts) == ("D", 1, 145)
    for e, top in ((512, unbuilt[("P", 2)].max_fraction), (145, u.max_fraction)):
        r = resident_slot_count(e, top)
        scratch_slot_count(r, e)  # the rank accepts the recommended fraction
        with pytest.raises(ValueError):
            scratch_slot_count(r + 1, e)


def test_a_rank_launched_past_the_dry_run_refuses_at_load_not_at_the_flip(
        monkeypatch):
    """The same W120 at the repack door, for a boot that bypassed the
    launcher. Pre-fix: ``frac >= 1.0`` returned silently and P stage 2 kept
    the plain stack. The NEXTN draft (fully resident by design, its block is
    ``model.layers.0``) and a boot without a Version-2 Karte are untouched."""
    from sglang.srt.layers.moe import expert_offload, expert_store

    fr_p, fr_d = X100
    karte = em.build_nested(
        total=512, ratios=[183, 137, 168],
        fr_pp=[float(x) for x in fr_p.split(",")],
        fr_tp=[float(x) for x in fr_d.split(",")],
        p_layer_stage=[0] * 29 + [1] * 11 + [2] * 8, pad_tp=1)
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")

    def _layer(layer_id, excluded):
        return SimpleNamespace(num_local_experts=512, num_experts=512,
                               layer_id=layer_id, moe_tp_rank=0,
                               _expert_offload_fraction=1.0,
                               _moe_offload_excluded=excluded)

    monkeypatch.setattr(expert_store, "expert_map", lambda: karte)
    with pytest.raises(RuntimeError, match="W120 Weg2PlatztauschBufferUnbuilt: layer 42"):
        expert_offload.presplit_expert_offload_after_repack(_layer(42, False))
    expert_offload.presplit_expert_offload_after_repack(_layer(0, True))
    monkeypatch.setattr(expert_store, "expert_map", lambda: None)
    expert_offload.presplit_expert_offload_after_repack(_layer(42, False))


# ---------------------------------------------------------------------------
# 2. The join refuses a buffer only one group publishes (W74), rank-uniformly.
# ---------------------------------------------------------------------------

_UNIT = 160  # rows of one expert in w13_weight_packed


def _piece(name, rows, tag):
    return xm.ManifestPiece(param_name=name, tensor_class=name.rsplit(".", 1)[1],
                            rows_full=rows, cols_full=2560, itemsize=4, tag=tag,
                            nbytes=rows * 2560 * 4)


def _mans(p_stage1_name):
    dense = "model.layers.0.input_layernorm.weight"
    buf = "model.layers.{}.mlp.experts.weg2_experts_w13_weight_packed"
    d = []
    for r, slots in enumerate((1, 2, 3)):
        pieces = [_piece(buf.format(0), slots * _UNIT, "weights_0"),
                  _piece(buf.format(1), slots * _UNIT, "weights_0")]
        if r == 0:
            pieces.append(_piece(dense, 1, "weights_0"))
        d.append(xm.RankManifest(group="D", rank=r, card=r, region_tag="weights",
                                 boot_token="b", pieces=tuple(pieces), tp_rank=r))
    p = [xm.RankManifest(group="P", rank=0, card=0, region_tag="weights",
                         boot_token="b", pp_rank=0, pieces=(
                             _piece(dense, 1, "weights_0"),
                             _piece(buf.format(0), 6 * _UNIT, "weights_0"))),
         xm.RankManifest(group="P", rank=1, card=1, region_tag="weights",
                         boot_token="b", pp_rank=1, pieces=(
                             _piece(p_stage1_name, 6 * _UNIT, "weights_0"),))]
    return d + p


def test_a_buffer_the_other_group_publishes_as_the_plain_stack_is_refused():
    """Pre-fix: the join logged 'destination-only-names=...' and planned the
    rest; D's layer-1 buffer then had no depositor target and P's plain stack
    no source, in BOTH directions."""
    xm.clear_join_memo()
    with pytest.raises(wx.Weg2XchgSourceMissing) as exc:
        xm.join_manifests(_mans("model.layers.1.mlp.experts.w13_weight_packed"))
    msg = str(exc.value)
    assert "W74" in msg
    assert "model.layers.1.mlp.experts.weg2_experts_w13_weight_packed" in msg
    assert "'model.layers.1.mlp.experts.w13_weight_packed' instead" in msg
    assert "D rank 0, D rank 1, D rank 2" in msg


def test_the_same_buffer_on_both_sides_still_joins():
    xm.clear_join_memo()
    join = xm.join_manifests(
        _mans("model.layers.1.mlp.experts.weg2_experts_w13_weight_packed"))
    got = {t.param_name: t.tp_widths for t in join.tensors}
    assert got["model.layers.1.mlp.experts.weg2_experts_w13_weight_packed"] == (
        _UNIT, 2 * _UNIT, 3 * _UNIT)


# ---------------------------------------------------------------------------
# 3. A sleeping rank with NO plan does not release its bytes silently (W106).
# ---------------------------------------------------------------------------


class _FakeWorker:
    def __init__(self):
        self.model_runner = SimpleNamespace(model=None, model_config=None)


def _manager(monkeypatch, *, group, rank):
    monkeypatch.setattr(Manager, "_weg2_group_name", lambda self: group)
    monkeypatch.setattr(Manager, "_weg2_rank", lambda self: rank)
    monkeypatch.setattr(Manager, "_weg2_device_index", lambda self: rank)
    return Manager(tp_worker=_FakeWorker(), draft_worker=None, tp_cpu_group=None,
                   memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
                   is_fully_idle=lambda *a, **k: True)


def test_a_refused_join_on_the_sleep_side_is_a_w106_not_a_skipped_deposit(
        monkeypatch):
    """Pre-fix: ``plan is None`` logged WEG2-XCHG-DEPOSIT-SKIPPED and returned,
    and the pause right after released the tag with nobody holding it."""
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_AUTHORITATIVE)
    monkeypatch.setenv(wx.WEIGHTS_CPU_BACKUP_ENV, wx.WEIGHTS_CPU_BACKUP_OFF)
    monkeypatch.setenv(xr.ENV_REGION_BOOT, "x100desk")
    m = _manager(monkeypatch, group="D", rank=1)
    monkeypatch.setattr(
        Manager, "_weg2_shadow_plan",
        lambda self, hook, g, r, agreed=None, require_agreement=None:
            (None, "unjoinable: W74 Weg2XchgSourceMissing: 96 Platztausch ..."))
    monkeypatch.setattr(Manager, "_weg2_tag_resident_bytes",
                        lambda self, tag: 1086324736)
    with pytest.raises(Weg2XchgWakeSourceGapRefused) as exc:
        m._weg2_xchg_deposit_before_sleep(flip_index=1, tag="weights_14")
    msg = str(exc.value)
    assert "W106" in msg and "tag=weights_14" in msg
    assert "NO exchange plan" in msg and "W74" in msg


# ---------------------------------------------------------------------------
# 4. A dead leg is named to the waiters at once (W121), not after 120 s.
# ---------------------------------------------------------------------------


def test_the_x100_interleaving_every_waiter_learns_of_the_dead_leg(
        monkeypatch, tmp_path):
    """D TP1 fails its sleep leg of flip 1; P PP1 (the co-located waker on
    card 1, other group) is in its liveness-probed lane wait and in its VRAM
    credit wait. Pre-fix: liveness answered True (TP1's process lived on in
    its fence) and the credit wait had no way to learn anything -- both rode
    out the full budget."""
    monkeypatch.setattr(la, "DEFAULT_ROOT", str(tmp_path))
    monkeypatch.setenv(xr.ENV_REGION_BOOT, "x100desk")
    sleeper = _manager(monkeypatch, group="D", rank=1)
    sleeper._weg2_flip_index_now = 1
    monkeypatch.setattr(Manager, "_weg2_fence_is_armed", lambda self: False)
    sleeper._weg2_leg_failed(
        "release_memory_occupation FAILED on this rank",
        Weg2XchgWakeSourceGapRefused("W106 ... tag=weights_14 ..."))

    waker = _manager(monkeypatch, group="P", rank=1)
    waker._weg2_flip_index_now = 1
    assert waker._weg2_cocard_peer_alive() is False
    credit = VramCredit("GPU-x100-card1", credit_dir=str(tmp_path / "credit"))
    t0 = time.monotonic()
    with pytest.raises(la.Weg2FlipPeerLegAborted) as exc:
        credit.wait_for(4210 * 1024 * 1024, budget_s=120.0, tag="weights_10",
                        free_bytes_now=0,
                        abort_reader=waker._weg2_foreign_leg_aborts)
    assert time.monotonic() - t0 < 5.0
    assert "W121" in str(exc.value) and "D rank 1" in str(exc.value)
    assert "weights_14" in str(exc.value)


def test_an_abort_of_another_flip_or_of_this_rank_itself_stops_nothing(
        monkeypatch, tmp_path):
    """Negative branch: the key is (boot, flip) and the reader skips itself --
    a previous flip's corpse or this rank's own post must not end a wait."""
    monkeypatch.setattr(la, "DEFAULT_ROOT", str(tmp_path))
    monkeypatch.setenv(xr.ENV_REGION_BOOT, "x100desk")
    la.post(boot_nonce="x100desk", flip=0, group="D", rank=1, reason="old flip")
    la.post(boot_nonce="x100desk", flip=1, group="P", rank=1, reason="myself")
    la.post(boot_nonce="other-boot", flip=1, group="D", rank=0, reason="old boot")
    waker = _manager(monkeypatch, group="P", rank=1)
    waker._weg2_flip_index_now = 1
    assert waker._weg2_foreign_leg_aborts() == ""
