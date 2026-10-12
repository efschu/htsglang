"""G13 (NF-GGUF, stage A of 12.10.2026): the D boot died in the first CUDA-graph warm-up forward, all three ranks,
``KeyError: 87 / 73 / 84`` in ``MoEExpertOffloadCache._pool_layout`` (``host_row.append(int(pool_index[e]) ...)``).

ROOT.  The environment carried ``SGLANG_MOE_COLD_TIER_SHM=1`` (the NF profile sets it) and the card probe gave a link
ratio, so ``FusedMoE._gguf_cold_shard_context`` returned a #394 context.  The G4 GGUF door
(``gguf_presplit.presplit_gguf_param``) planned with it: ``plan.spill_ids`` shrank to this rank's share of its own cold
experts (76 / 13 / 36) and the rest became ``plan.delegated_ids`` (105 / 58 / 56).  That door builds NO shared cold
tier (the streaming door does, ``_gguf_cold_tier_owner``); it wrote only ``spill_ids`` into the store and "released the
delegated experts without a copy".  ``_moe_offload_store_index`` therefore named only the owned share, and the pool
layout asked it for the first non-resident expert no tier held: local 87 (TP0), 73 (TP1), 84 (TP2).  The safetensors /
store path never has the gap: its Marlin door passes no cold shard (``refuse_cold_shard_at_repack_door``), so the map is
built over ALL cold experts.

THE FIX (two layers).  (1) The G4 door plans without a cold shard (``gguf_presplit.door_cold_shard``) -- every
non-resident expert gets a store row / pool row, ranks decide from the environment only -- and refuses at load a plan or
index with a gap (``assert_cold_tier_complete``).  (2) ``_pool_layout`` itself names a remaining gap (expert ids, tier)
instead of ``KeyError``.  Nothing is masked with ``.get()`` or a default.

These tests drive the REAL ``FusedMoE.materialize_gguf_weights`` on CPU tensors (the G4 harness), the REAL
``MoEExpertOffloadCache`` and the REAL ``_pool_layout``; the first reproduces the death of the boot on the unfixed tree
(RED on 8d5b142727), and the last checks the 12.10. numbers (87 / 73 / 84) from the real Karte of that boot.
"""

from __future__ import annotations

import json
import logging
import os
import types

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# the G4 harness: toy NF boot (12 experts, P one unsharded stage, D two expert-dim ranks with a trailing pad, nested Karte)
from test_gguf_g4_platztausch_store_1009 import (  # noqa: E402,F401  (boot is a fixture)
    FR_D,
    ROW_SHAPES,
    TOTAL,
    _bank,
    _expert_bytes,
    _gguf_layer,
    _stage,
    _store_row,
    boot,
)

from sglang.srt.layers.moe import expert_offload as eo  # noqa: E402
from sglang.srt.layers.moe import expert_store as es  # noqa: E402
from sglang.srt.layers.moe import gguf_presplit as gp  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=30, suite="stage-a-weg2-unit")

#: rank 0 : rank 1 = 1 : 3 -> of the two cold experts of each toy D rank, ONE is delegated on both ranks
HOST_RATIO = "1,3"
D_RANKS = (0, 1)


@pytest.fixture
def cold_tier_on(monkeypatch):
    """What the NF profile + the card probe do on the rig: the shared-tier switch and a non-equal link ratio."""
    monkeypatch.setenv("SGLANG_MOE_COLD_TIER_SHM", "1")
    monkeypatch.setenv("SGLANG_MOE_HOST_SHARD_RATIO", HOST_RATIO)
    monkeypatch.setattr(gp, "_COLD_SHARD_NOTE_LOGGED", False, raising=False)


def _cold_local(layer):
    """Local ids of ALL the layer's non-resident experts: everything the door did not make resident.

    Deliberately NOT the frozen layout's spill list: on the unfixed tree that list had already lost the delegated
    experts, which is the bug -- a test that took its expectation from it could not see the hole."""
    resident, _spill = layer._moe_offload_frozen_layout
    res = {int(e) for e in resident}
    return [e for e in range(int(layer._moe_offload_full_experts)) if e not in res], sorted(res)


def _cache(layer):
    """The real offload cache over a staged layer, the way ``install`` builds it (CPU: only the maps are read)."""
    layer.moe_runner_config = types.SimpleNamespace(routed_scaling_factor=1.0)
    return eo.MoEExpertOffloadCache(layer, layer._expert_offload_fraction)


# ---------------------------------------------------------------------------------------------------------------
# 1. the death of the boot, reproduced: a host-shard ratio under the store door
# ---------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rank", D_RANKS)
def test_the_pool_layout_of_a_staged_gguf_d_rank_has_a_row_for_every_cold_expert(boot, cold_tier_on, rank):
    """RED on 8d5b142727: ``KeyError`` out of ``_pool_layout`` -- the stage-A death -- for the delegated cold expert."""
    boot.group("D")
    layer = _stage(_gguf_layer(1, rank=rank))
    cold, resident = _cold_local(layer)
    assert len(cold) == 2 and not set(cold) & set(resident)
    cache = _cache(layer)
    hot_slot_of, host_row = cache._pool_layout()
    assert len(host_row) == cache.num_local_experts
    for e in range(cache.num_local_experts):
        if e in resident:
            assert host_row[e] == -1 and e in hot_slot_of
        else:
            assert host_row[e] >= 0, f"cold expert {e} has no host row"
    # one row per cold expert, all distinct: the map is a bijection onto the cold set
    assert sorted(h for h in host_row if h >= 0) == sorted(set(h for h in host_row if h >= 0))
    assert len([h for h in host_row if h >= 0]) == len(cold)


@pytest.mark.parametrize("rank", D_RANKS)
def test_the_store_index_names_every_cold_expert_and_the_store_holds_its_bytes(boot, cold_tier_on, rank):
    boot.group("D")
    layer = _stage(_gguf_layer(1, rank=rank))
    cold, _res = _cold_local(layer)
    assert sorted(layer._moe_offload_store_index) == cold
    assert not getattr(layer, "_moe_offload_delegated_experts", None)
    lo, _hi = layer._gguf_expert_range
    for attr in gp.GGUF_EXPERT_ATTRS:
        for local in cold:
            slot = layer._moe_offload_store_index[local]
            assert torch.equal(_store_row(boot, 1, attr, slot), _expert_bytes(1, attr, lo + local)), (attr, local)


def test_every_rank_decides_the_same_way_from_the_environment_alone(boot, cold_tier_on):
    """Ranks never disagree: the plan of each rank has no delegated experts, whatever its ratio share would have been."""
    boot.group("D")
    plans = []
    for rank in D_RANKS:
        layer = _stage(_gguf_layer(2, rank=rank))
        plans.append(layer._gguf_presplit_plan[1])
    assert [p.delegated_ids for p in plans] == [(), ()]
    assert [len(p.spill_ids) for p in plans] == [2, 2]


def test_a_private_pool_door_keeps_every_cold_expert_too(monkeypatch, cold_tier_on):
    """The seat-only / Karte-only form (no store) has the same hole: the delegated rows were dropped uncopied."""
    from sglang.srt.weg2 import d_seat_vram as sv

    monkeypatch.delenv(es.STORE_DIR_ENV, raising=False)
    monkeypatch.delenv(es.EXPERT_MAP_ENV, raising=False)
    monkeypatch.setenv("SGLANG_WEG2_D_SEAT_EXPERT_ROWS", "2")
    monkeypatch.setattr(es, "expert_map", lambda: None)
    monkeypatch.delenv("SGLANG_WEG2_GROUP", raising=False)
    monkeypatch.setattr(sv, "presplit_seat_rows", lambda layer: 2)
    monkeypatch.setattr(
        sv, "seat_expert_buffer",
        lambda *, rows, extra, tail, dtype, device, in_tag_pool=True, name="", spans=None, granule=None:
        torch.zeros((rows + extra,) + tuple(tail), dtype=dtype),
    )
    layer = _stage(_gguf_layer(1, rank=1))
    cold, _res = _cold_local(layer)
    _bank_t, spill = layer._moe_offload_presplit["w13_qweight"]
    assert spill.shape[0] == len(cold) == 2
    cache = _cache(layer)
    _hot, host_row = cache._pool_layout()
    assert all(host_row[e] >= 0 for e in cold)


def test_without_the_cold_tier_switch_nothing_changes_and_nothing_is_logged(boot, monkeypatch, caplog):
    monkeypatch.delenv("SGLANG_MOE_COLD_TIER_SHM", raising=False)
    monkeypatch.delenv("SGLANG_MOE_HOST_SHARD_RATIO", raising=False)
    monkeypatch.setattr(gp, "_COLD_SHARD_NOTE_LOGGED", False, raising=False)
    boot.group("D")
    with caplog.at_level(logging.WARNING):
        layer = _stage(_gguf_layer(0, rank=0))
    assert not [r for r in caplog.records if "G13" in r.getMessage()]
    cold, _ = _cold_local(layer)
    assert sorted(layer._moe_offload_store_index) == cold


def test_the_operator_is_told_once_that_the_share_was_not_applied(boot, cold_tier_on, caplog):
    boot.group("D")
    with caplog.at_level(logging.WARNING):
        for lid in (0, 1, 2):
            _stage(_gguf_layer(lid, rank=0))
    lines = [r.getMessage() for r in caplog.records if "G13" in r.getMessage()]
    assert len(lines) == 1, lines
    assert "NOT applied" in lines[0] and "#394" in lines[0]


def test_a_malformed_host_shard_ratio_is_still_a_hard_error(boot, cold_tier_on, monkeypatch):
    """Asking the context first keeps the operator-typed-it-meant-it contract of ``_host_shard_ratio_from_env``."""
    monkeypatch.setenv("SGLANG_MOE_HOST_SHARD_RATIO", "1,2,3")  # three weights for two ranks
    boot.group("D")
    with pytest.raises(ValueError, match="entries"):
        _stage(_gguf_layer(0, rank=0))


# ---------------------------------------------------------------------------------------------------------------
# 2. the invariant is checked at LOAD, by name
# ---------------------------------------------------------------------------------------------------------------


def _plan(delegated=(), spill=(5, 6), resident=(0, 1, 2, 3, 4), n=7):
    return types.SimpleNamespace(delegated_ids=tuple(delegated), spill_ids=tuple(spill),
                                 resident_ids=tuple(resident), num_experts=n)


def test_a_plan_with_delegated_experts_is_refused_at_load_with_the_ids():
    with pytest.raises(gp.GGUFPresplitRefused, match=r"delegated to a peer rank's host tier.*\[3, 4\]"):
        gp.assert_cold_tier_complete(7, "w13_qweight", _plan(delegated=(3, 4), spill=(5, 6)))


def test_a_store_index_missing_a_cold_expert_is_refused_at_load_with_the_ids():
    with pytest.raises(gp.GGUFPresplitRefused, match=r"without a store row: \[6\]"):
        gp.assert_cold_tier_complete(7, "w2_qweight", _plan(), {5: 0})
    with pytest.raises(gp.GGUFPresplitRefused, match=r"row without a staged expert: \[1\]"):
        gp.assert_cold_tier_complete(7, "w2_qweight", _plan(), {1: 3, 5: 0, 6: 1})
    gp.assert_cold_tier_complete(7, "w2_qweight", _plan(), {5: 0, 6: 1})   # complete: silent


# ---------------------------------------------------------------------------------------------------------------
# 3. a gap that survives is a NAMED abort of the pool layout, not a KeyError
# ---------------------------------------------------------------------------------------------------------------


def test_a_gap_in_the_pool_map_is_a_named_abort_with_ids_and_tier(boot, cold_tier_on):
    boot.group("D")
    layer = _stage(_gguf_layer(1, rank=0))
    cold, _res = _cold_local(layer)
    cache = _cache(layer)
    gone = cold[0]
    cache._spill_pool_index = {e: r for e, r in cache._spill_pool_index.items() if e != gone}
    cache.planner.delegated_ids = frozenset([gone])
    with pytest.raises(RuntimeError) as exc:
        cache._pool_layout()
    msg = str(exc.value)
    assert not isinstance(exc.value, KeyError)
    assert f"first [{gone}]" in msg and "delegated to a peer rank's host tier" in msg and "layer 1" in msg
    # the same gap without a delegation record: the other tier
    cache.planner.delegated_ids = None
    with pytest.raises(RuntimeError, match=r"in neither the resident set nor this rank's host/store tier index"):
        cache._pool_layout()


def test_the_default_static_pool_layout_is_unchanged():
    """No pool index = the static ``id - R`` layout of every non-GGUF path: byte for byte what it was."""
    fake = types.SimpleNamespace(
        resident_count=3, num_local_experts=8, _spill_pool_index=None,
        planner=types.SimpleNamespace(resident_ids=None, resident_slot=None, delegated_ids=None),
        layer=types.SimpleNamespace(layer_id=5),
    )
    hot, host = eo.MoEExpertOffloadCache._pool_layout(fake)
    assert hot == {0: 0, 1: 1, 2: 2} and host == [-1, -1, -1, 0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------------------------------------------
# 4. the numbers of the 12.10. boot, from its real Karte (evidence file; skipped where it is not mounted)
# ---------------------------------------------------------------------------------------------------------------

_KARTE_OF_THE_BOOT = "/spinning/docker-acceptance/nf/evidence/expert_map_dkrnfdggufdbar110120619.json"


@pytest.mark.skipif(not os.path.exists(_KARTE_OF_THE_BOOT), reason="the stage-A evidence is not mounted")
def test_the_stage_a_karte_and_ratio_give_exactly_the_logged_deaths():
    """87 / 73 / 84 are the first non-resident experts the 1:0.45:0.92 share left in no tier; owned 76 / 13 / 36.

    Pins the diagnosis to the log (``[06:32:17 TP0] KeyError: 87`` ...) with the plan function of the tree: the Karte's
    D residents + the pad, the measured ratio of the boot (rank0=0.4214 rank1=0.1891 rank2=0.3895)."""
    m = json.load(open(_KARTE_OF_THE_BOOT))
    lo, span = m["bounds"][:3], m["spans"]
    weights = [0.4214, 0.1891, 0.3895]
    first_missing, owned_n = [], []
    for rank in range(3):
        res = {int(g) - lo[rank] for g in m["phases"]["D"]["resident"][rank]} | {span[rank]}
        E = span[rank] + 1
        cold = [e for e in range(E) if e not in res]
        owned = set(eo.partition_cold_experts(cold, weights)[rank])
        first_missing.append(next(e for e in cold if e not in owned))
        owned_n.append(len(owned))
    assert first_missing == [87, 73, 84] and owned_n == [76, 13, 36]


# ---------------------------------------------------------------------------------------------------------------
# 5. BEFUND 2 of the same boot: ``Parameter model.embed_tokens.qweight not found`` on the expert workers is by design
# ---------------------------------------------------------------------------------------------------------------

_STAGE_A_D_LOG = (
    "/spinning/gpu-arb/deskq/done/gguf-stufeA-g12-1012/"
    "boot_weg2_dkrnfdggufdbar110120619_8d5b142727_1012_062031.D.log"
)


def test_a_form_a_worker_holds_experts_and_router_and_no_embedding_or_head():
    """The two ``embed_tokens.qweight`` lines (TP1, TP2) are 2 of 1230 per worker: a Form A worker (rank-tp-ratio 0)
    builds no dense module, so the GGUF stream's dense tensors have nowhere to land (``load_weights`` logs and goes on).
    TP0 (the host) holds embed_tokens / lm_head, its census says so, and it printed none of these lines."""
    from sglang.srt import rank_role as rr

    for name in ("model.embed_tokens.qweight", "model.embed_tokens.qweight_type", "lm_head.qweight",
                 "lm_head.qweight_type", "model.layers.3.mlp.shared_expert.down_proj.qweight",
                 "model.layers.3.linear_attn.in_proj_qkv.qweight", "model.hyper_connection_mixer.hc_norm.weight"):
        assert not rr.worker_keeps_parameter(name, 512), name
    assert rr.worker_keeps_parameter("model.layers.3.mlp.experts.17.gate_proj.qweight", 512)
    assert rr.worker_keeps_parameter("model.layers.3.mlp.gate.weight", 512)


@pytest.mark.skipif(not os.path.exists(_STAGE_A_D_LOG), reason="the stage-A D log is not mounted")
def test_every_not_found_line_of_the_stage_a_workers_names_a_parameter_a_worker_does_not_keep():
    """The harmlessness claim against the real log: 1230 lines on TP1 and on TP2, none for TP0, and not ONE of the
    names is a routed expert or the router (the things a worker must have)."""
    import re

    from sglang.srt import rank_role as rr

    pat = re.compile(r"\[[^\]]* (TP\d)\] Parameter (\S+) not found while loading Qwen4-Exp VL weights")
    by_rank = {}
    for line in open(_STAGE_A_D_LOG, errors="replace"):
        m = pat.search(line)
        if m:
            by_rank.setdefault(m.group(1), []).append(m.group(2))
    assert "TP0" not in by_rank
    assert {r: len(v) for r, v in by_rank.items()} == {"TP1": 1230, "TP2": 1230}
    for rank, names in by_rank.items():
        kept = [n for n in names if rr.worker_keeps_parameter(n, 512)]
        assert kept == [], (rank, kept[:5])
    assert "model.embed_tokens.qweight" in by_rank["TP1"] and "model.embed_tokens.qweight" in by_rank["TP2"]
