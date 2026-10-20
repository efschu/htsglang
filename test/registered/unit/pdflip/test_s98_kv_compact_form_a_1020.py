"""S98: KV-COMPACT refused on NF rc22 every time -- the Form A rank forms.

THE METAL (NF rc22 9c40fe40a2, boot 1020_094747, D.log): 51/51 ``PDFLIP D-MEM-SCHED
KV-COMPACT ... verdict=refused``, ``moved`` 0, with two reasons:

* TP0 (the Form A attention HOST, ``#239 S3a ZERO-SHARE CELL: rank 0 owns no
  full-attention token (vector [0, 3, 5])``): ``target.full_kv_pool.k_buffer[0]: 64
  rows match no slot geometry of the 524352-token space``. The token cut gives TP0 an
  owner share of 0, ``dcp_compact_pool_rows`` sizes its FA pool at 0 rows
  (``KV Cache is allocated ... #tokens: 0``), and an MHA pool always carries
  ``size + page_size`` rows: the 64 rows are the pad page alone. The owner rule maps
  no slot there -- nothing token-indexed lives in that pool, nothing has to move.
* TP1/TP2 (Form A EXPERT WORKERS, ``Draft-solo placement: rank N is a draft SHADOW
  (meta-device draft, no draft weights/KV/graphs ...)``): ``draft?: pool not found``.
  A solo shadow has no draft pool by design (``kv_cache_builder.get_draft_kv_pool``,
  ``PDFLIP-D-PARK DRAFT-SKIP``); the draft rows of every global token live on the
  solo host TP0 (``Draft-solo placement: rank 0 HOSTS the unsharded solo draft``,
  ``KV Cache is allocated ... #tokens: 524288``), and TP0 moves them.

Consequence: 81x ``hold S9 (wanted S8)``, 15x ``hold S9 (wanted S0)``, rows_on at
WAKE-RESHARD TP0 4 / TP1 2 / TP2 1 (rc16 bs2 ~21).

WHAT MUST HOLD.
(1) TP0 with the REAL pool classes (QSATokenToKVPool at size 0 under the token cut,
    the solo draft MHATokenToKVPool over the whole context) and the REAL stage cap:
    RED on 9c40fe40a2 with the metal text, GREEN: S8, the QSA index groups and the
    draft rows of every moved page carry their bytes.
(2) TP1/TP2 with the real owner-share pool (no QSA index on the worker) and a solo
    shadow draft runner: RED with ``draft?: pool not found``, GREEN: S8, the owner
    share rows moved.
(3) The two forms make the SAME plan for the same replicated tree (digest), so the
    group check does not crash-stop on Form A; the plan order uses replicated state
    only when the room suffices (``backed`` can differ between the host and a worker).
(4) Fail-closed stays: a draft runner without a pool that does NOT declare itself a
    solo shadow, a declared shadow that holds a pool anyway (walked, not skipped), a
    size-0 pool with rows beyond the pad page, a 64-row buffer of a pool that is not
    size 0, and an unknown token-indexed tensor beside the empty FA pool all refuse
    (or move) by name -- no silent KV corruption.
"""
from __future__ import annotations

import logging
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache.memory_pool import MHATokenToKVPool  # noqa: E402
from flliper.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool  # noqa: E402
from flliper.srt.pdflip import d_kv_compact as KC  # noqa: E402
from flliper.srt.pdflip import d_seat_vram as dsv  # noqa: E402

from test_s77_kv_compact_s9_hold_1019 import (  # noqa: E402,F401
    FULL, PG, S8_PAGE, Tree, _alloc, _node, _take, floor_env, tick_env,
)
from test_s84_kv_compact_listener_1019 import _armed_metal, _end_event  # noqa: E402
from test_pdflip_d_mem_sched_0929 import FLOOR_LADDER  # noqa: E402

TOP = FLOOR_LADDER[-1]                 # 524288: the stage form's top, the global slot space
RATIO = 4                              # QSA tokens per compressed slot
CUT = (0, 3, 5)                        # the rc22 token vector: S = 8
S = sum(CUT)


def _fill(t):
    flat = t.view(-1).view(torch.uint8)
    flat.copy_((torch.arange(flat.numel(), dtype=torch.int64) * 7 % 251).to(torch.uint8))
    return t


def _qsa_pool(size, *, index_on_rank):
    """The real NF target pool on the desk (CPU, tiny heads), built with the stage form
    off (no saver region) -- the shapes are the ones the boot builds."""
    with mock.patch.object(dsv, "stage_form", lambda env=None: None):
        return QSATokenToKVPool(
            size=size, dtype=torch.bfloat16, page_size=PG, head_num=1, head_dim=4,
            full_attention_layer_ids=[3, 7], device="cpu", mamba_pool=types.SimpleNamespace(),
            qsa_index_kv_heads=1, qsa_index_head_dim=8, qsa_compress_ratio=RATIO,
            qsa_token_topk=8, num_request_slots=4, qsa_slot_space=TOP,
            qsa_index_on_rank=index_on_rank)


def _solo_draft_pool():
    """The solo host's draft pool: the whole context (``#tokens: 524288``), 1 layer."""
    with mock.patch.object(dsv, "stage_form", lambda env=None: None):
        pool = MHATokenToKVPool(TOP, PG, torch.bfloat16, 1, 4, 1, "cpu", False,
                                enable_alt_stream=False)
    _fill(pool.k_buffer[0])
    _fill(pool.v_buffer[0])
    return pool


def _eagle_v2(runner):
    """EAGLEWorkerV2: ``draft_worker.draft_worker`` is an EagleDraftWorker with
    ``draft_runner`` and ``draft_worker.model_runner`` (the same runner)."""
    eagle = types.SimpleNamespace(draft_runner=runner,
                                  draft_worker=types.SimpleNamespace(model_runner=runner))
    return types.SimpleNamespace(draft_worker=eagle)


def _host_draft_worker(pool):
    return _eagle_v2(types.SimpleNamespace(token_to_kv_pool=pool, is_draft_solo_host=True,
                                           is_draft_solo_shadow=False))


def _shadow_draft_worker(pool=None):
    """``Draft-solo placement: rank N is a draft SHADOW``: the runner exposes
    ``token_to_kv_pool`` as None (``kv_cache_builder.get_draft_kv_pool``)."""
    return _eagle_v2(types.SimpleNamespace(token_to_kv_pool=pool, is_draft_solo_host=False,
                                           is_draft_solo_shadow=True))


def _tp0(monkeypatch):
    monkeypatch.setattr(dsv, "kv_owner_block", lambda: (0, 0))     # share 0: hi == lo
    target = _qsa_pool(0, index_on_rank=True)
    _fill(target.qsa_compressed_flat)
    return target, _host_draft_worker(_solo_draft_pool())


def _tp1(monkeypatch):
    ratio = CUT[1]
    monkeypatch.setattr(dsv, "kv_owner_block", lambda: (S, ratio))
    size = (TOP // S + 1) * ratio                                    # 196611 (rc22 TP1)
    target = _qsa_pool(size, index_on_rank=False)
    for t in target.full_kv_pool.k_buffer + target.full_kv_pool.v_buffer:
        _fill(t)
    return target, _shadow_draft_worker()


def _form_metal(floor_env, monkeypatch, form):
    dsv_, sched, caps, alloc, tree, born = _armed_metal(floor_env, monkeypatch)
    monkeypatch.setattr(dsv_, "_KV_BORN", [])
    target, dw = form(monkeypatch)
    alloc._kvcache = target
    sched.draft_worker = dw
    return dsv_, sched, caps, alloc, tree, target, dw


def _refusals(caplog, dsv_):
    return [m for m in caplog.messages if dsv_.KV_COMPACT_MARK in m and "verdict=refused" in m]


# ---- (1) TP0: the attention host, owner share 0 --------------------------------------

def test_the_real_tp0_pool_has_the_metal_shape():
    """The desk reproduces the boot: the FA sub-pool of TP0 is 0 tokens + the pad page."""
    t = _qsa_pool(0, index_on_rank=True)
    assert t.full_kv_pool.size == 0
    assert [tuple(b.shape)[0] for b in t.full_kv_pool.k_buffer] == [PG, PG]


def test_rc22_tp0_the_empty_fa_share_no_longer_refuses_and_qsa_and_draft_move(
        floor_env, monkeypatch, caplog):
    """RED on 9c40fe40a2: ``target.full_kv_pool.k_buffer[0]: 64 rows match no slot
    geometry of the 524352-token space``. GREEN: S8; every moved page's QSA groups and
    draft rows carry their bytes; the empty FA share is no buffer of the move."""
    caplog.set_level(logging.INFO)
    dsv_, sched, caps, alloc, tree, target, dw = _form_metal(floor_env, monkeypatch, _tp0)
    draft = dw.draft_worker.draft_runner.token_to_kv_pool
    q0 = target.qsa_compressed_flat.clone()
    d0 = (draft.k_buffer[0].clone(), draft.v_buffer[0].clone())
    ms, st, turn, v_old = _end_event(dsv_, sched, alloc, tree)
    assert st.changed and ms.stage == 8, (st.reason, _refusals(caplog, dsv_))
    moved = [m for m in caplog.messages if dsv_.KV_COMPACT_MARK in m and "verdict=moved" in m]
    assert moved and "buffers=3" in moved[0], moved          # QSA flat + draft K + draft V
    v_new = turn.component_data[FULL].value
    assert int((v_new // PG).max()) <= S8_PAGE
    assert torch.equal(draft.k_buffer[0][v_new], d0[0][v_old])
    assert torch.equal(draft.v_buffer[0][v_new], d0[1][v_old])
    cap = target.qsa_compressed_capacity
    assert torch.equal(target.qsa_compressed_flat.view(2, cap, -1)[:, v_new // RATIO],
                       q0.view(2, cap, -1)[:, v_old // RATIO])
    names, why = [n for n, _t, _g in dsv.kv_page_buffers(sched)[0]], dsv.kv_page_buffers(sched)[1]
    assert why is None and not any("full_kv_pool" in n for n in names), names


# ---- (2) TP1/TP2: the expert workers, solo draft shadows ------------------------------

def test_rc22_tp12_a_solo_draft_shadow_no_longer_refuses_and_the_owner_share_moves(
        floor_env, monkeypatch, caplog):
    """RED on 9c40fe40a2: ``draft?: pool not found``. GREEN: S8, the owner share rows of
    every moved page ((P // S) * ratio = 24 rows a page) carry their bytes."""
    caplog.set_level(logging.INFO)
    dsv_, sched, caps, alloc, tree, target, dw = _form_metal(floor_env, monkeypatch, _tp1)
    k0 = target.full_kv_pool.k_buffer[1].clone()
    ms, st, turn, v_old = _end_event(dsv_, sched, alloc, tree)
    assert st.changed and ms.stage == 8, (st.reason, _refusals(caplog, dsv_))
    v_new = turn.component_data[FULL].value
    per = (PG // S) * CUT[1]
    k1 = target.full_kv_pool.k_buffer[1]
    for po, pn in zip((v_old[::PG] // PG).tolist(), (v_new[::PG] // PG).tolist()):
        assert torch.equal(k1[pn * per:(pn + 1) * per], k0[po * per:(po + 1) * per])
    assert [n for n, _p in dsv._kv_pools_of(sched)] == ["target"]


# ---- (3) rank agreement on Form A ----------------------------------------------------

def _replica(target, dw, *, backed_flip=False):
    """One rank's replicated allocator + tree: the running decodes below S8, two idle
    turns above it (one backed, one not) -- the same steps on every rank."""
    alloc = _alloc(TOP // PG, PG)
    alloc._kvcache = target
    tree = Tree(alloc)
    _take(alloc, list(range(1, 5000)))
    _node(tree, alloc, list(range(6145, 6200)), PG, backed=not backed_flip)
    _node(tree, alloc, list(range(6300, 6330)), PG, backed=backed_flip)
    return types.SimpleNamespace(token_to_kv_pool_allocator=alloc, page_size=PG,
                                 draft_worker=dw, tree_cache=tree)


def test_host_and_worker_make_the_same_plan_for_the_same_tree(floor_env, monkeypatch):
    t0, dw0 = _tp0(monkeypatch)
    plans0, why0, cop0 = dsv._kv_compact_rank_plans(_replica(t0, dw0), [S8_PAGE])
    t1, dw1 = _tp1(monkeypatch)
    plans1, why1, cop1 = dsv._kv_compact_rank_plans(_replica(t1, dw1), [S8_PAGE])
    assert plans0[0] is not None and plans1[0] is not None, (why0, why1)
    assert KC.plan_digest(8, plans0[0]) == KC.plan_digest(8, plans1[0])
    assert len(cop0.items) == 3 and len(cop1.items) == 4          # QSA+draft K/V | 2x K/V


def test_the_plan_does_not_depend_on_the_rank_local_backed_flag_when_room_suffices(
        floor_env, monkeypatch):
    """``backed`` reads host copies and mamba anchors, which the host and an expert
    worker hold differently; with room for every mover it must not reorder the plan
    (a different digest is a crash-stop before anything moved)."""
    t0, dw0 = _tp0(monkeypatch)
    a, _w, _c = dsv._kv_compact_rank_plans(_replica(t0, dw0), [S8_PAGE])
    b, _w, _c = dsv._kv_compact_rank_plans(_replica(t0, dw0, backed_flip=True), [S8_PAGE])
    assert a[0] is not None and b[0] is not None
    assert KC.plan_digest(8, a[0]) == KC.plan_digest(8, b[0])


def test_without_room_the_unbacked_node_still_takes_the_room_first():
    """The fallback order is unchanged: short of room, the un-backed node is moved and
    the backed one evicted (its host copy stays), never the other way round."""
    alloc = _alloc(64, 4)
    tree = Tree(alloc)
    _take(alloc, list(range(1, 30)))                                 # free below 32: 30, 31, 32
    nb = _node(tree, alloc, [50, 51], 4, backed=True)              # the newer pages: first by id
    nu = _node(tree, alloc, [40, 41], 4, backed=False)
    plan, why = KC.plan_for(KC.survey(tree, alloc, 4, 32), 32)
    assert plan is not None, why
    assert [m.node for m, _s, _d in plan.moves] == [nu]
    assert [m.node for m in plan.evicts] == [nb]


# ---- (4) fail-closed stays ------------------------------------------------------------

def _sched(target, dw=None):
    alloc = _alloc(TOP // PG, PG)
    alloc._kvcache = target
    return types.SimpleNamespace(token_to_kv_pool_allocator=alloc, page_size=PG, draft_worker=dw)


def test_a_draft_runner_without_a_pool_that_is_no_declared_shadow_still_refuses(
        floor_env, monkeypatch):
    t1, _dw = _tp1(monkeypatch)
    plain = _eagle_v2(types.SimpleNamespace(token_to_kv_pool=None))
    bufs, why = dsv.kv_page_buffers(_sched(t1, plain))
    assert bufs == [] and why and "draft?" in why


def test_a_declared_shadow_that_holds_a_pool_is_walked_not_skipped(floor_env, monkeypatch):
    t1, _dw = _tp1(monkeypatch)
    pool = _solo_draft_pool()
    bufs, why = dsv.kv_page_buffers(_sched(t1, _shadow_draft_worker(pool)))
    names = [n for n, _t, _g in bufs]
    assert why is None and "draft.k_buffer[0]" in names and "draft.v_buffer[0]" in names


def test_a_mixed_draft_side_with_a_poolless_non_shadow_runner_refuses(floor_env, monkeypatch):
    """A shadow beside a runner that should hold a pool and does not: still ``draft?``."""
    t1, _dw = _tp1(monkeypatch)
    shadow = types.SimpleNamespace(token_to_kv_pool=None, is_draft_solo_shadow=True)
    lost = types.SimpleNamespace(token_to_kv_pool=None)
    dw = types.SimpleNamespace(draft_worker=types.SimpleNamespace(
        draft_runner=shadow, draft_worker=types.SimpleNamespace(model_runner=lost)))
    bufs, why = dsv.kv_page_buffers(_sched(t1, dw))
    assert bufs == [] and why and "draft?" in why


@pytest.mark.parametrize("rows", [PG + 1, 2 * PG])
def test_a_size_zero_pool_with_rows_beyond_the_pad_page_refuses(floor_env, monkeypatch, rows):
    t0, dw0 = _tp0(monkeypatch)
    t0.full_kv_pool.k_buffer[0] = torch.zeros(rows, 1, 4, dtype=torch.bfloat16)
    bufs, why = dsv.kv_page_buffers(_sched(t0, dw0))
    assert bufs == [] and why and "full_kv_pool.k_buffer[0]" in why


def test_a_pad_page_sized_buffer_of_a_non_empty_pool_still_refuses(floor_env, monkeypatch):
    t0, dw0 = _tp0(monkeypatch)
    t0.full_kv_pool.size = 4096                       # a pool that DOES hold tokens
    bufs, why = dsv.kv_page_buffers(_sched(t0, dw0))
    assert bufs == [] and why and "64 rows match no slot geometry" in why


def test_an_unknown_token_indexed_tensor_beside_the_empty_fa_share_refuses(floor_env, monkeypatch):
    t0, dw0 = _tp0(monkeypatch)
    t0.full_kv_pool.mystery = torch.zeros(TOP + PG, 2, dtype=torch.uint8)
    bufs, why = dsv.kv_page_buffers(_sched(t0, dw0))
    assert bufs == [] and why and "full_kv_pool.mystery" in why


def test_a_tp0_peer_refusal_keeps_the_worker_on_s9(floor_env, monkeypatch, caplog):
    """Rank agreement: the worker's own answer is yes, a peer says no -> group MIN 0,
    no rank moves, the worker's pages stay where they are."""
    caplog.set_level(logging.INFO)
    dsv_, sched, caps, alloc, tree, target, dw = _form_metal(floor_env, monkeypatch, _tp1)
    seen = []
    real = dsv_._kv_compact_rank_plans

    def counted(sched_, boundaries):
        seen.append(len(boundaries))
        return real(sched_, boundaries)
    monkeypatch.setattr(dsv_, "_kv_compact_rank_plans", counted)
    # the peer clears no stage: every KV-COMPACT flag (2 per candidate) is 0 in the MIN
    sched._pdflip_group_min_ints = lambda vals: (
        [0] * len(vals) if seen and len(vals) == 2 * seen[-1] else list(vals))
    ms, st, turn, v_old = _end_event(dsv_, sched, alloc, tree)
    assert not st.changed and ms.stage == 9
    assert torch.equal(turn.component_data[FULL].value, v_old)
    r = _refusals(caplog, dsv_)
    assert r and "a peer rank cannot clear the boundary" in r[0], r
