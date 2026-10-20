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
    """Rank agreement: the worker's own answer is yes, a peer votes no (ok = 0 in the
    group MIN) -> no rank moves, the worker's pages stay where they are."""
    caplog.set_level(logging.INFO)
    dsv_, sched, caps, alloc, tree, target, dw = _form_metal(floor_env, monkeypatch, _tp1)
    votes = []

    def peer_says_no(sched_, vec):
        votes.append(list(vec))
        return [0] + list(vec[1:])
    monkeypatch.setattr(dsv_, "_kv_compact_group_vote", peer_says_no)
    ms, st, turn, v_old = _end_event(dsv_, sched, alloc, tree)
    assert votes and votes[0][0] == 1, "the worker itself could not vote yes"
    assert not st.changed and ms.stage == 9
    assert torch.equal(turn.component_data[FULL].value, v_old)
    r = _refusals(caplog, dsv_)
    assert r and "a peer rank cannot clear the boundary" in r[0], r


# ---- (5) planner order 20.10.: the eviction choice is rank-agreed (group AND of backed) ----
#
# rc23 d76cae6851 (boot 1020_105621, 11:05:06) died of the same class in another place:
# ``S86 SLEEP-L3-RELEASE ... slots=431/433/433 ... digest=DIVERGED kept=aux_host_only:24``
# (TP0) vs ``aux_host_only:8`` (TP1/TP2) -- the host and the expert workers do not hold
# the same host-side state. KV-COMPACT's ``backed`` reads exactly such state.

from test_s77_kv_compact_s9_hold_1019 import MAMBA  # noqa: E402


def _three_ranks(monkeypatch, *, host_backs_x):
    """TP0 host, TP1/TP2 workers; the same replicated allocator and tree steps. Node X:
    the host holds its real GDN state without a host copy (unbacked there), the workers'
    R12 shadow holds no state (backed there) -- unless ``host_backs_x``. Node Y backed
    everywhere, node Z unbacked everywhere. Below S8 only 10 pages are free: X (8) + Z (2)
    fit, Y must be evicted -- the plan NEEDS an eviction."""
    ranks = []
    for name, form in (("TP0", _tp0), ("TP1", _tp1), ("TP2", _tp2)):
        target, dw = form(monkeypatch)
        alloc = _alloc(TOP // PG, PG)
        alloc._kvcache = target
        tree = Tree(alloc)
        _take(alloc, [p for p in range(1, S8_PAGE + 1) if p not in range(5000, 5010)])
        x = _node(tree, alloc, list(range(6200, 6208)), PG, backed=True)
        if name == "TP0" and not host_backs_x:
            x.component_data[MAMBA] = types.SimpleNamespace(value=torch.tensor([1]), host_value=None,
                                                            lock_ref=0)
        y = _node(tree, alloc, list(range(6300, 6308)), PG, backed=True)
        z = _node(tree, alloc, list(range(6400, 6402)), PG, backed=False)
        sched = types.SimpleNamespace(token_to_kv_pool_allocator=alloc, page_size=PG,
                                      draft_worker=dw, tree_cache=tree)
        sv, why, cop = dsv._kv_compact_rank_survey(sched, S8_PAGE)
        assert why is None, (name, why)
        ranks.append((name, sched, sv, {"x": x, "y": y, "z": z}))
    return ranks


def _tp2(monkeypatch):
    ratio = CUT[2]
    monkeypatch.setattr(dsv, "kv_owner_block", lambda: (S, ratio))
    target = _qsa_pool((TOP // S + 1) * ratio, index_on_rank=False)
    return target, _shadow_draft_worker()


def _vote_and_plan(ranks, top=TOP // PG):
    vecs = [dsv._kv_compact_vote(sv, None, S8_PAGE, top) for _n, _s, sv, _x in ranks]
    assert len({len(v) for v in vecs}) == 1, "the ranks enter the collective with unequal lengths"
    group = [min(col) for col in zip(*vecs)]                         # gloo MIN over 3 ranks
    out = []
    for name, _s, sv, nodes in ranks:
        plans, why = dsv._kv_compact_plans_from_vote(sv, group, S8_PAGE, top, [S8_PAGE])
        out.append((name, plans[0], why, nodes))
    return out


def test_three_ranks_with_unequal_backed_choose_the_same_eviction_and_digest(floor_env, monkeypatch):
    ranks = _three_ranks(monkeypatch, host_backs_x=False)
    local_backed = {n: {k: sv.owner[min(v.component_data[FULL].value.tolist()) // PG].backed
                        for k, v in nodes.items()} for n, _s, sv, nodes in ranks}
    assert local_backed["TP0"]["x"] is False and local_backed["TP1"]["x"] is True   # the premise
    # the class: planned on each rank's OWN survey (beef9035eb and before), the host keeps X
    # and evicts Y while the workers evict X -- different digests, the W86-style group death
    rank_local = [KC.plan_for(sv, S8_PAGE)[0] for _n, _s, sv, _x in ranks]
    assert len({KC.plan_digest(8, p) for p in rank_local}) > 1
    res = _vote_and_plan(ranks)
    digests = set()
    for name, plan, why, nodes in res:
        assert plan is not None, (name, why)
        assert plan.evicts, "the plan was meant to need an eviction"
        assert [m.node for m in plan.evicts] == [nodes["y"]], name              # never X
        assert {id(m.node) for m, _s, _d in plan.moves} == {id(nodes["x"]), id(nodes["z"])}, name
        digests.add(KC.plan_digest(8, plan))
    assert len(digests) == 1, "the ranks' plans differ (crash-stop on the metal)"


def test_three_ranks_backed_everywhere_evict_by_the_replicated_order(floor_env, monkeypatch):
    ranks = _three_ranks(monkeypatch, host_backs_x=True)
    res = _vote_and_plan(ranks)
    assert len({KC.plan_digest(8, p) for _n, p, _w, _x in res}) == 1
    assert all(p is not None and p.evicts for _n, p, _w, _x in res)


def test_a_page_live_on_one_rank_and_free_on_another_refuses_everywhere(floor_env, monkeypatch):
    ranks = _three_ranks(monkeypatch, host_backs_x=True)
    name, sched, sv, nodes = ranks[2]
    sv.live_above = [p for p in sv.live_above if p != 6400]                  # TP2: page free
    sv.owner.pop(6400)
    res = _vote_and_plan(ranks)
    for name, plan, why, _x in res:
        assert plan is None and "live on some ranks and free on others" in why, (name, why)


def test_a_page_blocked_on_one_rank_blocks_every_rank(floor_env, monkeypatch):
    ranks = _three_ranks(monkeypatch, host_backs_x=True)
    ranks[1][2].owner[6300].pinned = True                                     # TP1: Y pinned
    res = _vote_and_plan(ranks)
    assert all(p is None and "blocks on a rank" in w for _n, p, w, _x in res), \
        [w for _n, _p, w, _x in res]


def test_a_destination_free_on_one_rank_only_is_no_destination(floor_env, monkeypatch):
    ranks = _three_ranks(monkeypatch, host_backs_x=True)
    ranks[0][2].free = [p for p in ranks[0][2].free if p != 5000]            # TP0: 5000 taken
    res = _vote_and_plan(ranks)
    for _n, plan, _w, _x in res:
        assert plan is not None and 5000 not in [d for _m, _s, ds in plan.moves for d in ds]
    assert len({KC.plan_digest(8, p) for _n, p, _w, _x in res}) == 1


def test_the_vote_is_one_collective_on_the_tick(floor_env, monkeypatch):
    """No new collective: the survey step enters exactly one group MIN (the vote)."""
    dsv_, sched, caps, alloc, tree, target, dw = _form_metal(floor_env, monkeypatch, _tp1)
    seen = []
    real = dsv_._kv_compact_group_vote
    monkeypatch.setattr(dsv_, "_kv_compact_group_vote",
                        lambda s_, v: seen.append(len(v)) or real(s_, v))
    ms, st, turn, v_old = _end_event(dsv_, sched, alloc, tree)
    assert st.changed and len(seen) == 1


def test_a_higher_candidate_takes_its_destinations_above_the_lowest_boundary(floor_env, monkeypatch):
    """The vote covers every candidate at once: a page free between the lowest boundary
    (S4) and S8 is a destination for S8 -- the move, not an eviction."""
    t0, dw0 = _tp0(monkeypatch)
    sched = _replica(t0, dw0)
    s4 = FLOOR_LADDER[4] // PG
    plans, why, _c = dsv._kv_compact_rank_plans(sched, [s4, S8_PAGE])
    assert plans[0] is None                                       # 1..4999 run below S8, above S4
    assert plans[1] is not None and not plans[1].evicts, why
    dst = [d for _m, _s, ds in plans[1].moves for d in ds]
    assert dst and min(dst) > s4 and max(dst) <= S8_PAGE


def test_a_page_withheld_by_the_cap_is_no_destination_in_the_vote(floor_env, monkeypatch):
    """Not live is not free: ids the stage cap withholds (above it) are no destination
    for any candidate, even though nothing lives there."""
    from flliper.srt.managers.kv_backing_relief import KvRowCap

    t0, dw0 = _tp0(monkeypatch)
    sched = _replica(t0, dw0)
    alloc = sched.token_to_kv_pool_allocator
    cap = alloc._pdflip_kv_stage_cap = KvRowCap(alloc)
    cap.engage(5050)                                             # 51 free pages below it
    s4 = FLOOR_LADDER[4] // PG                                   # the vote's floor: S4, so the
    plans, why, _c = dsv._kv_compact_rank_plans(sched, [s4, S8_PAGE])   # withheld ids sit above it
    assert plans[1] is not None, why
    dst = [d for _m, _s, ds in plans[1].moves for d in ds]
    assert dst and max(dst) <= 5050
    assert plans[1].evicts, "85 pages do not fit into 51: the backed node is the fallback"


def test_a_rank_without_a_plan_still_enters_the_digest_collective(floor_env, monkeypatch, caplog):
    """The group voted yes, but this rank's tree yields no plan (a structure its peers do
    not share): it must not skip the digest collective its peers enter -- it answers -1.
    S98c: the unequal digest is a UNIFORM, named refusal (nothing moved yet), no crash."""
    caplog.set_level(logging.INFO)
    dsv_, sched, caps, alloc, tree, target, dw = _form_metal(floor_env, monkeypatch, _tp1)
    monkeypatch.setattr(dsv_, "_kv_compact_plans_from_vote",
                        lambda sv, group, fp, tp, bounds: ([None] * len(bounds), "no plan here"))
    calls = []

    def same(sched_, value):
        calls.append(int(value))
        return int(value), 12345                                       # a peer has its plan
    monkeypatch.setattr(dsv_, "_group_same", same)
    ms, st, turn, v_old = _end_event(dsv_, sched, alloc, tree)
    assert calls == [-1]
    assert not st.changed and ms.stage == 9
    assert torch.equal(turn.component_data[FULL].value, v_old)
    assert any("verdict=refused-plan-diff" in m for m in caplog.messages)


def test_no_plan_on_every_rank_after_a_yes_vote_is_a_refusal(floor_env, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    dsv_, sched, caps, alloc, tree, target, dw = _form_metal(floor_env, monkeypatch, _tp1)
    monkeypatch.setattr(dsv_, "_kv_compact_plans_from_vote",
                        lambda sv, group, fp, tp, bounds: ([None] * len(bounds), "no plan here"))
    ms, st, turn, v_old = _end_event(dsv_, sched, alloc, tree)
    assert not st.changed and ms.stage == 9
    r = _refusals(caplog, dsv_)
    assert r and "no plan here" in r[0], r


# ---- (6) rang review nf-review-s98-1020.md (S98c): the same pages, other node cuts ------
#
# rc24 781238542c (_113244, D.log 21695-21697, 11:39:51): the same 177 device slots in 32
# nodes on TP0 and 35 on TP1/TP2 -- node splits are rank-local.

def _split_ranks(monkeypatch, *, free_below):
    """Three ranks, the SAME pages above S8 (6200..6207 backed, 6300..6301 unbacked); TP0
    holds 6200..6207 as ONE node, TP1/TP2 split it into 6200..6203 -> 6204..6207."""
    ranks = []
    for name, form in (("TP0", _tp0), ("TP1", _tp1), ("TP2", _tp2)):
        target, dw = form(monkeypatch)
        alloc = _alloc(TOP // PG, PG)
        alloc._kvcache = target
        tree = Tree(alloc)
        _take(alloc, [p for p in range(1, S8_PAGE + 1) if p not in free_below])
        if name == "TP0":
            _node(tree, alloc, list(range(6200, 6208)), PG, backed=True)
        else:
            a = _node(tree, alloc, list(range(6200, 6204)), PG, backed=True)
            _node(tree, alloc, list(range(6204, 6208)), PG, backed=True, parent=a)
        _node(tree, alloc, [6300, 6301], PG, backed=False)
        sched = types.SimpleNamespace(token_to_kv_pool_allocator=alloc, page_size=PG,
                                      draft_worker=dw, tree_cache=tree)
        sv, why, cop = dsv._kv_compact_rank_survey(sched, S8_PAGE)
        assert why is None, (name, why)
        ranks.append((name, sched, sv, {}))
    assert len({len(sv.nodes) for _n, _s, sv, _x in ranks}) == 2       # the premise: other cuts
    return ranks


def test_three_ranks_with_other_node_cuts_make_the_same_page_plan_and_move_alike(
        floor_env, monkeypatch):
    ranks = _split_ranks(monkeypatch, free_below=range(5000, 5020))      # room for all 10
    res = _vote_and_plan(ranks)
    plans = [p for _n, p, _w, _x in res]
    assert all(p is not None and not p.evicts for p in plans), [w for _n, _p, w, _x in res]
    assert len({KC.plan_digest(8, p) for p in plans}) == 1
    want = list(zip(list(range(6200, 6208)) + [6300, 6301], range(5000, 5010)))
    assert all(p.pairs == want for p in plans)
    # executed locally, grouped by each rank's own nodes: the same pages land alike
    for (name, sched, sv, _x), plan in zip(ranks, plans):
        KC.execute(sched.tree_cache, sched.token_to_kv_pool_allocator, plan, _NullCopier())
    vals = []
    for name, sched, _sv, _x in ranks:
        pages = sorted(int(x) for n in sched.tree_cache._collect_all_nodes()
                       if n is not sched.tree_cache.root_node
                       for x in (n.component_data[FULL].value // PG).unique().tolist())
        vals.append(pages)
        free = sorted(sched.token_to_kv_pool_allocator.free_pages.tolist())
        assert free[:3] == [5010, 5011, 5012], name
    assert vals[0] == vals[1] == vals[2] == list(range(5000, 5010))


def test_the_rc24_split_with_a_needed_eviction_is_agreed_or_refused_uniformly(
        floor_env, monkeypatch):
    """Short of room the fallback evicts NODES (cut-dependent): the digest compares the
    pages they free. TP0 frees 6200..6207 with its one node; TP1/TP2 evict the leaf
    6204..6207 only -- the page sets differ, so the digests differ: the group refuses
    (uniform, nothing moved) instead of a crash-stop."""
    ranks = _split_ranks(monkeypatch, free_below=range(5000, 5006))      # 6 of 10 pages
    res = _vote_and_plan(ranks)
    plans = [p for _n, p, _w, _x in res]
    # TP0 evicts its one node (all 8 pages); TP1/TP2 move the leaf and cannot evict the
    # parent (a device child that stays) -- no plan there
    assert plans[0] is not None and plans[0].evict_pages == list(range(6200, 6208))
    assert plans[1] is None and plans[2] is None
    digests = [KC.plan_digest(8, p) if p is not None else -1 for p in plans]
    lo, hi = min(digests), max(digests)
    assert lo != hi        # -> _kv_compact_for_shrink: uniform refusal, nothing moved
                           #    (test_a_rank_without_a_plan_still_enters_the_digest_collective)


class _NullCopier:
    page_size = PG

    def copy(self, src, dst):
        return 0


def test_an_eviction_that_frees_pages_below_the_floor_differs_by_those_pages(floor_env, monkeypatch):
    """The digest counts EVERY page an eviction frees, not only the ones above the floor:
    TP0's one node 6140..6143 + 6200..6207 frees 6140..6143 with it, TP1's split leaf
    6200..6207 does not -- the allocators would part, so the digests must differ."""
    plans = []
    for name, form in (("TP0", _tp0), ("TP1", _tp1)):
        target, dw = form(monkeypatch)
        alloc = _alloc(TOP // PG, PG)
        alloc._kvcache = target
        tree = Tree(alloc)
        _take(alloc, list(range(1, S8_PAGE + 1)))                     # no room below S8
        low, high = list(range(6140, 6144)), list(range(6200, 6208))
        if name == "TP0":                                             # low: taken above
            _node(tree, alloc, high, PG, backed=True).component_data[FULL].value = \
                torch.cat([_tokens_of(low), _tokens_of(high)])
        else:
            a = _node(tree, alloc, [], PG, backed=True)
            a.component_data[FULL].value = _tokens_of(low)
            _node(tree, alloc, high, PG, backed=True, parent=a)
        sched = types.SimpleNamespace(token_to_kv_pool_allocator=alloc, page_size=PG,
                                      draft_worker=dw, tree_cache=tree)
        sv, why, _c = dsv._kv_compact_rank_survey(sched, S8_PAGE)
        assert why is None, why
        plans.append((sv, sched))
    vecs = [dsv._kv_compact_vote(sv, None, S8_PAGE, TOP // PG) for sv, _s in plans]
    group = [min(c) for c in zip(*vecs)]
    got = [dsv._kv_compact_plans_from_vote(sv, group, S8_PAGE, TOP // PG, [S8_PAGE])[0][0]
           for sv, _s in plans]
    assert got[0] is not None and got[1] is not None
    assert got[0].evict_pages == low + high and got[1].evict_pages == high
    assert KC.plan_digest(8, got[0]) != KC.plan_digest(8, got[1])


def _tokens_of(pages):
    return torch.tensor([p * PG + o for p in pages for o in range(PG)], dtype=torch.int64)
