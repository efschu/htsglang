"""Q-1190 DUAL D-ARENA-YIELD (27B NVFP4 dual y9d1, boot
dkr27bnvfp4dual1mpsleepsharebar1fs10040258 @507edf614c, 03:07-03:16Z; analysis
deskq/done/1190-dual-wedge.out).

The KV L2 arena is ONE file shared by group P and group D. Q-697c made P give its own
references back (P PP0 own_held 574601 -> 90262), but D's tree held the arena: every D
rank 'ARENA-REF-HOLDERS pool=FULL tree=675527 tree_in_use=0 arena_pinned=694560' of
720896 slots -- host-only leaves of pages D had adopted from P's hand-offs, used by no
request. A slot frees only when no rank of either group references it: P's claims were
refused ('#1427 ARENA-DROP freed=0' 2893x, PASS-STALL 871x), P's hand-off pages and
Mamba anchors did not reach the store, D refused them (W50) and P prefilled them twice
(W53 PdFlipStoreHandbackFailed).

The fix (dual layout only): a refused claim posts its page need next to the shared arena;
D's TP0 takes it in the tick, the need rides the tick's group collective, every D rank
spills host-only H-leaves of its tree (L3 copy first) -- P's next claim finds its room.

Hermetic: the real C arena on a temp file (two pools = two processes on the same file),
the real ArenaMHAHostPool, the real ``_pdflip_direct_claim`` / ``_w3_arena_spill`` /
``_evict_for_claim``; pages of 64 B, page_size 1. The new module functions are looked up
with getattr so that on the base the tests fail by BEHAVIOUR (P's claim stays refused),
not by an import error."""

import logging
import os
import shutil
import sys
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "pdflip"))
import test_w3_arena_spill_0929 as W  # noqa: E402
import test_w3_dual_host_only_spill_q697c_1004 as Q  # noqa: E402

from flliper.srt.pdflip import dual_arena_spill as DS  # noqa: E402
from flliper.srt.pdflip import dual_d_kv_stage as DK  # noqa: E402
from flliper.srt.pdflip import dual_p_kv_stage as PK  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

FULL = W.FULL
DUAL_P = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "P", PK.MAX_TOKENS_ENV: "131072"}
DUAL_D = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D", DK.MAX_TOKENS_ENV: "131072"}
GATE_KEYS = ("FLLIPER_PDFLIP_DUAL_LAYOUT", "FLLIPER_PDFLIP_GROUP", PK.MAX_TOKENS_ENV, DK.MAX_TOKENS_ENV)
NOT_DUAL = [
    {},
    {"FLLIPER_PDFLIP_GROUP": "P"},
    {"FLLIPER_PDFLIP_GROUP": "D"},
    {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1"},
    {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "P"},   # no P KV cap: not armed
    {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D"},   # no D KV cap: not armed
    {"FLLIPER_PDFLIP_GROUP": "D", DK.MAX_TOKENS_ENV: "131072"},      # no dual layout (flip)
]


def _env(monkeypatch, env):
    for k in GATE_KEYS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


@pytest.fixture(autouse=True)
def _fresh_counters():
    if hasattr(DS, "_Y"):
        DS._Y.update(posts=0, yields=0, no_pool=0, quiet_until=0.0)
    yield


def _y9d1(tmp_path):
    """Two processes on ONE arena file: D's tree holds every slot through host-only leaves
    (tree_in_use=0), P's tree holds none (Q-697c gave them back) and claims two pages."""
    pd, arena, root, td, d_nodes = Q._host_only_tree(tmp_path)          # group D: the hoard
    pp, arena_p, _root = W._pool(tmp_path, Q.SLOTS)                      # group P: same file
    assert arena_p.path == arena.path
    tp = W._tree(pp)
    tp._components_tuple = (tp.components[FULL],)
    tp.tree_components = (FULL,)
    g = Q._Group(pp)                                                     # the 27B's HostPoolGroup
    tp.cache_controller.mem_pool_host = g
    tp._pdflip_direct_pool = lambda: g
    return tp, td, d_nodes, root, arena


def _d_side(td, tp_rank=0):
    """What D's tick does with the need (TP0 takes, the group collective carries it)."""
    take = getattr(DS, "d_take_need", None)
    yld = getattr(DS, "d_yield_arena", None)
    sched = types.SimpleNamespace(tree_cache=td, tp_rank=tp_rank)
    if take is None or yld is None:
        return 0, None                                                   # the base: nobody listens
    need = take(sched)
    return need, (yld(sched, need) if need else None)


def test_y9d1_d_tree_holds_the_shared_arena_p_claim_gets_room_after_d_yield(tmp_path, monkeypatch, caplog):
    """RED on 507edf614c: P's claim is refused (#1421 arena_claim) and STAYS refused --
    D never hears of it. GREEN: the refusal posts the need, D yields host-only leaves
    (L3 copy first), P's next claim gets its two slots."""
    caplog.set_level(logging.INFO)
    tp, td, d_nodes, root, arena = _y9d1(tmp_path)
    _env(monkeypatch, DUAL_P)
    m = W._claimer(tp)
    assert tp._pdflip_direct_claim(m) is False                             # the metal line, reproduced
    assert tp.refused == ["arena_claim"]
    _env(monkeypatch, DUAL_D)
    need, got = _d_side(td)
    assert need == 2, "P's refused claim did not reach group D"
    assert got is not None and got["released"] >= 2
    _env(monkeypatch, DUAL_P)
    pre = tp._pdflip_direct_claim(m)
    assert pre is not False, f"P's claim still refused after D's yield: {tp.refused}"
    assert pre is not None and int(pre.numel()) == 2
    for n in d_nodes:                                                    # #257: no page left L2 without L3
        if not Q._in_tree(td, n):
            st = n.hash_value[0] + "_sfx"
            assert (root / (st + ".bin")).read_bytes() == bytes([0x40 + int(n.hash_value[0][1:])]) * W.PAGE
    msgs = [r.getMessage() for r in caplog.records]
    assert any("Q-1190 DUAL ARENA-NEED POSTED" in s for s in msgs)
    assert any("Q-1190 DUAL D-ARENA-YIELD" in s for s in msgs)


def test_the_need_is_taken_once_and_only_by_tp0(tmp_path, monkeypatch):
    tp, td, d_nodes, root, arena = _y9d1(tmp_path)
    _env(monkeypatch, DUAL_P)
    tp._pdflip_direct_claim(W._claimer(tp))
    _env(monkeypatch, DUAL_D)
    assert DS.d_take_need(types.SimpleNamespace(tree_cache=td)) == 2
    assert DS.d_take_need(types.SimpleNamespace(tree_cache=td)) == 0     # cleared
    assert all(Q._in_tree(td, n) for n in d_nodes)                       # taking alone spills nothing


def test_a_d_leaf_without_an_l3_copy_stays(tmp_path, monkeypatch):
    """#257: D gives a slot back only with its L3 copy for every page."""
    pd, arena, root, td, d_nodes = Q._host_only_tree(tmp_path, refuse=("a7_sfx",))
    _env(monkeypatch, DUAL_D)
    got = DS.d_yield_arena(types.SimpleNamespace(tree_cache=td), 2)
    assert Q._in_tree(td, d_nodes[7]) and d_nodes[7].component_data[FULL].host_value is not None
    assert not (root / "a7_sfx.bin").exists()
    assert got["unsecured"] >= 1 and got["released"] == 0               # the chain's tail blocks the rest
    assert DS.post_need(td._pdflip_direct_pool(), 5) is True
    assert DS.d_take_need(types.SimpleNamespace(tree_cache=td)) == 0     # backoff after an empty yield
    DS._Y["quiet_until"] = 0.0
    assert DS.d_take_need(types.SimpleNamespace(tree_cache=td)) == 5     # the need waited, not lost


def test_no_spill_pool_is_a_named_stop(monkeypatch, caplog):
    _env(monkeypatch, DUAL_D)
    caplog.set_level(logging.INFO)
    t = types.SimpleNamespace(_pdflip_direct_pool=lambda: types.SimpleNamespace(), page_size=1)
    got = DS.d_yield_arena(types.SimpleNamespace(tree_cache=t), 64)
    assert got["released"] == 0
    assert any("Q-1190 DUAL D-ARENA-YIELD" in r.getMessage() and "STOP no_spill_pool" in r.getMessage()
               for r in caplog.records)


def test_flip_unchanged_a_refused_claim_off_the_gate_posts_nothing(tmp_path, monkeypatch):
    """Off the dual layout (flip, 27B INT8, NF, half-set gates): the claim is refused as
    before and no file appears next to the arena."""
    for env in NOT_DUAL:
        _env(monkeypatch, env)
        sub = tmp_path / ("e%d" % len(os.listdir(tmp_path)))
        sub.mkdir()
        tp, td, d_nodes, root, arena = _y9d1(sub)
        assert tp._pdflip_direct_claim(W._claimer(tp)) is False, env
        assert tp.refused == ["arena_claim"], env
        assert not os.path.exists(arena.path + ".dualneed"), env
        assert all(Q._in_tree(td, n) for n in d_nodes), env


# ---------------------------------------------------------------- the tick wiring
import test_pdflip_dual_d_kv_stage_0930 as T  # noqa: E402


def _tick(monkeypatch, tp_rank, group, take=0):
    _env(monkeypatch, DUAL_D)
    a = T.DemandReplayOfBsffsv._actor(None, 8192)
    sched = T._Sched(a, running=[], waiting=[], group=group)
    sched.tp_rank = tp_rank
    calls = []
    with mock.patch.object(DK._pk, "max_live_id", lambda *x: 0), \
            mock.patch.object(DS, "d_take_need", lambda s, env=None: take, create=True), \
            mock.patch.object(DS, "d_yield_arena", lambda s, n: calls.append(n), create=True):
        DK.tick(sched)
    return calls


def test_tick_every_d_rank_yields_the_group_need(monkeypatch):
    """TP1 took nothing itself; TP0's need arrives through the collective (MAX via MIN of
    negatives) and TP1 yields the same pages. RED on the base: the vector has no slot."""
    def group(vals):
        out = list(vals)
        if len(out) > 9:
            out[9] = min(out[9], -5000)
        return out

    assert _tick(monkeypatch, 1, group) == [5000]


def test_tick_tp0_puts_its_need_on_the_collective_and_no_need_no_yield(monkeypatch):
    seen = {}

    def group(vals):
        seen["vals"] = list(vals)
        return list(vals)

    assert _tick(monkeypatch, 0, group, take=7) == [7]
    assert len(seen["vals"]) == 10 and seen["vals"][9] == -7
    assert _tick(monkeypatch, 0, group, take=0) == []
