"""Q-1190b DUAL ARENA-AUX-SPILL (27B NVFP4 dual, image int3 = 69cdc118a4, boot
dkr27bnvfp4dual262kbar1fs10061713, window hhe53y; two needles 190k + 258k at once from 17:25:54Z,
21 min without one token).

METAL. The shared KV arena stood at complete=718623 pinned=718623 of 720896 slots from 17:27:25Z to
the end, and every reference was a TREE reference ('ARENA-REF-HOLDERS pool=FULL tree=718621
tree_in_use=0 prefetch=0 queue=0 gap=0 own_held=718621' on every D rank; P PP0 'tree=674347 ...
gap=0'; refs 4146136 = 3 x 718621 + 674347 + 670251 + 645675 -- no stray, no release-queue leak).
D's yield found nothing: 'PKVWAIT-INSTR marker=spill_dyield nodes=17 n.aux_host_only=9
n.has_children=8 candidates=0' -- every leaf of D's tree is a finished request's END node with its
mamba anchor host-only, ``dual_arena_spill._reason`` refuses 'aux_host_only' unconditionally, and the
inner nodes never become leaves. P's V2 trim looped START -> 3 empty orders -> PAUSED -> RESUME 21x.

The fix (dual layout only, switch SGLANG_WEG2_DUAL_ARENA_AUX_SPILL_S, default 30 s, 0 = off): once the
wall has stood that long, the D yield (TP0's clock, a bit on the tick's collective) and the P trim
(PP0's clock, ``aux=1`` on the wire order) also take END-anchor leaves -- after every plain leaf, L3
copy of every KV page first, never a locked / in-flight / V1-held node.

Hermetic: the real C arena on a temp file, the real ArenaMHAHostPool, the real
``_weg2_direct_claim`` / ``_evict_host_leaf``; the mamba component is a recording fake (its host
free is the reference going back to its own arena). New names are looked up with getattr so the
tests fail on the base by BEHAVIOUR (the claim stays refused), not by an import error."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import logging  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402
import unittest.mock as mock  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "weg2"))
import test_dual_d_arena_yield_q1190_1004 as Y  # noqa: E402
import test_w3_arena_spill_0929 as W  # noqa: E402
import test_w3_dual_host_only_spill_q697c_1004 as Q  # noqa: E402

from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    ComponentType,
    EvictLayer,
)
from sglang.srt.weg2 import dual_arena_spill as DS  # noqa: E402
from sglang.srt.weg2 import dual_d_kv_stage as DK  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")

FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
AUX_ENV = getattr(DS, "AUX_ENV", "SGLANG_WEG2_DUAL_ARENA_AUX_SPILL_S")
AUX_FLAG = getattr(DS, "AUX_FLAG", 1 << 40)
AUX_MARK = getattr(DS, "AUX_MARK", "Q-1190b DUAL ARENA-AUX-SPILL")
DUAL_P, DUAL_D = Y.DUAL_P, Y.DUAL_D


def _env(monkeypatch, env, aux_s=None):
    Y._env(monkeypatch, env)
    monkeypatch.delenv(AUX_ENV, raising=False)
    if aux_s is not None:
        monkeypatch.setenv(AUX_ENV, str(aux_s))


@pytest.fixture(autouse=True)
def _fresh():
    saved = (dict(DS._N), dict(DS._Y))
    DS._Y.update(posts=0, yields=0, no_pool=0, quiet_until=0.0)
    getattr(DS, "_reset_aux_for_tests", lambda: None)()
    getattr(DS, "_reset_trim_for_tests", lambda: None)()
    yield
    getattr(DS, "_reset_aux_for_tests", lambda: None)()
    getattr(DS, "_reset_trim_for_tests", lambda: None)()
    DS._N.clear(); DS._N.update(saved[0])
    DS._Y.clear(); DS._Y.update(saved[1])


class _NoLRU:
    def in_list(self, node):
        return False


class _MambaComp:
    """The mamba component's host half: the free gives the anchor's reference back (recorded)."""

    component_type = MAMBA

    def __init__(self):
        self.freed = []

    def evict_component(self, node, target=EvictLayer.DEVICE):
        cd = node.component_data[MAMBA]
        if EvictLayer.HOST in target and cd.host_value is not None:
            self.freed.append(int(cd.host_value[0]))
            cd.host_value = None
            return 0, 1
        return 0, 0


def _with_mamba(t, nodes, anchors):
    """Give the tree a mamba component and host-only END anchors on ``anchors`` (node indices)."""
    mc = _MambaComp()
    t.components[MAMBA] = mc
    t._components_tuple = (t.components[FULL], mc)
    t.tree_components = (FULL, MAMBA)
    t.lru_lists[MAMBA] = _NoLRU()
    t.host_lru_lists[MAMBA] = _NoLRU()
    for k in anchors:
        nodes[k].component_data[MAMBA].host_value = torch.tensor([900 + k], dtype=torch.int64)
        nodes[k].component_data[MAMBA].value = None
    return mc


def _wedge(tmp_path):
    """The hhe53y shape: D's tree holds every slot, one chain whose leaf (and one inner node) carries
    a host-only END anchor -> the plain yield has no candidate at all; P's tree claims two pages."""
    tp, td, d_nodes, root, arena = Y._y9d1(tmp_path)
    mc = _with_mamba(td, d_nodes, anchors=(3, 7))
    return tp, td, d_nodes, root, arena, mc


def _p_claim(monkeypatch, tp, m, aux_s):
    _env(monkeypatch, DUAL_P, aux_s)
    return tp._weg2_direct_claim(m)


def _d_round(monkeypatch, td, aux_s, tp_rank=0):
    _env(monkeypatch, DUAL_D, aux_s)
    DS._Y["quiet_until"] = 0.0              # the 0.5-s backoff after an empty yield is not under test
    return Y._d_side(td, tp_rank)


# ------------------------------------------------------------------------------------ the defect, the fix
def test_hhe53y_wall_of_end_anchor_leaves_d_gives_them_after_the_wall_stood(tmp_path, monkeypatch, caplog):
    """RED on 69cdc118a4: D's yield finds candidates=0 on every round, P's claim stays refused for as long
    as the wall stands (21 min on the metal). GREEN: once the wall stood AUX_S, TP0's need carries the AUX
    bit, D gives the END-anchor leaves back (L3 copy first) and P's claim gets its room."""
    caplog.set_level(logging.INFO)
    tp, td, d_nodes, root, arena, mc = _wedge(tmp_path)
    m = W._claimer(tp)
    assert _p_claim(monkeypatch, tp, m, 0.2) is False                    # the metal line
    need, got = _d_round(monkeypatch, td, 0.2)
    assert need == 2 and got is not None and got["released"] == 0 and got["candidates"] == 0
    assert all(Q._in_tree(td, n) for n in d_nodes)                       # the stall, reproduced
    time.sleep(0.3)                                                      # the wall stands ...
    assert _p_claim(monkeypatch, tp, m, 0.2) is False                    # ... P keeps posting
    need, got = _d_round(monkeypatch, td, 0.2)
    assert got is not None and got["released"] >= 2, f"D still gave nothing back: {got}"
    pre = _p_claim(monkeypatch, tp, m, 0.2)
    assert pre is not False, f"P's claim still refused after the aux yield: {tp.refused}"
    assert pre is not None and int(pre.numel()) == 2
    gone = [n for n in d_nodes if not Q._in_tree(td, n)]
    assert d_nodes[7] in gone                                            # the END-anchor leaf went first
    assert 907 in mc.freed                                               # its anchor reference went back
    for n in gone:                                                       # #257: never without an L3 copy
        st = n.hash_value[0] + "_sfx"
        assert (root / (st + ".bin")).read_bytes() == bytes([0x40 + int(n.hash_value[0][1:])]) * W.PAGE
    msgs = [r.getMessage() for r in caplog.records]
    assert any(AUX_MARK in s and "ON side=D" in s for s in msgs)
    assert any("Q-1190 DUAL D-ARENA-YIELD" in s and "aux=1" in s for s in msgs)


def test_before_the_wall_stood_aux_s_nothing_changes(tmp_path, monkeypatch, caplog):
    """The normal case: a need younger than AUX_S is served exactly as before (here: nothing, the
    END-anchor leaves stay), no AUX line."""
    caplog.set_level(logging.INFO)
    tp, td, d_nodes, root, arena, mc = _wedge(tmp_path)
    m = W._claimer(tp)
    for _ in range(3):
        assert _p_claim(monkeypatch, tp, m, 3600) is False
        need, got = _d_round(monkeypatch, td, 3600)
        assert need == 2 and got["released"] == 0
    assert all(Q._in_tree(td, n) for n in d_nodes) and mc.freed == []
    assert not any(AUX_MARK in r.getMessage() for r in caplog.records)
    assert not any("aux=1" in r.getMessage() for r in caplog.records)


def test_switch_zero_is_the_old_refusal_forever(tmp_path, monkeypatch):
    tp, td, d_nodes, root, arena, mc = _wedge(tmp_path)
    m = W._claimer(tp)
    for _ in range(3):
        assert _p_claim(monkeypatch, tp, m, 0) is False
        time.sleep(0.05)
        need, got = _d_round(monkeypatch, td, 0)
        assert need == 2 and got["released"] == 0
    assert all(Q._in_tree(td, n) for n in d_nodes) and mc.freed == []


def test_the_aux_bit_reaches_every_d_rank_and_each_runs_the_same_pass(tmp_path, monkeypatch):
    """TP1 never reads the need file; the bit arrives with TP0's need through the collective. Two replica
    trees given the same flagged need give back the same leaves."""
    trees = []
    for i in range(2):
        sub = tmp_path / ("r%d" % i)
        sub.mkdir()
        trees.append(_wedge(sub))
    _env(monkeypatch, DUAL_D, 30)
    outs = [DS.d_yield_arena(types.SimpleNamespace(tree_cache=t[1], tp_rank=r), 2 | AUX_FLAG)
            for r, t in enumerate(trees)]
    assert outs[0]["released"] >= 2 and outs[0]["released"] == outs[1]["released"]
    keys = [[tuple(n.hash_value) for n in t[2] if not Q._in_tree(t[1], n)] for t in trees]
    assert keys[0] == keys[1] and keys[0]


def test_tick_carries_the_flagged_need_unchanged_to_tp1(monkeypatch):
    def group(vals):
        out = list(vals)
        if len(out) > 9:
            out[9] = min(out[9], -(5000 | AUX_FLAG))
        return out

    assert Y._tick(monkeypatch, 1, group) == [5000 | AUX_FLAG]


def test_the_clock_restarts_after_a_gap_without_need(tmp_path, monkeypatch):
    clock = getattr(DS, "_d_aux_clock", None)
    assert clock is not None, "no wall clock on the base"
    monkeypatch.setattr(DS, "AUX_GAP_S", 10.0)
    env = {AUX_ENV: "30"}
    assert clock(5, 100.0, env) is False                       # the wall starts
    for t in (108.0, 116.0, 124.0):                            # needs keep coming (gaps < 10 s)
        assert clock(5, t, env) is False
    assert clock(0, 129.0, env) is False                       # a tick without a need: no reset yet
    assert clock(5, 131.0, env) is True                        # stood 31 s
    assert clock(0, 145.0, env) is False                       # 14 s without a need: gone
    assert clock(5, 146.0, env) is False                       # a new wall, a new clock
    for t in (155.0, 164.0, 173.0):
        assert clock(5, t, env) is False
    assert clock(5, 177.0, env) is True
    assert clock(5, 200.0, env) is False                       # a 23-s hole is a new wall, too


# ------------------------------------------------------------------------------- what stays even with AUX
def test_aux_never_takes_a_page_without_l3_a_locked_an_in_flight_or_a_v1_held_end_anchor(tmp_path, monkeypatch):
    _env(monkeypatch, DUAL_D, 30)
    pd, arena, root, td, nodes = Q._host_only_tree(tmp_path, refuse=("a0_sfx",), chain=False)
    _with_mamba(td, nodes, anchors=range(8))                      # every leaf an END-anchor leaf
    nodes[1].component_data[MAMBA].host_lock_ref = 1             # a load reads the anchor
    td.ongoing_write_through[nodes[2].id] = object()              # pages not COMPLETE yet
    nodes[3]._weg2_end_anchor = True                              # V1: D may not have read it yet
    td._weg2_direct_mamba_rows = {nodes[4].id: object()}          # #1427 direct write in flight
    got = DS.d_yield_arena(types.SimpleNamespace(tree_cache=td), 100 | AUX_FLAG)
    for k in (0, 1, 2, 3, 4):
        assert Q._in_tree(td, nodes[k]), k
    assert not (root / "a0_sfx.bin").exists()
    assert [Q._in_tree(td, nodes[k]) for k in (5, 6, 7)] == [False] * 3
    assert got["aux_leaves"] == 3 and got["unsecured"] == 1


def test_plain_leaves_go_before_end_anchor_leaves(tmp_path, monkeypatch):
    _env(monkeypatch, DUAL_D, 30)
    pd, arena, root, td, nodes = Q._host_only_tree(tmp_path, chain=False)
    _with_mamba(td, nodes, anchors=(0, 1, 2, 3))                 # the oldest four carry anchors
    got = DS.spill_host_only(td, pd, 4, 1, set(), who="dyield", allow_aux=True)
    assert got["released"] == 4 and got["aux_leaves"] == 0
    assert [Q._in_tree(td, n) for n in nodes] == [True] * 4 + [False] * 4
    got = DS.spill_host_only(td, pd, 2, 1, set(), who="dyield", allow_aux=True)
    assert got["aux_leaves"] == 2 and [Q._in_tree(td, n) for n in nodes[:4]] == [False, False, True, True]


def test_without_allow_aux_the_spill_is_the_old_one(tmp_path, monkeypatch):
    _env(monkeypatch, DUAL_D, 30)
    pd, arena, root, td, nodes = Q._host_only_tree(tmp_path, chain=False)
    _with_mamba(td, nodes, anchors=(0, 1, 2, 3))
    got = DS.spill_host_only(td, pd, 100, 1, set(), who="dyield")
    assert got == {"released": 4, "leaves": 4, "candidates": 4, "unsecured": 0, "braked": 0}
    assert [Q._in_tree(td, n) for n in nodes] == [True] * 4 + [False] * 4


# ------------------------------------------------------------------------------------------- group P (trim)
import test_dual_arena_trim_v2_1004 as V  # noqa: E402


def _p_stage(monkeypatch, n, aux_s):
    V._env(monkeypatch, DUAL_P)
    monkeypatch.delenv(AUX_ENV, raising=False)
    monkeypatch.setenv(AUX_ENV, str(aux_s))
    sched, cache, pool, nodes = V._stage(n)
    return sched, cache, pool, nodes


def _anchor_all(cache, nodes):
    """Every P leaf an END-anchor leaf (mamba host-only; the host free is recorded, see _recording_evict)."""
    for n in nodes:
        n.component_data[MAMBA].host_value = torch.tensor([7], dtype=torch.int64)
        n.component_data[MAMBA].value = None


def test_p_trim_orders_carry_aux_only_after_the_wall_stood(monkeypatch, caplog):
    """RED on the base: the trim over END-anchor leaves gives nothing, START -> PAUSED forever (hhe53y:
    21x). GREEN: PP0's orders after AUX_S carry aux=1 and every stage gives the leaves back."""
    caplog.set_level(logging.INFO)
    sched, cache, pool, nodes = _p_stage(monkeypatch, 97, 30)
    freed = []
    with mock.patch.object(type(cache), "_evict_component_and_detach_lru",
                           _recording_evict(type(cache)._evict_component_and_detach_lru, freed)):
        _anchor_all(cache, nodes)
        t, cmds = 100.0, []
        for _ in range(30):                                       # 60 s of passes, 2 s apart
            cmds.append(V._tick(sched, t))
            t += 2.0
    issued = [c for c in cmds if c is not None]
    assert issued, "PP0 never ordered"
    assert all(int(getattr(c, "aux", 0)) == 0 for c in issued[:1])   # the first order: no aux yet
    assert any(int(getattr(c, "aux", 0)) == 1 for c in issued), "no aux order after 60 s of wall"
    assert V._fill(pool) <= 0.80 + 1e-9, "the arena stayed at %.2f" % V._fill(pool)
    assert any(AUX_MARK in r.getMessage() and "ON side=P" in r.getMessage() for r in caplog.records)


def _recording_evict(orig, freed):
    """The fixture's mamba host pool knows no row 7 -- record the mamba host free instead of running it."""
    def _evict(self, node, comp, target=EvictLayer.DEVICE, tracker=None):
        if comp.component_type == MAMBA and EvictLayer.HOST in target:
            cd = node.component_data[MAMBA]
            if cd.host_value is not None and cd.value is None:
                freed.append(node.id)
                cd.host_value = None
                return 0, 1
        return orig(self, node, comp, target=target, tracker=tracker)
    return _evict


def test_p_trim_switch_zero_never_orders_aux(monkeypatch):
    sched, cache, pool, nodes = _p_stage(monkeypatch, 97, 0)
    _anchor_all(cache, nodes)
    t, cmds = 100.0, []
    for _ in range(30):
        cmds.append(V._tick(sched, t))
        t += 2.0
    assert all(int(getattr(c, "aux", 0)) == 0 for c in cmds if c is not None)
    assert all(n.id in V._live(cache) for n in nodes)


def test_p_clock_clears_at_or_below_hi(monkeypatch):
    clock = getattr(DS, "_p_aux_clock", None)
    assert clock is not None, "no P wall clock on the base"
    env = {AUX_ENV: "30"}
    assert clock(0.99, 0.0, 0.90, env) is False
    assert clock(0.99, 31.0, 0.90, env) is True
    assert clock(0.90, 32.0, 0.90, env) is False                 # at HI: the wall is gone
    assert clock(0.99, 33.0, 0.90, env) is False                 # a new wall
    assert clock(None, 40.0, 0.90, env) is False
    assert clock(0.99, 41.0, 0.90, env) is False


def test_off_the_gate_no_clock_runs_and_no_bit_is_set(tmp_path, monkeypatch):
    """Flip / 27B INT8 / NF / half-set gates: d_take_need reads nothing (0), the trim decides nothing."""
    for env in Y.NOT_DUAL:
        _env(monkeypatch, env, 0.0001)
        sub = tmp_path / ("e%d" % len(os.listdir(tmp_path)))
        sub.mkdir()
        tp, td, d_nodes, root, arena, mc = _wedge(sub)
        assert DS.post_need(td._weg2_direct_pool(), 5) is False, env      # nothing written off the gate
        assert DS.d_take_need(types.SimpleNamespace(tree_cache=td)) == 0, env
        assert getattr(DS, "_A", {}).get("d_since") is None, env
