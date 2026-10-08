"""#1390f UNBACKED-DROP (y9d4d/B9, desk analysis 1390 Fix 2): dual D, full arena, waiting P.

Hermetic (fakes, no GPU): the pure proposal rule, the tree-side gate on the real
``UnifiedRadixCache._evict_device_leaf``, the tick's collective element (rank agreement), and the
D tick's use of the order (default OFF byte-identical; holds / demand / short wait never drop)."""

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.pdflip import dual_d_kv_stage as DK  # noqa: E402
from flliper.srt.pdflip import dual_d_unbacked_drop as UD  # noqa: E402

DUAL_D = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D", DK.MAX_TOKENS_ENV: "131072"}
ON = dict(DUAL_D, **{UD.ENV_NAME: "1"})


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for k in list(os.environ):
        if k.startswith("FLLIPER_PDFLIP_"):
            monkeypatch.delenv(k, raising=False)


def setenv(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)


# ---------------------------------------------------------------- gate / default off
def test_default_off_and_gate(monkeypatch):
    assert UD.switch_on() is False
    setenv(monkeypatch, DUAL_D)
    assert UD.switch_on() is False                      # dual D armed, switch unset = OFF
    monkeypatch.setenv(UD.ENV_NAME, "1")
    assert UD.switch_on() is True
    for bad in ({}, {"FLLIPER_PDFLIP_GROUP": "P"}, {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "P"},
                {"FLLIPER_PDFLIP_GROUP": "D", DK.MAX_TOKENS_ENV: "131072"},      # flip form (no dual layout)
                {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D"}):  # no D cap: not armed
        assert UD.switch_on(dict(bad, **{UD.ENV_NAME: "1"})) is False


def test_default_off_collective_unchanged(monkeypatch):
    actor = types.SimpleNamespace(_drop_propose=True)
    assert DK._drop_order_extra(actor) == []            # the old collective, element for element
    assert DK._drop_order_step(actor, [0] * 11, 5, True, 99.0, False) is False
    assert not hasattr(actor, "_arena_refused_t")
    setenv(monkeypatch, ON)
    assert DK._drop_order_extra(actor) == [-1]


# ---------------------------------------------------------------- proposal rule
def test_propose_table():
    f = UD.propose
    assert f(True, 9.0, 0, False, True, 8.0) is True
    assert f(False, 9.0, 0, False, True, 8.0) is False  # P does not wait
    assert f(True, 7.9, 0, False, True, 8.0) is False   # below the threshold
    assert f(True, 9.0, 1, False, True, 8.0) is False   # group demand
    assert f(True, 9.0, 0, True, True, 8.0) is False    # HOLD: a parked / W50 request is never dropped for
    assert f(True, 9.0, 0, False, False, 8.0) is False  # arena not refusing


# ---------------------------------------------------------------- tree-side gate (real _evict_device_leaf)
class FakeNode:
    def __init__(self, children=(), lock_ref=0, nid=1):
        self.children = dict.fromkeys(children, 1)
        self.lock_ref = lock_ref
        self.id = nid
        self.backuped = False


def make_tree(order, ongoing=None):
    from flliper.srt.mem_cache import unified_radix_cache as U

    t = types.SimpleNamespace(
        cache_controller=types.SimpleNamespace(write_policy="write_back"),
        ongoing_write_through=ongoing or {}, dropped=[], subtree=[])
    t._is_device_leaf = lambda n: True
    t.write_backup = lambda node, **kw: 0              # arena full: the backup is refused
    t.writing_check = lambda **kw: None
    t._ud_drop_unbacked_leaf = lambda node, tracker: (t.dropped.append(node), tracker.__setitem__(
        U.BASE_COMPONENT_TYPE, tracker.get(U.BASE_COMPONENT_TYPE, 0) + 100))
    t._ud_drop_unbacked_subtree = lambda node, sub, tracker: t.subtree.append(node)
    t._ud_dual_d_order_drop = types.MethodType(U.UnifiedRadixCache._ud_dual_d_order_drop, t)
    if order:
        setattr(t, UD.ORDER_ATTR, True)
    return t, U


def evict(t, U, node):
    tracker = {}
    U.UnifiedRadixCache._evict_device_leaf(t, node, tracker)
    return tracker


def test_tree_default_keeps_leaf(monkeypatch):
    t, U = make_tree(order=False)
    evict(t, U, FakeNode())
    assert t.dropped == []                              # today's behaviour: the leaf stays on the card


def test_tree_order_drops_unbacked_leaf():
    t, U = make_tree(order=True)
    tr = evict(t, U, FakeNode())
    assert len(t.dropped) == 1 and tr[U.BASE_COMPONENT_TYPE] == 100
    assert getattr(t, UD.STATS_ATTR) == [1, 100]


@pytest.mark.parametrize("node,ongoing", [
    (FakeNode(children=("c",)), None),                  # childless only (#841)
    (FakeNode(lock_ref=1), None),                       # a holder's lock: NEVER dropped
    (FakeNode(nid=7), {7: object()}),                   # write-through in flight
])
def test_tree_order_never_drops_held_or_inner(node, ongoing):
    t, U = make_tree(order=True, ongoing=ongoing)
    evict(t, U, node)
    assert t.dropped == []


def test_evict_with_order_clears_flag_and_reports():
    seen = []
    tree = types.SimpleNamespace()

    def ev(params):
        seen.append(getattr(tree, UD.ORDER_ATTR))
        UD.note_drop(tree, 40)
        UD.note_drop(tree, 2)
        raise_after = getattr(tree, "boom", False)
        if raise_after:
            raise RuntimeError("x")

    tree.evict = ev
    assert UD.evict_with_order(tree, object()) == (2, 42)
    assert seen == [True] and getattr(tree, UD.ORDER_ATTR) is False
    tree.boom = True
    with pytest.raises(RuntimeError):
        UD.evict_with_order(tree, object())
    assert getattr(tree, UD.ORDER_ATTR) is False        # cleared in finally


# ---------------------------------------------------------------- rank agreement
def test_three_ranks_one_order(monkeypatch):
    """Three D ranks with DIFFERENT local arena-refusal times / proposals see the SAME group value of the
    collective element and therefore the same order; the order is the MAX of the proposals."""
    setenv(monkeypatch, ON)
    clock = [100.0]
    monkeypatch.setattr(DK._pk, "_now", lambda: clock[0])
    actors = [types.SimpleNamespace(_drop_propose=p) for p in (False, True, False)]
    elems = [DK._drop_order_extra(a)[0] for a in actors]
    group = [min(elems)]                                # gmin of the negated values = MAX of the proposals
    g = [0] * 10 + group
    orders = [DK._drop_order_step(a, g, 0, True, 1.0, False) for a in actors]
    assert orders == [True, True, True]
    g0 = [0] * 10 + [min(DK._drop_order_extra(types.SimpleNamespace(_drop_propose=False)) * 3)]
    assert [DK._drop_order_step(a, g0, 0, True, 1.0, False) for a in actors] == [False] * 3


def test_proposal_needs_arena_refusal_then_expires(monkeypatch):
    setenv(monkeypatch, ON)
    clock = [100.0]
    monkeypatch.setattr(DK._pk, "_now", lambda: clock[0])
    a = types.SimpleNamespace()
    g = [0] * 11
    DK._drop_order_step(a, g, 0, True, 20.0, False)
    assert a._drop_propose is False                     # no arena refusal seen yet
    DK._drop_order_step(a, g, 64, True, 20.0, False)
    assert a._drop_propose is True
    clock[0] += UD.ARENA_RECENT_S + 1
    DK._drop_order_step(a, g, 0, True, 20.0, False)
    assert a._drop_propose is False                     # arena refusal too old
    DK._drop_order_step(a, g, 0, True, 20.0, True)      # holds
    assert a._drop_propose is False


# ---------------------------------------------------------------- the D tick end to end
def run_tick(monkeypatch, env, *, holds, demand, p_wait_ms, order_elem, arena_need=0):
    setenv(monkeypatch, env)
    calls = []
    from flliper.srt.pdflip import d_seat_vram as SV, dual_arena_spill as DAS
    from flliper.srt.pdflip import card_kv_ledger as CL

    monkeypatch.setattr(SV, "_air", lambda s: 0)
    monkeypatch.setattr(CL, "peek", lambda path: types.SimpleNamespace(demand={"P": 1}, pressure={}, pid={"P": 1}))
    monkeypatch.setattr(DK._pk, "max_live_id", lambda alloc, page: 0)
    monkeypatch.setattr(DK._pk, "phys_check", lambda *a, **k: None)
    monkeypatch.setattr(DK, "d_demand", lambda s: demand)
    monkeypatch.setattr(DK, "d_locked_rows", lambda s, a: 0)
    monkeypatch.setattr(DK, "d_avail_rows", lambda s, a: DK.NO_AVAIL)
    monkeypatch.setattr(DK, "publish_d_signal", lambda s, a: None)
    monkeypatch.setattr(DAS, "d_take_need", lambda s: arena_need)
    monkeypatch.setattr(DAS, "d_yield_arena", lambda s, n: None)
    monkeypatch.setattr(DK, "cache_yield", lambda sched, actor, live=None, drop_order=-1.0:
                        calls.append(drop_order) or 0)
    seen = {}

    def gmin(vals):
        seen["vals"] = list(vals)
        out = list(vals)
        if len(out) > 10:
            out[10] = order_elem
        return out

    actor = types.SimpleNamespace(
        gmin=gmin, mapped_tokens=1024, step=1024, _below=0, page=1, allocator=None,
        ledger=types.SimpleNamespace(path="x", clear_pressure=lambda: None), group_shrink=lambda *a: False,
        group_grow=lambda *a: None, _p_wait_since=1.0)
    monkeypatch.setattr(DK._pk, "_now", lambda: 1.0 + p_wait_ms / 1000.0)
    sched = types.SimpleNamespace(
        tp_rank=0, pdflip_d_parked=[object()] if holds else [], tree_cache=None,
        tp_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(**{DK.ACTOR_ATTR: actor})))
    DK.tick(sched)
    return calls, seen


def test_tick_default_off_never_orders(monkeypatch):
    calls, seen = run_tick(monkeypatch, DUAL_D, holds=False, demand=0, p_wait_ms=30000, order_elem=-1)
    assert calls == [-1.0]                              # plain cache_yield, as before
    assert len(seen["vals"]) == 10                      # the collective is the old one


def test_tick_on_orders_drop_when_idle_and_waiting(monkeypatch):
    calls, seen = run_tick(monkeypatch, ON, holds=False, demand=0, p_wait_ms=30000, order_elem=-1)
    assert len(seen["vals"]) == 11
    assert calls == [30.0]                              # order set, p_wait_s handed to the log


def test_tick_on_without_group_order_no_drop(monkeypatch):
    calls, _ = run_tick(monkeypatch, ON, holds=False, demand=0, p_wait_ms=30000, order_elem=0)
    assert calls == [-1.0]


def test_tick_holds_never_drop(monkeypatch):
    """A parked / W50-held request (holds=1): even with the group order on, nothing is dropped."""
    calls, _ = run_tick(monkeypatch, ON, holds=True, demand=0, p_wait_ms=30000, order_elem=-1)
    assert calls == [-1.0]


def test_tick_demand_no_drop(monkeypatch):
    calls, _ = run_tick(monkeypatch, ON, holds=False, demand=500, p_wait_ms=30000, order_elem=-1)
    assert calls == [-1.0]                              # only the existing LIVE yield runs: never the drop order


def test_log_line_keys_and_rate_limit(caplog):
    a = types.SimpleNamespace()
    with caplog.at_level(logging.INFO):
        UD.log_drop(a, 3, 34905, 12.5, 1.0, 100.0)
        UD.log_drop(a, 3, 34905, 12.5, 1.0, 101.0)      # inside the gap: suppressed
        UD.log_drop(a, 1, 10, 13.0, 1.0, 106.0)
    lines = [r.getMessage() for r in caplog.records if UD.MARK in r.getMessage()]
    assert len(lines) == 2
    assert "dropped_leaves=3 dropped_tok=34905 p_wait_s=12.5 arena_fill=1.000" in lines[0]
