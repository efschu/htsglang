# SPDX-License-Identifier: Apache-2.0
"""#1500i PKVWAIT-INSTR (y9d4d, desk analysis 1390 Fix 1): the log-only census of "why does a full-arena D
hand no VRAM to a waiting P".

PINNED
* the census of ``spill_host_only``: one leaf per refusal reason -> the exact ``n.<reason>`` / ``pg.<reason>``
  numbers, the L3 split of the unsecured leaves (lost / short), the candidates; the line carries the fixed
  prefix and key=value fields;
* NO BEHAVIOUR CHANGE: ``_eligible`` is ``_reason is None``; two identical trees spilled with the instrument
  on and off end identical (return value, surviving nodes);
* the gate: no dual layout (flip / NF / INT8 / half-set / not capped), or the switch at 0 -> no line, no
  state, no census pass (``_host_pages`` is never asked); the default is ON under the dual layout;
* rate limit: one line per marker per 5 s, the skipped calls come back as ``suppressed=n``;
* D's cache yield: ``ev_before`` / ``ev_after`` and the device-leaf census (un-backed / backed / locked);
* the owner of the topmost live row (request / tree node / none_found / error) in the SHRINK-BLOCKED path,
  with every blocking condition printed on its own;
* D-ARENA-YIELD reaches the census with ``who=dyield``.
All hermetic: hand-built trees and pools (real ``UnifiedTreeNode``), no arena file, no GPU.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode
from sglang.srt.weg2 import dual_arena_spill as D
from sglang.srt.weg2 import dual_d_kv_stage as DK
from sglang.srt.weg2 import dual_p_kv_stage as PK
from sglang.srt.weg2 import dual_pkvwait_instr as PI

FULL = ComponentType.FULL
GATE_KEYS = ("SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP", PK.MAX_TOKENS_ENV, DK.MAX_TOKENS_ENV, PI.ENV_NAME)
DUAL_P = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P", PK.MAX_TOKENS_ENV: "131072"}
DUAL_D = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D", DK.MAX_TOKENS_ENV: "262144"}
NOT_DUAL = [
    {},
    {"SGLANG_WEG2_GROUP": "P"},
    {"SGLANG_WEG2_DUAL_LAYOUT": "1"},
    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"},           # no P KV cap: not armed
    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"},           # no D KV cap: not armed
]


def _env(monkeypatch, env):
    for k in GATE_KEYS:
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    # dual_arena_spill keeps process-global counters that gate its OWN log lines (``k <= 8`` of
    # "Q-697c DUAL HOST-ONLY SPILL", the D-yield backoff ``quiet_until``): the calls of this file must not
    # use them up for the neighbour files that run in the same process (test_w3_dual_host_only_spill).
    saved = (dict(D._N), dict(D._Y))
    PI._reset_for_tests()
    for k in GATE_KEYS:
        monkeypatch.delenv(k, raising=False)
    yield
    PI._reset_for_tests()
    D._N.clear()
    D._N.update(saved[0])
    D._Y.clear()
    D._Y.update(saved[1])


def _lines(caplog, marker=None):
    out = []
    for r in caplog.records:
        m = r.getMessage()
        if m.startswith(PI.MARK) and (marker is None or ("marker=%s " % marker) in m):
            out.append(m)
    return out


def _kv(line):
    return dict(tok.split("=", 1) for tok in line.split()[len(PI.MARK.split()):])


# ---------------------------------------------------------------- hand-built tree and pool
class _Cache:
    """What ``spill_host_only`` / ``_reason`` ask of the tree; ``_is_host_leaf`` is the REAL one."""

    _is_host_leaf = UnifiedRadixCache._is_host_leaf
    tree_components = (FULL,)

    def __init__(self):
        self.root_node = UnifiedTreeNode((FULL,))
        self.ongoing_write_through = {}
        self._components_tuple = (types.SimpleNamespace(component_type=FULL),)
        self.nodes = []

    def _collect_all_nodes(self):
        return [self.root_node] + list(self.nodes)

    def _evict_host_leaf(self, node, tracker):
        cd = node.component_data[FULL]
        tracker[FULL] = int(cd.host_value.numel())
        cd.host_value = None
        for k, c in list(node.parent.children.items()):
            if c is node:
                del node.parent.children[k]
        self.nodes.remove(node)


class _Pool:
    staging_rows = 1

    def __init__(self, lost=(), short=()):
        self.lost, self.short = set(lost), set(short)

    def secure_rows_to_l3(self, hv):
        k = int(hv[0])
        if k in self.lost:
            return {"lost": 1, "pages": 1, "on_disk": 0, "written": 0}
        if k in self.short:
            return {"lost": 0, "pages": 0, "on_disk": 0, "written": 0}
        return {"lost": 0, "pages": 1, "on_disk": 0, "written": 1}


def _host_node(cache, k, parent=None, rows=None, hashed=True):
    n = UnifiedTreeNode((FULL,))
    n.parent = parent or cache.root_node
    n.parent.children[("k%d" % k,)] = n
    n.hash_value = ["h%d" % k] if hashed else None
    n.component_data[FULL].value = None
    n.component_data[FULL].host_value = torch.tensor(rows if rows is not None else [k, k + 1], dtype=torch.int64)
    cache.nodes.append(n)
    return n


def _refusal_tree():
    """One leaf per refusal reason plus three candidates (A ok, B lost, C short)."""
    c = _Cache()
    t = types.SimpleNamespace()
    t.A = _host_node(c, 10)
    t.B = _host_node(c, 20)
    t.C = _host_node(c, 30)
    t.dev = _host_node(c, 40)
    t.dev.component_data[FULL].value = torch.arange(3, dtype=torch.int64)            # device resident
    t.locked = _host_node(c, 50)
    t.locked.component_data[FULL].host_lock_ref = 1
    t.parent = _host_node(c, 60)
    t.child = _host_node(c, 61, parent=t.parent, hashed=False)                       # leaf, but no hash
    t.ongoing = _host_node(c, 70)
    c.ongoing_write_through[t.ongoing.id] = object()
    t.blocked = _host_node(c, 80)
    t.blocked.write_through_pending_id = 7
    t.staging = _host_node(c, 90, rows=[0, 5])                                       # min row < staging_rows
    t.skipped = _host_node(c, 100)
    t.empty = _host_node(c, 110, rows=[])                                            # zero host rows
    return c, t


# ---------------------------------------------------------------- census
def test_spill_census_names_every_refusal_reason_exactly(monkeypatch, caplog):
    _env(monkeypatch, DUAL_P)
    caplog.set_level(logging.INFO)
    c, t = _refusal_tree()
    got = D.spill_host_only(c, _Pool(lost=(20,), short=(30,)), 100, 1, {id(t.skipped)}, who="trim")
    assert got["released"] == 2 and got["leaves"] == 1 and got["candidates"] == 3 and got["unsecured"] == 2
    ls = _lines(caplog, "spill_trim")
    assert len(ls) == 1, ls
    assert ls[0].startswith("#1500i PKVWAIT-INSTR marker=spill_trim ")
    f = _kv(ls[0])
    assert f["who"] == "trim" and f["want"] == "100" and f["released_pages"] == "2"
    assert f["candidates"] == "3" and f["cand_pages"] == "6" and f["nodes"] == "12"
    assert f["unsecured"] == "2" and f["unsec_lost"] == "1" and f["unsec_short"] == "1"
    assert f["stale_pop"] == "0" and f["braked"] == "0" and f["suppressed"] == "0"
    one = ["device_resident", "host_locked", "has_children", "no_hash", "write_through_ongoing",
           "blocked_wt_pending_id", "staging", "skip", "no_host_value"]
    for r in one:
        assert f["n." + r] == "1", (r, f)
    assert f["pg.device_resident"] == "2" and f["pg.has_children"] == "2" and f["pg.no_host_value"] == "0"
    assert f["pg.staging"] == "2" and f["pg.skip"] == "2"
    assert not any(k.startswith("n.") and k[2:] not in one for k in f), f


def test_eligible_is_reason_none_and_the_instrument_changes_nothing(monkeypatch):
    c, t = _refusal_tree()
    pool = _Pool()
    skip = {id(t.skipped)}
    for n in c._collect_all_nodes():
        assert D._eligible(c, n, pool, skip) == (D._reason(c, n, pool, skip) is None)
    results = []
    for env in (DUAL_P, dict(DUAL_P, **{PI.ENV_NAME: "0"})):
        _env(monkeypatch, env)
        PI._reset_for_tests()
        c, t = _refusal_tree()
        got = D.spill_host_only(c, _Pool(lost=(20,), short=(30,)), 100, 1, {id(t.skipped)})
        left = sorted(name for name, n in vars(t).items() if n in c.nodes)
        results.append((got, left))
    assert results[0] == results[1]


def test_blocked_kinds_and_end_anchor_are_named(monkeypatch):
    c = _Cache()
    n = _host_node(c, 10)
    n.write_through_pending_id = 3
    assert D._reason(c, n, _Pool(), set()) == "blocked_wt_pending_id"
    n.write_through_pending_id = None
    c._weg2_direct_mamba_rows = {n.id: 1}
    assert D._reason(c, n, _Pool(), set()) == "blocked_direct_mamba_rows"
    c._weg2_direct_mamba_rows = {}
    assert D._reason(c, n, _Pool(), set()) is None


# ---------------------------------------------------------------- gate
@pytest.mark.parametrize("env", NOT_DUAL)
def test_off_the_dual_layout_nothing_is_counted_or_logged(monkeypatch, caplog, env):
    _env(monkeypatch, env)
    caplog.set_level(logging.INFO)

    def _boom(*a, **k):
        raise AssertionError("the census ran off the gate")

    monkeypatch.setattr(D, "_host_pages", _boom)
    monkeypatch.setattr(PI, "count", _boom)
    c, t = _refusal_tree()
    got = D.spill_host_only(c, _Pool(), 100, 1, set())
    assert got["released"] >= 2
    assert not _lines(caplog)
    assert PI._S["last"] == {} and PI._S["suppressed"] == {} and PI._S["emitted"] == {}
    assert PI.begin("anything") is None
    sched = types.SimpleNamespace(tree_cache=_YTree())
    DK.cache_yield(sched, types.SimpleNamespace(mapped_tokens=1))
    DK._instr_shrink_blocked(sched, types.SimpleNamespace(step=4, mapped_tokens=8, page=1), "holds", 0, 5, 5,
                             1.0, True, False, False, 0)
    assert not _lines(caplog)


def test_switch_off_is_silent_and_the_default_is_on(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    _env(monkeypatch, DUAL_P)
    assert PI.enabled() and PI.switch_on()
    _env(monkeypatch, dict(DUAL_D, **{PI.ENV_NAME: "0"}))
    assert not PI.enabled() and PI.begin("x") is None
    c, t = _refusal_tree()
    D.spill_host_only(c, _Pool(), 100, 1, set())
    assert not _lines(caplog)
    # the env-dict form (the one the gates of the neighbours use)
    assert PI.enabled(DUAL_D) and not PI.enabled(dict(DUAL_D, **{PI.ENV_NAME: "0"})) and not PI.enabled({})


def test_environ_registers_the_switch_default_on():
    from sglang.srt.environ import envs

    assert envs.SGLANG_WEG2_DUAL_PKVWAIT_INSTR.get() is True


# ---------------------------------------------------------------- rate limit
def test_one_line_per_marker_per_five_seconds_and_the_skips_are_counted(monkeypatch):
    _env(monkeypatch, DUAL_P)
    assert PI.begin("m", now=100.0) == 0
    assert PI.begin("m", now=101.0) is None
    assert PI.begin("m", now=104.9) is None
    assert PI.begin("other", now=101.0) == 0            # another marker has its own slot
    assert PI.begin("m", now=105.1) == 2                # the two skipped calls are reported
    assert PI.begin("m", now=106.0) is None
    assert PI.begin("m", now=110.2) == 1


def test_line_format_is_prefix_marker_then_key_value():
    line = PI.format_line("x", [("a", 1), ("b", 0.5), ("c", "u v"), ("d", "")], 3)
    assert line == "#1500i PKVWAIT-INSTR marker=x a=1 b=0.500 c=u_v d=- suppressed=3"


def test_spill_census_is_rate_limited_per_caller(monkeypatch, caplog):
    _env(monkeypatch, DUAL_P)
    caplog.set_level(logging.INFO)
    for _ in range(3):
        c, t = _refusal_tree()
        D.spill_host_only(c, _Pool(), 100, 1, set(), who="trim")
    c, t = _refusal_tree()
    D.spill_host_only(c, _Pool(), 100, 1, set(), who="dyield")        # another marker: its own slot
    assert len(_lines(caplog, "spill_trim")) == 1 and len(_lines(caplog, "spill_dyield")) == 1
    assert PI._S["suppressed"]["spill_trim"] == 2
    PI._S["last"]["spill_trim"] -= 10.0
    c, t = _refusal_tree()
    D.spill_host_only(c, _Pool(), 100, 1, set(), who="trim")
    assert _kv(_lines(caplog, "spill_trim")[-1])["suppressed"] == "2"


# ---------------------------------------------------------------- D-ARENA-YIELD reaches the census
def test_d_arena_yield_prints_the_census_with_who_dyield(monkeypatch, caplog):
    _env(monkeypatch, DUAL_D)
    caplog.set_level(logging.INFO)
    c, t = _refusal_tree()
    c.page_size = 1
    pool = _Pool()
    pool.arena = types.SimpleNamespace(stats=lambda: {"complete": 7, "slots": 9})
    c._weg2_direct_pool = lambda: pool
    got = D.d_yield_arena(types.SimpleNamespace(tree_cache=c), 10)
    assert got["candidates"] == 4                      # A, B, C and `skipped` (D's yield passes no skip set)
    ls = _lines(caplog, "spill_dyield")
    assert len(ls) == 1
    f = _kv(ls[0])
    assert f["who"] == "dyield" and f["want"] == str(D.D_YIELD_MIN_PAGES) and f["candidates"] == "4"
    assert f["arena_complete"] == "7" and f["arena_slots"] == "9"


def test_candidates_zero_is_explained_by_the_reasons(monkeypatch, caplog):
    """The y9d4d shape: nothing is a candidate -- the line says what everything was instead."""
    _env(monkeypatch, DUAL_D)
    caplog.set_level(logging.INFO)
    c = _Cache()
    for k in (10, 20, 30):
        _host_node(c, k).component_data[FULL].host_lock_ref = 1
    for k in (40, 50):
        _host_node(c, k).component_data[FULL].value = torch.arange(2, dtype=torch.int64)
    got = D.spill_host_only(c, _Pool(), 100, 1, set(), who="dyield")
    assert got["candidates"] == 0
    f = _kv(_lines(caplog, "spill_dyield")[0])
    assert f["candidates"] == "0" and f["n.host_locked"] == "3" and f["n.device_resident"] == "2"
    assert f["nodes"] == "5"


# ---------------------------------------------------------------- D's cache yield
class _YTree:
    def __init__(self):
        self.ev = 150116
        self.evictable_device_leaves = set()
        self.evicts = 0

    def evictable_size(self):
        return self.ev

    def protected_size(self):
        return 55

    def evict(self, params):
        self.evicts += 1
        self.ev = 119710          # the y9d4d number: the evict did not free what was evictable


def _dev_leaf(tokens, backed, lock=0):
    n = UnifiedTreeNode((FULL,))
    cd = n.component_data[FULL]
    cd.value = torch.arange(tokens, dtype=torch.int64)
    cd.host_value = torch.arange(tokens, dtype=torch.int64) if backed else None
    cd.lock_ref = lock
    return n


def test_cache_yield_prints_ev_before_after_and_the_device_leaf_census(monkeypatch, caplog):
    _env(monkeypatch, DUAL_D)
    caplog.set_level(logging.INFO)
    tree = _YTree()
    tree.evictable_device_leaves = {_dev_leaf(3, False, lock=1), _dev_leaf(5, False), _dev_leaf(7, True)}
    actor = types.SimpleNamespace(mapped_tokens=155648)
    got = DK.cache_yield(types.SimpleNamespace(tree_cache=tree), actor, live=(155645, 36.0))
    assert got == 150116 and tree.evicts == 1                      # the yield itself is unchanged
    f = _kv(_lines(caplog, "cache_yield")[0])
    assert f["live"] == "1" and f["ev_before"] == "150116" and f["ev_after"] == "119710"
    assert f["protected"] == "55" and f["mapped"] == "155648"
    assert f["dev_leaves"] == "3" and f["dev_unbacked"] == "2" and f["dev_unbacked_tok"] == "8"
    assert f["dev_backed"] == "1" and f["dev_backed_tok"] == "7" and f["dev_locked"] == "1"
    # the rate limit: the next yield inside 5 s prints nothing, and is counted
    tree.ev = 150116
    DK.cache_yield(types.SimpleNamespace(tree_cache=tree), actor)
    assert len(_lines(caplog, "cache_yield")) == 1 and PI._S["suppressed"]["cache_yield"] == 1


def test_cache_yield_without_evictable_stays_zero_and_silent(monkeypatch, caplog):
    _env(monkeypatch, DUAL_D)
    caplog.set_level(logging.INFO)
    tree = _YTree()
    tree.ev = 0
    assert DK.cache_yield(types.SimpleNamespace(tree_cache=tree), types.SimpleNamespace(mapped_tokens=1)) == 0
    assert tree.evicts == 0 and not _lines(caplog)


# ---------------------------------------------------------------- owner of the topmost live row
def _sched(top_row=155645, parked=(), running=(), tree=None):
    r2t = torch.zeros((4, 16), dtype=torch.int64)
    r2t[0, :3] = torch.tensor([5, 6, top_row], dtype=torch.int64)
    return types.SimpleNamespace(
        req_to_token_pool=types.SimpleNamespace(req_to_token=r2t),
        running_batch=types.SimpleNamespace(reqs=list(running)), chunked_req=None, waiting_queue=[],
        weg2_d_parked=list(parked), tree_cache=tree)


def _req(rid="weg2-0-119", idx=0):
    return types.SimpleNamespace(rid=rid, req_pool_idx=idx, origin_input_ids=[1, 2, 3], output_ids=[])


class _OTree:
    def __init__(self, nodes):
        self.root_node = UnifiedTreeNode((FULL,))
        self.nodes = nodes
        for n in nodes:
            n.parent = self.root_node

    def _collect_all_nodes(self):
        return [self.root_node] + self.nodes


def test_owner_is_a_request_when_a_request_holds_the_row():
    out = PI.top_row_owner(_sched(parked=[_req()]), 155645, 1)
    assert out["owner"] == "req" and out["req_state"] == "parked" and out["rid"] == "weg2-0-119"
    out = PI.top_row_owner(_sched(running=[_req("r1")]), 155645, 1)
    assert out["owner"] == "req" and out["req_state"] == "running" and out["rid"] == "r1"


def test_owner_is_a_tree_node_with_its_state_when_no_request_holds_it():
    node = _dev_leaf(2, False, lock=2)
    node.component_data[FULL].value = torch.tensor([155644, 155645], dtype=torch.int64)
    other = _dev_leaf(2, True)
    sched = _sched(tree=_OTree([other, node]))
    out = PI.top_row_owner(sched, 155645, 1)
    assert out["owner"] == "tree" and out["top_lock"] == 2 and out["top_backuped"] == 0
    assert out["top_node_id"] == node.id and out["top_children"] == 0 and out["top_tok"] == 2


def test_owner_none_found_and_error_are_named():
    sched = _sched(tree=_OTree([_dev_leaf(2, True)]))
    out = PI.top_row_owner(sched, 999999, 1)
    assert out["owner"] == "none_found" and out["nodes_scanned"] == 1

    class _Bad:
        @property
        def req_to_token_pool(self):
            raise RuntimeError("boom")

    assert PI.top_row_owner(_Bad(), 5, 1)["owner"] == "err:RuntimeError"


def test_shrink_blocked_line_prints_every_condition_and_the_owner(monkeypatch, caplog):
    _env(monkeypatch, DUAL_D)
    caplog.set_level(logging.INFO)
    sched = _sched(parked=[_req()])
    actor = types.SimpleNamespace(step=4096, mapped_tokens=155648, page=1)
    DK._instr_shrink_blocked(sched, actor, "holds", 0, 155645, 155645, 36.0, True, False, False, 12345)
    f = _kv(_lines(caplog, "shrink_blocked")[0])
    assert f["reason"] == "holds" and f["mapped"] == "155648" and f["floor"] == "155645"
    assert f["floor_blocks"] == "1" and f["floor_is_local"] == "1" and f["live_row"] == "155645"
    assert f["holds"] == "1" and f["parked"] == "1" and f["p_missing"] == "0" and f["regrow_hold"] == "0"
    assert f["p_wait_s"] == "36.000" and f["avail_min"] == "12345"
    assert f["owner"] == "req" and f["req_state"] == "parked" and f["rid"] == "weg2-0-119"


def test_shrink_blocked_reason_none_is_printed_as_none(monkeypatch, caplog):
    _env(monkeypatch, DUAL_D)
    caplog.set_level(logging.INFO)
    sched = _sched()
    actor = types.SimpleNamespace(step=4096, mapped_tokens=155648, page=1)
    DK._instr_shrink_blocked(sched, actor, None, 0, 100, 5, 1.0, False, False, False, 0)
    f = _kv(_lines(caplog, "shrink_blocked")[0])
    assert f["reason"] == "none" and f["floor_blocks"] == "0" and f["floor_is_local"] == "0"
    assert f["owner"] == "none_found"


def test_shrink_blocked_reason_function_is_unchanged():
    kw = dict(mapped=155648, need=0, floor=155645, step=4096, p_missing=False, regrow_hold=False)
    assert DK.shrink_blocked_reason(holds=True, **kw) == "holds"
    assert DK.shrink_blocked_reason(holds=False, **kw) == "live_floor"
