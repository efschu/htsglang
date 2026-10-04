"""TOLD-ANCHOR-HOLD (dual y9d4 09:34:14Z, b73e808a0f): between a request's told and its
admission nothing in the tree held the anchor the told names. Metal chain, rid weg2-0-77:

  PP0 told=22528 / PP1+PP2 "#1400 FOLLOWER SATISFIED LOCALLY told=22528"  (09:34:10)
  END-ANCHOR weg2-0-72 anchor=26955 on all three ranks -> WEG2 PATH-CAP cap=4 (09:34:10)
  PP1/PP2 "#1042 EXTENT set extent=16665" (09:34:11): the next END anchor up the path
  09:34:14 PP1 loads 16665, "#968 PREFIX MATERIALISATION SHORTFALL" deficit 22528.

Hermetic CPU: real UnifiedRadixCache methods on real UnifiedTreeNode chains, a stub arena pool whose
slot references are the tree's host values (a tree that gave its reference back makes the slot
droppable -- what ARENA-DROP stage i takes), three ranks fed the same sequence.

RED-FIRST: every ``*_without_hold`` test is the chain on the old way (switch 0 == no hold object,
the b73e808a0f behaviour): the admission lands on 16665 < told. The ``*_hold`` tests are the same
sequence with the hold: it lands on told.
"""
from __future__ import annotations

import logging
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from sglang.srt.mem_cache import unified_radix_cache as URC  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode  # noqa: E402
from sglang.srt.weg2 import dual_told_anchor_hold as TAH  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

M = ComponentType.MAMBA
TC = (ComponentType.FULL, ComponentType.MAMBA)
T = 22528            # PP0's told for weg2-0-77
END_OLD = 16665      # weg2-0-3's END anchor -- what the admission fell back to on y9d4
TAIL = 26955         # weg2-0-72's END anchor, the insert that runs the cap


@pytest.fixture(autouse=True)
def dual_p_env(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.setenv("SGLANG_WEG2_MAMBA_MAX_STATES_PER_PATH", "4")
    monkeypatch.delenv(TAH.ENV, raising=False)
    monkeypatch.delenv(TAH.ENV_MAX, raising=False)
    monkeypatch.delenv(TAH.ENV_RUNS, raising=False)
    monkeypatch.setattr(TAH, "_ACTIVE", None)
    yield


class _Pool:
    """Stub mamba arena pool: slots 0..99 are arena ids; a slot is REFERENCED while a tree node's
    host value names it; an unreferenced COMPLETE slot is what a claim-time drop may take."""

    def __init__(self):
        self.nodes = []
        self.dropped = set()

    def is_arena_id(self, i):
        return 0 <= int(i) < 100

    def settled_anchor_slots(self, hv):
        return [int(x) for x in hv.tolist()]

    def referenced(self):
        return {int(n.component_data[M].host_value[0]) for n in self.nodes
                if n.component_data[M].host_value is not None}

    def arena_drop(self, want):
        """stage i: unreferenced slots only"""
        free = [s for s in range(100) if s in self._all() and s not in self.referenced()
                and s not in self.dropped]
        take = free[:want]
        self.dropped.update(take)
        return take

    def _all(self):
        return {int(n.component_data[M].host_value[0]) for n in self.nodes
                if n.component_data[M].host_value is not None} | self._released

    _released = set()


class _Mc:
    def evict_component(self, node, target=None):
        node.component_data[M].host_value = None


class Rank:
    """One group-P rank: a tree with the 27B path of y9d4 after the earlier caps ran:
    END(16665) - 22528 - 24000 - 25000 - [tail END 26955 inserted by the test]."""

    def __init__(self, rid_told=None):
        self.told = {} if rid_told is None else rid_told
        self.pool = _Pool()
        self.pool._released = set()
        c = object.__new__(UnifiedRadixCache)
        c.root_node = UnifiedTreeNode(TC)
        c.components = {M: _Mc()}
        c.cache_controller = None
        c._weg2_cap_tail = None
        c._weg2_direct_mamba_rows = {}
        c._weg2_cap_deferred = {}
        c._evict_component_and_detach_lru = self._evict
        c._update_evictable_leaf_sets = lambda n: None
        c._weg2_mamba_pool = lambda: self.pool
        self.c = c
        self.nodes = {}
        parent, slot = c.root_node, 1
        for depth, end in ((END_OLD, True), (T, False), (24000, False), (25000, False)):
            parent = self._add(parent, depth, slot, end)
            slot += 1

    def _evict(self, node, comp, target=None, tracker=None):
        cd = node.component_data[M]
        if cd.host_value is not None:
            self.pool._released.add(int(cd.host_value[0]))
        cd.host_value = None
        cd.value = None
        return 0, 1

    def _add(self, parent, depth, slot, end):
        n = UnifiedTreeNode(TC)
        n.key = [0] * (depth - self._depth(parent))
        n.parent = parent
        parent.children[depth] = n
        n._weg2_anchored = True
        n._weg2_end_anchor = end
        n.component_data[M].host_value = torch.tensor([slot], dtype=torch.int64)
        self.pool.nodes.append(n)
        self.nodes[depth] = n
        return n

    @staticmethod
    def _depth(n):
        d = 0
        while n.parent is not None:
            d += len(n.key)
            n = n.parent
        return d

    def attach(self):
        return TAH.attach(self.c, self.told)

    def insert_tail_end_anchor(self):
        """weg2-0-72's END anchor insert: the node is anchored, the cap runs on it."""
        tail = self._add(self.nodes[25000], TAIL, 9, True)
        self.c._weg2_cap_tail = tail
        return self.c._weg2_cap_after_insert()

    def admit(self, told):
        """The admission's match: the deepest usable resume anchor not past `told`."""
        best = 0
        for depth, n in sorted(self.nodes.items()):
            if depth > told:
                break
            if (n.component_data[M].host_value is not None and not getattr(n, "_weg2_capped", False)
                    and n._weg2_anchored):
                best = depth
        return best

    def state(self):
        return {d: (bool(getattr(n, "_weg2_capped", False)), n.component_data[M].host_value is not None)
                for d, n in sorted(self.nodes.items())}


def _ranks(n=3, hold=True):
    out = [Rank({"weg2-0-77": T}) for _ in range(n)]
    if hold:
        for r in out:
            r.attach()
    return out


# ---------------------------------------------------------------------------------------------
# RED: the chain on the old way
# ---------------------------------------------------------------------------------------------
def test_chain_without_hold_loses_the_told_anchor_on_every_rank():
    """b73e808a0f: the told stands (22528), the END-anchor insert of -72 runs the cap, the admission
    lands on the old END anchor (16665): deficit 5863, as the metal."""
    ranks = _ranks(hold=False)
    for r in ranks:
        assert r.c._weg2_told_hold_depths() == {}
        assert r.insert_tail_end_anchor() == 1
    assert [r.admit(T) for r in ranks] == [END_OLD] * 3
    assert all(r.state()[T][0] for r in ranks)          # 22528 is _weg2_capped on every rank


def test_switch_zero_is_the_old_way(monkeypatch):
    monkeypatch.setenv(TAH.ENV, "0")
    ranks = _ranks(hold=True)
    for r in ranks:
        assert r.c._weg2_told_hold_depths() == {}
        r.insert_tail_end_anchor()
    assert [r.admit(T) for r in ranks] == [END_OLD] * 3


# ---------------------------------------------------------------------------------------------
# GREEN: the same chain with the hold
# ---------------------------------------------------------------------------------------------
def test_path_cap_skips_the_held_anchor_on_every_rank():
    ranks = _ranks()
    taken = [r.insert_tail_end_anchor() for r in ranks]
    assert taken == [1, 1, 1]                           # the cap still bites: the NEXT eligible anchor
    assert [r.admit(T) for r in ranks] == [T] * 3
    for r in ranks:
        st = r.state()
        assert st[T] == (False, True)                   # held: not capped, reference kept
        assert st[24000][0] is True                     # the next eligible was taken instead
        assert st[END_OLD][0] is False                  # END anchors stay as before


def test_all_ranks_decide_alike():
    """rank-uniform by construction: same told table + same inserts -> identical state on every rank,
    whatever the rank's own copies/locks/clock say (a rank-local host lock on the held node changes
    nothing in the decision)."""
    ranks = _ranks()
    ranks[1].nodes[T].component_data[M].host_lock_ref = 1      # rank-local lock
    ranks[2].nodes[24000].component_data[M].lock_ref = 1       # rank-local device lock
    for r in ranks:
        r.insert_tail_end_anchor()
    st = [{d: v[0] for d, v in r.state().items()} for r in ranks]
    assert st[0] == st[1] == st[2]


def test_q610_claim_room_and_release_ended_do_not_give_the_held_anchor_back(monkeypatch):
    ranks = _ranks()
    for r in ranks:
        n = r.nodes[T]
        n._weg2_anchored = False                     # a plain settled prefix anchor (as after the cap)
        assert r.c._weg2_dual_releasable(n, r.pool) is False
        # Q-610 claim room: gives back every other settled anchor, not this one
        mc_node = UnifiedTreeNode(TC)
        released, ended, prefix, kept = r.c._weg2_dual_claim_room(mc_node, r.pool)
        assert released >= 1
        assert n.component_data[M].host_value is not None
        assert r.c._weg2_told_held(n)
    # the pop (admission) lifts the hold: the same node is releasable again
    for r in ranks:
        r.told.pop("weg2-0-77")
        assert r.c._weg2_dual_releasable(r.nodes[T], r.pool) is True


def test_q610_end_anchor_release_skips_the_held_end_anchor():
    r = _ranks(1)[0]
    n = r.nodes[END_OLD]                              # an END anchor of a done rid at the held depth
    r.told["weg2-0-77"] = END_OLD
    assert r.c._weg2_dual_releasable(n, r.pool) is False


def test_arena_drop_takes_only_unreferenced_slots_held_anchor_stays_referenced():
    """ARENA-DROP stage i takes only slots no tree references. The held anchor never loses its tree
    reference to cap / Q-610 / inner release, so its slot is not droppable; without the hold the
    release makes it droppable."""
    held = _ranks()[0]
    held.insert_tail_end_anchor()
    held.c._weg2_dual_claim_room(UnifiedTreeNode(TC), held.pool)
    slot_T = int(held.nodes[T].component_data[M].host_value[0])
    assert slot_T in held.pool.referenced()
    assert slot_T not in held.pool.arena_drop(100)

    old = _ranks(hold=False)[0]
    old.insert_tail_end_anchor()
    old.c._weg2_dual_claim_room(UnifiedTreeNode(TC), old.pool)
    assert old.nodes[T].component_data[M].host_value is None   # reference gone: droppable


def test_inner_anchor_release_keeps_the_held_anchor(monkeypatch):
    monkeypatch.setattr(URC, "_WEG2_END_ANCHOR", True, raising=False)
    monkeypatch.setenv("SGLANG_WEG2_MAMBA_INNER_ANCHOR_RELEASE", "1")
    r = _ranks(1)[0]
    r.nodes[T]._weg2_anchored = False
    below = r.nodes[24000]                            # the chain moved past 22528 (node below it)
    below.component_data[M].host_value = None         # nearest host-valued ancestor is the T node
    assert r.c._weg2_release_inner_anchor(below, r.pool) is False
    assert r.nodes[T].component_data[M].host_value is not None
    r.told.pop("weg2-0-77")
    assert r.c._weg2_release_inner_anchor(below, r.pool) is True


def test_h19_victims_exclude_held_anchors():
    from sglang.srt.weg2 import mamba_arena_displace as mad

    r = _ranks(1)[0]
    owned = [mad.OwnedAnchor(node=r.nodes[d], depth=d, rid="x", slots=[1]) for d in (END_OLD, T, 24000)]
    assert [a.depth for a in r.c._weg2_told_unheld(owned)] == [END_OLD, 24000]
    r.told.clear()
    assert r.c._weg2_told_unheld(owned) == owned


# ---------------------------------------------------------------------------------------------
# lifecycle + bounds
# ---------------------------------------------------------------------------------------------
def test_hold_ends_where_the_told_ends():
    r = _ranks(1)[0]
    assert r.c._weg2_told_hold_depths() == {T: ["weg2-0-77"]}
    r.told.pop("weg2-0-77")                           # admission / abort / Q-580 TOLD-FORGET
    assert r.c._weg2_told_hold_depths() == {}
    r.insert_tail_end_anchor()
    assert r.admit(T) == END_OLD                      # no hold -> the old way


def test_max_bound_holds_the_oldest_told_entries_only(monkeypatch, caplog):
    monkeypatch.setenv(TAH.ENV_MAX, "1")
    r = _ranks(1)[0]
    r.told["weg2-0-78"] = 24000
    with caplog.at_level(logging.WARNING):
        d = r.c._weg2_told_hold_depths(tick=True)
    assert d == {T: ["weg2-0-77"]}
    assert any("OVER-MAX" in x.message and "weg2-0-78" in x.message for x in caplog.records)


def test_runs_bound_expires_the_hold_with_a_named_warning(monkeypatch, caplog):
    monkeypatch.setenv(TAH.ENV_RUNS, "2")
    r = _ranks(1)[0]
    with caplog.at_level(logging.WARNING):
        assert r.c._weg2_told_hold_depths(tick=True) == {T: ["weg2-0-77"]}
        assert r.c._weg2_told_hold_depths(tick=True) == {T: ["weg2-0-77"]}
        assert r.c._weg2_told_hold_depths(tick=True) == {}
    assert any("EXPIRED" in x.message and "weg2-0-77" in x.message for x in caplog.records)
    # and the old way applies from then on: the cap takes the anchor
    r.insert_tail_end_anchor()
    assert r.nodes[T]._weg2_capped is True


def test_expiry_state_is_cleared_when_the_rid_leaves_the_table(monkeypatch):
    """a retried rid (same name, new instance) is held again: the stale expired/age state goes with
    the pop (a tick sweeps it)"""
    monkeypatch.setenv(TAH.ENV_RUNS, "1")
    r = _ranks(1)[0]
    r.c._weg2_told_hold_depths(tick=True)
    assert r.c._weg2_told_hold_depths(tick=True) == {}          # expired
    r.told.pop("weg2-0-77")
    r.c._weg2_told_hold_depths(tick=True)                       # sweep
    r.told["weg2-0-77"] = T
    assert r.c._weg2_told_hold_depths(tick=True) == {T: ["weg2-0-77"]}


# ---------------------------------------------------------------------------------------------
# F3 instruments
# ---------------------------------------------------------------------------------------------
def test_path_cap_line_names_a_told_anchor_beyond_the_sampling_limit(caplog):
    URC.UnifiedRadixCache._weg2_cap_n = 10_000       # far past the n <= 16 window
    r = _ranks(1, hold=False)[0]
    r.attach()
    r.told["weg2-0-77"] = T
    with caplog.at_level(logging.INFO):
        TAH_OFF = os.environ.__setitem__(TAH.ENV, "0")  # switch off: the anchor IS taken, and named
        try:
            r.insert_tail_end_anchor()
        finally:
            os.environ.pop(TAH.ENV, None)
    lines = [x.message for x in caplog.records if "WEG2 PATH-CAP" in x.message]
    assert lines and "TOLD-ANCHOR TAKEN" in lines[-1] and str(T) in lines[-1]


def test_path_cap_line_with_hold_names_kept_told_hold(caplog):
    URC.UnifiedRadixCache._weg2_cap_n = 10_000
    r = _ranks(1)[0]
    with caplog.at_level(logging.INFO):
        r.insert_tail_end_anchor()
    lines = [x.message for x in caplog.records if "WEG2 PATH-CAP" in x.message]
    assert lines and "kept_told_hold=1" in lines[-1]


def test_extent_regression_is_a_warning_with_the_cause(caplog):
    """y9d4: extent 22528 (set=273) -> 16665 (set=275) between SATISFIED and admission"""
    r = _ranks(1, hold=False)[0]
    h = r.attach()
    os.environ[TAH.ENV] = "0"
    try:
        r.insert_tail_end_anchor()                     # the cap takes 22528 (old way), noted
    finally:
        os.environ.pop(TAH.ENV, None)
    req = type("Q", (), {"rid": "weg2-0-77"})()
    with caplog.at_level(logging.WARNING):
        TAH.report_extent(req, T)                      # 09:34:10 extent=22528
        TAH.report_extent(req, END_OLD)                # 09:34:11 extent=16665
        TAH.report_extent(req, END_OLD)                # once per rid
    ws = [x.message for x in caplog.records if "EXTENT REGRESSED" in x.message]
    assert len(ws) == 1
    assert "told=22528" in ws[0] and "22528 -> 16665" in ws[0]
    assert "PATH-CAP" in ws[0] and "depth=22528" in ws[0] and "end_anchor=False" in ws[0]
    assert h.takes


def test_extent_regression_silent_when_fine(caplog):
    r = _ranks(1)[0]
    req = type("Q", (), {"rid": "weg2-0-77"})()
    with caplog.at_level(logging.WARNING):
        TAH.report_extent(req, T)
        TAH.report_extent(req, T + 512)
        r.told.pop("weg2-0-77")
        TAH.report_extent(req, END_OLD)                # told gone (admitted): not a regression
    assert not [x for x in caplog.records if "EXTENT REGRESSED" in x.message]


def test_extent_stamp_calls_the_report(monkeypatch):
    from sglang.srt.managers import pp_admission_congruence as PAC

    seen = []
    monkeypatch.setattr(TAH, "report_extent", lambda req, extent: seen.append(extent))
    monkeypatch.setattr(PAC, "state_aligned_load_back_len", lambda req: 4242)
    req = type("Q", (), {"rid": "r", "host_hit_length": 5})()
    assert PAC.stamp_state_aligned_extent(req) == 4242
    assert seen == [4242]


# ---------------------------------------------------------------------------------------------
# wiring: told table <-> tree
# ---------------------------------------------------------------------------------------------
class _Sched:
    def __init__(self, tree):
        self.tree_cache = tree


def test_armed_attaches_the_told_table_to_the_tree(monkeypatch):
    from sglang.srt.managers import weg2_store_told as ST

    monkeypatch.setattr(ST, "_resolve_armed", lambda s: True)
    monkeypatch.setattr(ST, "_paced_env", lambda: False)
    tree = object.__new__(UnifiedRadixCache)
    s = _Sched(tree)
    s.ps = type("P", (), {"pp_rank": 1})()
    try:
        ST.armed(s)
    except Exception:  # noqa: BLE001 -- only the attach is under test; later init lines need a real scheduler
        pass
    h = getattr(tree, "_weg2_told_hold", None)
    assert h is not None and h.source is s._weg2_store_told
    s._weg2_store_told["weg2-0-77"] = T
    assert h.depths() == {T: ["weg2-0-77"]}


def test_intake_attaches_lazily(monkeypatch):
    from sglang.srt.managers import weg2_store_told as ST

    tree = object.__new__(UnifiedRadixCache)
    tree.root_node = None
    s = _Sched(tree)
    s.ps = type("P", (), {"pp_rank": 1})()
    s._weg2_store_held = {}
    s._weg2_store_told = {}
    monkeypatch.setattr(ST, "forget_rid_leftovers", lambda *a, **k: None)
    monkeypatch.setattr(ST, "_follower_early_allowed", lambda sch: False)
    req = type("Q", (), {"rid": "weg2-0-77"})()
    ST.intake(s, req, lambda g: None)
    assert getattr(tree, "_weg2_told_hold", None) is not None


def test_prefix_trace_default_on_in_dual_p_only(monkeypatch):
    from sglang.srt.weg2 import prefix_trace as PT

    monkeypatch.delenv(PT.ENV, raising=False)
    assert PT._read()[0] is True                       # dual P (fixture), unset
    monkeypatch.setenv(PT.ENV, "0")
    assert PT._read()[0] is False                      # explicit 0 wins
    monkeypatch.delenv(PT.ENV, raising=False)
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "D")
    assert PT._read()[0] is False
    monkeypatch.delenv("SGLANG_WEG2_GROUP")
    monkeypatch.delenv("SGLANG_WEG2_DUAL_LAYOUT")
    assert PT._read()[0] is False                      # flip / NF: unchanged


# ---------------------------------------------------------------------------------------------
# flip / NF / INT8 / group D unchanged
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("env", [
    {},                                                                  # flip / NF / INT8: no dual layout
    {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"},          # group D
    {"SGLANG_WEG2_GROUP": "P"},                                          # group P without the dual layout
])
def test_flip_unchanged_no_hold_object_no_exemption(monkeypatch, env):
    for k in ("SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("SGLANG_WEG2_MAMBA_MAX_STATES_PER_PATH", "4")
    r = Rank({"weg2-0-77": T})
    assert r.attach() is None
    assert TAH.armed() is False
    assert getattr(r.c, "_weg2_told_hold", None) is None
    assert r.c._weg2_told_hold_depths() == {} and r.c._weg2_told_held(r.nodes[T]) is False
    if env.get("SGLANG_WEG2_GROUP") == "P":
        r.insert_tail_end_anchor()
        assert r.nodes[T]._weg2_capped is True        # the cap takes it as before
    # the stamp's report is a free no-op without a hold
    TAH.report_extent(type("Q", (), {"rid": "weg2-0-77"})(), 1)
