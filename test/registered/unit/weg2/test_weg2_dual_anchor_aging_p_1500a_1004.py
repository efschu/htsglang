"""#1500a ANCHOR-AGING-P (dual group P, env SGLANG_WEG2_DUAL_ANCHOR_AGING_P, default OFF).

Befund deskq/done/1530-b9g-boot5-auswertung.md: after D's aging the 112-slot mamba arena stayed
36-42 pinned; ARENA-REF-HOLDERS pool=MAMBA tree=30 (PP0) tree_in_use=0, D tree=9 -- P's own radix
tree. P gave references back only at a refused claim or for an ended END anchor.

Hermetic CPU: the REAL ``UnifiedRadixCache._weg2_dual_retain_release`` / ``_weg2_dual_p_aging_tick``
/ ``_weg2_dual_d_held`` / ``_weg2_dual_releasable`` / ``_weg2_dual_release_ref`` on real
``UnifiedTreeNode`` trees, a stub arena pool (a released slot stays COMPLETE and readable until a
claim-time drop takes it), three PP ranks fed the same sequence of retains. The tick is the retain
count (``_weg2_dual_gen``). Every ``*_mutant`` test removes a guard and shows the red outcome.
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
from sglang.srt.weg2 import dual_anchor_release as DAR  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

M = ComponentType.MAMBA
F = ComponentType.FULL
TC = (ComponentType.FULL, ComponentType.MAMBA)
SLOTS = 112
N = 5                  # small N (retains) for the tests; production default 40


@pytest.fixture(autouse=True)
def dual_p_env(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_DUAL_LAYOUT", "1")
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    monkeypatch.setenv("SGLANG_WEG2_DUAL_ANCHOR_AGING_P", "1")
    monkeypatch.setenv("SGLANG_WEG2_DUAL_ANCHOR_AGING_TICKS", str(N))
    monkeypatch.delenv("SGLANG_WEG2_ENABLE_DUAL_ANCHOR_RELEASE", raising=False)
    monkeypatch.setattr(DAR, "_NP", [0])
    monkeypatch.setattr(DAR, "_AGING_P_FIRST", [True])
    yield


class _Arena:
    slots = SLOTS

    def __init__(self, pool):
        self.pool = pool

    def ref_census(self):
        return (len(self.pool.referenced()), 0)

    def slot_stem(self, slot):
        return "stem%d" % slot


class _Backend:
    def __init__(self, on_disk):
        self.on_disk = on_disk

    def _stat_stems(self, stems):
        return {s: 1 for s in stems if s in self.on_disk}


class _Pool:
    def __init__(self, on_disk=()):
        self.nodes = []
        self.complete = set()
        self.arena = _Arena(self)
        self._backend = _Backend(set(on_disk))

    def is_arena_id(self, i):
        return 0 <= int(i) < SLOTS

    def settled_anchor_slots(self, hv):
        return [int(x) for x in hv.tolist()]

    def referenced(self):
        return {int(n.component_data[M].host_value[0]) for n in self.nodes
                if n.component_data[M].host_value is not None}

    def readable(self, slot):
        return slot in self.complete


class _Mc:
    def evict_component(self, node, target=None):
        node.component_data[M].host_value = None


class _Told:
    def __init__(self, depths):
        self._d = depths

    def depths(self, tick=False):
        return dict(self._d)


class Rank:
    """One PP rank of group P: root with `n` anchor nodes (one arena slot each, key length 10*(i+1))."""

    def __init__(self, n=30, on_disk=()):
        self.pool = _Pool(on_disk)
        c = object.__new__(UnifiedRadixCache)
        c.root_node = UnifiedTreeNode(TC)
        c.components = {M: _Mc()}
        c.cache_controller = type("CC", (), {"tp_rank": 0})()
        c._weg2_direct_mamba_rows = {}
        c._evict_component_and_detach_lru = self._evict
        c._update_evictable_leaf_sets = lambda node: None
        c._weg2_mamba_pool = lambda: self.pool
        c.pp_rank = 0
        self.c = c
        self.nodes = []
        for i in range(n):
            self._add(i, 10 * (i + 1))

    def _evict(self, node, comp, target=None, tracker=None):
        node.component_data[M].host_value = None
        node.component_data[M].value = None
        return 0, 1

    def _add(self, slot, klen):
        n = UnifiedTreeNode(TC)
        n.key = [0] * klen
        n.parent = self.c.root_node
        self.c.root_node.children[slot] = n
        n._weg2_end_anchor = False
        n.component_data[M].host_value = torch.tensor([slot], dtype=torch.int64)
        self.pool.nodes.append(n)
        self.pool.complete.add(slot)
        self.nodes.append(n)
        return n

    def hit(self, i):
        self.nodes[i].last_access_time = URC.get_and_increase_time_counter()

    def retain(self):
        """One finished request: the retain hook (counts the generation, then the aging tick)."""
        UnifiedRadixCache._weg2_dual_retain_release(self.c)

    def retains(self, k):
        for _ in range(k):
            self.retain()

    def held(self):
        return sorted(self.pool.referenced())


# ---------------------------------------------------------------------------------------------
# the switch: default OFF, dual group P only
# ---------------------------------------------------------------------------------------------
def test_default_off_is_unchanged(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_DUAL_ANCHOR_AGING_P", raising=False)
    r = Rank(n=5)
    r.retains(10 * N)
    assert r.held() == [0, 1, 2, 3, 4]                       # nothing given back, no stamp written
    assert all(getattr(n, "weg2_age_stamp", None) is None for n in r.nodes)


def test_default_off_mutant_ignoring_the_switch_would_release(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_DUAL_ANCHOR_AGING_P", raising=False)
    monkeypatch.setattr(DAR, "aging_p_armed", lambda env=None: True)
    r = Rank(n=5)
    r.retains(10 * N)
    assert r.held() == []                                    # the switch test above has teeth


@pytest.mark.parametrize("env", [
    {"SGLANG_WEG2_GROUP": "D"},                              # group D has its own aging
    {"SGLANG_WEG2_DUAL_LAYOUT": "0"},                        # not the dual layout
    {"SGLANG_WEG2_DUAL_ANCHOR_AGING_P": "0"},
    {"SGLANG_WEG2_DUAL_ANCHOR_AGING_P": "garbage"},
])
def test_inert_outside_dual_group_p_and_without_the_switch(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert DAR.aging_p_armed() is False
    r = Rank(n=3)
    r.retains(10 * N)
    assert r.held() == [0, 1, 2]


def test_armed_in_dual_p_with_the_switch():
    assert DAR.aging_p_armed() is True
    assert DAR.aging_p_armed({"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}) is False


def test_the_d_switch_alone_does_not_age_p(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_DUAL_ANCHOR_AGING_P", raising=False)
    monkeypatch.setenv("SGLANG_WEG2_DUAL_ANCHOR_AGING", "1")
    r = Rank(n=3)
    r.retains(10 * N)
    assert r.held() == [0, 1, 2]


# ---------------------------------------------------------------------------------------------
# the rule: age in retains, soft release
# ---------------------------------------------------------------------------------------------
def test_fresh_anchor_stays_stale_anchor_is_released_softly():
    r = Rank(n=3)
    r.retains(N)                                             # first sight at retain 1; age N-1 .. not stale yet
    assert r.held() == [0, 1, 2]
    r.retains(2)
    assert r.held() == []                                    # unchanged for more than N retains
    assert all(r.pool.readable(s) for s in (0, 1, 2))        # soft: still COMPLETE and readable


def test_hit_resets_the_age():
    r = Rank(n=2)
    r.retains(3)                                             # ticks 1..3: both stamped at tick 1
    r.hit(0)                                                 # a match through node 0 (restamped at tick 4)
    r.retains(3)                                             # ticks 4..6: nobody is stale yet
    assert r.held() == [0, 1]
    r.retains(1)                                             # tick 7: node 1 unchanged for 6 > N retains
    assert r.held() == [0]                                   # node 0 restarted at tick 4: 3 retains old
    r.retains(N + 1)
    assert r.held() == []


def test_tick_is_the_retain_count_not_event_loop_rounds_or_wall_time(monkeypatch):
    r = Rank(n=2)
    r.c._1028_round = 10 ** 9                                # a free-running P event-loop counter: irrelevant
    monkeypatch.setattr("time.monotonic", lambda: 10 ** 9)
    monkeypatch.setattr("time.time", lambda: 10 ** 9)
    r.retains(2)
    assert r.held() == [0, 1]                                # N retains have NOT passed


def test_end_anchor_of_a_pending_rid_is_kept_of_a_done_rid_released(monkeypatch):
    r = Rank(n=3)
    for i in (0, 1):
        r.nodes[i]._weg2_end_anchor = True
    r.nodes[0].weg2_anchor_rid = "rid-pending"
    r.nodes[1].weg2_anchor_rid = "rid-done"
    monkeypatch.setattr(DAR, "rid_done", lambda rid, registered_at=None, now=None: rid == "rid-done")
    r.retains(6 * N)
    assert r.held() == [0]                                   # pending END anchor stays; done + plain released


def test_end_anchor_mutant_without_the_rid_done_guard_would_release_the_pending_one(monkeypatch):
    r = Rank(n=2)
    r.nodes[0]._weg2_end_anchor = True
    r.nodes[0].weg2_anchor_rid = "rid-pending"
    monkeypatch.setattr(DAR, "rid_done", lambda *a, **k: True)
    r.retains(6 * N)
    assert r.held() == []


def _protect(r, kind, i):
    n = r.nodes[i]
    if kind == "device_lock":
        n.component_data[F].lock_ref = 1
    elif kind == "mamba_lock":
        n.component_data[M].lock_ref = 1
    elif kind == "host_lock":
        n.component_data[M].host_lock_ref = 1
    elif kind == "write_pending":
        n.write_through_pending_id = 7
    elif kind == "direct_rows":
        r.c._weg2_direct_mamba_rows = {n.id: object()}
    elif kind == "told":
        r.c._weg2_told_hold = _Told({10 * (i + 1): ["rid-x"]})
    else:
        raise AssertionError(kind)


KINDS = ["device_lock", "mamba_lock", "host_lock", "write_pending", "direct_rows", "told"]


@pytest.mark.parametrize("kind", KINDS)
def test_protected_anchor_is_never_released_the_others_are(kind):
    r = Rank(n=3)
    _protect(r, kind, 1)
    r.retains(6 * N)
    assert r.held() == [1], kind                             # in-use / told-held / in-flight: untouched


@pytest.mark.parametrize("kind", KINDS)
def test_protected_anchor_mutant_without_the_guard_would_be_released(kind, monkeypatch):
    r = Rank(n=3)
    _protect(r, kind, 1)
    monkeypatch.setattr(UnifiedRadixCache, "_weg2_dual_releasable", lambda self, n, mp, claimer=None: True)
    r.retains(6 * N)
    assert r.held() == []                                    # the guard tests above have teeth


def test_protected_anchor_released_once_the_protection_ends():
    r = Rank(n=2)
    _protect(r, "device_lock", 0)
    r.retains(3 * N)
    assert r.held() == [0]
    r.nodes[0].component_data[F].lock_ref = 0                # the request ended
    r.retain()
    assert r.held() == []                                    # stale since long: released at once


def test_released_anchor_stays_readable_until_a_claim_drop_and_is_restamped_when_readopted():
    r = Rank(n=1)
    r.retains(2 * N)
    assert r.held() == [] and r.pool.readable(0)
    n = r.nodes[0]
    n.component_data[M].host_value = torch.tensor([0], dtype=torch.int64)   # a match re-adopted it
    n.last_access_time = URC.get_and_increase_time_counter()
    r.retains(N - 1)
    assert r.held() == [0]                                   # restarted: not stale again yet
    r.retains(3)
    assert r.held() == []


# ---------------------------------------------------------------------------------------------
# L3 copy: counted, never a gate (rank-local race)
# ---------------------------------------------------------------------------------------------
def test_unsecured_anchor_is_still_released_and_counted(caplog):
    r = Rank(n=3, on_disk={"stem0", "stem1"})                # slot 2 has no L3 copy yet
    with caplog.at_level(logging.INFO, logger=DAR.logger.name):
        r.retains(2 * N)
    assert r.held() == []                                    # NOT gated on the copy
    line = [m for m in caplog.messages if DAR.MARKER_AGING_P in m and "soft_released=3" in m]
    assert line and "secured=2 unsecured=1" in line[0]


def test_l3_stat_failure_counts_as_unsecured_and_never_raises():
    r = Rank(n=2)
    r.pool._backend = type("B", (), {"_stat_stems": staticmethod(lambda s: (_ for _ in ()).throw(OSError("x")))})()
    r.retains(2 * N)
    assert r.held() == []


# ---------------------------------------------------------------------------------------------
# rank agreement: the retain count is the only clock; same input, same decision on every PP rank
# ---------------------------------------------------------------------------------------------
def test_three_pp_ranks_decide_alike_retain_by_retain(monkeypatch):
    hits = {2: (0, 5), 4: (1,), 6: (0,), 8: (5, 6), 11: (0,), 14: (6,)}
    ranks = [Rank(n=8) for _ in range(3)]
    for i, rk in enumerate(ranks):
        rk.c.pp_rank = i
    seen = [[] for _ in ranks]
    # wall time and the free-running event-loop counter differ per rank; they must not matter
    for step in range(1, 5 * N):
        for i, rk in enumerate(ranks):
            rk.c._1028_round = step * (7 + 11 * i) * 1000
            for a in hits.get(step, ()):
                rk.hit(a)
        for i, rk in enumerate(ranks):
            rk.retain()
            seen[i].append(tuple(rk.held()))
    assert seen[0] == seen[1] == seen[2]


def test_aging_p_scan_is_a_pure_function_of_stamps_and_tick():
    r1, r2 = Rank(n=6), Rank(n=6)
    for t in range(1, 4 * N):
        s1 = DAR.aging_scan(r1.nodes, tick=t, ticks_n=N)
        s2 = DAR.aging_scan(r2.nodes, tick=t, ticks_n=N)
        assert [r1.nodes.index(n) for n in s1] == [r2.nodes.index(n) for n in s2]


# ---------------------------------------------------------------------------------------------
# robustness and marker
# ---------------------------------------------------------------------------------------------
def test_aging_p_never_takes_the_retain_down(monkeypatch, caplog):
    r = Rank(n=2)
    monkeypatch.setattr(DAR, "aging_scan", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with caplog.at_level(logging.WARNING, logger=DAR.logger.name):
        r.retains(3)                                         # must not raise
    assert any("tick raised RuntimeError" in m for m in caplog.messages)
    assert r.held() == [0, 1]


def test_no_arena_pool_is_a_noop():
    r = Rank(n=2)
    r.c._weg2_mamba_pool = lambda: None
    r.retains(3 * N)
    assert r.held() == [0, 1]


def test_marker_line_shape_and_rate_limit(caplog):
    with caplog.at_level(logging.INFO, logger=DAR.logger.name):
        for _ in range(200):
            DAR.log_aging_p(stale=1, soft_released=1, secured=1, unsecured=0, anchors=3, pinned=40,
                            slots=SLOTS, ticks_n=N, tick=9, pp_rank=1)
    lines = [m for m in caplog.messages if DAR.MARKER_AGING_P in m]
    assert 64 < len(lines) < 70                              # first 64, then every 64th
    assert ("stale=1 soft_released=1 secured=1 unsecured=0 anchors=3 pinned=40/112 ticks_n=5 tick=9 pp_rank=1"
            in lines[0])


def test_marker_quiet_when_nothing_to_say(caplog):
    DAR._AGING_P_FIRST[0] = False
    with caplog.at_level(logging.INFO, logger=DAR.logger.name):
        DAR.log_aging_p(stale=0, soft_released=0, secured=0, unsecured=0, anchors=3, pinned=40,
                        slots=SLOTS, ticks_n=N, tick=9)
    assert not [m for m in caplog.messages if DAR.MARKER_AGING_P in m]
