"""#1500a ANCHOR-AGING (dual group D, env FLLIPER_PDFLIP_DUAL_ANCHOR_AGING, default OFF).

Befund deskq/done/1500-anchor-lebensdauer.md (dual 1650, ed0803afa6): the 112-slot mamba arena
was complete 111-112/112 and 95-111 pinned from 16:57; D's tree held 84-98 anchors and Q-650 is
count-based (cap 84, or < 14 unpinned) -- no age, an idle anchor stays pinned for ever; P paid
15x ARENA-DROP freed=0 / 11x BACKUP-REFUSED why=mamba_claim.

Hermetic CPU: the REAL ``UnifiedRadixCache._pdflip_dual_d_tick`` / ``_pdflip_dual_d_held`` /
``_pdflip_dual_releasable`` / ``_pdflip_dual_release_ref`` on real ``UnifiedTreeNode`` trees, a stub
arena whose slot references are the tree's host values (a tree that gave its reference back makes
the slot droppable -- what a claim's ARENA-DROP stage i takes; until then it stays COMPLETE and
readable), three ranks fed the same sequence. Every ``*_mutant`` test runs the same chain with
the guard removed and shows the red outcome the guard prevents.
"""
from __future__ import annotations

import logging
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402

from flliper.srt.mem_cache import unified_radix_cache as URC  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache, UnifiedTreeNode  # noqa: E402
from flliper.srt.pdflip import dual_anchor_release as DAR  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

M = ComponentType.MAMBA
F = ComponentType.FULL
TC = (ComponentType.FULL, ComponentType.MAMBA)
SLOTS = 112
N_TICKS = 5            # a small N for the tests; the production default is 40
R = DAR.D_TICK_ROUNDS


@pytest.fixture(autouse=True)
def dual_d_env(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_LAYOUT", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_ANCHOR_AGING", "1")
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_ANCHOR_AGING_TICKS", str(N_TICKS))
    monkeypatch.delenv("FLLIPER_PDFLIP_ENABLE_DUAL_ANCHOR_RELEASE", raising=False)
    monkeypatch.setattr(DAR, "publish_room", lambda **kw: None)   # no /dev/shm file in a unit test
    monkeypatch.setattr(DAR, "_NA", [0])
    monkeypatch.setattr(DAR, "_AGING_FIRST", [True])
    monkeypatch.setattr(DAR, "_ND", [0])
    yield


class _Arena:
    slots = SLOTS

    def __init__(self, pool):
        self.pool = pool

    def ref_census(self):
        return (len(self.pool.referenced()), 0)


class _Pool:
    """Stub mamba arena pool: a slot is REFERENCED while a tree node's host value names it; a
    released slot stays COMPLETE (readable) until a claim-time drop (stage i: unreferenced) takes it."""

    def __init__(self):
        self.nodes = []
        self.complete = set()
        self.arena = _Arena(self)

    def is_arena_id(self, i):
        return 0 <= int(i) < SLOTS

    def settled_anchor_slots(self, hv):
        return [int(x) for x in hv.tolist()]

    def referenced(self):
        return {int(n.component_data[M].host_value[0]) for n in self.nodes
                if n.component_data[M].host_value is not None}

    def readable(self, slot):
        return slot in self.complete

    def claim_drop(self, want):
        """ARENA-DROP stage i: oldest unreferenced COMPLETE slots only."""
        free = sorted(s for s in self.complete if s not in self.referenced())
        take = free[:want]
        self.complete -= set(take)
        return take


class _Mc:
    def evict_component(self, node, target=None):
        node.component_data[M].host_value = None


class _Told:
    def __init__(self, depths):
        self._d = depths

    def depths(self, tick=False):
        return dict(self._d)


class Rank:
    """One group-D rank: root with `n` anchor nodes (one arena slot each, key length 10*(i+1))."""

    def __init__(self, n=30):
        self.pool = _Pool()
        c = object.__new__(UnifiedRadixCache)
        c.root_node = UnifiedTreeNode(TC)
        c.components = {M: _Mc()}
        c.cache_controller = type("CC", (), {"tp_rank": 0})()
        c._pdflip_direct_mamba_rows = {}
        c._evict_component_and_detach_lru = self._evict
        c._update_evictable_leaf_sets = lambda node: None
        c._pdflip_mamba_pool = lambda: self.pool
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
        n._pdflip_end_anchor = False
        n.component_data[M].host_value = torch.tensor([slot], dtype=torch.int64)
        self.pool.nodes.append(n)
        self.pool.complete.add(slot)
        self.nodes.append(n)
        return n

    def hit(self, i):
        """what a match through the node does: a new logical access time."""
        self.nodes[i].last_access_time = URC.get_and_increase_time_counter()

    def tick(self, k):
        """The D tick at HICACHE round k*D_TICK_ROUNDS (the rank-lockstep counter)."""
        self.c._1028_round = k * R
        UnifiedRadixCache._pdflip_dual_d_tick(self.c)

    def held_slots(self):
        return sorted(self.pool.referenced())


def _slot(r, i):
    return int(r.nodes[i].component_data[M].host_value[0])


# ---------------------------------------------------------------------------------------------
# (d) the switch: default OFF, dual group D only
# ---------------------------------------------------------------------------------------------
def test_default_off_is_unchanged(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_ANCHOR_AGING", raising=False)
    assert DAR.aging_armed() is False
    r = Rank()
    for k in range(1, 4 * N_TICKS):
        r.tick(k)
    assert len(r.held_slots()) == 30                       # 30 < cap 84, room: Q-650 gives nothing
    assert all(getattr(n, "pdflip_age_stamp", None) is None for n in r.nodes)   # not even a stamp written


def test_default_off_mutant_aging_ignoring_the_switch_would_release(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_ANCHOR_AGING", raising=False)
    monkeypatch.setattr(DAR, "aging_armed", lambda env=None: True)       # the mutant: switch ignored
    r = Rank()
    for k in range(1, 4 * N_TICKS):
        r.tick(k)
    assert len(r.held_slots()) == 0                         # the guard test above has teeth


@pytest.mark.parametrize("env", [
    {"FLLIPER_PDFLIP_DUAL_LAYOUT": "0"},                       # flip / NF / INT8: no dual layout
    {"FLLIPER_PDFLIP_GROUP": "P"},                             # group P: aging is D's tick
    {"FLLIPER_PDFLIP_GROUP": ""},
    {"FLLIPER_PDFLIP_ENABLE_DUAL_ANCHOR_RELEASE": "0"},        # the Q-610/Q-650 switch off
    {"FLLIPER_PDFLIP_DUAL_ANCHOR_AGING": "0"},
    {"FLLIPER_PDFLIP_DUAL_ANCHOR_AGING": ""},
])
def test_inert_outside_dual_group_d_and_without_the_switch(monkeypatch, env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    assert DAR.aging_armed() is False
    r = Rank()
    for k in range(1, 4 * N_TICKS):
        r.tick(k)
    assert len(r.held_slots()) == 30
    assert all(getattr(n, "pdflip_age_stamp", None) is None for n in r.nodes)


def test_armed_in_dual_d_with_the_switch():
    assert DAR.aging_armed() is True
    assert DAR.aging_ticks() == N_TICKS


def test_ticks_env_default_and_garbage(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_ANCHOR_AGING_TICKS")
    assert DAR.aging_ticks() == DAR.AGING_TICKS_DEFAULT == 40
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_ANCHOR_AGING_TICKS", "x")
    assert DAR.aging_ticks() == 40
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_ANCHOR_AGING_TICKS", "0")
    assert DAR.aging_ticks() == 1
    monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_ANCHOR_AGING_TICKS", "12")
    assert DAR.aging_ticks() == 12
    assert DAR.aging_armed({"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D"}) is False   # switch absent
    assert DAR.aging_armed({"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "D",
                            "FLLIPER_PDFLIP_DUAL_ANCHOR_AGING": "1"}) is True


# ---------------------------------------------------------------------------------------------
# (a)+(b) the rule: fresh stays, stale is released softly and stays readable until a claim
# ---------------------------------------------------------------------------------------------
def test_fresh_anchor_stays_stale_anchor_is_released_softly():
    r = Rank(n=4)
    for k in range(1, N_TICKS + 2):                          # ticks 1..6: first sight at tick 1, N=5
        r.tick(k)
        assert len(r.held_slots()) == 4, k                   # nobody is stale yet
    r.hit(0)                                                 # a match touches anchor 0 before tick 7
    r.tick(N_TICKS + 2)                                      # tick 7: 7 - 1 > 5 -> 1,2,3 stale
    assert r.held_slots() == [0]                             # the fresh one stays referenced
    for i in (1, 2, 3):                                      # released softly: the page is still there
        assert r.nodes[i].component_data[M].host_value is None
        assert r.pool.readable(i)
    assert r.pool.readable(0)


def test_hit_resets_the_age():
    r = Rank(n=2)
    for k in range(1, 40):
        if k % 3 == 0:
            r.hit(0)                                         # anchor 0 hit every 3rd tick (<< N)
        r.tick(k)
    assert _slot(r, 0) == 0                                  # still held after 39 ticks
    assert r.nodes[1].component_data[M].host_value is None   # the cold one went at tick 7


def test_stale_anchor_is_still_readable_until_a_claim_takes_it():
    r = Rank(n=6)
    for k in range(1, 4):
        r.tick(k)
    r.hit(2)
    r.hit(1)                                                 # restamped at tick 4 -> stale from tick 10
    r.tick(4)
    r.hit(4)                                                 # restamped at tick 5 -> stale from tick 11
    for k in range(5, 11):
        r.tick(k)
    assert r.held_slots() == [4]                             # 0,3,5 went at tick 7, 1,2 at tick 10
    # nothing was deleted: the pages are still COMPLETE and readable
    assert all(r.pool.readable(s) for s in (0, 1, 2, 3, 4, 5))
    taken = r.pool.claim_drop(10)                            # P's claim: ONLY unreferenced slots are takeable
    assert sorted(taken) == [0, 1, 2, 3, 5]
    assert r.pool.readable(4) and not r.pool.readable(0)


def test_release_order_is_oldest_access_first():
    order = []
    r = Rank(n=5)
    orig = r._evict

    def evict(node, comp, target=None, tracker=None):
        order.append(int(node.component_data[M].host_value[0]))
        return orig(node, comp, target, tracker)

    r.c._evict_component_and_detach_lru = evict
    r.hit(3)
    r.hit(1)
    r.hit(4)                                                 # access order: 0 < 2 < 3 < 1 < 4
    for k in range(1, N_TICKS + 3):
        r.tick(k)
    assert order == [0, 2, 3, 1, 4]


def test_restart_of_age_for_a_readopted_anchor():
    r = Rank(n=1)
    for k in range(1, N_TICKS + 3):
        r.tick(k)
    assert r.held_slots() == []
    assert r.nodes[0].pdflip_age_stamp is None                  # the stamp left with the reference


# ---------------------------------------------------------------------------------------------
# never: locked / host-locked / write in flight / direct rows / standing told
# ---------------------------------------------------------------------------------------------
def _protect(r, kind, i):
    n = r.nodes[i]
    if kind == "device_lock":
        n.component_data[F].lock_ref = 1                      # a running request holds the node
    elif kind == "mamba_lock":
        n.component_data[M].lock_ref = 1
    elif kind == "host_lock":
        n.component_data[M].host_lock_ref = 1
    elif kind == "write_pending":
        n.write_through_pending_id = 7
    elif kind == "direct_rows":
        r.c._pdflip_direct_mamba_rows = {n.id: object()}
    elif kind == "told":
        r.c._pdflip_told_hold = _Told({10 * (i + 1): ["rid-x"]})
    else:
        raise AssertionError(kind)


KINDS = ["device_lock", "mamba_lock", "host_lock", "write_pending", "direct_rows", "told"]


@pytest.mark.parametrize("kind", KINDS)
def test_protected_anchor_is_never_released_the_others_are(kind):
    r = Rank(n=3)
    _protect(r, kind, 1)
    for k in range(1, 6 * N_TICKS):
        r.tick(k)
    assert r.held_slots() == [1], kind                        # only the protected one is still held


@pytest.mark.parametrize("kind", KINDS)
def test_protected_anchor_mutant_without_the_guard_would_be_released(kind, monkeypatch):
    r = Rank(n=3)
    _protect(r, kind, 1)
    monkeypatch.setattr(UnifiedRadixCache, "_pdflip_dual_releasable", lambda self, n, mp, claimer=None: True)
    for k in range(1, 6 * N_TICKS):
        r.tick(k)
    assert r.held_slots() == []                               # the guard tests above have teeth


def test_protected_anchor_released_once_the_protection_ends():
    r = Rank(n=2)
    _protect(r, "device_lock", 0)
    for k in range(1, 3 * N_TICKS):
        r.tick(k)
    assert r.held_slots() == [0]
    r.nodes[0].component_data[F].lock_ref = 0                 # the request ended
    r.tick(3 * N_TICKS)
    assert r.held_slots() == []                               # stale since long: released at once


# ---------------------------------------------------------------------------------------------
# rank agreement: same input, same decision on every rank; no clock, no per-rank reading
# ---------------------------------------------------------------------------------------------
def _drive(rank, hits_by_tick, ticks):
    seen = []
    for k in range(1, ticks + 1):
        for i in hits_by_tick.get(k, ()):
            rank.hit(i)
        rank.tick(k)
        seen.append(tuple(rank.held_slots()))
    return seen


def test_three_ranks_decide_alike_tick_by_tick(monkeypatch):
    hits = {2: (0, 5), 4: (1,), 6: (0,), 8: (5, 6), 11: (0,), 14: (6,)}
    ranks = [Rank(n=8) for _ in range(3)]
    walls = iter([10.0, 5000.0, 123456.0])
    out = []
    for r in ranks:
        # each rank has its own wildly different wall clock: aging must not read it
        t0 = next(walls)
        monkeypatch.setattr(time, "time", lambda t0=t0: t0)
        monkeypatch.setattr(time, "monotonic", lambda t0=t0: t0)
        out.append(_drive(r, hits, 20))
    assert out[0] == out[1] == out[2]
    assert out[0][-1] != out[0][0]                            # and they did release something
    # the rank-local time counter differs between ranks (it is a process-wide counter that others
    # advance): shift rank 1's by an offset -- only the ORDER of last_access_time is read
    r = Rank(n=8)
    URC.get_and_increase_time_counter()
    URC.get_and_increase_time_counter()
    assert _drive(r, hits, 20) == out[0]


def test_aging_scan_is_a_pure_function_of_stamps_and_tick():
    class N:
        def __init__(self, lat):
            self.last_access_time = lat

    a, b = [N(1.0), N(2.0)], [N(1.0), N(2.0)]
    for tick in range(1, 10):
        sa = DAR.aging_scan(a, tick=tick, ticks_n=3)
        sb = DAR.aging_scan(b, tick=tick, ticks_n=3)
        assert [n.last_access_time for n in sa] == [n.last_access_time for n in sb]
    assert [n.last_access_time for n in DAR.aging_scan(a, tick=9, ticks_n=3)] == [1.0, 2.0]
    assert DAR.aging_scan([N(5.0)], tick=100, ticks_n=3) == []     # a first sight is never stale


# ---------------------------------------------------------------------------------------------
# (c) d_release_need with the stale count; the regular pass asks only what is left
# ---------------------------------------------------------------------------------------------
def _old_need(*, slots, pinned, d_held):
    """the Q-650 body before #1500a, verbatim."""
    slots, pinned, d_held = int(slots), int(pinned), int(d_held)
    if slots <= 0 or d_held <= 0:
        return 0
    over_cap = max(0, d_held - DAR.d_cap(slots))
    short_room = max(0, DAR.p_room(slots) - (slots - pinned))
    return min(d_held, max(over_cap, short_room))


def test_d_release_need_default_is_the_q650_rule_differential():
    for slots in (0, 8, 32, 112):
        for pinned in range(0, slots + 1, 7):
            for d_held in range(0, slots + 1, 5):
                assert DAR.d_release_need(slots=slots, pinned=pinned, d_held=d_held) == \
                    _old_need(slots=slots, pinned=pinned, d_held=d_held)
                assert DAR.d_release_need(slots=slots, pinned=pinned, d_held=d_held, stale=0) == \
                    _old_need(slots=slots, pinned=pinned, d_held=d_held)


def test_d_release_need_with_stale_counts_what_aging_already_gave():
    # 100 held, 100 pinned (cap 84, room 14): Q-650 alone gives 16 back
    assert DAR.d_release_need(slots=112, pinned=100, d_held=100) == 16
    # aging gave 20 back in this tick: 80 held, 80 pinned -> under the cap, 32 unpinned >= 14: nothing more
    assert DAR.d_release_need(slots=112, pinned=100, d_held=100, stale=20) == 0
    # aging gave only 5: 95 held / 95 pinned -> over the cap by 11, room 17 -> 11
    assert DAR.d_release_need(slots=112, pinned=100, d_held=100, stale=5) == 11
    # more stale than held clamps
    assert DAR.d_release_need(slots=112, pinned=100, d_held=10, stale=99) == 0


def test_tick_aging_then_regular_pass_does_not_double_release():
    # 100 held > cap 84, 100 pinned: Q-650 alone gives the 16 least recently used back.
    r0 = Rank(n=100)
    monk = pytest.MonkeyPatch()
    monk.setenv("FLLIPER_PDFLIP_DUAL_ANCHOR_AGING", "0")
    try:
        r0.tick(1)
    finally:
        monk.undo()
    assert len(r0.held_slots()) == 84
    # with aging: 20 anchors (50..69, not the LRU ones) have been stale for ages -> aging gives
    # exactly those back, the regular pass asks the cap/room of the 80 that are left: nothing more
    r = Rank(n=100)
    for i in range(50, 70):
        r.nodes[i].pdflip_age_stamp = (r.nodes[i].last_access_time, -100)
    r.tick(1)
    gone = {i for i in range(100) if r.nodes[i].component_data[M].host_value is None}
    assert gone == set(range(50, 70))
    assert len(r.held_slots()) == 80


def test_q650_still_works_when_aging_raises(monkeypatch, caplog):
    def boom(*a, **kw):
        raise RuntimeError("aging bug")

    monkeypatch.setattr(DAR, "aging_scan", boom)
    r = Rank(n=100)
    with caplog.at_level(logging.WARNING):
        r.tick(1)
    assert len(r.held_slots()) == 84                          # the cap pass ran anyway
    assert any("raised RuntimeError" in m for m in caplog.messages)


# ---------------------------------------------------------------------------------------------
# the marker
# ---------------------------------------------------------------------------------------------
def test_marker_line_shape_and_rate_limit(caplog):
    r = Rank(n=3)
    with caplog.at_level(logging.INFO):
        for k in range(1, 6 * N_TICKS):
            r.tick(k)
    lines = [m for m in caplog.messages if m.startswith("#1500a ANCHOR-AGING")]
    assert lines, caplog.messages
    first, last = lines[0], lines[-1]
    assert "stale=0 soft_released=0 anchors=3 pinned=3/112 ticks_n=5" in first        # proof of life at tick 1
    rel = [m for m in lines if "soft_released=3" in m]
    assert rel and "stale=3 soft_released=3 anchors=0 pinned=0/112 ticks_n=5" in rel[0]
    assert len(lines) == 2                                     # first sight + the release; quiet ticks say nothing


def test_marker_is_bounded(caplog):
    with caplog.at_level(logging.INFO):
        for _ in range(500):
            DAR.log_aging(stale=1, soft_released=1, anchors=2, pinned=3, slots=112, ticks_n=5)
    n = sum(1 for m in caplog.messages if m.startswith("#1500a ANCHOR-AGING"))
    assert n == 64 + len([x for x in range(65, 501) if x % 64 == 0])      # first 64, then every 64th
