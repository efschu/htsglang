"""R12 (fLLiper release table row 12): every rank of a Form A D group keeps the
SAME host and anchor entries.

THE ROOT (rc9i/rc9k/rc9l/rc9m/rc9o). NF D in Form A: TP0 owns the arena host
pools, TP1/TP2 are expert workers with 0-byte plain host pools. At the store
ack TP0 rebinds a node's rows to the arena and keeps them; a worker frees them
as transit (``_weg2_release_chain_piece_host``: HOST layer of every component,
the mamba anchor included) -- rc9m D log: ``WEG2 PUBLISH-CHAIN host released``
on TP1/TP2 only. At the next device eviction TP0's node stays a host node with
its anchor (``#1469 EVICT ... backuped=True host=True``), the worker's node is
deleted (``backuped=False host=False``) -> anchors at different depths.

Hermetic (no CUDA). Each rank is a tree object carrying the REAL
UnifiedRadixCache methods that decide a node's host life -- the storage-queue
drain (``_drain_storage_control_queues_impl``), the transit release, the
load-back ack (``loading_check``), the eviction funnel, ``_evict_device_leaf``
/ ``_evict_to_host`` / ``_evict_host_leaf``, ``evict_host``, the #1421 refusal
and the H19 displacement -- on real ``UnifiedTreeNode`` objects, each run
under that rank's REAL installed Form A role plan. The request broadcast is a
pickle round trip of TP0's list to every rank.

RED on d1c7094ba6 (the rc9m form: workers end with an empty tree, TP0 with
three host nodes and their anchors); GREEN with R12.
"""

from __future__ import annotations

import contextlib
import importlib
import pickle
from array import array
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

from sglang.srt import rank_role
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
    EvictLayer,
    TreeComponent,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedLRUList
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as U
from sglang.srt.mem_cache.unified_radix_cache import UnifiedTreeNode

try:  # absent on d1c7094ba6 -- that absence is part of the red
    m = importlib.import_module("sglang.srt.mem_cache.form_a_host_shadow")
except ImportError:  # pragma: no cover - base tree
    m = None

SWITCH = "SGLANG_WEG2_ENABLE_FORM_A_HOST_SHADOW"
ROLES = ("host", "worker", "worker")
PAGE = 64
NODE = 2 * PAGE  # every node of the chain: two pages
FULL, MAMBA = ComponentType.FULL, ComponentType.MAMBA
COMPS = (FULL, MAMBA)


# ------------------------------------------------------------------ harness
class _Comp:
    """Stand-in for Full/Mamba component: frees rows, as the real ones do
    (Full defers ``value = None`` to the cascade)."""

    node_has_component_data = TreeComponent.node_has_component_data

    def __init__(self, ct, tree):
        self.component_type = ct
        self.cache = tree

    def evict_component(self, node, target=EvictLayer.DEVICE):
        cd = node.component_data[self.component_type]
        freed = host = 0
        if EvictLayer.DEVICE in target and cd.value is not None:
            freed = len(cd.value)
            if self.component_type != FULL:
                cd.value = None
        if EvictLayer.HOST in target and cd.host_value is not None:
            host = len(cd.host_value)
            cd.host_value = None
        return freed, host

    def eviction_priority(self, is_leaf):
        if self.component_type == FULL:
            return 0 if is_leaf else 2
        return 0

    def drive_host_eviction(self, num_tokens, tracker):
        ct = self.component_type
        for n in sorted(self.cache.evictable_host_leaves, key=lambda x: -x.id):
            if tracker[ct] >= num_tokens:
                break
            if n in self.cache.evictable_host_leaves:
                self.cache._evict_host_leaf(n, tracker)


class _Rank:
    """One D rank's tree cache: the REAL host-life methods on real nodes."""

    _drain_storage_control_queues_impl = U._drain_storage_control_queues_impl
    _weg2_release_chain_piece_host = U._weg2_release_chain_piece_host
    _evict_component_and_detach_lru = U._evict_component_and_detach_lru
    _evict_device_leaf = U._evict_device_leaf
    _evict_to_host = U._evict_to_host
    _evict_host_leaf = U._evict_host_leaf
    _cascade_evict = U._cascade_evict
    _is_device_leaf = U._is_device_leaf
    _is_host_leaf = U._is_host_leaf
    _update_evictable_leaf_sets = U._update_evictable_leaf_sets
    _remove_leaf_from_parent = U._remove_leaf_from_parent
    _iteratively_delete_tombstone_leaf = U._iteratively_delete_tombstone_leaf
    _for_each_component_lru = U._for_each_component_lru
    _count_ready_acks = U._count_ready_acks
    loading_check = U.loading_check
    evict_host = U.evict_host
    _1421_refused = U._1421_refused
    _weg2_release_anchor = U._weg2_release_anchor
    _r12_rec = None
    page_size = PAGE
    enable_storage_metrics = False
    storage_metrics_collector = None
    staging_write_ring = None
    _drain_depth_every = 0
    _mamba_pin_budget = 0
    pp_rank = 0

    def __init__(self, idx):
        self.idx = idx
        self.is_host = ROLES[idx] == "host"
        self.fail_rebind = set()
        self.tree_components = COMPS
        self.components = {ct: _Comp(ct, self) for ct in COMPS}
        self._components_tuple = tuple(self.components.values())
        self.lru_lists = {ct: UnifiedLRUList(ct, COMPS) for ct in COMPS}
        self.host_lru_lists = {ct: UnifiedLRUList(ct, COMPS, use_host_ptr=True) for ct in COMPS}
        self.evictable_device_leaves = set()
        self.evictable_host_leaves = set()
        self.ongoing_backup = {}
        self.ongoing_prefetch = {}
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.cache_controller = SimpleNamespace(
            prefetch_revoke_queue=Queue(), ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(), extra_host_mem_release_queues={},
            ack_load_queue=[], write_policy="write_through",
            mem_pool_host=SimpleNamespace(arena_read=self.is_host, size=4096),
        )
        self.root_node = UnifiedTreeNode(COMPS)
        self.root_node.key = RadixKey(array("q"), None)
        self.root_node.component_data[FULL].value = []
        self.root_node.hash_value = []
        for ct in COMPS:
            self.root_node.component_data[ct].lock_ref = 1
        self.state = m._State() if m is not None else None
        self._ids = 0

    # -- stubs for what is not under test -------------------------------
    def _record_remove_event(self, node, medium=None):
        pass

    def dec_host_lock_ref(self, node, params):
        pass

    def dec_lock_ref(self, node, params):
        pass

    def _mamba_pins_held(self):
        return 0

    def _weg2_host_is_transit(self):
        return True  # SGLANG_HICACHE_ARENA_DIR set, as on NF D

    def _weg2_rebind_host_to_arena(self, node):
        # TP0: the arena pool rebinds (#1424); a worker's plain pool cannot.
        return self.is_host and node.hash_value[-1] not in self.fail_rebind

    # -- building ---------------------------------------------------------
    def add_chain(self, n_nodes):
        parent, depth = self.root_node, 0
        for i in range(n_nodes):
            node = UnifiedTreeNode(COMPS)
            node.parent = parent
            node.key = RadixKey(array("q", range(depth, depth + NODE)), None)
            node.hash_value = [f"h{depth + PAGE}", f"h{depth + NODE}"]
            node.component_data[FULL].value = torch.arange(depth, depth + NODE)
            node.component_data[FULL].host_value = torch.arange(depth, depth + NODE)
            node.component_data[MAMBA].value = torch.tensor([i])
            node.component_data[MAMBA].host_value = torch.tensor([i])
            parent.children[node.key.child_key(PAGE)] = node
            self.lru_lists[MAMBA].insert_mru(node)
            parent, depth = node, depth + NODE
        self._leaf_sets()

    def _leaf_sets(self):
        for n in self.nodes():
            self._update_evictable_leaf_sets(n)

    def nodes(self):
        out, stack = [], [self.root_node]
        while stack:
            n = stack.pop()
            for c in n.children.values():
                out.append(c)
                stack.append(c)
        return out

    def node_at(self, depth):
        for n in self.nodes():
            if n.hash_value and n.hash_value[-1] == f"h{depth}":
                return n
        return None

    def ack_store_writes(self):
        """Every node's store write acked, drained on THIS rank (H74: the
        backup acks drain rank-locally)."""
        for n in self.nodes():
            self._ids += 1
            op = SimpleNamespace(id=self._ids, completed_tokens=0)
            self.ongoing_backup[op.id] = (n, None)
            self.cache_controller.ack_backup_queue.put(op)
        self._drain_storage_control_queues_impl(
            n_revoke=None, n_backup=None, n_release=None,
            extra_release_counts=None, log_metrics=False)

    def evict_all_device(self):
        """Device pressure: evict every device leaf until none is left, the
        same order on every rank (the device trees are lockstep)."""
        tracker = {ct: 0 for ct in COMPS}
        while True:
            leaves = sorted(self.evictable_device_leaves, key=lambda x: -x.id)
            if not leaves:
                return
            self._evict_device_leaf(leaves[0], tracker)

    def snapshot(self):
        """depth -> (KV host, anchor host) for every node in the tree."""
        out = {}
        for n in self.nodes():
            d, p = 0, n
            while p.parent is not None:
                d += len(p.key)
                p = p.parent
            out[d] = (int(n.component_data[FULL].host_value is not None),
                      int(n.component_data[MAMBA].host_value is not None))
        return dict(sorted(out.items()))


@contextlib.contextmanager
def _as_rank(rank):
    """This rank's installed Form A plan and (R12) its process state."""
    prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
    rank_role.set_form_a_role_plan(rank_role.RankRolePlan(ROLES), rank.idx)
    prev_state = None
    if m is not None:
        prev_state, m._S = m._S, rank.state
        m._S.tree = rank
    try:
        yield
    finally:
        rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev
        if m is not None:
            m._S = prev_state


@contextlib.contextmanager
def _switch(value):
    from sglang.srt.environ import envs

    field = getattr(envs, SWITCH, None)
    if field is None:  # d1c7094ba6: no switch (= the old path)
        yield
        return
    with field.override(value):
        yield


def _ranks(n_nodes=3):
    ranks = [_Rank(i) for i in range(3)]
    for r in ranks:
        r.add_chain(n_nodes)
    return ranks


def _each(ranks, fn):
    for r in ranks:
        with _as_rank(r):
            fn(r)


def _broadcast(ranks):
    """One ``recv_requests``: TP0 attaches, ``tp<-reqs`` carries the list (a
    pickle round trip), every rank consumes before the pass's requests."""
    if m is None:
        return [[] for _ in ranks]
    with _as_rank(ranks[0]):
        wire = pickle.dumps(m.attach(["req-a"]))
    out = []
    for r in ranks:
        with _as_rank(r):
            out.append(m.consume(pickle.loads(wire), r))
    return out


def _same(ranks):
    snaps = [r.snapshot() for r in ranks]
    assert snaps[1] == snaps[0] and snaps[2] == snaps[0], (
        "host/anchor entries per rank (depth -> (kv_host, anchor_host)): "
        f"TP0 {snaps[0]} TP1 {snaps[1]} TP2 {snaps[2]}")
    return snaps[0]


# ------------------------------------------------------------------ the rc9m form
def test_rc9m_form_after_eviction_every_rank_holds_the_same_entries():
    """Store acks, one broadcast, device eviction of the whole chain: TP0 keeps
    three host nodes with their anchors -- so must TP1/TP2. RED on d1c7094ba6:
    the workers freed the rows at the ack and deleted every node."""
    ranks = _ranks()
    with _switch(True):
        _each(ranks, lambda r: r.ack_store_writes())
        _broadcast(ranks)
        _each(ranks, lambda r: r.evict_all_device())
    snap = _same(ranks)
    assert snap == {NODE: (1, 1), 2 * NODE: (1, 1), 3 * NODE: (1, 1)}


def test_the_worker_keeps_its_rows_at_the_store_ack():
    ranks = _ranks()
    with _switch(True):
        _each(ranks, lambda r: r.ack_store_writes())
    assert _same(ranks) == {NODE: (1, 1), 2 * NODE: (1, 1), 3 * NODE: (1, 1)}
    assert all(n.l3_present for r in ranks for n in r.nodes())


def test_the_worker_keeps_its_rows_after_a_load_back():
    """``loading_check``: a store-loaded span (l3_present) on the card again;
    the worker released its rows there (no arena_read), TP0 kept them."""
    ranks = _ranks(1)

    class _Ev:
        def query(self):
            return True

        def synchronize(self):
            pass

    def load(r):
        node = r.nodes()[0]
        node.l3_present = True
        r.ongoing_load_back[7] = (node, None, None)
        r.cache_controller.ack_load_queue.append((_Ev(), _Ev(), [7]))
        r.loading_check()

    with _switch(True):
        _each(ranks, load)
    assert _same(ranks) == {NODE: (1, 1)}


# ------------------------------------------------------------------ TP0's own decisions
def test_a_failed_rebind_on_tp0_is_released_on_every_rank_at_the_broadcast():
    """TP0's arena rebind fails for the middle node: TP0 used to free it at
    the ack; now it defers and every rank runs the same release at the next
    broadcast -- in between, and after, the trees agree."""
    ranks = _ranks()
    ranks[0].fail_rebind.add(f"h{2 * NODE}")
    with _switch(True):
        _each(ranks, lambda r: r.ack_store_writes())
        _same(ranks)  # before the broadcast: nobody has released yet
        out = _broadcast(ranks)
    assert out == [["req-a"]] * 3  # the verdict never reaches the request path
    assert _same(ranks) == {NODE: (1, 1), 2 * NODE: (0, 0), 3 * NODE: (1, 1)}
    assert all(r.node_at(2 * NODE).component_data[FULL].value is not None for r in ranks)


def test_an_h19_displacement_on_tp0_reaches_the_workers():
    """TP0 displaces the shallowest anchor of a request (H19, its arena only):
    the workers drop the same anchor at the next broadcast, the KV stays."""
    ranks = _ranks()
    with _switch(True):
        _each(ranks, lambda r: r.ack_store_writes())
        with _as_rank(ranks[0]):
            victim = SimpleNamespace(node=ranks[0].node_at(NODE), slots=[3], rid="weg2-18-15", depth=NODE)
            st = SimpleNamespace(displaced_share=0, displaced_full=0, dropped=0)
            mp = SimpleNamespace(drop_unreferenced=lambda slots: len(slots))
            ranks[0]._weg2_release_anchor(victim, mp, st, why="share", for_rid="weg2-18-15")
        _broadcast(ranks)
    assert _same(ranks) == {NODE: (1, 0), 2 * NODE: (1, 1), 3 * NODE: (1, 1)}


def test_tp0_evict_host_reaches_the_workers_and_a_worker_never_evicts_on_its_own():
    ranks = _ranks()
    with _switch(True):
        _each(ranks, lambda r: r.ack_store_writes())
        _broadcast(ranks)
        _each(ranks, lambda r: r.evict_all_device())
        with _as_rank(ranks[1]):
            assert ranks[1].evict_host(NODE) == 0  # refused, named, nothing dropped
        with _as_rank(ranks[0]):
            assert ranks[0].evict_host(NODE) == NODE  # TP0's host leaf goes
        _broadcast(ranks)
    assert _same(ranks) == {NODE: (1, 1), 2 * NODE: (1, 1)}


def test_a_backup_tp0_refused_is_dropped_on_the_worker_once_its_own_write_landed():
    """TP0 refuses a node's backup (#1421 arena_claim); the worker backed it up
    and its write is still in flight (host lock) -> the drop waits, then goes."""
    ranks = _ranks()
    with _switch(True):
        _each(ranks, lambda r: r.ack_store_writes())
        tail = [r.node_at(3 * NODE) for r in ranks]
        tail[0].component_data[FULL].host_value = None  # TP0 never got the backup
        tail[0].component_data[MAMBA].host_value = None
        with _as_rank(ranks[0]):
            ranks[0]._1421_refused("arena_claim", tail[0])
        for t in tail[1:]:
            t.component_data[FULL].host_lock_ref = 1  # the worker's write in flight
        _broadcast(ranks)
        assert ranks[1].snapshot()[3 * NODE] == (1, 1)  # pending, not dropped under the lock
        for t in tail[1:]:
            t.component_data[FULL].host_lock_ref = 0
        _broadcast(ranks)  # nothing new from TP0: the pending drop is retried
    assert _same(ranks) == {NODE: (1, 1), 2 * NODE: (1, 1), 3 * NODE: (0, 0)}


def test_a_later_tp0_backup_supersedes_the_pending_drop():
    ranks = _ranks()
    with _switch(True):
        _each(ranks, lambda r: r.ack_store_writes())
        tail = [r.node_at(3 * NODE) for r in ranks]
        saved = tail[0].component_data[FULL].host_value
        tail[0].component_data[FULL].host_value = None
        with _as_rank(ranks[0]):
            ranks[0]._1421_refused("arena_claim", tail[0])
        for t in tail[1:]:
            t.component_data[FULL].host_lock_ref = 1
        _broadcast(ranks)
        tail[0].component_data[FULL].host_value = saved  # the next sweep backed it up
        with _as_rank(ranks[0]):
            m.note_backup_ok(ranks[0], tail[0])
        for t in tail[1:]:
            t.component_data[FULL].host_lock_ref = 0
        _broadcast(ranks)
    assert _same(ranks) == {NODE: (1, 1), 2 * NODE: (1, 1), 3 * NODE: (1, 1)}


# ------------------------------------------------------------------ wire, capacity, switch
def test_the_verdict_rides_first_and_leaves_the_list():
    ranks = _ranks(1)
    ranks[0].fail_rebind.add(f"h{NODE}")
    with _switch(True):
        _each(ranks, lambda r: r.ack_store_writes())
        with _as_rank(ranks[0]):
            sent = m.attach(["a", "b"])
            assert isinstance(sent[0], m.FormAHostVerdict) and sent[1:] == ["a", "b"]
            assert m.attach(["c"]) == ["c"]  # the ledger went with the first
        back = pickle.loads(pickle.dumps(sent))
        with _as_rank(ranks[2]):
            assert m.consume(back, ranks[2]) == ["a", "b"]


def test_wire_only_on_a_pure_tp_group():
    ps = SimpleNamespace(pp_size=1, tp_size=3)
    with _switch(True), _as_rank(_Rank(1)):
        assert m.wire_active(SimpleNamespace(enable_dp_attention=False), ps)
        assert not m.wire_active(SimpleNamespace(enable_dp_attention=True), ps)
        assert not m.wire_active(SimpleNamespace(enable_dp_attention=False),
                                 SimpleNamespace(pp_size=3, tp_size=1))
    with _switch(False), _as_rank(_Rank(1)):
        assert not m.wire_active(SimpleNamespace(enable_dp_attention=False), ps)


def test_a_worker_anchor_pool_gets_the_arena_rows_byteless_only(monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_HOST", "1")
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_MAMBA_SLOTS", "32")  # rc9m
    with _switch(True):
        with _as_rank(_Rank(1)):
            assert m.mamba_shadow_extra_rows(0) == 64
            assert m.mamba_shadow_extra_rows(58834944) == 0  # a pool with bytes
        with _as_rank(_Rank(0)):
            assert m.mamba_shadow_extra_rows(0) == 0  # TP0 has the arena itself
    with _switch(False), _as_rank(_Rank(1)):
        assert m.mamba_shadow_extra_rows(0) == 0


def test_a_tree_reset_drops_every_event():
    ranks = _ranks()
    ranks[0].fail_rebind.add(f"h{NODE}")
    with _switch(True):
        _each(ranks, lambda r: r.ack_store_writes())
        with _as_rank(ranks[0]):
            m.on_tree_reset(ranks[0])
            assert m.attach(["x"]) == ["x"]


def test_switch_off_is_the_old_path():
    """Guard: off, the workers free at the ack and the rc9m split comes back."""
    ranks = _ranks()
    with _switch(False):
        _each(ranks, lambda r: r.ack_store_writes())
        _each(ranks, lambda r: r.evict_all_device())
    assert ranks[0].snapshot() == {NODE: (1, 1), 2 * NODE: (1, 1), 3 * NODE: (1, 1)}
    assert ranks[1].snapshot() == {} and ranks[2].snapshot() == {}


def test_a_classic_boot_is_the_old_path():
    """No Form A plan installed: the shadow never acts, whatever the switch."""
    r = _Rank(1)
    r.add_chain(1)
    with _switch(True):
        prev = (rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK)
        rank_role.set_form_a_role_plan(None, 1)
        try:
            if m is not None:
                assert m.role() is None
            r.ack_store_writes()
        finally:
            rank_role._INSTALLED_PLAN, rank_role._INSTALLED_RANK = prev
    assert r.snapshot() == {NODE: (0, 0)}


if m is None:  # pragma: no cover - base tree: the module-only cases cannot run
    for _name in list(globals()):
        if _name.startswith("test_") and _name not in (
            "test_rc9m_form_after_eviction_every_rank_holds_the_same_entries",
            "test_the_worker_keeps_its_rows_at_the_store_ack",
            "test_the_worker_keeps_its_rows_after_a_load_back",
            "test_switch_off_is_the_old_path",
            "test_a_classic_boot_is_the_old_path",
        ):
            globals()[_name] = pytest.mark.skip(reason="R12 module absent (base)")(globals()[_name])
