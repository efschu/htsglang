# SPDX-License-Identifier: Apache-2.0
"""S39 (18.10.): D's peel paid 0 while the tree reported 448960 evictable tokens on the frontier -- ADMISSION-WEDGE.

MEASURED (fLLiper rc11 49f9c2ea4d, boot ..._dauer10181332_49f9c2ea4d_1018_133242.D.log, D TP0 13:46:27-13:48:03Z,
96 s without a round, ``ADMISSION-WEDGE: 7 queued, 0 running``): ``EVICT-FRONTIER-CENSUS request=36544
delivered_before=0 delivered_after_repair=0 reported_evictable=467328 on_frontier=448960``. Each frontier leaf:
``#1421 BACKUP-REFUSED why=arena_claim node=241 tokens=178816`` (the full arena) + ``UD-REFUSED
why=not_local_pp_floor node=241 children=1`` -- and ``L3-DROP FORMA`` 0x in the boot: node 241 is the load-back of
pdflip-12-40 (its 2794 pages on L3: ``#248 PARK-DEMOTE l3_pages=2794 complete=1``, the wake read marked it
``l3_present``), its child the host-only wake-read span of the next turn (pdflip-12-41), and it carried a device anchor
(``#924D station=donate node=241``). ``_weg2_l3_drop_forma`` refused every such leaf.

THE SHAPE HERE ("Device voll, 0 running, Arena voll, Seiten ohne Arena-Kopie, Load-Back kommt"): the REAL
``UnifiedRadixCache`` of D TP0 and of a worker (replicas, the fixture of test_weg2_read_room_d_1015), a store-backed
device leaf (``l3_present``, no host copy -- the #248 release) with a host-only ``l3_present`` child, every arena claim
refused, no request running; the REAL ``_evict_device_leaf`` on both ranks, the REAL R12 wire (``attach`` /
``consume``). RED on 24d1d517a9 (rc11): the peel frees 0 on both ranks, and still 0 after the broadcast. GREEN: the
peel still refuses on BOTH ranks (no rank-local drop), TP0 drops the leaf with its subtree at its broadcast and the
worker applies the same L3DROP verdict -- the load-back's rows are free on both ranks before the next admission.
"""

import os
import threading

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from sglang.srt.mem_cache import form_a_host_shadow as R12
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

from test_weg2_read_room_d_1015 import _broadcast, _real_d_tree, _WbController, roles  # noqa: F401 (fixture)

FULL = ComponentType.FULL
MAMBA = ComponentType.MAMBA


def _s39():
    try:
        from sglang.srt.weg2 import forma_l3_subtree as m
    except ImportError:   # the base (rc11) has no S39 module: the repro runs, the assertions say why it is red
        return None
    return m


@pytest.fixture(autouse=True)
def _clean_s39():
    R12._reset_state_for_tests()
    m = _s39()
    if m is not None:
        m._reset_for_tests()
    yield
    R12._reset_state_for_tests()
    if m is not None:
        m._reset_for_tests()


def _attached(t, node):
    return R12._attached(node) and any(c is node for c in node.parent.children.values())


def _rank(roles, tag, child=True, child_l3=True, anchor=False):
    """One rank's tree: leaf 1..8 (device, l3_present, no host copy) and, below it, the host-only child 9..12
    (host rows 90..93, l3_present). Arena full: every claim refused."""
    me = threading.current_thread().name
    roles.now[me] = "host" if tag == "tp0" else "worker"
    t = _real_d_tree([(1, list(range(8)))], None, tag)
    t.cache_controller = _WbController()
    leaf = t._t_nodes[1]
    leaf.component_data[FULL].host_value = None        # #248 LOADED-HOST-RELEASE: the arena rows went
    leaf.l3_present = True                             # the wake read: the pages came out of the store
    if child:
        t.insert(InsertParams(key=RadixKey(list(range(1, 13)), None),
                              value=t._t_alloc.alloc(12).to(dtype=torch.int64)))
        c = next(iter(leaf.children.values()))
        t._t_alloc.free(c.component_data[FULL].value)
        c.component_data[FULL].value = None            # host-only: the next turn's wake-read span
        c.component_data[FULL].host_value = torch.tensor([90, 91, 92, 93], dtype=torch.int64)
        c.hash_value = ["c9", "c10", "c11", "c12"]
        c.l3_present = child_l3
        t._t_child = c
    if anchor:
        leaf.component_data[MAMBA].value = torch.tensor([4], dtype=torch.int64)

    def no_claim(*a, **k):
        return 0                                       # #1421 arena_claim: the arena is full

    t.write_backup = no_claim
    return t


def _peel(roles, t):
    me = threading.current_thread().name
    roles.now[me] = "host" if t._t_tag == "tp0" else "worker"
    tracker = {ct: 0 for ct in t.tree_components}
    leaf = t._t_nodes[1]
    if t._is_device_leaf(leaf):
        t._evict_device_leaf(leaf, tracker)
    return tracker[FULL]


def _free(t):
    return int(t._t_alloc.available_size())


def test_s39_repro_device_full_arena_full_the_load_back_gets_its_rows_on_every_rank(roles):
    """THE REPRO. Pass i: the peel refuses the store-backed leaf on BOTH ranks (equal trees, no rank-local drop).
    TP0's broadcast: TP0 drops it with its host-only child, the worker applies the same verdict. Pass i+1: the 8
    device rows are free on both ranks; nothing of it is in the tree any more (its KV comes back from the store)."""
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    free0 = (_free(tp0), _free(worker))
    got = (_peel(roles, tp0), _peel(roles, worker))
    assert got == (0, 0), got
    assert _attached(tp0, tp0._t_nodes[1]) and _attached(worker, worker._t_nodes[1])   # same tree in pass i
    _broadcast(roles, tp0, worker)
    freed = (_free(tp0) - free0[0], _free(worker) - free0[1])
    assert freed == (8, 8), ("the load-back's rows: the peel delivered nothing on (TP0, worker)", freed)
    for t in (tp0, worker):
        assert not _attached(t, t._t_nodes[1]) and not _attached(t, t._t_child)
        assert not t.root_node.children


def test_s39_a_leaf_without_store_copy_never_leaves(roles):
    """Danger direction DATA LOSS: an un-backed device leaf WITHOUT l3_present has no copy anywhere -- it stays."""
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    for t in (tp0, worker):
        t._t_nodes[1].l3_present = False
        _peel(roles, t)
    _broadcast(roles, tp0, worker)
    assert _attached(tp0, tp0._t_nodes[1]) and _attached(worker, worker._t_nodes[1])
    assert _attached(tp0, tp0._t_child) and _attached(worker, worker._t_child)


def test_s39_a_child_outside_the_store_keeps_the_leaf(roles):
    """Danger direction DATA LOSS: a host-only child whose pages the store does not hold (no l3_present) is never
    deleted to make the leaf droppable."""
    tp0, worker = _rank(roles, "tp0", child_l3=False), _rank(roles, "worker", child_l3=False)
    _peel(roles, tp0), _peel(roles, worker)
    _broadcast(roles, tp0, worker)
    for t in (tp0, worker):
        assert _attached(t, t._t_nodes[1]) and _attached(t, t._t_child)


@pytest.mark.parametrize("busy", ["host_lock", "device_lock", "write_in_flight"])
def test_s39_a_busy_child_keeps_the_leaf(roles, busy):
    """A child read by a load-back (host lock), locked by a request, or under a write stays -- and so does the leaf."""
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    for t in (tp0, worker):
        c = t._t_child
        if busy == "host_lock":
            c.component_data[FULL].host_lock_ref = 1
        elif busy == "device_lock":
            c.component_data[FULL].lock_ref = 1
        else:
            c.write_through_pending_id = 4711        # a split child keeps the old write's id (#1421 V1 guard)
        _peel(roles, t)
    _broadcast(roles, tp0, worker)
    for t in (tp0, worker):
        assert _attached(t, t._t_nodes[1]) and _attached(t, t._t_child)


class _MambaPool:
    def __init__(self, held):
        self.held = held
        self.probed = []

    def anchor_held(self, h):
        self.probed.append(h)
        return self.held


def test_s39_a_device_anchor_leaves_only_when_the_store_holds_it(roles):
    """Danger direction DATA LOSS (D-NORECOMPUTE): the leaf's device anchor goes only when the store holds the state
    of its last page; otherwise TP0 issues the anchor-only copy and the leaf waits -- on BOTH ranks."""
    tp0, worker = _rank(roles, "tp0", anchor=True), _rank(roles, "worker", anchor=True)
    mp = _MambaPool(held=False)
    issued = []
    tp0._weg2_mamba_pool = lambda: mp
    tp0.write_backup_anchor_only = lambda n: issued.append(n.id) or 1
    _peel(roles, tp0), _peel(roles, worker)
    _broadcast(roles, tp0, worker)
    assert _attached(tp0, tp0._t_nodes[1]) and _attached(worker, worker._t_nodes[1])
    assert issued == [tp0._t_nodes[1].id], issued
    mp.held = True                                   # the anchor-only copy landed
    _broadcast(roles, tp0, worker)
    for t in (tp0, worker):
        assert not _attached(t, t._t_nodes[1]) and not t.root_node.children


def test_s39_the_worker_follows_only_the_verdict_never_its_own_peel(roles):
    """Danger direction RANK DISAGREEMENT: a worker never drops on its own (its peel refuses, it queues nothing);
    without TP0's verdict both trees stay equal, with it both drop."""
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    _peel(roles, worker)
    _broadcast(roles, tp0, worker)                   # TP0 never peeled: no verdict
    assert _attached(tp0, tp0._t_nodes[1]) and _attached(worker, worker._t_nodes[1])
    _peel(roles, tp0)
    assert _attached(tp0, tp0._t_nodes[1])           # not inside the round: at the broadcast
    _broadcast(roles, tp0, worker)
    assert not _attached(tp0, tp0._t_nodes[1]) and not _attached(worker, worker._t_nodes[1])


def test_s39_a_worker_that_cannot_follow_keeps_the_event_pending_then_names_it(roles):
    """A worker whose child is host-locked (its own load in flight) keeps the drop pending and applies it once the
    lock is gone; it never drops a busy node."""
    m = _s39()
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    worker._t_child.component_data[FULL].host_lock_ref = 1
    _peel(roles, tp0)
    _broadcast(roles, tp0, worker)
    assert not _attached(tp0, tp0._t_nodes[1])
    assert _attached(worker, worker._t_nodes[1]) and _attached(worker, worker._t_child)
    worker._t_child.component_data[FULL].host_lock_ref = 0
    _broadcast(roles, tp0, worker)                   # the pending verdict applies at the next consume
    assert not _attached(worker, worker._t_nodes[1]), m._ST.n if m else None


def test_s39_childless_anchorless_leaf_keeps_the_immediate_drop(roles):
    """The pre-S39 path is unchanged: a childless, anchorless store-backed leaf leaves inside the peel, on both."""
    tp0, worker = _rank(roles, "tp0", child=False), _rank(roles, "worker", child=False)
    assert (_peel(roles, tp0), _peel(roles, worker)) == (8, 8)


def test_s39_tp0_queues_only_the_tokens_the_peel_asked_for(roles):
    m = _s39()
    me = threading.current_thread().name
    roles.now[me] = "host"
    tp0 = _rank(roles, "tp0")
    a, b = tp0._t_nodes[1], tp0._t_child
    tp0._s39_peel = [4, 0]
    assert m.note_refused(tp0, a, "children")
    assert not m.note_refused(tp0, b, "children")    # 8 >= 4 queued already
    assert list(m._ST.queue) == [id(a)]


def test_s39_switch_off_is_the_old_refusal(roles, monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_FORMA_L3_SUBTREE", "0")
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    _peel(roles, tp0), _peel(roles, worker)
    _broadcast(roles, tp0, worker)
    assert _attached(tp0, tp0._t_nodes[1]) and _attached(worker, worker._t_nodes[1])


def test_s39_a_tree_reset_drops_the_queue(roles):
    m = _s39()
    tp0 = _rank(roles, "tp0")
    _peel(roles, tp0)
    assert m._ST.queue
    R12.on_tree_reset(tp0)
    assert not m._ST.queue


class _Arena:
    slot_bytes = 64

    def slot_stem(self, s):
        return f"m{s}"


class _Backend:
    def __init__(self, on_disk, lost):
        self.on_disk, self.lost, self.secured = set(on_disk), lost, []

    def _stat_stems(self, stems):
        return [s for s in stems if s in self.on_disk]

    def arena_secure_to_disk(self, arena, items):
        self.secured.extend(s for s, *_ in items)
        if self.lost:
            return {"on_disk": 0, "written": 0, "lost": len(items)}
        return {"on_disk": 0, "written": len(items), "lost": 0}


class _AnchorPool:
    def __init__(self, backend):
        self.arena, self._backend = _Arena(), backend

    def settled_anchor_slots(self, hv):
        return [int(x) for x in hv.tolist()]

    def anchor_held(self, h):
        return True


def test_s39_a_host_anchor_below_gets_its_l3_copy_before_its_node_leaves(roles):
    """Danger direction DATA LOSS: the child's host anchor (the next turn's resume state) leaves L2 only with an L3
    copy -- a failed copy keeps the leaf and the child on both ranks; a written copy lets both go."""
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    for t in (tp0, worker):
        t._t_child.component_data[MAMBA].host_value = torch.tensor([7], dtype=torch.int64)
    be = _Backend(on_disk=(), lost=True)
    tp0._weg2_mamba_pool = lambda: _AnchorPool(be)
    _peel(roles, tp0), _peel(roles, worker)
    _broadcast(roles, tp0, worker)
    assert be.secured == [7], be.secured
    for t in (tp0, worker):
        assert _attached(t, t._t_nodes[1]) and _attached(t, t._t_child)
    be.lost = False
    _peel(roles, tp0)
    _broadcast(roles, tp0, worker)
    for t in (tp0, worker):
        assert not _attached(t, t._t_nodes[1]) and not _attached(t, t._t_child)


def test_s39_a_host_anchor_already_on_disk_is_not_written_again(roles):
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    for t in (tp0, worker):
        t._t_child.component_data[MAMBA].host_value = torch.tensor([7], dtype=torch.int64)
    be = _Backend(on_disk=("m7",), lost=True)
    tp0._weg2_mamba_pool = lambda: _AnchorPool(be)
    _peel(roles, tp0), _peel(roles, worker)
    _broadcast(roles, tp0, worker)
    assert be.secured == [] and not _attached(tp0, tp0._t_nodes[1]) and not _attached(worker, worker._t_nodes[1])


def test_s39_tp0_never_drops_a_subtree_a_worker_holds_on_its_device(roles, monkeypatch):
    """S22 (R12 SPILL-DIVERGED class): a host-only leaf of the subtree that a worker has device-resident is in the
    workers' nodelete set -- TP0 leaves the whole drop alone (trees stay equal)."""
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    key = R12.node_key(tp0._t_child)
    monkeypatch.setattr(R12, "worker_nodelete_keys", lambda tree: {key})
    _peel(roles, tp0), _peel(roles, worker)
    _broadcast(roles, tp0, worker)
    for t in (tp0, worker):
        assert _attached(t, t._t_nodes[1]) and _attached(t, t._t_child)


def test_s39_the_broadcast_rechecks_the_store_copy(roles):
    """Danger direction DATA LOSS: TP0 decides at its broadcast from the tree AS IT IS THEN -- a leaf whose store
    mark was taken back since the peel queued it (K12 A2-orphan clears ``l3_present``) is never dropped."""
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    _peel(roles, tp0), _peel(roles, worker)
    for t in (tp0, worker):
        t._t_nodes[1].l3_present = False
    _broadcast(roles, tp0, worker)
    for t in (tp0, worker):
        assert _attached(t, t._t_nodes[1]) and _attached(t, t._t_child)
