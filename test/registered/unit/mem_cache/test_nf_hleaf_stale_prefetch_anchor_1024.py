# SPDX-License-Identifier: Apache-2.0
"""HLEAF-STALE-1024: a storage read completed below a node that had left the tree while the read was in flight.

METAL (NF rc39 49cfdde0d0, boot boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwfl3kartenbar1sw310240748_49cfdde0d0_1024_074822,
D.log): D TP1 and TP2 died at 08:02:03Z in ``sanity_check`` (on_idle -> _check_tree_cache_bounded):

    H-leaf extra: [314]
    1 stale nodes in host_leaves: [314]
    mamba host LRU: +S3=set(), +lru={314}

Node 314 is named nowhere else in the log; its neighbours tell its life (D.log lines):

  76208  08:01:48  #924D station=load_back rid=pdflip-4-107 node=310 H->D COW target   (310 = 6400 tokens, depth
                   119552..125952, back on the device for pdflip-4-107; no Full host copy -- a store-backed leaf)
  76692  08:01:52  MATCH-CENSUS-DEEP rid=pdflip-4-108 reached=125952 accepted=125952  (pdflip-4-108's first read
                   registers with last_host_node = 310; 3 pages = 192 tokens to fetch)
  76726  08:01:53  S39 L3-DROP FORMA n=30 node=310 tokens=6400 children=3 (TP0's broadcast drops 310 with its host-only
                   subtree; the workers apply the same drop at this broadcast)
  76765  08:01:53  LEAF-SET DETACHED n=1 node=310 caller=.../check_prefetch_progress/_insert_helper_host   (all 3 ranks:
                   the read's completion walks DOWN from the removed 310 and hangs its 192 tokens there -- node 314)
  76768  08:01:53  LEAF-SET DETACHED n=2 node=310 caller=.../check_prefetch_progress/dec_host_lock_ref
  76836  08:01:53  FA PREFETCH-FROM-ANCHOR rid=pdflip-4-108 anchor=119552   (the second read, 6592 tokens, from where the
                   tree really ends now)
  78032  08:02:03  first idle walk after pdflip-4-99 ends: TP1/TP2 raise (TP0 was already in the next pass's
                   request broadcast -- the walk is rank-local and duty-cycled, the state was the same on all three)

THE WURZEL. ``prefetch_from_storage`` protects the read's anchor with ``inc_host_lock_ref(last_host_node)``; on a node
without a Full host copy that is a NO-OP (``full_component.acquire_component_lock(lock_host=True)`` returns on
``host_value is None``). Form A makes such anchors common (store-backed device leaves, ``l3_present``, no arena copy)
and S39 deletes exactly those. TWO repairs, both here:

  1. S39 treats the anchor of an in-flight read as a holder: TP0 does not drop it (``prefetch_anchor``, a silent
     dequeue like ``locked``), a worker whose own read anchors there keeps the drop pending.
  2. ``check_prefetch_progress`` refuses the insert when the anchor left the tree on ANY rank (a new slot in the
     packed MIN -- SOLL R11, one answer for the group), in the #937 shape: the span goes back to its pool.
"""

import logging
import os
import threading

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest
import torch

from flliper.srt.mem_cache import form_a_host_shadow as R12
from flliper.srt.mem_cache import unified_radix_cache as urc
from flliper.srt.mem_cache.base_prefix_cache import InsertParams
from flliper.srt.mem_cache.hicache_phase_binding import binding_state
from flliper.srt.mem_cache.radix_cache import RadixKey
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from flliper.srt.mem_cache.unified_radix_cache import _OngoingPrefetch

from test_pdflip_read_room_d_1015 import _broadcast, roles  # noqa: F401 (fixture)
from test_pdflip_s39_forma_l3_subtree_1018 import _attached, _free, _peel, _rank
from test_1157_reaper_prices_requested_span import REAP_REQ, REAP_TOKENS, _reap_scenario

FULL = ComponentType.FULL


@pytest.fixture(autouse=True)
def _clean():
    from flliper.srt.pdflip import forma_l3_subtree as s39

    R12._reset_state_for_tests()
    s39._reset_for_tests()
    binding_state().reset()
    yield
    R12._reset_state_for_tests()
    s39._reset_for_tests()
    binding_state().reset()


def _register_read(t, rid, anchor):
    """An in-flight read anchored on ``anchor`` -- what ``prefetch_from_storage`` leaves in ``ongoing_prefetch``
    (only the anchor is read by the drop paths)."""
    lock = t.inc_host_lock_ref(anchor).to_dec_params()   # the registration's lock: a no-op on a device-only node
    t.ongoing_prefetch[rid] = _OngoingPrefetch(anchor, RadixKey([1, 2, 3, 4], None), None, None, lock, {})


# --------------------------------------------------------------------------- 1. S39 spares the anchor


def test_hleaf_repro_s39_never_drops_the_anchor_of_an_inflight_read(roles):
    """THE REPRO (node 310): the store-backed device leaf with its host-only child is S39's -- and the anchor of a
    read still in flight on both ranks. RED on 807de14309: TP0 drops it at its broadcast and the worker follows, the
    read would complete below a removed node. GREEN: it stays on both ranks; once the read has completed, the next
    broadcast drops it as before (S39 still pays the room)."""
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    for t in (tp0, worker):
        leaf = t._t_nodes[1]
        _register_read(t, "pdflip-4-108", leaf)
        assert leaf.component_data[FULL].host_lock_ref == 0   # THE HOLE: the registration's lock held nothing
    free0 = (_free(tp0), _free(worker))
    assert (_peel(roles, tp0), _peel(roles, worker)) == (0, 0)
    _broadcast(roles, tp0, worker)
    for t in (tp0, worker):
        assert _attached(t, t._t_nodes[1]), ("the in-flight read's anchor left the tree on", t._t_tag)
        assert _attached(t, t._t_child)
    assert (_free(tp0), _free(worker)) == free0
    # the read completes: the record leaves ongoing_prefetch -- the leaf is S39's again
    for t in (tp0, worker):
        t.ongoing_prefetch.clear()
    _peel(roles, tp0), _peel(roles, worker)
    _broadcast(roles, tp0, worker)
    assert (_free(tp0) - free0[0], _free(worker) - free0[1]) == (8, 8)
    for t in (tp0, worker):
        assert not _attached(t, t._t_nodes[1]) and not t.root_node.children


class _Arena:
    def __init__(self, path):
        self.path = path


def _arena_dir(monkeypatch, tmp_path):
    """The worker's S22/S41 key files go next to the arena (here: tmp) -- the channel TP0 reads at its broadcast."""
    arena = _Arena(str(tmp_path / "arena-0.bin"))
    monkeypatch.setattr(R12, "_tree_arena", lambda tree: arena)


def _as(roles, role):
    roles.now[threading.current_thread().name] = role


def _round(roles, tp0, worker):
    """One pass: the peel on both ranks (TP0 queues), TP0's broadcast, the worker's consume (its refresh publishes)."""
    _peel(roles, tp0), _peel(roles, worker)
    _broadcast(roles, tp0, worker)


def test_hleaf_a1_a_workers_own_read_is_vetoed_at_tp0_through_insert_pin_and_load_back(roles, monkeypatch, tmp_path):
    """MUSS A1 (review): only the WORKER has a read anchored on the leaf L (Form A HP1-END: TP0's anchor is another
    node). The real course: the read registers (publishes its anchor at once), completes -- its span C is inserted
    below L and pinned (#1417) --, the admission loads C back onto the device, the request ends and C goes back to the
    host. RED on 6f673b4dd6: TP0 drops L at the first broadcast, the worker holds it pending, C is pinned
    (child_host_locked), loaded back (child_on_device) -> S39 DIVERGED, L stays on the worker's device and is gone on
    TP0. GREEN: TP0 leaves the drop alone as long as the worker could not follow (``s39`` veto set), no DIVERGED, and
    once C is a plain host-only node again the drop is taken on BOTH ranks (C with it)."""
    from flliper.srt.pdflip import forma_l3_subtree as s39

    _arena_dir(monkeypatch, tmp_path)
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    lw = worker._t_nodes[1]
    trail = []

    def _state(step):
        trail.append((step, _attached(tp0, tp0._t_nodes[1]), _attached(worker, lw), int(s39._ST.n["diverged"])))

    _as(roles, "worker")
    _register_read(worker, "pdflip-4-108", lw)
    reg = getattr(R12, "worker_read_registered", None)  # the registration publishes the anchor at once
    (reg or R12._dev_refresh)(worker)
    _as(roles, "host")

    _round(roles, tp0, worker)
    _state("read in flight")

    # the read completes on the worker: its 4 tokens become host node C below L, pinned until the admission
    _as(roles, "worker")
    res = worker._insert_helper_host(lw, RadixKey([13, 14, 15, 16], None),
                                     torch.tensor([80, 81, 82, 83], dtype=torch.int64), ["r13", "r14", "r15", "r16"])
    c = res.inserted_host_node
    assert c is not None and c.parent is lw
    c.l3_present = True
    pin = worker.inc_host_lock_ref(c).to_dec_params()
    worker.ongoing_prefetch.clear()
    _as(roles, "host")
    _round(roles, tp0, worker)
    _state("C inserted + pinned")

    # the admission loads C back onto the device and lets the pin go
    _as(roles, "worker")
    c.component_data[FULL].value = worker._t_alloc.alloc(4).to(dtype=torch.int64)
    worker.dec_host_lock_ref(c, pin)
    worker._update_evictable_leaf_sets(lw)
    _as(roles, "host")
    _round(roles, tp0, worker)
    _state("C loaded back")

    # the request ended: C back to the host only (store-backed) -- now the worker can follow
    _as(roles, "worker")
    worker._t_alloc.free(c.component_data[FULL].value)
    c.component_data[FULL].value = None
    worker._update_evictable_leaf_sets(c)
    worker._update_evictable_leaf_sets(lw)
    _as(roles, "host")
    _round(roles, tp0, worker)                          # publishes the cleared veto
    _round(roles, tp0, worker)                          # TP0 decides on it
    _state("C host-only again")

    assert s39._ST.n["diverged"] == 0, ("S39 DIVERGED: L stays on the worker's device, gone on TP0", trail)
    for step, on_tp0, on_worker, _d in trail[:-1]:
        assert on_tp0 and on_worker, ("TP0 dropped the leaf while the worker could not follow", step, trail)
    assert s39._ST.n["worker_veto"] >= 3, s39._ST.n
    for t in (tp0, worker):
        assert not t.root_node.children, ("both ranks dropped L with its subtree", t._t_tag, trail)


def test_hleaf_a1_without_published_sets_the_worker_still_holds_the_drop_pending(roles):
    """The fallback when no worker file is readable (``_worker_files_live`` 0): TP0 decides alone, the worker whose
    own read anchors the leaf keeps the drop PENDING instead of deleting the anchor under its read."""
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    _register_read(worker, "pdflip-4-108", worker._t_nodes[1])
    _round(roles, tp0, worker)
    assert not _attached(tp0, tp0._t_nodes[1])
    assert _attached(worker, worker._t_nodes[1]), "the worker dropped the anchor of its own in-flight read"


def test_hleaf_s1_tp0_never_queues_the_anchor_of_its_own_read(roles):
    """SOLL S1: TP0's in-flight read anchors the leaf -- the peel does not queue it, room_pending() promises nothing
    (PW-R / D-WALL-HEAD would hold the head for room S39 does not deliver)."""
    from flliper.srt.pdflip import forma_l3_subtree as s39

    tp0 = _rank(roles, "tp0")
    _register_read(tp0, "pdflip-4-108", tp0._t_nodes[1])
    _peel(roles, tp0)
    assert s39.room_pending(tp0) == 0 and not s39._ST.queue
    assert s39._ST.n["anchor_skipped"] >= 1
    assert s39.queue_for_room(tp0, [tp0._t_nodes[1]]) == 0
    tp0.ongoing_prefetch.clear()
    _peel(roles, tp0)
    assert s39.room_pending(tp0) == 8                  # CAN-FAIL: the same leaf without the read is queued


def test_hleaf_a_read_anchored_elsewhere_does_not_hold_the_leaf(roles):
    """CAN-FAIL COUNTERWEIGHT: only the read's OWN anchor holds -- a read anchored at the root leaves S39 alone."""
    tp0, worker = _rank(roles, "tp0"), _rank(roles, "worker")
    for t in (tp0, worker):
        _register_read(t, "other", t.root_node)
    free0 = (_free(tp0), _free(worker))
    _peel(roles, tp0), _peel(roles, worker)
    _broadcast(roles, tp0, worker)
    assert (_free(tp0) - free0[0], _free(worker) - free0[1]) == (8, 8)


# --------------------------------------------------------------------------- 2. the completion never hangs below a removed node


def _completion_on_device_anchor(detach: bool):
    """The healthy completion of test_1157 (16 tokens, probed, write_back -- D's policy: the #841 gate is off there)
    re-anchored on a DEVICE-ONLY node X (no Full host copy, as 310). ``detach``: X left the tree while the read was in
    flight (what S39 did to 310)."""
    cache, operation = _reap_scenario(probed=True)
    cache.cache_controller.write_policy = "write_back"
    alloc = cache.token_to_kv_pool_allocator.alloc(8).to(dtype=torch.int64)
    cache.insert(InsertParams(key=RadixKey(list(range(101, 109)), None), value=alloc))
    x = next(iter(cache.root_node.children.values()))
    assert x.component_data[FULL].host_value is None and x.component_data[FULL].value is not None
    rec = cache.ongoing_prefetch[REAP_REQ]
    cache.ongoing_prefetch[REAP_REQ] = rec._replace(anchor_node=x)
    if detach:
        cache.root_node.children.pop(x.key.child_key(cache.page_size))
        cache.evictable_device_leaves.discard(x)
    return cache, x


def _reachable(cache):
    out, stack = set(), [cache.root_node]
    while stack:
        n = stack.pop()
        out.add(n)
        stack.extend(n.children.values())
    return out


def test_hleaf_repro_the_completion_refuses_a_detached_anchor():
    """THE REPRO (node 314): RED on 807de14309 -- the 16 tokens become a host node below the removed X, in
    ``evictable_host_leaves`` and unreachable ("H-leaf extra" + "stale nodes in host_leaves"). GREEN: nothing is
    hung below X, the leaf sets hold only tree nodes, the outcome loads 0 and the refusal is named."""
    cache, x = _completion_on_device_anchor(detach=True)
    records = []
    h = logging.Handler()
    h.emit = records.append
    lg = logging.getLogger(urc.__name__)
    lg.addHandler(h)
    try:
        assert cache.check_prefetch_progress(REAP_REQ)
    finally:
        lg.removeHandler(h)
    assert not x.children, ("the read's span was hung below the removed anchor", [c.id for c in x.children.values()])
    stale = cache.evictable_host_leaves - _reachable(cache)
    assert not stale, ("stale nodes in host_leaves", [n.id for n in stale])
    assert int(cache.prefetch_loaded_tokens_by_reqid[REAP_REQ]) == 0
    assert any("HLEAF-STALE PREFETCH ANCHOR DETACHED" in r.getMessage() for r in records)


def test_hleaf_a_peer_ranks_detached_anchor_refuses_on_this_rank_too():
    """SOLL R11: the anchor left the tree on a PEER rank only (its MIN proposal is 0); this rank's anchor is in its
    tree. RED on 807de14309 (no such vote: this rank inserts, the peer cannot -- the trees part). GREEN: every rank
    of the group refuses alike."""
    cache, x = _completion_on_device_anchor(detach=False)
    cache.tp_world_size = 2
    seen = {}

    def _peer_min(packed, op, label=""):
        if label != "check_prefetch_progress":
            return
        slot = getattr(urc, "_REAP_SLOT_ANCHOR_ATTACHED", None)
        seen["len"] = int(packed.numel())
        if slot is not None:
            seen["local"] = int(packed[slot].item())
            packed[slot] = 0                                    # the peer's anchor left its tree

    cache._all_reduce_attn_groups = _peer_min
    assert cache.check_prefetch_progress(REAP_REQ)
    assert seen.get("local") == 1, seen                         # this rank proposed "in my tree"
    assert seen["len"] == urc._REAP_PACKED_LEN
    assert not x.children, "this rank inserted while its peer refused -- the trees part (R11)"
    assert int(cache.prefetch_loaded_tokens_by_reqid[REAP_REQ]) == 0


def test_hleaf_an_attached_device_anchor_still_takes_the_read():
    """CAN-FAIL COUNTERWEIGHT: the anchor is in the tree -- the read's 16 tokens are inserted below it as before."""
    cache, x = _completion_on_device_anchor(detach=False)
    assert cache.check_prefetch_progress(REAP_REQ)
    assert len(x.children) == 1
    child = next(iter(x.children.values()))
    assert child.component_data[FULL].host_value is not None and len(child.key) == REAP_TOKENS
    assert int(cache.prefetch_loaded_tokens_by_reqid[REAP_REQ]) == REAP_TOKENS


def test_hleaf_the_attached_predicate_reads_the_whole_chain_and_this_trees_root():
    """A node whose PARENT left the tree is detached too; a node of a tree ``_reset_full`` replaced (its chain ends at
    another root) is detached; the root and a node in the tree are attached."""
    cache, x = _completion_on_device_anchor(detach=False)
    f = urc.UnifiedRadixCache._pdflip_prefetch_anchor_attached
    assert f(cache, cache.root_node) and f(cache, x)
    alloc = cache.token_to_kv_pool_allocator.alloc(12).to(dtype=torch.int64)
    cache.insert(InsertParams(key=RadixKey(list(range(101, 113)), None), value=alloc))
    y = next(iter(x.children.values()))
    assert f(cache, y)
    cache.root_node.children.pop(x.key.child_key(cache.page_size))
    assert not f(cache, x) and not f(cache, y)                  # y still in x.children -- x is gone
    cache.root_node.children[x.key.child_key(cache.page_size)] = x
    old_root = cache.root_node
    cache.root_node = type(old_root)(cache.tree_components)     # a new tree: the old chain ends at another root
    assert not f(cache, x)


# --------------------------------------------------------------------------- review S2 / S3


class _Group3:
    """Three ranks in threads, one collective sequence: every ``_all_reduce_attn_groups`` is an element-wise MIN /
    MAX / SUM over the three tensors (what gloo hands back), every barrier a barrier."""

    def __init__(self, world=3):
        self.world = world
        self.bar = threading.Barrier(world, timeout=20)
        self.slots = {}
        self.labels = {r: [] for r in range(world)}

    def bind(self, cache, rank):
        def _reduce(t, op, label="hicache"):
            self.labels[rank].append((label, int(t.numel())))
            self.slots[rank] = t.clone()
            self.bar.wait()
            stack = torch.stack([self.slots[r] for r in range(self.world)])
            if op == torch.distributed.ReduceOp.MAX:
                red = stack.max(dim=0).values
            elif op == torch.distributed.ReduceOp.SUM:
                red = stack.sum(dim=0)
            else:
                red = stack.min(dim=0).values
            self.bar.wait()
            t.copy_(red.to(t.dtype))

        cache._all_reduce_attn_groups = _reduce
        cache._barrier_attn_groups = lambda label="hicache": self.bar.wait()
        cache.tp_world_size = self.world


def _released(cache):
    out = []
    q = cache.cache_controller.host_mem_release_queue
    while not q.empty():
        out.extend(int(x) for x in q.get_nowait().tolist())
    return sorted(out)


def test_hleaf_s2_three_ranks_one_detached_every_rank_refuses_and_returns_its_rows_once():
    """SOLL S2: three ranks, the anchor left the tree on rank 2 only (the boot's TP0 dropped 310 first). RED on
    40b569aee8: ranks 0/1 insert below their anchor, rank 2 hangs the span below the removed node (the stale host leaf).
    GREEN: every rank refuses alike, the same collective sequence on all three, and per rank: nothing below the
    anchor, ``ongoing_prefetch`` empty, ``prefetch_tokens_occupied`` back to 0, the release queue holds exactly the
    read's 16 host rows, once."""
    ranks = []
    for r in range(3):
        cache, x = _completion_on_device_anchor(detach=(r == 2))
        rows = sorted(int(v) for v in cache.ongoing_prefetch[REAP_REQ].host_indices.tolist())
        ranks.append((cache, x, rows))
    g = _Group3()
    for r, (cache, _, _) in enumerate(ranks):
        g.bind(cache, r)
    errs = []

    def _run(r):
        try:
            assert ranks[r][0].check_prefetch_progress(REAP_REQ)
        except BaseException as e:  # noqa: BLE001
            errs.append((r, e))
            g.bar.abort()

    th = [threading.Thread(target=_run, args=(r,)) for r in range(3)]
    for t in th:
        t.start()
    for t in th:
        t.join(30)
    assert not errs, errs
    assert g.labels[0] == g.labels[1] == g.labels[2], g.labels
    for r, (cache, x, rows) in enumerate(ranks):
        assert not x.children, ("rank", r, "inserted below its anchor while a peer refused")
        assert not (cache.evictable_host_leaves - _reachable(cache)), ("rank", r, "stale host leaf")
        assert REAP_REQ not in cache.ongoing_prefetch
        assert cache.cache_controller.prefetch_tokens_occupied == 0
        assert _released(cache) == rows, ("rank", r, "release queue != the read's rows once")
        assert int(cache.prefetch_loaded_tokens_by_reqid[REAP_REQ]) == 0


def test_hleaf_s2_the_refusals_insert_result_releases_the_reads_mamba_anchor_and_hangs_nothing():
    """SOLL S2, the ``+lru={314}`` half: the REAL ``MambaComponent.commit_hicache_transfer`` PREFETCH branch fed with the
    refusal's InsertResult (``host_span_unclaimed``, no inserted node, no matched end) releases the read's anchor slot
    to the pool and puts nothing on the mamba host LRU. The pre-fix shape (the inserted node 314 as target) is what put
    314 there -- shown as the counterweight."""
    import types

    from flliper.srt.mem_cache.base_prefix_cache import InsertResult
    from flliper.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
    from flliper.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent
    from flliper.srt.mem_cache.unified_cache_components.tree_component import CacheTransferPhase

    MAMBA = ComponentType.MAMBA

    class _Lru:
        def __init__(self):
            self.ids = set()

        def in_list(self, n):
            return id(n) in self.ids

        def insert_mru(self, n):
            self.ids.add(id(n))

    def _stub():
        released = []
        cc = types.SimpleNamespace(append_host_mem_release=lambda **kw: released.append(kw))
        cache = types.SimpleNamespace(cache_controller=cc, host_lru_lists={MAMBA: _Lru()}, root_node=object())
        comp = types.SimpleNamespace(component_type=MAMBA, cache=cache)
        comp._prefetch_anchor_target = MambaComponent._prefetch_anchor_target.__get__(comp)
        return comp, released

    def _xfer():
        return PoolTransfer(name=PoolName.MAMBA, host_indices=torch.tensor([7], dtype=torch.int64))

    hit = types.SimpleNamespace(extra_pool_hit_pages={PoolName.MAMBA: 1})
    comp, released = _stub()
    t = _xfer()
    res = InsertResult(prefix_len=0, host_span_unclaimed=True)
    MambaComponent.commit_hicache_transfer(comp, object(), CacheTransferPhase.PREFETCH, [t],
                                           insert_result=res, pool_storage_result=hit)
    assert released == [{"extra_pools": [t]}] and not comp.cache.host_lru_lists[MAMBA].ids
    # counterweight: the pre-fix completion named the node it hung below the removed anchor
    comp, released = _stub()
    n314 = types.SimpleNamespace(component_data={
        ComponentType.FULL: types.SimpleNamespace(host_value=torch.tensor([1]), value=None),
        MAMBA: types.SimpleNamespace(host_value=None, value=None)}, id=314)
    MambaComponent.commit_hicache_transfer(comp, object(), CacheTransferPhase.PREFETCH, [_xfer()],
                                           insert_result=InsertResult(prefix_len=0, inserted_host_node=n314),
                                           pool_storage_result=hit)
    assert not released and comp.cache.host_lru_lists[MAMBA].ids == {id(n314)}


def test_hleaf_s3_a_refused_read_does_not_end_the_handoffs_protection(monkeypatch):
    """SOLL S3: the full read the group refused for its detached anchor took nothing over -- ``_pdflip_handoff_consumed``
    stays silent (P's hand-off keeps its #243 protection for the re-read). CAN-FAIL: the same read on an attached
    anchor ends it."""
    calls = []
    monkeypatch.setattr(urc, "_pdflip_handoff_consumed", lambda rid, where, **kv: calls.append((rid, where)))
    cache, _ = _completion_on_device_anchor(detach=True)
    assert cache.check_prefetch_progress(REAP_REQ)
    assert not calls, calls
    cache, _ = _completion_on_device_anchor(detach=False)
    assert cache.check_prefetch_progress(REAP_REQ)
    assert calls == [(REAP_REQ, "fetch")], calls
