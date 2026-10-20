"""S108 (20.10.): D copies every finished turn L1 -> L2 at once, and the scheduler thread never waits.

User 20.10.: "JA NATÜRLICH" (D copies finished turns to L2 at once) and "UND WEHE DAS SCHREIBEN IN
DEN CACHE BLOCKIERT DURCH IRGENDEINE SYNCHRON ODER WARTESCHEIßE".

THE NAIVE WAY IS SYNCHRONOUS (S108 Vorbefund, rc27 1a9b2d9d03): ``write_backup`` -> the hybrid
``write``/``start_writing`` read device values on D's token cut -- ``int(rows.max())``/``min()`` (#923),
``compact[owned]`` (boolean mask), ``owned.to(cpu)``, the direct backend's ``.cpu()`` and, on TP1/TP2,
``_backup_owner_rows``' per-layer ``.to("cpu")``. Every one of them waits for the forward in flight.

WHAT MUST HOLD (CPU only, no GPU):
(1) the turn-end issue in the scheduler thread reads NO device value and waits on NO event or stream
    (a sync detector counts every device->host read: 0) -- the inline path counts > 0 (detector live);
(2) decode rounds run while the turn-end copy is still held by the copier, and the turn end returned
    before the copy (mutant "synchroner Copy-Wait": red);
(3) D hooks the finished turn (mutant "nur beim Flip" / gate off: red);
(4) three ranks with differently cut trees and the Form A owner cut [0, 9, 7] (S=16): each copies its
    own share exactly, the union is every token once, the ack is not ready before the copier ran;
(5) a full arena defers by name and never spills; the deferred node is retried at the next turn end;
(6) a #923 hit on the copier is raised by name in the scheduler thread at the ack poll;
(7) the instrument lines carry bytes, share, issue/blocked ms and copy ms; reset joins the copier.
"""
from __future__ import annotations

import inspect
import logging
import os
import threading
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402
import torch  # noqa: E402
from torch.overrides import TorchFunctionMode  # noqa: E402

from flliper.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hcc  # noqa: E402
from flliper.srt.mem_cache import unified_radix_cache as urc  # noqa: E402
from flliper.srt.pdflip import d_turn_end_l2 as te  # noqa: E402

PAGE = 64
S16 = 16
CUT = {0: (S16, 0, 0), 1: (S16, 0, 9), 2: (S16, 9, 16)}   # NF D: Form A x #239 token cut [0, 9, 7]


# ----------------------------------------------------------------------------- sync detector
class DevTensor(torch.Tensor):
    """Marks a tensor that lives on the card in production (CPU in this test)."""


def dev(t: torch.Tensor) -> DevTensor:
    return torch.Tensor._make_subclass(DevTensor, t)


def _has_dev(x) -> bool:
    if isinstance(x, DevTensor):
        return True
    if isinstance(x, (tuple, list)):
        return any(_has_dev(y) for y in x)
    return False


def _bool_index(x) -> bool:
    if isinstance(x, torch.Tensor):
        return x.dtype == torch.bool
    if isinstance(x, (tuple, list)):
        return any(_bool_index(y) for y in x)
    return False


_T = torch.Tensor
_SYNC_FUNCS = {_T.item, _T.tolist, _T.__int__, _T.__index__, _T.__bool__, _T.__float__, _T.cpu,
               _T.numpy, _T.nonzero, torch.nonzero, torch.masked_select, _T.masked_select,
               torch.unique, _T.unique}


class SyncCounter(TorchFunctionMode):
    """Counts every device->host read on a DevTensor in THIS thread (modes are thread-local):
    scalar conversions, host copies, data-dependent shapes (boolean masks, nonzero, unique)."""

    def __init__(self):
        super().__init__()
        self.hits = []

    def __torch_function__(self, func, types_, args=(), kwargs=None):
        kwargs = kwargs or {}
        if _has_dev(args) or _has_dev(list(kwargs.values())):
            name = getattr(func, "__name__", str(func))
            if func in _SYNC_FUNCS:
                self.hits.append(name)
            elif func is _T.to and any(
                    (a == "cpu" or (isinstance(a, torch.device) and a.type == "cpu"))
                    for a in list(args[1:]) + [kwargs.get("device")]):
                self.hits.append("to(cpu)")
            elif func is _T.__getitem__ and _bool_index(args[1:]):
                self.hits.append("bool_mask_getitem")
        return func(*args, **kwargs)


# ----------------------------------------------------------------------------- fake card
class _Event:
    def __init__(self, dm):
        self.dm = dm
        self.recorded_in = None

    def record(self, stream=None):
        self.recorded_in = threading.current_thread().name

    def wait(self, stream):
        self.dm.waits.append(threading.current_thread().name)

    def query(self):
        return True

    def synchronize(self):
        self.dm.syncs.append(threading.current_thread().name)


class _Stream:
    def __init__(self, dm, name):
        self.dm, self.name = dm, name

    def wait_event(self, ev):
        self.dm.waits.append(threading.current_thread().name)

    def query(self):
        return True

    def synchronize(self):
        self.dm.syncs.append(threading.current_thread().name)


class _Ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Dev:
    def __init__(self):
        self.waits, self.syncs = [], []
        self.compute = _Stream(self, "compute")

    def Event(self):  # noqa: N802
        return _Event(self)

    def stream(self, s):
        return _Ctx()

    def current_stream(self):
        return self.compute


class _HostPool:
    layout = "layer_first"
    entry_map = {}

    def __init__(self, gate=None):
        self.pairs = []
        self.gate = gate
        self._next = 0
        self.copier_threads = set()

    def alloc(self, n):
        h = torch.arange(self._next, self._next + n, dtype=torch.int64)
        self._next += n
        return h

    def free(self, h):
        pass

    def backup_from_device_all_layer(self, device_pool, host, device, io_backend, pool_transfers=None):
        if self.gate is not None:
            assert self.gate.wait(10), "test gate never opened"
        self.copier_threads.add(threading.current_thread().name)
        self.pairs.append((host.clone(), torch.Tensor(device).clone()))

    def backup_from_device_indices(self, *a, **kw):  # pragma: no cover - not taken (dcp)
        raise AssertionError("device-index form is refused under the owner rule")


class _DevPool:
    def __init__(self, rows):
        self.k_buffer = [torch.zeros(rows, 2, 4, dtype=torch.bfloat16) for _ in range(3)]
        self.v_buffer = [torch.zeros(rows, 2, 4, dtype=torch.bfloat16) for _ in range(3)]


def _controller(ctx, gate=None, compact_rows=32768):
    c = object.__new__(hcc.HybridCacheController)
    c.write_queue = []
    c.ack_write_queue = []
    c.io_backend = "direct"
    c.mem_pool_host = _HostPool(gate)
    c.mem_pool_device = _DevPool(compact_rows)
    c.write_stream = _Stream(None, "write")
    c.draft_tier_armed = lambda kind: False
    c._record_transfer_indices_on_stream = lambda *a, **kw: None
    c._dcp_owner_ctx_cache = ctx
    c.enable_storage = False
    return c


@pytest.fixture
def card(monkeypatch):
    dm = _Dev()
    monkeypatch.setattr(hcc, "device_module", dm)
    monkeypatch.setattr(hcc, "consume_gate", lambda *a, **kw: True)
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "D")
    return dm


def _slots(n_pages, seed=0):
    """Page-aligned allocator slots for positions 0..n_pages*64-1 (paged allocator: slot % 64 ==
    position % 64; pages drawn from a shuffled free list)."""
    g = torch.Generator().manual_seed(seed)
    pages = torch.randperm(4096 // PAGE * S16 // S16 * 8, generator=g)[:n_pages]
    return (pages[:, None] * PAGE + torch.arange(PAGE)[None, :]).reshape(-1).to(torch.int64)


def _compact(slots, ctx):
    S, lo, hi = ctx
    off = slots % S
    owned = (off >= lo) & (off < hi)
    return ((slots // S) * (hi - lo) + (off - lo))[owned]


def _wait_copier(c, t=10):
    cp = c.__dict__.get("_s108_copier")
    assert cp is None or cp.join(t)


# ----------------------------------------------------------------------------- (1) no sync
def test_turn_end_issue_reads_no_device_value_and_waits_on_nothing(card):
    gate = threading.Event()
    c = _controller(CUT[1], gate)
    slots = dev(_slots(4, seed=1))
    with SyncCounter() as sc, te.deferred_issue(c, "rid-1"):
        host = c.write(slots, node_id=11)
    assert host is not None
    assert sc.hits == [], sc.hits                              # no device->host read in this thread
    me = threading.current_thread().name
    assert me not in card.syncs and me not in card.waits       # no event/stream wait in this thread
    assert len(c.ack_write_queue) == 1
    assert c.ack_write_queue[0].finish_event.query() is False  # the copy is still the copier's
    gate.set()
    _wait_copier(c)
    assert c.ack_write_queue[0].finish_event.query() is True
    (h, d), = c.mem_pool_host.pairs
    assert torch.equal(d, _compact(torch.Tensor(slots), CUT[1]))
    assert me not in c.mem_pool_host.copier_threads


def test_the_detector_is_live_the_inline_issue_reads_the_card(card):
    """RED form (the naive way, rc27 inline write on D): #923 int(rows.max()), compact[owned],
    owned.to(host) -- counted."""
    c = _controller(CUT[1])
    with SyncCounter() as sc:
        assert c.write(dev(_slots(2, seed=2)), node_id=12) is not None
    assert {"__int__", "bool_mask_getitem"} <= set(sc.hits), sc.hits


# ----------------------------------------------------------------------------- (2) decode runs on
def _turn_end_returns_while_copier_holds(c, card, rounds=5):
    """True when the turn end returns while the copier still holds the op, and decode rounds
    (here: the scheduler's non-blocking ack poll + a forward record) run meanwhile without waiting."""
    gate = c.mem_pool_host.gate
    out = {}

    def scheduler():
        t0 = time.perf_counter()
        with te.deferred_issue(c, "rid-2"):
            c.write(dev(_slots(2, seed=3)), node_id=21)
        out["issue_s"] = time.perf_counter() - t0
        polls = []
        for _ in range(rounds):                       # decode rounds during the copy
            polls.append(all(a.finish_event.query() for a in c.ack_write_queue))
            card.compute.wait_event  # noqa: B018 - a forward is launched (nothing waits)
        out["polls"] = polls

    th = threading.Thread(target=scheduler, name="scheduler")
    th.start()
    th.join(2.0)
    returned = not th.is_alive()
    gate.set()
    th.join(10)
    _wait_copier(c)
    return returned and out.get("polls") == [False] * rounds


def test_decode_rounds_run_during_the_turn_end_copy_without_extra_wait(card):
    c = _controller(CUT[2], threading.Event())
    assert _turn_end_returns_while_copier_holds(c, card)
    assert "scheduler" not in card.syncs


def test_mutant_synchronous_copy_wait_is_red(card, monkeypatch):
    real = hcc.HybridCacheController._s108_submit

    def submit_and_wait(self):
        real(self)
        self.ack_write_queue[-1].finish_event.synchronize()   # the mutant: wait for the copy

    monkeypatch.setattr(hcc.HybridCacheController, "_s108_submit", submit_and_wait)
    c = _controller(CUT[2], threading.Event())
    assert not _turn_end_returns_while_copier_holds(c, card)


# ----------------------------------------------------------------------------- (3) D hooks the finish
class _Node:
    _n = 100

    def __init__(self, parent, ntok, slots):
        _Node._n += 1
        self.id = _Node._n
        self.parent = parent
        self.children = {}
        self.key = list(range(ntok))
        self.evicted = False
        self.backuped = False
        self.l3_present = False
        self.write_through_pending_id = None
        self.component_data = {urc.BASE_COMPONENT_TYPE: types.SimpleNamespace(value=slots)}
        if parent is not None:
            parent.children[self.id] = self


def _tree(cc, cuts, slots, refuse=()):
    """A stub tree over `cc`: the chain `cuts` (token counts, page multiples) over `slots`; its
    sweep backs each node through the REAL controller write (parents first), refusing `refuse`."""
    t = types.SimpleNamespace()
    t.cache_controller = cc
    t.enable_storage = True
    root = _Node(None, 0, None)
    t.root_node = root
    chain, p, at = [], root, 0
    for n in cuts:
        p = _Node(p, n, dev(slots[at:at + n]))
        chain.append(p)
        at += n
    t.chain = chain
    t._pdflip_chain_nodes = lambda key: list(chain)
    t._s108_in_tree = lambda n: urc.UnifiedRadixCache._s108_in_tree(t, n)
    t.calls = []
    t.refuse = set(refuse)

    def sweep(max_issue, clock, first, chain_only):
        t.calls.append(dict(first=[n.id for n in first], chain_only=chain_only,
                            deferred=cc.__dict__.get("_s108_defer_issue"),
                            active=t.__dict__.get("_s108_turn_end_active")))
        st = {"unbacked": 0, "issued": 0, "refused_by": {}}
        for n in first:
            if n.backuped or n.write_through_pending_id is not None:
                continue
            st["unbacked"] += 1
            if n.id in t.refuse or (n.parent is not root and not n.parent.backuped
                                  and n.parent.write_through_pending_id is None):
                st["refused_by"]["arena_claim"] = st["refused_by"].get("arena_claim", 0) + 1
                continue
            if cc.write(n.component_data[urc.BASE_COMPONENT_TYPE].value, node_id=n.id) is not None:
                n.write_through_pending_id = n.id
                st["issued"] += 1
        return st

    t.publish_unbacked_sweep = sweep
    return t


def _turn_end(t, rid="rid-x"):
    req = types.SimpleNamespace(rid=rid)
    urc.UnifiedRadixCache._s108_turn_end_l2(t, req, radix_key=object())


def test_d_copies_the_finished_turns_chain_through_the_copier(card):
    c = _controller(CUT[1])
    t = _tree(c, [128, 64], _slots(3, seed=4))
    with SyncCounter() as sc:
        _turn_end(t)
    assert sc.hits == []
    assert t.calls == [dict(first=[n.id for n in t.chain], chain_only=True, deferred=True, active=True)]
    assert c.__dict__.get("_s108_defer_issue") is False and t._s108_turn_end_active is False
    _wait_copier(c)
    assert len(c.mem_pool_host.pairs) == 2 and all(a.finish_event.query() for a in c.ack_write_queue)


def test_the_finish_path_calls_the_turn_end_hook_after_the_insert():
    src = inspect.getsource(urc.UnifiedRadixCache.cache_finished_req)
    i_ins = src.index("result = self.insert(insert_params)")
    i_hook = src.index("UnifiedRadixCache._s108_turn_end_l2(self, req, radix_key)")
    assert i_ins < i_hook
    assert te.turn_end_on({"FLLIPER_PDFLIP_GROUP": "D"}) is True
    assert te.turn_end_on({"FLLIPER_PDFLIP_GROUP": "P"}) is False


def test_mutant_only_at_the_flip_is_red(card, monkeypatch):
    """Mutant: the turn end does nothing on D (the copy waits for the flip's sweep) -- no op."""
    monkeypatch.setattr(te, "turn_end_on", lambda env=None: False)
    c = _controller(CUT[1])
    t = _tree(c, [64], _slots(1, seed=5))
    _turn_end(t)
    assert t.calls == [] and c.ack_write_queue == []          # what the real form never does


# ----------------------------------------------------------------------------- (4) three ranks
def test_three_ranks_differently_cut_trees_each_copy_their_share(card):
    slots = _slots(6, seed=6)                                  # 384 tokens, one replicated slot list
    cuts = {0: [128, 256], 1: [64, 128, 192], 2: [384]}        # rank-local splits
    gate = threading.Event()
    ctl = {r: _controller(CUT[r], gate) for r in range(3)}
    trees = {r: _tree(ctl[r], cuts[r], slots) for r in range(3)}
    for r in range(3):
        with SyncCounter() as sc:
            _turn_end(trees[r], rid="rid-3")
        assert sc.hits == [], (r, sc.hits)
        assert ctl[r].ack_write_queue and not any(a.finish_event.query() for a in ctl[r].ack_write_queue)
    gate.set()
    for r in range(3):
        _wait_copier(ctl[r])
        assert all(a.finish_event.query() for a in ctl[r].ack_write_queue)
    seen = torch.zeros(len(slots), dtype=torch.int64)
    for r in range(3):
        got = torch.cat([d for _h, d in ctl[r].mem_pool_host.pairs])
        assert torch.equal(got, _compact(slots, CUT[r])), r
        S, lo, hi = CUT[r]
        seen += ((slots % S >= lo) & (slots % S < hi)).to(torch.int64)
    assert int(ctl[0].mem_pool_host.pairs[0][1].numel()) == 0   # TP0: owner share 0 (index/draft only)
    assert torch.equal(seen, torch.ones_like(seen))             # every token exactly once over the group


# ----------------------------------------------------------------------------- (5) full arena
def test_full_arena_defers_by_name_never_spills_and_retries(card, caplog):
    c = _controller(CUT[2])
    t = _tree(c, [64, 64], _slots(2, seed=7))
    a, b = t.chain
    t.refuse = {a.id}                                           # the arena refuses the head's claim
    with caplog.at_level(logging.WARNING):
        _turn_end(t, rid="rid-full")
    assert any("S108 TURN-END-DEFER rid=rid-full why=arena_claim:2 nodes=2" in r.message
               for r in caplog.records), [r.message for r in caplog.records]
    assert set(t._s108_deferred) == {a.id, b.id}               # kept, not dropped
    assert c.ack_write_queue == []                              # nothing issued, nothing waited for
    # the spill never runs inside a turn end
    stub = types.SimpleNamespace(_s108_turn_end_active=True)
    assert urc.UnifiedRadixCache._w3_arena_spill(stub, object(), 4) == 0 and stub._s108_spill_skipped == 1
    # the next turn end (another request, no chain of its own here) retries them, now with room
    t.refuse = set()
    t._pdflip_chain_nodes = lambda key: []
    _turn_end(t, rid="rid-next")
    assert t.calls[-1]["first"] == [a.id, b.id]
    _wait_copier(c)
    assert not t._s108_deferred and len(c.mem_pool_host.pairs) == 2


# ----------------------------------------------------------------------------- (6) #923 on the copier
def test_a_923_hit_on_the_copier_is_raised_by_name_at_the_ack_poll(card):
    c = _controller(CUT[1], compact_rows=8)                    # a pool far too small for the rows
    with te.deferred_issue(c, "rid-923"):
        assert c.write(dev(_slots(1, seed=9)), node_id=31) is not None
    _wait_copier(c)
    with pytest.raises(te.TurnEndCopyError, match="#923 HICACHE write ROW OUT OF RANGE"):
        c.ack_write_queue[0].finish_event.query()
    assert c.mem_pool_host.pairs == []                          # nothing was copied


# ----------------------------------------------------------------------------- (7) instrument, reset
def test_instrument_lines_and_reset_joins_the_copier(card, caplog, monkeypatch):
    gate = threading.Event()
    c = _controller(CUT[1], gate)
    t = _tree(c, [128], _slots(2, seed=10))
    with caplog.at_level(logging.INFO):
        _turn_end(t, rid="rid-i")
        line = next(r.message for r in caplog.records if r.message.startswith("S108 TURN-END-L2"))
        assert "rid=rid-i chain=1 unbacked=1 issued=1 deferred=0 ops=1 tokens=128 share=9/16" in line
        assert "blocked_ms=" in line and "issue_cpu_ms=" in line
        monkeypatch.setattr(hcc.BaseHiCacheController, "reset", lambda self: None)
        threading.Timer(0.2, gate.set).start()
        c.reset()                                                # waits for the copier, then resets
        assert c._s108_copier.inflight == 0
    copy = next(r.message for r in caplog.records if r.message.startswith("S108 TURN-END-COPY"))
    row_bytes = 3 * 2 * (2 * 4 * 2)                              # 3 layers x K,V x (2x4 bf16)
    assert f"tag=rid-i kv_rows={128 * 9 // 16} kv_bytes={128 * 9 // 16 * row_bytes}" in copy
    assert "copy_ms=" in copy


# ----------------------------------------------------------------------------- (5b) no room-making
def test_turn_end_claims_only_free_room_never_makes_room(monkeypatch):
    """The claim's own room-making (reap, _evict_for_claim with its L3 secure -- y9e: 19-109 ms in
    the scheduler thread) never runs for a turn end: the probe sees no free room and defers."""
    from flliper.srt.mem_cache.pool_host import arena_pool as ap

    class _Arena:
        def __init__(self, free):
            self.free = free

        def find_states(self, stems):
            return [0 for _ in stems]                           # every page absent

        def stats(self):
            return {"slots": 10, "complete": 10 - self.free, "claimed": 0}

    pool = object.__new__(ap.ArenaMHAHostPool)
    pool.arena, pool._backend = _Arena(free=1), object()
    pool._stems = lambda hs: list(hs)
    assert pool.claim_needs_room(["a", "b"]) is True
    pool.arena.free = 2
    assert pool.claim_needs_room(["a", "b"]) is False

    pool.arena.free = 0
    pool.alloc_write = lambda hs: (_ for _ in ()).throw(AssertionError("claimed although no free room"))
    why = []
    node = types.SimpleNamespace(id=7, hash_value=["a"], component_data={
        urc.BASE_COMPONENT_TYPE: types.SimpleNamespace(value=torch.arange(PAGE))})
    stub = types.SimpleNamespace(_s108_turn_end_active=True, page_size=PAGE,
                                 cache_controller=None, _pdflip_direct_pool=lambda: pool,
                                 _1421_refused=lambda w, n: why.append(w))
    assert urc.UnifiedRadixCache._pdflip_direct_claim(stub, node) is False
    assert why == ["arena_room"] and stub._pdflip_sweep_last_refusal == "arena_claim"
