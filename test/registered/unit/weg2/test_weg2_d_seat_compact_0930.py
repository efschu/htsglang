"""D-SEAT-REWAKE COMPACT (NF-Operator 30.09., Produktentscheid): an EMPTY D moves
the tree's GDN states above a smaller seat count's slot limit DOWN, so it can
shrink to the fewest seats.

Wörtlich: "KOMPAKTIEREN, nicht verdrängen. Die GDN-Zustände des Baums, die über
dem Limit liegen, ziehen per Device-Kopie in freie Slots darunter um; die
Slot-Referenzen im Baum werden dabei umgehängt ... Verdrängen nur als Rückfall:
wenn unten kein Platz frei ist und der Zustand schon im L2 gesichert ist
(backuped). Nie einen ungesicherten Zustand verlieren."

Befund y4x (16:33:26-36Z): D empty, n=6, the tree held GDN states above the
limit of n=1 (7); the idle SHRINK of 50bed49e3a went only as far as the highest
held slot allowed.

Real tensors: a real UnifiedRadixCache over a real HybridReqToTokenPool (its
MambaPool's conv/temporal on CPU, its MambaSlotAllocator), states written into
the slots, the move checked byte for byte.
"""
from __future__ import annotations

import logging
import os
import types
from array import array
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape  # noqa: E402
from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE  # noqa: E402
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator  # noqa: E402
from sglang.srt.mem_cache.base_prefix_cache import InsertParams  # noqa: E402
from sglang.srt.mem_cache.cache_init_params import CacheInitParams  # noqa: E402
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool  # noqa: E402
from sglang.srt.mem_cache.radix_cache import RadixKey  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler  # noqa: E402
from sglang.srt.weg2 import d_seat_compact as C  # noqa: E402
from sglang.srt.weg2 import d_seat_rewake as R  # noqa: E402
from sglang.srt.weg2 import d_seat_vram as dsv  # noqa: E402

NUM_LAYERS = 8
FULL = (3, 7)
LIN = [i for i in range(NUM_LAYERS) if i not in FULL]
SIZE = 38                      # the NF form: limits 7/13/19/25/32/38 at 6 seats
MC = ComponentType.MAMBA


def _build():
    sa = ServerArgs(model_path="dummy", page_size=1)
    sa._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    sa.max_running_requests = 6
    sa.enable_hierarchical_cache = False
    sa.disable_radix_cache = False
    sa.disable_overlap_schedule = True
    set_global_server_args_for_scheduler(sa)
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(tp_world_size=1, intermediate_size=256, n_groups=1, num_heads=2,
                                        head_dim=16, state_size=16, conv_kernel=4)
        cp = Mamba2CacheParams(shape=shape, layers=LIN)
    pool = HybridReqToTokenPool(size=10, mamba_size=SIZE, mamba_spec_state_size=10, max_context_len=256,
                                device="cpu", enable_memory_saver=False, cache_params=cp,
                                mamba_layer_ids=LIN, enable_mamba_extra_buffer=False,
                                speculative_num_draft_tokens=3)
    kv = HybridLinearKVPool(size=512, dtype=torch.bfloat16, page_size=1, head_num=2, head_dim=64,
                            full_attention_layer_ids=list(FULL), device="cpu", enable_memory_saver=False,
                            mamba_pool=pool.mamba_pool)
    alloc = TokenToKVPoolAllocator(size=512, dtype=torch.bfloat16, device="cpu", kvcache=kv, need_sort=False)
    params = CacheInitParams(req_to_token_pool=pool, token_to_kv_pool_allocator=alloc, page_size=1,
                             disable=False, sliding_window_size=None,
                             tree_components=(ComponentType.FULL, ComponentType.MAMBA),
                             enable_mamba_extra_buffer=False, enable_kv_cache_events=False,
                             eviction_policy="lru", is_eagle=False)
    return UnifiedRadixCache(params=params), pool, alloc


def _take(mal, slots):
    """Put the allocator in the state 'exactly these slots in use' (real ledger)."""
    got = mal.alloc(SIZE)
    keep = torch.tensor(sorted(slots), dtype=torch.int64)
    mal.free(got[~torch.isin(got, keep)])


def _state(pool, slot, seed):
    """Write a distinctive state into ``slot`` (every conv tensor + temporal)."""
    g = torch.Generator().manual_seed(seed)
    mc = pool.mamba_pool.mamba_cache
    for t in list(mc.conv) + [mc.temporal]:
        t[:, slot] = torch.randn(t[:, slot].shape, generator=g).to(t.dtype)


def _bytes_at(pool, slot):
    mc = pool.mamba_pool.mamba_cache
    return [t[:, slot].clone() for t in list(mc.conv) + [mc.temporal]]


def _tree(slots, backed=(), base=1000):
    """A real tree whose leaves hold GDN states in ``slots`` (one leaf each)."""
    cache, pool, kalloc = _build()
    _take(pool.mamba_allocator, slots)
    nodes = {}
    for i, s in enumerate(slots):
        _state(pool, s, seed=s)
        toks = [base + 10 * i + j for j in range(4)]
        cache.insert(InsertParams(key=RadixKey(array("q", toks)), value=kalloc.alloc(4),
                                  mamba_value=torch.tensor([s], dtype=torch.int64)))
        node = [x for x in cache._collect_all_nodes()
                if x.component_data[MC].value is not None and int(x.component_data[MC].value[0]) == s][0]
        if s in backed:
            node.component_data[MC].host_value = torch.tensor([s], dtype=torch.int64)
        nodes[s] = node
    return cache, pool, nodes


# ---------------------------------------------------------------- the move, byte for byte

def test_the_tree_states_above_the_limit_move_down_by_device_copy():
    cache, pool, nodes = _tree([10, 20, 30])
    before = {s: _bytes_at(pool, s) for s in nodes}
    mal = pool.mamba_allocator
    view = C.TreeView(cache)
    plan, why = C.plan_for(C.used_ids(mal.slot_used), view.anchors(), dsv.phase_slot_limit(SIZE, 1, 6))
    assert why == "" and plan.limit == 7 and not plan.evicts
    assert sorted(d for _a, d in plan.moves) == [1, 2, 3]
    res = C.execute(cache, mal, view, plan)
    assert (res.moved, res.evicted) == (3, 0)
    assert res.bytes == 3 * C.slot_bytes(pool.mamba_pool) > 0
    for s, node in nodes.items():
        new = int(node.component_data[MC].value[0])
        assert new <= 7, "the slot reference was re-pointed"
        assert all(torch.equal(a, b) for a, b in zip(before[s], _bytes_at(pool, new))), "every byte moved"
    assert C.used_ids(mal.slot_used) == [1, 2, 3]
    assert mal.set_phase_limit(7, seats=1), "nothing above the limit of n=1 is in use now"
    assert cache.component_evictable_size_[MC] == 3


def test_a_moved_state_still_answers_its_prefix():
    cache, pool, nodes = _tree([25])
    view = C.TreeView(cache)
    plan, _ = C.plan_for(C.used_ids(pool.mamba_allocator.slot_used), view.anchors(), 7)
    C.execute(cache, pool.mamba_allocator, view, plan)
    from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams

    m = cache.match_prefix(MatchPrefixParams(key=RadixKey(array("q", [1000, 1001, 1002, 1003]))))
    assert len(m.device_indices) == 4 and m.last_device_node is nodes[25]
    assert int(nodes[25].component_data[MC].value[0]) == 1


# ---------------------------------------------------------------- the fallback, never losing a state

def test_no_free_slot_below_an_unbacked_state_is_never_lost():
    low = list(range(1, 8))
    cache, pool, nodes = _tree(low + [20])
    plan, why = C.plan_for(C.used_ids(pool.mamba_allocator.slot_used), C.TreeView(cache).anchors(), 7)
    assert plan is None and "un-backed" in why and "[20]" in why
    assert nodes[20].component_data[MC].value is not None


def test_no_free_slot_below_a_backed_state_is_evicted_its_host_copy_stays():
    low = list(range(1, 8))
    cache, pool, nodes = _tree(low + [20], backed=(20,))
    mal = pool.mamba_allocator
    view = C.TreeView(cache)
    plan, why = C.plan_for(C.used_ids(mal.slot_used), view.anchors(), 7)
    assert why == "" and plan.moves == [] and [a.slot for a in plan.evicts] == [20]
    res = C.execute(cache, mal, view, plan)
    assert (res.moved, res.evicted) == (0, 1)
    cd = nodes[20].component_data[MC]
    assert cd.value is None and cd.host_value is not None, "device tombstone, the L2 copy stays"
    assert nodes[20].component_data[ComponentType.FULL].value is not None, "its KV stays"
    assert 20 not in C.used_ids(mal.slot_used)


def test_unbacked_states_take_the_free_slots_first_backed_ones_are_evicted():
    cache, pool, nodes = _tree(list(range(1, 7)) + [20, 30], backed=(30,))
    view = C.TreeView(cache)
    plan, why = C.plan_for(C.used_ids(pool.mamba_allocator.slot_used), view.anchors(), 7)
    assert [(a.slot, d) for a, d in plan.moves] == [(20, 7)] and [a.slot for a in plan.evicts] == [30]


def test_a_pinned_or_in_flight_state_does_not_move():
    cache, pool, nodes = _tree([20])
    nodes[20].component_data[MC].lock_ref = 1
    plan, why = C.plan_for(C.used_ids(pool.mamba_allocator.slot_used), C.TreeView(cache).anchors(), 7)
    assert plan is None and "pinned" in why
    nodes[20].component_data[MC].lock_ref = 0
    cache.ongoing_write_through[nodes[20].id] = object()     # the D2H copy reads the source
    a = C.TreeView(cache).anchors()[0]
    assert a.pinned and not a.backed
    cache.ongoing_write_through.clear()
    cache.ongoing_load_back[nodes[20].id] = object()         # the H2D copy writes it
    assert C.TreeView(cache).anchors()[0].pinned


def test_a_slot_held_outside_the_tree_is_named():
    cache, pool, nodes = _tree([20])
    mal = pool.mamba_allocator
    extra = mal.alloc(SIZE - 1)                              # everything else ...
    mal.free(extra[extra != 22])                             # ... but slot 22 goes back
    plan, why = C.plan_for(C.used_ids(mal.slot_used), C.TreeView(cache).anchors(), 7)
    assert plan is None and "22 held outside the tree" in why


def test_the_claim_takes_only_free_slots():
    cache, pool, nodes = _tree([20])
    mal = pool.mamba_allocator
    assert not mal.claim_free_slots(torch.tensor([20]))       # in use
    assert not mal.claim_free_slots(torch.tensor([3, 3]))     # twice
    n0 = mal.available_size()
    assert mal.claim_free_slots(torch.tensor([3, 4]))
    assert mal.available_size() == n0 - 2 and bool(mal.slot_used[3]) and bool(mal.slot_used[4])


# ---------------------------------------------------------------- the tick: idle, riegel, group MIN

class _Ctl:
    def __init__(self):
        self.calls, self.rows_on, self.mamba_keep = [], 20, 7

    def reseat_live(self, n, stage):
        self.calls.append(n)


def _sched(cache, pool, n=4, gm=None):
    s = types.SimpleNamespace(running_batch=types.SimpleNamespace(reqs=[], batch_is_full=True),
                              waiting_queue=[], chunked_req=None, weg2_dormant=False, tree_cache=cache)
    s._weg2_d_seat_phase = dsv.PhaseState(epoch="e1", n=n, cap=6, has_n=True, done=True, apply_ms=40.0)
    s.tp_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(req_to_token_pool=pool))
    calls = []

    def group_min(flags):
        calls.append(list(flags))
        return gm(flags, len(calls)) if gm else [1 if f else 0 for f in flags]
    s._weg2_group_min_flags = group_min
    return s, calls


def _idle(s, ctl, ticks, clock):
    got = None
    ctxs = [envs.SGLANG_WEG2_D_SEAT_REWAKE.override(True),
            mock.patch.object(dsv, "armed", lambda env=None: True),
            mock.patch.object(dsv, "controller", lambda sched: ctl),
            mock.patch.object(dsv, "stage_form", lambda env=None: None),
            mock.patch.object(dsv, "_unmerged_extend", lambda sched, running: []),
            mock.patch.object(R.time, "monotonic", lambda: clock[0])]
    for c in ctxs:
        c.__enter__()
    try:
        for _ in range(ticks):
            clock[0] += 0.2
            got = R.tick(s) or got
    finally:
        for c in reversed(ctxs):
            c.__exit__(None, None, None)
    return got


def test_y4x_an_empty_d_compacts_then_shrinks_to_one(caplog):
    """y4x: D empty, n=4 (limit 25), the tree holds states in 10, 20, 24 ->
    COMPACT moved=3, then SHRINK n=4->1 (limit 7); the ranks agreed on the
    target in ONE group MIN and on the move's success in a second."""
    caplog.set_level(logging.INFO)
    cache, pool, nodes = _tree([10, 20, 24])
    before = {s: _bytes_at(pool, s) for s in nodes}
    assert pool.mamba_allocator.set_phase_limit(25, seats=4)
    ctl = _Ctl()
    s, calls = _sched(cache, pool)
    got = _idle(s, ctl, R.IDLE_ASK_ROUNDS, [1000.0])
    assert got == "shrink" and ctl.calls == [1] and s._weg2_d_seat_phase.n == 1
    assert pool.mamba_allocator.phase_limit == 7
    for sl, node in nodes.items():
        new = int(node.component_data[MC].value[0])
        assert new <= 7 and all(torch.equal(a, b) for a, b in zip(before[sl], _bytes_at(pool, new)))
    assert len(calls) == 2 and len(calls[0]) == 1 + 2 * 3 and calls[1] == [True]
    line = [m for m in caplog.messages if "D-SEAT-REWAKE COMPACT moved=3" in m][-1]
    assert "bytes=%d" % (3 * C.slot_bytes(pool.mamba_pool)) in line and "ms=" in line
    shrink = [i for i, m in enumerate(caplog.messages) if "D-SEAT-REWAKE SHRINK n=4->1" in m]
    compact = [i for i, m in enumerate(caplog.messages) if "D-SEAT-REWAKE COMPACT" in m]
    assert shrink and compact and compact[-1] < shrink[-1], "COMPACT first, then SHRINK"
    assert getattr(s, R.ATTR).counters["compact_moved"] == 3


def test_another_rank_that_reaches_less_sets_the_target():
    """The group MIN: another rank reaches n=2 at best (its flag for n=1 is 0)
    -> every rank compacts to the limit of n=2 (13) and shrinks to 2."""
    cache, pool, nodes = _tree([10, 20, 24])
    assert pool.mamba_allocator.set_phase_limit(25, seats=4)
    ctl = _Ctl()

    def gm(flags, i):
        out = [1 if f else 0 for f in flags]
        if i == 1:
            out[1] = 0                                   # reach(n=1) on the other rank: no
        return out
    s, calls = _sched(cache, pool, gm=gm)
    assert _idle(s, ctl, R.IDLE_ASK_ROUNDS, [1000.0]) == "shrink"
    assert ctl.calls == [2] and pool.mamba_allocator.phase_limit == 13
    assert max(int(nd.component_data[MC].value[0]) for nd in nodes.values()) <= 13


def test_a_rank_whose_move_failed_keeps_n_on_every_rank(caplog):
    caplog.set_level(logging.INFO)
    cache, pool, nodes = _tree([10, 20, 24])
    assert pool.mamba_allocator.set_phase_limit(25, seats=4)
    ctl = _Ctl()
    s, calls = _sched(cache, pool, gm=lambda flags, i: [0] if i == 2 else [1 if f else 0 for f in flags])
    assert _idle(s, ctl, R.IDLE_ASK_ROUNDS, [1000.0]) is None
    assert ctl.calls == [] and s._weg2_d_seat_phase.n == 4 and pool.mamba_allocator.phase_limit == 25
    assert any("SHRINK HELD" in m and "compact_refused" in m for m in caplog.messages)


def test_the_riegel_hold_the_compaction_too():
    for attr, value in (("weg2_d_parked", [object()]), ("weg2_dormant_hold", {"x": 1}),
                        ("weg2_post_wake_settle", [object()]), ("chunked_req", types.SimpleNamespace(rid="c"))):
        cache, pool, nodes = _tree([10, 20])
        assert pool.mamba_allocator.set_phase_limit(25, seats=4)
        ctl = _Ctl()
        s, calls = _sched(cache, pool)
        setattr(s, attr, value)
        _idle(s, ctl, R.IDLE_ASK_ROUNDS * 2, [1000.0])
        assert calls == [] and ctl.calls == [], attr
        assert sorted(int(nd.component_data[MC].value[0]) for nd in nodes.values()) == [10, 20], attr


def test_a_busy_d_does_not_compact():
    """Only in the idle loop: with a running request the round trigger keeps
    the partial shrink of 50bed49e3a (nothing moves)."""
    cache, pool, nodes = _tree([20])
    assert pool.mamba_allocator.set_phase_limit(25, seats=4)
    ctl = _Ctl()
    s, calls = _sched(cache, pool)
    s.running_batch.reqs = [types.SimpleNamespace(rid="r0")]
    _idle(s, ctl, R.SHRINK_ASK_ROUNDS, [1000.0])
    assert ctl.calls == []
    assert int(nodes[20].component_data[MC].value[0]) == 20


def test_an_unreachable_target_is_held_by_name_and_not_walked_again(caplog):
    caplog.set_level(logging.INFO)
    cache, pool, nodes = _tree(list(range(1, 8)) + list(range(8, 14)) + list(range(14, 20)) + [22])
    assert pool.mamba_allocator.set_phase_limit(25, seats=4)
    ctl = _Ctl()
    s, calls = _sched(cache, pool)
    walks = []
    real = C.TreeView.anchors

    def counted(self):
        walks.append(1)
        return real(self)
    with mock.patch.object(C.TreeView, "anchors", counted):
        _idle(s, ctl, R.IDLE_ASK_ROUNDS * 3, [1000.0])
    held = [m for m in caplog.messages if "SHRINK HELD" in m]
    assert ctl.calls == [] and len(held) == 1
    assert "un-backed" in held[0] and len(walks) == 1
    # KEIL: the rank still votes -- no for every n' below 4
    assert len(calls) == 3 and all(not any(c[1:4]) for c in calls)


# ---------------------------------------------------------------- three ranks, one collective

def test_three_ranks_one_compacts_all_agree_no_rank_skips_the_second_min(caplog):
    """Rank 1's tree holds states in 10/20/24, ranks 0/2 hold none. All three
    vote n=1 reachable (rank 1 with a move) -> rank 1 moves, ranks 0/2 still
    enter the second MIN (the "a move is needed" flag is agreed), all shrink
    to 1. Real threads, a barrier all-reduce with a timeout."""
    import importlib.util
    import threading

    spec = importlib.util.spec_from_file_location(
        "_t_keil_c", os.path.join(os.path.dirname(__file__), "test_weg2_d_seat_rewake_keil_0930.py"))
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    caplog.set_level(logging.INFO)
    trees = [_tree([]), _tree([10, 20, 24]), _tree([])]
    for _c, pool, _n in trees:
        assert pool.mamba_allocator.set_phase_limit(25, seats=4)
    group = K._Group()
    scheds = []
    for r, (cache, pool, _n) in enumerate(trees):
        s, _calls = _sched(cache, pool)
        s._weg2_group_min_flags = group.member(r)
        s._ctl = _Ctl()
        scheds.append(s)
    clock = threading.local()
    errors = {}

    def body(rank):
        clock.t = 1000.0
        try:
            for _ in range(R.IDLE_ASK_ROUNDS):
                clock.t += 0.2
                R.tick(scheds[rank])
        except BaseException as exc:  # noqa: BLE001
            errors[rank] = exc

    ctxs = [envs.SGLANG_WEG2_D_SEAT_REWAKE.override(True),
            mock.patch.object(dsv, "armed", lambda env=None: True),
            mock.patch.object(dsv, "controller", lambda sched: sched._ctl),
            mock.patch.object(dsv, "stage_form", lambda env=None: None),
            mock.patch.object(dsv, "_unmerged_extend", lambda sched, running: []),
            mock.patch.object(R.time, "monotonic", lambda: clock.t)]
    for c in ctxs:
        c.__enter__()
    try:
        ths = [threading.Thread(target=body, args=(r,), daemon=True) for r in range(3)]
        for t in ths:
            t.start()
        for t in ths:
            t.join(K.TIMEOUT_S * 4)
        assert not [r for r, t in enumerate(ths) if t.is_alive()], "a rank hangs"
    finally:
        for c in reversed(ctxs):
            c.__exit__(None, None, None)
    assert not errors, errors
    assert group.rounds == 2, "the vote and the move's success: two collectives, all ranks"
    assert [s._weg2_d_seat_phase.n for s in scheds] == [1, 1, 1]
    assert max(int(nd.component_data[MC].value[0]) for nd in trees[1][2].values()) <= 7
    assert sum("D-SEAT-REWAKE COMPACT moved=3" in m for m in caplog.messages) == 1
