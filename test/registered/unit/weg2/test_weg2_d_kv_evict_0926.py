"""--d-kv-evict-for-placement (27B row 24c, 26.09.): cold L2-backed radix KV
makes room for the bandwidth placement.

Pinned without a GPU (design: /spinning/gpu-arb/docs/DYN_D_RESHARD.md sec. 14):
  * POLICY -- the trigger sits exactly below the placement's own regime switch
    (placement_weights says 'bandwidth' below U_switch, 'capacity' above);
    need/excess from the agreed fill; nothing below the trigger;
  * VICTIMS -- only agreed-backed candidates, the class over its share first,
    the colder one on a tie, caps honoured, un-backed ones only published;
  * GROUP-UNIFORM -- three replica trees (real UnifiedRadixCache) with an
    ASYMMETRIC host state run one pass through a real MIN rendezvous: every
    rank evicts the SAME list; a node one rank has no L2 copy of is evicted
    nowhere; a tree divergence is a RankDivergence on every rank;
  * NO ACTIVE READER -- a locked (running) node is never evicted, however cold;
    a node touched inside the idle window is not a candidate;
  * COPY-FREE DEMOTION -- the victim stays in the tree with its host rows, its
    device rows are back in the free list, the placement re-interleaved;
  * OFF IDENTICAL -- no env: the hook disarms itself on the first lookup, the
    tree and the allocator are untouched; the launcher default ships no env;
    'on' without --d-token-placement bandwidth is refused.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from unittest import mock

import torch

from sglang.srt.weg2 import d_kv_evict as E
from sglang.srt.weg2 import d_reshard as D
from sglang.srt.weg2 import d_token_placement as P
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PREFIX = (0, 21, 43, 64)          # installed vector (21, 22, 21), S = 64
PSPEC = P.PlacementSpec("bandwidth", P.RC9_EFF_BW_GBS)
SIZE = 64 * 20                    # 1280 slots: class totals 419/440/420


def _static(prefix=PREFIX):
    return lambda: prefix


# ---------------------------------------------------------------------------
# pure policy
# ---------------------------------------------------------------------------


class TestPolicy(unittest.TestCase):
    def test_trigger_matches_placement_regime_switch(self):
        totals = P.slot_class_totals(SIZE, 64, PREFIX)
        sw = E.switch_level(totals, PSPEC.shares, PSPEC.fill_switch, PSPEC.lookahead)
        T = sum(totals)
        for U, want in ((int(sw) - 2, "bandwidth"), (int(sw) + 2, "capacity")):
            used = [round(U * s) for s in PSPEC.normalized_shares()]
            used[0] += U - sum(used)
            free = [t - u for t, u in zip(totals, used)]
            _, regime = P.placement_weights(totals, free, PSPEC)
            self.assertEqual(regime, want, (U, sw))
        h = 0.02
        self.assertFalse(E.triggered_by_total(totals, int(sw - h * T) - 1, PSPEC.shares, 0.85, 0.02, h))
        self.assertTrue(E.triggered_by_total(totals, int(sw - h * T) + 1, PSPEC.shares, 0.85, 0.02, h))

    def test_pressure_need_and_excess(self):
        totals = [4000, 4000, 4000]
        shares = PSPEC.shares
        low = E.pressure(totals, [1000, 1000, 1000], shares, 0.85, 0.02, 0.02)
        self.assertFalse(low.triggered)
        self.assertEqual(low.need, 0)
        hi = E.pressure(totals, [3300, 2800, 2800], shares, 0.85, 0.02, 0.02)
        self.assertTrue(hi.triggered)
        self.assertEqual(hi.need, int(hi.used - hi.goal + 0.999999))
        self.assertGreaterEqual(sum(hi.excess), hi.need)
        self.assertGreater(hi.excess[0], 0)    # the 5090 class holds more than its share of the goal

    def test_select_prefers_excess_class_then_colder(self):
        p = E.Pressure(total=1000, used=900, switch=800, goal=760, triggered=True, need=100, excess=(100, 0, 0))
        spec = E.EvictSpec()
        lengths = [100, 100, 100]
        cc = [[0, 100, 0], [100, 0, 0], [100, 0, 0]]     # 0 useless, 1 and 2 useful (1 colder)
        ev, pub = E.select_victims(lengths, cc, [1, 1, 1], [0, 0, 0], p, spec)
        self.assertEqual(ev, [1])
        self.assertEqual(pub, [])

    def test_only_agreed_backed_and_publish_fallback(self):
        p = E.Pressure(1000, 900, 800, 760, True, 150, (80, 40, 30))
        spec = E.EvictSpec(publish_max=2)
        lengths = [50, 50, 50, 50]
        cc = [[25, 15, 10]] * 4
        ev, pub = E.select_victims(lengths, cc, [0, 1, 0, 0], [1, 0, 1, 1], p, spec)
        self.assertEqual(ev, [1])
        self.assertEqual(pub, [0, 2])                     # coldest useful un-backed, capped at 2

    def test_caps_and_untriggered(self):
        p = E.Pressure(10_000, 9000, 8000, 7600, True, 5000, (5000, 5000, 5000))
        spec = E.EvictSpec(max_evict_nodes=3, max_evict_tokens=250)
        lengths = [100] * 10
        cc = [[40, 30, 30]] * 10
        ev, _ = E.select_victims(lengths, cc, [1] * 10, [0] * 10, p, spec)
        self.assertEqual(ev, [0, 1])                      # token cap 250 -> two nodes
        off = E.Pressure(10_000, 1000, 8000, 7600, False, 0, (0, 0, 0))
        self.assertEqual(E.select_victims(lengths, cc, [1] * 10, [1] * 10, off, spec), ([], []))

    def test_spec_roundtrip_and_validation(self):
        s = E.EvictSpec(min_idle_s=5.0, publish_max=0)
        self.assertEqual(E.EvictSpec.from_json(s.to_json()), s)
        with self.assertRaises(E.EvictError):
            E.EvictSpec(every=0).validate()
        self.assertIsNone(E.spec_from_env({}))
        self.assertEqual(E.spec_from_env({E.ENV: s.to_json()}), s)

    def test_cold_clock_floor(self):
        c = E.ColdClock(20_000)
        self.assertIsNone(c.floor())
        c.record(0, 10.0)
        self.assertIsNone(c.floor())
        c.record(10_000, 20.0)
        self.assertIsNone(c.floor())
        c.record(25_000, 30.0)
        self.assertEqual(c.floor(), 10.0)
        c.record(31_000, 40.0)
        self.assertEqual(c.floor(), 20.0)
        self.assertLessEqual(len(c.ring), 3)

    def test_unpack_detects_divergence(self):
        a = E.pack(1, 5, 111, 2, 900, [300, 300, 300], [1, 1], [0, 0], 4)
        b = E.pack(1, 7, 222, 2, 910, [305, 300, 305], [1, 0], [0, 1], 4)
        m = torch.minimum(a, b)
        with self.assertRaises(E.RankDivergence):
            E.unpack(m, 3, 4)
        b2 = E.pack(1, 7, 111, 2, 910, [305, 300, 305], [1, 0], [0, 1], 4)
        ag = E.unpack(torch.minimum(a, b2), 3, 4)
        self.assertEqual((ag.t_ms, ag.used_total_max, ag.used_per_class_max), (5, 910, (305, 300, 305)))
        self.assertEqual((ag.evict_ok, ag.publish_ok), ((1, 0), (0, 0)))


# ---------------------------------------------------------------------------
# real UnifiedRadixCache replicas
# ---------------------------------------------------------------------------


def _build_cache():
    from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
    from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    rtp = ReqToTokenPool(size=8, max_context_len=SIZE, device="cpu", enable_memory_saver=False)
    kv = MHATokenToKVPool(size=SIZE, page_size=1, dtype=torch.bfloat16, head_num=1, head_dim=4,
                          layer_num=1, device="cpu", enable_memory_saver=False)
    alloc = TokenToKVPoolAllocator(size=SIZE, dtype=torch.bfloat16, device="cpu", kvcache=kv, need_sort=False)
    alloc.set_owner_placement(P.OwnerPlacement(PSPEC, prefix_fn=_static()))
    cache = UnifiedRadixCache(params=CacheInitParams(
        req_to_token_pool=rtp, token_to_kv_pool_allocator=alloc, page_size=1, disable=False,
        tree_components=(ComponentType.FULL,)))
    return cache, alloc


def _slots_in_class(cls: int, n: int, start: int) -> torch.Tensor:
    ids = [i for i in range(1, SIZE + 1) if P.class_of(torch.tensor([i]), 64, PREFIX).item() == cls]
    return torch.tensor(ids[start:start + n], dtype=torch.int64)


class Replica:
    """One D rank: a tree, its allocator, and prefixes inserted from fixed slots."""

    def __init__(self):
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey

        self.cache, self.alloc = _build_cache()
        self.InsertParams, self.RadixKey = InsertParams, RadixKey
        self.leaves = {}

    def insert(self, name: str, first_token: int, slots: torch.Tensor, host: bool = True):
        from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

        # take the slots out of the free list, as a real extend would
        fp = self.alloc.free_pages
        keep = ~torch.isin(fp, slots)
        self.alloc.free_pages = fp[keep]
        toks = list(range(first_token, first_token + len(slots)))
        self.cache.insert(self.InsertParams(key=self.RadixKey(toks), value=slots.clone()))
        node = self.find(toks)
        if host:
            node.component_data[BASE_COMPONENT_TYPE].host_value = torch.arange(len(slots)) + 10_000
        self.leaves[name] = node
        return node

    def find(self, toks):
        node = self.cache.root_node
        for child in node.children.values():
            if child.key is not None and list(child.key.token_ids[:1]) == toks[:1]:
                return child
        raise KeyError(toks[:1])

    def counter(self):
        best = 0.0
        stack = [self.cache.root_node]
        while stack:
            n = stack.pop()
            best = max(best, float(n.last_access_time))
            stack.extend(n.children.values())
        return best + 1.0


def _populate(rep: Replica, hot_touch: bool = False):
    """Eight 100-token prefixes, 800 of 1280 slots used (> trigger ~766).
    'a0'..'a3' sit entirely in class 0 (the 5090: 400 used against its share
    of the goal ~323, the only class in excess), the rest in classes 1/2."""
    starts = {0: 0, 1: 0, 2: 0}
    plan = [("a0", 0), ("b1", 1), ("a1", 0), ("c2", 2), ("a2", 0), ("d1", 1), ("a3", 0), ("e2", 2)]
    for i, (name, cls) in enumerate(plan):
        slots = _slots_in_class(cls, 100, starts[cls])
        starts[cls] += 100
        rep.insert(name, 1000 * (i + 1), slots)


class Rendezvous:
    """A real MIN all_reduce across threads (one per 'rank')."""

    def __init__(self, n):
        self.n = n
        self.barrier = threading.Barrier(n, timeout=20)
        self.buf = [None] * n

    def reducer(self, rank):
        def _reduce(t):
            self.buf[rank] = t.clone()
            self.barrier.wait()
            m = torch.stack(self.buf).min(0).values
            self.barrier.wait()
            t.copy_(m)
        return _reduce


def _run_group(evictors):
    out, err = [None] * len(evictors), [None] * len(evictors)

    def go(i):
        try:
            out[i] = evictors[i].run_pass()
        except BaseException as e:  # noqa: BLE001
            err[i] = e
    ts = [threading.Thread(target=go, args=(i,), daemon=True) for i in range(len(evictors))]
    for t in ts:
        t.start()
    for t in ts:
        t.join(30)
    return out, err


class Clock:
    def __init__(self):
        self.t = 0

    def __call__(self):
        return self.t


class TestReplicas(unittest.TestCase):
    def _group(self, n=3, spec=None, mutate=None):
        spec = spec or E.EvictSpec(min_idle_s=20.0, publish_max=0)
        reps = [Replica() for _ in range(n)]
        for r in reps:
            _populate(r)
        if mutate:
            mutate(reps)
        rv = Rendezvous(n)
        clock = Clock()
        evs = [E.DKvEvictor(r.cache, r.alloc, spec, rv.reducer(i), rank=i, now_ms=clock,
                            counter=r.counter) for i, r in enumerate(reps)]
        return reps, evs, clock

    def _two_passes(self, evs, clock):
        """Pass 1 agrees the fill and starts the clock, pass 2 is 25 s later
        (the floor is read from the passes BEFORE the current one, so the
        first cold floor exists at pass 3 = pass-1 counter)."""
        for t in (0, 25_000):
            clock.t = t
            out, err = _run_group(evs)
            self.assertEqual(err, [None] * len(evs))
        clock.t = 30_000
        return _run_group(evs)

    def test_group_uniform_under_asymmetric_host(self):
        from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

        def asym(reps):
            # rank 1 has NO L2 copy of 'a0' (its write failed / was released)
            reps[1].leaves["a0"].component_data[BASE_COMPONENT_TYPE].host_value = None

        reps, evs, clock = self._group(mutate=asym)
        out, err = self._two_passes(evs, clock)
        self.assertEqual(err, [None, None, None])
        victims = [o["victims"] for o in out]
        self.assertEqual(victims[0], victims[1])
        self.assertEqual(victims[1], victims[2])
        self.assertEqual(len(victims[0]), 1)
        for r in reps:
            self.assertIsNotNone(r.leaves["a0"].component_data[BASE_COMPONENT_TYPE].value)   # nowhere evicted
            self.assertIsNone(r.leaves["a1"].component_data[BASE_COMPONENT_TYPE].value)      # the class-0 twin
            self.assertIsNotNone(r.leaves["a1"].component_data[BASE_COMPONENT_TYPE].host_value)
            self.assertIn(r.leaves["a1"], r.cache.evictable_host_leaves)                     # stays in the tree (L2)
        self.assertEqual(out[0]["freed_per_class"], [100, 0, 0])
        for r in reps[1:]:
            self.assertTrue(torch.equal(torch.sort(r.alloc.free_pages).values,
                                        torch.sort(reps[0].alloc.free_pages).values))

    def test_rank_local_lock_is_no_divergence_and_blocks_eviction(self):
        """A write pin still held on ONE rank (its ack is rank-local) must not
        split the candidate digest -- and the pinned node is evicted nowhere."""
        from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

        def pin_on_rank2(reps):
            reps[2].cache.inc_lock_ref(reps[2].leaves["a0"])

        reps, evs, clock = self._group(mutate=pin_on_rank2)
        out, err = self._two_passes(evs, clock)
        self.assertEqual(err, [None, None, None])
        self.assertEqual(out[0]["victims"], out[2]["victims"])
        for r in reps:
            self.assertIsNotNone(r.leaves["a0"].component_data[BASE_COMPONENT_TYPE].value)
            self.assertIsNone(r.leaves["a1"].component_data[BASE_COMPONENT_TYPE].value)

    def test_tree_divergence_is_a_crash_on_every_rank(self):
        def diverge(reps):
            reps[2].insert("x", 99_000, _slots_in_class(1, 20, 300))

        reps, evs, clock = self._group(mutate=diverge)
        out, err = self._two_passes(evs, clock)
        self.assertTrue(all(isinstance(e, E.RankDivergence) for e in err), err)
        from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE
        for r in reps:
            self.assertTrue(all(n.component_data[BASE_COMPONENT_TYPE].value is not None
                                for n in r.leaves.values()))          # nothing evicted before the verdict

    def test_active_reader_and_hot_node_never_evicted(self):
        from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

        reps, evs, clock = self._group(n=1)
        rep = reps[0]
        rep.cache.inc_lock_ref(rep.leaves["a0"])       # a running request reads a0
        for t in (0, 25_000):
            clock.t = t
            _run_group(evs)
        rep.cache._touch_node(rep.leaves["a1"])        # a1 hit inside the idle window
        clock.t = 30_000
        out, err = _run_group(evs)
        self.assertEqual(err, [None])
        self.assertIsNotNone(rep.leaves["a0"].component_data[BASE_COMPONENT_TYPE].value)
        self.assertIsNotNone(rep.leaves["a1"].component_data[BASE_COMPONENT_TYPE].value)
        self.assertEqual(out[0]["evicted"], 1)
        self.assertIsNone(rep.leaves["a2"].component_data[BASE_COMPONENT_TYPE].value)   # next cold class-0 node
        self.assertEqual(out[0]["freed_per_class"], [100, 0, 0])

    def test_unbacked_leaf_is_published_not_evicted(self):
        from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE

        def unback_all(reps):
            for r in reps:
                for n in r.leaves.values():
                    n.component_data[BASE_COMPONENT_TYPE].host_value = None
                r.cache.cache_controller = object()     # publishable needs a controller
                r.published = []
                r.cache.publish_unbacked_sweep = (
                    lambda max_issue, first, chain_only, _r=r: _r.published.append(len(first)) or {"issued": len(first)})

        reps, evs, clock = self._group(n=1, spec=E.EvictSpec(min_idle_s=20.0, publish_max=3), mutate=unback_all)
        out, err = self._two_passes(evs, clock)
        self.assertEqual(err, [None])
        self.assertEqual(out[0]["evicted"], 0)
        self.assertEqual(out[0]["published"], 3)
        self.assertTrue(all(n.component_data[BASE_COMPONENT_TYPE].value is not None
                            for n in reps[0].leaves.values()))

    def test_below_trigger_nothing_happens_and_placement_holds(self):
        reps, evs, clock = self._group(n=1)
        rep = reps[0]
        ids = torch.cat([n.component_data[0].value for n in list(rep.leaves.values())[:3]])
        # release three prefixes from the tree bookkeeping: fill 500 < trigger
        rep.alloc.free_pages = torch.cat([rep.alloc.free_pages, ids])
        out, err = self._two_passes(evs, clock)
        self.assertEqual(err, [None])
        self.assertEqual(out[0]["flag"], 0)
        self.assertEqual(out[0]["evicted"], 0)

    def test_placement_reapplied_after_eviction(self):
        reps, evs, clock = self._group(n=1)
        out, err = self._two_passes(evs, clock)
        self.assertEqual(err, [None])
        self.assertGreaterEqual(out[0]["evicted"], 1)
        self.assertEqual(reps[0].alloc._owner_placement.last_regime, "bandwidth")


# ---------------------------------------------------------------------------
# off = unchanged; launcher
# ---------------------------------------------------------------------------


class _Sched:
    pass


class TestOff(unittest.TestCase):
    def test_hook_disarms_without_env(self):
        s = _Sched()
        s.tree_cache = mock.MagicMock()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(E.ENV, None)
            E.scheduler_step(s)
            E.scheduler_step(s)
        self.assertIsNone(s._weg2_d_kv_evictor)
        self.assertEqual(s.tree_cache.mock_calls, [])

    def test_env_without_placement_refused(self):
        s = _Sched()
        rep = Replica()
        rep.alloc._owner_placement = None
        s.tree_cache = rep.cache
        with mock.patch.dict(os.environ, {E.ENV: E.EvictSpec().to_json()}):
            with self.assertRaises(RuntimeError):
                E.scheduler_step(s)


class TestLauncher(unittest.TestCase):
    def setUp(self):
        from sglang.srt.weg2 import launcher as L
        self.L = L

    def _model(self, cfg):
        d = tempfile.mkdtemp()
        with open(os.path.join(d, "config.json"), "w") as f:
            json.dump(cfg, f)
        return d

    def test_default_off_no_env(self):
        ns = self.L.build_parser().parse_args(["--tree", "/t", "--tag", "t"])
        self.assertEqual(ns.d_kv_evict_for_placement, "off")
        self.L.apply_d_token_placement(ns)
        self.L.apply_d_kv_evict(ns)
        self.assertEqual(self.L.d_kv_evict_env(), {})

    def test_on_needs_bandwidth_and_ships_env(self):
        m = self._model({"text_config": dict(D._QWEN38_27B)})
        ns = self.L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--model", m,
                                               "--d-kv-evict-for-placement", "on"])
        self.L.apply_d_token_placement(ns)
        with self.assertRaises(SystemExit) as cm:
            self.L.apply_d_kv_evict(ns)
        self.assertIn("REFUSED without --d-token-placement bandwidth", str(cm.exception.code))
        self.assertEqual(self.L.d_kv_evict_env(), {})
        ns = self.L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--model", m,
                                               "--d-token-placement", "bandwidth",
                                               "--d-kv-evict-for-placement", "on",
                                               "--d-kv-evict-min-idle-s", "15"])
        self.L.apply_d_token_placement(ns)
        self.L.apply_d_kv_evict(ns)
        spec = E.EvictSpec.from_json(self.L.d_kv_evict_env()[E.ENV])
        self.assertEqual(spec.min_idle_s, 15.0)
        self.assertEqual(spec.publish_max, 8)


if __name__ == "__main__":
    unittest.main()
