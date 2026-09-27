# SPDX-License-Identifier: Apache-2.0
"""ED/EF (27B rc12o b1, PP0 13:56:43Z, weg2-8-62): 'Available full tokens: 149167 (46 + evictable
149121)' and 'EVICTION UNDER-DELIVERED: asked for 512 tokens, the pool received 0', with no #1421
BACKUP-REFUSED on PP0 in the whole boot -- no leaf was even tried.

The reported FULL-evictable count includes nodes whose FULL lock is 0 but whose MAMBA lock is not
(a mamba lock pins the node alone), and their FULL-unlocked ancestors; ``_is_device_leaf`` refuses
all of them, so the peel can never pay them. ED: admission reads the deliverable count. EF: an
under-delivering peel re-derives leaf membership once, retries, and names the census.
"""

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache import evict_frontier_census as EF  # noqa: E402

FULL, MAMBA = 0, 1


class _CD:
    def __init__(self, value, lock_ref=0):
        self.value = value
        self.lock_ref = lock_ref


class _Node:
    _n = 0

    def __init__(self, parent, full_len, full_lock=0, mamba_lock=0):
        _Node._n += 1
        self.id = _Node._n
        self.parent = parent
        self.children = {}
        self.evicted = False
        self.component_data = [_CD(list(range(full_len)), full_lock), _CD([1], mamba_lock)]
        if parent is not None:
            parent.children[self.id] = self


def _cache():
    root = _Node(None, 0)
    c = types.SimpleNamespace(root_node=root, component_evictable_size_={FULL: 0},
                              evictable_device_leaves=set())
    return c, root


def _env(**kv):
    return mock.patch.dict(os.environ, {k: str(v) for k, v in kv.items()})


class Deliverable(unittest.TestCase):
    def test_b1_shape_mamba_locked_tail_blocks_its_chain(self):
        c, root = _cache()
        a = _Node(root, 69096)                 # weg2-8-61's host-backed prefix, unlocked
        b = _Node(a, 80000)                    # further unlocked chain
        tail = _Node(b, 25, mamba_lock=1)      # the anchor node: FULL 0, MAMBA 1
        free_leaf = _Node(root, 5000)          # an ordinary leaf elsewhere
        c.component_evictable_size_[FULL] = 69096 + 80000 + 25 + 5000
        EF.note_aux_lock(c, tail, True)
        with _env(**{EF.ENV_DELIVERABLE: 1}):
            self.assertEqual(EF.blocked_tokens(c, FULL), 69096 + 80000 + 25)
            self.assertEqual(EF.deliverable_evictable(c, FULL), 5000)
        with _env(**{EF.ENV_DELIVERABLE: 0}):
            self.assertEqual(EF.deliverable_evictable(c, FULL), 154121, "switch off: the reported count")
        EF.note_aux_lock(c, tail, False)
        with _env(**{EF.ENV_DELIVERABLE: 1}):
            self.assertEqual(EF.deliverable_evictable(c, FULL), 154121, "lock gone: nothing blocked")

    def test_walk_stops_at_a_full_locked_ancestor_and_counts_once(self):
        c, root = _cache()
        locked = _Node(root, 1000, full_lock=1)
        a = _Node(locked, 300)
        t1 = _Node(a, 10, mamba_lock=1)
        t2 = _Node(a, 20, mamba_lock=1)
        c.component_evictable_size_[FULL] = 330
        EF.note_aux_lock(c, t1, True)
        EF.note_aux_lock(c, t2, True)
        with _env(**{EF.ENV_DELIVERABLE: 1}):
            self.assertEqual(EF.blocked_tokens(c, FULL), 10 + 20 + 300)
            self.assertEqual(EF.deliverable_evictable(c, FULL), 0)

    def test_a_full_locked_aux_node_blocks_nothing(self):
        c, root = _cache()
        t = _Node(root, 50, full_lock=1, mamba_lock=1)
        EF.note_aux_lock(c, t, True)
        self.assertEqual(EF.blocked_tokens(c, FULL), 0)

    def test_mamba_component_tracks_the_lock_transitions(self):
        from sglang.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent

        c, root = _cache()
        c.component_evictable_size_ = {FULL: 0, MAMBA: 1}
        c.component_protected_size_ = {FULL: 0, MAMBA: 0}
        c._pin_trace_every = 0
        n = _Node(root, 7)
        comp = types.SimpleNamespace(cache=c, component_type=MAMBA)
        res = types.SimpleNamespace(skip_lock_node_ids={})
        MambaComponent.acquire_component_lock(comp, n, res)
        self.assertIn(id(n), getattr(c, EF.AUX_LOCKED_ATTR))
        MambaComponent.acquire_component_lock(comp, n, res)
        MambaComponent.release_component_lock(comp, n, None)
        self.assertIn(id(n), getattr(c, EF.AUX_LOCKED_ATTR), "still locked once")
        MambaComponent.release_component_lock(comp, n, None)
        self.assertNotIn(id(n), getattr(c, EF.AUX_LOCKED_ATTR))

    def test_fundable_extend_reads_the_deliverable_count(self):
        from sglang.srt.mem_cache import common

        class _Tree:  # the deliverable count is read from the CLASS (not a duck-typed getattr)
            token_to_kv_pool_allocator = types.SimpleNamespace(available_size=lambda: 46)

            def evictable_size(self):
                return 149121

            def deliverable_evictable_size(self):
                return 0

        class _Plain(_Tree):
            deliverable_evictable_size = None

        self.assertEqual(common.fundable_extend_tokens(_Tree()), 46, "b1: 512 is not fundable -> parks")
        self.assertEqual(common.fundable_extend_tokens(_Plain()), 149167)

    def test_a_duck_typed_stand_in_falls_back_to_the_reported_count(self):
        from unittest import mock

        from sglang.srt.mem_cache import common

        m = mock.MagicMock()  # answers every getattr: must not be read as a count
        self.assertEqual(common.deliverable_evictable_or(m, lambda: 7), 7)

    def test_the_adder_budget_reads_it_on_hybrid_ssm(self):
        from sglang.srt.managers import schedule_policy as sp

        src = open(sp.__file__).read()
        i = src.index("def rem_total_tokens(self)")
        blk = src[i:i + 1500]
        self.assertIn("deliverable_evictable_or(self.tree_cache, self.tree_cache.full_evictable_size)", blk)


class FrontierRepair(unittest.TestCase):
    def _cache(self, stale=True):
        from sglang.srt.mem_cache.unified_cache_components.full_component import FullComponent

        c, root = _cache()
        leaf = _Node(root, 600)
        c.component_evictable_size_[FULL] = 600
        c.evictable_device_leaves = set() if stale else {leaf}
        c.eviction_strategy = types.SimpleNamespace(get_priority=lambda n: n.id)
        c._collect_all_nodes = lambda: [root, leaf]

        def upd(n):
            if n is not root and not n.children and all(cd.lock_ref == 0 for cd in n.component_data):
                c.evictable_device_leaves.add(n)

        def evict_leaf(n, tracker):
            tracker[FULL] += len(n.component_data[FULL].value)
            c.evictable_device_leaves.discard(n)

        c._update_evictable_leaf_sets = upd
        c._evict_device_leaf = evict_leaf
        comp = types.SimpleNamespace(cache=c, component_type=FULL)
        comp._peel = FullComponent._peel.__get__(comp)
        return FullComponent, comp, leaf

    def test_stale_membership_is_repaired_and_paid(self):
        FC, comp, _ = self._cache()
        tracker = {FULL: 0}
        with _env(**{EF.ENV: 1}), self.assertLogs(EF.logger, level="WARNING") as cap:
            FC.drive_eviction(comp, types.SimpleNamespace(num_tokens=512), tracker)
        self.assertEqual(tracker[FULL], 600)
        self.assertIn("EVICT-FRONTIER-CENSUS request=512 delivered_before=0 delivered_after_repair=600", cap.output[0])
        self.assertIn("stale_added=1", cap.output[0])

    def test_switch_off_is_the_old_peel(self):
        FC, comp, _ = self._cache()
        tracker = {FULL: 0}
        with _env(**{EF.ENV: 0}):
            FC.drive_eviction(comp, types.SimpleNamespace(num_tokens=512), tracker)
        self.assertEqual(tracker[FULL], 0)

    def test_a_peel_that_pays_scans_nothing(self):
        FC, comp, _ = self._cache(stale=False)
        comp.cache._collect_all_nodes = lambda: (_ for _ in ()).throw(AssertionError("no scan"))
        tracker = {FULL: 0}
        FC.drive_eviction(comp, types.SimpleNamespace(num_tokens=512), tracker)
        self.assertEqual(tracker[FULL], 600)

    def test_census_names_the_aux_locked_chain(self):
        c, root = _cache()
        a = _Node(root, 1000)
        t = _Node(a, 20, mamba_lock=1)
        c._collect_all_nodes = lambda: [root, a, t]
        c._update_evictable_leaf_sets = lambda n: None
        out = EF.census_and_repair(c, FULL)
        self.assertEqual(out["full_unlocked_tokens"], 1020)
        self.assertEqual(out["aux_locked_tokens"], {"1": 20})
        self.assertEqual(out["device_child_tokens"], 1000)


if __name__ == "__main__":
    unittest.main()


class CostAndRanks(unittest.TestCase):
    def _chain(self):
        c, root = _cache()
        c.component_protected_size_ = {FULL: 0}
        a = _Node(root, 1000)
        t = _Node(a, 20, mamba_lock=1)
        c.component_evictable_size_[FULL] = 1020
        EF.note_aux_lock(c, t, True)
        return c, t

    def test_memo_scans_once_until_an_input_moves(self):
        c, t = self._chain()
        with _env(**{EF.ENV_DELIVERABLE: 1}):
            for _ in range(5):
                self.assertEqual(EF.deliverable_evictable(c, FULL), 0)
            st = getattr(c, EF._STATS_ATTR)
            self.assertEqual((st["calls"], st["scans"]), (5, 1))
            c.component_evictable_size_[FULL] += 7   # an insert moved the count
            EF.deliverable_evictable(c, FULL)
            EF.note_aux_lock(c, t, False)             # the lock went
            self.assertEqual(EF.deliverable_evictable(c, FULL), 1027)
            self.assertEqual(st["scans"], 3)

    def test_the_cost_line(self):
        c, _ = self._chain()
        with _env(**{EF.ENV_DELIVERABLE: 1}), self.assertLogs(EF.logger, level="INFO") as cap:
            EF.deliverable_evictable(c, FULL)
        self.assertIn("ED-DELIVERABLE calls=1 scans=1 scan_us_mean=", cap.output[0])

    def test_a_form_a_worker_keeps_the_reported_count(self):
        c, _ = self._chain()
        setattr(c, EF.EXEMPT_ATTR, True)
        with _env(**{EF.ENV_DELIVERABLE: 1}):
            self.assertEqual(EF.deliverable_evictable(c, FULL), 1020)

    def test_the_scheduler_marks_form_a_workers(self):
        from sglang.srt.managers import scheduler as S

        src = open(S.__file__).read()
        i = src.index("adder.form_a_admission_follow = self._form_a_admission_follow_fn()")
        blk = src[i:i + 900]
        self.assertIn("setattr(self.tree_cache, _ef.EXEMPT_ATTR, _is_host is False)", blk)
