# SPDX-License-Identifier: Apache-2.0
"""ARENAFULL-1015: the shared host arena stays drainable under 210k-token load (NF h16n deaths, PP1).

MEASURED (boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwbar1dauer10150714_17c2563647_1015_071520.P.log, PP1):

  07:30:30 #1427 ARENA-CLAIM REFUSED pages=3 statuses=[4]          (a 192-token backup of weg2-17-57)
  07:30:30 W3-ARENA SPILL need=3 released_pages=3232 nodes=1 l3=on_disk:3232 ref_free=0/3232
           (node 208, 206848 tokens, still on the device: its host copy left, PP0/PP2 still held the slots)
  07:30:35 #1469 EVICT node=211..214 backuped=True ... HOSTLOCK-LRU-SKIP ct=MAMBA host_lock_ref=1
  07:30:35 #1421 BACKUP-REFUSED why=arena_claim node=208 tokens=206848 parent_is_root=True
  07:30:35 UD-REFUSED why=host_children:child_host_locked node=208 tokens=206848 holder=211
  07:30:35 LB-VERDICT ROW-ROOM SHORT rid=weg2-16-53 kv_tokens=209472 evictable=255040 evicted=48192 -> group stop

and boot ..._1015_070103 PP1 07:11:54: node 107 (unbacked, backup refused 60x), its host-only child 108 host-locked
(HOSTLOCK-LRU-SKIP site=evict_to_host), ``EVICT-FRONTIER-CENSUS on_frontier=2816 behind_device_child=205312`` ->
PrefillOutOfPages need 16448.

Driven through the REAL ``UnifiedRadixCache._evict_device_leaf`` / ``_weg2_l3_drop`` / ``_ud_clear_host_children`` /
``_w3_arena_spill`` and the real ``weg2_told_fallback`` ack functions; the trees' I/O steps are stubs.
RED on 4f58ca0256 (desk/nf-h16o-int-1015), GREEN on the fix.
"""

import os
import time
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import (  # noqa: E402
    BASE_COMPONENT_TYPE as FULL,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.weg2 import pp_slot_fidelity as SF  # noqa: E402

# the numbers of 07:30:35 (tokens) -- page 1, so rows == tokens
NODE208 = 206848
TAIL211 = 231


class _CD:
    def __init__(self, value=None, host_value=None, host_lock=0):
        self.value = value
        self.host_value = host_value
        self.lock_ref = 0
        self.host_lock_ref = host_lock


class _Node:
    _n = 200

    def __init__(self, parent, *, device=0, host=0, evicted=False, backuped=False, host_lock=0, l3=False):
        _Node._n += 1
        self.id = _Node._n
        self.parent = parent
        self.children = {}
        self.evicted = evicted
        self.backuped = backuped
        self.l3_present = l3
        self.key = list(range(device or host))
        self.component_data = [
            _CD(list(range(device)) if device else None, list(range(host)) if host else None, host_lock),
            _CD(), _CD(),
        ]
        if parent is not None:
            parent.children[self.id] = self


class _Tree:
    """The P stage's tree around node 208: the real eviction methods, the I/O stubbed."""

    _evict_device_leaf = UnifiedRadixCache._evict_device_leaf
    _is_device_leaf = UnifiedRadixCache._is_device_leaf
    _is_host_leaf = UnifiedRadixCache._is_host_leaf
    _ud_clear_host_children = UnifiedRadixCache._ud_clear_host_children

    def __init__(self, local_pp=True):
        self.root_node = _Node(None)
        self.cache_controller = types.SimpleNamespace(write_policy="write_back")
        self.ongoing_write_through = {}
        self.ongoing_backup = {}
        self.ongoing_prefetch = {}
        self.ack_queue = []          # the backup thread's acks, not yet drained
        self.tree_components = (FULL,)
        setattr(self, SF.FLOOR_LOCAL_PP_ATTR, local_pp)
        self.dropped, self.host_evicted, self.claims = [], [], 0

    def write_backup(self, node, write_back=False, kv_only_if_mamba_refused=False):
        self.claims += 1
        return 0  # '#1421 BACKUP-REFUSED why=arena_claim': the arena has no free slot

    def writing_check(self, write_back=False):
        return None

    def _evict_host_leaf(self, node, tracker):
        assert self._is_host_leaf(node)
        node.parent.children.pop(node.id)
        tracker[FULL] += len(node.component_data[FULL].host_value)
        self.host_evicted.append(node.id)

    def _ud_drop_unbacked_leaf(self, node, tracker):
        assert not node.children, "#841: never drop a node with children"
        node.parent.children.pop(node.id)
        tracker[FULL] += len(node.component_data[FULL].value)
        self.dropped.append(node.id)

    # the store write of a host node holds its host lock until its ack is drained (write_backup_storage)
    def store_write(self, node, op_id, acked=True):
        for cd in node.component_data:
            if cd.host_value is not None or cd is node.component_data[0]:
                cd.host_lock_ref += 1
        self.ongoing_backup[op_id] = (node, None)
        if acked:
            self.ack_queue.append(op_id)

    def drain_backup_acks_uncoordinated(self):
        n = 0
        while self.ack_queue:
            op = self.ack_queue.pop(0)
            node, _p = self.ongoing_backup.pop(op)
            node.l3_present = True
            for cd in node.component_data:
                cd.host_lock_ref = max(0, cd.host_lock_ref - 1)
            n += 1
        return n


def _death2(tree, *, acked=True, holder="l3_write"):
    """node 208 spilled by W3 (device rows, no host copy, l3_present); its tail 211 demoted to the host by the
    same peel, host-locked by its own store write (or by something else)."""
    n208 = _Node(tree.root_node, device=NODE208, l3=True)
    n211 = _Node(n208, host=TAIL211, evicted=True, backuped=True, l3=True)
    if holder == "l3_write":
        tree.store_write(n211, op_id=77, acked=acked)
    else:  # a prefetch / pin holder: no store write behind the lock
        n211.component_data[0].host_lock_ref = 1
    return n208, n211


def _ov(name, value):
    """``envs.<name>.override(value)``; on a tree without the switch (the base 4f58ca0256) a no-op, so the scenario
    runs the older behaviour and goes red by BEHAVIOUR, never by an import error."""
    import contextlib

    field = getattr(envs, name, None)
    return field.override(value) if field is not None else contextlib.nullcontext()


def _env(on=True, wait_ms=None):
    stack = [_ov("SGLANG_WEG2_L3_DROP", on)]
    if wait_ms is not None:
        stack.append(_ov("SGLANG_WEG2_L3_DROP_ACK_WAIT_MS", wait_ms))
    return stack


class _Envs:
    def __init__(self, *cms):
        self.cms = cms

    def __enter__(self):
        for c in self.cms:
            c.__enter__()

    def __exit__(self, *a):
        for c in reversed(self.cms):
            c.__exit__(*a)


class L3DropTest(unittest.TestCase):
    """Death 2 (07:30:35) and its neighbours."""

    def setUp(self):
        self._sf = os.environ.get(SF.ENV)
        os.environ[SF.ENV] = "1"

    def tearDown(self):
        if self._sf is None:
            os.environ.pop(SF.ENV, None)
        else:
            os.environ[SF.ENV] = self._sf

    def test_07_30_35_node_208_leaves_the_device(self):
        """RED on 4f58ca0256: BACKUP-REFUSED + UD-REFUSED child_host_locked -> 0 of 206848 paid -> ROW-ROOM SHORT."""
        tree = _Tree()
        n208, n211 = _death2(tree)
        tracker = {FULL: 0}
        with _Envs(*_env(True)):
            tree._evict_device_leaf(n208, tracker)
        self.assertEqual(tree.dropped, [n208.id], "the store-backed leaf is dropped (its KV is in L3)")
        self.assertEqual(tree.host_evicted, [n211.id], "its host-only tail leaves the host first (UD-H)")
        self.assertGreaterEqual(tracker[FULL], NODE208)
        self.assertEqual(tree.claims, 0, "no new arena claim for a store-backed node (no refusal, no next spill)")
        self.assertEqual(tree.ongoing_backup, {}, "the tail's store-write ack was drained on this rank")

    def test_unacked_store_write_waits_bounded_then_refuses_by_name(self):
        """The store write has not acked within the bound: nothing is dropped, the old path runs (named)."""
        tree = _Tree()
        n208, _ = _death2(tree, acked=False)
        tracker = {FULL: 0}
        with _Envs(*_env(True, wait_ms=30)), self.assertLogs("sglang.srt.mem_cache.unified_radix_cache",
                                                              level="WARNING") as cm:
            tree._evict_device_leaf(n208, tracker)
        self.assertEqual((tracker[FULL], tree.dropped), (0, []))
        self.assertTrue(any("L3-DROP REFUSED" in m and "holder_source=l3_write" in m for m in cm.output))

    def test_a_prefetch_holder_is_never_drained_over(self):
        """A lock that is not the child's own store write: no wait, no drop (UD-H's all-or-nothing stands)."""
        tree = _Tree()
        n208, n211 = _death2(tree, holder="prefetch")
        tree.ongoing_backup[9] = (_Node(tree.root_node, host=8, evicted=True, backuped=True), None)  # unrelated write
        tracker = {FULL: 0}
        t0 = time.perf_counter()
        with _Envs(*_env(True, wait_ms=1000)):
            tree._evict_device_leaf(n208, tracker)
        self.assertLess(time.perf_counter() - t0, 0.5, "no wait on a holder no drain can free")
        self.assertEqual((tracker[FULL], tree.dropped, tree.host_evicted), (0, [], []))
        self.assertEqual(n211.component_data[0].host_lock_ref, 1)

    def test_off_the_local_pp_floor_nothing_changes(self):
        """D / any TP group: a rank-local drop would split the replicas."""
        tree = _Tree(local_pp=False)
        n208, _ = _death2(tree)
        tracker = {FULL: 0}
        with _Envs(*_env(True)):
            tree._evict_device_leaf(n208, tracker)
        self.assertEqual((tracker[FULL], tree.dropped, tree.host_evicted), (0, [], []))

    def test_switch_off_is_the_old_claim_first_path(self):
        tree = _Tree()
        n208, _ = _death2(tree)
        tracker = {FULL: 0}
        with _Envs(*_env(False)):
            tree._evict_device_leaf(n208, tracker)
        self.assertEqual((tracker[FULL], tree.dropped), (0, []))
        self.assertGreaterEqual(tree.claims, 1, "off: the claim is tried first, as before")

    def test_a_backed_node_is_untouched(self):
        """A node WITH its host copy demotes as before -- L3-DROP never takes a backed node."""
        tree = _Tree()
        n = _Node(tree.root_node, device=64, host=64, backuped=True, l3=True)
        fn = getattr(UnifiedRadixCache, "_weg2_l3_drop", None)
        if fn is None:
            self.skipTest("tree without L3-DROP")
        self.assertFalse(fn(tree, n, {FULL: 0}))

    def test_a_childless_spilled_leaf_is_dropped_without_a_claim(self):
        tree = _Tree()
        n = _Node(tree.root_node, device=4096, l3=True)
        tracker = {FULL: 0}
        with _Envs(*_env(True)):
            tree._evict_device_leaf(n, tracker)
        self.assertEqual((tree.dropped, tracker[FULL], tree.claims), ([n.id], 4096, 0))


class UdAckDrainTest(unittest.TestCase):
    """Death 1 (07:11:54): node 107 is NOT in L3 (a fresh tail, backup refused); its host-only child 108, just
    demoted by the peel, is host-locked by its own store write. The plain UD drop drains that ack too."""

    def setUp(self):
        self._sf = os.environ.get(SF.ENV)
        os.environ[SF.ENV] = "1"

    def tearDown(self):
        if self._sf is None:
            os.environ.pop(SF.ENV, None)
        else:
            os.environ[SF.ENV] = self._sf

    def test_07_11_54_node_107_is_paid_and_the_chain_behind_frees(self):
        """RED on 4f58ca0256: UD-REFUSED child_host_locked node=107 holder=108 -> behind_device_child=205312."""
        tree = _Tree()
        n106 = _Node(tree.root_node, device=205312, host=205312, backuped=True)
        n107 = _Node(n106, device=2816)                       # un-backed, not in L3
        n108 = _Node(n107, host=64, evicted=True, backuped=True)
        tree.store_write(n108, op_id=5)
        tracker = {FULL: 0}
        with _Envs(*_env(True)):
            tree._evict_device_leaf(n107, tracker)
        self.assertEqual(tree.dropped, [n107.id])
        self.assertEqual(tree.host_evicted, [n108.id])
        self.assertTrue(tree._is_device_leaf(n106), "106 is the next frontier leaf: 205312 tokens behind it are paid")

    def test_switch_off_keeps_the_old_refusal(self):
        tree = _Tree()
        n107 = _Node(tree.root_node, device=2816)
        n108 = _Node(n107, host=64, evicted=True, backuped=True)
        tree.store_write(n108, op_id=5)
        tracker = {FULL: 0}
        with _Envs(*_env(False)):
            tree._evict_device_leaf(n107, tracker)
        self.assertEqual((tracker[FULL], tree.dropped), (0, []))


class _SpillPool:
    staging_rows = 0

    def __init__(self):
        self.arena = object()
        self.secured = []

    def secure_rows_to_l3(self, host_indices):
        n = int(host_indices.numel())
        self.secured.append(n)
        return {"pages": n, "on_disk": n, "written": 0, "lost": 0}


class W3KeepPayableTest(unittest.TestCase):
    """07:30:30: the spill took node 208 (3232 pages) for a 3-page claim; once its tail is host-locked the node can
    leave the device only by a drop that UD-H cannot clear. W3-KEEP-PAYABLE keeps such a node's host copy."""

    def setUp(self):
        self._sf = os.environ.get(SF.ENV)
        os.environ[SF.ENV] = "1"

    def tearDown(self):
        if self._sf is None:
            os.environ.pop(SF.ENV, None)
        else:
            os.environ[SF.ENV] = self._sf

    def _tree(self):
        from sglang.srt.mem_cache.base_prefix_cache import InsertParams
        from sglang.srt.mem_cache.radix_cache import RadixKey
        from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType

        from test_unified_radix_cache_unittest import CacheConfig, build_fixture

        cache, allocator, _ = build_fixture(CacheConfig(page_size=1, components=(ComponentType.FULL,)))
        # A = node 208's shape (64 tokens + a 16-token tail), B = an unrelated 48-token conversation
        for first, n in ((1, 64), (1, 80), (5000, 48)):
            cache.insert(InsertParams(key=RadixKey(list(range(first, first + n)), None),
                                      value=allocator.alloc(n).to(dtype=torch.int64)))
        row = 100
        nodes = {}
        for node in cache._collect_all_nodes():
            if node is cache.root_node:
                continue
            k = len(node.key)
            node.component_data[ComponentType.FULL].host_value = torch.arange(row, row + k, dtype=torch.int64)
            row += k
            nodes[(node.key.token_ids[0], k)] = node
        a, tail, b = nodes[(1, 64)], nodes[(65, 16)], nodes[(5000, 48)]
        # the tail went to the host (demoted) and is host-locked: held by something UD-H cannot clear
        tail.component_data[ComponentType.FULL].value = None
        tail.component_data[ComponentType.FULL].host_lock_ref = 1
        SF.mark_floor_scope(cache, True)
        return cache, a, tail, b

    def _spill(self, cache, on):
        released = []

        def _release(self_, node, comp, target=None, tracker=None):
            released.append(node.id)
            hv = node.component_data[FULL].host_value
            node.component_data[FULL].host_value = None
            return 0, int(hv.numel())

        with _ov("SGLANG_WEG2_W3_SPILL_KEEP_PAYABLE", on), \
                envs.SGLANG_WEG2_ENABLE_W3_SPILL_ANCHOR_POOL.override(True), \
                mock.patch.object(UnifiedRadixCache, "_evict_component_and_detach_lru", _release):
            got = cache._w3_arena_spill(_SpillPool(), 128)
        return got, released

    def test_07_30_30_the_node_with_a_held_tail_keeps_its_host_copy(self):
        """RED on 4f58ca0256: node A (the 208 shape) is spilled and becomes undroppable."""
        cache, a, _tail, b = self._tree()
        _got, released = self._spill(cache, True)
        self.assertNotIn(a.id, released, "a spill must not make a device node undroppable")
        self.assertIn(b.id, released, "the payable candidate is still spilled")
        self.assertTrue(a.backuped)

    def test_switch_off_spills_every_written_device_node(self):
        cache, a, _tail, b = self._tree()
        _got, released = self._spill(cache, False)
        self.assertIn(a.id, released)
        self.assertIn(b.id, released)


class AckRoomTest(unittest.TestCase):
    """H64-AUFLAGE 1 (B1) and ACK-ROOM-PAYABLE: the follower's told ack never names a depth it cannot load back."""

    def _sched(self, avail, reported, payable_tree=None):
        tree = payable_tree or types.SimpleNamespace()
        tree.token_to_kv_pool_allocator = types.SimpleNamespace(available_size=lambda: avail)
        tree.evictable_size = lambda: reported
        return types.SimpleNamespace(tree_cache=tree, ps=types.SimpleNamespace(pp_rank=1, pp_size=3))

    def _req(self):
        return types.SimpleNamespace(rid="weg2-16-53")

    def test_b1_unresumable_branch_is_room_checked(self):
        """RED on 4f58ca0256: res < own acked res unchecked; with GROUP-FLOOR equal acks admit res -> ROW-ROOM SHORT."""
        from sglang.srt.managers import weg2_told_fallback as fb
        from sglang.srt.managers import weg2_told_fidelity as tf

        s = self._sched(avail=4096, reported=0)
        with mock.patch.object(tf, "rank_resumable", lambda sch, req, n: 127360), \
                mock.patch.object(fb, "_loadback_rows", lambda sch, req, n: n), \
                mock.patch.object(fb, "_ack_evictable", lambda tree: 48192, create=True):
            self.assertEqual(fb._resumable_own(s, self._req(), "weg2-16-53", 212480), 0)

    def test_b1_satisfied_branch_is_room_checked(self):
        from sglang.srt.managers import weg2_told_fallback as fb
        from sglang.srt.managers import weg2_told_fidelity as tf

        s = self._sched(avail=4096, reported=0)
        with mock.patch.object(tf, "rank_resumable", lambda sch, req, n: 127360), \
                mock.patch.object(tf, "enabled", lambda: True), \
                mock.patch.object(fb, "env_on", lambda: True), \
                mock.patch.object(fb, "_loadback_rows", lambda sch, req, n: n), \
                mock.patch.object(fb, "_ack_evictable", lambda tree: 48192, create=True):
            self.assertEqual(fb._satisfied_resumable(s, self._req(), "weg2-16-53", 212480), 0)

    def test_b1_with_room_the_reach_stands(self):
        from sglang.srt.managers import weg2_told_fallback as fb
        from sglang.srt.managers import weg2_told_fidelity as tf

        s = self._sched(avail=262144, reported=0)
        with mock.patch.object(tf, "rank_resumable", lambda sch, req, n: 127360), \
                mock.patch.object(fb, "_loadback_rows", lambda sch, req, n: n), \
                mock.patch.object(fb, "_ack_evictable", lambda tree: 0, create=True):
            self.assertEqual(fb._resumable_own(s, self._req(), "weg2-16-53", 212480), 127360)

    def test_07_30_35_reported_evictable_is_not_room(self):
        """The 07:30:35 pool: avail 4096, reported evictable 255040, one peel pays 48192 (node 208 undroppable).
        RED on 4f58ca0256: room = 4096 + 255040 >= 209472 -> the ack says told and PP0 admits a load-back the
        stage cannot hold. GREEN: the ack counts what the peel pays -> 0, PP0 answers told=0 for every rank."""
        from sglang.srt.managers import weg2_told_fallback as fb
        from sglang.srt.weg2 import pp_room_vote as PR

        tree = types.SimpleNamespace(cache_controller=None)
        setattr(tree, SF.FLOOR_LOCAL_PP_ATTR, True)
        s = self._sched(avail=4096, reported=255040, payable_tree=tree)
        with mock.patch.object(fb, "_loadback_rows", lambda sch, req, n: 209472), \
                mock.patch.object(PR, "estimate_payable", lambda t, local_floor, host_free: 48192), \
                _ov("SGLANG_WEG2_ACK_ROOM_PAYABLE", True):
            self.assertEqual(fb._room_own(s, self._req(), "weg2-16-53", 209472), 0)
        with mock.patch.object(fb, "_loadback_rows", lambda sch, req, n: 209472), \
                mock.patch.object(PR, "estimate_payable", lambda t, local_floor, host_free: 48192), \
                _ov("SGLANG_WEG2_ACK_ROOM_PAYABLE", False):
            self.assertEqual(fb._room_own(s, self._req(), "weg2-16-53", 209472), 209472,
                             "off: the reported evictable size, as before")

    def test_07_23_55_no_room_while_a_210k_prefill_holds_the_pool_stays_zero(self):
        """weg2-6-18 07:23:55: PP1 held weg2-0-6's 208320-token load (running, locked) -> room 50752 < 211776.
        That ack 0 was RIGHT on both trees (two 210k prompts do not fit one 262144-token stage); pinned."""
        from sglang.srt.managers import weg2_told_fallback as fb

        s = self._sched(avail=2560, reported=48192)
        with mock.patch.object(fb, "_loadback_rows", lambda sch, req, n: 211776), \
                mock.patch.object(fb, "_ack_evictable", lambda tree: 48192, create=True):
            self.assertEqual(fb._room_own(s, self._req(), "weg2-6-18", 211776), 0)


class OrphanClaimReapTest(unittest.TestCase):
    """(b) boot 070103 mamba arena: 'claimed=1 oldest=slot 20/274319ms/pid788/host-write/g1/open0' -- a claim with
    no open writer and no byte for 274 s, never reaped: every reader that met the stem JOINED it (which refreshes
    touched_ms, the age the reap read), waited and gave the claim up again. Real C arena on a temp file."""

    SLOT = 4096

    def setUp(self):
        import shutil
        import tempfile

        if shutil.which("gcc") is None:
            self.skipTest("needs gcc for the C arena")
        from sglang.srt.mem_cache.storage.file.hicache_arena import ShmArena

        self.dir = tempfile.mkdtemp()
        self.arena = ShmArena(os.path.join(self.dir, "arena-mamba.bin"), self.SLOT, 4)

    def tearDown(self):
        import shutil

        self.arena.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _orphan(self, stem):
        """host-write claim (fresh), a quarter merged, the writer resolved: CLAIMED, open 0, never COMPLETE."""
        (s, st, g), = self.arena.claim_slots([stem], [self.SLOT])
        assert st == 0
        assert self.arena.complete_slots([s], [g], [(0, self.SLOT // 4)]) == [0]
        return s, g

    def _reader_joins_and_gives_up(self, stem):
        (s, st, g), = self.arena.claim_slots([stem], [self.SLOT])
        assert st == 1, "a JOIN of the orphan"
        assert self.arena.unclaim([s], [g]) == 1

    def test_070103_a_joined_orphan_is_reaped_after_its_byte_age(self):
        """RED on 4f58ca0256: the joins keep touched_ms young -- reap_partial never takes the slot."""
        s, _g = self._orphan("anchor-stem")
        for _ in range(6):
            time.sleep(0.05)
            self._reader_joins_and_gives_up("anchor-stem")   # every 50 ms a reader meets the stem
        freed = self.arena.reap_partial(0.2)                  # no byte for >= 300 ms, touched 0 ms ago
        self.assertEqual(freed, [s])
        self.assertEqual(self.arena.find_slots(["anchor-stem"]), [(-1, 0)], "the stem is claimable afresh")

    def test_an_open_writer_is_never_reaped(self):
        """A writer that claimed and has not resolved (open 1) keeps its claim, however old."""
        (s, st, g), = self.arena.claim_slots(["live"], [self.SLOT])
        assert st == 0
        time.sleep(0.25)
        self.assertEqual(self.arena.reap_partial(0.2), [])

    def test_a_young_orphan_stays(self):
        s, _g = self._orphan("young")
        self.assertEqual(self.arena.reap_partial(5.0), [])


if __name__ == "__main__":
    unittest.main()
