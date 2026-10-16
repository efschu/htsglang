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


# ---------------------------------------------------------------------------------------------------------------
# H16P-NACHBESSERUNG (Opus review nf-arenafull-review-1015, JA_MIT_AUFLAGE): A1 / A2 / A3 and E1 / E2.
# ---------------------------------------------------------------------------------------------------------------


class _OkTree(_Tree):
    """The arena HAS room: the backup claim succeeds, the leaf is demoted (review probe P4 / P3)."""

    def __init__(self, local_pp=True, room=10**9):
        super().__init__(local_pp)
        # backup_room_tokens(pool) reads pool.available_size() on a pool without a bound arena
        self.cache_controller = types.SimpleNamespace(
            write_policy="write_back", mem_pool_host=types.SimpleNamespace(available_size=lambda: room))
        self.demoted = []

    def write_backup(self, node, write_back=False, kv_only_if_mamba_refused=False):
        self.claims += 1
        node.component_data[0].host_value = list(range(len(node.key)))
        node.backuped = True
        return len(node.key)

    def _evict_to_host(self, node, tracker):
        node.component_data[0].value = None
        tracker[FULL] += len(node.key)
        self.demoted.append(node.id)


class _SfBase(unittest.TestCase):
    def setUp(self):
        self._sf = os.environ.get(SF.ENV)
        os.environ[SF.ENV] = "1"

    def tearDown(self):
        if self._sf is None:
            os.environ.pop(SF.ENV, None)
        else:
            os.environ[SF.ENV] = self._sf


class DrainBudgetPerChainTest(_SfBase):
    """A1 (B1 HOCH): the ack-drain wait is ONE budget per eviction chain, not one per call and node."""

    def test_P1_one_leaf_waits_at_most_the_bound(self):
        """Was 0.608 s at a 0.3 s bound (L3-DROP waits, then the UD branch of the same call waits again)."""
        tree = _Tree()
        n208, _ = _death2(tree, acked=False)
        t0 = time.perf_counter()
        with _Envs(*_env(True, wait_ms=300)):
            tree._evict_device_leaf(n208, {FULL: 0})
        el = time.perf_counter() - t0
        self.assertLess(el, 0.45, f"one eviction call waited {el:.3f}s for a 0.3 s bound")
        self.assertGreaterEqual(el, 0.25, "the one wait is still taken (the ack may come)")

    def test_P2_a_peel_over_k_leaves_waits_one_bound_in_total(self):
        """Was 1.621 s for 4 leaves at a 0.2 s bound (4 x 2 x bound)."""
        tree = _Tree()
        leaves = []
        for i in range(4):
            n = _Node(tree.root_node, device=1024, l3=True)
            c = _Node(n, host=64, evicted=True, backuped=True)
            tree.store_write(c, op_id=100 + i, acked=False)
            leaves.append(n)
        t0 = time.perf_counter()
        with _Envs(*_env(True, wait_ms=200)):
            for n in leaves:
                tree._evict_device_leaf(n, {FULL: 0})
        el = time.perf_counter() - t0
        self.assertLess(el, 0.45, f"one peel over 4 leaves blocked {el:.3f}s for a 0.2 s bound")

    def test_a_late_ack_inside_the_budget_still_drops(self):
        """The wait is not removed: an ack that lands inside the bound lets the leaf leave."""
        import threading

        tree = _Tree()
        n208, n211 = _death2(tree, acked=False)
        threading.Timer(0.05, lambda: tree.ack_queue.append(77)).start()
        tracker = {FULL: 0}
        with _Envs(*_env(True, wait_ms=1000)):
            tree._evict_device_leaf(n208, tracker)
        self.assertEqual(tree.dropped, [n208.id])
        self.assertGreaterEqual(tracker[FULL], NODE208)

    def test_an_exhausted_chain_still_drains_once_without_waiting(self):
        """Budget spent: the next drain runs ONCE (an ack already in the queue is taken), bound 0."""
        tree = _Tree()
        tree._weg2_l3_drain_spent_ms = 5000.0
        n208, _ = _death2(tree, acked=True)         # the ack is in the queue
        t0 = time.perf_counter()
        with _Envs(*_env(True, wait_ms=1000)):
            tree._evict_device_leaf(n208, {FULL: 0})
        self.assertLess(time.perf_counter() - t0, 0.2)
        self.assertEqual(tree.dropped, [n208.id], "the single no-wait drain took the queued ack")

    def test_a_spent_budget_is_not_waited_again_even_without_a_stall_mark(self):
        """The budget alone (no timeout happened yet, e.g. earlier leaves used it up by late acks) ends the wait."""
        tree = _Tree()
        tree._weg2_l3_drain_spent_ms = 1000.0
        n208, _ = _death2(tree, acked=False)
        t0 = time.perf_counter()
        with _Envs(*_env(True, wait_ms=1000)):
            tree._evict_device_leaf(n208, {FULL: 0})
        self.assertLess(time.perf_counter() - t0, 0.2, "a chain that spent its whole budget waits no more")

    def test_a_part_spent_budget_waits_only_the_rest(self):
        tree = _Tree()
        tree._weg2_l3_drain_spent_ms = 800.0
        n208, _ = _death2(tree, acked=False)
        t0 = time.perf_counter()
        with _Envs(*_env(True, wait_ms=1000)):
            tree._evict_device_leaf(n208, {FULL: 0})
        el = time.perf_counter() - t0
        self.assertGreaterEqual(el, 0.15)
        self.assertLess(el, 0.4, f"waited {el:.3f}s of a 0.2 s rest")

    def test_after_a_timeout_the_next_chains_do_not_wait_until_the_stall_ends(self):
        tree = _Tree()
        n208, _ = _death2(tree, acked=False)
        with _Envs(*_env(True, wait_ms=200)):
            tree._evict_device_leaf(n208, {FULL: 0})           # times out -> stalled
            self.assertGreater(tree._weg2_l3_drain_stalled_until, time.monotonic())
            UnifiedRadixCache._weg2_l3_drain_budget_reset(tree)  # a NEW chain
            t0 = time.perf_counter()
            tree._evict_device_leaf(n208, {FULL: 0})
            self.assertLess(time.perf_counter() - t0, 0.1, "a stalled store is not waited for again")
            tree._weg2_l3_drain_stalled_until = 0.0              # the stall is over
            UnifiedRadixCache._weg2_l3_drain_budget_reset(tree)
            t0 = time.perf_counter()
            tree._evict_device_leaf(n208, {FULL: 0})
            self.assertGreaterEqual(time.perf_counter() - t0, 0.15, "a fresh chain after the stall waits again")

    def _chain_tree(self, k):
        tree = _Tree()
        leaves = []
        for i in range(k):
            n = _Node(tree.root_node, device=512, l3=True)
            c = _Node(n, host=32, evicted=True, backuped=True)
            tree.store_write(c, op_id=300 + i, acked=False)
            leaves.append(n)
        tree.disable = False
        tree.tree_components = (FULL,)
        tree.evict = types.MethodType(UnifiedRadixCache.evict, tree)
        tree.update_eviction_metrics = lambda *a, **k: None
        tree.evictable_device_leaves = set()

        class _Comp:
            def drive_eviction(self_, params, tracker):
                for n in leaves:
                    tree._evict_device_leaf(n, tracker)

        tree._components_tuple = (_Comp(),)
        return tree

    def test_evict_is_one_chain_k_leaves_one_bound_and_the_next_evict_starts_fresh(self):
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        tree = self._chain_tree(3)
        with _Envs(*_env(True, wait_ms=200)):
            t0 = time.perf_counter()
            tree.evict(EvictParams(num_tokens=1))
            first = time.perf_counter() - t0
            self.assertLess(first, 0.4, f"one evict() over 3 leaves waited {first:.3f}s for a 0.2 s bound")
            self.assertGreaterEqual(first, 0.15)
            tree._weg2_l3_drain_stalled_until = 0.0
            self.assertGreater(tree._weg2_l3_drain_spent_ms, 150.0)
            t0 = time.perf_counter()
            tree.evict(EvictParams(num_tokens=1))   # a new outermost evict(): the budget is back
            self.assertGreaterEqual(time.perf_counter() - t0, 0.15, "evict() opens a fresh budget")

    def test_a_nested_evict_does_not_refill_the_budget(self):
        from sglang.srt.mem_cache.base_prefix_cache import EvictParams

        tree = self._chain_tree(2)
        inner = tree._components_tuple[0]
        orig = inner.drive_eviction

        inside = []

        def nested(params, tracker):
            orig(params, tracker)
            if not inside:
                inside.append(1)
                tree.evict(EvictParams(num_tokens=1))   # e.g. the mamba component evicting inside the peel

        inner.drive_eviction = nested
        with _Envs(*_env(True, wait_ms=200)):
            t0 = time.perf_counter()
            tree.evict(EvictParams(num_tokens=1))
            self.assertLess(time.perf_counter() - t0, 0.4)


class L3PresentCompleteTest(_SfBase):
    """A2 (B2 MITTEL): l3_present only from a COMPLETE store write; unpaired pages are not an L3 copy."""

    class _Ack(_Tree):
        """The real ``_drain_storage_control_queues_impl`` over one acked store write."""

        _drain_storage_control_queues_impl = UnifiedRadixCache._drain_storage_control_queues_impl
        drain_backup_acks_uncoordinated = UnifiedRadixCache.drain_backup_acks_uncoordinated

        def __init__(self, completed, pages=4, page_size=16):
            super().__init__()
            from queue import Queue

            self.page_size = page_size
            self.cache_controller = types.SimpleNamespace(
                write_policy="write_back", ack_backup_queue=Queue(), prefetch_revoke_queue=Queue(),
                host_mem_release_queue=Queue(), mem_pool_host=types.SimpleNamespace(),
                extra_host_mem_release_queues={})
            self._drain_depth_every = 0
            self.pp_rank = 1
            self.enable_storage_metrics = False
            self.storage_metrics_collector = None
            self.staging_write_ring = None
            self.node = _Node(self.root_node, device=pages * page_size, host=pages * page_size, backuped=True)
            self.op = types.SimpleNamespace(id=9, completed_tokens=completed, hash_value=["h"] * pages)
            self.ongoing_backup[9] = (self.node, ("lock", 1))
            self.cache_controller.ack_backup_queue.put(self.op)
            self.unlocked = []

        def _drain_agreement_is_collective(self):
            return False

        def dec_host_lock_ref(self, node, params):
            self.unlocked.append(node.id)

        def _weg2_rebind_host_to_arena(self, node):
            return False

        def _weg2_host_is_transit(self):
            return False

    def _ack(self, completed):
        tree = self._Ack(completed)
        self.assertEqual(tree.drain_backup_acks_uncoordinated(), 1)
        self.assertEqual(tree.unlocked, [tree.node.id], "the ack always releases its own host lock")
        return tree

    def test_a_complete_write_sets_l3_present(self):
        self.assertTrue(self._ack(4 * 16).node.l3_present)

    def test_a_partial_ack_does_not(self):
        self.assertFalse(self._ack(1 * 16).node.l3_present)

    def test_a_stale_refused_operation_does_not(self):
        """operation_is_stale: the backup thread refuses the write and acks it anyway (completed_tokens 0)."""
        self.assertFalse(self._ack(0).node.l3_present)

    def test_switch_off_every_ack_sets_it(self):
        with _ov("SGLANG_WEG2_L3_PRESENT_STRICT", False):
            self.assertTrue(self._ack(0).node.l3_present)

    def test_a_partial_ack_leads_to_no_l3_drop(self):
        """The partial ack's node is NOT an L3-DROP candidate: the claim-first path runs (claims >= 1)."""
        tree = self._Ack(1 * 16)
        tree.drain_backup_acks_uncoordinated()
        node = tree.node
        node.backuped = False
        node.component_data[0].host_value = None   # the host copy is gone
        tracker = {FULL: 0}
        with _Envs(*_env(True)):
            tree._evict_device_leaf(node, tracker)
        self.assertGreaterEqual(tree.claims, 1, "no L3-DROP: the claim-first path ran")

    def test_the_complete_ack_node_is_an_l3_drop_candidate(self):
        tree = self._Ack(4 * 16)
        tree.drain_backup_acks_uncoordinated()
        node = tree.node
        node.backuped = False
        node.component_data[0].host_value = None
        with _Envs(*_env(True)):
            tree._evict_device_leaf(node, {FULL: 0})
        self.assertEqual((tree.claims, tree.dropped), (0, [node.id]))


class _UnpairedPool(_SpillPool):
    def __init__(self, unpaired=0, lost=0):
        super().__init__()
        self.unpaired, self.lost = unpaired, lost

    def secure_rows_to_l3(self, host_indices):
        n = int(host_indices.numel())
        if self.unpaired:
            return {"pages": n, "on_disk": 0, "written": 0, "lost": self.lost, "unpaired": self.unpaired,
                    "sidecar_absent": self.unpaired}
        return super().secure_rows_to_l3(host_indices)


class UnpairedSpillTest(W3KeepPayableTest):
    """A2: a spill whose pages are unpaired (KV page without its QSA index, not written) is a lost spill."""

    def _spill_with(self, cache, pool):
        released = []

        def _release(self_, node, comp, target=None, tracker=None):
            released.append(node.id)
            hv = node.component_data[FULL].host_value
            node.component_data[FULL].host_value = None
            return 0, int(hv.numel())

        with _ov("SGLANG_WEG2_W3_SPILL_KEEP_PAYABLE", True), \
                envs.SGLANG_WEG2_ENABLE_W3_SPILL_ANCHOR_POOL.override(True), \
                mock.patch.object(UnifiedRadixCache, "_evict_component_and_detach_lru", _release):
            got = cache._w3_arena_spill(pool, 128)
        return got, released

    def test_an_unpaired_spill_leaves_the_node_backed_and_names_unpaired(self):
        cache, _a, _tail, b = self._tree()
        with self.assertLogs("sglang.srt.mem_cache.unified_radix_cache", level="INFO") as cm:
            got, released = self._spill_with(cache, _UnpairedPool(unpaired=1))
        self.assertEqual((got, released), (0, []), "no release of the host copy")
        self.assertTrue(b.backuped)
        self.assertFalse(getattr(b, "l3_present", False), "no l3_present: the page is not in L3")
        line = [m for m in cm.output if "W3-ARENA SPILL" in m]
        self.assertTrue(line and "unpaired=1" in line[0], line)

    def test_switch_off_ignores_unpaired_as_before(self):
        cache, _a, _tail, b = self._tree()
        with _ov("SGLANG_WEG2_L3_PRESENT_STRICT", False):
            got, released = self._spill_with(cache, _UnpairedPool(unpaired=1))
        self.assertIn(b.id, released)
        self.assertTrue(b.l3_present)

    # the inherited W3KeepPayableTest scenarios are run by their own class; not again here
    test_07_30_30_the_node_with_a_held_tail_keeps_its_host_copy = None
    test_switch_off_spills_every_written_device_node = None


class SecureRowsPassUnpairedTest(unittest.TestCase):
    def test_secure_rows_to_l3_passes_unpaired_and_sidecar_absent_on(self):
        from sglang.srt.mem_cache.pool_host import arena_pool as ap

        pool = types.SimpleNamespace(
            arena=types.SimpleNamespace(slot_bytes=64, slot_states=lambda sl: [2] * len(sl)), _page_bytes=64,  # L3-COMPLETE A4: fail-closed without the header read
            _backend=types.SimpleNamespace(arena_secure_to_disk=lambda arena, cands: {
                "on_disk": 1, "written": 0, "lost": 0, "unpaired": 2, "sidecar_absent": 3}))
        with mock.patch.object(ap, "arena_ref_slots", lambda p, h: torch.tensor([5, 6, 7])):
            out = ap.ArenaMHAHostPool.secure_rows_to_l3(pool, torch.arange(3))
        self.assertEqual((out["pages"], out["on_disk"], out["unpaired"], out["sidecar_absent"]), (3, 1, 2, 3))


class HostOnlySpillUnpairedTest(_SfBase):
    def test_a_host_only_leaf_with_unpaired_pages_stays(self):
        from test_w3_spill_host_leaves_int22_1008 import SPAN, _AnchorPool, _death_shape, _spill

        class _Pool(_AnchorPool):
            def secure_rows_to_l3(self, host_indices):
                out = super().secure_rows_to_l3(host_indices)
                out.update(unpaired=1, on_disk=0)
                return out

        cache, top, mid, leaf = _death_shape()
        self.assertEqual(_spill(cache, _Pool(), top, on=True), 0)
        self.assertIn(leaf, cache._collect_all_nodes(), "an unpaired host-only leaf is not dropped")
        cache2, top2, mid2, leaf2 = _death_shape()
        with _ov("SGLANG_WEG2_L3_PRESENT_STRICT", False):
            self.assertEqual(_spill(cache2, _Pool(), top2, on=True), 2 * SPAN)


class NotInL3IsNeverDroppedTest(_SfBase):
    """A3 (Testluecke HOCH, review probe P3 / mutant MZ): L3-DROP needs l3_present."""

    def test_P3_a_node_not_in_l3_is_demoted_when_its_claim_succeeds(self):
        tree = _OkTree()
        n = _Node(tree.root_node, device=4096, l3=False)
        with _Envs(*_env(True)):
            tree._evict_device_leaf(n, {FULL: 0})
        self.assertEqual(tree.dropped, [], "a node with no L3 copy was dropped although its claim would succeed")
        self.assertEqual(tree.claims, 1)
        self.assertEqual(tree.demoted, [n.id])

    def test_P3b_even_without_arena_room_a_non_l3_node_is_claimed_first(self):
        """No room AND not in L3: L3-DROP must still not fire (the claim is tried; the UD drop is the old path)."""
        tree = _OkTree(room=0)
        n = _Node(tree.root_node, device=4096, l3=False)
        with _Envs(*_env(True)):
            tree._evict_device_leaf(n, {FULL: 0})
        self.assertEqual(tree.claims, 1)


class DropOnlyUnderPressureTest(_SfBase):
    """E1 (B3): with arena room an l3_present leaf is demoted to L2 (hits stay); without room it is dropped."""

    def test_P4_with_room_the_l3_leaf_is_demoted_not_dropped(self):
        tree = _OkTree(room=10**9)
        n = _Node(tree.root_node, device=4096, l3=True)
        with _Envs(*_env(True)):
            tree._evict_device_leaf(n, {FULL: 0})
        self.assertEqual((tree.dropped, tree.claims, tree.demoted), ([], 1, [n.id]))

    def test_room_equal_to_the_tokens_is_room(self):
        tree = _OkTree(room=4096)
        n = _Node(tree.root_node, device=4096, l3=True)
        with _Envs(*_env(True)):
            tree._evict_device_leaf(n, {FULL: 0})
        self.assertEqual((tree.dropped, tree.demoted), ([], [n.id]))

    def test_one_token_short_of_room_the_leaf_is_dropped_without_a_claim(self):
        tree = _OkTree(room=4095)
        n = _Node(tree.root_node, device=4096, l3=True)
        with _Envs(*_env(True)):
            tree._evict_device_leaf(n, {FULL: 0})
        self.assertEqual((tree.dropped, tree.claims), ([n.id], 0))

    def test_switch_off_is_the_arenafull_behaviour_drop_even_with_room(self):
        tree = _OkTree(room=10**9)
        n = _Node(tree.root_node, device=4096, l3=True)
        with _Envs(*_env(True)), _ov("SGLANG_WEG2_L3_DROP_ONLY_UNDER_PRESSURE", False):
            tree._evict_device_leaf(n, {FULL: 0})
        self.assertEqual((tree.dropped, tree.claims), ([n.id], 0))

    def test_an_unreadable_room_reads_as_pressure(self):
        tree = _Tree()         # cache_controller without mem_pool_host: room 0
        n = _Node(tree.root_node, device=64, l3=True)
        with _Envs(*_env(True)):
            tree._evict_device_leaf(n, {FULL: 0})
        self.assertEqual((tree.dropped, tree.claims), ([n.id], 0))


class OwnWritesOnlyTest(_SfBase):
    """E2 (B8): the drain runs only when a child's host lock IS its own open store writes."""

    def _mixed(self, tree):
        n208, n211 = _death2(tree, acked=False)
        n211.component_data[0].host_lock_ref += 1     # + a prefetch / pin reference no ack frees
        return n208, n211

    def test_a_child_with_an_extra_lock_is_not_waited_for(self):
        tree = _Tree()
        n208, n211 = self._mixed(tree)
        t0 = time.perf_counter()
        with _Envs(*_env(True, wait_ms=1000)):
            tree._evict_device_leaf(n208, {FULL: 0})
        self.assertLess(time.perf_counter() - t0, 0.3, "the extra lock is no store write: the wait is for nothing")
        self.assertEqual((tree.dropped, tree.host_evicted), ([], []))

    def test_an_acked_write_plus_an_extra_lock_is_still_refused(self):
        tree = _Tree()
        n208, n211 = _death2(tree, acked=True)
        n211.component_data[0].host_lock_ref += 1
        with _Envs(*_env(True, wait_ms=100)):
            tree._evict_device_leaf(n208, {FULL: 0})
        self.assertEqual(tree.dropped, [])

    def test_two_own_writes_two_references_are_drained(self):
        """KV write + sidecar write of one child: two lock references, both own."""
        tree = _Tree()
        n208, n211 = _death2(tree, acked=True)
        tree.store_write(n211, op_id=78, acked=True)
        self.assertEqual(max(cd.host_lock_ref for cd in n211.component_data), 2)
        with _Envs(*_env(True, wait_ms=500)):
            tree._evict_device_leaf(n208, {FULL: 0})
        self.assertEqual(tree.dropped, [n208.id])

    def test_holder_source_reports_mixed_for_a_shared_lock(self):
        tree = _Tree()
        n208, n211 = self._mixed(tree)
        self.assertEqual(UnifiedRadixCache._weg2_hostlock_source(tree, n211.id), "mixed")

    def test_holder_source_own_write_only_is_l3_write(self):
        tree = _Tree()
        _n208, n211 = _death2(tree, acked=False)
        self.assertEqual(UnifiedRadixCache._weg2_hostlock_source(tree, n211.id), "l3_write")

    def test_holder_source_write_plus_prefetch_is_mixed(self):
        tree = _Tree()
        _n208, n211 = _death2(tree, acked=False)
        tree.ongoing_prefetch["r"] = (n211,)
        self.assertEqual(UnifiedRadixCache._weg2_hostlock_source(tree, n211.id), "mixed")


if __name__ == "__main__":
    unittest.main()
