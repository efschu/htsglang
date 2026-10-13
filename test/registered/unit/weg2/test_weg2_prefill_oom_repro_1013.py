# SPDX-License-Identifier: Apache-2.0
"""PREFILL-OOM fix 43, the two REPRO tests (NF 13.10.2026 11:59:14Z; P.log 1214208-1214253 of boot ..._b60520f01d_1013_073910).

Depends on nothing the fix adds except the attribute under test, so it runs RED on 0ea6d17efd (int24f6):
(A) ``estimate_payable`` counts the un-backable leaf [6193] (135872 rows) payable although the peel demotes its two
host-locked device children first and then cannot drop it -- 150656 instead of 320, the 320 rows the peel really delivered;
(B) the allocation raises a bare ``RuntimeError`` (no ``mem_cache.common.PrefillOutOfPages`` with the terms).
GREEN on the fix. Hermetic: stubs only, no CUDA.
"""

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.mem_cache import common as mc  # noqa: E402
from sglang.srt.weg2 import pp_room_vote as PR  # noqa: E402


# ---------------------------------------------------------------------------
# (A) the walk follows the peel's order
# ---------------------------------------------------------------------------


class _CD:
    def __init__(self, n=0, lock=0, host_lock=0):
        self.value = torch.arange(n) if n else None
        self.lock_ref = lock
        self.host_lock_ref = host_lock


class _Node:
    def __init__(self, parent, *, n=0, backuped=False, evicted=False, lock=0, host_lock=0):
        self.id = id(self)
        self.parent = parent
        self.children = {}
        self.backuped = backuped
        self.evicted = evicted
        self.component_data = [_CD(n, lock, host_lock), _CD(), _CD()]
        if parent is not None:
            parent.children[self.id] = self


def _incident(child_host_lock=1, with_children=True):
    """[6182] 3008 locked -> [6190] 14464 backed -> [6193] 135872 UN-backed -> [6194] 192, [6195] 128 (backed by the tail
    write-through at 11:59:13, device nodes, holding the end anchor -> host-locked)."""
    root = _Node(None)
    a = _Node(root, n=3008, lock=1)
    b = _Node(a, n=14464, backuped=True)
    leaf = _Node(b, n=135872, backuped=False)
    kids = []
    if with_children:
        kids = [_Node(leaf, n=192, backuped=True, host_lock=child_host_lock),
                _Node(leaf, n=128, backuped=True, host_lock=child_host_lock)]
    return root, leaf, kids


def _tree(root, ongoing=None):
    return types.SimpleNamespace(
        root_node=root, ongoing_write_through=ongoing or {},
        cache_controller=types.SimpleNamespace(write_policy="write_back", mem_pool_host=None),
    )


def _payable(root, *, local_floor=True, host_free=0, ongoing=None):
    return PR.estimate_payable(_tree(root, ongoing), local_floor=local_floor, host_free=host_free)


class WalkFollowsThePeelTest(unittest.TestCase):
    def test_6193_is_not_payable_when_the_children_it_needs_gone_are_host_locked(self):
        """THE REPRO. The peel demotes [6194]/[6195] (320 rows, exactly what it delivered), then cannot clear them from the
        host (host-locked) -> [6193] cannot be dropped -> nothing behind it is paid. RED: 150656."""
        root, _, _ = _incident(child_host_lock=1)
        self.assertEqual(_payable(root), 320)

    def test_the_same_leaf_is_payable_when_the_demoted_children_can_be_cleared(self):
        root, _, _ = _incident(child_host_lock=0)
        self.assertEqual(_payable(root), 14464 + 135872 + 192 + 128)

    def test_a_childless_unbacked_leaf_is_dropped(self):
        root, _, _ = _incident(with_children=False)
        self.assertEqual(_payable(root), 14464 + 135872)

    def test_write_through_in_flight_or_no_local_floor_blocks_the_drop(self):
        root, leaf, _ = _incident(child_host_lock=0)
        self.assertEqual(_payable(root, ongoing={leaf.id: 1}), 320)
        self.assertEqual(_payable(root, local_floor=False), 320)

    def test_a_host_only_child_that_is_host_locked_blocks_the_drop_as_before(self):
        root = _Node(None)
        leaf = _Node(root, n=1000, backuped=False)
        _Node(leaf, backuped=True, evicted=True, host_lock=1)
        self.assertEqual(_payable(root), 0)
        root = _Node(None)
        leaf = _Node(root, n=1000, backuped=False)
        _Node(leaf, backuped=True, evicted=True)
        self.assertEqual(_payable(root), 1000)

    def test_a_demoted_chain_is_unchanged(self):
        """Backed nodes are demoted, no UD involved: the old arithmetic."""
        root = _Node(None)
        a = _Node(root, n=4096, backuped=True)
        _Node(a, n=4096, backuped=True)
        self.assertEqual(_payable(root), 8192)


# ---------------------------------------------------------------------------
# (B) the typed exception
# ---------------------------------------------------------------------------


class _Alloc:
    page_size = 64

    def __init__(self, free=13568):
        self._free = free

    def available_size(self):
        return self._free

    def alloc_extend(self, *a, **k):
        return None

    def backup_state(self):
        return "state"


def _alloc_tree(alloc, evictable=150336):
    tc = mock.MagicMock()
    tc.is_chunk_cache.return_value = False
    tc.token_to_kv_pool_allocator = alloc
    tc.uniform_avail_floor = None
    tc.uniform_admitted_since_floor = 0
    tc.evictable_size.return_value = evictable
    tc.full_evictable_size.return_value = evictable
    tc.available_and_evictable_str.return_value = f"avail={alloc._free} evictable={evictable}"
    return tc


class TypedExceptionTest(unittest.TestCase):
    def setUp(self):
        self._saved = list(mc._extend_relief_providers)
        mc.clear_extend_relief_providers()

    def tearDown(self):
        mc.clear_extend_relief_providers()
        mc._extend_relief_providers.extend(self._saved)

    def test_the_raise_names_its_terms_and_stays_a_runtime_error(self):
        t = torch.tensor([0], dtype=torch.int64)
        with self.assertRaises(mc.PrefillOutOfPages) as cm:
            mc.alloc_paged_token_slots_extend(
                tree_cache=_alloc_tree(_Alloc()), prefix_lens=t, prefix_lens_cpu=t, seq_lens=t, seq_lens_cpu=t,
                last_loc=t, extend_num_tokens=16384)
        e = cm.exception
        self.assertIsInstance(e, RuntimeError)
        self.assertIn("Prefill out of memory", str(e))
        self.assertEqual(e.need, 16384 + 64)
        self.assertEqual(e.free, 13568)
        self.assertEqual(e.evictable_reported, 150336)
        self.assertGreaterEqual(e.evict_asked, 0)


# ---------------------------------------------------------------------------
# (C) the load-back entrance: AgreedRoomShort (NF 13.10.2026 18:30:21Z, int24f6 0ea6d17efd, P.log 1012731-1012800)
# ---------------------------------------------------------------------------


class _LockRes:
    delta = 0

    def __init__(self, name, log):
        self.name, self.log = name, log

    def to_dec_params(self):
        return self.name


def _incident_load_back_tree(*, agreed=191744, avail=92032, delivered=64, kv_tokens=92288, evictable=99712):
    """The real ``UnifiedRadixCache.load_back`` on a stub ``self``: rid weg2-302-1956 loads 92288 rows back, the stages agreed
    191744 (= avail + every evictable row), this pool holds 92032 + 99712 evictable of which the peel pays 64."""
    from sglang.srt.mem_cache.unified_cache_components.tree_component import BASE_COMPONENT_TYPE
    from sglang.srt.weg2 import pp_slot_fidelity as SF

    log = []
    state = {"avail": avail}
    kv_xfer = types.SimpleNamespace(host_indices=list(range(kv_tokens)), nodes_to_load=())
    base = types.SimpleNamespace(build_hicache_transfers=lambda node, phase, **k: [kv_xfer])

    def evict(params):
        got = min(delivered, params.num_tokens)
        state["avail"] += got
        return types.SimpleNamespace(num_tokens_evicted=got)

    tree = types.SimpleNamespace(
        cache_controller=object(), components={BASE_COMPONENT_TYPE: base}, _components_tuple=(),
        load_back_threshold=0, uniform_avail_floor=avail, weg2_pp_room_cap=agreed,
        token_to_kv_pool_allocator=types.SimpleNamespace(available_size=lambda: state["avail"]),
        evictable_size=lambda: evictable, evict=evict,
        inc_host_lock_ref=lambda n: _LockRes("host", log), inc_lock_ref=lambda n: _LockRes("dev", log),
        dec_lock_ref=lambda n, p: log.append(("dec_lock_ref", p)),
        dec_host_lock_ref=lambda n, p: log.append(("dec_host_lock_ref", p)),
        _1424_verify_load_chain=lambda *a, **k: None, _build_sidecar_transfers=lambda *a, **k: [],
    )
    setattr(tree, SF.FLOOR_LOCAL_PP_ATTR, True)
    return tree, log


class LoadBackAgreedRoomShortTest(unittest.TestCase):
    def setUp(self):
        self._env = os.environ.pop("SGLANG_WEG2_PP_SLOT_FIDELITY", None)

    def tearDown(self):
        if self._env is not None:
            os.environ["SGLANG_WEG2_PP_SLOT_FIDELITY"] = self._env

    def _run(self, **kw):
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        tree, log = _incident_load_back_tree(**kw)
        req = types.SimpleNamespace(rid="weg2-302-1956")
        node = types.SimpleNamespace(id=1)
        try:
            return UnifiedRadixCache.load_back(tree, node, None, req=req), log
        except Exception as e:  # noqa: BLE001
            return e, log

    def test_the_incident_raises_with_its_terms_and_leaves_no_lock_behind(self):
        """THE REPRO. RED on 0ea6d17efd: the exception carries no terms (no .kv_tokens) and load_back left both lock refs
        taken -- a rank that does not die (the retreat) would leak them. GREEN: terms + both refs unwound."""
        from sglang.srt.weg2.pp_slot_fidelity import AgreedRoomShort

        exc, log = self._run()
        self.assertIsInstance(exc, AgreedRoomShort)
        self.assertIsInstance(exc, RuntimeError)
        self.assertIn("PR AGREED-ROOM SHORT rid=weg2-302-1956 kv_tokens=92288 agreed=191744 avail=92032 evictable=99712 "
                      "evicted=64 avail_after=92096", str(exc))
        self.assertEqual((exc.kv_tokens, exc.agreed, exc.avail, exc.evictable, exc.evicted, exc.avail_after),
                         (92288, 191744, 92032, 99712, 64, 92096))
        self.assertEqual(sorted(x for x in log if isinstance(x, tuple)),
                         [("dec_host_lock_ref", "host"), ("dec_lock_ref", "dev")])

    def test_a_peel_that_delivers_loads_in_this_pass_as_before(self):
        res, log = self._run(delivered=256)
        self.assertIsInstance(res, Exception)  # past the room check the stub has no cache_controller.load: that is the proof
        from sglang.srt.weg2.pp_slot_fidelity import AgreedRoomShort

        self.assertNotIsInstance(res, AgreedRoomShort)


if __name__ == "__main__":
    unittest.main()
