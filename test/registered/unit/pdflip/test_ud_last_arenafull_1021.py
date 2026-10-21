# SPDX-License-Identifier: Apache-2.0
"""UD-LAST (NF rc27 reboot 1021_023225, P PP0 02:37:15) -- pdflip/ud_last.py.

Metal (P.log boot_weg2_dkrnfint4h6ablxcw3spmlrhlvwflbar1dauer10210232_893d09fa3f_1021_023225):
pdflip-0-22 computed [170240, 184512); the arena (6485/6485 pinned) refused every backup (#1421 arena_claim,
#1427 ARENA-DROP freed=0). The SF LOADBACK-ROOM peel for pdflip-0-19 (need 113408-85760 = 27648) then dropped the
fresh tail in LRU order -- EVICT-UNBACKED-DROP node=16 1920, 19 4096, 18 4096, 17 4096, 15 64 = 14272 -- and demoted
node 9 (151360): evicted=165632, exactly the metal figure, with 78016 evictable tokens left. pdflip-1-23 recomputed
the 14272 (lost=14272).

The tree below is that chain plus one backed leaf elsewhere (a stand-in for the evictable rest). With UD-LAST the
peel holds the un-backed tail and pays from the backed leaf: nothing fresh is dropped. When the backed leaves cannot
pay, pass 2 drops in the old order (the rc12q liveness), only as much as needed.
"""

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.pdflip import pp_slot_fidelity as SF  # noqa: E402
from flliper.srt.pdflip import ud_last as UDL  # noqa: E402

FULL = 0


def _env(**kv):
    return mock.patch.dict(os.environ, {k: str(v) for k, v in kv.items()})


class _Node:
    def __init__(self, nid, parent, tokens, backuped=False, prio=0.0):
        self.id = nid
        self.parent = parent
        self.children = {}
        self.key = list(range(tokens))
        self.value = list(range(tokens))
        self.backuped = backuped
        self.evicted = False
        self.prio = prio
        if parent is not None:
            parent.children[nid] = self

    def __lt__(self, other):          # heap ties: deterministic, like TreeNode
        return str(self.id) < str(other.id)


def _fake_tree(local_pp=True):
    from flliper.srt.mem_cache import unified_radix_cache as U

    class T:
        _evict_device_leaf = U.UnifiedRadixCache._evict_device_leaf
        _ud_drop_unbacked_leaf = U.UnifiedRadixCache._ud_drop_unbacked_leaf

    t = T()
    root = _Node("root", None, 0, backuped=True)
    t.root_node = root
    t.cache_controller = types.SimpleNamespace(write_policy="write_back")
    t.ongoing_write_through = {}
    t._components_tuple = (types.SimpleNamespace(component_type=FULL),)
    t.dropped, t.demoted, t.backup_tries = [], [], []
    setattr(t, SF.FLOOR_LOCAL_PP_ATTR, local_pp)

    def is_leaf(n):
        return n is not root and not n.evicted and not n.children

    t._is_device_leaf = is_leaf

    def write_backup(n, write_back=False, kv_only_if_mamba_refused=False):
        t.backup_tries.append(n.id)
        return 0                       # arena full: #1421 arena_claim

    t.write_backup = write_backup
    t.writing_check = lambda write_back=False: None
    t._record_remove_event = lambda n, medium=None: None

    def evict_comp(n, comp, target=None, tracker=None):
        tracker[FULL] += len(n.value)

    t._evict_component_and_detach_lru = evict_comp

    def remove(n):
        n.parent.children.pop(n.id, None)
        t.dropped.append(n.id)

    t._remove_leaf_from_parent = remove

    def upd(n):
        if is_leaf(n):
            t.evictable_device_leaves.add(n)
        else:
            t.evictable_device_leaves.discard(n)

    t._update_evictable_leaf_sets = upd
    t._iteratively_delete_tombstone_leaf = lambda n, tracker: None

    def evict_to_host(n, tracker):
        tracker[FULL] += len(n.value)
        n.evicted = True
        t.demoted.append(n.id)
        t.evictable_device_leaves.discard(n)
        upd(n.parent)

    t._evict_to_host = evict_to_host
    t.eviction_strategy = types.SimpleNamespace(get_priority=lambda n: n.prio)
    t.evictable_device_leaves = set()
    return t, root


def _metal_tree(backed_rest=40000, local_pp=True):
    """root <- sys(18880, backed) <- 9(151360, backed) <- 15(64) <- 17(4096) <- 18(4096) <- 19(4096) <- 16(1920),
    15..16 = pdflip-0-22's fresh, un-backed tail (backup refused); plus a backed leaf 11 that is more recently used
    than the tail (the LRU order of the metal peel reached the tail first)."""
    t, root = _fake_tree(local_pp)
    sys_ = _Node("sys", root, 18880, backuped=True, prio=0.0)
    n9 = _Node(9, sys_, 151360, backuped=True, prio=1.0)
    n15 = _Node(15, n9, 64, prio=2.0)
    n17 = _Node(17, n15, 4096, prio=2.0)
    n18 = _Node(18, n17, 4096, prio=2.0)
    n19 = _Node(19, n18, 4096, prio=2.0)
    n16 = _Node(16, n19, 1920, prio=2.0)
    nodes = {"sys": sys_, 9: n9, 15: n15, 17: n17, 18: n18, 19: n19, 16: n16}
    leaves = {n16}
    if backed_rest:
        n11 = _Node(11, sys_, backed_rest, backuped=True, prio=5.0)
        nodes[11] = n11
        leaves.add(n11)
    t.evictable_device_leaves = leaves
    return t, nodes


def _peel(t, request):
    from flliper.srt.mem_cache.unified_cache_components.full_component import FullComponent

    comp = types.SimpleNamespace(cache=t, component_type=FULL)
    tracker = {FULL: 0}
    FullComponent._peel(comp, request, tracker)
    return tracker[FULL]


TAIL = [16, 19, 18, 17, 15]   # the metal drop order (EVICT-UNBACKED-DROP n=1..5)
NEED = 113408 - 85760         # SF LOADBACK-ROOM pdflip-0-19: kv_tokens - floor


class MetalCase(unittest.TestCase):
    def test_metal_one_pass_reproduced_with_the_switch_off(self):
        """UD-LAST off = the rc27 peel: the fresh tail is dropped, then node 9 demoted -- evicted 165632."""
        with _env(**{SF.ENV: 1, UDL.ENV: 0}):
            t, _ = _metal_tree()
            got = _peel(t, NEED)
        self.assertEqual(t.dropped, TAIL)
        self.assertEqual(t.backup_tries, TAIL)
        self.assertEqual(sum(len(range(n)) for n in (1920, 4096, 4096, 4096, 64)), 14272)
        self.assertEqual(got, 165632, "the metal figure: SF LOADBACK-ROOM ... evicted=165632")
        self.assertEqual(t.demoted, [9])

    def test_unbacked_fresh_tail_is_held_and_backed_leaf_pays(self):
        """DANGER DIRECTION 'unbacked fresh node dropped': with UD-LAST on, no node of the tail is dropped."""
        with _env(**{SF.ENV: 1}):
            t, nodes = _metal_tree()
            with self.assertLogs(UDL.logger, level="INFO") as cap:
                got = _peel(t, NEED)
        self.assertEqual(t.dropped, [], "an un-backed fresh node was dropped although backed leaves could pay")
        self.assertGreaterEqual(got, NEED)
        self.assertEqual(t.demoted, [11])
        self.assertIn(nodes[16], t.evictable_device_leaves, "the held leaf stays evictable for a later peel/backup")
        self.assertFalse(nodes[16].evicted)
        self.assertTrue(any("UD-LAST HELD" in m and "held_tokens=1920" in m for m in cap.output), cap.output)

    def test_backed_leaves_short_then_pass_two_drops_old_order_only_as_needed(self):
        """rc12q liveness: when only the drop can pay, pass 2 drops in the old order and stops at the request."""
        with _env(**{SF.ENV: 1}):
            t, _ = _metal_tree(backed_rest=10000)
            with self.assertLogs(UDL.logger, level="WARNING") as cap:
                got = _peel(t, 10000 + 6000)
        self.assertEqual(t.demoted, [11], "the backed leaf goes first")
        self.assertEqual(t.dropped, [16, 19], "only as many held leaves as the request needs, old order")
        self.assertEqual(t.backup_tries, [16, 19], "pass 2 claims no second time for a leaf pass 1 held")
        self.assertGreaterEqual(got, 16000)
        self.assertTrue(any("UD-LAST DROP-AFTER" in m for m in cap.output), cap.output)

    def test_only_unbacked_chain_still_peels_to_the_backed_parent(self):
        """No backed leaf at all (the rc12q shape): pass 2 = the old peel, the chain peels down to node 9."""
        with _env(**{SF.ENV: 1}):
            t, _ = _metal_tree(backed_rest=0)
            got = _peel(t, NEED)
        self.assertEqual(t.dropped, TAIL)
        self.assertEqual(t.demoted, [9])
        self.assertEqual(got, 165632)

    def test_group_floor_unchanged(self):
        """Off the local-PP floor UD never dropped and UD-LAST never holds (no rank-local verdict on a TP group)."""
        with _env(**{SF.ENV: 1}):
            t, _ = _metal_tree(backed_rest=0, local_pp=False)
            got = _peel(t, NEED)
        self.assertEqual(t.dropped, [])
        self.assertEqual(got, 0)
        self.assertIsNone(getattr(t, UDL.HOLD_ATTR, None))

    def test_hold_list_closed_after_the_peel_and_nested_peels_restore(self):
        with _env(**{SF.ENV: 1}):
            t, nodes = _metal_tree()
            outer = UDL.begin(t)
            self.assertEqual(outer, (None, None))
            outer_list = getattr(t, UDL.HOLD_ATTR)
            _peel(t, NEED)                       # an inner peel opens and closes its own list
            self.assertIs(getattr(t, UDL.HOLD_ATTR), outer_list)
            self.assertEqual(outer_list, [], "the inner peel's holds never leak into the outer list")
            self.assertIsNone(getattr(t, UDL.REFUSED_ATTR))
            UDL.end(t, outer)
        self.assertIsNone(getattr(t, UDL.HOLD_ATTR))
        self.assertIsNone(getattr(t, UDL.REFUSED_ATTR))
        # outside any peel a direct _evict_device_leaf drops as before (callers other than the full peel)
        with _env(**{SF.ENV: 1}):
            t, nodes = _metal_tree()
            tracker = {FULL: 0}
            t._evict_device_leaf(nodes[16], tracker)
        self.assertEqual(t.dropped, [16])

    def test_switch_default_on(self):
        self.assertTrue(UDL.on({}))
        self.assertFalse(UDL.on({UDL.ENV: "0"}))


if __name__ == "__main__":
    unittest.main()
