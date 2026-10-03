# SPDX-License-Identifier: Apache-2.0
"""NF rc12q (dkrnfh91dprbar1dauer09271603, 21d7e3b188).

UD, PP2 16:28:19Z: EVICT-FRONTIER-CENSUS request=16448 on_frontier=1856 behind_device_child=249856,
UNDER-DELIVERED 6016 -> OOM. The one frontier leaf (node 738, 1856 tokens) refused its write_back
backup (#1421 parent_unbacked -> 739 parent_unbacked -> 734 arena_claim: the arena had no free slot),
so it stayed on the device and every node behind it with it. On the local-PP floor the leaf is now
DROPPED (write_through's delete), and the chain peels.

EB, 16:26:57 weg2-14-52: D answered the stream with W88 Weg2StoreLoadNotProgressing before any
content; the front committed a 200 with no text. Now: 503, as its non-stream twin weg2-14-53 got.
"""

import os
import types
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import pp_slot_fidelity as SF  # noqa: E402

FULL = 0


def _env(**kv):
    return mock.patch.dict(os.environ, {k: str(v) for k, v in kv.items()})


class _Node:
    def __init__(self, nid, parent, tokens, backuped=False):
        self.id = nid
        self.parent = parent
        self.children = {}
        self.key = list(range(tokens))
        self.value = list(range(tokens))
        self.backuped = backuped
        self.evicted = False
        if parent is not None:
            parent.children[nid] = self


def _tree(local_pp=True):
    from sglang.srt.mem_cache import unified_radix_cache as U

    class T:
        _evict_device_leaf = U.UnifiedRadixCache._evict_device_leaf
        _ud_drop_unbacked_leaf = U.UnifiedRadixCache._ud_drop_unbacked_leaf

    t = T()
    root = _Node("root", None, 0, backuped=True)
    t.root_node = root
    t.cache_controller = types.SimpleNamespace(write_policy="write_back")
    t.ongoing_write_through = {}
    t._components_tuple = (types.SimpleNamespace(component_type=FULL),)
    t.evictable_device_leaves = set()
    t.dropped, t.demoted = [], []
    setattr(t, SF.FLOOR_LOCAL_PP_ATTR, local_pp)

    def is_leaf(n):
        return n is not root and not n.evicted and not n.children

    t._is_device_leaf = is_leaf
    t.write_backup = lambda n, write_back=False, kv_only_if_mamba_refused=False: 0  # arena full
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
        upd(n.parent)

    t._evict_to_host = evict_to_host
    t.eviction_strategy = types.SimpleNamespace(get_priority=lambda n: 0)
    # the NF chain: 737 (backed) <- 734 (arena_claim refused) <- 739 <- 738 (the frontier leaf)
    n737 = _Node(737, root, 245664, backuped=True)
    n734 = _Node(734, n737, 4096)
    n739 = _Node(739, n734, 4096)
    n738 = _Node(738, n739, 1856)
    t.evictable_device_leaves = {n738}
    return t, (n737, n734, n739, n738)


def _peel(t, request):
    from sglang.srt.mem_cache.unified_cache_components.full_component import FullComponent

    comp = types.SimpleNamespace(cache=t, component_type=FULL)
    tracker = {FULL: 0}
    FullComponent._peel(comp, request, tracker)
    return tracker[FULL]


class UnbackedDrop(unittest.TestCase):
    def test_nf_chain_peels_when_the_unbackable_leaves_are_dropped(self):
        with _env(**{SF.ENV: 1}):
            t, _ = _tree()
            with self.assertLogs(SF.logger, level="WARNING") as cap:
                got = _peel(t, 6016)
        self.assertGreaterEqual(got, 6016)
        self.assertEqual(t.dropped[:3], [738, 739, 734])
        self.assertTrue(any("EVICT-UNBACKED-DROP node=738" in m for m in cap.output))

    def test_the_metal_behaviour_switch_off_delivers_nothing(self):
        with _env(**{SF.ENV: 0}):
            t, _ = _tree()
            self.assertEqual(_peel(t, 6016), 0)
            self.assertEqual(t.dropped, [])

    def test_a_group_floor_never_drops(self):
        with _env(**{SF.ENV: 1}):
            t, _ = _tree(local_pp=False)
            self.assertEqual(_peel(t, 6016), 0)

    def test_guards(self):
        with _env(**{SF.ENV: 1}):
            t, (_, n734, n739, n738) = _tree()
            self.assertFalse(SF.unbacked_drop_allowed(t, n739), "#841: a node with children")
            t.ongoing_write_through[738] = object()
            self.assertFalse(SF.unbacked_drop_allowed(t, n738), "write-through in flight")
            del t.ongoing_write_through[738]
            self.assertTrue(SF.unbacked_drop_allowed(t, n738))


class ErrorBeforeContent(unittest.TestCase):
    W88 = (b'event: message_start\ndata: {"type":"message_start"}\n\n'
           b'event: error\ndata: {"type":"error","error":{"message":"W88 Weg2StoreLoadNotProgressing '
           b'rid=weg2-14-52 arm=host_pool_shortfall"}}\n\n')

    def test_anthropic_error_before_content_is_named(self):
        from sglang.srt.weg2 import front as F

        self.assertIn("W88", F.inband_error_before_content(self.W88, "/v1/messages"))
        with_content = self.W88.replace(b"event: error", b"event: content_block_start")
        self.assertIsNone(F.inband_error_before_content(with_content, "/v1/messages"))
        self.assertIsNone(F.inband_error_before_content(
            self.W88 + b"event: content_block_delta\n", "/v1/messages"), "content already sent")

    def test_openai_first_chunk_error(self):
        from sglang.srt.weg2 import front as F

        self.assertIsNotNone(F.inband_error_before_content(b'data: {"error": {"message": "W88"}}\n\n',
                                                           "/v1/chat/completions"))
        self.assertIsNone(F.inband_error_before_content(b'data: {"choices": [{"delta": {}}]}\n\n',
                                                        "/v1/chat/completions"))

    def test_switch_and_wiring(self):
        from sglang.srt.weg2 import front as F

        with _env(SGLANG_WEG2_LEG2_ERROR_BEFORE_CONTENT=0):
            self.assertFalse(F._eb_enabled())
        self.assertTrue(F._eb_enabled({}))
        src = open(F.__file__).read()
        i = src.index("_eb = inband_error_before_content(first_chunk, request.path)")
        self.assertLess(i, src.index("await resp.prepare(request)", i), "decided before the commit")
        blk = src[i:i + 2600]
        self.assertIn("if _eb_reroute_enabled():", blk)
        self.assertIn("return await self._requeue_after_x_refusal(", blk)
        self.assertIn("status=503", blk)
        self.assertTrue(F._eb_reroute_enabled({}))
        with _env(SGLANG_WEG2_LEG2_ERROR_REROUTE=0):
            self.assertFalse(F._eb_reroute_enabled())


if __name__ == "__main__":
    unittest.main()
