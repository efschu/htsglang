"""#1481: the mamba pool's interior-node eviction skips a marked END-ANCHOR
node while its state is un-backed, and evicts it once backed up.

END-ANCHOR-GUARD-1024 (#231, SOLL R7; 22-194): "backed up" meant the KV only -- an END anchor whose node went
KV-only (S108 turn end, P-FUND) still had its mamba state on the device alone and lost it here. Such a node is
now held in the eviction's pass 1 and tombstoned in pass 2 only when nothing else pays; with a host copy it is
evictable as before (pdflip/end_anchor_guard.py)."""
import os
import types
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.srt.mem_cache.unified_cache_components.mamba_component import MambaComponent


def _node(nid, end_anchor=False, backuped=False, host_value=None):
    return types.SimpleNamespace(id=nid, backuped=backuped, _pdflip_end_anchor=end_anchor,
                                 component_data={"mamba": types.SimpleNamespace(value=object(), host_value=host_value)})


def _run(nodes, request=1):
    """Drive the interior branch over `nodes` (LRU order) with stubbed cache/LRU."""
    evicted = []
    order = list(nodes)
    lru = types.SimpleNamespace(
        get_lru_no_lock=lambda: order[0] if order else None,
        get_prev_no_lock=lambda x: order[order.index(x) + 1] if order.index(x) + 1 < len(order) else None,
        in_list=lambda x: x is not None and x in order,
    )
    cache = types.SimpleNamespace(
        lru_lists={"mamba": lru},
        evictable_device_leaves=[],
        _evict_component_and_detach_lru=lambda x, comp, target=None, tracker=None: (evicted.append(x.id), tracker.__setitem__("mamba", tracker["mamba"] + 1)),
        _cascade_evict=lambda x, comp, tracker: None,
    )
    comp = types.SimpleNamespace(cache=cache, component_type="mamba")
    tracker = {"mamba": 0}
    MambaComponent.drive_eviction(comp, types.SimpleNamespace(mamba_num=request), tracker)
    return evicted


class Test1481(unittest.TestCase):
    def test_unbacked_end_anchor_is_skipped(self):
        self.assertEqual(_run([_node(1, end_anchor=True), _node(2)]), [2])

    def test_kv_backed_end_anchor_without_copy_is_held_while_others_pay(self):
        # END-ANCHOR-GUARD-1024: was test_backed_end_anchor_is_evictable (== [1]) -- the 22-194 gap, pinned
        self.assertEqual(_run([_node(1, end_anchor=True, backuped=True), _node(2)]), [2])

    def test_kv_backed_end_anchor_without_copy_goes_in_pass2_when_it_must(self):
        self.assertEqual(_run([_node(1, end_anchor=True, backuped=True), _node(2)], request=2), [2, 1])

    def test_end_anchor_with_host_copy_is_evictable(self):
        self.assertEqual(_run([_node(1, end_anchor=True, host_value=object()), _node(2)]), [1])

    def test_plain_interior_node_evicts_first(self):
        self.assertEqual(_run([_node(1), _node(2, end_anchor=True)]), [1])


if __name__ == "__main__":
    unittest.main()
