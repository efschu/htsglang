"""rc12 Dauerlauf, second death (boot dkrnfh91bar1dauer09262302, 530fb713ce):
a #1417 prefetch pin outlives the sleep flush and is released on the dead tree.

MEASURED (D.log, all three ranks identical):

    23:11:26-29 #1423 INSERT-PLACED req=pdflip-0-7 / -0-8 / -1-1 / -2-1
                reg_node=59 ... deepest=60 / 62 / 63 / 64   (prefetch inserts,
                pinned by #1417 for the held requests)
    23:12:45    PDFLIP-D-PARK park_running ... queued-behind=['pdflip-1-9',
                'pdflip-0-7', 'pdflip-0-8', ...]; FlushCacheReqInput;
                #1427 ARENA-REF RESET-RELEASE ... nodes=8 skipped_in_use=5;
                Cache flushed successfully!  (x2, the sleep's two flushes)
    23:12:47    Sanity check FAILED (2 violations across 7 nodes):
                  H-leaf extra: [62, 64, 60, 63]
                  4 stale nodes in host_leaves: [62, 64, 60, 63]

`_reset_full` builds a new root and fresh leaf sets but kept
`_prefetch_span_pins`. The held request's admission then called
`pop_prefetch_loaded_tokens` -> `_unpin_prefetched_span` ->
`dec_host_lock_ref(old node)` -> `_update_evictable_leaf_sets(old node)`,
which filed the dead tree's host leaves (60, 62, 63, 64 -- not 61, the split
node above 62-64) into the new tree's set. The OOM before it only made D slow
enough to park; the flush with held, pinned prefetches is the defect.

Hermetic: the #1417b fixture (real UnifiedRadixCache FULL + MAMBA on CPU,
real insert, pin, commit, reset and sanity_check).
"""

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import unittest

from flliper.test.test_utils import CustomTestCase

from test_prefetch_pin_host_lru_sanity_1417b import MAMBA, _metal_form
from test_prefetch_pin_host_lru_sanity_1417b import _tree as _tree_1417b


def _tree():
    return _tree_1417b()


def _flush(cache):
    """The sleep's flush on this fixture: the tree reset. The stub controller
    has no host pools to clear, so it is detached for the reset itself."""
    cc, cache.cache_controller = cache.cache_controller, None
    try:
        cache.reset()
    finally:
        cache.cache_controller = cc


class PrefetchPinAcrossReset(CustomTestCase):
    def test_admission_after_the_flush_leaves_the_new_tree_clean(self):
        """RED before the fix: '... stale nodes in host_leaves: [...]'."""
        cache = _tree()
        _metal_form(cache)
        _flush(cache)
        cache.pop_prefetch_loaded_tokens("pdflip-16-31")
        cache.pop_prefetch_loaded_tokens("pdflip-16-38")
        self.assertEqual(cache.evictable_host_leaves, set())
        cache.sanity_check()

    def test_the_flush_takes_the_pins_with_the_tree(self):
        cache = _tree()
        n249, n250 = _metal_form(cache)
        _flush(cache)
        self.assertFalse(getattr(cache, "_prefetch_span_pins", {}))
        self.assertEqual(n249.component_data[MAMBA].host_lock_ref, 0)
        self.assertEqual(n250.component_data[MAMBA].host_lock_ref, 0)

    def test_without_a_flush_the_pins_still_hold(self):
        """The fix does not weaken #1417: pins live until admission."""
        cache = _tree()
        n249, _ = _metal_form(cache)
        self.assertEqual(n249.component_data[MAMBA].host_lock_ref, 1)
        cache.sanity_check()


if __name__ == "__main__":
    unittest.main()
