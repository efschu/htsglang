"""#257 (b): a short read keeps only what a recurrent anchor can resume.

Vision boot 0928 (P PP0 06:20:28, weg2-4-27): the read ended at 14016 of 52032
host tokens; the first recurrent (mamba) anchor of that prefix sat at 16384.
The 14016 tokens were loaded to the device anyway (WEG2-LOAD-DEVICE 43 ms) and
inserted -- and the prefill started at 0, because a hybrid model resumes only
from an anchor. The reap now asks the store for the deepest anchor inside the
pages that landed (the same mamba presence question the probe asked,
``_presence_pool_transfers``), rides it on the existing packed MIN, and cuts
the claim there: below the first anchor nothing is loaded, above it only up to
the anchor.

Driven through the real ``check_prefetch_progress`` on the #937/#1157 harness
(a real CPU tree, a real host pool, a real ``PrefetchOperation``)."""

import importlib.util
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache.hicache_phase_binding import binding_state  # noqa: E402
from sglang.srt.mem_cache.hicache_storage import PoolName  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_t_1157", os.path.join(os.path.dirname(__file__), "test_1157_reaper_prices_requested_span.py")
)
h1157 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h1157)

REQ = h1157.REAP_REQ
SPAN = h1157.REAP_TOKENS      # 16 pages of 1 token
READ = 10                     # the read ended here (page 219 of 813 on the metal)


class _Store:
    """The store's mamba presence over the pages that landed."""

    def __init__(self, anchor_pages):
        self.anchor_pages = anchor_pages
        self.asked = []

    def batch_exists_v2(self, keys, pool_transfers=None, extra_info=None):
        self.asked.append(len(keys))
        hits = {PoolName.MAMBA: self.anchor_pages} if self.anchor_pages else {}
        return types.SimpleNamespace(kv_hit_pages=len(keys), extra_pool_hit_pages=hits)


def _short_read(anchor_pages):
    cache, op = h1157._reap_scenario(probed=False)
    op.hash_value = [f"h{i}" for i in range(SPAN)]
    op.probed_hit_tokens = SPAN
    op.increment(READ)
    op.mark_terminate()           # the read broke at READ (a page neither in L2 nor L3)
    store = _Store(anchor_pages)
    cache.cache_controller.storage_backend = store
    cache.cache_controller._presence_pool_transfers = lambda: [types.SimpleNamespace(name=PoolName.MAMBA)]
    return cache, store


class BelowTheFirstAnchorNothingIsLoaded(CustomTestCase):
    def setUp(self):
        self.addCleanup(binding_state().reset)

    def test_weg2_4_27_a_read_below_the_first_anchor_loads_nothing(self):
        """RED on a9e3a842ae: the 10 read tokens are claimed and loaded
        (the metal's 14016 below the anchor at 16384). GREEN: named, cut to 0."""
        cache, store = _short_read(anchor_pages=0)
        with self.assertLogs("sglang.srt.mem_cache.unified_radix_cache", "WARNING") as cm:
            cache.check_prefetch_progress(REQ)
        self.assertEqual(int(cache.prefetch_loaded_tokens_by_reqid[REQ]), 0)
        self.assertIn("#257 PREFETCH BELOW-ANCHOR", "\n".join(cm.output))
        self.assertEqual(store.asked, [READ])

    def test_a_read_past_an_anchor_keeps_it_and_drops_the_rest(self):
        cache, _ = _short_read(anchor_pages=6)
        cache.check_prefetch_progress(REQ)
        self.assertEqual(int(cache.prefetch_loaded_tokens_by_reqid[REQ]), 6)

    def test_a_full_read_asks_nothing(self):
        cache, op = h1157._reap_scenario(probed=True)
        store = _Store(anchor_pages=0)
        cache.cache_controller.storage_backend = store
        cache.cache_controller._presence_pool_transfers = lambda: [types.SimpleNamespace(name=PoolName.MAMBA)]
        cache.check_prefetch_progress(REQ)
        self.assertEqual(int(cache.prefetch_loaded_tokens_by_reqid[REQ]), SPAN)
        self.assertEqual(store.asked, [])

    def test_no_mamba_component_no_cut(self):
        """A model without a recurrent state keeps the plain KV prefix."""
        cache, _ = _short_read(anchor_pages=0)
        cache.cache_controller._presence_pool_transfers = lambda: []
        cache.check_prefetch_progress(REQ)
        self.assertEqual(int(cache.prefetch_loaded_tokens_by_reqid[REQ]), READ)
