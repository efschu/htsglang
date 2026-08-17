"""#681 remainder: the paged extend path asks for relief and throws the catch away.

#679 built the degradation: when an extend allocation fails, a rank-local relief
provider is consulted and the allocation is RETRIED before the raise.
``alloc_token_slots`` does exactly that (common.py:534-536):

    freed = _attempt_extend_relief(num_tokens)
    if freed > 0:
        out_cache_loc = allocator.alloc(num_tokens)   # <- retry
        logger.warning(... "the retry %s" ...)

Its page_size > 1 twin, ``alloc_paged_token_slots_extend``, calls the same
provider, logs that relief SUCCEEDED, and then falls through to the raise
without ever spending what it was given (common.py:1197-1205). So on that path
the net is asked, the catch is announced, and the batch dies anyway -- which is
the shape of the 01:46 specimen: a raise past a degradation that had already
produced the memory to avoid it.

Reachability, stated plainly: this rig runs page_size 1, so the twin is not on
today's hot path -- but ``_alloc_page_size`` notes that DCP swaps in an
allocator whose page_size is ``server_args.page_size * dcp_size``, so any
dcp_size > 1 boot takes it. Uneven DCP is a shipped feature, so this is a live
path on a supported configuration, not a hypothetical.

The fix is the #679 discipline applied verbatim, no new policy: retry once after
relief, keep the same fail-loud raise when the retry still fails.

Hermetic: mocks only, no CUDA.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.mem_cache import common as mc


class _Alloc:
    """Allocator whose alloc_extend fails until relief has run."""

    page_size = 2

    def __init__(self, succeed_after_relief=True):
        self.calls = 0
        self.succeed_after_relief = succeed_after_relief
        self.freed = False

    def available_size(self):
        # Reached through evict_from_tree_cache -> uniform_avail_for_evict.
        # Zero keeps the eviction trigger armed, which is the specimen's state.
        return 0

    def alloc_extend(self, *a, **k):
        self.calls += 1
        if self.freed and self.succeed_after_relief:
            return torch.tensor([1, 2, 3], dtype=torch.int64)
        return None

    def backup_state(self):
        return "state"


def _tree(alloc):
    tc = MagicMock()
    tc.is_chunk_cache.return_value = False
    tc.token_to_kv_pool_allocator = alloc
    tc.uniform_avail_floor = None
    tc.uniform_admitted_since_floor = 0
    tc.evictable_size.return_value = 0
    tc.full_evictable_size.return_value = 0
    tc.available_and_evictable_str.return_value = "avail=0 evictable=0"
    return tc


def _args(tree):
    t = torch.tensor([0], dtype=torch.int64)
    return dict(
        tree_cache=tree,
        prefix_lens=t,
        prefix_lens_cpu=t,
        seq_lens=t,
        seq_lens_cpu=t,
        last_loc=t,
        extend_num_tokens=512,
    )


class PagedExtendMustSpendTheReliefItAsksFor(unittest.TestCase):
    def setUp(self):
        # clear_extend_relief_providers() is the module's own test seam; using
        # it keeps this test off a private list whose name could drift.
        self._saved = list(mc._extend_relief_providers)
        mc.clear_extend_relief_providers()

    def tearDown(self):
        mc.clear_extend_relief_providers()
        mc._extend_relief_providers.extend(self._saved)

    def test_a_successful_relief_is_retried_not_discarded(self):
        """THE FALSIFIER. Relief frees the memory; the batch must not die."""
        alloc = _Alloc()

        def provider(n):
            alloc.freed = True
            return n

        mc.register_extend_relief_provider(provider)
        out = mc.alloc_paged_token_slots_extend(**_args(_tree(alloc)))
        self.assertIsNotNone(out)
        self.assertGreaterEqual(
            alloc.calls, 2, "the allocation was never retried after relief"
        )

    def test_it_still_raises_when_the_retry_also_fails(self):
        """Fail-loud is preserved -- relief is a net, never a guarantee."""
        alloc = _Alloc(succeed_after_relief=False)

        def provider(n):
            alloc.freed = True
            return n

        mc.register_extend_relief_provider(provider)
        with self.assertRaises(RuntimeError):
            mc.alloc_paged_token_slots_extend(**_args(_tree(alloc)))

    def test_no_relief_available_still_raises_without_a_second_attempt(self):
        """A provider that frees nothing must not trigger a pointless retry."""
        alloc = _Alloc()
        mc.register_extend_relief_provider(lambda n: 0)
        with self.assertRaises(RuntimeError):
            mc.alloc_paged_token_slots_extend(**_args(_tree(alloc)))
        self.assertEqual(alloc.calls, 1)

    def test_the_sibling_path_already_behaves_this_way(self):
        """Pins the asymmetry this closes: alloc_token_slots retries today."""
        alloc = MagicMock()
        alloc.available_size.return_value = 0
        seq = [None, torch.tensor([7], dtype=torch.int64)]
        alloc.alloc.side_effect = lambda n: seq.pop(0) if seq else None
        alloc.free_group = []
        tc = _tree(alloc)
        mc.register_extend_relief_provider(lambda n: n)
        out = mc.alloc_token_slots(tc, 512)
        self.assertIsNotNone(out)


if __name__ == "__main__":
    unittest.main()
