"""#843 -- why every window-6 storage prefetch reported ``loaded=0``.

THE FINDING. On the window-6 boot (integ/round6 @ 241e7ac385,
``boot_window6_0824_0118.log``) 339 storage prefetches completed and EVERY ONE
reported ``loaded=0``. The raw counts, from the log::

    309 x  completed_synced=0     matched=0      loaded=0
      6 x  completed_synced=24576 matched=24576  loaded=0
     15 x  completed_synced=24576 matched=57     loaded=0
      3 x  completed_synced=40960 matched=45     loaded=0
      3 x  completed_synced=24576 matched=8192   loaded=0
      3 x  completed_synced=16384 matched=50     loaded=0

The first 315 are arithmetic, not defects: 309 fetched NOTHING (there was no
hit in the storage tier) and 6 matched their whole span (the tree already held
it). ``loaded = min_completed - matched`` is 0 in both by construction.

The remaining 24 fetched a large unmatched tail -- 24519, 40915, 16384, 16334
tokens -- and still reported 0. Those are the real question, and this file is
their answer.

THE LANDING CONDITION, which is what these tests pin. ``_insert_helper_host``
walks down from ``last_host_node`` through EXISTING TREE CHILDREN and attaches
the fetched tail wherever the walk stops. Since #841 that attachment is refused
when the landing parent carries no host copy, because a backed child under an
un-backed parent is the state that killed both window-5 boots on the idle path.

So a prefetch can only land in one of two situations:

  * the walk matches NOTHING, so it lands on the root, which is exempt; or
  * the walk stops on a node that is itself host-backed.

**A note on the word "matched", because the window-6 reading turns on it:**
``matched`` counts tokens matched against tree nodes of ANY tier. A large
``matched`` does NOT mean the walk found host nodes. In window 6 it means the
DEVICE tree already held that prefix -- and every one of those walks then landed
on a device-only node and was refused.

WHY NOTHING WAS BACKED, i.e. why the second situation never occurred. This is
NOT the gate's doing, and the tree says so in its own voice
(``hicache_phase_guard.py:221-236``, once per rank, in the window-6 log)::

    HiCache write is DISARMED for the duration of the TP decode phase: this
    controller is bound to the pool it was built with (the boot PP stack's),
    which is not the pool the model is using now. ... Prefixes cached in this
    phase are therefore not staged from device; the host and storage tiers are
    unaffected, and THE PP PHASE RESUMES NORMAL STAGING AFTER THE NEXT FLIP.

The guard is `active_phase() != bound_phase()` and it is correct -- a copy
against the retired pool was the #760 SIGSEGV. It is also **bounded by design:
bounded by the next ``tp_to_pp`` flip.** Window 6 performed exactly one flip,
``pp_to_tp`` at epoch 1, and never flipped back -- the #834/#839 level deadlock
makes ``tp_to_pp`` unreachable. So a disarm designed to be temporary became
permanent, nothing was ever staged to host, and the flip-writeback confirms it
with ``already_staged=0`` and ``skipped 16 node(s) whose parent carries no host
copy``.

THE CHAIN, therefore, and note where the #841 gate actually sits in it::

    TP-sticky boot (no tp_to_pp, #834/#839)
      -> active_phase != bound_phase for the whole run
        -> every device->host write refused (#760 guard, by design)
          -> no node is ever backed (already_staged=0)
            -> every host insert lands on an un-backed node
              -> #841 refuses it (correctly -- attaching is the window-5 crash)
                -> loaded=0

The gate is the LAST link, not the first. Removing it would not restore
retention; it would restore the crash, and the "loaded=16334" that window 5
reported at ``matched=45`` was that crash being counted as a hit.
"""

import unittest

import torch

from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import (
    BASE_COMPONENT_TYPE,
    UnifiedRadixCache,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

# ~6s: three tiny CPU-only radix trees. No accelerator, no group, no boot.
register_cpu_ci(est_time=6, suite="base-a-test-cpu")

#: The window-6 shape, to the token: a 24576-token fetch whose first 57 tokens
#: the device tree already holds.
W6_DEVICE_PREFIX = 57
W6_FETCH = 200  # stands in for 24576; the arithmetic is what matters


def build_cache(size: int = 4096) -> UnifiedRadixCache:
    """A real UnifiedRadixCache on the CPU (the #841 harness).

    ``cache_controller`` stays None, which is what arms the parent-backed
    invariant -- it is suppressed only for the ``write_back`` policy, and the
    window-6 boot ran ``--hicache-write-policy write_through``.
    """
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
    return UnifiedRadixCache(
        params=CacheInitParams(
            req_to_token_pool=ReqToTokenPool(
                size=8, max_context_len=size, device="cpu", enable_memory_saver=False
            ),
            token_to_kv_pool_allocator=TokenToKVPoolAllocator(
                size=size,
                dtype=torch.bfloat16,
                device="cpu",
                kvcache=MHATokenToKVPool(
                    size=size,
                    page_size=1,
                    dtype=torch.bfloat16,
                    head_num=2,
                    head_dim=8,
                    layer_num=2,
                    device="cpu",
                    enable_memory_saver=False,
                ),
                need_sort=False,
            ),
            page_size=1,
            disable=False,
            tree_components=(ComponentType.FULL,),
        )
    )


def prefetch(cache, fetched, base=5000):
    """Exactly what ``check_prefetch_progress`` does on completion."""
    return cache._insert_helper_host(
        cache.root_node,
        RadixKey(list(fetched)),
        torch.arange(base, base + len(fetched), dtype=torch.int64),
        [f"h{i}" for i in range(len(fetched))],
    )


def reported_loaded(result, min_completed):
    """The `loaded=` field exactly as check_prefetch_progress computes it."""
    if result.host_span_unclaimed:
        return 0
    return min_completed - result.prefix_len


class TestTheWindow6Shape(CustomTestCase):
    def test_a_device_prefix_makes_the_prefetch_land_on_an_unbacked_node(self):
        """The 24 real window-6 cases, reproduced.

        Nothing is backed (the TP-phase disarm), the device tree holds the
        first 57 tokens, so the walk stops on a device-only node and the
        fetched tail has nowhere legal to attach.
        """
        cache = build_cache()
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(1, W6_DEVICE_PREFIX + 1))),
                value=torch.arange(W6_DEVICE_PREFIX, dtype=torch.int64),
            )
        )

        result = prefetch(cache, range(1, W6_FETCH + 1))

        self.assertEqual(result.prefix_len, W6_DEVICE_PREFIX)  # "matched=57"
        self.assertTrue(result.host_span_unclaimed)
        self.assertIsNone(result.inserted_host_node)
        self.assertEqual(reported_loaded(result, W6_FETCH), 0)  # "loaded=0"
        self.assertEqual(cache._host_insert_refused_unbacked_parent, 1)
        cache.sanity_check()

    def test_the_whole_span_already_present_is_arithmetic_not_a_refusal(self):
        """The 6 cases with matched == completed.

        These report loaded=0 because there was nothing new to load, NOT
        because anything was refused -- the counter stays at zero. Reading
        them as failures would inflate the finding by a quarter.
        """
        cache = build_cache()
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(1, W6_FETCH + 1))),
                value=torch.arange(W6_FETCH, dtype=torch.int64),
            )
        )

        result = prefetch(cache, range(1, W6_FETCH + 1))

        self.assertEqual(result.prefix_len, W6_FETCH)
        self.assertFalse(result.host_span_unclaimed)
        self.assertEqual(cache._host_insert_refused_unbacked_parent, 0)
        self.assertEqual(reported_loaded(result, W6_FETCH), 0)


class TestTheLoadFormThatWouldHaveLanded(CustomTestCase):
    """The prescription: what a driver must produce to make `loaded>0` possible.

    These are the danger direction for the whole finding. If they went red,
    the #841 gate really would have disabled the storage tier outright, and
    the honest verdict would be a retention regression rather than a load-form
    condition.
    """

    def test_a_key_with_no_device_prefix_lands_under_the_root(self):
        """matched=0 lands even with a completely empty host tier.

        The root is exempt, so a prefetch for content the DEVICE tree does not
        already hold attaches and reports the whole span as loaded -- no
        backed ancestor required. This is the cheapest load form that
        distinguishes "the tier is broken" from "the driver never asked it
        anything it could answer".
        """
        cache = build_cache()
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(1, 40))),
                value=torch.arange(39, dtype=torch.int64),
            )
        )

        # Disjoint key: shares no first token with anything in the tree.
        result = prefetch(cache, range(900, 900 + W6_FETCH))

        self.assertEqual(result.prefix_len, 0)
        self.assertFalse(result.host_span_unclaimed)
        self.assertIsNotNone(result.inserted_host_node)
        self.assertEqual(reported_loaded(result, W6_FETCH), W6_FETCH)
        cache.sanity_check()

    def test_a_backed_landing_parent_lands_the_tail(self):
        """The PP-phase shape: with staging armed, the ancestor is backed and
        the deep prefetch attaches beneath it."""
        cache = build_cache()
        first = prefetch(cache, range(1, W6_DEVICE_PREFIX + 1))
        self.assertIsNotNone(first.inserted_host_node)
        self.assertTrue(first.inserted_host_node.backuped)

        result = prefetch(cache, range(1, W6_FETCH + 1), base=9000)

        self.assertEqual(result.prefix_len, W6_DEVICE_PREFIX)
        self.assertFalse(result.host_span_unclaimed)
        self.assertIs(result.inserted_host_node.parent, first.inserted_host_node)
        self.assertEqual(reported_loaded(result, W6_FETCH), W6_FETCH - W6_DEVICE_PREFIX)
        cache.sanity_check()

    def test_the_refusal_is_what_keeps_the_tree_sane(self):
        """Why the gate may not simply be dropped to recover `loaded>0`.

        Force the pre-#841 behaviour by hand -- attach the tail under the
        device-only node -- and the idle-path check that killed both window-5
        boots fires. `loaded>0` bought that way is the defect, not retention.
        """
        cache = build_cache()
        cache.insert(
            InsertParams(
                key=RadixKey(list(range(1, W6_DEVICE_PREFIX + 1))),
                value=torch.arange(W6_DEVICE_PREFIX, dtype=torch.int64),
            )
        )
        parent = cache.root_node.children[RadixKey([1]).child_key(cache.page_size)]
        self.assertFalse(parent.backuped)

        child = cache._add_new_node(
            parent,
            RadixKey(list(range(W6_DEVICE_PREFIX + 1, W6_FETCH + 1))),
            torch.arange(W6_FETCH - W6_DEVICE_PREFIX, dtype=torch.int64),
        )
        child.component_data[BASE_COMPONENT_TYPE].value = None
        child.component_data[BASE_COMPONENT_TYPE].host_value = torch.arange(
            W6_FETCH - W6_DEVICE_PREFIX, dtype=torch.int64
        )
        cache._update_evictable_leaf_sets(child)

        with self.assertRaises(AssertionError) as caught:
            cache.sanity_check()
        self.assertIn("backed up but parent", str(caught.exception))


class TestTheRefusalIsMeasurableAtInfo(CustomTestCase):
    """#843 -- the instrumentation half, and why it needs a pin.

    The #841 refusal was observable only through a ``logger.debug`` line and an
    in-process counter that is never logged. The boot recipe runs
    ``log_level='info'``, so on window 6 the refusal left NO trace at all: 339
    prefetches reported ``loaded=0`` and nothing in the log said which of them
    were refused and which simply had nothing to load. W-841's acceptance
    criterion 2 asks the operator to count exactly that line, so the criterion
    was unmeasurable as written.

    Pinned at the source rather than by driving ``check_prefetch_progress``,
    which would need a controller, an ongoing-prefetch entry and a collective.
    The property under test is the WIRING -- that the INFO line carries the
    field and that the field is fed from the refusal flag -- and that is
    exactly what source inspection can assert.
    """

    def _source(self):
        import inspect

        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        return inspect.getsource(UnifiedRadixCache.check_prefetch_progress)

    def test_the_info_line_reports_whether_the_insert_was_refused(self):
        src = self._source()
        self.assertIn(
            "refused=%d",
            src,
            "the HiCache prefetch INFO line must carry refused=, or loaded=0 "
            "cannot be split into 'nothing to load' and 'the tree declined the "
            "tail' without a code archaeology pass (window 6 cost one).",
        )

    def test_the_refused_field_is_fed_from_the_refusal_flag(self):
        src = self._source()
        self.assertIn(
            "int(insert_result.host_span_unclaimed)",
            src,
            "refused= must be fed from host_span_unclaimed, the single flag "
            "that #841 sets when it declines an insert. A field computed from "
            "anything else would measure something other than the refusal.",
        )


if __name__ == "__main__":
    unittest.main()
