"""#743: mamba slot pressure must be observable per event.

THE GAP, from ``NOTE_743_mamba_slot_hitrate.md`` §4.1-4.2. Two halves of one
event were silent:

* a SUCCESSFUL ``evict_mamba`` -- the pool yielding a slot by destroying a
  cached anchor -- emitted nothing at all. ``_tombstone_internal_node``
  (``mamba_radix_cache.py``) decrements ``mamba_evictable_size_`` and clears
  ``node.mamba_value``; that is the whole record. ``_log_mamba_slot_starvation``
  covers only the pool FAILING to yield, and
  ``BasePrefixCache.update_eviction_metrics`` -- fed by every sibling cache --
  is never called on either mamba lineage.
* a prefix match cut short because the anchor was gone was indistinguishable
  from a genuine cache miss. The only report was the ``SGLANG_MAMBA_CKPT_DEBUG``
  line, default off and confined to the ``--mamba-checkpoint-interval``
  lineage; ``hi_mamba_radix_cache._match_post_processor`` had no emitter at all.

So "did the 12-slot pool cost us prefix reuse" could not be answered from a
boot log, which is why §5 of the note puts the instrument BEFORE the
agent-shaped soak: without it the soak produces the same unanswerable logs.

WHAT IS PINNED HERE, in three layers:

1. the observer as a pure unit against a FAKE CLOCK -- cadence, the
   suppression rollup, the running totals, and the wording, none of which
   needs a radix tree;
2. the WIRING, on a real ``MambaRadixCache`` built entirely on CPU tensors
   (the ``test_mamba_slot_starvation.py`` fixture shape), driving a genuine
   eviction and asserting the line the operator will actually grep;
3. can-fail counterweights: an eviction that frees nothing stays silent, a
   healthy match stays silent, and rate 0 restores the pre-#743 silence
   exactly.
"""

import unittest
from array import array

import torch

from sglang.srt.configs.mamba_utils import Mamba2CacheParams, Mamba2StateShape
from sglang.srt.environ import envs
from sglang.srt.layers.attention.fla.chunk_delta_h import CHUNK_SIZE as FLA_CHUNK_SIZE
from sglang.srt.mem_cache.allocator import TokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import InsertParams, MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
from sglang.srt.mem_cache.mamba_slot_observer import (
    DEFAULT_LOG_BURST,
    DEFAULT_LOG_RATE_PER_S,
    LOG_PREFIX,
    MambaSlotObserver,
    anchor_depth_tokens,
    probe_available,
)
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, HybridReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15)

NUM_LAYERS = 8
GLOBAL_INTERVAL = 4
KV_POOL_SIZE = 512
MAX_CONTEXT_LEN = 128
CACHE_LOGGER = "sglang.srt.mem_cache.mamba_radix_cache"


# --------------------------------------------------------------------------
# Layer 1: the observer alone, on a fake clock.
# --------------------------------------------------------------------------
class TestTheObserverCadence743(CustomTestCase):
    def test_the_first_eviction_of_a_boot_is_always_emitted(self):
        """RED-FIRST on the silence itself. An empty starting bucket would
        swallow exactly the events an instrument exists to catch."""
        obs = MambaSlotObserver(rate_per_s=1.0, burst=1)
        lines = obs.note_eviction(now=0.0, requested=1, evicted=1, nodes=[(7, 40)])
        self.assertEqual(len(lines), 1)
        self.assertIn(f"{LOG_PREFIX} EVICT", lines[0])

    def test_the_line_names_the_node_and_the_anchor_tokens(self):
        """The quantity #743 §1 could not get: the PREFIX cost, not the slot
        count."""
        obs = MambaSlotObserver(rate_per_s=1.0, burst=1)
        (line,) = obs.note_eviction(
            now=0.0,
            requested=2,
            evicted=2,
            nodes=[(7, 40), (9, 128)],
            evictable=3,
            protected=8,
            available=0,
            lineage="device",
        )
        self.assertIn("freed 2 of 2 requested slot(s)", line)
        self.assertIn("168 anchor tok", line)
        self.assertIn("node 7@40tok", line)
        self.assertIn("node 9@128tok", line)
        self.assertIn("available=0", line)
        self.assertIn("evictable=3", line)
        self.assertIn("held-by-running=8", line)

    def test_an_unmeasured_quantity_prints_a_question_mark(self):
        """A swallowed probe must never yield a plausible 0 -- the #698/#714
        shape."""
        obs = MambaSlotObserver(rate_per_s=1.0, burst=1)
        (line,) = obs.note_eviction(now=0.0, requested=1, evicted=1, nodes=[(1, 5)])
        self.assertIn("available=?", line)
        self.assertNotIn("available=0", line)

    def test_an_eviction_that_freed_nothing_is_not_recorded(self):
        """That is the STARVATION case, which _log_mamba_slot_starvation owns.
        Counting it here would inflate the rate #743 wants measured."""
        obs = MambaSlotObserver(rate_per_s=10.0, burst=10)
        self.assertEqual(obs.note_eviction(now=0.0, requested=1, evicted=0), [])
        self.assertEqual(obs.evictions, 0)
        self.assertEqual(obs.slots_evicted, 0)

    def test_over_rate_events_are_suppressed_but_still_counted(self):
        """The whole reason this is a bucket and not 'first 3 then every 1000':
        no event is ever lost from the arithmetic, only from the lines."""
        obs = MambaSlotObserver(rate_per_s=1.0, burst=1)
        obs.note_eviction(now=0.0, requested=1, evicted=1, nodes=[(1, 10)])
        for i in range(5):
            self.assertEqual(
                obs.note_eviction(now=0.0, requested=1, evicted=1, nodes=[(2, 10)]),
                [],
                f"event {i} should be suppressed at rate 1/s",
            )
        self.assertEqual(obs.evictions, 6)
        self.assertEqual(obs.slots_evicted, 6)
        self.assertEqual(obs.anchor_tokens_lost, 60)

    def test_the_rollup_repays_the_suppressed_events(self):
        obs = MambaSlotObserver(rate_per_s=1.0, burst=1)
        obs.note_eviction(now=0.0, requested=1, evicted=1, nodes=[(1, 10)])
        for _ in range(3):
            obs.note_eviction(now=0.0, requested=1, evicted=1, nodes=[(2, 10)])
        lines = obs.note_eviction(now=5.0, requested=1, evicted=1, nodes=[(3, 10)])
        self.assertEqual(len(lines), 2, f"expected rollup + event, got {lines}")
        self.assertIn(f"{LOG_PREFIX} SUPPRESSED 3 eviction(s)", lines[0])
        self.assertIn("30 anchor tok", lines[0])
        self.assertIn("over 5.0s", lines[0])
        self.assertIn(f"{LOG_PREFIX} EVICT", lines[1])

    def test_the_rollup_is_paid_once(self):
        obs = MambaSlotObserver(rate_per_s=1.0, burst=1)
        obs.note_eviction(now=0.0, requested=1, evicted=1, nodes=[(1, 10)])
        obs.note_eviction(now=0.0, requested=1, evicted=1, nodes=[(2, 10)])
        first = obs.note_eviction(now=5.0, requested=1, evicted=1, nodes=[(3, 10)])
        second = obs.note_eviction(now=10.0, requested=1, evicted=1, nodes=[(4, 10)])
        self.assertEqual(len(first), 2)
        self.assertEqual(len(second), 1, f"the rollup must not repeat: {second}")

    def test_rate_zero_restores_the_pre_743_silence(self):
        """CAN-FAIL: the instrument must be switchable off completely."""
        obs = MambaSlotObserver(rate_per_s=0.0, burst=0)
        self.assertFalse(obs.enabled)
        self.assertEqual(
            obs.note_eviction(now=0.0, requested=1, evicted=1, nodes=[(1, 10)]), []
        )
        self.assertEqual(
            obs.note_truncation(now=0.0, rid="r", matched_tokens=100, usable_tokens=0),
            [],
        )

    def test_a_burst_inside_one_scheduler_step_is_reported_in_full(self):
        """WHY CAPACITY IS DECOUPLED FROM RATE. Slot pressure does not arrive
        at a steady 2/s -- it arrives as several evictions inside one step. A
        bucket sized to its own drain rate would print the first and suppress
        exactly the burst that matters."""
        obs = MambaSlotObserver()  # shipped defaults
        self.assertEqual(obs.capacity, DEFAULT_LOG_BURST)
        self.assertGreater(obs.capacity, obs.rate_per_s)
        emitted = 0
        for _ in range(int(DEFAULT_LOG_BURST)):
            if obs.note_eviction(now=0.0, requested=1, evicted=1, nodes=[(1, 4)]):
                emitted += 1
        self.assertEqual(emitted, int(DEFAULT_LOG_BURST))
        self.assertEqual(
            obs.note_eviction(now=0.0, requested=1, evicted=1, nodes=[(1, 4)]),
            [],
            "past the burst the rollup must take over",
        )

    def test_the_sustained_default_is_not_a_firehose(self):
        """A permanently-on WARNING instrument pays its sustained cost on every
        boot forever; 20/s was 1200 lines a minute on a thrashing rig."""
        self.assertLessEqual(DEFAULT_LOG_RATE_PER_S, 2.0)

    def test_a_lowered_rate_keeps_its_burst_detail(self):
        obs = MambaSlotObserver(rate_per_s=0.5)
        self.assertEqual(obs.capacity, DEFAULT_LOG_BURST)

    def test_a_raised_rate_gets_a_capacity_at_least_as_large(self):
        obs = MambaSlotObserver(rate_per_s=64.0)
        self.assertGreaterEqual(obs.capacity, 64.0)

    def test_would_emit_does_not_consume_a_token(self):
        """The caller peeks before paying for the per-node depth walks; a peek
        that consumed would make every other event lose its detail."""
        obs = MambaSlotObserver(rate_per_s=1.0, burst=1)
        self.assertTrue(obs.would_emit(0.0))
        self.assertTrue(obs.would_emit(0.0))
        self.assertEqual(
            len(obs.note_eviction(now=0.0, requested=1, evicted=1, nodes=[(1, 1)])), 1
        )
        self.assertFalse(obs.would_emit(0.0))


class TestTheTruncationWording743(CustomTestCase):
    def test_a_truncation_says_slot_pressure_not_cache_miss(self):
        """The distinction the instrument exists to restore."""
        obs = MambaSlotObserver(rate_per_s=1.0, burst=1)
        (line,) = obs.note_truncation(
            now=0.0,
            rid="req-1",
            matched_tokens=4096,
            usable_tokens=512,
            node_id=11,
            lineage="device",
        )
        self.assertIn(f"{LOG_PREFIX} TRUNCATED", line)
        self.assertIn("rid=req-1", line)
        self.assertIn("matched 4096 tok", line)
        self.assertIn("only 512 tok are backed", line)
        self.assertIn("3584 tok are re-prefilled", line)
        self.assertIn("SLOT PRESSURE, not a cache miss", line)

    def test_a_full_match_is_not_a_truncation(self):
        """CAN-FAIL: a healthy match must stay silent, or the instrument
        becomes the noise it was added to cut through."""
        obs = MambaSlotObserver(rate_per_s=1.0, burst=1)
        self.assertEqual(
            obs.note_truncation(
                now=0.0, rid="r", matched_tokens=512, usable_tokens=512
            ),
            [],
        )
        self.assertEqual(obs.truncations, 0)

    def test_the_two_lineages_are_labelled(self):
        obs = MambaSlotObserver(rate_per_s=100.0, burst=100)
        (a,) = obs.note_truncation(
            now=0.0, rid="r", matched_tokens=10, usable_tokens=0, lineage="device"
        )
        (b,) = obs.note_truncation(
            now=0.0, rid="r", matched_tokens=10, usable_tokens=0, lineage="host-tier"
        )
        self.assertIn("(device)", a)
        self.assertIn("(host-tier)", b)


class TestProbeArmour743(CustomTestCase):
    def test_a_raising_allocator_yields_none_not_zero(self):
        class Boom:
            def available_size(self):
                raise RuntimeError("no")

        self.assertIsNone(probe_available(Boom()))
        self.assertIsNone(probe_available(None))

    def test_anchor_depth_ignores_the_root(self):
        """A root-anchored node must not report the root's (absent) tokens."""

        class Node:
            def __init__(self, parent, value):
                self.parent = parent
                self.value = value

        root = Node(None, None)
        a = Node(root, [0] * 7)
        b = Node(a, [0] * 5)
        self.assertEqual(anchor_depth_tokens(root), 0)
        self.assertEqual(anchor_depth_tokens(a), 7)
        self.assertEqual(anchor_depth_tokens(b), 12)


# --------------------------------------------------------------------------
# Layer 2: the wiring, on a real CPU-resident MambaRadixCache.
# --------------------------------------------------------------------------
def _build_tree(mamba_size=6, max_num_reqs=10):
    """MambaRadixCache + pools pinned to CPU (no accelerator required).

    The shape is lifted from ``test_mamba_slot_starvation.py`` deliberately:
    the wiring must be exercised through the real production objects, not a
    stub, because the defect being closed is that the production path emitted
    nothing.
    """
    server_args = ServerArgs(model_path="dummy", page_size=1)
    server_args._mamba_cache_chunk_size = FLA_CHUNK_SIZE
    set_global_server_args_for_scheduler(server_args)

    device = "cpu"
    full_attention_layer_ids = [
        i for i in range(GLOBAL_INTERVAL - 1, NUM_LAYERS, GLOBAL_INTERVAL)
    ]
    mamba_layers = [i for i in range(NUM_LAYERS) if i not in full_attention_layer_ids]
    with envs.SGLANG_MAMBA_SSM_DTYPE.override("bfloat16"):
        shape = Mamba2StateShape.create(
            tp_world_size=1,
            intermediate_size=512,
            n_groups=4,
            num_heads=8,
            head_dim=64,
            state_size=32,
            conv_kernel=4,
        )
        cache_params = Mamba2CacheParams(shape=shape, layers=mamba_layers)

    req_to_token_pool = HybridReqToTokenPool(
        size=max_num_reqs,
        mamba_size=mamba_size,
        mamba_spec_state_size=max_num_reqs,
        max_context_len=MAX_CONTEXT_LEN,
        device=device,
        enable_memory_saver=False,
        cache_params=cache_params,
        mamba_layer_ids=mamba_layers,
        enable_mamba_extra_buffer=False,
        enable_linear_replayssm=False,
    )
    kv_pool = HybridLinearKVPool(
        size=KV_POOL_SIZE,
        dtype=torch.bfloat16,
        page_size=1,
        head_num=2,
        head_dim=64,
        full_attention_layer_ids=full_attention_layer_ids,
        device=device,
        enable_memory_saver=False,
        mamba_pool=req_to_token_pool.mamba_pool,
    )
    allocator = TokenToKVPoolAllocator(
        size=KV_POOL_SIZE,
        dtype=torch.bfloat16,
        device=device,
        kvcache=kv_pool,
        need_sort=False,
    )
    tree = MambaRadixCache(
        params=CacheInitParams(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            disable=False,
            enable_kv_cache_events=False,
            enable_mamba_extra_buffer=False,
        )
    )
    return tree, allocator, req_to_token_pool


def _insert_cached(tree, allocator, pool, token_ids, with_state=True):
    """Insert one cached prefix, optionally carrying a mamba state."""
    slot = pool.mamba_allocator.alloc(1) if with_state else None
    kv_indices = allocator.alloc(len(token_ids))
    tree.insert(
        InsertParams(
            key=RadixKey(array("q", token_ids)),
            value=kv_indices,
            mamba_value=slot,
        )
    )
    return slot


class TestTheEvictionIsNoLongerSilent743(CustomTestCase):
    def test_a_successful_eviction_emits_a_line(self):
        """RED-FIRST: this is the event that produced no output at all."""
        tree, allocator, pool = _build_tree(mamba_size=4)
        _insert_cached(tree, allocator, pool, list(range(1000, 1008)))
        _insert_cached(tree, allocator, pool, list(range(2000, 2008)))
        self.assertGreater(tree.mamba_evictable_size(), 0, "nothing to evict")

        with self.assertLogs(CACHE_LOGGER, level="WARNING") as captured:
            freed = tree.evict_mamba(1)
        self.assertGreater(freed, 0, "the pool must actually yield a slot")
        evict_lines = [m for m in captured.output if f"{LOG_PREFIX} EVICT" in m]
        self.assertEqual(len(evict_lines), 1, captured.output)
        self.assertIn("(device)", evict_lines[0])
        self.assertIn("anchor tok of resumable prefix", evict_lines[0])

    def test_the_line_carries_the_pool_state_the_note_asked_for(self):
        tree, allocator, pool = _build_tree(mamba_size=4)
        _insert_cached(tree, allocator, pool, list(range(1000, 1008)))
        _insert_cached(tree, allocator, pool, list(range(2000, 2008)))
        with self.assertLogs(CACHE_LOGGER, level="WARNING") as captured:
            tree.evict_mamba(1)
        (line,) = [m for m in captured.output if f"{LOG_PREFIX} EVICT" in m]
        self.assertIn("pool now available=", line)
        self.assertIn("evictable=", line)
        self.assertIn("held-by-running=", line)
        self.assertIn("cumulative:", line)
        self.assertNotIn("available=?", line, "the CPU fixture can be asked")

    def test_a_node_id_and_a_nonzero_anchor_depth_are_reported(self):
        """Without the depth the line says a slot went; #743 needs to know how
        much resumable prefix went with it."""
        tree, allocator, pool = _build_tree(mamba_size=4)
        _insert_cached(tree, allocator, pool, list(range(1000, 1008)))
        _insert_cached(tree, allocator, pool, list(range(2000, 2008)))
        with self.assertLogs(CACHE_LOGGER, level="WARNING") as captured:
            tree.evict_mamba(1)
        (line,) = [m for m in captured.output if f"{LOG_PREFIX} EVICT" in m]
        self.assertIn("node ", line)
        self.assertNotIn("dropping 0 anchor tok", line)
        self.assertNotIn("per-node detail not collected", line)

    def test_an_eviction_that_frees_nothing_stays_silent(self):
        """CAN-FAIL: the starved pool is a different event with its own
        emitter, and must not gain a second one here."""
        tree, allocator, pool = _build_tree(mamba_size=4)
        self.assertEqual(tree.evict_mamba(1), 0)
        logger_obj = __import__("logging").getLogger(CACHE_LOGGER)
        with self.assertNoLogs(logger_obj, level="WARNING"):
            tree.evict_mamba(1)

    def test_rate_zero_makes_the_wiring_silent_again(self):
        """CAN-FAIL: byte-identical behaviour to before #743 on demand."""
        tree, allocator, pool = _build_tree(mamba_size=4)
        _insert_cached(tree, allocator, pool, list(range(1000, 1008)))
        _insert_cached(tree, allocator, pool, list(range(2000, 2008)))
        from sglang.srt.mem_cache.mamba_slot_observer import observer_of

        observer_of(tree).rate_per_s = 0.0
        logger_obj = __import__("logging").getLogger(CACHE_LOGGER)
        with self.assertNoLogs(logger_obj, level="WARNING"):
            self.assertGreater(tree.evict_mamba(1), 0)

    def test_the_eviction_still_evicts(self):
        """The instrument must not change what the cache does."""
        tree, allocator, pool = _build_tree(mamba_size=4)
        _insert_cached(tree, allocator, pool, list(range(1000, 1008)))
        _insert_cached(tree, allocator, pool, list(range(2000, 2008)))
        before = pool.mamba_allocator.available_size()
        freed = tree.evict_mamba(1)
        self.assertGreater(freed, 0)
        self.assertGreater(pool.mamba_allocator.available_size(), before)


class TestTheTruncationIsNoLongerSilent743(CustomTestCase):
    def test_a_match_beyond_the_last_anchor_emits_a_line(self):
        """The event, built the way production reaches it: a chain of cached
        prefixes, the MIDDLE one's state evicted by slot pressure (tombstoned,
        so its KV stays), then a request whose prefix ends there. The radix
        matches both nodes; only the first is backed by a surviving state.

        The deepest node is locked, and the shallowest too, so the LRU walk has
        exactly one legal choice -- the test must pin the emitter, not the LRU
        order."""
        tree, allocator, pool = _build_tree(mamba_size=6)
        a = list(range(3000, 3008))
        b = a + [3100, 3101]
        c = b + [3200, 3201]
        _insert_cached(tree, allocator, pool, a)
        _insert_cached(tree, allocator, pool, b)
        _insert_cached(tree, allocator, pool, c)

        node_a = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", a)))
        ).last_device_node
        node_c = tree.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", c)))
        ).last_device_node
        tree.inc_lock_ref(node_a)
        tree.inc_lock_ref(node_c)

        self.assertGreater(tree.evict_mamba(1), 0, "the middle node must yield")
        tree.dec_lock_ref(node_c)

        with self.assertLogs(CACHE_LOGGER, level="WARNING") as captured:
            tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", b))))
        lines = [m for m in captured.output if f"{LOG_PREFIX} TRUNCATED" in m]
        self.assertEqual(len(lines), 1, captured.output)
        self.assertIn("SLOT PRESSURE, not a cache miss", lines[0])
        self.assertIn("(device)", lines[0])
        self.assertIn("re-prefilled", lines[0])

    def test_a_fully_anchored_match_stays_silent(self):
        """CAN-FAIL counterweight to the test above."""
        tree, allocator, pool = _build_tree(mamba_size=4)
        base = list(range(4000, 4008))
        _insert_cached(tree, allocator, pool, base, with_state=True)
        logger_obj = __import__("logging").getLogger(CACHE_LOGGER)
        with self.assertNoLogs(logger_obj, level="WARNING"):
            tree.match_prefix(MatchPrefixParams(key=RadixKey(array("q", base))))


if __name__ == "__main__":
    unittest.main()
