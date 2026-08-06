"""Hermetic (CPU-only) tests for the #581 mamba pin trace on the class
production actually runs: UnifiedRadixCache with the MAMBA component.

Boot 26 logged

    Tree cache initialized: source=default impl=UnifiedRadixCache
      hybrid_swa=False hybrid_ssm=True hierarchical=True

and emitted ZERO trace lines with SGLANG_MAMBA_PIN_TRACE=50 armed, because
the first trace landed in `HiMambaRadixCache`. `registry.py:106-109` routes
`--enable-hierarchical-cache` + hybrid SSM to `_create_unified_radix_cache`
unconditionally, so the hierarchical MAMBA path is UnifiedRadixCache and the
Mamba* classes never see this configuration.

The trace reports the same ledger as the Hi variant, with the pin counts
resolved through the unified registries (`_OngoingWriteThrough` /
`_OngoingLoadBack` carry the acquire's skip set, so an entry that skipped
MAMBA is correctly NOT counted as holding a mamba pin).
"""

# The unified fixture lives beside this file; the suite runs without a
# package __init__, so load it by path rather than by relative import.
import importlib.util
import os
import unittest
from array import array
from collections import Counter

from sglang.srt.environ import envs
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache_components import ComponentType
from sglang.test.ci.ci_register import register_cpu_ci

_spec = importlib.util.spec_from_file_location(
    "_unified_fixture",
    os.path.join(os.path.dirname(__file__), "test_unified_radix_cache_unittest.py"),
)
_fixture = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fixture)
# The shared fixture builds on the accelerator; these tests are CPU-only.
_fixture.get_device = lambda *args, **kwargs: "cpu"
CacheConfig = _fixture.CacheConfig
build_fixture = _fixture.build_fixture

register_cpu_ci(est_time=15)

TRACE_LOGGER = "sglang.srt.mem_cache.unified_radix_cache"


def _mamba_cfg() -> CacheConfig:
    return CacheConfig(components=(ComponentType.FULL, ComponentType.MAMBA))


def _key(token_ids) -> RadixKey:
    return RadixKey(array("q", token_ids))


def _insert(cache, allocator, pool, token_ids):
    slot = pool.mamba_allocator.alloc(1)
    assert slot is not None, "test setup: mamba pool exhausted"
    cache.insert(
        InsertParams(
            key=_key(token_ids),
            value=allocator.alloc(len(token_ids)),
            mamba_value=slot,
        )
    )
    return cache.match_prefix(MatchPrefixParams(key=_key(token_ids))).last_device_node


class TestUnifiedPinTrace(unittest.TestCase):
    def test_trace_line_renders_with_the_pin_ledger(self):
        with envs.SGLANG_MAMBA_PIN_TRACE.override(1):
            cache, allocator, pool = build_fixture(_mamba_cfg())
            self.assertEqual(cache._pin_trace_every, 1)
            node = _insert(cache, allocator, pool, list(range(100, 116)))
            cache.inc_lock_ref(node)
            with self.assertLogs(TRACE_LOGGER, level="INFO") as captured:
                cache.check_hicache_events()

        line = next(m for m in captured.output if "MAMBA-PIN-TRACE" in m)
        for field in (
            "impl=unified",
            "tick=",
            "ack_write=",
            "ack_load=",
            "wt_mamba_pins=",
            "lb_mamba_pins=",
            "ongoing_wt=",
            "ongoing_lb=",
            "ongoing_backup=",
            "protected=",
            "evictable=",
            "mamba_avail=",
            "ops[",
        ):
            self.assertIn(field, line)

        # The lock this test took is attributed to THIS function, and it moved
        # a real mamba ref (the node carries a checkpoint).
        self.assertIn("inc@test_trace_line_renders_with_the_pin_ledger=1", line)
        self.assertIn("inc_mamba@test_trace_line_renders_with_the_pin_ledger=1", line)
        # One checkpoint, locked -> protected, nothing evictable.
        self.assertIn("protected=1", line)
        self.assertIn("evictable=0", line)

    def test_release_is_attributed_to_its_own_site(self):
        with envs.SGLANG_MAMBA_PIN_TRACE.override(1):
            cache, allocator, pool = build_fixture(_mamba_cfg())
            node = _insert(cache, allocator, pool, list(range(100, 116)))
            cache.inc_lock_ref(node)
            self._release(cache, node)
            with self.assertLogs(TRACE_LOGGER, level="INFO") as captured:
                cache.check_hicache_events()

        line = next(m for m in captured.output if "MAMBA-PIN-TRACE" in m)
        self.assertIn("dec@_release=1", line)
        self.assertIn("dec_mamba@_release=1", line)
        # Released -> the checkpoint is cache again.
        self.assertIn("protected=0", line)
        self.assertIn("evictable=1", line)

    def _release(self, cache, node):
        cache.dec_lock_ref(node)

    def test_a_tombstone_lock_is_counted_as_a_call_but_not_as_a_mamba_ref(self):
        """The inc/inc_mamba split is what separates 'lock traffic' from
        'pool pressure': only the latter can exhaust the state pool."""
        with envs.SGLANG_MAMBA_PIN_TRACE.override(1):
            cache, allocator, pool = build_fixture(_mamba_cfg())
            token_ids = list(range(100, 116))
            node = _insert(cache, allocator, pool, token_ids)
            # Drop the checkpoint, keeping the node: a mamba tombstone.
            cd = node.component_data[ComponentType.MAMBA]
            pool.mamba_allocator.free(cd.value)
            cache.component_evictable_size_[ComponentType.MAMBA] -= len(cd.value)
            cd.value = None

            cache.inc_lock_ref(node)
            with self.assertLogs(TRACE_LOGGER, level="INFO") as captured:
                cache.check_hicache_events()

        line = next(m for m in captured.output if "MAMBA-PIN-TRACE" in m)
        self.assertIn(
            "inc@test_a_tombstone_lock_is_counted_as_a_call_but_not_as_a_mamba_ref=1",
            line,
        )
        self.assertNotIn("inc_mamba@", line)

    def test_counters_reset_between_lines(self):
        with envs.SGLANG_MAMBA_PIN_TRACE.override(1):
            cache, allocator, pool = build_fixture(_mamba_cfg())
            node = _insert(cache, allocator, pool, list(range(100, 116)))
            cache.inc_lock_ref(node)
            with self.assertLogs(TRACE_LOGGER, level="INFO") as first:
                cache.check_hicache_events()
            with self.assertLogs(TRACE_LOGGER, level="INFO") as second:
                cache.check_hicache_events()

        self.assertIn("inc_mamba@", first.output[0])
        self.assertIn("ops[]", next(m for m in second.output if "MAMBA-PIN-TRACE" in m))

    def test_interval_throttles_the_line(self):
        with envs.SGLANG_MAMBA_PIN_TRACE.override(3):
            cache, _, _ = build_fixture(_mamba_cfg())
            with self.assertLogs(TRACE_LOGGER, level="INFO") as captured:
                for _ in range(6):
                    cache.check_hicache_events()
        self.assertEqual(sum(1 for m in captured.output if "MAMBA-PIN-TRACE" in m), 2)

    def test_default_is_off(self):
        cache, _, _ = build_fixture(_mamba_cfg())
        self.assertEqual(cache._pin_trace_every, 0)
        cache.check_hicache_events()
        self.assertEqual(cache._pin_trace_ops, Counter())

    def test_pin_count_ignores_entries_whose_acquire_skipped_mamba(self):
        """A write-through/load-back lock taken on a mamba TOMBSTONE holds no
        mamba pin; counting it as one would hide the real pin pressure."""
        cache, allocator, pool = build_fixture(_mamba_cfg())
        node = _insert(cache, allocator, pool, list(range(100, 116)))

        holds_pin = _FakeEntry(node, DecLockRefParams())
        skipped = _FakeEntry(
            node,
            DecLockRefParams(skip_lock_node_ids={ComponentType.MAMBA: {node.id}}),
        )
        no_lock = _FakeEntry(node, None)

        self.assertEqual(cache._mamba_pins_in({1: holds_pin}), 1)
        self.assertEqual(cache._mamba_pins_in({1: skipped}), 0)
        self.assertEqual(cache._mamba_pins_in({1: no_lock}), 0)
        self.assertEqual(
            cache._mamba_pins_in({1: holds_pin, 2: skipped, 3: no_lock}), 1
        )


class _FakeEntry:
    """Shape of the `_Ongoing*` NamedTuples the trace reads."""

    def __init__(self, node, lock_params):
        self.node = node
        self.lock_params = lock_params


if __name__ == "__main__":
    unittest.main()
