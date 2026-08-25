"""W35: enqueue and consume are separate calls, and the cutover lands between.

#760 established this for WRITE-BACKS and fixed it in one place. The W35 sweep
found the same shape uncovered in three others:

  * `HiCacheController.start_loading` had NEITHER consume-time check, while
    `start_writing` had both;
  * `HybridCacheController.start_writing` and `.start_loading` -- the
    mamba/hybrid path, i.e. THIS RIG -- had neither either; their `write()`
    and `load()` carry the ENQUEUE-time checks only.

WHY THE LOAD SIDE IS THE WORSE ONE. A stale write-back corrupts a host copy
and tends to trip an ownership assertion. A stale LOAD fills device rows from
host slots the incoming phase does not own, and the tree then marks that
prefix RESIDENT -- attention reads KV nobody wrote, with no assertion
anywhere. This codebase ranks a silent wrong answer worse than a crash.

ONE AUTHORITY, FOUR CALLERS. The fix is `consume_gate`, and the four consume
points call it. Four copies of one rule is the shape that cost W32, so the
tests below pin the callers as well as the behaviour.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import inspect
import types
import unittest

from sglang.srt.managers.cache_controller import consume_gate
from sglang.test.test_utils import CustomTestCase


class _Op:
    def __init__(self, gen):
        self.binding_generation = gen


class _Ctl:
    def __init__(self, ops):
        self.write_queue = list(ops)
        self.load_queue = list(ops)


def _gen_at(n):
    """Pin the ONE stamp authority to generation n."""
    from sglang.srt.mem_cache import hicache_phase_binding as b

    b._STATE.reset()
    for _ in range(n):
        b._STATE.advance("tp")


class TestTheStaleBatchIsRefused(CustomTestCase):
    """The STAMP arm in isolation.

    `device_tier_disarmed` is pinned False here on purpose: in a bare test
    process no flip runtime is registered, so the phase predicate answers
    "disarmed" and the gate short-circuits before the stamp filter is ever
    reached. Pinning it isolates the arm under test -- and the arm it is
    pinned away from has its own class below, so neither goes unexercised.
    """

    def setUp(self):
        import sglang.srt.managers.cache_controller as cc

        _gen_at(2)
        self._orig = cc.device_tier_disarmed
        cc.device_tier_disarmed = lambda d: False
        self.addCleanup(setattr, cc, "device_tier_disarmed", self._orig)

    def test_a_batch_stamped_at_an_older_generation_is_dropped(self):
        ctl = _Ctl([_Op(1), _Op(1)])
        self.assertFalse(consume_gate(ctl, "load_queue", "load"))
        self.assertEqual(ctl.load_queue, [], "nothing stale may be consumed")

    def test_a_current_batch_proceeds(self):
        ctl = _Ctl([_Op(2)])
        self.assertTrue(consume_gate(ctl, "load_queue", "load"))
        self.assertEqual(len(ctl.load_queue), 1)

    def test_a_mixed_batch_keeps_only_the_current_ops(self):
        ctl = _Ctl([_Op(1), _Op(2), _Op(1)])
        self.assertTrue(consume_gate(ctl, "write_queue", "write"))
        self.assertEqual([o.binding_generation for o in ctl.write_queue], [2])

    def test_the_refusal_is_counted_by_name(self):
        # Never silent: the sweep's rule is that a dropped stale op must be
        # countable, or the next boot cannot tell "clean" from "blind".
        ctl = _Ctl([_Op(1)])
        consume_gate(ctl, "load_queue", "load")
        self.assertEqual(getattr(ctl, "_load_stamp_refusals", 0), 1)

    def test_an_empty_queue_is_not_a_refusal_event(self):
        ctl = _Ctl([])
        self.assertFalse(consume_gate(ctl, "load_queue", "load"))
        self.assertEqual(getattr(ctl, "_load_stamp_refusals", 0), 0)


class TestTheDisarmPredicateAlsoRefuses(CustomTestCase):
    """The stamp alone cannot cover it: with --phase-flip-rebind-hicache off
    the binding never advances, so every stamp matches by construction and the
    generation check is dead code. The phase predicate is the one that knows."""

    def test_a_disarmed_direction_clears_the_queue_even_when_stamps_match(self):
        import sglang.srt.managers.cache_controller as cc

        _gen_at(1)
        ctl = _Ctl([_Op(1)])  # stamp is CURRENT
        orig = cc.device_tier_disarmed
        try:
            cc.device_tier_disarmed = lambda d: True
            self.assertFalse(consume_gate(ctl, "load_queue", "load"))
        finally:
            cc.device_tier_disarmed = orig
        self.assertEqual(ctl.load_queue, [])
        self.assertEqual(getattr(ctl, "_load_phase_refusals", 0), 1)


class TestAllFourConsumePointsUseTheOneGate(CustomTestCase):
    """CAN-FAIL in the direction that actually bit: a consume point that stops
    calling the gate is a fifth copy of the rule, and silently exempt."""

    def test_every_consume_point_calls_it(self):
        from sglang.srt.managers.cache_controller import HiCacheController
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        for name, fn in (
            ("HiCacheController.start_writing", HiCacheController.start_writing),
            ("HiCacheController.start_loading", HiCacheController.start_loading),
            (
                "HybridCacheController.start_writing",
                HybridCacheController.start_writing,
            ),
            (
                "HybridCacheController.start_loading",
                HybridCacheController.start_loading,
            ),
        ):
            self.assertIn(
                "consume_gate", inspect.getsource(fn), f"{name} must call the gate"
            )

    def test_the_rule_is_not_reimplemented_inline_anywhere(self):
        from sglang.srt.managers.cache_controller import HiCacheController
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        for fn in (
            HiCacheController.start_writing,
            HiCacheController.start_loading,
            HybridCacheController.start_writing,
            HybridCacheController.start_loading,
        ):
            src = inspect.getsource(fn)
            self.assertNotIn("write_back_stamp_is_current(", src)

    def test_the_gate_is_checked_before_a_producer_is_allocated(self):
        # A refused load must cost nothing downstream; allocating a producer
        # for a batch that is then dropped leaks a layer-done slot.
        from sglang.srt.managers.cache_controller import HiCacheController

        src = inspect.getsource(HiCacheController.start_loading)
        self.assertLess(
            src.find("consume_gate"),
            src.find("update_producer"),
        )


class TestAgainstTheRealEnqueueSites(CustomTestCase):
    """The six-instance lesson: the gap was enqueue-vs-consume being SEPARATE
    calls. Pin that they really are separate, or this whole fix is moot."""

    def test_enqueue_and_consume_are_different_methods(self):
        from sglang.srt.managers.cache_controller import HiCacheController

        self.assertIsNot(HiCacheController.load, HiCacheController.start_loading)
        self.assertIn("device_tier_disarmed", inspect.getsource(HiCacheController.load))

    def test_the_hybrid_enqueue_had_the_check_and_the_consume_did_not(self):
        # Documents the exact asymmetry the sweep found, so a future reader
        # does not "simplify" the consume gate away as duplicated.
        from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
            HybridCacheController,
        )

        self.assertIn(
            "device_tier_disarmed", inspect.getsource(HybridCacheController.write)
        )


if __name__ == "__main__":
    unittest.main()
