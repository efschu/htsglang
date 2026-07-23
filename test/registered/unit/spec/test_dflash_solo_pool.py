"""Unit tests for T156 task D: the small DFLASH solo draft-KV pool.

CPU-only tests of the pure pieces: the global->draft slot mapper
(allocation, read/write translation, hole semantics, free-listener drain,
clear reset, LRU reclaim, readable exhaustion error) and the ctx-cap
resolution over the cross-algo shapes stash. GPU behavior (pool shrink,
corridor, DFLASH function on the small pool) is covered by the live
validation protocol, not here.
"""

import os
import types
import unittest

import torch

from sglang.srt.speculative.dflash_solo_pool import (
    SOLO_POOL_CAP_ENV,
    DraftKVSlotMapper,
    resolve_dflash_solo_pool_cap,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _mapper(num_global=1000, num_draft=16, cap=64):
    return DraftKVSlotMapper(
        num_global_slots=num_global,
        num_draft_slots=num_draft,
        ctx_cap=cap,
        device="cpu",
    )


def _t(vals):
    return torch.tensor(vals, dtype=torch.int64)


class TestMapperBasics(CustomTestCase):
    def test_write_allocates_and_read_translates(self):
        m = _mapper()
        d = m.translate_write(_t([100, 200, 300]))
        self.assertEqual(d.shape, (3,))
        # Distinct fresh globals -> distinct non-hole draft slots.
        self.assertEqual(len(set(d.tolist())), 3)
        self.assertNotIn(0, d.tolist())
        # A later read of the same globals sees the same slots.
        r = m.translate_read(_t([300, 100, 200]))
        self.assertEqual(r.tolist(), [d[2], d[0], d[1]])
        # Re-write hits the existing mapping (no growth).
        d2 = m.translate_write(_t([100, 200]))
        self.assertEqual(d2.tolist(), [d[0], d[1]])
        self.assertEqual(m.stats()["mapped"], 3)

    def test_global_slot_zero_is_hole(self):
        m = _mapper()
        self.assertEqual(m.translate_read(_t([0])).tolist(), [0])
        self.assertEqual(m.translate_write(_t([0])).tolist(), [0])

    def test_unmapped_read_returns_hole_and_counts(self):
        m = _mapper()
        m.translate_write(_t([5]))
        r = m.translate_read(_t([5, 77, 88]))
        self.assertEqual(r[1].item(), 0)
        self.assertEqual(r[2].item(), 0)
        self.assertNotEqual(r[0].item(), 0)
        self.assertEqual(m.holes_read_total, 2)

    def test_valid_mask_blocks_allocation(self):
        m = _mapper()
        locs = torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int64)
        valid = torch.tensor([[True, True, False], [False, False, False]])
        d = m.translate_write(locs, valid=valid)
        self.assertEqual(d.shape, (2, 3))
        self.assertNotEqual(d[0, 0].item(), 0)
        self.assertNotEqual(d[0, 1].item(), 0)
        # Invalid rows -> hole, nothing allocated for them.
        self.assertEqual(d[0, 2].item(), 0)
        self.assertEqual(d[1].tolist(), [0, 0, 0])
        self.assertEqual(m.stats()["mapped"], 2)

    def test_free_listener_recycles_via_drain(self):
        m = _mapper(num_draft=4)  # slots 1..3 usable
        d = m.translate_write(_t([10, 11, 12]))
        self.assertEqual(m.stats()["free"], 0)
        # Scheduler thread frees two globals; applied at the next translate.
        m.on_global_free(_t([10, 12]))
        r = m.translate_write(_t([50, 51]))  # needs the recycled slots
        self.assertNotIn(0, r.tolist())
        self.assertEqual(m.stats()["mapped"], 3)
        # The freed globals are unmapped now.
        self.assertEqual(m.translate_read(_t([10])).tolist(), [0])
        # Slot of the surviving global unchanged.
        self.assertEqual(m.translate_read(_t([11])).tolist(), [d[1].item()])

    def test_free_ignores_unmapped_and_out_of_range(self):
        m = _mapper()
        m.translate_write(_t([10]))
        m.on_global_free(_t([999999, -3, 55]))  # none mapped / out of range
        m.translate_read(_t([10]))  # drains without error
        self.assertEqual(m.stats()["mapped"], 1)

    def test_clear_resets_everything(self):
        m = _mapper(num_draft=4)
        m.translate_write(_t([10, 11, 12]))
        m.on_global_clear()
        r = m.translate_read(_t([10]))  # drains the reset
        self.assertEqual(r.tolist(), [0])
        s = m.stats()
        self.assertEqual(s["mapped"], 0)
        self.assertEqual(s["free"], 3)

    def test_lru_reclaim_prefers_oldest(self):
        m = _mapper(num_draft=5)  # 4 usable slots
        m.begin_round()
        m.translate_write(_t([1, 2]))  # round 1: old
        m.begin_round()
        m.translate_write(_t([3, 4]))  # round 2: newer
        m.begin_round()
        # Pool full; two more allocations force a reclaim of the oldest.
        d = m.translate_write(_t([5, 6]))
        self.assertNotIn(0, d.tolist())
        self.assertGreaterEqual(m.reclaim_events, 1)
        # The oldest entries (globals 1, 2) were dropped.
        self.assertEqual(m.translate_read(_t([1, 2])).tolist(), [0, 0])
        # The newer entries survive unless the sweep needed them.
        self.assertEqual(m.stats()["mapped"], 4)

    def test_exhaustion_raises_readable_error(self):
        m = _mapper(num_draft=4)
        m.begin_round()
        m.translate_write(_t([1, 2, 3]))
        # Everything was touched THIS round -> nothing reclaimable.
        with self.assertRaises(RuntimeError) as cm:
            m.translate_write(_t([4, 5]))
        self.assertIn("exhausted", str(cm.exception))

    def test_epoch_protects_current_round_from_reclaim(self):
        m = _mapper(num_draft=4)
        m.begin_round()
        m.translate_write(_t([1]))
        m.begin_round()
        m.translate_write(_t([2, 3]))  # full now
        # Same round: global 1 (older epoch) is the only candidate.
        d = m.translate_write(_t([9]))
        self.assertNotIn(0, d.tolist())
        self.assertEqual(m.translate_read(_t([1])).tolist(), [0])
        r = m.translate_read(_t([2, 3]))
        self.assertNotIn(0, r.tolist())


class _Args:
    def __init__(self, shapes=None):
        if shapes is not None:
            self.speculative_cross_shapes = shapes


class TestConsumptionGate(CustomTestCase):
    """The measured-KV consumption gate's pure verdict, incl. the T156-D
    component-shift bypass (a structural pool release must not freeze the
    correction as 'unconsumed fantasy growth')."""

    @staticmethod
    def _frozen(**kw):
        from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
            ModelRunnerKVCacheMixin,
        )

        base = dict(
            delta_b=1 << 30,
            prev_b=291 << 20,
            prev_leftover_b=4 << 30,
            free_b=8 << 30,
            component_shift=False,
        )
        base.update(kw)
        return ModelRunnerKVCacheMixin.correction_growth_frozen(**base)

    def test_unconsumed_growth_frozen(self):
        # Leftover did not shrink -> freeze (the stock rule).
        self.assertTrue(self._frozen())

    def test_consumed_growth_passes(self):
        # Leftover shrank by > 128 MiB -> previous growth consumed.
        self.assertFalse(self._frozen(free_b=3 << 30, prev_leftover_b=4 << 30))

    def test_component_shift_bypasses_freeze(self):
        # The T156-D case: pool released GiBs, leftover grew structurally.
        self.assertFalse(self._frozen(component_shift=True))

    def test_negative_delta_always_applies(self):
        self.assertFalse(self._frozen(delta_b=-(1 << 30)))

    def test_no_previous_correction_passes(self):
        self.assertFalse(self._frozen(prev_b=0))

    def test_no_previous_leftover_passes(self):
        self.assertFalse(self._frozen(prev_leftover_b=None))


class TestCapResolution(CustomTestCase):
    def test_env_off_and_explicit(self):
        os.environ[SOLO_POOL_CAP_ENV] = "off"
        try:
            cap, _ = resolve_dflash_solo_pool_cap(_Args())
            self.assertIsNone(cap)
        finally:
            del os.environ[SOLO_POOL_CAP_ENV]
        os.environ[SOLO_POOL_CAP_ENV] = "12345"
        try:
            cap, src = resolve_dflash_solo_pool_cap(_Args())
            self.assertEqual(cap, 12345)
            self.assertIn("explicit", src)
        finally:
            del os.environ[SOLO_POOL_CAP_ENV]

    def test_not_cross_algo_disabled(self):
        cap, src = resolve_dflash_solo_pool_cap(_Args())
        self.assertIsNone(cap)
        self.assertIn("full-context", src)

    def test_auto_uses_gate_threshold(self):
        args = _Args({"force": "auto", "ctx_gate": {"threshold": 8192}})
        self.assertEqual(resolve_dflash_solo_pool_cap(args)[0], 8192)
        args = _Args({"force": "auto", "ctx_gate": {"threshold": None}})
        self.assertIsNone(resolve_dflash_solo_pool_cap(args)[0])

    def test_policy_stage_bound(self):
        table = [(0, ("dflash", 16)), (4096, ("nextn", 3))]
        args = _Args(
            {
                "force": "policy",
                "policy_table": table,
                "ctx_gate": {"threshold": 8192},
            }
        )
        self.assertEqual(resolve_dflash_solo_pool_cap(args)[0], 4096)

    def test_policy_gate_caps_stage_bound(self):
        # Stage runs to 8192 but the gate fences at 6000 -> cap 6000.
        table = [(0, ("dflash", 16)), (8192, ("nextn", 3))]
        args = _Args(
            {
                "force": "policy",
                "policy_table": table,
                "ctx_gate": {"threshold": 6000},
            }
        )
        self.assertEqual(resolve_dflash_solo_pool_cap(args)[0], 6000)

    def test_policy_unbounded_stage_disables(self):
        # DFLASH is the LAST stage and the gate is off -> no cap.
        table = [(0, ("nextn", 3)), (4096, ("dflash", 16))]
        args = _Args(
            {
                "force": "policy",
                "policy_table": table,
                "ctx_gate": {"threshold": None},
            }
        )
        self.assertIsNone(resolve_dflash_solo_pool_cap(args)[0])
        # With a gate, the gate bounds the trailing stage.
        args = _Args(
            {
                "force": "policy",
                "policy_table": table,
                "ctx_gate": {"threshold": 8192},
            }
        )
        self.assertEqual(resolve_dflash_solo_pool_cap(args)[0], 8192)

    def test_policy_without_dflash_stage_minimal(self):
        table = [(0, ("nextn", 3))]
        args = _Args(
            {
                "force": "policy",
                "policy_table": table,
                "ctx_gate": {"threshold": 8192},
            }
        )
        self.assertEqual(resolve_dflash_solo_pool_cap(args)[0], 0)

    def test_static_and_schedule_disabled(self):
        for force in ("dflash", "nextn", "schedule"):
            args = _Args({"force": force, "ctx_gate": {"threshold": 8192}})
            self.assertIsNone(resolve_dflash_solo_pool_cap(args)[0])


if __name__ == "__main__":
    unittest.main()
