"""Task #364: a cap on RESIDENT GDN/Mamba state slots, with idle-vacate.

Four questions, each its own falsifier:

1. Is the cap respected -- does the pool get built with the capped slot
   count, and do the bytes the cap keeps out actually stay out?
2. Does ONLY an idle session vacate? An active session losing its recurrent
   state is not a slowdown, it is wrong output, so this is the test that
   decides whether the feature may ship at all.
3. Does a state blob survive the round trip BIT-identically, including into
   a DIFFERENT slot than it came from (that is the whole point of vacating)?
4. Is the default path -- cap unset -- byte-identical to today?

The blob test drives the real ``MambaPool`` methods against synthetic CPU
tensors rather than a re-implementation, so a field the pool starts carrying
and the blob does not carry fails here instead of in production.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.gdn_slot_ladder import (
    cap_is_binding,
    effective_state_slots,
    freed_state_bytes,
    vacate_plan,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")


def _session(sid, *, arrival, fast=False, spill_class=None):
    """A stand-in carrying exactly the #242 protection fields."""
    return SimpleNamespace(
        session_id=sid,
        kv_arrival_seq=arrival,
        is_fast_lane=fast,
        spill_class=spill_class,
    )


class TestResidentCap(CustomTestCase):
    def test_cap_lowers_the_slot_count(self):
        self.assertEqual(effective_state_slots(64, 4), 4)
        self.assertEqual(effective_state_slots(64, 1), 1)

    def test_cap_above_the_profiled_count_is_a_ceiling_not_a_demand(self):
        # Growing the pool past what the budget profiled is how a boot turns
        # into an OOM; the flag may only lower.
        self.assertEqual(effective_state_slots(8, 64), 8)

    def test_cap_unset_is_the_identity(self):
        self.assertEqual(effective_state_slots(64, None), 64)
        self.assertFalse(cap_is_binding(None, 64))
        self.assertFalse(cap_is_binding(64, 64))
        self.assertTrue(cap_is_binding(4, 64))

    def test_a_zero_cap_is_rejected(self):
        with self.assertRaises(ValueError):
            effective_state_slots(64, 0)

    def test_freed_bytes_are_the_slots_the_cap_keeps_out(self):
        # 64 profiled slots, cap 4, 3 MiB per set -> 60 sets never allocated.
        per_set = 3 << 20
        self.assertEqual(freed_state_bytes(64, 4, per_set), 60 * per_set)
        # Not binding -> nothing freed, and nothing to reconcile.
        self.assertEqual(freed_state_bytes(64, None, per_set), 0)
        self.assertEqual(freed_state_bytes(64, 64, per_set), 0)


class TestIdleOnlyVacate(CustomTestCase):
    def test_an_active_session_never_vacates(self):
        # FALSIFIER for the correctness half. Four sessions, cap 2, and the
        # two that would be picked by pure youth are the ACTIVE ones.
        sessions = [_session(f"s{i}", arrival=i) for i in range(4)]
        plan = vacate_plan(
            resident_slots=2,
            sessions=sessions,
            active_ids=["s2", "s3"],
        )
        self.assertNotIn("s2", plan.vacate)
        self.assertNotIn("s3", plan.vacate)
        self.assertEqual(plan.skipped["s2"], "active session (has work in this tick)")
        self.assertEqual(sorted(plan.vacate), ["s0", "s1"])

    def test_vacate_order_is_the_242_protection_order(self):
        # Youngest (highest arrival_seq) is least protected and goes first.
        sessions = [_session(f"s{i}", arrival=i) for i in range(5)]
        plan = vacate_plan(resident_slots=2, sessions=sessions, active_ids=[])
        self.assertEqual(plan.vacate[:3], ["s4", "s3", "s2"])

    def test_a_fast_lane_session_outranks_every_normal_one(self):
        sessions = [
            _session("normal_old", arrival=0),
            _session("fast_young", arrival=99, fast=True),
        ]
        plan = vacate_plan(resident_slots=1, sessions=sessions, active_ids=[])
        self.assertEqual(plan.vacate, ["normal_old"])

    def test_a_never_class_session_sorts_last(self):
        sessions = [
            _session("never_young", arrival=99, spill_class="never"),
            _session("normal_old", arrival=0),
        ]
        plan = vacate_plan(resident_slots=1, sessions=sessions, active_ids=[])
        self.assertEqual(plan.vacate, ["normal_old"])

    def test_resume_is_planned_before_the_next_tick_and_is_never_a_victim(self):
        sessions = [_session(f"s{i}", arrival=i) for i in range(3)]
        plan = vacate_plan(
            resident_slots=2,
            sessions=sessions,
            active_ids=[],
            resumed_ids=["s2"],
            parked_ids=["s2"],
        )
        self.assertEqual(plan.restore, ["s2"])
        self.assertNotIn("s2", plan.vacate)
        self.assertEqual(plan.skipped["s2"], "resuming this tick")

    def test_no_overshoot_means_no_movement(self):
        sessions = [_session(f"s{i}", arrival=i) for i in range(2)]
        plan = vacate_plan(resident_slots=4, sessions=sessions, active_ids=[])
        self.assertTrue(plan.is_empty)

    def test_all_sessions_active_reports_why_nothing_freed(self):
        # A shortfall must name the refusing sessions, not just fail.
        sessions = [_session(f"s{i}", arrival=i) for i in range(3)]
        plan = vacate_plan(
            resident_slots=1,
            sessions=sessions,
            active_ids=["s0", "s1", "s2"],
        )
        self.assertEqual(plan.vacate, [])
        self.assertEqual(len(plan.skipped), 3)


class TestStateBlobRoundTrip(CustomTestCase):
    """The blob must come back bit-identical, into ANY free slot."""

    num_layers = 3
    num_slots = 5  # size 4 + the dummy slot 0

    def build_pool(self, *, with_replayssm: bool):
        """A MambaPool-shaped stand-in built from real torch tensors.

        The pool's own __init__ needs cache params and a device allocator, so
        the state container is assembled directly and the REAL methods are
        bound to it -- the methods are what this test is about.
        """
        from sglang.srt.mem_cache.memory_pool import MambaPool

        g = torch.Generator().manual_seed(1234)
        conv = [
            torch.randn(self.num_layers, self.num_slots, 4, 7, generator=g)
            for _ in range(2)
        ]
        temporal = torch.randn(self.num_layers, self.num_slots, 3, 5, generator=g)
        kw = {}
        if with_replayssm:
            kw = dict(
                replayssm_d=torch.randn(
                    self.num_layers, self.num_slots, 2, 6, 4, generator=g
                ),
                replayssm_k=torch.randn(
                    self.num_layers, self.num_slots, 2, 6, 3, generator=g
                ),
                replayssm_g=torch.randn(
                    self.num_layers, self.num_slots, 2, 6, generator=g
                ),
            )
        pool = MambaPool.__new__(MambaPool)
        pool.size = self.num_slots - 1
        pool.mamba_cache = MambaPool.State(conv=conv, temporal=temporal, **kw)
        pool.replayssm_write_pos = (
            torch.arange(self.num_slots, dtype=torch.int32)
            if with_replayssm
            else None
        )
        return pool

    def test_round_trip_into_the_same_slot_is_bit_identical(self):
        for with_replayssm in (False, True):
            with self.subTest(replayssm=with_replayssm):
                pool = self.build_pool(with_replayssm=with_replayssm)
                slot = 2
                before = [t[:, slot].clone() for t in pool.mamba_cache.conv]
                before.append(pool.mamba_cache.temporal[:, slot].clone())
                blob = pool.export_state_blob(slot)
                # Scribble over the slot, then restore it.
                for t in pool.mamba_cache.conv:
                    t[:, slot] = 0
                pool.mamba_cache.temporal[:, slot] = 0
                pool.import_state_blob(slot, blob)
                after = [t[:, slot] for t in pool.mamba_cache.conv]
                after.append(pool.mamba_cache.temporal[:, slot])
                for b, a in zip(before, after):
                    self.assertTrue(torch.equal(b, a), "state is not bit-identical")

    def test_round_trip_into_a_DIFFERENT_slot(self):
        # Vacating is only useful if the session can come back somewhere
        # else: the slot it left is handed to another session.
        pool = self.build_pool(with_replayssm=True)
        src, dst = 1, 4
        blob = pool.export_state_blob(src)
        pool.import_state_blob(dst, blob)
        for t in pool.mamba_cache.conv:
            self.assertTrue(torch.equal(t[:, src], t[:, dst]))
        self.assertTrue(
            torch.equal(
                pool.mamba_cache.temporal[:, src], pool.mamba_cache.temporal[:, dst]
            )
        )
        self.assertTrue(
            torch.equal(
                pool.mamba_cache.replayssm_d[:, src],
                pool.mamba_cache.replayssm_d[:, dst],
            )
        )
        self.assertEqual(
            int(pool.replayssm_write_pos[dst]), int(pool.replayssm_write_pos[src])
        )

    def test_the_blob_carries_the_replayssm_cursor(self):
        # A restored ring buffer with a stale write cursor reads its own
        # history at the wrong offset -- silently wrong, not a crash.
        pool = self.build_pool(with_replayssm=True)
        self.assertIn("replayssm_write_pos", pool.state_blob_fields())
        blob = pool.export_state_blob(3)
        self.assertIn("replayssm_write_pos", blob)

    def test_transient_spec_scratch_is_not_in_the_blob(self):
        pool = self.build_pool(with_replayssm=False)
        self.assertNotIn("intermediate_ssm", pool.state_blob_fields())
        self.assertNotIn("intermediate_conv_window", pool.state_blob_fields())

    def test_a_blob_from_a_different_layout_is_refused(self):
        # Restoring a partial recurrent state is silently wrong output.
        pool = self.build_pool(with_replayssm=True)
        blob = pool.export_state_blob(1)
        blob.pop("replayssm_g")
        with self.assertRaises(ValueError):
            pool.import_state_blob(2, blob)

    def test_the_dummy_slot_never_vacates(self):
        pool = self.build_pool(with_replayssm=False)
        with self.assertRaises(ValueError):
            pool.export_state_blob(0)
        with self.assertRaises(ValueError):
            pool.export_state_blob(pool.size + 1)


class TestDefaultPathUnchanged(CustomTestCase):
    def test_server_arg_defaults_to_unset(self):
        # Read the dataclass field rather than constructing ServerArgs: a
        # full construction resolves a device and cannot run CPU-only.
        from sglang.srt.server_args import ServerArgs

        field = ServerArgs.__dataclass_fields__["gdn_resident_state_slots"]
        self.assertIsNone(field.default)

    def test_a_zero_cap_is_rejected_at_argument_time(self):
        from sglang.srt.server_args import ServerArgs

        args = ServerArgs.__new__(ServerArgs)
        args.gdn_resident_state_slots = 0
        args.gdn_state_set_ladder = None
        args.gdn_state_set_ladder_hysteresis = 2
        with self.assertRaises(ValueError):
            ServerArgs._handle_gdn_state_set_ladder(args)
        # ...and a valid cap passes the same handler.
        args.gdn_resident_state_slots = 4
        ServerArgs._handle_gdn_state_set_ladder(args)

    def test_unset_cap_leaves_the_sizing_untouched(self):
        for profiled in (1, 8, 64, 4096):
            self.assertEqual(effective_state_slots(profiled, None), profiled)
            self.assertEqual(freed_state_bytes(profiled, None, 1 << 20), 0)

    def test_unset_cap_plans_nothing_even_when_sessions_exceed_the_pool(self):
        sessions = [_session(f"s{i}", arrival=i) for i in range(9)]
        plan = vacate_plan(
            resident_slots=effective_state_slots(64, None),
            sessions=sessions,
            active_ids=[],
        )
        self.assertTrue(plan.is_empty)


if __name__ == "__main__":
    unittest.main()
