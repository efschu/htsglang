"""The between-tick executor of the GDN slot ladder (#364 slice 2).

Slice 1's honest remainder, stated as a falsifier first: with only the shipped
slice-1 code, a capped server PLANS a vacate every round and still refuses the
session the plan was for. ``test_slice1_plans_but_never_admits`` is that red
arm -- it drives ``vacate_plan`` exactly as slice 1 offers it, and asserts the
allocator is untouched and the waiting session cannot be admitted. The green
arm runs the same scenario through :class:`GdnSlotExecutor` and admits it.

The rest pins what makes the executor safe rather than merely functional:

* it refuses to run outside the armed between-tick window (#52/#53: the pool
  is graph-addressed, so a wrongly-placed call is silent corruption),
* vacates run before restores, so a FULL pool can still resume a session,
* every failure names the session and the cause,
* the blob keeps its identity through the #224 flat form, and a tier that
  refuses a put fails over to local instead of losing state.

Real ``MambaPool.export_state_blob`` / ``import_state_blob`` against real
torch tensors on CPU; the allocator is a plain-int stand-in with the
``MambaSlotAllocator`` surface the executor actually uses. No CUDA, no server.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.gdn_slot_executor import (
    GdnSlotError,
    GdnSlotExecutor,
    GdnSlotWindowError,
    LocalGdnBlobStore,
    TieredGdnBlobStore,
    flatten_blob,
    unflatten_blob,
)
from sglang.srt.mem_cache.gdn_slot_ladder import vacate_plan
from sglang.test.test_utils import CustomTestCase

NUM_LAYERS = 3


def _session(sid, *, arrival, fast=False, spill_class=None):
    """Carries exactly the #242 protection fields, as in the slice-1 suite."""
    return SimpleNamespace(
        session_id=sid,
        kv_arrival_seq=arrival,
        is_fast_lane=fast,
        spill_class=spill_class,
    )


class FakeSlotAllocator:
    """The slice of ``MambaSlotAllocator`` the executor touches.

    Plain ints on purpose: the executor's tensor adaptation is one of the
    things under test, and a CPU suite must not need a device string that
    only means something with CUDA present.
    """

    device = "cpu"

    def __init__(self, size):
        self.size = size
        self.free_slots = list(range(1, size + 1))

    def alloc(self, n):
        if n > len(self.free_slots):
            return None
        got, self.free_slots = self.free_slots[:n], self.free_slots[n:]
        return got

    def free(self, idx):
        if hasattr(idx, "tolist"):
            idx = idx.tolist()
        self.free_slots.extend(int(i) for i in idx)

    def available_size(self):
        return len(self.free_slots)


class Fixture:
    """A capped pool plus the session->slot bookkeeping the scheduler owns."""

    def __init__(self, cap=2, num_slots=None):
        from sglang.srt.mem_cache.memory_pool import MambaPool

        slots = num_slots if num_slots is not None else cap
        g = torch.Generator().manual_seed(7)
        conv = [torch.randn(NUM_LAYERS, slots + 1, 4, 7, generator=g) for _ in range(2)]
        temporal = torch.randn(NUM_LAYERS, slots + 1, 3, 5, generator=g)
        pool = MambaPool.__new__(MambaPool)
        pool.size = slots
        pool.mamba_cache = MambaPool.State(conv=conv, temporal=temporal)
        pool.replayssm_write_pos = None
        self.pool = pool
        self.cap = cap
        self.alloc = FakeSlotAllocator(slots)
        self.slots = {}  # session id -> slot

    def admit(self, sid):
        """What the scheduler does when a session gets a state slot."""
        got = self.alloc.alloc(1)
        if got is None:
            return None
        self.slots[sid] = int(got[0])
        return self.slots[sid]

    def executor(self, store=None):
        return GdnSlotExecutor(
            mamba_pool=self.pool,
            mamba_allocator=self.alloc,
            slot_of=lambda sid: self.slots.get(sid),
            bind=lambda sid, slot: self.slots.__setitem__(sid, slot),
            unbind=lambda sid: self.slots.pop(sid, None),
            blob_store=store,
        )

    def slot_state(self, slot):
        return [t[:, slot].clone() for t in self.pool.mamba_cache.conv] + [
            self.pool.mamba_cache.temporal[:, slot].clone()
        ]


class TestSliceOneCannotAdmit(CustomTestCase):
    """The red arm: planning is not admitting."""

    def test_slice1_plans_but_never_admits(self):
        fx = Fixture(cap=2)
        sessions = [_session("s1", arrival=1), _session("s2", arrival=2)]
        self.assertEqual(fx.admit("s1"), 1)
        self.assertEqual(fx.admit("s2"), 2)

        # s3 arrives; s2 is idle this tick, so the ladder has a candidate.
        sessions.append(_session("s3", arrival=3))
        plan = vacate_plan(
            resident_slots=fx.cap,
            sessions=sessions,
            active_ids=["s1", "s3"],
        )
        # Slice 1 does its job: it names the victim.
        self.assertEqual(plan.vacate, ["s2"])

        # ...and that is ALL it does. Nothing was freed, so the session the
        # plan exists for is still refused. This is the remainder slice 1
        # reported, reproduced as a test rather than as prose.
        self.assertEqual(fx.alloc.available_size(), 0)
        self.assertIsNone(fx.admit("s3"))
        self.assertNotIn("s3", fx.slots)
        self.assertEqual(fx.slots, {"s1": 1, "s2": 2})

    def test_the_executor_admits_the_session_the_plan_was_for(self):
        """The green arm: same scenario, same plan, executed."""
        fx = Fixture(cap=2)
        sessions = [
            _session("s1", arrival=1),
            _session("s2", arrival=2),
            _session("s3", arrival=3),
        ]
        fx.admit("s1")
        fx.admit("s2")
        victim_state = fx.slot_state(2)

        plan = vacate_plan(
            resident_slots=fx.cap, sessions=sessions, active_ids=["s1", "s3"]
        )
        ex = fx.executor()
        with ex.between_ticks():
            result = ex.run(plan)

        self.assertEqual(result.vacated, ["s2"])
        self.assertEqual(result.failed, {})
        self.assertEqual(fx.alloc.available_size(), 1)
        self.assertNotIn("s2", fx.slots)
        self.assertEqual(ex.parked_ids, ["s2"])

        # The acceptance criterion of the whole slice.
        self.assertIsNotNone(fx.admit("s3"))
        self.assertIn("s3", fx.slots)

        # And s2's state survived the eviction untouched.
        stored = ex._store.pop("s2")
        for got, want in zip(stored["conv"], victim_state[:2]):
            self.assertTrue(torch.equal(got, want))
        self.assertTrue(torch.equal(stored["temporal"], victim_state[2]))


class TestRestore(CustomTestCase):
    def test_a_parked_session_comes_back_bit_identical_in_another_slot(self):
        fx = Fixture(cap=2)
        fx.admit("s1")
        fx.admit("s2")
        before = fx.slot_state(2)
        sessions = [
            _session("s1", arrival=1),
            _session("s2", arrival=2),
            _session("s3", arrival=3),
        ]
        ex = fx.executor()
        with ex.between_ticks():
            ex.run(
                vacate_plan(
                    resident_slots=fx.cap, sessions=sessions, active_ids=["s1", "s3"]
                )
            )
        fx.admit("s3")  # s3 takes the freed slot 2
        self.assertEqual(fx.slots["s3"], 2)
        # Scribble, so a "restore" that does nothing cannot pass.
        for t in fx.pool.mamba_cache.conv:
            t[:, 2] = 0
        fx.pool.mamba_cache.temporal[:, 2] = 0

        # s2 resumes; s3 is now the idle one and pays for it.
        sessions = [
            _session("s1", arrival=1),
            _session("s2", arrival=2),
            _session("s3", arrival=3),
        ]
        plan = vacate_plan(
            resident_slots=fx.cap,
            sessions=sessions,
            active_ids=["s1", "s2"],
            resumed_ids=["s2"],
            parked_ids=ex.parked_ids,
        )
        self.assertEqual(plan.restore, ["s2"])
        with ex.between_ticks():
            result = ex.run(plan)
        self.assertEqual(result.restored, ["s2"])
        self.assertEqual(result.failed, {})
        slot = fx.slots["s2"]
        after = fx.slot_state(slot)
        for got, want in zip(after, before):
            self.assertTrue(torch.equal(got, want))

    def test_vacates_run_before_restores_so_a_full_pool_can_resume(self):
        """Reverse order would deadlock: every restore needs a slot and the
        only slots are the ones the vacates have not released yet."""
        fx = Fixture(cap=2)
        fx.admit("s1")
        fx.admit("s2")
        ex = fx.executor()
        sessions = [_session("s1", arrival=1), _session("s2", arrival=2)]
        with ex.between_ticks():
            ex.run(vacate_plan(resident_slots=1, sessions=sessions, active_ids=["s1"]))
        self.assertEqual(ex.parked_ids, ["s2"])
        fx.admit("s3")  # pool full again
        self.assertEqual(fx.alloc.available_size(), 0)

        sessions = [
            _session("s1", arrival=1),
            _session("s2", arrival=2),
            _session("s3", arrival=3),
        ]
        plan = vacate_plan(
            resident_slots=2,
            sessions=sessions,
            active_ids=["s1", "s2"],
            resumed_ids=["s2"],
            parked_ids=ex.parked_ids,
        )
        self.assertEqual(plan.restore, ["s2"])
        self.assertEqual(plan.vacate, ["s3"])
        with ex.between_ticks():
            result = ex.run(plan)
        self.assertEqual(result.vacated, ["s3"])
        self.assertEqual(result.restored, ["s2"])
        self.assertEqual(result.failed, {})


class TestGraphSafety(CustomTestCase):
    """#52/#53: the pool is graph-addressed, so placement is correctness."""

    def _plan_with_one_vacate(self, fx):
        return vacate_plan(
            resident_slots=1,
            sessions=[_session("s1", arrival=1), _session("s2", arrival=2)],
            active_ids=["s1"],
        )

    def test_run_outside_the_window_is_refused(self):
        fx = Fixture(cap=2)
        fx.admit("s1")
        fx.admit("s2")
        ex = fx.executor()
        plan = self._plan_with_one_vacate(fx)
        self.assertFalse(ex.armed)
        with self.assertRaises(GdnSlotWindowError) as cm:
            ex.run(plan)
        msg = str(cm.exception)
        self.assertIn("between-tick window", msg)
        self.assertIn("#52/#53", msg)
        # Refused means REFUSED: nothing moved.
        self.assertEqual(fx.alloc.available_size(), 0)
        self.assertEqual(fx.slots, {"s1": 1, "s2": 2})

    def test_the_window_disarms_even_when_the_body_raises(self):
        fx = Fixture(cap=2)
        ex = fx.executor()
        with self.assertRaises(ValueError):
            with ex.between_ticks():
                self.assertTrue(ex.armed)
                raise ValueError("boom")
        self.assertFalse(ex.armed)
        with self.assertRaises(GdnSlotWindowError):
            ex.run(self._plan_with_one_vacate(fx))

    def test_a_nested_window_is_refused(self):
        ex = Fixture(cap=2).executor()
        with ex.between_ticks():
            with self.assertRaises(GdnSlotWindowError):
                with ex.between_ticks():
                    pass


class TestFailuresAreNamed(CustomTestCase):
    def test_vacating_a_session_that_holds_no_slot_names_it(self):
        fx = Fixture(cap=2)
        ex = fx.executor()
        plan = SimpleNamespace(vacate=["ghost"], restore=[], skipped={})
        with ex.between_ticks():
            result = ex.run(plan)
        self.assertEqual(result.vacated, [])
        self.assertIn("ghost", result.failed)
        self.assertIn("holds no", result.failed["ghost"])

    def test_restoring_without_a_blob_is_refused_not_zero_filled(self):
        fx = Fixture(cap=2)
        ex = fx.executor()
        plan = SimpleNamespace(vacate=[], restore=["s9"], skipped={})
        with ex.between_ticks():
            result = ex.run(plan)
        self.assertEqual(result.restored, [])
        self.assertIn("no parked blob", result.failed["s9"])

    def test_a_restore_with_no_free_slot_says_so(self):
        fx = Fixture(cap=1)
        fx.admit("s1")
        ex = fx.executor()
        ex._store.put("s2", fx.pool.export_state_blob(1))
        plan = SimpleNamespace(vacate=[], restore=["s2"], skipped={})
        with ex.between_ticks():
            result = ex.run(plan)
        self.assertIn("no resident", result.failed["s2"])
        self.assertIn("1 slots", result.failed["s2"])


class TestBlobFlatForm(CustomTestCase):
    """The #224 route: one flat buffer plus a manifest, and back."""

    def test_flatten_unflatten_is_bit_identical(self):
        fx = Fixture(cap=3)
        blob = fx.pool.export_state_blob(2)
        manifest, flat = flatten_blob(blob)
        self.assertEqual(flat.dtype, torch.uint8)
        back = unflatten_blob(manifest, flat)
        self.assertEqual(set(back), set(blob))
        for name, value in blob.items():
            if isinstance(value, list):
                for a, b in zip(back[name], value):
                    self.assertTrue(torch.equal(a, b))
            else:
                self.assertTrue(torch.equal(back[name], value))

    def test_a_short_buffer_is_refused(self):
        fx = Fixture(cap=3)
        manifest, flat = flatten_blob(fx.pool.export_state_blob(2))
        with self.assertRaises(GdnSlotError) as cm:
            unflatten_blob(manifest, flat[: flat.numel() // 2])
        self.assertIn("silently wrong output", str(cm.exception))

    def test_the_flat_blob_still_imports_into_the_pool(self):
        fx = Fixture(cap=3)
        before = fx.slot_state(2)
        manifest, flat = flatten_blob(fx.pool.export_state_blob(2))
        for t in fx.pool.mamba_cache.conv:
            t[:, 3] = 0
        fx.pool.mamba_cache.temporal[:, 3] = 0
        fx.pool.import_state_blob(3, unflatten_blob(manifest, flat))
        for got, want in zip(fx.slot_state(3), before):
            self.assertTrue(torch.equal(got, want))


class _FakeTier:
    name = "fake"

    def __init__(self, accept=True):
        self.accept = accept
        self.blobs = {}

    def put(self, key, tensor):
        if not self.accept:
            return False
        self.blobs[key] = tensor.clone()
        return True

    def get_into(self, key, tensor):
        if key not in self.blobs:
            return False
        tensor.copy_(self.blobs[key])
        return True


class TestTieredStore(CustomTestCase):
    def test_a_blob_round_trips_through_a_tier(self):
        fx = Fixture(cap=2)
        tier = _FakeTier()
        store = TieredGdnBlobStore(tier, key_fn=lambda sid: f"gdn/{sid}")
        fx.admit("s1")
        fx.admit("s2")
        before = fx.slot_state(2)
        ex = fx.executor(store=store)
        plan = SimpleNamespace(vacate=["s2"], restore=[], skipped={})
        with ex.between_ticks():
            ex.run(plan)
        self.assertIn("gdn/s2", tier.blobs)
        self.assertEqual(ex.parked_ids, ["s2"])
        for t in fx.pool.mamba_cache.conv:
            t[:, 2] = 0
        fx.pool.mamba_cache.temporal[:, 2] = 0
        with ex.between_ticks():
            result = ex.run(SimpleNamespace(vacate=[], restore=["s2"], skipped={}))
        self.assertEqual(result.restored, ["s2"])
        for got, want in zip(fx.slot_state(fx.slots["s2"]), before):
            self.assertTrue(torch.equal(got, want))

    def test_a_tier_that_refuses_the_put_falls_back_to_local(self):
        fx = Fixture(cap=2)
        tier = _FakeTier(accept=False)
        store = TieredGdnBlobStore(tier, key_fn=lambda sid: f"gdn/{sid}")
        fx.admit("s1")
        fx.admit("s2")
        before = fx.slot_state(2)
        ex = fx.executor(store=store)
        with ex.between_ticks():
            ex.run(SimpleNamespace(vacate=["s2"], restore=[], skipped={}))
        self.assertEqual(tier.blobs, {})
        self.assertTrue(store.has("s2"))
        with ex.between_ticks():
            ex.run(SimpleNamespace(vacate=[], restore=["s2"], skipped={}))
        for got, want in zip(fx.slot_state(fx.slots["s2"]), before):
            self.assertTrue(torch.equal(got, want))

    def test_a_lost_tier_blob_raises_instead_of_resuming_on_garbage(self):
        fx = Fixture(cap=2)
        tier = _FakeTier()
        store = TieredGdnBlobStore(tier, key_fn=lambda sid: f"gdn/{sid}")
        fx.admit("s1")
        fx.admit("s2")
        ex = fx.executor(store=store)
        with ex.between_ticks():
            ex.run(SimpleNamespace(vacate=["s2"], restore=[], skipped={}))
        tier.blobs.clear()  # the tier lost it
        with ex.between_ticks():
            result = ex.run(SimpleNamespace(vacate=[], restore=["s2"], skipped={}))
        self.assertIn("not readable from tier", result.failed["s2"])


class TestDefaultPathUnchanged(CustomTestCase):
    def test_an_empty_plan_moves_nothing_and_takes_no_window_work(self):
        fx = Fixture(cap=4)
        fx.admit("s1")
        ex = fx.executor()
        plan = vacate_plan(
            resident_slots=4,
            sessions=[_session("s1", arrival=1)],
            active_ids=["s1"],
        )
        self.assertTrue(plan.is_empty)
        with ex.between_ticks():
            result = ex.run(plan)
        self.assertEqual(result.moved, 0)
        self.assertEqual(fx.slots, {"s1": 1})
        self.assertEqual(fx.alloc.available_size(), 3)
        self.assertEqual(ex.parked_ids, [])

    def test_forget_drops_a_dead_session_blob(self):
        store = LocalGdnBlobStore()
        store.put("s1", {"conv": []})
        fx = Fixture(cap=2)
        ex = fx.executor(store=store)
        self.assertEqual(ex.parked_ids, ["s1"])
        ex.forget("s1")
        self.assertEqual(ex.parked_ids, [])
        ex.forget("s1")  # idempotent


if __name__ == "__main__":
    unittest.main()
