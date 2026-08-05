"""The scheduler-side driver of the GDN slot ladder (#364 slice 2).

The executor is verified separately; this suite is about the part that is
specific to THIS scheduler: turning running batch + waiting queue into the
four id lists the planner wants, and rebinding a restored session so the batch
builder reads its NEW slot.

The last point is the one worth a test of its own. ``req.mamba_pool_idx`` is
not the only place a slot is recorded -- the batch builder reads
``req_index_to_mamba_index_mapping[req_pool_idx]``. A rebind that updates only
the request would leave it pointing at the slot another session now owns for
exactly one forward, which is not a crash but another session's recurrent
state read as this one's.

No CUDA, no server: the driver is fed hand-built request stand-ins.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.gdn_slot_runtime import GdnSlotRuntime
from sglang.test.test_utils import CustomTestCase

NUM_LAYERS = 2


class FakeSlotAllocator:
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


def _req(rid, *, arrival, slot=None, req_pool_idx=None):
    return SimpleNamespace(
        rid=rid,
        kv_arrival_seq=arrival,
        is_fast_lane=False,
        spill_class=None,
        mamba_pool_idx=slot,
        req_pool_idx=req_pool_idx,
    )


def _batch(reqs):
    return SimpleNamespace(reqs=list(reqs))


class Harness:
    def __init__(self, slots=2):
        from sglang.srt.mem_cache.memory_pool import MambaPool

        g = torch.Generator().manual_seed(11)
        conv = [torch.randn(NUM_LAYERS, slots + 1, 3, 4, generator=g)]
        temporal = torch.randn(NUM_LAYERS, slots + 1, 2, 3, generator=g)
        pool = MambaPool.__new__(MambaPool)
        pool.size = slots
        pool.mamba_cache = MambaPool.State(conv=conv, temporal=temporal)
        pool.replayssm_write_pos = None
        self.mamba_pool = pool
        self.alloc = FakeSlotAllocator(slots)
        self.mapping = torch.zeros(16, dtype=torch.int32)
        self.req_to_token_pool = SimpleNamespace(
            mamba_pool=pool,
            mamba_allocator=self.alloc,
            req_index_to_mamba_index_mapping=self.mapping,
        )
        self.idle_holders = []
        self.rt = GdnSlotRuntime(
            mamba_pool=pool,
            mamba_allocator=self.alloc,
            req_to_token_pool=self.req_to_token_pool,
            resident_slots=slots,
            idle_holders_fn=lambda: self.idle_holders,
        )

    def give_slot(self, req):
        got = self.alloc.alloc(1)
        assert got is not None
        req.mamba_pool_idx = int(got[0])
        if req.req_pool_idx is not None:
            self.mapping[req.req_pool_idx] = req.mamba_pool_idx
        return req.mamba_pool_idx


class TestInventory(CustomTestCase):
    def test_an_idle_slot_holder_vacates_for_a_queued_session(self):
        h = Harness(slots=2)
        a = _req("a", arrival=1, req_pool_idx=0)
        b = _req("b", arrival=2, req_pool_idx=1)
        h.give_slot(a)
        h.give_slot(b)
        queued = _req("c", arrival=3, req_pool_idx=2)

        self.assertEqual(h.alloc.available_size(), 0)
        # a runs; b is an idle slot holder (spilled session: holds its mamba
        # slot, has no work in the batch); c is queued and needs a slot.
        h.idle_holders = [b]
        h.rt.on_round(_batch([a]), [queued])

        self.assertIsNone(b.mamba_pool_idx)
        self.assertEqual(h.rt._executor.parked_ids, ["b"])
        self.assertEqual(h.alloc.available_size(), 1)
        # The point of the whole slice: c can now be admitted.
        self.assertIsNotNone(h.give_slot(queued))

    def test_an_active_session_is_never_the_victim(self):
        h = Harness(slots=2)
        a = _req("a", arrival=1, req_pool_idx=0)
        b = _req("b", arrival=2, req_pool_idx=1)
        h.give_slot(a)
        h.give_slot(b)
        queued = _req("c", arrival=3, req_pool_idx=2)
        h.rt.on_round(_batch([a, b]), [queued])
        # Both hold work this tick; nothing may move, and nothing did.
        self.assertEqual(a.mamba_pool_idx, 1)
        self.assertEqual(b.mamba_pool_idx, 2)
        self.assertEqual(h.alloc.available_size(), 0)
        self.assertEqual(h.rt._executor.parked_ids, [])

    def test_nothing_happens_while_the_cap_is_not_binding(self):
        h = Harness(slots=4)
        a = _req("a", arrival=1, req_pool_idx=0)
        h.give_slot(a)
        h.rt.on_round(_batch([a]), [])
        self.assertEqual(h.rt.rounds, 0)
        self.assertEqual(a.mamba_pool_idx, 1)
        self.assertEqual(h.alloc.available_size(), 3)

    def test_an_empty_scheduler_is_a_no_op(self):
        h = Harness(slots=2)
        h.rt.on_round(_batch([]), [])
        self.assertEqual(h.rt.rounds, 0)


class TestRebind(CustomTestCase):
    def test_a_restored_session_rebinds_the_req_index_mapping(self):
        h = Harness(slots=2)
        a = _req("a", arrival=1, req_pool_idx=0)
        b = _req("b", arrival=2, req_pool_idx=1)
        h.give_slot(a)
        h.give_slot(b)
        before = [t[:, b.mamba_pool_idx].clone() for t in h.mamba_pool.mamba_cache.conv]
        queued = _req("c", arrival=3, req_pool_idx=2)

        # b vacates for c.
        h.idle_holders = [b]
        h.rt.on_round(_batch([a]), [queued])
        h.give_slot(queued)
        self.assertEqual(queued.mamba_pool_idx, 2)
        self.assertEqual(int(h.mapping[2]), 2)

        # b comes back (queued while parked) and c is now the idle holder.
        h.idle_holders = [queued]
        h.rt.on_round(_batch([a]), [b])
        self.assertIsNotNone(b.mamba_pool_idx)
        self.assertIsNone(queued.mamba_pool_idx)
        # The mapping followed the rebind, not just the request object.
        self.assertEqual(int(h.mapping[b.req_pool_idx]), b.mamba_pool_idx)
        after = [t[:, b.mamba_pool_idx] for t in h.mamba_pool.mamba_cache.conv]
        for got, want in zip(after, before):
            self.assertTrue(torch.equal(got, want))

    def test_a_session_that_dies_while_parked_does_not_leak_its_blob(self):
        h = Harness(slots=2)
        a = _req("a", arrival=1, req_pool_idx=0)
        b = _req("b", arrival=2, req_pool_idx=1)
        h.give_slot(a)
        h.give_slot(b)
        h.idle_holders = [b]
        h.rt.on_round(_batch([a]), [_req("c", arrival=3, req_pool_idx=2)])
        self.assertEqual(h.rt._executor.parked_ids, ["b"])
        # b never comes back: the next round no longer knows it.
        h.idle_holders = []
        h.rt.on_round(_batch([a]), [])
        self.assertEqual(h.rt._executor.parked_ids, [])


class TestBuilder(CustomTestCase):
    def test_a_model_without_a_state_pool_is_a_named_no_op(self):
        from sglang.srt.managers.gdn_slot_runtime import build_gdn_slot_executor

        scheduler = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(mamba_pool=None, mamba_allocator=None)
        )
        self.assertIsNone(build_gdn_slot_executor(scheduler))


class TestParkedSessionsAreVisibleToTheLadder(CustomTestCase):
    """#551: a #224-PARKED session still holds its GDN state slot.

    THE CORRECTNESS GAP. ``build_gdn_slot_executor``'s ``idle_holders`` closure
    enumerated live offload sessions through ``kv_session_offload.spills``. But
    ``_commit_park`` POPS a session out of ``spills`` when it parks to a deeper
    tier, while the session stays alive and keeps its req-pool slot, its radix
    tree lock and -- the part that matters here -- its Mamba/GDN state slot.

    A parked session was therefore invisible to the ladder's inventory, and
    ``on_round`` treats an id it cannot find as DEAD::

        # A parked session that died while parked leaves a blob nobody will
        # ever ask for; drop it here rather than grow the store forever.
        for sid in parked_ids:
            if sid not in self._by_id:
                self._executor.forget(sid)

    So a session whose GDN state had been exported to a blob, and which then
    parked its KV to the next tier, had that blob DISCARDED. It later unparks,
    restores its KV, resumes -- and continues with recurrent state that no
    longer exists. No crash, no log, wrong tokens. The comment was not wrong,
    its premise was: absence from the inventory meant "dead" only as long as
    every live session was in ``spills``.

    CAN-FAIL: point ``idle_holders`` back at ``manager.spills`` and
    ``test_the_builder_sees_a_parked_session`` goes red -- the parked request
    is missing from the inventory, which is exactly the state that makes the
    ladder throw its blob away.
    """

    def _manager(self, spilled, parked):
        from sglang.srt.managers.kv_session_offload import KVSessionOffloadManager

        mgr = KVSessionOffloadManager.__new__(KVSessionOffloadManager)
        mgr.spills = {
            r.req_pool_idx: SimpleNamespace(req=r) for r in spilled
        }
        mgr._dest = SimpleNamespace(
            parked={r.rid: SimpleNamespace(slot=SimpleNamespace(req=r)) for r in parked}
        )
        return mgr

    def test_live_offload_reqs_covers_spilled_and_parked(self):
        spilled = _req("s", arrival=1, req_pool_idx=0)
        parked = _req("p", arrival=2, req_pool_idx=1)
        mgr = self._manager([spilled], [parked])
        rids = sorted(r.rid for r in mgr.live_offload_reqs())
        self.assertEqual(rids, ["p", "s"])
        # ...and `spills` alone is exactly the set that misses the parked one.
        self.assertEqual([s.req.rid for s in mgr.spills.values()], ["s"])

    def test_live_offload_reqs_without_a_destination_chain(self):
        """Unarmed chain: byte-identical to the old spills-only enumeration."""
        spilled = _req("s", arrival=1, req_pool_idx=0)
        mgr = self._manager([spilled], [])
        mgr._dest = None
        self.assertEqual([r.rid for r in mgr.live_offload_reqs()], ["s"])

    def test_the_builder_sees_a_parked_session(self):
        from sglang.srt.managers.gdn_slot_runtime import build_gdn_slot_executor

        h = Harness(slots=2)
        spilled = _req("s", arrival=1, req_pool_idx=0)
        parked = _req("p", arrival=2, req_pool_idx=1)
        h.give_slot(spilled)
        h.give_slot(parked)
        scheduler = SimpleNamespace(
            req_to_token_pool=h.req_to_token_pool,
            kv_session_offload=self._manager([spilled], [parked]),
        )
        rt = build_gdn_slot_executor(scheduler)
        self.assertIsNotNone(rt)
        rids = sorted(r.rid for r in rt._idle_holders())
        self.assertEqual(
            rids,
            ["p", "s"],
            "a parked session is missing from the ladder inventory; on_round "
            "would forget its exported GDN state blob while it is still alive",
        )

    def test_no_offload_manager_is_still_an_empty_inventory(self):
        from sglang.srt.managers.gdn_slot_runtime import build_gdn_slot_executor

        h = Harness(slots=2)
        scheduler = SimpleNamespace(
            req_to_token_pool=h.req_to_token_pool, kv_session_offload=None
        )
        rt = build_gdn_slot_executor(scheduler)
        self.assertEqual(rt._idle_holders(), [])


if __name__ == "__main__":
    unittest.main()
