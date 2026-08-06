"""#616: request-pool slot ownership across the streaming-session lifecycle.

A streaming session parks a request's KV in a ``SessionSlot`` between turns.
``SessionSlot.save_from_req`` moves ownership of ``req_pool_idx`` from the
request to the slot; ``restore_to_req`` lends it to the next turn's request but
DELIBERATELY keeps its own copy, so a scheduler retry can restore again
idempotently.

That leaves the row named twice. The only thing keeping the session from
returning a row a live request still holds is the session's in-flight flag,
which makes ``SessionController._close`` defer the release. ``find_active_slot``
clears exactly that flag when it detaches a pre-aborted request -- after the row
may already have been lent out. The release then stops deferring, the slot
returns the row, and the detached request returns it again on its own abort
path. The free list carries the row twice and hands it to two concurrent
requests, while ``available_size()`` reports more rows than the pool owns.

These tests drive the real ``ReqToTokenPool``, the real ``SessionSlot`` and the
real ``StreamingSession`` bodies -- no reimplementation of the logic under test.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.session.streaming_session import SessionSlot, StreamingSession


def _make_pool(size: int = 4, max_context_len: int = 16) -> ReqToTokenPool:
    return ReqToTokenPool(
        size=size,
        max_context_len=max_context_len,
        device="cpu",
        enable_memory_saver=False,
    )


class _FakeSession:
    """The slice of Session the streaming slot lifecycle touches."""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.streaming = True
        self.inflight = False

    def abort_req(self):
        self.inflight = False


def _make_req(session=None, **kwargs):
    base = dict(
        req_pool_idx=None,
        kv_committed_len=0,
        kv_allocated_len=0,
        swa_evicted_seqlen=0,
        last_node=None,
        cache_protected_len=0,
        swa_uuid_for_lock=None,
        mamba_pool_idx=None,
        mamba_ping_pong_track_buffer=None,
        mamba_next_track_idx=None,
        mamba_last_track_seqlen=None,
        mamba_branching_seqlen=None,
        inflight_middle_chunks=0,
        to_finish=None,
        session=session,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


class _FreeRecordingAllocator:
    def __init__(self):
        self.freed = []

    def free(self, indices):
        self.freed.append(indices)


def _make_session_cache(pool):
    inner = SimpleNamespace(
        req_to_token_pool=pool,
        token_to_kv_pool_allocator=_FreeRecordingAllocator(),
        page_size=1,
        dec_lock_ref=lambda node, params=None: None,
    )
    return StreamingSession(inner)


class ReqSlotOwnershipTest(unittest.TestCase):
    def test_pool_hands_out_only_in_range_rows(self):
        """Baseline: an untouched pool only ever yields rows it owns."""
        pool = _make_pool(size=4)
        reqs = [_make_req() for _ in range(4)]
        got = pool.alloc(reqs)
        self.assertIsNotNone(got)
        for idx in got:
            self.assertGreater(idx, 0, "row 0 is the cuda-graph padding row")
            self.assertLess(idx, pool.req_to_token.shape[0])
        self.assertEqual(len(set(got)), len(got), "rows must be unique")

    def test_pre_abort_detach_of_a_lent_row_returns_it_once(self):
        """#616 root: the reachable double return, driven end to end.

        1. turn 1 takes a row and finishes -> the slot parks it
        2. turn 2 is created (session in flight) and an earlier scheduling
           cycle lends it the row via restore_to_req
        3. turn 2 is pre-aborted; find_active_slot detaches it, clearing the
           in-flight flag that made the close defer
        4. the session closes -> release_session
        5. turn 2 is torn down on the normal abort path -> pool.free

        The row must reach the free list exactly once.
        """
        pool = _make_pool(size=4)
        cache = _make_session_cache(pool)
        session = _FakeSession("s0")
        session_id = session.session_id

        # 1. turn 1 takes a row and parks it in the slot.
        turn1 = _make_req(session=session)
        pool.alloc([turn1])
        row = turn1.req_pool_idx
        slot = SessionSlot()
        cache.slots[session_id] = slot
        slot.save_from_req(turn1, is_first=True)
        self.assertEqual(slot.req_pool_idx, row)

        # 2. turn 2 is in flight and an earlier cycle lent it the row.
        turn2 = _make_req(session=session)
        session.inflight = True
        slot.restore_to_req(turn2)
        self.assertEqual(turn2.req_pool_idx, row)

        # 3. turn 2 is pre-aborted and detached.
        turn2.to_finish = object()
        self.assertIsNone(
            cache.find_active_slot(turn2),
            "a pre-aborted req must be detached, not handed the slot",
        )
        self.assertFalse(
            session.inflight,
            "detach clears the in-flight flag -- the close no longer defers",
        )

        # 4. the session closes now that nothing defers it.
        cache.release_session(session_id)

        # 5. the detached req is torn down on the normal path.
        self.assertEqual(turn2.req_pool_idx, row, "the detached req still owns the row")
        pool.free(turn2)

        occurrences = pool.free_slots.count(row)
        self.assertEqual(
            occurrences,
            1,
            f"row {row} reached the free list {occurrences} times",
        )

    def test_detached_row_is_not_handed_to_two_requests(self):
        """The consequence the root fix removes: one row, two requests."""
        pool = _make_pool(size=4)
        cache = _make_session_cache(pool)
        session = _FakeSession("s0")

        turn1 = _make_req(session=session)
        pool.alloc([turn1])
        slot = SessionSlot()
        cache.slots[session.session_id] = slot
        slot.save_from_req(turn1, is_first=True)

        turn2 = _make_req(session=session)
        session.inflight = True
        slot.restore_to_req(turn2)
        turn2.to_finish = object()
        cache.find_active_slot(turn2)
        cache.release_session(session.session_id)
        pool.free(turn2)

        drained = []
        while pool.available_size() > 0:
            r = _make_req()
            if pool.alloc([r]) is None:
                break
            drained.append(r.req_pool_idx)

        duplicates = {i for i in drained if drained.count(i) > 1}
        self.assertEqual(
            duplicates,
            set(),
            f"rows {sorted(duplicates)} were handed to more than one request",
        )

    def test_free_list_never_exceeds_the_pool(self):
        """A free list longer than the pool is proof of a double return."""
        pool = _make_pool(size=4)
        cache = _make_session_cache(pool)

        for n in range(3):
            session = _FakeSession(f"s{n}")
            turn1 = _make_req(session=session)
            pool.alloc([turn1])
            slot = SessionSlot()
            cache.slots[session.session_id] = slot
            slot.save_from_req(turn1, is_first=True)
            turn2 = _make_req(session=session)
            session.inflight = True
            slot.restore_to_req(turn2)
            turn2.to_finish = object()
            cache.find_active_slot(turn2)
            cache.release_session(session.session_id)
            pool.free(turn2)

        self.assertLessEqual(
            len(pool.free_slots),
            pool.size,
            f"free list holds {len(pool.free_slots)} entries for a "
            f"{pool.size}-row pool: {pool.free_slots}",
        )

    def test_slot_keeps_its_row_when_the_request_never_borrowed_it(self):
        """The relinquish must be narrow: an unrelated abort keeps the park.

        A pre-aborted request that never took the slot's row (a fresh first
        turn, or one the scheduler never restored) leaves the slot intact, so
        the session's parked KV survives for the next turn.
        """
        pool = _make_pool(size=4)
        cache = _make_session_cache(pool)
        session = _FakeSession("s0")

        parked = _make_req(session=session)
        pool.alloc([parked])
        row = parked.req_pool_idx
        slot = SessionSlot()
        cache.slots[session.session_id] = slot
        slot.save_from_req(parked, is_first=True)

        # A different request on the same session, never lent the row.
        other = _make_req(session=session, to_finish=object())
        cache.find_active_slot(other)

        self.assertEqual(
            slot.req_pool_idx,
            row,
            "the slot must keep a row the aborted request never held",
        )


class FreeSlotRefusalTest(unittest.TestCase):
    """The pool refuses a corrupt return by name instead of absorbing it."""

    def test_double_return_is_refused(self):
        pool = _make_pool(size=4)
        req = _make_req()
        pool.alloc([req])
        row = req.req_pool_idx
        pool.free_slot(row)
        with self.assertRaises(ValueError) as ctx:
            pool.free_slot(row)
        self.assertIn(str(row), str(ctx.exception))
        self.assertIn("already free", str(ctx.exception))

    def test_out_of_range_return_is_refused(self):
        pool = _make_pool(size=4)
        with self.assertRaises(ValueError) as ctx:
            pool.free_slot(pool._alloc_size)
        self.assertIn("0..", str(ctx.exception))

    def test_negative_return_is_refused(self):
        pool = _make_pool(size=4)
        with self.assertRaises(ValueError):
            pool.free_slot(-1)

    def test_null_return_is_refused(self):
        pool = _make_pool(size=4)
        with self.assertRaises(ValueError):
            pool.free_slot(None)

    def test_normal_return_still_works(self):
        pool = _make_pool(size=4)
        req = _make_req()
        pool.alloc([req])
        row = req.req_pool_idx
        pool.free(req)
        self.assertIsNone(req.req_pool_idx)
        self.assertEqual(pool.free_slots.count(row), 1)
        self.assertEqual(len(pool.free_slots), pool.size)


class AllocRollbackTest(unittest.TestCase):
    """#616, second producer: a deferred batch must hand back only its own rows.

    ``HybridReqToTokenPool.alloc`` defers the batch when the mamba pool is
    exhausted and nothing is evictable (#581), rolling back what it took. The
    rollback used the return value of ``super().alloc``, which names a row for
    EVERY request in the batch -- including a chunked continuation that arrived
    already owning one. Those live rows went back on the free list.
    """

    def test_base_alloc_returns_rows_for_reusing_requests_too(self):
        """Pins the semantics that made the rollback wrong."""
        pool = _make_pool(size=4)
        first = _make_req()
        pool.alloc([first])
        held = first.req_pool_idx

        # A chunked continuation arrives already owning `held`.
        first.inflight_middle_chunks = 1
        fresh = _make_req()
        returned = pool.alloc([first, fresh])

        self.assertIn(
            held,
            returned,
            "alloc returns a row for the reusing request as well, so its "
            "return value is not the set of rows the call took",
        )
        self.assertEqual(first.req_pool_idx, held, "the reusing req keeps it")

    def test_rollback_keeps_a_chunked_continuation_row(self):
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

        pool = object.__new__(HybridReqToTokenPool)
        pool.free_slots = [3, 4]
        pool._alloc_size = 5
        pool.enable_mamba_extra_buffer = False
        pool.mamba_allocator = SimpleNamespace(free=lambda idx: None)

        # req A continues a chunked prefill on row 1 (allocated earlier);
        # req B was given row 2 by the call being rolled back.
        chunked = _make_req(req_pool_idx=1, inflight_middle_chunks=1)
        allocated = _make_req(req_pool_idx=2)

        pool._rollback_alloc([chunked, allocated], [2], [], [])

        self.assertEqual(
            chunked.req_pool_idx,
            1,
            "a chunked continuation must keep the row it already owned",
        )
        self.assertNotIn(
            1,
            pool.free_slots,
            "row 1 is still owned by the chunked request and must not be free",
        )
        self.assertIsNone(allocated.req_pool_idx)
        self.assertIn(2, pool.free_slots, "the row this call took goes back")
        self.assertEqual(
            pool.free_slots[0], 2, "rolled-back rows go to the head for reuse"
        )

    def test_rollback_refuses_to_return_an_already_free_row(self):
        from sglang.srt.mem_cache.memory_pool import HybridReqToTokenPool

        pool = object.__new__(HybridReqToTokenPool)
        pool.free_slots = [2, 3, 4]
        pool._alloc_size = 5
        pool.enable_mamba_extra_buffer = False
        pool.mamba_allocator = SimpleNamespace(free=lambda idx: None)

        stale = _make_req(req_pool_idx=2)
        with self.assertRaises(ValueError):
            pool._rollback_alloc([stale], [2], [], [])


if __name__ == "__main__":
    unittest.main()
