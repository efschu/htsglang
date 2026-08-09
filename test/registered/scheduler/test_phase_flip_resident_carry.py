# SPDX-License-Identifier: Apache-2.0
"""#631 J.3: the resident decode set must SURVIVE the cutover.

Every pin here is written from a MEASURED mechanism, not an imagined one:

* the drop site is ``Scheduler.init_pp_loop_state``, which rebinds
  ``running_mbs`` -- the resident decode set under ``event_loop_pp`` -- to
  fresh empty batches. The REAL method is driven here, not a re-statement
  of it, because the whole defect was that its rebind is invisible at the
  call site;
* the observable of record is the POOL CENSUS bracket
  (``resident_reqs`` before and after), so the pins compare exactly what
  the metal log reports: membership, by ``(rid, req_pool_idx)``;
* the can-fail arms reproduce the two shapes that actually happened -- a
  request dropped by the swap, and a request reachable only through a
  stale view.

CPU-only and hermetic. The batches are duck-typed: what is under test is
WHICH batch objects are carried WHERE, never ``merge_batch``'s tensor
arithmetic (that is the scheduler's own primitive, used unchanged).
"""

import unittest
from types import SimpleNamespace

from sglang.srt.managers.phase_flip_resident_carry import (
    ResidentCarryError,
    assert_no_orphan_resident_reqs,
    harvest_resident_batches,
    install_resident_set,
    merge_resident_batches,
    promote_slot_zero_to_running_batch,
    resident_req_identity,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _req(rid, idx):
    return SimpleNamespace(rid=rid, req_pool_idx=idx)


def _mreq(rid, idx, mamba_idx):
    """A request that also holds a mamba/GDN slot (the hybrid model case)."""
    return SimpleNamespace(rid=rid, req_pool_idx=idx, mamba_pool_idx=mamba_idx)


class _FakeBatch:
    """Duck-typed ScheduleBatch: reqs, is_empty, merge_batch."""

    def __init__(self, reqs=None):
        self.reqs = list(reqs or [])
        self.merged_from = []

    def is_empty(self):
        return len(self.reqs) == 0

    def merge_batch(self, other):
        self.merged_from.append(other)
        self.reqs.extend(other.reqs)


class _PPStub:
    """The REAL init_pp_loop_state driven over an attribute shell.

    Only the attributes that method reads are provided; anything else it
    touches would be a genuine finding about the seam.
    """

    def __init__(self, pp_size, reqs_per_slot=()):
        self.ps = SimpleNamespace(pp_size=pp_size)
        self.server_args = SimpleNamespace(
            pp_async_batch_depth=0,
            enable_dsa_prefill_context_parallel=False,
        )
        self.running_mbs = [_FakeBatch(r) for r in reqs_per_slot]
        self.last_mbs = [None] * len(self.running_mbs)
        self.running_batch = _FakeBatch()
        self.last_batch = None

    init_pp_loop_state = SchedulerPPMixin.init_pp_loop_state


class TestHarvest(CustomTestCase):
    def test_running_batch_aliasing_a_slot_is_harvested_once(self):
        """The alias is the duplication hazard: ``running_batch`` IS a slot
        object under event_loop_pp, and merging it into itself would
        double every request in it."""
        sched = _PPStub(3, ([_req("a", 1)], [], []))
        sched.running_batch = sched.running_mbs[0]
        got = harvest_resident_batches(sched)
        self.assertEqual(len(got), 1)
        self.assertIs(got[0], sched.running_mbs[0])

    def test_every_resident_slot_is_harvested_not_just_the_current_one(self):
        """J.1's lesson, restated as a carry pin: the resident set spans
        slots, and the hook fires at an arbitrary one."""
        sched = _PPStub(3, ([_req("a", 1)], [], [_req("b", 2)]))
        self.assertEqual(
            resident_req_identity(sched), [("a", 1), ("b", 2)]
        )

    def test_can_fail_one_request_in_two_batches_is_refused(self):
        """Merging those would carry the same Req twice -- duplicate rows
        and a double free. It must be loud, not absorbed."""
        shared = _req("a", 1)
        sched = _PPStub(3, ([shared], [shared], []))
        with self.assertRaisesRegex(ResidentCarryError, "TWO distinct batches"):
            harvest_resident_batches(sched)

    def test_can_fail_a_request_only_in_last_mbs_is_refused(self):
        """At a quiescent boundary this cannot happen, so it means the
        quiescence predicate admitted a boundary that is not one."""
        sched = _PPStub(3, ([_req("a", 1)], [], []))
        sched.last_mbs[1] = _FakeBatch([_req("ghost", 7)])
        with self.assertRaisesRegex(ResidentCarryError, "only.*through last_mbs"):
            assert_no_orphan_resident_reqs(sched)

    def test_orphan_check_passes_when_last_mbs_only_mirrors_running_mbs(self):
        """The steady state: last_mbs holds the batch already merged into
        the slot, i.e. the SAME Req objects. Not an orphan."""
        r = _req("a", 1)
        sched = _PPStub(3, ([r], [], []))
        sched.last_mbs[0] = _FakeBatch([r])
        assert_no_orphan_resident_reqs(sched)


class TestRealInitPreservesResidents(CustomTestCase):
    """THE DEFECT ITSELF, pinned against the real code object."""

    def test_init_pp_loop_state_no_longer_drops_the_resident_set(self):
        sched = _PPStub(3, ([_req("a", 1)], [], [_req("b", 2)]))
        before = resident_req_identity(sched)
        sched.init_pp_loop_state()
        self.assertEqual(resident_req_identity(sched), before)
        self.assertEqual(before, [("a", 1), ("b", 2)])

    def test_the_set_survives_a_topology_change_that_shrinks_the_array(self):
        """PP(3 slots) -> TP(1 slot) is where a per-slot carry would have
        had nowhere to put slots 1 and 2. Merging is not optional."""
        sched = _PPStub(3, ([_req("a", 1)], [_req("b", 2)], [_req("c", 3)]))
        sched.ps.pp_size = 1
        sched.init_pp_loop_state()
        self.assertEqual(len(sched.running_mbs), 1)
        self.assertEqual(
            resident_req_identity(sched), [("a", 1), ("b", 2), ("c", 3)]
        )

    def test_boot_path_is_bit_for_bit_unchanged(self):
        """Nothing resident -> the rule is a no-op, which is what makes it
        safe to state unconditionally."""
        sched = _PPStub(3)
        sched.init_pp_loop_state()
        self.assertEqual(len(sched.running_mbs), 3)
        self.assertTrue(all(mb.is_empty() for mb in sched.running_mbs))
        self.assertEqual(sched.last_mbs, [None, None, None])
        self.assertIsNone(sched.last_batch)

    def test_repeated_init_does_not_duplicate_requests(self):
        """event_loop_pp calls this at its entry, right after the cutover
        already called it. Merging is NOT idempotent, so the carry has to
        be -- a duplicated Req is a double free."""
        sched = _PPStub(3, ([_req("a", 1)], [_req("b", 2)], []))
        sched.init_pp_loop_state()
        sched.init_pp_loop_state()
        sched.init_pp_loop_state()
        self.assertEqual(
            resident_req_identity(sched), [("a", 1), ("b", 2)]
        )
        total = sum(len(mb.reqs) for mb in sched.running_mbs)
        self.assertEqual(total, 2)


class TestPpRingSurvivesTheTopologySwap(CustomTestCase):
    """#631 DEFECT M: the PP chain's ring is not the live ps's.

    The cutover rewrites ps per phase, and the TP phase gets pp_rank=0,
    pp_size=1. Deriving the chain ring from ps therefore made UPSTREAM ==
    SELF on every rank, and the flip-commit hygiene check then compared a
    rank's own dict SEND counter against its own dict CONSUME counter --
    two different wires. Rank 0 sends proxy dicts and consumes none, so
    the imbalance was permanent: measured 8889 withheld rounds and
    "tensor-dict wire has 24 unconsumed message(s) from rank 0" (itself),
    with tp_to_pp abandoning for want of a quorum it could not form.
    """

    def _sched(self, pp_rank, pp_size, ring_rank=None, ring_n=None):
        sched = SimpleNamespace(ps=SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size))
        if ring_rank is not None:
            sched.pp_flip_counters = SimpleNamespace(rank=ring_rank, n_ranks=ring_n)
        sched._pp_flip_ring = SchedulerPPMixin._pp_flip_ring.__get__(sched)
        sched._pp_flip_upstream = SchedulerPPMixin._pp_flip_upstream.__get__(sched)
        sched._pp_flip_downstream = SchedulerPPMixin._pp_flip_downstream.__get__(sched)
        return sched

    def test_ring_holds_after_the_ps_is_rewritten_to_the_tp_topology(self):
        # ps says pp_rank=0/pp_size=1 (the TP phase); the counters carry
        # the PP topology this rank was booted with.
        for rank in (0, 1, 2):
            sched = self._sched(0, 1, ring_rank=rank, ring_n=3)
            self.assertEqual(sched._pp_flip_upstream(), (rank - 1) % 3)
            self.assertEqual(sched._pp_flip_downstream(), (rank + 1) % 3)

    def test_can_fail_the_old_ps_derived_ring_makes_upstream_self(self):
        """The falsifier for the OLD path, so the defect stays dead."""
        sched = self._sched(0, 1)  # no counters -> falls back to ps
        self.assertEqual(sched._pp_flip_upstream(), 0)
        self.assertEqual(sched._pp_flip_downstream(), 0)

    def test_pp_phase_ring_is_unchanged(self):
        """In the PP phase both sources agree, so nothing moves."""
        for rank in (0, 1, 2):
            sched = self._sched(rank, 3, ring_rank=rank, ring_n=3)
            self.assertEqual(sched._pp_flip_upstream(), (rank - 1) % 3)


class TestGdnSlotsFollowTheResidentSet(CustomTestCase):
    """#631 J.1's SECOND occurrence, found by audit and fixed here.

    The GDN leg enumerated ``scheduler.running_batch`` -- one microbatch
    slot -- so a request resident in another slot had its conv/ssm state
    left behind while (since J.1) its KV was carried correctly. Decoding
    on with truncated linear state raises nothing: #212's shape.
    """

    def test_slots_span_every_resident_slot(self):
        from sglang.srt.managers.gdn_flip_mover import resident_mamba_slots

        sched = _PPStub(3, ([_mreq("a", 1, 5)], [], [_mreq("b", 2, 9)]))
        got = resident_mamba_slots(sched).tolist()
        self.assertEqual(got, [5, 9])

    def test_can_fail_a_resident_request_without_a_mamba_slot_is_refused(self):
        from sglang.srt.layers.dcp.reshard_plan import KvReshardError
        from sglang.srt.managers.gdn_flip_mover import resident_mamba_slots

        sched = _PPStub(3, ([_req("a", 1)], [], []))
        with self.assertRaisesRegex(KvReshardError, "no mamba slot"):
            resident_mamba_slots(sched)

    def test_reading_one_slot_would_have_missed_the_other(self):
        """The falsifier for the OLD code path: current-slot-only
        enumeration returns a strict subset, and silently."""
        from sglang.srt.managers.gdn_flip_mover import resident_mamba_slots

        sched = _PPStub(3, ([_mreq("a", 1, 5)], [], [_mreq("b", 2, 9)]))
        sched.running_batch = sched.running_mbs[0]
        old_style = sorted(
            {r.mamba_pool_idx for r in sched.running_batch.reqs}
        )
        self.assertEqual(old_style, [5])
        self.assertEqual(resident_mamba_slots(sched).tolist(), [5, 9])


class TestChunkedPrefillIsResident(CustomTestCase):
    """#631 DEFECT O: the chunked prefill is resident and is in NO batch.

    get_next_batch_to_run deliberately moves it out of the batch ("so that
    we can merge only finished requests to running_batch"), so every
    batch-based enumeration misses it -- while it holds committed KV and a
    mamba slot. Until this was enumerated, quiescence had to refuse on
    chunked_req outright, which meant a flip armed BECAUSE of pending
    prefill could only commit once that prefill had already finished:
    measured 19 s of "NOT QUIESCENT: a chunked prefill is half-written",
    the whole 32768-token prefill running in the TP layout at 1525 tok/s
    against 4553 tok/s in PP, and two cutovers paid for nothing.
    """

    def test_live_reqs_enumerates_the_chunked_request(self):
        from sglang.srt.managers.phase_flip_runtime import _live_reqs

        sched = _PPStub(3, ([_req("a", 1)], [], []))
        sched.chunked_req = _req("chunk", 7)
        got = sorted(r.rid for r in _live_reqs(sched))
        self.assertEqual(got, ["a", "chunk"])

    def test_the_chunked_request_is_not_double_counted(self):
        """If it IS also reachable through a batch, it is still one row."""
        from sglang.srt.managers.phase_flip_runtime import _live_reqs

        shared = _req("chunk", 7)
        sched = _PPStub(3, ([shared], [], []))
        sched.chunked_req = shared
        self.assertEqual([r.rid for r in _live_reqs(sched)], ["chunk"])

    def test_a_parked_chunk_does_not_block_quiescence(self):
        """Between chunks is a SETTLED boundary: the prefix is stashed and
        the extend range is accounted, which is what the carry can move.
        What must be quiet is the FORWARD, not the request's existence."""
        from sglang.srt.managers.phase_flip_runtime import build_flip_quiescence_fn

        sched = _quiescence_stub()
        sched.chunked_req = _req("chunk", 7)
        self.assertTrue(build_flip_quiescence_fn(sched)())

    def test_can_fail_a_chunk_without_a_pool_row_still_blocks(self):
        """Mid-admission, before the allocator has given it a row, there is
        nothing coherent to carry."""
        from sglang.srt.managers.phase_flip_runtime import build_flip_quiescence_fn

        sched = _quiescence_stub()
        sched.chunked_req = SimpleNamespace(rid="chunk", req_pool_idx=None)
        fn = build_flip_quiescence_fn(sched)
        self.assertFalse(fn())
        self.assertIn("no pool row", fn.why_not())


def _quiescence_stub():
    import torch

    return SimpleNamespace(
        chunked_req=None,
        last_batch=None,
        last_mbs=[],
        result_queue=[],
        mbs=[],
        running_mbs=[],
        running_batch=SimpleNamespace(reqs=[]),
        tree_cache=SimpleNamespace(
            all_values_flatten=lambda: torch.tensor([], dtype=torch.int64)
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros((4, 10), dtype=torch.int64)
        ),
        phase_flip_runtime=None,
        flip_spec_algorithm=None,
    )


class TestInstallDestinations(CustomTestCase):
    def test_tp_leg_moves_the_set_into_running_batch_and_empties_the_slots(self):
        """The TP loops read running_batch. A copy left in the slot array
        is a second ageing view that the NEXT flip would resurrect."""
        sched = _PPStub(3, ([_req("a", 1)], [_req("b", 2)], []))
        sched.init_pp_loop_state()
        promote_slot_zero_to_running_batch(sched)
        self.assertEqual(len(sched.running_batch.reqs), 2)
        self.assertTrue(all(mb.is_empty() for mb in sched.running_mbs))
        self.assertIsNone(sched.last_batch)

    def test_pp_leg_puts_the_set_where_event_loop_pp_reads_it(self):
        sched = _PPStub(1, ([_req("a", 1)],))
        sched.ps.pp_size = 3
        sched.init_pp_loop_state()
        self.assertEqual([r.rid for r in sched.running_mbs[0].reqs], ["a"])
        self.assertIsNone(sched.last_batch)

    def test_merge_of_nothing_is_none_not_an_empty_batch(self):
        self.assertIsNone(merge_resident_batches([]))
        self.assertIsNone(merge_resident_batches([_FakeBatch(), _FakeBatch()]))

    def test_install_uses_the_schedulers_own_merge_primitive(self):
        """Not a re-implementation: the accumulator is fed through
        merge_batch, so sampling info, penalizers and tensors are merged
        by the code that owns them."""
        a, b = _FakeBatch([_req("a", 1)]), _FakeBatch([_req("b", 2)])
        sched = _PPStub(1)
        sched.running_mbs = [_FakeBatch()]
        install_resident_set(sched, [a, b], to_tp=True)
        self.assertEqual(a.merged_from, [b])


if __name__ == "__main__":
    unittest.main()
