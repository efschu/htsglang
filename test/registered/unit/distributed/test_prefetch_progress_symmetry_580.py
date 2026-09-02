"""#580: entry into the HiCache prefetch-progress collectives must be
all-ranks-or-none, not decided by this rank's own pool.

THE DEFECT THIS PINS CLOSED
---------------------------
``UnifiedRadixCache.check_prefetch_progress`` carries two collectives on the
attn/TP groups: the MAX reduce inside ``can_terminate_prefetch``
(``unified_radix_cache.py``, ``_all_reduce_attn_groups(states, MAX,
label="can_terminate_prefetch")``) and the MIN reduce over completed tokens
plus sidecar hit pages (``label="check_prefetch_progress"``).

It used to be called from INSIDE the prefill admission loop of
``Scheduler._get_new_batch_prefill_raw``, i.e. only for the prefix of the
waiting queue that the loop actually reached. Every exit that bounds that
prefix reads THIS rank's memory:

  * ``running_batch.batch_is_full`` -- a flag carried across scheduler
    iterations, set from ``get_num_allocatable_reqs`` and from the adder's
    ``AddReqResult.NO_TOKEN`` verdict, which reads
    ``token_to_kv_pool_allocator.available_size()``;
  * the ``res != AddReqResult.CONTINUE`` break, same source;
  * the early returns above the loop, on ``batch_is_full``, on
    ``min_free_slots_delayer.should_delay`` and on
    ``get_num_allocatable_reqs(...) <= 0``.

Under uneven TP/DCP the per-rank pools differ by construction, so near a pool
boundary one rank stops at request N while a peer walks on to N+1 and enters
the collectives alone. The peer then meets the stopping rank's NEXT
collective instead of its own -- two different ops of two different widths on
one gloo group, which is the production abort of 2026-08-05 06:23,
``op.preamble.length <= op.nbytes. 16 vs 4``.

The existing #580 vote (``prefetch_from_storage`` ->
``prefetch_participation_vote``) makes prefetch REGISTRATION rank-uniform. It
says nothing about who later walks into the progress check. That is the gap
these tests close.

THE FIX, AND WHY IT IS NOT A MASK
---------------------------------
``Scheduler._drain_prefetch_progress`` runs the progress check over the WHOLE
waiting queue -- a replicated list -- under the rank-uniform
``enable_hicache_storage`` config flag, before any rank-local admission
predicate. The admission loop then only READS the drained verdict via
``_prefetch_done_for``. The entry decision no longer has a rank-local input
at all; it is not a wider margin around one, which is why the divergent
fixtures below stop diverging instead of merely diverging less often.

Hermetic: no CUDA, no process group, no model, no pools. The scheduler
surface the decision touches is faked and the REAL methods are bound onto it,
so the production source is what runs.
"""

import types
import unittest
from typing import List, Optional
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers import scheduler as scheduler_mod  # noqa: E402
from sglang.srt.managers.schedule_policy import AddReqResult  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeReq:
    """Only the surface the admission loop reads off a queued request."""

    def __init__(self, rid: str):
        self.rid = rid
        self.lora_id = None
        self.mamba_pool_idx = None
        self.mamba_cow_src_index = None
        self.mamba_needs_clear = False
        self.storage_hit_length = 0
        self.session = None

    def init_next_round_input(self, tree_cache, cow_mamba=True):
        pass

    def finished(self):
        return False


class _RecordingTreeCache:
    """Records which requests this rank asked the progress check about.

    That sequence IS the collective sequence: each call with a registered
    prefetch enters ``can_terminate_prefetch``'s MAX reduce and then the
    ``check_prefetch_progress`` MIN reduce. Two ranks whose sequences differ
    are two ranks issuing different collectives on the same group.
    """

    def __init__(self, done_rids=None):
        self.asked: List[str] = []
        self._done_rids = done_rids
        self.req_to_token_pool = types.SimpleNamespace(mamba_allocator=None)

    def check_prefetch_progress(self, rid: str) -> bool:
        self.asked.append(rid)
        if self._done_rids is None:
            return True
        return rid in self._done_rids

    def pop_prefetch_loaded_tokens(self, rid: str) -> int:
        return 0

    def check_hicache_events(self):
        pass


class _FakeBatch:
    def __init__(self, batch_is_full=False):
        self.batch_is_full = batch_is_full
        self.reqs: List[_FakeReq] = []

    def is_empty(self) -> bool:
        return not self.reqs


class _FakeAdder:
    """A ``PrefillAdder`` that admits according to a fixed script.

    ``can_run_list`` deliberately stays empty so the caller returns right
    after the loop; the loop itself -- the break placement under test -- runs
    for real.
    """

    def __init__(self, verdicts: List[AddReqResult]):
        self._verdicts = list(verdicts)
        self.can_run_list: List[_FakeReq] = []
        self.preempt_list: List[_FakeReq] = []
        self.new_chunked_req = None

    def add_one_req(self, req, truncation_align_size=None):
        if not self._verdicts:
            return AddReqResult.CONTINUE
        return self._verdicts.pop(0)

    def preempt_to_schedule(self, req, server_args):
        return False


class _FakeScheduler:
    """Only the surface ``_get_new_batch_prefill_raw`` touches on the way to
    (and through) the admission loop. Built by hand rather than by
    constructing a real Scheduler: the decision under test depends on none of
    the model, device or process-group state the constructor wants."""

    def __init__(
        self,
        waiting_queue: List[_FakeReq],
        adder_verdicts: Optional[List[AddReqResult]] = None,
        batch_is_full: bool = False,
        enable_hicache_storage: bool = True,
        done_rids=None,
        tp_size: int = 3,
    ):
        self.waiting_queue = waiting_queue
        self.tree_cache = _RecordingTreeCache(done_rids=done_rids)
        self.enable_hicache_storage = enable_hicache_storage
        self.enable_hierarchical_cache = True
        self.ps = types.SimpleNamespace(tp_size=tp_size)
        self.running_batch = _FakeBatch(batch_is_full=batch_is_full)

        self.grammar_manager = types.SimpleNamespace(has_waiting_grammars=lambda: False)
        self.server_args = types.SimpleNamespace(
            enable_flexkv=False, prefill_max_requests=None
        )
        self.enable_priority_preemption = False
        self.is_hybrid_swa = False
        self.chunked_req = None
        self.min_free_slots_delayer = None
        self.policy = types.SimpleNamespace(calc_priority=lambda q, b: None)
        self.chunked_prefill_size = 8192
        # Bound off the real class, not stubbed: the sizing decision moved
        # into its own method (#656 first-chunk dynamic chunking), and a
        # fake that reimplemented it could keep passing while the real one
        # changed underneath.
        self.dynamic_chunked_prefill_size = (
            scheduler_mod.Scheduler.dynamic_chunked_prefill_size.__get__(self)
        )
        self.enable_dynamic_chunking = False
        self.kv_session_offload = None
        self.prefill_delayer = None
        self.page_size = 16
        self.token_to_kv_pool_allocator = object()
        self.new_token_ratio_tracker = types.SimpleNamespace(current=1.0)
        self.max_prefill_tokens = 1 << 20
        self.is_mixed_chunk = False
        self.priority_scheduling_preemption_threshold = 0
        self.max_prefill_bs = 0
        self.admission_limiter = types.SimpleNamespace(current=1024)
        self.dllm_config = None
        self.enable_lora = False
        self.lora_drainer = None
        self.req_to_token_pool = types.SimpleNamespace(mamba_allocator=None)
        self.disaggregation_mode = scheduler_mod.DisaggregationMode.NULL
        self.truncation_align_size = None
        self._adder_verdicts = adder_verdicts or []
        # Stubbed after _update_uniform_pool_budget now reads
        # self.server_args.dcp_size (scheduler.py:3503) and calls
        # uniform_budget_deficit() (scheduler.py:4366).
        self.server_args = types.SimpleNamespace(dcp_size=1, prefill_max_requests=None)
        self._uniform_budget_deficit = 0

    def uniform_budget_deficit(self):
        """Mirror the real Scheduler method; always 0 in the stub."""
        return 0

    def get_num_allocatable_reqs(self, running_bs):
        return 1024

    # The production code under test, bound unmodified.
    _drain_prefetch_progress = Scheduler._drain_prefetch_progress
    _prefetch_done_for = Scheduler._prefetch_done_for
    _get_new_batch_prefill_raw = Scheduler._get_new_batch_prefill_raw
    # #1068 weg1 slice 3 (0ad85647cb): `_get_new_batch_prefill_raw` now runs
    # the A12.2 rate-limited-deferral retry on every pass, ABOVE the progress
    # drain (`if self.enable_hicache_storage: self._retry_deferred_prefetches()`).
    # Bound off the real class, not stubbed (#624 stub-drift class): with no
    # `prefetch_deferred` mark on any _FakeReq it returns 0 before touching
    # anything, and the retry is rank-local by ruling (spec A12.6 CORRECTION,
    # 597a7f21eb) -- it must never add an entry into the progress collectives
    # this module pins, and binding the real method is what would show it if
    # it ever did.
    _retry_deferred_prefetches = Scheduler._retry_deferred_prefetches


def _run_rank(sched: _FakeScheduler) -> List[str]:
    """Run one scheduler iteration's prefill admission and return the
    sequence of requests this rank entered the progress check for."""
    with mock.patch.object(
        scheduler_mod,
        "PrefillAdder",
        lambda *a, **kw: _FakeAdder(sched._adder_verdicts),
    ):
        sched._get_new_batch_prefill_raw(
            prefill_delayer_single_pass=None, running_batch=sched.running_batch
        )
    return sched.tree_cache.asked


def _queue(n: int) -> List[_FakeReq]:
    return [_FakeReq(f"r{i}") for i in range(n)]


class PrefetchProgressEntryIsRankUniform(unittest.TestCase):
    """THE FALSIFIERS. Each plants a divergence that the pre-fix code turned
    into two different collective sequences on one group."""

    def test_a_short_rank_and_a_long_rank_enter_the_same_checks(self):
        """The binding rank's pool stops its adder at the first request; the
        peer's pool funds all three.

        Pre-fix the binding rank asked about ['r0'] and the peer about
        ['r0', 'r1', 'r2'] -- one rank posted the r0 progress reduce and then
        left, the peer stood in r1's. Post-fix both walk the same queue.
        """
        binding = _FakeScheduler(_queue(3), adder_verdicts=[AddReqResult.NO_TOKEN])
        peer = _FakeScheduler(_queue(3), adder_verdicts=[AddReqResult.CONTINUE] * 3)

        # The fixture must really be divergent, or the assertion below proves
        # nothing: the adder verdicts do split the two ranks' admission.
        self.assertNotEqual(binding._adder_verdicts, peer._adder_verdicts)

        self.assertEqual(_run_rank(binding), ["r0", "r1", "r2"])
        self.assertEqual(_run_rank(peer), ["r0", "r1", "r2"])

    def test_a_rank_that_returns_early_on_batch_is_full_still_enters(self):
        """``batch_is_full`` is carried across iterations, so a rank that
        filled up last iteration returns from
        ``_get_new_batch_prefill_raw`` before the loop exists. Pre-fix it
        entered ZERO progress checks while its peers entered three -- the
        widest form of the split."""
        full = _FakeScheduler(_queue(3), batch_is_full=True)
        peer = _FakeScheduler(_queue(3), adder_verdicts=[AddReqResult.CONTINUE] * 3)

        asked_full = _run_rank(full)
        asked_peer = _run_rank(peer)
        self.assertEqual(asked_full, ["r0", "r1", "r2"])
        self.assertEqual(asked_full, asked_peer)

    def test_no_allocatable_reqs_early_return_still_enters(self):
        """The ``get_num_allocatable_reqs(...) <= 0`` early return is the
        other pre-loop exit, and it reads ``req_to_token_pool.available_size()``
        through the same rank-local path."""
        starved = _FakeScheduler(_queue(3))
        starved.get_num_allocatable_reqs = lambda running_bs: 0
        peer = _FakeScheduler(_queue(3), adder_verdicts=[AddReqResult.CONTINUE] * 3)

        self.assertEqual(_run_rank(starved), ["r0", "r1", "r2"])
        self.assertEqual(_run_rank(peer), ["r0", "r1", "r2"])

    def test_a_not_done_prefetch_still_leaves_the_sequences_equal(self):
        """The loop ``continue``s past a request whose prefetch has not
        landed. The verdict is read from the drain, so the skip cannot make
        the two ranks' collective sequences differ either."""
        done = {"r0", "r2"}
        binding = _FakeScheduler(
            _queue(3), adder_verdicts=[AddReqResult.NO_TOKEN], done_rids=done
        )
        peer = _FakeScheduler(
            _queue(3), adder_verdicts=[AddReqResult.CONTINUE] * 3, done_rids=done
        )
        self.assertEqual(_run_rank(binding), _run_rank(peer))

    def test_three_ranks_of_the_crashing_boot_agree(self):
        """Uneven TP=3: three different pools, three different admission
        depths, one collective sequence."""
        scripts = [
            [AddReqResult.NO_TOKEN],
            [AddReqResult.CONTINUE, AddReqResult.NO_TOKEN],
            [AddReqResult.CONTINUE] * 4,
        ]
        seen = [_run_rank(_FakeScheduler(_queue(4), adder_verdicts=s)) for s in scripts]
        self.assertEqual(seen, [["r0", "r1", "r2", "r3"]] * 3)
        self.assertEqual(len({tuple(s) for s in seen}), 1)


class TheGateCanStillAnswerNo(unittest.TestCase):
    """A test that only ever asserts "everyone enters" would pass against a
    drain that enters unconditionally for every configuration. These pin the
    NEGATIVE direction: nobody enters."""

    def test_hicache_storage_off_takes_no_check_at_all(self):
        """The default (non-HiCache) path must be byte-identical."""
        for verdicts in ([AddReqResult.NO_TOKEN], [AddReqResult.CONTINUE] * 3):
            sched = _FakeScheduler(
                _queue(3), adder_verdicts=verdicts, enable_hicache_storage=False
            )
            self.assertEqual(_run_rank(sched), [])

    def test_an_empty_queue_takes_no_check_on_any_rank(self):
        sched = _FakeScheduler([])
        self.assertEqual(_run_rank(sched), [])

    def test_the_drain_returns_an_empty_map_when_storage_is_off(self):
        sched = _FakeScheduler(_queue(3), enable_hicache_storage=False)
        self.assertEqual(sched._drain_prefetch_progress(), {})
        self.assertEqual(sched.tree_cache.asked, [])


class TheVerdictIsNeverRecomputedLocally(unittest.TestCase):
    """#606's getattr-default trap: a fallback that reads as present in the
    source and silently reinstates the rank-local call at runtime."""

    def test_a_missing_verdict_is_refused_on_a_group(self):
        sched = _FakeScheduler(_queue(1), tp_size=3)
        with self.assertRaises(RuntimeError) as ctx:
            sched._prefetch_done_for(_FakeReq("not-queued"), {})
        self.assertIn("#580", str(ctx.exception))
        self.assertIn("not-queued", str(ctx.exception))
        # And it did NOT quietly enter the collective to answer.
        self.assertEqual(sched.tree_cache.asked, [])

    def test_a_single_rank_may_still_answer_locally(self):
        """With one rank there is nothing to diverge from, so refusing would
        be a regression rather than a protection."""
        sched = _FakeScheduler(_queue(1), tp_size=1)
        self.assertTrue(sched._prefetch_done_for(_FakeReq("not-queued"), {}))
        self.assertEqual(sched.tree_cache.asked, ["not-queued"])

    def test_a_false_verdict_is_honoured_and_not_treated_as_missing(self):
        """``if not verdict`` instead of ``if verdict is None`` would be the
        same silent-local bug wearing a different hat: a request whose
        prefetch has NOT landed reads False, which is a real answer."""
        sched = _FakeScheduler(_queue(1), tp_size=3)
        self.assertFalse(sched._prefetch_done_for(_FakeReq("r0"), {"r0": False}))
        self.assertEqual(sched.tree_cache.asked, [])


class TheDrainSitsAboveEveryRankLocalPredicate(unittest.TestCase):
    """Placement is the whole fix. Pin it against the real source so a later
    edit that moves the drain below an early return -- or restores the
    in-loop call -- fails here rather than in production."""

    @staticmethod
    def _source() -> List[str]:
        import inspect

        return inspect.getsource(Scheduler._get_new_batch_prefill_raw).splitlines()

    def _index_of(self, needle: str) -> int:
        src = self._source()
        for i, line in enumerate(src):
            if needle in line:
                return i
        self.fail(f"{needle!r} not found in _get_new_batch_prefill_raw")

    def test_the_drain_precedes_every_early_return_and_the_loop(self):
        drain = self._index_of("_drain_prefetch_progress()")
        for later in (
            "running_batch.batch_is_full or len(self.waiting_queue) == 0",
            "self.min_free_slots_delayer is not None",
            "self.get_num_allocatable_reqs(running_bs) <= 0",
            "for req in self.waiting_queue:",
        ):
            self.assertLess(
                drain,
                self._index_of(later),
                msg=f"the #580 drain must run before {later!r}",
            )

    def test_the_admission_loop_does_not_call_the_progress_check_itself(self):
        src = "\n".join(self._source())
        self.assertNotIn("self.tree_cache.check_prefetch_progress(", src)
        # #791b reformatted the call (the local verdict now feeds the group
        # ballot); the pin follows the shape but keeps its meaning: the loop
        # reads the DRAINED verdict, never the live progress check.
        self.assertIn("self._prefetch_done_for(", src)
        self.assertIn("req, prefetch_verdicts", src)
        # ...and the decision the loop consumes is the BALLOT's, resolved
        # through the module (instr22/23: the local verdict split the TP
        # replicas; see prefetch_ballot.py).
        self.assertIn("prefetch_ballot.prefetch_done_under_ballot(", src)

    def test_the_drain_reads_no_pool_state(self):
        import inspect

        src = inspect.getsource(Scheduler._drain_prefetch_progress)
        body = src.split('"""')[-1]
        for rank_local in (
            "available_size",
            "batch_is_full",
            "can_run_list",
            "evictable_size",
            "backuped",
        ):
            self.assertNotIn(
                rank_local,
                body,
                msg=f"{rank_local!r} is rank-local; the drain must not read it",
            )


if __name__ == "__main__":
    unittest.main()
