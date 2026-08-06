"""Hermetic (CPU-only) tests for the three #610 siblings of the
"a rank-local condition fences entry into a group collective" family.

The family already produced #580 (prefetch-progress vote), #581 (writing_check
MIN-freeze) and #607-E (decode-side backuped gate). These three were verified
open against the current tree:

S1 (wall-clock abort gates)
    ``Scheduler._abort_on_waiting_timeout`` (scheduler.py) tests
    ``0 < req.time_stats.wait_queue_entry_time < time.perf_counter() - timeout``
    -- two rank-local quantities -- and the abort body calls
    ``tree_cache.release_aborted_request``, which enters a TP barrier
    (``_barrier_attn_groups`` at unified_radix_cache.py:2480 and
    hiradix_cache.py:1921). A rank whose clock has crossed the deadline enters
    that barrier alone. ``_abort_on_running_timeout`` has the same shape and
    splits batch composition instead.

S2 (HiRadixCache prefetch participation)
    ``HiRadixCache.query_storage_hit_length`` runs a MIN ``all_reduce`` and
    used to sit behind ``prefetch_rate_limited()`` and the caller's
    ``last_host_node.backuped``; ``prefetch_from_storage`` registered
    ``ongoing_prefetch`` under rank-local host-alloc success, which makes
    ``check_prefetch_progress``'s ``req_id not in ongoing_prefetch`` early
    return rank-local too. f081654e8d fixed Scheduler._prefetch_kvcache and
    af5e0c947e the decode side; the HiRadixCache class itself never got the
    symmetrization.

S3 (PrefillAdder NO_TOKEN)
    ``PrefillAdder.budget_state`` decides NO_TOKEN from
    ``available_size() + evictable_size()``, rank-local under uneven DCP. The
    pin existed only when the kv-session-offload manager was constructed.

The collective tests run two threads as two TP ranks against a shared barrier,
so a rank that skips a collective its peer enters shows up as a broken barrier
rather than as a silent pass.
"""

import inspect
import threading
import time
import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.scheduler import Scheduler
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=15)

_BARRIER_TIMEOUT_S = 3.0
_NRANKS = 2


def accepts(fn, kwarg: str) -> bool:
    """Whether ``fn`` takes ``kwarg``.

    The can-fail check reverts ONLY the source, and the pre-fix functions have
    neither the ballot helper nor the ``locally_eligible`` parameters. Probing
    for them lets the test fall back to the pre-fix CALL PATTERN -- the caller
    applying its own rank-local gate -- so a reverted tree fails on the
    divergence symptom rather than on an AttributeError/TypeError at import.
    """
    try:
        return kwarg in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


class BrokenCollective(AssertionError):
    """Raised when a rank reaches a collective its peer never entered."""


class ThreadCollective:
    """A two-thread stand-in for a gloo process group.

    ``all_reduce`` rendezvouses both ranks and computes the real elementwise
    reduction. A rank that never calls it leaves the peer to time out on the
    barrier -- which is precisely the production symptom of this defect family,
    so the test observes the hang as a failure instead of missing it.
    """

    def __init__(self, nranks: int = _NRANKS):
        self.nranks = nranks
        self._barrier = threading.Barrier(nranks)
        self._lock = threading.Lock()
        self._slots = {}
        self._labels = []

    def all_reduce(self, tensor, op=None, group=None):
        rank = threading.current_thread().rank
        with self._lock:
            self._slots[rank] = tensor.clone()
        try:
            self._barrier.wait(timeout=_BARRIER_TIMEOUT_S)
        except threading.BrokenBarrierError as exc:
            raise BrokenCollective("peer rank never entered this collective") from exc
        with self._lock:
            vals = list(self._slots.values())
        acc = vals[0].clone()
        for other in vals[1:]:
            if op is torch.distributed.ReduceOp.MAX:
                acc = torch.maximum(acc, other)
            else:
                acc = torch.minimum(acc, other)
        tensor.copy_(acc)
        try:
            self._barrier.wait(timeout=_BARRIER_TIMEOUT_S)
        except threading.BrokenBarrierError as exc:
            raise BrokenCollective("peer rank left the collective early") from exc
        return None

    def get_world_size(self, group=None):
        return self.nranks

    def abort(self):
        self._barrier.abort()


def run_ranks(fn, nranks: int = _NRANKS):
    """Run ``fn(rank)`` on ``nranks`` threads; return (results, errors)."""
    results = [None] * nranks
    errors = []

    def target(rank):
        threading.current_thread().rank = rank
        try:
            results[rank] = fn(rank)
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append((rank, exc))

    threads = []
    for rank in range(nranks):
        t = threading.Thread(target=target, args=(rank,))
        t.rank = rank
        threads.append(t)
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10.0)
    return results, errors


# ---------------------------------------------------------------------------
# S1 -- wall-clock abort gates
# ---------------------------------------------------------------------------


class BarrierTreeCache:
    """``release_aborted_request`` enters a barrier, as every hierarchical
    cache class really does (hiradix_cache.py:1921, unified:2480,
    hi_mamba:2296)."""

    def __init__(self, collective: ThreadCollective):
        self._collective = collective
        self.released = []

    def release_aborted_request(self, rid):
        self.released.append(rid)
        # The real body reaches `_barrier_attn_groups`; a barrier is an
        # all_reduce with no payload for our purposes.
        self._collective.all_reduce(torch.zeros(1, dtype=torch.int64))


class FakeReq:
    """Identity-hashed, because the abort body puts requests in a set."""

    def __init__(self, rid, entry_time, forward_time=0.0):
        self.rid = rid
        self.to_finish = None
        self.time_stats = SimpleNamespace(
            wait_queue_entry_time=entry_time,
            forward_entry_time=forward_time,
            trace_ctx=SimpleNamespace(abort=lambda **kw: None),
        )

    def finished(self):
        return False


def make_req(rid, entry_time, forward_time=0.0):
    return FakeReq(rid, entry_time, forward_time)


class TimeoutHarness:
    """Binds the REAL Scheduler methods onto a minimal attribute surface.

    Testing the real functions (not a transcription of them) is what makes the
    revert-only-source check meaningful.
    """

    # Pre-existing names, bound unconditionally: these are the functions under
    # test. `_uniform_timeout_ballot` is the new helper and is attached only
    # when it exists, so a reverted source still imports and still runs its own
    # (rank-local) logic.
    _abort_on_waiting_timeout = Scheduler._abort_on_waiting_timeout
    _abort_on_running_timeout = Scheduler._abort_on_running_timeout
    if hasattr(Scheduler, "_uniform_timeout_ballot"):
        _uniform_timeout_ballot = Scheduler._uniform_timeout_ballot

    def __init__(self, collective, waiting_queue, enable_hicache_storage=True):
        self.tp_cpu_group = object()
        self.waiting_queue = waiting_queue
        self.enable_hicache_storage = enable_hicache_storage
        self.tree_cache = BarrierTreeCache(collective)
        self.ipc_channels = SimpleNamespace(
            send_to_tokenizer=SimpleNamespace(send_output=lambda *a, **k: None)
        )


class WallClockAbortGateTest(unittest.TestCase):
    """S1. The two ranks disagree on the wall clock by construction: rank 0's
    request is one jitter-width past the deadline, rank 1's is one short of it.
    That is the whole defect -- perf_counter is read at each rank's own point
    in its own scheduler iteration."""

    def _run(self, timeout_s=1.0):
        collective = ThreadCollective()

        def body(rank):
            now = time.perf_counter()
            # rank 0: expired. rank 1: not yet. Same request, same queue
            # position, different local clock reading.
            entry = now - timeout_s - 0.5 if rank == 0 else now - timeout_s + 0.5
            queue = [make_req("rid-0", entry)]
            harness = TimeoutHarness(collective, queue)
            with (
                unittest.mock.patch.object(
                    torch.distributed, "all_reduce", collective.all_reduce
                ),
                unittest.mock.patch.object(
                    torch.distributed, "get_world_size", collective.get_world_size
                ),
                unittest.mock.patch(
                    "sglang.srt.managers.scheduler.envs.SGLANG_REQ_WAITING_TIMEOUT.get",
                    lambda: timeout_s,
                ),
            ):
                harness._abort_on_waiting_timeout()
            return harness

        results, errors = run_ranks(body)
        collective.abort()
        return results, errors

    def test_ranks_agree_on_the_abort_set(self):
        results, errors = self._run()
        self.assertEqual(
            errors,
            [],
            f"a rank broke a collective -- desync: {errors}",
        )
        released = [tuple(h.tree_cache.released) for h in results]
        self.assertEqual(
            released[0],
            released[1],
            "the ranks released different requests: the abort verdict is not "
            f"rank-uniform ({released})",
        )
        queues = [len(h.waiting_queue) for h in results]
        self.assertEqual(
            queues[0],
            queues[1],
            f"waiting-queue composition diverged across ranks: {queues}",
        )

    def test_running_timeout_verdict_is_rank_uniform(self):
        timeout_s = 1.0
        collective = ThreadCollective()

        def body(rank):
            now = time.perf_counter()
            fwd = now - timeout_s - 0.5 if rank == 0 else now - timeout_s + 0.5
            req = make_req("rid-0", 0.0, forward_time=fwd)
            batch = SimpleNamespace(reqs=[req], is_empty=lambda: False)
            harness = TimeoutHarness(collective, [])
            with (
                unittest.mock.patch.object(
                    torch.distributed, "all_reduce", collective.all_reduce
                ),
                unittest.mock.patch.object(
                    torch.distributed, "get_world_size", collective.get_world_size
                ),
                unittest.mock.patch(
                    "sglang.srt.managers.scheduler.envs.SGLANG_REQ_RUNNING_TIMEOUT.get",
                    lambda: timeout_s,
                ),
            ):
                harness._abort_on_running_timeout(batch)
            return req.to_finish is not None

        results, errors = run_ranks(body)
        collective.abort()
        self.assertEqual(errors, [], f"a rank broke a collective: {errors}")
        self.assertEqual(
            results[0],
            results[1],
            "the ranks disagreed on whether the running request timed out; "
            "the next decode batch would differ in composition "
            f"({results})",
        )


# ---------------------------------------------------------------------------
# S2 -- HiRadixCache prefetch participation
# ---------------------------------------------------------------------------


class FakeHostPool:
    def __init__(self, size, alloc_ok=True):
        self.size = size
        self._alloc_ok = alloc_ok
        self.released = []

    def alloc(self, n):
        return torch.arange(n, dtype=torch.int64) if self._alloc_ok else None

    def available_size(self):
        return self.size if self._alloc_ok else 0

    def get_dummy_flat_data_page(self):
        return torch.zeros(1, dtype=torch.int64)


class FakeController:
    def __init__(self, host_pool, rate_limited=False, hit_count=64):
        self.mem_pool_host = host_pool
        self._rate_limited = rate_limited
        self._hit_count = hit_count
        self.prefetch_tokens_occupied = 0
        self.prefetch_capacity_limit = int(0.5 * host_pool.size)
        self.released = []

    def prefetch_rate_limited(self):
        return self._rate_limited

    def _storage_hit_query(self, operation):
        return ([], self._hit_count)

    def prefetch(self, *args, **kwargs):
        return SimpleNamespace(host_indices=torch.zeros(1), hash_value=[])

    def append_host_mem_release(self, host_indices, *args, **kwargs):
        self.released.append(host_indices)


class FakeHostNode:
    def __init__(self):
        self.key = SimpleNamespace(extra_key=None)
        self.protected = 0

    def protect_host(self):
        self.protected += 1

    def release_host(self):
        self.protected -= 1


def make_hiradix(collective, controller, symmetric=True):
    """A REAL HiRadixCache object with only the attributes these two methods
    touch populated. Pinning against the real class is deliberate: a
    transcribed copy would not notice a change to the class under test."""
    from sglang.srt.mem_cache.hiradix_cache import HiRadixCache

    cache = HiRadixCache.__new__(HiRadixCache)
    cache.enable_storage = True
    cache.cache_controller = controller
    cache.tp_world_size = _NRANKS
    cache.page_size = 32
    cache.prefetch_threshold = 32
    cache.is_eagle = False
    cache.ongoing_prefetch = {}
    cache._get_extra_pools = lambda: {}
    cache.evict_host = lambda n: None
    cache._hicache_prefetch_symmetric = lambda: symmetric
    cache._all_reduce_attn_groups = lambda t, op: collective.all_reduce(t, op)
    return cache


class HiRadixPrefetchParticipationTest(unittest.TestCase):
    """S2. Rank 1 is locally ineligible (rate-limited / no host memory), rank 0
    is not -- the asymmetry weighted DCP produces by construction, since the
    host pool is sized from the per-rank device pool."""

    def test_storage_hit_query_is_entered_by_every_rank(self):
        collective = ThreadCollective()

        def body(rank):
            pool = FakeHostPool(4096)
            controller = FakeController(pool, rate_limited=(rank == 1))
            cache = make_hiradix(collective, controller)
            locally_eligible = rank == 0
            if accepts(cache.query_storage_hit_length, "locally_eligible"):
                return cache.query_storage_hit_length(
                    FakeHostNode(),
                    list(range(256)),
                    locally_eligible=locally_eligible,
                )
            # Pre-fix call pattern: `_build_decode_prefix_match` applied the
            # `last_host_node.backuped` gate itself and simply did not call.
            if not locally_eligible:
                return 0
            return cache.query_storage_hit_length(FakeHostNode(), list(range(256)))

        results, errors = run_ranks(body)
        collective.abort()
        self.assertEqual(
            errors,
            [],
            "a rank skipped the storage-hit MIN all_reduce its peer entered "
            f"-- the #580 shape: {errors}",
        )
        self.assertEqual(
            results[0],
            results[1],
            f"the ranks derived different L3 hit lengths: {results}",
        )
        self.assertEqual(
            results[0], 0, "one ineligible rank must pull the MIN ballot to 0"
        )

    def test_prefetch_registration_is_all_or_none(self):
        collective = ThreadCollective()

        def body(rank):
            # Rank 1's host pool cannot serve the allocation -- the weighted-DCP
            # asymmetry. Under the vote neither rank may register.
            pool = FakeHostPool(4096, alloc_ok=(rank == 0))
            controller = FakeController(pool)
            cache = make_hiradix(collective, controller)
            if accepts(cache.prefetch_from_storage, "locally_eligible"):
                cache.prefetch_from_storage(
                    "rid-0",
                    FakeHostNode(),
                    list(range(256)),
                    locally_eligible=True,
                )
            else:
                cache.prefetch_from_storage("rid-0", FakeHostNode(), list(range(256)))
            return set(cache.ongoing_prefetch)

        results, errors = run_ranks(body)
        collective.abort()
        self.assertEqual(errors, [], f"a rank skipped the participation vote: {errors}")
        self.assertEqual(
            results[0],
            results[1],
            "ongoing_prefetch registration is not all-or-none; "
            "check_prefetch_progress's `req_id not in ongoing_prefetch` early "
            "return is therefore rank-local and its two reduces will be "
            f"entered by a subset of ranks: {results}",
        )
        self.assertEqual(
            results[0], set(), "a rank that could not allocate must veto the vote"
        )


# ---------------------------------------------------------------------------
# S3 -- PrefillAdder NO_TOKEN
# ---------------------------------------------------------------------------


class BudgetHarness:
    _update_uniform_pool_budget = Scheduler._update_uniform_pool_budget
    # Absent before the fix; the caller then reads a 0 deficit, which is
    # exactly the pre-fix admission behaviour.
    if hasattr(Scheduler, "uniform_budget_deficit"):
        uniform_budget_deficit = Scheduler.uniform_budget_deficit
    else:

        def uniform_budget_deficit(self):
            return 0

    def __init__(self, collective, avail, evictable, dcp_size=_NRANKS):
        self.tp_cpu_group = object()
        self.kv_session_offload = None
        self.token_to_kv_pool_allocator = SimpleNamespace(available_size=lambda: avail)
        self.tree_cache = SimpleNamespace(
            evictable_size=lambda: evictable,
            full_evictable_size=None,
        )
        self.server_args = SimpleNamespace(dcp_size=dcp_size)


class PrefillAdmissionBudgetTest(unittest.TestCase):
    """S3. Two ranks with different pool ownership (the weighted-DCP split) must
    admit against the same budget, or `budget_state` returns NO_TOKEN on one and
    CONTINUE on the other and the prefill batch is composed differently."""

    # Rank 0 is the slack rank, rank 1 the binding one.
    AVAIL = {0: 4000, 1: 100}
    EVICT = {0: 500, 1: 50}

    def _budgets(self):
        collective = ThreadCollective()

        def body(rank):
            harness = BudgetHarness(collective, self.AVAIL[rank], self.EVICT[rank])
            with (
                unittest.mock.patch.object(
                    torch.distributed, "all_reduce", collective.all_reduce
                ),
                unittest.mock.patch.object(
                    torch.distributed, "get_world_size", collective.get_world_size
                ),
                unittest.mock.patch(
                    "sglang.srt.distributed.utils.uneven_dcp_active", lambda *a: True
                ),
            ):
                harness._update_uniform_pool_budget()
            local = self.AVAIL[rank] + self.EVICT[rank]
            return local - harness.uniform_budget_deficit()

        results, errors = run_ranks(body)
        collective.abort()
        return results, errors

    def test_admission_budget_is_rank_uniform(self):
        results, errors = self._budgets()
        self.assertEqual(errors, [], f"a rank broke the budget reduce: {errors}")
        self.assertEqual(
            results[0],
            results[1],
            "the ranks admit against different budgets; budget_state would "
            "return NO_TOKEN on one and CONTINUE on the other "
            f"({results})",
        )
        self.assertEqual(
            results[0],
            min(self.AVAIL[r] + self.EVICT[r] for r in (0, 1)),
            "the shared budget must be the BINDING rank's, so no rank is "
            "asked to hold a request it cannot",
        )

    def test_budget_state_agrees_across_ranks(self):
        """The symptom itself, through the real PrefillAdder predicate: with a
        per-request demand between the two ranks' local budgets, the unpinned
        code makes them disagree on NO_TOKEN."""
        from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder

        budgets, errors = self._budgets()
        self.assertEqual(errors, [], f"a rank broke the budget reduce: {errors}")

        verdicts = []
        for rank in (0, 1):
            adder = PrefillAdder.__new__(PrefillAdder)
            adder.prefill_spill_deep_taken = False
            adder.is_hybrid_swa = False
            adder.is_all_swa = False
            adder.is_hybrid_ssm_cache = False
            adder.rem_mamba_slots = None
            adder.rem_input_tokens = 1 << 20
            adder.rem_chunk_tokens = None
            adder.dllm_config = None
            adder.rem_total_token_offset = 1000
            adder.cur_rem_token_offset = 1000
            adder.token_to_kv_pool_allocator = SimpleNamespace(
                available_size=lambda r=rank: self.AVAIL[r]
            )
            adder.tree_cache = SimpleNamespace(
                evictable_size=lambda r=rank: self.EVICT[r]
            )
            adder.dcp_avail_deficit = (
                self.AVAIL[rank] + self.EVICT[rank] - budgets[rank]
            )
            verdicts.append(adder.budget_state())

        self.assertEqual(
            verdicts[0],
            verdicts[1],
            "the ranks returned different AddReqResult verdicts -- divergent "
            f"batch composition: {verdicts}",
        )
        self.assertEqual(verdicts[0], AddReqResult.NO_TOKEN)


if __name__ == "__main__":
    unittest.main()
