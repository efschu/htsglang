"""#701: hermetic falsifier for the chunked-prefill self-deadlock.

Anatomy in the deploy tree:
  radix_cache.py:546        cache_unfinished_req -> inc_lock_ref: the committed
                            prefix becomes PROTECTED
  schedule_policy.py:809-825 rem_total_tokens = available + evictable, so
                            protected space is excluded from the budget
  radix_cache.py:569-575    evict walks evictable_leaves only -- a request
                            cannot evict its own locked prefix
  schedule_policy.py:993-995 the tree names the behaviour: "today's wedge/wait
                            behaviour"

Hermetic: pure arithmetic, no CUDA, no allocator.
"""

import pytest
from sglang.srt.planner.chunked_deadlock import (
    COMPLETED,
    DEADLOCKED,
    FIX_ADMISSION_GATE,
    FIX_NONE,
    FIX_SELF_EVICTABLE,
    ChunkedDeadlockError,
    Request,
    deadlock_length_threshold,
    simulate,
)

POOL = 10_000
CHUNK = 512


def test_a_request_that_fits_completes():
    r = simulate(POOL, CHUNK, [Request("a", 5_000)])
    assert r.outcome == COMPLETED
    assert r.committed["a"] == 5_000


def test_a_request_longer_than_the_pool_SELF_deadlocks():
    """THE falsifier. Admission checks the CHUNK; the failure is on the TOTAL.

    The request is admitted because its first 512-token chunk fits, then eats
    its own runway one locked chunk at a time until nothing is left for the
    next one -- and it cannot evict itself, because its own prefix is what is
    holding the pool.
    """
    r = simulate(POOL, CHUNK, [Request("a", POOL + 4_000)])
    assert r.outcome == DEADLOCKED
    assert "SELF-DEADLOCK" in r.detail
    assert r.admitted == ("a",), "it WAS admitted -- that is the defect"
    # It got most of the way, which is why this looks like a hang not a reject.
    assert r.committed["a"] >= POOL - CHUNK


def test_the_threshold_is_the_pool_not_the_chunk():
    """One token past the pool is enough; chunk size does not rescue it."""
    assert deadlock_length_threshold(POOL, CHUNK) == POOL + 1
    assert simulate(POOL, CHUNK, [Request("a", POOL)]).outcome == COMPLETED
    assert simulate(POOL, CHUNK, [Request("a", POOL + 1)]).outcome == DEADLOCKED
    # A larger chunk changes the granularity, not the outcome.
    assert simulate(POOL, 4096, [Request("a", POOL + 1)]).outcome == DEADLOCKED


def test_requests_that_each_fit_can_deadlock_COLLECTIVELY():
    """The concurrent shape, which no per-request check would catch.

    Each request is individually admissible against the budget it saw. Their
    committed prefixes then sum past the pool, and every one of them is locked.
    """
    each = 6_000  # < POOL individually
    r = simulate(POOL, CHUNK, [Request("a", each), Request("b", each)])
    assert r.outcome == DEADLOCKED
    assert set(r.admitted) == {"a", "b"}
    assert sum(r.committed.values()) >= POOL - CHUNK


def test_the_admission_gate_fix_prevents_it_by_refusing_early():
    """Option 1: never admit a request whose FULL length cannot fit."""
    r = simulate(POOL, CHUNK, [Request("a", POOL + 4_000)], fix=FIX_ADMISSION_GATE)
    assert r.outcome == COMPLETED
    assert r.refused == ("a",), "refused at admission, not wedged mid-flight"
    assert r.committed["a"] == 0


def test_the_admission_gate_also_serialises_the_concurrent_case():
    r = simulate(
        POOL,
        CHUNK,
        [Request("a", 6_000), Request("b", 6_000)],
        fix=FIX_ADMISSION_GATE,
    )
    assert r.outcome == COMPLETED
    assert r.admitted == ("a",) and r.refused == ("b",)


def test_the_self_evictable_fix_completes_a_request_larger_than_the_pool():
    """Option 2: let the request's own prefix spill to host for its tail.

    This is the only option that can serve a prompt LONGER than the device
    pool, which the admission gate can only refuse. The price is host traffic.
    """
    r = simulate(
        POOL,
        CHUNK,
        [Request("a", POOL + 4_000)],
        fix=FIX_SELF_EVICTABLE,
        host_spill_budget=50_000,
    )
    assert r.outcome == COMPLETED
    assert r.committed["a"] > 0
    assert r.spilled_tokens > 0, "it paid for it in host traffic"


def test_the_self_evictable_fix_still_deadlocks_without_host_budget():
    """The fix is the host tier, not the flag: with no room to spill, nothing
    changes. #703 is a genuine dependency, not a nicety."""
    r = simulate(
        POOL,
        CHUNK,
        [Request("a", POOL + 4_000)],
        fix=FIX_SELF_EVICTABLE,
        host_spill_budget=0,
    )
    assert r.outcome == DEADLOCKED


def test_the_two_fixes_differ_in_WHAT_THEY_CAN_SERVE():
    """The comparison that decides the recommendation.

    Both stop the wedge. Only one keeps serving the request.
    """
    long_req = [Request("a", POOL + 4_000)]
    gate = simulate(POOL, CHUNK, long_req, fix=FIX_ADMISSION_GATE)
    spill = simulate(
        POOL, CHUNK, long_req, fix=FIX_SELF_EVICTABLE, host_spill_budget=50_000
    )
    assert gate.outcome == spill.outcome == COMPLETED
    assert gate.committed["a"] == 0, "the gate refuses the request outright"
    assert spill.committed["a"] > 0, "the spill actually serves it"


def test_no_fix_is_the_shipped_behaviour():
    assert simulate(POOL, CHUNK, [Request("a", POOL + 1)], fix=FIX_NONE).outcome == (
        DEADLOCKED
    )


def test_foreign_profile():
    """Different pool, chunk and mix entirely."""
    r = simulate(1_000, 64, [Request("x", 900)])
    assert r.outcome == COMPLETED
    assert simulate(1_000, 64, [Request("x", 1_100)]).outcome == DEADLOCKED


def test_malformed_inputs_are_refused():
    with pytest.raises(ChunkedDeadlockError, match="positive"):
        simulate(0, CHUNK, [Request("a", 1)])
    with pytest.raises(ChunkedDeadlockError, match="unknown fix"):
        simulate(POOL, CHUNK, [Request("a", 1)], fix="wishful")
