"""#701: the chunked-prefill SELF-deadlock, as arithmetic.

A chunked request's own committed prefix can fill the pool that its own next
chunk must allocate from, and it cannot evict itself to make room. This module
models that arithmetic so the reachability condition can be stated exactly and
the candidate fixes priced without a GPU.

**The anatomy, at file:line in the deploy tree.**

1. Each committed chunk LOCKS its prefix.
   ``mem_cache/radix_cache.py:546`` — ``cache_unfinished_req`` ends with
   ``self.inc_lock_ref(new_last_node)``. The prefix moves from *evictable* to
   *protected*.
2. The admission budget EXCLUDES protected space.
   ``managers/schedule_policy.py:809-825`` — ``rem_total_tokens`` is
   ``available_size() + <tree>_evictable_size()``. Nothing protected counts.
3. So every chunk the request commits SHRINKS the budget its next chunk is
   checked against. The request eats its own runway.
4. Eviction cannot recover it. ``radix_cache.py:569-575`` — ``evict`` walks
   ``self.evictable_leaves`` only, and the request's own prefix is locked, so
   it is not a candidate. **The request cannot evict itself to fund itself.**

**The tree already knows.** ``schedule_policy.py:993-995`` documents the
current behaviour by name: *"the prefill input must transiently fit the device.
If not, this is the DEEP case (PS2) -> reject, today's wedge/wait behaviour"*,
and the born-spill admission logs that it is admitting a request whose *"full
lifetime would wedge"*. The defect is acknowledged; what is missing is the
arithmetic that says exactly when.

**The reachability condition.** With a budget ``A`` (available + evictable) at
admission and chunk size ``C``, a request of length ``L`` commits
``floor(L/C)`` chunks, each removing ``C`` from ``A``. Its next chunk fails once
``A - k*C < C``. So a single request self-deadlocks **iff its total length
exceeds the budget it was admitted against** — because admission checks the
CHUNK, never the TOTAL. Concurrently, the condition is on the sum: several
requests each individually admissible can deadlock collectively once their
committed prefixes plus one more chunk exceed the pool.

That is the sharp edge: **admission is per-chunk, but the failure is per-total.**
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence


class ChunkedDeadlockError(ValueError):
    """A modelling question that cannot be answered as posed."""


PROGRESSED = "progressed"
COMPLETED = "completed"
DEADLOCKED = "deadlocked"

#: Candidate fixes, named so they can be compared rather than assumed.
FIX_NONE = "none"
#: Never admit a chunked request whose FULL length cannot fit the budget.
FIX_ADMISSION_GATE = "admission_gate"
#: Let a request's own committed prefix be evicted to host for its tail chunks.
FIX_SELF_EVICTABLE = "self_evictable"


@dataclasses.dataclass(frozen=True)
class Request:
    rid: str
    length: int


@dataclasses.dataclass
class _State:
    #: Tokens of this request already prefilled. Monotone -- progress.
    progress: int = 0
    #: Tokens currently PROTECTED on device. Spilling reduces this without
    #: losing progress, which is the entire point of the host-tier option.
    device_held: int = 0
    admitted: bool = False
    done: bool = False


@dataclasses.dataclass(frozen=True)
class SimResult:
    outcome: str
    committed: dict[str, int]
    admitted: tuple[str, ...]
    refused: tuple[str, ...]
    spilled_tokens: int
    detail: str


def simulate(
    pool_tokens: int,
    chunk_tokens: int,
    requests: Sequence[Request],
    fix: str = FIX_NONE,
    host_spill_budget: int = 0,
) -> SimResult:
    """Run the chunk loop and report whether it wedges.

    Deliberately models only the three quantities that matter: the pool, the
    protected (committed) prefixes, and the chunk. Everything else in the real
    allocator is irrelevant to this failure.
    """
    if pool_tokens <= 0 or chunk_tokens <= 0:
        raise ChunkedDeadlockError("pool and chunk sizes must be positive.")
    if fix not in (FIX_NONE, FIX_ADMISSION_GATE, FIX_SELF_EVICTABLE):
        raise ChunkedDeadlockError(f"unknown fix {fix!r}.")

    state = {r.rid: _State() for r in requests}
    refused: list[str] = []
    spilled = 0

    def protected() -> int:
        return sum(s.device_held for s in state.values() if not s.done)

    # Admission. The shipped path checks only that the NEXT CHUNK fits. The
    # gate option RESERVES the full length, which is what makes it a gate --
    # checking the length without reserving it lets two requests both pass and
    # then collide, which is the concurrent shape this is meant to stop.
    reserved = 0
    for r in requests:
        if fix == FIX_ADMISSION_GATE:
            if r.length <= pool_tokens - reserved:
                state[r.rid].admitted = True
                reserved += r.length
            else:
                refused.append(r.rid)
        else:
            if min(chunk_tokens, r.length) <= pool_tokens - protected():
                state[r.rid].admitted = True
            else:
                refused.append(r.rid)

    live = [r for r in requests if state[r.rid].admitted]
    guard = 0
    max_steps = sum(r.length // chunk_tokens + 2 for r in live) * 4 + 32

    while any(not state[r.rid].done for r in live):
        guard += 1
        if guard > max_steps:
            raise ChunkedDeadlockError("simulation failed to terminate.")
        moved = False
        for r in live:
            st = state[r.rid]
            if st.done:
                continue
            want = min(chunk_tokens, r.length - st.progress)
            if want <= 0:
                st.done = True
                moved = True
                continue
            free = pool_tokens - protected()
            if want > free and fix == FIX_SELF_EVICTABLE:
                # Push this request's OWN protected prefix to host so it can
                # fund its own tail. Progress is preserved; only the device
                # residency is given up.
                need = want - free
                room = min(st.device_held, max(0, host_spill_budget - spilled))
                if room > 0:
                    move = min(st.device_held, max(need, 0))
                    move = min(move, room)
                    st.device_held -= move
                    spilled += move
                    free = pool_tokens - protected()
            if want <= free:
                st.progress += want
                st.device_held += want
                moved = True
                if st.progress >= r.length:
                    st.done = True
        if not moved:
            stuck = {
                r.rid: state[r.rid].progress for r in live if not state[r.rid].done
            }
            return SimResult(
                DEADLOCKED,
                {r.rid: state[r.rid].progress for r in requests},
                tuple(r.rid for r in live),
                tuple(refused),
                spilled,
                "SELF-DEADLOCK: no request can allocate its next chunk, and "
                f"every held prefix is protected so none can be evicted. "
                f"protected={protected()} of pool={pool_tokens}, "
                f"chunk={chunk_tokens}, stuck at {stuck}.",
            )

    return SimResult(
        COMPLETED,
        {r.rid: state[r.rid].progress for r in requests},
        tuple(r.rid for r in live),
        tuple(refused),
        spilled,
        "all admitted requests completed.",
    )


def deadlock_length_threshold(pool_tokens: int, chunk_tokens: int) -> int:
    """Smallest single-request length that self-deadlocks on an empty pool.

    A request commits chunks until its protected prefix leaves less than one
    chunk free, so anything longer than the pool wedges. Stated as a function
    so the caller need not rediscover the off-by-one.
    """
    if pool_tokens <= 0 or chunk_tokens <= 0:
        raise ChunkedDeadlockError("pool and chunk sizes must be positive.")
    return pool_tokens + 1
