"""fnFL2x105: the HiCache drain before a sleep is a GROUP loop, not a rank loop.

THE HANG (boot fnFL2x104, 2026-09-23 23:19:13 -> 23:21:13). Group D (TP=3,
PP=1) was told to sleep with a 97,841 + 3,891-token context in its pools. The
release leg drained HiCache before the idle assert -- but only on the ranks that
were locally not idle (``if not is_fully_idle(): drain``), and the drain loop
ran on each rank's OWN blocker list. TP0 had no blocker, skipped the loop, slept
and waited in the group fence. TP1/TP2 carried ``hicache_backup(1)`` and polled
``check_hicache_events``, whose ``drain_storage_control_queues`` is an
all_reduce over the attention group -- the group TP0 had just left. Both sides
waited on each other in different collectives until TP0's monitored_barrier
expired after 120 s ("Ranks 1, 2 failed to pass monitoredBarrier"); the
py-spy stacks of 23:21:11 show exactly that pair. Boot fnFL2x105 (40ac644fe0,
23:39:28 -> 23:41:28) repeated it with a 4,521-token context: the size is not
the condition. The condition is a store-loaded prefix that a later D request
SPLITS (see ``UnifiedRadixCache._split_node``: the split parent lost
``l3_present``, so the sleep sweep wrote it again on the workers only, and the
MIN-reduced ack drain left them one backup short of idle for ever).

THE LAW. Every collective ``check_hicache_events`` posts is posted by every
rank of its group, so the number of polls must be a GROUP number. Each pass of
the loop therefore reduces (MAX) this rank's terms over the SAME group
``check_hicache_events`` reduces over, and every rank reads the same verdict:
drained, a non-HiCache blocker, or the bound expired. The bound is part of the
reduced vector, so a clock skew between ranks cannot split the decision either.
A group that is still blocked when the verdict says stop is refused BY NAME on
every rank at once (W120), which the leg wrapper turns into one W29 group stop
-- seconds after the sleep began, not 120 s of silence.
"""

from __future__ import annotations

import time
from typing import Callable, List, Sequence, Tuple

import msgspec

# The bound on the whole drain. It was 30 s (weg2zr1); measured, it never
# decided anything: every drain on record either finished in the first polls
# or never (weg2sb1: 30 s / 2964 polls on an orphaned prefetch). The front's
# sleep-kv stall bound is 11.6 s (fnFL2x104), so a drain longer than this is
# already a stalled flip -- refuse by name before the front has to say so.
WEG2_SLEEP_DRAIN_BOUND_S = 10.0
WEG2_SLEEP_DRAIN_POLL_S = 0.01


class Weg2SleepDrainRefused(RuntimeError):
    """W120: the group's HiCache terms did not drain before the sleep.

    Raised on EVERY rank of the group in the same pass (the verdict it reads
    is a group reduction), so the leg wrapper's fence sees a unanimous False
    and stops the group with W29 instead of one rank sleeping alone.
    """


class SleepDrainVerdict(msgspec.Struct, frozen=True, kw_only=True):
    """The group's reading of one drain pass (MAX over the attention group)."""

    hicache_terms: int
    other_terms: int
    expired: bool

    @property
    def idle(self) -> bool:
        return self.hicache_terms == 0 and self.other_terms == 0

    @property
    def done(self) -> bool:
        # A non-HiCache blocker does not drain by polling HiCache, so the
        # loop stops on it at once, on every rank.
        return self.idle or self.other_terms > 0 or self.expired


def local_drain_terms(blockers: Sequence[str], *, expired: bool) -> List[int]:
    """This rank's contribution: [hicache terms, other terms, expired]."""
    hicache = sum(1 for b in blockers if b.startswith("hicache"))
    return [hicache, len(blockers) - hicache, int(expired)]


def drain_until_group_verdict(
    *,
    idle_blockers: Callable[[], Sequence[str]],
    check_hicache_events: Callable[[], object],
    group_max: Callable[[List[int]], List[int]],
    bound_s: float,
    clock: Callable[[], float] = time.monotonic,
    pause: Callable[[float], object] = time.sleep,
    poll_s: float = WEG2_SLEEP_DRAIN_POLL_S,
) -> Tuple[SleepDrainVerdict, int]:
    """Poll ``check_hicache_events`` until the GROUP verdict says stop.

    ``group_max`` must reduce over the group ``check_hicache_events`` posts
    its collectives on; every rank then leaves the loop after the same number
    of polls. Returns ``(verdict, polls)``.
    """
    t0 = clock()
    polls = 0
    while True:
        local = local_drain_terms(
            list(idle_blockers()), expired=clock() - t0 >= bound_s
        )
        hicache, other, expired = group_max(local)
        verdict = SleepDrainVerdict(
            hicache_terms=int(hicache), other_terms=int(other), expired=bool(expired)
        )
        if verdict.done:
            return verdict, polls
        check_hicache_events()
        polls += 1
        pause(poll_s)


def refusal_message(
    *,
    verdict: SleepDrainVerdict,
    own_blockers: Sequence[str],
    waited_s: float,
    polls: int,
    bound_s: float,
    rank_desc: str,
) -> str:
    return (
        f"W120 Weg2SleepDrainRefused: the group is not idle after the sleep "
        f"drain (group max: hicache_terms={verdict.hicache_terms} "
        f"other_terms={verdict.other_terms} expired={verdict.expired}; "
        f"waited {waited_s:.2f} s, {polls} group polls, bound "
        f"{bound_s:g} s). This rank [{rank_desc}] blocks on "
        f"[{', '.join(own_blockers) or 'none'}]. Every rank of the group raises "
        f"this in the same pass -- nothing was paused."
    )
