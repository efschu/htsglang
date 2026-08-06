# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#583: a standing cross-rank census of collective COUNTS, by family.

WHAT THIS ANSWERS
-----------------
Every desync in this family ends the same way: one rank issues a different
NUMBER of group collectives than its peers, the ranks stop pairing up, and
~30 s later a spin kernel takes its abort path. The traceback then says where
each rank was STANDING, which is a downstream fact. On 2026-08-05 three
crashes (9, 10, 11) all read "ranks 0 and 1 in one collective, rank 2 in a
different one" -- and none of them said which collective rank 2 had SKIPPED.

The per-rank prefill line already decomposes wait time by family
(``tp.all_reduce 243.4/129x``), so the counts exist. They were never compared
ACROSS ranks, which is the one comparison that names the defect: 129 vs 128
on ``tp.all_reduce`` turns "rank 2 was somewhere else" into "rank 2 skipped
its Nth tp.all_reduce".

This module makes that comparison standing.

MEASUREMENT, NOT MITIGATION
---------------------------
Nothing here changes what the ranks do. It cannot repair a divergence and
does not try: by the time the counts differ the forward is already wrong.
It reports, so the NEXT crash arrives with its cause attached.

TWO REPORTING PATHS, FOR TWO DIFFERENT FAILURE STATES
-----------------------------------------------------
1. ``compare_across_ranks`` -- the detector. Runs on the gloo CPU group
   (``tp_cpu_group``), NEVER the device/BAR1 path, so a wedged device group
   does not disable the instrument that is supposed to explain it. Rides an
   existing once-per-iteration scheduler point on a replicated cadence, so
   every rank enters it in the same round or none do. It fires while the
   ranks are still healthy -- typically within one tick of the divergence
   and ~30 s before the deadline abort.

2. ``format_local_census`` -- the wedge-proof half. At abort time the peers
   may already be dead (in crashes 9/10/11 the third rank aborted 30 s after
   the first two), so a collective would hang exactly when the evidence is
   needed. This one takes NO collective: each rank dumps its OWN counts into
   the log, and the three dumps are diffed after the fact.

COST
----
One dict increment per collective on the hot path, behind a module-level
bool. At the production rate (~129 collectives per prefill forward) this is
noise. The cross-rank comparison is one ``all_gather_object`` of a small
dict every ``interval`` scheduler iterations.

ARMED BY DEFAULT, like the #605 flight-recorder marks: an instrument that
must be switched on before it can explain a crash explains no crashes. Set
``SGLANG_COLLECTIVE_CENSUS=0`` to disable the counting entirely.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "CollectiveCensus",
    "Divergence",
    "census",
    "census_enabled",
    "census_heartbeat",
    "census_interval",
    "format_local_census",
    "format_local_history",
]

#: Kill switch. Default ON -- see the module docstring.
ENV_ENABLE = "SGLANG_COLLECTIVE_CENSUS"

#: Scheduler iterations between cross-rank comparisons.
ENV_INTERVAL = "SGLANG_COLLECTIVE_CENSUS_INTERVAL"

#: Scheduler iterations between coarse local heartbeats.
ENV_HEARTBEAT = "SGLANG_COLLECTIVE_CENSUS_HEARTBEAT"

#: Default cadence. At the production iteration rate this is a comparison
#: every couple of seconds -- an order of magnitude inside the ~30 s spin
#: deadline, so a divergence is named well before it aborts, while the cost
#: stays one small ``all_gather_object`` per interval.
DEFAULT_INTERVAL = 50

#: Coarse heartbeat cadence. Deliberately far apart: its job is to prove over
#: a long boot that the detector is STILL ticking, not to report anything the
#: comparison would not already have escalated. One modulo on a counter that
#: is incremented anyway, so the hot path is untouched.
DEFAULT_HEARTBEAT = 10000

#: Default ring-buffer capacity for per-collective history. Stores the most
#: recent N (family, nbytes) pairs so the abort-time dump can show the
#: sequence of collectives, not just the cumulative counts.
DEFAULT_HISTORY_LEN = 64


def census_enabled() -> bool:
    return os.environ.get(ENV_ENABLE, "1") not in ("0", "false", "False")


def census_heartbeat() -> int:
    """Local-snapshot heartbeat cadence, in scheduler iterations. Non-positive
    disables the heartbeat only."""
    try:
        return int(os.environ.get(ENV_HEARTBEAT, DEFAULT_HEARTBEAT))
    except ValueError:
        return DEFAULT_HEARTBEAT


def census_interval() -> int:
    """Cadence in scheduler iterations. A non-positive value disables the
    periodic comparison while leaving the counting (and the abort-time dump)
    armed."""
    try:
        return int(os.environ.get(ENV_INTERVAL, DEFAULT_INTERVAL))
    except ValueError:
        return DEFAULT_INTERVAL


class Divergence:
    """One family whose counts are not equal across the group."""

    __slots__ = ("family", "counts", "leader", "behind")

    def __init__(self, family: str, counts: Sequence[int]) -> None:
        self.family = family
        self.counts: Tuple[int, ...] = tuple(counts)
        #: The highest count seen. Ranks below it have SKIPPED collectives;
        #: naming the max (rather than rank 0) makes "behind by" a
        #: non-negative deficit no matter which rank is the odd one out.
        self.leader: int = max(self.counts) if self.counts else 0
        #: (rank, deficit) for every rank that is behind the leader.
        self.behind: List[Tuple[int, int]] = [
            (r, self.leader - c) for r, c in enumerate(self.counts) if c < self.leader
        ]

    def describe(self) -> str:
        behind = ", ".join(f"rank {r} behind by {d}" for r, d in self.behind)
        return f"{self.family}: counts {list(self.counts)} -- {behind}"


class CollectiveCensus:
    """Monotonic per-family collective counters for THIS process."""

    def __init__(self) -> None:
        self._counts: Dict[str, int] = {}
        self._round: int = 0
        #: Families DECLARED from replicated config, as opposed to families
        #: that happen to have fired. See :meth:`declare_families`.
        self._declared: set = set()
        #: Number of comparisons that actually ran, so a test (or a boot log)
        #: can prove the instrument is live rather than dormant.
        self.comparisons = 0
        #: Number of comparisons that found a divergence.
        self.divergences_seen = 0
        #: Arming is announced once, from the first TICK rather than at
        #: import: an import-time line proves only that the module loaded,
        #: while a line from the tick proves the wired production path is
        #: actually reached on this flagset. Silence from an instrument that
        #: is silent when healthy is otherwise unfalsifiable (#380).
        self._armed_announced = False
        #: Skip-warnings are emitted ONCE. A tick that cannot run usually
        #: cannot run EVERY tick (a wrong attribute, a missing group), so
        #: per-tick warning produces thousands of identical lines that bury
        #: the one that mattered -- 2661 of them in a single boot.
        self._skip_warned = False
        #: Ring buffer of the most recent (family, nbytes) entries, bounded
        #: at DEFAULT_HISTORY_LEN. Per-instance, not global shared state.
        self._history: deque = deque(maxlen=DEFAULT_HISTORY_LEN)

    # -- declaration ----------------------------------------------------

    def declare_families(self, families: Sequence[str]) -> None:
        """Seed families to zero from REPLICATED config, before any fire.

        WHY THIS EXISTS (#610 lesson: pack size is decided by replicated
        config, never by rank-local state). ``snapshot`` is what
        ``all_gather_object`` packs, so the KEY SET is the payload width. Left
        to grow on first use, that key set is a function of what a rank has
        happened to execute -- which is precisely the rank-local state this
        instrument exists to catch diverging. A rank that has not yet issued
        its first ``tp.broadcast`` would pack a narrower dict than its peers,
        and the widths would differ for the same reason the counts do.

        Group construction is replicated -- every rank builds the same groups
        under the same names -- so declaring each group's families at
        construction makes the key set identical on every rank from the first
        comparison onward, independent of execution.

        A second effect worth having: a declared family that never fires is
        now VISIBLE as ``0x`` rather than absent, so "this family was never
        counted" and "this family is not instrumented" stop looking alike.
        """
        for family in families:
            self._declared.add(family)
            self._counts.setdefault(family, 0)

    # -- hot path -------------------------------------------------------

    def bump(self, family: str, nbytes: int = 0) -> None:
        """Count one collective. One dict increment; no allocation.

        ``nbytes`` is an optional byte-count for the history ring buffer;
        defaults to 0 for backward compatibility.
        """
        self._counts[family] = self._counts.get(family, 0) + 1
        self._history.append((family, nbytes))

    def snapshot(self) -> Dict[str, int]:
        return dict(self._counts)

    def format_local_history(self, rank: int) -> str:
        """This rank's recent collective sequence, oldest-first.

        Intended for the abort path: each rank dumps its own history into
        the log; the three lines are diffed after the fact to find where
        the sequences diverged.
        """
        entries = list(self._history)
        count = len(entries)
        if not entries:
            return f"collective history (rank {rank}, {count} entries): no entries recorded"
        body = " ".join(f"{f}:{n}" for f, n in entries)
        return (
            f"collective history (rank {rank}, {count} entries): {body}. "
            f"Diff this line against the peers' to find where the sequences "
            f"diverged."
        )

    # -- reporting ------------------------------------------------------

    def next_round(self) -> int:
        """Advance the REPLICATED round counter.

        Must be called exactly once per scheduler iteration by every rank,
        from a point every rank reaches unconditionally -- the cadence gate
        below is this counter, never a local verdict and never the counts
        themselves. Same rule the kv-pressure ladder states for its own
        consensus cadence, and for the same reason: a cadence read off local
        state would itself become a rank-local test before a group
        collective.
        """
        self._round += 1
        return self._round

    def due(self, interval: int) -> bool:
        return interval > 0 and self._round % interval == 0

    def announce_armed_once(self, rank: int, interval: int, heartbeat: int) -> None:
        """One INFO line per scheduler rank, the first time the tick runs."""
        if self._armed_announced:
            return
        self._armed_announced = True
        logger.info(
            "collective census armed (rank %d): interval=%d iterations, "
            "heartbeat=%d, families tracked=%d (%s). Counting is always on; "
            "the cross-rank comparison runs on the gloo tp_cpu_group and "
            "reports only when the ranks disagree.",
            rank,
            interval,
            heartbeat,
            len(self._counts),
            ", ".join(sorted(self._counts)) or "none yet",
        )

    def warn_skipped_once(self, exc: BaseException) -> None:
        """Report an unusable tick ONCE, loudly enough to act on."""
        if self._skip_warned:
            return
        self._skip_warned = True
        logger.error(
            "collective census is NOT RUNNING: the per-iteration tick raised "
            "%s: %s. Counting and the cross-rank comparison are both dead for "
            "this process; the abort-time dump will be empty. This message is "
            "printed once, not per tick.",
            type(exc).__name__,
            exc,
        )

    def heartbeat(self, rank: int) -> None:
        """Coarse proof-of-life carrying this rank's local counts."""
        logger.info("%s", format_local_census(rank))

    def compare_across_ranks(
        self, group, world_size: int, rank: int
    ) -> Optional[List[Divergence]]:
        """Gather every rank's counts over gloo and diff them.

        Returns the list of diverging families (empty when the ranks agree),
        or ``None`` when the comparison could not be made at all.

        WARN-NEVER-RAISE. This is an instrument: if it cannot do its job it
        says so and gets out of the way. It must never be the reason a
        healthy forward fails, and it must never mask the defect it is
        watching for by raising something else first.
        """
        try:
            import torch.distributed as dist

            if group is None or world_size <= 1:
                return None
            local = self.snapshot()
            gathered: List[Optional[Dict[str, int]]] = [None] * world_size
            dist.all_gather_object(gathered, local, group=group)
            self.comparisons += 1
            found = self._diff(gathered)
            if found:
                self.divergences_seen += 1
                self._log(found, rank)
            return found
        except Exception as exc:  # noqa: BLE001 - instrument must not raise
            logger.warning(
                "collective census: cross-rank comparison unavailable (%s: %s); "
                "counting continues locally and the abort-time dump is "
                "unaffected",
                type(exc).__name__,
                exc,
            )
            return None

    @staticmethod
    def _diff(gathered: Sequence[Optional[Dict[str, int]]]) -> List[Divergence]:
        """Diff over the UNION of families.

        A rank that skipped the only collective of some family has no key for
        it at all, and that is exactly the case worth catching -- so a
        missing key reads as zero rather than excluding the family.
        """
        families = set()
        for per_rank in gathered:
            if per_rank:
                families.update(per_rank.keys())
        out: List[Divergence] = []
        for family in sorted(families):
            counts = [int((per_rank or {}).get(family, 0)) for per_rank in gathered]
            if len(set(counts)) > 1:
                out.append(Divergence(family, counts))
        return out

    @staticmethod
    def _log(found: Sequence[Divergence], rank: int) -> None:
        logger.error(
            "COLLECTIVE CENSUS DIVERGENCE (rank %d): the ranks have issued "
            "different numbers of group collectives, so they are no longer "
            "pairing up and a deadline abort is expected within the spin "
            "window. Diverging families: %s. The rank that is BEHIND skipped "
            "the collective; look for a rank-local predicate gating that "
            "family's call site.",
            rank,
            "; ".join(d.describe() for d in found),
        )


#: Process-wide instance. One per scheduler process, which is one per rank.
_CENSUS = CollectiveCensus()


def census() -> CollectiveCensus:
    return _CENSUS


def format_local_census(rank: int) -> str:
    """This rank's counts, for the abort path. Takes NO collective."""
    counts = _CENSUS.snapshot()
    if not counts:
        return f"collective census (rank {rank}): no collectives counted"
    body = ", ".join(f"{k} {v}x" for k, v in sorted(counts.items()))
    return (
        f"collective census (rank {rank}, cumulative since boot): {body}. "
        f"Compare this line against the peers' -- the family whose count "
        f"differs is the one that diverged."
    )


def format_local_history(rank: int) -> str:
    """This rank's recent collective sequence, for the abort path.

    Mirrors the shape of ``format_local_census`` but shows the ring-buffer
    history instead of the cumulative counts, so the reader can diff the
    per-rank sequences to find where they diverged.
    """
    return _CENSUS.format_local_history(rank)
