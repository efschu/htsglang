"""fnFL2 H23: DECODE-ROUND-COST -- the first decode rounds after a Weg-2 wake.

WHY A SEPARATE LINE. ``Decode rank batch`` prints every round, so the first
rounds after a wake are in the log -- but finding them means joining the
round numbers against the wake by hand, and the question every flip boot asks
("is round 1 after the wake stationary, and if not, which term is cold?") was
answered by hand from x132 (24.09.): round 1 = 66.5 gpu-ms on TP0 (compute
10.0, pool.fetch 39.4, all_reduce 16.3) against a median of 34.0 from round 9
on -- and a no-flip extend of the same size (64 new tokens on 3840 cached)
opened its decode with 76.9 gpu-ms (pool.fetch 48.9). The excess is the pool's
compulsory expert misses after ANY extend, not a wake effect. This line makes
the next boot state that itself, five lines per wake per rank.

WHAT IS PRICED. Rounds ``n = 1..ROUNDS`` opened after ``arm`` (the scheduler
arms on the first pass after the wake, i.e. after the extend was launched):

* ``gpu_ms`` / ``compute_ms`` / ``fetch_ms`` / ``allreduce_ms`` -- the same
  device clock ``Decode rank batch`` reads; phase prefixes (``spec_verify:``)
  are folded, so ``fetch_ms`` is the pool's host->device expert traffic;
* ``wall_ms`` -- host wall from this round's open to the next round's open
  (or to the drain that emitted it, when no later round opened yet);
* ``since_wake_ms`` -- this round's open minus the wake (DORMANT cleared).
  For n=1 that is the post-wake host path up to the first decode launch --
  the part of the flip time that follows the weight legs;
* ``graph`` -- every forward of the round replayed a captured graph;
* ``cold`` -- the largest single term of the round (``compute`` when the
  round is compute-bound, i.e. stationary on this form; ``pool.fetch`` when
  the expert pool missed; ``eager`` when a forward did not replay a graph;
  ``unsplit`` when the clock could not split the round).

COST. Unarmed: one attribute test per round open and per emitted round.
Armed: one ``perf_counter`` per round open and one log line per priced round,
for five rounds. No device work, no allocation on the device, no sync.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import msgspec

logger = logging.getLogger(__name__)

__all__ = ["RoundCost", "WakeRoundCensus", "cold_term"]

#: The families the line names on its own, after the phase prefix is folded.
FETCH_FAMILY = "pool.fetch"
ALL_REDUCE_FAMILY = "tp.all_reduce"


class RoundCost(msgspec.Struct, frozen=True, kw_only=True):
    """One priced round after the wake."""

    n: int
    round_id: int
    rank: int
    gpu_ms: float
    #: None when the clock could not split the round.
    compute_ms: Optional[float]
    fetch_ms: float
    allreduce_ms: float
    wall_ms: float
    #: None when the wake time was unknown at the arm.
    since_wake_ms: Optional[float]
    graphed: bool
    cold: str

    def line(self) -> str:
        return (
            "DECODE-ROUND-COST n=%d round=%d rank=%d gpu_ms=%.1f compute_ms=%s "
            "fetch_ms=%.1f allreduce_ms=%.1f wall_ms=%.1f since_wake_ms=%s "
            "graph=%s cold=%s"
            % (
                self.n,
                self.round_id,
                self.rank,
                self.gpu_ms,
                "-" if self.compute_ms is None else "%.1f" % self.compute_ms,
                self.fetch_ms,
                self.allreduce_ms,
                self.wall_ms,
                "-" if self.since_wake_ms is None else "%.0f" % self.since_wake_ms,
                "yes" if self.graphed else "no",
                self.cold,
            )
        )


def fold_families(families: Dict[str, float]) -> Dict[str, float]:
    """``spec_verify:pool.fetch`` and ``pool.fetch`` are one term."""
    folded: Dict[str, float] = {}
    for name, ms in families.items():
        key = name.rsplit(":", 1)[-1]
        folded[key] = folded.get(key, 0.0) + float(ms)
    return folded


def cold_term(
    *, graphed: bool, compute_ms: Optional[float], folded: Dict[str, float]
) -> str:
    """The largest single term of a round; see the module docstring."""
    if not graphed:
        return "eager"
    if compute_ms is None:
        return "unsplit"
    best_name, best_ms = "compute", float(compute_ms)
    for name in sorted(folded):
        if folded[name] > best_ms:
            best_name, best_ms = name, folded[name]
    return best_name


class WakeRoundCensus:
    """Prices the first ``ROUNDS`` decode rounds opened after ``arm``."""

    ROUNDS: int = 5

    def __init__(self, *, rank: int) -> None:
        self._rank = int(rank)
        self._armed = False
        self._wake_mono: Optional[float] = None
        #: round_id -> open time, for the rounds opened while armed; at most
        #: ROUNDS + 1 entries (the +1 closes the last priced round's wall).
        self._opens: Dict[int, float] = {}

    @property
    def armed(self) -> bool:
        return self._armed

    def arm(self, *, wake_mono: Optional[float]) -> None:
        """A new wake: forget the previous census, price the next rounds."""
        self._armed = True
        self._wake_mono = wake_mono
        self._opens = {}

    def note_open(self, *, round_id: int, mono: float) -> None:
        """Called at every round open, BEFORE the drain that may price the
        previous round -- so that round's wall can end at this open."""
        if not self._armed or len(self._opens) > self.ROUNDS:
            return
        self._opens[int(round_id)] = float(mono)

    def on_round(
        self,
        *,
        round_id: int,
        gpu_ms: float,
        compute_ms: Optional[float],
        families: Dict[str, float],
        graphed: bool,
        now_mono: float,
    ) -> Optional[RoundCost]:
        """Price one emitted round if it is among the first ROUNDS opened
        after the arm; log and return it, else None."""
        if not self._armed:
            return None
        opened = self._opens.get(int(round_id))
        if opened is None:
            return None
        n = self._position(round_id)
        if n > self.ROUNDS:
            return None
        cost = self._price(
            round_id=int(round_id),
            opened=opened,
            gpu_ms=gpu_ms,
            compute_ms=compute_ms,
            families=families,
            graphed=graphed,
            now_mono=now_mono,
        )
        logger.info(cost.line())
        # Rounds are emitted in order; once position ROUNDS is priced the
        # census is done (a round that issued no timed forward is never
        # emitted and must not keep the census armed forever).
        if n >= self.ROUNDS:
            self._armed = False
        return cost

    def _position(self, round_id: int) -> int:
        """1-based position of ``round_id`` among the rounds opened after arm."""
        return 1 + sum(1 for r in self._opens if r < int(round_id))

    def _next_open(self, round_id: int) -> Optional[float]:
        later = [t for r, t in self._opens.items() if r > round_id]
        return min(later) if later else None

    def _price(
        self,
        *,
        round_id: int,
        opened: float,
        gpu_ms: float,
        compute_ms: Optional[float],
        families: Dict[str, float],
        graphed: bool,
        now_mono: float,
    ) -> RoundCost:
        folded = fold_families(families)
        end = self._next_open(round_id)
        return RoundCost(
            n=self._position(round_id),
            round_id=round_id,
            rank=self._rank,
            gpu_ms=float(gpu_ms),
            compute_ms=None if compute_ms is None else float(compute_ms),
            fetch_ms=folded.get(FETCH_FAMILY, 0.0),
            allreduce_ms=folded.get(ALL_REDUCE_FAMILY, 0.0),
            wall_ms=1000.0 * ((now_mono if end is None else end) - opened),
            since_wake_ms=(
                None
                if self._wake_mono is None
                else 1000.0 * (opened - self._wake_mono)
            ),
            graphed=bool(graphed),
            cold=cold_term(graphed=graphed, compute_ms=compute_ms, folded=folded),
        )
