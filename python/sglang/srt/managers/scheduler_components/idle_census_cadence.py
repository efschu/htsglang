"""#1262: the idle pool census must cost the scheduler loop a BOUNDED share of
its own wall time, whatever the pool size.

THE SPECIMEN THIS EXISTS FOR (boot ``weg2t2a``, 2026-09-08,
``/spinning/gpu-arb/weg2/BOOT_weg2t2a_0908.md``). All SIX scheduler ranks,
``active+gil``, in one identical stack, unchanged across two py-spy dumps 30 s
apart::

    read_free_rows (kv_row_ownership.py:996)
    _check_full_pool (scheduler_components/invariant_checker.py:275)
    _check_all_pools (scheduler_components/invariant_checker.py:763)
    on_idle (scheduler.py:14123)

``WEG2-FLIP begin epoch=0`` was logged at 11:57:07Z and never completed; seven
minutes later the front still read ``state=flipping epoch=0 flips=0``.

WHY IT FIRED ON THAT BOOT AND NOT THE ONE BEFORE, and why the fix is not a
threshold. ``read_free_rows`` materialises two Python ``frozenset``s of one int
per KV row -- per idle pass, per rank, under the GIL. #1259 deferred the
drafter's ``lm_head`` and lifted PP2's pool from 162 435 rows to 410 857, a
factor of **2.53**. The instrument's cost is LINEAR IN THE POOL, so every
capacity win this fork lands makes the idle check more expensive; there is no
row count at which it is "safe", only one at which it has not bitten yet. An
instrument whose price grows with the thing it measures cannot sit on the
loop's per-tick path at all.

THE TWO PROPERTIES THIS CLASS GIVES IT, and neither is a tuning knob:

1. **The census is the DIAGNOSTIC, never the check.** The caller decides
   whether to enumerate from an O(1) comparison of the allocator's own
   counters (``invariant_checker._check_full_pool``); this class is only
   consulted once those counters already disagree.
2. **A DUTY CYCLE, derived from the census's own measured cost.** After an
   enumeration that cost ``t`` ms, the next one is refused until
   ``t * (1/duty - 1)`` ms have elapsed, so the census can never occupy more
   than ``duty`` of wall time -- at 410 857 rows exactly as at 162 435, and at
   whatever the next capacity win makes it. No row count, no seconds, no env
   knob appears anywhere in that rule.

THE EMITTER IS #926's, NOT A SECOND ONE. ``match_refusal_census.emit``
(mem_cache/match_refusal_census.py:439-464) is the fork's existing shape for
"rate-limit the periodic line, but NEVER sample away the line that is the
discriminator". Copied here in behaviour, not in code, because the
discriminator differs: there it is a REFUSED verdict, here it is a pass whose
own wall time exceeded the loop's idle poll interval -- the reading that says
the instrument is competing with the loop's service of control messages, which
is the whole finding of #1262 and must not be sampled away.

DENOMINATOR LAW. Every count this class prints names its population, and the
suppressed-line count rides on every emitted line: a rate-limited emitter whose
suppressed count is invisible reads as a zero (speed-mode block, measured
2026-08-31, four instances in one campaign).
"""

from __future__ import annotations

import time
from typing import Callable, Optional, Tuple

from sglang.srt.managers.scheduler_components.idle_sleeper import IDLE_POLL_CAP_MS

#: The largest share of wall time the ROW-ENUMERATING pool census may occupy.
#:
#: Dimensionless on purpose: it is a statement about the LOOP ("the diagnostic
#: may have a twentieth of the idle loop, the loop keeps the rest"), not about
#: the pool, so it stays true across every pool size and does not have to be
#: revisited when the next capacity fix lands. It is the one design constant
#: here, and it replaces the row-count threshold a naive fix would have needed.
IDLE_CENSUS_MAX_DUTY: float = 0.05

#: #926 emitter shape: after the first enumerating pass (never sampled away --
#: it is the one that establishes the cost), one line per this many enumerating
#: passes, each carrying its own suppressed count. A pass whose wall time
#: exceeded the loop's poll interval is the discriminator and is ALWAYS
#: emitted, at WARNING, whatever this number says.
IDLE_CENSUS_LOG_EVERY: int = 20


class IdleCensusCadence:
    """How often the row-enumerating pool census may actually run, and what it
    cost when it did.

    Rank-local, single-threaded (the scheduler loop owns it), no locking. The
    clock is injectable so the duty rule can be pinned without sleeping.
    """

    __slots__ = (
        "max_duty",
        "log_every",
        "poll_interval_ms",
        "_clock",
        "passes",
        "agreed",
        "disagreed",
        "enumerated",
        "deferred_cadence",
        "deferred_control",
        "last_rows",
        "last_cost_ms",
        "max_cost_ms",
        "total_cost_ms",
        "over_poll_passes",
        "_next_allowed",
        "_suppressed_lines",
    )

    def __init__(
        self,
        max_duty: float = IDLE_CENSUS_MAX_DUTY,
        log_every: int = IDLE_CENSUS_LOG_EVERY,
        poll_interval_ms: float = IDLE_POLL_CAP_MS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 < max_duty <= 1.0:
            raise ValueError(
                f"max_duty must be in (0, 1]; got {max_duty!r}. A duty of 1.0 "
                f"means 'the census may have the whole loop', which is the "
                f"pre-#1262 behaviour, not a disable switch."
            )
        self.max_duty = float(max_duty)
        self.log_every = max(1, int(log_every))
        self.poll_interval_ms = float(poll_interval_ms)
        self._clock = clock
        #: idle passes that reached the pool check at all -- the DENOMINATOR
        #: of every other count here.
        self.passes = 0
        #: passes whose O(1) counters balanced, so nothing was enumerated.
        self.agreed = 0
        #: passes whose O(1) counters did NOT balance, i.e. passes for which
        #: the enumerating census is the diagnostic.
        self.disagreed = 0
        #: of those, the ones that actually enumerated.
        self.enumerated = 0
        #: of those, the ones the duty cycle held back.
        self.deferred_cadence = 0
        #: of those, the ones a queued control message held back (#1262 (2)).
        self.deferred_control = 0
        self.last_rows: Optional[int] = None
        self.last_cost_ms = 0.0
        self.max_cost_ms = 0.0
        self.total_cost_ms = 0.0
        #: enumerating passes that alone cost more than the loop's own idle
        #: poll interval -- the #1262 reading, never rate-limited away.
        self.over_poll_passes = 0
        self._next_allowed: Optional[float] = None
        self._suppressed_lines = 0

    # -- what the caller reports back -------------------------------------

    def note_agreement(self) -> None:
        """The O(1) counters balanced: nothing to enumerate, nothing to pay."""
        self.passes += 1
        self.agreed += 1

    def note_disagreement(self) -> None:
        """The O(1) counters did not balance: the census is now the question."""
        self.passes += 1
        self.disagreed += 1

    def note_deferred(self, *, control: bool) -> None:
        """A disagreeing pass that did NOT enumerate, and which gate held it.

        ``control=True`` is #1262 (2) -- a control message was already queued,
        and serving it outranks any diagnostic. ``control=False`` is the duty
        cycle. Counted apart because they answer different questions: one says
        the loop is busy, the other says the instrument is expensive.
        """
        if control:
            self.deferred_control += 1
        else:
            self.deferred_cadence += 1

    # -- the gate ----------------------------------------------------------

    def may_enumerate(self) -> bool:
        """Whether the duty cycle permits an enumerating census right now.

        The FIRST one is always permitted: with no measured cost there is no
        duty to honour, and refusing it would make the instrument unable to
        establish its own price.
        """
        if self._next_allowed is None:
            return True
        return self._clock() >= self._next_allowed

    def seconds_until_allowed(self) -> float:
        if self._next_allowed is None:
            return 0.0
        return max(0.0, self._next_allowed - self._clock())

    # -- recording ---------------------------------------------------------

    def record(
        self, rows: Optional[int], cost_ms: float
    ) -> Optional[Tuple[bool, str]]:
        """Book one enumerating pass and re-arm the duty cycle.

        Returns ``(is_warning, line)`` when this pass should be logged, or
        ``None`` when the #926 rate limit swallows it. The caller does the
        logging so this class never imports a logger and stays trivially
        testable.
        """
        cost_ms = max(0.0, float(cost_ms))
        self.enumerated += 1
        self.last_rows = rows
        self.last_cost_ms = cost_ms
        self.max_cost_ms = max(self.max_cost_ms, cost_ms)
        self.total_cost_ms += cost_ms
        # THE DUTY RULE. A pass costing t may occupy at most `max_duty` of the
        # wall time it belongs to, so the window it belongs to is t/max_duty
        # and the part of it that is NOT census is t*(1/max_duty - 1). Derived
        # from the census's own measurement -- the more the pool grows, the
        # rarer the census gets, with nothing to re-tune.
        self._next_allowed = self._clock() + (cost_ms / 1000.0) * (
            1.0 / self.max_duty - 1.0
        )
        over_poll = cost_ms > self.poll_interval_ms
        if over_poll:
            self.over_poll_passes += 1
        # #926: the discriminator is never sampled away; the periodic line
        # exists only to give the log a denominator.
        if over_poll or self.enumerated == 1 or self.enumerated % self.log_every == 0:
            line = self.format_line(over_poll)
            self._suppressed_lines = 0
            return over_poll, line
        self._suppressed_lines += 1
        return None

    # -- the line ----------------------------------------------------------

    def format_line(self, over_poll: bool) -> str:
        rows = "UNKNOWN" if self.last_rows is None else str(self.last_rows)
        head = (
            f"IDLE-POOL-CENSUS rows={rows} wall={self.last_cost_ms:.1f} ms "
            f"(max {self.max_cost_ms:.1f} ms, total {self.total_cost_ms:.0f} ms) "
            f"enumerated={self.enumerated} of {self.disagreed} disagreeing "
            f"pass(es) in {self.passes} idle pass(es) "
            f"[agreed={self.agreed} deferred_cadence={self.deferred_cadence} "
            f"deferred_control={self.deferred_control} "
            f"lines_suppressed={self._suppressed_lines}] "
            f"next enumeration not before +{self.seconds_until_allowed():.2f} s "
            f"(duty <= {self.max_duty * 100:.1f}% of wall)"
        )
        if not over_poll:
            return head
        return (
            head
            + f" -- WARNING: this ONE pass cost {self.last_cost_ms:.1f} ms, more "
            f"than the scheduler's own idle poll interval "
            f"({self.poll_interval_ms:.0f} ms, idle_sleeper.IDLE_POLL_CAP_MS). "
            f"The instrument is competing with the loop's service of control "
            f"messages -- this is the #1262 reading, and it is never rate-limited "
            f"away"
        )
