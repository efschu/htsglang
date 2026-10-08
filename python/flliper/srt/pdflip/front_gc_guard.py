# SPDX-License-Identifier: Apache-2.0
"""GC-GUARD (02.10.2026): the front's FULL cyclic-GC pass leaves the flip path.

MEASURED (boot N5q ...10021513_9126170083_1002_151409.front.log, epoch 4, the
D>P Vorlauf 501 ms): ``PDFLIP-FRONT GC-PAUSE generation=2 ms=92 collected=0
frozen=806126`` at 15:17:14.037, between the P>D flip's done (13.854) and the
D>P flip's begin (14.042) -- inside the controller pass that decided the flip.
The same boot printed 62, 69 and 83 ms passes; all of them freed nothing
(collected=0/20/63), i.e. pure walking.

WHO TRIGGERS IT: CPython's own allocation-driven schedule (no ``gc.collect()``
in the front's path). A generation-2 pass walks every tracked object that is
NOT in the permanent generation. The H78 ``gc.freeze()`` runs once, at the end
of the launcher prewarm (15:16:45.3 in N5q); everything the front builds after
that -- the X-EXACT rendering stack (tokenizer, template manager, OpenAI and
Anthropic serving objects, ready 15:16:48.3), the lazy imports of the first
requests and flips -- stays in generation 2 and is walked again by every full
pass: ~0.3 us per object (H78 desk: 731k objects 219-225 ms), so 92 ms is some
300k unfrozen long-lived objects.

THE GUARD, three parts, one switch (``FLLIPER_PDFLIP_FRONT_GC_GUARD``, default on):

1. **No automatic full pass.** ``gc.set_threshold(t0, t1, GEN2_OFF)``: the
   young generations are collected exactly as before (cheap, proportional to
   what was just allocated); CPython never starts a generation-2 pass on its
   own. ``gc.get_count()[2]`` keeps counting the generation-1 passes, so the
   guard knows when CPython WOULD have wanted one (count > the original t2).
2. **The full pass in a quiet moment.** :meth:`GcGuard.step` (the front's
   sampler, every :data:`POLL_S`) runs the due pass only while nothing is
   critical -- no flip open, no queued verdict waiting for one. While
   critical it is deferred; after :data:`MAX_DEFER_S` (env
   ``FLLIPER_PDFLIP_FRONT_GC_MAX_DEFER_S``) it runs at the next moment no FLIP is
   open even with work queued, so the cycles are never left uncollected for
   longer than that bound (memory stays bounded). It never runs inside a flip.
   Marker ``PDFLIP-FRONT GC-GUARD full reason=idle|max-defer ...``.
3. **Refreeze.** The warm-up end (launcher, flip-path imports, sidecar,
   X-EXACT load all done) freezes the heap once more (``PDFLIP-FRONT GC-GUARD
   warm-freeze``), and a full pass that still took >= :data:`REFREEZE_MS`
   freezes its survivors (``PDFLIP-FRONT GC-GUARD refreeze``), at most
   :data:`REFREEZE_MAX` times per process -- a slow pass means long-lived
   objects (late lazy imports) were walked; frozen, the next pass skips them.
   The price, as with every ``gc.freeze()``: a cycle among objects alive at a
   freeze that later becomes garbage is never collected (reference counting
   still frees everything acyclic) -- bounded by the cap.

Python >= 3.14 runs an incremental collector whose thresholds mean something
else: the guard does not arm there and says so (``armed=0 reason=...``).

Switch off: nothing here runs -- the thresholds stay CPython's, as before.
"""

from __future__ import annotations

import gc
import logging
import sys
import time
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger("pdflip.front")

#: Generation-2 threshold while armed: CPython's count never reaches it.
GEN2_OFF = 1 << 30
#: How often the front's sampler asks the guard.
POLL_S = 0.25
#: A due pass waits at most this long for a moment without queued work
#: (default of FLLIPER_PDFLIP_FRONT_GC_MAX_DEFER_S).
MAX_DEFER_S = 30.0
#: Two guard passes are at least this far apart (CPython's own schedule also
#: spaces full passes: the 25 % long-lived ratio rule, unreadable from Python).
MIN_INTERVAL_S = 5.0
#: A full pass at least this slow freezes its survivors (~70k objects walked).
REFREEZE_MS = 20.0
#: Refreezes per process at most (the uncollectable-cycle bound).
REFREEZE_MAX = 8

#: Set while the guard itself collects -- the GC-PAUSE probe names the trigger.
_GUARD_COLLECTING = False


def guard_collecting() -> bool:
    """True while :meth:`GcGuard.step` runs its own full pass."""
    return _GUARD_COLLECTING


def supported() -> Tuple[bool, str]:
    """Whether this interpreter's collector is the generational one the guard drives."""
    if sys.version_info >= (3, 14):
        return False, "incremental-gc-py%d.%d" % sys.version_info[:2]
    t0 = gc.get_threshold()[0]
    if t0 <= 0:
        return False, "threshold0=0 (automatic collection already off)"
    return True, ""


class GcGuard:
    """See the module docstring. One per front process; :meth:`arm` once."""

    def __init__(self, max_defer_s: float = MAX_DEFER_S, min_interval_s: float = MIN_INTERVAL_S,
                 refreeze_ms: float = REFREEZE_MS, refreeze_max: int = REFREEZE_MAX,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.max_defer_s = float(max_defer_s)
        self.min_interval_s = float(min_interval_s)
        self.refreeze_ms = float(refreeze_ms)
        self.refreeze_max = int(refreeze_max)
        self.clock = clock
        self.armed = False
        self.orig: Optional[Tuple[int, int, int]] = None
        self.first_due: Optional[float] = None
        self.last_full: Optional[float] = None
        self.refreezes = 0
        self.counters: Dict[str, int] = {"full_idle": 0, "full_max_defer": 0, "deferred_steps": 0,
                                         "refreeze": 0, "warm_freeze": 0}

    # -- arming --------------------------------------------------------------
    def arm(self) -> bool:
        ok, why = supported()
        if not ok:
            logger.info("PDFLIP-FRONT GC-GUARD armed=0 reason=%s -- CPython's own schedule stays", why)
            return False
        self.orig = tuple(gc.get_threshold())  # type: ignore[assignment]
        gc.set_threshold(self.orig[0], self.orig[1], GEN2_OFF)
        self.armed = True
        logger.info("PDFLIP-FRONT GC-GUARD armed=1 threshold=%s->%s max_defer_s=%.0f min_interval_s=%.0f "
                    "refreeze_ms=%.0f refreeze_max=%d (no automatic generation-2 pass; the due pass "
                    "runs when no flip is open and no verdict waits, FLLIPER_PDFLIP_FRONT_GC_GUARD)",
                    self.orig, gc.get_threshold(), self.max_defer_s, self.min_interval_s,
                    self.refreeze_ms, self.refreeze_max)
        return True

    def disarm(self) -> None:
        if self.armed and self.orig is not None:
            gc.set_threshold(*self.orig)
        self.armed = False

    # -- the schedule ----------------------------------------------------------
    def due(self) -> bool:
        """CPython would have started a generation-2 pass by now."""
        return bool(self.armed and self.orig is not None and gc.get_count()[2] > self.orig[2])

    def step(self, critical: bool, flipping: bool) -> Optional[str]:
        """One sampler beat. ``critical``: a flip is open or a verdict waits for
        one; ``flipping``: a flip is open (never collected inside). Returns the
        reason of the pass it ran, or None."""
        if not self.due():
            self.first_due = None
            return None
        if not gc.isenabled():
            # someone paused the collector on purpose (the H75 launcher import
            # in its worker thread): no pass of ours inside that window either
            return None
        now = self.clock()
        if self.first_due is None:
            self.first_due = now
        if self.last_full is not None and now - self.last_full < self.min_interval_s:
            return None
        if flipping or (critical and now - self.first_due < self.max_defer_s):
            self.counters["deferred_steps"] += 1
            return None
        reason = "max-defer" if critical else "idle"
        deferred_s = now - self.first_due
        ms, collected = self._collect()
        self.last_full = self.clock()
        self.first_due = None
        self.counters["full_max_defer" if critical else "full_idle"] += 1
        frozen = gc.get_freeze_count()
        refroze = False
        if ms >= self.refreeze_ms and self.refreezes < self.refreeze_max:
            gc.freeze()
            self.refreezes += 1
            self.counters["refreeze"] += 1
            refroze = True
        logger.info("PDFLIP-FRONT GC-GUARD full reason=%s ms=%.1f collected=%d deferred_s=%.1f frozen=%d "
                    "refreeze=%d (generation 2, run where no flip waits on the loop)",
                    reason, ms, collected, deferred_s, frozen, int(refroze))
        if refroze:
            logger.info("PDFLIP-FRONT GC-GUARD refreeze n=%d/%d frozen=%d->%d (a %.1f ms pass walked "
                        "long-lived objects; frozen, the next pass skips them)", self.refreezes,
                        self.refreeze_max, frozen, gc.get_freeze_count(), ms)
        return reason

    def warm_freeze(self, what: str) -> int:
        """Freeze what the warm-up built (no pass first: O(1), never on the loop's clock)."""
        before = gc.get_freeze_count()
        gc.freeze()
        self.counters["warm_freeze"] += 1
        after = gc.get_freeze_count()
        logger.info("PDFLIP-FRONT GC-GUARD warm-freeze after=%s frozen=%d->%d (+%d objects built after "
                    "the H78 launcher freeze; no full pass walks them again)", what, before, after,
                    after - before)
        return after - before

    @staticmethod
    def _collect() -> Tuple[float, int]:
        global _GUARD_COLLECTING
        _GUARD_COLLECTING = True
        try:
            t0 = time.perf_counter()
            n = gc.collect(2)
            return (time.perf_counter() - t0) * 1000.0, int(n)
        finally:
            _GUARD_COLLECTING = False
