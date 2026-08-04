# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#546 -- the translator gives its VRAM back while nobody is talking.

The tenant held ~5.9 GiB on the shared 5090 around the clock for a workload
that is bursty by nature: a conversation happens, then hours of nothing. The
user's order was blunt and correct -- if it is to be kept warm, it shall
release the VRAM and spill into system RAM.

Nothing here moves a byte. The movement is the #286 ``audio_modules`` asset
class and its ledger (:mod:`sglang.srt.translator.ledger`), which already
parks the TTS modules to host RAM and restores them; #546 added the recognizer
to the same class through a park route, page-locked the host copies, and made
the pages actually go back to the driver. What this module owns is the one
question that machinery does not answer: WHEN.

WHY NOT A TIMER
===============

A bare "idle for N seconds" park is wrong for this tenant, and wrong in the
expensive direction. Inside a live conversation the gap between requests is
seconds -- a turn, a pause, the other person answering. Any fixed N small
enough to reclaim memory promptly after a conversation is also small enough
to fire inside one, and each false park costs a full park+restore on the next
utterance, in front of a person who is mid-sentence. A timer cannot tell the
two apart because it looks at the wrong thing: it measures the CURRENT gap
against a constant, when the question is whether the current gap is unusual
FOR THIS CONVERSATION.

So the threshold is derived from the traffic, in the shape the #536 spill-tick
controller (``kv_session_offload.SpillTickController``) established for a
measured control signal: a trailing window, a margin that acts as the
deadzone, and a dwell that rate-limits transitions independently of what the
signal says. Three terms, each answering a different failure:

1. **Inter-arrival term** -- ``p95(recent gaps) x margin``. Answers "is this
   silence unusual for this conversation?". During a conversation the gaps are
   seconds, so the term is small and the floor binds; the moment the
   conversation ends, the idle span walks past it. Gaps that span a park are
   NOT recorded: the gap that ends a parked period is by definition not a
   conversational gap, and folding it in would poison the percentile upward
   and stop the tenant ever parking again.
2. **Break-even term** -- ``break_even x measured_restore_s``. Answers "is
   parking worth it at all?". Parking to reclaim memory for a minute is a bad
   trade if getting back costs seconds. The restore latency is measured on
   every real wake (the ledger records it per asset), so this term
   self-calibrates: a slow restore raises the bar for parking, automatically.
3. **Floor** -- the operator's absolute minimum silence. Answers "how eager
   may this get, at most?", and is what binds on a fresh process with no
   measurements at all.

and a **ceiling**, so a single long mid-conversation pause cannot push the
adaptive term out to hours.

    threshold = clamp(
        max(floor, p95(gaps) x margin, break_even x restore_s),
        floor, ceiling,
    )

ANTI-FLAP
=========

The dwell is separate from the threshold and deliberately redundant with it:
after a wake completes, no park may start for ``dwell_s``, whatever the
threshold computes. It is the guard that still holds when the threshold logic
is wrong -- for instance if some path touches the GPU without announcing
itself, so the idle clock looks longer than the truth. A wrong threshold then
costs one park per dwell instead of a park/restore loop.

WORKED EXAMPLES
===============

*Live conversation.* Gaps 2-9 s, p95 ~8 s, margin 4 -> 32 s. Measured restore
0.9 s, break-even 20 -> 18 s. Floor 120 s binds. The tenant does not park
during the conversation, and would not even if the floor were 40 s.

*Overnight.* The conversation ends at 23:10. The gap ring still holds the
conversation's gaps, so the threshold stays at the 120 s floor. At 23:12 the
sweeper sees 120 s of silence, no session busy, dwell expired -> park, ~5.9
GiB back to the driver, one ``park_complete`` residency event. At 08:30 the
first request arrives: ``wake_start`` fires before the restore, the recognizer
comes back first and the turn starts while the talker is still landing. One
park and one wake for nine hours of idle.

THE STATE MACHINE
=================

``RESIDENT -> PARKING -> PARKED -> RESTORING -> RESIDENT``. Both transitional
states are published while the movement runs rather than held under a lock,
so a request that arrives mid-park QUEUES on the condition variable, waits for
the park to finish and then restores -- instead of racing it, and instead of
being refused. That is why the mover is called outside the lock and the state
is set inside it.

RESTORING is not a single step. The wake is STAGED in the pipeline's need
order (``ledger.DEFAULT_WAKE_RANKS``): the recognizer first, then the talker,
then the codec, with each waiter released at the rank it asked for. A turn is
therefore served as soon as it can begin, not when the last module lands --
which matters because a wake needs SPACE, and under #553 that space may first
have to be made by the serving engine spilling something of its own. See
:meth:`IdleParkController.ensure_awake` for what that leaves to #553.
"""

from __future__ import annotations

import dataclasses
import enum
import logging
import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

from sglang.srt.translator import residency
from sglang.srt.translator.ledger import AudioAssetLedger

logger = logging.getLogger(__name__)

__all__ = [
    "IdleParkConfig",
    "IdleParkController",
    "ParkDecision",
    "ParkState",
    "WakeTimeout",
]


class ParkState(enum.Enum):
    RESIDENT = "resident"
    PARKING = "parking"
    PARKED = "parked"
    RESTORING = "restoring"


class WakeTimeout(RuntimeError):
    """A wake that waited longer than its budget for another transition."""


@dataclasses.dataclass(frozen=True)
class IdleParkConfig:
    """The operator surface. The measured terms self-calibrate; these do not.

    ``never_park`` is the hard override and beats ``enabled``: it exists so an
    operator debugging a latency complaint can pin the tenant resident without
    having to reason about which of several knobs wins.
    """

    enabled: bool = True
    #: Hard override. Set, nothing ever parks, whatever ``enabled`` says.
    never_park: bool = False
    #: Absolute minimum silence before a park may be considered. The term that
    #: binds on a fresh process, before any gap or restore has been measured.
    floor_s: float = 120.0
    #: Upper bound on the whole threshold, so one long mid-conversation pause
    #: cannot push the adaptive term out to hours.
    ceiling_s: float = 900.0
    #: Multiple of the recent-gap p95. Doubles as the controller's deadzone:
    #: the decision is one-sided (only the park is controller-initiated; the
    #: wake is driven by a request), so a one-sided margin is the whole of it.
    gap_margin: float = 4.0
    #: Multiple of the MEASURED restore latency the idle span must also clear.
    break_even: float = 20.0
    #: No park may start within this long after a wake completes, whatever the
    #: threshold says. The guard that survives a wrong threshold.
    dwell_s: float = 180.0
    #: Trailing window of inter-arrival gaps.
    gap_window: int = 32
    #: Below this many recorded gaps the inter-arrival term is undefined and
    #: is not used -- the floor governs. Guessing a percentile from two
    #: samples is how a controller learns the wrong thing confidently.
    min_gap_samples: int = 4
    #: Longest a wake may wait for another transition to clear before it gives
    #: up. Generous: it is a deadlock detector, not a latency budget.
    wake_timeout_s: float = 120.0

    def active(self) -> bool:
        return bool(self.enabled) and not bool(self.never_park)

    def validate(self) -> None:
        if self.floor_s <= 0:
            raise ValueError("idle-park floor must be positive")
        if self.ceiling_s < self.floor_s:
            raise ValueError(
                f"idle-park ceiling ({self.ceiling_s}s) is below the floor "
                f"({self.floor_s}s); the clamp would invert"
            )
        if self.gap_margin <= 0:
            raise ValueError("idle-park gap margin must be positive")
        if self.break_even < 0:
            raise ValueError("idle-park break-even multiple must be >= 0")
        if self.dwell_s < 0:
            raise ValueError("idle-park dwell must be >= 0")
        if self.gap_window < 1:
            raise ValueError("idle-park gap window must be at least 1")


@dataclasses.dataclass(frozen=True)
class ParkDecision:
    """Why the controller did or did not park on this tick. Always logged."""

    parked: bool
    reason: str
    idle_s: float
    threshold_s: float
    freed_bytes: int = 0
    #: The three terms, so a threshold can be explained without a debugger.
    terms: Dict[str, float] = dataclasses.field(default_factory=dict)

    def to_json(self) -> Dict[str, object]:
        return {
            "parked": self.parked,
            "reason": self.reason,
            "idle_s": round(self.idle_s, 1),
            "threshold_s": round(self.threshold_s, 1),
            "freed_mib": round(self.freed_bytes / (1 << 20), 2),
            "terms": {k: round(v, 2) for k, v in sorted(self.terms.items())},
        }


def percentile(values: List[float], q: float) -> float:
    """Nearest-rank percentile. No interpolation, no numpy for six samples."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, min(len(ordered), int(round(q * len(ordered) + 0.5))))
    return ordered[rank - 1]


class IdleParkController:
    """Decides when the translator's assets leave the card, and brings them back.

    Thread-safe by construction: the sweeper decides on one thread while turns
    run on others, and a half-parked tenant is the one state that must never
    be observable from a request path.
    """

    def __init__(
        self,
        ledger: AudioAssetLedger,
        config: Optional[IdleParkConfig] = None,
        busy_probe: Optional[Callable[[], bool]] = None,
        clock: Callable[[], float] = time.monotonic,
        tenant_id: Optional[str] = None,
    ) -> None:
        self.config = config or IdleParkConfig()
        self.config.validate()
        self.ledger = ledger
        self.tenant_id = tenant_id or getattr(ledger, "tenant_id", "translator")
        #: True while work is in flight that must not be parked underneath.
        #: The idle clock alone cannot see a long-running turn, because a turn
        #: announces itself when it starts, not continuously.
        self._busy = busy_probe if busy_probe is not None else (lambda: False)
        self._clock = clock

        self._cv = threading.Condition(threading.RLock())
        self._state = ParkState.RESIDENT
        self._last_activity_s = clock()
        self._last_wake_done_s: Optional[float] = None
        self._gaps: Deque[float] = deque(maxlen=self.config.gap_window)
        #: True while the tenant has been parked since the last recorded
        #: activity, so the gap that ends a parked period is skipped.
        self._gap_spans_park = False

        #: Highest need rank fully restored in the wake now running. -1 while
        #: nothing has landed; WAKE_RANK_ALL once the stack is whole.
        self._restored_rank = -1
        self._wake_attempt = 0
        self._wake_error: Optional[BaseException] = None
        self._wake_error_attempt: Optional[int] = None
        self._wake_started_s = clock()
        self._first_serve_ms: Optional[float] = None

        self.parks = 0
        self.wakes = 0
        self.park_refusals = 0
        #: SPLIT latency. ``last_first_serve_ms`` is what a user waits for --
        #: the first rank, i.e. being able to serve the request at all --
        #: while ``last_wake_ms`` is the whole stack. Reporting only the
        #: second would price the feature at a cost nobody actually pays;
        #: reporting only the first would hide how long the card stays
        #: half-claimed.
        self.last_wake_ms: Optional[float] = None
        self.last_first_serve_ms: Optional[float] = None
        self.last_freed_bytes = 0
        self.last_decision: Optional[ParkDecision] = None

    # -- observation --------------------------------------------------------

    def notify_activity(self) -> None:
        """One request arrived. Records the inter-arrival gap and resets idle.

        Called from every path that will touch a model: the audio frames, the
        control messages, the REST endpoints that run a stage. Cheap on
        purpose -- it is on the hot path of every audio frame.
        """
        with self._cv:
            now = self._clock()
            gap = now - self._last_activity_s
            if self._gap_spans_park:
                # Not a conversational gap: it is the silence the park was
                # for. Folding it into the percentile would raise the
                # threshold by the length of the idle period, i.e. the tenant
                # would park less the longer it had just been idle.
                self._gap_spans_park = False
            elif gap > 0.0:
                self._gaps.append(gap)
            self._last_activity_s = now

    def idle_seconds(self) -> float:
        with self._cv:
            return max(0.0, self._clock() - self._last_activity_s)

    @property
    def state(self) -> ParkState:
        with self._cv:
            return self._state

    @property
    def parked(self) -> bool:
        return self.state is ParkState.PARKED

    # -- the threshold ------------------------------------------------------

    def measured_restore_s(self) -> float:
        """Worst measured restore across the assets, in seconds.

        Worst, not mean: a wake restores every parked asset, so what a request
        waits for is the sum -- and the sum is bounded below by the worst.
        Using the worst single asset keeps the term honest without pretending
        the assets restore in parallel (they do not; the ledger is serial).
        """
        best = 0.0
        for name in self.ledger.names():
            measured = self.ledger.get(name).measured_restore_ms
            if measured is not None:
                best = max(best, measured / 1000.0)
        if self.last_wake_ms is not None:
            best = max(best, self.last_wake_ms / 1000.0)
        return best

    def threshold(self) -> Tuple[float, Dict[str, float]]:
        """The park threshold in seconds, plus its three terms for the log."""
        cfg = self.config
        with self._cv:
            gaps = list(self._gaps)
        gap_term = 0.0
        if len(gaps) >= cfg.min_gap_samples:
            gap_term = percentile(gaps, 0.95) * cfg.gap_margin
        break_even_term = self.measured_restore_s() * cfg.break_even
        raw = max(cfg.floor_s, gap_term, break_even_term)
        threshold = min(max(raw, cfg.floor_s), cfg.ceiling_s)
        return threshold, {
            "floor_s": cfg.floor_s,
            "gap_p95_x_margin_s": gap_term,
            "break_even_s": break_even_term,
            "ceiling_s": cfg.ceiling_s,
            "gap_samples": float(len(gaps)),
        }

    def dwell_remaining_s(self) -> float:
        with self._cv:
            if self._last_wake_done_s is None:
                return 0.0
            elapsed = self._clock() - self._last_wake_done_s
            return max(0.0, self.config.dwell_s - elapsed)

    def evaluate(self) -> ParkDecision:
        """Should a park start now? Pure -- moves nothing, so it is loggable."""
        cfg = self.config
        idle_s = self.idle_seconds()
        threshold, terms = self.threshold()

        def no(reason: str) -> ParkDecision:
            return ParkDecision(
                parked=False,
                reason=reason,
                idle_s=idle_s,
                threshold_s=threshold,
                terms=terms,
            )

        if cfg.never_park:
            return no("parking is pinned off (never-park)")
        if not cfg.enabled:
            return no("idle park is disabled")
        state = self.state
        if state is not ParkState.RESIDENT:
            return no(f"nothing to do in state {state.value}")
        if not self.ledger.names():
            return no("no ledgered assets to park")
        if self._busy():
            return no("a turn is in flight")
        dwell = self.dwell_remaining_s()
        if dwell > 0.0:
            return no(f"anti-flap dwell has {dwell:.0f}s left")
        if idle_s < threshold:
            return no(f"idle {idle_s:.0f}s has not reached {threshold:.0f}s")
        return ParkDecision(
            parked=True,
            reason=f"idle {idle_s:.0f}s cleared the {threshold:.0f}s threshold",
            idle_s=idle_s,
            threshold_s=threshold,
            terms=terms,
        )

    # -- the transitions ----------------------------------------------------

    def tick(self) -> ParkDecision:
        """One sweeper step: evaluate, and park if the answer is yes."""
        decision = self.evaluate()
        if not decision.parked:
            self.last_decision = decision
            logger.debug("idle park: %s", decision.reason)
            return decision
        freed = self.park_now(decision.reason)
        decision = dataclasses.replace(decision, freed_bytes=freed)
        self.last_decision = decision
        return decision

    def park_now(self, reason: str = "explicit request") -> int:
        """Park every asset. Returns bytes freed; 0 when nothing was parked.

        Idempotent and race-safe: a second caller finding the tenant already
        parked, or a park already running, gets 0 rather than an error. Two
        planners racing must not be able to crash the tenant.
        """
        with self._cv:
            if self._state is not ParkState.RESIDENT:
                self.park_refusals += 1
                return 0
            if self._busy():
                self.park_refusals += 1
                return 0
            self._state = ParkState.PARKING
            self._cv.notify_all()

        freed = 0
        try:
            freed = int(self.ledger.park_all())
        except BaseException:
            # PARKED, not RESIDENT, on failure. A park that died part-way may
            # have moved some assets, and the wake path's ensure_resident is
            # exactly the repair for a partial state -- while claiming
            # RESIDENT would let the next turn run against a half-parked
            # module. Over-claiming parked is a wasted restore; under-claiming
            # it is a crash mid-turn.
            with self._cv:
                self._state = ParkState.PARKED
                self._gap_spans_park = True
                self._cv.notify_all()
            logger.exception("idle park failed part-way; treating as parked")
            raise

        with self._cv:
            self._state = ParkState.PARKED
            self._gap_spans_park = True
            self.parks += 1
            self.last_freed_bytes = freed
            self._cv.notify_all()

        logger.info(
            "parked the translator's audio assets: %.1f MiB freed (%s)",
            freed / (1 << 20), reason,
        )
        self._emit(
            residency.EVENT_PARK_COMPLETE,
            self.ledger.parked_bytes_by_device(),
            {"reason": reason, "freed_mib": round(freed / (1 << 20), 2)},
        )
        return freed

    def ensure_awake(
        self, up_to_rank: Optional[int] = None, timeout_s: Optional[float] = None
    ) -> float:
        """Wait until the assets a caller needs are back. Returns waited ms.

        ``up_to_rank`` is a NEED rank from :data:`ledger.DEFAULT_WAKE_RANKS`.
        A recognizer-stage caller passes 0 and is released the moment the
        recognizer is resident, while the talker and the codec keep restoring
        behind it. None means "the whole stack".

        Zero when already resident, which is the overwhelmingly common case --
        so this is safe to call on the hot path of every request, and it is
        the only correct place to call it: the alternative, waking from the
        sweeper, would have the tenant guess when the next request is coming.

        WHY THE RESTORE IS NOT A BARRIER. A wake needs SPACE on the card, and
        under #553 that space may first have to be made by the serving engine
        spilling something of its own. An all-or-nothing restore would make
        the first turn wait for the last byte of the last module even though
        the pipeline cannot reach that module for seconds. So the wake is
        staged: the wake-start event goes out FIRST carrying the per-card MiB
        needed, then the ranks come back in pipeline order, and each waiter is
        released at its own rank.

        WHAT STILL WAITS FOR #553: nothing here asks anyone to make room. In
        v1 the room already exists (the co-tenant's reserve was sized against
        this tenant's declared budget), so the wake always succeeds
        immediately. The sequence and the event contract are already the ones
        a room-granting consumer needs, so #553 adds a reaction rather than a
        redesign.

        ONE LOOP over the four states rather than a wait-then-act sequence.
        Every caller re-decides on every pass, so a caller that arrived during
        a park becomes the driver when the park finishes, and a wake whose
        driver died is picked up by the next waiter instead of deadlocking a
        thread that has no way to notice.
        """
        from sglang.srt.translator.ledger import WAKE_RANK_ALL

        needed = WAKE_RANK_ALL if up_to_rank is None else int(up_to_rank)
        budget = self.config.wake_timeout_s if timeout_s is None else timeout_s
        started = self._clock()
        deadline = started + budget

        #: The attempt this caller joined. A failure is reported to every
        #: caller already waiting when it happened -- its own attempt or a
        #: later one, since either way the wake it was waiting for did not
        #: happen -- and to nobody who arrived afterwards. Report it too
        #: narrowly and a waiter sails on believing the assets are back, which
        #: is a turn against meta tensors; report it too widely and one bad
        #: wake poisons the tenant for the life of the process.
        my_attempt: Optional[int] = None

        while True:
            drive = False
            with self._cv:
                self._last_activity_s = self._clock()
                if (
                    self._wake_error is not None
                    and my_attempt is not None
                    and self._wake_error_attempt is not None
                    and self._wake_error_attempt >= my_attempt
                ):
                    raise self._wake_error
                if self._state is ParkState.RESIDENT:
                    return (self._clock() - started) * 1000.0
                if self._state is ParkState.RESTORING:
                    if my_attempt is None:
                        my_attempt = self._wake_attempt
                    if self._restored_rank >= needed:
                        return (self._clock() - started) * 1000.0
                    self._wait_or_timeout(deadline, budget)
                    continue
                if self._state is ParkState.PARKING:
                    self._wait_or_timeout(deadline, budget)
                    continue
                # PARKED: this caller drives the wake.
                self._state = ParkState.RESTORING
                self._restored_rank = -1
                self._wake_attempt += 1
                my_attempt = self._wake_attempt
                self._wake_error = None
                self._wake_error_attempt = None
                self._wake_started_s = self._clock()
                self._first_serve_ms = None
                drive = True
                self._cv.notify_all()

            if drive:
                # BEFORE the first byte moves, and before this thread commits
                # to waiting: a consumer that has to free room can only act
                # while there is still time to act.
                self._emit(
                    residency.EVENT_WAKE_START,
                    self.ledger.parked_bytes_by_device(),
                    {
                        "reason": "a request arrived",
                        "ranks": list(self.ledger.parked_wake_ranks()),
                    },
                )
                threading.Thread(
                    target=self._wake_worker, name="translator-wake", daemon=True
                ).start()

    def _wait_or_timeout(self, deadline: float, budget: float) -> None:
        """Condition wait with a wall deadline. Caller holds the lock."""
        remaining = deadline - self._clock()
        if remaining <= 0.0:
            raise WakeTimeout(
                f"waited {budget:.0f}s in state {self._state.value} "
                f"(rank {self._restored_rank} restored); the mover is stuck"
            )
        self._cv.wait(remaining)

    def _wake_worker(self) -> None:
        """Restore rank by rank, releasing waiters as each rank lands."""
        from sglang.srt.translator.ledger import WAKE_RANK_ALL

        with self._cv:
            started = self._wake_started_s
            attempt = self._wake_attempt
        per_asset: Dict[str, float] = {}
        try:
            for rank in self.ledger.parked_wake_ranks():
                per_asset.update(self.ledger.restore_rank(rank))
                with self._cv:
                    self._restored_rank = rank
                    if self._first_serve_ms is None:
                        self._first_serve_ms = (self._clock() - started) * 1000.0
                    self._cv.notify_all()
        except BaseException as exc:  # noqa: BLE001 - handed to the waiters
            with self._cv:
                self._state = ParkState.PARKED
                self._wake_error = exc
                self._wake_error_attempt = attempt
                self._cv.notify_all()
            logger.exception("wake failed; the assets stay parked")
            return

        with self._cv:
            elapsed_ms = (self._clock() - started) * 1000.0
            # State and rank flip together, so a waiter for the whole stack
            # can never be released before the state says RESIDENT.
            self._state = ParkState.RESIDENT
            self._restored_rank = WAKE_RANK_ALL
            self._last_activity_s = self._clock()
            self._last_wake_done_s = self._last_activity_s
            self._gap_spans_park = True
            self.wakes += 1
            self.last_wake_ms = elapsed_ms
            self.last_first_serve_ms = self._first_serve_ms
            self.last_freed_bytes = 0
            self._cv.notify_all()

        logger.info(
            "woke the translator's audio assets: first stage %.0f ms, "
            "full stack %.0f ms (%s)",
            self.last_first_serve_ms or 0.0,
            elapsed_ms,
            ", ".join(f"{k} {v:.0f}ms" for k, v in sorted(per_asset.items())) or "-",
        )
        self._emit(
            residency.EVENT_WAKE_COMPLETE,
            self.ledger.resident_bytes_by_device(),
            {
                "wake_ms": round(elapsed_ms, 1),
                "first_serve_ms": (
                    None if self.last_first_serve_ms is None
                    else round(self.last_first_serve_ms, 1)
                ),
                "per_asset_ms": {k: round(v, 1) for k, v in per_asset.items()},
            },
        )

    def _emit(
        self, event: str, by_device: Dict[str, int], detail: Dict[str, object]
    ) -> None:
        try:
            residency.emit(
                residency.ResidencyEvent(
                    tenant_id=self.tenant_id,
                    event=event,
                    cards=residency.cards_from_bytes(by_device),
                    detail=detail,
                )
            )
        except Exception:  # noqa: BLE001 - telemetry never breaks the tenant
            logger.exception("could not emit residency event %s", event)

    # -- reporting ----------------------------------------------------------

    def to_json(self) -> Dict[str, object]:
        threshold, terms = self.threshold()
        with self._cv:
            gaps = list(self._gaps)
        return {
            "state": self.state.value,
            "enabled": self.config.enabled,
            "never_park": self.config.never_park,
            "idle_s": round(self.idle_seconds(), 1),
            "threshold_s": round(threshold, 1),
            "terms": {k: round(v, 2) for k, v in sorted(terms.items())},
            "dwell_remaining_s": round(self.dwell_remaining_s(), 1),
            "recent_gaps_s": [round(g, 2) for g in gaps[-8:]],
            "parks": self.parks,
            "wakes": self.wakes,
            "park_refusals": self.park_refusals,
            "last_wake_ms": (
                None if self.last_wake_ms is None else round(self.last_wake_ms, 1)
            ),
            "last_first_serve_ms": (
                None if self.last_first_serve_ms is None
                else round(self.last_first_serve_ms, 1)
            ),
            "parked_mib": round(self.last_freed_bytes / (1 << 20), 2),
            "measured_restore_s": round(self.measured_restore_s(), 3),
            "last_decision": (
                None if self.last_decision is None else self.last_decision.to_json()
            ),
        }
