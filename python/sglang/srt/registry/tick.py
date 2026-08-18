# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#305: the periodic idle-set evaluation the arbiter names as missing.

``arbiter.py``'s ``return_to_idle`` documents its own absent caller in its
docstring -- *"Idle handling is explicit rather than a background thread: the
control plane calls it on its own tick"* -- and until now there was no tick.
This is it.

It is deliberately NOT ``return_to_idle`` on a timer. That actuator takes every
engine outside ``default_hot`` straight to COLD, which throws away the middle
rung on exactly the class that implements it: a Class 1 engine parked at
TEIL-HOT still holds its process, its CUDA context and its compiled graphs, and
comes back with a resume rather than a boot. Walking one rung at a time is the
whole point of having a ladder.

**Every step this module takes is an edge ``ladder.py`` declares reachable.**
It never asks for a transition and hopes; it asks ``ladder.step_down_target``
which rung this class can actually reach next, and if the answer is that there
is none, the engine is left alone and the refusal is logged with the
architecture's own reason. A rung skipped because the class does not implement
it (Class 1 has no WARM, so TEIL-HOT steps to COLD) is reported in the
decision, never silent.

**Default off.** ``interval_s`` defaults to 0 and ``start()`` on a disabled
tick is a no-op that says so. Today's boots have exactly one model and no
control plane; nothing about them changes until an operator passes
``--tick-interval-s``.

What it will NOT do:

* It does not promote. Waking a model is the request path's job (#305 binding);
  a tick that promoted on its own would be the autonomous-promotion policy of
  cut 4, whose gate is recorded UNFULFILLED in #375.
* It does not call ``#286``'s ``RealMovementBackend``. The steps here are
  ladder edges driven through the arbiter's existing actuators; the per-item
  mover has zero production callers and this does not become its first.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

from sglang.srt.registry import ladder
from sglang.srt.registry.ladder import LadderRefusal

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sglang.srt.registry.arbiter import EngineRegistry

logger = logging.getLogger(__name__)

#: Environment override for the interval, so a deployment can turn the tick on
#: without editing the launch line. Absent or 0 means off, same as the flag.
TICK_INTERVAL_ENV = "SGLANG_REGISTRY_TICK_INTERVAL_S"

#: What a decision can be. ``stepped`` is the only one that moved anything.
STEPPED = "stepped"
HELD = "held"
REFUSED = "refused"
FAILED = "failed"


@dataclass(frozen=True)
class TickDecision:
    """One engine, one tick, and the reason -- including for doing nothing.

    A tick that only logged what it changed would be unreadable: the
    interesting question when a model is still hot an hour later is which of
    the five hold reasons applied, and that answer has to be in the report
    rather than reconstructed from a memory graph.
    """

    engine_id: str
    action: str
    src_rung: str
    dst_rung: str | None = None
    reason: str = ""
    skipped_rungs: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "engine_id": self.engine_id,
            "action": self.action,
            "src_rung": self.src_rung,
            "dst_rung": self.dst_rung,
            "reason": self.reason,
            "skipped_rungs": list(self.skipped_rungs),
        }


@dataclass(frozen=True)
class TickReport:
    ts: float
    decisions: tuple[TickDecision, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> tuple[str, ...]:
        return tuple(d.engine_id for d in self.decisions if d.action == STEPPED)

    @property
    def refused(self) -> tuple[str, ...]:
        return tuple(d.engine_id for d in self.decisions if d.action == REFUSED)

    def of(self, engine_id: str) -> TickDecision | None:
        for decision in self.decisions:
            if decision.engine_id == engine_id:
                return decision
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "ts": self.ts,
            "changed": list(self.changed),
            "refused": list(self.refused),
            "decisions": [d.to_json() for d in self.decisions],
        }


class ControlTick:
    """Periodic idle-set evaluation over one registry.

    ``evaluate()`` decides and moves nothing; ``run_once()`` decides and then
    executes. They are split because the decision is the part worth testing
    exhaustively, and a test of the decision must not need an adapter that can
    boot.
    """

    def __init__(
        self,
        registry: "EngineRegistry",
        *,
        interval_s: float = 0.0,
        idle_after_s: float | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.registry = registry
        self.interval_s = float(interval_s or 0.0)
        if self.interval_s < 0:
            raise ValueError("--tick-interval-s must not be negative")
        self._idle_after_s = idle_after_s
        self._clock = clock or getattr(registry, "_clock", time.time)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.last_report: TickReport | None = None

    @property
    def enabled(self) -> bool:
        return self.interval_s > 0.0

    @property
    def idle_after_s(self) -> float:
        if self._idle_after_s is not None:
            return float(self._idle_after_s)
        return float(getattr(self.registry, "idle_after_s", 900.0))

    # -- decision ----------------------------------------------------------

    def evaluate(self) -> TickReport:
        """Decide, for every registered engine, without moving any of them."""
        now = self._clock()
        wanted = set(self.registry.default_hot)
        decisions: list[TickDecision] = []
        for instance in self.registry.engines():
            decisions.append(self._decide(instance, wanted=wanted, now=now))
        return TickReport(ts=now, decisions=tuple(decisions))

    def _decide(self, instance: Any, *, wanted: set, now: float) -> TickDecision:
        engine_id = instance.engine_id
        try:
            src = ladder.rung_of_state(instance.state)
        except LadderRefusal as exc:
            return TickDecision(
                engine_id, REFUSED, str(instance.state), reason=str(exc)
            )

        if instance.spec.pinned:
            return TickDecision(engine_id, HELD, src, reason="pinned")
        if engine_id in wanted:
            return TickDecision(engine_id, HELD, src, reason="in the default_hot set")
        inflight = self._inflight(engine_id)
        if inflight:
            return TickDecision(
                engine_id,
                HELD,
                src,
                reason=f"{inflight} request(s) in flight",
            )
        idle_for = now - float(instance.last_used_ts or 0.0)
        if instance.last_used_ts and idle_for < self.idle_after_s:
            return TickDecision(
                engine_id,
                HELD,
                src,
                reason=(
                    f"used {idle_for:.0f}s ago, below the {self.idle_after_s:.0f}s "
                    "idle threshold"
                ),
            )

        klass = instance.spec.adapter
        try:
            dst = ladder.step_down_target(klass, src)
        except LadderRefusal as exc:
            # Loud: the registry believes this engine sits on a rung its own
            # class does not declare. That is a table/adapter disagreement, not
            # a routine hold, and it is refused rather than guessed around.
            logger.warning("registry tick: %s refused -- %s", engine_id, exc)
            return TickDecision(engine_id, REFUSED, src, reason=str(exc))
        if dst is None:
            return TickDecision(
                engine_id, HELD, src, reason=f"{src} is the lowest rung {klass} has"
            )
        skipped = ladder.skipped_rungs(klass, src, dst)
        if skipped:
            logger.info(
                "registry tick: %s steps %s -> %s, over %s which %s does not implement",
                engine_id,
                src,
                dst,
                list(skipped),
                klass,
            )
        return TickDecision(
            engine_id, STEPPED, src, dst_rung=dst, skipped_rungs=skipped
        )

    # -- execution ---------------------------------------------------------

    def run_once(self) -> TickReport:
        """Evaluate, then take exactly the steps the evaluation named."""
        planned = self.evaluate()
        executed: list[TickDecision] = []
        for decision in planned.decisions:
            if decision.action != STEPPED:
                executed.append(decision)
                continue
            assert decision.dst_rung is not None
            target = ladder.RUNG_TO_STATE[decision.dst_rung]
            try:
                # Re-checked at the moment of movement, not only at decision
                # time: this is the gate the determination asked for, and a
                # gate that only ran in the planner would be bypassed by any
                # future caller of the executor.
                klass = self.registry.instance(decision.engine_id).spec.adapter
                ladder.check_transition(klass, decision.src_rung, decision.dst_rung)
                self.registry.ensure_state(decision.engine_id, target)
            except LadderRefusal as exc:
                logger.warning(
                    "registry tick: refusing to move %s %s -> %s: %s",
                    decision.engine_id,
                    decision.src_rung,
                    decision.dst_rung,
                    exc,
                )
                executed.append(
                    TickDecision(
                        decision.engine_id,
                        REFUSED,
                        decision.src_rung,
                        dst_rung=decision.dst_rung,
                        reason=str(exc),
                    )
                )
                continue
            except Exception as exc:  # noqa: BLE001 - one engine must not stop the tick
                logger.warning(
                    "registry tick: %s %s -> %s failed: %s",
                    decision.engine_id,
                    decision.src_rung,
                    decision.dst_rung,
                    exc,
                )
                executed.append(
                    TickDecision(
                        decision.engine_id,
                        FAILED,
                        decision.src_rung,
                        dst_rung=decision.dst_rung,
                        reason=str(exc),
                    )
                )
                continue
            logger.info(
                "registry tick: %s %s -> %s",
                decision.engine_id,
                decision.src_rung,
                decision.dst_rung,
            )
            executed.append(decision)
        report = TickReport(ts=planned.ts, decisions=tuple(executed))
        self.last_report = report
        return report

    # -- background --------------------------------------------------------

    def start(self) -> bool:
        """Start the background thread. Returns whether one is now running."""
        if not self.enabled:
            logger.info(
                "registry tick: disabled (interval 0). Idle demotion stays "
                "operator-driven via POST /registry/idle."
            )
            return False
        if self._thread is not None and self._thread.is_alive():
            return True
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="registry-control-tick", daemon=True
        )
        self._thread.start()
        logger.info(
            "registry tick: every %.0fs, idle threshold %.0fs",
            self.interval_s,
            self.idle_after_s,
        )
        return True

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)

    def _loop(self) -> None:  # pragma: no cover - exercised by start/stop test
        while not self._stop.wait(self.interval_s):
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001 - a tick must not die silently
                logger.exception("registry tick: evaluation failed: %s", exc)

    # -- helpers -----------------------------------------------------------

    def _inflight(self, engine_id: str) -> int:
        getter = getattr(self.registry, "inflight", None)
        if getter is None:
            return 0
        try:
            return int(getter(engine_id))
        except Exception:  # noqa: BLE001 - accounting must never block a tick
            return 0


def interval_from_env(default: float = 0.0) -> float:
    """Tick interval from the environment, off unless it says otherwise."""
    import os  # noqa: PLC0415

    raw = (os.environ.get(TICK_INTERVAL_ENV) or "").strip()
    if not raw:
        return float(default)
    try:
        return max(0.0, float(raw))
    except ValueError:
        logger.warning(
            "registry tick: %s=%r is not a number; the tick stays off",
            TICK_INTERVAL_ENV,
            raw,
        )
        return 0.0


def build_tick(
    registry: "EngineRegistry",
    *,
    interval_s: float | None = None,
    idle_after_s: float | None = None,
) -> ControlTick:
    resolved = interval_from_env() if interval_s is None else float(interval_s)
    return ControlTick(registry, interval_s=resolved, idle_after_s=idle_after_s)


__all__: Sequence[str] = (
    "ControlTick",
    "TickDecision",
    "TickReport",
    "STEPPED",
    "HELD",
    "REFUSED",
    "FAILED",
    "TICK_INTERVAL_ENV",
    "build_tick",
    "interval_from_env",
)
