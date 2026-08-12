# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""#604: notice that serving died, and say so -- with a defined restart policy.

The failure this exists for is NOT the crash. A crashed process leaves an
exit status, and systemd already restarts it correctly. The failure that cost
this rig real time is the WEDGE (#622/#649 family): the process is alive, the
port is open, ``/health`` returns 200, and every generation hangs forever. To
every mechanism that watches processes or ports, a wedged server is a healthy
server. Only asking it to generate something tells the two apart.

So the liveness definition here is deliberately expensive: **a real generation
probe**. HTTP 200 is necessary and nowhere near sufficient.

Three structural decisions, each paid for by a specific incident:

**The watchdog never starts serving.** #638: a serving process spawned by the
watchdog inherits the watchdog's cgroup -- ``setsid`` escapes the session, not
the cgroup -- so every watchdog restart killed the live production server
alongside it (observed 2026-08-06 00:24). The installed unit worked around it
with ``KillMode=process``. This module removes the cause instead: the only
restart action it can emit is "ask systemd to restart the serving unit", so
the new process is started BY systemd, into the serving unit's own cgroup.
A detector cannot leak its cgroup into something it does not spawn.

**The state machine is a pure function.** :func:`step` takes a state, an
observation and a clock reading, and returns a new state plus an action. It
opens no socket and reads no clock of its own. That is what makes the wedge
path, the give-up path and the backoff ladder reachable in a unit test rather
than only during an outage at 03:00.

**Giving up is a state, not a silence.** After ``max_restarts`` inside the
window the watchdog stops restarting and keeps alarming. A lane that fails to
come back is a human problem; thrashing it hides the evidence and burns the
cards. ``GIVEN_UP`` is loud and terminal until an operator intervenes.
"""

from __future__ import annotations

import dataclasses
from typing import Optional, Tuple

__all__ = [
    "BOOTING", "HEALTHY", "SUSPECT", "WEDGED", "DEAD", "GIVEN_UP", "PHASES",
    "ACT_NONE", "ACT_PROBE_GENERATION", "ACT_RESTART", "ACT_ALARM", "ACTIONS",
    "Observation", "WatchdogState", "Decision", "Policy", "step", "initial",
]

# -- phases ----------------------------------------------------------------

#: A boot is in progress and slowness is expected. JIT cold caches (#172/#615)
#: make a legitimate first boot minutes long; without this phase the watchdog
#: would kill every cold boot just before it finished.
BOOTING = "booting"
HEALTHY = "healthy"
#: At least one generation probe failed, but not enough to convict. A single
#: failed probe is not a wedge: a long legitimate batch can starve it.
SUSPECT = "suspect"
#: Confirmed generating-dead while still answering HTTP. The state that only
#: a generation probe can reach.
WEDGED = "wedged"
#: Not answering at all, past the boot grace.
DEAD = "dead"
#: Restart budget exhausted. Loud, and no longer restarting.
GIVEN_UP = "given_up"

PHASES = (BOOTING, HEALTHY, SUSPECT, WEDGED, DEAD, GIVEN_UP)

# -- actions ---------------------------------------------------------------

ACT_NONE = "none"
ACT_PROBE_GENERATION = "probe_generation"
#: "Ask systemd to restart the serving unit." Never "spawn a server".
ACT_RESTART = "restart"
ACT_ALARM = "alarm"

ACTIONS = (ACT_NONE, ACT_PROBE_GENERATION, ACT_RESTART, ACT_ALARM)


@dataclasses.dataclass(frozen=True)
class Policy:
    """The tunables, already validated by config parsing."""

    poll_s: float = 20.0
    generation_probe_s: float = 120.0
    wedge_confirmations: int = 3
    backoff_s: Tuple[int, ...] = (30, 60, 120, 300, 600)
    max_restarts: int = 5
    restart_window_s: float = 3600.0
    #: How long after a restart the lane is allowed to be unreachable before
    #: it counts as DEAD rather than BOOTING.
    boot_grace_s: float = 1800.0

    def backoff_for(self, restarts_done: int) -> int:
        """Delay before the Nth restart. The ladder's last value repeats."""
        if not self.backoff_s:
            return 0
        i = min(max(restarts_done, 0), len(self.backoff_s) - 1)
        return self.backoff_s[i]


@dataclasses.dataclass(frozen=True)
class Observation:
    """One tick's view of the lane.

    ``generation`` is tri-state on purpose. ``None`` means "not probed this
    tick", which is the common case -- the probe is expensive and runs on its
    own cadence. Collapsing it into ``False`` would make every cheap tick look
    like evidence of a wedge; collapsing it into ``True`` would make the
    watchdog blind. A missing measurement is not a measurement.
    """

    port_open: bool
    api_ok: bool
    generation: Optional[bool] = None


@dataclasses.dataclass(frozen=True)
class WatchdogState:
    phase: str = BOOTING
    #: Consecutive FAILED generation probes. Reset by any success.
    gen_failures: int = 0
    #: Monotonic timestamps of restarts we asked for, within the window.
    restarts: Tuple[float, ...] = ()
    last_gen_probe_at: Optional[float] = None
    #: Set when we enter BOOTING; the lane may be unreachable until then.
    boot_deadline: Optional[float] = None
    #: Earliest time the next restart may be requested (backoff).
    next_restart_at: float = 0.0

    def restarts_in_window(self, now: float, window_s: float) -> int:
        return sum(1 for t in self.restarts if now - t < window_s)


@dataclasses.dataclass(frozen=True)
class Decision:
    state: WatchdogState
    action: str
    #: One line, already phrased for the log. Always names the evidence.
    reason: str

    @property
    def alarming(self) -> bool:
        return self.action in (ACT_RESTART, ACT_ALARM)


def initial(now: float, policy: Policy) -> WatchdogState:
    """A watchdog that has just started assumes a boot may be in progress.

    Starting in BOOTING rather than HEALTHY matters when the watchdog and the
    serving unit come up together at host boot: an optimistic initial state
    would see an unreachable port and immediately count a restart against a
    lane that is loading weights.
    """
    return WatchdogState(phase=BOOTING, boot_deadline=now + policy.boot_grace_s)


def _prune(restarts: Tuple[float, ...], now: float,
           window_s: float) -> Tuple[float, ...]:
    return tuple(t for t in restarts if now - t < window_s)


def _want_restart(state: WatchdogState, now: float, policy: Policy,
                  phase: str, why: str) -> Decision:
    """Common path for both death kinds: budget, then backoff, then act."""
    restarts = _prune(state.restarts, now, policy.restart_window_s)
    done = len(restarts)

    if done >= policy.max_restarts:
        return Decision(
            state=dataclasses.replace(state, phase=GIVEN_UP, restarts=restarts),
            action=ACT_ALARM,
            reason=(f"GIVEN_UP after {done} restarts in "
                    f"{int(policy.restart_window_s)}s: {why}. No further "
                    f"restarts; operator intervention required"))

    if now < state.next_restart_at:
        return Decision(
            state=dataclasses.replace(state, phase=phase, restarts=restarts),
            action=ACT_NONE,
            reason=(f"{phase}: {why}; holding off "
                    f"{state.next_restart_at - now:.0f}s for backoff"))

    delay = policy.backoff_for(done + 1)
    new = dataclasses.replace(
        state,
        phase=BOOTING,
        gen_failures=0,
        restarts=restarts + (now,),
        boot_deadline=now + policy.boot_grace_s,
        next_restart_at=now + delay,
        last_gen_probe_at=None,
    )
    return Decision(
        state=new, action=ACT_RESTART,
        reason=(f"RESTART #{done + 1} ({phase}): {why}; next restart no "
                f"sooner than {delay}s from now"))


def step(state: WatchdogState, obs: Observation, now: float,
         policy: Policy) -> Decision:
    """One tick. Pure: no clock, no socket, no side effect.

    Evaluation order is the contract:

    1. ``GIVEN_UP`` absorbs -- once we have stopped restarting, nothing an
       observation says starts it again. Recovery is an operator action.
    2. Reachability, because a generation verdict about an unreachable server
       is meaningless.
    3. The generation verdict, which is the only thing that can find a wedge.
    """
    # 1. absorbing state
    if state.phase == GIVEN_UP:
        if obs.api_ok and obs.generation is True:
            # It came back on its own (an operator restarted it, or a long
            # stall cleared). Resume supervision rather than stay blind.
            return Decision(
                state=dataclasses.replace(state, phase=HEALTHY, gen_failures=0,
                                          last_gen_probe_at=now),
                action=ACT_NONE,
                reason="recovered from GIVEN_UP: generation probe succeeded")
        return Decision(state=state, action=ACT_ALARM,
                        reason="still GIVEN_UP; not restarting")

    reachable = obs.api_ok

    # 2. unreachable
    if not reachable:
        if state.phase == BOOTING:
            deadline = state.boot_deadline
            if deadline is not None and now < deadline:
                return Decision(
                    state=state, action=ACT_NONE,
                    reason=(f"booting: not answering yet, "
                            f"{deadline - now:.0f}s of grace left"
                            + (" (port is open)" if obs.port_open else "")))
            return _want_restart(
                state, now, policy, DEAD,
                f"boot grace of {int(policy.boot_grace_s)}s expired with no "
                f"API response")
        return _want_restart(state, now, policy, DEAD,
                             "API stopped answering"
                             + (" while the port stays open"
                                if obs.port_open else " and the port is shut"))

    # 3. reachable -- now the only question is whether it GENERATES.
    if obs.generation is None:
        due = (state.last_gen_probe_at is None
               or now - state.last_gen_probe_at >= policy.generation_probe_s)
        phase = state.phase
        if phase in (BOOTING, DEAD):
            # It answers again. Do not call it healthy on an HTTP 200 -- that
            # is precisely the claim the wedge falsifies. Demand a generation.
            return Decision(
                state=dataclasses.replace(state, phase=SUSPECT),
                action=ACT_PROBE_GENERATION,
                reason="API answers again; a generation probe must confirm it "
                       "before this counts as healthy")
        if due:
            return Decision(state=state, action=ACT_PROBE_GENERATION,
                            reason="generation probe due")
        return Decision(state=state, action=ACT_NONE,
                        reason=f"{phase}: API ok, next generation probe in "
                               f"{policy.generation_probe_s - (now - (state.last_gen_probe_at or now)):.0f}s")

    if obs.generation:
        was = state.phase
        new = dataclasses.replace(state, phase=HEALTHY, gen_failures=0,
                                  last_gen_probe_at=now)
        if was == HEALTHY:
            reason = "healthy: generation probe succeeded"
        else:
            reason = f"recovered ({was} -> healthy): generation probe succeeded"
        return Decision(state=new, action=ACT_NONE, reason=reason)

    # A failed generation probe against a server that answers HTTP. This is
    # the wedge signature.
    fails = state.gen_failures + 1
    if fails < policy.wedge_confirmations:
        return Decision(
            state=dataclasses.replace(state, phase=SUSPECT, gen_failures=fails,
                                      last_gen_probe_at=now),
            action=ACT_NONE,
            reason=(f"suspect: generation probe failed {fails}/"
                    f"{policy.wedge_confirmations} while the API returns 200"))

    convicted = dataclasses.replace(state, gen_failures=fails,
                                    last_gen_probe_at=now)
    return _want_restart(
        convicted, now, policy, WEDGED,
        f"WEDGED -- {fails} consecutive generation probes failed while the "
        f"API kept answering (the #622 signature: HTTP 200, no tokens)")
