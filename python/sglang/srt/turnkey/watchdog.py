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
    #: Generation-based liveness probing. RETIRED by user order 2026-08-14:
    #: nothing on this box may prove liveness by generating on a timer.
    #: Detection stays passive -- port open, ``/get_model_info``, boot-log
    #: age, NVML -- and a real generation is run ONCE, by hand, at teardown
    #: or restore. The default is False so that a bare ``Policy()`` cannot
    #: reintroduce the probe by omission; the config key of the same name
    #: is the only way to turn it back on.
    generation_probe_enabled: bool = False
    #: #799: consume the #699/#739 admission-wedge verdict, published passively
    #: by the scheduler that computes it (``managers/wedge_status.py``).
    #:
    #: This is what replaces the retired generation probe, and it is ON by
    #: default deliberately. The probe was retired because proving liveness by
    #: GENERATING on a timer is forbidden on this box; reading a file the
    #: scheduler already wrote is not that. Leaving the replacement off by
    #: default would recreate the exact condition this flag exists to end --
    #: a correct detector with no consumer, which is how boot 0822_0829 spent
    #: thirteen minutes serving nobody while every supervisor read "healthy".
    wedge_signal_enabled: bool = True
    #: Consecutive confirmations required, shared by the generation probe and
    #: the wedge signal. ONE budget, not two: ``_systemctl_restart`` documents
    #: what two rate limiters in series cost when only one of them is visible.
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
    #: #799: the scheduler's own admission-wedge verdict, read passively from
    #: the file it publishes. Tri-state for the same reason ``generation`` is:
    #: ``None`` means "not measured this tick" -- export off, no file yet, or
    #: every file stale -- and must never be read as "fine".
    #:
    #: What this channel CANNOT see, stated here rather than discovered later:
    #: it carries ``admission_wedge_verdict``'s verdict unchanged, and that
    #: function returns "not wedged" whenever ``running > 0``
    #: (``invariant_checker.py``). The #536 fast-lane starvation class -- a
    #: request starved behind a co-tenant that is genuinely running -- is
    #: therefore invisible to this watchdog too. Transporting a verdict does
    #: not widen it.
    wedged: Optional[bool] = None


@dataclasses.dataclass(frozen=True)
class WatchdogState:
    phase: str = BOOTING
    #: Consecutive FAILED generation probes. Reset by any success.
    gen_failures: int = 0
    #: #799: consecutive ticks on which the scheduler published a wedge.
    #: A separate counter from ``gen_failures`` because they are separate
    #: pieces of evidence; sharing one would make a single failed generation
    #: probe and a single wedge report add up to a conviction neither earned.
    wedge_hits: int = 0
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
        wedge_hits=0,
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
        recovered_by_wedge_signal = (policy.wedge_signal_enabled
                                     and obs.wedged is False)
        if obs.api_ok and (obs.generation is True or recovered_by_wedge_signal):
            # It came back on its own (an operator restarted it, or a long
            # stall cleared). Resume supervision rather than stay blind.
            #
            # #799 added the second door on purpose. With the generation probe
            # retired, ``obs.generation`` is permanently None, so the original
            # condition could never be satisfied: GIVEN_UP was absorbing in
            # the strict sense -- an operator who fixed the lane got a
            # watchdog that kept alarming and never supervised again. A
            # POSITIVE no-wedge verdict from the scheduler is evidence of the
            # same kind and reopens supervision; a missing verdict (None) does
            # not, because that is not a measurement.
            return Decision(
                state=dataclasses.replace(state, phase=HEALTHY, gen_failures=0,
                                          wedge_hits=0, last_gen_probe_at=now),
                action=ACT_NONE,
                reason=("recovered from GIVEN_UP: "
                        + ("the scheduler reports no admission wedge"
                           if recovered_by_wedge_signal
                           else "generation probe succeeded")))
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

    # 3. reachable, and the scheduler has told us what it sees (#799).
    #
    #    This is evaluated BEFORE reachability is allowed to mean "healthy",
    #    because the whole failure class is a server that is reachable and
    #    serving nobody. An HTTP 200 outranking the scheduler's own wedge
    #    verdict would reproduce the blindness this channel exists to remove.
    #
    #    Restart is the action, and the reasoning is not "restart is what
    #    watchdogs do". The cheap in-process remedy already ran and lost: the
    #    detector makes ONE forced-admission attempt per episode
    #    (``invariant_checker``'s #788 rung) sixty seconds before this
    #    watchdog can convict, and boot 0822_0829 alarmed 146 times after it
    #    with zero decode batches. The wedge is also not self-clearing --
    #    #698 measured 52 minutes of it. What bounds the damage is not
    #    refusing to restart, it is the budget below: ``max_restarts`` inside
    #    ``restart_window_s`` and then GIVEN_UP, loud and terminal. A lane
    #    that wedges again after five restarts is a human problem, and
    #    thrashing it destroys the evidence needed to solve it.
    if policy.wedge_signal_enabled and obs.wedged is not None:
        if obs.wedged:
            hits = state.wedge_hits + 1
            if hits < policy.wedge_confirmations:
                return Decision(
                    state=dataclasses.replace(state, phase=SUSPECT,
                                              wedge_hits=hits),
                    action=ACT_NONE,
                    reason=(f"suspect: the scheduler reports an admission "
                            f"wedge {hits}/{policy.wedge_confirmations} while "
                            f"the API returns 200"))
            return _want_restart(
                dataclasses.replace(state, wedge_hits=hits), now, policy, WEDGED,
                f"WEDGED -- the scheduler published an admission-wedge verdict "
                f"on {hits} consecutive polls while the API kept answering "
                f"(#699/#739 signal: queue age against first-token progress, "
                f"which forward_ct and health-200 cannot see)")
        was = state.phase
        return Decision(
            state=dataclasses.replace(state, phase=HEALTHY, wedge_hits=0,
                                      gen_failures=0),
            action=ACT_NONE,
            reason=("healthy: the scheduler reports no admission wedge"
                    if was == HEALTHY else
                    f"recovered ({was} -> healthy): the scheduler reports no "
                    "admission wedge"))

    # 3b. reachable, no wedge measurement. With the generation probe retired,
    #    passive reachability IS the verdict, and what that gives up is named
    #    rather than hidden:
    #    with neither a generation verdict nor a wedge verdict, an HTTP 200 is
    #    the only evidence there is, and the #622 wedge (HTTP 200, no tokens)
    #    is invisible again. Reaching this branch is therefore a DEGRADED
    #    supervision mode, not the intended one: it means the scheduler is
    #    publishing nothing -- export disabled, or a publisher that stopped.
    #    Crash detection is unaffected either way; a lane that stops answering
    #    still reaches DEAD and still restarts, above.
    if obs.generation is None and not policy.generation_probe_enabled:
        was = state.phase
        new = dataclasses.replace(state, phase=HEALTHY, gen_failures=0,
                                  wedge_hits=0)
        return Decision(
            state=new, action=ACT_NONE,
            reason=("healthy: API answers; generation probe retired"
                    if was == HEALTHY else
                    f"recovered ({was} -> healthy): API answers; generation "
                    "probe retired"))

    # 3b. reachable and the probe is enabled -- the only question is whether
    #     it GENERATES.
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
