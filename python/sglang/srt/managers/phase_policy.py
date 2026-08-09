# Copyright 2023-2024 SGLang Team
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
"""#631 Route A -- the automatic phase policy.

Route A gives one instance two layouts of the same three ranks: a PP layout
that prefills fast, and a TP layout that decodes fast (speculation and the
decode CUDA graphs exist only there). Until now the choice between them was
a human POST to ``/phase_flip``, so a live server rested in whatever layout
the last manual call left it in. This module is the policy that makes the
choice automatically.

It is deliberately NARROW. The #363 regime controller already owns a general
"name the load, pick a stage" machine and even has the phase axis wired
(``regime_act._stage_wants_phase``), but its ``act`` mode is refused at parse
time without the four-item DESIGN_363 entry-gate evidence AND it requires a
#297 KV-reshard or #330 VRAM-dial actuator to be wired -- it does not count
the phase flip as an actuator at all. Routing this policy through it would
mean clearing a large evidence gate and granting a production server
permission to move its KV placement and VRAM budget as a side effect of
wanting a layout change. So this policy actuates ONE axis and nothing else,
behind its own flag. #363 remains the general answer.

THE RESTING STATE IS THE PREFILL LAYOUT (PP), by design
-------------------------------------------------------
Idle flips are free -- no request is waiting on them -- and the first thing
any request does is prefill. Resting in PP therefore means a long prompt
arriving at a quiet server gets PP-class prefill with ZERO flip cost inside
its TTFT. Resting in TP would put a ~3.2 s round trip in the latency path of
exactly the requests that are most latency-sensitive to it. The trade is
real but one-sided for this deployment: TP-resting only wins for
short-prompt / short-output traffic, where the prefill is too small to repay
a flip anyway (and such traffic simply runs in whatever layout is current --
see the threshold below). ``rest_state`` makes it configurable rather than
assumed.

THE THRESHOLD N, AND WHY IT IS A BREAK-EVEN AND NOT A GUESS
-----------------------------------------------------------
Flipping is not free, so a prefill should only drag the server out of TP if
the prefill is big enough to repay the flip. With

    C  = round-trip flip cost, seconds (measured, client-observed)
    X  = TP-phase prefill throughput, tok/s (measured)
    P  = PP-phase prefill throughput, tok/s (measured)

a prefill of n tokens costs ``n/X`` in TP and ``C + n/P`` if we flip first.
Flipping pays off when

    n/X  >  C + n/P      <=>      n  >  C / (1/X - 1/P)

so the break-even token count is

    N = C / (1/X - 1/P)

which is ``DEFAULT_FLIP_TOKENS`` below, computed from the measured
constants. NOTE the sign condition: this is only meaningful when ``X < P``
(TP prefill slower than PP prefill). If a measurement ever shows ``X >= P``,
the premise of Route A's prefill layout is gone and the policy REFUSES to
invent a threshold rather than silently emitting a negative or infinite N --
see ``break_even_tokens``.

N alone does not bound thrash: a burst of arrivals each just above N would
flip on every burst. So the policy carries an independent MINIMUM DWELL
timer -- no flip within ``min_dwell_s`` of the previous one, whatever the
token count says. The two controls are orthogonal on purpose: N is about
whether a flip is worth it for one prefill, dwell is about how often the
server is allowed to change its mind.

REPLICATION
-----------
``decide`` is a PURE function of its inputs and holds no rank-local state
beyond the timestamps threaded through ``PhasePolicyState``. That is not
stylistic: ``PhaseFlipRuntime.arm`` is a replicated call and the flip only
commits at a consensus boundary where EVERY rank is armed, so a policy that
reached different verdicts on different ranks would arm a flip that can
never commit and would park requests until the deadline abandoned it. Every
input to ``decide`` must therefore be rank-replicated, and the caller is
responsible for that guarantee (see ``phase_policy_inputs`` in the
scheduler). Keeping the decision pure is what makes that guarantee auditable
by reading one function.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-POLICY"

# Layout names, matching phase_flip_runtime.PHASE_PP / PHASE_TP. Imported
# lazily by callers; duplicated here as plain strings so this module stays
# importable (and unit-testable) without the flip runtime.
PHASE_PP = "pp"
PHASE_TP = "tp"
PP_TO_TP = "pp_to_tp"
TP_TO_PP = "tp_to_pp"

# Which layout the server returns to when it has nothing to do.
REST_PREFILL = "prefill"  # -> PP, the default; see the module docstring
REST_DECODE = "decode"  # -> TP
REST_STATES = (REST_PREFILL, REST_DECODE)
_PHASE_OF_REST = {REST_PREFILL: PHASE_PP, REST_DECODE: PHASE_TP}

# -- measured constants, #631 mainrig -----------------------------------------
# Round-trip flip cost, client-observed. Per-rank one-way cutovers measured
# 997 / 1246 / 1720 ms (PROD_BRINGUP_BENCH, commit 9a929352c9); the round
# trip is bounded by the slowest rank in each direction.
DEFAULT_FLIP_COST_S = 3.2
# PP-phase prefill at the 8192-token rung, tok/s (PROD_BRINGUP_BENCH
# acceptance ladder: 4236.8 / 7245.5 / 6842.6 at 2048 / 8192 / 32768).
DEFAULT_PP_PREFILL_TOK_S = 7245.5
# TP-phase prefill at the same rung, tok/s. MEASURED, not assumed: see
# scripts/route_a_631_prefill_ladder_quiet.py and the policy section of
# PROD_BRINGUP_BENCH. Overridable so a different rig can re-derive N from
# its own ladder without editing code.
ENV_TP_TOK_S = "SGLANG_PHASE_POLICY_TP_TOK_S"
# TP-phase prefill at the 8192-token rung, tok/s. MEASURED on this rig
# (2026-08-08, commit 2bcc6b7d25, quiet-gated ladder with zero contended
# draws): 2134.1 / 1681.0 / 1484.1 at 2048 / 8192 / 32768, spreads 0.25 /
# 0.14 / 0.03 %. Against the PP ladder's 4236.8 / 7245.5 / 6842.6 that is
# a 2.0x / 4.3x / 4.6x prefill penalty for serving a long prompt from the
# decode layout -- a 32768-token prefill costs 22.08 s in TP against
# 4.79 s in PP. This is the number that makes the policy worth having,
# and it is why the resting layout is PP.
#
# Overridable so another rig re-derives N from its own ladder rather than
# inheriting this one's (see scripts/route_a_631_prefill_ladder_quiet.py).
DEFAULT_TP_PREFILL_TOK_S = float(os.environ.get(ENV_TP_TOK_S, "1681.0") or 1681.0)

ENV_ENABLE = "SGLANG_PHASE_POLICY"
ENV_FLIP_TOKENS = "SGLANG_PHASE_POLICY_FLIP_TOKENS"
ENV_MIN_DWELL = "SGLANG_PHASE_POLICY_MIN_DWELL_S"
ENV_IDLE_DWELL = "SGLANG_PHASE_POLICY_IDLE_DWELL_S"
ENV_REST_STATE = "HTSGLANG_PHASE_IDLE_STATE"
ENV_PP_WINDOW = "SGLANG_PHASE_POLICY_PP_WINDOW_S"
ENV_TP_FLOOR = "SGLANG_PHASE_POLICY_TP_DECODE_FLOOR_S"

# -- THE FAIRNESS BOUND (starvation fix, metal-observed 2026-08-09) ----------
# Both thresholds above are LOAD-TRIGGERED, and that is not enough. Under
# CONTINUOUS mixed arrivals -- which is exactly what a real agent backend
# produces -- neither of them ever fires in the PP direction:
#
#   21:15:25Z .. 21:16:15Z  PHASE-POLICY holding in pp: prefilling in pp
#                           (302757 tok pending, running bs 2)   x 6 samples
#   same window: 87 Decode batches, 6 Prefill batches, cuda graph: False,
#   gen throughput 35.4 tok/s
#
# The PP->TP rule is ``pending <= N``. With arrivals sustaining a backlog far
# above N=7004 that predicate is never true, so the server pins itself in the
# PREFILL layout and serves DECODE there -- no speculation, no decode CUDA
# graphs, 35 tok/s. The docstring above anticipated the ``== 0`` form of this
# trap and replaced it with ``<= N``; that fixed the batching case and left
# the sustained-backlog case, which is the same trap one threshold further
# out. Absolute prefill priority can never reach the decode layout under a
# load that always has prefill pending.
#
# The bound is therefore a TIME-SLICE, and it must be symmetric or it just
# moves the starvation to the other side:
#
#   pp_window_s      max continuous seconds in PP while decode work waits.
#                    On expiry PP->TP arms REGARDLESS of pending prefill;
#                    the residents carry across (phase_flip_draft_bootstrap)
#                    and new prefill queues behind the existing admission
#                    hold until the next PP window.
#   tp_decode_floor_s minimum seconds in TP before a pending-prefill-driven
#                    TP->PP may arm. Without it the TP phase would last one
#                    min_dwell (3 s here) before the ever-present backlog
#                    dragged it straight back, and decode would starve in
#                    the mirror image of the defect.
#
# Both apply ONLY while the other side actually has work (``running_bs > 0``
# for the PP window; the TP floor likewise). A phase with nothing to protect
# is left free to flip on the load rule immediately, so an arriving long
# prompt at a decode-idle server still gets PP-class prefill in its TTFT.
#
# Cost accounting, and it is deliberate: one 15 s + 10 s cycle carries two
# round trips (~3.2 s measured), i.e. ~11 % of wall clock in cutovers. That
# is the price of both sides making progress; it is a knob, not a constant.
DEFAULT_PP_WINDOW_S = 15.0
DEFAULT_TP_DECODE_FLOOR_S = 10.0

# Defaults for the two timers. The minimum dwell is deliberately larger than
# one round trip: a server that flips, and is allowed to flip straight back,
# spends more wall clock moving KV than serving.
DEFAULT_MIN_DWELL_S = 10.0
# The idle return is the OTHER half of resting in PP, and it has to be
# fast to be worth anything. The serving cycle is: prefill in PP -> flip
# to TP so the draft/verify of speculation runs where the decode CUDA
# graphs are -> decode -> and then back to PP so the NEXT request's
# prefill is already in the fast layout. If this dwell is long, the last
# step does not happen in time and the next request prefills in TP at
# 1681 tok/s instead of 7245 -- which is the whole defect the policy
# exists to remove.
#
# 3 s is just under two flip round trips: a gap shorter than the flip
# itself cannot trigger a return (there would be no time to profit from
# it), while any real inter-request gap does. Nothing waits on this flip,
# so its cost is not in anyone's latency path.
DEFAULT_IDLE_DWELL_S = 3.0


class PhasePolicyError(ValueError):
    """Raised for a configuration that cannot be honoured."""


def break_even_tokens(
    flip_cost_s: float,
    tp_tok_s: float,
    pp_tok_s: float,
) -> int:
    """N = C / (1/X - 1/P); see the module docstring.

    Refuses rather than guesses when the premise does not hold: a
    non-positive throughput is not a measurement, and ``tp_tok_s >=
    pp_tok_s`` means the PP layout is not the faster prefill layout, in
    which case there is no token count at which flipping to it repays the
    cost and the caller must be told so instead of handed a nonsense number.
    """
    if flip_cost_s <= 0:
        raise PhasePolicyError(f"flip cost must be positive, got {flip_cost_s!r}")
    if tp_tok_s <= 0 or pp_tok_s <= 0:
        raise PhasePolicyError(
            f"prefill throughputs must be positive and measured, got "
            f"tp={tp_tok_s!r} pp={pp_tok_s!r}"
        )
    if tp_tok_s >= pp_tok_s:
        raise PhasePolicyError(
            f"no break-even exists: TP prefill {tp_tok_s:g} tok/s is not "
            f"slower than PP prefill {pp_tok_s:g} tok/s, so flipping to the "
            f"PP layout never repays the {flip_cost_s:g}s round trip. Route "
            f"A's premise is that PP prefills faster; re-measure before "
            f"setting a threshold."
        )
    return int(round(flip_cost_s / (1.0 / tp_tok_s - 1.0 / pp_tok_s)))


@dataclass(frozen=True)
class PhasePolicyConfig:
    """Static configuration. Built once at boot, never mutated."""

    enabled: bool = False
    flip_tokens: int = 0
    min_dwell_s: float = DEFAULT_MIN_DWELL_S
    idle_dwell_s: float = DEFAULT_IDLE_DWELL_S
    rest_state: str = REST_PREFILL
    #: Fairness bound; 0 disables it and restores the pure load-triggered
    #: behaviour that starves under a sustained backlog. See the block
    #: comment at DEFAULT_PP_WINDOW_S.
    pp_window_s: float = DEFAULT_PP_WINDOW_S
    tp_decode_floor_s: float = DEFAULT_TP_DECODE_FLOOR_S

    def __post_init__(self) -> None:
        if self.rest_state not in REST_STATES:
            raise PhasePolicyError(
                f"{ENV_REST_STATE}={self.rest_state!r} is not a known resting "
                f"state; use one of {', '.join(REST_STATES)}"
            )
        if self.enabled and self.flip_tokens <= 0:
            raise PhasePolicyError(
                "the phase policy needs a positive flip threshold; set "
                f"{ENV_FLIP_TOKENS} or supply a measured TP prefill "
                f"throughput via {ENV_TP_TOK_S}"
            )

    @property
    def rest_phase(self) -> str:
        return _PHASE_OF_REST[self.rest_state]


@dataclass
class PhasePolicyState:
    """The only mutable state. Timestamps are on the caller's clock."""

    last_flip_at: float = 0.0
    idle_since: Optional[float] = None
    flips_armed: int = 0
    last_reason: str = ""
    #: When the CURRENT phase was entered, on the caller's clock. Distinct
    #: from ``last_flip_at`` on purpose: that one is an ARM stamp and is 0
    #: until the first flip, while the fairness bound has to measure the
    #: very first PP occupancy too. Maintained by ``observe_idle`` from the
    #: OBSERVED phase, so a manual POST /phase_flip -- which the policy
    #: never armed -- also restarts the clock.
    #: None means NOT YET OBSERVED, and both window rules treat that as
    #: "inapplicable" rather than as "infinitely long ago". Without that
    #: reading, any caller reaching ``decide`` without having called
    #: ``observe_idle`` first -- every pre-window unit test, and any future
    #: caller that forgets -- would see a window that expired at t=0 and
    #: flip immediately. Degrading to the pre-window behaviour is the safe
    #: direction: an unobserved phase simply keeps the load-triggered rules.
    #: A sentinel rather than 0.0 because 0.0 is a legitimate clock value
    #: and the two readings must not be confusable.
    phase_since: Optional[float] = None
    last_phase: Optional[str] = None


@dataclass(frozen=True)
class PhasePolicyInputs:
    """A replicated snapshot of the load. Every field MUST be identical on
    every rank -- see the module docstring on replication."""

    phase: str
    pending_prefill_tokens: int
    running_bs: int
    now: float


@dataclass(frozen=True)
class PhasePolicyDecision:
    direction: Optional[str]
    reason: str

    @property
    def wants_flip(self) -> bool:
        return self.direction is not None


def _no(reason: str) -> PhasePolicyDecision:
    return PhasePolicyDecision(direction=None, reason=reason)


def decide(
    cfg: PhasePolicyConfig,
    state: PhasePolicyState,
    inp: PhasePolicyInputs,
) -> PhasePolicyDecision:
    """The whole policy. Pure: mutates nothing, reads nothing global.

    ``state`` is read but not written -- the caller commits state changes via
    ``note_flip_armed`` / ``observe_idle`` so that a decision which is
    refused downstream (guards, a losing consensus) does not leave the
    policy believing it flipped.
    """
    if not cfg.enabled:
        return _no("policy disabled")

    idle = inp.running_bs == 0 and inp.pending_prefill_tokens == 0

    # The minimum dwell is checked FIRST and applies to every flip, so no
    # branch below can bypass the thrash bound. It is the only guarantee
    # that survives an adversarial arrival pattern.
    since_flip = inp.now - state.last_flip_at
    if state.last_flip_at > 0 and since_flip < cfg.min_dwell_s:
        return _no(
            f"min dwell: {since_flip:.1f}s since last flip < "
            f"{cfg.min_dwell_s:g}s"
        )

    if inp.phase == PHASE_TP:
        # Enough queued prefill to repay the round trip -> go to the prefill
        # layout. Below the threshold the prefill runs in TP as-is, which is
        # the correct answer for short prompts by construction of N.
        if inp.pending_prefill_tokens > cfg.flip_tokens:
            # THE DECODE FLOOR. Under purity every token of prefill has to
            # wait for a PP window, so the backlog is essentially always
            # above N and this rule would otherwise fire the instant
            # min_dwell expires -- giving decode a 3 s window per cycle and
            # starving it in the mirror image of the defect the PP window
            # fixes. Only applies while decode work actually exists: with
            # nothing decoding there is nothing to protect, and an arriving
            # long prompt should reach the PP layout inside its TTFT.
            in_phase = (
                None
                if state.phase_since is None
                else inp.now - state.phase_since
            )
            if (
                cfg.tp_decode_floor_s > 0
                and in_phase is not None
                and inp.running_bs > 0
                and in_phase < cfg.tp_decode_floor_s
            ):
                return _no(
                    f"decode floor: {in_phase:.1f}s in tp < "
                    f"{cfg.tp_decode_floor_s:g}s, {inp.running_bs} req "
                    f"decoding ({inp.pending_prefill_tokens} tok prefill "
                    f"waiting for the next pp window)"
                )
            return PhasePolicyDecision(
                TP_TO_PP,
                f"pending prefill {inp.pending_prefill_tokens} tok > "
                f"N={cfg.flip_tokens}",
            )
        # Nothing to do, and the resting layout is PP -> return to rest.
        if idle and cfg.rest_phase == PHASE_PP:
            if state.idle_since is None:
                return _no("idle, waiting for the idle dwell to start")
            idle_for = inp.now - state.idle_since
            if idle_for >= cfg.idle_dwell_s:
                return PhasePolicyDecision(
                    TP_TO_PP,
                    f"idle {idle_for:.1f}s >= {cfg.idle_dwell_s:g}s, "
                    f"returning to the {cfg.rest_state} resting layout",
                )
            return _no(
                f"idle {idle_for:.1f}s < {cfg.idle_dwell_s:g}s idle dwell"
            )
        if inp.pending_prefill_tokens:
            return _no(
                f"pending prefill {inp.pending_prefill_tokens} tok <= "
                f"N={cfg.flip_tokens}, running it in tp"
            )
        return _no("decoding in tp")

    if inp.phase == PHASE_PP:
        # Prefill is done (or what is left is not worth staying for) and
        # there is decode work -> go to the decode layout, where the
        # draft/verify of speculation and the decode CUDA graphs live.
        #
        # The test is ``<= N`` rather than ``== 0`` because of BATCHING.
        # Under continuous arrivals the queue may never reach exactly
        # zero, and an ``== 0`` rule would pin the server in PP and decode
        # there -- with no speculation and no decode graphs, which is the
        # mirror of the defect this policy exists to remove. ``<= N`` is
        # the same break-even that governs the other direction, read the
        # other way: prefill worth less than a flip is prefill worth
        # running in whatever layout we are about to be in anyway. That
        # makes the two rules one hysteresis band around N instead of two
        # unrelated thresholds, so no arrival pattern can satisfy both.
        if inp.pending_prefill_tokens <= cfg.flip_tokens and inp.running_bs > 0:
            return PhasePolicyDecision(
                PP_TO_TP,
                f"prefill down to {inp.pending_prefill_tokens} tok "
                f"(<= N={cfg.flip_tokens}), {inp.running_bs} req decoding",
            )
        # THE PP WINDOW. The rule above needs prefill to DRAIN below N; a
        # sustained backlog never does, and under strict purity the PP
        # phase may not decode at all, so without this bound the instance
        # parks in PP with decode work it is forbidden to run. On expiry
        # PP->TP arms regardless of the backlog: the residents carry over
        # and decode batched in the TP window, and the prefill that did not
        # fit waits for the next PP window. This is also the deadlock
        # breaker for a PP phase that cannot ADMIT its pending prefill (no
        # free state slot) -- see phase_purity's deadlock section.
        if (
            cfg.pp_window_s > 0
            and state.phase_since is not None
            and inp.running_bs > 0
            and (inp.now - state.phase_since) >= cfg.pp_window_s
        ):
            return PhasePolicyDecision(
                PP_TO_TP,
                f"pp window {inp.now - state.phase_since:.1f}s >= "
                f"{cfg.pp_window_s:g}s with {inp.running_bs} req waiting to "
                f"decode ({inp.pending_prefill_tokens} tok prefill deferred "
                f"to the next pp window)",
            )
        if idle and cfg.rest_phase == PHASE_TP:
            if state.idle_since is None:
                return _no("idle, waiting for the idle dwell to start")
            idle_for = inp.now - state.idle_since
            if idle_for >= cfg.idle_dwell_s:
                return PhasePolicyDecision(
                    PP_TO_TP,
                    f"idle {idle_for:.1f}s >= {cfg.idle_dwell_s:g}s, "
                    f"returning to the {cfg.rest_state} resting layout",
                )
            return _no(
                f"idle {idle_for:.1f}s < {cfg.idle_dwell_s:g}s idle dwell"
            )
        if idle:
            return _no("idle at rest")
        return _no(f"prefilling in pp ({inp.pending_prefill_tokens} tok pending)")

    return _no(f"unknown phase {inp.phase!r}")


def observe_idle(state: PhasePolicyState, inp: PhasePolicyInputs) -> None:
    """Maintain the idle clock. Called every round, before ``decide``.

    Split out from ``decide`` so the decision stays pure and so the clock is
    driven by observation rather than by the policy's own verdict.
    """
    idle = inp.running_bs == 0 and inp.pending_prefill_tokens == 0
    if idle:
        if state.idle_since is None:
            state.idle_since = inp.now
    else:
        state.idle_since = None
    # Phase-entry clock for the fairness bound. Driven by the OBSERVED
    # phase rather than by the policy's own verdict, so it is correct for
    # the first phase (no flip has happened yet, last_flip_at == 0) and for
    # a phase this policy did not choose (manual /phase_flip).
    if state.last_phase != inp.phase:
        state.last_phase = inp.phase
        state.phase_since = inp.now


def note_flip_armed(
    state: PhasePolicyState, decision: PhasePolicyDecision, now: float
) -> None:
    """Commit the dwell clock, once a decision was actually armed."""
    state.last_flip_at = now
    state.idle_since = None
    state.flips_armed += 1
    state.last_reason = decision.reason


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise PhasePolicyError(f"{name}={raw!r} is not a number") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise PhasePolicyError(f"{name}={raw!r} is not an integer") from exc


def config_from_env(enabled: bool) -> PhasePolicyConfig:
    """Build the boot configuration.

    ``enabled`` comes from the server arg; the tuning knobs come from the
    environment so a running deployment can be re-tuned without a code
    change, and so the acceptance run can exercise a short dwell without
    shipping a short dwell as the default.
    """
    rest_state = os.environ.get(ENV_REST_STATE) or REST_PREFILL
    min_dwell = _env_float(ENV_MIN_DWELL, DEFAULT_MIN_DWELL_S)
    idle_dwell = _env_float(ENV_IDLE_DWELL, DEFAULT_IDLE_DWELL_S)

    explicit = _env_int(ENV_FLIP_TOKENS, 0)
    if explicit > 0:
        flip_tokens = explicit
        source = f"{ENV_FLIP_TOKENS}={explicit}"
    elif DEFAULT_TP_PREFILL_TOK_S > 0:
        flip_tokens = break_even_tokens(
            DEFAULT_FLIP_COST_S,
            DEFAULT_TP_PREFILL_TOK_S,
            DEFAULT_PP_PREFILL_TOK_S,
        )
        source = (
            f"break-even {DEFAULT_FLIP_COST_S:g}s / (1/"
            f"{DEFAULT_TP_PREFILL_TOK_S:g} - 1/{DEFAULT_PP_PREFILL_TOK_S:g})"
        )
    elif enabled:
        raise PhasePolicyError(
            "the phase policy is enabled but has no threshold: set "
            f"{ENV_FLIP_TOKENS} explicitly, or supply the measured TP-phase "
            f"prefill throughput in {ENV_TP_TOK_S} so the "
            f"break-even N can be derived. A threshold is a measurement, "
            f"not a default."
        )
    else:
        flip_tokens = 0
        source = "unset (policy off)"

    cfg = PhasePolicyConfig(
        enabled=enabled,
        flip_tokens=flip_tokens,
        min_dwell_s=min_dwell,
        idle_dwell_s=idle_dwell,
        rest_state=rest_state,
        pp_window_s=_env_float(ENV_PP_WINDOW, DEFAULT_PP_WINDOW_S),
        tp_decode_floor_s=_env_float(ENV_TP_FLOOR, DEFAULT_TP_DECODE_FLOOR_S),
    )
    if enabled:
        logger.warning(
            "%s armed: N=%d tok (%s), min dwell %gs, idle dwell %gs, "
            "pp window %gs, tp decode floor %gs, resting layout %s (%s)",
            LOG_PREFIX,
            cfg.flip_tokens,
            source,
            cfg.min_dwell_s,
            cfg.idle_dwell_s,
            cfg.pp_window_s,
            cfg.tp_decode_floor_s,
            cfg.rest_phase,
            cfg.rest_state,
        )
    return cfg


__all__ = [
    "PHASE_PP",
    "PHASE_TP",
    "PP_TO_TP",
    "TP_TO_PP",
    "REST_PREFILL",
    "REST_DECODE",
    "REST_STATES",
    "PhasePolicyConfig",
    "PhasePolicyDecision",
    "PhasePolicyError",
    "PhasePolicyInputs",
    "PhasePolicyState",
    "break_even_tokens",
    "config_from_env",
    "decide",
    "note_flip_armed",
    "observe_idle",
]
