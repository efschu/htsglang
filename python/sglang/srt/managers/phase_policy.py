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

THE RESTING STATE IS THE DECODE LAYOUT (TP) -- CHANGED 2026-08-14 (user)
------------------------------------------------------------------------
EVERY request decodes; only a large-prefill request benefits from PP. Since
decode in the PP layout is forbidden, resting in PP means every request pays
at least one ``pp_to_tp`` cutover to decode, and the server then idles back
to PP and pays it again on the next one. That is not a corner case, it is a
thrash cycle, and it was measured on this rig 2026-08-14: 882 flips in a
single boot, arming at 184 pending prefill tokens, ~4.8 s of seam per
request, TTFT ~2.9 s on a 65-CHARACTER prompt.

Resting in TP costs the opposite case: a long prompt arriving at a QUIET
server now pays ~2.7 s of ``tp_to_pp`` inside its TTFT that it previously got
free. That cost is bounded and, above the threshold N, repaid by the prefill
saving by construction -- N is precisely the point where it is.

#777, AND READ THIS BEFORE QUOTING N AT A SINGLE PROMPT. This paragraph used
to end "Below N the prompt never wanted PP anyway and now prefills in TP
without any flip at all", which is a claim about ONE prompt that the code does
not make. N is compared against ``inp.pending_prefill_tokens`` -- the AGGREGATE
of admitted-but-not-yet-computed prompt tokens across the waiting queue, the
chunked remainder and in-flight arrivals (see ``Scheduler._pending_prefill_
tokens``). There is no per-request comparison against N anywhere in this
module. Two consequences the old wording denied:

  * a 1.8k-token prompt reaches PP prefill whenever it lands in a backlog
    whose SUM crosses N, even though 1.8k is far below it;
  * once the layout is resident in PP, every prompt admitted into it prefills
    there at any size. The only ways back to TP are the aggregate exit rules,
    the residency and stall caps, and the idle dwell -- none of which look at
    the size of a newly arrived prompt.

N is a phase-wide backlog break-even. It is not, and never was, a small-prompt
shield. Whether it SHOULD also gate per request is a policy question and
belongs to the planner, not to this docstring.

The previous argument for PP-resting ("a long prompt gets PP-class prefill
with ZERO flip cost") was written when prefill could not run in TP AT ALL, so
every request had to reach PP regardless of size. With that prohibition
withdrawn the argument only covers prompts above N, which are the ones that
can afford the flip. ``rest_state`` remains configurable
(``HTSGLANG_PHASE_IDLE_STATE``) for a deployment whose traffic is dominated
by large prompts arriving at an idle server.

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
import math
import os
from dataclasses import dataclass, field
from typing import Dict, Optional

# #901: the single knob-resolution authority. Imported as a module rather than
# by name so every use below reads as "the authority said this", and so a new
# helper cannot be added to the import list and then quietly shadowed by a
# local of the same name.
from sglang.srt import knob_resolution as _knob

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-POLICY"

# Layout names, matching phase_flip_runtime.PHASE_PP / PHASE_TP. Imported
# lazily by callers; duplicated here as plain strings so this module stays
# importable (and unit-testable) without the flip runtime.
PHASE_PP = "pp"
PHASE_TP = "tp"
BOTH_BLOCKED = "BOTH BLOCKED"

#: #677: the layout HOLD. Bounded, because an unbounded hold is a starvation
#: bug wearing a fix's clothes.
#:
#: 8 rounds is derived, not round: a cutover costs ~2.6 s of seam (#690, seam
#: total 2.56-2.59 s measured), and the scheduler's rounds under prefill run at
#: roughly the chunk cadence, so 8 rounds bounds a hold at well under one seam's
#: worth of held layout. Holding longer than it costs to leave is never worth it.
LAYOUT_HOLD_MAX_ROUNDS: int = 8

HOLD_FOR_UNSERVED = "HOLD: work pending in this layout"


def next_hold_rounds(
    prev_rounds: int, prev_phase: str, phase: str, pending: int
) -> int:
    """The hold counter's lifecycle, as a function so it can be pinned (#677).

    ``layout_hold_verdict`` takes ``hold_rounds_so_far`` from its caller, and a
    caller-maintained counter is exactly the kind of state that goes stale
    silently. A counter carried across a phase change would make the FIRST hold
    of the next episode start half-exhausted -- the layout would release early,
    and the C2 fix would evaporate for the arrival that needed it most.

    RESET ON EITHER BOUNDARY: a phase change (a new episode owns a fresh bound)
    or the pending work reaching zero (the episode this bound was counting is
    over). Otherwise advance by one.
    """
    if phase != prev_phase:
        return 0
    if int(pending) <= 0:
        return 0
    return max(0, int(prev_rounds)) + 1


def layout_hold_verdict(
    phase: str,
    prefill_pending_tokens: int,
    decode_waiting: int,
    hold_rounds_so_far: int = 0,
    max_hold_rounds: int = LAYOUT_HOLD_MAX_ROUNDS,
    seam_funded: bool = True,
    mid_flip: bool = False,
):
    """``(allow_flip, reason)`` -- demand decides the layout, not the timer.

    MEASURED PROBLEM (#713 quantisation table, 2026-08-17 06:19). Demand-PULL
    already worked: C1 arrived 06:19:06.56 and was served by the tp_to_pp at
    :09, one seam plus overhead. What failed was the step after -- the box
    flipped BACK at :12, so C2, which arrived 06:19:09.71 just as PP began, was
    not served until :15. The layout left on a timer while the prefill that
    pulled it was still unserved, and TTFT quantised to whole cycles
    (0.1 / 3.1 / 5.9 s, nothing between).

    So the same arriving-tokens signal that PULLS a cutover must also HOLD it
    until the work it pulled for is served. Demand overrides the timer in BOTH
    directions; that symmetry is the rule, not two rules.

    THIS FUNCTION CARRIES ONE OF THE TWO DIRECTIONS -- the HOLD in pp. The pull
    out of tp is a rule, not a veto, and it lives where the rules live: the
    ``pending prefill > threshold`` arm and its DECODE FLOOR guard (see the
    "THE DECODE FLOOR" comment in ``_decide_from_load``). #820 removed the
    ``phase == "tp"`` branch that used to sit here; the argument is in that
    comment, at the guard that actually does the job.

    SAFETY, in precedence order:
      * never mid-flip -- a cutover in progress owns the layout;
      * never against an unfunded seam -- a pull that cannot pay is an abandon;
      * BOUNDED BOTH WAYS -- see below.

    THE BOTH-SIDES TIE. When prefill is unserved in PP *and* decode is waiting
    for TP, the timer must not break it and neither may an unbounded hold: that
    is one starvation direction traded for the other. The tie goes to the #677
    economic comparison, backlog-weighted, and the hold is bounded by
    ``max_hold_rounds`` so a decode queue can never be held out forever. The
    mirror bound is that a pull may not preempt an unserved prefill while this
    hold is live, which is exactly what the hold branch expresses.
    """
    pend = max(0, int(prefill_pending_tokens))
    dec = max(0, int(decode_waiting))
    held = max(0, int(hold_rounds_so_far))

    if mid_flip:
        return False, "no decision: a cutover is in progress and owns the layout"
    if not seam_funded:
        return False, "no flip: the seam is unfunded -- arming it would abandon"

    if phase == "pp":
        if pend <= 0:
            return True, "no prefill pending in pp: the timer may have the layout"
        if held >= max_hold_rounds:
            return True, (
                f"hold EXHAUSTED after {held} rounds with {pend} prefill tokens "
                f"still pending and {dec} decode waiting: releasing so the "
                f"decode side cannot be starved by an unbounded hold"
            )
        if dec > 0:
            return False, (
                f"{HOLD_FOR_UNSERVED}: BOTH SIDES have work ({pend} prefill "
                f"tokens unserved here, {dec} decode waiting) -- held for the "
                f"economic comparison, round {held + 1} of {max_hold_rounds}, "
                f"bounded so neither side starves"
            )
        return False, (
            f"{HOLD_FOR_UNSERVED}: {pend} prefill tokens pulled this layout and "
            f"are still unserved; the timer does not get to take it away "
            f"(round {held + 1} of {max_hold_rounds})"
        )

    # NO VERDICT FOR ANY OTHER PHASE, AND IT ALLOWS -- #820.
    #
    # This used to return False, and False here means HOLD: the sole consumer
    # is a veto (`decide`, "if not allow: return a wait"), so an unrecognised
    # input fell toward SWALLOWING the arm. That is the exact failure mode
    # #817 removed one level up, where an unrecognised arm used to be swallowed
    # by a substring denylist. Removing the "tp" branch turned "tp" from a
    # recognised phase into an unrecognised one, so leaving the old default
    # would have GROWN that surface instead of shrinking it. An input this
    # function does not understand cannot be grounds for holding a layout.
    return True, (
        f"no decision for phase {phase!r}: this lever only reads the pp-side "
        f"hold, so the rules' own arm stands unmodified"
    )


PP_TO_TP = "pp_to_tp"
#: Reason prefix of the #688 deadlock escape. A CONSTANT, because #689's
#: formation gate must recognise that decision and refuse to hold it, and a
#: literal repeated in two modules is a rename away from silently
#: reintroducing the idle window.
IDLE_LOCKED = "IDLE-LOCKED"
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
#: The PP-phase counterpart. It did not exist until 2026-08-15, which
#: meant the PP rate was the ONLY input to the break-even that could not
#: be re-measured per checkpoint -- so a rig whose PP prefill differed
#: from this module's 7245.5 had no way to say so, and silently solved N
#: against another rig's number.
ENV_PP_TOK_S = "SGLANG_PHASE_POLICY_PP_TOK_S"
#: THE DECODE-STARVATION CONSTRAINT that replaces the PP stopwatch.
#:
#: `pp_window_s` was a hand-set duty cycle, and it ejected the PP phase on a
#: clock regardless of what was left to drain. Live at 12:25 on 2026-08-15:
#:
#:   arming pp_to_tp: pp window 15.0s >= 15s with 3 req waiting to decode
#:                    (23313 tok prefill deferred to the next pp window)
#:
#: 23k of prefill still standing, ~12 s of seam burned across two round trips
#: to defer work that one longer window would have drained, then straight back
#: into PP as pending regrew. That is a see-saw on a timer, and a hand-set
#: constant deciding it is the same provenance defect the ladder fix removed
#: from arming.
#:
#: What actually bounds a PP residency is not a duty cycle, it is how long a
#: carried decode may be stalled -- a LATENCY constraint. Declare that, and the
#: residency cap is SOLVED from it and the measured seam:
#:
#:     pp_residency_cap = decode_stall_slo_s - 2 * flip_cost_s
#:
#: The 2x is not a fudge: a decode carried into PP resumes only once the
#: instance is back in TP, so it pays the seam in BOTH directions on top of the
#: residency itself. 0 means no cap is declared and the phase is governed
#: purely by drain, which is the existing default (`pp_window_s` also defaults
#: to 0).
ENV_DECODE_STALL_SLO = "SGLANG_PHASE_POLICY_DECODE_STALL_SLO_S"
#: Tokens at or below which the PP phase is considered DRAINED.
#:
#: This is NOT the entry break-even N, and conflating the two was a real
#: defect: the instance left PP at ~10k pending and prefilled the remainder in
#: TP at a third of the rate. N prices ENTRY -- two seams against a mountain
#: that has not been climbed yet. Once you are already IN PP the economics are
#: asymmetric:
#:
#:   finish here : R / r_pp  + the return seam, which is due anyway
#:   leave now   : the return seam NOW + R / r_tp
#:
#: The seam is common to both, so it cancels, and what remains is R/r_pp
#: against R/r_tp. Since r_pp > r_tp by construction (it is why the PP layout
#: exists), leaving early LOSES for every R > 0. There is no crossover to
#: find: the correct exit is "the prefill is done", and the only thing allowed
#: to cut it short is the decode-starvation SLO.
#:
#: "Done" means below one chunk, not exactly zero, because a chunked prefill
#: leaves a partial chunk in flight and an ==0 test can miss it forever. The
#: scheduler supplies its real chunked_prefill_size at boot.
ENV_PP_EXIT_TOKENS = "SGLANG_PHASE_POLICY_PP_EXIT_TOKENS"

# -- SEAM STAGING IS EVALUATED, NOT FITTED (#665-F1 item 7) -----------------
# An earlier version of this file carried an empirical want = base + k*chunk^2
# fit over three measured anchors. It was falsified on a held-out point (chunk
# 1024: predicted 1240 MiB, actual 1539-2398) and the anchors turned out to be
# one-per-RANK, so the curve was fitting per-rank arena_tail offsets rather
# than chunk width at all.
#
# There is nothing to fit. `PhaseFlipRuntime._staging_bytes` computes the want
# exactly, from the transfer plan and the static layout:
#
#     want = arena_tail + max(wave_peak, draft_restore)
#     wave_peak = incoming + max(outgoing, local) + one_layer_window
#                 + backing_slack
#
# and the ONLY live-set input to the plan is the slot count. So the boot-time
# projection lives with the runtime (see project_staging_bytes there), which
# owns those terms and knows its own rank -- not here, where a curve could
# only ever approximate them and would silently rot the next time the formula
# changed.
DEFAULT_PP_EXIT_TOKENS = 0
DEFAULT_DECODE_STALL_SLO_S = 0.0
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
ENV_FLIP_COST_S = "SGLANG_PHASE_POLICY_FLIP_COST_S"
ENV_DECODE_STRAND_WEIGHT = "SGLANG_PHASE_POLICY_DECODE_STRAND_WEIGHT"
ENV_MIN_DWELL = "SGLANG_PHASE_POLICY_MIN_DWELL_S"
ENV_IDLE_DWELL = "SGLANG_PHASE_POLICY_IDLE_DWELL_S"
ENV_REST_STATE = "HTSGLANG_PHASE_IDLE_STATE"
ENV_PP_WINDOW = "SGLANG_PHASE_POLICY_PP_WINDOW_S"
ENV_TP_FLOOR = "SGLANG_PHASE_POLICY_TP_DECODE_FLOOR_S"
ENV_REFUSAL_BACKOFF_CAP = "SGLANG_PHASE_POLICY_REFUSAL_BACKOFF_CAP_S"
ENV_REFUSAL_DEGRADE_AFTER = "SGLANG_PHASE_POLICY_REFUSAL_DEGRADE_AFTER"
ENV_DECODE_CONTENTION = "SGLANG_PHASE_POLICY_DECODE_CONTENTION"

# -- THE COUNTERFACTUAL, MEASURED (#665-F1, 2026-08-15) ----------------------
# `effective_flip_threshold` charges every decoding request a full PP window
# for the stranding a cutover causes, and charges the alternative -- leaving
# the prefill in TP -- nothing at all. That one-sidedness made the gate
# unreachable on the dev instance: the ladder ran 7004 / 39835 / 72666 /
# 105498 / 138329 tokens for 0..4 decoding requests, so at
# --max-running-requests 4 no prompt this model can hold could ever flip, and
# a live 72,257-token backlog at running_bs 2 was refused by 409 tokens.
#
# The premise behind the zero was never measured. It is false here. Measured
# against the live instance (2 decode streams + one 72k prompt, client-side
# token timing, /spinning/evidence-665-f1/measure_decode_contention.py):
#
#   decode throughput, undisturbed            53.2 tok/s
#   decode throughput, during a TP prefill      0.0 tok/s
#   A-vs-A noise floor                          0.4 %
#   TP prefill wall clock for 72k tokens       60.6 s
#
# Decode does not degrade beside a co-resident TP prefill -- it STOPS, for the
# whole minute the prefill takes. So the decodes are stranded in BOTH branches
# and the only question is for how long: ~60 s if we stay, ~16 s (two seams
# plus a 9.9 s PP prefill) if we flip. The surcharge was protecting the
# decodes by stalling them roughly four times longer.
#
# `decode_contention` is that measurement: the fraction of decode throughput
# lost while a prefill is co-resident in TP. 1.0 = decode stops dead, which is
# what this scheduler does unconditionally: batch selection reads `if
# new_batch is not None: # Run prefill first if possible`, so an iteration
# with any prefill chunk pending runs THAT batch and never reaches the decode
# branch. That is absolute prefill priority per iteration, not a property of
# --disable-overlap-schedule or any other flag. 0.0 means "not
# measured here" and keeps the old surcharge byte-identically, the same
# measurement gate `flip_cost_s` already uses.
#
# THE TWO ALTERNATIVES, AND WHY THEY LOST.
#
# CAP THE SURCHARGE at some multiple of `flip_tokens`. It would work, and the
# multiple would be fiat: nothing in the evidence picks one, and it would need
# re-picking whenever the seam or the window moved, because the model under it
# would still be one-sided. Pricing the counterfactual turns out to DERIVE the
# cap instead -- 2 x N at sigma = 1 -- so this option is not so much rejected
# as explained.
#
# ADMIT-THEN-FLIP-AT-DRAIN: admit the long prefill, queue the flip, take it
# once the decodes drain below a bar. Rejected, and the measurement is what
# rejects it. At sigma = 1 the decodes CANNOT drain while the admitted prefill
# runs in TP -- the prefill is precisely what is stopping them -- so the
# trigger waits on a condition its own admission prevents. It is also the trap
# the block comment at DEFAULT_PP_WINDOW_S documents twice already: a
# load-triggered predicate that a sustained backlog never satisfies, one
# threshold further out. Making it terminate needs a deadline, i.e. a forced
# flip, and forced flips are what raised the flip rate that killed three boots
# with an allocation failure at the cutover seam.
#
# WHAT THIS COSTS, STATED PLAINLY. TP->PP arms at the effective threshold while
# PP->TP arms at plain `flip_tokens`, so the two rules form a hysteresis band.
# Making the gate reachable NARROWS that band -- at running_bs 2 it goes from
# [7004, 72666] to [7004, 11673] -- and a narrower band is easier to cross
# twice. It does not raise the rate CEILING, which `min_dwell_s` and
# `tp_decode_floor_s` already own: a cycle cannot be shorter than
# tp_decode_floor + seam + min_dwell + seam, ~16 s with the booted 10 / 3.2 / 3,
# i.e. at most ~7 flips/min. But before this change the load-driven direction
# essentially never armed, so the REALISED rate was ~0 and that ceiling was
# theoretical; now it is reachable, and 12 flips in four minutes is the rate
# that once killed three boots.
#
# So the realised rate under sustained load is a number that has to be
# MEASURED, not argued (see acceptance GATE C in
# /spinning/evidence-665-f1/acceptance_665_f1.py, which reports flips/min
# alongside a 100 ms NVML corridor series). If it comes out too high, the
# remedy is `tp_decode_floor_s` -- the knob that already exists for exactly
# this -- and not a third mechanism bolted on here.
DEFAULT_DECODE_CONTENTION = 0.0

# Returned when the differential model proves no backlog can repay a flip --
# reachable only with the fairness window disabled (pp_window_s <= 0), where
# nothing bounds how long a carried decode waits. Deliberately a finite,
# recognisable number rather than sys.maxsize: it is compared against pending
# token counts, and a value that shows up in a log should read as "the policy
# says never", not as arithmetic that overflowed.
UNREACHABLE_FLIP_THRESHOLD = 1 << 40

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
#
# DEFAULTS ARE 0 = DISABLED, and that is a MEASURED retreat, not caution.
# 15/10 was the first guess and it killed three consecutive boots on this
# rig (2026-08-09, HANDOFF_658 section 4e), including one running the exact
# configuration that had survived 40+ minutes with zero exceptions -- the
# only difference being these windows. Signature was twice an allocation
# failure at the cutover seam (cuMemCreate / torch OOM with ~100 MiB free
# on a 3080) after 12 flips in four minutes.
#
# The mechanism is structural, not a bad constant: the windows raise the
# FLIP RATE, every cutover re-commits the KV backing, and that seam has a
# memory PEAK against ranks measured sitting only ~530-610 MiB above the
# corridor floor at runtime. Raising the flip rate is the wrong axis for
# fixing starvation while a cutover costs 1.0-1.7 s/rank AND spikes
# memory; make the seam memory-flat first, then re-tune these in MINUTES
# rather than seconds.
#
# Set > 0 deliberately, per deployment, with the corridor sampled in the
# PP phase specifically.
DEFAULT_PP_WINDOW_S = 0.0
DEFAULT_TP_DECODE_FLOOR_S = 0.0

# -- THE REFUSED-ARM CLOCK (#656, boot E) -----------------------------------
# A minute is long against a flip (1.0-1.7 s/rank) and short against an
# operator noticing, which is the window a hold that MIGHT heal should sit
# in: long enough that the retry costs nothing measurable, short enough that
# a transient -- an occupancy trough, a spill that lands, a peer's cache
# returned to the driver -- is picked up on its own.
DEFAULT_REFUSAL_BACKOFF_CAP_S = 60.0
# 8, the same number as the runtime's DEFAULT_SEAM_ABANDON_CAP, and for the
# same reason: past it, the evidence says the ask is structural.
DEFAULT_REFUSAL_DEGRADE_AFTER = 8

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


class FlipCostEstimator:
    """#677: the flip cost, MEASURED, not believed.

    ``DEFAULT_FLIP_COST_S = 3.2`` was derived from 997/1246/1720 ms in the
    PINNED-image era. The file-backed arm measured 22-24 s per leg (5 flips x 3
    ranks, ``NOTE_690_refill_commit_split.md``), and the live boot's own policy
    line still reads ``break-even 3.2s ... N=7004`` against a true break-even of
    ~49,250 tokens. Flips were priced **7.03x too cheaply**, which also makes
    #759's IDLE-LOCK floor -- itself ``flip_tokens`` -- 7x too permissive.

    #802 REPRICED THE LEG, so do not plan against 22-24 s any more. Most of
    that number was not transfer at all: the refill copied straight off the
    file-backed mapping and took one synchronous major fault per 4 KiB page.
    Reading the file instead (A/B on one binary, same load, same bytes moved)
    took the slowest rank's leg from 11.070 s to 4.246 s and the whole flip
    from 12.121 s to 4.998 s. The signature is in the RATES: on the fault path
    the ranks converge (821 and 775 MiB/s) despite PCIe links differing by
    1.80x, which is what says no DMA took part; with the read path they
    diverge again (2651 / 2602 / 3751 MiB/s) because the link matters once the
    transfer is real.

    None of this is a constant here, and deliberately so -- the estimator is
    fed the measured leg by ``observe_flip_cost`` at every refill, so a regime
    change like #802's reprices itself. The numbers above are provenance for
    the reader, not inputs.

    A constant cannot survive a regime change it does not know about, so the
    cost is estimated from the seam's own marks instead. Four properties, each
    pinned by ``test_flip_cost_calibration_677.py``:

    * **a measurement beats the seed OUTRIGHT.** The first observation replaces
      the config value rather than being averaged with it. Blending evidence
      with a belief is exactly how a 3.2 s constant would survive several 22 s
      flips.
    * **EMA thereafter**, so a regime change (pinned <-> file-backed) is tracked
      in both directions rather than latched.
    * **BOUNDED per sample.** One pathological reading is clamped into a band
      around the current estimate. An unbounded estimator would let a single
      600 s outlier price the break-even so high that flipping stops -- which is
      not a conservative failure, it is a different one.
    * **inert until it measures.** With no observation, ``value()`` is the seed,
      so an un-instrumented path is byte-identical to before.

    Non-finite and negative samples are REJECTED rather than clamped: they are
    not implausible readings, they are not readings.
    """

    #: Deliberately brisk. The regime change this exists for is a step, not
    #: drift, and a slow filter would spend a dozen mispriced flips converging.
    ALPHA = 0.3
    #: Per-sample clamp, as a multiple of the current estimate, both ways.
    OUTLIER_BAND = 3.0
    #: Absolute floor under the estimate, so the BAND can never collapse.
    #:
    #: The band is multiplicative, so an estimate near zero has a near-zero
    #: band and every later reading is clamped back into it. Measured on the
    #: pre-fix code: a first sample of 1e-6 s -- a no-op refill leg -- priced
    #: break-even at ZERO tokens (every pending prefill funds a flip, #748's
    #: churn signature from the other end) and needed 40 samples to recover,
    #: which at 22.5 s a leg is a quarter of an hour.
    #:
    #: 50 ms is 20x below the fastest per-rank leg this rig has ever measured
    #: (997 ms, the pinned era quoted in this docstring), so it cannot bind on
    #: a genuine reading. A floor that could would be a second wrong constant,
    #: not a guard -- pinned by
    #: test_flip_cost_clamp_directions_677.test_the_floor_is_far_below_any_real_measurement.
    MIN_ESTIMATE_S = 0.05
    #: Consecutive out-of-band samples ON THE SAME SIDE that make a regime.
    #:
    #: THE CLAMP BOUNDS AN OUTLIER, NOT A REGIME, and conflating the two is
    #: what made this estimator slow. Its stated purpose is that "one
    #: pathological reading" must not price flipping out -- ONE. Rate-limiting
    #: a SUSTAINED change is the EMA's job, and doing it a second time in the
    #: clamp cost eight samples on a step the seam had already reported twice
    #: (measured: 3.1 -> 22.5 took 8 samples, 2.4 -> 22.5 took 8, 22.5 -> 3.1
    #: took 13). A second consecutive reading on the same side is no longer a
    #: pathology, it is the new cost, and it is taken on the same rule the
    #: first sample is: evidence beats the current belief outright.
    #:
    #: The counter RESETS on any in-band sample, so two outliers an hour apart
    #: never read as a regime.
    REGIME_CONFIRM_SAMPLES = 2

    def __init__(self, seed_s: float, alpha: float = ALPHA):
        self.seed_s = float(seed_s)
        self.alpha = float(alpha)
        self._ema: Optional[float] = None
        self.samples = 0
        #: Length and sign of the current run of out-of-band samples.
        self._out_of_band = 0
        self._out_of_band_dir = 0

    @property
    def calibrated(self) -> bool:
        return self._ema is not None

    def value(self) -> float:
        return self.seed_s if self._ema is None else self._ema

    def observe(self, seconds) -> None:
        try:
            x = float(seconds)
        except (TypeError, ValueError):
            return
        # A ZERO-SECOND LEG IS NOT A FAST FLIP, IT IS A BROKEN TIMER, and the
        # module's own rule for a non-reading is to ignore it rather than
        # clamp it. The leg brackets a multi-GiB arena refill
        # (phase_flip_boot, arena_refill), so zero is only reachable when the
        # copy was skipped or short-circuited -- which says nothing about what
        # a flip costs. Measured on the pre-fix code: `observe(0.0)` as the
        # FIRST sample set the estimate to 0.0, collapsed the band to [0, 0],
        # and LATCHED there -- 50 subsequent 22.5 s flips moved it not at all
        # -- after which `break_even_tokens` raises "flip cost must be
        # positive" from inside config construction. The mid-stream zero was
        # already guarded by the clamp; only the first one was not, because
        # sample one takes the "measurement beats the seed outright" path
        # before any clamp exists.
        if not math.isfinite(x) or x <= 0.0:
            return
        if self._ema is None:
            # First measurement REPLACES the seed. See the class docstring.
            self._ema = max(x, self.MIN_ESTIMATE_S)
            self.samples = 1
            return
        lo = self._ema / self.OUTLIER_BAND
        hi = self._ema * self.OUTLIER_BAND
        if x > hi:
            direction = 1
        elif x < lo:
            direction = -1
        else:
            direction = 0
        if direction == 0:
            # In band: whatever run was building is over. Resetting HERE is
            # what keeps two outliers an hour apart from reading as a regime.
            self._out_of_band = 0
            self._out_of_band_dir = 0
        else:
            if direction == self._out_of_band_dir:
                self._out_of_band += 1
            else:
                self._out_of_band_dir = direction
                self._out_of_band = 1
            if self._out_of_band >= self.REGIME_CONFIRM_SAMPLES:
                # CONFIRMED. The seam has now reported the same new cost twice
                # running; it is the regime, not a pathology, and it is taken
                # outright for the same reason the first sample is.
                self._ema = max(x, self.MIN_ESTIMATE_S)
                self.samples += 1
                self._out_of_band = 0
                self._out_of_band_dir = 0
                return
            x = min(max(x, lo), hi)
        self._ema = max(self.MIN_ESTIMATE_S, self._ema + self.alpha * (x - self._ema))
        self.samples += 1


class RoundTripFlipCost:
    """C is a ROUND TRIP, so it is priced as TWO LEGS (#856).

    THE DEFECT THIS CLOSES, in this module's own words. The model at the top
    of this file defines ``C = round-trip flip cost, seconds``, and
    ``break_even_tokens`` refuses a bad premise by saying flipping "never
    repays the {flip_cost_s}s round trip". But ``observe_flip_leg`` feeds ONE
    LEG per sample -- its own docstring even computes the round trip it is not
    feeding, "tp_to_pp 11490 + pp_to_tp 5681 = 17171 ms" -- and both
    directions went into ONE estimator.

    THE TWO LEGS ARE NOT THE SAME QUANTITY. W25 measured, per flip, on the
    binding rank: ``tp_to_pp`` 10466-13181 ms against ``pp_to_tp`` 5078-6545
    ms. Feeding both to one EMA converges to neither: the seam figure the
    policy spent was 8.50 s, while the legs it was built from were 11.6 and
    6.4 and their SUM -- the actual round trip -- was 18.06 s.

    #819's own closing sentence is the rule it then broke one level up: "a
    component and its container are different quantities and an EMA fed both
    alternately converges to neither". Two directions are different
    quantities too.

    REPRODUCED EXACTLY, which is what makes this a measurement and not a
    reading of the log. Replaying PP0's eleven ``PHASE-FLIP DONE`` totals from
    boot_w25_0824_1125.log through one estimator at ALPHA=0.3 yields
    5.0779 / 6.6944 / 6.2494 / 7.5450 / 7.2426 / 9.0241 / 8.2740 / 9.2457 /
    8.4356 / 9.3990 / 8.5041 -- and the boot's own decision lines printed
    N=15853 / 18110 / 18464 / 18614 at exactly the samples that price to
    7.2426 / 8.2740 / 8.4356 / 8.5041. The bar also OSCILLATED by ~2000 tok
    with flip-direction parity (8.50 after a pp_to_tp, 9.40 after a
    tp_to_pp), which is a pure artifact of the blend and nothing about cost.

    SO EACH DIRECTION KEEPS ITS OWN ESTIMATOR and C is their SUM. The leg
    estimator is REUSED, not rebuilt: every property #677 pinned on it (a
    measurement beats the seed outright, the outlier band, the two-sample
    regime confirmation, tracking DOWN as readily as up) holds per leg.

    THE SEED IS SPLIT IN HALF, so an uncalibrated instance values exactly the
    round-trip seed and the pre-#856 default path is unchanged byte for byte.
    A half-measured state -- one leg measured, one still on its seed half --
    is a REAL state with its own name (see ``provenance``), because "measured"
    covering it would be the silent multi-valued word this build keeps
    removing.
    """

    #: The two legs of a round trip. A sample naming anything else is not
    #: attributable and is refused rather than filed under a guess.
    LEGS = ("tp_to_pp", "pp_to_tp")

    def __init__(self, seed_s: float, alpha: Optional[float] = None):
        self.seed_s = float(seed_s)
        self.alpha = FlipCostEstimator.ALPHA if alpha is None else float(alpha)
        self._legs = {
            leg: FlipCostEstimator(seed_s=self.seed_s / 2.0, alpha=self.alpha)
            for leg in self.LEGS
        }

    def leg(self, direction: str) -> Optional[FlipCostEstimator]:
        return self._legs.get(str(direction))

    @property
    def calibrated(self) -> bool:
        """Any leg measured -- repricing engages on the FIRST flip, as before.

        Deliberately not ``all``: waiting for both legs would leave the whole
        bar on the seed through an entire first phase, which is the state
        #819 exists to end. A one-leg price is already strictly better than
        an unmeasured constant; ``provenance`` is what stops it being read as
        more than it is.
        """
        return any(est.calibrated for est in self._legs.values())

    @property
    def fully_calibrated(self) -> bool:
        return all(est.calibrated for est in self._legs.values())

    def provenance(self) -> str:
        """What the live C actually rests on. Three states, three words."""
        measured = [leg for leg in self.LEGS if self._legs[leg].calibrated]
        if not measured:
            return "seed"
        if len(measured) == len(self.LEGS):
            return "measured"
        return f"half-measured ({measured[0]} only)"

    def value(self) -> float:
        """The round trip: both legs, always summed."""
        return float(sum(est.value() for est in self._legs.values()))

    def observe(self, seconds, direction=None) -> None:
        """One leg when the direction is known; a whole round trip when not.

        An UNDIRECTED sample is a round-trip reading and is split evenly
        across the legs, so ``value()`` still returns exactly what was
        observed. That keeps every caller that measured a round trip honest
        without inventing a direction for it -- and a direction this class
        does not know is refused outright, because filing a ``pp_to_dcp`` leg
        under ``tp_to_pp`` would corrupt both.
        """
        if direction is None:
            try:
                half = float(seconds) / 2.0
            except (TypeError, ValueError):
                return
            for est in self._legs.values():
                est.observe(half)
            return
        est = self._legs.get(str(direction))
        if est is None:
            return
        est.observe(seconds)


#: Process-wide, because the seam that measures a flip and the policy that
#: prices the next one are different modules on the same rank.
_FLIP_COST_ESTIMATOR: Optional[RoundTripFlipCost] = None


#: What `config_from_env` actually froze into `PhasePolicyConfig.flip_tokens`,
#: and the two throughputs it was priced with. Module-global for the same
#: reason the estimator is: the seam that measures a flip and the config that
#: priced it are built in different places on the same rank. None until a
#: config has been built (a process that never arms the policy says nothing).
_FLIP_TOKENS_AT_BOOT: Optional[int] = None
_FLIP_TOKENS_PRICING: Optional[tuple] = None
#: One-shot latch for the staleness warning (#777). The condition is true on
#: every flip once it is true at all, and this is a boot-configuration fact,
#: not a per-flip event.
_FLIP_TOKENS_STALE_SAID = False


def note_flip_tokens_pricing(flip_tokens, tp_tok_s, pp_tok_s, explicit) -> None:
    """Record how N was priced, so a later measurement can say it went stale."""
    global _FLIP_TOKENS_AT_BOOT, _FLIP_TOKENS_PRICING, _FLIP_TOKENS_STALE_SAID
    _FLIP_TOKENS_AT_BOOT = int(flip_tokens)
    _FLIP_TOKENS_PRICING = (float(tp_tok_s), float(pp_tok_s), bool(explicit))
    _FLIP_TOKENS_STALE_SAID = False


def repriced_flip_tokens() -> Optional[tuple]:
    """(n_boot, n_now, seam_boot, seam_now), or None when there is nothing to say.

    #777. N is priced once, at boot, off an UNMEASURED seam seed, and never
    repriced; the estimator meanwhile tracks what flips really cost. This
    reports what N would be if it were priced on the measurements taken since
    -- a statement, not an action. Nothing here changes the live threshold.

    None for every reason the comparison would be meaningless, and they are
    kept apart at the call site rather than collapsed here: no config built
    yet, no measurement yet, N pinned explicitly by the operator (then it is
    an assertion, not a derivation), or a pricing the break-even formula
    refuses.
    """
    est = _FLIP_COST_ESTIMATOR
    if est is None or not est.calibrated:
        return None
    if _FLIP_TOKENS_AT_BOOT is None or _FLIP_TOKENS_PRICING is None:
        return None
    tp_tok_s, pp_tok_s, explicit = _FLIP_TOKENS_PRICING
    if explicit or tp_tok_s <= 0:
        return None
    try:
        n_now = break_even_tokens(est.value(), tp_tok_s, pp_tok_s)
    except PhasePolicyError:
        return None
    return (_FLIP_TOKENS_AT_BOOT, int(n_now), est.seed_s, est.value())


def observe_flip_cost(seconds, direction=None) -> None:
    """Feed one measured flip-leg duration to the estimator (#677).

    #856: ``direction`` names WHICH LEG this reading prices. Omitting it means
    the reading is a whole round trip (see ``RoundTripFlipCost.observe``), not
    "either leg, whichever" -- a leg filed under the wrong direction is worse
    than one not filed at all.

    #777: and SAY IT when that measurement has left the live threshold behind.
    The estimator following the regime while N stays frozen at the seed is the
    counter-without-an-actuator shape; the actuator is the planner's call, but
    the silence was this module's.
    """
    global _FLIP_TOKENS_STALE_SAID
    if _FLIP_COST_ESTIMATOR is None:
        return
    _FLIP_COST_ESTIMATOR.observe(seconds, direction)
    if _FLIP_TOKENS_STALE_SAID:
        return
    repriced = repriced_flip_tokens()
    if repriced is None:
        return
    n_boot, n_now, seam_boot, seam_now = repriced
    if n_now == n_boot:
        return
    _FLIP_TOKENS_STALE_SAID = True
    logger.warning(
        "%s #777 N IS STALE AND STAYS STALE. The live threshold is N=%d tok, "
        "priced at boot off the UNMEASURED seam seed %gs. Measured flips put "
        "the seam at %gs, which prices the same break-even at %d tok (%.2fx). "
        "N is built once, in config_from_env, and nothing reprices it, so the "
        "server keeps arming against %d for the rest of this process. "
        "Repricing is a policy decision (it moves when the server flips at "
        "all) and belongs to the planner -- this line only refuses to let the "
        "gap stay silent. Set %s to pin N deliberately.",
        LOG_PREFIX,
        n_boot,
        seam_boot,
        seam_now,
        n_now,
        (n_now / n_boot) if n_boot else float("nan"),
        n_boot,
        ENV_FLIP_TOKENS,
    )


def observe_flip_leg(stats) -> None:
    """Feed the estimator THE WHOLE LEG, not one step of it (#819).

    WHAT WAS BEING PRICED. The only feeder was ``_timed_arena_refill``, which
    brackets the weights-arena copy and says so ("the number is the refill leg
    proper and nothing else"). The arena refill is ONE STEP of a flip --
    phase_flip_runtime's own header calls the weights-arena refill and the KV
    seam "separate steps of the flip" -- so the estimator was pricing a
    component and the policy was spending it as if it were the whole.

    MEASURED, on boot_window3_0823_1733.log. The estimator reported the seam
    at 3.60287 / 3.66144 / 4.62869 s, while the same boot's own PHASE-FLIP
    DONE lines put a leg at 5681-12023 ms, and a round trip at
    tp_to_pp 11490 + pp_to_tp 5681 = 17171 ms. The refill leg is roughly half
    of one leg, so the flip was priced at well under half of what it cost.

    WHY THIS MATTERS BEYOND ACCURACY, and why #834 needs it: ``total_ms``
    CONTAINS ``movers_ms`` and ``cutover_ms``, and the refill leg contains
    neither. ``SGLANG_SEAM_SHRINK`` shrinks the cutover -- so with only the
    refill fed, the seam shrink could not move the price by construction, no
    matter how much it saved. The coupling was not mistuned, it was absent.

    ONE SAMPLE PER LEG. The refill feed is retired rather than added to, because
    a component and its container are different quantities and an EMA fed both
    alternately converges to neither.
    """
    if not isinstance(stats, dict):
        return
    total_ms = stats.get("total_ms")
    if total_ms is None:
        return
    try:
        seconds = float(total_ms) / 1000.0
    except (TypeError, ValueError):
        return
    # #856: the seam stamps every completed flip with its direction
    # (phase_flip_runtime, `stats["direction"]`), so the leg is attributable
    # at the only place that knows it. A leg whose direction is missing is
    # priced as a round trip rather than guessed into one of the two.
    observe_flip_cost(seconds, stats.get("direction"))


def flip_cost_estimator() -> Optional[RoundTripFlipCost]:
    return _FLIP_COST_ESTIMATOR


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


def live_flip_tokens(cfg: PhasePolicyConfig) -> int:
    """N, PRICED FROM THE MEASURED SEAM instead of frozen at the boot seed (#819).

    THE ACTUATOR #777 DECLINED TO BUILD. ``repriced_flip_tokens`` already
    computed this number and said so in a WARNING -- "N IS STALE AND STAYS
    STALE ... nothing reprices it" -- explicitly leaving the decision to the
    planner because repricing moves when the server flips at all. This is that
    decision, taken: the threshold follows the measurement.

    THE SEED IS A COLD-START PRIOR, NOT A PIN (#770 family). ``flip_tokens``
    is priced once in ``config_from_env`` off ``DEFAULT_FLIP_COST_S = 3.2``, a
    PINNED-IMAGE-ERA constant that no longer describes this seam; on the
    window-3 boot it produced N=7004 while the same process measured the seam
    at 3.60287 / 3.66144 / 4.62869 s on its three ranks, pricing the same
    break-even at 7886 / 8014 / 10131. The seed does not decay on a timer --
    it is SUPERSEDED, on the estimator's own rule that a measurement beats a
    belief outright on the first sample.

    RANK SAFETY, since the three prices above differ and a per-rank bar would
    be a #616g divergence: the policy is NOT evaluated independently per rank.
    ``recv_requests`` runs the hook only on the request-origin rank and
    BROADCASTS the arm (request_receiver.py:169-193; its comment records the
    measured alternative -- "1/2/3 arms on PP0/PP1/PP2, a 12765-line
    capture-census flood, and a self-kill"). One rank prices, one rank decides,
    every rank obeys the same broadcast arm. No new collective is taken here,
    and none is available: the one reduce that runs per iteration is a
    one-element MIN over pool availability.

    THE LIMITATION IS NAMED, NOT ASSUMED AWAY. The deciding rank prices off
    ITS OWN leg, while ``phase_flip_boot`` states "the flip's cost is the
    SLOWEST rank's copy". When the decider is the fast rank the bar is priced
    low. This is strictly better than an unmeasured constant -- the error goes
    from seed-vs-reality to fast-rank-vs-slowest-rank -- but it is an error,
    and closing it needs a group MAX that no existing collective carries.

    Returns the frozen ``cfg.flip_tokens`` unchanged for every reason the
    repricing would be meaningless (no measurement yet, no config recorded, an
    operator-pinned N, or a pricing the break-even formula refuses), so an
    un-instrumented deployment is byte-identical to before.
    """
    repriced = repriced_flip_tokens()
    if repriced is None:
        return int(cfg.flip_tokens)
    return int(repriced[1])


def flip_cost_measured() -> bool:
    """Whether the live bar rests on a measurement or still on the seed (#819).

    PROVENANCE, IN THE DECISION LINE ITSELF. A repriced threshold that does not
    say what repriced it is the counter-without-an-actuator shape from the
    other end: a reader seeing N move has no way to tell a measured seam from
    an edited constant. Cheap to print, and it is the first thing anyone
    debugging a surprising bar will want.
    """
    return repriced_flip_tokens() is not None


def flip_cost_fully_measured() -> bool:
    """BOTH legs measured -- the only state that is evidence, not assumption.

    #856: `flip_cost_measured` engages repricing as soon as EITHER leg has a
    measurement, which is right for pricing (one measured leg beats an
    unmeasured constant). It is NOT right for the #838 detector, whose gate
    refuses to question a bar priced off the seed on the stated ground that
    "an assumption is not the policy's own claim". A half-measured round trip
    is still half assumption, so it is refused on exactly the same ground
    rather than promoted to evidence by a boolean that cannot see the
    difference. The blast radius is one-directional: the detector can only
    DECLINE more often, never alarm more often.
    """
    if repriced_flip_tokens() is None:
        return False
    est = _FLIP_COST_ESTIMATOR
    return bool(est is not None and est.fully_calibrated)


def flip_cost_provenance() -> str:
    """The SAME provenance, but able to say "half" (#856).

    ``flip_cost_measured`` is a boolean over a quantity that has three states
    once C is a round trip of two independently-measured legs: neither leg
    measured, one measured, both measured. A boolean prints the middle state
    as "measured", and a reader then has no way to tell a fully-priced round
    trip from one whose second leg is still the seed half -- the silent
    multi-valued word this build has removed twice already (#851 class,
    #853(i) on the exposure gate, #854 on the economy detector).
    """
    if repriced_flip_tokens() is None:
        return "seed"
    est = _FLIP_COST_ESTIMATOR
    return "measured" if est is None else est.provenance()


def live_flip_cost_s(cfg: PhasePolicyConfig) -> float:
    """The seam cost C the ladder is priced with, MEASURED where possible (#819).

    Repricing N alone is not enough, and the reason is in
    ``_differential_flip_threshold``: that solve derives the ladder from ``C``,
    ``X`` and ``P`` DIRECTLY -- "`base` survives only as the floor it should
    always have been" -- so a repriced N left the whole surcharge ladder still
    priced off the boot seed. Measured on the window-3 numbers: N moved
    7004 -> 10131 while the 4-decode ceiling stayed at 12608, i.e. the bar
    moved and the band above it did not.

    GATED ON THE SAME CONDITION AS ``live_flip_tokens``, deliberately. Either
    the seam has been measured and BOTH the break-even and the ladder follow
    it, or nothing is repriced and the path is byte-identical to before. A
    build that repriced one and not the other is the state this function
    exists to make unreachable.
    """
    if repriced_flip_tokens() is None:
        return float(cfg.flip_cost_s)
    est = _FLIP_COST_ESTIMATOR
    return float(cfg.flip_cost_s) if est is None else float(est.value())


def effective_flip_threshold(cfg: PhasePolicyConfig, running_bs: int) -> int:
    """Pending-prefill tokens required to justify `tp_to_pp` RIGHT NOW.

    ``break_even_tokens`` prices the seam and nothing else. A cutover also
    PAUSES every request that is decoding: it is carried into the PP layout,
    where decode is forbidden, and waits out the PP window before emitting
    another token. Stranding is therefore a real cost of the flip, it scales
    with how many requests are in flight, and it belongs in the same
    comparison rather than in a separate ad-hoc guard.

    The seconds go into the same C that produced N, so the threshold simply
    scales:

        C_eff = flip_cost_s + weight x running_bs x stranded_decode_s(cfg)
        N_eff = flip_tokens x C_eff / flip_cost_s

    #893: THE SECONDS ARE THE RESIDENCY THAT GOVERNS, not the requested knob.
    This term used to read ``cfg.pp_window_s`` directly, which #889 showed is
    unreachable whenever a decode-stall SLO supersedes it -- on the live line
    that charged a flip for 15 s of stranding while the decodes waited 173.6 s,
    understating the ladder ~9.7x, and on the configuration
    ``validate_purity_policy_pair`` recommends (window cleared, SLO declared)
    it multiplied by zero and switched the surcharge off entirely.

    Degenerate cases return the unchanged threshold, so a deployment that has
    not measured its seam behaves exactly as before:
      * ``flip_cost_s <= 0``  -- seam not measured here;
      * ``running_bs <= 0``   -- nothing to strand;
      * ``weight == 0``       -- surcharge explicitly disabled.

    Under ``strict`` purity (``prefill_runs_in_tp`` False) the threshold is 0
    by construction -- a sub-N prompt could not run in TP at all -- and the
    surcharge must not resurrect one, or short prompts would never run.
    """
    if not cfg.prefill_runs_in_tp:
        return 0
    # #819: the whole ladder is derived from N, so N is taken LIVE here. Were
    # the base frozen while the surcharge scaled, the lower bar and the upper
    # band would be priced off two different seams.
    base = live_flip_tokens(cfg)
    flip_cost_s = live_flip_cost_s(cfg)
    if flip_cost_s <= 0 or running_bs <= 0:
        return base
    if cfg.decode_contention > 0.0:
        return _differential_flip_threshold(cfg, running_bs, base)
    if cfg.decode_strand_weight <= 0:
        return base
    # #893: `stranded_decode_s`, never `cfg.pp_window_s`. `base` stays OUTSIDE
    # the surcharge on purpose -- at running_bs 0 there is nothing to strand,
    # and the reprice may raise the ladder's SLOPE, never its floor.
    stranded_s = cfg.decode_strand_weight * float(running_bs) * stranded_decode_s(cfg)
    return int(round(base * (flip_cost_s + stranded_s) / flip_cost_s))


def unpriced_seam_note(cfg: PhasePolicyConfig, running_bs: int = 0) -> Optional[str]:
    """The seam costs seconds and NOTHING is required to justify paying it.

    #874. A two-token health-check ping took 15.6-16.6 s on boot w40, three
    times over four and a half hours, on a server with no other traffic. None
    of it was compute. The ping is 13 tokens; under ``strict`` purity prefill
    cannot run in the TP layout, so the 13 tokens armed ``tp_to_pp``, the
    drain armed ``pp_to_tp`` right back, and the request paid two cutovers of
    ~6.2 s each plus the 3 s minimum dwell between them.

    EVERY ONE OF THAT BOOT'S CUTOVERS CARRIED ZERO KV (``sent 0 cells /
    0.00 MiB``, 611 of 611). The seconds are the weights arena: 16.4 GiB per
    rank, refilled host-to-device on every layout entry because the PP and TP
    layouts hold disjoint shards and the inactive one is not resident. The
    seam census names it directly -- ``refill_highwater->weights_refill``,
    4868 ms of a 6299 ms walk. That term is occupancy-independent, so it does
    not shrink for a small request: a 13-token ping and a 40k-token prompt pay
    the same seam.

    NOTHING HERE IS A BUG, AND THAT IS THE POINT. ``effective_flip_threshold``
    returns 0 under strict purity deliberately -- a sub-N prompt could not run
    in TP at all, and a threshold that survived would leave short prompts
    unrunnable, which is the wedge recorded at the scheduler's wiring site ("a
    one-token health check wedged an otherwise idle server"). The arm then
    fires on any pending token, deliberately, against the user's law
    "Break-even ist NICHT der Trigger". Both halves are correct in isolation.

    WHAT IS MISSING IS THAT NOBODY SAYS THE PRODUCT OUT LOUD. A large fixed
    cost whose price gate has been switched off will fire at whatever rate the
    arrival pattern dictates, and the instance reports the rate (flip counts)
    and the unit cost (DONE lines) in two different places while never
    multiplying them. The state is decidable from config alone, so it is
    reported at boot rather than discovered by timing a ping:

      * ``live_flip_cost_s > 0``               the seam costs something, and
      * ``effective_flip_threshold(...) == 0``  nothing is required to buy it.

    THE FIGURE IS THE ROUND TRIP, not one leg, because that is what a single
    request costs: out to the prefill layout and back to decode. Reporting one
    leg would halve the number an operator has to act on.

    Returns None whenever there is nothing to report -- a priced seam, or a
    seam that costs nothing (unmeasured, or a deployment where the layouts
    share their weights). A note on every config would be a note nobody reads.
    """
    cost_s = live_flip_cost_s(cfg)
    if cost_s <= 0:
        return None
    if effective_flip_threshold(cfg, int(running_bs)) > 0:
        return None
    return (
        f"UNPRICED SEAM: the flip threshold is 0, so ANY pending prefill token "
        f"arms a cutover, and one request pays the round trip -- "
        f"{2.0 * cost_s:.1f}s at the seam priced RIGHT NOW ({cost_s:.1f}s per "
        f"leg; the estimator supersedes the boot seed on its first sample), "
        f"plus {cfg.min_dwell_s:g}s of minimum dwell between the two legs. "
        f"That cost is occupancy-independent (it is the weights arena, not "
        f"the KV), so a 1-token health check pays the same as a full backlog. "
        f"This is the DESIGNED behaviour of purity mode 'strict', not a "
        f"defect: prefill cannot run in the tp layout there, so every prefill "
        f"must reach pp regardless of size. Purity mode 'prefill_in_tp' "
        f"restores the break-even gate (and is this tree's default), at the "
        f"cost of allowing prefill in the decode layout below it. #874."
    )


#: How many chunk-cadences of silence make a stall a WEDGE rather than a slow
#: tick. Small on purpose: the rule already refuses to fire while anything is
#: moving, so its only job is to outlast ordinary jitter between chunks.
PROGRESS_STALL_CHUNKS = 3


def pp_progress_stall_window_s(cfg: PhasePolicyConfig) -> float:
    """How long prefill may make NO progress before PP is declared wedged.

    SOLVED, NOT SET, and from the two quantities that already describe the
    phase: one chunk takes ``pp_exit_tokens / pp_prefill_tok_s`` seconds at the
    measured rate, so several of those passing with nothing admitted is not a
    slow drain, it is a drain that has stopped.

    FLOORED AT ``2 * flip_cost_s``. A cutover is the one interval in which
    prefill legitimately makes no progress at all, so a rig fast enough to
    solve a window shorter than its own seam would otherwise exit on the seam
    it just paid for -- arming a flip because it was flipping.

    0 disables the rule, which is what an unusable rate must produce: a
    division by zero here would be a wedge-breaker that wedges.

    WHY THIS IS NOT ANOTHER STOPWATCH, and the distinction is the whole point
    of #677(a). The user's pure-drain decision removed the clock deliberately:
    a real drain must never be cut short however long it takes. This rule
    cannot cut one short, because ANY progress resets it -- one chunk per
    window is enough to hold PP forever. It fires only when the thing the
    drain is waiting for has stopped happening.
    """
    rate = float(cfg.pp_prefill_tok_s)
    chunk = int(cfg.pp_exit_tokens)
    floor = 2.0 * float(cfg.flip_cost_s)
    if rate <= 0 or chunk <= 0:
        return 0.0
    return max(PROGRESS_STALL_CHUNKS * (chunk / rate), floor)


def prefill_suppressed_in_tp(
    cfg: PhasePolicyConfig,
    phase: str,
    flip_unavailable: bool = False,
    running_bs: int = -1,
) -> bool:
    """#677 hot fix 2: may prefill be admitted while the TP layout is up?

    Under drain mode, no. The TP window exists to finish a decode bundle, and
    prefill admitted into it competes with exactly the work the window was
    entered for -- measured as visible "Prefill batch" lines inside TP while
    carriers went unfinished across repeated flips.

    CARDS PARTIALLY IDLE DURING TP IS ACCEPTED, deliberately, and is the
    user's stated model: the alternative measured worse, because a bundle that
    never finishes costs a whole extra round trip and returns the same
    carriers.

    IT DEFERS TO THE PURITY VALVE, and that clause is the whole lesson of two
    live wedges. 2026-08-16 06:47:48: CorridorGuard refused the seam staging on
    two ranks with static numbers -- PP1 want 2163 MiB against 2456 free with
    an arming floor of 1536, PP2 want 2858 against 3560 -- because the staging
    a 4-carrier bundle needs EXCEEDS the floor that was supposed to guarantee
    it. Every degradation had already stood down; the arm refused 76 times in
    a row.

    That alone would have been a slow boot. It became a TOTAL wedge because
    suppressing prefill in TP removed the fallback the instance used to have:
    before drain mode, a refused tp_to_pp still prefilled in the TP layout and
    the backlog drained slowly instead of not at all. An idle server with
    727004 tokens waiting is strictly worse than a slow one.

    Then 07:02 said it again in a shape the first fix could not see. tp_to_pp
    was not REFUSED, it was DELAYED: one rank withheld its entry-margin yield
    on a predicted sub-law trough, and the withhold is deliberately exempt
    from the stand-down cap, so "consecutive delayed attempts" climbed 15, 16,
    17 with no exit while my fix watched a refusal counter that never moved.
    A different counter, the same wedge.

    SO THE QUESTION IS NOT "WAS IT REFUSED" BUT "IS PP REACHABLE". Drain
    mode's premise is that prefill belongs in PP and should wait for PP; that
    premise holds only while PP can be reached. ``flip_unavailable_reason``
    already answers exactly this, over BOTH books -- the seam's own
    ``_seam_abandons_in_a_row``, which delays DO advance, and the policy's
    ``arm_refusals`` -- so both wedge shapes leave through one door.

    ONE BOUND, NOT TWO. This replaces the threshold of my own that the first
    fix introduced. Keying on the valve inherits its bound and its two books;
    a second number would have been one more thing to tell the wrong state,
    which is the entire failure mode of this chain.

    NOT A LATCH. ``arm_refusals`` is reset by the first successful arm, so an
    instance that recovers returns to the user's semantics by itself. This
    chain has spent four tasks removing one-way ratchets; it is not adding a
    fifth.

    False for every phase but TP, and False everywhere when drain mode is off,
    so a deployment that has not opted in is byte-identical.
    """
    if not bool(cfg.drain_mode) or phase != PHASE_TP:
        return False
    if flip_unavailable:
        return False
    # AND THERE MUST BE A BUNDLE TO PROTECT. Measured 2026-08-16 08:03 on my
    # own fix: an 18-token request hung for two minutes with `running bs 0`
    # and GPU at 0%, while the policy logged "pending prefill 18 tok <= N=7004,
    # running it in tp".
    #
    # THE POLICY WAS RIGHT AND THE SUPPRESSION OVERRODE IT. Below the
    # break-even N a flip costs more than it saves, so the policy deliberately
    # keeps the work in TP and arms nothing. The valve cannot help: the flip
    # is not FAILING, it simply was not asked for -- so `flip_unavailable` is
    # false and suppression held a request that nothing was ever going to run.
    # Long prompts hid it, because 25625 tokens sit above N and go to PP.
    #
    # Drain mode exists to stop a TP window admitting the work it was entered
    # to escape -- that window is defined by a decode bundle in flight. With
    # `running_bs == 0` the bundle is finished, there is nothing to drain, and
    # the only thing suppression can still do is idle the instance.
    #
    # -1 means the caller did not measure it, which must not be read as "no
    # bundle": an unmeasured input never becomes a licence.
    if int(running_bs) == 0:
        return False
    return True


def pp_residency_cap_s(cfg: PhasePolicyConfig) -> float:
    """Seconds the PP phase may hold decodes, SOLVED from the declared SLO.

    Returns 0 when nothing is declared, meaning drain alone governs the phase.
    Never returns a negative cap: an SLO tighter than the round trip it must
    contain cannot be honoured by leaving PP sooner, because the seams are
    already inside it -- so it collapses to "leave as soon as the drain rule or
    the dwell allows" rather than to a nonsense negative deadline.
    """
    if cfg.decode_stall_slo_s <= 0.0:
        return 0.0
    return max(0.0, cfg.decode_stall_slo_s - 2.0 * cfg.flip_cost_s)


#: #889 -- the three things that can end a PP residency on a CLOCK, named so a
#: boot line can say which one is live instead of printing a knob that is not.
PP_EXIT_BY_DRAIN = "drain only"
PP_EXIT_BY_SLO_CAP = "decode-stall cap"
PP_EXIT_BY_STOPWATCH = "hand-set stopwatch"


def effective_pp_exit_term(cfg: PhasePolicyConfig) -> tuple[str, float]:
    """Which timed bound on PP residency is REACHABLE, and how big it is (#889).

    THE DEFECT THIS EXISTS TO END. ``decide`` reaches the hand-set stopwatch
    only through ``cap <= 0 and cfg.pp_window_s > 0``, so a declared
    ``decode_stall_slo_s`` above ``2 * flip_cost_s`` makes
    ``--phase-policy-pp-window-s`` UNREACHABLE. That supersession is intended
    (``phase_purity.validate_purity_policy_pair`` accepts the SLO as the
    substitute bound on purpose) -- what was not intended is that it happened
    without a word, while the boot line kept printing the inert number. Live on
    the w38b/w39/w40 line: window 15 s declared, SLO 180 s, seam 3.2 s, so the
    phase actually ran to 173.6 s -- 11.573x longer than every reader believed.

    THE GUARD ORDER HERE MIRRORS ``decide`` DELIBERATELY, and
    ``test_decide_agrees_with_the_reported_term_on_both_sides`` pins that they
    cannot drift apart. A reporter that derives the answer independently would
    be a second authority on the same question, which is the shape of the bug,
    not of the fix.
    """
    cap = pp_residency_cap_s(cfg)
    if cap > 0:
        return PP_EXIT_BY_SLO_CAP, cap
    if cfg.pp_window_s > 0:
        return PP_EXIT_BY_STOPWATCH, float(cfg.pp_window_s)
    return PP_EXIT_BY_DRAIN, 0.0


def stranded_decode_s(cfg: PhasePolicyConfig) -> float:
    """How long a decode carried into PP is REALLY stranded there (#893).

    THE ONE READER THE ENTRY ECONOMY IS ALLOWED TO HAVE. Every price a flip
    pays for the generations it pauses -- the one-sided surcharge in
    ``effective_flip_threshold``, and the regime boundary plus the saturation
    term inside ``_differential_flip_threshold`` -- is a number of seconds a
    carried decode waits before it can emit again. That is the PP residency
    the phase will actually run to, which #889 established is NOT
    ``cfg.pp_window_s`` whenever a decode-stall SLO supersedes it.

    Delegates rather than re-derives, deliberately. ``effective_pp_exit_term``
    already mirrors ``decide``'s guard order and is pinned against it; a second
    solve of "which bound governs" is precisely the shape of #889, so this is a
    NAME for that answer in the economy's vocabulary, not another authority on
    it. ``test_the_named_term_is_the_one_889_reports`` pins the delegation.

    Zero means "no timed bound is declared at all", which the callers must read
    as UNBOUNDED stranding (a carried decode waits out the whole prefill) --
    never as free stranding. Reading it as free is the second half of #893: on
    the configuration ``validate_purity_policy_pair`` recommends -- window
    cleared, SLO declared -- the surcharge term ``weight x running_bs x 0``
    switched the entire stranded-decode ladder off while decodes stranded for
    173.6 s.
    """
    return effective_pp_exit_term(cfg)[1]


def superseded_pp_bound_warning(cfg: PhasePolicyConfig) -> Optional[str]:
    """The line a boot must print when one PP bound silences the other (#889).

    ``None`` when at most one of the two is declared: a configuration that was
    never ambiguous must not acquire a warning, or the warning stops being read.

    WARNING, NOT REFUSAL -- decided on the danger direction, not on taste.

    * The combination is SANCTIONED, not broken. ``validate_purity_policy_pair``
      (phase_purity.py) explicitly accepts a declared SLO as the bound the
      stopwatch used to supply, and calls it "a better one". Refusing at parse
      what another validator accepts by design would make the two disagree
      about the same configuration.
    * The failure being fixed is a MISREAD, and its blast radius is a wrong
      belief. The failure a refusal would introduce is a DEAD BOOT, and its
      blast radius is the instance -- on a shipping config, since
      ``deploy/turnkey/stack.rig3.toml:41`` and ``ship_env.capture:38`` set the
      window unconditionally. Trading a documentation defect for an
      availability defect is the worse direction, and it is the direction a
      refusal picks.
    * Nor may the code silently "honour both" with ``min(cap, window)``. On the
      live pair that would cut the effective residency 173.6 s -> 15 s, i.e.
      raise the flip rate by ~11x -- which is verbatim the mechanism the block
      comment at ``DEFAULT_PP_WINDOW_S`` records as having killed three
      consecutive boots (12 flips in four minutes, allocation failure at the
      cutover seam). The safe move is to state the truth, not to change it.

    Both directions of supersession are reported, because both are silent: an
    SLO at or below the round trip collapses the cap to zero and is itself the
    inert knob.
    """
    window = float(cfg.pp_window_s)
    slo = float(cfg.decode_stall_slo_s)
    if window <= 0 or slo <= 0:
        return None
    seam = float(cfg.flip_cost_s)
    cap = pp_residency_cap_s(cfg)
    if cap > 0:
        direction = "LONGER" if cap > window else "SHORTER"
        ratio = cap / window
        return (
            f"{LOG_PREFIX} #889 SUPERSEDED KNOB: --phase-policy-pp-window-s "
            f"={window:g}s is INERT. A decode stall SLO of {slo:g}s is declared, "
            f"so the reachable bound on PP residency is the solved decode-stall "
            f"cap {cap:g}s (= {slo:g}s - 2x{seam:g}s seam), and the stopwatch "
            f"arm sits behind `cap <= 0` and can never fire. EFFECTIVE PP "
            f"RESIDENCY IS {ratio:.3g}x {direction} THAN THE REQUESTED WINDOW "
            f"({cap:g}s vs {window:g}s). Intended supersession, announced rather "
            f"than refused -- but read every PP-residency number against "
            f"{cap:g}s, not against {window:g}s. To make the window govern, "
            f"clear SGLANG_PHASE_POLICY_DECODE_STALL_SLO_S / "
            f"--phase-policy-decode-stall-slo-s."
        )
    return (
        f"{LOG_PREFIX} #889 SUPERSEDED KNOB: --phase-policy-decode-stall-slo-s "
        f"={slo:g}s is INERT. It does not clear the round trip it must contain "
        f"(2x{seam:g}s seam = {2 * seam:g}s), so the solved cap collapses to 0 "
        f"and the hand-set stopwatch governs instead: PP residency is bounded "
        f"at {window:g}s, not at {slo:g}s. Declare an SLO above {2 * seam:g}s "
        f"for it to take effect."
    )


def solved_tp_decode_floor_s(cfg: PhasePolicyConfig) -> float:
    """Minimum TP dwell, SOLVED: one full round trip.

    A cycle that spends less time serving decode in TP than it just spent
    entering and leaving TP has put more than half its wall clock into
    cutovers. `2 * flip_cost_s` is therefore the floor below which the phase
    machinery costs more than it moves -- measured, not chosen.
    """
    return 2.0 * cfg.flip_cost_s


def drain_stall_deadline_s(cfg: PhasePolicyConfig) -> float:
    """How long drain mode waits on an admitted set that is NOT shrinking.

    #833. Drain mode's exit condition is an empty decode bundle, and under
    sustained load that is a state admission structurally never revisits: it
    refills every slot the bundle frees. The bound here is what makes the wait
    terminate, so its size is the whole trade-off:

    * too SHORT and it re-creates the defect #677 hot fix 2 closed -- a bundle
      cut in half, its carriers coming back unfinished window after window;
    * too LONG and the instance serves single-phase indefinitely, which is the
      22-minute drought measured on boot_window3_0823_1733.

    SOLVED, not chosen: one full decode window. ``solved_tp_decode_floor_s`` is
    already this module's answer to "the shortest TP residency that is worth
    its own cutovers" (2 x flip_cost_s), and a bundle that has not retired a
    single request in that long is not being served by waiting longer. The
    floor keeps it meaningful when ``flip_cost_s`` is unset or tiny, where an
    unfloored deadline would arm on the first observation and read as thrash.

    Booted values, for the record: flip_cost_s 3.2 s gives 6.4 s, which the
    10 s floor raises to 10 s -- so a genuinely draining bundle keeps a full
    decode window, and a stalled one costs at most that.
    """
    return max(10.0, solved_tp_decode_floor_s(cfg))


def with_decode_contention(
    cfg: PhasePolicyConfig, value: object
) -> PhasePolicyConfig:
    """Return ``cfg`` with a new MEASURED decode-contention fraction.

    Lives here rather than in the scheduler's ``set_internal_state`` chain so
    the validation sits with the thing it validates and can be tested without
    standing up a Scheduler. Range checking is not repeated: ``replace`` re-runs
    ``__post_init__``, which already refuses anything outside [0, 1].

    Raises ``PhasePolicyError`` for a value that is not a fraction, so the
    caller can report it and refuse the update rather than half-applying one.
    """
    import dataclasses

    try:
        frac = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise PhasePolicyError(
            f"{ENV_DECODE_CONTENTION} must be a number in [0, 1] -- it is the "
            f"FRACTION of decode throughput lost to a co-resident TP prefill "
            f"-- got {value!r}"
        ) from None
    return dataclasses.replace(cfg, decode_contention=frac)


def _demand_outweighs_a_retry(
    cfg: PhasePolicyConfig,
    inp: PhasePolicyInputs,
    state: PhasePolicyState,
) -> bool:
    """Should a pending backlog override the staging rate limit?

    THE DAMPER THIS REMOVES. A rate limit that postpones a seam the load is
    actively demanding is a damper, not a guard. Live at 12:49-12:51: 70-90k
    tokens sat in TP behind "next staging in 12.4s ... capped at 60s" while
    the prefill crawled at the TP rate. Whatever the odds that the retry
    abandons again, waiting was strictly worse than trying.

    So the limiter yields whenever the flip is worth more than the wait. Both
    sides are seconds and both are measured:

        saved   = N x (1/r_tp - 1/r_pp) - 2C     what landing it is worth now
        waited  = the limiter's current interval

    At 90k on this checkpoint that is ~47 s against ~12.8 s: attempt. At the
    break-even rung it is ~0 s against 11.8 s: let the limiter hold. And under
    strict purity with one token pending -- the #656 storm shape -- saved is
    ~0, which is exactly the case the guard exists for.

    The guard therefore still bounds storms, and can no longer bound progress.
    """
    if not cfg.prefill_runs_in_tp:
        # Purity: a sub-N prompt cannot run in TP at all, so "demand" here
        # carries no time saving to weigh -- this is the #656 shape.
        return False
    tp, pp = float(cfg.tp_prefill_tok_s), float(cfg.pp_prefill_tok_s)
    if tp <= 0 or pp <= tp:
        return False
    saved_s = inp.pending_prefill_tokens * (1.0 / tp - 1.0 / pp) - 2.0 * cfg.flip_cost_s
    k = max(1, state.arm_refusals.get(TP_TO_PP, 1))
    base_s = solved_tp_decode_floor_s(cfg) if cfg.flip_cost_s > 0 else cfg.min_dwell_s
    waited_s = min(base_s * (2**k), cfg.refusal_backoff_cap_s)
    return saved_s > waited_s


def _differential_flip_threshold(
    cfg: PhasePolicyConfig, running_bs: int, base: int
) -> int:
    """The threshold once the counterfactual is priced too (#665-F1).

    The old surcharge compares the stranding a flip causes against a
    stay-in-TP branch assumed to cost the decodes nothing. Measured, that
    branch costs them everything (see DEFAULT_DECODE_CONTENTION). Both
    branches are therefore priced as aggregate delay-seconds, with
    ``sigma = cfg.decode_contention`` the measured fraction of decode
    throughput lost while a prefill is co-resident in TP:

        stay = N/X + B x sigma x N/X
        flip = (C + N/P) + B x (2C + min(W, N/P))

    ``2C`` and not ``C`` because a stranded decode waits out the cutover in
    BOTH directions -- it resumes only once the instance is back in TP -- while
    the prefill only pays the way in. ``min(W, N/P)`` because the PP residency
    is bounded: past ``W x P`` tokens the prefill no longer fits one residency,
    the decodes are released at ``W``, and the remainder waits for the next one.

    ``W`` IS ``stranded_decode_s(cfg)``, the residency that governs -- the
    solved decode-stall cap where one is declared, the hand-set stopwatch where
    it is not, 0 when neither is (#893). It is not ``cfg.pp_window_s``: #889
    established that knob is unreachable under a declared SLO, and this solve
    reads W three times.

    Flipping is justified when ``flip < stay``. Solving for N is linear in
    each of the two regimes ``min()`` selects, so the result is still a single
    scalar threshold that the caller compares against pending tokens:

        N/P <= W :  N > N0 (1-r)(1+2B) / [ (1+B*sigma) - (1+B) r ]
        N/P >  W :  N > N0 (1-r)[(1+2B) + B W / C] / [ (1+B*sigma) - r ]

    with ``r = X/P`` the prefill speed ratio and ``N0`` the seam break-even.
    Both reduce to ``N0`` at ``B = 0``, as they must.

    THE BOUND IS DERIVED, NOT DECREED. At the measured ``sigma = 1`` the first
    regime collapses to ``N0 x (1 + 2B) / (1 + B)`` -- with ``N0`` the
    UNROUNDED break-even ``C / (1/X - 1/P)``, since the solve uses the
    measurements rather than the rounded integer ``flip_tokens`` -- whose
    supremum is exactly ``2 x N0``: with decode stalled either way, even an infinitely busy server
    needs at most twice the break-even backlog, and the factor two is the
    round trip the decodes additionally pay. The threshold still RISES with
    the number of decodes stranded -- the surcharge keeps doing its job of
    delaying a marginal flip -- it just can no longer diverge to a number no
    prompt can reach.

    AND AT sigma = 1 THE RATIO CANCELS. ``(1-r)`` divides out of numerator and
    denominator, so the whole threshold depends only on ``N0`` and ``B`` -- not
    on the prefill ladder. That matters operationally: this rig re-ships
    the serving instance on re-solved memory and KV vectors, and a calibration that
    needed ``r`` re-measured after every such change would be stale the moment
    it landed. sigma = 1 is not a fitted constant either; it is what this
    scheduler does by construction. Batch selection reads ``if new_batch is
    not None: # Run prefill first if possible``, so an iteration with any
    prefill chunk pending runs THAT batch and never reaches the decode
    branch -- absolute prefill priority per iteration, decode getting zero
    steps until the backlog is chunked through. Measured 1.000 at running_bs
    2 and 3, on both the loose and the law-fitted vector, with A-vs-A noise
    floors of 0.4 % and 0.0 %. ``r`` is consulted only for a partial sigma,
    where a deployment that really does interleave would need its own ladder
    anyway.
    """
    tp_tok_s = float(cfg.tp_prefill_tok_s)
    pp_tok_s = float(cfg.pp_prefill_tok_s)
    if tp_tok_s <= 0 or pp_tok_s <= 0 or tp_tok_s >= pp_tok_s:
        # No speedup to solve against; fall back to the seam break-even
        # rather than invent a threshold from an unusable ratio.
        return base
    b = float(running_bs)
    sigma = float(cfg.decode_contention)
    # #819: MEASURED where the seam has reported, seed otherwise. This is the
    # term that made a repriced N inert -- the ladder is solved from C here,
    # not from `base`.
    cost = live_flip_cost_s(cfg) * (1.0 + 2.0 * b)

    # Solved from C, X and P DIRECTLY rather than by substituting
    # N0 = C / (1/X - 1/P) and cancelling. The substitution is only valid
    # while `flip_tokens` is exactly the break-even of THESE three numbers,
    # and it need not be: `config_from_env` derives the default N from the
    # module constant DEFAULT_FLIP_COST_S while `flip_cost_s` itself is
    # overridable, and an operator may pin `flip_tokens` outright. Deriving
    # from the measurements makes the surcharge independent of that, and
    # `base` survives only as the floor it should always have been.
    den_a = (1.0 + b * sigma) / tp_tok_s - (1.0 + b) / pp_tok_s

    # #893: W IS THE RESIDENCY THAT GOVERNS, NOT THE REQUESTED KNOB. All three
    # uses below -- the "is there a bound at all" guard, the regime boundary
    # `N/P <= W`, and regime B's saturation term -- are the same quantity: how
    # long a carried decode waits before it may emit again. Read off
    # `cfg.pp_window_s` that is silently wrong whenever a decode-stall SLO
    # supersedes the stopwatch (#889), and wrong in BOTH directions here: a
    # declared window under an SLO under-charges the stranding, while a CLEARED
    # window under an SLO sent this solve into the branch below, which denies
    # that any bound exists and can answer UNREACHABLE for a phase that is in
    # fact capped at `slo - 2C`.
    window_s = stranded_decode_s(cfg)

    if window_s <= 0.0:
        # NO TIMED BOUND OF ANY KIND -- neither stopwatch nor solved cap. The
        # PP phase is not time-limited, so a carried decode waits out the whole
        # prefill: the stall is 2C + N/P and regime A is the ONLY regime.
        # Regime B must not be reached here -- it models the decode charge
        # SATURATING at W, and with W = 0 that reads as "stranding is free",
        # the opposite of what an absent bound means. Falling through to it
        # produced a threshold that spiked toward the singularity at
        # den_a -> 0+ and then dropped by half a million tokens once den_a
        # went negative.
        if den_a <= 0.0:
            # No positive N satisfies the inequality: with this little
            # contention and this many decodes, flipping does not repay at
            # ANY backlog. That is a real answer, not a failure, and it is
            # reached only when NEITHER bound is declared, so nothing at all
            # limits the stranding. #893: it used to be reachable with a
            # declared SLO too, because this branch was selected off the
            # requested window rather than off the residency that governs --
            # i.e. a phase capped at slo - 2C was priced as uncapped.
            return UNREACHABLE_FLIP_THRESHOLD
        return max(base, int(round(cost / den_a)))

    if den_a > 0.0:
        n_a = cost / den_a
        # Regime A holds only where the prefill fits inside one PP residency.
        if n_a <= window_s * pp_tok_s:
            return max(base, int(round(n_a)))

    # Beyond the window the decode charge saturates at W, which makes this
    # denominator strictly positive (sigma >= 0 and ratio < 1), so the
    # threshold is always solvable and always finite.
    den_b = (1.0 + b * sigma) / tp_tok_s - 1.0 / pp_tok_s
    return max(base, int(round((cost + b * window_s) / den_b)))


@dataclass(frozen=True)
class PhasePolicyConfig:
    """Static configuration. Built once at boot, never mutated."""

    enabled: bool = False
    flip_tokens: int = 0
    min_dwell_s: float = DEFAULT_MIN_DWELL_S
    idle_dwell_s: float = DEFAULT_IDLE_DWELL_S
    rest_state: str = REST_DECODE
    #: Fairness bound; 0 disables it and restores the pure load-triggered
    #: behaviour that starves under a sustained backlog. See the block
    #: comment at DEFAULT_PP_WINDOW_S.
    pp_window_s: float = DEFAULT_PP_WINDOW_S
    tp_decode_floor_s: float = DEFAULT_TP_DECODE_FLOOR_S
    #: W30: can a resident the cutover RETRACTS be re-admitted in the layout
    #: it is flipped into?
    #:
    #: The tp-ward arm's whole justification is "N requests are decoding, so
    #: take them to the decode layout". Under the #856 no-carry rule the
    #: cutover RETRACTS those very requests, so the justification only holds
    #: if the target layout can then re-admit them by read-through. When it
    #: cannot, the arm's own execution destroys the reason it armed -- which
    #: is not a subtle failure: W30 measured 150 flips in 17 minutes, zero
    #: decode batches and zero completed requests, with the arm auditor
    #: calling the verdict wrong 12 times.
    #:
    #: True whenever the purity contract permits the re-admission -- either
    #: the mode allows prefill in TP outright, or the seam-transport
    #: exemption covers it (`phase_purity.seam_transport_exempt`). Static
    #: boot config, therefore identical on every rank, which is what lets the
    #: arm predicate read it without splitting the group.
    seam_readmit_available: bool = True
    #: #689 WINDOW FORMATION. How many completed carriers a PP window should
    #: accumulate before the tp-ward arm is allowed, normally
    #: max_running_requests. 0 or 1 disables the gate entirely and restores
    #: the previous behaviour exactly.
    formation_target: int = 0
    #: The latency bound on that accumulation. A PLAIN TIME CAP, chosen from
    #: this rig's own receipts rather than from an economics term: one flip
    #: EXECUTES in 2459-3186 ms (five FLIP DONE receipts, 2026-08-16), so
    #: spending up to about one flip's worth of waiting to quadruple the
    #: window it opens is obviously worth it, and spending much more is not
    #: obviously anything. #677's economics replaces this number later; until
    #: then it is a bound, not an optimum, and it is written here as such.
    formation_cap_s: float = 3.0
    #: The MEASURED seam round trip (s) that ``flip_tokens`` was derived from.
    #: 0.0 means "not measured here", which disables the stranded-decode
    #: surcharge below and reproduces the previous policy exactly. The
    #: surcharge is gated on a MEASUREMENT rather than a flag on purpose: its
    #: entire justification is that the seam and the window were measured.
    flip_cost_s: float = 0.0
    #: How much of a PP window a stranded decode is charged. 1.0 = the full
    #: window (a paused generation really does wait it out); 0.0 disables the
    #: surcharge while keeping ``flip_cost_s`` available for reporting.
    #:
    #: Consulted only while ``decode_contention`` is 0. Once the
    #: counterfactual is measured there is nothing left for a hand-set weight
    #: to express -- the stranding is priced against what NOT flipping costs
    #: the same decodes.
    decode_strand_weight: float = 1.0
    #: MEASURED fraction of decode throughput lost while a prefill is
    #: co-resident in the TP layout, in [0, 1]. 0.0 = "not measured here" and
    #: the old one-sided surcharge is used unchanged. See the block comment at
    #: DEFAULT_DECODE_CONTENTION for why the un-measured default of zero made
    #: the gate unreachable, and _differential_flip_threshold for the model
    #: this feeds.
    decode_contention: float = DEFAULT_DECODE_CONTENTION
    #: Max seconds a carried decode may be stalled by a cutover, DECLARED as a
    #: latency constraint. The PP residency cap is solved from it; see
    #: ENV_DECODE_STALL_SLO. 0 = not declared, drain governs alone.
    decode_stall_slo_s: float = DEFAULT_DECODE_STALL_SLO_S
    #: Drained threshold for the PP phase, in tokens. Set from the scheduler's
    #: real chunked_prefill_size at boot; see ENV_PP_EXIT_TOKENS for why this
    #: is emphatically not flip_tokens.
    #: #677 hot fix 2: the user's target semantics -- prefill until empty,
    #: decode the bundle TO COMPLETION, then prefill again. Off by default, and
    #: every rule it gates is byte-identical to the previous behaviour until it
    #: is set, so no existing deployment moves.
    drain_mode: bool = False
    #: #856 STRICT PHASE BATCHING (user directive 2026-08-24). All pending
    #: prefill is processed in PP, all decode in TP, and NEVER any work in the
    #: wrong layout -- "egal wie lang der flip dauert". The trigger is then
    #: DRAIN-AND-FLIP and nothing else: the break-even band answers "is this
    #: backlog worth the seam?", which is precisely the question this mode does
    #: not ask, because the alternative to flipping is running prefill in the
    #: decode layout and the mode forbids that outright.
    #:
    #: AN EXTENSION OF ``drain_mode``, NOT A RIVAL TO IT: the exit it relies on
    #: lives in the drain block, and ``--phase-flip-policy`` is left alone
    #: because that enum selects the manual-vs-auto ENGINE, not the arming
    #: rule. Off by default; the economic mode stays byte-identical for every
    #: other workload.
    drain_mode_strict: bool = False
    pp_exit_tokens: int = DEFAULT_PP_EXIT_TOKENS
    #: The scheduler's chunked_prefill_size, filled in at boot. Drives the
    #: seam staging estimate below.
    prefill_chunk_tokens: int = 0
    #: The prefill ladder the threshold is solved against. Only the RATIO
    #: matters to the differential model; ``flip_tokens`` already carries the
    #: absolute scale.
    tp_prefill_tok_s: float = DEFAULT_TP_PREFILL_TOK_S
    pp_prefill_tok_s: float = DEFAULT_PP_PREFILL_TOK_S
    #: How long a REFUSED arm holds the policy off that direction, doubling
    #: per consecutive refusal from ``min_dwell_s`` and clamped here.
    #:
    #: WHY A REFUSAL NEEDS ITS OWN CLOCK AT ALL. ``min_dwell_s`` bounds
    #: THRASH -- how soon after a flip the next one may happen -- and a
    #: refused arm is not a flip, so it must not reset that clock (a refusal
    #: moved no request and changed no layout). But without a clock of its
    #: own the policy would then re-arm on the very next round, which is
    #: worse than the defect this replaces. So the refusal gets a clock
    #: whose growth matches what a repeating refusal means: the seam is
    #: short by a CONFIGURATION-determined amount, and retrying an
    #: unfundable configuration cannot fund it.
    refusal_backoff_cap_s: float = DEFAULT_REFUSAL_BACKOFF_CAP_S
    #: Consecutive refusals in one direction before the hold is announced as
    #: a degradation. Mirrors the runtime's ``SEAM_ABANDON_CAP`` on purpose:
    #: that cap is the point at which the SEAM has decided the ask cannot be
    #: met, and this is the point at which the POLICY says so out loud.
    refusal_degrade_after: int = DEFAULT_REFUSAL_DEGRADE_AFTER
    #: False when STRICT PHASE PURITY forbids prefill in the TP layout.
    #:
    #: THIS COLLAPSES N TO ZERO, and it must. N is a BREAK-EVEN: it
    #: compares running a prefill of n tokens in TP (n/X) against flipping
    #: first (C + n/P). That comparison presupposes the TP option EXISTS.
    #: Under purity it does not -- prefill in TP is refused by the
    #: scheduler gate -- so "too small to be worth a flip" becomes "too
    #: small to ever run", and the request waits forever.
    #:
    #: METAL, 2026-08-09 21:39:50Z, the first purity boot: a single
    #: health-check prompt arrived while the instance sat in TP with
    #: nothing decoding. "PHASE-POLICY holding in tp: pending prefill 1 tok
    #: <= N=7004, running it in tp (running bs 0)" -- and purity refused to
    #: run it. The server was alive, idle, and permanently unable to answer
    #: a one-token prompt. Under purity ANY pending prefill must move the
    #: instance to PP, because PP is the only place it can happen.
    prefill_runs_in_tp: bool = True

    def __post_init__(self) -> None:
        # #856: strict batching EXTENDS drain; it does not replace it. The exit
        # strict relies on lives in the drain block, so running strict with
        # drain off would gate the economic band away and then fall through to
        # the very economics the mode exists to remove -- a half-applied mode
        # that looks configured and behaves like neither.
        if self.drain_mode_strict and not self.drain_mode:
            raise PhasePolicyError(
                "phase policy strict batching requires drain mode: strict is an "
                "extension of the drain exit, not an alternative to it. Set "
                "--phase-policy-drain-mode (or drop --phase-policy-drain-mode-strict)."
            )
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
        if self.pp_window_s < 0.0:
            # 0 means "no fairness window" and is a supported configuration;
            # a NEGATIVE window is not a weaker version of that, it is a typo
            # that would flow into the stranding charge as negative seconds.
            raise PhasePolicyError(
                f"{ENV_PP_WINDOW}={self.pp_window_s!r} is negative; use 0 to "
                f"disable the fairness window, or a positive number of seconds"
            )
        if not 0.0 <= self.decode_contention <= 1.0:
            raise PhasePolicyError(
                f"{ENV_DECODE_CONTENTION}={self.decode_contention!r} is not a "
                f"fraction of decode throughput; it must be in [0, 1], where "
                f"1 means decode stops dead while a prefill shares the TP "
                f"batch (measured on this rig) and 0 means it is unaffected"
            )

    @property
    def rest_phase(self) -> str:
        return _PHASE_OF_REST[self.rest_state]


@dataclass
class PhasePolicyState:
    """The only mutable state. Timestamps are on the caller's clock."""

    last_flip_at: float = 0.0
    idle_since: Optional[float] = None
    #: #748 REFAIL: when the CURRENT layout was first observed unable to build
    #: a batch of either class, on the caller's clock. A SEPARATE clock from
    #: ``idle_since`` because the two measure different states and only one of
    #: them exists during an idle lock: ``idle_since`` needs an EMPTY box
    #: (``running_bs == 0 and pending == 0``), while an idle lock is precisely
    #: a box with work it cannot run. #759 keyed its persistence qualifier on
    #: ``idle_since`` and was therefore inert on the path it was written for --
    #: 160 of 160 armings in boot_735_nohc.log took its "unobserved" branch.
    #: None means NOT YET OBSERVED and is read as "no evidence of transience",
    #: which keeps #689's "a layout that can run NOTHING leaves at once".
    #: Maintained by ``observe_idle`` only -- it is an observation, never a
    #: verdict, so ``note_flip_armed`` deliberately does not clear it.
    nothing_can_run_since: Optional[float] = None
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
    #: Consecutive REFUSED arms per direction, reset by the first success.
    arm_refusals: dict = field(default_factory=dict)
    #: Clock, per direction, before which no further arm may be issued.
    arm_hold_until: dict = field(default_factory=dict)
    #: Per direction, the length of the hold currently in force. Only used to
    #: recover when that hold STARTED; see `_last_refusal_at`.
    arm_hold_last_span: dict = field(default_factory=dict)
    #: When a staging attempt last ABANDONED, per direction. The storm guard
    #: is paced off this and nothing else -- see the rate-limiter block in
    #: `decide`. Cleared by a completion, so a funded path never carries a
    #: penalty from an earlier failure.
    last_abandon_at: dict = field(default_factory=dict)
    #: The runtime's own words for a direction that has been refused past
    #: ``refusal_degrade_after``. Truthy = degraded; cleared by a success.
    arm_degraded: dict = field(default_factory=dict)
    #: The reason the LAST arm of each direction was abandoned, kept whether
    #: or not the degrade threshold was reached. The purity valve prints it
    #: when it stands down, so the receipt names the SHORTFALL that idled the
    #: instance ("staging 1614 MiB needed but only 1595 MiB is spendable")
    #: instead of only the streak count, which says nothing about the cause.
    arm_last_detail: dict = field(default_factory=dict)
    #: What ``note_flip_armed`` overwrote, so a refusal can put it back.
    #: A tuple (direction, previous_last_flip_at, previous_idle_since); None
    #: when no arm is outstanding.
    pending_arm: Optional[tuple] = None
    #: Metrics. ``arm_refusals_total`` counts every refused arm ever;
    #: ``arm_degrade_events`` counts DEGRADATIONS, not refusals, so it stays
    #: readable as "how many times did this instance stop being able to
    #: flip" rather than as a retry tally.
    arm_refusals_total: int = 0
    arm_degrade_events: int = 0
    #: #677(a): the last tick at which prefill DEMONSTRABLY moved -- pending
    #: strictly decreased. Maintained by ``observe_idle`` so ``decide`` stays
    #: pure, the same split the idle clock uses. None means not yet observed,
    #: which reads as "inapplicable" rather than "stalled since t=0": a caller
    #: that reaches ``decide`` without observing must degrade to the previous
    #: behaviour, never to an immediate exit.
    last_prefill_progress_at: Optional[float] = None
    #: What pending was at the previous observation, so a DECREASE can be
    #: recognised. Compared, never trusted as a level.
    last_pending_prefill_tokens: Optional[int] = None
    #: #833: when the ADMITTED SET last got smaller, and what it was.
    #:
    #: The #677(a) pair above measures PENDING, and that is the wrong axis for
    #: drain mode: pending is the quantity PRESSURE inflates, so under load it
    #: keeps moving and always "reads as progress" -- the failure this module
    #: already records at the `nothing_can_run` rule ("it needs pending FROZEN,
    #: and pending was still creeping ... which reads as progress"). Drain mode
    #: waits on the admitted set, so the admitted set is what has to be watched
    #: for a stall. Same shape, honest axis.
    last_bundle_progress_at: Optional[float] = None
    #: The previous observation's ``running_bs``. Compared, never a level.
    last_running_bs: Optional[int] = None
    #: #677 hot fix 2: how many requests were decoding when this TP window
    #: opened, so the exit receipt can name the BUNDLE it finished rather than
    #: only the instant it ended. Set by ``observe_idle`` at the phase change.
    bundle_at_phase_entry: int = 0
    #: Cutovers that actually MOVED BYTES. Distinct from ``flips_armed`` on
    #: purpose: boot E armed 179 flips and completed none, and every summary
    #: that read the arm count called that instance healthy.
    flips_completed: int = 0
    #: When the current PP window began accumulating carriers, or 0.0 when it
    #: is not accumulating. Stamped by ``observe_idle`` (the caller), because
    #: ``decide`` is pure and may not mutate state.
    formation_started_at: float = 0.0
    #: #677 hold lifecycle. Carried on the STATE, and reset through
    #: next_hold_rounds on every phase change / pend->0, because a
    #: caller-maintained counter that goes stale would start the next
    #: episode's first hold half-exhausted.
    hold_rounds: int = 0
    hold_phase: str = ""


#: #861f: decode steps a TP phase owes a live bundle before a flip may pull the
#: layout away. Measured in COMPLETED STEPS, not seconds: a seconds floor
#: cannot distinguish a fast bundle from a stalled one, and W37-D/d4 produced
#: exactly one token per flip cycle while every seconds-based guard was happy.
#: Small on purpose -- this is an anti-chop floor, not a scheduling quantum.
MIN_DECODE_STEPS_PER_PHASE = 8


@dataclass(frozen=True)
class PhasePolicyInputs:
    """A replicated snapshot of the load. Every field MUST be identical on
    every rank -- see the module docstring on replication."""

    phase: str
    pending_prefill_tokens: int
    running_bs: int
    now: float

    #: W32: tokens of requests the #856 cutover retracted and RE-ADMITTED,
    #: awaiting a read-through in the layout the flip just entered. They are
    #: already excluded from `pending_prefill_tokens` (that is PP work, and
    #: these are not) -- this field exists so the decode-empty rule can tell
    #: "the bundle never arrived" from "the bundle is one round away".
    #: Replicated: the cutover is group-unanimous and the queue it re-admits
    #: into is replicated, so every rank computes the same number.
    seam_transport_tokens: int = 0

    #: #861j: the SERVICEABLE-HERE subset of ``seam_transport_tokens`` -- the
    #: tokens of stamped re-admissions that the purity gate's exemption WILL
    #: admit in the current (TP) layout this round: premise verified on the
    #: retract credit, phase is TP, and the transport-debt clock has not
    #: lapsed. Supplied by the build site; 0 on every stand-in and in the PP
    #: layout, so an unsupplied field reproduces the pre-#861j behaviour
    #: exactly.
    #:
    #: WHY A SECOND FIELD AND NOT A CONSUMER OF THE FIRST: W37-F's Door 1 was
    #: the W32 exclusion and the #861c existence term CLASSIFYING THE SAME
    #: SEVEN REQUESTS OPPOSITELY -- transport-serviceable-here vs PP demand --
    #: and the demand side winning by branch order, so the tp-ward arm undid
    #: its own flip within one round, forever (21+21 flips, zero decode
    #: batches, three boots). The demand term below subtracts exactly this
    #: field, so ONE side claims the population and the other releases it --
    #: the #861g fix principle, applied to the pair that broke it.
    #: Replicated: derived from the replicated queue and the group-unanimous
    #: stamp, same argument as ``seam_transport_tokens``.
    seam_serviceable_tokens: int = 0

    #: #861c: PROMPT TOKENS OWED A PREFILL PASS, CACHED OR NOT.
    #:
    #: THE SECOND SEMANTICS, and the field exists because ONE number was being
    #: asked TWO questions. ``pending_prefill_tokens`` sums
    #: ``uncached_prompt_tokens`` and is right for the ECONOMICS question --
    #: the break-even compares PP against TP prefill throughputs, and a cached
    #: token is read at a layout-independent cost, so it cancels. It is wrong
    #: for the EXISTENCE question, because a fully cached prompt still needs an
    #: extend pass to place its KV in the device pool and enter the running
    #: batch.
    #:
    #: W37-C measured the cost of conflating them: six requests queued, every
    #: token prefetched into HiCache, ``pending_prefill_tokens=0``, so the
    #: policy read "no prefill work", never demanded the flip to PP, strict
    #: purity correctly refused the pass in TP -- 18 flips, ZERO completions,
    #: 0 % GPU, pool `avail=468981`.
    #:
    #: THE SPLIT IS BY DECISION CLASS, and the classification is pinned by
    #: `test_counter_semantics_861c.py` so a future consumer cannot pick the
    #: wrong one silently:
    #:   ECONOMICS  (break-even N, price bands, LAYOUT-ECONOMY holds, the #819
    #:              price line, the #838 economy check) -> pending_prefill_tokens
    #:   EXISTENCE  (idle determination, drain exits, "is there work at all",
    #:              admission simulation) -> admissible_prefill_tokens
    #:
    #: Replicated by the same argument as every other field: it is a pure
    #: function of the replicated waiting queue.
    #:
    #: DEFAULTED so every stand-in and every older construction site keeps
    #: working; a caller that does not supply it gets 0, and the consumers
    #: below take `max(...)` with the economics number, so an unsupplied field
    #: can only reproduce today's behaviour and never invent new work.
    admissible_prefill_tokens: int = 0

    #: #861e: decode work the cutover RETRACTED but that is not FINISHED --
    #: requests with output tokens already produced, sitting back in the queue
    #: waiting to resume. Supplied by the build site; 0 on every stand-in and
    #: on every non-flip deployment, so this field can only ever ADD decode
    #: work that genuinely exists.
    retracted_unfinished_bs: int = 0

    #: #861f: decode steps COMPLETED since this TP phase began. The d4-thrash
    #: protection, measured in work actually done rather than in wall seconds
    #: (a seconds floor cannot tell a fast bundle from a stalled one) and
    #: rather than in "requests that exist" (which is what deadlocked W37-E).
    #: 0 on every stand-in, which reproduces the pre-#861f behaviour exactly.
    decode_steps_this_phase: int = 0

    #: #869: MAY A DECODE STEP EXECUTE IN THE CURRENT LAYOUT AT ALL?
    #:
    #: The phase axis of ``decode_steps_this_phase``, and the reason the
    #: counter alone cannot answer the question its consumer asks. Supplied by
    #: the build site from the purity authority (``decode_allowed_in_pp``),
    #: which is the one place that knows; see ``bundle_is_mid_flight`` for what
    #: goes wrong without it.
    #:
    #: DEFAULTS TRUE, which reproduces the pre-#869 behaviour exactly on every
    #: stand-in and on every deployment without phase purity -- where decode
    #: does run in whatever layout is up, so the default is not merely
    #: compatible but correct.
    #:
    #: Replicated like every other field here: purity is parsed once from the
    #: server args at boot and the phase is group-unanimous, so every rank
    #: computes the same boolean.
    decode_runs_in_this_phase: bool = True

    def bundle_is_mid_flight(self) -> bool:
        """Is a decode bundle running that a flip would chop? #861f.

        REPLACES the #861e formulation that deadlocked. The question is not
        "does decode work exist somewhere" -- a retracted request in the queue
        is not decoding and answering yes for it is what wedged W37-E. The
        question is whether THIS TP phase has a live bundle that has not yet
        had a fair share of steps.

        d4 (thrash): residents decoding, few steps done -> True, hold.
        W37-E (wedge): nothing resident, 7 queued retracted -> False, flip.
        """
        # #869: A BUNDLE CANNOT BE MID-FLIGHT IN A LAYOUT THAT CANNOT FLY IT.
        #
        # THE PHASE AXIS, which none of #861e/f/i checked. `decode_steps_this_
        # phase` is produced by exactly one site -- the decode forward funnel
        # in `scheduler.run_batch`, under `batch.forward_mode.is_decode()` --
        # and is reset to 0 at every cutover (`phase_flip_runtime`, "#861i: the
        # step budget is PER PHASE"). Under strict purity decode is FORBIDDEN
        # in the PP layout (`PhasePurity.decode_allowed_in_pp` -> False), so
        # across the whole PP phase the counter is 0 BY CONSTRUCTION -- never
        # because a bundle is young, which is the only thing the floor below
        # can read it as.
        #
        # Read in PP, the test therefore degenerates to `running_bs > 0`:
        # permanently True whenever any resident exists, and unable to become
        # False through progress, because the progress it waits for is the one
        # kind of work that layout may not do. `demand_prefill_tokens()` then
        # short-circuits to 0, and `_strict_holds_pp` -- the #861d-2 guard
        # whose entire stated job is that "the two directions cannot disagree
        # about whether work exists" -- is disarmed in the very phase it
        # guards.
        #
        # THE CLASS: a predicate evaluated in both phases whose governing input
        # can only be produced in one of them is not a measurement in the other
        # -- it is a constant wearing a measurement's name. The anti-chop floor
        # is meaningful exactly where decode steps can accrue; where they
        # cannot there is no bundle in flight to chop, and saying otherwise
        # suppresses the demand that would send the work to the layout that can
        # serve it.
        #
        # Third generation of this same term (#861e counted retracted requests
        # and deadlocked W37-E, #861f replaced the measure, #861i wired the
        # producer after finding it written nowhere). Each fix moved the defect
        # along one axis; this one closes the axis they shared.
        if not self.decode_runs_in_this_phase:
            return False
        if int(self.running_bs or 0) <= 0:
            return False
        return int(self.decode_steps_this_phase or 0) < MIN_DECODE_STEPS_PER_PHASE

    def decode_work_bs(self) -> int:
        """Decode work that EXISTS, not decode work currently installed. #861e.

        THE MANUFACTURED-STATE CLASS, and this is its one authority.

        ``running_bs`` is the size of the running batch, and the #856 cutover
        EMPTIES that batch as its second step -- it retracts every resident and
        puts them back at the front of the queue. So in the round after a
        cutover ``running_bs == 0`` is true, and it is true BECAUSE of the
        transition, not because the work is gone. W37-D/d4, verbatim, four
        consecutive log lines::

            arming pp_to_tp: DRAINED ... 2 req decoding
            RESIDENTS RELEASED for pp_to_tp: 7 request(s) retracted
            SEAM RE-ADMISSION: 7 put back at the FRONT of the waiting queue
            arming tp_to_pp: pending prefill 18586 tok > 0 (... nothing decoding)

        "Nothing decoding" one line after "7 retracted". Every policy term that
        FIRES BECAUSE ``running_bs == 0`` therefore fires on a state the seam
        produced: the demand term (mine), the idle determination, the starved
        dwell-bypass, and both flip-threshold shortcuts. Terms gated
        ``running_bs > 0`` are unaffected -- a manufactured 0 makes them not
        fire, which is the safe direction.

        THE COHERENT READ: a retracted request that has already produced output
        tokens is UNFINISHED DECODE WORK. It is not queued prefill and it is not
        idle; it is a bundle mid-flight that the seam parked for one round.

        THE DISCRIMINATOR IS "HAS IT PRODUCED OUTPUT", and it is what separates
        the two metal specimens that any fix here must satisfy at once:

          d2 wedge   7 queued, none ever started, 0 output tokens
                     -> decode_work_bs 0 -> the demand FIRES (correct: nothing
                        is served by staying)
          d4 thrash  7 retracted with n=2..13 output tokens each
                     -> decode_work_bs 7 -> the demand is SILENT (correct: the
                        bundle this TP window exists to finish is mid-flight)

        Per-rid evidence that these really are mid-flight and not restarts:
        rid be636087 across epochs read n=2,3,4,5,...,13 -- strictly
        increasing, never re-bootstrapped (#775 refuted on this boot).
        """
        # #861f ROOT FIX: GENUINELY RESIDENT DECODING ONLY.
        #
        # W37-E proved the #861e formulation wrong by deadlock. Adding
        # `retracted_unfinished_bs` here made a RETRACTED request count as
        # decode work, so the demand term stayed silent for requests that were
        # not decoding at all -- they were sitting in the waiting queue needing
        # a prefill pass, which TP may not run. Nothing flipped, nothing
        # prefilled, GPU 0 % for 198 s with 7 queued.
        #
        # THE CLEAN ROOT: a retracted-unfinished request IS NOT DECODE WORK.
        # It is PREFILL work waiting for the pp layout -- it needs a pass to be
        # re-materialised before it can decode again. So it belongs on the
        # DEMAND side, and counting it here was the category error.
        #
        # `retracted_unfinished_bs` is KEPT on the input (the seam still
        # reports it, and the d4 thrash pin still reads it) but it no longer
        # suppresses the flip. What protects a live bundle from being chopped
        # is `decode_steps_this_phase` below -- a floor measured in COMPLETED
        # DECODE STEPS rather than in "requests that exist somewhere", which is
        # what d4 actually needed and what this field was standing in for.
        return int(self.running_bs or 0)

    def demand_prefill_tokens(self) -> int:
        """Tokens of UNSTARTED queued prefill that justify DEMANDING a flip.

        #861d-2, and it is a narrower question than ``work_exists()``.

        THE TWO SPECIMENS THIS MUST SATISFY AT ONCE, both from tonight:

          d2  7 queued, **0 running**, GPU 0 %, no first token for 589 s.
              The demand MUST fire: nothing is decoding, so nothing is served
              by staying, and the queued work can only run in the other layout.
          d3  queued seam-transport re-admissions and **2 decoding**. The
              demand must NOT fire: 18 armings chopped decode mid-bundle,
              epochs 13/14 in six minutes, COMPLETIONS 0. The bundle this TP
              window exists to finish was destroyed by the arm meant to help
              the queue.

        The discriminator is therefore ``running_bs``, not the token count:
        DRAIN, THEN FLIP. While decode is in flight the queue waits -- that is
        the user's law, not a compromise with it ("aller Decode in TP, dann
        aller Prefill in PP"). With nothing decoding, any queued prefill work
        demands the layout that can run it, whatever its cached status.

        This is also W30's own rule read forward: AN ARM MAY NOT DESTROY ITS
        OWN JUSTIFICATION. An arm justified by "the queue is starving" that
        interrupts the decode which would empty the running set destroys
        exactly the progress it claims to want.

        Returns 0 while decode is in flight, so the caller's verdict and its
        message can both be this single number (#713).
        """
        # #861f: hold only for a bundle that is genuinely MID-FLIGHT. A
        # retracted request in the queue is prefill work, not decode work, and
        # treating it as decode work is what deadlocked W37-E.
        if self.bundle_is_mid_flight():
            return 0
        # #861j DOOR 1: work the CURRENT layout will serve this round is not
        # demand for the OTHER layout. The admissible term deliberately counts
        # every stamped re-admission (a seam-retracted request owes a full
        # pass); but when the purity gate's exemption will admit that pass
        # HERE (premise verified, debt clock live -- the build site's
        # ``seam_serviceable_tokens``), demanding a flip for it makes the
        # tp-ward flip undo itself: the W37-F oscillation, measured three
        # boots in a row. Subtracted from the ADMISSIBLE term only -- the
        # economics number was already corrected at the input boundary (W32)
        # -- and floored at 0, so fresh unstamped work keeps demanding
        # undiminished.
        return max(
            int(self.pending_prefill_tokens or 0),
            max(
                0,
                int(self.admissible_prefill_tokens or 0)
                - int(self.seam_serviceable_tokens or 0),
            ),
        )

    def work_exists(self) -> bool:
        """Is a prefill pass owed SOMEWHERE? The existence question, once.

        One method rather than ``max(a, b) > 0`` at each site: the whole defect
        was two questions sharing one expression, and the fix is worth nothing
        if the next consumer re-derives it a third way.
        """
        return (
            max(
                int(self.pending_prefill_tokens or 0),
                int(self.admissible_prefill_tokens or 0),
            )
            > 0
        )

    #: THE ROUND JUST FAILED TO BUILD A BATCH OF EITHER WORK CLASS.
    #:
    #: Not "the queue is empty" and not "the box looks quiet": both classes
    #: had work and neither could be scheduled in the layout that is running.
    #: Measured 2026-08-16 09:42:39-45, six seconds of total silence with
    #: 572715 tok of prefill queued and four carriers ready to decode --
    #: decode forbidden in PP, and prefill unadmittable because those same
    #: four carriers' KV held ~352k of the 472k-row pool. Nothing could run,
    #: and the only rule that armed the flip out of it was the 180 s
    #: decode-stall cap.
    #:
    #: REPLICATED like every other field here: it is derived from the round's
    #: own build outcome, which every rank reaches by running the same
    #: scheduler over the same replicated queue.
    nothing_can_run: bool = False

    #: The OTHER layout can build at least one of those classes.
    #:
    #: This is what makes the arm ONE-SIDED and therefore non-oscillating by
    #: construction, with no timer floor: the flip is armed only when the
    #: current layout provably cannot run anything AND the target provably
    #: can, so after the flip the target runs by premise and the same
    #: condition cannot immediately hold again in the new layout.
    target_can_admit: bool = False

    #: #689: completed carriers waiting for a decode window. NOT running_bs --
    #: with #677 phase-1 parking live, carriers PP cannot decode are
    #: deliberately discounted from the admission cap, so the two numbers have
    #: diverged and a formation rule keyed on running_bs would read 0 exactly
    #: when the window is fullest.
    ready_carriers: int = 0
    #: Whether anything is still queued behind them. When the queue is empty
    #: there is nothing left to accumulate and the window opens at once.
    queue_nonempty: bool = False

    #: #708: KV rows available, RANK-UNIFORM, so the BOTH-BLOCKED decline can
    #: NAME its binding resource from a measurement instead of asserting it.
    #:
    #: The decline used to state "the binding resource is KV" unconditionally.
    #: It happened to be right on all three live specimens (19,004 avail vs
    #: 97,922 pending; 107,881 vs 160,514; 7,085 vs 217,048 -- available well
    #: under pending every time), but a claim that cannot come out the other
    #: way is not a diagnosis, and it would have said exactly the same thing
    #: with a pool standing empty.
    #:
    #: MUST BE THE GROUP MIN, not this rank's own pool. Every field here is
    #: replicated by contract, and under uneven DCP the local availability
    #: differs per rank -- feeding a local value in would make the decline text
    #: (and any rule keyed on it) rank-dependent, which is the #616g divergence
    #: this codebase already pays to avoid. `uniform_avail_for_evict` is the
    #: existing accessor for exactly this quantity.
    #:
    #: None = not measured. The decline then says so rather than guessing.
    kv_available_tokens: Optional[int] = None


@dataclass(frozen=True)
class PhasePolicyDecision:
    direction: Optional[str]
    reason: str

    #: #817 THE HOLD ALLOWLIST, CARRIED INSTEAD OF SNIFFED.
    #:
    #: The #677 hold may veto exactly one kind of arm: the plain timer /
    #: economics exit, the one whose premise the hold contradicts ("prefill is
    #: unserved here, so do not let a clock take the layout away"). Every other
    #: arm is chosen by a RULE for a reason the hold cannot see, and vetoing
    #: those is how a legitimate exit becomes a wedge.
    #:
    #: That membership used to be decided by reading the reason STRING -- a
    #: denylist of three substrings, with everything unrecognised silently
    #: swallowed. The blocked-admission exit was unrecognised, so the exit
    #: built to end a live wedge (403779 tok frozen, every slot held by a
    #: carried decode) was itself converted into a hold that printed "the timer
    #: does not get to take it away" about an exit that is not a timer. The
    #: wrapper's own comment prescribed the remedy: "If a fourth exemption ever
    #: appears, invert this into an allowlist rather than adding it."
    #:
    #: So eligibility is now a PROPERTY OF THE ARM, set where the arm is built
    #: and by the code that knows what it means. The default is False, which
    #: makes the safe direction the structural one: an arm nobody marked is an
    #: EXIT. A future arm added without reading this comment cannot be
    #: swallowed by forgetting to name it -- only by explicitly claiming to be
    #: eligible.
    #:
    #: THE ADMISSION CONDITION, and it is the whole rule:
    #:
    #:     An arm may carry hold_eligible=True only if, in EVERY state where
    #:     that arm fires, a SECOND INDEPENDENT anti-starvation bound is armed.
    #:
    #: The hold's permission to veto "the plain timer/economics exit" was never
    #: about the arm being a timer. It rested on the unstated assumption that
    #: something else would still stop the layout from pinning. Veto the last
    #: bound and the hold is unbounded, which is the exact condition the
    #: starvation regression tests exist to prevent and the shape of the live
    #: wedge family. The assumption outranks the prose.
    #:
    #: THE LIST IS EMPTY TODAY, and that is the honest state rather than an
    #: oversight. The one arm that looked eligible -- the legacy pp_window
    #: stopwatch -- sits behind a `cap <= 0` guard, so it fires ONLY when the
    #: decode-starvation cap is absent and it is therefore the last bound in
    #: every state it fires in. It fails the condition by construction.
    #:
    #: THE SEAM STAYS ANYWAY. It is the socket for a future arm that really is
    #: backstopped, and keeping it means such an arm is added by stating the
    #: claim rather than by re-deriving this whole argument. It also keeps the
    #: swallow structurally impossible in the meantime: with no member, no arm
    #: can be held at all.
    #:
    #: WHERE #677's ECONOMICS ACTUALLY LIVES, so nobody re-wires it here: in
    #: the window-length machinery, and in the threshold repricing (#819). It
    #: belongs on the flip-DECISION side, where it can weigh a flip before one
    #: is chosen -- not as a veto on an exit the rules already decided.
    hold_eligible: bool = False

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
    """The load rules, then batch formation. Pure, mutates nothing.

    #689 WINDOW FORMATION IS APPLIED HERE, AS A WRAPPER, so it sits at ONE
    place instead of at the nine ``return PhasePolicyDecision`` sites inside
    the rules. It only ever converts a pp_to_tp arm into a wait; it can never
    create an arm, and it never sees the tp_to_pp direction at all.
    """
    d = _decide_rules(cfg, state, inp)
    # #677 HOLD, APPLIED AS A WRAPPER for the same reason #689 is: one place
    # instead of the nine return sites inside the rules. It only ever converts
    # an ARM into a wait -- it can never create one -- and its reason string is
    # what the boot's acceptance is read from, so it must reach the log exactly
    # as the rules' own reasons do.
    # THE DECODE STARVATION CAP OUTRANKS THIS ABSOLUTELY. That SLO is the
    # system's guarantee that PP can never pin the server, and my round bound
    # is only a secondary backstop -- vetoing the cap would put a local timer
    # above a global guarantee and reintroduce the pinning it prevents.
    #
    # THREE EXEMPTIONS IS A SHAPE, NOT A COINCIDENCE: idle-locked, drained and
    # the starvation cap are all arms the RULES chose for reasons this lever
    # cannot see. The lever only understands "prefill is unserved here", so it
    # may only veto the arm that contradicts that -- the plain timer/economics
    # exit. If a fourth exemption ever appears, invert this into an allowlist
    # rather than adding it.
    #
    # #817 DID THE INVERSION, and the fourth case that forced it was not
    # hypothetical: the blocked-admission exit (#677's own, one day older than
    # this wrapper) was never in the denylist, so it was swallowed. It is not a
    # timer -- its commit spends a paragraph on exactly that distinction,
    # because the wedge it ends is the one no clock was watching -- yet the
    # hold it became announced "the timer does not get to take it away" over
    # the live specimen's own 403779 frozen tokens. A denylist of substrings
    # cannot tell an unrecognised arm from an exempt one, and it fails toward
    # swallowing.
    #
    # The membership test is now `d.hold_eligible`, set at the arm. This
    # paragraph is kept because it states the RULE the allowlist encodes, and
    # the rule outlives its implementation: the lever may veto the plain timer
    # exit and nothing else.
    #
    # THE DRAINED EXIT ALSO OUTRANKS THIS, on #669's economics. A residual
    # ABOVE one chunk stays in PP -- which is what this hold wants anyway --
    # but a SUB-CHUNK residual is finishing regardless, and holding the layout
    # for it would spend a ~2.6 s seam to save a fraction of a chunk. #669
    # moved the anti-pinning guarantee to the starvation cap precisely so the
    # drain could exit cleanly; vetoing it here would re-pin the server in PP,
    # which is the defect that contract was written to prevent.
    #
    # PP_TO_TP ONLY, exactly as #689's formation gate is. The rules arm for
    # reasons this lever knows nothing about -- the idle return leg, the
    # decode-stall SLO -- and a gate that vetoed every arm because MY rule did
    # not recognise it would suppress all of them. The C2 defect is
    # specifically "pp_to_tp took the layout while prefill was unserved", so
    # that is the only arm this converts into a wait. It can never create one.
    if d.direction == PP_TO_TP and d.hold_eligible:
        # #688 OUTRANKS #677 EXACTLY AS IT OUTRANKS #689, and for the same
        # reason spelled out below for formation: the idle-locked arm fires
        # only when the current layout can build NOTHING. Holding that layout
        # to serve "pending work" is waiting inside a layout that cannot serve
        # it -- the zero-GPU window #688 exists to remove. The deadlock escape
        # is never held.
        # #820 REMOVED TWO ARGUMENTS FROM THIS CALL, and what they were is the
        # tell. `decode_serving` and `decode_rounds_so_far` were read ONLY by
        # the verdict's "tp" branch -- a branch this call site can never reach,
        # because every arm that can carry PP_TO_TP is built under
        # `inp.phase == PHASE_PP`. A call site feeding arguments its own phase
        # cannot consume is the fingerprint of a built-never-wired seam.
        # `decode_rounds_so_far` was worse than unused: it passed `hold_rounds`,
        # the HOLD counter, as a count of DECODE rounds. The caller never had
        # the quantity that branch needed, so wiring it would have been wrong on
        # day one.
        allow, hold_why = layout_hold_verdict(
            phase=inp.phase,
            prefill_pending_tokens=int(getattr(inp, "pending_prefill_tokens", 0) or 0),
            decode_waiting=int(getattr(inp, "ready_carriers", 0) or 0),
            hold_rounds_so_far=int(getattr(state, "hold_rounds", 0) or 0),
            seam_funded=True,
            mid_flip=False,
        )
        if not allow:
            return PhasePolicyDecision(direction=None, reason=hold_why)
    if d.direction != PP_TO_TP:
        return d
    if (d.reason or "").startswith(IDLE_LOCKED):
        # #688 OUTRANKS #689, STRUCTURALLY. The idle-locked arm fires only
        # when the current layout can build NOTHING; holding it to accumulate
        # carriers would be waiting for a wider window inside a layout that
        # cannot fill one, which is the exact zero-GPU window #688 removes.
        # Caught by test_the_deadlock_escape_is_not_delayed_by_formation,
        # which failed on the first draft of this wrapper.
        return d
    return _apply_formation_gate(cfg, state, inp, d)


def _formation_target(cfg: PhasePolicyConfig, inp: PhasePolicyInputs) -> int:
    want = int(getattr(cfg, "formation_target", 0) or 0)
    return max(0, want)


def _apply_formation_gate(
    cfg: PhasePolicyConfig,
    state: PhasePolicyState,
    inp: PhasePolicyInputs,
    d: PhasePolicyDecision,
) -> PhasePolicyDecision:
    """Hold a tp-ward arm until the decode window is worth opening.

    THE DEFECT, MEASURED. Decode ran bs=1 on 261 of 288 batches and bs=4 on
    12, while 13 requests sat queued, token usage was 0.44 and the mamba pool
    held 2 of 12 slots. Nothing was throttling ADMISSION -- what was wrong was
    WINDOW FORMATION. Under pure drain a TP window can only decode the
    carriers whose prefill finished in the PRECEDING PP window, and the flip
    was fired by stall caps rather than by readiness, so a window opened with
    one ready carrier and then served bs=1 for its whole length while the
    other twelve waited for the next one.

    So the arm waits for carriers to ACCUMULATE, bounded two ways:

    * it never waits when the queue is empty -- there is nothing left to
      accumulate, and waiting would be pure latency;
    * it never waits longer than ``formation_cap_s``, so a single carrier
      that no one follows still gets its window.

    IT CANNOT REINTRODUCE THE IDLE WINDOW. The #688 idle-locked arm is
    returned by ``_decide_rules`` BEFORE any of this and is not a load rule,
    so a layout that can run nothing leaves immediately and is never held
    here. This gate only ever fires while the PP layout is still doing work,
    which is the whole difference between "shape the window" and "stall".

    DELIBERATELY A PLAIN TIME CAP, not an economics term. #677's window
    economics is a separate piece of work; this is the bounded first cut that
    makes bs=4 real, and it is labelled as such rather than presented as the
    optimum.
    """
    target = _formation_target(cfg, inp)
    if target <= 1:
        return d
    ready = int(getattr(inp, "ready_carriers", 0) or 0)
    if ready >= target:
        return d
    if not bool(getattr(inp, "queue_nonempty", False)):
        # Nothing more can arrive to fill the window; opening it now is right.
        return d
    started = float(state.formation_started_at or 0.0)
    if started <= 0.0:
        # First round of this formation window. The caller stamps the clock;
        # a pure function cannot, so the wait begins on the NEXT round and
        # this one still holds -- one round, not one cap.
        return PhasePolicyDecision(
            None,
            f"forming decode window: {ready} of {target} carrier(s) ready, "
            f"queue still filling -- holding the tp-ward arm",
        )
    waited = float(inp.now) - started
    cap = float(getattr(cfg, "formation_cap_s", 0.0) or 0.0)
    if cap > 0.0 and waited >= cap:
        return PhasePolicyDecision(
            d.direction,
            f"{d.reason} [formation cap: {ready} of {target} carrier(s) after "
            f"{waited:.1f}s >= {cap:g}s -- opening the window short rather "
            f"than making the ready carrier wait longer]",
        )
    return PhasePolicyDecision(
        None,
        f"forming decode window: {ready} of {target} carrier(s) ready after "
        f"{waited:.1f}s of {cap:g}s -- holding the tp-ward arm so the window "
        f"opens at width instead of at 1",
    )


def _decide_rules(
    cfg: PhasePolicyConfig,
    state: PhasePolicyState,
    inp: PhasePolicyInputs,
) -> PhasePolicyDecision:
    """The load rules, then the refused-arm hold. Pure, mutates nothing.

    THE HOLD IS APPLIED LAST, not folded into the rules, because the two
    answer different questions. The rules answer "which layout does this
    load want"; the hold answers "did the last attempt to get there work".
    Keeping them apart is what lets the hold be per-DIRECTION: a tp_to_pp
    seam that cannot be funded says nothing about pp_to_tp, which is
    staged from the other leg's buffers entirely (#656 boot E refused only
    tp_to_pp, and gagging both would have been a second bug).
    """
    d = _decide_from_load(cfg, state, inp)
    if not d.wants_flip:
        return d

    # THE STORM GUARD, as a RATE LIMITER rather than a latch.
    #
    # Two dampers were removed from this path: the doubling backoff became
    # dwell pacing, and the guards-layer latch that blocked re-pricing after 8
    # abandons is going. This is what remains, and it has one job -- bound the
    # rate of EXPENSIVE failures without ever refusing a flip that could now be
    # funded.
    #
    # So it paces on abandons, not on arms. An arm that never reaches staging
    # costs a broadcast; an arm that stages and abandons costs the seam's
    # memory peak, and 12 of those in four minutes once killed three boots.
    # Decoupling the two is the whole point: probes stay cheap and frequent,
    # expensive failures get a floor between them.
    #
    # The floor is SOLVED, not chosen: one failed staging per
    # `solved_tp_decode_floor_s` = 2 x flip_cost_s. That is already the period
    # below which cutover work exceeds half the wall clock, so applying it to
    # failed work bounds waste by the same rule that bounds useful work.
    #
    # It is not a latch and cannot become one: it holds nothing, remembers one
    # timestamp, never exceeds 2 x flip_cost_s, and a COMPLETION clears it
    # outright. The instant conditions allow funding, the next arm goes.
    last_abandon = state.last_abandon_at.get(d.direction)
    if last_abandon is not None and not _demand_outweighs_a_retry(cfg, inp, state):
        # Interval starts at one round trip and doubles per consecutive
        # abandon, capped. Doubling because a second failure against the same
        # conditions is evidence the conditions have not moved; capped because
        # they might, and past the cap the right behaviour is to keep asking
        # once a minute forever rather than to give up -- which is the
        # difference between a rate limiter and a latch.
        k = max(1, state.arm_refusals.get(d.direction, 1))
        # The base interval is one round trip WHERE THE SEAM WAS MEASURED, and
        # min_dwell_s where it was not. A safety limiter must not switch itself
        # off for want of a measurement -- that is the right gate for a
        # threshold term, and exactly the wrong one here, where the failure
        # mode is an arm storm against an unfundable seam.
        base_s = (
            solved_tp_decode_floor_s(cfg) if cfg.flip_cost_s > 0 else cfg.min_dwell_s
        )
        floor_s = min(base_s * (2**k), cfg.refusal_backoff_cap_s)
        since = inp.now - last_abandon
        if floor_s > 0 and since < floor_s:
            return _no(
                f"{d.direction} arm refused by the staging rate limit: {k} "
                f"consecutive abandons, "
                f"last {since:.1f}s ago, next staging in {floor_s - since:.1f}s "
                f"(interval {floor_s:.1f}s = {base_s:.1f}s doubled {k}x, capped "
                f"at {cfg.refusal_backoff_cap_s:g}s) "
                f"-- a rate limit, not a latch: it never stops re-probing and "
                f"a completion clears it outright"
            )
    hold_until = state.arm_hold_until.get(d.direction)
    if hold_until is not None and inp.now < hold_until:
        k = state.arm_refusals.get(d.direction, 0)
        degraded = state.arm_degraded.get(d.direction)
        # #662: WHILE THE WORK IS STILL THERE, A REFUSAL IS NOT A REASON TO
        # WAIT -- IT IS A REASON TO PAY.
        #
        # Reaching here means `_decide_from_load` already said this load wants
        # the other layout, so the arming condition PERSISTS. Backing off then
        # is not damping, it is the defect with a timer on it: measured on
        # this rig, tp_to_pp was refused for want of ~380 MiB while tens of
        # thousands of tokens sat pending, the direction was declared
        # unfundable, and the instance held in TP at 1000-1600 tok/s where the
        # PP layout does 4000-7000. Waiting 48 s changed nothing, because
        # nothing about waiting frees memory.
        #
        # What DOES free memory is the KV rung: the pool being full is what
        # makes the flip worth taking AND what pays for it, once the rung is
        # actually asked. So while the condition holds we re-attempt at the
        # plain dwell cadence -- each attempt runs the gate, which asks the
        # rung -- instead of doubling a backoff or honouring a degrade latch.
        #
        # The exponential backoff and the latch both survive for the case they
        # were written for: a load that has STOPPED wanting the flip, where
        # `wants_flip` is false and this branch is never reached.
        paced_until = hold_until
        if k > 0:
            paced_until = min(
                hold_until, _last_refusal_at(state, d.direction) + cfg.min_dwell_s
            )
        if inp.now >= paced_until:
            return d
        if degraded:
            return _no(
                f"{d.direction} refused {k} times and treated as unfundable "
                f"({degraded}); the load still wants this layout, so the next "
                f"attempt is in {paced_until - inp.now:.1f}s and it will ask "
                f"the KV rung again rather than wait out a backoff"
            )
        return _no(
            f"{d.direction} arm refused {k} in a row; the load still wants "
            f"this layout, so the next attempt is in "
            f"{paced_until - inp.now:.1f}s (dwell-paced, not backed off)"
        )

    return d


def _last_refusal_at(state: PhasePolicyState, direction: str) -> float:
    """When the current hold started, derived from the hold itself.

    The state carries the hold's END, not its start, so the start is recovered
    by subtracting the backoff that produced it. Kept as a helper so the
    arithmetic has one home and the caller reads as intent.
    """
    return float(state.arm_hold_until.get(direction, 0.0)) - float(
        state.arm_hold_last_span.get(direction, 0.0)
    )


def _decide_from_load(
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

    # #861c EXISTENCE class: a fully cached backlog is still work. Reading
    # the economics number here called a box with six queued requests IDLE.
    # #861e: a box holding 7 retracted mid-flight bundles is not idle.
    idle = inp.decode_work_bs() == 0 and not inp.work_exists()

    # NOTHING CAN RUN HERE -- CHECKED BEFORE THE DWELL, AND THAT IS THE POINT.
    #
    # Every other rule below asks "is flipping WORTH it", and a dwell floor is
    # the right damper for that question: it bounds thrash between two layouts
    # that can both do work. This rule asks a different question -- "can this
    # layout do ANY work at all" -- and when the answer is no, waiting is not
    # a trade-off, it is dead time. Measured on metal 2026-08-16 09:42:39-45:
    # six seconds of zero GPU, zero PCIe, py-spy showing all three ranks
    # spinning get_next_batch_to_run and building nothing, released only by
    # the 180 s decode-stall cap. DRAINED could not fire (it needs pending
    # <= one chunk; there were 572715 tok) and the #677(a) progress exit could
    # not fire either (it needs pending FROZEN, and pending was still creeping
    # 572792 -> 572715, which reads as progress).
    #
    # A TIMER IS THE WRONG INSTRUMENT FOR A FACT THAT IS ALREADY KNOWN. The
    # round has just finished failing to build a batch of either class; that
    # is not a prediction to be confirmed by waiting, it is a completed
    # observation. So the arm is event-driven off that failure and carries no
    # interval of its own.
    #
    # IT CANNOT OSCILLATE, and not because a damper stops it. The condition is
    # ONE-SIDED: it requires that the current layout can build nothing AND
    # that the target can build something. After the flip the target runs by
    # premise, so the same condition is false there. Bypassing the dwell is
    # therefore safe in exactly this branch and nowhere else -- which is why
    # it is written here, above the dwell, rather than as an exception inside
    # it.
    if inp.nothing_can_run and not inp.target_can_admit:
        # BOTH SIDES BLOCKED IS AN EVICT PROBLEM, NOT A LAYOUT PROBLEM.
        #
        # This branch exists because the first version of this rule did not
        # have it, and shipped a flip loop: on 2026-08-16 10:24 the policy
        # armed tp_to_pp, pp_to_tp and tp_to_pp again in seven seconds, with
        # 2 requests resident and 910140 tokens queued, because each layout
        # was certified as the other's escape while NEITHER could run. When
        # the target is not admissible either, changing layout cannot help --
        # the binding resource is KV, and the action that unblocks a side is
        # freeing it. Declining here is what routes the caller to the evict
        # rung instead of to a cutover.
        # #708: NAME THE BINDING RESOURCE FROM THE MEASUREMENT, not from a
        # hardcoded string. The routing conclusion below is unchanged and does
        # not depend on which resource binds -- when the target cannot admit
        # either, changing layout cannot help, so this is never a flip. What
        # changes is the diagnosis handed to whoever reads the line: it now
        # says what it measured, and admits when it measured nothing.
        avail = inp.kv_available_tokens
        if avail is None:
            binding = (
                "the binding resource was NOT MEASURED here (no rank-uniform "
                "KV availability was supplied), so this line does not name one"
            )
        elif avail < inp.pending_prefill_tokens:
            binding = (
                f"the binding resource is KV ({avail} rows available against "
                f"{inp.pending_prefill_tokens} pending)"
            )
        else:
            binding = (
                f"KV is NOT the binding resource ({avail} rows available "
                f"against {inp.pending_prefill_tokens} pending) -- look at the "
                f"state-slot bound (mamba/GDN slots) before blaming the pool"
            )
        return _no(
            f"{BOTH_BLOCKED}: nothing can run in the {inp.phase} layout and the "
            f"target cannot admit either ({inp.running_bs} req resident, "
            f"{inp.pending_prefill_tokens} tok pending) -- {binding}, so this "
            f"is an evict trigger and NOT a flip. Flipping here is the 10:24 "
            f"ping-pong."
        )

    if inp.nothing_can_run and inp.target_can_admit:
        direction = PP_TO_TP if inp.phase == PHASE_PP else TP_TO_PP
        pending = int(inp.pending_prefill_tokens or 0)

        # #759: THE ESCAPE HATCH NEEDED A FLOOR. This branch used to arm
        # unconditionally -- "arming immediately rather than waiting for a
        # stall timer to notice" -- and comp4 Gate A failed 3 of 3 on it:
        # 8 small requests produced two armings 13 s apart,
        #   06:32:37 tp_to_pp  (0 req resident,  68 tok pending)
        #   06:32:50 pp_to_tp  (1 req resident,   0 tok pending)
        # a flip out and straight back, funded by 68 tokens and then by
        # nothing. Note this path never reaches the economic window at all, so
        # #677's cost terms could not have refused it -- the floor has to live
        # here.
        # NOT gated on pending_prefill_tokens > 0. #689's own fixture proves
        # that measure wrong: it expresses real queued work with
        # ``queue_nonempty=True`` while ``pending_prefill_tokens`` is 0, so a
        # "nothing pending means nothing to serve" rule would refuse a genuine
        # deadlock escape. The qualifier below is PERSISTENCE, not a work
        # count.
        # #819: the MEASURED break-even, not the boot seed. The estimator's own
        # docstring names this floor as collateral damage of the stale price --
        # "#759's IDLE-LOCK floor -- itself `flip_tokens` -- 7x too
        # permissive". Repricing the bar without repricing this floor would
        # leave the two disagreeing about what a flip costs.
        floor = int(live_flip_tokens(cfg) or 0)
        if pending < floor or (floor <= 0 and pending <= 0):
            # Below the MEASURED break-even, so this flip cannot repay itself
            # on the backlog it would carry. It is still a real escape route,
            # so it is delayed rather than refused: a transient gap between two
            # small requests is not a lock, and treating it as one is what
            # produced the ping-pong. A lock that PERSISTS still gets out --
            # refusing outright would trade Gate A's thrash for the #748 wedge.
            #
            # #748 REFAIL: THE QUALIFIER WAS RIGHT, THE CLOCK WAS DEAD.
            #
            # #759 named PERSISTENCE as the qualifier and then read it off
            # ``state.idle_since``, which ``observe_idle`` stamps only while
            # ``running_bs == 0 and pending_prefill_tokens == 0``. An IDLE-LOCK
            # is by definition the opposite state -- work EXISTS and this
            # layout cannot run it -- so on this path the stamp is structurally
            # absent. Measured on boot_735_nohc.log: of 160 IDLE-LOCK armings,
            # ZERO had (0 resident, 0 pending), and all 160 log lines carry the
            # "no idle observation" branch below. The floor was live and the
            # delay was unreachable, which is why the #759 desk tests were
            # green while the metal signature did not move.
            #
            # ``nothing_can_run_since`` is the clock that CAN see it: stamped by
            # ``observe_idle`` from the same ``nothing_can_run`` term this
            # branch is keyed on. ``idle_since`` remains the fallback so a
            # caller that stamps only the idle clock -- every #759 fixture --
            # keeps its exact answer.
            lock_since = state.nothing_can_run_since
            if lock_since is None:
                lock_since = state.idle_since
            if lock_since is None:
                # UNOBSERVED IS NOT BRIEF. With no lock stamp there is no
                # evidence the lock is transient, and #689's invariant --
                # "a layout that can run NOTHING leaves at once" -- governs.
                # Delaying on absence of evidence would convert an unproven
                # suspicion into a wedge, which is the #748 direction.
                return PhasePolicyDecision(
                    direction,
                    f"{IDLE_LOCKED}: no batch of either work class can be "
                    f"built in the {inp.phase} layout ({inp.running_bs} req "
                    f"resident, {pending} tok prefill pending) and the target "
                    f"layout can run one -- below the {floor}-tok break-even "
                    f"but with no lock observation to call it transient, so "
                    f"the deadlock escape is not delayed",
                )
            idle_for = inp.now - lock_since
            if idle_for < cfg.idle_dwell_s:
                return _no(
                    f"{IDLE_LOCKED} with {pending} tok < break-even {floor} "
                    f"tok, and idle {idle_for:.1f}s < {cfg.idle_dwell_s:g}s "
                    f"idle dwell: a transient gap is not a lock"
                )
            return PhasePolicyDecision(
                direction,
                f"{IDLE_LOCKED}: no batch of either work class can be built in "
                f"the {inp.phase} layout ({inp.running_bs} req resident, "
                f"{pending} tok prefill pending) and the target layout can run "
                f"one. Below the {floor}-tok break-even, so it waited: idle "
                f"{idle_for:.1f}s >= {cfg.idle_dwell_s:g}s idle dwell",
            )

        return PhasePolicyDecision(
            direction,
            f"{IDLE_LOCKED}: no batch of either work class can be built in the "
            f"{inp.phase} layout ({inp.running_bs} req resident, "
            f"{pending} tok prefill pending) and the "
            f"target layout can run one -- arming immediately rather than "
            f"waiting for a stall timer to notice",
        )

    # The minimum dwell is checked FIRST and applies to every flip, so no
    # branch below can bypass the thrash bound. It is the only guarantee
    # that survives an adversarial arrival pattern.
    since_flip = inp.now - state.last_flip_at
    # #768: the dwell is a THRASH bound, and thrashing needs work in flight.
    # With nothing running and prefill pending, holding protects no throughput
    # -- there is none to protect -- and nothing can end the wait either: the
    # request that would restart the clock is the one the hold refuses to
    # admit. That is a livelock, not a bound. It wedged serving for 464s
    # ("ADMISSION-WEDGE: 1 queued, 0 running, and NO first token", 41 of them)
    # on the specimen numbers 5813 tok pending / 0 running / "0.0s since last
    # flip < 3s", with health still answering 200 the whole time.
    # #861c EXISTENCE class: 'can this layout do ANY work at all'. A
    # prefetched backlog starves the box exactly as an uncached one does,
    # and the dwell floor must not hold the flip that would release it.
    # #861e: reading the manufactured 0 here bypasses the dwell floor in the
    # one round where the floor matters most -- immediately after a cutover.
    starved = int(inp.decode_work_bs()) == 0 and (
        max(
            int(getattr(inp, "pending_prefill_tokens", 0) or 0),
            int(getattr(inp, "admissible_prefill_tokens", 0) or 0),
        )
        > 0
    )
    if state.last_flip_at > 0 and since_flip < cfg.min_dwell_s and not starved:
        return _no(
            f"min dwell: {since_flip:.1f}s since last flip < {cfg.min_dwell_s:g}s"
        )

    if inp.phase == PHASE_TP:
        # Enough queued prefill to repay the round trip -> go to the prefill
        # layout. Below the threshold the prefill runs in TP as-is, which is
        # the correct answer for short prompts by construction of N.
        # Under purity the break-even is meaningless (see
        # prefill_runs_in_tp): the only threshold that terminates is zero.
        tp_threshold = effective_flip_threshold(cfg, inp.decode_work_bs())  # #861e
        # #819: ONE READING. Taken once here and used for both the comparison
        # and every message below it, so the bar the policy APPLIED and the bar
        # the log REPORTS can never be two different numbers -- which is what a
        # repriced threshold printed against `cfg.flip_tokens` would have been.
        n_live = live_flip_tokens(cfg)
        seam_s = live_flip_cost_s(cfg)
        priced = flip_cost_provenance()
        if (
            cfg.prefill_runs_in_tp
            # #856 STRICT BATCHING: the band cannot apply here. It sits ABOVE
            # the drain exit and returns `_no`, so a correct drain rule below
            # is not sufficient -- the band has to stop being consulted at all,
            # or it holds TP and prefill runs in the decode layout, which is
            # the one thing this mode forbids.
            and not cfg.drain_mode_strict
            and inp.running_bs > 0
            and n_live < inp.pending_prefill_tokens <= tp_threshold
        ):
            # #853(iii): THE BAND'S PREMISE IS A CLAIM ABOUT A RATE, AND IT
            # CAN BE FALSE.
            #
            # Everything below this point holds TP because "prefilling it in
            # tp beats the round trip". That is not a statement about the
            # backlog's size alone -- it is priced at `tp_prefill_tok_s`, and
            # it is only true while the backlog is ACTUALLY BEING PREFILLED
            # HERE. If no chunk is being computed, the arithmetic is denominated
            # in a rate that is not happening, and the hold becomes #833's
            # shape one branch up: a wait for something that has stopped.
            #
            # THIS WAS THE ONLY HOLD IN THIS FUNCTION THAT COULD NOT BE WRONG.
            # Every neighbour already carries a bound on its own premise: the
            # min dwell yields to `starved` (#768), drain mode yields to the
            # #833 stall deadline, the idle lock yields to the idle dwell
            # (#748). The band yielded to nothing, so a backlog parked between
            # the two bars with a decode resident held the layout for as long
            # as it stayed there.
            #
            # THE AXIS IS PREFILL PROGRESS, NOT THE DECODE BUNDLE, and the
            # difference is load-bearing. W24's ticket proposed breaking the
            # band when the decode bundle is not draining. At the measured
            # `decode_contention` = 1 the scheduler gives prefill absolute
            # priority per iteration, so while prefill is pending in TP the
            # bundle CANNOT shrink -- by construction, not by defect. Breaking
            # on it would collapse the band to the plain break-even for every
            # load with a request decoding, silently deleting the #665-F1
            # differential model. So the falsifier is taken on the axis the
            # claim is actually made on.
            #
            # `pending_prefill_tokens` is "admitted but not yet computed",
            # measured at the chunk fill boundary, so it drops by a chunk every
            # round a chunk is computed -- even mid-way through one long
            # prompt. Frozen for a whole decode window therefore means no
            # chunk was computed for a whole decode window. The clock is
            # #677(a)'s, already maintained by `observe_idle`; unstamped reads
            # as no stall, so a caller that never observed cannot flip on it.
            prefill_stall_s = (
                0.0
                if state.last_prefill_progress_at is None
                else inp.now - state.last_prefill_progress_at
            )
            band_deadline_s = drain_stall_deadline_s(cfg)
            if band_deadline_s > 0 and prefill_stall_s >= band_deadline_s:
                return PhasePolicyDecision(
                    TP_TO_PP,
                    f"prefill progress STALLED in tp: "
                    f"{inp.pending_prefill_tokens} tok has not gone down for "
                    f"{prefill_stall_s:.1f}s (deadline {band_deadline_s:.1f}s) "
                    f"while the backlog sits in the secondary band "
                    f"(> N={n_live}, <= {tp_threshold}) with "
                    f"{inp.running_bs} req decoding. That band holds because "
                    f"prefilling it here is supposed to beat the round trip -- "
                    f"a claim priced at the tp prefill rate, and no chunk has "
                    f"been computed for a whole decode window, so the rate it "
                    f"is priced on is not happening. Flipping is what makes "
                    f"that backlog runnable (seam {seam_s:.2f}s {priced})",
                )
            # Repays the seam, but not the seam PLUS the generations this
            # cutover would pause for a whole PP window. Named explicitly so
            # the log says which term refused, not just "below threshold".
            if cfg.decode_contention > 0.0:
                # Under the differential model the refusal is NOT "a flip
                # would strand them" -- they are stranded either way -- but
                # "the backlog is not yet big enough that flipping shortens
                # the stall". Say which, or the log invites the wrong fix.
                return _no(
                    f"pending prefill {inp.pending_prefill_tokens} tok > N="
                    f"{n_live} but <= {tp_threshold} with "
                    f"{inp.running_bs} req decoding: too short for the round "
                    f"trip to beat prefilling it in tp "
                    f"(seam {seam_s:.2f}s {priced})"
                )
            # #893: the seconds NAMED here are the seconds CHARGED above, and
            # both are the residency that governs -- not `cfg.pp_window_s`,
            # which under a declared SLO is the number #889 showed can never
            # end the phase. A refusal line quoting the inert knob invites the
            # operator to widen a knob that does nothing.
            return _no(
                f"pending prefill {inp.pending_prefill_tokens} tok > N="
                f"{n_live} but <= {tp_threshold} with "
                f"{inp.running_bs} req decoding: flipping would strand them "
                f"in pp for {stranded_decode_s(cfg):g}s "
                f"({effective_pp_exit_term(cfg)[0]}) "
                f"(seam {seam_s:.2f}s {priced})"
            )
        if cfg.drain_mode and inp.pending_prefill_tokens > cfg.pp_exit_tokens:
            # #677 HOT FIX 2: THE BUNDLE IS FINISHED BEFORE THE LAYOUT MOVES.
            #
            # Measured live: `arming tp_to_pp: pending > N=7004` fired with
            # running bs 2-3, cutting a decode bundle in half -- and under
            # purity the backlog is ALWAYS above N, so that rule cannot be a
            # reason to leave without meaning "never finish anything". Five
            # blocked-admission exits in one boot were the same carriers
            # coming back unfinished, window after window.
            #
            # The user's semantics: prefill until empty, decode the bundle to
            # completion, prefill again. So the backlog stops being an exit
            # condition here and BS==0 becomes one. The min-dwell, the decode
            # floor, the 180s cap and the #677a progress exit are all
            # untouched underneath -- this changes what ends a HEALTHY window,
            # never what rescues a broken one.
            if inp.running_bs > 0:
                # #833: THE WAIT IS BOUNDED BY THE ADMITTED SET'S OWN PROGRESS.
                #
                # `running_bs == 0` is an exit condition sustained load never
                # revisits. With `--max-running-requests 8` and ~42k-token
                # prompts chunked at 4096 (~11 chunks each), an admitted
                # request occupies the bundle for many rounds and admission
                # refills every slot it frees, so the set never reaches zero.
                #
                # MEASURED, boot_window3_0823_1733: pending prefill was driven
                # to 836,048 tok -- 119x the bar N=7004 and 65x the scaled
                # ceiling -- across three escalating load shapes, and the arm
                # refused under every one of them, 151 times, naming the same
                # bundle. Driving pending HARDER made it worse, because more
                # pressure lengthens the admitted set's occupancy. A condition
                # that recedes as you approach it is divergent, and no amount
                # of load will ever satisfy it.
                #
                # THE BINDING QUANTITY IS THE ONE THAT CONVERGES. The bundle
                # getting smaller is progress toward an empty set; the bundle
                # holding station is not, however much work it retires. So the
                # wait is bounded by a STALL on that axis, and the direction of
                # the pressure derivative is inverted by construction: more
                # pressure keeps the set full, which makes the stall trip
                # SOONER, so the refusal rate falls with load instead of rising
                # with it. #677's semantics are untouched where they were ever
                # true -- a bundle that is genuinely draining is never cut, and
                # a set that empties still exits through the branch below.
                #
                # THE NEIGHBOURING TICKET, NOT BUILT HERE: #819 would let
                # admission itself stop refilling the bundle while a flip is
                # pending, which is the other half of this and a policy change
                # of its own. This ticket only bounds the wait.
                stall_s = (
                    0.0
                    if state.last_bundle_progress_at is None
                    else inp.now - state.last_bundle_progress_at
                )
                deadline_s = drain_stall_deadline_s(cfg)
                if stall_s < deadline_s:
                    return _no(
                        f"decode bundle running: {inp.running_bs} of "
                        f"{max(state.bundle_at_phase_entry, inp.running_bs)} req "
                        f"still decoding, {inp.pending_prefill_tokens} tok prefill "
                        f"waiting -- drain mode finishes the bundle before "
                        f"flipping (bundle last shrank {stall_s:.1f}s ago, "
                        f"stall deadline {deadline_s:.1f}s)"
                    )
                return PhasePolicyDecision(
                    TP_TO_PP,
                    f"decode bundle STALLED, not draining: {inp.running_bs} of "
                    f"{max(state.bundle_at_phase_entry, inp.running_bs)} req "
                    f"still decoding and the set has not shrunk for "
                    f"{stall_s:.1f}s (deadline {deadline_s:.1f}s), while "
                    f"{inp.pending_prefill_tokens} tok of prefill waits. "
                    f"Admission is refilling the bundle as fast as it retires, "
                    f"so waiting for it to empty cannot converge -- more "
                    f"pressure would only lengthen its occupancy (#833). "
                    f"Flipping is what makes that backlog runnable",
                )
            in_phase = (
                None if state.phase_since is None else inp.now - state.phase_since
            )
            bundle = max(state.bundle_at_phase_entry, 0)
            elapsed = 0.0 if in_phase is None else in_phase
            if bundle == 0 and inp.seam_transport_tokens > 0:
                # W32: UNDER NO-CARRY, EMPTY AT ENTRY IS THE NORMAL STATE.
                #
                # The rule below was written for a seam that CARRIED its
                # residents, where "entered TP with 0 decoding" really did mean
                # the bundle never existed. Under #856 no-carry the cutover
                # RETRACTS every resident, so the bundle is absent at entry BY
                # CONSTRUCTION and arrives one round later through the
                # re-admission. The old rule therefore fired on every single
                # flip -- 26 times in W32 -- and armed tp_to_pp straight back
                # out of the layout the bundle was about to be served in.
                #
                # The invariant that DOES exist now: an empty decode phase is a
                # defect only when NO re-admission is in flight. While these
                # tokens are outstanding the right action is to WAIT for them,
                # not to flip away from them.
                return _no(
                    f"decode phase entered empty, which is normal under "
                    f"no-carry: {inp.seam_transport_tokens} tok of seam "
                    f"re-admission are in flight and are served in THIS "
                    f"layout by read-through. Waiting for the bundle the "
                    f"cutover just retracted, not flipping away from it"
                )
            if bundle == 0:
                # #730: ZERO WORK MUST NOT READ AS ALL WORK.
                #
                # This branch used to emit "decode bundle complete: 0 reqs
                # decoded ... exit condition: decode drained" -- the same
                # sentence a genuinely drained bundle produces, differing only
                # by a number nobody reads as a verdict. Measured 2026-08-17
                # 14:48:22: the tp phase was entered with NOTHING decoding,
                # sat 183.6 s, and left claiming completion while 13,777 tok of
                # prefill waited. Six such cycles in one boot, and the warmup
                # generation never reached a first token.
                #
                # `bundle_at_phase_entry` is `running_bs` AT ENTRY (:2093), so
                # 0 here means the bundle was never resident -- there was no
                # work to drain and nothing was accomplished by the visit.
                # Leaving is still right (a decode layout with nothing to
                # decode should yield to prefill); what was wrong is calling it
                # completion.
                #
                # The #699 admission-wedge detector fired 17 times on this same
                # boot and recorded "no phase-policy corroboration seen". This
                # line is that corroboration: an empty decode phase beside a
                # non-empty backlog is a symptom of work that cannot be
                # admitted, and it now says so where the operator is already
                # looking.
                return PhasePolicyDecision(
                    TP_TO_PP,
                    f"decode phase ran EMPTY: no bundle was ever resident "
                    f"(entered with 0 decoding, {elapsed:.1f} s ago) while "
                    f"{inp.pending_prefill_tokens} tok of prefill waited. "
                    f"NOTHING WAS DRAINED -- this is not a completed bundle, "
                    f"it is a decode window that had no work to do. If this "
                    f"repeats with a non-empty backlog, the defect is upstream "
                    f"in ADMISSION, not in the layout: work is queued that "
                    f"nothing is making runnable (#731, and see the #699 "
                    f"admission-wedge line for queued-vs-running)",
                )
            return PhasePolicyDecision(
                TP_TO_PP,
                f"decode bundle complete: {bundle} req bundle drained in "
                f"{elapsed:.1f} s "
                f"({inp.pending_prefill_tokens} tok prefill waiting) -- exit "
                f"condition: decode drained",
            )
        # #861d SECOND HALF: THE DEMAND SIDE, and the question is not the same
        # question as the price.
        #
        # `pending_prefill_tokens > tp_threshold` asks "is the backlog worth a
        # flip" -- ECONOMICS, and correct where economics decides. Under STRICT
        # purity economics does NOT decide: the user's law is drain-and-flip,
        # not break-even ("Break-even ist NICHT der Trigger"). Prefill simply
        # cannot run here, so ANY queued prefill work must demand the flip
        # regardless of its price, and a cached prompt is queued work.
        #
        # MEASURED, W37-D retry: `holding in tp: pending prefill 0 tok <=
        # N=28544, running it in tp` while SIX requests sat queued, every token
        # HiCache-cached so `uncached_prompt_tokens` was 0 -- and
        # ADMISSION-WEDGE alarmed for 368 s. The #861c existence term had
        # reached `_layout_admits` (prefill correctly refused in tp, rung 10
        # green) but NOT this branch, so the requests starved behind a CORRECT
        # refusal with nothing ever demanding the layout that could serve them.
        # Half a fix is its own failure mode.
        #
        # THE CLASSIFICATION ERROR THIS CORRECTS IS MINE. #861c's sweep put the
        # threshold arms in the ECONOMICS class and routed only the four
        # existence sites. That is right in relaxed mode and wrong in strict:
        # the QUESTION a site asks depends on the purity mode, not on the site
        # alone. Recorded as a sweep-completeness failure rather than a new
        # class -- the class ("one number, two questions") was already named.
        # #861d-2: ONE READ FEEDS THE VERDICT AND THE MESSAGE (#713).
        #
        # The first cut computed the verdict from `work_exists()` and printed
        # `pending_prefill_tokens`, so all 18 armings of boot d3 read
        # "pending prefill 0 tok > 0" -- a verdict of >0 beside a printed 0.
        # That is unreadable, and it is the exact rule #713 exists for: a
        # decision and its explanation must come from the SAME read.
        demand_tokens = inp.demand_prefill_tokens()
        strict_demands_flip = demand_tokens > 0 and not cfg.prefill_runs_in_tp
        # #861j: the state the subtraction above produces must be NAMED, not
        # fall through to a generic hold -- the W37-F specimen's whole cost
        # was that the closed door was invisible from the log. Reached
        # exactly when the only outstanding prefill is the cutover's own
        # re-admission and the exemption will serve it here.
        if (
            not cfg.prefill_runs_in_tp
            and not strict_demands_flip
            and int(getattr(inp, "seam_serviceable_tokens", 0) or 0) > 0
            and inp.pending_prefill_tokens <= tp_threshold
        ):
            return _no(
                f"seam transport: {inp.seam_serviceable_tokens} tok of the "
                f"cutover's own re-admission are serviceable in THIS layout "
                f"by read-through (premise verified on the retract credit) "
                f"-- holding for the transport batch instead of flipping "
                f"away from it. Bounded: the transport-debt clock lapses "
                f"this credit after the drain-stall deadline, and the "
                f"demand then fires (#861j)"
            )
        if inp.pending_prefill_tokens > tp_threshold or strict_demands_flip:
            # THE DECODE FLOOR. Under purity every token of prefill has to
            # wait for a PP window, so the backlog is essentially always
            # above N and this rule would otherwise fire the instant
            # min_dwell expires -- giving decode a 3 s window per cycle and
            # starving it in the mirror image of the defect the PP window
            # fixes. Only applies while decode work actually exists: with
            # nothing decoding there is nothing to protect, and an arriving
            # long prompt should reach the PP layout inside its TTFT.
            #
            # #820: THIS IS WHERE THE "TP MIRROR" BELONGED, and it is already
            # here. `layout_hold_verdict` used to carry a second, parallel
            # version of exactly this protection (MIN_DECODE_ROUNDS: "a pull
            # must not preempt a RUNNING decode batch mid-batch"), written in
            # 332cb3b345 for the same reason this floor exists. It was never
            # reachable: `decide` consults that verdict only for PP_TO_TP arms,
            # and every PP_TO_TP arm is built under `inp.phase == PHASE_PP`.
            # The author's own commit message records why the wiring stopped
            # there -- "My first wiring vetoed EVERY arm, including the idle
            # return leg -- 11 tests red" -- which is what a hold-veto does to
            # a tp branch whose `pend <= 0` answer is "no pull": read as a veto,
            # "no pull" becomes "never leave tp", unbounded, because that
            # return never sees `max_hold_rounds`.
            #
            # The two branches answered DIFFERENT QUESTIONS. The pp branch asks
            # "may the timer take the layout" -- a veto, which is what the
            # wrapper consumes. The tp branch asks "does demand pull the layout
            # out" -- an arm-CREATION, which that wrapper can never do by
            # construction. So the mirror is not a hold at all; it is this rule,
            # and this rule states it against the real phase clock in seconds
            # rather than against a round counter the caller does not have.
            # Removed there, recorded here.
            in_phase = (
                None if state.phase_since is None else inp.now - state.phase_since
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
            # #713 / #861d-2: the number printed is the number the verdict
            # used. d3 printed `pending_prefill_tokens` beside a verdict taken
            # from a different read and produced 18 lines of "0 tok > 0".
            _shown = (
                inp.pending_prefill_tokens
                if inp.pending_prefill_tokens > tp_threshold
                else demand_tokens
            )
            return PhasePolicyDecision(
                TP_TO_PP,
                f"pending prefill {_shown} tok > "
                f"{'N=' + str(n_live) + f' (seam {seam_s:.2f}s {priced})' if cfg.prefill_runs_in_tp else '0 (purity: prefill cannot run in tp, nothing decoding)'}",
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
            return _no(f"idle {idle_for:.1f}s < {cfg.idle_dwell_s:g}s idle dwell")
        if inp.work_exists():  # #861c EXISTENCE class
            # #861d: THE COMMENT THAT USED TO STAND HERE WAS FALSIFIED, and it
            # is worth keeping the correction visible. It read: "Unreachable
            # under purity: tp_threshold is 0 there, so any non-zero pending
            # already armed above." That is true only while "pending" and
            # "work exists" are the same quantity. They are not: a queue of
            # fully cached prompts has `pending_prefill_tokens == 0` and real
            # work, so `0 > 0` is False, the arm above did not fire, and
            # control reached this supposedly unreachable hold -- for 368 s,
            # with six requests starving. The arm above now consults the
            # existence term under strict purity, which restores the comment's
            # claim by making it TRUE rather than by asserting it.
            #
            # Same class as #861c/F1 and #861d: a premise stated in prose and
            # never checked. Third instance, so it is the SWEEP that was
            # incomplete, not the class that was unknown.
            return _no(
                f"pending prefill {inp.pending_prefill_tokens} tok <= "
                f"N={n_live}, running it in tp (seam {seam_s:.2f}s {priced})"
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
        # #861d THE MIRROR, and without it the other half is a ping-pong.
        #
        # This arm means "prefill is done, take the decoders to TP". With a
        # fully cached backlog `pending_prefill_tokens` is 0, so it reads
        # "done" while requests are still QUEUED and never computed -- PP
        # leaves, TP refuses to prefill (strict), the TP arm above demands the
        # flip straight back, and the instance oscillates without serving.
        # Fixing only the tp->pp side would have produced exactly that, so the
        # two directions are one change.
        #
        # Under STRICT purity, PP must not hand the layout back while queued
        # prefill work exists, whatever that work costs. Under relaxed purity
        # the economics stand: TP can prefill there, so leaving is a price
        # question and `pending <= N` is the right one to ask.
        # #861d-2: the SAME single read as the tp->pp arm, so the two
        # directions cannot disagree about whether work exists.
        _strict_holds_pp = (
            inp.demand_prefill_tokens() > 0 and not cfg.prefill_runs_in_tp
        )
        if (
            inp.pending_prefill_tokens <= cfg.pp_exit_tokens
            and inp.running_bs > 0
            and not _strict_holds_pp
        ):
            # W30: AN ARM MAY NOT DESTROY ITS OWN JUSTIFICATION.
            #
            # The reason logged below is "N req decoding", i.e. take these
            # requests to the layout that decodes. Under #856 no-carry the
            # cutover retracts exactly those requests, so unless the target
            # can re-admit them the flip arrives with nothing to do, the
            # policy flips back, and the requests re-prefill in PP -- for
            # ever. Refusing here is what keeps that at 3 flips instead of
            # 150, and it is a REFUSAL rather than a silent hold so the
            # operator sees which contract is missing.
            if not getattr(cfg, "seam_readmit_available", True):
                return _no(
                    f"NOT ARMING pp_to_tp despite {inp.running_bs} req "
                    f"decoding and {inp.pending_prefill_tokens} tok pending: "
                    f"the cutover would RETRACT those requests (#856 "
                    f"no-carry) and the target layout cannot re-admit them, "
                    f"so this arm would destroy the reason it armed. W30 "
                    f"measured that as 150 flips and zero decode batches",
                )
            return PhasePolicyDecision(
                PP_TO_TP,
                f"DRAINED: {inp.pending_prefill_tokens} tok remaining "
                f"(<= one chunk of {cfg.pp_exit_tokens}), {inp.running_bs} req "
                f"decoding -- exit condition: drained",
            )
        # #677(a) BLOCKED ADMISSION, CHECKED BEFORE THE RESIDENCY CAP.
        #
        # Measured live 2026-08-16 06:04: pending frozen at 403779 tok from
        # 06:04:47 with running bs 4 -- every max_running_requests slot held by
        # a carried decode. PP may not decode under strict purity, so no slot
        # could free; admission needs a slot, so no chunk could land; and the
        # DRAINED rule waits for pending to fall below one chunk, which it
        # never would. The instance was not slow, it was waiting for an event
        # that could no longer happen, and the user saw a dead server.
        #
        # This is not a deadline. It fires on the absence of PROGRESS, which
        # any single admitted chunk resets, so a genuine drain is never cut
        # short however long it takes -- the property the pure-drain decision
        # requires. See `pp_progress_stall_window_s` for the solve.
        stall_window = pp_progress_stall_window_s(cfg)
        stalled_for = (
            None
            if state.last_prefill_progress_at is None
            else inp.now - state.last_prefill_progress_at
        )
        if (
            stall_window > 0
            and stalled_for is not None
            and stalled_for >= stall_window
            and inp.pending_prefill_tokens > cfg.pp_exit_tokens
            and inp.running_bs > 0
        ):
            return PhasePolicyDecision(
                PP_TO_TP,
                f"blocked admission: pending frozen at "
                f"{inp.pending_prefill_tokens} tok for {stalled_for:.1f}s with "
                f"bs {inp.running_bs} carried (no chunk admitted in "
                f"{stall_window:.1f}s, solved as {PROGRESS_STALL_CHUNKS}x the "
                f"{cfg.pp_exit_tokens}tok/{cfg.pp_prefill_tok_s:g}tok-s chunk "
                f"cadence, floored at 2x{cfg.flip_cost_s:g}s seam) -- exit "
                f"condition: blocked admission",
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
        # THE DECODE-STARVATION CAP, solved from the declared latency
        # constraint and the measured seam. Unlike the stopwatch it replaces,
        # this is the only thing that may cut a drain short, and it says why in
        # units someone can argue with: a carried decode has been stalled for
        # its whole budget.
        in_pp = None if state.phase_since is None else inp.now - state.phase_since
        cap = pp_residency_cap_s(cfg)
        if cap > 0 and in_pp is not None and inp.running_bs > 0 and in_pp >= cap:
            return PhasePolicyDecision(
                PP_TO_TP,
                f"decode stall cap: {inp.running_bs} req stalled "
                f"{in_pp + 2 * cfg.flip_cost_s:.1f}s of the "
                f"{cfg.decode_stall_slo_s:g}s budget (residency {in_pp:.1f}s "
                f">= {cap:.1f}s solved as slo - 2x{cfg.flip_cost_s:g}s seam); "
                f"{inp.pending_prefill_tokens} tok prefill still pending -- exit "
                f"condition: decode starvation cap",
            )

        # LEGACY STOPWATCH. Retained so a deployment that set pp_window_s keeps
        # its behaviour, but it now has to say what the drain-based policy
        # would have done instead -- the whole point being that the next audit
        # can compare the two without re-deriving anything.
        if (
            cap <= 0
            and cfg.pp_window_s > 0
            and in_pp is not None
            and inp.running_bs > 0
            and in_pp >= cfg.pp_window_s
        ):
            rung = effective_flip_threshold(cfg, inp.decode_work_bs())  # #861e
            # #819: the drain target is the same bar as everywhere else, so it
            # is read the same way. Left frozen, this line would report a
            # drain-based counterfactual against a target the policy itself no
            # longer uses.
            n_live = live_flip_tokens(cfg)
            seam_s = live_flip_cost_s(cfg)
            would_drain = inp.pending_prefill_tokens <= n_live
            drain_s = (
                inp.pending_prefill_tokens / cfg.pp_prefill_tok_s
                if cfg.pp_prefill_tok_s > 0
                else float("nan")
            )
            return PhasePolicyDecision(
                PP_TO_TP,
                # #817: NOT hold_eligible, and this is the arm that looked like
                # it should be. The #677 hold was written to veto "the plain
                # timer/economics exit", and this is that timer -- but that
                # permission rests on an unstated assumption: that vetoing the
                # timer leaves some OTHER backstop armed. Look at the guard
                # this arm sits behind: `cap <= 0`. It fires only when the
                # decode-starvation cap is absent, i.e. exactly in the states
                # where it is the LAST anti-pinning bound there is. Vetoing the
                # last bound is an unbounded hold, which is verbatim the
                # condition test_sustained_backlog_still_leaves_pp_via_the_
                # window exists to prevent ("returned 'holding in pp' on every
                # call, without end") and the shape of the live wedge family.
                # The assumption outranks the prose, so this arm is an EXIT.
                reason=f"pp window {in_pp:.1f}s >= {cfg.pp_window_s:g}s with "
                f"{inp.running_bs} req waiting to decode "
                f"({inp.pending_prefill_tokens} tok prefill deferred to the "
                f"next pp window) -- HAND-SET STOPWATCH; drain-based policy "
                f"would {'also leave' if would_drain else 'STAY'} "
                f"(pending {inp.pending_prefill_tokens} vs drain target "
                f"{n_live}, active rung {rung}, ~{drain_s:.1f}s more "
                f"in pp to drain at {cfg.pp_prefill_tok_s:g} tok/s vs "
                f"{2 * seam_s:.1f}s of seam to leave and return); "
                f"declare {ENV_DECODE_STALL_SLO} to solve this instead",
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
            return _no(f"idle {idle_for:.1f}s < {cfg.idle_dwell_s:g}s idle dwell")
        if idle:
            return _no("idle at rest")
        return _no(f"prefilling in pp ({inp.pending_prefill_tokens} tok pending)")

    return _no(f"unknown phase {inp.phase!r}")


def observe_idle(state: PhasePolicyState, inp: PhasePolicyInputs) -> None:
    """Maintain the idle clock. Called every round, before ``decide``.

    Split out from ``decide`` so the decision stays pure and so the clock is
    driven by observation rather than by the policy's own verdict.
    """
    # #677 HOLD COUNTER, maintained HERE for the same reason the idle and
    # formation clocks are: ``decide`` is pure and must not mutate, so a bound
    # driven from inside the decision would never advance. Without this the
    # 8-round bound was a comment-claimed invariant with no wiring -- the
    # documented-but-inert class -- and every evaluation saw round 0 forever,
    # so EXHAUSTED was unreachable and the SLO cap was silently the only
    # backstop.
    #
    # FIRST SIGHT COUNTS AS ROUND ONE: `hold_phase` is empty before the first
    # observation, and treating that as a phase CHANGE would spend round one
    # resetting, so a hold would always be one round shorter than its bound
    # claims. A genuine phase change still resets, which is the case the reset
    # exists for.
    _prev_hold_phase = getattr(state, "hold_phase", "") or inp.phase
    state.hold_rounds = next_hold_rounds(
        int(getattr(state, "hold_rounds", 0) or 0),
        _prev_hold_phase,
        inp.phase,
        int(inp.pending_prefill_tokens or 0),
    )
    state.hold_phase = inp.phase

    # #861c EXISTENCE class: a fully cached backlog is still work. Reading
    # the economics number here called a box with six queued requests IDLE.
    # #861e: a box holding 7 retracted mid-flight bundles is not idle.
    idle = inp.decode_work_bs() == 0 and not inp.work_exists()
    if idle:
        if state.idle_since is None:
            state.idle_since = inp.now
    else:
        state.idle_since = None

    # #748 REFAIL: THE LOCK CLOCK, and why it is not the idle clock.
    #
    # The clause above stamps only an EMPTY box. An idle lock is the opposite
    # state -- work is present and this layout cannot run it -- so a qualifier
    # keyed on ``idle_since`` can never fire during one. #759 keyed its
    # break-even delay there and the metal signature did not move: 160 armings
    # in boot_735_nohc.log, none of them with (0 resident, 0 pending), all 160
    # taking the "unobserved" branch. This stamps the term the escape is
    # actually keyed on, so persistence becomes measurable where it is used.
    #
    # DRIVEN BY OBSERVATION, NOT BY THE VERDICT, exactly as the idle and
    # formation clocks are -- a clock the policy resets when it arms would
    # restart on every arm and could never accumulate.
    if bool(getattr(inp, "nothing_can_run", False)):
        if state.nothing_can_run_since is None:
            state.nothing_can_run_since = inp.now
    else:
        state.nothing_can_run_since = None

    # #689 FORMATION CLOCK. Stamped on the first round in which a PP window
    # holds a carrier but not yet a full one; cleared whenever the window is
    # full, the queue drains, or the phase is not PP. Kept HERE, in the
    # observer, for the same reason the idle clock is: ``decide`` is pure and
    # must not mutate, and a clock driven by the policy's own verdict would
    # restart every time the verdict changed.
    forming = (
        inp.phase == PHASE_PP
        and int(getattr(inp, "ready_carriers", 0) or 0) > 0
        and bool(getattr(inp, "queue_nonempty", False))
    )
    if forming:
        if not state.formation_started_at:
            state.formation_started_at = inp.now
    else:
        state.formation_started_at = 0.0
    # Phase-entry clock for the fairness bound. Driven by the OBSERVED
    # phase rather than by the policy's own verdict, so it is correct for
    # the first phase (no flip has happened yet, last_flip_at == 0) and for
    # a phase this policy did not choose (manual /phase_flip).
    if state.last_phase != inp.phase:
        state.last_phase = inp.phase
        state.phase_since = inp.now
        # A fresh phase inherits no stall. The clock restarts here so a wedge
        # has to be demonstrated in THIS residency, not carried in from the
        # last one.
        state.last_prefill_progress_at = inp.now
        state.last_pending_prefill_tokens = None
        # #677 hot fix 2: the bundle this window inherits. Recorded at the
        # boundary because by the time it drains there is nothing left to count.
        state.bundle_at_phase_entry = int(inp.running_bs)
        # #833: the admitted-set stall clock restarts with the phase, for the
        # same reason the prefill one does -- a wedge must be demonstrated in
        # THIS residency and never inherited from the last.
        state.last_bundle_progress_at = inp.now
        state.last_running_bs = None
    # #677(a) PREFILL PROGRESS, MEASURED. The wedge signature is pending
    # frozen at a value while every slot is held by a carried decode, so the
    # observable that separates it from a slow drain is whether the backlog
    # ever goes DOWN. Recorded here rather than derived in `decide` for the
    # same reason the idle clock is: a decision that measures its own history
    # is not reproducible from its inputs.
    prev = state.last_pending_prefill_tokens
    if prev is None or inp.pending_prefill_tokens < prev:
        state.last_prefill_progress_at = inp.now
    state.last_pending_prefill_tokens = inp.pending_prefill_tokens
    # #833 ADMITTED-SET PROGRESS, on the axis drain mode actually waits on.
    # "Progress" here is the bundle getting SMALLER. A bundle that is refilled
    # as fast as it retires is not progress toward an empty set, however much
    # work it completes -- and that distinction is the whole difference between
    # a drain that ends and one that does not.
    prev_bs = state.last_running_bs
    if prev_bs is None or int(inp.running_bs) < int(prev_bs):
        state.last_bundle_progress_at = inp.now
    state.last_running_bs = int(inp.running_bs)


def note_flip_armed(
    state: PhasePolicyState, decision: PhasePolicyDecision, now: float
) -> None:
    """Commit the dwell clock, once a decision was actually armed.

    PROVISIONAL, and the word is load-bearing. The arm travels the broadcast
    pipe and its verdict comes back later, so between dispatch and verdict
    the policy must already be holding -- otherwise every round in that gap
    arms again. So the dwell is committed here and the displaced value is
    remembered in ``pending_arm``; ``note_flip_outcome`` either keeps it (the
    flip happened) or puts it back (it did not).
    """
    state.pending_arm = (decision.direction, state.last_flip_at, state.idle_since)
    state.last_flip_at = now
    state.idle_since = None
    state.flips_armed += 1
    state.last_reason = decision.reason


def note_flip_outcome(
    cfg: PhasePolicyConfig,
    state: PhasePolicyState,
    direction: Optional[str],
    ok: bool,
    message: str,
    now: float,
) -> None:
    """Feed the ARM VERDICT back into the policy. The missing return path.

    WHAT WAS BROKEN (#656 boot E, 2026-08-12). ``decide`` is pure and
    documents that "a decision which is refused downstream (guards, a losing
    consensus) does not leave the policy believing it flipped" -- but nothing
    ever told it about the refusal. ``handle_phase_flip`` has ``(ok, msg)``
    in hand and, for an INTERNAL request (the policy's own), returned None
    without recording either. The policy therefore saw only its own dispatch,
    held for ``min_dwell_s``, and armed again -- 179 times, against a seam the
    runtime had already declared unfundable and installed a blocking guard
    for. Under strict phase purity the pending prefill could not run in TP, so
    the instance answered /health and produced no tokens.

    A refusal is therefore recorded three ways, each covering a different
    failure of the old code:

    * the dwell rollback, so a refused arm is not mistaken for a flip;
    * a growing hold, so the retry rate is not the dwell rate;
    * a degradation, so a refusal that keeps repeating is NAMED with the
      runtime's own numbers instead of accumulating as log noise.

    NOT A LATCH. The degraded direction is still re-probed at the capped
    interval: staging is a function of the live set as well as of the layer
    map, and an occupancy trough or a landed spill can fund a seam that was
    short a moment ago. What must never happen again is the SILENT spin, not
    the retry.
    """
    if direction is None:
        return
    if ok:
        # ACCEPTED IS NOT COMPLETED, and conflating the two is the half of
        # this defect that survives the obvious fix. ``arm`` returning True
        # only means no guard blocked the request; the seam is priced rounds
        # later and may abandon, which is how boot E burnt its first eight
        # attempts per episode with the policy believing all was well. So the
        # arm stays OUTSTANDING here -- ``note_flip_completed`` retires it,
        # and an abandon reported through this same function retires it as a
        # refusal, with the dwell rolled back.
        return
    pending = state.pending_arm
    state.pending_arm = None

    # Refused: put back what the provisional commit displaced. Only if the
    # verdict belongs to the arm still outstanding -- an out-of-order or
    # duplicated verdict must not rewind a dwell some LATER flip earned.
    if pending is not None and pending[0] == direction:
        state.last_flip_at = pending[1]
        state.idle_since = pending[2]
        state.flips_armed = max(0, state.flips_armed - 1)

    # An outcome of False means a staging attempt was made and ABANDONED --
    # the expensive failure, the one with a memory peak at the seam. Arms that
    # never reach staging do not land here, which is what lets the storm guard
    # be paced on real cost instead of on probe count.
    state.last_abandon_at[direction] = now
    k = state.arm_refusals.get(direction, 0) + 1
    state.arm_refusals[direction] = k
    state.arm_refusals_total += 1
    hold = min(cfg.min_dwell_s * (2**k), cfg.refusal_backoff_cap_s)
    state.arm_hold_until[direction] = now + hold
    # The span, so a persisting arming condition can pace from the refusal
    # instant rather than from the end of a backoff it is going to ignore.
    state.arm_hold_last_span[direction] = hold
    detail = (message or "no reason given").strip()
    # Kept unconditionally, so the purity valve can name the shortfall that
    # idled the box even on the FIRST abandon -- which, since the stand-down
    # bound became 1, is the abandon that opens it.
    state.arm_last_detail[direction] = detail

    if k >= cfg.refusal_degrade_after and not state.arm_degraded.get(direction):
        state.arm_degraded[direction] = detail
        state.arm_degrade_events += 1
        logger.error(
            "%s %s REFUSED %d times consecutively and is being treated as "
            "unfundable: %s. Serving continues in the current layout; under "
            "strict phase purity that means the work only the other layout "
            "can do will NOT drain. Re-probing every %gs in case the live "
            "set changes.",
            LOG_PREFIX,
            direction,
            k,
            detail,
            cfg.refusal_backoff_cap_s,
        )
    else:
        logger.warning(
            "%s %s arm refused (%d in a row), holding %.1fs: %s",
            LOG_PREFIX,
            direction,
            k,
            hold,
            detail,
        )


def note_flip_completed(
    cfg: PhasePolicyConfig,
    state: PhasePolicyState,
    direction: Optional[str],
    now: float,
) -> None:
    """A CUTOVER actually happened. The only event that clears a refusal.

    The distinction this enforces is the one #656 was measured against:
    boot E logged 179 arms and 0 completed cutovers, and every summary that
    counted arms as flips read that boot as healthy. A flip is completed
    bytes, not an accepted request.
    """
    if direction is None:
        return
    state.pending_arm = None
    state.last_flip_at = now
    state.idle_since = None
    # #748: THE LAYOUT CHANGED, so a lock measured in the layout that is gone
    # is not evidence about the one that replaced it. Cleared here and NOT in
    # ``note_flip_armed``, because an ARM is a request and the old layout is
    # still up until the cutover commits.
    state.nothing_can_run_since = None
    state.flips_completed += 1
    state.arm_refusals.pop(direction, None)
    state.arm_hold_until.pop(direction, None)
    state.last_abandon_at.pop(direction, None)
    if state.arm_degraded.pop(direction, None):
        logger.warning(
            "%s %s completed a cutover and is fundable again after being "
            "declared unfundable; the flip has resumed",
            LOG_PREFIX,
            direction,
        )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise PhasePolicyError(f"{name}={raw!r} is not a number") from exc


def _env_flag(name: str, default: bool) -> bool:
    """A boolean knob. Unset keeps the default, which for every flag added
    after a deployment exists must be the deployment's current behaviour."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise PhasePolicyError(f"{name}={raw!r} is not an integer") from exc


#: #677 hot fix 2: opt in to "prefill until empty, decode the bundle to
#: completion, prefill again". Off unless set, so no deployment moves without
#: asking for it.
ENV_DRAIN_MODE = "SGLANG_PHASE_POLICY_DRAIN_MODE"
ENV_DRAIN_MODE_STRICT = "SGLANG_PHASE_POLICY_DRAIN_MODE_STRICT"


#: How a knob's value was arrived at, in the words the boot log prints.
#: #896: an EFFECTIVE value with no printed provenance is a silent knob. The
#: decode-stall SLO booted at 180 s and governed a real cutover while the only
#: trace of where 180 came from was the 27 KB ``ServerArgs`` repr -- readable in
#: principle, unfindable in practice, so two follow-up tickets hung their
#: semantics on a number nobody could source. These three words are the whole
#: fix: they say FLAG, ENV or DEFAULT at the point the value starts governing.
#:
#: #901: re-exported from the authority rather than redefined here. Four
#: modules grew this vocabulary independently in one day; one definition is
#: the point of that ticket.
PROVENANCE_DEFAULT = _knob.PROVENANCE_DEFAULT


def _env_source(env_name: str) -> str:
    """Provenance for a knob that has no CLI flag yet: the env var, or the default.

    An env var set to the empty string is NOT a source -- ``_env_float`` and
    friends fall through to the default for it, so reporting "env" there would
    name a source that did not supply the value. That rule now lives in
    ``knob_resolution.env_present_nonempty`` (#901), beside the other presence
    rule (``is not None``, which #894 S5's site needs) so the difference
    between them is visible in one place instead of being re-decided per
    module.
    """
    return _knob.env_provenance(env_name)


def _flag_or_env(
    server_args,
    field: str,
    env_name: str,
    env_reader,
    default,
    record: Optional[Dict[str, str]] = None,
):
    """Resolve one knob: the CLI FLAG wins, the env var is a deprecated bridge.

    #781. These knobs used to come from the environment only. The boot env was
    assembled by concatenating a captured shell environment, a heredoc and
    EXTRA_ENV, which silently resolved a key written twice as "last one wins" --
    SGLANG_PHASE_POLICY_TP_DECODE_FLOOR_S was 10 in one half and 8 in the other,
    and the 10 had been dead the whole time with nobody aware.

    So the flag is authoritative. The env is still read when the flag is unset,
    which keeps every existing deployment byte-identical, and warns so the
    remaining users are visible instead of silent.

    ``record``, when supplied, collects the PROVENANCE of every knob resolved
    through here, keyed by ``field`` (#896).

    #901: THE LADDER IS NOW DECLARED, NOT HAND-WALKED. The three rungs below
    say the same thing the if/elif chain said -- flag, then a non-empty env,
    then the default -- but they say it in the vocabulary
    ``knob_resolution.resolve_knob`` shares with the other three reporters.
    The ORDER is still this site's own: flag-over-env is #781's deliberate
    promotion, and the authority takes a precedence order as a parameter
    precisely so it never has to have an opinion about one.

    The env and default rungs both read through ``env_reader``, lazily, which
    is byte-identical to the old tail (it returned ``env_reader(env_name,
    default)`` for both) and additionally means a losing rung is never parsed.
    """
    flag_value = getattr(server_args, field, None) if server_args is not None else None

    def _read():
        return env_reader(env_name, default)

    resolution = _knob.resolve_knob(
        [
            _knob.KnobSource(
                source=_knob.flag_source(field),
                value=flag_value,
                present=flag_value is not None,
                kind=_knob.KIND_FLAG,
            ),
            _knob.KnobSource(
                source=_knob.env_source(env_name),
                present=_knob.env_present_nonempty(env_name),
                reader=_read,
                kind=_knob.KIND_ENV,
            ),
            _knob.KnobSource(
                source=_knob.PROVENANCE_DEFAULT,
                present=True,
                reader=_read,
                kind=_knob.KIND_DEFAULT,
            ),
        ]
    )
    if record is not None:
        record[field] = resolution.source
    if resolution.winner.kind == _knob.KIND_ENV:
        try:
            from sglang.srt.environ import _warn_deprecated_env_to_cli_flag

            _warn_deprecated_env_to_cli_flag(env_name, "--" + field.replace("_", "-"))
        except Exception:
            # A missing/renamed helper must never cost a boot: the warning is
            # advisory, the value is what matters.
            pass
    return resolution.value


def config_from_env(
    enabled: bool,
    chunk_tokens: int = 0,
    formation_target: int = 0,
    server_args=None,
) -> PhasePolicyConfig:
    """Build the boot configuration.

    ``enabled`` comes from the server arg. The tuning knobs now come from
    SERVER ARGS FIRST (#781) and fall back to the environment only when the
    corresponding flag is unset -- see ``_flag_or_env``. ``server_args`` is
    optional so the many existing unit tests that build a config without a
    ServerArgs keep working unchanged; when it is None every knob resolves
    exactly as it did before.
    """
    # #896: every knob resolved below records WHERE its value came from, so the
    # arming block can print it. Filled as a side effect of the resolution
    # itself -- a provenance re-derived after the fact would be a second
    # opinion about the first one, and those drift.
    prov: Dict[str, str] = {}
    rest_state = os.environ.get(ENV_REST_STATE) or REST_DECODE
    min_dwell = _flag_or_env(
        server_args,
        "phase_policy_min_dwell_s",
        ENV_MIN_DWELL,
        _env_float,
        DEFAULT_MIN_DWELL_S,
        record=prov,
    )
    idle_dwell = _env_float(ENV_IDLE_DWELL, DEFAULT_IDLE_DWELL_S)
    prov["idle_dwell_s"] = _env_source(ENV_IDLE_DWELL)

    # THE THREE MEASUREMENTS THE WHOLE LADDER RESTS ON, resolved from the
    # environment ONCE and then used everywhere -- for the break-even N, for
    # the stranded-decode surcharge, and for the boot log.
    #
    # This used to read the module CONSTANTS here while the surcharge read the
    # env, so the two halves of one ladder were solved against different
    # numbers. Booting with a measured 5.918 s seam produced
    # [7004, 19430, 21589, 22669, 23316]: rung 0 did not move at all, because
    # the seam knob never reached it. A solved number silently replaced by a
    # constant is exactly the provenance defect this policy exists to avoid.
    # #677: SEED, then measure. The env/constant is what the policy uses until
    # the seam reports a real flip. The live harvest boot priced 22-24 s flips
    # at 3.2 s and derived N=7004 against a true break-even of ~49,250 --
    # 7.03x too cheap.
    #
    # #777, THE HALF THIS COMMENT USED TO OVERSTATE. It said "from the first
    # measurement on, the estimate wins". The ESTIMATOR does follow the
    # measurements -- it is process-global and `observe_flip_cost` feeds it
    # after every real flip. N does NOT. `config_from_env` has exactly one
    # caller (`Scheduler.__init__`), `flip_tokens` is assigned in exactly one
    # place (the config built just below), and the only `dataclasses.replace`
    # on the live config touches `decode_contention`. So the seam value read
    # on the next line is whatever the estimator held AT BOOT -- necessarily
    # the seed, because no flip has happened yet -- and it is frozen there for
    # the life of the process. A calibrated estimator never reaches N.
    #
    # This is deliberately NOT fixed by repricing N here: moving N from 7004
    # to a measured ~49,250 is a policy change with a 7x blast radius on when
    # the server flips at all, and that decision is the planner's. What is
    # fixed is the silence: `observe_flip_cost` now SAYS that N has gone stale
    # and by how much, instead of the estimator quietly tracking a number
    # nothing consumes.
    global _FLIP_COST_ESTIMATOR
    _seed_s = _env_float(ENV_FLIP_COST_S, DEFAULT_FLIP_COST_S)
    if _FLIP_COST_ESTIMATOR is None or _FLIP_COST_ESTIMATOR.seed_s != _seed_s:
        _FLIP_COST_ESTIMATOR = RoundTripFlipCost(seed_s=_seed_s)
    seam_s = _FLIP_COST_ESTIMATOR.value()
    # The seam has TWO provenances and both matter: where the SEED came from,
    # and whether the estimator is still sitting on that seed or has since
    # measured a real flip. Printing only the first would read as "measured".
    prov["flip_cost_s"] = (
        f"{_env_source(ENV_FLIP_COST_S)} seed, estimator {flip_cost_provenance()}"
    )
    prov["decode_strand_weight"] = _env_source(ENV_DECODE_STRAND_WEIGHT)
    tp_tok_s = _env_float(ENV_TP_TOK_S, DEFAULT_TP_PREFILL_TOK_S)
    pp_tok_s = _env_float(ENV_PP_TOK_S, DEFAULT_PP_PREFILL_TOK_S)
    # The two MEASUREMENTS N is solved from. #665-F1's defect was a solved
    # number silently replaced by a constant, so "measured or assumed" is the
    # first thing to know about either of them. Note DEFAULT_TP_PREFILL_TOK_S
    # itself reads ENV_TP_TOK_S at import time, so the env can reach this value
    # by two routes -- both of them are "env", which is what gets printed.
    prov["tp_prefill_tok_s"] = _env_source(ENV_TP_TOK_S)
    prov["pp_prefill_tok_s"] = _env_source(ENV_PP_TOK_S)
    prov["pp_exit_tokens"] = _env_source(ENV_PP_EXIT_TOKENS)
    prov["refusal_backoff_cap_s"] = _env_source(ENV_REFUSAL_BACKOFF_CAP)
    prov["refusal_degrade_after"] = _env_source(ENV_REFUSAL_DEGRADE_AFTER)
    prov["rest_state"] = _env_source(ENV_REST_STATE)

    explicit = _env_int(ENV_FLIP_TOKENS, 0)
    if explicit > 0:
        flip_tokens = explicit
        source = f"{ENV_FLIP_TOKENS}={explicit}"
    elif tp_tok_s > 0:
        flip_tokens = break_even_tokens(seam_s, tp_tok_s, pp_tok_s)
        source = f"break-even {seam_s:g}s / (1/{tp_tok_s:g} - 1/{pp_tok_s:g})"
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
    # #901: flip_tokens' provenance is a DERIVATION, not a flag/env/default
    # rung -- the string above names the measurement N was priced from. It
    # goes into the same record as the rest so the provenance line can be
    # built by one loop instead of having this one field appended by hand.
    prov["flip_tokens"] = source

    # #777: remember HOW N was priced, so the first real flip can say whether
    # the measurement has left it behind. Recorded for the explicit case too --
    # `repriced_flip_tokens` needs to know that N is an operator assertion
    # rather than a derivation, which is a different reason to stay quiet than
    # "no measurement yet".
    note_flip_tokens_pricing(flip_tokens, tp_tok_s, pp_tok_s, explicit > 0)

    # W30: can the layout a cutover flips INTO re-admit the residents that
    # same cutover retracts? Derived from the purity contract, once, at boot:
    #   * a mode that allows prefill in TP outright re-admits trivially;
    #   * under strict/threshold the seam-transport exemption
    #     (`phase_purity.seam_transport_exempt`) covers exactly this
    #     population, so it is available too.
    # It is therefore True for every mode this tree ships. It is computed and
    # passed rather than hard-coded because the tp-ward arm now DEPENDS on it
    # (see the DRAINED branch), and a future mode that cannot re-admit must
    # make that arm stand down rather than reproduce W30's 150-flip livelock
    # silently. Static config, so identical on every rank.
    seam_readmit_available = True

    cfg = PhasePolicyConfig(
        enabled=enabled,
        seam_readmit_available=seam_readmit_available,
        drain_mode=_flag_or_env(
            server_args,
            "phase_policy_drain_mode",
            ENV_DRAIN_MODE,
            _env_flag,
            False,
            record=prov,
        )
        or _flag_or_env(
            server_args,
            "phase_policy_drain_mode_strict",
            ENV_DRAIN_MODE_STRICT,
            _env_flag,
            False,
            record=prov,
        ),
        drain_mode_strict=_flag_or_env(
            server_args,
            "phase_policy_drain_mode_strict",
            ENV_DRAIN_MODE_STRICT,
            _env_flag,
            False,
            record=prov,
        ),
        flip_tokens=flip_tokens,
        min_dwell_s=min_dwell,
        idle_dwell_s=idle_dwell,
        rest_state=rest_state,
        pp_window_s=_flag_or_env(
            server_args,
            "phase_policy_pp_window_s",
            ENV_PP_WINDOW,
            _env_float,
            DEFAULT_PP_WINDOW_S,
            record=prov,
        ),
        tp_decode_floor_s=_flag_or_env(
            server_args,
            "phase_policy_tp_decode_floor_s",
            ENV_TP_FLOOR,
            _env_float,
            DEFAULT_TP_DECODE_FLOOR_S,
            record=prov,
        ),
        # #689: the caller passes max_running_requests; the env can override
        # it, and 0/1 disables the gate and restores the previous behaviour.
        formation_target=int(
            _env_float("SGLANG_PHASE_FORMATION_TARGET", float(formation_target))
        ),
        formation_cap_s=_env_float("SGLANG_PHASE_FORMATION_CAP_S", 3.0),
        # The seam cost the threshold was derived from, carried so the
        # stranded-decode surcharge can price a cutover against the
        # generations it would pause. Without this the surcharge is inert.
        flip_cost_s=seam_s,
        decode_strand_weight=_env_float(ENV_DECODE_STRAND_WEIGHT, 1.0),
        # The measured counterfactual. With it the surcharge prices a flip
        # against what NOT flipping costs the same decodes; without it the
        # old one-sided form is kept, unchanged.
        decode_contention=_flag_or_env(
            server_args,
            "phase_policy_decode_contention",
            ENV_DECODE_CONTENTION,
            _env_float,
            DEFAULT_DECODE_CONTENTION,
            record=prov,
        ),
        decode_stall_slo_s=_flag_or_env(
            server_args,
            "phase_policy_decode_stall_slo_s",
            ENV_DECODE_STALL_SLO,
            _env_float,
            DEFAULT_DECODE_STALL_SLO_S,
            record=prov,
        ),
        pp_exit_tokens=_env_int(ENV_PP_EXIT_TOKENS, DEFAULT_PP_EXIT_TOKENS),
        # Passed in rather than read from env: it is a runtime fact of THIS
        # scheduler, and the armed line below prices the seam from it -- so it
        # has to be known before that line is emitted, not filled in after.
        prefill_chunk_tokens=int(chunk_tokens or 0),
        tp_prefill_tok_s=tp_tok_s,
        pp_prefill_tok_s=pp_tok_s,
        refusal_backoff_cap_s=_env_float(
            ENV_REFUSAL_BACKOFF_CAP, DEFAULT_REFUSAL_BACKOFF_CAP_S
        ),
        refusal_degrade_after=_env_int(
            ENV_REFUSAL_DEGRADE_AFTER, DEFAULT_REFUSAL_DEGRADE_AFTER
        ),
    )
    if enabled:
        logger.warning(
            "%s armed: N=%d tok (%s), min dwell %gs, idle dwell %gs, "
            "pp window %gs, tp decode floor %gs, seam %gs, strand weight %g, "
            "decode contention %g, N ladder by decoding reqs %s, "
            "effective pp exit %s %gs, resting layout %s (%s)",
            LOG_PREFIX,
            cfg.flip_tokens,
            source,
            cfg.min_dwell_s,
            cfg.idle_dwell_s,
            cfg.pp_window_s,
            cfg.tp_decode_floor_s,
            cfg.flip_cost_s,
            cfg.decode_strand_weight,
            cfg.decode_contention,
            # The whole ladder, not just the one-decode rung: an unreachable
            # top rung is exactly the defect #665-F1 was, and it is invisible
            # if only the first rung is logged.
            [effective_flip_threshold(cfg, b) for b in range(5)],
            # #889: `pp window` above is the REQUESTED knob, and a declared
            # decode-stall SLO makes it unreachable without touching it. The
            # same line therefore has to carry the bound that actually ends the
            # PP residency, or the one artifact an operator reads keeps
            # reporting a number the policy no longer uses. This pair (15s
            # requested, 173.6s effective) is the whole of #889.
            *effective_pp_exit_term(cfg),
            cfg.rest_phase,
            cfg.rest_state,
        )
        # #889: and when the two disagree, say so in its own line rather than
        # leaving it to be spotted in a 14-field one. See
        # `superseded_pp_bound_warning` for why this warns instead of refusing.
        superseded = superseded_pp_bound_warning(cfg)
        if superseded:
            logger.warning("%s", superseded)
        # #896: the SECOND line, and the reason this ticket exists. The line
        # above prints VALUES; this one prints where each value came from, and
        # it carries `decode_stall_slo_s` -- which the line above never printed
        # at all, so a 180 s cap governed a live cutover with no boot-log trace
        # of either its value or its origin. Kept as its own line rather than
        # widened into the one above: the armed line is grepped by name in
        # standing runsheets, and provenance is a different question from
        # arming. Every knob, not just the SLO -- a per-knob exception is how
        # the next one goes silent.
        #
        # #901: the field and the line are built by the authority
        # (``knob_resolution.provenance_field`` / ``provenance_line``), so the
        # separator rule and the ``name=<value> from <source>`` grammar have
        # one definition for all four migrated reporters. The knob table stays
        # here -- WHICH knobs govern a phase cutover is this module's fact, not
        # the authority's.
        logger.warning(
            "%s",
            _knob.provenance_line(
                LOG_PREFIX,
                [
                    _knob.provenance_field(
                        name, value, prov.get(key, PROVENANCE_DEFAULT), fmt
                    )
                    for name, key, value, fmt in (
                        (
                            "min_dwell_s",
                            "phase_policy_min_dwell_s",
                            cfg.min_dwell_s,
                            "g",
                        ),
                        ("idle_dwell_s", "idle_dwell_s", cfg.idle_dwell_s, "g"),
                        (
                            "pp_window_s",
                            "phase_policy_pp_window_s",
                            cfg.pp_window_s,
                            "g",
                        ),
                        (
                            "tp_decode_floor_s",
                            "phase_policy_tp_decode_floor_s",
                            cfg.tp_decode_floor_s,
                            "g",
                        ),
                        ("flip_cost_s", "flip_cost_s", cfg.flip_cost_s, "g"),
                        (
                            "decode_strand_weight",
                            "decode_strand_weight",
                            cfg.decode_strand_weight,
                            "g",
                        ),
                        (
                            "decode_contention",
                            "phase_policy_decode_contention",
                            cfg.decode_contention,
                            "g",
                        ),
                        (
                            "decode_stall_slo_s",
                            "phase_policy_decode_stall_slo_s",
                            cfg.decode_stall_slo_s,
                            "g",
                        ),
                        (
                            "tp_prefill_tok_s",
                            "tp_prefill_tok_s",
                            cfg.tp_prefill_tok_s,
                            "g",
                        ),
                        (
                            "pp_prefill_tok_s",
                            "pp_prefill_tok_s",
                            cfg.pp_prefill_tok_s,
                            "g",
                        ),
                        ("pp_exit_tokens", "pp_exit_tokens", cfg.pp_exit_tokens, "g"),
                        (
                            "refusal_backoff_cap_s",
                            "refusal_backoff_cap_s",
                            cfg.refusal_backoff_cap_s,
                            "g",
                        ),
                        (
                            "refusal_degrade_after",
                            "refusal_degrade_after",
                            cfg.refusal_degrade_after,
                            "g",
                        ),
                        # flip_tokens is a count and rest_state a string; the
                        # per-field format spec is what lets them ride the same
                        # builder instead of being appended by hand, which is
                        # how they were carried before #901.
                        ("flip_tokens", "flip_tokens", cfg.flip_tokens, "d"),
                        ("rest_state", "rest_state", cfg.rest_state, ""),
                    )
                ],
            ),
        )
    return cfg


__all__ = [
    "PHASE_PP",
    "PHASE_TP",
    "PP_TO_TP",
    "TP_TO_PP",
    "prefill_suppressed_in_tp",
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
    # #889 -- the effective PP exit term and the supersession announcement.
    "PP_EXIT_BY_DRAIN",
    "PP_EXIT_BY_SLO_CAP",
    "PP_EXIT_BY_STOPWATCH",
    "effective_pp_exit_term",
    "superseded_pp_bound_warning",
    "decide",
    "note_flip_armed",
    "note_flip_completed",
    "note_flip_outcome",
    "observe_idle",
]
