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
saving by construction -- N is precisely the point where it is. Below N the
prompt never wanted PP anyway and now prefills in TP without any flip at all.

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


def effective_flip_threshold(cfg: "PhasePolicyConfig", running_bs: int) -> int:
    """Pending-prefill tokens required to justify `tp_to_pp` RIGHT NOW.

    ``break_even_tokens`` prices the seam and nothing else. A cutover also
    PAUSES every request that is decoding: it is carried into the PP layout,
    where decode is forbidden, and waits out the PP window before emitting
    another token. Stranding is therefore a real cost of the flip, it scales
    with how many requests are in flight, and it belongs in the same
    comparison rather than in a separate ad-hoc guard.

    The seconds go into the same C that produced N, so the threshold simply
    scales:

        C_eff = flip_cost_s + weight x running_bs x pp_window_s
        N_eff = flip_tokens x C_eff / flip_cost_s

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
    base = int(cfg.flip_tokens)
    if cfg.flip_cost_s <= 0 or running_bs <= 0:
        return base
    if cfg.decode_contention > 0.0:
        return _differential_flip_threshold(cfg, running_bs, base)
    if cfg.decode_strand_weight <= 0:
        return base
    stranded_s = cfg.decode_strand_weight * float(running_bs) * cfg.pp_window_s
    return int(round(base * (cfg.flip_cost_s + stranded_s) / cfg.flip_cost_s))


#: How many chunk-cadences of silence make a stall a WEDGE rather than a slow
#: tick. Small on purpose: the rule already refuses to fire while anything is
#: moving, so its only job is to outlast ordinary jitter between chunks.
PROGRESS_STALL_CHUNKS = 3


def pp_progress_stall_window_s(cfg: "PhasePolicyConfig") -> float:
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


def prefill_suppressed_in_tp(cfg: "PhasePolicyConfig", phase: str) -> bool:
    """#677 hot fix 2: may prefill be admitted while the TP layout is up?

    Under drain mode, no. The TP window exists to finish a decode bundle, and
    prefill admitted into it competes with exactly the work the window was
    entered for -- measured as visible "Prefill batch" lines inside TP while
    carriers went unfinished across repeated flips.

    CARDS PARTIALLY IDLE DURING TP IS ACCEPTED, deliberately, and is the
    user's stated model: the alternative measured worse, because a bundle that
    never finishes costs a whole extra round trip and returns the same
    carriers.

    False for every phase but TP, and False everywhere when drain mode is off,
    so a deployment that has not opted in is byte-identical.
    """
    return bool(cfg.drain_mode) and phase == PHASE_TP


def pp_residency_cap_s(cfg: "PhasePolicyConfig") -> float:
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


def solved_tp_decode_floor_s(cfg: "PhasePolicyConfig") -> float:
    """Minimum TP dwell, SOLVED: one full round trip.

    A cycle that spends less time serving decode in TP than it just spent
    entering and leaving TP has put more than half its wall clock into
    cutovers. `2 * flip_cost_s` is therefore the floor below which the phase
    machinery costs more than it moves -- measured, not chosen.
    """
    return 2.0 * cfg.flip_cost_s


def with_decode_contention(
    cfg: "PhasePolicyConfig", value: object
) -> "PhasePolicyConfig":
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
    cfg: "PhasePolicyConfig",
    inp: "PhasePolicyInputs",
    state: "PhasePolicyState",
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
    waited_s = min(base_s * (2 ** k), cfg.refusal_backoff_cap_s)
    return saved_s > waited_s


def _differential_flip_threshold(
    cfg: "PhasePolicyConfig", running_bs: int, base: int
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
    the prefill only pays the way in. ``min(W, N/P)`` because the PP window
    bounds how long the instance may sit in PP: past ``W x P`` tokens the
    prefill no longer fits one window, the decodes are released at ``W``, and
    the remainder waits for the next window.

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
    cost = cfg.flip_cost_s * (1.0 + 2.0 * b)

    # Solved from C, X and P DIRECTLY rather than by substituting
    # N0 = C / (1/X - 1/P) and cancelling. The substitution is only valid
    # while `flip_tokens` is exactly the break-even of THESE three numbers,
    # and it need not be: `config_from_env` derives the default N from the
    # module constant DEFAULT_FLIP_COST_S while `flip_cost_s` itself is
    # overridable, and an operator may pin `flip_tokens` outright. Deriving
    # from the measurements makes the surcharge independent of that, and
    # `base` survives only as the floor it should always have been.
    den_a = (1.0 + b * sigma) / tp_tok_s - (1.0 + b) / pp_tok_s

    if cfg.pp_window_s <= 0.0:
        # NO WINDOW BOUND. The PP phase is not time-limited, so a carried
        # decode waits out the whole prefill: the stall is 2C + N/P and
        # regime A is the ONLY regime. Regime B must not be reached here --
        # it models the decode charge SATURATING at W, and with W = 0 that
        # reads as "stranding is free", the opposite of what an absent
        # window means. Falling through to it produced a threshold that
        # spiked toward the singularity at den_a -> 0+ and then dropped by
        # half a million tokens once den_a went negative.
        if den_a <= 0.0:
            # No positive N satisfies the inequality: with this little
            # contention and this many decodes, flipping does not repay at
            # ANY backlog. That is a real answer, not a failure, and it is
            # reached only when the operator has disabled the fairness
            # window that would otherwise bound the stranding.
            return UNREACHABLE_FLIP_THRESHOLD
        return max(base, int(round(cost / den_a)))

    if den_a > 0.0:
        n_a = cost / den_a
        # Regime A holds only where the prefill fits inside one PP window.
        if n_a <= cfg.pp_window_s * pp_tok_s:
            return max(base, int(round(n_a)))

    # Beyond the window the decode charge saturates at W, which makes this
    # denominator strictly positive (sigma >= 0 and ratio < 1), so the
    # threshold is always solvable and always finite.
    den_b = (1.0 + b * sigma) / tp_tok_s - 1.0 / pp_tok_s
    return max(base, int(round((cost + b * cfg.pp_window_s) / den_b)))


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
    #: #677 hot fix 2: how many requests were decoding when this TP window
    #: opened, so the exit receipt can name the BUNDLE it finished rather than
    #: only the instant it ended. Set by ``observe_idle`` at the phase change.
    bundle_at_phase_entry: int = 0
    #: Cutovers that actually MOVED BYTES. Distinct from ``flips_armed`` on
    #: purpose: boot E armed 179 flips and completed none, and every summary
    #: that read the arm count called that instance healthy.
    flips_completed: int = 0


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
            solved_tp_decode_floor_s(cfg)
            if cfg.flip_cost_s > 0
            else cfg.min_dwell_s
        )
        floor_s = min(base_s * (2 ** k), cfg.refusal_backoff_cap_s)
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


def _last_refusal_at(state: "PhasePolicyState", direction: str) -> float:
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

    idle = inp.running_bs == 0 and inp.pending_prefill_tokens == 0

    # The minimum dwell is checked FIRST and applies to every flip, so no
    # branch below can bypass the thrash bound. It is the only guarantee
    # that survives an adversarial arrival pattern.
    since_flip = inp.now - state.last_flip_at
    if state.last_flip_at > 0 and since_flip < cfg.min_dwell_s:
        return _no(
            f"min dwell: {since_flip:.1f}s since last flip < {cfg.min_dwell_s:g}s"
        )

    if inp.phase == PHASE_TP:
        # Enough queued prefill to repay the round trip -> go to the prefill
        # layout. Below the threshold the prefill runs in TP as-is, which is
        # the correct answer for short prompts by construction of N.
        # Under purity the break-even is meaningless (see
        # prefill_runs_in_tp): the only threshold that terminates is zero.
        tp_threshold = effective_flip_threshold(cfg, inp.running_bs)
        if (
            cfg.prefill_runs_in_tp
            and inp.running_bs > 0
            and cfg.flip_tokens < inp.pending_prefill_tokens <= tp_threshold
        ):
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
                    f"{cfg.flip_tokens} but <= {tp_threshold} with "
                    f"{inp.running_bs} req decoding: too short for the round "
                    f"trip to beat prefilling it in tp"
                )
            return _no(
                f"pending prefill {inp.pending_prefill_tokens} tok > N="
                f"{cfg.flip_tokens} but <= {tp_threshold} with "
                f"{inp.running_bs} req decoding: flipping would strand them "
                f"in pp for a {cfg.pp_window_s:g}s window"
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
                return _no(
                    f"decode bundle running: {inp.running_bs} of "
                    f"{max(state.bundle_at_phase_entry, inp.running_bs)} req "
                    f"still decoding, {inp.pending_prefill_tokens} tok prefill "
                    f"waiting -- drain mode finishes the bundle before flipping"
                )
            in_phase = (
                None if state.phase_since is None else inp.now - state.phase_since
            )
            bundle = max(state.bundle_at_phase_entry, 0)
            return PhasePolicyDecision(
                TP_TO_PP,
                f"decode bundle complete: {bundle} reqs decoded in "
                f"{0.0 if in_phase is None else in_phase:.1f} s "
                f"({inp.pending_prefill_tokens} tok prefill waiting) -- exit "
                f"condition: decode drained",
            )
        if inp.pending_prefill_tokens > tp_threshold:
            # THE DECODE FLOOR. Under purity every token of prefill has to
            # wait for a PP window, so the backlog is essentially always
            # above N and this rule would otherwise fire the instant
            # min_dwell expires -- giving decode a 3 s window per cycle and
            # starving it in the mirror image of the defect the PP window
            # fixes. Only applies while decode work actually exists: with
            # nothing decoding there is nothing to protect, and an arriving
            # long prompt should reach the PP layout inside its TTFT.
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
            return PhasePolicyDecision(
                TP_TO_PP,
                f"pending prefill {inp.pending_prefill_tokens} tok > "
                f"{'N=' + str(cfg.flip_tokens) if cfg.prefill_runs_in_tp else '0 (purity: prefill cannot run in tp)'}",
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
        if inp.pending_prefill_tokens:
            # Unreachable under purity: tp_threshold is 0 there, so any
            # non-zero pending already armed above. Reaching it would mean
            # the deadlock of 21:39:50Z had returned.
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
        if inp.pending_prefill_tokens <= cfg.pp_exit_tokens and inp.running_bs > 0:
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
            rung = effective_flip_threshold(cfg, inp.running_bs)
            would_drain = inp.pending_prefill_tokens <= cfg.flip_tokens
            drain_s = (
                inp.pending_prefill_tokens / cfg.pp_prefill_tok_s
                if cfg.pp_prefill_tok_s > 0
                else float("nan")
            )
            return PhasePolicyDecision(
                PP_TO_TP,
                f"pp window {in_pp:.1f}s >= {cfg.pp_window_s:g}s with "
                f"{inp.running_bs} req waiting to decode "
                f"({inp.pending_prefill_tokens} tok prefill deferred to the "
                f"next pp window) -- HAND-SET STOPWATCH; drain-based policy "
                f"would {'also leave' if would_drain else 'STAY'} "
                f"(pending {inp.pending_prefill_tokens} vs drain target "
                f"{cfg.flip_tokens}, active rung {rung}, ~{drain_s:.1f}s more "
                f"in pp to drain at {cfg.pp_prefill_tok_s:g} tok/s vs "
                f"{2 * cfg.flip_cost_s:.1f}s of seam to leave and return); "
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
        # A fresh phase inherits no stall. The clock restarts here so a wedge
        # has to be demonstrated in THIS residency, not carried in from the
        # last one.
        state.last_prefill_progress_at = inp.now
        state.last_pending_prefill_tokens = None
        # #677 hot fix 2: the bundle this window inherits. Recorded at the
        # boundary because by the time it drains there is nothing left to count.
        state.bundle_at_phase_entry = int(inp.running_bs)
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


def config_from_env(
    enabled: bool, chunk_tokens: int = 0
) -> PhasePolicyConfig:
    """Build the boot configuration.

    ``enabled`` comes from the server arg; the tuning knobs come from the
    environment so a running deployment can be re-tuned without a code
    change, and so the acceptance run can exercise a short dwell without
    shipping a short dwell as the default.
    """
    rest_state = os.environ.get(ENV_REST_STATE) or REST_DECODE
    min_dwell = _env_float(ENV_MIN_DWELL, DEFAULT_MIN_DWELL_S)
    idle_dwell = _env_float(ENV_IDLE_DWELL, DEFAULT_IDLE_DWELL_S)

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
    seam_s = _env_float(ENV_FLIP_COST_S, DEFAULT_FLIP_COST_S)
    tp_tok_s = _env_float(ENV_TP_TOK_S, DEFAULT_TP_PREFILL_TOK_S)
    pp_tok_s = _env_float(ENV_PP_TOK_S, DEFAULT_PP_PREFILL_TOK_S)

    explicit = _env_int(ENV_FLIP_TOKENS, 0)
    if explicit > 0:
        flip_tokens = explicit
        source = f"{ENV_FLIP_TOKENS}={explicit}"
    elif tp_tok_s > 0:
        flip_tokens = break_even_tokens(seam_s, tp_tok_s, pp_tok_s)
        source = (
            f"break-even {seam_s:g}s / (1/{tp_tok_s:g} - 1/{pp_tok_s:g})"
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
        drain_mode=_env_flag(ENV_DRAIN_MODE, False),
        flip_tokens=flip_tokens,
        min_dwell_s=min_dwell,
        idle_dwell_s=idle_dwell,
        rest_state=rest_state,
        pp_window_s=_env_float(ENV_PP_WINDOW, DEFAULT_PP_WINDOW_S),
        tp_decode_floor_s=_env_float(ENV_TP_FLOOR, DEFAULT_TP_DECODE_FLOOR_S),
        # The seam cost the threshold was derived from, carried so the
        # stranded-decode surcharge can price a cutover against the
        # generations it would pause. Without this the surcharge is inert.
        flip_cost_s=seam_s,
        decode_strand_weight=_env_float(ENV_DECODE_STRAND_WEIGHT, 1.0),
        # The measured counterfactual. With it the surcharge prices a flip
        # against what NOT flipping costs the same decodes; without it the
        # old one-sided form is kept, unchanged.
        decode_contention=_env_float(
            ENV_DECODE_CONTENTION, DEFAULT_DECODE_CONTENTION
        ),
        decode_stall_slo_s=_env_float(
            ENV_DECODE_STALL_SLO, DEFAULT_DECODE_STALL_SLO_S
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
            "resting layout %s (%s)",
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
            cfg.rest_phase,
            cfg.rest_state,
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
    "decide",
    "note_flip_armed",
    "note_flip_completed",
    "note_flip_outcome",
    "observe_idle",
]
