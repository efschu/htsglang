"""#631 PHASE PURITY -- each layout runs the work it is for, when that pays.

CORRECTION 2026-08-14 (user, explicit) -- READ BEFORE THE HISTORY BELOW
----------------------------------------------------------------------
The blanket rule "NOT A SINGLE TOKEN may be prefilled in the TP layout" is
WITHDRAWN, and the record that the user ordered it as a hard rule is struck:
that instruction rested on wrong input data. Small prefills do not always
repay a seam round trip, so the default is now the SENSIBLE setup -- let the
policy's measured break-even N decide whether flipping to PP is worth it.

Measured on this rig 2026-08-14, which is what forced the correction: under a
workload of many small requests the blanket rule produced 882 flips in one
boot, arming `tp_to_pp` at 184 pending prefill tokens against a policy
break-even of N=7004, ~4.8 s of seam per request (tp_to_pp ~2.7 s + pp_to_tp
~2.1 s), and TTFT ~2.9 s on a 65-CHARACTER prompt. The flip cost dominated
everything it was supposed to accelerate.

WHAT STANDS, AND WHAT DOES NOT
- STANDS: decode in the PP layout is forbidden. That half has its own metal
  measurement (below) and is unchanged by this correction.
- WITHDRAWN: the unconditional prefill-in-TP prohibition. Prefill in TP is
  slower per token, so the policy still prefers PP -- but only once the
  pending prefill is large enough to amortise the seam, which is exactly what
  ``phase_policy.break_even_tokens`` already computes.

HISTORICAL RATIONALE (2026-08-09) -- the decode half remains valid
------------------------------------------------------------------
- Decode in the PP layout is COMPLETELY FORBIDDEN. Decode work is DEFERRED
  and executed BATCHED in the TP layout after the flip.
- (withdrawn) Prefill was likewise confined to the PP layout.

The server therefore alternates:

    accumulate prefill -> flip to PP -> run ALL pending prefill
    (new decode work queues) -> flip to TP -> run ALL deferred decode,
    batched, with graphs and speculation (new prefill queues) -> repeat

WHY, in one measurement
-----------------------
Without this rule the server pins itself in the layout that is wrong for
the work it is doing. Metal, 2026-08-09 21:15:25-21:16:15Z, live agent
traffic:

    PHASE-POLICY holding in pp: prefilling in pp (302757 tok pending,
    running bs 2)                                        x 6 samples
    same window: 87 Decode batch records, 6 Prefill batch records,
    cuda graph: False, gen throughput 35.4 tok/s

87 decode batches executed in the PREFILL layout -- no speculation, no
decode CUDA graphs, 35 tok/s -- while the policy refused to leave PP
because prefill was pending, and prefill barely advanced because decode
was taking the rounds. Each half starved the other in the same layout.
Purity removes the interleaving that makes that state reachable.

THE COST, accepted by the ordering user and stated plainly
-----------------------------------------------------------
A request mid-decode when prefill pressure arrives is PAUSED: it stays
resident, is carried across the cutover by the existing resident-carry and
draft-bootstrap machinery, and resumes -- batched -- in the next TP window.
Its inter-token latency therefore contains an entire PP window. That is a
deliberate trade of tail latency for layout-correct throughput on both
sides, not an oversight.

MODES
-----
``--phase-flip-purity``:

    prefill_in_tp   (DEFAULT since 2026-08-14) no decode in PP; prefill MAY
                    run in TP, so the policy's break-even N decides when a
                    flip to PP repays its seam. Small prefills stay in TP.
    strict          no decode in PP, no prefill in TP. The pre-2026-08-14
                    default; collapses the policy's break-even N to 0, so
                    ANY pending prefill forces a cutover.
    threshold:<n>   ESCAPE HATCH: decode may still run in the PP layout
                    while at most <n> requests are decoding. Above <n> the
                    strict rule applies again. n=0 is exactly ``strict``.
                    The prefill-in-TP prohibition is NOT relaxed by this --
                    prefill in TP is slower per token, which is why the
                    amortisation lives in the policy's break-even N rather
                    than in a second purity threshold.
    off             both prohibitions lifted: the pre-purity interleaving.
                    Kept reachable so the defect above can be reproduced
                    on demand for an A/B, not because it is a supported
                    way to run.

Anything else is a loud parse error. A silently-ignored purity setting
would serve decode from the prefill layout while the operator believes it
cannot happen, which is the exact failure this module exists to make
impossible.

THE DEADLOCK THIS OPENS, and what closes it
--------------------------------------------
Purity makes a new state reachable: the PP phase has pending prefill it
CANNOT admit (no free mamba/GDN state slot, or the KV pool is held by
paused decodes), and decode -- which would release those slots -- is
forbidden here. Nothing progresses, and the load-triggered PP->TP rule
(``pending <= N``) is false, so nothing flips either.

The closer is ``phase_policy``'s bounded PP window (``pp_window_s``): after
a bounded time in PP with decode work waiting, PP->TP arms REGARDLESS of
pending prefill. Under purity that window is not a fairness nicety -- it is
the deadlock breaker, and disabling it (0) while purity is strict is
therefore refused by ``validate_purity_policy_pair``.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-PURITY"

MODE_STRICT = "strict"
MODE_THRESHOLD = "threshold"
MODE_OFF = "off"
#: THE DEFAULT since 2026-08-14. Lifts ONLY the prefill-in-TP prohibition, so
#: the policy's break-even N governs the cutover; decode in PP stays forbidden.
MODE_PREFILL_IN_TP = "prefill_in_tp"

ENV_PURITY = "SGLANG_PHASE_FLIP_PURITY"
#: DEFAULT CHANGED 2026-08-14 (user, explicit): `prefill_in_tp`, not `strict`.
#: The blanket prefill-in-TP prohibition rested on wrong input data; with a
#: SMALL pending prefill the seam round trip is not worth paying, so the
#: sensible setup is to let the policy's break-even N decide. Decode in PP
#: remains forbidden -- that half has its own measurement and is unchanged.
DEFAULT_PURITY = MODE_PREFILL_IN_TP


class PhasePurityError(ValueError):
    """An unusable purity configuration. Raised, never defaulted around."""


@dataclass(frozen=True)
class PhasePurity:
    """The parsed rule. Immutable; built once at boot."""

    mode: str = DEFAULT_PURITY
    #: Only meaningful for ``threshold``: how many requests may decode in
    #: the PP layout before the strict prohibition applies again.
    decode_in_pp_threshold: int = 0

    @property
    def strict(self) -> bool:
        return self.mode == MODE_STRICT

    @property
    def enforced(self) -> bool:
        return self.mode != MODE_OFF

    @property
    def decode_forbidden_in_pp(self) -> bool:
        """Is decode in the PP layout impossible at EVERY batch size?

        The semantic property, as distinct from ``strict`` which is a mode
        NAME. Consumers that need "no decode ever runs in PP" -- the spill
        machinery releases the draft weights for the whole PP phase on
        exactly this guarantee -- must test this, not the name, or a mode
        that provides the guarantee by a different route is refused for no
        reason. (That is precisely what happened when `prefill_in_tp` was
        added: the spill guard compared the string and rejected a mode whose
        decode prohibition is identical to strict's.)
        """
        return self.mode in (MODE_STRICT, MODE_PREFILL_IN_TP)

    def decode_allowed_in_pp(self, running_bs: int) -> bool:
        """May a decode batch execute in the PP layout right now?

        ``running_bs`` is the number of requests that would decode. The
        threshold is on the BATCH, not on a per-request count, because the
        cost being traded is one decode step in the slow layout.
        """
        if self.mode == MODE_OFF:
            return True
        if self.mode == MODE_STRICT:
            return False
        if self.mode == MODE_PREFILL_IN_TP:
            # Explicit, not fall-through: this mode leaves
            # decode_in_pp_threshold at its 0 default, and `running_bs <= 0`
            # would then read as "a zero-sized decode batch is allowed" and
            # quietly re-admit decode into PP. Decode in PP is the half of the
            # 2026-08-09 rule the starvation measurement actually indicts, so
            # it stays forbidden here regardless of batch size.
            return False
        return int(running_bs) <= self.decode_in_pp_threshold

    def prefill_allowed_in_tp(self) -> bool:
        """May a prefill batch be BUILT in the TP layout right now?

        No threshold form HERE on purpose: the amortisation is not a second
        purity knob, it is ``phase_policy.break_even_tokens`` -- the pending
        prefill above which paying the seam beats prefilling in the slower
        layout. This method only says whether that machinery is allowed to
        run at all; ``strict`` forces it to 0 and every prefill flips.
        """
        return self.mode in (MODE_OFF, MODE_PREFILL_IN_TP)

    def describe(self) -> str:
        if self.mode == MODE_THRESHOLD:
            return f"threshold:{self.decode_in_pp_threshold}"
        return self.mode


def parse_purity(raw: Optional[str]) -> PhasePurity:
    """Parse ``strict`` / ``threshold:<n>`` / ``off``. Loud on anything else."""
    if raw is None or raw == "":
        return PhasePurity(mode=DEFAULT_PURITY)
    value = str(raw).strip().lower()
    if value == MODE_STRICT:
        return PhasePurity(mode=MODE_STRICT)
    if value == MODE_OFF:
        return PhasePurity(mode=MODE_OFF)
    if value == MODE_PREFILL_IN_TP:
        return PhasePurity(mode=MODE_PREFILL_IN_TP)
    if value.startswith(MODE_THRESHOLD):
        rest = value[len(MODE_THRESHOLD) :]
        if not rest.startswith(":"):
            raise PhasePurityError(
                f"{LOG_PREFIX} purity {raw!r} is missing its count; the "
                f"threshold mode is written 'threshold:<n>', e.g. "
                f"'threshold:2' to allow decode in the PP layout while at "
                f"most 2 requests are decoding"
            )
        try:
            n = int(rest[1:])
        except ValueError:
            raise PhasePurityError(
                f"{LOG_PREFIX} purity {raw!r} has a non-integer count "
                f"{rest[1:]!r}; write 'threshold:<n>'"
            )
        if n < 0:
            raise PhasePurityError(
                f"{LOG_PREFIX} purity threshold {n} is negative; use 0 (which "
                f"is exactly 'strict') or a positive count"
            )
        if n == 0:
            return PhasePurity(mode=MODE_STRICT)
        return PhasePurity(mode=MODE_THRESHOLD, decode_in_pp_threshold=n)
    raise PhasePurityError(
        f"{LOG_PREFIX} unknown purity {raw!r}; valid values are "
        f"'{MODE_STRICT}' (default), 'threshold:<n>', '{MODE_OFF}'"
    )


def purity_from_server_args(server_args) -> PhasePurity:
    """Boot resolution: the server arg wins, the environment is the
    fallback so an A/B needs no boot-script edit."""
    raw = getattr(server_args, "phase_flip_purity", None)
    if raw in (None, ""):
        raw = os.environ.get(ENV_PURITY)
    purity = parse_purity(raw)
    logger.warning(
        "%s mode=%s -- decode in the PP layout: %s; prefill in the TP layout: %s",
        LOG_PREFIX,
        purity.describe(),
        (
            "forbidden"
            if purity.strict
            else (
                "allowed"
                if purity.mode == MODE_OFF
                else f"allowed up to bs {purity.decode_in_pp_threshold}"
            )
        ),
        "allowed" if purity.prefill_allowed_in_tp() else "forbidden",
    )
    return purity


def validate_purity_policy_pair(purity: PhasePurity, policy_cfg) -> None:
    """Refuse the one combination that deadlocks.

    With purity enforced, the PP phase can reach a state where it may not
    decode and cannot admit prefill (no free state slot). The only exit is
    the policy's bounded PP window; without it the instance parks forever
    with both queues non-empty. Caught at boot rather than at 03:00.
    """
    if not purity.enforced:
        return
    window = float(getattr(policy_cfg, "pp_window_s", 0.0) or 0.0)
    if window > 0:
        return
    # The SOLVED equivalent (#665-F1). What this guard actually requires is a
    # BOUND on the PP residency, so that a phase which may not decode and
    # cannot admit prefill still has an exit. The hand-set stopwatch was one
    # way to supply it; a declared decode-stall SLO is another, and a better
    # one -- it bounds the same residency in units of the thing being
    # protected. Accept it, and only it: an SLO so tight that the solved cap
    # collapses to zero is no bound at all and must still be refused.
    slo = float(getattr(policy_cfg, "decode_stall_slo_s", 0.0) or 0.0)
    if slo > 0:
        seam = float(getattr(policy_cfg, "flip_cost_s", 0.0) or 0.0)
        if slo - 2.0 * seam > 0:
            return
        raise PhasePurityError(
            f"{LOG_PREFIX} purity={purity.describe()} has a decode stall SLO "
            f"of {slo:g}s, but the measured seam is {seam:g}s each way, so the "
            f"solved PP residency cap is {slo - 2 * seam:g}s -- not a bound, "
            f"and the PP phase would have no exit. Declare an SLO above "
            f"{2 * seam:g}s, or set SGLANG_PHASE_POLICY_PP_WINDOW_S > 0."
        )
    raise PhasePurityError(
        f"{LOG_PREFIX} purity={purity.describe()} requires a positive "
        f"phase-policy PP window, but pp_window_s is {window!r}. Under "
        f"purity the PP phase may not decode, so a PP phase that cannot "
        f"admit its pending prefill (no free mamba/GDN slot, KV held by "
        f"paused decodes) has NO exit except the bounded window: the "
        f"load-triggered rule needs prefill to drain, and prefill cannot "
        f"drain. Declare SGLANG_PHASE_POLICY_DECODE_STALL_SLO_S (preferred: "
        f"it bounds the residency in units of what is being protected), or set "
        f"SGLANG_PHASE_POLICY_PP_WINDOW_S > 0, or run --phase-flip-purity off."
    )


def _active_phase(scheduler) -> Optional[str]:
    """The layout this scheduler is currently running, or None when the
    flip is not enabled at all (then nothing is gated)."""
    if not getattr(scheduler.server_args, "enable_phase_flip", False):
        return None
    return getattr(scheduler, "phase_flip_active_stack", None)


#: Consecutive GROUP abandons of one direction after which the purity
#: prohibition on the work class that direction serves is lifted. Small on
#: purpose: ``/health`` times out at 20 s and a refused round costs about 3 s
#: on this rig, so a larger bound would open the valve after the instance has
#: already stopped answering -- which is the state this exists to prevent.
#:
#: LEFT AT 4 DELIBERATELY, 2026-08-16. Lowering it to 1 was tried as a way to
#: close the sub-10 s idle windows and is the WRONG FIX, which the suite says
#: out loud: it reds
#:   test_purity_stand_down_656 :: test_a_short_abandon_streak_is_not_a_stand_down
#:   test_decode_slo_starvation_662 :: test_it_holds_with_NO_count_REACHING_ITS_BOUND
#: Those are deliberate invariants -- a SHORT streak must not stand purity
#: down -- and an idle window is not a licence to delete them. The valve is
#: for a PERSISTENTLY unreachable layout; a single abandoned flip is a
#: latency defect in the seam, and it is fixed where it is caused. See the
#: gap decomposition before reaching for this constant again.
ENV_STAND_DOWN_AFTER = "SGLANG_PHASE_PURITY_STAND_DOWN_AFTER"
DEFAULT_STAND_DOWN_AFTER = 4

#: The direction whose failure starves each work class. Decode runs in TP, so
#: decode waiting in PP is starved by a ``pp_to_tp`` that will not commit.
_STARVED_BY = {"decode": "pp_to_tp", "prefill": "tp_to_pp"}


def stand_down_after() -> int:
    try:
        n = int(os.environ.get(ENV_STAND_DOWN_AFTER, DEFAULT_STAND_DOWN_AFTER))
    except ValueError:
        return DEFAULT_STAND_DOWN_AFTER
    return max(1, n)


def _decode_slo_s(scheduler) -> float:
    """The operator's decode-stall SLO in seconds, or 0.0 when unset."""
    try:
        cfg = getattr(scheduler, "phase_policy_cfg", None)
        if cfg is None:
            return 0.0
        return max(0.0, float(getattr(cfg, "decode_stall_slo_s", 0.0) or 0.0))
    except Exception:  # noqa: BLE001 - a safety valve must not raise
        return 0.0


def _group_starvation_signal(scheduler, work: str) -> bool:
    """Has the GROUP failed to fund the flip this work class needs?

    THE EVENT THE CLOCK IS STAMPED FROM, and the reason this function exists
    at all. It replaces "this rank has a non-empty batch", which was measured
    WRONG on metal 2026-08-15 in the only topology that matters here.

    THE PREMISE THAT FAILED. The clock used to start when decode was blocked
    in PP with ``running_bs > 0``, argued group-uniform because "phase, purity
    and batch are identical on every rank at that point". That is true under
    TP and FALSE UNDER PP. In a pipeline the HEAD holds the requests and the
    downstream ranks do not see them until it forwards them -- and a head that
    is holding decode never forwards. So PP1 and PP2 sat at ``running_bs = 0``,
    ``_clear_decode_starving`` reset their clocks on every iteration, and only
    rank 0 ever crossed the SLO.

    WHAT THAT COST, exactly, on boot_slo_proof_r2.log: one
    "RELAXING PURITY FOR DECODE" line, on PP0 alone, against two
    "FLIP ABANDONED" lines on every rank. Rank 0 then admitted a decode batch
    into the PP layout while its peers still refused decode, blocked in
    ``_pp_commit_comm_work`` on a proxy-tensor send whose matching receives
    nobody posted, while PP1 and PP2 blocked in ``recv_requests`` waiting for a
    forward rank 0 could no longer make. All three ranks alive, all three cards
    at 0% utilisation, and not one decode step ever ran. A safety valve that
    deadlocks the instance it is meant to rescue is worse than the stall.

    SO THE SIGNAL IS THE ONE THE GROUP ALREADY AGREES ON: the seam abandon
    streak for the direction that starves this work class. It is incremented
    from ``reduced_fit``, a collective MIN, so every rank advances it on the
    same iteration -- which the same log confirms, two lines per rank, exactly.
    No new collective is introduced; this reads a number the group already
    reduced.

    IT IS ALSO THE HONEST WORK SIGNAL. The streak only advances when the policy
    ARMED the flip, and the policy arms the direction that serves decode
    because decode is waiting. So "an empty batch is not starvation" still
    holds, sourced from a quantity every rank can see rather than from one only
    the head can.

    BOTH GROUP-UNIFORM COUNTERS COUNT, for the reason
    ``flip_unavailable_reason`` already reads both: the seam streak alone
    CANNOT advance once the policy's backoff engages, because the policy then
    declines the arm without entering the seam at all. Measured 2026-08-13
    15:40-15:44Z: three group abandons armed the backoff, the abandon counter
    froze at 3, and the policy went on logging "arm refused (7 in a row)" while
    work sat unrunnable. A signal keyed only to the inner counter is a signal
    the damping layer holds at zero.

    A NON-ZERO COUNT IS NOT THE SAME AS A COUNT THAT REACHED ITS BOUND, and
    that distinction is the whole point. The bounds (``stand_down_after``, the
    abandon cap) are what the SLO exists to outlive; ONE refusal is merely
    evidence that a funding failure is happening at all, which is exactly the
    precondition the invariant is stated over. The clock still has to run out.
    """
    rt = getattr(scheduler, "phase_flip_runtime", None)
    if rt is None:
        return False
    direction = _STARVED_BY.get(work)
    if direction is None:
        return False
    try:
        book = getattr(rt, "_seam_abandons_in_a_row", None) or {}
        if int(book.get(direction, 0) or 0) > 0:
            return True
        state = getattr(scheduler, "phase_policy_state", None)
        refusals = getattr(state, "arm_refusals", None) or {}
        return int(refusals.get(direction, 0) or 0) > 0
    except Exception:  # noqa: BLE001 - a safety valve must not raise
        return False


def _note_decode_starving(scheduler) -> None:
    """Stamp the instant the GROUP first failed to fund decode's layout.

    GROUP UNIFORMITY IS THE WHOLE PROPERTY HERE, and it is now sourced the way
    this file's other two causes are: from a group-reduced counter. See
    :func:`_group_starvation_signal` for the metal deadlock that proved a
    per-rank batch reading cannot carry it.
    """
    if getattr(scheduler, "_decode_starved_since", None) is None:
        scheduler._decode_starved_since = time.monotonic()


def _clear_decode_starving(scheduler) -> None:
    """Decode is moving (or has nothing to do): retire the clock."""
    if getattr(scheduler, "_decode_starved_since", None) is not None:
        scheduler._decode_starved_since = None


def _decode_starved_beyond_slo(scheduler) -> Optional[str]:
    """The SLO cause: decode held past the operator's bound by a FUNDING
    failure rather than by a terminal one.

    THE HOLE THIS CLOSES. The two existing causes are COUNTS -- a blocking
    guard, or an abandon streak reaching its bound. The SLO is a TIME. Nothing
    bridged them, so a funding failure that never accumulates the count could
    hold decode for ever inside the bound: on 2026-08-15 the seam abandoned
    repeatedly with the rate limiter pacing the retries, and the abandon-cap
    guard was deliberately stood down while work was waiting, so neither count
    arrived. That window was harmless only because the running batch was
    empty. With decodes resident it is an unbounded stall.

    So time is a cause in its own right: whatever the counts say, decodes are
    never held past the SLO by a funding failure.
    """
    slo = _decode_slo_s(scheduler)
    if slo <= 0.0:
        return None
    since = getattr(scheduler, "_decode_starved_since", None)
    if since is None:
        return None
    waited = time.monotonic() - since
    if waited < slo:
        return None
    return (
        f"decode has been held {waited:.1f}s by an unfunded flip, past the "
        f"{slo:g}s decode-stall SLO"
    )


def flip_unavailable_reason(scheduler, work: str) -> Optional[str]:
    """Why the flip cannot serve ``work`` right now, or None.

    #656 C22. Two causes, and BOTH inputs are already group-reduced, which is
    what lets this be read from a gate that must stay rank-uniform:

    * a BLOCKING GUARD -- the seam-abandon cap's terminal verdict, installed
      on every rank from the same unanimous abandon. The flip can never arm
      again on this boot, so the other layout is unreachable, for ever.
    * a streak of consecutive GROUP abandons of the direction this work class
      needs. ``_seam_abandons_in_a_row`` is booked from the reduced verdict --
      "all three ranks increment together" -- and reset to 0 by a committed
      cutover, so the streak means the same thing on every rank and clears on
      every rank at the same moment.

    Deliberately keyed on the DIRECTION rather than on any abandon: a stuck
    ``tp_to_pp`` starves prefill and says nothing about decode. Relaxing on
    the wrong one is how a safety valve becomes the normal path.
    """
    # THE SLO IS A CAUSE, and it is checked FIRST because it is the only one
    # with a bound the operator stated. The two below are counts; this is the
    # promise that no count can outlive.
    if work == "decode":
        starved = _decode_starved_beyond_slo(scheduler)
        if starved is not None:
            return starved
    rt = getattr(scheduler, "phase_flip_runtime", None)
    if rt is None:
        return None
    guards = tuple(getattr(rt, "blocking_guards", ()) or ())
    if guards:
        return "; ".join(str(g) for g in guards)
    direction = _STARVED_BY.get(work)
    if direction is None:
        return None
    bound = stand_down_after()
    book = getattr(rt, "_seam_abandons_in_a_row", None) or {}
    spent = int(book.get(direction, 0) or 0)
    if spent >= bound:
        return (
            f"{direction} abandoned {spent} times consecutively (bound "
            f"{bound}); the layout {work} needs is not reachable"
        )
    # AND THE POLICY'S OWN REFUSAL STREAK, because the seam streak alone
    # CANNOT REACH THE BOUND once the backoff engages. Measured on metal
    # 2026-08-13 15:40-15:44Z: three group abandons of tp_to_pp armed the
    # seam backoff, which then DECLINED the next arms without entering the
    # seam at all, so the abandon counter froze at 3 while the policy logged
    # "tp_to_pp arm refused (7 in a row)" and a 9-token prefill sat unrunnable
    # for four minutes. A valve keyed only on the inner counter is a valve
    # the damping layer holds shut.
    #
    # Group-uniform in the same way: every rank runs the same policy over the
    # same reduced verdicts, and the three ranks printed identical refusal
    # counts on every line of that window.
    state = getattr(scheduler, "phase_policy_state", None)
    refused = int((getattr(state, "arm_refusals", None) or {}).get(direction, 0) or 0)
    if refused >= bound:
        return (
            f"{direction} arm refused {refused} times consecutively (bound "
            f"{bound}); the layout {work} needs is not reachable"
            f"{_last_abandon_detail(state, direction)}"
        )
    # W30, THE FOURTH CAUSE: THE FLIP WORKS AND STILL SERVES NOTHING.
    #
    # The three causes above all describe a flip that does not HAPPEN --
    # guarded, abandoned, or refused. W30 found the state none of them can
    # see: every flip COMMITTED (150 of them in 17 minutes, 72 pp_to_tp
    # against 69 tp_to_pp) and the layout they committed into built no batch,
    # so the instance executed ZERO decode batches for ten minutes, timed out
    # every client request, and stood purity down exactly 0 times. A valve
    # keyed only on flips that fail is a valve blind to a flip that succeeds
    # pointlessly.
    #
    # The signal is the arm auditor's own, which was already computing this
    # and writing it to a log nobody consumed: `ARM-VERDICT-WRONG` fires when
    # a COMMITTED cutover builds nothing for `ARM_VERDICT_ROUNDS` rounds.
    #
    # RANK-UNIFORMITY -- THE WEAKEST OF THE FOUR, SAID PLAINLY. The three
    # causes above read group-REDUCED books. This one reads a per-rank round
    # counter, so a one-round skew between ranks is possible in principle.
    # It is bounded away rather than assumed away: the streak must reach
    # `stand_down_after()` SEPARATE fruitless arms, each of which is already
    # `ARM_VERDICT_ROUNDS` rounds deep, so the trigger sits ~32 batchless
    # rounds in. Nothing transient survives that depth; what does survive is
    # a config-determined livelock, and a config-determined state is by
    # construction the same on every rank. One built batch anywhere resets it.
    #
    # LOUD AND SCORED, NOT A QUIET PATH. Standing purity down here runs work
    # in the wrong layout, which w29_score.py counts as a violation and this
    # module's own log marks as a hazard. That is deliberate: this valve
    # firing means the PRIMARY fix (the seam-transport exemption) did not
    # work, and an acceptance run must fail loudly rather than pass on the
    # net. Serving degraded still beats serving nothing (#656 C22's rule),
    # but it must never be mistaken for the target mode.
    livelock = int(getattr(scheduler, "_arm_verdict_wrong_streak", 0) or 0)
    if livelock >= bound:
        return (
            f"LIVELOCK: {livelock} consecutive arms COMMITTED into the target "
            f"layout and built no batch there (bound {bound}, each arm "
            f"already watched for several rounds). The flip is not broken -- "
            f"it works and serves nothing, which no abandon or refusal "
            f"counter can see. {work} is starved by a seam that keeps "
            f"succeeding"
        )
    return None


def _last_abandon_detail(state, direction: str) -> str:
    """The shortfall that caused the stand-down, for the receipt.

    NAMED, NOT COUNTED. "abandoned 1 time consecutively" tells an operator
    that the valve opened and nothing about WHY, and the why is the number
    that gets acted on: the 2026-08-16 idle windows were a 19 MiB shortfall
    (staging 1614 MiB needed, 1595 MiB spendable), which is small enough that
    the evict rung could have funded it outright. A receipt that carries the
    figure is what makes that follow-up obvious instead of archaeological.
    """
    detail = (getattr(state, "arm_last_detail", None) or {}).get(direction)
    if not detail:
        return ""
    return f". Last abandon: {detail}"


def _relaxed(scheduler, work: str) -> bool:
    """True when purity must yield for ``work``, logged once per stand-down.

    A SERVING INSTANCE BEATS A CORRECT-LOOKING SILENT ONE. Purity is a
    throughput rule -- its own modes ``threshold:<n>`` and ``off`` run decode
    in the PP layout as supported configurations, and the documented cost is
    latency and throughput, never a wrong answer. So when the layout a work
    class needs is unreachable, running that work in the layout the instance
    is actually in is strictly better than emitting nothing until ``/health``
    times out with every stack idle.

    Not a mode change: the moment a flip commits, the streak resets and the
    prohibition is back, without anything having to remember to restore it.
    """
    reason = flip_unavailable_reason(scheduler, work)
    # EDGE-TRIGGER ON THE CAUSE, NOT ON ITS WORDING. The reason string carries
    # the streak COUNT, which grows every round, so keying the "log once" on
    # the whole string re-announced the same stand-down on every arm refusal
    # -- measured on boot_v2, once a minute for the life of the degrade.
    key = (work, reason.split(" ", 1)[0] if reason else None)
    if reason is None:
        # Edge-triggered: re-arm the log so the NEXT stand-down is announced.
        if getattr(scheduler, "_phase_purity_stood_down", None) is not None:
            scheduler._phase_purity_stood_down = None
            logger.warning(
                "%s the flip is working again; the purity rule is back in force for %s",
                LOG_PREFIX,
                work,
            )
        return False
    if getattr(scheduler, "_phase_purity_stood_down", None) != key:
        scheduler._phase_purity_stood_down = key
        logger.warning(
            "%s PHASE FLIP STOOD DOWN -- RELAXING PURITY FOR %s: %s. This "
            "instance would otherwise hold %s for ever and answer /health "
            "with 503 while every rank is alive and idle. %s now runs in the "
            "layout the instance is IN, which is slower and correct; the "
            "prohibition returns by itself the moment a flip commits. Set "
            "%s to change the bound.",
            LOG_PREFIX,
            work.upper(),
            reason,
            work,
            work.capitalize(),
            ENV_STAND_DOWN_AFTER,
        )
    return True


def prefill_blocked_here(scheduler, running_bs: int = -1) -> bool:
    """True when a prefill batch must NOT be built this iteration.

    Rank-uniform by construction, which is the load-bearing property: the
    purity rule is static boot config and the active layout is replicated
    (a cutover commits on every rank or on none). A rank-local input here
    would split the group across branches with mismatched collectives --
    the failure family documented at ``_update_uniform_pool_budget``.
    """
    from sglang.srt.managers.phase_policy import (
        PHASE_TP,
        prefill_suppressed_in_tp,
    )

    if _active_phase(scheduler) != PHASE_TP:
        return False
    # #677 HOT FIX 2: DRAIN MODE OUTRANKS THE PURITY MODE ON THIS ONE AXIS.
    #
    # Checked BEFORE `prefill_allowed_in_tp` on purpose. The deployed purity
    # mode is prefill_in_tp -- the 2026-08-14 correction that let the policy's
    # break-even N decide -- and that correction stands for its own workload.
    # Drain mode is a different contract, chosen by the user for this one:
    # prefill until empty, decode the bundle to completion, prefill again. A
    # TP window entered to finish a bundle must not admit the work it was
    # entered to escape, which is what "Prefill batch" lines inside TP were.
    #
    # Rank-uniform on the same argument as the rest of this function: the
    # drain-mode flag is static boot config and the phase is replicated.
    # The attribute is `phase_policy_cfg`. Named wrong once while writing
    # this, which a getattr default turns into a feature that silently never
    # fires -- the same shape as the #684 serving tick's NameError. Pinned by
    # `test_the_purity_hook_reads_the_real_scheduler_attribute`.
    policy = getattr(scheduler, "phase_policy_cfg", None)
    # THE VALVE OUTRANKS DRAIN MODE. Measured 2026-08-16 07:02 on the boot
    # that carried the first fix: tp_to_pp was DELAYED, not refused -- one
    # rank withheld its entry-margin yield on a predicted sub-law trough,
    # which is exempt from the stand-down cap, so the delay streak climbed to
    # 17 with no exit while the box sat at bs 0, GPU 0%, 794179 tok pending.
    #
    # `flip_unavailable_reason` had the answer all along: it reads the seam's
    # `_seam_abandons_in_a_row` (which delays DO advance) AND the policy's
    # `arm_refusals`, against one bound. What broke was ORDER -- suppression
    # was checked FIRST and returned True, so the valve never ran and the
    # seam's own promise that "the purity valve lets the starved work class
    # run meanwhile" was FALSE on metal. Asking the valve first makes that
    # promise true, and both wedge shapes leave through the same door.
    unavailable = None
    if policy is not None and bool(getattr(policy, "drain_mode", False)):
        try:
            unavailable = flip_unavailable_reason(scheduler, "prefill")
        except Exception:  # noqa: BLE001 - a guard never breaks the loop
            unavailable = None
    if policy is not None and prefill_suppressed_in_tp(
        policy, PHASE_TP, flip_unavailable=bool(unavailable), running_bs=running_bs
    ):
        return True
    # THE YIELD IS A RECEIPT-BEARING EVENT, and loud on purpose. Prefilling in
    # the TP layout is what drain mode exists to forbid; it resumes only
    # because an idle instance is worse. EDGE-TRIGGERED -- this runs every
    # scheduler iteration, and a per-iteration line would bury the signal it
    # exists to raise. Cleared on recovery, so a flapping rig logs each
    # engagement rather than only the first in the process's life.
    if unavailable:
        if not getattr(scheduler, "_drain_yield_announced", False):
            scheduler._drain_yield_announced = True
            logger.warning(
                "%s DRAIN-MODE SUPPRESSION YIELDED: %s. This TP layout resumes "
                "PREFILLING -- the behaviour drain mode exists to forbid, taken "
                "only because idling the instance with work waiting is worse. "
                "THE SEAM COULD NOT BE ENTERED, and that is the defect this "
                "line points at; the yield is the symptom. Clears on the first "
                "flip that commits.",
                LOG_PREFIX,
                unavailable,
            )
    elif getattr(scheduler, "_drain_yield_announced", False):
        scheduler._drain_yield_announced = False
        logger.warning(
            "%s the seam is reachable again; drain mode is back in force and "
            "the TP layout has stopped prefilling.",
            LOG_PREFIX,
        )
    if purity_of(scheduler).prefill_allowed_in_tp():
        return False
    # W30: FLIP TRANSPORT IS NOT WORKLOAD, and this is the one carve-out.
    #
    # See `seam_transport_exempt` for the full argument and the W30 specimen.
    # In one line: the #856 cutover RETRACTS its residents, so the only way a
    # decode-ready request can exist in the layout it was flipped into is to
    # be re-admitted there by a read-through that recomputes nothing. Strict
    # purity forbade that read-through, so the request could never cross the
    # seam and the instance ping-ponged 150 times executing zero decode
    # batches. Exempting a cache restore of already-computed tokens keeps
    # "never any WORK in the wrong layout" exactly as true as it was.
    if seam_transport_exempt(scheduler):
        return False
    return not _relaxed(scheduler, "prefill")


#: W30: attribute the seam stamps on every request it retracts under the #856
#: no-carry rule. Set ONLY in `build_cutover_release._retract`; deliberately
#: NOT `Req.is_retracted`, which decode-OOM preemption sets too.
SEAM_READMIT_ATTR = "seam_readmit_epoch"

#: Scheduler flag naming the round in which the seam-transport exemption is
#: open, read by the prefill builder to keep the batch to transport only.
SEAM_TRANSPORT_ROUND_ATTR = "_seam_transport_round"


def seam_readmit_candidates(scheduler) -> list:
    """Queued requests the #856 cutover retracted and must re-admit.

    Read off ``waiting_queue``, which is replicated across the ranks
    (scheduler.py's own note: "``self.waiting_queue``, which is replicated
    across the TP ranks"), so this list is the same on every rank in the same
    round -- the property the purity branch is required to have.
    """
    out = []
    for req in getattr(scheduler, "waiting_queue", ()) or ():
        if getattr(req, SEAM_READMIT_ATTR, None) is not None:
            out.append(req)
    return out


def seam_transport_exempt(scheduler) -> bool:
    """Is this round's TP prefill a SEAM RE-ADMISSION, i.e. flip transport?

    THE W30 LIVELOCK, and why this exists rather than a purity stand-down.
    Measured 2026-08-24 (SPECIMEN_w30_a1_purity_nocarry_livelock.log): 150
    flips in 17 minutes, 129 prefill batches, **zero decode batches**, every
    client request timing out at 600 s. The scheduler's own arm auditor named
    it 12 times -- "armed pp_to_tp (... 1 req decoding ...), the cutover
    COMMITTED into the target layout, and it still built no batch in 8
    rounds ... target_can_admit=False".

    The chain is short and every link is a shipped design decision:
      1. the policy arms pp_to_tp BECAUSE a request has drained prefill and
         is ready to decode;
      2. the #856 seam then RETRACTS that very request -- no-carry, "their KV
         is in the canonical store from the fence; the new layout re-admits
         them and serves the prefix by read-through";
      3. re-admitting it in TP therefore needs a read-through PREFILL batch;
      4. strict purity forbids prefill in TP absolutely -- the W30 arm logged
         `Prefill batch phase=tp` exactly 0 times;
      5. so TP builds nothing, the policy flips back, the request re-prefills
         in PP, drains, and arms the same flip again. For ever.
    The arm's own justification is destroyed by the arm's own execution.

    WHY THIS IS AN EXEMPTION AND NOT A RELAXATION. A purity stand-down would
    let ORDINARY prefill run in TP, which both the #838 detector and
    w29_score.py count as wrong-layout work -- so it would make the very
    acceptance this is meant to pass unpassable by our own scorers, and it
    would be dishonest about the user's rule. What crosses here is different
    in kind: the tokens were already prefilled in the PP window, their KV is
    in the canonical store, and the re-admission recomputes nothing -- it is
    a cache restore, i.e. SEAM MECHANICS, the same category as the KV the
    flip moves. "Never any work in the wrong layout" is untouched, because
    this is not work.

    THE DANGEROUS DIRECTION IS PINNED. A genuine, never-retracted request
    must still be refused in TP. That is why the stamp is seam-specific
    (`SEAM_READMIT_ATTR`, set only by the cutover's own retract closure) and
    not `Req.is_retracted`, which decode-OOM preemption sets identically --
    keying on the latter would silently exempt every preempted request's
    re-prefill, which IS work. The builder additionally keeps the batch to
    stamped requests only, so a new arrival cannot ride along inside an
    exempt batch.

    RANK-UNIFORM: the stamp comes from a group-unanimous cutover and is read
    off the replicated ``waiting_queue``, so every rank takes this branch in
    the same round. A rank-local input here would split the group across
    branches with mismatched collectives.
    """
    if not seam_readmit_candidates(scheduler):
        # RE-DERIVED EVERY ROUND, NEVER LATCHED. If the flag stayed set after
        # the debt was paid, a later round would filter the prefill builder
        # down to stamped requests that no longer exist and build nothing --
        # trading the W30 livelock for a quieter one.
        try:
            setattr(scheduler, SEAM_TRANSPORT_ROUND_ATTR, False)
        except Exception:  # noqa: BLE001 - a flag may never break the gate
            pass
        return False
    try:
        setattr(scheduler, SEAM_TRANSPORT_ROUND_ATTR, True)
    except Exception:  # noqa: BLE001 - the flag is an optimisation, not a gate
        return False
    if not getattr(scheduler, "_seam_transport_announced", False):
        scheduler._seam_transport_announced = True
        logger.warning(
            "%s SEAM TRANSPORT ADMITTED in the TP layout: the #856 cutover "
            "retracted its residents, so their re-admission is a read-through "
            "that recomputes nothing -- flip transport, not workload, and the "
            "purity rule on WORK is untouched. Only requests the cutover "
            "itself stamped are admitted; a genuine new request is still "
            "refused here. Without this the instance ping-pongs for ever "
            "executing zero decode batches (W30).",
            LOG_PREFIX,
        )
    return True


def decode_blocked_here(scheduler, running_bs: int) -> bool:
    """True when a decode step must NOT execute this iteration."""
    from sglang.srt.managers.phase_policy import PHASE_PP

    if _active_phase(scheduler) != PHASE_PP:
        _clear_decode_starving(scheduler)
        return False
    if purity_of(scheduler).decode_allowed_in_pp(running_bs):
        _clear_decode_starving(scheduler)
        return False
    # THE CLOCK RUNS ONLY WHEN THERE IS DECODE TO HOLD, AND "THERE IS DECODE TO
    # HOLD" MUST BE A GROUP FACT. It was ``running_bs > 0`` -- this rank's own
    # batch -- and under PP only the head has one, so only the head ever
    # crossed the SLO and the group half-relaxed into a deadlock. See
    # _group_starvation_signal for the measurement. The signal is now the seam
    # abandon streak, which every rank advances off the same collective MIN.
    if _group_starvation_signal(scheduler, "decode"):
        _note_decode_starving(scheduler)
    else:
        _clear_decode_starving(scheduler)
    return not _relaxed(scheduler, "decode")


def purity_of(scheduler) -> PhasePurity:
    """The scheduler's purity rule, resolved once and cached."""
    cached = getattr(scheduler, "_phase_purity", None)
    if cached is not None:
        return cached
    purity = purity_from_server_args(scheduler.server_args)
    scheduler._phase_purity = purity
    return purity
