"""#838 THE LAYOUT-CONFORMANCE DETECTOR. The code says it, not the operator.

WHY THIS MODULE EXISTS. Three times in a row the fact that prefill was
executing in the TP layout while PP was the intended one was discovered by a
HUMAN READING A LOG AFTER THE FACT. boot_window3_0823_1733 is the specimen:
2253 ``Prefill batch phase=tp`` lines against 96 ``phase=pp``, over a span in
which the policy printed ``holding in tp`` on every round and nothing in the
process ever said that this was remarkable. A defect that only a person
looking at the right grep can see is a defect that recurs, so this module
makes the running process say it, on the round it happens.

TWO CLASSES, AND THEY ASK DIFFERENT QUESTIONS.

  CLASS 1, HARD CONFORMANCE -- "is the layout the batch RAN in the layout the
  decision was MADE in?" A pure intent-versus-execution comparison. It carries
  no opinion about which layout is right; it only refuses to let the two
  diverge silently. Every arm of it is a binary fact, so it is expected to
  read zero on a healthy boot and any non-zero count is a defect.

  CLASS 2, ECONOMIC DIVERGENCE -- "does the policy's own measured price
  contradict the policy's own behaviour?" Also carries no opinion about which
  layout is right. It reads the #819 price line -- the numbers the policy
  itself computed and acted on -- and alarms only when the policy holds a
  layout that its OWN arithmetic says it should have left, for a reason that
  is not one of the legitimate ones.

WHAT THIS MODULE MUST NEVER DO, and the rule that decides every threshold in
it: PREFILL RUNNING IN TP AT AN HONESTLY HIGH MEASURED FLIP PRICE IS CORRECT
ECONOMICS AND MUST NOT ALARM. There is no hardcoded layout wish anywhere in
here. The class-2 hinge is ``pending > live_flip_tokens``, and
``live_flip_tokens`` is the repriced break-even: an expensive measured seam
RAISES that bar, which switches the detector OFF by construction. Only a
CHEAP measured price -- the policy saying "flipping pays" and then not
flipping -- can trip it. Both directions are pinned by test.

NO KNOB. The detector is always on wherever the phase flip is on; there is no
env var and no flag to turn it off (#837's law: configuration is planner-
derived or a flag, and a detector that can be silenced is a detector that will
be silenced on the boot that needed it). The only quantity that is configured
at all is the economy window ``N``, and that is DERIVED from the policy's own
existing size rather than pinned here -- see ``economy_window_s``.

RANK-LOCAL BY CONSTRUCTION. Every reading this module consumes is a plain
attribute or module-global read on the rank that calls it
(``phase_flip_tp_routing_active`` is a module global in parallel_state;
``PhaseFlipRuntime.phase`` is an instance attribute; the policy cfg/state/
inputs are already replicated by their own contract). This module adds NO
collective. A conformance detector that needed a reduce to answer would be
unable to run on the per-iteration path it has to run on, and would add a new
member to the shared-collective family it exists to police.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

#: The two grep keys. They are BOOT-ACCEPTANCE CRITERIA (queue ticket W17), so
#: they are literals with no interpolation in them and they never move.
ALARM_CONFORMANCE = "LAYOUT-CONFORMANCE VIOLATION (#838)"
ALARM_ECONOMY = "LAYOUT-ECONOMY ANOMALY (#838)"

#: W25/#854: the DECLINE marker, and it is a grep key of equal standing.
#:
#: WHY IT EXISTS. W25 ran a sustained TP-sticky prefill phase that the USER
#: found BY EYE, while `grep -c "LAYOUT-ECONOMY ANOMALY" = 0` for the whole
#: boot. The detector was RIGHT to stay silent -- pending was 16-20k against a
#: live bar of 18614, so the policy's own economy favoured TP and an alarm
#: would have been the over-eager mutant this file exists to exclude. But a
#: zero that means "ran and correctly declined" is indistinguishable from a
#: zero that means "never ran", "not wired", or "threw", and a reader cannot
#: tell which without reading the source. That is the #851 defect class -- the
#: silent multi-valued zero -- occurring inside this detector, one commit after
#: #853(i) had to remove exactly the same shape from the exposure gate.
#:
#: So the detector now says what it decided, every time, at a heartbeat
#: cadence. Silence becomes impossible: either the box is not running this
#: code, or a line appears. The count of these is NOT an error count and must
#: never be read as one -- it is the proof that the instrument is alive.
CHECKED_ECONOMY = "LAYOUT-ECONOMY CHECKED (#838)"

#: Class-1 arms, named in the line so a grep can split them without parsing.
KIND_ADMIT_VS_EXEC = "admit_vs_exec"
KIND_VERDICT_VS_ROUTING = "verdict_vs_routing"
KIND_STALE_RING_RESTORE = "stale_ring_restore"

#: Class-2 illegitimacy labels, likewise.
ILLEGITIMATE_BELOW_BAR = "below-bar-while-pending-exceeds-bar"
ILLEGITIMATE_BUNDLE_NOT_DRAINING = "decode-bundle-not-draining"

#: Seconds before an alarm of the SAME SHAPE is printed again. The counter
#: below is incremented on every occurrence regardless -- the throttle bounds
#: the LOG, never the measurement, which is the split #631's own policy
#: throttle got right after a 12765-line flood cost this feature a self-kill.
ALARM_REANNOUNCE_S: float = 10.0

#: Seconds between DECLINE heartbeats. Deliberately much longer than the alarm
#: cadence: an alarm is urgent and a decline is a liveness proof, so it must be
#: cheap enough that nobody is ever tempted to switch it off. One line per
#: minute per rank is the price of never using a human eye as the instrument
#: again.
DECLINE_REANNOUNCE_S: float = 60.0


@dataclass
class LayoutConformanceCounters:
    """Cumulative per-class counts for this rank, for the periodic line.

    Cumulative and never reset in serving: the periodic line reports a LEVEL,
    so a reader comparing two lines gets the delta, and a reader who joined
    late still sees that something happened earlier in the boot.
    """

    conformance_violations: int = 0
    economy_anomalies: int = 0
    #: W25/#854: economy verdicts that RAN and declined. Not an error count --
    #: it is what separates "healthy" from "never ran" when c2 reads 0.
    economy_checks: int = 0

    def as_field(self) -> str:
        """The fragment appended to the periodic stats line."""
        return (
            f"layout-conformance (#838): c1={self.conformance_violations}, "
            f"c2={self.economy_anomalies}, c2ok={self.economy_checks}"
        )


#: Process-global, rank-local. A module global rather than scheduler state for
#: the same reason `phase_policy` keeps its cost estimator here: the emitter
#: (metrics_reporter) and the two detection sites (scheduler) do not share an
#: object, and threading one through three call chains to carry two integers
#: would be a wider change than the feature.
_COUNTERS = LayoutConformanceCounters()

#: Last log time per alarm shape, for ALARM_REANNOUNCE_S.
_LAST_SAID: dict = {}


def counters() -> LayoutConformanceCounters:
    return _COUNTERS


def reset_for_test() -> None:
    """Test-only: clear the process globals. Never called in serving."""
    global _COUNTERS
    _COUNTERS = LayoutConformanceCounters()
    _LAST_SAID.clear()


def _should_say(key: str, now: float, every_s: Optional[float] = None) -> bool:
    """Throttle one shape. `every_s` defaults to the ALARM cadence.

    The cadence is a PARAMETER rather than a second copy of this function
    because the decline heartbeat needs a slower one, and two throttles that
    could drift apart is the shape this file already spends its length
    avoiding.
    """
    window = ALARM_REANNOUNCE_S if every_s is None else every_s
    last = _LAST_SAID.get(key)
    if last is not None and now - last < window:
        return False
    _LAST_SAID[key] = now
    return True


def _quote(reason: Optional[str]) -> str:
    """Render a verdict for the alarm line.

    The caller passes the reason it ALREADY read. This function never reaches
    back for it, which is the #713 rule ("ONE reading, used by the verdict AND
    by the message that reports it") applied to this detector: an alarm that
    re-read the policy would be free to quote a verdict the comparison never
    saw, and the quote is the whole evidentiary value of the line.
    """
    if not reason:
        return "<no verdict recorded>"
    return reason.replace('"', "'")


# ---------------------------------------------------------------------------
# CLASS 1 -- hard conformance. Pure verdicts; the caller does the logging.
# ---------------------------------------------------------------------------


def admit_vs_exec_verdict(
    admitted_phase: Optional[str],
    executing_phase: str,
    rid: str,
    mb_id: int,
    verdict_reason: Optional[str],
) -> Tuple[bool, str]:
    """Did this batch execute in the layout it was ADMITTED in?

    THE QUESTION THIS ANSWERS, and why it is the right one. Admission decides
    WHICH requests enter the batch, in what order, and how many -- against the
    layout that was live when the scheduling pass ran. The forward then routes
    through whichever process groups ``phase_flip_tp_routing_active()`` names
    at the instant it runs. If a cutover commits between those two points, a
    batch shaped for one group is executed on another. That is the
    "flip committed, old-group batch processed afterwards" form, and it is
    unambiguously a defect at any count: there is no load, no price and no
    policy setting under which it is the intended behaviour.

    ``admitted_phase`` is None when the batch did not come through the
    scheduling funnel that stamps it (the decoupled spill batch is the live
    example). Unknown provenance is NOT a violation -- it is silence, which is
    the safe direction for a detector whose false positives would train the
    operator to ignore it.
    """
    if admitted_phase is None:
        return False, "batch carries no admission stamp: nothing to compare"
    if admitted_phase == executing_phase:
        return False, (
            f"batch admitted and executed in phase={executing_phase}: conformant"
        )
    return True, (
        f"{ALARM_CONFORMANCE} kind={KIND_ADMIT_VS_EXEC} "
        f"admitted={admitted_phase} executed={executing_phase} "
        f"rid={rid} mb_id={mb_id} "
        f"class1_total={_COUNTERS.conformance_violations + 1} "
        f'verdict="{_quote(verdict_reason)}". A cutover committed between the '
        f"scheduling pass that shaped this batch and the forward that ran it, "
        f"so the batch was built against the {admitted_phase} groups and "
        f"executed on the {executing_phase} ones"
    )


def verdict_vs_routing_verdict(
    verdict_phase: Optional[str],
    routing_phase: str,
    mb_id: int,
    verdict_reason: Optional[str],
) -> Tuple[bool, str]:
    """Did the policy reason about the layout that is actually routing?

    ``PhasePolicyInputs.phase`` comes from ``PhaseFlipRuntime.phase``;
    ``phase_flip_tp_routing_active()`` is the module global the model code
    itself branches on. The cutover writes both, in the same function, one
    after the other -- so they agree except in the window between the two
    writes, and outside that window a disagreement means one of the three
    mirrors of the live layout has drifted. A verdict computed against a
    layout that is not the one executing is an ADMISSION VERDICT ABOUT THE
    WRONG LAYOUT, which is the class this ticket exists for.

    Deliberately compared against the ROUTING flag rather than against
    ``scheduler.phase_flip_active_stack``: the routing flag is the one the
    forward obeys, and a label that disagrees with the thing it labels is
    exactly what #758 found and fixed on the batch line.
    """
    if verdict_phase is None:
        return False, "no verdict phase recorded: nothing to compare"
    if verdict_phase == routing_phase:
        return False, (
            f"policy verdict and routing agree on phase={routing_phase}: conformant"
        )
    return True, (
        f"{ALARM_CONFORMANCE} kind={KIND_VERDICT_VS_ROUTING} "
        f"verdict_phase={verdict_phase} routing_phase={routing_phase} "
        f"rid=- mb_id={mb_id} "
        f"class1_total={_COUNTERS.conformance_violations + 1} "
        f'verdict="{_quote(verdict_reason)}". The phase-flip runtime and the '
        f"module-level routing flag disagree about the live layout, so this "
        f"round's policy verdict was computed against a layout the forward "
        f"is not using"
    )


def stale_ring_restore_verdict(
    armed_passes: Optional[int],
    arm_epoch: Optional[int],
    live_epoch: Optional[int],
    arm_mb_id: Optional[int],
    verdict_reason: Optional[str],
) -> Tuple[bool, str]:
    """The W12 form: a COMMIT at zero armed passes carrying a retired slot.

    boot_window2_0823_1554 @ f9d7637f04. A committed cutover reached the
    armed-window falling edge with ``_pp_flip_armed_passes == 0`` and an
    ``_pp_flip_arm_mb_id`` recorded against the ring the cutover had just
    rebuilt. The slot was restored onto the fresh ring, PP0 re-entered on slot
    2 while its downstream was on slot 0, and one second later the group died
    on a ``PROXY LEFTOVER REFUSED``.

    #829 already REFUSES that restore. This verdict does not repeat the
    refusal and does not change it -- it makes the SHAPE audible. A guard that
    silently declines tells nobody that the condition it guards against
    occurred, and this condition occurring means a commit and a slot-ring
    rebuild raced, which is a finding whether or not the guard caught it.

    Zero passes with a MATCHING epoch is the ordinary abandon that W4b
    restores on purpose, and is not a violation.
    """
    if armed_passes is None or armed_passes != 0:
        return False, "armed window ran passes (or none was open): not the W12 form"
    if arm_epoch is None or live_epoch is None:
        return False, "no epoch on one side: falls back to the slot-only reading"
    if arm_epoch == live_epoch:
        return False, "zero passes within one ring generation: an ordinary abandon"
    return True, (
        f"{ALARM_CONFORMANCE} kind={KIND_STALE_RING_RESTORE} "
        f"armed_passes=0 arm_epoch={arm_epoch} live_epoch={live_epoch} "
        f"rid=- mb_id={arm_mb_id} "
        f"class1_total={_COUNTERS.conformance_violations + 1} "
        f'verdict="{_quote(verdict_reason)}". A cutover committed with zero '
        f"armed passes and the slot recorded at arming names ring generation "
        f"{arm_epoch}, which no longer exists -- the live ring is generation "
        f"{live_epoch}"
    )


def work_layout_verdict(
    *,
    batch_class: str,
    phase: str,
    strict: bool,
    transport_verified: bool,
    n_reqs: int,
    new_tokens: int,
    cached_tokens: int,
    now: float,
) -> Optional[str]:
    """#861d: WORK IN THE WRONG LAYOUT. The term this detector did not have.

    THE GAP, measured. W37-D formed 258 prefill batches in the TP layout and
    this module flagged ZERO of them, because every existing term asks about
    the ECONOMY of a hold or the provenance of a routing decision -- none asks
    the user's actual law: *never any work in the wrong layout*. The detector
    ran through all 258 and stayed quiet, which is worse than not existing,
    because its silence was read as conformity.

    WHAT COUNTS AS A VIOLATION, and the distinction is the whole point.
    Prefill in TP is permitted for SEAM TRANSPORT -- a re-admitted request
    whose KV is restored from the canonical store recomputes nothing, so it is
    mechanics, not work. That exemption is legitimate ONLY while its premise
    holds, and W37-D proved the premise can be false: `#new-token: 4096,
    #cached-token: 0` on every one of the 258, i.e. cold prefill wearing
    transport's clothes.

    So the verdict keys on the MEASURED bytes, not on the caller's intent:
    a batch claiming transport while recomputing tokens is a violation, and
    `cached_tokens == 0 and new_tokens > 0` is exactly that shape.

    Returns the detail string when it is a violation, else None. The caller
    passes it to ``note_conformance_violation`` so counting and throttling stay
    in one place.
    """
    if not strict:
        return None
    wrong_layout = (batch_class == "prefill" and phase == "tp") or (
        batch_class == "decode" and phase == "pp"
    )
    if not wrong_layout:
        return None
    # A verified restore that actually restored is mechanics, not work.
    if transport_verified and int(cached_tokens) > 0 and int(new_tokens) <= 0:
        return None
    recomputing = int(new_tokens) > 0
    return (
        f"{ALARM_CONFORMANCE} kind=work_in_wrong_layout class={batch_class} "
        f"phase={phase} reqs={n_reqs} new_tokens={new_tokens} "
        f"cached_tokens={cached_tokens} transport_claimed={transport_verified} "
        f"recomputing={recomputing} "
        f"class1_total={_COUNTERS.conformance_violations + 1} -- the user's "
        f"strict-batch law forbids work in this layout. A batch that claims "
        f"SEAM TRANSPORT while recomputing tokens is not transport: W37-D ran "
        f"258 such batches at #new-token 4096 / #cached-token 0 and this "
        f"detector flagged none of them, because it had no term for the law "
        f"itself."
    )


def note_conformance_violation(detail: str, now: float) -> bool:
    """Count it always, log it at most once per ALARM_REANNOUNCE_S per shape.

    Returns whether the line was emitted, so a caller under test can assert on
    the throttle without reading the log.
    """
    _COUNTERS.conformance_violations += 1
    # Key on the SHAPE, not the text: the detail carries live rids and slot
    # numbers, so keying on it would make every occurrence look new and the
    # throttle would never engage -- measured on #631's own first attempt.
    kind = "unknown"
    for token in detail.split():
        if token.startswith("kind="):
            kind = token
            break
    if not _should_say(f"c1:{kind}", now):
        return False
    logger.error("%s", detail)
    return True


# ---------------------------------------------------------------------------
# CLASS 2 -- economic divergence.
# ---------------------------------------------------------------------------


def economy_window_s(drain_stall_deadline: float) -> float:
    """N: how long a hold must last before its economics are questioned.

    DERIVED, NOT PINNED. The caller passes
    ``phase_policy.drain_stall_deadline_s(cfg)``, whose own docstring already
    settles what it is: "SOLVED, not chosen: one decode window", built as
    ``max(10.0, solved_tp_decode_floor_s(cfg))`` where the solved floor is
    ``2 * flip_cost_s``, i.e. one full round trip. That is exactly the
    quantity this detector needs -- the shortest residency that is worth its
    own cutovers -- so it is reused rather than re-derived, and a future
    change to the policy's idea of a decode window moves this detector with
    it instead of leaving a second, stale copy behind.

    A non-positive input (a cfg with no flip cost at all) yields 0.0, which
    the verdict below reads as "no window declared" and declines on, rather
    than as "alarm immediately".
    """
    return max(0.0, float(drain_stall_deadline))


def economy_divergence_verdict(
    *,
    phase: str,
    held_s: float,
    window_s: float,
    pending_prefill_tokens: int,
    live_flip_tokens: int,
    applied_bar_tokens: int,
    live_flip_cost_s: float,
    price_measured: bool,
    hold_reason: Optional[str],
    since_flip_s: float,
    min_dwell_s: float,
    staging_active: bool,
    running_bs: int,
    bundle_at_phase_entry: int,
    bundle_stall_s: float,
) -> Tuple[bool, str]:
    """Does the policy's own measured price contradict its own behaviour?

    THE ONE ARM THIS CLAIMS, and why the mirror is not claimed. The check is
    written for a hold in the TP layout, because that is the direction in
    which the #819 price line is DENOMINATED: ``live_flip_tokens`` is a
    break-even in PENDING PREFILL TOKENS, so "pending exceeds the bar" is
    literally the policy's own statement that leaving TP pays. There is no
    equally direct statement in the other direction -- the quantity that would
    make TP pay while sitting in PP is decode latency, which this price line
    does not carry -- so the PP arm is DECLINED rather than approximated. An
    approximated mirror would be a hardcoded layout wish wearing arithmetic,
    which is the one thing this module may not contain.

    THE FIVE GATES, in the order they are cheapest to fail:

    1. The hold is in TP. (Otherwise: not claimed, see above.)
    2. A window was declared. (``window_s > 0``.)
    3. The layout has been held longer than one decode window.
    4. THE PRICE IS MEASURED. A seeded price is an assumption, not the
       policy's own claim, and a detector that alarmed on an assumption would
       be reporting the seed's opinion as a defect.
    5. THE PRICE FAVOURS LEAVING: ``pending`` exceeds THE BAR THE POLICY
       ACTUALLY APPLIED.

       THIS IS THE GATE THAT MAKES HONEST ECONOMICS SAFE. ``live_flip_tokens``
       IS the repriced break-even; an expensive measured seam raises it, so a
       genuinely costly flip pushes ``pending`` below the bar and this gate
       closes. Prefill running in TP because flipping is honestly expensive
       therefore CANNOT reach the alarm. Only a cheap measured price -- the
       policy having computed that flipping pays, and then not flipping --
       can get past here. Both directions are pinned by test.

       #853(iii): THE BREAK-EVEN IS NOT THE ONLY BAR, and reading it as if it
       were produced a false positive on metal. Above ``live_flip_tokens``
       sits the #665-F1 SECONDARY BAND, whose upper edge is
       ``phase_policy.effective_flip_threshold(cfg, running_bs)`` -- the
       break-even plus the cost of stranding the requests currently decoding.
       A hold inside that band is the policy's differential arithmetic
       working, not a divergence from it. W24 09:01:37: this detector printed
       ``pending_tok=22887 bar_tok=20057`` and alarmed, while the hold it
       indicted said in its own text ``> N=20057 but <= 30086``. Two bars for
       one comparison, which is exactly what #819's ONE READING rule forbids
       inside ``phase_policy`` -- the rule simply had no reach across this
       module boundary.

       So the applied bar is PASSED IN from the one authority that computes
       it, and is REQUIRED rather than defaulted: a default is the mechanism
       by which a caller silently re-creates the divergence.

       THE GATE TAKES THE HIGHER OF THE TWO, which bounds the blast radius of
       this change to exactly one direction: it can only ever RAISE the bar,
       so it can only remove false positives and can never invent a new
       alarm. That also handles strict purity, where ``effective_flip_
       threshold`` returns 0 by construction (a sub-N prompt cannot run in TP
       at all) -- there the break-even stands, unchanged.

    Then, and only then, the hold must be for a reason that is not legitimate.
    LEGITIMACY IS DECIDED STRUCTURALLY, NEVER BY READING THE REASON STRING.
    #817 removed exactly that anti-pattern from this code path (a denylist of
    three substrings, with everything unrecognised silently swallowed), and
    re-introducing it here would re-introduce the same failure: a reason
    nobody thought of becomes legitimate by accident. The reason string is
    QUOTED for the human and is never parsed.

    The legitimate holds:

      * MIN DWELL -- ``since_flip_s < min_dwell_s``. A thrash bound the policy
        is entitled to enforce whatever the price says.
      * STAGING LIMIT WITH STAGING ACTUALLY ACTIVE. The rate limit is
        legitimate while a staging attempt is in flight; the caller passes
        whether one is.
      * DECODE BUNDLE DURING A REAL DRAIN. "Real" is two conditions, and
        #833's own argument is why both are needed: "the bundle getting
        smaller is progress toward an empty set; the bundle holding station is
        not, however much work it retires". So a real drain is NET progress
        (``running_bs < bundle_at_phase_entry``) AND RECENT progress (the set
        shrank inside the window). A set oscillating around its entry size
        while admission refills it is holding station, and that is the
        window-3 shape: ``6 of 6``, ``7 of 7``, ``5 of 5``, ``6 of 6``, for
        twenty minutes, with a prefill batch every three seconds in TP.
    """
    from sglang.srt.managers.phase_policy import PHASE_TP

    if phase != PHASE_TP:
        return False, (
            f"hold is in phase={phase}: the #819 price line is denominated in "
            f"pending prefill tokens and states no economics for this "
            f"direction, so no claim is made"
        )
    if window_s <= 0.0:
        return False, "no decode window declared: nothing to measure the hold against"
    if held_s < window_s:
        return False, (
            f"held {held_s:.1f}s < one decode window {window_s:.1f}s: too "
            f"early to question"
        )
    if not price_measured:
        return False, (
            "flip price provenance is seed or half-measured, not measured: an "
            "assumption is not the policy's own claim and is not evidence of "
            "divergence. #856: C is a ROUND TRIP of two independently measured "
            "legs, so one measured leg still leaves half the price a seed"
        )
    # The bar the policy APPLIED, never the break-even alone. See gate 5.
    applied_bar = max(int(live_flip_tokens), int(applied_bar_tokens))
    if pending_prefill_tokens <= applied_bar:
        band = (
            ""
            if applied_bar == live_flip_tokens
            else (
                f" -- inside the secondary band (> N={live_flip_tokens}, "
                f"<= {applied_bar}), where the bar prices the "
                f"{running_bs} req this cutover would strand on top of the "
                f"seam"
            )
        )
        return False, (
            f"pending {pending_prefill_tokens} tok <= bar {applied_bar} "
            f"tok at a measured seam of {live_flip_cost_s:.2f}s: the policy's "
            f"own arithmetic says holding is right, which is correct "
            f"economics and not an anomaly{band}"
        )
    if since_flip_s < min_dwell_s:
        return False, (
            f"min dwell: {since_flip_s:.1f}s since the last flip < "
            f"{min_dwell_s:g}s -- a legitimate thrash bound"
        )
    if staging_active:
        return False, (
            "a staging attempt is in flight: the staging rate limit is a "
            "legitimate hold while it is actually staging"
        )
    bundle_draining = (
        bundle_at_phase_entry > 0
        and running_bs < bundle_at_phase_entry
        and bundle_stall_s < window_s
    )
    if bundle_draining:
        return False, (
            f"decode bundle is genuinely draining: {running_bs} running vs "
            f"{bundle_at_phase_entry} at phase entry, last shrank "
            f"{bundle_stall_s:.1f}s ago -- a legitimate drain"
        )
    illegitimate = (
        ILLEGITIMATE_BUNDLE_NOT_DRAINING
        if bundle_at_phase_entry > 0
        else ILLEGITIMATE_BELOW_BAR
    )
    return True, (
        f"{ALARM_ECONOMY} held={phase} held_s={held_s:.1f} "
        f"window_s={window_s:.1f} pending_tok={pending_prefill_tokens} "
        f"bar_tok={live_flip_tokens} applied_bar_tok={applied_bar} "
        f"seam_s={live_flip_cost_s:.2f} "
        f"provenance=measured running_bs={running_bs} "
        f"bundle_entry={bundle_at_phase_entry} "
        f"bundle_stall_s={bundle_stall_s:.1f} illegitimate={illegitimate} "
        f"class2_total={_COUNTERS.economy_anomalies + 1} "
        f'verdict="{_quote(hold_reason)}". The policy has held {phase} for '
        f"{held_s:.1f}s while its OWN measured price ({live_flip_cost_s:.2f}s "
        f"seam, applied bar {applied_bar} tok) says the "
        f"{pending_prefill_tokens} tok waiting would be served faster in the "
        f"other layout, and the hold is not min-dwell, not active staging, "
        f"and not a draining bundle"
    )


def note_economy_anomaly(detail: str, now: float) -> bool:
    """Count it always, log it at most once per ALARM_REANNOUNCE_S."""
    _COUNTERS.economy_anomalies += 1
    if not _should_say("c2", now):
        return False
    logger.error("%s", detail)
    return True


def note_economy_declined(detail: str, now: float) -> bool:
    """The verdict RAN and declined. Count always, log on a heartbeat.

    W25/#854. The caller must invoke this on EVERY non-alarming verdict, so
    that the absence of anomalies is a fact in the log rather than an
    inference. The pair is mutually exclusive by construction at the call
    site: a verdict either alarms or declines, never both, and the tests pin
    that -- a detector that emitted both would restore the ambiguity while
    passing every other assertion.

    Logged at INFO, not DEBUG: W25 ran at INFO and would not have seen a DEBUG
    line, which is the same mistake #853(i) had to correct on the exposure
    clamp. A liveness proof nobody's log level shows is not a liveness proof.
    """
    _COUNTERS.economy_checks += 1
    if not _should_say("c2ok", now, every_s=DECLINE_REANNOUNCE_S):
        return False
    logger.info("%s %s", CHECKED_ECONOMY, detail)
    return True
