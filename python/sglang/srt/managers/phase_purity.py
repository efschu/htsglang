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
        "%s mode=%s -- decode in the PP layout: %s; prefill in the TP " "layout: %s",
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
    raise PhasePurityError(
        f"{LOG_PREFIX} purity={purity.describe()} requires a positive "
        f"phase-policy PP window, but pp_window_s is {window!r}. Under "
        f"purity the PP phase may not decode, so a PP phase that cannot "
        f"admit its pending prefill (no free mamba/GDN slot, KV held by "
        f"paused decodes) has NO exit except the bounded window: the "
        f"load-triggered rule needs prefill to drain, and prefill cannot "
        f"drain. Set SGLANG_PHASE_POLICY_PP_WINDOW_S > 0, or run "
        f"--phase-flip-purity off."
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
        )
    return None


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
                "%s the flip is working again; the purity rule is back in "
                "force for %s",
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


def prefill_blocked_here(scheduler) -> bool:
    """True when a prefill batch must NOT be built this iteration.

    Rank-uniform by construction, which is the load-bearing property: the
    purity rule is static boot config and the active layout is replicated
    (a cutover commits on every rank or on none). A rank-local input here
    would split the group across branches with mismatched collectives --
    the failure family documented at ``_update_uniform_pool_budget``.
    """
    from sglang.srt.managers.phase_policy import PHASE_TP

    if _active_phase(scheduler) != PHASE_TP:
        return False
    if purity_of(scheduler).prefill_allowed_in_tp():
        return False
    return not _relaxed(scheduler, "prefill")


def decode_blocked_here(scheduler, running_bs: int) -> bool:
    """True when a decode step must NOT execute this iteration."""
    from sglang.srt.managers.phase_policy import PHASE_PP

    if _active_phase(scheduler) != PHASE_PP:
        return False
    if purity_of(scheduler).decode_allowed_in_pp(running_bs):
        return False
    return not _relaxed(scheduler, "decode")


def purity_of(scheduler) -> PhasePurity:
    """The scheduler's purity rule, resolved once and cached."""
    cached = getattr(scheduler, "_phase_purity", None)
    if cached is not None:
        return cached
    purity = purity_from_server_args(scheduler.server_args)
    scheduler._phase_purity = purity
    return purity
