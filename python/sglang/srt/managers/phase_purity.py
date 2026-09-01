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
    #: #887. Only meaningful for ``strict``: how many CHUNKS of genuinely
    #: computed prefill the TP layout may run per TP phase before the strict
    #: prohibition applies again. 0 -- the default on every mode -- is exactly
    #: the pre-#887 behaviour. See ``prefill_allowed_in_tp_now``.
    tp_compute_chunk_budget: int = 0

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

        THE MODE QUESTION, NOT THE ROUND QUESTION, and #887 makes the two
        distinct rather than collapsing them. This answers "may the
        break-even machinery run in TP at all", which is what BOOT-TIME
        consumers need: ``scheduler.py`` collapses the policy's threshold from
        it, and ``model_runner_kv_cache_mixin`` sizes a survivable TP prefill
        from it. Both must go on reading ``False`` under a budgeted ``strict``
        -- the one-chunk valve is a bounded exception, not a mode change, and
        the flip is still demanded for the pending prefill behind it.

        No threshold form HERE on purpose: the amortisation is not a second
        purity knob, it is ``phase_policy.break_even_tokens`` -- the pending
        prefill above which paying the seam beats prefilling in the slower
        layout. ``strict`` forces it to 0 and every prefill flips.

        The per-ROUND question is ``prefill_allowed_in_tp_now``.
        """
        return self.mode in (MODE_OFF, MODE_PREFILL_IN_TP)

    def prefill_allowed_in_tp_now(self, chunks_spent: int) -> bool:
        """#887: may a prefill batch be built in TP THIS ROUND?

        THE CLASS THIS METHOD EXISTS TO END: a YES/NO where the user's rule is
        a BUDGET. The user permitted the TP phase to prefill up to ONE CHUNK
        itself (2026-08-25, during the #857 acceptance boot) -- and the code
        had nowhere to put a "once". ``prefill_allowed_in_tp`` takes no token
        argument at all, so ``strict`` meant never and ``prefill_in_tp`` meant
        always, with the quantity delegated to a different axis entirely: the
        policy's break-even N, measured at 13791 tokens in
        WINDOW_TICKET_874.md -- 3.4x the permission that was given. Either side
        of that gap was measured (W29-RESULT.md): 153 TP prefill batches under
        ``prefill_in_tp``, 0 under ``strict``. The truth the user asked for is
        between them and no signature could express it.

        SCOPE IS THE TP PHASE, NOT THE PROCESS. "Die TP-Phase darf bis zu EINEM
        Chunk selbst prefillen" is per phase; ``chunks_spent`` is re-keyed on
        the flip epoch by ``tp_compute_chunks_spent``, so a cutover restores
        the allowance without a reset hook anybody has to remember to call.

        COMPUTE ONLY. A HiCache/radix restore never reaches this method: the
        seam-transport exemption is checked ABOVE it in ``prefill_blocked_here``
        and returns unconditionally. That is the user's own qualification --
        *"wenn das hicache reinladen als prefill gilt, dann darf es das
        natuerlich ueber einen chunk hinaus tun"* -- carried by ORDER rather
        than by an exemption clause that could rot.
        """
        if self.prefill_allowed_in_tp():
            return True
        return self.tp_compute_budget_remaining(chunks_spent) > 0

    def tp_compute_budget_remaining(self, chunks_spent: int) -> int:
        """Chunks of computed TP prefill still owed in this phase. Never < 0."""
        return max(0, int(self.tp_compute_chunk_budget) - max(0, int(chunks_spent)))

    def describe(self) -> str:
        if self.mode == MODE_THRESHOLD:
            return f"threshold:{self.decode_in_pp_threshold}"
        if self.mode == MODE_STRICT and self.tp_compute_chunk_budget > 0:
            return f"{MODE_STRICT}:{self.tp_compute_chunk_budget}"
        return self.mode


def parse_purity(raw: Optional[str]) -> PhasePurity:
    """Parse ``strict`` / ``threshold:<n>`` / ``off``. Loud on anything else."""
    if raw is None or raw == "":
        return PhasePurity(mode=DEFAULT_PURITY)
    value = str(raw).strip().lower()
    if value == MODE_STRICT:
        return PhasePurity(mode=MODE_STRICT)
    if value.startswith(MODE_STRICT + ":"):
        # #887: `strict:<n>` -- strict, with n chunks of COMPUTED prefill
        # permitted in the TP layout per TP phase. Written where the mode is
        # written, and in the grammar `threshold:<n>` already established, so
        # an operator reading one boot line sees the rule AND its exception
        # rather than having to correlate a mode with a separate env var.
        rest = value[len(MODE_STRICT) + 1 :]
        try:
            chunks = int(rest)
        except ValueError:
            raise PhasePurityError(
                f"{LOG_PREFIX} purity {raw!r} has a non-integer chunk budget "
                f"{rest!r}; write 'strict:<n>', e.g. 'strict:1' to let the TP "
                f"phase compute at most ONE chunk of prefill per TP phase"
            )
        if chunks < 0:
            raise PhasePurityError(
                f"{LOG_PREFIX} purity chunk budget {chunks} is negative; use 0 "
                f"(which is exactly 'strict') or a positive count of chunks"
            )
        # `strict:0` COLLAPSES STRUCTURALLY, not by a branch, and the #887
        # mutation harness is what established the difference: a
        # `if chunks == 0: return PhasePurity(mode=MODE_STRICT)` special case
        # sat here first and SURVIVED its own mutant -- deleting it changed no
        # behaviour, because the frozen dataclass already compares equal to a
        # bare `strict` at budget 0 and `describe()` already prints `strict`.
        # A guard that cannot fail is not a guard; it is a claim that the
        # collapse needs defending when it does not. `threshold:0` needs its
        # branch because it maps onto a DIFFERENT mode; this one does not.
        return PhasePurity(mode=MODE_STRICT, tp_compute_chunk_budget=chunks)
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
        (
            "allowed"
            if purity.prefill_allowed_in_tp()
            # #887: "forbidden" alone would misreport a budgeted strict as
            # the mode it is not. The exception is a deliberate operator
            # choice and has to be legible in the one line that states the
            # rule, or nobody can tell the two strict boots apart.
            else (
                f"forbidden, EXCEPT {purity.tp_compute_chunk_budget} computed "
                f"chunk(s) per TP phase (#887; a HiCache restore is separate "
                f"and unbounded)"
                if purity.tp_compute_chunk_budget > 0
                else "forbidden"
            )
        ),
    )
    return purity


def validate_tp_exit_pair(purity: PhasePurity, policy_cfg) -> None:
    """#858b: the TP mirror of `validate_purity_policy_pair`, and it was missing.

    PP has a bounded window and that guard says why: a phase that may not
    decode and cannot admit prefill "has NO exit except the bounded window".
    TP has no such bound -- ``--phase-policy-tp-decode-floor-s`` is a MINIMUM
    dwell, not a maximum -- and under strict the TP phase may not admit
    prefill either. So a tp_to_pp arm that waits for a chunked prefill to
    finish waits for work the TP layout forbids, with nothing to time it out.

    THE ASYMMETRY IS THE MISSING EXIT. Measured on
    boot_w40_857strict_0825_1931: 225 of 228 quiescence holds were tp_to_pp,
    258 ADMISSION-WEDGE reports, 11 queued / 0 running, no first token for
    535 s. The runtime predicate that produced them was later DELETED
    outright (#1065, 2026-09-01: the strict quiescence clause and its
    runnability plumbing are gone -- an incomplete chunk no longer holds a
    flip at all); this refuses the deadlocking
    CONFIGURATION at parse time, where refusing is free -- the same reason
    the PP guard exists rather than a runtime recovery.

    Checkable triple: strict purity + strict drain mode + no bounded TP
    residency.

    #894 S3 -- THERE IS EXACTLY ONE BOUNDED TP EXIT, AND IT IS THE SLO.
    This function used to read a second escape,
    ``getattr(policy_cfg, "tp_window_s", 0.0)``, and let ``tp_window > 0``
    return early. ``PhasePolicyConfig`` has no ``tp_window_s`` field and never
    had one, so that read was the constant ``0.0``: the branch was UNREACHABLE,
    the refusal was stricter than its own source read, and -- the part that
    cost operator time -- the message below told the reader to "set a bounded
    TP window", a knob that does not exist as a flag, an env var or a config
    field. Following the instruction changed nothing and hit the same refusal.

    The repair is the DELETION, not a new knob. A ``tp_window_s`` field would
    need a consumer inside ``phase_policy.decide``; there is none, and adding
    one here would manufacture a fresh instance of the class #889/#894 exist to
    close (a shipped knob that produces nothing and says nothing). Behaviour is
    unchanged for every real configuration, because the branch was already
    dead. What changes is that the code and the message stop describing an
    escape that is not there. ``TestNoPhantomPolicyFields`` in
    ``test_silent_superseded_knobs_894.py`` walks this module's AST and fails
    on the next defaulted read of a name the dataclass does not carry.
    """
    if not purity.strict:
        return
    if not bool(getattr(policy_cfg, "drain_mode_strict", False)):
        return
    slo = float(getattr(policy_cfg, "decode_stall_slo_s", 0.0) or 0.0)
    if slo > 0:
        return
    raise PhasePurityError(
        f"{LOG_PREFIX} purity={purity.describe()} with strict drain mode has "
        f"no bounded TP residency: decode_stall_slo_s={slo!r} is the only "
        f"bound this layout has, and it is not declared. `tp_decode_floor_s` "
        f"is a MINIMUM dwell and cannot end a hold, and there is no TP "
        f"counterpart to `pp_window_s` -- the SLO is the whole of the TP exit. "
        f"Under strict the TP phase may not admit prefill, so a tp_to_pp arm "
        f"waiting on a pending prefill waits for work this layout forbids and "
        f"nothing times it out -- the PP guard above refuses exactly this "
        f"shape for the other direction. Declare "
        f"SGLANG_PHASE_POLICY_DECODE_STALL_SLO_S "
        f"(--phase-policy-decode-stall-slo-s), or run --phase-flip-purity off."
    )


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


#: #887: the scheduler attribute holding ``(epoch, chunks_spent)`` for the
#: one-chunk compute exception. KEYED ON THE EPOCH RATHER THAN RESET AT THE
#: CUTOVER, deliberately: the allowance is scoped to a TP phase, and a ledger
#: whose scope is derived from the event cannot drift from it the way a
#: separate reset hook can be forgotten at one of the seam's several commit
#: paths. A stale key simply reads as a fresh phase, which is the correct
#: answer for the phase that key belongs to.
TP_COMPUTE_LEDGER_ATTR = "_tp_compute_prefill_ledger"


def _tp_phase_epoch(scheduler) -> int:
    """The flip epoch, or 0 when there is no runtime to ask.

    Group-uniform: the epoch advances on a cutover that commits on every rank
    or on none, which is the same property the seam's own readmit stamp relies
    on (``phase_flip_runtime`` stamps ``seam_readmit_epoch`` from it). A
    rank-local input here would split the group across branches with mismatched
    collectives -- the family this whole module argues against.
    """
    rt = getattr(scheduler, "phase_flip_runtime", None)
    try:
        return int(getattr(rt, "epoch", 0) or 0)
    except (TypeError, ValueError):
        return 0


def tp_compute_chunks_spent(scheduler) -> int:
    """Chunks of computed TP prefill this TP phase has already spent."""
    book = getattr(scheduler, TP_COMPUTE_LEDGER_ATTR, None)
    try:
        epoch, spent = book
    except (TypeError, ValueError):
        return 0
    if epoch != _tp_phase_epoch(scheduler):
        return 0
    return max(0, int(spent))


def tp_compute_budget_remaining(scheduler, purity=None) -> int:
    """Chunks of computed TP prefill still owed in this TP phase."""
    rule = purity_of(scheduler) if purity is None else purity
    return rule.tp_compute_budget_remaining(tp_compute_chunks_spent(scheduler))


def tp_compute_fits_in_one_chunk(scheduler, inflight=None) -> Optional[bool]:
    """#887: does ALL the pending prefill fit inside ONE chunk? None = unknown.

    THE GATE AND THE #838 DETECTOR MUST BE ONE RULE, and without this term they
    were two. `layout_conformance.tp_compute_exception_verdict` permits a batch
    only when ``new_tokens < chunk_tokens`` -- #870's measured discriminator,
    whose whole argument is that a batch REACHING the cap was TRUNCATED by it,
    so more prefill stands behind it and what is really running in TP is a large
    cold prefill. A gate that granted the chunk regardless would hand the batch
    builder a round whose resulting batch its own detector then calls a
    violation: the instance alarming on the exception it was configured to take,
    once per TP phase, for as long as a big prefill is pending.

    IT IS ALSO WHAT THE USER ASKED FOR. The permission is for the case where
    letting TP finish the work "einfacher funktionieren wuerde" -- the request
    completes in this layout and no cutover is needed for it at all. When more
    than a chunk is pending the cutover is coming anyway, and computing 4096
    tokens at TP's 1681 tok/s instead of PP's 7245 spends ~1.9 s of the slow
    layout on work the flip was going to do properly. That is the throughput
    extension the rule explicitly reserves for a separate user decision.

    UNKNOWN IS REFUSAL, never permission -- the same direction as an unresolved
    chunk size at the detector. A scheduler stand-in without the accessor or
    without a chunk size gets None and the strict rule stands.

    Rank-uniform on this module's standing argument: read in the TP layout only,
    where ``waiting_queue`` and ``chunked_req`` are replicated across the ranks
    -- the property ``seam_readmit_candidates`` already relies on.
    """
    chunk = getattr(getattr(scheduler, "server_args", None), "chunked_prefill_size", 0)
    try:
        chunk = int(chunk or 0)
    except (TypeError, ValueError):
        return None
    if chunk <= 0:
        return None
    fn = getattr(scheduler, "_pending_prefill_tokens", None)
    if not callable(fn):
        return None
    try:
        # #942c: ASK THE SAME QUANTITY THE FLIP POLICY COMPARES AGAINST.
        #
        # `_pending_prefill_tokens` takes the #713 `inflight` batch -- requests
        # pulled off the wire that have NOT yet reached `waiting_queue` -- and
        # its own docstring says "passing it is what the flip policy must do".
        # This probe is read on the flip-policy path and did NOT pass it, so on
        # a FRESH ARRIVAL it measured 0 while the policy measured the real
        # prompt. `0 < 0 < chunk` is False, the grant collapsed to 0, and the
        # #942 suppression could never fire for the one case #887 exists for.
        # Measured on boot_855_1011idle: 11 grants, 0 suppressions, 10 tp-ward
        # arms on `pending prefill 51 tok > 0`.
        #
        # Default None keeps every other caller byte-identical (the #363
        # observer's quantity and the break-even denominator, which the
        # docstring warns must not move). The fallback keeps stand-in
        # schedulers whose accessor takes no argument working as before,
        # rather than turning a signature mismatch into a silent refusal.
        try:
            pending = int(fn(inflight) or 0)
        except TypeError:
            pending = int(fn() or 0)
    except Exception:  # noqa: BLE001 - an unreadable queue is not a permission
        return None
    # Strictly LESS than the chunk, matching the detector's boundary exactly:
    # pending == chunk would fill the batch to the cap and be flagged there.
    return 0 < pending < chunk


def _spend_tp_compute_chunk(scheduler) -> None:
    """Book one chunk against this TP phase's allowance.

    SPENT AT THE GRANT, NOT AT THE BATCH, and the direction of that choice is
    the safe one. The grant is what ``prefill_blocked_here`` hands the batch
    builder; if the builder then finds nothing to build, the allowance is gone
    for this phase. That is conservative in the direction the user's law runs
    -- never MORE than the permitted chunk -- and it keeps the ledger on the
    one rank-uniform decision path rather than on a measured instrument that
    is wrapped in a blanket ``try/except`` and may legitimately not run.
    """
    spent = tp_compute_chunks_spent(scheduler)
    try:
        setattr(
            scheduler, TP_COMPUTE_LEDGER_ATTR, (_tp_phase_epoch(scheduler), spent + 1)
        )
    except Exception:  # noqa: BLE001 - a ledger may never break the gate
        pass


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
    # W31: SEAM TRANSPORT OUTRANKS EVERYTHING BELOW, AND ORDER IS THE WHOLE
    # POINT OF PUTTING IT HERE.
    #
    # This check first sat after `prefill_allowed_in_tp`, which is BELOW the
    # drain-mode suppression. W31 arm 1 measured what that costs: with
    # `--phase-policy-drain-mode` in the recipe, `prefill_suppressed_in_tp`
    # returned True and this function returned before the exemption was ever
    # evaluated. The seam retracted 87 requests across 39 pp_to_tp flips, and
    # `SEAM TRANSPORT ADMITTED` was logged 0 times, `Prefill batch phase=tp` 0
    # times -- the W30 livelock reproduced exactly, with the fix for it
    # installed and unreachable. Same defect shape as the ORDER bug recorded
    # in this very function ("What broke was ORDER -- suppression was checked
    # FIRST and returned True, so the valve never ran").
    #
    # AND IT IS NOT MERELY A PLACEMENT CONVENIENCE. Drain mode's own stated
    # reason for forbidding TP prefill is that "a TP window entered to finish
    # a bundle must not admit the work it was entered to escape". A request
    # the cutover ITSELF retracted is not that work: it IS the bundle this
    # window was entered to finish, sent back to the queue by the seam a
    # moment earlier. Suppressing it does not defend the drain contract, it
    # makes the contract unsatisfiable -- the bundle can never complete.
    if seam_transport_exempt(scheduler) and seam_transport_premise_holds(scheduler):
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
    purity = purity_of(scheduler)
    if purity.prefill_allowed_in_tp():
        return False
    # #887 THE ONE-CHUNK EXCEPTION, and it sits HERE for two reasons that are
    # both about order.
    #
    # BELOW DRAIN-MODE SUPPRESSION, which therefore still outranks it. Drain
    # mode's contract is "a TP window entered to finish a bundle must not admit
    # the work it was entered to escape", and its suppression lifts by itself
    # at ``running_bs == 0`` (phase_policy: "with running_bs == 0 the bundle is
    # finished"). That is exactly the state the valve exists for -- the #858
    # wedge was 11 queued / 0 running -- so the exception reaches its own case
    # without being able to interrupt a draining bundle.
    #
    # BELOW THE SEAM-TRANSPORT EXEMPTION at the top of this function, which is
    # what carries the user's own qualification: *"wenn das hicache reinladen
    # als prefill gilt, dann darf es das natuerlich ueber einen chunk hinaus
    # tun"*. A verified restore returns up there and never reaches the ledger,
    # so unbounded restore is a property of ORDER, not of a second exemption
    # clause that could rot out of agreement with the first.
    #
    # SPENDING HERE IS SPENDING ON THE ONE DECISION PATH. This function is
    # called once per round from ``get_next_batch_to_run``; the hypothetical
    # ``target_can_admit`` probe asks ``prefill_allowed_in_tp_now`` directly and
    # books nothing, which is what keeps a probe from emptying the valve
    # without a batch ever being built (the W33 divergence class, in the new
    # currency). Rank-uniform on this function's own standing argument: static
    # purity config, a replicated active layout, and a group-unanimous epoch.
    if purity.tp_compute_chunk_budget > 0:
        spent = tp_compute_chunks_spent(scheduler)
        # AND THE WORK MUST FIT, or the gate and the #838 detector are two
        # rules: the detector permits only a batch BELOW the chunk cap, so
        # granting a round whose batch fills the cap makes the instance alarm on
        # its own configured exception. See `tp_compute_fits_in_one_chunk`.
        if purity.prefill_allowed_in_tp_now(spent) and tp_compute_fits_in_one_chunk(
            scheduler
        ):
            _spend_tp_compute_chunk(scheduler)
            if not getattr(scheduler, "_tp_compute_exception_announced", False):
                scheduler._tp_compute_exception_announced = True
                logger.warning(
                    "%s ONE-CHUNK EXCEPTION TAKEN: purity=%s permits %d chunk(s) "
                    "of COMPUTED prefill per TP phase and this phase has now "
                    "spent %d. Granted by the user on 2026-08-25 as a valve "
                    "against the #858 shape (TP may admit no prefill -> hold "
                    "with no exit -> wedge), NOT as a relaxation of the "
                    "strict-batch mode: the flip is still demanded for the "
                    "prefill behind this chunk, and the allowance returns at "
                    "the next cutover. A HiCache restore does not spend it.",
                    LOG_PREFIX,
                    purity.describe(),
                    purity.tp_compute_chunk_budget,
                    spent + 1,
                )
            return False
    # (The seam-transport exemption is checked at the TOP of this function --
    # see the W31 note there. It must outrank the drain-mode suppression, so
    # it cannot live down here.)
    return not _relaxed(scheduler, "prefill")


#: W30: attribute the seam stamps on every request it retracts under the #856
#: no-carry rule. Set ONLY in `build_cutover_release._retract`; deliberately
#: NOT `Req.is_retracted`, which decode-OOM preemption sets too.
SEAM_READMIT_ATTR = "seam_readmit_epoch"

#: Scheduler flag naming the round in which the seam-transport exemption is
#: open, read by the prefill builder to keep the batch to transport only.
SEAM_TRANSPORT_ROUND_ATTR = "_seam_transport_round"

#: #890: attribute stamped on a request whose seam restore was REFUSED where it
#: is executed, i.e. whose tokens the refusal sends back to be RECOMPUTED.
#: Written only by `schedule_batch.restore_seam_state`'s two refusal branches
#: and cleared there by a restore that actually happens, so it is a statement
#: about the last attempt rather than a life sentence. Read by
#: `seam_transport_premise_holds`, which is the whole point: the exemption is
#: granted on the claim that a re-admission recomputes nothing, and this is the
#: one signal that says the claim was false for this request.


#: #906: the grant this request has already SPENT. The seam stamp
#: (`SEAM_READMIT_ATTR`) says a cutover retracted this request; it does not say
#: how much prefill that buys. One chunk is the answer, and this is the ledger
#: entry that makes it one -- set where the exempt admission happens, cleared
#: only by a fresh cutover stamp.
SEAM_GRANT_CONSUMED_ATTR = "seam_grant_consumed"


def seam_grant_is_open(req) -> bool:
    """Does this request still hold an UNSPENT seam-transport grant? #906.

    THE LEAK THE USER SAW. `seam_transport_exempt` derives the whole round's
    exemption from "a stamped request exists", and the stamp is a fact about
    the past that never expires. Live specimen: a 20 000-token wave running
    `phase=tp` under SEAM-TRANSPORT-CREDIT, chunk after chunk, EVERY admitted
    chunk `#cached-token: 0` -- the premise ("this is a restore, it recomputes
    nothing") contradicted by the scheduler's own batches, and nothing in the
    chain able to notice, because one stamp licensed an unbounded number of
    chunks.

    A GRANT IS A QUANTITY, NOT A PROPERTY. The exemption's justification is
    per-chunk: THIS chunk's tokens were already computed in the PP window and
    are being read back. Nothing in that argument extends to the next chunk,
    which is why the grant is spent when it is used and the next chunk needs
    its own. #858's shape, and #501's before it: a permission checked at issue
    and never debited at execution.

    WHY CONSUMPTION IS EXPRESSED HERE AND NOT AT THE TWO GATES. This predicate
    feeds `seam_readmit_candidates`, which is THE ONE AUTHORITY with two
    callers (`seam_transport_exempt` for "may TP build a prefill batch" and
    `seam_transport_pending_tokens` for "do these tokens demand a flip back to
    PP") -- the file's own rule, one predicate, one clock. Debiting here moves
    BOTH in the same step, and that is what makes the refusal safe:

        a spent grant drops the request out of candidacy
          -> `seam_transport_exempt` stops exempting the round (no more
             unbounded TP prefill), AND
          -> `seam_transport_pending_tokens` stops SUBTRACTING its remaining
             tokens from pending prefill, so the policy sees them as ordinary
             prefill and arms `tp_to_pp`.

    THAT SECOND HALF IS THE WHOLE ANTI-WEDGE ARGUMENT and it is why the debit
    could not be a local check at the builder. The transport tokens are
    excluded from `pending_prefill_tokens` precisely so the tp-ward flip does
    not undo itself; refusing the next chunk while still excluding its tokens
    would leave the request with no layout willing to run it and no policy
    input able to say so -- a mid-prefill wedge, which is the one outcome this
    posten may not produce. Spending the grant hands the remainder back to PP,
    where it is honest work in the right layout.

    RANK-UNIFORM: the stamp comes from a group-unanimous cutover and the debit
    happens in the replicated builder on the same round on every rank, so the
    candidate list stays identical across ranks -- the property the purity
    branch is required to have.
    """
    if getattr(req, SEAM_READMIT_ATTR, None) is None:
        return False
    return not bool(getattr(req, SEAM_GRANT_CONSUMED_ATTR, False))


def consume_seam_grant(req) -> bool:
    """Debit this request's seam grant. Returns True if one was open.

    Idempotent: spending an already-spent grant is not an error, it is the
    common case on the second chunk and must stay quiet.
    """
    if not seam_grant_is_open(req):
        return False
    try:
        setattr(req, SEAM_GRANT_CONSUMED_ATTR, True)
    except Exception:  # noqa: BLE001 - a ledger entry may never break a build
        return False
    return True


def reissue_seam_grant(req) -> None:
    """A fresh cutover stamp is a fresh grant. #906.

    Called where the seam stamps `SEAM_READMIT_ATTR`, so a request retracted
    again by a later cutover is transported again -- the grant is per
    retraction, not per lifetime.
    """
    try:
        setattr(req, SEAM_GRANT_CONSUMED_ATTR, False)
    except Exception:  # noqa: BLE001
        pass


def seam_readmit_candidates(scheduler) -> list:
    """Queued requests the #856 cutover retracted and must re-admit.

    Read off ``waiting_queue``, which is replicated across the ranks
    (scheduler.py's own note: "``self.waiting_queue``, which is replicated
    across the TP ranks"), so this list is the same on every rank in the same
    round -- the property the purity branch is required to have.

    #906: a request whose grant is SPENT is no longer a candidate. See
    `seam_grant_is_open` for why the debit belongs here rather than at either
    gate -- both of this function's callers must move together, or refusing
    the next chunk wedges the request instead of sending it back to PP.
    """
    out = []
    for req in getattr(scheduler, "waiting_queue", ()) or ():
        if seam_grant_is_open(req):
            out.append(req)
    return out


def seam_transport_pending_tokens(scheduler) -> int:
    """Pending prefill tokens that are SEAM TRANSPORT, not PP workload (W32).

    THE ONE AUTHORITY, WITH TWO CALLERS. `prefill_blocked_here` asks
    `seam_transport_exempt` whether a TP prefill batch may be built; the phase
    policy asks THIS whether those same tokens justify flipping back to PP.
    Both answers are derived from `seam_readmit_candidates` -- one predicate,
    one clock. That is deliberate and it is the whole lesson of the last two
    windows: the exemption was implemented as a COPY at one enforcement site
    while another site kept its own, and a correct mechanism that a second
    copy overrides is a mechanism that never runs.

    W32 measured the cost, in the policy's own words, 23 times:

        PHASE-POLICY arming tp_to_pp: pending prefill 1 tok > 0
          (purity: prefill cannot run in tp)

    The seam had just re-admitted its retracted residents, so those tokens
    appeared as "pending prefill". The policy's copy of the purity rule read
    that as work requiring the PP layout and armed tp_to_pp -- leaving TP
    before the exemption at the batch builder could be consulted. The
    exemption fired ONCE in 144 pp_to_tp flips; the other 143 times the
    instance had already flipped away.

    THESE TOKENS ARE NOT PP WORK. They are the seam's own re-admission,
    destined to be served in the layout the flip just entered, by a
    read-through that recomputes nothing. Counting them as pending PP prefill
    makes the tp-ward flip undo itself, every time.

    Excluded at the INPUT BOUNDARY rather than at each trigger, so every
    consumer of `pending_prefill_tokens` in the policy -- the drain exit, the
    break-even band, the tp-ward arm -- sees the same, correct quantity. A
    per-trigger subtraction would be a fourth copy of the same judgement.
    """
    total = 0
    for req in seam_readmit_candidates(scheduler):
        try:
            n = len(getattr(req, "origin_input_ids", None) or ())
            done = int(getattr(req, "cache_protected_len", 0) or 0)
            total += max(0, n - done)
        except Exception:  # noqa: BLE001 - an accounting probe, never a gate
            continue
    return total


def seam_transport_premise_holds(scheduler) -> bool:
    """Is the re-admission actually a RESTORE? Checked, not asserted. #861d.

    THE EXEMPTION RESTS ON A FACTUAL CLAIM, AND THE CLAIM IS FALSIFIABLE.
    ``seam_transport_exempt`` permits prefill in the TP layout on the ground
    that "the tokens were already prefilled in the PP window, their KV is in
    the canonical store, and the re-admission recomputes nothing -- it is a
    cache restore". That is what makes it an exemption rather than a breach of
    the user's law: transport is not work.

    W37-D FALSIFIED IT ON METAL. 258 TP prefill batches, every one
    ``#new-token: 4096, #cached-token: 0``, mean cached 0.0 % -- and
    ``#cached-token`` was 0 on ALL 1441 occurrences in the whole boot, PP
    included. HiCache wrote (write_backup 207, load_to_device 207, fence 573)
    and served ZERO hits (storage_hit 0). So the re-admission was not restoring
    anything; it was re-prefilling 4096 cold tokens in the decode layout, 258
    times. Real work, wrong layout, the user's law broken -- by an exemption
    whose justification was true only in design.

    SAME CLASS AS #861c/F1, and that is the finding rather than this function:
    a guard or an exemption that ASSERTS a premise in prose and never verifies
    it at runtime. F1 assumed two host pools have equal slot counts; this
    assumed a re-admission hits cache. Both were wrong for a whole window, both
    were silent, and both are fixed the same way -- check the premise where it
    is relied upon.

    THE CHECK: a restore was COMPUTED AND FENCED, and the evidence must
    survive the retraction that raises the question. The first cut read
    ``cache_protected_len`` -- but ``reset_for_retract`` zeroes that field on
    every request the seam stamps, so the premise was structurally False for
    the whole population it judges (#861j: three boots, zero decode batches,
    this refusal never even reached because the policy armed away first).
    The surviving evidence is ``cached_prompt_tokens_at_retract``, stamped
    from the measured fill boundary BEFORE the fields are cleared: non-zero
    means the tokens were computed in the PP window and the fence persisted
    them. A cold request (retracted before any fill) carries 0 and stays
    refused -- the can-fail direction. Whether the canonical store then
    actually SERVES the prefix is measured where it can be:
    ``layout_conformance.work_in_wrong_layout`` scores the batch's own
    ``cached_tokens``, so a transport batch that recomputes is loud, not
    silent (the W37-D falsifier, kept).

    WHY THIS IS SAFE AGAINST THE W30 LIVELOCK, and the ordering matters: the
    exemption exists because W30 ping-ponged with ZERO decode batches, and the
    ROOT of that ping-pong was the blind counter #861c/F2 fixed -- the policy
    read 0 pending and never demanded the flip. With the existence term in
    place a held TP prefill DEMANDS the flip to PP, and PP admits it, so the
    hold now has somewhere to go. The exemption was a workaround for a defect
    that is fixed; narrowing it to its own premise is safe only BECAUSE F2
    landed first.

    LOUD, ONCE. A refused exemption is a state somebody has to be able to see.
    """
    reqs = seam_readmit_candidates(scheduler)
    if not reqs:
        return False
    restored = 0
    revoked = 0
    for req in reqs:
        try:
            # #890: THE GRANT IS WITHDRAWN BY THE EXECUTION THAT DISPROVED IT.
            #
            # Everything below this line is EVIDENCE -- what was true when the
            # request was retracted. This is OUTCOME: `restore_seam_state`
            # refused the copy and said so in its own words, "Dropped; these
            # tokens are recomputed" (W38: 90 and 21 occurrences). A recompute
            # is the one thing the exemption promised would not happen, so the
            # request stops counting as restore evidence until a restore
            # actually succeeds for it again.
            #
            # IT CANNOT BE FOLDED INTO THE EVIDENCE TERM, which is why it is a
            # field of its own: the recompute the refusal forces re-fills the
            # prefix, so the NEXT retraction re-stamps
            # `cached_prompt_tokens_at_retract` from the measured boundary and
            # the evidence reads "computed and fenced" at exactly the moment
            # the copy has proven unusable. The premise would then be issued
            # again on a claim this request has already falsified on metal --
            # the #501 shape (a grant checked at issue, never revoked at
            # execution).
            # #1043: the #890 revocation is deleted WITH the carry it existed
            # to describe. Its ground -- 'the copy was dropped and the tokens
            # go back to be recomputed' -- was measured false: the store
            # serves the prefix. With no carry there are no carry failures to
            # revoke on, and the read-through re-admission is no longer forced
            # under the #887 one-chunk cap by a false premise -- which is the
            # user order quoted at the top of this module: the HiCache
            # load-in, counting as prefill, MAY exceed one chunk.
            # #861j: EVIDENCE THAT SURVIVES THE RETRACTION. The seam's own
            # `reset_for_retract` zeroes `cache_protected_len` on every
            # request it stamps, so keying the premise on that field alone
            # refused the exact population the exemption exists for -- the
            # W37-D manufactured-state class, committed by the check meant to
            # close it (three metal boots, zero `Decode batch phase=`,
            # SEAM TRANSPORT REFUSED unreachable behind the policy's own
            # arm). `cached_prompt_tokens_at_retract` is stamped from the
            # measured fill boundary BEFORE those fields are cleared and is
            # exactly the claim this premise rests on: the tokens WERE
            # computed and the fence persisted them.
            if (
                int(getattr(req, "cache_protected_len", 0) or 0) > 0
                or int(getattr(req, "cached_prompt_tokens_at_retract", 0) or 0) > 0
            ):
                restored += 1
        except (TypeError, ValueError):
            continue
    if restored:
        # #890 EDGE-TRIGGERED, on this module's own rule for exactly this shape:
        # "Cleared on recovery, so a flapping rig logs each engagement rather
        # than only the first in the process's life" (`_drain_yield_announced`).
        # The refusal below latched instead -- which would have made the
        # revocation this fix installs observable exactly once per process, and
        # a revocation nobody can see recur is a revocation nobody can measure.
        if getattr(scheduler, "_seam_premise_refused_announced", False):
            scheduler._seam_premise_refused_announced = False
        return True
    if not getattr(scheduler, "_seam_premise_refused_announced", False):
        scheduler._seam_premise_refused_announced = True
        logger.error(
            "%s SEAM TRANSPORT REFUSED: the exemption's premise does not hold. "
            "%d stamped request(s) are queued and NONE carries restore "
            "evidence (cache_protected_len=0 AND "
            "cached_prompt_tokens_at_retract=0 for all), or %d of them had "
            "their restore REFUSED at execution (#890 -- the copy was dropped "
            "and the tokens go back to be recomputed), so re-admitting "
            "them in the TP "
            "layout would be a COLD PREFILL of real work, not a cache restore "
            "-- the user's strict-batch law, broken by the exemption meant to "
            "respect it. Holding instead; the #861c existence term raises the "
            "flip demand and the work runs in the layout that owns it. "
            "Measured W37-D: 258 such batches at #cached-token 0; W38: 90 and "
            "21 SEAM RESTORE REFUSED (LAYOUT).",
            LOG_PREFIX,
            len(reqs),
            revoked,
        )
    return False


def seam_transport_deduction(tokens, *, in_tp: bool, premise_holds: bool) -> int:
    """#869c: how many seam-transport tokens may be deducted from pending.

    ONE RULE FOR BOTH SUBTRACTIONS, which is the whole point of this function
    existing rather than a second inline condition.

    THE DEFECT IT CLOSES. The economics subtraction (``_pending_now -=
    _seam_transport_now``) ran UNCONDITIONALLY, while its twin twelve lines
    below -- the #861j serviceable credit -- was gated on the TP phase AND on
    ``seam_transport_premise_holds``. Both rest on the SAME justification: that
    a seam re-admission is cheap flip transport rather than real workload,
    because "their prefixes are served by read-through from the canonical
    store". #861j verified that premise for the existence term and never
    backported the verification to the economics term.

    WHY THE PREMISE MUST BE CHECKED AND NOT ASSUMED. When read-through cannot
    serve those prefixes, the tokens are not transport at all -- they are a full
    cold prefill of real work, and deducting them tells the policy that work
    does not exist. The consumer that pays is the one still reading RAW pending:
    the #677(a) blocked-admission stall escape, whose threshold is
    ``pending > pp_exit_tokens``. A deflated pending holds a genuine stall below
    its own escape, which is the wedge that escape was written to end.

    PHASE, TOO, for the same reason the twin carries it: a re-admission is only
    "transport that will land here" in the layout that can admit it. In PP the
    stamp buys nothing, so the deduction has no premise to stand on at all.

    Pure and total, so both directions are falsifiable without a scheduler.
    Returns the DEDUCTION, never the remainder -- the caller keeps its own
    ``max(0, ...)`` floor, so this can never manufacture negative pending.
    """
    if not (in_tp and premise_holds):
        return 0
    return max(0, int(tokens or 0))


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
