"""S42 HEAD-VOTE HOLD: the head vote is measured on the passes that read it, not in every decode round.

MEASURED (S40, boot ..._49f9c2ea4d_1018_133242.D.log, TP0, 14:13:00-14:13:59, ``#queue-req: 6``): 902 decode rounds
at 66 ms each; ``DECODE-HOST-SPLIT sched_ms=56.9`` against 2.6-3.2 ms without a waiting request. Per round 4x
``#1042 EXTENT LIFECYCLE set`` (one per priced waiting rid, ~155x each for the rids 36-117/119/124, 38-126) and ~14
``MAMBA-HOST-RESUME`` acceptances, no ``X-GATE`` line in the whole minute: the admission pass itself was held by DQH
(weg2/d_head_hold.py) -- what ran every round was the #823 head vote at the budget site.

THE ROOT: ``Scheduler._update_uniform_pool_budget`` builds the head arms of its packed MIN reduce on EVERY TP-loop
iteration -- ``_local_head_prefix_matches`` (a full ``match_prefix_for_req`` per canonical waiting rid: the extent
stamp, the mamba host-resume walk, the match census), the usable arm (on Form A ``admission_probe``: a second match
plus a list copy of the whole prompt, ~3.4 ms per 114k-token rid at the desk), the store and completion arms. Those
values have exactly one reader: the admission pass of the same iteration (``_take_uniform_head_inputs`` /
``_uniform_usable_floor``, consume-once). DQH holds that pass while nothing the head's verdict reads has changed, so
on a held iteration the vote was measured for nobody. DQH (08.10., FIXES-NACH-FREEZE Nr. 9, Planer's Commit 3) held
the PASS and left its INPUTS running.

THE RULE: on group D the DQH decision of an iteration is taken ONCE, at the budget site, after the prefetch drain
(its fingerprint reads the landed verdicts) and before the head vote. Held -> the head arms contribute their
MIN-neutral abstention (no match, no probe, no store or completion vote) and the pass is held -- the elif chain
consumes the same decision, it does not ask again. Evaluated -> the vote is measured fresh, exactly as before, and
the pass runs on it. So every pass that READS the vote reads one measured in its own iteration; a vote is never
reused.

GROUP-UNIFORM (SOLL R11): the decision is DQH's, from replicated inputs (head rid, waiting and running rid sets,
landed prefetch verdicts, the binding generation) and a pass counter every rank advances alike -- every rank holds
on the same iteration and contributes the same neutral payload. Should a rank ever hold alone, its abstention
MIN-reduces the group's head arms to ABSENT: the evaluating ranks DELAY the head (no skew, no realize round on any
rank) -- never a collective-count split, never a forced admission.

RE-EVALUATION (when the vote is measured again): a new arrival or a left request (waiting set), a finished or
retracted decode (running set -- the space it held is free), a landed prefetch, a flip/cutover (binding
generation), a chunked continuation; and at the latest every ``d_head_hold.REEVAL_EVERY_PASSES`` held passes, for
what is rank-local (device evictions, host arena, write-through acks). A change that lands between the budget site
and the elif chain of the same iteration is taken by the NEXT iteration's budget site (one round), because the vote
of this iteration was not measured.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: the decision of THIS iteration, written at the budget site, consumed by the elif chain
DECISION_ATTR = "_s42_dqh_decision"
#: set by get_next_batch_to_run right before it calls the budget site (only that caller has the elif chain)
ARMED_ATTR = "_s42_head_vote_armed"

_STATS = {"held": 0, "measured": 0}


def arm(sched: Any) -> None:
    """get_next_batch_to_run, right before ``_update_uniform_pool_budget``: this budget site is followed by the
    elif chain that consumes the decision. A stale decision from an iteration that never reached the chain goes."""
    d = sched.__dict__
    d.pop(DECISION_ATTR, None)
    d[ARMED_ATTR] = True


def disarm(sched: Any) -> None:
    """Right after the budget site: an early return there (single rank, offload path) took no decision; the arm
    must not reach another caller of the budget site."""
    sched.__dict__.pop(ARMED_ATTR, None)


def decide(sched: Any, running_batch: Any, prefetch_verdicts: Optional[dict]) -> bool:
    """Budget site, after the drain, before the head vote. True = this iteration's pass is held, so the head arms
    abstain. Only an armed call (get_next_batch_to_run) decides; any other caller of the budget site measures as
    before and leaves no decision."""
    if not sched.__dict__.pop(ARMED_ATTR, False):
        return False
    dqh = getattr(sched, "weg2_d_head_hold", None)
    if dqh is None:
        return False
    if running_batch is None:  # the budget site's own rule: None reads self.running_batch
        running_batch = getattr(sched, "running_batch", None)
    fp = sched._weg2_d_head_fingerprint(running_batch, prefetch_verdicts)
    held = bool(dqh.should_hold(fp))
    sched.__dict__[DECISION_ATTR] = held
    _STATS["held" if held else "measured"] += 1
    return held


#: S42r A3 group-split counter (status line), not a decision input
_SPLIT = {"n": 0}


def build_hold_payload(held: bool) -> list:
    """S42r A3: this rank's hold vote as an (x, -x) pair on the packed MIN reduce -> the group's MIN and MAX."""
    h = 1 if held else 0
    return [h, -h]


def resolve_group(sched: Any, reduced_pair) -> Optional[bool]:
    """S42r A3 (review nf-review-s42-1018): replace this rank's decision by the GROUP's.

    ``reduced_pair`` = the MIN-reduced (h, -h) slots: ``lo = MIN(h)``, ``hi = -MIN(-h) = MAX(h)``. Only an armed
    decision (DECISION_ATTR present) is replaced; an unarmed caller voted 0 and keeps none.

    held_group = hi: the pass is held when ANY rank held. A holding rank abstained from every head arm, so the
    group's head is ABSENT this iteration; a rank that ran its pass on it would see the X gate ABSTAIN (no refusal
    on an absent match) and reach ``add_one_req``'s Form A per-candidate verdict gather ALONE -- a collective the
    holding ranks never enter (a14cf046ba claimed "divergent DQH only delays, same collective count"; that held
    for the budget-site reduce only). lo != hi = a split: every rank disarms DQH, so the next iteration evaluates
    on every rank, jointly, on a measured vote."""
    d = sched.__dict__
    if DECISION_ATTR not in d:
        return None
    lo, hi = int(reduced_pair[0]), -int(reduced_pair[1])
    held = hi >= 1
    if lo != hi:
        _SPLIT["n"] += 1
        dqh = getattr(sched, "weg2_d_head_hold", None)
        if dqh is not None:
            dqh.state.fingerprint = None
            dqh.state.held = 0
        n = _SPLIT["n"]
        if n <= 8 or (n & (n - 1)) == 0:
            logger.warning(
                "S42r DQH GROUP-SPLIT n=%d local_held=%s group_min=%d group_max=%d: the ranks decided the hold "
                "differently -- the whole group holds this pass (no rank runs a pass on an abstained head vote) "
                "and every rank re-evaluates the next one; a split means a DQH input was not replicated",
                n, d.get(DECISION_ATTR), lo, hi,
            )
    d[DECISION_ATTR] = held
    return held


def take(sched: Any) -> Optional[bool]:
    """The elif chain: the budget site's decision of this iteration (consumed), or None when the budget site took
    none (single rank, offload path, an unarmed caller) -- then the chain asks DQH itself, as before."""
    return sched.__dict__.pop(DECISION_ATTR, None)


def stats() -> dict:
    return dict(_STATS)


def reset_for_tests() -> None:
    for k in _STATS:
        _STATS[k] = 0
    _SPLIT["n"] = 0
