"""AP L15-09 (2026-09-30): the hot D->P flip -- ordering note and pure state machine.

ORDERING NOTE -- the steps of a D->P flip, in today's order (every anchor verified
against this tree on 2026-09-30):

  1. D quiesce / publish. The front parks D's decodes (the park RPC, recorded by
     front_state_ipc.py:281 ``DpFlipClock.note_park`` -- the clock's decode-end) and
     opens the flip (front_state_ipc.py:287 ``DpFlipClock.begin``: "A D->P flip
     begins").
  2. D sleep leg (kv pause). D publishes, joins and PAUSES its kv_cache tag. A
     paused pool is UNMAPPED memory -- touching it is the fault W114 names, never a
     report of it (weight_updater.py:9770). In the D->P direction this pause is
     what frees the VRAM the waking side lives from (a shard, ~1.35 GB per tag --
     wake_kv.py:50).
  3. P wake leg (weights). The waking group's weight tags come over the BAR1
     lanes -- "D->P flip: TP<src> deposits into PP<dst>" (bar1_lanes.py:599
     ``Bar1Lanes``) -- and are collected per tag (weight_updater.py:6034
     ``_weg2_wake_collect_one``).
  4. P kv resume (early / defer / late / done, wake_kv.py:31 ``wake_kv_plan``,
     called at weight_updater.py:9963). The fit is checked per rank against the
     card's free bytes (weight_updater.py:9747 ``kv_resume_fit_refusal``) with a
     bounded WAIT because the sleeper's leg "may still be releasing"
     (weight_updater.py:9754 ``wait_for_kv_fit``); the resume itself maps P's pool
     (weight_updater.py:9784); a mid-leg variant exists (weight_updater.py:10343
     ``kv_mid_ok``); the verdict is GROUP-UNIFORM
     (weight_updater.py:9803 ``_weg2_kv_group_verdict``): the tag is resumed on
     every card or on none.
  5. Lanes up. The depositor's credit cycle
     (bar1_lanes.py:759 ``Bar1Lanes.deposit_cycle``) runs on the chain stated at
     wake_kv.py:50: "a deposit completes only when the collect runs, the collect
     only after the resume, the resume only with VRAM".
  6. First P leg 1. The flip clock fires on the first leg 1 dispatched after the
     flip (front_state_ipc.py:307 ``DpFlipClock.first_prefill``).

THE WINDOW. A window in which D's pool is still mapped AND P's kv tag is already
resumed on every card DOES NOT EXIST today. Step 2 unmaps D's pool BEFORE step 4
can map P's, and it must: the resume may run "only with VRAM" and "nothing may be
counted on the peer's later pauses" (wake_kv.py:50 -- xsn376 stalled 120 s in
exactly the D->P direction, resume band ~2.9 GB vs pause shard ~1.35 GB, "the pool
taken early starved that chain"); where the sleeper has not released yet, the
waking rank WAITS (weight_updater.py:9754) instead of overlapping the two
mappings. So by the time the group verdict lands P's tag on the last card, D's
pages are gone and the old path re-reads the whole context from L2/L3.

WHAT MUST MOVE. The hot handover needs that window, so the kv pause inside step 2
must move AFTER step 4's group verdict: D keeps its pool mapped through P's wake
(P's resume then funded without D's pause -- L1.5's idle P-layout VRAM on the
3080s is that tier), the handover copies D's prefix KV + anchor into P's rows
[0, n) while BOTH are mapped, and only then does D pause. The rid-keyed manifest
through /dev/shm (written at handoff.py:39, read once at handoff.py:55, #1442) is
the existing control-plane pattern this machine's ``manifest_complete`` stands for.

THIS MODULE IS PURE (no CUDA, no I/O, no env): the plan picker :func:`decide` and
the per-request state machine :class:`Handover`, so the ordering and the fallback
logic can be tested and wired before any boot runs on them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

PLANNED = "planned"
D_DEPOSITED = "d_deposited"
P_ADOPTED = "p_adopted"
FALLEN_BACK = "fallen_back"


@dataclass(frozen=True)
class HandoverPlan:
    """One hot handover: ``n_tokens`` tokens of prefix at ``anchor_depth``, to be
    landed on P's rows starting at ``p_row0``."""

    rid: str
    n_tokens: int
    anchor_depth: int
    p_row0: int = 0


def decide(candidates: Sequence[Mapping], p_free_rows: int) -> Optional[HandoverPlan]:
    """Pick the handover for THIS flip, or ``None`` to fall back.

    ``candidates`` is the front's list of follow-up prefill(s) for P, dicts
    ``{rid, hot_in_d, prefix_tokens, anchor_depth}``. The first candidate (in
    INPUT order) whose three conditions all hold wins; a candidate failing any
    condition is skipped and the next one is tried:

    * ``hot_in_d`` -- its KV is resident on the D side;
    * ``anchor_depth == prefix_tokens`` -- F3: never KV without its anchor at
      depth (an anchor shallower or deeper than the prefix means the pages the
      handover would copy are not the pages the anchor vouches for);
    * ``prefix_tokens <= p_free_rows`` -- P's rows can carry it.

    P destination convention: the handover lands on rows ``[0, n)`` right after
    P's reset, hence ``p_row0 = 0``.
    """
    for c in candidates:
        if not c.get("hot_in_d"):
            continue
        prefix = int(c.get("prefix_tokens", 0))
        anchor = int(c.get("anchor_depth", 0))
        if anchor != prefix:
            continue  # F3: KV without its anchor at depth is not handable over
        if prefix > int(p_free_rows):
            continue
        return HandoverPlan(
            rid=str(c.get("rid", "")),
            n_tokens=prefix,
            anchor_depth=anchor,
            p_row0=0,
        )
    return None


class Handover:
    """One plan's journey ``planned -> d_deposited -> p_adopted``, with the only
    legal exits: ``fallen_back`` (named reason) from ``planned`` or
    ``d_deposited``. Anything else raises ValueError naming from/to -- a flip
    that "lost" a handover silently is the failure class this machine exists to
    make loud (the W114 rule: name the refusal, never skip it silently)."""

    def __init__(self, plan: HandoverPlan) -> None:
        self.plan = plan
        self.state = PLANNED
        self.reason = ""
        self.bytes_local = 0
        self.bytes_lane = 0

    # -- legs ------------------------------------------------------------
    def deposit_done(self, bytes_local: int, bytes_lane: int) -> str:
        """D's deposit landed: ``bytes_local`` straight into P's pool (the card
        window of the hot handover) and ``bytes_lane`` over the BAR1 lanes.
        Legal only from ``planned``."""
        self._go(D_DEPOSITED, allowed_from=(PLANNED,))
        self.bytes_local = int(bytes_local)
        self.bytes_lane = int(bytes_lane)
        return self.state

    def adopt(self, manifest_complete: bool, fp_agree: bool) -> str:
        """P checked the handover: the manifest carries every page the plan
        names, and the fingerprint agrees with what D deposited. ``p_adopted``
        only when BOTH hold; else a named ``fallen_back`` (the manifest is
        checked first -- pages that are not there explain a mismatch too).
        Legal only from ``d_deposited``: adopting before the deposit is a lie
        about a copy that never ran."""
        self._go(P_ADOPTED, allowed_from=(D_DEPOSITED,))
        if not manifest_complete:
            self.state, self.reason = FALLEN_BACK, "incomplete_manifest"
        elif not fp_agree:
            self.state, self.reason = FALLEN_BACK, "disagree"
        return self.state

    def fall_back(self, reason: str) -> str:
        """Give up on THIS handover with a NAMED reason (the old path --
        resume_via_p's full re-prefill -- still serves the request). Legal from
        ``planned`` and from ``d_deposited``; a terminal state stays terminal."""
        self._go(FALLEN_BACK, allowed_from=(PLANNED, D_DEPOSITED))
        self.reason = str(reason)
        return self.state

    # -- plumbing ----------------------------------------------------------
    def _go(self, to: str, allowed_from: Sequence[str]) -> None:
        if self.state not in allowed_from:
            raise ValueError(
                "hot-handover %s: illegal transition %s -> %s"
                % (self.plan.rid, self.state, to)
            )
        self.state = to

    @property
    def line(self) -> str:
        """One log line per state, the WEG2 way: name it, do not hint at it."""
        return "HOT-HANDOVER rid=%s n=%d state=%s reason=%s" % (
            self.plan.rid, self.plan.n_tokens, self.state, self.reason)


# -- L15-10 S2: real candidates from the front ------------------------------

def front_candidates(queue, sess_prev, d_live) -> list:
    """decide()'s input from the front's own state at a D->P flip.

    ``queue``: the waiting follow-ups (objects with ``.rid``), in order;
    ``sess_prev``: rid -> (previous rid of the same session, common token
    prefix) -- the SESSION-PREFIX bookkeeping; ``d_live``: rids D still holds
    (running or parked: their KV is on D). A follow-up is hot when its
    session's previous rid is live on D; its handable prefix is the common
    token prefix, and the anchor sits at that depth (D parks/ends a request at
    its END anchor, the prefix the follow-up shares)."""
    out = []
    for q in queue:
        rid = str(getattr(q, "rid", ""))
        prev = (sess_prev or {}).get(rid)
        hot = bool(prev) and str(prev[0]) in d_live
        n = int(prev[1]) if hot else 0
        out.append({"rid": rid, "hot_in_d": hot, "prefix_tokens": n,
                    "anchor_depth": n})
    return out


def plan_line(epoch: int, candidates, p_free_rows: int) -> str:
    """One HOT-HANDOVER-PLAN line: waiting, hot count, hot tokens, and the
    handover decide() would choose (none -> the store read of today)."""
    hot = [c for c in candidates if c.get("hot_in_d")]
    plan = decide(candidates, p_free_rows)
    chosen = ("chosen=%s n=%d" % (plan.rid, plan.n_tokens)) if plan else "chosen=none"
    return ("HOT-HANDOVER-PLAN epoch=%d waiting=%d hot=%d hot_tokens=%d %s"
            % (int(epoch), len(candidates), len(hot),
               sum(int(c.get("prefix_tokens", 0)) for c in hot), chosen))
