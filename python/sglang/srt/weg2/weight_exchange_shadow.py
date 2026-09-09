# SPDX-License-Identifier: Apache-2.0
"""#1273 slice S5 -- THE SHADOW.  The metal proof with ZERO AUTHORITY.

WEG2_REUSE_SPEC_0908 section 6 / S5.  This module runs the whole exchange --
S1's plan, S3's region and gates, S4's two lanes -- on a real flip leg, into
buffers nothing reads, and then compares what it pulled against the bytes the
HOST RING actually restored.  The ring stays authoritative throughout: its
bytes land as they land today, this module observes them.

**WHY IT EXISTS.**  It is the only measurement anywhere in the record of BOTH
GROUPS MOVING AT ONCE (ADDENDUM 1 section H.4 says in those words that this was
never measured, and every ordering conclusion of spec section 3.4 rests on a
one-direction arm plus today's ``waited=0.000s``).  And it is the only
device-space proof of the fused sub-block offsets: the 24/24 IDENTICAL-SUBBLOCK
probe is CHECKPOINT-space (spec section 2.4 rule 1, R2-6), so a wrong device
offset writes ``k`` over ``q`` on ranks 1 and 2 -- plausible garbage, no error,
the worst class this campaign has.  S5 finds that at zero risk; S6 would find
it with a live model.

**ZERO AUTHORITY, ENUMERATED, because the first thing that will be asked of a
shadow that finds something is exactly the thing it may not do:**

1. it may not vote a gate (its own gate is separate and advisory);
2. it may not refuse a flip -- every refusal here decides only whether the
   SHADOW runs;
3. it may not change a slot state on the authoritative path;
4. it may not call ``vote_failure``;
5. it may not raise into the leg: :func:`shadow_transport` catches
   ``BaseException`` and turns it into a logged line -- with ONE stated arm,
   ``explicit=True``, where an operator asked for a named class subset by hand
   and gets :class:`Weg2XchgShadowUnaffordable` instead of a silent degrade
   (the W56 two-arm shape; it has no producer on the boot path, TODO(S6));
6. it may not hold the flip: every wait here is bounded by ITS OWN small
   budget, never by ``WEG2_GROUP_FENCE_BUDGET_S``, because a shadow that can
   block a leg for 120 s has authority over the flip's wall.

Point 6 is a DEVIATION from spec section 3.2's "no second timeout constant",
and it is the one deviation this module takes: that rule exists so the
authoritative path cannot grow a second, differently-tuned deadline.  A budget
that bounds an observer is not that; using the fence budget here would be.

**THE MISMATCH IS COUNTED, NEVER ACTED ON.**  W59 names the class, the stripe,
the byte counts and both checksums, and the flip continues.  Spec section 3.3
rule 3 is obeyed at the one place it matters: ``checksum_is_representable``
(``model_executor/weights_arena.py:133``) is asked BEFORE anything is called a
mismatch, because a value outside ``[0, 255 * nbytes]`` was never a checksum
and #656 register C22 killed an instance for a corruption that had not
happened.
"""

from __future__ import annotations

import ctypes
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.srt.weg2 import weight_exchange_transport as tp

MIB = xr.MIB

# ---------------------------------------------------------------------------
# Names and markers.
# ---------------------------------------------------------------------------

#: The third value of ``--weg2-weight-source``.  ``ring`` is today's path,
#: ``exchange`` is S6's, and this one is ring-authoritative WITH the exchange
#: running beside it into buffers nobody reads.
WEIGHT_SOURCE_SHADOW = "shadow"

SHADOW_LINE_PREFIX = "WEG2-XCHG-SHADOW"
SHADOW_GATE_LINE_PREFIX = "WEG2-XCHG-SHADOW-GATE"
SHADOW_BUDGET_LINE_PREFIX = "WEG2-XCHG-SHADOW-BUDGET"

MISMATCH_MARKER = "W59 Weg2XchgShadowMismatch"
UNAFFORDABLE_MARKER = "W61 Weg2XchgShadowUnaffordable"

#: S5c.  THE ACCEPTANCE LINE OF THE DERIVED PLAN, one per leg per rank.
#:
#: It shares its prefix with S1's ``emit_plan_line`` on purpose -- one event
#: name for "this is the plan that will move" -- and the two are told apart by
#: their FIRST field: S1's is ``dir=``, this one is ``card=``.  A grep for
#: ``WEG2-XCHG-PLAN `` gets both, which is what a reader asking "what did this
#: boot plan" wants; ``grep 'WEG2-XCHG-PLAN card='`` is the shadow's own.
PLAN_LINE_PREFIX = "WEG2-XCHG-PLAN"

#: S5c.  THE PER-CLASS COMPARE LINE.  ``mismatch_message`` (W59) names a class
#: only when something went WRONG, so a boot whose shadow agreed carried no
#: per-class evidence at all -- the reader could see ``match=7`` and not which
#: seven.  This line is emitted for MATCH and MISMATCH alike, one per class,
#: and it is what item 4 of the S5c brief asks for.
COMPARE_LINE_PREFIX = "WEG2-XCHG-SHADOW-COMPARE"

#: W64 -- the six ranks did not derive the SAME plan.
#:
#: The gate already refused six ranks that chose different CLASS SUBSETS
#: (``classes_hash``).  S5c gives the plan a product producer, and a producer
#: has failure modes a subset hash cannot see: a rank whose chunk geometry came
#: from a stale environment, a rank whose wave partition names a tag this
#: boot's family does not carry, a rank that enumerated a different class
#: rotation because its card carries different tags.  All four are "the ranks
#: are about to shadow different things", all four are silent today, and the
#: subset hash agrees in every one of them because the subset is CHOSEN from
#: the rotation and two different rotations can offer the same name.
#:
#: TWO SCOPES, ONE CODE, the same shape as W61's ``scope=vram|hop``:
#:
#: * ``scope=group`` -- the rows this hook expects do not carry one
#:   ``plan_digest``.  The digest is over the GROUP-UNIFORM derivation facts
#:   (chunk geometry, the family tag tuple, the wave partition, the card
#:   vector, the class rotation and the producer that answered), so a
#:   disagreement is a boot-configuration divergence and never a property of
#:   which card a rank sits on.
#: * ``scope=oncard-peer`` -- this rank and the CO-LOCATED rank of the other
#:   group derived different CARD geometry for the same classes.  That one is
#:   not a configuration fault: it is the honest reading of "the two groups do
#:   not hold the same bytes on this card", which on the P=PP / D=TP form of
#:   this rig is the expected answer.  It is refused BY NAME here rather than
#:   left to become a W54 byte-count disagreement three seams later inside the
#:   on-card consumer, or -- worse -- a red compare that reads as a defect in
#:   the exchange when it is a statement about the two layouts.
PLAN_DIVERGED_MARKER = "W64 Weg2XchgShadowPlanDiverged"

#: The comparison granularity, spec section 6/S5: "per-64-MiB-stripe checksum
#: on device, no host round trip".  It is ALSO the size of the scratch the spec
#: budgets ("VRAM delta = 64 MiB per consumer rank"), and deliberately the same
#: number: the scratch is what makes the sum a device sum, so a stripe larger
#: than it would need a second one.
STRIPE_BYTES = 64 * MIB

#: The shadow's own waits.  Small, and NOT ``xr.fence_budget_s()`` -- see the
#: module docstring, deviation 6.
SHADOW_GATE_BUDGET_S = 5.0
SHADOW_TRANSPORT_BUDGET_S = 30.0

#: THE ONE WALL A FLIP LEG PAYS FOR HAVING AN OBSERVER, and the only bound the
#: user law ("never delays the leg's own completion beyond a named bound")
#: can be read off in one place.
#:
#: MEASURED-BY-REVIEW DEFECT (S5b refuter, must_fix 4): the only ENFORCED bound
#: was :data:`SHADOW_HOP_BOUND_MS_DEFAULT` (60 ms), which grades the priced
#: on-card hop and nothing else -- while the attach, the gate wait (5 s), the
#: transport (30 s) and the compare all sat on the leg's critical path outside
#: it.  The honest bound on the added wall was therefore the SUM of three
#: constants a reader had to find in three places, and the code enforced one of
#: them.  Here the two WAITS are carved out of ONE deadline: whatever the gate
#: spends, the transport cannot spend again, so the two together can never
#: exceed this number no matter how the sub-budgets are tuned.
#:
#: TWICE FALSE WHEN WRITTEN, and both halves are fixed rather than re-worded
#: (S5c refuter, must_fix 2 and 6):
#:
#: * the transport got ONE NUMBER and performs MANY waits, each restarting its
#:   own clock -- worst case ``(batches + slots + 1) x`` this constant.  It now
#:   receives the DEADLINE as a callable (``budget_left``) and re-evaluates it
#:   per wait; see ``weight_exchange_transport._wait_budget``.
#: * the DERIVATION ran in the adapter before the hook's clock started and was
#:   outside the number entirely.  It is now subtracted from the deadline and
#:   printed as ``derive_ms=``, and ``budget=ok|OVER`` grades
#:   :attr:`ShadowResult.observer_ms`, which is the sum.
#:
#: ITS VALUE IS THE GATE BUDGET, DELIBERATELY, and that is the whole rule: the
#: shadow may cost the flip ONE rendezvous, never a rendezvous plus a transport
#: plus a compare.  It is a policy number sitting on top of S5's, not a fourth
#: measured constant -- the same shape as :data:`SHADOW_HOP_BOUND_FACTOR`.
SHADOW_HOOK_BUDGET_S = SHADOW_GATE_BUDGET_S

#: The floor the shadow's VRAM is priced against.  ``Reserve-Semantik``: 1024
#: MiB per card is the user's free space, not an internal allowance, and the
#: VRAM corridor band (819-1229 MiB NVML-free per card under load) is measured
#: in the same regime the shadow runs in -- a live boot with load -- which is
#: what makes it the right floor here and a conservative one in W55's unloaded
#: window.
SHADOW_FLOOR_MIB = 1024.0

#: WHY THE 5090 WILL REFUSE AT FULL SIZE, with the reading that says so.  Boot
#: weg2sb5f measured **474 MiB** NVML-free on the 5090 under load -- already
#: below the 1024 MiB floor before the shadow asks for a byte.  So the shadow
#: does NOT get to shadow a whole leg on this rig: it takes a bounded CLASS
#: SUBSET, rotating, and prints the subset on its line.  A shadow sized by hope
#: would OOM the flip it is observing, which is the one failure a zero-
#: authority instrument must not be able to cause.
SB5F_FREE_MIB_NOTE = (
    "boot weg2sb5f measured 5090 free=474 MiB under load, below the 1024 MiB "
    "corridor floor -- the full-leg shadow is refused on this rig by "
    "arithmetic, not by preference"
)


class Weg2XchgShadowMismatch(RuntimeError):
    """W59 -- a pulled stripe differs from the destination's own restored bytes.

    **RAISED ONLY WHERE IT CANNOT REACH A FLIP** (a test, or a deliberate
    ``strict=True`` caller).  On the flip path it is a COUNTER and a log line:
    the shadow's whole point is that the exchange has no authority yet, and an
    instrument that stops the thing it is measuring has stopped being one.

    The message names the tensor CLASS, the stripe, the byte range and both
    checksums -- never "checksum mismatch" alone, which is what a reader cannot
    act on (spec 6/S5 red-first ``test_shadow_mismatch_names_the_parameter``).
    """


class Weg2XchgShadowPlanDiverged(RuntimeError):
    """W64 -- the ranks of one leg did not derive the same plan.

    NOT a flip refusal, exactly like every other refusal in this module: it
    switches the SHADOW off for this leg and the ring proceeds untouched.  It
    is raised only on the ``explicit=True`` arm (an operator who asked for the
    shadow by name), and on the automatic path it is a verdict reason, a log
    line under :data:`PLAN_DIVERGED_MARKER` and a ``vote=no`` row.

    WHY IT IS A SEPARATE CODE FROM THE CLASS-SUBSET DISAGREEMENT the gate
    already reports: that one says "six ranks picked different classes OUT OF
    THE SAME LIST", which is a rotation bug; this one says "six ranks are not
    reading the same list at all", which is a boot-configuration or a layout
    fact.  Folding the two would make a census of "legs the ranks disagreed on"
    unable to say which question they disagreed about, and the two have
    different fixes.
    """


class Weg2XchgShadowUnaffordable(RuntimeError):
    """W61 -- the shadow's buffers do not fit this card's free column.

    NOT a flip refusal.  Raised only when a caller EXPLICITLY demanded a class
    subset (``classes=``, which has no launcher flag yet: TODO(S6)); the
    automatic path logs the same message and votes NO in the shadow gate.  That
    gate is RANK-UNIFORM, so one card's refusal switches the shadow off on ALL
    SIX RANKS for that leg -- not "shadows nothing on that card", which is what
    this sentence used to say and is not what the gate does (S5 refuter,
    wording).  A shadow that refuses a FLIP would have taken the authority this
    slice exists not to have; refusing ITSELF is the whole mechanism.  Same
    shape as W56's two arms in S4: logged when the degrade is automatic, raised
    when the operator asked for the thing that is unavailable.
    """


# ---------------------------------------------------------------------------
# The shadow's rows in the dir area.  S4 owns [DIR_OFF, DATA_OFF); this is the
# next free byte after its release area, and the disjointness is a test.
# ---------------------------------------------------------------------------

SHADOW_AREA_OFF = tp.DIR_USED_BYTES
#: S5c WIDENED IT FROM 64 TO 128, and the two new words are why.
#:
#: The row carried seven words plus its seal, which filled 64 bytes exactly, so
#: the plan digest and the card fingerprint had nowhere to go.  Folding either
#: into ``classes_hash`` was the alternative and is refused: the gate reports
#: that field's disagreement as "the ranks chose different class subsets", and
#: a plan divergence reported under that sentence sends the reader to the
#: rotation, which is not where it is.  The dir area has 832,768 bytes of slack
#: behind this table (``DATA_OFF - DIR_OFF - SHADOW_AREA_END``, asserted by
#: ``test_the_shadow_rows_are_disjoint_from_s4s``), so the space is free and
#: 128 keeps the row a power of two the way every other table here is.
SHADOW_ROW_BYTES = 128
#: ``epoch_hash, leg, vote, classes_hash, need_mib, pid, ts_ns, plan_digest,
#: card_digest`` + a seal.
SHADOW_ROW_STRUCT = struct.Struct("<9Q")
SHADOW_SEAL_OFF = SHADOW_ROW_STRUCT.size
SHADOW_AREA_BYTES = xr.N_RANKS * SHADOW_ROW_BYTES
SHADOW_AREA_END = SHADOW_AREA_OFF + SHADOW_AREA_BYTES

VOTE_NO = 0
VOTE_YES = 1


def _row_off(row: int) -> int:
    if not 0 <= int(row) < xr.N_RANKS:
        raise ValueError(f"row must be 0..{xr.N_RANKS - 1}, not {row!r}")
    return SHADOW_AREA_OFF + int(row) * SHADOW_ROW_BYTES


def write_shadow_vote(region: xr.XchgRegion, row: int, *, leg: int, vote: bool,
                      classes_hash: int, need_mib: int,
                      plan_digest: int = 0, card_digest: int = 0) -> None:
    """Publish this rank's shadow vote.  ONE WRITER PER ADDRESS, sealed.

    Same discipline as every other row in this region (S3's gate rows, S4's
    on-card rows): a rank writes only its own row, the row is sealed, and a
    half-written row is not a signal.

    ``plan_digest`` and ``card_digest`` default to 0 so a caller that has no
    derived plan (a refusal publishing its NO, S3's and S4's own callers) is
    byte-unchanged; the gate only COMPARES them, and a table of zeros is
    uniform, so a leg where nobody derived a plan is not turned into a
    divergence by the fields existing.
    """
    view = region.dir_view()
    off = _row_off(row)
    payload = SHADOW_ROW_STRUCT.pack(
        region.epoch_hash, int(leg), VOTE_YES if vote else VOTE_NO,
        int(classes_hash) & ((1 << 64) - 1), int(need_mib), os.getpid(),
        time.time_ns(), int(plan_digest) & ((1 << 64) - 1),
        int(card_digest) & ((1 << 64) - 1))
    addr = ctypes.addressof(view)
    ctypes.memmove(addr + off, payload, len(payload))
    ctypes.memmove(addr + off + SHADOW_SEAL_OFF,
                   struct.pack("<Q", xr._seal(payload)), 8)


def read_shadow_vote(region: xr.XchgRegion, row: int) -> Dict[str, int]:
    view = region.dir_view()
    off = _row_off(row)
    addr = ctypes.addressof(view)
    payload = ctypes.string_at(addr + off, SHADOW_ROW_STRUCT.size)
    seal = struct.unpack("<Q", ctypes.string_at(addr + off + SHADOW_SEAL_OFF, 8))[0]
    eh, leg, vote, classes_hash, need_mib, pid, ts_ns, plan_digest, \
        card_digest = SHADOW_ROW_STRUCT.unpack(payload)
    return {"epoch_hash": eh, "leg": leg, "vote": vote,
            "classes_hash": classes_hash, "need_mib": need_mib, "pid": pid,
            "ts_ns": ts_ns, "plan_digest": plan_digest,
            "card_digest": card_digest,
            "sealed": int(seal == xr._seal(payload))}


@dataclass(frozen=True)
class ShadowVerdict:
    """Whether the shadow runs on this leg, and why not when it does not."""

    run: bool
    joined: int
    refusers: Tuple[int, ...]
    reason: str
    waited_s: float
    #: THE SPREAD OF THE SIX RANKS' OWN PUBLICATION STAMPS, in ms -- which is
    #: what "skew" means.  MEASURED-BY-REVIEW finding (S5 refuter): the result
    #: line's ``gate_skew_ms`` carried ``waited_s``, i.e. THIS rank's wait,
    #: and grading spec 6/S5's ``gate_skew_ms <= 100`` against a wait
    #: conflates two different numbers -- a rank that arrives last waits ~0 and
    #: would have reported a skew of zero on the leg with the largest one.
    #: The rows already carry ``ts_ns``, so the real number is free.
    skew_ms: float = 0.0
    #: HOW MANY ROWS THIS RANK WAS ENTITLED TO EXPECT -- the denominator of
    #: ``joined=``, and never a constant again.  See :func:`shadow_gate`'s
    #: ``expect_rows``: the two hooks of one flip sit at OPPOSITE ends of it,
    #: so a source rank that waits for all six waits for rows that physically
    #: cannot exist yet, and printing ``joined=3/6`` for a complete rendezvous
    #: reads as a half-failed one.
    expected: int = xr.N_RANKS

    def line(self, *, leg: int, epoch: str) -> str:
        return (
            f"{SHADOW_GATE_LINE_PREFIX} leg={leg} epoch={epoch} "
            f"joined={self.joined}/{self.expected} "
            f"run={'yes' if self.run else 'no'} "
            f"refusers={','.join(str(r) for r in self.refusers) or 'none'} "
            f"waited_s={self.waited_s:.3f} skew_ms={self.skew_ms:.3f} "
            f"reason={self.reason}"
        )


def plan_divergence_message(*, scope: str, leg: int, epoch: str,
                            rows: Dict[int, int], field: str) -> str:
    """W64, naming the scope, the field and every row's own value.

    Every row is printed, not just the odd one out: with six rows and two
    values there is no "the odd one out", and a message that picks one has
    decided which side is wrong without knowing.
    """
    return (
        f"{PLAN_DIVERGED_MARKER} scope={scope} leg={leg} epoch={epoch} "
        f"field={field} "
        + " ".join(f"[row={r} {field}={v:#x}]" for r, v in sorted(rows.items()))
        + " -- the ranks of this leg did not derive the same plan, so the "
        "shadow switches itself off for this leg rather than compare two "
        "different experiments; the flip proceeds on the ring, which is and "
        "stays the only authority for weight bytes"
    )


#: THE PLACEMENT REFUSAL, one grep-able line.  Deliberately NOT a W-code: no
#: rank is wrong, nothing is corrupt, and a W-code in a boot log is a claim
#: that something went wrong.  What happened is that the observer was asked to
#: use a lane whose two ends are sequential on this placement, and it declined
#: before spending the flip's time on it.
ONCARD_NOT_DRAINABLE_PREFIX = "WEG2-XCHG-SHADOW-ONCARD-REFUSED"


def oncard_not_drainable_message(*, rank: int, row: int, peer_row: int,
                                 leg: int, epoch: str, is_source: bool,
                                 descs: int) -> str:
    """Why this leg compared digests and moved no bytes (S5c refuter, must_fix 3)."""
    return (
        f"{ONCARD_NOT_DRAINABLE_PREFIX} rank={rank} row={row} "
        f"peer_row={peer_row} leg={leg} epoch={epoch} "
        f"hook={'source' if is_source else 'destination'} oncard_descs={descs} "
        "-- the co-located end of this lane cannot run while this hook holds "
        "its scheduler thread (the source hook is upstream of the pause loop "
        "that publishes the C14 credit the co-located destination hook's "
        "resume is fenced on), so a producer here would fill a bounce, block "
        "in drain-final and time out at the budget for zero compared bytes; "
        "the gate rendezvous ran and plan_digest/card_digest WERE compared, "
        "the ring is and stays the only authority for weight bytes"
    )


def deposit_refusal_message(*, reason: str, rank: int, row: int, peer_row: int,
                            leg: int, epoch: str, is_source: bool,
                            batches: int, slots: int, slot_bytes: int,
                            budget_bytes: int,
                            slots_max: int = tp.ONCARD_SLOTS_MAX) -> str:
    """W65: this lane's store-and-forward deposit does not fit (#1273 S6).

    THE SIBLING OF :func:`oncard_not_drainable_message`, at the same placement
    and under the same contract: the gate rendezvous has already run, the two
    digests HAVE been compared, and no byte moves.  What differs is the claim.
    The other line says the lane's two ends are sequential on this placement --
    true of the hooks, nobody's fault, no W-code.  This one says a deposit was
    asked for whose slots cannot hold its batches, or whose host bytes the
    #1269 ledger did not charge for, and both of those are configurations that
    would move bytes into storage nothing guarantees.

    IT REPLACES THE OTHER REFUSAL ONLY WHERE THE BOUNCE REMOVES THE WAIT.  On
    the ``ipc`` arm the deposit cannot exist at all (an exported bounce dies
    with its leg), so that lane keeps the old, blameless line.
    """
    return (
        f"W65 Weg2XchgDepositUnfundable rank={rank} row={row} "
        f"peer_row={peer_row} leg={leg} epoch={epoch} "
        f"hook={'source' if is_source else 'destination'} reason={reason} "
        f"oncard_batches={batches} oncard_slots={slots} "
        f"oncard_slot_mib={slot_bytes / MIB:.0f} "
        f"deposit_mib={slots * slot_bytes / MIB:.0f} "
        f"host_budget_mib={budget_bytes / MIB:.0f} slots_max={slots_max} "
        "-- the store-and-forward deposit is what lets this hook's lane run "
        "with no concurrent peer (the source fills one slot per batch and "
        "returns; the destination reads the same shm file in its own later "
        "leg), and this shape does not fit: with fewer slots than batches a "
        "slot is overwritten before anything read it, and above the charged "
        "host budget the deposit is pinned host bytes no ledger term carries "
        "against a reap mark that is a hard bound.  The gate rendezvous ran "
        "and plan_digest/card_digest WERE compared, NO bytes moved, and the "
        "ring is and stays the only authority for weight bytes"
    )


def shadow_gate(region: xr.XchgRegion, row: int, *, leg: int, vote: bool,
                classes_hash: int, need_mib: int,
                log: Callable[[str], None],
                budget_s: float = SHADOW_GATE_BUDGET_S,
                poll_s: float = xr.GATE_POLL_S,
                monotonic: Callable[[], float] = time.monotonic,
                expect_rows: Optional[Sequence[int]] = None,
                plan_digest: int = 0, card_digest: int = 0,
                peer_row: Optional[int] = None) -> ShadowVerdict:
    """UNIFORM ACROSS THE ROWS IT MAY EXPECT: any of them refusing is a NO.

    ``expect_rows`` IS THE ROWS THIS RANK MAY EXPECT AT THIS INSTANT, and it
    defaults to all six -- S3's and S4's callers are unchanged.

    MEASURED-BY-REVIEW DEFECT (S5b refuter, must_fix 1), and it is a
    STRUCTURAL one rather than a tuning one: the two ``weight_updater`` hooks
    sit at OPPOSITE ENDS OF THE FLIP.  The source hook runs on the SLEEPING
    group before its ``pause``; the destination hook runs on the WAKING group
    after its ``resume`` and after the disk reload -- and on a co-located card
    the waking rank is BLOCKED on the C14 credit the sleeping rank publishes
    inside its pause loop, i.e. AFTER its own hook.  A source rank that waits
    for all six rows is therefore waiting for three rows that cannot exist
    until it stops waiting: a circular wait, broken only by the expiry, paid
    once per sleeping rank per flip, on the critical path of the credit a
    co-located waking rank is fenced on.  The shadow never ran, and the flip
    paid the full budget for it.

    So the ROWS ARE THE HOOK'S: the source expects its OWN group's three
    (the ranks that are at the same instant of the same flip), the destination
    expects all six (the source rows were sealed earlier in this same flip,
    with this region's ``epoch_hash`` and this ``leg``, so they are free).
    Spec section 6/S5 is written this way too -- step 1 pushes into the
    staging region before ``pause``, step 2 pulls "after ``family_complete``"
    -- so the asymmetry is the design's, not an accident of the placement.

    The #802 rule (``kv_reshard.py:1010``'s discipline, spec section 3.3)
    applied to an observer: the six ranks derive their own affordability from
    their own card's free column, and a subset of ranks running the exchange is
    not a smaller experiment -- it is a producer with no consumer, which blocks
    on a slot until a budget expires inside somebody's flip leg.

    THE EXPIRY IS A "NO", NOT A REFUSAL.  A rank that never published is a rank
    that is not going to answer, and the honest response from an instrument is
    to switch itself off for this leg and say so.  The alternative -- waiting
    the fence budget and then raising -- is the shadow taking the flip down,
    which is deviation 6 of the module docstring inverted.

    It is also why ``classes_hash`` is in the row: six ranks shadowing
    DIFFERENT class subsets is six half-experiments, and the disagreement is
    detectable here for free rather than as a W54 three seams later.
    """
    wanted = (tuple(range(xr.N_RANKS)) if expect_rows is None
              else tuple(sorted({int(r) for r in expect_rows} | {int(row)})))
    write_shadow_vote(region, row, leg=leg, vote=vote,
                      classes_hash=classes_hash, need_mib=need_mib,
                      plan_digest=plan_digest, card_digest=card_digest)
    started = monotonic()
    while True:
        rows = {i: read_shadow_vote(region, i) for i in wanted}
        fresh = [i for i, r in rows.items()
                 if r["sealed"] and r["epoch_hash"] == region.epoch_hash
                 and r["leg"] == int(leg)]
        if len(fresh) == len(wanted):
            refusers = tuple(i for i in wanted if not rows[i]["vote"])
            hashes = {rows[i]["classes_hash"] for i in wanted}
            if refusers:
                return ShadowVerdict(
                    False, len(fresh), refusers,
                    "a rank refused: " + " ".join(
                        f"[row={i} pid={rows[i]['pid']} "
                        f"need_mib={rows[i]['need_mib']}]" for i in refusers),
                    monotonic() - started, expected=len(wanted))
            if len(hashes) != 1:
                return ShadowVerdict(
                    False, len(fresh), wanted,
                    "the ranks chose different class subsets: " + " ".join(
                        f"[row={i} classes_hash={rows[i]['classes_hash']:#x}]"
                        for i in wanted),
                    monotonic() - started, expected=len(wanted))
            # W64, scope=group.  THE DERIVATION'S OWN AGREEMENT, checked in the
            # rendezvous that already exists rather than in a second one: every
            # field of this digest is read from the LAUNCHER's environment or
            # from a pure function of it (S5c refuter, must_fix 4 took the one
            # live-tensor field back out), so a disagreement here is a rank
            # reading a different boot configuration -- a stale chunk geometry,
            # a wave partition from another family -- and never a property of
            # which card the rank sits on.  A class-set disagreement is NOT
            # this check: it is ``classes_hash`` two branches up, which says so
            # in its own words.
            digests = {i: int(rows[i]["plan_digest"]) for i in wanted}
            if len(set(digests.values())) != 1:
                message = plan_divergence_message(
                    scope="group", leg=int(leg), epoch=str(region.epoch),
                    rows=digests, field="plan_digest")
                log(message)
                return ShadowVerdict(False, len(fresh), wanted,
                                     "plan-diverged-group",
                                     monotonic() - started,
                                     expected=len(wanted))
            # W64, scope=oncard-peer.  THE CO-LOCATED PAIR's own geometry, and
            # it is a DIFFERENT question from the one above: the two ranks that
            # share a card are the two ends of the on-card lane, and if they
            # framed the same classes differently the transport would discover
            # it as a W54 byte-count disagreement inside the consumer, after
            # the source has already filled a bounce and while it waits out its
            # drain.  Asked here it costs one comparison and names the layouts.
            if peer_row is not None and int(peer_row) in digests:
                pair = {int(row): int(card_digest),
                        int(peer_row): int(rows[int(peer_row)]["card_digest"])}
                if len(set(pair.values())) != 1:
                    message = plan_divergence_message(
                        scope="oncard-peer", leg=int(leg),
                        epoch=str(region.epoch), rows=pair, field="card_digest")
                    log(message)
                    return ShadowVerdict(False, len(fresh),
                                         tuple(sorted(pair)),
                                         "plan-diverged-oncard-peer",
                                         monotonic() - started,
                                         expected=len(wanted))
            stamps = [int(rows[i]["ts_ns"]) for i in wanted]
            return ShadowVerdict(True, len(fresh), (),
                                 f"all {len(wanted)} expected rows joined",
                                 monotonic() - started,
                                 (max(stamps) - min(stamps)) / 1e6,
                                 expected=len(wanted))
        if monotonic() - started >= budget_s:
            missing = tuple(i for i in wanted if i not in fresh)
            return ShadowVerdict(
                False, len(fresh), missing,
                f"only {len(fresh)}/{len(wanted)} expected ranks published a "
                f"vote for leg={leg} within {budget_s}s -- the shadow switches "
                f"itself off for this leg rather than wait inside the flip",
                monotonic() - started, expected=len(wanted))
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# Tensor classes and the rotating subset.
# ---------------------------------------------------------------------------


def tensor_class(param_name: str) -> str:
    """The spec section 2.2 CLASS of a parameter, from its own name.

    ``model.layers.7.self_attn.qkv_proj.weight`` -> ``qkv_proj``: the last name
    component that is not a leaf attribute (``weight``, ``bias``,
    ``weight_scale``, ...) and not a layer index.  The class is what section
    2.2 tables by copy primitive, and it is the unit the shadow rotates over,
    because the offsets that can be wrong are wrong for a whole class at once
    (R5: a wrong device offset writes ``k`` over ``q`` for every layer).
    """
    parts = [p for p in str(param_name).split(".") if p]
    if not parts:
        return "?"
    leafs = {"weight", "bias", "weight_scale", "weight_zero_point", "scale",
             "input_scale", "weight_packed", "weight_shape", "g_idx"}
    for part in reversed(parts):
        if part in leafs or part.isdigit():
            continue
        return part
    return parts[-1]


def classes_hash(classes: Sequence[str]) -> int:
    """A stable 64-bit hash of a class subset, for the gate row.

    ``hash()`` is per-process salted (PYTHONHASHSEED), so six ranks would
    publish six different numbers for the same subset and the gate would refuse
    every leg.  This is the same defect class as a plan id built from pointers.
    """
    return xr.epoch_hash("|".join(sorted(classes)))


@dataclass(frozen=True)
class ShadowSubset:
    """One leg's shadowed classes and the descriptors they select."""

    classes: Tuple[str, ...]
    descs: Tuple[object, ...]
    bytes_by_rank: Dict[int, int]
    rotation: Tuple[str, ...]
    leg: int

    @property
    def nbytes(self) -> int:
        return sum(int(d.nbytes) for d in self.descs)

    @property
    def hash(self) -> int:
        return classes_hash(self.classes)

    @property
    def tags(self) -> Tuple[str, ...]:
        """The TAGS the chosen classes fall in, for the acceptance line.

        Spec section 6/S5's line names ``tag=``, and its sentence is "per flip,
        ONE tag is shadowed, rotating".  THIS SLICE ROTATES OVER THE TENSOR
        CLASS INSTEAD, and the deviation is stated rather than papered over
        (S5 review, must_fix 5): the offsets that can be wrong are wrong for a
        whole CLASS at once (R5 -- a wrong device offset writes ``k`` over
        ``q`` for every layer of every tag), so a class is the unit that either
        proves or fails, and a tag is a save-order bucket the class cuts
        across.  The line therefore carries the tags the chosen classes
        actually touch, which is one tag in the common case and says so when it
        is not.
        """
        return tuple(sorted({str(d.tag) for d in self.descs}))


def rotation_of(descs: Sequence[object]) -> Tuple[str, ...]:
    """Every class in the plan, in a DETERMINISTIC order all six ranks share.

    Sorted, never plan order: the plan is sorted by
    ``(wave, dst_rank, param_name, ...)``, so two ranks holding different
    directed subsets would enumerate the classes in different orders and pick
    different ones for the same leg index.
    """
    return tuple(sorted({tensor_class(d.param_name) for d in descs}))


def select_subset(descs: Sequence[object], *, leg: int,
                  classes: Sequence[str] = (),
                  per_leg: int = 1) -> ShadowSubset:
    """The bounded class subset this leg shadows.

    ROTATING, and derived from the LEG INDEX rather than from anything a rank
    measures: every rank computes the same subset from the same sorted class
    list and the same flip counter, so the six agree without a message and the
    gate's ``classes_hash`` check is a cross-check rather than the mechanism.

    ``classes`` pins the subset explicitly (an operator arm); ``per_leg``
    widens it.  Both stay bounded by the budget check, which runs afterwards --
    a wider subset does not become affordable by being asked for.
    """
    rotation = rotation_of(descs)
    if classes:
        chosen = tuple(c for c in rotation if c in set(classes))
        if not chosen:
            chosen = tuple(sorted(set(classes)))
    elif not rotation:
        chosen = ()
    else:
        n = max(1, int(per_leg))
        start = int(leg) % len(rotation)
        chosen = tuple(rotation[(start + i) % len(rotation)]
                       for i in range(min(n, len(rotation))))
    wanted = set(chosen)
    picked = tuple(d for d in descs
                   if d.kind != tp.ZEROFILL and tensor_class(d.param_name) in wanted)
    by_rank: Dict[int, int] = {}
    for d in picked:
        by_rank[int(d.dst_rank)] = by_rank.get(int(d.dst_rank), 0) + int(d.nbytes)
    return ShadowSubset(tuple(sorted(chosen)), picked, by_rank, rotation, int(leg))


# ---------------------------------------------------------------------------
# The budget.  Priced against the card's OWN free column, refused by name.
# ---------------------------------------------------------------------------


#: The two things a budget line can be about.  ``subset`` is what this leg
#: will actually run and is the only one that grades anything; ``full`` is the
#: whole leg, priced and printed for the reader and nothing else.
SCOPE_SUBSET = "subset"
SCOPE_FULL = "full"


@dataclass(frozen=True)
class ShadowPrice:
    card: str
    dst_mib: int
    scratch_mib: int
    bounce_mib: int
    free_mib: int
    floor_mib: float
    reserve_mib: int = 0
    scope: str = SCOPE_SUBSET
    graded: bool = True

    @property
    def need_mib(self) -> int:
        return self.dst_mib + self.scratch_mib + self.bounce_mib

    @property
    def affordable(self) -> bool:
        return (self.free_mib - self.need_mib - self.reserve_mib
                >= self.floor_mib)

    def message(self) -> str:
        return (
            f"{UNAFFORDABLE_MARKER} card={self.card} scope={self.scope} "
            f"need_mib={self.need_mib} "
            f"(dst_buffers={self.dst_mib} + scratch={self.scratch_mib} + "
            f"oncard_bounce={self.bounce_mib}) "
            f"resume_reserve_mib={self.reserve_mib} "
            f"free_mib={self.free_mib} floor_mib={self.floor_mib:g} "
            f"left_mib={self.free_mib - self.need_mib - self.reserve_mib} "
            f"-- the shadow does not run on this card for this leg; the flip "
            f"is untouched.  {SB5F_FREE_MIB_NOTE}"
        )

    def line(self) -> str:
        return (
            f"{SHADOW_BUDGET_LINE_PREFIX} card={self.card} scope={self.scope} "
            f"need_mib={self.need_mib} dst_mib={self.dst_mib} "
            f"scratch_mib={self.scratch_mib} bounce_mib={self.bounce_mib} "
            f"resume_reserve_mib={self.reserve_mib} "
            f"free_mib={self.free_mib} floor_mib={self.floor_mib:g} "
            f"graded={'yes' if self.graded else 'no'} "
            f"verdict={'AFFORDABLE' if self.affordable else 'REFUSED'}"
        )


def price_shadow(card: str, need_bytes: int, free_mib: int, *,
                 floor_mib: float = SHADOW_FLOOR_MIB,
                 scratch_bytes: int = STRIPE_BYTES,
                 bounce_bytes: int = 0,
                 reserve_bytes: int = 0,
                 scope: str = SCOPE_SUBSET,
                 graded: bool = True) -> ShadowPrice:
    """Can this card carry the shadow's buffers on top of what it is doing?

    ``need`` is the destination buffers (raw ``cudaMalloc``, OUTSIDE every TMS
    region, so the saver never pauses them and they never enter its census)
    plus ONE stripe-sized scratch for the device checksum -- the spec's own
    "VRAM delta = 64 MiB per consumer rank".

    ``bounce_bytes`` IS THE THIRD TERM AND IT WAS MISSING.  MEASURED-BY-REVIEW
    DEFECT (S5 review must_fix 3, refuter must_fix 2): a SOURCE rank priced
    ``need=0`` and then allocated ``slots x slot_bytes`` of raw ``cudaMalloc``
    on its own card inside ``run_leg``'s diagonal (:class:`tp.OnCardBounce`) --
    up to 2 x 128 MiB on exactly the card the boot ticket expects to refuse.
    An instrument whose only forbidden failure is "OOM the flip it observes"
    had an unpriced VRAM term on the source leg.

    ``reserve_bytes`` IS THE FOURTH, and it is the timing hole the refuter
    named (must_fix 4): ``free_mib`` is read at HOOK time, but the
    destination's ring ``resume`` maps its image AFTER that instant while the
    shadow's buffers are still alive.  A subset that fits the free column now
    can OOM the authoritative ``cu_mem_create`` a moment later.  The caller
    that knows the not-yet-resumed demand passes it here; it defaults to 0 and
    the line PRINTS the number, so a boot where nobody passed it says so
    (``resume_reserve_mib=0``) instead of looking priced.  TODO(S6): the
    destination-side hook in ``weight_updater`` is where that number exists.

    ``free_mib`` is the card's LIVE NVML free column, read by the caller.  Not
    a census, not a projection: R6 (the allocator cache) is unbounded in
    principle and this is an instrument that must not OOM the boot it observes,
    so the number it prices against is the one the driver reports right now.
    """
    up = lambda n: int(-(-int(n) // MIB))  # noqa: E731 -- one line, three uses
    return ShadowPrice(str(card), up(need_bytes), up(scratch_bytes),
                       up(bounce_bytes), int(free_mib), float(floor_mib),
                       up(reserve_bytes), str(scope), bool(graded))


def oncard_lane_descs(descs: Sequence[object], rank: int) -> List[object]:
    """THIS CARD's diagonal inside ``descs`` -- ``run_leg``'s own filter.

    ONE DEFINITION, because the set is used four times now (the bounce is
    priced from it, the hop is priced from it, S6's slot count is derived from
    it and S6's deposit verdict is graded on it) and two copies of a filter is
    how the sum-across-cards denominator got in.
    """
    return [d for d in descs
            if d.kind != tp.ZEROFILL and _is_on_card(d)
            and int(d.src_rank) == int(rank)]


def oncard_lane_bytes(descs: Sequence[object], rank: int) -> int:
    """THIS CARD's diagonal in BYTES."""
    return sum(int(d.nbytes) for d in oncard_lane_descs(descs, rank))


def oncard_lane_batches(descs: Sequence[object], rank: int,
                        slot_bytes: int) -> int:
    """HOW MANY BATCHES the producer will actually cut -- the batcher's count.

    NOT ``ceil(bytes / slot_bytes)``, and the difference is the whole reason
    this function exists (#1273 S6).  ``ceil`` is the HOP model's denominator
    and it is a lower bound on the truth: :func:`tp.batch_descs` starts a new
    batch whenever the next piece does not fit, so a diagonal whose descriptors
    do not pack flush cuts MORE batches than the byte count implies.  A slot
    count derived from the ceil model would then be smaller than the number of
    batches -- exactly the ``slots < batches`` overwrite the deposit forbids,
    arrived at by using a number that names something else.
    """
    return len(tp.batch_descs(oncard_lane_descs(descs, rank), int(slot_bytes)))


def price_leg(card_uuid: str, descs: Sequence[object], *, rank: int,
              is_source: bool, oncard_mode: str, free_mib: int,
              oncard_slots: int = tp.ONCARD_SLOTS,
              oncard_slot_bytes: Optional[int] = None,
              floor_mib: float = SHADOW_FLOOR_MIB,
              stripe_bytes: int = STRIPE_BYTES,
              reserve_bytes: int = 0,
              scope: str = SCOPE_SUBSET,
              graded: bool = True) -> Tuple[ShadowPrice, int]:
    """Every VRAM term ONE RANK's shadow costs ON ITS OWN CARD, and the slot.

    Returns the price and the diagonal slot size, together, because they are
    the same decision: the slot sizes the bounce, and the bounce is a term of
    the price.  Computing them apart is how the source's bounce came to be
    allocated without ever being priced.

    THE DIAGONAL IS THIS CARD'S, NOT THE SUM.  ``run_leg`` selects its on-card
    descriptors with ``src_rank == rank``
    (``weight_exchange_transport.py:2500``): the lane runs ONCE PER CARD
    between the two co-located processes.  MEASURED-BY-REVIEW DEFECT (S5
    refuter, must_fix 5): the caller priced the slot from the subset's on-card
    bytes summed across ALL THREE cards -- the exact denominator error
    ``XchgPlan.oncard_bytes_by_rank`` exists to stop, re-committed one seam
    over, and it inflated the unpriced bounce of the defect above by ~3x.

    Both co-located processes compute this from the same ``src_rank == rank``
    filter over the same descriptors, so they agree by construction and the
    W52 slot_bytes cross-check in the diagonal stays a cross-check.
    """
    # THE SAME SET :func:`shadow_transport` ACTUALLY ALLOCATES FOR.
    # MEASURED-BY-REVIEW DEFECT (S5b refuter, non-blocking finding, carried
    # from S5): this filtered ``kind != ZEROFILL`` first, while the transport's
    # own ``mine_dst`` does not -- a ZEROFILL descriptor addressed to this rank
    # IS shadowed there (it must be: leaving its original ``dst_ptr`` in the
    # leg would have ``run_leg`` memset the RING's live destination, which is
    # danger (a)), so the buffer was larger than the number that was priced.
    # An instrument whose one forbidden failure is "OOM the flip it observes"
    # may under-price nothing; the price follows the allocation, not the other
    # way round.
    mine_dst = [d for d in descs if int(d.dst_rank) == int(rank)]
    oncard_bytes = oncard_lane_bytes(descs, rank)
    diag_slot = (int(tp.plan_oncard_slot_bytes(oncard_bytes).slot_bytes)
                 if oncard_slot_bytes is None else int(oncard_slot_bytes))
    # The IMPORTING side maps the exporter's allocation and allocates none of
    # its own; the ``host`` degrade's bounce is a shm file, not VRAM.  So the
    # term is the EXPORTER's, and only on the IPC arm.
    exports = (bool(is_source) and oncard_bytes > 0
               and str(oncard_mode) == tp.ONCARD_MODE_IPC)
    price = price_shadow(
        card_uuid,
        0 if is_source else shadow_layout(mine_dst)[1],
        free_mib,
        floor_mib=floor_mib,
        scratch_bytes=0 if is_source else int(stripe_bytes),
        bounce_bytes=int(oncard_slots) * diag_slot if exports else 0,
        reserve_bytes=int(reserve_bytes),
        scope=str(scope), graded=bool(graded))
    return price, diag_slot


# ---------------------------------------------------------------------------
# Stripes: the comparison, and the representability question.
# ---------------------------------------------------------------------------


@dataclass
class Stripe:
    """One 64-MiB unit of PAYLOAD, summed on both sides."""

    index: int
    tensor_class: str
    param_name: str
    nbytes: int = 0
    shadow_sum: int = 0
    ring_sum: int = 0
    first_run_dst: int = 0

    @property
    def match(self) -> bool:
        return self.shadow_sum == self.ring_sum


MATCH = "MATCH"
MISMATCH = "MISMATCH"
NOT_REPRESENTABLE = "NOT-REPRESENTABLE"
#: A compare that RAN and had nothing to compare.  Not MATCH -- see
#: :meth:`ShadowCounters.verdict`.
NO_STRIPES = "NO-STRIPES"


def checksum_representable(value: int, nbytes: int) -> bool:
    """Spec section 3.3 rule 3, asked from ONE place in this module.

    MEASURED-BY-REVIEW DEFECT (S5 review, must_fix 1): the question was asked
    for STRIPES only.  The SLOT-checksum path -- the transport's own
    producer-vs-consumer comparison, which reaches the same W59 marker --
    logged a mismatch on a bare inequality, so a batch whose two ends framed
    the field differently was reported as a data corruption.  That is #656
    register C22 exactly, and it is why rule 3 says "before ANY mismatch is
    reported" rather than "before a stripe is".

    The import is LAZY because ``weights_arena`` imports torch and this module
    is exercised at the desk without it; the function is IMPORTED, never
    re-derived (rule 3 says so in as many words).
    """
    from sglang.srt.model_executor.weights_arena import checksum_is_representable

    return bool(checksum_is_representable(int(value), int(nbytes)))


def classify(stripe: Stripe) -> str:
    """MATCH / MISMATCH / NOT-REPRESENTABLE -- in that order of questions.

    SPEC SECTION 3.3 RULE 3, and it is not a formality.  ``uint8_checksum`` of
    ``n`` bytes lies in ``[0, 255n]`` and nowhere else, so a value outside that
    range was never a checksum of this payload -- the field was read at the
    wrong offset, or the payload was framed differently at the two ends.  #656
    register C22 reported exactly that as a data corruption and an instance was
    killed for a corruption that had not happened.

    Asked through :func:`checksum_representable`, which is also what the SLOT
    path asks -- one question, one caller, so the two cannot drift again.
    """
    for value in (stripe.shadow_sum, stripe.ring_sum):
        if not checksum_representable(value, stripe.nbytes):
            return NOT_REPRESENTABLE
    return MATCH if stripe.match else MISMATCH


def mismatch_message(stripe: Stripe, *, verdict: str, leg: int,
                     epoch: str) -> str:
    """W59, naming the class, the parameter, the offset and both sums."""
    return (
        f"{MISMATCH_MARKER} leg={leg} epoch={epoch} verdict={verdict} "
        f"class={stripe.tensor_class} param={stripe.param_name} "
        f"stripe={stripe.index} dst_off={stripe.first_run_dst:#x} "
        f"nbytes={stripe.nbytes} shadow_checksum={stripe.shadow_sum} "
        f"ring_checksum={stripe.ring_sum} "
        + ("-- one of the two values is outside [0, 255*nbytes], so it was "
           "never a checksum of this payload: the bytes were framed "
           "differently at the two ends, and the DATA is not what is wrong "
           "(#656 C22)"
           if verdict == NOT_REPRESENTABLE else
           "-- the shadow pulled different bytes than the ring restored at "
           "the same destination offsets; the RING's bytes are authoritative "
           "and were served, this is a finding about the EXCHANGE")
    )


def compare_class_line(*, tensor_class: str, stripes: int, nbytes: int,
                       match: int, mismatch: int, leg: int, epoch: str) -> str:
    """One line per shadowed CLASS, on the destination, MATCH or not.

    ``bytes=`` is the PAYLOAD the two sides actually summed for this class --
    not the plan's claim and not the descriptor span -- because that is the
    denominator of ``stripes=`` and the number a reader needs to know how much
    of the class the shadow really looked at.  A class whose stripes are all
    ``MATCH`` gets a line too: the S5b boot ticket could grade only
    ``mismatch=0``, which is satisfied just as well by a compare that looked at
    nothing.
    """
    return (
        f"{COMPARE_LINE_PREFIX} leg={leg} epoch={epoch} class={tensor_class} "
        f"stripes={stripes} bytes={nbytes} match={match} mismatch={mismatch} "
        f"verdict={MATCH if match and not mismatch else (MISMATCH if mismatch else NO_STRIPES)}"
    )


def slot_checksum_verdict(report: "tp.ChecksumReport") -> str:
    """MATCH / MISMATCH / NOT-REPRESENTABLE for one STAGED batch.

    The same three answers in the same order as :func:`classify`, for the other
    checksum in this slice: the transport's producer-published sum against the
    consumer's own sum over the staged bytes.  ``ChecksumReport`` states in its
    own docstring that it "cannot tell a corruption from a framing error (that
    is what ``checksum_is_representable`` is for)"; this is the caller that
    obeys it.
    """
    if report.match:
        return MATCH
    for value in (report.expected, report.got):
        if not checksum_representable(value, report.nbytes):
            return NOT_REPRESENTABLE
    return MISMATCH


def slot_checksum_message(report: "tp.ChecksumReport", *, verdict: str,
                          leg: int, epoch: str) -> str:
    """W59 for a staged batch, naming the lane, the slot and both sums."""
    return (
        f"{MISMATCH_MARKER} leg={leg} epoch={epoch} verdict={verdict} "
        f"lane={report.lane} pair={report.pair} slot={report.slot} "
        f"seq={report.seq} nbytes={report.nbytes} "
        f"producer_checksum={report.expected} consumer_checksum={report.got} "
        + ("-- one of the two values is outside [0, 255*nbytes], so it was "
           "never a checksum of this batch: the two ends framed the field "
           "differently and the STAGED BYTES are not what is wrong (#656 C22)"
           if verdict == NOT_REPRESENTABLE else
           "-- the STAGED bytes changed between publish and read; counted, "
           "never acted on")
    )


def report_slot_checksum(report: "tp.ChecksumReport", counters: "ShadowCounters",
                         *, leg: int, epoch: str,
                         log: Callable[[str], None]) -> str:
    """Count one staged batch's checksum.  COUNT AND LOG, never act.

    Returns the verdict so a caller can assert on it; the flip is never told.
    """
    counters.checksum_reports += 1
    verdict = slot_checksum_verdict(report)
    if verdict == MATCH:
        return verdict
    if verdict == NOT_REPRESENTABLE:
        counters.checksum_not_representable += 1
    else:
        counters.checksum_mismatch += 1
    log(slot_checksum_message(report, verdict=verdict, leg=leg, epoch=epoch))
    return verdict


def payload_runs(desc: object, shadow_ptr: int, shadow_off: int):
    """The (shadow, ring, nbytes) runs of one descriptor's PAYLOAD.

    A FLAT descriptor is one run.  A STRIDED2D descriptor is ``rows`` runs of
    ``run_bytes`` at ``dpitch`` -- and walking it row-wise is the whole reason
    the comparison is not a single span compare: between two runs of a 2-D
    destination sit bytes belonging to OTHER tensors of the same fused
    parameter, which the shadow neither wrote nor should read.  A span compare
    would report those as mismatches on a perfectly correct flip.

    The shadow buffer keeps the destination's own pitch, so a run sits at the
    same relative offset on both sides.  That is what makes the comparison a
    pairing rather than a re-derivation.
    """
    ring_base = int(desc.dst_ptr) + int(desc.dst_off)
    if desc.kind == tp.STRIDED2D:
        for row in range(int(desc.rows)):
            yield (shadow_ptr + shadow_off + row * int(desc.dpitch),
                   ring_base + row * int(desc.dpitch),
                   int(desc.run_bytes))
    else:
        yield (shadow_ptr + shadow_off, ring_base, int(desc.nbytes))


def desc_span(desc: object) -> int:
    """The bytes a descriptor's destination occupies, padding included."""
    if desc.kind == tp.STRIDED2D:
        rows = int(desc.rows)
        return (rows - 1) * int(desc.dpitch) + int(desc.run_bytes) if rows else 0
    return int(desc.nbytes)


def compare_stripes(descs: Sequence[object], layout: Dict[int, int],
                    shadow_ptr: int,
                    sum_bytes: Callable[[int, int], int],
                    *, stripe_bytes: int = STRIPE_BYTES) -> List[Stripe]:
    """Sum both sides run by run, accumulating into 64-MiB payload stripes.

    THE SUMS ARE ADDITIVE, which is what lets a stripe be built out of runs:
    ``uint8_checksum`` is an exact integer sum, and ``weights_arena`` says so
    where it explains why chunking cannot change the value.  So a stripe's
    checksum is the sum of its runs' checksums, and a 6 KiB run of a 2-D class
    contributes to it without being a stripe of its own.

    ``layout`` maps ``id(desc) -> offset in the shadow buffer``; it is built by
    :func:`shadow_layout` and is the ONLY place the two address spaces are
    paired.
    """
    stripes: List[Stripe] = []
    current: Optional[Stripe] = None
    index = 0
    for desc in descs:
        off = layout[id(desc)]
        cls = tensor_class(desc.param_name)
        for shadow_addr, ring_addr, nbytes in payload_runs(desc, shadow_ptr, off):
            remaining = int(nbytes)
            done = 0
            while remaining > 0:
                if current is None or current.nbytes >= int(stripe_bytes):
                    current = Stripe(index=index, tensor_class=cls,
                                     param_name=str(desc.param_name),
                                     first_run_dst=ring_addr + done)
                    stripes.append(current)
                    index += 1
                take = min(remaining, int(stripe_bytes) - current.nbytes)
                current.shadow_sum += int(sum_bytes(shadow_addr + done, take))
                current.ring_sum += int(sum_bytes(ring_addr + done, take))
                current.nbytes += take
                done += take
                remaining -= take
    return stripes


def shadow_layout(descs: Sequence[object]) -> Tuple[Dict[int, int], int]:
    """Where each descriptor's destination sits inside the shadow buffer.

    Packed in plan order with each descriptor's own SPAN (pitch included), so
    the 2-D classes keep their geometry and the consumer's scatter is the same
    ``cudaMemcpy2DAsync`` the authoritative path would issue.  A compacted
    shadow would test a copy the real flip never makes -- which is how a
    shadow ends up proving the wrong thing.
    """
    layout: Dict[int, int] = {}
    total = 0
    for desc in descs:
        layout[id(desc)] = total
        total += desc_span(desc)
    return layout, total


def to_shadow(descs: Sequence[object], layout: Dict[int, int],
              shadow_ptr: int) -> List[object]:
    """The same descriptors with their DESTINATION redirected into the buffer.

    Source side untouched: the shadow reads the same still-mapped VRAM the
    authoritative source reads, which is the point -- it is the exchange's own
    read path under test, not a copy of it.
    """
    return [d.replace(dst_ptr=int(shadow_ptr) + layout[id(d)], dst_off=0)
            for d in descs]


# ---------------------------------------------------------------------------
# The device summer.
# ---------------------------------------------------------------------------


def make_device_scratch(nbytes: int = STRIPE_BYTES, device: Optional[int] = None):
    """THE ONE PRODUCER of the summer's scratch: a uint8 device tensor.

    Torch-owned because :func:`~sglang.srt.model_executor.weights_arena.uint8_checksum`
    takes a tensor, and that is what makes the sum a DEVICE sum.  The shadow's
    destination BUFFERS are raw ``cudaMalloc`` instead (see
    :class:`ShadowBuffers`), deliberately: two allocators, two reasons, both
    stated.  ITS SIZE IS THE STRIPE SIZE and the summer re-reads it from the
    tensor, so the two numbers cannot be given independently and drift.
    """
    import torch

    return torch.empty(int(nbytes), dtype=torch.uint8,
                       device="cuda" if device is None else f"cuda:{int(device)}")


def device_summer(ops: tp.DeviceOps, stream: int, scratch,
                  checksum: Optional[Callable[[object], int]] = None):
    """``sum_bytes(addr, nbytes)`` over DEVICE memory, on the device.

    The range is copied D2D into ``scratch`` and summed by
    :func:`~sglang.srt.model_executor.weights_arena.uint8_checksum`, which is
    imported and never re-derived (spec 3.3 rule 3).  No host round trip of the
    payload: the only thing that crosses is the scalar, at the one sync
    ``uint8_checksum`` already does.

    **THE SCRATCH IS THE ARGUMENT, AND ITS SIZE IS READ FROM IT.**
    MEASURED-BY-REVIEW DEFECT (S5 refuter, must_fix 1): the first version took
    a raw ``scratch_ptr`` plus a byte count and bound the TENSOR afterwards
    through a ``bind_scratch`` attribute that had no caller anywhere.  Unbound,
    the tensor was ``torch.empty(0)``, ``scratch[:n]`` was empty, and
    ``uint8_checksum`` returned 0 for BOTH sides of every stripe -- a summer
    that cannot go red, on an instrument whose acceptance is ``mismatch=0``.
    Nothing coupled the memcpy target to the tensor's storage either.  Here the
    tensor IS the target (``data_ptr()``) and IS the bound
    (``numel() * element_size()``), so the dead-instrument state is not
    reachable, and a range larger than the scratch is REFUSED rather than
    silently truncated to whatever the slice returns.

    OPEN ITEM, named rather than hidden: a 2-D class contributes ~6 KiB runs,
    so this issues one small D2D per run.  It is an observer, bounded by the
    class subset, and the cost lands in ``compare_ms`` on the line -- but it is
    not free and it is not the shape S6 would use for an authoritative check.
    """
    scratch_ptr = int(scratch.data_ptr())
    scratch_bytes = int(scratch.numel()) * int(scratch.element_size())
    if scratch_bytes <= 0:
        raise ValueError(
            "the device summer was handed an EMPTY scratch -- every sum would "
            "be 0 and every stripe would match itself; the scratch is the "
            "stripe (spec 6/S5's 64 MiB per consumer rank) and it must be "
            "allocated before the summer exists")

    def sum_bytes(addr: int, nbytes: int) -> int:
        n = int(nbytes)
        if n <= 0:
            return 0
        if n > scratch_bytes:
            raise ValueError(
                f"a stripe of {n} bytes does not fit the {scratch_bytes}-byte "
                f"shadow scratch -- the scratch IS the stripe size (spec 6/S5's "
                f"64 MiB per consumer rank), so these two numbers may not drift")
        ops.memcpy_async(scratch_ptr, int(addr), n, stream)
        ops.synchronize(stream)
        if checksum is not None:
            return int(checksum(scratch[:n]))
        from sglang.srt.model_executor.weights_arena import uint8_checksum

        return int(uint8_checksum(scratch[:n]))

    return sum_bytes


# ---------------------------------------------------------------------------
# The leg.
# ---------------------------------------------------------------------------


@dataclass
class ShadowCounters:
    """What the shadow found.  Counters, never actions."""

    stripes: int = 0
    match: int = 0
    mismatch: int = 0
    not_representable: int = 0
    checksum_reports: int = 0
    checksum_mismatch: int = 0
    checksum_not_representable: int = 0
    errors: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        """MISMATCH / MATCH / NO-STRIPES -- and the third one is the point.

        MEASURED-BY-REVIEW finding (S5 refuter): a comparison that produced NO
        stripes returned ``MATCH``, so a destination whose subset selected
        nothing on this rank printed the same word as one that compared a whole
        class and agreed.  Only the boot ticket's "``mismatch=0`` WITH
        ``stripes>0``" grep stood between that and an unarmed instrument
        reading as a passed one -- a compensating reader, which is the thing
        this campaign refuses to rely on.  The code says it now.
        """
        if self.mismatch or self.not_representable:
            return MISMATCH
        if not self.stripes:
            return NO_STRIPES
        return MATCH


@dataclass
class ShadowResult:
    leg: int
    epoch: str
    subset: ShadowSubset
    counters: ShadowCounters
    oncard_ms: float = 0.0
    cross_ms: float = 0.0
    #: NO PRODUCER IN THIS SLICE, and the line says ``n/a`` rather than
    #: ``0.000``.  MEASURED-BY-REVIEW finding (S5 refuter): a zero prints as a
    #: measurement, and ``ring_ms`` (the authoritative restore's own wall) and
    #: ``lock_wait_ms`` (``_split_decisions`` contention) are both filled by
    #: the ``weight_updater`` hooks S6 owns.  ``None`` is the honest value of a
    #: field nothing has measured yet.
    ring_ms: Optional[float] = None
    compare_ms: float = 0.0
    issue_ms: float = 0.0
    slot_wait_ms: float = 0.0
    gate_skew_ms: float = 0.0
    lock_wait_ms: Optional[float] = None
    pieces: int = 0
    #: THE SHADOW'S OWN PRICED HOP, so the boot ticket can put a measurement
    #: beside a prediction.  The PLAN line's ``oncard_hop_ms_priced`` prices the
    #: FULL diagonal; a shadow leg moves a class SUBSET, so grading its
    #: ``oncard_ms`` against the plan's number compares two different lanes.
    #: ``ONCARD_PER_BATCH_MS`` carries its own measured arm (S5-pre, 32 MiB
    #: slot, 16 batches, consumer side, RTX 5090).
    oncard_slot_mib: float = 0.0
    oncard_batches: int = 0
    oncard_hop_ms_priced: Optional[float] = None
    #: S6.  THE DEPOSIT'S SHAPE AND WHERE THE COUNT CAME FROM.  ``slots`` was
    #: never on this line because it was never a choice -- it was
    #: ``tp.ONCARD_SLOTS`` on every product path.  It is one now, so it is
    #: printed with its provenance and with the host bytes it implies, and a
    #: reader can check ``oncard_slots >= oncard_batches`` (the property that
    #: removes the drain wait) without opening the source.
    oncard_slots: int = 0
    oncard_slots_source: str = "caller"
    oncard_deposit_mib: float = 0.0
    direction: str = "?"
    ran: bool = False
    reason: str = ""
    #: S5b.  Which of the two ``weight_updater`` hooks emitted this line --
    #: ``source`` (the sleep leg) or ``destination`` (the wake leg).  A boot
    #: whose lines are all one word has ONE hook wired, which is a wiring
    #: finding a reader cannot make out of ``dir=`` alone (both hooks of one
    #: flip carry different directions on different ranks).
    hook: str = "?"
    #: S5b, item 5.  THE ADDED WALL THIS HOOK COST ITS LEG, measured from the
    #: first statement of :func:`run_leg_hook` to its ``finally``, so it
    #: includes the attach, the semaphore census, the gate wait and the
    #: compare -- everything the flip paid for having an observer.  It is a
    #: MEASUREMENT and never a refusal: ``hop_bound_ms`` is what refuses, and
    #: it refuses a PREDICTION, before the wall is spent.
    shadow_ms: float = 0.0
    #: The launcher-given bound the priced hop was graded against.
    hop_bound_ms: float = 0.0
    #: S5b fix, refuter must_fix 4.  THE ONE DEADLINE THE HOOK'S TWO WAITS
    #: WERE CARVED OUT OF (:data:`SHADOW_HOOK_BUDGET_S`, in ms).  On the line
    #: beside ``shadow_ms`` so the reader compares the wall that was SPENT
    #: against the wall that was ALLOWED without opening the source; a
    #: ``shadow_ms`` above it is the compare's own work overrunning (work is
    #: measured, waits are bounded) and reads as ``budget=OVER``.
    hook_budget_ms: float = 0.0
    #: THE DERIVATION'S WALL, which the adapter paid BEFORE this hook's clock
    #: started (S5c refuter, must_fix 6).  It is on the line and it is inside
    #: the ``budget=`` verdict below: the bound the user law is read off must
    #: cover everything the observer costs the leg, not everything the observer
    #: happened to time.
    derive_ms: float = 0.0
    #: ``verify_sem_arm``'s census (24) when every count was armed, ``None``
    #: when it refused or could not run.  ``None`` prints ``n/a``: an absent
    #: check and a passed one are different findings (denominator law).
    sems_armed: Optional[int] = None
    #: S5b, item 2.  The ring's OWN not-yet-mapped image demand for this leg,
    #: in MiB, as the destination hook read it from ``_weg2_tag_bytes`` before
    #: the ``resume`` loop.  It is on the acceptance line as well as on the
    #: budget line so a reader who greps only ``WEG2-XCHG-SHADOW `` can still
    #: tell a priced leg from an unwired one -- ``0`` on the source hook, where
    #: there is no resume to reserve for, and the ``hook=`` token says which.
    resume_reserve_mib: int = 0
    #: S5c, item 5.  THE WALL THIS LEG SPENT BLOCKED ON ITS CO-LOCATED PEER,
    #: from ``OnCardStats.drain_wait_s``.  The two hooks sit at opposite ends
    #: of one flip, so the producer's terminal slot drain and the consumer's
    #: fill wait can legitimately span it; the block is BOUNDED (it is carved
    #: out of :data:`SHADOW_HOOK_BUDGET_S` through ``run_leg``'s ``budget_s``)
    #: and this is what lets a boot PRICE it rather than infer it from
    #: ``shadow_ms`` minus everything else.  ``0.0`` here is a real
    #: measurement of "never waited", not an absence: the field is filled on
    #: every leg that reached the transport, and a leg that did not reached
    #: no lane to wait in and says so with ``ran=no``.
    blocked_ms: float = 0.0
    #: S5c.  The GROUP-UNIFORM digest of the derivation facts this rank voted
    #: with (:attr:`LegPlanFacts.digest`), on the line so a boot log can be
    #: censused for the divergence W64 refuses without re-deriving anything.
    plan_digest: int = 0

    @property
    def observer_ms(self) -> float:
        """EVERYTHING THIS OBSERVER COST THE LEG, in one number.

        ``shadow_ms`` plus ``derive_ms``, and the sum is what ``budget=`` is
        graded against (S5c refuter, must_fix 6).  The derivation runs in the
        adapter before the hook's own clock exists, so a verdict over
        ``shadow_ms`` alone graded the observer against a bound that excluded
        part of the observer.
        """
        return float(self.shadow_ms) + float(self.derive_ms)
    def line(self) -> str:
        """THE acceptance line.  One line, both token sets, and here is why.

        The S5 brief names ``leg= epoch= classes= stripes= match= mismatch=
        oncard_ms= cross_ms= ring_ms= verdict=``; spec section 6/S5 names
        ``dir= tag= pieces= mismatch= xchg_ms= issue_ms= slot_wait_ms=
        gate_skew_ms= lock_wait_ms=``.  They are the same event described by
        two readers, and emitting two lines would make a grep for either one
        under-count the flips.  So it is ONE line carrying the union, and the
        two greps both work.

        ``lock_wait_ms`` is an OBSERVATION, never a budget (R2-10): with one
        producer and one consumer per card and a direction-split key there is
        no second contender by construction, so a non-zero value is a finding
        about ``_split_decisions``, not about the exchange.  It and ``ring_ms``
        print ``n/a`` until a hook fills them -- see the field comments.

        ``tag=`` carries the tags the shadowed CLASSES fall in; the rotation
        unit of this slice is the class, not the tag, and
        :attr:`ShadowSubset.tags` states why.
        """
        counters = self.counters
        ms = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
        return (
            f"{SHADOW_LINE_PREFIX} leg={self.leg} epoch={self.epoch} "
            f"dir={self.direction} ran={'yes' if self.ran else 'no'} "
            f"classes={len(self.subset.classes)} "
            f"subset={','.join(self.subset.classes) or 'none'} "
            f"rotation={len(self.subset.rotation)} "
            f"tag={','.join(self.subset.tags) or 'none'} "
            f"pieces={self.pieces} stripes={counters.stripes} "
            f"match={counters.match} mismatch={counters.mismatch} "
            f"not_representable={counters.not_representable} "
            f"slot_checksums={counters.checksum_reports} "
            f"slot_checksum_mismatch={counters.checksum_mismatch} "
            f"slot_checksum_not_representable="
            f"{counters.checksum_not_representable} "
            f"oncard_slot_mib={self.oncard_slot_mib:g} "
            f"oncard_batches={self.oncard_batches} "
            f"oncard_slots={self.oncard_slots} "
            f"oncard_slots_source={self.oncard_slots_source} "
            f"oncard_deposit_mib={self.oncard_deposit_mib:g} "
            f"oncard_hop_ms_priced={ms(self.oncard_hop_ms_priced)} "
            f"oncard_ms={self.oncard_ms:.3f} cross_ms={self.cross_ms:.3f} "
            f"ring_ms={ms(self.ring_ms)} compare_ms={self.compare_ms:.3f} "
            f"xchg_ms={self.oncard_ms + self.cross_ms:.3f} "
            f"issue_ms={self.issue_ms:.3f} slot_wait_ms={self.slot_wait_ms:.3f} "
            f"gate_skew_ms={self.gate_skew_ms:.3f} "
            f"lock_wait_ms={ms(self.lock_wait_ms)} "
            f"hook={self.hook} shadow_ms={self.shadow_ms:.3f} "
            f"derive_ms={self.derive_ms:.3f} "
            f"hook_budget_ms={self.hook_budget_ms:.3f} "
            f"budget={'OVER' if self.hook_budget_ms and self.observer_ms > self.hook_budget_ms else 'ok'} "
            f"hop_bound_ms={self.hop_bound_ms:.3f} "
            f"sems_armed={'n/a' if self.sems_armed is None else self.sems_armed} "
            f"resume_reserve_mib={self.resume_reserve_mib} "
            f"blocked_ms={self.blocked_ms:.3f} "
            f"plan_digest={self.plan_digest:#x} "
            f"verdict={counters.verdict if self.ran else 'NOT-RUN'} "
            f"reason={self.reason or 'ok'}"
        )


class ShadowBuffers:
    """The raw ``cudaMalloc`` destinations, outside every TMS region.

    RAW, not torch, and the reason is the same one S4's ``OnCardBounce``
    states: a torch allocation under ``TMS_HOOK_MODE_TORCH`` is intercepted by
    the saver, which would put the shadow's buffers inside the weights region
    -- paused with it, unmapped with it, and counted into the census the ring
    is sized from.  An observer that changes the thing it observes is not one.

    ONE ALLOCATION, and the scratch is NOT here.  MEASURED-BY-REVIEW DEFECT
    (S5 refuter, must_fix 3): this used to issue two raw ``cudaMalloc``s in one
    ``__init__``, and a failure of the second one -- which is what happens on
    the tight card, the only card where this matters -- propagated with the
    first already leaked for the life of the boot.  The summer's scratch is a
    torch tensor owned by :func:`make_device_scratch` (one object, one
    pointer, one size), so there is nothing left to pair with here.
    """

    def __init__(self, ops: tp.DeviceOps, device: int, nbytes: int):
        self.ops = ops
        self.nbytes = int(nbytes)
        self.ptr = ops.raw_malloc(int(device), self.nbytes) if self.nbytes else 0

    def close(self) -> None:
        if self.ptr:
            try:
                self.ops.raw_free(self.ptr)
            except Exception:  # noqa: BLE001 -- an observer's unwind
                pass
        self.ptr = 0


class ShadowRun:
    """One leg's shadow: gate, transport, and (on the destination) the compare.

    TWO PHASES, because the ground truth only exists in the second one.  The
    transport runs where the authoritative leg runs -- the source before its
    ``pause``, the destination around its ``resume`` -- but the RING's bytes
    are not restored until ``family_complete``, so the comparison is a separate
    call the leg makes afterwards.  Spec section 6/S5 step 3 states the reason
    this is race-free: comparing the shadow's buffer against the destination's
    OWN restored tensor needs no ordering between the two legs at all.
    """

    def __init__(self, result: ShadowResult, buffers: Optional[ShadowBuffers],
                 descs: Sequence[object], layout: Dict[int, int],
                 stripe_bytes: int = STRIPE_BYTES):
        self.result = result
        self.buffers = buffers
        self.descs = list(descs)
        self.layout = layout
        # ONE number for the stripe and the scratch, carried from the caller:
        # the scratch IS the stripe (spec 6/S5's 64 MiB per consumer rank), and
        # two independent copies of that size is a stripe that does not fit its
        # own scratch.
        self.stripe_bytes = int(stripe_bytes)

    def compare(self, sum_bytes: Optional[Callable[[int, int], int]],
                log: Callable[[str], None]) -> ShadowResult:
        """Compare the pulled bytes against the ring-restored ones.  COUNT ONLY.

        Called after ``family_complete`` on the destination.  Every mismatch is
        a W59 log line and a counter; nothing here refuses, retries, or touches
        a tag.  If the caller passed no summer there is nothing to compare and
        the line says ``verdict=NOT-RUN`` rather than ``MATCH`` -- an unarmed
        instrument may not read as a passed one (spec section 4.2).
        """
        result = self.result
        try:
            if not result.ran or self.buffers is None or sum_bytes is None:
                result.ran = False
                if not result.reason:
                    result.reason = "no-summer"
                return result
            started = time.perf_counter()
            stripes = compare_stripes(self.descs, self.layout,
                                      self.buffers.ptr, sum_bytes,
                                      stripe_bytes=self.stripe_bytes)
            result.compare_ms = (time.perf_counter() - started) * 1e3
            counters = result.counters
            counters.stripes = len(stripes)
            per_class: Dict[str, List[int]] = {}
            for stripe in stripes:
                verdict = classify(stripe)
                # PER CLASS, FOR EVERY VERDICT (S5c item 4): W59 names a class
                # only when something went wrong, so a boot whose shadow AGREED
                # carried no per-class evidence at all -- ``match=7`` and no way
                # to say which seven, over which bytes.  The class is the unit
                # this slice rotates over, so it is the unit the reader needs.
                row = per_class.setdefault(
                    stripe.tensor_class, [0, 0, 0, 0])
                row[0] += 1
                row[1] += int(stripe.nbytes)
                if verdict == MATCH:
                    counters.match += 1
                    row[2] += 1
                    continue
                row[3] += 1
                if verdict == NOT_REPRESENTABLE:
                    counters.not_representable += 1
                else:
                    counters.mismatch += 1
                log(mismatch_message(stripe, verdict=verdict, leg=result.leg,
                                     epoch=result.epoch))
            for cls in sorted(per_class):
                n, nbytes, ok, bad = per_class[cls]
                log(compare_class_line(
                    tensor_class=cls, stripes=n, nbytes=nbytes, match=ok,
                    mismatch=bad, leg=result.leg, epoch=result.epoch))
        except BaseException as exc:  # noqa: BLE001 -- an observer never raises
            result.counters.errors.append(f"{type(exc).__name__}: {exc}")
            result.reason = f"compare-failed:{type(exc).__name__}"
        finally:
            self.close()
        return result

    def close(self) -> None:
        if self.buffers is not None:
            self.buffers.close()
            self.buffers = None


def shadow_transport(
    *,
    region: xr.XchgRegion,
    sems: tp.SemSet,
    ops: tp.DeviceOps,
    row: int,
    rank: int,
    device: int,
    card_uuid: str,
    uuid_of_card: Sequence[str],
    descs: Sequence[object],
    is_source: bool,
    oncard_mode: str,
    peer_row: int,
    wave: int,
    leg: int,
    direction: str,
    epoch: str,
    free_mib: int,
    log: Callable[[str], None],
    sum_bytes: Optional[Callable[[int, int], int]] = None,
    classes: Sequence[str] = (),
    per_leg: int = 1,
    budget_s: float = SHADOW_TRANSPORT_BUDGET_S,
    #: THE LEG'S REMAINING DEADLINE, as a callable, forwarded verbatim to
    #: ``tp.run_leg`` (S5c refuter, must_fix 2).  ``budget_s`` bounds ONE wait
    #: and the transport performs many; only a callable can bound their sum.
    budget_left: Optional[Callable[[], float]] = None,
    gate_budget_s: float = SHADOW_GATE_BUDGET_S,
    gate_rows: Optional[Sequence[int]] = None,
    #: S5c refuter, must_fix 3.  ``False`` means "the co-located peer of this
    #: lane cannot run while this hook holds its thread", which is the truth on
    #: ``weight_updater``'s placement.  The gate still runs -- the digests are
    #: compared -- and the TRANSPORT is refused by name afterwards, before a
    #: bounce is allocated or a handle awaited.  See
    #: :attr:`ShadowLegInputs.oncard_drainable`.
    oncard_drainable: bool = True,
    #: S6.  What one card's store-and-forward deposit may cost on the host,
    #: from the #1269 ledger's own charge, and whether this caller asked for
    #: the deposit at all.  ``0``/``False`` keeps every S3/S4/S5 caller
    #: byte-unchanged: with ``oncard_drainable=True`` neither is read.
    host_bounce_budget_bytes: int = 0,
    oncard_store_forward: bool = False,
    #: S5c.  The two digests this rank votes with, and the peer whose card
    #: geometry it is checked against.  Zero/``None`` keeps S3's and S4's
    #: callers byte-unchanged: a table of zeros is uniform, so the checks are
    #: inert until a derivation fills them.
    plan_digest: int = 0,
    card_digest: int = 0,
    check_peer_card: bool = False,
    floor_mib: float = SHADOW_FLOOR_MIB,
    resume_reserve_bytes: int = 0,
    explicit: bool = False,
    slot_bytes: int = xr.SLOT_BYTES,
    oncard_slot_bytes: Optional[int] = None,
    oncard_slots: int = tp.ONCARD_SLOTS,
    stripe_bytes: int = STRIPE_BYTES,
) -> ShadowRun:
    """Run the exchange for one leg, into buffers nothing reads.

    THE ORDER IS: subset -> price -> rank-uniform gate -> allocate -> transport.
    Pricing before the gate is what lets a rank vote its own card's arithmetic;
    allocating after the gate is what stops a rank that will not run from
    taking VRAM anyway.

    EVERY FAILURE PATH ENDS IN A LOG LINE AND A RETURNED RESULT, WITH ONE
    STATED EXCEPTION.  No exception leaves this function on the automatic path,
    including the ones the transport raises by design (W52, W53, W54): on the
    authoritative path those are refusals that stop a flip, and here the same
    event means "the shadow did not get its measurement".  The exception is
    ``explicit=True`` -- an operator who asked for a named class subset gets
    :class:`Weg2XchgShadowUnaffordable` raised instead of a degrade, the same
    two-arm shape as W56 in S4, and that arm has no producer on the boot path
    (there is no launcher flag for it: TODO(S6)).  ``vote_failure`` is a
    RECORDER, not the transport's group vote -- passing the real one would let
    an observer's thread take six ranks down.

    ``resume_reserve_bytes`` is the destination's not-yet-resumed image demand;
    see :func:`price_shadow`.  It defaults to 0 and the budget line prints it,
    so an unwired boot is visibly unwired rather than quietly optimistic.
    """
    subset = select_subset(descs, leg=leg, classes=classes, per_leg=per_leg)
    result = ShadowResult(leg=int(leg), epoch=str(epoch), subset=subset,
                          counters=ShadowCounters(), direction=str(direction))
    run = ShadowRun(result, None, (), {}, stripe_bytes=stripe_bytes)
    try:
        # THE FULL LEG, PRICED AND PRINTED, GRADING NOTHING.  MEASURED-BY-
        # REVIEW DEFECT (S5 review, must_fix 4): the automatic path prices only
        # the SUBSET, so no full-size budget line could ever appear on a
        # default shadow boot -- and the boot ticket's prediction ("the 5090
        # refuses at full size; no refusal there is itself a finding") was
        # graded against an absence the code guaranteed.  An absence that
        # cannot occur is not evidence.  This line is scope=full graded=no: it
        # is the ticket's number, and it decides nothing.
        full_price, _ = price_leg(
            card_uuid, descs, rank=rank, is_source=is_source,
            oncard_mode=oncard_mode, free_mib=free_mib,
            oncard_slots=oncard_slots, oncard_slot_bytes=oncard_slot_bytes,
            floor_mib=floor_mib, stripe_bytes=stripe_bytes,
            reserve_bytes=resume_reserve_bytes, scope=SCOPE_FULL, graded=False)
        log(full_price.line())
        price, diag_slot = price_leg(
            card_uuid, subset.descs, rank=rank, is_source=is_source,
            oncard_mode=oncard_mode, free_mib=free_mib,
            oncard_slots=oncard_slots, oncard_slot_bytes=oncard_slot_bytes,
            floor_mib=floor_mib, stripe_bytes=stripe_bytes,
            reserve_bytes=resume_reserve_bytes, scope=SCOPE_SUBSET)
        log(price.line())
        # The lane's own geometry, on the line, from the same two numbers that
        # sized the bounce -- a prediction the boot can be graded against.
        # BEFORE THE GATE, because it is a property of the PLAN and not of the
        # run: a leg the gate switches off still prints what it would have
        # cost, which is the number a refused boot needs most.
        diag_bytes = oncard_lane_bytes(subset.descs, rank)
        result.oncard_slot_mib = diag_slot / MIB
        result.oncard_batches = -(-diag_bytes // diag_slot) if diag_bytes else 0
        result.oncard_hop_ms_priced = (result.oncard_batches
                                       * tp.ONCARD_PER_BATCH_MS)
        # S6, on the same line and before the same gate: the deposit's shape.
        result.oncard_slots = int(oncard_slots)
        result.oncard_slots_source = ("store-forward-batches"
                                      if oncard_store_forward else "caller")
        result.oncard_deposit_mib = int(oncard_slots) * diag_slot / MIB
        if not price.affordable:
            log(price.message())
            if explicit:
                raise Weg2XchgShadowUnaffordable(price.message())
        verdict = shadow_gate(region, row, leg=leg, vote=price.affordable,
                              classes_hash=subset.hash, need_mib=price.need_mib,
                              log=log, budget_s=gate_budget_s,
                              expect_rows=gate_rows,
                              plan_digest=plan_digest, card_digest=card_digest,
                              peer_row=peer_row if check_peer_card else None)
        log(verdict.line(leg=leg, epoch=epoch))
        # THE SIX RANKS' SPREAD, not this rank's wait -- see
        # :attr:`ShadowVerdict.skew_ms`.
        result.gate_skew_ms = verdict.skew_ms
        if not verdict.run:
            result.reason = verdict.reason.split(":")[0].replace(" ", "-")
            return run
        store_forward = False
        if not oncard_drainable and any(_is_on_card(d) for d in subset.descs):
            # THE LANE HAS NO CONCURRENT PEER ON THIS PLACEMENT, and the gate
            # has just done the only work that does not need one: the six rows
            # agreed on ``plan_digest`` and the co-located pair on
            # ``card_digest``.  Refusing HERE rather than earlier is the whole
            # value -- an earlier return would skip the rendezvous and the
            # digests would never be compared at all, which is refuter finding
            # 9 with the sign flipped.
            #
            # S6 PUTS A LANE BACK IN FRONT OF THAT REFUSAL, and only where it
            # genuinely removes the wait.  A store-and-forward deposit needs no
            # concurrent peer at all: with one slot per batch the source fills,
            # publishes and RETURNS, and the destination reads the same shm
            # file in its own later leg.  So the refusal below now has two
            # arms -- W65 when the deposit itself does not fit or is not funded
            # (a claim that something is misconfigured), and the older
            # blameless line when no deposit is possible on this arm at all.
            # THE BATCHER'S OWN COUNT, not the hop model's ceil -- see
            # :func:`oncard_lane_batches`.  The verdict is about whether every
            # batch gets its own slot, so it must be graded on the number of
            # batches that will actually be cut.
            deposit_batches = oncard_lane_batches(subset.descs, rank, diag_slot)
            deposit_reason = tp.deposit_refusal_reason(
                batches=deposit_batches, slots=int(oncard_slots),
                slot_bytes=int(diag_slot),
                budget_bytes=int(host_bounce_budget_bytes),
                mode=str(oncard_mode))
            if not oncard_store_forward or deposit_reason == tp.DEPOSIT_REASON_IPC:
                log(oncard_not_drainable_message(
                    rank=rank, row=row, peer_row=peer_row, leg=leg, epoch=epoch,
                    is_source=is_source,
                    descs=sum(1 for d in subset.descs if _is_on_card(d))))
                result.reason = "oncard-not-drainable"
                return run
            if deposit_reason:
                log(deposit_refusal_message(
                    reason=deposit_reason, rank=rank, row=row,
                    peer_row=peer_row, leg=leg, epoch=epoch,
                    is_source=is_source, batches=deposit_batches,
                    slots=int(oncard_slots), slot_bytes=int(diag_slot),
                    budget_bytes=int(host_bounce_budget_bytes)))
                result.reason = "deposit-unfundable"
                return run
            store_forward = True
        mine_dst = [d for d in subset.descs if int(d.dst_rank) == int(rank)]
        layout, total = shadow_layout(mine_dst)
        buffers = None
        leg_descs: Sequence[object] = subset.descs
        if not is_source:
            buffers = ShadowBuffers(ops, device, total)
            # THE RUN OWNS THE BUFFER FROM THE INSTANT IT EXISTS.  MEASURED-BY-
            # REVIEW DEFECT (S5 refuter, must_fix 3): the run was built AFTER
            # the descriptor rewrite below, so an exception in between reached
            # the handler, which called ``close()`` on the STALE run
            # (``buffers=None``) and leaked the whole destination buffer for
            # the life of the boot -- on the card where allocations raise,
            # which is the only card where any of this matters.
            run = ShadowRun(result, buffers, (), layout,
                            stripe_bytes=stripe_bytes)
            shadowed = to_shadow(mine_dst, layout, buffers.ptr)
            keep = {id(d) for d in mine_dst}
            leg_descs = [d for d in subset.descs if id(d) not in keep] + shadowed
            # THE COMPARE KEEPS THE ORIGINAL DESCRIPTORS, and this is the one
            # place the two address spaces must not be confused.  MEASURED
            # DEFECT, caught by this slice's own can-fail control: handing the
            # SHADOWED descriptors to the comparison made both of its sides the
            # shadow buffer, so every stripe matched itself and the instrument
            # could not go red.  A comparison whose two sides are the same
            # pointer is the "gate that cannot fail" this campaign keeps
            # naming, and it was one line.
            run.descs = list(mine_dst)
        counters = result.counters

        def on_checksum(report: tp.ChecksumReport) -> None:
            # REPRESENTABILITY FIRST, on this path too (S5 review, must_fix 1).
            report_slot_checksum(report, counters, leg=leg, epoch=epoch, log=log)

        started = time.perf_counter()
        out = tp.run_leg(
            region, sems, ops, row=row, rank=rank, device=device,
            card_uuid=card_uuid, uuid_of_card=uuid_of_card, descs=leg_descs,
            is_source=is_source, oncard_mode=oncard_mode, peer_row=peer_row,
            wave=wave, log=log,
            # A RECORDER, not the group vote.  See the docstring.
            vote_failure=lambda exc: counters.errors.append(
                f"leg-vote {type(exc).__name__}: {exc}"),
            budget_s=budget_s, budget_left=budget_left,
            checksum_bytes=sum_bytes,
            on_checksum=on_checksum, slot_bytes=slot_bytes,
            # PRICED FROM THIS CARD'S OWN DIAGONAL WITHIN THE SUBSET, by the
            # same :func:`price_leg` call that sized the bounce from it: the
            # shadow moves a class subset, so its batch count -- and therefore
            # the hop this leg pays -- is a different number from the full
            # leg's, and the lane runs once per CARD, so it is a different
            # number from the subset's sum across the three cards too.
            oncard_slot_bytes=diag_slot,
            oncard_slots=oncard_slots,
            # S6.  ``True`` ONLY after the deposit verdict above said this
            # shape fits and is funded -- so a source leg that reaches here
            # fills its slots and returns without a terminal drain, and the
            # destination's own later leg reads them.
            oncard_store_forward=store_forward)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        result.cross_ms = sum(p.elapsed_s for p in out.pairs) * 1e3
        result.oncard_ms = (out.oncard.elapsed_s * 1e3) if out.oncard else 0.0
        result.slot_wait_ms = sum(p.slot_wait_s for p in out.pairs) * 1e3
        result.blocked_ms = (out.oncard.drain_wait_s * 1e3) if out.oncard else 0.0
        # THE REMAINDER NAMES WHAT IT SUBTRACTED.  ``issue_ms`` is
        # ``elapsed - cross - oncard``, and the producer's TERMINAL drain sits
        # outside ``oncard.elapsed_s`` by design -- so without this term a wait
        # that can span a whole flip would be reported as issue overhead and
        # graded against spec 6/S5's "issue_ms <= 5 % of xchg_ms".
        outside_ms = ((out.oncard.drain_wait_outside_s * 1e3)
                      if out.oncard else 0.0)
        result.issue_ms = max(0.0, elapsed_ms - result.cross_ms
                              - result.oncard_ms - outside_ms)
        result.pieces = sum(p.pieces for p in out.pairs) + len(
            [d for d in leg_descs if _is_on_card(d)])
        result.ran = True
    except Weg2XchgShadowUnaffordable:
        raise
    except BaseException as exc:  # noqa: BLE001 -- an observer never raises
        result.counters.errors.append(f"{type(exc).__name__}: {exc}")
        result.reason = f"transport-failed:{type(exc).__name__}"
        result.ran = False
        # THE BLOCK IS PRICED ON THE PATH WHERE IT MATTERS MOST.  A leg that
        # died in its drain is a leg that waited out its whole budget, and
        # reporting ``blocked_ms=0.000`` there would say the opposite of what
        # happened -- the unwired-instrument-reads-as-a-passed-one shape, in
        # the one field this round exists to add.  ``run_leg`` carries its
        # partial stats on the exception for exactly this.
        partial = getattr(exc, "weg2_leg_result", None)
        oncard = getattr(partial, "oncard", None) if partial is not None else None
        if oncard is not None:
            result.blocked_ms = oncard.drain_wait_s * 1e3
        run.close()
    return run


def _is_on_card(desc: object) -> bool:
    return int(desc.src_rank) == int(desc.dst_rank)


# ===========================================================================
# S5b -- THE TWO LEG HOOKS.  The shadow's only callers in the product.
#
# SECTION 1ai-S5-fix's UNPROVEN 2 named the gap in these words: "the two hooks
# inside ``weight_updater``'s legs are NOT written ... until then the shadow
# has no caller in the product at all."  This section is that caller, and it
# is deliberately a pair of FREE FUNCTIONS over plain arguments rather than
# methods on the scheduler mixin: the mixin cannot be constructed without a
# model runner, a torch process group and a device, so a hook written into it
# is a hook no hermetic test can drive.  ``weight_updater`` keeps two thin
# adapters that gather the arguments and call these; the adapters are pinned
# by SOURCE (the WiringTest precedent, test_weg2_xchg_cover_1273.py:862) and
# the behaviour is proven here.
# ===========================================================================

#: Which side of the flip a hook is on.  The SOURCE hook runs on the SLEEP leg
#: (the group giving its weights up) and the DESTINATION hook on the WAKE leg.
HOOK_SOURCE = "source"
HOOK_DESTINATION = "destination"

#: THE FACTOR, NAMED.  :data:`tp.ONCARD_HOP_BUDGET_MS` (20 ms) is the spec's
#: diagonal target for the AUTHORITATIVE lane, and the shadow is not that lane:
#: it moves a class subset on a card that is simultaneously carrying a real
#: flip leg, so a bound equal to the authoritative target would refuse the
#: observer for being an observer.  Three is chosen as the smallest integer
#: multiple that still refuses the shape this bound exists to refuse -- a
#: subset whose priced hop has grown to the FULL diagonal's order (the 10.28
#: GiB / 109-batch figure prices at 19.8 ms, so a full-diagonal subset lands at
#: ~1x the budget and passes; the 32 MiB shipping slot prices the same bytes at
#: 329 batches ~ 118 ms and is refused at 60 ms).  It is a factor and not a
#: second measured constant precisely so the reader can see it is a POLICY
#: number sitting on top of a measured one, not an arm of its own.
SHADOW_HOP_BOUND_FACTOR = 3.0
SHADOW_HOP_BOUND_MS_DEFAULT = tp.ONCARD_HOP_BUDGET_MS * SHADOW_HOP_BOUND_FACTOR

#: The launcher-given bound.  ``--weg2-shadow-hop-bound-ms`` publishes it; the
#: env read is the same shape as :data:`tp.ENV_ONCARD_MODE`, so a rank needs no
#: new plumbing to see a flag the launcher set.
ENV_HOP_BOUND_MS = "SGLANG_WEG2_SHADOW_HOP_BOUND_MS"

#: W63 -- this RANK could not join the shadow while the mode is armed.
#:
#: It exists because item 6 of the S5b brief is a real hazard and not a style
#: rule: the shadow gate is rank-uniform, so a rank that quietly returns
#: without voting is not "one rank less" -- it is five ranks waiting out the
#: gate budget inside their own flip legs and then reading an EXPIRY, which
#: looks identical to a card that refused on arithmetic.  A skip that cannot be
#: told from a refusal is a silent divergence, and this names it.
RANK_LOCAL_SKIP_MARKER = "W63 Weg2XchgShadowRankLocalSkip"


class Weg2XchgShadowRankLocalSkip(RuntimeError):
    """W63 -- the shadow is armed for this boot but this rank cannot join it.

    Raised ONLY with ``explicit=True`` (an operator who asked for the shadow by
    name on this rank), the same two-arm shape as :class:`
    Weg2XchgShadowUnaffordable` and W56 before it.  On the automatic path it is
    a log line and a ``reason=`` token on the leg's own
    ``WEG2-XCHG-SHADOW`` line, because a zero-authority observer that raises
    into a flip leg has taken the authority this slice exists not to have.

    The reasons it names, all of which are "this rank, locally":

    * ``no-region`` -- ``SGLANG_WEG2_XCHG_REGION`` is unset or the file will not
      open (the launcher's ``prepare_shadow_env`` did not run, or ran for
      another boot);
    * ``no-sems`` -- the 24 names are not openable without ``O_CREAT``;
    * ``no-ops`` -- ``libcudart`` did not load;
    * ``no-plan`` -- nothing handed this leg descriptors.  **S5c ENDED THE
      STANDING STATE**: :func:`derive_leg_plan` is ``build_plan``'s product
      caller and the two ``weight_updater`` adapters call it, so a shadow boot
      no longer prints this on every leg of every rank by construction.  It is
      now a REFUSAL WITH A CAUSE, and the cause is on the line
      (``ShadowLegInputs.plan_reason``): ``no-ring-layout`` for a form with no
      chunked weights family (a P-only boot, a stock boot),
      ``stale-wave-map``, ``tag-not-in-family``, ``no-chunk-classes``,
      ``plan-refused:<W-code>``.  What has NOT changed is the prohibition the
      old wording existed for: the plan is derived from the RING's own layout
      producers, never from a rank's private guess, because a plan a rank
      invents on its own compares the ring's restored bytes against a copy of
      those same bytes and matches by construction -- the "instrument that
      cannot go red" this campaign has now paid for three times
      (``20c5fb9048``, refuter 1, refuter 2).
    * ``stale-run`` -- a previous leg's shadow was never closed (see
      :class:`ShadowLeg`).
    """


def hop_bound_ms(value: Optional[float] = None) -> float:
    """The bound this boot grades the priced hop against, with its provenance.

    ``value`` (the launcher flag, passed down) wins; then the environment;
    then :data:`SHADOW_HOP_BOUND_MS_DEFAULT`.  A malformed environment value is
    NOT silently replaced by the default -- it raises, here, at the top of the
    leg, where a ``ValueError`` is caught by the hook and printed as a reason.
    A bound nobody can read is a bound nobody is graded against.
    """
    if value is not None:
        return float(value)
    raw = (os.environ.get(ENV_HOP_BOUND_MS, "") or "").strip()
    if not raw:
        return float(SHADOW_HOP_BOUND_MS_DEFAULT)
    return float(raw)


def hop_refusal_message(*, card: str, priced_ms: float, bound_ms: float,
                        batches: int, slot_mib: float, leg: int) -> str:
    """W61 for the TIME term, the same marker the VRAM term already uses.

    ONE code for one class of event -- "the shadow cannot afford to run on this
    leg" -- and the line says which resource by naming both numbers.  A second
    W-code for the same decision would make a census of "legs the shadow
    refused itself" read low by exactly the time-refused ones.
    """
    return (
        f"{UNAFFORDABLE_MARKER} card={card} scope=hop leg={leg} "
        f"priced_hop_ms={priced_ms:.3f} bound_ms={bound_ms:.3f} "
        f"batches={batches} slot_mib={slot_mib:g} "
        f"factor={SHADOW_HOP_BOUND_FACTOR:g}x{tp.ONCARD_HOP_BUDGET_MS:g}ms "
        f"-- the shadow does not run on this leg; the flip is untouched and "
        f"the ring remains the only authority for weight bytes"
    )


def rank_local_skip_message(*, reason: str, rank: int, leg: int, epoch: str,
                            detail: str = "") -> str:
    """W63, naming the rank, the leg and what was missing."""
    return (
        f"{RANK_LOCAL_SKIP_MARKER} rank={rank} leg={leg} epoch={epoch} "
        f"reason={reason}{(' detail=' + detail) if detail else ''} "
        f"-- this rank does not join the shadow for this leg.  The rank-uniform "
        f"gate turns that into a NO for all {xr.N_RANKS} rows (a non-publisher "
        f"is a gate expiry, which is a NO), so no rank runs a half experiment; "
        f"the flip proceeds on the ring either way"
    )


# ---------------------------------------------------------------------------
# S5c -- THE DERIVATION.  build_plan gets its product caller, and every fact
# it is built from has a named producer in the ring's own code.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegPlanFacts:
    """The GROUP-UNIFORM inputs one leg's plan is derived from.

    "Group-uniform" is the load-bearing word and it is a property of WHERE
    each field comes from, not a hope about the six ranks agreeing:

    * ``chunk_layers`` / ``chunk_count`` -- ``weg2_memory_saver.
      weight_chunk_geometry()``, read from the two environment variables the
      LAUNCHER writes into both groups' environments in one call
      (``launcher.build_env``), so the six ranks read one value;
    * ``family_tags`` -- ``weg2_memory_saver.weights_family_tags(count)``, a
      pure function of ``chunk_count``, in the order the front pauses them;
    * ``waves`` -- ``weight_exchange.derive_waves`` over those tags;
    * ``cards`` -- ``weight_exchange_region.N_CARDS``, the module that states
      "rank *n* of either group runs on ``cards[n]``";
    * ``classes`` -- the tensor classes of the CHUNK tags, THE ONE FIELD READ
      FROM LIVE TENSORS, and therefore the one field deliberately NOT in
      :attr:`digest`.

    MEASURED-BY-REVIEW DEFECT (S5c refuter, must_fix 4).  This docstring said
    "nothing on this list is read from a card, a rank, a pointer or a live
    tensor" while ``classes`` was built from this rank's own
    ``named_parameters()`` AND filtered by ``ParamGeom.of`` succeeding on this
    rank's live tensors -- so it was group-uniform only if one ASSERTS that
    every stage's layer band carries the same module classes.  Under P=PP a
    rank's inventory is its stage's band alone, and on a hybrid layer stack
    (GDN + full attention) a band that lacks one layer type yields a different
    class set.  The code did not check that assertion, it HASHED it, so the
    divergence would have surfaced as W64 ``scope=group`` under a cause
    sentence naming a stale chunk geometry -- sending the reader to the wrong
    place (Instrument-Text-luegt, class A).

    So the class agreement is checked where it already WAS checked, with the
    right cause sentence: ``classes_hash`` in the same gate row ("the ranks
    chose different class subsets"), and the CARD digest, which is a per-card
    reading by construction.  Removing it from here removes a second
    bookkeeping of one fact, not a check.

    What REMAINS on this list is read from the launcher's environment, from
    pure functions of it, and from a module constant -- never from a card, a
    rank, a pointer or a live tensor, which is what makes :attr:`digest`
    comparable across all six rows of the shadow gate.
    """

    chunk_layers: int
    chunk_count: int
    family_tags: Tuple[str, ...]
    waves: Tuple[Tuple[str, ...], ...]
    cards: Tuple[int, ...]
    classes: Tuple[str, ...]
    source: str

    @property
    def digest(self) -> int:
        """A stable 64-bit digest, through the SAME hash the gate rows use.

        ``xr.epoch_hash`` and never ``hash()``: the built-in is salted per
        process (PYTHONHASHSEED), so six ranks would publish six numbers for
        one plan and the gate would refuse every leg -- the identical defect
        :func:`classes_hash` names one section up.
        """
        # ``classes`` IS DELIBERATELY ABSENT -- see the class docstring.  It is
        # the one field of this dataclass that is read from live tensors, and a
        # per-rank fact inside a group digest is a gate that refuses for the
        # wrong reason and names the wrong cause.
        return xr.epoch_hash("|".join((
            f"L={self.chunk_layers}", f"N={self.chunk_count}",
            "family=" + ",".join(self.family_tags),
            "waves=" + ";".join("+".join(w) for w in self.waves),
            "cards=" + ",".join(str(c) for c in self.cards),
            "source=" + self.source,
        )))


@dataclass(frozen=True)
class LegPlan:
    """One leg's derived plan, its provenance, and the two digests.

    ``descs`` is what :func:`run_leg_hook` hands the transport; ``classes`` is
    the rotation the subset is chosen FROM (never the subset itself -- the
    subset is a function of the leg index and is chosen by
    :func:`select_subset`, which stays the one owner of that arithmetic).
    """

    facts: LegPlanFacts
    descs: Tuple[object, ...]
    card: int
    tags: Tuple[str, ...]
    card_digest: int
    undescribed: int = 0
    #: THE POPULATION THIS PLAN WAS DERIVED OVER, from
    #: ``weight_exchange.walk_live_tensors`` -- the ring's own named producer
    #: for the question "what live tensors sit under an exchanged tag".
    population: int = 0
    #: Of that population, the ones this plan TRANSPORTS (parameters).
    planned: int = 0
    #: Of that population, the ones it does NOT: family-tagged buffers and
    #: plain module attributes.  MEASURED-BY-REVIEW DEFECT (S5c refuter,
    #: must_fix 5): the derivation walked ``named_parameters()`` alone while
    #: ``undescribed=`` counted only ``ParamGeom.of`` refusals, so a boot could
    #: print ``undescribed=0`` with an entire population never enumerated --
    #: and the largest member of it is the rope ``cos_sin_cache``, measured at
    #: +300/+181/+210 MiB on D's ``weights_0`` by ``walk_live_tensors``' own
    #: docstring.  Counted and printed, never silently outside the denominator.
    unplanned: int = 0
    unplanned_bytes: int = 0
    #: THE DERIVATION'S OWN WALL (S5c refuter, must_fix 6).  It runs on the
    #: flip's critical path before ``run_leg_hook`` starts its clock, so
    #: without this field it was outside ``shadow_ms``, outside
    #: ``hook_budget_ms`` and outside every ``budget=ok|OVER`` reading -- while
    #: ``SHADOW_HOOK_BUDGET_S`` claimed to be "the ONE wall a flip leg pays for
    #: having an observer".  The hook now subtracts it from its own deadline.
    derive_ms: float = 0.0

    @property
    def classes(self) -> Tuple[str, ...]:
        return self.facts.classes

    @property
    def slots(self) -> int:
        """The DESCRIPTOR count -- what the on-card lane batches into slots.

        Named ``slots`` on the line because that is the word the brief's
        acceptance grep uses; it is not the bounce's ``ONCARD_SLOTS`` depth,
        which is a constant of the transport and is printed by the budget line
        as ``oncard_slot_mib``/``oncard_batches``.
        """
        return len(self.descs)

    @property
    def nbytes(self) -> int:
        return sum(int(d.nbytes) for d in self.descs)

    def line(self) -> str:
        """THE PROVENANCE LINE, once per leg per rank.

        ``source=`` names the PRODUCERS, not this module: the whole point of
        S5c is that no number here is a hand list, so the line has to say
        which functions were asked.  A reader who doubts a field greps the
        name and lands on the code that answered.
        """
        return (
            f"{PLAN_LINE_PREFIX} card={self.card} "
            f"tags={','.join(self.tags) or 'none'} "
            f"classes={','.join(self.classes) or 'none'} "
            f"slots={self.slots} bytes={self.nbytes} "
            # THE WAVE MAP IS AN ASSUMPTION AND SAYS SO (S5c refuter, finding
            # 8).  ``derive_waves`` was handed ``{}`` -- "uniform" -- because a
            # rank holds only its own PP stage's layer count, so this number is
            # NOT the per-card order the front reads from
            # ``WEG2-FLIP-ORDER MAP``, which for group P names several.  A
            # reader comparing the two lines in one boot log must be able to
            # see that from the line and not from this source file.
            f"wave_map=uniform-assumed waves={len(self.facts.waves)} "
            f"population={self.population} planned={self.planned} "
            f"unplanned={self.unplanned} "
            f"unplanned_bytes={self.unplanned_bytes} "
            f"undescribed={self.undescribed} "
            f"derive_ms={self.derive_ms:.3f} "
            f"plan_digest={self.facts.digest:#x} "
            f"card_digest={self.card_digest:#x} "
            f"source={self.facts.source}"
        )


#: The producers, named once, in the order :func:`derive_leg_plan` asks them.
#: This string goes on the acceptance line as ``source=`` and into the group
#: digest, so a rank that derived its plan from a DIFFERENT set of producers
#: (a future second derivation, a monkeypatched test double left armed in a
#: product process) diverges at the gate instead of silently shadowing
#: something else.
PLAN_SOURCE = (
    "weg2_memory_saver.weight_chunk_geometry+weights_family_tags"
    "+tag_of_parameter_name,weight_exchange.walk_live_tensors"
    "+derive_waves+build_plan,weight_exchange_region.N_CARDS"
)


def _plan_refusal(reason: str, detail: str = "") -> Tuple[None, str]:
    return None, (f"{reason}:{detail}" if detail else reason)


def derive_leg_plan(
    *,
    hook: str,
    group: str,
    peer: str,
    rank: int,
    model,
    region_tag: str = "",
    chunk_geometry: Optional[Callable[[], Tuple[int, int]]] = None,
    family_tags: Optional[Callable[[int], Sequence[str]]] = None,
    tag_of: Optional[Callable[..., str]] = None,
    waves_of: Optional[Callable[..., Sequence[Sequence[str]]]] = None,
    n_cards: Optional[int] = None,
) -> Tuple[Optional[LegPlan], str]:
    """ONE derivation, from the ring's own layout facts.  ``(plan, reason)``.

    THE PRODUCT CALLER OF ``build_plan`` (SECTION 1ai-S5b-fix's UNPROVEN 2, in
    its own words: *"``build_plan`` has no product caller ... still the largest
    gap"*).  Every input is READ FROM A PRODUCER and none is a literal:

    ==================  =======================================================
    what                who answers
    ==================  =======================================================
    layer chunking      ``weg2_memory_saver.weight_chunk_geometry()``
    the family tags     ``weg2_memory_saver.weights_family_tags(count)``
    a parameter's tag   ``weight_exchange.tag_of_parameter_name(name, ...)``
    the flip order      ``weight_exchange.derive_waves(tags, {}, cards)``
    the card vector     ``weight_exchange_region.N_CARDS``
    the geometry        ``ParamGeom.of(<the live tensor>)`` -- stride/itemsize
    the pointers        ``tensor.data_ptr()``, THIS rank's side only
    ==================  =======================================================

    **THE PLAN IS THIS CARD'S DIAGONAL**, and that is a derivation and not a
    narrowing: ``weight_exchange_region`` states that rank *n* of either group
    runs on ``cards[n]``, so the pair ``src_rank == dst_rank == rank`` is the
    ON-CARD lane S0 probed on silicon and S4 built -- the one lane that exists
    between two processes that share a card.  Both groups are given the PP form
    (``tp_size=1``, whole tensors, ``stage`` naming the holder) with DISJOINT
    ``base`` numbers, which is what makes ``_check_bases``' 6x6 matrix real and
    ``_check_cards``' on-card theorem true.

    **WHY THE BASE TAG'S CLASSES ARE NOT IN THE ROTATION.**
    ``chunk_tag_cards``' own docstring says it: the base tag "is NOT in the map
    -- its bytes (embeddings on the first stage, head on the last, buffers
    everywhere) are not a layer band".  A rotation built over those classes is
    a rotation whose CONTENT depends on which card asks, so the six ranks would
    enumerate different lists, pick different classes for the same leg index
    and refuse every leg at the ``classes_hash`` check -- a gate that can never
    open.  The chunk tags ARE layer bands and every card's bands carry the same
    module classes, so the rotation over them is card-uniform by construction.
    The base tag's parameters stay in the inventory (they are still planned and
    still counted) -- they are simply never the rotation's unit.

    **WHAT THIS IS NOT.**  It is not the full cross-group exchange plan.  That
    one needs BOTH groups' shard vectors and the unsharded extent of every
    parameter, which is cross-group knowledge no single rank holds, and it is
    S6/S8's.  What the shadow can honestly claim from this card alone is: *the
    bytes rank n of one group holds for class C, pushed through the exchange's
    own on-card lane, are the bytes rank n of the other group holds for class C
    after the ring restored them.*  On a form where the two groups do NOT hold
    the same bytes on a card -- this rig's P=PP / D=TP shipping form -- that
    claim is false, and it is refused BY NAME as W64 ``scope=oncard-peer`` at
    the gate, on the ``card_digest``, BEFORE a byte moves.  A named refusal is
    the honest answer there; a red compare would read as a defect in the
    exchange when it is a statement about two layouts, and a fabricated match
    would be the instrument-that-cannot-go-red this campaign has paid for three
    times.

    ``(None, reason)`` on every refusal, and each reason is its own word so a
    boot log can be censused by cause:

    ``no-model``, ``population-refused`` (``walk_live_tensors`` could not
    enumerate this model), ``no-ring-layout`` (this boot has no chunked weights family,
    which is every form without a ring layout -- a P-only boot, a stock boot),
    ``stale-wave-map`` (the wave partition is not a permutation of THIS boot's
    family: the flip order map a rank derived does not belong to the tags it
    carries), ``tag-not-in-family`` (a live tensor carries a weights-family tag
    this boot's family does not contain), ``no-carried-tags``,
    ``no-chunk-classes``, ``plan-refused:<W-code exception>`` (``build_plan``'s
    own refusals, forwarded verbatim: an empty plan, a barren wave tag, a
    geometry the shard arithmetic cannot name).
    """
    from sglang.srt.managers import weg2_memory_saver as ms
    from sglang.srt.weg2 import weight_exchange as wx

    _t0 = time.perf_counter()
    chunk_geometry = chunk_geometry or ms.weight_chunk_geometry
    family_tags = family_tags or ms.weights_family_tags
    tag_of = tag_of or wx.tag_of_parameter_name
    waves_of = waves_of or wx.derive_waves
    cards = tuple(range(int(xr.N_CARDS if n_cards is None else n_cards)))
    region_tag = region_tag or wx.GPU_MEMORY_TYPE_WEIGHTS

    if model is None:
        return _plan_refusal("no-model")
    if not 0 <= int(rank) < len(cards):
        return _plan_refusal("wrong-card", f"rank={rank} cards={list(cards)}")

    chunk_layers, chunk_count = chunk_geometry()
    if int(chunk_layers) <= 0 or int(chunk_count) <= 0:
        return _plan_refusal(
            "no-ring-layout",
            f"weight_chunk_geometry()=({chunk_layers}, {chunk_count})")
    family = tuple(str(t) for t in family_tags(int(chunk_count)))

    # THE FLIP ORDER MAP.  ``{}`` for the tag->card map is READ, not assumed:
    # ``chunk_tag_cards``' own docstring says an empty map means UNIFORM, and a
    # rank cannot produce a non-empty one -- the map needs the layer count of
    # EVERY PP stage and a rank holds only its own.  Uniform yields the single
    # wave that is this boot's shipping shape (``derive_waves`` refuses the
    # nine-wave arm on measured transport grounds in its own docstring), and,
    # unlike a per-card map, it is the same on all six ranks, which is what the
    # group digest requires.  STATED DEVIATION, carried into UNPROVEN.
    waves = tuple(tuple(str(t) for t in w) for w in waves_of(family, {}, cards))
    flat = [t for w in waves for t in w]
    if sorted(flat) != sorted(family):
        return _plan_refusal(
            "stale-wave-map",
            f"waves={waves} family={family}")

    # THE POPULATION COMES FROM THE RING'S OWN PRODUCER (S5c refuter,
    # must_fix 5).  ``walk_live_tensors`` is the documented three-population
    # walk -- parameters, buffers, plain module attributes -- and it names, in
    # its own docstring, exactly the bytes a ``named_parameters()``-only
    # derivation drops: the rope ``cos_sin_cache`` measured at +300/+181/+210
    # MiB on D's ``weights_0``.  Those bytes carry CHUNK tags, which is to say
    # the tags this plan claims to cover, and the ring restores them.  The plan
    # still TRANSPORTS parameters only -- ``ParamGeom`` needs the live tensor
    # and this walk hands back records, not tensors -- but the omission is now
    # ENUMERATED BY THE PRODUCER and counted on the line rather than being
    # invisible behind ``undescribed=0``.  UPSTREAM-MINIMAL: the population
    # question has one owner and this is not a second one.
    try:
        live = [t for t in wx.walk_live_tensors(model, region_tag=region_tag)
                if ms.is_weights_family_tag(str(t.tag))]
    except BaseException as exc:  # noqa: BLE001 -- a derivation never raises
        return _plan_refusal("population-refused",
                             f"{type(exc).__name__}: {exc}")
    for t in live:
        if str(t.tag) not in family:
            return _plan_refusal("tag-not-in-family",
                                 f"{t.name} tag={t.tag} family={family}")
    unplanned = [t for t in live if str(t.kind) != wx.PARAMETER]

    inventory = []
    tensor_of: Dict[str, object] = {}
    carried: set = set()
    undescribed = 0
    for name, param in model.named_parameters():
        tag = str(tag_of(str(name), region_tag=region_tag))
        if not ms.is_weights_family_tag(tag):
            continue
        if tag not in family:
            # BOTH CHECKS STAY, and they are two questions: the walk above
            # covers the BUFFER and ATTRIBUTE populations the parameter loop
            # cannot see, and this one covers an injected ``tag_of`` that
            # disagrees with the walk's own tagging.  Neither subsumes the
            # other.
            return _plan_refusal("tag-not-in-family",
                                 f"{name} tag={tag} family={family}")
        try:
            geom = wx.ParamGeom.of(param, name=str(name), tag=tag,
                                   shard_axis=wx.REPLICATED, shard_total=0,
                                   stage=int(rank))
        except BaseException:  # noqa: BLE001 -- a shape this plan cannot name
            # COUNTED AND PRINTED, never silently dropped: ``undescribed=`` is
            # on the acceptance line, so a boot whose plan quietly narrowed is
            # visible without opening the source.  The classes such a parameter
            # would have contributed are simply not in the rotation, so the
            # shadow never claims to have checked them.
            undescribed += 1
            continue
        inventory.append(geom)
        tensor_of[str(name)] = param
        carried.add(tag)

    if not inventory:
        return _plan_refusal("no-carried-tags", f"family={family}")

    # THE ROTATION, over the CHUNK tags only -- see the docstring.
    classes = tuple(sorted({
        tensor_class(g.name) for g in inventory
        if ms.is_weights_chunk_tag(g.tag)}))
    if not classes:
        return _plan_refusal("no-chunk-classes",
                             f"carried={sorted(carried)}")

    waves_here = [[t for t in w if t in carried] for w in waves]
    waves_here = [w for w in waves_here if w]
    skip_tags = tuple(t for t in family if t not in carried)

    is_source = str(hook) == HOOK_SOURCE
    src_name, dst_name = ((str(group), str(peer)) if is_source
                          else (str(peer), str(group)))
    src = wx.GroupLayout(name=src_name, cards=cards, tp_size=1, base=0)
    dst = wx.GroupLayout(name=dst_name, cards=cards, tp_size=1,
                         base=len(cards))

    def ptr_of(g: str, r: int, name: str) -> Optional[int]:
        # ``None`` ON THE SIDE THIS RANK DOES NOT OWN, which is XchgDesc's own
        # documented contract.  The source hook fills src_ptr and the
        # destination hook fills dst_ptr, and the destination's dst_ptr is what
        # the compare reads as "the ring-restored bytes".
        if g != str(group) or int(r) != int(rank):
            return None
        tensor = tensor_of.get(str(name))
        return None if tensor is None else int(tensor.data_ptr())

    def geom_of(g: str, r: int, name: str):
        if g != str(group) or int(r) != int(rank):
            return None
        return tensor_of.get(str(name))

    try:
        plan = wx.build_plan(inventory, src, dst, waves=waves_here,
                             ptr_of=ptr_of, geom_of=geom_of,
                             skip_tags=skip_tags)
    except BaseException as exc:  # noqa: BLE001 -- a derivation never raises
        return _plan_refusal("plan-refused", f"{type(exc).__name__}: {exc}")

    facts = LegPlanFacts(
        chunk_layers=int(chunk_layers), chunk_count=int(chunk_count),
        family_tags=family, waves=waves, cards=cards, classes=classes,
        source=PLAN_SOURCE)
    return LegPlan(facts=facts, descs=tuple(plan.descs), card=int(rank),
                   tags=tuple(sorted(carried)),
                   card_digest=card_geometry_digest(inventory, classes,
                                                    unplanned=unplanned),
                   undescribed=undescribed,
                   population=len(live),
                   planned=len(inventory),
                   unplanned=len(unplanned),
                   unplanned_bytes=sum(int(t.nbytes) for t in unplanned),
                   derive_ms=(time.perf_counter() - _t0) * 1e3), ""


def card_geometry_digest(inventory: Sequence[object],
                         classes: Sequence[str],
                         unplanned: Sequence[object] = ()) -> int:
    """THIS CARD's storage geometry for the rotation's classes, as one number.

    Computed from the INVENTORY and never from the descriptors, and the
    difference is a measured trap rather than taste: ``_emit`` reads the live
    pitch through ``geom_of`` for the side the rank owns and DERIVES it for the
    other, so the source hook and the destination hook of one card build
    descriptors that differ in ``spitch``/``dpitch`` even when the two
    processes hold byte-identical storage.  A digest over descriptors would
    call that a divergence and refuse every leg -- a gate that cannot open,
    which is exactly the failure mode the base-tag rotation was cut to avoid.

    What the two co-located processes must agree on is the STORAGE: same
    parameter names, same extents, same itemsize, same tag.  That is what this
    hashes, and a disagreement in it is the honest reading of "these two groups
    do not hold the same bytes on this card".
    """
    wanted = set(str(c) for c in classes)
    terms = sorted(
        f"{g.name}:{g.tag}:{g.rows_full}x{g.cols_full}x{g.itemsize}"
        for g in inventory if tensor_class(g.name) in wanted)
    # THE UNPLANNED POPULATION IS IN HERE TOO (S5c refuter, must_fix 5).  It is
    # not transported, but it IS bytes this card holds under an exchanged tag,
    # and a digest computed over the truncated set let two co-located ranks
    # with different buffer populations agree that they hold the same thing.
    # Name, tag and size come from ``walk_live_tensors``' own record; there is
    # no ``ParamGeom`` for these, which is precisely why they are unplanned.
    terms += sorted(f"{t.name}:{t.tag}:{t.nbytes}:{t.kind}" for t in unplanned)
    return xr.epoch_hash("|".join(terms))


# ---------------------------------------------------------------------------
# The plan seam.  ONE named absence, not a silent one.
# ---------------------------------------------------------------------------

#: The descriptor producer for a shadow leg.  ``None`` is the shipping value.
#:
#: STILL EMPTY IN THE PRODUCT, AND STILL NOT A STUB -- but for a different
#: reason than before S5c.  The product's producer is
#: :func:`derive_leg_plan`, called by the two ``weight_updater`` adapters,
#: which pass a whole :class:`LegPlan` (descriptors AND the digests the gate
#: needs) through ``run_leg_hook(plan=...)``.  This module-level slot remains
#: the seam a TEST installs, and a future caller that has descriptors but no
#: facts can still use it; what it may never carry is a plan a rank invented
#: from its own ``named_parameters()`` with no cross-check, which is refused
#: in W63's ``no-plan`` bullet with the reason.
_PLAN_PROVIDER: Optional[Callable[[str, int, int], Sequence[object]]] = None


def set_plan_provider(
    provider: Optional[Callable[[str, int, int], Sequence[object]]],
) -> Optional[Callable[[str, int, int], Sequence[object]]]:
    """Install the descriptor producer (S6's, or a test's).  Returns the old one."""
    global _PLAN_PROVIDER
    previous = _PLAN_PROVIDER
    _PLAN_PROVIDER = provider
    return previous


def plan_for_leg(direction: str, leg: int, rank: int) -> Sequence[object]:
    """The descriptors for this leg, or ``()`` when nothing produces them."""
    provider = _PLAN_PROVIDER
    if provider is None:
        return ()
    return tuple(provider(str(direction), int(leg), int(rank)))


# ---------------------------------------------------------------------------
# The leg's own shadow: created at leg start, closed at leg end.
# ---------------------------------------------------------------------------

#: The ONE shadow run the CALLING THREAD owns, or ``None``.  A module-level
#: slot and not a scheduler attribute, because the thing it guards against is a
#: run that OUTLIVES the object that made it: a leg that raised between the
#: allocation and the close leaves a raw ``cudaMalloc`` alive for the life of
#: the boot on the tight card, which is refuter must_fix 3 one scope up.
#:
#: THREAD-LOCAL, and the reason is the ownership rule and not convenience: the
#: leg is owned by the thread that runs it.  In the product that is the
#: scheduler's own thread, one per RANK PROCESS, so the two hooks of one flip
#: are two processes and a process-wide slot would be the same thing.  A
#: process-wide slot is strictly WRONG, though, wherever a second leg shares
#: the interpreter -- this round's own two-sided test, and any future caller
#: that runs a leg off the scheduler thread: the second leg adopted the slot
#: and CLOSED the first one out from under itself, which presented as a gate
#: expiry ("only n/6 ranks published") five seconds inside a flip.  A stale run
#: carried across FLIPS is the same thread and is still caught.
_ACTIVE = threading.local()


def _active_leg() -> Optional["ShadowLeg"]:
    return getattr(_ACTIVE, "leg", None)


def _set_active_leg(leg: Optional["ShadowLeg"]) -> None:
    _ACTIVE.leg = leg


@dataclass(frozen=True)
class ShadowLegInputs:
    """What the leg knows and the shadow needs.  Every field has a producer.

    ``resume_reserve_bytes`` and ``ring_ms`` are the two S5b brief calls
    "producers", and both are READ from the ring rather than estimated:

    * ``resume_reserve_bytes`` is the sum of ``_weg2_tag_bytes(tag)`` over the
      weights family, read in the wake leg BEFORE the ``resume`` loop -- i.e.
      the ring's own not-yet-mapped image demand, the exact number
      :func:`price_shadow`'s fourth term was added for (refuter must_fix 4).
    * ``ring_ms`` is the sum of the per-tag walls the leg already measures and
      already prints on its ``WEG2-FLIP-TAG`` lines (``weg2_per_tag[tag][1]``),
      so the shadow's line and the ring's lines carry the SAME instrument and a
      reader can subtract them.  It is ``None`` on the source hook, which runs
      before the leg's own wall exists -- and ``None`` prints ``n/a``, which is
      the point of that field.
    """

    leg: int
    epoch: str
    direction: str
    hook: str
    rank: int
    row: int
    peer_row: int
    device: int
    card_uuid: str
    uuid_of_card: Tuple[str, ...] = ()
    wave: int = 0
    free_mib: int = 0
    resume_reserve_bytes: int = 0
    ring_ms: Optional[float] = None
    #: THE ROWS THIS HOOK MAY EXPECT TO SEE VOTED AT THIS INSTANT, or ``None``
    #: for all six.  Its producer is the adapter, because the adapter is the
    #: only thing that knows which GROUP this rank is in -- and the source and
    #: the destination hook are at opposite ends of the flip, so the answer is
    #: different for each.  See :func:`shadow_gate`'s ``expect_rows``.
    gate_rows: Optional[Tuple[int, ...]] = None
    #: S5c.  WHY THIS LEG HAS NO PLAN, when it has none.  The ``no-plan``
    #: refusal used to print one fixed sentence -- "build_plan has no product
    #: caller" -- which was true while there was no derivation and is a LIE the
    #: moment there is one: a boot that refuses because its chunk geometry is
    #: unset, or because a rank carries a tag outside the family, would still
    #: have printed the sentence naming an absence that no longer exists.  The
    #: adapter derives, and hands the derivation's own reason word down here.
    plan_reason: str = ""
    #: CAN THE CO-LOCATED PEER DRAIN THIS RANK'S BOUNCE WHILE THIS HOOK HOLDS
    #: ITS THREAD?  Its producer is the ADAPTER, because the answer is a
    #: property of WHERE the two hooks are placed, which only the adapter knows.
    #:
    #: MEASURED-BY-REVIEW DEFECT (S5c refuter, must_fix 3).  On
    #: ``weight_updater``'s placement the answer is NO, structurally: the source
    #: hook runs before the sleep leg's pause loop, ``credit.publish`` runs
    #: INSIDE that loop, and the co-located waking rank's destination hook runs
    #: after a ``resume`` that is fenced (C14) on exactly that credit.  So the
    #: consumer of this bounce cannot exist while the producer blocks on it.
    #: With a non-empty plan -- which is what S5c added -- every armed source
    #: leg would therefore fill a bounce, block in ``drain-final``, time out at
    #: the budget, mark its PROD row FAILED and raise W53; the destination hook
    #: would then find a FAILED row and raise at once.  Net product effect:
    #: seconds added to every sleep leg, on the critical path of the C14 credit,
    #: and ZERO bytes ever compared.
    #:
    #: So the lane is refused BY NAME, before a bounce exists, and what the
    #: shadow still delivers on this placement is the part that needs no lane:
    #: the derivation, its provenance line, and the gate rendezvous in which
    #: ``plan_digest`` and ``card_digest`` ARE compared across the rows -- which
    #: is also the honest answer to refuter finding 9, since no byte now moves
    #: under a plan its consumer has not seen.  ``True`` is the default because
    #: it is the truth for every concurrent caller (S6's RPC handler, and every
    #: hermetic test that drives both ends of the lane at once).
    #:
    #: S6: ``False`` no longer means the lane is refused.  It means the lane
    #: needs the STORE-AND-FORWARD shape, which needs no concurrent peer -- and
    #: the refusal now fires only when that shape does not fit either (W65,
    #: :func:`deposit_refusal_message`) or when the arm cannot carry it (the
    #: ``ipc`` arm, which keeps the old blameless line).
    oncard_drainable: bool = True
    #: WHAT THE #1269 HOST LEDGER CHARGED FOR ONE CARD'S DEPOSIT, in bytes
    #: (:func:`sglang.srt.weg2.host_ledger.xchg_bounce_bytes_per_card`).  Its
    #: producer is the ADAPTER, for the same reason ``gate_rows`` has one: the
    #: budget is a property of the BOOT's arm, and a rank hook cannot read the
    #: launcher's ladder.  ``0`` is "no ledger answer reached this rank" and
    #: refuses the deposit by name -- an absent measurement never becomes a
    #: quiet zero, and pinned host bytes above what a term carries are exactly
    #: the shape ``host-schwelle-nie-uebertreten`` forbids.
    host_bounce_budget_bytes: int = 0


class ShadowLeg:
    """ONE flip leg's shadow.  The leg owns it; nothing else may close it.

    LIFETIME, which is item 4 of the S5b brief and a real defect class rather
    than hygiene:

    * :meth:`start` refuses to begin while another leg's run is still open
      (``reason=stale-run``) and closes the stale one, so a boot cannot
      accumulate one raw ``cudaMalloc`` per flip on the card that is short of
      VRAM by 550 MiB before the shadow asks for a byte;
    * :meth:`close` is IDEMPOTENT and OWNERSHIP-CHECKED: a caller that did not
      create this run cannot close it (``owner=`` must match the token
      :meth:`start` handed back).  The handler that wraps the hook therefore
      cannot free a buffer a LATER leg is using, which is the same shape as
      the stale-``ShadowRun`` leak refuter must_fix 3 found one layer down;
    * the buffer is freed BEFORE the hook returns, on every path, including the
      ones where the transport raised: the ``finally`` is in :func:`run_leg_hook`
      and the ownership token makes the double-close a no-op rather than a
      free of somebody else's pointer.
    """

    def __init__(self, inputs: ShadowLegInputs, log: Callable[[str], None]):
        self.inputs = inputs
        self.log = log
        self.token = f"{os.getpid()}:{inputs.epoch}:{inputs.leg}:{inputs.hook}"
        self.region: Optional[xr.XchgRegion] = None
        self.sems: Optional[tp.SemSet] = None
        self.ops: Optional[tp.DeviceOps] = None
        self.run: Optional[ShadowRun] = None
        #: WHICH OF THE THREE THIS LEG OPENED ITSELF.  MEASURED DEFECT, found
        #: by this round's own end-to-end test as a SIGSEGV and a junit.xml
        #: that was never written: :meth:`close` used to close all three
        #: unconditionally, so a leg handed a region and a device layer by its
        #: CALLER unmapped them out from under that caller -- and in the
        #: product's own shape (two co-located ranks, one region) the source
        #: leg's close would land while the destination leg is still reading.
        #: "The handler cannot close a run it does not own" is the rule; a run
        #: does not own what it was handed either.
        self._opened: set = set()
        self.sems_armed: Optional[int] = None
        self.sems_reason: str = ""
        self.closed = False

    # -- lifetime ---------------------------------------------------------

    def adopt(self) -> "ShadowLeg":
        """Become this thread's active run, closing a stale one BY NAME."""
        stale = _active_leg()
        if stale is not None and stale is not self:
            self.log(rank_local_skip_message(
                reason="stale-run", rank=self.inputs.rank,
                leg=self.inputs.leg, epoch=self.inputs.epoch,
                detail=f"previous={stale.token}"))
            stale.close(owner=stale.token)
        _set_active_leg(self)
        return self

    def close(self, *, owner: str) -> bool:
        """Free everything this run owns.  ``False`` when the caller is not it."""
        if owner != self.token:
            return False
        if self.closed:
            return True
        self.closed = True
        for name in ("run", "ops", "sems", "region"):
            obj = getattr(self, name, None)
            setattr(self, name, None)
            # ``run`` is always this leg's own -- :func:`shadow_transport`
            # builds it here and its buffer is the one raw cudaMalloc that must
            # not survive the leg.  The other three are closed ONLY if this leg
            # opened them; see :attr:`_opened`.
            if obj is None or (name != "run" and name not in self._opened):
                continue
            try:
                obj.close()
            except BaseException:  # noqa: BLE001 -- an observer's unwind
                pass
        if _active_leg() is self:
            _set_active_leg(None)
        return True

    # -- attach -----------------------------------------------------------

    def attach(self, *, region=None, sems=None, ops=None) -> str:
        """Open region, semaphores and device ops.  ``""`` on success, else a reason.

        Every one of the three is INJECTABLE, which is what makes the hook
        testable at all: the hermetic double hands in S4's fake device layer and
        a region on ``tmp_path``, and the product hands in nothing and gets the
        env-driven real ones.
        """
        i = self.inputs
        if region is not None:
            self.region = region
        else:
            path = (os.environ.get(xr.ENV_REGION_PATH, "") or "").strip()
            boot = (os.environ.get(xr.ENV_REGION_BOOT, "") or "").strip()
            if not path or not boot:
                return "no-region"
            try:
                self.region = xr.XchgRegion.open(path, expect_boot=boot)
                self._opened.add("region")
            except BaseException as exc:  # noqa: BLE001
                self.sems_reason = f"{type(exc).__name__}: {exc}"
                return "no-region"
            try:
                # THE FLIP TOKEN IS COMPOSED FROM THIS REGION'S OWN BOOT NONCE,
                # not from the RPC's epoch, and the two are not the same string:
                # the RPC carries ``weg2_memory_saver.credit_epoch`` =
                # ``<TMS_HOST_RING_EPOCH>.<flip>`` while the region was created
                # under the launcher's xchg boot nonce, and ``begin_flip``
                # REFUSES a token whose boot half is not its own (W52).  The
                # FLIP half is the shared quantity and it is what
                # ``ShadowLegInputs.leg`` carries -- derived, in the adapter,
                # from that same epoch token rather than from a counter this
                # class does not have.
                self.region.begin_flip(f"{self.region.boot_nonce}.{int(i.leg)}")
            except BaseException as exc:  # noqa: BLE001
                self.sems_reason = f"{type(exc).__name__}: {exc}"
                return "no-region"
        if sems is not None:
            self.sems = sems
        else:
            try:
                self.sems = tp.SemSet(self.region.boot_nonce)
                self._opened.add("sems")
            except BaseException as exc:  # noqa: BLE001
                self.sems_reason = f"{type(exc).__name__}: {exc}"
                return "no-sems"
        if ops is not None:
            self.ops = ops
        else:
            try:
                self.ops = tp.CudartDeviceOps()
                self._opened.add("ops")
            except BaseException as exc:  # noqa: BLE001
                self.sems_reason = f"{type(exc).__name__}: {exc}"
                return "no-ops"
        return ""

    def verify_sems(self) -> str:
        """W62 AT LEG START -- **counted here, never raised into the leg**.

        SECTION 1ai-S5's UNPROVEN 9 said ``verify_sem_arm`` has no caller.  This
        is the caller, and the placement is a deliberate narrowing of that
        function's own ``TODO(S6)``: it names the RPC preamble, which is where a
        refusal that STOPS A FLIP belongs, and the shadow may not stop a flip.
        So the same check runs at the start of the shadow's leg and its refusal
        decides only whether the SHADOW runs -- ``reason=w62-stale`` on the
        line, ``sems=stale`` in the census field, and the flip proceeds on the
        ring.  When S6 puts it in the preamble as an authoritative refusal this
        call becomes redundant and should be deleted, not kept as a second one.
        """
        if self.sems is None:
            return "no-sems"
        try:
            census = tp.verify_sem_arm(self.sems, leg=self.inputs.leg,
                                       epoch=self.inputs.epoch, log=self.log)
        except tp.Weg2XchgSemaphoreNotRearmed as exc:
            self.sems_armed = None
            self.sems_reason = str(exc)
            self.log(str(exc))
            return "w62-stale"
        except BaseException as exc:  # noqa: BLE001
            self.sems_reason = f"{type(exc).__name__}: {exc}"
            return "sem-check-failed"
        self.sems_armed = int(census.get("checked", 0))
        return ""


def oncard_mode_word(mode: str) -> int:
    """The Gate-0 word for an on-card mode.  THE MODE WORD'S PRODUCER.

    SECTION 1ai-S5's UNPROVEN 10: "the Gate-0 mode word has no producer -- every
    rank publishes UNSTATED".  ``gate0_publish`` takes the word; this is the one
    place that turns the launcher's arm (``--weg2-xchg-oncard`` ->
    ``SGLANG_WEG2_XCHG_ONCARD`` -> the mode the transport resolved) into it, so
    a Gate-0 publisher no longer has to invent one.  An unknown mode maps to
    ``UNSTATED`` rather than to a guess: ``gate0_check`` treats UNSTATED as
    "not a disagreement" and a wrong word would be one.
    """
    return int(xr.ONCARD_MODE_WORD.get(str(mode), xr.ONCARD_MODE_UNSTATED))


def resolve_shadow_oncard_mode() -> str:
    """The mode the shadow's lane uses, from the launcher flag, never probed.

    ``ipc`` is the default and ``host`` the operator's degrade; the shadow does
    NOT run ``tp.resolve_oncard_mode``'s probe, because that probe allocates and
    an observer that probes has changed the card it is observing before it has
    been allowed to run at all.  An unreadable value is ``host``, the degrade,
    not a raise: this decision may not stop a leg.
    """
    requested = (os.environ.get(tp.ENV_ONCARD_MODE, "") or "").strip()
    if requested == tp.ONCARD_MODE_HOST:
        return tp.ONCARD_MODE_HOST
    if requested in ("", tp.ONCARD_MODE_IPC):
        return tp.ONCARD_MODE_IPC
    return tp.ONCARD_MODE_HOST


def shadow_armed() -> bool:
    """Is ``--weg2-weight-source shadow`` the arm of this boot?

    THE ONE GATE ON BOTH HOOKS.  On ``ring`` -- the default and every boot that
    has ever run -- this is False, both adapters return before they touch a
    single symbol of this module's machinery, and the leg is byte-identical to
    today's.  ``test_the_hooks_are_never_reached_on_the_ring_arm`` is the proof.
    """
    from sglang.srt.weg2 import weight_exchange as wxm

    # ONE DEFINITION OF ONE DECISION (S5b refuter, non-blocking finding):
    # ``weight_exchange.shadow_armed`` is the owner and this delegates to it.
    # It used to re-spell the comparison (``weight_source() == SHADOW``), which
    # is a second copy of a predicate -- the Zweitbuchhaltung shape
    # UPSTREAM-MINIMAL refuses, and the shape that lets two arms drift.
    return bool(wxm.shadow_armed())


def publish_no_vote(leg: ShadowLeg, *, reason: str,
                    classes_hash: int = 0, need_mib: int = 0) -> bool:
    """Publish this rank's NO before returning, when the region is already open.

    MEASURED-BY-REVIEW DEFECT (S5b refuter, must_fix 5): every refusal that
    happens BETWEEN the attach and the gate returned without writing a row --
    so a rank that refused on its own arithmetic was indistinguishable, to its
    five peers, from a rank that crashed, and they paid a whole gate budget
    inside their own flip legs discovering it.  The hazard is named verbatim in
    W63's own docstring, and the code then only logged it.

    A row IS available on those paths (``leg.attach`` succeeded and
    ``begin_flip`` stamped the region), so the NO costs one 64-byte sealed
    write and turns every peer's wait into an immediate, named refusal.

    ``classes_hash`` may be 0 here and that is not a disagreement: the gate
    checks refusers BEFORE it compares class hashes, so a NO row's hash is
    never read.  ``False`` when there was no region to write into -- the
    pre-attach reasons (``no-region``, ``no-sems``, ``no-ops``, ``no-plan``)
    have nowhere to publish, by construction, and W63 names them instead.
    """
    region = getattr(leg, "region", None)
    if region is None:
        return False
    i = leg.inputs
    try:
        write_shadow_vote(region, int(i.row), leg=int(i.leg), vote=False,
                          classes_hash=int(classes_hash), need_mib=int(need_mib))
    except BaseException:  # noqa: BLE001 -- an observer never raises
        return False
    leg.log(
        f"{SHADOW_GATE_LINE_PREFIX} leg={i.leg} epoch={i.epoch} row={i.row} "
        f"vote=no reason={reason} -- published BEFORE the gate, so no peer "
        f"pays a gate budget inside its own flip leg discovering this row is "
        f"not coming"
    )
    return True


def _make_summer(ops: tp.DeviceOps, device: int, stripe_bytes: int,
                 result: ShadowResult):
    """``(sum_bytes, stream, scratch)`` -- THE PRODUCT PRODUCER of the summer.

    MEASURED-BY-REVIEW DEFECT (S5b refuter, must_fix 6): the ``weight_updater``
    adapter passed no ``sum_bytes``, so :meth:`ShadowRun.compare` took its
    ``no-summer`` exit and ``tp.run_leg`` ran with its slot checksums disabled
    -- the shipping shape reported ``verdict=NOT-RUN`` for TWO independent
    reasons while the record named one.  :func:`make_device_scratch` and
    :func:`device_summer` had no product caller anywhere; this is it.

    IT IS BUILT AFTER THE GATE AND AFTER THE TRANSPORT, never before: the
    scratch is a 64 MiB device allocation, and "allocate only once the gate
    said yes" is the rule :func:`shadow_transport` states for the destination
    buffers.  It is the term :func:`price_leg` already prices as
    ``scratch_bytes`` on the destination side, so it is priced before it is
    taken, not after.

    Every failure is a counted error and ``(None, None, None)`` -- an
    unavailable summer reads as ``verdict=NOT-RUN reason=no-summer``, which is
    the honest reading, and never as a match.
    """
    try:
        stream = ops.create_stream(int(device))
    except BaseException as exc:  # noqa: BLE001 -- an observer never raises
        result.counters.errors.append(f"no-summer-stream {type(exc).__name__}: {exc}")
        return None, None, None
    try:
        scratch = make_device_scratch(int(stripe_bytes), int(device))
        return device_summer(ops, stream, scratch), stream, scratch
    except BaseException as exc:  # noqa: BLE001
        result.counters.errors.append(f"no-summer-scratch {type(exc).__name__}: {exc}")
        destroy = getattr(ops, "destroy_stream", None)
        if destroy is not None:
            try:
                destroy(stream)
            except BaseException:  # noqa: BLE001
                pass
        return None, None, None


def run_leg_hook(
    inputs: ShadowLegInputs,
    *,
    log: Callable[[str], None],
    descs: Optional[Sequence[object]] = None,
    #: S5c.  THE DERIVED PLAN, from :func:`derive_leg_plan`.  When it is given,
    #: it supplies the descriptors, the class rotation the subset is chosen
    #: from, and the two digests the gate compares -- and its provenance line
    #: is logged ONCE, here, before anything is opened.  ``descs`` stays for
    #: the hermetic callers that hand descriptors directly.
    plan: Optional[LegPlan] = None,
    region=None,
    sems=None,
    ops=None,
    sum_bytes: Optional[Callable[[int, int], int]] = None,
    classes: Sequence[str] = (),
    per_leg: int = 1,
    bound_ms: Optional[float] = None,
    explicit: bool = False,
    armed: Optional[bool] = None,
    # THE ONE DEADLINE THE TWO WAITS ARE CARVED OUT OF.  See
    # :data:`SHADOW_HOOK_BUDGET_S`; ``gate_budget_s`` and ``budget_s`` are
    # CEILINGS on their step, and what is left of this deadline is the other
    # ceiling, so their sum can never be paid.
    hook_budget_s: float = SHADOW_HOOK_BUDGET_S,
    gate_budget_s: float = SHADOW_GATE_BUDGET_S,
    # The staging geometry, FORWARDED and never re-derived.  These are already
    # parameters of :func:`shadow_transport` with exactly these defaults; the
    # hook passes them through so a caller whose region has a different
    # geometry (the hermetic double's 4 KiB slot) drives the same code path the
    # product does instead of a second one written for it.
    slot_bytes: int = xr.SLOT_BYTES,
    stripe_bytes: int = STRIPE_BYTES,
    budget_s: float = SHADOW_TRANSPORT_BUDGET_S,
    oncard_slot_bytes: Optional[int] = None,
) -> Optional[ShadowResult]:
    """ONE leg's shadow, start to close.  Returns ``None`` when not armed.

    THE ORDER, and every step is a refusal point that decides only the shadow:

    1. the arm (``--weg2-weight-source shadow``) -- ``None`` on every other one;
    2. adopt the process's active-run slot, closing a stale one (W63);
    3. attach region + semaphores + device ops (W63 on each, by name);
    4. ``verify_sem_arm`` (W62, counted);
    5. the plan (W63 ``no-plan`` -- the standing state, see :func:`plan_for_leg`);
    6. the priced hop against the launcher's bound (W61 ``scope=hop``);
    7. the transport, and on the destination hook the compare;
    8. ``close()``, in a ``finally``, with the ownership token.

    Step 6 runs BEFORE step 7 on purpose: the bound is a PREDICTION the leg is
    allowed to act on, and ``shadow_ms`` is the MEASUREMENT it is graded
    against afterwards.  A measured overshoot is a finding on the line, never a
    retro-active refusal -- there is nothing to refuse once the wall has been
    spent, and pretending otherwise is the compensating-reader shape.

    **EVERY WAIT IS INSIDE ONE DEADLINE** (:data:`SHADOW_HOOK_BUDGET_S`;
    S5b refuter must_fix 4).  ``bound_ms`` grades the priced on-card HOP and
    nothing else -- it never covered the attach, the gate, the transport or the
    compare, all of which sat on the leg's critical path.  The two things that
    can WAIT (the gate rendezvous and the transport's slot protocol) now draw
    from one deadline.  The gate is ONE wait and is given ``min(its ceiling,
    what is left)``; the TRANSPORT performs many and is given the deadline
    itself as a callable, so each of its waits is re-clamped (S5c refuter,
    must_fix 2 -- one number handed down was a per-wait ceiling being read as a
    total).  Work is not a wait and is not clamped -- the compare's device sums
    are bounded by the class subset instead -- but the total IS measured
    (``shadow_ms``), the DERIVATION's wall is subtracted from the deadline and
    printed (``derive_ms=``, must_fix 6), and ``budget=ok|OVER`` grades their
    sum against ``hook_budget_ms``, so an overrun is a reading rather than a
    hidden cost.
    """
    if armed is None:
        armed = shadow_armed()
    if not armed:
        return None
    started = time.perf_counter()
    hook_budget_s = max(0.0, float(hook_budget_s))
    # THE DERIVATION IS INSIDE THE DEADLINE (S5c refuter, must_fix 6).  It runs
    # in the adapter, BEFORE this clock starts, on the flip's critical path: a
    # walk over every named parameter plus a regex tag call and a ``ParamGeom``
    # per tensor, then ``build_plan``'s sort and its per-parameter emit.  With
    # it outside, ``SHADOW_HOOK_BUDGET_S``'s claim to be "the ONE wall a flip
    # leg pays for having an observer ... the only bound the user law can be
    # read off in one place" was false by exactly that wall.  It is subtracted
    # here rather than re-timed, because the adapter already measured it and a
    # second measurement of one quantity is the defect this round removed from
    # three other places.
    derive_ms = float(getattr(plan, "derive_ms", 0.0) or 0.0)
    deadline = started + max(0.0, hook_budget_s - derive_ms / 1e3)
    left = lambda: max(0.0, deadline - time.perf_counter())  # noqa: E731
    summer_stream = None
    leg = ShadowLeg(inputs, log).adopt()
    result = ShadowResult(leg=int(inputs.leg), epoch=str(inputs.epoch),
                          subset=select_subset((), leg=int(inputs.leg)),
                          counters=ShadowCounters(),
                          direction=str(inputs.direction))
    result.ring_ms = inputs.ring_ms
    bound = float(SHADOW_HOP_BOUND_MS_DEFAULT)
    try:
        try:
            bound = hop_bound_ms(bound_ms)
        except (TypeError, ValueError) as exc:
            result.reason = f"bad-bound:{type(exc).__name__}"
            return result
        # THE PLAN IS ASKED FOR FIRST, BEFORE A FILE IS OPENED.  Not a
        # preference: a leg with nothing to move must not map a 385 MiB region,
        # open 24 semaphores and load libcudart to find that out -- and the
        # standing state of this boot IS "no plan" (see :func:`plan_for_leg`),
        # so an attach-first order would report ``no-region`` on every rank of
        # every boot that has no region either, and the two reasons would be
        # indistinguishable in exactly the case the reader cares about.
        plan_digest = card_digest = 0
        if plan is not None:
            # ONCE PER LEG, BEFORE ANYTHING IS OPENED.  A plan that is refused
            # further down (unaffordable, gate expiry) still leaves its own
            # provenance on the log, which is the line a reader needs most on a
            # leg that did not run.
            log(plan.line())
            descs = plan.descs
            classes = classes or plan.classes
            plan_digest = plan.facts.digest
            card_digest = plan.card_digest
        plan_descs = tuple(descs) if descs is not None else plan_for_leg(
            inputs.direction, inputs.leg, inputs.rank)
        if not plan_descs:
            log(rank_local_skip_message(
                reason="no-plan", rank=inputs.rank, leg=inputs.leg,
                epoch=inputs.epoch,
                detail=(inputs.plan_reason
                        or "no derivation was handed to this leg")))
            if explicit:
                raise Weg2XchgShadowRankLocalSkip("no-plan")
            result.reason = "no-plan"
            return result
        reason = leg.attach(region=region, sems=sems, ops=ops)
        if reason:
            log(rank_local_skip_message(
                reason=reason, rank=inputs.rank, leg=inputs.leg,
                epoch=inputs.epoch, detail=leg.sems_reason))
            if explicit:
                raise Weg2XchgShadowRankLocalSkip(reason)
            result.reason = reason
            return result
        sem_reason = leg.verify_sems()
        if sem_reason:
            publish_no_vote(leg, reason=sem_reason)
            result.reason = sem_reason
            return result
        is_source = inputs.hook == HOOK_SOURCE
        mode = resolve_shadow_oncard_mode()
        subset = select_subset(plan_descs, leg=inputs.leg, classes=classes,
                               per_leg=per_leg)
        # NO HAND ARITHMETIC: the slot, the batch count and the hop all come
        # from ``plan_oncard_slot_bytes``'s own :class:`tp.OnCardSlotPlan`,
        # which carries the measured coefficient and prints its own cost model.
        # Recomputing ``batches x per_batch_ms`` here would be a second copy of
        # a formula that already has one producer -- and the previous cut of
        # this line did exactly that and divided by the PLAN OBJECT.
        # A PINNED slot is pinned through the SAME producer (floor == ceiling),
        # so the batch count and the hop stay its arithmetic and not ours.
        lane_bytes = oncard_lane_bytes(subset.descs, inputs.rank)
        # S6: THE SLOT COUNT COMES OUT OF THE SAME PRODUCER AS THE SLOT SIZE.
        # ``run_leg_hook`` had no ``oncard_slots`` parameter, so every product
        # path ran the lane at the module constant ``tp.ONCARD_SLOTS = 2`` --
        # a hand number one seam below a planner that already knew the batch
        # count.  A deposit needs one slot per batch, so the count is derived
        # here, printed with its provenance (``oncard_slots_source=``), and
        # handed to ``shadow_transport`` as ONE reading, exactly as
        # ``diag_slot`` already is: the W52 cross-check between the two
        # co-located processes then still compares two readings of one
        # derivation rather than two derivations.
        store_forward = not bool(inputs.oncard_drainable)
        budget_bytes = int(inputs.host_bounce_budget_bytes)
        lane = (tp.plan_oncard_slot_bytes(
                    lane_bytes, store_forward=store_forward,
                    host_budget_bytes=budget_bytes)
                if oncard_slot_bytes is None
                else tp.plan_oncard_slot_bytes(
                    lane_bytes, floor_bytes=int(oncard_slot_bytes),
                    ceiling_bytes=int(oncard_slot_bytes),
                    store_forward=store_forward,
                    host_budget_bytes=budget_bytes))
        diag_slot = int(lane.slot_bytes)
        batches = int(lane.batches)
        # ONE SLOT PER BATCH THE BATCHER WILL ACTUALLY CUT.  ``lane.slots`` is
        # derived from the hop model's ``ceil(bytes / slot)``, which is a LOWER
        # bound on that count (:func:`oncard_lane_batches`), so the deposit
        # takes the larger of the two.  ``or tp.ONCARD_SLOTS`` covers the empty
        # diagonal, where the plan derives 0 and ``require_oncard_slots``
        # refuses a depth of zero one layer down.
        diag_slots = int(tp.ONCARD_SLOTS)
        if store_forward:
            diag_slots = max(int(lane.slots),
                             oncard_lane_batches(subset.descs, inputs.rank,
                                                 diag_slot)) or int(tp.ONCARD_SLOTS)
        priced_ms = float(lane.hop_ms)
        if priced_ms > bound:
            message = hop_refusal_message(
                card=inputs.card_uuid, priced_ms=priced_ms, bound_ms=bound,
                batches=batches, slot_mib=diag_slot / MIB, leg=inputs.leg)
            log(message)
            result.subset = subset
            result.oncard_slot_mib = diag_slot / MIB
            result.oncard_batches = batches
            result.oncard_slots = diag_slots
            result.oncard_slots_source = lane.slots_source
            result.oncard_deposit_mib = diag_slots * diag_slot / MIB
            result.oncard_hop_ms_priced = priced_ms
            result.reason = "hop-over-bound"
            # THE ASYMMETRIC REFUSAL, PUBLISHED.  ``hop_ms`` is priced from
            # THIS CARD's diagonal, so one card can refuse while the other two
            # do not -- exactly the case where a silent return costs five
            # peers a gate budget each.
            publish_no_vote(leg, reason="hop-over-bound",
                            classes_hash=subset.hash, need_mib=0)
            if explicit:
                raise Weg2XchgShadowUnaffordable(message)
            return result
        run = shadow_transport(
            region=leg.region, sems=leg.sems, ops=leg.ops, row=inputs.row,
            rank=inputs.rank, device=inputs.device,
            card_uuid=inputs.card_uuid,
            uuid_of_card=tuple(inputs.uuid_of_card), descs=plan_descs,
            is_source=is_source, oncard_mode=mode, peer_row=inputs.peer_row,
            wave=inputs.wave, leg=inputs.leg, direction=inputs.direction,
            epoch=inputs.epoch, free_mib=inputs.free_mib, log=log,
            sum_bytes=sum_bytes, classes=classes, per_leg=per_leg,
            resume_reserve_bytes=inputs.resume_reserve_bytes,
            explicit=explicit, slot_bytes=slot_bytes,
            stripe_bytes=stripe_bytes,
            # BOTH WAITS OUT OF ONE DEADLINE -- see the docstring.  The gate
            # is ONE wait, so clamping it once at entry is exact; the transport
            # performs MANY, so it gets the deadline itself as a callable
            # (S5c refuter, must_fix 2) and not just a first slice of it.
            budget_s=min(float(budget_s), left()),
            budget_left=left,
            gate_budget_s=min(float(gate_budget_s), left()),
            oncard_drainable=bool(inputs.oncard_drainable),
            # S6: the deposit's two inputs, from the two producers that own
            # them -- the slot count from the lane plan above, the host budget
            # from the adapter's reading of the #1269 ledger's charge.
            oncard_slots=diag_slots,
            host_bounce_budget_bytes=budget_bytes,
            oncard_store_forward=store_forward,
            gate_rows=inputs.gate_rows,
            plan_digest=plan_digest, card_digest=card_digest,
            # THE CO-LOCATED PAIR IS ONLY VISIBLE TO THE DESTINATION.  The
            # source expects its own group's three rows (S5b must_fix 1), and
            # its peer's row is one of the three it does NOT wait for -- asking
            # it to compare a row it never reads would reintroduce exactly the
            # circular wait that fix removed.
            check_peer_card=(not is_source) and card_digest != 0,
            # ONE NUMBER FOR THE DIAGONAL SLOT.  The hook already priced the hop
            # from it; letting ``shadow_transport`` derive it a second time is
            # two computations of one quantity, and the W52 cross-check between
            # the two co-located processes would then be comparing two
            # derivations rather than two readings of the same one.
            oncard_slot_bytes=diag_slot)
        leg.run = run
        result = run.result
        result.ring_ms = inputs.ring_ms
        if not is_source:
            # THE COMPARE IS THE DESTINATION HOOK'S WHOLE POINT and it runs
            # here, after the ring's own restore, against the ring's own bytes.
            if sum_bytes is None and result.ran:
                # The scratch itself is held by the closure the summer IS
                # (``device_summer`` closes over the tensor), so there is no
                # second reference here to forget to drop.
                sum_bytes, summer_stream, _scratch = _make_summer(
                    leg.ops, int(inputs.device), stripe_bytes, result)
                del _scratch
            run.compare(sum_bytes, log)
        return result
    except Weg2XchgShadowUnaffordable:
        raise
    except Weg2XchgShadowRankLocalSkip:
        raise
    except BaseException as exc:  # noqa: BLE001 -- a hook never raises into a leg
        result.counters.errors.append(f"{type(exc).__name__}: {exc}")
        result.reason = f"hook-failed:{type(exc).__name__}"
        result.ran = False
        return result
    finally:
        # THE SUMMER'S STREAM BEFORE THE LEG'S OPS, because ``leg.close`` nulls
        # them: a stream destroyed through a ``None`` is a leaked stream on the
        # card the shadow is not allowed to disturb.
        destroy = getattr(leg.ops, "destroy_stream", None)
        if summer_stream is not None and destroy is not None:
            try:
                destroy(summer_stream)
            except BaseException:  # noqa: BLE001 -- an observer's unwind
                pass
        leg.close(owner=leg.token)
        result.shadow_ms = (time.perf_counter() - started) * 1e3
        result.hook_budget_ms = hook_budget_s * 1e3
        result.hop_bound_ms = bound
        result.sems_armed = leg.sems_armed
        result.hook = str(inputs.hook)
        result.resume_reserve_mib = int(
            -(-int(inputs.resume_reserve_bytes) // MIB))
        result.plan_digest = plan.facts.digest if plan is not None else 0
        result.derive_ms = derive_ms
        log(result.line())
