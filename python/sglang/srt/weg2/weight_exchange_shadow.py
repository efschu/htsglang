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
SHADOW_ROW_BYTES = 64
#: ``epoch_hash, leg, vote, classes_hash, need_mib, pid, ts_ns`` + a seal.
SHADOW_ROW_STRUCT = struct.Struct("<7Q")
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
                      classes_hash: int, need_mib: int) -> None:
    """Publish this rank's shadow vote.  ONE WRITER PER ADDRESS, sealed.

    Same discipline as every other row in this region (S3's gate rows, S4's
    on-card rows): a rank writes only its own row, the row is sealed, and a
    half-written row is not a signal.
    """
    view = region.dir_view()
    off = _row_off(row)
    payload = SHADOW_ROW_STRUCT.pack(
        region.epoch_hash, int(leg), VOTE_YES if vote else VOTE_NO,
        int(classes_hash) & ((1 << 64) - 1), int(need_mib), os.getpid(),
        time.time_ns())
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
    eh, leg, vote, classes_hash, need_mib, pid, ts_ns = \
        SHADOW_ROW_STRUCT.unpack(payload)
    return {"epoch_hash": eh, "leg": leg, "vote": vote,
            "classes_hash": classes_hash, "need_mib": need_mib, "pid": pid,
            "ts_ns": ts_ns, "sealed": int(seal == xr._seal(payload))}


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

    def line(self, *, leg: int, epoch: str) -> str:
        return (
            f"{SHADOW_GATE_LINE_PREFIX} leg={leg} epoch={epoch} "
            f"joined={self.joined}/{xr.N_RANKS} "
            f"run={'yes' if self.run else 'no'} "
            f"refusers={','.join(str(r) for r in self.refusers) or 'none'} "
            f"waited_s={self.waited_s:.3f} skew_ms={self.skew_ms:.3f} "
            f"reason={self.reason}"
        )


def shadow_gate(region: xr.XchgRegion, row: int, *, leg: int, vote: bool,
                classes_hash: int, need_mib: int,
                log: Callable[[str], None],
                budget_s: float = SHADOW_GATE_BUDGET_S,
                poll_s: float = xr.GATE_POLL_S,
                monotonic: Callable[[], float] = time.monotonic) -> ShadowVerdict:
    """RANK-UNIFORM: any rank refusing means NO rank runs the shadow.

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
    write_shadow_vote(region, row, leg=leg, vote=vote,
                      classes_hash=classes_hash, need_mib=need_mib)
    started = monotonic()
    while True:
        rows = [read_shadow_vote(region, r) for r in range(xr.N_RANKS)]
        fresh = [r for r in rows
                 if r["sealed"] and r["epoch_hash"] == region.epoch_hash
                 and r["leg"] == int(leg)]
        if len(fresh) == xr.N_RANKS:
            refusers = tuple(i for i, r in enumerate(rows) if not r["vote"])
            hashes = {r["classes_hash"] for r in rows}
            if refusers:
                return ShadowVerdict(
                    False, len(fresh), refusers,
                    "a rank refused: " + " ".join(
                        f"[row={i} pid={rows[i]['pid']} "
                        f"need_mib={rows[i]['need_mib']}]" for i in refusers),
                    monotonic() - started)
            if len(hashes) != 1:
                return ShadowVerdict(
                    False, len(fresh), tuple(range(xr.N_RANKS)),
                    "the ranks chose different class subsets: " + " ".join(
                        f"[row={i} classes_hash={r['classes_hash']:#x}]"
                        for i, r in enumerate(rows)),
                    monotonic() - started)
            stamps = [int(r["ts_ns"]) for r in rows]
            return ShadowVerdict(True, len(fresh), (), "all six joined",
                                 monotonic() - started,
                                 (max(stamps) - min(stamps)) / 1e6)
        if monotonic() - started >= budget_s:
            missing = tuple(i for i, r in enumerate(rows)
                            if not (r["sealed"]
                                    and r["epoch_hash"] == region.epoch_hash
                                    and r["leg"] == int(leg)))
            return ShadowVerdict(
                False, len(fresh), missing,
                f"only {len(fresh)}/{xr.N_RANKS} ranks published a vote for "
                f"leg={leg} within {budget_s}s -- the shadow switches itself "
                f"off for this leg rather than wait inside the flip",
                monotonic() - started)
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


def oncard_lane_bytes(descs: Sequence[object], rank: int) -> int:
    """THIS CARD's diagonal inside ``descs`` -- ``run_leg``'s own filter.

    ONE DEFINITION, because the number is used twice (the bounce is priced from
    it, and the hop is priced from it) and two copies of a filter is how the
    sum-across-cards denominator got in.
    """
    return sum(int(d.nbytes) for d in descs
               if d.kind != tp.ZEROFILL and _is_on_card(d)
               and int(d.src_rank) == int(rank))


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
    moved = [d for d in descs if d.kind != tp.ZEROFILL]
    mine_dst = [d for d in moved if int(d.dst_rank) == int(rank)]
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
            f"oncard_hop_ms_priced={ms(self.oncard_hop_ms_priced)} "
            f"oncard_ms={self.oncard_ms:.3f} cross_ms={self.cross_ms:.3f} "
            f"ring_ms={ms(self.ring_ms)} compare_ms={self.compare_ms:.3f} "
            f"xchg_ms={self.oncard_ms + self.cross_ms:.3f} "
            f"issue_ms={self.issue_ms:.3f} slot_wait_ms={self.slot_wait_ms:.3f} "
            f"gate_skew_ms={self.gate_skew_ms:.3f} "
            f"lock_wait_ms={ms(self.lock_wait_ms)} "
            f"hook={self.hook} shadow_ms={self.shadow_ms:.3f} "
            f"hop_bound_ms={self.hop_bound_ms:.3f} "
            f"sems_armed={'n/a' if self.sems_armed is None else self.sems_armed} "
            f"resume_reserve_mib={self.resume_reserve_mib} "
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
            for stripe in stripes:
                verdict = classify(stripe)
                if verdict == MATCH:
                    counters.match += 1
                    continue
                if verdict == NOT_REPRESENTABLE:
                    counters.not_representable += 1
                else:
                    counters.mismatch += 1
                log(mismatch_message(stripe, verdict=verdict, leg=result.leg,
                                     epoch=result.epoch))
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
        if not price.affordable:
            log(price.message())
            if explicit:
                raise Weg2XchgShadowUnaffordable(price.message())
        verdict = shadow_gate(region, row, leg=leg, vote=price.affordable,
                              classes_hash=subset.hash, need_mib=price.need_mib,
                              log=log)
        log(verdict.line(leg=leg, epoch=epoch))
        # THE SIX RANKS' SPREAD, not this rank's wait -- see
        # :attr:`ShadowVerdict.skew_ms`.
        result.gate_skew_ms = verdict.skew_ms
        if not verdict.run:
            result.reason = verdict.reason.split(":")[0].replace(" ", "-")
            return run
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
            budget_s=budget_s, checksum_bytes=sum_bytes,
            on_checksum=on_checksum, slot_bytes=slot_bytes,
            # PRICED FROM THIS CARD'S OWN DIAGONAL WITHIN THE SUBSET, by the
            # same :func:`price_leg` call that sized the bounce from it: the
            # shadow moves a class subset, so its batch count -- and therefore
            # the hop this leg pays -- is a different number from the full
            # leg's, and the lane runs once per CARD, so it is a different
            # number from the subset's sum across the three cards too.
            oncard_slot_bytes=diag_slot,
            oncard_slots=oncard_slots)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        result.cross_ms = sum(p.elapsed_s for p in out.pairs) * 1e3
        result.oncard_ms = (out.oncard.elapsed_s * 1e3) if out.oncard else 0.0
        result.slot_wait_ms = sum(p.slot_wait_s for p in out.pairs) * 1e3
        result.issue_ms = max(0.0, elapsed_ms - result.cross_ms - result.oncard_ms)
        result.pieces = sum(p.pieces for p in out.pairs) + len(
            [d for d in leg_descs if _is_on_card(d)])
        result.ran = True
    except Weg2XchgShadowUnaffordable:
        raise
    except BaseException as exc:  # noqa: BLE001 -- an observer never raises
        result.counters.errors.append(f"{type(exc).__name__}: {exc}")
        result.reason = f"transport-failed:{type(exc).__name__}"
        result.ran = False
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
    * ``no-plan`` -- nothing handed this leg descriptors.  **This is the
      standing state**: :func:`plan_for_leg` has NO PRODUCER in this slice and
      says so, so a shadow boot today prints ``verdict=NOT-RUN reason=no-plan``
      on every leg of every rank.  That is the honest reading of "the exchange
      has no plan on the boot path yet" (``weight_exchange.build_plan`` has
      zero product callers, verified by ``test_the_plan_seam_has_no_producer``)
      and it is deliberately a NAMED ABSENCE rather than a fabricated identity
      plan: a plan a rank derives on its own would compare the ring's restored
      bytes against a copy of those same bytes and match by construction --
      the "instrument that cannot go red" this campaign has now paid for three
      times (``20c5fb9048``, refuter 1, refuter 2).
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
# The plan seam.  ONE named absence, not a silent one.
# ---------------------------------------------------------------------------

#: The descriptor producer for a shadow leg.  ``None`` is the shipping value.
#:
#: WHY IT IS EMPTY AND WHY THAT IS NOT A STUB.  ``weight_exchange.build_plan``
#: needs the inventory and the ``GroupLayout`` of BOTH groups -- cross-group
#: knowledge no single rank holds -- and it has no product caller anywhere
#: (``test_the_plan_seam_has_no_producer`` asserts that, so the day S6 wires
#: one this test goes red and this comment gets updated rather than rotting).
#: The alternative was an identity plan built from this rank's own
#: ``named_parameters()``; it is refused above, in W63's ``no-plan`` bullet,
#: with the reason.
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

#: The ONE shadow run this process owns, or ``None``.  A module-level slot and
#: not a scheduler attribute, because the thing it guards against is a run that
#: OUTLIVES the object that made it: a leg that raised between the allocation
#: and the close leaves a raw ``cudaMalloc`` alive for the life of the boot on
#: the tight card, which is refuter must_fix 3 one scope up.
_ACTIVE_LEG: Optional["ShadowLeg"] = None


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
        self.sems_armed: Optional[int] = None
        self.sems_reason: str = ""
        self.closed = False

    # -- lifetime ---------------------------------------------------------

    def adopt(self) -> "ShadowLeg":
        """Become the process's active run, closing a stale one BY NAME."""
        global _ACTIVE_LEG
        stale = _ACTIVE_LEG
        if stale is not None and stale is not self:
            self.log(rank_local_skip_message(
                reason="stale-run", rank=self.inputs.rank,
                leg=self.inputs.leg, epoch=self.inputs.epoch,
                detail=f"previous={stale.token}"))
            stale.close(owner=stale.token)
        _ACTIVE_LEG = self
        return self

    def close(self, *, owner: str) -> bool:
        """Free everything this run owns.  ``False`` when the caller is not it."""
        global _ACTIVE_LEG
        if owner != self.token:
            return False
        if self.closed:
            return True
        self.closed = True
        for name in ("run", "ops", "sems", "region"):
            obj = getattr(self, name, None)
            if obj is None:
                continue
            try:
                obj.close()
            except BaseException:  # noqa: BLE001 -- an observer's unwind
                pass
            setattr(self, name, None)
        if _ACTIVE_LEG is self:
            _ACTIVE_LEG = None
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
            except BaseException as exc:  # noqa: BLE001
                self.sems_reason = f"{type(exc).__name__}: {exc}"
                return "no-sems"
        if ops is not None:
            self.ops = ops
        else:
            try:
                self.ops = tp.CudartDeviceOps()
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

    return wxm.weight_source() == WEIGHT_SOURCE_SHADOW


def run_leg_hook(
    inputs: ShadowLegInputs,
    *,
    log: Callable[[str], None],
    descs: Optional[Sequence[object]] = None,
    region=None,
    sems=None,
    ops=None,
    sum_bytes: Optional[Callable[[int, int], int]] = None,
    classes: Sequence[str] = (),
    per_leg: int = 1,
    bound_ms: Optional[float] = None,
    explicit: bool = False,
    armed: Optional[bool] = None,
    # The staging geometry, FORWARDED and never re-derived.  These are already
    # parameters of :func:`shadow_transport` with exactly these defaults; the
    # hook passes them through so a caller whose region has a different
    # geometry (the hermetic double's 4 KiB slot) drives the same code path the
    # product does instead of a second one written for it.
    slot_bytes: int = xr.SLOT_BYTES,
    stripe_bytes: int = STRIPE_BYTES,
    budget_s: float = SHADOW_TRANSPORT_BUDGET_S,
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
    """
    if armed is None:
        armed = shadow_armed()
    if not armed:
        return None
    started = time.perf_counter()
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
            result.reason = sem_reason
            return result
        plan = tuple(descs) if descs is not None else plan_for_leg(
            inputs.direction, inputs.leg, inputs.rank)
        if not plan:
            log(rank_local_skip_message(
                reason="no-plan", rank=inputs.rank, leg=inputs.leg,
                epoch=inputs.epoch,
                detail="weight_exchange.build_plan has no product caller"))
            if explicit:
                raise Weg2XchgShadowRankLocalSkip("no-plan")
            result.reason = "no-plan"
            return result
        is_source = inputs.hook == HOOK_SOURCE
        mode = resolve_shadow_oncard_mode()
        subset = select_subset(plan, leg=inputs.leg, classes=classes,
                               per_leg=per_leg)
        diag_slot = tp.plan_oncard_slot_bytes(
            oncard_lane_bytes(subset.descs, inputs.rank))
        diag_bytes = oncard_lane_bytes(subset.descs, inputs.rank)
        batches = -(-diag_bytes // diag_slot) if diag_bytes else 0
        priced_ms = batches * tp.ONCARD_PER_BATCH_MS
        if priced_ms > bound:
            message = hop_refusal_message(
                card=inputs.card_uuid, priced_ms=priced_ms, bound_ms=bound,
                batches=batches, slot_mib=diag_slot / MIB, leg=inputs.leg)
            log(message)
            result.subset = subset
            result.oncard_slot_mib = diag_slot / MIB
            result.oncard_batches = batches
            result.oncard_hop_ms_priced = priced_ms
            result.reason = "hop-over-bound"
            if explicit:
                raise Weg2XchgShadowUnaffordable(message)
            return result
        run = shadow_transport(
            region=leg.region, sems=leg.sems, ops=leg.ops, row=inputs.row,
            rank=inputs.rank, device=inputs.device,
            card_uuid=inputs.card_uuid,
            uuid_of_card=tuple(inputs.uuid_of_card), descs=plan,
            is_source=is_source, oncard_mode=mode, peer_row=inputs.peer_row,
            wave=inputs.wave, leg=inputs.leg, direction=inputs.direction,
            epoch=inputs.epoch, free_mib=inputs.free_mib, log=log,
            sum_bytes=sum_bytes, classes=classes, per_leg=per_leg,
            resume_reserve_bytes=inputs.resume_reserve_bytes,
            explicit=explicit, slot_bytes=slot_bytes,
            stripe_bytes=stripe_bytes, budget_s=budget_s)
        leg.run = run
        result = run.result
        result.ring_ms = inputs.ring_ms
        if not is_source:
            # THE COMPARE IS THE DESTINATION HOOK'S WHOLE POINT and it runs
            # here, after the ring's own restore, against the ring's own bytes.
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
        leg.close(owner=leg.token)
        result.shadow_ms = (time.perf_counter() - started) * 1e3
        result.hop_bound_ms = bound
        result.sems_armed = leg.sems_armed
        result.hook = str(inputs.hook)
        result.resume_reserve_mib = int(
            -(-int(inputs.resume_reserve_bytes) // MIB))
        log(result.line())
