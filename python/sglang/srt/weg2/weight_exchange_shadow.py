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
5. it may not raise into the leg: :func:`shadow_leg` catches ``BaseException``
   and turns it into a logged line;
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
    subset (``--weg2-shadow-classes``); the automatic path logs the same
    message and shadows nothing on that card, because a shadow that refuses a
    flip has taken the authority this slice exists not to have.  Same shape as
    W56's two arms in S4: logged when the degrade is automatic, raised when the
    operator asked for the thing that is unavailable.
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

    def line(self, *, leg: int, epoch: str) -> str:
        return (
            f"{SHADOW_GATE_LINE_PREFIX} leg={leg} epoch={epoch} "
            f"joined={self.joined}/{xr.N_RANKS} "
            f"run={'yes' if self.run else 'no'} "
            f"refusers={','.join(str(r) for r in self.refusers) or 'none'} "
            f"waited_s={self.waited_s:.3f} reason={self.reason}"
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
            return ShadowVerdict(True, len(fresh), (), "all six joined",
                                 monotonic() - started)
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


@dataclass(frozen=True)
class ShadowPrice:
    card: str
    need_mib: int
    free_mib: int
    floor_mib: float
    scratch_mib: int

    @property
    def affordable(self) -> bool:
        return self.free_mib - self.need_mib >= self.floor_mib

    def message(self) -> str:
        return (
            f"{UNAFFORDABLE_MARKER} card={self.card} need_mib={self.need_mib} "
            f"(dst_buffers={self.need_mib - self.scratch_mib} + "
            f"scratch={self.scratch_mib}) free_mib={self.free_mib} "
            f"floor_mib={self.floor_mib:g} "
            f"left_mib={self.free_mib - self.need_mib} -- the shadow does not "
            f"run on this card for this leg; the flip is untouched.  "
            f"{SB5F_FREE_MIB_NOTE}"
        )

    def line(self) -> str:
        return (
            f"{SHADOW_BUDGET_LINE_PREFIX} card={self.card} "
            f"need_mib={self.need_mib} free_mib={self.free_mib} "
            f"floor_mib={self.floor_mib:g} "
            f"verdict={'AFFORDABLE' if self.affordable else 'REFUSED'}"
        )


def price_shadow(card: str, need_bytes: int, free_mib: int, *,
                 floor_mib: float = SHADOW_FLOOR_MIB,
                 scratch_bytes: int = STRIPE_BYTES) -> ShadowPrice:
    """Can this card carry the shadow's buffers on top of what it is doing?

    ``need`` is the destination buffers (raw ``cudaMalloc``, OUTSIDE every TMS
    region, so the saver never pauses them and they never enter its census)
    plus ONE stripe-sized scratch for the device checksum -- the spec's own
    "VRAM delta = 64 MiB per consumer rank".

    ``free_mib`` is the card's LIVE NVML free column, read by the caller.  Not
    a census, not a projection: R6 (the allocator cache) is unbounded in
    principle and this is an instrument that must not OOM the boot it observes,
    so the number it prices against is the one the driver reports right now.
    """
    need = int(-(-int(need_bytes) // MIB)) + int(-(-int(scratch_bytes) // MIB))
    return ShadowPrice(str(card), need, int(free_mib), float(floor_mib),
                       int(-(-int(scratch_bytes) // MIB)))


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


def classify(stripe: Stripe) -> str:
    """MATCH / MISMATCH / NOT-REPRESENTABLE -- in that order of questions.

    SPEC SECTION 3.3 RULE 3, and it is not a formality.  ``uint8_checksum`` of
    ``n`` bytes lies in ``[0, 255n]`` and nowhere else, so a value outside that
    range was never a checksum of this payload -- the field was read at the
    wrong offset, or the payload was framed differently at the two ends.  #656
    register C22 reported exactly that as a data corruption and an instance was
    killed for a corruption that had not happened.

    The import is LAZY because ``weights_arena`` imports torch and this module
    is exercised at the desk without it; the function is IMPORTED, never
    re-derived (rule 3 says so in as many words).
    """
    from sglang.srt.model_executor.weights_arena import checksum_is_representable

    for value in (stripe.shadow_sum, stripe.ring_sum):
        if not checksum_is_representable(value, stripe.nbytes):
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


def torch_device_summer(ops: tp.DeviceOps, device: int, stream: int,
                        scratch_ptr: int, scratch_bytes: int = STRIPE_BYTES):
    """``sum_bytes(addr, nbytes)`` over DEVICE memory, on the device.

    The range is copied D2D into a torch-owned uint8 scratch and summed by
    :func:`~sglang.srt.model_executor.weights_arena.uint8_checksum`, which is
    imported and never re-derived (spec 3.3 rule 3).  No host round trip of the
    payload: the only thing that crosses is the scalar, at the one sync
    ``uint8_checksum`` already does.

    The scratch is a torch tensor because that is what makes the sum a DEVICE
    sum; the shadow's destination BUFFERS are raw ``cudaMalloc`` instead,
    deliberately, so they sit outside every TMS region and can never be paused,
    unmapped or counted into the saver's census.  Two allocators, two reasons,
    both stated.

    OPEN ITEM, named rather than hidden: a 2-D class contributes ~6 KiB runs,
    so this issues one small D2D per run.  It is an observer, bounded by the
    class subset, and the cost lands in ``compare_ms`` on the line -- but it is
    not free and it is not the shape S6 would use for an authoritative check.
    """
    import torch

    scratch = torch.empty(0)  # placeholder; bound below

    def sum_bytes(addr: int, nbytes: int) -> int:
        from sglang.srt.model_executor.weights_arena import uint8_checksum

        n = int(nbytes)
        if n <= 0:
            return 0
        if n > int(scratch_bytes):
            raise ValueError(
                f"a stripe of {n} bytes does not fit the {int(scratch_bytes)}-byte "
                f"shadow scratch -- the scratch IS the stripe size (spec 6/S5's "
                f"64 MiB per consumer rank), so these two numbers may not drift")
        ops.memcpy_async(scratch_ptr, int(addr), n, stream)
        ops.synchronize(stream)
        return int(uint8_checksum(scratch[:n]))

    def bind(tensor) -> None:
        nonlocal scratch
        scratch = tensor

    sum_bytes.bind_scratch = bind  # type: ignore[attr-defined]
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
    errors: List[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        if self.mismatch or self.not_representable:
            return MISMATCH
        return MATCH


@dataclass
class ShadowResult:
    leg: int
    epoch: str
    subset: ShadowSubset
    counters: ShadowCounters
    oncard_ms: float = 0.0
    cross_ms: float = 0.0
    ring_ms: float = 0.0
    compare_ms: float = 0.0
    issue_ms: float = 0.0
    slot_wait_ms: float = 0.0
    gate_skew_ms: float = 0.0
    lock_wait_ms: float = 0.0
    pieces: int = 0
    direction: str = "?"
    ran: bool = False
    reason: str = ""

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
        about ``_split_decisions``, not about the exchange.
        """
        counters = self.counters
        return (
            f"{SHADOW_LINE_PREFIX} leg={self.leg} epoch={self.epoch} "
            f"dir={self.direction} ran={'yes' if self.ran else 'no'} "
            f"classes={len(self.subset.classes)} "
            f"subset={','.join(self.subset.classes) or 'none'} "
            f"rotation={len(self.subset.rotation)} "
            f"pieces={self.pieces} stripes={counters.stripes} "
            f"match={counters.match} mismatch={counters.mismatch} "
            f"not_representable={counters.not_representable} "
            f"slot_checksums={counters.checksum_reports} "
            f"slot_checksum_mismatch={counters.checksum_mismatch} "
            f"oncard_ms={self.oncard_ms:.3f} cross_ms={self.cross_ms:.3f} "
            f"ring_ms={self.ring_ms:.3f} compare_ms={self.compare_ms:.3f} "
            f"xchg_ms={self.oncard_ms + self.cross_ms:.3f} "
            f"issue_ms={self.issue_ms:.3f} slot_wait_ms={self.slot_wait_ms:.3f} "
            f"gate_skew_ms={self.gate_skew_ms:.3f} "
            f"lock_wait_ms={self.lock_wait_ms:.3f} "
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
    """

    def __init__(self, ops: tp.DeviceOps, device: int, nbytes: int,
                 scratch_bytes: int = STRIPE_BYTES):
        self.ops = ops
        self.nbytes = int(nbytes)
        self.scratch_bytes = int(scratch_bytes)
        self.ptr = ops.raw_malloc(int(device), self.nbytes) if self.nbytes else 0
        self.scratch = (ops.raw_malloc(int(device), self.scratch_bytes)
                        if self.scratch_bytes else 0)

    def close(self) -> None:
        for ptr in (self.ptr, self.scratch):
            if ptr:
                try:
                    self.ops.raw_free(ptr)
                except Exception:  # noqa: BLE001 -- an observer's unwind
                    pass
        self.ptr = self.scratch = 0


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

    EVERY FAILURE PATH ENDS IN A LOG LINE AND A RETURNED RESULT.  There is no
    exception that leaves this function, including the ones the transport
    raises by design (W52, W53, W54): on the authoritative path those are
    refusals that stop a flip, and here the same event means "the shadow did
    not get its measurement".  ``vote_failure`` is a RECORDER, not the
    transport's group vote -- passing the real one would let an observer's
    thread take six ranks down.
    """
    subset = select_subset(descs, leg=leg, classes=classes, per_leg=per_leg)
    result = ShadowResult(leg=int(leg), epoch=str(epoch), subset=subset,
                          counters=ShadowCounters(), direction=str(direction))
    run = ShadowRun(result, None, (), {}, stripe_bytes=stripe_bytes)
    try:
        mine = [d for d in subset.descs
                if int(d.dst_rank if not is_source else d.src_rank) == int(rank)]
        need_bytes = 0 if is_source else shadow_layout(
            [d for d in subset.descs if int(d.dst_rank) == int(rank)])[1]
        price = price_shadow(card_uuid, need_bytes, free_mib,
                             floor_mib=floor_mib,
                             scratch_bytes=0 if is_source else stripe_bytes)
        log(price.line())
        if not price.affordable:
            log(price.message())
            if explicit:
                raise Weg2XchgShadowUnaffordable(price.message())
        verdict = shadow_gate(region, row, leg=leg, vote=price.affordable,
                              classes_hash=subset.hash, need_mib=price.need_mib,
                              log=log)
        log(verdict.line(leg=leg, epoch=epoch))
        result.gate_skew_ms = verdict.waited_s * 1e3
        if not verdict.run:
            result.reason = verdict.reason.split(":")[0].replace(" ", "-")
            return run
        layout, total = shadow_layout([d for d in subset.descs
                                       if int(d.dst_rank) == int(rank)])
        buffers = None
        leg_descs: Sequence[object] = subset.descs
        if not is_source:
            buffers = ShadowBuffers(ops, device, total,
                                    scratch_bytes=stripe_bytes)
            mine_dst = [d for d in subset.descs if int(d.dst_rank) == int(rank)]
            shadowed = to_shadow(mine_dst, layout, buffers.ptr)
            layout = {id(new): layout[id(old)]
                      for old, new in zip(mine_dst, shadowed)}
            keep = {id(d) for d in mine_dst}
            leg_descs = [d for d in subset.descs if id(d) not in keep] + shadowed
            run = ShadowRun(result, buffers, shadowed, layout,
                            stripe_bytes=stripe_bytes)
        counters = result.counters

        def on_checksum(report: tp.ChecksumReport) -> None:
            counters.checksum_reports += 1
            if not report.match:
                counters.checksum_mismatch += 1
                log(f"{MISMATCH_MARKER} lane={report.lane} pair={report.pair} "
                    f"slot={report.slot} seq={report.seq} "
                    f"nbytes={report.nbytes} producer_checksum={report.expected} "
                    f"consumer_checksum={report.got} leg={leg} epoch={epoch} "
                    f"-- the STAGED bytes changed between publish and read; "
                    f"counted, never acted on")

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
            # PRICED FROM THE SUBSET'S OWN DIAGONAL, not inherited: the shadow
            # moves a class subset, so its batch count -- and therefore the
            # hop this leg pays -- is a different number from the full leg's.
            # Passing the flip's slot size here would price a lane that is not
            # the one running.
            oncard_slot_bytes=(
                tp.plan_oncard_slot_bytes(
                    sum(int(d.nbytes) for d in leg_descs if _is_on_card(d))
                ).slot_bytes if oncard_slot_bytes is None else oncard_slot_bytes),
            oncard_slots=oncard_slots)
        elapsed_ms = (time.perf_counter() - started) * 1e3
        result.cross_ms = sum(p.elapsed_s for p in out.pairs) * 1e3
        result.oncard_ms = (out.oncard.elapsed_s * 1e3) if out.oncard else 0.0
        result.slot_wait_ms = sum(p.slot_wait_s for p in out.pairs) * 1e3
        result.issue_ms = max(0.0, elapsed_ms - result.cross_ms - result.oncard_ms)
        result.pieces = sum(p.pieces for p in out.pairs) + len(
            [d for d in leg_descs if _is_on_card(d)])
        result.ran = True
        _ = mine
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
