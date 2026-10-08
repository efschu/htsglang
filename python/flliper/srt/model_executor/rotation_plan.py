# SPDX-License-Identifier: Apache-2.0
"""#809/W28: chunk-rotation residency for the flip images.

THE DESIGN (user's, 2026-08-24). Never hold BOTH layouts in system RAM. RAM
holds exactly ONE layout image -- the INACTIVE one -- plus a small OVERSHOOT.
At the flip:

    stream the new layout chunk-wise RAM -> VRAM      (H2D)
    concurrently stream the old layout's vacated VRAM
    chunks -> RAM, into the pages the H2D just freed  (D2H)

PCIe is full duplex, so the copy-back rides the idle return direction. After
the cutover RAM again holds exactly the now-inactive layout, primed for the
next flip.

THE COPY-BACK IS NOT WRITE-BACK. The weights are immutable and nothing is
being saved; it is residency PLACEMENT for the next flip, which is what a
single-layout RAM budget requires. A reader who mistakes it for a write-back
will "optimise" it away and break the following flip.

WHY THIS AND NOT A PARTIAL PIN. W26 proved the dual pin impossible here: both
pin arms were OOM-killed in the LAUNCH phase, before any flip. Holding one
layout plus eps (~30 GiB against ~68.7 GiB) fits, and it removes the disk from
the steady-state critical path entirely -- which is what reaches the physics
floor, where a partial pin by construction leaves a disk share behind and W26
measured the leg 99.8-100 % storage-bound.

THIS MODULE IS THE ARITHMETIC ONLY, deliberately: the overshoot sizing and the
interleaved schedule, as pure functions over byte counts. Every invariant the
scheme rests on is therefore falsifiable without a GPU -- the same split #852's
estimator and #856(a)'s bound phrase use, and for the same reason: a rule that
can only be exercised on metal is one this corpus has repeatedly shipped
inert.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Tuple


class RotationPlanError(ValueError):
    """A rotation that cannot be scheduled within its RAM budget."""


def rotation_overshoot_bytes(
    pp_image_bytes: int, tp_image_bytes: int, chunk_bytes: int, depth: int
) -> int:
    """The RAM headroom a rank needs beyond ONE layout image.

    TWO TERMS, and both are load-bearing:

    * **the size asymmetry.** The two layouts differ per rank -- W26 measured
      PP0 15925.8 vs 16362.7 MiB, PP1 8573.8 vs 8961.3, PP2 8573.8 vs 9481.6.
      When the layout being COPIED BACK is larger than the one being streamed
      out, RAM must hold the difference on top. Sized from the LARGER
      direction, because the overshoot is a single fixed reservation that has
      to cover whichever direction the next flip takes; sizing it from a mean
      is the OOM.
    * **the in-flight window.** `depth` chunks are outstanding at once, so the
      copy-back can be that far ahead of the pages the H2D has freed.

    Refuses rather than guesses on a non-positive chunk or depth: a schedule
    built on either is not a schedule, and returning a plausible number would
    hide that.
    """
    if chunk_bytes <= 0:
        raise RotationPlanError(f"chunk_bytes must be positive, got {chunk_bytes}")
    if depth <= 0:
        raise RotationPlanError(f"depth must be positive, got {depth}")
    if pp_image_bytes < 0 or tp_image_bytes < 0:
        raise RotationPlanError("image sizes must be non-negative")
    asymmetry = abs(int(pp_image_bytes) - int(tp_image_bytes))
    return asymmetry + int(depth) * int(chunk_bytes)


@dataclass(frozen=True)
class RotationStep:
    """One interleaved step: push a chunk out, pull a chunk back.

    Either side may be absent (``length == 0``) when its stream is exhausted --
    the two layouts differ in size, so the tails never line up.
    """

    h2d_offset: int
    h2d_len: int
    d2h_offset: int
    d2h_len: int

    @property
    def overlaps(self) -> bool:
        return self.h2d_len > 0 and self.d2h_len > 0


def plan_rotation(
    *,
    incoming_bytes: int,
    outgoing_bytes: int,
    chunk_bytes: int,
    overshoot_bytes: int,
) -> List[RotationStep]:
    """Interleave the H2D of the incoming layout with the D2H of the outgoing.

    ``incoming_bytes`` is the layout currently in RAM and about to enter VRAM.
    ``outgoing_bytes`` is the layout currently in VRAM and about to be placed
    back into the RAM the incoming one vacates.

    THE INVARIANT THIS ENFORCES, and it is the whole point of the scheme:

        bytes_copied_back - bytes_streamed_out  <=  overshoot_bytes

    i.e. the copy-back may never run further ahead of the freed pages than the
    overshoot allows. Violating it is how "one layout + eps" silently becomes
    "two layouts" and meets the OOM killer that took both of W26's pin arms.

    The schedule is emitted, not executed: the caller owns the streams. What is
    fixed here is the ORDER and the bound, which is what can be checked without
    a device.
    """
    if chunk_bytes <= 0:
        raise RotationPlanError(f"chunk_bytes must be positive, got {chunk_bytes}")
    if overshoot_bytes < 0:
        raise RotationPlanError("overshoot must be non-negative")
    if incoming_bytes < 0 or outgoing_bytes < 0:
        raise RotationPlanError("byte counts must be non-negative")

    steps: List[RotationStep] = []
    out = 0  # bytes streamed RAM -> VRAM (incoming), i.e. RAM pages freed
    back = 0  # bytes copied VRAM -> RAM (outgoing), i.e. RAM pages consumed
    while out < incoming_bytes or back < outgoing_bytes:
        h_len = min(chunk_bytes, max(0, incoming_bytes - out))
        h_off = out
        # The copy-back may only take pages the H2D has already freed, plus the
        # overshoot. `out + h_len` is what will be free once this step's push
        # completes -- the two run concurrently, which is exactly why the
        # overshoot has to cover the in-flight window.
        allowed = out + h_len + overshoot_bytes - back
        d_len = min(chunk_bytes, max(0, outgoing_bytes - back), max(0, allowed))
        d_off = back
        if h_len == 0 and d_len == 0:
            raise RotationPlanError(
                f"rotation stalled at out={out}/{incoming_bytes} "
                f"back={back}/{outgoing_bytes}: the overshoot "
                f"({overshoot_bytes} B) cannot cover the remaining "
                f"copy-back, so RAM would have to hold both layouts"
            )
        steps.append(RotationStep(h_off, h_len, d_off, d_len))
        out += h_len
        back += d_len
    return steps


def peak_ram_bytes(steps: Iterator[RotationStep], resident_bytes: int) -> int:
    """Peak host bytes the schedule holds, given the image already resident.

    Models what the pages actually do: a completed H2D chunk frees its RAM,
    a completed D2H chunk consumes RAM. The peak is what the boot ledger has
    to have reserved, so it is computed from the schedule rather than asserted
    about it.
    """
    held = int(resident_bytes)
    peak = held
    for s in steps:
        # Worst case within a step: the copy-back lands before the push frees
        # its page. Taking the pessimistic order is the point -- the optimistic
        # one would under-report exactly when it matters.
        held += s.d2h_len
        peak = max(peak, held)
        held -= s.h2d_len
    return peak


def rotation_totals(steps: List[RotationStep]) -> Tuple[int, int, int]:
    """(h2d bytes, d2h bytes, overlapped steps) -- the acceptance figures.

    ``overlapped`` is what falsifies a serialized implementation: a scheme
    whose duplex never actually overlaps produces a schedule where the two
    directions do not co-occur, and W28's whole premise is that they do.
    """
    h = sum(s.h2d_len for s in steps)
    d = sum(s.d2h_len for s in steps)
    both = sum(1 for s in steps if s.overlaps)
    return h, d, both
