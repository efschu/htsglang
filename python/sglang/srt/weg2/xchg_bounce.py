"""#1332 (S6 slice 1, steps 1-2): the BOUNCE TERM -- how much host RAM the
weight exchange may hold, and the refusals when it cannot be covered.

USER LAW, 2026-09-11, verbatim: *"notfalls wird das layer auf einem
(vertretbar kleinen) hostpuffer vollstaendig zusammengesetzt und jede karte
nimmt sich von dem was er braucht (oder ihn komplett). die 27gb (oder so
aehnlich) an layerbytes muessen nicht mehr dauerhaft im systemram gehalten
werden. das ist das ziel."*  Design note: WEG2_REUSE_SPEC_0908.md section 10.

WHAT THIS MODULE IS. One pure function that DERIVES the bounce term from the
boot's own figures, plus the refusals that keep it a bound rather than a wish.
It allocates nothing, opens no device, reads no environment: the ledger charges
its answer at both moments (`charge_terms(..., xchg_bounce_host_bytes=)`) and
the launcher refuses before either group starts.

WHAT IT REPLACES. `Sigma H`, the host weights term, measured 42.96 GiB on boot
weg2xsn8 (`f03e405e7d`): a region preallocated at Sigma H that the legs copy
THROUGH, charged once and never shrinking for the life of the boot. That is the
permanent residency the law forbids.

NEVER PINNED, and this is the part a reader must be able to check. Every term
below is read from the boot; the module ships no rig constant. The figures in
the doctests are THIS rig's instance of the expression
(`oncard 10.28 + cross 16.84 = 27.12 GiB` over 64 layers at
`--pp-stage-ratio 32,18,14`, 8 layers/chunk x 8 chunks), not defaults:

    buffer   = ceil(bytes_per_direction / n_layers) * depth
    staging  = pairs * slots_per_pair * slot_bytes
    bounce   = buffer + staging

    433.9 MiB * 2  +  6 * 2 * 64 MiB  =  867.8 + 768  =  1.60 GiB
    against Sigma H 42.96 GiB -> 41.37 GiB released, 26.9x

THE BOUND IS THE WIDEST LAYER, NEVER THE MEAN (section 10.5). `bytes_per_
direction / n_layers` is a MEAN and it is the right term for SIZING the steady
stream; it is the wrong term for the REFUSAL, because a boot sized on the mean
dies on whichever layer is above it. Both numbers are therefore carried and the
refusal grades against `widest_layer_bytes`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

MIB = 1024 * 1024
GIB = 1024 * MIB

#: #1277 measured 64 MiB DOUBLE-BUFFERED at 99 % of the link ceiling. E2's
#: earlier "32/64/128 indistinguishable, <0.6 %" was measured on the
#: SINGLE-buffered form and does not carry to this lane -- quoting it here
#: would be an instrument reused outside the arm it was measured on.
SLOT_BYTES_DEFAULT = 64 * MIB

#: Slots per directed pair. Two is the double buffer itself: D2H fills slot k
#: while H2D drains slot k-1.
SLOTS_PER_PAIR = 2

#: Layers in flight in the ASSEMBLE buffer. Two, and not three: depth 2 is what
#: buys the assemble/copy-out overlap (section 10.4), and a third layer buys
#: nothing once the x4 link is saturated while costing another whole layer of
#: host RAM. Raising it must be PRICED, never assumed.
ASSEMBLE_DEPTH_DEFAULT = 2


class Weg2XchgBounceUnderCovered(RuntimeError):
    """W71-form: the buffer cannot hold this boot's widest layer.

    Raised at ARM time, before either group starts, for the reason W71 itself
    exists: a refusal that arrives mid-flip arrives after VRAM has already
    been mutated. It reuses W71's code and shape rather than taking a new
    number -- the refusal is "the residency this arm needs is not arm-able",
    which is what W71 already names.
    """


@dataclass(frozen=True)
class BounceTerms:
    """The bounce term and every input it was derived from.

    Frozen because it is a statement about one boot's figures, and because the
    ARM line prints it: a term that could be mutated after it was charged is a
    term the log cannot vouch for.
    """

    bytes_per_direction: int
    n_layers: int
    widest_layer_bytes: int
    depth: int
    pairs: int
    slot_bytes: int
    mean_layer_bytes: int
    buffer_bytes: int
    staging_bytes: int

    @property
    def total_bytes(self) -> int:
        """The one number the ledger charges as ``xchg_bounce_host_bytes``."""
        return int(self.buffer_bytes) + int(self.staging_bytes)

    @property
    def covers_widest_layer(self) -> bool:
        """Can ONE depth-slot of the buffer hold the widest layer?

        The per-slot width is what an assembly writes into, so the comparison
        is against `buffer_bytes / depth` and not against the whole buffer:
        two layers in flight do not make one layer fit.
        """
        if self.depth <= 0:
            return False
        return (int(self.buffer_bytes) // int(self.depth)) >= int(
            self.widest_layer_bytes
        )

    def expression(self) -> str:
        """The derivation, printed so the total is checkable rather than taken.

        Section 10.2's rule: the arming line must print the EXPRESSION's terms,
        not just the total, because the total alone is indistinguishable from a
        pinned constant.
        """
        return (
            f"buffer={self.buffer_bytes // MIB} MiB "
            f"(widest_layer={self.widest_layer_bytes // MIB} MiB x depth={self.depth}; "
            f"mean_layer={self.mean_layer_bytes // MIB} MiB over n_layers={self.n_layers} "
            f"of bytes_per_direction={self.bytes_per_direction // MIB} MiB) "
            f"+ staging={self.staging_bytes // MIB} MiB "
            f"(pairs={self.pairs} x slots={SLOTS_PER_PAIR} x slot={self.slot_bytes // MIB} MiB) "
            f"= bounce_total={self.total_bytes // MIB} MiB"
        )


def bounce_terms(
    *,
    bytes_per_direction: int,
    n_layers: int,
    widest_layer_bytes: int,
    pairs: int,
    depth: int = ASSEMBLE_DEPTH_DEFAULT,
    slot_bytes: int = SLOT_BYTES_DEFAULT,
) -> BounceTerms:
    """Derive the bounce term. Pure; raises only on inputs that cannot mean anything.

    THE BUFFER IS SIZED ON THE WIDEST LAYER, not the mean. The mean is carried
    for the log (it is what the steady stream costs) but a buffer sized on it
    fails on the first layer above it -- section 10.5, and the danger direction
    of this whole slice.
    """
    if int(n_layers) <= 0:
        raise ValueError(
            f"n_layers must be positive: {n_layers!r}. A boot whose layer "
            "count could not be read cannot be sized, and sizing it against a "
            "default is how a pinned constant re-enters through the back door"
        )
    if int(bytes_per_direction) <= 0:
        raise ValueError(
            f"bytes_per_direction must be positive: {bytes_per_direction!r} "
            "(the plan's own oncard + cross halves)"
        )
    if int(widest_layer_bytes) <= 0:
        raise ValueError(
            f"widest_layer_bytes must be positive: {widest_layer_bytes!r}. "
            "Absent is NOT 'use the mean': the refusal grades against this "
            "number and a missing one would silently become a mean-sized bound"
        )
    if int(depth) <= 0:
        raise ValueError(f"depth must be positive: {depth!r}")
    if int(pairs) < 0 or int(slot_bytes) <= 0:
        raise ValueError(f"pairs/slot_bytes invalid: {pairs!r}/{slot_bytes!r}")
    mean_layer = -(-int(bytes_per_direction) // int(n_layers))  # ceil
    return BounceTerms(
        bytes_per_direction=int(bytes_per_direction),
        n_layers=int(n_layers),
        widest_layer_bytes=int(widest_layer_bytes),
        depth=int(depth),
        pairs=int(pairs),
        slot_bytes=int(slot_bytes),
        mean_layer_bytes=mean_layer,
        buffer_bytes=int(widest_layer_bytes) * int(depth),
        staging_bytes=int(pairs) * SLOTS_PER_PAIR * int(slot_bytes),
    )


def under_coverage_refusal(terms: BounceTerms, *, widest_layer_name: str = "") -> str:
    """The W71-form message for a buffer that cannot hold the widest layer."""
    per_slot = int(terms.buffer_bytes) // max(1, int(terms.depth))
    return (
        f"W71 Weg2XchgResidencyUnarmable (bounce under-coverage): the assemble "
        f"buffer holds {per_slot // MIB} MiB per depth-slot and this boot's "
        f"WIDEST layer"
        + (f" ({widest_layer_name})" if widest_layer_name else "")
        + f" is {terms.widest_layer_bytes // MIB} MiB. Refused at ARM time, "
        f"before either group starts: a layer that cannot be assembled whole "
        f"cannot be sliced by its destinations, and discovering that mid-flip "
        f"is discovering it after VRAM has been mutated. The bound is the "
        f"WIDEST layer and never the mean ({terms.mean_layer_bytes // MIB} "
        f"MiB) -- a boot sized on the mean dies on whichever layer is above "
        f"it. {terms.expression()}"
    )


def arm_line(terms: BounceTerms, *, sigma_h_bytes: int = 0) -> str:
    """The acceptance line of section 10.6.

    ``sigma_h_mib`` is printed BESIDE the bounce and never folded into it: the
    slice's whole acceptance is that the first goes to 0 while the second
    stands in for it, and one number carrying both would make that
    unobservable.
    """
    released = max(0, int(sigma_h_bytes) - terms.total_bytes)
    return (
        f"WEG2-XCHG-BOUNCE layers={terms.n_layers} "
        f"widest_layer_mib={terms.widest_layer_bytes // MIB} "
        f"buffer_mib={terms.buffer_bytes // MIB} depth={terms.depth} "
        f"path_a_staging_mib={terms.staging_bytes // MIB} "
        f"slot_mib={terms.slot_bytes // MIB} "
        f"bounce_total_mib={terms.total_bytes // MIB} "
        f"sigma_h_mib={int(sigma_h_bytes) // MIB} "
        f"released_gib={released / GIB:.2f} "
        f"covers_widest={'yes' if terms.covers_widest_layer else 'NO'} "
        f"-- {terms.expression()}"
    )
