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
    def staging_per_card(self) -> int:
        """This term's staging, per card -- what ONE rank may deposit.

        ``staging_bytes // pairs`` expressed through the shared arithmetic
        rather than by dividing, so a term built with ``pairs=0`` (a caller
        pricing a boot without path (a)) answers 0 instead of raising.
        """
        return staging_bytes_per_card(self.slot_bytes) if self.pairs else 0

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


#: ONE AUTHORITY FOR THE ASSEMBLE BUFFER'S SIZE -- the PRICE and the
#: ALLOCATION now call the same two functions, because they did not, and the
#: gap was measurable to the byte:
#:
#:   PRICED     xchg_bounce.py:189  widest_layer_bytes * depth
#:              = 756323776 * 2 = 1512647552 B (1442 MiB)
#:   ALLOCATED  weight_exchange_bounce.py:1119 slots = depth + (1 if comparing)
#:              weight_exchange_bounce.py:502  nbytes = slot_bytes * depth
#:              = 756323776 * 3 = 2268971328 B (2163.9 MiB)
#:
#: -- exactly ONE widest layer (756323776 B, 721.4 MiB) unpriced whenever
#: `mode=shadow`, i.e. on every S6I boot, since S6I boots shadow first. Boot
#: weg2xsn25 printed the allocated number verbatim:
#: `WEG2-XCHG-HOST-SLOT ... event=alloc bytes=2268971328`.
#:
#: NOT NAMED `bounce_bytes`. That name belonged to
#: `host_ledger.xchg_bounce_bytes`/`...bytes_per_card`, which plan AMENDMENT 5
#: DELETED as a second producer (launcher.py:4929 still names it as such).
#: Reviving a retired name for the function that exists to end second
#: bookkeeping would be the joke telling itself.
#:
#: THE ARM IS READ HERE, not threaded through the launcher: `inject_mode()` is
#: "THE one reader" (weight_exchange.py:2145), so a caller that passed its own
#: opinion could disagree with it -- which is the very shape being closed. The
#: parameter stays available for tests and for a caller that genuinely knows
#: better, and defaults to the arm.


def assemble_slots(depth: int, *, comparing: Optional[bool] = None) -> int:
    """Depth-slots the assemble buffer holds, INCLUDING the shadow's extra one.

    The compare needs the live bytes beside the staged ones, so a shadow leg
    holds one slot more than its depth. That was already true in the allocator
    and false in the price.
    """
    if comparing is None:
        from sglang.srt.weg2 import weight_exchange as wx

        comparing = wx.inject_mode() == wx.INJECT_SHADOW
    return int(depth) + (1 if comparing else 0)


def assemble_buffer_bytes(widest_layer_bytes: int, depth: int, *,
                          comparing: Optional[bool] = None) -> int:
    """The assemble buffer, in bytes.  The ONLY expression for that number."""
    return int(widest_layer_bytes) * assemble_slots(int(depth),
                                                    comparing=comparing)


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
        buffer_bytes=assemble_buffer_bytes(widest_layer_bytes, depth),
        staging_bytes=int(pairs) * SLOTS_PER_PAIR * int(slot_bytes),
    )


#: The channel the launcher's PRICED term reaches a rank on.  A rank may not
#: re-derive the term: the census is a checkpoint read and a second reader of
#: it would be a second sizing authority (the defect AMENDMENT 5 just retired
#: one instance of).  The launcher publishes the scalars it priced WITH, the
#: rank rebuilds the term through `bounce_terms` -- the same one function -- so
#: the two objects are identical by construction rather than by agreement.
ENV_BOUNCE_TERMS = "SGLANG_WEG2_XCHG_BOUNCE_TERMS"

_TERM_FIELDS = ("bytes_per_direction", "n_layers", "widest_layer_bytes",
                "pairs", "depth", "slot_bytes")


def publish_terms(terms: BounceTerms) -> str:
    """The launcher's side: the term's INPUTS as one compact value.

    The inputs and not the outputs, deliberately.  Publishing
    `buffer_bytes`/`staging_bytes` would let a rank hold a total whose
    derivation it cannot check, and a total is exactly the thing that reads
    right while its factors drift (measured: two 384 MiB staging figures from
    different factorisations).  Publishing the inputs makes the rank recompute
    through `bounce_terms`, so a disagreement is impossible instead of merely
    unlikely.
    """
    return ",".join(f"{f}={int(getattr(terms, f))}" for f in _TERM_FIELDS)


def read_published_terms(raw: Optional[str] = None) -> Optional[BounceTerms]:
    """The rank's side: the launcher's term, rebuilt, or ``None``.

    ``None`` means NOT PUBLISHED -- an unarmed boot, or a launcher that did not
    price a bounce -- and the caller must treat that as "no authority to
    inject", never as a zero-sized buffer.  A MALFORMED value raises rather
    than returning None: an env var that exists and cannot be parsed is a
    launcher defect, and swallowing it would inject against a size nobody
    priced.
    """
    import os

    text = (os.environ.get(ENV_BOUNCE_TERMS, "") if raw is None else raw)
    text = (text or "").strip()
    if not text:
        return None
    kw = {}
    for part in text.split(","):
        key, _, value = part.partition("=")
        key = key.strip()
        if key not in _TERM_FIELDS:
            raise ValueError(
                f"{ENV_BOUNCE_TERMS} carries an unknown field {key!r}; the "
                f"published term is the launcher's own and its fields are "
                f"{_TERM_FIELDS}")
        kw[key] = int(value)
    missing = [f for f in _TERM_FIELDS if f not in kw]
    if missing:
        raise ValueError(
            f"{ENV_BOUNCE_TERMS} is missing {missing}; a partially published "
            f"term would be sized against defaults, which is how a pinned "
            f"constant re-enters through the back door")
    return bounce_terms(**kw)


def staging_bytes_per_card(slot_bytes: int) -> int:
    """One card's share of path (a)'s staging: ``SLOTS_PER_PAIR x slot_bytes``.

    THE ONE ARITHMETIC, so the launcher's charge and a rank's runtime budget
    cannot be two copies of it (AMENDMENT 5 consequence 2).
    ``bounce_terms`` multiplies this by ``pairs``; a rank enforcing its own
    deposit multiplies it by nothing.  Before AMENDMENT 5 the runtime read
    ``host_ledger.xchg_bounce_bytes_per_card`` -- the S6b deposit's CEILING
    (``ONCARD_SLOTS_MAX 8 x ONCARD_SLOT_BYTES_MAX``) -- while the ledger
    charged this, so the two disagreed by 4x on the same payload.  That
    ceiling is retired by design: the diagonal store-and-forward IS path (a)'s
    staging for the co-located pairs.

    THE DEPOSIT IS NEVER SIZED UP, AND THIS TERM IS THE LEVER THAT SAYS SO --
    ``DEPOSIT_REASON_UNFUNDED``, not ``DEPOSIT_REASON_BATCHES`` (#1333).  The
    sentence standing here until then named the wrong refusal, and the
    difference is not cosmetic because the two words point at different knobs.
    ``weight_exchange_transport.deposit_refusal_reason`` grades ``batches``
    against ``slots_max``, whose default is ``ONCARD_SLOTS_MAX`` (8) and which
    the ONE production caller (``weight_exchange_shadow``'s leg planner) does
    not pass -- that is the transport's ROW AREA, sized once for both processes.
    A plan needing three batches at the published slot therefore fails on THIS
    number: ``slots x slot_bytes > budget_bytes``.  Sending a reader to
    ``ONCARD_SLOTS_MAX`` would send them to a knob that is not binding here.
    Measured at the desk (#1333): the only test that read ``BATCHES`` for a
    3-batch plan passed ``slots_max=SLOTS_PER_PAIR`` explicitly, a value no
    product call site supplies.

    ``slot_bytes`` is the LAUNCHER'S PUBLISHED VALUE
    (``--weg2-xchg-oncard-slot-mib``, reaching a rank as
    ``weight_exchange_transport.ENV_ONCARD_SLOT_MIB``), never a module
    default: one value, both sides.
    """
    return SLOTS_PER_PAIR * int(slot_bytes)


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
