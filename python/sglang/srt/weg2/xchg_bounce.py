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

#1397 OPTION 3 ADDENDUM (2026-09-14), appended rather than rewritten: #1374
("OPTION 1") sizes `lane_buffer_bytes` from the LARGEST TAG a lane ever pauses
(`tag_slots`/`BounceTerms.lane_slots`), because the pause granularity was the
tag and a tag's deposit had to complete before its own collector could run at
all (weg2xsn30's circular wait, `weight_exchange_bounce.py`'s own module
docstring). That is still true for a DIAGONAL lane (a card talking to
itself) and cannot be relaxed without touching `weg2_memory_saver.pause`
semantics -- out of this module's scope and, on the design doc's own finding,
not needed. It is NOT true for a CROSS lane (a directed card pair): that
lane's collector's VRAM credit gate names the collector's OWN co-located
peer, a rank whose deposit does not wait on anything from this lane, so a
per-BAND drain credit (`weight_exchange_bounce.CrossSlotRendezvous.
wait_band_drained`/`post_band_drained`) can safely let a cross lane reuse a
slot WITHIN one tag instead of sizing the whole tag in. `BounceTerms.
band_credit` + `n_cross_lanes` price that smaller cross share
(`cross_lane_buffer_bytes`) while the diagonal share keeps the #1374 floor,
unconditionally. See `/spinning/gpu-arb/weg2/DESIGN_option3_band_credit_0914.md`
for the full derivation, including why the #1386 `max(buffer_bytes,
lane_buffer_bytes)` trap had to be sidestepped rather than reused for the
cross share.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

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


class Weg2XchgLanesConcurrentInvalid(RuntimeError):
    """W103-form: ``--xchg-lanes-concurrent`` was given a value that cannot
    mean anything (N < 1).

    Refused at ARM time, before either group starts -- the same moment
    :class:`Weg2XchgBounceUnderCovered` (W71) and ``host_ledger``'s
    ``Weg2XchgLanesUnmeasured`` (W102) refuse, and for the same reason: a
    bounce whose lane budget cannot be priced is wrong before a single byte
    moves, never discovered mid-flip.

    NOT FOLDED INTO THE 0-SENTINEL. `BounceTerms.lanes_concurrent` uses 0 to
    mean "the flag was never given" (byte-identical, every lane runs) --
    exactly the convention `max_tag_bytes` already uses. An EXPLICIT 0 or a
    negative number from argv is a different fact (the operator typed
    something that cannot be a lane count) and refusing it here, rather than
    silently reading it as "uncapped", is what keeps
    ``--xchg-lanes-concurrent 0`` from running every lane anyway while the
    argv says the opposite -- the "printed advisory without a reader" shape
    #1256 names.

    W103 IS THE NEXT FREE NUMBER, not a guessed digit: a census of this tree
    (``grep -rhoE 'W[0-9]+' python/ test/``) tops out at W102, and W52/W53/
    W54/W56/W58 are a DOCUMENTED COLLISION CLASS (#1265/#1306) of numbers
    picked from memory landing on codes already spoken for.
    """


def resolve_lanes_concurrent(raw: Optional[int]) -> int:
    """The ONE place ``--xchg-lanes-concurrent`` is validated.

    ``raw`` is the argparse value, and ``None`` -- the flag never given -- is
    the ONLY way to reach the 0 sentinel (:class:`BounceTerms`'s "not
    stated", byte-identical default). Both the ledger's call
    (:func:`bounce_terms` by way of ``choose_host_ledger``) and the ranks'
    publication call (``main``'s ``widest_layer_terms``) resolve through this
    one function, so a flag that would refuse one side and pass the other --
    a boot arming with a number the ranks then reject -- cannot happen; a
    second, slightly different validation is exactly the two-sources defect
    #1358's whole family exists to close.

    An explicit ``0`` or a negative value REFUSES rather than becoming the
    sentinel: collapsing them would let ``--xchg-lanes-concurrent 0`` run
    every lane concurrently anyway, the silent degradation this flag exists
    to replace with a named refusal (#1256).
    """
    if raw is None:
        return 0
    n = int(raw)
    if n < 1:
        raise Weg2XchgLanesConcurrentInvalid(
            f"W103 Weg2XchgLanesConcurrentInvalid: --xchg-lanes-concurrent="
            f"{raw!r} cannot mean anything -- at least one lane must run to "
            f"move any bytes at all. Refused at ARM time, before either "
            f"group starts: folding this into the 'not stated' sentinel "
            f"would run every lane concurrently while the argv says the "
            f"opposite, which is the printed-advisory-without-a-reader "
            f"shape #1256 names, not a smaller boot."
        )
    return n


def resolve_cross_lanes(n_lanes: int, *, max_diag_lanes: Optional[int] = None) -> int:
    """#1397 VERDRAHTUNG (2026-09-14): the SAFE lower bound on how many of
    `n_lanes` are genuinely CROSS pairs, for a caller that arms
    `band_credit` without having measured THIS flip's real split.

    ``max_diag_lanes`` defaults to `weight_exchange_region.N_CARDS`, imported
    LAZILY -- the same reason `assemble_slots`/`tag_slots` (two functions
    up) import `weight_exchange` lazily rather than at module level: this
    file stays free of an unconditional dependency on the rest of weg2, and
    the ONE non-measured topology fact this function needs (a diagonal lane
    is one physical card talking to itself, and there are never more of
    those than there are cards) is read from its OWN owner rather than
    duplicated as a second literal here -- two ``N_CARDS = 3``s is exactly
    the "two sources" defect #1358's whole family exists to close, just for
    a topology constant instead of a measured figure.

    NEVER OVER-COUNTS, BY THE PIGEONHOLE PRINCIPLE, not by an assumption
    about any PARTICULAR flip. A diagonal lane requires `pair_of(r, r) is
    None` (`weight_exchange_bounce.py`), i.e. one lane per physical card --
    so out of ANY `n_lanes` active lanes (`n_lanes` is itself #1358's own
    MEASURED count, `group_descs_by_pair`'s enumeration, never assumed),
    AT MOST `max_diag_lanes` of them can be diagonal, REGARDLESS of which
    specific lanes those are. The remainder, `n_lanes - max_diag_lanes`, is
    therefore a hard LOWER BOUND on the genuinely-cross count for THIS
    flip, not a probabilistic guess: this is what answers DESK10's
    per-rank question (2026-09-14, cross-session) of whether a rank with a
    DIFFERENT specific set of cross pairs than assumed could exceed what
    was priced -- it cannot, because the bound never depended on which
    lanes are cross, only on how many diagonal lanes could possibly exist
    at all.

    A caller that HAS measured the real split (a boot's own
    `group_descs_by_pair` census) may state a larger, more precise
    ``n_cross_lanes`` directly to :func:`bounce_terms` instead of calling
    this at all -- this function is the answer for a caller that has not
    measured it, the exact gap #1397's own wiring left open (see the
    module docstring's Option 3 addendum and
    ``DESIGN_option3_band_credit_0914.md`` section 9).
    """
    if max_diag_lanes is None:
        from sglang.srt.weg2 import weight_exchange_region as xr

        max_diag_lanes = xr.N_CARDS
    return max(0, int(n_lanes) - int(max_diag_lanes))


def lanes_concurrent_line(terms: BounceTerms) -> str:
    """Named and counted, never silent (Wand 11b): the flip-time cost of a cap.

    A lane that does not get its own buffer waits for an earlier lane's
    collector to drain it (`CrossSlotRendezvous.wait_drained`, the per-tag
    handshake #1374 already gives every lane) and then reuses the freed
    buffer -- serial where the uncapped boot ran concurrent, and that costs
    flip time nobody can see on the size line alone. This line puts a number
    beside it: `lanes_concurrent=off` states plainly that no cap was asked
    for (every pre-#1385 boot's own reading), never a blank or an implied 0.
    """
    cap = int(terms.lanes_concurrent)
    return (
        f"WEG2-XCHG-LANES-CONCURRENT "
        f"lanes_concurrent={cap if cap > 0 else 'off'} "
        f"lanes_total={int(terms.n_lanes)} "
        f"lanes_priced={int(terms.lanes_priced)} "
        f"serialised={int(terms.lanes_serialised)}"
    )


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
    #: #1358: HOW MANY ASSEMBLE BUFFERS THIS BOOT ACTUALLY CREATES. One file
    #: per LANE (`bounce_path`: a lane is one directed card pair, or the
    #: diagonal's card), enumerated at runtime by `group_descs_by_pair`.
    #: Default 1 keeps every pre-#1358 caller byte-identical.
    n_lanes: int = 1
    #: #1374 OPTION 1: THE LARGEST TAG THIS BOOT PAUSES, in bytes. The pause
    #: granularity is the tag (`memory_saver_adapter.pause(tag)`), so a tag's
    #: whole deposit has to fit the buffer or the deposit needs a collector
    #: that cannot run until this rank has paused -- boot weg2xsn30's deadlock.
    #: 0 means "not stated", and then the old depth-sized geometry stands and
    #: the leg refuses a band that would wrap instead of overwriting it.
    max_tag_bytes: int = 0
    #: #1385 (Wand 11b, the one host lever named for the cushion floor W98
    #: latches on xsn31/3): THE CAP on how many of this boot's `n_lanes`
    #: assemble buffers may be pinned in tmpfs AT ONCE. 0 means "not stated" --
    #: every lane prices and every pre-#1385 caller is byte-identical, exactly
    #: the convention `max_tag_bytes` already uses two fields above. A value
    #: >= 1 charges `min(n_lanes, lanes_concurrent)` buffers instead of
    #: `n_lanes`; the excess lanes are SERIALISED -- they wait for an earlier
    #: lane's collector to drain (`CrossSlotRendezvous.wait_drained`, the
    #: per-tag handshake #1374 already gives every lane) and then reuse its
    #: buffer -- which is a flip-time cost, never a silent one: see
    #: `lanes_serialised` and `lanes_concurrent_line`.
    lanes_concurrent: int = 0
    #: #1397 OPTION 3 (DESIGN_option3_band_credit_0914.md): whether a CROSS
    #: lane (a directed card pair, never the diagonal) may reuse a slot
    #: WITHIN one tag under a per-BAND drain credit
    #: (`weight_exchange_bounce.CrossSlotRendezvous.wait_band_drained`/
    #: `post_band_drained`) instead of sizing the whole tag into the buffer.
    #: ``False`` keeps every pre-#1397 caller byte-identical: `lane_slots`
    #: is untouched by this field, so this default alone changes nothing.
    #:
    #: THIS CAN NEVER SHRINK A DIAGONAL LANE, and that is not a missing
    #: feature -- it is the finding the design doc is written around. A
    #: diagonal lane's own collector's `resume` is gated (C14) on THIS SAME
    #: rank's not-yet-published VRAM credit (weight_updater.py's pause/
    #: credit loop), which only follows this rank's WHOLE tag deposit; a
    #: wait for that collector's drain, inside this rank's own deposit loop,
    #: BEFORE that credit is published, is the exact circular wait #1374
    #: deleted after boot weg2xsn30. A CROSS lane has no such cycle: its
    #: collector's credit gate names a DIFFERENT rank (the collector's own
    #: co-located peer), whose deposit does not wait on anything from this
    #: lane -- see the design doc's "why cross but never diagonal" section.
    band_credit: bool = False
    #: Of `lanes_priced`, how many are genuinely CROSS pairs (never the
    #: diagonal) and therefore eligible for `band_credit`'s smaller sizing.
    #: A CONCRETE INT ON THIS FIELD ALWAYS -- the RESOLVED value, never the
    #: sentinel. `bounce_terms`'s own `n_cross_lanes` PARAMETER is
    #: `Optional[int] = None` (the identical `raw is None` convention
    #: `resolve_lanes_concurrent` already uses): `None` (never given) alongside
    #: `band_credit=True` auto-derives a SAFE, structural value via
    #: :func:`resolve_cross_lanes` before this field is ever set (#1397
    #: VERDRAHTUNG, 2026-09-14) -- "built but nobody pulls the lever"
    #: (#1367/#1375) is closed for this ONE input by construction, so
    #: arming reduces to the single `band_credit` boolean. An EXPLICIT `0`
    #: is respected literally (this boot truly has no cross lanes -- band_
    #: credit then prices and does nothing, safely) and never promoted. A
    #: caller that states a larger count than `lanes_priced` cannot
    #: inflate the eligible share past 100 % (`n_cross_lanes_priced` clamps
    #: it) -- the direction that matters is UNDER-stating, which only ever
    #: costs bytes back to the diagonal-safe floor, never under-charges.
    n_cross_lanes: int = 0

    @property
    def n_cross_lanes_priced(self) -> int:
        """`n_cross_lanes`, clamped to `lanes_priced` and to `band_credit`.

        Clamped rather than trusted: a caller that mis-states more cross
        lanes than exist would otherwise UNDER-charge `total_bytes` against
        what `run_bounce_leg` actually allocates for the excess (which it
        would price as diagonal, i.e. the bigger number, on the metal).
        """
        if not self.band_credit:
            return 0
        return max(0, min(int(self.n_cross_lanes), int(self.lanes_priced)))

    @property
    def n_diag_lanes_priced(self) -> int:
        """The remainder of `lanes_priced` -- always priced at the whole-tag
        floor, `band_credit` or not."""
        return max(0, int(self.lanes_priced) - int(self.n_cross_lanes_priced))

    @property
    def cross_lane_slots(self) -> int:
        """Option 3's floor for a CROSS lane: `depth`-order, never
        `ceil(max_tag_bytes / slot_bytes)`.

        NEVER read for a diagonal lane -- `run_bounce_leg` only takes this
        branch when its `rendezvous.pair is not None`, i.e. never for the
        diagonal's `card=` form. Reuses `assemble_slots` rather than
        inventing a third depth-to-slots expression, the same "one producer"
        rule `lane_slots` itself follows.
        """
        return assemble_slots(int(self.depth), comparing=None)

    @property
    def cross_lane_buffer_bytes(self) -> int:
        """A CROSS lane's file under Option 3: `cross_lane_slots x slot_bytes`.

        DELIBERATELY NOT maxed against `buffer_bytes` the way
        `lane_buffer_bytes` is charged in `total_bytes`. Under Option 1
        (`max_tag_bytes > 0`, a precondition of `band_credit` ever mattering)
        `run_bounce_leg`/`leg_slot_bytes` never allocate `buffer_bytes` at
        all for ANY Option-1 lane -- `leg_slot_bytes`'s own docstring: "Option
        1 ACTIVE ... never a third number computed here" -- so `buffer_bytes`
        is provably dead weight for a band-credit lane too. Charging it
        anyway would silently re-substitute the bigger, unallocated number
        through the very `max()` `total_bytes` uses for the diagonal share,
        which is the #1386 trap this ticket (#1397) was opened to avoid: a
        printed saving the buffer never actually gets.
        """
        return int(self.cross_lane_slots) * int(self.slot_bytes)

    @property
    def lanes_priced(self) -> int:
        """How many lane buffers THIS TERM actually charges.

        `n_lanes` when the cap is not stated (0) -- byte-identical to every
        caller before #1385 -- otherwise the smaller of the two. Never the
        cap alone: a cap larger than the measured lane count would be a
        knob that can INFLATE the charge past what any boot of this cut ever
        creates, which is the opposite direction from #1358's under-charge
        but the same defect class (a priced number nothing on the boot
        produces).
        """
        cap = int(self.lanes_concurrent)
        return int(self.n_lanes) if cap <= 0 else min(int(self.n_lanes), cap)

    @property
    def lanes_serialised(self) -> int:
        """Lanes that do NOT get their own buffer and must wait for one.

        Named and counted rather than folded into the total: a boot that
        pays this in flip-time has nothing else on the ARM line that would
        tell it why -- `total_bytes` alone reads like a smaller boot, not a
        slower one.
        """
        return max(0, int(self.n_lanes) - int(self.lanes_priced))

    @property
    def lane_slots(self) -> int:
        """Slots ONE lane's buffer holds -- the one producer of that number.

        `assemble_slots(depth)` is the FLOOR (and the whole answer before
        #1374); Option 1 raises it to whatever the largest tag needs, in whole
        slots, because a partial slot cannot hold a band.
        """
        return tag_slots(int(self.max_tag_bytes), int(self.slot_bytes),
                         int(self.depth), comparing=None,
                         floor=assemble_slots(int(self.depth), comparing=False))

    @property
    def lane_buffer_bytes(self) -> int:
        """What one lane's FILE really is: slots x slot_bytes.

        Distinct from `buffer_bytes`, which states the WIDEST-LAYER claim the
        #1332 guard grades (`covers_widest_layer`) and is left untouched. The
        charge below takes the larger of the two, so the ledger can never
        under-charge whichever geometry the leg actually allocates.
        """
        return int(self.lane_slots) * int(self.slot_bytes)

    @property
    def total_bytes(self) -> int:
        """The one number the ledger charges as ``xchg_bounce_host_bytes``.

        #1358 PER LANE, and this was a 5.64 GiB under-charge measured three
        ways on boot weg2xsn28. The term charged ONE buffer while the lane
        created FIVE separate files on tmpfs:

            /dev/shm/weg2-xchg-<epoch>/bounce.bin.{c0,c1,p1,p2,p4}
                5 x 1.409 GiB = 7.044 GiB   (filesystem, 0 holders)
            d_shmem across the source leg
                          +7.043 GiB        (cgroup sampler, independent)
            the lane's OWN HOST-SLOT lines
                 4.226 + 2.818 = 7.044 GiB  (this boot's own log)
            ARM line xchg_bounce
                            2.160 GiB       (this term, before the fix)

        The counter-proof that the instrument is sound: the same method on the
        same boot summed the host ring to 46.40 GiB against
        `host_weights=46.40` -- exact. The ledger priced the ring right and the
        bounce wrong, so it was not a tmpfs or sparse-file artefact.

        ALL FIVE ARE CONCURRENT (measured: the two legs overlap 115 s of
        120 s), so this is not a peak-vs-sum question -- reducing the count is
        a lane change, not an accounting one.

        #1385: PRICED, NOT `n_lanes` DIRECTLY. `lanes_priced` is `n_lanes`
        whenever the cap is unstated (0), so every caller from before this
        field is byte-identical; a cap only ever LOWERS the multiplier, never
        raises it, which is the half of #1358's lesson this flag exists to
        reuse in the other direction: xsn31/3 measured 5 lanes x 3.00 GiB +
        0.75 GiB staging = 15.75 GiB (`n_lanes=5, lanes_concurrent=0`) and a
        boot arming `--xchg-lanes-concurrent 2` on the identical checkpoint
        prices 2 x 3.00 + 0.75 = 6.75 GiB -- the SAME formula, a smaller
        multiplier, never a second one.
        """
        # #1397 OPTION 3: a DIFFERENT sum, not a smaller `per_lane` -- the
        # diagonal share stays at the #1374 floor (the max() below is still
        # correct FOR THOSE LANES, unchanged) and only the counted-cross
        # share prices at `cross_lane_buffer_bytes`, deliberately WITHOUT
        # that max() (see the property's own docstring for why `buffer_bytes`
        # is dead weight there). `n_cross_lanes_priced` is 0 whenever
        # `band_credit` is unset, so this branch is inert and the formula
        # below is the only one that ever runs for every pre-#1397 caller.
        if self.band_credit and int(self.n_cross_lanes_priced) > 0:
            diag_per_lane = max(int(self.buffer_bytes),
                                int(self.lane_buffer_bytes))
            diag_total = diag_per_lane * int(self.n_diag_lanes_priced)
            cross_total = (int(self.cross_lane_buffer_bytes)
                          * int(self.n_cross_lanes_priced))
            return diag_total + cross_total + int(self.staging_bytes)
        # #1374: THE LARGER OF THE TWO GEOMETRIES, per lane. `buffer_bytes` is
        # the widest-layer statement and `lane_buffer_bytes` is the file Option
        # 1 allocates; charging the smaller would under-charge the one the leg
        # creates, which is the #1358 defect one geometry over.
        per_lane = max(int(self.buffer_bytes), int(self.lane_buffer_bytes))
        return per_lane * int(self.lanes_priced) + int(self.staging_bytes)

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
        # PER SLOT, AND THE SLOT COUNT IS NOT THE DEPTH. This divided by
        # `depth` while the buffer holds `assemble_slots(depth)` -- one more
        # under a comparing arm. The division therefore reported a per-slot
        # width 1.5x the real one, and a MEAN-sized buffer could answer "yes,
        # it covers the widest layer" when it does not: the exact fail-open
        # this property exists to prevent, introduced by the very commit that
        # unified the size. Found by a 1332 pin, not by me.
        return (int(self.buffer_bytes) // assemble_slots(int(self.depth))) >= int(
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


def tag_slots(max_tag_bytes: int, slot_bytes: int, depth: int, *,
              comparing: Optional[bool] = None, floor: Optional[int] = None) -> int:
    """#1374 OPTION 1: slots one lane needs so a TAG's deposit never waits.

    ONE PRODUCER, read by the terms, by the lane allocation and by the ledger
    charge, because three derivations of this number are three chances for the
    buffer, the file and the price to disagree -- which is how boot weg2xsn28
    charged one buffer for five files.

    `max_tag_bytes` 0 means the boot did not state it: the floor stands and the
    leg refuses a wrapping band by name rather than pricing a wrap nobody can
    see. The `+1` for a comparing arm is the shadow's live-bytes slot, exactly
    as `assemble_slots` has always added it.
    """
    if comparing is None:
        from sglang.srt.weg2 import weight_exchange as wx

        comparing = wx.inject_mode() == wx.INJECT_SHADOW
    # THE SHADOW SLOT HAS ONE OWNER (#1330 ratchet): `assemble_slots(0, ...)`
    # IS that expression -- 1 under a comparing arm, 0 otherwise -- so this
    # function never re-spells `+1 if comparing`.
    shadow = int(assemble_slots(0, comparing=comparing))
    base = (int(assemble_slots(int(depth), comparing=comparing))
            if floor is None else int(floor) + shadow)
    if int(max_tag_bytes) <= 0 or int(slot_bytes) <= 0:
        return max(base, 1)
    need = -(-int(max_tag_bytes) // int(slot_bytes))     # ceil
    return max(base, need + shadow)


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
    n_lanes: int = 1,
    max_tag_bytes: int = 0,
    lanes_concurrent: int = 0,
    band_credit: bool = False,
    n_cross_lanes: Optional[int] = None,
) -> BounceTerms:
    """Derive the bounce term. Pure; raises only on inputs that cannot mean anything.

    THE BUFFER IS SIZED ON THE WIDEST LAYER, not the mean. The mean is carried
    for the log (it is what the steady stream costs) but a buffer sized on it
    fails on the first layer above it -- section 10.5, and the danger direction
    of this whole slice.

    ``lanes_concurrent`` IS A RAW MAGNITUDE, NEVER THE VALIDATED FLAG. 0 is
    the "not stated" sentinel (byte-identical) and a NEGATIVE value is folded
    to 0 rather than refused here, on purpose: this function prices a boot's
    OWN figures and knows nothing about argv, so it cannot tell "the flag was
    never given" from "the flag was given something meaningless" -- both
    reach it as an absence of a stated cap. The refusal for an explicit,
    meaningless CLI value (`N < 1`) belongs to :func:`resolve_lanes_concurrent`,
    the one place that still has the caller's intent (a flag that was set)
    beside the number -- folding it in here would let a mistyped `0` silently
    become "uncapped" instead of failing the boot that asked for it.
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
    # #1397 VERDRAHTUNG (2026-09-14). `band_credit` still needs an EXPLICIT
    # `True` from the caller -- this function manufactures no opt-in signal
    # from nothing, and "unset stays byte-identical" (`band_credit=False`'s
    # own default) is unconditional. But GIVEN that explicit `True`, a
    # caller who has not measured this flip's real cross/diagonal split
    # would otherwise get `n_cross_lanes_priced == 0` (band_credit priced,
    # nobody pulls it) -- exactly the "built but nobody pulls the lever"
    # class #1367/#1375 name. `resolve_cross_lanes` (this file) answers the
    # ONE remaining question this module CAN answer without a boot-specific
    # measurement -- how many lanes MUST be cross, structurally, given
    # `n_lanes` and this rig's fixed `weight_exchange_region.N_CARDS` -- so
    # arming reduces to the single boolean; nothing else to coordinate.
    # A caller that DID measure the real split still states its own
    # `n_cross_lanes`, INCLUDING AN EXPLICIT ``0`` (this boot truly has none)
    # -- `None` (never given, the parameter's own default) is the ONLY
    # trigger for auto-derivation, the identical `raw is None` convention
    # `resolve_lanes_concurrent` already uses one screen up, so an explicit
    # zero is respected literally rather than silently promoted.
    _n_cross_lanes = (resolve_cross_lanes(max(1, int(n_lanes)))
                      if bool(band_credit) and n_cross_lanes is None
                      else max(0, int(n_cross_lanes or 0)))
    # #1397 x #1385 INTERACTION, NAMED RATHER THAN LEFT TO COINCIDE.
    # `band_credit`'s own smaller sizing (`cross_lane_slots x slot_bytes`,
    # the small ONCARD unit) has exactly one precondition: Option 1
    # (`max_tag_bytes > 0`) is what makes `leg_slot_bytes` return
    # `terms.slot_bytes` at all -- `leg_slot_bytes`'s OTHER branch (Option 1
    # absent) returns `leg_geometry(terms)[0]`, i.e. `widest_layer_bytes`,
    # a DIFFERENT and much larger slot unit. A `band_credit=True` term with
    # `max_tag_bytes` unset would therefore PRICE at the small oncard
    # geometry while `run_bounce_leg` -- `_band_credit_leg` is `_option1_leg
    # and ...`, so it silently falls back to the Option-1-absent path --
    # ALLOCATES at the widest-layer geometry instead: the identical
    # slot-size/slot-count mismatch #1385 round 3 (`461fdebca0`/
    # `ccd0173d45`, "the allocator now builds the buffer the ledger
    # priced") measured at 5.6x on boot weg2xsn31/4, just with `band_credit`
    # as the new second reader of one decision. Refused HERE, at
    # construction, rather than left to coincide: a caller that means to
    # price the cross share smaller must state `max_tag_bytes` too. GRADED
    # AGAINST THE RESOLVED `_n_cross_lanes`, not the raw parameter, so this
    # is REACHABLE from the single-boolean arming path above: `band_credit
    # =True` alone (n_lanes > N_CARDS, max_tag_bytes unset) auto-derives a
    # positive count and hits this raise -- no caller has to ALSO
    # misconfigure `n_cross_lanes` by hand to reach it.
    if bool(band_credit) and _n_cross_lanes > 0 and int(max_tag_bytes) <= 0:
        raise ValueError(
            "band_credit=True with n_cross_lanes stated (or auto-derived "
            "from n_lanes via resolve_cross_lanes) needs max_tag_bytes > 0 "
            "(Option 1 active) -- band_credit's cross-lane price is the "
            "ONCARD slot unit (`terms.slot_bytes`), which `run_bounce_leg` "
            "(`leg_slot_bytes`) only ever allocates when Option 1 is active; "
            "with max_tag_bytes unset the leg falls back to the "
            "widest-layer-sized slot instead, and pricing the small number "
            "while allocating the big one is the #1385-round-3 mismatch "
            "class (measured 5.6x on boot weg2xsn31/4), not a saving"
        )
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
        n_lanes=max(1, int(n_lanes)),
        max_tag_bytes=max(0, int(max_tag_bytes)),
        lanes_concurrent=max(0, int(lanes_concurrent)),
        band_credit=bool(band_credit),
        n_cross_lanes=_n_cross_lanes,
    )


#: The channel the launcher's PRICED term reaches a rank on.  A rank may not
#: re-derive the term: the census is a checkpoint read and a second reader of
#: it would be a second sizing authority (the defect AMENDMENT 5 just retired
#: one instance of).  The launcher publishes the scalars it priced WITH, the
#: rank rebuilds the term through `bounce_terms` -- the same one function -- so
#: the two objects are identical by construction rather than by agreement.
ENV_BOUNCE_TERMS = "SGLANG_WEG2_XCHG_BOUNCE_TERMS"

_TERM_FIELDS = ("bytes_per_direction", "n_layers", "widest_layer_bytes",
                "pairs", "depth", "slot_bytes",
                # #1358: the lane count rides with the INPUTS, so the rank
                # recomputes the same total instead of holding one it cannot
                # check -- the rule this tuple already existed for.
                "n_lanes",
                # #1374: the tag size rides with the inputs too, or the ranks
                # rebuild a SMALLER buffer than the launcher charged for and
                # the deposit meets the wrap refusal the launcher priced away.
                "max_tag_bytes",
                # #1385: the cap rides with the inputs for the SAME reason --
                # a rank that recomputed `total_bytes` without it would price
                # every lane again, silently un-capping the very number the
                # host ledger read the smaller charge from.
                "lanes_concurrent",
                # #1397: both ride with the inputs for the identical reason --
                # a rank that rebuilt `total_bytes` without them would price
                # every lane at the diagonal floor again, silently erasing the
                # cross share's saving the launcher already charged for.
                "band_credit", "n_cross_lanes")


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
    # #1397 FIX: `band_credit` is the ONE field of `_TERM_FIELDS` typed
    # `bool`, not `int` -- `Dict[str, Any]`, not the uniform `Dict[str,
    # int]` a bare `{}` would infer from every OTHER field's assignment,
    # so `bounce_terms(**kw)` hands it a real `bool` rather than an `int`
    # that happens to be truthy. Harmless at runtime today (Python's own
    # `bool` is an `int` subtype, and `bounce_terms` casts with
    # `bool(band_credit)` regardless) but the wrong type on this ONE key
    # is exactly the seam a genuinely wrong value would cross silently:
    # BOOT7's pyright gate (2026-09-14) is what caught the type, before any
    # caller ever published a nonzero `band_credit` for real.
    kw: Dict[str, Any] = {}
    for part in text.split(","):
        key, _, value = part.partition("=")
        key = key.strip()
        if key not in _TERM_FIELDS:
            raise ValueError(
                f"{ENV_BOUNCE_TERMS} carries an unknown field {key!r}; the "
                f"published term is the launcher's own and its fields are "
                f"{_TERM_FIELDS}")
        if key == "band_credit":
            # VALIDATED, NOT MERELY CAST: `publish_terms` only ever writes
            # `int(True)`/`int(False)` (`0` or `1`) for this ONE bool field
            # among otherwise-int fields, so any OTHER value here is
            # corruption (a hand-edited env var, a future producer bug), not
            # a legitimate "how many" the way every other field's int is.
            # Coercing it with a bare `bool(...)` -- Python's own truthiness
            # -- would ARM this feature on a `2`, a `-1`, anything nonzero,
            # SILENTLY, exactly the "a feature arms itself where nobody is
            # looking" direction BOOT7's pyright gate (2026-09-14) surfaced
            # as a type error one line up from this comment's history.
            _bc = int(value)
            if _bc not in (0, 1):
                raise ValueError(
                    f"{ENV_BOUNCE_TERMS} carries band_credit={value!r}, "
                    f"which is neither 0 nor 1 -- publish_terms never writes "
                    f"anything else for this field, so this is a corrupted "
                    f"or hand-edited value. Refusing rather than coercing "
                    f"it with bare truthiness, which would silently ARM "
                    f"band_credit on ANY nonzero value nobody intended as "
                    f"'on'.")
            kw[key] = bool(_bc)
        else:
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
