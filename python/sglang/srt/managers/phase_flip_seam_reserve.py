# SPDX-License-Identifier: Apache-2.0
"""#656: size the KV pool so the FLIP SEAM can still be paid for.

THE DEFECT THIS EXISTS FOR (boot E, 2026-08-12)
-----------------------------------------------
With the TP pool sized to the PP id space, the pool stopped being an operator
number and became whatever the VRAM backs. It backed 683150 tokens, held the
1024 MiB corridor with zero breaches for its whole life -- and produced no
tokens at all. On rank1::

    staging 464 MiB needed but only 444 MiB is spendable
    (driver free 1468 MiB, reserve 1024 MiB kept free)

The sizer had filled the card down to the corridor floor and left nothing for
the seam. A flip is not free: crossing between the layouts needs device bytes
that exist NOWHERE in the pool arithmetic -- the weights-arena tail the TP
phase released and the PP phase must re-commit, the spilled drafter's restore,
and the wave-boundary slack of the backing swap. Short by 20 MiB, every
``tp_to_pp`` flip abandoned; under strict phase purity prefill can only run in
PP; so the instance answered /health with 200 and served nothing.

A pool that cannot afford to leave its own layout is not capacity.

THE LAW
-------
    pool = VRAM - corridor - staging

with ``staging`` the seam's AT-REST requirement -- what the seam costs with an
empty live set, which is the floor no amount of draining can lower. Anything
above that floor is genuine occupancy pressure and correctly answered by the
runtime's existing "retry when occupancy drops".

THE FIXED POINT, AND WHY IT IS SOLVED RATHER THAN ITERATED
----------------------------------------------------------
``staging`` is not a constant: the wave-boundary slack is
``drift_layers x row_nbytes x pool_rows``, so it GROWS with the pool it is
being subtracted from. Writing it with the ADDITIVE pool-independent floor
``A`` (the weights-arena tail), the ALTERNATIVE pool-independent floor ``F``
(the drafter's restore) and the per-row coefficient ``a``::

    staging(T) = A + max(F, a*T)      T*cell + staging(T) <= R'

TWO FLOORS, BECAUSE THEY COMBINE DIFFERENTLY, and the split is a measurement
rather than a refinement (#656, MERGE-R9 12.4). This was one flat
``max(F, a*T)`` with ``F = max(arena tail, draft restore)``, mirroring the
runtime's own ``_staging_bytes``, on the argument that the wave staging and
the re-commits belong to different instants of the seam and never coexist.

That argument holds for the DRAFTER and fails for the ARENA TAIL:
``stacks.refill`` is a PRE-cutover function, so its commit lands while the
wave state is still outstanding, while rung 2's restore runs inside
``_cutover`` after the waves' buffers are dead. The stage walk of one
``tp_to_pp`` cutover settles it -- entry 2464 MiB free, a 1386 MiB wave peak,
the refill reached at 1250 with 1214 MiB still outstanding, and the refill's
own 238 MiB taking the card to 1012: twelve MiB under a corridor law that
``max(1386, 238)`` called 54 MiB clear.

Setting ``A = 0`` reproduces the previous arithmetic byte for byte, which is
exactly what a record written before the split reads back as.

Two regimes, and exactly one of them holds for any given ``R'``::

    A. the floor binds  ->  T = (R' - A - F) / cell,   while a*T <= F
    B. the slack binds  ->  T = (R' - A) / (cell + a), while a*T >  F

Both are closed forms; the branch is chosen by testing a candidate against
its own condition. Iterating instead lands on the wrong side -- it charges
the per-row term at a pool size that will not exist once the term is charged,
so it settles BELOW the fixed point and silently gives away pool, on a ticket
whose whole subject is pool that was silently given away.

WHERE F AND a COME FROM: MEASUREMENT, NOT ARITHMETIC
----------------------------------------------------
Both are properties of the two weight LAYOUTS and the wave plan, and neither
exists until the TP stack is built -- which happens AFTER the PP pool is
sized. There is no ordering that makes them knowable in time, so they are not
computed here at all: they are MEASURED at the end of the flip boot, from the
runtime's own methods (so this number cannot drift from the number the gate
will actually check), written to a fingerprinted record, and read by the next
boot with the same configuration.

That is the house pattern, not a new one -- ``note_post_capture_leftover``
already sizes this rig's KV from the previous boot's measured leftover, for
the same reason and with the same two-boot convergence.

A COLD RECORD IS TODAY'S BEHAVIOUR, EXACTLY. No record, no correction: the
first boot of a new configuration sizes as it does now, measures, and says so
in the log. The second boot is the one that is seam-safe. This is stated
loudly rather than smoothed over, because a capacity number that depends on
an on-disk record from a previous boot is a harness trap when it is silent
(#188 records what that cost the last time it was silent).

NOT A SAFETY MARGIN. F is the seam's own arithmetic, and it is subtracted
once. Nothing here adds a percentage, a fudge, or a rounding-down on top --
choosing to run closer to the edge than the seam's own numbers allow is not
something this can express, which is the point.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "PHASE-FLIP-SEAM-RESERVE"

#: Turn the whole term off, restoring boot-E sizing exactly. A VALUE of the
#: same mechanism rather than a second code path, so the off switch cannot
#: drift from the on switch.
ENV_ENABLE = "SGLANG_PHASE_FLIP_SEAM_RESERVE"
#: Override the measured floor with an absolute MiB value. For bringing up a
#: configuration whose record does not exist yet without spending a boot on
#: it -- and for pinning a reproducible capacity when a byte gate needs one.
ENV_FIXED_MIB = "SGLANG_PHASE_FLIP_SEAM_RESERVE_MIB"

#: Margin, in MiB, the solver keeps between the measured position and the seam
#: floor. WHY A MARGIN AT ALL (#656 R4, boot K3): ``seam_allowed_tokens``
#: returns the largest T with ``have(T) >= need(T)``, i.e. it targets EQUALITY,
#: so the boot it sizes lands exactly ON its own floor and has nothing to
#: absorb second-order error. Boot K3 derived 610942 and then re-measured
#: rank2 at 1430 MiB against a 1455 MiB floor -- 25 MiB the wrong side of its
#: own gate, logging ``CANNOT FUND ITS OWN FLIP`` while completing all 30
#: cutovers. The error is allocator granularity and arena drift along the pool
#: axis, and it is one-sided in the dangerous direction.
#:
#: THIS IS NOT THE GATE'S C20 ENTRY MARGIN. The flip gate wants 512 MiB of
#: entry headroom on top of the staging, but it satisfies that at flip time
#: from transient reclaim (measured: ``CORRIDOR-GUARD cleared on device 0:
#: ... reclaimed 136 MiB from [allocator-cache]``), so reserving the gate's
#: number here would charge the pool twice for one requirement -- roughly
#: 65k tokens on this rig's binding rank. What the SIZER owes is only the
#: measurement's own error bar, and 192 MiB is ~8x the largest observed
#: deviation.
ENV_MARGIN_MIB = "SGLANG_PHASE_FLIP_SEAM_MARGIN_MIB"
DEFAULT_MARGIN_MIB = 192

#: #678: how a measured corridor breach EXPIRES. It is monotonic-max while it
#: is being observed (a shallower breach later does not retire a deeper one)
#: and halved by every flip boot that measures its seam without seeing it. A
#: breach that recurs is re-raised on the spot; one that cannot be reproduced
#: is gone in a bounded number of boots instead of taxing the arming floor for
#: the life of the configuration.
SHORTFALL_DECAY_DEN = 2

#: Below this the residue is written off to exactly 0, so the decay terminates
#: instead of leaving a long tail that still moves the floor.
SHORTFALL_FORGET_BYTES = 16 << 20

@dataclasses.dataclass(frozen=True)
class SizingPin:
    """A shipped pool size WITH the regime it was derived under (#678).

    WHY THE PROVENANCE IS IN THE ARTIFACT AND NOT IN A COMMIT MESSAGE. The
    number this replaces -- a hand-pinned 550000 -- was an empirical bound
    from a boot that armed and flipped 219+ times, and it was correct when it
    was taken. It then survived a regime change silently: `48ba9fe72a`
    ("the pool must reserve for the floor the GATE arms at") folded the arming
    floor into sizing, every seam record written after it came back smaller,
    and the pin kept being quoted as the target the sizer was failing to
    reach. A whole task was opened to find "the missing 115k" that had never
    existed as a recoverable quantity -- it was the distance between two
    regimes.

    THE RIG'S OWN METAL ALREADY BRACKETED THE HONEST ANSWER, and both ends are
    recorded here because the upper one is what makes the pin falsifiable:

      * 482490 tokens -- SAFE. Every rank above its arming floor
        (2167/3056/3233 free against 1536/1772/2414) and a full round trip
        under load.
      * 537076 tokens -- UNSAFE. 987/2286/1475 free, below the floor on two of
        three ranks, with the pre-arm ladder finding 40-46 MiB against a
        650-726 MiB gap.

    537076 is 13k BELOW the withdrawn 550000 pin. So the pin did not merely
    drift out of regime; it named a pool larger than one this rig has measured
    as unable to hold its own floor.

    HOW THE NEXT REGIME CHANGE INVALIDATES THIS. `basis_per_row_bytes` and
    `basis_arming_floor_mib` are what the rig's own records and gate said when
    the pin was derived. A registered test compares them against the live
    values; when a change moves either, the test goes red and the pin has to
    be re-derived instead of quietly outliving its inputs, which is exactly
    what 550000 did.
    """

    #: The largest pool this rig has DEMONSTRATED safe under the current
    #: regime -- not a target the sizer is asserted to reach today.
    demonstrated_safe_tokens: int
    #: The smallest pool measured UNABLE to hold its floors. Nothing may ship
    #: at or above this.
    demonstrated_unsafe_tokens: int
    #: The commit that defines the regime these numbers belong to.
    regime_commit: str
    #: Per-rank seam slope and arming floor the derivation stood on.
    basis_per_row_bytes: Tuple[float, ...]
    basis_arming_floor_mib: Tuple[int, ...]
    derived_at: str
    #: What was replaced, so a re-introduction is recognisable on sight.
    withdrawn_pin: int = 0
    withdrawn_reason: str = ""


#: #678: the re-derived pin, from the current regime's six solves (three ranks
#: x floor and seam), anchored on the measured bracket above. The slopes are
#: this rig's own records of 2026-08-16T03:56Z; the floors are the levels the
#: gate solved on the same boot.
SHIP_PIN = SizingPin(
    demonstrated_safe_tokens=482490,
    demonstrated_unsafe_tokens=537076,
    regime_commit="48ba9fe72a",
    basis_per_row_bytes=(2360.3031340235552, 424.1172657292698, 550.6682501797533),
    basis_arming_floor_mib=(1728, 1825, 2467),
    derived_at="2026-08-16",
    withdrawn_pin=550000,
    withdrawn_reason=(
        "cross-regime transfer: derived before 48ba9fe72a folded the arming "
        "floor into sizing, and 13k above the measured-unsafe 537076"
    ),
)

PROVENANCE_COLD = "cold"
PROVENANCE_STORED = "stored"
PROVENANCE_OVERRIDE = "override"
PROVENANCE_DISABLED = "disabled"
PROVENANCE_MALFORMED = "malformed"


@dataclass(frozen=True)
class SeamReserve:
    """What the seam costs this rank at rest, and where the number came from."""

    #: Pool-INDEPENDENT floor that is an ALTERNATIVE to the wave slack: the
    #: drafter's restore. Present with an empty live set and unaffected by
    #: draining. Rung 2's restore runs inside ``_cutover``, after the waves'
    #: buffers are dead, so it and the slack cannot both be resident -- which
    #: is why this one stays inside a ``max``.
    fixed_bytes: int = 0
    #: Pool-INDEPENDENT floor that is ADDED to whatever else the seam holds:
    #: the weights-arena tail (#656, MERGE-R9 12.4).
    #:
    #: SEPARATED FROM ``fixed_bytes`` BECAUSE THE TWO COMBINE DIFFERENTLY,
    #: and the measurement says so. ``stacks.refill`` is a PRE-cutover
    #: function, so its commit lands while the wave state is still
    #: outstanding; the stage walk of one tp_to_pp cutover
    #: (evidence-631/remediation-656/boot_m1.log, rank 1) entered at 2464 MiB
    #: free, reached the refill at 1250 with 1214 MiB outstanding, and the
    #: refill's own 238 MiB took the card to 1012 -- twelve MiB under the
    #: corridor law. Folded into one ``max`` the model predicted a 1250 MiB
    #: trough and could not see the breach.
    #:
    #: DEFAULT ZERO IS THE BACK-COMPAT PATH, exactly. A record written before
    #: this field existed carries ``fixed_bytes = max(arena, draft)`` and no
    #: arena entry, so it reads back as A=0 and the arithmetic below is
    #: byte-identical to what that record was written for. The next boot of
    #: that configuration re-measures and fills the field in -- the two-boot
    #: protocol this module already documents, carrying one more quantity.
    arena_fixed_bytes: int = 0
    #: Device bytes per KV POOL ROW: the wave-boundary backing slack, which
    #: scales with the pool and is therefore the term that makes this a fixed
    #: point rather than a subtraction.
    per_row_bytes: float = 0.0
    #: Bytes that WERE spendable above the corridor law when this record was
    #: taken, and the id space they were measured at. The pair is what makes
    #: the correction need no model of the sizer's other posts: it is a
    #: measured position plus the slope along which moving the pool moves it.
    have_bytes: int = 0
    id_space: int = 0
    #: The deepest CORRIDOR SHORTFALL this rank has been observed to make,
    #: in bytes: how far below the law its card went at the worst instant,
    #: recorded by the runtime's own corridor audit. ZERO until a boot
    #: actually measures one, so a rig that has never breached is sized
    #: exactly as before -- this term can only ever appear as a consequence
    #: of a measurement, which is what keeps it from becoming another
    #: constant carried between rigs (#656).
    corridor_shortfall_bytes: int = 0
    #: #678: the largest commit ONE LEG actually makes, ``max`` over the two
    #: directions of that direction's own ``arena tail + draft restore``.
    #:
    #: WHY THIS EXISTS SEPARATELY FROM ``total_fixed_bytes``. The two stored
    #: terms are each a ``max`` over BOTH directions, and they are maxed by
    #: DIFFERENT legs: on every record this rig has written, the arena tail is
    #: committed only on ``tp_to_pp`` (entering PP refills the weights arena)
    #: and the drafter's restore only on ``pp_to_tp`` (the drafter comes back
    #: with the TP layout). Summing the two maxima therefore prices a seam
    #: that commits both, which no seam does.
    #:
    #: Measured on rank 2, record 03d16efef3ad:
    #:   tp_to_pp: arena tail 1456 + draft restore 0    = 1456 MiB
    #:   pp_to_tp: arena tail 0    + draft restore 139  =  139 MiB
    #:   total_fixed_bytes (the sum of the maxima)      = 1595 MiB
    #: 1595 is not a draw any leg makes; 1456 is the honest worst leg.
    #:
    #: ZERO on a record written before this field existed, and callers fall
    #: back to ``total_fixed_bytes`` there -- the previous arithmetic exactly,
    #: which is what makes an old record readable.
    worst_leg_fixed_bytes: int = 0
    #: #678: the RESTING FREE COLUMN at the measured id space, in bytes, and
    #: the rung credit that was folded into ``have_bytes`` beside it.
    #:
    #: ``have_bytes`` is already net of the band floor AND of what the KV rung
    #: was judged able to pay (``free - band_floor + rung_fund``), so the raw
    #: free column cannot be recovered from it. Without the raw number the
    #: sizer can only APPROXIMATE the arming-floor constraint by subtracting
    #: from the budget -- which is what #678 measured going wrong in both
    #: directions: 284181 tokens when the subtrahend double-charged, 537076
    #: when it was removed and two of three ranks came up below their floor.
    #:
    #: With it, the constraint is solved instead of approximated:
    #: ``free(T) = free_at_measure + (id_space - T) * cell``, and the largest
    #: T with ``free(T) >= floor + margin`` is an equality, not an estimate.
    #:
    #: ZERO on a record written before this field existed, and the sizer falls
    #: back to the subtrahend there -- the previous arithmetic exactly.
    free_at_measure_bytes: int = 0
    rung_fund_bytes: int = 0
    #: #696: what the KV rung is GUARANTEED to deliver, as distinct from
    #: ``rung_fund_bytes`` which is what it happened to hold when the record
    #: was written. Zero means "no guarantee has ever been measured", never
    #: "guaranteed zero is fine" -- the arming floor treats the two the same
    #: way on purpose, by declining to excuse anything.
    rung_guaranteed_bytes: int = 0
    provenance: str = PROVENANCE_COLD
    written_at: Optional[str] = None
    detail: str = ""

    @property
    def total_fixed_bytes(self) -> int:
        """The seam's whole at-rest floor: the additive term plus the
        alternative one. What a caller wants when it needs ONE number for
        "the seam costs at least this much with an empty live set".

        NOT THE RIGHT NUMBER FOR AN ARMING FLOOR -- see
        :attr:`worst_leg_fixed_bytes` and :meth:`arming_draw_bytes`. It is a
        bound on what the seam machinery holds across a whole flip cycle, and
        it is deliberately left as it was so the sizing solve that was fitted
        against it does not move under this change.
        """
        return max(0, int(self.arena_fixed_bytes)) + max(0, int(self.fixed_bytes))

    def arming_draw_bytes(self) -> int:
        """What ONE seam draws -- the number a gate arming per flip wants.

        Falls back to :attr:`total_fixed_bytes` when the per-leg measurement
        is absent, so a pre-#678 record is priced exactly as before rather
        than optimistically.

        #685: THE ARENA TAIL DROPS OUT WHERE THE RUNG DEMONSTRABLY FUNDS IT,
        and this is the "third payment" ``arming_floor_subtrahend_bytes``
        already warns about, finally made true.

        The tail is real and is not disputed here: ``arena_fixed_bytes`` is
        ``max(0, pp_bytes - tp_bytes)``, the weights the PP layout needs
        beyond the TP layout's on this rank, and on the 2026-08-16 boot it
        reconciles three independent ways -- the ``rung 3 released`` lines,
        this record, and that definition (PP0 0, PP1 815, PP2 1456 MiB). It is
        largest on the SMALLEST card because uneven TP shrinks that card's TP
        shard while PP still hands it a full stage. Geometry, not a defect.

        What was wrong was charging it to the arming floor, which is a
        PERMANENT free-VRAM reservation, when the same record already showed
        the KV rung able to pay it AT FLIP TIME: PP1 954 MiB of rung_fund
        against an 815 MiB tail, PP2 1595 against 1456. Reserved permanently,
        it cost the binding rank its whole headroom -- 2474 MiB free at rest
        against a 2467 MiB floor, an equality solve with 7 MiB to spare, and
        ~47k tokens of pool.

        CONDITIONAL ON THE RECORDED FUNDING, never an unconditional removal. A
        rank whose rung cannot cover its own tail keeps paying for it in the
        floor, because for that rank the ladder genuinely cannot find it; and
        ``rung_fund_bytes`` is 0 on a record that never measured one, which
        means "no funding known", never "funded".

        NEVER BELOW THE LEG THE RUNG DOES NOT FUND. The drafter's restore is
        not arena tail and is not paid by this provider, so it floors the
        result -- relieving the arena must not relieve the other leg by
        arithmetic accident.

        GATED ON METAL. The rung that must pay was latched off until
        38c1161fd4 (its recovery aimed above the arena's immutable
        reservation and failed 59/59). On the boot carrying that fix the log
        shows the tail released 22 times at flip cadence and no recovery
        failures, which is the evidence this relief was held back for.
        """
        leg = max(0, int(self.worst_leg_fixed_bytes))
        if leg <= 0:
            return self.total_fixed_bytes
        arena = max(0, int(self.arena_fixed_bytes))
        # #696 THE EXCUSE NOW REQUIRES A GUARANTEE, NOT A SIGHTING.
        #
        # This read ``rung_fund_bytes`` -- what the rung HELD when the record
        # was written -- and used it to cancel a PERMANENT reservation. The two
        # are not the same quantity, and the difference is the whole defect.
        #
        # Measured 2026-08-16 under lane load: PP1 arena 814.9 MiB,
        # rung_fund_bytes 953.8 MiB, so the draw fell 815 -> 138.9 and 815 MiB
        # stopped being reserved. The pool was sized that much larger. Then, at
        # the fill where a seam actually runs:
        #
        #     current=473088 rows, floor=471983, slack=1105  -> 12.3 MiB
        #     staging 733 MiB needed but only 691 MiB is spendable      (x8)
        #
        # 954 MiB promised, 12.3 MiB delivered, 42 MiB short -- the same figure
        # eight times over, because it is a SIZING constant and not a runtime
        # fluctuation. pp_to_tp abandoned every ~3 s, three ARM-UNFUNDED in
        # nine seconds, decode at 187.5 s against a 180 s budget.
        #
        # The rung's deliverable is ``(current_rows - floor_rows) * bytes_per_row``
        # and ``floor_rows`` tracks the LIVE SET, so the funding collapses
        # exactly when occupancy is high -- which is exactly when a seam is
        # hard to fund. An excuse granted at low fill is spent at high fill.
        #
        # WHICH IS WHY THE TWO OBVIOUS REPAIRS ARE ONE REPAIR. "Price the
        # excuse against the worst case at max fill" and "drop the excuse" give
        # the same answer, because at max fill ``current`` approaches
        # ``floor_rows`` and the guaranteed deliverable is ZERO. So the
        # predicate is written in the general form and reads a GUARANTEE; a
        # record that has never measured one excuses nothing. The mechanism is
        # preserved for a future measurement rather than deleted, and #685's
        # own words -- "conditional on the recorded funding, never an
        # unconditional removal" -- finally hold, because the condition is now
        # a bound the rung cannot fall below instead of a number it was once
        # observed at.
        guaranteed = max(0, int(self.rung_guaranteed_bytes))
        if arena > 0 and guaranteed >= arena:
            return max(0, leg - arena, max(0, int(self.fixed_bytes)))
        return leg

    @property
    def active(self) -> bool:
        return self.id_space > 0 and (
            self.fixed_bytes > 0 or self.arena_fixed_bytes > 0 or self.per_row_bytes > 0
        )


def seam_reserve_enabled() -> bool:
    raw = os.environ.get(ENV_ENABLE)
    if raw is None or raw == "":
        return True
    return raw.strip().lower() not in ("0", "false", "no", "off")


def seam_margin_bytes(reserve: Optional["SeamReserve"] = None) -> int:
    """Bytes the solver stands back from the measured position.

    TWO TERMS, and only the first is a constant:

    * the MEASUREMENT ERROR BAR (``DEFAULT_MARGIN_MIB``), which is what the
      sizer owes because ``seam_allowed_tokens`` targets equality;
    * the MEASURED CORRIDOR SHORTFALL for this rank, if any boot has ever
      recorded one. This is zero on a rig that has never breached, so it
      cannot travel between configurations as a number somebody picked.

    WHY THE SECOND TERM EXISTS (#656 acceptance, 2026-08-13). The sizer
    guarantees the seam is fundable AT REST. It says nothing about the
    level the card actually reaches while the seam RUNS, and on this rig
    that drawdown measured 1814-1852 MiB against a gate that assumes 512.
    Five cutovers entered through a gate with no objection and took the
    card to 886 MiB, 138 below the law. The shortfall is knowable only by
    measurement -- it is a property of the geometry, the load and the
    card -- so the honest term is the one the runtime writes down after it
    has seen it, and the two-boot protocol this file already documents is
    exactly the vehicle for carrying it to the next boot.

    ``SGLANG_PHASE_FLIP_SEAM_MARGIN_MIB`` still overrides BOTH terms, so a
    bring-up can pin the whole margin to one number.

    Read from the environment every call rather than cached: the same
    discipline as ``seam_reserve_enabled``, so a boot can be re-run with a
    different margin without a code change, and a test can set it per case.
    A malformed or negative value falls back to the default rather than
    disabling the margin -- the failure mode this exists to prevent is a
    zero-margin pool, so an unparsable override must not produce one.
    """
    raw = os.environ.get(ENV_MARGIN_MIB)
    if raw is not None:
        try:
            mib = int(raw)
        except (TypeError, ValueError):
            mib = -1
        if mib >= 0:
            return mib << 20
        # Malformed or negative: fall through to the derived margin rather
        # than to zero. The failure mode this exists to prevent is a
        # zero-margin pool, so an unparsable override must not produce one.
    measured = 0 if reserve is None else max(0, int(reserve.corridor_shortfall_bytes))
    return (DEFAULT_MARGIN_MIB << 20) + measured


#: #662-F4 / A0: how much free VRAM the ARMING FLOOR needs on top of the
#: corridor law, as a load margin. The floor is a level the card must hold at
#: the INSTANT an arm is considered, and a card that sits exactly on it arms
#: only while nothing else moves. Same shape and same default as the seam
#: measurement's error bar, for the same reason.
ENV_ARMING_MARGIN_MIB = "SGLANG_PHASE_FLIP_ARMING_MARGIN_MIB"
DEFAULT_ARMING_MARGIN_MIB = 192


def configured_arming_floor_mib(server_args=None) -> int:
    """The operator's arming-floor override, or 0.

    READ FROM THE SAME TWO PLACES THE GUARD READS, in the same order, because
    the whole point of this term is that the sizer reserves for the number the
    gate will actually enforce. This rig sets ``SGLANG_CORRIDOR_FLOOR_MIB=1536``
    while the derived value is 1331: a sizer that reserved for the derived one
    would leave every rank 205 MiB short of the floor its own gate arms at, and
    the resulting boot looks exactly like the defect this fix exists for.
    """
    from sglang.srt.managers.phase_flip_spill import CORRIDOR_FLOOR_ENV

    raw = os.environ.get(CORRIDOR_FLOOR_ENV) or getattr(
        server_args, "phase_flip_corridor_floor_mib", None
    )
    try:
        return max(0, int(raw)) if raw else 0
    except (TypeError, ValueError):
        return 0


def arming_floor_target_bytes(
    configured_mib: int = 0, measured_draw_mib: int = 0
) -> int:
    """The free VRAM a rank must keep so a flip can ARM at all.

    RESOLVED EXACTLY AS ``get_corridor_guard`` RESOLVES IT -- ``max(configured,
    derived-from-the-measured-draw)``, highest wins -- and not by a second
    derivation that happens to agree today. Two numbers that must be equal and
    are computed twice are the shape this corpus has already paid for more than
    once; here the consequence would be silent, because a pool sized against a
    lower floor produces a healthy-looking boot that simply never flips.

    THIS IS THE TERM THE POOL SOLVE WAS MISSING, and its absence is a boot
    that cannot flip in either direction no matter what happens at runtime.

    Measured 2026-08-15 (boot_maxfill.log): with a COLD seam record the sizer
    charged NO flip term at all -- ``SeamReserve.active`` is False when every
    measured field is zero, so ``seam_adjusted_budget_bytes`` returned the
    budget unchanged -- and filled the pool to the corridor law. Rank 1 came
    up with ~875 MiB free against an arming floor of 1536 MiB, so ``tp_to_pp``
    was refused on every arm, every prefill stayed in the TP layout, and
    NOTHING AT RUNTIME COULD RECOVER IT: the pool is fixed at boot. The
    operator had to hand-pin ``--max-total-tokens`` to work around it. That is
    the planner failing to solve something it has every input for.

    WHY IT IS SEPARATE FROM THE SEAM RESERVE, rather than a cold default for
    it. They are different quantities measured at different instants. The seam
    reserve is what a flip COSTS WHILE IT RUNS, is genuinely knowable only by
    measurement, and legitimately reads zero on a first boot. The arming floor
    is a LEVEL THE GATE COMPARES AGAINST, is derived from the corridor law the
    operator already stated, and is knowable at boot on every rig. Folding the
    second into the first would make a derived constant look like a
    measurement -- the #584 ``derived_provenance`` defect, which this chain
    has already paid for once.

    So this term is charged on EVERY flip boot, cold or stored: a cold record
    means the staging cost is unknown, never that the arming level is.
    """
    from sglang.srt.managers import corridor_guard as cg

    derived_mib = cg.arming_floor_mib(
        seam_entry_reserve_mib=max(
            cg.DEFAULT_SEAM_ENTRY_RESERVE_MIB, max(0, int(measured_draw_mib or 0))
        ),
        law_mib=cg.corridor_law_mib(),
    )
    floor_mib = max(max(0, int(configured_mib or 0)), int(derived_mib))
    return (floor_mib << 20) + _arming_margin_bytes()


def _arming_margin_bytes() -> int:
    """The load margin above the floor. Malformed overrides fall back to the
    default rather than to zero, for the reason ``seam_margin_bytes`` gives:
    the failure this exists to prevent is a pool with no margin at all."""
    raw = os.environ.get(ENV_ARMING_MARGIN_MIB)
    if raw is not None:
        try:
            mib = int(raw)
        except (TypeError, ValueError):
            mib = -1
        if mib >= 0:
            return mib << 20
    return DEFAULT_ARMING_MARGIN_MIB << 20


def seam_solve_reserved_free_bytes(reserve: "SeamReserve") -> int:
    """Free VRAM the SEAM SOLVE already arranges to leave at rest, or 0.

    #678. ``seam_allowed_tokens`` picks the largest id space with
    ``have(T) >= need(T)``, and ``have`` was measured as
    ``free - band_floor + rung_fund``. Targeting equality therefore sizes the
    pool so that the resting free column is about
    ``band_floor + seam_draw - rung_fund``. That is not a coincidence and it is
    not slack: it is the same requirement the arming floor states, arrived at
    from the other side.

    So a boot with a MEASURED record has already paid for the arming floor
    once, inside the solve. Charging the whole floor again on top is the
    "double insurance paid twice" this rig measured: the unpinned solve came
    back at 284181 tokens where the hand-pinned 550000 arms, flips 219+ times
    and clears its guard.

    ZERO WITHOUT AN ACTIVE RECORD, which is the case that must still pay in
    full. A cold boot's solve reserves nothing for the seam, so nothing has
    been paid and the floor is charged whole -- exactly the r2 behaviour that
    landed every rank above its floor.

    The ``rung_fund`` term is deliberately NOT added back. The solve counts the
    KV rung as a payer at seam time while the gate wants free VRAM at arm
    time, so a gap of that size can remain -- and covering it is precisely
    what the pre-arm relief ladder is for. Reserving it here as well would be
    the third payment for one requirement.

    THE MEASUREMENT'S ERROR BAR IS NOT COUNTED HERE, AND THE METAL SAYS WHY.
    ``seam_allowed_tokens`` does solve against ``have - seam_margin_bytes``, so
    on paper that margin is additional free VRAM at the solved id space, and
    counting it drives the residual charge to zero. Booted (boot_678_final.log,
    537076 tokens) the cards came up at 987/2286/1475 MiB against arming floors
    of 1536/1633/2275 -- BELOW the floor on two of three ranks, with the
    pre-arm ladder able to find only 40-46 MiB against a 650-726 MiB gap.

    The reason is the ``rung_fund`` term excluded above: the solve counts the
    KV rung as a payer, so it permits the resting free column to land that much
    lower, and the paper margin is spent covering it. A baseline that assumes
    both is optimistic by exactly the amount that matters. So the error bar
    stays out, the charge stays at the load margin, and the pool the planner
    picks is one whose cards can actually hold their own arming floor --
    measured at 482490 tokens with free 2167/3056/3233 against 1536/1772/2414
    and a full round trip under load.
    """
    if not reserve.active:
        return 0
    return _band_floor_bytes(_corridor_law_bytes()) + reserve.arming_draw_bytes()


def arming_floor_subtrahend_bytes(
    arming_floor_bytes: int, already_reserved_bytes: int = 0
) -> int:
    """KV bytes the pool must give up so the floor is actually free.

    WHAT IS ALREADY HELD FREE IS NOT OWED AGAIN. Two things can already be
    holding memory back at this point, and the charge is the shortfall against
    the larger of them:

    * the corridor LAW, which the sizer holds free on every boot. Charging the
      whole floor over it would reserve the law twice -- the same double-count
      ``required_free_bytes`` refuses with its ``max``.
    * the SEAM SOLVE, on a boot with a measured record. See
      :func:`seam_solve_reserved_free_bytes` for why that solve already leaves
      about the arming floor free, and for the 284181-vs-550000 measurement
      that charging it twice produced.

    Zero when the floor is covered already, so a rig whose arming floor sits
    below its corridor -- or whose seam solve already reserves more than the
    floor -- is sized exactly as before.

    ``already_reserved_bytes`` defaults to 0, which reduces to the law-only
    baseline: the previous arithmetic, and what every caller that knows
    nothing about a seam record should get.
    """
    baseline = max(int(_corridor_law_bytes()), max(0, int(already_reserved_bytes)))
    return max(0, int(arming_floor_bytes) - baseline)


def record_path(server_args, world_rank: int) -> str:
    """One record per (configuration, rank).

    PER RANK, and not per configuration alone: the arena tail is
    ``max(0, pp_bytes - tp_bytes)`` on THIS rank's two layouts, and on the
    ship boot that is 1436 MiB on rank2 against 466 MiB on rank1 and 0 on
    rank0. A shared record would size every rank from one rank's seam --
    exactly the cross-stage trap ``_measured_kv_budget_cache_path`` grew its
    ``-stage{pp_rank}`` suffix to close.
    """
    from sglang.srt.uneven_perf import measured_kv_budget_cache_path

    base = measured_kv_budget_cache_path(server_args)
    root, _ext = os.path.splitext(base)
    return f"{root}-seam-rank{int(world_rank)}.json"


def read_seam_reserve(server_args, world_rank: int) -> SeamReserve:
    """This rank's seam floor for THIS boot. Never raises."""
    if not seam_reserve_enabled():
        return SeamReserve(provenance=PROVENANCE_DISABLED)

    override = os.environ.get(ENV_FIXED_MIB)
    if override not in (None, ""):
        try:
            mib = int(float(override))
        except ValueError:
            logger.warning(
                "%s %s=%r is not a number; ignoring the override",
                LOG_PREFIX,
                ENV_FIXED_MIB,
                override,
            )
        else:
            return SeamReserve(
                fixed_bytes=max(0, mib) * (1 << 20),
                provenance=PROVENANCE_OVERRIDE,
                detail=f"{ENV_FIXED_MIB}={mib} MiB",
            )

    path = record_path(server_args, world_rank)
    if not os.path.exists(path):
        return SeamReserve(provenance=PROVENANCE_COLD, detail=path)
    try:
        with open(path) as fh:
            rec = json.load(fh)
        return SeamReserve(
            fixed_bytes=int(rec["fixed_bytes"]),
            # Absent in records written before #656 MERGE-R9 12.4. Absent
            # means "this boot did not separate the two floors", and the
            # zero it defaults to reproduces that record's own arithmetic
            # exactly rather than guessing a split for it.
            arena_fixed_bytes=int(rec.get("arena_fixed_bytes", 0)),
            # ABSENT ON A PRE-#678 RECORD, and 0 makes arming_draw_bytes fall
            # back to total_fixed_bytes -- the previous pricing exactly, which
            # is what keeps an old record readable rather than optimistic.
            worst_leg_fixed_bytes=int(rec.get("worst_leg_fixed_bytes", 0)),
            free_at_measure_bytes=int(rec.get("free_at_measure_bytes", 0)),
            rung_fund_bytes=int(rec.get("rung_fund_bytes", 0)),
            rung_guaranteed_bytes=int(rec.get("rung_guaranteed_bytes", 0)),
            per_row_bytes=float(rec["per_row_bytes"]),
            have_bytes=int(rec.get("have_bytes", 0)),
            id_space=int(rec.get("id_space", 0)),
            # Absent in records written before #656; absent means "never
            # measured", which is exactly the zero the term defaults to.
            corridor_shortfall_bytes=int(rec.get("corridor_shortfall_bytes", 0)),
            provenance=PROVENANCE_STORED,
            written_at=rec.get("written_at"),
            detail=rec.get("detail", ""),
        )
    except Exception as e:
        logger.warning("%s record %s unreadable (%s); sizing cold", LOG_PREFIX, path, e)
        return SeamReserve(provenance=PROVENANCE_MALFORMED, detail=path)


def write_seam_reserve(
    server_args,
    world_rank: int,
    fixed_bytes: int,
    per_row_bytes: float,
    detail: str,
    have_bytes: int = 0,
    id_space: int = 0,
    arena_fixed_bytes: int = 0,
    worst_leg_fixed_bytes: int = 0,
    free_at_measure_bytes: int = 0,
    rung_fund_bytes: int = 0,
    rung_guaranteed_bytes: int = 0,
) -> Optional[str]:
    """Persist this boot's measurement for the next one. Never raises.

    Written at the END of the flip boot, where both layouts, the arena
    carrier and the drafter all exist and the numbers are exact rather than
    predicted.
    """
    path = record_path(server_args, world_rank)
    # PRESERVED WHEN THIS BOOT SAW IT, DECAYED WHEN IT DID NOT (#678).
    #
    # The shortfall is written by a different event (the runtime's corridor
    # audit, mid-run) than this measurement (end of the flip boot), so a boot
    # that re-measures its seam must not silently discard a breach an earlier
    # boot paid to learn about. It must also not RENEW one indefinitely: the
    # term is added straight to the arming floor's load margin, and the floor
    # binds the pool on two of three ranks, so a value nobody can reproduce
    # sizes the instance forever.
    #
    # Measured 2026-08-16: the rank-0 record carried 1004 MiB while every
    # record from the day before carried 0, and the boot reading it logged no
    # breach of its own. The event it descends from is a foreign process that
    # held 4.29 GiB on that card for a few seconds.
    #
    # SAME PID MEANS THIS BOOT SAW IT. Both writers live in the same process,
    # so the pid the audit stamped is the exact discriminator between "this
    # boot observed a breach" and "this value was inherited". A boot that
    # completed a seam measurement without its audit firing is evidence
    # against the inherited value, and evidence is what retires it.
    prior_shortfall = 0
    prior_pid = None
    try:
        with open(path) as fh:
            _prior = json.load(fh)
        prior_shortfall = int(_prior.get("corridor_shortfall_bytes", 0))
        prior_pid = _prior.get("corridor_shortfall_pid")
    except Exception:
        prior_shortfall = 0
        prior_pid = None
    if prior_shortfall > 0 and prior_pid != os.getpid():
        decayed = int(prior_shortfall) // SHORTFALL_DECAY_DEN
        if decayed < SHORTFALL_FORGET_BYTES:
            decayed = 0
        logger.info(
            "%s corridor shortfall %d MiB was inherited, not observed by this "
            "boot: decaying to %d MiB. A breach that recurs is re-observed and "
            "re-raised to its worst on the spot; one that does not is retired "
            "rather than charged to the arming floor forever (#678).",
            LOG_PREFIX,
            int(prior_shortfall) >> 20,
            decayed >> 20,
        )
        prior_shortfall = decayed
    payload = {
        "fixed_bytes": int(fixed_bytes),
        "arena_fixed_bytes": int(arena_fixed_bytes),
        "worst_leg_fixed_bytes": int(worst_leg_fixed_bytes),
        "free_at_measure_bytes": int(free_at_measure_bytes),
        "rung_fund_bytes": int(rung_fund_bytes),
        "rung_guaranteed_bytes": int(rung_guaranteed_bytes),
        "per_row_bytes": float(per_row_bytes),
        "have_bytes": int(have_bytes),
        "id_space": int(id_space),
        "corridor_shortfall_bytes": max(0, prior_shortfall),
        "corridor_shortfall_pid": prior_pid if prior_shortfall > 0 else None,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "detail": detail,
    }
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(payload, fh, indent=1)
        # Atomic, so a boot that dies mid-write leaves the previous record
        # intact rather than a truncated one the next boot must discard.
        os.replace(tmp, path)
        return path
    except Exception as e:
        logger.warning("%s could not write %s: %s", LOG_PREFIX, path, e)
        return None


def record_corridor_shortfall(
    server_args, world_rank: int, shortfall_bytes: int
) -> Optional[int]:
    """Persist a MEASURED corridor shortfall for this rank. Never raises.

    Called by the runtime's corridor audit when the continuous minimum on
    this rank's card has gone below the law. The next boot of the same
    configuration reads it back through :func:`read_seam_reserve` and the
    solver stands that much further back -- the two-boot protocol this file
    already documents, carrying one more measured quantity.

    A MONOTONIC MAXIMUM, deliberately. A shallower breach later does not
    mean the deeper one cannot recur; the pool must be sized for the worst
    instant that has ever been seen, which is the same rule the corridor law
    itself is stated with. Returns the value now on record, or None if it
    could not be written.
    """
    want = max(0, int(shortfall_bytes))
    if want <= 0:
        return None
    path = record_path(server_args, world_rank)
    try:
        with open(path) as fh:
            rec = json.load(fh)
    except Exception:
        # No record yet: a cold boot has nothing to append to, and inventing
        # one here would hand the next boot a seam floor nobody measured.
        return None
    prior = int(rec.get("corridor_shortfall_bytes", 0))
    if want <= prior:
        return prior
    rec["corridor_shortfall_bytes"] = want
    rec["corridor_shortfall_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    #: #678: WHICH PROCESS SAW IT. `write_seam_reserve` decays a shortfall it
    #: did not observe itself, and both writers run in this process, so the pid
    #: is the discriminator between a live measurement and an inherited one.
    rec["corridor_shortfall_pid"] = os.getpid()
    try:
        tmp = f"{path}.tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            json.dump(rec, fh, indent=1)
        os.replace(tmp, path)
    except Exception as e:
        logger.warning(
            "%s could not record the shortfall in %s: %s", LOG_PREFIX, path, e
        )
        return None
    logger.warning(
        "%s recorded a MEASURED corridor shortfall of %d MiB for rank %d "
        "(was %d MiB). The next boot of this configuration will stand that "
        "much further back when it sizes the pool, which is the two-boot "
        "protocol carrying one more measured quantity. This term is zero "
        "until a breach is actually observed, so it can never travel to "
        "another rig as a number somebody picked.",
        LOG_PREFIX,
        want >> 20,
        int(world_rank),
        prior >> 20,
    )
    return want


# ---------------------------------------------------------------------------
# The arithmetic. Pure, so it is testable without a GPU, a model or a boot.
# ---------------------------------------------------------------------------


def required_free_bytes(
    headroom_bytes: int, corridor_bytes: int, seam_fixed_bytes: int
) -> int:
    """Bytes that must remain free after the pool is placed.

    MAX, NOT SUM, against the existing headroom. ``headroom_bytes`` is the
    activation reserve the sizer already leaves; the corridor law and the
    seam floor must BOTH be satisfiable out of what is left free, and on this
    rig the existing headroom already covers the corridor with room to spare
    (boot E rested at 1468 MiB free against a 1024 MiB law). Adding the terms
    would reserve the corridor twice and give back most of the capacity this
    change exists to keep. Taking the max reserves exactly the binding one --
    on boot E's rank1, 1024 + 464 = 1488 against a 1468 MiB headroom, i.e.
    the 20 MiB that were actually missing.
    """
    return max(int(headroom_bytes), int(corridor_bytes) + int(seam_fixed_bytes))


def solve_pool_tokens(
    corridor_relaxed_bytes: int,
    cell_bytes: int,
    fixed_bytes: int,
    per_row_bytes: float,
    arena_fixed_bytes: int = 0,
    received_layers: Optional[int] = None,
) -> int:
    """The largest T with ``T*cell + staging(T) <= R'``.

    ``R'`` is ``corridor_relaxed_bytes``: the KV byte budget plus whatever
    headroom the sizer already leaves, minus the corridor law -- i.e. every
    byte that could become KV if the corridor were the only thing to respect.

    ``staging(T) = A + max(F, a*T)``, and the SHAPE is the #656 MERGE-R9 12.4
    correction rather than an elaboration:

    * ``A`` (``arena_fixed_bytes``) is the weights-arena tail, committed by
      ``stacks.refill`` at the PRE-cutover seam and therefore held WHILE the
      wave state is still outstanding. It ADDS.
    * ``F`` (``fixed_bytes``) is the drafter's restore, which runs inside
      ``_cutover`` after the waves' buffers are dead. It and the slack cannot
      coexist, so it stays an ALTERNATIVE.

    The whole floor used to be one ``max`` against the slack, on the argument
    that the peaks belong to different instants and summing them "would model
    a peak that does not occur". The stage walk falsified that for the arena
    tail and only for the arena tail: one tp_to_pp cutover entered at 2464 MiB
    free, reached the refill at 1250 with 1214 MiB outstanding, and the
    refill's 238 MiB took it to 1012 -- under the 1024 MiB law that
    ``max(1214, 238)`` said was 226 MiB clear.

    ``A = 0`` restores the previous arithmetic exactly, which is what a
    pre-#656 record reads back as.

    Two regimes, one of which always holds:

      A. the floor binds  ->  T = (R' - A - F) / cell,   valid while a*T <= F
      B. the slack binds  ->  T = (R' - A) / (cell + a), valid while a*T >  F

    Both are exact; the branch is chosen by checking the candidate against
    its own condition, so there is no iteration and no tolerance.
    """
    cell = int(cell_bytes)
    if cell <= 0:
        return 0
    R = max(0, int(corridor_relaxed_bytes))
    F = max(0, int(fixed_bytes))
    A = max(0, int(arena_fixed_bytes))
    a = max(0.0, float(per_row_bytes))
    # #685 THE ZERO-RECEIVE RANK PAYS NO PER-TOKEN SEAM.
    #
    # ``per_row_bytes`` is a MEASURED per-boot record, and on a rank that
    # receives no attention layers at the flip it measures BASELINE cost --
    # checksums, the one-layer streaming window, allocator grain -- not
    # transfer. Charging it as a per-token seam reserve makes that rank hold
    # back memory for bytes that never move, which shrinks its pool and its
    # spendable headroom at exactly the moment the seam needs them.
    #
    # Measured 2026-08-16: the binding rank was short 192 MiB on a 1059 MiB
    # need whose per-token term was ~190 MiB, i.e. essentially the whole
    # shortfall was a charge for a transfer it does not perform. received_r
    # comes from seam_slope.received_attention_layers; rank 0 (received 1)
    # keeps its measured slope, which matches the derivation to 1.4%.
    #
    # None means "not supplied" and is NOT zero: callers that cannot derive
    # the layout keep the previous arithmetic exactly.
    if received_layers is not None and int(received_layers) <= 0:
        a = 0.0

    t_floor = max(0, (R - A - F) // cell)
    if a * t_floor <= F:
        return int(t_floor)
    return int(max(0, (R - A) // (cell + a)))


def corridor_relaxed_bytes(
    budget_bytes: int, headroom_bytes: int, corridor_bytes: int
) -> int:
    """``R' = budget + headroom - corridor``.

    The sizer hands out ``budget_bytes`` of KV and leaves ``headroom_bytes``
    free. The corridor law needs ``corridor_bytes`` of that free, and
    everything else is fungible with KV -- so this is the honest ceiling to
    solve the seam against, rather than pretending the existing headroom is
    untouchable. On boot E's rank1 the headroom (1468 MiB) exceeded the law
    (1024 MiB) by 444 MiB, and it was 464 MiB of staging that had to come out
    of it: the 20 MiB shortfall falls straight out of this subtraction.
    """
    return int(budget_bytes) + int(headroom_bytes) - int(corridor_bytes)


def seam_allowed_tokens(
    cell_bytes: int,
    reserve: "SeamReserve",
    *,
    abandon_is_survivable: bool = False,
    received_layers: Optional[int] = None,
) -> int:
    """The largest id space whose seam this rank can still fund.

    ``abandon_is_survivable`` says what a refused seam COSTS, and it is the
    difference between paying for the guarantee with a smaller pool and
    refusing to. Under strict purity a refused ``tp_to_pp`` means prefill
    never runs (boot E held the corridor and served nothing), so the pool
    pays. Where the other layout may do the work -- ``prefill_in_tp`` --
    the refusal is a runtime abandon that costs one flip, and the corridor
    law wins instead. Derived from the purity mode, never configured
    separately: it is a consequence of the boot, not a preference.

    ANCHORED ON A MEASUREMENT, NOT ON A MODEL OF THE SIZER. The previous
    boot recorded, at a known id space ``T_m``, how many bytes were actually
    spendable above the corridor law (``have_m``). Every KV row given back
    returns ``cell`` bytes to that pool, so along the pool axis::

        have(T) = have_m + (T_m - T) * cell
        need(T) = A + max(F, a*T)

    and the answer is the largest T with ``have(T) >= need(T)``:

        A. the floor binds  ->  T <= T_m + (have_m - A - F) / cell
        B. the slack binds  ->  T <= (have_m + T_m*cell - A) / (cell + a)

    ``A`` is the ADDITIVE floor -- the weights-arena tail, committed at the
    pre-cutover seam while the wave state is still outstanding (#656,
    MERGE-R9 12.4, and ``solve_pool_tokens`` for the walk that shows it).
    ``F`` is the drafter's restore and stays an alternative to the slack.
    ``A = 0`` is exactly the pre-#656 arithmetic, which is what a record
    written before the split reads back as.

    This needs no term for the activation reserve, the capture peak, the
    arena, the TP stack or the invisible carve-out -- all of them are already
    inside ``have_m``, because it was measured with all of them resident.
    That is why it works on the pre-capture sizing path, where no headroom
    quantity exists to reason about (boot H: the post-capture path never ran,
    so a correction keyed to its headroom was never applied and the pool came
    back at 683150 unchanged).
    """
    cell = int(cell_bytes)
    if cell <= 0 or not reserve.active:
        return 0
    # The margin comes off the MEASURED POSITION, not off the floor. In the
    # floor-bound regime the two are algebraically identical; in the
    # slack-bound regime only this one has any effect, because ``F`` appears
    # there solely in the regime test. Taking it here therefore means the same
    # thing on both branches: "stand this far back from where we measured".
    have_m = max(0, int(reserve.have_bytes) - seam_margin_bytes(reserve))
    t_m = int(reserve.id_space)
    F = max(0, int(reserve.fixed_bytes))
    A = max(0, int(reserve.arena_fixed_bytes))
    a = max(0.0, float(reserve.per_row_bytes))
    # See the note in solve_pool_tokens: a zero-receive rank's measured
    # per_row_bytes is baseline, not seam, and must not be reserved per token.
    if received_layers is not None and int(received_layers) <= 0:
        a = 0.0

    t_floor = t_m + (have_m - A - F) // cell
    if a * max(0, t_floor) <= F:
        solved = int(max(0, t_floor))
    else:
        solved = int(max(0, (have_m + t_m * cell - A) // (cell + a)))
    if not abandon_is_survivable:
        # An abandoned seam is FATAL here, so the pool must pay for the
        # guarantee. This is boot E's lesson and it is not being unlearned.
        return solved
    return _clamped_to_the_law(solved, t_m, have_m, cell)


def _clamped_to_the_law(solved: int, t_m: int, have_m: int, cell: int) -> int:
    """The reserve may SPEND slack above the corridor law. It may not MAKE it.

    ``have(T) = have_m + (t_m - T) * cell`` is the whole reason this solver
    can answer at all, and it says plainly where the bytes come from: every
    token cut off the id space turns into free VRAM. So a solve that returns
    ``T < t_m`` is not finding memory, it is MANUFACTURING free VRAM by
    making the KV pool permanently smaller.

    That is legitimate only while it is spending slack the instance already
    had. Once ``have_m`` is exhausted -- i.e. once the measured position sits
    ON the law -- any further cut pushes free VRAM ABOVE the law and holds it
    there for the life of the instance.

    THE CORRIDOR LAW HAS TWO HALVES AND THIS BROKE THE SECOND ONE. The law is
    "~1024 MiB free per card, and best-filled": 2-3 GiB too much free is the
    same defect as too little, because those bytes buy no tokens and serve no
    request. Measured on this rig 2026-08-15, one boot, one config:

        t_m = 590000, have_m = 0 MiB spendable  ->  solved 486403
        at-rest free per card: 1139 / 1462 / 2279  ->  3153 / 4604 / 3977

    The instance was ON the law and the reserve moved it ~2 GiB per card above
    it, permanently, to guarantee a seam could be funded at an occupancy the
    pool now could not even reach. The guarantee is not worth a pool: an
    unfundable seam is a runtime ABANDON, which is free, unanimous and leaves
    every request intact, while the bytes it was bought with are gone for the
    whole boot.

    So the cut stops at the measured position. When that is not enough to
    fund the seam the caller is told, loudly, that flips may abandon at this
    pool size -- which is a true statement about a serving instance, not a
    reason to shrink one.
    """
    if solved >= t_m or t_m <= 0:
        return solved
    if have_m <= 0:
        logger.warning(
            "%s the seam wants a SMALLER POOL than the corridor law already "
            "pays for: solved %d tokens against a measured position of %d, "
            "with 0 MiB spendable above the law. Cutting there would hold "
            "roughly %.0f MiB per card free ABOVE the law for the life of "
            "the instance, and free VRAM above the law buys no tokens. The "
            "id space STAYS at %d. A seam that cannot be funded at this size "
            "abandons at runtime -- free, unanimous, every request intact.",
            LOG_PREFIX,
            solved,
            t_m,
            (t_m - solved) * cell / (1 << 20),
            t_m,
        )
        return int(t_m)
    # Slack above the law exists: spend it, and no more than it.
    spendable_tokens = int(have_m // cell)
    floor_at_law = max(0, t_m - spendable_tokens)
    if solved >= floor_at_law:
        return solved
    logger.warning(
        "%s the seam's cut is bounded by the corridor law: solved %d tokens, "
        "but only %d MiB sits above the law, which is %d tokens' worth. The "
        "id space stops at %d instead. Cutting further would manufacture free "
        "VRAM above the law rather than spend slack the instance had.",
        LOG_PREFIX,
        solved,
        have_m // (1 << 20),
        spendable_tokens,
        floor_at_law,
    )
    return int(floor_at_law)


def floor_allowed_tokens(
    cell_bytes: int, reserve: "SeamReserve", floor_target_bytes: int
) -> Optional[int]:
    """Largest id space whose RESTING FREE COLUMN still holds the arming floor.

    #678, and it replaces an approximation with an equality. The constraint has
    always been "every rank must rest at or above ``floor + margin``, or it can
    never arm a flip and the pool is fixed at boot". Expressing it as a
    SUBTRAHEND on the KV budget could not state that, because the budget is not
    the free column: what the pool gives up and what the card ends up holding
    free are related by the sizer's other posts, and the gap between them is
    exactly where both failures of this ticket lived.

        284181 tokens  the subtrahend double-charged, reserving a floor the
                       seam solve had already reserved
        537076 tokens  the double charge removed, and two of three ranks came
                       up BELOW their floor -- 987 MiB against 1536, with the
                       pre-arm ladder finding 46 MiB of a 650 MiB gap

    Both are the same error in opposite directions: a quantity that must be
    solved for was being adjusted towards.

    THE SOLVE. The record now carries the resting free column at the measured
    id space, so free moves along the pool axis exactly as ``have`` does::

        free(T) = free_at_measure + (id_space - T) * cell

    and the answer is the largest T with ``free(T) >= floor + margin``::

        T <= id_space + (free_at_measure - floor - margin) / cell

    One line, exact, and it needs no model of the activation reserve, the
    capture peak, the arena or the carve-out -- all of them were resident when
    ``free_at_measure`` was taken, which is the same argument
    ``seam_allowed_tokens`` makes for its own anchor.

    RETURNS None WHEN IT CANNOT BE SOLVED -- a cold record, no cell, or a
    record written before the free column was persisted. The caller falls back
    to the subtrahend there, which is the previous behaviour exactly. None and
    zero are deliberately different: zero would mean "no tokens are allowed",
    which is a verdict, and this function must not invent one.
    """
    cell = int(cell_bytes)
    if cell <= 0 or not reserve.active:
        return None
    free_m = int(reserve.free_at_measure_bytes)
    t_m = int(reserve.id_space)
    if free_m <= 0 or t_m <= 0:
        return None
    target = max(0, int(floor_target_bytes))
    return max(0, t_m + (free_m - target) // cell)


def seam_adjusted_budget_bytes(
    budget_bytes: int,
    cell_bytes: int,
    reserve: "SeamReserve",
    *,
    abandon_is_survivable: bool = False,
    arming_floor_bytes: int = 0,
    received_layers: Optional[int] = None,
) -> Tuple[int, int]:
    """(new_budget_bytes, allowed_tokens). Never GROWS the budget.

    TWO CHARGES, AND THEY ARE NOT THE SAME KIND OF THING.

    * ``arming_floor_bytes`` is the LEVEL the gate compares against. It is
      charged whenever it is given, INCLUDING on a cold record, because a
      first boot not knowing what a flip costs is not the same as it not
      knowing what level the gate wants. Passing 0 -- the default -- is
      byte-identical to the previous behaviour, which is what every non-flip
      boot gets.
    * the seam RESERVE is what the flip costs while it runs, is knowable only
      by measurement, and is legitimately absent on a first boot. Unchanged:
      a cold record, a disabled term or a configurator with no single
      per-token cell still charges nothing for it.

    THE FLOOR IS CHARGED LAST, AFTER THE MIN, AND THAT ORDERING IS THE FIX.
    Charging it against the budget FIRST looks equivalent and is not: the
    result is ``min(budget, allowed * cell)``, so whenever the seam solve binds
    -- which is whenever a MEASURED record exists -- the min picks
    ``allowed * cell`` and the subtraction is silently discarded.

    Measured, and it cost a boot. The first version of this charge worked on a
    COLD record (``allowed`` is 0 there, so the budget path applies) and did
    nothing at all on the WARM one written by that same boot: pool 385927 ->
    491445, free landing at 1515/3130/1983 MiB against guard floors of
    1772/1964/2414 -- below the floor on two of three ranks, which is the exact
    condition this term exists to prevent. A term that only fires on a first
    boot is worse than no term, because the second boot looks like the fixed
    one.

    So the charge is applied to whatever the two solves settle on. It can only
    ever REDUCE the result, which is why it is safe to apply after a min that
    already never grows it.
    """
    budget = int(budget_bytes)
    # #678: SOLVE THE FLOOR WHERE THE RECORD ALLOWS IT, approximate only where
    # it does not. floor_allowed_tokens returns the largest id space whose
    # RESTING FREE COLUMN still holds the floor -- an equality -- and a
    # constraint solved exactly needs no charge against the budget at all.
    floor_allowed = floor_allowed_tokens(cell_bytes, reserve, arming_floor_bytes)
    if floor_allowed is None:
        # No free column on record (cold, no cell, or a pre-#678 record): fall
        # back to the subtrahend, which is the previous arithmetic exactly.
        #
        # SAID OUT LOUD, because "which path sized this pool" is the first
        # question any acceptance of #678 has to answer and reconstructing it
        # from the pool number alone is an investigation. One line, one grep.
        logger.info(
            "%s ARMING FLOOR APPROXIMATED (no at-rest free column on record: "
            "%s): the pool is sized by the budget subtrahend, the pre-#678 "
            "arithmetic. A boot that WROTE a column at rest makes the next one "
            "solve instead.",
            LOG_PREFIX,
            reserve.provenance,
        )
        charge = arming_floor_subtrahend_bytes(
            arming_floor_bytes, seam_solve_reserved_free_bytes(reserve)
        )
        if not reserve.active or int(cell_bytes) <= 0:
            return max(0, budget - charge), 0
        allowed = seam_allowed_tokens(
            cell_bytes,
            reserve,
            abandon_is_survivable=abandon_is_survivable,
            received_layers=received_layers,
        )
        return max(0, min(budget, allowed * int(cell_bytes)) - charge), allowed

    # TWO CONSTRAINTS, ONE MIN, AND NEITHER IS SUBTRACTED FROM THE OTHER. The
    # seam must be fundable AND the card must rest above its arming floor;
    # they bind on different ranks at different vectors, so the pool is the
    # smaller of the two id spaces rather than one adjusted by the other.
    seam_allowed = seam_allowed_tokens(
        cell_bytes,
        reserve,
        abandon_is_survivable=abandon_is_survivable,
        received_layers=received_layers,
    )
    allowed = min(int(seam_allowed), int(floor_allowed))
    logger.info(
        "%s ARMING FLOOR SOLVED from the at-rest free column: %d MiB free at "
        "an id space of %d -> this rank permits %d tokens; the seam solve "
        "permits %d; the binding one is %s at %d tokens. No budget subtrahend "
        "was applied -- the constraint is an equality, not an adjustment.",
        LOG_PREFIX,
        int(reserve.free_at_measure_bytes) >> 20,
        int(reserve.id_space),
        int(floor_allowed),
        int(seam_allowed),
        "the FLOOR" if floor_allowed <= seam_allowed else "the SEAM",
        allowed,
    )
    return min(budget, allowed * int(cell_bytes)), allowed


def _worst_case_fixed_bytes(runtime, direction: str) -> Tuple[int, int]:
    """(arena_tail, draft_restore) the seam WILL have to commit, not the
    amount pending right now.

    MEASURED IN THE WRONG PHASE IS MEASURED WRONG (boot F, 2026-08-12). Both
    of the runtime's live accessors are state readings:
    ``pending_tail_bytes`` is ``want - committed``, and
    ``pending_restore_bytes`` returns 0 unless the drafter is CURRENTLY
    spilled. At the first round the instance sits in its boot phase (PP) with
    the arena fully committed and nothing spilled, so both read ZERO -- and a
    reserve of zero is exactly the sizing that wedged boot E, whose refusal
    named 464 MiB. That 464 MiB is the tail rung 3 releases on ENTERING TP,
    which is a state this measuring point never sees.

    So the sizing term is the commit the seam would face from the OTHER
    phase: the arena span above the TP layout (what rung 3 releases and the
    tp->pp leg must take back) and the drafter's whole payload (what rung 2
    releases and the pp->tp leg must take back). Both are static layout
    quantities, which is what makes them safe to read at any time.
    """
    scheduler = getattr(runtime, "_census_scheduler", None)
    stacks = getattr(scheduler, "phase_flip_stacks", None) if scheduler else None
    if stacks is None:
        return 0, 0

    arena_tail = 0
    if direction == "tp_to_pp":
        try:
            # What rung 3 releases in TP = the arena span the PP layout needs
            # above what the TP layout does. Zero on a rank whose TP layout is
            # the larger one, which is where the "PP is always bigger"
            # assumption killed three ranks at the first flip.
            high_water = int(stacks.refill_high_water_bytes())
            arena_tail = max(0, high_water - int(stacks.layout_tp.total_bytes))
        except Exception:
            arena_tail = 0

    draft_restore = 0
    if direction == "pp_to_tp":
        try:
            from sglang.srt.managers.phase_flip_spill import carrier_of

            carrier = carrier_of(getattr(stacks, "draft_worker", None))
            if carrier is not None:
                draft_restore = int(carrier.payload_bytes)
        except Exception:
            draft_restore = 0
    return arena_tail, draft_restore


def measure_at_rest(runtime) -> Tuple[int, int, float, str, int]:
    """(arena_fixed, fixed, per_row, detail, worst_leg_fixed) for this boot.

    FOUR VALUES SINCE #656 MERGE-R9 12.4, not three. The arena tail and the
    drafter's restore used to be folded into one ``max`` here, matching the
    runtime's own; they are returned separately now because they combine
    differently -- see :class:`SeamReserve` for which is which and why.

    Computed from the runtime's OWN methods against an EMPTY live set, so the
    number recorded here is the number the gate will check -- not a parallel
    model of it that can drift. ``build_phase_flip_transition`` already
    handles a zero-row slot tensor (every send/recv guard evaluates false),
    which is what makes "at rest" expressible rather than approximated.

    ``_arena_tail_bytes`` and ``_draft_restore_bytes`` read static layout and
    carrier counters; ``_flip_waves`` is a pure function of the layer map and
    the vector; ``src.num_rows`` is the pool's physical row CAPACITY, not the
    live count. So none of this needs a round to have run, allocates
    anything, or performs a collective.
    """
    import torch

    from sglang.srt.layers.dcp.phase_flip_plan import build_phase_flip_transition

    empty = torch.empty(0, dtype=torch.int64)
    # The id space the sizer solves for: the PP allocator's capacity, which is
    # what every "T tokens" in this module means.
    scheduler = getattr(runtime, "_census_scheduler", None)
    id_space = max(1, int(getattr(scheduler, "max_total_num_tokens", 0) or 0))
    fixed = 0
    arena_fixed = 0
    worst_leg = 0
    per_row = 0.0
    parts = []
    for direction in ("pp_to_tp", "tp_to_pp"):
        src, dst = runtime._src_dst(direction)
        waves = runtime._flip_waves(direction)
        tr = build_phase_flip_transition(
            empty,
            runtime._map,
            runtime._n_layers,
            runtime._vec,
            runtime._rank,
            direction,
        )
        total = int(runtime._staging_bytes(tr, direction, src, dst, waves))
        slack = int(runtime._backing_slack_bytes(direction, src, dst, waves))
        arena, draft = _worst_case_fixed_bytes(runtime, direction)
        # SPLIT, NOT MAXED (#656, MERGE-R9 12.4). ``d_fixed = max(arena,
        # draft)`` was the sizer's mirror of the runtime's flat max(), and it
        # inherited the same defect: whichever of the two was smaller vanished
        # from the floor entirely. The arena tail is committed at the
        # PRE-cutover seam and is therefore additive against everything else
        # the seam holds; the drafter's restore is not. Kept as two numbers so
        # the solver can combine them the way the walk says.
        d_fixed = draft
        # NORMALISED BY THE ID SPACE, NOT BY src.num_rows (boot F).
        #
        # The sizer's T is the GLOBAL pool -- the id space every rank shares.
        # ``src.num_rows`` is this rank's PHYSICAL row count, and under the TP
        # layout that is its token SHARE of the id space, so tp_to_pp divided
        # by roughly T/3 and reported a coefficient ~3x too large (measured on
        # boot F: 5393.8 B/row against a pp_to_tp reading of 2360.7 for the
        # same 1396 MiB of slack). Dividing both directions by the id space
        # makes the two comparable and makes the number mean what the sizer
        # multiplies it by.
        d_per_row = float(slack) / float(id_space)
        fixed = max(fixed, d_fixed)
        arena_fixed = max(arena_fixed, arena)
        # #678: AND THE WORST SINGLE LEG, which is not the sum of the two
        # maxima above. Those are maxed independently and, on this rig, by
        # DIFFERENT directions -- the arena tail only on tp_to_pp, the
        # drafter's restore only on pp_to_tp -- so their sum prices a commit
        # no leg makes. A gate that arms per flip needs what one flip draws.
        worst_leg = max(worst_leg, int(arena) + int(d_fixed))
        per_row = max(per_row, d_per_row)
        mib = 1 << 20
        parts.append(
            f"{direction}: staging {total / mib:.0f} MiB now, seam commit "
            f"{(arena + d_fixed) / mib:.0f} MiB (arena tail {arena / mib:.0f} "
            f"ADDITIVE + draft restore {draft / mib:.0f} vs slack), wave "
            f"slack {slack / mib:.0f} MiB over an id space of {id_space} = "
            f"{d_per_row:.1f} B/token [this rank holds {int(src.num_rows)} rows]"
        )
    return arena_fixed, fixed, per_row, "; ".join(parts), worst_leg


def _band_floor_bytes(law_bytes: int) -> int:
    """The band floor in bytes, from the one declaration of the tolerance.

    Falls back to the law itself if the band cannot be read: reserving the
    centre sizes a smaller pool, which costs tokens, while reserving less than
    the floor would size a pool whose seam breaches. The fallback goes the
    costly-but-lawful way.
    """
    try:
        from sglang.srt.managers.corridor_guard import CORRIDOR_BAND_FRACTION

        # ROUNDED IN MiB, then converted -- the runtime staging reserve does
        # the same, and the two must be the SAME number rather than two
        # roundings of one intention that differ by a fifth of a MiB.
        mib = 1 << 20
        floor_mib = int(round((int(law_bytes) / mib) * (1.0 - CORRIDOR_BAND_FRACTION)))
        return max(0, floor_mib) * mib
    except Exception:  # pragma: no cover - sizing must never fail on this
        return int(law_bytes)


def _rung_fundable_for_seam(scheduler, arena_fixed: int, fixed: int) -> int:
    """Bytes the KV rung can put toward the seam, bounded by the FIXED floor.

    THE BOUND IS THE POINT. The rung's honest capacity is "everything above
    the live high-water mark", which at rest is nearly the whole pool -- and
    sizing against that would grow the id space without limit and then fail
    under a live set the measurement never saw. So it may cover at most the
    seam's FIXED floor: the weights-arena tail plus the drafter's restore.

    Those two are one-shot commits at the cutover, which is exactly the moment
    the rung can release rows and exactly the moment it takes them back again;
    on this rig they are also the dominant term (815 MiB and 1456 MiB of arena
    tail on the two 3080 ranks). The PER-ROW slack is deliberately NOT covered:
    it is held across the whole wave walk and it scales with the very pool this
    would be growing, which is the shape that runs away.

    ASKED AS A PREDICATE, NOT OF AN OBJECT, and that is an ordering fact rather
    than a preference. The relief is installed at the first corridor gate,
    which happens AFTER both the pool sizing and this measurement -- measured
    on this rig, the MEASURED line is five lines above "KV backing relief is
    available" in the same millisecond. Asking the scheduler for the object
    here returns None every time, and a term that is silently zero is worse
    than one that is absent. So the question asked is the one that can be
    answered now: WILL there be a rung able to pay
    (:func:`kv_backing_relief.rung_can_pay`, the same disqualifiers the
    provider applies), and if so the fixed floor need not also be held in idle
    VRAM.

    Returns 0 whenever no such rung will exist -- the env switch off, no
    allocator, a chunkless arena, unreadable geometry. Zero reproduces the
    previous sizing exactly, which is the direction a term that GROWS a pool
    has to fail in, and it is the can-fail arm: with
    ``SGLANG_KV_BACKING_RELIEF=0`` the reserve charges what it always charged.
    """
    from sglang.srt.managers.kv_backing_relief import rung_can_pay

    ceiling = max(0, int(arena_fixed)) + max(0, int(fixed))
    if ceiling <= 0:
        return 0
    try:
        can_pay = bool(rung_can_pay(scheduler))
    except Exception as e:
        logger.warning(
            "%s could not decide whether a KV rung will be able to pay at the "
            "seam (%s); sizing as if it cannot",
            LOG_PREFIX,
            e,
        )
        return 0
    if not can_pay:
        logger.info(
            "%s no KV rung will be able to return bytes at the seam, so the "
            "%.0f MiB fixed floor stays charged to the pool exactly as before",
            LOG_PREFIX,
            ceiling / (1 << 20),
        )
        return 0
    logger.info(
        "%s a KV rung will be able to return backing at the seam, so the "
        "%.0f MiB fixed floor (%.0f arena tail + %.0f draft restore) is "
        "counted as spendable instead of held free. The per-row slack stays "
        "charged: it is held across the whole wave walk and scales with the "
        "pool this term would grow.",
        LOG_PREFIX,
        ceiling / (1 << 20),
        max(0, int(arena_fixed)) / (1 << 20),
        max(0, int(fixed)) / (1 << 20),
    )
    return ceiling


def measure_and_record(scheduler, runtime) -> None:
    """Measure this boot's seam and leave the record for the next one.

    Never raises: a bookkeeping write must not be able to take down an
    instance that is otherwise serving.
    """
    if not seam_reserve_enabled():
        return
    try:
        arena_fixed, fixed, per_row, detail, worst_leg = measure_at_rest(runtime)
    except Exception as e:
        logger.warning("%s could not measure the seam at rest: %s", LOG_PREFIX, e)
        return

    mib = 1 << 20
    try:
        import torch

        free_bytes = int(torch.cuda.mem_get_info()[0])
    except Exception:
        free_bytes = 0
    law = _corridor_law_bytes()
    rung_fund = _rung_fundable_for_seam(scheduler, arena_fixed, fixed)
    # THE MEASURED POSITION. Taken with every unnamed post already resident
    # -- activation reserve, capture peak, arena, TP stack, carve-out -- so
    # the next boot's correction needs no model of any of them.
    #
    # #662: AND FREE VRAM IS NO LONGER THE ONLY THING THAT CAN PAY. The KV
    # relief rung returns unoccupied backing at the seam and takes it back
    # after the cutover, which is bytes arriving exactly when the commit needs
    # them. Counting only the free column is what made the pool shrink until
    # enough VRAM sat idle to cover the seam -- the ~12.7 GiB this ticket
    # exists to remove. See _rung_fundable_for_seam for what it may cover and
    # why that bound is the fixed floor and not a token more.
    # #662 JOINT CONSTRAINT: the seam's demand is a term in the POOL SOLVE,
    # not something the flip discovers at runtime.
    #
    # Sizing reserved the law CENTRE here while the runtime seam reserves the
    # band FLOOR, and the two disagreeing is what let a pool gain eat the
    # seam. Causally, on this rig: deriving the arming floor from the band
    # recovered ~205 MiB/rank, that went into pool, device 0's under-load free
    # fell from ~2.5 GiB to ~1.9-2.0 GiB, and the seam's 1059 MiB staging then
    # missed by 59 MiB and latched "unfundable". The earlier boot flipped
    # BECAUSE its pool was smaller -- pool-fill and seam fundability were being
    # maximised as separate criteria and collided.
    #
    # Subtracting the BAND FLOOR makes the solve reserve exactly
    # ``band_floor + seam_draw``: at the solved id space the resting free
    # column is the floor plus what a cutover draws, so the seam can always be
    # staged and the dip bottoms out at the floor -- which is lawful, because
    # the verdict is the continuous minimum against the floor. It is the same
    # number the runtime staging reserve now uses, so sizing and the gate can
    # no longer disagree, and every future pool gain is taken from what is
    # left AFTER the seam's demand rather than out of it.
    have = max(0, free_bytes - _band_floor_bytes(law)) + rung_fund
    id_space = max(0, int(getattr(scheduler, "max_total_num_tokens", 0) or 0))
    path = write_seam_reserve(
        scheduler.server_args,
        int(runtime._rank),
        fixed,
        per_row,
        detail,
        have_bytes=have,
        id_space=id_space,
        arena_fixed_bytes=arena_fixed,
        worst_leg_fixed_bytes=worst_leg,
        free_at_measure_bytes=free_bytes,
        rung_fund_bytes=rung_fund,
    )
    logger.info(
        "%s MEASURED (rank %d): floor %.0f MiB (%.0f arena tail ADDITIVE "
        "+ %.0f draft restore vs the wave slack), %.1f B/row. %s. Recorded "
        "in %s for the next boot with this configuration.",
        LOG_PREFIX,
        int(runtime._rank),
        (arena_fixed + fixed) / mib,
        arena_fixed / mib,
        fixed / mib,
        per_row,
        detail,
        path,
    )

    # LIVE VERDICT for the boot that is running right now. The record helps
    # the NEXT boot; an operator watching this one needs to know today
    # whether its flips can be funded, because the failure mode is an
    # instance that answers /health and serves nothing (#656 boot E).
    try:
        spendable = have
        # THE WHOLE at-rest floor, both terms. Checking the alternative
        # term alone would let a boot whose arena tail is the binding
        # cost report that it can fund a flip it cannot.
        if spendable < arena_fixed + fixed:
            logger.error(
                "%s (rank %d): THIS BOOT CANNOT FUND ITS OWN FLIP. The seam "
                "needs %.0f MiB at rest and only %.0f MiB is spendable above "
                "the %.0f MiB corridor law (driver free %.0f MiB). Under "
                "strict phase purity the layout that cannot be reached will "
                "not run its work at all. Re-boot to pick up the record just "
                "written, or set %s.",
                LOG_PREFIX,
                int(runtime._rank),
                fixed / mib,
                spendable / mib,
                law / mib,
                int(free_bytes) / mib,
                ENV_FIXED_MIB,
            )
        else:
            logger.info(
                "%s (rank %d): seam fundable at rest -- needs %.0f MiB, has "
                "%.0f MiB above the corridor law.",
                LOG_PREFIX,
                int(runtime._rank),
                fixed / mib,
                spendable / mib,
            )
    except Exception as e:  # pragma: no cover - diagnosis only
        logger.warning("%s live verdict unavailable: %s", LOG_PREFIX, e)


def _corridor_law_bytes() -> int:
    from sglang.srt.managers.vram_dial import corridor_law_floor_bytes

    return int(corridor_law_floor_bytes())


def describe(reserve: SeamReserve, path: str) -> str:
    """The boot line. Emitted on EVERY flip boot, cold included (#188)."""
    mib = 1 << 20
    if reserve.provenance == PROVENANCE_DISABLED:
        return (
            f"seam reserve OFF ({ENV_ENABLE}); the pool is sized without a "
            f"flip-seam term, which is the sizing that produced an instance "
            f"that held the corridor and served nothing (#656 boot E)."
        )
    if reserve.provenance == PROVENANCE_OVERRIDE:
        return (
            f"seam reserve {reserve.fixed_bytes / mib:.0f} MiB from "
            f"{reserve.detail} -- an operator value, not this boot's "
            f"measurement. The measurement still lands in {path} for the "
            f"next boot."
        )
    if reserve.provenance == PROVENANCE_STORED:
        return (
            f"seam reserve {reserve.fixed_bytes / mib:.0f} MiB fixed + "
            f"{reserve.per_row_bytes:.1f} B/row, MEASURED BY A PREVIOUS BOOT "
            f"({path}, written {reserve.written_at}). {reserve.detail} "
            f"Identical commands against a different record state will size "
            f"differently; pin --max-total-tokens for a reproducible pool."
        )
    return (
        f"seam reserve is COLD (no record at {path}): this boot sizes with NO "
        f"flip-seam term and may produce an instance whose flips cannot be "
        f"funded (#656 boot E). It measures the seam at the end of the flip "
        f"boot and writes the record, so the NEXT identical boot is the "
        f"seam-safe one. Watch for 'FLIP ABANDONED' on this boot."
    )


def staging_post_bytes(
    projected_ask_bytes: int = 0, entry_margin_bytes: int = 0
) -> int:
    """#771: the seam's MANDATORY staging ask, charged into the pool solve.

    THE THIRD TERM, and it is not either of the two above it. The seam reserve
    is what a flip COSTS WHILE IT RUNS and is only knowable by measurement. The
    arming floor is a LEVEL THE GATE COMPARES AGAINST, derived from the stated
    corridor law. This one is what the KV rung must be ABLE TO FUND for the
    seam to enter at all -- a per-rank, per-leg quantity that
    ``project_staging_bytes`` computes exactly, because "the plan's only
    live-set input is the slot count, and everything else is static layout".

    WHY THE POOL MUST CHARGE IT. ``collective_kv_backing_relief`` is
    deliberately all-or-nothing: it grants nothing when the mandatory ask
    exceeds what the rung can return above its admission floor, on the
    reasoning that laundering a short ask into a seam that then does not fit
    is worse than a clean abandon. That rule is correct and is not touched
    here. But it turns any sizing shortfall into a PERMANENT one: measured on
    this rig, the rung reported it could return 1902-2037 MiB, granted 0, and
    logged ``evicted 0 rows over 0 shrinks so far`` 72 times in a single boot
    while the flip never happened. Sizing the pool so ``mandatory <=
    fundable`` holds by construction is what keeps that rule from ever firing.

    AN UNKNOWN ASK CHARGES ZERO, LOUDLY -- never a fabricated constant. A
    projection that could not be computed is a missing measurement, and
    inventing a number here would be the ``derived_provenance`` defect this
    module already names: it would make a guess indistinguishable from the
    exact figure ``project_staging_bytes`` returns. The caller logs which of
    the two it charged.
    """
    ask = max(0, int(projected_ask_bytes or 0))
    if ask <= 0:
        return 0
    return ask + max(0, int(entry_margin_bytes or 0))


def pool_flip_posts_bytes(
    arming_floor_bytes: int = 0, staging_post_bytes_: int = 0
) -> int:
    """Every flip term the POOL SOLVE owes, summed in one place.

    ADDITIVE, because the two are needed at the same instant and for different
    things: the gate refuses unless the arming floor is free, and the rung then
    has to find the staging ask on top of it. A pool that reserved only the
    larger of the two would satisfy whichever term is quoted in the log and
    still abandon on the other.
    """
    return max(0, int(arming_floor_bytes or 0)) + max(0, int(staging_post_bytes_ or 0))
