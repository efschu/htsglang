"""#656 spec item 15a/15b: SPILL BEFORE THE ALLOCATION, not after the alarm.

WHY THIS IS NOT ``kv_pressure_runtime``
---------------------------------------
``KvPressureRuntime`` (#287) already drives a spill ladder, and it already
takes a ``spill_fn``. It is nonetheless the wrong shape for item 15a, and the
difference is the whole point of the user's order:

    "SPILL-BEFORE-ALLOC -- check AT the allocation (free-X >= 1024, otherwise
     spill synchronously first), not reactive threshold watching"

(Translated from the user's German; the original wording is in the commit that
introduced this module.)

``on_round`` observes occupancy at a ROUND BOUNDARY and reacts. Between two
boundaries an allocation can take the card below the floor, and the corridor
law is a CONTINUOUS minimum -- it is broken by the trough, not by the average,
and a sampler at 100 ms will see a dip that a per-round controller never
notices. Worse, the allocation that most needs guarding is the seam's
``commit_range``, which does not happen on a round boundary at all: it happens
inside the flip's no-return region, and it has killed this instance before
(``cuMemCreate failed: CUDA_ERROR_OUT_OF_MEMORY``, 2026-08-09).

So this module is a GATE AT THE CALL SITE, synchronous, returning only once
the headroom exists or once it is provably unobtainable. It does not replace
the pressure ladder; the ladder still handles slow, planned residency changes.
This handles the instant a specific allocation is about to happen.

TWO WATERMARKS (item 15b), and why one is not enough
----------------------------------------------------
Freeing exactly enough to clear the floor guarantees that the NEXT allocation
of any size spills again. That is thrashing: every allocation pays a spill,
and the spill/restore pair costs far more than the allocation it enabled. So
the gate arms at the FLOOR and frees up to ``floor + delta``:

    arm at   free - want <  floor           (the corridor law's own bound)
    free to  free - want >= floor + delta   (headroom for the next few asks)

``delta`` is deliberately a byte figure and not a percentage: the corridor law
is stated in absolute MiB, and a percentage of a 20 GiB card and a 32 GiB card
would arm at different absolute headrooms on hardware whose floor is the same.

WHAT A PROVIDER IS
------------------
A payload class that can give memory back synchronously: the draft-weight
carrier, idle GDN/mamba slots at bs<4, the inactive layout's arena tail,
session KV via kvso. Each registers a callable that frees "up to N bytes and
returns how many it actually freed", plus a COST RANK. The gate spends the
cheapest first, which is the reclaim-ordering law this chain already follows
elsewhere: coldest and cheapest-to-restore goes first, hot data last.

ITEM 15c IS A PROVIDER, NOT A SPECIAL CASE. "If everything resident is hot,
keep computing over the host tier (kvso) -- the price is tempo, never a
corridor breach" is expressed here as the most expensive provider in the
order. When the cheap ones are exhausted the gate reaches it, pays the
latency, and still does not breach. The gate NEVER returns "allocate anyway".

FAILURE IS A REFUSAL, NOT A BREACH
----------------------------------
If every provider is exhausted and the headroom still is not there, the gate
says so and the caller must not allocate. At the seam that means abandoning
the flip -- which the affordability verdict already knows how to do
unanimously and for free -- rather than dying inside the no-return region.
A gate that shrugs and lets the allocation proceed would be worse than no
gate, because it would launder a corridor breach as a check that passed.

ITEM 16: THE CARDS FILL EVENLY, AND HOST RAM IS LAST
----------------------------------------------------
    "CARDS MUST FILL EVENLY BEFORE ANY HOST SPILL. Host spill happens ONLY
     when ALL three cards are at the floor -- never while one card binds and
     the others have headroom."

"Even" is defined on FREE HEADROOM, not on bytes held: this rig's cards are
32/20/20 GiB, so equal bytes would mean permanently unequal pressure and the
20 GiB cards would bind forever. The objective is therefore a water-filling
one over the per-card FREE column, and the metric that says whether it has
been achieved is the SPREAD of that column (:func:`free_spread_mib`), which
belongs in every corridor CSV so that "evenly filled" is provable rather
than asserted.

This splits the provider order into a TIER above the cost:

    RELIEF_LOCAL      give back memory NO payload owns. Torch's caching
                      allocator holds blocks no tensor uses; returning them
                      moves NVML's free column, moves no payload anywhere,
                      and costs only the re-``cudaMalloc`` of whatever asks
                      next. It levels nothing, so it is not a rebalance; it
                      spills nothing, so it is not a host tier. It is simply
                      free money and must be spent before anything that
                      moves a real payload.
    RELIEF_REBALANCE  make the free column more level. In TP this is the
                      uneven-DCP token vector (``distributed.corridor_vector``
                      already solves it against per-card corridor capacity):
                      steer tokens to the freest card, physical backing
                      follows the vector, VA stays geometric. In PP the KV is
                      layer-bound and cannot be token-steered, so levelling
                      means evacuating everything NOT layer-bound from the
                      heavy card -- the drafter (done), idle mamba slots,
                      arena slack, graph pools.
    RELIEF_PARK       park a cold payload in ANOTHER card's surplus free
                      space. Still VRAM, still fast, and it levels.
    RELIEF_HOST       system RAM. Gated: see below.

**Tier outranks cost, and that is the point.** Cost still orders providers
WITHIN a tier -- item 15's cheapest-first law is untouched -- but it may not
promote a host spill ahead of a rebalance merely because the host spill is
cheaper to execute. Those are answers to different questions.

**The host gate is a FLEET predicate.** A guard that sees only its own card
cannot distinguish "everything is full" from "I am the only one that is
full", and those two states have opposite correct answers. So the guard reads
the whole free column and admits the host tier only when
:func:`fleet_is_level` holds. Without a fleet probe the host tier stays shut:
item 16 is a permission that must be proven, not assumed.

**Why the drafter is a REBALANCE and not a host spill, even though its bytes
land in host RAM.** The tier names what the action does to the free column,
not where the bytes go. Evacuating a non-layer-bound payload from the binding
card is precisely the PP levelling move the user prescribes ("drafter done;
mamba slots, arena slack, graph pools next"), and it is not the KV/cold spill
class item 16 gates. If a successor finds the user meant the destination
rather than the effect, the change is one ``tier=`` argument at the
registration site -- deliberately, so that it is a one-line decision and not
a refactor.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "CORRIDOR-GUARD"

_MIB = 1024 * 1024

#: The user's corridor law, in the unit the law is stated in. THE ONE
#: DECLARATION. Every other module that needs the law imports it from here
#: rather than repeating the literal -- ``corridor_trace.summary`` used to
#: default to its own ``1024`` and ``phase_flip_seam_census`` to another, so
#: three modules each held a private copy of a number the operator states
#: once.
CORRIDOR_LAW_MIB = 1024
#: Historical name, kept because it is the ``floor_mib`` default of the guard
#: and callers pass it positionally. It IS the law.
DEFAULT_FLOOR_MIB = CORRIDOR_LAW_MIB

#: How far above the LAW the gate starts working. The arming floor is not a
#: second policy number: it is the law plus what a seam is expected to draw
#: while it runs, and it must be DECLARED WITH the law or the two drift.
#:
#: #656 acceptance (2026-08-13) is what that drift costs. The gate armed at
#: 1536 MiB while the verdict was read against 1024, so a 512 MiB entry
#: allowance was being asked to cover a draw that the corridor sampler
#: measured at 1814-1852 MiB on GPU0 -- 3.6x. Five cutovers entered cleanly
#: through a gate that had no objection and took the card to 886 MiB, 138
#: below the law, and NOTHING IN THE PROCESS NOTICED: the runtime's own
#: verdict is read at 1024 and its own gate arms at 1536, so the breach fell
#: in the gap between the two numbers.
#:
#: The default keeps the shipped 512 MiB allowance, so a boot that does not
#: measure its draw behaves exactly as before. What changes is that the pair
#: is now derived and logged together, and an arming floor BELOW the law --
#: which would let the gate bless an allocation the law forbids -- is
#: refused instead of silently accepted.
DEFAULT_SEAM_ENTRY_RESERVE_MIB = 512


#: Overrides the law. Read HERE and nowhere else: three modules used to read
#: it with their own ``"1024"`` fallback (``kv_vmm_backing``,
#: ``phase_flip_seam_census``, and ``corridor_trace``'s default argument), so
#: the law could be moved for one of them and not the others -- a divergence
#: with no symptom until a breach is judged twice and answered differently.
LAW_ENV = "SGLANG_CORRIDOR_LAW_FLOOR_MIB"


def corridor_law_mib() -> int:
    """The law in force, in MiB. THE reader of :data:`LAW_ENV`.

    Read per call, not frozen at import: a rank can be told the law late
    (the ``kv_vmm_backing`` preempt path is reached long after import), and
    a value captured at import cannot be corrected by a boot that sets the
    variable afterwards.
    """
    raw = os.environ.get(LAW_ENV)
    if raw is None:
        return CORRIDOR_LAW_MIB
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return CORRIDOR_LAW_MIB


def corridor_law_bytes() -> int:
    """The law in force, in bytes."""
    return corridor_law_mib() * _MIB


#: THE CORRIDOR IS A BAND, NOT A POINT (user relaxation, 2026-08-15).
#:
#: The law was stated as "~1024 MiB free per card, best-filled", and both
#: halves were being read as an exact target: a sample at 1000 MiB counted as
#: a breach and a card resting at 2500 MiB counted as over-filled. Neither
#: reading is what the number is for. The operator's relaxation makes the
#: tolerance explicit at +-20 %, which is what lets the boot-time gate and the
#: planner solve be simple: a solve that has to land on a point has no
#: feasible region, and one that has to land in a band does.
#:
#: The centre stays the target -- the self-correcting margin pulls back to it,
#: because a mechanism that aims at the edge of its own tolerance has none.
#: The FLOOR is the verdict: below it is a breach, and nothing inside the band
#: is. Measured consequence on this rig: the cutover transient at the
#: `weights_refill` stage bottomed at 895-935 MiB on gpu0, which is inside the
#: band, so that class stops being chased.
CORRIDOR_BAND_FRACTION = 0.20


def corridor_band_floor_mib() -> int:
    """Below this is a breach. The law minus its tolerance."""
    law = corridor_law_mib()
    return int(law - law * CORRIDOR_BAND_FRACTION)


def corridor_band_ceiling_mib() -> int:
    """Above this at REST is over-filled: VRAM buying no tokens.

    The second half of "best-filled", and the one a boot-time gate needs in
    order to refuse a configuration that leaves gibibytes idle.

    WHAT "BUYING NO TOKENS" MEANS, because the flip made it ambiguous (#784).
    This ceiling grades the USER-FREE column, not the raw NVML free field. An
    armed phase-flip boot holds an arming floor above the band floor on every
    rank; those MiB are working capital the guard actively defends so a flip
    can arm, so they buy the flip and are not idle. Grade
    :func:`net_free_mib`, never the raw reading -- see that function for why
    the raw comparison is unsatisfiable by construction.

    ROUNDED, NOT TRUNCATED. ``int()`` gave 1228 here while the acceptance
    verdict (``corridor_verdict_774.sh``) computed ``int(round(...))`` = 1229,
    so the code and the gate that decides pass/fail disagreed by 1 MiB on the
    number under test. Rounding is also the symmetric choice: the floor's
    ``int(819.2)`` and ``round(819.2)`` are the same 819, so only this side
    ever moved.
    """
    law = corridor_law_mib()
    return int(round(law + law * CORRIDOR_BAND_FRACTION))


def committed_arming_mib(arming_mib: int) -> int:
    """The part of an arming floor that is CHARGED, not free.

    An arming floor is ``corridor_band_floor_mib() + seam draw + margin``. The
    band floor half belongs to the user -- it is the external headroom the
    corridor law promises. Everything ABOVE it is internal demand: working
    capital the flip spends while it runs, which the guard reclaims and spills
    to defend. Under the reserve semantics a reserve is the user's external
    free space and internal demand is booked exactly, in the ledger, so this
    part is a ledger post and must not be graded as free VRAM.

    Returns 0 for any floor at or below the band floor: such a rank holds
    nothing on the flip's behalf. Never negative.
    """
    return max(0, int(arming_mib) - corridor_band_floor_mib())


def net_free_mib(free_mib: int, arming_mib: int) -> int:
    """The quantity the corridor band actually grades: free MINUS the credit.

    THE DEFECT THIS EXISTS FOR (#784). The shipped constants make the raw
    comparison unsatisfiable. The smallest arming floor any rig can ask for is
    ``819 + 512 + 192 = 1523`` MiB -- the band floor, the shipped seam entry
    allowance and the arming margin -- while the ceiling a boot is graded
    against is 1229 MiB. ``1523 > 1229`` on every rank, unconditionally, so an
    armed phase-flip boot was graded against a threshold its own arming rule
    forbade it to reach, and no gate said so: this module's only pair check
    refused an arming floor BELOW the law and never looked at the ceiling
    side, and :func:`corridor_band_ceiling_mib` had no production consumer.

    Crediting the committed part resolves the pair rather than widening the
    band: a rank resting exactly at its arming floor reads exactly the band
    floor and is in band, whatever its measured seam draw. This is the
    Option-A formula the acceptance verdict has documented since 2026-07-22
    (``net = free_min - sum(registered posts)``) and never implemented.

    IT CANNOT MANUFACTURE A PASS, which is the whole risk of the change. The
    credit is a rank's own floor and nothing more, so a rank resting far above
    that floor keeps every stranded MiB visible: boot instr9's ranks 0 and 1
    rest at 5229 / 6612 MiB against a 1523 MiB floor and still read 4525 /
    5908 MiB net, still far over the ceiling, still a failed acceptance. The
    credit makes the grading honest; the capacity defect stays the planner's
    to solve.
    """
    return int(free_mib) - committed_arming_mib(arming_mib)


def corridor_band_mib():
    """``(floor, centre, ceiling)`` -- the whole band in one read."""
    return corridor_band_floor_mib(), corridor_law_mib(), corridor_band_ceiling_mib()


def arming_floor_mib(
    seam_entry_reserve_mib: int = DEFAULT_SEAM_ENTRY_RESERVE_MIB,
    law_mib: int = CORRIDOR_LAW_MIB,
) -> int:
    """The gate's watermark, derived from the law it protects.

    ``seam_entry_reserve_mib`` is the MEASURED draw a seam makes while it
    runs, where a measurement exists; the default is the shipped allowance.
    Deriving it means an operator cannot raise one number and leave the
    other behind, which is the failure this function exists to make
    impossible.

    BUILT ON THE BAND FLOOR, not the centre. The gate exists to keep the
    worst instant out of breach, and breach is now defined at
    :func:`corridor_band_floor_mib`. Arming from the centre instead would
    reserve the band's whole tolerance on top of the seam's draw on every
    card, for every boot -- roughly 205 MiB per rank here -- which is pool
    given up to protect a threshold that is not the verdict. The centre is
    what the self-correcting margin aims at; the floor is what the gate
    defends.
    """
    return corridor_band_floor_mib() + max(0, int(seam_entry_reserve_mib))


#: The one machine-readable line a boot emits so its acceptance verdict can
#: credit what the flip has committed. THE FORMAT LIVES HERE AND NOWHERE ELSE:
#: ``corridor_verdict_774.sh`` imports :func:`parse_corridor_posts` rather than
#: re-deriving the field names in awk, which is how the verdict and the runtime
#: came to disagree about the band ceiling by 1 MiB in the first place.
CORRIDOR_POST_PREFIX = "CORRIDOR-POST"


def format_corridor_post(
    rank: int, gpu_id: int, arming_mib: int, uuid: Optional[str] = None
) -> str:
    """One rank's committed arming reserve, as the verdict reads it.

    ``gpu_id`` is the PHYSICAL device index, not the process-local one: with
    per-rank ``CUDA_VISIBLE_DEVICES`` isolation every rank sees itself as
    ``cuda:0``, while the verdict enumerates NVML in physical order. The UUID
    rides along because NVML's enumeration order can shift between boots and
    driver states, so a reader that doubts the index has the identity.
    """
    line = (
        f"{CORRIDOR_POST_PREFIX} rank={int(rank)} gpu={int(gpu_id)} "
        f"arming_floor_mib={int(arming_mib)} "
        f"band_floor_mib={corridor_band_floor_mib()} "
        f"committed_mib={committed_arming_mib(arming_mib)}"
    )
    if uuid:
        line += f" uuid={uuid}"
    return line


def parse_corridor_posts(text: str) -> dict:
    """``{physical_gpu_id: committed MiB}`` from a boot log.

    TWO RULES, BOTH OF THEM LOAD-BEARING.

    *Co-resident ranks ADD.* Two ranks sharing a physical GPU each hold their
    own arming floor, so the card's committed column is the sum.

    *A rank is counted ONCE, at its latest reading.* A phase-flip boot
    resolves its arming floor in the PP phase and again when the TP stack is
    built, so the same rank emits more than once in a single log. Summing the
    LINES would charge the card twice and hand the verdict a pass it did not
    earn, which is the one failure mode a credit must not have. So the last
    line per rank wins -- including its card, so a rank that moved is not
    counted on both.

    A malformed line is skipped rather than raising: this runs inside an
    acceptance verdict, where dying on a truncated log would turn a gradeable
    boot into no verdict at all.
    """
    latest = {}
    for raw in str(text).splitlines():
        line = raw.strip()
        if CORRIDOR_POST_PREFIX not in line:
            continue
        fields = {}
        for token in line[line.index(CORRIDOR_POST_PREFIX) :].split():
            if "=" in token:
                key, _, value = token.partition("=")
                fields[key] = value
        try:
            rank = int(fields["rank"])
            gpu = int(fields["gpu"])
            committed = int(fields["committed_mib"])
        except (KeyError, TypeError, ValueError):
            continue
        latest[rank] = (gpu, max(0, committed))
    posts: dict = {}
    for gpu, committed in latest.values():
        posts[gpu] = posts.get(gpu, 0) + committed
    return posts


def check_threshold_pair(
    arming_mib: int,
    law_mib: int = CORRIDOR_LAW_MIB,
    band_floor_mib: Optional[int] = None,
    band_ceiling_mib: Optional[int] = None,
) -> None:
    """Refuse a pair that cannot mean what it says.

    An arming floor below the law lets the gate return "no reclaim needed"
    for an allocation that ends under the corridor -- the guard would be
    laundering a breach as a passed check, which is the one thing its own
    refusal message says it must never do.

    THE CEILING SIDE (#784). The floor check alone let a second unsatisfiable
    pair through in silence: a band whose floor sits above its own ceiling has
    no feasible region at all, so every boot graded under it fails an
    acceptance no configuration could pass. The band is inverted exactly when
    the fraction exceeds 1.0, which no shipped value does -- but the pair was
    accepted without a word when it happened, and "accepted in silence" is the
    same failure mode the floor side exists to prevent.

    The arming floor itself is deliberately NOT compared against the ceiling.
    It exceeds it by construction on every rig (1523 > 1229 with the shipped
    constants) and that is not an error: the excess is charged working
    capital, credited by :func:`net_free_mib`, not idle VRAM. Refusing it here
    would refuse every armed flip boot.

    ``band_floor_mib`` / ``band_ceiling_mib`` override the derived band for
    tests that need to construct an inverted one; production passes neither.
    """
    floor = corridor_band_floor_mib() if band_floor_mib is None else int(band_floor_mib)
    ceiling = (
        corridor_band_ceiling_mib()
        if band_ceiling_mib is None
        else int(band_ceiling_mib)
    )
    if floor > ceiling:
        raise ValueError(
            f"corridor band is INVERTED: floor {floor} MiB is above ceiling "
            f"{ceiling} MiB, so the band has no feasible region and no boot "
            f"can ever be graded against it. The band is the law "
            f"{law_mib} MiB plus and minus {CORRIDOR_BAND_FRACTION:.0%}; a "
            f"fraction above 100 % inverts it."
        )
    if int(arming_mib) < floor:
        raise ValueError(
            f"corridor arming floor {arming_mib} MiB is BELOW the corridor "
            f"law {law_mib} MiB. The gate would clear allocations the law "
            f"forbids and no refusal would ever fire. The arming floor is "
            f"the law PLUS the seam's expected draw, never less than it."
        )


#: How far ABOVE the floor a reclaim frees, so the next few allocations do not
#: each pay a spill. 256 MiB is one seam's worth of slack on this rig's
#: measured seam trough (1196 MiB free at the tightest instant, floor 1024).
DEFAULT_DELTA_MIB = 256

#: Relief tiers (item 16). Lower is tried first, unconditionally: the tier
#: outranks the cost, so a cheap host spill can never overtake an expensive
#: rebalance. Cost still orders providers within one tier.
RELIEF_LOCAL = -1  # give back memory NO payload owns (torch's allocator cache)
RELIEF_REBALANCE = 0  # level the free column: token vector, evacuate the heavy card
RELIEF_PARK = 1  # park a cold payload in another card's surplus VRAM
RELIEF_HOST = 2  # system RAM -- only once every card is at the floor

_TIER_NAMES = {
    RELIEF_LOCAL: "local",
    RELIEF_REBALANCE: "rebalance",
    RELIEF_PARK: "park",
    RELIEF_HOST: "host",
}


def free_spread_mib(free_column) -> int:
    """max - min over the per-card free column, in MiB. The levelness metric.

    This is the number that goes in the corridor CSV. A run whose spread
    stays small was evenly filled; a run whose spread is thousands of MiB was
    not, whatever its minimum says. The corridor law's minimum and item 16's
    spread are independent axes and a CSV needs both.
    """
    col = [int(f) for f in free_column]
    if not col:
        return 0
    return int((max(col) - min(col)) // _MIB)


def water_fill_targets(free_column):
    """The equal-free-headroom targets for a fleet, in bytes.

    The objective is EQUAL FREE, so every card's target is the mean of the
    column regardless of its total size. Returned as a vector (rather than a
    scalar) because the caller subtracts each card's current free to get the
    signed transfer it should aim for: negative means "this card should give
    bytes up", positive means "this card can take them".
    """
    col = [int(f) for f in free_column]
    if not col:
        return []
    mean = sum(col) // len(col)
    return [mean] * len(col)


def water_fill_transfers(free_column):
    """The signed PAYLOAD move each card needs to reach equal free, in bytes.

    Positive means **this card should SHED that many bytes of payload**;
    negative means it can ABSORB them. The vector sums to zero, because a
    transfer is conserved.

    WHY THIS EXISTS ALONGSIDE :func:`water_fill_targets` RATHER THAN INSTEAD
    OF IT. The targets function is the objective and this is one subtraction
    away from it -- computed FROM it here, so there is one derivation and not
    two. What it adds is a sign convention a reader can act on without a
    decoder ring: ``water_fill_targets`` states its sign in terms of FREE
    bytes, and "this card should give bytes up" reads backwards to anyone
    thinking about where the payload goes, because the card that should give
    FREE bytes up is the card that should TAKE payload.

    IT IS NOW BOTH AN INSTRUMENT AND AN OBJECTIVE, AND THE SPLIT MATTERS.
    Item 16's first relief stage is "redistribute onto the card with the most
    headroom", and that has two halves with different fates:

    * the SHED half -- get the bytes off the card that binds -- is actuated,
      by :meth:`CorridorGuard.lend_to_level` driven from
      ``corridor_rebalance.RebalanceLender``. This function supplies its
      bound: the lender frees up to the transfer named here and no further.
    * the ABSORB half -- put those same bytes on the peer card -- is still
      missing, and structurally so for KV: the DCP owner rule makes a token's
      row a pure function of the token vector (``layers/dcp/owner.py:348``),
      the vector is boot-constant (``phase_flip_runtime.py:866``), and the
      one actuator that may change it needs a fully idle instance
      (``kv_reshard.py:781``). In the PP phase KV is layer-bound anyway.

    So a positive transfer here is a move that WILL be attempted from the
    tight card, and a negative one is capacity that stays unusable to its
    peers. Before HANDOFF_679 ``water_fill_targets`` had exactly one caller
    in the tree and it was a test: the objective was computed by nothing.

    A spread figure alone does not tell a successor whether the actuator is
    worth building; a per-card "shed 900 MiB / absorb 900 MiB" does.
    """
    col = [int(f) for f in free_column]
    if not col:
        return []
    return [t - f for t, f in zip(water_fill_targets(col), col)]


def describe_water_fill(free_column) -> str:
    """One human-readable clause naming the levelling move, or ''.

    Empty when the fleet is unknown or already level, so a log line does not
    grow a clause that says nothing.
    """
    transfers = water_fill_transfers(free_column)
    if not transfers or all(t == 0 for t in transfers):
        return ""
    shed = max(range(len(transfers)), key=lambda i: transfers[i])
    absorb = min(range(len(transfers)), key=lambda i: transfers[i])
    if transfers[shed] <= 0:
        return ""
    return (
        f"item 16 water-fill wants card {shed} to SHED "
        f"{transfers[shed] / _MIB:.0f} MiB onto card {absorb} "
        f"(which can absorb {-transfers[absorb] / _MIB:.0f} MiB). The SHED "
        "half is actuated by the rebalance lender, bounded by this figure; "
        "the ABSORB half has no actuator (the DCP owner rule fixes KV "
        "ownership to a boot-constant token vector), so any residue is "
        "levelling NOT performed"
    )


def fleet_is_level(free_column, floor_mib: int, delta_mib: int) -> bool:
    """True when EVERY card sits at the floor, i.e. host RAM is permitted.

    "At the floor" is a band and not a point -- a card is never exactly at
    1024 MiB, so a point predicate would be unreachable and the host tier
    would be dead code. The band is the same ``floor + delta`` upper
    watermark the gate frees to, which keeps one number in charge of both.

    An EMPTY column is not level. No evidence is not evidence.
    """
    col = [int(f) for f in free_column]
    if not col:
        return False
    ceiling = (int(floor_mib) + int(delta_mib)) * _MIB
    return all(f <= ceiling for f in col)


class CorridorBreachRefused(RuntimeError):
    """The allocation cannot be made without breaking the corridor law.

    Raised only when the caller asked for ``raise_on_refusal``; the default
    is a returned verdict, because the seam's caller wants to abandon the
    flip cleanly rather than unwind an exception from inside a cutover.
    """


@dataclass(order=True)
class _Provider:
    # Ordered by TIER first (item 16), then by cost (item 15). The two-key
    # sort is the whole enforcement: a cheap host spill sorts after every
    # rebalance no matter how expensive the rebalance is.
    tier: int
    cost: int
    name: str = field(compare=False)
    free_up_to: Callable[[int], int] = field(compare=False)


@dataclass
class GuardResult:
    """What the gate did, in bytes, so a caller can log or account for it."""

    ok: bool
    free_before: int
    free_after: int
    want: int
    reclaimed: int
    used_providers: Tuple[str, ...]
    detail: str = ""
    #: The verdict HELD but the residual sits under the corridor law. Carried
    #: so the caller can WARN (user decision 2026-08-16: the law is advisory
    #: at seam entry, and a dip has to be sayable to be warned about).
    law_breached: bool = False

    @property
    def reclaimed_mib(self) -> float:
        return self.reclaimed / _MIB


class CorridorGuard:
    """Synchronous spill-before-alloc gate for ONE device.

    One guard per rank per device. The rank's own device is named
    explicitly rather than read from ``current_device()``: under
    ``--rank-gpu-id`` each worker sees exactly one card, but an absolute
    memory figure should still say which card it is about.
    """

    def __init__(
        self,
        device_index: int,
        *,
        floor_mib: int = DEFAULT_FLOOR_MIB,
        delta_mib: int = DEFAULT_DELTA_MIB,
        probe: Optional[Callable[[], int]] = None,
        fleet_probe: Optional[Callable[[], Sequence[int]]] = None,
        law_floor_mib: Optional[int] = None,
    ) -> None:
        self.device_index = int(device_index)
        self.floor_mib = int(floor_mib)
        # THE ARMING WATERMARK AND THE LAW ARE NOT THE SAME NUMBER, and
        # conflating them wedged this instance on 2026-08-10.
        #
        # ``floor_mib`` is a POLICY target: where the gate starts working and
        # how far it frees. ``law_floor_mib`` is the user's corridor law, the
        # only thing a REFUSAL may be justified by. When they are equal (the
        # default) nothing changes. When the policy floor is raised -- for a
        # proof run, or by a future per-card policy -- a shared threshold
        # makes the gate refuse allocations that the law permits perfectly
        # well, and on the pp->tp leg that is not a conservative choice: it
        # is a DEADLOCK. Strict purity forbids decode in PP, so a permanently
        # refused pp->tp flip means decode never runs again, and nothing in
        # the PP phase can free the memory that would end the refusal.
        # Measured: 411 abandons, 0 requests completed in 6 minutes, /health
        # 503 while every rank was alive and logging normally.
        self.law_floor_mib = int(
            law_floor_mib if law_floor_mib is not None else floor_mib
        )
        self.law_floor_bytes = self.law_floor_mib * _MIB
        self.delta_mib = int(delta_mib)
        self.floor_bytes = int(floor_mib) * _MIB
        self.delta_bytes = int(delta_mib) * _MIB
        self._probe = probe
        # Item 16's fleet predicate. Deliberately a per-card NVML read rather
        # than a collective: this gate runs inside the flip's no-return
        # region, and a collective there is a deadlock waiting for the one
        # rank that took a different branch. NVML sees every card regardless
        # of CUDA_VISIBLE_DEVICES, so no rank has to be asked.
        self._fleet_probe = fleet_probe
        self._providers: List[_Provider] = []
        self.arm_count = 0
        self.refuse_count = 0
        self.host_blocked_count = 0
        #: Times the host tier was admitted onto an UNLEVEL fleet because
        #: refusing would have deadlocked the caller. Every one of these is a
        #: levelling failure that item 16 wanted avoided, so the counter is
        #: the honest measure of how much the missing rebalance tier costs.
        self.host_forced_count = 0
        self.reclaimed_total = 0
        #: Item 16's REBALANCE lender (see :meth:`lend_to_level`). Counted
        #: apart from ``arm_count`` on purpose: an arm is an allocation that
        #: would have breached, a lend is relief taken BEFORE any allocation
        #: asked, and averaging the two would hide which one moved the trough.
        self.lend_count = 0
        self.lent_total = 0

    # -- registration ----------------------------------------------------

    def register(
        self,
        name: str,
        cost: int,
        free_up_to: Callable[[int], int],
        tier: int = RELIEF_REBALANCE,
    ) -> None:
        """Add a payload class the gate may spend.

        ``free_up_to(nbytes)`` must free AT MOST ``nbytes``, synchronously,
        and return the bytes it actually gave back to the DRIVER -- not to
        torch's cache. The corridor law is stated in NVML's free column, and
        a provider that only returns memory to the caching allocator has
        freed nothing the law can see.

        ``cost`` orders the spend: lower is cheaper to give up and to get
        back. Ties are resolved by registration order.

        ``tier`` (item 16) outranks ``cost`` entirely -- see the module
        docstring. It defaults to ``RELIEF_REBALANCE`` so that a provider
        written before tiers existed keeps working: defaulting to
        ``RELIEF_HOST`` would silently switch off relief that already works,
        while this direction forces the gated class to be declared on
        purpose, which is the way an omission should fall.
        """
        if not callable(free_up_to):
            raise TypeError(f"{LOG_PREFIX} provider {name!r} is not callable")
        if tier not in _TIER_NAMES:
            raise ValueError(
                f"{LOG_PREFIX} provider {name!r}: unknown relief tier {tier!r}, "
                f"expected one of {sorted(_TIER_NAMES)}"
            )
        self._providers.append(_Provider(int(tier), int(cost), str(name), free_up_to))
        self._providers.sort(key=lambda p: (p.tier, p.cost))
        logger.info(
            "%s registered provider %r in tier %s at cost %d (device %d); "
            "spend order is now: %s",
            LOG_PREFIX,
            name,
            _TIER_NAMES[tier],
            cost,
            self.device_index,
            ", ".join(f"{p.name}[{_TIER_NAMES[p.tier]}]" for p in self._providers),
        )

    @property
    def providers(self) -> Tuple[str, ...]:
        return tuple(p.name for p in self._providers)

    # -- the gate --------------------------------------------------------

    def free_bytes(self) -> int:
        if self._probe is not None:
            return int(self._probe())
        import torch

        return int(torch.cuda.mem_get_info(self.device_index)[0])

    def fleet_free(self) -> List[int]:
        """The per-card free column, in bytes. Empty when unknown."""
        if self._fleet_probe is None:
            return []
        try:
            return [int(f) for f in self._fleet_probe()]
        except Exception as e:
            # An unreadable fleet is an UNPROVEN fleet, and item 16's host
            # permission must be proven. Degrading to "empty" therefore
            # closes the host tier rather than opening it.
            logger.warning("%s fleet probe failed: %s", LOG_PREFIX, e)
            return []

    def _host_tier_permitted(self, column: Sequence[int]) -> bool:
        return fleet_is_level(column, self.floor_mib, self.delta_mib)

    def ensure_headroom(
        self,
        want_bytes: int,
        *,
        reason: str = "",
        raise_on_refusal: bool = False,
        refusal_is_fatal: bool = False,
        must_reclaim: bool = False,
    ) -> GuardResult:
        """Make ``want_bytes`` allocatable without breaching the floor.

        Returns a verdict. ``ok=False`` means DO NOT ALLOCATE -- the caller
        must take its own refusal path (at the seam: abandon the flip).

        ``refusal_is_fatal`` tells the gate that the caller has NO survivable
        refusal path, and it opens the host tier even on an unlevel fleet.
        Item 16 is a preference, not a suicide pact: withholding host RAM
        while a peer has headroom is right when the caller can wait, and
        catastrophic when it cannot. On the pp->tp leg it cannot -- strict
        purity forbids decode in PP, so a refused pp->tp starves decode and
        nothing in PP can free the memory that would end the refusal.
        Item 15c already authorises this in the user's terms: the price of the
        host tier is tempo, never a corridor breach. Refusing forever does not
        protect the corridor, which is fine in that state; it kills serving.

        It opens the tier; it does NOT reorder the ladder. Rebalance and park
        are still spent first, so the escape is only reached when nothing
        cheaper exists.
        """
        want = max(0, int(want_bytes))
        free_before = self.free_bytes()
        # The corridor law's own bound. Note it is checked against the
        # allocation that is ABOUT to happen, which is the entire difference
        # between this and a threshold observer: after the fact, a breach has
        # already been recorded by a 100 ms sampler and cannot be undone.
        # #689: THIS BRANCH IS WHERE THE FALSE SUCCESS RETURNED. Telling a
        # caller "no reclaim needed" is correct for an allocator about to
        # allocate `want`, and misleading for one that already accounted the
        # free column and needs this ladder to RELEASE more -- the 12:29 seam
        # asks returned here three times, ok=True, having freed nothing, while
        # spendable sat at 609 against a need of 788. Under must_reclaim the
        # ladder actually runs and the verdict is the measured delta.
        if free_before - want >= self.floor_bytes and not must_reclaim:
            return GuardResult(
                True, free_before, free_before, want, 0, (), "no reclaim needed"
            )

        self.arm_count += 1
        # Free to the UPPER watermark, not merely to the floor, so the next
        # few allocations do not each pay a spill.
        target = self.floor_bytes + self.delta_bytes + want
        if must_reclaim:
            # #689 THE TARGET IS RELATIVE UNDER must_reclaim. The ladder spends
            # only against a DEFICIT (`deficit = target - free_now`), so with
            # 1428 MiB free and a 178 MiB ask there is no deficit and it
            # correctly spends nothing -- which is why the 12:29 asks freed
            # zero. A caller that already accounted the free column is asking
            # for `want` MORE bytes, so the target has to be measured from
            # where the column stands now, not from the floor.
            target = max(target, free_before + want)
        # Item 16: read the fleet ONCE, before spending. Re-reading it after
        # each provider would let a rebalance that just filled a peer card
        # close the host gate mid-ladder on the strength of its own effect,
        # which is a feedback loop, not a policy.
        column = self.fleet_free()
        fleet_level = self._host_tier_permitted(column)
        host_ok = fleet_level or bool(refusal_is_fatal)
        host_forced = bool(refusal_is_fatal) and not fleet_level
        reclaimed, free_now, used, used_host, host_blocked = self._spend_ladder(
            target=target,
            free_now=free_before,
            column=column,
            host_ok=host_ok,
            host_forced=host_forced,
            max_tier=RELIEF_HOST,
            reason=reason,
        )

        self.reclaimed_total += reclaimed
        # USER DECISION 2026-08-16: THE LAW IS ADVISORY HERE, OOM IS NOT.
        #
        # This line used to read `ok = (free_now - want) >= law_floor_bytes`,
        # and that single comparison produced the 06:47:48 wedge: PP1's want
        # of 2163 MiB FIT inside 2456 MiB free, but the 293 MiB residual sat
        # under the law, so the seam was refused 76 times in a row while
        # 727004 tokens waited on an idle GPU. It protected a few hundred MiB
        # of headroom by stopping the machine.
        #
        # The ~1024 line exists because the planner was not filling VRAM well
        # enough. It is a FILL-QUALITY target and it remains the planner's
        # job; it was never a safety device, and it may not block, delay or
        # refuse anything on its own. What it still does is SPEAK: a dip is
        # carried out on the verdict and warned about by the caller.
        #
        # WHAT STAYS HARD is the only thing that was ever unsurvivable -- an
        # allocation larger than free. That is not a corridor dip, it is an
        # OOM, and softening it would trade a warning for a dead worker.
        ok = free_now >= want
        law_breached = ok and (free_now - want) < self.law_floor_bytes
        if not ok:
            self.refuse_count += 1
        # COUNTED WHEN IT MATTERED, and "mattered" is no longer the same as
        # "refused". Since the law stopped gating (2026-08-16), a withheld
        # host tier usually ends in a verdict that HOLDS but dips under the
        # law -- item 16's decision is exactly as consequential as before, so
        # a counter keyed on refusal alone would silently stop recording it.
        if host_blocked and (not ok or law_breached):
            self.host_blocked_count += 1
        detail = (
            f"want {want / _MIB:.0f} MiB, free {free_before / _MIB:.0f} -> "
            f"{free_now / _MIB:.0f} MiB, reclaimed {reclaimed / _MIB:.0f} MiB "
            f"from [{', '.join(used) or 'nothing'}], arming floor "
            f"{self.floor_bytes / _MIB:.0f} MiB, corridor law "
            f"{self.law_floor_mib} MiB" + (f" ({reason})" if reason else "")
        )
        # #689 A RECLAIM ASK IS JUDGED BY WHAT MOVED.
        #
        # ``free_now >= want`` asks "is want allocatable", which is exactly
        # right for an allocator about to allocate it -- every existing caller
        # is one, so their verdict is untouched and must_reclaim defaults off.
        #
        # It is the WRONG question for a caller that already accounted the
        # memory and needs this ladder to FREE more. Measured 2026-08-16
        # 12:29, three consecutive seam asks on the binding rank:
        #     asked the corridor guard for 178 MiB (pp_to_tp): ok=True,
        #     spendable now 609 MiB against a need of 788 MiB
        # 609 never moved. With 1428 MiB of driver-free, "are 178 MiB free"
        # was trivially true while the ladder reclaimed nothing, so the seam
        # was told it was funded and abandoned anyway. A success that is true
        # of the world before the call is not a report about the call.
        #
        # It also propagates: fundable_width's pre-arm picture is built from
        # what this returns, so an optimistic guard forms a window the seam
        # cannot carry and moves the failure later, into an abandon.
        if must_reclaim and ok and reclaimed < want:
            ok = False
            # THE MESSAGE MUST NOT QUOTE TERMS IT DID NOT JUDGE. The default
            # verdict weighs free against the floor; this one does not weigh
            # them at all, and a refusal that recites "free 2518, arming floor
            # 1536, corridor law 1024" next to "want 6" reads as an inversion
            # -- it was reported as one. Under must_reclaim the ONLY quantities
            # in the verdict are asked-vs-reclaimed, so those are the only ones
            # stated, with the reason the free column is irrelevant here.
            detail = (
                f"want {want / _MIB:.0f} MiB INCREMENTAL, reclaimed "
                f"{reclaimed / _MIB:.0f} MiB from "
                f"[{', '.join(used) if used else 'nothing'}] "
                f"({reason}). REFUSED under must_reclaim, which judges the "
                f"DELTA and nothing else: the caller has already accounted the "
                f"free column and is asking this ladder to RELEASE more, so "
                f"free memory it did not release cannot fund the ask and is "
                f"not weighed. Providers available: "
                f"{', '.join(self.providers) or 'none'}"
            )
        if host_forced and used_host:
            detail += (
                "; host tier admitted on an UNLEVEL fleet because refusal "
                f"would deadlock (free column {[int(f // _MIB) for f in column]} "
                f"MiB, spread {free_spread_mib(column)} MiB)"
            )
            clause = describe_water_fill(column)
            if clause:
                detail += f". {clause}"
        if host_blocked:
            detail += (
                "; host tier withheld -- fleet is not level"
                f" (free column {[int(f // _MIB) for f in column]} MiB, spread "
                f"{free_spread_mib(column)} MiB): item 16 spends host RAM only "
                "once every card is at the floor"
            )
            # NAME THE MOVE, not just the unevenness. "spread 879 MiB" does
            # not say which card to fix or by how much, and the decision a
            # successor faces -- is the missing rebalance tier worth building
            # -- turns on exactly that number.
            clause = describe_water_fill(column)
            if clause:
                detail += f". {clause}"
        if ok and law_breached:
            self.law_dip_count = getattr(self, "law_dip_count", 0) + 1
            logger.warning(
                "%s CANNOT FULLY HOLD THE CORRIDOR FLOOR through this seam "
                "entry on device %d: predicted trough %d MiB below the %d MiB "
                "law. PROCEEDING -- the law is a fill-quality target, not a "
                "gate (user decision 2026-08-16), and the allocation itself "
                "fits, so this is a dip and not an OOM. %s",
                LOG_PREFIX,
                self.device_index,
                (self.law_floor_bytes - (free_now - want)) / _MIB,
                self.law_floor_mib,
                detail,
            )
        elif ok:
            logger.info(
                "%s cleared on device %d: %s", LOG_PREFIX, self.device_index, detail
            )
        else:
            logger.error(
                "%s REFUSED on device %d: %s. Every provider is exhausted, so "
                "this allocation cannot be made without breaking the corridor "
                "law. The caller must take its refusal path -- allocating "
                "anyway would launder a breach as a passed check.",
                LOG_PREFIX,
                self.device_index,
                detail,
            )
            if raise_on_refusal:
                raise CorridorBreachRefused(f"{LOG_PREFIX} {detail}")
        return GuardResult(
            ok,
            free_before,
            free_now,
            want,
            reclaimed,
            tuple(used),
            detail,
            law_breached=law_breached,
        )

    # -- the ladder, shared by the gate and the lender ---------------------

    def _spend_ladder(
        self,
        *,
        target: int,
        free_now: int,
        column: Sequence[int],
        host_ok: bool,
        host_forced: bool,
        max_tier: int,
        reason: str,
    ):
        """Spend providers, cheapest tier first, until ``target`` free or dry.

        ONE ladder, two callers. :meth:`ensure_headroom` runs it with
        ``max_tier=RELIEF_HOST`` (spend anything the fleet permits);
        :meth:`lend_to_level` runs it with ``max_tier=RELIEF_REBALANCE``, so
        item 16's first relief stage physically cannot reach a park or a host
        spill. Extracting it was the alternative to a second spend loop, which
        in this module is how two policies drift apart while both look right.

        ``max_tier`` may terminate the loop rather than skip, because
        ``_providers`` is kept sorted by ``(tier, cost)``.
        """
        used: List[str] = []
        used_host = False
        host_blocked = False
        reclaimed = 0
        for provider in self._providers:
            if free_now >= target:
                break
            if provider.tier > max_tier:
                break
            if provider.tier == RELIEF_HOST and host_forced and not used_host:
                used_host = True
                self.host_forced_count += 1
                logger.warning(
                    "%s spending HOST RAM on device %d while the fleet is NOT "
                    "level (free column %s MiB, spread %d MiB), because "
                    "refusing here would deadlock the caller. Item 16 wanted "
                    "these bytes rebalanced onto a peer card instead -- this "
                    "counter is the cost of the missing rebalance tier, not a "
                    "licence. (%s)",
                    LOG_PREFIX,
                    self.device_index,
                    [int(f // _MIB) for f in column],
                    free_spread_mib(column),
                    reason or "no reason given",
                )
            if provider.tier == RELIEF_HOST and not host_ok:
                # Never while a peer still has headroom: the bytes belong on
                # that card, not in RAM.
                host_blocked = True
                continue
            deficit = target - free_now
            try:
                got = int(provider.free_up_to(deficit))
            except Exception as e:
                # A provider that fails must not take the allocation down;
                # the gate simply has less to spend and may still refuse.
                logger.warning(
                    "%s provider %r raised while freeing: %s",
                    LOG_PREFIX,
                    provider.name,
                    e,
                )
                continue
            if got <= 0:
                continue
            reclaimed += got
            used.append(provider.name)
            # Re-probe rather than trusting the provider's arithmetic: the
            # law is what the DRIVER reports, and a provider that returns
            # its payload size while the pages went to torch's cache has
            # freed nothing the corridor can see.
            free_now = self.free_bytes()

        return reclaimed, free_now, used, used_host, host_blocked

    # -- item 16's first relief stage --------------------------------------

    def lend_to_level(
        self,
        bound_bytes: int,
        *,
        column: Sequence[int],
        max_tier: int = RELIEF_REBALANCE,
        reason: str = "",
    ) -> GuardResult:
        """Give up to ``bound_bytes`` back on THIS card because the fleet is
        unlevel -- before any allocation has asked for them.

        WHAT MAKES THIS THE REBALANCE TIER AND NOT A SECOND GATE. The gate is
        reactive by construction: it arms on an allocation that would breach,
        which means the trough has already been reached by the time relief is
        spent. s34's green window measured that trough at 19 MiB of margin on
        the binding card while a peer held 3280 MiB free -- a PLACEMENT
        problem, not a capacity one. The lender is the same ladder, spent on
        the schedule the water-fill dictates instead of the schedule the
        allocator dictates.

        THE BOUND IS THE OBJECTIVE, NOT A DEFICIT. ``bound_bytes`` comes from
        :func:`water_fill_transfers` -- the payload this card must shed to sit
        level with its peers. Freeing more than that would evacuate a card
        that is no longer the tightest, which is the same unevenness with the
        sign flipped, and it would pay restore costs for nothing.

        IT PHYSICALLY CANNOT REACH HOST RAM. ``max_tier`` defaults to
        ``RELIEF_REBALANCE`` and the ladder terminates there, so the lender
        can never do what item 16 forbids -- spill to RAM while a peer has
        headroom -- no matter what a caller passes as a bound. Host RAM stays
        exactly where item 15c put it: the last stage, reached only through
        :meth:`ensure_headroom`, only once every card is at the floor.

        It never touches the KV rung either: that one is collective (it moves
        ``available_size()`` and therefore admission), and a rank-local lender
        that shrank the pool would be "a smaller pool as the fix", which the
        standing rule forbids and which this method is the alternative to.
        """
        bound = max(0, int(bound_bytes))
        free_before = self.free_bytes()
        if bound == 0:
            return GuardResult(
                True, free_before, free_before, 0, 0, (), "nothing to lend"
            )
        target = free_before + bound
        # HARD CEILING, not a default a caller can lift. Park is legitimate
        # here -- parking a cold payload in a peer card's surplus IS the
        # redistribution item 16 asks for -- but host RAM is the last stage by
        # user order, and the lender runs precisely when the fleet is UNLEVEL,
        # which is the one state in which host RAM is forbidden.
        ceiling = min(int(max_tier), RELIEF_PARK)
        claimed, free_now, used, _used_host, _blocked = self._spend_ladder(
            target=target,
            free_now=free_before,
            column=list(column),
            # The fleet is unlevel by construction -- that is why the lender
            # was called -- so the host tier is shut and cannot be forced.
            host_ok=False,
            host_forced=False,
            max_tier=ceiling,
            reason=reason,
        )
        # THE MEASURED DELTA, NOT THE SUM OF THE PROVIDERS' CLAIMS. The gate
        # can afford to credit claims because its verdict is re-probed anyway;
        # the lender's whole output IS the number, and this chain has three
        # times credited bytes that went to an allocator free-list instead of
        # to the driver. A fall in free between the probes (another process
        # taking memory) reads as 0 rather than as a negative.
        measured = max(0, free_now - free_before)
        self.lend_count += 1
        self.lent_total += measured
        # DELIBERATELY NOT added to ``reclaimed_total``. That counter answers
        # "what did the GATE spend", and ``reclaimed_total / arm_count`` is a
        # figure a successor will compute; folding lends into it pollutes both
        # and sets up a double count for anyone who later sums the two.
        # ``lent_total`` carries the lender's own bytes.
        #
        # One caveat this figure carries, named because the negative case is
        # already named and the positive one was not: the delta credits the
        # lend with any memory ANOTHER process released between the two
        # probes. It over-reports in the permissive direction.
        detail = (
            f"lent {measured / _MIB:.0f} MiB of a {bound / _MIB:.0f} MiB "
            f"water-fill bound, free {free_before / _MIB:.0f} -> "
            f"{free_now / _MIB:.0f} MiB, from [{', '.join(used) or 'nothing'}], "
            f"column {[int(f // _MIB) for f in column]} MiB, spread "
            f"{free_spread_mib(column)} MiB"
            + (
                f" (providers claimed {claimed / _MIB:.0f} MiB)"
                if claimed != measured
                else ""
            )
            + (f" ({reason})" if reason else "")
        )
        return GuardResult(
            True, free_before, free_now, 0, measured, tuple(used), detail
        )


def nvml_fleet_probe() -> Callable[[], List[int]]:
    """A fleet free-column probe over NVML, for item 16's host gate.

    NVML, deliberately, and not a collective. This gate runs inside the
    flip's no-return region; a collective there deadlocks the moment one rank
    takes a different branch, and the whole reason the seam is guarded is
    that ranks can disagree about affordability. NVML sidesteps the question:
    it sees every physical card regardless of ``CUDA_VISIBLE_DEVICES``, so a
    rank pinned to one card can still read the other two without asking
    anyone.

    The column is in NVML index order and includes cards this instance does
    not use. That is correct for the predicate being asked -- "is there
    headroom anywhere else" -- and it is conservative in the safe direction:
    a foreign card with free memory keeps the host tier SHUT, which costs
    tempo and never costs the corridor.

    Only called when the gate arms, which is rare, so no cache: a cached free
    column can only be wrong in the direction that opens the host gate.
    """

    def probe() -> List[int]:
        from sglang.srt.registry import nvml as registry_nvml

        with registry_nvml.nvml_session() as pynvml:
            out: List[int] = []
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                out.append(int(pynvml.nvmlDeviceGetMemoryInfo(handle).free))
            return out

    return probe


def allocator_cache_provider(
    probe: Callable[[], int],
    *,
    empty_cache: Optional[Callable[[], None]] = None,
) -> Callable[[int], int]:
    """Return torch's unused cached blocks to the driver. The cheapest relief.

    WHY THIS IS A PROVIDER AND NOT AN IMPLEMENTATION DETAIL. The seam already
    reclaims the allocator cache -- but inside ``_staging_affordable``, which
    runs AFTER the corridor gate. So the gate formed its verdict against a
    free column that understated the truth by the size of the hoard, and on
    this rig the hoard is 1028-1426 MiB per card at idle (register: flush
    contamination). The gate could therefore REFUSE a ``pp->tp`` flip -- the
    leg that starves decode outright under strict purity -- or force the host
    tier onto an unlevel fleet, while a gibibyte of nobody's memory sat on the
    card. Registering it makes the gate spend it FIRST, before any payload
    moves anywhere.

    THE RETURN VALUE IS A MEASURED DELTA, NOT ``memory_reserved() -
    memory_allocated()``. That difference is the size of the hoard, not the
    size of the release: the allocator keeps whole segments it is still
    carving from, so the two numbers routinely disagree. Crediting the hoard
    would be the same class of error as crediting a free-list push -- bytes
    the corridor law cannot see -- which this chain has now made three times.
    So the provider probes, flushes, probes again, and reports the difference.

    A fall in driver-free between the two probes (another process taking
    memory) is reported as 0 rather than as a negative number: the guard sums
    these into ``reclaimed_total`` and a negative would make the accounting
    lie in the permissive direction.

    ``nbytes`` is ignored, honestly rather than silently: ``empty_cache`` is
    all-or-nothing and cannot free a bounded amount. Over-delivering is not a
    failure -- the guard re-probes the driver and stops asking once the target
    is met.
    """

    def free_up_to(_nbytes: int) -> int:
        try:
            before = int(probe())
            if empty_cache is not None:
                empty_cache()
            else:
                import torch

                torch.cuda.empty_cache()
            after = int(probe())
        except Exception as e:
            logger.warning("%s allocator cache reclaim failed: %s", LOG_PREFIX, e)
            return 0
        return max(0, after - before)

    return free_up_to


def draft_carrier_provider(carrier) -> Callable[[int], int]:
    """Adapt a :class:`VmmDraftWeightCarrier` to the provider protocol.

    Cheapest real provider on this rig and the natural first registration:
    the payload is already proven to return its pages to the driver, the
    restore is priced into the seam's affordability verdict, and under strict
    purity the drafter is idle for the whole PP phase.

    It is ALL-OR-NOTHING -- ``decommit_range`` releases whole extents and the
    drafter is either resident or not -- so a request for fewer bytes than
    the payload still frees the whole payload. That is reported honestly
    rather than clipped, because the guard re-probes the driver anyway and
    over-delivering is not a failure.
    """

    def free_up_to(_nbytes: int) -> int:
        if carrier is None or carrier.spilled:
            return 0
        return int(carrier.spill() * _MIB)

    return free_up_to
