"""#656 spec item 15a/15b: SPILL BEFORE THE ALLOCATION, not after the alarm.

WHY THIS IS NOT ``kv_pressure_runtime``
---------------------------------------
``KvPressureRuntime`` (#287) already drives a spill ladder, and it already
takes a ``spill_fn``. It is nonetheless the wrong shape for item 15a, and the
difference is the whole point of the user's order:

    "SPILL-VOR-ALLOC -- Pruefung an der Allokation (frei-X >= 1024 sonst erst
     synchron spillen), nicht reaktive Schwellen-Beobachtung"

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
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "CORRIDOR-GUARD"

_MIB = 1024 * 1024

#: The user's corridor law, in the unit the law is stated in.
DEFAULT_FLOOR_MIB = 1024

#: How far ABOVE the floor a reclaim frees, so the next few allocations do not
#: each pay a spill. 256 MiB is one seam's worth of slack on this rig's
#: measured seam trough (1196 MiB free at the tightest instant, floor 1024).
DEFAULT_DELTA_MIB = 256

#: Relief tiers (item 16). Lower is tried first, unconditionally: the tier
#: outranks the cost, so a cheap host spill can never overtake an expensive
#: rebalance. Cost still orders providers within one tier.
RELIEF_REBALANCE = 0  # level the free column: token vector, evacuate the heavy card
RELIEF_PARK = 1  # park a cold payload in another card's surplus VRAM
RELIEF_HOST = 2  # system RAM -- only once every card is at the floor

_TIER_NAMES = {
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
        if free_before - want >= self.floor_bytes:
            return GuardResult(
                True, free_before, free_before, want, 0, (), "no reclaim needed"
            )

        self.arm_count += 1
        # Free to the UPPER watermark, not merely to the floor, so the next
        # few allocations do not each pay a spill.
        target = self.floor_bytes + self.delta_bytes + want
        used: List[str] = []
        used_host = False
        reclaimed = 0
        free_now = free_before
        # Item 16: read the fleet ONCE, before spending. Re-reading it after
        # each provider would let a rebalance that just filled a peer card
        # close the host gate mid-ladder on the strength of its own effect,
        # which is a feedback loop, not a policy.
        column = self.fleet_free()
        fleet_level = self._host_tier_permitted(column)
        host_ok = fleet_level or bool(refusal_is_fatal)
        host_forced = bool(refusal_is_fatal) and not fleet_level
        host_blocked = False
        for provider in self._providers:
            if free_now >= target:
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

        self.reclaimed_total += reclaimed
        # Judged against the LAW, not the arming watermark: see __init__.
        ok = (free_now - want) >= self.law_floor_bytes
        if not ok:
            self.refuse_count += 1
        if host_blocked and not ok:
            self.host_blocked_count += 1
        detail = (
            f"want {want/_MIB:.0f} MiB, free {free_before/_MIB:.0f} -> "
            f"{free_now/_MIB:.0f} MiB, reclaimed {reclaimed/_MIB:.0f} MiB "
            f"from [{', '.join(used) or 'nothing'}], arming floor "
            f"{self.floor_bytes/_MIB:.0f} MiB, corridor law "
            f"{self.law_floor_mib} MiB"
            + (f" ({reason})" if reason else "")
        )
        if host_forced and used_host:
            detail += (
                "; host tier admitted on an UNLEVEL fleet because refusal "
                f"would deadlock (free column {[int(f // _MIB) for f in column]} "
                f"MiB, spread {free_spread_mib(column)} MiB)"
            )
        if host_blocked:
            detail += (
                "; host tier withheld -- fleet is not level"
                f" (free column {[int(f // _MIB) for f in column]} MiB, spread "
                f"{free_spread_mib(column)} MiB): item 16 spends host RAM only "
                "once every card is at the floor"
            )
        if ok:
            logger.info("%s cleared on device %d: %s", LOG_PREFIX, self.device_index, detail)
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
            ok, free_before, free_now, want, reclaimed, tuple(used), detail
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
