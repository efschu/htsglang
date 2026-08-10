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
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

LOG_PREFIX = "CORRIDOR-GUARD"

_MIB = 1024 * 1024

#: The user's corridor law, in the unit the law is stated in.
DEFAULT_FLOOR_MIB = 1024

#: How far ABOVE the floor a reclaim frees, so the next few allocations do not
#: each pay a spill. 256 MiB is one seam's worth of slack on this rig's
#: measured seam trough (1196 MiB free at the tightest instant, floor 1024).
DEFAULT_DELTA_MIB = 256


class CorridorBreachRefused(RuntimeError):
    """The allocation cannot be made without breaking the corridor law.

    Raised only when the caller asked for ``raise_on_refusal``; the default
    is a returned verdict, because the seam's caller wants to abandon the
    flip cleanly rather than unwind an exception from inside a cutover.
    """


@dataclass(order=True)
class _Provider:
    # Ordered by cost first: the cheapest thing to give back goes first.
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
    ) -> None:
        self.device_index = int(device_index)
        self.floor_bytes = int(floor_mib) * _MIB
        self.delta_bytes = int(delta_mib) * _MIB
        self._probe = probe
        self._providers: List[_Provider] = []
        self.arm_count = 0
        self.refuse_count = 0
        self.reclaimed_total = 0

    # -- registration ----------------------------------------------------

    def register(
        self, name: str, cost: int, free_up_to: Callable[[int], int]
    ) -> None:
        """Add a payload class the gate may spend.

        ``free_up_to(nbytes)`` must free AT MOST ``nbytes``, synchronously,
        and return the bytes it actually gave back to the DRIVER -- not to
        torch's cache. The corridor law is stated in NVML's free column, and
        a provider that only returns memory to the caching allocator has
        freed nothing the law can see.

        ``cost`` orders the spend: lower is cheaper to give up and to get
        back. Ties are resolved by registration order.
        """
        if not callable(free_up_to):
            raise TypeError(f"{LOG_PREFIX} provider {name!r} is not callable")
        self._providers.append(_Provider(int(cost), str(name), free_up_to))
        self._providers.sort(key=lambda p: p.cost)
        logger.info(
            "%s registered provider %r at cost %d (device %d); spend order "
            "is now: %s",
            LOG_PREFIX,
            name,
            cost,
            self.device_index,
            ", ".join(p.name for p in self._providers),
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

    def ensure_headroom(
        self,
        want_bytes: int,
        *,
        reason: str = "",
        raise_on_refusal: bool = False,
    ) -> GuardResult:
        """Make ``want_bytes`` allocatable without breaching the floor.

        Returns a verdict. ``ok=False`` means DO NOT ALLOCATE -- the caller
        must take its own refusal path (at the seam: abandon the flip).
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
        reclaimed = 0
        free_now = free_before
        for provider in self._providers:
            if free_now >= target:
                break
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
        ok = (free_now - want) >= self.floor_bytes
        if not ok:
            self.refuse_count += 1
        detail = (
            f"want {want/_MIB:.0f} MiB, free {free_before/_MIB:.0f} -> "
            f"{free_now/_MIB:.0f} MiB, reclaimed {reclaimed/_MIB:.0f} MiB "
            f"from [{', '.join(used) or 'nothing'}], floor "
            f"{self.floor_bytes/_MIB:.0f} MiB"
            + (f" ({reason})" if reason else "")
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
