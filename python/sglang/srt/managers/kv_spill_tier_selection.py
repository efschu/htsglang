# SPDX-License-Identifier: Apache-2.0
"""The KV spill ladder, asked of the #407 tier registry instead of asserted.

#659. Today the kv-session-offload rung spills to exactly one place -- the
local pinned host pool -- and the ORDER of any further #224 park tiers is a
hardcoded rule: ``destinations_error`` (``kv_session_spill_destination.py:146``)
refuses any list whose first entry is not ``local``, quoting a measurement in
prose ("3.43 GB/s RDMA write ... orders of magnitude below local host RAM").

That rule is right, and it is also the wrong KIND of statement. It is a name
check standing in for a capability check, with its evidence in a comment where
nothing can re-derive it. #407's own design says so at the field that exists
for precisely this edge (``tiers.TierTransport.stages_through``):

    "#224's ``local`` must be first law is exactly this edge -- the device
    D2H copy lands in locally pinned host RAM, so every blob tier stages
    through the local host tier -- and recording it now is what lets that law
    stop being a hardcoded total order later."

This module is that "later". It builds a :class:`TierRegistry` whose FIRST
registered tier is local host RAM, and derives the ladder from measured
capacity and bandwidth. Two things follow, and the second is the point:

*   **The default is byte-identical.** With only the local tier registered and
    headroom for the ask, the selection is the local tier, which is what the
    rung does today. Nothing about the spill path changes.
*   **Local-first becomes a DERIVED result that can be falsified.** The
    hardcoded law and the measured ladder must agree; if they ever stop
    agreeing, :func:`local_first_disagreement` says so by name instead of the
    two drifting apart silently. See ``test_kv_spill_tier_selection_659``.

WHAT THIS MODULE IS NOT. It does not move a byte and it does not choose a
transport. The D2H hop is a physical fact -- pinned host RAM is where a device
copy can land -- so ``local`` is not merely first by cost, it is the only
possible first. The registry records that as ``stages_through``, so a park
tier is ranked as what it is: a destination BELOW local that local stages for.

THE BUDGET, and why it lives here (#659 b). The local host tier is the one
that can invoke the OOM killer: its pool is PINNED, this box has no swap, and
with ``--kv-session-offload-host-ram-gib`` at its default 0 the pool is sized
from ``--context-length`` alone with no bound at all (see the comment block at
``kv_session_offload.py:588-616``). ``pinned_host_budget`` already guards the
BOOT allocation and already sums every pinned post honestly. What it cannot do
is answer the RUNTIME question -- "does one more region fit?" -- because it
guards allocations, not occupancy. :func:`local_host_kv_tier` carries both:
the pool's own bytes as ``total``, and everything already claimed as
``reserved``, so ``headroom()`` is the live answer and exhaustion becomes a
named refusal that degrades to the next tier rather than an unbudgeted pin.

RANK-UNIFORMITY. Every function here is pure over its arguments and none of
them queries live host memory. The caller supplies the numbers, which is what
lets a hermetic test build a foreign rig, and what keeps a per-rank
free-memory reading out of a path whose divergence would be an NCCL hang
rather than a wrong answer (the same reasoning as
``kv_session_offload.host_ram_budget_error``).
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

from sglang.srt.memtier.registry import (
    Refusal,
    TierQuery,
    TierRegistry,
    TierSelection,
)
from sglang.srt.memtier.tiers import (
    PayloadClass,
    TierCapacity,
    TierCaps,
    TierDescriptor,
    TierHealth,
    TierKind,
    TierTransport,
    Volatility,
    host_tier_id,
)
from sglang.srt.planner.cost_model import Rate

#: Spilled KV is reconstructable ONLY by redoing user-visible work -- a
#: re-prefill of the session. That is #224's whole point and it is why the
#: payload may never be dropped to make room. It carries no
#: ``OFFLOAD_CLASSES`` name (the vocabulary has none for a spilled session, cf.
#: HiCache pages and hibernate images), so it is gated by volatility alone.
KV_SPILL_PAYLOAD = PayloadClass.EXPENSIVE_RECONSTRUCTABLE

#: The accounting bucket a KV spill reservation posts to, in #400 terms.
KV_SPILL_LEDGER_KEY = "kv_session_spill"

#: ``local`` in #224's spelling, so the two halves cannot drift on the token.
LOCAL_DESTINATION = "local"


def local_host_kv_tier(
    *,
    host: str,
    pool_bytes: int,
    occupied_bytes: int = 0,
    other_pinned_bytes: int = 0,
    bandwidth_gbs: Optional[Rate] = None,
    latency_us: Optional[Rate] = None,
    profile_id: str = "",
) -> TierDescriptor:
    """The local pinned host pool as the FIRST tier of the KV spill ladder.

    ``total`` is the pool that was actually allocated -- not the machine's RAM.
    A tier's capacity is what the tier HAS, and conflating the two is how the
    pinned-pool question got answered against ``/proc/meminfo`` six times over
    (the defect ``pinned_host_budget`` was built to end).

    ``reserved`` carries what is already claimed: bytes occupied by live
    spilled regions, plus any OTHER pinned post in this process. The second
    term is the join #407 declared and never made -- ``TierCapacity.reserved``
    is documented as "bytes declared by other holders, read from the
    cross-process ledger", and ``pinned_host_budget.registered_posts()`` IS
    that ledger. Until now nothing populated it from a live source, so every
    headroom the registry could compute was the headroom of an empty machine.

    ``floor`` is 0 and measured: the whole pool exists to be spilled into, so
    none of it is non-reclaimable AT THIS TIER. The OS reserve that protects
    the machine is a different quantity and it guards the ALLOCATION, in
    ``pinned_host_budget``; charging it here as well would price it twice.
    """
    return TierDescriptor(
        id=host_tier_id(host),
        kind=TierKind.HOST,
        host=host,
        capacity=TierCapacity(
            total=Rate.measured(
                int(pool_bytes),
                "allocated pinned kv-session-offload host pool",
                unit="bytes",
                label="total",
            ),
            floor=Rate.measured(
                0,
                "the pinned spill pool is reclaimable in full; the OS reserve "
                "guards the allocation in pinned_host_budget, not this tier",
                unit="bytes",
                label="floor",
            ),
            reserved=max(0, int(occupied_bytes)) + max(0, int(other_pinned_bytes)),
        ),
        volatility=Volatility.EXPENSIVE_OK,
        caps=TierCaps(
            latency_us=latency_us
            or Rate.absent(
                "no point-access probe has been taken for the pinned host pool "
                "(DESIGN_407 4 M1); absent, not assumed",
                unit="us",
                label="latency_us",
            ),
            bandwidth_gbs=bandwidth_gbs
            or Rate.absent(
                "no D2H bandwidth probe recorded for this pool",
                unit="GB/s",
                label="bandwidth_gbs",
            ),
            aperture_bytes=Rate.absent(
                "not a windowed tier; a host pool has no BAR aperture",
                unit="bytes",
                label="aperture_bytes",
            ),
            ledger_key=KV_SPILL_LEDGER_KEY,
        ),
        health=TierHealth(reachable=True),
        transport=TierTransport(
            name="pcie",
            handle="pool_host.MHATokenToKVPoolHost.backup_from_device_all_layer",
        ),
        profile_id=profile_id,
    )


def park_tier(
    *,
    tier_id: str,
    kind: TierKind,
    host: str,
    volatility: Volatility,
    capacity: TierCapacity,
    bandwidth_gbs: Rate,
    latency_us: Rate,
    transport_name: str,
    stages_through: Optional[str],
    handle: str = "",
    reachable: bool = True,
    verdict: str = "ok",
    reason: str = "",
    profile_id: str = "",
) -> TierDescriptor:
    """A #224 park tier, recorded with the edge that makes it a park tier.

    ``stages_through`` is not decoration. A blob tier cannot receive a device
    copy; the bytes reach it THROUGH the local host pool. Recording the edge is
    what lets the ladder be derived rather than asserted, and it is what makes
    a refusal readable: "below local" is a statement about an edge, not about
    a name.

    An unreachable tier is still built, with ``verdict="block"`` and a reason.
    Omitting it is how a spill target silently becomes a different spill
    target -- ``TierHealth``'s own words, and the registry refuses it by name.
    """
    return TierDescriptor(
        id=tier_id,
        kind=kind,
        host=host,
        capacity=capacity,
        volatility=volatility,
        caps=TierCaps(
            latency_us=latency_us,
            bandwidth_gbs=bandwidth_gbs,
            aperture_bytes=Rate.absent(
                "not a windowed tier", unit="bytes", label="aperture_bytes"
            ),
            ledger_key=KV_SPILL_LEDGER_KEY,
        ),
        health=TierHealth(reachable=reachable, verdict=verdict, reason=reason),
        transport=TierTransport(
            name=transport_name,
            handle=handle,
            stages_through=stages_through,
        ),
        profile_id=profile_id,
    )


def kv_spill_registry(
    tiers: Sequence[TierDescriptor],
    *,
    local_host: str,
    profile_id: str = "kv-spill-live",
    caveat: str = "",
) -> TierRegistry:
    """The KV spill ladder as a registry.

    The local host tier must be present: without it there is no ladder at all,
    because there is nowhere a device copy can land. Refusing here rather than
    returning an empty selection keeps the physical fact and the configuration
    error distinguishable.
    """
    if not any(t.kind is TierKind.HOST and t.host == local_host for t in tiers):
        raise ValueError(
            "the KV spill ladder has no local host tier. The device D2H copy "
            "lands in locally pinned host RAM, so the local host tier is not "
            "the cheapest rung -- it is the only possible FIRST rung, and "
            "every park tier stages through it."
        )
    return TierRegistry(
        tiers,
        profile_id=profile_id,
        local_host=local_host,
        caveat=caveat,
    )


def kv_spill_query(
    bytes_needed: int,
    *,
    min_bandwidth_gbs: Optional[float] = None,
    require_measured_bandwidth: bool = False,
    allow_unmeasured_bandwidth: bool = True,
) -> TierQuery:
    """The ask, spelled once.

    ``allow_unmeasured_bandwidth`` defaults TRUE, which is the opposite of what
    a cost-driven caller wants and is deliberate here: the local host pool has
    no D2H bandwidth probe on this rig, and refusing the only tier that can
    physically receive the copy -- because nobody has benchmarked it -- would
    turn a missing measurement into a dropped session. A caller that is
    CHOOSING between tiers on cost passes a floor or
    ``require_measured_bandwidth`` and gets #286's rule back ("an unmeasured
    path is NEVER assumed usable for parking, however tempting").
    """
    return TierQuery(
        payload=KV_SPILL_PAYLOAD,
        bytes_needed=max(0, int(bytes_needed)),
        min_bandwidth_gbs=min_bandwidth_gbs,
        require_measured_bandwidth=require_measured_bandwidth,
        allow_unmeasured_bandwidth=allow_unmeasured_bandwidth,
    )


def choose_kv_spill_tier(
    registry: TierRegistry,
    bytes_needed: int,
    **query_kw,
) -> Tuple[Optional[str], TierSelection]:
    """``(tier id or None, the full selection)``.

    The id is the cheapest sufficient tier. ``None`` means every tier refused,
    and the selection carries one named refusal per tier so the caller can say
    WHICH tier declined and WHY -- never a silent fallback, and never a silent
    drop.
    """
    selection = registry.select(kv_spill_query(bytes_needed, **query_kw))
    if not selection.candidates:
        return None, selection
    return selection.candidates[0].tier.id, selection


def refusal_report(selection: TierSelection, bytes_needed: int) -> str:
    """An operator-facing decline naming every tier and its reason.

    One line per tier. A refusal that reports only a total tells an operator
    that something did not fit; this one tells them which knob moves it --
    the same standard ``pinned_host_budget`` sets for the boot-time guard.
    """
    if not selection.refusals:
        return f"no tier refused {int(bytes_needed)} bytes"
    lines = [
        f"no KV spill tier accepted {int(bytes_needed)} bytes; "
        f"{len(selection.refusals)} tier(s) declined:"
    ]
    lines.extend(_refusal_line(r) for r in selection.refusals)
    return "\n".join(lines)


def _refusal_line(refusal: Refusal) -> str:
    return f"  - {refusal.tier_id} [{refusal.rule.value}]: {refusal.reason}"


def local_first_disagreement(
    registry: TierRegistry,
    bytes_needed: int,
    *,
    local_host: str,
    **query_kw,
) -> Optional[str]:
    """``None`` when the MEASURED ladder agrees that local comes first.

    The falsifier for this whole cut. #224 asserts local-first as a rule; this
    module derives the order from capacity and cost. The two must agree, and
    the moment they do not, one of them is wrong about the machine -- which is
    exactly the drift a comment-borne measurement cannot catch.

    Disagreement is REPORTED, never acted on. The hardcoded law stays in force
    because it encodes a physical constraint (a device copy has nowhere else
    to land), so a cheaper-looking park tier is evidence of a bad number, not
    a reason to reorder the ladder.
    """
    selection = registry.select(kv_spill_query(bytes_needed, **query_kw))
    if not selection.candidates:
        return None
    first = selection.candidates[0].tier
    if first.kind is TierKind.HOST and first.host == local_host:
        return None
    return (
        f"the measured ladder ranks {first.id} above the local host tier for "
        f"{int(bytes_needed)} bytes, but #224's law makes the local pinned "
        "pool the only tier a device copy can reach. One of the two is wrong "
        "about this machine: either a park tier's capacity/bandwidth record "
        "is overstated, or the local tier's is understated. The law stands; "
        "this is a report, not a reorder."
    )


def occupied_and_other_pinned(
    *,
    occupied_bytes: int,
    posts: Iterable,
    own_post_name: str,
) -> Tuple[int, int]:
    """Split the live claims into (this pool's occupancy, other pools' bytes).

    ``posts`` is ``pinned_host_budget.registered_posts()``. This pool's OWN
    post is excluded from the second term because it is the tier's ``total``,
    not a claim against it -- counting it in both places would charge the pool
    for existing and make the headroom of an empty pool zero.
    """
    others = 0
    for post in posts:
        if getattr(post, "name", None) == own_post_name:
            continue
        others += max(0, int(getattr(post, "nbytes", 0) or 0))
    return max(0, int(occupied_bytes)), others


def live_kv_spill_ladder(
    manager,
    *,
    local_host: str,
    own_post_name: str = "kv-session-offload spill pool",
    park_tiers: Sequence[TierDescriptor] = (),
) -> Optional[TierRegistry]:
    """The ladder built from THIS process's live state, or ``None``.

    The one impure function in this module, kept to one place on purpose: it
    reads the pool's occupancy and the pinned-post ledger and hands the numbers
    to the pure builders above, so everything that DECIDES stays testable
    against a synthetic rig.

    ``None`` when the host pool is not attached (kv-session-offload off, which
    is the ship config): there is no ladder without a pool, and returning an
    empty registry would let a caller believe it had asked.
    """
    from sglang.srt.mem_cache.pinned_host_budget import registered_posts
    from sglang.srt.observability.spill_tiers import kv_session_host_bytes

    if getattr(manager, "host_pool", None) is None:
        return None
    # Returns None (not a pair) when session offload is off -- unpacking it
    # directly would raise inside the caller's blanket except and turn a
    # perfectly ordinary "no pool" into an invisible skip.
    sizes = kv_session_host_bytes(manager)
    if not sizes:
        return None
    used, total = sizes
    if not total:
        return None
    occupied, others = occupied_and_other_pinned(
        occupied_bytes=int(used or 0),
        posts=registered_posts(),
        own_post_name=own_post_name,
    )
    local = local_host_kv_tier(
        host=local_host,
        pool_bytes=int(total),
        occupied_bytes=occupied,
        other_pinned_bytes=others,
        profile_id="kv-spill-live",
    )
    return kv_spill_registry([local, *park_tiers], local_host=local_host)


def ladder_rows(registry: TierRegistry) -> List[Tuple[str, str, str, str]]:
    """``(tier id, role, headroom, bandwidth)`` per tier, for a log line.

    Renders the ABSENCES rather than hiding them: a tier whose headroom cannot
    be computed prints why, because "unknown" and "zero" lead to opposite
    decisions and a dash would conflate them.
    """
    rows: List[Tuple[str, str, str, str]] = []
    for tier in registry.tiers():
        headroom = tier.capacity.headroom()
        bandwidth = tier.caps.bandwidth_gbs
        rows.append(
            (
                str(tier.id),
                tier.role(local_host=registry.local_host),
                (
                    f"absent ({headroom.source})"
                    if headroom.is_absent
                    else f"{headroom.require('headroom') / 1e9:.2f} GB "
                    f"[{headroom.provenance.value}]"
                ),
                (
                    f"absent ({bandwidth.source})"
                    if bandwidth.is_absent
                    else f"{bandwidth.require('bandwidth_gbs'):.2f} GB/s "
                    f"[{bandwidth.provenance.value}]"
                ),
            )
        )
    return rows
