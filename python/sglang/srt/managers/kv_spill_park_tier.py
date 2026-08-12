# SPDX-License-Identifier: Apache-2.0
"""The second KV spill rung: a #224 file park backend, MEASURED at registration.

#659 (c). Cut 1 (``kv_spill_tier_selection``) put the #407 registry under the
KV spill rung as an OBSERVER with exactly one tier on it -- local pinned host
RAM. This module supplies the second rung and lets the registry stop observing.

WHY A LOCAL FILESYSTEM AND NOT THE REMOTE TIER THE BRIEF PREFERRED. Successor
41 measured the two candidates the brief named and refused both, on independent
grounds recorded in ``docs/dev/631/TIER2_LINK_MEASUREMENTS.md``: Rig-2 RAM is
unreachable from this container (its 100G/40G NICs sit on unroutable subnets,
the only routable path measures 75 MB/s, and the far box offers 8.6 GiB of
SWAP-BACKED RAM against a ~12.9 GB KV region), and peer VRAM is the REBALANCE
tier, closed by triple falsification. A local filesystem is neither fast nor
glamorous, but it is REACHABLE, it is large, and -- unlike either refused
candidate -- it can honour a persistence contract. It is a tier, not a remote
tier, and that distinction is stated rather than smuggled.

THE MEASUREMENT DISCIPLINE, WHICH IS THE POINT OF THIS FILE (register C24).
``memtier/profiles/rig1.json`` carries a row for this very rig claiming
2.83 GB/s "measured" for a link that does not exist from this process. A number
that travels in a file outlives the geometry it was taken on (register law 1).
So NOTHING here is read from a profile: :func:`probe_park_filesystem` runs a
real write/read/fsync probe against the real park directory at registration
time, and the resulting :class:`Rate` values carry ``Provenance.MEASURED`` with
a source string naming the probe. If the probe cannot run, the entry says
``absent`` with the reason -- it never falls back to a literature value.

TWO MEASUREMENT TRAPS THIS PROBE ACTIVELY AVOIDS, both found on this rig:

*   **Compression.** The park directory here is ZFS. Probing with a buffer of
    zeros measured 6.2 GB/s with ``fdatasync`` -- that is the compressor, not
    the disk. The probe writes INCOMPRESSIBLE bytes (``os.urandom``) for this
    reason, which brought the same measurement to ~3.1 GB/s.
*   **Read-back from cache.** Reading a file straight after writing it measures
    the page cache / ARC, not the medium. The read probe is therefore reported
    as what it is -- a read-back rate with the cache warm, an UPPER bound --
    and the tier is ranked on the WRITE rate, which is the one a park pays and
    the one an fsync makes honest.

GROUP UNIFORMITY (standing law; register laws 8 and 14). Every rank runs this
probe against the same filesystem and will measure slightly different numbers.
A ladder ordered on raw per-rank floats is a rank-local decision over
replicated state -- exactly the shape that made three ranks take turns being
short in C20's delay budget. So every ordering-relevant number is QUANTIZED
(:func:`quantize_bandwidth`, :func:`quantize_bytes`) before it can influence an
order: measurement noise cannot flip a rung, and only a difference big enough
to be real survives the rounding. The raw value is still recorded on the entry
for the ledger; it is the quantized one that sorts.

THE BUDGET IS EXPLICIT AND df-AWARE (#558 family). A park directory shares a
filesystem with everything else on the box, so an unbounded park tier is a
disk-full outage with extra steps. The tier's ``total`` is
``min(configured budget, free space - df headroom)``, and the df headroom is
subtracted BEFORE the budget is honoured, not after: the operator's number is a
ceiling, never a licence to fill the volume.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from typing import Dict, List, Optional, Sequence, Tuple

from sglang.srt.managers.kv_spill_tier_selection import kv_spill_query, park_tier
from sglang.srt.memtier.registry import TierRegistry, TierSelection
from sglang.srt.memtier.tiers import (
    TierCapacity,
    TierDescriptor,
    TierKind,
    Volatility,
    filesystem_tier_id,
)
from sglang.srt.planner.cost_model import Rate

logger = logging.getLogger(__name__)

#: Bytes written by the registration probe.
#:
#: THIS NUMBER IS AN UPPER BOUND AND THE ENTRY SAYS SO. A registration probe
#: has to finish at boot, so it measures a SHORT write, and a short write on a
#: caching filesystem is partly absorbed rather than sustained. Measured on
#: this rig's park volume (successor 42, incompressible payload, fsync'd):
#:
#:     probe size    64 MiB   256 MiB   512 MiB   1 GiB   2 GiB   8 GiB
#:     write GB/s      8.69      6.65      6.80    4.27    4.92    3.13
#:     probe wall      83 ms    127 ms    184 ms  510 ms  1285 ms    ~2.6 s
#:
#: i.e. the honest sustained rate here is ~3 GB/s and a 64 MiB probe overstates
#: it by 2.8x. 256 MiB is the chosen compromise: 127 ms of boot per rank, and
#: within ~2x of sustained. The value is used for ORDERING tiers (quantized,
#: see :func:`quantize_bandwidth`) and for the measured/absent distinction --
#: it is NOT a capacity plan and NOT an ETA, and nothing may quote it as one.
PARK_PROBE_BYTES = 256 * 1024 * 1024

#: Payload of one latency sample: a page-sized write plus ``fdatasync``. The
#: quantity of interest is the SYNC, which is what a park commit pays.
PARK_LATENCY_SAMPLE_BYTES = 4096
PARK_LATENCY_SAMPLES = 64

#: Ordering grid. Bandwidth sorts in 0.25 GB/s buckets and capacity in 1 GiB
#: buckets, so two ranks measuring the same medium cannot disagree about an
#: order while a genuinely different medium still separates.
BANDWIDTH_QUANTUM_GBS = 0.25
BYTES_QUANTUM = 1024 * 1024 * 1024

#: Default free-space headroom kept below any park budget. The park directory
#: shares its volume with the model cache, the evidence tree and the logs; a
#: park tier that fills the last gigabyte takes the whole box down with it.
DEFAULT_DF_HEADROOM_BYTES = 32 * BYTES_QUANTUM

#: How many faults on a tier before the selection policy stops choosing it.
#: A park that fails is a park that has to be redone somewhere else; repeating
#: it forever is the silent-fallback shape this cut exists to remove.
PARK_FAULT_WARN = 1
PARK_FAULT_BLOCK = 3


def quantize_bandwidth(value: float) -> float:
    """Round a GB/s reading down onto the ordering grid.

    DOWN, not nearest: a tier may never be ranked above what it demonstrated.
    """
    if value <= 0:
        return 0.0
    return (int(value / BANDWIDTH_QUANTUM_GBS)) * BANDWIDTH_QUANTUM_GBS


def quantize_bytes(value: int) -> int:
    """Round a byte count down onto the ordering grid (see module docstring)."""
    if value <= 0:
        return 0
    return (int(value) // BYTES_QUANTUM) * BYTES_QUANTUM


def probe_park_filesystem(
    directory: str,
    *,
    probe_bytes: int = PARK_PROBE_BYTES,
    latency_samples: int = PARK_LATENCY_SAMPLES,
) -> Tuple[Rate, Rate, Dict[str, float]]:
    """Measure the park directory. Returns ``(bandwidth, latency, raw)``.

    Runs at REGISTRATION, against the directory that will actually hold the
    blobs -- never against a proxy path and never from a profile (C24).

    The write probe uses incompressible bytes and ends in an ``fsync``, so the
    number is a rate the medium actually sustained rather than a rate the page
    cache accepted. The read-back is measured too, but only recorded in ``raw``
    and in the entry's properties: with the cache warm it is an upper bound,
    and ranking a tier on it would credit it with a speed a park never gets.

    On any failure the rates come back ABSENT with the reason as their source.
    An absent bandwidth is refused by the registry by name
    (``RefusalRule.BANDWIDTH_ABSENT``) when a caller asks for a measured one --
    an unprobed path is never assumed usable (#286).
    """
    raw: Dict[str, float] = {}
    probe_dir = os.path.join(directory, ".kvso-park-probe")
    path = os.path.join(probe_dir, f"probe.{os.getpid()}.bin")
    try:
        os.makedirs(probe_dir, exist_ok=True)
        chunk = os.urandom(min(int(probe_bytes), 32 * 1024 * 1024) or 1)
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
        try:
            written = 0
            t0 = time.perf_counter()
            while written < probe_bytes:
                written += os.write(fd, chunk)
            os.fsync(fd)
            write_s = time.perf_counter() - t0
        finally:
            os.close(fd)
        write_gbs = (written / write_s) / 1e9 if write_s > 0 else 0.0
        raw["write_gbs"] = write_gbs
        raw["probe_bytes"] = float(written)

        fd = os.open(path, os.O_RDONLY)
        try:
            read = 0
            t0 = time.perf_counter()
            while True:
                block = os.read(fd, 8 * 1024 * 1024)
                if not block:
                    break
                read += len(block)
            read_s = time.perf_counter() - t0
        finally:
            os.close(fd)
        raw["readback_gbs"] = (read / read_s) / 1e9 if read_s > 0 else 0.0

        sample = b"\0" * PARK_LATENCY_SAMPLE_BYTES
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC)
        samples: List[float] = []
        try:
            for _ in range(max(1, int(latency_samples))):
                t0 = time.perf_counter()
                os.write(fd, sample)
                os.fdatasync(fd)
                samples.append((time.perf_counter() - t0) * 1e6)
        finally:
            os.close(fd)
        samples.sort()
        latency_us = samples[len(samples) // 2]
        raw["latency_us_p50"] = latency_us
        raw["latency_us_p90"] = samples[min(len(samples) - 1, int(len(samples) * 0.9))]
    except OSError as exc:
        reason = f"park probe failed at {directory}: {exc.strerror or exc}"
        logger.warning("kv-spill park tier: %s", reason)
        return (
            Rate.absent(reason, unit="GB/s", label="bandwidth_gbs"),
            Rate.absent(reason, unit="us", label="latency_us"),
            raw,
        )
    finally:
        try:
            os.unlink(path)
            os.rmdir(probe_dir)
        except OSError:
            pass

    source = (
        f"park probe at {directory}: {int(raw['probe_bytes'])} incompressible "
        f"bytes written and fsynced at registration"
    )
    return (
        Rate.measured(write_gbs, source, unit="GB/s", label="bandwidth_gbs"),
        Rate.measured(
            latency_us,
            f"park probe at {directory}: p50 of {len(samples)} "
            f"{PARK_LATENCY_SAMPLE_BYTES}-byte write+fdatasync samples",
            unit="us",
            label="latency_us",
        ),
        raw,
    )


def park_filesystem_capacity(
    directory: str,
    *,
    budget_bytes: int,
    df_headroom_bytes: int = DEFAULT_DF_HEADROOM_BYTES,
    parked_bytes: int = 0,
) -> TierCapacity:
    """The park tier's capacity: the operator's ceiling, floored by df.

    ``total`` is ``min(budget, free - headroom)`` and never the raw free space.
    The order matters and is the #558-family discipline: the df headroom comes
    off FIRST, so a generous budget on a nearly-full volume yields a small tier
    rather than a large tier that fails at the worst moment.

    ``reserved`` carries what this tier already holds, so ``headroom()`` is the
    live answer to "does one more region fit?" -- the same runtime question
    ``pinned_host_budget`` cannot answer for the host rung because it guards
    allocations rather than occupancy.

    A budget of 0 means "derive it entirely from df", which is the honest
    reading of an unset flag: not unbounded.
    """
    try:
        usage = shutil.disk_usage(directory)
        free = int(usage.free)
    except OSError as exc:
        reason = f"statvfs failed at {directory}: {exc.strerror or exc}"
        return TierCapacity(
            total=Rate.absent(reason, unit="bytes", label="total"),
            floor=Rate.absent(reason, unit="bytes", label="floor"),
            reserved=max(0, int(parked_bytes)),
        )
    headroom_kept = max(0, int(df_headroom_bytes))
    allowed = max(0, free - headroom_kept)
    if budget_bytes > 0:
        total = min(int(budget_bytes), allowed)
        source = (
            f"min(configured park budget {budget_bytes} B, free {free} B - df "
            f"headroom {headroom_kept} B) at {directory}"
        )
    else:
        total = allowed
        source = (
            f"free {free} B - df headroom {headroom_kept} B at {directory} "
            f"(no explicit park budget configured)"
        )
    return TierCapacity(
        total=Rate.measured(total, source, unit="bytes", label="total"),
        floor=Rate.measured(
            0,
            "a park file is fully reclaimable by deleting it",
            unit="bytes",
            label="floor",
        ),
        reserved=max(0, int(parked_bytes)),
    )


def park_health(faults: int) -> Tuple[bool, str, str]:
    """Turn the #224 fault tally for one tier into a #407 health verdict.

    This is one of the three places the park counters stop being write-only.
    A tier that has failed parks is not merely noisy: every fault is a session
    that had to be redone somewhere else, so the tally is exactly the signal a
    selection policy should consult. Below ``PARK_FAULT_WARN`` it is silent,
    then it warns, then it blocks -- and a blocked tier is still ENUMERATED
    with its reason, because a spill target that silently becomes a different
    spill target is the failure mode #224's own docstring names.
    """
    faults = max(0, int(faults))
    if faults >= PARK_FAULT_BLOCK:
        return (
            False,
            "block",
            f"{faults} failed or timed-out parks on this tier "
            f"(>= {PARK_FAULT_BLOCK}); it is not taking overflow",
        )
    if faults >= PARK_FAULT_WARN:
        return (
            True,
            "warn",
            f"{faults} failed or timed-out parks on this tier",
        )
    return (True, "ok", "")


def file_park_tier(
    *,
    host: str,
    directory: str,
    backend_name: str = "file",
    budget_bytes: int = 0,
    df_headroom_bytes: int = DEFAULT_DF_HEADROOM_BYTES,
    parked_bytes: int = 0,
    faults: int = 0,
    local_host_tier_id: Optional[str] = None,
    probe: Optional[Tuple[Rate, Rate, Dict[str, float]]] = None,
    profile_id: str = "kv-spill-live",
) -> TierDescriptor:
    """Build the file park tier entry, with its metrics MEASURED here.

    ``volatility`` is :attr:`Volatility.PERSISTENT`: a parked blob is a file
    and survives process exit. That is a real capability of the medium, and it
    is declared as such even though today's consumer only needs
    ``EXPENSIVE_OK`` -- #407's table admits an expensive-reconstructable
    payload on a persistent tier, and understating the class would make the
    entry useless to the #89 hibernate consumer that will want exactly this
    property.

    ``stages_through`` names the local host tier, because that is the physical
    edge: a device copy cannot land in a file. It is what makes "below local" a
    statement about an edge rather than about a name.
    """
    if probe is None:
        probe = probe_park_filesystem(directory)
    bandwidth, latency, raw = probe
    capacity = park_filesystem_capacity(
        directory,
        budget_bytes=budget_bytes,
        df_headroom_bytes=df_headroom_bytes,
        parked_bytes=parked_bytes,
    )
    reachable, verdict, reason = park_health(faults)
    tier = park_tier(
        tier_id=filesystem_tier_id(host, directory),
        kind=TierKind.FILESYSTEM,
        host=host,
        volatility=Volatility.PERSISTENT,
        capacity=capacity,
        bandwidth_gbs=bandwidth,
        latency_us=latency,
        transport_name="filesystem",
        stages_through=local_host_tier_id,
        handle=f"hicache_storage.HiCacheFile({backend_name}) at {directory}",
        reachable=reachable,
        verdict=verdict,
        reason=reason,
        profile_id=profile_id,
    )
    properties = {
        "medium": "filesystem",
        "backend": backend_name,
        "directory": directory,
        "park_faults": str(int(faults)),
        "df_headroom_bytes": str(int(df_headroom_bytes)),
        "configured_budget_bytes": str(int(budget_bytes)),
    }
    if "readback_gbs" in raw:
        # Recorded, deliberately NOT ranked on: the read probe runs with the
        # cache warm, so it is an upper bound rather than a rate a park pays.
        properties["readback_gbs_cache_warm"] = f"{raw['readback_gbs']:.3f}"
    if "write_gbs" in raw:
        properties["write_gbs_raw"] = f"{raw['write_gbs']:.3f}"
    if "latency_us_p90" in raw:
        properties["latency_us_p90"] = f"{raw['latency_us_p90']:.1f}"
    return tier.evolve(properties=properties)


def _ordering_bandwidth(tier: TierDescriptor) -> float:
    value = tier.caps.bandwidth_gbs.or_none()
    return quantize_bandwidth(value) if value else 0.0


def choose_park_tier(
    registry: TierRegistry,
    bytes_needed: int,
    *,
    park_tier_ids: Sequence[str],
) -> Tuple[Optional[int], TierSelection]:
    """The selection policy: cheapest SUFFICIENT park tier, or a named refusal.

    Returns ``(index into park_tier_ids, selection)``. ``None`` means every
    park tier was refused, and the selection carries one named refusal per
    tier -- there is no silent fallback, which is the whole ask.

    "Cheapest sufficient" is evaluated by the #407 registry, not here: the
    registry already refuses on health, volatility, capacity and bandwidth in
    that order and ranks survivors by provenance then bandwidth. This function
    only restricts the answer to the configured park tiers and translates the
    winner back into the index #224's transfer records.

    ORDERING IS QUANTIZED, so the answer is group-uniform. Two ranks probing
    the same filesystem measure different floats; if those floats sorted
    directly, two ranks could park the same session to two different tiers and
    the divergence would surface as a hang rather than an error (register law
    14's shape). The quantum is coarser than the spread between ranks and
    finer than the gap between media.
    """
    selection = registry.select(kv_spill_query(int(bytes_needed)))
    wanted = list(park_tier_ids)
    ranked = [
        candidate
        for candidate in selection.candidates
        if candidate.tier.id in wanted
    ]
    ranked.sort(
        key=lambda c: (
            -_ordering_bandwidth(c.tier),
            wanted.index(c.tier.id),
        )
    )
    if not ranked:
        return None, selection
    return wanted.index(ranked[0].tier.id), selection


def park_refusal_lines(
    selection: TierSelection,
    park_tier_ids: Sequence[str],
    bytes_needed: int,
) -> List[str]:
    """One line per refused park tier, each naming the tier and the reason."""
    lines: List[str] = []
    for tier_id in park_tier_ids:
        refusal = selection.refusal_for(tier_id)
        if refusal is None:
            continue
        lines.append(
            f"{tier_id} refused for {bytes_needed} B "
            f"[{refusal.rule.value}]: {refusal.reason}"
        )
    return lines


def park_counter_row(counters: Dict[str, int]) -> str:
    """Render the #224 park counters for the ledger line.

    The second place the counters stop being write-only. They were complete
    and correct for three shifts and nothing ever read them, so a park that
    failed and a park that never happened produced the same silence.

    Bytes are reported as MOVED, not as occupancy, and the wording says so:
    ``park_bytes_out`` counts every byte ever written to a park tier, which is
    a traffic total. Occupancy is a different question with a different source
    (``observability.spill_tiers.park_bytes_by_tier``), and conflating the two
    is the mistake that module's own comment warns about.
    """
    get = lambda k: int(counters.get(k, 0))  # noqa: E731
    return (
        f"parks started/committed/failed/timeout="
        f"{get('parks_started')}/{get('parks_committed')}/"
        f"{get('parks_failed')}/{get('parks_timeout')} "
        f"unparks started/committed/failed/timeout="
        f"{get('unparks_started')}/{get('unparks_committed')}/"
        f"{get('unparks_failed')}/{get('unparks_timeout')} "
        f"moved out/in={get('park_bytes_out')}/{get('unpark_bytes_in')} B "
        f"identity_miss={get('unpark_identity_miss')} "
        f"orphaned={get('blobs_orphaned')} aborted={get('parked_aborted')}"
    )


def park_fault_key(tier_name: str) -> str:
    """Counter key for per-tier faults, so health is per tier and not global.

    The flat counters answer "how many parks failed"; a selection policy needs
    "which tier failed them". Same tally, keyed by the tier that took it.
    """
    return f"park_faults:{tier_name}"


__all__ = [
    "BANDWIDTH_QUANTUM_GBS",
    "BYTES_QUANTUM",
    "DEFAULT_DF_HEADROOM_BYTES",
    "PARK_FAULT_BLOCK",
    "PARK_FAULT_WARN",
    "PARK_PROBE_BYTES",
    "choose_park_tier",
    "file_park_tier",
    "park_counter_row",
    "park_fault_key",
    "park_filesystem_capacity",
    "park_health",
    "park_refusal_lines",
    "probe_park_filesystem",
    "quantize_bandwidth",
    "quantize_bytes",
]
