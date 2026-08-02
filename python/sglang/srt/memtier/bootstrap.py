# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""What an UNKNOWN machine gets: tiers from live facts, every cap absent (#407).

This is the probe-first bootstrap directive #434 requires, and the answer to
the question cut 1 got wrong. Cut 1's default was the bundled rig-1 profile, so
a machine nobody had ever measured received rig-1's host RAM, rig-1's ZFS pool
and rig-1's 40G peer. This module is what that default becomes: an enumeration
built entirely from what the local machine reports about itself, in which

*   **capacity is measured** -- NVML for cards, ``/proc/meminfo`` for host RAM,
    ``statvfs`` for filesystems. These are readings, not estimates, and they
    are available on any Linux box with or without a driver;
*   **every cost is absent, and says which probe would fill it.** A card's
    membw, a host's DRAM bandwidth, an NVMe's latency: none of them can be
    known without running something, so none of them is filled in. The absence
    carries the probe id from :data:`sglang.srt.memtier.probe.PROBES`, so the
    dashboard shows a work item rather than a blank cell;
*   **nothing is declared that was not seen.** No remote host, no peer rig, no
    filesystem the caller did not name. An unknown machine's registry contains
    exactly its own memories, and the NORDSTERN-shaped remote rows come from a
    matched profile or not at all.

Volatility and admission are NOT measurements and are assigned here: a device's
VRAM is ``DEVICE_BOUND_ONLY`` on every machine ever built, host RAM dies with
the process, and a filesystem's persistence follows from its filesystem type
rather than from its path. Those are properties of the KIND, so assigning them
introduces no rig constant -- and the one that is genuinely path-dependent,
whether a mount survives a reboot, is read from the mount's own type rather
than guessed (a ``tmpfs`` pointed at by ``--hibernate-dir`` is the
silent-correctness hole this makes checkable).
"""

from __future__ import annotations

import os
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from sglang.srt.memtier.profile import (
    CardFact,
    FilesystemFact,
    LocalFacts,
    RigProfile,
)
from sglang.srt.memtier.tiers import (
    TierCapacity,
    TierCaps,
    TierDescriptor,
    TierHealth,
    TierKind,
    TierTransport,
    Volatility,
    device_tier_id,
    filesystem_tier_id,
    host_tier_id,
)
from sglang.srt.planner.cost_model import Rate

__all__ = [
    "BOOTSTRAP_PROFILE_ID",
    "NETWORK_FS_TYPES",
    "NON_PERSISTENT_FS_TYPES",
    "bootstrap_profile",
    "bootstrap_tiers",
    "collect_fs_types",
    "default_host_name",
    "filesystem_kind_properties",
]

#: The profile id a bootstrapped record carries. Not a rig name: the point is
#: that this record was NOT measured on any named rig.
BOOTSTRAP_PROFILE_ID = "bootstrap"

#: Filesystem types whose contents do not survive a reboot. A tier on one of
#: these is ``EXPENSIVE_OK`` and never ``PERSISTENT``, whatever its path looks
#: like -- ``/var/hibernate`` on a tmpfs is still a tmpfs.
NON_PERSISTENT_FS_TYPES: Tuple[str, ...] = ("tmpfs", "ramfs", "devtmpfs")

#: Filesystem types where POSIX advisory locking does not hold across the
#: clients that share the mount. ``flock`` is reported ``no`` for these, which
#: is what makes #89's dependency a capability query rather than a path check.
NETWORK_FS_TYPES: Tuple[str, ...] = (
    "nfs",
    "nfs4",
    "cifs",
    "smb3",
    "9p",
    "ceph",
    "glusterfs",
    "fuse.sshfs",
)

_ABSENT_PREFIX = (
    "bootstrap enumeration: this machine has no matched profile, so no cost "
    "for this tier has been measured"
)


def default_host_name() -> str:
    """The logical host component of every local tier id.

    ``socket.gethostname()`` deliberately: a bootstrapped record is
    machine-local and never shared, so a readable name beats an opaque one.
    A profile that wants a stable logical name across container renames sets
    :attr:`RigProfile.host` instead, which is why that field exists.
    """
    import socket

    return socket.gethostname() or "localhost"


def _absent(reason: str, unit: str) -> Rate:
    return Rate.absent(f"{_ABSENT_PREFIX}; {reason}", unit=unit)


def _unmeasured_caps(*, ledger_key: str, probes: str, aperture: str) -> TierCaps:
    return TierCaps(
        latency_us=_absent(f"a point latency would come from {probes}", "us"),
        bandwidth_gbs=_absent(f"a bandwidth would come from {probes}", "GB/s"),
        aperture_bytes=_absent(aperture, "bytes"),
        ledger_key=ledger_key,
    )


def filesystem_kind_properties(fs_type: str) -> Dict[str, str]:
    """Capability properties that follow from a filesystem TYPE, not a path.

    The three consumers that care -- #89's ``flock`` dependency, #306's
    in-place rewrite, #224's persistence requirement -- each asked a different
    name-shaped question of a path. Answering from the mount's type makes the
    answer checkable and makes a wrong ``--hibernate-dir`` a refusal.
    """
    lowered = (fs_type or "").strip().lower()
    known = bool(lowered)
    network = lowered in NETWORK_FS_TYPES
    persistent = known and lowered not in NON_PERSISTENT_FS_TYPES
    if not known:
        # An unreadable mount type makes every capability below unknowable.
        # Recording "unknown" rather than a default is what makes a
        # ``require={"flock": "yes"}`` query refuse this tier by name instead
        # of accepting it on an assumption -- #89's whole failure mode.
        return {
            "medium": "unknown",
            "fs_type": "unknown",
            "flock": "unknown",
            "persistent_across_reboot": "unknown",
            "rewritable": "unknown",
            "pointer_io": "no",
        }
    return {
        "medium": "tmpfs" if lowered in NON_PERSISTENT_FS_TYPES else "disk",
        "fs_type": lowered,
        "persistent_across_reboot": "yes" if persistent else "no",
        # POSIX advisory locks are a property of the mount, and a network
        # filesystem is where they stop working -- which is why #89 needs a
        # capability filter and not a filesystem-kind check.
        "flock": "no" if network else "yes",
        "rewritable": "yes",
        "pointer_io": "no",
    }


def bootstrap_tiers(
    facts: LocalFacts,
    *,
    host: str,
    fs_types: Optional[Mapping[str, str]] = None,
) -> Tuple[TierDescriptor, ...]:
    """Every memory this machine reports, with measured size and absent cost.

    ``fs_types`` maps a mount to its filesystem type. It is a parameter rather
    than a lookup so this function stays pure and a hermetic test can present a
    tmpfs and an ext4 without mounting either; :func:`collect_fs_types` is the
    live reader.
    """
    types = dict(fs_types or {})
    tiers: List[TierDescriptor] = [
        _device_tier(card, host=host) for card in facts.cards
    ]
    if facts.host_total_bytes is not None:
        tiers.append(_host_tier(facts, host=host))
    for fact in facts.filesystems:
        tiers.append(
            _filesystem_tier(fact, host=host, fs_type=types.get(fact.mount, ""))
        )
    return tuple(tiers)


def _device_tier(card: CardFact, *, host: str) -> TierDescriptor:
    return TierDescriptor(
        id=device_tier_id(card.uuid),
        kind=TierKind.DEVICE,
        host=host,
        capacity=TierCapacity(
            total=Rate.measured(
                float(card.total_bytes),
                f"NVML total for {card.uuid} ({card.model})",
                unit="bytes",
            ),
            floor=_absent(
                "the #330 pinned floor is measured once per boot as "
                "nvml_process_used_bytes - vmm_backed_bytes; it does not exist "
                "before a boot",
                "bytes",
            ),
            # #330's absolutely-free corridor is a property of a card under
            # this fork's own rule, not a measurement, and it is the same on
            # every card. It stays out of the bootstrap record because a
            # corridor is a policy the ledger owns; the registry reads it from
            # the ledger at cut 4 rather than restating it here.
            corridor=0,
        ),
        volatility=Volatility.DEVICE_BOUND_ONLY,
        admits=frozenset(),
        caps=_unmeasured_caps(
            ledger_key="vram",
            probes="a card membw probe (rigmon card_probe) for bandwidth and "
            "probe M1 for point latency",
            aperture="the effective BAR1 aperture comes from probe M2 "
            "(p2p_readiness capability_matrix); the nominal window is not it",
        ),
        health=TierHealth(reachable=True, verdict="ok"),
        transport=TierTransport(
            name="cuda-local",
            handle="torch device allocator",
            # The card's own BDF is one segment and is known; what is upstream
            # of it is not, so the path is INCOMPLETE and #423's disjointness
            # gate returns UNKNOWN rather than a false all-clear.
            link_path=(f"pcie:{card.bdf}",) if card.bdf else (),
            link_path_complete=False,
        ),
        properties={"medium": "vram"},
        profile_id=BOOTSTRAP_PROFILE_ID,
        card_model=card.model,
    )


def _host_tier(facts: LocalFacts, *, host: str) -> TierDescriptor:
    total = int(facts.host_total_bytes or 0)
    available = facts.host_available_bytes
    floor = (
        Rate.measured(
            float(total - int(available)),
            "live host memory total minus available: what is resident at this "
            "tier and is accounted in no ledger",
            unit="bytes",
        )
        if available is not None
        else _absent("MemAvailable could not be read", "bytes")
    )
    return TierDescriptor(
        id=host_tier_id(host),
        kind=TierKind.HOST,
        host=host,
        capacity=TierCapacity(
            total=Rate.measured(
                float(total), f"live host memory total ({facts.source})", unit="bytes"
            ),
            floor=floor,
        ),
        volatility=Volatility.EXPENSIVE_OK,
        admits=frozenset(),
        caps=_unmeasured_caps(
            ledger_key="host_ram",
            probes="probe M4, a sustained host-DRAM read arm",
            aperture="host RAM is not an aperture-gated tier",
        ),
        health=TierHealth(reachable=True, verdict="ok"),
        transport=TierTransport(name="pcie", handle="torch pinned H2D/D2H"),
        properties={"medium": "dram"},
        profile_id=BOOTSTRAP_PROFILE_ID,
    )


def _filesystem_tier(
    fact: FilesystemFact, *, host: str, fs_type: str
) -> TierDescriptor:
    properties = filesystem_kind_properties(fs_type)
    persistent = properties["persistent_across_reboot"] == "yes"
    return TierDescriptor(
        id=filesystem_tier_id(host, fact.mount),
        kind=TierKind.FILESYSTEM,
        host=host,
        capacity=TierCapacity(
            total=Rate.measured(
                float(fact.total_bytes), f"live statvfs on {fact.mount}", unit="bytes"
            ),
            floor=Rate.measured(
                float(fact.total_bytes - fact.available_bytes),
                f"live statvfs on {fact.mount}: bytes already in use by other "
                "content, which this tier cannot reclaim",
                unit="bytes",
            ),
        ),
        volatility=Volatility.PERSISTENT if persistent else Volatility.EXPENSIVE_OK,
        admits=frozenset(),
        caps=_unmeasured_caps(
            ledger_key=f"fs{fact.mount.replace('/', '_')}",
            probes="probe M5 for read latency and M6 for write-back behaviour",
            aperture="a filesystem is not an aperture-gated tier",
        ),
        health=TierHealth(reachable=True, verdict="ok"),
        transport=TierTransport(
            name="posix",
            handle=f"open/pread/pwrite on {fact.mount}",
            stages_through=host_tier_id(host),
        ),
        properties=properties,
        profile_id=BOOTSTRAP_PROFILE_ID,
    )


def bootstrap_profile(
    facts: LocalFacts,
    *,
    host: Optional[str] = None,
    fs_types: Optional[Mapping[str, str]] = None,
) -> RigProfile:
    """A :class:`RigProfile` over live facts alone, carrying no stored numbers.

    It has no hardware keys on purpose: a bootstrap record is not a claim about
    hardware, it is a reading of it, and it is never persisted from here. The
    save path is :func:`sglang.srt.memtier.profile_store.save_profile`, which
    keys what it writes from the live fingerprint.
    """
    resolved = host or default_host_name()
    return RigProfile(
        profile_id=BOOTSTRAP_PROFILE_ID,
        host=resolved,
        title="live enumeration, no stored profile matched this hardware",
        caveat=(
            "Sizes are live readings; every cost is ABSENT and names the probe "
            "that would fill it. No number in this record was measured on any "
            "other machine."
        ),
        tiers=bootstrap_tiers(facts, host=resolved, fs_types=fs_types),
    )


def collect_fs_types(mounts: Sequence[str]) -> Dict[str, str]:
    """``{mount: fs_type}`` from ``/proc/mounts``. The only I/O here.

    Resolved by longest matching mount point rather than by exact string, so a
    directory inside a mount inherits its filesystem type -- which is the case
    that matters: ``--hibernate-dir /dev/shm/img`` is on a tmpfs even though
    ``/dev/shm/img`` is not itself a mount point.
    """
    table: List[Tuple[str, str]] = []
    try:
        with open("/proc/mounts", "r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3:
                    table.append((parts[1], parts[2]))
    except OSError:
        return {}
    table.sort(key=lambda row: len(row[0]), reverse=True)
    resolved: Dict[str, str] = {}
    for mount in mounts:
        target = os.path.abspath(mount)
        for point, fs_type in table:
            if target == point or target.startswith(point.rstrip("/") + "/"):
                resolved[mount] = fs_type
                break
    return resolved
