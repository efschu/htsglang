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
"""Rig profiles: the measured numbers, as data (#407 cut 1).

The registry hardcodes no capacity, no bandwidth and no latency. It loads a
**rig profile** -- a JSON document whose every number carries the probe it came
off and the machine it was measured on -- and the one bundled here
(``profiles/rig1.json``) describes exactly one machine: the htsglang
development rig. A different rig points ``SGLANG_MEMTIER_PROFILE`` at its own
file, or overlays a partial one on top of the bundled record.

Three shapes live in a profile, for three different lifetimes:

*   **device model templates** -- per card MODEL, because a membw figure is a
    property of a 5090 and not of one particular 5090. They bind to a
    :data:`~sglang.srt.memtier.tiers.TierId` only when a live card is
    enumerated, so nothing in the file names a UUID that will differ on the
    next machine;
*   **declared tiers** -- host RAM, filesystems, the remote rig -- which have
    stable ids because they are named by host and mount rather than by serial;
*   **live facts** -- what NVML, ``/proc/meminfo`` and ``statvfs`` say right
    now. :func:`apply_local_facts` folds them over the declared record, and
    that is the only path by which a capacity number becomes current.

The override rule, which is the reason this is a file and not a table: an
overlay may replace any field of any tier, but every rate it supplies must
carry its own source. A number without a source is refused at load, because
the failure this module exists to prevent is a measured figure from one rig
being read as a fact about all of them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import msgspec

from sglang.srt.memtier.tiers import (
    TierCapacity,
    TierCaps,
    TierDescriptor,
    TierHealth,
    TierId,
    TierKind,
    TierTransport,
    Volatility,
    device_tier_id,
)
from sglang.srt.planner.cost_model import Provenance, Rate

__all__ = [
    "BUNDLED_PROFILE_PATH",
    "CardFact",
    "DeviceModelTemplate",
    "FilesystemFact",
    "LocalFacts",
    "ProfileError",
    "RigProfile",
    "apply_local_facts",
    "bind_device_tiers",
    "bundled_profile",
    "collect_local_facts",
    "load_profile",
    "profile_from_json",
]

#: The profile that ships with the fork. It describes ONE machine.
BUNDLED_PROFILE_PATH = Path(__file__).with_name("profiles") / "rig1.json"

#: Environment override, so a different rig needs no code change.
PROFILE_PATH_ENV = "SGLANG_MEMTIER_PROFILE"

_SUPPORTED_SCHEMA_VERSION = 1


class ProfileError(ValueError):
    """A profile document that cannot be trusted, and why."""


# ---------------------------------------------------------------------------
# Decoding, with the provenance rules enforced at the door
# ---------------------------------------------------------------------------


def _rate_from_json(data: Mapping[str, Any], *, where: str, unit: str) -> Rate:
    """One JSON rate, or :class:`ProfileError` naming what is wrong with it.

    Two rules are enforced here rather than downstream, because a profile is
    read once and consulted forever:

    * ``value`` is ``None`` exactly when ``provenance`` is ``absent`` -- the
      same invariant ``Rate`` enforces, checked with the file's own name in
      the message;
    * ``source`` is never empty. For a measured rate it names the probe; for
      an absent one it is the entire content of the record. An unsourced
      number is how a rig figure becomes a universal truth.
    """
    try:
        provenance = Provenance(str(data["provenance"]))
    except KeyError as exc:
        raise ProfileError(f"{where}: rate has no provenance field") from exc
    except ValueError as exc:
        raise ProfileError(
            f"{where}: {data.get('provenance')!r} is not one of "
            f"{[p.value for p in Provenance]}"
        ) from exc
    source = str(data.get("source", "")).strip()
    if not source:
        raise ProfileError(
            f"{where}: a rate must name its source -- the probe it came off, "
            "or the reason it is absent"
        )
    value = data.get("value")
    if provenance is Provenance.ABSENT:
        if value is not None:
            raise ProfileError(f"{where}: an absent rate must not carry a value")
        return Rate.absent(source, unit=str(data.get("unit", unit)))
    if value is None:
        raise ProfileError(
            f"{where}: a {provenance.value} rate must carry a value; use "
            'provenance "absent" to record that it does not exist'
        )
    return Rate(float(value), provenance, source, str(data.get("unit", unit)), "")


def _capacity_from_json(data: Mapping[str, Any], *, where: str) -> TierCapacity:
    return TierCapacity(
        total=_rate_from_json(data["total"], where=f"{where}.total", unit="bytes"),
        floor=_rate_from_json(data["floor"], where=f"{where}.floor", unit="bytes"),
        reserved=int(data.get("reserved", 0)),
        corridor=int(data.get("corridor", 0)),
    )


def _caps_from_json(data: Mapping[str, Any], *, where: str) -> TierCaps:
    ledger_key = str(data.get("ledger_key", "")).strip()
    if not ledger_key:
        raise ProfileError(
            f"{where}: caps.ledger_key is required -- a reservation on this "
            "tier has to post to a NAMED bucket"
        )
    return TierCaps(
        latency_us=_rate_from_json(
            data["latency_us"], where=f"{where}.latency_us", unit="us"
        ),
        bandwidth_gbs=_rate_from_json(
            data["bandwidth_gbs"], where=f"{where}.bandwidth_gbs", unit="GB/s"
        ),
        aperture_bytes=_rate_from_json(
            data["aperture_bytes"], where=f"{where}.aperture_bytes", unit="bytes"
        ),
        ledger_key=ledger_key,
    )


def _health_from_json(data: Mapping[str, Any]) -> TierHealth:
    return TierHealth(
        reachable=bool(data.get("reachable", False)),
        verdict=str(data.get("verdict", "ok")),
        reason=str(data.get("reason", "")),
        last_seen_s=float(data.get("last_seen_s", 0.0)),
    )


def _transport_from_json(data: Mapping[str, Any]) -> TierTransport:
    return TierTransport(
        name=str(data.get("name", "")),
        handle=str(data.get("handle", "")),
        stages_through=data.get("stages_through") or None,
        link_path=tuple(str(s) for s in data.get("link_path", ())),
        link_path_complete=bool(data.get("link_path_complete", False)),
    )


class DeviceModelTemplate(msgspec.Struct, frozen=True, kw_only=True):
    """Everything known about a card MODEL, before a card is enumerated.

    Bound to a real card by :func:`bind_device_tiers`, which replaces the
    nominal capacity with what NVML says about that particular card.
    """

    model: str
    capacity: TierCapacity
    caps: TierCaps
    properties: Mapping[str, str] = {}
    volatility: Volatility = Volatility.DEVICE_BOUND_ONLY
    admits: frozenset = frozenset()


class RigProfile(msgspec.Struct, frozen=True, kw_only=True):
    """One machine's measured record, as loaded."""

    profile_id: str
    #: The rig's LOGICAL name -- the host component of every local tier id.
    #: Logical rather than ``socket.gethostname()`` so a container rename does
    #: not rewrite every tier id in a persisted ledger.
    host: str
    hostnames: Tuple[str, ...] = ()
    title: str = ""
    caveat: str = ""
    device_models: Mapping[str, DeviceModelTemplate] = {}
    tiers: Tuple[TierDescriptor, ...] = ()
    path: str = ""
    #: The hardware this profile's numbers were measured on, verbatim as the
    #: document states it: ``{"version", "cards": [...], "models": [...]}``.
    #: :func:`sglang.srt.memtier.fingerprint.match_profile` derives the keys
    #: from it. An empty block means the profile makes an unverifiable claim,
    #: and the matcher refuses it wholesale rather than trusting it.
    hardware: Mapping[str, Any] = {}

    def tier(self, tier_id: TierId) -> Optional[TierDescriptor]:
        for tier in self.tiers:
            if tier.id == tier_id:
                return tier
        return None


def _tier_from_json(data: Mapping[str, Any], *, profile_id: str) -> TierDescriptor:
    tier_id = str(data["id"])
    return TierDescriptor(
        id=tier_id,
        kind=TierKind(str(data["kind"])),
        host=str(data["host"]),
        capacity=_capacity_from_json(data["capacity"], where=f"{tier_id}.capacity"),
        volatility=Volatility(str(data["volatility"])),
        admits=frozenset(str(c) for c in data.get("admits", ())),
        caps=_caps_from_json(data["caps"], where=f"{tier_id}.caps"),
        health=_health_from_json(data.get("health", {})),
        transport=_transport_from_json(data.get("transport", {})),
        properties={
            str(k): str(v) for k, v in dict(data.get("properties", {})).items()
        },
        profile_id=profile_id,
        card_model=str(data.get("card_model", "")),
    )


def _device_model_from_json(model: str, data: Mapping[str, Any]) -> DeviceModelTemplate:
    return DeviceModelTemplate(
        model=model,
        capacity=_capacity_from_json(data["capacity"], where=f"{model}.capacity"),
        caps=_caps_from_json(data["caps"], where=f"{model}.caps"),
        properties={
            str(k): str(v) for k, v in dict(data.get("properties", {})).items()
        },
        volatility=Volatility(
            str(data.get("volatility", Volatility.DEVICE_BOUND_ONLY.value))
        ),
        admits=frozenset(str(c) for c in data.get("admits", ())),
    )


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    """Overlay wins per leaf; nested mappings merge rather than replace."""
    merged: Dict[str, Any] = dict(base)
    for key, value in overlay.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _merge_documents(
    base: Mapping[str, Any], overlay: Mapping[str, Any]
) -> Dict[str, Any]:
    """Merge two profile documents. Tiers are matched by ``id``, not by index.

    Positional matching is exactly the class of bug this package refuses
    elsewhere: an overlay that dropped one entry would silently retarget every
    later one.
    """
    merged = _deep_merge(
        {k: v for k, v in base.items() if k != "tiers"},
        {k: v for k, v in overlay.items() if k != "tiers"},
    )
    by_id: Dict[str, Dict[str, Any]] = {
        str(t["id"]): dict(t) for t in base.get("tiers", ())
    }
    order = list(by_id)
    for tier in overlay.get("tiers", ()):
        tier_id = str(tier["id"])
        if tier_id in by_id:
            by_id[tier_id] = _deep_merge(by_id[tier_id], tier)
        else:
            by_id[tier_id] = dict(tier)
            order.append(tier_id)
    merged["tiers"] = [by_id[i] for i in order]
    return merged


def profile_from_json(
    document: Mapping[str, Any],
    *,
    base: Optional[Mapping[str, Any]] = None,
    path: str = "",
) -> RigProfile:
    """Build a profile from a decoded document, optionally over a base one."""
    data = _merge_documents(base, document) if base is not None else dict(document)
    version = int(data.get("schema_version", 0))
    if version != _SUPPORTED_SCHEMA_VERSION:
        raise ProfileError(
            f"{path or '<document>'}: schema_version {version} is not the "
            f"supported {_SUPPORTED_SCHEMA_VERSION}"
        )
    profile_id = str(data["profile_id"])
    tiers = tuple(
        _tier_from_json(t, profile_id=profile_id) for t in data.get("tiers", ())
    )
    seen: Dict[str, int] = {}
    for tier in tiers:
        seen[tier.id] = seen.get(tier.id, 0) + 1
    duplicates = sorted(i for i, n in seen.items() if n > 1)
    if duplicates:
        raise ProfileError(f"{path or '<document>'}: duplicate tier ids {duplicates}")
    return RigProfile(
        profile_id=profile_id,
        hardware=dict(data.get("hardware") or {}),
        host=str(data["host"]),
        hostnames=tuple(str(h) for h in data.get("hostnames", ())),
        title=str(data.get("title", "")),
        caveat=str(data.get("caveat", "")),
        device_models={
            model: _device_model_from_json(model, spec)
            for model, spec in dict(data.get("device_models", {})).items()
        },
        tiers=tiers,
        path=path,
    )


def load_profile(
    path: Optional[os.PathLike] = None, *, over_bundled: bool = False
) -> RigProfile:
    """Load a profile from disk.

    ``path`` defaults to ``$SGLANG_MEMTIER_PROFILE`` and then to the bundled
    rig record. ``over_bundled`` reads the file as a partial OVERLAY on the
    bundled document, which is how a second machine states only what differs.
    """
    chosen = Path(path) if path is not None else _default_profile_path()
    try:
        document = json.loads(chosen.read_text())
    except FileNotFoundError as exc:
        raise ProfileError(f"no memory-tier profile at {chosen}") from exc
    except json.JSONDecodeError as exc:
        raise ProfileError(f"{chosen}: not valid JSON ({exc})") from exc
    base = None
    if over_bundled and chosen != BUNDLED_PROFILE_PATH:
        base = json.loads(BUNDLED_PROFILE_PATH.read_text())
    return profile_from_json(document, base=base, path=str(chosen))


def _default_profile_path() -> Path:
    override = os.environ.get(PROFILE_PATH_ENV, "").strip()
    return Path(override) if override else BUNDLED_PROFILE_PATH


def bundled_profile() -> RigProfile:
    """The record measured on the htsglang development rig. One machine."""
    return load_profile(BUNDLED_PROFILE_PATH)


# ---------------------------------------------------------------------------
# Live facts
# ---------------------------------------------------------------------------


class CardFact(msgspec.Struct, frozen=True, kw_only=True):
    """One card as NVML sees it. UUID-keyed; no ordinal, ever.

    ``bdf`` is the #397 readable secondary and is carried because the
    ``p2p_readiness`` capability matrix keys its rows by PCI address rather
    than by UUID (``barlink_path_rates.py:145-147``); the adapter that ingests
    it needs a BDF -> UUID resolution and refuses without one. It is NOT part
    of the hardware fingerprint: a card that moves slots keeps its UUID.
    """

    uuid: str
    model: str
    total_bytes: int
    bdf: str = ""


class FilesystemFact(msgspec.Struct, frozen=True, kw_only=True):
    """One mount as ``statvfs`` sees it."""

    mount: str
    total_bytes: int
    available_bytes: int


class LocalFacts(msgspec.Struct, frozen=True, kw_only=True):
    """What this machine says about itself right now.

    Every field is optional and every absence is honest: a fact that could not
    be collected leaves the declared (measured-earlier) number in place rather
    than overwriting it with a zero.
    """

    cards: Tuple[CardFact, ...] = ()
    host_total_bytes: Optional[int] = None
    host_available_bytes: Optional[int] = None
    filesystems: Tuple[FilesystemFact, ...] = ()
    source: str = "live"


def bind_device_tiers(
    profile: RigProfile, cards: Sequence[CardFact]
) -> Tuple[TierDescriptor, ...]:
    """Turn card-model templates plus enumerated cards into DEVICE tiers.

    A card whose model has no template still gets a tier -- with its NVML
    capacity and every cap ABSENT, naming the missing template. Dropping it
    would be the omission failure DESIGN_407 §3.4 forbids: an unenumerated
    spill target silently becomes a different spill target.
    """
    tiers = []
    for card in cards:
        template = profile.device_models.get(card.model)
        caps = (
            template.caps if template is not None else _unknown_model_caps(card.model)
        )
        capacity = TierCapacity(
            total=Rate.measured(
                float(card.total_bytes),
                f"NVML total for {card.uuid} ({card.model})",
                unit="bytes",
            ),
            floor=(
                template.capacity.floor
                if template is not None
                else Rate.absent(
                    "no per-boot floor measurement for this card", unit="bytes"
                )
            ),
            corridor=template.capacity.corridor if template is not None else 0,
        )
        properties = dict(template.properties) if template is not None else {}
        if template is None:
            properties["model_template"] = (
                f"absent: profile {profile.profile_id} has no measured record "
                f"for {card.model!r}; capacity is live, every cap is absent"
            )
        tiers.append(
            TierDescriptor(
                id=device_tier_id(card.uuid),
                kind=TierKind.DEVICE,
                host=profile.host,
                capacity=capacity,
                volatility=(
                    template.volatility
                    if template is not None
                    else Volatility.DEVICE_BOUND_ONLY
                ),
                admits=template.admits if template is not None else frozenset(),
                caps=caps,
                health=TierHealth(reachable=True, verdict="ok"),
                transport=TierTransport(
                    name="cuda-local",
                    handle="torch device allocator; peer moves via barlink BAR1",
                ),
                properties=properties,
                profile_id=profile.profile_id,
                card_model=card.model,
            )
        )
    return tuple(tiers)


def _unknown_model_caps(model: str) -> TierCaps:
    reason = (
        f"no measured record for card model {model!r} in this profile; a "
        "roofline is not substituted (#348b D4: a missing bandwidth must not "
        "read as an extremely slow but usable tier)"
    )
    return TierCaps(
        latency_us=Rate.absent(reason, unit="us"),
        bandwidth_gbs=Rate.absent(reason, unit="GB/s"),
        aperture_bytes=Rate.absent(reason, unit="bytes"),
        ledger_key="vram",
    )


def apply_local_facts(
    tiers: Iterable[TierDescriptor], facts: LocalFacts
) -> Tuple[TierDescriptor, ...]:
    """Fold live capacity over declared tiers. Pure; nothing here does I/O.

    Only capacity moves. A live reading is a capacity fact and never a
    bandwidth one, so no cap is touched -- a tier does not become measured
    because a ``statvfs`` succeeded.
    """
    by_mount = {f.mount: f for f in facts.filesystems}
    updated = []
    for tier in tiers:
        if tier.kind is TierKind.HOST and facts.host_total_bytes is not None:
            updated.append(_with_host_facts(tier, facts))
            continue
        mount = tier.parsed.mount
        if tier.kind is TierKind.FILESYSTEM and mount in by_mount:
            updated.append(_with_filesystem_facts(tier, by_mount[mount]))
            continue
        updated.append(tier)
    return tuple(updated)


def _with_host_facts(tier: TierDescriptor, facts: LocalFacts) -> TierDescriptor:
    total = int(facts.host_total_bytes or 0)
    available = facts.host_available_bytes
    floor = tier.capacity.floor
    if available is not None:
        floor = Rate.measured(
            float(total - int(available)),
            "live host memory total minus available: what is resident at this "
            "tier and is accounted in no ledger",
            unit="bytes",
        )
    capacity = TierCapacity(
        total=Rate.measured(
            float(total), f"live host memory total ({facts.source})", unit="bytes"
        ),
        floor=floor,
        reserved=tier.capacity.reserved,
        corridor=tier.capacity.corridor,
    )
    return tier.evolve(capacity=capacity)


def _with_filesystem_facts(
    tier: TierDescriptor, fact: FilesystemFact
) -> TierDescriptor:
    capacity = TierCapacity(
        total=Rate.measured(
            float(fact.total_bytes), f"live statvfs on {fact.mount}", unit="bytes"
        ),
        floor=Rate.measured(
            float(fact.total_bytes - fact.available_bytes),
            f"live statvfs on {fact.mount}: bytes already in use by other "
            "content, which this tier cannot reclaim",
            unit="bytes",
        ),
        reserved=tier.capacity.reserved,
        corridor=tier.capacity.corridor,
    )
    return tier.evolve(capacity=capacity)


def collect_local_facts(
    *,
    mounts: Sequence[str] = (),
    cards: Optional[Sequence[CardFact]] = None,
) -> LocalFacts:
    """Read what this machine says about itself. The only I/O in this module.

    ``cards`` is injectable and defaults to NOT touching NVML: the coarse,
    driver-free query is what #89's arg-parse-time gate needs, because the
    CUDA half of an identity map creates a context worth a few hundred MiB on
    every visible card. Pass :func:`nvml_card_facts` explicitly when a driver
    is wanted.
    """
    total, available = _host_memory_bytes()
    filesystems = []
    for mount in mounts:
        try:
            stat = os.statvfs(mount)
        except OSError:
            continue
        filesystems.append(
            FilesystemFact(
                mount=mount,
                total_bytes=stat.f_frsize * stat.f_blocks,
                available_bytes=stat.f_frsize * stat.f_bavail,
            )
        )
    return LocalFacts(
        cards=tuple(cards or ()),
        host_total_bytes=total,
        host_available_bytes=available,
        filesystems=tuple(filesystems),
    )


def honest_host_memory_bytes(
    meminfo_total: Optional[int],
    meminfo_available: Optional[int],
    cgroup_max: Optional[int],
    cgroup_anon: Optional[int],
    cgroup_kernel: Optional[int],
    cgroup_shmem: Optional[int] = None,
    swap_free: Optional[int] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """``(total, available)`` a PINNED allocation may believe, or ``(None, None)``.

    WHY THIS IS NOT JUST ``/proc/meminfo``. Inside a container ``/proc/meminfo``
    is synthesised by lxcfs and is not a description of what this process may
    have. Two ways it lies, both observed on this rig:

    * ``MemAvailable`` can EXCEED ``MemTotal`` -- an arithmetic impossibility on
      a real machine, so any guard comparing a request against it is comparing
      against a number that does not denote anything;
    * with ``memory.max`` unlimited it reports the HOST's figures, which
      over-promise on a box whose RAM other containers are also spending.

    ``/sys/fs/cgroup`` is the honest source: ``memory.max`` is the hard ceiling
    this process group will be killed at, and ``memory.stat``'s ``anon`` is what
    is genuinely resident and unreclaimable, as opposed to ``file`` (page cache),
    which a pinning allocation can reclaim.

    SHMEM IS THE EXCEPTION TO THAT LAST CLAUSE, and it is why this function
    grew two more inputs (#695). In cgroup v2 tmpfs/shmem pages are accounted
    under ``file``, not under ``anon`` -- so a process holding a large
    ``MAP_SHARED`` region was priced here as holding reclaimable page cache.
    Measured on the PP=3 boot of 2026-08-12: ``anon`` 13.6 GiB, ``file``
    84.9 GiB, of which ``shmem`` 75.07 GiB, and ``SwapTotal: 0``. Shmem with
    no swap has nowhere to be reclaimed TO -- the pages are pinned for the
    lifetime of the mapping -- yet all 75.07 GiB counted as free. Nine
    cumulative cgroup OOM kills followed, one of them presenting as a silent
    rank death.

    The correction is deliberately NOT "subtract ``file``": the paragraph above
    is right that page cache is reclaimable and that subtracting it would
    refuse boots that would have succeeded. It is to subtract the
    UNRECLAIMABLE SUBSET of ``file``, which the cgroup already reports under
    its own key. Reclaimability of shmem is a function of swap rather than a
    constant, so the rule is stated against swap: only the part of ``shmem``
    that exceeds FREE SWAP is unreclaimable. On a swapless box that reduces to
    all of it, which is this rig; on a box with swap the term shrinks by
    itself and no boot is refused that would have fitted.

    The rules, in order:

    1. A FINITE ``memory.max`` is the ceiling, whatever ``/proc/meminfo`` says
       (and never above ``MemTotal`` when that is known and smaller -- the
       kernel cannot hand out RAM the machine does not have). Available is that
       ceiling minus what is anonymously resident and minus kernel memory; page
       cache is deliberately NOT subtracted, since it is reclaimable and
       subtracting it would refuse boots that would have succeeded.
    2. With no finite limit, ``MemTotal`` is the ceiling and ``MemAvailable``
       the estimate -- but CLAMPED to the ceiling, which is the whole fix for
       the impossible reading, and clamped again by the cgroup's own resident
       accounting when it is readable, so the number never promises memory this
       cgroup has already spent.
    3. Anything that cannot be established stays ``None`` rather than being
       guessed. A caller that cannot get a number must say so, not invent one.

    Pure: every input is passed in, so the decision is testable without a
    container, a cgroup, or a particular machine.
    """
    total: Optional[int] = None
    if cgroup_max is not None and cgroup_max > 0:
        total = cgroup_max
        if meminfo_total is not None and meminfo_total > 0:
            total = min(total, meminfo_total)
    elif meminfo_total is not None and meminfo_total > 0:
        total = meminfo_total
    if total is None:
        return None, None

    resident = 0
    have_resident = False
    for part in (cgroup_anon, cgroup_kernel):
        if part is not None and part >= 0:
            resident += int(part)
            have_resident = True

    # #695: the UNRECLAIMABLE part of `file`. shmem can only be reclaimed to
    # swap, so free swap is exactly how much of it is not pinned; whatever
    # exceeds that is as unavailable as `anon` and is charged next to it. No
    # double count -- shmem is inside `file`, and `file` is never charged.
    # Unknown shmem stays UNCHARGED rather than guessed, per rule 3: a caller
    # that cannot establish the term must degrade to the old answer, not to an
    # invented one.
    if cgroup_shmem is not None and cgroup_shmem >= 0:
        swappable = int(swap_free) if swap_free is not None and swap_free > 0 else 0
        resident += max(0, int(cgroup_shmem) - swappable)
        have_resident = True

    if cgroup_max is not None and cgroup_max > 0:
        if not have_resident:
            # A ceiling with no usage accounting tells us nothing about how much
            # of it is still free; refuse to guess.
            return total, None
        return total, max(0, total - resident)

    if meminfo_available is None:
        return total, (max(0, total - resident) if have_resident else None)
    available = min(int(meminfo_available), total)
    if have_resident:
        available = min(available, max(0, total - resident))
    return total, max(0, available)


def _read_meminfo() -> Tuple[Optional[int], Optional[int]]:
    values: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    values[key] = int(rest.strip().split()[0]) * 1024
    except OSError:
        return None, None
    return values.get("MemTotal"), values.get("MemAvailable")


def _read_cgroup_memory() -> (
    Tuple[Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]]
):
    """``(memory.max, anon, kernel, shmem, swap_free)``, any of them ``None``.

    ``memory.max`` reads ``"max"`` when unlimited, which becomes ``None`` here:
    "no ceiling I can name" rather than a number. cgroup v1 and a missing
    cgroupfs both fall out as all-``None``, so the caller degrades to
    ``/proc/meminfo`` on those.

    ``shmem`` (#695) is read from the same ``memory.stat`` as ``anon``. It is a
    SUBSET of that file's ``file`` key, not of ``anon``, which is the whole
    reason it needed its own term: see :func:`honest_host_memory_bytes`.

    ``swap_free`` is how much of that shmem could still be pushed out, and the
    honest source depends on the cgroup. A finite ``memory.swap.max`` bounds
    this process group regardless of what the machine has left, so the headroom
    is ``memory.swap.max - memory.swap.current``. With swap unlimited the bound
    is the machine's, so ``/proc/meminfo``'s ``SwapFree`` is read instead --
    the one figure in that file that lxcfs does not synthesise into an
    impossible reading, since a container has no swap of its own to report.
    """
    root = "/sys/fs/cgroup"
    limit: Optional[int] = None
    try:
        with open(f"{root}/memory.max", "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
        limit = None if raw == "max" else int(raw)
    except (OSError, ValueError):
        limit = None
    anon: Optional[int] = None
    kernel: Optional[int] = None
    shmem: Optional[int] = None
    try:
        with open(f"{root}/memory.stat", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(" ")
                if key == "anon":
                    anon = int(rest.strip())
                elif key == "kernel":
                    kernel = int(rest.strip())
                elif key == "shmem":
                    shmem = int(rest.strip())
    except (OSError, ValueError):
        pass
    return limit, anon, kernel, shmem, _swap_headroom_bytes(root)


def _swap_headroom_bytes(root: str) -> Optional[int]:
    """Bytes of swap this cgroup could still push shmem into, or ``None``."""
    swap_max: Optional[int] = None
    try:
        with open(f"{root}/memory.swap.max", "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
        swap_max = None if raw == "max" else int(raw)
    except (OSError, ValueError):
        swap_max = None
    if swap_max is not None:
        try:
            with open(f"{root}/memory.swap.current", "r", encoding="utf-8") as handle:
                used = int(handle.read().strip())
        except (OSError, ValueError):
            used = 0
        return max(0, swap_max - used)
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                if key == "SwapFree":
                    return int(rest.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def host_memory_bytes_for_pinning() -> Tuple[Optional[int], Optional[int]]:
    """Public name for :func:`_host_memory_bytes`.

    The #407 charter makes this module the OWNER of the host-memory number, so
    consumers that must size a PINNED pool ask here instead of adding another
    ``psutil.virtual_memory()`` call site. It is deliberately the whole pair --
    a caller that only wants "available" still has to look at the ceiling it is
    a fraction of.
    """
    return _host_memory_bytes()


def _host_memory_bytes() -> Tuple[Optional[int], Optional[int]]:
    """``(total, available)`` in bytes, or ``(None, None)``.

    ``/proc/meminfo`` plus ``/sys/fs/cgroup`` rather than ``psutil`` on purpose:
    DESIGN_407 §2.9 counts eight unrelated ``psutil.virtual_memory()`` call
    sites with no owner, and this module is meant to become that owner rather
    than the ninth. :func:`honest_host_memory_bytes` carries the reasoning for
    why the cgroup has to be consulted at all.
    """
    meminfo_total, meminfo_available = _read_meminfo()
    (
        cgroup_max,
        cgroup_anon,
        cgroup_kernel,
        cgroup_shmem,
        swap_free,
    ) = _read_cgroup_memory()
    return honest_host_memory_bytes(
        meminfo_total,
        meminfo_available,
        cgroup_max,
        cgroup_anon,
        cgroup_kernel,
        cgroup_shmem,
        swap_free,
    )


def nvml_card_facts() -> Tuple[CardFact, ...]:
    """Enumerate cards through NVML. Creates no CUDA context.

    Deliberately separate from :func:`collect_local_facts` so that the
    driver-free path stays driver-free and a caller has to ask for the driver
    by name.
    """
    from sglang.srt.registry.nvml import list_devices

    return tuple(
        CardFact(
            uuid=d.uuid,
            model=d.name,
            total_bytes=int(d.total_bytes),
            bdf=getattr(d, "pci_bus_id", "") or "",
        )
        for d in list_devices()
    )
