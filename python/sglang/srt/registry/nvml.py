# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""NVML device identity: the physical-GPU key every tenant agrees on.

Physical GPU identity in the registry, the VRAM ledger and every tenant is
the **NVML UUID**, never the CUDA enumeration index. CUDA defaults to a ``FASTEST_FIRST`` device order while
NVML enumerates in PCI bus order, so ``cuda:1`` and NVML index 1 are
routinely different cards on a mixed rig. That divergence has already
produced a real defect here: a reserve line computed against the wrong
card's total memory because it reused a CUDA index as an NVML index. Every
consumer therefore keys on the UUID string
(``GPU-xxxxxxxx-...``) and converts to an index only at the moment it has to
talk to a specific API.

``pynvml`` is imported lazily behind :func:`_pynvml` so that the module
imports cleanly on a host without the NVIDIA management library, and so that
tests of the pure-arithmetic modules never touch a driver. Every entry point
raises :class:`NvmlUnavailableError` with an actionable message when the
library or the driver is missing.

CLI::

    python -m sglang.srt.registry.nvml --json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator, Sequence

logger = logging.getLogger(__name__)

MIB = 1024 * 1024


class NvmlUnavailableError(RuntimeError):
    """NVML cannot answer: the binding is missing, or the driver is not loaded."""


class DeviceNotFoundError(LookupError):
    """A requested device identity does not match any NVML device."""


class DeviceOrderUnresolvedError(RuntimeError):
    """The CUDA order of the present cards is not knowable in this process.

    Raised instead of inventing one (#397). Every historical instance of the
    device-order family -- the torch-vs-NVML memory read, runbook 6.1, the
    #331 audit, #349 sweep-3 arm L / #392 -- is a piece of code that answered
    "which card is index i" without being able to know, and was believed. A
    caller that cannot get a bridge has exactly two honest options: leave the
    CUDA-space value unset, or refuse. Both need this error to carry WHAT
    could not be placed and WHY, which is what the message does.
    """


@dataclass(frozen=True)
class DeviceInfo:
    """One physical GPU as NVML sees it.

    ``index`` is the NVML index, which is stable within a driver session but
    not across boots. ``uuid`` is the durable identity and the key used by
    the reservation ledger and the shard planner.
    """

    index: int
    uuid: str
    name: str
    total_bytes: int
    pci_bus_id: str

    @property
    def total_mib(self) -> int:
        return self.total_bytes // MIB

    def describe(self) -> str:
        return (
            f"nvml[{self.index}] {self.name} {self.total_mib} MiB "
            f"{self.pci_bus_id} {self.uuid}"
        )


def _pynvml():
    """Import ``pynvml`` or raise a message that says what to install."""
    try:
        import pynvml  # noqa: PLC0415 - deliberately lazy, see module docstring
    except ImportError as exc:
        raise NvmlUnavailableError(
            "pynvml is not installed, so physical GPU identity cannot be "
            "resolved. Install nvidia-ml-py (pip install nvidia-ml-py) or "
            "pass device identities explicitly."
        ) from exc
    return pynvml


def is_available() -> bool:
    """True when NVML can be initialised. Never raises."""
    try:
        with nvml_session():
            return True
    except Exception:
        return False


@contextmanager
def nvml_session() -> Iterator[Any]:
    """Init/shutdown pair around a batch of NVML queries.

    NVML reference-counts init, but pairing it here keeps every query in this
    module from leaking a handle when a caller raises mid-loop.
    """
    pynvml = _pynvml()
    try:
        pynvml.nvmlInit()
    except Exception as exc:
        raise NvmlUnavailableError(
            f"nvmlInit() failed ({exc}); the NVIDIA driver is not loaded or is "
            "not reachable from this process."
        ) from exc
    try:
        yield pynvml
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # pragma: no cover - shutdown failure is not actionable
            pass


def _decode(value: object) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def list_devices() -> list[DeviceInfo]:
    """Every physical GPU, in NVML index order.

    The returned ``index`` is the NVML index. It is intentionally *not* usable
    as a ``torch.cuda`` index: see the module docstring for why conflating the
    two has already cost a defect here.
    """
    with nvml_session() as pynvml:
        devices: list[DeviceInfo] = []
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            devices.append(
                DeviceInfo(
                    index=index,
                    uuid=_decode(pynvml.nvmlDeviceGetUUID(handle)),
                    name=_decode(pynvml.nvmlDeviceGetName(handle)),
                    total_bytes=int(mem.total),
                    pci_bus_id=_decode(pynvml.nvmlDeviceGetPciInfo(handle).busId),
                )
            )
        return devices


def device_by_uuid(uuid: str) -> DeviceInfo:
    for device in list_devices():
        if device.uuid == uuid:
            return device
    raise DeviceNotFoundError(
        f"no NVML device with UUID {uuid!r}; present: "
        f"{[d.uuid for d in list_devices()]}"
    )


def total_bytes_for_uuid(uuid: str) -> int:
    """Card total as NVML reports it. The ledger invariant's right-hand side."""
    return device_by_uuid(uuid).total_bytes


@dataclass(frozen=True)
class MemoryInfo:
    """One card's memory as the driver sees it, not as any tenant believes it."""

    total_bytes: int
    free_bytes: int
    used_bytes: int

    @property
    def free_mib(self) -> int:
        return self.free_bytes // MIB


def memory_info_for_uuid(uuid: str) -> MemoryInfo:
    """Total, free and used for one card.

    The corridor guard (#330: at least 400 MiB absolutely free) is a statement
    about *free*, which no ledger can compute -- reservations say what tenants
    promised to use, the driver says what is actually mapped, and a CUDA
    context that nobody declared still consumes bytes. Verification therefore
    reads here, independently of the ledger.
    """
    with nvml_session() as pynvml:
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            if _decode(pynvml.nvmlDeviceGetUUID(handle)) != uuid:
                continue
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            return MemoryInfo(
                total_bytes=int(mem.total),
                free_bytes=int(mem.free),
                used_bytes=int(mem.used),
            )
    raise DeviceNotFoundError(f"no NVML device with UUID {uuid!r}")


@dataclass(frozen=True)
class PcieLink:
    """The PCIe slot a card sits in: negotiated lanes, and link generation."""

    #: Lanes actually negotiated for the SLOT (``CurrPcieLinkWidth``).
    width: int
    #: Generation the link is CAPABLE of (``MaxPcieLinkGeneration``).
    generation: int
    #: True when ``width`` had to come from ``MaxPcieLinkWidth`` because the
    #: current-link query was unavailable. Such a width describes the CARD,
    #: not the slot, and consumers that switch on width should know.
    width_from_max: bool = False


def pcie_link_for_uuid(uuid: str) -> PcieLink | None:
    """The PCIe link of one card, or ``None`` when it cannot be read (#736).

    THE ONE AUTHORITY for this question. Two consumers ask it -- the
    expert-offload link weights (``layers/moe/expert_offload.py``) and the
    #732 per-peer transport binding
    (``distributed/device_communicators/barlink_peer_transport.py``) -- and
    they used to carry separate copies of the rule below. It lives here
    because this is the IdentityMap's home and both consumers import DOWN into
    it; the reverse (a transport reaching up into ``layers.moe``) would drag
    the MoE package into world build.

    WIDTH COMES FROM THE CURRENT LINK, GENERATION FROM THE MAXIMUM, and the
    asymmetry is measured rather than stylistic.

    ``nvmlDeviceGetMaxPcieLinkWidth`` reports what the CARD can do, not what
    the SLOT gives it: on the reference rig it returns x16 for all three cards
    while the slots are wired x4 / x8 / x8. Deriving width from it produces an
    equal ratio for the offload weights and collapses every edge to "fast" for
    the transport policy -- i.e. both features silently disabled on exactly
    the box they exist for. ``CurrPcieLinkWidth`` reports 4 / 8 / 8 and is the
    physical wiring.

    Generation is the other way round. The current LINK STATE idles down (all
    three cards report gen 1 at rest and would report 4 under load), so a
    current-generation read taken at weight-load time describes the power
    state rather than the slot. Only ratios matter to the callers and the
    generation is uniform across a single board's slots, so the maximum is
    both stable and sufficient.

    The card is resolved through the IdentityMap by UUID and only then
    converted to an NVML index. Never positionally: CUDA enumerates
    FASTEST_FIRST and NVML in bus order, so a rank's CUDA ordinal is not its
    NVML index on a mixed rig, and #392 is what happens when those are
    conflated.
    """
    try:
        card = identity_map().require(uuid)
        with nvml_session() as pynvml:
            handle = pynvml.nvmlDeviceGetHandleByIndex(int(card.nvml_index))
            generation = int(pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle))
            width, from_max = 0, False
            try:
                width = int(pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle))
            except Exception:  # noqa: BLE001 - older binding: fall back below
                width = 0
            if width <= 0:
                # Deliberate last resort, and known to over-report (see above).
                # Flagged, so a consumer that switches on width can tell.
                width = int(pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle))
                from_max = True
    except Exception:  # noqa: BLE001 - absent driver/binding is not an error
        return None
    if width <= 0:
        return None
    return PcieLink(width=width, generation=generation, width_from_max=from_max)


def process_bytes_on_uuid(uuid: str) -> dict[int, int]:
    """``{pid: device bytes}`` for every compute process on that card.

    This is the measured side of the ledger for a process-isolated tenant: the
    registry knows a tenant's pids, and NVML knows what those pids hold. No
    cooperation from inside the tenant is required, which is the point -- a
    tenant that lies or that has not reported yet is still accounted for.
    """
    with nvml_session() as pynvml:
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            if _decode(pynvml.nvmlDeviceGetUUID(handle)) != uuid:
                continue
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
            except AttributeError:  # older binding
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            out: dict[int, int] = {}
            for proc in procs:
                used = getattr(proc, "usedGpuMemory", None)
                # NVML reports None when the caller may not see the value
                # (another user's process under a restricted driver mode).
                out[int(proc.pid)] = int(used) if used is not None else 0
            return out
    raise DeviceNotFoundError(f"no NVML device with UUID {uuid!r}")


def resolve_devices_by_name_fragment(fragment: str) -> list[DeviceInfo]:
    """Every device whose product name contains ``fragment``, case-insensitive."""
    needle = fragment.lower()
    return [d for d in list_devices() if needle in d.name.lower()]


def resolve_index_by_name_fragment(fragment: str) -> int:
    """NVML index of the single device matching ``fragment``.

    Test harnesses use this to find, say, the 5090 at runtime instead of
    hard-coding an index that shifts between boots. Ambiguity is an error
    rather than a silent first-match, because "the 3080" on a two-3080 rig is
    not a well-formed question.
    """
    matches = resolve_devices_by_name_fragment(fragment)
    if not matches:
        raise DeviceNotFoundError(
            f"no NVML device name contains {fragment!r}; present: "
            f"{[d.name for d in list_devices()]}"
        )
    if len(matches) > 1:
        raise DeviceNotFoundError(
            f"{fragment!r} matches {len(matches)} devices "
            f"({[(d.index, d.name) for d in matches]}); use a longer fragment "
            "or resolve by UUID."
        )
    return matches[0].index


def _visible_device_tokens() -> list[str] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    return [tok for tok in (t.strip() for t in raw.split(",")) if tok]


def current_device_uuid() -> str:
    """NVML UUID of the GPU this process is pinned to.

    The supported pinning is one physical GPU per process via
    ``CUDA_VISIBLE_DEVICES``, which is what the Class-3 executor and the
    multi-rank-per-card layout both use: inside such a process ``cuda:0`` is
    unambiguous and no logical-to-physical mapping table is needed. Both
    forms of the variable are accepted, index and ``GPU-...`` UUID.

    When the variable is unset or names more than one device, the current
    torch device is bridged to NVML through the PCI bus id rather than
    through the index.
    """
    tokens = _visible_device_tokens()
    devices = list_devices()
    if tokens is not None and len(tokens) == 1:
        token = tokens[0]
        if token.startswith("GPU-") or token.startswith("MIG-"):
            for device in devices:
                if device.uuid == token:
                    return device.uuid
            raise DeviceNotFoundError(
                f"CUDA_VISIBLE_DEVICES={token!r} names a UUID NVML does not report"
            )
        try:
            index = int(token)
        except ValueError:
            raise DeviceNotFoundError(
                f"CUDA_VISIBLE_DEVICES={token!r} is neither an index nor a UUID"
            ) from None
        # CUDA_VISIBLE_DEVICES indices are NVML/PCI-order indices, unlike the
        # in-process torch indices they produce.
        for device in devices:
            if device.index == index:
                return device.uuid
        raise DeviceNotFoundError(
            f"CUDA_VISIBLE_DEVICES={index} is out of range for "
            f"{len(devices)} NVML device(s)"
        )
    return _current_device_uuid_via_torch(devices)


def _current_device_uuid_via_torch(devices: list[DeviceInfo]) -> str:
    try:
        import torch  # noqa: PLC0415 - optional, only for the unpinned case
    except ImportError as exc:
        raise NvmlUnavailableError(
            "CUDA_VISIBLE_DEVICES does not pin exactly one device and torch is "
            "not importable, so the current device cannot be identified. Pin "
            "the process to one physical GPU."
        ) from exc
    if not torch.cuda.is_available():
        raise NvmlUnavailableError(
            "CUDA_VISIBLE_DEVICES does not pin exactly one device and torch "
            "reports no CUDA device."
        )
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    bus = getattr(props, "pci_bus_id", None)
    if bus is None:
        raise NvmlUnavailableError(
            "torch does not expose pci_bus_id on this build, so the CUDA "
            "device cannot be bridged to NVML by bus id."
        )
    for device in devices:
        # NVML formats the bus id as "00000000:BB:00.0"; torch reports the bus
        # number as an int.
        if int(device.pci_bus_id.split(":")[1], 16) == bus:
            return device.uuid
    raise DeviceNotFoundError(
        f"current CUDA device sits on PCI bus {bus:#x}, which matches no NVML device"
    )


# ===========================================================================
# Canonical card identity (AUDIT #331)
#
# Everything above answers "what does NVML see". This section answers the
# question every *persisted* artifact has to answer instead: "which physical
# card is this record about, given that the enumeration it was written under
# is gone". The answer is the UUID; the indices are derived views of it,
# recomputed live at process start and never stored as the key.
#
# Three enumerations coexist on one host and disagree routinely:
#
#   NVML index      PCI bus order, what nvidia-smi prints.
#   CUDA ordinal    what torch.cuda sees, FASTEST_FIRST by default, and
#                   further remapped by CUDA_VISIBLE_DEVICES.
#   PCI BDF         the slot. Stable across boots, readable by a human,
#                   but not usable as a key when a card is reseated.
#
# On the reference rig the 5090 is CUDA ordinal 0 and NVML index 1. Every
# defect in this family (#305-M1, #336, #339, #340, #272) is one of those
# three being written down and read back as another.
# ===========================================================================


@dataclass(frozen=True)
class CardIdentity:
    """One physical GPU under all four names it answers to.

    ``uuid`` is the key. ``nvml_index`` and ``cuda_ordinal`` are the views
    valid for *this* process only, and ``cuda_ordinal`` is ``None`` when the
    card is not visible to torch at all (masked by ``CUDA_VISIBLE_DEVICES``,
    or torch absent).
    """

    uuid: str
    nvml_index: int
    pci_bus_id: str
    name: str
    total_bytes: int
    cuda_ordinal: int | None = None

    @property
    def total_mib(self) -> int:
        return self.total_bytes // MIB

    def describe(self) -> str:
        cuda = "-" if self.cuda_ordinal is None else str(self.cuda_ordinal)
        return (
            f"uuid={self.uuid}  nvml={self.nvml_index}  cuda={cuda}  "
            f"bdf={self.pci_bus_id}  {self.name} {self.total_mib} MiB"
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "uuid": self.uuid,
            "nvml_index": self.nvml_index,
            "cuda_ordinal": self.cuda_ordinal,
            "pci_bus_id": self.pci_bus_id,
            "name": self.name,
            "total_mib": self.total_mib,
        }


class IdentityMap:
    """The live UUID <-> NVML index <-> CUDA ordinal <-> BDF resolver.

    Built once per process from NVML (the source of truth) plus, when torch
    is importable, the CUDA-ordinal side bridged over the PCI bus id. Every
    persisted artifact that used to store an index resolves it through here
    on read and rewrites itself UUID-keyed.

    A resolver instance is a snapshot. It is deliberately not cached at
    module level: hotplug (#329) will invalidate it, and a stale map that
    silently answers is exactly the failure this audit removes.
    """

    def __init__(self, cards: Sequence[CardIdentity]):
        self.cards: tuple[CardIdentity, ...] = tuple(cards)
        self._by_uuid = {c.uuid: c for c in self.cards}
        self._by_nvml = {c.nvml_index: c for c in self.cards}
        self._by_cuda = {
            c.cuda_ordinal: c for c in self.cards if c.cuda_ordinal is not None
        }
        self._by_bdf = {_normalize_bdf(c.pci_bus_id): c for c in self.cards}

    # -- lookups ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self.cards)

    def __iter__(self) -> Iterator[CardIdentity]:
        return iter(self.cards)

    @property
    def uuids(self) -> tuple[str, ...]:
        return tuple(c.uuid for c in self.cards)

    def get(self, uuid: str) -> CardIdentity | None:
        return self._by_uuid.get(uuid)

    def require(self, uuid: str) -> CardIdentity:
        """The card, or a named error. Never a silent nearest match.

        A UUID that is no longer present means a card was pulled, disabled or
        moved to another host. The only correct answer is to say so: binding
        the record to whatever now occupies that index is the bug.
        """
        card = self._by_uuid.get(uuid)
        if card is None:
            raise DeviceNotFoundError(
                f"card {uuid} is not present on this host; visible now: "
                + (
                    ", ".join(f"{c.uuid} ({c.name}, nvml {c.nvml_index})" for c in self.cards)
                    or "no NVML devices"
                )
            )
        return card

    def by_nvml_index(self, index: int) -> CardIdentity | None:
        return self._by_nvml.get(int(index))

    def by_cuda_ordinal(self, ordinal: int) -> CardIdentity | None:
        return self._by_cuda.get(int(ordinal))

    def by_pci_bus_id(self, bus_id: str) -> CardIdentity | None:
        return self._by_bdf.get(_normalize_bdf(bus_id))

    def nvml_index_of(self, uuid: str) -> int:
        return self.require(uuid).nvml_index

    def cuda_ordinal_of(self, uuid: str) -> int:
        card = self.require(uuid)
        if card.cuda_ordinal is None:
            raise DeviceNotFoundError(
                f"card {uuid} ({card.name}, nvml {card.nvml_index}) is not "
                "visible to torch in this process; CUDA_VISIBLE_DEVICES masks it"
            )
        return card.cuda_ordinal

    # -- the CUDA side, all-or-nothing (#397) ------------------------------

    @property
    def unbridged(self) -> tuple[CardIdentity, ...]:
        """Cards whose CUDA ordinal this process could not determine."""
        return tuple(c for c in self.cards if c.cuda_ordinal is None)

    def require_cuda_ordinals(self, context: str) -> dict[str, int]:
        """``{uuid: CUDA ordinal}`` for EVERY card, or a named error.

        All-or-nothing on purpose. A partial bridge is worse than none: the
        caller fills the gaps with the number it already has -- the NVML
        index, or the list position -- and that silent substitution is the
        whole device-order defect family. ``context`` names the caller so the
        message says which decision was refused, not just that one was.
        """
        missing = self.unbridged
        if missing:
            raise DeviceOrderUnresolvedError(
                f"{context}: the CUDA enumeration order of "
                f"{len(missing)} of {len(self.cards)} card(s) could not be "
                "resolved, so no CUDA-space index can be produced for them. "
                "Unplaced:\n  "
                + "\n  ".join(c.describe() for c in missing)
                + f"\nReason: {cuda_bridge_diagnosis()}"
            )
        return {c.uuid: int(c.cuda_ordinal) for c in self.cards}

    def cuda_to_nvml(self, context: str) -> dict[int, int]:
        """``{CUDA ordinal: NVML index}``, the bridge in the direction the
        engine flags need it. Same all-or-nothing contract."""
        self.require_cuda_ordinals(context)
        return {int(c.cuda_ordinal): c.nvml_index for c in self.cards}

    # -- legacy migration --------------------------------------------------

    def adopt_legacy_indices(
        self, indices: Sequence[int], *, order: str = "nvml"
    ) -> list[str]:
        """Index-keyed legacy record -> UUIDs, under a stated assumption.

        This is the *migration* path, not a resolution path: an old artifact
        recorded an index without recording which enumeration it meant, so
        the best that can be done is to resolve it under the enumeration the
        writer is known to have used and rewrite the artifact UUID-keyed at
        once. ``order`` names that assumption explicitly rather than leaving
        it in a comment -- ``"nvml"`` for anything derived from nvidia-smi or
        NVML, ``"cuda"`` for anything derived from ``torch.cuda``.

        An index with no card behind it raises rather than being dropped: a
        legacy file naming card 3 on a two-card host is not a record whose
        remaining entries can be trusted either.
        """
        if order not in ("nvml", "cuda"):
            raise ValueError(f"order must be 'nvml' or 'cuda', not {order!r}")
        out: list[str] = []
        for raw in indices:
            index = int(raw)
            card = (
                self.by_nvml_index(index)
                if order == "nvml"
                else self.by_cuda_ordinal(index)
            )
            if card is None:
                raise DeviceNotFoundError(
                    f"legacy record names {order} index {index}, which matches "
                    f"no card on this host ({len(self.cards)} present: "
                    + ", ".join(c.describe() for c in self.cards)
                    + "). The record predates UUID keying and cannot be "
                    "migrated; re-measure it."
                )
            out.append(card.uuid)
        return out

    def describe(self) -> str:
        return "\n".join(c.describe() for c in self.cards)

    def to_json(self) -> list[dict[str, Any]]:
        return [c.to_json() for c in self.cards]


def _normalize_bdf(bus_id: str) -> str:
    """``00000000:2D:00.0`` and ``0000:2d:00.0`` are the same slot.

    NVML pads the domain to eight hex digits and upper-cases; lspci and the
    kernel use four and lower-case. Comparing the raw strings silently fails
    to match, so every BDF entering a lookup passes through here.
    """
    text = (bus_id or "").strip().lower()
    parts = text.split(":")
    if len(parts) == 3:
        domain, bus, rest = parts
        try:
            domain = f"{int(domain, 16):04x}"
        except ValueError:
            pass
        return f"{domain}:{bus}:{rest}"
    return text


def _cuda_ordinals_by_bus(allow_cuda_init: bool = False) -> dict[str, int]:
    """``{normalized BDF: CUDA ordinal}`` for the devices torch can see.

    Bridged over the bus id, never over the index: that identity is the whole
    point. Returns ``{}`` when torch is absent or reports no device, which is
    the desk case and not an error -- the map is then UUID+NVML only.

    ``allow_cuda_init`` guards a real side effect. ``get_device_properties``
    goes through ``torch.cuda._lazy_init``, which creates a CUDA context and
    costs a few hundred MiB of VRAM on every visible card. Most callers only
    want to know *which card* a stored uuid is -- the boot-path presence
    checks, the arbitration files -- and must not pay a context, least of all
    in the launcher before it forks its workers. So by default the bridge is
    built only when a context already exists; a caller that genuinely needs
    the CUDA side (the planner, resolving ``--rank-gpu-id``) asks for it.
    """
    try:
        import torch  # noqa: PLC0415 - optional; the map is useful without it
    except ImportError:
        return {}
    try:
        if not torch.cuda.is_available():
            return {}
        if not allow_cuda_init and not torch.cuda.is_initialized():
            return {}
        out: dict[str, int] = {}
        for ordinal in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(ordinal)
            bus = getattr(props, "pci_bus_id", None)
            if bus is None:
                continue
            domain = getattr(props, "pci_domain_id", 0) or 0
            device = getattr(props, "pci_device_id", 0) or 0
            out[_normalize_bdf(f"{domain:04x}:{bus:02x}:{device:02x}.0")] = ordinal
        return out
    except Exception as exc:  # noqa: BLE001 - a broken torch must not break NVML
        logger.debug("card identity: CUDA ordinal bridge unavailable: %s", exc)
        return {}


def cuda_bridge_diagnosis() -> str:
    """One sentence naming why the CUDA side of the map may be missing.

    Called only on the failure path, to turn "unresolved" into something the
    operator can act on. Never raises: a diagnosis that itself blows up would
    replace the real error with its own.
    """
    try:
        import torch  # noqa: PLC0415 - failure-path only
    except ImportError:
        return (
            "torch is not importable in this process, and only torch can "
            "report the CUDA enumeration order"
        )
    try:
        if not torch.cuda.is_available():
            return "torch reports no CUDA device in this process"
        if not torch.cuda.is_initialized():
            return (
                "no CUDA context exists in this process and the caller did "
                "not pass allow_cuda_init=True, so the CUDA order was not "
                "queried (creating a context costs VRAM on every visible card)"
            )
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            return (
                f"CUDA_VISIBLE_DEVICES={visible!r} masks part of the rig, so "
                "torch cannot place the cards it does not see"
            )
        return (
            f"torch sees {torch.cuda.device_count()} device(s) but none of "
            "them could be matched to an NVML card by PCI bus id (this torch "
            "build may not expose pci_bus_id on device properties)"
        )
    except Exception as exc:  # noqa: BLE001 - diagnosis must never mask the error
        return f"the CUDA side could not be inspected ({exc})"


def identity_map(
    devices: Sequence[DeviceInfo] | None = None,
    cuda_ordinals_by_bus: dict[str, int] | None = None,
    *,
    allow_cuda_init: bool = False,
) -> IdentityMap:
    """Build the live resolver. NVML is the source of truth for the card set.

    Both inputs are injectable so the hermetic tests can construct a rig
    whose CUDA and NVML orders deliberately disagree (the real shape here:
    5090 = CUDA 0 / NVML 1) without a driver.

    ``allow_cuda_init`` is the licence to create a CUDA context in order to
    fill in ``cuda_ordinal``; see :func:`_cuda_ordinals_by_bus`. Without it
    the CUDA side is filled only when a context already exists, and is
    ``None`` otherwise -- which is what every caller that keys on UUIDs
    actually needs.
    """
    infos = list(devices) if devices is not None else list_devices()
    bus_to_ordinal = (
        cuda_ordinals_by_bus
        if cuda_ordinals_by_bus is not None
        else _cuda_ordinals_by_bus(allow_cuda_init)
    )
    normalized = {_normalize_bdf(k): v for k, v in bus_to_ordinal.items()}
    return IdentityMap(
        [
            CardIdentity(
                uuid=info.uuid,
                nvml_index=info.index,
                pci_bus_id=info.pci_bus_id,
                name=info.name,
                total_bytes=info.total_bytes,
                cuda_ordinal=normalized.get(_normalize_bdf(info.pci_bus_id)),
            )
            for info in infos
        ]
    )


def cuda_to_nvml_index_map(*, allow_cuda_init: bool = True) -> dict[int, int]:
    """``{CUDA ordinal: NVML index}`` for the cards this process can place.

    The lenient form of :meth:`IdentityMap.cuda_to_nvml`, and the single
    delegation target of the deprecated bridges kept for their existing
    callers (#397). ``{}`` when the map cannot be built at all or when torch
    places no card -- which every one of those callers already treats as
    "no bridge". It is partial by construction, so a caller that must not
    substitute an index for a missing entry uses the strict form instead.
    """
    try:
        imap = identity_map(allow_cuda_init=allow_cuda_init)
    except Exception as exc:  # noqa: BLE001 - callers degrade, they do not crash
        logger.warning(
            "CUDA -> NVML index bridge unavailable (%s); %s",
            exc,
            cuda_bridge_diagnosis(),
        )
        return {}
    return {
        int(c.cuda_ordinal): c.nvml_index
        for c in imap
        if c.cuda_ordinal is not None
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sglang.srt.registry.nvml",
        description=(
            "Print physical GPU identity (NVML index -> uuid/name/total). "
            "Test harnesses derive card assignments from this instead of "
            "hard-coding indices."
        ),
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument(
        "--name-fragment",
        help="print only the NVML index of the single device matching this fragment",
    )
    parser.add_argument(
        "--map",
        action="store_true",
        help=(
            "print the full identity map (uuid <-> NVML index <-> CUDA ordinal "
            "<-> PCI BDF) as every persisted artifact resolves it"
        ),
    )
    args = parser.parse_args(argv)

    try:
        if args.name_fragment:
            print(resolve_index_by_name_fragment(args.name_fragment))
            return 0
        if args.map:
            imap = identity_map()
            if args.json:
                print(json.dumps(imap.to_json(), indent=2))
            else:
                print(imap.describe())
            return 0
        devices = list_devices()
    except (NvmlUnavailableError, DeviceNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([asdict(d) for d in devices], indent=2))
    else:
        for device in devices:
            print(device.describe())
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(_main())
