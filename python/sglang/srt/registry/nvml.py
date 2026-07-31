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
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Iterator

MIB = 1024 * 1024


class NvmlUnavailableError(RuntimeError):
    """NVML cannot answer: the binding is missing, or the driver is not loaded."""


class DeviceNotFoundError(LookupError):
    """A requested device identity does not match any NVML device."""


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
    args = parser.parse_args(argv)

    try:
        if args.name_fragment:
            print(resolve_index_by_name_fragment(args.name_fragment))
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
