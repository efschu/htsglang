"""Shared helpers for the P2P readiness probe package.

Pure-parsing / planning functions live here so they can be unit-tested
CPU-only (test/registered/unit/distributed/test_p2p_readiness_scripts.py).
Nothing in this module touches CUDA at import time; the CUDA runtime is
reached only through the ctypes helpers, and only when a probe actually runs.

Device identity: every result row is keyed by PCI bus id, never by a bare
enumeration index. torch/cudart order (CUDA_DEVICE_ORDER, default
FASTEST_FIRST) and NVML/nvidia-smi order are DIFFERENT enumerations of the
same cards (measured on this rig, TP5-emulation notes); joining them by index
is the standing device-order trap. The join is done via
cudaDeviceGetPCIBusId <-> nvmlDeviceGetPciInfo.
"""

import ctypes
import datetime
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

KIB = 1024
MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024

# The nominal small-BAR window on cards without resizable-BAR-sized BAR1
# (the two RTX 3080 here). NOMINAL: how much of it is actually usable is a
# measurement, not a datum -- see effective-aperture probing.
SMALL_BAR_WINDOW_BYTES = 256 * MIB

SCHEMA_VERSION = 3


# --------------------------------------------------------------------------
# result schema
# --------------------------------------------------------------------------


@dataclass
class DeviceInfo:
    pci_bus_id: str
    name: str
    uuid: str
    nvml_index: int
    cuda_index: Optional[int]  # in-process cudart index after the PCI join
    vram_total_bytes: int
    bar1_total_bytes: Optional[int]  # NOMINAL upper bound (NVML/lspci)
    bar1_classification: Optional[str] = None  # "full" | "windowed" | None


@dataclass
class DirectedPairResult:
    """One ordered (src -> dst) pair. dst's BAR window is the constraint that
    matters for peer writes INTO dst."""

    src_pci: str
    dst_pci: str
    can_access_peer: Optional[bool] = None
    dst_bar1_nominal_bytes: Optional[int] = None
    dst_bar1_classification: Optional[str] = None
    # EFFECTIVE aperture, measured -- may be less than nominal
    # (addressability, reservations). None means "not measured yet".
    effective_max_single_copy_bytes: Optional[int] = None
    effective_max_region_chunked_bytes: Optional[int] = None
    probe_errors: List[str] = field(default_factory=list)


def result_envelope(kind: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "timestamp": datetime.datetime.now().isoformat(),
        "host": os.uname().nodename,
    }


def write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=asdict_fallback)
        f.write("\n")


def asdict_fallback(obj):
    try:
        return asdict(obj)
    except TypeError:
        return str(obj)


# --------------------------------------------------------------------------
# parsing (unit-tested, CPU-only)
# --------------------------------------------------------------------------


def classify_bar(bar1_total_bytes: Optional[int], vram_total_bytes: int) -> str:
    """'full' when the whole VRAM is BAR-addressable (resizable BAR active),
    'windowed' when only an aperture is. The label speaks about the NOMINAL
    size; effective usability is measured separately."""
    if bar1_total_bytes is None:
        return "unknown"
    return "full" if bar1_total_bytes >= vram_total_bytes else "windowed"


def parse_smi_bar1(text: str) -> Dict[str, int]:
    """nvidia-smi -q -d MEMORY -> {pci_bus_id: bar1_total_bytes}.

    The section looks like:
        GPU 00000000:01:00.0
            ...
            BAR1 Memory Usage
                Total                             : 256 MiB
    """
    out: Dict[str, int] = {}
    current_pci = None
    in_bar1 = False
    for line in text.splitlines():
        m = re.match(r"^GPU\s+([0-9A-Fa-f:.]+)\s*$", line.strip())
        if m:
            current_pci = normalize_pci(m.group(1))
            in_bar1 = False
            continue
        if re.match(r"^\s*BAR1 Memory Usage", line):
            in_bar1 = True
            continue
        if in_bar1:
            m = re.match(r"^\s*Total\s*:\s*(\d+)\s*(KiB|MiB|GiB)\s*$", line)
            if m and current_pci:
                mult = {"KiB": KIB, "MiB": MIB, "GiB": GIB}[m.group(2)]
                out[current_pci] = int(m.group(1)) * mult
                in_bar1 = False
    return out


def parse_lspci_regions(text: str) -> List[int]:
    """Region sizes (bytes) from one device's `lspci -vv` block:
        Region 1: Memory at ... (64-bit, prefetchable) [size=256M]
    Returns all memory region sizes, largest first -- the largest
    prefetchable one is the BAR1 candidate. NOMINAL only."""
    sizes = []
    for m in re.finditer(r"Region\s+\d+:\s+Memory at .*\[size=(\d+)([KMGT])\]", text):
        mult = {"K": KIB, "M": MIB, "G": GIB, "T": 1024 * GIB}[m.group(2)]
        sizes.append(int(m.group(1)) * mult)
    return sorted(sizes, reverse=True)


def normalize_pci(pci: str) -> str:
    """'00000000:01:00.0' / '0000:01:00.0' / '01:00.0' -> '0000:01:00.0'."""
    pci = pci.strip().lower()
    parts = pci.split(":")
    if len(parts) == 2:
        pci = "0000:" + pci
    elif len(parts) == 3 and len(parts[0]) > 4:
        pci = parts[0][-4:] + ":" + ":".join(parts[1:])
    return pci


def parse_topo_matrix(text: str) -> Dict[Tuple[str, str], str]:
    """`nvidia-smi topo -m` -> {(gpuA, gpuB): link} with keys like 'GPU0'.

    Keeps the raw link classes (X, PHB, PIX, PXB, NODE, SYS, NV#). The raw
    text should be stored alongside; this is only for programmatic diffing.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    header = None
    out: Dict[Tuple[str, str], str] = {}
    for ln in lines:
        cols = re.split(r"\s+", ln.strip())
        if header is None:
            if cols and cols[0].startswith("GPU"):
                # header row variant without leading cell
                header = cols
            elif len(cols) > 1 and cols[1].startswith("GPU"):
                header = cols[1:]
            continue
        if not cols[0].startswith("GPU"):
            continue
        row = cols[0]
        for i, val in enumerate(cols[1 : 1 + len(header)]):
            col = header[i]
            if col.startswith("GPU"):
                out[(row, col)] = val
    return out


NCCL_TRANSPORT_RE = re.compile(
    r"Channel\s+(\d+)(?:/\d+)?\s*:\s*(\d+)\[(\w+)\]\s*->\s*(\d+)\[(\w+)\]\s*"
    r"(?:\[(?:send|receive)\]\s*)?via\s+([A-Za-z0-9/_]+)"
)


def parse_nccl_transports(log_text: str) -> List[dict]:
    """NCCL_DEBUG=INFO lines like
        NCCL INFO Channel 00/0 : 0[1000] -> 1[2000] via P2P/CUMEM
        NCCL INFO Channel 00 : 0[0] -> 1[1] via SHM/direct/direct
        NCCL INFO Channel 00/0 : 1[2000] -> 0[1000] via NET/Socket/0
    -> [{channel, src_rank, src_busid, dst_rank, dst_busid, transport}, ...]
    The transport's FIRST path segment is the class (P2P / SHM / NET)."""
    rows = []
    for m in NCCL_TRANSPORT_RE.finditer(log_text):
        transport_full = m.group(6)
        rows.append(
            {
                "channel": int(m.group(1)),
                "src_rank": int(m.group(2)),
                "src_busid": m.group(3),
                "dst_rank": int(m.group(4)),
                "dst_busid": m.group(5),
                "transport": transport_full,
                "transport_class": transport_full.split("/")[0].upper(),
            }
        )
    return rows


def summarize_transport_classes(rows: List[dict]) -> Dict[str, str]:
    """{'0->1': 'P2P', ...}; mixed channels are reported as a sorted union
    like 'P2P+SHM' -- a finding, not an error."""
    by_pair: Dict[str, set] = {}
    for r in rows:
        by_pair.setdefault(f"{r['src_rank']}->{r['dst_rank']}", set()).add(
            r["transport_class"]
        )
    return {k: "+".join(sorted(v)) for k, v in by_pair.items()}


# --------------------------------------------------------------------------
# probe planning (unit-tested, CPU-only)
# --------------------------------------------------------------------------


def size_ladder(
    start: int = 64 * KIB,
    stop: int = 1 * GIB,
    around: int = SMALL_BAR_WINDOW_BYTES,
) -> List[int]:
    """Doubling ladder start..stop that ALWAYS brackets the small-BAR window
    boundary tightly: window-1MiB, window, window+1MiB are inserted so a
    windowed aperture shows up as a knee, not as a gap between two doubling
    points."""
    sizes = []
    s = start
    while s <= stop:
        sizes.append(s)
        s *= 2
    for extra in (around - MIB, around, around + MIB):
        if start <= extra <= stop and extra not in sizes:
            sizes.append(extra)
    return sorted(sizes)


def aperture_search_plan(largest_ok: int, first_fail: Optional[int]) -> Optional[int]:
    """Next probe size for the effective-aperture search, or None when done.

    Growing phase (first_fail is None): double from largest_ok (from 1 MiB).
    Refine phase: bisect [largest_ok, first_fail] to 1-MiB resolution.
    """
    if first_fail is None:
        return max(MIB, largest_ok * 2)
    if first_fail - largest_ok <= MIB:
        return None
    return largest_ok + (first_fail - largest_ok) // 2


# --------------------------------------------------------------------------
# ctypes CUDA runtime (touched only when a probe RUNS, never at import)
# --------------------------------------------------------------------------

_cudart = None


def cudart():
    global _cudart
    if _cudart is None:
        _cudart = ctypes.CDLL("libcudart.so")
    return _cudart


def cuda_check(err: int, what: str) -> None:
    if err != 0:
        lib = cudart()
        lib.cudaGetErrorString.restype = ctypes.c_char_p
        raise RuntimeError(
            f"{what}: cudaError {err} ({lib.cudaGetErrorString(err).decode()})"
        )


def cuda_device_count() -> int:
    n = ctypes.c_int()
    cuda_check(cudart().cudaGetDeviceCount(ctypes.byref(n)), "cudaGetDeviceCount")
    return n.value


def cuda_pci_bus_id(dev: int) -> str:
    buf = ctypes.create_string_buffer(64)
    cuda_check(
        cudart().cudaDeviceGetPCIBusId(buf, 64, dev), f"cudaDeviceGetPCIBusId({dev})"
    )
    return normalize_pci(buf.value.decode())


def cuda_can_access_peer(dev: int, peer: int) -> bool:
    r = ctypes.c_int()
    cuda_check(
        cudart().cudaDeviceCanAccessPeer(ctypes.byref(r), dev, peer),
        f"cudaDeviceCanAccessPeer({dev},{peer})",
    )
    return bool(r.value)


def nvml_devices() -> List[DeviceInfo]:
    """All GPUs via NVML, keyed by PCI. BAR1 via nvmlDeviceGetBAR1MemoryInfo
    where the binding has it; None otherwise (lspci then fills nominal)."""
    import pynvml

    pynvml.nvmlInit()
    try:
        devs = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            pci = pynvml.nvmlDeviceGetPciInfo(h)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            bar1 = None
            try:
                bar1 = int(pynvml.nvmlDeviceGetBAR1MemoryInfo(h).bar1Total)
            except Exception:  # noqa: BLE001 -- optional API, absence is data
                pass
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            uuid = pynvml.nvmlDeviceGetUUID(h)
            if isinstance(uuid, bytes):
                uuid = uuid.decode()
            busid = pci.busId.decode() if isinstance(pci.busId, bytes) else pci.busId
            devs.append(
                DeviceInfo(
                    pci_bus_id=normalize_pci(busid),
                    name=name,
                    uuid=uuid,
                    nvml_index=i,
                    cuda_index=None,
                    vram_total_bytes=int(mem.total),
                    bar1_total_bytes=bar1,
                )
            )
        return devs
    finally:
        pynvml.nvmlShutdown()


def join_cuda_indices(devs: List[DeviceInfo]) -> None:
    """Fill cuda_index by PCI join -- THE device-order-trap counter."""
    by_pci = {d.pci_bus_id: d for d in devs}
    for ci in range(cuda_device_count()):
        pci = cuda_pci_bus_id(ci)
        if pci in by_pci:
            by_pci[pci].cuda_index = ci
