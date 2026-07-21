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
"""CUDA-order <-> NVML-order device-index bridge for the planner dashboard.

Two DIFFERENT index spaces name the same physical cards:

  * CUDA / rank space -- the space the ENGINE flags live in. ``--rank-gpu-id``
    and ``--base-gpu-id`` values end up in ``CUDA_VISIBLE_DEVICES``
    (``utils/common.py maybe_reindex_device_id``), which is interpreted in
    CUDA enumeration order. With no ``CUDA_DEVICE_ORDER`` set (our launches)
    that is FASTEST_FIRST: the fastest card is ``cuda:0``.
  * NVML / physical space -- what pynvml / nvidia-smi enumerate, in PCI bus
    order. All telemetry (power, temperature, memory) is sampled here.

On a mixed rig the two DIVERGE (the reference box: cuda:0 = RTX 5090 =
nvml:1; the two RTX 3080s are cuda:1 = nvml:0 and cuda:2 = nvml:2). Any
index that crosses between the spaces without translation silently names a
different card. This module is the planner's ONE bridge: per physical card
``{uuid, name, total_mib, nvml_index, cuda_index}``, resolved by

  1. torch enumeration matched to NVML by GPU UUID (source ``"torch"``) --
     exact, the same technique as the engine's
     ``server_args._torch_to_nvml_gpu_index_mapping`` (which bridges via PCI
     bus IDs); skipped when ``CUDA_VISIBLE_DEVICES`` filters the process
     (torch would then see a subset in a remapped order);
  2. else a documented FASTEST_FIRST EMULATION (source ``"heuristic"``):
     stable sort by the SEED_PROFILES fp16 GEMM peak, descending. CUDA's
     FASTEST_FIRST ranks by device performance and breaks ties in PCI order,
     which the stable sort reproduces; unknown card names rank 0 and keep
     NVML order. Callers must surface ``source == "heuristic"`` to the user.

Never crashes without a GPU: no NVML -> an empty map with ``source=None``.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Dict, List, Optional, Sequence

__all__ = [
    "DeviceMapEntry",
    "DeviceMap",
    "build_device_map",
    "device_map",
    "emulate_cuda_order",
    "seed_fp16_peak",
    "norm_uuid",
]


def norm_uuid(u) -> str:
    """Normalize any GPU-UUID spelling ('GPU-xxxx-...', bytes, uuid.UUID) to
    plain lowercase hex, so torch / NVML / nvidia-smi spellings compare."""
    s = u.decode() if isinstance(u, bytes) else str(u)
    return "".join(ch for ch in s.lower() if ch in "0123456789abcdef")


def seed_fp16_peak(name) -> float:
    """Best-effort fp16 tensor-core peak for a card NAME from the planner's
    SEED_PROFILES (bidirectional substring match, mirroring
    ``flags._gpu_flops``). Unknown names return 0.0."""
    try:
        from sglang.srt.planner.profiles import SEED_PROFILES

        low = str(name or "").lower()
        for p in SEED_PROFILES.values():
            if p.name.lower() in low or low in p.name.lower():
                return float(p.peak_gemm_tflops_fp16 or 0.0)
    except Exception:
        pass
    return 0.0


def emulate_cuda_order(names: Sequence[str]) -> List[int]:
    """FASTEST_FIRST emulation: the CUDA index each input position would get.

    ``names[i]`` is the card at NVML/list position ``i``; the returned list
    gives its emulated CUDA index. Stable sort by SEED_PROFILES fp16 peak
    descending -- CUDA's FASTEST_FIRST orders by device performance and keeps
    PCI order among equals, which a stable descending sort reproduces. This is
    a HEURISTIC (CUDA's internal ranking is not exactly the fp16 GEMM peak);
    callers must mark results from this path as such."""
    order = sorted(
        range(len(names)), key=lambda i: (-seed_fp16_peak(names[i]), i)
    )
    out = [0] * len(names)
    for cuda_idx, pos in enumerate(order):
        out[pos] = cuda_idx
    return out


@dataclasses.dataclass(frozen=True)
class DeviceMapEntry:
    """One physical card under BOTH index spaces."""

    nvml_index: int
    cuda_index: int
    name: str
    total_mib: int
    #: normalized lowercase hex (see :func:`norm_uuid`); "" when unreadable.
    uuid: str

    def to_json(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class DeviceMap:
    entries: tuple
    #: "torch" (exact, UUID-bridged) | "heuristic" (FASTEST_FIRST emulation)
    #: | None (no GPU inventory at all).
    source: Optional[str]

    def nvml_to_cuda(self) -> Dict[int, int]:
        return {e.nvml_index: e.cuda_index for e in self.entries}

    def cuda_to_nvml(self) -> Dict[int, int]:
        return {e.cuda_index: e.nvml_index for e in self.entries}

    def cuda_for_uuid(self, uuid) -> Optional[int]:
        u = norm_uuid(uuid) if uuid else ""
        if not u:
            return None
        for e in self.entries:
            if e.uuid == u:
                return e.cuda_index
        return None

    def cuda_for_nvml(self, nvml_index: int) -> Optional[int]:
        for e in self.entries:
            if e.nvml_index == nvml_index:
                return e.cuda_index
        return None

    def to_json(self) -> dict:
        return {
            "source": self.source,
            "entries": [e.to_json() for e in self.entries],
        }


def _nvml_cards(nvml=None) -> List[dict]:
    """[{nvml_index, name, total_mib, uuid}] from pynvml (or an injected
    module-alike; an injected ``nvml`` is NOT init/shutdown -- the caller owns
    it, mirroring ``live_metrics.read_gpu_live``). [] when nothing is
    readable."""
    own = nvml is None
    if own:
        try:
            import pynvml as nvml  # type: ignore[no-redef]

            nvml.nvmlInit()
        except Exception:
            return []
    try:
        out = []
        for i in range(nvml.nvmlDeviceGetCount()):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            name = nvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode("utf-8", "replace")
            try:
                uuid = norm_uuid(nvml.nvmlDeviceGetUUID(h))
            except Exception:
                uuid = ""
            try:
                total_mib = int(nvml.nvmlDeviceGetMemoryInfo(h).total / 2**20)
            except Exception:
                total_mib = 0
            out.append(
                {
                    "nvml_index": i,
                    "name": str(name),
                    "total_mib": total_mib,
                    "uuid": uuid,
                }
            )
        return out
    except Exception:
        return []
    finally:
        if own:
            try:
                nvml.nvmlShutdown()
            except Exception:
                pass


def _torch_cuda_uuids() -> Optional[List[str]]:
    """Normalized GPU UUIDs in torch's CUDA enumeration order, or None when
    torch/CUDA is unavailable or the order cannot be trusted.

    Untrusted when ``CUDA_VISIBLE_DEVICES`` filters THIS process: torch then
    enumerates a remapped subset, which is not the bare CUDA order the engine
    flags (launched without CVD) are interpreted in."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") not in (None, ""):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        out = []
        for i in range(torch.cuda.device_count()):
            uu = getattr(torch.cuda.get_device_properties(i), "uuid", None)
            if uu is None:
                return None
            out.append(norm_uuid(uu))
        return out
    except Exception:
        return None


def build_device_map(nvml=None, allow_torch: bool = True) -> DeviceMap:
    """Build a fresh (uncached) :class:`DeviceMap`. ``nvml`` is injectable for
    tests; an injected fake rig will not match the real torch UUIDs and thus
    deterministically exercises the heuristic path."""
    cards = _nvml_cards(nvml)
    if not cards:
        return DeviceMap(entries=(), source=None)
    cuda_per_card: Optional[List[int]] = None
    source = "heuristic"
    if allow_torch:
        tu = _torch_cuda_uuids()
        if tu is not None and len(tu) == len(cards):
            by_uuid = {u: i for i, u in enumerate(tu)}
            if all(c["uuid"] and c["uuid"] in by_uuid for c in cards):
                cuda_per_card = [by_uuid[c["uuid"]] for c in cards]
                source = "torch"
    if cuda_per_card is None:
        cuda_per_card = emulate_cuda_order([c["name"] for c in cards])
    entries = tuple(
        DeviceMapEntry(
            nvml_index=c["nvml_index"],
            cuda_index=cuda_per_card[i],
            name=c["name"],
            total_mib=c["total_mib"],
            uuid=c["uuid"],
        )
        for i, c in enumerate(cards)
    )
    return DeviceMap(entries=entries, source=source)


_CACHE: Optional[DeviceMap] = None


def device_map(refresh: bool = False) -> DeviceMap:
    """The cached live bridge for THIS host (built once; ``refresh=True``
    rebuilds, e.g. after a driver reload). Never raises."""
    global _CACHE
    if _CACHE is None or refresh:
        try:
            _CACHE = build_device_map()
        except Exception:  # pragma: no cover - defensive
            _CACHE = DeviceMap(entries=(), source=None)
    return _CACHE
