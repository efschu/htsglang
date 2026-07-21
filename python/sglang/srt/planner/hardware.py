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
"""HardwareSpec — the planner's abstract hardware description (design §2.1).

The planner never talks to NVML directly; it consumes a ``HardwareSpec``
built from one of three sources, in graceful-degradation order:

  1. ``nvml``    — live inventory via the rig-dashboard sampler
                   (``tools/rig_dashboard/server.py:sample_nvml``: pynvml
                   first, nvidia-smi CSV parse as fallback), reused when the
                   repo checkout is present, with a minimal built-in
                   nvidia-smi fallback otherwise;
  2. ``manual``  — a hand-declared inventory ("RTX 5090:32607", ...): the
                   whole point of an OFFLINE planner — it must work on a
                   machine with no GPU at all;
  3. a JSON file with the same fields (for saved / composed rigs).

``free_mib`` exists only for the ``nvml`` source; manual specs plan against
``total_mib`` minus the user's per-card free-reserve (design §2.5).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import os
import subprocess
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "GpuDescriptor",
    "HardwareSpec",
    "HardwareUnavailable",
    "hardware_from_nvml",
    "hardware_from_manual",
    "hardware_from_json",
    "parse_manual_gpu",
]


class HardwareUnavailable(RuntimeError):
    """No live GPU inventory could be read (and no manual spec was given)."""


@dataclasses.dataclass(frozen=True)
class GpuDescriptor:
    """One physical card.

    INDEX SPACES (the device-order trap): ``index`` is the NVML/PCI-bus
    enumeration index for source="nvml" specs (list position for manual/json
    specs). The ENGINE flags (``--rank-gpu-id`` / ``--base-gpu-id``) live in
    CUDA enumeration order (FASTEST_FIRST) instead, which diverges from NVML
    order on mixed rigs -- that is ``cuda_index``, bridged per UUID/heuristic
    by ``planner.device_map`` (None when no bridge is available, e.g. manual
    specs for another host)."""

    index: int
    name: str
    total_mib: int
    #: Live free VRAM; only known for source="nvml" (None for manual specs).
    free_mib: Optional[int] = None
    uuid: Optional[str] = None
    pcie_gen: Optional[int] = None
    pcie_width: Optional[int] = None
    #: CUDA-order index of this card (the --rank-gpu-id/--base-gpu-id space);
    #: None when unbridged. See planner.device_map.
    cuda_index: Optional[int] = None


@dataclasses.dataclass(frozen=True)
class HardwareSpec:
    gpus: Tuple[GpuDescriptor, ...]
    #: "nvml" | "nvidia-smi" | "manual" | "json"
    source: str
    host_ram_mib: Optional[int] = None
    driver: Optional[str] = None
    #: How the per-gpu ``cuda_index`` values were resolved: "torch" (exact,
    #: UUID-bridged) | "heuristic" (FASTEST_FIRST emulation -- surface this
    #: to the user) | None (no bridge / not a live spec).
    cuda_index_source: Optional[str] = None

    def gpu(self, index: int) -> GpuDescriptor:
        for g in self.gpus:
            if g.index == index:
                return g
        raise ValueError(
            f"--rank-gpu-id names GPU {index}, but this hardware spec "
            f"({self.source}) only declares {len(self.gpus)} device(s): "
            f"{[f'{g.index}: {g.name} {g.total_mib} MiB' for g in self.gpus]}."
        )


# ---------------------------------------------------------------------------
# Source 1: live NVML, reusing the rig-dashboard sampler when available.
# ---------------------------------------------------------------------------


def _load_rig_dashboard_sampler():
    """Import ``sample_nvml`` from tools/rig_dashboard/server.py (the repo
    checkout layout: <repo>/python/sglang/srt/planner/hardware.py ->
    <repo>/tools/rig_dashboard/server.py). Returns None when the file is not
    present (e.g. an installed wheel)."""
    here = os.path.abspath(__file__)
    repo = os.path.dirname(  # repo root
        os.path.dirname(  # python/
            os.path.dirname(  # sglang/
                os.path.dirname(os.path.dirname(here))  # srt/planner
            )
        )
    )
    path = os.path.join(repo, "tools", "rig_dashboard", "server.py")
    if not os.path.isfile(path):
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            "_sglang_rig_dashboard_server", path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.sample_nvml
    except Exception:
        return None


def _smi_inventory_fallback() -> List[dict]:
    """Minimal built-in nvidia-smi parse (mirror of the rig-dashboard
    ``_nvml_via_smi`` fields the planner needs), for installed-wheel
    environments where tools/ is absent."""
    q = "index,name,uuid,memory.total,memory.used,pcie.link.gen.current,pcie.link.width.current"
    txt = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
        text=True,
    )
    out = []
    for line in txt.strip().splitlines():
        c = [x.strip() for x in line.split(",")]

        def _i(v):
            try:
                return int(float(v))
            except Exception:
                return None

        out.append(
            {
                "index": _i(c[0]),
                "name": c[1],
                "uuid": c[2],
                "mem_total_mib": _i(c[3]),
                "mem_used_mib": _i(c[4]),
                "pcie_gen": _i(c[5]),
                "pcie_width": _i(c[6]),
            }
        )
    return out


def _host_ram_mib() -> Optional[int]:
    try:
        return int(
            os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2**20
        )
    except (ValueError, OSError, AttributeError):
        return None


def hardware_from_nvml() -> HardwareSpec:
    """Live inventory: rig-dashboard ``sample_nvml`` (pynvml -> nvidia-smi)
    when the checkout is present, else the built-in nvidia-smi fallback.
    Raises :class:`HardwareUnavailable` when nothing can be read — callers
    should then fall back to a manual spec."""
    sample = _load_rig_dashboard_sampler()
    cards, source = [], None
    if sample is not None:
        cards, source = sample()
        if not cards:
            source = None
    if source is None:
        try:
            cards, source = _smi_inventory_fallback(), "nvidia-smi"
        except Exception as e:
            raise HardwareUnavailable(
                "No live GPU inventory: pynvml/nvidia-smi both unavailable "
                f"({e}). Declare the hardware manually (--gpu 'NAME:TOTAL_MIB' "
                "per card, or --hardware-json)."
            ) from e
    gpus = []
    for c in cards:
        total = c.get("mem_total_mib")
        used = c.get("mem_used_mib")
        gpus.append(
            GpuDescriptor(
                index=int(c["index"]),
                name=str(c.get("name") or "unknown"),
                total_mib=int(total),
                free_mib=(int(total - used) if used is not None else None),
                uuid=c.get("uuid"),
                pcie_gen=c.get("pcie_gen"),
                pcie_width=c.get("pcie_width"),
            )
        )
    if not gpus:
        raise HardwareUnavailable(
            "The NVML/nvidia-smi inventory returned zero GPUs. Declare the "
            "hardware manually (--gpu 'NAME:TOTAL_MIB' per card)."
        )
    gpus, cuda_src = _annotate_cuda_indices(gpus)
    return HardwareSpec(
        gpus=tuple(gpus),
        source=source,
        host_ram_mib=_host_ram_mib(),
        cuda_index_source=cuda_src,
    )


def _annotate_cuda_indices(gpus):
    """Attach each live card's CUDA-order index (the --rank-gpu-id /
    --base-gpu-id space) via the planner's device_map bridge: by UUID when
    the card carries one, else by NVML index. Returns ``(gpus, source)``;
    on any failure the descriptors pass through unbridged (never crashes)."""
    try:
        from sglang.srt.planner.device_map import device_map, norm_uuid

        dm = device_map()
    except Exception:  # pragma: no cover - defensive
        return gpus, None
    if not dm.entries:
        return gpus, None
    n2c = dm.nvml_to_cuda()
    out = []
    for g in gpus:
        cu = dm.cuda_for_uuid(norm_uuid(g.uuid)) if g.uuid else None
        if cu is None:
            cu = n2c.get(g.index)
        out.append(dataclasses.replace(g, cuda_index=cu))
    return out, dm.source


# ---------------------------------------------------------------------------
# Sources 2/3: manual / JSON — the offline path (no GPU required).
# ---------------------------------------------------------------------------


def parse_manual_gpu(text: str, index: int) -> GpuDescriptor:
    """Parse one ``--gpu`` CLI item: ``NAME:TOTAL_MIB`` (e.g.
    ``"RTX 5090:32607"``). A ``g``/``G`` suffix reads as GiB
    (``"RTX 3080:20g"`` -> 20480 MiB)."""
    name, sep, mem = text.rpartition(":")
    if not sep or not name.strip():
        raise ValueError(
            f"--gpu {text!r}: expected 'NAME:TOTAL_MIB' "
            "(e.g. 'RTX 5090:32607' or 'RTX 3080:20g')."
        )
    mem = mem.strip().lower()
    try:
        if mem.endswith("g"):
            total_mib = int(float(mem[:-1]) * 1024)
        else:
            total_mib = int(mem)
    except ValueError as e:
        raise ValueError(
            f"--gpu {text!r}: cannot parse the VRAM size {mem!r} "
            "(MiB integer, or GiB with a 'g' suffix)."
        ) from e
    if total_mib <= 0:
        raise ValueError(f"--gpu {text!r}: VRAM must be positive.")
    return GpuDescriptor(index=index, name=name.strip(), total_mib=total_mib)


def hardware_from_manual(items: Sequence[str]) -> HardwareSpec:
    """A hand-declared inventory; card i in the list is physical index i."""
    gpus = tuple(parse_manual_gpu(t, i) for i, t in enumerate(items))
    return HardwareSpec(gpus=gpus, source="manual", host_ram_mib=_host_ram_mib())


def hardware_from_json(path: str) -> HardwareSpec:
    """Load a saved/composed rig: ``{"gpus": [{"name", "total_mib",
    ["index"], ["free_mib"], ...}], ["host_ram_mib"]}``."""
    with open(path) as f:
        data = json.load(f)
    gpus = []
    for i, g in enumerate(data["gpus"]):
        gpus.append(
            GpuDescriptor(
                index=int(g.get("index", i)),
                name=str(g.get("name", f"gpu{i}")),
                total_mib=int(g["total_mib"]),
                free_mib=(
                    int(g["free_mib"]) if g.get("free_mib") is not None else None
                ),
                uuid=g.get("uuid"),
                pcie_gen=g.get("pcie_gen"),
                pcie_width=g.get("pcie_width"),
                cuda_index=(
                    int(g["cuda_index"])
                    if g.get("cuda_index") is not None
                    else None
                ),
            )
        )
    return HardwareSpec(
        gpus=tuple(gpus),
        source="json",
        host_ram_mib=data.get("host_ram_mib", _host_ram_mib()),
    )
