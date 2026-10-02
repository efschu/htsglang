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
    order on mixed rigs -- that is ``cuda_index``, resolved per UUID against
    the #331 identity map (None when it cannot be resolved: manual specs for
    another host, or a rig whose CUDA order this process cannot see -- never
    a guessed value, #397)."""

    index: int
    name: str
    total_mib: int
    #: Live free VRAM; only known for source="nvml" (None for manual specs).
    free_mib: Optional[int] = None
    uuid: Optional[str] = None
    pcie_gen: Optional[int] = None
    pcie_width: Optional[int] = None
    #: CUDA-order index of this card (the --rank-gpu-id/--base-gpu-id space);
    #: None when unbridged. Resolved through registry.nvml.IdentityMap.
    cuda_index: Optional[int] = None
    #: Compute capability ``(major, minor)`` (HW-GENERIC 1002): for live specs
    #: what NVML answered (``nvmlDeviceGetCudaComputeCapability`` via
    #: ``registry.nvml.DeviceInfo``, matched by UUID); for manual/json specs
    #: what the user declared. None when nothing answered -- never derived
    #: from the name here (``planner.flags.gpu_cc`` owns the exact-name
    #: catalogue fallback and names an unknown arch).
    cc: Optional[Tuple[int, int]] = None


@dataclasses.dataclass(frozen=True)
class HardwareSpec:
    gpus: Tuple[GpuDescriptor, ...]
    #: "nvml" | "nvidia-smi" | "manual" | "json"
    source: str
    host_ram_mib: Optional[int] = None
    driver: Optional[str] = None
    #: How the per-gpu ``cuda_index`` values were resolved: "identity-map"
    #: (the #331 UUID/BDF resolver, the only way they can be resolved since
    #: #397) | None (no bridge / not a live spec -- then cuda_index is None
    #: too, never a guess).
    cuda_index_source: Optional[str] = None

    def gpu(self, index: int) -> GpuDescriptor:
        """The card a ``--rank-gpu-id`` / ``--base-gpu-id`` value names.

        Those flags live in CUDA enumeration order, so the lookup resolves
        against ``cuda_index`` whenever this spec knows it (live NVML specs,
        and json specs that declare it). Matching on ``index`` there returns
        the card at the same NVML index, which on a mixed rig is a DIFFERENT
        card -- the planner's copy of #392, and the same defect the engine's
        budget guard carried: a per-card budget checked against one card and
        spent on another.

        Manual and unbridged specs carry no ``cuda_index``; there the list
        position is the only declared identity and stays the meaning of the
        value, as documented on :class:`GpuDescriptor`.
        """
        if any(g.cuda_index is not None for g in self.gpus):
            for g in self.gpus:
                if g.cuda_index == index:
                    return g
            declared = [
                f"cuda {g.cuda_index} (nvml {g.index}): {g.name} {g.total_mib} MiB"
                for g in self.gpus
            ]
            raise ValueError(
                f"--rank-gpu-id names CUDA device {index}, but this hardware "
                f"spec ({self.source}) only declares {len(self.gpus)} "
                f"device(s): {declared}."
            )
        for g in self.gpus:
            if g.index == index:
                return g
        raise ValueError(
            f"--rank-gpu-id names GPU {index}, but this hardware spec "
            f"({self.source}) only declares {len(self.gpus)} device(s): "
            f"{[f'{g.index}: {g.name} {g.total_mib} MiB' for g in self.gpus]}."
        )

    def rank_gpu_id_of(self, gpu: GpuDescriptor) -> int:
        """The ``--rank-gpu-id`` value that names ``gpu``.

        The inverse of :meth:`gpu`, and the only correct way to derive a
        placement from this spec: the engine reads those values in CUDA
        order, so a default placement written in NVML indices names other
        cards than it means on a rig where the two diverge.
        """
        return gpu.index if gpu.cuda_index is None else gpu.cuda_index


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
                cc=_cc_of(c),
            )
        )
    if not gpus:
        raise HardwareUnavailable(
            "The NVML/nvidia-smi inventory returned zero GPUs. Declare the "
            "hardware manually (--gpu 'NAME:TOTAL_MIB' per card)."
        )
    gpus = _annotate_compute_capability(gpus)
    gpus, cuda_src = _annotate_cuda_indices(gpus)
    return HardwareSpec(
        gpus=tuple(gpus),
        source=source,
        host_ram_mib=_host_ram_mib(),
        cuda_index_source=cuda_src,
    )


def _cc_of(rec) -> Optional[Tuple[int, int]]:
    """A declared compute capability from a sampler/json record: ``cc`` as
    ``[8, 6]`` / ``"8.6"`` / ``"sm86"``, or ``cc_major`` + ``cc_minor``; None
    when the record carries none (or an unreadable one)."""
    get = rec.get if isinstance(rec, dict) else (lambda k: getattr(rec, k, None))
    major, minor = get("cc_major"), get("cc_minor")
    if major is not None and minor is not None:
        try:
            return (int(major), int(minor))
        except (TypeError, ValueError):
            return None
    for key in ("cc", "compute_capability", "compute_cap"):
        cc = _parse_cc_text(get(key))
        if cc is not None:
            return cc
    return None


def _parse_cc_text(value) -> Optional[Tuple[int, int]]:
    if value is None or value == "":
        return None
    if isinstance(value, (tuple, list)):
        try:
            return (int(value[0]), int(value[1])) if len(value) == 2 else None
        except (TypeError, ValueError):
            return None
    text = str(value).strip().lower()
    if "." in text:
        major, _, minor = text.partition(".")
        if major.isdigit() and minor.isdigit() and len(minor) == 1:
            return (int(major), int(minor))
        return None
    text = text[2:].lstrip("_") if text.startswith("sm") else text
    text = text.rstrip("af")
    if text.isdigit() and len(text) >= 2:
        major, minor = divmod(int(text), 10)
        return (major, minor)
    return None


def _annotate_compute_capability(gpus):
    """Attach the NVML compute capability to every live card that does not
    carry one yet, matched by UUID against ``registry.nvml.list_devices()``
    (``nvmlDeviceGetCudaComputeCapability``). Never by NVML index or name: a
    card that cannot be matched keeps ``cc=None``. Any NVML failure leaves the
    descriptors as they are."""
    if all(g.cc is not None for g in gpus):
        return gpus
    try:
        from sglang.srt.registry import nvml as _nvml

        by_uuid = {
            str(d.uuid): d.compute_capability for d in _nvml.list_devices()
        }
    except Exception:  # noqa: BLE001 - unanswered = None, never a guess
        return gpus
    out = []
    for g in gpus:
        cc = g.cc
        if cc is None and g.uuid:
            cc = by_uuid.get(str(g.uuid))
        out.append(g if cc == g.cc else dataclasses.replace(g, cc=tuple(cc)))
    return out


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
    """Parse one ``--gpu`` CLI item: ``NAME:TOTAL_MIB[:CC]`` (e.g.
    ``"RTX 5090:32607"``, ``"RTX 3090:24576:sm86"``, ``"RTX A6000:48g:8.6"``).
    A ``g``/``G`` suffix reads as GiB (``"RTX 3080:20g"`` -> 20480 MiB). The
    optional third field declares the compute capability (HW-GENERIC 1002)."""
    cc = None
    head, sep3, last = text.rpartition(":")
    # Only a third field SHAPED like a cc ('sm..', 'X.Y') is one; anything
    # else keeps the old reading (the name may itself contain ':').
    if sep3 and ":" in head and (
        last.strip().lower().startswith("sm") or "." in last
    ):
        cc = _parse_cc_text(last)
        if cc is None:
            raise ValueError(
                f"--gpu {text!r}: cannot parse the compute capability "
                f"{last!r} (e.g. 'sm86', 'sm120', '8.6')."
            )
        text = head
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
    return GpuDescriptor(index=index, name=name.strip(), total_mib=total_mib, cc=cc)


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
                cc=_cc_of(g),
            )
        )
    return HardwareSpec(
        gpus=tuple(gpus),
        source="json",
        host_ram_mib=data.get("host_ram_mib", _host_ram_mib()),
    )
