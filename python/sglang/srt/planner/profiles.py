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
"""Hardware-profile library + rig composition (design #97 stage S4, §2.7).

The planner-core is a pure function of ``(model_config, HardwareSpec)`` (S1),
so varying the *hardware* side is free: the same ``plan()`` that answers "my
local rig -> which config" answers "an arbitrary COMPOSED rig -> what becomes
possible". This module is only a new SOURCE for ``HardwareSpec`` (§2.1) —
nothing in the engine changes.

Contents:
  * ``GpuProfile``      — one catalog card (hardware-only, non-sensitive:
    name, VRAM, sm-arch, typical PCIe/NVLink, TDP).
  * ``SEED_PROFILES``   — a curated seed set (the memory rig + common
    consumer / datacenter GPUs).
  * ``ProfileLibrary``  — load/save JSON, add, and **populate from parsed
    RESULTS-issue fingerprints** (S2 already emits those anonymously: card
    model + count) — closing the loop submissions -> library.
  * ``compose_rig``     — assemble N profiles into a
    ``HardwareSpec(source="library-composition")``.

HONESTY (design §8): a composed rig has NO measured free-VRAM and NO measured
interconnect — it is an *estimate*. That is carried structurally by
``HardwareSpec.source == "library-composition"`` (and ``free_mib is None`` on
every card); the explorer (``explorer.py``) turns that into a visible
"estimate — assumes pcie/nvlink topology, not measured" label, distinct from
a real (live-NVML / booted) rig. Capacity/feasibility is pure VRAM math
(interconnect-independent), so the composed number is honest as *capacity* —
never as throughput.
"""

from __future__ import annotations

import dataclasses
import json
import os
from typing import Dict, List, Optional, Sequence

from sglang.srt.planner.hardware import GpuDescriptor, HardwareSpec

__all__ = [
    "GpuProfile",
    "SEED_PROFILES",
    "ProfileLibrary",
    "compose_rig",
]


@dataclasses.dataclass(frozen=True)
class GpuProfile:
    """A catalog card — hardware-only, non-identifying (design §2.7). Optional
    measured perf fields (``gemm_tflops`` / ``membw_gbs``) fill in ONLY from a
    submission that carried a cached probe; absent, capacity/feasibility still
    work (they need only the VRAM total)."""

    name: str
    total_mib: int
    sm_arch: Optional[str] = None
    pcie_gen: Optional[int] = None
    pcie_width: Optional[int] = None
    nvlink: bool = False
    tdp_w: Optional[int] = None
    #: Measured-only; None unless a submission supplied a cached probe.
    gemm_tflops: Optional[float] = None
    membw_gbs: Optional[float] = None

    def to_descriptor(self, index: int) -> GpuDescriptor:
        """Map to the S1 planner card. ``free_mib`` stays None — a composed /
        library card has no live free-VRAM measurement (design §8)."""
        return GpuDescriptor(
            index=index,
            name=self.name,
            total_mib=self.total_mib,
            free_mib=None,
            pcie_gen=self.pcie_gen,
            pcie_width=self.pcie_width,
        )


#: Curated seed set. VRAM totals are the NVML-nominal MiB per card. The two
#: entries for the local rig (RTX 5090 32GB + the 20GB RTX 3080 variant) come
#: from this system's inventory (MEMORY: hardware rig). Common consumer +
#: datacenter cards round out the library so the explorer can compose rigs the
#: user does not physically own.
SEED_PROFILES: Dict[str, GpuProfile] = {
    p.name: p
    for p in [
        # -- this system's rig (MEMORY) ------------------------------------
        GpuProfile("RTX 5090", 32607, "sm120", 5, 16, False, 575),
        GpuProfile("RTX 3080 20GB", 20480, "sm86", 4, 16, False, 320),
        # -- common consumer -----------------------------------------------
        GpuProfile("RTX 5080", 16303, "sm120", 5, 16, False, 360),
        GpuProfile("RTX 4090", 24564, "sm89", 4, 16, False, 450),
        GpuProfile("RTX 4080", 16376, "sm89", 4, 16, False, 320),
        GpuProfile("RTX 3090", 24576, "sm86", 4, 16, True, 350),
        GpuProfile("RTX 3090 Ti", 24564, "sm86", 4, 16, True, 450),
        GpuProfile("RTX 3080", 10240, "sm86", 4, 16, False, 320),
        GpuProfile("RTX 3060", 12288, "sm86", 4, 16, False, 170),
        # -- workstation / datacenter --------------------------------------
        GpuProfile("RTX A6000", 49140, "sm86", 4, 16, True, 300),
        GpuProfile("L40S", 46068, "sm89", 4, 16, False, 350),
        GpuProfile("A100 40GB", 40960, "sm80", 4, 16, True, 400),
        GpuProfile("A100 80GB", 81920, "sm80", 4, 16, True, 400),
        GpuProfile("H100 80GB", 81559, "sm90", 5, 16, True, 700),
        GpuProfile("H200", 143771, "sm90", 5, 16, True, 700),
        GpuProfile("MI300X", 196608, "cdna3", 5, 16, True, 750),
    ]
}


def _canonical(name: str) -> str:
    """Loose key for name lookup (case/space-insensitive)."""
    return " ".join(str(name).lower().replace("nvidia", "").split())


class ProfileLibrary:
    """A catalog of ``GpuProfile``s keyed by card name, seeded from
    ``SEED_PROFILES`` and grown from submitted RESULTS fingerprints (§2.7).
    JSON persistence, no server."""

    def __init__(self, profiles: Optional[Dict[str, GpuProfile]] = None):
        self._by_key: Dict[str, GpuProfile] = {}
        for p in (profiles or SEED_PROFILES).values():
            self._by_key[_canonical(p.name)] = p

    # -- access -------------------------------------------------------------

    def names(self) -> List[str]:
        return sorted(p.name for p in self._by_key.values())

    def get(self, name: str) -> GpuProfile:
        key = _canonical(name)
        if key not in self._by_key:
            raise KeyError(
                f"unknown GPU profile {name!r}. Known: {', '.join(self.names())}"
            )
        return self._by_key[key]

    def has(self, name: str) -> bool:
        return _canonical(name) in self._by_key

    def add(self, profile: GpuProfile, overwrite: bool = False) -> bool:
        """Add/refresh a profile. Returns True when the library changed. An
        existing entry is kept unless ``overwrite`` (a submission never
        silently downgrades a curated entry's perf fields to None)."""
        key = _canonical(profile.name)
        if key in self._by_key and not overwrite:
            # Fill only missing perf fields from the newcomer (measured wins
            # over absent, never the reverse).
            cur = self._by_key[key]
            merged = dataclasses.replace(
                cur,
                gemm_tflops=cur.gemm_tflops or profile.gemm_tflops,
                membw_gbs=cur.membw_gbs or profile.membw_gbs,
            )
            if merged != cur:
                self._by_key[key] = merged
                return True
            return False
        self._by_key[key] = profile
        return True

    # -- populate from S2 RESULTS fingerprints ------------------------------

    def populate_from_fingerprint(self, fingerprint) -> int:
        """Add each card in an S2 ``HardwareFingerprint`` (``.cards`` =
        ``[(count, name, total_mib), ...]``, already anonymous — no UUIDs).
        Returns the number of library entries added/changed. A card with an
        unknown VRAM total (0, e.g. from a boot-log-only fingerprint) is
        skipped rather than poisoning the library with a bogus total."""
        changed = 0
        for count, name, total_mib in getattr(fingerprint, "cards", []):
            if not total_mib or total_mib <= 0:
                continue
            prof = GpuProfile(name=str(name), total_mib=int(total_mib))
            if self.add(prof):
                changed += 1
        return changed

    def populate_from_results_dicts(self, entries: Sequence[dict]) -> int:
        """Populate from a list of parsed RESULTS entries of the shape
        ``{"cards": [[count, name, total_mib], ...]}`` (e.g. a curated
        ``results.json``)."""
        changed = 0
        for e in entries:
            for item in e.get("cards", []):
                count, name, total_mib = item
                if total_mib and total_mib > 0 and self.add(
                    GpuProfile(name=str(name), total_mib=int(total_mib))
                ):
                    changed += 1
        return changed

    # -- persistence --------------------------------------------------------

    def save(self, path: str) -> None:
        data = {
            "profiles": [dataclasses.asdict(p) for p in self._by_key.values()]
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=1)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, seed: bool = True) -> "ProfileLibrary":
        """Load a saved library. ``seed=True`` starts from ``SEED_PROFILES``
        and overlays the file (so a saved library never loses the seed set)."""
        lib = cls() if seed else cls(profiles={})
        with open(path) as f:
            data = json.load(f)
        for pd in data.get("profiles", []):
            lib.add(GpuProfile(**pd), overwrite=True)
        return lib


def compose_rig(
    profiles: Sequence,
    library: Optional[ProfileLibrary] = None,
    host_ram_mib: Optional[int] = None,
) -> HardwareSpec:
    """Compose ``profiles`` (names resolved via ``library``, or
    ``GpuProfile`` objects) into a ``HardwareSpec(source="library-composition")``.

    Card ``i`` in the list is physical index ``i``. ``free_mib`` is None on
    every card (no live measurement) and the source marks the whole spec an
    ESTIMATE — the explorer surfaces that as the required "assumes topology,
    not measured" label (design §8). Duplicates are allowed (a homogeneous or
    heterogeneous pile: e.g. ["RTX 5090", "RTX 3080 20GB", "RTX 3080 20GB"]).
    """
    library = library or ProfileLibrary()
    gpus = []
    for i, item in enumerate(profiles):
        prof = item if isinstance(item, GpuProfile) else library.get(item)
        gpus.append(prof.to_descriptor(i))
    if not gpus:
        raise ValueError("compose_rig needs at least one profile.")
    return HardwareSpec(
        gpus=tuple(gpus),
        source="library-composition",
        host_ram_mib=host_ram_mib,
    )
