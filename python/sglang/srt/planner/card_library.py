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
"""GPU-model catalogue + rig composition (design #97 stage S4, §2.7).

Called a CARD LIBRARY, not a profile library. Three unrelated things on this
fork were competing for the word "profile": a launchable server preset
(``planner.flags.Profile``, the one users meet), a measurement file
(``hw_profile-*.json``, ``power_profile.json``) and this catalogue of GPU
models. A control panel that uses one word for three things is unreadable, so
this one is a card, its container a library, and "profile" is left to the
preset.

The planner-core is a pure function of ``(model_config, HardwareSpec)`` (S1),
so varying the *hardware* side is free: the same ``plan()`` that answers "my
local rig -> which config" answers "an arbitrary COMPOSED rig -> what becomes
possible". This module is only a new SOURCE for ``HardwareSpec`` (§2.1) —
nothing in the engine changes.

Contents:
  * ``CardSpec``     — one catalogue card (hardware-only, non-sensitive:
    name, VRAM, sm-arch, typical PCIe/NVLink, TDP, nameplate peaks).
  * ``SEED_CARDS``   — a curated seed set (the memory rig + common
    consumer / datacenter GPUs).
  * ``CardLibrary``  — load/save JSON, add, and **populate from parsed
    RESULTS-issue fingerprints** (S2 already emits those anonymously: card
    model + count) — closing the loop submissions -> library.
  * ``compose_rig``  — assemble N cards into a
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
    "CardSpec",
    "SEED_CARDS",
    "CardLibrary",
    "compose_rig",
]


@dataclasses.dataclass(frozen=True)
class CardSpec:
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
    #: Measured-only; None unless a submission supplied a cached probe. These
    #: are the MEASURED micro-probe scores (uneven_perf's on-device GEMM /
    #: membw benchmark) — they WIN over the nameplate peaks below when present.
    gemm_tflops: Optional[float] = None
    membw_gbs: Optional[float] = None
    #: NAMEPLATE peak specs (public datasheet numbers), used ONLY by the
    #: roofline throughput ESTIMATE (roofline.py) as the fallback when no
    #: measured probe exists. Deliberately distinct from the measured
    #: ``gemm_tflops`` / ``membw_gbs`` above — a peak is a theoretical ceiling
    #: the hardware never actually reaches, so the roofline that consumes it is
    #: labelled a rough ballpark, never a measured number.
    #:   * ``peak_membw_gbs``          — HBM/GDDR peak bandwidth (GB/s).
    #:   * ``peak_gemm_tflops_fp16``   — dense fp16/bf16 tensor-core peak
    #:     (no 2:4 sparsity), fp32/fp16 accumulate.
    #:   * ``peak_gemm_tflops_fp8``    — dense fp8 (E4M3) tensor-core peak;
    #:     None on pre-Ada archs that have NO fp8 tensor cores (Ampere: an fp8
    #:     model up-casts to the fp16 tensor path, so the roofline falls back
    #:     to ``peak_gemm_tflops_fp16`` there).
    peak_membw_gbs: Optional[float] = None
    peak_gemm_tflops_fp16: Optional[float] = None
    peak_gemm_tflops_fp8: Optional[float] = None

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
#: Peak specs below are NAMEPLATE datasheet numbers (dense tensor peak, no 2:4
#: sparsity; peak GDDR/HBM bandwidth) — theoretical ceilings, ~ballpark, used
#: only by the roofline estimate. fp8 peak is None on Ampere (sm8x: no fp8
#: tensor cores).
SEED_CARDS: Dict[str, CardSpec] = {
    p.name: p
    for p in [
        # -- this system's rig (MEMORY) ------------------------------------
        CardSpec("RTX 5090", 32607, "sm120", 5, 16, False, 575,
                   peak_membw_gbs=1792.0, peak_gemm_tflops_fp16=419.0,
                   peak_gemm_tflops_fp8=838.0),
        CardSpec("RTX 3080 20GB", 20480, "sm86", 4, 16, False, 320,
                   peak_membw_gbs=760.0, peak_gemm_tflops_fp16=119.0),
        # -- common consumer -----------------------------------------------
        CardSpec("RTX 5080", 16303, "sm120", 5, 16, False, 360,
                   peak_membw_gbs=960.0, peak_gemm_tflops_fp16=225.0,
                   peak_gemm_tflops_fp8=450.0),
        CardSpec("RTX 4090", 24564, "sm89", 4, 16, False, 450,
                   peak_membw_gbs=1008.0, peak_gemm_tflops_fp16=330.0,
                   peak_gemm_tflops_fp8=660.0),
        CardSpec("RTX 4080", 16376, "sm89", 4, 16, False, 320,
                   peak_membw_gbs=717.0, peak_gemm_tflops_fp16=195.0,
                   peak_gemm_tflops_fp8=390.0),
        CardSpec("RTX 3090", 24576, "sm86", 4, 16, True, 350,
                   peak_membw_gbs=936.0, peak_gemm_tflops_fp16=142.0),
        CardSpec("RTX 3090 Ti", 24564, "sm86", 4, 16, True, 450,
                   peak_membw_gbs=1008.0, peak_gemm_tflops_fp16=160.0),
        CardSpec("RTX 3080", 10240, "sm86", 4, 16, False, 320,
                   peak_membw_gbs=760.0, peak_gemm_tflops_fp16=119.0),
        CardSpec("RTX 3060", 12288, "sm86", 4, 16, False, 170,
                   peak_membw_gbs=360.0, peak_gemm_tflops_fp16=51.0),
        # -- workstation / datacenter --------------------------------------
        CardSpec("RTX A6000", 49140, "sm86", 4, 16, True, 300,
                   peak_membw_gbs=768.0, peak_gemm_tflops_fp16=155.0),
        CardSpec("L40S", 46068, "sm89", 4, 16, False, 350,
                   peak_membw_gbs=864.0, peak_gemm_tflops_fp16=362.0,
                   peak_gemm_tflops_fp8=733.0),
        CardSpec("A100 40GB", 40960, "sm80", 4, 16, True, 400,
                   peak_membw_gbs=1555.0, peak_gemm_tflops_fp16=312.0),
        CardSpec("A100 80GB", 81920, "sm80", 4, 16, True, 400,
                   peak_membw_gbs=2039.0, peak_gemm_tflops_fp16=312.0),
        CardSpec("H100 80GB", 81559, "sm90", 5, 16, True, 700,
                   peak_membw_gbs=3350.0, peak_gemm_tflops_fp16=989.0,
                   peak_gemm_tflops_fp8=1979.0),
        CardSpec("H200", 143771, "sm90", 5, 16, True, 700,
                   peak_membw_gbs=4800.0, peak_gemm_tflops_fp16=989.0,
                   peak_gemm_tflops_fp8=1979.0),
        CardSpec("MI300X", 196608, "cdna3", 5, 16, True, 750,
                   peak_membw_gbs=5300.0, peak_gemm_tflops_fp16=1307.0,
                   peak_gemm_tflops_fp8=2614.0),
    ]
}


#: Vendor/brand words that a driver-reported name carries and a catalogue key
#: does not. NVML reports consumer NVIDIA cards as "NVIDIA GeForce RTX 5090"
#: while the seed key is "RTX 5090", so stripping only "nvidia" left every
#: GeForce card unmatchable -- and `--pp-solve-cut` looks the card up by the
#: name the CENSUS recorded, which is the driver's. Model tokens are never
#: listed here: "RTX 3080" and "RTX 3080 20GB" must stay distinct.
_VENDOR_WORDS = ("nvidia", "geforce")


def _canonical(name: str) -> str:
    """Loose key for name lookup (case/space-insensitive)."""
    text = str(name).lower()
    for word in _VENDOR_WORDS:
        text = text.replace(word, "")
    return " ".join(text.split())


class CardLibrary:
    """A catalog of ``CardSpec``s keyed by card name, seeded from
    ``SEED_CARDS`` and grown from submitted RESULTS fingerprints (§2.7).
    JSON persistence, no server."""

    def __init__(self, profiles: Optional[Dict[str, CardSpec]] = None):
        self._by_key: Dict[str, CardSpec] = {}
        for p in (profiles or SEED_CARDS).values():
            self._by_key[_canonical(p.name)] = p

    # -- access -------------------------------------------------------------

    def names(self) -> List[str]:
        return sorted(p.name for p in self._by_key.values())

    def get(self, name: str) -> CardSpec:
        key = _canonical(name)
        if key not in self._by_key:
            raise KeyError(
                f"unknown GPU profile {name!r}. Known: {', '.join(self.names())}"
            )
        return self._by_key[key]

    def has(self, name: str) -> bool:
        return _canonical(name) in self._by_key

    def add(self, profile: CardSpec, overwrite: bool = False) -> bool:
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
            prof = CardSpec(name=str(name), total_mib=int(total_mib))
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
                    CardSpec(name=str(name), total_mib=int(total_mib))
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
    def load(cls, path: str, seed: bool = True) -> "CardLibrary":
        """Load a saved library. ``seed=True`` starts from ``SEED_CARDS``
        and overlays the file (so a saved library never loses the seed set)."""
        lib = cls() if seed else cls(profiles={})
        with open(path) as f:
            data = json.load(f)
        for pd in data.get("profiles", []):
            lib.add(CardSpec(**pd), overwrite=True)
        return lib


def compose_rig(
    profiles: Sequence,
    library: Optional[CardLibrary] = None,
    host_ram_mib: Optional[int] = None,
) -> HardwareSpec:
    """Compose ``profiles`` (names resolved via ``library``, or
    ``CardSpec`` objects) into a ``HardwareSpec(source="library-composition")``.

    Card ``i`` in the list is physical index ``i``. ``free_mib`` is None on
    every card (no live measurement) and the source marks the whole spec an
    ESTIMATE — the explorer surfaces that as the required "assumes topology,
    not measured" label (design §8). Duplicates are allowed (a homogeneous or
    heterogeneous pile: e.g. ["RTX 5090", "RTX 3080 20GB", "RTX 3080 20GB"]).
    """
    library = library or CardLibrary()
    gpus = []
    for i, item in enumerate(profiles):
        prof = item if isinstance(item, CardSpec) else library.get(item)
        gpus.append(prof.to_descriptor(i))
    if not gpus:
        raise ValueError("compose_rig needs at least one profile.")
    return HardwareSpec(
        gpus=tuple(gpus),
        source="library-composition",
        host_ram_mib=host_ram_mib,
    )
