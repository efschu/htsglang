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
"""Phase footprints: the prefill activation peak and the graph-capture cost.

WHY THIS MODULE EXISTS. Both quantities were inherited as formulas, and the
2026-08-05 reference window falsified both -- in OPPOSITE directions:

    activation   booked 3968 MiB   (512 + tokens*1.5 + tp*pp/8*1024)
                 measured <= 1766 MiB on the tightest card
                 -> OVER-charged, costs KV pool

    graph capture booked  192 MiB   (96 captured tokens x 2 MiB/token)
                 measured 633-730 MiB per rank
                 -> UNDER-charged ~3.3-3.8x, the dangerous direction

The activation falsification is not a matter of degree. The 3080 at CUDA
ordinal 1 finished a 70 018-token prefill with 1766 MiB free on the card. Had
the activation peak really needed the 3968 MiB the heuristic books, that boot
would have died. It did not, so the heuristic is wrong by construction and no
amount of averaging rescues it.

The capture figures come from the boot's own log: every rank prints
``avail mem`` at "Capture ... begin", i.e. free memory after the weights, the
KV pool and the state pools are placed and before the graphs are built. The
difference between that and the post-boot steady free is what capture actually
took.

WHAT THIS MODULE WILL NOT DO. It will not fall back to either formula. A
falsified formula that still runs is worse than a refusal, because it produces
a number that looks like an answer -- which is exactly how 3968 survived long
enough to become the denominator of a solver. When nothing is calibrated for
the live profile, :func:`resolve_phase_footprint` returns ``None`` and the
ledger turns that into an UNBOUNDED item, which refuses the boot and names the
probe that would fix it.

PROVENANCE LADDER, in strict precedence order:

1. ``MEASURED_PEAK`` -- ``torch.cuda.memory_stats()`` peak counters captured by
   an instrumented boot. The honest number, and the only one that is a point
   estimate rather than a bound.
2. ``UPPER_BOUND`` -- derived from card-level free memory that a real workload
   survived. Conservative and safe (an over-estimate cannot OOM a boot, it can
   only cost KV), and strictly better than the heuristic it replaces because it
   is a bound the hardware actually demonstrated. Carries the probe command
   that would replace it.
3. nothing -- refusal.

The cache follows the #213/#513 probe canon: same directory, same
sha1-of-canonical-inputs key, same "a version bump invalidates rather than
reinterprets" rule. The key is the HARDWARE fingerprint plus the ACTIVATION
PROFILE, because these quantities depend on both: the same card with a
different chunked prefill size is a different measurement, and the same config
on a different card is too.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
import json
import logging
import os
from typing import Any, Dict, Mapping, Optional, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "PHASE_FOOTPRINT_VERSION",
    "FootprintProvenance",
    "PhaseFootprint",
    "ActivationProfile",
    "profile_key",
    "footprint_cache_path",
    "load_footprints",
    "save_footprints",
    "resolve_phase_footprint",
    "REFERENCE_WINDOW_FINGERPRINT",
]

PHASE_FOOTPRINT_VERSION = 1

from sglang.srt.rigmon.card_probe import CACHE_DIR  # noqa: E402

#: The rig the 2026-08-05 reference window measured. Recorded so the shipped
#: upper bounds below cannot leak onto another rig: a different card set,
#: driver or wheel produces a different fingerprint and therefore a miss.
REFERENCE_WINDOW_FINGERPRINT = "a191a0712717"


class FootprintProvenance(str, enum.Enum):
    MEASURED_PEAK = "measured_peak"
    UPPER_BOUND = "upper_bound"


@dataclasses.dataclass(frozen=True)
class ActivationProfile:
    """What a phase footprint depends on, besides the card.

    Deliberately explicit rather than "the whole ServerArgs": a key that
    includes everything is a key that never hits, and a key that includes too
    little silently reuses one config's measurement for another. These are the
    fields that move the prefill activation peak and the captured-graph set.
    """

    architectures: tuple
    chunked_prefill_size: int
    tp_size: int
    pp_size: int
    kv_cache_dtype: str = ""
    speculative_num_draft_tokens: int = 0
    decode_max_bs: int = 0

    def canonical(self) -> list:
        return [
            sorted(str(a) for a in self.architectures),
            int(self.chunked_prefill_size),
            int(self.tp_size),
            int(self.pp_size),
            str(self.kv_cache_dtype or ""),
            int(self.speculative_num_draft_tokens or 0),
            int(self.decode_max_bs or 0),
        ]


def profile_key(profile: ActivationProfile) -> str:
    """Stable digest of the profile. Same discipline as the card probe key."""
    blob = json.dumps(profile.canonical(), sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


@dataclasses.dataclass(frozen=True)
class PhaseFootprint:
    """One rank's prefill activation peak and graph-capture cost, in MiB."""

    activation_mib: int
    capture_mib: int
    provenance: FootprintProvenance
    #: Free-text statement of where the numbers came from. Printed in the
    #: ledger row, so a reader can tell a measured point estimate from a bound
    #: without opening this file.
    source: str
    card_uuid: str = ""
    profile_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "provenance", FootprintProvenance(self.provenance))
        if self.activation_mib < 0 or self.capture_mib < 0:
            raise ValueError("a phase footprint cannot be negative")
        if not str(self.source).strip():
            raise ValueError(
                "a phase footprint must state its source; an unattributed "
                "number is the defect this module replaces"
            )

    @property
    def is_bound(self) -> bool:
        return self.provenance is FootprintProvenance.UPPER_BOUND

    def to_json(self) -> dict:
        d = dataclasses.asdict(self)
        d["provenance"] = self.provenance.value
        return d

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "PhaseFootprint":
        return cls(
            activation_mib=int(d["activation_mib"]),
            capture_mib=int(d["capture_mib"]),
            provenance=FootprintProvenance(d["provenance"]),
            source=str(d["source"]),
            card_uuid=str(d.get("card_uuid", "")),
            profile_digest=str(d.get("profile_digest", "")),
        )


# ---------------------------------------------------------------------------
# The 2026-08-05 reference window, recorded as BOUNDS
# ---------------------------------------------------------------------------
#
# Read off the reference boot (/spinning/boot-ref-584.log) plus 10 Hz
# card sampling during a 70 018-token prefill:
#
#   activation upper bound = free memory the card still had while surviving
#   that prefill. It is a BOUND, not a measurement of the peak: nvidia-smi
#   reports the caching allocator's RESERVATION, so a transient that fits
#   inside an existing segment is invisible. What it does prove is that the
#   peak did not exceed this, which is enough to falsify 3968 MiB and not
#   enough to replace it with a point estimate.
#
#   capture = avail_mem at "Capture ... begin" minus post-boot steady free.
#   Both endpoints are real allocations, so this one is close to a measurement;
#   it is still filed as a bound because the two endpoints come from different
#   instruments (torch free vs nvidia-smi used).
#
# The probe in scripts/vram_ledger/probe_activation.py replaces both with
# torch.cuda.memory_stats() peak counters.
_REFERENCE_WINDOW_BOUNDS: Dict[str, Dict[str, int]] = {
    # RTX 5090, CUDA ordinal 0, rank TP0
    "GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d": {
        "activation_mib": 4195,
        "capture_mib": 730,
    },
    # RTX 3080, CUDA ordinal 1, rank TP1 -- the binding card
    "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7": {
        "activation_mib": 1766,
        "capture_mib": 640,
    },
    # RTX 3080, CUDA ordinal 2, rank TP2
    "GPU-62dbbae1-e859-9ccc-f9c2-d9f2443a84f4": {
        "activation_mib": 2644,
        "capture_mib": 633,
    },
}

_REFERENCE_WINDOW_PROFILE = ActivationProfile(
    architectures=("Qwen3_5ForConditionalGeneration",),
    chunked_prefill_size=2048,
    tp_size=3,
    pp_size=1,
    kv_cache_dtype="fp8_e4m3",
    speculative_num_draft_tokens=4,
    decode_max_bs=24,
)

_REFERENCE_SOURCE = (
    "2026-08-05 reference window: activation is the free VRAM the card retained "
    "while completing a 70018-token prefill (an upper bound -- nvidia-smi shows "
    "allocator reservation, so a transient inside an existing segment is "
    "invisible); capture is avail_mem at 'Capture ... begin' minus post-boot "
    "steady free. Replaces the falsified 512+tokens*1.5+tp*pp/8*1024 heuristic "
    "(3968 MiB), which exceeded the 1766 MiB the binding card had free and "
    "would therefore have OOMed that boot. Run "
    "scripts/vram_ledger/probe_activation.py for measured peaks."
)


def reference_window_footprints() -> Dict[str, PhaseFootprint]:
    """The shipped bounds, keyed by card UUID. Empty of meaning on any other
    rig -- the caller gates them on the hardware fingerprint."""
    digest = profile_key(_REFERENCE_WINDOW_PROFILE)
    return {
        uuid: PhaseFootprint(
            activation_mib=v["activation_mib"],
            capture_mib=v["capture_mib"],
            provenance=FootprintProvenance.UPPER_BOUND,
            source=_REFERENCE_SOURCE,
            card_uuid=uuid,
            profile_digest=digest,
        )
        for uuid, v in _REFERENCE_WINDOW_BOUNDS.items()
    }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def footprint_cache_path(
    hw_fingerprint: str, profile_digest: str, cache_dir: Optional[str] = None
) -> str:
    return os.path.join(
        cache_dir or CACHE_DIR,
        f"phase_footprint-{hw_fingerprint}-{profile_digest}.json",
    )


def save_footprints(
    footprints: Mapping[str, PhaseFootprint],
    *,
    hw_fingerprint: str,
    profile: ActivationProfile,
    cache_dir: Optional[str] = None,
) -> str:
    digest = profile_key(profile)
    path = footprint_cache_path(hw_fingerprint, digest, cache_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "version": PHASE_FOOTPRINT_VERSION,
        "hw_fingerprint": hw_fingerprint,
        "profile_digest": digest,
        "profile": profile.canonical(),
        "cards": {k: v.to_json() for k, v in footprints.items()},
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)
    return path


def load_footprints(
    *,
    hw_fingerprint: str,
    profile: ActivationProfile,
    cache_dir: Optional[str] = None,
) -> Dict[str, PhaseFootprint]:
    """Cached footprints for this (hardware, profile), or ``{}``."""
    digest = profile_key(profile)
    path = footprint_cache_path(hw_fingerprint, digest, cache_dir)
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("phase footprint %s unreadable (%s); ignoring.", path, e)
        return {}
    if int(d.get("version", -1)) != PHASE_FOOTPRINT_VERSION:
        logger.info(
            "phase footprint %s was written by version %s, this is version %s; "
            "ignoring it rather than reinterpreting its fields.",
            path,
            d.get("version"),
            PHASE_FOOTPRINT_VERSION,
        )
        return {}
    if d.get("hw_fingerprint") != hw_fingerprint or d.get("profile_digest") != digest:
        return {}
    return {k: PhaseFootprint.from_json(v) for k, v in (d.get("cards") or {}).items()}


def resolve_phase_footprint(
    card_uuid: str,
    *,
    hw_fingerprint: Optional[str],
    profile: ActivationProfile,
    cache_dir: Optional[str] = None,
) -> Optional[PhaseFootprint]:
    """The footprint for one card, or ``None``.

    Precedence, strictly:

    1. a cached MEASURED_PEAK / UPPER_BOUND for this (hardware, profile);
    2. the shipped reference-window bounds, but ONLY when the live hardware
       fingerprint and the profile both match the rig they were taken on;
    3. ``None``.

    ``None`` is a real answer and the caller must treat it as one. There is
    deliberately no fourth branch: reinstating either falsified formula here
    would restore the exact failure this module was written to end, and would
    do it invisibly.
    """
    if not hw_fingerprint:
        return None
    cached = load_footprints(
        hw_fingerprint=hw_fingerprint, profile=profile, cache_dir=cache_dir
    )
    hit = cached.get(card_uuid)
    if hit is not None:
        return hit
    if hw_fingerprint == REFERENCE_WINDOW_FINGERPRINT and profile_key(
        profile
    ) == profile_key(_REFERENCE_WINDOW_PROFILE):
        return reference_window_footprints().get(card_uuid)
    return None


def profile_from_server_args(
    server_args, architectures: Sequence[str]
) -> ActivationProfile:
    """Build the profile key from a ServerArgs without importing torch."""
    decode_max_bs = 0
    try:
        decode_max_bs = int(server_args.cuda_graph_config.decode.max_bs or 0)
    except Exception:  # pragma: no cover - config shape differences
        decode_max_bs = 0
    return ActivationProfile(
        architectures=tuple(str(a) for a in architectures),
        chunked_prefill_size=int(getattr(server_args, "chunked_prefill_size", 0) or 0),
        tp_size=int(getattr(server_args, "tp_size", 1) or 1),
        pp_size=int(getattr(server_args, "pp_size", 1) or 1),
        kv_cache_dtype=str(getattr(server_args, "kv_cache_dtype", "") or ""),
        speculative_num_draft_tokens=int(
            getattr(server_args, "speculative_num_draft_tokens", 0) or 0
        ),
        decode_max_bs=decode_max_bs,
    )
