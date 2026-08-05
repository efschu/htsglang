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
"""Hardware residuals: the ledger terms that cannot be derived, measured once.

WHAT BELONGS HERE, AND WHY IT IS A SHORT LIST. Most of a card's demand follows
from configuration: a shard's bytes, a KV cell's bytes, a pool's slot count, a
graph ladder's captured tokens. Those are MODELED and never appear in this
module. What does not follow from configuration is the part that belongs to the
CARD, the DRIVER and the BUILD:

* the per-process CUDA context (a fixed cost per process on a device, which
  differs by architecture and driver, and which every co-located rank pays
  again -- #237/#403 also make the parent/tokenizer process pay it);
* the JIT / kernel workspace pools the attention and comm backends allocate on
  first use, whose size depends on which kernels the installed wheel compiled;
* the caching allocator's granularity, i.e. the bytes it rounds a request up to
  and the bytes it holds in a segment it cannot return.

These were previously literals: ``_PREDICT_OVERHEAD_MIB = 1280`` in
``uneven_perf``, ``FIXED_PROCESS_POST_MIB = 1536`` in the planner's solver. Two
different guesses for the same physical quantity, neither traceable to a
measurement, both silently wrong on a card nobody tried. This module replaces
them with ONE measurement per hardware profile.

THE FINGERPRINT IS THE CONTRACT. The cache key follows the #213 card-probe
canon exactly (same cache directory, same sha1-of-sorted-inputs discipline,
same atomic replace, same "a version bump invalidates rather than
reinterprets" rule as #513) and adds the two inputs a VRAM residual depends on
that a rate probe does not: the torch/CUDA build, because a different wheel
compiles different kernels and therefore different workspaces, and the driver.
A calibration whose fingerprint does not match the live rig is not used and not
adjusted -- it is a MISS, and a miss is honest.

WHAT A MISS DOES. It does not fall back to a literal, because a literal is the
defect. It makes the term UNCALIBRATED, and an uncalibrated hardware residual
is an :attr:`~sglang.srt.mem_ledger.terms.CardVramLedger.unbounded` item, which
refuses the boot with an instruction to run the probe. The probe is cheap
(single-digit seconds, one small allocation per card) and its result is cached
for the lifetime of the driver/wheel/card set, so paying it once is the whole
cost of never guessing again.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "VRAM_CALIBRATION_VERSION",
    "CardResidual",
    "CalibrationProfile",
    "calibration_fingerprint",
    "calibration_cache_path",
    "load_calibration",
    "save_calibration",
    "measure_calibration",
    "CalibrationMiss",
]


#: Bumped whenever the MEANING of a stored field changes. #513's rule: a
#: version mismatch invalidates the file, it never reinterprets its fields.
VRAM_CALIBRATION_VERSION = 1

#: Same directory as the #213 card probe on purpose -- one cache location for
#: everything keyed on "which rig is this", so an operator clearing stale
#: hardware state clears all of it at once.
from sglang.srt.rigmon.card_probe import CACHE_DIR  # noqa: E402


class CalibrationMiss(RuntimeError):
    """No calibration matches this rig's fingerprint."""


@dataclasses.dataclass(frozen=True)
class CardResidual:
    """The measured hardware residual of ONE card, per process on it.

    Every field is bytes measured on this card under this driver and wheel.
    None of them is a policy number, and none of them is scaled by anything
    the configuration chooses -- the CONFIGURATION-dependent part of a
    workspace is modeled elsewhere; what is here is the part that is present
    the moment a process touches the device and would be there for any config.
    """

    uuid: str
    name: str
    #: Bytes the CUDA primary context costs a process on this device, measured
    #: as (free before first allocation) - (free after context creation).
    cuda_context_bytes: int
    #: Bytes the caching allocator holds beyond what was asked for, measured
    #: by allocating a known size and reading reserved-minus-allocated. This
    #: is the granularity/segment residue, not fragmentation under load.
    allocator_granularity_bytes: int
    #: Bytes the attention/comm backends' lazy workspaces cost on first use,
    #: measured by running one tiny forward-shaped allocation cycle. Zero when
    #: the probe could not exercise a backend, and a zero here is reported as
    #: such rather than smoothed.
    lazy_workspace_bytes: int
    #: Every measurement is tagged with the state it was taken in (#149).
    note: str = ""

    @property
    def total_bytes(self) -> int:
        return (
            self.cuda_context_bytes
            + self.allocator_granularity_bytes
            + self.lazy_workspace_bytes
        )

    @property
    def total_mib(self) -> int:
        return self.total_bytes // (1 << 20)

    def to_json(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "CardResidual":
        return cls(
            uuid=str(d["uuid"]),
            name=str(d.get("name", "")),
            cuda_context_bytes=int(d["cuda_context_bytes"]),
            allocator_granularity_bytes=int(d["allocator_granularity_bytes"]),
            lazy_workspace_bytes=int(d.get("lazy_workspace_bytes", 0)),
            note=str(d.get("note", "")),
        )


@dataclasses.dataclass(frozen=True)
class CalibrationProfile:
    """Residuals for every card of one hardware profile, plus its fingerprint."""

    fingerprint: str
    driver: str
    build: str
    cards: Tuple[CardResidual, ...]
    measured_at: float = 0.0
    version: int = VRAM_CALIBRATION_VERSION

    def by_uuid(self) -> Dict[str, CardResidual]:
        return {c.uuid: c for c in self.cards}

    def residual(self, uuid: str) -> Optional[CardResidual]:
        return self.by_uuid().get(uuid)

    def to_json(self) -> dict:
        return {
            "version": self.version,
            "fingerprint": self.fingerprint,
            "driver": self.driver,
            "build": self.build,
            "measured_at": self.measured_at,
            "cards": [c.to_json() for c in self.cards],
        }

    @classmethod
    def from_json(cls, d: dict) -> "CalibrationProfile":
        return cls(
            fingerprint=str(d["fingerprint"]),
            driver=str(d.get("driver", "")),
            build=str(d.get("build", "")),
            cards=tuple(CardResidual.from_json(c) for c in d.get("cards", [])),
            measured_at=float(d.get("measured_at", 0.0)),
            version=int(d.get("version", -1)),
        )


def _build_id() -> str:
    """The wheel identity a workspace footprint depends on.

    torch version plus its CUDA runtime version: those two decide which
    kernels exist and therefore which workspaces get allocated. Degrades to a
    marker string rather than raising, because a fingerprint that cannot be
    computed must still be a STABLE key -- an unstable key would silently
    re-measure forever.
    """
    try:
        import torch

        return f"torch{torch.__version__}+cuda{torch.version.cuda}"
    except Exception:  # pragma: no cover - torch-less environments
        return "torch-unavailable"


def calibration_fingerprint(
    uuids: Sequence[str],
    driver: Optional[str],
    build: Optional[str] = None,
) -> str:
    """The cache key: sorted card UUIDs + driver + build + format version.

    Byte-identical discipline to :func:`rigmon.card_probe.card_probe_cache_path`
    -- sha1 over a canonical json list, truncated to 12 hex chars -- so the two
    caches are read the same way and an operator learns one convention.
    """
    key = json.dumps(
        [
            sorted(str(u) for u in uuids),
            driver or "",
            build if build is not None else _build_id(),
            VRAM_CALIBRATION_VERSION,
        ]
    )
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def calibration_cache_path(fingerprint: str, cache_dir: Optional[str] = None) -> str:
    return os.path.join(cache_dir or CACHE_DIR, f"vram_calibration-{fingerprint}.json")


def live_fingerprint(
    inventory: Optional[Tuple[Sequence[dict], Optional[str]]] = None,
) -> Optional[Tuple[str, List[dict], str]]:
    """``(fingerprint, gpus, driver)`` for the rig visible right now, or None.

    ``inventory`` is the test injection point; production callers pass none and
    the #213 inventory (CUDA-order to NVML-order bridged over PCI BDF) answers.
    """
    if inventory is None:
        try:
            from sglang.srt.rigmon.card_probe import _inventory

            gpus, driver = _inventory()
        except Exception as e:
            logger.debug("VRAM calibration: no inventory (%s)", e)
            return None
    else:
        gpus, driver = inventory
    if not gpus:
        return None
    gpus = list(gpus)
    fp = calibration_fingerprint([g["uuid"] for g in gpus], driver)
    return fp, gpus, driver or ""


def rig_fingerprint() -> Optional[Tuple[str, List[dict], str]]:
    """``live_fingerprint`` for the WHOLE rig, independent of CVD masking.

    :func:`live_fingerprint` describes the cards the *calling process* can
    see, because its inventory is built from ``torch.cuda``. That is right in
    the launcher and wrong in a pinned worker: each window-5 rank
    fingerprinted only its own card (``fad5762191c9`` / ``9ce0ed6a79fc`` /
    ``b0c936f23170``) and ingest correctly refused all three against the
    rig's ``a191a0712717`` (#589). A dump has to name the rig it was measured
    on, not the slice of it one process was pinned to.

    NVML ignores ``CUDA_VISIBLE_DEVICES``, so its device list *is* the rig.
    The hashing is not duplicated: that inventory is handed to
    ``live_fingerprint``'s own injection point, so a pinned worker and an
    unmasked launcher compute the identical fingerprint through the identical
    code (``calibration_fingerprint`` sorts the UUIDs, so enumeration order
    cannot change the answer either).
    """
    try:
        from sglang.srt.registry import nvml as registry_nvml

        devices = registry_nvml.list_devices()
        driver = registry_nvml.driver_version()
    except Exception as e:
        logger.debug("VRAM calibration: no NVML rig inventory (%s)", e)
        return None
    gpus = [
        {
            # No cuda_index: the masked process cannot place the cards it
            # cannot see, and this field is not a fingerprint input anyway.
            "cuda_index": None,
            "uuid": d.uuid,
            "name": d.name,
            "total_mib": d.total_mib,
        }
        for d in sorted(devices, key=lambda d: d.index)
    ]
    return live_fingerprint(inventory=(gpus, driver))


def load_calibration(
    *,
    cache_dir: Optional[str] = None,
    inventory: Optional[Tuple[Sequence[dict], Optional[str]]] = None,
) -> Optional[CalibrationProfile]:
    """The calibration for THIS rig, or None.

    Never measures as a side effect: every read path (boot ledger, dashboard,
    planner) must be able to ask without spending GPU seconds, and the boot
    contract turns a None into a named refusal with the probe command in it.
    """
    live = live_fingerprint(inventory=inventory)
    if live is None:
        return None
    fingerprint, _gpus, _driver = live
    path = calibration_cache_path(fingerprint, cache_dir)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning("VRAM calibration %s unreadable (%s); ignoring it.", path, e)
        return None
    if int(d.get("version", -1)) != VRAM_CALIBRATION_VERSION:
        logger.info(
            "VRAM calibration %s was written by version %s, this is version "
            "%s; ignoring it rather than reinterpreting its fields.",
            path,
            d.get("version"),
            VRAM_CALIBRATION_VERSION,
        )
        return None
    profile = CalibrationProfile.from_json(d)
    if profile.fingerprint != fingerprint:
        # Belt and braces: the filename already encodes the key, but a file
        # copied between machines would otherwise be trusted.
        logger.info(
            "VRAM calibration %s carries fingerprint %s but this rig is %s; "
            "ignoring it.",
            path,
            profile.fingerprint,
            fingerprint,
        )
        return None
    return profile


def save_calibration(
    profile: CalibrationProfile, *, cache_dir: Optional[str] = None
) -> str:
    path = calibration_cache_path(profile.fingerprint, cache_dir)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(profile.to_json(), f, indent=1)
    os.replace(tmp, path)
    logger.info("VRAM calibration written to %s", path)
    return path


def _nvml_free_bytes(uuid: str) -> Optional[int]:
    """Free bytes on *uuid* read through NVML, which does NOT create a CUDA
    context. Returns None when NVML cannot answer."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByUUID(uuid.encode())
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).free)
    except Exception as e:  # pragma: no cover - NVML shape differences
        logger.debug("NVML free read failed for %s: %s", uuid, e)
        return None


def _measure_one_card(
    device_index: int, uuid: Optional[str] = None
) -> Tuple[int, int, int, str]:
    """``(context_bytes, granularity_bytes, workspace_bytes, note)`` on one card.

    Deliberately three small measurements rather than one big one: the terms
    have different lifetimes (a context lives as long as the process, a
    workspace until the backend is torn down, granularity is per allocation),
    and a single number would not tell a later reader which of them moved when
    the total moves.

    THE BASELINE MUST COME FROM NVML, NOT FROM TORCH. ``torch.cuda.mem_get_info``
    initialises CUDA in order to answer, so using it for the "before" reading
    measures the context AFTER the context already exists -- and reports the
    context as a couple of MiB instead of a few hundred. The reference window of
    2026-08-05 measured 2 MiB that way against a true 505 MiB on the 5090, an
    undercount of 503 MiB PER RANK PROCESS that would have been charged to
    nobody. Importing torch is fine (it allocates nothing); the first CUDA
    *call* is the line that must not be crossed before the baseline is taken.
    """
    import torch

    free_before = _nvml_free_bytes(uuid) if uuid else None
    if free_before is None:
        # No NVML: refuse rather than fall back to the torch reading, which is
        # the defect above wearing a plausible number.
        raise RuntimeError(
            "cannot read a pre-context NVML baseline for this card; refusing "
            "to measure the CUDA context with torch.cuda.mem_get_info, which "
            "creates the context it is supposed to measure and yields a "
            "few MiB instead of a few hundred"
        )
    # Creating the primary context: the first real allocation forces it.
    torch.cuda.set_device(device_index)
    anchor = torch.empty(1, dtype=torch.uint8, device=f"cuda:{device_index}")
    torch.cuda.synchronize(device_index)
    free_after_ctx, _ = torch.cuda.mem_get_info(device_index)
    context_bytes = max(int(free_before - free_after_ctx), 0)

    # Allocator granularity: ask for a size that is deliberately not a round
    # segment, and read what the caching allocator actually reserved for it.
    probe_bytes = 3 * (1 << 20) + 7
    before_reserved = torch.cuda.memory_reserved(device_index)
    block = torch.empty(probe_bytes, dtype=torch.uint8, device=f"cuda:{device_index}")
    torch.cuda.synchronize(device_index)
    after_reserved = torch.cuda.memory_reserved(device_index)
    granularity_bytes = max(int(after_reserved - before_reserved) - probe_bytes, 0)
    del block

    # Lazy workspace: one matmul is enough to force cuBLAS to create its
    # handle and workspace, which is the largest of the lazily-created pools
    # and the one that surprises a budget.
    ws_before = torch.cuda.memory_reserved(device_index)
    a = torch.zeros((256, 256), dtype=torch.float16, device=f"cuda:{device_index}")
    b = torch.zeros((256, 256), dtype=torch.float16, device=f"cuda:{device_index}")
    _ = a @ b
    torch.cuda.synchronize(device_index)
    free_after_ws, _ = torch.cuda.mem_get_info(device_index)
    ws_after = torch.cuda.memory_reserved(device_index)
    # The handle/workspace lives OUTSIDE the caching allocator, so it shows up
    # as free memory that disappeared without the allocator reserving it.
    allocator_delta = int(ws_after - ws_before)
    workspace_bytes = max(int(free_after_ctx - free_after_ws) - allocator_delta, 0)
    del a, b, anchor

    note = ""
    try:
        note = f"driver-visible free at probe: {free_before // (1 << 20)} MiB"
    except Exception:  # pragma: no cover
        pass
    return context_bytes, granularity_bytes, workspace_bytes, note


def measure_calibration(
    *,
    cache_dir: Optional[str] = None,
    inventory: Optional[Tuple[Sequence[dict], Optional[str]]] = None,
    save: bool = True,
) -> CalibrationProfile:
    """Measure the hardware residuals of every visible card and cache them.

    THE ONLY function in the ledger that touches a GPU. Kept separate from
    every read path so that "the boot printed a ledger" never means "the boot
    silently spent GPU seconds", and so the hermetic tests can exercise the
    whole ledger with an injected profile.
    """
    import time

    import torch

    live = live_fingerprint(inventory=inventory)
    if live is None:
        raise CalibrationMiss(
            "no CUDA/NVML inventory to calibrate against; the VRAM ledger "
            "cannot measure hardware residuals on this host"
        )
    fingerprint, gpus, driver = live
    residuals: List[CardResidual] = []
    for gpu in gpus:
        index = int(gpu["cuda_index"])
        ctx, gran, ws, note = _measure_one_card(index, str(gpu["uuid"]))
        residuals.append(
            CardResidual(
                uuid=str(gpu["uuid"]),
                name=str(gpu.get("name", "")),
                cuda_context_bytes=ctx,
                allocator_granularity_bytes=gran,
                lazy_workspace_bytes=ws,
                note=note,
            )
        )
        torch.cuda.empty_cache()
    profile = CalibrationProfile(
        fingerprint=fingerprint,
        driver=driver,
        build=_build_id(),
        cards=tuple(residuals),
        measured_at=time.time(),
    )
    if save:
        save_calibration(profile, cache_dir=cache_dir)
    return profile
