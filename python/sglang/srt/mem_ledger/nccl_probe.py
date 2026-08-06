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
"""Measure the NCCL communicator buffers, which is the only way to price them.

WHY THIS MODULE HAS TO EXIST. ``TERM_NCCL_BUFFERS`` in ``engine.py`` shipped
with a supply path that nothing ever filled: ``DemandInputs`` declares
``nccl_buffer_mib_per_gpu`` and ``nccl_signature``, #598 wired
``communicator_groups`` so the term could reach NOT_APPLICABLE, but no caller
in the tree ever assigned the two fields that reach ``priced``. The
consequence was total rather than partial: on any launch where some group
builds a communicator -- i.e. every TP>1 boot -- the term was permanently
UNBOUNDED, so ``--enable-vram-ledger`` refused at parse time and could never
boot at all. This module closes that path.

WHY ``torch.cuda.memory_stats`` IS THE WRONG INSTRUMENT HERE, which is the
mirror image of the lesson in ``activation_probe``. There the counters were
right and ``nvidia-smi`` was wrong. Here it inverts: libnccl allocates its
buffers with its own ``cudaMalloc`` calls, entirely outside PyTorch's caching
allocator, so ``memory_stats`` cannot see them at all and would report a flat
zero no matter how large they are. The instrument that does see them is
``torch.cuda.mem_get_info``, which asks the DRIVER how much of the card is
free.

WHAT IS NETTED OUT. The driver-level delta across the construction window
would also capture any torch allocation that happens to land inside it, so
the reserved-bytes delta is subtracted:

    nccl_mib = (free_before - free_after) - (reserved_after - reserved_before)

That leaves exactly the non-torch growth, which on this window is libnccl.
The result is clamped at zero: a negative value means something FREED memory
concurrently, which makes the sample uninformative rather than negative.

MEASUREMENT VALIDITY, stated because it bounds where this may be trusted. The
window is a wall-clock bracket around one constructor, and it attributes every
non-torch byte that appears on the card during it to libnccl. That is sound
when one rank owns the card, which is the pinned-``CUDA_VISIBLE_DEVICES``
arrangement every rank here runs under. It is NOT sound for co-located ranks
sharing a physical card: two ranks constructing communicators concurrently
each see the other's allocation in their own delta, and both over-report. Such
a sample is recorded with ``exclusive: false`` and ingest refuses it rather
than folding it in, because an over-report of this term inflates the reserve
and silently shrinks the KV pool -- the #602 failure direction.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from typing import Dict, Mapping, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "DUMP_ENV",
    "SIGNATURE_ENV",
    "is_armed",
    "published_signature",
    "publish_signature",
    "measure_communicator_init",
    "write_nccl_dump",
    "ingest_dumps",
    "load_nccl_buffers",
    "nccl_cache_path",
    "save_nccl_buffers",
]

#: Arms the in-process measurement. Same discipline as
#: ``SGLANG_PHASE_FOOTPRINT_DUMP``: unset means the serving path does not pay
#: for an instrument nobody asked for.
DUMP_ENV = "SGLANG_NCCL_BUFFER_DUMP"

#: The launcher publishes the communicator-set signature here and the ranks
#: inherit it. Straight from the #605 boot-id lesson: the ledger is built
#: during argument resolution in the LAUNCHER, the measurement is taken by the
#: RANKS, and if each side derived the key independently the two halves would
#: be filed under different names and nothing would ever match.
SIGNATURE_ENV = "SGLANG_NCCL_SIGNATURE"


def is_armed() -> bool:
    return bool(os.environ.get(DUMP_ENV))


def publish_signature(signature: str) -> None:
    """Launcher side: state what the ranks are about to measure."""
    if signature:
        os.environ[SIGNATURE_ENV] = str(signature)


def published_signature() -> str:
    return (os.environ.get(SIGNATURE_ENV) or "").strip()


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


class measure_communicator_init:
    """Context manager bracketing one PyNccl communicator construction.

    Never raises into the caller. A probe that breaks a boot is worse than a
    term that stays unbounded, and the whole point of the bracket is that the
    boot it measures is a boot that would have happened anyway.
    """

    def __init__(self, group_name: str, device_index: Optional[int] = None):
        self.group_name = group_name
        self.device_index = device_index
        self.mib: Optional[float] = None
        self._free_before: Optional[int] = None
        self._reserved_before: Optional[int] = None

    def _sample(self):
        import torch

        idx = self.device_index
        if idx is None:
            idx = torch.cuda.current_device()
        free, _total = torch.cuda.mem_get_info(idx)
        reserved = torch.cuda.memory_reserved(idx)
        return int(free), int(reserved)

    def __enter__(self):
        if not is_armed():
            return self
        try:
            import torch

            torch.cuda.synchronize()
            self._free_before, self._reserved_before = self._sample()
        except Exception as e:  # pragma: no cover - CUDA availability
            logger.debug("NCCL buffer probe could not open its bracket: %s", e)
            self._free_before = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if not is_armed() or self._free_before is None or exc_type is not None:
            return False
        try:
            import torch

            torch.cuda.synchronize()
            free_after, reserved_after = self._sample()
            driver_delta = self._free_before - free_after
            torch_delta = reserved_after - (self._reserved_before or 0)
            self.mib = max(0.0, (driver_delta - torch_delta) / (1024.0 * 1024.0))
            _record(self.group_name, self.mib)
        except Exception as e:  # pragma: no cover - CUDA availability
            logger.debug("NCCL buffer probe could not close its bracket: %s", e)
        return False


#: Per-process accumulation. A rank builds SEVERAL groups (world, tp, ...) and
#: each one allocates its own buffers, so the card's charge is their SUM, not
#: the last one measured.
_per_group: Dict[str, float] = {}


def _record(group_name: str, mib: float) -> None:
    _per_group[group_name] = float(mib)
    _write_current()


def _resolve_identity() -> Optional[dict]:
    try:
        from sglang.srt.mem_ledger.calibration import rig_fingerprint
        from sglang.srt.registry import nvml as registry_nvml

        # The RIG fingerprint, not this process's -- identical reasoning to
        # activation_probe._resolve_identity (#589): every rank is pinned to
        # one card, and a per-process fingerprint would give three ranks three
        # different keys, none of them the rig's.
        live = rig_fingerprint()
        return {
            "card_uuid": registry_nvml.current_device_uuid(),
            "hw_fingerprint": live[0] if live else "",
        }
    except Exception as e:  # pragma: no cover - NVML availability
        logger.debug("NCCL buffer probe cannot identify this rank: %s", e)
        return None


def _exclusive_card() -> bool:
    """True when this process is the only rank bound to its card.

    Read from ``CUDA_VISIBLE_DEVICES``: the pinned arrangement gives each rank
    exactly one visible device. This is a necessary condition, not a
    sufficient one -- two ranks can be pinned to the SAME single device -- so
    the co-location count is what ingest actually checks. Recorded here so the
    dump carries the fact rather than ingest having to guess it.
    """
    cvd = (os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    return len([p for p in cvd.split(",") if p.strip()]) == 1


def _write_current() -> None:
    identity = _resolve_identity()
    if not identity:
        return
    write_nccl_dump(
        card_uuid=identity["card_uuid"],
        hw_fingerprint=identity["hw_fingerprint"],
        signature=published_signature(),
        per_group_mib=dict(_per_group),
        exclusive=_exclusive_card(),
    )


def write_nccl_dump(
    *,
    card_uuid: str,
    hw_fingerprint: str,
    signature: str,
    per_group_mib: Mapping[str, float],
    exclusive: bool = True,
    dump_dir: Optional[str] = None,
) -> Optional[str]:
    """One file per card. Rewritten as each group is measured, so a rank that
    dies after the TP group still leaves that group's number behind.
    """
    directory = dump_dir or os.environ.get(DUMP_ENV)
    if not directory:
        return None
    try:
        os.makedirs(directory, exist_ok=True)
        stem = card_uuid or "unknown"
        path = os.path.join(directory, f"nccl_buffers_{stem}.json")
        payload = {
            "card_uuid": card_uuid,
            "hw_fingerprint": hw_fingerprint,
            "nccl_signature": signature,
            "per_group_mib": {str(k): float(v) for k, v in per_group_mib.items()},
            "total_mib": float(sum(per_group_mib.values())),
            "exclusive": bool(exclusive),
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, path)
        return path
    except OSError as e:  # pragma: no cover - filesystem differences
        logger.warning("could not write the NCCL buffer dump: %s", e)
        return None


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

from sglang.srt.rigmon.card_probe import CACHE_DIR  # noqa: E402


def nccl_cache_path(
    hw_fingerprint: str, signature: str, cache_dir: Optional[str] = None
) -> str:
    return os.path.join(
        cache_dir or CACHE_DIR, f"nccl_buffers-{hw_fingerprint}-{signature}.json"
    )


def save_nccl_buffers(
    *,
    hw_fingerprint: str,
    signature: str,
    per_uuid_mib: Mapping[str, float],
    cache_dir: Optional[str] = None,
) -> str:
    path = nccl_cache_path(hw_fingerprint, signature, cache_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "hw_fingerprint": hw_fingerprint,
        "nccl_signature": signature,
        "per_uuid_mib": {str(k): float(v) for k, v in per_uuid_mib.items()},
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    os.replace(tmp, path)
    return path


def load_nccl_buffers(
    hw_fingerprint: str, signature: str, cache_dir: Optional[str] = None
) -> Optional[Dict[str, float]]:
    """``{card uuid: MiB}`` measured for this (rig, communicator set), or None.

    Both keys must match. The fingerprint invalidates on a card/driver/build
    change, the signature on a communicator-set change; neither alone is
    enough, which is why the term carries the signature next to the number.
    """
    path = nccl_cache_path(hw_fingerprint, signature, cache_dir)
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if d.get("hw_fingerprint") != hw_fingerprint:
        return None
    if d.get("nccl_signature") != signature:
        return None
    per = d.get("per_uuid_mib") or {}
    if not isinstance(per, dict) or not per:
        return None
    return {str(k): float(v) for k, v in per.items()}


def ingest_dumps(
    dump_dir: str, cache_dir: Optional[str] = None
) -> "tuple[Optional[str], list]":
    """Fold per-card dumps into the fingerprinted cache.

    Returns ``(cache path or None, notes)``. Refuses rather than folds when a
    dump is not attributable: no signature, a signature the dumps disagree on,
    a missing fingerprint, or a non-exclusive card. Every refusal is a note,
    because a silently dropped card is a card the ledger then prices from
    nothing.
    """
    notes = []
    files = sorted(glob.glob(os.path.join(dump_dir, "nccl_buffers_*.json")))
    if not files:
        return None, ["No NCCL dumps in %s" % dump_dir]

    per_uuid: Dict[str, float] = {}
    signatures = set()
    fingerprints = set()
    for path in files:
        try:
            with open(path) as f:
                d = json.load(f)
        except (OSError, ValueError) as e:
            notes.append(f"{os.path.basename(path)}: unreadable ({e})")
            continue
        uuid = str(d.get("card_uuid") or "")
        sig = str(d.get("nccl_signature") or "")
        fp = str(d.get("hw_fingerprint") or "")
        if not uuid:
            notes.append(f"{os.path.basename(path)}: no card uuid, skipped")
            continue
        if not sig:
            notes.append(
                f"{uuid}: no communicator signature -- the launcher did not "
                f"publish {SIGNATURE_ENV}, so this measurement cannot be said "
                "to be valid for anything. Skipped."
            )
            continue
        if not fp:
            notes.append(f"{uuid}: no rig fingerprint, skipped")
            continue
        if not d.get("exclusive", True):
            notes.append(
                f"{uuid}: measured on a card this rank did not have to itself, "
                "so the driver delta contains another rank's allocation too. "
                "Skipped rather than over-charged."
            )
            continue
        signatures.add(sig)
        fingerprints.add(fp)
        per_uuid[uuid] = float(d.get("total_mib") or 0.0)

    if not per_uuid:
        return None, notes + ["Nothing attributable to ingest."]
    if len(signatures) > 1:
        return None, notes + [
            "The dumps disagree on the communicator signature (%s). They did "
            "not come from one launch; refusing to fold them."
            % ", ".join(sorted(signatures))
        ]
    if len(fingerprints) > 1:
        return None, notes + [
            "The dumps disagree on the rig fingerprint (%s); refusing."
            % ", ".join(sorted(fingerprints))
        ]

    path = save_nccl_buffers(
        hw_fingerprint=next(iter(fingerprints)),
        signature=next(iter(signatures)),
        per_uuid_mib=per_uuid,
        cache_dir=cache_dir,
    )
    notes.append("Ingested %d card(s)." % len(per_uuid))
    return path, notes
