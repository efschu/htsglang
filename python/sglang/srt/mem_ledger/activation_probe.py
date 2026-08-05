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
"""In-process hook that records one rank's phase footprint.

Armed only by ``SGLANG_PHASE_FOOTPRINT_DUMP``; absent that variable every
function here returns immediately and allocates nothing, so an unarmed boot is
byte-identical to one without this module.

The counters are ``torch.cuda.memory_stats()``, not ``nvidia-smi``, and the
difference is the entire point (see ``scripts/vram_ledger/probe_activation.py``):
``allocated_bytes.all.peak`` tracks LIVE allocations, so it sees a prefill
transient that fits inside a segment the caching allocator already holds.
``nvidia-smi`` does not, which is why the 2026-08-05 window could only bound the
activation peak instead of measuring it.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

__all__ = [
    "DUMP_ENV",
    "is_armed",
    "reset_peaks",
    "read_peaks",
    "write_footprint_dump",
]

DUMP_ENV = "SGLANG_PHASE_FOOTPRINT_DUMP"


def is_armed() -> bool:
    return bool(os.environ.get(DUMP_ENV))


def reset_peaks(device_index: int = 0) -> None:
    """Zero the peak counters. Called after the KV pool is sized so the
    measurement starts from the steady baseline rather than from the load."""
    if not is_armed():
        return
    try:
        import torch

        torch.cuda.reset_peak_memory_stats(device_index)
    except Exception as e:  # pragma: no cover - torch shape differences
        logger.debug("could not reset peak memory stats: %s", e)


def read_peaks(device_index: int = 0) -> dict:
    """``{allocated_peak_bytes, reserved_peak_bytes, allocated_bytes}``."""
    try:
        import torch

        stats = torch.cuda.memory_stats(device_index)
        return {
            "allocated_peak_bytes": int(stats.get("allocated_bytes.all.peak", 0)),
            "reserved_peak_bytes": int(stats.get("reserved_bytes.all.peak", 0)),
            "allocated_bytes": int(stats.get("allocated_bytes.all.current", 0)),
        }
    except Exception as e:  # pragma: no cover - torch shape differences
        logger.debug("could not read memory stats: %s", e)
        return {}


def write_footprint_dump(
    *,
    rank: int,
    card_uuid: str,
    hw_fingerprint: str,
    profile_canonical: list,
    activation_peak_bytes: int,
    capture_bytes: int,
    reserved_peak_bytes: int = 0,
    prefill_tokens: Optional[int] = None,
    dump_dir: Optional[str] = None,
) -> Optional[str]:
    """Write this rank's dump. One file per rank, so no collective is needed
    and a rank that dies mid-run simply contributes nothing."""
    directory = dump_dir or os.environ.get(DUMP_ENV)
    if not directory:
        return None
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"phase_footprint_rank{rank}.json")
        payload = {
            "rank": rank,
            "card_uuid": card_uuid,
            "hw_fingerprint": hw_fingerprint,
            "profile": profile_canonical,
            "activation_peak_bytes": int(activation_peak_bytes),
            "capture_bytes": int(capture_bytes),
            "reserved_peak_bytes": int(reserved_peak_bytes),
            "prefill_tokens": prefill_tokens,
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=1)
        os.replace(tmp, path)
        logger.info("phase footprint written to %s", path)
        return path
    except OSError as e:  # pragma: no cover - filesystem differences
        logger.warning("could not write the phase footprint dump: %s", e)
        return None
