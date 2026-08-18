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
    "note_capture_begin",
    "note_capture_end",
    "record_prefill_peak",
]

DUMP_ENV = "SGLANG_PHASE_FOOTPRINT_DUMP"


def is_armed() -> bool:
    return bool(os.environ.get(DUMP_ENV))


def reset_peaks(device_index: int = 0) -> None:
    """Re-base the peak counters, and record the floor they were re-based to.

    ``torch.cuda.reset_peak_memory_stats`` does not zero the peak: it re-bases
    it at whatever is allocated *right now*, which after model load and KV
    sizing is weights + KV pool + captured graphs. The counter read back later
    is therefore an ABSOLUTE resident figure, not the activation cost. That is
    what the window-5 dumps recorded -- 26555 / 17306 / 16368 MiB, the whole
    footprint of each rank rather than its prefill transient (#589).

    So every re-base records its floor here, and the activation number the
    ledger ingests is peak MINUS that floor. The raw peak is still dumped
    alongside it, labelled, because it is the figure the operator sees in
    ``nvidia-smi`` and dropping it would make the two impossible to reconcile.
    """
    global _peak_floor_bytes
    if not is_armed():
        return
    try:
        import torch

        torch.cuda.reset_peak_memory_stats(device_index)
    except Exception as e:  # pragma: no cover - torch shape differences
        logger.debug("could not reset peak memory stats: %s", e)
        return
    _peak_floor_bytes = int(read_peaks(device_index).get("allocated_bytes", 0))


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
    peak_floor_bytes: Optional[int] = None,
) -> Optional[str]:
    """Write this rank's dump. One file per rank, so no collective is needed
    and a rank that dies mid-run simply contributes nothing.

    Three activation numbers are written, not one, because they answer
    different questions and window 5 proved they get confused otherwise:

    ``activation_peak_bytes``   the raw counter -- ABSOLUTE, weights and KV
                                included, because the reset re-bases rather
                                than zeroes (see :func:`reset_peaks`).
    ``peak_floor_bytes``        what was resident when the bracket opened.
    ``activation_delta_bytes``  peak minus floor: the prefill transient, and
                                the only one of the three the ledger reserves
                                for. ``None`` when no floor was recorded --
                                an honest absence, never a silent fallback to
                                the raw peak, which would re-introduce the
                                exact over-charge this field exists to fix.
    """
    directory = dump_dir or os.environ.get(DUMP_ENV)
    if not directory:
        return None
    delta: Optional[int] = None
    if peak_floor_bytes is not None:
        delta = max(0, int(activation_peak_bytes) - int(peak_floor_bytes))
    try:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"phase_footprint_rank{rank}.json")
        payload = {
            "rank": rank,
            "card_uuid": card_uuid,
            "hw_fingerprint": hw_fingerprint,
            "profile": profile_canonical,
            "activation_peak_bytes": int(activation_peak_bytes),
            "peak_floor_bytes": (
                None if peak_floor_bytes is None else int(peak_floor_bytes)
            ),
            "activation_delta_bytes": delta,
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


# ---------------------------------------------------------------------------
# Serving-path wiring
# ---------------------------------------------------------------------------
#
# The three call sites below are what turn this module from a design into a
# measurement. Before them the probe had unit tests and no callers, so
# ``SGLANG_PHASE_FOOTPRINT_DUMP`` armed nothing and ``probe_activation.py
# ingest`` could only ever report "No rank dumps" -- the ledger stayed on its
# shipped UPPER_BOUNDs with no way to reach MEASURED_PEAK.
#
# State is module-level (process-wide) ON PURPOSE. ``torch.cuda.memory_stats``
# is per-device and process-wide, and a rank IS a process, so the number the
# ledger reserves for is the process peak -- not one model runner's. This also
# makes the speculative case right for free: the target runner and the NEXTN
# draft runner both capture graphs in the same process, and their capture costs
# must ADD rather than overwrite.

_baseline_allocated: Optional[int] = None
_capture_bytes_total: int = 0
_activation_peak_bytes: int = 0
_identity: Optional[dict] = None
#: What was resident at the last peak re-base; see :func:`reset_peaks`. None
#: until a bracket is opened, which is why the delta can be honestly absent.
_peak_floor_bytes: Optional[int] = None


def _device_index() -> int:
    try:
        import torch

        return int(torch.cuda.current_device())
    except Exception:  # pragma: no cover - non-CUDA platforms
        return 0


def note_capture_begin() -> None:
    """Baseline immediately before graph capture (KV pool already sized)."""
    global _baseline_allocated
    if not is_armed():
        return
    peaks = read_peaks(_device_index())
    _baseline_allocated = int(peaks.get("allocated_bytes", 0))
    reset_peaks(_device_index())


def note_capture_end() -> None:
    """Fold this capture's PERSISTENT cost in, then re-baseline for prefill.

    Capture is charged as the delta in LIVE allocations across the capture,
    not as the peak: the ledger reserves for the memory the captured graphs go
    on holding, and a transient spike during capture is already gone by the
    time the first prefill runs. The peak counters are reset afterwards so the
    activation measurement starts from the post-capture steady state.
    """
    global _capture_bytes_total, _baseline_allocated
    if not is_armed():
        return
    if _baseline_allocated is not None:
        current = int(read_peaks(_device_index()).get("allocated_bytes", 0))
        _capture_bytes_total += max(0, current - _baseline_allocated)
        _baseline_allocated = None
    reset_peaks(_device_index())


def _resolve_identity(model_runner) -> Optional[dict]:
    """Card UUID, rig fingerprint and activation profile for the dump."""
    global _identity
    # (see _global_rank below for why this is not tp_rank)
    if _identity is not None:
        return _identity or None
    try:
        from sglang.srt.mem_ledger.activation import profile_from_server_args
        from sglang.srt.mem_ledger.calibration import rig_fingerprint
        from sglang.srt.mem_ledger.engine import _model_architectures
        from sglang.srt.registry import nvml as registry_nvml

        server_args = model_runner.server_args
        # The RIG fingerprint, not this process's. Every rank here is pinned
        # to one card by CUDA_VISIBLE_DEVICES, and ``live_fingerprint`` would
        # hash only that card -- three ranks, three different fingerprints,
        # none of them the rig's, and ingest rightly refuses all three (#589).
        live = rig_fingerprint()
        profile = profile_from_server_args(
            server_args, _model_architectures(server_args)
        )
        _identity = {
            "card_uuid": registry_nvml.current_device_uuid(),
            "hw_fingerprint": live[0] if live else "",
            "profile_canonical": profile.canonical(),
            "rank": _global_rank(model_runner),
        }
    except Exception as e:  # pragma: no cover - NVML/config availability
        logger.warning("phase footprint probe cannot identify this rank: %s", e)
        _identity = {}
    return _identity or None


def _global_rank(model_runner) -> int:
    """The rank that makes this dump's FILENAME unique.

    NOT ``tp_rank``. Under pure pipeline parallelism -- pp_size 3, tp_size 1 --
    every rank's tp_rank is 0, so all three wrote
    ``phase_footprint_rank0.json`` over one another and the ingest saw a single
    card. The dumps were never wrong, only two of the three were destroyed, and
    the ledger then went on refusing the terms it had in fact measured: boot
    v7pp5 left exactly one file behind for three cards.

    The distributed rank is unique across the job by definition, which is the
    property the filename needs. tp_rank is kept as the fallback for a run with
    no process group, where it is 0 and correct because there is one process.
    """
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return int(dist.get_rank())
    except Exception:  # noqa: BLE001 - identity may never break the probe
        pass
    return int(getattr(model_runner, "tp_rank", 0) or 0)


def record_prefill_peak(model_runner, num_tokens: int) -> None:
    """After a prefill: keep the running peak and rewrite this rank's dump.

    Rewritten on every new high rather than once at exit, because there is no
    reliable "last prefill" to hook and a rank that is killed mid-run should
    still leave its best measurement behind. The deepest prefill the rank is
    driven through is the one that sets the number, which is why the operator
    drives a representative deep prefill before ingesting.
    """
    global _activation_peak_bytes
    if not is_armed():
        return
    peaks = read_peaks(_device_index())
    peak = int(peaks.get("allocated_peak_bytes", 0))
    if peak <= _activation_peak_bytes:
        return
    _activation_peak_bytes = peak
    identity = _resolve_identity(model_runner)
    if not identity:
        return
    write_footprint_dump(
        rank=identity["rank"],
        card_uuid=identity["card_uuid"],
        hw_fingerprint=identity["hw_fingerprint"],
        profile_canonical=identity["profile_canonical"],
        activation_peak_bytes=peak,
        capture_bytes=_capture_bytes_total,
        reserved_peak_bytes=int(peaks.get("reserved_peak_bytes", 0)),
        prefill_tokens=int(num_tokens),
        peak_floor_bytes=_peak_floor_bytes,
    )
