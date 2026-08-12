# SPDX-License-Identifier: Apache-2.0
"""#644 discriminator: is the residual host anon RETAINED or merely UNTRIMMED?

THE OPEN QUESTION
-----------------
VAL-R4 measured, on the real Qwen3.6-35B-A3B GGUF checkpoint, that ~16 GB of
host ``RssAnon`` survives model load **on both sides of the #644 fix**:

    merged tip (#644 in)     peak 17.144 GB   plateau 15.974 GB
    same tree, hunk reverted peak 17.521 GB   plateau 16.646 GB

The fix moves the number in the right direction (672 MB lower plateau) but the
headline "the expert set is released from host RAM" does not survive contact
with a 16 GB checkpoint. RSS alone cannot say WHY the residue is there, and
that distinction decides what to do about it:

* **RETENTION** -- Python references to CPU tensors genuinely outlive load.
  That is a leak, it scales with the checkpoint, and it needs a named holder
  fixed at file:line.
* **ALLOCATOR** -- the bytes were freed by Python and are held by glibc's
  malloc arenas (or by torch's CPU caching), which never returned them to the
  kernel. Benign-ish: bounded by the peak, and addressable with a trim or an
  arena cap rather than by changing ownership.

RSS is identical in both cases. Nothing in a sampler can separate them.

WHY THIS IS IN-PROCESS
----------------------
The textbook discriminator is ``malloc_trim(0)`` under gdb. **gdb is not
installed on this box**, and installing a debugger to answer one question on a
serving rig is a worse trade than a gated instrument that ships. So this runs
inside the loading process, at the end of load, behind
``SGLANG_644_DISCRIMINATOR``, and is a no-op otherwise.

WHAT IT MEASURES, AND WHY THREE INSTRUMENTS RATHER THAN ONE
-----------------------------------------------------------
1. **Named holders.** ``param.data_container`` / ``param.expert_data_map`` are
   the two holders #644 names. If they are non-empty after load, the fix did
   not take on this path and the question is answered on the spot.
2. **Live CPU storages (the discriminating instrument).** A gc walk over every
   live ``torch.Tensor``, summing CPU storages deduplicated by data pointer,
   so aliasing views are not counted twice. This does not care WHICH holder is
   responsible: if ~16 GB of CPU tensor storage is reachable, references
   persist, whoever owns them. If the sum is small while RssAnon is ~16 GB,
   references do NOT persist and the residue cannot be a tensor leak.
3. **malloc_trim(0).** Releases free arena pages back to the kernel. What RSS
   gives back is a direct read of how much of the residue was already-freed
   memory the allocator was sitting on.

The verdict is only asserted when instruments 2 and 3 agree; a disagreement is
reported as ``MIXED`` with both numbers rather than resolved by preference.

The caller's RSS sampler stays valuable as the outside view -- this instrument
is the inside view, and neither replaces the other.
"""

from __future__ import annotations

import ctypes
import gc
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

ENV_FLAG = "SGLANG_644_DISCRIMINATOR"

#: Live CPU-storage bytes above which "references genuinely persist" is the
#: only reading. Deliberately far above incidental load-time residue (a few
#: hundred MB of tokenizer/config/pinned staging is normal) and far below the
#: ~16 GB the question is about, so neither answer depends on the threshold.
RETENTION_BYTES_THRESHOLD = 2 << 30

#: Fraction of the residue that malloc_trim must hand back before the residue
#: is called allocator-held rather than genuinely resident.
TRIM_FRACTION_THRESHOLD = 0.25


def _proc_status_kb(field: str, pid: str = "self") -> int:
    """One /proc/<pid>/status field in kB, or 0 if unreadable."""
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def rss_anon_kb(pid: str = "self") -> int:
    """RssAnon, NOT VmRSS.

    The GGUF is mmap'd, so VmRSS counts file pages that the #391 page dropper
    is supposed to release, which would hide exactly the anon population this
    is about.
    """
    return _proc_status_kb("RssAnon", pid)


def malloc_trim() -> bool:
    """``malloc_trim(0)`` via libc. False if libc has no such symbol."""
    try:
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim.argtypes = [ctypes.c_size_t]
        libc.malloc_trim.restype = ctypes.c_int
        libc.malloc_trim(0)
        return True
    except (OSError, AttributeError) as exc:  # noqa: BLE001 -- diagnosis only
        logger.warning("#644 discriminator: malloc_trim unavailable (%s)", exc)
        return False


def live_cpu_storage_bytes() -> Tuple[int, List[Tuple[int, str]]]:
    """(total CPU storage bytes reachable, largest few as (bytes, describe)).

    Deduplicated by storage data pointer: a narrow()/view of a 4 GiB storage
    keeps the WHOLE storage alive, and counting the view's own nbytes would
    under-report the retention by exactly the amount that matters. Pointer
    identity is therefore the unit, not the tensor.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover -- torch is always present in-tree
        return 0, []

    seen: Dict[int, int] = {}
    describe: Dict[int, str] = {}
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, torch.Tensor):
                continue
            if obj.device.type != "cpu":
                continue
            storage = obj.untyped_storage()
            ptr = storage.data_ptr()
            nbytes = storage.nbytes()
        except Exception:  # noqa: BLE001 -- a dead/odd tensor must not stop the census
            continue
        if ptr == 0 or nbytes <= 0:
            continue
        if ptr not in seen:
            seen[ptr] = nbytes
            describe[ptr] = f"{tuple(obj.shape)} {obj.dtype}"
    total = sum(seen.values())
    top = sorted(
        ((nb, describe.get(ptr, "?")) for ptr, nb in seen.items()), reverse=True
    )[:5]
    return total, top


def named_holder_residue(model: Any) -> List[str]:
    """The two holders #644 names, if any of them is still populated.

    Reported per parameter rather than as a count, because "which parameter"
    is the difference between a missed branch and a whole-model regression.
    """
    findings: List[str] = []
    if model is None:
        return findings
    try:
        named_params = list(model.named_parameters())
    except Exception:  # noqa: BLE001
        return findings
    for name, param in named_params:
        for attr in ("data_container", "expert_data_map"):
            holder = getattr(param, attr, None)
            if not holder:
                continue
            try:
                count = len(holder)
            except TypeError:
                count = -1
            if count:
                findings.append(f"{name}.{attr} holds {count} entries")
    return findings


def enabled() -> bool:
    """Read per call: the flag is set in the boot environment of the arm under
    test, and a value frozen at import would silently disarm an override."""
    return os.environ.get(ENV_FLAG, "").lower() in ("1", "true", "yes", "y")


def run_discriminator(model: Any = None, tag: str = "") -> Optional[Dict[str, Any]]:
    """Answer RETENTION vs ALLOCATOR for this process. No-op unless enabled.

    Returns the measurement dict (also logged), or None when disabled.
    """
    if not enabled():
        return None

    before_kb = rss_anon_kb()
    holders = named_holder_residue(model)

    gc.collect()
    after_gc_kb = rss_anon_kb()

    live_bytes, top = live_cpu_storage_bytes()

    trimmed = malloc_trim()
    after_trim_kb = rss_anon_kb()

    freed_kb = max(0, after_gc_kb - after_trim_kb)
    residue_kb = max(1, after_gc_kb)
    trim_fraction = freed_kb / residue_kb

    retention = live_bytes >= RETENTION_BYTES_THRESHOLD
    allocator = trim_fraction >= TRIM_FRACTION_THRESHOLD

    if retention and not allocator:
        verdict = "RETENTION"
    elif allocator and not retention:
        verdict = "ALLOCATOR"
    elif retention and allocator:
        verdict = "MIXED"
    else:
        # Neither instrument fired: the residue is anon that Python does not
        # own as tensors AND that the allocator will not give back -- the
        # signature of driver/pinned host allocations rather than of a leak.
        verdict = "NEITHER"

    result = {
        "tag": tag,
        "verdict": verdict,
        "rss_anon_before_mib": round(before_kb / 1024, 1),
        "rss_anon_after_gc_mib": round(after_gc_kb / 1024, 1),
        "rss_anon_after_trim_mib": round(after_trim_kb / 1024, 1),
        "trim_released_mib": round(freed_kb / 1024, 1),
        "trim_fraction": round(trim_fraction, 4),
        "malloc_trim_available": trimmed,
        "live_cpu_storage_mib": round(live_bytes / (1 << 20), 1),
        "named_holder_residue": holders,
        "largest_live_cpu_storages": [
            f"{nb / (1 << 20):.1f} MiB {desc}" for nb, desc in top
        ],
    }

    logger.warning(
        "#644-DISCRIMINATOR %s verdict=%s rss_anon %.1f -> %.1f MiB after "
        "gc, -> %.1f MiB after malloc_trim (released %.1f MiB, %.1f%% of the "
        "residue); live CPU tensor storage %.1f MiB; named holders: %s",
        tag or "(untagged)",
        verdict,
        result["rss_anon_before_mib"],
        result["rss_anon_after_gc_mib"],
        result["rss_anon_after_trim_mib"],
        result["trim_released_mib"],
        100.0 * trim_fraction,
        result["live_cpu_storage_mib"],
        "; ".join(holders) if holders else "all clear",
    )
    for line in result["largest_live_cpu_storages"]:
        logger.warning("#644-DISCRIMINATOR %s largest live CPU storage: %s", tag, line)

    return result
