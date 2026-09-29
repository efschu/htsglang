"""BOOTZEIT 3 Stufe 2b: the ranges group P kept in the page cache, released after D.

Stufe 2b lets P (``SGLANG_WEIGHT_LOADER_SHARED_CACHE=keep``, the first reader)
leave the NON-expert tensor ranges in the page cache for D (``=drop``, the last
reader), which drops every range it reads right after reading it. What D never
reads stays kept: z30u P kept 6702 MiB (PP0 3057 + PP1 1914 + PP2 1731), D
advised 4642 MiB away -- the ~2 GiB remainder had no one to let go of it.

This module closes that: each P rank writes the ranges it kept to a manifest
in the boot's manifest directory (``SGLANG_WEIGHT_LOADER_SHARED_CACHE_MANIFEST``,
set per group by the launcher); the launcher, AFTER group D is ready -- i.e.
after D's take-over, never inside the load path, so the boot time is untouched
-- drops exactly those ranges and MEASURES what that did: page-cache residency
of the ranges (``mincore``) before and after, and the cgroup's
``memory.current``/``file``/``shmem`` around it.

What the line says is a measurement, not a promise: a range the kernel had
already reclaimed under the cgroup ceiling counts in ``advised_mib`` and NOT in
``evicted_mib``. And a released page is page cache the kernel could have
traded anyway -- the W98 cushion (``file - shmem``) counts it as cushion, so the
release moves it from ``file`` to the free pool; it does not create room.

Stdlib only: the launcher imports it without torch.
"""
from __future__ import annotations

import glob
import json
import mmap
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

MANIFEST_ENV = "SGLANG_WEIGHT_LOADER_SHARED_CACHE_MANIFEST"
MARKER = "WEG2-SHARED-CACHE-RELEASE"
MIB = 1 << 20
GIB = 1 << 30

Range = Tuple[str, int, int]  # (path, offset, length)


def manifest_dir_for(group_log: str) -> str:
    """ONE directory per boot, the same for P and D: next to the group logs
    (``<base>.P.log`` / ``<base>.D.log`` -> ``<base>.shared_cache``)."""
    base = group_log
    for suf in (".P.log", ".D.log"):
        if base.endswith(suf):
            base = base[: -len(suf)]
            break
    return f"{base}.shared_cache"


def write_keep_manifest(d: str, ranges: List[Range], kept_bytes: int) -> Optional[str]:
    """P side: one stream's kept ranges, atomically (a rank may run several
    streams -- target and draft -- each gets its own file). None = no directory."""
    if not d:
        return None
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"keep_{os.getpid()}_{time.time_ns()}.json")
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump({"pid": os.getpid(), "kept_bytes": int(kept_bytes),
                   "ranges": [[p, int(o), int(n)] for p, o, n in ranges]}, f)
    os.replace(tmp, path)
    return path


def read_manifests(d: str) -> Tuple[List[Range], int, int]:
    """(ranges, kept_bytes, manifests) over every P rank's manifest."""
    ranges: List[Range] = []
    kept = n = 0
    for path in sorted(glob.glob(os.path.join(d, "keep_*.json"))):
        try:
            with open(path) as f:
                m = json.load(f)
        except (OSError, ValueError):
            continue
        n += 1
        kept += int(m.get("kept_bytes") or 0)
        for p, o, ln in m.get("ranges") or ():
            ranges.append((str(p), int(o), int(ln)))
    return ranges, kept, n


def merge_ranges(ranges: List[Range]) -> Dict[str, List[Tuple[int, int]]]:
    """Per file, overlapping/adjacent ranges merged -- residency and advice are
    then counted once per byte even where two P ranks kept the same range."""
    by: Dict[str, List[Tuple[int, int]]] = {}
    for p, o, n in ranges:
        if n > 0:
            by.setdefault(p, []).append((o, o + n))
    out: Dict[str, List[Tuple[int, int]]] = {}
    for p, iv in by.items():
        iv.sort()
        merged = [list(iv[0])]
        for a, b in iv[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out[p] = [(a, b - a) for a, b in merged]
    return out


def resident_bytes(fd: int, offset: int, length: int) -> int:
    """Bytes of [offset, offset+length) in the page cache (mincore over a
    read-only map of the range; the pages are not touched). 0 when unknown."""
    import ctypes

    page = mmap.PAGESIZE
    lo = offset & ~(page - 1)
    span = offset + length - lo
    if span <= 0:
        return 0
    try:
        m = mmap.mmap(fd, span, access=mmap.ACCESS_COPY, offset=lo)
    except (OSError, ValueError):
        return 0
    anchor = None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        npages = (span + page - 1) // page
        vec = (ctypes.c_ubyte * npages)()
        anchor = ctypes.c_char.from_buffer(m)
        if libc.mincore(ctypes.c_void_p(ctypes.addressof(anchor)), ctypes.c_size_t(span), vec) != 0:
            return 0
        return min(length, page * sum(1 for v in vec if v & 1))
    except Exception:  # noqa: BLE001 -- an instrument, never a boot killer
        return 0
    finally:
        del anchor
        try:
            m.close()
        except BufferError:
            pass


def release(d: str, *, read_pressure: Optional[Callable[[], dict]] = None) -> Optional[dict]:
    """Drop every range P kept; measure it. None = no manifest (2b off)."""
    ranges, kept, n = read_manifests(d)
    if not n:
        return None
    t0 = time.perf_counter()
    before = read_pressure() if read_pressure is not None else {}
    merged = merge_ranges(ranges)
    advised = res0 = res1 = 0
    missing = 0
    for path, iv in sorted(merged.items()):
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            missing += 1
            continue
        try:
            for off, ln in iv:
                res0 += resident_bytes(fd, off, ln)
                try:
                    os.posix_fadvise(fd, off, ln, os.POSIX_FADV_DONTNEED)
                    advised += ln
                except OSError:
                    pass
                res1 += resident_bytes(fd, off, ln)
        finally:
            os.close(fd)
    after = read_pressure() if read_pressure is not None else {}
    return {
        "manifests": n, "files": len(merged), "files_missing": missing,
        "kept_mib": round(kept / MIB), "advised_mib": round(advised / MIB),
        "resident_before_mib": round(res0 / MIB), "resident_after_mib": round(res1 / MIB),
        "evicted_mib": round((res0 - res1) / MIB),
        "cg_current_before_gib": _r(before.get("current_gib")),
        "cg_current_after_gib": _r(after.get("current_gib")),
        "cushion_before_gib": _cushion(before), "cushion_after_gib": _cushion(after),
        "shmem_gib": _r(after.get("shmem_gib")),
        "ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def _r(v) -> Optional[float]:
    return None if v is None else round(float(v), 3)


def _cushion(pr: dict) -> Optional[float]:
    f, sh = pr.get("file_gib"), pr.get("shmem_gib")
    return None if f is None or sh is None else round(float(f) - float(sh), 3)


def line(rec: dict) -> str:
    def g(k):
        v = rec.get(k)
        return "unreadable" if v is None else f"{v:.2f}"

    return (
        f"{MARKER} kept_mib={rec['kept_mib']} advised_mib={rec['advised_mib']} "
        f"evicted_mib={rec['evicted_mib']} resident_before_mib={rec['resident_before_mib']} "
        f"resident_after_mib={rec['resident_after_mib']} manifests={rec['manifests']} "
        f"files={rec['files']} missing={rec['files_missing']} memory.current "
        f"{g('cg_current_before_gib')} -> {g('cg_current_after_gib')} GiB cushion "
        f"{g('cushion_before_gib')} -> {g('cushion_after_gib')} GiB ms={rec['ms']} "
        "(BOOTZEIT 3 Stufe 2b: the ranges P kept for D, dropped after D is ready; "
        "advised = DONTNEED issued, evicted = resident before - after by mincore -- "
        "a range the cgroup had already reclaimed is advised, not evicted)"
    )
