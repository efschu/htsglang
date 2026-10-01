# SPDX-License-Identifier: Apache-2.0
"""Dual-model: the MODEL IMAGE -- a sleeping model's weights in a file.

Why (DUAL-MODEL-FLIP-KONZEPT-1001.md section 8.1). Under today's arm a waking
group refills its weights from the OTHER group of the same model (exchange
collect, ``weg2_memory_saver.exchange_owns_wake_refill``). A model flip puts
BOTH groups of a model to sleep, so there is no live source; a disk reload is
undefined on quantized checkpoints (the post-load pass replaces parameters);
TMS host backup is decided per allocation at load time and would make every
PHASE flip pay a full D2H.

What. At model sleep the awake group copies every byte of every TMS
allocation of each weights tag -- raw, already post-processed, exactly what
the captured graphs read -- into one file per rank on XFS (page cache: clean,
reclaimable, refaulted from NVMe when evicted). After the pause/resume of the
model wake the same allocations sit at the same VAs (TMS keeps the VA
reservation), and the bytes are copied back. No loader, no exchange, no
collective, rank-local.

Allocation rows come from the TMS fork (``tms_tag_allocations``). A device
checksum (int64 byte sum per allocation) is recorded at save and verified at
load, on the device, so the check costs no host pass over the bytes.

Refusals, all named, none silent:

* an allocation not fully mapped or not active at save (a span plan would
  leave unmapped holes the image cannot carry);
* at load: a different allocation set (pointer or size) than at save -- the
  image would land at addresses the graphs do not use;
* a checksum mismatch after load;
* a TMS build without ``tms_tag_allocations``.

The volume and file: one ``<dir>/<boot>-<group>-r<rank>.img`` + a JSON
manifest beside it. ``dir`` must not be tmpfs (that is the RAM this exists to
release); the caller checks with the #89 verdict (``hibernate_dir_verdict``).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

ALIGN = 2 << 20          # file offsets on 2 MiB, matches the TMS granule
CHUNK = 64 << 20         # host bounce per copy step
MANIFEST_SCHEMA = "weg2.model_image/1"


class ModelImageRefused(RuntimeError):
    """The image cannot be written or restored safely; named."""


@dataclass(frozen=True)
class Alloc:
    ptr: int
    size: int
    mapped: int
    active: bool


@dataclass
class Entry:
    tag: str
    ptr: int
    size: int
    offset: int
    checksum: Optional[int] = None


@dataclass
class ImagePlan:
    entries: List[Entry]
    total_bytes: int
    file_bytes: int
    meta: Dict[str, str] = field(default_factory=dict)

    def manifest(self) -> dict:
        return {"schema": MANIFEST_SCHEMA, "meta": dict(self.meta),
                "total_bytes": self.total_bytes, "file_bytes": self.file_bytes,
                "entries": [asdict(e) for e in self.entries]}

    @staticmethod
    def from_manifest(m: dict) -> "ImagePlan":
        if m.get("schema") != MANIFEST_SCHEMA:
            raise ModelImageRefused(f"manifest schema {m.get('schema')!r} != {MANIFEST_SCHEMA!r}")
        return ImagePlan(entries=[Entry(**e) for e in m["entries"]],
                         total_bytes=int(m["total_bytes"]), file_bytes=int(m["file_bytes"]),
                         meta=dict(m.get("meta") or {}))


# ---------------------------------------------------------------------------
# pure planning
# ---------------------------------------------------------------------------


def plan_image(allocs_by_tag: Dict[str, Sequence[Alloc]], meta: Optional[Dict[str, str]] = None) -> ImagePlan:
    entries: List[Entry] = []
    off = 0
    total = 0
    for tag in sorted(allocs_by_tag):
        rows = list(allocs_by_tag[tag])
        if not rows:
            raise ModelImageRefused(f"tag {tag!r} has no allocations -- nothing to image is a defect")
        for a in sorted(rows, key=lambda r: r.ptr):
            if not a.active:
                raise ModelImageRefused(f"tag {tag!r} allocation {a.ptr:#x} is PAUSED at save")
            if a.mapped != a.size:
                raise ModelImageRefused(
                    f"tag {tag!r} allocation {a.ptr:#x}: mapped {a.mapped} of {a.size} B "
                    f"(span plan) -- the image would carry unmapped holes")
            entries.append(Entry(tag=tag, ptr=a.ptr, size=a.size, offset=off))
            total += a.size
            off += (a.size + ALIGN - 1) // ALIGN * ALIGN
    return ImagePlan(entries=entries, total_bytes=total, file_bytes=off, meta=dict(meta or {}))


def check_wake(plan: ImagePlan, allocs_by_tag: Dict[str, Sequence[Alloc]]) -> None:
    saved: Dict[str, set] = {}
    for e in plan.entries:
        saved.setdefault(e.tag, set()).add((e.ptr, e.size))
    errs = []
    for tag in sorted(set(saved) | set(allocs_by_tag)):
        now = {(a.ptr, a.size) for a in allocs_by_tag.get(tag, ())}
        was = saved.get(tag, set())
        if now != was:
            gone = sorted(was - now)[:3]
            new = sorted(now - was)[:3]
            errs.append(f"tag {tag!r}: {len(was - now)} saved allocation(s) missing "
                        f"{[(hex(p), s) for p, s in gone]}, {len(now - was)} new "
                        f"{[(hex(p), s) for p, s in new]}")
        inactive = [hex(a.ptr) for a in allocs_by_tag.get(tag, ()) if not a.active]
        if inactive:
            errs.append(f"tag {tag!r}: allocation(s) {inactive[:3]} still PAUSED at load -- resume first")
    if errs:
        raise ModelImageRefused("model image does not fit the live allocations:\n  " + "\n  ".join(errs))


# ---------------------------------------------------------------------------
# I/O (device side injected so tests run on CPU tensors)
# ---------------------------------------------------------------------------


def _device_view(ptr: int, size: int, device):
    """A zero-copy uint8 CUDA tensor over [ptr, ptr+size)."""
    import torch

    class _Iface:
        __cuda_array_interface__ = {"shape": (size,), "typestr": "|u1",
                                    "data": (ptr, False), "version": 3}

    return torch.as_tensor(_Iface(), device=device)


def _checksum(t) -> int:
    import torch

    return int(t.sum(dtype=torch.int64).item())


def save(path: str, plan: ImagePlan, *, view: Callable = None, device=None,
         pin: bool = True, chunk: int = CHUNK) -> dict:
    """Write every entry's bytes to ``path`` and the manifest to ``path + '.json'``."""
    import torch

    view = view or (lambda p, n: _device_view(p, n, device))
    bounce = torch.empty(chunk, dtype=torch.uint8, pin_memory=pin)
    t0 = time.time()
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.ftruncate(fd, plan.file_bytes)
        for e in plan.entries:
            src = view(e.ptr, e.size)
            e.checksum = _checksum(src)
            done = 0
            while done < e.size:
                n = min(chunk, e.size - done)
                bounce[:n].copy_(src[done:done + n])
                os.pwrite(fd, memoryview(bounce[:n].numpy()), e.offset + done)
                done += n
        if os.environ.get("SGLANG_WEG2_MODEL_IMAGE_FSYNC") == "1":
            os.fsync(fd)
    finally:
        os.close(fd)
    with open(path + ".json", "w") as f:
        json.dump(plan.manifest(), f)
    return {"bytes": plan.total_bytes, "entries": len(plan.entries),
            "ms": round((time.time() - t0) * 1000, 1)}


def load(path: str, allocs_by_tag: Dict[str, Sequence[Alloc]], *, view: Callable = None,
         device=None, pin: bool = True, chunk: int = CHUNK) -> dict:
    """Copy the image back into the live allocations; verify every checksum."""
    import torch

    with open(path + ".json") as f:
        plan = ImagePlan.from_manifest(json.load(f))
    check_wake(plan, allocs_by_tag)
    view = view or (lambda p, n: _device_view(p, n, device))
    bounce = torch.empty(chunk, dtype=torch.uint8, pin_memory=pin)
    t0 = time.time()
    bad = []
    fd = os.open(path, os.O_RDONLY)
    try:
        for e in plan.entries:
            dst = view(e.ptr, e.size)
            done = 0
            while done < e.size:
                n = min(chunk, e.size - done)
                got = os.preadv(fd, [memoryview(bounce[:n].numpy())], e.offset + done)
                if got != n:
                    raise ModelImageRefused(f"short read {got}/{n} at {e.offset + done} in {path}")
                dst[done:done + n].copy_(bounce[:n])
                done += n
            if e.checksum is not None and _checksum(dst) != e.checksum:
                bad.append(f"{e.tag}@{e.ptr:#x}")
    finally:
        os.close(fd)
    if bad:
        raise ModelImageRefused(f"checksum mismatch after load: {bad[:5]} ({len(bad)} total)")
    return {"bytes": plan.total_bytes, "entries": len(plan.entries),
            "ms": round((time.time() - t0) * 1000, 1)}


# ---------------------------------------------------------------------------
# TMS binding
# ---------------------------------------------------------------------------


def tag_allocations(tag: str) -> List[Alloc]:
    """Every TMS allocation of ``tag`` via ``tms_tag_allocations``."""
    import ctypes

    from sglang.srt.utils.torch_memory_saver_adapter import _weg2_ring_symbol

    fn = _weg2_ring_symbol("tms_tag_allocations")
    if fn is None:
        raise ModelImageRefused("this TMS build has no tms_tag_allocations (dual-model fork patch)")
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
                   ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_int), ctypes.c_size_t]
    cap = 256
    while True:
        P = (ctypes.c_uint64 * cap)()
        S = (ctypes.c_uint64 * cap)()
        M = (ctypes.c_uint64 * cap)()
        A = (ctypes.c_int * cap)()
        n = int(fn(tag.encode(), P, S, M, A, cap))
        if n <= cap:
            return [Alloc(int(P[i]), int(S[i]), int(M[i]), bool(A[i])) for i in range(n)]
        cap = n
