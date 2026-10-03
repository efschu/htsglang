"""What SASS the installed sgl_kernel wheel ACTUALLY carries -- read from the
wheel's own ELF bytes, never from a card name.

SM89-DURCHSPIEL-1002: the release wheel is built
``SGL_KERNEL_LIMIT_CUDA_ARCHS=86;120a``, so ``sgl_kernel.fp8_scaled_mm`` on
sm_89 dispatches into a CUTLASS template that is compiled to
``CUTLASS_NOT_IMPLEMENTED()`` (printf + brkpt) inside the sm_86 pass -- a
device trap. The FP8 dispatch therefore asks the wheel instead of trusting
a capability number.

THE RECORD: every CUDA cubin carries a ``.note.nv.cuinfo`` ELF note --
header ``namesz=12, descsz=8, type=1000``, name ``NVIDIA Corp\\0``, a
descriptor whose u16 at offset 2 is the SM the cubin was compiled for
(verified against the installed wheel ``sm100/common_ops.abi3.so``: 52
notes sm_86 + 52 notes sm_120, no sm_89; ``flashmla_ops.abi3.so``: 25/25/25
of sm_90/sm_100/sm_103). Fatbins built by recent nvcc store their cubins in
ZSTD frames (magic ``28 b5 2f fd``); those are inflated with the
``zstandard`` module or the ``zstd`` binary and scanned the same way.

UNKNOWN is an honest answer: if neither decoder is available (or nothing
parses), :func:`wheel_carries_sass` returns ``None`` and the callers MUST
not treat that as "carries" -- a conservative fallback beats a device trap.
"""

from __future__ import annotations

import functools
import logging
import re
import struct
import subprocess
from pathlib import Path
from typing import List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

__all__ = [
    "scan_sass_archs",
    "sgl_kernel_so_paths",
    "sgl_kernel_sass_archs",
    "wheel_carries_sass",
]

#: ``.note.nv.cuinfo`` note header + name (see module docstring). The
#: descriptor begins at +24 (u16 tag 2, u16 SM, u32 CUDA version), so the
#: SM reads as a u16 at +26.
_CUINFO = struct.pack("<III", 12, 8, 1000) + b"NVIDIA Corp\x00"
_CUINFO_SM_OFF = 26
#: ZSTD frame magic (nvcc's fatbin entry compression).
_ZSTD_FRAME = b"\x28\xb5\x2f\xfd"
#: Per-frame input cap: a fatbin entry is far below this; the cap bounds a
#: false-positive scan hit's cost.
_ZSTD_FRAME_CAP = 16 << 20
#: Hard cap on inflated frames per file (runaway guard; real wheels: 104).
_ZSTD_FRAMES_MAX = 512


def _scan_notes(data: bytes) -> Set[int]:
    archs: Set[int] = set()
    for m in re.finditer(re.escape(_CUINFO), data):
        off = m.start() + _CUINFO_SM_OFF
        if off + 2 <= len(data):
            archs.add(int(struct.unpack_from("<H", data, off)[0]))
    return archs


def _inflate_zstd_frames(data: bytes) -> List[bytes]:
    """Inflate the ZSTD frames embedded in ``data`` (empty list when no
    decoder is available -- the caller reports UNKNOWN, never a guess)."""
    starts = [m.start() for m in re.finditer(re.escape(_ZSTD_FRAME), data)][:_ZSTD_FRAMES_MAX]
    if not starts:
        return []
    out: List[bytes] = []
    try:
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        for s in starts:
            try:
                out.append(dctx.decompressobj().decompress(
                    data[s:s + _ZSTD_FRAME_CAP], max_output_size=_ZSTD_FRAME_CAP))
            except Exception:  # noqa: BLE001 -- a garbage "frame" is not an arch
                continue
        return out
    except ImportError:
        pass
    # No python decoder: the zstd CLI, one subprocess per frame (this path
    # runs lazily, once per process, only where the answer is actually asked).
    for s in starts:
        try:
            p = subprocess.run(["zstd", "-d", "-c"], input=data[s:s + _ZSTD_FRAME_CAP],
                               capture_output=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return out
        # The zstd CLI prints the FIRST frame cleanly and only then complains
        # about the following bytes ("unsupported format", rc!=0) -- the
        # decoded prefix is exactly what the scan needs, so a nonzero rc
        # discards nothing that came out.
        if p.stdout:
            out.append(p.stdout[:_ZSTD_FRAME_CAP])
    return out


def scan_sass_archs(data: bytes) -> Set[int]:
    """The SM numbers whose cubin notes appear in ``data`` (raw cubins and,
    when a decoder is available, ZSTD-framed fatbin entries)."""
    archs = _scan_notes(data)
    for frame in _inflate_zstd_frames(data):
        archs |= _scan_notes(frame)
    return archs


def sgl_kernel_so_paths(cc: Optional[Tuple[int, int]] = None) -> List[str]:
    """The installed sgl_kernel's extension files that the loader would use
    for ``cc``: the package root (flashmla/infllm/...) plus the variant dir
    chosen exactly like ``sgl_kernel.load_utils`` -- ``sm90/`` for a CC-90
    device, ``sm100/`` for everything else (including no GPU)."""
    import importlib.util

    try:
        spec = importlib.util.find_spec("sgl_kernel")
    except (ImportError, ValueError):
        return []
    if spec is None or not spec.submodule_search_locations:
        return []
    root = Path(next(iter(spec.submodule_search_locations)))
    variant = "sm90" if (cc is not None and cc[0] == 9) else "sm100"
    paths: List[str] = []
    for d in (root, root / variant):
        try:
            paths.extend(sorted(str(p) for p in d.glob("*.so")))
        except OSError:
            continue
    return paths


@functools.lru_cache(maxsize=None)
def sgl_kernel_sass_archs(cc: Optional[Tuple[int, int]] = None) -> Optional[frozenset]:
    """The SM set the wheel carries for ``cc``'s loader variant, or ``None``
    when it cannot be determined (package not installed, no files readable,
    no zstd decoder and no parseable raw cubins)."""
    paths = sgl_kernel_so_paths(cc)
    if not paths:
        return None
    archs: Set[int] = set()
    read_any = False
    for p in paths:
        try:
            with open(p, "rb") as fh:
                data = fh.read()
        except OSError:
            continue
        read_any = True
        try:
            found = scan_sass_archs(data)
        except Exception:  # noqa: BLE001 -- a probe never breaks a boot
            logger.warning("wheel_sass: scan of %s failed", p, exc_info=True)
            continue
        archs |= found
    if not read_any or not archs:
        return None
    return frozenset(archs)


def wheel_carries_sass(cc: Tuple[int, int]) -> Optional[bool]:
    """True: the wheel carries ``code=sm_<cc>`` SASS. False: it demonstrably
    does not. None: unknown -- callers MUST treat unknown as not-carries
    (the conservative route: a fallback kernel beats a device trap)."""
    archs = sgl_kernel_sass_archs(cc)
    if archs is None:
        return None
    return (int(cc[0]) * 10 + int(cc[1])) in archs
