#!/usr/bin/env python3
"""#306 step 1 -- whole-image measurement for the disk-image asset class.

A 16 MiB chunk sample is the right unit for an expert tensor (that is the
granule a cold-tier miss moves) but the WRONG unit for a hibernate image: the
image is written and read as a whole, so its ratio is
``total bytes / total compressed bytes``, not the median of chunk ratios. The
two differ sharply here because the image is not homogeneous -- part of it is
an all-zero region, and a chunk sample either hits it or does not.

This script streams the whole file, so it also separates the two mechanisms
that a "compress the hibernate image" feature would conflate:

* **hole fraction** -- the share of 4 KiB pages that are entirely one byte
  value. Removing those needs no codec at all: a sparse write, or
  ``fallocate(FALLOC_FL_PUNCH_HOLE)``, gets exactly that factor for free.
* **residual codec gain** -- what zstd finds in the pages that are NOT holes.
  This is the part that actually costs CPU on every read.

Reporting them separately matters because the cheap mechanism and the
expensive one look identical in a single blended ratio.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import zstandard as zstd

CHUNK = 8 << 20
PAGE = 4096


def scan(path: Path) -> dict:
    """One pass: hole fraction at page granularity, plus a byte histogram."""
    size = path.stat().st_size
    hole_pages = 0
    total_pages = 0
    hist = np.zeros(256, dtype=np.int64)
    with open(path, "rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            a = np.frombuffer(buf, dtype=np.uint8)
            hist += np.bincount(a, minlength=256)
            n = a.size // PAGE
            if n:
                pages = a[: n * PAGE].reshape(n, PAGE)
                mn = pages.min(axis=1)
                mx = pages.max(axis=1)
                hole_pages += int(((mn == mx) & (mn == 0)).sum())
                total_pages += n
    p = hist[hist > 0] / hist.sum()
    return {
        "size": size,
        "hole_pages": hole_pages,
        "total_pages": total_pages,
        "hole_fraction": hole_pages / total_pages,
        "sparse_only_ratio": 1.0 / (1.0 - hole_pages / total_pages),
        "h0_bits": float(-(p * np.log2(p)).sum()),
    }


def stream_codec(path: Path, level: int, threads: int) -> dict:
    """Whole-file compress, then decompress, both streaming."""
    size = path.stat().st_size
    cctx = zstd.ZstdCompressor(level=level, threads=threads)
    sink = _Counter()
    t0 = time.perf_counter()
    with open(path, "rb") as f:
        with cctx.stream_writer(sink) as w:
            while True:
                buf = f.read(CHUNK)
                if not buf:
                    break
                w.write(buf)
    comp_s = time.perf_counter() - t0
    comp_out = sink.n

    # Decompress from a re-compressed temp stream held on disk, so the
    # decompress timing is not distorted by holding 7 GB of frames in RAM.
    tmp = path.parent / ".306_tmp.zst"
    try:
        with open(path, "rb") as f, open(tmp, "wb") as g:
            with zstd.ZstdCompressor(level=level, threads=threads).stream_writer(g) as w:
                while True:
                    buf = f.read(CHUNK)
                    if not buf:
                        break
                    w.write(buf)
        sink = _Counter()
        t0 = time.perf_counter()
        with open(tmp, "rb") as g:
            zstd.ZstdDecompressor().copy_stream(g, sink, read_size=CHUNK, write_size=CHUNK)
        dec_s = time.perf_counter() - t0
        assert sink.n == size, (sink.n, size)
    finally:
        tmp.unlink(missing_ok=True)

    return {
        "level": level,
        "threads": threads,
        "comp_bytes": comp_out,
        "ratio": size / comp_out,
        "comp_mbs": size / comp_s / 1e6,
        "decomp_mbs": size / dec_s / 1e6,
    }


class _Counter:
    """Write sink that only counts, so nothing large is retained."""

    def __init__(self) -> None:
        self.n = 0

    def write(self, b) -> int:
        self.n += len(b)
        return len(b)

    def flush(self) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image",
        default="/spinning/gpu-battery-results/2026-07-31_333_m1_registry"
        "/hibernate-qwen35-9b/rank0_GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d.pt",
    )
    ap.add_argument("--out", default="/spinning/wt-306-ratio/.probe-data/image_whole.json")
    args = ap.parse_args()
    path = Path(args.image)

    rec = {"path": str(path), **scan(path)}
    print(json.dumps(rec, indent=2), flush=True)
    rec["arms"] = []
    for level, threads in ((3, 0), (3, 16), (19, 16)):
        arm = stream_codec(path, level, threads)
        rec["arms"].append(arm)
        print(json.dumps(arm), flush=True)
    Path(args.out).write_text(json.dumps(rec, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
