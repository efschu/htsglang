#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#456 -- what does the sparse hibernate write actually buy, on THIS box?

#306 §J projected 1.1447x from hole removal on a real 6.68 GiB rank image, and
in the same table showed the `/spinning` ZFS dataset already returning 1.1735x
by itself. Those two numbers are not additive, and the projection was never
measured against a filesystem that folds zeros on its own. This script settles
it by writing the same synthetic image dense and sparse, on the real pool and
on tmpfs, and reporting BOTH axes:

* allocated bytes (``st_blocks`` x 512) -- what sparse adds ON TOP of whatever
  the filesystem already does to zeros;
* wall time including ``fsync`` -- the write-path win, which exists whether or
  not the filesystem would have folded the zeros anyway, because the skipped
  bytes never cross the syscall boundary at all.

Arms are interleaved (dense, sparse, dense, sparse, ...) and an A-vs-A floor is
taken first (dense vs dense) so a reported delta can be read against the noise
of this box rather than against zero.

Usage:
    python scripts/dev/456_sparse_write/measure_sparse_write.py \
        --gib 3 --reps 3 --dirs /spinning/tmp /dev/shm
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "python"))

from sglang.srt.model_loader.sparse_write import (  # noqa: E402
    PAGE_SIZE,
    SparseFileWriter,
    filesystem_supports_holes,
)

CHUNK = 8 << 20


def build_image(nbytes: int, hole_fraction: float, seed: int = 456) -> np.ndarray:
    """A synthetic image with the #306-measured zero-page share.

    The zeros are laid out as a handful of LARGE contiguous regions, because
    that is how they occur in a real image: `torch.save` parks whole
    pre-allocated buffers, so the holes are buffer-sized, not scattered pages.
    A scattered layout would flatter the run-length encoder and understate the
    number of syscalls a real image costs.
    """
    rng = np.random.default_rng(seed)
    npages = nbytes // PAGE_SIZE
    # High-entropy body: quantized weights compress badly (order-0 entropy of
    # the real image was 7.38 bits/byte), so random bytes are the right filler.
    a = rng.integers(0, 256, size=nbytes, dtype=np.uint8)
    want_holes = int(round(npages * hole_fraction))
    n_regions = 12
    per = want_holes // n_regions
    placed = 0
    starts = np.linspace(0, npages - per - 1, n_regions).astype(np.int64)
    for i, s in enumerate(starts):
        n = per if i < n_regions - 1 else want_holes - placed
        a[s * PAGE_SIZE : (s + n) * PAGE_SIZE] = 0
        placed += n
    return a


def measured_hole_fraction(a: np.ndarray) -> float:
    pages = a[: (a.size // PAGE_SIZE) * PAGE_SIZE].reshape(-1, PAGE_SIZE)
    return float((~pages.any(axis=1)).sum()) / pages.shape[0]


def write_dense(path: str, a: np.ndarray) -> float:
    mv = memoryview(a).cast("B")
    t0 = time.perf_counter()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        off = 0
        while off < mv.nbytes:
            off += os.write(fd, mv[off : off + CHUNK])
        os.fsync(fd)
    finally:
        os.close(fd)
    return time.perf_counter() - t0


def write_sparse(path: str, a: np.ndarray) -> float:
    mv = memoryview(a).cast("B")
    t0 = time.perf_counter()
    w = SparseFileWriter(path)
    try:
        off = 0
        while off < mv.nbytes:
            w.write(mv[off : off + CHUNK])
            off += CHUNK
        os.fsync(w._fd)  # noqa: SLF001 - measurement wants the durable cost
    finally:
        w.close()
    return time.perf_counter() - t0


def settled_allocated(path: str, *, tries: int = 12, wait: float = 3.0) -> int:
    """``st_blocks`` x 512, after it stops moving.

    ZFS accounts allocation at txg sync, not at ``fsync`` return, so a stat
    taken immediately after a write reports a partially-committed figure --
    the first run of this script read 0.30 GB for a file that settled at
    2.82 GB. Poll until two consecutive reads agree.
    """
    prev = -1
    for _ in range(tries):
        cur = os.stat(path).st_blocks * 512
        if cur == prev and cur > 0:
            return cur
        prev = cur
        time.sleep(wait)
    return prev


def run_dir(directory: str, a: np.ndarray, reps: int, settle: float = 1.0) -> dict:
    os.makedirs(directory, exist_ok=True)
    holes_ok = filesystem_supports_holes(directory)
    dense_p = os.path.join(directory, ".456_dense.bin")
    sparse_p = os.path.join(directory, ".456_sparse.bin")
    aa_p = os.path.join(directory, ".456_aa.bin")

    rec: dict = {
        "dir": directory,
        "st_blocks_reflects_holes": holes_ok,
        "apparent_bytes": int(a.nbytes),
        "dense_s": [],
        "sparse_s": [],
        "aa_s": [],
    }
    try:
        # Three interleaved arms: two IDENTICAL dense writes (the A-vs-A floor)
        # and the sparse one. The arm ORDER rotates every rep -- the first
        # version of this script kept a fixed order and the always-first arm
        # came out systematically slower (11 % A-vs-A floor between two
        # identical writes), which is a position effect, not noise.
        arms = [
            ("aa_s", lambda: write_dense(aa_p, a)),
            ("dense_s", lambda: write_dense(dense_p, a)),
            ("sparse_s", lambda: write_sparse(sparse_p, a)),
        ]
        for i in range(reps):
            for j in range(3):
                key, fn = arms[(i + j) % 3]
                # Drain first, THEN time. ZFS defers work to the transaction
                # group, so without this the previous arm's flush lands inside
                # the next arm's window and the ordering effect swamps the
                # signal (three untimed trials of the same dense write spread
                # 1.33-2.43 s).
                os.sync()
                time.sleep(settle)
                rec[key].append(fn())
        rec["dense_allocated"] = settled_allocated(dense_p)
        rec["sparse_allocated"] = settled_allocated(sparse_p)
        rec["bytes_identical"] = (
            open(dense_p, "rb").read() == open(sparse_p, "rb").read()
        )
    finally:
        for p in (dense_p, sparse_p, aa_p):
            if os.path.exists(p):
                os.unlink(p)

    med = statistics.median
    rec["aa_med_s"] = med(rec["aa_s"])
    rec["dense_med_s"] = med(rec["dense_s"])
    rec["sparse_med_s"] = med(rec["sparse_s"])
    rec["aa_floor_pct"] = (
        100.0 * abs(rec["dense_med_s"] - rec["aa_med_s"]) / rec["aa_med_s"]
    )
    rec["settle_s"] = settle
    rec["time_speedup"] = rec["dense_med_s"] / rec["sparse_med_s"]
    rec["alloc_ratio"] = (
        rec["dense_allocated"] / rec["sparse_allocated"]
        if rec["sparse_allocated"]
        else float("inf")
    )
    rec["fs_own_ratio"] = rec["apparent_bytes"] / rec["dense_allocated"]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gib", type=float, default=3.0)
    ap.add_argument("--hole-fraction", type=float, default=0.12643438728613668)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--dirs", nargs="+", default=[tempfile.gettempdir()])
    ap.add_argument("--settle", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    nbytes = int(args.gib * (1 << 30)) // PAGE_SIZE * PAGE_SIZE
    a = build_image(nbytes, args.hole_fraction)
    out = {
        "apparent_bytes": nbytes,
        "requested_hole_fraction": args.hole_fraction,
        "measured_hole_fraction": measured_hole_fraction(a),
        "reps": args.reps,
        "dirs": [],
    }
    for d in args.dirs:
        rec = run_dir(d, a, args.reps, args.settle)
        out["dirs"].append(rec)
        print(json.dumps(rec, indent=2), flush=True)
    print(json.dumps({k: v for k, v in out.items() if k != "dirs"}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
