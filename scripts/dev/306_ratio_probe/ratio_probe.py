#!/usr/bin/env python3
"""#306 step 1 -- lossless compression ratio + throughput matrix.

For every sample produced by ``sample_extract.py``, runs
``codec x byte-layout`` and records:

* ``ratio``            = uncompressed bytes / compressed bytes  (>1 is a win)
* ``comp_mbs``         = compression rate in UNCOMPRESSED MB/s
* ``decomp_mbs``       = decompression rate in UNCOMPRESSED MB/s, INCLUDING the
                         inverse byte-permutation for the non-raw layouts. This
                         is the load-bearing number: a slow-link cell only wins
                         if bytes can be turned back into the original layout
                         faster than the link would have delivered them raw.
* ``decomp_codec_mbs`` = the same without the inverse permutation, so the
                         permutation's share is visible rather than folded in.

A byte-layout is a pure PERMUTATION of the sample's bytes, compressed as a
single frame (not one frame per plane -- separate frames would forfeit the
shared window, which is the whole point of grouping like-entropy bytes).
Every (sample, codec, layout) triple is verified byte-identical after the
round trip; a mismatch aborts the run.

Rates are MB/s = 1e6 bytes/s (not MiB/s), to match the GB/s link figures the
verdicts are compared against.
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocks import (  # noqa: E402
    LAYOUTS,
    nibble_join,
    nibble_split,
    plane_join,
    plane_split,
    stride_join,
    stride_split,
)

MT_THREADS = 16
CHUNK_BYTES = 4 << 20


# --------------------------------------------------------------------------
# byte layouts: (forward permutation, inverse permutation)
# --------------------------------------------------------------------------
def make_layouts(sample: dict) -> dict[str, tuple]:
    n = sample["n_bytes"]
    out: dict[str, tuple] = {"raw": (lambda b: b, lambda b: b)}
    qname = sample.get("block_layout")
    if qname and qname in LAYOUTS:
        lay = LAYOUTS[qname]
        out["plane"] = (
            lambda b, lay=lay: b"".join(p for _, p in plane_split(b, lay)),
            lambda b, lay=lay, n=n: plane_join(_resplit_plane(b, lay, n), lay, n),
        )
        k = lay.block_bytes
        out[f"stride{k}"] = (
            lambda b, k=k: b"".join(p for _, p in stride_split(b, k)),
            lambda b, k=k, n=n: stride_join(_resplit_even(b, k), k, n),
        )
    else:
        for k in (2, 4):
            out[f"stride{k}"] = (
                lambda b, k=k: b"".join(p for _, p in stride_split(b, k)),
                lambda b, k=k, n=n: stride_join(_resplit_even(b, k), k, n),
            )
    if n % 2 == 0:  # the nibble pack works on byte pairs
        out["nibble"] = (
            lambda b: b"".join(p for _, p in nibble_split(b)),
            lambda b, n=n: nibble_join(_resplit_even(b, 2), n),
        )
    return out


def _resplit_plane(buf: bytes, lay, n: int) -> list[tuple[str, bytes]]:
    nblocks = n // lay.block_bytes
    a = np.frombuffer(buf, dtype=np.uint8)
    out, pos = [], 0
    for fname, start, end in lay.fields:
        w = (end - start) * nblocks
        out.append((fname, a[pos : pos + w].tobytes()))
        pos += w
    return out


def _resplit_even(buf: bytes, k: int) -> list[tuple[str, bytes]]:
    a = np.frombuffer(buf, dtype=np.uint8)
    w = a.size // k
    return [(f"s{j}", a[j * w : (j + 1) * w].tobytes()) for j in range(k)]


# --------------------------------------------------------------------------
# codecs
# --------------------------------------------------------------------------
def _zstd_c(level: int, threads: int = 0):
    return lambda b: zstd.ZstdCompressor(level=level, threads=threads).compress(b)


def _zstd_d(b: bytes, n: int) -> bytes:
    return zstd.ZstdDecompressor().decompress(b, max_output_size=n)


CODECS = {
    "zstd-3": (_zstd_c(3), _zstd_d, True),
    "zstd-19": (_zstd_c(19), _zstd_d, True),
    "zlib-6": (lambda b: zlib.compress(b, 6), lambda b, n: zlib.decompress(b), True),
    "lzma-fast": (
        lambda b: lzma.compress(b, preset=1),
        lambda b, n: lzma.decompress(b),
        True,
    ),
}

# Multi-thread arms run on the raw layout only: threading is a property of the
# codec driver, not of the byte permutation.
MT_CODECS = {
    f"zstd-3-mt{MT_THREADS}": _zstd_c(3, MT_THREADS),
    f"zstd-19-mt{MT_THREADS}": _zstd_c(19, MT_THREADS),
}


def timed(fn, min_seconds: float, max_reps: int):
    """Best-of timing: repeat until min_seconds of wall time or max_reps."""
    best = float("inf")
    total = 0.0
    reps = 0
    result = None
    while reps == 0 or (reps < max_reps and total < min_seconds):
        t0 = time.perf_counter()
        result = fn()
        dt = time.perf_counter() - t0
        best = min(best, dt)
        total += dt
        reps += 1
    return best, reps, result


def chunked_parallel_decompress(buf: bytes, level: int, workers: int) -> dict:
    """Independent-frame chunking: the shape a cold tier actually stores.

    A single zstd frame decompresses on one core no matter how many are free.
    Storing the asset as independent CHUNK_BYTES frames is the only way to get
    aggregate decompress bandwidth above one core, at a small ratio cost from
    the reset window. Both effects are measured here.
    """
    c = zstd.ZstdCompressor(level=level)
    chunks = [buf[i : i + CHUNK_BYTES] for i in range(0, len(buf), CHUNK_BYTES)]
    frames = [c.compress(ch) for ch in chunks]
    comp = sum(len(f) for f in frames)
    sizes = [len(ch) for ch in chunks]

    def run():
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda a: _zstd_d(a[0], a[1]), zip(frames, sizes, strict=True)))

    best, _, out = timed(run, 0.3, 5)
    if b"".join(out) != buf:
        raise AssertionError("chunked parallel round trip mismatch")
    return {
        "chunk_bytes": CHUNK_BYTES,
        "n_chunks": len(chunks),
        "workers": workers,
        "ratio": len(buf) / comp,
        "decomp_mbs": len(buf) / best / 1e6,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/spinning/wt-306-ratio/.probe-data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--only-class", default=None)
    args = ap.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "samples.json").read_text())
    out_path = Path(args.out) if args.out else data / "results.jsonl"
    fout = open(out_path, "w")

    samples = manifest["samples"]
    if args.only_class:
        samples = [s for s in samples if s["asset_class"] == args.only_class]

    t_start = time.time()
    for idx, s in enumerate(samples):
        buf = Path(s["file"]).read_bytes()
        assert len(buf) == s["n_bytes"]
        layouts = make_layouts(s)
        for lname, (fwd, inv) in layouts.items():
            permuted = fwd(buf)
            assert len(permuted) == len(buf), lname
            if inv(permuted) != buf:
                raise AssertionError(f"{s['sample_id']}/{lname}: permutation is not invertible")
            # Cost of the inverse permutation on its own.
            perm_t, _, _ = timed(lambda p=permuted: inv(p), 0.2, 5)
            for cname, (comp, decomp, _) in CODECS.items():
                slow = cname in ("zstd-19", "lzma-fast")
                ct, creps, cbuf = timed(lambda p=permuted: comp(p), 0.0 if slow else 0.3, 1 if slow else 5)
                dt, _, dbuf = timed(lambda c=cbuf: decomp(c, len(buf)), 0.2, 5)
                if lname == "raw":
                    restored = dbuf
                    dt_total = dt
                else:
                    restored = inv(dbuf)
                    dt_total = dt + perm_t
                if restored != buf:
                    raise AssertionError(f"LOSSY: {s['sample_id']}/{lname}/{cname}")
                rec = {
                    "sample_id": s["sample_id"],
                    "asset_class": s["asset_class"],
                    "ggml_type": s["ggml_type"],
                    "n_bytes": len(buf),
                    "layout": lname,
                    "codec": cname,
                    "comp_bytes": len(cbuf),
                    "ratio": len(buf) / len(cbuf),
                    "comp_mbs": len(buf) / ct / 1e6,
                    "comp_reps": creps,
                    "decomp_mbs": len(buf) / dt_total / 1e6,
                    "decomp_codec_mbs": len(buf) / dt / 1e6,
                    "inv_perm_mbs": len(buf) / perm_t / 1e6 if perm_t else None,
                    "verified_lossless": True,
                }
                fout.write(json.dumps(rec) + "\n")
        # multi-thread + chunked arms, raw layout only
        for cname, comp in MT_CODECS.items():
            ct, creps, cbuf = timed(lambda: comp(buf), 0.0, 1)
            dt, _, dbuf = timed(lambda c=cbuf: _zstd_d(c, len(buf)), 0.2, 5)
            if dbuf != buf:
                raise AssertionError(f"LOSSY: {s['sample_id']}/raw/{cname}")
            fout.write(
                json.dumps(
                    {
                        "sample_id": s["sample_id"],
                        "asset_class": s["asset_class"],
                        "ggml_type": s["ggml_type"],
                        "n_bytes": len(buf),
                        "layout": "raw",
                        "codec": cname,
                        "comp_bytes": len(cbuf),
                        "ratio": len(buf) / len(cbuf),
                        "comp_mbs": len(buf) / ct / 1e6,
                        "comp_reps": creps,
                        "decomp_mbs": len(buf) / dt / 1e6,
                        "decomp_codec_mbs": len(buf) / dt / 1e6,
                        "inv_perm_mbs": None,
                        "verified_lossless": True,
                    }
                )
                + "\n"
            )
        ch = chunked_parallel_decompress(buf, 3, 8)
        fout.write(
            json.dumps(
                {
                    "sample_id": s["sample_id"],
                    "asset_class": s["asset_class"],
                    "ggml_type": s["ggml_type"],
                    "n_bytes": len(buf),
                    "layout": "raw",
                    "codec": "zstd-3-chunk4M-x8",
                    "comp_bytes": int(len(buf) / ch["ratio"]),
                    "ratio": ch["ratio"],
                    "comp_mbs": None,
                    "comp_reps": None,
                    "decomp_mbs": ch["decomp_mbs"],
                    "decomp_codec_mbs": ch["decomp_mbs"],
                    "inv_perm_mbs": None,
                    "verified_lossless": True,
                    "n_chunks": ch["n_chunks"],
                    "workers": ch["workers"],
                }
            )
            + "\n"
        )
        fout.flush()
        el = time.time() - t_start
        print(
            f"[{idx + 1}/{len(samples)}] {s['sample_id']:34s} "
            f"{el / 60:5.1f} min elapsed, eta {(el / (idx + 1) * (len(samples) - idx - 1)) / 60:5.1f} min",
            flush=True,
        )
    fout.close()
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "99")
    raise SystemExit(main())
