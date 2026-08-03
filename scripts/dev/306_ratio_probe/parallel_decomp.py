#!/usr/bin/env python3
"""#306 step 1 -- how much CPU a compressed cold tier would cost.

The main sweep fixes the parallel-decompress arm at 8 workers. That is an
arbitrary point, and it decides the verdict: the serial break-even
``r_min = D/(D-L)`` falls as ``D`` rises, so "does compression pay" is really
"how many cores are you willing to spend on decompressing". This script sweeps
the worker count for the classes whose ratio cleared the 1.08 kill criterion
and reports, at every point, both speedups and the cores it took.

Independent 4 MiB frames, `zstd -3` -- the shape a cold tier would actually
store, since a single frame decompresses on one core however many are free.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import zstandard as zstd

CHUNK = 4 << 20
LINKS = [
    ("T3 NVMe", 1.8e9),
    ("T4 40G sockets", 2.07e9),
    ("T4 40G staged RDMA", 2.83e9),
    ("PCIe H2D x4", 6.4e9),
]


def run_sample(buf: bytes, workers_list: list[int], level: int = 3) -> dict:
    c = zstd.ZstdCompressor(level=level)
    chunks = [buf[i : i + CHUNK] for i in range(0, len(buf), CHUNK)]
    frames = [c.compress(ch) for ch in chunks]
    sizes = [len(ch) for ch in chunks]
    ratio = len(buf) / sum(len(f) for f in frames)
    out = {"ratio": ratio, "n_chunks": len(chunks), "rates": {}}

    # A ZstdDecompressor is NOT thread-safe -- one per call, as in ratio_probe.
    def _dec(a):
        return zstd.ZstdDecompressor().decompress(a[0], max_output_size=a[1])

    for w in workers_list:
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            with ThreadPoolExecutor(max_workers=w) as ex:
                res = list(ex.map(_dec, zip(frames, sizes, strict=True)))
            dt = time.perf_counter() - t0
            best = min(best, dt)
        if b"".join(res) != buf:
            raise AssertionError("round trip mismatch")
        out["rates"][w] = len(buf) / best / 1e6
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/spinning/wt-306-ratio/.probe-data")
    ap.add_argument(
        "--classes",
        default="qwen27b_fp8,qwen27b_int8,dsv4f_ud_q3kxl_mxfp4,dsv4f_ud_q3kxl_iq3_xxs",
        help="comma-separated; the last one is a DEAD control",
    )
    ap.add_argument("--workers", default="1,2,4,8,16,32")
    ap.add_argument("--per-class", type=int, default=4)
    args = ap.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "samples.json").read_text())
    wanted = args.classes.split(",")
    workers = [int(x) for x in args.workers.split(",")]

    by_class = defaultdict(list)
    for s in manifest["samples"]:
        if s["asset_class"] in wanted:
            by_class[s["asset_class"]].append(s)

    results = {}
    for cls in wanted:
        recs = [run_sample(Path(s["file"]).read_bytes(), workers)
                for s in by_class[cls][: args.per_class]]
        ratio = st.median([r["ratio"] for r in recs])
        rates = {w: st.median([r["rates"][w] for r in recs]) for w in workers}
        results[cls] = {"ratio": ratio, "rates_mbs": rates, "n": len(recs)}
        print(f"{cls:28s} ratio {ratio:.4f}  " +
              "  ".join(f"{w}T={rates[w]:.0f}" for w in workers), flush=True)

    (data / "parallel_decomp.json").write_text(json.dumps(results, indent=2))

    print("\n| asset class | ratio | workers | D (MB/s) | " +
          " | ".join(f"serial @ {n}" for n, _ in LINKS) + " |")
    print("|---" * (4 + len(LINKS)) + "|")
    for cls, r in results.items():
        for w in workers:
            d = r["rates_mbs"][w] * 1e6
            cells = []
            for _n, lbps in LINKS:
                s = 1.0 / (1.0 / r["ratio"] + lbps / d)
                cells.append(f"{s:.3f}x")
            print(f"| `{cls}` | {r['ratio']:.4f} | {w} | {d / 1e6:.0f} | " +
                  " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
