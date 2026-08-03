#!/usr/bin/env python3
"""#306 step 1 -- the entropy ceiling, and a "did you try hard enough" arm.

Two things the ratio matrix on its own cannot say:

1. **The ceiling.** A byte-oriented entropy coder cannot beat the payload's
   order-0 byte entropy. ``ceil0 = 8 / H0`` is therefore an upper bound on any
   memoryless coder, and ``ceil0`` computed per PLANE (weighted by plane size)
   is the upper bound on the byte-plane-split family specifically. If the
   measured ratio already sits at the ceiling, no further codec tuning exists
   to find -- the payload is out of redundancy, not the coder out of effort.
   The bound is one-sided: a coder can still beat ``ceil0`` by modelling
   ORDER (repeated substrings), which is exactly what LZ does, so a measured
   ratio above ceil0 is legitimate and reported as such.

2. **The effort ceiling.** ``zstd --ultra -22 --long=27`` and ``xz -9e`` are
   the strongest generally available settings. Running them on a subset shows
   whether the -3/-19 verdict is an artefact of insufficient effort.

Sampled at ``--per-class`` samples per class (default 2) because -22/-9e are
minutes-per-sample.
"""

from __future__ import annotations

import argparse
import json
import lzma
import statistics as st
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import zstandard as zstd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocks import LAYOUTS, plane_split, stride_split  # noqa: E402


def h0(buf: bytes) -> float:
    """Order-0 byte entropy in bits/byte."""
    counts = np.bincount(np.frombuffer(buf, dtype=np.uint8), minlength=256).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def plane_ceiling(buf: bytes, planes: list[tuple[str, bytes]]) -> float:
    """Size-weighted order-0 ceiling of a plane decomposition."""
    bits = sum(len(p) * h0(p) for _, p in planes)
    return len(buf) * 8.0 / bits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/spinning/wt-306-ratio/.probe-data")
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = Path(args.data)
    manifest = json.loads((data / "samples.json").read_text())
    out_path = Path(args.out) if args.out else data / "ceiling.jsonl"

    by_class = defaultdict(list)
    for s in manifest["samples"]:
        by_class[s["asset_class"]].append(s)

    with open(out_path, "w") as fout:
        for cls, ss in sorted(by_class.items()):
            for s in ss[: args.per_class]:
                buf = Path(s["file"]).read_bytes()
                rec = {
                    "sample_id": s["sample_id"],
                    "asset_class": cls,
                    "n_bytes": len(buf),
                    "h0_raw_bits": h0(buf),
                    "ceil0_raw": 8.0 / h0(buf),
                }
                qname = s.get("block_layout")
                c19 = zstd.ZstdCompressor(level=19)
                if qname and qname in LAYOUTS:
                    lay = LAYOUTS[qname]
                    planes = plane_split(buf, lay)
                    rec["ceil0_plane"] = plane_ceiling(buf, planes)
                    rec["ceil0_stride"] = plane_ceiling(buf, stride_split(buf, lay.block_bytes))
                    # Where the (small) gain actually sits: per-field byte
                    # share, order-0 entropy, and a real zstd-19 ratio. A field
                    # that compresses 3x but is 1/17 of the payload contributes
                    # only share*(1 - 1/ratio) of total saving.
                    rec["planes"] = {
                        f: {
                            "share": len(p) / len(buf),
                            "h0": round(h0(p), 4),
                            "zstd19_ratio": len(p) / len(c19.compress(p)),
                        }
                        for (f, p) in planes
                    }
                else:
                    rec["ceil0_stride4"] = plane_ceiling(buf, stride_split(buf, 4))

                t0 = time.perf_counter()
                c = zstd.ZstdCompressor(
                    compression_params=zstd.ZstdCompressionParameters.from_level(
                        22, window_log=27, enable_ldm=1
                    ),
                ).compress(buf)
                rec["zstd22_long_ratio"] = len(buf) / len(c)
                rec["zstd22_long_comp_s"] = time.perf_counter() - t0

                t0 = time.perf_counter()
                c = lzma.compress(buf, preset=9 | lzma.PRESET_EXTREME)
                rec["xz9e_ratio"] = len(buf) / len(c)
                rec["xz9e_comp_s"] = time.perf_counter() - t0

                fout.write(json.dumps(rec) + "\n")
                fout.flush()
                print(
                    f"{s['sample_id']:34s} H0={rec['h0_raw_bits']:.4f} b/B  "
                    f"ceil0={rec['ceil0_raw']:.4f}  zstd22L={rec['zstd22_long_ratio']:.4f}  "
                    f"xz9e={rec['xz9e_ratio']:.4f}",
                    flush=True,
                )

    print("\n=== per class ===")
    rows = [json.loads(ln) for ln in out_path.read_text().splitlines()]
    per = defaultdict(list)
    for r in rows:
        per[r["asset_class"]].append(r)
    for cls, rs in sorted(per.items()):
        print(
            f"{cls:34s} H0 {st.median([r['h0_raw_bits'] for r in rs]):.4f}  "
            f"ceil0 {st.median([r['ceil0_raw'] for r in rs]):.4f}  "
            f"ceil0-plane {st.median([r.get('ceil0_plane') or r.get('ceil0_stride4') for r in rs]):.4f}  "
            f"zstd22L {st.median([r['zstd22_long_ratio'] for r in rs]):.4f}  "
            f"xz9e {st.median([r['xz9e_ratio'] for r in rs]):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
