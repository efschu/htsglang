#!/usr/bin/env python3
"""#306 step 1 -- extract REAL cold-tier asset samples for the ratio probe.

Seed-fixed (``--seed``, default 306) so the sample set is reproducible. Nothing
is synthesised: every sample is a byte-for-byte slice of a file that exists on
this box, and each one records its own provenance (path, tensor, offset).

Asset classes:

* ``dsv4f_<quant>``   -- routed-expert tensors of DeepSeek-V4-Flash GGUF
                        (UD-IQ3_XXS and UD-Q3_K_XL), RAW QUANTIZED BYTES read
                        at the tensor's file offset. Never dequantised: the
                        cold tier stores the quantised bytes.
* ``qwen_moe_<quant>`` -- routed-expert tensors of Qwen3.6-35B-A3B UD-Q3_K_M,
                        included because it is the only real Q3_K / IQ4_XS /
                        Q6_K expert sample on this box.
* ``qwen27b_int8``   -- I8 weight tensors of Qwen3.6-27B-INT8-W8A8.
* ``qwen27b_fp8``    -- F8_E4M3 weight tensors of Qwen3.6-27B-FP8.
* ``hibernate_img``  -- chunks of the #89 hibernate image written by the
                        2026-07-31 battery (Qwen3.5-9B Q4_K_M, torch ZIP_STORED
                        container, so the chunks are raw tensor bytes plus zip
                        framing -- i.e. exactly what the disk tier would move).

Usage:
    python sample_extract.py --out /spinning/wt-306-ratio/.probe-data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from blocks import LAYOUTS  # noqa: E402

MODELS = Path("/spinning/llm_stuff/club-3090/models-cache")
HIBERNATE = Path(
    "/spinning/gpu-battery-results/2026-07-31_333_m1_registry/hibernate-qwen35-9b"
    "/rank0_GPU-31d7ef41-f574-4d0e-21ad-e773fd938f6d.pt"
)

GGUF_SOURCES = [
    (
        "dsv4f_ud_iq3xxs",
        MODELS / "DeepSeek-V4-Flash-0731-GGUF/UD-IQ3_XXS",
        "DeepSeek-V4-Flash UD-IQ3_XXS routed experts",
    ),
    (
        "dsv4f_ud_q3kxl",
        MODELS / "DeepSeek-V4-Flash-0731-GGUF/UD-Q3_K_XL",
        "DeepSeek-V4-Flash UD-Q3_K_XL routed experts",
    ),
    (
        "qwen35ba3b_ud_q3km",
        MODELS / "Qwen3.6-35B-A3B-MTP-UD-Q3_K_M-GGUF",
        "Qwen3.6-35B-A3B UD-Q3_K_M routed experts",
    ),
]

SAFETENSOR_SOURCES = [
    ("qwen27b_int8", MODELS / "Qwen3.6-27B-INT8-W8A8", "I8", "Qwen3.6-27B INT8-W8A8"),
    ("qwen27b_fp8", MODELS / "Qwen3.6-27B-FP8", "F8_E4M3", "Qwen3.6-27B FP8 (E4M3)"),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def read_slice(path: Path, offset: int, n: int) -> bytes:
    with open(path, "rb") as f:
        f.seek(offset)
        buf = f.read(n)
    if len(buf) != n:
        raise OSError(f"short read at {path}:{offset} ({len(buf)} of {n})")
    return buf


def gguf_samples(rng: random.Random, target_bytes: int, per_type: int) -> list[dict]:
    import gguf

    out: list[dict] = []
    for cls_prefix, directory, human in GGUF_SOURCES:
        by_type: dict[str, list[tuple[Path, object]]] = {}
        for path in sorted(directory.glob("*.gguf")):
            reader = gguf.GGUFReader(str(path), "r")
            for tensor in reader.tensors:
                if "_exps." not in tensor.name:
                    continue
                by_type.setdefault(tensor.tensor_type.name, []).append(
                    (
                        path,
                        {
                            "name": tensor.name,
                            "type": tensor.tensor_type.name,
                            "shape": [int(x) for x in tensor.shape],
                            "n_bytes": int(tensor.n_bytes),
                            "file_offset": int(reader.data_offset + tensor.data_offset),
                        },
                    )
                )
            del reader
        for qtype, entries in sorted(by_type.items()):
            layout = LAYOUTS.get(qtype)
            if layout is None:
                print(f"  SKIP {cls_prefix}/{qtype}: no block layout transcribed", file=sys.stderr)
                continue
            block = layout.block_bytes
            n = (target_bytes // block) * block
            entries = [e for e in entries if e[1]["n_bytes"] >= n]
            if not entries:
                continue
            # Draw per_type samples. When a quant type has fewer than per_type
            # tensors in the model (the UD mixes leave some types with 1-3),
            # tensors are reused at independent random offsets rather than
            # shrinking the class below the probe's >= 8 contract.
            order = rng.sample(entries, len(entries))
            picks = [order[i % len(order)] for i in range(per_type)]
            for path, meta in picks:
                # Block-aligned offset inside the tensor, uniformly drawn.
                max_block = (meta["n_bytes"] - n) // block
                blk = rng.randint(0, max_block)
                off_in_tensor = blk * block
                out.append(
                    {
                        "asset_class": f"{cls_prefix}_{qtype.lower()}",
                        "human": f"{human} [{qtype}]",
                        "source_path": str(path),
                        "tensor": meta["name"],
                        "ggml_type": qtype,
                        "tensor_shape": meta["shape"],
                        "tensor_bytes": meta["n_bytes"],
                        "offset_in_tensor": off_in_tensor,
                        "file_offset": meta["file_offset"] + off_in_tensor,
                        "n_bytes": n,
                        "elem_bytes": 1,
                        "block_layout": qtype,
                    }
                )
    return out


def safetensors_index(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return header, 8 + n


def safetensor_samples(rng: random.Random, target_bytes: int, count: int) -> list[dict]:
    out: list[dict] = []
    for cls, directory, dtype, human in SAFETENSOR_SOURCES:
        cands: list[dict] = []
        for path in sorted(directory.glob("*.safetensors")):
            header, data_off = safetensors_index(path)
            for name, meta in header.items():
                if name == "__metadata__" or meta.get("dtype") != dtype:
                    continue
                start, end = meta["data_offsets"]
                if end - start < target_bytes:
                    continue
                cands.append(
                    {
                        "path": path,
                        "name": name,
                        "shape": meta["shape"],
                        "abs_start": data_off + start,
                        "n": end - start,
                    }
                )
        if not cands:
            print(f"  SKIP {cls}: no {dtype} tensor >= {target_bytes} B", file=sys.stderr)
            continue
        for c in rng.sample(cands, min(count, len(cands))):
            # 4096-byte-aligned offset keeps the stride arms well defined.
            max_off = (c["n"] - target_bytes) // 4096
            off = rng.randint(0, max_off) * 4096
            out.append(
                {
                    "asset_class": cls,
                    "human": human,
                    "source_path": str(c["path"]),
                    "tensor": c["name"],
                    "ggml_type": dtype,
                    "tensor_shape": c["shape"],
                    "tensor_bytes": c["n"],
                    "offset_in_tensor": off,
                    "file_offset": c["abs_start"] + off,
                    "n_bytes": target_bytes,
                    "elem_bytes": 1,
                    "block_layout": None,
                }
            )
    return out


def hibernate_samples(rng: random.Random, target_bytes: int, count: int) -> list[dict]:
    if not HIBERNATE.exists():
        print("  SAMPLE-ABSENT hibernate_img: no #89 image on this box", file=sys.stderr)
        return []
    size = HIBERNATE.stat().st_size
    out = []
    # Skip the first 64 MiB (zip directory / small tensors) so the chunks land
    # in the bulk weight region, which is what a disk tier actually moves.
    lo = 64 << 20
    hi = size - target_bytes
    for i in range(count):
        off = rng.randint(lo // 4096, hi // 4096) * 4096
        out.append(
            {
                "asset_class": "hibernate_img",
                "human": "#89 hibernate image (Qwen3.5-9B Q4_K_M, torch ZIP_STORED)",
                "source_path": str(HIBERNATE),
                "tensor": f"<image chunk {i}>",
                "ggml_type": "mixed-Q4_K/Q6_K/F32",
                "tensor_shape": None,
                "tensor_bytes": size,
                "offset_in_tensor": off,
                "file_offset": off,
                "n_bytes": target_bytes,
                "elem_bytes": 1,
                "block_layout": None,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/spinning/wt-306-ratio/.probe-data")
    ap.add_argument("--seed", type=int, default=306)
    ap.add_argument("--mib", type=int, default=16, help="target sample size in MiB (>= 8)")
    ap.add_argument("--per-type", type=int, default=8, help="samples per (source, quant type)")
    args = ap.parse_args()
    if args.mib < 8:
        ap.error("--mib must be >= 8 (probe contract)")

    outdir = Path(args.out)
    (outdir / "samples").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    target = args.mib << 20

    samples = []
    samples += gguf_samples(rng, target, args.per_type)
    samples += safetensor_samples(rng, target, args.per_type)
    samples += hibernate_samples(rng, target, args.per_type)

    manifest = []
    for i, s in enumerate(samples):
        sid = f"{s['asset_class']}_{i:03d}"
        dst = outdir / "samples" / f"{sid}.bin"
        buf = read_slice(Path(s["source_path"]), s["file_offset"], s["n_bytes"])
        dst.write_bytes(buf)
        s["sample_id"] = sid
        s["file"] = str(dst)
        s["sha256"] = hashlib.sha256(buf).hexdigest()
        manifest.append(s)
        print(f"  {sid:38s} {s['n_bytes'] >> 20:3d} MiB  {s['tensor']}")

    meta = {
        "seed": args.seed,
        "sample_mib": args.mib,
        "per_type": args.per_type,
        "host": os.uname().nodename,
        "n_samples": len(manifest),
        "samples": manifest,
    }
    (outdir / "samples.json").write_text(json.dumps(meta, indent=2))
    print(f"\n{len(manifest)} samples -> {outdir / 'samples.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
