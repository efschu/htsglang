#!/usr/bin/env python3
"""Does a quantised DFLASH drafter dequantize to its dense reference?

CPU-only, no GPU and no model instantiation: one tensor at a time, through the
very name map the loader uses (``model_loader/gguf_dflash.build_dflash_name_map``)
and the ``gguf`` package's reference dequant kernels, so peak RSS stays at the
largest single tensor.

Written for #290, where a drafter that loaded cleanly still accepted 1.005
tokens per round. This separated the two candidate halves in one run: the FILE
was a faithful 0.57% Q8_0 of the BF16 release, which moved the search off the
checkpoint and onto the loader -- where the bug was, the runtime having built a
dense skeleton and dropped every packed tensor.

A Q8_0 round trip is ~0.6% relative error. An order of magnitude above that, or
any shape disagreement, is a load bug rather than quantization.

    python scripts/diag/q8_dflash_weight_diff.py \\
        --gguf .../Qwen3.6-27B-DFlash-Q8_0.gguf \\
        --safetensors .../qwen3.6-27b-dflash/model.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_MODEL_ROOT = "/spinning/llm_stuff/club-3090/models-cache"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gguf",
        default=f"{_MODEL_ROOT}/qwen3.6-27b-dflash-gguf/Qwen3.6-27B-DFlash-Q8_0.gguf",
    )
    ap.add_argument(
        "--safetensors",
        default=f"{_MODEL_ROOT}/qwen3.6-27b-dflash/model.safetensors",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="config.json of the GGUF drafter (default: beside the .gguf)",
    )
    ap.add_argument("--rel-tol", type=float, default=0.05)
    args = ap.parse_args()

    import gguf
    from safetensors import safe_open

    from sglang.srt.model_loader.gguf_dflash import (
        audit_dflash_name_map,
        build_dflash_name_map,
    )

    config_path = args.config or os.path.join(os.path.dirname(args.gguf), "config.json")
    with open(config_path) as f:
        num_layers = int(json.load(f)["num_hidden_layers"])

    class _Cfg:
        num_hidden_layers = num_layers

    name_map = build_dflash_name_map(_Cfg())

    reader = gguf.GGUFReader(args.gguf, "r")
    by_name = {t.name: t for t in reader.tensors}
    audit = audit_dflash_name_map(name_map, by_name.keys())
    print(f"name map: {audit if audit else f'exact ({len(name_map)}/{len(by_name)})'}")

    results = []
    with safe_open(args.safetensors, "pt") as st:
        st_keys = set(st.keys())
        for gguf_name in sorted(name_map):
            hf_name = name_map[gguf_name]
            if hf_name not in st_keys:
                print(f"MISSING in the reference: {hf_name}")
                continue
            tensor = by_name[gguf_name]
            # GGUF ne is reversed w.r.t. torch's [out, in].
            shape = tuple(int(x) for x in reversed(tensor.shape))
            deq = gguf.quants.dequantize(tensor.data, tensor.tensor_type).reshape(shape)
            ref = st.get_tensor(hf_name).float().numpy()
            if deq.shape != ref.shape:
                note = "TRANSPOSED" if deq.shape == ref.shape[::-1] else "SHAPE"
                print(f"{note} {gguf_name} -> {hf_name}: {deq.shape} vs {ref.shape}")
                results.append((float("inf"), gguf_name, hf_name, note))
                continue
            denom = float(np.abs(ref).mean()) or 1.0
            rel = float(np.abs(deq - ref).mean()) / denom
            results.append((rel, gguf_name, hf_name, "ok"))
            del deq, ref

    results.sort(reverse=True)
    print(f"\n{'rel_mean_err':>13}  tensor")
    for rel, gname, hname, _ in results[:8]:
        print(f"{rel:13.6f}  {gname} -> {hname}")
    print(f"{'...':>13}")
    for rel, gname, hname, _ in results[-3:]:
        print(f"{rel:13.6f}  {gname} -> {hname}")

    bad = [r for r in results if r[0] > args.rel_tol or r[3] != "ok"]
    print(f"\ntensors above rel-tol {args.rel_tol} or mis-shaped: {len(bad)}")
    for rel, gname, hname, note in bad:
        print(f"  {rel:.6f} {note} {gname} -> {hname}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
