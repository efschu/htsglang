# Pricing facts of an EXTERNAL DFlash draft checkpoint (Weg 2, #1017 weight
# line / #593 VRAM ledger).
#
# The cost model prices the NEXTN head from the TARGET config (one MTP layer
# of the target's own geometry). A DFlash draft is a separate checkpoint with
# its own geometry (Qwen3.8-27B-DFlash2: 5 layers, 32 q / 8 kv heads, fc over
# five captured hiddens, a top-K candidate selector with two fp32 codebooks),
# so its bytes are read off the checkpoint's own safetensors headers -- exact
# bytes per tensor, whatever the quantisation -- and grouped by how they
# shard on group D:
#
#   attn  layers.*.self_attn.*  -> by kv-head share (the draft's own head grid)
#   mlp   layers.*.mlp.*        -> by the MLP ratio vector
#   repl  everything else       -> replicated on every rank
#         (fc, hidden_norm, norm, candidate_selector, the grouped convs,
#         the per-layer norms)
#
# Measured 2026-09-17 (boot df2g3, TP3 3991/1000/1000, W8 draft): resident
# draft weights 1792 / 1167 / 1126 MiB per rank against 1536 / 783 / 750 MiB
# of checkpoint bytes dealt by these rules; the difference is the loader's
# dequant scratch and allocator rounding, a CALIBRATED residual, not a weight
# term.
from __future__ import annotations

import json
import os
import struct
from typing import Dict

_DTYPE_BYTES = {
    "BF16": 2, "F16": 2, "F32": 4, "F64": 8,
    "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8, "BOOL": 1,
    "F8_E4M3": 1, "F8_E5M2": 1,
}


def _safetensors_headers(model_path: str):
    """Every (name, shape, dtype) the checkpoint declares."""
    files = []
    index = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weight_map = json.load(f).get("weight_map", {})
        files = sorted({os.path.join(model_path, v) for v in weight_map.values()})
    else:
        files = sorted(
            os.path.join(model_path, n)
            for n in os.listdir(model_path)
            if n.endswith(".safetensors")
        )
    if not files:
        raise FileNotFoundError(f"no safetensors file under {model_path}")
    for path in files:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            yield name, meta["shape"], meta["dtype"]


def dflash_draft_family_bytes(model_path: str) -> Dict[str, int]:
    """Checkpoint bytes of the draft by shard family: attn / mlp / repl."""
    out = {"attn": 0, "mlp": 0, "repl": 0}
    for name, shape, dtype in _safetensors_headers(model_path):
        nbytes = _DTYPE_BYTES[dtype]
        for d in shape:
            nbytes *= int(d)
        if name.startswith("layers.") and ".self_attn." in name:
            out["attn"] += nbytes
        elif name.startswith("layers.") and ".mlp." in name:
            out["mlp"] += nbytes
        else:
            out["repl"] += nbytes
    return out


def dflash_draft_kv_heads(model_path: str) -> int:
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    heads = int(cfg.get("num_key_value_heads") or 0)
    if heads <= 0:
        raise ValueError(f"{model_path}: config.json declares no num_key_value_heads")
    return heads
