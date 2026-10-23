#!/usr/bin/env python3
"""Lossless repack: AutoRound ``auto_round:auto_gptq`` W4A16 (sym, no act-order)
-> compressed-tensors ``pack-quantized`` W4A16 (strategy group).

Why (27b-int4ar, 23.10.): the pdflip weight exchange declares geometry for the
compressed-tensors WNA16 parameter set (``weight_packed`` / ``weight_scale``,
MIXED_FUSED_COLS weg2xsn258 on the 27B lued W8A16 checkpoint, NF int4 line);
the GPTQ-Marlin parameter set (``qweight``/``scales``/``qzeros``/``g_idx``) has
never been through a flip. Both loaders end in the same ``gptq_marlin_repack``
Marlin layout, so the repack changes names and storage orientation only.

Bit-exactness argument (checked per tensor below, not assumed):
  * GPTQ ``qweight`` int32 [K/8, N]: nibble j of qweight[i, n] is q[8i+j, n].
    CT ``weight_packed`` int32 [N, K/8]: nibble j of wp[n, i] is q[n, 8i+j]
    (compressed_tensors pack_to_int32, packed_dim=1).  => wp = qweight.T.
  * GPTQ sym 4-bit stores unsigned q in [0,15] with zero point 8; auto_gptq v1
    stores ``qzeros`` = zero-1 = 7 in every nibble (0x77777777).  CT sym int4
    stores signed v in [-8,7] as v+8 = the same unsigned nibble.  The script
    REFUSES any qzeros word that is not 0x77777777 (asymmetric would need a
    zero-point tensor, not supported here).
  * ``scales`` fp16 [K/g, N] -> ``weight_scale`` [N, K/g]; cast to the model
    dtype (bf16), the same cast the GPTQ-Marlin loader does when it copies the
    fp16 checkpoint scales into its params_dtype parameter.
  * ``g_idx`` must be absent or the trivial i//g (no act-order).
Everything else (bf16 weights, norms, vision, embed, lm_head, mtp.fc) is copied
byte-identical.

Vision tower (default on, ``--no-vision-shard`` to keep the source layout): every
``visual`` tensor goes into ONE dedicated shard ``model-visual.safetensors``.  The
pdflip transient-vision loader (planner/vision_stage_load.py ``tower_extent``)
reads the tower as a single extent of a single shard and refuses a tower that
spans shards (the AutoRound export splits it over shards 4 and 5).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Dict

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SYM_ZERO_WORD = 0x77777777  # eight nibbles of 7 (auto_gptq v1: zero 8 stored as 7)


def _ignore_list(cfg: dict) -> list:
    """CT ignore list: every module the AutoRound export kept in bf16."""
    ign = [
        "lm_head",
        "re:.*visual.*",
        "re:.*linear_attn[.]in_proj_a$",
        "re:.*linear_attn[.]in_proj_b$",
        "re:.*(^|[.])fc$",  # mtp.fc (bf16 in the export)
    ]
    return ign


def ct_quant_config(src_q: dict) -> dict:
    bits, gs, sym = int(src_q["bits"]), int(src_q["group_size"]), bool(src_q["sym"])
    if bits != 4 or not sym or gs <= 0:
        raise SystemExit(f"unsupported source quantization bits={bits} sym={sym} group={gs}")
    if str(src_q.get("packing_format", "")) != "auto_round:auto_gptq":
        raise SystemExit(f"unsupported packing_format {src_q.get('packing_format')!r}")
    if src_q.get("desc_act") or src_q.get("act_order"):
        raise SystemExit("act-order checkpoints are not supported")
    return {
        "config_groups": {
            "group_0": {
                "format": "pack-quantized",
                "input_activations": None,
                "output_activations": None,
                "targets": ["Linear"],
                "weights": {
                    "actorder": None, "block_structure": None, "dynamic": False,
                    "group_size": gs, "num_bits": 4, "observer": "minmax",
                    "observer_kwargs": {}, "scale_dtype": None, "strategy": "group",
                    "symmetric": True, "type": "int", "zp_dtype": None,
                },
            }
        },
        "format": "pack-quantized",
        "global_compression_ratio": None,
        "ignore": _ignore_list(src_q),
        "kv_cache_scheme": None,
        "quant_method": "compressed-tensors",
        "quantization_status": "compressed",
        "sparsity_config": {},
        "transform_config": {},
        "version": "0.17.1",
        "_converted_from": {
            "quant_method": src_q.get("quant_method"),
            "packing_format": src_q.get("packing_format"),
            "autoround_version": src_q.get("autoround_version"),
            "tool": "tools/quant/autoround_gptq_to_ct.py (lossless repack)",
        },
    }


def is_vision(key: str) -> bool:
    return ".visual." in "." + key


def convert_group(prefix: str, t: Dict[str, torch.Tensor], group: int,
                  scale_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
    qw = t[prefix + ".qweight"]
    sc = t[prefix + ".scales"]
    qz = t.get(prefix + ".qzeros")
    gi = t.get(prefix + ".g_idx")
    if qw.dtype != torch.int32 or qw.dim() != 2:
        raise SystemExit(f"{prefix}: qweight {qw.dtype} {tuple(qw.shape)}")
    k8, n = qw.shape
    k = k8 * 8
    if k % group or sc.shape != (k // group, n):
        raise SystemExit(f"{prefix}: scales {tuple(sc.shape)} vs K={k} N={n} g={group}")
    if qz is not None:
        if qz.shape != (k // group, n // 8):
            raise SystemExit(f"{prefix}: qzeros {tuple(qz.shape)}")
        bad = int((qz != torch.tensor(SYM_ZERO_WORD, dtype=torch.int64).to(torch.int32)).sum())
        if bad:
            raise SystemExit(f"{prefix}: {bad} qzeros words != 0x77777777 (not symmetric zero 8)")
    if gi is not None:
        want = torch.arange(k, dtype=gi.dtype) // group
        if not torch.equal(gi, want):
            raise SystemExit(f"{prefix}: non-trivial g_idx (act-order)")
    return {
        prefix + ".weight_packed": qw.t().contiguous(),
        prefix + ".weight_scale": sc.t().contiguous().to(scale_dtype),
        prefix + ".weight_shape": torch.tensor([n, k], dtype=torch.int64),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--scale-dtype", default="bfloat16")
    ap.add_argument("--no-vision-shard", action="store_true",
                    help="keep visual tensors in their source shards (default: one dedicated shard)")
    args = ap.parse_args()
    src, dst = os.path.abspath(args.src), os.path.abspath(args.dst)
    if os.path.exists(dst) and os.listdir(dst):
        raise SystemExit(f"{dst} exists and is not empty")
    os.makedirs(dst, exist_ok=True)
    cfg = json.load(open(os.path.join(src, "config.json")))
    src_q = cfg["quantization_config"]
    group = int(src_q["group_size"])
    scale_dtype = getattr(torch, args.scale_dtype)
    cfg["quantization_config"] = ct_quant_config(src_q)
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    new_map: Dict[str, str] = {}
    n_groups = n_copied = 0
    total = 0
    vision: Dict[str, torch.Tensor] = {}
    VISION_SHARD = "model-visual.safetensors"
    for fname in sorted(set(wmap.values())):
        tensors: Dict[str, torch.Tensor] = {}
        with safe_open(os.path.join(src, fname), "pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        prefixes = sorted({k[: -len(".qweight")] for k in tensors if k.endswith(".qweight")})
        out: Dict[str, torch.Tensor] = {}
        consumed = set()
        for p in prefixes:
            out.update(convert_group(p, tensors, group, scale_dtype))
            consumed.update(p + s for s in (".qweight", ".scales", ".qzeros", ".g_idx"))
            n_groups += 1
        for key, val in tensors.items():
            if key in consumed:
                continue
            if key.endswith((".scales", ".qzeros", ".g_idx")):
                raise SystemExit(f"orphan quant tensor {key}")
            n_copied += 1
            if not args.no_vision_shard and is_vision(key):
                vision[key] = val
                continue
            out[key] = val
        if not out:  # a shard that held only vision tensors
            print(f"{fname}: empty after the vision split, not written", flush=True)
            del tensors
            continue
        save_file(out, os.path.join(dst, fname), metadata={"format": "pt"})
        for key, val in out.items():
            new_map[key] = fname
            total += val.numel() * val.element_size()
        print(f"{fname}: groups={len(prefixes)} tensors_out={len(out)}", flush=True)
        del tensors, out
    if vision:
        save_file(vision, os.path.join(dst, VISION_SHARD), metadata={"format": "pt"})
        for key, val in vision.items():
            new_map[key] = VISION_SHARD
            total += val.numel() * val.element_size()
        print(f"{VISION_SHARD}: tensors_out={len(vision)}", flush=True)
    json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(new_map.items()))},
              open(os.path.join(dst, "model.safetensors.index.json"), "w"), indent=2)
    json.dump(cfg, open(os.path.join(dst, "config.json"), "w"), indent=2)
    skip = {"config.json", "model.safetensors.index.json", "quantization_config.json"}
    for name in sorted(os.listdir(src)):
        sp = os.path.join(src, name)
        if name in skip or name.endswith(".safetensors") or not os.path.isfile(sp) or name.startswith("."):
            continue
        shutil.copy2(sp, os.path.join(dst, name))
    print(f"DONE groups={n_groups} copied={n_copied} total_bytes={total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
