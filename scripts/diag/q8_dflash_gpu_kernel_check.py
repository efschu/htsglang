#!/usr/bin/env python3
"""#290 confirmation: do the DFLASH drafter's PACKED kernels compute the right thing?

NO SERVER BOOT, ONE CARD, ~1.7 GiB, well under a minute of card time.

Why this exists rather than "just re-run the boot". The #290 root cause is
proven on CPU (the runtime built a dense drafter skeleton and silently dropped
all 36 packed tensors). The fix makes the drafter GGUF-resident -- which means
``fused_mul_mat_gguf`` runs over the drafter's shapes for the FIRST time. Every
earlier DFLASH run, working or broken, used dense matmuls for the draft trunk.
That is a genuinely new GPU path, and it is falsifiable without loading a 27B
target, capturing graphs, or holding the card for a serving window.

What is checked, per module class the drafter actually has:

  fc            ReplicatedLinear      25600 -> 5120   (the new one: a packed
                                                      REPLICATED weight had no
                                                      loader before this fix)
  qkv_proj      QKVParallelLinear     5120  -> 6144   (merged, 3 shards)
  gate_up_proj  MergedColumnParallel  5120  -> 34816  (merged, 2 shards)
  down_proj     RowParallelLinear     17408 -> 5120

Each module's ``quant_method.apply`` is compared against a dense reference
matmul built from the BF16 release of the same drafter. Q8_0 through a fused
kernel in bf16 lands well inside 2% relative error; a mis-sliced merged shard
or a mis-read block layout lands at 100%+.

    CUDA_VISIBLE_DEVICES=<one free card> \\
    PYTHONPATH=<worktree>/python <venv>/bin/python \\
        scripts/diag/q8_dflash_gpu_kernel_check.py

Exit 0 = the packed drafter computes the reference result and the boot is worth
its window. Exit 1 = do not spend the window; the kernel path is the next bug.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types

MODEL_ROOT = "/spinning/llm_stuff/club-3090/models-cache"
DEFAULT_GGUF = f"{MODEL_ROOT}/qwen3.6-27b-dflash-gguf/Qwen3.6-27B-DFlash-Q8_0.gguf"
DEFAULT_HF = f"{MODEL_ROOT}/qwen3.6-27b-dflash/model.safetensors"

# The drafter proposes a BLOCK of this many rows per round, so this is the row
# count the fused GEMV gate has to admit (DEFAULT_DFLASH_BLOCK_SIZE). Checking
# at 1 and at the block size covers both sides of that gate.
ROW_COUNTS = (1, 16)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default=DEFAULT_GGUF)
    ap.add_argument("--safetensors", default=DEFAULT_HF)
    ap.add_argument("--rel-tol", type=float, default=0.02)
    ap.add_argument("--time-budget-s", type=float, default=240.0)
    args = ap.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device visible")
        return 1
    started = time.perf_counter()

    from transformers import PretrainedConfig

    from sglang.srt.runtime_context import _CONTEXT, get_parallel

    if getattr(_CONTEXT, "_server_args", None) is None:
        # RotaryEmbedding reads exactly one field during construction.
        _CONTEXT._server_args = types.SimpleNamespace(rl_on_policy_target=None)

    from sglang.srt.layers.quantization.gguf import GGUFConfig
    from sglang.srt.model_loader.gguf_dflash import (
        build_dflash_name_map,
        dflash_unquantized_module_prefixes,
    )
    from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator
    from sglang.srt.models.dflash import DFlashDraftModel

    with open(os.path.join(os.path.dirname(args.gguf), "config.json")) as f:
        cfg = PretrainedConfig(**json.load(f))

    quant_config = GGUFConfig()
    for prefix in dflash_unquantized_module_prefixes(cfg):
        if prefix not in quant_config.modules_to_not_convert:
            quant_config.modules_to_not_convert.append(prefix)

    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.bfloat16)
    with get_parallel().override(
        tp_size=1, tp_rank=0, world_size=1, world_rank=0, pp_size=1, pp_rank=0
    ):
        with device:
            model = DFlashDraftModel(cfg, quant_config=quant_config, prefix="")
        # Raises if any parameter is left unloaded -- the #290 guard itself.
        model.load_weights(
            gguf_quant_weights_iterator(args.gguf, build_dflash_name_map(cfg))
        )
        for _, module in model.named_modules():
            quant_method = getattr(module, "quant_method", None)
            if quant_method is not None and hasattr(
                quant_method, "process_weights_after_loading"
            ):
                quant_method.process_weights_after_loading(module)

    load_s = time.perf_counter() - started
    if load_s > args.time_budget_s:
        # The card is shared. Overrunning here means something is paging or
        # dequantizing that should not be; release rather than push on.
        print(
            f"FAIL: load took {load_s:.0f}s, over the {args.time_budget_s:.0f}s budget"
        )
        return 1

    resident = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"drafter resident: {resident / 2**30:.2f} GiB (dense bf16 = 3.22 GiB)")
    if resident > 2.5 * 2**30:
        print("FAIL: the drafter is not packed -- the skeleton is still dense")
        return 1

    from safetensors import safe_open

    # module attribute path -> the reference tensors, in the module's own
    # output order. Merged modules concatenate along the output dim.
    cases = {
        "fc": ["fc.weight"],
        "layers.0.self_attn.qkv_proj": [
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
        ],
        "layers.0.mlp.gate_up_proj": [
            "layers.0.mlp.gate_proj.weight",
            "layers.0.mlp.up_proj.weight",
        ],
        "layers.0.mlp.down_proj": ["layers.0.mlp.down_proj.weight"],
    }

    failures = []
    with safe_open(args.safetensors, "pt") as st:
        for path, hf_names in cases.items():
            module = model.get_submodule(path)
            reference = torch.cat(
                [
                    st.get_tensor(n).to(device=device, dtype=torch.bfloat16)
                    for n in hf_names
                ],
                dim=0,
            )
            in_features = int(reference.shape[1])
            for rows in ROW_COUNTS:
                # Sampled on the CPU on purpose: on-GPU RNG is not identical
                # across architectures, and this check is meant to read the
                # same on a 3080 and a 5090.
                x = torch.randn(rows, in_features, dtype=torch.bfloat16).to(device)
                packed = module.quant_method.apply(module, x)
                dense = torch.matmul(x, reference.T)
                if packed.shape != dense.shape:
                    print(
                        f"FAIL {path} rows={rows}: packed{tuple(packed.shape)} "
                        f"vs dense{tuple(dense.shape)}"
                    )
                    failures.append(path)
                    continue
                denom = dense.abs().mean().clamp_min(1e-6)
                rel = float((packed - dense).abs().mean() / denom)
                verdict = "ok" if rel < args.rel_tol else "FAIL"
                print(f"{verdict:4s} {path:32s} rows={rows:3d} rel_mean_err={rel:.5f}")
                if rel >= args.rel_tol:
                    failures.append(path)
            del reference

    elapsed = time.perf_counter() - started
    print(f"elapsed {elapsed:.1f}s (budget {args.time_budget_s:.0f}s)")
    if failures:
        print(f"FAIL: {sorted(set(failures))}")
        return 1
    print("PASS: the packed drafter reproduces the dense reference")
    return 0


if __name__ == "__main__":
    sys.exit(main())
