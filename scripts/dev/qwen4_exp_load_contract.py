#!/usr/bin/env python3
"""Prove the qwen4_exp loader contract by DRIVING THE REAL LOADER.

Register #1036. No GPU compute, no weight bytes moved, no network.

WHY THIS EXISTS. A model file for a brand-new architecture fails in exactly one
place first: a checkpoint tensor with no destination parameter, or a parameter no
checkpoint tensor ever reaches. On metal the first surfaces as a `KeyError` after a
175 GiB read; the second is far worse -- it never raises at all, and serves
plausible garbage from whatever the parameter was initialised to. `py_compile` and
an import smoke are both structurally blind to both. This check is chosen to MATCH
THAT FAILURE CLASS.

WHAT CHANGED, and why this is a stronger check than the one it replaces. The first
version of this script carried its own copy of the mapping rules and compared their
output against `named_parameters()`. It passed judgement on a loader it modelled.
Run against upstream's `qwen4_exp.py` it reported 99 unmatched destinations -- and
the RULES were wrong, not the checkpoint: upstream rewrites the `language_model.`
prefix away and FUSES q/k/v into `qkv_proj` and gate/up into `gate_up_proj`, none of
which the rules knew. A contract that models the loader can drift from it silently.
This version calls the loader instead, so it cannot.

HOW. The model is built on `meta`; every parameter's `weight_loader` is wrapped by a
recorder that delegates to the real one; `torch.Tensor.copy_` is made a no-op for
the duration of the walk; and all 296,474 checkpoint names are fed to
`model.load_weights()` as `meta` tensors carrying each tensor's REAL dtype and shape
(from the safetensors headers, captured beside the index by --shapes).

Suppressing ONLY the final byte copy is the point: every shape assertion, every
`narrow`, every shard-id and expert-id computation inside the real weight loaders
still runs. What is NOT proven is numerical correctness of the copy itself.

Then BOTH directions are asserted, because each is a different bug:
  * a checkpoint tensor no parameter consumed -> silently unloaded weights
  * a parameter no checkpoint tensor fed      -> silently random weights

Usage:
    python3 scripts/dev/qwen4_exp_load_contract.py \
        --index  /spinning/qwen38-flash-next/ckpt/model.safetensors.index.json \
        --config /spinning/qwen38-flash-next/ckpt
"""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import re
import sys
import tempfile
from collections import Counter, defaultdict

# Checkpoint name -> pattern, so 296,474 names collapse to a reviewable set.
_COLLAPSE = (
    (re.compile(r"\.layers\.\d+\."), ".layers.{L}."),
    (re.compile(r"\.experts\.\d+\."), ".experts.{E}."),
    (re.compile(r"\.blocks\.\d+\."), ".blocks.{B}."),
    (re.compile(r"\.shard_\d+\."), ".shard_{S}."),
    (re.compile(r"\.shard_\d+\b"), ".shard_{S}"),
)

# safetensors dtype string -> torch dtype name.
_DT = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I8": "int8",
    "I16": "int16",
    "I32": "int32",
    "I64": "int64",
    "U8": "uint8",
    "BOOL": "bool",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}


def collapse(name: str) -> str:
    for rx, repl in _COLLAPSE:
        name = rx.sub(repl, name)
    return name


# Two sets are legitimately unmatched, and each is DECLARED here with its reason
# so that anything NOT on these lists still fails the contract. A blanket
# "ignore what did not match" would defeat the whole check.
ALLOWED_UNFED = [
    (
        re.compile(r"\.experts\.w(13|2)(_weight)?_g_idx(_sort_indices)?$"),
        "GPTQ activation-reordering indices. CompressedTensorsWNA16MoEMethod "
        "allocates them unconditionally; this checkpoint is AWQ without act-order, "
        "so they are zero-filled placeholders the kernel reads as identity. Fed "
        "would be the surprise, not unfed.",
    ),
]

ALLOWED_UNCONSUMED = [
    (
        re.compile(r"^mtp\."),
        "The MTP/NEXTN draft head is a SEPARATE model (models/qwen4_exp_mtp.py) "
        "loaded only when speculative decoding is enabled, with its own "
        "load_weights. The target model is right to leave these alone. NOTE: that "
        "module does not import in this fork yet (_mtp_quant_config missing from "
        "models/qwen3_5_mtp.py), so these names are unproven rather than proven "
        "elsewhere -- tracked, not waved through.",
    ),
]


def excused(name, rules):
    for rx, why in rules:
        if rx.search(name):
            return why
    return None




def build_model(model_dir: str, raw_config: dict):
    """Construct the real model on `meta`. Every prerequisite here was found by
    RUNNING this script, not by reading: each one is a construction-time global the
    layers read while building."""
    import torch

    sys.path.insert(0, "python")

    # sglang's layers read the parallel state at construction time, so a 1-rank
    # group must exist first. gloo over a file:// rendezvous keeps this CPU-only and
    # PORTLESS: no CUDA context, no sockets, nothing a running serving boot could
    # observe.
    from sglang.srt.distributed import (
        init_distributed_environment,
        initialize_model_parallel,
    )

    rdzv = osp.join(tempfile.mkdtemp(prefix="qwen4_contract_"), "rdzv")
    init_distributed_environment(
        world_size=1,
        rank=0,
        distributed_init_method=f"file://{rdzv}",
        local_rank=0,
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1)

    from sglang.srt.configs.device_config import DeviceConfig
    from sglang.srt.configs.load_config import LoadConfig
    from sglang.srt.configs.model_config import ModelConfig
    from sglang.srt.layers.dp_attention import initialize_dp_attention
    from sglang.srt.model_loader.loader import (
        _get_quantization_config,
        _initialize_model,
    )
    from sglang.srt.model_loader.utils import set_default_torch_dtype
    from sglang.srt.runtime_context import lane_scope
    from sglang.srt.server_args import ServerArgs

    server_args = ServerArgs(model_path=model_dir)
    model_config = ModelConfig.from_server_args(server_args)

    # Some layers read the DP-attention globals while building. Defaults only
    # (dp_size 1, dp attention off): this is a name/shape contract, not a
    # parallelism test.
    initialize_dp_attention(server_args, model_config)

    # Construct through the PRODUCTION path -- DefaultModelLoader.load_model's own
    # three lines -- rather than calling the model class directly. That buys three
    # things a direct call does not: `get_model_architecture` resolves the class
    # through the registry (so the registry wiring is under test too),
    # `_get_quantization_config` parses the checkpoint's own
    # `quantization_config` (without it FusedMoE builds unquantized `w13_weight`
    # instead of the compressed-tensors `w13_weight_packed` set, and the loader
    # dies on the first expert tensor -- measured), and `set_default_torch_dtype`
    # fixes the dtype the same way a boot does.
    #
    # `lane_scope` installs an OVERLAY in a context variable rather than
    # overwriting the process-wide server-args slot -- the fork's own documented
    # idiom and the path its tests use.
    load_config = LoadConfig()
    with lane_scope(None, server_args):
        quant_config = _get_quantization_config(model_config, load_config)
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = _initialize_model(model_config, load_config, quant_config)
    print(f"quantization: {type(quant_config).__name__ if quant_config else None}")
    return model


def drive_loader(model, weight_map, shapes, expert_cap):
    """Feed every checkpoint name through the model's REAL load_weights.

    Returns (consumed, fed_params, errors, skipped) where `consumed` is the set of
    checkpoint names that reached a weight loader, `fed_params` the set of parameter
    names that received something, `errors` a name -> (kind, message) map, and
    `skipped` the names held back by --expert-cap.
    """
    import torch

    params = dict(model.named_parameters())
    by_id = {id(p): n for n, p in params.items()}

    consumed: set[str] = set()
    fed: set[str] = set()
    errors: dict[str, tuple[str, str]] = {}
    current = {"name": None}

    def wrap(param, real):
        def recorder(*a, **kw):
            dest = by_id.get(id(param), "<unknown-param>")
            fed.add(dest)
            if current["name"] is not None:
                consumed.add(current["name"])
            if real is not None:
                return real(*a, **kw)
            # No weight_loader of its own: the loader would have called
            # default_weight_loader, whose only contract is the shape assert.
            loaded = a[1] if len(a) > 1 else kw.get("loaded_weight")
            if loaded is not None and tuple(loaded.shape) != tuple(param.shape):
                raise AssertionError(
                    f"shape {tuple(loaded.shape)} != param {tuple(param.shape)}"
                )
            return None

        return recorder

    for _n, p in params.items():
        p.weight_loader = wrap(p, getattr(p, "weight_loader", None))

    # Suppress ONLY the byte copy. Every shape assert, narrow and expert-id
    # computation above it still runs.
    orig_copy_ = torch.Tensor.copy_

    def no_copy(self, *a, **kw):
        return self

    # Some loaders assign into a slice instead of calling copy_.
    orig_setitem = torch.Tensor.__setitem__

    def no_setitem(self, *a, **kw):
        return None

    skipped: set[str] = set()

    def weights():
        for name in weight_map:
            if expert_cap is not None:
                m = re.search(r"\.experts\.(\d+)\.", name)
                if m and int(m.group(1)) >= expert_cap:
                    skipped.add(name)
                    continue
            dt, shape = shapes[name]
            tdt = getattr(torch, _DT[dt])
            current["name"] = name
            yield name, torch.empty(shape, dtype=tdt, device="meta")

    # The PLE lane BYPASSES weight_loader entirely: n-gram shard rows are copied
    # straight into the embedding, and PLE metadata straight into buffers.
    # Instrumenting only weight_loader therefore reported 128 shards + 3 buffers as
    # "silently unloaded" when the loader had in fact placed every one -- a false
    # positive in THIS SCRIPT, not a defect in the model.
    #
    # The copy helper is a function NESTED inside load_weights and cannot be
    # patched, so the authority used instead is the loader's OWN bookkeeping:
    # load_weights folds loaded_buffers and loaded_shard_params into the set it
    # returns (qwen4_exp.py:2105-2106,2118). Deriving the verdict from the loader's
    # own record rather than from a guess is the same principle as driving the
    # loader rather than modelling it.
    torch.Tensor.copy_ = no_copy
    torch.Tensor.__setitem__ = no_setitem
    returned: set[str] = set()
    try:
        try:
            returned = set(model.load_weights(weights()) or ())
        except Exception as exc:  # noqa: BLE001 - classify, do not raise
            kind = type(exc).__name__
            errors[current["name"] or "<before-first-name>"] = (kind, str(exc)[:300])
    finally:
        torch.Tensor.copy_ = orig_copy_
        torch.Tensor.__setitem__ = orig_setitem

    # Parameters and buffers the loader says it filled.
    fed.update(n for n in returned if n in params)

    # A PLE shard is consumed exactly when the loader recorded the embedding it
    # feeds. The rewrite mirrors the loader's own (qwen4_exp.py:1958).
    for name in weight_map:
        if name in skipped or name in consumed:
            continue
        inner = name.replace("model.language_model.", "model.")
        if inner in returned:            # PLE metadata buffers, recorded by name
            consumed.add(name)
            continue
        m = re.match(r"(.*)\.ngram_embedding\.shard_\d+\.weight$", inner)
        if m and f"{m.group(1)}.ngram_embedding.weight" in returned:
            consumed.add(name)

    return consumed, fed, errors, skipped, set(params)


def report(title, names, limit=25):
    print(f"\n{title} ({len(names)}):")
    pats = Counter(collapse(n) for n in names)
    for pat, n in sorted(pats.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]:
        print(f"  {n:>7d}x  {pat}")
    if len(pats) > limit:
        print(f"  ... and {len(pats) - limit} further patterns")
    return pats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--config", required=True, help="checkpoint dir or config.json")
    ap.add_argument(
        "--shapes",
        help="tensor_shapes.json (name -> [dtype, shape]); default: beside --index",
    )
    ap.add_argument(
        "--expert-cap",
        type=int,
        help="only feed experts with id < N (speed knob; weakens the proof, and "
        "the report says so)",
    )
    ap.add_argument(
        "--dump",
        metavar="REGEX",
        help="print every MODULE and PARAMETER whose name matches, then exit. The "
        "answer to 'what is this thing actually called', which is the question "
        "every name-mapping bug turns out to be.",
    )
    ap.add_argument(
        "--full", action="store_true", help="do not truncate the pattern reports"
    )
    ap.add_argument(
        "--allow-rules-only",
        action="store_true",
        help="do not fail merely because the model modules are not importable yet",
    )
    args = ap.parse_args()

    if osp.isdir(args.config):
        model_dir, config_path = args.config, osp.join(args.config, "config.json")
    else:
        config_path = args.config
        model_dir = osp.dirname(osp.abspath(args.config))

    with open(args.index) as fh:
        weight_map = json.load(fh)["weight_map"]
    with open(config_path) as fh:
        raw_config = json.load(fh)

    shapes_path = args.shapes or osp.join(osp.dirname(args.index), "tensor_shapes.json")
    if not osp.exists(shapes_path):
        print(f"FATAL: no tensor shape map at {shapes_path}")
        print("  This contract feeds REAL dtypes and shapes so the loaders' own")
        print("  assertions stay live. Capture it from the safetensors headers.")
        return 2
    with open(shapes_path) as fh:
        shapes = json.load(fh)

    patterns = Counter(collapse(k) for k in weight_map)
    print(f"checkpoint: {len(weight_map)} tensors -> {len(patterns)} name patterns")
    print(f"shards:     {len(set(weight_map.values()))}")
    missing_shapes = set(weight_map) - set(shapes)
    if missing_shapes:
        print(f"FATAL: {len(missing_shapes)} indexed tensors absent from the shape map")
        return 2

    try:
        model = build_model(model_dir, raw_config)
    except Exception as exc:  # noqa: BLE001
        print(f"\nCOULD NOT CONSTRUCT the model ({type(exc).__name__}: {exc})")
        print("  The parameter-existence half of this contract did NOT run.")
        return 0 if args.allow_rules_only else 2

    params = dict(model.named_parameters())
    print(f"\nCONSTRUCTED on meta: {len(params)} parameters")

    if args.dump:
        rx = re.compile(args.dump)
        print(f"\nMODULES matching /{args.dump}/:")
        for n, m in model.named_modules():
            if n and rx.search(n):
                print(f"  {n}  <{type(m).__name__}>")
        print(f"\nPARAMETERS matching /{args.dump}/:")
        for n, p in params.items():
            if rx.search(n):
                print(f"  {n}  {tuple(p.shape)} {p.dtype}")
        print(f"\nBUFFERS matching /{args.dump}/:")
        for n, b in model.named_buffers():
            if rx.search(n):
                print(f"  {n}  {tuple(b.shape)} {b.dtype}")
        return 0

    consumed, fed, errors, skipped, all_params = drive_loader(
        model, weight_map, shapes, args.expert_cap
    )

    offered = set(weight_map) - skipped
    unconsumed_all = offered - consumed
    unfed_all = all_params - fed

    # Split each direction into DECLARED-and-explained versus genuinely wrong.
    unconsumed = {n for n in unconsumed_all if not excused(n, ALLOWED_UNCONSUMED)}
    unfed = {n for n in unfed_all if not excused(n, ALLOWED_UNFED)}
    excused_unconsumed = unconsumed_all - unconsumed
    excused_unfed = unfed_all - unfed

    print(f"\ndrove load_weights over {len(offered)} checkpoint tensors")
    print(f"  consumed by a weight loader: {len(consumed)}")
    print(f"  parameters fed:              {len(fed)} / {len(all_params)}")
    if skipped:
        print(f"  HELD BACK by --expert-cap:   {len(skipped)}  (proof is WEAKENED)")

    for label, names, rules in (
        ("checkpoint tensors", excused_unconsumed, ALLOWED_UNCONSUMED),
        ("parameters", excused_unfed, ALLOWED_UNFED),
    ):
        if not names:
            continue
        print(f"\nDECLARED-UNMATCHED {label} ({len(names)}) -- excused WITH a reason:")
        seen = {}
        for n in names:
            seen.setdefault(excused(n, rules), []).append(n)
        for why, group in seen.items():
            print(f"  {len(group)}x  e.g. {collapse(sorted(group)[0])}")
            print(f"      {why}")

    print(
        f"\nfully accounted for: "
        f"{len(consumed) + len(excused_unconsumed)}/{len(offered)} tensors, "
        f"{len(fed) + len(excused_unfed)}/{len(all_params)} parameters"
    )

    if errors:
        print(f"\nLOADER RAISED ({len(errors)}):")
        for name, (kind, msg) in list(errors.items())[:10]:
            print(f"  {collapse(name)}\n      {kind}: {msg}")

    if unconsumed:
        report("CHECKPOINT TENSORS NO PARAMETER CONSUMED -> silently unloaded", unconsumed)
    if unfed:
        report("PARAMETERS NO CHECKPOINT TENSOR FED -> silently random", unfed)

    if errors or unconsumed or unfed:
        print("\nCONTRACT FAILED.")
        return 1

    print("\nOK: every checkpoint tensor reached a parameter, and every parameter")
    print("    was fed, through the model's OWN load_weights.")
    print(f"    mode=DRIVEN  tensors={len(offered)}  params={len(all_params)}")
    if args.expert_cap:
        print("    NOTE: --expert-cap was set, so this is a subset proof.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
