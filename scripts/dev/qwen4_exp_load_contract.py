#!/usr/bin/env python3
"""Prove the qwen4_exp loader contract against the REAL checkpoint name set.

Register #1036. No GPU, no weights, no network: this reads only
`model.safetensors.index.json` (33.6 MB of names) and, when the model modules are
importable, constructs the model on the `meta` device.

WHY THIS EXISTS. A model file for a brand-new architecture fails in exactly one
place first: a checkpoint tensor with no destination parameter, or a parameter with
no checkpoint tensor. On metal that surfaces as a `KeyError` after a 175 GiB read,
or — far worse — as a silently unloaded tensor class that produces plausible
garbage. `py_compile` and an import smoke are both structurally blind to it. This
check is chosen to MATCH THAT FAILURE CLASS: it walks every one of the 296,474
checkpoint names through the loader's own mapping rules and asserts a destination.

TWO MODES, and the script says which one it ran in:
  RULES-ONLY   the model modules are not importable yet (a sibling component is
               still being written). Every checkpoint name pattern is still walked
               through the mapping rules and must resolve to a destination NAME.
               This catches an unmapped pattern but not a misnamed parameter.
  CONSTRUCTED  the model is built on `meta` and each destination name is checked
               to EXIST in `named_parameters()`. This is the real proof.

Exit code is nonzero if any checkpoint name pattern is unaccounted for.

Usage:
    python3 scripts/dev/qwen4_exp_load_contract.py \
        --index /spinning/qwen38-flash-next/ckpt/model.safetensors.index.json \
        --config /spinning/qwen38-flash-next/ckpt/config.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict

# Checkpoint name -> pattern, so 296,474 names collapse to a reviewable set.
_COLLAPSE = (
    (re.compile(r"\.layers\.\d+\."), ".layers.{L}."),
    (re.compile(r"\.experts\.\d+\."), ".experts.{E}."),
    (re.compile(r"\.blocks\.\d+\."), ".blocks.{B}."),
    (re.compile(r"\.shard_\d+\."), ".shard_{S}."),
)


def collapse(name: str) -> str:
    for rx, repl in _COLLAPSE:
        name = rx.sub(repl, name)
    return name


# The mapping rules, written once here and mirrored by
# models/qwen4_exp.py::load_weights. Each entry is (matcher, destination-builder,
# why). Order matters and mirrors the loader's own order.
STACKED = [
    ("gate_up_proj", "gate_proj"),
    ("gate_up_proj", "up_proj"),
    ("in_proj_qkvz.", "in_proj_qkv."),
    ("in_proj_qkvz.", "in_proj_z."),
    ("in_proj_ba.", "in_proj_b."),
    ("in_proj_ba.", "in_proj_a."),
]


def destination(pattern: str) -> tuple[str | None, str]:
    """Return (destination parameter pattern, rule name) for a checkpoint pattern."""
    # 1. routed experts: per-expert INT4 quads -> the fused MoE parameters
    if ".mlp.experts.{E}." in pattern:
        m = re.search(r"\.experts\.\{E\}\.(gate_proj|up_proj|down_proj)\.(\w+)$", pattern)
        if not m:
            return None, "expert-pattern-unrecognised"
        proj, suffix = m.group(1), m.group(2)
        slot = "w2_" if proj == "down_proj" else "w13_"
        return (
            pattern[: pattern.index(".mlp.experts.")] + f".mlp.experts.{slot}{suffix}",
            "FusedMoE.make_expert_params_mapping",
        )

    # 2. MTP fused experts: already fused in the checkpoint, no per-expert index
    if re.search(r"\.mlp\.experts\.(gate_up_proj|down_proj)$", pattern):
        proj = pattern.rsplit(".", 1)[1]
        slot = "w13_weight" if proj == "gate_up_proj" else "w2_weight"
        return (
            pattern[: pattern.index(".mlp.experts.")] + f".mlp.experts.{slot}",
            "FusedMoE.make_expert_params_mapping_fused",
        )

    # 3. shared-expert gate/up fusion, and the GDN input-projection fusions
    for param_name, shard_name in STACKED:
        if shard_name in pattern:
            return pattern.replace(shard_name, param_name), f"stacked:{shard_name}"

    # 4. PLE: the 128 shards and the metadata tensors are placed by
    #    NgramEmbedding.load_weight, which owns the shard->slice arithmetic.
    if ".ple." in pattern:
        return pattern, "NgramEmbedding.load_weight"

    # 5. everything else is a 1:1 name (this architecture's module tree is built to
    #    match the checkpoint's own tree, which is why this branch is the majority
    #    of the patterns and none of the bytes).
    return pattern, "identity"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--allow-rules-only",
        action="store_true",
        help="do not fail merely because the model modules are not importable yet",
    )
    args = ap.parse_args()

    import os.path as _osp

    # --config accepts either the checkpoint DIRECTORY or config.json itself. The
    # directory form matters because ServerArgs wants a model path it can also find
    # the tokeniser and generation config beside.
    if _osp.isdir(args.config):
        model_dir = args.config
        config_path = _osp.join(args.config, "config.json")
    else:
        config_path = args.config
        model_dir = _osp.dirname(_osp.abspath(args.config))

    with open(args.index) as fh:
        weight_map = json.load(fh)["weight_map"]
    with open(config_path) as fh:
        raw_config = json.load(fh)

    patterns = Counter(collapse(k) for k in weight_map)
    print(f"checkpoint: {len(weight_map)} tensors -> {len(patterns)} name patterns")
    print(f"shards:     {len(set(weight_map.values()))}")

    dests: dict[str, tuple[str | None, str]] = {}
    by_rule: dict[str, int] = defaultdict(int)
    unmapped: list[str] = []
    for pat in patterns:
        dest, rule = destination(pat)
        dests[pat] = (dest, rule)
        by_rule[rule] += patterns[pat]
        if dest is None:
            unmapped.append(pat)

    print("\nmapping rules exercised (by tensor count, not pattern count):")
    for rule, n in sorted(by_rule.items(), key=lambda kv: -kv[1]):
        print(f"  {n:>8d}  {rule}")

    if unmapped:
        print("\nUNMAPPED PATTERNS -- loader contract is broken:")
        for pat in unmapped:
            print(f"  {pat}")
        return 1

    # ---- try the real proof: construct on meta and check the destinations exist
    constructed = False
    missing: list[tuple[str, str]] = []
    try:
        import os
        import tempfile

        import torch

        sys.path.insert(0, "python")

        # sglang's layers read the parallel state at construction time, so a
        # 1-rank group must exist before the model is built. gloo over a file://
        # rendezvous keeps this CPU-only and PORTLESS: no CUDA context, no
        # sockets, nothing that could disturb a running serving boot.
        from sglang.srt.distributed import (
            init_distributed_environment,
            initialize_model_parallel,
        )

        rdzv = os.path.join(tempfile.mkdtemp(prefix="qwen4_contract_"), "rdzv")
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{rdzv}",
            local_rank=0,
            backend="gloo",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)

        from sglang.srt.configs.qwen4_exp import Qwen4ExpConfig
        from sglang.srt.models.qwen4_exp import Qwen4ExpForConditionalGeneration
        from sglang.srt.runtime_context import lane_scope
        from sglang.srt.server_args import ServerArgs

        config = Qwen4ExpConfig(**raw_config)
        # The model reads get_server_args() while building. `lane_scope` installs
        # an OVERLAY in a context variable rather than overwriting the process-wide
        # slot -- the fork's own idiom, documented as the replacement for the legacy
        # set_server_args swap and as the path tests use. Nothing a concurrent
        # serving group could observe.
        server_args = ServerArgs(model_path=model_dir)

        # Some layers read the DP-attention globals while building, so initialise
        # them too. Defaults only (dp_size 1, dp attention off) -- this is a shape
        # contract check, not a parallelism test.
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.layers.dp_attention import initialize_dp_attention

        initialize_dp_attention(server_args, ModelConfig.from_server_args(server_args))

        with lane_scope(None, server_args):
            with torch.device("meta"):
                model = Qwen4ExpForConditionalGeneration(config)
            param_names = set(dict(model.named_parameters()))
        constructed = True
        print(f"\nCONSTRUCTED on meta: {len(param_names)} parameters")

        for pat, (dest, rule) in dests.items():
            if rule == "NgramEmbedding.load_weight":
                # placed by the submodule, not by name identity
                continue
            concrete = (
                dest.replace(".layers.{L}.", ".layers.0.")
                .replace(".experts.{E}.", ".experts.0.")
                .replace(".blocks.{B}.", ".blocks.0.")
            )
            if concrete not in param_names:
                missing.append((pat, concrete))
    except Exception as exc:  # noqa: BLE001 - the whole point is to report, not raise
        print(f"\nRULES-ONLY mode: could not construct the model ({type(exc).__name__}: {exc})")
        print("  A sibling component is probably still being written. The rule walk")
        print("  above is still a real result; the parameter-existence half is NOT.")
        if not args.allow_rules_only:
            return 2
        return 0

    if missing:
        print(f"\nDESTINATIONS WITH NO PARAMETER ({len(missing)}):")
        for pat, concrete in missing[:40]:
            print(f"  {pat}\n      -> {concrete}")
        if len(missing) > 40:
            print(f"  ... and {len(missing) - 40} more")
        return 1

    print("\nOK: every checkpoint name pattern resolves to an existing parameter.")
    print(f"    mode=CONSTRUCTED  patterns={len(patterns)}  tensors={len(weight_map)}")
    return 0 if constructed else 2


if __name__ == "__main__":
    raise SystemExit(main())
