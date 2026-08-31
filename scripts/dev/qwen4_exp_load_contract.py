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
import struct
import sys
import tempfile
from collections import Counter, defaultdict

_HDR_CACHE: dict[str, dict] = {}


def safetensors_header(path: str) -> dict | None:
    """The header alone: an 8-byte little-endian length, then that many bytes of JSON.

    [#1036] Exists so the shape sidecar can be a CACHE rather than an authority. The
    sidecar is captured from the checkpoint's headers at one moment; any later repoint
    -- adopting an fp8 PLE table ADDS a `weight_scale` name -- leaves it stale, and a
    stale cache must never be the reason a valid checkpoint is rejected. Reading a
    header is a seek plus a few KB, not a file read.
    """
    if path in _HDR_CACHE:
        return _HDR_CACHE[path]
    try:
        with open(path, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            hdr = json.loads(fh.read(n))
    except (OSError, ValueError, struct.error):
        return None
    hdr.pop("__metadata__", None)
    _HDR_CACHE[path] = hdr
    return hdr

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
        "load_weights, so the TARGET model is right to leave these alone. Pass "
        "--mtp to build that head and drive it too, which proves these 31 names "
        "instead of excusing them.",
    ),
]

# The draft head's own direction. Its `lm_head` and `embed_tokens` are TIED to the
# target's at runtime, never loaded from the checkpoint.
ALLOWED_UNFED_MTP = [
    (
        re.compile(r"^(lm_head\.weight|model\.embed_tokens\.weight)$"),
        "Shared with the target model, not loaded from the checkpoint: the runtime "
        "calls set_embed_and_head / set_lm_head_from_target "
        "(qwen3_5_mtp.py:176,207) to point the draft head at the target's tensors. "
        "A checkpoint copy would be a second, silently divergent embedding.",
    ),
]


def excused(name, rules):
    for rx, why in rules:
        if rx.search(name):
            return why
    return None




def build_model(model_dir: str, raw_config: dict, with_mtp: bool = False):
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
    mtp = None
    with lane_scope(None, server_args):
        quant_config = _get_quantization_config(model_config, load_config)
        with set_default_torch_dtype(model_config.dtype):
            with torch.device("meta"):
                model = _initialize_model(model_config, load_config, quant_config)
                if with_mtp:
                    # The draft head is a SEPARATE model with its own load_weights;
                    # it is built here, inside the same lane and the same meta
                    # device, because the distributed and DP-attention globals are
                    # process-wide and must not be initialised twice.
                    from sglang.srt.models.qwen4_exp_mtp import (
                        Qwen4ExpForCausalLMMTP,
                    )
                    mtp = Qwen4ExpForCausalLMMTP(
                        model_config.hf_config, quant_config, prefix="mtp"
                    )
    print(f"quantization: {type(quant_config).__name__ if quant_config else None}")
    # A meta construction should cost NO device memory. It used to cost 1.19 GiB,
    # all of it from three nn.Linear calls in layers/hyperconnection.py whose device
    # helper returned a CUDA index whenever CUDA was merely available, overriding
    # the ambient meta device -- which made every desk run race the standing serving
    # job for VRAM. Printed, not assumed, so a regression is visible here.
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 2**20
        print(f"peak VRAM allocated during construction: {peak:.1f} MiB")
    return model, mtp


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
    ap.add_argument(
        "--mtp",
        action="store_true",
        help="also build the MTP draft head and drive it with the same stream, so "
        "the checkpoint's `mtp.*` tensors are PROVEN rather than excused. Requires "
        "models/qwen4_exp_mtp.py to import.",
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

    # WHERE SHAPES COME FROM, and the order matters. [#1036] The local safetensors
    # HEADERS are the authority; the sidecar is only a cache for the case where the
    # files are not on disk yet (it was captured by HTTP Range request before the
    # download). A cache must never be the reason a valid checkpoint is rejected, and
    # it must never be the reason a WRONG dtype is fed.
    #
    # Both failure modes were real, one after the other, from the same fp8 PLE adoption:
    #   * the adoption ADDED one name (`weight_scale`) -> the sidecar had 296,474
    #     against a 296,475-name index, and the contract died with a FATAL that said
    #     nothing about the checkpoint. The instrument rejecting a valid tree.
    #   * the adoption REPOINTED 128 names at fp8 files while their sidecar entries
    #     still read BF16 -> the contract fed bf16 meta tensors for an fp8 store and
    #     the model warned "downcasting is lossy". A green exit over a wrong dtype,
    #     which is worse than the crash.
    shapes: dict[str, list] = {}
    from_header, unresolved, drift = 0, [], []
    for name, shard in weight_map.items():
        hdr = safetensors_header(osp.join(model_dir, shard))
        if hdr is not None and name in hdr:
            shapes[name] = [hdr[name]["dtype"], hdr[name]["shape"]]
            from_header += 1
        else:
            unresolved.append(name)

    shapes_path = args.shapes or osp.join(osp.dirname(args.index), "tensor_shapes.json")
    cached: dict[str, list] = {}
    if osp.exists(shapes_path):
        with open(shapes_path) as fh:
            cached = json.load(fh)
        # Fill only what no header could answer, and NAME the disagreements: a
        # sidecar that contradicts a header is drift, and drift is a finding.
        for name in list(unresolved):
            if name in cached:
                shapes[name] = cached[name]
                unresolved.remove(name)
        for name, val in shapes.items():
            if name in cached and cached[name] != val and len(drift) < 4000:
                drift.append(name)

    patterns = Counter(collapse(k) for k in weight_map)
    print(f"checkpoint: {len(weight_map)} tensors -> {len(patterns)} name patterns")
    print(f"shards:     {len(set(weight_map.values()))}")
    print(f"shapes:     {from_header} from local headers, "
          f"{len(shapes) - from_header} from the sidecar cache")
    if drift:
        ex = drift[0]
        print(f"  sidecar DRIFT on {len(drift)} name(s) -- headers win. e.g. {ex}")
        print(f"    cached {cached[ex]!r} vs header {shapes[ex]!r}")
    if unresolved:
        print(f"FATAL: {len(unresolved)} indexed tensors answered by neither a local "
              f"header nor the sidecar")
        for n in unresolved[:5]:
            print(f"  {n} -> {weight_map[n]}")
        return 2

    try:
        model, mtp = build_model(model_dir, raw_config, with_mtp=args.mtp)
    except Exception as exc:  # noqa: BLE001
        print(f"\nCOULD NOT CONSTRUCT the model ({type(exc).__name__}: {exc})")
        print("  The parameter-existence half of this contract did NOT run.")
        return 0 if args.allow_rules_only else 2

    params = dict(model.named_parameters())
    print(f"\nCONSTRUCTED on meta: {len(params)} parameters")
    if mtp is not None:
        print(
            f"CONSTRUCTED MTP draft head: "
            f"{len(dict(mtp.named_parameters()))} parameters"
        )

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

    # --- the draft head, as its own subject. Its loader takes the SAME full stream
    # and filters `if "mtp" not in name: continue`, so it is driven with the whole
    # weight map and judged only on the `mtp.*` slice. This is what turns those 31
    # names from "excused" into "proven", and it matters because the MTP loader has
    # its own silent skip (`if name_mapped not in params_dict: continue`,
    # qwen3_5_mtp.py:413-414) that only the parameter direction can catch.
    mtp_verdict = None
    if mtp is not None:
        m_consumed, m_fed, m_errors, m_skipped, m_params = drive_loader(
            mtp, weight_map, shapes, args.expert_cap
        )
        mtp_names = {n for n in weight_map if n.startswith("mtp.")} - m_skipped
        m_unconsumed = mtp_names - m_consumed
        m_unfed_all = m_params - m_fed
        m_unfed = {n for n in m_unfed_all if not excused(n, ALLOWED_UNFED_MTP)}
        m_excused = m_unfed_all - m_unfed
        print(
            f"\nMTP draft head: {len(m_consumed & mtp_names)}/{len(mtp_names)} "
            f"`mtp.*` tensors consumed, "
            f"{len(m_fed) + len(m_excused)}/{len(m_params)} parameters accounted for"
        )
        for n in sorted(m_excused):
            print(f"  declared-unfed: {n}\n      {excused(n, ALLOWED_UNFED_MTP)}")
        if m_errors:
            print(f"  MTP LOADER RAISED ({len(m_errors)}):")
            for name, (kind, msg) in list(m_errors.items())[:5]:
                print(f"    {collapse(name)}\n        {kind}: {msg}")
        if m_unconsumed:
            report("  MTP: `mtp.*` TENSORS NO PARAMETER CONSUMED", m_unconsumed, 10)
        if m_unfed:
            report("  MTP: PARAMETERS NO CHECKPOINT TENSOR FED", m_unfed, 10)
        mtp_verdict = not (m_errors or m_unconsumed or m_unfed)
        print(f"  MTP verdict: {'PROVEN' if mtp_verdict else 'FAILED'}")

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

    if errors or unconsumed or unfed or mtp_verdict is False:
        print("\nCONTRACT FAILED.")
        return 1

    print("\nOK: every checkpoint tensor reached a parameter, and every parameter")
    print("    was fed, through the model's OWN load_weights.")
    print(f"    mode=DRIVEN  tensors={len(offered)}  params={len(all_params)}")
    if mtp_verdict:
        print("    the MTP draft head's `mtp.*` tensors are PROVEN, not excused.")
    if args.expert_cap:
        print("    NOTE: --expert-cap was set, so this is a subset proof.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
