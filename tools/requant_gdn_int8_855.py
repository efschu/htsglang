#!/usr/bin/env python3
"""#855 — requant the GDN dense projections of a compressed-tensors W8A8
checkpoint from BF16 to int8, matching the checkpoint's OWN scheme.

Background
----------
`Qwen3.8-27B-INT8` is a compressed-tensors `int-quantized` checkpoint: int8
per-output-channel symmetric weights plus dynamic per-token int8 activations.
Its producer excluded `re:.*linear_attn.*`, so the 48 gated-delta-net layers'
dense projections stayed BF16 — 10,560 MiB of BF16 weights that the rest of the
model would have carried at 5,280 MiB.  ANALYSE_854 §3.3 showed the KV-cache win
everybody attributed to the W8A16 scheme is in fact this coverage gap, and
NOTE_855 §3.4 priced the speed half of it (BF16 GDN costs 1.39x/1.46x of prefill
linear time).  This tool closes the gap on the W8A8 lane.

Scope — exactly 144 projections, and the boundary is not arbitrary
-----------------------------------------------------------------
Quantized (3 families x 48 layers):

    linear_attn.in_proj_qkv.weight   (10240, 5120)   4800.0 MiB BF16
    linear_attn.in_proj_z.weight     ( 6144, 5120)   2880.0 MiB BF16
    linear_attn.out_proj.weight      ( 5120, 6144)   2880.0 MiB BF16
                                                  = 10560.0 MiB  -> 5280.0 MiB

Left alone: `in_proj_a` / `in_proj_b` (48x5120 gates, 22.5 MiB each), `conv1d`,
`norm`, `A_log`, `dt_bias`, and the separate embed/lm_head/vision axis (#727).

That split is forced by the runtime, not chosen for taste.  `qwen3_5.py`
:1283-1288 packs `in_proj_qkv + in_proj_z -> in_proj_qkvz` and
`in_proj_b + in_proj_a -> in_proj_ba`, and `should_ignore_layer`
(`compressed_tensors/utils.py:53-79`) raises `ValueError` if the shards of one
packed module disagree about being ignored.  So the only legal cut lines are
"all of qkvz" and "all of ba".  Quantizing qkv but not z would not be a quality
tradeoff, it would be a hard load failure.

Method — data-free RTN, because that is what the incumbent itself used
---------------------------------------------------------------------
The incumbent's `weights` group is `strategy: channel, symmetric: true,
num_bits: 8, dynamic: false, observer: memoryless_minmax`.  `memoryless_minmax`
per output channel IS round-to-nearest on amax/127 — no calibration data enters
it.  So requantizing the GDN projections the same way is not a cheaper
approximation of the incumbent's method, it is the identical method applied to
the tensors its producer skipped.  Scales are BF16 `[out, 1]`, matching the
companion layout already in the checkpoint.

Cost discipline
---------------
Per-output-channel scales make row blocking EXACT (each output row is
independent), so the blocked result is bit-identical to a whole-tensor one.
Only shards carrying a target are rewritten; the rest are hardlinked, so the new
directory costs ~the rewritten bytes, not another full copy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from typing import Dict, List, Optional, Sequence

# The three families that make up ANALYSE_854 §3.2's 10,560 MiB
# "GDN dense projections" row.  Anchored on `.weight` so a scale companion
# from a re-run can never be re-quantized.
TARGET_RE = re.compile(r"\.linear_attn\.(in_proj_qkv|in_proj_z|out_proj)\.weight$")

# The single ignore entry that must go, and what replaces it.  Dropping
# `re:.*linear_attn.*` outright would also expose in_proj_a/in_proj_b, whose
# int8 weights this tool does not produce -- the loader would then look for
# `in_proj_ba` scales that do not exist.  conv1d and norm are already covered by
# the checkpoint's own `re:.*conv1d.*` / `re:.*norm.*` entries, so they need no
# replacement of their own; a/b do.
IGNORE_DROP = "re:.*linear_attn.*"
IGNORE_ADD = [
    r"re:.*linear_attn\.in_proj_a.*",
    r"re:.*linear_attn\.in_proj_b.*",
]

DTYPE_BYTES = {"BF16": 2, "F16": 2, "F32": 4, "I8": 1, "I64": 8, "F8_E4M3": 1}


def quantize_per_channel_symmetric(weight):
    """Symmetric per-output-channel int8. Returns ``(int8, scale_bf16[out,1])``.

    Carried over verbatim from the #727 `requant_vocab_int8.py` quantizer -- the
    scheme is the same one, so it should be the same code.

    A zero row would divide by zero; its scale is clamped to a positive floor.
    Quantizing it to all-zeros is exact either way, but a NaN scale would poison
    the whole tensor on load.
    """
    import torch

    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D projection, got shape {tuple(weight.shape)}")

    # ROW-BLOCKED so the fp32 upcast never materializes the whole matrix.
    # Per-output-channel scales make the blocking EXACT, not approximate.
    rows = weight.shape[0]
    block = max(1, min(rows, 4096))
    q = torch.empty(weight.shape, dtype=torch.int8)
    scale = torch.empty((rows, 1), dtype=torch.bfloat16)
    tiny = torch.finfo(torch.float32).tiny
    for start in range(0, rows, block):
        stop = min(start + block, rows)
        w = weight[start:stop].to(torch.float32)
        amax = w.abs().amax(dim=1, keepdim=True)

        # The scale is STORED as bf16 (that is the checkpoint's companion
        # layout), so bf16 is the scale the runtime will dequantize with.
        # Quantizing against the fp32 scale and only then rounding the scale
        # down to bf16 leaves an error the ideal RTN bound does not cover:
        # bf16 carries 8 mantissa bits, so a scale off by up to 2^-9 relative
        # multiplies through a code of up to 127, i.e. ~0.25 extra steps, and
        # if the stored scale lands BELOW the true one the amax element also
        # clips. Both disappear by quantizing against the bf16 scale itself.
        #
        # The scale is additionally nudged up to the next bf16 value whenever
        # rounding put it below amax/127, which guarantees |w/s| <= 127 and so
        # makes the clamp below dead code rather than a silent error source.
        s = (amax / 127.0).clamp(min=tiny).to(torch.bfloat16)
        s_eff = s.to(torch.float32)
        low = s_eff * 127.0 < amax
        if bool(low.any()):
            bumped = (s_eff * (1.0 + 2.0**-8)).to(torch.bfloat16)
            s = torch.where(low, bumped, s)
            s_eff = s.to(torch.float32)

        q[start:stop] = torch.round(w / s_eff).clamp(-127, 127).to(torch.int8)
        scale[start:stop] = s
        del w, amax, s, s_eff, low
    return q, scale


def dequantize_per_channel(q, scale):
    """The inverse the runtime applies. Kept here so the tool can prove its own
    round trip rather than assert it."""
    return q.to(scale.dtype).mul(scale)


def tensor_error_stats(weight, q, scale) -> Dict[str, float]:
    """Round-trip error + the outlier statistic that decides whether data-free
    RTN is defensible for this tensor.

    `outlier_ratio` = max over channels of (channel amax / channel rms).  It is
    the quantity that governs RTN damage: the step is amax/127, so a channel
    whose energy sits far below its peak spends its 8 bits on empty range.
    """
    import torch

    w = weight.to(torch.float32)
    dq = dequantize_per_channel(q, scale.to(torch.float32))
    err = w - dq
    wn = torch.linalg.vector_norm(w).item()
    en = torch.linalg.vector_norm(err).item()
    rel_fro = en / wn if wn > 0 else 0.0

    amax = w.abs().amax(dim=1)
    rms = w.pow(2).mean(dim=1).sqrt()
    crest = (amax / rms.clamp(min=torch.finfo(torch.float32).tiny))

    # Bounded by 1/254 = 0.003937 for correct RTN; a violation means a bug,
    # not bad data.
    step = (scale.to(torch.float32)).squeeze(1)
    max_norm_err = (err.abs().amax(dim=1) / step.clamp(min=torch.finfo(torch.float32).tiny)).amax().item()

    return {
        "rel_fro": rel_fro,
        "snr_db": (20.0 * torch.log10(torch.tensor(1.0 / rel_fro)).item()) if rel_fro > 0 else float("inf"),
        "max_err_over_step": max_norm_err,
        "crest_max": crest.amax().item(),
        "crest_median": crest.median().item(),
        "amax_max": amax.amax().item(),
        "amax_median": amax.median().item(),
        "amax_ratio": (amax.amax() / amax.median().clamp(min=torch.finfo(torch.float32).tiny)).item(),
    }


def family_of(name: str) -> str:
    m = TARGET_RE.search(name)
    return m.group(1) if m else "?"


def rewrite_ignore(ignore: Sequence[str]) -> List[str]:
    """Swap the blanket linear_attn exclusion for the two gate-only entries."""
    if IGNORE_DROP not in ignore:
        raise ValueError(
            f"expected {IGNORE_DROP!r} in the ignore list, found {list(ignore)!r}; "
            "refusing to guess what this checkpoint excludes."
        )
    out: List[str] = []
    for entry in ignore:
        if entry == IGNORE_DROP:
            out.extend(IGNORE_ADD)
        else:
            out.append(entry)
    return out


def _header(path: str):
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def index_total_size(directory: str, weight_map: Dict[str, str]) -> int:
    """Recompute `metadata.total_size` from the shards actually written."""
    total = 0
    for shard in sorted(set(weight_map.values())):
        for name, info in _header(os.path.join(directory, shard)).items():
            if name == "__metadata__":
                continue
            n = 1
            for d in info["shape"]:
                n *= d
            total += n * DTYPE_BYTES[info["dtype"]]
    return total


def self_test() -> int:
    """Hermetic: synthetic tensors, no checkpoint, no CUDA."""
    failures: List[str] = []
    ran: List[str] = []

    def check(label: str, cond: bool) -> None:
        ran.append(label)
        if not cond:
            failures.append(label)

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(f"self-test cannot run: {exc}")
        return 2

    torch.manual_seed(0)

    # 1. Round trip is within the RTN bound, and the bound is the real one.
    w = torch.randn(64, 128, dtype=torch.bfloat16)
    q, s = quantize_per_channel_symmetric(w)
    check("dtype int8", q.dtype == torch.int8)
    check("scale bf16", s.dtype == torch.bfloat16)
    check("scale shape", tuple(s.shape) == (64, 1))
    check("shape preserved", tuple(q.shape) == (64, 128))
    check("int8 range", int(q.abs().amax()) <= 127)
    st = tensor_error_stats(w, q, s)
    check("err within RTN bound", st["max_err_over_step"] <= 0.5 + 1e-3)

    # 2. Row blocking is EXACT, not approximate -- the load-bearing claim that
    #    lets this run without an 8 GiB upcast spike. Compare a blocked run
    #    against a deliberately unblocked reference.
    big = torch.randn(9000, 64, dtype=torch.bfloat16)
    qb, sb = quantize_per_channel_symmetric(big)
    bigf = big.to(torch.float32)
    ref_amax = bigf.abs().amax(dim=1, keepdim=True)
    ref_s = (ref_amax / 127.0).clamp(min=torch.finfo(torch.float32).tiny).to(torch.bfloat16)
    ref_eff = ref_s.to(torch.float32)
    ref_low = ref_eff * 127.0 < ref_amax
    ref_s = torch.where(ref_low, (ref_eff * (1.0 + 2.0**-8)).to(torch.bfloat16), ref_s)
    ref_q = torch.round(bigf / ref_s.to(torch.float32)).clamp(-127, 127).to(torch.int8)
    check("blocking bit-identical (q)", torch.equal(qb, ref_q))
    check("blocking bit-identical (s)", torch.equal(sb, ref_s))

    # The nudge must make the clamp dead code: no channel may saturate because
    # its scale rounded down. A saturating amax element is a silent error the
    # ideal-RTN bound would not reveal.
    check("no clip: scale covers amax", bool((sb.to(torch.float32) * 127.0 >= ref_amax).all()))
    stb = tensor_error_stats(big, qb, sb)
    check("blocked err within bound", stb["max_err_over_step"] <= 0.5 + 1e-4)

    # 3. A zero row does not produce NaN.
    z = torch.zeros(4, 32, dtype=torch.bfloat16)
    z[1] = 1.0
    qz, sz = quantize_per_channel_symmetric(z)
    check("zero row finite", bool(torch.isfinite(sz.to(torch.float32)).all()))
    check("zero row exact", int(qz[0].abs().amax()) == 0)

    # 4. Target regex selects exactly the three families and nothing else.
    names = [
        "model.language_model.layers.3.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.3.linear_attn.in_proj_z.weight",
        "model.language_model.layers.3.linear_attn.out_proj.weight",
        "model.language_model.layers.3.linear_attn.in_proj_a.weight",
        "model.language_model.layers.3.linear_attn.in_proj_b.weight",
        "model.language_model.layers.3.linear_attn.conv1d.weight",
        "model.language_model.layers.3.linear_attn.norm.weight",
        "model.language_model.layers.3.linear_attn.A_log",
        "model.language_model.layers.3.self_attn.o_proj.weight",
        "model.language_model.layers.3.mlp.down_proj.weight",
        # a scale companion must never be re-selected on a second run
        "model.language_model.layers.3.linear_attn.out_proj.weight_scale",
    ]
    hit = [n for n in names if TARGET_RE.search(n)]
    check("regex selects 3", len(hit) == 3)
    check("regex skips gates", not any("in_proj_a" in n or "in_proj_b" in n for n in hit))
    check("regex skips scale companion", not any(n.endswith("_scale") for n in hit))
    check("regex families", {family_of(n) for n in hit} == {"in_proj_qkv", "in_proj_z", "out_proj"})

    # 5. Ignore-list surgery keeps every non-GDN exclusion and both gates.
    src_ignore = [
        "re:.*(vision|visual).*",
        "lm_head",
        "re:.*embed_tokens.*",
        "re:.*norm.*",
        "re:.*conv1d.*",
        "re:.*linear_attn.*",
    ]
    new = rewrite_ignore(src_ignore)
    check("ignore drops blanket", IGNORE_DROP not in new)
    check("ignore keeps vision", "re:.*(vision|visual).*" in new)
    check("ignore keeps lm_head", "lm_head" in new)
    check("ignore keeps embed", "re:.*embed_tokens.*" in new)
    check("ignore keeps norm", "re:.*norm.*" in new)
    check("ignore keeps conv1d", "re:.*conv1d.*" in new)
    check("ignore adds gates", all(a in new for a in IGNORE_ADD))
    try:
        rewrite_ignore(["lm_head"])
        check("ignore refuses unknown list", False)
    except ValueError:
        check("ignore refuses unknown list", True)

    # 6. THE decisive semantic test: replay the runtime's own matcher over the
    #    new ignore list. This is what makes the packed-module boundary a proven
    #    property rather than a comment.
    def is_ignored(layer: str, ignore: Sequence[str]) -> bool:
        for t in ignore:
            if t.startswith("re:"):
                if re.match(t[3:], layer):
                    return True
            elif t.lower() in layer.lower():
                return True
        return False

    base = "model.language_model.layers.3.linear_attn."
    check("qkv not ignored", not is_ignored(base + "in_proj_qkv", new))
    check("z not ignored", not is_ignored(base + "in_proj_z", new))
    check("out_proj not ignored", not is_ignored(base + "out_proj", new))
    check("a still ignored", is_ignored(base + "in_proj_a", new))
    check("b still ignored", is_ignored(base + "in_proj_b", new))
    check("conv1d still ignored", is_ignored(base + "conv1d", new))
    check("norm still ignored", is_ignored(base + "norm", new))
    # packed-module agreement: both shards of each packed module must agree,
    # or compressed_tensors/utils.py:74-79 raises.
    check(
        "packed in_proj_qkvz agrees",
        is_ignored(base + "in_proj_qkv", new) == is_ignored(base + "in_proj_z", new),
    )
    check(
        "packed in_proj_ba agrees",
        is_ignored(base + "in_proj_a", new) == is_ignored(base + "in_proj_b", new),
    )
    # the gate patterns must not leak onto the packed sibling names
    check("gate pattern misses qkv", not is_ignored(base + "in_proj_qkv", IGNORE_ADD))
    check("non-GDN untouched", not is_ignored("model.language_model.layers.3.mlp.down_proj", new))
    check("vision still ignored", is_ignored("model.visual.blocks.0.attn.qkv", new))

    for label in ran:
        print(f"  {'FAIL' if label in failures else 'ok  '}  {label}")
    print(f"\n{len(ran) - len(failures)}/{len(ran)} passed")
    return 1 if failures else 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="requant GDN dense projections to int8 (#855)")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--dry-run", action="store_true", help="plan only, write nothing")
    ap.add_argument("--stats-out", help="write per-tensor error stats JSON here")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not (args.src and args.dst):
        ap.print_help()
        return 2

    try:
        import torch  # noqa: F401
        from safetensors import safe_open
        from safetensors.torch import save_file
    except Exception as exc:
        print(f"cannot run: {exc}")
        return 2

    index_path = os.path.join(args.src, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)
    weight_map: Dict[str, str] = index["weight_map"]

    targets = sorted(n for n in weight_map if TARGET_RE.search(n))
    if len(targets) != 144:
        print(f"cannot run: expected 144 GDN dense projections, found {len(targets)}")
        return 2
    target_set = set(targets)
    touched = {weight_map[n] for n in targets}
    all_shards = set(weight_map.values())

    print(f"targets       : {len(targets)} tensors")
    for fam in ("in_proj_qkv", "in_proj_z", "out_proj"):
        print(f"  {fam:14s} {sum(1 for n in targets if family_of(n) == fam)}")
    print(f"shards to redo: {len(touched)}")
    print(f"shards linked : {len(all_shards - touched)}")

    with open(os.path.join(args.src, "config.json")) as f:
        config = json.load(f)
    qc = config.get("quantization_config", {})
    before = list(qc.get("ignore", []))
    try:
        after = rewrite_ignore(before)
    except ValueError as exc:
        print(f"cannot run: {exc}")
        return 2
    print(f"ignore: {before}\n     -> {after}")

    if args.dry_run:
        print("dry run: nothing written")
        return 0

    os.makedirs(args.dst, exist_ok=True)
    stats: Dict[str, Dict[str, float]] = {}
    done = 0

    for shard in sorted(all_shards):
        src_shard = os.path.join(args.src, shard)
        dst_shard = os.path.join(args.dst, shard)
        if os.path.exists(dst_shard):
            os.unlink(dst_shard)
        if shard not in touched:
            try:
                os.link(src_shard, dst_shard)
            except OSError:
                shutil.copy2(src_shard, dst_shard)
            print(f"[link] {shard}")
            continue

        tensors = {}
        with safe_open(src_shard, framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if name in target_set:
                    q, s = quantize_per_channel_symmetric(t)
                    stats[name] = tensor_error_stats(t, q, s)
                    tensors[name] = q
                    tensors[name + "_scale"] = s
                    done += 1
                    del t
                else:
                    tensors[name] = t
        save_file(tensors, dst_shard, metadata={"format": "pt"})
        del tensors
        print(f"[redo] {shard}  ({done}/144 quantized)")

    # Index: scale companions are new entries, and total_size must be recomputed
    # from what was actually written rather than adjusted arithmetically.
    for name in targets:
        weight_map[name + "_scale"] = weight_map[name]
    index["weight_map"] = weight_map
    index.setdefault("metadata", {})["total_size"] = index_total_size(args.dst, weight_map)
    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    qc["ignore"] = after
    config["quantization_config"] = qc
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    for extra in os.listdir(args.src):
        if extra in ("config.json", "model.safetensors.index.json"):
            continue
        if extra.endswith(".safetensors"):
            continue
        s, d = os.path.join(args.src, extra), os.path.join(args.dst, extra)
        if os.path.isfile(s) and not os.path.exists(d):
            try:
                os.link(s, d)
            except OSError:
                shutil.copy2(s, d)

    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump(stats, f, indent=1)
        print(f"stats: {args.stats_out}")

    print(f"total_size: {index['metadata']['total_size']} B")
    print(f"written: {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
