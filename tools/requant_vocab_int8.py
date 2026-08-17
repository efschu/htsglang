#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#727: requantize embed_tokens / lm_head to INT8 in a NEW checkpoint dir.

WHY THE CHECKPOINT HAS TO CHANGE. On the served
``Qwen3.8-27B-INT8`` family the vocab weights are BF16 by the producer's own
instruction: ``quantization_config.ignore`` carries ``lm_head`` and
``re:.*embed_tokens.*``, and the safetensors headers agree
(``lm_head.weight`` and ``model.language_model.embed_tokens.weight`` are both
BF16 ``[248320, 5120]``). So there is nothing to dequant-on-gather today and no
wiring change can produce the saving -- the bytes are simply not quantized.
This tool produces a checkpoint where they are.

THE PRIZE, exactly. 248320 x 5120 = 1,271,398,400 elements per tensor:

    BF16   2,542,796,800 B = 2425.0 MiB
    INT8   1,271,398,400 B = 1212.5 MiB  + scale 248320 x 2 B = 0.5 MiB
    saving                   1212.0 MiB per tensor

Under the serving geometry (``--tp-size 1 --pp-size 3``) the two tensors do NOT
land on the same rank: ``embed_tokens`` lives on the FIRST stage and
``lm_head`` on the LAST, so this is 1212 MiB off PP0 and 1212 MiB off PP2, not
2424 MiB off one card. Quote it per stage.

THE TWO HALVES CARRY DIFFERENT RISK, and this tool keeps them separable
(``--targets``) for that reason:

* ``embed_tokens`` is a GATHER. Per-row (per-vocab-token) scales make dequant
  exact per row and cost one multiply on the few rows a batch touches. The
  result feeds a layernorm, which absorbs a per-row scale error. LOW risk.
* ``lm_head`` is a GEMM producing LOGITS directly. Per-output-channel scales
  put a ~0.4% relative error on each logit, and softmax/argmax care about
  logit DIFFERENCES, so near-ties can flip. This is the half the producer's
  ignore list was plausibly protecting, and it is the half that needs the
  quality A/B before it ships.

FORMAT IS COPIED FROM THE CHECKPOINT, NOT INVENTED. The existing quantized
linears in this very checkpoint carry ``weight`` as I8 ``[out, in]`` beside
``weight_scale`` as BF16 ``[out, 1]``, and ``config_groups`` declares
``strategy: channel``, ``symmetric: true``, ``num_bits: 8``, ``type: int``,
``observer: memoryless_minmax``. This tool reproduces exactly that: symmetric
per-output-channel min-max int8.

DISK. Unchanged shards are HARD-LINKED, not copied -- a full copy would be
~28 GiB and this filesystem runs at 92% . Only the shards that actually contain
a target tensor are rewritten. Hardlinks mean the new directory costs roughly
the size of those shards alone; it also means the source must not be mutated in
place afterwards (it is a read-only cache, so that holds).

USAGE

    python tools/requant_vocab_int8.py --self-test           # hermetic
    python tools/requant_vocab_int8.py \\
        --src /spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-yarn1.5 \\
        --dst /spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-vocabint8 \\
        --targets embed            # or: lm_head, or: embed,lm_head

Exit: 0 = written, 1 = a check failed, 2 = could not run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

MIB = 1024 * 1024

#: Suffix-matched so the tool does not depend on the model's prefix layout.
TARGET_SUFFIXES = {
    "embed": "embed_tokens.weight",
    "lm_head": "lm_head.weight",
}

#: What must be removed from ``quantization_config.ignore`` for a requantized
#: tensor to be READ as quantized. Leaving the entry in place would produce a
#: checkpoint whose bytes are int8 and whose config says they are not -- the
#: worst of both, and silent.
IGNORE_PATTERNS = {
    "embed": ("re:.*embed_tokens.*",),
    "lm_head": ("lm_head",),
}


@dataclass(frozen=True)
class QuantResult:
    name: str
    elements: int
    bf16_mib: float
    int8_mib: float

    @property
    def saved_mib(self) -> float:
        return self.bf16_mib - self.int8_mib


def quantize_per_channel_symmetric(weight):
    """Symmetric per-output-channel int8, matching the checkpoint's scheme.

    Returns ``(int8_weight, scale_bf16)`` with scale shaped ``[out, 1]``.

    A zero row (a never-used vocab slot is possible in a padded vocab) would
    divide by zero, so its scale is clamped to a positive floor. Quantizing it
    to all-zeros is exact for that row either way, but a NaN scale would
    poison the whole tensor on load.
    """
    import torch

    if weight.dim() != 2:
        raise ValueError(f"expected a 2-D vocab matrix, got shape {tuple(weight.shape)}")

    # ROW-BLOCKED so the fp32 upcast does not materialize the whole matrix.
    # A [248320, 5120] bf16 vocab tensor is 2.4 GiB and its fp32 view is
    # 4.9 GiB; doing it in one shot spikes ~8 GiB beside a live serving
    # process. Per-output-channel scales make the blocking exact -- each row
    # is independent, so a blocked result is bit-identical to the whole-tensor
    # one, not an approximation of it.
    rows = weight.shape[0]
    block = max(1, min(rows, 8192))
    q = torch.empty_like(weight, dtype=torch.int8)
    scale = torch.empty((rows, 1), dtype=torch.float32)
    tiny = torch.finfo(torch.float32).tiny
    for start in range(0, rows, block):
        stop = min(start + block, rows)
        w = weight[start:stop].to(torch.float32)
        amax = w.abs().amax(dim=1, keepdim=True)
        s = (amax / 127.0).clamp(min=tiny)
        q[start:stop] = torch.round(w / s).clamp(-127, 127).to(torch.int8)
        scale[start:stop] = s
        del w, amax, s
    return q, scale.to(torch.bfloat16)


def dequantize_per_channel(q, scale):
    """The inverse the runtime must apply. Kept here so the tool can prove
    its own round trip rather than asserting it."""
    return q.to(scale.dtype).mul(scale)


def resolve_targets(spec: str) -> List[str]:
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part not in TARGET_SUFFIXES:
            raise ValueError(
                f"unknown target {part!r}; choose from {sorted(TARGET_SUFFIXES)}"
            )
        out.append(part)
    if not out:
        raise ValueError("no targets given")
    return out


def find_target_tensors(weight_map: Dict[str, str], targets: Sequence[str]):
    """Map each requested target to the ONE tensor name that carries it."""
    found: Dict[str, str] = {}
    for target in targets:
        suffix = TARGET_SUFFIXES[target]
        names = [n for n in weight_map if n == suffix or n.endswith("." + suffix)]
        if len(names) != 1:
            raise ValueError(
                f"target {target!r} matched {len(names)} tensors ({names!r}); "
                "expected exactly one -- refusing rather than guessing."
            )
        found[target] = names[0]
    return found


def strip_ignore(ignore: Sequence[str], targets: Sequence[str]) -> List[str]:
    """Remove the entries that would keep a requantized tensor unread."""
    drop = set()
    for t in targets:
        drop.update(IGNORE_PATTERNS[t])
    return [entry for entry in ignore if entry not in drop]


def _shard_of(weight_map: Dict[str, str], name: str) -> str:
    return weight_map[name]


def self_test() -> int:
    """Hermetic. Small synthetic tensors -- the real ones are 2.4 GiB each."""
    failures: List[str] = []
    ran: List[str] = []

    def check(label: str, cond: bool) -> None:
        ran.append(label)
        if not cond:
            failures.append(label)

    try:
        import torch
    except Exception:
        print("cannot self-test: torch unavailable")
        return 2

    # -- quantization is symmetric, per row, and round-trips within its step
    torch.manual_seed(0)
    w = torch.randn(64, 32, dtype=torch.bfloat16) * 3.0
    q, s = quantize_per_channel_symmetric(w)
    check("int8 dtype", q.dtype == torch.int8)
    check("shape preserved", tuple(q.shape) == (64, 32))
    check("scale is per output channel", tuple(s.shape) == (64, 1))
    check("scale is bf16 like the checkpoint's", s.dtype == torch.bfloat16)
    check("no value exceeds the symmetric range", int(q.abs().max()) <= 127)

    deq = dequantize_per_channel(q, s.to(torch.float32))
    err = (deq - w.to(torch.float32)).abs()
    rowmax = w.to(torch.float32).abs().amax(dim=1, keepdim=True)
    # Symmetric int8 has a half-step of amax/254; allow one step for rounding.
    check(
        "round trip is within one quantization step",
        bool((err <= (rowmax / 127.0) * 1.01 + 1e-6).all()),
    )
    # The largest-magnitude entry of each row must land on the rail, which is
    # what makes the scale minmax rather than arbitrary.
    check("row maxima saturate the rail", int(q.abs().amax(dim=1).min()) == 127)

    # -- a zero row must not produce NaN
    wz = torch.zeros(3, 8, dtype=torch.bfloat16)
    qz, sz = quantize_per_channel_symmetric(wz)
    check("zero row quantizes to zero", int(qz.abs().max()) == 0)
    check("zero row scale is finite", bool(torch.isfinite(sz).all()))

    # -- a non-2D input is refused rather than silently flattened
    try:
        quantize_per_channel_symmetric(torch.zeros(4, dtype=torch.bfloat16))
        check("1-D input is refused", False)
    except ValueError:
        check("1-D input is refused", True)

    # -- target resolution
    wm = {
        "model.language_model.embed_tokens.weight": "a.safetensors",
        "lm_head.weight": "b.safetensors",
        "model.language_model.layers.0.mlp.down_proj.weight": "c.safetensors",
    }
    got = find_target_tensors(wm, ["embed", "lm_head"])
    check("embed resolves", got["embed"] == "model.language_model.embed_tokens.weight")
    check("lm_head resolves", got["lm_head"] == "lm_head.weight")
    check("targets can be requested separately", list(find_target_tensors(wm, ["embed"])) == ["embed"])
    try:
        find_target_tensors({}, ["embed"])
        check("a missing target is refused", False)
    except ValueError:
        check("a missing target is refused", True)
    try:
        resolve_targets("nonsense")
        check("an unknown target name is refused", False)
    except ValueError:
        check("an unknown target name is refused", True)

    # -- the ignore list must actually lose the entry, or the bytes are int8
    #    while the config still says they are not
    ig = [
        "re:.*(vision|visual).*",
        "lm_head",
        "re:.*embed_tokens.*",
        "re:.*norm.*",
    ]
    check(
        "embed-only strip leaves lm_head ignored",
        strip_ignore(ig, ["embed"]) == ["re:.*(vision|visual).*", "lm_head", "re:.*norm.*"],
    )
    check(
        "both strips remove both",
        strip_ignore(ig, ["embed", "lm_head"])
        == ["re:.*(vision|visual).*", "re:.*norm.*"],
    )
    check("unrelated entries survive", "re:.*norm.*" in strip_ignore(ig, ["embed"]))

    # -- the byte accounting the ticket quotes
    els = 248320 * 5120
    bf16 = els * 2 / MIB
    int8 = els * 1 / MIB
    check("bf16 vocab tensor is 2425.0 MiB", abs(bf16 - 2425.0) < 0.5)
    check("int8 vocab tensor is 1212.5 MiB", abs(int8 - 1212.5) < 0.5)
    check("the saving is ~1212 MiB per tensor", abs((bf16 - int8) - 1212.5) < 0.5)

    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    rejects = sum(1 for x in ran if "refused" in x)
    print(f"self-test: OK ({len(ran)} checks, {rejects} asserting a refusal)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--targets", default="embed")
    ap.add_argument("--dry-run", action="store_true", help="plan only, write nothing")
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

    try:
        targets = resolve_targets(args.targets)
    except ValueError as exc:
        print(f"cannot run: {exc}")
        return 2

    index_path = os.path.join(args.src, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        print(f"cannot run: no index at {index_path}")
        return 2
    with open(index_path) as f:
        index = json.load(f)
    weight_map: Dict[str, str] = index["weight_map"]

    try:
        found = find_target_tensors(weight_map, targets)
    except ValueError as exc:
        print(f"cannot run: {exc}")
        return 2

    touched_shards = {_shard_of(weight_map, n) for n in found.values()}
    print(f"targets       : {found}")
    print(f"shards to redo: {sorted(touched_shards)}")
    print(f"shards linked : {len(set(weight_map.values()) - touched_shards)}")
    if args.dry_run:
        print("dry run: nothing written")
        return 0

    os.makedirs(args.dst, exist_ok=True)

    # Link every untouched shard; rewrite only the ones carrying a target.
    for shard in sorted(set(weight_map.values())):
        src_shard = os.path.join(args.src, shard)
        dst_shard = os.path.join(args.dst, shard)
        if os.path.exists(dst_shard):
            os.unlink(dst_shard)
        if shard not in touched_shards:
            try:
                os.link(src_shard, dst_shard)
            except OSError:
                shutil.copy2(src_shard, dst_shard)
            continue

        tensors = {}
        with safe_open(src_shard, framework="pt") as f:
            for name in f.keys():
                t = f.get_tensor(name)
                if name in found.values():
                    q, s = quantize_per_channel_symmetric(t)
                    tensors[name] = q
                    tensors[name + "_scale"] = s
                    print(f"  quantized {name}: {tuple(t.shape)} bf16 -> int8")
                else:
                    tensors[name] = t
        save_file(tensors, dst_shard, metadata={"format": "pt"})

    # Index: the scale companions are new entries.
    for name in found.values():
        weight_map[name + "_scale"] = weight_map[name]
    index["weight_map"] = weight_map
    with open(os.path.join(args.dst, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    # Config: drop the ignore entries, or the bytes are int8 while the config
    # still declares them excluded.
    with open(os.path.join(args.src, "config.json")) as f:
        config = json.load(f)
    qc = config.get("quantization_config", {})
    before = list(qc.get("ignore", []))
    qc["ignore"] = strip_ignore(before, targets)
    config["quantization_config"] = qc
    with open(os.path.join(args.dst, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"ignore: {before} -> {qc['ignore']}")

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
    print(f"written: {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
