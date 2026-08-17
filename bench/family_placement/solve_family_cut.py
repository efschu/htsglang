#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Solve the FAMILY-OBJECTIVE pipeline cut: GDN compute on the 5090, KV off it.

THE OBJECTIVE, as asked: put the compute-heavy linear/GDN layers (no KV) on the
big card and leave the full-attention layers (the KV carriers) on the small
ones. This script answers whether that is reachable, and at what price, using
only funded terms.

THE STRUCTURAL FACT THAT DECIDES IT, and it is arithmetic rather than opinion.
Pipeline stages are CONTIGUOUS layer ranges -- activations flow forward, so a
stage cannot be a scattered subset. The checkpoint interleaves full attention
uniformly at ``full_attention_interval = 4`` (FA at layer indices
3, 7, 11, ... 63). Therefore the number of FA layers in a stage is pinned by
how many layers it holds:

    max_layers_on_a_stage = 4 * (its FA count) + 3

Shedding one FA layer from the first stage costs it exactly FOUR layers, three
of them linear. "More GDN on the 5090" and "less KV on the 5090" are therefore
the SAME KNOB PULLED IN OPPOSITE DIRECTIONS, not two independent goals. That
tension is the whole problem, and this script prices it instead of narrating
it.

WHAT ``--pp-attn-stage-ratio`` ACTUALLY BUYS (#485). Without it, a stage-ratio
request snaps the boundary to a multiple of the period: scores 31,17,16 derive
[32,16,16] with FA [8,4,4]. With it, [31,17,16] with FA [7,5,4] is reachable --
one FA layer and one total layer off the first stage. So the decoupling is real
but bounded: it moves the boundary WITHIN a period, buying at most one FA layer
per boundary. It cannot break the 4:1 frontier above.

FUNDED TERMS ONLY. Weights are MEASURED from the checkpoint's safetensors index
via ``planner.pp_cut.checkpoint_weight_terms`` (the docstring there is explicit
that formula-derived attention layers are wrong by 30 MiB each on this family,
and the measured 355.1 MiB confirms it). KV per token per FA layer comes from
the serving kv-cache dtype. Anything this script cannot fund it prints as
UNFUNDED rather than estimating -- in particular per-stage FREE VRAM, which is
a boot artifact and is what decides absolute feasibility.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

MIB = 1024 * 1024

#: Serving default; every number scales linearly in this.
INCUMBENT_MAX_TOTAL_TOKENS = 436275
INCUMBENT_CUT = (31, 17, 16)
INCUMBENT_ATTN = (7, 5, 4)


@dataclass(frozen=True)
class Stage:
    layers: int
    fa: int
    gdn: int
    weight_mib: float
    kv_mib: float

    @property
    def total_mib(self) -> float:
        return self.weight_mib + self.kv_mib


@dataclass(frozen=True)
class Cut:
    bounds: Tuple[int, ...]
    stages: Tuple[Stage, ...]

    @property
    def layer_counts(self) -> Tuple[int, ...]:
        return tuple(s.layers for s in self.stages)

    @property
    def fa_counts(self) -> Tuple[int, ...]:
        return tuple(s.fa for s in self.stages)


def fa_positions(layer_types: Sequence[str]) -> Tuple[int, ...]:
    return tuple(i for i, t in enumerate(layer_types) if t == "full_attention")


def build_cut(
    boundaries: Sequence[int],
    n_layers: int,
    fa_idx: Sequence[int],
    *,
    attn_mib: float,
    gdn_mib: float,
    kv_mib_per_fa_layer: float,
) -> Cut:
    """Price one contiguous cut. ``boundaries`` are exclusive stage ends."""
    stages: List[Stage] = []
    prev = 0
    for end in boundaries:
        fa = sum(1 for x in fa_idx if prev <= x < end)
        layers = end - prev
        gdn = layers - fa
        stages.append(
            Stage(
                layers=layers,
                fa=fa,
                gdn=gdn,
                weight_mib=fa * attn_mib + gdn * gdn_mib,
                kv_mib=fa * kv_mib_per_fa_layer,
            )
        )
        prev = end
    return Cut(tuple(boundaries), tuple(stages))


def frontier(n_layers: int, fa_idx: Sequence[int]) -> dict:
    """max layers on stage 0 for each achievable stage-0 FA count."""
    best: dict = {}
    for b1 in range(1, n_layers):
        fa = sum(1 for x in fa_idx if x < b1)
        if fa not in best or b1 > best[fa]:
            best[fa] = b1
    return best


def enumerate_cuts(
    n_layers: int,
    fa_idx: Sequence[int],
    pp_size: int,
    **priced,
) -> List[Cut]:
    if pp_size != 3:
        raise ValueError("only the 3-stage rig geometry is priced here")
    out: List[Cut] = []
    for b1 in range(1, n_layers - 1):
        for b2 in range(b1 + 1, n_layers):
            out.append(build_cut((b1, b2, n_layers), n_layers, fa_idx, **priced))
    return out


def kv_mib_per_fa_layer(tokens: int, kv_bytes_per_token_per_attn_layer: int) -> float:
    return tokens * kv_bytes_per_token_per_attn_layer / MIB


def self_test() -> int:
    """Hermetic. The geometry is the real checkpoint's, stated inline."""
    failures: List[str] = []
    ran: List[str] = []

    def check(label: str, cond: bool) -> None:
        ran.append(label)
        if not cond:
            failures.append(label)

    n = 64
    lt = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(n)]
    fa = fa_positions(lt)
    check("16 full-attention layers", len(fa) == 16)
    check("period 4 places the first at index 3", fa[0] == 3)

    # THE FRONTIER. This is the load-bearing claim of the whole analysis.
    fr = frontier(n, fa)
    check("FA0=7 allows at most 31 layers", fr[7] == 31)
    check("FA0=6 allows at most 27 layers", fr[6] == 27)
    check("FA0=4 allows at most 19 layers", fr[4] == 19)
    check(
        "the frontier is exactly 4*FA+3",
        all(v == 4 * k + 3 for k, v in fr.items() if k < 15),
    )
    # The incumbent is ON the frontier: it is already the most GDN compute
    # obtainable for 7 FA layers. This is why "do the opposite" is not
    # available as a free move.
    check("the incumbent sits on the frontier", fr[7] == INCUMBENT_CUT[0])

    priced = dict(attn_mib=355.1, gdn_mib=476.1, kv_mib_per_fa_layer=852.0)
    inc = build_cut((31, 48, 64), n, fa, **priced)
    check("incumbent layer counts", inc.layer_counts == INCUMBENT_CUT)
    check("incumbent attn counts", inc.fa_counts == INCUMBENT_ATTN)
    check("stage0 holds 24 gdn layers", inc.stages[0].gdn == 24)

    # Shedding one FA layer from stage 0 costs four layers, three of them GDN.
    alt = build_cut((27, 48, 64), n, fa, **priced)
    check("stage0 loses exactly one FA layer", alt.stages[0].fa == inc.stages[0].fa - 1)
    check("and exactly four layers", alt.stages[0].layers == inc.stages[0].layers - 4)
    check("three of them linear", alt.stages[0].gdn == inc.stages[0].gdn - 3)

    moved = inc.stages[0].total_mib - alt.stages[0].total_mib
    check("the move is ~2.6 GiB off stage 0", 2500 < moved < 2700)
    # and it lands on a SMALLER card -- the whole tension, in one assertion.
    check(
        "all of it lands on stage 1",
        abs((alt.stages[1].total_mib - inc.stages[1].total_mib) - moved) < 1.0,
    )

    # GDN layers are HEAVIER than FA layers on this checkpoint, so
    # "concentrate GDN" also concentrates weight, not only compute.
    check("gdn layer is heavier than an fa layer", 476.1 > 355.1)

    # rejects
    check("a 1-layer stage still prices", build_cut((1, 2, n), n, fa, **priced).stages[0].layers == 1)
    try:
        enumerate_cuts(n, fa, 4, **priced)
        check("non-3-stage geometry is refused", False)
    except ValueError:
        check("non-3-stage geometry is refused", True)

    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print(f"self-test: OK ({len(ran)} checks)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--model", default="/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-yarn1.5")
    ap.add_argument("--tokens", type=int, default=INCUMBENT_MAX_TOTAL_TOKENS)
    ap.add_argument("--kv-bytes", type=int, default=2048)
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    try:
        from sglang.srt.planner.pp_cut import checkpoint_weight_terms
    except Exception as exc:
        print(f"cannot run: {exc}")
        return 2
    with open(f"{args.model}/config.json") as f:
        lt = json.load(f)["text_config"]["layer_types"]
    terms = checkpoint_weight_terms(args.model)
    fa = fa_positions(lt)
    n = len(lt)
    priced = dict(
        attn_mib=terms.attn_layer_weight_bytes / MIB,
        gdn_mib=terms.linear_layer_weight_bytes / MIB,
        kv_mib_per_fa_layer=kv_mib_per_fa_layer(args.tokens, args.kv_bytes),
    )

    print(f"model      {args.model}")
    print(f"layers     {n} ({len(fa)} full-attention, period {fa[1]-fa[0]})")
    print(f"FA layer   {priced['attn_mib']:.1f} MiB   GDN layer {priced['gdn_mib']:.1f} MiB "
          f"(GDN is {priced['gdn_mib']/priced['attn_mib']:.2f}x heavier)")
    print(f"KV/FA layer at {args.tokens} tokens: {priced['kv_mib_per_fa_layer']:.1f} MiB")
    print()
    fr = frontier(n, fa)
    print("FRONTIER -- max layers on stage 0 per stage-0 FA count (= 4*FA+3):")
    for k in sorted(fr):
        if k <= 8:
            print(f"   FA0={k:2d} -> at most {fr[k]:2d} layers")
    print()

    inc = build_cut((31, 48, n), n, fa, **priced)
    print("stage        layers  FA  GDN   weights MiB   KV MiB   total MiB")
    for label, cut in (("INCUMBENT", inc), ("FA0=6 alt", build_cut((27, 48, n), n, fa, **priced))):
        print(f"-- {label}  cut={cut.layer_counts} attn={cut.fa_counts}")
        for i, s in enumerate(cut.stages):
            print(f"   PP{i}        {s.layers:5d} {s.fa:3d} {s.gdn:4d} "
                  f"{s.weight_mib:12.1f} {s.kv_mib:8.1f} {s.total_mib:11.1f}")
    print()
    print("UNFUNDED here, and it decides absolute feasibility: per-stage FREE")
    print("VRAM. That is a boot artifact (residency census), not a checkpoint")
    print("property, so this script reports DELTAS and refuses to claim fit.")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "frontier": fr,
                    "incumbent": {
                        "cut": inc.layer_counts,
                        "attn": inc.fa_counts,
                        "stages": [s.__dict__ for s in inc.stages],
                    },
                },
                f,
                indent=2,
                default=float,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
