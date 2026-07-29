"""Pre-boot predicate for a dual-group lane plan (#274 families slice).

Answers, on the CPU and without a model load, two questions that otherwise
cost a boot:

1. Does a candidate ``(base, mlp, moe, vocab)`` ratio quadruple NEST for this
   model's unit counts?  (the 65-of-497 class from DESIGN_121 §3.3)
2. What does each BIG rank actually hold, so the VRAM fixpost calculation can
   be done before the cards are touched?

Usage:
    python scripts/dual_group/lane_plan_probe.py <hf-config-dir> \
        --base 2,1,1 [--mlp 6,1,1] [--vocab 6,1,1] [--moe 6,1,1]
    python scripts/dual_group/lane_plan_probe.py <hf-config-dir> --search
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.distributed.dual_group import (  # noqa: E402
    NestedGroupPlan,
    derive_nested_plan,
    nesting_failures,
    transformer_nesting_probes,
)
from sglang.srt.distributed.utils import ACTIVATION_VEC_ELEMS  # noqa: E402


def read_cfg(path: str) -> dict:
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def geometry(cfg: dict) -> dict:
    vocab = cfg.get("vocab_size")
    return {
        "num_attention_heads": cfg["num_attention_heads"],
        "num_kv_heads": cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        "intermediate_size": cfg.get("intermediate_size"),
        "num_experts": cfg.get("num_experts"),
        "linear_attn_units": cfg.get("gdn_tp_units") or cfg.get("linear_num_key_heads"),
        "vocab_units": (vocab + 63) // 64 if vocab else None,
    }


def make_plan(base, fams) -> NestedGroupPlan:
    plan = derive_nested_plan(tuple(base))
    return NestedGroupPlan(
        big_ratio=plan.big_ratio,
        segments=plan.segments,
        family_ratios=tuple((n, tuple(v)) for n, v in fams if v),
    )


def check(cfg: dict, base, fams) -> list:
    plan = make_plan(base, fams)
    probes = transformer_nesting_probes(plan, **geometry(cfg))
    return nesting_failures(plan, probes)


def _parse_vec(s):
    return [int(x) for x in s.split(",")] if s else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--base", default="2,1,1")
    ap.add_argument("--mlp", default=None)
    ap.add_argument("--moe", default=None)
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--search", action="store_true")
    ap.add_argument("--max-weight", type=int, default=8)
    args = ap.parse_args()

    cfg = read_cfg(args.model)
    geo = geometry(cfg)
    print(f"model   : {args.model}")
    print(f"geometry: {geo}")
    if geo["intermediate_size"]:
        i = geo["intermediate_size"]
        print(f"  mlp units = {i // math.gcd(i, ACTIVATION_VEC_ELEMS)}")

    if not args.search:
        base = _parse_vec(args.base)
        fams = [
            ("mlp", _parse_vec(args.mlp)),
            ("moe", _parse_vec(args.moe)),
            ("vocab", _parse_vec(args.vocab)),
        ]
        fails = check(cfg, base, fams)
        plan = make_plan(base, fams)
        print(f"plan    : {plan.describe()}")
        if fails:
            print("NESTING: FAIL")
            for f in fails:
                print("  " + f)
            return 1
        print("NESTING: OK")
        return 0

    # Search: base vectors of length 3 with rank 0 first (shared rank), plus a
    # matching mlp/vocab vector carrying the same shape.
    n = 3
    hits = []
    rng = range(1, args.max_weight + 1)
    for base in itertools.product(rng, repeat=n):
        if math.gcd(*base) != 1:
            continue
        for fam in [None, *itertools.product(rng, repeat=n)]:
            if fam is not None and math.gcd(*fam) != 1:
                continue
            fams = (
                []
                if fam is None
                else [("mlp", list(fam)), ("moe", list(fam)), ("vocab", list(fam))]
            )
            try:
                if not check(cfg, list(base), fams):
                    hits.append((list(base), None if fam is None else list(fam)))
            except Exception:  # pragma: no cover - search is best-effort
                continue
    print(f"{len(hits)} nesting quadruples found (base, family):")
    for b, f in hits[:60]:
        print(f"  base={b} fam={f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
