#!/usr/bin/env python3
"""Task #343: which layer stops matching, and on which forward step.

Reads two ``layers_rank*.jsonl`` traces written by
``sglang.srt.model_executor.layer_fingerprint`` and joins them on ``astep``
(the forward index counted from the arming point, so boot warmups cannot shift
one trace against the other) and on the tensor name.

Three verdicts per tensor, and the distinction matters:

  MATCH       identical sha256 over the fp32-cast bytes.
  DIFFER      same shape, different bytes. This is the finding.
  SHAPE       different shape, so not comparable. Under --rank-tp-ratio the
              per-rank attention head slice (``L<i>.attn_shard``) is
              deliberately a different width on every rank and against TP=1;
              flagging it as a deviation would be noise, not a result.

Where full tensors were dumped for the step (``full_rank*_astep*.pt``) the
exact max-abs delta is computed; otherwise the delta columns fall back to what
the JSONL carries -- the leading values and max|x| -- and are labelled as such,
because an unlabelled approximation is worse than no number.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

_ROLE_ORDER = {
    "input_ids": 0,
    "attn_shard": 1,
    "o_proj": 2,
    "mlp": 3,
    "out": 4,
}
_LAYER_RE = re.compile(r"^L(\d+)\.(\w+)")


def sort_key(name: str) -> Tuple[int, int, int, str]:
    """Forward order: input, embed, then layer by layer, then head."""
    if name == "input_ids":
        return (0, 0, 0, name)
    if name == "embed":
        return (1, 0, 0, name)
    match = _LAYER_RE.match(name)
    if match:
        return (2, int(match.group(1)), _ROLE_ORDER.get(match.group(2), 9), name)
    if name == "final_norm":
        return (3, 0, 0, name)
    if name.startswith("logits"):
        return (4, 0, 0, name)
    return (5, 0, 0, name)


def load(path: str, astep: int) -> Dict[str, dict]:
    rows: Dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("astep") == astep:
                # First occurrence wins: a name recorded twice in one forward
                # would be a hook registered twice, which is a harness bug and
                # must not be silently averaged away.
                rows.setdefault(row["name"], row)
    return rows


def full_path(dump_dir: str, rank: int, astep: int, name: str) -> str:
    safe = name.replace("/", "_")
    return os.path.join(dump_dir, f"full_rank{rank}_astep{astep:03d}_{safe}.pt")


def exact_delta(
    ref_dir: str, cmp_dir: str, ref_rank: int, cmp_rank: int, astep: int, name: str
) -> Optional[float]:
    a = full_path(ref_dir, ref_rank, astep, name)
    b = full_path(cmp_dir, cmp_rank, astep, name)
    if not (os.path.exists(a) and os.path.exists(b)):
        return None
    import torch

    ta = torch.load(a, map_location="cpu")
    tb = torch.load(b, map_location="cpu")
    if ta.shape != tb.shape:
        return None
    return float((ta - tb).abs().max().item())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="reference dump dir (TP=1)")
    ap.add_argument("--cmp", required=True, help="comparison dump dir")
    ap.add_argument("--ref-rank", type=int, default=0)
    ap.add_argument("--cmp-rank", type=int, default=0)
    ap.add_argument("--astep", type=int, default=1)
    ap.add_argument("--label", default="")
    ap.add_argument("--write", default="")
    ap.add_argument(
        "--all-rows",
        action="store_true",
        help="print every tensor, not only the first differing ones",
    )
    args = ap.parse_args()

    ref = load(os.path.join(args.ref, f"layers_rank{args.ref_rank}.jsonl"), args.astep)
    cmp_ = load(os.path.join(args.cmp, f"layers_rank{args.cmp_rank}.jsonl"), args.astep)

    lines: List[str] = []
    label = args.label or f"{args.ref} vs {args.cmp}"
    lines.append(f"# layer delta, astep={args.astep}: {label}")
    lines.append(f"# ref rank{args.ref_rank}: {len(ref)} tensors")
    lines.append(f"# cmp rank{args.cmp_rank}: {len(cmp_)} tensors")
    if not ref or not cmp_:
        lines.append("VOID: one side recorded nothing at this astep")
        print("\n".join(lines))
        return 1

    first_differ: Optional[str] = None
    n_match = n_differ = n_shape = n_missing = 0
    rows_out: List[str] = []
    for name in sorted(set(ref) | set(cmp_), key=sort_key):
        a, b = ref.get(name), cmp_.get(name)
        if a is None or b is None:
            n_missing += 1
            side = "ref" if a is None else "cmp"
            rows_out.append(f"  {name:24s} MISSING (absent on {side} side)")
            continue
        if a["shape"] != b["shape"]:
            n_shape += 1
            rows_out.append(
                f"  {name:24s} SHAPE   {a['shape']} vs {b['shape']} "
                "(per-rank shard, not comparable)"
            )
            continue
        if a["sha256"] == b["sha256"]:
            n_match += 1
            if args.all_rows:
                rows_out.append(f"  {name:24s} MATCH   shape={a['shape']}")
            continue
        n_differ += 1
        if first_differ is None:
            first_differ = name
        exact = exact_delta(
            args.ref, args.cmp, args.ref_rank, args.cmp_rank, args.astep, name
        )
        head_delta = max(
            (abs(x - y) for x, y in zip(a["head"], b["head"])), default=float("nan")
        )
        absmax_delta = abs(a["absmax"] - b["absmax"])
        # Relative to the reference tensor's own scale. An absolute delta is
        # unreadable across a decoder stack whose activation magnitudes span
        # three orders of magnitude between an o_proj output and a layer-1 MLP
        # output; "0.09" is noise on one and a defect on the other.
        scale = a["absmax"] or 1.0
        if exact is not None:
            delta = f"max|d|={exact:.4g}  rel={exact / scale:.3g}"
        else:
            delta = (
                f"max|d|(first {len(a['head'])} values)={head_delta:.4g}  "
                f"rel={head_delta / scale:.3g}  |absmax d|={absmax_delta:.4g}"
            )
        rows_out.append(f"  {name:20s} DIFFER  ref|absmax|={a['absmax']:.4g}  {delta}")

    lines.extend(rows_out)
    lines.append(
        f"# summary: {n_match} MATCH, {n_differ} DIFFER, {n_shape} SHAPE, "
        f"{n_missing} MISSING"
    )
    lines.append(f"# first differing tensor in forward order: {first_differ or 'none'}")
    text = "\n".join(lines)
    print(text)
    if args.write:
        os.makedirs(os.path.dirname(os.path.abspath(args.write)), exist_ok=True)
        with open(args.write, "w") as f:
            f.write(text + "\n")
    return 0 if first_differ is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
