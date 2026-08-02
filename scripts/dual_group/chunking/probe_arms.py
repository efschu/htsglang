#!/usr/bin/env python3
"""#274 §13.10 measurement window: chunked lane prefill, priced and gated.

One boot, arms from per-job overrides (the enqueue whitelist carries
``prefill_chunk``), so every comparison is same-boot:

* REFERENCE  chunk 0 (single whole-prompt forward), ``--ref-draws`` times.
  The draws are the instrument's own A-vs-A band (#328/#363 discipline):
  the GDN prefill is not reproducible past ~109 tokens on this family, so a
  single draw is a sample, not a reference, and byte identity is never the
  criterion by itself.
* CHUNK ARMS one job per requested chunk size, spec off and (if the boot has
  a head) spec on.

Three verdicts per arm, separated because they fail for different reasons:

* STRUCTURE (hard): the row must carry ceil(n/chunk) chunk timings summing
  to its ``prefill_ms``; a chunk-0 row must carry none. Integer arithmetic
  over the row -- a red here is a broken vehicle, not a measurement.
* COHERENCE (graded, three states): the chunked trajectory against the
  reference SET. GREEN when it lands in the set or diverges no earlier than
  the reference draws diverge among themselves (the measured band); RED when
  it leaves the trajectory before the instrument's own noise floor; VOID
  when the reference draws disagree from position 0 -- then this boot's
  instrument has no floor and the arm is not judged (never a silent pass:
  VOID is reported as VOID).
* PRICE (reported, never judged): ms/chunk per size, prefill tokens/s, delta
  against the reference floor. §13.10 duty 4 -- the solo floors move when
  the grain moves, so they are re-collected here rather than compared to
  older windows.

The gates are pure functions over result rows so the hermetic suite runs
them without a server; ``main()`` only moves rows.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

_HERE = os.path.dirname(os.path.abspath(__file__))
_DG = os.path.dirname(_HERE)
if _DG not in sys.path:
    sys.path.insert(0, _DG)


# ---------------------------------------------------------------------------
# pure gates
# ---------------------------------------------------------------------------


@dataclass
class Verdict:
    state: str  # "green" | "red" | "void"
    reason: str
    detail: Dict[str, Any]


def structure_gate(row: Dict[str, Any], n_tokens: int, chunk: int) -> Verdict:
    """The row's chunk bookkeeping as integer arithmetic.

    ``prefill_chunk_ms`` entries are rounded to 3 decimals and ``prefill_ms``
    to 2, so the sum check carries a rounding allowance of 0.5 ms plus a
    millipoint per chunk -- wide enough for rounding, far too narrow for a
    missing or double-counted chunk.
    """
    chunk_ms = row.get("prefill_chunk_ms")
    if chunk <= 0:
        if chunk_ms is None and row.get("prefill_chunks") is None:
            return Verdict("green", "single-forward row carries no chunk fields", {})
        return Verdict(
            "red",
            "chunk fields on a chunk-0 row: the single-forward path changed",
            {"prefill_chunks": row.get("prefill_chunks")},
        )
    expect = math.ceil(n_tokens / chunk)
    if chunk_ms is None:
        return Verdict(
            "red",
            "no prefill_chunk_ms on a chunked arm -- the override did not reach "
            "the lane (enqueue whitelist?) or the dispatch did not engage",
            {"expect_chunks": expect},
        )
    if row.get("prefill_chunks") != expect or len(chunk_ms) != expect:
        return Verdict(
            "red",
            "chunk count does not tile the prompt",
            {
                "expect_chunks": expect,
                "prefill_chunks": row.get("prefill_chunks"),
                "len_chunk_ms": len(chunk_ms),
            },
        )
    total = row.get("prefill_ms")
    allowance = 0.5 + 0.001 * expect
    if total is None or abs(sum(chunk_ms) - total) > allowance:
        return Verdict(
            "red",
            "chunk timings do not sum to prefill_ms",
            {"sum": round(sum(chunk_ms), 3), "prefill_ms": total},
        )
    return Verdict("green", "chunks tile and sum", {"chunks": expect})


def _divergence_index(a: List[int], b: List[int]) -> Optional[int]:
    """First position where two trajectories differ; None when equal
    (length difference counts as divergence at the shorter length)."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def coherence_gate(arm_ids: List[int], ref_sets: List[List[int]]) -> Verdict:
    """The graded, three-state reading of one chunked trajectory against the
    same-boot reference set (#328 band discipline, #404 set reading)."""
    if not ref_sets:
        return Verdict("void", "no reference draws", {})
    # The instrument's own floor: how early the reference draws diverge
    # among themselves. None = all draws byte-identical (exact instrument).
    ref_div: List[int] = []
    for i in range(len(ref_sets)):
        for j in range(i + 1, len(ref_sets)):
            d = _divergence_index(ref_sets[i], ref_sets[j])
            if d is not None:
                ref_div.append(d)
    floor = min(ref_div) if ref_div else None
    if floor == 0:
        return Verdict(
            "void",
            "reference draws disagree from position 0 -- no floor, no verdict",
            {"ref_divergences": ref_div},
        )
    arm_div = [_divergence_index(arm_ids, ref) for ref in ref_sets]
    if any(d is None for d in arm_div):
        return Verdict("green", "trajectory in the reference set", {"floor": floor})
    best = max(d for d in arm_div if d is not None)
    if floor is None:
        # Exact instrument, inexact arm: with a byte-identical reference set
        # any divergence is beyond the measured band -- red, with the index.
        return Verdict(
            "red",
            "diverges from a byte-identical reference set",
            {"divergence_index": best},
        )
    if best >= floor:
        return Verdict(
            "green",
            "diverges no earlier than the reference band",
            {"divergence_index": best, "floor": floor},
        )
    return Verdict(
        "red",
        "diverges before the instrument's own noise floor",
        {"divergence_index": best, "floor": floor},
    )


def price_table(
    rows: Dict[int, Dict[str, Any]], ref_ms: List[float], n_tokens: int
) -> List[Dict[str, Any]]:
    """ms/chunk and prefill rate per arm, floor delta against the same-boot
    reference mean. Reported, never judged (#284 rule 4: no share claims)."""
    ref_mean = sum(ref_ms) / len(ref_ms) if ref_ms else None
    out = []
    for chunk, row in sorted(rows.items()):
        ms = row.get("prefill_ms")
        entry = {
            "chunk": chunk,
            "prefill_ms": ms,
            "tokens_per_s": round(n_tokens / ms * 1000.0, 1) if ms else None,
            "ms_per_chunk_mean": (
                round(sum(row["prefill_chunk_ms"]) / len(row["prefill_chunk_ms"]), 3)
                if row.get("prefill_chunk_ms")
                else None
            ),
            "vs_ref_ms": (
                round(ms - ref_mean, 2) if ms is not None and ref_mean else None
            ),
        }
        out.append(entry)
    return out


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _build_prompt(tokenizer_path: str, base: str, n_tokens: int) -> List[int]:
    from lane_accept_probe import PROMPTS, tokenize

    text = " ".join(PROMPTS) + " "
    ids: List[int] = []
    while len(ids) < n_tokens:
        ids = tokenize(base, text, tokenizer_path)
        text += text
        if len(text) > 4_000_000:
            break
    if len(ids) < n_tokens:
        raise RuntimeError(f"could not synthesize {n_tokens} prompt tokens")
    return ids[:n_tokens]


def run(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="http://host:port of the boot")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--chunks", default="512,1024,2048")
    ap.add_argument("--prompt-tokens", type=int, default=1600)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--ref-draws", type=int, default=3)
    ap.add_argument("--spec", choices=["off", "on", "both"], default="both")
    ap.add_argument("--out", default="chunking_results.json")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="also exit non-zero on a red COHERENCE verdict. A red STRUCTURE "
        "verdict always exits non-zero -- that is a broken vehicle, while a "
        "coherence red is a finding (r404 rule: a divergent arm is not an "
        "abort, it is the measurement). VOID never fails the run; it fails "
        "the claim.",
    )
    args = ap.parse_args(argv)

    from lane_accept_probe import lane_run

    chunks = [int(c) for c in args.chunks.split(",") if int(c) > 0]
    if args.tokenizer:
        prompt = _build_prompt(args.tokenizer, args.base, args.prompt_tokens)
    else:
        # Smoke path: synthetic ids. The fake server judges shapes, not
        # content, and a real boot always passes --tokenizer.
        prompt = [(11 + i) % 32000 for i in range(args.prompt_tokens)]
    spec_arms = {"off": [False], "on": [True], "both": [False, True]}[args.spec]

    report: Dict[str, Any] = {
        "prompt_tokens": len(prompt),
        "chunks": chunks,
        "arms": {},
    }
    failed_structure = False
    failed_coherence = False
    for spec in spec_arms:
        tag = "spec" if spec else "nospec"
        refs: List[Dict[str, Any]] = []
        for _ in range(args.ref_draws):
            rows = lane_run(
                args.base,
                {
                    "input_ids": prompt,
                    "max_new_tokens": args.max_new,
                    "spec": spec,
                    "prefill_chunk": 0,
                },
            )
            refs.append(rows[-1])
        ref_ids = [r.get("output_ids") or [] for r in refs]
        ref_ms = [r["prefill_ms"] for r in refs if r.get("prefill_ms")]
        arm_rows: Dict[int, Dict[str, Any]] = {}
        verdicts: Dict[str, Any] = {}
        for ref_index, ref_row in enumerate(refs):
            sv = structure_gate(ref_row, len(prompt), 0)
            verdicts[f"ref{ref_index}/structure"] = asdict(sv)
            failed_structure |= sv.state == "red"
        for chunk in chunks:
            rows = lane_run(
                args.base,
                {
                    "input_ids": prompt,
                    "max_new_tokens": args.max_new,
                    "spec": spec,
                    "prefill_chunk": chunk,
                },
            )
            row = rows[-1]
            arm_rows[chunk] = row
            sv = structure_gate(row, len(prompt), chunk)
            cv = coherence_gate(row.get("output_ids") or [], ref_ids)
            verdicts[f"chunk{chunk}/structure"] = asdict(sv)
            verdicts[f"chunk{chunk}/coherence"] = asdict(cv)
            failed_structure |= sv.state == "red"
            failed_coherence |= cv.state == "red"
        report["arms"][tag] = {
            "ref_prefill_ms": ref_ms,
            "ref_output_ids": ref_ids,
            "rows": {str(k): v for k, v in arm_rows.items()},
            "verdicts": verdicts,
            "price": price_table(arm_rows, ref_ms, len(prompt)),
        }

    with open(args.out, "w") as f:
        json.dump(report, f, indent=1)
    for tag, arm in report["arms"].items():
        print(f"== {tag}")
        for name, verdict in arm["verdicts"].items():
            print(f"  {name:24s} {verdict['state'].upper():5s} {verdict['reason']}")
        for entry in arm["price"]:
            print(
                f"  chunk {entry['chunk']:>5d}: {entry['prefill_ms']} ms "
                f"({entry['tokens_per_s']} tok/s, "
                f"mean {entry['ms_per_chunk_mean']} ms/chunk, "
                f"{entry['vs_ref_ms']} ms vs ref)"
            )
    print(f"report: {args.out}")
    if failed_structure:
        print("STRUCTURE: broken vehicle, at least one red row", file=sys.stderr)
        return 1
    if args.strict and failed_coherence:
        print("STRICT: at least one red coherence verdict", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(run())
