"""#274 families slice 2: the per-arm coherence gate, one arm per boot.

WHAT IT DECIDES

For a model family that has never carried a lane, the question is not "how
fast" -- it is "does the lane compute the same model". The gate answers that
with the reproducibility standard round 8/#284 arrived at, applied to a
NON-speculative arm:

1. FLOOR, serving side. The same greedy request twice. If the serving group
   does not reproduce itself on a prompt, that prompt carries no verdict in
   either direction and is marked void. (GDN prefill is known to be
   non-reproducible past ~109 tokens, so the gate stays short on purpose.)
2. FLOOR, lane side. The same lane job twice, same rule.
3. COHERENCE. The lane's trajectory against the serving group's, on the
   prompts whose two floors are green.

Only a content divergence on a green-floored prompt fails the gate. A void
prompt is reported as void, never as a pass and never as a failure -- the
instrument saying it cannot see, which is a third outcome.

The trajectories are compared as TOKEN IDS, not as text: a detokenizer can
map two different id sequences onto the same string.

Usage (from inside the boot recipe, which owns the card):

    python family_gate.py --port 30090 --tokenizer <dir> --out report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lane_accept_probe import (  # noqa: E402
    PROMPTS,
    _get,
    lane_run,
    serving_run,
    tokenize,
)


def _lane_ids(base: str, ids: List[int], tokens: int) -> Optional[List[int]]:
    rows = lane_run(
        base,
        {
            "lane_id": 0,
            "input_ids": ids,
            "max_new_tokens": tokens,
            "spec": False,
        },
    )
    if not rows:
        return None
    row = rows[-1] if isinstance(rows, list) else rows
    return list(row.get("output_ids") or [])


def _serving_ids(base: str, ids: List[int], tokens: int) -> Optional[List[int]]:
    res = serving_run(base, ids, tokens)
    if not res:
        return None
    # /generate carries the ids at the TOP level of the response; meta_info is
    # the fallback for older shapes. Reading only meta_info returns None on
    # every request, which the floor then reports as "not reproducible" --
    # an instrument defect that looks exactly like a finding.
    out = res.get("output_ids") or (res.get("meta_info") or {}).get("output_ids")
    return list(out) if out else None


def gate_one(base: str, tokenizer: str, name: str, tokens: int) -> Dict[str, Any]:
    ids = tokenize(base, PROMPTS[name], tokenizer)
    row: Dict[str, Any] = {"prompt": name, "n_prompt_ids": len(ids)}

    serving_a = _serving_ids(base, ids, tokens)
    serving_b = _serving_ids(base, ids, tokens)
    row["serving_floor_identical"] = serving_a is not None and serving_a == serving_b
    lane_a = _lane_ids(base, ids, tokens)
    lane_b = _lane_ids(base, ids, tokens)
    row["lane_floor_identical"] = lane_a is not None and lane_a == lane_b

    if not (row["serving_floor_identical"] and row["lane_floor_identical"]):
        row["verdict"] = "void"
        row["why"] = "a floor is not reproducible; the comparison has no basis"
        return row

    row["n_out_serving"] = len(serving_a)
    row["n_out_lane"] = len(lane_a)
    if lane_a == serving_a:
        row["verdict"] = "byte_identical"
        return row
    first = next((i for i, (x, y) in enumerate(zip(lane_a, serving_a)) if x != y), None)
    if first is None:
        row["verdict"] = "prefix_identical_different_length"
    else:
        row["verdict"] = "content_divergence"
        row["first_divergent_index"] = first
        row["lane_tail"] = lane_a[max(0, first - 2) : first + 3]
        row["serving_tail"] = serving_a[max(0, first - 2) : first + 3]
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--prompts", default="alphabet,squares,code")
    ap.add_argument("--deadline-s", type=float, default=600.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    deadline = time.time() + args.deadline_s
    report: Dict[str, Any] = {"prompts": [], "verdict": None}
    for name in [p for p in args.prompts.split(",") if p]:
        if time.time() > deadline:
            report["truncated_at"] = name
            break
        try:
            row = gate_one(base, args.tokenizer, name, args.tokens)
        except Exception as exc:  # pragma: no cover - card-window robustness
            row = {"prompt": name, "verdict": "error", "why": repr(exc)}
        report["prompts"].append(row)
        print(f"  {name:10s} {row['verdict']}", flush=True)

    verdicts = [r["verdict"] for r in report["prompts"]]
    judged = [v for v in verdicts if v not in ("void", "error")]
    if any(v == "content_divergence" for v in verdicts):
        report["verdict"] = "RED: lane diverges from the serving group"
    elif not judged:
        report["verdict"] = "VOID: no prompt had two green floors"
    elif all(v == "byte_identical" for v in judged):
        report["verdict"] = f"GREEN: byte-identical on {len(judged)} judged prompt(s)"
    else:
        report["verdict"] = f"GREEN(prefix): {verdicts}"

    try:
        info = _get(base, "/get_server_info") or {}
        report["lane_stats"] = [
            lane
            for st in (info.get("internal_states") or [])
            for lane in (st.get("dual_group_lanes") or [])
        ]
    except Exception:
        report["lane_stats"] = None

    print(report["verdict"], flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0 if str(report["verdict"]).startswith("GREEN") else 1


if __name__ == "__main__":
    raise SystemExit(main())
