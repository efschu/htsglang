#!/usr/bin/env python3
"""Task #340 posten 2: one arm's greedy trajectory, with its own noise floor.

The instrument is the one the #336 ARM B reference used, so the numbers are
comparable across commits and arms: the prompts and the request payload come
from ``lane_accept_probe`` (PROMPTS / serving_run), the sampling is greedy
(temperature 0, ignore_eos), and every prompt is run TWICE. The second run is
not a repeat for confidence -- it is the A-vs-A floor. A prompt whose own two
runs disagree carries no verdict in either direction, and is reported VOID
rather than as a deviation.

Two bounds keep this usable inside a card window:

  --req-timeout-s   caps a single HTTP call. ``lane_accept_probe._post``
                    defaults to 600 s, which on six calls is an hour of
                    hanging; it is rebound here rather than reimplemented so
                    the request payload stays defined in exactly one place.
  --deadline-s      caps the whole probe. Prompts not reached by then are
                    recorded as SKIPPED, which is a different fact from a red
                    floor and is kept distinct in the JSON.

The two bounds are composed rather than merely both applied: a single call is
given whichever of the two is SMALLER at the moment it starts, so a hung server
cannot push the probe past its deadline by up to two request timeouts. The
whole run therefore ends within a few seconds of --deadline-s no matter what
the server does, and always writes its JSON.

Exit code 0 when at least one prompt has a green floor, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

MIN_REQ_TIMEOUT_S = 5.0


def _load_probe(module_dir: str, budget):
    """Import the shared probe and bound every request it makes.

    ``budget()`` returns the seconds this call may take: the smaller of the
    per-request cap and what is left of the whole-probe deadline.
    """
    sys.path.insert(0, module_dir)
    import lane_accept_probe as lap  # noqa: E402

    original_post = lap._post

    def _bounded_post(base, path, payload, timeout=None):
        return original_post(base, path, payload, timeout=budget())

    lap._post = _bounded_post
    return lap


def _ids(res: Optional[Dict[str, Any]]) -> Optional[List[int]]:
    if not res:
        return None
    out = res.get("output_ids") or (res.get("meta_info") or {}).get("output_ids")
    return list(out) if out else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="", help="free-text launch summary")
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--prompts", default="alphabet,squares,code")
    ap.add_argument(
        "--module-dir",
        default="/spinning/wt-340/scripts/dual_group",
        help="directory holding lane_accept_probe.py",
    )
    ap.add_argument("--req-timeout-s", type=float, default=60.0)
    ap.add_argument("--deadline-s", type=float, default=120.0)
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    stop_at = time.time() + args.deadline_s

    def budget() -> float:
        return max(MIN_REQ_TIMEOUT_S, min(args.req_timeout_s, stop_at - time.time()))

    lap = _load_probe(args.module_dir, budget)

    report: Dict[str, Any] = {
        "label": args.label,
        "config": args.config,
        "port": args.port,
        "tokens": args.tokens,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompts": [],
    }

    for name in [p for p in args.prompts.split(",") if p]:
        row: Dict[str, Any] = {"prompt": name}
        if time.time() >= stop_at:
            row.update(status="SKIPPED", floor_identical=False, ids=None)
            report["prompts"].append(row)
            print(f"  {name:10s} SKIPPED (probe deadline)", flush=True)
            continue
        t0 = time.time()
        try:
            ids = lap.tokenize(base, lap.PROMPTS[name], args.tokenizer)
            row["n_prompt_ids"] = len(ids)
            a = _ids(lap.serving_run(base, ids, args.tokens))
            b = _ids(lap.serving_run(base, ids, args.tokens))
            row["ids"] = a
            row["ids_b"] = b
            row["floor_identical"] = a is not None and a == b
            row["status"] = "OK" if row["floor_identical"] else "FLOOR_RED"
        except Exception as exc:  # card-window robustness: never abort the arm
            row["error"] = repr(exc)
            row["ids"] = None
            row["floor_identical"] = False
            row["status"] = "ERROR"
        row["seconds"] = round(time.time() - t0, 2)
        report["prompts"].append(row)
        print(
            f"  {name:10s} {row['status']:9s} floor={row['floor_identical']} "
            f"head={(row.get('ids') or [])[:4]} ({row['seconds']}s)",
            flush=True,
        )

    green = [r["prompt"] for r in report["prompts"] if r.get("floor_identical")]
    report["green_prompts"] = green
    report["verdict"] = (
        f"USABLE on {green}" if green else "VOID: no green floor on any prompt"
    )
    print(f"{args.label}: {report['verdict']}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
