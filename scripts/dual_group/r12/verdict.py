#!/usr/bin/env python3
"""#274 / #284: read the control arms and name the world, A or B.

Input: the two JSON reports ``stock_spec_control.py`` wrote, one per boot
(``--nospec`` and ``--spec``). Output: for every prompt, the classification
of the stock speculative trajectory against the stock greedy one, and -- at
the first divergent index -- the top-2 logprob margin the greedy arm measured
there, together with where that margin sits in the distribution of all
margins of the same run.

THE DECISION RULE, written down before the numbers were in:

* Stock speculation does NOT leave the stock greedy trajectory
  -> the divergence needs the lane -> WORLD B, the lane is the carrier.
* Stock speculation DOES leave it, and the flipped position is a near tie
  (its margin is at or near the smallest of the run)
  -> the mechanism is batch-shape numerics and it exists without any lane
  -> WORLD A, and the lane-spec gate is measuring the mechanism, not the lane.
* Stock leaves it at a COMFORTABLE margin -> neither: the control itself is
  broken, and nothing may be concluded about the lane from it.

The margin is the load-bearing quantity, and the band it is compared against
is MEASURED here rather than assumed. Two arms that agree on a token still
report slightly different logprobs for it, and that difference IS the numeric
perturbation between the 2-row verify path and the 1-row decode path on this
vehicle -- with Q3_K weights and an fp8 KV cache it is nothing like the 1e-3
a plain fp32 reassociation would give, so guessing it would have decided the
answer by the guess. The perturbation band is therefore read off the
agreeing positions of the two arms, and a flip counts as explained when the
margin at the flipped position sits inside it.

``NEAR_TIE_ABS`` stays as a coarse, pre-registered secondary check so a
degenerate band (one position, or a band that swallows everything) cannot on
its own turn a real defect into a near tie.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graded import score as graded_score  # noqa: E402

NEAR_TIE_ABS = 0.05
"""Margin (in nats) at or below which a position counts as a near tie.

Not tuned to the answer: it is two orders above the ~1e-3 a reassociated
reduction moves a logit, so it is generous to world B -- a flip at a margin
above this is NOT explained by batch-shape numerics and would be reported as
such. The report prints the raw margin next to it either way.
"""


def _rank_of(value: float, pool: List[float]) -> Dict[str, Any]:
    """Where `value` sits among `pool`: rank, and the pool's shape."""
    ordered = sorted(pool)
    rank = sum(1 for x in ordered if x < value)
    return {
        "rank_from_smallest": rank,
        "n_positions": len(ordered),
        "pool_min": round(ordered[0], 6) if ordered else None,
        "pool_median": round(statistics.median(ordered), 6) if ordered else None,
        "pool_max": round(ordered[-1], 6) if ordered else None,
    }


def perturbation_band(nospec: Dict, spec: Dict) -> Dict[str, Any]:
    """How much the two kernel paths move a top-2 GAP, measured on this pair.

    At a position where both arms committed the same token, each arm still
    reports its own top-2 margin. The absolute difference between those two
    margins is what the batch-shape change did to the very quantity a flip
    depends on -- not a logit in isolation, but the GAP that decides the
    argmax. It is the directly relevant band, and it is measured rather than
    assumed: with Q3_K weights and an fp8 KV cache the perturbation is
    nothing like the 1e-3 a plain fp32 reassociation would give, so a
    constant chosen in advance would have decided the answer by itself.

    Positions at or after the first divergence are excluded: past that point
    the arms condition on different prefixes and the difference is no longer
    only numeric.
    """
    ref_ids = nospec["run_a"]["output_ids"]
    got_ids = spec["run_a"]["output_ids"]
    ref_m = nospec["run_a"].get("margins") or []
    got_m = spec["run_a"].get("margins") or []
    deltas: List[float] = []
    for i, (a, b) in enumerate(zip(ref_ids, got_ids)):
        if a != b:
            break
        if i < len(ref_m) and i < len(got_m):
            if ref_m[i] is None or got_m[i] is None:
                continue
            deltas.append(abs(float(ref_m[i]) - float(got_m[i])))
    if not deltas:
        return {"n": 0, "max": None, "median": None, "p90": None}
    ordered = sorted(deltas)
    return {
        "n": len(ordered),
        "max": round(ordered[-1], 6),
        "median": round(statistics.median(ordered), 6),
        "p90": round(ordered[int(0.9 * (len(ordered) - 1))], 6),
    }


def judge_prompt(name: str, nospec: Dict, spec: Dict) -> Dict[str, Any]:
    if nospec.get("void") or spec.get("void"):
        return {
            "prompt": name,
            "verdict": "void",
            "why": nospec.get("void") or spec.get("void"),
        }

    ref = nospec["run_a"]["output_ids"]
    got = spec["run_a"]["output_ids"]
    cmp_ = compare(ref, got)
    entry: Dict[str, Any] = {
        "prompt": name,
        "classification": cmp_["classification"],
        "first_divergent_index": cmp_["first_divergent_index"],
        "n_out_nospec": len(ref),
        "n_out_spec": len(got),
        "spec_accept_length": spec["run_a"].get("spec_accept_length"),
    }

    # The graded score sits next to the classification on purpose. A flip at a
    # near tie that leaves the answer just as correct is the signature of
    # world A; the identity check cannot see the difference between that and
    # a trajectory that fell apart, and the score can.
    s_ref = graded_score(name, nospec["run_a"].get("text") or "")
    s_got = graded_score(name, spec["run_a"].get("text") or "")
    entry["graded"] = {
        "nospec": s_ref,
        "spec": s_got,
        "delta": s_got["score"] - s_ref["score"],
    }

    idx = cmp_["first_divergent_index"]
    margins: List[Optional[float]] = nospec["run_a"].get("margins") or []
    pool = [m for m in margins if m is not None]
    if idx is not None and idx < len(margins) and margins[idx] is not None:
        m = margins[idx]
        entry["margin_at_divergence"] = m
        entry["margin_context"] = _rank_of(m, pool)
        band = perturbation_band(nospec, spec)
        entry["perturbation_band"] = band
        entry["inside_band"] = (
            band["max"] is not None and band["n"] >= 3 and m <= band["max"]
        )
        entry["near_tie"] = m <= NEAR_TIE_ABS
        entry["explained_by_numerics"] = bool(
            entry["inside_band"] or entry["near_tie"]
        )
        entry["ref_top2"] = {
            "nospec_token": ref[idx] if idx < len(ref) else None,
            "spec_token": got[idx] if idx < len(got) else None,
        }
    elif idx is not None:
        entry["margin_at_divergence"] = None
        entry["near_tie"] = None
        entry["why"] = "no top-2 logprobs recorded at the divergent index"
    return entry


def compare(ref: List[int], got: List[int]) -> Dict[str, Any]:
    """The r8 gate's classifier, inlined so this script has no server import.

    Kept literally equivalent to
    ``scripts/dual_group/r8/lane_spec_window.compare_trajectories`` -- the
    control must be judged by the same instrument as the thing it controls,
    or the comparison is between two rules and not between two arms.
    """
    first = None
    for i, (x, y) in enumerate(zip(got, ref)):
        if x != y:
            first = i
            break
    if first is not None:
        cls = "content_divergence"
    elif len(got) == len(ref):
        cls = "identical"
    else:
        cls = "length_end_only"
    return {"classification": cls, "first_divergent_index": first}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nospec", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()

    with open(a.nospec) as f:
        nospec = json.load(f)
    with open(a.spec) as f:
        spec = json.load(f)

    names = [n for n in nospec["prompts"] if n in spec["prompts"]]
    prompts = [judge_prompt(n, nospec["prompts"][n], spec["prompts"][n]) for n in names]

    judged = [p for p in prompts if p.get("verdict") != "void"]
    diverging = [p for p in judged if p["classification"] == "content_divergence"]
    explained = [p for p in diverging if p.get("explained_by_numerics") is True]

    if not judged:
        world = "undecided: no prompt carried a verdict (floors void)"
    elif not diverging:
        world = (
            "B-leaning: stock speculation did NOT leave the stock greedy "
            "trajectory on any judged prompt, so the lane is required to "
            "produce the divergence"
        )
    elif len(explained) == len(diverging):
        world = (
            "A: stock speculation leaves the stock greedy trajectory WITHOUT "
            "any lane, and every flip sits inside the measured batch-shape "
            "perturbation band -- the mechanism is numerics, not the lane"
        )
    else:
        world = (
            "control broken or world B: stock diverges at a margin OUTSIDE "
            "the measured perturbation band, which batch-shape "
            "reassociation on this vehicle does not reach"
        )

    report = {
        "near_tie_threshold_nats": NEAR_TIE_ABS,
        "prompts": prompts,
        "judged_prompts": len(judged),
        "diverging_prompts": len(diverging),
        "explained_by_numerics": len(explained),
        "world": world,
    }
    print(json.dumps(report, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(report, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
