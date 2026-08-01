#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#328: the semantic chain-quality gate — graded, against a measured band.

A reusable verdict for "did turning the lane chain on make the OUTPUT worse",
built from the two rules this fork paid for:

* #360 / #365 — TEXT IDENTITY IS NOT AN INSTRUMENT. Greedy speculation only
  reproduces the greedy trajectory if the verify forward is bit-identical to
  the decode forward, and it is not (2-row verify vs 1-row decode reassociates
  the reduction). A gate that fails on a flip at a near-tie position measures
  the margin at that position, not the lane. So the gate scores CONTENT,
  graded, via ``r12/graded.py``.
* #274 — THE BAND IS MEASURED, NEVER PRE-REGISTERED. The obvious constant for
  "how much may a score move" would have decided the answer by itself: with
  Q3_K weights and an fp8 KV cache the numeric perturbation is nothing like
  the 1e-3 a plain fp32 reassociation gives, and a 1e-3 threshold would have
  reported world B where the measurement says world A. The band therefore
  comes from the SAME BOOT, as the arms' own A-vs-A repeat spread.

THE RULE

    band   = max(|ref_a - ref_b|, |cand_a - cand_b|)      # A-vs-A, same boot
    margin = |cand_a - ref_a|                             # the cross-arm delta
    GREEN  <=> margin <= band

The band is a max over BOTH arms' self-noise, not the reference arm's alone:
a candidate arm that is itself unstable has a noise floor of its own, and
judging its cross-arm delta against a quieter arm's floor would report its own
jitter as a regression.

WHAT THE GATE REFUSES TO DO

It never substitutes a constant when the band cannot be measured. A missing
repeat, a prompt with no scorer, or a score the grader could not produce comes
back VOID with the reason attached. A VOID is not a pass and not a failure --
it says the instrument was not present, which is the one thing a threshold
would have hidden.

It also does not average prompts into a single number. Each prompt carries its
own band (its own noise), so each is judged on its own and the run verdict is
the worst of them; a prompt that scores full marks cannot mask one that
collapsed.

Machine-readable by construction: :func:`judge_run` returns a dict, and
``--json`` prints it. The CLI consumes the report shape
``r12/stock_spec_control.py`` already writes (``prompts[name].run_a/run_b``),
so nothing new has to be plumbed to use it.

Hermetic: no server, no card, no torch.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "r12"))

from graded import score as graded_score  # noqa: E402

#: Verdict vocabulary. Deliberately three-valued: a gate that can only say
#: pass/fail has to call a missing instrument one of them.
GREEN = "GREEN"
RED = "RED"
VOID = "VOID"


def band_of(ref_a: float, ref_b: float, cand_a: float, cand_b: float) -> float:
    """The empirical A-vs-A band: ``max(|ref_a-ref_b|, |cand_a-cand_b|)``.

    Both arms contribute their own repeat spread and the wider one wins --
    see the module docstring for why the reference arm's floor alone is the
    wrong denominator for an unstable candidate.
    """
    return max(abs(float(ref_a) - float(ref_b)), abs(float(cand_a) - float(cand_b)))


def judge_scores(
    name: str,
    ref_a: Optional[float],
    ref_b: Optional[float],
    cand_a: Optional[float],
    cand_b: Optional[float],
    max_score: Optional[float] = None,
) -> Dict[str, Any]:
    """One prompt's verdict from four graded scores.

    ``None`` anywhere, or a grader sentinel (-1 = "no scorer for this
    prompt"), is VOID with the reason named -- never silently treated as 0,
    which would read as "the model got nothing right".
    """
    values = {"ref_a": ref_a, "ref_b": ref_b, "cand_a": cand_a, "cand_b": cand_b}
    missing = [k for k, v in values.items() if v is None]
    if missing:
        return {
            "prompt": name,
            "verdict": VOID,
            "reason": (
                f"missing score(s) {', '.join(sorted(missing))}: the band needs "
                f"both arms' A-vs-A repeat, and no constant stands in for it"
            ),
            **values,
        }
    unscored = [k for k, v in values.items() if float(v) < 0]
    if unscored:
        return {
            "prompt": name,
            "verdict": VOID,
            "reason": (
                f"grader returned no score for {', '.join(sorted(unscored))} "
                f"(prompt {name!r} has no scorer); an ungraded prompt is not "
                f"evidence either way"
            ),
            **values,
        }

    band = band_of(ref_a, ref_b, cand_a, cand_b)
    margin = abs(float(cand_a) - float(ref_a))
    inside = margin <= band
    entry: Dict[str, Any] = {
        "prompt": name,
        "verdict": GREEN if inside else RED,
        "band": band,
        "margin": margin,
        "ref_a": float(ref_a),
        "ref_b": float(ref_b),
        "cand_a": float(cand_a),
        "cand_b": float(cand_b),
        "ref_repeat_delta": abs(float(ref_a) - float(ref_b)),
        "cand_repeat_delta": abs(float(cand_a) - float(cand_b)),
        "direction": (
            "better"
            if float(cand_a) > float(ref_a)
            else ("worse" if float(cand_a) < float(ref_a) else "equal")
        ),
    }
    if max_score is not None:
        entry["max_score"] = float(max_score)
    if inside and band == 0.0 and margin == 0.0:
        # Both arms held perfectly still AND agree. Worth naming: it is the
        # strongest possible pass, and it is also the case a reader would
        # otherwise mistake for "the band swallowed the delta".
        entry["note"] = "both arms held still and agree; band and margin are 0"
    elif not inside:
        entry["reason"] = (
            f"cross-arm delta {margin:g} exceeds the same-boot A-vs-A band "
            f"{band:g} (ref repeat {entry['ref_repeat_delta']:g}, candidate "
            f"repeat {entry['cand_repeat_delta']:g})"
        )
    return entry


def _score_of(run: Optional[Dict], prompt: str) -> Optional[float]:
    """Graded score of one run's text, or ``None`` when there is no text."""
    if not run:
        return None
    text = run.get("text")
    if text is None:
        return None
    return float(graded_score(prompt, text)["score"])


def judge_prompt_reports(
    name: str, ref_entry: Dict, cand_entry: Dict
) -> Dict[str, Any]:
    """One prompt's verdict from the two arms' ``stock_spec_control`` entries.

    Honours the harness's own ``void`` marker: an arm that did not hold still
    was already disqualified as a control by the producer, and re-judging it
    here would overrule a decision made closer to the measurement.
    """
    for label, entry in (("reference", ref_entry), ("candidate", cand_entry)):
        if entry.get("void"):
            return {
                "prompt": name,
                "verdict": VOID,
                "reason": f"{label} arm marked void by the harness: {entry['void']}",
            }
    got = graded_score(name, "")
    return judge_scores(
        name,
        _score_of(ref_entry.get("run_a"), name),
        _score_of(ref_entry.get("run_b"), name),
        _score_of(cand_entry.get("run_a"), name),
        _score_of(cand_entry.get("run_b"), name),
        max_score=(got["max_score"] if got["max_score"] >= 0 else None),
    )


def judge_run(reference: Dict, candidate: Dict) -> Dict[str, Any]:
    """The whole gate: every prompt judged, and the run verdict.

    The run verdict is the WORST prompt verdict (RED beats VOID beats GREEN):
    a gate that averaged would let a full-marks prompt pay for a collapsed
    one. VOID outranks GREEN because a run with an unmeasurable prompt has
    not been fully judged, and saying GREEN would overstate what was checked.
    """
    ref_prompts = reference.get("prompts") or {}
    cand_prompts = candidate.get("prompts") or {}
    names = sorted(set(ref_prompts) | set(cand_prompts))
    entries: List[Dict[str, Any]] = []
    for name in names:
        if name not in ref_prompts or name not in cand_prompts:
            entries.append(
                {
                    "prompt": name,
                    "verdict": VOID,
                    "reason": (
                        f"prompt {name!r} is present in only one arm; there is "
                        f"nothing to compare"
                    ),
                }
            )
            continue
        entries.append(
            judge_prompt_reports(name, ref_prompts[name], cand_prompts[name])
        )

    verdicts = [e["verdict"] for e in entries]
    if not entries:
        run_verdict = VOID
    elif RED in verdicts:
        run_verdict = RED
    elif VOID in verdicts:
        run_verdict = VOID
    else:
        run_verdict = GREEN
    return {
        "gate": "chain_quality",
        "method": (
            "graded content scores vs the same-boot A-vs-A band "
            "max(|ref_a-ref_b|, |cand_a-cand_b|); never text identity, never a "
            "pre-registered constant"
        ),
        "verdict": run_verdict,
        "n_prompts": len(entries),
        "n_green": verdicts.count(GREEN),
        "n_red": verdicts.count(RED),
        "n_void": verdicts.count(VOID),
        "reference_arm": reference.get("arm"),
        "candidate_arm": candidate.get("arm"),
        "prompts": entries,
    }


def format_report(result: Dict[str, Any]) -> str:
    """Human-readable rendering; the JSON stays the machine surface."""
    lines = [
        f"chain-quality gate: {result['verdict']} "
        f"({result['n_green']} green, {result['n_red']} red, "
        f"{result['n_void']} void of {result['n_prompts']})",
        f"  method: {result['method']}",
    ]
    for e in result["prompts"]:
        if e["verdict"] == VOID:
            lines.append(f"  {e['prompt']:9s} VOID  -- {e.get('reason', '')}")
            continue
        lines.append(
            f"  {e['prompt']:9s} {e['verdict']:5s} "
            f"score {e['cand_a']:g} vs ref {e['ref_a']:g} "
            f"(margin {e['margin']:g}, band {e['band']:g}, {e['direction']})"
        )
        if e.get("reason"):
            lines.append(f"            {e['reason']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", required=True, help="reference arm report JSON")
    ap.add_argument("--candidate", required=True, help="candidate arm report JSON")
    ap.add_argument("--json", action="store_true", help="print the machine surface")
    ap.add_argument("--out", help="write the JSON verdict here")
    args = ap.parse_args(argv)

    with open(args.reference) as f:
        reference = json.load(f)
    with open(args.candidate) as f:
        candidate = json.load(f)

    result = judge_run(reference, candidate)
    print(json.dumps(result, indent=2) if args.json else format_report(result))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    # GREEN 0, RED 1, VOID 2 -- a void must not read as a pass in CI.
    return {GREEN: 0, RED: 1, VOID: 2}[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
