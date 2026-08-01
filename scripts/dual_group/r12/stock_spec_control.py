#!/usr/bin/env python3
"""#274 / #284: the control arm the divergence verdict was missing.

THE QUESTION. The lane's speculative chain leaves the lane's own greedy
trajectory (alphabet at index 7, squares at index 18, #284 b4a7def95c) with
every A-vs-A floor green on both sides. Two worlds fit that observation:

(A) the 2-row verify forward is not bit-identical to the 1-row decode
    forward, so a position whose top-2 logit margin is under that numeric
    difference flips. Inherent, quality-neutral, and NOT the lane's -- it
    would happen to any speculative decoder on this model.
(B) the lane's verify path computes something different -- wrong logits,
    wrong KV row, a head shift of the r3 kind. A defect.

Both worlds predict "spec leaves no-spec". Only (B) predicts that the LANE is
required to produce it. So the discriminator is a control arm with NO LANE AT
ALL: plain NEXTN speculation against plain greedy decoding, same model, same
partition, same prompts. If stock speculation leaves the stock greedy
trajectory the same way, the lane cannot be the carrier of a phenomenon that
occurs without it.

This script is one ARM of that control. It runs entirely over ``/generate``
and never touches the lane, so it works on a boot that has no lane compiled
into it at all. The boot decides which arm it is:

    --speculative-algorithm NEXTN ...   -> the `spec` arm
    (no speculative flags)              -> the `nospec` arm

WHAT IT RECORDS, and why each field is needed to tell A from B:

* ``output_ids`` twice per prompt. The A-vs-A floor. An arm whose own two
  runs disagree carries no verdict -- the #284 lesson, applied to the control
  as well, because a control with a moving denominator controls nothing.
* the top-2 logprobs at every emitted position. This is the instrument that
  separates the two worlds at the divergence index itself: world (A) requires
  the flipped position to be a NEAR TIE, because float non-associativity moves
  a logit by ~1e-3 and cannot flip a decision made by a margin of 1. A flip at
  a comfortable margin is world (B) and nothing else.

Usage (inside a boot recipe that owns the card):

    python stock_spec_control.py --port 30082 --arm spec \
        --tokenizer <dir> --out /tmp/r12/spec.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lane_accept_probe import PROMPTS, _post, tokenize  # noqa: E402


def serving_run_logprobs(
    base: str, input_ids: List[int], tokens: int
) -> Dict[str, Any]:
    """One greedy continuation with the top-2 logprobs at every position.

    ``ignore_eos`` so the length is fixed by the caller and not by the
    content: two arms that stop at different places are not comparable, and
    the gate's length-end rule exists precisely because a speculative block
    can overrun. ``temperature 0`` because the whole question is about the
    greedy argmax.
    """
    return _post(
        base,
        "/generate",
        {
            "input_ids": input_ids,
            "sampling_params": {
                "max_new_tokens": tokens,
                "temperature": 0,
                "ignore_eos": True,
            },
            "return_logprob": True,
            "top_logprobs_num": 2,
        },
    )


def _margins(meta: Dict[str, Any]) -> List[Optional[float]]:
    """Per emitted position: logprob(top1) - logprob(top2), or None.

    ``output_top_logprobs`` is a list (one entry per position) of lists of
    ``(logprob, token_id, token_text)``, already sorted best first. A position
    that carries fewer than two candidates yields None rather than a fake
    number -- an absent margin must not read as a huge one.
    """
    top = meta.get("output_top_logprobs")
    if not top:
        return []
    out: List[Optional[float]] = []
    for cand in top:
        if not cand or len(cand) < 2:
            out.append(None)
            continue
        try:
            out.append(round(float(cand[0][0]) - float(cand[1][0]), 6))
        except (TypeError, ValueError, IndexError):
            out.append(None)
    return out


def _ids(meta: Dict[str, Any]) -> List[int]:
    """The emitted token ids, from the logprob channel.

    Read off ``output_token_logprobs`` rather than re-tokenizing the text:
    a trajectory comparison must run on the ids the sampler actually chose,
    and detokenize/retokenize is not a round trip for every token.
    """
    rows = meta.get("output_token_logprobs") or []
    out: List[int] = []
    for r in rows:
        try:
            out.append(int(r[1]))
        except (TypeError, ValueError, IndexError):
            pass
    return out


def run_arm(
    base: str,
    tokenizer: str,
    prompt_names: List[str],
    tokens: int,
    arm: str,
    deadline: float,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"arm": arm, "prompts": {}}
    for name in prompt_names:
        if time.time() > deadline:
            out["truncated_at"] = name
            break
        ids = tokenize(base, PROMPTS[name], tokenizer)

        runs = []
        for _ in range(2):
            res = serving_run_logprobs(base, ids, tokens)
            meta = res.get("meta_info") or {}
            runs.append(
                {
                    "output_ids": _ids(meta),
                    "margins": _margins(meta),
                    "text": res.get("text"),
                    "completion_tokens": meta.get("completion_tokens"),
                    "spec_accept_length": meta.get("spec_accept_length"),
                    "spec_verify_ct": meta.get("spec_verify_ct"),
                }
            )

        floor_ok = runs[0]["output_ids"] == runs[1]["output_ids"]
        entry = {
            "prompt_tokens": len(ids),
            "floor_byte_identical": floor_ok,
            "n_out": len(runs[0]["output_ids"]),
            "run_a": runs[0],
            "run_b": runs[1],
        }
        if not floor_ok:
            entry["void"] = (
                f"{arm} a-vs-a floor not byte-identical: this arm did not hold "
                "still, so it cannot serve as a control"
            )
        out["prompts"][name] = entry
        m = [x for x in runs[0]["margins"] if x is not None]
        print(
            f"  {arm:7s} {name:9s} floor={'ok' if floor_ok else 'VOID'} "
            f"n_out={len(runs[0]['output_ids'])} "
            f"accept={runs[0]['spec_accept_length']} "
            f"margin_min={min(m) if m else None} "
            f"margin_median={sorted(m)[len(m) // 2] if m else None}",
            flush=True,
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--arm", required=True, choices=["spec", "nospec"])
    ap.add_argument("--prompts", default="alphabet,squares")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--deadline-s", type=float, default=600.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    base = f"http://127.0.0.1:{a.port}"
    report = run_arm(
        base,
        a.tokenizer,
        [p for p in a.prompts.split(",") if p],
        a.tokens,
        a.arm,
        time.time() + a.deadline_s,
    )
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"written: {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
