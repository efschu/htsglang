#!/usr/bin/env python3
"""#700: ReplaySSM byte-identity probe. WINDOW-GATED.

Measures whether ``--enable-linear-replayssm`` changes output at all. The
decision rules live in ``sglang.srt.planner.replayssm_identity`` and are unit
tested without a GPU; this script only collects the samples and hands them over,
so the verdict cannot drift when the probe is edited.

Order is not negotiable:

  1. A-vs-A floor -- baseline against itself, back to back. Until this is
     bit-identical, an A-vs-B difference measures the harness.
  2. A-vs-B -- baseline against ReplaySSM at the shipped L.

Refusals before any GPU work: no gpu-arb claim, a probe above the ~109-token
GDN determinism ceiling, GPU-sampled inputs, or a KDA arm.

Usage:
    python scripts/replayssm/identity_probe.py --base-url http://127.0.0.1:30030
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import urllib.request

sys.path.insert(0, "python")

from sglang.srt.planner.replayssm_identity import (  # noqa: E402
    ProbePlan,
    classify_identity,
    gate_verdict,
    validate_probe_plan,
)

GPU_ARB = "/spinning/gpu-arb/holder-*"


def _require_claim() -> None:
    if not glob.glob(GPU_ARB):
        raise SystemExit(
            "REFUSED: no /spinning/gpu-arb/holder-* claim. This probe restarts "
            "serving on both arms; claim a window first."
        )


def _generate(base_url: str, prompt: str, max_tokens: int, seed: int) -> dict:
    req = urllib.request.Request(
        f"{base_url}/generate",
        data=json.dumps(
            {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0.0,
                    "max_new_tokens": max_tokens,
                    "seed": seed,
                },
                "return_logprob": True,
            }
        ).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as fh:
        return json.load(fh)


def _tokens_and_logprobs(resp: dict):
    meta = resp.get("meta_info", {})
    lp = meta.get("output_token_logprobs") or []
    toks = [int(t[1]) for t in lp] if lp else []
    vals = [float(t[0]) for t in lp] if lp else []
    return toks, vals


def _max_abs_delta(a, b) -> float:
    if len(a) != len(b):
        return float("inf")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30030")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--prompt", default="Explain a binary search in one sentence.")
    ap.add_argument("--seed", type=int, default=945747943)
    args = ap.parse_args()

    plan = ProbePlan(
        max_tokens=args.max_tokens,
        gate="gdn",
        sample_device="cpu",
        arms=("off", "on"),
    )
    validate_probe_plan(plan)
    _require_claim()

    print("[1/2] A-vs-A floor: baseline against itself, back to back")
    r1 = _generate(args.base_url, args.prompt, args.max_tokens, args.seed)
    r2 = _generate(args.base_url, args.prompt, args.max_tokens, args.seed)
    t1, l1 = _tokens_and_logprobs(r1)
    t2, l2 = _tokens_and_logprobs(r2)
    a_vs_a = classify_identity(t1, t2, _max_abs_delta(l1, l2))
    print(
        f"      byte_identical={a_vs_a.byte_identical} "
        f"max|delta|={a_vs_a.max_abs_logit_delta:g}"
    )

    if not a_vs_a.byte_identical:
        v = gate_verdict(a_vs_a, None)
        print(f"\nVERDICT: {'ENABLE' if v.enable else 'REFUSE'} -- {v.reason}")
        return 1

    print(
        "\n[2/2] A-vs-B needs the treatment arm booted with --enable-linear-replayssm."
    )
    print(
        "      Re-run with the server restarted on that flag and pass the "
        "baseline tokens in; this script deliberately does NOT restart "
        "serving itself."
    )
    print(json.dumps({"baseline_tokens": t1, "baseline_logprobs": l1}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
