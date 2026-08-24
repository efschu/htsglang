#!/usr/bin/env python
"""#651: coherence, greedy determinism, and SPEC ACCEPTANCE on a live server.

Three questions, because on this checkpoint they fail independently:

1. Coherence. HTTP 200 proves nothing. A misfiled expert shard or an unloaded
   router gate produces fluent, grammatical, WRONG text -- that is the failure
   mode this file is exposed to (#647/#318), so every prompt has a determined
   answer that is checked rather than eyeballed.

2. Greedy determinism. The same prompts run twice at temperature 0 must give
   byte-identical text. Non-determinism at temp 0 means a race or an
   uninitialised buffer, not a quality issue.

3. Speculative acceptance. This is the one a coherence battery CANNOT see. The
   #290 signature is a drafter that loads, reports "Load weight end", and then
   proposes noise: the target still verifies every token, so output stays
   perfectly correct while acceptance sits at ~1.005 tokens per round. On THIS
   checkpoint that is the expected outcome if blk.40's BF16 router gates were
   renamed to `.qweight` and dropped -- the exact defect mtp_gate_probe.py gates
   at desk. So the two probes are counterparts: one before the window, one on
   the card.

   Acceptance is read from meta_info as completion_tokens / spec_verify_ct.
   NOT from spec_ema_accept_len, which is a smoothed internal counter and has
   its own known measurement trap.

    python accept_probe.py [port] [--spec]

`--spec` makes a low acceptance a FAILURE rather than a note; without it the
acceptance section is reported but not judged (stages a and d run no drafter).
"""

from __future__ import annotations

import json
import sys
import urllib.request

PORT = 30040
EXPECT_SPEC = False
for arg in sys.argv[1:]:
    if arg == "--spec":
        EXPECT_SPEC = True
    else:
        PORT = int(arg)
BASE = f"http://127.0.0.1:{PORT}"

# A drafter proposing noise still yields correct text, so this floor is about
# the DRAFTER being alive, not about speed. 1.005 was the measured #290 corpse;
# anything at or below ~1.05 means essentially nothing is being accepted.
ACCEPT_FLOOR = 1.05

PROBES = [
    ("What is the capital of France? Answer with one word.", ["paris"]),
    ("What is 14 * 3? Reply with just the number.", ["42"]),
    ("What is 31 * 7? Reply with just the number.", ["217"]),
    ("Which planet is known as the Red Planet? One word.", ["mars"]),
    ("Complete the sequence with one number: 2, 4, 8, 16, ", ["32"]),
    (
        "In one short sentence: why does ice float on water?",
        ["less dense", "lower density", "denser", "density"],
    ),
]


def generate(prompt: str) -> tuple[str, dict]:
    """Native /generate, because meta_info is where spec_verify_ct lives."""
    body = json.dumps(
        {
            "text": prompt,
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 96},
        }
    ).encode()
    req = urllib.request.Request(
        f"{BASE}/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read())
    if isinstance(data, list):
        data = data[0]
    return data.get("text", ""), data.get("meta_info", {}) or {}


def main() -> int:
    try:
        with urllib.request.urlopen(f"{BASE}/health", timeout=10):
            pass
    except Exception as exc:
        print(f"REFUSE: no server answering on {BASE} ({type(exc).__name__}).")
        print("The probe judges a LIVE server; it has nothing to say about a dead one.")
        return 2

    rounds: list[list[str]] = []
    metas: list[dict] = []
    for rnd in (1, 2):
        texts = []
        for prompt, _ in PROBES:
            text, meta = generate(prompt)
            texts.append(text)
            if rnd == 1:
                metas.append(meta)
        rounds.append(texts)

    print("=== coherence (round 1, content-judged) ===")
    hits = 0
    for (prompt, accepted), text in zip(PROBES, rounds[0]):
        low = text.lower()
        ok = any(a in low for a in accepted)
        hits += ok
        flat = " ".join(text.split())[:70]
        print(f"  [{'ok ' if ok else 'BAD'}] {prompt[:44]:44s} -> {flat}")
    coherent = hits == len(PROBES)
    print(f"  {hits}/{len(PROBES)} -> {'COHERENT' if coherent else 'INCOHERENT'}")

    print("=== greedy determinism (round 1 vs round 2) ===")
    diffs = [i for i, (a, b) in enumerate(zip(rounds[0], rounds[1])) if a != b]
    deterministic = not diffs
    print(
        f"  {len(PROBES) - len(diffs)}/{len(PROBES)} identical -> "
        f"{'DETERMINISTIC' if deterministic else 'NON-DETERMINISTIC ' + str(diffs)}"
    )

    print("=== speculative acceptance (meta_info) ===")
    total_tok = total_ver = 0
    for meta in metas:
        ct = meta.get("completion_tokens") or 0
        vc = meta.get("spec_verify_ct") or 0
        total_tok += ct
        total_ver += vc
    if not total_ver:
        accept = None
        print("  spec_verify_ct absent or zero -> NO DRAFTER RAN")
    else:
        accept = total_tok / total_ver
        print(
            f"  completion_tokens={total_tok} spec_verify_ct={total_ver} "
            f"-> accept_len={accept:.3f} tokens/round"
        )
        if accept <= ACCEPT_FLOOR:
            print(
                f"  AT OR BELOW THE {ACCEPT_FLOOR} FLOOR: the drafter is proposing "
                "noise. This is the #290/#647 signature -- output stays correct "
                "because the target verifies everything. Check blk.40's router "
                "gates (mtp_gate_probe.py)."
            )

    ok = coherent and deterministic
    if EXPECT_SPEC:
        ok = ok and accept is not None and accept > ACCEPT_FLOOR

    print("VERDICT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
