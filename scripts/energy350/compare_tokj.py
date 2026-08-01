#!/usr/bin/env python3
# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Verdict for the #350 phase-2 tok/J validation.

Reads the two #146 harness reports (throughput-objective arm and
energy-objective arm) and states whether the MEASURED ranking agrees with the
solver's prediction.

The prediction is a TRADE, and the verdict checks both halves:

    the energy arm measures FEWER J/token   (the objective did something)
    the energy arm measures FEWER tok/s     (and paid the expected price)

Anything else is reported as a falsification with the numbers attached, not
massaged into a pass. In particular "energy arm wins both" means the
throughput arm was simply mis-planned, and "energy arm loses both" means the
solver's energy model does not describe this rig -- both are findings.

Desk-prepared; no GPU, no server, no torch.
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Dict, Optional

#: The harness prints human-readable lines; these pull the two numbers we
#: compare. Kept deliberately loose (the harness's exact wording has changed
#: before) and FAILING LOUDLY when a number is missing rather than defaulting.
_PATTERNS = {
    "decode_tok_s": re.compile(r"decode[^\n]*?([0-9]+\.?[0-9]*)\s*tok/s", re.I),
    "j_per_decode_token": re.compile(
        r"(?:J/tok|j_per_decode_token)[^\n]*?([0-9]+\.?[0-9]*)", re.I
    ),
}


def parse_report(path: str) -> Dict[str, Optional[float]]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    out: Dict[str, Optional[float]] = {}
    for key, pat in _PATTERNS.items():
        m = pat.search(text)
        out[key] = float(m.group(1)) if m else None
    return out


def verdict(tput: Dict[str, Optional[float]], energy: Dict[str, Optional[float]]) -> int:
    missing = [
        f"{arm}.{k}"
        for arm, d in (("throughput", tput), ("energy", energy))
        for k, v in d.items()
        if v is None
    ]
    if missing:
        print("INCONCLUSIVE: the harness reports are missing " + ", ".join(missing))
        print("Nothing is inferred from a missing number; re-run the arm.")
        return 2

    j_t, j_e = tput["j_per_decode_token"], energy["j_per_decode_token"]
    s_t, s_e = tput["decode_tok_s"], energy["decode_tok_s"]
    print(f"throughput arm: {s_t:.2f} tok/s, {j_t:.4f} J/tok  ({1 / j_t:.2f} tok/J)")
    print(f"energy arm    : {s_e:.2f} tok/s, {j_e:.4f} J/tok  ({1 / j_e:.2f} tok/J)")
    print(
        f"delta         : {(j_t - j_e) / j_t * 100:+.1f}% J/token, "
        f"{(s_e - s_t) / s_t * 100:+.1f}% tok/s (energy arm vs throughput arm)"
    )

    cheaper = j_e < j_t
    slower = s_e < s_t
    if cheaper and slower:
        print("GREEN: the predicted trade reproduces -- fewer J/token, fewer tok/s.")
        return 0
    if cheaper and not slower:
        print(
            "AMBER: the energy arm won BOTH axes. The energy objective is not "
            "wrong, but the throughput arm is then not the throughput optimum "
            "-- investigate the throughput plan before claiming this result."
        )
        return 1
    print(
        "RED: the energy arm did not reduce J/token. The solver's energy model "
        "does not describe this rig at this operating point; do not ship the "
        "objective as validated."
    )
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--throughput", required=True, help="throughput-arm report")
    ap.add_argument("--energy", required=True, help="energy-arm report")
    args = ap.parse_args(argv)
    return verdict(parse_report(args.throughput), parse_report(args.energy))


if __name__ == "__main__":
    sys.exit(main())
