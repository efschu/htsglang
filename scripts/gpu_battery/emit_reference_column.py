#!/usr/bin/env python3
"""Emit the reference column for an r7c accept boot.

Duty 7 of the r7c queue: every accept number in these boots is reported
AGAINST the reference -- 2.688 (prose) / 3.279 (code), the same FP8 vehicle at
pure NEXTN and K=3, docs/benchmarks/htsglang_tp3.json:87-90. Not against
2.75-2.82: that pair is two cells of a five-axis cross-algo battery and is not
a comparison for a K=3 measurement.

This is a JOIN, not an assessment. It writes measured, reference and their
ratio side by side and stops there. Whether a ratio of 0.45 means the reference
did not reproduce or that the vehicle has a ceiling is exactly the question the
boot exists to inform, and it is not the executor's to answer.

Prompts without a reference value (alphabet, squares, repeat) get a row with
reference=None. Dropping them would hide that they were measured.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "checks"))

from check_common import (  # noqa: E402
    REFERENCE_ACCEPT,
    REFERENCE_COLUMN_KIND,
    REFERENCE_COLUMN_SCHEMA_VERSION,
    REFERENCE_SOURCE,
    is_number,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accept", required=True, help="accept.json from the boot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", required=True)
    args = ap.parse_args()

    if not os.path.exists(args.accept):
        print(f"no {args.accept} -- no reference column possible", file=sys.stderr)
        return 1

    with open(args.accept) as f:
        report = json.load(f)

    rows = []
    for arm in report.get("arms", []):
        if not isinstance(arm, dict):
            continue
        prompt = arm.get("prompt")
        serving = arm.get("serving") or {}
        measured = serving.get("accept_len_mean")
        reference = REFERENCE_ACCEPT.get(prompt)
        ratio = None
        if reference and is_number(measured) and reference:
            ratio = round(float(measured) / float(reference), 4)
        rows.append(
            {
                "prompt": prompt,
                "measured": measured,
                "meta_spec_accept_length": serving.get("spec_accept_length"),
                "reference": reference,
                "ratio": ratio,
                "curve": serving.get("curve"),
                "rounds": serving.get("rounds"),
            }
        )

    payload = {
        "kind": REFERENCE_COLUMN_KIND,
        "schema_version": REFERENCE_COLUMN_SCHEMA_VERSION,
        "boot": args.boot,
        "timestamp": datetime.datetime.now().isoformat(),
        "reference_source": REFERENCE_SOURCE,
        "reference_note": (
            "same FP8 vehicle, pure NEXTN, K=3. Not 2.75-2.82 (two cells of "
            "a five-axis cross-algo battery, not a comparison for a K=3 "
            "measurement). The other end of the scale is the GGUF-Q3 serving "
            "group's own band, 1.15-1.53 (round 7b)."
        ),
        "k": report.get("steps"),
        "tokens": report.get("tokens"),
        "rows": rows,
    }

    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"{'prompt':<10} {'measured':>10} {'reference':>10} {'ratio':>12}")
    for row in rows:
        print(
            f"{str(row['prompt']):<10} {str(row['measured']):>10} "
            f"{str(row['reference']):>10} {str(row['ratio']):>12}"
        )
    print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
