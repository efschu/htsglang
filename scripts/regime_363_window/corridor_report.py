#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Corridor verdict from a 100 ms NVML FREE series. Refuses; never reassures.

WHY THIS IS A SCRIPT AND NOT A ONE-LINER. The #363 stage-clock window ran two
awk one-liners over its corridor series and both printed a reassuring minimum
-- because they compared the free column as a STRING, so "999" sorted above
"1495". The finding was caught only when the same series went through code
that could be made to fail on demand. That is the rule this file exists to
keep: a corridor verdict comes from an instrument with a proven can-fail arm.

INPUT is what the sampler writes, `nvidia-smi --query-gpu=index,memory.free
--format=csv,noheader,nounits -lms 100`: one `index, free_mib` row per card per
tick. OUTPUT is per card: samples, minimum, and the count of samples BELOW the
floor. Exit 1 on any breach, or on a series too short to be a time series.

THE FLOOR IS 1024 MiB PER CARD, and it is a LAW, not a target: the free column
is the user's own reserve, measured as a continuous time-series minimum under
load and never as a boot snapshot. `total - used` is not an accepted
substitute -- a carve-out of ~424/518 MiB is invisible to that subtraction.

    python scripts/regime_363_window/corridor_report.py --csv $OUT/corridor.csv
    python scripts/regime_363_window/corridor_report.py --smoke   # can-fail proof
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Optional, Tuple

#: The corridor law, MiB of NVML FREE per card, held continuously.
DEFAULT_FLOOR_MIB = 1024

#: Below this many samples per card the file is not a time series. A 12x
#: undersampled series' minimum is a LOWER bound on nothing (#493).
MIN_SAMPLES = 100


class CorridorError(RuntimeError):
    pass


def parse_csv(text: str) -> Dict[int, List[int]]:
    """``{card_index: [free_mib, ...]}`` from the sampler's own format.

    Every field is converted to an INT here, at the parse, which is the whole
    point of the file: a comparison against a string is the defect this
    replaces, and it cannot recur if no string reaches the comparison.
    """
    out: Dict[int, List[int]] = {}
    for line_no, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            raise CorridorError(f"line {line_no}: expected 'index, free_mib', got {line!r}")
        try:
            index = int(parts[0])
            free = int(float(parts[1]))
        except ValueError as exc:
            raise CorridorError(f"line {line_no}: {line!r} is not numeric ({exc})") from exc
        out.setdefault(index, []).append(free)
    if not out:
        raise CorridorError("the series is empty; a truncated file has no minimum")
    return out


def verdict(
    series: Dict[int, List[int]],
    *,
    floor_mib: int = DEFAULT_FLOOR_MIB,
    min_samples: int = MIN_SAMPLES,
) -> Tuple[bool, List[str]]:
    """``(passed, lines)``. Passed means: enough samples AND zero breaches."""
    lines: List[str] = []
    passed = True
    for index in sorted(series):
        vals = series[index]
        low = min(vals)
        below = sum(1 for v in vals if v < floor_mib)
        state = "OK" if below == 0 else f"BREACH x{below}"
        if below:
            passed = False
        if len(vals) < min_samples:
            passed = False
            state += f" / TOO SHORT ({len(vals)} < {min_samples} samples)"
        lines.append(
            f"gpu{index}: {len(vals)} samples, min {low} MiB, "
            f"{below} below {floor_mib} MiB -- {state}"
        )
    return passed, lines


def _smoke() -> int:
    """Can-fail proof: the same series, once clean and once with a plant."""
    clean = "\n".join(f"{g}, {1500 + i}" for i in range(120) for g in (0, 1, 2))
    ok, lines = verdict(parse_csv(clean))
    print("clean series:", "PASS" if ok else "FAIL")
    for line in lines:
        print("  " + line)
    planted = clean + "\n1, 900\n"
    bad, lines = verdict(parse_csv(planted))
    print("planted 900 MiB sample:", "PASS" if bad else "FAIL (expected)")
    for line in lines:
        print("  " + line)
    string_trap = "0, 999\n" * 120
    trap_ok, trap_lines = verdict(parse_csv(string_trap))
    print("all-999 series (the string-compare trap):", "PASS" if trap_ok else "FAIL (expected)")
    for line in trap_lines:
        print("  " + line)
    good = ok and not bad and not trap_ok
    print("smoke:", "3/3" if good else "FAILED -- the instrument cannot fail on demand")
    return 0 if good else 1


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", help="corridor series from the 100 ms sampler")
    p.add_argument("--floor-mib", type=int, default=DEFAULT_FLOOR_MIB)
    p.add_argument("--min-samples", type=int, default=MIN_SAMPLES)
    p.add_argument("--smoke", action="store_true", help="prove it can fail")
    args = p.parse_args(argv)
    if args.smoke:
        return _smoke()
    if not args.csv:
        print("refused: --csv or --smoke", file=sys.stderr)
        return 2
    try:
        with open(args.csv) as fh:
            series = parse_csv(fh.read())
        passed, lines = verdict(
            series, floor_mib=args.floor_mib, min_samples=args.min_samples
        )
    except (OSError, CorridorError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    for line in lines:
        print(line)
    print("CORRIDOR:", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
