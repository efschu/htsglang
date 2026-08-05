"""Solve the soak harness parameters from a crash-reference log.

WHY THIS EXISTS
---------------
Every earlier arm was aimed by a human picking numbers. That produced two
failures in a row, and the second one was subtle enough to nearly pass:

* 2026-08-05 arm 2 saturated mamba at 1.00 and died on allocation asserts,
  because the load's prefix diversity was unbounded.
* 2026-08-05 soak arm 1 ran at fraction 0.17 against the crash's 0.25 and was
  rejected as "too light" -- but the ABSOLUTE occupancy was identical (~16
  slots). Only the denominator differed (96 here, 64 there). Acting on the
  fraction would have raised the real load ABOVE the crash while the number on
  screen moved toward it.

So the harness targets ABSOLUTE quantities and derives them itself. An env
override is not a fix for this: it is the same guess with better ergonomics.

THE ONE RULE
------------
Never compare or rebuild a fraction across different denominators. A reference
log's `mamba usage` is meaningful only against that log's own
`max_mamba_cache_size`. This module converts to absolute slots at parse time,
using the reference's own denominator, and everything downstream is absolute.

WHY NOT JUST READ `mamba num`
-----------------------------
`mamba num` is the absolute count, but it is printed on DECODE lines only. In
CRASH_20260805_boot5 that is 14 samples (all 16, stdev 1.0) against 222
`mamba usage` samples spanning prefill and decode -- and the regime's peak
(0.80, i.e. 51 slots) occurs on PREFILL lines that carry no `mamba num` at all.
Reading only `mamba num` would report a flat regime and miss the peak by 3x.
So the series is reconstructed from `usage x denominator` for full coverage and
cross-checked against `mamba num` wherever both appear on the same line.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

#: Mamba slots one running request structurally holds. Mirrors
#: mamba_pool_floor.mamba_slots_per_running_request: 1 active + P ping-pong +
#: 1 donation + 1 pinned checkpoint. P defaults to 2.
DEFAULT_SLOTS_PER_RUNNING_REQ = 5

_DEN_RE = re.compile(r"max_mamba_cache_size=(\d+)")
_USAGE_RE = re.compile(r"mamba usage: ([0-9.]+)")
_NUM_USAGE_RE = re.compile(r"mamba num: (\d+), mamba usage: ([0-9.]+)")
_RUN_RE = re.compile(r"#running-req: (\d+)")
_NEW_RE = re.compile(r"#new-token: (\d+)")


@dataclass
class Reference:
    path: Path
    denominator: int
    slots: list[int]          # absolute, full coverage
    running: list[int]
    new_tokens: list[int]
    crosscheck_error: float   # max |usage*den - mamba num| where both present

    @property
    def median_slots(self) -> int:
        return int(statistics.median(self.slots))

    @property
    def peak_slots(self) -> int:
        return max(self.slots)

    @property
    def spikiness(self) -> float:
        """Peak over median. 1.0 means flat; the barlink crash ran ~3.2."""
        med = self.median_slots
        return self.peak_slots / med if med else float("inf")


def read_reference(path: Path) -> Reference:
    text = path.read_text(errors="replace")
    den_m = _DEN_RE.search(text)
    if not den_m:
        raise SystemExit(
            f"{path}: no max_mamba_cache_size= in the log. Without the "
            f"reference's own denominator its fractions cannot be converted "
            f"to absolute slots, and converting them against any other "
            f"denominator is the exact error this module exists to prevent."
        )
    den = int(den_m.group(1))
    usages = [float(u) for u in _USAGE_RE.findall(text)]
    if not usages:
        raise SystemExit(f"{path}: no 'mamba usage' samples")
    slots = [round(u * den) for u in usages]

    worst = 0.0
    for num_s, use_s in _NUM_USAGE_RE.findall(text):
        worst = max(worst, abs(float(use_s) * den - int(num_s)))

    return Reference(
        path=path,
        denominator=den,
        slots=slots,
        running=[int(r) for r in _RUN_RE.findall(text)] or [1],
        new_tokens=[int(n) for n in _NEW_RE.findall(text)] or [0],
        crosscheck_error=worst,
    )


@dataclass
class Target:
    mamba_cache: int
    prefix_pool: int
    sessions: int
    derivation: list[str]


def solve(ref: Reference,
          slots_per_req: int = DEFAULT_SLOTS_PER_RUNNING_REQ) -> Target:
    """Solve harness parameters from the reference's ABSOLUTE regime."""
    # 1. Same denominator as the reference. This is what keeps every later
    #    fraction comparable without any rescaling, and it is why the harness
    #    must not simply inherit production's pool size.
    mamba_cache = ref.denominator

    # 2. Concurrency: the reference's typical running set.
    sessions = max(1, int(statistics.median(ref.running)))

    # 3. Prefix diversity. Peak occupancy is the structural slots held by the
    #    running set plus one state per distinct cached prefix, so the pool
    #    that reproduces the reference PEAK is:
    #        pool = peak_slots - running_max * slots_per_req
    #    Solved on the PEAK, not the median: the pool is a ceiling on distinct
    #    prefixes, and the median falls out below it as prefixes are evicted
    #    between bursts. Solving on the median would cap the pool too low to
    #    ever reach the reference's excursions -- which is what made soak arm 1
    #    dead flat (min = median = max).
    running_max = max(ref.running)
    structural = running_max * slots_per_req
    prefix_pool = max(1, ref.peak_slots - structural)

    d = [
        f"reference        : {ref.path.name}",
        f"denominator      : max_mamba_cache_size={ref.denominator} (the "
        f"reference's own; all fractions converted against it)",
        f"absolute slots   : median={ref.median_slots} peak={ref.peak_slots} "
        f"spikiness={ref.spikiness:.2f}x  n={len(ref.slots)}",
        f"cross-check      : max |usage*den - mamba num| = "
        f"{ref.crosscheck_error:.2f} slots"
        + ("  (OK)" if ref.crosscheck_error <= 1.0 else "  (SUSPECT)"),
        f"running-req      : median={sessions} max={running_max}",
        f"-> MAMBA_CACHE   : {mamba_cache}  (match the reference denominator)",
        f"-> SESSIONS      : {sessions}  (reference median concurrency)",
        f"-> PREFIX_POOL   : {prefix_pool}  (peak {ref.peak_slots} - "
        f"{running_max} running x {slots_per_req} structural = {structural})",
    ]
    return Target(mamba_cache, prefix_pool, sessions, d)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference", type=Path)
    ap.add_argument("--slots-per-req", type=int,
                    default=DEFAULT_SLOTS_PER_RUNNING_REQ)
    ap.add_argument("--shell", action="store_true",
                    help="emit sourceable assignments only")
    args = ap.parse_args(argv)

    ref = read_reference(args.reference)
    target = solve(ref, args.slots_per_req)

    if args.shell:
        print(f"MAMBA_CACHE={target.mamba_cache}")
        print(f"PREFIX_POOL={target.prefix_pool}")
        print(f"SESSIONS={target.sessions}")
        return 0

    print("== soak target derived from the crash reference ==")
    for line in target.derivation:
        print("  " + line)
    if ref.spikiness < 1.2:
        print("  NOTE: the reference is nearly flat; a harness matched to it "
              "will not exercise excursion behaviour.")
    if ref.peak_slots >= ref.denominator:
        print("  WARNING: the reference SATURATED its pool "
              f"({ref.peak_slots}/{ref.denominator}). Targeting it reproduces "
              "a saturating regime, which the soak's own gate scores as "
              "REGIME FAIL. Pick the reference for the fault you are "
              "reproducing, not merely the most recent crash.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
