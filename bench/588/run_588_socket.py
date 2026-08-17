#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#588: decompose the prefill collective socket from a real boot log.

WHY THIS IS A READER AND NOT AN INSTRUMENT. #252 already built the per-rank
compute-vs-wait instrument, and it is **live by default**: the installer runs
unconditionally from ``SchedulerMetricsReporter.__post_init__``
(``metrics_reporter.py:373`` -> ``:458``), gated only on ``device == "cuda"``,
and ``collective_clock.py`` reads no environment flag at all. So closing #588
honestly does not need new wiring or a special boot -- it needs the existing
lines read, decomposed and judged. That is what this does.

Input is any boot log carrying the per-rank prefill lines::

    Prefill rank batch, #new-token: 2048, #cached-token: 0, #chunks: 1,
    gpu-ms: 1905.7 (compute 354.7, wait 1551.0) (wait by family:
    tp.all_reduce 932.2/129x, dcp.all_gather 366.2/48x,
    dcp.all_reduce 250.4/16x, tp.all_gather 0)

WHAT THE ORIGINAL FINDING ACTUALLY WAS, restated from the window-8 record so
this runner judges the real thing rather than the slogan. "Wait is 2-3x
compute" is a PER-RANK statement and the ranks disagree sharply -- in window 8,
TP0 sat at 1551.0/354.7 = **4.4x** while TP1 sat at 1328.4/857.4 = **1.5x**,
on the same batch. Reporting one aggregate ratio hides exactly the asymmetry
that decides where the lever must go, so this runner reports per rank and
refuses to average them.

The collective terms behave differently from each other, and that is the whole
decomposition:

* ``tp.all_reduce`` is IDENTICAL across ranks (932.2/129x on both) -- a true
  floor every rank pays. 129 = 2 x 64 layers + 1, i.e. one reduce for the
  attention output and one for the MLP output per layer. That is the DENSE
  pattern.
* ``dcp.all_gather`` is rank-dependent (366.2 vs 132.7 on 48 calls) -- skew,
  not floor. A lever aimed at the floor will not touch it and vice versa.

Exit: 0 = parsed and judged, 1 = a check failed, 2 = could not run.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

#: One per-rank prefill line. Families are parsed separately because their
#: count varies and `tp.all_gather 0` carries no `/Nx` suffix.
_LINE = re.compile(
    r"TP(?P<rank>\d+)\].*?Prefill rank batch.*?"
    r"gpu-ms:\s*(?P<gpu>[\d.]+)\s*\(compute\s*(?P<compute>[\d.]+),\s*"
    r"wait\s*(?P<wait>[\d.]+)\)"
    r"(?:.*?wait by family:(?P<families>[^)]*))?"
)
_FAMILY = re.compile(r"(?P<name>[a-z_]+\.[a-z_]+)\s+(?P<ms>[\d.]+)(?:/(?P<calls>\d+)x)?")


@dataclass
class RankSample:
    rank: int
    gpu_ms: float
    compute_ms: float
    wait_ms: float
    families: Dict[str, dict] = field(default_factory=dict)

    @property
    def wait_over_compute(self) -> Optional[float]:
        if self.compute_ms <= 0:
            return None
        return self.wait_ms / self.compute_ms

    def dominant_family(self) -> Optional[str]:
        if not self.families:
            return None
        return max(self.families, key=lambda k: self.families[k]["ms"])


def parse_line(line: str) -> Optional[RankSample]:
    m = _LINE.search(line)
    if not m:
        return None
    sample = RankSample(
        rank=int(m.group("rank")),
        gpu_ms=float(m.group("gpu")),
        compute_ms=float(m.group("compute")),
        wait_ms=float(m.group("wait")),
    )
    fam_text = m.group("families") or ""
    for fm in _FAMILY.finditer(fam_text):
        sample.families[fm.group("name")] = {
            "ms": float(fm.group("ms")),
            "calls": int(fm.group("calls")) if fm.group("calls") else 0,
        }
    return sample


def parse_log(text: str) -> List[RankSample]:
    out = []
    for line in text.splitlines():
        s = parse_line(line)
        if s is not None:
            out.append(s)
    return out


def family_medians(samples: Sequence[RankSample]) -> Dict[str, Dict[int, float]]:
    """Per-family, per-rank median ms. The median is the comparable unit.

    A single sample cannot be compared across ranks: the log carries many
    batches per rank, the families move over the run (``dcp.all_gather`` on
    TP0 walks from 5 ms to 442 ms across one boot), and pooling every sample
    into one min/max mixes TIME variation with RANK variation. That mistake
    labelled ``tp.all_reduce`` -- which every rank pays at 871-935 ms -- as
    rank-dependent, which is the opposite of what it is.
    """
    out: Dict[str, Dict[int, float]] = {}
    names = {n for s in samples for n in s.families}
    for name in sorted(names):
        per_rank: Dict[int, List[float]] = {}
        for s in samples:
            if name in s.families:
                per_rank.setdefault(s.rank, []).append(s.families[name]["ms"])
        out[name] = {r: statistics.median(v) for r, v in sorted(per_rank.items())}
    return out


def floor_families(
    samples: Sequence[RankSample], tol_pct: float = 10.0
) -> Dict[str, bool]:
    """Which families are a FLOOR (every rank pays it) vs SKEW (rank-dependent).

    A floor is what a hiding/overlap lever must attack; skew is what a
    balancing lever must attack. Conflating them is how prefill-rebalance came
    to be tried against a term that was never skew (#264 refuted it).

    Compared on per-rank MEDIANS, and the tolerance is deliberately loose:
    the question is "does every rank pay roughly this", not "are the ranks
    bit-identical". On the window-8 record ``tp.all_reduce`` sits at
    930/871/878 ms -- a 6.3% spread, i.e. a floor -- while
    ``dcp.all_gather`` sits at 250/135/146, which is not.
    """
    verdict: Dict[str, bool] = {}
    for name, per_rank in family_medians(samples).items():
        vals = list(per_rank.values())
        if len(vals) < 2:
            continue
        lo, hi = min(vals), max(vals)
        if hi <= 0:
            verdict[name] = True
            continue
        verdict[name] = ((hi - lo) / hi * 100.0) <= tol_pct
    return verdict


@dataclass
class Decomposition:
    samples: List[RankSample] = field(default_factory=list)
    floors: Dict[str, bool] = field(default_factory=dict)

    def render(self) -> str:
        lines = ["## #588 prefill collective socket", ""]
        if not self.samples:
            return "\n".join(lines + ["no per-rank prefill lines found."])
        lines.append(
            "rank   n   compute(med)  wait(med)  wait/compute  min..max  dominant"
        )
        by_rank: Dict[int, List[RankSample]] = {}
        for s in self.samples:
            by_rank.setdefault(s.rank, []).append(s)
        for rank, group in sorted(by_rank.items()):
            ratios = [s.wait_over_compute for s in group if s.wait_over_compute]
            cm = statistics.median([s.compute_ms for s in group])
            wm = statistics.median([s.wait_ms for s in group])
            med = statistics.median(ratios) if ratios else None
            span = (
                f"{min(ratios):.2f}..{max(ratios):.2f}" if ratios else "n/a"
            )
            dom = max(
                (s.dominant_family() for s in group if s.dominant_family()),
                key=lambda n: sum(
                    s.families.get(n, {}).get("ms", 0) for s in group
                ),
                default="-",
            )
            lines.append(
                f"TP{rank:<4d} {len(group):<3d} {cm:11.1f} {wm:10.1f} "
                f"{(f'{med:.2f}x' if med else 'n/a'):>13} {span:>10}  {dom}"
            )
        lines += ["", "family            floor?   per-rank MEDIAN ms"]
        for name, per_rank in family_medians(self.samples).items():
            per = [f"TP{r}={v:.1f}" for r, v in sorted(per_rank.items())]
            flag = self.floors.get(name)
            label = "FLOOR" if flag else ("skew" if flag is False else "?")
            lines.append(f"{name:<17} {label:<8} " + "  ".join(per))
        lines += [
            "",
            "FLOOR = identical across ranks, so a hiding/overlap lever is the "
            "only thing that can move it. skew = rank-dependent, which is a "
            "balancing question and a different lever (#264 refuted rebalance "
            "for the floor term; it was never skew).",
            "",
            "Ratios are reported PER RANK and deliberately not averaged: the "
            "window-8 record had 4.4x on TP0 and 1.5x on TP1 in the same "
            "batch, and an average would have hidden that.",
        ]
        return "\n".join(lines)


def self_test() -> int:
    """Hermetic. Fixtures are the VERBATIM window-8 lines, not invented ones."""
    failures: List[str] = []
    ran: List[str] = []

    def check(label: str, cond: bool) -> None:
        ran.append(label)
        if not cond:
            failures.append(label)

    tp0 = (
        "[2026-08-05 16:03:38 TP0] Prefill rank batch, #new-token: 2048, "
        "#cached-token: 0, #chunks: 1, gpu-ms: 1905.7 (compute 354.7, wait "
        "1551.0) (wait by family: tp.all_reduce 932.2/129x, dcp.all_gather "
        "366.2/48x, dcp.all_reduce 250.4/16x, tp.all_gather 0)"
    )
    tp1 = (
        "[2026-08-05 16:04:01 TP1] Prefill rank batch, #new-token: 2004, "
        "#cached-token: 0, #chunks: 1, gpu-ms: 2185.8 (compute 857.4, wait "
        "1328.4) (wait by family: tp.all_reduce 932.2/129x, dcp.all_reduce "
        "263.3/16x, dcp.all_gather 132.7/48x, tp.all_gather 0)"
    )

    a = parse_line(tp0)
    check("parses a real line", a is not None)
    check("reads the rank", a.rank == 0)
    check("reads compute", a.compute_ms == 354.7)
    check("reads wait", a.wait_ms == 1551.0)
    check("reads a family with calls", a.families["tp.all_reduce"] == {"ms": 932.2, "calls": 129})
    check("reads a zero family without a call count", a.families["tp.all_gather"]["ms"] == 0.0)
    check("dominant family is the all-reduce", a.dominant_family() == "tp.all_reduce")
    check("ratio is per rank", round(a.wait_over_compute, 2) == 4.37)

    b = parse_line(tp1)
    check("second rank parses", b.rank == 1)
    check("second rank ratio differs sharply", round(b.wait_over_compute, 2) == 1.55)
    # THE POINT: the slogan "2-3x" is true of neither rank in the record.
    check(
        "neither rank is actually in the 2-3x band",
        not (2.0 <= a.wait_over_compute <= 3.0)
        and not (2.0 <= b.wait_over_compute <= 3.0),
    )

    samples = [a, b]
    floors = floor_families(samples)
    check("tp.all_reduce is a FLOOR", floors["tp.all_reduce"] is True)
    check("dcp.all_gather is SKEW", floors["dcp.all_gather"] is False)
    # 250.4 vs 263.3 is ~5%: every rank pays it, so it is a FLOOR too. An
    # earlier 1%-tolerance version of this called it skew, which made almost
    # every family "skew" and the classification useless -- the question is
    # "does every rank pay roughly this", not "are the ranks bit-identical".
    check("dcp.all_reduce is a FLOOR as well", floors["dcp.all_reduce"] is True)
    # And the classification must still be able to say SKEW, or it is vacuous.
    check(
        "a 3x rank spread is still called skew",
        floor_families(
            [
                RankSample(0, 1, 1, 1, {"x.y": {"ms": 300.0, "calls": 1}}),
                RankSample(1, 1, 1, 1, {"x.y": {"ms": 100.0, "calls": 1}}),
            ]
        )["x.y"]
        is False,
    )

    # -- rejects
    check("a non-matching line is skipped", parse_line("nothing here") is None)
    check("an empty log parses to nothing", parse_log("") == [])
    check(
        "zero compute does not divide by zero",
        RankSample(0, 1.0, 0.0, 1.0).wait_over_compute is None,
    )
    check("no families means no dominant", RankSample(0, 1.0, 1.0, 1.0).dominant_family() is None)
    # A single rank cannot establish floor-vs-skew, and must not pretend to.
    check("one rank yields no floor verdict", floor_families([a]) == {})

    text = Decomposition(samples, floors).render()
    check("report marks the floor", "FLOOR" in text)
    check("report refuses to average", "not averaged" in text)
    check("empty report says so", "no per-rank prefill lines" in Decomposition().render())

    if failures:
        print("SELF-TEST FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    rejects = sum(1 for x in ran if "not " in x or "no " in x or "skipped" in x)
    print(f"self-test: OK ({len(ran)} checks, {rejects} asserting a refusal)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--log", help="boot log carrying per-rank prefill lines")
    ap.add_argument("--out")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    if not args.log:
        ap.print_help()
        return 2
    try:
        with open(args.log, errors="replace") as f:
            text = f.read()
    except OSError as exc:
        print(f"cannot run: {exc}")
        return 2

    samples = parse_log(text)
    if not samples:
        print(f"cannot run: no per-rank prefill lines in {args.log}.")
        print("The #252 instrument is on by default on any CUDA boot; if the")
        print("lines are absent the boot was not CUDA or the log is truncated.")
        return 2
    decomp = Decomposition(samples, floor_families(samples))
    print(decomp.render())
    if args.out:
        with open(args.out, "w") as f:
            json.dump(
                {
                    "samples": [
                        {
                            "rank": s.rank,
                            "gpu_ms": s.gpu_ms,
                            "compute_ms": s.compute_ms,
                            "wait_ms": s.wait_ms,
                            "wait_over_compute": s.wait_over_compute,
                            "families": s.families,
                        }
                        for s in samples
                    ],
                    "floors": decomp.floors,
                },
                f,
                indent=2,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
