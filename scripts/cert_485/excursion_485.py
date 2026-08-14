#!/usr/bin/env python3
"""#485 excursion analysis: what moved on the binding rank, and by how much.

THE QUESTION THIS TOOL ANSWERS. Certification window s50 recorded gpu1
crossing the 1024 MiB corridor floor at 669 MiB while gpu0 and gpu1's
neighbours reproduced their own minima to 6 and 50 MiB. The runsheet called
that "686 MiB of boot-to-boot spread on the binding rank" and treated it as a
variance to be bounded by repetition.

It is not a variance. The corridor minimum on the binding rank is, by
construction::

    min_free = at_rest_free(TP phase) - seam_transient(tp_to_pp)

and the fork already instruments the right-hand term per flip: the
``[#631 seam-census]`` line names the transient, the baseline it was drawn
from, and the stage that reached the trough. So the excursion is directly
observable in artifacts already on disk, without a window.

WHY A TOOL AND NOT A GREP. The census line is one very long line per flip per
rank -- a few hundred stage marks -- so the interesting quantity (the
distribution of the transient over the whole run) is not something a human
reads off the log. The distribution is the finding: a tight body with one
outlier is a mechanism, a broad smear is noise, and only counting decides
which.

SUBCOMMANDS
-----------
``census``      Transient and weights_refill-step distributions from a boot log.
``decompose``   Stage-by-stage budget of one named flip, against a modal one.
``judge``       The AMENDED criterion (C2'): margin against the WORST observed
                seam transient, not against the spread of window minima.
``smoke``       Red-on-demand self-tests on synthetic fixtures. No artifacts.

CPU only. Reads files; never touches a device, a server, or a process.
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

#: The corridor law. Denominated in the NVML FREE column, never total-used.
LAW_MIB = 1024

#: ``[<ts> PP<n>] [#631 seam-census] <direction> rank <r>: transient <t> MiB
#: (baseline free <b> MiB, trough <g> MiB at '<stage>')``
_HDR = re.compile(
    r"^\[(?P<date>\d{4}-\d{2}-\d{2}) (?P<ts>[0-9:]+) PP\d+\] "
    r"\[#631 seam-census\] (?P<dir>[a-z_]+) rank (?P<rank>\d+): "
    r"transient (?P<transient>\d+) MiB "
    r"\(baseline free (?P<baseline>\d+) MiB, trough (?P<trough>\d+) MiB "
    r"at '(?P<stage>\w+)'\)"
)

#: One stage mark inside the same line.
_STEP = re.compile(r"\| (\w+) free=(-?\d+) step([+-]\d+) slack=(-?\d+)")


class Flip(NamedTuple):
    ts: str
    direction: str
    rank: int
    transient: int
    baseline: int
    trough: int
    stage: str
    steps: Tuple[Tuple[str, int, int, int], ...]

    def step_of(self, stage: str) -> Optional[int]:
        """The free-delta charged to ``stage``, or None if it never ran."""
        for name, _free, step, _slack in self.steps:
            if name == stage:
                return step
        return None


def parse_log(path: str) -> List[Flip]:
    """Every seam-census line in ``path``, in file order.

    Tolerant by construction: a line that does not match the header is not an
    error, it is one of the tens of thousands of other lines in a boot log.
    """
    out: List[Flip] = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if "#631 seam-census" not in line:
                continue
            m = _HDR.match(line)
            if m is None:
                continue
            steps = tuple(
                (name, int(free), int(step), int(slack))
                for name, free, step, slack in _STEP.findall(line)
            )
            out.append(
                Flip(
                    ts=m.group("ts"),
                    direction=m.group("dir"),
                    rank=int(m.group("rank")),
                    transient=int(m.group("transient")),
                    baseline=int(m.group("baseline")),
                    trough=int(m.group("trough")),
                    stage=m.group("stage"),
                    steps=steps,
                )
            )
    return out


def _hist(values: Sequence[int]) -> List[Tuple[int, int]]:
    return sorted(collections.Counter(values).items())


def _fmt_hist(values: Sequence[int], indent: str = "    ") -> str:
    lines = []
    for value, count in _hist(values):
        lines.append(f"{indent}{value:7d} MiB  x{count}")
    return "\n".join(lines)


def cmd_census(args: argparse.Namespace) -> int:
    all_flips: List[Tuple[str, Flip]] = []
    for spec in args.log:
        label, _, path = spec.partition("=")
        if not path:
            label, path = path or spec, spec
        for f in parse_log(path):
            all_flips.append((label or path, f))
    if not all_flips:
        print("no seam-census lines found -- is this a phase-flip boot log?")
        return 2

    direction = args.direction
    rank = args.rank
    sel = [
        (lab, f)
        for lab, f in all_flips
        if f.direction == direction and f.rank == rank
    ]
    print(f"== seam census: direction={direction} rank={rank}")
    print(f"   logs: {', '.join(sorted({lab for lab, _ in all_flips}))}")
    print(f"   flips matched: {len(sel)} of {len(all_flips)} census lines")
    if not sel:
        return 2

    for lab in sorted({lab for lab, _ in sel}):
        sub = [f for lb, f in sel if lb == lab]
        print(f"\n-- {lab}: n={len(sub)}")
        print("   transient (baseline - trough):")
        print(_fmt_hist([f.transient for f in sub], "     "))
        print("   baseline free:")
        print(_fmt_hist([f.baseline for f in sub], "     "))
        print("   trough:")
        print(_fmt_hist([f.trough for f in sub], "     "))
        steps = [
            f.step_of(args.stage) for f in sub if f.step_of(args.stage) is not None
        ]
        if steps:
            print(f"   step charged to '{args.stage}':")
            print(_fmt_hist(steps, "     "))
        stages = collections.Counter(f.stage for f in sub)
        print(f"   stage at trough: {dict(stages)}")
        breaches = [f for f in sub if f.trough < LAW_MIB]
        print(f"   BREACHES (< {LAW_MIB} MiB): {len(breaches)}"
              + (f"  at {[f.ts for f in breaches]}" if breaches else ""))

    body = [f.transient for _, f in sel]
    worst = max(body)
    modal = collections.Counter(body).most_common(1)[0][0]
    rest = sorted(v for v in body if v != worst)
    print(f"\n== POOLED over all logs: n={len(body)}")
    print(f"   modal transient        {modal} MiB")
    print(f"   worst transient        {worst} MiB")
    print(f"   2nd worst              {rest[-1] if rest else 'n/a'} MiB")
    print(f"   worst - 2nd worst      {worst - rest[-1] if rest else 'n/a'} MiB")
    print(f"   baseline needed for the corridor law against the WORST: "
          f"{worst + LAW_MIB} MiB")
    for lab in sorted({lab for lab, _ in sel}):
        sub = [f for lb, f in sel if lb == lab]
        base = collections.Counter(f.baseline for f in sub).most_common(1)[0][0]
        print(f"   {lab}: modal baseline {base} MiB -> margin against worst "
              f"= {base - worst - LAW_MIB:+d} MiB")
    return 0


def cmd_decompose(args: argparse.Namespace) -> int:
    flips = [f for f in parse_log(args.log) if f.rank == args.rank]
    by_ts = {f.ts: f for f in flips if f.direction == args.direction}
    target = by_ts.get(args.at)
    if target is None:
        print(f"no {args.direction} flip on rank {args.rank} at {args.at}; "
              f"have {sorted(by_ts)[:8]}...")
        return 2
    others = [f for ts, f in by_ts.items() if ts != args.at]
    if not others:
        print("only one flip in the log; nothing to compare against")
        return 2
    modal_transient = collections.Counter(
        f.transient for f in others
    ).most_common(1)[0][0]
    modal = next(f for f in others if f.transient == modal_transient)

    print(f"== decompose rank {args.rank} {args.direction}")
    print(f"   TARGET {target.ts}: transient {target.transient} MiB, "
          f"baseline {target.baseline}, trough {target.trough}")
    print(f"   MODAL  {modal.ts}: transient {modal.transient} MiB, "
          f"baseline {modal.baseline}, trough {modal.trough}")
    print()
    m_steps: Dict[str, List[int]] = collections.defaultdict(list)
    for n, _f, s, _sl in modal.steps:
        m_steps[n].append(s)
    t_agg: Dict[str, int] = collections.defaultdict(int)
    t_order: List[str] = []
    for n, _f, s, _sl in target.steps:
        if n not in t_agg:
            t_order.append(n)
        t_agg[n] += s
    m_agg = {n: sum(v) for n, v in m_steps.items()}
    print(f"   {'stage':26s} {'target':>9s} {'modal':>9s} {'delta':>9s}")
    for n in t_order:
        d = t_agg[n] - m_agg.get(n, 0)
        flag = "   <== THE DIFFERENCE" if abs(d) >= 64 else ""
        print(f"   {n:26s} {t_agg[n]:9d} {m_agg.get(n, 0):9d} {d:9d}{flag}")
    return 0


def cmd_judge(args: argparse.Namespace) -> int:
    """The amended criterion.

    C2 as written in the runsheet is ``margin > spread`` over WINDOW MINIMA.
    That treats the minimum as a draw from a distribution whose width you
    estimate by repeating the window. It is the wrong statistic: each window
    contains ~100 flips, so a window's minimum is already an extreme-value
    statistic over its own flips, and comparing two of them throws away 200
    samples to keep 2.

    C2' compares the same margin against the WORST TRANSIENT observed over
    every flip in the reference class. That uses all the samples, and it is
    the quantity the corridor law is actually exposed to: the next flip draws
    from the transient distribution, not from the distribution of window
    minima.
    """
    rows = []
    for spec in args.window:
        label, _, path = spec.partition("=")
        flips = [
            f
            for f in parse_log(path)
            if f.direction == args.direction and f.rank == args.rank
        ]
        if not flips:
            print(f"{label}: no matching flips in {path}")
            return 2
        rows.append((label, flips))
    pooled = [f for _l, fs in rows for f in fs]
    worst = max(f.transient for f in pooled)
    worst_at = next(f for f in pooled if f.transient == worst)
    print("== C2' amended: margin against the WORST observed seam transient")
    print(f"   reference class: {len(pooled)} flips over {len(rows)} window(s)")
    print(f"   worst transient: {worst} MiB (at {worst_at.ts}, "
          f"stage {worst_at.stage})")
    print(f"   required at-rest free: {worst} + {LAW_MIB} = {worst + LAW_MIB} MiB")
    print()
    verdict = True
    for label, flips in rows:
        base = collections.Counter(f.baseline for f in flips).most_common(1)[0][0]
        margin = base - worst - LAW_MIB
        ok = margin >= 0
        verdict &= ok
        obs_min = min(f.trough for f in flips)
        print(f"   {'PASS' if ok else 'FAIL'}  {label:12s} modal baseline "
              f"{base:6d}  observed min {obs_min:6d}  "
              f"margin vs worst {margin:+6d} MiB")
    print()
    print("CERTIFIED (C2')" if verdict else "NOT CERTIFIED (C2')")
    if not verdict:
        print("   The binding rank has never had enough at-rest free memory to")
        print("   absorb the worst seam transient this reference class contains.")
        print("   A window that does not breach did not draw it; that is not a")
        print("   margin, it is a miss.")
    return 0 if verdict else 1


# ---------------------------------------------------------------------------
# smoke: red on demand, on synthetic fixtures only
# ---------------------------------------------------------------------------

def _fixture(ts, direction, rank, baseline, trough, stage, steps) -> str:
    body = " ".join(
        f"| {n} free={f} step{s:+d} slack={sl}" for n, f, s, sl in steps
    )
    return (
        f"[2026-08-12 {ts} PP{rank}] [#631 seam-census] {direction} rank {rank}: "
        f"transient {baseline - trough} MiB (baseline free {baseline} MiB, "
        f"trough {trough} MiB at '{stage}') {body}\n"
    )


def cmd_smoke(args: argparse.Namespace) -> int:
    import tempfile
    import os

    checks: List[Tuple[str, bool]] = []

    def check(name: str, ok: bool) -> None:
        checks.append((name, bool(ok)))

    # 1. the parser finds a census line and its stage marks
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "a.log")
        with open(p, "w") as fh:
            fh.write("noise that is not a census line\n")
            fh.write(
                _fixture("11:00:00", "tp_to_pp", 0, 7725, 1925, "weights_refill",
                         [("plan", 7725, 0, 700), ("weights_refill", 1925, -4278, 652)])
            )
        flips = parse_log(p)
        check("parses one flip from a noisy log", len(flips) == 1)
        check("transient derived from the header", flips[0].transient == 5800)
        check("stage step recovered", flips[0].step_of("weights_refill") == -4278)
        check("absent stage is None", flips[0].step_of("nope") is None)

    # 2. judge FAILS when the worst transient does not fit the baseline
    with tempfile.TemporaryDirectory() as d:
        p1, p2 = os.path.join(d, "w1.log"), os.path.join(d, "w2.log")
        with open(p1, "w") as fh:
            for i in range(5):
                fh.write(_fixture(f"11:0{i}:00", "tp_to_pp", 0, 7725, 1925,
                                  "weights_refill", [("weights_refill", 1925, -4278, 652)]))
            # the outlier: 7055 MiB transient
            fh.write(_fixture("11:44:06", "tp_to_pp", 0, 7723, 668,
                              "weights_refill", [("weights_refill", 668, -5536, 652)]))
        with open(p2, "w") as fh:
            for i in range(5):
                fh.write(_fixture(f"12:0{i}:00", "tp_to_pp", 0, 7314, 1356,
                                  "weights_refill", [("weights_refill", 1356, -4278, 652)]))
        ns = argparse.Namespace(window=[f"w1={p1}", f"w2={p2}"],
                                direction="tp_to_pp", rank=0)
        rc = cmd_judge(ns)
        check("judge REFUSES a clean window whose baseline cannot absorb the "
              "worst transient in the class", rc == 1)

        # 3. and PASSES once the baseline clears worst + law
        p3 = os.path.join(d, "w3.log")
        with open(p3, "w") as fh:
            for i in range(5):
                fh.write(_fixture(f"13:0{i}:00", "tp_to_pp", 0, 8200, 2400,
                                  "weights_refill", [("weights_refill", 2400, -4278, 652)]))
        ns = argparse.Namespace(window=[f"w3={p3}"], direction="tp_to_pp", rank=0)
        check("judge CERTIFIES when the margin clears the worst transient",
              cmd_judge(ns) == 0)

    # 4. decompose points at the stage that differs, not at the whole flip
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.log")
        with open(p, "w") as fh:
            for i in range(3):
                fh.write(_fixture(f"11:0{i}:00", "tp_to_pp", 0, 7725, 1925,
                                  "weights_refill",
                                  [("kv_pack", 7600, -125, 700),
                                   ("weights_refill", 1925, -4278, 652)]))
            fh.write(_fixture("11:44:06", "tp_to_pp", 0, 7723, 668,
                              "weights_refill",
                              [("kv_pack", 7598, -125, 700),
                               ("weights_refill", 668, -5536, 652)]))
        import io
        import contextlib
        buf = io.StringIO()
        ns = argparse.Namespace(log=p, rank=0, direction="tp_to_pp", at="11:44:06")
        with contextlib.redirect_stdout(buf):
            cmd_decompose(ns)
        text = buf.getvalue()
        marked = [
            ln for ln in text.splitlines() if "THE DIFFERENCE" in ln
        ]
        check("decompose marks exactly one stage as the difference",
              len(marked) == 1)
        check("and it is weights_refill",
              bool(marked) and "weights_refill" in marked[0])

    ok = sum(1 for _n, o in checks if o)
    print()
    for name, o in checks:
        print(f"  {'PASS' if o else 'FAIL'}  {name}")
    print(f"\n{ok}/{len(checks)} smoke checks pass")
    return 0 if ok == len(checks) else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("census", help="transient distributions from boot logs")
    c.add_argument("--log", action="append", required=True,
                   metavar="LABEL=PATH", help="repeatable")
    c.add_argument("--rank", type=int, default=0)
    c.add_argument("--direction", default="tp_to_pp")
    c.add_argument("--stage", default="weights_refill")
    c.set_defaults(fn=cmd_census)

    d = sub.add_parser("decompose", help="one flip against the modal one")
    d.add_argument("--log", required=True)
    d.add_argument("--at", required=True, help="HH:MM:SS of the target flip")
    d.add_argument("--rank", type=int, default=0)
    d.add_argument("--direction", default="tp_to_pp")
    d.set_defaults(fn=cmd_decompose)

    j = sub.add_parser("judge", help="the amended criterion C2'")
    j.add_argument("--window", action="append", required=True,
                   metavar="LABEL=PATH", help="repeatable")
    j.add_argument("--rank", type=int, default=0)
    j.add_argument("--direction", default="tp_to_pp")
    j.set_defaults(fn=cmd_judge)

    s = sub.add_parser("smoke", help="self-tests, no artifacts needed")
    s.set_defaults(fn=cmd_smoke)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
