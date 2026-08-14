#!/usr/bin/env python3
"""#485 planner-cut CERTIFICATION -- criteria encoded so they are stated BEFORE the run.

WHAT IS ACTUALLY BEING CERTIFIED, and it is not the speed.

The gain is not in doubt: +25.5 % deep prefill at 179200 against a same-shift
ship control (65.257 s vs 81.878 s), with per-arm spreads of 1.67 % and 0.31 %.
A 25 % effect against a <2 % instrument spread is not a marginal result and no
further window is needed to believe it.

What IS in doubt is the CORRIDOR MARGIN, and one number says why. The same
configuration -- 40,12,12 / attn 10,3,3, pool 280000, flip ON -- was booted
twice:

    s50   gpu0 5585   gpu1  669   gpu2 6075   -> 2 BREACHES, rank death
    s51   gpu0 5591   gpu1 1355   gpu2 6125   -> 0 breaches, clean

gpu0 reproduces to 6 MiB and gpu2 to 50 MiB. **gpu1, the binding rank, moves
686 MiB** -- against a measured margin above the 1024 MiB floor of 331 MiB.
HANDOFF_695 section 5 states the consequence plainly: "the margin is inside the
spread". That is the certification question, and it is a question about
VARIANCE, so it cannot be answered by one more clean window -- only by enough
windows to bound the variance, or by an explanation that removes s50 from the
reference class.

THE PRE-REGISTERED CRITERIA. Stated here, in code, before the windows run, so
they cannot be adjusted afterwards to fit the result. Every threshold cites
where its number comes from.

Per-window CLEAN (all must hold -- HANDOFF_695 section 4's own definition):
  W1 NVML corridor: 0 breaches below 1024 MiB on every card, 100 ms sampling,
     FREE column, time-series minimum (never total-used).
  W2 Seam census: 0 breaches. The second instrument; on s51 the two agreed to
     1 MiB (1355 vs 1354), so a disagreement is itself a finding.
  W3 Flips: > 0 in both directions and 0 ABANDONED.
  W4 Soak: err == 0 and 0 tracebacks.
  W5 Ranks: all 3 alive at the end.
  W6 Work-matched samples: every scored sample a verified real prefill --
     cache_hit_frac <= 0.05 and 0 rejected. s485's driver already enforces it.
  W7 n_scored >= --min-scored. A spread from 2 samples is a number, not a
     measurement -- the same rule #363 applies to its own bands.
     (This line used to read "Window 1 scored n=2 (arm_a_179k.json)". The
     merged runsheet sections 0 and 5.1b both record window 1 at n=5 -- the
     n=2 was an earlier arm file, superseded before the window was reported.
     Corrected on the port rather than carried, since the threshold argument
     does not depend on which it was.)

Across windows (this is the certification proper):
  C1 Every window CLEAN.
  C2 SPREAD vs MARGIN: with binding-rank minima m_i,
         margin = min(m_i) - 1024
         spread = max(m_i) - min(m_i)
     PASS requires **margin > spread**. This is the direct encoding of "the
     margin is inside the spread": a margin smaller than the movement already
     observed is not a margin, it is a coin flip that has not landed yet.
  C3 The binding rank must be the SAME card in every window. If the binding
     rank moves, the windows are not repeats of one experiment and C2's
     spread is computed over two different quantities.

THE GOVERNING SET IS NOT C1+C2+C3. Revision 2 of the runsheet (2026-08-14)
replaced C2 with C2' and ADDED C4 and C5; section 7 gate item 1 states it
verbatim: "the criteria are now C1 + C2' + C3 + C4 + C5". This file was written
before that revision and spent its life on a branch the line never merged, so
until the port it would print a bare "CERTIFIED" on three of the five governing
criteria -- the one word the entire runsheet is built to protect.

  C2' at_rest_free(binding rank, TP) - max_observed_transient >= 1024 MiB,
      taken over EVERY FLIP in the reference class, not over window minima. A
      window minimum is already an extreme-value statistic over its own ~100
      flips, so C2 compares two numbers and discards the other 194. C2' is
      computed by scripts/cert_485/excursion_485.py judge; this tool cannot
      recompute it (it reads corridor CSVs, not flip censuses) and therefore
      REQUIRES the margin be attested with --c2prime-margin.
  C4  the cut being certified must be the cut an honest gate ADMITTED
      (--cut must equal --c4-admitted-cut). Certifying a cut the admitting
      gate rejects is not a certification; it is a manual override with a
      stopwatch. Runsheet 5.1b/6a.
  C5  the transient census must have SEEN the cutover: every window must
      report a non-zero SEAM_* sample count (--c5-seam-samples). A census with
      no SEAM_* load state is one that could not see the seam, and its
      worst_transient may not be quoted as "the worst state the deployment
      will serve".

C2', C4 and C5 are ATTESTED here rather than recomputed, because their inputs
are flip censuses and gate output, not the corridor series this tool reads.
Attestation is still a gate: unattested is REFUSED, never assumed. The refusal
names which criteria have no evidence.

Note on why the checks are ordered C1 -> C3 -> C2 -> {C2', C4, C5}: the
recorded verdicts in docs/dev/485/WINDOW_VERDICT_485_R12.md quote this tool's
output for a ONE-window run, which must still fail at C2 with the same
sentence. A criteria change that edits an already-recorded result is precisely
what pre-registration exists to prevent, so the reconciliation may only ever
turn a CERTIFIED into a NOT CERTIFIED, never the reverse.

EXIT: 0 certified / 1 not certified (reason printed) / 2 usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

CORRIDOR_FLOOR_MIB = 1024          # the corridor law
CARDS = ("gpu0_free", "gpu1_free", "gpu2_free")

# Reference numbers from the first window, for the report's context lines.
S50 = {"gpu0_free": 5585, "gpu1_free": 669, "gpu2_free": 6075}
S51 = {"gpu0_free": 5591, "gpu1_free": 1355, "gpu2_free": 6125}


def _is_long_form(path: Path) -> bool:
    """Long form is what the runsheet's own sampler emits: one line PER CARD.

        nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits -lms 100
        0, 20053
        1, 32080
        2, 20050

    Wide form carries a `ts_ms,gpu0_free,...` header. The discriminator is that
    header, because long form has none at all.
    """
    with path.open() as f:
        first = f.readline()
    return "gpu0_free" not in first


def _read_corridor_long(path: Path) -> Dict[str, int]:
    """Per-card minimum from the long form, WITHOUT synthesising timestamps.

    The corridor verdict is a per-column minimum over the series, and no
    monotonic re-labelling of time can move a minimum -- so this needs no
    timestamps, which is why reading the shape natively is better than the
    converter that invented them from the file mtime.

    A tick is scored only when every card in it has been seen; a truncated
    final tick is dropped rather than half-counted, since a partial tick would
    otherwise weigh one card's reading against another card's absence.
    """
    mins: Dict[str, int] = {}
    cur: Dict[int, int] = {}

    def commit(tick: Dict[int, int]) -> None:
        if len(tick) != len(CARDS):
            return
        for idx, free in tick.items():
            k = f"gpu{idx}_free"
            if k not in mins or free < mins[k]:
                mins[k] = free

    with path.open() as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if len(parts) != 2:
                continue
            try:
                idx, free = int(parts[0]), int(parts[1])
            except ValueError:
                continue
            if idx in cur:          # a new tick began before the old one closed
                commit(cur)
                cur = {}
            cur[idx] = free
            if len(cur) == len(CARDS):
                commit(cur)
                cur = {}
    commit(cur)                     # a complete final tick still counts
    return mins


def read_corridor(path: Path) -> Dict[str, int]:
    """Per-card minimum of the NVML FREE column over the whole series.

    Accepts BOTH shapes: the wide `ts_ms,gpu0_free,gpu1_free,gpu2_free` this
    tool was written for, and the long form the documented sampler actually
    emits. The R12 shift discovered the mismatch on metal and worked around it
    with a converter in its evidence directory, which meant the chain in the
    runsheet did not run as written.
    """
    if _is_long_form(path):
        return _read_corridor_long(path)
    mins: Dict[str, int] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            for k in CARDS:
                v = (row.get(k) or "").strip()
                if not v.lstrip("-").isdigit():
                    continue
                iv = int(v)
                if k not in mins or iv < mins[k]:
                    mins[k] = iv
    return mins


def breaches(mins: Dict[str, int]) -> List[str]:
    return [k for k, v in mins.items() if v < CORRIDOR_FLOOR_MIB]


class Window:
    def __init__(self, name: str, corridor: Path, arm: Optional[Path],
                 seam_breaches: Optional[int], abandoned: Optional[int],
                 flips: Optional[int], soak_err: Optional[int],
                 tracebacks: Optional[int], ranks_alive: Optional[int]):
        self.name = name
        self.mins = read_corridor(corridor)
        self.arm = json.loads(arm.read_text()) if arm and arm.is_file() else None
        self.seam_breaches = seam_breaches
        self.abandoned = abandoned
        self.flips = flips
        self.soak_err = soak_err
        self.tracebacks = tracebacks
        self.ranks_alive = ranks_alive
        self.problems: List[str] = []

    @property
    def binding(self) -> Optional[str]:
        return min(self.mins, key=self.mins.get) if self.mins else None

    @property
    def binding_min(self) -> Optional[int]:
        b = self.binding
        return self.mins[b] if b else None

    def judge(self, min_scored: int) -> bool:
        p = self.problems
        if not self.mins:
            p.append("W1 corridor series carried no readable samples")
            return False
        br = breaches(self.mins)
        if br:
            p.append(
                f"W1 NVML BREACH below {CORRIDOR_FLOOR_MIB} MiB on "
                + ", ".join(f"{k}={self.mins[k]}" for k in br)
            )
        if self.seam_breaches is None:
            p.append("W2 seam-census breach count not supplied (second instrument absent)")
        elif self.seam_breaches > 0:
            p.append(f"W2 seam census reports {self.seam_breaches} breach(es)")
        if self.abandoned is None:
            p.append("W3 abandoned-flip count not supplied")
        elif self.abandoned > 0:
            p.append(f"W3 {self.abandoned} FLIP ABANDONED")
        if self.flips is not None and self.flips <= 0:
            p.append("W3 no flips observed; the flip path was not exercised")
        if self.soak_err is not None and self.soak_err > 0:
            p.append(f"W4 soak reported {self.soak_err} error(s)")
        if self.tracebacks is not None and self.tracebacks > 0:
            p.append(f"W4 {self.tracebacks} traceback/CUDA error(s)")
        if self.ranks_alive is not None and self.ranks_alive != 3:
            p.append(f"W5 {self.ranks_alive} of 3 ranks alive at the end")

        if self.arm is not None:
            n = self.arm.get("n_scored")
            rej = self.arm.get("n_rejected_cache_hit")
            if rej:
                p.append(f"W6 {rej} sample(s) rejected for cache hits")
            for s in self.arm.get("samples", []):
                chf = s.get("cache_hit_frac")
                if chf is not None and chf > 0.05 and not s.get("warmup"):
                    p.append(f"W6 scored sample idx={s.get('idx')} cache_hit_frac={chf}")
            if n is not None and n < min_scored:
                p.append(
                    f"W7 n_scored={n} below --min-scored {min_scored}: a spread "
                    f"from {n} samples is a number, not a measurement"
                )
        return not p


def judge(
    windows: List[Window],
    min_scored: int,
    verbose: bool = True,
    cut: Optional[str] = None,
    c4_admitted_cut: Optional[str] = None,
    c2prime_margin: Optional[int] = None,
    c5_seam_samples: Optional[Dict[str, int]] = None,
) -> int:
    """C1 + C2 are computed here; C2', C4, C5 are attested (see the module docstring).

    The four attestation arguments default to None = UNATTESTED = refused. They
    can only ever subtract a CERTIFIED, never add one.
    """
    if not windows:
        print("NOT CERTIFIED: no windows supplied")
        return 1

    clean = True
    for w in windows:
        okw = w.judge(min_scored)
        clean = clean and okw
        if verbose:
            b = w.binding
            print(
                f"{'CLEAN' if okw else 'DIRTY':<6} {w.name:<22} "
                + " ".join(f"{k.replace('_free','')}={w.mins.get(k,'?')}" for k in CARDS)
                + (f"  binding={b.replace('_free','')} margin={w.binding_min - CORRIDOR_FLOOR_MIB:+d} MiB"
                   if b else "")
            )
            for pr in w.problems:
                print(f"       - {pr}")

    if not clean:
        print("\nNOT CERTIFIED (C1): at least one window is not CLEAN.")
        return 1

    # C3 -- the binding rank must not move between windows.
    bindings = {w.binding for w in windows}
    if len(bindings) > 1:
        print(
            f"\nNOT CERTIFIED (C3): the binding rank MOVED between windows "
            f"({', '.join(sorted(b.replace('_free','') for b in bindings))}). "
            f"The windows are not repeats of one experiment, so a spread "
            f"computed across them is a spread of two different quantities."
        )
        return 1

    mins = [w.binding_min for w in windows]
    margin = min(mins) - CORRIDOR_FLOOR_MIB
    spread = max(mins) - min(mins)
    band = next(iter(bindings)).replace("_free", "")
    print(f"\nbinding rank      {band}, minima {mins}")
    print(f"margin            {margin} MiB  (min minus the {CORRIDOR_FLOOR_MIB} floor)")
    print(f"observed spread   {spread} MiB  over {len(windows)} window(s)")

    if len(windows) < 2:
        print(
            "\nNOT CERTIFIED (C2): one window cannot bound a variance. The "
            "certification question IS the boot-to-boot spread."
        )
        return 1

    if margin <= spread:
        print(
            f"\nNOT CERTIFIED (C2): the margin ({margin} MiB) does NOT exceed the "
            f"observed boot-to-boot spread ({spread} MiB). The margin is inside "
            f"the spread -- exactly the condition HANDOFF_695 section 5 named, and "
            f"a margin smaller than the movement already observed is not a margin."
        )
        return 1

    # C1, C2 and C3 hold. They are three of the five GOVERNING criteria
    # (runsheet section 6, revision 2). C2', C4 and C5 are attested, not
    # recomputed -- their inputs are flip censuses and gate output, which this
    # tool does not read. Unattested is refused.
    failures: List[tuple] = []

    if c2prime_margin is None:
        failures.append((
            "C2'",
            "UNATTESTED -- no --c2prime-margin supplied. C2 compares two window "
            "minima; C2' is the criterion that replaced it and is computed over "
            "every flip in the reference class by excursion_485.py judge.",
        ))
    elif c2prime_margin < 0:
        failures.append((
            "C2'",
            f"FAILS at {c2prime_margin} MiB: at-rest free on the binding rank does "
            f"not clear the worst observed transient by the 1024 MiB corridor "
            f"floor. A cut that cannot absorb a transient already observed in its "
            f"reference class is not certified by a window that did not draw one.",
        ))

    if c4_admitted_cut is None or cut is None:
        failures.append((
            "C4",
            "UNATTESTED -- --cut and --c4-admitted-cut are both required. A cut may "
            "not be certified while the gate that admitted it is unnamed; and per "
            "the #584 measurement pass, a gate that cannot run at all admits nothing.",
        ))
    elif cut.strip() != c4_admitted_cut.strip():
        failures.append((
            "C4",
            f"FAILS: certifying cut {cut!r} while the gate admitted "
            f"{c4_admitted_cut!r}. Certification follows admission; it does not "
            f"override it (runsheet 5.1b, 6a).",
        ))

    if c5_seam_samples is None:
        failures.append((
            "C5",
            "UNATTESTED -- no --c5-seam-samples supplied. A census with no SEAM_* "
            "load state could not see the cutover, so its worst transient is not "
            "'the worst state the deployment will serve'.",
        ))
    else:
        blind = [w.name for w in windows if c5_seam_samples.get(w.name, 0) <= 0]
        if blind:
            failures.append((
                "C5",
                f"FAILS: window(s) {', '.join(blind)} recorded no SEAM_* census "
                f"samples -- the census could not see the cutover it is being asked "
                f"to certify.",
            ))

    if failures:
        print()
        for criterion, msg in failures:
            print(f"NOT CERTIFIED ({criterion}): {msg}")
        print(
            f"\n(C1, C2 and C3 DO hold: margin {margin} MiB exceeds the observed "
            f"spread {spread} MiB over {len(windows)} windows. That is three of the "
            f"five governing criteria and is not a certification.)"
        )
        return 1

    print(f"\nCERTIFIED (C1 + C2' + C3 + C4 + C5): margin {margin} MiB exceeds the "
          f"observed spread {spread} MiB over {len(windows)} windows; "
          f"C2' margin {c2prime_margin} MiB; cut {cut} is the cut the gate "
          f"admitted; every window's census saw the cutover.")
    return 0


# ---------------------------------------------------------------------------
def cmd_flags(argv) -> int:
    """Verify the window's flags against THIS tree by building the real parser.

    Not a grep. #363 established why: flags derived from annotated dataclass
    field names are invisible to a literal search, so a grep-based check
    reports a false MISSING.
    """
    ap = argparse.ArgumentParser(prog="certify_485.py flags")
    ap.add_argument("--extra", action="append", default=[])
    args = ap.parse_args(argv)

    want = [
        "--pp-solve-cut", "--pp-stage-ratio", "--pp-attn-stage-ratio",
        "--pp-layer-ratio", "--max-total-tokens", "--rank-gpu-memory-mib",
        "--rank-gpu-id", "--enable-phase-flip",
    ] + list(args.extra)

    try:
        from sglang.srt.server_args import ServerArgs
    except Exception as exc:
        print(f"FAIL: cannot import server_args ({type(exc).__name__}: {exc})")
        return 1
    p = argparse.ArgumentParser()
    ServerArgs.add_cli_args(p)
    known = set()
    for a in p._actions:
        known.update(a.option_strings)

    missing = [f for f in want if f not in known]
    for f in want:
        print(f"{'PASS' if f in known else 'FAIL'}  {f}")
    fam = sorted(x for x in known if x.startswith("--phase-flip"))
    print(f"INFO  --phase-flip family: {fam}")
    if missing:
        print(f"-- {len(missing)} flag(s) MISSING from this tree: {missing}")
        return 1
    print(f"-- all {len(want)} flags accepted by server_args")
    return 0


def _fixture_corridor(path: Path, g0: int, g1: int, g2: int) -> None:
    with path.open("w") as f:
        f.write("ts_ms,gpu0_free,gpu1_free,gpu2_free\n")
        for i in range(50):
            f.write(f"{1000+i},{g0+i},{g1+i},{g2+i}\n")


def _win(name: str, path: Path, **kw) -> Window:
    d = dict(seam_breaches=0, abandoned=0, flips=200, soak_err=0,
             tracebacks=0, ranks_alive=3)
    d.update(kw)
    return Window(name, path, None, **d)


def smoke() -> int:
    red = 0
    total = 0
    print("== smoke: each case must behave as stated ==")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        a, b, c = d / "a.csv", d / "b.csv", d / "c.csv"

        # 1. a real breach must be caught
        total += 1
        print("-- case 1: an NVML breach must fail the window")
        _fixture_corridor(a, 5585, 669, 6075)
        rc = judge([_win("s50-like", a)], 5, verbose=False)
        print(f"   [exit={rc}] (1 required)")
        red += 1 if rc == 1 else 0

        # 2. one clean window is not a certification
        total += 1
        print("-- case 2: ONE clean window must not certify")
        _fixture_corridor(a, 5591, 1355, 6125)
        rc = judge([_win("s51", a)], 5, verbose=False)
        print(f"   [exit={rc}] (1 required)")
        red += 1 if rc == 1 else 0

        # 3. margin inside the spread must refuse
        total += 1
        print("-- case 3: margin inside the spread must refuse")
        _fixture_corridor(a, 5591, 1200, 6125)   # margin 176
        _fixture_corridor(b, 5591, 1900, 6125)   # spread 700
        rc = judge([_win("w1", a), _win("w2", b)], 5, verbose=False)
        print(f"   [exit={rc}] (1 required)")
        red += 1 if rc == 1 else 0

        # 4. margin clearing the spread must certify -- WITH the governing
        #    criteria attested. C1+C2+C3 alone no longer certify anything; this
        #    case is what proves the refusals in the pytest suite are a gate and
        #    not an unconditional "no".
        total += 1
        print("-- case 4: margin exceeding the spread must certify (fully attested)")
        _fixture_corridor(a, 5591, 1400, 6125)   # margin 376
        _fixture_corridor(b, 5591, 1450, 6125)   # spread 50
        rc = judge([_win("w1", a), _win("w2", b)], 5, verbose=False,
                   cut="37,14,13", c4_admitted_cut="37,14,13",
                   c2prime_margin=2589, c5_seam_samples={"w1": 99, "w2": 99})
        print(f"   [exit={rc}] (0 required)")
        red += 1 if rc == 0 else 0

        # 5. a moving binding rank must refuse
        total += 1
        print("-- case 5: a binding rank that MOVES must refuse")
        _fixture_corridor(a, 5591, 1400, 6125)   # binds gpu1
        _fixture_corridor(c, 1400, 5591, 6125)   # binds gpu0
        rc = judge([_win("w1", a), _win("w2", c)], 5, verbose=False)
        print(f"   [exit={rc}] (1 required)")
        red += 1 if rc == 1 else 0

        # 6. a thin arm file must fail W7
        total += 1
        print("-- case 6: n_scored below the floor must fail the window")
        _fixture_corridor(a, 5591, 1400, 6125)
        armf = d / "arm.json"
        armf.write_text(json.dumps({"n_scored": 2, "n_rejected_cache_hit": 0,
                                    "samples": []}))
        w = Window("thin", a, armf, seam_breaches=0, abandoned=0, flips=200,
                   soak_err=0, tracebacks=0, ranks_alive=3)
        rc = judge([w], 5, verbose=False)
        print(f"   [exit={rc}] (1 required)")
        red += 1 if rc == 1 else 0

        # 7. a cache-hit sample must fail W6
        total += 1
        print("-- case 7: a scored sample with a cache hit must fail the window")
        armf.write_text(json.dumps({
            "n_scored": 6, "n_rejected_cache_hit": 0,
            "samples": [{"idx": 1, "warmup": False, "cache_hit_frac": 0.4}]}))
        w = Window("cachey", a, armf, seam_breaches=0, abandoned=0, flips=200,
                   soak_err=0, tracebacks=0, ranks_alive=3)
        rc = judge([w], 5, verbose=False)
        print(f"   [exit={rc}] (1 required)")
        red += 1 if rc == 1 else 0

    print(f"== smoke: {red}/{total} cases behaved as required ==")
    return 0 if red == total else 1


def cmd_judge(argv) -> int:
    ap = argparse.ArgumentParser(prog="certify_485.py judge")
    ap.add_argument("--window", action="append", default=[], metavar="NAME=CORRIDOR.csv",
                    help="repeatable; one per certification window")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=ARM.json")
    ap.add_argument("--seam-breaches", action="append", default=[], metavar="NAME=N")
    ap.add_argument("--abandoned", action="append", default=[], metavar="NAME=N")
    ap.add_argument("--ranks-alive", action="append", default=[], metavar="NAME=N")
    # W3/W4 inputs. Without these three the CLI could express only four of the
    # seven per-window CLEAN criteria the runsheet lists, so "flips in both
    # directions", "soak err == 0" and "0 tracebacks" were unenforceable from
    # the documented command rather than merely unenforced.
    ap.add_argument("--flips", action="append", default=[], metavar="NAME=N",
                    help="observed flip count; 0 means the flip path never ran")
    ap.add_argument("--soak-err", action="append", default=[], metavar="NAME=N")
    ap.add_argument("--tracebacks", action="append", default=[], metavar="NAME=N")
    # The governing criteria that this tool attests rather than recomputes.
    ap.add_argument("--cut", help="the cut being certified, e.g. 37,14,13")
    ap.add_argument("--c4-admitted-cut",
                    help="the cut the planner gate ADMITTED (runsheet 5.1b P2)")
    ap.add_argument("--c2prime-margin", type=int, metavar="MIB",
                    help="C2' margin from excursion_485.py judge; negative fails")
    ap.add_argument("--c5-seam-samples", action="append", default=[],
                    metavar="NAME=N",
                    help="SEAM_* census sample count per window; 0 fails C5")
    ap.add_argument("--min-scored", type=int, default=5)
    args = ap.parse_args(argv)

    def kvmap(items):
        out = {}
        for it in items:
            if "=" not in it:
                print(f"bad --option {it!r}; want NAME=VALUE", file=sys.stderr)
                sys.exit(2)
            k, v = it.split("=", 1)
            out[k] = v
        return out

    corr = kvmap(args.window)
    arms = kvmap(args.arm)
    seam = {k: int(v) for k, v in kvmap(args.seam_breaches).items()}
    aband = {k: int(v) for k, v in kvmap(args.abandoned).items()}
    ranks = {k: int(v) for k, v in kvmap(args.ranks_alive).items()}
    flips = {k: int(v) for k, v in kvmap(args.flips).items()}
    soak = {k: int(v) for k, v in kvmap(args.soak_err).items()}
    tbs = {k: int(v) for k, v in kvmap(args.tracebacks).items()}
    seam_samples = ({k: int(v) for k, v in kvmap(args.c5_seam_samples).items()}
                    if args.c5_seam_samples else None)

    if not corr:
        print("need at least one --window NAME=corridor.csv", file=sys.stderr)
        return 2

    windows = []
    for name, path in corr.items():
        p = Path(path).expanduser()
        if not p.is_file():
            print(f"REFUSED: corridor series {p} does not exist")
            return 1
        ap_ = Path(arms[name]).expanduser() if name in arms else None
        windows.append(Window(
            name, p, ap_,
            seam_breaches=seam.get(name),
            abandoned=aband.get(name),
            flips=flips.get(name),
            soak_err=soak.get(name),
            tracebacks=tbs.get(name),
            ranks_alive=ranks.get(name),
        ))
    return judge(
        windows,
        args.min_scored,
        cut=args.cut,
        c4_admitted_cut=args.c4_admitted_cut,
        c2prime_margin=args.c2prime_margin,
        c5_seam_samples=seam_samples,
    )


def cmd_ordering(argv) -> int:
    """Is a corridor breach BEFORE or AFTER a named event? The reference-class test.

    s50's breach could be dismissed as an artifact of its dying rank only if it
    happened AFTER the death. It did not: breach 11:44:06Z, SIGKILL 11:48:55,
    with all three ranks still decoding at 11:48:41. This subcommand is that
    check, so the finding is reproducible rather than quoted.
    """
    import datetime

    ap = argparse.ArgumentParser(prog="certify_485.py ordering")
    ap.add_argument("--corridor", required=True)
    ap.add_argument("--card", default="gpu1_free")
    ap.add_argument("--floor", type=int, default=CORRIDOR_FLOOR_MIB)
    ap.add_argument("--event-utc", help="HH:MM:SS of the event to order against")
    args = ap.parse_args(argv)

    p = Path(args.corridor).expanduser()
    if not p.is_file():
        print(f"REFUSED: {p} does not exist")
        return 1
    if _is_long_form(p):
        # `judge` reads long form natively because a minimum needs no clock.
        # `ordering` DOES reason about wall-clock instants -- it is the
        # reference-class test that decides whether a breach preceded an event
        # -- so it must refuse rather than reason over invented time.
        print(
            f"REFUSED: {p} is the long-form sampler output and carries no ts_ms "
            f"column. `ordering` compares wall-clock instants, and synthesising "
            f"them from a file mtime would decide the reference-class question "
            f"from an artefact of when the file was closed."
        )
        return 1
    rows = list(csv.DictReader(p.open()))
    if not rows:
        print("REFUSED: empty series")
        return 1

    def iso(ms: int) -> str:
        return datetime.datetime.fromtimestamp(
            ms / 1000, datetime.timezone.utc
        ).strftime("%H:%M:%SZ")

    span = (int(rows[0]["ts_ms"]), int(rows[-1]["ts_ms"]))
    print(f"series {iso(span[0])} -> {iso(span[1])}  ({len(rows)} samples)")

    br = [r for r in rows if int(r[args.card]) < args.floor]
    if not br:
        print(f"no {args.card} sample below {args.floor} MiB")
        return 0
    first = min(br, key=lambda r: int(r["ts_ms"]))
    print(f"{len(br)} breach sample(s); FIRST at {iso(int(first['ts_ms']))} "
          f"{args.card}={first[args.card]}")
    for k in CARDS:
        if k != args.card:
            print(f"    at that instant {k}={first[k]}")

    if args.event_utc:
        day = datetime.datetime.fromtimestamp(
            span[0] / 1000, datetime.timezone.utc
        ).date()
        hh, mm, ss = (int(x) for x in args.event_utc.split(":"))
        ev = datetime.datetime.combine(
            day, datetime.time(hh, mm, ss), datetime.timezone.utc
        ).timestamp() * 1000
        delta = (ev - int(first["ts_ms"])) / 1000.0
        if delta > 0:
            print(f"VERDICT   breach precedes the event by {delta:.0f} s "
                  f"-- it CANNOT be an artifact of that event")
        else:
            print(f"VERDICT   breach follows the event by {-delta:.0f} s "
                  f"-- it may be downstream of it")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--smoke":
        return smoke()
    if not argv:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "flags":
        return cmd_flags(rest)
    if cmd == "judge":
        return cmd_judge(rest)
    if cmd == "ordering":
        return cmd_ordering(rest)
    if cmd == "smoke":
        return smoke()
    print(f"unknown command {cmd!r}; use flags | judge | ordering | smoke",
          file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
