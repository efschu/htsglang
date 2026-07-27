"""Compare two arms from their per-block jsonl records (Task #103).

Reports BOTH axes of the speculative-throughput factorisation

    tok/s  =  verify_rounds_per_second  x  accepted_tokens_per_round
              ^ hardware / pipeline axis  ^ content / speculation axis

because the A-vs-A control measured r(tok/s, accept) = 0.98: nearly all
residual end-to-end scatter is content variance, so the raw tok/s axis has a
~3.5 % detection limit at n=8 while the round-rate axis has ~0.7 %.

Every comparison prints the DETECTION LIMIT derived from the arms' own
scatter. An effect that does not clear it is reported as "below detection
limit", never as a small win.

Usage: cmp_arms.py <tagA> <tagB> [mode] [classes]
  e.g. cmp_arms.py base vocab single code,prosa,misch
"""

import glob
import json
import statistics as st
import sys

A, B = sys.argv[1], sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "single"
CLASSES = (sys.argv[4] if len(sys.argv) > 4 else "code,prosa,misch").split(",")
LOGS = "/spinning/r3val/logs"


def load(tag, mode, cls):
    """Blocks for one arm/mode/class, warm-up-contaminated points dropped."""
    hits = sorted(glob.glob(f"{LOGS}/nf_{tag}_{mode}_{cls}.jsonl")) or \
        sorted(glob.glob(f"{LOGS}/nf_{tag}*{cls}.jsonl"))
    rows = []
    for fn in hits:
        with open(fn) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def pacing_clock(rows):
    """Clock of GPU2, the thermally-limited pacing rank.

    There is NO absolute plateau threshold: single-session runs settle near
    1630 MHz while dual-session runs settle near 1690 MHz, because the duty
    cycle differs. Comparability is therefore established by running the
    IDENTICAL protocol (same warm-up burn, same class order) for every arm
    and checking here that the two arms actually sat at the same clock.
    A clock gap between arms invalidates the comparison outright.
    """
    v = [r["thermal"].get("gpu2", {}).get("sm_mhz_mean", 0) for r in rows]
    v = [x for x in v if x]
    if not v:
        return None
    return st.mean(v), min(v), max(v)


def stats(v):
    m = st.mean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    return m, sd, st.median(v), sd / (len(v) ** 0.5 if v else 1)


print(f"{'class':6s} {'metric':12s} {'A='+A:>18s} {'B='+B:>18s} "
      f"{'delta':>9s} {'det.limit':>10s}  verdict")
print("-" * 96)

for cls in CLASSES:
    ra, rb = load(A, MODE, cls), load(B, MODE, cls)
    if not ra or not rb:
        print(f"{cls:6s} MISSING data (A={len(ra)} B={len(rb)})")
        continue
    ca, cb = pacing_clock(ra), pacing_clock(rb)
    if ca and cb:
        gap = 100 * (cb[0] - ca[0]) / ca[0]
        flag = "  <-- THERMAL MISMATCH, comparison invalid" if abs(gap) > 1.5 else ""
        print(f"{cls:6s} pacing GPU2 clock: A {ca[0]:.0f} MHz ({ca[1]:.0f}-{ca[2]:.0f})"
              f"  B {cb[0]:.0f} MHz ({cb[1]:.0f}-{cb[2]:.0f})  gap {gap:+.2f}%{flag}")
    for metric, fn in (("tok/s", lambda r: r["rate"]),
                       ("accept", lambda r: r["accept"]),
                       ("round_rate", lambda r: r["rate"] / r["accept"])):
        va, vb = [fn(r) for r in ra], [fn(r) for r in rb]
        ma, sda, mda, sema = stats(va)
        mb, sdb, mdb, semb = stats(vb)
        delta = 100 * (mb - ma) / ma
        # pooled 95 % two-sided limit on the difference of two means
        lim = 1.96 * ((sema ** 2 + semb ** 2) ** 0.5) / ma * 100
        verdict = ("BELOW DETECTION LIMIT" if abs(delta) < lim
                   else ("gain" if delta > 0 else "loss"))
        print(f"{cls:6s} {metric:12s} {ma:9.3f}+-{sda:6.3f} {mb:9.3f}+-{sdb:6.3f} "
              f"{delta:+8.2f}% {lim:9.2f}%  {verdict}")
    print(f"{'':6s} {'n':12s} {len(ra):18d} {len(rb):18d}")
    print()
