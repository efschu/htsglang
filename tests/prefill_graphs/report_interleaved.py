"""Report the interleaved eager/graphs pairs on ms per fixed unit of work.

Metric. Every arm runs an IDENTICAL prompt set -- same prompts, same count,
same order, from a fixed seed -- so the work is fixed and ms/prefill is
directly comparable. This is valid because the power limit is identical across
all runs (200/400/200 W on this rig).

Pairing. Arms alternate E,G,E,G,E,G, and the delta is computed PER PAIR
(G_i against E_i, adjacent in time) before the pair deltas are combined. Slow
drift moves both members of a pair together, so pairing removes it rather than
charging it to the treatment.

Validity. Two things carry it, and neither is a clock:
  * the A-vs-A floor -- the spread across the eager replicates is what "no
    treatment at all" is worth on this rig; a paired delta inside that spread
    is not a result;
  * the pairing itself.

Clock and power are ANNOTATION. At a fixed power limit a lower clock often
means MORE work per cycle, not a disadvantage: a power-limited card downclocks
when it is doing more, so low clock with high power is a busy card and low
clock with low power is an idle one. They are printed together so the reader
can read the load character of each arm, and they never reject a point and
never normalise one.

Usage: report_interleaved.py <artifact-dir>
"""

import json
import os
import statistics as st
import sys

POINTS = ("1900", "256c4")
CAPPED_GPUS = ("0", "2")  # the 200 W 3080s; the 5090 at 400 W held flat


def load(out: str, arm: str, point: str):
    p = f"{out}/{arm}_perf_{point}.json"
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def replicates(out: str, prefix: str):
    found, i = [], 1
    while any(load(out, f"{prefix}{i}", pt) for pt in POINTS):
        found.append(f"{prefix}{i}")
        i += 1
    return found


def diag(d) -> str:
    c = d.get("clocks") or {}
    if not c:
        return "no telemetry"
    return " ".join(
        f"g{g}:{v['sm_median']}MHz/{v['watt_median']}W" for g, v in c.items()
    )


def capped_watts(d) -> float:
    c = d.get("clocks") or {}
    v = [c[g]["watt_median"] for g in CAPPED_GPUS if g in c]
    return st.mean(v) if v else float("nan")


def capped_clock(d) -> float:
    c = d.get("clocks") or {}
    v = [c[g]["sm_median"] for g in CAPPED_GPUS if g in c]
    return st.mean(v) if v else float("nan")


def main() -> int:
    out = sys.argv[1]
    e_arms, g_arms = replicates(out, "E"), replicates(out, "G")
    pairs = list(zip(e_arms, g_arms))
    print(f"\neager replicates: {e_arms or 'NONE'}")
    print(f"graph replicates: {g_arms or 'NONE'}")
    print(f"pairs: {pairs or 'NONE'}")
    # Transport is a per-run property; if the arms disagree, the comparison is
    # void regardless of how clean the numbers look.
    tps = set()
    for a in e_arms + g_arms:
        for pt in POINTS:
            d = load(out, a, pt)
            if d:
                tps.add(d.get("transport", "unstamped"))
    print(f"transport: {', '.join(sorted(tps)) or 'unknown'}")
    if len(tps) > 1:
        print("  ABORT: arms used DIFFERENT transports -- not comparable.")
        return 1
    if tps == {"unstamped"}:
        print("  WARNING: artifacts predate transport stamping; confirm by hand.")
    if not pairs:
        print("\nNot enough arms to report. Nothing claimed.")
        return 1

    for point in POINTS:
        eds = {a: load(out, a, point) for a in e_arms}
        gds = {a: load(out, a, point) for a in g_arms}
        if not all(eds.values()) or not all(gds.values()):
            print(f"\n[{point}] missing arms -- skipped")
            continue

        any_d = next(iter(eds.values()))
        print(
            f"\n########## [{point}] ms per prefill "
            f"({any_d['prefills']} prefills of ~{any_d['prompt_tokens_median']} "
            f"tokens, conc={any_d['concurrency']}) ##########"
        )

        walls = [d["wall_seconds"] for d in list(eds.values()) + list(gds.values())]
        print(
            f"  measured walls {min(walls):.1f}-{max(walls):.1f}s "
            f"(5-20 s band is the standard)"
        )

        # Per-arm lines with diagnostic annotation.
        for a in e_arms:
            d = eds[a]
            print(f"  {a:>3} eager   {d['ms_per_prefill']:8.1f} ms   diag[{diag(d)}]")
        for a in g_arms:
            d = gds[a]
            print(f"  {a:>3} graphs  {d['ms_per_prefill']:8.1f} ms   diag[{diag(d)}]")

        # PAIRED deltas: G_i vs E_i, adjacent in time.
        pair_deltas = []
        for ea, ga in pairs:
            e_ms, g_ms = eds[ea]["ms_per_prefill"], gds[ga]["ms_per_prefill"]
            dl = (g_ms / e_ms - 1) * 100
            pair_deltas.append(dl)
            print(
                f"  pair {ea}/{ga}: {dl:+6.2f} %  "
                f"(positive = graphs SLOWER on identical work)"
            )

        # A-vs-A floor from the eager replicates.
        ev = [eds[a]["ms_per_prefill"] for a in e_arms]
        floor = (
            max(abs((b / a - 1) * 100) for a in ev for b in ev) if len(ev) > 1 else None
        )
        delta = st.median(pair_deltas)
        print(
            f"  --> paired delta (median) {delta:+.2f} %   "
            f"A-vs-A floor "
            f"{('%.2f %%' % floor) if floor is not None else 'n/a (1 replicate)'}"
        )

        # Annotation only: state the load character, do not judge on it.
        e_clk = st.mean([capped_clock(d) for d in eds.values()])
        g_clk = st.mean([capped_clock(d) for d in gds.values()])
        e_w = st.mean([capped_watts(d) for d in eds.values()])
        g_w = st.mean([capped_watts(d) for d in gds.values()])
        print(
            f"  annotation (NOT a criterion): capped 3080s "
            f"eager {e_clk:.0f} MHz @ {e_w:.0f} W, "
            f"graphs {g_clk:.0f} MHz @ {g_w:.0f} W"
        )
        if g_clk < e_clk and g_w >= e_w:
            print(
                "    graphs ran at lower clock AND >= power: the card was "
                "busier, not slower-by-clock. Read with the ms figure."
            )

        if any(
            d["cached_tokens_total"] for d in list(eds.values()) + list(gds.values())
        ):
            verdict = "INVALID (cache hits -- prefill was served from cache)"
        elif floor is None:
            verdict = "NO FLOOR (need >= 2 eager replicates)"
        elif abs(delta) <= abs(floor):
            verdict = "INSIDE FLOOR -- not a result"
        else:
            verdict = f"REPORTABLE: {delta:+.2f} % on identical work"
        print(f"  VERDICT: {verdict}")

    print(
        "\nReminder: a delta is reportable on the ms figure and the floor "
        "alone.\nClock and power annotate the load character; they neither "
        "validate nor\nnormalise anything."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
