#!/usr/bin/env python3
"""#363 card gate 3 -- per-signal A-vs-A bands from two identical boots.

Gate 3 of DESIGN_363 section 11.7: measure, on this rig, how much each
classifier input differs between two runs that should be the same, and then
check every provisional constant of section 3.4 against its own signal's
band. A threshold inside its own noise is a coin flip with a number written
next to it.

The bands go through the CONTROLLER'S OWN ``signal_band`` and ``clears_band``
(``managers.regime_classifier``), so the experiment and the runtime cannot
drift apart -- if the runtime's idea of "clears" ever changes, this report
changes with it.

WINDOW ALIGNMENT, SETTLED AGAINST REAL TRACES
---------------------------------------------
Deliberately not designed until traces existed, and the real ones decided it.
The 2026-08-01 re-run wrote 34 954 consensus boundaries of which **28 were
active** (a window with any forward round); the rest were idle, and idle
windows carry no share at all. Aligning two runs by absolute boundary index
would therefore compare one run's idle stretch against the other's workload
and report the difference as noise.

So alignment is PER SIGNAL and on the boundaries where that signal EXISTS:

1. take, from each run, the subsequence of boundaries where the signal is
   present (``None`` means absent -- an idle window is no measurement, not a
   zero, and it is dropped rather than filled);
2. resample both subsequences onto ``K = min(len_a, len_b)`` positions on a
   normalised 0..1 timeline, so two runs whose active spans differ in length
   are still compared like-for-like;
3. band = ``signal_band(a_k, b_k)``.

Resampling can only INFLATE the band (it adds a small alignment error to a
real difference), and inflation is the conservative direction: it makes a
threshold harder to clear, never easier. The report states the two active
span lengths so a reader can see how much work the resampling did.

WHAT THIS REPORTS THAT A BAND ALONE WOULD NOT
----------------------------------------------
A threshold can fail in two different ways and they need different fixes:

* **INSIDE_BAND** -- the gap it defends is smaller than the signal's own
  noise. Re-derive the constant.
* **UNREACHED** -- the signal never got near it in either run, so the regime
  it gates cannot be entered at all. That is either a mis-set constant or a
  workload that never produced the shape, and the report says which numbers
  it saw rather than guessing between them.

The re-run already produced one of each kind of evidence, and the second is
the sharper one: ``prefill_share`` peaked at **0.000** across 34 954
boundaries. ``enter_prefill = 0.35`` is not merely high for this rig -- the
signal it reads never moved. The script reports that; it does not re-tune.

Card-less smoke: ``--smoke`` runs the whole pipeline on the trace of the
2026-08-01 re-run, split into two pseudo-arms, plus a synthetic pair with a
band known in advance.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import os
import sys
from typing import Dict, List, Sequence

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
)

from sglang.srt.managers.regime_classifier import (  # noqa: E402
    DEFAULT_ENTER_DECODE,
    DEFAULT_ENTER_PREFILL,
    DEFAULT_EXIT_DECODE,
    DEFAULT_EXIT_PREFILL,
    DEFAULT_WINDOW_ROUNDS,
    KV_ASCEND_MARK,
    KV_DESCEND_MARK,
    clears_band,
    signal_band,
)

#: The margin a THRESHOLD must clear its band by. One band away still leaves
#: the threshold on the wrong side of itself half the time (DESIGN_363 3.4).
THRESHOLD_MARGIN = 2.0

#: Below this many paired samples a band is a number, not a measurement.
MIN_PAIRED_SAMPLES = 8

#: The classifier inputs, and where each lives on a verdict record.
SIGNALS = {
    "prefill_share": "prefill_share",
    "decode_share": "decode_share",
    "occupancy": "occupancy",
    "queued_prompt_tokens": "queued_prompt_tokens",
    "rank_ms_spread_pct": "rank_ms_spread_pct",
}


class Constant:
    """One section-3.4 constant, and how to judge it against the data."""

    def __init__(self, name, value, signal, *, gap=None, partner=None, note=""):
        self.name = name
        self.value = value
        self.signal = signal
        #: The interval the constant defends. For a hysteresis pair it is the
        #: entry-to-exit gap; a bare threshold defends nothing and is judged
        #: on reachability only.
        self.gap = gap
        self.partner = partner
        self.note = note


CONSTANTS = [
    Constant(
        "enter_prefill",
        DEFAULT_ENTER_PREFILL,
        "prefill_share",
        gap=DEFAULT_ENTER_PREFILL - DEFAULT_EXIT_PREFILL,
        partner="exit_prefill",
    ),
    Constant(
        "enter_decode",
        DEFAULT_ENTER_DECODE,
        "decode_share",
        gap=DEFAULT_ENTER_DECODE - DEFAULT_EXIT_DECODE,
        partner="exit_decode",
    ),
    Constant(
        "kv_ascend_mark",
        KV_ASCEND_MARK,
        "occupancy",
        gap=KV_ASCEND_MARK - KV_DESCEND_MARK,
        partner="kv_descend_mark",
        note=(
            "INHERITED from #287 and deliberately not re-derived here: two "
            "independently-chosen thresholds on one physical quantity is how "
            "two controllers end up disagreeing about the same pool. Reported "
            "for information; a failure is a finding for #287, not a licence "
            "to set a second mark."
        ),
    ),
    Constant(
        "spread_veto_pct",
        25.0,
        "rank_ms_spread_pct",
        note="a bare veto threshold: judged on reachability and band only",
    ),
    Constant(
        "PRESTAGE_SINGLE_PROMPT_TOKENS",
        8192,
        "queued_prompt_tokens",
        note="queue mass, not a share; the trace records the total queued",
    ),
]


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _open(path):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)


def load_boundaries(path: str) -> List[Dict]:
    """One entry per consensus BOUNDARY, in order.

    A TP group writes one verdict per rank per boundary. The group agrees by
    construction (a disagreement is a desync, which gate 1 catches), so the
    boundary's value is any rank's -- but it must be counted ONCE, or a
    3-rank run looks like three times the samples and every band shrinks by
    the square root of a lie. With a rank stamp we take one rank; without
    one, the first verdict of each round.
    """
    verdicts = []
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if obj.get("kind") == "verdict":
                verdicts.append(obj)
    ranks = {v.get("rank") for v in verdicts}
    if ranks != {None}:
        pick = sorted(r for r in ranks if r is not None)[0]
        return [v for v in verdicts if v.get("rank") == pick]
    seen, out = set(), []
    for v in verdicts:
        if v.get("round") in seen:
            continue
        seen.add(v["round"])
        out.append(v)
    return out


def series(boundaries: Sequence[Dict], field: str) -> List[float]:
    """The signal's values where it EXISTS, in order.

    ``None`` is dropped, never filled: an idle window carries no share, and a
    zero there would be a different claim that would also drag the band.
    """
    out = []
    for b in boundaries:
        v = b.get(field)
        if v is None:
            continue
        out.append(float(v))
    return out


def resample(values: Sequence[float], k: int) -> List[float]:
    """``values`` onto ``k`` positions of a normalised 0..1 timeline.

    Nearest-neighbour on purpose: interpolating between two samples would
    invent a value the run never produced, and the band would then be partly
    a property of the interpolation.
    """
    n = len(values)
    if n == 0 or k <= 0:
        return []
    if n == k:
        return list(values)
    out = []
    for i in range(k):
        pos = 0 if k == 1 else i * (n - 1) / (k - 1)
        out.append(float(values[int(round(pos))]))
    return out


# ---------------------------------------------------------------------------
# The band, per signal
# ---------------------------------------------------------------------------


def band_for(a: Sequence[Dict], b: Sequence[Dict], field: str) -> Dict:
    sa, sb = series(a, field), series(b, field)
    res = {
        "signal": field,
        "n_a": len(sa),
        "n_b": len(sb),
        "observed_max": max(sa + sb) if (sa or sb) else None,
        "observed_min": min(sa + sb) if (sa or sb) else None,
    }
    if not sa or not sb:
        res.update(
            band=None,
            paired=0,
            status="NO_DATA",
            why="the signal is absent in at least one run",
        )
        return res
    k = min(len(sa), len(sb))
    res["paired"] = k
    res["band"] = signal_band(resample(sa, k), resample(sb, k))
    res["span_ratio"] = round(max(len(sa), len(sb)) / min(len(sa), len(sb)), 3)
    # A band that swallows the signal's whole observed range is not a noise
    # measurement: it says the two arms were doing different things at the
    # positions the alignment paired up. Reporting it as noise would make
    # every threshold look unclearable (or, for a bare threshold, make a real
    # difference look like jitter). Caught on the real trace, where two halves
    # of ONE run -- idle first, workload second -- produced a band equal to
    # the full occupancy range.
    # Compared against the WITHIN-ARM swing, not the pooled range. Two arms
    # that each hold steady at different levels differ by a real, reproducible
    # bias -- that IS the band, and calling it an alignment failure would
    # throw away the measurement. What is not comparable is a band as large as
    # the movement inside a single arm: then the pairing is lining up a quiet
    # stretch of one run against a busy stretch of the other.
    within = max(
        (max(sa) - min(sa)) if sa else 0.0,
        (max(sb) - min(sb)) if sb else 0.0,
    )
    if within > 0 and res["band"] >= 0.9 * within:
        res["status"] = "ARMS_DISSIMILAR"
        res["why"] = (
            f"the band ({res['band']:g}) is as large as the signal's own "
            f"movement within a single arm ({within:g}): "
            f"the two arms were not doing the same thing at the positions the "
            f"alignment paired. That is a comparability failure, not a noise "
            f"floor -- check that both boots ran the same workload and that "
            f"neither was truncated."
        )
        return res
    if k < MIN_PAIRED_SAMPLES:
        res["status"] = "UNDERPOWERED"
        res["why"] = (
            f"only {k} paired sample(s); a band from this few is a number, "
            f"not a measurement. Run a workload that produces more windows "
            f"in which this signal exists."
        )
    else:
        res["status"] = "OK"
        res["why"] = f"{k} paired samples after alignment"
    return res


def judge_constant(c: Constant, bands: Dict[str, Dict]) -> Dict:
    """Per-constant verdict: does it clear its band, and is it reachable?"""
    b = bands.get(c.signal, {})
    out = {
        "constant": c.name,
        "value": c.value,
        "signal": c.signal,
        "gap": c.gap,
        "band": b.get("band"),
        "observed_max": b.get("observed_max"),
        "note": c.note,
    }
    if b.get("status") in (None, "NO_DATA"):
        out.update(verdict="NO_DATA", why=b.get("why", "no band"))
        return out

    # Reachability FIRST. A threshold the signal never approaches is dead
    # whatever its band says: the regime it gates cannot be entered, and
    # reporting "clears its band" for it would be true and useless.
    top = b.get("observed_max")
    if top is not None and top < c.value:
        out.update(
            verdict="UNREACHED",
            why=(
                f"the signal peaked at {top:g}, below the constant "
                f"{c.value:g}: nothing this constant gates can ever have "
                f"triggered in these runs. Either the constant is mis-set for "
                f"this rig, or the workload never produced the shape it reads "
                f"-- this report does not choose between those, and does not "
                f"re-tune."
            ),
        )
        return out
    if b.get("status") in ("UNDERPOWERED", "ARMS_DISSIMILAR"):
        out.update(verdict=b["status"], why=b["why"])
        return out
    if c.gap is None:
        out.update(
            verdict="NO_GAP",
            why=(
                "a bare threshold defends no interval, so there is nothing to "
                f"compare against the band ({b['band']:g}). Reported for "
                "information."
            ),
        )
        return out
    ok = clears_band(c.gap, b["band"], margin=THRESHOLD_MARGIN)
    out.update(
        verdict="CLEARS" if ok else "INSIDE_BAND",
        why=(
            f"gap {c.gap:g} against {THRESHOLD_MARGIN:g}x the measured band "
            f"{b['band']:g} = {THRESHOLD_MARGIN * b['band']:g}"
        ),
    )
    return out


def report(path_a: str, path_b: str) -> Dict:
    a, b = load_boundaries(path_a), load_boundaries(path_b)
    bands = {name: band_for(a, b, field) for name, field in SIGNALS.items()}
    verdicts = [judge_constant(c, bands) for c in CONSTANTS]
    blocking = [
        v
        for v in verdicts
        if v["verdict"] in ("INSIDE_BAND", "UNREACHED", "NO_DATA", "ARMS_DISSIMILAR")
    ]
    return {
        "arm_a": {"path": path_a, "boundaries": len(a)},
        "arm_b": {"path": path_b, "boundaries": len(b)},
        "window_rounds": DEFAULT_WINDOW_ROUNDS,
        "bands": bands,
        "constants": verdicts,
        "passed": not blocking,
        "blocking": [f"{v['constant']}: {v['verdict']}" for v in blocking],
    }


def evidence_entry(rep: Dict, *, note: str = "") -> Dict:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    measured = ", ".join(
        f"{n}={b['band']:g}" for n, b in sorted(rep["bands"].items()) if b.get("band")
    )
    source = (
        f"gate-3 A-vs-A bands {stamp}: {rep['arm_a']['boundaries']} vs "
        f"{rep['arm_b']['boundaries']} boundaries; {measured}; every "
        f"section-3.4 constant checked at {THRESHOLD_MARGIN:g}x its own band "
        f"[{os.path.basename(rep['arm_a']['path'])}; "
        f"{os.path.basename(rep['arm_b']['path'])}]"
    )
    if note:
        source += f" -- {note}"
    return {"f3_bands_measured": {"passed": bool(rep["passed"]), "source": source}}


def render(rep: Dict) -> str:
    lines = [
        f"arm A: {rep['arm_a']['boundaries']} boundaries  {rep['arm_a']['path']}",
        f"arm B: {rep['arm_b']['boundaries']} boundaries  {rep['arm_b']['path']}",
        "",
        f"{'signal':<22}{'band':>12}{'paired':>8}{'max':>12}  status",
    ]
    for name, b in sorted(rep["bands"].items()):
        band = "absent" if b.get("band") is None else f"{b['band']:.6g}"
        top = "-" if b.get("observed_max") is None else f"{b['observed_max']:.6g}"
        lines.append(
            f"{name:<22}{band:>12}{b.get('paired', 0):>8}{top:>12}  {b['status']}"
        )
    lines += ["", f"{'constant':<32}{'value':>10}{'gap':>8}  verdict"]
    for v in rep["constants"]:
        gap = "-" if v["gap"] is None else f"{v['gap']:g}"
        lines.append(f"{v['constant']:<32}{v['value']:>10}{gap:>8}  {v['verdict']}")
        lines.append(f"    {v['why']}")
        if v["note"]:
            lines.append(f"    note: {v['note']}")
    lines += [
        "",
        "PASSED" if rep["passed"] else "NOT PASSED: " + "; ".join(rep["blocking"]),
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------

_REAL = "/spinning/gpu-battery-results/2026-08-01_363_gates_rerun/regime.jsonl.gz"


def _write(path, rows):
    with open(path, "w") as f:
        f.write(json.dumps({"kind": "header", "mode": "observe", "rank": 0}) + "\n")
        for i, r in enumerate(rows):
            rec = {"kind": "verdict", "rank": 0, "round": (i + 1) * 8}
            rec.update(r)
            f.write(json.dumps(rec) + "\n")


def smoke() -> int:
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        # 1. A synthetic pair whose band is known before the code runs.
        a = os.path.join(tmp, "a.jsonl")
        b = os.path.join(tmp, "b.jsonl")
        _write(
            a, [{"decode_share": 0.50 + 0.00, "prefill_share": None} for _ in range(20)]
        )
        _write(b, [{"decode_share": 0.53, "prefill_share": None} for _ in range(20)])
        rep = report(a, b)
        got = rep["bands"]["decode_share"]["band"]
        print(f"[known band ] decode_share band={got:.6g} (expected 0.03)")
        ok &= abs(got - 0.03) < 1e-9

        # 2. Different active-span lengths: alignment must still pair them.
        _write(b, [{"decode_share": 0.53, "prefill_share": None} for _ in range(7)])
        rep = report(a, b)
        bb = rep["bands"]["decode_share"]
        print(
            f"[ragged span] paired={bb['paired']} ratio={bb.get('span_ratio')} "
            f"status={bb['status']}"
        )
        ok &= bb["paired"] == 7 and bb["status"] == "UNDERPOWERED"

        # 3. An UNREACHED constant is called out as such, not as a pass.
        _write(a, [{"prefill_share": 0.01, "decode_share": 0.99} for _ in range(20)])
        _write(b, [{"prefill_share": 0.01, "decode_share": 0.99} for _ in range(20)])
        rep = report(a, b)
        vp = next(v for v in rep["constants"] if v["constant"] == "enter_prefill")
        print(f"[unreached  ] enter_prefill -> {vp['verdict']}")
        ok &= vp["verdict"] == "UNREACHED"
        ok &= not rep["passed"]

        # 4. THE REAL TRACE, split into two pseudo-arms. Not a real A-vs-A
        #    pair -- one boot cannot be two -- but it is the only material
        #    that exercises alignment on the real shape (34 954 boundaries of
        #    which 28 active), which is what this script was deferred for.
        if os.path.exists(_REAL):
            bounds = load_boundaries(_REAL)
            half = len(bounds) // 2
            ha, hb = os.path.join(tmp, "ha.jsonl"), os.path.join(tmp, "hb.jsonl")
            for dst, part in ((ha, bounds[:half]), (hb, bounds[half:])):
                with open(dst, "w") as f:
                    f.write(
                        json.dumps({"kind": "header", "mode": "observe", "rank": 0})
                        + "\n"
                    )
                    for r in part:
                        r = dict(r)
                        r["rank"] = 0
                        f.write(json.dumps(r) + "\n")
            rep = report(ha, hb)
            print("\n[real trace, halves as pseudo-arms]")
            print(render(rep))
            ok &= rep["bands"]["occupancy"]["band"] is not None
            # The halves of one run are NOT an A-vs-A pair, and the guard
            # says so rather than publishing their difference as a floor.
            ok &= rep["bands"]["occupancy"]["status"] == "ARMS_DISSIMILAR"
            print(
                "\n[guard      ] occupancy status="
                f"{rep['bands']['occupancy']['status']} (halves are not a pair)"
            )
        else:
            print(f"[real trace ] SKIPPED, {_REAL} not present")
    print("\nSMOKE OK" if ok else "\nSMOKE FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm-a", help="verdict trace of boot A")
    ap.add_argument("--arm-b", help="verdict trace of boot B (the repeat)")
    ap.add_argument("--evidence", help="evidence JSON to create/merge into")
    ap.add_argument("--note", default="")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        return smoke()
    if not (args.arm_a and args.arm_b):
        ap.error("--arm-a and --arm-b are required, or use --smoke")
    if os.path.realpath(args.arm_a) == os.path.realpath(args.arm_b):
        print(
            "arm A and arm B are the same file: a band measured against "
            "itself is zero by construction and every threshold would clear "
            "it for free. Gate 3 needs TWO boots.",
            file=sys.stderr,
        )
        return 2

    rep = report(args.arm_a, args.arm_b)
    print(json.dumps(rep, indent=2) if args.json else render(rep))
    entry = evidence_entry(rep, note=args.note)
    print("\nevidence entry:\n" + json.dumps(entry, indent=2))
    if not rep["passed"]:
        print("\nGATE 3 NOT PASSED -- not written to the evidence file.")
        return 1
    if args.evidence:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from readout import merge_into  # noqa: PLC0415 -- sibling script

        merge_into(args.evidence, entry)
        print(f"\nwritten to {args.evidence}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
