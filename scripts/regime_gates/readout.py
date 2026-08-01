#!/usr/bin/env python3
"""#363 card gate 1 -- read the observe trace and write the evidence entry.

Gate 1 of DESIGN_363 section 11.7: ``summary()["desyncs"] == 0`` over a real
workload. This turns one or more observer traces into a verdict and, when it
passes, into an entry in the evidence file that ``--regime-controller act``
demands -- in exactly the format the gate enforces, including the ``source``
attribution, because an unattributed pass is refused by the gate itself.

WHAT IT REFUSES TO CALL A PASS
------------------------------
* A trace with no ``summary`` line. The summary is written on close, so its
  absence means the server was killed and "zero desyncs" only means "zero so
  far" -- a different claim.
* A trace with no verdicts. A run where the observer never reached a
  consensus boundary has not exercised the thing under test.
* A trace whose regime never changed. Gate 1 is about rank agreement, but a
  workload that produced ONE regime never asked the ranks to agree about
  anything interesting, and recording that as evidence would put a number on
  an experiment that did not run.
* A multi-rank boot where only one rank's trace was supplied. The desync
  count is a property of the group; one rank's file cannot report it.

Card-less smoke: ``--smoke`` synthesises a passing and a failing trace, runs
both through the same verdict function, and prints the two answers.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import tempfile
from typing import Dict, List, Optional

GATE_KEY = "desyncs_zero"


def read_trace(path: str) -> Dict:
    """One trace file -> {verdicts, summary, header}. Tolerates a torn tail."""
    verdicts: List[Dict] = []
    summary: Optional[Dict] = None
    header: Optional[Dict] = None
    torn = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                torn += 1
                continue
            kind = obj.get("kind")
            if kind == "verdict":
                verdicts.append(obj)
            elif kind == "summary":
                summary = obj
            elif kind == "header":
                header = obj
    return {
        "path": path,
        "header": header,
        "verdicts": verdicts,
        "summary": summary,
        "torn_lines": torn,
    }


def regime_transitions(verdicts: List[Dict]) -> List[str]:
    """The regime sequence, collapsed to its changes."""
    out: List[str] = []
    for v in verdicts:
        r = v.get("regime")
        if not out or r != out[-1]:
            out.append(r)
    return out


def judge(traces: List[Dict], *, expect_ranks: int = 1) -> Dict:
    """The gate-1 verdict over one boot's traces (one file per rank)."""
    problems: List[str] = []
    if not traces:
        return {"passed": False, "problems": ["no trace files supplied"]}
    if len(traces) < expect_ranks:
        problems.append(
            f"{len(traces)} trace file(s) for a {expect_ranks}-rank boot: the "
            f"desync count is a property of the GROUP, and one rank's file "
            f"cannot report it. Pass one --trace per rank."
        )

    total_desyncs = 0
    total_verdicts = 0
    all_regimes: List[str] = []
    for t in traces:
        if t["summary"] is None:
            problems.append(
                f"{t['path']}: no summary line. It is written on close, so "
                f"its absence means the server was killed -- 'zero desyncs' "
                f"would only mean 'zero so far'."
            )
            continue
        if not t["verdicts"]:
            problems.append(
                f"{t['path']}: no verdicts. The observer never reached a "
                f"consensus boundary, so nothing was exercised."
            )
        if t["torn_lines"]:
            problems.append(
                f"{t['path']}: {t['torn_lines']} unparsable line(s); the file "
                f"is not a clean record of the run."
            )
        mode = (t["header"] or {}).get("mode")
        if mode not in ("observe", "act"):
            problems.append(f"{t['path']}: header mode is {mode!r}, expected observe")
        total_desyncs += int(t["summary"].get("desyncs", 0))
        total_verdicts += int(t["summary"].get("verdicts", 0))
        if t["summary"].get("uncoordinated"):
            problems.append(
                f"{t['path']}: the run was UNCOORDINATED (multi-rank group "
                f"with no consensus channel), so rank agreement was never "
                f"checked and a zero desync count means nothing."
            )
        all_regimes.extend(regime_transitions(t["verdicts"]))

    distinct = sorted(set(r for r in all_regimes if r))
    if len(distinct) < 2:
        problems.append(
            f"the workload produced only {distinct or ['nothing']}: the ranks "
            f"were never asked to agree about a CHANGE, so this run is not "
            f"evidence for gate 1. Run scripts/regime_gates/workload.py, "
            f"which walks the four named shapes."
        )
    if total_desyncs != 0:
        problems.append(
            f"{total_desyncs} desync(s) recorded. The classifier is not "
            f"rank-uniform on this workload, which BLOCKS act: the same "
            f"disagreement under an actuator is the #94/#194/#259 hang."
        )

    return {
        "passed": not problems,
        "problems": problems,
        "desyncs": total_desyncs,
        "verdicts": total_verdicts,
        "regimes_seen": distinct,
        "transitions": len([1 for _ in all_regimes]),
        "traces": [t["path"] for t in traces],
    }


def evidence_entry(verdict: Dict, *, note: str = "") -> Dict:
    """The gate-format entry. ``source`` is mandatory and is built from the
    facts of the run, not typed by hand -- the gate refuses an unattributed
    pass, and a source a human composes is the easiest thing to overstate."""
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    source = (
        f"observe run {stamp}: {verdict['verdicts']} verdicts across "
        f"{len(verdict['traces'])} rank trace(s), regimes "
        f"{','.join(verdict['regimes_seen'])}, desyncs {verdict['desyncs']} "
        f"[{'; '.join(os.path.basename(p) for p in verdict['traces'])}]"
    )
    if note:
        source += f" -- {note}"
    return {GATE_KEY: {"passed": bool(verdict["passed"]), "source": source}}


def merge_into(path: str, entry: Dict) -> Dict:
    """Merge the entry into an existing evidence file, or create one."""
    data: Dict = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise SystemExit(f"{path} does not hold a JSON object")
    data.update(entry)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    return data


def _synth(path: str, *, desyncs: int, regimes: List[str], summary: bool = True):
    with open(path, "w") as f:
        f.write(json.dumps({"kind": "header", "mode": "observe"}) + "\n")
        for i, r in enumerate(regimes):
            f.write(
                json.dumps({"kind": "verdict", "round": (i + 1) * 8, "regime": r})
                + "\n"
            )
        if summary:
            f.write(
                json.dumps(
                    {
                        "kind": "summary",
                        "verdicts": len(regimes),
                        "desyncs": desyncs,
                        "uncoordinated": False,
                        "actuations": 0,
                    }
                )
                + "\n"
            )


def smoke() -> int:
    """Card-less: a passing trace and three failing ones, same verdict code."""
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        good = os.path.join(tmp, "good.jsonl")
        _synth(
            good,
            desyncs=0,
            regimes=["mixed", "prefill_heavy", "decode_heavy", "mixed"],
        )
        v = judge([read_trace(good)])
        print(f"[pass case ] passed={v['passed']} problems={v['problems']}")
        ok &= v["passed"]
        entry = evidence_entry(v, note="smoke")
        print(f"             source={entry[GATE_KEY]['source']}")
        ok &= bool(entry[GATE_KEY]["source"])

        bad = os.path.join(tmp, "desync.jsonl")
        _synth(bad, desyncs=3, regimes=["mixed", "prefill_heavy"])
        v = judge([read_trace(bad)])
        print(f"[desync    ] passed={v['passed']} :: {v['problems'][0][:70]}")
        ok &= not v["passed"]

        killed = os.path.join(tmp, "killed.jsonl")
        _synth(killed, desyncs=0, regimes=["mixed", "prefill_heavy"], summary=False)
        v = judge([read_trace(killed)])
        print(f"[no summary] passed={v['passed']} :: {v['problems'][0][:70]}")
        ok &= not v["passed"]

        flat = os.path.join(tmp, "flat.jsonl")
        _synth(flat, desyncs=0, regimes=["mixed", "mixed", "mixed"])
        v = judge([read_trace(flat)])
        print(f"[one regime] passed={v['passed']} :: {v['problems'][0][:70]}")
        ok &= not v["passed"]

        # The evidence file round-trips through the gate that will read it.
        ev = os.path.join(tmp, "gate.json")
        merge_into(ev, evidence_entry(judge([read_trace(good)]), note="smoke"))
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "..", "..", "python")
        )
        from sglang.srt.managers.regime_stages import load_gate_evidence

        gate = load_gate_evidence(ev)
        print(f"[gate read ] {GATE_KEY} accepted={GATE_KEY in gate.passed}")
        ok &= GATE_KEY in gate.passed
    print("\nSMOKE OK" if ok else "\nSMOKE FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", action="append", default=[], help="one per rank")
    ap.add_argument("--ranks", type=int, default=1)
    ap.add_argument("--evidence", help="evidence JSON to create/merge into")
    ap.add_argument("--note", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        return smoke()
    if not args.trace:
        ap.error("--trace is required (one per rank), or use --smoke")

    verdict = judge([read_trace(p) for p in args.trace], expect_ranks=args.ranks)
    print(json.dumps(verdict, indent=2))
    entry = evidence_entry(verdict, note=args.note)
    print("\nevidence entry:")
    print(json.dumps(entry, indent=2))
    if not verdict["passed"]:
        print("\nGATE 1 NOT PASSED -- not written to the evidence file.")
        return 1
    if args.evidence:
        merge_into(args.evidence, entry)
        print(f"\nwritten to {args.evidence}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
