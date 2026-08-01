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
import collections
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


def completeness(trace: Dict) -> Dict:
    """Is this trace a COMPLETE record, whatever ended the process?

    The summary line was the original proof, and it turned out to be
    unobtainable: on shutdown the fork's launcher calls
    ``kill_process_tree(..., include_parent=False)``, which SIGKILLs the
    scheduler children. Neither a ``finally`` nor a SIGTERM handler runs under
    SIGKILL, so a normal shutdown can never write that line and a gate that
    requires it can never be recorded. Found on the second gates window --
    after the shutdown hook covering the other three paths had already landed,
    which is the point: the hook was right and the CONTRACT was wrong.

    Completeness is therefore proved from the verdicts themselves. Each rank
    emits one verdict every ``interval`` rounds, in order, so its round
    numbers form an arithmetic sequence; contiguous per rank means nothing was
    lost between the first verdict and the last, which is exactly the property
    the summary line stood in for. A gap is a genuinely torn file and still
    refuses. A summary, when present, remains the strongest ending: it proves
    the process chose to stop rather than being stopped.
    """
    by_rank: Dict[object, List[int]] = {}
    for v in trace["verdicts"]:
        by_rank.setdefault(v.get("rank"), []).append(int(v.get("round") or 0))
    if not by_rank:
        return {"complete": False, "why": "no verdicts", "ranks": 0}
    if list(by_rank) == [None]:
        # No rank stamp (a trace from before it landed). The same proof still
        # works on the MULTIPLICITY: N ranks writing one interleaved file
        # produce each round exactly N times, because every rank reaches every
        # consensus boundary. Contiguous rounds at a constant multiplicity
        # therefore prove both completeness AND how many ranks are in the
        # file -- a subset would show a lower or ragged multiplicity.
        rounds = sorted(by_rank[None])
        counts = collections.Counter(rounds)
        mult = set(counts.values())
        keys = sorted(counts)
        steps = {b - a for a, b in zip(keys, keys[1:])}
        if len(mult) == 1 and len(steps) <= 1 and keys:
            return {
                "complete": True,
                "why": (
                    f"no rank stamp, but {len(keys)} contiguous rounds each "
                    f"appear exactly {mult.pop()} times -- every rank reached "
                    f"every boundary and none is missing"
                ),
                "ranks": counts[keys[0]],
            }
        return {
            "complete": False,
            "why": (
                f"the verdicts carry no rank and the round multiplicity is "
                f"ragged ({sorted(mult)[:4]}) or the rounds are not contiguous "
                f"(steps {sorted(steps)[:4]}), so neither completeness nor the "
                f"rank count can be established"
            ),
            "ranks": 0,
        }
    gaps = []
    for rank, rounds in sorted(by_rank.items(), key=lambda kv: str(kv[0])):
        rounds.sort()
        if len(rounds) < 2:
            continue
        step = rounds[1] - rounds[0]
        for a, b in zip(rounds, rounds[1:]):
            if b - a != step:
                gaps.append(f"rank {rank}: round {a} -> {b} (step should be {step})")
                break
    if gaps:
        return {"complete": False, "why": "; ".join(gaps), "ranks": len(by_rank)}
    return {
        "complete": True,
        "why": (
            "every rank's round sequence is contiguous, so nothing was lost "
            "between the first verdict and the last"
        ),
        "ranks": len(by_rank),
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
    # Checked on RANKS SEEN, not on file count. A group's desync number must
    # come from the whole group -- but since the rank stamp (and the
    # multiplicity proof for traces without it) one interleaved file can carry
    # all of it honestly, and demanding one file per rank would refuse a
    # complete record for its filename. The subset check moved to rank_count
    # below, which is the property that actually matters.

    total_desyncs = 0
    total_verdicts = 0
    all_regimes: List[str] = []
    endings: List[str] = []
    rank_count = 0
    for t in traces:
        # FACTS FIRST, judgement second. This loop used to `continue` past a
        # missing summary, so a trace holding 93 603 verdicts was reported as
        # "0 verdicts, regimes []" next to the real refusal -- the verdict was
        # right and the diagnostics were misleading, which is the failure mode
        # this whole tool exists to avoid (2026-08-01 window).
        all_regimes.extend(regime_transitions(t["verdicts"]))
        comp = completeness(t)
        rank_count = max(rank_count, comp.get("ranks", 0))
        if t["summary"] is not None:
            endings.append(f"{os.path.basename(t['path'])}: clean summary")
            total_desyncs += int(t["summary"].get("desyncs", 0))
            total_verdicts += int(t["summary"].get("verdicts", 0))
            if t["summary"].get("uncoordinated"):
                problems.append(
                    f"{t['path']}: the run was UNCOORDINATED (multi-rank "
                    f"group with no consensus channel), so rank agreement was "
                    f"never checked and a zero desync count means nothing."
                )
        else:
            # Count what the verdicts themselves show, so the reader sees the
            # run's size even when its ending is missing.
            total_verdicts += len(t["verdicts"])
            total_desyncs += sum(1 for v in t["verdicts"] if v.get("agreed") is False)
            endings.append(
                f"{os.path.basename(t['path'])}: killed, "
                f"{'complete' if comp['complete'] else 'TORN'}"
            )
            if not comp["complete"]:
                problems.append(
                    f"{t['path']}: no summary line AND the record is not "
                    f"provably complete -- {comp['why']}. A torn trace cannot "
                    f"report a desync count: 'zero' would only mean 'zero so "
                    f"far'."
                )
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

    if rank_count and rank_count < expect_ranks:
        problems.append(
            f"the traces carry {rank_count} rank(s) for a {expect_ranks}-rank "
            f"boot: the desync count is a property of the GROUP and a subset "
            f"cannot report it."
        )
    return {
        "passed": not problems,
        "problems": problems,
        "ranks_seen": rank_count,
        "endings": endings,
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
        f"{verdict.get('ranks_seen', '?')} rank(s) in "
        f"{len(verdict['traces'])} trace file(s), regimes "
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
        f.write(json.dumps({"kind": "header", "mode": "observe", "rank": 0}) + "\n")
        for i, r in enumerate(regimes):
            f.write(
                json.dumps(
                    {
                        "kind": "verdict",
                        "rank": 0,
                        "round": (i + 1) * 8,
                        "regime": r,
                    }
                )
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

        # THE CASE THE SECOND GATES WINDOW MADE NECESSARY: the server was
        # SIGKILLed by its own launcher (the normal shutdown on this fork), so
        # there is no summary -- but every rank's rounds are contiguous, so
        # the record is provably complete and the counts stand.
        killed_ok = os.path.join(tmp, "killed-complete.jsonl")
        _synth(
            killed_ok,
            desyncs=0,
            regimes=["mixed", "prefill_heavy", "decode_heavy", "mixed"],
            summary=False,
        )
        v = judge([read_trace(killed_ok)])
        print(f"[killed ok ] passed={v['passed']} endings={v['endings']}")
        ok &= v["passed"]

        killed = os.path.join(tmp, "killed.jsonl")
        with open(killed, "w") as f:
            f.write(json.dumps({"kind": "header", "mode": "observe", "rank": 0}) + "\n")
            for rnd, r in ((8, "mixed"), (16, "prefill_heavy"), (40, "mixed")):
                f.write(
                    json.dumps(
                        {"kind": "verdict", "rank": 0, "round": rnd, "regime": r}
                    )
                    + "\n"
                )
        v = judge([read_trace(killed)])
        print(f"[torn      ] passed={v['passed']} :: {v['problems'][0][:70]}")
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
