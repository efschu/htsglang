#!/usr/bin/env python3
"""#363 card gate 2 -- F2 self-conditioning replay against a LIVE trace.

Phase 2 proved F2 on a synthetic workload; gate 2 of DESIGN_363 section 11.7
is the same question asked of a real observe run: does the classifier react to
anything the CONTROLLER caused, rather than to the load?

WHY AN OBSERVE TRACE CAN ANSWER IT AT ALL
------------------------------------------
Observe actuates nothing. So an observe trace is, by construction, an
OPEN-LOOP record: every regime in it was produced by the workload. Replaying
its own inputs through a fresh classifier must therefore reproduce its own
regime sequence exactly. Two things can break that, and they are the two
findings this tool exists to produce:

1. **Non-determinism.** The replay diverges from the recorded sequence even
   though nothing actuated. The classifier depends on something the trace does
   not carry -- which means the phase-3 act path would be steering on an input
   nobody can reconstruct, and F2 cannot be evaluated at all until it is
   fixed.
2. **Self-conditioning, counterfactually.** Feeding the SAME inputs to a
   classifier whose stage selection is allowed to change the capacity
   denominator (the mechanism DESIGN_363 section 7.3 named and phase 2
   reproduced) produces transitions the open-loop trace does not contain.
   Those transitions would have been the controller's own doing.

The second is the real gate. It is a counterfactual on measured inputs, which
is the strongest thing a run that never acted can say about acting -- and it
is why gates 1+2 ride along on an observe boot instead of needing one of their
own.

Card-less smoke: ``--smoke`` builds a synthetic trace with a known answer and
runs both arms over it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
)

from sglang.srt.managers.regime_classifier import (  # noqa: E402
    REGIME_DECODE_HEAVY,
    REGIME_PREFILL_HEAVY,
    RegimeSample,
    RegimeSensor,
    Stage,
    StageTable,
)

#: Fields the replay needs off each verdict. A trace missing any of them
#: cannot be replayed, and saying so is better than replaying a guess.
REQUIRED = (
    "round",
    "prefill_share",
    "decode_share",
    # The NUMERATOR, not the ratio. Gate 2's counterfactual varies the
    # capacity denominator, so a trace carrying only ``occupancy`` would hold
    # occupancy constant under exactly the change it exists to expose. A trace
    # without this field is unreplayable and says so.
    "held_tokens",
    "queued_reqs",
    "queued_prompt_tokens",
    "regime",
)


def load(path: str) -> Tuple[List[Dict], Optional[Dict]]:
    verdicts, summary = [], None
    with open(path) as f:
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
            elif obj.get("kind") == "summary":
                summary = obj
    return verdicts, summary


def _sample(v: Dict, *, capacity: int, window: int = 64) -> RegimeSample:
    """Rebuild the classifier input from a recorded verdict.

    The trace stores SHARES rather than round counts, so the counts are
    reconstructed against a fixed window. That is exact for the classifier,
    which reads only the ratio -- and the reconstruction is stated here rather
    than hidden, because a replay whose inputs differ from the original is not
    a replay.

    ``held_tokens`` is taken VERBATIM and ``capacity`` is the arm's, never the
    trace's: held tokens are a property of the load and capacity is the
    property the controller changes. Deriving held from a recorded occupancy
    would move the numerator with the denominator and make every
    counterfactual come back clean.
    """
    share = v.get("prefill_share")
    if share is None:
        p = d = 0
    else:
        p = int(round(float(share) * window))
        d = window - p
    held = int(v.get("held_tokens") or 0)
    return RegimeSample(
        round_index=int(v.get("round") or 0),
        prefill_rounds=p,
        decode_rounds=d,
        held_tokens=held,
        capacity_tokens=capacity,
        queued_reqs=int(v.get("queued_reqs") or 0),
        queued_prompt_tokens=int(v.get("queued_prompt_tokens") or 0),
        rank_ms_spread_pct=v.get("sample_spread_pct"),
    )


def one_rank(verdicts: List[Dict]) -> List[Dict]:
    """The GROUP's verdict sequence: one entry per consensus boundary.

    A TP group writes one verdict per rank per boundary. Replaying all of them
    through a single sensor feeds it every boundary N times, which shortens
    its hysteresis windows by N and manufactures transitions -- the live trace
    replayed 13 where the run recorded 7, and the tool called that
    NON-DETERMINISM. It was the replay's own doing.

    The verdict at a boundary is a GROUP verdict (the ranks agree by
    construction, and a disagreement is a desync gate 1 catches), so the
    honest reconstruction keeps one entry per round. With a rank stamp the
    same thing is done by selecting a rank; without one, by de-duplicating on
    the round, which is equivalent when the ranks agree and refuses to hide it
    when they do not.
    """
    ranks = {v.get("rank") for v in verdicts}
    if ranks != {None}:
        pick = sorted(r for r in ranks if r is not None)[0]
        return [v for v in verdicts if v.get("rank") == pick]
    seen = set()
    out = []
    for v in verdicts:
        rnd = v.get("round")
        if rnd in seen:
            continue
        seen.add(rnd)
        out.append(v)
    return out


def transitions(seq: List[str]) -> List[str]:
    return [r for i, r in enumerate(seq) if i == 0 or r != seq[i - 1]]


def replay_open(verdicts: List[Dict], capacity: int) -> List[str]:
    """Arm 1: the trace's own inputs, capacity FIXED. Must reproduce it."""
    sensor = RegimeSensor(enter_window=2, exit_window=4)
    return [sensor.observe(_sample(v, capacity=capacity)) for v in verdicts]


def replay_closed(
    verdicts: List[Dict], booted: Stage, target: Stage, *, guarded: bool = True
) -> List[str]:
    """Arm 2: the same inputs, but capacity FOLLOWS the stage a controller
    would have selected -- the counterfactual. Any transition here that the
    open arm does not produce would have been self-caused.

    ``guarded=False`` drops the admissibility interlock, which is how the tool
    reports WHY a clean result is clean: if the unguarded arm is also clean the
    workload never approached the trap, and if only the guarded arm is clean
    the interlock is what earned it.
    """
    table = StageTable([booted, target], reference=booted.name)
    sensor = RegimeSensor(enter_window=2, exit_window=4)
    stage = booted
    out: List[str] = []
    for v in verdicts:
        sample = _sample(v, capacity=stage.max_total_num_tokens)
        regime = sensor.observe(sample)
        out.append(regime)
        if guarded:
            chosen, _why = table.select(regime, sample, current=stage.name)
        else:
            cand = table.for_regime(regime)
            chosen = cand if cand is not None and cand.name != stage.name else None
        if chosen is not None:
            stage = chosen
    return out


def judge(verdicts: List[Dict], booted: Stage, target: Stage) -> Dict:
    missing = sorted({k for v in verdicts[:1] for k in REQUIRED if k not in v})
    if missing:
        return {
            "passed": False,
            "problems": [
                f"the trace is missing {missing}; it cannot be replayed, and "
                f"replaying a guess is not evidence"
            ],
        }
    # One entry per boundary, or the replay outpaces the run it is replaying.
    verdicts = one_rank(verdicts)
    recorded = transitions([v["regime"] for v in verdicts])
    cap = booted.max_total_num_tokens
    open_seq = transitions(replay_open(verdicts, cap))
    closed_seq = transitions(replay_closed(verdicts, booted, target))
    unguarded_seq = transitions(replay_closed(verdicts, booted, target, guarded=False))

    problems: List[str] = []
    if open_seq != recorded:
        problems.append(
            f"NON-DETERMINISTIC: the open-loop replay ({open_seq}) does not "
            f"reproduce the recorded sequence ({recorded}). The classifier "
            f"depends on something the trace does not carry, so F2 cannot be "
            f"evaluated until that input is recorded."
        )
    extra = [t for t in closed_seq if t not in open_seq]
    if extra:
        problems.append(
            f"SELF-CONDITIONING: the counterfactual closed loop produces "
            f"{extra}, which the open-loop trace does not. Those transitions "
            f"would have been the controller's own doing -- the DESIGN_363 "
            f"section 7.3 mechanism, on measured inputs."
        )
    # Why a clean result is clean. Not a pass/fail input -- a finding: if the
    # unguarded arm is also clean the workload never approached the trap, and
    # this run is weak evidence rather than strong.
    unguarded_extra = [t for t in unguarded_seq if t not in open_seq]
    return {
        "passed": not problems,
        "problems": problems,
        "recorded": recorded,
        "open_loop": open_seq,
        "closed_loop": closed_seq,
        "unguarded_closed_loop": unguarded_seq,
        "interlock_was_load_bearing": bool(unguarded_extra) and not extra,
        "trap_approached": bool(unguarded_extra),
        "verdicts": len(verdicts),
    }


def evidence_entry(verdict: Dict, *, traces: List[str], note: str = "") -> Dict:
    import datetime

    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    source = (
        f"F2 live replay {stamp}: {verdict['verdicts']} verdicts, recorded "
        f"{'->'.join(verdict['recorded'])}, open-loop reproduces it, "
        f"counterfactual adds nothing "
        f"[{'; '.join(os.path.basename(p) for p in traces)}]"
    )
    if note:
        source += f" -- {note}"
    return {"f2_live_replay": {"passed": bool(verdict["passed"]), "source": source}}


def _stages(booted_pool: int, target_pool: int):
    booted = Stage(
        name="booted",
        regime=REGIME_DECODE_HEAVY,
        weight_vector=None,
        kv_token_vector=(7, 3, 3),
        vram_budget_mib=(29607, 17780, 17780),
        max_total_num_tokens=booted_pool,
        measured_gain_pct=0.0,
        measured_band_pct=0.0,
        flip_cost_s=0.0,
    )
    target = Stage(
        name="prefill",
        regime=REGIME_PREFILL_HEAVY,
        weight_vector=None,
        kv_token_vector=(2, 11, 10),
        vram_budget_mib=(29607, 17780, 17780),
        max_total_num_tokens=target_pool,
        measured_gain_pct=22.6,
        measured_band_pct=4.2,
        flip_cost_s=0.4,
    )
    return booted, target


def _record_trace(path: str, inputs: List[Dict], capacity: int) -> None:
    """Write a trace by RUNNING the classifier over ``inputs``.

    Generated rather than hand-written on purpose: a hand-written regime
    sequence is a guess about what the sensor does, and a smoke fixture that
    guesses wrong reports a tool bug that is really a fixture bug (which is
    exactly what the first version of this smoke did -- it omitted the MIXED
    warm-up every fresh sensor emits and then flagged the replay as
    non-deterministic).
    """
    sensor = RegimeSensor(enter_window=2, exit_window=4)
    with open(path, "w") as f:
        f.write(json.dumps({"kind": "header", "mode": "observe"}) + "\n")
        for i, spec in enumerate(inputs):
            sample = RegimeSample(
                round_index=(i + 1) * 8,
                prefill_rounds=spec["prefill_rounds"],
                decode_rounds=spec["decode_rounds"],
                held_tokens=spec["held"],
                capacity_tokens=capacity,
            )
            regime = sensor.observe(sample)
            f.write(
                json.dumps(
                    {
                        "kind": "verdict",
                        "round": sample.round_index,
                        "prefill_share": sample.prefill_share,
                        "decode_share": sample.decode_share,
                        "occupancy": sample.occupancy,
                        "held_tokens": sample.held_tokens,
                        "capacity_tokens": sample.capacity_tokens,
                        "queued_reqs": 0,
                        "queued_prompt_tokens": 0,
                        "regime": regime,
                        "sample_spread_pct": None,
                    }
                )
                + "\n"
            )
        f.write(json.dumps({"kind": "summary", "desyncs": 0}) + "\n")


def smoke() -> int:
    """Card-less: three cases with known answers, through the same code the
    live trace will go through."""
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        # 90 000 held tokens, prefill-heavy: 20 % of the booted pool and 93 %
        # of the #354 prefill arm's. The phase-2 fixture, as a trace.
        inputs = [
            {"prefill_rounds": 48, "decode_rounds": 16, "held": 90_000}
            for _ in range(30)
        ]
        path = os.path.join(tmp, "t.jsonl")
        _record_trace(path, inputs, 453_632)
        verdicts, summary = load(path)
        assert summary is not None

        booted, target = _stages(453_632, 380_000)
        v = judge(verdicts, booted, target)
        print(
            f"[harmless    ] passed={v['passed']} recorded={v['recorded']} "
            f"trap_approached={v['trap_approached']}"
        )
        ok &= v["passed"] and not v["problems"]

        booted, target = _stages(453_632, 96_256)
        v = judge(verdicts, booted, target)
        print(
            f"[trap        ] passed={v['passed']} "
            f"interlock_load_bearing={v['interlock_was_load_bearing']}"
        )
        print(f"               unguarded={v['unguarded_closed_loop']}")
        # The guarded arm stays clean AND the unguarded one does not: that
        # pair is what says the interlock earned the result rather than the
        # workload never testing it.
        ok &= v["passed"]
        ok &= v["interlock_was_load_bearing"]

        stripped = [{k: x[k] for k in x if k != "held_tokens"} for x in verdicts]
        v = judge(stripped, booted, target)
        print(f"[unreplayable] passed={v['passed']} :: {v['problems'][0][:66]}")
        ok &= not v["passed"]
    print("\nSMOKE OK" if ok else "\nSMOKE FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", help="an observe-mode verdict trace (JSONL)")
    ap.add_argument("--booted-pool", type=int, default=453_632)
    ap.add_argument(
        "--target-pool",
        type=int,
        default=96_256,
        help=(
            "max_total_num_tokens of the stage a controller would have "
            "selected. Use the real one from the boot stage table; the "
            "default is the #354 FP8 prefill arm, the tightest case."
        ),
    )
    ap.add_argument("--evidence", help="evidence JSON to create/merge into")
    ap.add_argument("--note", default="")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)

    if args.smoke:
        return smoke()
    if not args.trace:
        ap.error("--trace is required, or use --smoke")

    verdicts, summary = load(args.trace)
    if not verdicts:
        print("no verdicts in the trace", file=sys.stderr)
        return 2
    if summary is None:
        # Not a refusal any more: the fork's launcher SIGKILLs its scheduler
        # children on shutdown, so a normal run never writes one. Completeness
        # is proved from the verdicts by readout.completeness(), and this tool
        # defers to that single definition rather than keeping a second one.
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from readout import completeness  # noqa: PLC0415 -- sibling script

        comp = completeness({"verdicts": verdicts})
        if not comp["complete"]:
            print(
                f"the trace is neither summarised nor provably complete: "
                f"{comp['why']}",
                file=sys.stderr,
            )
            return 2
        print(f"note: no summary line; completeness proved instead -- {comp['why']}")
    booted, target = _stages(args.booted_pool, args.target_pool)
    verdict = judge(verdicts, booted, target)
    print(json.dumps(verdict, indent=2))
    entry = evidence_entry(verdict, traces=[args.trace], note=args.note)
    print("\nevidence entry:")
    print(json.dumps(entry, indent=2))
    if not verdict["passed"]:
        print("\nGATE 2 NOT PASSED -- not written to the evidence file.")
        return 1
    if args.evidence:
        from readout import merge_into  # noqa: PLC0415 -- sibling script

        merge_into(args.evidence, entry)
        print(f"\nwritten to {args.evidence}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
