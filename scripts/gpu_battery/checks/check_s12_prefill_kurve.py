#!/usr/bin/env python3
"""s12 check -- is the prefill curve a measurement, not just a pile of numbers?

Verified:

  * every PLANNED session count has a number in BOTH arms. Half a pair is not a
    comparison, and an arm that broke off mid-run leaves exactly that,
  * the arms really alternated. A,B,A,B is the difference between a comparison
    and two afternoons; blockwise measurement is the one methodological error
    that cannot be repaired afterwards,
  * each point's boot ran the arm it claims. For bar1: every communicator group
    reports ACHIEVED=bar1 -- the requested name says bar1 even when the group
    fell back to gloo, and one mixed group makes the point a mixed point. For
    the baseline: NO HTCCL group at all,
  * a decode point at bs=1 and one at bs=16 per arm,
  * an output sample per arm is persisted. A fast garbage run looks good in a
    throughput table (measurement rule 5),
  * no boot died on an OOM / NCCL error / traceback. Eight boots that each
    broke off in a prefill OOM still hand in throughput numbers, and without
    this gate the table looks exactly like a healthy one,
  * the baseline reproduces the known host-path numbers within the tolerance.
    It does not: STOP -- this environment is not the one those numbers came
    from, and comparing against them would be comparing two rigs.

Deliberately NOT judged -- and this is the point of the whole step: whether the
bar1 curve is flat or rising, and by how much it beats the baseline. Flat is a
finding, rising is a finding. A check that graded it would be deciding the
question the measurement exists to answer.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_common import (  # noqa: E402
    CheckFail,
    CheckStop,
    is_number,
    load_json,
    require_envelope,
    run_check,
)

STEP = "s12_prefill_kurve"
KIND = "bar1_prefill_kurve"
# Arm names as the producer writes them into the artifact -- "grundlinie" is a
# value in prefill_kurve.json, not prose, so it stays as it is.
ARMS = ("bar1", "grundlinie")
DECODE_BATCHES = (1, 16)
REQUIRED_GROUP_PREFIXES = ("tp", "dcp")
MIN_SAMPLE_CHARS = 50


def _rate(payload: dict, arm: str, sessions: int):
    return (payload.get("kurven") or {}).get(arm, {}).get(str(sessions))


def _check_fatal(payload: dict) -> None:
    """The fatal harvest per boot. A missing field is a FAIL, not a pass:
    "nobody looked" and "nothing found" must not produce the same verdict."""
    if "fatal" not in payload:
        raise CheckFail(
            "prefill_kurve.json carries no fatal field -- the boots' fatal harvest "
            "(logs/*.fatal.txt) was never evaluated at all"
        )
    unharvested = payload.get("fatal_ungeprueft") or []
    if unharvested:
        first = unharvested[0]
        raise CheckFail(
            f"{len(unharvested)} boot(s) ohne Fatal-Ernte, first: "
            f"{first.get('arm')}/{first.get('sessions')} sessions -- without that "
            "file, 'no fatal' is not proven, only unknown"
        )
    fatal = payload.get("fatal") or []
    if fatal:
        first = fatal[0]
        raise CheckFail(
            f"{len(fatal)} boot(s) with a fatal, first {first.get('arm')}/"
            f"{first.get('sessions')} sessions -- {first.get('zeile')}"
        )


def check(step_dir: str) -> None:
    path = os.path.join(step_dir, "prefill_kurve.json")
    if not os.path.exists(path):
        raise CheckStop(f"prefill_kurve.json missing ({path}) -- the step never ran")
    payload = load_json(path, "prefill_kurve.json")
    require_envelope(payload, KIND, "prefill_kurve.json", 2)

    if not payload.get("host_erreichbar"):
        raise CheckStop("host unreachable -- nothing was measured")
    if not payload.get("integration_vorhanden"):
        raise CheckStop(
            "the BAR1 integration is not in the worktree under test -- set "
            "BAR1_HOST_WT"
        )
    if payload.get("blockiert"):
        raise CheckStop(f"step blocked, blockiert={payload['blockiert']}")

    plan = payload.get("sessions_geplant") or []
    if not plan:
        raise CheckStop("no session plan in the result -- nothing to check")

    # --- completeness --------------------------------------------------------
    aborted = payload.get("abbruch")
    for sessions in plan:
        for arm in ARMS:
            rate = _rate(payload, arm, sessions)
            if not is_number(rate) or rate <= 0:
                raise CheckFail(
                    f"no throughput for arm {arm!r} at {sessions} session(s) "
                    f"(value {rate!r})"
                    + (f"; run aborted: {aborted}" if aborted else "")
                )

    # --- interleaved, not blockwise -----------------------------------------
    order = payload.get("reihenfolge") or []
    if len(order) < 2 * len(plan):
        raise CheckFail(
            f"only {len(order)} measured points for {2 * len(plan)} expected "
            "(two arms per session count)"
        )
    for a, b in zip(order, order[1:]):
        if a.get("arm") == b.get("arm"):
            raise CheckFail(
                f"two identical arms back to back ({a.get('arm')} at "
                f"{a.get('sessions')} and {b.get('sessions')} sessions) -- measured "
                "blockweise, so the comparison is not interleaved"
            )

    _check_fatal(payload)

    # --- did each boot run the arm it claims? -------------------------------
    for entry in order:
        arm = entry.get("arm")
        if not entry.get("beleg_vorhanden"):
            raise CheckFail(
                f"no transport Beleg for {arm}/{entry.get('sessions')} -- without "
                "it the measured value's arm is unsupported"
            )
        groups = entry.get("gruppen") or []
        if arm == "grundlinie":
            if groups:
                raise CheckFail(
                    f"Grundlinie boot at {entry.get('sessions')} sessions reports "
                    f"HTCCL groups {[g.get('group') for g in groups]} -- the "
                    "baseline must not see a single SGLANG_HTCCL* variable"
                )
            continue
        if not groups:
            raise CheckFail(
                f"bar1 boot at {entry.get('sessions')} sessions reports not a "
                "single ERREICHT line -- HTCCL was not on"
            )
        for prefix in REQUIRED_GROUP_PREFIXES:
            hits = [g for g in groups if str(g.get("group", "")).startswith(prefix)]
            if not hits:
                raise CheckFail(
                    f"bar1 boot at {entry.get('sessions')} sessions without group "
                    f"{prefix!r} (reported: {[g.get('group') for g in groups]})"
                )
            foreign = [g for g in hits if g.get("achieved") != "bar1"]
            if foreign:
                raise CheckFail(
                    f"bar1 boot at {entry.get('sessions')} sessions: group "
                    f"{foreign[0].get('group')!r} runs "
                    f"ACHIEVED={foreign[0].get('achieved')!r} -- mixed point, "
                    "not a bar1 point"
                )

    # --- decode points -------------------------------------------------------
    decode = payload.get("decode") or []
    for arm in ARMS:
        for batch in DECODE_BATCHES:
            hits = [
                d
                for d in decode
                if d.get("arm") == arm
                and d.get("batch") == batch
                and is_number(d.get("decode_tok_s"))
            ]
            if not hits:
                raise CheckFail(f"no decode point for arm {arm!r} at bs={batch}")

    # --- output samples ------------------------------------------------------
    samples = payload.get("output_samples") or {}
    for arm in ARMS:
        sample = samples.get(arm)
        if not sample or len(str(sample)) < MIN_SAMPLE_CHARS:
            raise CheckFail(
                f"no output Sample persisted for arm {arm!r} "
                f"({str(sample)[:40]!r}) -- a fast garbage run looks good in a "
                "throughput table"
            )

    # --- does the baseline reproduce? ---------------------------------------
    tol = payload.get("toleranz_pct")
    if not is_number(tol):
        tol = 5.0
    deviation = payload.get("grundlinie_abweichung_pct") or {}
    known = payload.get("grundlinie_bekannt") or {}
    worst = None
    for sessions in plan:
        key = str(sessions)
        if key not in known:
            continue
        if key not in deviation:
            raise CheckStop(
                f"no deviation against the known baseline value computed at "
                f"{sessions} session(s)"
            )
        if worst is None or abs(deviation[key]) > abs(deviation[worst]):
            worst = key
    if worst is not None and abs(deviation[worst]) > tol:
        raise CheckStop(
            f"Grundlinie reproduziert nicht: at {worst} session(s) "
            f"{_fmt(_rate(payload, 'grundlinie', int(worst)))} instead of "
            f"{known[worst]} tok/s ({deviation[worst]:+.1f} %, allowed "
            f"+-{tol} %) -- this environment is not the one the known numbers came "
            f"from ({payload.get('grundlinie_quelle')})"
        )


def _fmt(value) -> str:
    return f"{value:.1f}" if is_number(value) else str(value)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
