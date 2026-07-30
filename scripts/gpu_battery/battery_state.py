#!/usr/bin/env python3
"""Run state, resume and step selection for the GPU battery.

A battery run is not a single sitting. Cards get taken away, a driver gets
updated mid-queue, a boot fails and the bugfixer needs three hours. What was
already green must not have to be paid for twice -- and when there is genuine
doubt about a green result, it must be re-runnable on purpose rather than by
deleting files.

So: every verdict is recorded in <run dir>/state.json, and every later
invocation reads it.

  * RESUME IS THE DEFAULT. A step whose recorded verdict is PASS is skipped
    with a SKIP line naming when it passed.
  * --force <ids> re-runs green steps anyway. That is the "justified doubt"
    path, and it is explicit, per step, and logged as a new attempt.
  * --only / --from / --to / --skip select what runs at all.
  * Dependencies are ARTIFACT dependencies, not ordering. s08 needs the s01
    and s06 files; it does not need the boots. A resume can therefore run s08
    alone months later, as long as the run dir still holds what it consumes.

Subcommands:
  init    create/refresh state.json for a run dir
  plan    print the ordered list of steps with RUN / SKIP / BLOCKED and why
  gate    one step: may it run now? (used by run_step.sh)
  record  write a step's verdict
  status  print the table of what is where
  field   print one field of the step table (used by run_step.sh)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from typing import List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from battery_steps import (  # noqa: E402
    STEP_ORDER,
    STEPS_BY_ID,
    VERDICT_PASS,
    VERDICTS,
    resolve_ids,
    total_expected_min,
)

STATE_KIND = "gpu_battery_state"
STATE_SCHEMA_VERSION = 1


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def state_path(run_dir: str) -> str:
    return os.path.join(run_dir, "state.json")


def load_state(run_dir: str) -> dict:
    path = state_path(run_dir)
    if not os.path.exists(path):
        return new_state(run_dir)
    with open(path) as f:
        payload = json.load(f)
    if payload.get("kind") != STATE_KIND:
        raise SystemExit(
            f"{path}: kind {payload.get('kind')!r}, expected {STATE_KIND!r}"
        )
    payload.setdefault("steps", {})
    return payload


def new_state(run_dir: str) -> dict:
    return {
        "kind": STATE_KIND,
        "schema_version": STATE_SCHEMA_VERSION,
        "run_dir": os.path.abspath(run_dir),
        "run_id": os.path.basename(os.path.abspath(run_dir)),
        "created": _now(),
        "commit": _git_commit(),
        "steps": {},
    }


def save_state(run_dir: str, payload: dict) -> None:
    os.makedirs(run_dir, exist_ok=True)
    tmp = state_path(run_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    os.replace(tmp, state_path(run_dir))


def _git_commit() -> Optional[str]:
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(
            ["git", "-C", here, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def verdict_of(state: dict, step_id: str) -> Optional[str]:
    entry = state.get("steps", {}).get(step_id)
    return entry.get("verdict") if entry else None


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def select_ids(
    only: Optional[str] = None,
    from_id: Optional[str] = None,
    to_id: Optional[str] = None,
    skip: Optional[str] = None,
) -> List[str]:
    ids = list(STEP_ORDER)
    if only:
        wanted = set(resolve_ids(only))
        ids = [i for i in ids if i in wanted]
    if from_id:
        start = resolve_ids(from_id)[0]
        ids = ids[ids.index(start) :] if start in ids else []
    if to_id:
        end = resolve_ids(to_id)[0]
        if end in ids:
            ids = ids[: ids.index(end) + 1]
    if skip:
        dropped = set(resolve_ids(skip))
        ids = [i for i in ids if i not in dropped]
    return ids


def plan(
    state: dict,
    selected: List[str],
    forced: Optional[List[str]] = None,
    rerun_all: bool = False,
) -> List[Tuple[str, str, str]]:
    """Return [(step_id, action, reason)] with action in RUN|SKIP|BLOCKED.

    BLOCKED is deliberately distinct from SKIP: a blocked step is one the
    executor MUST NOT quietly pass over, because its inputs are not there.
    """
    forced_set = set(forced or [])
    rows: List[Tuple[str, str, str]] = []
    will_run = set()
    for step_id in selected:
        step = STEPS_BY_ID[step_id]
        recorded = verdict_of(state, step_id)

        # Skip BEFORE the dependency gate: an already-green step has its
        # artifacts on disk and is not going to run, so what its inputs look
        # like today is irrelevant. Checking deps first would report a finished
        # step as BLOCKED and stall a resume on a step that is done.
        if recorded == VERDICT_PASS and not rerun_all and step_id not in forced_set:
            when = state["steps"][step_id].get("finished", "?")
            rows.append((step_id, "SKIP", f"already PASS on {when}"))
            continue

        unmet = [
            d
            for d in step.deps
            if verdict_of(state, d) != VERDICT_PASS and d not in will_run
        ]
        if unmet:
            rows.append(
                (step_id, "BLOCKED", f"dependency not PASS: {','.join(unmet)}")
            )
            continue

        if recorded == VERDICT_PASS:
            # "trotz PASS" is asserted on by test_gpu_battery_checks.py --
            # the marker stays German, the rest of the line does not.
            reason = "forced re-run trotz PASS"
        elif recorded:
            reason = f"last verdict {recorded}"
        else:
            reason = "not run yet"
        rows.append((step_id, "RUN", reason))
        will_run.add(step_id)
    return rows


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------


def cmd_init(args) -> int:
    os.makedirs(args.run_dir, exist_ok=True)
    if os.path.exists(state_path(args.run_dir)):
        state = load_state(args.run_dir)
        print(
            f"state exists: {state_path(args.run_dir)} (created {state.get('created')})"
        )
    else:
        state = new_state(args.run_dir)
        save_state(args.run_dir, state)
        print(f"state created: {state_path(args.run_dir)}")
    return 0


def cmd_plan(args) -> int:
    state = load_state(args.run_dir)
    selected = select_ids(args.only, getattr(args, "from"), args.to, args.skip)
    forced = resolve_ids(args.force) if args.force else []
    rows = plan(state, selected, forced, args.rerun_all)
    to_run = [sid for sid, action, _ in rows if action == "RUN"]

    if args.format == "ids":
        for sid in to_run:
            print(sid)
        return 0

    for sid, action, reason in rows:
        step = STEPS_BY_ID[sid]
        print(
            f"{action:<8} {sid:<26} {step.model:<7} ~{step.expected_min:>3} min  {reason}"
        )
    blocked = [sid for sid, action, _ in rows if action == "BLOCKED"]
    print(
        f"\n{len(to_run)} step(s) to run, estimated "
        f"{total_expected_min(to_run)} min; {len(blocked)} blocked"
    )
    return 0


def cmd_gate(args) -> int:
    """One step. Prints exactly one line and exits:
    0 = run it, 3 = skip it (already green), 2 = blocked / must stop."""
    state = load_state(args.run_dir)
    step_id = resolve_ids(args.step)[0]
    rows = plan(state, [step_id], [step_id] if args.force else [], args.rerun_all)
    _, action, reason = rows[0]
    if action == "RUN":
        print(f"RUN {step_id}: {reason}")
        return 0
    if action == "SKIP":
        print(f"BATTERY-SKIP {step_id}: {reason}")
        return 3
    print(f"BATTERY-STOP {step_id}: {reason}")
    return 2


def cmd_record(args) -> int:
    if args.verdict not in VERDICTS:
        raise SystemExit(f"unknown verdict {args.verdict!r}, allowed: {VERDICTS}")
    state = load_state(args.run_dir)
    step_id = resolve_ids(args.step)[0]
    prev = state["steps"].get(step_id, {})
    entry = {
        "verdict": args.verdict,
        "line": args.line or "",
        "reason": args.reason or "",
        "started": args.started or prev.get("started"),
        "finished": _now(),
        "duration_s": args.duration_s,
        "attempts": int(prev.get("attempts", 0)) + 1,
        "step_dir": os.path.join(state["run_dir"], step_id),
        "history": (prev.get("history") or [])
        + (
            [{"verdict": prev["verdict"], "finished": prev.get("finished")}]
            if prev.get("verdict")
            else []
        ),
    }
    state["steps"][step_id] = entry
    state["updated"] = entry["finished"]
    save_state(args.run_dir, state)
    print(f"{step_id}: {args.verdict} (attempt {entry['attempts']})")
    return 0


def cmd_status(args) -> int:
    state = load_state(args.run_dir)
    print(f"run:    {state.get('run_id')}  ({state.get('run_dir')})")
    print(f"commit: {state.get('commit')}")
    print(f"{'step':<26} {'verdict':<8} {'attempts':>8}  {'finished':<20} reason")
    done_min = 0
    for step_id in STEP_ORDER:
        entry = state.get("steps", {}).get(step_id, {})
        verdict = entry.get("verdict", "-")
        if verdict == VERDICT_PASS:
            done_min += STEPS_BY_ID[step_id].expected_min
        print(
            f"{step_id:<26} {verdict:<8} {entry.get('attempts', 0):>8}  "
            f"{str(entry.get('finished', '-')):<20} {entry.get('reason', '')[:60]}"
        )
    total = total_expected_min()
    print(f"\ngreen: {done_min} of {total} min of step time")
    open_ids = [i for i in STEP_ORDER if verdict_of(state, i) != VERDICT_PASS]
    if open_ids:
        print(f"open: {', '.join(open_ids)} (~{total_expected_min(open_ids)} min)")
    else:
        print("open: nothing -- the battery is done")
    return 0


def cmd_field(args) -> int:
    step = STEPS_BY_ID[resolve_ids(args.step)[0]]
    value = getattr(step, args.field)
    if isinstance(value, bool):
        print("1" if value else "0")
    elif isinstance(value, (list, tuple)):
        print(",".join(value))
    else:
        print(value)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--run-dir", default=os.environ.get("BATTERY_RUN"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(func=cmd_init)

    p = sub.add_parser("plan")
    p.add_argument("--only")
    p.add_argument("--from", dest="from")
    p.add_argument("--to")
    p.add_argument("--skip")
    p.add_argument("--force", help="steps to run again even though they are PASS")
    p.add_argument("--rerun-all", action="store_true", help="run everything, ignore PASS")
    p.add_argument("--format", choices=("table", "ids"), default="table")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("gate")
    p.add_argument("step")
    p.add_argument("--force", action="store_true")
    p.add_argument("--rerun-all", action="store_true")
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("record")
    p.add_argument("step")
    p.add_argument("--verdict", required=True)
    p.add_argument("--line", default="")
    p.add_argument("--reason", default="")
    p.add_argument("--started", default=None)
    p.add_argument("--duration-s", type=int, default=None, dest="duration_s")
    p.set_defaults(func=cmd_record)

    sub.add_parser("status").set_defaults(func=cmd_status)

    p = sub.add_parser("field")
    p.add_argument("step")
    p.add_argument("field")
    p.set_defaults(func=cmd_field)

    args = ap.parse_args()
    if not args.run_dir and args.cmd != "field":
        raise SystemExit("--run-dir or BATTERY_RUN required")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
