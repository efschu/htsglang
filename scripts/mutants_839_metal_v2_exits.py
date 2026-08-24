#!/usr/bin/env python3
"""#839-METAL v2 mutants: each is a wrong implementation this fix could regress into.

M1 IS THE SHIPPED v1 BUG, restored deliberately. It is not hypothetical: it ran
on metal for a whole GPU window (window 6, integ/round6 @ 241e7ac385) and
produced 570 flip arms, 0 completions and total silence. If M1 ever survives
again, the fix has been undone.

Run:  CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree>/python \\
        <venv>/bin/python scripts/mutants_839_metal_v2_exits.py
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RELIEF = ROOT / "python/sglang/srt/managers/kv_backing_relief.py"
SUITE = "test/registered/unit/managers/test_floor_need_exits_839_v2.py"

MUTANTS = [
    (
        "M1 THE v1 BUG: a non-raising commit is taken as success (window 6)",
        "        reached = int(self._current_rows())\n        if reached < target:",
        "        reached = int(self._current_rows())\n        if False:",
    ),
    (
        "M2 the span reverts to the raw row id (off by one)",
        "        self._group_live_need = row + 1",
        "        self._group_live_need = row",
    ),
    (
        "M3 the clamp is counted but not refused (exit named, operator not told)",
        "            self._record_floor_need_refusal(\n                floor,\n                target,\n                f\"the pool CLAMPED:",
        "            _unused = (\n                floor,\n                target,\n                f\"the pool CLAMPED:",
    ),
    (
        "M4 NO-GROUP-VERDICT returns a bare 0 again, unnamed",
        "            self._note_floor_need_exit(\n                FLOOR_NEED_NO_GROUP_VERDICT, floor=floor, need=need\n            )\n            return 0, FLOOR_NEED_NO_GROUP_VERDICT",
        "            return 0, FLOOR_NEED_NO_GROUP_VERDICT",
    ),
    (
        "M5 a rank that is not the floor grows too (the min moves by zero)",
        "        local = int(self._current_rows())\n        if local > floor:",
        "        local = int(self._current_rows())\n        if False:",
    ),
    (
        "M6 the exit census stops counting (asserting at a desk becomes impossible)",
        "        self._floor_need_exit_counts[reason] = (\n            self._floor_need_exit_counts.get(reason, 0) + 1\n        )",
        "        pass",
    ),
]


def run_suite() -> bool:
    r = subprocess.run(
        [sys.executable, "-m", "pytest", SUITE, "-q", "-x"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def main() -> int:
    if not run_suite():
        print("REFUSING: the suite is not green before mutating.")
        return 2
    dead = alive = 0
    for name, old, new in MUTANTS:
        src = RELIEF.read_text()
        if old not in src:
            print(f"  SKIP(anchor missing)  {name}")
            alive += 1
            continue
        RELIEF.write_text(src.replace(old, new, 1))
        try:
            killed = not run_suite()
        finally:
            RELIEF.write_text(src)
        if killed:
            dead += 1
            print(f"  DEAD  {name}")
        else:
            alive += 1
            print(f"  ALIVE {name}   <-- the suite would not have caught this")
    print(f"\n{dead} dead / {dead + alive} total")
    if not run_suite():
        print("WARNING: sources were not restored cleanly.")
        return 3
    return 0 if alive == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
