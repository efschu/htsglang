#!/usr/bin/env python3
"""#829 mutation check: does the suite actually hold the fix in place?

Each mutant is a way the #829 fix could be weakened or over-applied by a
later edit. A mutant that the suite does not kill is a guard nobody is
guarding, so every one of these MUST be reported DEAD.

The danger direction is LOOSENING. M1 and M2 put the tree back the way it
was on boot_window2_0823_1554; M4 is the tempting "simpler" version of
the fix that silently removes #757's drain. M3 and M5 are the opposite
error -- over-correcting until #824 W4b's own case breaks -- and they are
here because a suite that only forbids one direction lets a fix pass by
deleting the feature.

CPU-only, no GPU, no server. Restores the source from an in-memory copy
in a finally block, so an interrupted run cannot leave the tree mutated.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree>/python \\
        <venv>/bin/python scripts/mutants_829_arm_slot_ring.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "python" / "sglang" / "srt" / "managers" / "scheduler_pp_mixin.py"
TEST = (
    ROOT
    / "test"
    / "registered"
    / "unit"
    / "managers"
    / "test_pp_arm_slot_outlives_ring_829.py"
)

# (name, what it models, old, new)
MUTANTS = [
    (
        "M1-drop-the-epoch-term",
        "the falling edge stops distinguishing a commit from an abandon "
        "-- exactly the f9d7637f04 behaviour that killed window 2",
        "        if would_restore and ring_rebuilt:",
        "        if False and would_restore and ring_rebuilt:",
    ),
    (
        "M2-unwire-the-ring-rebuild",
        "the backstop helper still exists but init_pp_loop_state no "
        "longer calls it: a fix written and never run",
        "        pp_flip_forget_ring_scoped_slots(self)",
        "        pass  # mutant: backstop unwired",
    ),
    (
        "M3-delete-the-restore",
        "over-correction: the cheapest way to pass the two root arms is "
        "to remove #824 W4b altogether, reopening boot_827",
        "        elif would_restore:",
        "        elif False and would_restore:",
    ),
    (
        "M4-also-clear-the-pass-counter",
        "the tempting simpler fix: clearing _pp_flip_armed_passes on the "
        "rebuild suppresses the falling edge, taking #757's leftover "
        "drain off the commit path without saying so",
        "    holder._pp_flip_resume_slot = None",
        "    holder._pp_flip_resume_slot = None\n    holder._pp_flip_armed_passes = None",
    ),
    (
        "M5-refuse-on-every-falling-edge",
        "over-broad refusal: treating an abandon as a rebuild too, which "
        "breaks #824 W4b's own case",
        "            and int(arm_epoch) != int(now_epoch)",
        "            and True",
    ),
]


def run_suite() -> bool:
    """True when the suite passes."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", str(TEST), "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def main() -> int:
    original = SRC.read_text()
    results = []
    try:
        if not run_suite():
            print("BASELINE IS RED -- fix the tree before measuring mutants.")
            return 2
        print("baseline: GREEN\n")

        for name, why, old, new in MUTANTS:
            if original.count(old) != 1:
                print(f"{name}: ANCHOR NOT UNIQUE ({original.count(old)} hits) -- "
                      "the mutant could not be applied, which is itself a failure")
                results.append((name, False, why))
                continue
            SRC.write_text(original.replace(old, new, 1))
            killed = not run_suite()
            results.append((name, killed, why))
            print(f"{name}: {'DEAD (suite caught it)' if killed else 'SURVIVED'}")
            print(f"    {why}")
            SRC.write_text(original)
    finally:
        SRC.write_text(original)

    survivors = [n for n, killed, _ in results if not killed]
    print()
    if survivors:
        print(f"{len(survivors)} MUTANT(S) SURVIVED: {', '.join(survivors)}")
        return 1
    print(f"all {len(results)} mutants dead; source restored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
