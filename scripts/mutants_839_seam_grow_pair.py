#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#839: apply each mutation, run the suites, report DEAD or ALIVE, revert.

WHY THIS EXISTS. Both #839 suites passed on the first run against the fixed
tree, and a test that has never been observed to fail is a test whose failure
mode is unknown. The red-first proof against the base tree covers the two
headline arms; this covers the DANGER DIRECTIONS -- the specific ways the fix
can be silently undone by a later edit that looks correct in isolation.

EACH MUTANT IS A REAL MISTAKE SOMEBODY COULD MAKE. M1 and M2 are the shipped
#833 lines, restored: they are what a reviewer would write if they thought the
arena stamp was ceremony. M5 is the tempting simplification of the publication
("it is called the exposure CLAMP, so it should only clamp"). M7 is #834's
ordering comment followed to the letter after #839 changed what it means.

Run with the measurement environment, hermetically::

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree>/python \\
      /spinning/htsglang-gpu/.venv/bin/python \\
      scripts/mutants_839_seam_grow_pair.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RELIEF = ROOT / "python/sglang/srt/managers/kv_backing_relief.py"
RUNTIME = ROOT / "python/sglang/srt/managers/phase_flip_runtime.py"
SPILL = ROOT / "python/sglang/srt/managers/phase_flip_spill.py"

SCHED_TESTS = ROOT / "test/registered/scheduler"
MGR_TESTS = ROOT / "test/registered/unit/managers"

SCHED_SUITE = ["test_grow_debt_payment_839.py", "test_seam_shrink_834.py"]
MGR_SUITE = [
    "test_exposure_publication_839.py",
    "test_group_exposure_floor_833.py",
    "test_exposure_backing_invariant_816.py",
]


@dataclass
class Mutant:
    name: str
    why: str
    path: Path
    old: str
    new: str
    suite: str  # "sched" or "mgr"
    killed_by: list[str] = field(default_factory=list)


MUTANTS: list[Mutant] = [
    Mutant(
        name="M1 the clamp compares against the raw floor again (#833 as shipped)",
        why=(
            "the exact line window 4 booted. The arena check looks like "
            "ceremony until you notice the floor and the backing are readings "
            "of two different layouts"
        ),
        path=RELIEF,
        old="        ceiling = self._exposure_ceiling(backed)",
        new="        ceiling = group_exposure_ceiling(backed, self._group_backed_floor)",
        suite="mgr",
    ),
    Mutant(
        name="M2 the floor is recorded without its arena",
        why=(
            "drops the stamp, so every floor claims to belong to whichever "
            "arena is active when it is READ -- the comparison is then always "
            "'legal' and never checked"
        ),
        path=RELIEF,
        old="        self._group_floor_arena = self._arena_key()\n\n    def _arena_key(self):",
        new="        self._group_floor_arena = None\n\n    def _arena_key(self):",
        suite="mgr",
    ),
    Mutant(
        name="M3 a stale-arena floor may raise after all",
        why=(
            "keeps the stamp but drops the hold, which is the half that "
            "actually closes the divergence -- the stamp alone only makes the "
            "hazard observable"
        ),
        path=RELIEF,
        old="        held = self._published_exposure.get(arena)\n        if held is not None:\n            ceiling = min(ceiling, int(held))",
        new="        held = self._published_exposure.get(arena)\n        if held is not None:\n            ceiling = max(ceiling, int(held))",
        suite="mgr",
    ),
    Mutant(
        name="M4 nothing is ever recorded as published",
        why=(
            "the never-raise rule with nothing to defend. Every call still "
            "runs; the rule silently degrades to #833's behaviour"
        ),
        path=RELIEF,
        old="        self._published_exposure[self._arena_key()] = int(level)",
        new="        _ = int(level)",
        suite="mgr",
    ),
    Mutant(
        name="M5 the publication only ever lowers",
        why=(
            "the tempting simplification: 'it is a clamp, so it should clamp'. "
            "It also turns the fix for A into the defect from B"
        ),
        path=RELIEF,
        old="        moved = int(self.level_recovery_to(level))",
        new="        moved = int(self.level_recovery_to(min(level, self.exposed_rows())))",
        suite="sched",
    ),
    Mutant(
        name="M6 the rung records the group floor and never publishes it",
        why=(
            "#833 exactly as it shipped: the value is taken and nothing acts "
            "on it in the same round. This is the state window 4 measured"
        ),
        path=SPILL,
        old='            publish = getattr(cap_relief, "publish_group_exposure", None)',
        new='            publish = None if True else getattr(cap_relief, "x", None)',
        suite="sched",
    ),
    Mutant(
        name="M7 the payment clamps back to the booked level",
        why=(
            "#834's ordering comment followed to the letter after #839 changed "
            "what the level means -- the payment takes itself back and re-books "
            "the same debt"
        ),
        path=RUNTIME,
        old="            if published is not None:\n                level = published if level is None else max(int(level), published)",
        new="            if published is not None:\n                level = level if level is not None else published",
        suite="sched",
    ),
    Mutant(
        name="M8 the debt is read from the booking, not from the pool",
        why=(
            "#834's own 'RE-READ, never trust the booking' rule, violated the "
            "way #834 violated it -- a settled debt keeps shouting and the "
            "alarm becomes noise"
        ),
        path=RUNTIME,
        old="            settled = max(int(level), int(exposed))",
        new="            settled = int(level)",
        suite="sched",
    ),
    Mutant(
        name="M9 the raw actuator records a published level",
        why=(
            "NOT hypothetical -- this was written, shipped into the branch for "
            "one run, and turned the window-4 reproduction red again. It looks "
            "like completeness ('every publisher should record'), and it is "
            "the opposite: reconcile_to is reached with RANK-LOCAL targets too, "
            "so recording here lets a rank-local level pose as a group verdict "
            "and license the exact raise the rule refuses"
        ),
        path=RELIEF,
        old="            self._cap.engage(level)\n        # #839 A: AND IT DELIBERATELY DOES NOT RECORD",
        new="            self._cap.engage(level)\n        self._record_published(level)\n        # #839 A: AND IT DELIBERATELY DOES NOT RECORD",
        suite="mgr",
    ),
]


def _run(suite: str) -> tuple[bool, list[str]]:
    cwd = SCHED_TESTS if suite == "sched" else MGR_TESTS
    files = SCHED_SUITE if suite == "sched" else MGR_SUITE
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = f"{ROOT / 'python'}:{cwd}"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *files,
            "-q",
            "-rf",
            "--color=no",
            "-p",
            "no:cacheprovider",
        ],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )
    failed = [
        line.split("::", 1)[1].split(" ")[0].strip()
        for line in proc.stdout.splitlines()
        if line.startswith("FAILED") and "::" in line
    ]
    return proc.returncode == 0, failed


def main() -> int:
    print("#839 mutant run. Baseline first: a mutant report against a red")
    print("baseline is unreadable, so the baseline is proved before anything")
    print("is mutated.\n")
    for suite in ("sched", "mgr"):
        ok, failed = _run(suite)
        if not ok:
            print(f"BASELINE RED for suite {suite}: {failed}")
            return 2
        print(f"  baseline {suite}: green")
    print()

    alive: list[str] = []
    for m in MUTANTS:
        src = m.path.read_text()
        if src.count(m.old) != 1:
            print(f"SKIP {m.name}: anchor not found exactly once in {m.path.name}")
            alive.append(m.name + " (ANCHOR LOST)")
            continue
        m.path.write_text(src.replace(m.old, m.new))
        try:
            ok, failed = _run(m.suite)
        finally:
            m.path.write_text(src)
        m.killed_by = failed
        verdict = "DEAD" if not ok else "ALIVE"
        if ok:
            alive.append(m.name)
        print(f"{verdict:5s} {m.name}")
        print(f"      why: {m.why}")
        for f in failed[:5]:
            print(f"      killed by: {f}")
        if len(failed) > 5:
            print(f"      ... and {len(failed) - 5} more")
        print()

    ok, _ = _run("sched")
    ok2, _ = _run("mgr")
    print(f"restored green: sched={ok} mgr={ok2}")
    if alive:
        print(f"\nALIVE MUTANTS ({len(alive)}) -- each is an untested claim:")
        for a in alive:
            print(f"  {a}")
        return 1
    print(f"\nall {len(MUTANTS)} mutants DEAD")
    return 0 if (ok and ok2) else 3


if __name__ == "__main__":
    raise SystemExit(main())
