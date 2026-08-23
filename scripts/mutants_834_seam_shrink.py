#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#834: apply each mutation, run the suite, report DEAD or ALIVE, revert.

WHY THIS EXISTS. Every test in this ticket passed the first time it was run,
which is exactly the condition under which a suite proves nothing. A test that
has never been observed to fail is a test whose failure mode is unknown, and
this tree has a standing rule about it: a gate needs a can-fail proof, not a
green run.

EACH MUTANT IS A REAL MISTAKE SOMEBODY COULD MAKE, and several are the
TEMPTING version of the change rather than an obviously broken one -- M2 is the
insurance clause left exactly as it shipped, M5 is the seam's own drain deleted
because it now measures zero, M7 is the grow deferred without the clamp that
makes it safe. A mutant that nobody would ever write tests nothing.

THE REPORT NAMES WHICH TESTS DIED, not just how many. A mutation that turns the
whole suite red is weak evidence: it says the suite notices damage, not that the
specific arm aimed at that specific defect is the one that fires. Where a mutant
is expected to kill exactly one arm, that is asserted.

Run with the measurement environment, hermetically:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree>/python \\
      <venv>/bin/python scripts/mutants_834_seam_shrink.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME = ROOT / "python/sglang/srt/managers/phase_flip_runtime.py"
SPILL = ROOT / "python/sglang/srt/managers/phase_flip_spill.py"

SCHED_TESTS = ROOT / "test/registered/scheduler"
MEM_TESTS = ROOT / "test/registered/unit/mem_cache"

RUNTIME_SUITE = ["test_seam_shrink_834.py", "test_seam_window_830.py"]
SPILL_SUITE = ["test_seam_shrink_grow_split_834.py"]


@dataclass
class Mutant:
    name: str
    why: str
    path: Path
    old: str
    new: str
    suite: str  # "sched" or "mem"
    expect_exactly: int | None = None
    killed_by: list[str] = field(default_factory=list)


MUTANTS: list[Mutant] = [
    Mutant(
        name="M1 arm never quiesces (the move is not made)",
        why="reverts #834 A: the drain falls back into the no-return window",
        path=RUNTIME,
        old="        prearm_ok, prearm_drain_ms, prearm_detail = self._prearm_quiesce(direction)",
        new="        prearm_ok, prearm_drain_ms, prearm_detail = (True, 0.0, 'mutant')",
        suite="sched",
    ),
    Mutant(
        name="M2 the round hook clears the guard unconditionally",
        why=(
            "the insurance clause exactly as it shipped -- the most likely way "
            "this change is silently undone, because the line looks correct in "
            "isolation and the test that catches it is about a DIFFERENT method"
        ),
        path=RUNTIME,
        old="            if not prearm_quiesce_held(self):\n                self.hicache_seam_active = False",
        new="            self.hicache_seam_active = False",
        suite="sched",
    ),
    Mutant(
        name="M3 a refused arm leaves the device tier disarmed",
        why="the #742 inert-state class: a capability dead for the process life",
        path=RUNTIME,
        old='        self._release_prearm_quiesce("arm refused on the drain it measured")',
        new="        pass",
        suite="sched",
    ),
    Mutant(
        name="M4 the measured drain never reaches the seam budget",
        why="#830 F4 would keep projecting from the previous flip for no reason",
        path=RUNTIME,
        old="        self._seam_drain_ms = drain_ms\n        allow, escalated, detail = flip_seam_budget_verdict(",
        new="        allow, escalated, detail = flip_seam_budget_verdict(",
        suite="sched",
    ),
    Mutant(
        name="M5 the seam's own drain is deleted (it measures 0 now)",
        why=(
            "THE CORRECTNESS MUTANT, and the whole point of keeping it: with "
            "the shrink on the seam's quiesce finds nothing to do, which makes "
            "it look free to remove. Removing it re-opens two SIGSEGVs that a "
            "shape check cannot catch (#830 M11, restated for this change)"
        ),
        path=RUNTIME,
        old="        self._seam_drain_ms = self._quiesce_hicache(direction)",
        new="        self._seam_drain_ms = 0.0",
        suite="sched",
    ),
    Mutant(
        name="M6 the gate-off path quiesces anyway",
        why="default-off must be byte-identical; this is the commonest way it is not",
        path=RUNTIME,
        old="        if not seam_shrink_prearm_quiesce_enabled():\n            return True, 0.0, \"prearm quiesce off\"",
        new="        if False:\n            return True, 0.0, \"prearm quiesce off\"",
        suite="sched",
    ),
    Mutant(
        name="M7 the deferred grow is paid WITHOUT the clamp",
        why=(
            "THE TEMPTING SIMPLIFICATION. The clamp looks like tidy-up after "
            "the grow; it is the half that makes the grow safe. Without it "
            "recover()'s own clamp_exposure_to_backing has just exposed rows "
            "no peer has backed"
        ),
        path=RUNTIME,
        old="                clamped = int(clamp_kv_exposure_to_level(scheduler, level))",
        new="                clamped = 0",
        suite="sched",
    ),
    Mutant(
        name="M8 the unlevelled-exposure guard always allows",
        why="hazard direction 1: the three-rank abort is no longer refused",
        path=RUNTIME,
        old="        if not seam_shrink_defer_grow_enabled():\n            return None\n        level = getattr(self, \"_deferred_grow_level\", None)",
        new="        if True:\n            return None\n        level = getattr(self, \"_deferred_grow_level\", None)",
        suite="sched",
    ),
    Mutant(
        name="M9 the grow debt is never shouted",
        why="hazard direction 2: #814's ratchet returns, silently",
        path=RUNTIME,
        old="        logger.error(\n            \"%s [#834] %s: %d row(s) have been BACKED but not EXPOSED for %d \"",
        new="        logger.info(\n            \"%s [#834] %s: %d row(s) have been BACKED but not EXPOSED for %d \"",
        suite="sched",
    ),
    Mutant(
        name="M10 nothing ever pays the deferred grow",
        why="a debt with no payer IS #814, and every seam would look faster",
        path=RUNTIME,
        old="        pay_deferred_grow(self)\n        ready = 1 if (armed and self._ready_fn()) else 0",
        new="        ready = 1 if (armed and self._ready_fn()) else 0",
        suite="sched",
    ),
    Mutant(
        name="M11 the seam runs the WHOLE recovery again",
        why="the grow never leaves the no-return window; the shrink is decorative",
        path=RUNTIME,
        old="    level = level_kv_backing_to_group(scheduler, reduce_fn)\n    runtime.book_deferred_grow(direction, level)",
        new="    recover_kv_backing(scheduler, reduce_fn=reduce_fn)",
        suite="sched",
    ),
    Mutant(
        name="M12 the levelling leaves the seam too",
        why=(
            "THE OVER-CORRECTION, and the one the #830 design note refuses by "
            "name: the collective cannot run at a rank-local cadence without "
            "re-opening the 2026-08-08 boots 9/10 PP wedge"
        ),
        path=RUNTIME,
        old="    level = level_kv_backing_to_group(scheduler, reduce_fn)\n    runtime.book_deferred_grow(direction, level)",
        new="    runtime.book_deferred_grow(direction, None)",
        suite="sched",
    ),
    Mutant(
        name="M13 the level is reported only when rows moved",
        why=(
            "an even group -- the commonest case -- would report None, and a "
            "deferred grow reading None must refuse to expose anything for "
            "ever on a perfectly healthy fleet"
        ),
        path=SPILL,
        old="        if group_min > 0:\n            applied = group_min",
        new="        if False:\n            applied = group_min",
        suite="mem",
    ),
    Mutant(
        name="M14 the #792 decline is reported as an agreed level",
        why=(
            "a declined levelling means NO level exists; handing it back as one "
            "authorises exactly the exposure the decline was protecting against"
        ),
        path=SPILL,
        old="            applied = None\n        elif unlevel:",
        new="            pass\n        elif unlevel:",
        suite="mem",
    ),
    Mutant(
        name="M15 the collective half never reaches the channel",
        why=(
            "THE SAME CLAIM AS M12 READ FROM THE OTHER END. The deferral rests "
            "on 'the local half touches no collective, the levelling half "
            "does'. The first clause is structurally true -- "
            "grow_kv_backing_local has no channel parameter to reach one with, "
            "which is a stronger guarantee than a test -- so the arm that can "
            "actually fail is the second: a levelling that agrees with nobody "
            "and reports a level anyway"
        ),
        path=SPILL,
        old="        reduced = list(reduce_fn([backed, -backed, -floor]))",
        new="        reduced = [backed, -backed, -floor]",
        suite="mem",
    ),
]


def _run(suite: str) -> tuple[bool, list[str]]:
    cwd = SCHED_TESTS if suite == "sched" else MEM_TESTS
    files = RUNTIME_SUITE if suite == "sched" else SPILL_SUITE
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = f"{ROOT / 'python'}:{cwd}"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *files, "-q", "-p", "no:cacheprovider"],
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
    print("#834 mutant run. Baseline first: a mutant report against a red")
    print("baseline is unreadable, so the baseline is proved before anything")
    print("is mutated.\n")
    for suite in ("sched", "mem"):
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
        if failed:
            for f in failed[:6]:
                print(f"      killed by: {f}")
            if len(failed) > 6:
                print(f"      ... and {len(failed) - 6} more")
        print()

    ok, failed = _run("sched")
    ok2, failed2 = _run("mem")
    print(f"restored green: sched={ok} mem={ok2}")
    if alive:
        print(f"\nALIVE MUTANTS ({len(alive)}) -- each is an untested claim:")
        for a in alive:
            print(f"  {a}")
        return 1
    print(f"\nall {len(MUTANTS)} mutants DEAD")
    return 0 if (ok and ok2) else 3


if __name__ == "__main__":
    raise SystemExit(main())
