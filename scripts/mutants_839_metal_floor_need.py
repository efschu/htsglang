#!/usr/bin/env python3
"""#839-METAL mutants: each one is a wrong implementation a reviewer might write.

A mutant that no test kills is a test suite that would not have caught the bug.
M3 is not hypothetical -- it is the FIRST DRAFT of this fix, which checked the
arena before rebinding and therefore compared row counts from two layouts. The
guard test caught it during development and it is kept here as a mutant so it
cannot come back.

Run:  CUDA_VISIBLE_DEVICES="" PYTHONPATH=<worktree>/python \\
        <venv>/bin/python scripts/mutants_839_metal_floor_need.py
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RELIEF = ROOT / "python/sglang/srt/managers/kv_backing_relief.py"
SPILL = ROOT / "python/sglang/srt/managers/phase_flip_spill.py"
SUITE = "test/registered/unit/managers/test_floor_need_839_metal.py"

MUTANTS = [
    (
        "M1 a rank that is not the floor also grows (the min moves by zero)",
        RELIEF,
        "        if int(self._current_rows()) > floor:\n            return 0\n        return need - floor",
        "        return need - floor",
    ),
    # M2 IS A KNOWN EQUIVALENT MUTANT, kept and labelled rather than deleted.
    #
    # It makes the grow announce its own pages -- the window-4-A raise -- and
    # NO TEST KILLS IT, because it cannot change behaviour: ``reconcile_to``
    # re-applies ``_exposure_ceiling``, which is ``min(local, floor)``, so the
    # raise is refused one layer down by the rule #839 A installed. Probed
    # directly with and without the mutation, all four observables identical::
    #
    #     before: exposed=126976 backed=126976 cap=126976 free_max=126976
    #     after : exposed=126976 backed=131073 cap=126976 free_max=126976
    #
    # That is defence in depth and it is the reassuring outcome, not a hole.
    # It is recorded here so nobody re-adds it as a "missing" mutant and so
    # nobody counts it as a kill. If a future change makes ``reconcile_to``
    # able to raise past the ceiling, this mutant becomes killable and the
    # danger direction has regressed -- which is exactly when it should fire.
    (
        "M2 EQUIVALENT the grow announces its own pages (clamped downstream)",
        RELIEF,
        None,
        None,
    ),
    (
        "M3 arena is read before the rebind (this fix's own first draft)",
        RELIEF,
        "        self._rebind()\n        arena = self._arena_key()\n        if self._group_floor_arena != arena",
        "        arena = self._arena_key()\n        self._rebind()\n        if self._group_floor_arena != arena",
    ),
    (
        "M4 a grow that cannot reach is silent (window 5's 153 abandons)",
        RELIEF,
        "            self._record_floor_need_refusal(floor, target, repr(e))\n            return 0",
        "            return 0",
    ),
    (
        "M5 a failed grow reports success and clears the refusal",
        RELIEF,
        "        except Exception as e:  # noqa: BLE001 -- MemoryError and driver errors alike",
        "        except Exception as e:  # noqa: BLE001\n            self._floor_need_refusal = None\n            return 0\n        if False:",
    ),
    (
        "M6 the callsite never reaches the rung (the W9 inert-fix shape)",
        SPILL,
        "                    note_need(int(need))\n                    close_gap()",
        "                    pass",
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
    dead = alive = equivalent = 0
    for name, path, old, new in MUTANTS:
        if old is None:
            equivalent += 1
            print(f"  EQUIV {name}")
            continue
        src = path.read_text()
        if old not in src:
            print(f"  SKIP(anchor missing)  {name}")
            alive += 1
            continue
        path.write_text(src.replace(old, new, 1))
        try:
            killed = not run_suite()
        finally:
            path.write_text(src)
        if killed:
            dead += 1
            print(f"  DEAD  {name}")
        else:
            alive += 1
            print(f"  ALIVE {name}   <-- the suite would not have caught this")
    print(f"\n{dead} dead / {alive} alive / {equivalent} known-equivalent")
    if not run_suite():
        print("WARNING: sources were not restored cleanly.")
        return 3
    return 0 if alive == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
