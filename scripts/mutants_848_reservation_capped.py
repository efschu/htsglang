#!/usr/bin/env python3
"""#848 mutants: wrong ways to consult the reservation.

M1 IS WINDOW 7's SHIPPED BEHAVIOUR restored deliberately -- it ran on metal and
produced a refusal naming the driver instead of the reservation. M2 is the
off-by-one that would be WORSE than the defect: refusing the last legal row
turns a working grow into a permanent named refusal.
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RELIEF = ROOT / "python/sglang/srt/managers/kv_backing_relief.py"
SUITE = "test/registered/unit/managers/test_reservation_capped_848.py"

MUTANTS = [
    ("M1 WINDOW 7's BUG: the reservation is never consulted",
     "        ceiling = self._reserved_rows()\n        if ceiling is not None and target > ceiling:",
     "        ceiling = self._reserved_rows()\n        if False:"),
    ("M2 off-by-one: refuses a target EQUAL to the reservation (worse than the bug)",
     "        if ceiling is not None and target > ceiling:",
     "        if ceiling is not None and target >= ceiling:"),
    ("M3 an unreadable reservation is treated as a ceiling of zero",
     "        if ceiling is not None and target > ceiling:",
     "        if (ceiling or 0) >= 0 and target > (ceiling or 0):"),
    ("M4 the exit is not named (refusal recorded, census silent)",
     "            self._note_floor_need_exit(\n                FLOOR_NEED_RESERVATION_CAPPED,",
     "            _skip = (\n                FLOOR_NEED_RESERVATION_CAPPED,"),
    ("M5 the refusal is silent (census named, operator not told)",
     "            self._record_floor_need_refusal(\n                floor,\n                target,\n                f\"the pool's IMMUTABLE VA reservation is",
     "            _skip2 = (\n                floor,\n                target,\n                f\"the pool's IMMUTABLE VA reservation is"),
]

def run():
    r = subprocess.run([sys.executable, "-m", "pytest", SUITE, "-q", "-x"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode == 0

def main():
    if not run():
        print("REFUSING: suite not green before mutating.")
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
            killed = not run()
        finally:
            RELIEF.write_text(src)
        print(f"  {'DEAD ' if killed else 'ALIVE'} {name}")
        dead, alive = (dead + 1, alive) if killed else (dead, alive + 1)
    print(f"\n{dead} dead / {dead+alive} total")
    if not run():
        print("WARNING: sources not restored.")
        return 3
    return 0 if alive == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
