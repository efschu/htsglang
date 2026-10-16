#!/usr/bin/env python3
"""Mutants of the READ-ROOM review conditions A1-A3, A6 (read_room.py, pdflip_told_fallback.deadline, the store_told call
site). Each must turn test_pdflip_read_room_review_1015.py red. Run from the worktree root (ONE lock turn):
  PYTEST_PYTHON=<venv python> /spinning/gpu-arb/pytest_gedeckelt.sh scripts/mutants_arenaread_review_1015.py -q -s
Sources are restored byte for byte after every mutant."""
import os
import shutil
import subprocess
import sys

RR = "python/flliper/srt/pdflip/read_room.py"
FB = "python/flliper/srt/managers/pdflip_told_fallback.py"
ST = "python/flliper/srt/managers/pdflip_store_told.py"
TEST = "test/registered/unit/mem_cache/test_pdflip_read_room_review_1015.py"

MUTANTS = [
    ("E1 the Frist ignores the room lifesign", FB, "intake_at + total + room))", "intake_at + total))"),
    ("E2 the lifesign is not capped by the absolute ceiling", FB,
     "               _env_float(ENV_WAIT_ABS_S, WAIT_ABS_S_DEFAULT))\n", "               1e9)\n"),
    ("E3 PP0 hands 0 to the Frist", ST, "            _room_s = _read_room_lifesign(_intake, now)\n",
     "            _room_s = 0.0\n"),
    ("E4 an armed room wait records no lifesign", RR, "    with _LOCK:\n        _ST.waits.append(iv)\n    return iv",
     "    return iv"),
    ("E5 overlapping waits count twice (sum, not union)", RR,
     "        if cur_b is None or a > cur_b:", "        if True:"),
    ("E6 the SHORT class reads the process lifetime", RR,
     "    spilled = sum(r for k, r in list(_ST.serve_log) if k > int(served_before))",
     "    spilled = _ST.spilled_pages"),
    ("E7 a rank spills again for a read it already let go of", RR, "    d = max(0, d - mine)\n", ""),
    ("E8 the read id is per poster, not the read's content", RR, "    if stems:\n        import hashlib",
     "    if False:\n        import hashlib"),
    ("E9 'mine' is checked by slot number, not page identity", RR,
     "        if callable(finder) and all(st for _s, st in pages):", "        if False:"),
    ("E10 two reads of one process share a slot", RR, "        if pid == me and rid == read_id:",
     "        if pid == me:"),
    ("E11 a still-short wait that spilled is called nothing_to_spill", RR,
     '"served_nothing_to_spill" if spilled <= 0 else "served_spilled_short"', '"served_nothing_to_spill"'),
    ("E12 a rebind keeps the old maps", RR, "        for p in [p for p in _ST.maps if p != keep]:",
     "        for p in []:"),
]


def run_tests():
    env = dict(os.environ)
    if os.environ.get("MUTANTS_INSIDE_LOCK"):
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH="python:test/registered/unit/pdflip:test/registered/unit/mem_cache")
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", "no:randomly", TEST, "-x"]
    else:
        cmd = ["/spinning/gpu-arb/pytest_gedeckelt.sh", TEST, "-x", "-q"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return r.returncode, (r.stdout + r.stderr)[-600:]


def test_all_mutants_are_killed():
    os.environ["MUTANTS_INSIDE_LOCK"] = "1"
    assert main(quiet=False) == 0


def main(quiet=True):
    bad = 0
    only = [x for x in os.environ.get("MUTANTS_ONLY", "").split(",") if x]
    for name, path, old, new in MUTANTS:
        if only and name.split()[0] not in only:
            continue
        src = open(path).read()
        if src.count(old) != 1:
            print(f"{name}: PATTERN {'NOT FOUND' if not src.count(old) else 'NOT UNIQUE'} in {path}")
            bad += 1
            continue
        shutil.copy(path, path + ".orig")
        try:
            open(path, "w").write(src.replace(old, new, 1))
            rc, tail = run_tests()
        finally:
            shutil.move(path + ".orig", path)
        killed = rc != 0
        print(f"{name}: {'KILLED' if killed else 'SURVIVED'} (rc={rc})", flush=True)
        if not killed:
            bad += 1
            print(tail)
    print("SURVIVORS:", bad, flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
