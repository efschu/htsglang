#!/usr/bin/env python3
"""Mutants of READ-ROOM 1015 (weg2/read_room.py + the two seams). Each mutant must turn
test_weg2_read_room_1015.py red. Run from the worktree root:
  PYTEST_PYTHON=<venv python> /spinning/gpu-arb/pytest_gedeckelt.sh scripts/mutants_arenaread_1015.py
(the file is ALSO a pytest module: ONE lock turn of the shared pytest lock for all mutants; run directly
with python3 it uses the wrapper once per mutant and waits for the lock each time).
Sources are restored byte for byte after every mutant."""
import os
import shutil
import subprocess
import sys

RR = "python/sglang/srt/weg2/read_room.py"
HS = "python/sglang/srt/mem_cache/hicache_storage.py"
UR = "python/sglang/srt/mem_cache/unified_radix_cache.py"
TEST = "test/registered/unit/mem_cache/test_weg2_read_room_1015.py"

MUTANTS = [
    ("M1 post never posts", RR, "    if need <= 0 or not served(arena):\n        return False\n    key = _path(arena)",
     "    return False\n    key = _path(arena)"),
    ("M2 tick never serves", RR, "        return serve(cache, pool, arena, need)", "        return 0"),
    ("M3 armed idle bound = the old bound", RR,
     "    if not served(arena):\n        return int(default_ms)\n    try:", "    return int(default_ms)\n    try:"),
    ("M4 deficit ignores the need", RR, "    return max(0, int(need) - max(0, free) - max(0, clock)), c",
     "    return 0, c"),
    ("M5 the clock is not room (over-asks)", RR, "    clock = (c[\"complete\"] - c[\"pinned\"]) if c[\"pinned\"] >= 0 else 0",
     "    clock = 0"),
    ("M6 a served request is never marked seen (served every round)", RR,
     "        seen[(key, s)] = (seq, t)\n        need = max(need, int(n))", "        need = max(need, int(n))"),
    ("M7 stale requests are served", RR, "or now - t > _STALE_MS:", "or False:"),
    ("M8 not armed outside the local-PP floor is armed anyway", RR,
     "        return bool(_sf.enabled() and getattr(cache, _sf.FLOOR_LOCAL_PP_ATTR, False))",
     "        return bool(_sf.enabled())"),
    ("M9 the switch is ignored", RR, "        return bool(envs.SGLANG_WEG2_READ_ROOM.get())", "        return True"),
    ("M10 the cut is silent", RR, "    _ST.short += 1\n    k = _ST.short", "    return\n    _ST.short += 1\n    k = _ST.short"),
    ("M11 the fill never posts (both sites)", HS,
     "            _read_room.post(arena, len(still_full), site=\"fill\")\n", "            pass\n"),
    ("M12 the hook is not in the round", UR, "        _read_room_tick(self)\n", "        pass\n"),
    ("M14 posters share slots", RR, "    return (free if free is not None else oldest[0]), need", "    return 0, need"),
    ("M15 a second post forgets the larger need", RR,
     "return s, (max(need, int(n)) if now - t < 3000 else need)", "return s, need"),
    ("M16 a foreign arena (mamba anchor) is served too", RR, "    return bool(a) and a == b", "    return True"),
    ("M13 re-posts are not paced", RR, "    if last is not None and now - last < _repost_ms():\n        return False\n",
     ""),
]


def run_tests():
    env = dict(os.environ)
    if os.environ.get("MUTANTS_INSIDE_LOCK"):
        # already inside the wrapper's lock turn: plain pytest, same env the wrapper sets
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH="python:test/registered/unit/weg2")
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", TEST, "-x"]
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
        if old not in src:
            print(f"{name}: PATTERN NOT FOUND in {path}")
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
