#!/usr/bin/env python3
"""Mutants of the NF h16s P-gap fixes (desk/nf-h16s-pgap-1016). Each mutant must turn its test red.
Run from the worktree root (ONE lock turn of the shared pytest lock for all mutants):
  /spinning/gpu-arb/pytest_gedeckelt.sh scripts/mutants_pgap_1016.py
Sources are restored byte for byte after every mutant. MUTANTS_ONLY=F3a,F2b selects some."""
import os
import shutil
import subprocess
import sys

ARB = "python/sglang/srt/weg2/room_wait_arbiter.py"
RV = "python/sglang/srt/weg2/room_verdicts.py"
RR = "python/sglang/srt/weg2/read_room.py"
HS = "python/sglang/srt/mem_cache/hicache_storage.py"
CC = "python/sglang/srt/managers/cache_controller.py"
T_F3 = "test/registered/unit/mem_cache/test_weg2_room_arbiter_1016.py"
T_F2 = "test/registered/unit/mem_cache/test_weg2_room_verdicts_1016.py"

MUTANTS = [
    # F3 ROOM-ARBITER: only the oldest request's fill waits; PP0 decides
    ("F3r RED: the base behaviour (arbiter switched off = the old wait byte for byte)", ARB,
     "def enabled() -> bool:\n    raw = (os.environ.get(ENV, \"\") or \"\").strip().lower()\n",
     "def enabled() -> bool:\n    return False\n    raw = (os.environ.get(ENV, \"\") or \"\").strip().lower()\n", T_F3),
    ("F3a PP0 announces the YOUNGEST", ARB,
     "    h, k, _pr = min(((h, k, pr) for h, k, _p, pr in regs), key=lambda hkp: _order_key(*hkp))",
     "    h, k, _pr = max(((h, k, pr) for h, k, _p, pr in regs), key=lambda hkp: _order_key(*hkp))", T_F3),
    ("F3b a fill never yields", ARB,
     "    if since_ms is not None and o[3] <= int(since_ms):\n        return True\n    return False\n",
     "    return True\n", T_F3),
    ("F3c PP0 never announces (the round does not decide)", RR,
     "        _arbiter.decide(arena)\n", "        pass\n", T_F3),
    ("F3d the fill's request is not named", CC,
     "_arbiter.fill_scope(getattr(operation, \"request_id\", None))", "_arbiter.fill_scope(None)", T_F3),
    ("F3e a follower decides too", ARB,
     "    if not enabled() or not is_pp0():\n        return None\n    regs", "    if not enabled():\n        return None\n    regs",
     T_F3),
    ("F3f a newcomer is sent away by an announcement made without it", ARB,
     "    if since_ms is not None and o[3] <= int(since_ms):\n        return True\n", "", T_F3),
    ("F3g the arrival number is ignored (registration order)", ARB,
     "    age = k if k else (1 << 62) + _A.first_seen[h]\n",
     "    age = _A.first_seen[h]\n", T_F3),
    ("F3p the priority is ignored (arrival only)", ARB,
     "    return (_pa.key(prio, age), _A.first_seen[h])", "    return (_pa.key(0, age), _A.first_seen[h])", T_F3),
    ("F3q the registration carries no priority", ARB,
     "_W.pack(h, age_key(rid), me, _now_ms(), 1, priority_of(rid))", "_W.pack(h, age_key(rid), me, _now_ms(), 1, 0)",
     T_F3),
    ("F3s the intake does not note the priority", "python/sglang/srt/managers/scheduler.py",
     "            _arbiter.note_priority(req.rid, _pa.lane_of_req(req))\n", "            pass\n", T_F3),
    # F2 ROOM-VERDICTS: a wait no P rank can answer ends on that fact
    ("F2r RED: the base behaviour (verdicts switched off = the old wait byte for byte)", RV,
     "def enabled() -> bool:\n    raw = (os.environ.get(ENV, \"\") or \"\").strip().lower()\n",
     "def enabled() -> bool:\n    return False\n    raw = (os.environ.get(ENV, \"\") or \"\").strip().lower()\n", T_F2),
    ("F2a one rank's answer is enough", RV, "    if len(fresh) < pp_size:\n        return None\n", "", T_F2),
    ("F2b a write in flight does not hold the wait", RV,
     "    if any(r or tr or q for _p, _t, r, tr, q in fresh.values()):",
     "    if any(r or q for _p, _t, r, tr, q in fresh.values()):", T_F2),
    ("F2c a spill in its backoff counts as having looked", RV,
     "    if any(r or tr or q for _p, _t, r, tr, q in fresh.values()):",
     "    if any(r or tr for _p, _t, r, tr, q in fresh.values()):", T_F2),
    ("F2d answers from before the last progress count", RV, "        if v[1] > int(since_ms):\n", "        if True:\n",
     T_F2),
    ("F2e the serve writes no verdict", RR,
     "    _rv.note_verdict(arena, released, _rv.transient_blocked(cache), _quiet)\n", "", T_F2),
    ("F2f the fill never asks", HS, "                ev = _room_verdicts.futile(arena, rv[0])\n",
     "                ev = None\n", T_F2),
    ("F2g transient ignores the load-back host lock", RV,
     "                    or any(int(getattr(cd, \"host_lock_ref\", 0) or 0) > 0 for cd in node.component_data)):",
     "                    or False):", T_F2),
]


def run_tests(test):
    env = dict(os.environ)
    if os.environ.get("MUTANTS_INSIDE_LOCK"):
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH="python:test/registered/unit/weg2")
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", test, "-x"]
    else:
        cmd = ["/spinning/gpu-arb/pytest_gedeckelt.sh", test, "-x", "-q"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return r.returncode, (r.stdout + r.stderr)[-600:]


def test_all_mutants_are_killed():
    os.environ["MUTANTS_INSIDE_LOCK"] = "1"
    assert main(quiet=False) == 0


def main(quiet=True):
    bad = 0
    only = [x for x in os.environ.get("MUTANTS_ONLY", "").split(",") if x]
    for name, path, old, new, test in MUTANTS:
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
            rc, tail = run_tests(test)
        finally:
            shutil.move(path + ".orig", path)
        killed = rc != 0
        print(f"{name}: {'KILLED' if killed else 'SURVIVED'}")
        if not killed:
            bad += 1
            if not quiet:
                print(tail)
    return bad


if __name__ == "__main__":
    sys.exit(main())
