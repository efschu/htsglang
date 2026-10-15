#!/usr/bin/env python3
"""Mutants of READ-ROOM-D 1015 (weg2/read_room.py: arming, the Form A dispatch, ``form_a_spill``). Each mutant must
turn test_weg2_read_room_d_1015.py red. Run from the worktree root:
  PYTEST_PYTHON=<venv python> /spinning/gpu-arb/pytest_gedeckelt.sh scripts/mutants_arenaread_d_1015.py -q -s
(ONE lock turn of the shared pytest lock for all mutants). Sources are restored byte for byte after every mutant."""
import os
import shutil
import subprocess
import sys

RR = "python/sglang/srt/weg2/read_room.py"
TEST = "test/registered/unit/mem_cache/test_weg2_read_room_d_1015.py"

MUTANTS = [
    ("D1 Form A TP0 not armed", RR,
     '    return _local_pp(cache) or form_a_role() in ("host", "worker")\n',
     "    return _local_pp(cache)\n"),
    ("D2 Form A TP0 does not serve", RR,
     '    return _local_pp(cache) or form_a_role() == "host"\n',
     "    return _local_pp(cache)\n"),
    ("D3 a worker serves on its own", RR,
     '    return _local_pp(cache) or form_a_role() == "host"\n',
     '    return _local_pp(cache) or form_a_role() in ("host", "worker")\n'),
    ("D4 TP0's serve goes to the W3 spill (returns 0 under R12)", RR,
     '    if form_a_role() == "host":\n', "    if False:\n"),
    ("D5 no R12 verdict (the workers keep their references)", RR,
     '        _r12.record_state(cache, node, why="read-room")\n', ""),
    ("D6 a node holding a Mamba anchor on the host is released", RR,
     "        if mamba is not None and node.component_data[mamba].host_value is not None:\n"
     "            skip[\"anchor\"] += 1\n            continue\n", ""),
    ("D7 a page without an L3 copy is released", RR,
     '        if int(sec.get("lost", 0)) or not int(sec.get("pages", 0)):\n',
     '        if not int(sec.get("pages", 0)):\n'),
    ("D8 no L3 copy is made before the release", RR,
     "        sec = pool.secure_rows_to_l3(hv)\n", '        sec = {"pages": 1, "lost": 0}\n'),
    ("D9 a busy node (host lock / pending write) is released", RR,
     '            skip["busy"] += 1\n            continue\n', '            skip["busy"] += 1\n'),
    ("D10 a host-only node is released", RR,
     '            skip["host_only"] += 1\n            continue\n', '            skip["host_only"] += 1\n'),
    ("D11 form_a_spill runs on a worker (rank-local release)", RR,
     '    if _r12.role() != "host":\n        return 0   # never rank-local',
     '    if _r12.role() is None:\n        return 0   # never rank-local'),
    ("D13 the hybrid group's anchor host pool is not resolved", RR,
     "        pool = host if host is not None and hasattr(host, \"secure_rows_to_l3\") else None\n",
     "        pool = None\n"),
    ("D12 the want is ignored (everything goes)", RR,
     "        if released >= want:\n            break\n        hv = node.component_data[BASE_COMPONENT_TYPE].host_value\n",
     "        hv = node.component_data[BASE_COMPONENT_TYPE].host_value\n"),
]


def run_tests():
    env = dict(os.environ)
    if os.environ.get("MUTANTS_INSIDE_LOCK"):
        env.update(CUDA_VISIBLE_DEVICES="", PYTHONPATH="python:test/registered/unit/weg2:test/registered/unit/mem_cache")
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
