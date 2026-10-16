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
T_F5 = "test/registered/unit/mem_cache/test_weg2_quiesce_publish_f5_1016.py"
BPUB = "python/sglang/srt/managers/weg2_bubble_publish.py"
UR = "python/sglang/srt/mem_cache/unified_radix_cache.py"
NBK = "python/sglang/srt/managers/weg2_flush_nonblock.py"

MUTANTS = [
    # F3 ROOM-ARBITER (+PRIO, + review 1016 A3/A4/A5)
    ("F3r RED: the base behaviour (arbiter switched off = the old wait byte for byte)", ARB,
     "def enabled() -> bool:\n    raw = (os.environ.get(ENV, \"\") or \"\").strip().lower()\n",
     "def enabled() -> bool:\n    return False\n    raw = (os.environ.get(ENV, \"\") or \"\").strip().lower()\n", T_F3),
    ("F3a PP0 announces the YOUNGEST", ARB,
     "    h, k, _pr = min(((h, k, pr) for h, k, _p, pr in regs), key=lambda hkp: _order_key(*hkp))",
     "    h, k, _pr = max(((h, k, pr) for h, k, _p, pr in regs), key=lambda hkp: _order_key(*hkp))", T_F3),
    ("F3b a fill never yields", HS, "                    arb[2] = True\n                    arb[5] = _held\n                    break\n",
     "                    pass\n", T_F3),
    ("F3c PP0 never announces (the round does not decide)", RR, "        _arbiter.decide(arena)\n", "        pass\n", T_F3),
    ("F3d the fill's request is not named", CC,
     "_arbiter.fill_scope(getattr(operation, \"request_id\", None), operation)", "_arbiter.fill_scope(None, operation)",
     T_F3),
    ("F3e a follower decides too", ARB,
     "    if not enabled() or not is_pp0():\n        return None\n    t_dec", "    if not enabled():\n        return None\n    t_dec",
     T_F3),
    ("F3f a newcomer is sent away by an announcement made without it", ARB,
     "    if since_ms is not None and o[3] <= int(since_ms):\n        return True\n", "", T_F3),
    ("F3g the arrival number is ignored (registration order)", ARB,
     "    age = k if k else (1 << 62) + _A.first_seen[h]\n", "    age = _A.first_seen[h]\n", T_F3),
    ("F3p the priority is ignored (arrival only)", ARB,
     "    return (_pa.key(prio, age), _A.first_seen[h])", "    return (_pa.key(0, age), _A.first_seen[h])", T_F3),
    ("F3q the registration carries no priority", ARB, "1, priority_of(rid), max(0, int(held)))", "1, 0, max(0, int(held)))",
     T_F3),
    ("F3s the intake does not note the priority", "python/sglang/srt/managers/scheduler.py",
     "            _arbiter.note_priority(req.rid, _pa.lane_of_req(req))\n", "            pass\n", T_F3),
    # (A3a "a fill that holds nothing yields too" is EQUIVALENT: such a fill defers before its claim round and never
    #  reaches the yield check -- A3c covers the deferral)
    ("A3b a younger fill yields without its own futility evidence", HS,
     "                if rv is None or _room_verdicts.futile(arena, rv[0], until_ms=_t_claim, rid=arb[1],",
     "                if True or _room_verdicts.futile(arena, rv[0], until_ms=_t_claim, rid=arb[1],", T_F3),
    ("A3c a non-owner that holds nothing competes for the owner's room", HS, "            if _defer:\n                continue\n", "",
     T_F3),
    ("A4 a terminated operation does not end the wait", HS,
     "            if _op is not None and _arbiter.op_terminated(_op):", "            if False:", T_F3),
    ("A5 PP0 stamps after its read", ARB, "        m[0:_HDR.size] = _HDR.pack(_MAGIC, h, k, os.getpid(), _A.seq, t_dec, _A.pp_size, 0)",
     "        m[0:_HDR.size] = _HDR.pack(_MAGIC, h, k, os.getpid(), _A.seq, _now_ms(), _A.pp_size, 0)", T_F3),
    ("A3d an ended owner still sends fills away", ARB,
     "    if not any(h == o[0] for h, _k, _p, _pr in registrations(arena)):\n        return True", "    if False:\n        return True",
     T_F3),
    # F2 ROOM-VERDICTS (+ review 1016 A1/A2/A7, foreign references)
    ("F2r RED: the base behaviour (verdicts switched off = the old wait byte for byte)", RV,
     "def enabled() -> bool:\n    raw = (os.environ.get(ENV, \"\") or \"\").strip().lower()\n",
     "def enabled() -> bool:\n    return False\n    raw = (os.environ.get(ENV, \"\") or \"\").strip().lower()\n", T_F2),
    ("F2a one rank's answer is enough", RV, "    if len(fresh) < pp_size:\n        return None\n", "", T_F2),
    ("F2b I/O in flight does not hold the wait", RV, "    if any(r or tr or q or rs for", "    if any(r or q or rs for", T_F2),
    ("F2c a spill in its backoff counts as having looked", RV, "    if any(r or tr or q or rs for", "    if any(r or tr or rs for", T_F2),
    ("F2d answers from before the last progress count", RV,
     "        if v[1] > int(since_ms) and (until_ms is None or v[1] <= int(until_ms)):",
     "        if (until_ms is None or v[1] <= int(until_ms)):", T_F2),
    ("F2e the serve writes no verdict", RR, "    _rv.note_verdict(arena, released, _rv.transient_blocked(cache), _quiet or _w3rs.undecided(cache))\n", "",
     T_F2),
    ("F2f the fill never asks", HS, "                ev = _room_verdicts.futile(arena, rv[0], until_ms=_t_claim,",
     "                ev = None and _room_verdicts.futile(arena, rv[0], until_ms=_t_claim,", T_F2),
    ("A1a a serve that finds room writes a futility answer", RV, "    if any(r or tr or q or rs for", "    if any(r or tr or q for", T_F2),
    ("A1b answers after the last claim round count", RV,
     "        if v[1] > int(since_ms) and (until_ms is None or v[1] <= int(until_ms)):", "        if v[1] > int(since_ms):", T_F2),
    ("A2a a pending give-back is not room on its way", RV, "    if givebacks(arena):\n        return None\n", "", T_F2),
    ("A2b a yielded fill records no give-back", HS, "                    _room_verdicts.note_giveback(arena, _arb_rid, _arb[5])\n",
     "", T_F3),
    ("A2c another request's holding fill is ignored", RV, "    if not ignore_holders and _arb.holders(arena, exclude_rid=rid):\n        return None",
     "    if False:\n        return None", T_F2),
    ("A2d a give-back is cleared before its reap", RV, "    done = [rid for rid in list(_S.givebacks) if ongoing is None or rid not in ongoing]",
     "    done = list(_S.givebacks)", T_F2),
    ("A7a transient misses the store writes", RV,
     "    for attr in (\"ongoing_write_through\", \"ongoing_backup\", \"ongoing_load_back\", \"_weg2_direct_mamba_rows\"):",
     "    for attr in (\"ongoing_write_through\", \"ongoing_load_back\", \"_weg2_direct_mamba_rows\"):", T_F2),
    ("A7b transient misses a running chunk's device lock", RV,
     "            n += sum(1 for v in prot.values() if int(v or 0) > 0)", "            n += 0", T_F2),
    ("Dref foreign references are ignored", RV, "        if int(refs_now) > sum(held):\n            return None\n", "", T_F2),
    # F5 quiesce publish (a) bubble pause, (b) memo-lite, (c) FLUSH-UNBACKED
    ("F5r RED: the base behaviour (bubble pause switched off)", BPUB,
     "def quiesce_pause_on() -> bool:\n", "def quiesce_pause_on() -> bool:\n    return False\n", T_F5),
    ("F5a the quiesce never opens", BPUB,
     "    if not getattr(scheduler, \"_weg2_quiesce_open\", False):\n        scheduler._weg2_quiesce_open = True\n",
     "    if False:\n        scheduler._weg2_quiesce_open = True\n", T_F5),
    ("F5b a launched batch does not end the quiesce", BPUB,
     "    if getattr(scheduler, \"_weg2_quiesce_open\", False):\n        quiesce_end(scheduler, \"batch\")\n", "", T_F5),
    ("F5c memo-lite never consulted", UR,
     "            _in_memo = _memo is not None and node.id in _memo\n", "            _in_memo = False\n", T_F5),
    ("F5d memo-lite never records", UR, "                        _memo[node.id] = True\n", "                        pass\n", T_F5),
    ("F5e a memo skip does not engage the sweep's full-arena mode", UR,
     "                    _full_arena = _full_arena or _full_arena_armed\n                _issue_s", "                _issue_s", T_F5),
    ("F5f the census counts backed nodes", NBK,
     "        if node is root or getattr(node, \"evicted\", False) or getattr(node, \"backuped\", False) \\\n",
     "        if node is root or getattr(node, \"evicted\", False) \\\n", T_F5),
    ("F5g a follower's intake does not open the quiesce", "python/sglang/srt/managers/scheduler_components/request_receiver.py",
     "                    _f5_bubble.quiesce_begin(self)\n", "                    pass\n", T_F5),
    ("F1x an undecided group spill answers like a looking one", RR,
     "_rv.transient_blocked(cache), _quiet or _w3rs.undecided(cache))", "_rv.transient_blocked(cache), _quiet)",
     "test/registered/unit/mem_cache/test_weg2_readroom_f1_f2_joint_1016.py"),
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
