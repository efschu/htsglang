"""S62 mutants (danger directions of 4a TOLD-TREE-ANCHOR and 4b SYS-GRID TRACK). Usage:
python mutants_s62_recompute_reste_1019.py <copy-worktree>

Each mutant is one exact text replacement in a COPY worktree; its test file must go red. Every pytest run is
detached in its own 4-GiB scope (rig rule), CUDA hidden. The file is restored after each mutant.

Danger directions: zu tiefer Told / falsche KV (a told past the depth PP0's tree resumes; a grid track gathered
at another position than the tree inserts), Rang-Uneinigkeit (PP0's record != the wire told; the deeper told
without the ack stream that lets a follower report a shortfall; a follower acking its store record instead of
its tree; a grid position derived from rank-local state), Haenger (a twin held for a store publish; a promise
booked that is never written nor declined), Slot-Leck (a stale grid plan's slot not freed; the tree not
consuming a grid plan), byte-identity / defaults of the switches, cost bounds (grid outside the first message,
no point cap)."""
import os
import subprocess
import sys
import time

ST = "python/flliper/srt/managers/pdflip_store_told.py"
FB = "python/flliper/srt/managers/pdflip_told_fallback.py"
TA = "python/flliper/srt/pdflip/turn_anchor.py"
EN = "python/flliper/srt/environ.py"
URC = "python/flliper/srt/mem_cache/unified_radix_cache.py"
T4A = "test/registered/unit/managers/test_pdflip_s62_told_tree_anchor_1019.py"
T4B = "test/registered/unit/pdflip/test_pdflip_s62_sys_grid_anchor_1019.py"

MUTANTS = [
    # ---- 4a
    ("A1 told = the KV head, not the tree's resumable depth (too deep: wrong state)", ST, T4A,
     "    final = min(told, int(reach))\n", "    final = told\n"),
    ("A2 PP0's record keeps the store clamp while the wire says the tree depth (start split on PP0)", ST, T4A,
     "        tree._prefetch_completed_tokens[rid] = clamped\n",
     "        tree._prefetch_completed_tokens[rid] = min(clamped, int(_anchor_clamp(scheduler, req, told)))\n"),
    ("A3 the deeper told without the PF ack stream (a short follower can only refuse: rank exit)", ST, T4A,
     "        if not bool(getattr(scheduler, \"_pdflip_told_fallback_on\", False)) or not _tf.enabled():",
     "        if not _tf.enabled():"),
    ("A4 the twin is held while the store lacks the anchor (Haenger: waits for a publish)", ST, T4A,
     "        held.pop(rid, None)\n        _intake = intake_t.pop(rid, now)\n",
     "        if getattr(scheduler, \"_s62_tree_told_n\", 0):\n            continue\n"
     "        held.pop(rid, None)\n        _intake = intake_t.pop(rid, now)\n"),
    ("A5 a satisfied follower acks its store record, not its tree (#869b form)", FB, T4A,
     "    if rid in satisfied:\n        return _satisfied_resumable(scheduler, req, rid, int(told))\n",
     "    if rid in satisfied:\n        return int(_st._completed_prefix(scheduler.tree_cache, rid))\n"),
    ("A6 switch default off (new features on)", ST, T4A,
     "    return str(env.get(ENV_TOLD_TREE_ANCHOR, \"1\") or \"1\")",
     "    return str(env.get(ENV_TOLD_TREE_ANCHOR, \"0\") or \"0\")"),
    # review S62 (19.10.): the follower checks the tree told stands on, and the dual-layout guard
    ("A7 a satisfied follower acks told unprobed (pre-#968 form: KV held, state not)", FB, T4A,
     "    if res is None or int(res) >= told:\n        return told\n    # H64-AUFLAGE 1",
     "    return told\n    # H64-AUFLAGE 1"),
    ("A8 the tree raise on the dual layout (satisfied acks unprobed there)", ST, T4A,
     "    if (os.environ.get(\"FLLIPER_PDFLIP_DUAL_LAYOUT\", \"\") or \"\").strip() == \"1\":\n        # Review S62 (A1)",
     "    if False:\n        # Review S62 (A1)"),
    ("A9 a reading follower acks its KV record unprobed (ACK-RESUMABLE off)", FB, T4A,
     "        res = _tf.rank_resumable(scheduler, req, int(own))\n    except Exception:  # noqa: BLE001 - a probe never breaks the ack\n        return own\n",
     "        res = None\n    except Exception:  # noqa: BLE001 - a probe never breaks the ack\n        return own\n"),
    # ---- 4b
    ("B1 the forward gathers one page below the inserted position (wrong KV/state)", TA, T4B,
     "        desc.add(row, slot, t, prefix, end - prefix)\n        plans.append((t, slot, desc,",
     "        desc.add(row, slot, t - max(1, int(page)), prefix, end - prefix)\n        plans.append((t, slot, desc,"),
    ("B2 a window snaps to its DEEPEST group boundary (the built-in -> MCP line is missed)", TA, T4B,
     "        pts.add(min(win) if win else g)", "        pts.add(max(win) if win else g)"),
    ("B3 promise booked before the slot (no slot: a twin waits for nothing -- Haenger)", TA, T4B,
     "        slot = _alloc_slot(batch)\n        if slot is None:\n            n = _count(\"grid_no_slot\")",
     "        _tw.note_track_planned(rid, t)\n        slot = _alloc_slot(batch)\n        if slot is None:\n"
     "            n = _count(\"grid_no_slot\")"),
    ("B4 a stale grid plan is freed but its promise not declined (Haenger)", TA, T4B,
     "        _free_slot(batch, sg_pend[1])\n        _decline_unconsumed(req, sg_pend[0])\n",
     "        _free_slot(batch, sg_pend[1])\n"),
    ("B5 a stale grid plan's slot leaks", TA, T4B,
     "        _free_slot(batch, sg_pend[1])\n        _decline_unconsumed(req, sg_pend[0])\n",
     "        _decline_unconsumed(req, sg_pend[0])\n"),
    ("B6 the tree does not consume a grid-only plan (anchor never lands, slots leak)", URC, T4B,
     "                or getattr(req, _GEN_PENDING_ATTR, None) is not None\n"
     "                or getattr(req, _SYSGRID_PENDING_ATTR, None)):",
     "                or getattr(req, _GEN_PENDING_ATTR, None) is not None):"),
    ("B7 pop_all_pending drops the grid plans", TA, T4B,
     "        out.extend((\"grid\", p) for p in grid)\n", ""),
    ("B8 position reads rank-local state (R11/#1004)", TA, T4B,
     "    grid = (max(int(grid), page) // page) * page\n",
     "    grid = (max(int(grid), page) // page) * page if not getattr(ids, \"pp_rank\", 0) else page\n"),
    ("B9 switch off ignored (not byte-identical)", TA, T4B,
     "        return bool(envs.FLLIPER_PDFLIP_SYS_GRID_ANCHOR.get()) and int(",
     "        return True or int("),
    ("B10 default off (new features on)", EN, T4B,
     "    FLLIPER_PDFLIP_SYS_GRID_ANCHOR = EnvBool(True)", "    FLLIPER_PDFLIP_SYS_GRID_ANCHOR = EnvBool(False)"),
    ("B11 no SHARED check (two slots, two inserts at one position)", TA, T4B,
     "        if t in taken:\n            n = _count(\"grid_shared\")",
     "        if False:\n            n = _count(\"grid_shared\")"),
    ("B12 grid past the first message (cost: anchors nobody forks at)", TA, T4B,
     "    while g <= end_page:\n", "    while g <= len(ids):\n"),
    ("B13 no point cap (a 200k first message: 48 anchors through a 21-slot share)", TA, T4B,
     "    if max_points > 0 and end_page // grid > max_points:", "    if False:"),
]


def run_tests(wt, tag, test):
    out = os.path.join(os.environ.get("MUT_OUT", "/tmp/s62/mut"), f"mut_{tag}.out")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = out + ".done"
    for f in (out, done):
        if os.path.exists(f):
            os.remove(f)
    cmd = (f"cd {wt} && /spinning/shvllm/.venv/bin/python -m pytest -q -p no:cacheprovider -x {test} "
           f"> {out} 2>&1; echo $? > {done}")
    subprocess.run(["setsid", "-f", "systemd-run", "--scope", "-q", "-p", "MemoryMax=4G",
                    "-E", "PYTHONPATH=python:test/registered/unit/pdflip:test/registered/unit/managers",
                    "-E", "CUDA_VISIBLE_DEVICES=", "bash", "-c", cmd], check=True)
    t0 = time.time()
    while not os.path.exists(done):
        time.sleep(1)
        if time.time() - t0 > 600:
            return None, "timeout"
    time.sleep(0.2)
    rc = int(open(done).read().strip() or 1)
    tail = [ln for ln in open(out).read().splitlines() if "passed" in ln or "failed" in ln or "FAILED" in ln]
    return rc, (tail[-1] if tail else "")


def main(wt):
    for test in (T4A, T4B):
        rc, line = run_tests(wt, "base_" + os.path.basename(test)[:24], test)
        print(f"BASE {test} rc={rc} {line}", flush=True)
        if rc != 0:
            print("base not green -- abort")
            return 2
    killed = 0
    for i, (name, path, test, old, new) in enumerate(MUTANTS):
        p = os.path.join(wt, path)
        src = open(p).read()
        if src.count(old) != 1 and not (path == URC and src.count(old) == 2):
            print(f"{name}: PATTERN count={src.count(old)} -- not applied", flush=True)
            continue
        open(p, "w").write(src.replace(old, new))
        try:
            rc, line = run_tests(wt, f"m{i + 1}", test)
        finally:
            open(p, "w").write(src)
        dead = rc not in (0, None)
        killed += dead
        print(f"{name}: {'KILLED' if dead else 'SURVIVED'} rc={rc} {line}", flush=True)
    print(f"{killed}/{len(MUTANTS)} killed")
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
