"""S57 mutants (danger directions of the GEN-FORK TRACK). Usage:
python mutants_s57_gen_fork_anchor_1019.py <copy-worktree>

Each mutant is one exact text replacement in a COPY worktree; the S57 test file must go red. Every pytest run is
detached in its own 4-GiB scope (rig rule), CUDA hidden. The file is restored after each mutant.

Danger directions: falscher/zu tiefer Anker = falsche KV (the forward gathers at another position than the tree
inserts; a position past the sibling's claim; a track where the END anchor already serves), Haenger (a promise
booked that is never written nor declined: a twin waits until its Frist; a leaked slot), Rang-Uneinigkeit (a
position derived from rank-local state instead of the ids), byte-identity / default of the switch, the tree not
consuming the plan (the anchor never lands)."""
import os
import subprocess
import sys
import time

TA = "python/flliper/srt/pdflip/turn_anchor.py"
EN = "python/flliper/srt/environ.py"
URC = "python/flliper/srt/mem_cache/unified_radix_cache.py"
TEST = "test/registered/unit/pdflip/test_pdflip_s57_gen_fork_anchor_1019.py"

MUTANTS = [
    ("M1 position from N-1 instead of the generation prompt (too deep: past the sibling)", TA,
     "    g = _fa.fork_cut(ids, tok, tail)\n    if g is None:\n        return None\n    grain = max(1, int(grain))\n    t = (int(g) // grain) * grain",
     "    g = _fa.fork_cut(ids, tok, tail)\n    if g is None:\n        return None\n    grain = max(1, int(grain))\n    t = ((len(ids) - 1) // grain) * grain"),
    ("M2 the forward gathers one page below the inserted position (wrong KV/state)", TA,
     "    desc.add(row, slot, t, prefix, end - prefix)\n    setattr(req, GEN_PENDING_ATTR,",
     "    desc.add(row, slot, t - max(1, int(page)), prefix, end - prefix)\n    setattr(req, GEN_PENDING_ATTR,"),
    ("M3 promise booked before the slot (no slot: a twin waits for nothing -- Haenger)", TA,
     "    slot = _alloc_slot(batch)\n    if slot is None:\n        n = _count(\"gen_no_slot\")",
     "    from flliper.srt.pdflip import twin_anchor as _tw0\n    _tw0.note_track_planned(rid, t)\n    slot = _alloc_slot(batch)\n    if slot is None:\n        n = _count(\"gen_no_slot\")"),
    ("M4 a stale gen plan is freed but its promise not declined (Haenger)", TA,
     "        _free_slot(batch, gn_stale[1])\n        _decline_unconsumed(req, gn_stale[0])\n",
     "        _free_slot(batch, gn_stale[1])\n"),
    ("M5 a stale gen plan's slot leaks", TA,
     "        _free_slot(batch, gn_stale[1])\n        _decline_unconsumed(req, gn_stale[0])\n",
     "        _decline_unconsumed(req, gn_stale[0])\n"),
    ("M6 position reads rank-local state (R11/#1004)", TA,
     "    grain = max(1, int(grain))\n    t = (int(g) // grain) * grain\n    return t if t > 0 else None\n\n\ndef req_gen_pos",
     "    grain = max(1, int(grain)) if not getattr(ids, \"pp_rank\", 0) else 1\n    t = (int(g) // grain) * grain\n    return t if t > 0 else None\n\n\ndef req_gen_pos"),
    ("M7 switch off ignored (not byte-identical)", TA,
     "        return bool(envs.FLLIPER_PDFLIP_GEN_FORK_ANCHOR.get())", "        return True"),
    ("M8 default off (new features on)", EN,
     "    FLLIPER_PDFLIP_GEN_FORK_ANCHOR = EnvBool(True)", "    FLLIPER_PDFLIP_GEN_FORK_ANCHOR = EnvBool(False)"),
    ("M9 a track also where the END anchor serves (no main-track bound: wasted slot, unproven geometry)", TA,
     "    t = step_target(prefix, end, req_gen_pos(req, tok, page), chunk, main_track)",
     "    t = step_target(prefix, end, req_gen_pos(req, tok, page), chunk, end)"),
    ("M10 the tree does not consume a gen-only plan (anchor never lands, slot leaks)", URC,
     "                or getattr(req, _FORK_PENDING_ATTR, None) is not None\n                or getattr(req, _GEN_PENDING_ATTR, None) is not None):",
     "                or getattr(req, _FORK_PENDING_ATTR, None) is not None):"),
    ("M11 no SHARED check (two slots, two inserts at one position)", TA,
     "    if t in taken:\n        n = _count(\"gen_shared\")",
     "    if False:\n        n = _count(\"gen_shared\")"),
    ("M12 pop_all_pending drops the gen plan", TA,
     "        out.append((\"gen\", gen))\n", ""),
    ("M13 the gen track books no promise (a twin of the source is not held: recompute)", TA,
     "    _tw.note_track_planned(rid, t)\n    n = _count(\"gen_planned\")",
     "    n = _count(\"gen_planned\")"),
]


def run_tests(wt, tag):
    out = os.path.join(os.environ.get("MUT_OUT", "/tmp/s57/mut"), f"mut_{tag}.out")
    done = out + ".done"
    for f in (out, done):
        if os.path.exists(f):
            os.remove(f)
    cmd = (f"cd {wt} && /spinning/shvllm/.venv/bin/python -m pytest -q -p no:cacheprovider -x {TEST} "
           f"> {out} 2>&1; echo $? > {done}")
    subprocess.run(["setsid", "-f", "systemd-run", "--scope", "-q", "-p", "MemoryMax=4G",
                    "-E", "PYTHONPATH=python:test/registered/unit/pdflip", "-E", "CUDA_VISIBLE_DEVICES=",
                    "bash", "-c", cmd], check=True)
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
    rc, line = run_tests(wt, "base")
    print(f"BASE rc={rc} {line}", flush=True)
    if rc != 0:
        print("base not green -- abort")
        return 2
    killed = 0
    for i, (name, path, old, new) in enumerate(MUTANTS):
        p = os.path.join(wt, path)
        src = open(p).read()
        if src.count(old) != 1 and not (path == URC and src.count(old) == 2):
            print(f"{name}: PATTERN count={src.count(old)} -- not applied", flush=True)
            continue
        open(p, "w").write(src.replace(old, new))
        try:
            rc, line = run_tests(wt, f"m{i + 1}")
        finally:
            open(p, "w").write(src)
        dead = rc not in (0, None)
        killed += dead
        print(f"{name}: {'KILLED' if dead else 'SURVIVED'} rc={rc} {line}", flush=True)
    print(f"{killed}/{len(MUTANTS)} killed")
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
