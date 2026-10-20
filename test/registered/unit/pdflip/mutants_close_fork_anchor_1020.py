"""CLOSE-FORK TRACK mutants (danger directions). Usage:
python mutants_close_fork_anchor_1020.py <copy-worktree>

Each mutant is one exact text replacement in a COPY worktree; the close-fork test file must go red. Runs are
SERIAL (rig rule 19.10.: no parallel pytest), each in its own 2-GiB scope, CUDA hidden. The file is restored after
each mutant.

Danger directions: falscher/zu tiefer Anker = falsche KV (a position past the side request's claim; the forward
gathering at another position than the tree inserts; no separator check), Haenger (a promise booked that is never
written nor declined; a leaked slot), Rang-Uneinigkeit (a position from rank-local state), the tree not consuming
the plan (the anchor never lands), wasted slots (a track where the turn track already serves), switch coupling
(no own switch: armed with the turn anchor)."""
import os
import subprocess
import sys

TA = "python/flliper/srt/pdflip/turn_anchor.py"
URC = "python/flliper/srt/mem_cache/unified_radix_cache.py"
TEST = "test/registered/unit/pdflip/test_pdflip_close_fork_anchor_1020.py"

MUTANTS = [
    ("M1 claim bound without the bigram -1 (too deep: past the side request's claim)", TA,
     "    t = ((e - 1) // grain) * grain\n    return t if t > 0 else None\n\n\ndef req_close_pos",
     "    t = (e // grain) * grain\n    return t if t > 0 else None\n\n\ndef req_close_pos"),
    ("M2 separator length 1 (wrong position)", TA, "CLOSE_SEP = 2\n", "CLOSE_SEP = 1\n"),
    ("M3 separator not verified (an anchor where no message closes)", TA,
     "    if sep_last != sep_gen or int(tok) in sep_last:\n        return None\n",
     "    if int(tok) in sep_last:\n        return None\n"),
    ("M4 the forward gathers one page below the inserted position (wrong state)", TA,
     "    desc.add(row, slot, t, prefix, end - prefix)\n    setattr(req, CLOSE_PENDING_ATTR,",
     "    desc.add(row, slot, t - max(1, int(page)), prefix, end - prefix)\n    setattr(req, CLOSE_PENDING_ATTR,"),
    ("M5 promise booked before the slot (no slot: a twin waits for nothing -- Haenger)", TA,
     "    slot = _alloc_slot(batch)\n    if slot is None:\n        n = _count(\"close_no_slot\")",
     "    from flliper.srt.pdflip import twin_anchor as _tw0\n    _tw0.note_track_planned(rid, t)\n    slot = _alloc_slot(batch)\n    if slot is None:\n        n = _count(\"close_no_slot\")"),
    ("M6 a stale close plan is freed but its promise not declined (Haenger)", TA,
     "        _free_slot(batch, cl_stale[1])\n        _decline_unconsumed(req, cl_stale[0])\n",
     "        _free_slot(batch, cl_stale[1])\n"),
    ("M7 a stale close plan's slot leaks", TA,
     "        _free_slot(batch, cl_stale[1])\n        _decline_unconsumed(req, cl_stale[0])\n",
     "        _decline_unconsumed(req, cl_stale[0])\n"),
    ("M8 position reads rank-local state (R11/#1004)", TA,
     "    grain = max(1, int(grain))\n    t = ((e - 1) // grain) * grain",
     "    grain = max(1, int(grain)) if not getattr(ids, \"pp_rank\", 0) else 1\n    t = ((e - 1) // grain) * grain"),
    ("M9 a track also where the main track lies at/below it (no main-track bound)", TA,
     "    t = step_target(prefix, end, req_close_pos(req, tok, page), chunk, main_track)",
     "    t = step_target(prefix, end, req_close_pos(req, tok, page), chunk, end)"),
    ("M10 the tree does not consume a close plan (anchor never lands, slot leaks)", URC,
     "                or getattr(req, _SYSGRID_PENDING_ATTR, None)\n                or getattr(req, _CLOSE_PENDING_ATTR, None) is not None):",
     "                or getattr(req, _SYSGRID_PENDING_ATTR, None)):"),
    ("M11 no SHARED check (two slots, two inserts at one position)", TA,
     "    if t in taken:\n        n = _count(\"close_shared\")",
     "    if False:\n        n = _count(\"close_shared\")"),
    ("M12 pop_all_pending drops the close plan", TA, "        out.append((\"close\", close))\n", ""),
    ("M13 the close track books no promise (a twin of the source is not held: recompute)", TA,
     "    _tw.note_track_planned(rid, t)\n    n = _count(\"close_planned\")",
     "    n = _count(\"close_planned\")"),
    ("M14 coupled to the GEN-FORK switch (an own switch by proxy)", TA,
     "    if tok is not None:\n        # CLOSE-FORK TRACK: armed wherever the turn anchor is (``tok``), no own switch\n",
     "    if tok is not None and gen_fork_armed():\n        # CLOSE-FORK TRACK: armed wherever the turn anchor is (``tok``), no own switch\n"),
    ("M15 the close track is drawn even where the turn track already serves (wasted slot)", TA,
     "    if t in taken:\n        n = _count(\"close_shared\")",
     "    if t in taken and t != req_anchor_pos(req, tok, page):\n        n = _count(\"close_shared\")"),
    ("M16 the close track is never drawn (rc28)", TA,
     "    t = step_target(prefix, end, req_close_pos(req, tok, page), chunk, main_track)",
     "    t = None"),
]


def run_tests(wt, tag):
    out_dir = os.environ.get("MUT_OUT", "/tmp/closefork/mut")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"mut_{tag}.out")
    cmd = (f"cd {wt} && /spinning/shvllm/.venv/bin/python -m pytest -q -p no:cacheprovider -x {TEST} > {out} 2>&1")
    r = subprocess.run(["systemd-run", "--scope", "-q", "-p", "MemoryMax=2G",
                        "-E", "PYTHONPATH=python:test/registered/unit/pdflip", "-E", "CUDA_VISIBLE_DEVICES=",
                        "timeout", "600", "bash", "-c", cmd])
    tail = [ln for ln in open(out).read().splitlines() if "passed" in ln or "failed" in ln or "FAILED" in ln]
    return r.returncode, (tail[-1] if tail else "")


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
        want = 2 if path == URC else 1
        if src.count(old) != want:
            print(f"{name}: PATTERN count={src.count(old)} -- not applied", flush=True)
            continue
        open(p, "w").write(src.replace(old, new))
        try:
            rc, line = run_tests(wt, f"m{i + 1}")
        finally:
            open(p, "w").write(src)
        dead = rc != 0
        killed += dead
        print(f"{name}: {'KILLED' if dead else 'SURVIVED'} rc={rc} {line}", flush=True)
    print(f"{killed}/{len(MUTANTS)} killed")
    return 0 if killed == len(MUTANTS) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
