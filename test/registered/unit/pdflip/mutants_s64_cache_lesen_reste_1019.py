"""S64 mutants: each one in the working tree, the S64 test file in ONE 2-GiB
scope (serial, never two at once), the file restored from HEAD afterwards.
Blocking systemd-run scope per mutant (review S64 Auflage 3). Run from the worktree root: python test/registered/unit/pdflip/mutants_s64_cache_lesen_reste_1019.py"""
import os
import subprocess
import sys
import time

T = "test/registered/unit/pdflip/test_pdflip_s64_cache_lesen_reste_1019.py"
RZ = "python/flliper/srt/managers/pdflip_told_raise.py"
ST = "python/flliper/srt/managers/pdflip_store_told.py"
CC = "python/flliper/srt/managers/cache_controller.py"
SC = "python/flliper/srt/managers/scheduler.py"
MUTANTS = [
    # (name, danger, file, old, new)
    ("M1 raise admits without PP0's resumable proof", "falsche KV", RZ,
     "    if not _pp0_resumes_at(scheduler, req, st.target, absolute):", "    if False:"),
    ("M1b dense walk reads past the first short batch", "falsche KV (over-credit)", CC,
     "        if hit < len(b):\n            break", "        if False:\n            break"),
    ("M2 raise on differing acks (max)", "Rang-Uneinigkeit", RZ,
     "    if len(vals) != 1:\n        return None\n    m = vals.pop()", "    m = max(vals)"),
    ("M2b raise releases PP0's own read", "Rang-Uneinigkeit", ST,
     "                setattr(admit, _raise.WIRE_RAISE, 1)",
     "                _fb.release_own_read(scheduler, rid)\n                setattr(admit, _raise.WIRE_RAISE, 1)"),
    ("M3 raise wait unbounded", "Haenger", RZ, "        if now - st.since > wait_s():", "        if False:"),
    ("M3b partial wait ignores the Frist", "Haenger", RZ,
     "            if o is not None and _fb._frist_verdict(scheduler, rid, o, now, pass_n) is None:",
     "            if True:"),
    ("M3c completion mark leaks past its call", "Haenger/falsche Mindestlesung", ST,
     "    finally:\n        setattr(req, COMPLETION_READ_ATTR, None)", "    finally:\n        pass"),
    ("M4a completion min default off", "Feature an", ST,
     'get(COMPLETION_READ_ENV, "1")', 'get(COMPLETION_READ_ENV, "0")'),
    ("M4b raise default off", "Feature an", RZ, 'get(ENV_RAISE, "1")', 'get(ENV_RAISE, "0")'),
    ("M4c dense full span default off", "Feature an", CC,
     'get(S64_DENSE_FULL_SPAN_ENV, "1")', 'get(S64_DENSE_FULL_SPAN_ENV, "0")'),
    ("M4d fresh arm default off", "Feature an", SC,
     'get(S64_STORE_ARM_FRESH_ENV, "1")', 'get(S64_STORE_ARM_FRESH_ENV, "0")'),
    ("M5 short arm answer reused across passes", "nichts schaetzen", SC,
     "                        or int(cached[1]) * page_size >= len(span)):", "                        or True):"),
    ("M7 raise admits without record == m", "Tod (PdFlipStoreToldMismatch auf PP0)", RZ,
     "    if own != int(st.target):", "    if False:"),
    ("M7b raise read unbounded", "Tod (PdFlipStoreToldMismatch auf PP0)", RZ,
     "            verdict = _st.issue_completion_read(lambda r: fn(r, limit_tokens=_lim), req)",
     "            verdict = _st.issue_completion_read(fn, req)"),
    ("M6 regress named as store shortness", "Benennung", "python/flliper/srt/pdflip/short_read_cause.py",
     "        return CAPACITY_REGRESS", "        return STORE_HELD"),
]


def run_one(tag):
    """One BLOCKING 2-GiB scope (no setsid, no polling): serial by construction
    (order after the kernel OOM of 19.10. 11:31:59Z)."""
    cmd = ["systemd-run", "--scope", "-q", "-p", "MemoryMax=2G",
           "-E", "PYTHONPATH=python:test/registered/unit/pdflip", "-E", "CUDA_VISIBLE_DEVICES=",
           "/spinning/shvllm/.venv/bin/python", "-m", "pytest", "-q", "-p", "no:cacheprovider", T]
    p = subprocess.run(cmd, capture_output=True, text=True)
    tail = [l for l in (p.stdout + p.stderr).splitlines() if " passed" in l or " failed" in l]
    return p.returncode, (tail[-1] if tail else "")


def main():
    res = []
    for i, (name, danger, path, old, new) in enumerate(MUTANTS):
        src = open(path).read()
        assert src.count(old) == 1, (name, src.count(old))
        open(path, "w").write(src.replace(old, new))
        try:
            rc, summary = run_one(i)
        finally:
            subprocess.run(["git", "checkout", "--", path], check=True)
        killed = rc != 0
        res.append((name, danger, "KILLED" if killed else "SURVIVED", summary[-120:]))
        print(res[-1], flush=True)
    print("killed %d of %d" % (sum(r[2] == "KILLED" for r in res), len(res)))


if __name__ == "__main__":
    sys.exit(main())
