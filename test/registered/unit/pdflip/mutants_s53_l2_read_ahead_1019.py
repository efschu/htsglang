"""S53 mutants (danger directions of L2-READ-AHEAD). Usage: python mutants_s53_l2_read_ahead_1019.py <worktree>.

Each mutant is one exact text replacement in a COPY worktree; the S53 test file must go red. Every pytest run is
detached in its own 4-GiB scope (rig rule), CUDA hidden. The file is restored after each mutant."""
import os
import subprocess
import sys
import time

L2 = "python/flliper/srt/pdflip/l2_read_ahead.py"
FR = "python/flliper/srt/pdflip/front.py"
SC = "python/flliper/srt/managers/scheduler.py"
TEST = "test/registered/unit/pdflip/test_pdflip_s53_l2_read_ahead_1019.py"

MUTANTS = [
    ("M1 read-ahead takes references (pins L2 slots)", L2,
     "room_wait=False,\n                                           take_ref=False)",
     "room_wait=False,\n                                           take_ref=True)"),
    ("M2 read-ahead room-waits (READ-ROOM spill for a request not on P)", L2,
     "prefix=True, room_wait=False,", "prefix=True, room_wait=True,"),
    ("M3 armed on followers too", L2,
     "if int(getattr(ps, \"pp_rank\", 0) or 0) != 0:\n            return False", "pass"),
    ("M4 front hints a phase-cap wait too (hook at the cap break)", FR,
     "                idx = _exempt_index()\n                if idx is None:\n                    break",
     "                idx = _exempt_index()\n                if idx is None:\n                    if on_pool_hold is not None:\n                        on_pool_hold(queue[0])\n                    break"),
    ("M5 dormant P keeps the job", L2,
     "if bool(getattr(s, \"pdflip_dormant\", False)):\n            return \"dormant\"", "pass"),
    ("M6 scheduler ignores ple=False (PLE admit on an awake P)", SC,
     "        if not getattr(recv_req, \"ple\", True):\n            return  #", "        if False:\n            return  #"),
    ("M7 a newer hint does not replace the waiting one", L2,
     "                self.waiting = {\"rid\": rid, \"ids\": ids, \"span\": int(span), \"t\": self._now()}",
     "                self.waiting = self.waiting or {\"rid\": rid, \"ids\": ids, \"span\": int(span), \"t\": self._now()}"),
    ("M8 span not cut (pages past start_hint read)", L2,
     "span = (min(int(span), max(0, usable)) // page) * page", "span = (max(0, usable) // page) * page"),
    ("M9 no lead gate (starts at the hint)", L2,
     "if rem > lead_tokens(self.scheduler):\n            return None", "pass"),
    ("M10 resident pages asked again", L2,
     "missing = [s for s, r in zip(stems, resident) if not r]", "missing = list(stems)"),
    ("M11 front hints while P is asleep", L2,
     "if sent or client_gone or skip_leg1 or awake != \"P\":", "if sent or client_gone or skip_leg1:"),
    ("M12 front re-hints every hold (no once-per-request mark)", FR,
     "        p._l2_ra_sent = True   # once per request", "        pass   # once per request"),
    ("M13 no protect scope over the span (own L2 head evicted)", L2,
     "_pdflip_protect_stems=list(stems))", "_pdflip_protect_stems=[])"),
    ("M14 unigram keys on a bigram tree", L2,
     "bigram = bool(getattr(tree, \"is_eagle\", False))", "bigram = False"),
]


def run_tests(wt, tag):
    out = f"/tmp/s53/mut_{tag}.out"
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


def main(wt, only=None):
    results = []
    rc, line = run_tests(wt, "base")
    print(f"BASE rc={rc} {line}", flush=True)
    if rc != 0:
        print("base not green -- abort")
        return 1
    for i, (name, rel, old, new) in enumerate(MUTANTS, 1):
        if only and not name.startswith(only + " "):
            continue
        path = os.path.join(wt, rel)
        src = open(path).read()
        if src.count(old) != 1:
            print(f"{name}: ANCHOR count={src.count(old)} -- not applied", flush=True)
            results.append((name, "anchor"))
            continue
        open(path, "w").write(src.replace(old, new))
        try:
            rc, line = run_tests(wt, f"m{i}")
        finally:
            open(path, "w").write(src)
        verdict = "KILLED" if rc not in (0, None) else ("SURVIVED" if rc == 0 else "TIMEOUT")
        print(f"{name}: {verdict} ({line})", flush=True)
        results.append((name, verdict))
    killed = sum(1 for _, v in results if v == "KILLED")
    print(f"SUMMARY killed={killed}/{len(results)}", flush=True)
    return 0 if killed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None))
