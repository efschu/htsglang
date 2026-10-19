"""S55 mutants (danger directions of P-POOL-FROM-P). Usage: python mutants_s55_p_pool_from_p_1019.py <worktree>.

Each mutant is one exact text replacement in a COPY worktree; the S55 test file must go red. Every pytest run is
detached in its own 4-GiB scope (rig rule), CUDA hidden. The file is restored after each mutant.

Danger directions: R11/#1004 (ONE reading, the min-reduced realised pool, decided by the front -- never the argv pin,
never a rank-local value), over-plan (a budget above P's pool: a leg 1 whose pages P cannot hold, READ-ROOM fight,
recompute), Haenger (the read gating the drain / startup), byte-identity of the off forms (switch 0, rule 0)."""
import os
import subprocess
import sys
import time

PP = "python/flliper/srt/pdflip/phase_policy.py"
FR = "python/flliper/srt/pdflip/front.py"
TEST = "test/registered/unit/pdflip/test_pdflip_s55_p_pool_from_p_1019.py"

MUTANTS = [
    ("M1 reads the argv pin max_total_tokens (R11: not P's realised pool)", PP,
     'raw = info.get("max_total_num_tokens")', 'raw = info.get("max_total_num_tokens", info.get("max_total_tokens"))'),
    ("M1b reads only the argv pin", PP,
     'raw = info.get("max_total_num_tokens")', 'raw = info.get("max_total_tokens")'),
    ("M2 a written value above P's pool is not clamped (over-plan)", PP,
     "    if c > r:\n        return r, P_POOL_SRC_CLAMPED\n", ""),
    ("M3 a written value is ignored (always P's pool)", PP,
     "    if not written:\n        return r, P_POOL_SRC_REALISED\n", "    return r, P_POOL_SRC_REALISED\n"),
    ("M4 the rule-off form (0) is switched on by P's reading", PP,
     "    if c <= 0:\n        return 0, P_POOL_SRC_OFF\n", ""),
    ("M5 unread P = no token plan (0: unlimited overlap)", PP,
     "        return c, P_POOL_SRC_CONFIGURED", "        return 0, P_POOL_SRC_CONFIGURED"),
    ("M6 a zero/negative reading is taken", PP,
     "    return v if v > 0 else None", "    return v"),
    ("M7 the boot read gives up after one failure (budget never follows)", FR,
     "            if self._apply_p_pool_info(body):\n                return\n",
     "            self._apply_p_pool_info(body)\n            return\n"),
    ("M8 switch off still reads", FR,
     "if not envs.FLLIPER_PDFLIP_P_POOL_FROM_P.get() or self.p_pool_tokens_configured <= 0:",
     "if self.p_pool_tokens_configured <= 0:"),
    ("M9 written marked after the defaults filled it (the default counts as written: no raise)", FR,
     '    _p_pool_written = getattr(args, "p_pool_tokens", None) is not None\n'
     '    phase_policy.resolve_front_defaults(args, bool(envs.FLLIPER_PDFLIP_STANDARD_FORM.get()))',
     '    phase_policy.resolve_front_defaults(args, bool(envs.FLLIPER_PDFLIP_STANDARD_FORM.get()))\n'
     '    _p_pool_written = getattr(args, "p_pool_tokens", None) is not None'),
    ("M10 startup awaits the read (Haenger: the front waits on P before serving)", FR,
     'app["p_pool_from_p"] = asyncio.create_task(self._p_pool_boot())', 'await self._p_pool_boot()'),
    ("M11 the read asks D (wrong group)", FR,
     'g = self.groups.get("P") if isinstance(getattr(self, "groups", None), dict) else None',
     'g = self.groups.get("D") if isinstance(getattr(self, "groups", None), dict) else None'),
    ("M12 an unusable body zeroes the budget", FR,
     "        if realised is None:\n            return False\n",
     "        if realised is None:\n            self.p_pool_tokens = 0\n            return False\n"),
]


def run_tests(wt, tag):
    out = os.path.join(os.environ.get("MUT_OUT", "/tmp/s55"), f"mut_{tag}.out")
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
