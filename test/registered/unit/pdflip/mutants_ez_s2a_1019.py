"""EZ-S2a mutants (plan R.6: M6 M8 M11 M12 M16, plus the S2a glue): each mutates ONE spot of the source, runs
``test_decide_set_1019`` + ``test_decide_s4_compat_1019`` in its own process and expects RED; the source is restored
byte for byte afterwards. Usage (serial): ``python mutants_ez_s2a_1019.py [name ...]`` from the tree root."""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PP = "python/flliper/srt/pdflip/phase_policy.py"
FR = "python/flliper/srt/pdflip/front.py"
TESTS = ["test/registered/unit/pdflip/test_decide_set_1019.py", "test/registered/unit/pdflip/test_decide_s4_compat_1019.py"]

MUTANTS = {
    # M6: R6 counts "granted, not started" as running
    "M6": (PP, "    sv = set_verdict(keep, **kw)\n",
           "    kw['d_prefill_running_uncached'] = int(kw.get('d_prefill_running_uncached', 0)) + int(kw.get('d_granted_unstarted', 0))\n"
           "    sv = set_verdict(keep, **kw)\n"),
    # M8: S from S_P (the deduplicated figure) instead of the members' own u
    "M8": (PP, "s = sum(max(0, int(m.own_uncached or m.uncached)) for m in members) + running",
           "s = sum(max(0, int(m.uncached)) for m in members) + running"),
    # M11: a kurz member counts in S
    "M11": (PP, "        if c == CLASS_KURZ:\n            out.append(m.rid)\n            continue\n",
            "        if False:\n            out.append(m.rid)\n            continue\n"),
    # M12a/b: the static X instead of x_own (policy / front)
    "M12a": (PP, "c = classify_arrival(m.own_uncached or m.uncached, int(m.x_own) or xc, half_frac, m.p_only)",
             "c = classify_arrival(m.own_uncached or m.uncached, xc, half_frac, m.p_only)"),
    "M12b": (FR, "x_own=(int(xs.get(r, 0)) if _xo else 0))", "x_own=0)"),
    # M16: p_only per state under X (A5)
    "M16": (FR, "                _p_only = (bool(getattr(q, \"p_only\", False)) or int(getattr(q, \"reroutes\", 0) or 0) > 0\n"
            "                           or (int(getattr(q, \"x_requeues\", 0) or 0) > 0 and bool(getattr(q, \"x_ref_nonum\", False))))\n",
            "                _p_only = bool(getattr(q, \"p_only\", False)) or (u < xq and not _meas)\n"),
    # glue
    "G1-second-set": (FR, "        if _fd.enabled():\n            # EZ-S2a (R1, plan A8 contradiction 3, #41)", "        if False:\n            # EZ-S2a (R1, plan A8 contradiction 3, #41)"),
    "G2-economics-verdict": (FR, "            return True\n        return ok", "            return ok\n        return ok"),
    "G3-needs-p-state": (PP, "    if decider_on():\n        for p in queue:", "    if False:\n        for p in queue:"),
    "G4-needs-p-boundary": (PP, "return int(est_uncached) >= max(1, int(x_own))", "return int(est_uncached) > max(1, int(x_own))"),
    "G5-leg2-static-x": (FR, "_x = Front._x_own(self, ct) if _fd.enabled() else self.tp_prefill_max_tokens",
                         "_x = self.tp_prefill_max_tokens"),
    "G6-foreseeable-never": (FR, "        return ev is None or ev[1].verdict not in (phase_policy.V_COLLECT, phase_policy.V_D_AFTER_TIMER)",
                             "        return False"),
    "G8-nonum-stale-estimate": (PP, "if p_only or int(reroutes or 0) > 0 or no_number:", "if p_only or int(reroutes or 0) > 0:"),
    "G9-kurz-only-hold": (FR, "            if members and not _dec.in_set:", "            if False:"),
    "G7-s49-tag-lost": (PP, 'return SetVerdict(V_COLLECT, "NS:S49 X-SUM', 'return SetVerdict(V_COLLECT, "X-SUM'),
}


def run(name):
    path, old, new = MUTANTS[name]
    p = os.path.join(ROOT, path)
    orig = open(p, "rb").read()
    src = orig.decode()
    if src.count(old) != 1:
        return "NOT-APPLIED (%d matches)" % src.count(old)
    try:
        open(p, "w").write(src.replace(old, new))
        env = dict(os.environ, PYTHONPATH="python:test/registered/unit/pdflip", CUDA_VISIBLE_DEVICES="")
        r = subprocess.run(["systemd-run", "--scope", "-q", "-p", "MemoryMax=2G", "/spinning/shvllm/.venv/bin/python", "-m",
                            "pytest", "-q", "-p", "no:cacheprovider", "-x"] + TESTS, cwd=ROOT, env=env,
                           capture_output=True, text=True)
        tail = [l for l in r.stdout.splitlines() if "passed" in l or "failed" in l or l.startswith("FAILED")]
        return ("RED" if r.returncode != 0 else "SURVIVED") + " :: " + (tail[-1][:160] if tail else "")
    finally:
        open(p, "wb").write(orig)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(MUTANTS)):
        print("%-22s %s" % (n, run(n)), flush=True)
