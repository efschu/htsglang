"""rc30 R8 mutants (SOLL R8, R2, R6k, R3): each mutates ONE spot of the source, runs ``test_nf_rc30_r8_1021`` in its own
process (systemd-run scope, MemoryMax=2G, no device access, CUDA_VISIBLE_DEVICES="") and expects RED; the source is
restored byte for byte afterwards. Usage (serial): ``python mutants_nf_rc30_r8_1021.py [name ...]`` from the tree root.

  R8-collect-without-decode   the R8 branch goes through the S49 hold again (the rc30 06:36:19Z COLLECT at d_decoding=0)
  R8-fallback-hold            the S1 path (decider off) holds at R8 again
  R2-kurz-grant-counts        kurz grants count into the X-SUM term again (decider path)
  R3-hold-gone                the S49 hold at the R3 timer end (D decodes, halb grant) is dropped -- the Gegenfall
"""
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
PP = "python/flliper/srt/pdflip/phase_policy.py"
FR = "python/flliper/srt/pdflip/front.py"
TESTS = ["test/registered/unit/pdflip/test_nf_rc30_r8_1021.py"]

MUTANTS = {
    "R8-collect-without-decode": (
        PP, 'return SetVerdict(V_D_AFTER_TIMER, "R8 no decode on D: decided now, S=%d < X=%d" % (s, x), 0.0, s, hx)',
        'return _d_verdict("R8 no decode on D: decided now, S=%d < X=%d" % (s, x), 0.0)'),
    "R8-fallback-hold": (
        FR, "        if not decoding:\n            return sv\n        # S56: the rows D has not begun",
        "        # S56: the rows D has not begun"),
    "R2-kurz-grant-counts": (
        FR, "d_pf_unstarted=(Front._d_pf_pending_tokens(self, unstarted_only=True, skip_kurz=True)",
        "d_pf_unstarted=(Front._d_pf_pending_tokens(self, unstarted_only=True)"),
    "R3-hold-gone": (
        PP, "        if gu > 0 and s + gu > x:          # S49",
        "        if False:          # S49"),
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
        r = subprocess.run(["systemd-run", "--scope", "-q", "-p", "MemoryMax=2G", "-p", "DevicePolicy=closed",
                            "/spinning/shvllm/.venv/bin/python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x"] + TESTS,
                           cwd=ROOT, env=env, capture_output=True, text=True)
        tail = [l for l in r.stdout.splitlines() if "passed" in l or "failed" in l or l.startswith("FAILED")]
        return ("RED" if r.returncode != 0 else "SURVIVED") + " :: " + (tail[-1][:160] if tail else "")
    finally:
        open(p, "wb").write(orig)


if __name__ == "__main__":
    for n in (sys.argv[1:] or list(MUTANTS)):
        print("%-28s %s" % (n, run(n)), flush=True)
