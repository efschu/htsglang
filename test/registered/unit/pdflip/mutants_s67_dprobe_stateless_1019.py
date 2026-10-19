"""S67 mutants (danger directions of the stateless D-MATCH-PROBE tokenization). Usage, from anywhere:

    python mutants_s67_dprobe_stateless_1019.py <worktree>

Each mutant is one exact text replacement in <worktree>; ``test_pdflip_s67_dprobe_stateless_1019.py`` must go red.
STRICTLY SERIAL (rig rule 19.10. after the kernel OOM): one pytest at a time, in the FOREGROUND, each in its own
``systemd-run --scope -p MemoryMax=2G``, CUDA hidden -- no ``setsid -f``, no ``xargs -P``. The source file is restored
byte for byte after every mutant (also on Ctrl-C).

Danger directions:
  * the probe REGISTERS a request state under ``<rid>~probe`` (M1, M2): an in-flight request nobody finishes;
  * the probe is tokenized stateful again (M3): the rc16 class, every probe a KeyError, no exact D number;
  * the real request path loses its invariant (M4): a missing ReqState is silently skipped for every request;
  * S67r A3: a real request tokenized stateless (M5): its time_stats lost silently;
  * S67r A2: the probe's image chain on the /health loop again (M6): the D-HEALTH stop;
"""
import os
import subprocess
import sys

HS = "python/flliper/srt/entrypoints/http_server.py"
TMG = "python/flliper/srt/managers/tokenizer_manager.py"
MMO = "python/flliper/srt/managers/mm_tokenize_offload.py"
TEST = "test/registered/unit/pdflip/test_pdflip_s67_dprobe_stateless_1019.py"

# S67r A4: the anchor carries the S67 comment -- after the rc17 merge ``/pdflip/tokenize_ahead`` holds a second
# ``_tokenize_one_request(obj, stateless=True)`` line, and a bare-call anchor would run as "not applied"
_CALL = ("            # S67: the probe is no request of this server -- tokenized WITHOUT a ReqState (rc16: the stateful path\n"
         "            # read rid_to_state['<rid>~probe'] and every probe died of KeyError before the scheduler was asked)\n"
         "            tokenized = await tm._tokenize_one_request(obj, stateless=True)\n")
_CALL_HEAD = _CALL.rsplit("            tokenized", 1)[0]

MUTANTS = [
    ("M1 the probe registers a ReqState under ~probe (and tokenizes stateful)", HS, _CALL,
     _CALL_HEAD
     + "            from types import SimpleNamespace as _NS\n"
       "            tm.rid_to_state[obj.rid] = _NS(time_stats=_NS(set_tokenize_finish_time=lambda: None))\n"
       "            tokenized = await tm._tokenize_one_request(obj)\n"),
    ("M2 the stateless branch creates the state it does not read", TMG,
     "        if stateless:\n",
     "        if stateless:\n"
     "            from types import SimpleNamespace as _NS\n"
     "            self.rid_to_state.setdefault(obj.rid, _NS(time_stats=None))\n"),
    ("M3 the probe is tokenized stateful again (rc16)", HS, _CALL,
     _CALL_HEAD + "            tokenized = await tm._tokenize_one_request(obj)\n"),
    ("M4 every request skips a missing ReqState silently", TMG,
     "        if stateless:\n",
     "        if stateless or obj.rid not in self.rid_to_state:\n"),
    # S67r A3 (reviewer M5): a real request tokenized stateless loses its time_stats silently
    ("M5 generate_request tokenizes stateless", TMG,
     "                        tokenized_obj = await self._tokenize_one_request(obj)\n"
     "                        state = self.rid_to_state[obj.rid]\n",
     "                        tokenized_obj = await self._tokenize_one_request(obj, stateless=True)\n"
     "                        state = self.rid_to_state[obj.rid]\n"),
    # S67r A2: the probe's image chain back on the /health loop
    ("M6 the probe section never offloads", MMO,
     "    offload = bool(is_mm) and offload_permitted()\n    stats = MmTokenizeStats",
     "    offload = False and bool(is_mm) and offload_permitted()\n    stats = MmTokenizeStats"),
]


def run_tests(wt):
    cmd = ["systemd-run", "--scope", "-q", "-p", "MemoryMax=2G",
           "-E", "PYTHONPATH=python:test/registered/unit/pdflip", "-E", "CUDA_VISIBLE_DEVICES=",
           "-E", "PYTHONDONTWRITEBYTECODE=1",
           "/spinning/shvllm/.venv/bin/python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--color=no", TEST]
    p = subprocess.run(cmd, cwd=wt, capture_output=True, text=True, timeout=900)
    tail = [ln for ln in (p.stdout + p.stderr).splitlines() if " passed" in ln or " failed" in ln]
    return p.returncode, (tail[-1].strip() if tail else "")


def main(wt):
    rc, line = run_tests(wt)
    print(f"BASE rc={rc} {line}", flush=True)
    if rc != 0:
        print("base not green -- abort")
        return 1
    bad = 0
    for name, rel, old, new in MUTANTS:
        path = os.path.join(wt, rel)
        src = open(path, "rb").read()
        text = src.decode()
        if text.count(old) != 1:
            print(f"{name}: ANCHOR count={text.count(old)} -- not applied", flush=True)
            bad += 1
            continue
        try:
            open(path, "w").write(text.replace(old, new, 1))
            rc, line = run_tests(wt)
        finally:
            open(path, "wb").write(src)
        killed = rc != 0
        bad += 0 if killed else 1
        print(f"{name}: {'KILLED' if killed else 'SURVIVED'} rc={rc} {line}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")))
