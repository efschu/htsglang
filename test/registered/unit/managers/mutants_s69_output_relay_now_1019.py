"""S69 mutants (danger directions of OUTPUT-RELAY-NOW). Usage: python mutants_s69_output_relay_now_1019.py <worktree>.

Each mutant is one exact text replacement in the worktree; the S69 test file must go red. STRICTLY SERIAL (rig rule
19.10., incident parallel test runs -> kernel OOM): one pytest at a time, each in its own
``systemd-run --scope -p MemoryMax=2G``, foreground, CUDA hidden. The file is restored after each mutant (finally).

Danger directions:
  * token duplicated or dropped on the relay hop (#631 class: a token the client never sees, or a stray message
    that puts every later receive off by one) -- M1, M2;
  * the join of a relayed send moves EARLIER than the old ring's join of the same message (#1015e: the join is the
    cycle; a wedge, not a slowdown) -- M3;
  * the relay where the old relay is the only correct one: the last rank (source), the gapped wire (its own
    forward_now), XPU (parity-ordered, blocking isend), async depth > 0, switch 0 -- M4..M8;
  * the relay moved in the loop (before the proxy send = wire order changes; after the chain flush) or the
    cutover reset dropped -- M9..M11.
"""
import os
import subprocess
import sys

MX = "python/flliper/srt/managers/scheduler_pp_mixin.py"
TEST = "test/registered/unit/managers/test_pp_output_relay_now_s69_1019.py"
PY = "/spinning/shvllm/.venv/bin/python"

MUTANTS = [
    ("M1 relay posts AND keeps pp_outputs (the next exchange forwards it again: duplicate)", MX,
     "                if self._pp_relay_output_now(next_pp_outputs):\n                    self.pp_outputs = None\n",
     "                if self._pp_relay_output_now(next_pp_outputs):\n                    self.pp_outputs = next_pp_outputs\n"),
    ("M2 relay claims the pass but posts nothing (drop)", MX,
     "        if outputs:\n            with torch.profiler.record_function(\"send_res_dict_to_next_stage\"):\n"
     "                work = self._pp_send_dict_to_next_stage(",
     "        if outputs and False:\n            with torch.profiler.record_function(\"send_res_dict_to_next_stage\"):\n"
     "                work = self._pp_send_dict_to_next_stage("),
    ("M3 relayed send filed at the current lap (joined one lap earlier than the old ring)", MX,
     "self._pp_post_send(pending_output_work, lap_offset=1)",
     "self._pp_post_send(pending_output_work, lap_offset=0)"),
    ("M4 the last rank relays too", MX,
     '    if getattr(getattr(holder, "pp_group", None), "is_last_rank", True):\n        return False\n', ""),
    ("M5 the gapped wire relays too", MX,
     '    if getattr(holder, "_pp_gapped_wire", False):\n        return False\n    if is_xpu():', "    if is_xpu():"),
    ("M6 XPU relays too", MX,
     "    if is_xpu():\n        return False\n    depth =", "    depth ="),
    ("M7 async batch depth > 0 relays too", MX,
     "    return int(depth or 0) == 0\n", "    return True\n"),
    ("M8 switch 0 still relays", MX,
     "    if not envs.FLLIPER_PP_OUTPUT_RELAY_NOW.get():\n        return False\n", ""),
    ("M9 relay before the proxy send in the loop (wire order changes)", MX,
     "                    if cur_batch:\n                        self.device_module.current_stream().wait_event(\n"
     "                            self.launch_event\n                        )\n                        with torch.profiler.record_function(\n"
     "                            \"send_proxy_dict_to_next_stage\"",
     "                    _s69_early = self._pp_relay_output_now(next_pp_outputs)\n"
     "                    if cur_batch:\n                        self.device_module.current_stream().wait_event(\n"
     "                            self.launch_event\n                        )\n                        with torch.profiler.record_function(\n"
     "                            \"send_proxy_dict_to_next_stage\""),
    ("M10 cutover reset forgets the relay mark", MX,
     "        self._pp_relay_posted = False\n        self.launch_event = None\n", "        self.launch_event = None\n"),
    ("M11 relay mark not cleared by the commit that files it", MX,
     "                self._pp_relay_posted = False\n                self._pp_post_send(pending_output_work, lap_offset=1)",
     "                self._pp_post_send(pending_output_work, lap_offset=1)"),
]


def run_tests(wt):
    cmd = ["systemd-run", "--scope", "-q", "-p", "MemoryMax=2G",
           "-E", "PYTHONPATH=python:test/registered/unit/managers", "-E", "CUDA_VISIBLE_DEVICES=",
           "-E", "PYTHONDONTWRITEBYTECODE=1",
           PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-x", TEST]
    try:
        p = subprocess.run(cmd, cwd=wt, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return None, "timeout"
    tail = [ln for ln in (p.stdout + p.stderr).splitlines() if " passed" in ln or " failed" in ln]
    return p.returncode, (tail[-1].strip() if tail else (p.stdout + p.stderr)[-300:])


def main(wt, only=None):
    results = []
    rc, line = run_tests(wt)
    print(f"BASE rc={rc} {line}", flush=True)
    if rc != 0:
        print("base not green -- abort")
        return 1
    for name, rel, old, new in MUTANTS:
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
            rc, line = run_tests(wt)
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
