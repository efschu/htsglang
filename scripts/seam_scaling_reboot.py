#!/usr/bin/env python3
"""#631 boot replay: relaunch the LIVE serving instance with ONE thing moved.

Why a replay of /proc rather than another run of the boot script: a
capacity or seam measurement only means something if nothing else moved.
The boot carries ~85 environment variables and 56 arguments; re-deriving
them by hand is how a second variable sneaks in, and it did -- successor
21's hand-built environment silently turned ``PP_STAGE_RATIO=15,10,7``
into an achieved 16/9/7 split and cost two boots.

WHAT CHANGED HERE, AND WHY IT IS NOT A REFINEMENT (successor 22,
2026-08-10). This script used to read ``/tmp/boot_cmdline.txt`` and
``/tmp/boot_env.txt``, files written by whoever last ran the boot script.
Those files were 10 hours stale against the running server and differed
from it in load-bearing ways:

    pp-stage-ratio         2,1,1              live: 14,10,8
    rank-gpu-memory-mib    22700,11920,11970  live: 31800,17400,17450
    SGLANG_UNEVEN_TOKEN_VECTOR  28,26,20      live: 14,10,8
    PHASE_FLIP_PURITY      off                live: strict

So "replay the captured boot with one variable moved" would have booted a
different geometry WITH STRICT PURITY DISABLED and reported the result as
a one-variable step. A tool whose whole purpose is single-variable
discipline cannot take its baseline from a file nobody re-captures. It
now reads the LIVE process by default, and refuses a capture it cannot
tie to a running server unless that is asked for explicitly.

Usage:
  seam_scaling_reboot.py 500000
      Replay the live boot with --max-total-tokens 500000 (the historical
      form; the positional argument is still the pool).
  seam_scaling_reboot.py --set-arg --chunked-prefill-size 8192
  seam_scaling_reboot.py --add-flag --enable-dynamic-chunking
  seam_scaling_reboot.py --set-env PP_STAGE_RATIO 14,10,8 --set-arg ...
      Any number of explicit substitutions. Every one is printed, and the
      full effective diff against the live boot is written next to the log
      so the row in the bench can name what moved.
  seam_scaling_reboot.py --dry-run ...
      Show the substitutions and the diff, launch nothing.

The lever for a pure capacity step is --max-total-tokens, deliberately,
and NOT --rank-gpu-memory-mib: the per-card budget must stay put so the
cards, the weights and the corridor are untouched. Lowering the budget
instead would change free memory AND the pool at once, and the resulting
peak difference would be unattributable.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time

LOG = "/spinning/serving-30030.boot.log"
CAPTURE_DIR = "/spinning/evidence-631/boot-captures"

# Regenerated per boot by the runtime; replaying a stale one would make two
# boots claim the same phase-flip instance id.
DROP_ENV = {"SGLANG_PHASE_FLIP_INSTANCE"}

# Set by whatever shell launched the previous boot; carrying it forward
# pins a stale interpreter path into the replay.
DROP_ENV_PREFIXES = ("_=",)


def find_live_server(port: str = "30030") -> int:
    """PID of the running launch_server for ``port``, or raise.

    Matched on the cmdline rather than on a pidfile: the pidfile is
    written by the boot script and this tool exists precisely because
    boot-script bookkeeping goes stale.
    """
    hits = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as fh:
                argv = fh.read().split(b"\0")
        except OSError:
            continue
        if not argv or b"sglang.launch_server" not in b" ".join(argv):
            continue
        if b"--port" in argv and argv[argv.index(b"--port") + 1] == port.encode():
            hits.append(int(entry))
    if not hits:
        raise SystemExit(
            f"no live sglang.launch_server on port {port}. Boot one with "
            f"PROD_BRINGUP_BENCH.md section 7 first, or pass "
            f"--from-capture <cmdline> <env> and accept that the baseline "
            f"is whatever those files happen to hold."
        )
    if len(hits) > 1:
        raise SystemExit(f"several servers on port {port}: {hits}")
    return hits[0]


def read_proc(pid: int):
    with open(f"/proc/{pid}/cmdline", "rb") as fh:
        argv = [a.decode() for a in fh.read().split(b"\0") if a]
    env = {}
    with open(f"/proc/{pid}/environ", "rb") as fh:
        for item in fh.read().split(b"\0"):
            if not item or b"=" not in item:
                continue
            k, v = item.decode("utf-8", "replace").split("=", 1)
            env[k] = v
    return argv, env


def read_files(cmdline_path: str, env_path: str):
    with open(cmdline_path) as fh:
        argv = [ln.rstrip("\n") for ln in fh if ln.strip() != ""]
    env = {}
    with open(env_path) as fh:
        for ln in fh:
            ln = ln.rstrip("\n")
            if "=" in ln:
                k, v = ln.split("=", 1)
                env[k] = v
    return argv, env


def set_arg(argv, name, value):
    """``--name value``: replace in place, or append if absent."""
    if name in argv:
        idx = argv.index(name)
        if idx + 1 >= len(argv) or argv[idx + 1].startswith("--"):
            raise SystemExit(f"{name} is a bare flag in the live boot, not a value arg")
        old = argv[idx + 1]
        argv[idx + 1] = value
        return f"arg {name}: {old} -> {value}"
    argv.extend([name, value])
    return f"arg {name}: (absent) -> {value}"


def add_flag(argv, name):
    if name in argv:
        return f"flag {name}: already set (no change)"
    argv.append(name)
    return f"flag {name}: (absent) -> set"


def del_flag(argv, name):
    if name not in argv:
        return f"flag {name}: already absent (no change)"
    argv.remove(name)
    return f"flag {name}: set -> removed"


def main():
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("pool", nargs="?", help="--max-total-tokens (historical form)")
    ap.add_argument("--port", default="30030")
    ap.add_argument("--set-arg", nargs=2, action="append", metavar=("NAME", "VALUE"))
    ap.add_argument("--add-flag", action="append", metavar="NAME")
    ap.add_argument("--del-flag", action="append", metavar="NAME")
    ap.add_argument("--set-env", nargs=2, action="append", metavar=("NAME", "VALUE"))
    ap.add_argument("--del-env", action="append", metavar="NAME")
    ap.add_argument("--from-capture", nargs=2, metavar=("CMDLINE", "ENV"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-stop",
        action="store_true",
        help="do not stop the live server first (it will fight for the port)",
    )
    ap.add_argument("--stop-timeout", type=float, default=60.0)
    args = ap.parse_args()

    if args.from_capture:
        argv, env = read_files(*args.from_capture)
        source = f"capture files {args.from_capture[0]} + {args.from_capture[1]}"
        print(
            "WARNING: baseline is a CAPTURE FILE, not a running server. "
            "Nothing verifies it matches anything currently serving."
        )
    else:
        pid = find_live_server(args.port)
        argv, env = read_proc(pid)
        source = f"live pid {pid} on port {args.port}"

    base_argv = list(argv)
    base_env = dict(env)

    changes = []
    if args.pool:
        changes.append(set_arg(argv, "--max-total-tokens", str(int(args.pool))))
    for name, value in args.set_arg or []:
        changes.append(set_arg(argv, name, value))
    for name in args.add_flag or []:
        changes.append(add_flag(argv, name))
    for name in args.del_flag or []:
        changes.append(del_flag(argv, name))
    for name, value in args.set_env or []:
        old = env.get(name, "(absent)")
        env[name] = value
        changes.append(f"env {name}: {old} -> {value}")
    for name in args.del_env or []:
        old = env.pop(name, None)
        changes.append(f"env {name}: {old} -> (removed)")

    for key in DROP_ENV:
        env.pop(key, None)
    for prefix in DROP_ENV_PREFIXES:
        for key in [k for k in env if f"{k}=".startswith(prefix)]:
            env.pop(key, None)

    if not changes:
        raise SystemExit(
            "no substitution requested -- this tool exists to move exactly "
            "one thing; relaunching an identical boot is what the boot "
            "script is for"
        )

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    print(f"baseline: {source}")
    print(f"substitutions ({len(changes)}):")
    for line in changes:
        print(f"  {line}")
    if len(changes) > 1:
        print(
            "NOTE: more than one variable moved. The result is not a "
            "single-variable step and must not be reported as one."
        )

    os.makedirs(CAPTURE_DIR, exist_ok=True)
    record = os.path.join(CAPTURE_DIR, f"replay-{stamp}.txt")
    with open(record, "w") as fh:
        fh.write(f"baseline: {source}\nstamp: {stamp}\n\nsubstitutions:\n")
        for line in changes:
            fh.write(f"  {line}\n")
        fh.write("\nbaseline argv:\n")
        fh.write("\n".join(base_argv) + "\n")
        fh.write("\nreplay argv:\n")
        fh.write("\n".join(argv) + "\n")
        fh.write("\nbaseline env:\n")
        for k in sorted(base_env):
            fh.write(f"{k}={base_env[k]}\n")
        fh.write("\nreplay env:\n")
        for k in sorted(env):
            fh.write(f"{k}={env[k]}\n")
    print(f"replay record: {record}")

    if args.dry_run:
        print("--dry-run: nothing launched")
        return 0

    # CAPTURE, THEN STOP, THEN LAUNCH -- in that order and inside one tool.
    # The baseline is read from the live process, so a successor who kills
    # the server first has nothing left to replay; doing it by hand in two
    # steps is how the stale capture files came to exist in the first
    # place. Stopped by PID (SIGTERM, then SIGKILL on the same pid after a
    # bounded wait) -- never a pattern kill: this box runs the router on
    # 30099 and other agents' processes, and a broad pkill has taken those
    # out before.
    if not args.from_capture and not args.no_stop:
        import signal

        print(f"stopping live pid {pid} (SIGTERM)")
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        deadline = time.time() + args.stop_timeout
        while time.time() < deadline:
            if not os.path.exists(f"/proc/{pid}"):
                break
            time.sleep(0.5)
        if os.path.exists(f"/proc/{pid}"):
            print(f"pid {pid} still up after {args.stop_timeout}s; SIGKILL")
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            time.sleep(3.0)
        # The schedulers are children and normally follow the parent down.
        # Verify rather than assume: a surviving scheduler still holds its
        # card, and the replacement would then boot into a card that is
        # already full and blame the pool for it.
        stragglers = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as fh:
                    cmd = fh.read()
            except OSError:
                continue
            if b"sglang::scheduler_PP" in cmd or b"sglang::detokenizer" in cmd:
                stragglers.append(int(entry))
        for spid in stragglers:
            print(f"straggler {spid} still holding a card; SIGKILL by pid")
            try:
                os.kill(spid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if stragglers:
            time.sleep(3.0)

    if os.path.exists(LOG):
        os.rename(LOG, f"/spinning/serving-30030.boot.{stamp}.log")
    with open(LOG, "wb") as out:
        proc = subprocess.Popen(
            argv,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    print(f"relaunched as pid {proc.pid}; log {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
