#!/usr/bin/env python3
"""#631 seam-scaling probe: relaunch the LIVE boot with one variable moved.

Why a replay of /proc rather than another run of the boot script: the
question is whether the phase-flip cutover's memory peak scales with the
KV POOL, and that only means something if nothing else moved. The boot
script carries ~85 environment variables and 56 arguments; re-deriving
them by hand is how a second variable sneaks in. So the launch is read
back from the running process and replayed verbatim, with exactly one
substitution.

The lever is --max-total-tokens, deliberately, and NOT --rank-gpu-memory-mib:
the per-card budget stays at 22700,11920,11970 so the cards, the weights
and the corridor are untouched. Lowering the budget instead would change
free memory AND the pool at once, and the resulting peak difference would
be unattributable.

Usage: seam_scaling_reboot.py <max_total_tokens>
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

CMDLINE = "/tmp/boot_cmdline.txt"
ENVFILE = "/tmp/boot_env.txt"
LOG = "/spinning/serving-30030.boot.log"

# Regenerated per boot by the runtime; replaying a stale one would make two
# boots claim the same phase-flip instance id.
DROP_ENV = {"SGLANG_PHASE_FLIP_INSTANCE"}


def read_lines(path):
    with open(path) as fh:
        return [ln.rstrip("\n") for ln in fh if ln != ""]


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    cap = str(int(sys.argv[1]))

    argv = [a for a in read_lines(CMDLINE) if a != ""]
    if "--max-total-tokens" not in argv:
        raise SystemExit("captured cmdline has no --max-total-tokens")
    argv[argv.index("--max-total-tokens") + 1] = cap

    env = {}
    for line in read_lines(ENVFILE):
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k not in DROP_ENV:
            env[k] = v

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if os.path.exists(LOG):
        os.rename(LOG, f"/spinning/serving-30030.boot.{stamp}.log")

    with open(LOG, "wb") as out:
        subprocess.Popen(
            argv,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    print(f"relaunched with --max-total-tokens {cap}; log {LOG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
