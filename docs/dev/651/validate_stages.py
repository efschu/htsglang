#!/usr/bin/env python
"""#651: run every stage's REAL cmdline through the real parser and validator.

A GPU window is the scarcest resource in this task, and the cheapest way to
waste one is an argument combination the tree refuses after CUDA init. The
predecessor's boot.sh STAGE=d did exactly that: it asked for PP=3 together with
NEXTN, which server_args.py:19303 refuses outright, and because that stage was
never executed nobody found out.

This takes each stage's cmdline from rig_boot.sh's own DRYRUN output -- not a
reimplementation of it, so the two cannot drift -- feeds it to
`prepare_server_args`, and then calls `check_server_args()`, which is what
`entrypoints/engine.py:889` calls on a real boot. Construction alone is NOT
enough: ServerArgs(...) accepts PP=3+NEXTN happily and only check_server_args
refuses it.

    PYTHONPATH=<tree>/python python validate_stages.py [stage ...]
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BOOT = os.path.join(HERE, "rig_boot.sh")

# What each stage is supposed to be, so a silently-renumbered ladder is caught
# rather than validated. (tp, pp, spec, flip)
EXPECTED = {
    "a": (1, 1, False, False),
    "b": (1, 1, True, False),
    "c": (2, 1, True, False),
    "d": (1, 3, False, False),
    "e": (1, 3, True, True),
}


def cmdline_for(stage: str, model: str) -> list[str]:
    env = dict(os.environ, STAGE=stage, DRYRUN="1", MODEL=model)
    out = subprocess.run(
        ["bash", BOOT], env=env, capture_output=True, text=True
    )
    if out.returncode != 0:
        raise SystemExit(f"stage {stage}: rig_boot.sh DRYRUN failed:\n{out.stderr}")
    for line in out.stdout.splitlines():
        if "launch_server" in line:
            parts = shlex.split(line)
            return parts[parts.index("sglang.launch_server") + 1 :]
    raise SystemExit(f"stage {stage}: no launch_server line in DRYRUN output")


def main() -> int:
    model = os.environ.get("MODEL")
    if not model:
        raise SystemExit("set MODEL=/abs/path (a .gguf, or a dir holding config.json)")

    stages = sys.argv[1:] or ["a", "b", "c", "d", "e"]
    from sglang.srt.server_args import prepare_server_args

    failures = 0
    for stage in stages:
        argv = cmdline_for(stage, model)
        try:
            sa = prepare_server_args(argv)
            sa.check_server_args()
            verdict, detail = "ACCEPTED", ""
        except AssertionError as exc:
            verdict, detail = "REFUSED", str(exc).replace("\n", " ")[:150]
            failures += 1
        except Exception as exc:  # a ValueError here is also a boot-time death
            verdict, detail = type(exc).__name__, str(exc).replace("\n", " ")[:150]
            failures += 1
            sa = None

        got = None
        if verdict == "ACCEPTED":
            got = (
                sa.tp_size,
                sa.pp_size,
                sa.speculative_algorithm is not None,
                bool(sa.enable_phase_flip),
            )
            if got != EXPECTED[stage]:
                verdict, detail = "SHAPE-MISMATCH", f"got {got}, expected {EXPECTED[stage]}"
                failures += 1

        print(f"stage {stage}: {verdict} {detail}".rstrip())
        if got and verdict == "ACCEPTED":
            print(f"          tp={got[0]} pp={got[1]} spec={got[2]} flip={got[3]}")

    print("VERDICT:", "ALL STAGES PARSE AND VALIDATE" if not failures else f"{failures} PROBLEM(S)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
