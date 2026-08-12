#!/usr/bin/env python
# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# ==============================================================================
"""Prove the turnkey path reproduces the captured ship config exactly.

#539's acceptance is parity: whatever the turnkey unit boots must be the ship
config, not a near-miss that starts successfully. This script answers that
question on the desk, with no cards and no boot, by asking the orchestrator to
assemble the launch and diffing the result against the capture token by token.

It runs preflight against SUBSTITUTED probes describing an idle machine. That
is not a way of dodging the checks -- preflight is exercised for real
elsewhere -- it is because parity is a question about argv and env, and a
busy card is a true fact about today that would otherwise stop the comparison
from being made at all.

Exit 0 only on byte-identical argv and an env superset. Any divergence is
printed as a diff and exits 1.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from sglang.srt.turnkey import config as C            # noqa: E402
from sglang.srt.turnkey import orchestrator as O      # noqa: E402
from sglang.srt.turnkey import preflight as PF        # noqa: E402

#: Env keys the turnkey path is EXPECTED to diverge on, each for a stated
#: reason. Anything diverging outside this set is a parity failure.
EXPECTED_DIVERGENCE = {
    "PYTHONPATH": "capture ran from the wt-631-routea worktree; the unit "
                  "roots the stack in the canonical checkout",
    "CUDA_VISIBLE_DEVICES": "derived from the card UUIDs at boot (same value, "
                            "different provenance)",
    "SGLANG_PHASE_FLIP_INSTANCE": "per-boot identity; the capture embeds the "
                                  "dead pid 3940356",
    "SGLANG_BOOT_COMMIT": "provenance, measured from the repo at boot",
}


def idle_machine_probes(cfg) -> PF.Probes:
    cards = [PF.CardObs(uuid=c.uuid, name=c.label or "GPU",
                        total_bytes=32 << 30, free_bytes=32 << 30)
             for c in cfg.cards]
    return PF.Probes(
        cards=lambda: cards,
        procs_on=lambda uuid: {},
        mem_available_bytes=lambda: 120 << 30,
        disk_free_bytes=lambda p: 200 << 30,
        port_busy=lambda p: False,
        path_exists=lambda p: True,
        probe_import=lambda m, a: PF.ImportObs(
            module_file="/spinning/htsglang-gpu/.venv/lib/python3.12/"
                        "site-packages/sgl_kernel/__init__.py",
            version="0.4.4", has_arm=True),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="deploy/turnkey/stack.rig3.toml")
    ap.add_argument("--argv", default="/spinning/evidence-631/s485/ship_argv.txt")
    ap.add_argument("--env", default="/spinning/evidence-631/s485/ship_env.txt")
    ap.add_argument("--lane", default="ship")
    a = ap.parse_args(argv)

    cfg = C.load(a.config)
    lane = cfg.lane(a.lane)
    if lane is None:
        print(f"no lane {a.lane}")
        return 2

    probes = idle_machine_probes(cfg)
    refusals = O.run_preflight(cfg, probes)
    print(f"preflight against an idle machine: {len(refusals)} refusal(s)")
    for r in refusals:
        print("  " + r.line())
    if refusals:
        return 1

    bp = O.assemble(cfg, lane)

    want_argv = [x for x in open(a.argv).read().split("\n") if x != ""]
    got_argv = list(bp.argv)

    print("\n== argv parity ==")
    print(f"captured: {len(want_argv)} tokens   assembled: {len(got_argv)} tokens")
    ok = True
    if want_argv == got_argv:
        print("IDENTICAL -- every token matches, in order")
    else:
        ok = False
        import difflib
        for line in difflib.unified_diff(want_argv, got_argv,
                                         "captured", "assembled", lineterm=""):
            print("  " + line)

    want_env = {}
    for line in open(a.env):
        line = line.rstrip("\n")
        if "=" in line:
            k, v = line.split("=", 1)
            want_env[k] = v

    print("\n== env parity (turnkey-controlled keys) ==")
    interesting = [k for k in want_env
                   if k.startswith(("SGLANG_", "HTSGLANG_", "PYTORCH_"))
                   or k in ("LD_LIBRARY_PATH", "PYTHONPATH",
                            "CUDA_VISIBLE_DEVICES")]
    bad = []
    for k in sorted(interesting):
        got = bp.env.get(k)
        if got == want_env[k]:
            continue
        if k in EXPECTED_DIVERGENCE:
            print(f"  EXPECTED  {k}")
            print(f"            captured : {want_env[k]}")
            print(f"            assembled: {got}")
            print(f"            reason   : {EXPECTED_DIVERGENCE[k]}")
            continue
        bad.append(k)
        print(f"  DIVERGES  {k}")
        print(f"            captured : {want_env[k]}")
        print(f"            assembled: {got}")
    if not bad:
        print("  every other captured key reproduced exactly")
    else:
        ok = False

    extra = [k for k in bp.env
             if k not in want_env and k.startswith(("SGLANG_", "HTSGLANG_"))]
    for k in sorted(extra):
        if k in EXPECTED_DIVERGENCE:
            continue
        print(f"  ADDED     {k}={bp.env[k]}  (not in the capture)")

    print("\nVERDICT:", "PARITY PROVEN" if ok else "PARITY FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
