#!/usr/bin/env python3
# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Boot arm for the Weg-2 corridor instrument -- the CLI half.

THE DURABLE PATH the record's [SECTION 1x] pointed at and did not have.  All
logic lives in ``sglang.srt.weg2.corridor_arm`` so it is importable and
tested; this file is argument parsing and an exit code.

USAGE, in the order a boot uses them:

    # 1. LIVE, at any moment during the boot (reads only, no GPU window):
    #    does the in-tree sampler reproduce an independent reader?
    scripts/weg2/corridor_arm_check.py --pair

    # 2. AFTER the boot, against its own front log: did it sample, and in
    #    which unit?
    scripts/weg2/corridor_arm_check.py --log /spinning/evidence-665-f1/<stem>.front.log

    # 3. Both, plus the band check (BELOW the floor is a capacity finding,
    #    not an instrument one -- opt in deliberately):
    scripts/weg2/corridor_arm_check.py --log <stem>.front.log --pair --require-in-band

Exit code 0 = every requested check passed, 1 = at least one failed, 2 = the
invocation itself was wrong.  The exit code is the acceptance; the printed
report says which card and by how much.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "python"))

from sglang.srt.weg2 import corridor_arm  # noqa: E402

#: The INDEPENDENT reader, and it lives HERE rather than in ``srt/weg2/`` on
#: purpose.  That package carries a sweep forbidding any second in-tree
#: ``--query-gpu=`` card-memory reader -- the very thing the corridor fix
#: deleted -- and it fired on this query the first time it was written inside
#: the package.  The pairing needs a reader that is genuinely outside the tree
#: whose reader it is checking, so this script owns it and hands the module
#: text.
SMI_ARGV = [
    "nvidia-smi",
    "--query-gpu=index,memory.free",
    "--format=csv,noheader,nounits",
]


def read_independent() -> str:
    return subprocess.run(SMI_ARGV, capture_output=True, text=True, timeout=30).stdout


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", help="a boot's <stem>.front.log")
    ap.add_argument(
        "--pair",
        action="store_true",
        help="read the cards now and pair the in-tree sampler against nvidia-smi",
    )
    ap.add_argument(
        "--smi-output",
        help="independent reader's CSV instead of shelling out (for offline checks)",
    )
    ap.add_argument(
        "--require-in-band",
        action="store_true",
        help="also fail when a per-card minimum is outside the corridor band",
    )
    ap.add_argument(
        "--tolerance-mib",
        type=int,
        default=1,
        help="rounding allowance for --pair (default 1; it is not a margin)",
    )
    args = ap.parse_args(argv)

    if not args.log and not args.pair:
        ap.error("nothing to check: pass --log, --pair, or both")
    if args.tolerance_mib >= min(corridor_arm.KNOWN_CARVE_OUT_MIB):
        ap.error(
            f"--tolerance-mib {args.tolerance_mib} is at or above the smallest known "
            f"driver carve-out ({min(corridor_arm.KNOWN_CARVE_OUT_MIB)} MiB): the pairing "
            "could no longer fail on the defect it exists to catch"
        )

    failed = False
    if args.log:
        rep = corridor_arm.arm_report(args.log, require_in_band=args.require_in_band)
        print(rep.report())
        failed |= not rep.ok
    if args.pair:
        smi = args.smi_output
        if smi is None:
            smi = read_independent()
        elif os.path.exists(smi):
            with open(smi) as f:
                smi = f.read()
        res = corridor_arm.live_pair(smi_text=smi, tolerance_mib=args.tolerance_mib)
        print(res.report())
        failed |= not res.ok
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
