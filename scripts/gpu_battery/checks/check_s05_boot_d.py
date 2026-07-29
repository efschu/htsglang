#!/usr/bin/env python3
"""s05 check -- r7c Boot D: lane re-seed A/B.

Boot D writes reseed.json, not accept.json: two arms per content in one boot,
re-seed on and off, driven through the job body.

Verified:
  * all three contents (squares, code, prose) present,
  * BOTH arms present per content -- an A/B with one arm is not an A/B,
  * accept_len_mean and decode_ms_mean are real numbers in both arms; the
    price of the re-seed is the point of the boot and it is read off
    decode_ms_mean, so a missing timing makes the boot worthless,
  * reseed_forwards is recorded on the re-seed arm, i.e. the arm actually
    re-seeded rather than silently behaving like the control,
  * output_identical is present as a boolean,
  * no OOM / NCCL error / traceback in the server log.

Explicitly NOT a failure: output_identical == False. Whether the two arms
produce the same tokens IS the measurement. A check that failed on it would be
a check that refuses to accept the answer it was sent to collect.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_common import (  # noqa: E402
    CheckFail,
    check_vram_summary,
    classify_missing_result,
    load_json,
    require_number,
    run_check,
    scan_log_for_fatals,
)

STEP = "s05_boot_d"
PROMPTS = ("squares", "code", "prose")


def check(step_dir: str) -> None:
    path = os.path.join(step_dir, "reseed.json")
    classify_missing_result(step_dir, "boot_d", path, "reseed.json")
    report = load_json(path, "boot_d: reseed.json")

    arms = report.get("arms")
    if not isinstance(arms, list) or not arms:
        raise CheckFail("boot_d: reseed.json hat keine arms")
    by_prompt = {a.get("prompt"): a for a in arms if isinstance(a, dict)}

    absent = [p for p in PROMPTS if p not in by_prompt]
    if absent:
        raise CheckFail(f"boot_d: Prompts ohne Arm: {','.join(absent)}")

    for prompt in PROMPTS:
        row = by_prompt[prompt]
        sub = row.get("arms")
        if not isinstance(sub, dict):
            raise CheckFail(f"boot_d/{prompt}: kein arms-Block")
        for key in ("True", "False"):
            if key not in sub:
                raise CheckFail(
                    f"boot_d/{prompt}: Arm reseed={key} fehlt -- ein A/B mit einem "
                    "Arm ist kein A/B"
                )
            arm = sub[key]
            require_number(
                arm.get("accept_len_mean"), f"boot_d/{prompt}/{key}: accept_len_mean"
            )
            require_number(
                arm.get("decode_ms_mean"), f"boot_d/{prompt}/{key}: decode_ms_mean"
            )
        if sub["True"].get("reseed_forwards") is None:
            raise CheckFail(
                f"boot_d/{prompt}: reseed_forwards fehlt im Re-Seed-Arm -- nicht "
                "belegbar, dass der Arm ueberhaupt re-seedet hat"
            )
        if not isinstance(row.get("output_identical"), bool):
            raise CheckFail(
                f"boot_d/{prompt}: output_identical ist {row.get('output_identical')!r}, "
                "kein Bool"
            )

    check_vram_summary(step_dir, "boot_d")

    fatal = scan_log_for_fatals(
        os.path.join(step_dir, "server.log"), "boot_d: server.log"
    )
    if fatal:
        raise CheckFail(f"boot_d: Fatal im Serverlog -- {fatal}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(STEP, lambda: check(args.step_dir))


if __name__ == "__main__":
    sys.exit(main())
