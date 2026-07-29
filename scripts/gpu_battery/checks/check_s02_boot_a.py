#!/usr/bin/env python3
"""s02 check -- r7c Boot A: did the boot produce an interpretable
accept result?

Content, not exit code. The recipe exits 0 on a run that produced five rows of
None just as happily as on a good one, so what is verified here is:

  * an arm for every one of the five contents (alphabet, squares, repeat,
    code, prose) -- the probe is only comparable across the same five,
  * accept_len_mean is a real number, which is the recipe's OWN abort
    criterion: None means the probe is off or the spec path is not running,
  * the PER-POSITION curve exists and covers positions 0..K-1. A mean is
    structurally blind to a positional pathology; that blindness is how the
    round-7a defect survived, and it is why the curve is mandatory here,
  * the reference column exists and names its source, so no accept number is
    reported on its own,
  * the minimum free MiB per card was recorded (queue item 4),
  * the server log carries no OOM, NCCL error or traceback.

What is deliberately NOT judged: the MAGNITUDE of the accept numbers. The
reproducing and the falsifying outcome are both results, and telling them
apart is the reader's job.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from check_common import check_accept_artifact, run_check  # noqa: E402

STEP = "s02_boot_a"
PROMPTS = ("alphabet", "squares", "repeat", "code", "prose")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--step-dir", required=True)
    args = ap.parse_args()
    return run_check(
        STEP,
        lambda: check_accept_artifact(args.step_dir, "boot_a", PROMPTS, steps_k=3),
    )


if __name__ == "__main__":
    sys.exit(main())
