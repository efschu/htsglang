#!/usr/bin/env python3
"""#862/#864 -- what a suite is actually TELLING you when it says nothing.

A gate that skips 1506 tests is indistinguishable, at the summary line, from a
gate that passed them. A gate that reports 16 failures is indistinguishable
from a gate with 16 defects -- unless somebody reads WHY. #868 §2.1 already
turned one such count (15 "failures") into a property of how the gate was
invoked, with zero defects in it. This tool asks that question mechanically
instead of by hand.

It runs a suite hermetically and classifies every non-pass outcome BY REASON:

  failures
    DEVICE      the test needs an accelerator and there is none. Not a defect;
                a property of the invocation. (#868 §0.7's class.)
    REAL        anything else -- until shown otherwise, a defect.

  skips
    BY DESIGN   the reason names a requirement the desk genuinely cannot meet
                (a device, a second node, a model file, a network endpoint).
                Darkness that was chosen.
    BY ACCIDENT the reason is an import error, an attribute error, an empty
                string, or a bare marker. NOBODY chose this: the test would
                very likely run here, and it is silently not running.
    UNCLASSIFIED reason text this tool does not recognise. Reported as its own
                bucket and never folded into either of the others, because
                quietly bucketing the unknown is how a dark suite reads green.

THE ASYMMETRY THAT MAKES THIS WORTH RUNNING. `BY DESIGN` is a cost.
`BY ACCIDENT` is a false green: a test that the desk could run, that is not
running, and whose absence no count reveals. Same shape as #868's partition
violation, on the skip axis instead of the partition axis.

CUDA_VISIBLE_DEVICES is forced empty and not overridable from the environment.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_partition_lib import ANSI, parse_log  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PY = os.environ.get("GATE_PY", "/spinning/htsglang-gpu/.venv/bin/python3")

# `-rs` short summary: "SKIPPED [12] path/to/test.py:34: <reason>"
SKIP_LINE = re.compile(r"^SKIPPED\s+\[(\d+)\]\s+([^:]+):(\d+):\s*(.*)$")

# A device-shaped failure. These are the signatures #868 §0.7 measured on the
# three modules that pass with cards visible and fail with CVD="".
DEVICE_FAIL = re.compile(
    r"No CUDA GPUs are available"
    r"|Torch not compiled with CUDA"
    r"|CUDA driver version"
    r"|no kernel image is available"
    r"|Found no NVIDIA driver"
    r"|CUDA (?:unknown )?error"
    r"|device-side assert"
    r"|requires (?:a )?(?:CUDA|GPU|accelerator)"
    r"|cuda(?::0)? is not available"
    r"|AssertionError: Torch not compiled",
    re.I,
)

# --- the buckets, in priority order ------------------------------------------
#
# CORRECTED BY THE DATA, and the correction is the point. The first version of
# this tool had two buckets, BY DESIGN and BY ACCIDENT, on the assumption that
# a skip either names hardware or is a mistake. The first real run put 636 of
# 1658 skips in neither: reasons like "requires SWA", "requires Mamba
# component", "page_size > 1 only". Those are not darkness at all -- the file
# runs one test body against a matrix of cache configurations, and each body
# only applies to some of them. A test that does not apply to this
# parametrisation is not a test that is being hidden.
#
# Folding those 636 into "dark" would have inflated the darkness number by 62 %
# and pointed the ticket at a problem that does not exist. Folding them into
# "by design" would have hidden them inside a bucket that means something else.
# They get their own name.

# The desk genuinely lacks the hardware. Dark, and dark ON PURPOSE.
HARDWARE = re.compile(
    r"\bcuda\b|\bgpu\b|accelerator|\bdevice\b|nvidia|nvml"
    r"|multi[- ]?node|second node|world_size|nccl"
    r"|amd|rocm|\bhip\b|\bxpu\b|\bnpu\b|\btpu\b|musa|ascend|mlx|cpu-only build",
    re.I,
)

# An optional third-party component is not installed. Dark on purpose too, but
# curable by installing something rather than by buying a card.
DEPENDENCY = re.compile(
    r"unavailable: please install|not installed|requires? the .* package"
    r"|backend unavailable|no module named ['\"]?(?:nixl|mooncake|infinistore)",
    re.I,
)

# The test body does not apply to THIS parametrisation. NOT darkness.
CONFIG = re.compile(
    r"requires? (?:swa|mamba|full|page_size|ps |sw |a? ?\w+[- ]only)"
    r"|page_size|sliding_window|fixture required|fixture does not support"
    r"|test scenario requires|only\b.*\bconfig|config\b.*\bonly"
    r"|\bonly$|covered on|keeps the (?:expected|chain)"
    # Added after READING all 36 reasons the first pass left UNCLASSIFIED, not
    # by widening the pattern until the bucket emptied. Every phrase below was
    # inspected individually; each names a component shape or a fixture scope,
    # none names a resource this desk lacks.
    r"|aux(?:iliary)? component|out of scope for this .*fixture"
    r"|[-\s]only path|accounts in pages|keeps the .* (?:simple|precise)",
    re.I,
)

# Nobody chose this: an import broke, or the reason is empty.
BY_ACCIDENT = re.compile(
    r"could not import|cannot import|importerror|attributeerror|modulenotfound"
    r"|^unconditional skip$|^skipped$|^no reason",
    re.I,
)


def classify_skip(reason: str) -> str:
    r = reason.strip()
    if not r:
        return "BY ACCIDENT"
    # Order matters: CONFIG is tested before HARDWARE because reasons like
    # "requires SWA-only config with node size >= cushion" contain no hardware
    # word, but "requires page_size=1 Full+Mamba" would trip \bdevice\b-free
    # hardware patterns in future edits. Narrowest meaning first.
    if BY_ACCIDENT.search(r):
        return "BY ACCIDENT"
    if DEPENDENCY.search(r):
        return "DEPENDENCY"
    if CONFIG.search(r):
        return "CONFIG (not darkness)"
    if HARDWARE.search(r):
        return "HARDWARE"
    return "UNCLASSIFIED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="suite path to illuminate")
    ap.add_argument("--out", default=None)
    ap.add_argument("--collect-only", action="store_true",
                    help="count what EXISTS without running it")
    ap.add_argument("--from-log", default=None,
                    help="re-classify an EXISTING run's log instead of running "
                         "the suite again. Classification is a reading of the "
                         "log, so improving the reading must never cost another "
                         "run of the tests.")
    ap.add_argument("extra", nargs="*", default=[])
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out = Path(args.out or f"/tmp/darkness_{stamp}.log")

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""          # forced, not overridable
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(ROOT / "python")  # derived, never typed
    cmd = [PY, "-m", "pytest", args.path, "-q", "-p", "no:randomly",
           "-p", "no:cacheprovider", "--color=no", "-rsfE"]
    if args.collect_only:
        cmd += ["--collect-only"]
    cmd += args.extra

    print(f"# darkness report {stamp}")
    print(f"# tree     {ROOT}")
    print(f"# commit   {subprocess.getoutput('git -C %s rev-parse --short HEAD' % ROOT)}")
    print(f"# suite    {args.path}")
    print('# hermetic CUDA_VISIBLE_DEVICES="" forced (this number is a CVD="" number '
          'and is NOT comparable to a cards-visible one)')

    if args.from_log:
        out = Path(args.from_log)
        rc, wall = 0, 0.0
        print(f"# RE-CLASSIFIED from {out} -- no tests were run for this report")
    else:
        t0 = time.time()
        with out.open("w") as fh:
            rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=fh,
                                 stderr=subprocess.STDOUT)
        wall = time.time() - t0

    res = parse_log(out)
    text = ANSI.sub("", out.read_text(errors="replace"))

    # ---- skips, with the tally gate the whole report depends on
    buckets: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    where: dict[str, set[str]] = defaultdict(set)
    extracted = 0
    for line in text.splitlines():
        m = SKIP_LINE.match(line.strip())
        if not m:
            continue
        n, path, _lineno, reason = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        extracted += n
        b = classify_skip(reason)
        buckets[b] += n
        by_reason[f"{b}\t{reason.strip()[:110]}"] += n
        where[reason.strip()[:110]].add(path)

    want_skipped = res.counts.get("skipped", 0)
    print(f"\n# wall {wall:.1f}s  rc={rc}  summary={res.counts}")
    print("\n=== tally gate ===")
    ok_f = res.tally_ok
    print(f"  failures: {'OK' if ok_f else 'BROKEN'}  names={len(res.all_names)} "
          f"summary={res.counts.get('failed', 0)} {res.tally_note}")
    ok_s = extracted == want_skipped
    print(f"  skips   : {'OK' if ok_s else 'BROKEN'}  extracted={extracted} "
          f"summary={want_skipped}")
    if not (ok_f and ok_s):
        print("\nINCONCLUSIVE -- the extraction is broken, not the run. A bucket "
              "count drawn from a broken extraction is worse than no count.")
        return 3

    # ---- failures, classified
    print("\n=== failures, by cause ===")
    if not res.all_names:
        print("  (none)")
    dev, real = [], []
    for name in sorted(res.all_names):
        blk = extract_block(text, name)
        (dev if DEVICE_FAIL.search(blk) else real).append((name, first_error_line(blk)))
    print(f"  DEVICE (harness artefact, not a defect): {len(dev)}")
    for n, e in dev:
        print(f"    {n}\n      {e}")
    print(f"  REAL (a defect until shown otherwise): {len(real)}")
    for n, e in real:
        print(f"    {n}\n      {e}")

    # ---- skips, classified
    print(f"\n=== skips, by cause ({want_skipped} total) ===")
    order = ("HARDWARE", "DEPENDENCY", "CONFIG (not darkness)", "BY ACCIDENT",
             "UNCLASSIFIED")
    for b in order:
        print(f"  {b:22s}: {buckets.get(b, 0)}")
    dark = buckets.get("HARDWARE", 0) + buckets.get("DEPENDENCY", 0)
    print(f"\n  DARK (would run elsewhere) : {dark}")
    print(f"  NOT DARK (wrong config)    : {buckets.get('CONFIG (not darkness)', 0)}")
    print("\n=== skip reasons, most tests first ===")
    for key, n in by_reason.most_common():
        b, reason = key.split("\t", 1)
        files = where[reason]
        print(f"  [{n:5d}] {b:12s} {reason}")
        for f in sorted(files)[:3]:
            print(f"                       {f}")
        if len(files) > 3:
            print(f"                       ... and {len(files) - 3} more file(s)")

    if buckets.get("BY ACCIDENT") or buckets.get("UNCLASSIFIED"):
        print("\n=== NOT DARK BY CHOICE ===")
        print("  Skips above that are BY ACCIDENT or UNCLASSIFIED are tests this")
        print("  desk may well be able to run. They are invisible in the summary")
        print("  line, which is the false-green direction: a suite that skips them")
        print("  reads exactly like a suite that passed them.")
    return 0 if rc in (0, 1, 5) else rc


def extract_block(text: str, name: str) -> str:
    """The FAILURES section block for one test id, or the whole text as fallback."""
    short = name.split("::")[-1]
    m = re.search(rf"^_+ .*{re.escape(short)} _+$(.*?)(?=^_{{3,}}|^=+ )", text,
                  re.S | re.M)
    return m.group(1) if m else ""


def first_error_line(block: str) -> str:
    for line in block.splitlines():
        s = line.strip()
        if s.startswith("E "):
            return s[2:].strip()[:200]
    return "(no E-line found in the failure block)"


if __name__ == "__main__":
    raise SystemExit(main())
