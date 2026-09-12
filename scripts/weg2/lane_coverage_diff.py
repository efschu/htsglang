#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""#1348 -- read the per-rank lane-coverage dumps and print what did NOT run.

    # 1. boot with the launcher flag (and nothing else changed)
    python -m sglang.srt.weg2.launcher --tag s6i --xchg-coverage-diff ...
    # 2. after the boot, fold the per-rank dumps into the complement
    python scripts/weg2/lane_coverage_diff.py --dump-dir <evidence dir>

WHAT COMES OUT, and why each line is shaped the way it is.

``WEG2-COVERAGE UNEXECUTED module=<path> rank=<n> lines=[a-b,c,...]
executed_pct=<x> ...``
    Per module, per rank: the statement lines this boot never reached.  This
    is the READING LIST -- each entry is a dead branch, an untriggered
    refusal, or the next wall, and only a reader can say which.

``WEG2-COVERAGE UNION module=<path> group=<G> ...``
    The lines unexecuted on EVERY rank of a group.  A line missed by one rank
    and hit by another is a rank-local path (information, and it stays in the
    per-rank lines); a line missed by all of them is a seam nothing in this
    boot reached, which is the stronger finding.

``WEG2-COVERAGE NO-OBSERVATION module=<path> rank=<n> reason=<r>`` + ``W90``
    THE INDICATOR LAW, and the reason half this file exists.  A coverage
    reader that prints "0 unexecuted" when its input is missing hands the
    reader an ABSENCE OF OBSERVATION dressed as a FULL SWEEP, and closes
    exactly the seam it was built to open.  Four ways in: the dump directory
    is not there, the dump file is empty, the module is absent from the dump,
    the module never got imported.  Each prints a NAMED refusal and no number.

``WEG2-COVERAGE TALLY-REFUSED ...`` + ``W91``, exit 2
    ``executed + unexecuted == executable`` is an identity.  Where it fails,
    the numerator and the denominator were not produced by the same analysis
    of the same source, so their difference is meaningless.  Refused, never
    reported.

``WEG2-COVERAGE SOURCE-DRIFT ...``, exit 2
    The dump carries a sha256 per module, taken in the booting process.  Read
    against a different checkout, every line number is off by an unknown
    amount and the output would be confidently wrong.  Same class as the
    codegraph's pin line: an answer without its provenance is not checkable.

TWO BOUNDS THIS TOOL PRINTS RATHER THAN HIDES.

1.  ``imported_before_arm=1``.  The tracer goes in at the lane's first leg,
    not at interpreter start, so a module already imported by then had its
    module-scope lines (imports, ``def``, ``class``, constants) executed
    UNOBSERVED.  They read as unexecuted and are an artefact.  The flag is on
    every line that has it; a reader who ignores it over-reports exactly those
    lines and nothing else.
2.  ``off_statement=N``.  Raw traced events that map to no statement line --
    docstrings, mostly.  coverage.py's own reports absorb these silently; here
    they are counted beside the number, because a percentage whose dropped
    remainder is invisible cannot be argued with.
"""

from __future__ import annotations

import argparse
import ast
import glob
import hashlib
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Set, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_ROOT = os.path.dirname(os.path.dirname(_HERE))
if os.path.isdir(os.path.join(_DEFAULT_ROOT, "python", "sglang")):
    sys.path.insert(0, os.path.join(_DEFAULT_ROOT, "python"))

from sglang.srt.weg2 import lane_coverage as lc  # noqa: E402


def compress(lines: Sequence[int]) -> str:
    """``[41,42,43,67]`` -> ``41-43,67``.

    A seam is a RUN of lines, so a run is what a reader should see; the flat
    list of 400 numbers that the same data prints uncompressed is unreadable
    and, worse, hides the fact that they are contiguous -- which is the single
    strongest hint that they are one unentered branch rather than 400
    independent misses.
    """
    out: List[str] = []
    run_start = run_prev = None
    for n in sorted(lines):
        if run_start is None:
            run_start = run_prev = n
        elif n == run_prev + 1:
            run_prev = n
        else:
            out.append(str(run_start) if run_start == run_prev else f"{run_start}-{run_prev}")
            run_start = run_prev = n
    if run_start is not None:
        out.append(str(run_start) if run_start == run_prev else f"{run_start}-{run_prev}")
    return ",".join(out)


def function_line_ranges(path: str, names: Sequence[str]) -> Set[int]:
    """Every line inside the named top-level functions of ``path``.

    Used for ``launcher.py`` only.  The module runs in the launcher process
    and is 10k lines of which the arm/ledger half is the part whose unentered
    branches are a finding; the rest is argv assembly, and reporting its
    unexecuted lines would bury the eleven modules that matter under one that
    does not.
    """
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    keep: Set[int] = set()
    wanted = set(names)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in wanted:
            keep.update(range(node.lineno, (node.end_lineno or node.lineno) + 1))
    return keep


def sha256_of(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class Report:
    """Accumulates the lines, the refusals, and the exit code."""

    def __init__(self, root: str, strict: bool) -> None:
        self.root = root
        self.strict = strict
        self.out: List[str] = []
        self.rc = 0
        self.saw_no_observation = False
        #: module -> group -> [set of unexecuted lines per rank]
        self.per_group: Dict[str, Dict[str, List[Set[int]]]] = defaultdict(
            lambda: defaultdict(list)
        )

    def say(self, line: str) -> None:
        self.out.append(line)

    def no_observation(self, module: str, rank: str, group: str, reason: str) -> None:
        self.saw_no_observation = True
        self.say(
            f"WEG2-COVERAGE NO-OBSERVATION module={module} rank={rank} "
            f"group={group or '?'} reason={reason} refusal={lc.NO_OBSERVATION_CODE}"
        )
        if self.strict:
            self.rc = max(self.rc, 1)

    def refuse(self, line: str) -> None:
        self.say(line)
        self.rc = 2


def read_dump(path: str) -> Tuple[Optional[dict], str]:
    """``(blob, reason)`` -- a reason instead of an exception on every failure.

    An unreadable dump is an OBSERVATION about the boot (the rank died before
    the first checkpoint, the disk filled, the schema moved), not a crash of
    the reader, and the report has to be able to say which file and why.
    """
    try:
        raw = open(path, encoding="utf-8").read()
    except OSError as e:
        return None, f"unreadable:{type(e).__name__}"
    if not raw.strip():
        return None, "dump-file-empty"
    try:
        blob = json.loads(raw)
    except json.JSONDecodeError:
        return None, "dump-file-malformed"
    if blob.get("schema") != lc.SCHEMA:
        return None, f"schema-mismatch:{blob.get('schema')!r}"
    return blob, ""


def process_dump(rep: Report, path: str) -> None:
    blob, reason = read_dump(path)
    base = os.path.basename(path)
    if blob is None:
        for rel in lc.ALLOWLIST:
            rep.no_observation(rel, "?", "", f"{reason} file={base}")
        return

    group = str(blob.get("group", ""))
    rank = str(blob.get("rank", "?"))
    legs = blob.get("legs", "?")
    overhead = blob.get("overhead_ms", {})
    rep.say(
        f"WEG2-COVERAGE DUMP file={base} group={group or '?'} rank={rank} "
        f"legs={legs} instrument={blob.get('instrument', '?')} "
        f"arm_ms={overhead.get('arm', '?')} save_total_ms={overhead.get('save_total', '?')} "
        f"saves={overhead.get('saves', '?')} written_at={blob.get('written_at', '?')}"
    )
    modules = blob.get("modules") or {}

    for rel in lc.ALLOWLIST:
        entry = modules.get(rel)
        if entry is None:
            rep.no_observation(rel, rank, group, "module-absent-from-dump")
            continue
        src = os.path.join(rep.root, rel)
        if not os.path.isfile(src):
            rep.no_observation(rel, rank, group, "source-absent-from-this-checkout")
            continue
        tree_sha = sha256_of(src)
        dump_sha = entry.get("sha256") or ""
        if dump_sha and dump_sha != tree_sha:
            rep.refuse(
                f"WEG2-COVERAGE SOURCE-DRIFT module={rel} rank={rank} "
                f"dump_sha256={dump_sha[:12]} tree_sha256={tree_sha[:12]} -- the "
                f"dump was taken on a different source than this checkout; every "
                f"line number below would be off by an unknown amount"
            )
            continue
        if not entry.get("imported", False):
            rep.no_observation(rel, rank, group, "module-never-imported")
            continue

        executable = lc.executable_lines(src)
        if rel.endswith("launcher.py"):
            scope = function_line_ranges(src, lc.LAUNCHER_ARM_FUNCTIONS)
            executable &= scope
        if not executable:
            rep.no_observation(rel, rank, group, "no-executable-lines-in-scope")
            continue

        raw = [int(n) for n in entry.get("executed", [])]
        n_source_lines = sum(1 for _ in open(src, encoding="utf-8"))
        stray = [n for n in raw if n < 1 or n > n_source_lines]
        if stray:
            rep.refuse(
                f"WEG2-COVERAGE TALLY-REFUSED module={rel} rank={rank} "
                f"reason=recorded-line-outside-source stray={compress(stray)} "
                f"source_lines={n_source_lines} refusal={lc.TALLY_REFUSED_CODE}"
            )
            continue
        if not raw:
            rep.no_observation(rel, rank, group, "no-lines-recorded")
            continue

        translated = lc.translate(src, raw)
        executed = translated & executable
        off_statement = len(translated - executable)
        if not executed:
            rep.no_observation(rel, rank, group, "no-lines-recorded-in-scope")
            continue
        unexecuted = executable - executed

        # THE COUNT CHECK, literally.  It is an identity by construction here,
        # which is the point: if it ever fails, the construction changed and
        # the numbers stopped being comparable.
        if len(executed) + len(unexecuted) != len(executable):
            rep.refuse(
                f"WEG2-COVERAGE TALLY-REFUSED module={rel} rank={rank} "
                f"executed={len(executed)} unexecuted={len(unexecuted)} "
                f"executable={len(executable)} refusal={lc.TALLY_REFUSED_CODE}"
            )
            continue

        pct = 100.0 * len(executed) / len(executable)
        rep.say(
            f"WEG2-COVERAGE UNEXECUTED module={rel} rank={rank} "
            f"lines=[{compress(sorted(unexecuted))}] executed_pct={pct:.1f} "
            f"group={group or '?'} executed={len(executed)} "
            f"unexecuted_n={len(unexecuted)} executable={len(executable)} "
            f"off_statement={off_statement} "
            f"imported_before_arm={int(bool(entry.get('imported_before_arm')))}"
        )
        rep.per_group[rel][group].append(unexecuted)


def emit_unions(rep: Report) -> None:
    for rel in lc.ALLOWLIST:
        for group, per_rank in sorted(rep.per_group.get(rel, {}).items()):
            if not per_rank:
                continue
            union = set.intersection(*per_rank)
            rep.say(
                f"WEG2-COVERAGE UNION module={rel} group={group or '?'} "
                f"ranks={len(per_rank)} lines=[{compress(sorted(union))}] "
                f"unexecuted_on_all={len(union)}"
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "#1348: per-rank UNEXECUTED lines of the Weg-2 exchange lane. "
            "Reads the dumps written by sglang.srt.weg2.lane_coverage when the "
            "launcher was given --xchg-coverage-diff."
        )
    )
    ap.add_argument(
        "--dump-dir",
        required=True,
        help="directory holding phase_coverage_{GROUP}_rank{N}.json (the same "
        "dump directory #1292's footprint probe writes into)",
    )
    ap.add_argument(
        "--root",
        default=_DEFAULT_ROOT,
        help="repo root the dumps' line numbers refer to; the per-module "
        "sha256 in each dump is checked against it (default: this checkout)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if ANY module was not observed (default: a NO-OBSERVATION "
        "line is a reported finding, not a tool error, and exits 0)",
    )
    ns = ap.parse_args(argv)

    rep = Report(root=os.path.abspath(ns.root), strict=ns.strict)
    if not os.path.isdir(ns.dump_dir):
        for rel in lc.ALLOWLIST:
            rep.no_observation(rel, "?", "", f"dump-dir-absent dir={ns.dump_dir}")
        print("\n".join(rep.out))
        return rep.rc

    dumps = sorted(glob.glob(os.path.join(ns.dump_dir, "phase_coverage_*rank*.json")))
    if not dumps:
        for rel in lc.ALLOWLIST:
            rep.no_observation(rel, "?", "", f"no-dump-files-in dir={ns.dump_dir}")
        print("\n".join(rep.out))
        return rep.rc

    for path in dumps:
        process_dump(rep, path)
    emit_unions(rep)
    print("\n".join(rep.out))
    return rep.rc


if __name__ == "__main__":
    raise SystemExit(main())
