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
        #: (group, rank) pairs that produced a USABLE dump for THIS boot.
        self.reported: Set[Tuple[str, int]] = set()
        #: group -> how many ranks the BOOT said would report
        self.expect: Dict[str, int] = {}

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


def modules_for(group: str) -> List[str]:
    """The allowlist half that this dump's PROCESS could possibly have run.

    Applying all eleven to every dump produced ~16 refusals per boot that mean
    nothing by construction (review S-2): `launcher.py` is never imported in a
    rank, and the ten rank modules are not imported in the launcher. The one
    refusal that would have meant something -- a rank that never wrote -- was
    not printed at all. Noise up, signal down; both halves fixed here.
    """
    return list(lc.LAUNCHER_MODULES if group == lc.LAUNCHER_GROUP else lc.RANK_MODULES)


def parse_expect(spec: str) -> Dict[str, int]:
    """``"P=3,D=3,L=1"`` -> ``{"P": 3, "D": 3, "L": 1}``."""
    out: Dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        group, _, n = part.partition("=")
        out[group.strip()] = int(n)
    return out


def process_dump(rep: Report, path: str, boot_token: str) -> None:
    blob, reason = read_dump(path)
    base = os.path.basename(path)
    if blob is None:
        for rel in lc.ALLOWLIST:
            rep.no_observation(rel, "?", "", f"{reason} file={base}")
        return

    group = str(blob.get("group", ""))
    rank = str(blob.get("rank", "?"))
    dump_token = str(blob.get("boot_token", ""))

    # WHICH BOOT WROTE THIS. Refused by NAME, never skipped quietly: a dump
    # left in the (reused) evidence directory by an earlier boot is the thing
    # that fills the silence about a rank that did not report today.
    if dump_token != boot_token:
        rep.refuse(
            f"WEG2-COVERAGE STALE-DUMP file={base} group={group or '?'} rank={rank} "
            f"dump_token={dump_token or '<none>'} boot_token={boot_token} -- this "
            f"dump belongs to a different boot and is NOT counted as this "
            f"boot's reading for that rank; delete it or point --dump-dir at a "
            f"fresh directory"
        )
        return

    legs = blob.get("legs", "?")
    overhead = blob.get("overhead_ms", {})
    saves = overhead.get("saves") or 0
    total = overhead.get("save_total")
    per_leg = round(total / saves, 3) if (saves and isinstance(total, (int, float))) else "?"
    rep.say(
        f"WEG2-COVERAGE DUMP file={base} group={group or '?'} rank={rank} "
        f"legs={legs} instrument={blob.get('instrument', '?')} "
        f"arm_ms={overhead.get('arm', '?')} save_total_ms={total} "
        f"saves={saves} save_per_leg_ms={per_leg} "
        f"traced_ms={overhead.get('traced', '?')} timings=CONFOUNDED "
        f"threads_at_arm={blob.get('threads_at_arm', '?')} "
        f"written_at={blob.get('written_at', '?')}"
    )
    try:
        rep.reported.add((group, int(rank)))
    except (TypeError, ValueError):
        pass

    # A COLLECTOR THAT DIED LEAVES A COMPLETE-LOOKING DUMP: it parses, the
    # modules are present, the tally holds, and every line the legs after the
    # death executed is printed as a wall.
    if blob.get("dead"):
        rep.no_observation(
            "<all>", rank, group,
            f"collector-died where={blob.get('dead_where') or '?'} legs_completed={legs}",
        )
        return

    modules = blob.get("modules") or {}

    for rel in modules_for(group):
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
        if not dump_sha:
            # NOT "no drift". An empty hash was falsy at the old
            # `if dump_sha and ...`, so a file the booting process could not
            # read switched the source guard OFF where nobody could see it --
            # and the line numbers below would refer to a tree this ingest
            # never checked.
            rep.no_observation(rel, rank, group, "no-source-hash")
            continue
        if dump_sha != tree_sha:
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

        # THE GUARANTEED ARTEFACTS, NAMED RATHER THAN MIXED IN (review S-1).
        # If the module was already imported when the tracer went in, its
        # module-scope lines ran unobserved and appear in `unexecuted` as an
        # artefact -- 2104 of 8892 lines (23.7 %) across the allowlist, and on
        # a real boot 10 of 11 modules carry the flag. The old line told the
        # reader to subtract "exactly those lines" and did not show which they
        # were. They are NOT removed from the count: a module imported after
        # the arm has the same lines genuinely unexecuted, and this tool
        # cannot tell the two apart -- so it prints both numbers and says so.
        artefacts: set = set()
        if entry.get("imported_before_arm"):
            artefacts = lc.module_scope_lines(src) & unexecuted
        pct = 100.0 * len(executed) / len(executable)
        rep.say(
            f"WEG2-COVERAGE UNEXECUTED module={rel} rank={rank} "
            f"lines=[{compress(sorted(unexecuted))}] executed_pct={pct:.1f} "
            f"group={group or '?'} executed={len(executed)} "
            f"unexecuted_n={len(unexecuted)} executable={len(executable)} "
            f"off_statement={off_statement} "
            f"imported_before_arm={int(bool(entry.get('imported_before_arm')))} "
            f"artefact_n={len(artefacts)} artefact_lines=[{compress(sorted(artefacts))}]"
        )
        rep.per_group[rel][group].append(unexecuted)


def emit_unions(rep: Report) -> None:
    """Lines no rank of the group reached -- or a refusal if a rank is missing.

    AN INTERSECTION OVER A SUBSET OF THE RANKS IS A SUPERSET, and BOOT_QUEUE
    calls this line "the work list for the next wall". Over 2 of 3 ranks it
    contains every line the absent rank DID execute. The old code shrank
    quietly to `ranks=2` and printed the list anyway; the only tell was a
    number in the middle of the line. It refuses now.
    """
    for rel in lc.ALLOWLIST:
        for group, per_rank in sorted(rep.per_group.get(rel, {}).items()):
            if not per_rank:
                continue
            want = rep.expect.get(group)
            if want is not None and len(per_rank) < want:
                rep.no_observation(
                    rel, "*", group,
                    f"union-incomplete-group ranks={len(per_rank)}/{want}",
                )
                rep.say(
                    f"WEG2-COVERAGE UNION module={rel} group={group or '?'} "
                    f"ranks={len(per_rank)}/{want} REFUSED "
                    f"reason=incomplete-group -- an intersection over a subset "
                    f"of the ranks is a SUPERSET of the true union and would "
                    f"name lines the absent rank executed"
                )
                continue
            union = set.intersection(*per_rank)
            rep.say(
                f"WEG2-COVERAGE UNION module={rel} group={group or '?'} "
                f"ranks={len(per_rank)}/{want if want is not None else len(per_rank)} "
                f"lines=[{compress(sorted(union))}] unexecuted_on_all={len(union)}"
            )


def emit_missing_pairs(rep: Report) -> None:
    """Every (group, rank) the BOOT expected that produced no usable dump.

    THE LINE THE FIRST VERSION COULD NOT PRINT. It iterated the dumps the glob
    found, so a rank -- or a whole group -- that never armed was absent from
    the report rather than named in it, and the reader took the surviving
    group's reading list for the boot's. That is the #1329 shape, which is the
    shape this instrument exists to catch.
    """
    for group in sorted(rep.expect):
        want = rep.expect[group]
        for rank in range(want):
            if (group, rank) in rep.reported:
                continue
            rep.no_observation(
                "<all>", str(rank), group,
                f"rank-wrote-no-dump expected_by=boot-manifest modules="
                f"{len(modules_for(group))}",
            )
            # AN EXPECTED RANK THAT DID NOT REPORT IS AN ACCEPTANCE BREAK, not
            # a reading nuance, so it escalates the EXIT CODE without --strict
            # (review N-1: a caller that reads only `$?` used to read green
            # over a boot in which nothing was observed at all). Every other
            # NO-OBSERVATION stays rc=0 unless --strict, because those are
            # findings about a rank that DID report.
            rep.rc = max(rep.rc, 1)


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
        "--expect",
        default="",
        help="which (group, rank) pairs MUST report, e.g. 'P=3,D=3,L=1'. "
        "Overrides the launcher's " + lc.EXPECT_FILENAME + " manifest in the "
        "dump dir. Without either, this tool REFUSES: a report derived from "
        "the dumps it found can never notice a rank that wrote nothing, which "
        "is the failure this instrument exists to catch.",
    )
    ap.add_argument(
        "--boot-token",
        default="",
        help="the boot whose dumps to read. Overrides the manifest's token. "
        "Dumps carrying a different token are refused BY NAME, never silently "
        "skipped -- the dump directory is shared with #1292's and is reused "
        "across boots.",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit 1 if ANY module was not observed (default: a NO-OBSERVATION "
        "line is a reported finding, not a tool error, and exits 0)",
    )
    ns = ap.parse_args(argv)

    rep = Report(root=os.path.abspath(ns.root), strict=ns.strict)

    # THE EXPECTATION COMES FROM THE BOOT, never from the dumps.
    manifest = {}
    mpath = os.path.join(ns.dump_dir, lc.EXPECT_FILENAME)
    if os.path.isfile(mpath):
        try:
            manifest = json.loads(open(mpath, encoding="utf-8").read())
        except (OSError, json.JSONDecodeError):
            manifest = {}
    rep.expect = parse_expect(ns.expect) if ns.expect else {
        str(k): int(v) for k, v in (manifest.get("expect") or {}).items()
    }
    boot_token = ns.boot_token or str(manifest.get("boot_token") or "")

    if not rep.expect:
        rep.no_observation(
            "<all>", "?", "",
            f"no-expectation-list dir={ns.dump_dir} -- pass --expect or let the "
            f"launcher write {lc.EXPECT_FILENAME}; without it this tool cannot "
            f"tell 'no such rank' from 'this rank reported nothing', and a "
            f"silent report over an empty directory reads exactly like a clean "
            f"sweep",
        )
        rep.rc = max(rep.rc, 1)

    if not os.path.isdir(ns.dump_dir):
        rep.no_observation("<all>", "?", "", f"dump-dir-absent dir={ns.dump_dir}")
        emit_missing_pairs(rep)
        print("\n".join(rep.out))
        return rep.rc

    dumps = sorted(glob.glob(os.path.join(ns.dump_dir, "phase_coverage_*rank*.json")))
    if not dumps:
        # MR1: this branch used to be `if not dumps: <refuse>` but the refusal
        # only covered the ABSENT directory; an EMPTY one walked the loop zero
        # times and printed nothing at all -- which is what a reader sees when
        # no rank ever armed, i.e. the #1329 shape again.
        rep.no_observation("<all>", "?", "", f"no-dump-files-in dir={ns.dump_dir}")

    for path in dumps:
        process_dump(rep, path, boot_token)
    emit_missing_pairs(rep)
    emit_unions(rep)
    print("\n".join(rep.out))
    return rep.rc


if __name__ == "__main__":
    raise SystemExit(main())
