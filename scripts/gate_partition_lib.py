"""#868 -- shared parsing for the partitioned tier-2 gate.

Everything here exists to make ONE comparison trustworthy: the failure set of
a module measured solo versus the same module's failure set inside the full
serial run.  A set difference is only as good as the extraction that produced
it, so the extraction carries its own gate:

    the number of names pulled out of a log MUST equal the number the log's
    own summary reports.  If it does not, the EXTRACTION is broken, not the
    run, and the caller must refuse to draw a conclusion.

Known traps this encodes (each one has cost a wrong verdict before):
  * ANSI colour codes sit in front of ``FAILED`` and break ``^FAILED``;
  * parametrised subtests emit ``SUBFAILED``, not ``FAILED``;
  * the summary line is NOT the last line -- teardown/atexit output pushes it
    up -- so it is found by PATTERN, never by position;
  * #1207, all three measured on the gate run of 2026-09-04, which reported
    ``names failed=111 vs summary failed=113; names error=5 vs summary
    error=11`` and left every WEG-1 slice unmeasurable:
      - a subtest's DESCRIPTION is glued to the keyword with no separator, so
        ``SUBFAILED(abandon="'no_quorum'")`` is not ``SUBFAILED`` + space;
      - the PRODUCT's own logger writes ``ERROR    sglang.srt...`` in column
        0, which a keyword match reads as a test name;
      - the tally compared SET SIZES against pytest's OCCURRENCE COUNTS, so
        eight collection errors for one module (one per xdist worker) counted
        as one.
  * #1207 again, and this one has no arithmetic at all: a lane whose pytest
    run was INTERRUPTED during collection tallies PERFECTLY while not one of
    its tests ran.  No count can see that, so it is read off the log's own
    marker instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Short-summary lines, KEYWORD ONLY. SUBFAILED comes from parametrised
# subtests; ERROR covers both collection errors (``ERROR path``) and fixture
# errors (``ERROR path::test - msg``).
#
# The name is deliberately NOT part of this pattern. pytest builds the line as
# ``f"{verbose_word} {node}"`` (``_pytest/terminal.py:1552``) and
# pytest-subtests makes the word ``f"SUBFAILED{description}"``
# (``_pytest/subtests.py:397``), where the description is arbitrary user text
# that may contain spaces and brackets (``_pytest/subtests.py:88-97`` joins an
# optional ``[msg]`` with an optional ``(k=v, k2=v2)``). A field-position rule
# therefore cannot find the name on every real line; a SHAPE rule can.
NAME_LINE = re.compile(r"^(FAILED|SUBFAILED|ERROR)\b")

# A pytest node id: a path ending in ``.py``, optionally followed by
# ``::``-separated parts.
#
# THE HAZARD THIS SHAPE GUARDS: the product's own logger emits
# ``ERROR    sglang.srt.managers.phase_flip_runtime:phase_flip_runtime.py:8289
# PHASE-FLIP SEAM UNFUNDABLE ...`` at column 0, which is indistinguishable
# from pytest's summary by keyword alone -- seven such lines in the 2026-09-04
# run, every one taken as a test name. A ``file.py:LINE`` reference is not a
# node id, because there ``.py`` is followed by ONE colon, and no path
# component of a node id may contain a colon at all.
NODE_ID = re.compile(r"^[^\s:]+(?:/[^\s:]+)*\.py(?:::\S*)?$")

# ``__ ERROR collecting registered/unit/managers/test_x.py ___``. The module
# was never imported, so none of its tests ran. The captured path is relative
# to pytest's ROOTDIR, not to the tree root, so it is reported as the log
# gives it and never silently joined onto a tree path.
COLLECT_BANNER = re.compile(r"^_+ ERROR collecting (\S+) _+$")

# ``!!!! Interrupted: 1 error during collection !!!!``. pytest gave up before
# running anything it had already collected.
INTERRUPTED_LINE = re.compile(r"^!+ Interrupted: .*!+$")

# The trailing counts line, e.g.
#   "15 failed, 4111 passed, 18 skipped in 717.49s (0:11:57)"
# possibly wrapped in '=' when not under -q. Matched by pattern anywhere in
# the file, and the LAST match wins (xdist prints an inner summary first).
COUNT_TOKEN = re.compile(r"(\d+)\s+(failed|passed|skipped|errors?|xfailed|xpassed|deselected|warnings?)\b")
SUMMARY_LINE = re.compile(r"^=*\s*\d+\s+\w+.*\bin\s+[\d.]+s")
WALL_LINE = re.compile(r"\bin\s+([\d.]+)s")


@dataclass
class RunResult:
    path: str
    failures: set[str] = field(default_factory=set)
    errors: set[str] = field(default_factory=set)
    counts: dict[str, int] = field(default_factory=dict)
    wall: float | None = None
    rc: int | None = None
    solo_wall: float | None = None
    summary_found: bool = False
    tally_ok: bool = False
    tally_note: str = ""
    collected_nothing: bool = False
    #: Occurrences, not distinct names. pytest's summary counts REPORTS, and
    #: one node id can be reported many times -- one collection error per
    #: xdist worker, one SUBFAILED per failing subtest of one test. The sets
    #: above stay sets because the verdict is set arithmetic; the tally is
    #: measured against these.
    failure_lines: int = 0
    error_lines: int = 0
    #: Lines carrying a summary KEYWORD but no node id -- the product's own
    #: logger, whose level names collide with pytest's. Counted rather than
    #: dropped, so a filter that ever grew too wide is visible.
    unnamed_lines: int = 0
    #: Modules pytest could not import, as the log's own banner names them.
    collection_errors: set[str] = field(default_factory=set)
    #: pytest stopped the run during collection: nothing this lane collected
    #: was executed, however healthy the tally looks.
    interrupted: bool = False

    @property
    def all_names(self) -> set[str]:
        return self.failures | self.errors


def node_id_on(line: str) -> str | None:
    """The pytest node id on a short-summary line, found by SHAPE.

    The FIRST token that is a node id wins, because pytest writes
    ``{word} {node}[ - message]`` and the message is the only other part that
    can hold one.

    THE BOUND, named rather than hidden: a subtest description carrying a bare
    ``something.py`` token would be taken instead of the real name. The tally
    is two-sided, so a wrong COUNT is refused in either direction; a wrong
    NAME with a right count is not. No description in this tree has that
    shape, and one that grew it would be a defect in the test, not here.
    """
    for tok in line.split():
        if NODE_ID.match(tok):
            return tok
    return None


def parse_log(path: str | Path) -> RunResult:
    p = Path(path)
    text = ANSI.sub("", p.read_text(errors="replace"))
    res = RunResult(path=str(p))

    summary = None
    for line in text.splitlines():
        line = line.rstrip()
        stripped = line.strip()
        if INTERRUPTED_LINE.match(stripped):
            res.interrupted = True
            continue
        m = COLLECT_BANNER.match(stripped)
        if m:
            res.collection_errors.add(m.group(1))
            continue
        m = NAME_LINE.match(line)
        if m:
            name = node_id_on(line)
            if name is None:
                res.unnamed_lines += 1
                continue
            if m.group(1) == "ERROR":
                res.errors.add(name)
                res.error_lines += 1
            else:
                res.failures.add(name)
                res.failure_lines += 1
            continue
        if SUMMARY_LINE.match(line.strip()):
            summary = line.strip()
        if line.startswith("#SOLO_RC "):
            res.rc = int(line.split()[1])
        elif line.startswith("#SOLO_WALL "):
            res.solo_wall = float(line.split()[1])

    if summary is not None:
        res.summary_found = True
        for n, kind in COUNT_TOKEN.findall(summary):
            key = kind.rstrip("s") if kind in ("errors", "error", "warnings", "warning") else kind
            key = {"error": "error", "warning": "warning"}.get(key, key)
            res.counts[key] = int(n)
        w = WALL_LINE.search(summary)
        if w:
            res.wall = float(w.group(1))
    elif re.search(r"no tests ran", text):
        res.summary_found = True
        res.collected_nothing = True

    # THE TALLY GATE. Names extracted must equal names counted -- and both
    # sides must count the same THING, which before #1207 they did not: the
    # left was a set of distinct ids, the right pytest's count of reports.
    want_f = res.counts.get("failed", 0)
    want_e = res.counts.get("error", 0)
    got_f, got_e = res.failure_lines, res.error_lines
    if not res.summary_found:
        res.tally_note = "no summary line found in log"
    elif got_f != want_f or got_e != want_e:
        res.tally_note = (
            f"extraction mismatch: names failed={got_f} vs summary failed={want_f}; "
            f"names error={got_e} vs summary error={want_e}"
            + (f"; {res.unnamed_lines} keyword line(s) carried no node id"
               if res.unnamed_lines else "")
        )
    else:
        res.tally_ok = True
    return res


def module_of(test_id: str) -> str:
    """``a/b/test_x.py::C::t`` -> ``a/b/test_x.py`` (also handles bare paths)."""
    return test_id.split("::", 1)[0]


def by_module(names: set[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for n in names:
        out.setdefault(module_of(n), set()).add(n)
    return out
