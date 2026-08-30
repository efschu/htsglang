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
    up -- so it is found by PATTERN, never by position.
  * #1034a, three roots measured against .gate1029/{wide,narrow,serial}.log:
    (a) a parametrised subtest prints its params BEFORE the whitespace --
        ``SUBFAILED(abandon="'no_quorum'") test/...`` -- so a pattern that
        demands ``\\s`` right after the keyword drops it silently;
    (b) the tally compared a set of unique NAMES against the summary's count
        of EVENTS.  Two SUBFAILED subtests of ONE test are two events and one
        name, and eight fixture ERRORs of one module are eight events and one
        name (in wide.log they are even byte-identical lines).  Counting names
        against events mismatches on every log that has either;
    (c) ``^ERROR`` also matches the APPLICATION's own log records --
        ``ERROR    sglang.srt.managers.phase_flip_runtime:...:7750 PHASE-FLIP
        SEAM UNFUNDABLE`` -- 7 of them in wide.log, which were being reported
        as failing "tests" named after a logger.  The name must therefore be
        required to look like a test path, not merely be non-whitespace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Short-summary lines. SUBFAILED comes from parametrised subtests; ERROR
# covers both collection errors (``ERROR path``) and fixture errors
# (``ERROR path::test - msg``).
#
# The optional ``(...)`` group carries a subtest's parameters and sits BEFORE
# the whitespace.  It is greedy-with-backtracking rather than ``[^)]*`` so a
# nested paren in a param repr (``SUBFAILED(x=(1, 2)) test/...``) still parses;
# the required test-path anchor that follows makes the backtracking safe.
#
# The name must LOOK LIKE A TEST PATH.  Without that anchor the application's
# own ``ERROR <logger>:<file>:<line> ...`` records are harvested as test names.
NAME_LINE = re.compile(
    r"^(FAILED|SUBFAILED|ERROR)(?:\(.*\))?\s+((?:test|tests)/\S*\.py\S*)"
)

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
    # The tally gate compares EVENTS, because that is what the summary counts.
    # ``failures``/``errors`` stay NAME sets -- they are what the set
    # arithmetic downstream (solo vs serial) needs -- but a name set can be
    # smaller than the event count and must never be compared against it.
    failed_events: int = 0
    error_events: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    wall: float | None = None
    rc: int | None = None
    solo_wall: float | None = None
    summary_found: bool = False
    tally_ok: bool = False
    tally_note: str = ""
    collected_nothing: bool = False

    @property
    def all_names(self) -> set[str]:
        return self.failures | self.errors


def parse_log(path: str | Path) -> RunResult:
    p = Path(path)
    text = ANSI.sub("", p.read_text(errors="replace"))
    res = RunResult(path=str(p))

    summary = None
    failed_names: set[str] = set()  # FAILED only -- one summary event each
    subfailed_events = 0            # SUBFAILED -- one event per LINE
    for line in text.splitlines():
        line = line.rstrip()
        m = NAME_LINE.match(line)
        if m:
            kind, name = m.group(1), m.group(2)
            if kind == "ERROR":
                res.errors.add(name)
                res.error_events += 1
            else:
                res.failures.add(name)
                if kind == "SUBFAILED":
                    subfailed_events += 1
                else:
                    failed_names.add(name)
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

    # THE TALLY GATE. EVENTS extracted must equal EVENTS counted -- pytest's
    # summary counts one "failed" per FAILED test plus one per failing
    # subtest, and one "error" per ERROR line.
    res.failed_events = len(failed_names) + subfailed_events
    want_f = res.counts.get("failed", 0)
    want_e = res.counts.get("error", 0)
    got_f, got_e = res.failed_events, res.error_events
    if not res.summary_found:
        res.tally_note = "no summary line found in log"
    elif got_f != want_f or got_e != want_e:
        res.tally_note = (
            f"extraction mismatch: events failed={got_f} vs summary failed={want_f}; "
            f"events error={got_e} vs summary error={want_e}"
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
