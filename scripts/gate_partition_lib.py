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
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

# Short-summary lines. SUBFAILED comes from parametrised subtests; ERROR
# covers both collection errors (``ERROR path``) and fixture errors
# (``ERROR path::test - msg``).
NAME_LINE = re.compile(r"^(FAILED|SUBFAILED|ERROR)\s+(\S+)")

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

    @property
    def all_names(self) -> set[str]:
        return self.failures | self.errors


def parse_log(path: str | Path) -> RunResult:
    p = Path(path)
    text = ANSI.sub("", p.read_text(errors="replace"))
    res = RunResult(path=str(p))

    summary = None
    for line in text.splitlines():
        line = line.rstrip()
        m = NAME_LINE.match(line)
        if m:
            kind, name = m.group(1), m.group(2)
            (res.errors if kind == "ERROR" else res.failures).add(name)
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

    # THE TALLY GATE. Names extracted must equal names counted.
    want_f = res.counts.get("failed", 0)
    want_e = res.counts.get("error", 0)
    got_f, got_e = len(res.failures), len(res.errors)
    if not res.summary_found:
        res.tally_note = "no summary line found in log"
    elif got_f != want_f or got_e != want_e:
        res.tally_note = (
            f"extraction mismatch: names failed={got_f} vs summary failed={want_f}; "
            f"names error={got_e} vs summary error={want_e}"
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
