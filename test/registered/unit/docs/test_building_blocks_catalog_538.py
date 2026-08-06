"""Section 18 of the feature catalog must point at code that exists (#538).

The catalog's building-block section is only useful if its ``file:line``
pointers resolve. A module that is deleted, renamed or shortened past a cited
line makes this test fail instead of leaving a dead pointer in the catalog.

Deliberately cheap: no imports of the target modules, no torch, no GPU. It
resolves paths and counts lines, nothing else -- the checks themselves run in
well under a tenth of a second, and the only real cost is the shared ``sglang``
import that CI registration requires. It does NOT check that the
cited line still holds the cited symbol -- that would need a parser per
language and would fail on every unrelated insertion above; the line number is
a reading aid, the path is the contract, and a path that has lost the line
entirely is the failure worth catching.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
CATALOG = REPO_ROOT / "docs" / "dev" / "FEATURE_CATALOG.md"
SRT = REPO_ROOT / "python" / "sglang" / "srt"

#: Paths in section 18 are relative to ``python/sglang/srt`` unless they start
#: with one of these repo-root-relative prefixes.
ROOT_RELATIVE_PREFIXES = ("tests/", "test/", "docs/", "scripts/", "python/")

#: ``path.py:123`` inside backticks. The path may contain directories.
POINTER = re.compile(r"`([A-Za-z0-9_./-]+\.py):(\d+)`")

SECTION_HEADING = "## 18. Reusable building blocks"


def _section_18() -> str:
    text = CATALOG.read_text(encoding="utf-8")
    start = text.find(SECTION_HEADING)
    assert start != -1, f"{CATALOG} has no section 18 heading {SECTION_HEADING!r}"
    rest = text[start + len(SECTION_HEADING) :]
    nxt = re.search(r"^## ", rest, flags=re.MULTILINE)
    return rest[: nxt.start()] if nxt else rest


def _resolve(rel: str) -> Path:
    if rel.startswith(ROOT_RELATIVE_PREFIXES):
        return REPO_ROOT / rel
    return SRT / rel


def _pointers() -> list[tuple[str, int]]:
    return [(m.group(1), int(m.group(2))) for m in POINTER.finditer(_section_18())]


def test_section_18_has_pointers():
    """A section 18 that cites nothing is a section 18 that documents nothing."""
    pointers = _pointers()
    assert len(pointers) >= 60, f"section 18 cites only {len(pointers)} file:line pointers"


@pytest.mark.parametrize("rel,line", _pointers(), ids=lambda v: str(v))
def test_section_18_pointer_resolves(rel: str, line: int):
    path = _resolve(rel)
    assert path.is_file(), (
        f"FEATURE_CATALOG.md section 18 cites `{rel}:{line}`, which does not exist. "
        f"Looked at {path}. A module named in section 18 was moved or deleted "
        f"without updating its entry (catalog rule 5)."
    )
    count = sum(1 for _ in path.open("rb"))
    assert line <= count, (
        f"FEATURE_CATALOG.md section 18 cites `{rel}:{line}` but {rel} has only "
        f"{count} lines. The entry point moved; re-read it and correct the entry."
    )
