"""Section 18 of the feature catalog must point at code that exists (#538),
and at the SYMBOL it names (#826, shipped #1034b).

WHAT THIS USED TO CHECK, AND WHY THAT WAS NOT ENOUGH
----------------------------------------------------
Until #1034b this asserted exactly two things per anchor: the file exists and
``line <= len(file)``. That is LINE EXISTENCE. It catches a deleted module and
is blind to the failure that actually happens -- a line number that still
exists and no longer means anything.

Measured (#826): W6 added 37 lines to ``funding_authority.py``; this test
stayed green while three citations slid onto a comment, an ``if`` and a
dataclass field. Measured again 2026-08-30 with the checker this test is now
modelled on: 23 of 53 symbol-bearing anchors had drifted and one pointed at a
symbol that had moved file entirely -- all of them green under line existence.

A catalog whose §18 answers "is there already a module I should be calling" is
worth exactly as much as its pointers. A pointer that lands on a comment reads
like an answer, which is why nobody re-checks it.

TWO ORTHOGONAL QUESTIONS, NOT ONE
---------------------------------
  (1) IS THE SYMBOL STILL THERE?  Resolved BY NAME, so a symbol that merely
      MOVED is FOUND, not reported missing. A checker keyed on the line would
      call a moved-but-present building block absent and send someone to
      rebuild what already exists -- the exact incident class §18 was written
      to stop.
  (2) IS THE ANCHOR ACCURATE?  The cited line must be where the symbol is
      DEFINED. Drift here is RED: it is the silent-wrong-citation class.
A moved symbol is therefore GREEN on (1) and RED on (2), and the failure
message prints the corrected line so fixing it is one edit, not an
investigation.

HONEST COVERAGE -- the denominator is printed, not hidden
---------------------------------------------------------
Not every anchor names a symbol. A CONSUMERS list cites a CALL SITE, and the
catalog's syntax for "defined here" is a symbol in backticks IMMEDIATELY after
the pointer; a call-site citation therefore names its symbol in prose instead.
Those pointers keep the line-existence check, because nothing better is
derivable from them, and they are counted SEPARATELY. Reporting "103 anchors
checked" while only half were symbol-checked would be the denominator trap.

DELIBERATE DUPLICATION. ``/spinning/gpu-arb/devtools/catalog_anchor_check.py``
is the same check as a standalone tool and is not importable from the test tree
(it lives outside the repo). The regexes and the resolver here are kept
character-identical to it so the two cannot disagree about what an anchor IS.

Still cheap: no imports of the target modules, no torch, no GPU -- ``ast.parse``
over the cited files only.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

REPO_ROOT = Path(__file__).resolve().parents[4]
CATALOG = REPO_ROOT / "docs" / "dev" / "FEATURE_CATALOG.md"
SRT = REPO_ROOT / "python" / "sglang" / "srt"

#: Paths in section 18 are relative to ``python/sglang/srt`` unless they start
#: with one of these repo-root-relative prefixes.
ROOT_RELATIVE_PREFIXES = ("tests/", "test/", "docs/", "scripts/", "python/")

#: ``path.py:123`` inside backticks. The path may contain directories.
POINTER = re.compile(r"`([A-Za-z0-9_./-]+\.py):(\d+)`")

#: A symbol named next to the anchor: `` `name()` ``, `` `name` `` or
#: `` `Class.method()` ``. Only the window immediately after the anchor is
#: searched -- prose further along the line belongs to a different claim.
SYMBOL_NEAR = re.compile(r"^[\s,]*\(?`([A-Za-z_][A-Za-z0-9_.]*)\(?\)?`")

SYMBOL_WINDOW = 60

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


def _anchors() -> list[tuple[str, int, str | None]]:
    """``[(rel_path, line, symbol_or_None)]`` for every §18 pointer."""
    sec = _section_18()
    out = []
    for m in POINTER.finditer(sec):
        tail = sec[m.end() : m.end() + SYMBOL_WINDOW]
        sm = SYMBOL_NEAR.match(tail)
        out.append((m.group(1), int(m.group(2)), sm.group(1) if sm else None))
    return out


def _assign_targets(node: ast.AST):
    if isinstance(node, ast.Assign):
        for t in node.targets:
            if isinstance(t, ast.Name):
                yield t.id
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        yield node.target.id


def definition_lines(source: str, symbol: str) -> set[int] | None:
    """Lines where ``symbol`` is DEFINED: def, async def, class, or assignment.

    Returns ``None`` when the file cannot be parsed -- undecidable is its own
    answer and is never folded into "absent", which would invent drift out of a
    syntax error.

    A dotted ``Class.method`` resolves to the method inside that class, so an
    anchor may cite either the class or one of its methods. A decorated
    definition accepts either the decorator line or the ``def`` line, because a
    citation may reasonably point at either.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    head, _, tail = symbol.partition(".")
    lines: set[int] = set()

    def scan(node, want, inside=None):
        for child in ast.iter_child_nodes(node):
            name = getattr(child, "name", None)
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if name == want and (inside is None or inside):
                    lines.add(child.lineno)
                    for dec in getattr(child, "decorator_list", []):
                        lines.add(dec.lineno)
                if isinstance(child, ast.ClassDef) and tail and name == head:
                    scan(child, tail, inside=True)
                elif not tail:
                    scan(child, want, inside)
            else:
                if not tail:
                    for t in _assign_targets(child):
                        if t == want:
                            lines.add(child.lineno)
                    scan(child, want, inside)

    scan(tree, head)
    if tail:
        lines = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == head:
                for child in ast.iter_child_nodes(node):
                    if getattr(child, "name", None) == tail:
                        lines.add(child.lineno)
                        for dec in getattr(child, "decorator_list", []):
                            lines.add(dec.lineno)
    return lines


_ANCHORS = _anchors()
_SYMBOL_ANCHORS = [a for a in _ANCHORS if a[2] is not None]
_BARE_ANCHORS = [a for a in _ANCHORS if a[2] is None]


def test_section_18_has_pointers():
    """A section 18 that cites nothing is a section 18 that documents nothing."""
    assert len(_ANCHORS) >= 60, f"section 18 cites only {len(_ANCHORS)} file:line pointers"


def test_symbol_checked_share_is_not_silently_zero():
    """The denominator is part of the result.

    If a catalog edit stopped putting symbols next to anchors, every anchor
    would fall back to line existence and this suite would stay green while
    checking nothing. That silent downgrade is the failure mode this asserts
    against.
    """
    assert len(_SYMBOL_ANCHORS) >= 40, (
        f"only {len(_SYMBOL_ANCHORS)} of {len(_ANCHORS)} §18 anchors name a symbol; "
        f"the symbol check has been silently downgraded to line existence"
    )


@pytest.mark.parametrize("rel,line,symbol", _SYMBOL_ANCHORS, ids=lambda v: str(v))
def test_section_18_anchor_resolves_to_its_symbol(rel: str, line: int, symbol: str):
    path = _resolve(rel)
    assert path.is_file(), (
        f"FEATURE_CATALOG.md section 18 cites `{rel}:{line}` (`{symbol}`), which does "
        f"not exist. Looked at {path}. A module named in section 18 was moved or "
        f"deleted without updating its entry (catalog rule 5)."
    )
    lines = definition_lines(path.read_text(encoding="utf-8", errors="replace"), symbol)
    assert lines is not None, f"{rel} does not parse; cannot check `{symbol}`"
    assert lines, (
        f"FEATURE_CATALOG.md section 18 cites `{rel}:{line}` (`{symbol}`), but "
        f"`{symbol}` is not defined anywhere in {rel}. Either the building block "
        f"moved file -- find it and re-point the anchor, do NOT rebuild it -- or "
        f"the entry names a CALL SITE, in which case name the symbol in prose "
        f"rather than in backticks right after the pointer (#1034b)."
    )
    assert line in lines, (
        f"FEATURE_CATALOG.md section 18 cites `{rel}:{line}` (`{symbol}`), but "
        f"`{symbol}` is defined at {sorted(lines)}. The symbol is PRESENT and the "
        f"anchor is STALE: correct the entry to `{rel}:{max(lines)}`. A line that "
        f"still exists but no longer holds its symbol reads like an answer, which "
        f"is why this is red rather than a warning (#826)."
    )


@pytest.mark.parametrize("rel,line,symbol", _BARE_ANCHORS, ids=lambda v: str(v))
def test_section_18_bare_pointer_resolves(rel: str, line: int, symbol: None):
    """Pointers with no adjacent symbol keep the old line-existence check.

    Nothing better is derivable from a bare citation. Counted separately so the
    coverage of the symbol check above is never overstated.
    """
    path = _resolve(rel)
    assert path.is_file(), (
        f"FEATURE_CATALOG.md section 18 cites `{rel}:{line}`, which does not exist. "
        f"Looked at {path}."
    )
    count = sum(1 for _ in path.open("rb"))
    assert line <= count, (
        f"FEATURE_CATALOG.md section 18 cites `{rel}:{line}` but {rel} has only "
        f"{count} lines. The entry point moved; re-read it and correct the entry."
    )


# --------------------------------------------------------------------------
# The two properties, on synthetic sources: this is what makes the check above
# a check rather than a green light. Both were RED against the real catalog on
# 2026-08-30 (23 drifted, 1 moved-file) and are the reason #1034b exists.
# --------------------------------------------------------------------------

_SYNTH_BEFORE = "def alpha():\n    pass\n\n\ndef beta():\n    pass\n"
_SYNTH_AFTER = "# a new comment\n# and another\ndef alpha():\n    pass\n\n\ndef beta():\n    pass\n"


def test_property_drifted_anchor_is_red():
    """An anchor whose line no longer holds its symbol must NOT resolve."""
    assert definition_lines(_SYNTH_BEFORE, "alpha") == {1}
    # two lines inserted above: the old anchor 1 is now stale, and 1 is not in
    # the definition set -- which is exactly what makes the parametrised test
    # above fail rather than pass on drift.
    assert 1 not in definition_lines(_SYNTH_AFTER, "alpha")


def test_property_moved_but_present_symbol_is_found_by_name():
    """A symbol that merely moved is FOUND, not reported absent.

    The opposite behaviour is the expensive one: it sends someone to rebuild a
    building block that already exists.
    """
    moved = definition_lines(_SYNTH_AFTER, "alpha")
    assert moved == {3}, moved
    assert definition_lines(_SYNTH_AFTER, "gamma") == set()


def test_property_decorated_definition_accepts_either_line():
    src = "import functools\n\n\n@functools.cache\ndef alpha():\n    pass\n"
    assert definition_lines(src, "alpha") == {4, 5}


def test_property_unparsable_file_is_undecidable_not_absent():
    assert definition_lines("def (:\n", "alpha") is None
