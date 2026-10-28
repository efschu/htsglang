"""The launcher source as the source-pinned tests need it, independent of how many files it is split over.

``launcher.py`` keeps ``main`` and the shared core; classes and functions that moved out (behaviour-neutral refactor, 28.10.) live in
``launcher_types.py`` and ``launcher_*.py``. A test that pins source text of a launcher class or function reads it through here:
one merged view, the old names.
"""

import ast
import pathlib

_PDFLIP = pathlib.Path(__file__).resolve().parents[4] / "python" / "flliper" / "srt" / "pdflip"
LAUNCHER_PY = _PDFLIP / "launcher.py"
SPLIT_FILES = sorted(p for p in _PDFLIP.glob("launcher_*.py"))


def source_files():
    return [LAUNCHER_PY, *SPLIT_FILES]


def _segment(text: str, node) -> str:
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    return "\n".join(text.split("\n")[first - 1: node.end_lineno])


def definition_source(name: str) -> str:
    """Source text of the top-level ``class``/``def`` called ``name`` wherever it lives now (decorators included)."""
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        for node in ast.parse(text).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name == name:
                return _segment(text, node)
    raise KeyError(name)


def merged_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in source_files())
