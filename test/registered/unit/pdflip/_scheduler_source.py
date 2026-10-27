"""The ``Scheduler`` source as the source-pinned tests need it, independent of how many files it is split over.

``scheduler.py`` keeps the class and the shared core; methods that moved out (behaviour-neutral refactor, 27.10.) live in
``scheduler_*_mixin.py`` classes and module functions in ``scheduler_pdflip_helpers.py``. A test that pins source text or
AST of a ``Scheduler`` method or of a scheduler module function reads it through here: one merged view, the old names.
"""

import ast
import pathlib

_MANAGERS = pathlib.Path(__file__).resolve().parents[4] / "python" / "flliper" / "srt" / "managers"
SCHEDULER_PY = _MANAGERS / "scheduler.py"
SPLIT_FILES = sorted(_MANAGERS.glob("scheduler_*_mixin.py")) + [_MANAGERS / "scheduler_pdflip_helpers.py"]
# scheduler_pp_mixin.py is an older, separate mixin that tests pin on its own
SPLIT_FILES = [p for p in SPLIT_FILES if p.name != "scheduler_pp_mixin.py" and p.exists()]


def source_files():
    return [SCHEDULER_PY, *SPLIT_FILES]


def method_source(name: str) -> str:
    """Source text of ``Scheduler.<name>`` wherever it lives now (decorators included, no leading comments)."""
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                first = min([node.lineno] + [d.lineno for d in node.decorator_list])
                lines = text.split("\n")[first - 1: node.end_lineno]
                return "\n".join(lines)
    raise KeyError(name)


def file_defining(name: str) -> pathlib.Path:
    """The file that now holds ``def <name>`` (a Scheduler method or a module function)."""
    for path in source_files():
        if f"def {name}(" in path.read_text(encoding="utf-8"):
            return path
    raise KeyError(name)


def class_source() -> str:
    """``class Scheduler`` as it reads today: its own body plus the moved-method mixin classes."""
    import inspect

    from flliper.srt.managers import scheduler as sched

    parts = [inspect.getsource(sched.Scheduler)]
    for cls in sched.Scheduler.__mro__[1:]:
        if cls.__module__.startswith("flliper.srt.managers.scheduler_") and not cls.__module__.endswith("pp_mixin"):
            parts.append(inspect.getsource(cls))
    return "\n".join(parts)


def merged_text() -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in source_files())


def merged_tree() -> ast.Module:
    """scheduler.py's module with every moved method back in ``class Scheduler`` and every moved top-level node back
    in the module body (positions are those of their own file; nothing here depends on them)."""
    tree = ast.parse(SCHEDULER_PY.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    for path in SPLIT_FILES:
        other = ast.parse(path.read_text(encoding="utf-8"))
        for node in other.body:
            if isinstance(node, ast.ClassDef) and node.name.startswith("Scheduler") and node.name.endswith("Mixin"):
                cls.body.extend(n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Assign, ast.AnnAssign)):
                tree.body.append(node)
    return tree
