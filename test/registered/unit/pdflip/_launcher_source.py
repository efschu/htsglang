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


SPLIT_FUNCTIONS = ("main", "solve_p_cut")        # split behaviour-neutrally into _<name>_stepNN_* functions (refactor, 28.10.)


def _step_prefix_of(name: str) -> str:
    return "_%s_step" % name


def _is_step_def(node) -> bool:
    return isinstance(node, ast.FunctionDef) and any(node.name.startswith(_step_prefix_of(f)) for f in SPLIT_FUNCTIONS)


def _parse(path):
    text = path.read_text(encoding="utf-8")
    return text, text.split("\n"), ast.parse(text)


def expanded_source(name: str, decorators: bool = False) -> str:
    """``launcher.<name>`` as it read before it was split into ``_<name>_stepNN_*`` functions (behaviour-neutral refactor, 28.10.).

    Every line of the function that calls a step function (``[names = |return ]_<name>_stepNN_x(...)``) is replaced by that step's body, in
    call order (the steps may live in any ``launcher*.py``), so a source pin that asserts "A comes before B" or "calls X" reads the
    statements of the original function. The ``return`` a step got for the names later code reads is dropped; the docstring is dropped.
    ``decorators=True`` starts at the first decorator line."""
    import re

    prefix = _step_prefix_of(name)
    call = re.compile(r"^(\s+)(?:[\w, ]+ = |return )?(%s\d+_\w+)\(" % re.escape(prefix))
    steps, fn_node, fn_lines = {}, None, None
    for path in source_files():
        _, lines, tree = _parse(path)
        for n in tree.body:
            if isinstance(n, ast.FunctionDef) and n.name.startswith(prefix):
                body = n.body[1:]                               # without the docstring
                last = body[-1] if body else None
                if isinstance(last, ast.Return) and isinstance(last.value, (ast.Name, ast.Tuple)):
                    body = body[:-1]                            # the tool-added return of the names later code reads (a genuine ``return f(...)`` stays)
                first = n.body[0].end_lineno                    # keep comments/blank lines between the docstring and the first statement
                steps[n.name] = (lines[first: body[-1].end_lineno] if body else [], n.body[1].col_offset if len(n.body) > 1 else 4)
            elif isinstance(n, ast.FunctionDef) and n.name == name:
                fn_node, fn_lines = n, lines
    first_line = min([fn_node.lineno] + [d.lineno for d in fn_node.decorator_list]) if decorators else fn_node.lineno

    def expand(block):
        out = []
        for line in block:
            m = call.match(line)
            if m and m.group(2) in steps:
                body, body_indent = steps[m.group(2)]
                delta = len(m.group(1)) - body_indent           # a step lifted out of a nested block (or one hoisted by hand) sat at another depth
                for x in expand(body):                          # a later round may call steps of an earlier one
                    if not x.strip():
                        out.append(x)
                    elif delta >= 0:
                        out.append(" " * delta + x)
                    else:
                        out.append(x[-delta:])
            else:
                out.append(line)
        return out

    return "\n".join(expand(fn_lines[first_line - 1: fn_node.end_lineno]))


def expanded_main_source() -> str:
    """``launcher.main`` with its ``_main_stepNN_*`` calls expanded in call order, see :func:`expanded_source`."""
    return expanded_source("main")


def expanded_launcher_text() -> str:
    """The whole launcher source (``launcher.py`` + ``launcher_*.py``) with the split functions (:data:`SPLIT_FUNCTIONS`) expanded as in
    :func:`expanded_source` and every ``_<name>_stepNN_*`` definition removed, so text order inside them is the original order and nothing
    is counted twice."""
    parts = []
    for path in source_files():
        text, lines, tree = _parse(path)
        drop, fn_nodes = [], {}
        for n in tree.body:
            if _is_step_def(n):
                drop.append((n.lineno, n.end_lineno))
            elif isinstance(n, ast.FunctionDef) and n.name in SPLIT_FUNCTIONS:
                fn_nodes[n.lineno] = n
        drop = sorted(drop)
        out, i = [], 1
        while i <= len(lines):
            if i in fn_nodes:
                out.append(expanded_source(fn_nodes[i].name))
                i = fn_nodes[i].end_lineno + 1
                continue
            hit = next(((a, b) for a, b in drop if a == i), None)
            if hit:
                i = hit[1] + 1
                continue
            out.append(lines[i - 1])
            i += 1
        parts.append("\n".join(out))
    return "\n".join(parts)
