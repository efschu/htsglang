"""Repo-wide structural lint: no statements behind a top-level return/raise.

This is the generalisation of the guard added in 38220757cc for
`KVSessionOffloadManager`. The bug it catches is not "dead code" as a style
issue -- it is the fingerprint of a `def` that was accidentally written at
class-body indentation *inside* another function's body:

    class C:
        def __init__(self):
            stmt_1
            ...
        def inserted(self):     # <- meant to come after __init__
            return x

            stmt_20             # <- these were __init__'s remaining statements
            stmt_21             #    and are now `inserted`'s dead code

Python accepts this silently. `__init__` simply ends at the new `def`, and
every statement after it stops running. That happened once for real (the R1
fix truncated `KVSessionOffloadManager.__init__` from 63 statements to 38,
dropping `self._free_regions` -- an immediate AttributeError on any boot with
`--enable-kv-session-offload`) and the full CPU suite stayed green, because
the unit tests for that class build it with `__new__`.

The structural signature is always the same: the host function's tail ends up
behind the inserted function's `return`. This test scans the whole tree for
that signature, so the next occurrence is caught wherever it lands, not only
in the one file that already has a bespoke guard.

Deliberately NOT flagged:
  * generator functions -- `return` / `raise` followed by `yield` is the
    standard idiom for "empty generator" / "generator that only raises";
  * a trailing bare string literal, i.e. a prose block parked at the end of a
    function; that is documentation, not code.
"""

import ast
import pathlib
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=25, suite="base-a-test-cpu")


_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SCAN_ROOTS = [_REPO_ROOT / "python" / "sglang", _REPO_ROOT / "test"]

# Pre-existing, deliberate dead code inherited from upstream. Keyed by
# (path relative to repo root, function name) so it survives line drift.
# Anything NOT in here is a finding -- do not extend this list to silence a
# new hit without first proving the hit is not a truncated function.
_KNOWN_UPSTREAM_DEAD_CODE = {
    # Upstream TODO: rid handling is disabled with an early `return None`,
    # the real body is kept below it on purpose.
    (
        "python/sglang/srt/entrypoints/openai/serving_base.py",
        "_generate_request_id_base",
    ),
    # Upstream stub: `raise NotImplementedError("teacache is not supported
    # yet ...")` in front of the not-yet-wired teacache body.
    (
        "python/sglang/multimodal_gen/runtime/models/dits/hunyuanvideo.py",
        "should_skip_forward_for_cached_states",
    ),
}


class TestNoUnreachableAfterReturn(CustomTestCase):
    def test_no_unreachable_statements_after_return(self):
        offenders = []
        for root in _SCAN_ROOTS:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                offenders.extend(_find_unreachable(path))

        self.assertFalse(
            offenders,
            msg=(
                "Statements found behind a top-level `return`/`raise`. This is "
                "what an accidentally in-lined `def` looks like: the host "
                "function ends at the new `def` and its tail becomes dead code. "
                "Check the line numbers against the surrounding function "
                "definitions before assuming it is only unused code:\n  "
                + "\n  ".join(offenders)
            ),
        )


def _find_unreachable(path: pathlib.Path):
    try:
        source = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return []

    rel = path.relative_to(_REPO_ROOT).as_posix()
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if (rel, node.name) in _KNOWN_UPSTREAM_DEAD_CODE:
            continue
        if _is_generator(node):
            continue
        for index, stmt in enumerate(node.body[:-1]):
            if not isinstance(stmt, (ast.Return, ast.Raise)):
                continue
            tail = node.body[index + 1 :]
            if all(_is_prose(item) for item in tail):
                break
            kind = "return" if isinstance(stmt, ast.Return) else "raise"
            found.append(
                f"{rel}:{tail[0].lineno} in {node.name}() "
                f"(unreachable after the {kind} on line {stmt.lineno})"
            )
            break
    return found


def _is_generator(node) -> bool:
    """True if `node` itself is a generator (yields in nested defs don't count)."""
    stack = list(node.body)
    while stack:
        item = stack.pop()
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        if isinstance(item, (ast.Yield, ast.YieldFrom)):
            return True
        stack.extend(ast.iter_child_nodes(item))
    return False


def _is_prose(stmt: ast.stmt) -> bool:
    """A bare string literal statement -- a prose block, not code."""
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


if __name__ == "__main__":
    unittest.main()
