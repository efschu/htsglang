# SPDX-License-Identifier: Apache-2.0
"""Ring neighbour arithmetic pin for barlink_bar1_ext.py (#621, Test B).

INVARIANT BEING PINNED
----------------------
barlink_bar1_ext.py ~line 1303-1304:
  ``const int next_rank = (r + 1) % R;``
  ``const int prev_rank = (r + R - 1) % R;``

These expressions define the ring topology used for all-ring/bar1
collectives.  Rank ``r`` sends to ``next_rank`` and receives from
``prev_rank``.  The receive-from expression is algebraically equivalent
to ``(r - 1 + R) % R``.

WHAT THIS TEST PROVES
---------------------
 1. The ``next_rank`` and ``prev_rank`` expressions are extracted directly
    from the CUDA kernel source string (``_CUDA_SRC``) via regex, not
    hardcoded in the test.  If someone edits the expressions, the regex
    captures the new text and the test re-evaluates from it.
 2. For a ring of size R=3, the computed partner tables are exactly
    ``{0: (1, 2), 1: (2, 0), 2: (0, 1)}`` where each value is
    ``(send_partner, recv_partner)``.
 3. The test FAILS if either expression is changed to something that
    produces different partner tables for R=3.

Pure source analysis.  No nvcc, no GPU, no import of the bar1 module.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from sglang.test.test_utils import CustomTestCase


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------

_COMM = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/distributed/device_communicators"
)
_EXT = _COMM / "barlink_bar1_ext.py"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cuda_src() -> str:
    """Extract the ``_CUDA_SRC`` literal from the Python source file.

    Reads the file as text, parses the AST, and returns the string literal
    value of the module-level ``_CUDA_SRC`` assignment.  This is the
    approach used by ``test_barlink_bar1_ext_codegen.py``.
    """
    import ast

    tree = ast.parse(_EXT.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            target = node.targets[0]
            if getattr(target, "id", "") == "_CUDA_SRC":
                assert isinstance(node.value, ast.Constant)
                return node.value.value
    raise AssertionError(f"_CUDA_SRC not found in {_EXT.name}")


def _extract_neighbour_exprs(src: str) -> tuple[str, str]:
    """Parse ``next_rank`` and ``prev_rank`` expressions from the CUDA source.

    Returns ``(next_expr, prev_expr)`` as strings, e.g.
    ``("(r + 1) % R", "(r + R - 1) % R")``.

    Extracted by regex, not hardcoded: if the expressions are edited, the
    regex picks up the new text automatically.
    """
    next_m = re.search(
        r"const\s+int\s+next_rank\s*=\s*([^;]+)\s*;", src
    )
    prev_m = re.search(
        r"const\s+int\s+prev_rank\s*=\s*([^;]+)\s*;", src
    )
    assert next_m, "Could not find 'const int next_rank = ...' in _CUDA_SRC"
    assert prev_m, "Could not find 'const int prev_rank = ...' in _CUDA_SRC"
    return next_m.group(1).strip(), prev_m.group(1).strip()


def _eval_ring_expr(expr_text: str, r: int, R: int) -> int:
    """Evaluate a ring neighbour expression for rank ``r`` in a ring of size ``R``.

    The expression text is something like ``(r + 1) % R`` or
    ``(r + R - 1) % R``.  We substitute Python-friendly variable names
    and evaluate.

    SAFETY: the expression comes from our own source file, not user input.
    We restrict the eval namespace to math operations only.
    """
    # Build a safe namespace with only the two loop variables.
    # The expression uses ``r`` (current rank) and ``R`` (world size).
    ns = {"r": r, "R": R}
    return int(eval(expr_text, {"__builtins__": {}}, ns))  # noqa: S307


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

class TestRingNeighbours(CustomTestCase):
    """Pinned partner-table for the ring topology in bar1_collective.

    For R=3 the expected partner table is:
        rank 0: send -> 1, recv from 2
        rank 1: send -> 2, recv from 0
        rank 2: send -> 0, recv from 1

    Written as a dict: {0: (1, 2), 1: (2, 0), 2: (0, 1)}.
    """

    def setUp(self) -> None:
        self.src = _cuda_src()
        self.next_expr, self.prev_expr = _extract_neighbour_exprs(self.src)

    def test_next_rank_expr_is_canonical(self) -> None:
        """The extracted next_rank expression is the canonical ``r+1`` ring step.

        Verifies the expression text matches the expected form, so a
        refactoring that accidentally uses a different formula (e.g.
        ``(r - R + 1) % R`` which is algebraically equivalent but unusual)
        is flagged for review.
        """
        # The canonical form is "(r + 1) % R".  Allow equivalent but
        # cosmetically different forms like "(r+1)%R" (no spaces).
        self.assertRegex(
            self.next_expr,
            r"\(r\s*\+\s*1\)\s*%\s*R",
            msg=(
                f"next_rank expression '{self.next_expr}' does not match "
                "the canonical '(r + 1) % R' pattern. If this was a "
                "deliberate change, update the regex here with justification."
            ),
        )

    def test_prev_rank_expr_is_canonical(self) -> None:
        """The extracted prev_rank expression is the canonical ``r+R-1`` wrap.

        Verifies the expression text matches the expected form.
        Equivalent forms like ``(r - 1 + R) % R`` are also accepted.
        """
        self.assertRegex(
            self.prev_expr,
            r"\(r\s*\+\s*R\s*-\s*1\)\s*%\s*R|\(r\s*-\s*1\s*\+\s*R\)\s*%\s*R",
            msg=(
                f"prev_rank expression '{self.prev_expr}' does not match "
                "the canonical '(r + R - 1) % R' or '(r - 1 + R) % R' "
                "pattern. If this was a deliberate change, update the regex "
                "here with justification."
            ),
        )

    def test_partner_table_r3(self) -> None:
        """For R=3, the computed partner table is exactly the expected one.

        THE INVARIANT: computed from the parsed expressions, not hardcoded.
        If someone edits either expression, the eval here uses the NEW text
        and the assertion fails.
        """
        R = 3
        expected = {
            0: (1, 2),  # rank 0: send->1, recv<-2
            1: (2, 0),  # rank 1: send->2, recv<-0
            2: (0, 1),  # rank 2: send->0, recv<-1
        }

        actual: dict[int, tuple[int, int]] = {}
        for r in range(R):
            send_partner = _eval_ring_expr(self.next_expr, r, R)
            recv_partner = _eval_ring_expr(self.prev_expr, r, R)
            actual[r] = (send_partner, recv_partner)

        self.assertEqual(
            actual,
            expected,
            (
                f"Ring partner table for R={R} does not match expected. "
                f"next_rank expr: '{self.next_expr}', "
                f"prev_rank expr: '{self.prev_expr}'.\n"
                f"  Expected: {expected}\n"
                f"  Actual:   {actual}"
            ),
        )

    def test_partner_table_self_consistency(self) -> None:
        """Cross-check: rank A sends to B iff rank B receives from A.

        For any R, the ring must satisfy:
          next(A) == B  <=>  prev(B) == A

        This catches formula swaps (e.g. next_rank accidentally using the
        prev_rank formula, or vice versa).
        """
        R = 3
        for a in range(R):
            for b in range(R):
                if a == b:
                    continue
                send_to = _eval_ring_expr(self.next_expr, a, R)
                recv_from_b = _eval_ring_expr(self.prev_expr, b, R)
                if send_to == b:
                    self.assertEqual(
                        recv_from_b,
                        a,
                        (
                            f"Ring inconsistency: rank {a} sends to rank {b}, "
                            f"but rank {b} receives from rank {recv_from_b} "
                            f"(expected {a})"
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
