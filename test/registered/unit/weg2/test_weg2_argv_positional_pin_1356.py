# SPDX-License-Identifier: Apache-2.0
"""#1356 [fix] -- no NEW positional caller of the argv builders.

THE DEFECT THIS PINS. 2cc618b819 inserted `vision` in the middle of `argv_p`
and `common_flags` while their callers passed POSITIONALLY. Seven parameters
shifted; `RING_FORM_SENTINEL_DEPTH` landed in `vision` and the per-rank window
string landed in `depth`, so `str(int(depth))` raised

    ValueError: invalid literal for int() with base 10: '24,PP_0=96'

on the NORMAL start path. Every boot of the tip died there.

WHAT SAVED IT WAS A TYPE ACCIDENT, NOT A GUARD: `window_mib` happens to be a
string. Numeric, and the launch would have SUCCEEDED with seed, BAR1 window and
census interval silently re-bound, and the form key hashed over an argv nobody
meant. `common_flags` is the worse case -- every value it shifts is an int, so
nothing there would ever raise.

bfc252c172 closed the instance: `argv_p` is keyword-only from its first
optional and `vision` is keyword-only in `common_flags`, so that particular
insert cannot repeat. This pins the CLASS: the positional profile of every call
site is recorded, so a NEW positional caller -- or one that grows -- turns red
here instead of at the next boot.

THE KNOWN LIST IS A TO-DO, NOT AN ENDORSEMENT. Two `argv_d` call sites still
pass 18 positionals each. They are named below so the pin lands green on the
tip it was written for, and converting them is a separate, deliberate change
that must read `argv_d`'s real signature. This seat guessed those names twice
from `argv_p`'s and produced a NameError and then a SyntaxError, which is why
the conversion is not bundled here: a pin that documents a debt is honest, a
rushed conversion that re-binds arguments is the defect itself.
"""

import ast
import inspect
import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as lc
from sglang.test.test_utils import CustomTestCase

#: (function, positional-argument count) for every call site in launcher.py,
#: measured at bfc252c172. A new positional caller adds an entry; an existing
#: one that grows changes its count. Either way this list stops matching.
#:
#: THE TWO `argv_d` ENTRIES ARE THE OPEN DEBT (launcher.py:10727 and :10805 at
#: bfc252c172 -- line numbers drift, which is why they are not the key).
KNOWN_POSITIONAL_CALLS = {
    ("argv_d", 18): 2,
    ("common_flags", 10): 4,
}

#: Below this, a call is short enough that an inserted parameter cannot reach
#: an optional term -- `argv_p`'s seven required positionals, the smallest
#: capacity of the three.
_FLOOR = 7


def _profile():
    src = inspect.getsource(lc)
    out = {}
    for node in ast.walk(ast.parse(src)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in ("argv_p", "argv_d", "common_flags")
                and len(node.args) > _FLOOR):
            key = (node.func.id, len(node.args))
            out[key] = out.get(key, 0) + 1
    return out


class NoNewPositionalCaller(CustomTestCase):
    def test_the_positional_profile_is_unchanged(self):
        """AST, not grep: a source scan matches the comments describing this."""
        now = _profile()
        self.assertEqual(
            now, KNOWN_POSITIONAL_CALLS,
            "the positional call profile of the argv builders changed.\n"
            f"  now:   {sorted(now.items())}\n"
            f"  known: {sorted(KNOWN_POSITIONAL_CALLS.items())}\n"
            "A NEW positional caller re-binds every argument after any "
            "parameter someone inserts, and on common_flags every shifted "
            "value is an int -- nothing raises. Pass by keyword, or add the "
            "call here deliberately with a reason.")

    def test_argv_p_has_no_positional_caller_left(self):
        """The one that crashed is now structurally unreachable."""
        self.assertEqual(
            [k for k in _profile() if k[0] == "argv_p"], [],
            "argv_p is keyword-only from its first optional; a positional "
            "caller beyond its required arguments cannot be correct")

    def test_the_open_debt_is_exactly_two_argv_d_call_sites(self):
        """The to-do, asserted so it cannot quietly grow."""
        self.assertEqual(_profile().get(("argv_d", 18)), 2)

    def test_argv_p_is_keyword_only_from_its_first_optional(self):
        kinds = [p.kind for p in inspect.signature(lc.argv_p).parameters.values()]
        self.assertIn(inspect.Parameter.KEYWORD_ONLY, kinds)
        n_pos = sum(1 for k in kinds
                    if k == inspect.Parameter.POSITIONAL_OR_KEYWORD)
        self.assertEqual(n_pos, _FLOOR,
                         "argv_p's required positionals changed; the floor "
                         "this pin uses must be re-derived, not adjusted")

    def test_vision_cannot_shift_a_positional_in_either_builder(self):
        for fn in (lc.common_flags, lc.argv_p):
            with self.subTest(fn=fn.__name__):
                self.assertEqual(
                    inspect.signature(fn).parameters["vision"].kind,
                    inspect.Parameter.KEYWORD_ONLY)


if __name__ == "__main__":
    unittest.main()
