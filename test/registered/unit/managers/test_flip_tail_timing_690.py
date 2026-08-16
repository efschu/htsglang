"""#690: the flip's fixed cost is measured, not left as a residual.

WHAT THE LOGS SHOWED. Over 291 same-regime ``PHASE-FLIP DONE`` lines, the part
of the flip NOT covered by ``read + exchange + write`` is 2.0-2.1 s and is FLAT
across a 3600x range of live slots (123 -> 440 095). At 123 live slots it is
~81 % of the whole flip, so the fixed cost -- not the movement -- is what floors
every window #677's economics can solve, and what #692 prices depth against.

WHERE IT WAS HIDING. The three timers cover the wave loop only. The backing
swap is INSIDE ``write_ms`` (``t_write0`` is taken before
``release_wave``/``restore_wave``), so it was never the residual; what fell
outside was the tail after the loop:

    _pool_census("pre-cutover")        phase_flip_runtime.py
    for fn in self._pre_cutover_fns    the EXTRA MOVERS -- weights arena
                                       refill and the GDN state leg
    _cutover_fn(direction)             the group step
    _pool_census("post-cutover")

The movers are occupancy-INDEPENDENT by construction: the weights arena refill
is the same bytes whatever the KV live set holds, which is the leading
explanation for a residual that does not move with occupancy.

SO THEY ARE TIMED, SPLIT MOVERS vs CUTOVER, because they have different fixes.
A residual that has to be regressed across boots cannot be priced per flip; a
reported number can.
"""

import ast
import inspect
import re
import unittest

from sglang.srt.managers import phase_flip_runtime as pf
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5)


def _execute_source() -> str:
    return inspect.getsource(pf.PhaseFlipRuntime._execute)


class TheTailIsTimed(unittest.TestCase):
    def setUp(self):
        self.src = _execute_source()

    def test_the_movers_clock_opens_before_the_pre_cutover_census(self):
        i_clock = self.src.index("t_movers0 = self._clock()")
        i_census = self.src.index('self._pool_census("pre-cutover"')
        self.assertLess(
            i_clock,
            i_census,
            "the movers timer starts after the census it is meant to include",
        )

    def test_the_movers_clock_closes_after_the_mover_loop(self):
        i_loop = self.src.index("for fn in self._pre_cutover_fns:")
        i_close = self.src.index("movers_ms = (self._clock()")
        self.assertLess(i_loop, i_close)

    def test_the_cutover_clock_wraps_the_cutover(self):
        i_open = self.src.index("t_cutover0 = self._clock()")
        i_fn = self.src.index("self._cutover_fn(direction)")
        i_close = self.src.index("cutover_ms = (self._clock()")
        self.assertLess(i_open, i_fn)
        self.assertLess(i_fn, i_close)

    def test_the_three_original_timers_are_untouched(self):
        for name in ("read_ms", "xfer_ms", "write_ms"):
            self.assertIn(f"{name} +=", self.src, name)


class TheNumbersAreReported(unittest.TestCase):
    def setUp(self):
        self.src = _execute_source()

    def test_the_stats_dict_carries_both(self):
        self.assertIn('"movers_ms": movers_ms,', self.src)
        self.assertIn('"cutover_ms": cutover_ms,', self.src)

    def test_the_done_line_names_both(self):
        self.assertIn("movers %.1f ms", self.src)
        self.assertIn("cutover %.1f ms", self.src)


class TheDoneLineArityHolds(unittest.TestCase):
    """A %-format whose specifiers and arguments disagree raises at the moment
    the flip finishes -- the single worst place to learn it.

    Counted from the AST rather than by eye, so the next field added to this
    line is checked by a test instead of by a boot.
    """

    def _done_call(self):
        tree = ast.parse(inspect.getsource(pf).replace("\t", "    "))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "warning"):
                continue
            if not node.args:
                continue
            first = node.args[0]
            try:
                value = ast.literal_eval(first)
            except Exception:
                continue
            if isinstance(value, str) and value.startswith("%s DONE %s"):
                return value, node
        return None, None

    def test_the_done_line_exists_and_is_a_single_literal(self):
        fmt, node = self._done_call()
        self.assertIsNotNone(fmt, "the PHASE-FLIP DONE log call was not found")
        self.assertIsNotNone(node)

    def test_specifier_count_equals_argument_count(self):
        fmt, node = self._done_call()
        specs = re.findall(r"%(?:[-+ #0]*\d*(?:\.\d+)?)[a-zA-Z]", fmt.replace("%%", ""))
        args = len(node.args) - 1
        self.assertEqual(
            len(specs),
            args,
            f"the DONE line has {len(specs)} format specifiers and {args} "
            "arguments; the flip would raise at the moment it completes",
        )

    def test_the_two_new_fields_are_inside_that_count(self):
        fmt, _ = self._done_call()
        self.assertIn("movers", fmt)
        self.assertIn("cutover", fmt)


if __name__ == "__main__":
    unittest.main()
