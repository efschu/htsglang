# SPDX-License-Identifier: Apache-2.0
"""B4f: ring absence is a property of the INJECT ARM, not of the weight source.

Boot weg2xsn13's S6I shadow DRY-RUN stopped at rc=1 before a card was touched,
inside this module's own ledger call::

    ValueError: ring_absent_by_design with a non-zero ring
    (ring_bytes=46133149696, ring_span1_bytes=44559237120)
    host_ledger.py:1832 price <- :2626 choose <- launcher.py:4635
    choose_host_ledger <- :9263 main

THE LEDGER WAS RIGHT AND STAYS AS IT IS.  Its guard exists for exactly this:
*"the two inputs contradict each other, and a contradiction resolved silently
is how a charged term becomes invisible"*.  The same run had armed
21715+9758+12529 MiB = 42.97 GiB of ring and handed the ledger those very
bytes, while the launcher declared the ring absent.

THE DEFECT IS THE PREDICATE.  `ring_absent_by_design=(weight_source ==
"exchange")` reads ring absence off the WEIGHT SOURCE, and #1327's own comment
explains why that looked right: *"under `exchange` the launcher publishes no
TMS_HOST_RING_*"*.  That is true only when the exchange is the AUTHORITY.
Under `--weg2-xchg-inject shadow` -- the default, and the form the S6I order
grades -- the refill is still the authority and the ring is still armed (that
order's item (b): `host_weights=` UNCHANGED, and a `host_weights=0.00` there is
a FAIL).  So `exchange` + `shadow` is a real, ordered form in which the ring
is present, and it is the one combination the old predicate could not express.

The predicate is therefore a CONJUNCTION of the two arms, and it lives in one
named function so the two call sites cannot spell it differently.

Hermetic: the pure predicate, its composition with the ledger's own guard at
the exact bytes weg2xsn13 handed it, and the wiring pins.  No /proc, no NVML.
"""

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import launcher, weight_exchange
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

#: The bytes boot weg2xsn13 actually handed the ledger, from its own traceback.
XSN13_RING_BYTES = 46133149696
XSN13_SPAN1_BYTES = 44559237120


def _price(**kw):
    base = dict(
        memtotal_bytes=int(118.05 * hl.GIB), memavail_bytes=int(104.02 * hl.GIB),
        s_gb=1, m_mib=600, ranks_per_group=3, s_gb_d=4,
        cg_current_bytes=int(15.36 * hl.GIB),
        reclaimable_bytes=int(1.33 * hl.GIB),
    )
    base.update(kw)
    return hl.price(**base)


class RingAbsenceIsTheConjunctionOfBothArms(CustomTestCase):
    def test_the_whole_truth_table(self):
        """Three weight sources x two inject modes, stated rather than implied."""
        want = {
            ("ring", "shadow"): False,
            ("ring", "authoritative"): False,
            ("shadow", "shadow"): False,
            ("shadow", "authoritative"): False,
            ("exchange", "shadow"): False,          # <- weg2xsn13's form
            ("exchange", "authoritative"): True,    # <- the only absent one
        }
        for src in launcher.WEIGHT_SOURCE_CHOICES:
            for mode in weight_exchange.INJECT_CHOICES:
                self.assertEqual(
                    launcher.ring_absent_by_design(src, mode), want[(src, mode)],
                    f"{src}/{mode}")

    def test_the_shadow_inject_form_declares_the_ring_PRESENT(self):
        """The regression, in one line: this is what stopped weg2xsn13."""
        self.assertFalse(launcher.ring_absent_by_design(
            launcher.WEIGHT_SOURCE_EXCHANGE, weight_exchange.INJECT_SHADOW))

    def test_the_authoritative_form_still_declares_it_ABSENT(self):
        """Pin: the S6I AUTHORITATIVE boot must still get RING ABSENT BY DESIGN
        and `host weights term 0.00 GiB`.  Fixing the shadow form must not cost
        the form the declaration was written for."""
        self.assertTrue(launcher.ring_absent_by_design(
            launcher.WEIGHT_SOURCE_EXCHANGE, weight_exchange.INJECT_AUTHORITATIVE))

    def test_the_ring_form_is_byte_identical_under_both_modes(self):
        """Pin: the serving form's ledger input must not move either way."""
        for mode in weight_exchange.INJECT_CHOICES:
            self.assertFalse(launcher.ring_absent_by_design(
                launcher.WEIGHT_SOURCE_DEFAULT, mode))

    def test_an_unknown_inject_mode_is_not_authoritative(self):
        """An unrecognised mode must not silently zero a charged ring."""
        self.assertFalse(launcher.ring_absent_by_design("exchange", "nonsense"))
        self.assertFalse(launcher.ring_absent_by_design("exchange", ""))


class TheCompositionThatActuallyBlewUp(CustomTestCase):
    """The predicate joined to the ledger's own guard, at weg2xsn13's bytes.

    This is the end of the defect rather than a restatement of the fix: the
    ValueError came from ``price()`` reading a declaration that contradicted
    the bytes beside it, so the test prices THOSE bytes under THAT form.
    """

    def test_the_shadow_inject_form_prices_its_armed_ring(self):
        arm = _price(
            ring_bytes=XSN13_RING_BYTES, ring_span1_bytes=XSN13_SPAN1_BYTES,
            ring_absent_by_design=launcher.ring_absent_by_design(
                "exchange", weight_exchange.INJECT_SHADOW),
        )
        self.assertGreater(arm.terms["host_ring_gib"], 42.0)

    def test_the_authoritative_form_prices_no_ring_at_all(self):
        arm = _price(
            ring_bytes=0, ring_span1_bytes=0,
            ring_absent_by_design=launcher.ring_absent_by_design(
                "exchange", weight_exchange.INJECT_AUTHORITATIVE),
        )
        self.assertEqual(arm.terms["host_ring_gib"], 0.0)
        self.assertIs(arm.terms["ring_absent_by_design"], True)

    def test_the_ledgers_contradiction_guard_is_UNCHANGED(self):
        """It stays exactly as it is -- the ruling says so, and it was right."""
        with self.assertRaises(ValueError) as caught:
            _price(ring_bytes=XSN13_RING_BYTES, ring_span1_bytes=XSN13_SPAN1_BYTES,
                   ring_absent_by_design=True)
        self.assertIn("contradict each other", str(caught.exception))


class BothArmsReachTheOneCallSite(CustomTestCase):
    """Pins, because a predicate whose second argument never arrives is the
    old predicate wearing a conjunction."""

    def _src(self, fn):
        import inspect
        import re

        return re.sub(r"\s+", " ", inspect.getsource(fn))

    def test_choose_host_ledger_takes_the_inject_mode(self):
        import inspect

        sig = inspect.signature(launcher.choose_host_ledger)
        self.assertIn("inject_mode", sig.parameters)
        self.assertEqual(sig.parameters["inject_mode"].default,
                         weight_exchange.INJECT_SHADOW)

    def _kwargs_of_call(self, fn, callee):
        """``{keyword: unparsed value}`` of the call to ``callee`` inside ``fn``.

        STRUCTURAL, not textual, and a MUTANT bought this: the first version of
        the pin below asserted that ``inject_mode=ns.weg2_xchg_inject`` appears
        SOMEWHERE in ``main``, and it appears TWICE -- the ``prepare_xchg_env``
        call three statements earlier hands the same flag to a different
        consumer.  So a mutant that redirected the LEDGER's argument to a
        literal left the other occurrence standing and the pin passed.  A text
        scan cannot say WHICH call site an argument reached; an AST walk can,
        and it is immune to wrapping, indentation and neighbour order too --
        the other two traps this campaign has paid for.
        """
        import ast
        import inspect
        import textwrap

        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name == callee:
                return {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
        self.fail(f"{fn.__name__} contains no call to {callee}()")

    def test_choose_host_ledger_computes_it_through_the_helper(self):
        kw = self._kwargs_of_call(launcher.choose_host_ledger, "dict")
        self.assertEqual(kw.get("ring_absent_by_design"),
                         "ring_absent_by_design(weight_source, inject_mode)")

    def test_main_hands_the_LEDGER_call_the_flag_and_not_a_default(self):
        kw = self._kwargs_of_call(launcher.main, "choose_host_ledger")
        self.assertEqual(kw.get("inject_mode"), "ns.weg2_xchg_inject")

    def test_and_the_env_publisher_still_gets_it_too(self):
        """The other consumer of the same flag, so neither can be starved to
        satisfy the other."""
        kw = self._kwargs_of_call(launcher.main, "prepare_xchg_env")
        self.assertEqual(kw.get("inject_mode"), "ns.weg2_xchg_inject")

    def test_the_flags_default_is_shadow_so_the_bug_was_the_default_path(self):
        self.assertEqual(weight_exchange.INJECT_SHADOW, "shadow")
        self.assertEqual(weight_exchange.inject_mode.__module__,
                         "sglang.srt.weg2.weight_exchange")


if __name__ == "__main__":
    unittest.main()
