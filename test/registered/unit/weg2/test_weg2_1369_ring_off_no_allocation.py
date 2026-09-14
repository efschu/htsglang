"""#1369 CLOSEOUT -- USER ORDER 2026-09-14 ("DIE 48GB MUESSEN WEG. UND ZWAR
PRONTO"): closing the pincer DESK12's own inventory
(/spinning/gpu-arb/weg2/INVENTAR_ringleser_0914.md) opened. Paket C
(model_runner.py, landed 99e2bd9e27) and Paket B (weight_updater.py's
main_carried, landed c27f05e121) both gate whether an ALLOCATION ever asks
the ring for backup. This file closes the THIRD, previously unguarded leg:
whether the LAUNCHER creates the ring FILE at all.

THE GAP, verified myself (not assumed from the coordinator's trace) against
this worktree's own checkout of ``python/sglang/srt/weg2/launcher.py``:

* ``weg2_weights_cpu_backup_ring_kw`` (added by DESK9, ``0266010523``/
  ``6e68501c0c``) correctly zeroes ``ring_bytes``/``ring_span1_bytes`` and
  forces ``ring_absent_by_design=True`` for the LEDGER's own pricing --
  ``choose_host_ledger`` calls it and the numbers ``host_ledger.price()``
  sees are already right. That is the PRICE.
* ``prepare_host_ring`` -- the function that actually calls
  ``ring_table.solve(...)`` and then, per card, ``os.open`` +
  ``os.ftruncate`` a REAL file sized ``2 MiB header + H(c) MiB`` (its own
  docstring: "the per-card files are created BEFORE either group starts")
  -- has NO ``weights_cpu_backup_armed`` parameter at all, and its ONE
  production call site (``main()``, the ``step 1b`` comment) does not pass
  one either. It runs, and creates full-size files, UNCONDITIONALLY,
  regardless of ``--weg2-weights-cpu-backup off``. That is the FILE, and it
  is what a boot's ``du``/tmpfs accounting and any co-located rank's
  ``cudaHostRegister`` would actually see -- the ledger's own correct price
  tag on a box that still physically exists.

THE DANGER DIRECTION THIS TICKET NAMES, restated for this specific gap:
"Schalter off, Allokation passiert trotzdem -- die 43,83 GiB (boot
weg2ring1's own measured Sigma H, never quoted as a bare constant) bleiben
still liegen." ``TestOffSwitchNeverTouchesRingTableOrDisk`` is that mutant,
caught structurally (mocked ``ring_table.solve`` raises if reached) and
proven red-then-green by hand (see the commit message for the manual
injection/reversion this file's own class was written against).

WHY table=None IS SAFE HERE, checked rather than assumed (the risk this file
would otherwise be blind to: conflating "we could not measure a table" (R22,
a genuine refusal) with "we deliberately need none" (this ticket) would turn
an intentional off-switch into a boot-killing W20):

* ``HostRingPlan.host_weights_bytes``/``host_weights_span1_bytes`` both
  check ``self.table is None`` FIRST and return 0 -- before the
  ``self.armed`` branch that would otherwise add an "old flip form" charge
  meant for a GENUINE refusal, not an intentional absence.
* ``choose_host_ledger`` -> ``weg2_weights_cpu_backup_ring_kw`` forces
  ``(0, 0, True)`` whenever ``weights_cpu_backup_armed`` is False,
  REGARDLESS of what ``ring_bytes``/``ring_span1_bytes`` were handed in --
  so whether ``ring_plan.host_weights_bytes`` reads 0 because ``table`` is
  None, or reads some large number that gets zeroed downstream, the ledger
  ends up pricing the identical ``(0, 0, True)`` either way.
* ``host_ledger.price()``'s W20 branch is an ``elif`` that only fires when
  ``ring_absent_by_design`` is False -- with it True (which it always is
  once ``weights_cpu_backup_armed`` is False), the only check left is the
  CONTRADICTION guard (refuses on a NON-zero ring beside the declaration),
  which a genuine 0 satisfies trivially.
* ``refuse_unless_same_form_source``'s own docstring names ``table is
  None`` as a NON-CASE it returns ``None`` for outright: "prepare_host_ring
  has already refused (R22/W20) for a reason of its own; a second refusal
  here would bury it." It does not distinguish WHY table is None, and does
  not need to -- it never raises on it either way.

So ``prepare_host_ring`` returning ``table=None`` under an explicit off
switch is not a NEW code path through the refusal machinery; it is the
SAME "no table" state three other functions already handle safely,
reached for a new (non-refusal) reason.

FILE BOUNDARY: ``launcher.py`` only. NOT
``weight_updater.py``/``weg2_memory_saver.py`` (DESK10, Paket B) and NOT
``xchg_bounce.py``/``weight_exchange_bounce.py`` (DESK11, Paket D).
"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher, ring_table
from sglang.test.test_utils import CustomTestCase


def _measured_table() -> "ring_table.RingTable":
    """A table with REAL, non-trivial per-card MiB -- if anything in the
    off-switch path accidentally reused a zeroed table instead of skipping
    the solve altogether, this fixture's numbers (not zero) would show up
    in a created file's size and this file's own assertions would catch it.
    """
    rows = [
        ring_table.CardRing(uuid="GPU-aaaa", nvml_index=1, name="RTX 5090",
                            image_p_mib=13860, image_d_mib=13914,
                            max_tag_p_mib=2988, max_tag_d_mib=2856,
                            credit_d2p_mib=13441, credit_p2d_mib=12609),
        ring_table.CardRing(uuid="GPU-bbbb", nvml_index=0, name="RTX 3080",
                            image_p_mib=7548, image_d_mib=9680,
                            max_tag_p_mib=2986, max_tag_d_mib=2714,
                            credit_d2p_mib=7106, credit_p2d_mib=7741),
    ]
    return ring_table.RingTable(boot="weg2zr2", instrument="i", lines_read=1,
                                cards=rows, max_step_total_mib=3600)


class _SolveRaisesIfCalled:
    """Context manager: ``ring_table.solve`` raises AssertionError if ever
    invoked while active. This IS the danger-direction mutant catcher --
    "the switch says off but the allocation happens anyway" is exactly a
    reached ``ring_table.solve`` call under this context."""

    def __enter__(self):
        self._real = ring_table.solve

        def _boom(*_a, **_k):
            raise AssertionError(
                "ring_table.solve was called under weights_cpu_backup_armed="
                "False -- the off switch does not prevent the ring from "
                "being measured and sized, which is the #1369 defect this "
                "file exists to catch"
            )

        ring_table.solve = _boom
        return self

    def __exit__(self, *exc):
        ring_table.solve = self._real
        return False


class _SolveReturns:
    """Context manager: ``ring_table.solve`` returns a fixed (table, reason)
    pair, for the control path where the off switch is NOT engaged."""

    def __init__(self, table, reason="stubbed"):
        self.table = table
        self.reason = reason

    def __enter__(self):
        self._real = ring_table.solve
        ring_table.solve = lambda *a, **k: (self.table, self.reason)
        return self

    def __exit__(self, *exc):
        ring_table.solve = self._real
        return False


class TestOffSwitchNeverTouchesRingTableOrDisk(CustomTestCase):
    """THE CORE FIX: ``weights_cpu_backup_armed=False`` must short-circuit
    before ``ring_table.solve`` is ever called and before any per-card file
    is created."""

    def test_ring_table_solve_is_never_reached(self):
        lines = []
        with _SolveRaisesIfCalled():
            plan = launcher.prepare_host_ring(
                [], lines.append, "t1369off", "auto", "/nonexistent", "",
                True,  # dry -- belt and braces, the assertion below is the
                       # real proof either way
                p_argv=[], weights_cpu_backup_armed=False,
            )
        self.assertIsNone(plan.table)
        self.assertFalse(plan.armed)

    def test_no_directory_or_file_is_created_even_with_dry_false(self):
        import tempfile

        tmp = tempfile.mkdtemp(prefix="weg2ringoff-")
        ring_dir_glob = os.path.join(tmp, "*")
        lines = []
        with _SolveRaisesIfCalled():
            plan = launcher.prepare_host_ring(
                [], lines.append, "t1369off2", "auto", "/nonexistent", "",
                False,  # dry=False -- if the guard is missing, THIS would
                        # actually create files, which is exactly the
                        # allocation this test proves does not happen
                p_argv=[], weights_cpu_backup_armed=False,
            )
        self.assertEqual(plan.dir, "", "no ring directory was ever named")
        import glob
        self.assertEqual(glob.glob(ring_dir_glob), [],
                         "the off switch must create NOTHING on disk")

    def test_the_log_line_says_absent_by_design_not_a_refusal(self):
        lines = []
        with _SolveRaisesIfCalled():
            launcher.prepare_host_ring(
                [], lines.append, "t1369off3", "auto", "/nonexistent", "",
                True, p_argv=[], weights_cpu_backup_armed=False,
            )
        joined = "\n".join(lines)
        self.assertIn("ABSENT-BY-DESIGN", joined)
        self.assertIn("--weg2-weights-cpu-backup", joined)
        # THE DISTINCTION THAT MATTERS: this must never read like the A1-3
        # "no fallback, this boot refuses" text -- that text describes a
        # boot that COULD NOT get what it needed, this one deliberately
        # does not need it.
        for refusal_token in ("REFUSES", "W20", "W22", "W32", "W33", "W34"):
            self.assertNotIn(refusal_token, joined,
                             f"the off-switch line must not read as a "
                             f"refusal (found {refusal_token!r})")

    def test_host_weights_bytes_reads_zero_from_the_returned_plan(self):
        """The ONE property ``choose_host_ledger`` actually reads
        (``ring_plan.host_weights_bytes`` / ``_span1_bytes``) must be 0, so
        the ledger's own zeroing has nothing left to contradict."""
        lines = []
        with _SolveRaisesIfCalled():
            plan = launcher.prepare_host_ring(
                [], lines.append, "t1369off4", "auto", "/nonexistent", "",
                True, p_argv=[], weights_cpu_backup_armed=False,
            )
        self.assertEqual(plan.host_weights_bytes, 0)
        self.assertEqual(plan.host_weights_span1_bytes, 0)

    def test_refuse_unless_same_form_source_is_a_silent_no_op_on_this_plan(self):
        """The THIRD function that must not misread this ``table=None`` as a
        NEW refusal it should raise -- pinned by actually calling it, not by
        re-reading its docstring."""
        lines = []
        with _SolveRaisesIfCalled():
            plan = launcher.prepare_host_ring(
                [], lines.append, "t1369off5", "auto", "/nonexistent", "",
                True, p_argv=[], weights_cpu_backup_armed=False,
            )
        self.assertIsNone(
            launcher.refuse_unless_same_form_source(plan.table, "ring"))
        self.assertIsNone(
            launcher.refuse_unless_same_form_source(plan.table, "exchange"))


class TestDefaultTrueIsByteIdenticalToBeforeThisTicket(CustomTestCase):
    """Control group: the default (``weights_cpu_backup_armed=True``, and
    OMITTING the argument entirely) must reach ``ring_table.solve`` and arm
    the ring exactly as every pre-#1369 boot did."""

    def test_omitting_the_new_kwarg_still_solves_and_arms(self):
        lines = []
        with _SolveReturns(_measured_table()):
            plan = launcher.prepare_host_ring(
                [], lines.append, "t1369on", "MAP_SHARED", "/nonexistent",
                "", True, p_argv=[],
            )
        self.assertIsNotNone(plan.table)
        self.assertTrue(plan.armed)
        self.assertIn("WEG2-HOST-RING ARMED", "\n".join(lines))

    def test_explicit_true_is_the_same_as_omitting_it(self):
        lines_a, lines_b = [], []
        with _SolveReturns(_measured_table()):
            plan_a = launcher.prepare_host_ring(
                [], lines_a.append, "t1369on2", "MAP_SHARED", "/nonexistent",
                "", True, p_argv=[],
            )
        with _SolveReturns(_measured_table()):
            plan_b = launcher.prepare_host_ring(
                [], lines_b.append, "t1369on2", "MAP_SHARED", "/nonexistent",
                "", True, p_argv=[], weights_cpu_backup_armed=True,
            )
        self.assertEqual(plan_a.armed, plan_b.armed)
        self.assertEqual(lines_a, lines_b)


class TestTheGuardSitsBeforeTheSolveCallStructurally(CustomTestCase):
    """A source-order pin, belt-and-braces beside the behavioural tests
    above: ``if not weights_cpu_backup_armed`` (however it ends up spelled)
    must appear TEXTUALLY BEFORE ``ring_table.solve(`` inside
    ``prepare_host_ring``'s own body, so a future edit that moved the guard
    below the solve call -- which would keep every test above green only by
    accident of mock ordering in a differently-shaped regression -- is
    caught here instead."""

    def test_the_weights_cpu_backup_armed_check_precedes_the_solve_call(self):
        import inspect

        src = inspect.getsource(launcher.prepare_host_ring)
        self.assertIn("weights_cpu_backup_armed", src)
        guard = src.find("weights_cpu_backup_armed")
        solve_call = src.index("ring_table.solve(")
        # the FIRST mention must be the parameter itself (in the signature),
        # so find the guard's own conditional mention distinctly:
        cond = src.find("if not weights_cpu_backup_armed")
        self.assertNotEqual(cond, -1,
                            "no `if not weights_cpu_backup_armed` guard found")
        self.assertLess(cond, solve_call,
                        "the off-switch guard must sit BEFORE the "
                        "ring_table.solve call, not after it")


if __name__ == "__main__":
    unittest.main(verbosity=2)
