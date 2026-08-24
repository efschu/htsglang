"""#849: the census LINE carries the fourth owner's verdict, not only the audit.

THE SPECIMEN. Window 7 (boot_window7_0824_0252, series in
/spinning/evidence-665-f1/window7/pool_census_844.txt) printed, on one line:

    ... cur_slot_reqs=5 resident_reqs=0 resident_slots=[] unaccounted=40960 ...

while the ownership AUDIT of the very same census found exactly ONE unowned
row -- "1 committed row id(s) ... belong to no enumerated owner" -- because
the audit is fed ``_resident_rows`` (#822's fourth owner) and the LINE never
was. ``resident_reqs=`` on the line is the ``running_mbs`` slot scope, which
is 0 by construction outside the PP event loop, so in the TP regime the line
simultaneously asserted "40960 rows unowned" and "no resident requests".
That pairing is what 844-O2 was misread from, and what sent #849.

THE FIX UNDER TEST. ``resident_census_terms`` derives two additional fields,
appended at the END of the line (existing fields byte-identical, additive per
the I832 pattern): ``resident_rows_n=`` (the fourth owner's row count) and
``unowned_after_owners=`` (what remains unowned once the fourth owner is
subtracted -- the audit's arithmetic, on the line). ``None`` from the
enumeration prints ``no-verdict`` on both: asserting absorption from an
empty set is the #822 defect with an extra step. The census and the audit
share ONE enumeration per census, so the two can never disagree about a
moving working set.

CPU-only: no GPU, no flip, no controller.
"""

import logging
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.managers import phase_flip_runtime as pfr
from sglang.srt.managers.phase_flip_runtime import (
    PhaseFlipRuntime,
    resident_census_terms,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

from test_census_composite_free_832 import _Composite, _Opaque, _Paged, _Pages

register_cpu_ci(est_time=10)

POOL_SIZE = _Paged.size


class TestTheArithmetic(CustomTestCase):
    """resident_census_terms: derived, overlap-exact, and honest about None."""

    def test_none_is_no_verdict_never_absorption(self):
        """The #822 contract: None means the enumeration had no verdict.
        Treating it as an empty set would re-assert the whole working set as
        unowned -- or worse, absorb rows nobody vouched for."""
        self.assertEqual(
            resident_census_terms({1, 2, 3}, 3, None, POOL_SIZE, True),
            ("no-verdict", "no-verdict"),
        )

    def test_enumerable_branch_is_an_id_difference(self):
        """Overlap-exact: resident ids OUTSIDE the leaked set (rows the tree
        also claims, or ids out of space) must not over-subtract. A count
        subtraction would report 0 here; the id difference reports 1."""
        leaked = set(range(1, 101))
        resident = set(range(1, 100)) | {POOL_SIZE + 500}
        n, unowned = resident_census_terms(leaked, 100, resident, POOL_SIZE, True)
        self.assertEqual(n, 100)
        self.assertEqual(unowned, 1)

    def test_counted_branch_subtracts_only_in_space_ids(self):
        resident = set(range(1, 31)) | {POOL_SIZE + 1, POOL_SIZE + 2}
        n, unowned = resident_census_terms(set(), 50, resident, POOL_SIZE, False)
        self.assertEqual(n, 32)
        self.assertEqual(unowned, 20)

    def test_counted_branch_reports_negative_overlap_as_it_falls(self):
        """Same rule as `unaccounted` itself: a negative count difference is
        a finding (owners overlap the watermark), never clamped."""
        _, unowned = resident_census_terms(set(), 10, set(range(1, 16)), POOL_SIZE, False)
        self.assertEqual(unowned, -5)

    def test_unknown_arithmetic_passes_through(self):
        n, unowned = resident_census_terms(set(), "UNKNOWN", {1, 2}, POOL_SIZE, False)
        self.assertEqual(n, 2)
        self.assertEqual(unowned, "UNKNOWN")


def _req(pool_idx, seqlen):
    return SimpleNamespace(req_pool_idx=pool_idx, seqlen=seqlen)


def _scheduler(alloc, cached_ids=(), row_matrix=None, reqs=()):
    """A scheduler stub shaped like the 832 harness, plus the request pool.

    ``row_matrix`` is the ``req_to_token`` table: row i lists the KV row ids
    request i holds. Without it, ``_resident_rows`` answers None (no pool ->
    no verdict), which is the 832 harness's implicit state.
    """
    tree = SimpleNamespace(all_values_flatten=lambda: _Pages(cached_ids))
    sched = SimpleNamespace(
        token_to_kv_pool_allocator=alloc,
        tree_cache=tree,
        running_mbs=[],
        running_batch=SimpleNamespace(reqs=list(reqs)) if reqs else None,
        last_batch=None,
        phase_flip_stacks=None,
        tp_worker=None,
    )
    if row_matrix is not None:
        sched.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.tensor(row_matrix, dtype=torch.int64)
        )
    return sched


def _run_census(sched, stub_extra=None):
    stub = SimpleNamespace(_census_scheduler=sched)
    stub._owner_ident = PhaseFlipRuntime._owner_ident
    stub._owner_pool_of = PhaseFlipRuntime._owner_pool_of
    stub._census_owner_probe = lambda *a, **k: None
    for k, v in (stub_extra or {}).items():
        setattr(stub, k, v)
    stub._pool_census = PhaseFlipRuntime._pool_census.__get__(stub, SimpleNamespace)

    logger = logging.getLogger("sglang.srt.managers.phase_flip_runtime")
    with mock.patch.object(logger, "warning") as warn:
        stub._pool_census("at-arm", "tp_to_pp")
    assert warn.called, "the census must always emit"
    return [c[0][0] % tuple(c[0][1:]) for c in warn.call_args_list]


def _census_line(sched, stub_extra=None):
    for line in _run_census(sched, stub_extra):
        if "POOL CENSUS" in line and "free_src=" in line:
            return line
    raise AssertionError("no census line emitted")


class TestTheLine(CustomTestCase):
    """The window-7 shape, driven through the real `_pool_census`."""

    def _window7_shaped_scheduler(self):
        """Five requests hold rows 1..100; nothing cached; the rest free.

        The pre-#849 line for this state reads `unaccounted=100` with
        `resident_reqs=0` -- the miniature of `unaccounted=40960` against
        `resident_reqs=0` on the specimen. Every one of those rows is owned.
        """
        working = list(range(1, 101))
        free = [i for i in range(1, POOL_SIZE + 1) if i > 100]
        rows = [working[i * 20 : (i + 1) * 20] for i in range(5)]
        reqs = [_req(i, 20) for i in range(5)]
        return _scheduler(_Paged(free), row_matrix=rows, reqs=reqs)

    def test_the_working_set_is_absorbed_on_the_line(self):
        line = _census_line(self._window7_shaped_scheduler())
        self.assertIn("unaccounted=100", line)
        self.assertIn("resident_reqs=0", line, "mbs scope stays what it is")
        self.assertIn("resident_rows_n=100", line)
        self.assertIn("unowned_after_owners=0", line)

    def test_existing_fields_stay_byte_identical(self):
        """Additive means additive: the new terms hang off the END of the
        line and `unaccounted=` keeps its historical meaning, so every
        existing parser and specimen grep still reads what it always read."""
        line = _census_line(self._window7_shaped_scheduler())
        head, sep, tail = line.partition(" resident_rows_n=")
        self.assertTrue(sep, "the new fields must be present")
        self.assertIn("unaccounted=100", head)
        self.assertNotIn("resident_rows_n", head)
        self.assertTrue(head.rstrip().endswith("free_src=enumerated:%d" % (POOL_SIZE - 100)))

    def test_no_pool_prints_no_verdict_on_both(self):
        """No request pool -> `_resident_rows` has no verdict; the line must
        say so rather than absorb (or re-assert) anything."""
        free = list(range(1, POOL_SIZE - 99))
        line = _census_line(_scheduler(_Paged(free)))
        self.assertIn("resident_rows_n=no-verdict", line)
        self.assertIn("unowned_after_owners=no-verdict", line)

    def test_counted_allocator_gets_the_count_difference(self):
        working = list(range(1, 41))
        rows = [working[i * 20 : (i + 1) * 20] for i in range(2)]
        reqs = [_req(i, 20) for i in range(2)]
        sched = _scheduler(
            _Composite(available=POOL_SIZE - 40), row_matrix=rows, reqs=reqs
        )
        line = _census_line(sched)
        self.assertIn("unaccounted=40", line)
        self.assertIn("resident_rows_n=40", line)
        self.assertIn("unowned_after_owners=0", line)

    def test_unknown_allocator_passes_unknown_through(self):
        working = list(range(1, 21))
        sched = _scheduler(_Opaque(), row_matrix=[working], reqs=[_req(0, 20)])
        line = _census_line(sched)
        self.assertIn("unaccounted=UNKNOWN", line)
        self.assertIn("resident_rows_n=20", line)
        self.assertIn("unowned_after_owners=UNKNOWN", line)


class TestOneEnumerationPerCensus(CustomTestCase):
    """Line and audit must read the same snapshot of a moving working set."""

    def test_the_audit_receives_the_line_s_enumeration(self):
        received = []
        sched = self._sched = None
        working = list(range(1, 41))
        rows = [working[i * 20 : (i + 1) * 20] for i in range(2)]
        reqs = [_req(i, 20) for i in range(2)]
        sched = _scheduler(
            _Paged([i for i in range(1, POOL_SIZE + 1) if i > 40]),
            row_matrix=rows,
            reqs=reqs,
        )

        def _audit(why, alloc, size, free_reading, cached, withheld, **kw):
            received.append(kw)

        with mock.patch.object(
            pfr, "_resident_rows", side_effect=pfr._resident_rows
        ) as counted:
            _census_line(sched, stub_extra={"_census_ownership_audit": _audit})
        self.assertEqual(counted.call_count, 1, "ONE enumeration per census")
        self.assertEqual(len(received), 1)
        self.assertIn("resident_rows", received[0])
        self.assertEqual(received[0]["resident_rows"], set(working))

    def test_audit_fallback_enumerates_only_when_nothing_was_passed(self):
        """`resident_rows=None` passed explicitly is a verdict (`no verdict`)
        and must NOT be replaced by a fresh enumeration; only the UNSET
        sentinel may fall back."""
        stub = SimpleNamespace(
            _census_scheduler=_scheduler(_Paged([1])),
            _committed_backing_rows=lambda: None,
        )
        audit = PhaseFlipRuntime._census_ownership_audit.__get__(
            stub, SimpleNamespace
        )
        with mock.patch.object(pfr, "_resident_rows") as enum:
            audit("t", _Paged([1]), 4, {1}, set(), set(), resident_rows=None)
        enum.assert_not_called()
        with mock.patch.object(pfr, "_resident_rows", return_value=None) as enum:
            audit("t", _Paged([1]), 4, {1}, set(), set())
        enum.assert_called_once()


if __name__ == "__main__":
    unittest.main()
