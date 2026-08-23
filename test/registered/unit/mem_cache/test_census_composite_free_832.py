"""#832: the pool census must ask the allocator how it accounts free rows.

THE SPECIMEN, and what it cost. `_pool_census` computed its free set as

    free = set(alloc.free_pages.tolist()) | set(alloc.release_pages.tolist())

which hard-codes a PAGE-LIST allocator. The two unified composite allocators
stub both fields to a permanently empty tensor at construction, with the reason
stated in place -- "Base init left these None; we use watermark math, not
free-lists" (multi_ended_allocator.py:1734) and "Empty (not None) for the leak
checker" (:2070). On those flavors `free` is therefore UNCONDITIONALLY EMPTY,
and every genuinely free row falls into `unaccounted`. The tree records the
scale of that as ~94000 rows, 21% of a 448698-row pool, FLAT across four
censuses (phase_flip_runtime.py:4167) -- flat being the signature of a steady
available watermark rather than of a leak, because a leak accumulates. That
figure is a CITATION, not a measurement this suite reproduces; see the
restcheck paragraph below for what the retained logs can and cannot settle.

IT DID NOT STAY A LOG BUG. `_census_ownership_audit` feeds the same three sets
to the #822 authority (kv_row_ownership.py:567-571), which declares `free_list`
over them and then asks whether every committed row has an owner. With the free
set empty, that is ~94000 FALSE ownership violations per census.

BOTH DIRECTIONS, AND THE THIRD ONE. Teaching the census to read a watermark is
only a fix if it can still report a real leak afterwards, and only safe if an
allocator it does NOT understand produces a NAMED UNKNOWN rather than a silent
zero -- zero is the one wrong answer that turns a whole pool into a phantom
leak (the #606 getattr family). All three are asserted here.

WHAT THE AVAILABLE SPECIMENS DO AND DO NOT SHOW. The 2026-08-23 window-1/2/3
boot logs on this rig all carry `enable_unified_memory=False`, so the composite
allocators were never constructed in them and their census lines cannot confirm
or refute the ~94000 reading; that figure traces to a 2026-08-22 r5 boot which
is not among the retained logs. What those logs DO show is the same defect
SHAPE on a non-composite allocator: in 19 of 177 censuses the allocator's own
`available_size()` exceeds the census's deduplicated page-list count, by
4534-17771 rows. The premise of this fix -- that the census must read the
allocator's own free accounting rather than reconstruct it -- is measurable
there even though the composite instance is not.

No GPU, no flip, no scheduler: the shipped methods are driven directly.
"""

import logging
import unittest
import unittest.mock
from types import SimpleNamespace

import torch

from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime
from sglang.srt.mem_cache.allocator.base import BaseTokenToKVPoolAllocator
from sglang.srt.mem_cache.allocator.swa import PureSWATokenToKVPoolAllocator
from sglang.srt.mem_cache.kv_row_ownership import (
    FREE_COUNTED,
    FREE_COUNTED_UNDECLARED,
    FREE_ENUMERATED,
    FREE_UNKNOWN,
    FREE_WATERMARK,
    Law,
    RowOwnershipAuthority,
    RowSpace,
    audit_pool_census,
    read_free_rows,
)
from sglang.srt.mem_cache.multi_ended_allocator import (
    MultiEndedAllocator,
    UnifiedMambaTokenToKVPoolAllocator,
    UnifiedSWATokenToKVPoolAllocator,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5)

# The live specimen, as named constants so the docstring and the assertions
# cannot drift apart.
POOL_SIZE = 448698
FREE_ROWS = 94000
CACHED = 1000


class _Pages:
    """Stands in for an allocator's id tensors; only `.tolist()` is used."""

    def __init__(self, ids):
        self._ids = list(ids)

    def tolist(self):
        return list(self._ids)


def _run_census(alloc, cached_ids=()):
    """Drive the real `_pool_census` and return every line it logged."""
    tree = SimpleNamespace(all_values_flatten=lambda: _Pages(cached_ids))
    scheduler = SimpleNamespace(
        token_to_kv_pool_allocator=alloc,
        tree_cache=tree,
        running_mbs=[],
        running_batch=None,
        last_batch=None,
        phase_flip_stacks=None,
        tp_worker=None,
    )
    stub = SimpleNamespace(_census_scheduler=scheduler)
    stub._owner_ident = PhaseFlipRuntime._owner_ident
    stub._owner_pool_of = PhaseFlipRuntime._owner_pool_of
    stub._census_owner_probe = lambda *a, **k: None
    stub._pool_census = PhaseFlipRuntime._pool_census.__get__(stub, SimpleNamespace)

    logger = logging.getLogger("sglang.srt.managers.phase_flip_runtime")
    with unittest.mock.patch.object(logger, "warning") as warn:
        stub._pool_census("at-arm", "pp_to_tp")
    assert warn.called, "the census must always emit"
    return [c[0][0] % tuple(c[0][1:]) for c in warn.call_args_list]


def _census_line(alloc, cached_ids=()):
    """The POOL CENSUS line itself, not the explanatory line beside it."""
    for line in _run_census(alloc, cached_ids):
        if "POOL CENSUS" in line and "free_src=" in line:
            return line
    raise AssertionError("no census line emitted")


class _Composite:
    """A stubbed composite allocator, shaped exactly like the shipped ones.

    Empty page lists, watermark accounting, and the declaration that says so.
    """

    census_free_accounting = FREE_WATERMARK
    size = POOL_SIZE
    page_size = 1
    residency_withheld_slots = 0

    def __init__(self, available=FREE_ROWS):
        self.free_pages = _Pages([])
        self.release_pages = _Pages([])
        self._available = available

    def available_size(self):
        return self._available


class _UndeclaredComposite(_Composite):
    """The same allocator with the declaration forgotten -- the future-composite
    case the corroboration branch exists to catch."""

    census_free_accounting = None


class _Opaque:
    """An allocator that answers in no form the census knows."""

    size = POOL_SIZE
    page_size = 1
    residency_withheld_slots = 0
    free_pages = None
    release_pages = None


class _Paged:
    """An ordinary page-list allocator: real ids, corroborating available_size."""

    size = POOL_SIZE
    page_size = 1
    residency_withheld_slots = 0

    def __init__(self, free_ids):
        self.free_pages = _Pages(free_ids)
        self.release_pages = _Pages([])
        self._n = len(free_ids)

    def available_size(self):
        return self._n


class TestTheDefectItself(CustomTestCase):
    """RED-FIRST: the old expression, run against the specimen allocator."""

    def test_the_page_list_expression_produces_the_94000_band(self):
        """The line this fix replaces, evaluated on a composite allocator.

        This is the defect stated as arithmetic rather than as a story: the
        expression `_pool_census` used to run reads a composite's free capacity
        as ZERO, so the whole of it lands in `unaccounted`. Kept as a test so
        that anyone reverting the dispatch sees exactly what returns.
        """
        alloc = _Composite()
        old_free = set(alloc.free_pages.tolist()) | set(alloc.release_pages.tolist())
        self.assertEqual(old_free, set(), "the composite's page lists are empty")

        cached = set(range(1, CACHED + 1))
        old_unaccounted = len(set(range(1, POOL_SIZE + 1)) - old_free - cached)
        # 94000 genuinely free rows, every one of them reported as unaccounted.
        self.assertGreaterEqual(old_unaccounted, FREE_ROWS)
        self.assertEqual(old_unaccounted, POOL_SIZE - CACHED)

    def test_the_allocator_could_have_been_asked_all_along(self):
        """The correct answer was one call away on the same object."""
        self.assertEqual(_Composite().available_size(), FREE_ROWS)


class TestTheDispatch(CustomTestCase):
    """`read_free_rows` answers in the allocator's own shape, or says UNKNOWN."""

    def test_declared_watermark_is_counted_never_enumerated(self):
        r = read_free_rows(_Composite())
        self.assertEqual(r.kind, FREE_COUNTED)
        self.assertEqual(r.count, FREE_ROWS)
        # A count is not a set, and must not be dressed up as one.
        self.assertIsNone(r.rows)
        self.assertFalse(r.is_enumerable)
        self.assertTrue(r.is_answerable)

    def test_the_declaration_outranks_the_empty_page_lists(self):
        """MUTANT GUARD. Inspecting `free_pages` before the declaration is
        exactly the defect: the composites DO carry those fields, deliberately,
        as empty tensors. A dispatch that checks them first classifies a
        composite as a page-list allocator and rebuilds the 94000 band."""
        alloc = _Composite()
        self.assertIsNotNone(alloc.free_pages)
        self.assertIsNotNone(alloc.release_pages)
        self.assertEqual(read_free_rows(alloc).kind, FREE_COUNTED)

    def test_page_list_allocator_is_still_enumerated(self):
        r = read_free_rows(_Paged([1, 2, 3, 7]))
        self.assertEqual(r.kind, FREE_ENUMERATED)
        self.assertEqual(r.rows, frozenset({1, 2, 3, 7}))
        self.assertEqual(r.count, 4)
        self.assertTrue(r.is_enumerable)

    def test_a_genuinely_empty_page_list_stays_enumerated(self):
        """A full pool is not an unknown pool. When the page list is empty AND
        `available_size()` agrees, zero free rows is the truth and is reported
        as an enumerated zero -- not downgraded to a guess."""
        r = read_free_rows(_Paged([]))
        self.assertEqual(r.kind, FREE_ENUMERATED)
        self.assertEqual(r.count, 0)
        self.assertEqual(r.rows, frozenset())

    def test_an_undeclared_composite_is_caught_by_corroboration(self):
        """THE GENERALISATION. A future composite that forgets the declaration
        would fall through to the page-list branch, read zero, and silently
        rebuild this defect. Two authorities on the same object disagreeing is
        information; believing the empty one is how the class survives."""
        r = read_free_rows(_UndeclaredComposite())
        self.assertEqual(r.kind, FREE_COUNTED_UNDECLARED)
        self.assertEqual(r.count, FREE_ROWS)
        self.assertIsNone(r.rows)
        self.assertIn("census_free_accounting", r.detail)

    def test_an_unreadable_allocator_is_unknown_not_zero(self):
        r = read_free_rows(_Opaque())
        self.assertEqual(r.kind, FREE_UNKNOWN)
        self.assertIsNone(r.count)
        self.assertIsNone(r.rows)
        self.assertFalse(r.is_answerable)

    def test_a_declared_watermark_that_cannot_answer_is_unknown(self):
        """A declaration is not an answer. If `available_size()` raises or
        returns nonsense, the count is UNKNOWN -- not zero, and not the
        declaration taken on faith."""

        class _Broken(_Composite):
            def available_size(self):
                raise RuntimeError("no")

        self.assertEqual(read_free_rows(_Broken()).kind, FREE_UNKNOWN)

    def test_no_allocator_at_all_is_unknown(self):
        self.assertEqual(read_free_rows(None).kind, FREE_UNKNOWN)


class TestTheCensusLine(CustomTestCase):
    """The Fenster-4 criterion, asserted on the emitted text."""

    def test_the_composite_no_longer_shows_the_94000_band(self):
        text = _census_line(_Composite(), cached_ids=range(1, CACHED + 1))
        self.assertIn(f"free={FREE_ROWS}", text)
        self.assertNotIn(f"unaccounted={POOL_SIZE - CACHED}", text)
        # The 94000 free rows are now accounted for, so what is left over is
        # the rest of the pool, not the free capacity.
        self.assertIn(f"unaccounted={POOL_SIZE - FREE_ROWS - CACHED}", text)

    def test_the_census_names_the_allocator_it_read(self):
        """A census that does not say what it read cannot be checked against
        that thing's semantics afterwards -- which is precisely why settling
        the original 94000 reading needed a live probe."""
        text = _census_line(_Composite())
        self.assertIn("alloc=_Composite", text)
        self.assertIn(f"free_src={FREE_COUNTED}:{FREE_ROWS}", text)

    def test_the_unknown_allocator_prints_unknown_and_never_zero(self):
        """MUTANT GUARD, and the sharpest one. A silent `0` here reports every
        row in the pool as leaked; a silent `unaccounted=0` reports a clean
        bill of health. Both are lies, with opposite signs."""
        text = _census_line(_Opaque())
        self.assertIn("free=UNKNOWN", text)
        self.assertIn("unaccounted=UNKNOWN", text)
        self.assertNotIn("free=0 ", text)

    def test_the_reason_is_logged_verbatim_beside_the_line(self):
        lines = _run_census(_Opaque())
        self.assertTrue(
            any("free accounting:" in ln for ln in lines),
            "an unreadable allocator must say why, not merely print UNKNOWN",
        )

    def test_a_page_list_census_is_unchanged(self):
        """BACKWARD COMPATIBILITY. The ordinary path must emit exactly what it
        emitted before, including the single-line shape."""
        alloc = _Paged([1, 2, 3])
        lines = _run_census(alloc, cached_ids=[4, 5])
        self.assertEqual(len(lines), 1, "no extra line on the healthy path")
        text = lines[0]
        self.assertIn("free=3", text)
        self.assertIn("cached=2", text)
        self.assertIn(f"unaccounted={POOL_SIZE - 5}", text)
        self.assertIn(f"free_src={FREE_ENUMERATED}:3", text)

    def test_a_real_leak_is_still_reported_on_a_composite(self):
        """The other direction: subtracting a term is only a fix if the
        instrument can still say 'leak'. Free capacity drops to nothing, so
        everything not cached is genuinely unexplained."""
        text = _census_line(_Composite(available=0), cached_ids=range(1, CACHED + 1))
        self.assertIn("free=0", text)
        self.assertIn(f"unaccounted={POOL_SIZE - CACHED}", text)


class TestTheAuditConsumesTheSameSource(CustomTestCase):
    """One authority, two consumers -- the audit must not re-derive the shape."""

    def _authority(self):
        return RowOwnershipAuthority(RowSpace(exposed=POOL_SIZE, committed=POOL_SIZE))

    def test_a_counted_reading_raises_no_false_unowned_violations(self):
        """THE COST OF THE DEFECT, asserted directly. With the free rows
        unenumerable, 'this row belongs to nobody' is UNANSWERABLE -- and an
        unanswerable question must be suppressed, never answered from an empty
        set."""
        auth = self._authority()
        found = audit_pool_census(
            auth,
            exposed=POOL_SIZE,
            committed=POOL_SIZE,
            free_rows=None,
            free_count=FREE_ROWS,
            free_detail="watermark",
            cached_rows=range(1, CACHED + 1),
            resident_rows={"requests": ()},
            why="composite",
        )
        unowned = [v for v in found if v.law is Law.EXCLUSIVITY]
        self.assertEqual(unowned, [], f"false ownership violations: {found}")

    def test_the_empty_set_would_have_produced_them(self):
        """RED-FIRST for the audit leg: the same call with the empty set the
        old census handed over does report the pool as unowned."""
        auth = self._authority()
        found = audit_pool_census(
            auth,
            exposed=POOL_SIZE,
            committed=POOL_SIZE,
            free_rows=set(),
            cached_rows=range(1, CACHED + 1),
            resident_rows={"requests": ()},
            why="composite-as-page-list",
        )
        unowned = [v for v in found if v.law is Law.EXCLUSIVITY]
        self.assertTrue(unowned, "the defect must be reproducible")
        self.assertGreaterEqual(unowned[0].rows, FREE_ROWS)

    def test_no_free_list_claim_is_declared_for_a_counted_reading(self):
        """A claim over an empty set is not the absence of a claim: it asserts
        that nothing is free."""
        auth = self._authority()
        audit_pool_census(
            auth,
            exposed=POOL_SIZE,
            committed=POOL_SIZE,
            free_rows=None,
            free_count=FREE_ROWS,
            cached_rows=range(1, CACHED + 1),
            resident_rows={"requests": ()},
            why="composite",
        )
        self.assertNotIn("free_list", auth.owners())

    def test_a_stale_free_list_claim_is_withdrawn(self):
        """A claim that is not refreshed must not survive -- the same rule the
        resident owners already follow. Otherwise an earlier enumerated census
        keeps vouching for rows a later counted census cannot see."""
        auth = self._authority()
        auth.declare("free_list", range(1, 50))
        self.assertIn("free_list", auth.owners())
        audit_pool_census(
            auth,
            exposed=POOL_SIZE,
            committed=POOL_SIZE,
            free_rows=None,
            free_count=FREE_ROWS,
            cached_rows=range(1, CACHED + 1),
            resident_rows={"requests": ()},
            why="composite",
        )
        self.assertNotIn("free_list", auth.owners())

    def test_the_enumerated_path_still_audits_in_full(self):
        """The suppression must be scoped to the unanswerable case only."""
        auth = self._authority()
        found = audit_pool_census(
            auth,
            exposed=POOL_SIZE,
            committed=POOL_SIZE,
            free_rows=range(1, 100),
            cached_rows=range(100, 200),
            resident_rows={"requests": ()},
            why="paged",
        )
        self.assertIn("free_list", auth.owners())
        self.assertTrue(found, "a mostly-unowned pool must still be reported")


class TestTheShippedAllocatorsDeclareTheirShape(CustomTestCase):
    """The declaration is on the real classes, not only on the test doubles."""

    def test_both_composites_and_the_sub_allocator_declare_watermark(self):
        for cls in (
            UnifiedMambaTokenToKVPoolAllocator,
            UnifiedSWATokenToKVPoolAllocator,
            MultiEndedAllocator,
        ):
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.census_free_accounting, FREE_WATERMARK)

    def test_ordinary_allocators_do_not_declare(self):
        """The declaration must be an opt-in, or it means nothing."""
        self.assertIsNone(
            getattr(BaseTokenToKVPoolAllocator, "census_free_accounting", None)
        )


class _Staging:
    """A minimal allocator that runs the SHIPPED free-group methods.

    Base's real `reclaim_abandoned_free_group` / `_apply_free_group` are used,
    so what is under test is the override's decision to call them -- not a
    reimplementation of the recovery.
    """

    reclaim_abandoned_free_group = (
        BaseTokenToKVPoolAllocator.reclaim_abandoned_free_group
    )
    _apply_free_group = BaseTokenToKVPoolAllocator._apply_free_group

    def __init__(self, begin):
        self.is_not_in_free_group = True
        self.free_group = []
        self.freed = []
        self._begin = begin

    def free_group_begin(self):
        return self._begin(self)

    def free(self, tensor):
        self.freed.extend(int(x) for x in tensor.tolist())


class TestO8AbandonedWindowsAreRecoveredNotDiscarded(CustomTestCase):
    """#832 O-8: five `free_group_begin` overrides bypassed the W10/#827 fix.

    `BaseTokenToKVPoolAllocator.free_group_begin` was fixed to call
    `reclaim_abandoned_free_group()` first, because rows staged by a window
    nobody closed are out of the radix tree, in no free list, and held by no
    request -- so assigning `self.free_group = []` makes them unreachable for
    the life of the process, and the idle invariant then reports them as a
    fatal leak under the default SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.

    An override silently opts out of an inherited fix. Four classes did:
    `PureSWATokenToKVPoolAllocator`, `MultiEndedAllocator`, and both unified
    composites. (`HiSparseTokenToKVPoolAllocator.free_group_begin` is a bare
    `return` -- it never sets `is_not_in_free_group`, so `free` never stages
    and there is nothing to discard. Different shape, no hole; asserted below
    so the distinction is recorded rather than assumed.)
    """

    OVERRIDES = {
        "PureSWATokenToKVPoolAllocator": PureSWATokenToKVPoolAllocator,
        "MultiEndedAllocator": MultiEndedAllocator,
        "UnifiedMambaTokenToKVPoolAllocator": UnifiedMambaTokenToKVPoolAllocator,
        "UnifiedSWATokenToKVPoolAllocator": UnifiedSWATokenToKVPoolAllocator,
    }

    def test_every_override_recovers_an_abandoned_window(self):
        for name, cls in self.OVERRIDES.items():
            with self.subTest(allocator=name):
                alloc = _Staging(cls.free_group_begin)
                # A window is opened and rows are staged into it...
                alloc.free_group_begin()
                self.assertFalse(alloc.is_not_in_free_group)
                alloc.free_group.append(torch.tensor([11, 22, 33]))
                # ...and then nobody closes it. The next window opens.
                alloc.free_group_begin()
                self.assertEqual(
                    alloc.freed,
                    [11, 22, 33],
                    f"{name} discarded the staged rows instead of recovering them",
                )

    def test_the_ordinary_path_stays_silent(self):
        """0 recovered and no side effects when nothing was abandoned -- the
        recovery must not become a second free path."""
        for name, cls in self.OVERRIDES.items():
            with self.subTest(allocator=name):
                alloc = _Staging(cls.free_group_begin)
                alloc.free_group_begin()
                self.assertEqual(alloc.freed, [])
                self.assertEqual(alloc.free_group, [])

    def test_each_override_still_opens_the_window(self):
        """The recovery must not swallow the method's actual job."""
        for name, cls in self.OVERRIDES.items():
            with self.subTest(allocator=name):
                alloc = _Staging(cls.free_group_begin)
                alloc.free_group_begin()
                self.assertFalse(alloc.is_not_in_free_group)
                self.assertEqual(alloc.free_group, [])

    def test_hisparse_never_stages_so_it_has_no_hole(self):
        from sglang.srt.mem_cache.allocator.hisparse import (
            HiSparseTokenToKVPoolAllocator,
        )

        alloc = _Staging(HiSparseTokenToKVPoolAllocator.free_group_begin)
        alloc.free_group_begin()
        # It never opens a window, so `free` never stages and no rows can be
        # stranded. This is a different shape, not the same fix applied twice.
        self.assertTrue(alloc.is_not_in_free_group)


if __name__ == "__main__":
    unittest.main()
