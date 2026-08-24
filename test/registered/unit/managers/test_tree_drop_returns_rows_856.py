"""#856 W27-retry: the seam's tree drop must RETURN rows, not orphan them.

THE SPECIMEN. boot_w27r_0824_1551.log, on the THIRD retract+drop cycle, all
three ranks:

    ValueError: pool memory leak detected! [full] total=472864,
      available=126802, evictable=22, protected=0, session_held=0,
      uncached=0, withheld=345888

126802 + 22 + 345888 = 472712 against a total of 472864 -> **152 rows belong
to nobody**, and it accumulates once per cycle. `evictable=22` is the NEW
tree; the old tree's rows are gone from every owner's books, which is why the
detector can only see them as a total mismatch.

DERIVED FROM THE TREE CODE, NOT GUESSED. `MambaRadixCache.reset`
(mamba_radix_cache.py:555) installs a NEW `TreeNode()` as root and zeroes
`full_evictable_size_` / `full_protected_size_`. It frees no device row -- the
old tree is simply dereferenced. It is a BOOKKEEPING reset, correct for a
teardown where the pool is reset too, wrong for a seam that keeps serving. The
call that actually returns rows is `evict` -> `evict_full`, whose leaf path
frees through `token_to_kv_pool_allocator.free`.

So the drop is EVICT-THEN-RESET, and eviction is legitimate here only because
the #703 fence has already persisted these prefixes to the canonical store.
Without the fence this would be data loss -- which is why the seam order is
fence -> retract -> drop and not any permutation.

THE TWO DANGER DIRECTIONS ARE BOTH PINNED:
  * orphaning (the W27-retry defect) -- rows never returned;
  * DOUBLE-RETURN -- a row handed back twice, which is silent where the leak
    was loud. `_Allocator` below raises on a double free, so any
    evict-and-also-free-again implementation fails here.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

import unittest

from sglang.srt.managers.phase_flip_runtime import drop_prefix_tree_returning_rows
from sglang.test.test_utils import CustomTestCase


class _Allocator:
    """Models the one property the leak detector checks: every row has an
    owner. Raises on a double free, which is the danger direction."""

    def __init__(self, total):
        self.total = total
        self.free_rows = set(range(total))

    def free(self, rows):
        for r in rows:
            if r in self.free_rows:
                raise AssertionError(f"row {r} returned twice -- double free")
            self.free_rows.add(r)

    @property
    def available(self):
        return len(self.free_rows)


class _Tree:
    """A tree double with the REAL two-API split: `evict` returns rows to the
    allocator, `reset` only rebuilds bookkeeping.

    Shaped after the attribute-keeping caches (`MambaRadixCache`,
    `SWARadixCache`), which hold the count in `full_evictable_size_` AND
    expose it through the `full_evictable_size()` method
    (swa_radix_cache.py:844, mamba_radix_cache.py:1502). The method was
    missing here until W29, so this double had the private half and not the
    public one -- backwards from every shipped cache, and the reason ten
    green tests survived a boot that died of this exact defect.
    """

    def __init__(self, alloc, held_rows):
        self.alloc = alloc
        self.held = list(held_rows)
        for r in self.held:
            alloc.free_rows.discard(r)
        self.full_evictable_size_ = len(self.held)
        self.resets = 0

    def full_evictable_size(self) -> int:
        return self.full_evictable_size_

    def evict(self, params):
        n = min(int(params.num_tokens), len(self.held))
        rows, self.held = self.held[:n], self.held[n:]
        self.alloc.free(rows)
        self.full_evictable_size_ = len(self.held)

        class _R:
            num_tokens_evicted = n

        return _R()

    def reset(self):
        # THE DEFECT, modelled exactly: a new root, zeroed accounting, and the
        # rows simply dereferenced.
        self.held = []
        self.full_evictable_size_ = 0
        self.resets += 1


class TestTheDropReturnsTheRows(CustomTestCase):
    def test_rows_go_back_to_the_allocator(self):
        alloc = _Allocator(1000)
        tree = _Tree(alloc, range(100, 152))
        before = alloc.available
        returned = drop_prefix_tree_returning_rows(tree)
        self.assertEqual(returned, 52)
        self.assertEqual(alloc.available, before + 52)
        self.assertEqual(tree.resets, 1, "the tree is still reset afterwards")

    def test_a_bare_reset_is_what_leaks(self):
        # THE SPECIMEN, modelled: reset alone loses the rows. If this ever
        # stops leaking, the double has drifted from the real tree and every
        # other assertion here is worthless.
        alloc = _Allocator(1000)
        tree = _Tree(alloc, range(100, 152))
        before = alloc.available
        tree.reset()
        self.assertEqual(alloc.available, before, "reset returned nothing")
        self.assertEqual(tree.full_evictable_size_, 0, "and forgot them too")

    def test_the_leak_accumulates_per_cycle_without_the_fix(self):
        # W27-retry fired on the THIRD cycle, not the first. That shape is the
        # signature, so it is asserted rather than described.
        alloc = _Allocator(1000)
        lost = 0
        for cycle in range(3):
            tree = _Tree(alloc, range(100 + cycle * 60, 100 + cycle * 60 + 52))
            tree.reset()
            lost += 52
        self.assertEqual(alloc.available, 1000 - lost)

    def test_the_fix_does_not_accumulate(self):
        alloc = _Allocator(1000)
        for cycle in range(3):
            tree = _Tree(alloc, range(100 + cycle * 60, 100 + cycle * 60 + 52))
            drop_prefix_tree_returning_rows(tree)
        self.assertEqual(alloc.available, 1000, "every row came back")


class TestTheDoubleReturnDirection(CustomTestCase):
    """A row handed back twice is SILENT where the leak was loud."""

    def test_dropping_an_already_empty_tree_returns_nothing(self):
        alloc = _Allocator(1000)
        tree = _Tree(alloc, [])
        self.assertEqual(drop_prefix_tree_returning_rows(tree), 0)
        self.assertEqual(alloc.available, 1000)

    def test_dropping_twice_does_not_double_free(self):
        # The second drop must be a no-op. `_Allocator.free` raises on a
        # double return, so an implementation that evicted from a stale row
        # list would fail here rather than corrupt the pool quietly.
        alloc = _Allocator(1000)
        tree = _Tree(alloc, range(100, 152))
        drop_prefix_tree_returning_rows(tree)
        drop_prefix_tree_returning_rows(tree)
        self.assertEqual(alloc.available, 1000)


class TestItNeverBreaksTheSeam(CustomTestCase):
    """It runs after the requests are already retracted; aborting here would
    strand a flip that has released its state."""

    def test_a_tree_that_refuses_to_evict_still_gets_reset(self):
        class _Refuses(_Tree):
            def evict(self, params):
                raise RuntimeError("evict refused")

        alloc = _Allocator(1000)
        tree = _Refuses(alloc, range(100, 152))
        self.assertEqual(drop_prefix_tree_returning_rows(tree), 0)
        self.assertEqual(tree.resets, 1)

    def test_a_tree_with_no_evict_api_still_gets_reset(self):
        class _Bare:
            full_evictable_size_ = 7

            def __init__(self):
                self.resets = 0

            def reset(self):
                self.resets += 1

        t = _Bare()
        drop_prefix_tree_returning_rows(t)
        self.assertEqual(t.resets, 1)

    def test_an_unreadable_evictable_size_does_not_raise(self):
        class _Odd:
            full_evictable_size_ = "n/a"

            def __init__(self):
                self.resets = 0

            def reset(self):
                self.resets += 1

        t = _Odd()
        self.assertEqual(drop_prefix_tree_returning_rows(t), 0)
        self.assertEqual(t.resets, 1)


class _UnifiedShapedTree:
    """The tree shape the seam ACTUALLY meets on this rig, and the one the
    double above does not model.

    `UnifiedRadixCache` -- `tree_type=UnifiedRadixCache` in every W29 census
    line -- keeps its evictable count in `component_evictable_size_`, a dict
    keyed by component, and exposes it through the `full_evictable_size()`
    METHOD declared on `BasePrefixCache`. It has NO `full_evictable_size_`
    attribute. `_Tree` above has the attribute and not the method, which is
    exactly backwards from production, and that drift is why this suite was
    green through the W29 boot that died of this defect.
    """

    def __init__(self, alloc, held_rows):
        self.alloc = alloc
        self.held = list(held_rows)
        for r in self.held:
            alloc.free_rows.discard(r)
        self.component_evictable_size_ = {"full": len(self.held)}
        self.resets = 0

    def full_evictable_size(self) -> int:
        return self.component_evictable_size_.get("full", 0)

    def evict(self, params):
        n = min(int(params.num_tokens), len(self.held))
        rows, self.held = self.held[:n], self.held[n:]
        self.alloc.free(rows)
        self.component_evictable_size_["full"] = len(self.held)

        class _R:
            num_tokens_evicted = n

        return _R()

    def reset(self):
        self.held = []
        self.component_evictable_size_["full"] = 0
        self.resets += 1


class TestTheLiveTreeShape(CustomTestCase):
    """W29 (SPECIMEN_w29_a1_pool_leak_1row.log). The fix shipped for W27-retry
    read `full_evictable_size_`, an attribute the live tree does not have, so
    `getattr(tree, "full_evictable_size_", 0)` returned ZERO, the eviction was
    skipped outright, and `reset()` orphaned the tree's rows exactly as before.

    The seam's own #832 census named the orphan by id -- `unaccounted=1 [1]`,
    flat from the first flip that crossed a non-empty tree -- and the idle
    check killed all three ranks:

        pool memory leak detected! [full] total=469733, available=107041,
          evictable=1, protected=0, session_held=0, uncached=0,
          withheld=362690

    ONE row only because the tree held exactly one: a 1-token health check.
    The orphan is the size of the tree, not a constant, and this class pins
    both sizes so nobody re-reads the W29 arithmetic as a reserved slot.
    """

    def test_a_unified_shaped_tree_gets_its_rows_back(self):
        # RED before the fix: the old read finds no `full_evictable_size_`,
        # defaults to 0, skips the evict, and `reset()` drops the row.
        alloc = _Allocator(1000)
        tree = _UnifiedShapedTree(alloc, [1])
        before = alloc.available
        returned = drop_prefix_tree_returning_rows(tree)
        self.assertEqual(returned, 1, "the one row the tree held came back")
        self.assertEqual(alloc.available, before + 1)
        self.assertEqual(tree.resets, 1, "the tree is still reset afterwards")

    def test_the_orphan_scales_with_the_tree_not_with_a_constant(self):
        # W29 read as a CONSTANT unit deficit and was therefore mistaken for a
        # reserved/boundary slot. It is not: it is the tree's own occupancy.
        for held in (1, 13, 152):
            alloc = _Allocator(100000)
            tree = _UnifiedShapedTree(alloc, range(1, held + 1))
            self.assertEqual(drop_prefix_tree_returning_rows(tree), held)
            self.assertEqual(alloc.available, 100000, f"{held} rows came back")

    def test_repeated_flips_do_not_accumulate(self):
        alloc = _Allocator(1000)
        for cycle in range(5):
            tree = _UnifiedShapedTree(alloc, range(1 + cycle * 60, 1 + cycle * 60 + 52))
            drop_prefix_tree_returning_rows(tree)
        self.assertEqual(alloc.available, 1000)

    def test_a_unified_shaped_tree_is_never_double_freed(self):
        alloc = _Allocator(1000)
        tree = _UnifiedShapedTree(alloc, range(100, 152))
        drop_prefix_tree_returning_rows(tree)
        drop_prefix_tree_returning_rows(tree)
        self.assertEqual(alloc.available, 1000)


class TestTheContractIsTheOneTheRealTreesImplement(CustomTestCase):
    """The drift-detector. These assertions run against the REAL classes, not
    against a double, because a double is what hid this defect for a whole
    boot."""

    def test_every_shipped_prefix_cache_answers_full_evictable_size(self):
        from sglang.srt.mem_cache.base_prefix_cache import BasePrefixCache
        from sglang.srt.mem_cache.mamba_radix_cache import MambaRadixCache
        from sglang.srt.mem_cache.radix_cache import RadixCache
        from sglang.srt.mem_cache.swa_radix_cache import SWARadixCache
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        for cls in (
            BasePrefixCache,
            RadixCache,
            SWARadixCache,
            MambaRadixCache,
            UnifiedRadixCache,
        ):
            self.assertTrue(
                callable(getattr(cls, "full_evictable_size", None)),
                f"{cls.__name__} must answer the base contract the seam reads",
            )

    def test_the_drop_does_not_read_the_private_attribute(self):
        # ONE WRITER, ONE CLOCK. `full_evictable_size_` is an implementation
        # detail of three of the five caches; reading it is what silently
        # returned 0 on the two that keep the count elsewhere.
        import inspect

        from sglang.srt.managers.phase_flip_runtime import tree_evictable_full_rows

        def _body(fn) -> str:
            # The docstrings NAME the private attribute on purpose -- that is
            # the history. Only the executable half is pinned here.
            src = inspect.getsource(fn)
            return src.split('"""')[2] if src.count('"""') >= 2 else src

        for fn in (tree_evictable_full_rows, drop_prefix_tree_returning_rows):
            self.assertNotIn("full_evictable_size_", _body(fn), fn.__name__)
        self.assertIn("full_evictable_size", _body(tree_evictable_full_rows))

    def test_a_tree_that_cannot_answer_is_not_treated_as_empty(self):
        # `None` is not zero. Zero is a licence to skip the eviction, and
        # skipping it is the whole defect.
        from sglang.srt.managers.phase_flip_runtime import tree_evictable_full_rows

        class _Mute:
            pass

        self.assertIsNone(tree_evictable_full_rows(_Mute()))

        class _Raises:
            def full_evictable_size(self):
                raise RuntimeError("no")

        self.assertIsNone(tree_evictable_full_rows(_Raises()))


class TestTheCheckerStaysStrict(CustomTestCase):
    """CAN-FAIL, the other direction: the W29 abort was CORRECT and must stay
    correct. Nothing in this fix softens, widens or special-cases the leak
    arithmetic -- it fixes the owner of the row, and these assertions fail if
    a later change ever buys a green boot by relaxing the checker instead."""

    def _check(self, **kw):
        from sglang.srt.managers.scheduler_components.invariant_checker import (
            SchedulerInvariantChecker,
        )

        return SchedulerInvariantChecker._check_pool_invariant("full", **kw)

    def test_the_w29_one_row_deficit_still_aborts(self):
        leak, _ = self._check(
            available=107041,
            evictable=1,
            protected=0,
            session_held=0,
            total=469733,
            uncached=0,
            withheld=362690,
        )
        self.assertTrue(leak, "a one-row deficit is a leak, not a tolerance")

    def test_a_real_multi_row_orphan_still_aborts(self):
        leak, _ = self._check(
            available=126802,
            evictable=22,
            protected=0,
            session_held=0,
            total=472864,
            uncached=0,
            withheld=345888,
        )
        self.assertTrue(leak, "the W27-retry 152-row orphan must still abort")

    def test_a_balanced_pool_is_clean(self):
        leak, _ = self._check(
            available=107042,
            evictable=1,
            protected=0,
            session_held=0,
            total=469733,
            uncached=0,
            withheld=362690,
        )
        self.assertFalse(leak, "the checker must be able to pass, or it proves nothing")


class TestTheSeamUsesIt(CustomTestCase):
    def test_build_cutover_release_hands_back_the_returning_drop(self):
        # Pinned because the bare `tree_cache.reset` is the tempting one-liner
        # and it is exactly what leaked on metal.
        import inspect

        from sglang.srt.managers.phase_flip_runtime import build_cutover_release

        src = inspect.getsource(build_cutover_release)
        self.assertIn("drop_prefix_tree_returning_rows", src)
        self.assertNotIn("return _retract, tree_cache.reset", src)

    def test_the_seam_reports_how_many_rows_the_drop_returned(self):
        # W29: the drop computed the number and the seam discarded it, so a
        # drop returning ZERO rows on every flip read identically in the log
        # to one returning all of them. A number nobody prints is not a
        # measurement.
        import inspect

        from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime

        src = inspect.getsource(PhaseFlipRuntime._release_residents_for_cutover)
        self.assertIn("tree_rows_returned", src)
        self.assertIn("returning %s", src)
        self.assertIn("UNKNOWN", src, "an absent count is not the number zero")


if __name__ == "__main__":
    unittest.main()
