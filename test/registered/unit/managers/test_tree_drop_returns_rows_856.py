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
    allocator, `reset` only rebuilds bookkeeping."""

    def __init__(self, alloc, held_rows):
        self.alloc = alloc
        self.held = list(held_rows)
        for r in self.held:
            alloc.free_rows.discard(r)
        self.full_evictable_size_ = len(self.held)
        self.resets = 0

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


class TestTheSeamUsesIt(CustomTestCase):
    def test_build_cutover_release_hands_back_the_returning_drop(self):
        # Pinned because the bare `tree_cache.reset` is the tempting one-liner
        # and it is exactly what leaked on metal.
        import inspect

        from sglang.srt.managers.phase_flip_runtime import build_cutover_release

        src = inspect.getsource(build_cutover_release)
        self.assertIn("drop_prefix_tree_returning_rows", src)
        self.assertNotIn("return _retract, tree_cache.reset", src)


if __name__ == "__main__":
    unittest.main()
