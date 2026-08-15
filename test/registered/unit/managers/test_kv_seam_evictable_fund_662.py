"""#662 -- the seam's fund is EVICTABLE CONTENT, not VRAM held free.

THE DEFECT, stated as arithmetic. ``kv_backing_relief`` releases only
backing that NO row occupies: the slack between the live high-water mark
and the pool's reservation. On a pool sized to fill the corridor there IS
no such slack -- the mark sits just under the reservation -- so the rung
reports "no slack above the live set", funds nothing, and the seam can
only be paid for out of memory held free at rest. On this rig that was
3361/5070/4303 MiB, ~12.7 GiB, against a corridor law of ~1024 MiB/card.

These tests pin the two halves of the fix on ONE pool state:

  * WITHOUT the watermark actuator the rung funds ZERO. That is the
    can-fail proof -- it is the shipped behaviour, reproduced here, and if
    a future change makes it non-zero the premise of this work is gone.
  * WITH it the same rung funds the whole distance down to the RESIDENT
    half of the live set, which is the memory the seam actually needs.

The two runs differ ONLY by SGLANG_KV_RADIX_EVICT_RELIEF. Nothing about
the pool, the live set or the corridor changes between them.

Hermetic: tensor-backed fakes, no CUDA, no scheduler.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.managers import kv_backing_relief as kbr

MIB = 1024 * 1024

#: Rows the rung keeps allocatable above the mark (register C20).
RESERVE = 512

#: A pool whose backing is exactly its reservation: the corridor-filled state.
POOL_ROWS = 10_000

#: Rows a request in flight is holding. This is the true floor: eviction may
#: never touch these.
RESIDENT_CEILING = 1_000

#: The radix tree's ceiling. Recomputable, and therefore spendable.
TREE_CEILING = 9_900


class _Node:
    _next = 0

    def __init__(self, rows, parent=None):
        _Node._next += 1
        self.id = _Node._next
        self.value = torch.tensor(list(rows), dtype=torch.int64)
        self.children = {}
        self.parent = parent
        self.full_lock_ref = 0
        self.mamba_lock_ref = 0
        if parent is not None:
            parent.children[self.id] = self


class _Tree:
    """A flat radix tree holding the recomputable half of the live set."""

    def __init__(self, lo, hi, chunk=100):
        self.root_node = _Node([])
        self.freed = []
        for start in range(lo, hi + 1, chunk):
            _Node(range(start, min(start + chunk, hi + 1)), parent=self.root_node)

    def _delete_leaf(self, node):
        if node.parent is not None:
            node.parent.children.pop(node.id, None)
        node.parent = None

    class _Alloc:
        def __init__(self, tree):
            self.tree = tree

        def free(self, value):
            self.tree.freed.extend(int(v) for v in value)

    @property
    def token_to_kv_pool_allocator(self):
        return _Tree._Alloc(self)


class _FakeAllocator:
    def __init__(self, size):
        self.size = size
        self.page_size = 1
        self._free_listeners = []
        self.free_pages = torch.arange(1, size + 1, dtype=torch.int64)
        self.release_pages = torch.empty((0,), dtype=torch.int64)

    def register_free_listener(self, on_free, on_clear=None):
        self._free_listeners.append((on_free, on_clear))

    def available_size(self):
        return len(self.free_pages) + len(self.release_pages)

    def free(self, idx):
        self.free_pages = torch.cat((self.free_pages, idx))
        for on_free, _ in self._free_listeners:
            on_free(idx)


class _FakePool:
    def __init__(self, rows, bytes_per_row=MIB, card=None):
        self.size = rows
        self.page_size = 1
        self._bytes_per_row = bytes_per_row
        self._card = card
        self.supports_backing_spans = True
        self.calls = []

    def runtime_set_backing_rows(self, rows):
        rows = int(rows)
        self.calls.append(rows)
        released = max(0, self.size - rows) * self._bytes_per_row
        self.size = rows
        if self._card is not None:
            self._card.free += released
        return released


class _Card:
    def __init__(self, free_mib):
        self.free = free_mib * MIB

    def probe(self):
        return self.free


def _corridor_filled_rig(card_free_mib=1024):
    """The exact state the corridor law asks for: pool full, ~1024 MiB free.

    The live set runs to ``TREE_CEILING``, so the mark is 100 rows under the
    reservation and the plain floor (mark + 1 + reserve) is ABOVE it: the
    shipped rung has nothing to give.
    """
    card = _Card(card_free_mib)
    pool = _FakePool(POOL_ROWS, card=card)
    alloc = _FakeAllocator(POOL_ROWS)
    tree = _Tree(RESIDENT_CEILING + 1, TREE_CEILING)

    def live_slots():
        return torch.arange(1, TREE_CEILING + 1, dtype=torch.int64)

    # The side channel the flip's own enumeration publishes: who pins the
    # ceiling. Without it the rung must assume the whole live set is
    # resident, which is the safe direction and funds nothing.
    live_slots.last_split = {
        "tree_max": TREE_CEILING,
        "tree_rows": TREE_CEILING - RESIDENT_CEILING,
        "req_max": RESIDENT_CEILING,
        "req_rows": RESIDENT_CEILING,
    }

    relief = kbr.KvBackingRelief(
        pool,
        alloc,
        live_slots_fn=live_slots,
        bytes_per_row=pool._bytes_per_row,
        probe=card.probe,
        admission_reserve_rows=RESERVE,
        tree_cache_fn=lambda: tree,
    )
    return relief, pool, card, tree


class SeamFundIsEvictableContentTest(unittest.TestCase):
    def setUp(self):
        import os

        self._env = os.environ.get(kbr.KV_RADIX_EVICT_ENV)

    def tearDown(self):
        import os

        if self._env is None:
            os.environ.pop(kbr.KV_RADIX_EVICT_ENV, None)
        else:
            os.environ[kbr.KV_RADIX_EVICT_ENV] = self._env

    def _set(self, value):
        import os

        os.environ[kbr.KV_RADIX_EVICT_ENV] = value

    # -- the can-fail proof --------------------------------------------

    def test_without_the_actuator_a_corridor_filled_pool_funds_nothing(self):
        """THE SHIPPED BEHAVIOUR, and the reason the seam needed free VRAM.

        The mark is 100 rows under the reservation, so the plain floor
        (mark + 1 + 512) is above it and there is no slack to release.
        """
        self._set("0")
        relief, _pool, _card, _tree = _corridor_filled_rig()

        self.assertEqual(
            relief.fundable_bytes(),
            0,
            "with no watermark actuator a corridor-filled pool has nothing "
            "to give: this is exactly why the seam had to be funded from "
            "VRAM held free at rest",
        )

    def test_with_the_actuator_the_same_pool_funds_the_recomputable_half(self):
        """The fix, on the identical pool state. Only the env differs."""
        self._set("1")
        relief, _pool, _card, _tree = _corridor_filled_rig()

        expected_rows = POOL_ROWS - (RESIDENT_CEILING + 1 + RESERVE)
        self.assertEqual(
            relief.fundable_bytes(),
            expected_rows * MIB,
            "the rung may fund everything down to the RESIDENT half of the "
            "live set",
        )
        self.assertGreater(relief.fundable_bytes(), 8_000 * MIB)

    # -- the actuator actually moves the driver ------------------------

    def test_the_shrink_evicts_then_caps_then_unmaps_and_reports_driver_bytes(self):
        self._set("1")
        relief, pool, card, tree = _corridor_filled_rig()
        free_before = card.free

        got = relief.free_up_to(2_000 * MIB)

        self.assertGreaterEqual(got, 2_000 * MIB, "the ask must be covered")
        self.assertEqual(
            got,
            card.free - free_before,
            "reported bytes must equal what the driver's free column moved",
        )
        self.assertTrue(pool.calls, "the pool's backing must actually be resized")
        self.assertLess(pool.size, POOL_ROWS)
        self.assertGreater(relief.evicted_rows_total, 0, "rows were evicted")
        self.assertEqual(relief.evict_count, 1)

    def test_the_resident_half_is_never_evicted(self):
        """No row a request in flight holds may be given up, at any ask."""
        self._set("1")
        relief, pool, _card, tree = _corridor_filled_rig()

        relief.free_up_to(POOL_ROWS * MIB)  # ask for everything

        self.assertTrue(tree.freed, "the recomputable half was spent")
        self.assertGreater(
            min(tree.freed),
            RESIDENT_CEILING,
            "not one resident row may appear in the evicted set",
        )
        self.assertGreaterEqual(
            pool.size,
            RESIDENT_CEILING + 1,
            "backing must still cover every live row",
        )

    def test_an_unknown_resident_half_funds_nothing(self):
        """No split, no eviction. An unknown live set is not an empty one."""
        self._set("1")
        card = _Card(1024)
        pool = _FakePool(POOL_ROWS, card=card)
        alloc = _FakeAllocator(POOL_ROWS)
        tree = _Tree(RESIDENT_CEILING + 1, TREE_CEILING)

        def live_slots():
            return torch.arange(1, TREE_CEILING + 1, dtype=torch.int64)

        # No last_split side channel at all.
        relief = kbr.KvBackingRelief(
            pool,
            alloc,
            live_slots_fn=live_slots,
            bytes_per_row=pool._bytes_per_row,
            probe=card.probe,
            admission_reserve_rows=RESERVE,
            tree_cache_fn=lambda: tree,
        )
        self.assertEqual(
            relief.fundable_bytes(),
            0,
            "without a split the rung cannot tell recomputable rows from "
            "resident ones, and must therefore give up neither",
        )

    def test_no_tree_degrades_to_exactly_the_previous_behaviour(self):
        self._set("1")
        card = _Card(1024)
        pool = _FakePool(POOL_ROWS, card=card)
        alloc = _FakeAllocator(POOL_ROWS)

        def live_slots():
            return torch.arange(1, TREE_CEILING + 1, dtype=torch.int64)

        live_slots.last_split = {
            "tree_max": TREE_CEILING,
            "req_max": RESIDENT_CEILING,
            "tree_rows": 0,
            "req_rows": 0,
        }
        relief = kbr.KvBackingRelief(
            pool,
            alloc,
            live_slots_fn=live_slots,
            bytes_per_row=pool._bytes_per_row,
            probe=card.probe,
            admission_reserve_rows=RESERVE,
        )
        self.assertEqual(relief.fundable_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
