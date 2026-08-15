"""#662 -- id-targeted radix eviction that lowers the KV high-water mark.

RED FIRST. Each test states a property the seam's funding depends on and
would fail against the obvious wrong implementation:

  * LRU FALSIFIER -- evicting by coldness does not lower the mark. The
    whole reason this module exists is that the quantity pinning committed
    backing is max(row id), not the number of live rows, and those two are
    uncorrelated. If eviction here were LRU-ordered this test fails.
  * RESIDENCY FALSIFIER -- a row held by an in-flight request is never
    given up, and a target below the resident ceiling is refused WHOLE
    rather than applied in part.
  * CHAIN CONVERGENCE -- evicting a leaf exposes its parent; the mark must
    still come down.
  * MEASURE == ACT -- what the pricing call promises is what the actuator
    delivers, in the same unit.

CPU-only; no pool, no CUDA, no allocator. The tree is a stub whose shape
is the only thing under test.
"""

import pytest

from sglang.srt.managers.kv_radix_watermark import (
    evict_rows_above,
    evictable_rows_above,
    tree_ceiling,
)


class _Node:
    """A radix node: some rows, maybe children, maybe a lock."""

    _next_id = 0

    def __init__(self, rows, parent=None, lock=0):
        _Node._next_id += 1
        self.id = _Node._next_id
        self.value = list(rows)
        self.children = {}
        self.parent = parent
        self.full_lock_ref = lock
        self.mamba_lock_ref = 0
        if parent is not None:
            parent.children[self.id] = self


class _Tree:
    """Minimal stand-in exposing what the evictor actually uses."""

    def __init__(self):
        self.root_node = _Node([])
        self.root_node.value = []
        self.freed = []

    def _delete_leaf(self, node):
        parent = node.parent
        if parent is not None:
            parent.children.pop(node.id, None)
        node.parent = None

    class _Alloc:
        def __init__(self, tree):
            self.tree = tree

        def free(self, value):
            self.tree.freed.extend(list(value))

    @property
    def token_to_kv_pool_allocator(self):
        return _Tree._Alloc(self)


def _tree(*row_lists, lock_idx=()):
    """A flat tree: one leaf per row list, all children of the root."""
    t = _Tree()
    for i, rows in enumerate(row_lists):
        _Node(rows, parent=t.root_node, lock=1 if i in lock_idx else 0)
    return t


# -- the LRU falsifier ---------------------------------------------------------


def _evict_coldest(tree, num_tokens):
    """What ``tree_cache.evict(num_tokens)`` does: coldest-first, by COUNT.

    Insertion order stands in for last-access order. This is the
    implementation this module exists to NOT be, simulated here so the
    comparison is made by the test rather than asserted in a comment.
    """
    freed = 0
    for node in sorted(_leaves(tree), key=lambda n: n.id):
        if freed >= num_tokens:
            break
        for row in node.value:
            tree.freed.append(row)
        freed += len(node.value)
        tree._delete_leaf(node)
    return freed


def _leaves(tree):
    out, frontier = [], list(tree.root_node.children.values())
    while frontier:
        n = frontier.pop()
        kids = list(n.children.values())
        frontier.extend(kids)
        if not kids:
            out.append(n)
    return out


def test_eviction_is_ordered_by_row_id_not_by_coldness():
    """THE falsifier for this module, stated as a COMPARISON.

    Four cached nodes. The COLDEST (first inserted, an LRU pass's first
    victims) hold the LOWEST rows; the hottest holds the highest. Both
    halves run on identical trees with the same token budget:

      * a coldest-first pass gives up 3x more cache AND leaves the mark
        exactly where it started -- it freed rows nothing was pinned by;
      * the id-targeted pass gives up 2 rows and the mark falls to 8.

    If this module ever regresses to coldness ordering, the second half
    reproduces the first half's numbers and this test fails.
    """
    cold = _tree([0, 1, 2], [3, 4, 5], [6, 7, 8], [900, 901])
    assert tree_ceiling(cold) == 901
    cold_freed = _evict_coldest(cold, 6)
    assert cold_freed == 6, "the coldest pass gave up three whole nodes"
    assert tree_ceiling(cold) == 901, (
        "and the high-water mark DID NOT MOVE -- one hot row at a high id "
        "still pins every page beneath it. This is the defect."
    )

    t = _tree([0, 1, 2], [3, 4, 5], [6, 7, 8], [900, 901])
    freed = evict_rows_above(t, 99)

    assert freed == 2, f"only the 2 rows above the mark are given up, got {freed}"
    assert tree_ceiling(t) == 8, "the mark must come down to the highest kept row"
    assert sorted(t.freed) == [900, 901]
    # The cold, low-id cache is untouched: that is the cache-loss saving
    # the id-targeted order buys over an LRU pass.
    assert 0 not in t.freed and 6 not in t.freed
    assert freed < cold_freed, (
        "the id-targeted pass must lower the mark for STRICTLY less cache "
        "than the coldest-first pass gave up without lowering it at all"
    )


def test_lowering_the_mark_far_gives_up_everything_above_it():
    t = _tree([10, 11], [20, 21], [30, 31])
    freed = evict_rows_above(t, 15)
    assert freed == 4
    assert tree_ceiling(t) == 11


def test_a_mark_already_below_the_target_evicts_nothing():
    t = _tree([1, 2], [3, 4])
    assert evict_rows_above(t, 1000) == 0
    assert t.freed == []
    assert tree_ceiling(t) == 4


# -- the residency falsifier ---------------------------------------------------


def test_a_locked_node_is_never_evicted():
    """A row an in-flight request holds is not this rung's to give up."""
    t = _tree([5, 6], [700, 701], lock_idx=(1,))
    freed = evict_rows_above(t, 10)
    assert freed == 0, "a locked node must survive"
    assert t.freed == []
    assert tree_ceiling(t) == 701


def test_a_target_below_the_resident_ceiling_is_refused_whole():
    """Not partially applied: half a watermark is a fault, not a mode.

    Without this refusal the caller would unmap backing beneath rows that
    are still live -- the pool's backing would not cover its own live set.
    """
    t = _tree([10, 11], [800, 801])
    freed = evict_rows_above(t, 50, resident_ceiling=600)
    assert freed == 0, "nothing may be evicted toward an impossible target"
    assert t.freed == []
    assert tree_ceiling(t) == 801, "the tree is left exactly as it was"


def test_a_target_above_the_resident_ceiling_proceeds():
    t = _tree([10, 11], [800, 801])
    freed = evict_rows_above(t, 700, resident_ceiling=600)
    assert freed == 2
    assert tree_ceiling(t) == 11


# -- chain convergence ---------------------------------------------------------


def test_a_chain_converges_because_evicting_a_leaf_exposes_its_parent():
    """The mark can be pinned by a node that is not a leaf yet.

    root -> A(rows 500) -> B(rows 501) -> C(rows 502). Only C is a leaf, so
    a single pass frees 502 and stops. The mark must still reach 499.
    """
    t = _Tree()
    a = _Node([500], parent=t.root_node)
    b = _Node([501], parent=a)
    _Node([502], parent=b)
    assert tree_ceiling(t) == 502

    freed = evict_rows_above(t, 499)

    assert freed == 3, "the whole chain above the target comes down"
    assert tree_ceiling(t) == -1
    assert sorted(t.freed) == [500, 501, 502]


def test_a_chain_stops_at_a_locked_ancestor():
    """Convergence must not walk through a resident request's node."""
    t = _Tree()
    a = _Node([500], parent=t.root_node, lock=1)
    _Node([501], parent=a)

    freed = evict_rows_above(t, 499)

    assert freed == 1, "only the unlocked descendant is given up"
    assert tree_ceiling(t) == 500, "the locked node still pins the mark"


# -- measurement matches action ------------------------------------------------


def test_the_price_quoted_is_the_price_paid():
    """``evictable_rows_above`` is what the gate prices the rung with.

    If it over- or under-states what ``evict_rows_above`` then frees, the
    seam is funded against a number that was never true -- the failure this
    chain's own docstrings call out as having shipped repeatedly.
    """
    t = _tree([1, 2, 3], [400, 401], [402, 403, 404])

    rows, nodes = evictable_rows_above(t, 100)
    assert (rows, nodes) == (5, 2)

    freed = evict_rows_above(t, 100)
    assert freed == rows, "the actuator delivers exactly what was priced"


def test_pricing_is_pure_and_evicts_nothing():
    t = _tree([1, 2], [500, 501])
    before = tree_ceiling(t)
    rows, nodes = evictable_rows_above(t, 10)
    assert (rows, nodes) == (2, 1)
    assert tree_ceiling(t) == before, "pricing must not touch residency"
    assert t.freed == []


def test_an_empty_tree_prices_and_evicts_zero():
    t = _Tree()
    assert tree_ceiling(t) == -1
    assert evictable_rows_above(t, 0) == (0, 0)
    assert evict_rows_above(t, 0) == 0


def test_no_tree_at_all_is_not_an_error():
    assert evict_rows_above(None, 0) == 0
