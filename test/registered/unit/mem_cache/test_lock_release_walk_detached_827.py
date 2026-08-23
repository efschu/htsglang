"""#827 -- a lock-release walk over a detached node must not die on None.

MEASURED, boot_826_review_0823_0912, 2026-08-23 08:55:45, all three ranks at
once, ~1 minute after ready:

    cache_finished_req -> dec_lock_ref
    -> full_component.py:239  `if cur.id in skip_lock_node_ids`
    AttributeError: 'NoneType' object has no attribute 'id'

`release_component_lock` walks `cur = node` up to `self.cache.root_node` with
`while cur != root`. The loop's only exit is reaching root, so a node whose
ancestor chain has been broken -- detached, evicted, or replaced under a
rebuilt root -- walks PAST the top, `cur` becomes None, and the next
dereference throws. The traceback names a NoneType attribute, which says
nothing about the tree; the actual fault is a broken parent chain several
frames earlier.

WHAT THIS FIXES AND WHAT IT DOES NOT. This does not make a broken chain
legal. It makes it DIAGNOSABLE: a named error that says which node, which
component, and how far the walk got, instead of an AttributeError that
implicates the wrong line. A silent `break` would be worse than the crash --
it would leave `component_protected_size_` permanently overcounted and turn a
loud fault into a slow leak.

(The specific breakage that produced this specimen was #825's own tree reset
at the cutover, now off by default. This guard is independent of that: any
detach/evict racing a lock release reaches the same line.)
"""

import pytest

from sglang.srt.mem_cache.unified_cache_components.full_component import FullComponent


class _CD:
    def __init__(self, value=(1, 2), lock_ref=1):
        self.value = list(value)
        self.lock_ref = lock_ref
        self.host_value = None
        self.host_lock_ref = 0


class _Node:
    _next = 0

    def __init__(self, parent=None, ct=None, lock_ref=1):
        _Node._next += 1
        self.id = _Node._next
        self.parent = parent
        self.component_data = {ct: _CD(lock_ref=lock_ref)}


class _Cache:
    def __init__(self, root):
        self.root_node = root
        self.component_evictable_size_ = {}
        self.component_protected_size_ = {}
        self.updated = []

    def _update_evictable_leaf_sets(self, node):
        self.updated.append(node)


class _Self:
    """Stands in for the bound FullComponent; the walk touches only these."""

    def __init__(self, cache, ct):
        self.cache = cache
        self.component_type = ct


CT = FullComponent.component_type


def _chain(depth, lock_ref=1):
    """root <- n1 <- n2 ... ; returns (root, leaf)."""
    root = _Node(None, CT, lock_ref=lock_ref)
    cur = root
    for _ in range(depth):
        cur = _Node(cur, CT, lock_ref=lock_ref)
    return root, cur


def _run(root, leaf):
    cache = _Cache(root)
    for k in (CT,):
        cache.component_evictable_size_[k] = 0
        cache.component_protected_size_[k] = 0
    me = _Self(cache, CT)
    FullComponent.release_component_lock(me, node=leaf, params=None)
    return cache


def test_an_intact_chain_still_releases_every_node():
    """The regression arm: normal walks must be untouched."""
    root, leaf = _chain(3, lock_ref=1)
    cache = _run(root, leaf)
    # three non-root nodes, each dropped from 1 to 0
    cur, n = leaf, 0
    while cur is not root:
        assert cur.component_data[CT].lock_ref == 0
        cur = cur.parent
        n += 1
    assert n == 3
    assert cache.component_protected_size_[CT] == -6  # 3 nodes x len(value)=2


def test_a_detached_node_raises_a_NAMED_error_not_an_AttributeError():
    """THE CRASH PATTERN, reproduced exactly: the chain ends before root."""
    root, leaf = _chain(3)
    # Break the chain the way an evict/detach does: an ancestor loses its
    # parent while the leaf still holds a lock ref.
    leaf.parent.parent.parent = None

    with pytest.raises(Exception) as e:
        _run(root, leaf)
    assert not isinstance(e.value, AttributeError), (
        "the walk still dies on None.id -- this is the un-fixed crash"
    )
    msg = str(e.value)
    assert "parent chain" in msg.lower()
    # It must name the node it started from and the component, so the next
    # occurrence is diagnosable from the log line alone.
    assert str(leaf.id) in msg
    assert "FULL" in msg.upper()


def test_the_named_error_reports_how_far_the_walk_got():
    """Depth is the difference between 'detached leaf' and 'root replaced'."""
    root, leaf = _chain(4)
    leaf.parent.parent = None  # break two hops up
    with pytest.raises(Exception) as e:
        _run(root, leaf)
    assert "2" in str(e.value), str(e.value)


def test_it_does_not_silently_swallow_the_break():
    """A `break` would leave protected_size overcounted forever -- a loud
    fault traded for a slow leak. The walk must not simply return."""
    root, leaf = _chain(3)
    leaf.parent.parent.parent = None
    with pytest.raises(Exception):
        _run(root, leaf)


def test_a_node_that_IS_the_root_is_a_no_op():
    root, _ = _chain(0)
    cache = _run(root, root)
    assert cache.updated == []
