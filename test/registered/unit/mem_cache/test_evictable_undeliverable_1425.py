"""#1425/#1425b (xsn184, xsn185): under write_back an un-backed device node the
staging ring cannot absorb is not deliverable -- and neither is anything above
it, because the peel reaches a parent only after its children are gone."""

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache import unified_radix_cache as urc
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

CT = urc.BASE_COMPONENT_TYPE


def _node(tokens, backed, lock=0, parent=None):
    cds = [types.SimpleNamespace(value=None, host_value=None, lock_ref=0) for _ in range(len(list(urc.ComponentType)))]
    cds[CT] = types.SimpleNamespace(value=[0] * tokens, host_value=[0] * tokens if backed else None, lock_ref=lock)
    n = types.SimpleNamespace(component_data=cds, children={}, parent=parent, backuped=backed)
    if parent is not None:
        parent.children[id(n)] = n
    return n


def _tree(avail, root, policy="write_back"):
    t = object.__new__(UnifiedRadixCache)
    t.root_node = root
    total = 0
    stack = [root]
    while stack:
        n = stack.pop()
        stack.extend(n.children.values())
        cd = n.component_data[CT]
        if n is not root and cd.value is not None and cd.lock_ref == 0:
            total += len(cd.value)
    t.component_evictable_size_ = {CT: total}
    t.cache_controller = types.SimpleNamespace(
        write_policy=policy, mem_pool_host=types.SimpleNamespace(available_size=lambda: avail))
    return t


def _root():
    r = types.SimpleNamespace(children={}, parent=None, backuped=False)
    r.component_data = [types.SimpleNamespace(value=None, host_value=None, lock_ref=0) for _ in range(len(list(urc.ComponentType)))]
    return r


def test_unbacked_leaves_beyond_the_ring_are_not_counted():
    r = _root()
    _node(4096, True, parent=r)
    for _ in range(3):
        _node(4096, False, parent=r)
    t = _tree(avail=5000, root=r)
    assert t.evictable_size() == 16384
    # one un-backed leaf fits the 5000 free rows, two do not
    assert t.full_evictable_size() == 16384 - 2 * 4096


def test_backed_leaf_under_unbacked_parent_counts_but_the_parent_does_not():
    # xsn185 shape: root -> A(un-backed, 4096) -> B(backed, 4096); ring full.
    r = _root()
    a = _node(4096, False, parent=r)
    _node(4096, True, parent=a)
    t = _tree(avail=0, root=r)
    assert t.evictable_size() == 8192
    assert t.full_evictable_size() == 4096


def test_nothing_above_an_undeliverable_node_is_counted():
    # root -> A(backed) -> B(un-backed, ring full) -> C(backed): C peels, B stops, A is unreachable.
    r = _root()
    a = _node(4096, True, parent=r)
    b = _node(4096, False, parent=a)
    _node(1000, True, parent=b)
    t = _tree(avail=0, root=r)
    assert t.full_evictable_size() == 1000


def test_backed_chains_and_write_through_are_untouched():
    r = _root()
    a = _node(4096, True, parent=r)
    _node(4096, True, parent=a)
    assert _tree(avail=0, root=r).full_evictable_size() == 8192
    r2 = _root()
    _node(4096, False, parent=r2)
    assert _tree(avail=0, root=r2, policy="write_through").full_evictable_size() == 4096
