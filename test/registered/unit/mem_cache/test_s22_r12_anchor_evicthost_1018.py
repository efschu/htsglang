"""S22 part 3 (18.10., review nf-review-s22-1018 point 4): the two further TP0 delete paths without worker state.

h16v4b 02:09:44 (4 nodes, seq=326) and 02:18:01: ``K23 SPILL-DIVERGED outcome=unreconcilable`` right after
``MAMBA-ARENA DISPLACE why=flush_spill``: ``_weg2_release_anchor`` dropped the anchor of a device-evicted, childless node
without KV host -- the node is DELETED on TP0, its STATE(exists=0) met a worker that holds it interior/device-resident.
``evict_host`` (``FullComponent.drive_host_eviction``) deletes host leaves by TP0's LRU alone. Both now take only what
every worker can delete; in the SLEEP FLUSH the reset follows on every rank, so the anchor path sends no verdict there.
"""
import os
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(__file__))
from test_w3_arena_spill_forma_dual15_0930 import group  # noqa: E402,F401

from sglang.srt.mem_cache import form_a_host_shadow as R  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.full_component import FullComponent  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import BASE_COMPONENT_TYPE, UnifiedRadixCache  # noqa: E402


class _NS(types.SimpleNamespace):
    __hash__ = object.__hash__


def _node(h, depth, evicted=True, children=None, host=None, nid=1):
    root = _NS(parent=None, key=[], children={})
    n = _NS(
        id=nid, parent=root, key=list(range(depth)), hash_value=[h], evicted=evicted,
        children=children or {}, component_data={BASE_COMPONENT_TYPE: types.SimpleNamespace(host_value=host)})
    return n


class _MP:
    def drop_unreferenced(self, slots):
        return len(slots)


def _release(node, flush, nd, role="host"):
    calls = []
    me = types.SimpleNamespace(
        components={__import__("sglang.srt.mem_cache.unified_cache_components.tree_component",
                               fromlist=["ComponentType"]).ComponentType.MAMBA: object()},
        _weg2_flush_spill=flush,
        _evict_component_and_detach_lru=lambda *a, **k: calls.append("evict") or (0, 0),
        _update_evictable_leaf_sets=lambda n: None)
    st = types.SimpleNamespace(displaced_share=0, displaced_full=0, dropped=0)
    victim = types.SimpleNamespace(node=node, slots=[3], rid="r", depth=7)
    with mock.patch.object(R, "role", return_value=role), \
            mock.patch.object(R, "worker_nodelete_keys", return_value=nd), \
            mock.patch.object(R, "record_state", side_effect=lambda *a, **k: calls.append("state")):
        got = UnifiedRadixCache._weg2_release_anchor(me, victim, _MP(), st, why="full", for_rid="x")
    return got, calls


def setup_function(_f):
    R._reset_state_for_tests()


def test_anchor_release_keeps_a_leaf_a_worker_cannot_delete():
    n = _node("a", 4)
    got, calls = _release(n, flush=False, nd={R.node_key(n)})
    assert got == 0 and calls == [], "TP0 deleted a node a worker keeps (and sent STATE exists=0)"
    assert R._S.counts["kept_total_anchor_release_refused"] == 1     # the backstop; the choice leaves it out first


def test_anchor_release_proceeds_when_every_worker_can_delete():
    n = _node("a", 4)
    got, calls = _release(n, flush=False, nd=set())
    assert got == 1 and calls == ["evict", "state"]


def test_anchor_release_of_a_node_that_survives_is_untouched():
    n = _node("a", 4, host=object())          # it keeps its KV host: the anchor drop does not delete it
    got, calls = _release(n, flush=False, nd={R.node_key(n)})
    assert calls == ["evict", "state"]


def test_in_the_sleep_flush_the_state_is_sent_as_everywhere_no_reset_is_assumed():
    """/flush_cache can end WITHOUT a reset ("quiesced", "Cache not flushed": 3x on h16v4b TP0, two right after
    FLUSH-SPILLs 02:09:44) -- so the flush sends the verdict like every other path, unfiltered."""
    n = _node("a", 4)
    got, calls = _release(n, flush=True, nd={R.node_key(n)})
    assert got == 1 and calls == ["evict", "state"], calls
    assert R._S.counts["kept_total_anchor_flush_reset_covers"] == 1     # classification only


def test_after_a_flush_without_reset_the_worker_follows_the_state(group):
    """The flip of the old reset-covers case: no tree reset after the flush; the worker applies TP0's STATE for the
    deleted node and ends in the same tree (device-evicted leaf there -> deleted)."""
    g = group
    from test_w3_arena_spill_forma_dual15_0930 import FULL as F
    g.n1[7].component_data[F].value = None
    g.t1._evict_host_leaf = lambda node, tracker: g.n1[7].parent.children.pop(tuple(g.n1[7].key), None) \
        or g.n1[7].parent.children.clear()
    k = R.node_key(g.n1[7])
    with mock.patch.object(R, "role", return_value="worker"):
        c = R.apply(g.t1, [(R.STATE, k[0], k[1], 0, 0, 0)], seq=1)
    assert c["deleted"] == 1 and c["unreconcilable"] == 0, c


class _A(types.SimpleNamespace):
    __hash__ = object.__hash__


def _anchor(n, rid, depth):
    return _A(node=n, rid=rid, depth=depth, slots=[depth])


def test_victim_choice_skips_a_protected_anchor_and_takes_the_next_one():
    """Review point 5: the protected victim at the top used to end in ``return 0`` and block every displacement
    (callers do not look for another). The choice itself leaves it out now."""
    R._reset_state_for_tests()
    first, second = _node("a", 4, nid=1), _node("b", 8, nid=2)
    owned = [_anchor(first, "r", 4), _anchor(second, "r", 8)]
    me = types.SimpleNamespace(_weg2_flush_spill=False)
    me._r12_anchor_deletable = lambda a: UnifiedRadixCache._r12_anchor_deletable(me, a)
    me._weg2_anchor_of = None
    with mock.patch.object(R, "role", return_value="host"), \
            mock.patch.object(R, "worker_nodelete_keys", return_value={R.node_key(first)}):
        victim = UnifiedRadixCache._weg2_full_victim(me, types.SimpleNamespace(_weg2_end_anchor=False), "r", owned, None)
    assert victim is owned[1], "the protected first victim blocked the displacement"
    assert R._S.counts["kept_total_anchor_worker_cannot_delete"] == 1


def test_a_worker_or_a_plain_group_is_untouched():
    n = _node("a", 4)
    assert _release(n, flush=False, nd={R.node_key(n)}, role=None)[1] == ["evict"]


def _drive(leaves, nd, role="host"):
    seen = []
    ev = {id(l) for l in leaves}
    cache = types.SimpleNamespace(
        evictable_host_leaves=set(leaves),
        eviction_strategy=types.SimpleNamespace(get_priority=lambda n: n.id),
        _evict_host_leaf=lambda x, tr: (seen.append(x), tr.__setitem__("k", tr["k"] + 1)))
    comp = object.__new__(FullComponent)
    comp.cache = cache
    comp.component_type = "k"
    tracker = {"k": 0}
    with mock.patch.object(R, "role", return_value=role), \
            mock.patch.object(R, "worker_nodelete_keys", return_value=nd):
        comp.drive_host_eviction(10, tracker)
    return seen


def test_evict_host_skips_a_leaf_a_worker_cannot_delete():
    R._reset_state_for_tests()
    a, b = _node("a", 4, nid=1), _node("b", 4, nid=2)
    for n in (a, b):
        n.parent = _NS(parent=None, key=[], children={})
    seen = _drive([a, b], {R.node_key(a)})
    assert seen == [b], seen
    assert R._S.counts["kept_total_evict_host_worker_cannot_delete"] == 1


def test_evict_host_without_worker_state_drops_every_leaf():
    R._reset_state_for_tests()
    a, b = _node("a", 4, nid=1), _node("b", 4, nid=2)
    assert sorted(x.id for x in _drive([a, b], set())) == [1, 2]
    assert [x.id for x in _drive([a, b], {R.node_key(a)}, role=None)] == [1, 2]
