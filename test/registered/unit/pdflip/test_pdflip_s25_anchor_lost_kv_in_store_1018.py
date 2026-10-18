"""S25 (h16v5 1af79968a7, D TP0 03:29:35 node=38 / 03:47:08 node=378; h16v4b 7x, h16v3 23x):
``PDFLIP-ANCHOR-LOST at=evict ... backuped=False why=arena_claim``.

Root: the K15 save (anchor backup before the device mamba tombstone) took the anchor-only path only for a
``backuped`` node and asked ``write_backup`` for a FULL KV claim otherwise -- also for a store-backed node
(``l3_present``, Full host copy spilled by W3 / Form A reconcile) whose KV the store already holds. That claim
(1062 pages, 6485-slot KV arena full of the tree's own host copies: ``ARENA-CLAIM REFUSED statuses=[0,2,4]``) was
refused, the anchor tombstoned. ``_pdflip_anchor_only_candidate`` already admits ``l3_present``; the K15 branch did
not read the mark (the L3-DROP lesson, h16n 1015).
"""
from __future__ import annotations

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("FLLIPER_PDFLIP_EVICT_ANCHOR_SAVE", raising=False)
    monkeypatch.delenv("FLLIPER_PDFLIP_ANCHOR_ONLY_BACKUP", raising=False)


def _node(backuped=False, l3_present=False):
    cd = types.SimpleNamespace(value=object(), host_value=None)
    return types.SimpleNamespace(id=38, depth=86336, backuped=backuped, l3_present=l3_present,
                                 pdflip_anchor_secured=False, _pdflip_end_anchor=False,
                                 write_through_pending_id=None,
                                 component_data={ComponentType.MAMBA: cd})


def _tree(kv_arena_full=True):
    t = types.SimpleNamespace()
    t.cache_controller = types.SimpleNamespace(write_policy="write_back")
    t.calls = []

    def _full(node, write_back=False):
        t.calls.append("full_kv_backup")
        t._k15_last_refusal = "arena_claim" if kv_arena_full else None
        return 0

    def _anchor_only(node):
        t.calls.append("anchor_only")
        node.pdflip_anchor_secured = not node.backuped and node.l3_present   # detached slot, acked at the join
        if node.backuped:
            node.component_data[ComponentType.MAMBA].host_value = object()
        return 1

    t.write_backup = _full
    t.write_backup_anchor_only = _anchor_only
    t.writing_check = lambda write_back=False: t.calls.append("join")
    t._pdflip_mamba_pool = lambda: object()
    t.pdflip_node_depth = lambda n: n.depth
    t._pdflip_sweep_last_refusal = None
    return t


def _save(t, n):
    return UnifiedRadixCache.pdflip_save_anchor_before_tombstone(t, n)


def test_store_backed_node_saves_its_anchor_without_a_kv_claim(caplog):
    """RED before the fix: full_kv_backup refused (arena_claim) -> ANCHOR-LOST. GREEN: anchor_only, joined, saved."""
    t, n = _tree(), _node(l3_present=True)
    with caplog.at_level(logging.INFO):
        assert _save(t, n) is True
    assert t.calls == ["anchor_only", "join"]
    assert any("K15 ANCHOR-SAVED" in m for m in caplog.messages)
    assert not any("PDFLIP-ANCHOR-LOST" in m for m in caplog.messages)


def test_unbacked_unstored_node_still_takes_the_full_backup():
    t, n = _tree(), _node()
    assert _save(t, n) is False
    assert t.calls == ["full_kv_backup"]


def test_backuped_node_unchanged():
    t, n = _tree(), _node(backuped=True)
    assert _save(t, n) is True and t.calls == ["anchor_only", "join"]


def test_secured_anchor_is_not_saved_again(monkeypatch):
    t, n = _tree(), _node(l3_present=True)
    n.pdflip_anchor_secured = True
    monkeypatch.setattr(UnifiedRadixCache, "_pdflip_anchor_secured_holds", lambda self, node: True)
    assert _save(t, n) is True and t.calls == []


def test_anchor_only_switch_off_is_the_old_path(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ANCHOR_ONLY_BACKUP", "0")
    t, n = _tree(), _node(l3_present=True)
    assert _save(t, n) is False and t.calls == ["full_kv_backup"]


def test_anchor_lost_line_names_l3_present(caplog):
    t, n = _tree(), _node()
    with caplog.at_level(logging.WARNING):
        _save(t, n)
    assert any("PDFLIP-ANCHOR-LOST at=evict" in m and "l3_present=False" in m for m in caplog.messages)


# --- Teil 2: the W3 spill's aux release keeps the anchor findable (h16v5 03:46:05 node=378) ---

from flliper.srt.mem_cache import unified_radix_cache as _urc  # noqa: E402


def _spill_tree(held):
    mamba_comp = types.SimpleNamespace(component_type=ComponentType.MAMBA)
    t = types.SimpleNamespace(_components_tuple=(mamba_comp,))
    t.released = []
    t._evict_component_and_detach_lru = lambda node, comp, target=None, tracker=None: (
        t.released.append(node.id), setattr(node.component_data[ComponentType.MAMBA], "host_value", None))
    t._pdflip_mamba_pool = lambda: types.SimpleNamespace(anchor_held=lambda h: held)
    return t


def _hosted_node():
    n = _node(l3_present=True)
    cd = n.component_data[ComponentType.MAMBA]
    cd.host_value = object()
    cd.host_lock_ref = 0
    n.hash_value = [1, 2, 3]
    return n


def test_spill_release_of_the_host_anchor_marks_it_secured_when_the_slot_holds():
    """RED before: the mark stayed False -> K15 claimed a new slot / tombstoned (ANCHOR-LOST node=378)."""
    t, n = _spill_tree(True), _hosted_node()
    assert _urc._evict_aux_host_before_full_host(t, n, None) == 1
    assert n.pdflip_anchor_secured is True
    t2 = _tree()
    assert _save(t2, n) is True and t2.calls == []        # nothing claimed, the anchor is held


def test_spill_release_sets_no_mark_when_the_slot_is_gone_or_unknown():
    for held in (False, None):
        t, n = _spill_tree(held), _hosted_node()
        _urc._evict_aux_host_before_full_host(t, n, None)
        assert n.pdflip_anchor_secured is False


def test_save_after_an_unmarked_release_still_tries_anchor_only():
    t2, n = _tree(), _hosted_node()
    n.component_data[ComponentType.MAMBA].host_value = None
    assert _save(t2, n) is True and t2.calls == ["anchor_only", "join"]


# --- Teil 3: the once-per-node spill latch is cleared by a successful save (mamba_claim losses) ---

from flliper.srt.mem_cache.unified_cache_components.tree_component import ComponentType as _CT  # noqa: E402,F401


def _spill_tree_mamba(refusals):
    t = types.SimpleNamespace()
    t.cache_controller = types.SimpleNamespace(write_policy="write_back")
    t.calls, seq = [], list(refusals)

    def _ao(node):
        t.calls.append("anchor_only")
        if seq.pop(0):
            node.component_data[ComponentType.MAMBA].host_value = object()
            return 1
        t._k15_last_refusal = "mamba_claim"
        return 0

    t.write_backup = t.write_backup_anchor_only = _ao
    t.writing_check = lambda write_back=False: None
    t._pdflip_mamba_pool = lambda: object()
    t._pdflip_flush_spill_room = lambda node, mp: (t.calls.append("spill"), True)[1]
    t._pdflip_evict_spill_room = types.MethodType(UnifiedRadixCache._pdflip_evict_spill_room, t)
    t.pdflip_node_depth = lambda n: n.depth
    t._pdflip_sweep_last_refusal = None
    return t


def test_a_second_save_of_the_same_node_may_spill_again_after_the_first_succeeded():
    """RED before: 1st save: refused, spill, saved (latch stays); anchor lost again later; 2nd save: no spill -> LOST."""
    n = _node(backuped=True)
    t = _spill_tree_mamba([0, 1])
    assert _save(t, n) is True and t.calls == ["anchor_only", "spill", "anchor_only"]
    n.component_data[ComponentType.MAMBA].host_value = None      # loaded back, device state evicted again
    t.calls.clear()
    t2 = _spill_tree_mamba([0, 1])
    assert _save(t2, n) is True
    assert t2.calls == ["anchor_only", "spill", "anchor_only"]


def test_a_failing_node_still_spills_only_once():
    n = _node(backuped=True)
    t = _spill_tree_mamba([0, 0, 0, 0])
    _save(t, n)
    _save(t, n)
    assert t.calls.count("spill") == 1


# --- Teil 4 (review nf-review-s25-1018): the mark alone is no proof; store-aware backups ---

def test_mark_set_but_kv_not_in_store_gives_no_early_return(monkeypatch):
    """Review 1: pdflip_anchor_secured=True on a node whose KV is on the device only (not backuped, not l3_present)."""
    monkeypatch.setattr(UnifiedRadixCache, "_pdflip_anchor_secured_holds", lambda self, node: True)
    t, n = _tree(), _node()          # not backuped, not l3_present
    n.pdflip_anchor_secured = True
    assert _save(t, n) is False
    assert t.calls == ["full_kv_backup"]


def test_unbacked_anchors_counts_a_marked_node_without_store_kv(monkeypatch):
    monkeypatch.setattr(UnifiedRadixCache, "_pdflip_anchor_secured_holds", lambda self, node: True)

    def _tree_with(node):
        node.children, node.parent = {}, None
        node.component_data[ComponentType.MAMBA].value = object()
        root = types.SimpleNamespace(children={1: node})
        t = types.SimpleNamespace(root_node=root, pdflip_node_depth=lambda n: n.depth,
                                  _pdflip_anchor_secured_holds=lambda n: True)
        t.__dict__.setdefault("_pdflip_anchor_only_detached", set())
        return t

    lost = UnifiedRadixCache.pdflip_unbacked_anchors(_tree_with(_node()))
    assert [d for d, _ in lost] == [86336]
    assert UnifiedRadixCache.pdflip_unbacked_anchors(_tree_with(_node(l3_present=True))) == []


def _bs_tree():
    t = types.SimpleNamespace(calls=[])
    t.write_backup = lambda node: t.calls.append("full") or 1
    t.write_backup_anchor_only = lambda node: t.calls.append("anchor_only") or 1
    t.pdflip_node_depth = lambda n: n.depth
    return t


def test_store_backed_node_asks_no_full_backup_from_the_hit_path():
    """Review 3: RED before (write_backup for every un-backed node, 93376 tokens at D.log:99080)."""
    t = _bs_tree()
    UnifiedRadixCache._s25_backup_store_aware(t, _node(l3_present=True))
    assert t.calls == ["anchor_only"]
    t = _bs_tree()
    UnifiedRadixCache._s25_backup_store_aware(t, _node())
    assert t.calls == ["full"]


def test_hit_count_and_pin_reissue_route_through_the_store_aware_backup():
    import inspect
    assert "_s25_backup_store_aware(node)" in inspect.getsource(UnifiedRadixCache._inc_hit_count)
    assert "_s25_backup_store_aware(node)" in inspect.getsource(UnifiedRadixCache.release_acked_anchor_pin)
