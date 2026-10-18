"""S25 (h16v5 1af79968a7, D TP0 03:29:35 node=38 / 03:47:08 node=378; h16v4b 7x, h16v3 23x):
``WEG2-ANCHOR-LOST at=evict ... backuped=False why=arena_claim``.

Root: the K15 save (anchor backup before the device mamba tombstone) took the anchor-only path only for a
``backuped`` node and asked ``write_backup`` for a FULL KV claim otherwise -- also for a store-backed node
(``l3_present``, Full host copy spilled by W3 / Form A reconcile) whose KV the store already holds. That claim
(1062 pages, 6485-slot KV arena full of the tree's own host copies: ``ARENA-CLAIM REFUSED statuses=[0,2,4]``) was
refused, the anchor tombstoned. ``_weg2_anchor_only_candidate`` already admits ``l3_present``; the K15 branch did
not read the mark (the L3-DROP lesson, h16n 1015).
"""
from __future__ import annotations

import logging
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.mem_cache.unified_cache_components.tree_component import ComponentType  # noqa: E402


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_EVICT_ANCHOR_SAVE", raising=False)
    monkeypatch.delenv("SGLANG_WEG2_ANCHOR_ONLY_BACKUP", raising=False)


def _node(backuped=False, l3_present=False):
    cd = types.SimpleNamespace(value=object(), host_value=None)
    return types.SimpleNamespace(id=38, depth=86336, backuped=backuped, l3_present=l3_present,
                                 weg2_anchor_secured=False, _weg2_end_anchor=False,
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
        node.weg2_anchor_secured = not node.backuped and node.l3_present   # detached slot, acked at the join
        if node.backuped:
            node.component_data[ComponentType.MAMBA].host_value = object()
        return 1

    t.write_backup = _full
    t.write_backup_anchor_only = _anchor_only
    t.writing_check = lambda write_back=False: t.calls.append("join")
    t._weg2_mamba_pool = lambda: object()
    t.weg2_node_depth = lambda n: n.depth
    t._weg2_sweep_last_refusal = None
    return t


def _save(t, n):
    return UnifiedRadixCache.weg2_save_anchor_before_tombstone(t, n)


def test_store_backed_node_saves_its_anchor_without_a_kv_claim(caplog):
    """RED before the fix: full_kv_backup refused (arena_claim) -> ANCHOR-LOST. GREEN: anchor_only, joined, saved."""
    t, n = _tree(), _node(l3_present=True)
    with caplog.at_level(logging.INFO):
        assert _save(t, n) is True
    assert t.calls == ["anchor_only", "join"]
    assert any("K15 ANCHOR-SAVED" in m for m in caplog.messages)
    assert not any("WEG2-ANCHOR-LOST" in m for m in caplog.messages)


def test_unbacked_unstored_node_still_takes_the_full_backup():
    t, n = _tree(), _node()
    assert _save(t, n) is False
    assert t.calls == ["full_kv_backup"]


def test_backuped_node_unchanged():
    t, n = _tree(), _node(backuped=True)
    assert _save(t, n) is True and t.calls == ["anchor_only", "join"]


def test_secured_anchor_is_not_saved_again(monkeypatch):
    t, n = _tree(), _node(l3_present=True)
    n.weg2_anchor_secured = True
    monkeypatch.setattr(UnifiedRadixCache, "_weg2_anchor_secured_holds", lambda self, node: True)
    assert _save(t, n) is True and t.calls == []


def test_anchor_only_switch_off_is_the_old_path(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_ANCHOR_ONLY_BACKUP", "0")
    t, n = _tree(), _node(l3_present=True)
    assert _save(t, n) is False and t.calls == ["full_kv_backup"]


def test_anchor_lost_line_names_l3_present(caplog):
    t, n = _tree(), _node()
    with caplog.at_level(logging.WARNING):
        _save(t, n)
    assert any("WEG2-ANCHOR-LOST at=evict" in m and "l3_present=False" in m for m in caplog.messages)
