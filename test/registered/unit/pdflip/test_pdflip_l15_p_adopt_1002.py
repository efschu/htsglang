# SPDX-License-Identifier: Apache-2.0
"""L15-10 S1: P adopts a delivered hot prefix as device nodes.

The hot D->P handover lands the prefix's KV on P rows [0, n) and its END
anchor on one P mamba row (plan 2.0, L15-10 design S1). P must then:
reserve those rows in both allocators (so nothing else is handed them),
insert the prefix into its radix tree as device nodes with the anchor on the
end node, and leave the pools balanced -- checked with the REAL
SchedulerInvariantChecker._mamba_double_claimed (no slot both free and
cached) on a REAL UnifiedRadixCache (fixture of test_pdflip_l15_tree_rewrite).
A later match of the prefix is a full device hit.
"""

from __future__ import annotations

import importlib.util
import os
from array import array

import torch

from flliper.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker as InvariantChecker,
)
from flliper.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from flliper.srt.pdflip import l15_p_adopt

_HERE = os.path.dirname(__file__)


def _fx():
    spec = importlib.util.spec_from_file_location(
        "test_pdflip_l15_tree_rewrite_1001",
        os.path.join(_HERE, "test_pdflip_l15_tree_rewrite_1001.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m, m._fixture()


def test_adopt_reserves_rows_inserts_device_nodes_and_balances():
    m, fx = _fx()
    ids = list(range(500, 540))
    n = len(ids)
    rows = list(range(1, n + 1))      # P rows [1, n] (row 0 is padding)
    anchor = 2
    res = l15_p_adopt.adopt(
        fx.cache, fx.allocator, fx.pool.mamba_allocator,
        token_ids=ids, rows=rows, anchor_row=anchor)
    assert res.adopted == n
    hit = m._match(fx, ids)
    assert len(hit.device_indices) >= n - 1   # bigram/page alignment
    assert hit.device_indices.tolist() == rows[: len(hit.device_indices)]
    mv = hit.last_device_node.component_data[ComponentType.MAMBA].value
    assert mv is not None and int(mv.flatten()[0]) == anchor
    dup, _ids, shared = InvariantChecker._mamba_double_claimed(
        fx.pool.mamba_allocator, fx.cache)
    assert dup == 0 and shared == 0
    free_kv = set(int(x) for x in fx.allocator.free_pages.tolist()) \
        if hasattr(fx.allocator, "free_pages") else None
    if free_kv is not None:
        kept = set(rows[: len(hit.device_indices)])
        assert not (free_kv & kept), "adopted rows still on the free list"
        # L15-ADOPT-TAIL: a row the (bigram) tree does not file goes back
        assert set(rows) - kept <= free_kv, "an unfiled reserved row leaked"


def test_adopt_refuses_a_row_that_is_not_free():
    m, fx = _fx()
    taken = fx.allocator.alloc(4)
    t = [int(x) for x in taken.tolist()]
    try:
        l15_p_adopt.adopt(fx.cache, fx.allocator, fx.pool.mamba_allocator,
                          token_ids=[1, 2, 3, 4], rows=t, anchor_row=3)
    except l15_p_adopt.L15AdoptRefused as exc:
        assert "not free" in str(exc)
    else:
        raise AssertionError("adopting allocated rows must be refused")


def test_adopt_length_mismatch_refused():
    m, fx = _fx()
    try:
        l15_p_adopt.adopt(fx.cache, fx.allocator, fx.pool.mamba_allocator,
                          token_ids=[1, 2, 3], rows=[1, 2], anchor_row=1)
    except l15_p_adopt.L15AdoptRefused:
        pass
    else:
        raise AssertionError("token/row length mismatch must be refused")


def test_adopting_a_prefix_the_tree_already_holds_leaks_nothing():
    """L15-ADOPT-TAIL: the second adopt of the same prefix finds the chain and
    its state already in the tree: the tree frees the duplicate KV rows
    itself, the adopt gives the unused anchor row back (mamba_exist)."""
    m, fx = _fx()
    ids = list(range(600, 632))
    n = len(ids)
    l15_p_adopt.adopt(fx.cache, fx.allocator, fx.pool.mamba_allocator,
                      token_ids=ids, rows=list(range(1, n + 1)), anchor_row=2)
    rows2 = list(range(n + 1, 2 * n + 1))
    l15_p_adopt.adopt(fx.cache, fx.allocator, fx.pool.mamba_allocator,
                      token_ids=ids, rows=rows2, anchor_row=3)
    free_mb = set(int(x) for x in fx.pool.mamba_allocator.free_slots.tolist())
    assert 3 in free_mb, "the second anchor row leaked"
    free_kv = set(int(x) for x in fx.allocator.free_pages.tolist())
    assert set(rows2) <= free_kv, "duplicate KV rows leaked"
    dup, _ids, shared = InvariantChecker._mamba_double_claimed(
        fx.pool.mamba_allocator, fx.cache)
    assert dup == 0 and shared == 0


def test_unaligned_prefix_is_refused_before_anything_is_reserved():
    m, fx = _fx()
    fx.cache.page_size = 4
    before = fx.allocator.free_pages.clone()
    try:
        l15_p_adopt.adopt(fx.cache, fx.allocator, fx.pool.mamba_allocator,
                          token_ids=[1, 2, 3, 4, 5], rows=[1, 2, 3, 4, 5],
                          anchor_row=1)
    except l15_p_adopt.L15AdoptRefused as exc:
        assert "page 4" in str(exc)
    else:
        raise AssertionError("an unaligned adopt must be refused")
    assert torch.equal(before, fx.allocator.free_pages)


def test_an_adopted_anchor_can_be_released_without_a_double_free_refusal():
    """L15-MAMBA-LEDGER: reserve_mamba_slots marks the #924 ledger, so the
    tree's later release of the adopted anchor is a legitimate free."""
    m, fx = _fx()
    ids = list(range(700, 716))
    l15_p_adopt.adopt(fx.cache, fx.allocator, fx.pool.mamba_allocator,
                      token_ids=ids, rows=list(range(1, 17)), anchor_row=4)
    fx.pool.mamba_allocator.free(torch.tensor([4]))      # must not raise
    assert 4 in set(int(x) for x in fx.pool.mamba_allocator.free_slots.tolist())


def test_bigram_tree_tail_row_goes_back():
    m, fx = _fx()
    ids = list(range(800, 820))
    rows = list(range(1, 21))
    l15_p_adopt.adopt(fx.cache, fx.allocator, fx.pool.mamba_allocator,
                      token_ids=ids, rows=rows, anchor_row=2)
    free_kv = set(int(x) for x in fx.allocator.free_pages.tolist())
    hit = m._match(fx, ids)
    held = set(hit.device_indices.tolist())
    assert set(rows) <= held | free_kv, "a reserved row is in no node and not free"
    assert not (held & free_kv)
