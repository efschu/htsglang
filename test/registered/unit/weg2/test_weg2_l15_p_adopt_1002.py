# SPDX-License-Identifier: Apache-2.0
"""L15-10 S1: P adopts a delivered hot prefix as device nodes.

The hot D->P handover lands the prefix's KV on P rows [0, n) and its END
anchor on one P mamba row (plan 2.0, L15-10 design S1). P must then:
reserve those rows in both allocators (so nothing else is handed them),
insert the prefix into its radix tree as device nodes with the anchor on the
end node, and leave the pools balanced -- checked with the REAL
SchedulerInvariantChecker._mamba_double_claimed (no slot both free and
cached) on a REAL UnifiedRadixCache (fixture of test_weg2_l15_tree_rewrite).
A later match of the prefix is a full device hit.
"""

from __future__ import annotations

import importlib.util
import os
from array import array

import torch

from sglang.srt.managers.scheduler_components.invariant_checker import (
    SchedulerInvariantChecker as InvariantChecker,
)
from sglang.srt.mem_cache.unified_cache_components.tree_component import (
    ComponentType,
)
from sglang.srt.weg2 import l15_p_adopt

_HERE = os.path.dirname(__file__)


def _fx():
    spec = importlib.util.spec_from_file_location(
        "test_weg2_l15_tree_rewrite_1001",
        os.path.join(_HERE, "test_weg2_l15_tree_rewrite_1001.py"))
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
        assert not (free_kv & set(rows)), "adopted rows still on the free list"


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
