# SPDX-License-Identifier: Apache-2.0
"""ORPHAN-OWN (02.10.): the #1424g orphan give-back counts only the slots this
process holds a reference on.

N6d (..._ec4d492f58_1002_170955), 'PDFLIP-TREE-RESET-SUB': orphans 7-9 ms on PP0,
12.2 ms on PP1/PP2 -- dense torch passes (zeros, clone, subtract, clamp,
nonzero) over all 720896 slots of each arena, although orphan = max(held -
named, 0) is 0 wherever this process holds nothing. Switch
FLLIPER_PDFLIP_ORPHAN_OWN_SLOTS (default on; 0 = the dense form).

DANGER DIRECTION: a different give-back than the dense form (a reference
released that a holder still names, or an orphan kept). Pinned: the same
released slots and counts on the #1424g specimens and on a randomized ledger.
Real ``_reset_full`` / ``_pdflip_release_orphan_refs`` on the #1424g arena shell.
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest
import torch

_spec = importlib.util.spec_from_file_location(
    "_h1424g_orphan_own", Path(__file__).with_name("test_arena_reset_orphans_1424g.py"))
H = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(H)

arena = H.arena  # the fixture


@pytest.fixture(autouse=True)
def _no_deferred_census(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_RESET_CENSUS_DEFER_S", "0")


def _run(arena, monkeypatch, own: str, build):
    monkeypatch.setenv("FLLIPER_PDFLIP_ORPHAN_OWN_SLOTS", own)
    t = build(arena)
    before = list(H._refs(arena))
    t._reset_full()
    return before, list(H._refs(arena)), H._own(arena), dict(t._pdflip_reset_orphans or {})


def _specimen(arena):
    t, _pool, _slots = H._parked_two_rids_same_prefix(arena)
    return t


@pytest.mark.parametrize("own", ["1", "0"])
def test_the_1424g_specimen_leaves_no_reference_either_way(arena, monkeypatch, own):
    _before, after, held, _ = _run(arena, monkeypatch, own, _specimen)
    assert held == 0 and sum(after) == 0, (own, after)


def test_red_nothing_held_means_no_naming_walk(arena, monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_ORPHAN_OWN_SLOTS", "1")
    t, _pool, _slots = H._parked_two_rids_same_prefix(arena)
    calls = []
    orig = type(t)._pdflip_name_holders
    t._pdflip_name_holders = lambda a, n: (calls.append(1), orig(t, a, n))[1]
    with arena._ledger.lock:
        arena._ledger.held.zero_()
    assert t._pdflip_release_orphan_refs("reset") == 0
    assert calls == [], "the holders were walked although this process holds nothing"


def test_own_equals_dense_on_a_randomized_ledger(arena, monkeypatch, tmp_path):
    """Random references this process holds, some named by holders (carrier
    hold, a queued row, a tree node), some not: both forms give back exactly
    the same slots, as often."""
    from flliper.srt.mem_cache.storage.file.hicache_arena import ShmArena

    rng = random.Random(7)
    results = []
    for own in ("0", "1"):
        a = ShmArena(str(tmp_path / f"arena-rand-{own}.bin"), H.SLOT_BYTES, H.SLOTS)
        try:
            stems = [f"s{i}" for i in range(H.SLOTS)]
            slot_of = H._publish(a, stems)
            rng.seed(7)
            held = {k: rng.randint(0, 3) for k in stems}
            for k, n in held.items():
                for _ in range(n):
                    a.ref_slots_np(np.asarray([slot_of[k]], dtype=np.int64), +1)
            pool = H._pool(a)
            t, FULL = H._tree(a, pool)
            named = [k for k in stems if held[k] and rng.random() < 0.5]
            t.cache_controller.host_mem_release_queue.put(H._rows([slot_of[k] for k in named]))
            t.root_node = H._node(FULL, 0, None, {})
            monkeypatch.setenv("FLLIPER_PDFLIP_ORPHAN_OWN_SLOTS", own)
            # the queue is cleared by the controller reset before the orphan pass, so
            # name the same rows from a holder that survives it: the carrier hold
            t._pdflip_carrier_rows = {id(pool): (pool, [H._rows([slot_of[k] for k in named])])}
            t.cache_controller.host_mem_release_queue.queue.clear()
            given = t._pdflip_release_orphan_refs("reset")
            results.append((given, list(H._refs(a)), H._own(a)))
        finally:
            a.close()
    assert results[0] == results[1], results
    assert results[0][0] > 0, "the specimen must carry orphans"


def test_the_switch_reads_like_the_others(monkeypatch):
    from flliper.srt.mem_cache import unified_radix_cache as urc

    monkeypatch.delenv("FLLIPER_PDFLIP_ORPHAN_OWN_SLOTS", raising=False)
    assert urc._pdflip_orphan_own_on() is True
    monkeypatch.setenv("FLLIPER_PDFLIP_ORPHAN_OWN_SLOTS", "0")
    assert urc._pdflip_orphan_own_on() is False


def test_counts_at_matches_a_dense_bincount():
    from flliper.srt.mem_cache import unified_radix_cache as urc

    g = torch.Generator().manual_seed(3)
    n = 1000
    ns = urc._PdFlipNamedSlots(n)
    dense = torch.zeros(n, dtype=torch.int64)
    for _ in range(5):
        s = torch.randint(0, n, (50,), generator=g)
        ns.index_add_(0, s, torch.ones(50, dtype=torch.int64))
        dense.index_add_(0, s, torch.ones(50, dtype=torch.int64))
    own = torch.unique(torch.randint(0, n, (200,), generator=g))
    assert torch.equal(ns.counts_at(own), dense[own])
    assert ns.counts_at(torch.zeros(0, dtype=torch.int64)).numel() == 0
