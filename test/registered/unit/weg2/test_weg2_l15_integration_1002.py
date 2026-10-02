# SPDX-License-Identifier: Apache-2.0
"""L15-17 (hermetic integration): one sleep -> P phase -> wake across the
real retain/manifest/restore/decide/act code of three D ranks, fakes only
for the pools and the tree. Scenario of the N4a death (09:22:56Z): the
cap-0 rank's (empty) arm succeeds and it retains, the capped ranks' arms
fail (their manifests discarded, plain flush), REFILL is off -- the wake
must leave every rank's tree EMPTY (no rank keeps held chains whose slots
the restore just freed) and decide uniformly."""

from __future__ import annotations

import importlib.util
import os

from sglang.srt.weg2 import l15_manifest, l15_restore

_HERE = os.path.dirname(__file__)


def _chain():
    spec = importlib.util.spec_from_file_location(
        "test_weg2_l15_chain_1001", os.path.join(_HERE, "test_weg2_l15_chain_1001.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_n4a_shape_cap0_retained_capped_arms_failed_refill_off(monkeypatch, tmp_path):
    c = _chain()
    c._env(monkeypatch, tmp_path, mib="c1=64,c2=64")
    monkeypatch.delenv("SGLANG_WEG2_L15_REFILL", raising=False)
    # sleep: only the cap-0 rank keeps its manifest (the capped ranks' keep
    # arm failed -> manifest discarded -> plain flush, tree reset)
    c._sleep_all(tmp_path, (0,), os.getpid())
    scheds = {r: c._sched() for r in range(3)}
    scheds[0]._l15_tree_retained = True        # what the sleep hook records
    scheds[1]._l15_tree_retained = False
    scheds[2]._l15_tree_retained = False
    fss = {r: c._fake_self(scheds[r], r) for r in range(3)}
    # wake: every rank's restore, then the group vote and the act
    for r in range(3):
        assert c.WU._weg2_wake_restore_pools(fss[r]) is True
    votes = [c._hold_votes(fss[r]) for r in range(3)]
    gc = l15_restore.group_check(votes)
    assert votes == [None, None, None] and gc.verdict == "none"
    for r in range(3):
        c.WU._l15_wake_act(fss[r], scheds[r], gc.verdict, group_ok=True,
                           master_on=True)
    # the cap-0 rank dropped its held chains with the pools; the capped
    # ranks' trees were already empty from their plain flush
    assert scheds[0].tree_cache.resets == 1
    assert scheds[1].tree_cache.resets == 0 and scheds[2].tree_cache.resets == 0
    assert all(getattr(scheds[r], "_l15_tree_retained") is False for r in range(3))
    for r in range(3):
        free = {int(x) for x in scheds[r].token_to_kv_pool_allocator.free_pages.tolist()}
        assert free == set(range(1, c.ALLOC_SIZE + 1))
    # the record was consumed by the wake on the cap-0 rank
    assert not os.path.exists(l15_manifest.manifest_path("D", 0, os.environ))


def test_all_ranks_retained_cap0_without_refill_falls_back_uniformly(monkeypatch, tmp_path):
    c = _chain()
    c._env(monkeypatch, tmp_path, mib="c1=64,c2=64")
    monkeypatch.delenv("SGLANG_WEG2_L15_REFILL", raising=False)
    c._sleep_all(tmp_path, (0, 1, 2), os.getpid())
    scheds = {r: c._sched() for r in range(3)}
    for r in range(3):
        scheds[r]._l15_tree_retained = True
    fss = {r: c._fake_self(scheds[r], r) for r in range(3)}
    for r in range(3):
        assert c.WU._weg2_wake_restore_pools(fss[r]) is True
    votes = [c._hold_votes(fss[r]) for r in range(3)]
    gc = l15_restore.group_check(votes)
    assert votes[0] is None and votes[1] is not None and gc.verdict == "fallback"
    for r in range(3):
        c.WU._l15_wake_act(fss[r], scheds[r], gc.verdict, group_ok=True,
                           master_on=True)
    # rank 0 dropped at the restore (no hold here) and again, idempotently,
    # in the fallback act; ranks 1/2 drop in the act
    assert [scheds[r].tree_cache.resets for r in range(3)] == [2, 1, 1]
    for r in range(3):
        free = {int(x) for x in scheds[r].token_to_kv_pool_allocator.free_pages.tolist()}
        assert free == set(range(1, c.ALLOC_SIZE + 1))


def test_agreed_no_hold_wake_keeps_every_tree_for_the_parked_reads(monkeypatch, tmp_path):
    """N4f W50 class: the sleep agreement turned the round off on EVERY rank
    (nobody retained); the wake with verdict none must not reset any tree --
    the #248 hold reads of the parked requests (head + store credit) live
    there (#1455)."""
    c = _chain()
    c._env(monkeypatch, tmp_path, mib="c1=64,c2=64")
    scheds = {r: c._sched() for r in range(3)}
    for r in range(3):
        scheds[r]._l15_tree_retained = False        # agreement: nobody holds
    fss = {r: c._fake_self(scheds[r], r) for r in range(3)}
    for r in range(3):
        assert c.WU._weg2_wake_restore_pools(fss[r]) is True
    votes = [c._hold_votes(fss[r]) for r in range(3)]
    gc = l15_restore.group_check(votes)
    assert gc.verdict == "none"
    for r in range(3):
        c.WU._l15_wake_act(fss[r], scheds[r], gc.verdict, group_ok=True,
                           master_on=True)
    assert [scheds[r].tree_cache.resets for r in range(3)] == [0, 0, 0]
