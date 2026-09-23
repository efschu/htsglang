"""Blocker #104 (fnFL2x44-x50, 23.09.): D's first graph verify after a
request's extend died on TP0 with an illegal memory access; an eager verify
round in between healed it (SGLANG_SPEC_EAGER_VERIFY=first), and Form A
without the flip (fnFA22/23) never died.

Mechanism, black-box: the extend is an eager forward under the device-planned
pool. D runs it TOKEN-major (fnFA22/23: expert-major), so each wave fetches its
spill experts into rows [R, R + n_wave) in sorted order; an expert routed in two
waves lands in two different rows, and a row a later, smaller wave does not
reach keeps the earlier copy. ``sync_tables`` mapped both rows to the expert
while ``hot_phys`` names only one. The first decode step after that hits the
expert in one row, evicts the other one (the lowest row goes first), and the
eviction unmaps the expert it just hit: its lanes are routed to -1. The marlin
MoE runs without an expert map there, so a -1 block reads the expert BEFORE
the bank. A one-wave eager verify writes no twin and its sync frees every
other LRU row -- which is why "first" hid it.

The cases: the real eager path (run_waves + sync_pool_from_host of a CPU cache)
into the real first decode step, bytes checked per lane; the step on its own
with a stale twin (defence in the step, Triton program included).
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.layers.moe import expert_pool_device as ep
from sglang.srt.layers.moe.topk import StandardTopKOutput

# residents 0,1 in rows 0,1; scratch C=5 = LRU rows 2..5 + staging row 6
E, R, C, S, W = 10, 2, 5, 1, 4

DispatchOutput = namedtuple("DispatchOutput", "hidden_states hidden_states_scale topk_output")
CombineOutput = namedtuple("CombineOutput", "hidden_states")


def _pool_cache(monkeypatch, keep):
    monkeypatch.setenv("SGLANG_MOE_SCRATCH_SLOTS", str(C))
    monkeypatch.setenv("SGLANG_OPT_MOE_POOL_KEEP_LRU", "1" if keep else "0")
    monkeypatch.setenv("SGLANG_MOE_OFFLOAD_WAVE_ORDER", "token")
    cache = eo.MoEExpertOffloadCache(SimpleNamespace(num_local_experts=E, layer_id=23), R / E)
    assert (cache.resident_count, cache.scratch) == (R, C)
    spill = torch.zeros((E - R, W), dtype=torch.int32)
    for row in range(E - R):
        spill[row].fill_(R + row)  # a row's bytes ARE its expert id
    bank = torch.full((R + C, W), -1, dtype=torch.int32)
    for e in range(R):
        bank[e].fill_(e)
    cache._pinned = {"w13": spill}
    cache._resident = {"w13": bank}
    cache._installed = True
    hot_slot_of, host_row = cache._pool_layout()
    cache._pool_tables = ep.allocate_pool_tables("cpu", E, R + C, R, S, hot_slot_of, host_row)
    cache._pool_buffers = ep.allocate_step_buffers("cpu", E, 8)
    cache._pool_ready = True
    return cache


def _eager_forward(cache, ids):
    """The FusedMoE eager branch under the pool mode, verbatim order."""
    topk_ids = torch.tensor(ids, dtype=torch.int32)
    topk = StandardTopKOutput(
        topk_weights=torch.ones(topk_ids.shape), topk_ids=topk_ids, router_logits=None
    )
    cache.begin_eager_pool()
    cache.run_waves(
        DispatchOutput(torch.zeros(topk_ids.shape[0], W), None, topk),
        lambda sub: CombineOutput(hidden_states=sub.hidden_states),
    )
    cache.sync_pool_from_host()


def _decode_step(cache, ids):
    """The captured decode step on CPU: plan, copy the misses, route."""
    flat = torch.tensor(ids, dtype=torch.int32)
    pairs, _ = ep.step_reference(cache._pool_tables, flat, cache._pool_buffers)
    ep.copy_rows_reference([cache._pinned["w13"]], [cache._resident["w13"]], pairs)
    return cache._pool_buffers.routes[: len(ids)].tolist()


@pytest.mark.parametrize("keep", [False, True], ids=["free-lru", "keep-lru"])
def test_a_multi_wave_extend_never_routes_the_first_decode_away_from_an_experts_bytes(
    monkeypatch, keep
):
    cache = _pool_cache(monkeypatch, keep)
    # wave 1: token 0 needs spill 2..6 -> rows 2..6; wave 2: token 1 needs 5, 7
    # -> rows 2, 3. Row 5 still holds expert 5 from wave 1: a twin of row 2.
    _eager_forward(cache, [[2, 3, 4, 5, 6], [5, 7, 0, 1, 0]])
    # first decode step: hits 5, misses 8 -> the lowest LRU row is the victim
    routes = _decode_step(cache, [5, 8])
    bank = cache._resident["w13"]
    for lane, expert in enumerate([5, 8]):
        row = routes[lane]
        assert row >= 0, f"expert {expert} routed to -1"
        assert int(bank[row, 0]) == expert, f"lane {lane} reads expert {int(bank[row, 0])}"
    assert ep.bijection_breaks(cache._pool_tables) == 0


@pytest.mark.parametrize("keep", [False, True], ids=["free-lru", "keep-lru"])
def test_sync_tables_reports_the_twins_it_freed_and_none_for_one_wave(keep):
    t = ep.allocate_pool_tables("cpu", E, R + C, R, S, {0: 0, 1: 1}, [-1, -1] + list(range(E - R)))
    holds = {2: 5, 3: 7, 4: 4, 5: 5, 6: 6}  # expert 5 in rows 2 and 5
    assert ep.sync_tables(t, holds, keep_unwritten=keep).twins_freed == 1
    assert t.row_key.tolist()[2:6].count(5) == 1
    assert ep.bijection_breaks(t) == 0
    # one wave writes every expert once: nothing to free on a fresh pool
    fresh = ep.allocate_pool_tables("cpu", E, R + C, R, S, {0: 0, 1: 1}, [-1, -1] + list(range(E - R)))
    assert ep.sync_tables(fresh, {2: 5, 3: 7}, keep_unwritten=keep).twins_freed == 0
    # under keep, rewriting 5 into row 3 frees the kept row 5 -- still one owner
    if keep:
        assert ep.sync_tables(t, {3: 5}, keep_unwritten=True).twins_freed == 1
        assert ep.bijection_breaks(t) == 0 and t.hot_phys.tolist()[5] == 3


def test_evicting_a_stale_twin_never_unmaps_the_expert_the_step_hits():
    t = ep.allocate_pool_tables("cpu", E, R + C, R, S, {0: 0, 1: 1}, [-1, -1] + list(range(E - R)))
    b = ep.allocate_step_buffers("cpu", E, 8)
    # expert 5 lives in row 5; row 2 still names it (stale twin), rows 3, 4 hold 7, 4
    for r, e in ((2, 5), (3, 7), (4, 4), (5, 5)):
        t.row_key[r] = e
    t.hot_phys[5], t.hot_phys[7], t.hot_phys[4] = 5, 3, 4
    ep.step_reference(t, torch.tensor([5, 8], dtype=torch.int32), b)
    assert b.routes[:2].tolist() == [5, 2]
    assert t.hot_phys.tolist()[5] == 5 and t.hot_phys.tolist()[8] == 2


def test_the_triton_step_unmaps_only_the_owners_row():
    """The Triton program must carry the same owner check as the reference."""
    import inspect

    src = inspect.getsource(ep._step_kernel)
    evict = src[src.index("old = tl.load(row_key_ptr + victim_row)") :]
    evict = evict[: evict.index("tl.store(hot_phys_ptr + expert")]
    assert "owner = tl.load(hot_phys_ptr + old)" in evict
    assert "if owner == victim_row:" in evict
