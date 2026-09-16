"""WP8 expert lookahead (slotstream: the router of a LATER MoE block runs on
the current stream and that block's offload cache prefetches the predicted
spill experts while the current block computes).

Desk-provable parts: the sticky slot resolve, the prefetch into free scratch
slots with the holds map kept by every fetch, the merged rendezvous of
run_waves, the block chaining. The overlap itself is a stream property and
is measured on the metal (A/B, SGLANG_MOE_EXPERT_LOOKAHEAD=0/1/2)."""

from collections import namedtuple
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.moe import expert_offload as eo
from sglang.srt.layers.moe.topk import StandardTopKOutput

E, R, C, W = 12, 4, 3, 4  # experts, resident, scratch, row width


def _planner():
    return eo.ExpertResidencyPlanner(num_local_experts=E, resident_count=R, scratch=C)


def test_resolve_sticky_keeps_held_experts_and_fetches_only_the_rest():
    p = _planner()
    holds = {R + 0: 7, R + 1: 9}  # scratch slot -> expert already there
    slot_of, plan, reused = p.resolve_sticky([1, 7, 10, 9], holds)
    assert reused == 2 and plan == [(10, R + 2)]
    assert slot_of == {1: 1, 7: R + 0, 9: R + 1, 10: R + 2}
    assert p.stats.lookahead_hits == 2 and p.stats.lookahead_misses == 1
    assert p.stats.fetches == 1 and p.stats.misses == 1 and p.stats.hits == 1


def test_resolve_sticky_without_holds_is_the_sorted_layout_of_resolve():
    a, b = _planner(), _planner()
    needed = [11, 2, 6, 5]
    sa, fa = a.resolve(needed)
    sb, fb, reused = b.resolve_sticky(needed, {})
    assert (sa, fa) == (sb, fb) and reused == 0


def test_resolve_sticky_refuses_scratch_overflow_like_resolve():
    with pytest.raises(RuntimeError, match="scratch"):
        _planner().resolve_sticky([4, 5, 6, 7], {})


def _cache(monkeypatch, lookahead=True):
    monkeypatch.setenv("SGLANG_MOE_EXPERT_LOOKAHEAD", "1" if lookahead else "0")
    monkeypatch.setenv("SGLANG_MOE_SCRATCH_SLOTS", str(C))
    layer = SimpleNamespace(num_local_experts=E, layer_id=0)
    cache = eo.MoEExpertOffloadCache(layer, R / E)
    assert cache.resident_count == R and cache.scratch == C
    spill = torch.zeros((E - R, W), dtype=torch.int32)
    for row in range(E - R):
        spill[row].fill_(R + row)  # the row IS the expert id
    resident = torch.full((R + C, W), -1, dtype=torch.int32)
    for e in range(R):
        resident[e].fill_(e)
    cache._pinned = {"w13": spill}
    cache._resident = {"w13": resident}
    cache._installed = True
    return cache


def test_prefetch_fills_free_scratch_and_the_real_resolve_reuses_it(monkeypatch):
    cache = _cache(monkeypatch)
    assert cache.prefetch([1, 9, 7, 7]) == 2  # residents ignored, spill uniqued
    assert cache._scratch_holds == {R + 0: 7, R + 1: 9}
    assert int(cache._resident["w13"][R + 0, 0]) == 7
    assert int(cache._resident["w13"][R + 1, 0]) == 9
    assert cache.planner.stats.lookahead_prefetched == 2

    slot_of, plan, reused = cache.planner.resolve_sticky([9, 10, 2], cache._scratch_holds)
    cache._fetch(plan)
    assert reused == 1 and plan == [(10, R + 0)]  # 7's slot was free to take
    assert cache._scratch_holds == {R + 0: 10, R + 1: 9}
    assert int(cache._resident["w13"][slot_of[10], 0]) == 10
    assert int(cache._resident["w13"][slot_of[9], 0]) == 9

    # A prediction already held costs nothing; an over-long one is truncated.
    assert cache.prefetch([9, 10]) == 0
    assert cache.prefetch([4, 5, 6, 7, 8]) == 3
    assert sorted(cache._scratch_holds.values()) == [4, 5, 6]


def test_every_fetch_route_writes_the_holds_so_sticky_can_trust_them(monkeypatch):
    cache = _cache(monkeypatch)
    slot_of, plan = cache.planner.resolve([5, 11])  # the deterministic route
    cache._fetch(plan)
    assert cache._scratch_holds == {R + 0: 5, R + 1: 11}
    cache._scratch_holds.clear()
    assert cache.prefetch([11]) == 1  # cleared holds -> fetched again, no trust


DispatchOutput = namedtuple("DispatchOutput", "hidden_states hidden_states_scale topk_output")


def _dispatch(ids):
    topk_ids = torch.tensor(ids, dtype=torch.int32)
    topk = StandardTopKOutput(
        topk_weights=torch.ones(topk_ids.shape), topk_ids=topk_ids, router_logits=None
    )
    return DispatchOutput(torch.zeros(topk_ids.shape[0], W), None, topk)


def test_run_waves_carries_the_prediction_over_one_rendezvous(monkeypatch):
    this, nxt = _cache(monkeypatch), _cache(monkeypatch)
    seen = []

    def apply_fn(sub):
        seen.append(sub.topk_output.topk_ids.clone())
        return SimpleNamespace(hidden_states=sub.hidden_states)

    pred = torch.tensor([[2, 8, 6]], dtype=torch.int32)
    this.run_waves(_dispatch([[1, 5, 7]]), apply_fn, lookahead=(nxt, pred))
    # this layer resolved its own routing (5 and 7 fetched, 1 resident) ...
    assert this._scratch_holds == {R + 0: 5, R + 1: 7}
    assert seen[0].tolist() == [[1, R + 0, R + 1]]
    # ... and the LATER layer's cache holds the predicted spill experts.
    assert sorted(nxt._scratch_holds.values()) == [6, 8]
    assert nxt.planner.stats.lookahead_prefetched == 2
    assert nxt.planner.stats.forwards == 0  # a prefetch is not a forward

    # The later layer's real forward then pays only for the miss.
    nxt.run_waves(_dispatch([[6, 8, 10]]), apply_fn)
    assert nxt.planner.stats.lookahead_hits == 2 and nxt.planner.stats.fetches == 1


def test_a_prediction_into_an_uninstalled_cache_is_dropped_by_count(monkeypatch):
    this = _cache(monkeypatch)
    nxt = _cache(monkeypatch)
    nxt._installed = False
    this.run_waves(
        _dispatch([[1, 5, 7]]),
        lambda sub: SimpleNamespace(hidden_states=sub.hidden_states),
        lookahead=(nxt, torch.tensor([[2, 8, 6]], dtype=torch.int32)),
    )
    assert this.planner.stats.lookahead_dropped == 1
    assert nxt._scratch_holds == {}


def test_without_the_switch_the_deterministic_layout_is_untouched(monkeypatch):
    cache = _cache(monkeypatch, lookahead=False)
    assert cache.lookahead_sticky is False
    cache._scratch_holds = {R + 0: 11}
    seen = []
    cache.run_waves(
        _dispatch([[11, 5]]),
        lambda sub: (seen.append(sub.topk_output.topk_ids.tolist()) or SimpleNamespace(hidden_states=sub.hidden_states)),
    )
    # resolve(): sorted spill -> 5 at R+0, 11 at R+1, regardless of what was held.
    assert seen[0] == [[R + 1, R + 0]]
    assert cache._scratch_holds == {R + 0: 5, R + 1: 11}


def test_link_moe_lookahead_chains_moe_blocks_at_the_distance(monkeypatch):
    from sglang.srt.models.qwen2_moe import link_moe_lookahead

    def blk():
        return SimpleNamespace(lookahead_next=None, gate=object(), experts=object())

    dense = SimpleNamespace(mlp=SimpleNamespace())  # no gate/experts
    layers = [SimpleNamespace(mlp=blk()) for _ in range(4)]
    layers.insert(2, dense)  # [m0, m1, dense, m2, m3]
    assert link_moe_lookahead(layers, distance=1) == 2  # m0->m1, m2->m3
    assert layers[0].mlp.lookahead_next is layers[1].mlp
    assert layers[1].mlp.lookahead_next is None  # dense breaks the chain
    assert layers[3].mlp.lookahead_next is layers[4].mlp
    for l in layers:
        if hasattr(l.mlp, "lookahead_next"):
            l.mlp.lookahead_next = None
    # distance 2 over [m0, m1, dense, m2, m3]: m0 -> dense (no link),
    # m1 -> m2, m2 -> past the end. One link.
    assert link_moe_lookahead(layers, distance=2) == 1
    assert layers[0].mlp.lookahead_next is None
    assert layers[1].mlp.lookahead_next is layers[3].mlp
    monkeypatch.setenv("SGLANG_MOE_EXPERT_LOOKAHEAD", "0")
    assert link_moe_lookahead(layers) == 0


def test_fused_moe_hands_the_pending_lookahead_to_exactly_one_forward():
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE

    me = SimpleNamespace(_lookahead_pending=None, _expert_offload=SimpleNamespace(planner=SimpleNamespace(stats=eo.ResidencyStats())))
    nxt = SimpleNamespace(_expert_offload="next-cache")
    FusedMoE.set_lookahead(me, nxt, "ids")
    assert FusedMoE._take_lookahead(me) == ("next-cache", "ids")
    assert FusedMoE._take_lookahead(me) is None
    FusedMoE.set_lookahead(me, nxt, "ids")
    FusedMoE._drop_lookahead(me)
    assert me._lookahead_pending is None
    assert me._expert_offload.planner.stats.lookahead_dropped == 1
