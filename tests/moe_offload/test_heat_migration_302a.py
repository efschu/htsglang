# SPDX-License-Identifier: Apache-2.0
"""#302a -- hermetic falsifier for dynamic expert heat migration.

No CUDA (runs under ``CUDA_VISIBLE_DEVICES=99``), no model, no weights beyond a
handful of tiny CPU tensors. Four families:

1. **The policy is right.** ``plan_heat_swaps`` -- equal counts, coldest-first
   victims, hottest-first candidates, pinned never demoted, delegated never
   promoted, hysteresis, determinism.
2. **The lift is real, and the test can fail.** A synthetic router stream whose
   hot experts sit OUTSIDE the initial resident set. With migration on the hit
   rate rises; with the identical stream and migration off it does not. The
   can-fail arm is the second half of the same test, not a separate claim.
3. **The move is byte-exact and VRAM-neutral.** The executor runs over real
   CPU tensors: after a swap, every expert's bytes are findable, unchanged, at
   its new address; buffer shapes and total bytes are identical before/after.
4. **The refusals fire.** Graph capture, and a migration attempted after the
   capturable buffers were built.

Run: python -m pytest tests/moe_offload/test_heat_migration_302a.py -q
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from sglang.srt.layers.moe.expert_heat_migration import (  # noqa: E402
    HeatMigrationConfig,
    HeatMigrationStats,
    HeatWindow,
    plan_heat_swaps,
    refuse_heat_migration_under_graph_capture,
)


# --------------------------------------------------------------------------- #
# 1. the policy
# --------------------------------------------------------------------------- #
def test_equal_counts_is_structural():
    # Every promotion is paired with exactly one demotion, so residency size
    # cannot move. This is the #439 sizing latch's invariant.
    heat = {0: 1.0, 1: 1.0, 2: 100.0, 3: 90.0, 4: 80.0}
    swaps = plan_heat_swaps(heat, resident_ids=[0, 1], max_swaps=8)
    assert all(len(p) == 2 for p in swaps)
    resident = {0, 1}
    for hot, cold in swaps:
        resident.discard(cold)
        resident.add(hot)
    assert len(resident) == 2


def test_victims_are_coldest_first_and_candidates_hottest_first():
    heat = {0: 5.0, 1: 1.0, 2: 3.0, 7: 100.0, 8: 50.0}
    swaps = plan_heat_swaps(heat, resident_ids=[0, 1, 2], max_swaps=2)
    # hottest candidate 7 displaces coldest resident 1; then 8 displaces 2.
    assert swaps == [(7, 1), (8, 2)]


def test_pinned_expert_is_never_demoted():
    # id 3 is the #82 pad expert: hot by construction, and demoting it would
    # make every foreign token a miss.
    heat = {3: 1000.0, 0: 1.0, 9: 500.0}
    swaps = plan_heat_swaps(
        heat, resident_ids=[0, 3], pinned=frozenset({3}), max_swaps=4
    )
    assert [c for _, c in swaps] == [0]
    assert 3 not in [c for _, c in swaps]


def test_pinned_only_resident_set_yields_no_swaps():
    heat = {3: 10.0, 9: 500.0}
    swaps = plan_heat_swaps(heat, resident_ids=[3], pinned=frozenset({3}), max_swaps=4)
    assert swaps == []


def test_delegated_expert_is_never_promoted():
    # #394: its bytes live in a peer's segment, so this rank has no local pool
    # row to put the displaced victim into.
    st = HeatMigrationStats()
    heat = {0: 1.0, 9: 500.0, 8: 400.0}
    swaps = plan_heat_swaps(
        heat,
        resident_ids=[0, 1],
        delegated=frozenset({9}),
        max_swaps=4,
        stats=st,
    )
    assert [h for h, _ in swaps] == [8]
    assert st.skipped_delegated == 1


def test_hysteresis_blocks_a_marginal_swap():
    st = HeatMigrationStats()
    # candidate is 10 % hotter, margin demands 25 %.
    swaps = plan_heat_swaps(
        {0: 100.0, 9: 110.0},
        resident_ids=[0],
        hysteresis=0.25,
        min_gain=0.0,
        stats=st,
    )
    assert swaps == []
    assert st.skipped_hysteresis == 1
    # ... and passes once it clears the margin.
    assert plan_heat_swaps(
        {0: 100.0, 9: 126.0}, resident_ids=[0], hysteresis=0.25, min_gain=0.0
    ) == [(9, 0)]


def test_absolute_min_gain_blocks_a_noise_swap_the_relative_margin_lets_through():
    """The tail-of-the-distribution case the relative margin cannot see."""
    st = HeatMigrationStats()
    heat = {0: 3.0, 9: 6.0}  # 100 % hotter, and two activations of noise
    assert (
        plan_heat_swaps(heat, resident_ids=[0], hysteresis=0.25, min_gain=8.0, stats=st)
        == []
    )
    assert st.skipped_hysteresis == 1
    # The same relative margin with the floor off does take it.
    assert plan_heat_swaps(heat, resident_ids=[0], hysteresis=0.25, min_gain=0.0) == [
        (9, 0)
    ]


def test_max_swaps_bounds_the_burst():
    heat = {i: 1.0 for i in range(4)}
    heat.update({10 + i: 100.0 - i for i in range(4)})
    swaps = plan_heat_swaps(heat, resident_ids=list(range(4)), max_swaps=2)
    assert len(swaps) == 2


def test_plan_is_deterministic_under_ties():
    heat = {0: 1.0, 1: 1.0, 8: 50.0, 9: 50.0}
    a = plan_heat_swaps(heat, resident_ids=[0, 1], max_swaps=2)
    b = plan_heat_swaps(
        dict(reversed(list(heat.items()))), resident_ids=[1, 0], max_swaps=2
    )
    assert a == b == [(8, 0), (9, 1)]


def test_zero_max_swaps_is_a_no_op():
    assert plan_heat_swaps({0: 1.0, 9: 99.0}, resident_ids=[0], max_swaps=0) == []


# --------------------------------------------------------------------------- #
# 2. the lift is real -- synthetic router stream, with its can-fail arm
# --------------------------------------------------------------------------- #
def _router_stream(num_experts, hot_ids, steps, tokens=8, top_k=4, seed=1234):
    """Deterministic stream that routes ~3/4 of its mass into ``hot_ids``."""
    import random

    rng = random.Random(seed)
    cold = [e for e in range(num_experts) if e not in set(hot_ids)]
    out = []
    for _ in range(steps):
        rows = []
        for _t in range(tokens):
            row = []
            for k in range(top_k):
                pool = hot_ids if k < top_k - 1 else cold
                row.append(rng.choice(pool))
            rows.append(row)
        out.append(rows)
    return out


def _replay(stream, num_experts, resident_count, cfg, migrate=True):
    """Run the stream through the window; return (hit_rate, swaps).

    The residency here is a plain set, so this exercises the POLICY against a
    known-answer stream without any tensors -- the executor gets its own test
    below.
    """
    resident = set(range(resident_count))
    window = HeatWindow(cfg)
    hits = misses = 0
    swaps_done = 0
    for rows in stream:
        for row in rows:
            for e in row:
                if e in resident:
                    hits += 1
                else:
                    misses += 1
        window.observe(rows, frozenset(resident), resident_count)
        if migrate and window.due():
            for hot, cold in window.plan(resident):
                resident.discard(cold)
                resident.add(hot)
                swaps_done += 1
            window.close_round()
        elif window.forwards_since_round >= cfg.period_forwards:
            window.close_round()
    total = hits + misses
    return hits / total, swaps_done


HOT_OUTSIDE = [40, 41, 42, 43, 44, 45]
SIM_ARGS = dict(num_experts=64, resident_count=8)


def test_migration_lifts_hit_rate_on_a_synthetic_stream():
    stream = _router_stream(64, HOT_OUTSIDE, steps=400)
    cfg = HeatMigrationConfig(
        enabled=True,
        period_forwards=20,
        decay=0.5,
        hysteresis=0.1,
        min_gain=8.0,
        max_swaps=3,
    )
    rate_on, swaps = _replay(stream, cfg=cfg, **SIM_ARGS)
    assert swaps > 0, "no migration happened -- the arm did not run"
    # The hot set is 6 of 64 experts and holds 3/4 of the mass; a resident set
    # of 8 that reaches them should clear 0.7.
    assert rate_on > 0.70, rate_on


def test_can_fail_the_same_stream_without_migration_stays_low():
    """The can-fail arm. Identical stream, identical residency, migration off.

    If this ever rose to the migrating arm's number, the test above would be
    measuring the stream and not the feature.
    """
    stream = _router_stream(64, HOT_OUTSIDE, steps=400)
    cfg = HeatMigrationConfig(
        enabled=True,
        period_forwards=20,
        decay=0.5,
        hysteresis=0.1,
        min_gain=8.0,
        max_swaps=3,
    )
    rate_off, swaps = _replay(stream, cfg=cfg, migrate=False, **SIM_ARGS)
    assert swaps == 0
    assert rate_off < 0.30, rate_off
    rate_on, _ = _replay(stream, cfg=cfg, **SIM_ARGS)
    assert rate_on - rate_off > 0.40, (rate_on, rate_off)


def test_disabled_config_never_becomes_due():
    stream = _router_stream(64, HOT_OUTSIDE, steps=100)
    cfg = HeatMigrationConfig(enabled=False, period_forwards=1)
    window = HeatWindow(cfg)
    for rows in stream:
        window.observe(rows, frozenset(range(8)), 8)
        assert not window.due()


def _thrash_trace(hysteresis, min_gain, steps=600, period=20):
    stream = _router_stream(64, HOT_OUTSIDE, steps=steps)
    cfg = HeatMigrationConfig(
        enabled=True,
        period_forwards=period,
        decay=0.5,
        hysteresis=hysteresis,
        min_gain=min_gain,
        max_swaps=3,
    )
    resident = set(range(8))
    window = HeatWindow(cfg)
    per_round = []
    for rows in stream:
        window.observe(rows, frozenset(resident), 8)
        if window.due():
            plan = window.plan(resident)
            for hot, cold in plan:
                resident.discard(cold)
                resident.add(hot)
            per_round.append(len(plan))
            window.close_round()
    return per_round, resident


def test_hysteresis_stops_thrash_on_a_stationary_stream():
    """Once the hot set is resident, a stationary stream must go quiet.

    The claim is specifically about the tail: the hot set converges early and
    then stays put, rather than the resident set churning forever on sampling
    noise among the near-identical cold experts.
    """
    per_round, resident = _thrash_trace(hysteresis=0.25, min_gain=8.0)
    assert per_round, "no rounds ran"
    assert set(HOT_OUTSIDE) <= resident, resident
    # Front-loaded: the two opening rounds place the hot set, and the remaining
    # 28 rounds of the same stationary stream are near-silent.
    assert sum(per_round[:2]) == 6, per_round
    tail = per_round[len(per_round) // 2 :]
    assert sum(tail) <= 2, per_round


def test_can_fail_without_the_absolute_floor_the_tail_keeps_churning():
    """Can-fail arm for the anti-thrash claim.

    Identical stream, identical relative margin, absolute floor removed. The
    tail is then anything but quiet -- which is what makes the assertion above
    a property of the POLICY and not of the workload. Measured on the fixed
    seed: 9 swaps total / 1 in the tail with the floor, 42 / 20 without.
    """
    with_floor, res_a = _thrash_trace(hysteresis=0.25, min_gain=8.0)
    without, res_b = _thrash_trace(hysteresis=0.25, min_gain=0.0)
    tail_a = sum(with_floor[len(with_floor) // 2 :])
    tail_b = sum(without[len(without) // 2 :])
    assert tail_b >= 10, without
    assert tail_b > 5 * max(tail_a, 1), (with_floor, without)
    assert sum(without) > 3 * sum(with_floor), (with_floor, without)
    # Both converge on the hot set; only the churn differs.
    assert set(HOT_OUTSIDE) <= res_a
    assert set(HOT_OUTSIDE) <= res_b


def test_window_hit_rate_is_recorded_per_round():
    stream = _router_stream(64, HOT_OUTSIDE, steps=60)
    cfg = HeatMigrationConfig(
        enabled=True, period_forwards=20, max_swaps=3, min_gain=0.0
    )
    window = HeatWindow(cfg)
    resident = set(range(8))
    for rows in stream:
        window.observe(rows, frozenset(resident), 8)
        if window.due():
            for hot, cold in window.plan(resident):
                resident.discard(cold)
                resident.add(hot)
            window.close_round()
    assert window.stats.rounds == 3
    assert 0.0 <= window.stats.last_window_hit_rate <= 1.0


def test_decay_zero_forgets_and_decay_one_remembers():
    cfg0 = HeatMigrationConfig(enabled=True, period_forwards=1, decay=0.0)
    w0 = HeatWindow(cfg0)
    w0.observe([[5, 5, 5]], frozenset(), 0)
    w0.close_round()
    assert w0.heat == {}

    cfg1 = HeatMigrationConfig(enabled=True, period_forwards=1, decay=1.0)
    w1 = HeatWindow(cfg1)
    w1.observe([[5, 5, 5]], frozenset(), 0)
    w1.close_round()
    assert w1.heat == {5: 3.0}


def test_padding_ids_are_not_counted_as_heat():
    w = HeatWindow(HeatMigrationConfig(enabled=True))
    w.observe([[-1, -1, 3]], frozenset({3}), 8)
    assert w.heat == {3: 1.0}
    assert w.stats.window_hit_activations == 1
    assert w.stats.window_miss_activations == 0


# --------------------------------------------------------------------------- #
# 3. the executor -- byte-exact and VRAM-neutral, over real tensors
# --------------------------------------------------------------------------- #
torch = pytest.importorskip("torch")

from sglang.srt.layers.moe.expert_offload import (  # noqa: E402
    MoEExpertOffloadCache,
    plan_load_time_staging,
)


@pytest.fixture
def cpu_pin(monkeypatch):
    """``pin_memory()`` needs a CUDA context; these tests run without one.

    Same shim ``test_link_proportional_shards.py`` uses. Page-locking is a
    property of the host allocation, not of the swap logic under test.
    """
    monkeypatch.setattr(torch.Tensor, "pin_memory", lambda self: self, raising=True)


class _FakeLayer:
    """Enough of a FusedMoE for install() to split a full [E] tensor."""

    layer_id = 0

    def __init__(self, E, row, dtype=torch.float32):
        self.num_local_experts = E
        # A distinct, recognisable byte pattern per expert, so a misplaced row
        # is identifiable rather than merely unequal.
        self.w13_weight = (
            torch.arange(E, dtype=dtype).view(E, 1, 1).expand(E, *row).contiguous()
        )
        self.w2_weight = (
            torch.arange(E, dtype=dtype).view(E, 1, 1).expand(E, *row).contiguous()
            + 1000.0
        )
        self.moe_runner_config = None


def _build_cache(E=16, R=6, row=(2, 3)):
    layer = _FakeLayer(E, row)
    cache = MoEExpertOffloadCache(layer, fraction=R / E)
    assert cache.resident_count == R, cache.resident_count
    cache.install()
    return layer, cache


def _expert_bytes(cache, attr, expert_id):
    """Where expert ``expert_id``'s row lives right now, and its content."""
    resident_ids, resident_slot, pool_index = cache._current_layout_maps()
    if expert_id in resident_ids:
        return cache._resident[attr][resident_slot[expert_id]].clone()
    return cache._pinned[attr][pool_index[expert_id]].clone()


def test_swap_moves_the_bytes_and_nothing_else(cpu_pin):
    layer, cache = _build_cache()
    E, R = cache.num_local_experts, cache.resident_count
    before = {
        attr: {e: _expert_bytes(cache, attr, e) for e in range(E)}
        for attr in ("w13_weight", "w2_weight")
    }
    cache._heat = HeatWindow(HeatMigrationConfig(enabled=True))
    resident_ids, resident_slot, pool_index = cache._current_layout_maps()
    swaps = [(R + 1, 0), (R + 3, 2)]
    cache._apply_heat_swaps(swaps, resident_ids, resident_slot, pool_index)

    for attr in ("w13_weight", "w2_weight"):
        for e in range(E):
            assert torch.equal(_expert_bytes(cache, attr, e), before[attr][e]), (
                attr,
                e,
            )
    # And the promoted/demoted experts genuinely changed tier.
    assert R + 1 in cache.planner.resident_ids
    assert R + 3 in cache.planner.resident_ids
    assert 0 not in cache.planner.resident_ids
    assert 2 not in cache.planner.resident_ids


def test_swap_is_vram_neutral_byte_exact(cpu_pin):
    layer, cache = _build_cache()
    R = cache.resident_count

    def footprint():
        return {
            attr: (
                tuple(cache._resident[attr].shape),
                cache._resident[attr].numel() * cache._resident[attr].element_size(),
                tuple(cache._pinned[attr].shape),
                cache._pinned[attr].numel() * cache._pinned[attr].element_size(),
            )
            for attr in cache._resident
        }

    before = footprint()
    before_ptrs = {a: cache._resident[a].data_ptr() for a in cache._resident}
    cache._heat = HeatWindow(HeatMigrationConfig(enabled=True))
    ids, slot, pool = cache._current_layout_maps()
    cache._apply_heat_swaps([(R + 1, 0)], ids, slot, pool)

    assert footprint() == before
    # Not merely the same size -- the SAME buffer. A reallocation would be
    # VRAM-neutral in the report and a transient double peak in reality.
    assert {a: cache._resident[a].data_ptr() for a in cache._resident} == before_ptrs
    assert len(cache.planner.resident_ids) == R


def test_residency_count_is_invariant_over_many_rounds(cpu_pin):
    layer, cache = _build_cache(E=24, R=8)
    R = cache.resident_count
    cache._heat = HeatWindow(HeatMigrationConfig(enabled=True))
    for _ in range(6):
        ids, slot, pool = cache._current_layout_maps()
        cand = sorted(set(range(24)) - set(ids))[:2]
        vict = sorted(ids)[:2]
        cache._apply_heat_swaps(list(zip(cand, vict)), ids, slot, pool)
        assert len(cache.planner.resident_ids) == R
        assert len(cache._spill_pool_index) == 24 - R
        assert sorted(cache.planner.resident_slot.values()) == list(range(R))
        assert sorted(cache._spill_pool_index.values()) == list(range(24 - R))


def test_every_expert_stays_addressable_exactly_once(cpu_pin):
    layer, cache = _build_cache(E=20, R=7)
    cache._heat = HeatWindow(HeatMigrationConfig(enabled=True))
    ids, slot, pool = cache._current_layout_maps()
    cache._apply_heat_swaps([(9, 1), (14, 3)], ids, slot, pool)
    ids, slot, pool = cache._current_layout_maps()
    assert set(slot) | set(pool) == set(range(20))
    assert not (set(slot) & set(pool))


def test_fetch_reads_the_right_row_after_a_migration(cpu_pin):
    """The map install is what makes the swap visible to the fetch path."""
    layer, cache = _build_cache(E=16, R=6)
    cache._heat = HeatWindow(HeatMigrationConfig(enabled=True))
    ids, slot, pool = cache._current_layout_maps()
    cache._apply_heat_swaps([(7, 0)], ids, slot, pool)
    # Expert 0 is now cold. Fetch it into a scratch slot and check the bytes.
    scratch_slot = cache.resident_count
    cache._fetch([(0, scratch_slot)])
    got = cache._resident["w13_weight"][scratch_slot]
    assert torch.equal(got, torch.full_like(got, 0.0))
    # And expert 7 is resident at the slot expert 0 vacated.
    _, slot2, _ = cache._current_layout_maps()
    res = cache._resident["w13_weight"][slot2[7]]
    assert torch.equal(res, torch.full_like(res, 7.0))


def test_migration_round_end_to_end_through_run_waves_entry_point(cpu_pin):
    """The whole chain: observe -> due -> plan -> execute -> maps installed."""
    layer, cache = _build_cache(E=16, R=4)
    cfg = HeatMigrationConfig(
        enabled=True,
        period_forwards=1,
        decay=1.0,
        hysteresis=0.0,
        min_gain=0.0,
        max_swaps=2,
        min_observations=1,
    )
    cache._heat = HeatWindow(cfg)
    rows = [[9, 9, 9, 12], [9, 12, 12, 12]]
    cache._heat.observe(rows, cache.planner.resident_ids, cache.resident_count)
    assert cache._heat.due()
    cache._migrate_heat()
    assert {9, 12} <= set(cache.planner.resident_ids)
    assert cache._heat.stats.swaps == 2
    assert cache._heat.stats.rounds == 1
    assert cache._heat.stats.rounds_migrating == 1
    assert cache._heat.stats.h2d_bytes > 0
    assert cache._heat.stats.d2h_bytes == cache._heat.stats.h2d_bytes


def test_pinned_experts_published_by_the_staging_plan_survive_migration(cpu_pin):
    plan = plan_load_time_staging(16, fraction=0.4, pinned_experts=(15,))
    assert plan is not None
    assert plan.pinned_ids == (15,)
    assert 15 in plan.resident_ids

    layer, cache = _build_cache(E=16, R=6)
    layer._moe_offload_pinned_experts = [15]
    # Make 15 resident first (it is cold in the plain [0,R) layout).
    ids, slot, pool = cache._current_layout_maps()
    cache._heat = HeatWindow(
        HeatMigrationConfig(
            enabled=True,
            period_forwards=1,
            min_observations=1,
            hysteresis=0.0,
            min_gain=0.0,
            max_swaps=4,
            decay=1.0,
        )
    )
    cache._apply_heat_swaps([(15, 0)], ids, slot, pool)
    # Now route heavily to a cold expert; 15 must not be the victim even though
    # it is the coldest resident.
    cache._heat.observe([[8] * 8], cache.planner.resident_ids, cache.resident_count)
    cache._migrate_heat()
    assert 15 in cache.planner.resident_ids


# --------------------------------------------------------------------------- #
# 4. refusals
# --------------------------------------------------------------------------- #
def test_refused_under_graph_capture():
    from sglang.srt.environ import envs

    cfg = HeatMigrationConfig(enabled=True)
    with envs.SGLANG_MOE_OFFLOAD_CUDA_GRAPH.override(True):
        with pytest.raises(RuntimeError, match="SGLANG_MOE_OFFLOAD_CUDA_GRAPH"):
            refuse_heat_migration_under_graph_capture(cfg)


def test_not_refused_when_disabled():
    from sglang.srt.environ import envs

    cfg = HeatMigrationConfig(enabled=False)
    with envs.SGLANG_MOE_OFFLOAD_CUDA_GRAPH.override(True):
        refuse_heat_migration_under_graph_capture(cfg)  # must not raise


def test_migration_after_capturable_buffers_is_refused(cpu_pin):
    layer, cache = _build_cache()
    cache._heat = HeatWindow(HeatMigrationConfig(enabled=True))
    cache._capturable_ready = True
    ids, slot, pool = cache._current_layout_maps()
    with pytest.raises(RuntimeError, match="capturable"):
        cache._apply_heat_swaps([(7, 0)], ids, slot, pool)


def test_default_env_leaves_the_feature_off(cpu_pin):
    cfg = HeatMigrationConfig.from_env()
    assert cfg.enabled is False
    layer, cache = _build_cache()
    assert cache._heat is None


def test_stats_dict_shape_is_stable():
    st = HeatMigrationStats()
    d = st.as_dict()
    assert set(d) == {
        "rounds",
        "rounds_migrating",
        "swaps",
        "promoted",
        "demoted",
        "skipped_hysteresis",
        "skipped_delegated",
        "skipped_cap",
        "h2d_bytes",
        "d2h_bytes",
        "last_window_hit_rate",
    }


def test_expert_stats_snapshot_carries_the_counters():
    from sglang.srt.layers.moe.expert_stats import LayerExpertStats

    s = LayerExpertStats(layer_id=3, num_experts=16, resident_count=6)
    assert "heat_migration" not in s.snapshot()
    s.heat_migration = HeatMigrationStats(rounds=2, swaps=5)
    snap = s.snapshot()
    assert snap["heat_migration"]["rounds"] == 2
    assert snap["heat_migration"]["swaps"] == 5


# --------------------------------------------------------------------------- #
# 5. output byte-identity across a migration -- the whole chain
# --------------------------------------------------------------------------- #
def _dispatch(rows, hidden_dim=4):
    """A minimal StandardDispatchOutput over CPU tensors."""
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    T = len(rows)
    K = len(rows[0])
    g = torch.Generator().manual_seed(7)
    hidden = torch.randn(T, hidden_dim, generator=g)
    tw = torch.rand(T, K, generator=g)
    tid = torch.tensor(rows, dtype=torch.int32)
    return StandardDispatchOutput(
        hidden_states=hidden,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=tw, topk_ids=tid, router_logits=None
        ),
    )


def _apply_fn(layer):
    """Deterministic stand-in for quant_method.apply.

    Reads the RESIDENT BUFFER by slot id -- exactly what a real kernel does --
    so a migration that installed a wrong map, or moved the wrong bytes, shows
    up as a different output rather than as a passing test.
    """
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    def apply(dispatch_output):
        hs = dispatch_output.hidden_states
        tw = dispatch_output.topk_output.topk_weights
        tid = dispatch_output.topk_output.topk_ids
        w = layer.w13_weight  # [R+C, ...] -- indexed by SLOT
        out = torch.zeros_like(hs)
        for t in range(hs.shape[0]):
            for k in range(tid.shape[1]):
                slot = int(tid[t, k])
                if slot < 0:
                    continue
                out[t] += hs[t] * float(w[slot].reshape(-1)[0]) * float(tw[t, k])
        return StandardCombineInput(hidden_states=out)

    return apply


def test_output_is_byte_identical_across_a_migration(cpu_pin):
    """The load-bearing identity claim: same bytes computed, only location moves.

    Same routing, same weights, same reduction order -- once before any
    migration and once after the resident set has been permuted. The two
    outputs must be bit-for-bit equal, because a slot permutation changes which
    address an expert's weights are read from and nothing else.
    """
    rows = [[0, 7], [2, 9], [7, 2], [1, 0]]
    layer, cache = _build_cache(E=16, R=6)
    disp = _dispatch(rows)
    before = cache.run_waves(disp, _apply_fn(layer)).hidden_states.clone()

    cache._heat = HeatWindow(HeatMigrationConfig(enabled=True))
    ids, slot, pool = cache._current_layout_maps()
    cache._apply_heat_swaps([(7, 0), (9, 1)], ids, slot, pool)
    after = cache.run_waves(disp, _apply_fn(layer)).hidden_states

    assert torch.equal(before, after), (before, after)
    # ... and the migration really did happen (otherwise the equality is vacuous).
    assert {7, 9} <= set(cache.planner.resident_ids)
    assert not ({0, 1} & set(cache.planner.resident_ids))


def test_can_fail_output_identity_detects_a_corrupted_swap(cpu_pin):
    """Can-fail arm for the identity claim.

    Deliberately install the permuted MAPS without moving the BYTES. The chain
    then reads the previous occupant of each slot, and the same comparison that
    passes above fails -- so the test above is sensitive to exactly the failure
    mode it claims to exclude.
    """
    rows = [[0, 7], [2, 9], [7, 2], [1, 0]]
    layer, cache = _build_cache(E=16, R=6)
    disp = _dispatch(rows)
    before = cache.run_waves(disp, _apply_fn(layer)).hidden_states.clone()

    ids, slot, pool = cache._current_layout_maps()
    for hot, cold in [(7, 0), (9, 1)]:
        i = slot.pop(cold)
        j = pool.pop(hot)
        slot[hot] = i
        pool[cold] = j
        ids = (set(ids) - {cold}) | {hot}
    cache.planner.resident_ids = frozenset(ids)
    cache.planner.resident_slot = slot
    cache._spill_pool_index = pool

    after = cache.run_waves(disp, _apply_fn(layer)).hidden_states
    assert not torch.equal(before, after), "maps-only swap went undetected"
