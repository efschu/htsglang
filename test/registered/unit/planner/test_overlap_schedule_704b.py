"""#704b: pricing the "+40% is overlappable" claim.

DESIGN_704 sec 4.2a inherited the canonical plan's phrasing that the decoupling
collectives are "overlappable behind GDN/FFN compute". Pricing it splits that
into two statements with very different values, and one of them is false:

* the Q-broadcast + partial-gather + LSE merge is ON THE CRITICAL PATH -- layer
  L+1 consumes layer L's attention output, so the GDN layers after L cannot
  hide L's gather because they cannot START;
* KV placement is OFF the critical path -- it is a write no later layer in the
  chunk reads, so the following GDN layers hide it entirely.

The "48 GDN layers to hide behind" intuition counts compute that is
sequentially downstream of the very thing it was supposed to hide. Only
cross-chunk pipelining hides the dominant term, and this file prices what
building it would buy.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest
from sglang.srt.planner.decoupled_kv import KvGeometry, collective_bytes_per_chunk
from sglang.srt.planner.overlap_schedule import (
    OverlapScheduleError,
    solve_overlap,
)

MIB = 1024 * 1024
GEO = KvGeometry(
    num_attention_heads=24,
    head_dim=256,
    num_key_value_heads=4,
    kv_dtype_bytes=1,
    activation_dtype_bytes=2,
    num_attn_layers=16,
)
CHUNK = 512
GATHER_PER_LAYER = collective_bytes_per_chunk(GEO, 2, CHUNK) / GEO.num_attn_layers / MIB
PLACE_PER_LAYER = CHUNK * GEO.kv_bytes_per_token_per_attn_layer / MIB
MS_PER_LAYER = (1.7571, 7.740, 7.275)
FIXED_VECTOR = (0.135, 0.483, 0.382)  # the ladder's fixed free-proportional vector
X4, X8 = 3000.0, 6000.0  # PLACEHOLDER link classes, pending the pair matrix

INCUMBENT = {"attn_per_stage": (7, 5, 4), "layers_per_stage": (28, 20, 16)}
DEEP = {"attn_per_stage": (11, 2, 3), "layers_per_stage": (44, 10, 10)}


def _solve(case, bw=(X4, X8, X8)):
    return solve_overlap(
        ms_per_layer=MS_PER_LAYER,
        link_mib_per_s=bw,
        shares=FIXED_VECTOR,
        gather_mib_per_attn_layer=GATHER_PER_LAYER,
        placement_mib_per_attn_layer=PLACE_PER_LAYER,
        **case,
    )


def test_the_per_layer_volumes_match_the_cost_model():
    assert GATHER_PER_LAYER == pytest.approx(24.09, abs=0.05)
    assert PLACE_PER_LAYER == pytest.approx(1.00, abs=0.01)


def test_placement_hides_entirely_behind_the_following_gdn_layers():
    """The half of the claim that IS true, and it stays true at every rung."""
    for case in (INCUMBENT, DEEP):
        sched = _solve(case)
        for s in sched.stages:
            assert s.placement_exposed_ms == 0.0, (
                f"stage {s.stage} could not hide its placement write behind "
                f"{s.gdn_compute_ms:.1f} ms of GDN compute"
            )
            assert s.gdn_compute_ms > s.placement_ms


def test_the_gather_does_not_hide_within_a_chunk():
    """The half that is FALSE, stated as an equality.

    With no cross-chunk pipelining the exposed time IS the gather time: not a
    fraction of it, all of it. GDN compute downstream of the gather cannot
    reduce it, because it cannot begin until the gather completes.
    """
    sched = _solve(INCUMBENT)
    for s in sched.stages:
        assert s.exposed_no_pipeline_ms == pytest.approx(s.gather_ms)


def test_rank0_is_the_worst_stage_and_it_is_triple_jeopardy():
    """rank0 carries the most attention layers, the least compute to hide
    behind, and the slowest link -- the three disadvantages coincide."""
    for case in (INCUMBENT, DEEP):
        sched = _solve(case)
        assert sched.worst_stage_no_pipeline == 0
        s0 = sched.stages[0]
        assert s0.attn_layers == max(s.attn_layers for s in sched.stages)


def test_cross_chunk_pipelining_is_worth_about_eightfold():
    """The number this module exists to produce: what the mitigation buys."""
    inc = _solve(INCUMBENT)
    assert inc.worst_no_pipeline_ms == pytest.approx(56.2, abs=1.0)
    assert inc.worst_pipelined_ms == pytest.approx(7.0, abs=1.0)
    assert inc.pipelining_speedup > 6.0

    deep = _solve(DEEP)
    assert deep.worst_no_pipeline_ms == pytest.approx(88.3, abs=1.5)
    assert deep.worst_pipelined_ms == pytest.approx(11.0, abs=1.5)
    assert deep.pipelining_speedup > 6.0


def test_deepening_the_cut_makes_the_worst_stage_worse_on_both_sides():
    """Deep cuts move attention layers ONTO the slow-linked fast rank.

    So the traffic rises while the compute available to hide it rises more
    slowly -- and the modelled compute is itself the optimistic end, since
    fixed_ms=0 attributes all measured time to layers.
    """
    inc = _solve(INCUMBENT).stages[0]
    deep = _solve(DEEP).stages[0]
    assert deep.attn_layers > inc.attn_layers
    assert deep.gather_ms > inc.gather_ms
    assert deep.exposed_no_pipeline_ms > inc.exposed_no_pipeline_ms


def test_exposure_scales_inversely_with_the_link():
    fast = _solve(INCUMBENT, bw=(2 * X4, 2 * X8, 2 * X8))
    slow = _solve(INCUMBENT, bw=(X4, X8, X8))
    assert fast.worst_no_pipeline_ms == pytest.approx(
        slow.worst_no_pipeline_ms / 2.0, rel=1e-9
    )


def test_a_fast_enough_link_removes_the_exposure_even_without_pipelining():
    """The other mitigation: the problem is bandwidth, not only scheduling."""
    sched = _solve(INCUMBENT, bw=(24000.0, 48000.0, 48000.0))
    assert sched.worst_pipelined_ms == 0.0
    assert sched.worst_no_pipeline_ms < 10.0


def test_a_stage_without_attention_layers_pays_nothing():
    sched = solve_overlap(
        attn_per_stage=(16, 0, 0),
        layers_per_stage=(40, 12, 12),
        ms_per_layer=MS_PER_LAYER,
        link_mib_per_s=(X4, X8, X8),
        shares=FIXED_VECTOR,
        gather_mib_per_attn_layer=GATHER_PER_LAYER,
        placement_mib_per_attn_layer=PLACE_PER_LAYER,
    )
    assert sched.stages[1].gather_ms == 0.0
    assert sched.stages[1].exposed_no_pipeline_ms == 0.0


def test_malformed_inputs_are_refused():
    with pytest.raises(OverlapScheduleError, match="stages"):
        solve_overlap(
            attn_per_stage=(7, 5, 4),
            layers_per_stage=(28, 20),
            ms_per_layer=MS_PER_LAYER,
            link_mib_per_s=(X4, X8, X8),
            shares=FIXED_VECTOR,
            gather_mib_per_attn_layer=GATHER_PER_LAYER,
            placement_mib_per_attn_layer=PLACE_PER_LAYER,
        )
    with pytest.raises(OverlapScheduleError, match="attention layers out of"):
        solve_overlap(
            attn_per_stage=(30, 5, 4),
            layers_per_stage=(28, 20, 16),
            ms_per_layer=MS_PER_LAYER,
            link_mib_per_s=(X4, X8, X8),
            shares=FIXED_VECTOR,
            gather_mib_per_attn_layer=GATHER_PER_LAYER,
            placement_mib_per_attn_layer=PLACE_PER_LAYER,
        )
    with pytest.raises(OverlapScheduleError, match="link"):
        solve_overlap(
            attn_per_stage=(7, 5, 4),
            layers_per_stage=(28, 20, 16),
            ms_per_layer=MS_PER_LAYER,
            link_mib_per_s=(0.0, X8, X8),
            shares=FIXED_VECTOR,
            gather_mib_per_attn_layer=GATHER_PER_LAYER,
            placement_mib_per_attn_layer=PLACE_PER_LAYER,
        )
