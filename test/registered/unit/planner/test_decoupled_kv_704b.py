"""#704b: PP-KV decoupling -- cost model and vector feasibility, solved.

Decoupling token-shards the 16 full-attention layers' KV across all ranks, and
the owning stage computes attention over the distributed pool by Q-broadcast +
partial attention + LSE merge. Two questions decide whether it is worth
building, and both are arithmetic that must be done BEFORE any measurement
(Machbarkeit vor Messung):

1. **What does the collective actually cost?** The canonical plan estimates
   ~25 MiB per attention layer per 512-token chunk and "+10-20% chunk cost".
   The first number is right; the second silently assumes a link this rig's
   rank0 does not have.

2. **Is the phase-uniform token vector feasible where it matters?** The prize
   is choosing the SAME token vector as the TP phase, which makes the KV layout
   phase-uniform and deletes the seam KV move. But the TP vector puts 43.75% of
   all KV rows on rank0, and rank0 is exactly the rank that deep PP cuts load
   with weights.

Both are pinned here against census numbers, not estimates.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest
from sglang.srt.planner.decoupled_kv import (
    DecoupledKvError,
    KvGeometry,
    collective_bytes_per_chunk,
    deepest_feasible_rank0_layers,
    vector_feasibility,
)

MiB = 1024 * 1024
GiB = 1024**3

# Qwen3.8-27B on this rig, all config-derived (per the #704 D1 discipline).
GEO = KvGeometry(
    num_attention_heads=24,
    head_dim=256,
    num_key_value_heads=4,
    kv_dtype_bytes=1,  # fp8_e4m3
    activation_dtype_bytes=2,  # bf16
    num_attn_layers=16,
)

# Measured at the live [28,20,16] boot: free-for-KV per rank = 2*K_size +
# available_gpu_mem, in GiB.
FREE_GIB = (10.01, 6.19, 5.74)
POOL_TOKENS = 436_766
WEIGHT_MIB_PER_LAYER = 450.7
TP_VECTOR = (14, 10, 8)  # the shipped uneven-DCP token vector


def test_kv_cell_matches_the_boot_log():
    assert GEO.kv_bytes_per_token_per_attn_layer == 2048


def test_collective_traffic_is_independent_of_how_much_kv_is_remote():
    """THE structural fact the plan's estimate does not capture.

    Traffic is set by Q and the partial OUTPUT, both of which are full-size per
    remote rank regardless of how many KV rows that rank holds. A rank holding
    1% of the shard returns exactly the same bytes as one holding 99%. So
    "shard less aggressively to save bandwidth" does not work.
    """
    a = collective_bytes_per_chunk(GEO, n_remote_ranks=2, chunk_tokens=512)
    b = collective_bytes_per_chunk(GEO, n_remote_ranks=2, chunk_tokens=512)
    assert a == b
    # And it is strictly linear in the number of remote participants.
    one = collective_bytes_per_chunk(GEO, n_remote_ranks=1, chunk_tokens=512)
    assert a == pytest.approx(2 * one)
    # Zero remote ranks is the all-local degenerate case: no collective at all.
    assert collective_bytes_per_chunk(GEO, n_remote_ranks=0, chunk_tokens=512) == 0


def test_the_plan_estimate_of_25_mib_per_layer_per_chunk_is_confirmed():
    """The canonical plan's per-layer figure holds up against the geometry."""
    total = collective_bytes_per_chunk(GEO, n_remote_ranks=2, chunk_tokens=512)
    per_layer = total / GEO.num_attn_layers
    assert 24.0 <= per_layer / MiB <= 25.0


def test_the_plan_percentage_assumes_a_link_rank0_does_not_have():
    """+10-20% needs 6-12 GiB/s; rank0 sits on PCIe x4.

    This is the correction the cost model exists to surface: the same 385 MiB
    per chunk is +10% or +40% depending entirely on the link, and rank0 is the
    slow one.
    """
    total_mib = collective_bytes_per_chunk(GEO, 2, 512) / MiB
    serial_chunk_ms = 320.0

    def pct(bw_mib_s):
        return 100.0 * (total_mib / bw_mib_s * 1000.0) / serial_chunk_ms

    assert pct(12000.0) == pytest.approx(10.0, abs=1.0)
    assert pct(6000.0) == pytest.approx(20.0, abs=1.0)
    # The measured-class figure for rank0's x4 link.
    assert pct(3000.0) > 35.0


def test_the_tp_vector_fits_at_the_incumbent_cut():
    fit = vector_feasibility(
        shares=TP_VECTOR,
        free_gib=FREE_GIB,
        total_tokens=POOL_TOKENS,
        geometry=GEO,
    )
    assert all(fit.fits)
    assert fit.need_gib[0] == pytest.approx(5.83, abs=0.05)


def test_the_tp_vector_becomes_infeasible_at_the_deep_cuts_it_should_unlock():
    """The central contradiction of #704b, quantified.

    Decoupling exists to make deep cuts affordable. The phase-uniform vector
    puts 43.75% of KV on rank0 -- the rank a deep cut loads with weights. So
    the prize and the purpose fight each other, and the design must say which
    yields rather than assume they compose.
    """
    for n0, expect in ((28, True), (35, True), (38, False), (44, False)):
        free0 = FREE_GIB[0] - (n0 - 28) * WEIGHT_MIB_PER_LAYER / 1024.0
        fit = vector_feasibility(
            shares=TP_VECTOR,
            free_gib=(free0, FREE_GIB[1], FREE_GIB[2]),
            total_tokens=POOL_TOKENS,
            geometry=GEO,
        )
        assert fit.fits[0] is expect, f"rank0 at n0={n0}"


def test_the_depth_ceiling_under_the_phase_uniform_vector_is_solved():
    n0 = deepest_feasible_rank0_layers(
        shares=TP_VECTOR,
        free_gib=FREE_GIB,
        total_tokens=POOL_TOKENS,
        geometry=GEO,
        weight_mib_per_layer=WEIGHT_MIB_PER_LAYER,
        base_rank0_layers=28,
    )
    assert n0 == 37


def test_a_free_proportional_vector_at_a_deep_cut_is_feasible_and_larger():
    """The other arm: give up vector identity, keep the depth.

    At [44,10,10] the free-proportional shares are near the INVERSE of the TP
    vector on rank0 (13.5% vs 43.75%), because rank0 is weight-rich under a
    deep PP cut and weight-light under TP width-sharding. The two phases
    genuinely want opposite vectors.
    """
    cut = (44, 10, 10)
    base = (28, 20, 16)
    free = tuple(
        FREE_GIB[i] - (cut[i] - base[i]) * WEIGHT_MIB_PER_LAYER / 1024.0
        for i in range(3)
    )
    total = sum(free)
    shares = tuple(f / total for f in free)
    assert shares[0] < 0.20 and shares[1] > 0.40

    fit = vector_feasibility(
        shares=shares, free_gib=free, total_tokens=POOL_TOKENS, geometry=GEO
    )
    assert all(fit.fits)
    # And the world pool it admits is far above the incumbent's.
    assert fit.world_pool_tokens > 1.5 * POOL_TOKENS


def test_shares_are_normalised_not_required_to_be_fractions():
    """A raw vector like (14,10,8) and its normalised form must agree."""
    a = vector_feasibility(TP_VECTOR, FREE_GIB, POOL_TOKENS, GEO)
    norm = tuple(v / sum(TP_VECTOR) for v in TP_VECTOR)
    b = vector_feasibility(norm, FREE_GIB, POOL_TOKENS, GEO)
    assert a.need_gib == pytest.approx(b.need_gib)


def test_a_degenerate_vector_is_refused():
    with pytest.raises(DecoupledKvError, match="non-negative"):
        vector_feasibility((-1, 2, 3), FREE_GIB, POOL_TOKENS, GEO)
    with pytest.raises(DecoupledKvError, match="zero"):
        vector_feasibility((0, 0, 0), FREE_GIB, POOL_TOKENS, GEO)
    with pytest.raises(DecoupledKvError, match="ranks"):
        vector_feasibility((1, 1), FREE_GIB, POOL_TOKENS, GEO)


def test_traffic_scales_with_chunk_and_layer_count():
    base = collective_bytes_per_chunk(GEO, 2, 512)
    assert collective_bytes_per_chunk(GEO, 2, 1024) == pytest.approx(2 * base)
    half = KvGeometry(
        num_attention_heads=24,
        head_dim=256,
        num_key_value_heads=4,
        kv_dtype_bytes=1,
        activation_dtype_bytes=2,
        num_attn_layers=8,
    )
    assert collective_bytes_per_chunk(half, 2, 512) == pytest.approx(base / 2)


def test_structural_uniformity_is_separable_from_vector_identity():
    """The distinction the design turns on.

    Both phases token-sharding all 16 attention layers makes the LAYOUT KIND
    uniform, which is what a content-addressed cache key needs. Choosing the
    SAME shares additionally deletes the seam byte move. The first is free; the
    second costs depth. A design that conflates them would pay for both or
    neither.
    """
    from sglang.srt.planner.decoupled_kv import seam_rebalance_bytes

    # Same structure, different shares -> a rebalance, not a re-layout.
    moved = seam_rebalance_bytes(
        from_shares=(0.1353, 0.4827, 0.3820),
        to_shares=(0.4375, 0.3125, 0.2500),
        total_tokens=POOL_TOKENS,
        geometry=GEO,
    )
    assert moved > 0
    # Identical shares -> nothing moves at all. That is the phase-uniform prize.
    assert (
        seam_rebalance_bytes(
            from_shares=TP_VECTOR,
            to_shares=TP_VECTOR,
            total_tokens=POOL_TOKENS,
            geometry=GEO,
        )
        == 0
    )


# --------------------------------------------------------------------------
# Free-proportional is the PRIMARY case (operator decision): vector identity
# may lose the flip-frequency break-even, so the design must be correct without
# it. The question that then matters is whether the vector follows the cut.
# --------------------------------------------------------------------------

LADDER = [(31, 17, 16), (35, 14, 15), (38, 13, 13), (41, 11, 12), (44, 10, 10)]
BASE_CUT = (28, 20, 16)


def _free_for(cut):
    return tuple(
        FREE_GIB[i] - (cut[i] - BASE_CUT[i]) * WEIGHT_MIB_PER_LAYER / 1024.0
        for i in range(3)
    )


def test_one_fixed_vector_serves_every_rung_and_moves_no_rows():
    """The decision: the vector does NOT follow the cut.

    Letting it follow re-optimises each rung at the price of a KV rebalance on
    every step and buys nothing, because one vector chosen against the tightest
    per-rank constraint is feasible at all of them -- and a fixed vector makes a
    rung change move ZERO KV rows, which is what the ladder needs.
    """
    from sglang.srt.planner.decoupled_kv import fixed_vector_for_ladder

    vec = fixed_vector_for_ladder([_free_for(c) for c in LADDER], POOL_TOKENS, GEO)
    for cut in LADDER:
        fit = vector_feasibility(vec, _free_for(cut), POOL_TOKENS, GEO)
        assert all(fit.fits), f"fixed vector infeasible at {cut}"
        # Every rung must also beat the coupled incumbent pool, or decoupling
        # is not worth its collective cost.
        assert fit.world_pool_tokens > POOL_TOKENS

    # Held fixed, a rung change moves nothing.
    from sglang.srt.planner.decoupled_kv import seam_rebalance_bytes

    assert seam_rebalance_bytes(vec, vec, POOL_TOKENS, GEO) == 0


def test_a_per_rung_vector_would_pay_a_rebalance_on_every_step():
    """The cost avoided, quantified, so the choice is not a matter of taste."""
    from sglang.srt.planner.decoupled_kv import seam_rebalance_bytes

    GiB_ = 1024**3
    total = 0
    prev = None
    for cut in LADDER:
        free = _free_for(cut)
        shares = tuple(f / sum(free) for f in free)
        if prev is not None:
            moved = seam_rebalance_bytes(prev, shares, POOL_TOKENS, GEO)
            assert moved > 0, "adjacent rungs really do want different vectors"
            total += moved
        prev = shares
    assert total / GiB_ > 3.0


def test_the_binding_rank_differs_by_rung_which_is_why_the_minimum_is_per_rank():
    """Deepening starves rank0 while FREEING rank1 and rank2.

    So there is no single 'tightest rung'; the constraint for each rank comes
    from a different one, and a fixed vector must be built from the per-rank
    minimum over rungs rather than from any one rung's free vector.
    """
    shallow = _free_for(LADDER[0])
    deep = _free_for(LADDER[-1])
    assert deep[0] < shallow[0], "rank0 loses free memory as the cut deepens"
    assert deep[1] > shallow[1], "rank1 gains it"
    assert deep[2] > shallow[2], "rank2 gains it"


def test_an_impossible_ladder_is_refused():
    from sglang.srt.planner.decoupled_kv import fixed_vector_for_ladder

    with pytest.raises(DecoupledKvError, match="no rungs"):
        fixed_vector_for_ladder([], POOL_TOKENS, GEO)
    with pytest.raises(DecoupledKvError, match="disagree"):
        fixed_vector_for_ladder([(1.0, 2.0, 3.0), (1.0, 2.0)], POOL_TOKENS, GEO)
