"""#704: the canonical pool solve, assembled from INSTRUMENTS only.

Every term is read from a boot, never derived externally. The session's
repeated lesson: three external re-derivations of the sizer's arithmetic missed
by +20 %, -3.8 % and -12 %, because the reserve tracks per-rank CUDA-graph
capture and config cannot see it.

The four instrumented terms, and where each comes from:

1. **budget posts** -- `"KV budget posts (GiB): ... | rest=..."`, emitted at the
   profiler's success path (84abeb7f5b, live on metal).
2. **mamba ALLOCATED** -- `"Mamba Cache is allocated. ..."`, which already
   existed. Truth, unlike the budget post: the post under-charges the
   allocation by a constant 0.852 on every rank.
3. **available_bytes / cell_size / tokens** -- `"KV pool sizing: ..."`
   (2a6305dd3b), the last link.
4. **per-layout arming floor** -- the #676 solver.

The reserve is then RECOVERED, not modelled: `reserve = rest - available_bytes`.

What this module refuses to do is predict a cut whose reserve is unknown. That
is the whole discipline: the reserve does not transfer between layouts, so a
prediction for an unbooted cut is an extrapolation and must be labelled one.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest

from sglang.srt.planner.boot_instruments import (
    RankInstruments,
    recover_reserve_mib,
    verify_sizing_chain,
    world_pool_tokens,
)

MiB = 1024.0 * 1024.0

# Live [28,20,16] boot. Budget from rank_gpu_memory_mib; posts and rest from
# DATA_704_budget_posts.txt; mamba allocated from the Mamba Cache lines;
# arming floors measured. available_bytes is what the sizing line will publish;
# until that boot lands it is pinned here from tokens x cell_size, which the
# instrument will confirm or refute.
INCUMBENT = (
    RankInstruments(
        stage=0,
        budget_mib=31800.0,
        weights_runtime_mib=15.688 * 1024,
        mamba_post_mib=0.895 * 1024,
        mamba_allocated_mib=1.05 * 1024,
        gguf_scratch_mib=0.0,
        rest_mib=14.472 * 1024,
        available_bytes=8526565376,
        cell_size_bytes=7 * 2048,
        max_total_num_tokens=594766,
        arming_floor_mib=1728.0,
        attn_layers=7,
        gdn_layers=21,
    ),
    RankInstruments(
        stage=1,
        budget_mib=18800.0,
        weights_runtime_mib=9.826 * 1024,
        mamba_post_mib=0.639 * 1024,
        mamba_allocated_mib=0.75 * 1024,
        gguf_scratch_mib=0.0,
        rest_mib=7.894 * 1024,
        available_bytes=4741928960,
        cell_size_bytes=5 * 2048,
        max_total_num_tokens=463079,
        arming_floor_mib=1825.0,
        attn_layers=5,
        gdn_layers=15,
    ),
    RankInstruments(
        stage=2,
        budget_mib=19800.0,
        weights_runtime_mib=10.449 * 1024,
        mamba_post_mib=0.511 * 1024,
        mamba_allocated_mib=0.60 * 1024,
        gguf_scratch_mib=0.0,
        rest_mib=8.375 * 1024,
        available_bytes=3575365632,
        cell_size_bytes=4 * 2048,
        max_total_num_tokens=436446,
        arming_floor_mib=2467.0,
        attn_layers=4,
        gdn_layers=12,
    ),
)


def test_the_budget_chain_reconciles_on_every_rank():
    """rest = budget - sum(posts), reproduced exactly from the emitted terms."""
    for inst in INCUMBENT:
        verify_sizing_chain(inst)  # raises on mismatch


def test_a_broken_chain_is_refused_by_name():
    bad = INCUMBENT[2]._replace_rest(rest_mib=99.0)
    with pytest.raises(ValueError, match="budget chain"):
        verify_sizing_chain(bad)


def test_tokens_equal_available_bytes_over_cell_size():
    for inst in INCUMBENT:
        assert inst.max_total_num_tokens == inst.available_bytes // inst.cell_size_bytes


def test_the_reserve_is_recovered_not_modelled():
    """The term no config can predict: 1.88x spread across three stages."""
    reserves = [recover_reserve_mib(i) for i in INCUMBENT]
    assert reserves[0] == pytest.approx(6.531 * 1024, rel=0.02)
    assert reserves[1] == pytest.approx(3.478 * 1024, rel=0.02)
    assert reserves[2] == pytest.approx(5.045 * 1024, rel=0.02)
    assert max(reserves) / min(reserves) == pytest.approx(1.88, abs=0.05)


def test_the_post_measures_less_than_a_subset_of_what_it_covers():
    """The gap is real; its DECOMPOSITION is not established.

    The post is "mamba state pool + speculative intermediate state + prefill
    activation reserve"; the allocated line sums conv + ssm + intermediate_ssm
    + intermediate_conv. The post nominally covers a SUPERSET and is
    nevertheless smaller -- which cannot be legitimate -- but the 0.852 ratio
    is a ratio of non-identical quantities and must NOT be published as "the
    mamba post under-charges mamba by 14.8 percent".
    """
    for inst in INCUMBENT:
        assert inst.mamba_post_mib < inst.mamba_allocated_mib
        gap = inst.mamba_allocated_mib - inst.mamba_post_mib
        assert gap > 80.0  # MiB, on every stage


def test_the_allocation_is_the_term_the_solver_uses():
    for inst in INCUMBENT:
        assert inst.mamba_charge_mib == inst.mamba_allocated_mib


def test_world_pool_is_the_min_over_ranks_and_reproduces_the_boot():
    """Retro-prediction of the incumbent, from instruments alone."""
    pool, binder = world_pool_tokens(INCUMBENT)
    assert pool == 436446
    assert binder == 2  # PP2, as the boot reports


def test_predicting_an_unbooted_cut_without_its_reserve_is_refused():
    """The reserve does not transfer between layouts; saying so is the point."""
    from sglang.srt.planner.boot_instruments import predict_tokens_for_cut

    with pytest.raises(ValueError, match="reserve"):
        predict_tokens_for_cut(attn_layers=8, rest_mib=14000.0, reserve_mib=None)


def test_predicting_with_a_supplied_reserve_is_arithmetic_only():
    from sglang.srt.planner.boot_instruments import predict_tokens_for_cut

    # The measured PP2 stage, so it must land back on the boot.
    tokens = predict_tokens_for_cut(
        attn_layers=4, rest_mib=8.375 * 1024, reserve_mib=5.045 * 1024
    )
    assert tokens == pytest.approx(436446, rel=5e-3)


def test_the_arming_floor_is_inside_the_reserve_not_a_second_subtraction():
    """Double-counting it would understate every pool by ~2 GiB.

    `rest` is `budget - sum(posts)`, and the arming floor is not a post -- it is
    held back after the profiler, so the recovered `rest - available_bytes`
    already contains it. A solver that subtracted the floor again would charge
    it twice.
    """
    from sglang.srt.planner.boot_instruments import predict_tokens_for_cut

    pp2 = INCUMBENT[2]
    reserve = recover_reserve_mib(pp2)
    assert reserve > pp2.arming_floor_mib  # the floor fits inside it
    tokens = predict_tokens_for_cut(
        attn_layers=pp2.attn_layers, rest_mib=pp2.rest_mib, reserve_mib=reserve
    )
    assert tokens == pytest.approx(pp2.max_total_num_tokens, rel=5e-3)
