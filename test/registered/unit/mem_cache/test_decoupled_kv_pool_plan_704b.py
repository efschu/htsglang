"""#704b R6: KV pool sized by token share, not by layer ownership.

Today's shape is `own_attn_layers x ALL tokens`, from the ownership filter at
`model_runner_kv_cache_mixin.py:2496-2500`. B1 inverts it to
`ALL attn_layers x own token share`.

Calibration data for this rig (in the test, not the module): 64 layers,
`full_attention_interval` 4 → 16 attention layers; cut [28,20,16] gives
attention counts 7/5/4; cell 2048 B per token per attention layer
(fp8_e4m3, byte-exact against the boot log).

Hermetic: pure arithmetic, no CUDA, no pools.
"""

import pytest
from sglang.srt.mem_cache.decoupled_kv_pool_plan import (
    DECOUPLED,
    STAGE_LOCAL,
    KvPoolPlanError,
    layer_extents,
    plan_for_rank,
    realized_share,
    validate_plan,
    validate_world_conservation,
)

ATTN = tuple(i for i in range(64) if (i + 1) % 4 == 0)  # 16 layers: 3,7,...,63
CELL = 2048
T = 436_766
CUT = ((0, 28), (28, 48), (48, 64))  # [28,20,16] stage ranges
SHARES = (0.135, 0.483, 0.382)  # free-proportional, NOT layer-proportional
# The owner rule hands out whole residue slots, so a share is only realizable
# if it lands on the 1/PERIOD grid (owner.py:406-436). These shares are
# 135/483/382 of 1000 -- exact at PERIOD=1000, unrealizable at PERIOD=3.
PERIOD = 1000


def _stage(rank):
    lo, hi = CUT[rank]
    return plan_for_rank(ATTN, lo, hi, T, CELL, armed=False)


def _decoupled(rank):
    lo, hi = CUT[rank]
    return plan_for_rank(
        ATTN, lo, hi, T, CELL, armed=True, share=SHARES[rank], period=PERIOD
    )


def test_unarmed_reproduces_todays_ownership_filter_exactly():
    """Byte-identity pin. Unarmed must be the pre-#704b plan, not merely close."""
    assert len(ATTN) == 16
    for rank, expected_attn in enumerate((7, 5, 4)):
        p = _stage(rank)
        lo, hi = CUT[rank]
        assert p.mode == STAGE_LOCAL
        assert p.layer_ids == tuple(i for i in ATTN if lo <= i < hi)
        assert len(p.layer_ids) == expected_attn
        assert p.tokens == T, "stage-local holds ALL tokens for its own layers"
        assert p.bytes_total == expected_attn * T * CELL


def test_armed_holds_every_attention_layer_and_only_its_token_share():
    for rank in range(3):
        p = _decoupled(rank)
        assert p.mode == DECOUPLED
        assert p.layer_ids == ATTN, "every rank holds ALL attention layers"
        # The SHIPPED rule, not a second one: dcp_compact_pool_rows ceils to
        # a whole owner block (owner.py:155-181). round(share * T) would floor
        # here, which is the off-by-one that already cost an out-of-bounds
        # scatter.
        from sglang.srt.layers.dcp.owner import dcp_compact_pool_rows

        assert p.tokens == dcp_compact_pool_rows(
            T, PERIOD, round(SHARES[rank] * PERIOD)
        )
        assert p.tokens >= round(SHARES[rank] * T)
        assert p.bytes_total == 16 * p.tokens * CELL


def test_the_share_is_NOT_the_layer_fraction():
    """The whole point of the slice, stated as an inequality.

    rank0 owns 7 of 16 attention layers (43.75%) but takes a 13.5% token share.
    A plan that derived the share from layer count would rebuild exactly the
    ownership coupling this removes.
    """
    layer_fraction = 7 / 16
    assert abs(SHARES[0] - layer_fraction) > 0.25
    stage, dec = _stage(0), _decoupled(0)
    assert dec.bytes_total < stage.bytes_total, "rank0 shrinks under decoupling"


def test_the_world_total_is_CONSERVED_between_modes():
    """Decoupling redistributes capacity; it does not create it."""
    want = 16 * T * CELL
    assert sum(_stage(r).bytes_total for r in range(3)) == want
    validate_world_conservation([_decoupled(r) for r in range(3)], ATTN, T)


def test_a_share_vector_that_does_not_sum_to_one_is_caught():
    """CAN-FAIL: an under-summing vector silently shrinks the world pool."""
    bad = [
        plan_for_rank(ATTN, *CUT[r], T, CELL, armed=True, share=s, period=5)
        for r, s in enumerate((0.2, 0.2, 0.2))
    ]
    with pytest.raises(KvPoolPlanError, match="does not change it"):
        validate_world_conservation(bad, ATTN, T)


def test_a_WRONG_SIZED_armed_pool_fails_loudly():
    """The failure this slice exists to prevent.

    A rank that holds only its own layers cannot answer a read for a layer
    another stage owns, and the failure would surface as WRONG OUTPUT rather
    than as a missing row -- so validation must reject it by name.
    """
    import dataclasses

    good = _decoupled(0)
    wrong = dataclasses.replace(good, layer_ids=tuple(i for i in ATTN if i < 28))
    with pytest.raises(KvPoolPlanError, match="missing attention layer"):
        validate_plan(wrong, ATTN)
    validate_plan(good, ATTN)  # the good one passes


def test_arming_without_a_share_is_refused():
    with pytest.raises(KvPoolPlanError, match="no token share"):
        plan_for_rank(ATTN, 0, 28, T, CELL, armed=True)


def test_a_share_supplied_while_unarmed_is_refused():
    """A share that silently does nothing is how a decoupled pool ends up
    sized like a stage-local one."""
    with pytest.raises(KvPoolPlanError, match="not armed"):
        plan_for_rank(ATTN, 0, 28, T, CELL, armed=False, share=0.3)


def test_degenerate_shares_are_refused():
    with pytest.raises(KvPoolPlanError, match="not in"):
        plan_for_rank(ATTN, 0, 28, T, CELL, armed=True, share=0.0, period=PERIOD)
    with pytest.raises(KvPoolPlanError, match="not in"):
        plan_for_rank(ATTN, 0, 28, T, CELL, armed=True, share=1.5, period=PERIOD)
    # A zero-row pool is now UNREACHABLE by construction rather than caught:
    # the ceil hands out at least one whole owner block, so even a 1/1000000
    # share of 10 tokens gets a row. Kept as a pin because the refusal it
    # replaces was load-bearing before the sizing rule changed.
    tiny = plan_for_rank(ATTN, 0, 28, 10, CELL, armed=True, share=1e-6, period=1_000_000)
    assert tiny.tokens >= 1


def test_layer_extents_use_the_GLOBAL_slot_index():
    """The #706 seam. A rank-local index would put bytes in the wrong slot --
    the same silent failure shape as the canonical-page work found."""
    ext = layer_extents(_decoupled(1), ATTN)
    assert len(ext) == 16
    assert ext[0] == (0, 0, CELL)
    assert ext[15] == (15, 15 * CELL, CELL)
    # Stage-local rank1 owns global attention slots 7..11, NOT 0..4.
    stage_ext = layer_extents(_stage(1), ATTN)
    assert [slot for slot, _, _ in stage_ext] == [7, 8, 9, 10, 11]


def test_decoupled_residence_is_a_WHOLE_canonical_page():
    """Residence is whole; authorship is not.

    A canonical page is one token x all 16 attention slots. Under decoupling a
    rank covers every slot, so a page never straddles ranks -- which is what
    the canonical store wants. It does NOT remove the completeness marker:
    production stays layer-sharded, so the 16 slots still arrive from three
    writers.
    """
    covered = {slot for slot, _, _ in layer_extents(_decoupled(2), ATTN)}
    assert covered == set(range(16)), "whole page resident on one rank"
    partial = {slot for slot, _, _ in layer_extents(_stage(2), ATTN)}
    assert partial != set(range(16)), "stage-local is a partial page"


def test_a_non_attention_layer_has_no_page_slot():
    import dataclasses

    p = dataclasses.replace(_decoupled(0), layer_ids=(4,))  # 4 is linear
    with pytest.raises(KvPoolPlanError, match="no canonical page slot"):
        layer_extents(p, ATTN)


def test_a_foreign_geometry_plans_the_same_way():
    """No rig constants: different depth, interval, cell and shares."""
    attn = tuple(i for i in range(24) if (i + 1) % 2 == 0)  # 12 layers
    p = plan_for_rank(attn, 0, 12, 1000, 4096, armed=True, share=0.25, period=4)
    assert p.layer_ids == attn and p.tokens == (1000 // 4 + 1) * 1
    assert p.bytes_total == 12 * p.tokens * 4096


# --- reconciliation with the neighbour commits (see module REVIEW note) -------


def test_a_share_OFF_the_owner_rule_grid_is_refused():
    """CAN-FAIL, and the sharpest finding of the #635 reconciliation.

    The handover door #635 found already built scatters rows by the weighted
    owner rule, whose ownership granularity is 1/period. The SAME share is
    realizable at one period and not at another, so the period is what decides
    -- not the share's plausibility. 0.135 is exactly 135/1000, and nothing at
    all out of 3 residue slots.
    """
    plan_for_rank(ATTN, 0, 28, T, CELL, armed=True, share=0.135, period=1000)
    with pytest.raises(KvPoolPlanError, match="not realizable"):
        plan_for_rank(ATTN, 0, 28, T, CELL, armed=True, share=0.135, period=3)


def test_the_refusal_names_the_grid_it_wanted():
    """An error that does not say what would have worked is a dead end."""
    with pytest.raises(KvPoolPlanError, match=r"nearest 1 = 0\.333333"):
        plan_for_rank(ATTN, 0, 28, T, CELL, armed=True, share=0.4, period=3)


def test_an_unrealizable_share_is_REFUSED_not_rounded():
    """Rounding here would hide that the caller skipped the quantiser.

    planner/decoupled_kv.py:261 quantize_shares exists to land a fractional
    target on the grid deliberately, repairing over-commitment as it goes.
    Silently rounding at plan time would produce a pool sized for a share the
    solver never chose.
    """
    with pytest.raises(KvPoolPlanError, match="do not round it here"):
        plan_for_rank(ATTN, 0, 28, T, CELL, armed=True, share=0.3, period=7)


def test_arming_without_a_period_is_refused():
    """Without the period the realizability check is not merely skipped -- it
    is impossible, so the parameter is required rather than optional."""
    with pytest.raises(KvPoolPlanError, match="no owner-rule period"):
        plan_for_rank(ATTN, 0, 28, T, CELL, armed=True, share=0.5)


def test_a_period_supplied_while_unarmed_is_refused():
    with pytest.raises(KvPoolPlanError, match="not armed"):
        plan_for_rank(ATTN, 0, 28, T, CELL, armed=False, period=3)


def test_realized_share_matches_the_owner_rule_expression():
    """Pins the planner to owner.py:406-436 rather than to a second copy.

    Read and write there are deliberately one expression; a planner that
    computed the share its own way would be a third.
    """
    assert realized_share(16, 0, 2) == 2 / 16
    assert realized_share(16, 2, 7) == 5 / 16
    assert realized_share(16, 7, 16) == 9 / 16
    # The bands tile the period exactly, so the shares sum to one.
    assert sum(realized_share(16, lo, hi) for lo, hi in ((0, 2), (2, 7), (7, 16))) == 1.0
    for share, band in ((2 / 16, (0, 2)), (5 / 16, (2, 7)), (9 / 16, (7, 16))):
        plan_for_rank(ATTN, 0, 28, T, CELL, armed=True, share=share, period=16)
        assert realized_share(16, *band) == share


def test_a_band_outside_the_period_is_refused():
    with pytest.raises(KvPoolPlanError, match="not a sub-range"):
        realized_share(16, 12, 20)
    with pytest.raises(KvPoolPlanError, match="not a sub-range"):
        realized_share(16, 5, 5)
    with pytest.raises(KvPoolPlanError, match="must be positive"):
        realized_share(0, 0, 1)


def test_a_SECOND_SECTION_appended_to_the_layer_list_is_refused():
    """CAN-FAIL: the #646 rule (b851df7626) adopted for this list.

    #646's defect was a flat list that stopped being one section when the
    draft pool was appended, after which a boundary recovered by arithmetic
    mispaired silently. This module shards by token share whatever it is
    handed, so an appended section would be sized as target layers and split
    across ranks when such a section is owned whole.
    """
    with_draft = ATTN + ATTN[:2]  # a draft section appended to the same list
    with pytest.raises(KvPoolPlanError, match="repeat"):
        plan_for_rank(with_draft, 0, 28, T, CELL, armed=True, share=0.5, period=2)


def test_a_concatenated_layer_list_is_refused():
    scrambled = ATTN[8:] + ATTN[:8]
    with pytest.raises(KvPoolPlanError, match="not in increasing order"):
        plan_for_rank(scrambled, 0, 28, T, CELL, armed=True, share=0.5, period=2)


def test_the_unarmed_path_did_NOT_acquire_the_new_refusals():
    """The byte-identity pin outranks the new guards.

    A list the armed path refuses must still plan unarmed exactly as before,
    or this commit changed today's behaviour while claiming not to.
    """
    with_draft = ATTN + ATTN[:2]
    p = plan_for_rank(with_draft, 0, 28, T, CELL, armed=False)
    assert p.mode == STAGE_LOCAL
    assert p.layer_ids == tuple(i for i in with_draft if 0 <= i < 28)
    assert p.tokens == T
