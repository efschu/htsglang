"""#702 revision 5: attention-layer divisor + per-layout SOLVED arming floor.

Revision history, because three of five were wrong and the corrections matter:

* rev 2 -- pool as the SUM of per-rank capacities (the TP rule). Arm B
  [42,11,11] was armed on it and OOM'd twice.
* rev 3 -- SUM corrected to MIN (PP is layer-sharded), but still divided by the
  rank's TOTAL layer count.
* rev 4 -- divisor corrected to the rank's FULL-ATTENTION layer count
  (Slot-3's claim, verified below). Its "metal discriminator" test was
  CIRCULAR: the two free constants were solved from the two metal points, so
  reproducing them validated nothing.
* rev 5 -- here: the per-layout arming floor is consumed, and the calibration's
  limits are stated rather than implied.

THE ADJUDICATED KV-SCALING RULE. Token-scaling KV lives ONLY on full-attention
layers. ``HybridLinearKVPool`` is documented "KV cache with separate pools for
full and linear attention layers" (``mem_cache/memory_pool.py:3606``); its
``full_kv_pool`` takes ``layer_num=self.full_layer_nums`` (``:3688``) and
``full_layer_nums = len(full_attention_layer_ids)`` (``:3637``). Linear/GDN
layers hold per-SEQUENCE MambaPool slots -- a residency term, never a divisor.

WHAT THE TWO METAL POINTS DO AND DO NOT SETTLE. On this checkpoint the
attention layers are uniformly every 4th, so any multiple-of-4 cut has
``total/attention == 4`` on every rank and rev 3 and rev 4 agree exactly. Both
[28,20,16] and [32,16,16] are multiple-of-4, so **neither point discriminates
the two rules**. The rule is adopted on the allocator source, not on these
boots. The discriminating cut is a non-multiple-of-4 one such as [33,15,16].

THE COMMON MODE. Both rules over-predicted [32,16,16] by +9.6 % (456,715
against a live 416,796) -- larger than the disagreement between them. The
arming floor moves with the layout (2255/1728/2467 measured on [32,16,16]
against 1728/1825/2467 on the incumbent) while both models held it fixed.
Rank0's floor rises 527 MiB, which at 8 attention layers and 2048 B/token is
33,728 tokens: **84 % of the 39,919-token gap**, leaving ~1.5 % residual.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest

from sglang.srt.planner.pp_cut import (
    PhasePoolModel,
    pp_phase_pool,
    stage_pp_capacities,
)

INCUMBENT = (28, 20, 16)
ARM_C2 = (32, 16, 16)
ARM_B = (42, 11, 11)
DISCRIMINATOR = (33, 15, 16)

LIVE_INCUMBENT = 436_766.0
LIVE_ARM_C2 = 416_796.0
BOTH_RULES_PREDICTED_C2 = 456_715.0

# fp8_e4m3 KV (from the boot log), 2 * 4 kv-heads * 256 head_dim = 2048 B per
# token per ATTENTION layer. FIXED at its physical value, never fitted.
KV_MIB_PER_TOKEN_PER_ATTN_LAYER = 2048.0 / 1_048_576.0

MEASURED_FLOORS = {
    INCUMBENT: (1728.0, 1825.0, 2467.0),
    ARM_C2: (2255.0, 1728.0, 2467.0),
}


def _model(floors, free=(23_189.8, 15_104.3, 14_561.0)):
    return PhasePoolModel(
        free_mib=free,
        weight_mib_per_layer=450.7,
        kv_mib_per_token_per_attn_layer=KV_MIB_PER_TOKEN_PER_ATTN_LAYER,
        arming_floor_mib=floors,
    )


def _attn(counts):
    """Full-attention layers per contiguous stage: every 4th layer (i % 4 == 3)."""
    out, start = [], 0
    for c in counts:
        out.append(sum(1 for i in range(start, start + c) if i % 4 == 3))
        start += c
    return tuple(out)


def test_attention_pattern_helper_matches_the_census():
    assert sum(_attn((64,))) == 16
    assert _attn(INCUMBENT) == (7, 5, 4)  # matches census n_attn_layers 7/5/4
    assert _attn(ARM_C2) == (8, 4, 4)


def test_multiple_of_four_cuts_cannot_discriminate_the_two_rules():
    """States the limit of the available metal, so nobody claims otherwise."""
    for cut in (INCUMBENT, ARM_C2, (24, 20, 20)):
        for n, a in zip(cut, _attn(cut)):
            assert n / a == 4.0
    # The discriminating cut is not uniform, which is what makes it a test.
    ratios = [n / a for n, a in zip(DISCRIMINATOR, _attn(DISCRIMINATOR))]
    assert len(set(ratios)) > 1


def test_the_floor_must_be_supplied_per_layout_not_defaulted():
    """A constant floor is the defect; the model must consume a solved one."""
    with pytest.raises(TypeError):
        PhasePoolModel(
            free_mib=(1.0, 1.0, 1.0),
            weight_mib_per_layer=1.0,
            kv_mib_per_token_per_attn_layer=1.0,
        )


def test_both_metal_points_reproduce_with_solved_floors():
    for cut, pool in ((INCUMBENT, LIVE_INCUMBENT), (ARM_C2, LIVE_ARM_C2)):
        m = _model(MEASURED_FLOORS[cut])
        assert pp_phase_pool(cut, _attn(cut), m) == pytest.approx(pool, rel=3e-3)


def test_the_binding_rank_switches_between_the_two_points():
    inc = _model(MEASURED_FLOORS[INCUMBENT])
    c2 = _model(MEASURED_FLOORS[ARM_C2])
    caps_i = stage_pp_capacities(INCUMBENT, _attn(INCUMBENT), inc)
    caps_c = stage_pp_capacities(ARM_C2, _attn(ARM_C2), c2)
    assert caps_i.index(min(caps_i)) == 1  # rank1: 20 layers / 5 attention
    assert caps_c.index(min(caps_c)) == 0  # rank0: 32 layers / 8 attention


def test_the_sign_of_the_32_16_16_move_is_negative():
    """Rev 3's fatal error, pinned. Metal: [32,16,16] LOSES 4.4 %."""
    inc = pp_phase_pool(INCUMBENT, _attn(INCUMBENT), _model(MEASURED_FLOORS[INCUMBENT]))
    c2 = pp_phase_pool(ARM_C2, _attn(ARM_C2), _model(MEASURED_FLOORS[ARM_C2]))
    assert c2 < inc
    assert c2 / inc == pytest.approx(LIVE_ARM_C2 / LIVE_INCUMBENT, rel=5e-3)


def test_the_floor_delta_explains_most_of_the_common_mode():
    """Quantifies F4-r4's suspect instead of asserting it."""
    stale = _model(MEASURED_FLOORS[INCUMBENT])  # the wrong floor for this cut
    solved = _model(MEASURED_FLOORS[ARM_C2])
    lost = pp_phase_pool(ARM_C2, _attn(ARM_C2), stale) - pp_phase_pool(
        ARM_C2, _attn(ARM_C2), solved
    )
    assert lost == pytest.approx(33_728.0, rel=0.02)
    assert lost / (BOTH_RULES_PREDICTED_C2 - LIVE_ARM_C2) > 0.8


def test_arm_b_remains_far_below_the_pin():
    """[42,11,11] OOM'd, so metal gives an INEQUALITY, not a point. Do not fit it."""
    m = _model((2255.0, 1728.0, 2467.0))
    assert pp_phase_pool(ARM_B, _attn(ARM_B), m) < LIVE_INCUMBENT * 0.6


def test_rank2_is_not_identified_by_two_metal_points():
    """Honesty guard: 2 equations, 3 unknowns.

    Rank1 binds at the incumbent and rank0 at [32,16,16]; rank2 binds at
    neither, so its free constant is BOUNDED, not identified. Perturbing it
    upward changes neither reproduced point -- which is exactly why it cannot be
    reported as calibrated.
    """
    a = _model(MEASURED_FLOORS[ARM_C2])
    b = _model(MEASURED_FLOORS[ARM_C2], free=(23_189.8, 15_104.3, 19_000.0))
    assert pp_phase_pool(ARM_C2, _attn(ARM_C2), a) == pytest.approx(
        pp_phase_pool(ARM_C2, _attn(ARM_C2), b), rel=1e-12
    )


def test_linear_layers_do_not_scale_the_token_cost():
    """The adjudicated rule itself: only attention layers divide."""
    m = PhasePoolModel(
        free_mib=(20_000.0,) * 3,
        weight_mib_per_layer=0.0,  # isolate the divisor from the weight term
        kv_mib_per_token_per_attn_layer=KV_MIB_PER_TOKEN_PER_ATTN_LAYER,
        arming_floor_mib=(0.0, 0.0, 0.0),
    )
    a = stage_pp_capacities((20, 22, 22), (5, 6, 5), m)[0]
    b = stage_pp_capacities((28, 18, 18), (5, 6, 5), m)[0]
    assert a == pytest.approx(b, rel=1e-12)


def test_a_stage_with_no_attention_layer_is_refused():
    m = _model(MEASURED_FLOORS[INCUMBENT])
    with pytest.raises(ValueError, match="attention"):
        pp_phase_pool((3, 45, 16), (0, 12, 4), m)


def test_weight_overflow_is_refused_by_name():
    m = _model(MEASURED_FLOORS[INCUMBENT])
    with pytest.raises(ValueError, match="rank0"):
        pp_phase_pool((60, 2, 2), (15, 0, 1), m)
