"""#704: the PP pool's token-scaling term lives on ATTENTION layers only.

`PhasePoolModel` (#702 revision 3) prices a rank's token capacity as

    free_i / (layers_i * kv_mib_per_token_per_layer)

with `layers_i` the rank's TOTAL layer count and a flat per-layer cost. That
form is calibrated -- it reproduces the incumbent and it backtests the
`[42,11,11]` OOM -- but it is dimensionally wrong, and the calibration hides
the error because a contiguous cut correlates attention count with layer
count.

Ground truth is in the allocator, not in a fit. `HybridLinearKVPool` builds its
token-indexed pool with `layer_num=self.full_layer_nums`, i.e. only the
full-attention layers (`mem_cache/memory_pool.py:3609` sets `full_layer_nums =
len(full_attention_layer_ids)`, consumed at `:3687`). The linear (GDN) layers
are served by `MambaPool`, whose size follows the SEQUENCE count, not the token
count. So a rank's token-scaling footprint is

    max_total_tokens * attn_layers_i * kv_bytes_per_token_per_attn_layer

and its linear layers contribute a token-INDEPENDENT term.

The arithmetic that settles it: this checkpoint has num_key_value_heads=4 and
head_dim=256, so one token of K+V for one attention layer is 2*4*256 = 2048
elements. The flat model's calibrated cost is 960 bytes per token per layer,
i.e. **0.47 bytes per element** -- no dtype has a fractional byte width. Priced
on attention layers only, the same census yields 4096 B (bf16) to within the
census's own known pessimism.

The two forms agree on totals per token, because both are fitted to the same
observed pool. They disagree on the DISTRIBUTION across ranks, which is the
only thing a cut solver is for.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest
from sglang.srt.planner.pp_cut import (
    FamilyPoolModel,
    attention_counts,
    family_phase_pool,
    layer_families_from_config,
    stage_family_capacities,
)

INCUMBENT = (28, 20, 16)
LIVE_INCUMBENT_POOL = 434_878.0
CENSUS_FREE_MIB = (26721.2, 16051.3, 13984.3)
CENSUS_WEIGHT_MIB_PER_LAYER = 450.7

# Qwen3.8-27B INT8: 64 layers, full_attention_interval 4 -> 16 attention layers.
FAMILIES = layer_families_from_config(
    {"num_hidden_layers": 64, "full_attention_interval": 4}
)
KV_ELEMS_PER_TOKEN_PER_ATTN_LAYER = 2 * 4 * 256  # K and V, 4 kv heads, head_dim 256


def _model(mamba_mib_per_linear_layer: float = 0.0) -> FamilyPoolModel:
    return FamilyPoolModel(
        free_mib=CENSUS_FREE_MIB,
        weight_mib_per_layer=CENSUS_WEIGHT_MIB_PER_LAYER,
        kv_mib_per_token_per_attn_layer=(
            KV_ELEMS_PER_TOKEN_PER_ATTN_LAYER * 2 / (1024.0 * 1024.0)  # bf16
        ),
        mamba_mib_per_linear_layer=mamba_mib_per_linear_layer,
        layer_families=FAMILIES,
    )


def test_only_attention_layers_are_counted_in_the_token_denominator():
    """The defect in one assertion.

    Rank 0 at (32,16,16) and at (35,14,15) holds the SAME 8 attention layers --
    layers 32, 33, 34 are all linear under interval 4. The flat rule divides its
    capacity by 35 instead of 32, charging three linear layers a per-token cost
    they do not carry. Only their weights bite.
    """
    m = _model()
    assert attention_counts(FAMILIES, (32, 16, 16))[0] == 8
    assert attention_counts(FAMILIES, (35, 14, 15))[0] == 8

    shallow = stage_family_capacities((32, 16, 16), m)[0]
    deep = stage_family_capacities((35, 14, 15), m)[0]

    # The only thing that changed on rank0 is three layers' worth of weights.
    free_shallow = CENSUS_FREE_MIB[0] - CENSUS_WEIGHT_MIB_PER_LAYER * 32
    free_deep = CENSUS_FREE_MIB[0] - CENSUS_WEIGHT_MIB_PER_LAYER * 35
    assert deep / shallow == pytest.approx(free_deep / free_shallow, rel=1e-9)

    # And that is emphatically NOT the flat rule's answer, which would also
    # divide by the layer count.
    assert deep / shallow > (free_deep / free_shallow) * (32.0 / 35.0) * 1.05


def test_implied_cost_per_element_is_a_whole_number_of_bytes():
    """CAN-FAIL PROOF: a pool rule that fits but cannot be a dtype is a fit.

    The flat all-layer rule lands on 0.47 bytes per element. This one must land
    on a real width.
    """
    m = _model()
    bytes_per_elem = (
        m.kv_mib_per_token_per_attn_layer
        * 1024.0
        * 1024.0
        / KV_ELEMS_PER_TOKEN_PER_ATTN_LAYER
    )
    assert bytes_per_elem == pytest.approx(round(bytes_per_elem), abs=1e-9)
    assert bytes_per_elem >= 1.0


def test_census_reproduces_the_observed_incumbent_pool_within_its_pessimism():
    """The census free numbers are ~14-17% pessimistic; the FORM must still fit.

    Priced on attention layers with bf16 KV the incumbent lands at 0.83 of the
    observed 434,878 and binds on rank1. The flat rule reaches the same total
    only by distributing it wrongly.
    """
    m = _model()
    caps = stage_family_capacities(INCUMBENT, m)
    pool = family_phase_pool(INCUMBENT, m)
    assert pool == min(caps)
    assert caps.index(min(caps)) == 1, "rank1 is the binding rank at the incumbent"
    assert 0.80 <= pool / LIVE_INCUMBENT_POOL <= 0.90


def test_arm_b_still_fails_under_the_corrected_rule():
    """The corrected rule must not un-explain the OOM it inherited.

    A model that fixes its functional form but loses the metal failure it was
    validated against has been re-parameterized, not corrected.
    """
    m = _model()
    pool = family_phase_pool((42, 11, 11), m)
    caps = stage_family_capacities((42, 11, 11), m)
    assert caps[0] == min(caps), "rank0 binds arm B"
    assert pool < LIVE_INCUMBENT_POOL / 1.5


def test_mamba_term_is_token_independent_not_a_denominator():
    """GDN state memory scales with sequences, so it subtracts, never divides."""
    lean = family_phase_pool(INCUMBENT, _model(0.0))
    heavy = family_phase_pool(INCUMBENT, _model(50.0))
    assert heavy < lean
    # A subtractive term moves capacity strictly less than a divisive one would.
    m = _model()
    caps = stage_family_capacities(INCUMBENT, m)
    linear_on_binding_rank = 20 - attention_counts(FAMILIES, INCUMBENT)[1]
    assert (
        heavy / lean
        > 1.0
        - (50.0 * linear_on_binding_rank)
        / (CENSUS_FREE_MIB[1] - CENSUS_WEIGHT_MIB_PER_LAYER * 20)
        - 1e-9
    )
    assert caps[1] == pytest.approx(lean, rel=1e-12)


def test_a_zero_attention_stage_is_refused_not_priced_as_infinite():
    """A stage with no attention layer has no token capacity to divide.

    Under the flat rule such a stage silently prices as finite. Here it must be
    refused, because 'infinite pool on rank2' is exactly the kind of answer that
    arms a boot that cannot run.
    """
    m = _model()
    # Layers 0..2 are linear under interval 4, so a 3-layer leading stage holds
    # no attention layer at all.
    with pytest.raises(ValueError, match="attention"):
        stage_family_capacities((3, 45, 16), m)
