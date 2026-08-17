"""#704 D1: the per-token KV cell is CONSUMED from config, never fitted.

This file replaces test_pp_cut_family_pool_704.py. That suite pinned the
attention-only pool rule against a duplicate model of mine; the rule has since
been adjudicated correct and adopted as canonical in Slot-2's rev5, so the
duplicate is gone. What survives, and matters more, is WHY the dispute was
settled without a boot: a dimensional argument plus a byte-exact log line.

The flat all-layer rule implied 960 bytes per token per layer against a K+V of
2 x 4 kv-heads x 256 head_dim = 2048 elements -- 0.47 bytes per element, and no
dtype has a fractional byte width. That is what proves a fit is a fit.

The positive statement is here: the cell is
`2 x num_key_value_heads x head_dim x dtype_width`, taken from config. The
[28,20,16] boot log confirms it with zero free parameters -- at 436,766 tokens
it logs K sizes 2.92 / 2.08 / 1.67 GB against attention counts 7 / 5 / 4, i.e.
exactly `attn_i x 1024 B` per token for K.

The binding lesson of the review gate: calibration against the incumbent
silently absorbs exactly the layout-varying terms a new layout then exposes.
Every term from config or instruments; fit nothing.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest
from sglang.srt.planner.pp_cut import (
    kv_dtype_width_bytes,
    kv_mib_per_token_per_attn_layer_from_config,
)

QWEN38_CONFIG = {"num_key_value_heads": 4, "head_dim": 256, "num_hidden_layers": 64}
BOOT_TOKENS = 436_766
# K size logged per rank at that pool, GiB, against attention counts 7 / 5 / 4.
BOOT_K_GIB = {7: 2.92, 5: 2.08, 4: 1.67}


def test_cell_is_2048_bytes_under_the_shipped_fp8_dtype():
    cell = kv_mib_per_token_per_attn_layer_from_config(QWEN38_CONFIG, "fp8_e4m3")
    assert cell * 1024 * 1024 == pytest.approx(2048.0)


def test_cell_reproduces_the_boot_log_k_sizes_byte_exactly():
    """CAN-FAIL PROOF: the strongest evidence in the adjudication.

    K alone is half the cell. With no fitted parameter this must land on the
    logged GB figures to their printed precision.
    """
    cell = kv_mib_per_token_per_attn_layer_from_config(QWEN38_CONFIG, "fp8_e4m3")
    k_mib_per_token_per_layer = cell / 2.0
    for attn, logged_gib in BOOT_K_GIB.items():
        predicted_gib = k_mib_per_token_per_layer * attn * BOOT_TOKENS / 1024.0
        assert predicted_gib == pytest.approx(logged_gib, abs=0.005)


def test_a_bf16_reading_would_miss_the_log_by_exactly_2x():
    """The specific error this file exists to prevent.

    DESIGN_704 originally read the cell as 4096 B (bf16) because it was FITTED
    against an observed pool rather than read from config. The log falsifies
    that directly.
    """
    bf16 = kv_mib_per_token_per_attn_layer_from_config(QWEN38_CONFIG, "bf16")
    fp8 = kv_mib_per_token_per_attn_layer_from_config(QWEN38_CONFIG, "fp8_e4m3")
    assert bf16 == pytest.approx(2.0 * fp8)
    predicted_gib = (bf16 / 2.0) * 7 * BOOT_TOKENS / 1024.0
    assert predicted_gib > BOOT_K_GIB[7] * 1.9


def test_an_unknown_dtype_is_refused_rather_than_defaulted():
    """A wrong default is a silent 2x on every pool number."""
    with pytest.raises(ValueError, match="unknown kv_cache_dtype"):
        kv_dtype_width_bytes("float6_e3m2")


def test_auto_is_refused_because_it_names_no_width():
    with pytest.raises(ValueError, match="does not name a width"):
        kv_dtype_width_bytes("auto")


def test_head_dim_falls_back_to_hidden_over_heads_but_never_to_a_guess():
    derived = kv_mib_per_token_per_attn_layer_from_config(
        {"num_key_value_heads": 4, "hidden_size": 5120, "num_attention_heads": 20},
        "fp8_e4m3",
    )
    assert derived * 1024 * 1024 == pytest.approx(2 * 4 * 256 * 1)
    with pytest.raises(ValueError, match="refusing to fit"):
        kv_mib_per_token_per_attn_layer_from_config(
            {"num_key_value_heads": 4}, "fp8_e4m3"
        )


def test_the_flat_rule_implied_a_fractional_byte_width():
    """The dimensional argument that settled the dispute, kept as a record."""
    elements = 2 * 4 * 256
    flat_bytes_per_token_per_layer = 450.7 / 492129.0 * 1024 * 1024
    per_element = flat_bytes_per_token_per_layer / elements
    assert 0.4 < per_element < 0.5
    assert per_element != pytest.approx(round(per_element), abs=1e-6)
