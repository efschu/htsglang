"""#704: ONE pool entry point for the ladder, closed against metal.

Shaped to Slot-3's five interface requirements (attn derived not passed, floor
required, binder+per-rank returned, cheap for enumeration, KV cell from config)
and carrying this module's refusal semantics.

The numbers here are the four-boot gate's own data, from the boot that carries
the sizing instrument (2a6305dd3b):

    available_bytes  PP0 8,526,565,376 (cell 14336) -> 594,766
                     PP1 4,741,928,960 (cell 10240) -> 463,079
                     PP2 3,575,365,632 (cell  8192) -> 436,446
    world pool = min = 436,446, BINDING RANK = PP2
    rest             14.472 / 7.894 / 8.375 GiB
    recovered reserve 6.531 / 3.478 / 5.045 GiB

The cell sizes settle the #702 divisor on ABSOLUTE grounds, not just by ratio:
14336 / 10240 / 8192 is exactly 7 / 5 / 4 attention layers x 2048 B, so the
multiple-of-4 degeneracy that made [28,20,16] and [32,16,16] non-discriminating
is broken by the absolute value.

Hermetic: pure arithmetic, no CUDA.
"""

import json

import pytest

from sglang.srt.planner.pp_cut import LAYER_FAMILY_ATTENTION, LAYER_FAMILY_LINEAR
from sglang.srt.planner.rung_pool import (
    PoolContext,
    RungPoolSolution,
    solve_rung_pool,
)

CONFIG_PATH = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8/config.json"

INCUMBENT = (28, 20, 16)
DEEP = (33, 15, 16)

# Metal, from the boot log's "KV pool sizing" lines.
METAL_TOKENS = (594766, 463079, 436446)
METAL_POOL = 436446
METAL_BINDER = 2
METAL_CELLS = (14336, 10240, 8192)

REST_MIB = {INCUMBENT: (14.472 * 1024, 7.894 * 1024, 8.375 * 1024)}
RESERVE_MIB = {INCUMBENT: (6.531 * 1024, 3.478 * 1024, 5.045 * 1024)}
FLOORS = {INCUMBENT: (1728.0, 1825.0, 2467.0), DEEP: (2255.0, 1728.0, 2467.0)}


def _families():
    """64 layers, every 4th full-attention -- the checkpoint's pattern."""
    return tuple(
        LAYER_FAMILY_ATTENTION if i % 4 == 3 else LAYER_FAMILY_LINEAR for i in range(64)
    )


def _ctx():
    with open(CONFIG_PATH) as fh:
        cfg = json.load(fh)
    return PoolContext.from_config(cfg, "fp8_e4m3", _families())


def _solve(counts=INCUMBENT, **kw):
    defaults = dict(
        arming_floor_for=FLOORS,
        reserve_for=RESERVE_MIB,
        rest_for=REST_MIB,
        measured=True,
    )
    defaults.update(kw)
    return solve_rung_pool(counts, _ctx(), **defaults)


def test_the_kv_cell_comes_from_config_and_matches_metal_absolutely():
    ctx = _ctx()
    assert ctx.kv_mib_per_token_per_attn_layer * 1024 * 1024 == pytest.approx(2048.0)
    attn = ctx.attn_counts(INCUMBENT)
    assert attn == (7, 5, 4)
    for a, cell in zip(attn, METAL_CELLS):
        assert round(a * ctx.kv_mib_per_token_per_attn_layer * 1024 * 1024) == cell


def test_attn_counts_are_derived_not_passed():
    """Requirement 1: an inconsistent pair must be unrepresentable."""
    ctx = _ctx()
    assert ctx.attn_counts(DEEP) == (8, 4, 4)
    with pytest.raises(ValueError, match="layers"):
        ctx.attn_counts((30, 20, 16))  # sums to 66


def test_the_incumbent_reproduces_the_metal_pool_and_binder():
    sol = _solve()
    assert isinstance(sol, RungPoolSolution)
    assert sol.pool_tokens == pytest.approx(METAL_POOL, rel=2e-3)
    assert sol.binding_stage == METAL_BINDER
    for got, want in zip(sol.per_stage_tokens, METAL_TOKENS):
        assert got == pytest.approx(want, rel=2e-3)


def test_binder_and_per_rank_caps_are_returned():
    """Requirement 3: the controller must not recompute them."""
    sol = _solve()
    assert len(sol.per_stage_tokens) == 3
    assert sol.per_stage_tokens[sol.binding_stage] == sol.pool_tokens
    assert sol.pool_tokens == min(sol.per_stage_tokens)


def test_the_arming_floor_provider_is_required():
    with pytest.raises(ValueError, match="arming_floor_for"):
        _solve(arming_floor_for=None)


def test_an_unbooted_rung_without_a_reserve_is_refused():
    with pytest.raises(ValueError, match="reserve"):
        solve_rung_pool(
            DEEP,
            _ctx(),
            arming_floor_for=FLOORS,
            reserve_for=lambda c: None,
            rest_for=lambda c: (14000.0, 8000.0, 8500.0),
        )


def test_extrapolated_rungs_are_labelled():
    sol = _solve(measured=False)
    assert sol.provenance == "extrapolated"
    assert any("extrapolat" in c.lower() for c in sol.caveats)


def test_coverage_flags_rank2_against_the_discriminator():
    sol = _solve(reference_cut=DEEP)
    assert not sol.coverage[2].identified
    assert any("rank2" in c for c in sol.caveats)


def test_explain_returns_a_per_stage_breakdown():
    sol = _solve(explain=True)
    assert len(sol.terms) == 3
    pp2 = sol.terms[2]
    assert pp2.attn_layers == 4 and pp2.gdn_layers == 12
    assert pp2.cell_bytes == 8192
    assert pp2.free_for_kv_mib == pytest.approx(8.375 * 1024 - 5.045 * 1024, rel=1e-9)
    # The floor is reported but NOT subtracted: it lives inside the reserve.
    assert pp2.arming_floor_mib == 2467.0
    assert pp2.reserve_mib > pp2.arming_floor_mib


def test_a_stage_with_no_attention_layer_is_refused():
    ctx = PoolContext.from_config(
        json.load(open(CONFIG_PATH)),
        "fp8_e4m3",
        (LAYER_FAMILY_LINEAR,) * 60 + (LAYER_FAMILY_ATTENTION,) * 4,
    )
    with pytest.raises(ValueError, match="attention"):
        solve_rung_pool(
            (20, 20, 24),
            ctx,
            arming_floor_for=lambda c: (0.0, 0.0, 0.0),
            reserve_for=lambda c: (0.0, 0.0, 0.0),
            rest_for=lambda c: (1000.0, 1000.0, 1000.0),
        )


def test_an_unknown_kv_dtype_is_refused_not_defaulted():
    with pytest.raises(ValueError, match="unknown kv_cache_dtype"):
        PoolContext.from_config(json.load(open(CONFIG_PATH)), "fp6_wat", _families())
