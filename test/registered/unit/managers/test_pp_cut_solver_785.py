# SPDX-License-Identifier: Apache-2.0
"""#785/#702: the cut solver, pinned against the boot it was calibrated on.

The rig numbers live HERE, not in the module. A solver carrying one machine's
fitted vector would be the transfer this corpus forbids, so the shape is
general and the calibration is a fixture taken from a boot's own instruments.

CALIBRATION SOURCE: boot 735-tail785 (commit fa7d387a4e), cut 31/16/17, flip
vector 32,16,16. Every number below is quoted from that boot's log:

    layout_pp   16007.47 / 8008.96 / 10789.22 MiB   (ARENA-TAIL-PROBE, and the
    layout_tp   15925.80 / 8573.78 /  8573.78 MiB    TP-stack build agrees at
                                                     error 0.00)
    arming floor  1523 / 1523 / 3226 MiB
    KV available  7710 / 5504 / 1576 MiB
    cell_size    14336 / 8192 / 10240 B
    tokens      564006 / 704530 / 161378 -> global 161378
"""

import pytest

from sglang.srt.managers.pp_cut_solver import CutModel, calibrate_residuals, describe

N_LAYERS = 64
SHIPPED_CUT = [31, 16, 17]
SHIPPED_ATTN = [7, 4, 5]
MEASURED_AVAILABLE_MIB = [7710.0, 5504.0, 1576.0]
MEASURED_TOKENS = [564006, 704530, 161378]

PER_LAYER_MIB = 8008.96 / 16.0  # rank 1 carries neither embed nor lm_head


@pytest.fixture(scope="module")
def model():
    base = CutModel(
        budget_mib=(31800.0, 18800.0, 19800.0),
        layout_tp_mib=(15925.80, 8573.78, 8573.78),
        per_layer_mib=PER_LAYER_MIB,
        embed_mib=16007.47 - 31 * PER_LAYER_MIB,
        lm_head_mib=10789.22 - 17 * PER_LAYER_MIB,
        bytes_per_token_per_attn_layer=2048,
        attn_layer_ids=tuple(i for i in range(N_LAYERS) if i % 4 == 3),
        # Solved from the boot's own three floors against its three tails:
        # 819 (corridor law) + 192 (arming margin) sits on top of the tail,
        # with 1523 under it. rank 2 checks: 819 + 2215.4 + 192 = 3226.4.
        floor_base_mib=1523.0,
        floor_over_tail_mib=1011.0,
        residual_const_mib=0.0,
        residual_per_layer_mib=0.0,
        residual_per_attn_mib=0.0,
    )
    return calibrate_residuals(base, SHIPPED_CUT, MEASURED_AVAILABLE_MIB)


# ---------------------------------------------------------------------------
# The attention split is implied by the cut. This is what makes the search
# honest, and getting it wrong invents cuts that cannot be built.
# ---------------------------------------------------------------------------


def test_the_shipped_cut_implies_the_attention_ratio_the_boot_declares(model):
    """--pp-stage-ratio 31,16,17 implies --pp-attn-stage-ratio 7,4,5."""
    assert model.attn_counts(SHIPPED_CUT) == SHIPPED_ATTN


def test_the_candidate_cut_implies_its_own_attention_ratio(model):
    """32,17,15 implies 8,4,4 -- the ratio boot 735-full785 was launched with."""
    assert model.attn_counts([32, 17, 15]) == [8, 4, 4]


def test_attention_counts_always_total_the_models_attention_layers(model):
    for cut in ([31, 16, 17], [32, 17, 15], [20, 20, 24], [40, 12, 12]):
        assert sum(model.attn_counts(cut)) == len(model.attn_layer_ids)


def test_a_cut_that_starves_a_stage_of_attention_is_refused(model):
    """A stage with no full-attention layer has no KV cell to divide by."""
    assert model.evaluate([61, 2, 1]) is None


# ---------------------------------------------------------------------------
# Weight layouts, which #785 makes exact.
# ---------------------------------------------------------------------------


def test_the_layouts_reproduce_the_boots_measured_weight_mass(model):
    got = model.layout_pp_mib(SHIPPED_CUT)
    for derived, measured in zip(got, [16007.47, 8008.96, 10789.22]):
        assert abs(derived - measured) <= 0.05


def test_the_cell_is_the_attention_count_times_one_layers_kv(model):
    """14336 / 8192 / 10240 is exactly 2048 x (7, 4, 5)."""
    out = model.evaluate(SHIPPED_CUT)
    assert [o.cell_bytes for o in out] == [14336, 8192, 10240]


# ---------------------------------------------------------------------------
# The calibrated model reproduces the boot it was fitted to.
# ---------------------------------------------------------------------------


def test_it_reproduces_the_calibration_boots_available_column(model):
    out = model.evaluate(SHIPPED_CUT)
    for rank, measured in enumerate(MEASURED_AVAILABLE_MIB):
        assert abs(out[rank].kv_available_mib - measured) <= 1.0


def test_it_reproduces_the_calibration_boots_pool(model):
    """Within 0.05% of 161378 -- close enough to search with, no closer."""
    assert abs(model.pool_tokens(SHIPPED_CUT) - 161378) / 161378 < 5e-4


def test_the_pool_is_a_min_reduce_so_balance_beats_total(model):
    """Why the shipped cut sat at 161378 while two ranks could fund 4x it."""
    out = model.evaluate(SHIPPED_CUT)
    tokens = [o.tokens for o in out]
    assert model.pool_tokens(SHIPPED_CUT) == min(tokens)
    assert max(tokens) > 4 * min(tokens)


def test_the_binding_rank_is_the_one_carrying_the_arena_tail(model):
    """The whole reason the cut is the lever rather than the token vector."""
    out = model.evaluate(SHIPPED_CUT)
    binder = min(range(3), key=lambda r: out[r].tokens)
    assert binder == 2
    assert out[binder].arena_tail_mib == max(o.arena_tail_mib for o in out)
    assert out[binder].arming_floor_mib > 2 * model.floor_base_mib


# ---------------------------------------------------------------------------
# The search, and the bound that says what a cut can and cannot do.
# ---------------------------------------------------------------------------


def test_the_search_beats_the_shipped_cut_by_a_wide_margin(model):
    best_tokens, best_cut = model.rank_cuts(limit=1)[0]
    assert best_tokens > 2 * model.pool_tokens(SHIPPED_CUT)
    assert best_cut != SHIPPED_CUT


def test_no_cut_can_zero_every_arena_tail_and_the_sums_say_why(model):
    """A bound, not a search result.

    sum(layout_pp) is essentially cut-independent -- the layer mass is fixed
    and embed/lm_head sit on the end stages either way -- while sum(layout_tp)
    is fixed by the vector. Here the PP side is the larger by ~1732 MiB, so
    some rank must carry a tail no matter how the layers are dealt. The cut's
    value is therefore RELOCATING the tail onto a rank with headroom, not
    abolishing it.
    """
    total_tp = sum(model.layout_tp_mib)
    for cut in ([31, 16, 17], [32, 17, 15], [22, 21, 21], [40, 12, 12]):
        assert sum(model.layout_pp_mib(cut)) > total_tp

    for _tokens, cut in model.rank_cuts(limit=25):
        out = model.evaluate(cut)
        assert max(o.arena_tail_mib for o in out) > 0.0


def test_describe_reports_the_terms_a_reader_needs_to_check_it(model):
    d = describe(model, [32, 17, 15])
    assert d["buildable"] is True
    assert d["attn"] == [8, 4, 4]
    assert d["cell_bytes"] == [16384, 8192, 8192]
    assert d["pool_tokens"] == min(d["tokens"])


def test_an_unbuildable_cut_says_so_rather_than_returning_a_number(model):
    assert describe(model, [10, 10, 10])["buildable"] is False  # not 64 layers
    assert model.pool_tokens([10, 10, 10]) == 0
