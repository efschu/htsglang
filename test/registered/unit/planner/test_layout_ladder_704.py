"""#704: the prefill layout ladder is SOLVED, not tabulated.

Binding requirement: rungs, admission thresholds, hysteresis bands and the
move costs are outputs of the solver from measured inputs -- per-layer prefill
ms (self-probe), NVML free bytes, link bandwidths (pair matrix), checkpoint
weight bytes, and the checkpoint's own `layer_types` vector. This rig's
numbers ([32,16,16], 450.7 MiB/layer, a 42-layer rank0 cap) are CALIBRATION
DATA. None of them may appear as a constant in the solver.

The generality proof is by synthetic foreign profiles (#434 canon): different
card counts, different card sizes, different link bandwidths, different
attention/GDN ratios, and the degenerate 0-GDN pure-attention model. A solved
vector is never transferred across regimes.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest
from sglang.srt.planner.layout_ladder import (
    LadderInputs,
    solve_layout_ladder,
)
from sglang.srt.planner.pp_cut import FamilyPoolModel, layer_families_from_config

# --------------------------------------------------------------------------
# Profile A: this rig, as calibration data only.
# --------------------------------------------------------------------------
RIG_FAMILIES = layer_families_from_config(
    {"num_hidden_layers": 64, "full_attention_interval": 4}
)


def _rig_inputs(**over) -> LadderInputs:
    base = {
        "pool": FamilyPoolModel(
            free_mib=(26721.2, 16051.3, 13984.3),
            weight_mib_per_layer=450.7,
            kv_mib_per_token_per_attn_layer=2 * 4 * 256 * 2 / (1024.0 * 1024.0),
            layer_families=RIG_FAMILIES,
        ),
        "ms_per_layer": (1.7571, 7.740, 7.275),
        "fixed_ms": (0.0, 0.0, 0.0),
        # Measured PCIe reach per rank, MiB/s. Rank0 sits on x4 here.
        "link_mib_per_s": (3000.0, 6000.0, 6000.0),
        "min_pool_tokens": 200_000.0,
        "prefill_tokens_per_s": 20_000.0,
    }
    base.update(over)
    return LadderInputs(**base)


def _foreign_4card() -> LadderInputs:
    """Four cards of differing size and speed, fast links, 1:1 attention:GDN."""
    fam = layer_families_from_config(
        {"num_hidden_layers": 48, "full_attention_interval": 2}
    )
    return LadderInputs(
        pool=FamilyPoolModel(
            free_mib=(40000.0, 24000.0, 24000.0, 16000.0),
            weight_mib_per_layer=300.0,
            kv_mib_per_token_per_attn_layer=2 * 8 * 128 * 2 / (1024.0 * 1024.0),
            layer_families=fam,
        ),
        ms_per_layer=(2.0, 5.0, 5.0, 9.0),
        fixed_ms=(0.0, 0.0, 0.0, 0.0),
        link_mib_per_s=(20000.0, 20000.0, 20000.0, 20000.0),
        min_pool_tokens=50_000.0,
        prefill_tokens_per_s=60_000.0,
    )


def _uniform_rig() -> LadderInputs:
    """Identical cards at identical per-layer cost: there is no ladder here."""
    fam = layer_families_from_config(
        {"num_hidden_layers": 48, "full_attention_interval": 2}
    )
    return LadderInputs(
        pool=FamilyPoolModel(
            free_mib=(40000.0, 40000.0, 40000.0, 40000.0),
            weight_mib_per_layer=300.0,
            kv_mib_per_token_per_attn_layer=2 * 8 * 128 * 2 / (1024.0 * 1024.0),
            layer_families=fam,
        ),
        ms_per_layer=(4.0, 4.0, 4.0, 4.0),
        fixed_ms=(0.0, 0.0, 0.0, 0.0),
        link_mib_per_s=(20000.0, 20000.0, 20000.0, 20000.0),
        min_pool_tokens=50_000.0,
        prefill_tokens_per_s=60_000.0,
    )


def _foreign_pure_attention() -> LadderInputs:
    """0-GDN: every layer is full attention. The degenerate family split."""
    fam = layer_families_from_config({"num_hidden_layers": 32})
    assert set(fam) == {"full_attention"}
    return LadderInputs(
        pool=FamilyPoolModel(
            free_mib=(24000.0, 12000.0),
            weight_mib_per_layer=500.0,
            kv_mib_per_token_per_attn_layer=2 * 8 * 128 * 2 / (1024.0 * 1024.0),
            layer_families=fam,
        ),
        ms_per_layer=(2.0, 9.0),
        fixed_ms=(0.0, 0.0),
        link_mib_per_s=(5000.0, 5000.0),
        min_pool_tokens=20_000.0,
        prefill_tokens_per_s=15_000.0,
    )


ALL_PROFILES = {
    "rig": _rig_inputs,
    "foreign_4card": _foreign_4card,
    "foreign_pure_attention": _foreign_pure_attention,
}


def test_a_uniform_rig_has_no_ladder():
    """The ladder's precondition, pinned as a property.

    Piling layers onto a "fast" rank only buys pipeline time when some rank IS
    faster per layer. With identical cards at identical cost the balanced cut
    is optimal and every deviation is dominated, so the frontier collapses to a
    single rung. The solver must say so rather than manufacture steps.
    """
    ladder = solve_layout_ladder(_uniform_rig())
    assert len(ladder.rungs) == 1
    assert ladder.transitions == ()


# --------------------------------------------------------------------------
# Structural properties that must hold on EVERY profile.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(ALL_PROFILES))
def test_ladder_is_monotone_in_both_axes(name):
    """A ladder is only a ladder if the two axes oppose monotonically.

    Rungs are ordered from most pool / least speed to least pool / most speed.
    If a rung were beaten on BOTH axes by its neighbour it would be dominated
    and must not be on the ladder at all.
    """
    ladder = solve_layout_ladder(ALL_PROFILES[name]())
    assert len(ladder.rungs) >= 2, f"{name}: a one-rung ladder cannot step"
    for a, b in zip(ladder.rungs, ladder.rungs[1:]):
        assert b.pool_tokens < a.pool_tokens
        assert b.pipelined_speedup > a.pipelined_speedup


@pytest.mark.parametrize("name", sorted(ALL_PROFILES))
def test_no_constants_leak_the_rig_layout(name):
    """Every rung sums to the profile's own depth, on any card count."""
    inputs = ALL_PROFILES[name]()
    ladder = solve_layout_ladder(inputs)
    depth = len(inputs.pool.layer_families)
    n_ranks = len(inputs.pool.free_mib)
    for r in ladder.rungs:
        assert sum(r.counts) == depth
        assert len(r.counts) == n_ranks
        assert all(c >= 1 for c in r.counts)


@pytest.mark.parametrize("name", sorted(ALL_PROFILES))
def test_every_rung_clears_the_pool_floor(name):
    """`min_pool_tokens` is the corridor input; no rung may sit under it."""
    inputs = ALL_PROFILES[name]()
    ladder = solve_layout_ladder(inputs)
    for r in ladder.rungs:
        assert r.pool_tokens >= inputs.min_pool_tokens


@pytest.mark.parametrize("name", sorted(ALL_PROFILES))
def test_hysteresis_bands_do_not_cross(name):
    """The anti-oscillation invariant, and it must be DERIVED.

    For each adjacent pair the descend threshold (drop to the deeper, faster
    rung) must sit strictly below the ascend threshold (retreat to the
    shallower, roomier rung). If they crossed, one fill level would demand both
    moves and the controller would flip forever.
    """
    ladder = solve_layout_ladder(ALL_PROFILES[name]())
    for t in ladder.transitions:
        assert t.descend_below_tokens < t.ascend_above_tokens, (
            f"{name}: rung pair {t.shallower}->{t.deeper} has crossing bands"
        )


@pytest.mark.parametrize("name", sorted(ALL_PROFILES))
def test_move_cost_uses_the_measured_link_not_a_constant(name):
    """Halving the measured link bandwidth must double every move time."""
    inputs = ALL_PROFILES[name]()
    slow = solve_layout_ladder(
        ALL_PROFILES[name]().__class__(
            **{
                **inputs.__dict__,
                "link_mib_per_s": tuple(b / 2.0 for b in inputs.link_mib_per_s),
            }
        )
    )
    fast = solve_layout_ladder(inputs)
    by_pair = {(t.shallower, t.deeper): t for t in fast.transitions}
    for t in slow.transitions:
        ref = by_pair.get((t.shallower, t.deeper))
        if ref is None:
            continue
        assert t.weight_move_ms == pytest.approx(2.0 * ref.weight_move_ms, rel=1e-9)


def test_rank0_cap_is_derived_not_typed():
    """The 42-layer cap must fall out of arithmetic, on any card.

    Give rank0 a much larger card and the ladder must reach deeper cuts than it
    does on the rig. Nothing may pin the depth to this rig's number.
    """
    small = solve_layout_ladder(_rig_inputs())
    big = solve_layout_ladder(
        _rig_inputs(
            pool=FamilyPoolModel(
                free_mib=(90000.0, 16051.3, 13984.3),
                weight_mib_per_layer=450.7,
                kv_mib_per_token_per_attn_layer=2 * 4 * 256 * 2 / (1024.0 * 1024.0),
                layer_families=RIG_FAMILIES,
            )
        )
    )
    assert max(r.counts[0] for r in big.rungs) > max(r.counts[0] for r in small.rungs)


def test_pure_attention_model_has_no_gdn_plateau():
    """0-GDN: every layer carries KV, so no cut is free of pool cost.

    On the hybrid rig, moving a run of linear layers changes speed without
    changing the attention count -- the plateau the ladder exploits. With every
    layer an attention layer that plateau cannot exist, and the solver must
    reflect that rather than inheriting the hybrid's shape.
    """
    ladder = solve_layout_ladder(_foreign_pure_attention())
    for a, b in zip(ladder.rungs, ladder.rungs[1:]):
        # Every deeper rung moves rank0's attention count, hence its pool.
        assert b.attn_counts[0] > a.attn_counts[0]


def test_admission_threshold_never_exceeds_the_rung_pool():
    """A rung may not admit more live tokens than it can physically hold."""
    for name, factory in ALL_PROFILES.items():
        ladder = solve_layout_ladder(factory())
        for r in ladder.rungs:
            assert r.admit_up_to_tokens <= r.pool_tokens, name


def test_a_rung_whose_move_cannot_be_funded_is_pruned():
    """CAN-FAIL PROOF: an impossibly slow link must collapse the ladder.

    With a link slow enough that a rung change costs more than the fill window
    it buys, the bands cross and the pair is not a usable step. The solver must
    drop it rather than emit a ladder that oscillates on metal.
    """
    fast = solve_layout_ladder(_rig_inputs())
    crawling = solve_layout_ladder(
        _rig_inputs(link_mib_per_s=(0.5, 0.5, 0.5), prefill_tokens_per_s=200_000.0)
    )
    assert len(crawling.transitions) < len(fast.transitions)
