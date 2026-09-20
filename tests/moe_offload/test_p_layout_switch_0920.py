"""P-layout switch inside the running PP processes (user order 2026-09-20).

Hermetic: no CUDA, no model, no process group. Every device interaction goes
through an injected fake, which is the point -- the mover is testable exactly
because ``regions`` and ``DeviceOps`` are parameters rather than imports.

The rig numbers that appear here are MEASURED ones, carried in on purpose so
the worked example pins the design rather than illustrating it:
PLAN_BAR1_LANES_0918's BAR1 table (5090 -> 3080-x4 at 7.13 GB/s, 3080-x4 ->
3080-x8 at 6.58, 5090 -> 3080-x8 at 14.25) and the 64-layer / 27 GB INT8 gdncov
checkpoint.
"""

from __future__ import annotations

import os

import pytest

from sglang.srt.model_executor.layout_boundary import (
    LayoutBoundaryActuator,
    LayoutBoundaryError,
    LayoutBoundaryTorn,
)
from sglang.srt.planner.pp_cut import prefill_timing_from_measurement
from sglang.srt.weg2.p_layout_switch import (
    ACTION_FAST_THEN_LEAN,
    ACTION_STAY,
    ACTION_TO_FAST,
    BandGeometry,
    LayerBytes,
    LinkRates,
    MissingLayerBytes,
    MoveOp,
    PLayout,
    PLayoutRankDisagree,
    RingState,
    SwitchCalibration,
    SwitchNotQuiescent,
    UnalignedLayout,
    UnpricedFlip,
    UnpricedLink,
    accept_pp0_verdict,
    band_aligned_candidates,
    bands_freed,
    breakeven_tokens,
    course_breakeven_tokens,
    decide,
    emit_move_ops,
    layer_bytes_from_manifests,
    layer_index_of,
    move_seconds,
    plan_layer_moves,
    pp0_broadcast,
    preconditions_digest,
    prefill_seconds,
    require_switchable,
    run_move_ops,
    vram_freed_bytes,
)

N_LAYERS = 64
#: 27 GB INT8 gdncov over 64 layers -- the ~0.42 GB/layer of the briefing, used
#: as a UNIFORM stand-in. Real inventories are not uniform (an attention layer
#: and a GDN layer differ), which is why the code reads the manifest; the
#: uniform case is used here only where the per-layer split is not what is
#: under test.
BYTES_PER_LAYER = 420_000_000

P39 = PLayout(name="p39", counts=(39, 13, 12))
P32 = PLayout(name="p32", counts=(32, 16, 16))


def uniform_bytes(n_layers: int = N_LAYERS, per: int = BYTES_PER_LAYER) -> LayerBytes:
    return LayerBytes(
        per_layer={i: per for i in range(n_layers)}, stage_invariant_bytes=1_000_000
    )


#: The measured BAR1 directions of this rig, as stage pairs under
#: stage0 = 5090, stage1 = 3080-x4, stage2 = 3080-x8.
RIG_RATES = LinkRates(
    gbytes_per_s={
        (0, 1): 7.13,
        (1, 0): 6.56,
        (0, 2): 14.25,
        (2, 0): 13.15,
        (1, 2): 6.58,
        (2, 1): 6.58,
    }
)

GEOM8 = BandGeometry(layers_per_chunk=8, chunk_count=8)


# ---------------------------------------------------------------------------
# The layout type
# ---------------------------------------------------------------------------


def test_bounds_ranges_and_ownership_are_consistent():
    assert P39.bounds == (39, 52, 64)
    assert P39.ranges() == ((0, 39), (39, 52), (52, 64))
    assert P39.n_layers == 64
    assert P39.owner_of(0) == 0
    assert P39.owner_of(38) == 0
    assert P39.owner_of(39) == 1
    assert P39.owner_of(51) == 1
    assert P39.owner_of(52) == 2
    assert P39.owner_of(63) == 2
    # Ownership and the ranges must be the same statement, on every layer.
    for layer in range(N_LAYERS):
        stage = P39.owner_of(layer)
        start, end = P39.range_of(stage)
        assert start <= layer < end


def test_layout_refuses_an_empty_stage():
    # pp_cut rev 4: a stage with no layers has no full-attention layers either,
    # and HybridLinearKVPool divides by that count.
    with pytest.raises(UnalignedLayout, match="at least one layer"):
        PLayout(name="bad", counts=(64, 0, 0))


def test_layout_refuses_a_nameless_layout():
    with pytest.raises(UnalignedLayout, match="needs a name"):
        PLayout(name="  ", counts=(32, 16, 16))


def test_as_ratio_round_trips_the_flag_string():
    assert P39.as_ratio() == "39,13,12"


def test_switching_between_different_models_is_refused():
    with pytest.raises(UnalignedLayout, match="different models"):
        plan_layer_moves(
            P39, PLayout(name="short", counts=(10, 10, 10)), uniform_bytes()
        )


def test_switching_between_different_stage_counts_is_refused():
    with pytest.raises(UnalignedLayout, match="does not add or remove a rank"):
        plan_layer_moves(P39, PLayout(name="two", counts=(32, 32)), uniform_bytes())


# ---------------------------------------------------------------------------
# Bands: the unit in which VRAM is actually returned
# ---------------------------------------------------------------------------


def test_band_of_agrees_with_the_live_weight_chunk_tag():
    """The reproduced formula must BE ``weight_chunk_tag``, not resemble it."""
    from sglang.srt.managers import weg2_memory_saver as ms

    old = {
        ms.WEIGHT_CHUNK_ENV_LAYERS: os.environ.get(ms.WEIGHT_CHUNK_ENV_LAYERS),
        ms.WEIGHT_CHUNK_ENV_COUNT: os.environ.get(ms.WEIGHT_CHUNK_ENV_COUNT),
    }
    os.environ[ms.WEIGHT_CHUNK_ENV_LAYERS] = "8"
    os.environ[ms.WEIGHT_CHUNK_ENV_COUNT] = "8"
    try:
        live = BandGeometry.from_env()
        assert live == GEOM8
        for layer in range(N_LAYERS):
            assert live.tag_of(layer) == ms.weight_chunk_tag(layer)
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_chunking_off_is_a_refusal_not_a_geometry():
    with pytest.raises(UnalignedLayout, match="chunking-OFF"):
        BandGeometry(layers_per_chunk=0, chunk_count=0)


def test_a_seven_layer_move_frees_nothing_and_an_eight_layer_move_frees_a_band():
    """THE central fact of the design, as a test rather than a comment.

    Stage 0 gives VRAM back only for a band it stops owning ENTIRELY, because
    the memory saver pauses whole tags. 39 -> 33 moves six layers off the 5090
    and frees zero; 39 -> 32 moves seven and frees a whole 8-layer band, because
    32 is where band 4 stops being stage 0's at all.
    """
    bytes_ = uniform_bytes()
    p33 = PLayout(name="p33", counts=(33, 16, 15))

    assert bands_freed(P39, p33, 0, GEOM8) == ()
    assert vram_freed_bytes(P39, p33, 0, GEOM8, bytes_) == 0

    assert bands_freed(P39, P32, 0, GEOM8) == (4,)
    # The WHOLE band is released, all eight layers of it -- not the seven that
    # changed owner. That difference is the reason the function exists.
    assert vram_freed_bytes(P39, P32, 0, GEOM8, bytes_) == 8 * BYTES_PER_LAYER


def test_freed_bands_are_counted_per_stage_not_globally():
    bytes_ = uniform_bytes()
    # Stage 1 goes from [39,52) to [32,48): it gains band 4 and loses band 6.
    assert bands_freed(P39, P32, 1, GEOM8) == (6,)
    assert vram_freed_bytes(P39, P32, 1, GEOM8, bytes_) == 8 * BYTES_PER_LAYER
    # Stage 2 goes from [52,64) to [48,64): it loses nothing, it only gains.
    assert bands_freed(P39, P32, 2, GEOM8) == ()


def test_band_aligned_candidates_offers_only_band_edges_below_the_cap():
    cands = band_aligned_candidates(P39, GEOM8, max_stage0_layers=32)
    assert cands, "there are band-aligned cuts below 39; the enumerator found none"
    for layout in cands:
        assert layout.counts[0] % GEOM8.layers_per_chunk == 0
        assert layout.counts[0] < P39.counts[0]
        assert layout.counts[0] <= 32
        assert layout.n_layers == N_LAYERS
        # Every offered candidate must actually free a band; that is the point
        # of restricting the enumeration.
        assert bands_freed(P39, layout, 0, GEOM8)
    assert (32, 16, 16) in {c.counts for c in cands}


# ---------------------------------------------------------------------------
# Layer bytes from the manifests
# ---------------------------------------------------------------------------


class _Piece:
    def __init__(self, param_name, nbytes):
        self.param_name = param_name
        self.nbytes = nbytes


class _Manifest:
    def __init__(self, pieces):
        self.pieces = pieces


def test_layer_index_is_read_off_the_dotted_segment():
    assert layer_index_of("model.layers.37.self_attn.qkv_proj.weight") == 37
    assert layer_index_of("model.layers.0.mlp.down_proj.weight_scale") == 0
    assert layer_index_of("layers.5.norm.weight") == 5
    # Stage-invariant parameters carry no layer.
    assert layer_index_of("model.embed_tokens.weight") is None
    assert layer_index_of("lm_head.weight") is None
    assert layer_index_of("model.norm.weight") is None
    # A name that merely contains the word is not a layer.
    assert layer_index_of("model.num_layers_config") is None


def test_manifest_inventory_counts_scales_norms_and_biases_into_the_layer():
    """The reason the number is read off the manifest at all.

    A per-layer byte count computed from a layer's nominal weight shapes misses
    the quantisation scales, the biases and the norms -- which under INT8 are
    not a rounding error -- and a move priced without them under-copies.
    """
    manifests = [
        _Manifest(
            [
                _Piece("model.layers.3.self_attn.qkv_proj.weight", 1000),
                _Piece("model.layers.3.self_attn.qkv_proj.weight_scale", 50),
                _Piece("model.layers.3.self_attn.qkv_proj.bias", 20),
                _Piece("model.layers.3.input_layernorm.weight", 5),
                _Piece("model.embed_tokens.weight", 9999),
            ]
        )
    ]
    inv = layer_bytes_from_manifests(manifests)
    assert inv.of(3) == 1000 + 50 + 20 + 5
    assert inv.stage_invariant_bytes == 9999
    assert inv.n_layers == 1


def test_manifest_inventory_sums_across_stages():
    inv = layer_bytes_from_manifests(
        [
            _Manifest([_Piece("model.layers.0.w", 10)]),
            _Manifest([_Piece("model.layers.1.w", 20)]),
        ]
    )
    assert inv.of(0) == 10
    assert inv.of(1) == 20


def test_an_unknown_layer_is_a_refusal_not_a_guess():
    inv = LayerBytes(per_layer={0: 10, 1: 20})
    with pytest.raises(MissingLayerBytes, match="cannot be priced"):
        inv.of(7)


def test_inventory_digest_is_stable_and_discriminating():
    a = LayerBytes(per_layer={0: 10, 1: 20})
    b = LayerBytes(per_layer={1: 20, 0: 10})
    c = LayerBytes(per_layer={0: 10, 1: 21})
    assert a.digest() == b.digest()
    assert a.digest() != c.digest()


# ---------------------------------------------------------------------------
# The move plan
# ---------------------------------------------------------------------------


def test_a_single_boundary_move_cascades_across_three_stages():
    """Stage 1 both gains and loses, which a boundary-delta plan would miss."""
    plan = plan_layer_moves(P39, P32, uniform_bytes())
    gained_by_1 = set(plan.layers_gained_by(1))
    lost_by_1 = set(plan.layers_lost_by(1))
    assert gained_by_1 == set(range(32, 39))  # from stage 0
    assert lost_by_1 == set(range(48, 52))  # to stage 2
    assert set(plan.layers_gained_by(2)) == set(range(48, 52))
    assert set(plan.layers_lost_by(0)) == set(range(32, 39))
    assert plan.by_pair() == {
        (0, 1): 7 * BYTES_PER_LAYER,
        (1, 2): 4 * BYTES_PER_LAYER,
    }
    assert plan.total_bytes == 11 * BYTES_PER_LAYER


def test_moving_to_the_same_layout_moves_nothing():
    plan = plan_layer_moves(
        P39, PLayout(name="same", counts=(39, 13, 12)), uniform_bytes()
    )
    assert plan.moves == ()
    assert plan.total_bytes == 0
    assert move_seconds(plan, RIG_RATES) == 0.0


def test_move_seconds_is_the_busiest_link_direction():
    """Stage 1 receives 7 layers and sends 4 on ONE x4 link, duplex.

    Inbound 7 x 0.42 GB at 7.13 GB/s = 0.412 s; outbound 4 x 0.42 GB at 6.58 =
    0.255 s. The two run concurrently (full duplex), so the move costs the
    slower of them, not their sum.
    """
    plan = plan_layer_moves(P39, P32, uniform_bytes())
    inbound_1 = 7 * BYTES_PER_LAYER / (7.13 * 1e9)
    outbound_1 = 4 * BYTES_PER_LAYER / (6.58 * 1e9)
    assert inbound_1 > outbound_1
    assert move_seconds(plan, RIG_RATES) == pytest.approx(inbound_1, rel=1e-9)
    assert move_seconds(plan, RIG_RATES) == pytest.approx(0.412, abs=0.005)


def test_same_direction_peers_share_the_link_and_are_summed():
    """Two destinations off one card queue on that card's single PCIe link."""
    rates = LinkRates(gbytes_per_s={(0, 1): 10.0, (0, 2): 10.0, (1, 2): 10.0})
    frm = PLayout(name="a", counts=(4, 1, 1))
    to = PLayout(name="b", counts=(1, 1, 4))
    inv = LayerBytes(per_layer={i: 10_000_000_000 for i in range(6)})
    plan = plan_layer_moves(frm, to, inv)
    assert plan.by_pair() == {
        (0, 1): 10_000_000_000,
        (0, 2): 20_000_000_000,
        (1, 2): 10_000_000_000,
    }
    # Stage 0 sends 10 GB to stage 1 AND 20 GB to stage 2 over ONE outbound
    # link: 1 s + 2 s = 3 s. Stage 2 receives 20 + 10 on one inbound link: also
    # 3 s. Both are the binding load; neither is hidden by the other.
    assert move_seconds(plan, rates) == pytest.approx(3.0, rel=1e-9)


def test_an_unmeasured_link_is_refused_never_interpolated():
    rates = LinkRates(gbytes_per_s={(0, 1): 7.13})
    plan = plan_layer_moves(P39, P32, uniform_bytes())
    with pytest.raises(UnpricedLink, match="Refusing to interpolate"):
        move_seconds(plan, rates)


def test_a_nonpositive_rate_is_a_missing_measurement():
    with pytest.raises(UnpricedLink, match="missing measurement"):
        LinkRates(gbytes_per_s={(0, 1): 0.0})


def test_link_rates_are_directional():
    assert RIG_RATES.rate(0, 1) != RIG_RATES.rate(1, 0)


# ---------------------------------------------------------------------------
# Prefill cost
# ---------------------------------------------------------------------------


def rig_calibration(**overrides):
    """A calibration in the shape the launcher would hand over.

    The timing is calibrated from ONE measured cut, which is what
    ``prefill_timing_from_measurement`` is for and whose optimism its own
    docstring states (fixed_fraction 0 credits a reallocation with the full
    per-layer saving).

    The measured cut is NEAR-BALANCED in TIME, which is the only shape that
    makes 39/13/12 a sensible cut in the first place: the three stages take
    roughly the same milliseconds while holding 39, 13 and 12 layers, i.e. the
    5090 is about three times faster per layer than the 3080s. A calibration
    whose stage times were proportional to the LAYER COUNTS would imply equal
    per-layer speed on all three cards -- and under that implication the
    pipelined max is minimised by an EVEN split, so p32 would come out faster
    than p39 and the whole trade would invert. That is not a hypothetical: it
    is what this fixture said in its first draft, and the sign error survived
    until a test asserted the direction.
    """
    timing = prefill_timing_from_measurement(
        counts=P39.counts, stage_ms=(392.0, 386.0, 380.0)
    )
    base = dict(
        timing=timing,
        chunk_tokens=4096,
        rates=RIG_RATES,
        layer_bytes=uniform_bytes(),
        geom=GEOM8,
        flip_seconds={"p39": 4.0, "p32": 3.1},
        margin_s=0.0,
    )
    base.update(overrides)
    return SwitchCalibration(**base)


def test_prefill_is_charged_per_whole_chunk():
    calib = rig_calibration()
    one = prefill_seconds(P39, 1, calib)
    full = prefill_seconds(P39, 4096, calib)
    just_over = prefill_seconds(P39, 4097, calib)
    assert one == pytest.approx(full)
    assert just_over == pytest.approx(2 * full)
    assert prefill_seconds(P39, 0, calib) == 0.0


def test_the_lean_layout_is_the_slower_one_on_this_calibration():
    """Sanity: the whole trade only exists because P32 prefills slower."""
    calib = rig_calibration()
    assert prefill_seconds(P32, 100_000, calib) > prefill_seconds(P39, 100_000, calib)


def test_a_layout_with_no_measured_flip_is_refused():
    calib = rig_calibration(flip_seconds={"p39": 4.0})
    with pytest.raises(UnpricedFlip, match="no measured flip time"):
        calib.flip_of(P32)


def test_a_negative_margin_is_refused():
    with pytest.raises(Exception, match="negative"):
        rig_calibration(margin_s=-1.0)


def test_zero_chunk_tokens_is_refused():
    with pytest.raises(Exception, match="chunks of zero tokens"):
        rig_calibration(chunk_tokens=0)


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_a_small_backlog_stays_on_the_lean_layout():
    """The order's own case: do not pay the longer flip for a short prefill."""
    calib = rig_calibration()
    v = decide(current=P32, fast=P39, lean=P32, pending_tokens=4096, calib=calib)
    assert v.action == ACTION_STAY
    assert not v.switches
    assert v.chosen.total_s <= min(p.total_s for p in v.plans)
    assert "WEG2-PSWITCH action=stay" in v.as_line()


def test_a_large_backlog_leaves_the_lean_layout():
    calib = rig_calibration()
    v = decide(current=P32, fast=P39, lean=P32, pending_tokens=4_000_000, calib=calib)
    assert v.switches
    assert v.chosen.end_layout in ("p39", "p32")
    assert v.gain_s > 0


def test_fast_then_lean_is_found_when_the_flip_delta_is_large():
    """The course a two-way rule cannot express.

    With an expensive fast-layout flip, running the backlog on p39 and moving
    BACK to p32 before the flip beats both pure courses: it buys the fast
    prefill without paying the long flip, for the price of a second move.
    """
    calib = rig_calibration(flip_seconds={"p39": 30.0, "p32": 3.1})
    v = decide(current=P32, fast=P39, lean=P32, pending_tokens=4_000_000, calib=calib)
    assert v.action == ACTION_FAST_THEN_LEAN
    assert v.chosen.end_layout == "p32"
    assert v.chosen.back_switch_s > 0
    assert v.chosen.flip_s == 3.1
    # It must genuinely beat the pure fast course, not merely be listed.
    pure_fast = next(p for p in v.plans if p.action == ACTION_TO_FAST)
    assert v.chosen.total_s < pure_fast.total_s


def test_pure_fast_wins_when_the_flip_delta_is_small():
    calib = rig_calibration(flip_seconds={"p39": 3.15, "p32": 3.1})
    v = decide(current=P32, fast=P39, lean=P32, pending_tokens=4_000_000, calib=calib)
    assert v.action == ACTION_TO_FAST
    assert v.chosen.back_switch_s == 0.0


def test_the_deadband_suppresses_a_marginal_switch():
    calib = rig_calibration()
    n = breakeven_tokens(P32, P39, calib)
    assert n is not None
    # Just past the break-even, an unguarded rule switches...
    unguarded = decide(
        current=P32, fast=P39, lean=P32, pending_tokens=n + 4096, calib=calib
    )
    assert unguarded.switches
    assert "margin=unguarded" in unguarded.as_line()
    # ...and a rule with a deadband wider than the gain does not.
    guarded = decide(
        current=P32,
        fast=P39,
        lean=P32,
        pending_tokens=n + 4096,
        calib=rig_calibration(margin_s=unguarded.gain_s + 1.0),
    )
    assert guarded.action == ACTION_STAY
    assert "deadband" in guarded.why
    assert "margin=guarded" in guarded.as_line()


def test_the_decision_is_re_evaluated_from_the_layout_actually_in_force():
    """A new arrival while P is already on the fast layout must not re-charge
    the switch it already paid for."""
    calib = rig_calibration()
    from_lean = decide(
        current=P32, fast=P39, lean=P32, pending_tokens=200_000, calib=calib
    )
    from_fast = decide(
        current=P39, fast=P39, lean=P32, pending_tokens=200_000, calib=calib
    )
    stay_on_fast = next(p for p in from_fast.plans if p.action == ACTION_STAY)
    assert stay_on_fast.switch_s == 0.0
    assert stay_on_fast.end_layout == "p39"
    # And the sunk move is not counted again.
    assert from_fast.incumbent.moved_bytes == 0
    assert from_lean.incumbent.moved_bytes == 0


def test_a_growing_backlog_can_tip_a_decision_that_was_stay():
    """The order's opening case: pending arrive DURING the prefill."""
    calib = rig_calibration()
    small = decide(current=P32, fast=P39, lean=P32, pending_tokens=4096, calib=calib)
    assert small.action == ACTION_STAY
    grown = decide(
        current=P32, fast=P39, lean=P32, pending_tokens=2_000_000, calib=calib
    )
    assert grown.switches


def test_breakeven_refuses_when_the_fast_layout_is_not_faster():
    calib = rig_calibration()
    # Ask for the break-even of moving to a SLOWER layout: there is none.
    assert breakeven_tokens(P39, P32, calib) is None


def test_fast_then_lean_breaks_even_earlier_than_pure_fast():
    """It does not pay the flip delta, so it starts paying sooner.

    The property that makes a single per-layout break-even wrong.
    """
    calib = rig_calibration(flip_seconds={"p39": 30.0, "p32": 3.1})
    pure = course_breakeven_tokens(P32, P39, P39, calib)
    mixed = course_breakeven_tokens(P32, P39, P32, calib)
    assert pure is not None and mixed is not None
    assert mixed < pure
    reported = decide(
        current=P32, fast=P39, lean=P32, pending_tokens=0, calib=calib
    ).breakeven_tokens
    assert reported == mixed


def test_breakeven_agrees_with_the_rule_within_one_chunk():
    """The continuous solve and the chunk-quantised rule must not disagree by
    more than the quantisation, or one of them is wrong.

    Asked of the verdict's OWN break-even, which is the minimum over the
    courses on offer. Asking it of the pure to_fast break-even instead is what
    the first draft did, and it fails: fast_then_lean does not pay the flip
    delta, so it starts paying at a smaller backlog and the rule switches below
    the to_fast crossing -- correctly.
    """
    calib = rig_calibration()
    n = decide(
        current=P32, fast=P39, lean=P32, pending_tokens=0, calib=calib
    ).breakeven_tokens
    assert n is not None
    below = decide(
        current=P32, fast=P39, lean=P32, pending_tokens=max(0, n - 4096), calib=calib
    )
    above = decide(
        current=P32, fast=P39, lean=P32, pending_tokens=n + 2 * 4096, calib=calib
    )
    assert below.action == ACTION_STAY
    assert above.switches


def test_every_plan_reports_the_vram_it_frees_on_stage_zero():
    calib = rig_calibration()
    v = decide(current=P39, fast=P39, lean=P32, pending_tokens=1000, calib=calib)
    to_lean = next(p for p in v.plans if p.end_layout == "p32")
    assert to_lean.vram_freed_bytes == 8 * BYTES_PER_LAYER
    stay = next(p for p in v.plans if p.action == ACTION_STAY)
    assert stay.vram_freed_bytes == 0


# ---------------------------------------------------------------------------
# Quiescence
# ---------------------------------------------------------------------------


def test_a_drained_ring_at_a_chunk_boundary_is_switchable():
    require_switchable(
        RingState(at_chunk_boundary=True, inflight_microbatches=0, inflight_requests=0)
    )


def test_mid_chunk_is_refused():
    with pytest.raises(SwitchNotQuiescent, match="chunk boundary"):
        require_switchable(
            RingState(
                at_chunk_boundary=False, inflight_microbatches=0, inflight_requests=0
            )
        )


def test_an_undrained_ring_is_refused():
    with pytest.raises(SwitchNotQuiescent, match="still in\nthe PP ring|PP ring"):
        require_switchable(
            RingState(
                at_chunk_boundary=True, inflight_microbatches=2, inflight_requests=0
            )
        )


def test_inflight_requests_are_refused_for_the_gdn_state_reason():
    with pytest.raises(SwitchNotQuiescent, match="recurrent state"):
        require_switchable(
            RingState(
                at_chunk_boundary=True, inflight_microbatches=0, inflight_requests=1
            )
        )


# ---------------------------------------------------------------------------
# Rank uniformity: one verdict at PP0, checked everywhere
# ---------------------------------------------------------------------------


def digest_for(**overrides):
    base = dict(
        current=P32,
        fast=P39,
        lean=P32,
        geom=GEOM8,
        layer_bytes=uniform_bytes(),
        epoch=7,
        chunk_index=3,
    )
    base.update(overrides)
    return preconditions_digest(**base)


def test_agreeing_ranks_accept_pp0s_action():
    calib = rig_calibration()
    v = decide(current=P32, fast=P39, lean=P32, pending_tokens=4_000_000, calib=calib)
    d = digest_for()
    record = pp0_broadcast(v, d)
    for rank in (1, 2):
        assert accept_pp0_verdict(record, d, rank) == v.action


def test_a_rank_holding_a_different_layout_stops_the_group():
    v = decide(
        current=P32,
        fast=P39,
        lean=P32,
        pending_tokens=4_000_000,
        calib=rig_calibration(),
    )
    record = pp0_broadcast(v, digest_for())
    other = digest_for(lean=PLayout(name="p24", counts=(24, 20, 20)))
    with pytest.raises(PLayoutRankDisagree, match="owned twice or owned by"):
        accept_pp0_verdict(record, other, 2)


def test_a_verdict_from_a_previous_chunk_is_refused():
    """The digest carries WHICH switch point it is, so a stale verdict cannot
    be applied to this one."""
    v = decide(
        current=P32,
        fast=P39,
        lean=P32,
        pending_tokens=4_000_000,
        calib=rig_calibration(),
    )
    stale = pp0_broadcast(v, digest_for(chunk_index=2))
    with pytest.raises(PLayoutRankDisagree):
        accept_pp0_verdict(stale, digest_for(chunk_index=3), 1)


def test_a_verdict_from_a_previous_epoch_is_refused():
    v = decide(
        current=P32,
        fast=P39,
        lean=P32,
        pending_tokens=4_000_000,
        calib=rig_calibration(),
    )
    stale = pp0_broadcast(v, digest_for(epoch=6))
    with pytest.raises(PLayoutRankDisagree):
        accept_pp0_verdict(stale, digest_for(epoch=7), 1)


def test_a_differing_byte_inventory_stops_the_group():
    v = decide(
        current=P32,
        fast=P39,
        lean=P32,
        pending_tokens=4_000_000,
        calib=rig_calibration(),
    )
    record = pp0_broadcast(v, digest_for())
    other = digest_for(layer_bytes=uniform_bytes(per=BYTES_PER_LAYER + 1))
    with pytest.raises(PLayoutRankDisagree):
        accept_pp0_verdict(record, other, 1)


def test_a_differing_band_geometry_stops_the_group():
    v = decide(
        current=P32,
        fast=P39,
        lean=P32,
        pending_tokens=4_000_000,
        calib=rig_calibration(),
    )
    record = pp0_broadcast(v, digest_for())
    other = digest_for(geom=BandGeometry(layers_per_chunk=4, chunk_count=16))
    with pytest.raises(PLayoutRankDisagree):
        accept_pp0_verdict(record, other, 1)


def test_an_unnameable_action_is_refused_rather_than_read_as_stay():
    with pytest.raises(PLayoutRankDisagree, match="cannot name"):
        accept_pp0_verdict(
            {"action": "wobble", "digest": digest_for()}, digest_for(), 1
        )


def test_the_broadcast_carries_no_inputs_for_downstream_to_re_derive_from():
    """#968: downstream is verdict-FREE. Give it nothing to decide with."""
    v = decide(
        current=P32,
        fast=P39,
        lean=P32,
        pending_tokens=4_000_000,
        calib=rig_calibration(),
    )
    record = pp0_broadcast(v, digest_for())
    assert set(record) == {
        "action",
        "to",
        "run_on",
        "digest",
        "pending_tokens",
        "gain_s",
    }
    for forbidden in ("rates", "flip_seconds", "timing", "plans", "layer_bytes"):
        assert forbidden not in record


# ---------------------------------------------------------------------------
# The mover
# ---------------------------------------------------------------------------


class _Window:
    def __init__(self, dev_ptr, direct=False):
        self.dev_ptr = dev_ptr
        self.direct = direct


class _Regions:
    """Injected stand-in for the mapped BAR1 windows and layer regions."""

    def __init__(self, windows):
        self._windows = windows

    def src_ptr(self, layer_id):
        return 0x100000 + int(layer_id) * 0x1000

    def dst_ptr(self, layer_id):
        return 0x900000 + int(layer_id) * 0x1000

    def window(self, src_stage, dst_stage):
        return self._windows.get((int(src_stage), int(dst_stage)))


class _FakeDeviceOps:
    """Records what ``memcpy_async`` was asked to do. No CUDA anywhere."""

    def __init__(self):
        self.calls = []

    def memcpy_async(self, dst, src, nbytes, stream):
        self.calls.append((dst, src, nbytes, stream))


def test_a_sender_deposits_into_the_peer_window():
    plan = plan_layer_moves(P39, P32, uniform_bytes())
    regions = _Regions({(0, 1): _Window(0xB0000), (1, 2): _Window(0xC0000)})
    ops = emit_move_ops(plan, regions, this_stage=0)
    assert {o.kind for o in ops} == {"deposit"}
    assert [o.layer_id for o in ops] == list(range(32, 39))
    assert all(o.dst == 0xB0000 for o in ops)
    assert all(o.src == regions.src_ptr(o.layer_id) for o in ops)


def test_a_receiver_collects_out_of_its_own_window():
    plan = plan_layer_moves(P39, P32, uniform_bytes())
    regions = _Regions({(0, 1): _Window(0xB0000), (1, 2): _Window(0xC0000)})
    ops = emit_move_ops(plan, regions, this_stage=2)
    assert {o.kind for o in ops} == {"collect"}
    assert [o.layer_id for o in ops] == list(range(48, 52))
    assert all(o.src == 0xC0000 for o in ops)
    assert all(o.dst == regions.dst_ptr(o.layer_id) for o in ops)


def test_a_stage_that_both_gains_and_loses_emits_both_kinds():
    plan = plan_layer_moves(P39, P32, uniform_bytes())
    regions = _Regions({(0, 1): _Window(0xB0000), (1, 2): _Window(0xC0000)})
    ops = emit_move_ops(plan, regions, this_stage=1)
    kinds = {o.kind for o in ops}
    assert kinds == {"collect", "deposit"}
    collected = [o.layer_id for o in ops if o.kind == "collect"]
    deposited = [o.layer_id for o in ops if o.kind == "deposit"]
    assert collected == list(range(32, 39))
    assert deposited == list(range(48, 52))


def test_the_5090_receives_directly_and_the_receiver_collects_nothing():
    """PLAN_BAR1_LANES_0918: the 5090's 32 GiB BAR covers its whole VRAM, so the
    sender writes straight into the target region -- 'kein Collect'.

    A collect here would not merely be wasted: src and dst would be the same
    address.
    """
    frm = PLayout(name="a", counts=(30, 18, 16))
    to = PLayout(name="b", counts=(34, 14, 16))  # stage 0 GAINS layers 30-33
    inv = uniform_bytes()
    plan = plan_layer_moves(frm, to, inv)
    assert set(plan.layers_gained_by(0)) == set(range(30, 34))
    regions = _Regions({(1, 0): _Window(0xD0000, direct=True)})
    sender_ops = emit_move_ops(plan, regions, this_stage=1)
    assert {o.kind for o in sender_ops} == {"direct"}
    receiver_ops = emit_move_ops(plan, regions, this_stage=0)
    assert receiver_ops == ()


def test_a_missing_window_is_refused_not_silently_bounced_via_host():
    plan = plan_layer_moves(P39, P32, uniform_bytes())
    regions = _Regions({(0, 1): _Window(0xB0000)})  # (1,2) absent
    with pytest.raises(UnpricedLink, match="no BAR1 window"):
        emit_move_ops(plan, regions, this_stage=1)


def test_run_move_ops_issues_exactly_the_planned_copies():
    ops = (
        MoveOp(
            kind="deposit", layer_id=32, dst=0xB0, src=0xA0, nbytes=100, peer_stage=1
        ),
        MoveOp(
            kind="collect", layer_id=48, dst=0xD0, src=0xC0, nbytes=200, peer_stage=2
        ),
    )
    dev = _FakeDeviceOps()
    moved = run_move_ops(ops, dev, stream=0x1234)
    assert moved == 300
    assert dev.calls == [(0xB0, 0xA0, 100, 0x1234), (0xD0, 0xC0, 200, 0x1234)]


def test_the_whole_switch_moves_exactly_the_planned_bytes_across_all_stages():
    """End to end on the fakes: the bytes issued must equal the bytes planned.

    Counted over DEPOSITS and DIRECTS only -- a collect re-copies bytes that are
    already on the card, so counting it too would double the total and make the
    move look twice as expensive as the plan priced it.
    """
    plan = plan_layer_moves(P39, P32, uniform_bytes())
    regions = _Regions({(0, 1): _Window(0xB0000), (1, 2): _Window(0xC0000)})
    issued = 0
    for stage in (0, 1, 2):
        ops = emit_move_ops(plan, regions, this_stage=stage)
        dev = _FakeDeviceOps()
        run_move_ops(ops, dev, stream=1)
        issued += sum(o.nbytes for o in ops if o.kind in ("deposit", "direct"))
        assert len(dev.calls) == len(ops)
    assert issued == plan.total_bytes


# ---------------------------------------------------------------------------
# The actuator in MOVING mode (#704 extended)
# ---------------------------------------------------------------------------


class _Param:
    pass


class _Layer:
    """A real decoder layer: owns a parameter."""

    def parameters(self):
        return iter([_Param()])


class _Missing:
    """A PPMissingLayer: a parameterless pass-through."""

    def parameters(self):
        return iter([])


class _Model:
    def __init__(self, real_layers, n_layers=8):
        self.layers = [
            _Layer() if i in real_layers else _Missing() for i in range(n_layers)
        ]
        self._start_layer = 0
        self._end_layer = n_layers


def test_moving_mode_demands_a_residency_probe():
    model = _Model(real_layers=set(range(8)))
    with pytest.raises(LayoutBoundaryError, match="needs a residency_probe"):
        LayoutBoundaryActuator(
            model,
            {"wide": (0, 6), "narrow": (0, 4)},
            "wide",
            mover=lambda a, d, r: 0,
        )


def test_the_copy_nothing_path_is_untouched_without_a_mover():
    model = _Model(real_layers=set(range(8)))
    act = LayoutBoundaryActuator(model, {"wide": (0, 6), "narrow": (0, 4)}, "wide")
    report = act.flip("narrow", quiescent=True)
    assert report.bytes_copied == 0
    assert model._start_layer == 0 and model._end_layer == 4


def test_moving_mode_calls_the_mover_and_reports_its_bytes():
    model = _Model(real_layers=set(range(8)))
    resident = {0, 1, 2, 3}
    seen = {}

    def mover(activated, deactivated, rng):
        seen["activated"] = activated
        seen["deactivated"] = deactivated
        seen["range"] = rng
        resident.update(activated)
        resident.difference_update(deactivated)
        return 4242

    act = LayoutBoundaryActuator(
        model,
        {"narrow": (0, 4), "wide": (0, 6)},
        "narrow",
        mover=mover,
        residency_probe=lambda i: i in resident,
    )
    report = act.flip("wide", quiescent=True)
    assert seen["activated"] == (4, 5)
    assert seen["deactivated"] == ()
    assert seen["range"] == (0, 6)
    assert report.bytes_copied == 4242
    assert model._end_layer == 6


def test_moving_mode_refuses_to_start_from_a_non_resident_rung():
    model = _Model(real_layers=set(range(8)))
    with pytest.raises(LayoutBoundaryError, match="NOT resident"):
        LayoutBoundaryActuator(
            model,
            {"narrow": (0, 4), "wide": (0, 6)},
            "narrow",
            mover=lambda a, d, r: 0,
            residency_probe=lambda i: i in {0, 1},  # layers 2,3 paused
        )


def test_moving_mode_still_refuses_a_placeholder_it_could_never_fill():
    model = _Model(real_layers={0, 1, 2, 3})  # 4..7 are PPMissingLayer
    with pytest.raises(LayoutBoundaryError, match="PPMissingLayer placeholders"):
        LayoutBoundaryActuator(
            model,
            {"narrow": (0, 4), "wide": (0, 6)},
            "narrow",
            mover=lambda a, d, r: 0,
            residency_probe=lambda i: True,
        )


def test_a_mover_failure_is_torn_and_never_rolled_back():
    model = _Model(real_layers=set(range(8)))

    def mover(activated, deactivated, rng):
        raise OSError("BAR1 window vanished mid-leg")

    act = LayoutBoundaryActuator(
        model,
        {"narrow": (0, 4), "wide": (0, 6)},
        "narrow",
        mover=mover,
        residency_probe=lambda i: i < 4,
    )
    with pytest.raises(LayoutBoundaryTorn, match="must stop"):
        act.flip("wide", quiescent=True)
    # The range was NOT advanced, and the failure is a stop, not a retry.
    assert model._end_layer == 4


def test_a_mover_that_lies_about_residency_is_caught_after_it_runs():
    """A mover that returns a byte count without making the layers resident
    must not get the range moved anyway."""
    model = _Model(real_layers=set(range(8)))
    act = LayoutBoundaryActuator(
        model,
        {"narrow": (0, 4), "wide": (0, 6)},
        "narrow",
        mover=lambda a, d, r: 999,  # claims success, moves nothing
        residency_probe=lambda i: i < 4,
    )
    with pytest.raises(LayoutBoundaryError, match="NOT resident"):
        act.flip("wide", quiescent=True)
    assert model._end_layer == 4


def test_an_observer_refusal_after_a_move_is_torn_not_rolled_back():
    model = _Model(real_layers=set(range(8)))
    resident = {0, 1, 2, 3}

    def mover(activated, deactivated, rng):
        resident.update(activated)
        resident.difference_update(deactivated)
        return 1

    act = LayoutBoundaryActuator(
        model,
        {"narrow": (0, 4), "wide": (0, 6)},
        "narrow",
        mover=mover,
        residency_probe=lambda i: i in resident,
    )

    def refuse(report):
        raise RuntimeError("the KV pool cannot follow")

    act.add_observer(refuse)
    with pytest.raises(LayoutBoundaryTorn, match="no longer backed by weights"):
        act.flip("wide", quiescent=True)


def test_moving_mode_still_refuses_a_non_quiescent_flip():
    model = _Model(real_layers=set(range(8)))
    act = LayoutBoundaryActuator(
        model,
        {"narrow": (0, 4), "wide": (0, 6)},
        "narrow",
        mover=lambda a, d, r: 0,
        residency_probe=lambda i: True,
    )
    with pytest.raises(LayoutBoundaryError, match="not quiescent"):
        act.flip("wide", quiescent=False)


# ---------------------------------------------------------------------------
# The worked example, with this rig's measured numbers.
# ---------------------------------------------------------------------------


def test_worked_example_on_the_rig_numbers():
    """p39 <-> p32 on the 64-layer INT8 gdncov checkpoint, priced end to end.

    Pins the design's numbers so a later change to any of them shows up as a
    failing assertion rather than as a quietly different recommendation.
    """
    calib = rig_calibration()
    plan = plan_layer_moves(P32, P39, calib.layer_bytes)

    # 11 layers change owner: 7 from stage1 back to stage0, 4 from stage2 to 1.
    assert len(plan.moves) == 11
    assert plan.by_pair() == {
        (1, 0): 7 * BYTES_PER_LAYER,
        (2, 1): 4 * BYTES_PER_LAYER,
    }

    # The x4 link binds: stage 1 sends 2.94 GB at 6.56 GB/s while receiving
    # 1.68 GB at 6.58 -- duplex, so the send is the cost.
    switch_s = move_seconds(plan, calib.rates)
    assert switch_s == pytest.approx(7 * BYTES_PER_LAYER / (6.56 * 1e9), rel=1e-9)
    assert switch_s == pytest.approx(0.448, abs=0.005)

    # Going to p32 frees one whole 8-layer band on the 5090: 3.36 GB.
    freed = vram_freed_bytes(P39, P32, 0, calib.geom, calib.layer_bytes)
    assert freed == 8 * BYTES_PER_LAYER
    assert freed / 1e9 == pytest.approx(3.36, abs=0.01)

    # And the break-even is a token count the scheduler can compare against.
    n = breakeven_tokens(P32, P39, calib)
    assert n is not None and n > 0
    v = decide(current=P32, fast=P39, lean=P32, pending_tokens=n * 4, calib=calib)
    assert v.switches
    assert v.gain_s > 0
