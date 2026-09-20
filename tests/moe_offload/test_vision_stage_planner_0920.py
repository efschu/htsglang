# SPDX-License-Identifier: Apache-2.0
"""Task #58 slice 1 -- the transient vision stage's PLACEMENT term.

HERMETIC: no CUDA, no NVML, no GPU, no torch, no network.  Every number below
is either arithmetic from a config this box carries or a line a boot printed.

FIXTURE PROVENANCE
------------------
* ``TOWER_*`` -- read on 2026-09-20 out of
  ``/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-INT8-gdncov``:
  ``model.safetensors.index.json`` lists **333** ``*visual*`` tensors and the
  safetensors headers of ``model-00001-of-00018.safetensors`` put them all in
  that one shard, contiguous, from byte offset 4_889_688 to 926_349_880 --
  **921_460_192 bytes = 0.858 GiB**, every tensor BF16.  That is the xsn63
  manifest figure to the byte.
* ``READ_GBPS_ODIRECT`` / ``READ_GBPS_BUFFERED`` -- measured on this box the
  same day against that extent: O_DIRECT 3.85 GB/s (239 ms for the whole
  tower), buffered per-piece ``pread`` 1.08-1.15 GB/s (800-850 ms).
* ``FN8AJ_*`` -- boot fn8aj's own ``[vram-peak]`` lines, 2026-09-20,
  ``/spinning/evidence-665-f1/boot_fn_fn8aj_20260920T094757Z.server.log``,
  quoted through ``tests/moe_offload/test_expert_pool_budget_0920.py``.
* link rates -- memory ``RANG-LINK-ZUORDNUNG``: rank 1 sits on x4 (6.5 GB/s),
  ranks 0 and 2 on x8 (14.4 / 13.3 GB/s).  MEASURED rates, not link widths.

WHAT THIS PINS, and why each is a danger direction and not a ceremony:

1. The tower's bytes are the checkpoint's, and a NON-contiguous manifest is
   REFUSED rather than silently planned at the contiguous read rate.
2. The placement prefers a card that needs NO eviction, even when an eviction
   candidate on a faster card would finish sooner.  Moving zero bytes beats
   moving some, and the order calls the displacement the fallback.
3. The link tie-break is real on this rig: with equal air, the x8 card wins
   over the x4 card -- and it does so WITHOUT the 5090 being named anywhere,
   which is what memory ``vision-tower-platzierung`` (12.09.: the 5090's
   space is the most valuable, the placement must be choosable) asks for.
4. ``prefer_cards`` overrides the tie-break, because that memory says the
   placement is CHOOSABLE.
5. The displacement is the SMALLEST that closes the gap -- "ein kleiner Teil"
   in the user's words -- never the first block that fits.
6. A block inside a captured CUDA graph's private pool is present but NOT
   evictable, and the refusal says so instead of pretending it is absent.
7. Reserves are never booked: the default floor is 0 bytes, and a non-zero
   floor without a NAMED reason is a hard error, not a courtesy.
8. ``encode_seconds`` and therefore ``stage_seconds`` are ``None`` without a
   MEASURED rate.  FLOPs are arithmetic; seconds are a measurement.
9. A flip in flight refuses by its own name and never reads as "no room".
"""

import math

import pytest

from sglang.srt.planner import vision_stage as vs

GIB = vs.GIB
MIB = vs.MIB

# ---------------------------------------------------------------- fixtures --

TOWER_PIECES = 333
TOWER_FIRST_OFFSET = 4_889_688
TOWER_LAST_END = 926_349_880
TOWER_BYTES = 921_460_192
TOWER_SHARD = "model-00001-of-00018.safetensors"

READ_GBPS_ODIRECT = 3.85
READ_GBPS_BUFFERED = 1.08

#: memory RANG-LINK-ZUORDNUNG, measured H2D per card.
H2D_GBPS = {0: 14.4, 1: 6.5, 2: 13.3}

#: fn8aj ``[vram-peak] decode`` -- the quietest state that boot sampled, the
#: closest stand-in it carries for "idle, pools built, no prefill in flight".
#: The real number for a stage placement is ``free_idle_mib``
#: (weg2/corridor_budget.py:143); until a boot prints it for the P group,
#: these are what exist, and the tests say so rather than inventing better.
FN8AJ_FREE_IDLE = {0: 0.65 * GIB, 1: 1.63 * GIB, 2: 2.09 * GIB}
FN8AJ_FREE_LOAD = {0: 0.55 * GIB, 1: 0.85 * GIB, 2: 0.96 * GIB}
FN8AJ_TOTAL = {0: 31.34 * GIB, 1: 19.58 * GIB, 2: 19.58 * GIB}

VISION_CFG = vs.VisionEncoderConfig()  # the gdncov checkpoint's values


def tower(**kw) -> vs.TowerSpec:
    return vs.tower_from_span(
        TOWER_PIECES,
        TOWER_FIRST_OFFSET,
        TOWER_LAST_END,
        TOWER_BYTES,
        shard=TOWER_SHARD,
        **kw,
    )


def card(idx, free_gib, *, evictable=(), total_gib=None) -> vs.CardAir:
    return vs.CardAir(
        card=idx,
        ranks=(idx,),
        total_bytes=int((total_gib or FN8AJ_TOTAL[idx] / GIB) * GIB),
        free_bytes=int(free_gib * GIB),
        h2d_gbps=H2D_GBPS[idx],
        evictable=tuple(evictable),
        free_under_load_bytes=int(FN8AJ_FREE_LOAD[idx]),
        provenance="fn8aj [vram-peak] decode",
    )


def band(name, gib, card_idx) -> vs.EvictableBlock:
    rate = H2D_GBPS[card_idx]
    return vs.EvictableBlock(
        name=name, bytes=int(gib * GIB), out_gbps=rate, in_gbps=rate
    )


# ------------------------------------------------------- the tower's bytes --


def test_the_tower_is_one_contiguous_extent_and_that_is_checked():
    t = tower()
    assert t.pieces == TOWER_PIECES
    assert t.weight_bytes == TOWER_BYTES
    assert t.weight_bytes / GIB == pytest.approx(0.858, abs=0.001)
    assert TOWER_SHARD in t.source
    # With no extra posts the tower's footprint IS its weights.
    assert t.total_bytes == TOWER_BYTES


def test_a_scattered_manifest_is_refused_not_planned_as_contiguous():
    """The danger: a checkpoint whose tower is interleaved with other tensors
    would be planned at 3.85 GB/s and deliver 333 seeks."""
    with pytest.raises(vs.VisionStageTowerUnreadable) as e:
        vs.tower_from_span(
            TOWER_PIECES,
            TOWER_FIRST_OFFSET,
            TOWER_LAST_END + 4096,  # a gap: some other tensor sits inside
            TOWER_BYTES,
            shard=TOWER_SHARD,
        )
    assert "NOT one contiguous extent" in str(e.value)
    assert "4096" in str(e.value)


@pytest.mark.parametrize("pieces,nbytes", [(0, TOWER_BYTES), (TOWER_PIECES, 0)])
def test_an_unpriced_tower_refuses_before_any_placement(pieces, nbytes):
    with pytest.raises(vs.VisionStageTowerUnreadable):
        vs.TowerSpec(pieces=pieces, weight_bytes=nbytes)


def test_posts_are_explicit_and_sum_to_the_footprint():
    """User law: transients are BOOKED, never left to slack."""
    t = tower(ctx_bytes=int(0.45 * GIB), activation_bytes=int(0.30 * GIB),
              embedding_bytes=int(0.01 * GIB))
    assert t.total_bytes == sum(v for _, v in t.posts)
    assert [n for n, _ in t.posts] == [
        "tower weights",
        "cuda context",
        "encoder activation",
        "embeddings",
    ]


# ---------------------------------------------------------- encoder shapes --


def test_geometry_of_the_gdncov_tower_matches_its_config():
    cfg = VISION_CFG
    # a 1024x1024 still: 64x64 patches, merged 2x2 -> 1024 prefill rows
    assert cfg.patch_rows(1024, 1024) == 4096
    assert cfg.vision_tokens(1024, 1024) == 1024
    # deepstack is EMPTY on this checkpoint, so one embedding row is exactly
    # the LLM's hidden width -- nothing to split on the receiving side.
    assert cfg.deepstack_visual_indexes == ()
    assert cfg.embed_width == 5120


def test_a_deepstack_checkpoint_would_widen_the_row_and_this_says_so():
    """Not our checkpoint -- pinned so the constraint is visible if one ever
    is.  ``encode_server.py:441-450`` ships ``out_hidden * (1 + len(ds))``."""
    cfg = vs.VisionEncoderConfig(deepstack_visual_indexes=(8, 16, 24))
    assert cfg.embed_width == 4 * 5120


def test_encoder_flops_are_arithmetic_and_grow_quadratically_in_attention():
    cfg = VISION_CFG
    rows = cfg.patch_rows(1024, 1024)
    full = cfg.encoder_flops(rows)
    linear_only = cfg.encoder_flops(rows, full_attention=False)
    assert full > linear_only > 0
    # the linear part is exactly 27 blocks x rows x 2*(h*3h + h*h + h*i + i*h)
    h, i, d = cfg.hidden_size, cfg.intermediate_size, cfg.depth
    per_row_block = 2 * (h * 3 * h + h * h + h * i + i * h)
    merger = 2 * (rows // 4) * (h * 4) * cfg.out_hidden_size
    assert linear_only == rows * d * per_row_block + merger
    # doubling the rows more than doubles the full-attention cost
    assert cfg.encoder_flops(2 * rows) > 2 * full


def test_flops_refuse_an_empty_image():
    with pytest.raises(ValueError):
        VISION_CFG.encoder_flops(0)


# -------------------------------------------------------------- placement --


def test_it_picks_the_card_with_air_and_needs_no_eviction():
    """fn8aj's quietest sample: only card 2 (2.09 GiB) and card 1 (1.63 GiB)
    clear the 0.858 GiB tower plus a 0.45 GiB context; card 0 (0.65) does not.

    MEASURED CORRECTION, and this test is where it is pinned: between cards 1
    and 2 the modelled times are EQUAL, not different.  The pipelined load is
    ``max(read, h2d)``, the O_DIRECT read of the tower runs at 3.85 GB/s on
    this box, and every card's H2D is faster than that (6.5 GB/s on the x4,
    13.3 on the x8) -- so the read leg dominates on both and the seconds tie.
    The first draft of this test asserted the link would decide on TIME and
    was wrong; the link decides as an explicit tie-break instead.  The 5090
    is still named nowhere in the term.
    """
    t = tower(ctx_bytes=int(0.45 * GIB))
    cards = [card(i, FN8AJ_FREE_IDLE[i] / GIB) for i in (0, 1, 2)]
    plan = vs.plan_vision_stage(cards, t, read_gbps=READ_GBPS_ODIRECT)
    # the two legs really do tie -- that is the premise, so assert it
    assert plan.read_seconds > plan.h2d_seconds
    assert plan.card == 2
    assert plan.evicted == ()
    assert plan.slack_bytes == pytest.approx(
        FN8AJ_FREE_IDLE[2] - t.total_bytes, abs=1.0
    )
    reasons = dict(plan.rejected)
    assert 1 in reasons and "slower link" in reasons[1]
    assert 2 not in reasons
    assert 0 in reasons and "short by" in reasons[0]


def test_no_eviction_beats_eviction_even_when_eviction_would_be_faster():
    """The 5090 is on the fastest link AND could displace a band; card 2 has
    the air and sits on a slower link.  Card 2 must still win: the order
    makes the displacement the FALLBACK, not a cheaper route."""
    t = tower(ctx_bytes=int(0.45 * GIB))
    cards = [
        card(0, 0.20, evictable=[band("weights_3", 2.0, 0)]),
        card(2, FN8AJ_FREE_IDLE[2] / GIB),
    ]
    plan = vs.plan_vision_stage(cards, t, read_gbps=READ_GBPS_ODIRECT)
    assert plan.card == 2
    assert plan.evicted == ()
    assert any("displace" in why for _, why in plan.rejected)


def test_prefer_cards_overrides_the_tie_break():
    """memory vision-tower-platzierung, user 12.09.: the placement must be
    CHOOSABLE.  An override is an argument here, never a constant."""
    t = tower(ctx_bytes=int(0.45 * GIB))
    cards = [card(i, FN8AJ_FREE_IDLE[i] / GIB) for i in (0, 1, 2)]
    plan = vs.plan_vision_stage(
        cards, t, read_gbps=READ_GBPS_ODIRECT, prefer_cards=(1,)
    )
    assert plan.card == 1
    assert plan.evicted == ()


def test_prefer_cards_does_not_override_physics():
    """A preferred card that cannot hold the stage is still not chosen."""
    t = tower(ctx_bytes=int(0.45 * GIB))
    cards = [card(i, FN8AJ_FREE_IDLE[i] / GIB) for i in (0, 1, 2)]
    plan = vs.plan_vision_stage(
        cards, t, read_gbps=READ_GBPS_ODIRECT, prefer_cards=(0,)
    )
    assert plan.card != 0
    assert 0 in dict(plan.rejected)


# -------------------------------------------------------------- eviction --


def test_the_displacement_is_the_smallest_block_that_closes_the_gap():
    """'entweder ist dort irgendwo noch vram frei, oder wir offloaden solange
    einen KLEINEN TEIL' -- smallest sufficient, not first-fit."""
    t = tower()  # 0.858 GiB, no context: the stage runs in-process
    blocks = [
        band("weights_0", 4.0, 0),
        band("weights_1", 1.0, 0),
        band("weights_2", 2.5, 0),
    ]
    plan = vs.plan_vision_stage(
        [card(0, 0.20, evictable=blocks)], t, read_gbps=READ_GBPS_ODIRECT
    )
    assert [b.name for b in plan.evicted] == ["weights_1"]
    assert plan.slack_bytes >= 0


def test_several_blocks_are_combined_only_when_no_single_one_suffices():
    t = tower()
    blocks = [band("weights_0", 0.4, 2), band("weights_1", 0.5, 2),
              band("weights_2", 0.3, 2)]
    plan = vs.plan_vision_stage(
        [card(2, 0.10, evictable=blocks)], t, read_gbps=READ_GBPS_ODIRECT
    )
    freed = sum(b.bytes for b in plan.evicted)
    assert freed >= t.total_bytes - int(0.10 * GIB)
    # and no block is carried that the remainder does not need
    for b in plan.evicted:
        assert freed - b.bytes < t.total_bytes - int(0.10 * GIB)


def test_a_block_in_a_captured_graph_pool_is_present_but_not_evictable():
    """The danger: counting a graph-private block as free capacity, planning
    on it, and discovering at the flip that the graph is now invalid."""
    t = tower()
    blocked = vs.EvictableBlock(
        name="weights_0", bytes=int(4.0 * GIB), out_gbps=13.3, in_gbps=13.3,
        graph_safe=False,
    )
    c = card(2, 0.10, evictable=[blocked])
    assert c.evictable_bytes == 0
    with pytest.raises(vs.VisionStageNoRoom) as e:
        vs.plan_vision_stage([c], t, read_gbps=READ_GBPS_ODIRECT)
    assert "evictable 0.000 GiB" in str(e.value)


def test_eviction_can_be_forbidden_and_the_refusal_says_who_forbade_it():
    t = tower()
    c = card(2, 0.10, evictable=[band("weights_1", 4.0, 2)])
    with pytest.raises(vs.VisionStageNoRoom) as e:
        vs.plan_vision_stage(
            [c], t, read_gbps=READ_GBPS_ODIRECT, allow_eviction=False
        )
    assert "FORBIDDEN by the caller" in str(e.value)


def test_evict_and_restore_are_priced_separately_per_direction():
    b = vs.EvictableBlock(
        name="weights_1", bytes=int(1.0 * GIB), out_gbps=6.5, in_gbps=13.3
    )
    assert b.evict_seconds > b.restore_seconds
    assert b.round_trip_seconds == pytest.approx(
        b.evict_seconds + b.restore_seconds
    )


def test_a_placeholder_link_rate_is_refused():
    with pytest.raises(ValueError):
        vs.EvictableBlock(name="x", bytes=1, out_gbps=0.0, in_gbps=1.0)


# --------------------------------------------------------------- refusals --


def test_no_room_anywhere_refuses_by_name_with_every_cards_arithmetic():
    t = tower(ctx_bytes=int(0.45 * GIB))
    cards = [card(i, 0.10) for i in (0, 1, 2)]
    with pytest.raises(vs.VisionStageNoRoom) as e:
        vs.plan_vision_stage(cards, t, read_gbps=READ_GBPS_ODIRECT)
    msg = str(e.value)
    for i in (0, 1, 2):
        assert f"card{i}:" in msg
    assert "short" in msg
    assert "no reserve is raided" in msg
    exc = e.value
    assert len(exc.attempts) == 3
    assert all(short > 0 for _, _, _, short in exc.attempts)


def test_an_empty_card_list_refuses_by_the_same_name():
    with pytest.raises(vs.VisionStageNoRoom):
        vs.plan_vision_stage([], tower(), read_gbps=READ_GBPS_ODIRECT)


def test_a_flip_in_flight_refuses_by_its_OWN_name_not_as_no_room():
    """Distinct classes on purpose: a flip is a WAIT, capacity is a NO.  An
    operator reading 'no room' for what is a 2 s wait chases the wrong thing.
    The predicate is ``PhaseFlipRuntime.is_armed()``
    (phase_flip_runtime.py:8085), read fail-closed the way
    kv_backing_relief.py:4567 reads it."""
    cards = [card(2, 8.0)]  # plenty of air: capacity is NOT the reason
    with pytest.raises(vs.VisionStageFlipInFlight) as e:
        vs.plan_vision_stage(
            cards, tower(), read_gbps=READ_GBPS_ODIRECT,
            flip_in_flight=True, flip_direction="D->P",
        )
    assert "D->P" in str(e.value)
    assert "not a capacity refusal" in str(e.value)
    assert not isinstance(e.value, vs.VisionStageNoRoom)
    assert isinstance(e.value, vs.VisionStageRefused)


# ------------------------------------------------------------- user laws --


def test_the_default_floor_is_zero_bytes():
    """User 2026-09-19, verbatim: 'Reserven NIE, nicht ein Byte'."""
    assert vs.DEFAULT_FLOOR_BYTES == 0
    t = tower()
    c = card(2, t.total_bytes / GIB)  # exactly enough, not one byte more
    plan = vs.plan_vision_stage([c], t, read_gbps=READ_GBPS_ODIRECT)
    assert plan.slack_bytes == pytest.approx(0.0, abs=1.0)
    assert plan.floor_bytes == 0


def test_a_floor_without_a_named_reason_is_a_hard_error():
    with pytest.raises(ValueError) as e:
        vs.plan_vision_stage(
            [card(2, 8.0)], tower(), read_gbps=READ_GBPS_ODIRECT,
            floor_bytes=int(0.5 * GIB),
        )
    assert "named reason" in str(e.value)


def test_a_named_floor_is_carried_into_the_need_and_printed():
    t = tower()
    plan = vs.plan_vision_stage(
        [card(2, 8.0)], t, read_gbps=READ_GBPS_ODIRECT,
        floor_bytes=int(0.5 * GIB), floor_reason="fragmentation, fn8ak3",
    )
    assert plan.need_bytes == t.total_bytes + int(0.5 * GIB)
    assert "fragmentation, fn8ak3" in plan.report()


def test_encode_seconds_are_None_without_a_measured_rate():
    """Memory NULL-NUR-BEI-ERREICHTEM-EMITTER: FLOPs are arithmetic, seconds
    are a measurement.  And a total that is missing a leg is not printed as a
    total."""
    rows = VISION_CFG.patch_rows(1024, 1024)
    plan = vs.plan_vision_stage(
        [card(2, 8.0)], tower(), read_gbps=READ_GBPS_ODIRECT,
        encoder_flops=VISION_CFG.encoder_flops(rows),
    )
    assert plan.encode_flops > 0
    assert plan.encode_seconds is None
    assert plan.stage_seconds is None
    assert "rate not measured" in plan.report()


def test_a_measured_rate_turns_the_flops_into_the_stage_total():
    rows = VISION_CFG.patch_rows(1024, 1024)
    flops = VISION_CFG.encoder_flops(rows)
    plan = vs.plan_vision_stage(
        [card(2, 8.0)], tower(), read_gbps=READ_GBPS_ODIRECT,
        encoder_flops=flops, achieved_tflops=20.0,
    )
    assert plan.encode_seconds == pytest.approx(flops / 20e12)
    assert plan.stage_seconds == pytest.approx(
        plan.evict_seconds + plan.load_seconds
        + plan.encode_seconds + plan.restore_seconds
    )


# --------------------------------------------------------------- the cost --


def test_the_measured_read_rate_is_the_load_floor_on_this_box():
    """O_DIRECT 3.85 GB/s against buffered 1.08 GB/s, both measured here on
    the very extent this loads.  The difference is 0.61 s of TTFT."""
    t = tower()
    fast = vs.plan_vision_stage(
        [card(2, 8.0)], t, read_gbps=READ_GBPS_ODIRECT
    )
    slow = vs.plan_vision_stage(
        [card(2, 8.0)], t, read_gbps=READ_GBPS_BUFFERED
    )
    assert fast.read_seconds == pytest.approx(TOWER_BYTES / 3.85e9)
    assert fast.read_seconds < 0.25
    assert slow.read_seconds > 0.80
    assert slow.load_seconds - fast.load_seconds > 0.55


def test_the_pipelined_load_is_the_slower_of_the_two_legs():
    t = tower()
    c = card(1, 8.0)  # x4: 6.5 GB/s H2D is slower than the 3.85 GB/s read
    piped = vs.plan_vision_stage([c], t, read_gbps=READ_GBPS_ODIRECT)
    serial = vs.plan_vision_stage(
        [c], t, read_gbps=READ_GBPS_ODIRECT, pipelined_load=False
    )
    assert piped.load_seconds == pytest.approx(
        max(piped.read_seconds, piped.h2d_seconds)
    )
    assert serial.load_seconds == pytest.approx(
        serial.read_seconds + serial.h2d_seconds
    )
    assert serial.load_seconds > piped.load_seconds


def test_the_report_names_every_post_and_every_rejected_card():
    t = tower(ctx_bytes=int(0.45 * GIB), activation_bytes=int(0.30 * GIB))
    plan = vs.plan_vision_stage(
        [card(i, FN8AJ_FREE_IDLE[i] / GIB) for i in (0, 1, 2)],
        t, read_gbps=READ_GBPS_ODIRECT,
    )
    text = plan.report()
    for post in ("tower weights", "cuda context", "encoder activation"):
        assert post in text
    assert "card0 rejected" in text
    assert "card free WITH a prefill in flight" in text


# ---------------------------------------------------------------- shapes --


def test_a_self_inconsistent_card_census_is_rejected():
    with pytest.raises(ValueError):
        vs.CardAir(card=0, ranks=(0,), total_bytes=int(GIB),
                   free_bytes=int(2 * GIB), h2d_gbps=14.4)


def test_a_nominal_link_width_of_zero_is_rejected():
    with pytest.raises(ValueError):
        vs.CardAir(card=0, ranks=(0,), total_bytes=int(GIB),
                   free_bytes=0, h2d_gbps=0.0)


def test_read_rate_must_be_measured_not_zero():
    with pytest.raises(ValueError):
        vs.plan_vision_stage([card(2, 8.0)], tower(), read_gbps=0.0)
