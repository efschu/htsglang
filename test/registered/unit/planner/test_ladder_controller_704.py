"""#704: the ladder controller must not oscillate, on any fill trace.

The controller turns a solved ladder plus a live-fill reading into rung
changes. Its failure mode is not a wrong answer, it is a CHATTERING answer:
a fill level that sits near a boundary and drives a rung change every step,
each of which moves 451-901 MiB of weights over PCIe. On this rig that would
spend the entire link budget on layout churn.

So the properties below are falsifiers, not smoke tests. They drive adversarial
fill traces -- monotone ramps, sawtooths parked exactly on a threshold, and
step changes -- and assert the controller cannot be made to flap.

Hermetic: pure arithmetic, no CUDA, no server.
"""

import pytest
from sglang.srt.planner.ladder_controller import LadderController
from sglang.srt.planner.layout_ladder import LadderInputs, solve_layout_ladder
from sglang.srt.planner.pp_cut import (
    PhasePoolModel,
    kv_mib_per_token_per_attn_layer_from_config,
    layer_families_from_config,
)

RIG_FAMILIES = layer_families_from_config(
    {"num_hidden_layers": 64, "full_attention_interval": 4}
)

# Structural fixture only; see test_layout_ladder_704 for why no pool VALUE
# here is bootable. The controller tests assert control behaviour, which is
# invariant to the free-bytes vector.
RIG_KV_CELL = kv_mib_per_token_per_attn_layer_from_config(
    {"num_key_value_heads": 4, "head_dim": 256}, "fp8_e4m3"
)


def _rig_pool() -> PhasePoolModel:
    return PhasePoolModel(
        free_mib=(26721.2, 16051.3, 13984.3),
        weight_mib_per_layer=450.7,
        kv_mib_per_token_per_attn_layer=RIG_KV_CELL,
        arming_floor_mib=(1728.0, 1825.0, 2467.0),
        mamba_mib_per_linear_layer_per_slot=50.85,
        mamba_slots=1,
    )


def _flat_floor(counts):
    return (1728.0, 1825.0, 2467.0)


def _rig_ladder():
    return solve_layout_ladder(
        LadderInputs(
            pool=_rig_pool(),
            layer_families=RIG_FAMILIES,
            arming_floor_for=_flat_floor,
            ms_per_layer=(1.7571, 7.740, 7.275),
            fixed_ms=(0.0, 0.0, 0.0),
            link_mib_per_s=(3000.0, 6000.0, 6000.0),
            min_pool_tokens=200_000.0,
            prefill_tokens_per_s=20_000.0,
        )
    )


def _drive(controller, trace, quiescent=True):
    """Run a fill trace and return the list of committed rung changes."""
    moves = []
    for fill in trace:
        target = controller.observe(float(fill), quiescent=quiescent)
        if target is not None:
            moves.append((fill, target))
    return moves


def test_a_flat_trace_settles_and_then_stops():
    """Convergence, and the distinction the controller lives or dies on.

    A flat fill is NOT a no-op from an arbitrary starting rung: at 250k live
    tokens on the roomiest rung the ladder should descend to harvest the slack,
    which is its entire purpose. What must not happen is that it keeps moving.
    So: bounded transient, then silence.
    """
    ladder = _rig_ladder()
    c = LadderController(ladder)
    transient = _drive(c, [250_000.0] * 50)
    assert len(transient) <= len(ladder.rungs), "the transient did not terminate"
    # Settled. From here nothing more may move.
    assert _drive(c, [250_000.0] * 50) == []


def test_a_sawtooth_parked_on_a_threshold_does_not_flap():
    """THE anti-oscillation proof.

    Let the controller settle, then park the fill exactly on the live
    transition's descend threshold and jitter it by one token in each
    direction, forever. A bare-threshold controller emits a move on every
    sample. Hysteresis means a value low enough to trigger a descent is by
    construction too low to trigger the answering ascent, so the jitter must
    die out.
    """
    ladder = _rig_ladder()
    seed = ladder.transitions[len(ladder.transitions) // 2].descend_below_tokens
    c = LadderController(ladder)
    _drive(c, [seed] * 50)  # settle first; the transient is not flapping

    # Jitter around the boundary that is ACTIVE at the settled rung. Probing an
    # inactive threshold would pass vacuously -- verified: it does, and a
    # controller with its hysteresis collapsed survives that version unchanged.
    edge = c.active_descend_threshold()
    assert edge is not None
    trace = []
    for _ in range(200):
        trace += [edge - 1.0, edge + 1.0]

    # First pass may still settle: the descend thresholds are not monotone in
    # rung index, so crossing one boundary can unlock the next. That is
    # convergence, and it must be bounded by the ladder's own depth.
    first = _drive(c, trace)
    assert len(first) <= len(ladder.rungs)

    # Second pass is the actual proof: a fixed point, under identical jitter.
    # With the hysteresis collapsed to a bare threshold this pass emits one
    # move per sample (400) instead of none.
    assert _drive(c, trace) == [], "controller has no fixed point under jitter"


def test_a_monotone_ramp_only_ever_ascends():
    """Rising fill must walk one direction and never backtrack."""
    ladder = _rig_ladder()
    c = LadderController(ladder)
    # Start deep (low fill), then fill the pool steadily.
    c.force_rung(len(ladder.rungs) - 1)
    trace = [float(x) for x in range(0, 430_000, 2_000)]
    moves = _drive(c, trace)
    idxs = [ladder.rungs.index(m[1]) for m in moves]
    assert idxs == sorted(idxs, reverse=True), "a rising ramp backtracked"


def test_a_draining_trace_only_ever_descends():
    ladder = _rig_ladder()
    c = LadderController(ladder)
    c.force_rung(0)
    trace = [float(x) for x in range(430_000, 0, -2_000)]
    moves = _drive(c, trace)
    idxs = [ladder.rungs.index(m[1]) for m in moves]
    assert idxs == sorted(idxs), "a draining trace backtracked"


def test_the_controller_never_selects_a_rung_that_cannot_hold_the_live_set():
    """The safety invariant: no committed rung may be under the live fill."""
    ladder = _rig_ladder()
    c = LadderController(ladder)
    for fill in list(range(0, 430_000, 3_000)) + list(range(430_000, 0, -3_000)):
        target = c.observe(float(fill), quiescent=True)
        if target is not None:
            assert target.pool_tokens >= fill


def test_no_move_is_committed_while_not_quiescent():
    """Rung changes commit at a chunk boundary, never inside one."""
    ladder = _rig_ladder()
    c = LadderController(ladder)
    c.force_rung(0)
    busy = _drive(c, [float(x) for x in range(430_000, 0, -5_000)], quiescent=False)
    assert busy == []
    # And the pending decision is not lost -- it commits at the next boundary.
    assert c.observe(50_000.0, quiescent=True) is not None


def test_descend_only_happens_at_low_fill():
    """Descending is bounded to the regime where KV movement is cheap.

    Stated RELATIVE to the ladder's own solved threshold. An earlier version
    used absolute token counts, which silently encoded the KV dtype: when the
    cell was corrected from bf16 to the shipped fp8_e4m3 every pool doubled and
    those constants became meaningless. A control test must not carry a
    hardware constant.
    """
    ladder = _rig_ladder()
    c = LadderController(ladder)
    c.force_rung(0)
    edge = c.active_descend_threshold()
    assert edge is not None
    for above in (1.30, 1.15, 1.001):
        c.force_rung(0)
        assert c.observe(edge * above, quiescent=True) is None
    # And it DOES descend just below the same boundary -- otherwise the test
    # above would pass on a controller that never descends at all.
    c.force_rung(0)
    assert c.observe(edge * 0.999, quiescent=True) is not None


def test_a_single_ladder_step_per_observation():
    """No multi-rung leaps: each commit moves at most one rung.

    A leap would move several layers at once and blow the move budget the
    hysteresis was derived against.
    """
    ladder = _rig_ladder()
    c = LadderController(ladder)
    c.force_rung(0)
    prev = 0
    for fill in [float(x) for x in range(430_000, 0, -1_000)]:
        target = c.observe(fill, quiescent=True)
        if target is not None:
            idx = ladder.rungs.index(target)
            assert abs(idx - prev) == 1
            prev = idx


def test_an_empty_transition_set_pins_the_controller():
    """If no step could be funded, the controller must sit still.

    `solve_layout_ladder` prunes steps whose bands cross. A controller handed a
    ladder with no usable transitions must not invent one.
    """
    ladder = solve_layout_ladder(
        LadderInputs(
            pool=_rig_pool(),
            layer_families=RIG_FAMILIES,
            arming_floor_for=_flat_floor,
            ms_per_layer=(1.7571, 7.740, 7.275),
            fixed_ms=(0.0, 0.0, 0.0),
            link_mib_per_s=(0.5, 0.5, 0.5),
            min_pool_tokens=200_000.0,
            prefill_tokens_per_s=200_000.0,
        )
    )
    assert ladder.transitions == ()
    c = LadderController(ladder)
    assert _drive(c, [float(x) for x in range(0, 400_000, 5_000)]) == []


def test_force_rung_rejects_an_index_off_the_ladder():
    ladder = _rig_ladder()
    c = LadderController(ladder)
    with pytest.raises(IndexError):
        c.force_rung(len(ladder.rungs))
