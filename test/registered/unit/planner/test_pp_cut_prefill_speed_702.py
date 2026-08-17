"""#702: solve the PP cut for PREFILL SPEED, not for pool capacity.

The capacity solve (#602) correctly withdrew the cut: at the censused regime
the incumbent [28,20,16] is the pool optimum. But the question actually asked
was "more prefill on the 5090", which is a different objective and was never
solved for.

Two objectives, deliberately reported side by side because they disagree:

* SERIAL   sum_r layers_r * ms_per_layer_r  -- what a non-pipelined prefill
  costs today.
* PIPELINED max_r layers_r * ms_per_layer_r -- once the stages overlap, the
  bottleneck is the slowest stage, not the sum.

Calibration is from the measured stage times at the incumbent:
49.2 / 154.8 / 116.4 ms at [28, 20, 16], i.e. 1.757 / 7.740 / 7.275 ms per
layer.

The modelling limitation these tests pin: ONE measured cut gives ONE
(count, time) point per rank, which cannot separate a per-layer slope from a
fixed per-stage cost. `fixed_ms` defaults to zero, which is the OPTIMISTIC
choice, and `test_fixed_overhead_erodes_the_gain` exists so that assumption is
visible rather than buried.

Hermetic: pure arithmetic, no CUDA, no pool construction.
"""

import pytest

from sglang.srt.planner.pp_cut import (
    pipelined_prefill_ms,
    prefill_timing_from_measurement,
    serial_prefill_ms,
    solve_pp_cut_for_prefill_speed,
)

INCUMBENT = (28, 20, 16)
MEASURED_MS = (49.2, 154.8, 116.4)


def _timing():
    return prefill_timing_from_measurement(INCUMBENT, MEASURED_MS)


def test_calibration_backtests_the_measured_stage_times():
    """The model must reproduce the measurement it was calibrated on."""
    t = _timing()
    assert t.ms_per_layer[0] == pytest.approx(49.2 / 28, rel=1e-12)
    assert t.ms_per_layer[1] == pytest.approx(7.74, rel=1e-12)
    assert t.ms_per_layer[2] == pytest.approx(7.275, rel=1e-12)
    assert t.stage_ms(INCUMBENT) == pytest.approx(MEASURED_MS, rel=1e-12)
    assert serial_prefill_ms(INCUMBENT, t) == pytest.approx(320.4, rel=1e-12)
    assert pipelined_prefill_ms(INCUMBENT, t) == pytest.approx(154.8, rel=1e-12)


def test_the_hand_arithmetic_anchor_at_42_12_10():
    """Coordinator's anchor: [42,12,10] lands near 1.34x serial.

    By hand: 42*1.7571 + 12*7.74 + 10*7.275 = 73.80 + 92.88 + 72.75 = 239.43,
    against 320.4 -> 1.338x. If the solver disagrees materially, the solver is
    wrong, because this arithmetic is checkable without it.
    """
    t = _timing()
    counts = (42, 12, 10)
    assert serial_prefill_ms(counts, t) == pytest.approx(239.43, abs=0.01)
    assert 320.4 / serial_prefill_ms(counts, t) == pytest.approx(1.338, abs=0.005)
    # Pipelined, the same cut is worth far more: the max stage drops from
    # PP1's 154.8 to 92.88.
    assert pipelined_prefill_ms(counts, t) == pytest.approx(92.88, abs=0.01)
    assert 154.8 / pipelined_prefill_ms(counts, t) == pytest.approx(1.667, abs=0.005)


def test_serial_and_pipelined_objectives_disagree():
    """They rank cuts differently -- which is why both are reported."""
    t = _timing()
    # Balancing the SUM favours pushing layers onto the fast rank; balancing
    # the MAX favours equalising stages. [44,10,10] beats [42,12,10] on both,
    # but by different margins, and it is memory-infeasible (see the cap test).
    s_a = serial_prefill_ms((42, 12, 10), t)
    s_b = serial_prefill_ms((38, 16, 10), t)
    p_a = pipelined_prefill_ms((42, 12, 10), t)
    p_b = pipelined_prefill_ms((38, 16, 10), t)
    assert s_a < s_b  # serial prefers more on rank0
    assert p_a < p_b  # so does pipelined here
    # But the ratio of improvement differs, so the objectives are not
    # monotone transforms of one another.
    assert (s_b / s_a) != pytest.approx(p_b / p_a, rel=1e-3)


def test_rank0_layer_cap_is_enforced():
    """44 layers on rank0 is ~31,868 MiB against a 31,800 MiB budget.

    The cap is a hard constraint, not a preference: a solve that returns a
    45-layer rank0 is proposing arithmetic that ignores the budget.
    """
    t = _timing()
    cands = solve_pp_cut_for_prefill_speed(
        total_layers=64, timing=t, incumbent=INCUMBENT, max_rank0_layers=42
    )
    assert cands, "solver returned nothing"
    assert all(c.counts[0] <= 42 for c in cands)
    # The unconstrained time optimum sits above the cap, so the cap binds.
    loose = solve_pp_cut_for_prefill_speed(
        total_layers=64, timing=t, incumbent=INCUMBENT, max_rank0_layers=64
    )
    best_loose = min(loose, key=lambda c: c.serial_ms)
    assert best_loose.counts[0] > 42


def test_every_candidate_carries_both_speedups_and_a_pool_cost():
    """The user decides the trade on numbers, so all three must be present."""
    t = _timing()
    pool_calls = []

    def pool_fn(counts):
        pool_calls.append(tuple(counts))
        return 400_000.0 - 1000.0 * counts[0]

    cands = solve_pp_cut_for_prefill_speed(
        total_layers=64,
        timing=t,
        incumbent=INCUMBENT,
        max_rank0_layers=42,
        pool_fn=pool_fn,
    )
    assert pool_calls, "pool_fn was never consulted"
    for c in cands:
        assert c.serial_speedup > 0
        assert c.pipelined_speedup > 0
        assert c.pool_tokens is not None
    inc = [c for c in cands if c.counts == INCUMBENT]
    assert inc and inc[0].serial_speedup == pytest.approx(1.0, rel=1e-12)
    assert inc[0].pipelined_speedup == pytest.approx(1.0, rel=1e-12)


def test_pool_cost_is_none_when_no_pool_model_is_supplied():
    """Refuse to invent a pool number rather than defaulting one."""
    t = _timing()
    cands = solve_pp_cut_for_prefill_speed(
        total_layers=64, timing=t, incumbent=INCUMBENT, max_rank0_layers=42
    )
    assert all(c.pool_tokens is None for c in cands)


def test_fixed_overhead_erodes_the_gain():
    """Pins the calibration limitation, and that it is not free.

    One measured cut cannot separate slope from intercept. If a third of each
    stage's time is fixed rather than per-layer, the SAME reallocation buys
    materially less -- so the zero-intercept default is optimistic and must be
    reported as an assumption, not a result.
    """
    zero = _timing()
    fixed = prefill_timing_from_measurement(
        INCUMBENT, MEASURED_MS, fixed_fraction=1.0 / 3.0
    )
    # Backtest still holds: any split of slope and intercept must reproduce
    # the point it was fitted on.
    assert fixed.stage_ms(INCUMBENT) == pytest.approx(MEASURED_MS, rel=1e-9)
    gain_zero = 320.4 / serial_prefill_ms((42, 12, 10), zero)
    gain_fixed = 320.4 / serial_prefill_ms((42, 12, 10), fixed)
    assert gain_fixed < gain_zero
    assert gain_zero == pytest.approx(1.338, abs=0.005)


def test_counts_must_sum_to_total_layers():
    t = _timing()
    with pytest.raises(ValueError, match="sum"):
        serial_prefill_ms((42, 12, 11), t, total_layers=64)


def test_timing_rejects_a_zero_layer_calibration_stage():
    """Dividing a stage time by zero layers is not a per-layer cost."""
    with pytest.raises(ValueError, match="zero"):
        prefill_timing_from_measurement((28, 0, 36), (49.2, 154.8, 116.4))


# ---------------------------------------------------------------------------
# #702 revision: CO-SOLVE the KV/mamba token vector with the layer cut.
#
# The "cut costs pool" column in revision 1 was single-family optimization: it
# priced a layer move while holding the KV token vector PINNED, which the #485
# phase-matrix doctrine forbids. Layers moved to rank0 free exactly their
# weight bytes on ranks 1/2, and the uneven-DCP / rank-kv-ratio machinery
# relocates the displaced KV share onto those freed bytes (the #320
# capacity-matched pattern: prefill 10,1,1 with kv 2,11,10).
# ---------------------------------------------------------------------------

from sglang.srt.planner.pp_cut import (  # noqa: E402
    WorldMemory,
    cosolve_prefill_cut,
)

# Three ranks: 5090 then two 3080s, MiB.
WORLD = dict(
    vram_mib=(31800.0, 19456.0, 19456.0),
    nonlayer_weight_mib=(1200.0, 400.0, 400.0),
    weight_mib_per_layer=724.3,
    kv_mib_per_token_per_layer=0.0021,
)


def _world(**kw):
    d = dict(WORLD)
    d.update(kw)
    return WorldMemory(**d)


def test_world_pool_is_conserved_when_overheads_do_not_move_with_the_cut():
    """The user's claim, proven rather than asserted.

    Total weight bytes are the same 64 layers wherever they sit, and total VRAM
    is fixed, so total free bytes are invariant. Each layer's KV is stored once,
    on whichever rank owns that layer, so a token of world capacity costs the
    same total bytes regardless of the cut. World pool is therefore EXACTLY
    conserved -- not approximately.
    """
    w = _world()
    base = cosolve_prefill_cut((28, 20, 16), w, _timing())
    for counts in [(42, 12, 10), (38, 16, 10), (34, 20, 10), (20, 24, 20)]:
        got = cosolve_prefill_cut(counts, w, _timing())
        assert got.world_pool_tokens == pytest.approx(base.world_pool_tokens, rel=1e-9)
        assert got.world_pool_delta == pytest.approx(0.0, abs=1e-6)


def test_the_residual_is_exactly_the_itemized_second_order_terms():
    """No fudge factor: the delta equals the sum of the named terms."""
    w = _world()

    def overhead(counts):
        # Seam/staging grows with how far the cut moved from the incumbent.
        moved = abs(counts[0] - 28)
        return {
            "seam_staging_fwd": 2.0 * moved,
            "seam_staging_rev": 1.5 * moved,
            "tp_phase_redistribution": 0.5 * moved,
        }

    base = cosolve_prefill_cut((28, 20, 16), w, _timing(), overhead_fn=overhead)
    got = cosolve_prefill_cut((42, 12, 10), w, _timing(), overhead_fn=overhead)
    assert base.world_pool_delta == pytest.approx(0.0, abs=1e-9)
    itemized = sum(got.second_order.values())
    assert itemized > 0
    lost_tokens = itemized / (w.kv_mib_per_token_per_layer * 64)
    assert got.world_pool_delta == pytest.approx(-lost_tokens, rel=1e-9)
    assert set(got.second_order) == {
        "seam_staging_fwd",
        "seam_staging_rev",
        "tp_phase_redistribution",
    }


def test_kv_vector_shifts_onto_the_freed_3080_bytes():
    """The #320 capacity-matched pattern: layers to rank0, KV to ranks 1/2."""
    w = _world()
    inc = cosolve_prefill_cut((28, 20, 16), w, _timing())
    moved = cosolve_prefill_cut((42, 12, 10), w, _timing())
    # rank0 took 14 more layers, so its KV share must fall and the 3080s' rise.
    assert moved.kv_token_vector[0] < inc.kv_token_vector[0]
    assert moved.kv_token_vector[1] > inc.kv_token_vector[1]
    assert moved.kv_token_vector[2] > inc.kv_token_vector[2]
    # The vector is a split of the same world pool.
    assert sum(moved.kv_token_vector) == pytest.approx(
        moved.world_pool_tokens, rel=1e-9
    )


def test_rank0_cap_bounds_only_its_own_share_not_the_world_pool():
    """The revised constraint semantics, pinned."""
    w = _world()
    capped = cosolve_prefill_cut((42, 12, 10), w, _timing())
    inc = cosolve_prefill_cut((28, 20, 16), w, _timing())
    assert capped.world_pool_tokens == pytest.approx(inc.world_pool_tokens, rel=1e-9)
    # rank0 is tighter, but the world is not.
    assert capped.kv_token_vector[0] < inc.kv_token_vector[0]


def test_a_cut_that_overflows_rank0_is_refused_by_name():
    """Rank0 free bytes going negative is infeasible, not a small pool."""
    w = _world()
    with pytest.raises(ValueError, match="rank0"):
        cosolve_prefill_cut((60, 2, 2), w, _timing())


def test_speedups_survive_the_cosolve_unchanged():
    """The timing anchors are independent of the memory co-solve."""
    got = cosolve_prefill_cut((42, 12, 10), _world(), _timing())
    assert got.serial_speedup == pytest.approx(1.338, abs=0.005)
    assert got.pipelined_speedup == pytest.approx(1.667, abs=0.005)
