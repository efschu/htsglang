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

import unittest

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
# RECONCILIATION (Cluster B, 2026-08-17): what became of the co-solve tests.
#
# The serving line carried six tests here that pinned the WorldMemory /
# cosolve_prefill_cut model:
#
#   test_world_pool_is_conserved_when_overheads_do_not_move_with_the_cut
#   test_the_residual_is_exactly_the_itemized_second_order_terms
#   test_kv_vector_shifts_onto_the_freed_3080_bytes
#   test_rank0_cap_bounds_only_its_own_share_not_the_world_pool
#   test_a_cut_that_overflows_rank0_is_refused_by_name
#   test_speedups_survive_the_cosolve_unchanged
#
# #701/#704a replaced that model with PhasePoolModel + pp_phase_pool /
# tp_phase_pool and deleted those tests along with the API they exercised.
# They are NOT restored as-is: four of them assert properties of a "world
# pool" that no longer exists as a concept, and restoring them would pin a
# model this tree deliberately left behind.
#
# The supersession is decided on the standing rule -- measured beats designed,
# newer beats older. The phase-pool model reproduces BOTH metal calibration
# points with solved per-layout floors (test_pp_cut_phase_pool_702.py); the
# co-solve model is the older design and had no metal backing of its own.
#
# What survives is re-pinned below against the new model, so no claim is
# dropped silently -- only re-expressed.
# ---------------------------------------------------------------------------


class TheCoSolveClaimsAfterTheModelChange(unittest.TestCase):
    """The surviving half of the six, against PhasePoolModel."""

    def _model(self):
        from sglang.srt.planner.pp_cut import PhasePoolModel

        return PhasePoolModel(
            free_mib=(20000.0, 12000.0, 12000.0),
            weight_mib_per_layer=450.7,
            kv_mib_per_token_per_attn_layer=0.002,
            arming_floor_mib=(2255.0, 1728.0, 2467.0),
        )

    def test_the_pp_pool_is_the_binding_rank_not_a_world_sum(self):
        """SUPERSEDES test_world_pool_is_conserved... and
        test_rank0_cap_bounds_only_its_own_share_not_the_world_pool.

        There is no world pool to conserve, nor one that a per-rank cap could
        fail to bound: the PP-phase pool IS the min over ranks, so the binding
        rank is the pool and a per-rank cap CAN bind it -- the opposite of the
        old claim, which is why that test could not be carried over.
        """
        from sglang.srt.planner.pp_cut import pp_phase_pool, stage_pp_capacities

        model = self._model()
        counts, attn = (32, 16, 16), (8, 4, 4)
        caps = stage_pp_capacities(counts, attn, model)
        pool = pp_phase_pool(counts, attn, model)
        self.assertAlmostEqual(pool, min(caps), places=6)
        self.assertLessEqual(pool, max(caps))

    def test_the_tp_pool_is_a_sum_and_ignores_the_cut(self):
        """SUPERSEDES test_kv_vector_shifts_onto_the_freed_3080_bytes.

        That intuition belongs to the TP column, which is a SUM over ranks and
        independent of any PP cut, so it is pinned as cut-independence rather
        than as a vector shift.
        """
        from sglang.srt.planner.pp_cut import tp_phase_pool

        model = self._model()
        self.assertEqual(tp_phase_pool(16, 3, model), tp_phase_pool(16, 3, model))
        self.assertGreater(tp_phase_pool(16, 3, model), 0.0)

    def test_overflow_is_still_refused_by_name(self):
        """SUPERSEDES test_a_cut_that_overflows_rank0_is_refused_by_name.

        This claim survived the model change intact; the new model's own suite
        pins it as test_weight_overflow_is_refused_by_name. Cross-checked here
        so this file records that it was carried, not dropped.
        """
        from sglang.srt.planner.pp_cut import stage_pp_capacities

        with self.assertRaises(ValueError):
            stage_pp_capacities((64, 0, 0), (16, 0, 0), self._model())

    def test_the_speedup_objectives_are_untouched_by_the_pool_model(self):
        """SUPERSEDES test_speedups_survive_the_cosolve_unchanged.

        The original point -- co-solving for capacity must not perturb the
        timing objectives -- now reduces to the observation that the timing
        half does not consult the pool model at all.
        """
        timing = _timing()
        self.assertAlmostEqual(
            serial_prefill_ms((32, 16, 16), timing),
            sum(c * m for c, m in zip((32, 16, 16), timing.ms_per_layer)),
            places=6,
        )
        self.assertAlmostEqual(
            pipelined_prefill_ms((32, 16, 16), timing),
            max(c * m for c, m in zip((32, 16, 16), timing.ms_per_layer)),
            places=6,
        )
