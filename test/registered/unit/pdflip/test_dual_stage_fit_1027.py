"""Dual P stage fit from ``Prefill rank batch`` lines (planner/dual_stage_fit.py, 27.10.).

The dual boots print no #PGAP, so the only per-forward stage times are the rank lines, and those carry NO prefix depth.
These cases pin the derivations that a "looks equivalent" rewrite would break silently."""

import datetime

from flliper.srt.planner import dual_stage_fit as dsf

A = (100.0, 325.0, 336.0)       # ms at depth 0, per 1024-token chunk (the 27.10. W4A16 fits, rounded)
B = (1.46, 2.55, 2.20)          # ms per 1k of mean depth


def _line(t: float, rank: int, new: int, ms: float, rung: int = 0) -> str:
    stamp = datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
    return (f"[{stamp} PP{rank}] Prefill rank batch, #new-token: {new}, #cached-token: 0, #chunks: 1, "
            f"gpu-ms: {ms:.1f} (compute {ms:.1f}, wait 0.0) bubble_ms=1.0 (between forwards, mb=0) rung={rung}")


def _prompt_lines(t0: float, chunks: int, *, rung: int = 0):
    """One prompt of ``chunks`` 1024-token forwards on the three ranks, 0.4 s apart, times exactly on the lines."""
    lines = []
    for i in range(chunks):
        x = (i * 1024 + 512) / 1000.0
        for rank in range(3):
            lines.append(_line(t0 + 0.4 * i + 0.01 * rank, rank, 1024, A[rank] + B[rank] * x, rung))
    return lines


def test_fa_layers_follow_the_layer_index_not_the_stage_size():
    # every 4th layer (index % 4 == 3) is full attention; the 16 FA layers of the 64 sit where the cut puts them
    assert dsf.fa_layers((31, 17, 16)) == (7, 5, 4)
    assert dsf.fa_layers((33, 16, 15)) == (8, 4, 4)
    assert dsf.fa_layers((34, 15, 15)) == (8, 4, 4)
    assert dsf.fa_layers((36, 14, 14)) == (9, 3, 4)


def test_fit_rebuilds_depth_from_consecutive_forwards_and_restarts_at_a_gap():
    # two prompts 40 s apart (each lasts 16 s): a rank that did not restart the depth at the gap would fit a slope ~1/2 of the true one
    lines = _prompt_lines(1_000_000.0, 40) + _prompt_lines(1_000_040.0, 40)
    batches = dsf.parse_rank_batches(lines)
    for rank in range(3):
        fit = dsf.fit_line(dsf.depth_points(batches, rank=rank))
        assert abs(fit.a_ms - A[rank]) < 0.5, (rank, fit)
        assert abs(fit.b_ms_per_1k - B[rank]) < 0.02, (rank, fit)


def test_rungs_other_than_the_asked_one_are_not_mixed_into_the_fit():
    # a stage taking 2x under a lower P share (rung 1) must not bend the rung-0 line
    lines = _prompt_lines(2_000_000.0, 40) + _prompt_lines(2_000_040.0, 40, rung=1)
    lines = [ln.replace("gpu-ms: ", "gpu-ms: 1") if "rung=1" in ln else ln for ln in lines]
    batches = dsf.parse_rank_batches(lines)
    fit = dsf.fit_line(dsf.depth_points(batches, rank=1, rung=0))
    assert abs(fit.a_ms - A[1]) < 0.5


def test_costs_round_trip_and_one_layer_moves_one_layer_of_cost():
    fits = [dsf.StageFit(a_ms=A[r], b_ms_per_1k=B[r], sd_ms=0.0, n=99) for r in range(3)]
    cut = (31, 17, 16)
    costs = dsf.costs_from_fits(fits, cut)
    for depth in (0.5, 64.0):
        got = dsf.stage_ms(cut, costs, depth_1k=depth)
        want = tuple(A[r] + B[r] * depth for r in range(3))
        # the depth slope is pooled over the two 3080 stages (per FA layer), so a stage is exact only to ~2 %
        assert all(abs(g - w) < 0.02 * w for g, w in zip(got, want)), (depth, got, want)
    # the last stage keeps a per-forward remainder beyond its layers (PP2 is slower than PP1 per layer)
    assert costs.last_stage_ms > 0
    # 33,16,15 takes one layer off PP1 and one off PP2 (both FA counts stay 4): each stage shrinks by one slow layer
    moved = dsf.stage_ms((33, 16, 15), costs, depth_1k=0.5)
    base = dsf.stage_ms(cut, costs, depth_1k=0.5)
    assert abs((base[1] - moved[1]) - costs.layer_ms_slow - (costs.attn_ms_per_1k_slow * 0.5)) < 1e-6
    assert moved[2] < base[2] and moved[0] > base[0]


def test_calibration_makes_the_measured_cut_reproduce_its_own_ttft():
    fits = [dsf.StageFit(a_ms=A[r], b_ms_per_1k=B[r], sd_ms=0.0, n=99) for r in range(3)]
    cut = (31, 17, 16)
    costs = dsf.costs_from_fits(fits, cut)
    raw = dsf.ttft_s(cut, costs, prompt_tokens=65536)
    cal = dsf.calibration(cut, costs, {65536: raw * 1.1})
    assert abs(dsf.ttft_s(cut, costs, prompt_tokens=65536, calibration=cal) - raw * 1.1) < 1e-9
