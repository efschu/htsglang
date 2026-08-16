"""#702: the PP cut solved for PREFILL SPEED, with the pool price stated.

The user's question was direct: more layers on the fast card, what does it cost?
The capacity solver never covered that objective. This pins the answer and, more
importantly, the two shapes in it that a table would hide.

Calibration data for this rig, which lives here and NOT in the solver:

* metal ``available_bytes`` at the incumbent, from the four-boot gate:
  8,526,565,376 / 4,741,928,960 / 3,575,365,632 (PP2 the binder);
* 450.7 MiB per layer of weights, 50.85 MiB per GDN layer of state;
* 2048 B per token per attention layer (fp8_e4m3, config-derived);
* measured links x8 = 13 GB/s, x4 = 6.4 GB/s, with the 5090 on x8;
* per-layer prefill 1.7571 / 7.740 / 7.275 ms.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest
from sglang.srt.planner.prefill_frontier import (
    PrefillFrontierError,
    solve_prefill_frontier,
)

MIB = 1024 * 1024
CELL = 2048
W_PER_LAYER = 450.7 * MIB
MAMBA_PER_LINEAR = 50.85 * MIB
AVAIL0 = (8_526_565_376.0, 4_741_928_960.0, 3_575_365_632.0)
INCUMBENT = (28, 20, 16)
INCUMBENT_ATTN = (7, 5, 4)
OBSERVED_POOL = 436_766.0
MS_PER_LAYER = (1.7571, 7.740, 7.275)
X8 = 13.0e9 / MIB
X4 = 6.4e9 / MIB
LINKS = (X8, X8, X4)  # 5090 on x8; the x4 card carries a 3080
GATHER_MIB = 24.09
FAM = tuple((i + 1) % 4 == 0 for i in range(64))


def _attn_for(counts):
    out, s = [], 0
    for c in counts:
        out.append(sum(1 for i in range(s, s + c) if FAM[i]))
        s += c
    return tuple(out)


def _avail_for(counts, attn):
    return [
        AVAIL0[i]
        - (counts[i] - INCUMBENT[i]) * W_PER_LAYER
        - ((counts[i] - attn[i]) - (INCUMBENT[i] - INCUMBENT_ATTN[i]))
        * MAMBA_PER_LINEAR
        for i in range(3)
    ]


def _solve(**over):
    kw = {
        "total_layers": 64,
        "n_stages": 3,
        "incumbent": INCUMBENT,
        "incumbent_pool_tokens": OBSERVED_POOL,
        "ms_per_layer": MS_PER_LAYER,
        "attn_counts_for": _attn_for,
        "available_bytes_for": _avail_for,
        "kv_bytes_per_token_per_attn_layer": CELL,
        "total_attn_layers": 16,
        "gather_mib_per_attn_layer": GATHER_MIB,
        "link_mib_per_s": LINKS,
    }
    kw.update(over)
    return solve_prefill_frontier(**kw)


def test_the_decoupled_pool_is_exactly_cut_independent():
    """The structural fact that makes depth affordable at all.

    Total weight bytes and total GDN state are invariant under a re-cut -- only
    their DISTRIBUTION moves -- so the sum-rule pool does not depend on the cut.
    Not approximately: exactly.
    """
    f = _solve()
    pools = {round(p.decoupled_pool_tokens, 6) for p in f.points}
    assert len(pools) == 1
    only = f.points[0].decoupled_pool_tokens
    assert only == pytest.approx(514_034, rel=1e-3)
    # And it is ABOVE the incumbent's observed pool: the price is negative.
    assert only > OBSERVED_POOL


def test_the_coupled_pool_collapses_with_depth():
    """The price in the regime that exists today, and it is brutal."""
    f = _solve()
    by_cut = {p.counts[0]: p for p in f.points}
    assert by_cut[33].coupled_pool_tokens < OBSERVED_POOL
    assert by_cut[42].coupled_pool_tokens < 0.20 * OBSERVED_POOL
    deep = [p.coupled_pool_tokens for p in f.points if p.counts[0] >= 36]
    assert deep == sorted(deep, reverse=True), "coupled pool must fall with depth"


def test_net_speedup_without_pipelining_is_NOT_monotone():
    """THE result that makes this a solve rather than a table.

    Past some depth the collective overhead grows faster than the compute gain,
    so a deeper cut is actively WORSE -- not merely diminishing. A frontier that
    reported only compute speedup would recommend exactly those cuts.
    """
    f = _solve()
    nets = [p.net_no_pipelining for p in f.points]
    assert nets != sorted(nets), "expected a peak, not a monotone rise"
    best = f.best_without_pipelining()
    assert best.counts[0] < f.points[-1].counts[0], "the peak is interior"
    assert best.net_no_pipelining == pytest.approx(1.660, abs=0.03)


def test_with_pipelining_the_deepest_cut_wins():
    f = _solve()
    best = f.best_with_pipelining()
    assert best.counts == f.points[-1].counts
    assert best.net_pipelined == pytest.approx(2.000, abs=0.02)


def test_cuts_past_the_peak_are_flagged_as_needing_the_lever():
    """What the user must not be handed without a caveat."""
    f = _solve()
    peak = f.best_without_pipelining().counts[0]
    for p in f.points:
        if p.counts[0] > peak:
            assert p.needs_pipelining, f"{p.counts} should require pipelining"
    for p in f.points:
        if p.counts[0] <= peak:
            assert not p.needs_pipelining


def test_the_shallowest_decoupled_cuts_do_not_pay_for_themselves():
    """Decoupling is net-NEGATIVE until there is enough depth to justify it.

    At the incumbent depth the collective buys nothing and costs overhead, so
    the honest frontier starts below 1.0.
    """
    f = _solve()
    assert f.points[0].net_no_pipelining < 1.0


def test_results_are_self_labelled_as_extrapolated_by_default():
    assert _solve().measured is False
    assert _solve(measured=True).measured is True


def test_no_rig_constants_leak_into_the_solver():
    """A foreign profile: different depth, stages, links and geometry."""
    fam = tuple((i + 1) % 2 == 0 for i in range(24))

    def attn(counts):
        out, s = [], 0
        for c in counts:
            out.append(sum(1 for i in range(s, s + c) if fam[i]))
            s += c
        return tuple(out)

    f = solve_prefill_frontier(
        total_layers=24,
        n_stages=2,
        incumbent=(12, 12),
        incumbent_pool_tokens=100_000.0,
        ms_per_layer=(1.0, 4.0),
        attn_counts_for=attn,
        available_bytes_for=lambda c, a: [8e9, 8e9],
        kv_bytes_per_token_per_attn_layer=4096,
        total_attn_layers=12,
        gather_mib_per_attn_layer=10.0,
        link_mib_per_s=(20000.0, 20000.0),
    )
    assert len(f.points) >= 2
    for p in f.points:
        assert sum(p.counts) == 24
        assert len(p.counts) == 2


def test_malformed_inputs_are_refused():
    with pytest.raises(PrefillFrontierError, match="stages"):
        _solve(incumbent=(28, 20))
    with pytest.raises(PrefillFrontierError, match="sums to"):
        _solve(incumbent=(28, 20, 15))
    with pytest.raises(PrefillFrontierError, match="every stage"):
        _solve(link_mib_per_s=(X8, X8))
