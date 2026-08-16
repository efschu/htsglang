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
from sglang.srt.planner.seam_holdback import SeamRecord, available_bytes_for_cut

MIB = 1024 * 1024
CELL = 2048
W_PER_LAYER = 450.7 * MIB
MAMBA_PER_LINEAR = 50.85 * MIB
# MEASURED at the instrument boot c3e94878ff (PP-phase sizing 22:11:08). The
# chain is rest -> profiled -> holdback -> adjusted -> available_bytes -> tokens,
# and `adjusted` IS `available_bytes` to the capture's 0.1 MiB print precision.
AVAIL0 = (8_524_386_304.0, 4_740_280_320.0, 3_573_989_376.0)
# Per-rank holdback in that PP pass. Note the spread -- PP2 holds back far more
# than PP0/PP1 -- and note that the TP-stack pass holds back 0.000% on every
# rank. So the holdback is a PP-phase seam adjustment, not a flat reserve, and
# it must not be extrapolated as a constant across layouts.
HOLDBACK_PCT = (45.143, 44.074, 60.258)
INCUMBENT = (28, 20, 16)
INCUMBENT_ATTN = (7, 5, 4)
OBSERVED_POOL = 436_278.0  # binder PP2 on the instrument boot
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


# #707 CLOSED FORM (Slot-2). available_bytes comes from the seam cap, not from
# an extrapolation of my own:
#   allowed_tokens = id_space + (free_at_measure - arming_floor - margin) / cell
#   holdback_frac  = 1 - allowed_tokens / (profiled_bytes / cell)
# Everything except the layout SHIFT of free_at_measure is exact.
_ALLOWED = tuple(
    a * MIB / c for a, c in zip((8129.5, 4520.7, 3408.4), (14336, 10240, 8192))
)
_RECORD = SeamRecord(
    id_space_tokens=min(_ALLOWED),
    bracket_mib=tuple(
        (_ALLOWED[i] - min(_ALLOWED)) * (14336, 10240, 8192)[i] / MIB for i in range(3)
    ),
    cell_bytes=(14336, 10240, 8192),
)


def _avail_for(counts, attn):
    return available_bytes_for_cut(_RECORD, INCUMBENT, INCUMBENT_ATTN, counts, attn)


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


def test_the_coupled_pool_declines_with_depth_but_buys_the_first_steps_cheaply():
    """The price in the regime that exists today, under the #707 closed form.

    An earlier extrapolation of mine made this look catastrophic from the start
    (-17% at [33,15,16]). The seam cap is materially kinder: allowed_tokens is
    floored at id_space, which does not shrink, so the first few layers onto the
    fast card cost very little. It is a real decline, not a collapse.
    """
    f = _solve()
    by_cut = {p.counts[0]: p for p in f.points}
    assert by_cut[33].coupled_pool_tokens > 0.90 * OBSERVED_POOL, "cheap early"
    assert by_cut[42].coupled_pool_tokens < 0.50 * OBSERVED_POOL, "expensive late"
    assert by_cut[33].coupled_pool_tokens > by_cut[42].coupled_pool_tokens


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


def test_both_optima_are_INTERIOR_not_the_deepest_cut():
    """Neither curve rewards maximum depth, and for two different reasons.

    WITHOUT the lever, overhead outgrows the compute gain past [42,11,11].
    WITH it, the raw compute speedup itself peaks at [44,10,10] and falls after:
    piling layers onto the fast card eventually makes a TAIL stage the
    bottleneck, so the pipeline stops improving. "More layers on the 5090" has
    a limit that is not about memory at all.
    """
    f = _solve()
    deepest = f.points[-1]
    with_lever = f.best_with_pipelining()
    assert with_lever.net_pipelined == pytest.approx(2.000, abs=0.02)
    assert with_lever.counts != deepest.counts, "the deepest cut is not the best"
    assert with_lever.compute_speedup > deepest.compute_speedup

    without = f.best_without_pipelining()
    assert without.counts[0] < with_lever.counts[0], "overhead binds even earlier"


def test_the_seam_cap_bounds_how_deep_the_cut_may_go():
    """A hard boundary the closed form supplies and an extrapolation cannot.

    Past some depth a rank's free column no longer clears its arming floor, so
    the layout cannot ARM a flip. Those cuts are refused by the provider and
    never reach the frontier -- not priced as a tiny pool, absent.
    """
    f = _solve()
    assert max(p.counts[0] for p in f.points) < 64 - 2, "some depth is refused"


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


def test_the_booted_layout_is_measured_and_the_rest_are_not():
    """Provenance is PER LAYOUT, not per frontier.

    The incumbent's available_bytes come from its own sizing lines and are
    exact. Every other cut is extrapolated across a holdback that is known to
    vary by rank (45.1 / 44.1 / 60.3 %) and that vanishes entirely in the TP
    pass, so treating it as constant is precisely the assumption under test.
    One global flag would let the measured row launder the extrapolated ones.
    """
    f = _solve(measured_for=lambda c: tuple(c) == INCUMBENT)
    booted = [p for p in f.points if p.pool_measured]
    assert len(booted) == 1 and booted[0].counts == INCUMBENT
    assert not f.measured, "a frontier over candidate cuts is not measured"
    # And the measured row must reproduce the boot's own binder.
    assert booted[0].coupled_pool_tokens == pytest.approx(OBSERVED_POOL, rel=2e-3)


def test_the_measured_cell_confirms_the_attention_layer_rule():
    """cell = attn_layers x 2048, from the boot's own sizing line.

    Logged 14336 / 10240 / 8192 against attention counts 7 / 5 / 4 -- exact on
    every rank. This is the third independent confirmation of the divisor that
    #702 originally disputed, and the first from a live sizer rather than from
    allocator source or a K-size back-computation.
    """
    for attn, cell in zip(INCUMBENT_ATTN, (14336, 10240, 8192)):
        assert attn * CELL == cell


def test_holdback_is_not_a_constant_and_the_solver_does_not_pretend_it_is():
    """The open term, pinned as a property rather than a number.

    The spread across ranks (and its disappearance in the TP pass) is why an
    unbooted cut cannot be called measured. Slot-2 owns the formula that
    derives these percentages from the seam machinery; until it lands the
    extrapolation is labelled, not corrected.
    """
    assert max(HOLDBACK_PCT) - min(HOLDBACK_PCT) > 15.0
    f = _solve(measured_for=lambda c: tuple(c) == INCUMBENT)
    assert any(not p.pool_measured for p in f.points)


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
