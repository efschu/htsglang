"""#705 as PLANNER RULES: solve() picks family ratios, on any hardware.

Binding directive (PLAN_PERF_PIPELINE_2026-08-16, "PLANNER-SOLVED, UNIVERSAL"):
the family-split verdict must land as an objective/constraint set, not as a
hand-chosen ratio for this rig. Collective costs come from the census
instruments, bandwidths from the pair matrix, capacity from the ledger.

The rules being pinned, none of which mention this rig:

* A family's BLOCKING collectives vanish iff its placement is concentrated on
  exactly one rank. Partial concentration buys nothing -- a two-rank shard still
  all-reduces. DEFERRED collectives never vanish.
* Sharded bandwidth time is the MAX over participating ranks (the slowest binds);
  solo time is bytes / bandwidth of the host rank. Solo therefore wins only when
  it escapes a slow binding rank -- which is a property of the hardware, not of
  this model.
* Capacity is a hard per-rank constraint; the world total is conserved under the
  TP sum rule, so concentration is a redistribution, not a loss.
* A decision with unmeasured collective cost is REFUSED, not guessed.

The generality test that matters is `test_equal_bandwidth_rig_prefers_sharded`:
on uniform hardware the same solver must reject the very placement it
recommends here, because there is no slow rank to escape.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest

from sglang.srt.planner.family_split import (
    CollectiveCost,
    FamilySpec,
    RankHardware,
    solve_family_placement,
)

# This rig, from the census/pair matrix -- supplied as INPUT, never assumed.
RIG = (
    RankHardware(name="5090", bandwidth_mib_per_s=1_707_077.0, capacity_mib=31_064.0),
    RankHardware(name="3080a", bandwidth_mib_per_s=724_792.0, capacity_mib=19_031.0),
    RankHardware(name="3080b", bandwidth_mib_per_s=724_792.0, capacity_mib=19_031.0),
)

GDN = FamilySpec(
    name="gdn",
    n_layers=48,
    weight_mib_per_layer=110.5,
    state_mib_per_layer_per_slot=3.0,
    blocking_collectives_per_layer=1,
    deferred_collectives_per_layer=1,
)
ATTN = FamilySpec(
    name="full_attn",
    n_layers=16,
    weight_mib_per_layer=70.0,
    state_mib_per_layer_per_slot=0.0,
    blocking_collectives_per_layer=1,
    deferred_collectives_per_layer=1,
)
MOE = FamilySpec(
    name="moe",
    n_layers=64,
    weight_mib_per_layer=350.3,
    state_mib_per_layer_per_slot=0.0,
    blocking_collectives_per_layer=0,
    deferred_collectives_per_layer=1,
)

MEASURED = CollectiveCost(blocking_us=40.0, deferred_us=5.0, source="measured")
UNMEASURED = CollectiveCost(blocking_us=None, deferred_us=None, source="unmeasured")


def _solve(cost=MEASURED, ranks=RIG, families=(GDN, ATTN, MOE), slots=12, **kw):
    return solve_family_placement(
        families=families, ranks=ranks, collective_cost=cost, mamba_slots=slots, **kw
    )


def test_unmeasured_collective_cost_is_refused_not_guessed():
    """Provenance discipline: no verdict without a measurement."""
    with pytest.raises(ValueError, match="unmeasured"):
        _solve(cost=UNMEASURED)


def test_blocking_collectives_vanish_only_under_full_concentration():
    """The core rule. A two-rank shard still all-reduces."""
    sol = _solve()
    gdn = sol.by_family["gdn"]
    if gdn.is_solo:
        assert gdn.blocking_removed == GDN.n_layers
    # Deferred collectives are never removed, whatever the placement.
    assert gdn.deferred_removed == 0


def test_moe_never_contributes_a_removable_collective():
    """It declares zero blocking collectives, so concentrating it buys nothing."""
    sol = _solve()
    assert sol.by_family["moe"].blocking_removed == 0


def test_break_even_is_derived_not_hardcoded():
    """The solver must produce the threshold from its own inputs."""
    sol = _solve()
    gdn = sol.by_family["gdn"]
    # Cost of concentrating, in ms, divided by the collectives it removes.
    assert gdn.break_even_us == pytest.approx(
        gdn.bandwidth_delta_ms * 1000.0 / GDN.n_layers, rel=1e-9
    )


def test_the_baseline_shard_POLICY_moves_the_threshold_twofold():
    """Corrects NOTE_705's desk figure, which compared against the wrong baseline.

    The note priced solo against an EQUAL 1/3 shard, where the slow 3080s bind,
    and got a 14.3 us break-even. But this fork ships uneven TP
    (--rank-tp-ratio), so the honest baseline is a BANDWIDTH-PROPORTIONAL shard,
    under which every rank finishes together and the family costs
    total_bytes / sum(bandwidth). That baseline is much faster, so concentrating
    has more to repay and the threshold roughly DOUBLES.

    Expressing the verdict as rules rather than a number is what surfaced this;
    the note's 14.3 us understated the bar.
    """
    uneven = _solve().by_family["gdn"]
    equal = _solve(allow_uneven_shards=False).by_family["gdn"]
    assert equal.break_even_us == pytest.approx(14.3, abs=1.5)
    assert uneven.break_even_us == pytest.approx(30.5, abs=2.0)
    assert uneven.break_even_us > equal.break_even_us * 1.8


def test_this_rig_recommends_concentrating_gdn_on_the_fast_rank():
    sol = _solve()
    gdn = sol.by_family["gdn"]
    assert gdn.is_solo
    assert gdn.host_rank == 0  # the 5090, by bandwidth
    assert sol.net_ms > 0


def test_equal_bandwidth_rig_prefers_sharded():
    """GENERALITY: the same rules must reject solo on uniform hardware.

    With no slow rank to escape, concentrating a family multiplies the host's
    bytes without shortening the binding term, so the bandwidth delta can never
    be repaid. A solver that still said "solo" would be fitting this rig.
    """
    uniform = tuple(
        RankHardware(
            name=f"u{i}", bandwidth_mib_per_s=1_000_000.0, capacity_mib=40_000.0
        )
        for i in range(3)
    )
    sol = _solve(ranks=uniform)
    assert not sol.by_family["gdn"].is_solo


def test_a_slower_host_is_never_chosen_as_the_solo_rank():
    sol = _solve()
    gdn = sol.by_family["gdn"]
    if gdn.is_solo:
        bws = [r.bandwidth_mib_per_s for r in RIG]
        assert bws[gdn.host_rank] == max(bws)


def test_capacity_is_a_hard_constraint():
    """A host that cannot hold the concentrated family must not be chosen."""
    tiny = (
        RankHardware(
            name="fast_tiny", bandwidth_mib_per_s=1_707_077.0, capacity_mib=100.0
        ),
        RankHardware(
            name="3080a", bandwidth_mib_per_s=724_792.0, capacity_mib=19_031.0
        ),
        RankHardware(
            name="3080b", bandwidth_mib_per_s=724_792.0, capacity_mib=19_031.0
        ),
    )
    sol = _solve(ranks=tiny)
    assert not sol.by_family["gdn"].is_solo


def test_world_capacity_is_conserved_by_concentration():
    """TP sum rule: concentration redistributes, it does not consume."""
    sol = _solve()
    assert sum(sol.capacity_delta_mib) == pytest.approx(0.0, abs=1e-6)
    # And it is a real redistribution, not a no-op.
    assert max(sol.capacity_delta_mib) > 0
    assert min(sol.capacity_delta_mib) < 0


def test_sharded_ratios_follow_bandwidth_proportion():
    """The non-solo fallback splits by bandwidth, giving 2.4:1:1 here."""
    sol = _solve()
    ratios = sol.by_family["moe"].ratios
    assert sum(ratios) == pytest.approx(1.0, rel=1e-9)
    assert ratios[0] / ratios[1] == pytest.approx(
        RIG[0].bandwidth_mib_per_s / RIG[1].bandwidth_mib_per_s, rel=1e-9
    )
