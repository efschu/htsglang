"""#704: emit the TRUE per-rank holdback at its source.

The fourth missing instrument. The reserve between the profiler's ``rest`` and
what ``calculate_pool_sizes`` actually receives lands on metal at 6,688 / 3,561
/ 5,166 MiB, while ``derived_rank_auto_reserve_mib`` returns **4,160 uniformly**
for the same boot's arguments. So the held-back amount is NOT that function's
output, and three collinear data points cannot identify its form (see
DESIGN_704_reserve_vs_layout.md: single-factor layer models are already
falsified by non-monotonicity, and layers/attn/gdn are exactly collinear on any
multiple-of-4 cut).

Rather than fit a curve through a black box the process already knows, emit what
it actually holds back. ``_config_from_budget`` is the right site: its own
comment calls it "the one funnel every sizing path reaches", and the holdback is
precisely the delta across ``_seam_adjusted_budget`` there.

The helper is pure so it is testable without a boot, and it reports rather than
sanitises: a NEGATIVE holdback (the seam handing budget back) and a holdback
exceeding the input are both real states that must surface, not be clamped.

Hermetic: pure arithmetic, no CUDA.
"""

import pytest

from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    budget_holdback_mib,
)

_MIB = 1 << 20


def test_the_holdback_is_the_delta_across_the_seam_adjustment():
    got = budget_holdback_mib(profiled_bytes=8577 * _MIB, adjusted_bytes=3410 * _MIB)
    assert got == pytest.approx(5167.0, abs=1.0)


def test_it_reproduces_the_three_metal_reserves():
    """The numbers DESIGN_704 could not attribute; this is what settles them."""
    cases = (
        (14.472 * 1024, 8526565376 / _MIB, 6688.0),
        (7.894 * 1024, 4741928960 / _MIB, 3561.0),
        (8.375 * 1024, 3575365632 / _MIB, 5166.0),
    )
    for rest_mib, adjusted_mib, want in cases:
        got = budget_holdback_mib(
            profiled_bytes=int(rest_mib * _MIB), adjusted_bytes=int(adjusted_mib * _MIB)
        )
        assert got == pytest.approx(want, abs=2.0)


def test_a_zero_holdback_is_reported_as_zero_not_omitted():
    """A measured zero is information; treating it as absence hides a real state."""
    assert (
        budget_holdback_mib(profiled_bytes=1000 * _MIB, adjusted_bytes=1000 * _MIB)
        == 0.0
    )


def test_a_negative_holdback_is_surfaced_not_clamped():
    """The seam handing budget BACK is a real state and a loud one.

    Clamping it to zero would hide exactly the accounting error class this
    ticket has spent four instruments chasing.
    """
    got = budget_holdback_mib(profiled_bytes=1000 * _MIB, adjusted_bytes=1200 * _MIB)
    assert got == pytest.approx(-200.0, abs=0.01)
    assert got < 0


def test_a_holdback_consuming_the_whole_budget_is_representable():
    assert budget_holdback_mib(profiled_bytes=500 * _MIB, adjusted_bytes=0) == 500.0


def test_the_fraction_helper_refuses_a_zero_budget_rather_than_dividing():
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
        budget_holdback_fraction,
    )

    assert budget_holdback_fraction(1000 * _MIB, 750 * _MIB) == pytest.approx(0.25)
    assert budget_holdback_fraction(0, 0) is None
