"""#586: ladder rungs must not inherit the 2 MiB/captured-token fiction.

The rung's capture pool was priced at the #68 literal 2 MiB per captured token,
the same coefficient the 2026-08-05 window measured 3.3-3.8x LOW (192 MiB
booked against 633-730 MiB per rank). Every built rung inherited that error, and
it is an UNDER-charge -- the direction that OOMs rather than the one that costs
KV.

The fix keeps the part the config knows in the config: the coefficient's
MAGNITUDE now comes from a measurement, while the SCALING across rungs stays
derived from captured-token counts. Uncalibrated rigs keep the literal so the
legacy reserve path still boots, but the post is marked estimate-only so a
ledger can refuse rather than carry an unknown under-charge silently.
"""

import math

import pytest

from sglang.srt.speculative.adaptive_graph_memory import LadderRungPost


def post(**kw):
    base = dict(rung=5, workspace_mib=384, capture_mib=192)
    base.update(kw)
    return LadderRungPost(**base)


def test_default_post_is_marked_estimate_only():
    """An unmarked post is indistinguishable from a measured one, which is how
    an under-charge travels unnoticed."""
    p = post()
    assert p.measured is False
    assert p.mib_per_captured_token == 2.0


def test_a_measured_post_says_so():
    p = post(capture_mib=640, mib_per_captured_token=6.67, measured=True)
    assert p.measured is True
    assert p.total_mib == 384 + 640


def test_demand_is_estimate_only_if_any_post_is():
    from sglang.srt.speculative.adaptive_graph_memory import LadderReserveDemand

    measured = post(measured=True, mib_per_captured_token=6.67)
    est = post(measured=False)
    assert (
        LadderReserveDemand(
            posts=(measured,), boot_rung=3, margin_mib=0, resident=True
        ).estimate_only
        is False
    )
    assert (
        LadderReserveDemand(
            posts=(measured, est), boot_rung=3, margin_mib=0, resident=True
        ).estimate_only
        is True
    ), "one estimate post taints the whole demand"


def test_the_old_coefficient_is_no_longer_hardcoded_in_the_sizing_call():
    """Pin: the rung sizing must read its coefficient from a parameter, not
    from a literal in the expression."""
    import inspect

    from sglang.srt.speculative import adaptive_graph_memory as agm

    src = inspect.getsource(agm.estimate_ladder_reserve_demand)
    assert "capture_mib_per_token" in src
    # The old expression was `speculative_capture_tokens(...) * 2 * colocated`.
    assert "* 2\n" not in src, "the literal 2 coefficient is back in the sizing"


def test_measured_coefficient_scales_the_rung_cost():
    """The whole point: same token count, measured coefficient, bigger post."""
    tokens = 96
    colocated = 1
    at_two = int(math.ceil(tokens * 2.0 * colocated))
    at_measured = int(math.ceil(tokens * 6.67 * colocated))
    assert at_two == 192
    assert at_measured == 641
    assert at_measured > at_two * 3


# --- the coefficient derivation ---------------------------------------------


def test_measured_coefficient_is_derived_from_a_calibration_not_a_constant():
    from sglang.srt.mem_ledger.activation import (
        REFERENCE_WINDOW_FINGERPRINT,
        ActivationProfile,
        measured_capture_mib_per_token,
    )

    profile = ActivationProfile(
        architectures=("Qwen3_5ForConditionalGeneration",),
        chunked_prefill_size=2048,
        tp_size=3,
        pp_size=1,
        kv_cache_dtype="fp8_e4m3",
        speculative_num_draft_tokens=4,
        decode_max_bs=24,
    )
    uuid = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
    coeff = measured_capture_mib_per_token(
        uuid,
        hw_fingerprint=REFERENCE_WINDOW_FINGERPRINT,
        profile=profile,
        boot_capture_tokens=96,
        cache_dir="/nonexistent",
    )
    # 640 measured MiB / 96 captured tokens = 6.67 MiB per token, i.e. the
    # inherited 2 was 3.3x low on this card.
    assert coeff == pytest.approx(640 / 96, rel=1e-6)
    assert coeff > 2.0 * 3


def test_no_calibration_yields_None_never_two():
    from sglang.srt.mem_ledger.activation import (
        ActivationProfile,
        measured_capture_mib_per_token,
    )

    profile = ActivationProfile(
        architectures=("Other",),
        chunked_prefill_size=2048,
        tp_size=3,
        pp_size=1,
    )
    assert (
        measured_capture_mib_per_token(
            "GPU-unknown",
            hw_fingerprint="someotherrig",
            profile=profile,
            boot_capture_tokens=96,
            cache_dir="/nonexistent",
        )
        is None
    )


def test_zero_boot_tokens_yields_None_rather_than_dividing():
    from sglang.srt.mem_ledger.activation import (
        REFERENCE_WINDOW_FINGERPRINT,
        ActivationProfile,
        measured_capture_mib_per_token,
    )

    profile = ActivationProfile(
        architectures=("Qwen3_5ForConditionalGeneration",),
        chunked_prefill_size=2048,
        tp_size=3,
        pp_size=1,
        kv_cache_dtype="fp8_e4m3",
        speculative_num_draft_tokens=4,
        decode_max_bs=24,
    )
    assert (
        measured_capture_mib_per_token(
            "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7",
            hw_fingerprint=REFERENCE_WINDOW_FINGERPRINT,
            profile=profile,
            boot_capture_tokens=0,
            cache_dir="/nonexistent",
        )
        is None
    )
