"""Falsifiers for the phase-footprint terms (prefill activation, graph capture).

Hermetic. The inherited formulas were falsified by the 2026-08-05 reference
window IN OPPOSITE DIRECTIONS, so these tests guard both directions:

    activation   booked 3968 MiB, measured <= 1766 MiB on the binding card
                 -> over-charged; costs KV pool
    capture      booked  192 MiB, measured 633-730 MiB per rank
                 -> under-charged ~3.3-3.8x; the direction that OOMs

The activation falsification is categorical, not a matter of degree: the 3080
at CUDA ordinal 1 completed a 70018-token prefill with 1766 MiB free. Had the
peak needed 3968 MiB that boot would have died.
"""

import pytest

from sglang.srt.mem_ledger.activation import (
    REFERENCE_WINDOW_FINGERPRINT,
    ActivationProfile,
    FootprintProvenance,
    PhaseFootprint,
    load_footprints,
    profile_key,
    reference_window_footprints,
    resolve_phase_footprint,
    save_footprints,
)
from sglang.srt.mem_ledger.engine import (
    TERM_ACTIVATION,
    TERM_GRAPH_CAPTURE,
    CardFacts,
    DemandInputs,
    build_card_ledgers,
)
from sglang.srt.mem_ledger.terms import (
    DEFAULT_USER_RESERVE_MIB,
    LedgerOvercommit,
    Provenance,
)

BINDING_3080 = "GPU-5c648f96-be1d-42d5-0221-34d11ab137f7"
CARD = CardFacts(gpu_id=1, uuid=BINDING_3080, name="RTX 3080", total_mib=20480)

REF_PROFILE = ActivationProfile(
    architectures=("Qwen3_5ForConditionalGeneration",),
    chunked_prefill_size=2048,
    tp_size=3,
    pp_size=1,
    kv_cache_dtype="fp8_e4m3",
    speculative_num_draft_tokens=4,
    decode_max_bs=24,
)

#: The falsified heuristic, restated ONLY here so a test can assert the ledger
#: never produces it. Production must not contain this expression.
FALSIFIED_ACTIVATION_MIB = 512 + 2048 * 1.5 + 3 * 1 / 8 * 1024  # 3968


class FakeResidual:
    uuid = BINDING_3080
    name = "RTX 3080"
    cuda_context_bytes = 223 << 20
    allocator_granularity_bytes = 16 << 20
    lazy_workspace_bytes = 72 << 20
    total_bytes = (223 + 16 + 72) << 20
    total_mib = 311


class FakeCalibration:
    fingerprint = REFERENCE_WINDOW_FINGERPRINT

    def by_uuid(self):
        return {BINDING_3080: FakeResidual()}


def inputs(activation, capture_mib=None, fingerprint=REFERENCE_WINDOW_FINGERPRINT):
    return DemandInputs(
        weight_mib_per_rank=[0],
        activation_mib_per_rank=[activation],
        capture_tokens_per_rank=[96],
        capture_mib_per_rank=capture_mib,
        mamba_pool_mib_per_rank=[900.0],
        chunked_prefill_size=2048,
        phase_footprint_fingerprint=fingerprint,
        phase_footprint_source_per_rank=("[upper_bound] reference window",),
    )


def ledger(activation, capture_mib=None):
    return build_card_ledgers(
        inputs(activation, capture_mib),
        cards=[CARD],
        rank_gpu_id=[1],
        user_reserve_mib={1: DEFAULT_USER_RESERVE_MIB},
        calibration=FakeCalibration(),
    )[0]


# --- the falsification itself ----------------------------------------------


def test_the_inherited_heuristic_exceeds_what_the_card_had_free():
    """The measurement that killed the formula, as an assertion."""
    measured_free_on_binding_card = 1766
    assert FALSIFIED_ACTIVATION_MIB == 3968
    assert FALSIFIED_ACTIVATION_MIB > measured_free_on_binding_card, (
        "if this ever becomes false the falsification argument needs redoing"
    )


def test_ledger_never_produces_the_falsified_activation_number():
    x = ledger(1766.0)
    assert x.term(TERM_ACTIVATION).mib == 1766
    assert x.term(TERM_ACTIVATION).mib != int(FALSIFIED_ACTIVATION_MIB)


def test_activation_term_is_calibrated_not_modeled():
    term = ledger(1766.0).term(TERM_ACTIVATION)
    assert term.provenance is Provenance.CALIBRATED
    assert term.fingerprint == REFERENCE_WINDOW_FINGERPRINT


# --- refusal-honesty: no silent fallback -----------------------------------


def test_uncalibrated_activation_refuses_and_names_the_probe():
    """THE CORE REQUIREMENT. No probe result and no cached fingerprint must
    produce a REFUSAL, never the falsified heuristic."""
    x = ledger(None)
    assert x.unbounded
    assert not x.fits
    with pytest.raises(LedgerOvercommit) as excinfo:
        from sglang.srt.mem_ledger.contract import enforce_boot_contract

        enforce_boot_contract([x], log=False)
    text = str(excinfo.value)
    assert "probe_activation.py" in text
    assert "3968" in text  # it names what it refuses to fall back to
    assert "does NOT fall back" in text


def test_uncalibrated_activation_is_not_silently_zero():
    x = ledger(None)
    assert x.term(TERM_ACTIVATION) is None
    assert x.unbounded, "a missing activation must be unbounded, never absent"


def test_a_calibrated_term_without_a_fingerprint_is_rejected():
    """No 'unknown' sentinel: an un-invalidatable calibrated number is a
    literal wearing a label."""
    from sglang.srt.mem_ledger.terms import LedgerError

    with pytest.raises(LedgerError):
        build_card_ledgers(
            inputs(1766.0, fingerprint=""),
            cards=[CARD],
            rank_gpu_id=[1],
            user_reserve_mib={1: DEFAULT_USER_RESERVE_MIB},
            calibration=FakeCalibration(),
        )


# --- capture: the under-charge -----------------------------------------------


def test_measured_capture_overrides_the_token_estimate():
    est = ledger(1766.0).term(TERM_GRAPH_CAPTURE)
    meas = ledger(1766.0, capture_mib=[640.0]).term(TERM_GRAPH_CAPTURE)
    assert est.mib == 192  # 96 captured tokens x 2 MiB
    assert meas.mib == 640
    assert meas.provenance is Provenance.CALIBRATED
    assert meas.mib > est.mib * 3, "the window measured the estimate 3.3-3.8x low"


def test_the_token_estimate_declares_that_it_is_known_low():
    """While the estimate is still reachable it must say so -- an
    under-charge that does not announce itself is how a boot OOMs."""
    term = ledger(1766.0).term(TERM_GRAPH_CAPTURE)
    assert "KNOWN LOW" in term.derivation


# --- the profile key --------------------------------------------------------


def test_profile_key_separates_configs_that_change_the_footprint():
    base = profile_key(REF_PROFILE)
    import dataclasses as dc

    assert profile_key(dc.replace(REF_PROFILE, chunked_prefill_size=4096)) != base
    assert profile_key(dc.replace(REF_PROFILE, tp_size=2)) != base
    assert profile_key(dc.replace(REF_PROFILE, decode_max_bs=8)) != base
    assert profile_key(dc.replace(REF_PROFILE, kv_cache_dtype="auto")) != base
    # ...and is stable for the same inputs
    assert profile_key(dc.replace(REF_PROFILE)) == base


def test_reference_bounds_apply_only_to_the_rig_they_were_measured_on():
    hit = resolve_phase_footprint(
        BINDING_3080,
        hw_fingerprint=REFERENCE_WINDOW_FINGERPRINT,
        profile=REF_PROFILE,
        cache_dir="/nonexistent",
    )
    assert hit is not None
    assert hit.activation_mib == 1766
    assert hit.provenance is FootprintProvenance.UPPER_BOUND

    other_rig = resolve_phase_footprint(
        BINDING_3080,
        hw_fingerprint="someotherrig",
        profile=REF_PROFILE,
        cache_dir="/nonexistent",
    )
    assert other_rig is None, "a bound must not leak onto hardware it never saw"


def test_reference_bounds_do_not_apply_to_a_different_profile():
    import dataclasses as dc

    hit = resolve_phase_footprint(
        BINDING_3080,
        hw_fingerprint=REFERENCE_WINDOW_FINGERPRINT,
        profile=dc.replace(REF_PROFILE, chunked_prefill_size=8192),
        cache_dir="/nonexistent",
    )
    assert hit is None


def test_all_three_reference_cards_are_recorded():
    fps = reference_window_footprints()
    assert len(fps) == 3
    assert fps[BINDING_3080].activation_mib == 1766
    for f in fps.values():
        assert f.capture_mib >= 633
        assert f.provenance is FootprintProvenance.UPPER_BOUND


# --- cache round trip -------------------------------------------------------


def test_measured_peak_beats_the_shipped_bound(tmp_path):
    """Precedence: a real probe result must win over the shipped upper bound."""
    save_footprints(
        {
            BINDING_3080: PhaseFootprint(
                activation_mib=900,
                capture_mib=655,
                provenance=FootprintProvenance.MEASURED_PEAK,
                source="probe_activation.py torch peak counters",
                card_uuid=BINDING_3080,
            )
        },
        hw_fingerprint=REFERENCE_WINDOW_FINGERPRINT,
        profile=REF_PROFILE,
        cache_dir=str(tmp_path),
    )
    hit = resolve_phase_footprint(
        BINDING_3080,
        hw_fingerprint=REFERENCE_WINDOW_FINGERPRINT,
        profile=REF_PROFILE,
        cache_dir=str(tmp_path),
    )
    assert hit.provenance is FootprintProvenance.MEASURED_PEAK
    assert hit.activation_mib == 900


def test_cache_is_keyed_by_profile_not_only_hardware(tmp_path):
    import dataclasses as dc

    save_footprints(
        {
            BINDING_3080: PhaseFootprint(
                activation_mib=900,
                capture_mib=655,
                provenance=FootprintProvenance.MEASURED_PEAK,
                source="probe",
                card_uuid=BINDING_3080,
            )
        },
        hw_fingerprint=REFERENCE_WINDOW_FINGERPRINT,
        profile=REF_PROFILE,
        cache_dir=str(tmp_path),
    )
    other = load_footprints(
        hw_fingerprint=REFERENCE_WINDOW_FINGERPRINT,
        profile=dc.replace(REF_PROFILE, chunked_prefill_size=4096),
        cache_dir=str(tmp_path),
    )
    assert other == {}
