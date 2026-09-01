"""#631/#968: can-fail proof for the producer-phase instrument.

An instrument that cannot go RED when the thing did not happen, and GREEN when
it did, is not an instrument. Both directions are proven here, plus the third
state (NO_OBSERVATION) that this strand has already lost a reading to.

Hermetic: no GPU, no boot, no sglang runtime. The module's two runtime lookups
(`current_generation`, `bound_phase`) fail closed to "-" without sglang, and
the arming knob returns 0, so every path below is exercised with explicit
state instead of ambient state.
"""

import logging

import pytest
from sglang.srt.mem_cache.producer_phase_census import (
    AdoptionSource,
    ObservationState,
    ProducerPhase,
    ProducerPhaseCensus,
    arrival_stats,
    ledger_stats,
    note_arrival,
    note_consult,
    note_generation,
    note_store_write,
    payload_verdict,
    phase_of_generation,
    producer_phase_of,
    reset_for_test,
)


@pytest.fixture(autouse=True)
def _clean():
    reset_for_test()
    yield
    reset_for_test()


def _flip_happened():
    """The mission's shape: generation 1 was PP, generation 2 is TP."""
    note_generation(1, "pp")
    note_generation(2, "tp")


# -- the three states ----------------------------------------------------


def test_unfed_census_is_no_observation_not_zero():
    c = ProducerPhaseCensus()
    assert c.state() is ObservationState.NO_OBSERVATION
    assert c.log_fields()["state"] == "NO_OBSERVATION"


def test_fed_but_empty_population_is_empty_not_no_observation():
    c = ProducerPhaseCensus()
    c.observed = True  # armed and touched, but no walk completed
    assert c.state() is ObservationState.EMPTY


def test_counted_walks_are_value():
    c = ProducerPhaseCensus()
    c.note_walk(hit=False)
    assert c.state() is ObservationState.VALUE


# -- CAN-FAIL: RED when no PP3-produced hit occurred ---------------------


def test_red_when_every_hit_is_same_phase():
    """Same-phase host resume is the EVICTION axis. It must not read as ok."""
    _flip_happened()
    note_store_write("page-a", 2)  # written by TP itself

    c = ProducerPhaseCensus()
    for _ in range(40):
        c.note_walk(hit=False)
    for _ in range(12):
        c.note_walk(hit=True)
    assert producer_phase_of("page-a", 2) is ProducerPhase.SAME_PHASE
    c.note_accepted_tokens(49152, ProducerPhase.SAME_PHASE, AdoptionSource.BACKUP_HOST)

    f = c.log_fields()
    assert f["ok"] == 0, "same-phase hits must not be counted as the mission"
    assert f["denom"] == 52, "the denominator must ride the same line"
    assert f["hit_walks"] == 12, "hits happened; they were just the wrong kind"
    assert f["cross_tokens"] == 0


def test_red_survives_the_false_win_specimen():
    """The 2026-09-01 near-miss: a fat cached-token count inside TP.

    In-phase chunked continuation produces exactly this. Without the producer
    axis it is indistinguishable from the mission succeeding; with it, ok=0.
    """
    _flip_happened()
    note_store_write("chunk", 2)
    c = ProducerPhaseCensus()
    c.note_walk(hit=True)
    c.note_accepted_tokens(
        16384, producer_phase_of("chunk", 2), AdoptionSource.BACKUP_HOST
    )
    assert c.log_fields()["ok"] == 0


# -- CAN-FAIL: GREEN when a PP3-produced hit did occur -------------------


def test_green_when_a_cross_phase_hit_happened():
    _flip_happened()
    note_store_write("page-pp", 1)  # produced under the PP binding

    c = ProducerPhaseCensus()
    for _ in range(40):
        c.note_walk(hit=False)
    for _ in range(12):
        c.note_walk(hit=True)

    assert producer_phase_of("page-pp", 2) is ProducerPhase.CROSS_PHASE
    for _ in range(7):
        c.note_cross_phase_walk()
    c.note_accepted_tokens(57344, ProducerPhase.CROSS_PHASE, AdoptionSource.PREFETCH)
    c.note_accepted_tokens(4096, ProducerPhase.SAME_PHASE, AdoptionSource.BACKUP_HOST)

    f = c.log_fields()
    assert f["ok"] == 7
    assert f["denom"] == 52
    assert f["cross_tokens"] == 57344
    assert "cross:57344" in f["by_producer"]
    assert "prefetch:57344" in f["by_source"]


# -- provenance must never be guessed ------------------------------------


def test_unstamped_key_is_unknown_never_same():
    _flip_happened()
    assert producer_phase_of("never-written", 2) is ProducerPhase.UNKNOWN


def test_unnamed_generation_is_unknown_never_cross():
    """A different generation we cannot name is weaker than CROSS_PHASE."""
    note_store_write("k", 1)  # generation 1 never recorded a phase
    note_generation(2, "tp")
    assert phase_of_generation(1) is None
    assert producer_phase_of("k", 2) is ProducerPhase.UNKNOWN


def test_rewrite_attributes_to_the_later_phase():
    """A page rewritten in TP IS produced by TP. Over-reporting is worse."""
    _flip_happened()
    note_store_write("k", 1)
    note_store_write("k", 2)
    assert producer_phase_of("k", 2) is ProducerPhase.SAME_PHASE


# -- ordering: "never came" vs "came too late" ---------------------------


def test_late_arrival_is_its_own_term():
    note_consult("k", accepted=False)  # the walk looked, and missed
    note_arrival("k")  # the fetch landed afterwards
    s = arrival_stats()
    assert s["late"] == 1, "a fetch that landed after the walk is LATE"
    assert s["never_yet"] == 0


def test_never_arrived_is_distinct_from_late():
    note_consult("k", accepted=False)
    s = arrival_stats()
    assert s["late"] == 0
    assert s["never_yet"] == 1, "missed with no arrival is a different fix"


def test_a_hit_consult_is_not_a_miss():
    note_consult("k", accepted=True)
    note_arrival("k")
    assert arrival_stats()["late"] == 0


# -- the partition must be checkable -------------------------------------


def test_broken_partition_raises_rather_than_reporting():
    c = ProducerPhaseCensus()
    c.note_walk(hit=True)
    c.cross_phase_walks = 5  # more cross-phase than hits: impossible
    with pytest.raises(ValueError):
        c.log_fields()


def test_numerator_cannot_exceed_denominator():
    c = ProducerPhaseCensus()
    c.observed = True
    c.walks = 1
    c.hit_walks = 3
    with pytest.raises(ValueError):
        c.check_partition()


# -- a success label must measure its payload ----------------------------


def test_the_measured_17_of_18_are_not_successes():
    """THE specimen, from the FULL count (not the 4-line sample that said 18/18).

    17 of 18 'HiCache prefetch success' lines carry loaded=0 despite
    matched>0: the store had the page and nothing came across. That is a
    named state, and it is not success.
    """
    assert (
        payload_verdict(
            completed_local=8192, completed_synced=8192, matched=8192, loaded=0
        )
        == "matched_not_loaded"
    )


def test_only_arrived_bytes_are_delivery():
    """The one line of eighteen that actually loaded."""
    assert (
        payload_verdict(completed_local=4096, matched=4096, loaded=4096) == "delivered"
    )


def test_matched_not_loaded_is_not_reabsorbed_by_refused_or_empty():
    """A partial refusal must not hide the 17/18 state."""
    assert payload_verdict(matched=8192, loaded=0, refused=3) == "matched_not_loaded"
    assert (
        payload_verdict(completed_local=8192, matched=8192, loaded=0)
        == "matched_not_loaded"
    )


def test_payload_verdict_separates_the_remaining_worlds():
    assert payload_verdict(completed_local=1, refused=3) == "refused"
    assert payload_verdict(completed_local=1) == "empty"
    assert payload_verdict() == "no_completion", "the 7/18 genuinely-empty lines"


# -- the ledger must not forget silently ---------------------------------


def test_dropped_ledger_entries_are_counted():
    from sglang.srt.mem_cache import producer_phase_census as m

    original = m._LEDGER_MAX
    m._LEDGER_MAX = 2
    try:
        for i in range(5):
            note_store_write(f"k{i}", 1)
        s = ledger_stats()
        assert s["dropped"] == 3, "a forgotten key must leave a trace"
        assert s["keys"] == 2
    finally:
        m._LEDGER_MAX = original


# -- emission -------------------------------------------------------------


def test_cross_phase_window_is_never_sampled_away(monkeypatch, caplog):
    from sglang.srt.mem_cache import producer_phase_census as m

    monkeypatch.setattr(m, "census_armed", lambda: 1000)
    c = ProducerPhaseCensus()
    c.note_walk(hit=True)
    c.note_cross_phase_walk()
    c.note_accepted_tokens(64, ProducerPhase.CROSS_PHASE, AdoptionSource.PREFETCH)
    logger = logging.getLogger("t631")
    with caplog.at_level(logging.INFO, logger="t631"):
        m.emit(c, logger)
    assert any("#631 producer-phase" in r.getMessage() for r in caplog.records)


def test_disarmed_emits_nothing(caplog):
    from sglang.srt.mem_cache import producer_phase_census as m

    c = ProducerPhaseCensus()
    c.note_walk(hit=True)
    logger = logging.getLogger("t631b")
    with caplog.at_level(logging.INFO, logger="t631b"):
        m.emit(c, logger)
    assert not caplog.records, "disarmed must be silent, not zero-valued"
