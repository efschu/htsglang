"""#1269: the host watermark is a hard bound with a named margin.

User order 2026-09-08, verbatim, given AFTER the sb4 host OOM: "kein
uebertreten mehr der schwelle. fuehrt nur zum absturz."  The base weg2sb4
stood green while idling at 95.92 -> 96.36 -> 96.47 GiB against a 95.90 GiB
reap mark, growth entirely anon, and went into the OOM.  "Accept the risk"
had been offered and taken.  It is not an option at this threshold any more.

Two halves are pinned here: the BOOT refusal (an arm whose peak does not fit
under watermark - margin is refused by name, and the store shrinks rather
than the margin) and the RUNTIME breach (a controlled teardown verdict).
"""

import pytest

from sglang.srt.weg2.host_ledger import (
    FLIP_HOST_TRANSIENT_GIB,
    GIB,
    RESIDUAL_WINDOW_MIN,
    RING_ERA_FLIP_TRANSIENT_GIB,
    RUN_PEAK_RESIDUAL_GIB,
    measure_foreign_anon,
    OBSERVED_REAP_NONRECLAIM_BYTES,
    REAP_SAMPLE_EXCLUDED,
    REAP_SAMPLES_GIB,
    RING_GRANULE_GIB,
    Margin,
    resolve_margin,
    size_store_gib,
    watermark_breach_verdict,
    watermark_provenance,
)

WATERMARK = OBSERVED_REAP_NONRECLAIM_BYTES / GIB


# ------------------------------------------------------------------ margin
def test_margin_is_named_terms_not_a_hand_number():
    m = resolve_margin()
    assert m.transient_gib > 0 and m.residual_gib > 0 and m.drift_gib > 0
    assert m.total_gib == pytest.approx(
        m.transient_gib + m.residual_gib + m.drift_gib + m.foreign_gib
    )
    for t in ("transient", "residual", "drift", "foreign"):
        assert t in m.terms()
    assert (
        m.transient_source and m.residual_source and m.drift_source and m.foreign_source
    )


def test_drift_is_charged_only_beyond_the_residuals_own_window():
    """CORRECTION: the residual is (measured - predicted) over a ~60 min window,
    so it ALREADY contains that window's drift. Charging the full 90 min on top
    would double-count the very thing the residual measured."""
    m = resolve_margin(drift_mib_per_min=19.0, window_min=90.0)
    assert m.drift_gib == pytest.approx(19.0 * (90.0 - RESIDUAL_WINDOW_MIN) / 1024.0)
    m_short = resolve_margin(drift_mib_per_min=19.0, window_min=RESIDUAL_WINDOW_MIN)
    assert m_short.drift_gib == 0.0, (
        "inside the residual's window, drift is not re-added"
    )


def test_the_transient_is_ring_era_not_the_pre_ring_constant():
    """CORRECTION: 9.97 GiB is the PRE-RING per-allocation transient. With the
    shared registered ring it is 1.1 (rg2) / 2.88 (rg3); the max binds."""
    m = resolve_margin()
    assert m.transient_gib == pytest.approx(max(RING_ERA_FLIP_TRANSIENT_GIB.values()))
    assert m.transient_gib < FLIP_HOST_TRANSIENT_GIB
    assert "RING-ERA" in m.transient_source and "weg2rg3" in m.transient_source


def test_the_pre_ring_value_is_used_only_when_no_ring_sample_exists(monkeypatch):
    monkeypatch.setattr(
        "sglang.srt.weg2.host_ledger.RING_ERA_FLIP_TRANSIENT_GIB", {}, raising=True
    )
    m = resolve_margin()
    assert m.transient_gib == pytest.approx(FLIP_HOST_TRANSIENT_GIB)
    assert "PRE-RING fallback" in m.transient_source


def test_the_residual_is_the_ring_era_under_prediction():
    """THE REAL GAP: the estimator misses the steady state, not the flip."""
    m = resolve_margin()
    assert m.residual_gib == pytest.approx(max(RUN_PEAK_RESIDUAL_GIB.values()))
    assert "weg2sb4" in m.residual_source
    assert m.residual_gib > m.transient_gib, (
        "the residual, not the flip, is the big term"
    )


# ------------------------------------------------------------- foreign load
def test_foreign_anon_is_the_cgroup_minus_the_boots_own_pids():
    import os

    foreign, sglang, src = measure_foreign_anon(None, [os.getpid()])
    assert foreign is None and sglang > 0 and "unreadable" in src
    big = int((sglang + 7.0) * GIB)
    foreign, sglang2, src = measure_foreign_anon(big, [os.getpid()])
    assert foreign == pytest.approx(7.0, abs=0.05)
    assert "cgroup anon" in src and "sglang RssAnon" in src


def test_foreign_is_not_re_added_to_the_margin_by_default():
    """It is ALREADY in the origin (run_origin_gib returns a cgroup reading,
    which counts every process in the cgroup, Claude included)."""
    m = resolve_margin()
    assert m.foreign_gib == 0.0
    assert "already in the origin" in m.foreign_source


def test_the_breach_verdict_names_who_caused_it():
    import os

    m = resolve_margin()
    over = int((WATERMARK - m.total_gib + 0.5) * GIB)
    own = measure_foreign_anon(0, [os.getpid()])[1]
    v = watermark_breach_verdict(
        over,
        margin=m,
        cgroup_anon_bytes=int((own + 40.0) * GIB),
        own_pids=[os.getpid()],
    )
    assert "SPLIT sglang=" in v and "foreign=" in v
    assert "FOREIGN desk load dominates" in v


def test_a_measured_census_rate_supersedes_the_sb4_default():
    default = resolve_margin()
    measured = resolve_margin(drift_mib_per_min=0.4)
    assert "sb4 default" in default.drift_source
    assert "WEG2-IDLE-CENSUS" in measured.drift_source
    assert measured.total_gib < default.total_gib


def test_the_transient_term_is_floored_at_one_ring_granule():
    """The margin may never collapse to zero when a record is thin."""
    m = resolve_margin(flip_transient_gib=0.0)
    assert m.transient_gib == pytest.approx(RING_GRANULE_GIB)
    assert "granule" in m.transient_source


# -------------------------------------------------------------- provenance
def test_only_kernel_reaps_count_as_watermark_samples():
    assert set(REAP_SAMPLES_GIB) == {"weg2dk5", "weg2dk6"}
    assert "weg2sb4" in REAP_SAMPLE_EXCLUDED
    _, why = REAP_SAMPLE_EXCLUDED["weg2sb4"]
    assert "no kernel OOM" in why


def test_provenance_line_names_watermark_source_margin_and_bound():
    line = watermark_provenance()
    assert line.startswith("WEG2-HOST WATERMARK=")
    for token in ("source=[", "margin=", "hard bound", "excluded=["):
        assert token in line, line
    assert "weg2dk5" in line and "weg2dk6" in line
    assert "weg2sb4" in line  # named as EXCLUDED, never silently dropped


# ------------------------------------------------------------ boot refusal
def test_a_peak_that_crosses_the_line_leaves_no_store():
    """sb4's own chosen arm, from its boot log: predicted run peak 91.44 GiB
    with store 11, leftover 11.61, unsampled 0.58.  It reported FUNDABLE
    against the raw 95.90 mark and was 4.6-5.2 GiB over it on the metal."""
    m = resolve_margin()
    peak_without_store = 91.44 - 11.0
    sz = size_store_gib(11.61, peak_without_store, 0.58, margin_gib=m.total_gib)
    assert sz.reap_bound_gib < 8.0, "the 8 GiB store floor must not be reachable"
    assert sz.gib < 8.0
    assert 91.44 > WATERMARK - m.total_gib, "sb4's arm must not fit the hard bound"


def test_the_store_shrinks_and_the_margin_never_does():
    m = resolve_margin()
    wide = size_store_gib(40.0, 60.0, 0.0, margin_gib=0.0)
    tight = size_store_gib(40.0, 60.0, 0.0, margin_gib=m.total_gib)
    assert tight.gib == pytest.approx(wide.gib - m.total_gib, abs=1.0)
    assert tight.gib < wide.gib


def test_bound_leftover_may_never_exceed_watermark_minus_margin():
    """A small leftover must still be capped by the hard bound, not waved
    through because 'leftover' happened to be the smaller of the two."""
    m = resolve_margin()
    peak_without_store = WATERMARK - m.total_gib - 3.0  # only 3 GiB of room left
    sz = size_store_gib(999.0, peak_without_store, 0.0, margin_gib=m.total_gib)
    assert sz.bound == "reap"
    assert sz.gib <= 3.0
    assert sz.gib + peak_without_store <= WATERMARK - m.total_gib + 1e-6


def test_a_store_that_fits_under_the_hard_bound_is_accepted():
    m = resolve_margin()
    peak_without_store = 60.0
    sz = size_store_gib(20.0, peak_without_store, 0.0, margin_gib=m.total_gib)
    assert sz.gib >= 8.0, "this arm has room for a store above the floor"
    assert peak_without_store + sz.gib <= WATERMARK - m.total_gib + 1e-6


# --------------------------------------------------------- runtime breach
def test_no_verdict_while_below_the_hard_bound():
    m = resolve_margin()
    safe = int((WATERMARK - m.total_gib - 1.0) * GIB)
    assert watermark_breach_verdict(safe, margin=m) is None


def test_breach_issues_the_named_teardown_verdict():
    m = resolve_margin()
    over = int((WATERMARK - m.total_gib + 0.5) * GIB)
    v = watermark_breach_verdict(over, margin=m)
    assert v is not None
    assert v.startswith("W22 Weg2HostWatermarkBreached")
    for token in ("current=", "watermark=", "margin="):
        assert token in v, v
    assert "CONTROLLED TEARDOWN" in v
    assert "accept the risk" in v  # the offer that is now refused by name


def test_sb4_idle_reading_would_have_torn_down():
    """96.60 GiB: what the operator saw before killing sglang by hand."""
    v = watermark_breach_verdict(int(96.60 * GIB))
    assert v is not None and "96.60" in v


def test_the_verdict_is_pure_and_needs_no_processes():
    """Synthetic cgroup reading only -- no /proc, no cgroup, no front."""
    m = Margin(
        transient_gib=1.0,
        residual_gib=0.5,
        drift_gib=0.5,
        foreign_gib=0.0,
        drift_mib_per_min=1.0,
        window_min=1024.0,
        transient_source="t",
        residual_source="r",
        drift_source="d",
        foreign_source="f",
    )
    assert (
        watermark_breach_verdict(int(10 * GIB), margin=m, watermark_gib=100.0) is None
    )
    assert watermark_breach_verdict(int(99 * GIB), margin=m, watermark_gib=100.0)


def test_the_pre_order_ledger_would_have_funded_sb4_and_this_one_does_not():
    """THE REGRESSION THIS COMMIT EXISTS FOR, both verdicts side by side.

    sb4's chosen arm (S=1 M=2400, store 11 GiB) predicted a 91.44 GiB run peak.
    With NO margin -- the ledger as it stood -- 91.44 <= 95.90 and the boot was
    printed FUNDABLE; it then idled 4.6-5.2 GiB above its own prediction and
    the box OOMed.  With the named margin the same arm is refused.
    """
    predicted, store, leftover, unsampled = 91.44, 11.0, 11.61, 0.58
    peak_without_store = predicted - store

    # pre-order: margin 0
    old = size_store_gib(leftover, peak_without_store, unsampled, margin_gib=0.0)
    assert predicted <= WATERMARK, "the raw watermark passed this arm"
    assert old.gib >= 8.0, "and left a store above the floor -- FUNDABLE"

    # with the order's margin
    m = resolve_margin()
    new = size_store_gib(
        leftover, peak_without_store, unsampled, margin_gib=m.total_gib
    )
    assert predicted > WATERMARK - m.total_gib
    assert new.gib < 8.0, "no store fits under the hard bound -> W21"

    shortfall = predicted - (WATERMARK - m.total_gib)
    assert 4.0 < shortfall < 4.3, f"sb4 is over the hard bound by {shortfall:.2f} GiB"


def test_the_ring_form_DOES_fit_but_not_on_the_arm_sb4_chose():
    """THE ANSWER, pinned. With the corrected terms the binding constraint is
    NOT the host ring -- it is the mamba anchor pool M. sb4's three arms, from
    its own boot log (front.log 16:29:38Z):

        S=1 M=2400  peak 91.44 store 11  anchors 9.49 GiB
        S=1 M=1200  peak 91.50 store 16  anchors 4.75 GiB
        S=1 M=600   peak 92.04 store 19  anchors 2.37 GiB

    The ring (Sigma H 32.19-34.94 GiB) is inside ALL THREE peaks, so it cannot
    be what separates them. The ledger picked M=2400 because the pre-margin
    bound passed it; with the margin it is refused and the ladder falls to an
    arm that fits with room.
    """
    m = resolve_margin()
    hard = WATERMARK - m.total_gib
    arms = {
        "M=2400": (91.44, 11.0, 11.61),
        "M=1200": (91.50, 16.0, 16.54),
        "M=600": (92.04, 19.0, 19.01),
    }
    stores = {
        name: size_store_gib(left, peak - store, 0.58, margin_gib=m.total_gib).gib
        for name, (peak, store, left) in arms.items()
    }
    assert stores["M=2400"] < 8.0, "the arm sb4 actually chose is refused"
    assert stores["M=1200"] >= 8.0, "a smaller anchor pool fits with a real store"
    assert stores["M=600"] >= 8.0
    assert max(stores.values()) >= 8.0, "SOME store >= the floor fits the ring form"
    # and the peak without any store is under the bound on every arm, so the
    # arm itself is never the impossible part -- the store size is.
    for name, (peak, store, _) in arms.items():
        assert peak - store < hard, f"{name}: peak without store must fit"
