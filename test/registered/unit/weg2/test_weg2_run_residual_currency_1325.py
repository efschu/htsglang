"""#1325 -- the run residual was measured in the WRONG CURRENCY.

THE DEFECT. ``dormant_image_sample`` derives ``run_residual_gib`` as
``nonreclaimable - this boot's own charges - the measured image``, i.e. "what
the box holds that this ledger's term list does NOT name". It called
``charge_terms`` WITHOUT ``s_gb_d``, so on a two-budget boot it subtracted the
rings of an ``S_D == S`` arm while the boot actually held ``S_D=4`` of them.
The unsubtracted difference landed in the residual -- the one quantity whose
whole definition is "not named by the term list" -- and the ledger then adds
those same rings back at prediction time. Charged twice.

MEASURED with ``charge_terms`` itself, sn6s's arm (S=1 M=600 S_D=4, ranks=3):
boot charges 24.5047 GiB without ``s_gb_d`` against 33.2219 with it, i.e.
**8.7172 GiB under-subtracted**. sn6s recorded ``run_residual_gib`` 16.34 for
group P; corrected it is **7.62**, which is BELOW sn6p's idle 9.01, so
``run_origin_gib``'s ``max()`` keeps 9.01 and the arm prices exactly as it did
when acceptance (g) passed. The uncorrected 16.34 is what predicts
``run_peak 92.53`` against the 87.30 boot bound and fires ``W21
Weg2HostRunPeakRefused`` on every dry run of this form.

WHY THIS IS THE ROOT AND THE LOAD CLASS IS NOT. The record WAS sampled under
load (15:21:55Z, mid-120k-load) and that is worth recording -- but the ~7.3 GiB
gap between the loaded 16.34 and the idle 9.01 is almost entirely the 8.72 GiB
of unsubtracted D rings, not a load working set. A margin rule that dropped the
flip transient (2.88) for loaded records would have closed 2.88 of a 7.33 GiB
gap and left the boot refused, while leaving the double charge in place. So the
load class and the form key are RECORDED here and select nothing yet: pricing a
corrected sample by a rule designed for the uncorrected one is how a
compensation layer starts.

DANGER DIRECTION: this correction makes residuals SMALLER, i.e. it can fund an
arm that was refused. Every mutant must therefore make the suite red by
under-charging the box, never by moving a decimal:

  M1  the subtraction must carry S_D -> test_the_subtraction_carries_s_d
  M2  a single-budget arm must be byte-identical
      -> test_a_single_budget_arm_is_unchanged
  M3  the sn6s record must fund the arm that held on metal
      -> test_the_sn6s_record_no_longer_exceeds_the_boot_bound
  M4  "a quieter launch does not buy a bigger arm" must survive
      -> test_a_quiet_launch_still_charges_the_measured_floor
  M5  an absent load witness must never read as idle
      -> test_an_absent_load_witness_is_unknown_not_idle
"""

from sglang.srt.weg2 import host_ledger as hl

# sn6s, verbatim from BOOT_weg2sn6s_0910.md (g) and the XSN6 order.
SN6S_ARM = {"s_gb": 1, "m_mib": 600, "s_gb_d": 4}
SN6S_RECORDED_RESIDUAL = 16.34
SN6P_IDLE_RESIDUAL = 9.01
SN6S_IMAGE_GIB = 38.63
SN6S_WTAGS_GIB = 28.83
RANKS = 3


def _images(gib=SN6S_IMAGE_GIB):
    return hl.ImageTerms(
        p_gib=gib, d_gib=gib, p_source="sample", d_source="sample",
        p_measured=True, d_measured=True, extra_p_gib=0.0, extra_d_gib=0.0,
    )


def _sample(arm=None, witness=None, nonreclaim_gib=50.0, wtags=SN6S_WTAGS_GIB):
    return hl.dormant_image_sample(
        group="P", shmem_before_bytes=0, shmem_after_bytes=0, pids=[],
        weight_tags_gib=wtags, interleaved=True, boot_tag="t", commit="c",
        cg_current_bytes=int(nonreclaim_gib * hl.GIB), reclaimable_bytes=0,
        arm=SN6S_ARM if arm is None else arm, ranks_per_group=RANKS,
        load_witness=witness,
    )


# --------------------------------------------------------------------------
# M1 / M2: the currency
# --------------------------------------------------------------------------


def test_the_subtraction_carries_s_d():
    """M1: the gap is exactly the rings the old call did not subtract.

    Stated as a DELTA on purpose. The rings term depends only on the budgets,
    never on the measured image, so the correction moves every residual of this
    arm by the same 8.7172 GiB whatever the image was -- which is what lets the
    sn6s pin below be checked against the boot's own recorded number without
    reconstructing its image.
    """
    img = _images()
    without = hl._boot_charges_gib(hl.charge_terms(1, 600, RANKS, img))
    with_sd = hl._boot_charges_gib(
        hl.charge_terms(1, 600, RANKS, img, s_gb_d=4)
    )
    assert round(with_sd - without, 4) == 8.7172, (without, with_sd)
    # Image-independence of that delta, since the pin relies on it.
    z = _images(0.0)
    assert round(
        hl._boot_charges_gib(hl.charge_terms(1, 600, RANKS, z, s_gb_d=4))
        - hl._boot_charges_gib(hl.charge_terms(1, 600, RANKS, z)), 4
    ) == 8.7172

    # And the SAMPLE takes the WITH form: same box, two arms, 8.7172 apart.
    loaded = float(_sample()["run_residual_gib"])
    single = float(_sample(arm={"s_gb": 1, "m_mib": 600})["run_residual_gib"])
    assert abs((single - loaded) - 8.7172) < 0.01, (
        "the sample must subtract the rings the boot actually held"
    )
    assert "S_D=4" in str(_sample()["run_residual_note"])
    assert "#1325" in str(_sample()["run_residual_note"])


def test_a_single_budget_arm_is_unchanged():
    """M2: a pre-#1325 arm dict (no s_gb_d) keeps its old subtraction.

    `charge_terms` defaults `_s_d` to `s_gb`, so the corrected call is
    byte-identical for a single-budget boot -- which is every boot before D got
    its own budget, and every record already on disk. Checked against the
    sample's OWN arithmetic (`pids=[]` measures a zero image, so the image term
    is 0 here) rather than a hand-rolled formula.
    """
    expect = hl._boot_charges_gib(hl.charge_terms(1, 600, RANKS, _images(0.0)))
    r = _sample(arm={"s_gb": 1, "m_mib": 600})
    assert abs(float(r["run_residual_gib"]) - (50.0 - expect)) < 1e-6
    # Explicit S_D == S must agree with omitting it entirely.
    r2 = _sample(arm={"s_gb": 1, "m_mib": 600, "s_gb_d": 1})
    assert abs(float(r2["run_residual_gib"]) - float(r["run_residual_gib"])) < 1e-6


def test_the_sn6s_record_no_longer_exceeds_the_boot_bound():
    """M3: THE PIN THE OPERATOR ASKED FOR.

    The sn6s record must fund the same arm that held 88.48 GiB non-reclaimable
    on metal against the 92.46 runtime bound. Applied to sn6s's OWN recorded
    16.34 GiB via the image-independent delta pinned above -- the input is the
    boot's number, not a desk-chosen one.
    """
    corrected = SN6S_RECORDED_RESIDUAL - 8.7172
    assert abs(corrected - 7.62) < 0.02, corrected
    # A record already in its own currency, so the reader re-prices it by 0
    # and this test asserts the ORDERING and not the correction twice.
    r = dict(_sample(), arm={"s_gb": 1, "m_mib": 600},
             run_residual_gib=corrected)
    assert corrected < SN6P_IDLE_RESIDUAL, (
        "corrected, the loaded sample sits BELOW sn6p's idle floor, so "
        "run_origin_gib's max() keeps the idle floor and the arm prices as it "
        "did when (g) passed -- which is why no margin-currency change is "
        "needed to clear the W21"
    )
    # The origin picks the LARGER of the two records, i.e. the idle floor.
    idle = dict(r)
    idle["run_residual_gib"] = SN6P_IDLE_RESIDUAL
    idle["boot_tag"] = "weg2sn6p"
    idle["load_class"] = "idle"
    origin, src = hl.run_origin_gib(5.0, {"P": r, "D": idle})
    assert abs(origin - SN6P_IDLE_RESIDUAL) < 1e-6
    assert "weg2sn6p" in src and "load_class=idle" in src


def test_a_quiet_launch_still_charges_the_measured_floor():
    """M4: the fix must not reopen the dk6/dk7 hole.

    "A quieter launch does not buy a bigger arm" -- a launch reading below the
    measured floor still charges the floor. The correction lowers the floor; it
    must not turn it back into the launch reading.
    """
    r = _sample(nonreclaim_gib=60.0)
    # The RE-PRICED floor is what the origin compares against, so the
    # expectation is taken from the same reader the origin uses -- not from
    # the stored figure (the mistake the XSN6 record calls out: two ledger
    # numbers are comparable only when their inputs are).
    floor, _corr = hl.record_run_residual_gib(r)
    assert floor > 0
    origin, src = hl.run_origin_gib(floor - 3.0, {"P": r})
    assert abs(origin - floor) < 1e-6, "the floor binds, not the quiet launch"
    assert "quieter launch does not buy a bigger arm" in src
    # And a launch AT OR ABOVE the floor charges the launch reading.
    origin2, src2 = hl.run_origin_gib(floor + 3.0, {"P": r})
    assert abs(origin2 - (floor + 3.0)) < 1e-6
    assert "AT OR ABOVE" in src2


# --------------------------------------------------------------------------
# M5: the recorded load class and form key
# --------------------------------------------------------------------------


def test_a_loaded_sample_is_stamped_loaded():
    """The sn6s sampling moment, as a measurement rather than a date."""
    r = _sample(witness={"queued": 2, "outstanding": 1})
    assert r["load_class"] == "loaded"
    assert r["load_witness"] == {"queued": 2, "outstanding": 1}


def test_an_idle_sample_is_stamped_idle():
    r = _sample(witness={"queued": 0, "outstanding": 0})
    assert r["load_class"] == "idle"


def test_an_absent_load_witness_is_unknown_not_idle():
    """M5: an absent witness is not a quiet box.

    The #944 rule again: "I measured nothing in flight" and "I did not look"
    are different facts. Reading the second as `idle` would let a pre-#1325
    record, which carries no witness at all, claim a quiet box.
    """
    assert _sample(witness=None)["load_class"] == "unknown"
    assert _sample(witness={})["load_class"] == "unknown"


def test_the_form_key_is_independent_of_the_arm():
    """The key must name the host posten, not the budget being priced.

    Keying record selection on the arm would be circular -- the arm is what the
    record is used to price -- and the residual is already arm-normalised by
    the subtraction. Shadow/exchange forms change neither term, so they share a
    key, which is what the operator's reading of "same serving form" requires.
    """
    a = _sample(arm={"s_gb": 1, "m_mib": 600, "s_gb_d": 4})
    b = _sample(arm={"s_gb": 2, "m_mib": 2400, "s_gb_d": 1})
    assert a["form_key"] == b["form_key"] == f"ranks={RANKS};wtags=28.83"
    # ... and it DOES separate a different rank layout or a different image.
    assert _sample(wtags=12.5)["form_key"] != a["form_key"]
    c = hl.dormant_image_sample(
        group="P", shmem_before_bytes=0, shmem_after_bytes=0, pids=[],
        weight_tags_gib=SN6S_WTAGS_GIB, interleaved=True, boot_tag="t",
        commit="c", cg_current_bytes=int(50 * hl.GIB), reclaimable_bytes=0,
        arm=SN6S_ARM, ranks_per_group=2,
    )
    assert c["form_key"] != a["form_key"]


def test_the_arm_line_carries_both_currencies():
    """The count-check the operator asked for, on the origin's own source.

    A boot whose origin came from a sample taken under load must SAY so where
    the arm is printed, or the next reader cannot tell an idle floor from a
    load peak. A pre-#1325 record answers `?`, which is a fact about the record
    rather than a claim about the box.
    """
    loaded = _sample(witness={"queued": 5, "outstanding": 2})
    _, src = hl.run_origin_gib(0.0, {"P": loaded})
    assert "load_class=loaded" in src
    assert f"form=ranks={RANKS};wtags=28.83" in src
    legacy = {k: v for k, v in loaded.items()
              if k not in ("load_class", "form_key")}
    _, src2 = hl.run_origin_gib(0.0, {"P": legacy})
    assert "load_class=?" in src2 and "form=?" in src2


# --------------------------------------------------------------------------
# THE READER: the record that blocks the next boot is already on disk
# --------------------------------------------------------------------------

#: weg2sn6s's own P entry, copied VERBATIM from the live sidecar
#: (2026-09-10T15:21:55Z). Inlined rather than read from
#: /spinning/evidence-665-f1 on purpose: an evidence-tree-bound test skips on
#: the remote desk, and this is the pin that must run everywhere.
SN6S_P_ENTRY = {
    "group": "P",
    "at": "2026-09-10T15:21:55Z",
    "boot_tag": "weg2sn6s",
    "commit": "23dd8ab2c1",
    "rss_shmem_gib": 38.63,
    "weight_tags_gib": 28.83,
    "arm": {"m_mib": 600, "s_gb": 1, "s_gb_d": 4},
    "run_residual_gib": 16.34,
}


def test_the_reader_reprices_the_record_already_on_disk():
    """The sampler fix cannot reach a record that is already written.

    The sidecar is APPEND-ONLY -- history is evidence -- so the correction has
    to live in the reader or the blocking record stays blocking forever. This
    is the pin on the exact entry that fires the W21.
    """
    v, corr = hl.record_run_residual_gib(SN6S_P_ENTRY)
    assert round(corr, 4) == 8.7172, corr
    assert abs(v - 7.63) < 0.02, v
    assert v < SN6P_IDLE_RESIDUAL, (
        "re-priced, sn6s sits back inside the 7.42-9.01 GiB band every other "
        "boot of this line occupies"
    )
    # The STORED value is untouched: evidence is not rewritten.
    assert SN6S_P_ENTRY["run_residual_gib"] == 16.34


def test_every_single_budget_record_is_repriced_by_exactly_zero():
    """Byte-identical for every record written before D got its own budget."""
    for arm in ({"s_gb": 1, "m_mib": 600}, {"s_gb": 1, "m_mib": 1200},
                {"s_gb": 1, "m_mib": 600, "s_gb_d": 1}):
        e = dict(SN6S_P_ENTRY, arm=arm, run_residual_gib=9.01)
        v, corr = hl.record_run_residual_gib(e)
        assert corr == 0.0 and v == 9.01, (arm, v, corr)


def test_an_unrepriceable_record_keeps_its_stored_value():
    """The conservative direction, and never a silent 0.

    A record whose arm cannot be re-priced keeps the LARGER stored figure --
    an unusable input must not become a smaller charge.
    """
    for arm in ({"s_gb_d": 4}, {"s_gb": "x", "m_mib": 600, "s_gb_d": 4}, "not-a-dict"):
        v, corr = hl.record_run_residual_gib(dict(SN6S_P_ENTRY, arm=arm))
        assert v == 16.34 and corr == 0.0, arm
    v, corr = hl.record_run_residual_gib(
        dict(SN6S_P_ENTRY, run_residual_gib=None)
    )
    assert v is None and corr == 0.0


def test_the_origin_line_prints_both_the_stored_and_the_repriced_figure():
    """A re-priced number must be auditable against the boot that wrote it."""
    origin, src = hl.run_origin_gib(0.0, {"P": SN6S_P_ENTRY})
    assert abs(origin - 7.63) < 0.02, origin
    assert "stored=16.34" in src
    assert "RE-PRICED -8.72 for S_D (#1325)" in src
    # A record needing no correction says nothing about one.
    plain = dict(SN6S_P_ENTRY, arm={"s_gb": 1, "m_mib": 600},
                 run_residual_gib=9.01)
    _, src2 = hl.run_origin_gib(0.0, {"P": plain})
    assert "stored=9.01" in src2 and "RE-PRICED" not in src2


def test_the_origin_falls_to_the_launch_reading_once_the_floor_is_repriced():
    """THE UNBLOCK, as an assertion on the two numbers that produced the W21.

    XSN6's clean-host dry run read 8.00 GiB non-reclaimable at launch and was
    refused because the floor was sn6s's stored 16.34, predicting
    ``run_peak 92.53`` against the 87.30 boot bound. Re-priced, the floor is
    7.63, so the launch reading binds instead -- 8.34 GiB lower than the
    number that fired W21.
    """
    origin, src = hl.run_origin_gib(8.0, {"P": SN6S_P_ENTRY})
    assert abs(origin - 8.0) < 1e-6, origin
    assert "AT OR ABOVE" in src
    stored_origin, _ = hl.run_origin_gib(
        8.0, {"P": dict(SN6S_P_ENTRY, arm={"s_gb": 1, "m_mib": 600})}
    )
    assert abs(stored_origin - 16.34) < 1e-6
    assert abs(stored_origin - origin - 8.34) < 0.01, (
        "the whole gap between the refused prediction and a fundable one"
    )


def test_the_record_still_refuses_to_derive_a_residual_without_a_run_moment():
    """Unchanged and re-pinned: absence is never 0 (the pre-#1325 contract)."""
    r = hl.dormant_image_sample(
        group="P", shmem_before_bytes=0, shmem_after_bytes=0, pids=[],
        weight_tags_gib=SN6S_WTAGS_GIB, interleaved=True, boot_tag="t",
        commit="c", cg_current_bytes=None, reclaimable_bytes=None, arm=None,
    )
    assert r["run_residual_gib"] is None
    assert "not the run moment" in str(r["run_residual_note"])
    assert "not 0" in str(r["run_residual_note"])
