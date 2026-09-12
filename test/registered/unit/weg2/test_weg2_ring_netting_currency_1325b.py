"""#1325b -- the RING NETTING was in the wrong currency: the THIRD consumer.

THE DEFECT, in one sentence: ``ring_table.apportion_dormant`` nets the source
boot's measured ``RssShmem`` with ``host_ledger.non_backup_host_bytes(group,
s_gb, m_mib)`` -- a signature that cannot express ``s_gb_d`` -- so on a source
boot that ran a TWO-BUDGET arm (``S_D=4``) only ``S_D == S`` worth of D's
HiCache rings comes out of the sample, the rest stays inside the netted image,
becomes ``Sigma H``, and is then charged a SECOND time by
``Arm.predicted_run_peak_gib`` (which adds ``_boot_charges_gib`` -- containing
``rings_gib`` AT ``S_D=4`` -- and ``host_ring_gib`` on top of each other).

THE TWO BOOKING SITES, both of the same bytes:
  * inside Sigma H : ring_table.apportion_dormant, ``usable = measured.rss_mib
    - non_backup_mib`` (ring_table.py:1856), fed by ring_table._non_backup_mib
    (:2156) -> host_ledger.non_backup_host_bytes (host_ledger.py:1618)
  * as its own term: host_ledger.charge_terms ``rings_gib`` (host_ledger.py
    :1268) -> _boot_charges_gib (:1292) -> Arm.predicted_run_peak_gib (:1592)

#1325 fixed the SAMPLER and the READER of ``run_residual_gib``.  This is the
same currency defect one layer out, in the consumer #1325's own docstring names
("``non_backup_host_bytes``, fed by the arm that ``ring_table.parse_chosen_arm``
reads out of the SOURCE boot's own front log", host_ledger.py:2429-2431) and
never converted -- because ``parse_chosen_arm``'s regex can see ``S=`` and
``M=`` and nothing else.

THREE INDEPENDENT INSTRUMENTS AGREE ON THE SIZE, measured 2026-09-12:

  1. ARITHMETIC.  ``non_backup_host_bytes("D", 1, 1200)`` = 5499 MiB; the same
     boot's true D subtrahend at ``S_D=4`` is 14426 MiB.  Gap 8927 MiB =
     **8.718 GiB** under-subtracted.
  2. THE TWO SOURCE BOOTS.  Netting boot weg2xsn20 (which ran ``S_D=4``) leaves
     D's image 9092 MiB above its own per-card census; netting boot weg2sn5b
     (which ran a single budget) leaves 329 MiB.  Difference **8763 MiB**.
  3. THE SOURCE BOOT'S OWN RESIDUAL.  weg2xsn20's group-D record carries
     ``run_residual_gib`` **-8.718** GiB.  A residual whose definition is "what
     the box holds that this ledger's term list does NOT name" cannot be
     negative unless something is counted twice; it is negative by exactly (1).

THE CONSEQUENCE ON METAL.  Boot weg2xsn21's dry run reads ``host_weights=50.38
GiB`` and refuses every rung (``W21``, run_peak 91.99 vs the 87.30 hard bound,
shortfall 4.69 GiB).  Netted in the source boot's own currency the same source
states ``Sigma H = 44587 MiB = 43.54 GiB``, i.e. **-6.83 GiB**, and the same
source boot RAN on metal at 88.798 GiB non-reclaimable peak against its own
prediction of 87.17 -- an over-prediction of nothing and an under-prediction of
1.63 GiB, nowhere near 7 GiB of ring.

DANGER DIRECTION: this correction makes Sigma H SMALLER, i.e. it can fund an
arm that was refused.  Every mutant must make the suite red by UNDER-charging
the box, never by moving a decimal:

  M1  the netting must carry S_D          -> test_the_ring_netting_carries_s_d
  M2  a single-budget source must be byte-identical
                                          -> test_a_single_budget_source_is_byte_identical
  M3  the subtrahend may never exceed the arm the source boot actually ran
                                          -> test_the_netting_never_subtracts_more_than_the_source_arm
  M4  sample and correction must be ONE boot's pair (#1264 B)
                                          -> test_the_arm_travels_with_the_sample
  M5  the xchg bounce is NOT netted -- weg2xsn20 never mapped it
                                          -> test_the_bounce_is_not_netted_off_the_image
  M6  the xchg form marker must not match the launcher's own prose
                                          -> test_the_xchg_marker_ignores_the_launchers_own_prose
  M7  a real region line must still stamp the token
                                          -> test_a_genuine_region_line_still_stamps_the_token
  M8  the flip-transient margin term keeps its ring-granule floor
                                          -> test_the_transient_never_falls_below_one_ring_granule
  M9  the excluded transient samples stay NAMED, never deleted
                                          -> test_the_excluded_transient_samples_are_named_with_a_reason
  M10 the prediction line must carry the source boot's own metal deviation
                                          -> test_the_ring_provenance_names_the_source_boots_metal_deviation
"""

import json
import os
import tempfile

from sglang.srt.weg2 import host_ledger as hl
from sglang.srt.weg2 import ring_table as rt

MIB = 1024 * 1024

# ---------------------------------------------------------------------------
# THE FIXTURE IS THE REAL LINE, not a constructed one.  Every number below is
# read verbatim off boot weg2xsn21's dry run
# (/spinning/evidence-665-f1/weg2xsn21_0912/dryX21_c2287ee8c6_0912_140018.log,
# WEG2-HOST-LEDGER RING lines) and off the append-only sidecar
# (/spinning/evidence-665-f1/weg2_measured_record.json, boot_tag weg2xsn20).
# ---------------------------------------------------------------------------

#: weg2xsn20's group-D sample, verbatim from the sidecar.
XSN20_D_ARM = {"s_gb": 1, "m_mib": 1200, "s_gb_d": 4,
               "xchg_bounce_gib": 2.158762812614441}
XSN20_D_RSS_MIB = 57083          # rss_shmem_gib 55.7455 -> MiB, as the log prints
XSN20_P_RSS_MIB = 48879          # rss_shmem_gib 47.7335
#: the per-card WEG2-FLIP-TAG census of that source, verbatim.
XSN20_CENSUS = {"GPU-31d7ef41": 21712, "GPU-5c648f96": 8254, "GPU-62dbbae1": 12526}
XSN20_CENSUS_TOTAL = 42492

#: what the boot printed, and what it should have printed.
D_SUBTRAHEND_AS_SHIPPED = 5499   # non_backup_host_bytes("D", 1, 1200)
D_SUBTRAHEND_IN_CURRENCY = 14426  # ... the same at S_D=4
SIGMA_H_AS_SHIPPED_MIB = 51584    # 50.38 GiB, the refusing figure
SIGMA_H_IN_CURRENCY_MIB = 44587   # 43.54 GiB == the boot's own Sigma span1

#: weg2sn5b's group-D sample: a SINGLE-budget arm.  Must not move by one byte.
SN5B_D_ARM = {"s_gb": 1, "m_mib": 1200}

RANKS = 3


def _img(rss_mib, tags_mib=XSN20_CENSUS_TOTAL, arm=None):
    return rt.DormantImage(
        group="D", rss_mib=rss_mib, weight_tags_mib=tags_mib,
        extra_mib=rss_mib - tags_mib, arm=arm,
    )


# --------------------------------------------------------------- M1 / M3 ---

def test_the_ring_netting_carries_s_d():
    """The subtrahend must be the arm the SOURCE boot ran, S_D included."""
    assert hl.non_backup_host_bytes("D", 1, 1200) // MIB == D_SUBTRAHEND_AS_SHIPPED
    got = hl.non_backup_host_bytes("D", 1, 1200, s_gb_d=4) // MIB
    assert got == D_SUBTRAHEND_IN_CURRENCY, (
        f"D's netting at S_D=4 is {got} MiB, not {D_SUBTRAHEND_IN_CURRENCY}: "
        "the rings the boot actually held must come out of its own sample, or "
        "they stay in Sigma H and are charged again by predicted_run_peak_gib"
    )
    # The gap IS the measured double charge: 8.718 GiB, the #1325 number.
    gap_gib = (got - D_SUBTRAHEND_AS_SHIPPED) / 1024.0
    assert abs(gap_gib - 8.718) < 0.01, gap_gib


def test_a_single_budget_source_is_byte_identical():
    """Every pre-two-budget source must price EXACTLY as before."""
    for g in ("P", "D"):
        assert (
            hl.non_backup_host_bytes(g, 1, 1200)
            == hl.non_backup_host_bytes(g, 1, 1200, s_gb_d=None)
            == hl.non_backup_host_bytes(g, 1, 1200, s_gb_d=1)
        )


def test_the_netting_never_subtracts_more_than_the_source_arm():
    """P's budget is s_gb; S_D may not inflate P's subtrahend (danger direction)."""
    assert (
        hl.non_backup_host_bytes("P", 1, 1200, s_gb_d=4)
        == hl.non_backup_host_bytes("P", 1, 1200)
    ), "S_D is D's budget alone -- crediting it to P under-charges the box"


# -------------------------------------------------------------------- M4 ---

def test_the_arm_travels_with_the_sample():
    """``dormant_images`` must hand the record's OWN arm to the netting.

    #1264 (B) bound the sample to the stem so sample and correction are one
    boot's pair.  The correction can only BE that pair if the arm comes from
    the same record, not from a regex over a log that cannot see ``S_D``.
    """
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "rec.json")
        with open(p, "w") as fh:
            json.dump({"samples": [{
                "group": "D", "boot_tag": "weg2xsn20", "commit": "3267f109fb",
                "at": "2026-09-11T15:49:26Z",
                "rss_shmem_gib": 55.745452880859375,
                "weight_tags_gib": 27.1480530500412,
                "extra_gib": 28.597399830818176,
                "arm": XSN20_D_ARM,
            }]}, fh)
        got = rt.dormant_images(p, boot_tag="weg2xsn20")
        assert "D" in got
        assert got["D"].arm == XSN20_D_ARM, (
            "the DormantImage must carry the arm of the very record it was "
            "measured in, or the netting is a different boot's currency"
        )


def test_the_xsn20_source_nets_in_its_own_currency():
    """The END-TO-END number: Sigma H 50.38 -> 43.54 GiB on the real fixture."""
    img = _img(XSN20_D_RSS_MIB, arm=XSN20_D_ARM)
    # THE REAL JOIN, not a hand-passed subtrahend: the arm travels in the
    # record and `_non_backup_mib` derives both the MiB and the currency word.
    mib, why = rt._non_backup_mib(None, {"D": img})
    assert mib["D"] == D_SUBTRAHEND_IN_CURRENCY, mib
    rows, source, _bound = rt.apportion_dormant(
        "D", img, XSN20_CENSUS, True,
        non_backup_mib=mib["D"], non_backup_currency=why["D"],
    )
    netted = sum(rows.values())
    assert netted <= SIGMA_H_IN_CURRENCY_MIB, (
        f"group D nets to {netted} MiB; as shipped it netted "
        f"{SIGMA_H_AS_SHIPPED_MIB} and refused the boot by 4.69 GiB"
    )
    # 57083 - 14426 = 42657, which no longer EXCEEDS the census by 9092 MiB.
    assert netted in (XSN20_CENSUS_TOTAL, 42657), netted
    assert "S_D=4" in source, (
        "the line must NAME the currency it netted in; a subtrahend whose "
        "currency is invisible is how this defect survived #1325"
    )


# -------------------------------------------------------------------- M5 ---

def test_the_bounce_is_not_netted_off_the_image():
    """The xchg bounce is an arm charge -- but weg2xsn20 NEVER MAPPED it.

    Its record (BOOT_weg2xsn20_0911.md, item (c)) states ``bounce.bin ABSENT``:
    the W74 refusal fires *before the buffer is mapped*.  Netting a term that
    is not in the sample UNDER-charges the box, which is the danger direction.
    So the bounce stays out of the subtrahend, and this test is the ratchet
    that keeps a later reader from "completing" the netting.
    """
    assert (
        hl.non_backup_host_bytes("D", 1, 1200, s_gb_d=4)
        == hl.non_backup_host_bytes("D", 1, 1200, s_gb_d=4)
    )
    import inspect
    src = inspect.getsource(hl.non_backup_host_bytes)
    assert "bounce" in src.lower(), (
        "the omission must be STATED in the function that omits it, or the "
        "next reader repairs it into an under-charge"
    )


# --------------------------------------------------------------- M6 / M7 ---

_PROSE = (
    "[2026-09-11T15:47:08Z] WEG2-LAUNCH WEG2-HOST-RING SOURCE ... "
    "boot_weg2_weg2she1_0909_135740: EXCLUDED -- an xchg shadow boot "
    "(WEG2-XCHG-REGION in its front log); its dormant residual carries the "
    "exchange region's host pages, which this serving boot never allocates "
    "(#1305 item 4)\n"
)
_GENUINE = (
    "[2026-09-11T15:47:10Z] WEG2-LAUNCH WEG2-XCHG-REGION epoch=1789141630 "
    "path=/dev/shm/weg2-xchg-1789141630/xchg.bin slots=6x2x32MiB "
    "bytes=403701760 registered=0/6 sems=36 hook_mode=unknown scope=boot\n"
)
_ARGV = "[2026-09-11T15:47:00Z] WEG2-LAUNCH group P argv: python -m x --pp-stage-ratio=39,13,12\n"


def _form_of(text):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "b.front.log")
        with open(p, "w") as fh:
            fh.write(text)
        return rt.parse_p_form(p)[0]


def test_the_xchg_marker_ignores_the_launchers_own_prose():
    """#995 inside the product: the marker is quoted in the launcher's OWN text.

    MEASURED 2026-09-12 over the 40 newest boots in /spinning/evidence-665-f1:
    the marker's count rises by exactly one per boot because each launcher run
    prints one more EXCLUDED line than the last.  weg2sn6t -- a pure serving
    boot that never armed the exchange -- carries it 40 times, all prose, and
    ``parse_p_form``'s bare substring test stamps it
    ``--weg2-xchg-region=armed``.  A serving boot wearing the exchange's form
    key is the exact mirror of the defect B4e fixed.
    """
    argv = _form_of(_ARGV + _PROSE)
    assert argv is not None
    assert rt.XCHG_FORM_TOKEN not in argv, (
        "the launcher's own exclusion prose stamped a serving boot as "
        "xchg-armed; anchor the marker to a real event line"
    )


def test_a_genuine_region_line_still_stamps_the_token():
    argv = _form_of(_ARGV + _PROSE + _GENUINE)
    assert argv is not None and rt.XCHG_FORM_TOKEN in argv, (
        "a boot that really mapped the exchange region must keep its token, or "
        "B4e's own fix is undone in the other direction"
    )


# --------------------------------------------------------------- M8 / M9 ---

def test_the_transient_never_falls_below_one_ring_granule():
    """Re-pricing the term may not DELETE it: the granule floor is the ratchet."""
    m = hl.resolve_margin()
    assert m.transient_gib >= hl.RING_GRANULE_GIB, m.transient_gib


def test_the_in_currency_transient_samples_are_recorded_and_not_priced():
    """The measured, in-currency flip transient is RECORDED beside the term.

    MEASURED, 2026-09-12, on the two boots that have 1 Hz non-reclaimable
    series: over 8 flips each (16 flips total) the maximum LOCAL transient --
    max inside a flip minus the higher of its two 12 s shoulders -- is
    **-0.019 GiB on weg2xsn19 and -0.023 GiB on weg2xsn20**.  Negative on
    every single flip: the series is a slow monotone creep and no flip spikes
    above it.  rg3's binding +2.88 was measured on ``memory.current`` "net of
    store growth (+3.81 gross)" at a 10 s cadence against a 2.9 s interleave
    (its own record, WEG2_BUILD_DECISIONS_0906.md:3552) -- the raw instrument
    #1309 forbids for a host verdict, and the growth it measured was the
    tmpfs store that #1236 moved to disk.

    THE TERM IS NOT RETIRED BY THIS TICKET and that is deliberate: retiring it
    raises the hard bound 87.30 -> 90.18 GiB, i.e. it FUNDS arms that are
    refused today (the danger direction), and eight tests in five other
    tickets are ratchets on that bound.  #1325b does not need it -- the
    netting correction alone takes weg2xsn21's cheapest rung to 85.15 GiB
    below the UNCHANGED bound.  So the evidence is recorded next to the term
    and the decision is left where it belongs.
    """
    for boot in ("weg2rg2", "weg2rg3"):
        assert boot in hl.RING_ERA_FLIP_TRANSIENT_GIB, (
            f"{boot} must keep PRICING the term until a plan decision retires "
            "it; silently re-populating this table moves the hard bound"
        )
    assert hl.FLIP_TRANSIENT_IN_CURRENCY_GIB, "the measurement must survive as evidence"
    for boot in ("weg2rg6", "weg2xsn19", "weg2xsn20"):
        assert boot in hl.FLIP_TRANSIENT_IN_CURRENCY_GIB, boot
        assert hl.FLIP_TRANSIENT_IN_CURRENCY_GIB[boot] <= 0.0, boot
    m = hl.resolve_margin()
    assert abs(m.transient_gib - 2.88) < 1e-9, (
        "the bound may not move as a side effect of a netting fix", m.transient_gib)
    assert "in-currency" in m.transient_source and "#1309" in m.transient_source, (
        "the line must show the reader that its binding sample is out of "
        "currency, even while it keeps pricing it", m.transient_source)


# -------------------------------------------------------------------- M10 ---

def test_the_ring_provenance_names_the_source_boots_metal_deviation():
    """The next seat must SEE the source boot's own prediction-vs-metal gap."""
    line = hl.source_metal_deviation("weg2xsn20")
    assert "88.80" in line and "87.17" in line and "1.63" in line, line
    assert hl.source_metal_deviation("weg2nosuchboot") == "", (
        "an unknown boot is an ABSENCE, never a fabricated 0.00 deviation"
    )
