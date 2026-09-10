"""#1318b -- the rings term is NOT double-booked, and the S ladder is pinned.

THE QUESTION PUT TO THE DESK: does `rings_gib` model an allocation that some
other term already charges? Answered NO, from the code and from boot
weg2sn6e's own measurements, and the term is therefore UNCHANGED. What did
change is one instrument line that printed the old constants, and one
measured-but-unpriced post that is now named.

THE TWO THINGS BOTH CALLED "RING", which is where a double-book would hide:
* `host_ring_gib` (`price`) = `ring_bytes / GIB`, handed in by the launcher's
  ring_table and sized from the dormant weight IMAGE. sn6e measured
  `/dev/shm/weg2-hostring-weg2sn6e` at 43 GiB and it **did not change with
  S** -- exactly what an image-derived post should do.
* `rings_gib` (`charge_terms`) = `(RING_P + RING_D) * s_gb` = the L2 host KV
  POOLS, which do scale with S by construction.
Disjoint bytes: /dev/shm shmem for the first, anonymous MAP_SHARED for the
second. `price` even documents that the measured image is reported and not
charged a second time.

AND NEITHER IS CHARGED BY `anchors_gib` OR `heaps_gib`: anchors is the MAMBA
anchor pool (M-scaled, not S-scaled) and heaps is the per-rank process heaps.
Four terms, four allocations.
"""

import pytest

from sglang.srt.weg2 import host_ledger as hl

GIB = hl.GIB
# boot weg2sn6g's own readings, from its front log.
SN6G = dict(
    memtotal=int(118.05 * GIB), memavail=int(110.09 * GIB),
    cg_current=int(21.62 * GIB), ring=int(42.96 * GIB), span1=int(41.50 * GIB),
)
SN6G_RUN_PEAK_S1 = 75.43      # its printed run_peak at S=1 M=600
SN6G_HARD_BOUND = 87.30       # its printed hard bound; NOT changed by this posten


def _kw():
    return dict(ring_bytes=SN6G["ring"], ring_span1_bytes=SN6G["span1"],
                cg_current_bytes=SN6G["cg_current"],
                cg_ceiling_bytes=SN6G["memtotal"])


# --------------------------------------------------------------------------
# (1) no double-booking
# --------------------------------------------------------------------------

def test_only_the_rings_term_scales_with_s():
    """The S-dependence must live in exactly ONE term. If a second term moved
    with S, the two would be charging the same pool twice."""
    a = hl.charge_terms(1, 600, 3, hl.resolve_image_terms(None))
    b = hl.charge_terms(2, 600, 3, hl.resolve_image_terms(None))
    moved = {k for k in a if abs(a[k] - b[k]) > 1e-9}
    assert moved == {"rings_gib", "overhead_gib"}, (
        f"S moved {moved}; only rings (and the 4 % overhead posted ON it) may"
    )


def test_the_image_sized_ring_does_not_scale_with_s():
    """`host_ring_gib` is handed in, not derived from S -- which is why sn6e
    measured 43 GiB in /dev/shm regardless of S."""
    peaks = {}
    for s in (1, 2, 4):
        arm = hl.price(SN6G["memtotal"], SN6G["memavail"], s, 600, **_kw())
        peaks[s] = arm.terms["host_ring_gib"]
    assert len({round(v, 6) for v in peaks.values()}) == 1
    assert peaks[1] == pytest.approx(42.96, abs=0.01)


def test_the_rings_term_reproduces_ds_measured_l2_pool():
    """The arithmetic that identifies WHICH allocation this term models: D's
    half is 30,518 rows x 32,768 B x 3 ranks, and sn6e measured ~2.8 GiB of
    anonymous MAP_SHARED for it."""
    d_gib = hl.RING_D_MULT_GB_PER_S * hl.GB / GIB
    assert d_gib == pytest.approx(2.79, abs=0.02)
    rows, cell, ranks = 30518, 32768, 3
    assert rows * cell * ranks / hl.GB == pytest.approx(3.0, abs=0.01)


def test_anchors_scales_with_m_and_not_with_s():
    """The other candidate for a double-book: anchors is the MAMBA pool."""
    img = hl.resolve_image_terms(None)
    assert hl.charge_terms(1, 600, 3, img)["anchors_gib"] == pytest.approx(
        hl.charge_terms(4, 600, 3, img)["anchors_gib"]
    )
    assert hl.charge_terms(1, 1200, 3, img)["anchors_gib"] > hl.charge_terms(
        1, 600, 3, img
    )["anchors_gib"]


# --------------------------------------------------------------------------
# (2) the S ladder, anchored on the boot's OWN printed peak
# --------------------------------------------------------------------------

def test_the_per_s_increment_is_the_derived_multiplier_plus_its_overhead():
    """peak(S) = peak(1) + (S-1) * 4.63 GiB, and 4.63 = 4.45 * 1.04. Anchored
    on sn6g's printed run_peak rather than on a re-derived origin, because the
    origin depends on readings this test does not have."""
    rings_per_s = (hl.RING_P_MULT_GB_PER_S + hl.RING_D_MULT_GB_PER_S) * hl.GB / GIB
    assert rings_per_s == pytest.approx(4.45, abs=0.01)
    step = rings_per_s * (1.0 + hl.HOST_POOL_OVERHEAD)
    assert step == pytest.approx(4.63, abs=0.01)

    ladder = {s: SN6G_RUN_PEAK_S1 + (s - 1) * step for s in (1, 2, 3, 4)}
    assert ladder[1] == pytest.approx(75.43, abs=0.01)
    assert ladder[2] == pytest.approx(80.06, abs=0.02)
    assert ladder[3] == pytest.approx(84.69, abs=0.02)
    assert ladder[4] == pytest.approx(89.32, abs=0.02)
    # and the bound the Q7 boot is read against
    assert SN6G_HARD_BOUND - ladder[3] == pytest.approx(2.61, abs=0.02)
    assert ladder[4] > SN6G_HARD_BOUND, "S=4 must break the hard bound"


def test_s_three_is_the_last_fundable_rung_on_sn6g_readings():
    """The number Q7 needs: on this boot's own readings the hard bound admits
    S=1..3 and refuses S=4. Stated as the margin, not as a verdict, because
    the ledger takes the verdict itself at boot time from LIVE readings."""
    step = (hl.RING_P_MULT_GB_PER_S + hl.RING_D_MULT_GB_PER_S) * hl.GB / GIB * (
        1.0 + hl.HOST_POOL_OVERHEAD
    )
    fits = [s for s in (1, 2, 3, 4)
            if SN6G_RUN_PEAK_S1 + (s - 1) * step < SN6G_HARD_BOUND]
    assert fits == [1, 2, 3]


# --------------------------------------------------------------------------
# (3) the unpriced post, named and NOT folded into the margin
# --------------------------------------------------------------------------

def test_the_unpriced_anon_mapshared_post_is_named_with_its_arithmetic():
    """sn6e: shmem 51.3 = ring 43.0 + ~8.3 anon MAP_SHARED; the ledger charges
    4.45 of that 8.3 as rings, leaving ~3.85 unpriced -- which covers the
    measured 3.4 GiB under-prediction (79.3 against 75.86) on its own."""
    assert hl.UNPRICED_ANON_MAPSHARED_GIB == pytest.approx(3.4, abs=0.01)
    rings_s1 = (hl.RING_P_MULT_GB_PER_S + hl.RING_D_MULT_GB_PER_S) * hl.GB / GIB
    unpriced = 8.3 - rings_s1
    assert unpriced == pytest.approx(hl.UNPRICED_ANON_MAPSHARED_GIB, abs=0.5)


def test_the_unpriced_post_is_not_charged_and_not_in_the_margin():
    """It is a NAMED reading, never an actuator: folding an unmeasured post
    into a safety margin is the compensation constant the #1232 headroom term
    was deleted for. Neither the reap mark nor the margin may move."""
    import inspect

    src = inspect.getsource(hl)
    i = src.index("UNPRICED_ANON_MAPSHARED_GIB = 3.4")
    # it must not be referenced by any pricing or margin code
    after = src[i + 40:]
    assert "UNPRICED_ANON_MAPSHARED_GIB" not in after, (
        "the unpriced post is READ by pricing code -- it is a reading, not a term"
    )
    # 95.925 GiB is the constant; 95.90 is what the boot LINE prints, rounded.
    # Pinned against the constant with the display rounding named, so a future
    # reader does not "fix" the constant to match the line.
    assert hl.OBSERVED_REAP_CURRENT_BYTES / GIB == pytest.approx(95.93, abs=0.02)


def test_the_provenance_line_prints_the_derived_multipliers_not_the_old_pair():
    """The instrument defect this posten found: `:.0f` rendered 1.778 as "2",
    so boot weg2sn6g printed `rings=(2+3)xS GB (b0)` -- which reads as a
    hand-typed pair and hides that the term is DERIVED. The charge was right
    all along (rings=4.45 GiB at S=1); only the text lied."""
    import inspect

    src = inspect.getsource(hl)
    assert 'f"rings=({RING_P_MULT_GB_PER_S:.0f}' not in src, "the 0-decimal render is back"
    assert "DERIVED from rows x cell bytes" in src
    assert "RING_B0_TOTAL_MULT_GB_PER_S:.3f" in src, (
        "the line must print the reading it replaced, so the two can be compared"
    )
