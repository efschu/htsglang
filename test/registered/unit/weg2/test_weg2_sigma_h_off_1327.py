"""#1327 (S6 slice 2) -- the LEDGER PROOF: Sigma H = 0 releases the headroom,
and an absent ring is not an absent measurement.

THE BLOCKER THIS CLOSES. `price()` refuses by name when `ring_bytes <= 0`:

    W20 Weg2HostLedgerRefused: the host weights term has no measured source
    (ring_bytes=0, ring_span1_bytes=0). C19 deleted BACKUP_P_BYTES /
    BACKUP_D_BYTES because they were stale constants ... THIS BOOT REFUSES.

That refusal is correct for the case it was written for: a boot whose
PREDECESSOR logged no ring table, i.e. a number the planner would have to
guess. It is WRONG for `--weg2-weight-source exchange`, where there is no host
weights ring at all -- the launcher publishes no `TMS_HOST_RING_*`, Sigma H is
0 by design, and the arming line already prints `ring_H_mib=0`. So every
`exchange` boot would have died at the ledger before a byte moved.

Two facts, one spelling (`ring_bytes == 0`), separated by a DECLARATION the
caller makes at the one site that already knows the arm -- never by loosening
the guard, which would let the stale-table case through silently.

WHAT IS ACTUALLY RELEASED, and the S6 planning had it attached to the wrong
term. There are TWO rings in this ledger:

  * `rings_gib`   -- the HiCache HOST POOLS,
                     `RING_P_MULT_GB_PER_S * s_gb + RING_D_MULT_GB_PER_S * s_gb_d`,
                     12.83 GiB at S=1 / S_D=4. **The exchange does not touch it.**
  * `host_ring_gib` -- the WEIGHTS ring, Sigma H, `ring_bytes / GIB`, MEASURED
                     **32.19 GiB** on this rig at the run moment. **This is
                     what S6 sets to 0.**

So the headroom S6 buys is ~2.7x the figure the planning assumed, and it is
the term that makes S7's shared canonical ring fundable.

DANGER DIRECTION is a boot that prices as if it had no ring while it has one
(or the reverse), so every mutant must make the suite red by mispricing a
moment, never by moving a decimal:

  M1  the declaration must not double as the stale-table escape
      -> test_the_ring_form_still_refuses_a_boot_with_no_measured_table
  M2  a declared absence must actually zero BOTH moments
      -> test_both_moments_charge_zero_for_an_absent_ring
  M3  the released headroom must be exactly Sigma H, never twice
      -> test_the_release_is_exactly_sigma_h_at_the_run_moment
  M4  a contradiction between the declaration and the bytes must refuse
      -> test_a_declared_absence_with_a_non_zero_ring_is_a_contradiction
  M5  the HiCache rings must be UNAFFECTED (the wrong-term trap)
      -> test_the_hicache_rings_are_untouched_by_the_declaration
  M6  the correction to the S_D sizing must not be re-lost
      -> test_s_d_is_demand_driven_and_the_three_is_the_rank_count
"""

import pytest

from sglang.srt.weg2 import host_ledger as hl

RANKS = 3
# The rig's own figures, from the weg2sn6s ledger TERMS line.
SIGMA_H_BYTES = int(32.19 * hl.GIB)
SPAN1_BYTES = int(29.21 * hl.GIB)
MEMTOTAL = int(118.05 * hl.GIB)
MEMAVAIL = int(104.02 * hl.GIB)


#: A cgroup sample MUST be passed or `predicted_run_peak_gib()` answers None
#: by design ("a prediction has no origin to add to, and a number without its
#: origin is exactly the reading that made weg2dk5 look fundable"). The figures
#: are weg2sn6s's own TERMS line: memory.current 15.36 GiB, reclaimable 1.33.
CG_CURRENT = int(15.36 * hl.GIB)
CG_RECLAIMABLE = int(1.33 * hl.GIB)


def _price(**kw):
    base = dict(
        memtotal_bytes=MEMTOTAL, memavail_bytes=MEMAVAIL, s_gb=1, m_mib=600,
        ranks_per_group=RANKS, s_gb_d=4,
        cg_current_bytes=CG_CURRENT, reclaimable_bytes=CG_RECLAIMABLE,
    )
    base.update(kw)
    return hl.price(**base)


# --------------------------------------------------------------------------
# M1 / M4: the declaration is not an escape hatch
# --------------------------------------------------------------------------


def test_the_ring_form_still_refuses_a_boot_with_no_measured_table():
    """M1: the case W20 was written for must keep refusing.

    `ring_bytes == 0` WITHOUT the declaration still means "the predecessor
    logged no WEG2-CHUNK-BYTES / WEG2-FLIP-TAG lines", and the planner refuses
    to guess (R22). Loosening the guard instead of separating the two facts
    would have let exactly this through.
    """
    with pytest.raises(hl.Weg2HostLedgerRefused) as ei:
        _price(ring_bytes=0, ring_span1_bytes=0)
    assert "W20 Weg2HostLedgerRefused" in str(ei.value)
    assert "no measured source" in str(ei.value)
    # And a half-measured table is refused too, unchanged.
    with pytest.raises(hl.Weg2HostLedgerRefused):
        _price(ring_bytes=SIGMA_H_BYTES, ring_span1_bytes=0)


def test_a_declared_absence_with_a_non_zero_ring_is_a_contradiction():
    """M4: the declaration and the bytes may not disagree.

    A contradiction resolved silently is how a charged term becomes invisible.
    """
    with pytest.raises(ValueError) as ei:
        _price(ring_bytes=SIGMA_H_BYTES, ring_span1_bytes=SPAN1_BYTES,
               ring_absent_by_design=True)
    assert "contradict" in str(ei.value)
    with pytest.raises(ValueError):
        _price(ring_bytes=0, ring_span1_bytes=SPAN1_BYTES,
               ring_absent_by_design=True)


# --------------------------------------------------------------------------
# M2 / M3: what the declaration releases
# --------------------------------------------------------------------------


def test_both_moments_charge_zero_for_an_absent_ring():
    """M2: a declared absence prices, and prices the ring at 0 twice."""
    arm = _price(ring_bytes=0, ring_span1_bytes=0, ring_absent_by_design=True)
    t = arm.terms
    assert t["host_ring_gib"] == 0.0
    assert t["host_ring_span1_gib"] == 0.0
    assert t["ring_absent_by_design"] is True


def test_the_release_is_exactly_sigma_h_at_the_run_moment():
    """M3: the headroom is Sigma H, once -- not twice, not a fraction.

    Same box, same arm, only the ring differing. The run leftover must grow by
    exactly Sigma H and the launch leftover by exactly span 1, which is the
    R7 asymmetry (only span 1 is registered at launch).
    """
    with_ring = _price(ring_bytes=SIGMA_H_BYTES, ring_span1_bytes=SPAN1_BYTES)
    without = _price(ring_bytes=0, ring_span1_bytes=0,
                     ring_absent_by_design=True)
    run_delta = without.run_leftover_gib - with_ring.run_leftover_gib
    launch_delta = without.launch_leftover_gib - with_ring.launch_leftover_gib
    assert abs(run_delta - SIGMA_H_BYTES / hl.GIB) < 0.01, run_delta
    assert abs(launch_delta - SPAN1_BYTES / hl.GIB) < 0.01, launch_delta
    # 32.19 GiB at the run moment -- 2.7x the 12 GiB the S6 planning assumed,
    # because that figure was the HiCache pools and not the weights ring.
    assert run_delta > 30.0
    # And the predicted run peak falls by the same amount, once. Both sides
    # must be priced at all -- a None here would mean the fixture forgot the
    # cgroup sample, and comparing None would fail as a TypeError rather than
    # as the finding it looks like.
    a, b = with_ring.predicted_run_peak_gib(), without.predicted_run_peak_gib()
    assert a is not None and b is not None, (a, b)
    assert abs((a - b) - SIGMA_H_BYTES / hl.GIB) < 0.01, (a, b)


def test_the_terms_line_says_which_kind_of_zero_it_is():
    """An unarmed gate must never be readable as a passed one.

    A SOURCE pin, and deliberately not a `pytest.skip` on a formatter that
    does not exist: the TERMS line is built inside `choose()`, so there is no
    `format_terms` to call, and a skip here would be a test that reports
    success for a line nobody checked. The error class of this edit is "a
    branch dropped out of an f-string chain", so the check is that the branch
    is present AND that it hangs off the declaration rather than off the zero.
    """
    import inspect

    src = inspect.getsource(hl.choose)
    assert "RING ABSENT BY DESIGN" in src
    assert "DECLARED absence, not a missing measurement" in src
    assert 'if t.get("ring_absent_by_design") else ""' in src, (
        "the sentence must be conditioned on the DECLARATION; conditioning it "
        "on host_ring_gib == 0 would print it for the stale-table case too, "
        "which is the very confusion this slice removes"
    )


# --------------------------------------------------------------------------
# M5: the wrong-term trap
# --------------------------------------------------------------------------


def test_the_hicache_rings_are_untouched_by_the_declaration():
    """M5: `rings_gib` is the HiCache pools and the exchange does not free it.

    This is the pin against the mistake the S6 planning made: attributing the
    exchange's headroom to `rings_gib` (12.83 GiB) instead of `host_ring_gib`
    (Sigma H, 32.19 GiB). If a future edit "frees" the HiCache rings under the
    exchange, D's L2 disappears and one cap-sized store read stops fitting.
    """
    with_ring = _price(ring_bytes=SIGMA_H_BYTES, ring_span1_bytes=SPAN1_BYTES)
    without = _price(ring_bytes=0, ring_span1_bytes=0,
                     ring_absent_by_design=True)
    assert with_ring.terms["rings_gib"] == without.terms["rings_gib"]
    assert abs(without.terms["rings_gib"] - 12.83) < 0.05, (
        without.terms["rings_gib"]
    )
    # The two terms are different quantities and neither is the other.
    assert abs(without.terms["rings_gib"]
               - with_ring.terms["host_ring_gib"]) > 15.0


# --------------------------------------------------------------------------
# M6: the sizing correction, pinned so it cannot be re-lost
# --------------------------------------------------------------------------


def test_s_d_is_demand_driven_and_the_three_is_the_rank_count():
    """M6: the correction that struck the 24 GiB L2 target.

    Three claims, each checked against the code rather than restated:
      * `derive_d_hicache_size_gb` has NO host-budget term -- freeing the rings
        cannot grow S_D, and no pin is holding it at 4 either;
      * the 3.0 in `RING_D_MULT_GB_PER_S` is `sum(cells)/max(cells)`, i.e. the
        RANK COUNT, not a sidecar factor;
      * the invariant already holds at full cap.
    """
    import inspect

    src = inspect.getsource(hl.derive_d_hicache_size_gb)
    for forbidden in ("memavail", "memtotal", "cg_current", "base_gib",
                      "leftover", "margin"):
        assert forbidden not in src, (
            f"{forbidden!r} in the S_D derivation would make it supply-driven; "
            "it is demand-driven (cap x share x cell / fraction) and that is "
            "why freeing the rings does not enlarge it"
        )
    t = hl.derive_d_hicache_size_gb(262144, 0.3750,
                                    hl.CELL_BYTES_D_PER_RANK[0], 0.90)
    assert t["s_gb"] == 4.0
    assert abs(t["gb_exact"] - 3.579) < 0.01
    assert t["rows_needed"] == 98304.0
    assert t["rows_lendable"] == 109863.0
    assert t["rows_lendable"] >= t["rows_needed"], (
        "one cap-sized read must fit in ONE prefetch on the largest rank"
    )
    cells = hl.CELL_BYTES_D_PER_RANK
    assert abs(hl.RING_D_MULT_GB_PER_S - sum(cells) / max(cells)) < 1e-9
    assert abs(hl.RING_D_MULT_GB_PER_S - RANKS) < 1e-9, (
        "the 3.0 is the rank count; reading it as a sidecar factor is what "
        "produced the 24 GiB target"
    )
