"""#66 (fnFL2v71, 21.09.): the W11 draft-resident budget follows the build.

v71 got BOTH groups up (D READY after 432.7 s) and was then refused by the
launcher's own gate:

    W11 Weg2DraftResidentOverBudget: measured resident_mib=2202.6 vs budget
    1618.2+256 MiB

The refusal was correct -- the build really is bigger now, for two reasons
fixed the same day, and the budget is a MEASURED constant the module itself
says to replace ("The next boot's L2 resident_mib replaces 1618.2 here").
"""

from sglang.srt.weg2 import launcher


def test_the_budget_is_the_sum_of_the_measured_parts():
    # embed 615.7 (packed vocab, d84f1394fe) + mtp 1522.7 (INT4 draft
    # checkpoint, a905902f47) -- both read off fnFL2v71's producer line.
    assert launcher.P_DRAFT_RESIDENT_BUDGET_MIB == 615.7 + 1522.7


def test_the_v71_measurement_now_passes_inside_the_tolerance():
    got = launcher.check_draft_resident.__wrapped__ if hasattr(
        launcher.check_draft_resident, "__wrapped__"
    ) else launcher.check_draft_resident
    assert got is not None
    measured = 2202.6
    budget = launcher.P_DRAFT_RESIDENT_BUDGET_MIB
    tol = launcher.P_DRAFT_RESIDENT_TOL_MIB
    assert measured <= budget + tol, (measured, budget, tol)
    # and it is not slack: the old budget would still refuse it
    assert measured > 1618.2 + tol


def test_the_gate_still_refuses_a_real_overrun():
    """The tolerance must not have swallowed the gate."""
    budget = launcher.P_DRAFT_RESIDENT_BUDGET_MIB
    tol = launcher.P_DRAFT_RESIDENT_TOL_MIB
    # weg2dk2's 3994 MiB build -- the shape the gate exists for
    assert 3994.0 > budget + tol


def test_w11b_counts_the_tag_pool_cache_as_a_named_term():
    """#66: the fourth term is a DELTA, never the absolute pool occupancy.

    Measured on fnFL2v72's PP2: the pools hold 7649 MiB inactive after the
    load, long before the draft build. Subtracting that absolute number would
    drive W11b's residual strongly negative -- and the gate refuses BOTH
    directions ("an explanation that does not add up is not an explanation").
    """
    import inspect

    from sglang.srt.speculative import draft_kv_producer as dkp

    src = inspect.getsource(dkp.DraftKvProducer.load_resident_embedding)
    assert "after_pool - self._pool_inactive_before_mib" in src
    # the "before" leg is taken where the NVML "before" leg already is
    init = inspect.getsource(dkp.DraftKvProducer.__init__)
    assert "_pool_inactive_before_mib = _tag_pool_inactive_mib()" in init
    # an unmeasured pool (-1) must not silently become a credit of 0
    body = inspect.getsource(dkp._tag_pool_inactive_mib)
    assert "return -1.0" in body

    lau = inspect.getsource(launcher.check_draft_resident)
    # #66 fnFL2v92: der Posten `other_live` steht seither VOR den beiden
    # (lebende Nicht-Modell-Bytes, Attention-Workspace voran); der
    # Cache-Term selbst ist unveraendert.
    assert "r + other_live + released + pooled" in lau
    assert "float(pooled) < 0 else float(pooled)" in lau
