"""#1432: the L2 arena is a host term of its own; the fallback pools are priced
at their real sizes, not at the ladder's S/M."""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import host_ledger as hl


def _images():
    return hl.resolve_image_terms(None, want_digest="", form_key_match=False)


def test_arena_term_is_charged_and_the_fallback_pools_priced_real():
    base = hl.charge_terms(1, 2400, 3, _images())
    real = hl.charge_terms(1, 2400, 3, _images(), arena_gib=37.0, staging_gb=0.05, anchor_mib=100)
    assert real["arena_gib"] == 37.0 and base["arena_gib"] == 0.0
    assert real["anchors_gib"] < base["anchors_gib"] / 20   # 100 MiB instead of 2400
    assert real["rings_gib"] < base["rings_gib"] / 10       # 0.05 GB instead of 1 GB
    assert abs(hl._boot_charges_gib(real) - hl._boot_charges_gib(base)
               - (37.0 - (base["anchors_gib"] - real["anchors_gib"]) - (base["rings_gib"] - real["rings_gib"])
                  - (base["overhead_gib"] - real["overhead_gib"]))) < 1e-6


def test_hicache_off_zeroes_the_arena_too():
    off = hl.charge_terms(1, 2400, 3, _images(), arena_gib=37.0, hicache_disabled=True)
    assert off["arena_gib"] == 0.0


def test_the_priced_arm_carries_the_arena_in_its_run_peak():
    """xsn195: charge_terms knew the arena, the Arm's own terms dict did not --
    the run peak (origin + charges) silently dropped 37 GiB."""
    GIB = 1 << 30
    a0 = hl.price(118 * GIB, 100 * GIB, 1, 600, ranks_per_group=3, ring_bytes=1, ring_span1_bytes=1,
                  cg_current_bytes=20 * GIB, reclaimable_bytes=0, cg_ceiling_bytes=118 * GIB)
    a1 = hl.price(118 * GIB, 100 * GIB, 1, 600, ranks_per_group=3, ring_bytes=1, ring_span1_bytes=1,
                  cg_current_bytes=20 * GIB, reclaimable_bytes=0, cg_ceiling_bytes=118 * GIB,
                  arena_gib=30.0, staging_gb=0.05, anchor_mib=100)
    assert a1.terms["arena_gib"] == 30.0 and a0.terms.get("arena_gib", 0.0) == 0.0
    p0, p1 = a0.predicted_run_peak_gib(), a1.predicted_run_peak_gib()
    if p0 is not None and p1 is not None:
        assert p1 > p0 + 15.0, (p0, p1)  # +30 arena minus the rings/anchors priced real
