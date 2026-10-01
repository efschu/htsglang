"""2026-10-01 user order ("trage 105gb ein"): the reap mark follows a finite
cgroup memory.max (the operator's Docker --memory) instead of being capped by
the recorded CT999 host mark 95.90 GiB, which stays the fallback only when no
finite cgroup ceiling exists."""

from sglang.srt.weg2 import host_ledger as hl

GIB = hl.GIB
CT999 = hl.OBSERVED_REAP_NONRECLAIM_BYTES / GIB


def test_finite_cgroup_ceiling_is_the_mark():
    mark = hl.reap_mark_gib(105 * 1024**3, "cgroup memory.max")
    assert abs(mark - 105.0) < 1e-9
    assert mark > CT999


def test_no_ceiling_keeps_the_ct999_fallback():
    assert hl.reap_mark_gib() == CT999
    assert hl.reap_mark_gib(None, "") == CT999


def test_lxcfs_fallback_label_keeps_the_ct999_fallback():
    assert hl.reap_mark_gib(118 * 1024**3, "lxcfs MemTotal FALLBACK") == CT999


def test_small_container_still_binds_below_ct999():
    assert abs(hl.reap_mark_gib(84 * 1024**3, "cgroup memory.max") - 84.0) < 1e-9


def test_predicted_97_is_under_the_105g_hard_bound():
    """27B port: the margin is the line's own (resolve_margin, 8.60 GiB on 27B
    vs NF's smaller one), so the NF y6f specimen 97.0 GiB is not the 27B
    number. The rule is: a run peak just above the old CT999 bound is refused
    there and funded under the 105g container mark."""
    margin = hl.resolve_margin()
    hard_bound = hl.reap_mark_gib(105 * 1024**3, "cgroup memory.max") - margin.total_gib
    assert abs(hard_bound - (105.0 - margin.total_gib)) < 1e-9
    old_bound = CT999 - margin.total_gib
    predicted = old_bound + 1.0          # refused against the old CT999 mark ...
    assert predicted >= old_bound
    assert predicted < hard_bound        # ... funded under the 105g container mark
