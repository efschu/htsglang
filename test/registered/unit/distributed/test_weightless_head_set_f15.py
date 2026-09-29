"""F15 (5a): the weightless lane's ONE head generalised to a head SET W (Form B).
The lane is the set of one and answers byte-identically; under a set the heads
split over W by the SAME rule the W ranks' projections are built with."""

import pytest

from sglang.srt.distributed import utils as du
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _reset():
    yield
    du.set_weightless_kv_head_rank(None)


def test_the_lane_is_the_set_of_one():
    for h in (0, 1, 2):
        du.set_weightless_kv_head_rank(h)
        assert du.weightless_kv_active() and du.get_weightless_kv_head_rank() == h
        assert du.get_weightless_kv_weight_ranks() == (h,)
        assert du.weightless_head_counts(24, 3) == [24 if r == h else 0 for r in range(3)]
        assert [du.is_weightless_head_rank(r) for r in range(3)] == [r == h for r in range(3)]
        assert [du.weightless_worker_rank(r) for r in range(3)] == [r != h for r in range(3)]
    du.set_weightless_kv_head_rank(None)
    assert not du.weightless_kv_active() and du.get_weightless_kv_weight_ranks() is None
    assert not du.is_weightless_head_rank(0) and not du.weightless_worker_rank(0)


@pytest.mark.parametrize("w,ratios", [((0, 1), (77, 23)), ((0, 1), (3, 1)), ((1, 2), (1, 1)),
                                      ((0, 2), None)])
def test_a_head_set_splits_like_the_w_build(w, ratios):
    """27B geometry: 24 q heads, 4 kv heads. The counts equal what the W ranks'
    qkv projections get under rank_form.form_b_build_context (tp_partition_sizes
    over |W| with the W-restricted plan), with 0 on the KV-only rank."""
    du.set_weightless_kv_weight_ranks(w, ratios)
    assert du.get_weightless_kv_head_rank() == min(w)                      # the lead
    units = du.attn_q_partition_units(24, 4, len(w))
    groups = du.attn_q_partition_groups(4, len(w))
    q = du.weightless_head_counts(24, 3, units=units, groups=groups)
    kv = du.weightless_head_counts(4, 3, units=4)
    with du.scoped_tp_partition_ratios(list(ratios) if ratios else None):
        bq = du.tp_partition_sizes(24, len(w), units=units, groups=groups)
        bkv = du.tp_partition_sizes(4, len(w), units=4)
    assert [q[r] for r in w] == bq and [kv[r] for r in w] == bkv
    k = [r for r in range(3) if r not in w]
    assert [q[r] for r in k] == [0] and [kv[r] for r in k] == [0]
    assert du.weightless_worker_rank(k[0]) and all(du.is_weightless_head_rank(r) for r in w)


def test_a_bad_head_set_is_refused_by_name():
    for ranks, ratios in (([], None), ([0, 0], None), ([0, 1], [1]), ([0, 1], [1, 0])):
        with pytest.raises(ValueError, match="W181"):
            du.set_weightless_kv_weight_ranks(ranks, ratios)


def test_the_backend_plans_its_counts_through_one_helper():
    """F15 (5c): flashinfer's lane branch plans (q, kv) with
    weightless_dcp_head_counts: the lane [H,0,0]/[KV,0,0] as before, the head set
    in whole GQA groups by the W shares (= the W build)."""
    import inspect

    from sglang.srt.layers.attention import flashinfer_backend as fb

    assert "weightless_dcp_head_counts(" in inspect.getsource(fb.FlashInferAttnBackend.__init__)
    du.set_weightless_kv_head_rank(0)
    assert du.weightless_dcp_head_counts(24, 4, 3) == ([24, 0, 0], [4, 0, 0])
    du.set_weightless_kv_weight_ranks((0, 1), (77, 23))
    assert du.weightless_dcp_head_counts(24, 4, 3) == ([18, 6, 0], [3, 1, 0])
    du.set_weightless_kv_weight_ranks((1, 2), None)
    assert du.weightless_dcp_head_counts(24, 4, 3) == ([0, 12, 12], [0, 2, 2])
