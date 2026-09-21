"""fnFL2 v17 (21.09.): under Form A (--rank-tp-ratio 1,0,0) the canonical
mamba window for the attention host must cover the WHOLE blob; the unit
partition's one-unit-per-rank floor gave each worker one of 16 units and
cut the host to 7/8 (51480576 of 58834944 bytes)."""

from sglang.srt.mem_cache import hicache_migrate as hm


def _spec():
    return hm.MambaBlobSpec(
        num_layers=3, num_heads=32, head_dim=128, state_size=128, conv_dim=8192,
        conv_width=3, key_dim=2048, value_dim=4096, units=16,
        temporal_itemsize=4, conv_itemsize=2,
    )


def test_zero_ratio_workers_own_no_heads():
    spec = _spec()
    assert hm._partition(32, [1, 0, 0], 16) == [32, 0, 0]
    assert hm._partition(32, [1], 16) == [32]
    assert hm._partition(32, [1, 1], 16) == [16, 16]  # the classic path is unchanged


def test_host_window_under_form_a_is_the_whole_blob():
    spec = _spec()
    assert hm.temporal_extents(spec, [1, 0, 0], 0) == hm.temporal_extents(spec, [1], 0)
    assert hm.conv_extents(spec, [1, 0, 0], 0) == hm.conv_extents(spec, [1], 0)
    assert spec.shard_for_rank([1, 0, 0], 0).total_bytes == spec.total_bytes
    assert spec.shard_for_rank([1, 0, 0], 1).total_bytes == 0


def test_draft_migrate_uses_the_same_rule():
    from sglang.srt.mem_cache import draft_migrate as dm

    assert dm._partition(32, [1, 0, 0], 16) == [32, 0, 0]
