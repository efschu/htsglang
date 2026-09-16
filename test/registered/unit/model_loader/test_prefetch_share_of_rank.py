"""Checkpoint page-cache warm-up: the share each rank reads (fn1w boot
2026-09-16: under --rank-gpu-id every rank saw local_rank 0 / local_size 1
and all three warmed the same third of the files)."""

from sglang.srt.model_loader.weight_utils import prefetch_share_of_rank


def test_collapsed_local_view_falls_back_to_the_world_rank():
    assert prefetch_share_of_rank(0, 1, 0, 3) == (0, 3)
    assert prefetch_share_of_rank(0, 1, 2, 3) == (2, 3)
    files = list(range(38))
    shares = [files[i::n] for i, n in (prefetch_share_of_rank(0, 1, r, 3) for r in range(3))]
    assert sorted(sum(shares, [])) == files
    assert all(set(a).isdisjoint(b) for a in shares for b in shares if a is not b)


def test_real_local_view_is_kept():
    assert prefetch_share_of_rank(1, 4, 5, 8) == (1, 4)
    assert prefetch_share_of_rank(0, 1, 0, 1) == (0, 1)
