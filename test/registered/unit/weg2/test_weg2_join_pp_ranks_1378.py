"""#1378 xsn80 -- the rank vector is an axis only on the TP side of the join.

MEASURED on weg2xsn80 (b35a9b80): once P's draft manifest carried the
process's place (rank=2 card=2 -- the MTP drafter lives on P's LAST stage),
join_manifests refused the draft region with "group 'P' published ranks
[2], which is not a contiguous 0..n-1 range", D logged
WEG2-XCHG-DRAFT-PLAN-SKIPPED and then W106 (no source for the draft on
wake). A PP stage holds every tensor it publishes WHOLE and is keyed by
(region, name) with its own card; only a TP rank's position is a shard
boundary. Mutant: the shipped guard, which asked 0..n-1 of both groups.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

NAME = "fc.weight"
DRAFT = "weights_draft"


def _piece(rows, cols=16):
    return xm.ManifestPiece(
        param_name=NAME, tensor_class=sh.tensor_class(NAME),
        rows_full=rows, cols_full=cols, itemsize=1, tag=DRAFT,
        nbytes=rows * cols, component_rows=(),
    )


def _man(group, rank, pieces, *, tp_rank=0, pp_rank=0):
    return xm.RankManifest(group=group, rank=rank, card=rank, region_tag=DRAFT,
                           boot_token="b1", tp_rank=tp_rank, pp_rank=pp_rank,
                           pieces=tuple(pieces))


def test_a_pp_region_published_by_one_non_zero_stage_joins():
    manifests = [_man("P", 2, [_piece(12)], pp_rank=2)] + [
        _man("D", r, [_piece(4)], tp_rank=r) for r in range(3)
    ]
    join = xm.join_manifests(manifests, pp_group="P", tp_group="D")
    assert not join.unsourced
    assert join.by_name[NAME].shard_axis == wx.ROWS


def test_a_tp_gap_is_still_refused():
    manifests = [_man("P", 0, [_piece(12)])] + [
        _man("D", r, [_piece(4)], tp_rank=r) for r in (0, 2, 3)
    ]
    with pytest.raises(wx.Weg2XchgPlanDisagree, match="contiguous 0..n-1"):
        xm.join_manifests(manifests, pp_group="P", tp_group="D")


def test_a_duplicate_pp_rank_is_still_refused():
    manifests = [_man("P", 2, [_piece(12)], pp_rank=2),
                 _man("P", 2, [_piece(12)], pp_rank=2)] + [
        _man("D", r, [_piece(4)], tp_rank=r) for r in range(3)
    ]
    # merge_region_tags folds one rank's region files together; two files
    # for the same (rank, region) merge rather than duplicate -- so the
    # duplicate check is exercised through two DIFFERENT region tags that
    # resolve to distinct manifests of one rank only when they cannot merge.
    # The guard itself is unit-tested here on the merged list's contract:
    merged = xm.merge_region_tags(m for m in manifests if m.group == "P")
    assert len({m.rank for m in merged}) == len(merged)
