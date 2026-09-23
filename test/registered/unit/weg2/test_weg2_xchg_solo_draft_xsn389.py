"""xsn389 (19.09.2026): --speculative-draft-placement solo x the weight exchange.

The solo HOST (D rank 0) holds the DFlash draft WHOLE, the shadows hold a
meta draft (no bytes). The join must plan the draft on the host only:
ROWS with widths (whole, 0, 0); `_blocks_of` gives a 0-width rank NO block
(so `_emit` skips it); any other partial hold stays W74."""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

TAG = wx.GPU_MEMORY_TYPE_WEIGHTS
CARDS = (0, 1, 2)
MAIN = "model.layers.0.mlp.down_proj.weight"      # a normal column cut
DRAFT = "fc.weight_packed"                        # the solo-held draft tensor


def _piece(name, rows, cols, item=1, tag=TAG):
    from sglang.srt.weg2 import weight_exchange_shadow as sh
    return xm.ManifestPiece(param_name=name, tensor_class=sh.tensor_class(name),
                            rows_full=rows, cols_full=cols, itemsize=item,
                            tag=tag, nbytes=rows * cols * item)


def _manifests(*, shadow_rows=None):
    p = [xm.RankManifest(group="P", rank=0, card=0, region_tag=TAG, boot_token="b1",
                         tp_rank=0, pp_rank=0,
                         pieces=(_piece(MAIN, 2048, 1200), _piece(DRAFT, 1600, 20480)))]
    d = []
    for r, w in enumerate((400, 400, 400)):
        pieces = [_piece(MAIN, 2048, w)]
        if r == 0:
            pieces.append(_piece(DRAFT, 1600, 20480))
        elif shadow_rows is not None:
            pieces.append(_piece(DRAFT, *shadow_rows))
        d.append(xm.RankManifest(group="D", rank=r, card=CARDS[r], region_tag=TAG,
                                 boot_token="b1", tp_rank=r, pp_rank=0,
                                 pieces=tuple(pieces)))
    return p + d


def test_a_tensor_held_whole_by_one_tp_rank_joins_as_rows_whole_zero_zero():
    join = xm.join_manifests(_manifests(), pp_group="P", tp_group="D")
    by = {t.param_name: t for t in join.tensors}
    d = by[DRAFT]
    assert d.shard_axis == wx.ROWS and d.tp_widths == (1600, 0, 0)
    assert d.rows_full == 1600 and d.cols_full == 20480 and d.pp_stage == 0
    assert by[MAIN].shard_axis == wx.COLS and by[MAIN].tp_widths == (400, 400, 400)


@pytest.mark.parametrize("direction", [wx.LEGS_PP_TO_TP, wx.LEGS_TP_TO_PP])
def test_the_plan_moves_the_solo_tensor_to_and_from_the_host_only(direction):
    join = xm.join_manifests(_manifests(), pp_group="P", tp_group="D")
    plan = xm.plan_from_join(join, direction=direction)
    draft = [d for d in plan.descs if d.param_name == DRAFT]
    assert draft, "the solo tensor must be planned"
    tp_is_dst = direction == wx.LEGS_PP_TO_TP
    ranks = {int(d.dst_rank) if tp_is_dst else int(d.src_rank) for d in draft}
    assert ranks == {0}
    assert sum(int(d.nbytes) for d in draft) == 1600 * 20480
    main = [d for d in plan.descs if d.param_name == MAIN]
    assert {int(d.dst_rank) if tp_is_dst else int(d.src_rank) for d in main} == {0, 1, 2}


def test_sharded_meta_shapes_on_the_shadows_are_still_w68():
    # xsn389 as it happened: the shadows published their meta draft's sharded
    # shapes -- a genuine disagreement, refused by name (the fix is that the
    # shadows publish no meta rows at all, `card_inventory` skips them)
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        xm.join_manifests(_manifests(shadow_rows=(5120, 6400)), pp_group="P", tp_group="D")
    assert "W68" in str(exc.value)


def test_a_partial_hold_that_is_not_the_whole_stays_w74():
    mans = _manifests()
    d0 = mans[1]
    assert d0.group == "D" and d0.rank == 0
    pieces = tuple(_piece(DRAFT, 800, 20480) if p.param_name == DRAFT else p for p in d0.pieces)
    mans[1] = xm.RankManifest(group="D", rank=0, card=0, region_tag=TAG, boot_token="b1",
                              tp_rank=0, pp_rank=0, pieces=pieces)
    with pytest.raises(wx.Weg2XchgSourceMissing) as exc:
        xm.join_manifests(mans, pp_group="P", tp_group="D")
    assert "W74" in str(exc.value)


def test_a_zero_width_is_no_block_not_a_zero_size_block():
    geom = wx.ParamGeom(name=DRAFT, tag=TAG, shard_axis=wx.ROWS, rows_full=1600,
                        cols_full=20480, itemsize=1, stage=0, family=DRAFT,
                        dst_widths=(1600, 0, 0))
    geom.validate()
    layout = wx.GroupLayout(name="D", cards=CARDS, tp_size=3, base=3,
                            family_ratios={DRAFT: (1600, 0, 0)})
    blocks = wx._blocks_of(geom, layout, is_dst=True)
    assert [len(b) for b in blocks] == [1, 0, 0] and blocks[0][0].size == 1600
    src = wx._blocks_of(geom, layout, is_dst=False)
    assert [len(b) for b in src] == [1, 0, 0]


# ---------------------------------------------------------------------------
# xsn390: the draft REGION's own leg -- the shadows have no piece of it
# ---------------------------------------------------------------------------
DTAG = wx.GPU_MEMORY_TYPE_WEIGHTS_DRAFT


def _region_manifests():
    """P: stages 0..2 hold the main weights (whole, one per stage); the LAST
    stage also holds the draft whole. D: every rank a main cut; only rank 0
    (the solo host) holds the draft, whole."""
    out = []
    for r in range(3):
        pieces = [_piece(f"model.layers.{r}.mlp.down_proj.weight", 2048, 1200)]
        if r == 2:
            pieces.append(_piece(DRAFT, 1600, 20480, tag=DTAG))
        out.append(xm.RankManifest(group="P", rank=r, card=r, region_tag=TAG, boot_token="b1",
                                   tp_rank=0, pp_rank=r, pieces=tuple(pieces)))
    for r in range(3):
        pieces = [_piece(f"model.layers.{i}.mlp.down_proj.weight", 2048, 400) for i in range(3)]
        if r == 0:
            pieces.append(_piece(DRAFT, 1600, 20480, tag=DTAG))
        out.append(xm.RankManifest(group="D", rank=r, card=r, region_tag=TAG, boot_token="b1",
                                   tp_rank=r, pp_rank=0, pieces=tuple(pieces)))
    return out


@pytest.mark.parametrize("hook,group,rank_side", [("source", "D", "src_rank"),
                                                   ("wake", "D", "dst_rank")])
def test_the_draft_regions_leg_plans_the_solo_host_with_the_shadows_as_empty_rows(hook, group, rank_side):
    lines = []
    plan, why = xm.leg_plan_from_join(hook=hook, group=group, rank=0,
                                      manifests=_region_manifests(),
                                      region_tag=DTAG, log=lines.append)
    assert plan is not None, why
    descs = [d for d in plan.descs if d.param_name == DRAFT]
    assert descs and {int(getattr(d, rank_side)) for d in descs} == {0}
    assert sum(int(d.nbytes) for d in descs) == 1600 * 20480
    assert any("tp_ranks_without_pieces=[1, 2]" in l for l in lines)


def test_a_shadow_gets_no_draft_leg_and_is_told_so_not_a_crash():
    plan, why = xm.leg_plan_from_join(hook="source", group="D", rank=1,
                                      manifests=_region_manifests(),
                                      region_tag=DTAG, log=lambda *_: None)
    assert plan is None and "no-descriptors-for-rank" in why


def test_a_region_nobody_in_tp_holds_is_still_refused():
    mans = [m for m in _region_manifests()]
    # strip the draft from D rank 0 too: the region has no TP holder at all
    d0 = mans[3]
    mans[3] = xm.RankManifest(group="D", rank=0, card=0, region_tag=TAG, boot_token="b1",
                              tp_rank=0, pp_rank=0,
                              pieces=tuple(p for p in d0.pieces if p.tag != DTAG))
    plan, why = xm.leg_plan_from_join(hook="source", group="D", rank=0, manifests=mans,
                                      region_tag=DTAG, log=lambda *_: None)
    assert plan is None and why
