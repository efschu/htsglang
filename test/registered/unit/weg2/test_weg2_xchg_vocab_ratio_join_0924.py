# SPDX-License-Identifier: Apache-2.0
"""``--rank-vocab-ratio`` on weg2 group D: the manifest join must take the
RATIO-WEIGHTED padded vocabulary cut (27B line, xsn437 geometry).

RED on 5a3f533f0f (the xsn437 boot tree): ``xchg_manifest._axis_of``
recognised a padded cut only when every rank holds EXACTLY
``pad_vocab_size(ceil(full / n))`` rows -- the EVEN split (weg2xsn23, 3 x 82816
for a 248320 vocabulary).  Under ``--rank-vocab-ratio 58,25,25`` group D's
loader (``VocabParallelEmbedding``: padding ``lcm(64, 3) = 192`` under the
active uneven-TP plan, ``partition_units(1294, [58, 25, 25])``) holds
``[133440, 57600, 57408]`` rows of ``lm_head.weight``,
``model.embed_tokens.weight`` and ``model.embed_tokens.weight_scale``, and the
join raised ``W68 ... nor a PADDED cut``.  ``leg_plan_from_join`` turns that
into ``unjoinable``, and under ``--weg2-xchg-inject authoritative`` the first
wake raises ``W4 Weg2WakeRefused`` -- the boot would die at its first flip.

THE DANGER DIRECTION is unchanged: a genuine disagreement must never be read
as padding.  The ratio-weighted cut is recognised only as the tree's OWN
arithmetic -- every width a whole number of the loader's padding unit, the sum
exactly the loader's padded vocabulary, the surplus below one unit -- and only
on the tensors the loader pads (the vocabulary classes).  A three-row skew, a
surplus, a shortfall and an ordinary tensor that happens to sit on the unit
grid all still refuse.
"""

from __future__ import annotations

import math
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

#: xsn437's measured geometry (phase_manifest_*_weights.json of that boot,
#: 2026-09-24 21:06-21:07Z): P holds the vocabulary whole, embed_tokens (INT8
#: + FP32 per-row scale) on PP0 and lm_head (BF16) on PP2; D is TP3.
VOCAB_FULL = 248320
HIDDEN = 5120
CARDS = (0, 1, 2)
TAG = "weights"
A1_RATIO = (58, 25, 25)
A1_WIDTHS = (133440, 57600, 57408)

#: (param_name, cols, itemsize, P stage holding it whole)
VOCAB_TENSORS = (
    ("lm_head.weight", HIDDEN, 2, 2),
    ("model.embed_tokens.weight", HIDDEN, 1, 0),
    ("model.embed_tokens.weight_scale", 1, 4, 0),
)


def _loader_widths(full, ratios, n=3):
    """The D loader's own per-rank vocab widths, from the tree's functions.

    ``VocabParallelEmbedding.__init__``: under an active uneven-TP plan the
    padding unit is ``lcm(DEFAULT_VOCAB_PADDING_SIZE, tp)`` and the explicit
    vocab vector splits the PADDED vocabulary in those units
    (``partition_units``).  Recomputed from the tree, never restated.
    """
    from sglang.srt.distributed.utils import partition_units
    from sglang.srt.layers.vocab_parallel_embedding import (
        DEFAULT_VOCAB_PADDING_SIZE,
        pad_vocab_size,
    )

    unit = math.lcm(DEFAULT_VOCAB_PADDING_SIZE, n)
    padded = pad_vocab_size(full, unit)
    return tuple(u * unit for u in partition_units(padded // unit, list(ratios)))


def _piece(name, rows, cols, item, tag=TAG):
    from sglang.srt.weg2 import weight_exchange_shadow as sh

    return xm.ManifestPiece(param_name=name, tensor_class=sh.tensor_class(name),
                            rows_full=rows, cols_full=cols, itemsize=item,
                            tag=tag, nbytes=rows * cols * item)


def _manifests(widths=A1_WIDTHS, full=VOCAB_FULL, tensors=VOCAB_TENSORS,
               extra_p=(), extra_d=()):
    """P: the whole tensor on its stage.  D: ``widths`` rows per rank."""
    p = {r: [] for r in range(3)}
    d = {r: [] for r in range(3)}
    for name, cols, item, stage in tensors:
        p[stage].append(_piece(name, full, cols, item))
        for r, w in enumerate(widths):
            d[r].append(_piece(name, w, cols, item))
    for stage, piece in extra_p:
        p[stage].append(piece)
    for r, piece in extra_d:
        d[r].append(piece)
    out = []
    for r in range(3):
        out.append(xm.RankManifest(group="P", rank=r, card=CARDS[r],
                                   region_tag=TAG, boot_token="a1",
                                   tp_rank=0, pp_rank=r, pieces=tuple(p[r])))
    for r in range(3):
        out.append(xm.RankManifest(group="D", rank=r, card=CARDS[r],
                                   region_tag=TAG, boot_token="a1",
                                   tp_rank=r, pp_rank=0, pieces=tuple(d[r])))
    return out


def _join(**kw):
    return xm.join_manifests(_manifests(**kw), pp_group="P", tp_group="D")


def _plan(join, direction):
    return xm.plan_from_join(join, direction=direction,
                             src_addr=lambda n, r: 0x7000_0000,
                             dst_addr=lambda n, r: 0x1000)


# ---------------------------------------------------------------------------
# (1) the geometry is the loader's, and the pad sits in the LAST rank's tail
# ---------------------------------------------------------------------------

def test_the_a1_widths_are_the_loaders_own_partition():
    assert _loader_widths(VOCAB_FULL, A1_RATIO) == A1_WIDTHS
    assert sum(A1_WIDTHS) == 248448 and sum(A1_WIDTHS) - VOCAB_FULL == 128


def test_the_loader_puts_the_pad_in_the_last_ranks_tail():
    """Prefix-sum slices of the PADDED vocabulary, as in the even split: rank
    0 and 1 hold only real rows, rank 2 holds the 128 pad rows at its end.
    That is the layout ``_emit``'s pad branch (ZEROFILL past content) assumes."""
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    padded = sum(A1_WIDTHS)
    starts = []
    for r in range(3):
        idx = VocabParallelEmbedding._get_indices(
            padded, padded, VOCAB_FULL, VOCAB_FULL, r, 3,
            padded_org_sizes=list(A1_WIDTHS))
        starts.append(idx.padded_org_vocab_start_index)
        assert idx.num_elements_padded == A1_WIDTHS[r]
        real = idx.org_vocab_end_index - idx.org_vocab_start_index
        assert real == (A1_WIDTHS[r] if r < 2 else A1_WIDTHS[r] - 128)
    assert starts == [0, 133440, 191040]


# ---------------------------------------------------------------------------
# (2) THE RED ONE: the ratio-weighted padded cut joins
# ---------------------------------------------------------------------------

def test_the_ratio_weighted_vocab_cut_joins_as_a_padded_row_cut():
    join = _join()
    for name, cols, _item, stage in VOCAB_TENSORS:
        t = join.by_name[name]
        assert t.shard_axis == wx.ROWS, name
        assert t.tp_widths == A1_WIDTHS, name
        assert t.rows_full == 248448 and t.pad_units == 128, name
        assert t.cols_full == cols, name
        assert t.pp_stage == stage, name
    assert "padded=3" in join.line()


@pytest.mark.parametrize("ratio", [(2, 1, 1), (5, 2, 2), (1, 2, 1), (58, 25, 25)])
def test_any_vector_the_loader_can_produce_joins(ratio):
    """The join reads what the loader DID; the vector itself is not needed."""
    widths = _loader_widths(VOCAB_FULL, ratio)
    assert len(set(widths)) > 1
    t = _join(widths=widths).by_name["lm_head.weight"]
    assert t.tp_widths == widths and t.pad_units == sum(widths) - VOCAB_FULL


def test_the_even_padded_cut_is_unchanged():
    t = _join(widths=(82816,) * 3).by_name["lm_head.weight"]
    assert t.shard_axis == wx.ROWS and t.tp_widths == (82816,) * 3
    assert t.rows_full == 248448 and t.pad_units == 128


# ---------------------------------------------------------------------------
# (3) the plan: pad is ZEROFILL on rank 2 only, content moves exactly, both ways
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,cols,item,stage", VOCAB_TENSORS)
def test_p_to_d_moves_the_content_and_zerofills_only_rank2s_tail(name, cols, item, stage):
    join = _join()
    geom = join.by_name[name].geom(tp_is_dst=True)
    assert geom.content_units == VOCAB_FULL and geom.pad_units == 128
    plan = _plan(join, wx.LEGS_PP_TO_TP)
    row = cols * item
    moved = {r: 0 for r in range(3)}
    zero = {r: 0 for r in range(3)}
    for d in plan.raw_descs:
        if d.param_name != name:
            continue
        if d.kind == wx.ZEROFILL:
            assert d.src_ptr is None and d.src_rank == -1
            zero[d.dst_rank] += int(d.nbytes)
        else:
            assert d.src_rank == stage, (d.src_rank, stage)
            moved[d.dst_rank] += int(d.nbytes)
    assert zero == {0: 0, 1: 0, 2: 128 * row}
    assert moved == {0: 133440 * row, 1: 57600 * row, 2: 57280 * row}


@pytest.mark.parametrize("name,cols,item,stage", VOCAB_TENSORS)
def test_d_to_p_reassembles_exactly_the_checkpoint_rows(name, cols, item, stage):
    """The mirror: P receives 248320 rows, rank 2's pad rows are never read."""
    join = _join()
    plan = _plan(join, wx.LEGS_TP_TO_PP)
    row = cols * item
    got = {r: 0 for r in range(3)}
    for d in plan.raw_descs:
        if d.param_name != name:
            continue
        assert d.kind != wx.ZEROFILL, "P has no pad rows to fill"
        assert d.dst_rank == stage
        got[d.src_rank] += int(d.nbytes)
    assert got == {0: 133440 * row, 1: 57600 * row, 2: 57280 * row}
    assert sum(got.values()) == VOCAB_FULL * row


# ---------------------------------------------------------------------------
# (4) THE DANGER DIRECTION: nothing that is not the loader's rounding joins
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("widths,why", [
    ((133440 + 3, 57600 - 3, 57408), "three-row skew off the unit grid"),
    ((133440 + 64, 57600 - 64, 57408), "one DEFAULT pad unit, not the TP3 unit"),
    ((133440 + 192, 57600, 57408), "one unit of surplus over the rounding"),
    ((133440, 57600, 57408 - 192), "a shortfall: real rows missing"),
    ((133440, 57600, 0), "an empty rank the loader never produces"),
])
def test_only_the_loaders_rounding_counts_as_ratio_padding(widths, why):
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        _join(widths=widths, tensors=VOCAB_TENSORS[:1])
    assert "lm_head.weight" in str(exc.value), why
    assert "RATIO-WEIGHTED" in str(exc.value), why


def test_an_ordinary_tensor_on_the_unit_grid_is_never_vocab_padding():
    """THE MUTANT this guard is built against: a qkv-shaped tensor whose D rows
    happen to be multiples of 192 summing to pad(5120, 192) = 5184.  The
    arithmetic alone would call it padded; it is not a vocabulary tensor, the
    loader never pads it, and the join must refuse."""
    victim = "model.layers.3.self_attn.qkv_proj.weight"
    assert all(w % 192 == 0 for w in (2112, 1536, 1536))
    assert 2112 + 1536 + 1536 == 5184
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        _join(tensors=(), extra_p=[(0, _piece(victim, 5120, 512, 1))],
              extra_d=[(0, _piece(victim, 2112, 512, 1)),
                       (1, _piece(victim, 1536, 512, 1)),
                       (2, _piece(victim, 1536, 512, 1))])
    assert victim in str(exc.value)


def test_the_vocab_classes_are_read_from_the_class_rule_not_hand_listed():
    """The guard keys on ``weight_exchange_shadow.tensor_class`` -- the same
    class the manifest stores -- so embed_tokens' scale tensor is in it."""
    from sglang.srt.weg2 import weight_exchange_shadow as sh

    for name, *_ in VOCAB_TENSORS:
        assert sh.tensor_class(name) in xm.VOCAB_PADDED_CLASSES, name
    assert sh.tensor_class("model.layers.0.mlp.down_proj.weight") not in (
        xm.VOCAB_PADDED_CLASSES)


# ---------------------------------------------------------------------------
# (5) a2: --rank-mlp-ratio needs NO join change (exact cuts, any widths)
# ---------------------------------------------------------------------------

def test_the_a2_mlp_widths_join_as_exact_cuts():
    """``--rank-mlp-ratio 81,28,27`` on the INT8 target (unit 64 -> 272 units):
    [10368, 3584, 3456].  gate_up is the declared two-component fused cut,
    down_proj the column cut, the per-output scale of down_proj a replica."""
    from sglang.srt.distributed.utils import partition_units

    inter = 17408
    widths = tuple(u * 64 for u in partition_units(inter // 64, [81, 28, 27]))
    assert widths == (10368, 3584, 3456)

    def fused(rows_each, cols, item):
        p = _piece("model.layers.0.mlp.gate_up_proj.weight", 2 * rows_each,
                   cols, item)
        return xm.ManifestPiece(**{**p.as_json(),
                                   "component_rows": (rows_each, rows_each)})

    gu = "model.layers.0.mlp.gate_up_proj.weight"
    dn = "model.layers.0.mlp.down_proj.weight"
    dns = "model.layers.0.mlp.down_proj.weight_scale"
    extra_p = [(0, fused(inter, HIDDEN, 1)),
               (0, _piece(dn, HIDDEN, inter, 1)),
               (0, _piece(dns, HIDDEN, 1, 4))]
    extra_d = []
    for r, w in enumerate(widths):
        extra_d += [(r, fused(w, HIDDEN, 1)), (r, _piece(dn, HIDDEN, w, 1)),
                    (r, _piece(dns, HIDDEN, 1, 4))]
    # No vocabulary tensors in this join: a2 is judged on its own (it joined
    # before the ratio-padded case existed and must keep joining after it).
    join = _join(tensors=(), extra_p=extra_p, extra_d=extra_d)
    assert join.by_name[gu].shard_axis == wx.MIXED_FUSED
    assert join.by_name[dn].shard_axis == wx.COLS
    assert join.by_name[dn].tp_widths == widths
    assert join.by_name[dns].shard_axis == wx.REPLICATED
