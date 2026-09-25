"""G1 (27B line / NF, 2026-09-25): the flat container through the manifest join and
the real plan machinery -- plus the three GGUF shapes the ordinary classes already
carry (qweight_type REPLICATED, a row-parallel K cut in bytes at block boundaries,
padded vocabulary rows), pinned on the same executor.

RED on b5c7d01614: ``ManifestPiece`` has no ``flat_table``; and without a
declaration a flat container whose byte totals add up by coincidence classifies as
a plain COLS cut that lands the wrong bytes with no refusal (the mutant test keeps
showing that on the old classification).

Desk only, synthetic sizes, a plain memmove executor over ctypes buffers.
"""

from __future__ import annotations

import ctypes
import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import xchg_flat_segments as fs  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

TP = 3
TAG = "weights_4"
NAME = "model.layers.3.mlp.gate_up_proj.qweight"
ALIGN = 16
#: row widths that are multiples of 16 -> no pad, so the byte totals add up
#: (the coincidence that makes the old plain column cut look valid)
SPEC_NOPAD = [("0", 32, [("0", 6, (3, 2, 1))]), ("1", 48, [("1", 6, (2, 2, 2))])]
#: a fused 3-component segment and an odd row width -> pad on every rank
SPEC_PAD = [
    ("0+1+2", 24, [("0", 8, (4, 2, 2)), ("1", 4, (2, 1, 1)), ("2", 4, (2, 1, 1))]),
    ("3", 34, [("3", 8, (3, 3, 2))]),
]


def _layout(parts):
    segs, total = [], 0
    for key, row_bytes, comps in parts:
        total = -(-total // ALIGN) * ALIGN
        seg = fs.FlatSegment(key, total, row_bytes, tuple(comps))
        segs.append(seg)
        total += seg.nbytes
    return fs.FlatTable(nbytes=total, segments=tuple(segs))


def _whole(spec):
    return _layout(
        [(k, w, [fs.FlatComponent(c, f, 0, f) for c, f, _ in cs]) for k, w, cs in spec]
    )


def _rank(spec, r):
    return _layout(
        [
            (k, w, [fs.FlatComponent(c, f, sum(sp[:r]), sp[r]) for c, f, sp in cs])
            for k, w, cs in spec
        ]
    )


def _truth(seg, comp, row, byte):
    return (sum(map(ord, seg + comp)) % 97 + 7 * row + 3 * byte) % 251 + 1


def _image(table):
    buf = bytearray(table.nbytes)
    for seg in table.segments:
        for comp in seg.components:
            base = seg.component_offset(comp.key)
            for i in range(comp.rows):
                for b in range(seg.row_bytes):
                    buf[base + i * seg.row_bytes + b] = _truth(
                        seg.key, comp.key, comp.row_start + i, b
                    )
    return bytes(buf)


def _piece(table, *, declare=True, name=NAME, rows=None, cols=None, itemsize=1):
    rows = 1 if rows is None else rows
    cols = table.nbytes if cols is None else cols
    return xm.ManifestPiece(
        param_name=name,
        tensor_class="gate_up_proj",
        rows_full=rows,
        cols_full=cols,
        itemsize=itemsize,
        tag=TAG,
        nbytes=rows * cols * itemsize,
        flat_table=(table if declare else None),
    )


def _join(whole_piece, cut_pieces):
    manifests = [
        xm.RankManifest(
            group="P",
            rank=1,
            card=1,
            region_tag="weights",
            boot_token="t",
            pieces=(whole_piece,),
        )
    ]
    manifests += [
        xm.RankManifest(
            group="D",
            rank=r,
            card=r,
            region_tag="weights",
            boot_token="t",
            pieces=(cut_pieces[r],),
        )
        for r in range(TP)
    ]
    return xm.join_manifests(manifests)


def _bufs(sizes):
    return [ctypes.create_string_buffer(max(1, n)) for n in sizes]


def _apply(plan):
    for d in plan.descs:
        if d.kind == wx.ZEROFILL:
            ctypes.memset(d.dst_ptr + d.dst_off, 0, d.nbytes)
        elif d.kind == wx.FLAT:
            ctypes.memmove(d.dst_ptr + d.dst_off, d.src_ptr + d.src_off, d.nbytes)
        else:
            for r in range(d.rows):
                ctypes.memmove(
                    d.dst_ptr + d.dst_off + r * d.dpitch,
                    d.src_ptr + d.src_off + r * d.spitch,
                    d.run_bytes,
                )


def _run(join, direction, p_buf, d_bufs):
    def p_addr(name, rank):
        return ctypes.addressof(p_buf) if rank == 1 else None

    def d_addr(name, rank):
        return ctypes.addressof(d_bufs[rank])

    tp_is_dst = direction == wx.LEGS_PP_TO_TP
    plan = xm.plan_from_join(
        join,
        direction=direction,
        waves=[[TAG]],
        src_addr=(p_addr if tp_is_dst else d_addr),
        dst_addr=(d_addr if tp_is_dst else p_addr),
    )
    _apply(plan)
    return plan


# ---------------------------------------------------------------------------
# the manifest
# ---------------------------------------------------------------------------


def test_a_piece_carries_its_declaration_through_json_and_others_stay_byte_identical():
    t = _rank(SPEC_PAD, 1)
    p = _piece(t)
    assert xm.ManifestPiece.from_json(p.as_json()) == p
    plain = xm.ManifestPiece(
        param_name="x.weight",
        tensor_class="x",
        rows_full=2,
        cols_full=3,
        itemsize=2,
        tag=TAG,
        nbytes=12,
    )
    assert "flat_segments" not in plain.as_json()
    assert xm.ManifestPiece.from_json(plain.as_json()).flat_table is None


def test_pieces_from_inventory_reads_the_declaration_off_the_geom():
    t = _rank(SPEC_PAD, 0)
    geom = wx.ParamGeom(
        name=NAME,
        tag=TAG,
        shard_axis=wx.REPLICATED,
        rows_full=1,
        cols_full=t.nbytes,
        itemsize=1,
        flat_table=t,
    )
    (piece,) = xm.pieces_from_inventory([geom])
    assert piece.flat_table == t


# ---------------------------------------------------------------------------
# the join
# ---------------------------------------------------------------------------


def test_the_join_classifies_a_declared_container_flat_segments():
    join = _join(
        _piece(_whole(SPEC_PAD)), [_piece(_rank(SPEC_PAD, r)) for r in range(TP)]
    )
    (t,) = join.tensors
    assert t.shard_axis == wx.FLAT_SEGMENTS
    assert t.tp_widths == tuple(_rank(SPEC_PAD, r).nbytes for r in range(TP))
    g = t.geom(tp_is_dst=True)
    assert g.flat_join is not None and g.dst_widths is None


def test_a_declaration_on_one_side_only_is_w68():
    cut = [_piece(_rank(SPEC_PAD, r)) for r in range(TP)]
    with pytest.raises(wx.Weg2XchgPlanDisagree, match="one side only"):
        _join(_piece(_whole(SPEC_PAD), declare=False), cut)
    cut[2] = _piece(_rank(SPEC_PAD, 2), declare=False)
    with pytest.raises(wx.Weg2XchgPlanDisagree, match="one side only"):
        _join(_piece(_whole(SPEC_PAD)), cut)


def test_a_flat_geom_without_its_joined_declarations_is_w68():
    with pytest.raises(wx.Weg2XchgPlanDisagree, match="FLAT_SEGMENTS without"):
        wx.ParamGeom(
            name=NAME,
            tag=TAG,
            shard_axis=wx.FLAT_SEGMENTS,
            rows_full=1,
            cols_full=64,
            itemsize=1,
        ).validate()


# ---------------------------------------------------------------------------
# the bytes, both directions, on the real plan machinery
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [SPEC_PAD, SPEC_NOPAD], ids=["pad", "nopad"])
@pytest.mark.parametrize("direction", [wx.LEGS_PP_TO_TP, wx.LEGS_TP_TO_PP])
def test_every_byte_lands_where_the_checkpoint_says(spec, direction):
    whole, cut = _whole(spec), [_rank(spec, r) for r in range(TP)]
    join = _join(_piece(whole), [_piece(t) for t in cut])
    p_buf = ctypes.create_string_buffer(whole.nbytes)
    d_bufs = _bufs([t.nbytes for t in cut])
    if direction == wx.LEGS_PP_TO_TP:
        ctypes.memmove(p_buf, _image(whole), whole.nbytes)
        for b, t in zip(d_bufs, cut):
            ctypes.memset(b, 0xEE, t.nbytes)
    else:
        ctypes.memset(p_buf, 0xEE, whole.nbytes)
        for b, t in zip(d_bufs, cut):
            ctypes.memmove(b, _image(t), t.nbytes)
    plan = _run(join, direction, p_buf, d_bufs)
    assert {d.kind for d in plan.descs} <= {wx.FLAT, wx.ZEROFILL}
    if direction == wx.LEGS_PP_TO_TP:
        for b, t in zip(d_bufs, cut):
            assert b.raw[: t.nbytes] == _image(t)
    else:
        assert p_buf.raw[: whole.nbytes] == _image(whole)


def test_mutant_the_old_plain_column_cut_lands_the_wrong_bytes_silently():
    """THE DANGER DIRECTION: undeclared, the pad-free container's byte totals add
    up and the join reads a plain COLS cut -- it plans, it refuses nothing, and
    D rank 0 receives the whole's first bytes instead of its own segments."""
    whole, cut = _whole(SPEC_NOPAD), [_rank(SPEC_NOPAD, r) for r in range(TP)]
    assert sum(t.nbytes for t in cut) == whole.nbytes
    join = _join(_piece(whole, declare=False), [_piece(t, declare=False) for t in cut])
    (t,) = join.tensors
    assert t.shard_axis == wx.COLS
    p_buf = ctypes.create_string_buffer(whole.nbytes)
    ctypes.memmove(p_buf, _image(whole), whole.nbytes)
    d_bufs = _bufs([t.nbytes for t in cut])
    _run(join, wx.LEGS_PP_TO_TP, p_buf, d_bufs)
    assert d_bufs[1].raw[: cut[1].nbytes] != _image(cut[1])


def test_the_live_tensor_must_hold_the_declared_bytes():
    whole, cut = _whole(SPEC_PAD), [_rank(SPEC_PAD, r) for r in range(TP)]
    join = _join(_piece(whole), [_piece(t) for t in cut])
    geom = join.tensors[0].geom(tp_is_dst=True)
    p = wx.GroupLayout(name="P", cards=(0, 1, 2), tp_size=1, base=0)
    d = wx.GroupLayout(name="D", cards=(0, 1, 2), tp_size=TP, base=TP)

    def geom_of(group, rank, name):
        n = (
            whole.nbytes
            if group == "P"
            else cut[rank].nbytes - (16 if rank == 2 else 0)
        )
        return wx.StorageGeom(rows=1, cols=n, pitch=n, itemsize=1)

    with pytest.raises(wx.Weg2XchgPlanDisagree, match="declared container"):
        wx.build_plan([geom], p, d, waves=[[TAG]], geom_of=geom_of)


# ---------------------------------------------------------------------------
# the GGUF shapes the ordinary classes carry -- pinned, not changed
# ---------------------------------------------------------------------------


def _plain_join(name, whole_rc, cut_rc, itemsize=1):
    wp = xm.ManifestPiece(
        param_name=name,
        tensor_class="t",
        rows_full=whole_rc[0],
        cols_full=whole_rc[1],
        itemsize=itemsize,
        tag=TAG,
        nbytes=whole_rc[0] * whole_rc[1] * itemsize,
    )
    cps = [
        xm.ManifestPiece(
            param_name=name,
            tensor_class="t",
            rows_full=r,
            cols_full=c,
            itemsize=itemsize,
            tag=TAG,
            nbytes=r * c * itemsize,
        )
        for r, c in cut_rc
    ]
    return _join(wp, cps)


def test_qweight_type_is_replicated():
    join = _plain_join(
        "model.layers.3.mlp.gate_up_proj.qweight_type", (1, 2), [(1, 2)] * TP
    )
    assert join.tensors[0].shard_axis == wx.REPLICATED


@pytest.mark.parametrize("direction", [wx.LEGS_PP_TO_TP, wx.LEGS_TP_TO_PP])
def test_a_row_parallel_k_cut_in_bytes_at_block_boundaries(direction):
    """[N, K bytes] with K cut per rank on whole quant blocks (34-byte blocks,
    6/3/3 of 12): the ordinary COLS class, bytes exact both ways."""
    n, blk, blocks = 5, 34, (6, 3, 3)
    kb = blk * sum(blocks)
    widths = [blk * b for b in blocks]
    join = _plain_join(
        "model.layers.3.mlp.down_proj.qweight", (n, kb), [(n, w) for w in widths]
    )
    assert join.tensors[0].shard_axis == wx.COLS
    whole = bytes((7 * i + 1) % 251 for i in range(n * kb))
    p_buf = ctypes.create_string_buffer(n * kb)
    d_bufs = _bufs([n * w for w in widths])
    starts = [sum(widths[:r]) for r in range(TP)]
    rank_img = [
        b"".join(
            whole[row * kb + starts[r] : row * kb + starts[r] + widths[r]]
            for row in range(n)
        )
        for r in range(TP)
    ]
    if direction == wx.LEGS_PP_TO_TP:
        ctypes.memmove(p_buf, whole, len(whole))
    else:
        for b, img in zip(d_bufs, rank_img):
            ctypes.memmove(b, img, len(img))
    _run(join, direction, p_buf, d_bufs)
    if direction == wx.LEGS_PP_TO_TP:
        assert [b.raw[: len(img)] for b, img in zip(d_bufs, rank_img)] == rank_img
    else:
        assert p_buf.raw[: len(whole)] == whole


def test_padded_vocabulary_rows_are_zerofill():
    """A quantized vocabulary [V, row bytes]: every TP rank holds
    pad_vocab_size(ceil(V/3)) rows, the checkpoint's first, zero rows after."""
    rb = 18
    v = 2 * xm.vocab_pad_unit() + 2
    per = xm._pad_vocab_size(-(-v // TP))
    join = _plain_join("model.embed_tokens.qweight", (v, rb), [(per, rb)] * TP)
    (t,) = join.tensors
    assert t.shard_axis == wx.ROWS and t.pad_units == per * TP - v
    whole = bytes((5 * i + 3) % 251 for i in range(v * rb))
    p_buf = ctypes.create_string_buffer(v * rb)
    ctypes.memmove(p_buf, whole, len(whole))
    d_bufs = _bufs([per * rb] * TP)
    for b in d_bufs:
        ctypes.memset(b, 0xEE, per * rb)
    plan = _run(join, wx.LEGS_PP_TO_TP, p_buf, d_bufs)
    assert any(d.kind == wx.ZEROFILL for d in plan.descs)
    got = b"".join(b.raw[: per * rb] for b in d_bufs)
    assert got == whole + bytes(per * TP * rb - len(whole))


def test_the_inventory_walk_reads_what_the_loader_declared_on_the_parameter():
    import torch

    from sglang.srt.weg2 import weight_exchange_shadow as sh

    t = _rank(SPEC_PAD, 2)
    param = torch.nn.Parameter(
        torch.zeros(t.nbytes, dtype=torch.uint8), requires_grad=False
    )
    assert sh.declared_flat_table(param) is None
    param.xchg_flat_table = t.as_json()
    assert sh.declared_flat_table(param) == t
    geom = wx.ParamGeom.of(
        param,
        name=NAME,
        tag=TAG,
        shard_axis=wx.REPLICATED,
        shard_total=0,
        stage=0,
        flat_table=sh.declared_flat_table(param),
    )
    assert (geom.rows_full, geom.cols_full, geom.flat_table) == (1, t.nbytes, t)
    param.xchg_flat_table = fs.FlatTable(t.nbytes - 1, t.segments).as_json()
    with pytest.raises(wx.Weg2XchgPlanDisagree, match="W68"):
        sh.declared_flat_table(param)
