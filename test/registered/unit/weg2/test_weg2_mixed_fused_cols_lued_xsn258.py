"""weg2xsn258 (17.09.2026) -- W90 Weg2SeamDigestMismatch on the lued checkpoint.

PP1 graded ``moved=52`` content-changed pieces after the first D->P flip:
``gate_up_proj`` 26, ``in_proj_qkvz`` 18, ``qkv_proj`` 8 -- every FUSED
column-parallel class, ``weight_packed`` and ``weight_scale`` alike -- while
``o_proj``/``down_proj`` (row-parallel) matched. Both groups had loaded the
SAME checkpoint.

THE ROOT, read off the desk replay of the boot's own manifests
(``leg_plan_from_join`` over ``phase_manifest_*_weights.json``):

    model.layers.39.qkv_proj.weight_packed   PP1 [320, 57344]
        src_rank=0 STRIDED2D run=114688 dst_off=0
        src_rank=1 STRIDED2D run=57344  dst_off=114688
        src_rank=2 STRIDED2D run=57344  dst_off=172032        (pieces=1 each)

A PLAIN column cut. The lued checkpoint is compressed-tensors
``pack-quantized`` (Marlin): a column-parallel weight is stored ``[K/16,
N*pack]``, the OUTPUT dim -- and with it the q|k|v boundaries -- is the
storage COLUMN axis. D rank 0 holds q[0:6144] k[0:512] v[0:512] (its own
shard of EACH component, [6144,512,512] in ``weight_scale``'s declaration);
the plain cut copied that [q|k|v] shard onto the whole's first N/2 columns:
its q prefix landed right, its k and v landed over the whole's remaining q.

``_mixed_fused_axis`` (#1384) already knows fused components -- for the ROW
axis only (``sum(component_rows) == rows_full``); the packed tensor's
components sum to its COLUMNS, and ``weight_packed`` declared none at all
(``_qkv_component_rows`` stopped at ``.weight``/``.bias``/``.weight_scale``).
gdncov (``int-quantized``, channel, ``[out, in]``) never hit this: its output
dim is the row axis.

THE FIX: ``MIXED_FUSED_COLS`` -- the same declaration, read against the
column extent; ``weight_packed`` declares its components too, scaled by the
pack factor the tensor's own extent states (57344 / 14336 = 4).

THE DANGER DIRECTION, run on the real copy machinery: the old classification
(a plain ``COLS`` geom for the same tensor) lands the wrong bytes and no
refusal fires. The mutant below shows exactly that.
"""

from __future__ import annotations

import ctypes
import os
import struct
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

ITEMSIZE = 4
ROWS = 6                       # K/16 rows of a packed tensor, small
NAME = "model.layers.39.qkv_proj.weight_packed"
TAG = "weights_4"
PACK = 4
# output units (what the module declares): q=48, k=8, v=8 -> whole 64
Q_UNITS, KV_UNITS = 48, 8
# uneven TP3, [1/2, 1/4, 1/4] like the boot: q [24,12,12], k/v [4,2,2]
Q_RANK = [24, 12, 12]
KV_RANK = [4, 2, 2]
TP = 3


def _whole_comp():
    return (Q_UNITS * PACK, KV_UNITS * PACK, KV_UNITS * PACK)


def _rank_comp(r):
    return (Q_RANK[r] * PACK, KV_RANK[r] * PACK, KV_RANK[r] * PACK)


def _piece(rows, cols, comp, tag=TAG, name=NAME):
    return xm.ManifestPiece(
        param_name=name, tensor_class="qkv_proj", rows_full=rows,
        cols_full=cols, itemsize=ITEMSIZE, tag=tag,
        nbytes=rows * cols * ITEMSIZE, component_rows=tuple(comp))


def _whole_piece():
    return _piece(ROWS, sum(_whole_comp()), _whole_comp())


def _cut_pieces():
    return [_piece(ROWS, sum(_rank_comp(r)), _rank_comp(r)) for r in range(TP)]


# ---------------------------------------------------------------------------
# 1. classification
# ---------------------------------------------------------------------------


def test_axis_of_names_a_packed_fused_column_cut_mixed_fused_cols():
    axis, rows_full, cols_full, widths, pad = xm._axis_of(
        NAME, _whole_piece(), _cut_pieces())
    assert axis == wx.MIXED_FUSED_COLS
    assert (rows_full, cols_full) == (ROWS, sum(_whole_comp()))
    assert widths == tuple(sum(_rank_comp(r)) for r in range(TP))
    assert pad == 0


def test_axis_of_without_a_declaration_is_still_the_plain_column_cut():
    """Untouched path: a column cut with NO declared components stays COLS."""
    whole = _piece(ROWS, 256, ())
    cut = [_piece(ROWS, 128, ()), _piece(ROWS, 64, ()), _piece(ROWS, 64, ())]
    axis, *_ = xm._axis_of(NAME, whole, cut)
    assert axis == wx.COLS


def test_mixed_fused_axis_on_cols_resolves_each_component_as_a_split():
    comps = xm._mixed_fused_axis(_whole_piece(), _cut_pieces(), axis="cols")
    assert comps is not None
    axes = [a for a, _w, _r in comps]
    wholes = [w for _a, w, _r in comps]
    ranks = [r for _a, _w, r in comps]
    assert axes == [wx.ROWS, wx.ROWS, wx.ROWS]
    assert tuple(wholes) == _whole_comp()
    assert ranks == [tuple(Q_RANK[r] * PACK for r in range(TP)),
                     tuple(KV_RANK[r] * PACK for r in range(TP)),
                     tuple(KV_RANK[r] * PACK for r in range(TP))]
    # the row reading of the same pieces is NOT a fused row tensor
    assert xm._mixed_fused_axis(_whole_piece(), _cut_pieces()) is None


# ---------------------------------------------------------------------------
# 2. the writer: weight_packed declares, scaled by the pack factor
# ---------------------------------------------------------------------------


def test_qkv_component_rows_accepts_weight_packed():
    owner = types.SimpleNamespace(q_proj_shard_size=Q_UNITS,
                                  kv_proj_shard_size=KV_UNITS,
                                  v_proj_shard_size=KV_UNITS)
    model = types.SimpleNamespace(
        get_submodule=lambda path: owner if path == "model.layers.39.qkv_proj" else (_ for _ in ()).throw(AttributeError(path)))
    assert sh._qkv_component_rows(model, NAME) == (Q_UNITS, KV_UNITS, KV_UNITS)
    assert sh._qkv_component_rows(model, NAME.replace("weight_packed", "weight_scale")) == (Q_UNITS, KV_UNITS, KV_UNITS)
    assert sh._qkv_component_rows(model, NAME.replace("weight_packed", "weight_shape")) == ()


def test_pieces_from_inventory_scales_the_declaration_by_the_pack_factor():
    packed = types.SimpleNamespace(name=NAME, rows_full=ROWS,
                                   cols_full=sum(_whole_comp()), itemsize=ITEMSIZE,
                                   tag=TAG, component_rows=(Q_UNITS, KV_UNITS, KV_UNITS))
    scale = types.SimpleNamespace(name=NAME.replace("weight_packed", "weight_scale"),
                                  rows_full=2, cols_full=Q_UNITS + 2 * KV_UNITS,
                                  itemsize=2, tag=TAG,
                                  component_rows=(Q_UNITS, KV_UNITS, KV_UNITS))
    bf16 = types.SimpleNamespace(name="model.layers.0.qkv_proj.weight",
                                 rows_full=Q_UNITS + 2 * KV_UNITS, cols_full=16,
                                 itemsize=2, tag=TAG,
                                 component_rows=(Q_UNITS, KV_UNITS, KV_UNITS))
    by = {p.param_name: p for p in xm.pieces_from_inventory([packed, scale, bf16])}
    assert by[NAME].component_rows == _whole_comp()               # x PACK
    assert by[scale.name].component_rows == (Q_UNITS, KV_UNITS, KV_UNITS)  # already the column axis
    assert by[bf16.name].component_rows == (Q_UNITS, KV_UNITS, KV_UNITS)   # already the row axis


# ---------------------------------------------------------------------------
# 3. the bytes -- the real plan machinery, a plain memmove executor
# ---------------------------------------------------------------------------


def _cell(row: int, gcol: int) -> bytes:
    return struct.pack("<I", ((row & 0xFF) << 24) | (gcol & 0xFFFFFF))


def _alloc(rows: int, cols: int):
    return ctypes.create_string_buffer(max(1, rows * cols) * ITEMSIZE)


def _fill_identity(buf, rows, cols, gcol_of):
    base = ctypes.addressof(buf)
    for r in range(rows):
        for c in range(cols):
            ctypes.memmove(base + (r * cols + c) * ITEMSIZE, _cell(r, gcol_of(c)), ITEMSIZE)


def _read(buf, rows, cols, r, c):
    raw = ctypes.string_at(ctypes.addressof(buf) + (r * cols + c) * ITEMSIZE, ITEMSIZE)
    v = struct.unpack("<I", raw)[0]
    return v >> 24, v & 0xFFFFFF


def _apply(plan):
    for d in plan.descs:
        if d.kind == wx.ZEROFILL:
            ctypes.memset(d.dst_ptr + d.dst_off, 0, d.nbytes)
        elif d.kind == wx.FLAT:
            ctypes.memmove(d.dst_ptr + d.dst_off, d.src_ptr + d.src_off, d.nbytes)
        else:
            for r in range(d.rows):
                ctypes.memmove(d.dst_ptr + d.dst_off + r * d.dpitch,
                               d.src_ptr + d.src_off + r * d.spitch, d.run_bytes)


def _expected_gcol(rank: int, local_col: int) -> int:
    """THE INDEPENDENT ORACLE: which global column of the whole a D rank's
    local column holds, straight from the declared sizes."""
    gp, lp = 0, 0
    for i in range(3):
        whole = _whole_comp()[i]
        size = _rank_comp(rank)[i]
        if local_col < lp + size:
            start = sum(_rank_comp(rr)[i] for rr in range(rank))
            return gp + start + (local_col - lp)
        lp += size
        gp += whole
    raise AssertionError((rank, local_col))


def _geom_from_join(tp_is_dst: bool):
    manifests = [
        xm.RankManifest(group="P", rank=1, card=1, region_tag="weights",
                        boot_token="t", pieces=(_whole_piece(),)),
    ] + [
        xm.RankManifest(group="D", rank=r, card=r, region_tag="weights",
                        boot_token="t", pieces=(_cut_pieces()[r],))
        for r in range(TP)
    ]
    # the P side names its holder stage through the manifest's pp_rank
    manifests[0] = xm.RankManifest(group="P", rank=1, card=1, region_tag="weights",
                                   boot_token="t", pieces=(_whole_piece(),))
    join = xm.join_manifests(manifests)
    t = [jt for jt in join.tensors if jt.param_name == NAME][0]
    assert t.shard_axis == wx.MIXED_FUSED_COLS
    return t.geom(tp_is_dst=tp_is_dst)


def _layouts():
    p = wx.GroupLayout(name="P", cards=tuple(range(TP)), tp_size=1, base=0)
    d = wx.GroupLayout(name="D", cards=tuple(range(TP)), tp_size=TP, base=TP)
    return p, d


def test_tp_to_pp_lands_every_component_of_every_rank_where_it_belongs():
    """D -> P (the xsn258 leg): each D rank's local [q|k|v] slice lands on
    the whole's q, k and v regions at that rank's position within each."""
    geom = _geom_from_join(tp_is_dst=False)
    p, d = _layouts()
    W = sum(_whole_comp())
    p_buf = _alloc(ROWS, W)
    ctypes.memset(ctypes.addressof(p_buf), 0xEE, ROWS * W * ITEMSIZE)
    d_bufs = []
    for r in range(TP):
        cols = sum(_rank_comp(r))
        b = _alloc(ROWS, cols)
        _fill_identity(b, ROWS, cols, lambda c, r=r: _expected_gcol(r, c))
        d_bufs.append(b)

    def ptr_of(group, rank, name):
        return ctypes.addressof(p_buf) if group == "P" else ctypes.addressof(d_bufs[rank])

    plan = wx.build_plan([geom], d, p, waves=[[TAG]], ptr_of=ptr_of)
    assert len([x for x in plan.descs if x.param_name == NAME]) == 3 * TP, [
        (x.src_rank, x.dst_off, x.run_bytes) for x in plan.descs]
    _apply(plan)
    for r in range(ROWS):
        for gc in range(W):
            row, got = _read(p_buf, ROWS, W, r, gc)
            assert (row, got) == (r, gc), (r, gc, row, got)


def test_pp_to_tp_is_the_mirror():
    geom = _geom_from_join(tp_is_dst=True)
    p, d = _layouts()
    W = sum(_whole_comp())
    p_buf = _alloc(ROWS, W)
    _fill_identity(p_buf, ROWS, W, lambda c: c)
    d_bufs = [_alloc(ROWS, sum(_rank_comp(r))) for r in range(TP)]

    def ptr_of(group, rank, name):
        return ctypes.addressof(p_buf) if group == "P" else ctypes.addressof(d_bufs[rank])

    plan = wx.build_plan([geom], p, d, waves=[[TAG]], ptr_of=ptr_of)
    _apply(plan)
    for rk in range(TP):
        cols = sum(_rank_comp(rk))
        for r in range(ROWS):
            for c in range(cols):
                row, got = _read(d_bufs[rk], ROWS, cols, r, c)
                assert (row, got) == (r, _expected_gcol(rk, c)), (rk, r, c, row, got)


def test_mutant_the_old_plain_column_cut_lands_the_wrong_bytes_silently():
    """THE DANGER DIRECTION: the shipped classification before this fix -- a
    plain COLS geom for the same tensor -- copies rank 0's [q|k|v] onto the
    whole's first columns. It raises nothing and lands k/v bytes over q."""
    p, d = _layouts()
    W = sum(_whole_comp())
    widths = tuple(sum(_rank_comp(r)) for r in range(TP))
    geom = wx.ParamGeom(name=NAME, tag=TAG, shard_axis=wx.COLS, rows_full=ROWS,
                        cols_full=W, itemsize=ITEMSIZE, stage=1,
                        family=NAME)
    geom.validate()
    p_buf = _alloc(ROWS, W)
    ctypes.memset(ctypes.addressof(p_buf), 0xEE, ROWS * W * ITEMSIZE)
    d_bufs = []
    for r in range(TP):
        b = _alloc(ROWS, widths[r])
        _fill_identity(b, ROWS, widths[r], lambda c, r=r: _expected_gcol(r, c))
        d_bufs.append(b)

    def ptr_of(group, rank, name):
        return ctypes.addressof(p_buf) if group == "P" else ctypes.addressof(d_bufs[rank])

    # the source-side widths of a plain COLS cut come from the group ratio
    # vector; seed the same uneven widths the boot had
    d_flat = wx.GroupLayout(name="D", cards=tuple(range(TP)), tp_size=TP, base=TP,
                            family_ratios={NAME: tuple(widths)})
    plan = wx.build_plan([geom], d_flat, p, waves=[[TAG]], ptr_of=ptr_of)
    _apply(plan)
    wrong = 0
    for gc in range(W):
        _row, got = _read(p_buf, ROWS, W, 0, gc)
        wrong += int(got != gc)
    # rank 0's q prefix (24*4 columns) lands right; everything after it is wrong
    assert wrong > 0
    assert all(_read(p_buf, ROWS, W, 0, gc)[1] == gc for gc in range(Q_RANK[0] * PACK))
    assert _read(p_buf, ROWS, W, 0, Q_RANK[0] * PACK)[1] != Q_RANK[0] * PACK


def test_validate_refuses_a_cols_declaration_that_does_not_sum_to_the_columns():
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        wx.ParamGeom(name=NAME, tag=TAG, shard_axis=wx.MIXED_FUSED_COLS,
                     rows_full=ROWS, cols_full=sum(_whole_comp()) + 4,
                     itemsize=ITEMSIZE, stage=1,
                     component_rows=_whole_comp(),
                     component_axes=(wx.ROWS,) * 3,
                     component_rank_rows=tuple(
                         tuple(_rank_comp(r)[i] for r in range(TP)) for i in range(3))
                     ).validate()
    assert "cols_full" in str(exc.value)
