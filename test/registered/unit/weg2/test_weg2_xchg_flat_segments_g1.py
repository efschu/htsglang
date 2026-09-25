"""G1 (27B line / NF, 2026-09-25): the flat-segment container, as pure geometry.

A fused module stored as ONE byte buffer per rank: segments at 16-byte-aligned
offsets, each with its own row width, one of them fusing three components (a
q|k|v tensor: a TP rank's rows are [q_r | k_r | v_r]), plus a component held
whole on every rank (k/v under replicated KV). Desk only, small synthetic sizes
(no model's numbers): the declaration is built the way a loader lays the buffer
out, the copy plan is executed on bytearrays, and every destination byte is
compared with an oracle built straight from the "checkpoint".
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import xchg_flat_segments as fs  # noqa: E402

ALIGN = 16
TP = 3

#: (segment key, row bytes, [(component key, full rows, per-rank rows)]) --
#: a fused 3-component segment, a plain one with an odd row width (forces pad),
#: a q shard split by rows and a k shard held whole on every rank (replicated KV).
SPEC = [
    ("0+1+2", 24, [("0", 8, (4, 2, 2)), ("1", 4, (2, 1, 1)), ("2", 4, (2, 1, 1))]),
    ("3", 34, [("3", 8, (3, 3, 2))]),
    ("q", 20, [("q", 6, (2, 2, 2))]),
    ("k", 20, [("k", 2, None)]),  # None = whole on every rank
]


def _layout(parts):
    """``[(key, row_bytes, [FlatComponent])]`` -> FlatTable, offsets aligned the
    way the loader aligns them."""
    segs, total = [], 0
    for key, row_bytes, comps in parts:
        total = -(-total // ALIGN) * ALIGN
        seg = fs.FlatSegment(key, total, row_bytes, tuple(comps))
        segs.append(seg)
        total += seg.nbytes
    return fs.FlatTable(nbytes=total, segments=tuple(segs))


def _whole():
    return _layout(
        [
            (k, w, [fs.FlatComponent(c, f, 0, f) for c, f, _ in comps])
            for k, w, comps in SPEC
        ]
    )


def _rank(r):
    parts = []
    for k, w, comps in SPEC:
        cs = []
        for c, f, split in comps:
            if split is None:
                cs.append(fs.FlatComponent(c, f, 0, f))
            else:
                cs.append(fs.FlatComponent(c, f, sum(split[:r]), split[r]))
        parts.append((k, w, cs))
    return _layout(parts)


def _truth(seg, comp, row, byte):
    """The checkpoint: one deterministic byte per (segment, component, row, col)."""
    return (sum(map(ord, seg + comp)) % 97 + 7 * row + 3 * byte) % 251 + 1


def _materialise(table):
    """A rank's buffer as its loader would have written it (pad = 0)."""
    buf = bytearray(table.nbytes)
    for seg in table.segments:
        for comp in seg.components:
            base = seg.component_offset(comp.key)
            for i in range(comp.rows):
                for b in range(seg.row_bytes):
                    buf[base + i * seg.row_bytes + b] = _truth(
                        seg.key, comp.key, comp.row_start + i, b
                    )
    return buf


def _run(copies, fills, src_bufs, dst_bufs):
    for c in copies:
        dst_bufs[c.dst_rank][c.dst_off : c.dst_off + c.nbytes] = src_bufs[c.src_rank][
            c.src_off : c.src_off + c.nbytes
        ]
    for f in fills:
        dst_bufs[f.dst_rank][f.dst_off : f.dst_off + f.nbytes] = bytes(f.nbytes)


def test_the_loader_layout_has_pad_and_the_table_names_it():
    t = _rank(2)
    t.validate("x")
    assert t.pad_ranges(), "odd row widths must leave alignment pad"
    assert (
        sum(n for _o, n in t.pad_ranges()) + sum(s.nbytes for s in t.segments)
        == t.nbytes
    )
    assert all(s.offset % ALIGN == 0 for s in t.segments)


def test_json_round_trip():
    t = _rank(1)
    assert fs.FlatTable.from_json(t.as_json()) == t
    import json

    assert fs.FlatTable.from_json(json.dumps(t.as_json())) == t


@pytest.mark.parametrize(
    "bad",
    [
        lambda t: fs.FlatTable(t.nbytes, t.segments + (t.segments[0],)),  # key twice
        lambda t: fs.FlatTable(
            t.segments[0].nbytes - 1, t.segments[:1]
        ),  # past the end
        lambda t: fs.FlatTable(
            t.nbytes,
            (
                t.segments[0],
                fs.FlatSegment(
                    "z", t.segments[0].offset + 8, 4, (fs.FlatComponent("z", 2, 0, 2),)
                ),
            ),
        ),  # overlap
        lambda t: fs.FlatTable(
            t.nbytes, (fs.FlatSegment("z", 0, 4, (fs.FlatComponent("z", 2, 1, 2),)),)
        ),  # rows outside
    ],
)
def test_a_declaration_that_could_only_be_guessed_is_w68(bad):
    with pytest.raises(wx.Weg2XchgPlanDisagree, match="W68"):
        bad(_rank(0)).validate("x")


def test_join_reads_rows_and_replicated_components():
    j = fs.join_tables("x", _whole(), [_rank(r) for r in range(TP)])
    assert j.axis_of("0+1+2", "1") == fs.COMPONENT_ROWS
    assert j.axis_of("3", "3") == fs.COMPONENT_ROWS
    assert j.axis_of("k", "k") == fs.COMPONENT_REPLICATED


def test_join_refuses_what_is_not_one_container():
    cut = [_rank(r) for r in range(TP)]
    other_width = fs.FlatTable(
        cut[1].nbytes,
        tuple(
            (
                s
                if s.key != "3"
                else fs.FlatSegment(s.key, s.offset, s.row_bytes + 2, s.components)
            )
            for s in cut[1].segments
        ),
    )
    with pytest.raises(wx.Weg2XchgPlanDisagree, match="not the same container"):
        fs.join_tables("x", _whole(), [cut[0], other_width, cut[2]])
    gap = [_rank(0), _rank(0), _rank(2)]  # rank 1 declares rank 0's rows: gap + overlap
    with pytest.raises(wx.Weg2XchgPlanDisagree, match="do not tile"):
        fs.join_tables("x", _whole(), gap)
    with pytest.raises(wx.Weg2XchgPlanDisagree, match="not all"):
        fs.join_tables("x", _rank(0), cut)


@pytest.mark.parametrize("direction", ["pp_to_tp", "tp_to_pp"])
def test_every_destination_byte_lands_where_the_checkpoint_says(direction):
    """PP (one whole holder, rank 1 of 3) <-> TP3, executed on bytearrays."""
    whole, cut = _whole(), [_rank(r) for r in range(TP)]
    pp = [None, whole, None]
    if direction == "pp_to_tp":
        src, dst = pp, cut
    else:
        src, dst = cut, pp
    copies, fills = fs.copy_plan("x", src, dst)
    src_bufs = [None if t is None else _materialise(t) for t in src]
    dst_bufs = [None if t is None else bytearray(b"\xee" * t.nbytes) for t in dst]
    _run(copies, fills, src_bufs, dst_bufs)
    for r, t in enumerate(dst):
        if t is not None:
            assert dst_bufs[r] == _materialise(t), (direction, r)
    # replicated components come from the co-located rank when one holds them
    if direction == "tp_to_pp":
        k_off = whole.by_key()["k"].offset
        assert [c.src_rank for c in copies if c.dst_off == k_off] == [1]


def test_a_destination_row_nobody_holds_is_w74():
    cut = [_rank(r) for r in range(TP)]
    with pytest.raises(wx.Weg2XchgSourceMissing, match="W74"):
        fs.copy_plan("x", [cut[0], None, cut[2]], [None, _whole(), None])


def test_every_copy_is_whole_rows_and_pad_is_never_a_source():
    cut = [_rank(r) for r in range(TP)]
    copies, fills = fs.copy_plan("x", [None, _whole(), None], cut)
    for c in copies:
        seg = [
            s
            for s in cut[c.dst_rank].segments
            if s.offset <= c.dst_off < s.offset + s.nbytes
        ][0]
        assert (
            c.nbytes % seg.row_bytes == 0
            and (c.dst_off - seg.offset) % seg.row_bytes == 0
        )
    pads = {(r, o, n) for r in range(TP) for o, n in cut[r].pad_ranges()}
    assert {(f.dst_rank, f.dst_off, f.nbytes) for f in fills} == pads


def test_shard_keys_are_spelled_the_same_by_every_group():
    assert fs.shard_key((0, 1, 2)) == "0+1+2"
    assert fs.shard_key(3) == "3"
    assert fs.shard_key("q") == "q"
