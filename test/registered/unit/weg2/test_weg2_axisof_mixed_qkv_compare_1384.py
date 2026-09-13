# SPDX-License-Identifier: Apache-2.0
"""#1384 follow-up -- the per-component BYTE COMPARE for MIXED_FUSED tensors.

``test_weg2_axisof_mixed_qkv_1384.py`` closed the CLASSIFICATION half:
``_axis_of`` names a fused QKV tensor under ``kv < tp`` as ``MIXED_FUSED``
instead of raising W68. Classifying is not moving bytes -- before this file,
``ParamGeom.validate()`` still refused ``MIXED_FUSED`` outright, so
``plan_from_join``'s ``t.geom(...)`` call raised for EVERY leg that carried
such a tensor and no descriptor, let alone a Stripe compare, was ever built.
A Zwerg boot could not reach ``INJECT verdict=MATCH`` on these tensors --
it could not reach a plan at all.

**WHAT NOW COMPARES.**  ``weight_exchange.ParamGeom`` gains
``component_axes``/``component_rank_rows`` alongside the ``component_rows``
#1384 already added; ``ParamGeom.validate()`` accepts ``MIXED_FUSED`` only
when those three are mutually self-consistent (never a silent pass-through);
``_blocks_of``'s two branches (the PP-form whole-holder and the TP-form cut)
both grow a ``MIXED_FUSED`` case that lays out ONE ``Block`` per declared
component -- REPLICATED components at ``global_start=0`` on every rank,
ROWS components via the declared ``component_rank_rows`` prefix sum -- and
``_emit``'s EXISTING, unmodified block-intersection loop does the rest: it
was already axis-agnostic per block, it only needed blocks that exist.
``xchg_manifest.JoinedTensor``/``.geom()`` carry the resolved per-component
axis and per-rank vectors from ``_mixed_fused_axis`` onward, never
re-deriving them.

**NOTHING IS RE-COMPUTED.**  Every byte offset below traces to a
``component_rows``/``component_rank_rows`` entry that was DECLARED (by
``_qkv_component_rows`` at the real call site, or directly by a test here) --
this file never invents a second formula for where Q ends and K begins.

**THE DANGER DIRECTION, TESTED DIRECTLY.**  A wrongly wired axis does not
refuse -- it PAIRS THE WRONG BYTES and can still read as a match if nobody
checks content. Every mutant below runs the SAME real copy machinery
(``wx.build_plan`` -> ``XchgDesc`` -> a plain ``ctypes.memmove`` executor) and
then asks the one question that matters: does what LANDED equal what a
plain, independent formula says SHOULD have landed? For all three mutants the
answer is demonstrably no.
"""

from __future__ import annotations

import ctypes
import os
import struct

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.distributed import utils as du  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

HIDDEN = 2048
ITEMSIZE = 4          # 4 bytes/elem: keeps every row a whole number of u32 words
ROW_BYTES = HIDDEN * ITEMSIZE
NAME = "model.layers.0.qkv_proj.weight"
TAG = "weights_0"


# ---------------------------------------------------------------------------
# The real geometry (same derivation as test_weg2_axisof_mixed_qkv_1384.py).
# ---------------------------------------------------------------------------


def _real_geometry(*, total_heads, total_kv, tp, head_size):
    du.set_tp_partition_ratios([1] * tp)
    try:
        assert du.attn_kv_replicated(tp, total_kv)
        units = du.attn_q_partition_units(total_heads, total_kv, tp)
        groups = du.attn_q_partition_groups(total_kv, tp)
        q_rows = [
            du.tp_partition_size(total_heads, tp, r, units, groups=groups)
            * head_size
            for r in range(tp)
        ]
    finally:
        du.set_tp_partition_ratios(None)
    kv_row = total_kv * head_size
    return q_rows, kv_row


def _boot_geometry():
    """P=5120, D=[3072,2048,2048] -- the exact BOOT7/weg2zwerg2 numbers."""
    q_rows, kv_row = _real_geometry(total_heads=16, total_kv=2, tp=3, head_size=256)
    whole = (sum(q_rows), kv_row, kv_row)
    cut = [(q, kv_row, kv_row) for q in q_rows]
    assert sum(whole) == 5120
    assert [sum(c) for c in cut] == [3072, 2048, 2048]
    return whole, cut


# ---------------------------------------------------------------------------
# A byte-exact fake device: every row of the source ENCODES its own global
# row index, so any misrouting is a direct, unambiguous readback mismatch --
# no separate "expected content" oracle to keep in sync by hand.
# ---------------------------------------------------------------------------


def _row_pattern(global_row: int) -> bytes:
    word = struct.pack("<I", global_row & 0xFFFFFFFF)
    return word * (ROW_BYTES // 4)


def _alloc(rows: int) -> ctypes.Array:
    return ctypes.create_string_buffer(max(1, rows) * ROW_BYTES)


def _write_identity_rows(buf: ctypes.Array, rows: int) -> None:
    base = ctypes.addressof(buf)
    for r in range(rows):
        ctypes.memmove(base + r * ROW_BYTES, _row_pattern(r), ROW_BYTES)


def _zero(buf: ctypes.Array, rows: int) -> None:
    ctypes.memset(ctypes.addressof(buf), 0, max(1, rows) * ROW_BYTES)


def _read_row_index(buf: ctypes.Array, row: int) -> int:
    raw = ctypes.string_at(ctypes.addressof(buf) + row * ROW_BYTES, 4)
    return struct.unpack("<I", raw)[0]


def _apply_plan(plan: wx.XchgPlan) -> None:
    """The REAL copy primitive, unmodified: FLAT is one memmove, STRIDED2D is
    row-by-row -- exactly what ``_shape_piece`` describes and what a real
    ``cudaMemcpyAsync``/``cudaMemcpy2DAsync`` pair would be handed."""
    for d in plan.descs:
        if d.kind == wx.ZEROFILL:
            ctypes.memset(d.dst_ptr + d.dst_off, 0, d.nbytes)
            continue
        if d.kind == wx.FLAT:
            ctypes.memmove(d.dst_ptr + d.dst_off, d.src_ptr + d.src_off, d.nbytes)
        else:
            for r in range(d.rows):
                ctypes.memmove(d.dst_ptr + d.dst_off + r * d.dpitch,
                                d.src_ptr + d.src_off + r * d.spitch,
                                d.run_bytes)


# ---------------------------------------------------------------------------
# Building geoms and running the plan end to end.
# ---------------------------------------------------------------------------


def _mixed_geom(*, component_axes, component_rows, component_rank_rows,
                stage=0):
    return wx.ParamGeom(
        name=NAME, tag=TAG, shard_axis=wx.MIXED_FUSED,
        rows_full=sum(component_rows), cols_full=HIDDEN, itemsize=ITEMSIZE,
        stage=stage,
        component_rows=tuple(component_rows),
        component_axes=tuple(component_axes),
        component_rank_rows=tuple(tuple(r) for r in component_rank_rows),
    )


def _run(geom, tp: int, *, p_buf, d_bufs, ptr_of=None):
    src_layout = wx.GroupLayout(name="P", cards=tuple(range(tp)), tp_size=1, base=0)
    dst_layout = wx.GroupLayout(name="D", cards=tuple(range(tp)), tp_size=tp,
                                base=tp)

    def default_ptr_of(group, rank, name):
        if group == "P":
            return ctypes.addressof(p_buf)
        return ctypes.addressof(d_bufs[rank])

    plan = wx.build_plan([geom], src_layout, dst_layout, waves=[[TAG]],
                         ptr_of=ptr_of or default_ptr_of)
    _apply_plan(plan)
    return plan


def _expected_global_row(*, component_whole, component_axes, component_rank,
                         rank: int, local_row: int) -> int:
    """THE INDEPENDENT ORACLE -- plain arithmetic, no shared code with
    ``mixed_fused_blocks``. Which GLOBAL P row a D rank's local row must have
    come from, computed straight from the declared sizes."""
    global_prefix = 0
    local_prefix = 0
    for i, axis in enumerate(component_axes):
        whole = component_whole[i]
        if axis == wx.REPLICATED:
            size = whole
        else:
            size = component_rank[i][rank]
        if local_row < local_prefix + size:
            offset = local_row - local_prefix
            if axis == wx.REPLICATED:
                return global_prefix + offset
            rank_start = sum(component_rank[i][:rank])
            return global_prefix + rank_start + offset
        local_prefix += size
        global_prefix += whole
    raise AssertionError((rank, local_row, component_axes))


# ---------------------------------------------------------------------------
# 1. THE GOLDEN PATH -- correct declarations, real functions, byte-exact.
# ---------------------------------------------------------------------------


def test_mixed_fused_copies_the_correct_bytes_per_component():
    whole, cut = _boot_geometry()
    tp = 3
    axes = (wx.ROWS, wx.REPLICATED, wx.REPLICATED)
    rank_rows = tuple(tuple(cut[r][i] for r in range(tp)) for i in range(3))
    geom = _mixed_geom(component_axes=axes, component_rows=whole,
                       component_rank_rows=rank_rows)
    geom.validate()  # #1384 follow-up: this must now PASS, not refuse.

    rows_full = sum(whole)
    p_buf = _alloc(rows_full)
    _write_identity_rows(p_buf, rows_full)
    d_rows = [sum(c) for c in cut]
    d_bufs = [_alloc(r) for r in d_rows]
    for b, r in zip(d_bufs, d_rows):
        _zero(b, r)

    _run(geom, tp, p_buf=p_buf, d_bufs=d_bufs)

    for rank in range(tp):
        for local_row in range(d_rows[rank]):
            want = _expected_global_row(
                component_whole=whole, component_axes=axes,
                component_rank=rank_rows, rank=rank, local_row=local_row)
            got = _read_row_index(d_bufs[rank], local_row)
            assert got == want, (rank, local_row, got, want)


def test_mixed_fused_reverse_direction_tp_to_pp_is_also_correct():
    """The mirror direction: D is the SOURCE, P the DESTINATION. Same geom,
    same declared components -- ``_blocks_of`` must not care which side asked.

    Each D rank's local buffer is filled PER COMPONENT rather than with one
    running index: the Q slice carries a RANK-SPECIFIC base (so a wrong
    source rank for a ROWS component is directly visible), while K and V
    carry a base that is the SAME on every rank (so whichever rank the
    on-card tie-break in ``_pick_source`` happens to choose for a REPLICATED
    component, the expected content at P does not depend on that choice --
    exactly the property #1382's replication is supposed to have).
    """
    whole, cut = _boot_geometry()
    tp = 3
    axes = (wx.ROWS, wx.REPLICATED, wx.REPLICATED)
    rank_rows = tuple(tuple(cut[r][i] for r in range(tp)) for i in range(3))
    geom = _mixed_geom(component_axes=axes, component_rows=whole,
                       component_rank_rows=rank_rows)

    rows_full = sum(whole)
    d_rows = [sum(c) for c in cut]
    d_bufs = [_alloc(r) for r in d_rows]
    q_base, k_base, v_base = 10_000_000, 20_000_000, 30_000_000
    for rank, buf in enumerate(d_bufs):
        q_n, k_n, v_n = rank_rows[0][rank], rank_rows[1][rank], rank_rows[2][rank]
        base = ctypes.addressof(buf)
        for j in range(q_n):
            ctypes.memmove(base + j * ROW_BYTES,
                           _row_pattern(q_base + rank * 1_000 + j), ROW_BYTES)
        for j in range(k_n):
            ctypes.memmove(base + (q_n + j) * ROW_BYTES,
                           _row_pattern(k_base + j), ROW_BYTES)
        for j in range(v_n):
            ctypes.memmove(base + (q_n + k_n + j) * ROW_BYTES,
                           _row_pattern(v_base + j), ROW_BYTES)
    p_buf = _alloc(rows_full)
    _zero(p_buf, rows_full)

    src_layout = wx.GroupLayout(name="D", cards=tuple(range(tp)), tp_size=tp, base=0)
    dst_layout = wx.GroupLayout(name="P", cards=tuple(range(tp)), tp_size=1, base=tp)

    def ptr_of(group, rank, name):
        return ctypes.addressof(d_bufs[rank]) if group == "D" else ctypes.addressof(p_buf)

    plan = wx.build_plan([geom], src_layout, dst_layout, waves=[[TAG]], ptr_of=ptr_of)
    _apply_plan(plan)

    # Q: P's row for global Q offset o must equal the OWNING rank's pattern
    # at its own local Q offset -- a wrong source rank shows up as a
    # different rank's `q_base + rank*1000` prefix.
    prefix = 0
    for r in range(tp):
        for j in range(rank_rows[0][r]):
            assert _read_row_index(p_buf, prefix + j) == q_base + r * 1_000 + j
        prefix += rank_rows[0][r]
    # K, V: RANK-INDEPENDENT content -- correct regardless of which rank was
    # picked as the on-card source.
    for j in range(whole[1]):
        assert _read_row_index(p_buf, prefix + j) == k_base + j
    prefix += whole[1]
    for j in range(whole[2]):
        assert _read_row_index(p_buf, prefix + j) == v_base + j


# ---------------------------------------------------------------------------
# 2. VALIDATE() GATE -- MIXED_FUSED is refused until it truly carries.
# ---------------------------------------------------------------------------


def test_validate_still_refuses_mixed_fused_with_no_declared_components():
    geom = wx.ParamGeom(name=NAME, tag=TAG, shard_axis=wx.MIXED_FUSED,
                        rows_full=5120, cols_full=HIDDEN, itemsize=ITEMSIZE)
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        geom.validate()
    assert "W68" in str(exc.value) and "no declared component_rows" in str(exc.value)


def test_validate_refuses_a_self_inconsistent_component_declaration():
    geom = wx.ParamGeom(name=NAME, tag=TAG, shard_axis=wx.MIXED_FUSED,
                        rows_full=5120, cols_full=HIDDEN, itemsize=ITEMSIZE,
                        component_rows=(4096, 512, 511),  # sums to 5119, not 5120
                        component_axes=(wx.ROWS, wx.REPLICATED, wx.REPLICATED),
                        component_rank_rows=((2048, 1024, 1024),
                                             (512, 512, 512), (511, 511, 511)))
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        geom.validate()
    assert "W68" in str(exc.value) and "sum to" in str(exc.value)


def test_validate_refuses_a_rows_component_whose_rank_vector_disagrees():
    geom = wx.ParamGeom(name=NAME, tag=TAG, shard_axis=wx.MIXED_FUSED,
                        rows_full=5120, cols_full=HIDDEN, itemsize=ITEMSIZE,
                        component_rows=(4096, 512, 512),
                        component_axes=(wx.ROWS, wx.REPLICATED, wx.REPLICATED),
                        # Q's rank vector sums to 4095, not 4096.
                        component_rank_rows=((2048, 1024, 1023),
                                             (512, 512, 512), (512, 512, 512)))
    with pytest.raises(wx.Weg2XchgPlanDisagree) as exc:
        geom.validate()
    assert "W68" in str(exc.value) and "do not sum" in str(exc.value)


def test_validate_after_the_fix_accepts_a_genuinely_consistent_declaration():
    whole, cut = _boot_geometry()
    tp = 3
    rank_rows = tuple(tuple(cut[r][i] for r in range(tp)) for i in range(3))
    geom = _mixed_geom(component_axes=(wx.ROWS, wx.REPLICATED, wx.REPLICATED),
                       component_rows=whole, component_rank_rows=rank_rows)
    geom.validate()  # must not raise


# ---------------------------------------------------------------------------
# 3. MUTANTS -- the danger direction: WRONG bytes paired and never refused.
#    Each of these validates cleanly (self-consistent) and even tiles
#    cleanly (build_plan does not raise) -- the corruption is only visible in
#    the LANDED CONTENT, which is exactly the class of bug a naive "it built
#    a plan" check would miss.
# ---------------------------------------------------------------------------


def test_mutant_a_component_boundary_shifted_by_one_row():
    """One row moved from Q to K, on EVERY side consistently (so validate()
    and the tiling check both pass) -- a boundary that is wrong by exactly
    one row, not a corrupted declaration."""
    whole, cut = _boot_geometry()
    tp = 3
    axes = (wx.ROWS, wx.REPLICATED, wx.REPLICATED)
    rank_rows_correct = tuple(tuple(cut[r][i] for r in range(tp)) for i in range(3))

    # SHIFT: Q loses its last row (goes to the last rank), K gains one row.
    shifted_whole = (whole[0] - 1, whole[1] + 1, whole[2])
    q_ranks = list(rank_rows_correct[0])
    q_ranks[-1] -= 1
    shifted_rank_rows = (tuple(q_ranks), (whole[1] + 1,) * tp, rank_rows_correct[2])
    geom = _mixed_geom(component_axes=axes, component_rows=shifted_whole,
                       component_rank_rows=shifted_rank_rows)
    geom.validate()  # self-consistent -- the bug this class cannot catch

    rows_full = sum(shifted_whole)
    assert rows_full == sum(whole)  # same total, just misdrawn internally
    p_buf = _alloc(rows_full)
    _write_identity_rows(p_buf, rows_full)
    d_rows = [q_ranks[r] + (whole[1] + 1) + whole[2] for r in range(tp)]
    d_bufs = [_alloc(r) for r in d_rows]
    for b, r in zip(d_bufs, d_rows):
        _zero(b, r)
    _run(geom, tp, p_buf=p_buf, d_bufs=d_bufs)

    # THE CORRECT ORACLE uses the TRUE (unshifted) boundary. If the shifted
    # plan produced the same content, the boundary would not matter -- it
    # must NOT: the last rank's Q tail and the K block are now offset by one
    # row relative to the correct semantic content. A rank whose OWN row
    # count already differs from the correct geometry is counted as a
    # mismatch outright (compared only over the overlapping local range,
    # since the oracle has no "correct" answer past the correct extent).
    correct_d_rows = [sum(c) for c in cut]
    mismatches = 0
    for rank in range(tp):
        if d_rows[rank] != correct_d_rows[rank]:
            mismatches += 1
        for local_row in range(min(d_rows[rank], correct_d_rows[rank])):
            want = _expected_global_row(
                component_whole=whole, component_axes=axes,
                component_rank=rank_rows_correct, rank=rank,
                local_row=local_row)
            got = _read_row_index(d_bufs[rank], local_row)
            if got != want:
                mismatches += 1
    assert mismatches > 0, (
        "a one-row boundary shift produced byte-identical content to the "
        "correct geometry -- the mutant is not exercising the boundary")


def test_mutant_b_q_and_kv_regions_swapped():
    """The component ORDER is swapped (K, Q, V instead of Q, K, V) on every
    side consistently -- self-consistent and tiling-clean, wrong content."""
    whole, cut = _boot_geometry()
    tp = 3
    correct_axes = (wx.ROWS, wx.REPLICATED, wx.REPLICATED)
    correct_rank_rows = tuple(tuple(cut[r][i] for r in range(tp)) for i in range(3))

    swapped_whole = (whole[1], whole[0], whole[2])
    swapped_axes = (wx.REPLICATED, wx.ROWS, wx.REPLICATED)
    swapped_rank_rows = (correct_rank_rows[1], correct_rank_rows[0],
                        correct_rank_rows[2])
    geom = _mixed_geom(component_axes=swapped_axes, component_rows=swapped_whole,
                       component_rank_rows=swapped_rank_rows)
    geom.validate()

    rows_full = sum(whole)
    p_buf = _alloc(rows_full)
    _write_identity_rows(p_buf, rows_full)
    d_rows = [sum(c) for c in cut]
    d_bufs = [_alloc(r) for r in d_rows]
    for b, r in zip(d_bufs, d_rows):
        _zero(b, r)
    _run(geom, tp, p_buf=p_buf, d_bufs=d_bufs)

    mismatches = 0
    for rank in range(tp):
        for local_row in range(d_rows[rank]):
            want = _expected_global_row(
                component_whole=whole, component_axes=correct_axes,
                component_rank=correct_rank_rows, rank=rank,
                local_row=local_row)
            got = _read_row_index(d_bufs[rank], local_row)
            if got != want:
                mismatches += 1
    assert mismatches > 0, (
        "swapping the Q and K/V component order produced byte-identical "
        "content to the correct geometry")


def test_mutant_c_replicated_component_treated_as_partitioned():
    """K is genuinely replicated (every rank must hold the FULL block), but
    the geom LABELS it ROWS with a rank vector that partitions it instead --
    self-consistent (the vector sums to the whole) and tiling-clean, but every
    rank now receives only a FRACTION of K instead of the whole block."""
    whole, cut = _boot_geometry()
    tp = 3
    q_rank_rows = tuple(cut[r][0] for r in range(tp))
    # A plausible-looking partition of K's 512 rows across 3 ranks, instead
    # of every rank holding the replicated 512.
    k_partition = (172, 170, 170)
    assert sum(k_partition) == whole[1]
    axes = (wx.ROWS, wx.ROWS, wx.REPLICATED)  # K mislabelled ROWS
    rank_rows = (q_rank_rows, k_partition, tuple(cut[r][2] for r in range(tp)))
    geom = _mixed_geom(component_axes=axes, component_rows=whole,
                       component_rank_rows=rank_rows)
    geom.validate()  # self-consistent by every check this class can run

    rows_full = sum(whole)
    p_buf = _alloc(rows_full)
    _write_identity_rows(p_buf, rows_full)
    # Every rank's LOCAL extent is now smaller (K is split, not replicated).
    d_rows = [q_rank_rows[r] + k_partition[r] + cut[r][2] for r in range(tp)]
    d_bufs = [_alloc(r) for r in d_rows]
    for b, r in zip(d_bufs, d_rows):
        _zero(b, r)
    _run(geom, tp, p_buf=p_buf, d_bufs=d_bufs)

    correct_axes = (wx.ROWS, wx.REPLICATED, wx.REPLICATED)
    correct_rank_rows = tuple(tuple(cut[r][i] for r in range(tp)) for i in range(3))
    # THE OBSERVABLE DEFECT: rank 1 and rank 2 no longer hold 512 rows of K
    # at all -- their own local row COUNT already disagrees with what the
    # correct geometry says this parameter's per-rank size must be.
    correct_d_rows = [sum(c) for c in cut]
    assert d_rows != correct_d_rows, (
        "treating a replicated component as partitioned did not change any "
        "rank's own held byte count -- the mutant has no effect to detect")
    # And for the rows that DO exist, they are wrong content too (K's rank-1
    # local rows hold a slice of K that a correct replica would not call
    # rank 1's own row 0, 1, 2, ... -- they hold ROWS 172..341 of K instead
    # of the full 0..511).
    prefix_local = q_rank_rows[1]
    got0 = _read_row_index(d_bufs[1], prefix_local)
    want0_if_correct = _expected_global_row(
        component_whole=whole, component_axes=correct_axes,
        component_rank=correct_rank_rows, rank=1, local_row=prefix_local)
    assert got0 != want0_if_correct, (
        "rank 1's first K row coincidentally matched the correct replica's "
        "first K row")


# ---------------------------------------------------------------------------
# 4. kv >= tp PINNED BYTE-IDENTICAL AT THE COPY LAYER TOO.
# ---------------------------------------------------------------------------


def test_kv_ge_tp_ordinary_fused_copy_is_byte_identical_untouched_path():
    """The 27B production model's path (kv >= tp): an ORDINARY uniformly-
    ROWS-split fused parameter (``ParamGeom.blocks`` + ``device_block_offsets``
    -- the mechanism #1384 did not touch). ``_blocks_of``'s MIXED_FUSED branch
    is a NEW, EARLIER-returning branch; every other axis's code is the
    original, unedited ``if geom.blocks: ...`` / ``else: [geom.content_units]``
    path. This runs it end to end through the real copy machinery and checks
    the bytes land exactly where the untouched formula says they must."""
    tp = 3
    # A fused (q, k, v)-shaped parameter with NO replication at all -- every
    # block ratio-split the SAME way, e.g. kv=4 >= tp=3 (device_block_offsets'
    # own worked example shape, `in_proj_qkvz`-style).
    output_sizes = (900, 300, 300)  # q, k, v -- each divisible enough for tp=3
    geom = wx.ParamGeom(
        name=NAME, tag=TAG, shard_axis=wx.ROWS,
        rows_full=sum(output_sizes), cols_full=HIDDEN, itemsize=ITEMSIZE,
        stage=0, blocks=output_sizes,
    )
    geom.validate()

    rows_full = sum(output_sizes)
    p_buf = _alloc(rows_full)
    _write_identity_rows(p_buf, rows_full)
    blocks_per_rank = wx.device_block_offsets(output_sizes, None, tp)
    d_rows = [sum(b.size for b in blocks) for blocks in blocks_per_rank]
    d_bufs = [_alloc(r) for r in d_rows]
    for b, r in zip(d_bufs, d_rows):
        _zero(b, r)
    _run(geom, tp, p_buf=p_buf, d_bufs=d_bufs)

    for rank, blocks in enumerate(blocks_per_rank):
        for blk in blocks:
            for j in range(blk.size):
                want = blk.global_start + j
                got = _read_row_index(d_bufs[rank], blk.dev_row + j)
                assert got == want, (rank, blk, j, got, want)
