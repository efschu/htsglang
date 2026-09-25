"""G1 (27B line / NF, 2026-09-25): the GGUF loader DECLARES its flat container, and
the declaration carries the exchange end to end.

The real code path on CPU tensors: ``GGUFLinearMethod.create_weights``, the
``MergedColumnParallelLinear`` / ``QKVParallelLinear`` GGUF weight-loader branches
(fused multi-slot shard (0, 1, 2) split per component and re-fused, a plain shard,
q/k/v shards), ``_create_flat_weight_param`` (16-byte aligned segments, mixed row
widths, a dense shard cast to the params dtype) -- once at TP1 (the PP holder) and
once per rank at TP3. The declarations go through the inventory, the manifests, the
join and the plan; the descriptors are executed with memmove on the loaders' own
buffers, and every destination byte must equal what that destination's loader wrote.

RED on b5c7d01614 + G1.1/G1.2: the loader declares nothing (no ``xchg_flat_table``).
"""

from __future__ import annotations

import ctypes
import os
import types

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.layers.linear import (  # noqa: E402
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from sglang.srt.layers.quantization import gguf as G  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

TP = 3
TAG = "weights_4"
K = 64  # input elements; a Q8_0 row is K/32*34 = 68 bytes (not 16-aligned)
Q8_0, F16 = 8, 1
Q8_ROW = K // 32 * 34


def _qbytes(rows, salt):
    return torch.tensor(
        [
            [(salt * 31 + r * 7 + b * 3) % 251 + 1 for b in range(Q8_ROW)]
            for r in range(rows)
        ],
        dtype=torch.uint8,
    )


def _dense(rows, salt):
    return torch.tensor(
        [[((salt + r) * K + b) % 97 / 7.0 for b in range(K)] for r in range(rows)],
        dtype=torch.float16,
    )


#: in_proj_qkvz-like: a GGUF tensor fusing q|k|v (shard (0, 1, 2)) and z (3),
#: z shipped DENSE (F16, cast to bf16 at load) beside the Q8_0 fused part
MERGED_SIZES = [12, 6, 6, 9]
MERGED_SHARDS = [
    ((0, 1, 2), Q8_0, lambda: _qbytes(24, 1)),
    (3, F16, lambda: _dense(9, 2)),
]
#: qkv_proj-like: q, k, v as three Q8_0 shards
QKV = dict(q=12, k=6, v=6)


def _merged_layer(tp_size, tp_rank):
    layer = torch.nn.Module()
    method = G.GGUFLinearMethod(G.GGUFConfig([]))
    method.create_weights(
        layer,
        K,
        [s // tp_size for s in MERGED_SIZES],
        K,
        sum(MERGED_SIZES),
        torch.bfloat16,
    )
    fake = types.SimpleNamespace(
        output_sizes=MERGED_SIZES,
        tp_size=tp_size,
        tp_rank=tp_rank,
        tp_units=None,
        tp_family=None,
    )
    for sid, qtype, make in MERGED_SHARDS:
        MergedColumnParallelLinear.weight_loader(
            fake, layer.qweight_type, torch.tensor(qtype, dtype=torch.uint8), sid
        )
        MergedColumnParallelLinear.weight_loader(fake, layer.qweight, make(), sid)
    method._create_flat_weight_param(layer)
    return layer.qweight


def _qkv_layer(tp_size, tp_rank):
    layer = torch.nn.Module()
    method = G.GGUFLinearMethod(G.GGUFConfig([]))
    method.create_weights(
        layer,
        K,
        [n // tp_size for n in QKV.values()],
        K,
        sum(QKV.values()),
        torch.bfloat16,
    )
    fake = types.SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        tp_units=None,
        tp_family=None,
        tp_q_groups=None,
        num_kv_head_replicas=1,
    )
    for salt, (sid, rows) in enumerate(QKV.items()):
        QKVParallelLinear.weight_loader(
            fake, layer.qweight_type, torch.tensor(Q8_0, dtype=torch.uint8), sid
        )
        QKVParallelLinear.weight_loader(
            fake, layer.qweight, _qbytes(rows, 10 + salt), sid
        )
    method._create_flat_weight_param(layer)
    return layer.qweight


@pytest.mark.parametrize("build", [_merged_layer, _qkv_layer], ids=["merged", "qkv"])
def test_the_loader_declares_the_container_it_laid_out(build):
    """RED before G1.3: no declaration on the flat parameter."""
    whole = build(1, 0)
    table = sh.declared_flat_table(whole)
    assert table is not None and table.nbytes == whole.numel()
    views = [(off, rows, dim1) for off, rows, dim1 in whole.shard_offset_map.values()]
    assert sorted((s.offset, s.rows) for s in table.segments) == sorted(
        (off, rows) for off, rows, _ in views
    )
    for r in range(TP):
        t = sh.declared_flat_table(build(TP, r))
        assert all(s.offset % 16 == 0 for s in t.segments)


def test_the_fused_shard_is_three_row_ranges_per_rank():
    t = sh.declared_flat_table(_merged_layer(TP, 1))
    fused = t.by_key()["0+1+2"]
    assert [(c.key, c.full_rows, c.row_start, c.rows) for c in fused.components] == [
        ("0", 12, 4, 4),
        ("1", 6, 2, 2),
        ("2", 6, 2, 2),
    ]
    assert fused.row_bytes == Q8_ROW
    assert t.by_key()["3"].row_bytes == K * 2  # the dense shard, cast to bf16


def _manifest(group, rank, name, param):
    geom = wx.ParamGeom.of(
        param,
        name=name,
        tag=TAG,
        shard_axis=wx.REPLICATED,
        shard_total=0,
        stage=rank,
        flat_table=sh.declared_flat_table(param),
    )
    return xm.RankManifest(
        group=group,
        rank=rank,
        card=rank,
        region_tag="weights",
        boot_token="t",
        pieces=xm.pieces_from_inventory([geom]),
    )


def _apply(plan):
    for d in plan.descs:
        if d.kind == wx.ZEROFILL:
            ctypes.memset(d.dst_ptr + d.dst_off, 0, d.nbytes)
        else:
            assert d.kind == wx.FLAT, d
            ctypes.memmove(d.dst_ptr + d.dst_off, d.src_ptr + d.src_off, d.nbytes)


@pytest.mark.parametrize("build", [_merged_layer, _qkv_layer], ids=["merged", "qkv"])
@pytest.mark.parametrize("direction", [wx.LEGS_PP_TO_TP, wx.LEGS_TP_TO_PP])
def test_the_exchange_reproduces_every_loader_byte(build, direction):
    name = "model.layers.2.linear_attn.in_proj_qkvz.qweight"
    whole = build(1, 0)
    ranks = [build(TP, r) for r in range(TP)]
    manifests = [_manifest("P", 1, name, whole)] + [
        _manifest("D", r, name, p) for r, p in enumerate(ranks)
    ]
    join = xm.join_manifests(manifests)
    assert join.tensors[0].shard_axis == wx.FLAT_SEGMENTS
    want = {
        ("P", 1): bytes(whole.numpy()),
        **{("D", r): bytes(p.numpy()) for r, p in enumerate(ranks)},
    }
    if direction == wx.LEGS_PP_TO_TP:
        for p in ranks:
            p.data.fill_(0xEE)
    else:
        whole.data.fill_(0xEE)

    def p_addr(n, rank):
        return whole.data_ptr() if rank == 1 else None

    def d_addr(n, rank):
        return ranks[rank].data_ptr()

    tp_is_dst = direction == wx.LEGS_PP_TO_TP
    plan = xm.plan_from_join(
        join,
        direction=direction,
        waves=[[TAG]],
        src_addr=(p_addr if tp_is_dst else d_addr),
        dst_addr=(d_addr if tp_is_dst else p_addr),
    )
    _apply(plan)
    if tp_is_dst:
        for r, p in enumerate(ranks):
            assert bytes(p.numpy()) == want[("D", r)], r
    else:
        assert bytes(whole.numpy()) == want[("P", 1)]


def test_an_undeclared_shard_publishes_no_table():
    """A shard the loader did not declare (or declared rows that are not the
    rows it holds) leaves the container undeclared -- refused downstream by
    name, never guessed."""
    p = _merged_layer(TP, 0)
    shard_id = list(p.shard_offset_map)
    tensors = [
        torch.zeros(rows, dim1, dtype=torch.uint8)
        for _off, rows, dim1 in p.shard_offset_map.values()
    ]
    offsets = [off for off, _r, _d in p.shard_offset_map.values()]
    q = types.SimpleNamespace(xchg_src_rows={})
    assert (
        G._flat_container_declaration(q, shard_id, tensors, offsets, p.numel()) is None
    )
    q = types.SimpleNamespace(xchg_src_rows={k: v for k, v in p.xchg_src_rows.items()})
    first = next(iter(q.xchg_src_rows))
    q.xchg_src_rows[first] = tuple(
        (c[0], c[1], c[2], c[3] + 1) for c in q.xchg_src_rows[first]
    )
    assert (
        G._flat_container_declaration(q, shard_id, tensors, offsets, p.numel()) is None
    )
