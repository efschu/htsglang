"""#1378 xsn74 -- every fused column-parallel tensor is placed per component.

The first SEAM-DIGEST verdict ever reached (weg2xsn74, eb61d523): placement
identical (gone=0, arrived=0), 218 of 560 tensors content-changed, the first
eight named being embed_tokens and layer 0's conv1d, in_proj_ba, in_proj_qkvz
and gate_up_proj (+ their INT8 scales). Two defects, one shape:

  * only QKVParallelLinear declared ``component_rows``; gate_up_proj
    ([gate|up]), in_proj_qkvz ([q|k|v|z]), in_proj_ba ([b|a]) and the GDN
    conv1d ([key|key|value]) declared nothing;
  * ``_axis_of`` tried the plain ROWS cut BEFORE the declared components,
    and ``_mixed_fused_axis`` refused any declaration whose components all
    share one axis -- so even a declared all-ratio-split fusion was laid out
    as rank 0's [c0_0|c1_0] followed by rank 1's [c0_1|c1_1], where the whole
    is [c0_0 c0_1 .. | c1_0 c1_1 ..].

Mutants: the shipped reader (QKV only) and the shipped order/predicate.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

HIDDEN = 64
NAME = "model.layers.3.mlp.gate_up_proj.weight"


def _piece(name, rows, cols, *, component_rows=(), tag="weights_0"):
    return xm.ManifestPiece(
        param_name=name, tensor_class=sh.tensor_class(name),
        rows_full=rows, cols_full=cols, itemsize=1, tag=tag,
        nbytes=rows * cols, component_rows=tuple(component_rows),
    )


# --------------------------------------------------------------------------
# 1. the component reader
# --------------------------------------------------------------------------


class _Model:
    def __init__(self, mods):
        self._mods = mods

    def get_submodule(self, path):
        if path not in self._mods:
            raise AttributeError(path)
        return self._mods[path]


def test_reader_declares_merged_layers_conv1d_and_qkv_and_nothing_else():
    gdn = SimpleNamespace(key_dim=6, value_dim=10)
    model = _Model({
        "model.layers.0.self_attn.qkv_proj": SimpleNamespace(
            q_proj_shard_size=8, kv_proj_shard_size=2, v_proj_shard_size=2,
            output_partition_sizes=[8, 2, 2]),
        "model.layers.0.mlp.gate_up_proj": SimpleNamespace(
            output_partition_sizes=[5, 5]),
        "model.layers.1.linear_attn.in_proj_qkvz": SimpleNamespace(
            output_partition_sizes=[6, 6, 10, 10]),
        "model.layers.1.linear_attn.in_proj_ba": SimpleNamespace(
            output_partition_sizes=[3, 3]),
        "model.layers.1.linear_attn": gdn,
        "model.layers.1.linear_attn.conv1d": SimpleNamespace(
            output_partition_sizes=[22]),
        "model.layers.0.mlp.down_proj": SimpleNamespace(
            output_partition_sizes=[64]),
        "model.layers.0.input_layernorm": SimpleNamespace(),
    })
    f = sh._qkv_component_rows
    assert f(model, "model.layers.0.self_attn.qkv_proj.weight") == (8, 2, 2)
    assert f(model, "model.layers.0.mlp.gate_up_proj.weight") == (5, 5)
    assert f(model, "model.layers.0.mlp.gate_up_proj.weight_scale") == (5, 5)
    assert f(model, "model.layers.1.linear_attn.in_proj_qkvz.weight") == (6, 6, 10, 10)
    assert f(model, "model.layers.1.linear_attn.in_proj_ba.weight") == (3, 3)
    assert f(model, "model.layers.1.linear_attn.conv1d.weight") == (6, 6, 10)
    # single-output and non-linear owners declare nothing (plain cut)
    assert f(model, "model.layers.0.mlp.down_proj.weight") == ()
    assert f(model, "model.layers.0.input_layernorm.weight") == ()
    assert f(model, "model.layers.9.nothing.weight") == ()


# --------------------------------------------------------------------------
# 2. the classification: declared all-ROWS components win over the plain cut
# --------------------------------------------------------------------------


def _fused_geometry():
    """whole [gate 4 | up 4]; ranks 2:1:1 -> (2,2) (1,1) (1,1)."""
    whole = _piece(NAME, 8, HIDDEN, component_rows=(4, 4))
    cut = [_piece(NAME, 4, HIDDEN, component_rows=(2, 2)),
           _piece(NAME, 2, HIDDEN, component_rows=(1, 1)),
           _piece(NAME, 2, HIDDEN, component_rows=(1, 1))]
    return whole, cut


def test_a_declared_all_rows_fusion_classifies_mixed_fused():
    whole, cut = _fused_geometry()
    axis, rows_full, cols_full, widths, pad = xm._axis_of(NAME, whole, cut)
    assert axis == wx.MIXED_FUSED
    assert (rows_full, cols_full, widths, pad) == (8, HIDDEN, (4, 2, 2), 0)
    per_component = xm._mixed_fused_axis(whole, cut)
    assert per_component == ((wx.ROWS, 4, (2, 1, 1)), (wx.ROWS, 4, (2, 1, 1)))


def test_without_a_declaration_the_same_rows_are_the_plain_cut():
    """The shipped path (mutant): undeclared -> ROWS, i.e. interleaved."""
    whole = _piece(NAME, 8, HIDDEN)
    cut = [_piece(NAME, 4, HIDDEN), _piece(NAME, 2, HIDDEN), _piece(NAME, 2, HIDDEN)]
    axis, *_ = xm._axis_of(NAME, whole, cut)
    assert axis == wx.ROWS


def test_a_one_component_declaration_is_still_the_plain_cut():
    whole = _piece(NAME, 8, HIDDEN, component_rows=(8,))
    cut = [_piece(NAME, 4, HIDDEN, component_rows=(4,)),
           _piece(NAME, 2, HIDDEN, component_rows=(2,)),
           _piece(NAME, 2, HIDDEN, component_rows=(2,))]
    axis, *_ = xm._axis_of(NAME, whole, cut)
    assert axis == wx.ROWS


def test_the_join_carries_the_fusion_end_to_end():
    whole, cut = _fused_geometry()
    manifests = [
        xm.RankManifest(group="P", rank=0, card=0, region_tag="weights_0",
                        boot_token="b1", tp_rank=0, pp_rank=0, pieces=(whole,)),
    ] + [
        xm.RankManifest(group="D", rank=r, card=r, region_tag="weights_0",
                        boot_token="b1", tp_rank=r, pp_rank=0, pieces=(c,))
        for r, c in enumerate(cut)
    ]
    join = xm.join_manifests(manifests, pp_group="P", tp_group="D")
    assert not join.unsourced
    assert join.by_name[NAME].shard_axis == wx.MIXED_FUSED
