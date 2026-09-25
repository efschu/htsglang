"""27B FP8 (Qwen/Qwen3.8-27B-FP8, block 128x128) on the weg2 flip -- the simple
variant (--fp8-uniform-marlin, default off). Desk, no GPU.

Pinned here:
* the launcher decision: FP8 under the exchange without the flag is REFUSED
  (W160, the 5090 and the 3080s would hold two layouts of one tensor); the flag
  puts SGLANG_FORCE_FP8_MARLIN on BOTH groups; non-FP8 checkpoints and the
  default argv/env stay byte-identical;
* a non-incumbent checkpoint without a P-cut calibration record gets a NAMED line
  (argv_p would hand it the INT8 incumbent vector unasked);
* the Marlin lock workspace stays on the module: the exchange coverage books it
  as local_scratch and the wake re-zeroes it (NF mechanism, adopted; pinned in
  test_weg2_local_scratch_coverage_0924.py -- the FP8-only private registry of
  fdade8572f is gone);
* the FP8-Marlin byte geometry is the one the exchange already moves for
  compressed-tensors pack-quantized (weg2xsn258): ``.weight`` int32 [K/16, 4N]
  declares its fused components scaled by the pack factor, ``.weight_scale``
  [K/128, N] on the column axis, both classify MIXED_FUSED_COLS; a row-parallel
  weight is a plain ROWS cut.
"""

from __future__ import annotations

import json
import os
import types

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.weg2 import launcher as L  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

FP8_VARS = ("SGLANG_FORCE_FP8_MARLIN",)


def _ckpt(tmp_path, quant, nested=False):
    d = tmp_path / ("ckpt_%s_%s" % (quant or "none", int(nested)))
    d.mkdir()
    cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {"num_hidden_layers": 64}}
    if quant:
        qc = {"quant_method": quant}
        if quant == "fp8":
            qc.update(fmt="e4m3", weight_block_size=[128, 128], activation_scheme="dynamic")
        if nested:
            cfg["text_config"]["quantization_config"] = qc
        else:
            cfg["quantization_config"] = qc
    (d / "config.json").write_text(json.dumps(cfg))
    return str(d)


# -- A. the launcher decision ----------------------------------------------------


def test_checkpoint_quant_method_reads_top_level_and_text_config(tmp_path):
    assert L.checkpoint_quant_method(_ckpt(tmp_path, "fp8")) == "fp8"
    assert L.checkpoint_quant_method(_ckpt(tmp_path, "fp8", nested=True)) == "fp8"
    assert L.checkpoint_quant_method(_ckpt(tmp_path, "compressed-tensors")) == "compressed-tensors"
    assert L.checkpoint_quant_method(_ckpt(tmp_path, None)) == ""
    assert L.checkpoint_quant_method(str(tmp_path / "missing")) == ""


def test_fp8_under_the_exchange_without_the_flag_is_refused_by_name():
    with pytest.raises(L.Weg2LaunchRefused) as e:
        L.fp8_layout_decision("fp8", L.WEIGHT_SOURCE_EXCHANGE, False)
    msg = str(e.value)
    assert msg.startswith("W160 Weg2Fp8LayoutRefused")
    assert "--fp8-uniform-marlin" in msg and "weight_scale_inv" in msg


def test_the_flag_puts_one_layout_on_every_rank():
    env, line = L.fp8_layout_decision("fp8", L.WEIGHT_SOURCE_EXCHANGE, True)
    assert env == {v: "1" for v in FP8_VARS}
    assert "FP8-UNIFORM-MARLIN on" in line and all(v in line for v in FP8_VARS)


def test_ring_needs_no_forcing_and_non_fp8_is_untouched():
    env, line = L.fp8_layout_decision("fp8", "ring", False)
    assert env == {} and "native per-card" in line
    assert L.fp8_layout_decision("compressed-tensors", L.WEIGHT_SOURCE_EXCHANGE, False) == ({}, None)
    assert L.fp8_layout_decision("", L.WEIGHT_SOURCE_EXCHANGE, False) == ({}, None)
    env, line = L.fp8_layout_decision("compressed-tensors", L.WEIGHT_SOURCE_EXCHANGE, True)
    assert env == {} and "inert" in line


def test_default_argv_and_env_are_byte_identical(tmp_path, monkeypatch):
    for v in FP8_VARS:
        monkeypatch.delenv(v, raising=False)
    base = ["--tree", "/t", "--tag", "t"]
    ns = L.build_parser().parse_args(base)
    assert ns.fp8_uniform_marlin is False
    assert L._env_knobs(ns)["fp8_uniform_marlin"] is False
    env = L.build_env("tree", "venv", "0", "/tmp/store", False, "tag", group="P")
    assert not any(v in env for v in FP8_VARS)
    # the flag on an INT8 checkpoint stays inert, also in the env
    ns_int8 = L.build_parser().parse_args(base + ["--model", _ckpt(tmp_path, "compressed-tensors"), "--fp8-uniform-marlin"])
    assert L._env_knobs(ns_int8)["fp8_uniform_marlin"] is False


def test_the_flag_on_an_fp8_checkpoint_reaches_both_groups(tmp_path):
    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--model", _ckpt(tmp_path, "fp8"), "--fp8-uniform-marlin"])
    knobs = L._env_knobs(ns)
    assert knobs["fp8_uniform_marlin"] is True
    for group in ("P", "D"):
        env = L.build_env("tree", "venv", "0", "/tmp/store", False, "tag", group=group,
                          fp8_uniform_marlin=knobs["fp8_uniform_marlin"])
        assert all(env[v] == "1" for v in FP8_VARS), group


def test_uncalibrated_non_incumbent_checkpoint_is_named(monkeypatch):
    monkeypatch.setattr(L.host_ledger, "checkpoint_digest", lambda m: ("f" * 64, ""))
    monkeypatch.setattr(L.host_ledger, "read_pp_calibration", lambda dg: (None, "no PP calibration for model fff"))
    line = L.p_cut_calibration_line("/m", "fp8")
    assert line.startswith("WEG2 P-CUT UNCALIBRATED fp8 checkpoint ffffffffffff")
    assert "pcut_refit.py" in line and "--pp-cut-stage-fit" in line
    # the incumbent family and a checkpoint WITH a record print nothing
    assert L.p_cut_calibration_line("/m", "compressed-tensors") is None
    monkeypatch.setattr(L.host_ledger, "read_pp_calibration", lambda dg: ({"measured_counts": [42, 11, 11]}, ""))
    assert L.p_cut_calibration_line("/m", "fp8") is None


# -- B. the Marlin lock workspace -------------------------------------------------


def test_the_workspace_stays_on_the_module_and_the_walk_books_it_as_scratch():
    model = torch.nn.Module()
    model.layers = torch.nn.ModuleList([torch.nn.Module()])
    lin = torch.nn.Linear(8, 8, bias=False)
    model.layers[0].mlp = lin
    lin.workspace = torch.zeros(170, dtype=torch.int)  # prepare_fp8_layer_for_marlin's plain attribute
    kinds = {t.name: t.kind for t in wx.walk_live_tensors(model)}
    assert kinds.get("layers.0.mlp.workspace") == wx.ATTRIBUTE
    assert wx.is_local_scratch("layers.0.mlp.workspace")
    assert "SGLANG_FP8_MARLIN_PRIVATE_WORKSPACE" not in L.FP8_UNIFORM_MARLIN_ENV


# -- C. the FP8-Marlin byte geometry in the exchange ------------------------------

PACK = 4                     # four fp8 per int32 in the Marlin weight
K16 = 5120 // 16             # K/16 rows of the Marlin weight
KB = 5120 // 128             # K/128 rows of the group scales
Q, KV = 6144, 512            # qkv_proj output units (uneven TP3 below)
Q_R, KV_R = [3072, 1536, 1536], [256, 128, 128]
W_NAME = "model.layers.3.self_attn.qkv_proj.weight"
S_NAME = "model.layers.3.self_attn.qkv_proj.weight_scale"


def _piece(name, rows, cols, comp, item):
    return xm.ManifestPiece(param_name=name, tensor_class=sh.tensor_class(name), rows_full=rows,
                            cols_full=cols, itemsize=item, tag="weights_0", nbytes=rows * cols * item,
                            component_rows=tuple(comp))


def test_fp8_marlin_leaf_names_are_known_classes():
    assert sh.tensor_class(W_NAME) == "qkv_proj"
    assert sh.tensor_class(S_NAME) == "qkv_proj"


def test_the_fused_declaration_is_scaled_by_the_pack_factor_on_the_weight_only():
    w = types.SimpleNamespace(name=W_NAME, rows_full=K16, cols_full=(Q + 2 * KV) * PACK, itemsize=4,
                              tag="weights_0", component_rows=(Q, KV, KV))
    s = types.SimpleNamespace(name=S_NAME, rows_full=KB, cols_full=Q + 2 * KV, itemsize=2,
                              tag="weights_0", component_rows=(Q, KV, KV))
    by = {p.param_name: p for p in xm.pieces_from_inventory([w, s])}
    assert by[W_NAME].component_rows == (Q * PACK, KV * PACK, KV * PACK)
    assert by[S_NAME].component_rows == (Q, KV, KV)


@pytest.mark.parametrize("name,rows,pack,item", [(W_NAME, K16, PACK, 4), (S_NAME, KB, 1, 2)])
def test_weight_and_scales_classify_mixed_fused_cols(name, rows, pack, item):
    whole = _piece(name, rows, (Q + 2 * KV) * pack, (Q * pack, KV * pack, KV * pack), item)
    cut = [_piece(name, rows, (Q_R[r] + 2 * KV_R[r]) * pack,
                  (Q_R[r] * pack, KV_R[r] * pack, KV_R[r] * pack), item) for r in range(3)]
    axis, rows_full, cols_full, widths, pad = xm._axis_of(name, whole, cut)
    assert axis == wx.MIXED_FUSED_COLS
    assert (rows_full, cols_full, pad) == (rows, (Q + 2 * KV) * pack, 0)


def test_a_row_parallel_fp8_marlin_weight_is_a_plain_rows_cut():
    name = "model.layers.3.self_attn.o_proj.weight"
    k_r = [2560 // 16, 1280 // 16, 1280 // 16]          # K split across TP3, N = hidden 5120
    whole = _piece(name, sum(k_r), 5120 * PACK, (), 4)
    cut = [_piece(name, k_r[r], 5120 * PACK, (), 4) for r in range(3)]
    axis, *_ = xm._axis_of(name, whole, cut)
    assert axis == wx.ROWS


# -- D. a P-cut fit never crosses checkpoints -------------------------------------


def test_p_log_model_path_reads_the_first_server_args_line(tmp_path):
    log = tmp_path / "boot_weg2_x.P.log"
    log.write_text("[t] hello\n[t] server_args=ServerArgs(model_path='/a/b', x=1)\n"
                   "[t] server_args=ServerArgs(model_path='/c/d', x=1)\n")
    assert L.p_log_model_path(str(log)) == "/a/b"
    assert L.p_log_model_path(str(tmp_path / "missing.P.log")) == ""


def test_a_stage_fit_from_another_existing_checkpoint_is_refused(tmp_path, monkeypatch):
    int8 = _ckpt(tmp_path, "compressed-tensors")
    fp8 = _ckpt(tmp_path, "fp8")
    log = tmp_path / "boot_weg2_weg2xsnT_abc_0924_000000.P.log"
    log.write_text("[t] server_args=ServerArgs(model_path='%s', chunked_prefill_size=512, "
                   "pp_stage_ratio=[42, 11, 11], pp_attn_stage_ratio=[10, 3, 3], x=1)\n" % int8)
    from sglang.srt.planner import pgap_stage_fit as fit

    monkeypatch.setattr(fit, "read_pgap_log", lambda p: types.SimpleNamespace(chunk_tokens=512))
    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--model", fp8, "--pp-cut-stage-fit", str(log)])
    cards = [L.Card(1, "GPU-a", "NVIDIA GeForce RTX 5090", 32607),
             L.Card(0, "GPU-b", "NVIDIA GeForce RTX 3080", 20480),
             L.Card(2, "GPU-c", "NVIDIA GeForce RTX 3080", 20480)]
    with pytest.raises(L.Weg2LaunchRefused) as e:
        L.stage_fit_family_cost(ns, cards, 512, [400.0, 230.0, 230.0], lambda *_: None)
    assert "W40" in str(e.value) and "does not transfer between checkpoints" in str(e.value)
