"""27B line, model format (2): RadixArk Qwen3.8-27B-NVFP4 on the weg2 flip. Desk, no GPU.

The checkpoint (HF RadixArk, rev 319f741cce): ModelOpt MIXED_PRECISION -- FP8
per-tensor (attention q/k/v/o, GDN in_proj_qkv/in_proj_z/out_proj), NVFP4 g16
(MLP gate/up/down, lm_head), BF16 elsewhere (embedding, in_proj_ba, MTP head,
vision), KV FP8. Pinned here, gap by gap:

A  ModelOptFp8LinearMethod gets its Marlin branch -- ADOPTED from upstream
   sglang #31340 (7de8792758): use_marlin = SGLANG_FORCE_FP8_MARLIN or
   can_auto_enable_marlin_fp8(), prepare_fp8_layer_for_marlin, the Marlin apply;
   plus the fork's contract flag (marlin_packable_linear) on ModelOptFp8Config.
B  ModelOptMixedPrecisionConfig exposes weight_block_size (lcm of its listed
   algorithms' blocks: [128, 128] for RadixArk) and marlin_packable_linear, so
   the uneven-TP split lands on NVFP4's groups and Marlin's tiles.
C  the Marlin repacks of both ModelOpt methods run under weg2 H39 (the
   weg2xsn441 class: load residue in the flip tags), survivors born back in the
   tag pool; the workspace stays ON the module (NF local-scratch, previous
   commit) -- the NVFP4 lm_head predicate needs it there, DFlash2 included.
D  the launcher's --fp8-uniform-marlin covers quant_method modelopt: W160 under
   the exchange without it; with it SGLANG_FORCE_FP8_MARLIN=1 plus
   --fp4-gemm-backend marlin on both groups.
E  the exchange geometry of NVFP4-on-Marlin: fused gate_up weight/scales
   MIXED_FUSED_COLS (pack 2 on the weight), down_proj ROWS, global scale
   REPLICATED, and the Marlin-packed vocab-parallel lm_head as a PADDED COLS cut
   read in vocab units (W68 before).
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import types

import pytest
import torch

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.layers import linear as LIN  # noqa: E402
from sglang.srt.layers import logits_processor as LP  # noqa: E402
from sglang.srt.layers.quantization import marlin_utils_fp4 as MU4  # noqa: E402
from sglang.srt.layers.quantization import modelopt_quant as MQ  # noqa: E402
from sglang.srt.managers import weg2_memory_saver as MS  # noqa: E402
from sglang.srt.weg2 import launcher as L  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402
from sglang.srt.weg2 import weight_exchange_shadow as sh  # noqa: E402
from sglang.srt.weg2 import xchg_manifest as xm  # noqa: E402

RADIXARK = "/spinning/llm_stuff/club-3090/models-cache/Qwen3.8-27B-NVFP4-RadixArk"


def _recording_outside(state):
    @contextlib.contextmanager
    def fake(reason: str = "", *, into: str = "load"):
        state.setdefault("calls", []).append((reason, into))
        state["inside"] = state.get("inside", 0) + 1
        try:
            yield True
        finally:
            state["inside"] -= 1

    return fake


# -- A. ModelOptFp8LinearMethod: upstream #31340 -----------------------------------


def _fp8_method(monkeypatch, *, cuda=True, auto=False, force=False):
    monkeypatch.setattr(MQ, "is_cuda", lambda: cuda)
    monkeypatch.setattr(MQ, "can_auto_enable_marlin_fp8", lambda *a, **k: auto)
    monkeypatch.setattr(MQ, "cutlass_fp8_supported", lambda *a, **k: False)
    monkeypatch.setattr(MQ, "is_sm100_supported", lambda *a, **k: False)
    if force:
        monkeypatch.setenv("SGLANG_FORCE_FP8_MARLIN", "1")
    else:
        monkeypatch.delenv("SGLANG_FORCE_FP8_MARLIN", raising=False)
    return MQ.ModelOptFp8LinearMethod(MQ.ModelOptFp8Config(is_checkpoint_fp8_serialized=True))


def test_the_fp8_marlin_decision_is_upstreams(monkeypatch):
    assert _fp8_method(monkeypatch).use_marlin is False
    assert _fp8_method(monkeypatch, auto=True).use_marlin is True          # sm80..88
    assert _fp8_method(monkeypatch, force=True).use_marlin is True         # the flip's force
    assert _fp8_method(monkeypatch, cuda=False, force=True).use_marlin is False
    assert MQ.ModelOptFp8Config.marlin_packable_linear is True


def _fp8_layer(method, out_sizes=(256, 128, 128), k=256):
    layer = torch.nn.Module()
    method.create_weights(layer, k, list(out_sizes), k, sum(out_sizes), torch.bfloat16,
                          weight_loader=lambda *a, **kw: None)
    layer.weight.data.copy_(torch.randn(sum(out_sizes), k).clamp(-4, 4).to(torch.float8_e4m3fn))
    layer.weight_scale.data.copy_(torch.tensor([0.5, 0.25, 0.25]))
    layer.input_scale.data.copy_(torch.tensor([0.1, 0.2, 0.3]))
    return layer


def _cpu_scaled_fp8_quant(monkeypatch):
    """requantize_with_max_scale calls the CUDA scaled_fp8_quant; a CPU twin."""
    from sglang.srt.layers.quantization import utils as QU

    def quant(x, scale):
        return (x.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale

    monkeypatch.setattr(QU, "scaled_fp8_quant", quant)


def test_the_marlin_branch_repacks_after_the_requantisation_and_drops_the_input_scale(monkeypatch):
    _cpu_scaled_fp8_quant(monkeypatch)
    method = _fp8_method(monkeypatch, force=True)
    layer = _fp8_layer(method)
    assert layer.orig_dtype == torch.bfloat16          # upstream: create_weights records it
    seen = {}

    def fake_prepare(lyr, size_k_first=True, *, born_in=None):
        seen.update(shape=tuple(lyr.weight.shape), size_k_first=size_k_first, born_in=born_in,
                    input_scale=float(lyr.input_scale))
        lyr.workspace = torch.zeros(68, dtype=torch.int)

    monkeypatch.setattr(MQ, "prepare_fp8_layer_for_marlin", fake_prepare)
    monkeypatch.delenv("SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL", raising=False)
    method.process_weights_after_loading(layer)
    # [K, N] after requantize_with_max_scale's transpose, per upstream
    assert seen["shape"] == (256, 512) and seen["size_k_first"] is True
    assert seen["born_in"] is None and abs(seen["input_scale"] - 0.3) < 1e-6
    assert not hasattr(layer, "input_scale")          # Marlin: activations unquantised
    src = inspect.getsource(MQ.ModelOptFp8LinearMethod.apply)
    assert "torch.ops.sglang.apply_fp8_marlin_linear(" in src and "workspace=layer.workspace" in src


def test_under_h39_the_fp8_pass_steps_out_and_hands_the_survivor_hook_down(monkeypatch):
    _cpu_scaled_fp8_quant(monkeypatch)
    method = _fp8_method(monkeypatch, force=True)
    layer = _fp8_layer(method)
    state, seen = {}, {}
    monkeypatch.setattr(MS, "outside_tag_pool", _recording_outside(state))

    def fake_prepare(lyr, size_k_first=True, *, born_in=None):
        seen.update(inside=state.get("inside", 0) > 0, born_in=born_in)

    monkeypatch.setattr(MQ, "prepare_fp8_layer_for_marlin", fake_prepare)
    monkeypatch.setenv("SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL", "1")
    method.process_weights_after_loading(layer)
    assert seen == {"inside": True, "born_in": MS.back_into_tag_pool}
    assert state["calls"] == [("modelopt-fp8-marlin", "load")]


# -- B. the mixed config aligns the uneven split -----------------------------------


def _mixed(quantized_layers):
    return MQ.ModelOptMixedPrecisionConfig.from_config({
        "quantization": {"quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": "FP8",
                         "quantized_layers": quantized_layers, "exclude_modules": []},
        "packed_modules_mapping": {"gate_up_proj": ["gate_proj", "up_proj"]},
    })


def test_nvfp4_plus_fp8_aligns_on_the_nvfp4_block_and_folds_marlin():
    cfg = _mixed({"model.layers.0.mlp.gate_proj": {"quant_algo": "NVFP4", "group_size": 16},
                  "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"}})
    assert cfg.weight_block_size == [128, 128]
    assert cfg.marlin_packable_linear is True
    # gate_up output (17408 per shard) and down input: the SAME 136 units of 128
    assert LIN._quant_block_aligned_units(17408, 17408, cfg, 0) == 136
    assert LIN._quant_block_aligned_units(17408, 17408, cfg, 1) == 136
    # head-granular families pass through (GDN value heads of 384 elements)
    assert LIN._quant_block_aligned_units(6144, 16, cfg, 1) == 16


def test_an_fp8_only_mixed_export_exposes_no_block_and_marlin_still_folds():
    cfg = _mixed({"model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"}})
    assert cfg.weight_block_size is None
    tile = LIN._marlin_uneven_tp_block()
    assert LIN._quant_block_aligned_units(17408, 17408, cfg, 0) == 17408 // tile


@pytest.mark.skipif(not os.path.isfile(os.path.join(RADIXARK, "hf_quant_config.json")),
                    reason="RadixArk checkpoint absent")
def test_the_radixark_export_itself():
    with open(os.path.join(RADIXARK, "hf_quant_config.json")) as fh:
        hf = json.load(fh)
    from sglang.srt.models.qwen3_5 import Qwen3_5ForCausalLM

    hf["packed_modules_mapping"] = dict(Qwen3_5ForCausalLM.packed_modules_mapping)
    cfg = MQ.ModelOptMixedPrecisionConfig.from_config(hf)
    assert cfg.weight_block_size == [128, 128]
    expect = {"linear_attn.in_proj_qkvz": "FP8", "linear_attn.in_proj_ba": None,
              "linear_attn.out_proj": "FP8", "mlp.gate_up_proj": "NVFP4",
              "mlp.down_proj": "NVFP4"}
    for pre in ("model.layers.0.", "model.language_model.layers.0."):
        for mod, algo in expect.items():
            assert cfg.resolve_quant_algo(pre + mod) == algo, (pre, mod)
    assert cfg.resolve_quant_algo("model.layers.3.self_attn.qkv_proj") == "FP8"
    assert cfg.resolve_quant_algo("lm_head") == "NVFP4"
    assert cfg.resolve_quant_algo("mtp.layers.0.mlp.gate_up_proj") is None   # MTP stays BF16


# -- C. H39 for the NVFP4 Marlin repack; the workspace stays on the module ------------


def _fp4_layer(N=128, K=256):
    layer = torch.nn.Module()
    layer.output_size_per_partition, layer.input_size_per_partition = N, K
    layer.params_dtype = torch.bfloat16
    layer.weight = torch.nn.Parameter(torch.zeros(N, K // 2, dtype=torch.uint8), requires_grad=False)
    layer.weight_scale = torch.nn.Parameter(
        torch.ones(N, K // 16).to(torch.float8_e4m3fn), requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
    return layer


def test_the_nvfp4_survivors_are_born_under_the_hook_and_the_checkpoint_dies_first(monkeypatch):
    depth = {"n": 0}
    rec = []

    @contextlib.contextmanager
    def born_in():
        depth["n"] += 1
        try:
            yield True
        finally:
            depth["n"] -= 1

    layer = _fp4_layer()

    def fake_ws(device, max_blocks_per_sm=1):
        rec.append(("workspace", depth["n"] > 0))
        return torch.zeros(68, dtype=torch.int)

    def fake_repack(b_q_weight, perm, size_k, size_n, num_bits):
        rec.append(("repack", depth["n"] > 0, layer.weight is None))
        return torch.zeros(size_k // 16, size_n * 2, dtype=torch.int32)

    real_scales = MU4.nvfp4_marlin_process_scales

    def spy_scales(s):
        rec.append(("scales", depth["n"] > 0, layer.weight_scale is None))
        return real_scales(s)

    monkeypatch.setattr(MU4, "marlin_make_workspace", fake_ws)
    monkeypatch.setattr(MU4, "gptq_marlin_repack", fake_repack, raising=False)
    monkeypatch.setattr(MU4, "nvfp4_marlin_process_scales", spy_scales)
    MU4.prepare_nvfp4_layer_for_marlin(layer, born_in=born_in)
    assert ("workspace", True) in rec
    assert ("repack", True, True) in rec            # born inside, after the checkpoint weight died
    assert ("scales", False, True) in rec           # computed outside, ckpt scales already gone
    assert layer.weight.dtype == torch.int32 and tuple(layer.weight.shape) == (256 // 16, 128 * 2)
    assert layer.weight.numel() * 4 == 128 * 256 // 2      # == the uint8 checkpoint bytes
    assert tuple(layer.weight_scale.shape) == (256 // 16, 128)
    assert layer.workspace is not None                     # ON the module


def test_without_the_hook_the_order_is_the_old_one(monkeypatch):
    layer = _fp4_layer()
    alive = []
    monkeypatch.setattr(MU4, "marlin_make_workspace", lambda device, m=1: torch.zeros(68, dtype=torch.int))

    def fake_repack(b_q_weight, perm, size_k, size_n, num_bits):
        alive.append(layer.weight is not None)
        return torch.zeros(size_k // 16, size_n * 2, dtype=torch.int32)

    monkeypatch.setattr(MU4, "gptq_marlin_repack", fake_repack, raising=False)
    MU4.prepare_nvfp4_layer_for_marlin(layer)
    assert alive == [True]


def test_under_h39_the_nvfp4_repack_steps_out(monkeypatch):
    method = MQ.ModelOptFp4LinearMethod(
        MQ.ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16))
    layer = torch.nn.Module()
    method.create_weights(layer, 256, [128], 256, 128, torch.bfloat16,
                          weight_loader=lambda *a, **kw: None)
    layer.input_scale.data.fill_(0.5)
    layer.weight_scale_2.data.fill_(0.25)
    state, seen = {}, {}
    monkeypatch.setattr(MS, "outside_tag_pool", _recording_outside(state))
    monkeypatch.setattr(MQ, "get_fp4_gemm_runner_backend",
                        lambda: types.SimpleNamespace(is_marlin=lambda: True))

    def fake_prepare(lyr, *, born_in=None):
        seen.update(inside=state.get("inside", 0) > 0, born_in=born_in)

    monkeypatch.setattr(MQ, "prepare_nvfp4_layer_for_marlin", fake_prepare)
    monkeypatch.setenv("SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL", "1")
    method.process_weights_after_loading(layer)
    assert seen == {"inside": True, "born_in": MS.back_into_tag_pool}
    assert state["calls"] == [("modelopt-nvfp4-marlin", "load")]
    assert float(layer.weight_global_scale) == pytest.approx(0.25)


def test_every_marlin_repack_of_the_27b_flip_formats_runs_under_h39():
    from sglang.srt.layers.quantization import fp8
    from sglang.srt.layers.quantization.compressed_tensors.schemes import (
        compressed_tensors_wNa16 as wna16,
    )

    for fn in (fp8.Fp8LinearMethod.process_weights_after_loading,
               MQ.ModelOptFp8LinearMethod.process_weights_after_loading,
               MQ.ModelOptFp4LinearMethod.process_weights_after_loading,
               wna16.CompressedTensorsWNA16.process_weights_after_loading):
        src = inspect.getsource(fn)
        if "_weg2_marlin_outside_pool()" in src:  # Fp8LinearMethod gates through a helper
            src += inspect.getsource(fp8.Fp8LinearMethod._weg2_marlin_outside_pool)
        assert "SGLANG_WEG2_DENSE_REPACK_OUTSIDE_POOL" in src or "dense_repack_outside_pool_armed" in src, fn
        assert "outside_tag_pool" in src, fn


def test_the_nvfp4_marlin_lm_head_is_computed_by_its_quant_method_incl_dflash2():
    qm = object.__new__(MQ.ModelOptFp4LinearMethod)
    head = torch.nn.Module()
    head.weight = torch.nn.Parameter(torch.zeros(16, 32, dtype=torch.int32), requires_grad=False)
    head.weight_scale = torch.zeros(16, 16)
    head.weight_global_scale = torch.zeros(1)
    head.input_size_per_partition, head.output_size_per_partition = 256, 16
    head.workspace = torch.zeros(68, dtype=torch.int)
    head.quant_method = qm
    assert LP.should_apply_lm_head_quant_method(head, qm) is True
    from sglang.srt.speculative import dflash_worker_v2 as DF

    assert DF._resolve_lm_head_compute(head, "t") == (None, qm)
    # the reason the workspace must stay ON the module (the dropped FP8 registry
    # took it off): without it the Marlin head is not recognised
    del head.workspace
    assert LP.should_apply_lm_head_quant_method(head, qm) is False


# -- D. the launcher decision for ModelOpt ------------------------------------------


def _ckpt(tmp_path, qc=None, hf=None):
    d = tmp_path / ("ckpt_%d" % len(list(tmp_path.iterdir())))
    d.mkdir()
    cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": {"num_hidden_layers": 64}}
    if qc:
        cfg["quantization_config"] = qc
    (d / "config.json").write_text(json.dumps(cfg))
    if hf:
        (d / "hf_quant_config.json").write_text(json.dumps(hf))
    return str(d)


def test_the_modelopt_family_is_read_from_config_or_hf_quant_config(tmp_path):
    assert L.checkpoint_quant_method(_ckpt(tmp_path, {"quant_method": "modelopt",
                                                      "quant_algo": "MIXED_PRECISION"})) == "modelopt"
    assert L.checkpoint_quant_method(_ckpt(tmp_path, None, {"quantization": {"quant_algo": "NVFP4"}})) == "modelopt"
    assert L.checkpoint_quant_method(_ckpt(tmp_path)) == ""


def test_modelopt_under_the_exchange_needs_the_flag_and_gets_both_kernels_forced():
    with pytest.raises(L.Weg2LaunchRefused) as e:
        L.fp8_layout_decision("modelopt", L.WEIGHT_SOURCE_EXCHANGE, False)
    assert str(e.value).startswith("W160 Weg2Fp8LayoutRefused") and "NVFP4" in str(e.value)
    env, line = L.fp8_layout_decision("modelopt", L.WEIGHT_SOURCE_EXCHANGE, True)
    assert env == {"SGLANG_FORCE_FP8_MARLIN": "1"}
    assert "modelopt checkpoint" in line and "--fp4-gemm-backend marlin" in line
    env, line = L.fp8_layout_decision("modelopt", "ring", False)
    assert env == {} and "MODELOPT native per-card" in line
    assert L.uniform_marlin_argv("modelopt", True) == ["--fp4-gemm-backend", "marlin"]
    assert L.uniform_marlin_argv("modelopt", False) == []
    assert L.uniform_marlin_argv("fp8", True) == []


def test_the_fp4_backend_reaches_both_groups_once_and_a_contrary_pin_is_refused():
    argv = L.uniform_marlin_argv("modelopt", True)
    assert L.with_uniform_marlin_argv("--max-running-requests=2", argv) == \
        "--max-running-requests=2 --fp4-gemm-backend marlin"
    assert L.with_uniform_marlin_argv("", argv) == "--fp4-gemm-backend marlin"
    once = L.with_uniform_marlin_argv("--fp4-gemm-backend marlin", argv)
    assert once == "--fp4-gemm-backend marlin"
    assert L.with_uniform_marlin_argv("--fp4-gemm-backend=marlin", argv) == "--fp4-gemm-backend=marlin"
    with pytest.raises(L.Weg2LaunchRefused):
        L.with_uniform_marlin_argv("--fp4-gemm-backend cutlass", argv)
    assert L.with_uniform_marlin_argv("--x 1", []) == "--x 1"


def test_the_env_knob_covers_modelopt(tmp_path):
    ck = _ckpt(tmp_path, {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION"})
    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--model", ck, "--fp8-uniform-marlin"])
    assert L._env_knobs(ns)["fp8_uniform_marlin"] is True
    ns = L.build_parser().parse_args(["--tree", "/t", "--tag", "t", "--model", ck])
    assert L._env_knobs(ns)["fp8_uniform_marlin"] is False


# -- E. the exchange geometry of NVFP4 on Marlin --------------------------------------

K16 = 5120 // 16
I = 17408
I_R = [9344, 4096, 3968]            # 73/32/31 units of 128 (uneven TP3)
GU_W = "model.layers.0.mlp.gate_up_proj.weight"
GU_S = "model.layers.0.mlp.gate_up_proj.weight_scale"


def _piece(name, rows, cols, comp, item):
    return xm.ManifestPiece(param_name=name, tensor_class=sh.tensor_class(name), rows_full=rows,
                            cols_full=cols, itemsize=item, tag="weights_0", nbytes=rows * cols * item,
                            component_rows=tuple(comp))


def test_the_fused_gate_up_weight_declares_pack_two_and_the_scales_pack_one():
    w = types.SimpleNamespace(name=GU_W, rows_full=K16, cols_full=2 * I * 2, itemsize=4,
                              tag="weights_0", component_rows=(I, I))
    s = types.SimpleNamespace(name=GU_S, rows_full=K16, cols_full=2 * I, itemsize=1,
                              tag="weights_0", component_rows=(I, I))
    by = {p.param_name: p for p in xm.pieces_from_inventory([w, s])}
    assert by[GU_W].component_rows == (2 * I, 2 * I)
    assert by[GU_S].component_rows == (I, I)


@pytest.mark.parametrize("name,pack,item", [(GU_W, 2, 4), (GU_S, 1, 1)])
def test_gate_up_weight_and_scales_are_a_mixed_fused_column_cut(name, pack, item):
    whole = _piece(name, K16, 2 * I * pack, (I * pack, I * pack), item)
    cut = [_piece(name, K16, 2 * I_R[r] * pack, (I_R[r] * pack, I_R[r] * pack), item) for r in range(3)]
    axis, rows_full, cols_full, widths, pad = xm._axis_of(name, whole, cut)
    assert axis == wx.MIXED_FUSED_COLS and pad == 0 and cols_full == 2 * I * pack


def test_down_proj_is_a_row_cut_and_the_global_scale_a_replica():
    name = "model.layers.0.mlp.down_proj.weight"
    whole = _piece(name, I // 16, 5120 * 2, (), 4)
    cut = [_piece(name, I_R[r] // 16, 5120 * 2, (), 4) for r in range(3)]
    assert xm._axis_of(name, whole, cut)[0] == wx.ROWS
    g = "model.layers.0.mlp.down_proj.weight_global_scale"
    assert xm._axis_of(g, _piece(g, 1, 1, (), 2), [_piece(g, 1, 1, (), 2)] * 3)[0] == wx.REPLICATED


V, V_R = 248320, 82816          # vocab; per-rank padded (3 x 82816 = V + 128)


def test_the_marlin_packed_lm_head_is_a_padded_column_cut_in_vocab_units():
    name = "lm_head.weight"
    whole = _piece(name, K16, 2 * V, (), 4)
    cut = [_piece(name, K16, 2 * V_R, (), 4)] * 3
    axis, rows_full, cols_full, widths, pad = xm._axis_of(name, whole, cut)
    assert axis == wx.COLS and pad == 2 * (3 * V_R - V) == 256 and cols_full == 6 * V_R
    s = "lm_head.weight_scale"
    axis, _, _, _, pad = xm._axis_of(s, _piece(s, K16, V, (), 1), [_piece(s, K16, V_R, (), 1)] * 3)
    assert axis == wx.COLS and pad == 128


def test_the_pack_reading_is_bounded_to_an_int32_container():
    name = "lm_head.weight"
    with pytest.raises(wx.Weg2XchgPlanDisagree):
        xm._axis_of(name, _piece(name, K16, 2 * V, (), 2), [_piece(name, K16, 2 * V_R, (), 2)] * 3)
    with pytest.raises(wx.Weg2XchgPlanDisagree):  # a width no pack explains
        xm._axis_of(name, _piece(name, K16, 2 * V, (), 4), [_piece(name, K16, 2 * V_R + 64, (), 4)] * 3)
