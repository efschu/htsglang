"""H68b: the NVFP4 half of the MoE expert offload (ModelOpt NVFP4 on Marlin).

Before this, ``assert_expert_offload_quant_supported`` refused every NVFP4 MoE
method (#323b) -- on this rig that is a refusal of the checkpoint, because
nvidia/Qwen3.8-Flash-Next-NVFP4's 63 GiB of routed experts fit on no card set
without the offload. Pinned here, hermetically (CPU, no CUDA):

(a) the installer admits ``ModelOptNvFp4FusedMoEMethod`` exactly on a layer
    the half staged (marker), and still refuses it -- with a reason naming the
    Marlin requirement -- when it did not; the online / compressed-tensors
    NVFP4 methods stay refused; the per-expert global scales are expert tensors.
(b) create_weights stages the experts on the HOST only with the offload on,
    the Marlin path, a serialized checkpoint and a non-draft layer, and arms the
    per-layer stream presplit with the method's own shard counts.
(c)(d) the Marlin branch zeroes the generic-shard pad expert BEFORE the
    in-place repack, runs the presplit, marks only host-staged layers, and a
    second post-load pass is a no-op.
(e) the presplit stages the ``[E, 1]`` global scales like every other expert
    tensor (buffer, host store rows, Platztausch buffer); the cache install
    binds them; a fetch moves the right expert's scale into its scratch slot;
    the pool copy kernel's row view accepts 2-byte rows (int16 words) and keeps
    int32 words for every row size it saw before.
(f) the manifest reads an ``[n, 1]`` expert buffer as n ROWS (the expert row
    cut P vs D classifies as ROWS); a 1-D ``[n]`` one would be one row of n
    columns, which is why the scales are kept 2-D.
"""

import os
import types

import pytest

try:
    import torch

    from sglang.srt.layers.moe import expert_offload as eo
    from sglang.srt.layers.moe import expert_pool_device as ep
    from sglang.srt.layers.quantization import modelopt_quant as M
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

NVFP4 = "ModelOptNvFp4FusedMoEMethod"


def _named(cls_name):
    return type(cls_name, (), {})()


# --- (a) installer ----------------------------------------------------------


def test_nvfp4_is_refused_without_the_marker():
    layer = types.SimpleNamespace()
    with pytest.raises(RuntimeError) as ctx:
        eo.assert_expert_offload_quant_supported(_named(NVFP4), 7, layer=layer)
    msg = str(ctx.value)
    assert "_moe_offload_nvfp4_marlin_staged" in msg
    assert "--moe-runner-backend marlin" in msg
    assert "(layer_id=7)" in msg


def test_nvfp4_is_admitted_on_a_staged_layer():
    layer = types.SimpleNamespace(_moe_offload_nvfp4_marlin_staged=True)
    eo.assert_expert_offload_quant_supported(_named(NVFP4), 7, layer=layer)


@pytest.mark.parametrize("name", ["ModelOptNvFp4OnlineFusedMoEMethod", "CompressedTensorsW4A4Nvfp4MoE"])
def test_other_nvfp4_methods_stay_refused(name):
    layer = types.SimpleNamespace(_moe_offload_nvfp4_marlin_staged=True)
    with pytest.raises(RuntimeError, match="no load-time"):
        eo.assert_expert_offload_quant_supported(_named(name), 1, layer=layer)


def test_gguf_refusal_text_is_unchanged():
    with pytest.raises(RuntimeError, match=r"has a load-time offload half \(#123-GGUF\)"):
        eo.assert_expert_offload_quant_supported(
            _named("GGUFMoEMethod"), 1, layer=types.SimpleNamespace())


def test_global_scales_are_expert_tensors():
    attrs = eo.MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS
    assert "w13_weight_scale_2" in attrs and "w2_weight_scale_2" in attrs
    assert "w13_input_scale" not in attrs  # global, unread by Marlin


# --- (b) create_weights ------------------------------------------------------


def _method(marlin=True, serialized=True, online=False):
    meth = object.__new__(M.ModelOptNvFp4FusedMoEMethod)
    meth.quant_config = types.SimpleNamespace(
        is_checkpoint_nvfp4_serialized=serialized, group_size=16, is_nvfp4_online=online)
    meth.use_marlin_fallback = marlin
    meth.enable_flashinfer_trtllm_moe = False
    return meth


def _moe_layer(n_local=4, excluded=False):
    layer = torch.nn.Module()
    layer.num_local_experts = n_local
    layer.num_experts = n_local
    layer.moe_runner_config = types.SimpleNamespace(is_gated=True)
    layer._moe_offload_excluded = excluded
    return layer


@pytest.fixture
def offload_on(monkeypatch):
    from sglang.srt.layers.moe import resident_fraction

    monkeypatch.setattr(resident_fraction, "offload_active", lambda: True)


@pytest.mark.parametrize(
    "kw,excluded,want",
    [
        ({}, False, "cpu"),
        ({"marlin": False}, False, None),
        ({"serialized": False}, False, None),
        ({"online": True}, False, None),
        ({}, True, None),
    ],
)
def test_host_staging_decision(offload_on, kw, excluded, want):
    assert _method(**kw)._nvfp4_host_staging_device(_moe_layer(excluded=excluded)) == want


def test_no_staging_without_offload(monkeypatch):
    from sglang.srt.layers.moe import resident_fraction

    monkeypatch.setattr(resident_fraction, "offload_active", lambda: False)
    assert _method()._nvfp4_host_staging_device(_moe_layer()) is None


def test_create_weights_stages_on_host_and_arms_the_stream(offload_on):
    meth, layer = _method(), _moe_layer(n_local=5)
    layer._expert_shard_owned = 4  # generic shard: the pad never arrives
    meth.create_weights(layer, num_experts=5, hidden_size=64, intermediate_size_per_partition=32,
                        params_dtype=torch.bfloat16, weight_loader=lambda *a, **k: None)
    assert layer._moe_nvfp4_host_staged is True
    for n in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
              "w13_weight_scale_2", "w2_weight_scale_2"):
        assert getattr(layer, n).device.type == "cpu", n
    # CPU ambient in this test: not armed by create_weights itself ...
    assert getattr(layer, "_ct_stream_presplit", None) is None
    # ... armed with a CUDA ambient device, with the method's own counts
    meth._arm_stream_presplit(layer, 2, ambient=torch.device("cuda", 0))
    st = layer._ct_stream_presplit
    assert st["expected"] == {
        "w13_weight": 8, "w2_weight": 4, "w13_weight_scale": 8, "w2_weight_scale": 4,
        "w13_weight_scale_2": 8, "w2_weight_scale_2": 4,
    }
    assert st["device_ctx"] is False and st["done"] is False
    assert st["names"][id(layer.w13_weight_scale_2)] == "w13_weight_scale_2"
    assert id(layer.w13_input_scale) not in st["names"]  # global, not counted


# --- (c)(d) the Marlin branch --------------------------------------------------


def _marlin_ready_layer(staged):
    layer = _moe_layer(n_local=3)
    layer.w13_weight_scale_2 = torch.nn.Parameter(torch.ones(3, 2), requires_grad=False)
    layer._moe_nvfp4_host_staged = staged
    return layer


@pytest.mark.parametrize("staged", [True, False])
def test_marlin_branch_order_marker_and_idempotence(monkeypatch, staged):
    from sglang.srt.layers.moe import MoeRunnerBackend

    calls = []
    layer = _marlin_ready_layer(staged)
    layer.zero_expert_shard_pad = lambda: calls.append("pad")
    monkeypatch.setattr(M, "prepare_moe_nvfp4_layer_for_marlin_inplace",
                        lambda lyr, outputs_survive: calls.append(("repack", outputs_survive)))
    monkeypatch.setattr(eo, "presplit_expert_offload_after_repack",
                        lambda lyr: calls.append("presplit"))
    meth = _method()
    meth._moe_runner_backend = MoeRunnerBackend.MARLIN
    meth.process_weights_after_loading(layer)
    assert calls == ["pad", ("repack", not staged), "presplit"]
    assert tuple(layer.w13_weight_scale_2.shape) == (3,)  # gate column
    assert layer.is_marlin_converted is True
    assert getattr(layer, "_moe_offload_nvfp4_marlin_staged", False) is staged
    meth.process_weights_after_loading(layer)  # the loader's later pass
    assert len(calls) == 3


# --- (e) presplit, store, cache, pool -----------------------------------------

EX, K, N = 8, 64, 32


def _repacked_nvfp4_layer():
    """An NVFP4 MoE layer in its post-repack (Marlin) shapes, CPU tensors."""
    g = torch.Generator().manual_seed(1)
    layer = torch.nn.Module()
    layer.layer_id = 9
    layer.num_experts = EX
    layer.num_local_experts = EX  # a PP stage: local id == global id
    layer.moe_tp_rank = 0
    layer._expert_offload_fraction = 0.5

    def P(t):
        return torch.nn.Parameter(t, requires_grad=False)

    layer.w13_weight = P(torch.randint(0, 2**31 - 1, (EX, K // 16, 4 * N), generator=g, dtype=torch.int32))
    layer.w2_weight = P(torch.randint(0, 2**31 - 1, (EX, N // 16, 2 * K), generator=g, dtype=torch.int32))
    layer.w13_weight_scale = P(torch.rand(EX, K // 16, 2 * N, generator=g).to(torch.float8_e4m3fn))
    layer.w2_weight_scale = P(torch.rand(EX, N // 16, K, generator=g).to(torch.float8_e4m3fn))
    # one distinct value per expert: a mixed-up row shows
    layer.w13_weight_scale_2 = P((torch.arange(EX, dtype=torch.bfloat16) + 1).reshape(EX, 1))
    layer.w2_weight_scale_2 = P((torch.arange(EX, dtype=torch.bfloat16) + 101).reshape(EX, 1))
    layer.w13_input_scale = P(torch.ones(EX, 2))
    return layer


@pytest.fixture
def store_dir(tmp_path, monkeypatch):
    from sglang.srt.layers.moe import expert_store

    d = tmp_path / "store"
    monkeypatch.setenv(expert_store.STORE_DIR_ENV, str(d))
    return d


def test_presplit_stages_the_global_scales(store_dir):
    from sglang.srt.managers.weg2_memory_saver import expert_buffer_attr_name

    layer = _repacked_nvfp4_layer()
    ref = {a: getattr(layer, a).data.clone() for a in ("w13_weight_scale_2", "w2_weight_scale_2")}
    eo.presplit_expert_offload_after_repack(layer)
    pre = layer._moe_offload_presplit
    assert set(pre) == {"w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale",
                        "w13_weight_scale_2", "w2_weight_scale_2"}
    R = eo.resident_slot_count(EX, 0.5)
    store_index = layer._moe_offload_store_index
    for attr in ("w13_weight_scale_2", "w2_weight_scale_2"):
        buf, spill = pre[attr]
        assert buf.dim() == 2 and buf.shape[1] == 1  # [R+C, 1]
        assert torch.equal(buf[:R], ref[attr][:R])  # static plan: residents first
        for local, row in store_index.items():  # cold rows in the host store
            assert torch.equal(spill[row], ref[attr][local]), (attr, local)
        assert tuple(getattr(layer, attr).shape) == (0, 1)  # placeholder
        published = getattr(layer, expert_buffer_attr_name(attr))
        assert published.data_ptr() == buf.data_ptr()
        assert os.path.exists(store_dir / f"L9-{attr}.bin")
    assert tuple(layer.w13_input_scale.shape) == (EX, 2)  # untouched


def test_cache_install_and_fetch_move_the_right_scale(store_dir):
    layer = _repacked_nvfp4_layer()
    ref = layer.w13_weight_scale_2.data.clone()
    eo.presplit_expert_offload_after_repack(layer)
    cache = eo.MoEExpertOffloadCache(layer, 0.5)
    cache.install()
    bank = layer.w13_weight_scale_2
    assert isinstance(bank, torch.nn.Parameter) and bank.shape[1] == 1
    assert bank.shape[0] == cache.planner.buffer_size
    R = cache.resident_count
    cold = sorted(layer._moe_offload_store_index)[0] if getattr(layer, "_moe_offload_store_index", None) else R
    slot = R  # first scratch slot
    cache._fetch([(cold, slot)])
    assert torch.equal(bank.data[slot], ref[cold])
    assert torch.equal(layer.w2_weight_scale_2.data[slot].float(), torch.tensor([101.0 + cold]))


def test_pool_row_view_widths():
    two_byte = torch.zeros(6, 1, dtype=torch.bfloat16)
    assert ep._word_rows(two_byte).dtype == torch.int16
    four_byte = torch.zeros(6, 1, dtype=torch.float32)
    assert ep._word_rows(four_byte).dtype == torch.int32
    ct_like = torch.zeros(6, 160, 2560, dtype=torch.int32)  # every pre-H68b row
    w = ep._word_rows(ct_like)
    assert w.dtype == torch.int32 and tuple(w.shape) == (6, 160 * 2560)
    odd = torch.zeros(6, 3, dtype=torch.uint8)
    assert ep._word_rows(odd).dtype == torch.uint8


def test_pool_copy_moves_two_byte_rows():
    src = (torch.arange(10, dtype=torch.bfloat16) + 1).reshape(10, 1)
    dst = torch.zeros(4, 1, dtype=torch.bfloat16)
    ep.copy_rows([src], [dst], torch.tensor([7, 2]), torch.tensor([1, 3]), torch.tensor([2]))
    assert dst.view(-1).tolist() == [0.0, 8.0, 0.0, 3.0]


# --- (f) manifest geometry ----------------------------------------------------


def test_manifest_reads_expert_scale_rows():
    from sglang.srt.weg2 import weight_exchange as wx
    from sglang.srt.weg2 import xchg_manifest as xm

    g2 = wx.StorageGeom.of(torch.zeros(61, 1, dtype=torch.bfloat16))
    assert (g2.rows, g2.cols) == (61, 1)
    g1 = wx.StorageGeom.of(torch.zeros(61, dtype=torch.bfloat16))
    assert (g1.rows, g1.cols) == (1, 61)  # why the scales are kept 2-D

    def piece(rows, cols):
        return xm.ManifestPiece(
            param_name="model.layers.0.mlp.experts.weg2_experts_w13_weight_scale_2",
            tensor_class="weg2_experts_w13_weight_scale_2", rows_full=rows,
            cols_full=cols, itemsize=2, tag="weights_0", nbytes=rows * cols * 2)

    axis = xm._axis_of("model.layers.0.mlp.experts.weg2_experts_w13_weight_scale_2",
                       piece(515, 1), [piece(61, 1), piece(227, 1), piece(227, 1)])
    assert axis[0] == wx.ROWS
    one_d = xm._axis_of("model.layers.0.mlp.experts.weg2_experts_w13_weight_scale_2",
                        piece(1, 515), [piece(1, 61), piece(1, 227), piece(1, 227)])
    assert one_d[0] != wx.ROWS
