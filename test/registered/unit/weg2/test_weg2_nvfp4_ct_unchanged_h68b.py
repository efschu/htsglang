"""H68b GUARD for the NF production path: the compressed-tensors WNA16 door
(Qwen3.8-Flash-Next-INT4-Mixed-AutoRound-Minachist) does not change.

The NVFP4 half touched four shared places: ``EXPERT_TENSOR_ATTRS`` (+ the two
global scales), the installer's refusal table, the FusedMoE stream presplit
(a ``device_ctx`` key) and the pool copy kernel's row view (2-byte rows). Each
is pinned here against the CT door built from the Minachist quantization
config (its INT4 g128 routed-expert group, verbatim):

* the CT scheme's parameters share no name with the new expert tensors, and
  the presplit stages exactly the CT set it staged before;
* the CT stream state carries no ``device_ctx`` key and still runs its repack
  inside ``device_loading_context``; a ``device_ctx=False`` state does not;
* CT rows keep int32 words in the pool copy; the CT wrapper/scheme pass the
  installer.

The byte-level A/B (old modules at c701dd4d54 against these, same CT layer,
same store) is in the H68b commit message: identical buffers, spills, store
files and published attributes.
"""

import contextlib
import types

import pytest

try:
    import torch

    from sglang.srt.layers.moe import expert_offload as eo
    from sglang.srt.layers.moe import expert_pool_device as ep
    from sglang.test.ci.ci_register import register_cpu_ci
except RuntimeError as _import_err:  # pragma: no cover - leak-dependent
    pytest.skip(f"#249 import chain: {_import_err}", allow_module_level=True)

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

#: Minachist config.json, quantization_config: the routed-expert group (group_1)
#: and the fields the scheme reads.
MINACHIST_QC = {
    "quant_method": "compressed-tensors",
    "format": "pack-quantized",
    "quantization_status": "compressed",
    "config_groups": {
        "group_1": {
            "format": "pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["re:(model|language_model)\\..*\\.mlp\\.experts\\..*"],
            "weights": {
                "actorder": None, "block_structure": None, "dynamic": False,
                "group_size": 128, "num_bits": 4, "observer": "memoryless_minmax",
                "observer_kwargs": {}, "scale_dtype": None, "strategy": "group",
                "symmetric": True, "type": "int", "zp_dtype": None,
            },
        }
    },
    "ignore": [],
}

NEW_ATTRS = {"w13_weight_scale_2", "w2_weight_scale_2"}
E = 8


def _ct_layer():
    try:
        from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
            CompressedTensorsConfig,
        )
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16_moe import (
            CompressedTensorsWNA16MoE,
        )
    except Exception as ex:  # pragma: no cover - import-chain dependent
        pytest.skip(f"compressed-tensors not importable here: {ex}")
    cfg = CompressedTensorsConfig.from_config(MINACHIST_QC)
    wq = next(g["weights"] for g in cfg.target_scheme_map.values() if g.get("weights"))
    scheme = CompressedTensorsWNA16MoE(cfg, wq)
    layer = torch.nn.Module()
    layer.moe_tp_size = 1
    layer.num_local_experts = E
    layer.num_experts = E
    layer.layer_id = 3
    layer.moe_tp_rank = 0
    layer._expert_offload_fraction = 0.5
    scheme.create_weights(layer, num_experts=E, hidden_size=256, intermediate_size_per_partition=128,
                          params_dtype=torch.bfloat16, weight_loader=lambda *a, **k: None)
    g = torch.Generator().manual_seed(3)
    for _n, p in layer.named_parameters():
        if p.dtype == torch.int32:
            p.data.copy_(torch.randint(0, 2**31 - 1, p.shape, generator=g, dtype=torch.int32))
        else:
            p.data.copy_(torch.rand(p.shape, generator=g).to(p.dtype))
    return layer


def test_ct_parameters_share_no_name_with_the_new_expert_tensors():
    layer = _ct_layer()
    names = {n for n, _ in layer.named_parameters()}
    assert names.isdisjoint(NEW_ATTRS)
    assert NEW_ATTRS <= set(eo.MoEExpertOffloadCache.EXPERT_TENSOR_ATTRS)


def test_ct_presplit_stages_the_same_set(tmp_path, monkeypatch):
    from sglang.srt.layers.moe import expert_store

    monkeypatch.setenv(expert_store.STORE_DIR_ENV, str(tmp_path / "store"))
    layer = _ct_layer()
    eo.presplit_expert_offload_after_repack(layer)
    assert set(layer._moe_offload_presplit) == {
        "w13_weight_packed", "w2_weight_packed", "w13_weight_scale", "w2_weight_scale"}


def test_ct_rows_keep_int32_words():
    layer = _ct_layer()
    for n in ("w13_weight_packed", "w2_weight_packed", "w13_weight_scale", "w2_weight_scale"):
        t = getattr(layer, n).data
        old = t.view(torch.uint8).reshape(t.shape[0], -1).view(torch.int32)  # pre-H68b
        new = ep._word_rows(t)
        assert new.dtype == torch.int32 and new.shape == old.shape
        assert torch.equal(new, old)


def test_ct_wrapper_and_scheme_pass_the_installer():
    wrapper = type("CompressedTensorsFusedMoEMethod", (), {})()
    scheme = type("CompressedTensorsWNA16MoE", (), {})()
    eo.assert_expert_offload_quant_supported(wrapper, 3, scheme=scheme, layer=types.SimpleNamespace())


@pytest.mark.parametrize("state_extra,want_ctx", [({}, True), ({"device_ctx": False}, False)])
def test_stream_presplit_device_context_choice(monkeypatch, state_extra, want_ctx):
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.model_loader import loader

    entered = []

    @contextlib.contextmanager
    def fake_ctx(module, device):
        entered.append(device)
        yield module

    monkeypatch.setattr(loader, "device_loading_context", fake_ctx)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    ran = []
    fake_self = types.SimpleNamespace(
        layer_id=None,
        quant_method=types.SimpleNamespace(process_weights_after_loading=lambda s: ran.append(s)),
    )
    state = dict({"device": torch.device("cpu")}, **state_extra)
    FusedMoE._ct_stream_presplit_now(fake_self, state)
    assert ran == [fake_self]
    assert bool(entered) is want_ctx
