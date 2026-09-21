"""#66 (21.09.): die Transposition der Experten-Shards wandert aus dem
Hauptthread in die Datei-Worker.

GEMESSEN fnFL2v84/v85 (SGLANG_LOAD_PROFILE=1, drei Raenge einig): 43-50 % der
Ladezeit stehen in layer.py `loaded_weight.t().contiguous()`, seriell, waehrend
die acht Worker in threading.wait stehen und die NVMe bei Queue-Tiefe 0,64 nur
40-60 % Leselast meldet.

DIE EINE EIGENSCHAFT, die dieser Test sichert: die Transposition passiert
GENAU EINMAL. Zweimal laedt still falsch.
"""

import os
import types

import pytest
import torch

from sglang.srt.models import qwen4_exp as qx
from sglang.srt.layers.moe.fused_moe_triton import layer as fml


class _CT:
    """Eine compressed-tensors quant_config, wie das Modell sie traegt."""
    __class__ = type("CompressedTensorsConfig", (), {})


def _model(quant="CompressedTensorsConfig"):
    qc = type(quant, (), {})()
    return types.SimpleNamespace(quant_config=qc)


# -- der Schalter -----------------------------------------------------------

def test_off_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_LOAD_TRANSPOSE_IN_WORKER", raising=False)
    assert qx._transpose_in_worker() is False
    assert fml._transpose_done_in_worker() is False


def test_both_sides_read_the_same_switch(monkeypatch):
    for raw, want in (("1", True), ("0", False), ("", False), ("ja", False)):
        monkeypatch.setenv("SGLANG_LOAD_TRANSPOSE_IN_WORKER", raw)
        assert qx._transpose_in_worker() is want, raw
        assert fml._transpose_done_in_worker() is want, raw


# -- was der Worker anfasst und was nicht -----------------------------------

def test_only_compressed_tensors_expert_shards(monkeypatch):
    m = _model()
    yes = [
        "model.layers.0.mlp.experts.3.gate_proj.weight_packed",
        "model.layers.0.mlp.experts.3.down_proj.weight_scale",
        "model.layers.9.mlp.experts.511.up_proj.weight_zero_point",
    ]
    no = [
        "model.layers.0.self_attn.q_proj.weight",      # kein Expert
        "model.layers.0.mlp.experts.3.gate_proj.g_idx",  # falsches Suffix
        "model.embed_tokens.weight",
    ]
    for n in yes:
        assert qx._is_ct_wna16_expert_shard(n, m) is True, n
    for n in no:
        assert qx._is_ct_wna16_expert_shard(n, m) is False, n


def test_a_non_compressed_tensors_model_is_never_touched():
    m = _model(quant="AWQConfig")
    assert qx._is_ct_wna16_expert_shard(
        "model.layers.0.mlp.experts.3.gate_proj.weight_packed", m
    ) is False
    assert qx._is_ct_wna16_expert_shard("x", types.SimpleNamespace(quant_config=None)) is False


# -- die Eigenschaft: genau einmal -----------------------------------------

def test_the_switch_is_locked_because_it_loaded_wrong(monkeypatch):
    """fnFL2v87 starb eingeschaltet an

        RuntimeError: The size of tensor a (2560) must match the size of
        tensor b (80) at non-singleton dimension 1

    -- das Praedikat im Worker ist breiter als das des Verbrauchers, der je
    LAYER-METHODE entscheidet und Zweige hat, die vor der Transposition
    abbiegen. Und schneller war es auch nicht (PP2 36,7 -> 47,5 s). Also
    WIRFT der Pfad, statt still falsch zu laden."""
    src = torch.arange(2560 * 80, dtype=torch.int32).reshape(2560, 80)

    monkeypatch.setenv("SGLANG_LOAD_TRANSPOSE_IN_WORKER", "1")
    with pytest.raises(RuntimeError, match="DEFECT and locked"):
        qx.Qwen4ExpForConditionalGeneration.weight_post_load(
            _model(), "model.layers.0.mlp.experts.3.gate_proj.weight_packed", src
        )

    monkeypatch.setenv("SGLANG_LOAD_TRANSPOSE_IN_WORKER", "0")
    out = qx.Qwen4ExpForConditionalGeneration.weight_post_load(
        _model(), "model.layers.0.mlp.experts.3.gate_proj.weight_packed", src
    )
    assert out is src                           # aus: unveraendert durch
    assert not fml._transpose_done_in_worker()  # und der Verbraucher tut es


def test_the_consumer_reads_the_switch_at_the_right_place():
    import inspect

    src = inspect.getsource(fml.FusedMoE._weight_loader_impl)
    assert "_needs_ct_transpose and _transpose_done_in_worker()" in src
    assert "loaded_weight.t().contiguous() if _needs_ct_transpose else loaded_weight" in src


def test_the_lock_comes_before_any_shape_logic(monkeypatch):
    """Die Verriegelung greift VOR jeder Namens- oder Formpruefung -- sonst
    haengt die Sicherheit an demselben Praedikat, das den Defekt hatte."""
    monkeypatch.setenv("SGLANG_LOAD_TRANSPOSE_IN_WORKER", "1")
    t = torch.arange(8, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="DEFECT and locked"):
        qx.Qwen4ExpForConditionalGeneration.weight_post_load(
            _model(), "irgendein.name.ohne.experts", t
        )


# -- die Naht im Loader -----------------------------------------------------

def test_the_loader_applies_it_in_the_worker():
    import inspect

    from sglang.srt.model_loader import weight_utils as wu

    src = inspect.getsource(wu.buffered_multi_thread_safetensors_weights_iterator)
    assert "post_load" in src
    assert "result = {k: post_load(k, v) for k, v in result.items()}" in src
    # und NICHT gekapselt -- ein halb transformierter Satz laedt still falsch
    i = src.index("result = {k: post_load(k, v)")
    assert "try:" not in src[max(0, i - 200):i]


def test_the_loader_hands_the_models_hook_over():
    import inspect

    from sglang.srt.model_loader import loader as ld

    src = inspect.getsource(ld.DefaultModelLoader._get_all_weights)
    assert 'self._weight_post_load = getattr(model, "weight_post_load", None)' in src
    assert "self._weight_post_load = None" in src
    it = inspect.getsource(ld.DefaultModelLoader._get_weights_iterator)
    assert 'post_load=getattr(self, "_weight_post_load", None)' in it
