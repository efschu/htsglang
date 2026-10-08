"""#68a: EIN Praedikat fuer "wird dieser Tensor transponiert".

fnFL2v87 starb an der Asymmetrie: der VERBRAUCHER entschied an der Methode
des LAYERS (drei Klassennamen), der WORKER an der quant_config des MODELLS
("CompressedTensors" in ...) -- und das trifft mehr. Der Worker transponierte
Tensoren, die der Verbraucher nie anfasste:

    RuntimeError: The size of tensor a (2560) must match the size of
    tensor b (80) at non-singleton dimension 1

Der Test, der gefehlt hat, stellt BEIDE Seiten gegen dieselbe Namensliste.
"""
import pathlib
import types

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2] / "python" / "flliper"


def _strip_comments(text):
    return "\n".join(z for z in text.split("\n") if not z.lstrip().startswith("#"))


_LAYER = _strip_comments((_ROOT / "srt/layers/moe/fused_moe_triton/layer.py").read_text())
_QWEN = _strip_comments((_ROOT / "srt/models/qwen4_exp.py").read_text())


def test_there_is_only_one_list():
    # Die Klassennamen duerfen an genau EINER Stelle stehen. Zwei Listen sind
    # zwei Wahrheiten, und die zweite war breiter.
    assert _LAYER.count('"CompressedTensorsWNA16MarlinMoE"') == 1
    assert 'CompressedTensorsWNA16MarlinMoE' not in _QWEN


def test_consumer_calls_the_shared_predicate():
    assert "ct_method_transposes(method)" in _LAYER


def test_worker_calls_the_same_predicate():
    assert "ct_method_transposes(" in _QWEN
    # und NICHT mehr den Modelltyp
    assert '"CompressedTensors" in method' not in _QWEN


def test_predicate_decides_on_the_method():
    import sys, importlib
    m = importlib.import_module("flliper.srt.layers.moe.fused_moe_triton.layer")

    class CompressedTensorsWNA16MoE: ...
    class CompressedTensorsW8A8Int8: ...      # compressed-tensors, aber NICHT in der Liste

    assert m.ct_method_transposes(CompressedTensorsWNA16MoE()) is True
    assert m.ct_method_transposes(CompressedTensorsW8A8Int8()) is False, (
        "genau dieser Fall war v87: ein compressed-tensors-Schema, das der "
        "Verbraucher NICHT transponiert"
    )
    assert m.ct_method_transposes(None) is False


def test_without_layer_answer_is_false():
    # Ein Fehltreffer darf Geschwindigkeit kosten, nie Korrektheit: findet der
    # Worker den Layer nicht, transponiert der Verbraucher selbst.
    import importlib
    q = importlib.import_module("flliper.srt.models.qwen4_exp")
    model = types.SimpleNamespace(get_submodule=lambda n: (_ for _ in ()).throw(AttributeError()))
    assert q._expert_layer_for_name("model.layers.0.mlp.experts.1.w.weight_packed", model) is None
    assert q._is_ct_wna16_expert_shard("model.layers.0.mlp.experts.1.w.weight_packed", model) is False


def test_lock_is_gone_and_reason_named():
    src = (_ROOT / "srt/models/qwen4_exp.py").read_text()
    i = src.index("def weight_post_load")
    block = src[i : src.index("def weight_name_needed", i)]
    assert "DEFECT and locked" not in block, "die alte Verriegelung steht noch"
    assert "_is_ct_wna16_expert_shard(name, self)" in block, (
        "der Worker transponiert ohne das gemeinsame Praedikat zu fragen"
    )
    assert "tensor.t()" in block
    # Der ZWEITE Grund (post_load seriell je Datei) muss benannt bleiben --
    # er ist NICHT geloest, und wer den Schalter anwirft, muss das wissen.
    assert "serially per file" in block
