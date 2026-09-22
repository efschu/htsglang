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

_ROOT = pathlib.Path(__file__).resolve().parents[2] / "python" / "sglang"


def _ohne_kommentare(text):
    return "\n".join(z for z in text.split("\n") if not z.lstrip().startswith("#"))


_LAYER = _ohne_kommentare((_ROOT / "srt/layers/moe/fused_moe_triton/layer.py").read_text())
_QWEN = _ohne_kommentare((_ROOT / "srt/models/qwen4_exp.py").read_text())


def test_es_gibt_nur_noch_EINE_liste():
    # Die Klassennamen duerfen an genau EINER Stelle stehen. Zwei Listen sind
    # zwei Wahrheiten, und die zweite war breiter.
    assert _LAYER.count('"CompressedTensorsWNA16MarlinMoE"') == 1
    assert 'CompressedTensorsWNA16MarlinMoE' not in _QWEN


def test_der_verbraucher_ruft_das_gemeinsame_praedikat():
    assert "ct_method_transposes(method)" in _LAYER


def test_der_worker_ruft_DASSELBE_praedikat():
    assert "ct_method_transposes(" in _QWEN
    # und NICHT mehr den Modelltyp
    assert '"CompressedTensors" in method' not in _QWEN


def test_das_praedikat_entscheidet_an_der_methode():
    import sys, importlib
    m = importlib.import_module("sglang.srt.layers.moe.fused_moe_triton.layer")

    class CompressedTensorsWNA16MoE: ...
    class CompressedTensorsW8A8Int8: ...      # compressed-tensors, aber NICHT in der Liste

    assert m.ct_method_transposes(CompressedTensorsWNA16MoE()) is True
    assert m.ct_method_transposes(CompressedTensorsW8A8Int8()) is False, (
        "genau dieser Fall war v87: ein compressed-tensors-Schema, das der "
        "Verbraucher NICHT transponiert"
    )
    assert m.ct_method_transposes(None) is False


def test_ohne_layer_ist_die_antwort_False():
    # Ein Fehltreffer darf Geschwindigkeit kosten, nie Korrektheit: findet der
    # Worker den Layer nicht, transponiert der Verbraucher selbst.
    import importlib
    q = importlib.import_module("sglang.srt.models.qwen4_exp")
    modell = types.SimpleNamespace(get_submodule=lambda n: (_ for _ in ()).throw(AttributeError()))
    assert q._expert_layer_for_name("model.layers.0.mlp.experts.1.w.weight_packed", modell) is None
    assert q._is_ct_wna16_expert_shard("model.layers.0.mlp.experts.1.w.weight_packed", modell) is False
