"""#68d: Worker-Transpose und Verbraucher muessen DIESELBE Methode fragen.

fnFL2w59 starb nach 7,3 s an "The size of tensor a (2560) must match the
size of tensor b (80)". Grund: `_is_ct_wna16_expert_shard` fragte
`layer.quant_method`, der Verbraucher in `_weight_loader_impl` fragte
`self.scheme`. Die Namen in `_CT_TRANSPOSING_METHODS` sind SCHEMA-Namen --
`quant_method` traf nie einen, der Worker transponierte also nie, und der
Verbraucher sprang wegen `SGLANG_LOAD_TRANSPOSE_IN_WORKER=1` trotzdem ueber
seine eigene Transposition. Niemand transponierte.
"""
import inspect
import types

from sglang.srt.layers.moe.fused_moe_triton import layer as fml


class CompressedTensorsWNA16MarlinMoE:
    """Traegt nur den NAMEN -- genau den, auf den der Code prueft."""


class CompressedTensorsMoEMethod:
    """Die `quant_method`, unter der das Schema haengt. Transponiert NICHT."""


def test_schema_gewinnt_gegen_quant_method():
    layer = types.SimpleNamespace(
        quant_method=CompressedTensorsMoEMethod(),
        scheme=CompressedTensorsWNA16MarlinMoE(),
    )
    assert not fml.ct_method_transposes(layer.quant_method), (
        "die quant_method steht nicht in _CT_TRANSPOSING_METHODS -- genau "
        "deshalb war `layer.quant_method` die falsche Frage"
    )
    assert fml.ct_method_transposes(fml.ct_effective_method(layer))


def test_ohne_schema_zaehlt_die_quant_method():
    layer = types.SimpleNamespace(quant_method=CompressedTensorsWNA16MarlinMoE())
    assert fml.ct_method_transposes(fml.ct_effective_method(layer))


def test_ktep_wrapper_wird_ausgepackt():
    class KTEPWrapperMethod:
        gpu_method = CompressedTensorsWNA16MarlinMoE()

    layer = types.SimpleNamespace(quant_method=KTEPWrapperMethod())
    assert fml.ct_method_transposes(fml.ct_effective_method(layer))


def test_verbraucher_fragt_dieselbe_funktion():
    """Der Verbraucher darf die Kette nicht ein zweites Mal ausschreiben --
    sonst laufen die beiden Seiten wieder auseinander."""
    src = inspect.getsource(fml.FusedMoE._weight_loader_impl)
    code = "\n".join(z for z in src.split("\n") if not z.lstrip().startswith("#"))
    assert "ct_effective_method(self)" in code
    assert "self.scheme" not in code, (
        "der Verbraucher loest die Methode wieder selbst auf"
    )


def test_worker_fragt_dieselbe_funktion():
    from sglang.srt.models import qwen4_exp

    src = inspect.getsource(qwen4_exp._is_ct_wna16_expert_shard)
    code = "\n".join(z for z in src.split("\n") if not z.lstrip().startswith("#"))
    assert "ct_effective_method(layer)" in code
    assert 'getattr(layer, "quant_method"' not in code, (
        "der Worker fragt wieder nur die quant_method -- das war die w59-Wurzel"
    )


def test_worker_praedikat_am_echten_namen(monkeypatch):
    """Die Gegenprobe am Prädikat selbst, nicht am Quelltext: ein Layer,
    dessen SCHEMA transponiert und dessen quant_method nicht -- genau die
    Form des Checkpoints, an dem w59 starb."""
    from sglang.srt.models import qwen4_exp

    experts = types.SimpleNamespace(
        quant_method=CompressedTensorsMoEMethod(),
        scheme=CompressedTensorsWNA16MarlinMoE(),
    )

    class _Modell:
        def get_submodule(self, pfad):
            if pfad.endswith("layers.7.mlp.experts"):
                return experts
            raise AttributeError(pfad)

    name = "model.language_model.layers.7.mlp.experts.3.down_proj.weight_packed"
    assert qwen4_exp._is_ct_wna16_expert_shard(name, _Modell()) is True
    # kein Experten-Tensor -> nie
    assert not qwen4_exp._is_ct_wna16_expert_shard(
        "model.layers.7.mlp.gate.weight", _Modell()
    )


def test_auch_der_fused_einstieg_kennt_die_quittung():
    """devindex `where scheme kinds=read path~layers/moe` fand einen ZWEITEN
    Leser der Methodenkette: `weight_loader_fused`. Er transponiert mit
    einer eigenen, engeren Namensliste -- ohne die Quittung waere das eine
    zweite Transposition auf einem Tensor, den der Worker schon gedreht
    hat."""
    src = inspect.getsource(fml.FusedMoE.weight_loader_fused)
    code = "\n".join(z for z in src.split("\n") if not z.lstrip().startswith("#"))
    assert "ct_effective_method(self)" in code
    assert "not transpose_done_in_worker(self)" in code
    assert "self.scheme" not in code


def test_68f_zaehler_zaehlt_gedreht_und_angeboten(monkeypatch):
    """Ohne diese Zahl sind am Ende eines Boots 'der Schalter brachte
    nichts' und 'der Schalter griff nie' nicht unterscheidbar -- w58 (Env
    kam nicht an) und w59 (Praedikat sagte immer nein) haben je einen Boot
    gekostet."""
    import torch
    from sglang.srt.models import qwen4_exp as qx

    monkeypatch.setenv("SGLANG_LOAD_TRANSPOSE_IN_WORKER", "1")
    experts = types.SimpleNamespace(
        quant_method=CompressedTensorsMoEMethod(),
        scheme=CompressedTensorsWNA16MarlinMoE(),
    )

    class _M:
        def get_submodule(self, pfad):
            if pfad.endswith(".experts"):
                return experts
            raise AttributeError(pfad)

    vor = qx.worker_transpose_counts()
    t = torch.arange(6, dtype=torch.int32).reshape(2, 3)
    qx.Qwen4ExpForConditionalGeneration.weight_post_load(
        _M(), "model.layers.0.mlp.experts.3.gate_proj.weight_packed", t
    )
    qx.Qwen4ExpForConditionalGeneration.weight_post_load(
        _M(), "model.layers.0.self_attn.q_proj.weight", t
    )
    nach = qx.worker_transpose_counts()
    assert nach[0] - vor[0] == 1, "gedreht falsch gezaehlt"
    assert nach[1] - vor[1] == 2, "angeboten falsch gezaehlt"


def test_68f_der_zaehler_haelt_nichts_fest():
    """Ein Instrument darf das Gemessene nicht festhalten (21.09.: der
    Sampler hielt Frames ueber sein wait = +450 MiB reserved, zwei Boots
    tot). Hier stehen zwei ints, keine Tensoren."""
    from sglang.srt.models import qwen4_exp as qx

    assert all(isinstance(x, int) for x in qx._WORKER_TRANSPOSED)
    assert len(qx._WORKER_TRANSPOSED) == 2
