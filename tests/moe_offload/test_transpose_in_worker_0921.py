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


def _experts_modul(transponierend=True):
    """Ein FusedMoE-artiges Modul: die transponierende Methode haengt am
    SCHEMA, nicht an `quant_method` -- genau die Form des Checkpoints, an
    dem w59 starb."""
    m = types.SimpleNamespace(quant_method=object())
    if transponierend:
        m.scheme = type("CompressedTensorsWNA16MarlinMoE", (), {})()
    return m


def _model(quant="CompressedTensorsConfig", transponierend=True):
    """Ein Modell, das `get_submodule` beantwortet -- ohne das findet der
    Worker den Layer nicht und antwortet False, und der Test prueft dann
    nur noch seine eigene Attrappe (die Luecke, durch die w59 fiel)."""
    qc = type(quant, (), {})()
    experts = _experts_modul(transponierend and quant == "CompressedTensorsConfig")

    class _M(types.SimpleNamespace):
        def get_submodule(self, pfad):
            if pfad.endswith(".experts"):
                return experts
            raise AttributeError(pfad)

    return _M(quant_config=qc)


# -- der Schalter -----------------------------------------------------------

def test_off_by_default(monkeypatch):
    monkeypatch.delenv("SGLANG_LOAD_TRANSPOSE_IN_WORKER", raising=False)
    assert qx._transpose_in_worker() is False


def test_die_env_schaltet_nur_den_worker(monkeypatch):
    """#68e: die Env aktiviert den Worker. Ob der Verbraucher seine eigene
    Transposition auslaesst, entscheidet NICHT sie, sondern die Quittung am
    Layer -- sonst ueberspringt er auch dort, wo der Worker nichts getan
    hat (fnFL2w59, "2560 vs 80")."""
    for raw, want in (("1", True), ("0", False), ("", False), ("ja", False)):
        monkeypatch.setenv("SGLANG_LOAD_TRANSPOSE_IN_WORKER", raw)
        assert qx._transpose_in_worker() is want, raw
        # die Env allein quittiert NICHTS
        assert fml.transpose_done_in_worker(types.SimpleNamespace()) is False, raw


def test_die_quittung_steht_am_layer():
    leer = types.SimpleNamespace()
    assert fml.transpose_done_in_worker(leer) is False
    setattr(leer, fml.CT_WORKER_TRANSPOSED_ATTR, True)
    assert fml.transpose_done_in_worker(leer) is True


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
    m = _model(quant="AWQConfig", transponierend=False)
    assert qx._is_ct_wna16_expert_shard(
        "model.layers.0.mlp.experts.3.gate_proj.weight_packed", m
    ) is False
    assert qx._is_ct_wna16_expert_shard("x", types.SimpleNamespace(quant_config=None)) is False


# -- die Eigenschaft: genau einmal -----------------------------------------

def test_der_worker_transponiert_genau_einmal_und_quittiert(monkeypatch):
    """Die Verriegelung von fnFL2v87 ist mit #68a gefallen: beide Seiten
    fragen dieselbe Methode (#68d), und der Verbraucher liest die Quittung
    am Layer statt einer Env (#68e). Was bleibt, ist die Eigenschaft --
    GENAU EINMAL."""
    src = torch.arange(2560 * 80, dtype=torch.int32).reshape(2560, 80)
    name = "model.layers.0.mlp.experts.3.gate_proj.weight_packed"

    monkeypatch.setenv("SGLANG_LOAD_TRANSPOSE_IN_WORKER", "1")
    m = _model()
    out = qx.Qwen4ExpForConditionalGeneration.weight_post_load(m, name, src)
    assert out.shape == (80, 2560)
    assert torch.equal(out, src.t())
    # und der Verbraucher weiss es jetzt -- vom LAYER, nicht von der Env
    layer = m.get_submodule("model.layers.0.mlp.experts")
    assert fml.transpose_done_in_worker(layer) is True

    monkeypatch.setenv("SGLANG_LOAD_TRANSPOSE_IN_WORKER", "0")
    m2 = _model()
    out2 = qx.Qwen4ExpForConditionalGeneration.weight_post_load(m2, name, src)
    assert out2 is src                                    # aus: unveraendert
    layer2 = m2.get_submodule("model.layers.0.mlp.experts")
    assert fml.transpose_done_in_worker(layer2) is False  # -> Verbraucher tut es


def test_keine_quittung_wenn_der_worker_nichts_tat(monkeypatch):
    """Die Asymmetrie, an der w59 starb: sagt der Worker fuer diesen Layer
    nein, darf der Verbraucher NICHT ueberspringen -- auch nicht, wenn die
    Env an ist."""
    monkeypatch.setenv("SGLANG_LOAD_TRANSPOSE_IN_WORKER", "1")
    src = torch.arange(2560 * 80, dtype=torch.int32).reshape(2560, 80)
    m = _model(transponierend=False)   # Layer da, Methode transponiert nicht
    out = qx.Qwen4ExpForConditionalGeneration.weight_post_load(
        m, "model.layers.0.mlp.experts.3.gate_proj.weight_packed", src
    )
    assert out is src
    layer = m.get_submodule("model.layers.0.mlp.experts")
    assert fml.transpose_done_in_worker(layer) is False


def test_the_consumer_reads_the_switch_at_the_right_place():
    import inspect

    src = inspect.getsource(fml.FusedMoE._weight_loader_impl)
    assert "_needs_ct_transpose and transpose_done_in_worker(self)" in src
    # #68 (21.09.): der Verbraucher materialisiert nicht mehr -- `.t()` ohne
    # `.contiguous()`, weil jeder Endpunkt dieses Pfades ohnehin kopiert und
    # ein `copy_` strided Quellen selbst umsortiert (EINE Kopie statt zwei,
    # 2,24x auf dem grossen Shard). Am SCHALTER aendert das nichts: liest er
    # sich als "im Worker erledigt", laesst der Verbraucher die Transposition
    # weiterhin ganz aus.
    assert "loaded_weight.t() if _needs_ct_transpose else loaded_weight" in src


def test_ein_fremder_name_wird_nie_angefasst(monkeypatch):
    """Was kein Experten-Tensor ist, geht unveraendert durch -- auch bei
    eingeschaltetem Schalter."""
    monkeypatch.setenv("SGLANG_LOAD_TRANSPOSE_IN_WORKER", "1")
    t = torch.arange(8, dtype=torch.int32)
    out = qx.Qwen4ExpForConditionalGeneration.weight_post_load(
        _model(), "irgendein.name.ohne.experts", t
    )
    assert out is t


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
    # #68c: der pread-Pfad hat `post_load` schon angewandt, je Thread bzw.
    # je Tensor -- BEIDE Zweige von `pread_safetensors_file`. Liefe die
    # Schleife oben auch fuer ihn, waere es zweimal.
    ps = inspect.getsource(wu.pread_safetensors_file)
    assert ps.count("post_load") >= 2, (
        "ein Zweig von pread_safetensors_file wendet post_load nicht an -- "
        "genau daran starb fnFL2w59"
    )


def test_the_loader_hands_the_models_hook_over():
    import inspect

    from sglang.srt.model_loader import loader as ld

    src = inspect.getsource(ld.DefaultModelLoader._get_all_weights)
    assert 'self._weight_post_load = getattr(model, "weight_post_load", None)' in src
    assert "self._weight_post_load = None" in src
    it = inspect.getsource(ld.DefaultModelLoader._get_weights_iterator)
    assert 'post_load=getattr(self, "_weight_post_load", None)' in it
