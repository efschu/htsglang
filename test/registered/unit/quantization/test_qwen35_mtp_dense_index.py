"""The MTP draft is built unquantized when the checkpoint index carries no
quantized tensor under ``mtp.`` (Minachist / cyankiwi Qwen3.8-Flash-Next ship
a bf16 draft that the compressed-tensors config neither targets nor ignores);
it stays quantized when the index shows ``mtp.*.weight_scale``
(Qwen3.8-27B-INT8)."""

import json
import os
from types import SimpleNamespace

import pytest

from sglang.srt.models import qwen3_5_mtp as m


def _index(tmp_path, keys):
    os.makedirs(tmp_path, exist_ok=True)
    with open(os.path.join(tmp_path, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in keys}}, f)
    return str(tmp_path)


DENSE = ["model.layers.0.mlp.experts.0.gate_proj.weight_packed", "mtp.layers.0.self_attn.q_proj.weight", "mtp.fc.weight"]
QUANT = ["mtp.layers.0.self_attn.q_proj.weight", "mtp.layers.0.self_attn.q_proj.weight_scale", "mtp.fc.weight"]


def test_index_verdicts(tmp_path):
    assert m.mtp_index_is_dense(_index(tmp_path / "a", DENSE)) is True
    assert m.mtp_index_is_dense(_index(tmp_path / "b", QUANT)) is False
    assert m.mtp_index_is_dense(_index(tmp_path / "c", ["model.embed_tokens.weight"])) is None
    assert m.mtp_index_is_dense(str(tmp_path / "missing")) is None
    assert m.mtp_index_is_dense(None) is None


@pytest.mark.parametrize("keys,expect_quant", [(DENSE, False), (QUANT, True), (["model.norm.weight"], True)])
def test_mtp_quant_config_follows_the_index(tmp_path, monkeypatch, keys, expect_quant):
    path = _index(tmp_path / "x", keys)
    monkeypatch.setattr(m, "get_server_args", lambda: SimpleNamespace(model_path=path, speculative_draft_model_quantization=None))
    monkeypatch.setattr(m, "is_npu", lambda: False)
    ct = SimpleNamespace(get_name=lambda: "compressed-tensors")
    assert (m._mtp_quant_config(ct) is ct) is expect_quant
    # other quant families are untouched by the index rule
    fp8 = SimpleNamespace(get_name=lambda: "fp8")
    assert m._mtp_quant_config(fp8) is fp8
    assert m._mtp_quant_config(None) is None
