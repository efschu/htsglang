"""Task #46: the hyper-connection mixers can stay quantized (ReplicatedLinear
with a quant config) -- the mix() math must equal the widened BF16 path."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch
import torch.nn as nn

from sglang.srt.layers import hyperconnection as hc


class _StubQuantLinear(nn.Module):
    """Stands in for ReplicatedLinear(quant_config=...): returns (x W^T, None)."""

    made = []

    def __init__(self, in_f, out_f, bias=False, quant_config=None, prefix="", params_dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f, dtype=torch.float32) * 0.05)
        self.prefix = prefix
        _StubQuantLinear.made.append(prefix)

    def forward(self, x):
        return x.float() @ self.weight.t(), None


def _cfg():
    return hc.HyperConnectionConfig(
        hc_count=2, hidden_size=16, params_dtype=torch.float32, hc_lowrank=8,
        rms_norm_eps=1e-6, hc_per_branch_norm=True,
    )


def test_quantized_mixers_match_the_dense_math(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setenv("SGLANG_HC_MIXER_INT8", "1")
    import sglang.srt.layers.linear as lin

    monkeypatch.setattr(lin, "ReplicatedLinear", _StubQuantLinear)
    _StubQuantLinear.made.clear()
    q = hc.GatedResidual(_cfg(), use_mix=True, use_combine=False,
                         quant_config=object(), prefix="model.language_model.layers.3.attn_hyper_connection")
    assert q._mix_quantized and _StubQuantLinear.made == [
        "model.language_model.layers.3.attn_hyper_connection.input_mix_weight_down",
        "model.language_model.layers.3.attn_hyper_connection.input_mix_weight_up",
    ]
    monkeypatch.setenv("SGLANG_HC_MIXER_INT8", "0")
    d = hc.GatedResidual(_cfg(), use_mix=True, use_combine=False)
    assert not d._mix_quantized and isinstance(d.input_mix_weight_down, nn.Linear)
    with torch.no_grad():
        d.input_mix_weight_down.weight.copy_(q.input_mix_weight_down.weight)
        d.input_mix_weight_up.weight.copy_(q.input_mix_weight_up.weight)
        d.hc_norm.weight.copy_(q.hc_norm.weight)
    x = torch.randn(3, 32)
    out_q, (hi_q, hn_q) = q.mix(x)
    out_d, (hi_d, hn_d) = d.mix(x)
    assert out_q.shape == (3, 16)
    assert torch.allclose(out_q, out_d, atol=1e-5), (out_q - out_d).abs().max()
    assert torch.equal(hn_q, hn_d)


def test_switch_semantics(monkeypatch):
    monkeypatch.setenv("SGLANG_HC_MIXER_INT8", "1")
    assert hc.hc_mixer_int8_on(object()) and not hc.hc_mixer_int8_on(None)
    monkeypatch.setenv("SGLANG_HC_MIXER_INT8", "0")
    assert not hc.hc_mixer_int8_on(object())
    monkeypatch.delenv("SGLANG_HC_MIXER_INT8")
    assert not hc.hc_mixer_int8_on(object())
