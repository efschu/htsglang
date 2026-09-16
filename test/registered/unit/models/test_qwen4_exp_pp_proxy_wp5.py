"""WP5 (PP=3 prefill layout): Qwen4-Exp's forward under pipeline
parallelism. The activation that crosses a stage boundary is the
hyper-connection stream, carried as ``hidden_states`` with a None residual;
the first stage embeds, later stages take the stream out of the proxy, a
non-last stage returns a PPProxyTensors, and the PLE batch is only built
on the stage that owns a PLE layer. Pinned with stubs (no kernels)."""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import PPProxyTensors
from sglang.srt.models import qwen4_exp as m


class _Layer(torch.nn.Module):
    def __init__(self, tag, ple=None):
        super().__init__()
        self.tag, self.ple, self.calls = tag, ple, []

    def forward(self, positions, hidden_states, residual, forward_batch, ple_batch, captured_last_layer_outputs):
        self.calls.append((residual, ple_batch))
        return hidden_states + self.tag, None


def _stub(pp_rank, pp_size, start, end, n_layers=6, ple_at=None):
    layers = [_Layer(float(i + 1), ple=SimpleNamespace(start_prefetch=lambda *a: None) if i == ple_at else None) for i in range(n_layers)]
    st = SimpleNamespace(
        pp_group=SimpleNamespace(is_first_rank=pp_rank == 0, is_last_rank=pp_rank == pp_size - 1),
        embed_tokens=lambda ids: torch.zeros(ids.shape[0], 4) + 100.0,
        has_ple=ple_at is not None,
        ple_ngram_size=3,
        ple_ngram_eos_token_id=0,
        layers=layers,
        start_layer=start,
        end_layer=end,
        hyper_connection_mixer=SimpleNamespace(mix=lambda h: (h * 0.5, None)),
    )
    st._stage_has_ple = st.has_ple and any(getattr(layers[i], "ple", None) is not None for i in range(start, end))
    return st


def _forward(st, **kw):
    fb = SimpleNamespace(forward_mode=SimpleNamespace(is_idle=lambda: False), input_ids=None)
    return m.Qwen4ExpModel.forward(st, torch.tensor([1, 2]), torch.tensor([0, 1]), fb, **kw)


def test_first_stage_embeds_and_hands_the_stream_on(monkeypatch):
    monkeypatch.setattr(m, "_prepare_ple_batch", lambda *a, **k: "PLE")
    committed = []
    monkeypatch.setattr(m, "_commit_ple_batch", lambda b, fb: committed.append(b))
    st = _stub(0, 3, 0, 2, ple_at=1)
    out = _forward(st)
    assert isinstance(out, PPProxyTensors)
    assert torch.equal(out["hidden_states"], torch.full((2, 4), 103.0))  # 100 + 1 + 2
    assert "residual" not in out.tensors
    assert st.layers[1].calls[0][1] == "PLE" and committed == ["PLE"]


def test_middle_stage_takes_the_stream_from_the_proxy_and_builds_no_ple(monkeypatch):
    monkeypatch.setattr(m, "_prepare_ple_batch", lambda *a, **k: pytest.fail("PLE built off its stage"))
    st = _stub(1, 3, 2, 4, ple_at=1)  # the PLE layer (1) belongs to stage 0
    out = _forward(st, pp_proxy_tensors=PPProxyTensors({"hidden_states": torch.zeros(2, 4)}))
    assert isinstance(out, PPProxyTensors)
    assert torch.equal(out["hidden_states"], torch.full((2, 4), 7.0))  # 3 + 4
    assert st.layers[2].calls[0] == (None, None)


def test_last_stage_mixes_and_returns_the_hc_stream_too():
    st = _stub(2, 3, 4, 6)
    out = _forward(st, pp_proxy_tensors=PPProxyTensors({"hidden_states": torch.zeros(2, 4)}))
    hidden, hc = out
    assert torch.equal(hc, torch.full((2, 4), 11.0)) and torch.equal(hidden, hc * 0.5)


def test_a_later_stage_without_a_proxy_refuses():
    with pytest.raises(AssertionError, match="pp_proxy_tensors"):
        _forward(_stub(1, 3, 2, 4))


def test_single_stage_is_the_old_path(monkeypatch):
    monkeypatch.setattr(m, "_prepare_ple_batch", lambda *a, **k: "PLE")
    monkeypatch.setattr(m, "_commit_ple_batch", lambda b, fb: None)
    st = _stub(0, 1, 0, 6, ple_at=1)
    hidden, hc = _forward(st)
    assert torch.equal(hc, torch.full((2, 4), 121.0))
