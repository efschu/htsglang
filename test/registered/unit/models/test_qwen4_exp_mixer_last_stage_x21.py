"""fnFL2x21 (2026-09-23): the model-level hyper-connection mixer belongs to
the LAST pipeline stage only.

forward() hands every non-last stage's stream on before ``mix``, but the mixer
was built on every stage. The flip join takes the first stage that publishes a
replicated name as its holder, so the bytes moved to PP0: the one stage that
mixes (PP2) received nothing at a D->P wake, and PP1's dead copy had no source
at the P->D sleep -- ``W106 Weg2XchgWakeSourceGapRefused ... tag=weights
expected_bytes=23068672`` on P rank 1, every rank stopped in the fence.
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.models import qwen4_exp as m


def _stage(*, last: bool):
    return SimpleNamespace(pp_group=SimpleNamespace(is_last_rank=last))


def test_a_non_last_stage_builds_no_mixer(monkeypatch):
    """RED ON d8cdfa4198: every stage built a GatedResidual."""
    monkeypatch.setattr(
        m, "GatedResidual", lambda *a, **k: pytest.fail("mixer built off its stage")
    )
    mixer = m.Qwen4ExpModel._build_hyper_connection_mixer(_stage(last=False), None)
    assert isinstance(mixer, PPMissingLayer)


def test_the_last_stage_builds_the_mixer(monkeypatch):
    built = []
    monkeypatch.setattr(m, "GatedResidual", lambda cfg, **k: built.append(cfg) or "MIXER")
    assert m.Qwen4ExpModel._build_hyper_connection_mixer(_stage(last=True), "cfg") == "MIXER"
    assert built == ["cfg"]


def test_the_loader_skips_the_mixer_only_where_it_has_no_module():
    """Without the skip a non-last stage refuses the packed mixer tensors
    (``load_packed_hc_linear``: packed hyper-connection weight without a
    module) and never finishes loading."""
    name = "model.hyper_connection_mixer.input_mix_weight_down.weight_packed"
    foreign = SimpleNamespace(hyper_connection_mixer=PPMissingLayer())
    owner = SimpleNamespace(hyper_connection_mixer=torch.nn.Identity())
    assert m.mixer_is_foreign(foreign, name)
    assert not m.mixer_is_foreign(owner, name)
    # the per-layer mixers are decided by the stage range, never here
    layer_mixer = "model.layers.29.attn_hyper_connection.input_mix_weight_down.weight_packed"
    assert not m.mixer_is_foreign(foreign, layer_mixer)
