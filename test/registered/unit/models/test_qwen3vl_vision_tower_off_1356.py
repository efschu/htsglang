"""#1356 slice 2 -- text-only means NO TOWER, not a tower nobody calls.

MEASURED on weg2xsn63 (3dbc84099e): the argv carried ``--no-enable-multimodal``
and the server args printed ``enable_multimodal=False``, yet every P rank's
exchange manifest still listed 333 ``visual.*`` pieces (921,460,192 bytes).
The flag reached the tokenizer's image path and never the model constructor.

Three properties, each with the mutant that was shipped:
  * the predicate reads ONLY the explicit ``False`` of the tri-state
    (mutant: ``not enable_multimodal`` would also drop the tower on ``None``);
  * the dense loader skips a tower tensor BEFORE looking it up when there is
    no tower (mutant: the pre-#1356 loader, which renamed and loaded it);
  * with the tower built nothing is skipped (control -- a fixture that only
    has defect cases cannot tell a broken stub from a finding).

Hermetic: ``CUDA_VISIBLE_DEVICES=""``, no distributed init, no model build.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402

from sglang.srt.models import qwen3_5, qwen3_vl  # noqa: E402


class _Args:
    def __init__(self, enable_multimodal):
        self.enable_multimodal = enable_multimodal


def test_predicate_reads_only_the_explicit_false(monkeypatch):
    seen = {}
    for value, expect in ((False, True), (None, False), (True, False)):
        monkeypatch.setattr(qwen3_vl, "get_server_args", lambda v=value: _Args(v))
        seen[value] = qwen3_vl.vision_tower_forced_off()
        assert seen[value] is expect, (value, seen[value])


def test_predicate_is_false_without_server_args(monkeypatch):
    def _raise():
        raise RuntimeError("no server args on the desk")

    monkeypatch.setattr(qwen3_vl, "get_server_args", _raise)
    assert qwen3_vl.vision_tower_forced_off() is False


def test_skip_only_when_tower_absent_and_name_is_tower():
    assert qwen3_vl.skip_vision_weight("model.visual.blocks.0.attn.proj.bias", None)
    assert not qwen3_vl.skip_vision_weight("model.layers.0.input_layernorm.weight", None)
    assert not qwen3_vl.skip_vision_weight("model.visual.blocks.0.attn.proj.bias", object())


def _stub(visual):
    """The attributes ``Qwen3_5ForConditionalGeneration.load_weights`` touches
    on ``self`` for a single tower tensor -- nothing else is built."""
    calls = []
    param = torch.nn.Parameter(torch.zeros(4), requires_grad=False)
    param.weight_loader = lambda p, w, *rest: calls.append((w.shape, rest))
    stub = SimpleNamespace(
        visual=visual,
        config=SimpleNamespace(tie_word_embeddings=False),
        pp_group=SimpleNamespace(is_last_rank=True),
        # the stub carries the tower parameter EVEN when `visual` is None so
        # that the pre-guard loader (the mutant) has somewhere to land the
        # tensor and the call edge becomes observable
        named_parameters=lambda remove_duplicate=False: [
            ("visual.blocks.0.attn.proj.bias", param)
        ],
    )
    return stub, calls


_TOWER_TENSOR = ("model.visual.blocks.0.attn.proj.bias", torch.ones(4))


def test_dense_loader_skips_tower_tensor_when_no_tower():
    stub, calls = _stub(visual=None)
    loaded = qwen3_5.Qwen3_5ForConditionalGeneration.load_weights(
        stub, [_TOWER_TENSOR]
    )
    assert calls == [], f"tower tensor was loaded without a tower: {calls}"
    assert loaded == set()


def test_dense_loader_control_loads_tower_tensor_with_tower():
    stub, calls = _stub(visual=object())
    loaded = qwen3_5.Qwen3_5ForConditionalGeneration.load_weights(
        stub, [_TOWER_TENSOR]
    )
    assert len(calls) == 1, calls
    assert loaded == {"visual.blocks.0.attn.proj.bias"}


def test_base_loader_skips_tower_tensor_when_no_tower():
    stub, calls = _stub(visual=None)
    stub.model = SimpleNamespace(start_layer=0, end_layer=64)
    qwen3_vl.Qwen3VLForConditionalGeneration.load_weights(stub, [_TOWER_TENSOR])
    assert calls == [], calls


def test_image_input_refuses_without_tower():
    stub = SimpleNamespace(visual=None)
    try:
        qwen3_vl.Qwen3VLForConditionalGeneration._require_visual(stub, "image")
    except RuntimeError as exc:
        assert "text-only" in str(exc)
    else:
        raise AssertionError("image input reached a None tower without refusal")
