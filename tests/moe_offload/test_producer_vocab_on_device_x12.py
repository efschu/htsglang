"""fnFL2x12 (23.09.): the first D->P flip went through, then P PP2 died in its
first prefill:

    ct_embedding.py:292  packed = layer.weight_packed[x]
    RuntimeError: indices should be either on cpu or on the same device as
    the indexed tensor (cpu)

``load_resident_embedding`` rebuilds the draft vocab under the target's
quantization AFTER the draft runner's load, i.e. outside the loader's device
context and its memory-saver region -- VocabParallelEmbedding materialised
the packed table on the CPU. The rebuild has to run inside the drafter's own
scope, and the BF16 table the MTP build made has to go first (x12: it stayed
live, 1212.5 MiB, beside its replacement).
"""

import contextlib
import types

import torch

from sglang.srt.speculative import draft_kv_producer as dkp


def _producer(events):
    embed = torch.nn.Module()
    embed.weight = torch.nn.Parameter(torch.zeros(8, 4, dtype=torch.bfloat16),
                                      requires_grad=False)
    inner = torch.nn.Module()
    inner.embed_tokens = embed
    inner.config = object()

    def _rebuild(config, quant, prefix=""):
        events.append(("rebuild", "scope" in events, len(list(embed.parameters()))))
        new = torch.nn.Module()
        new.weight_packed = torch.nn.Parameter(
            torch.zeros(8, 1, dtype=torch.int32), requires_grad=False)
        return new

    inner._build_embed_tokens = _rebuild
    draft_model = torch.nn.Module()
    draft_model.model = inner
    producer = dkp.DraftKvProducer.__new__(dkp.DraftKvProducer)
    producer.embed_released_mib = 0.0
    producer.draft_runner = types.SimpleNamespace(model=draft_model)
    producer.draft_worker = types.SimpleNamespace(
        target_worker=types.SimpleNamespace(model_runner=types.SimpleNamespace(
            model=types.SimpleNamespace(quant_config=object(), lm_head=None))))

    @contextlib.contextmanager
    def _scope():
        events.append("scope")
        yield

    producer._draft_weights_scope = _scope
    return producer, inner, embed


def test_x12_the_vocab_is_rebuilt_inside_the_drafters_scope_after_the_bf16_release(
        monkeypatch):
    events = []
    producer, inner, old = _producer(events)
    monkeypatch.setattr(dkp, "_iter_checkpoint_tensors", lambda *a, **k: iter(()))
    try:
        producer.load_resident_embedding("/nonexistent")
    except RuntimeError:
        pass  # the fake checkpoint has no rows; the rebuild already ran
    rebuild = [e for e in events if isinstance(e, tuple)]
    assert rebuild, "the rebuild never ran"
    _, in_scope, old_params_left = rebuild[0]
    assert in_scope, "rebuilt outside the drafter's device/region scope (x12)"
    assert old_params_left == 0, "the BF16 table must be released first"
    assert producer.embed_released_mib > 0
    assert inner.embed_tokens is not old


def test_x13_the_scope_carries_the_drafters_dtype(monkeypatch):
    """x13: device alone was not enough -- the packed vocab took the process
    default (float32) as params_dtype, dequantized float32 rows and the MTP
    layer's gemma_rmsnorm refused them ('failed to dispatch data type Float').
    The scope opens the loader's dtype context like the draft build had."""
    from sglang.srt.managers import weg2_memory_saver as ms

    monkeypatch.setattr(ms, "weights_region",
                        lambda *a, **k: contextlib.nullcontext(), raising=True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0, raising=True)
    producer = dkp.DraftKvProducer.__new__(dkp.DraftKvProducer)
    producer.draft_runner = types.SimpleNamespace(
        _weg2_manifest_identity={"region_tag": "weights_draft"},
        server_args=types.SimpleNamespace(enable_weights_cpu_backup=False,
                                          enable_draft_weights_cpu_backup=False),
        memory_saver_adapter=None,
        model_config=types.SimpleNamespace(dtype=torch.bfloat16))
    before = torch.get_default_dtype()
    with producer._draft_weights_scope():
        assert torch.get_default_dtype() == torch.bfloat16
    assert torch.get_default_dtype() == before
