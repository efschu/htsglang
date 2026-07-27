"""PS2 (born-spilled prefill) x speculative decoding.

A born-spilled-deep prefill allocates NO device KV slots: its ``out_cache_loc``
is a row of HOST SENTINELS (``kv_session_offload.spill_extend_alloc``). The
draft extend reuses that very tensor through ``ForwardBatch.init_new(batch,
self.draft_runner)``, so running it would scatter draft KV far outside the
draft pool.

For such a request nothing ever reads the draft extend's output, so the extend
is skipped and a shape-valid stub is returned instead. These tests pin the skip
itself (no draft forward is issued) and the stub's shape contract. CPU only.
"""

import types

import pytest
import torch

from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker
from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
    MultiLayerEagleDraftWorker,
)


class _ExplodingRunner:
    """Any use of the draft runner in this path is the bug under test."""

    def __init__(self):
        self.spec_algorithm = types.SimpleNamespace(is_standalone=lambda: False)
        self.model_config = types.SimpleNamespace(
            spec_hidden_size=8, dtype=torch.float32, vocab_size=32
        )

    def forward(self, *a, **kw):  # pragma: no cover - must never be reached
        raise AssertionError(
            "draft extend ran on a born-spilled prefill: it would scatter "
            "draft KV at host sentinel slots"
        )


def _fake_draft_worker(cls):
    """A ``cls``-typed object carrying only what the guard path may touch."""
    w = cls.__new__(cls)
    w.topk = 2
    w.draft_runner = _ExplodingRunner()
    w.model_config = w.draft_runner.model_config
    w.speculative_algorithm = w.draft_runner.spec_algorithm
    return w


def _born_spilled_batch(bs=1):
    return types.SimpleNamespace(
        kv_session_prefill_spill=True,
        seq_lens=torch.zeros(bs, dtype=torch.int64),
        forward_mode=types.SimpleNamespace(is_idle=lambda: False),
    )


@pytest.mark.parametrize("cls", [EagleDraftWorker, MultiLayerEagleDraftWorker])
def test_draft_extend_for_prefill_is_skipped_when_born_spilled(cls):
    """The guard sits BEFORE anything that touches the draft runner, so the
    out-of-bounds scatter can no longer happen. Removing the guard makes
    _ExplodingRunner fire (EagleDraftWorker) or the OOB probe run on a fake
    model config (MultiLayerEagleDraftWorker) -- either way, red."""
    w = _fake_draft_worker(cls)
    batch = _born_spilled_batch()
    next_token_ids = torch.tensor([7], dtype=torch.int64)

    out = cls._draft_extend_for_prefill(
        w, batch, torch.zeros(4, 8), next_token_ids
    )

    # bonus_tokens is the one field a merge/filter step may still move around
    # before the request is adopted by the spill tick, so it must be real.
    assert out.bonus_tokens is next_token_ids
    assert out.num_tokens_per_req == 1


def test_stub_shapes_match_what_the_future_map_stashes():
    w = _fake_draft_worker(EagleDraftWorker)
    batch = _born_spilled_batch(bs=3)
    next_token_ids = torch.tensor([1, 2, 3], dtype=torch.int64)

    out = w.born_spilled_stub_draft_input(batch, next_token_ids)

    assert out.topk_p.shape == (3, w.topk)
    assert out.topk_index.shape == (3, w.topk)
    assert out.topk_index.dtype == torch.int64
    assert out.hidden_states.shape == (3, 8)
    assert out.hidden_states.dtype == torch.float32


def test_standalone_draft_carries_no_hidden_states():
    """STANDALONE skips hidden states end to end; the stub must not invent a
    buffer the rest of the pipeline would then try to consume."""
    w = _fake_draft_worker(EagleDraftWorker)
    w.draft_runner.spec_algorithm = types.SimpleNamespace(
        is_standalone=lambda: True
    )
    out = w.born_spilled_stub_draft_input(
        _born_spilled_batch(), torch.tensor([5], dtype=torch.int64)
    )
    assert out.hidden_states is None


def test_a_normal_prefill_is_not_affected():
    """Without the born-spilled marker the guard is inert -- the draft extend
    proceeds and reaches the draft runner (here: the exploding stub)."""
    w = _fake_draft_worker(EagleDraftWorker)
    batch = _born_spilled_batch()
    batch.kv_session_prefill_spill = False
    batch.input_ids = torch.zeros(4, dtype=torch.int64)
    batch.extend_lens = [4]
    batch.seq_lens = torch.tensor([4], dtype=torch.int64)
    with pytest.raises(Exception):
        EagleDraftWorker._draft_extend_for_prefill(
            w, batch, torch.zeros(4, 8), torch.tensor([7], dtype=torch.int64)
        )
