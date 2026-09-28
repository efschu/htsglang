"""#239 M1s: a Form A worker's eager forward resolved rows against STALE metadata.

Metal (rc12z30g 7bd3541c4f, D-only -st-cut-vsync, cut [0,32,32], D log
22:07:33): the S1 probe's second 4096-token chunk (an extend WITH a prefix, so
the worker joins the rows path) died on TP1/TP2 in
``qwen_sparse_attn_backend.form_a_worker_attention`` -> ``_rows_and_counts``
-> ``qsa/rows_resolve.py`` with ``ValueError: QSA top-k rows do not match
query rows``, right after the host's top-k gather (collective T) completed.

Root: the host builds attention metadata per forward in its eager runner; the
worker's eager entry ``ModelRunner._forward_form_a_worker`` bypasses that
runner, and the backend's lazy ``_resolve_metadata`` builds only when
``forward_metadata is None`` -- so the worker resolved the host's 4096 rows
against the last graph capture/replay metadata. Same-sized stale metadata (an
eager decode after a graph replay) attends wrong KV rows silently.

Hermetic, CPU: the real ``ModelRunner._forward_form_a_worker`` on a stand-in
runner, the real ``form_a_attends``. RED on 7bd3541c4f / ca2a9706ec, GREEN with
the fix.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.model_runner import ModelRunner

ROWS = 4096      # the metal chunk: GDN-EXTEND cu_seqlens=[0, 4096]
PREFIX = 4096    # second chunk of the S1 probe: the first 4096 are the prefix
GRAPH_BS = 6     # a decode-graph capture/replay batch (--cuda-graph-bs-decode 1..6)


class _Backend:
    """The worker backend's metadata surface: ``forward_metadata`` as the last
    graph capture/replay left it, and the per-forward build the host's eager
    runner performs."""

    def __init__(self, form_a_dcp=object()):
        self.form_a_dcp = form_a_dcp
        self.forward_metadata = SimpleNamespace(
            token_to_batch_idx=torch.arange(GRAPH_BS, dtype=torch.int32), is_cuda_graph=True
        )
        self.built = 0

    def init_forward_metadata(self, fb):
        self.built += 1
        bs = int(fb.extend_seq_lens.numel())
        self.forward_metadata = SimpleNamespace(
            token_to_batch_idx=torch.repeat_interleave(
                torch.arange(bs, dtype=torch.int32), fb.extend_seq_lens.long()
            ),
            is_cuda_graph=False,
        )


def _batch(mode=ForwardMode.EXTEND, rows=ROWS, prefix=PREFIX):
    return SimpleNamespace(
        forward_mode=mode,
        input_ids=torch.zeros(rows, dtype=torch.int64),
        seq_lens_cpu=torch.tensor([prefix + rows]),
        extend_seq_lens_cpu=torch.tensor([rows]),
        extend_seq_lens=torch.tensor([rows]),
    )


def _runner(backend, seen):
    def route(fb, n):
        # what the worker's rows path compares (rows_resolve.py Tq-check)
        seen.append((int(backend.forward_metadata.token_to_batch_idx.numel()), int(n)))

    return SimpleNamespace(
        attn_backend=backend,
        _prepare_eager_forward_batch=lambda fb: None,
        run_form_a_worker_route=route,
    )


def test_the_metal_chunk_resolves_against_this_forwards_rows():
    backend, seen = _Backend(), []
    ModelRunner._forward_form_a_worker(_runner(backend, seen), _batch())
    assert seen == [(ROWS, ROWS)]  # base: (6, 4096) -> 'QSA top-k rows do not match query rows'
    assert backend.built == 1


def test_stale_graph_metadata_is_exactly_the_metal_error():
    from sglang.srt.layers.attention.qsa.rows_resolve import qsa_rows_resolve

    stale = _Backend().forward_metadata
    with pytest.raises(ValueError, match="QSA top-k rows do not match query rows"):
        qsa_rows_resolve(
            torch.zeros((ROWS, 8), dtype=torch.int32), stale.token_to_batch_idx,
            torch.tensor([PREFIX + ROWS]), torch.tensor([1]), torch.zeros((2, 16), dtype=torch.int32),
        )


def test_eager_decode_after_a_replay_rebuilds_too():
    # the silent twin: same row count, wrong table
    backend, seen = _Backend(), []
    fb = _batch(ForwardMode.DECODE, rows=GRAPH_BS, prefix=1000)
    fb.extend_seq_lens = torch.ones(GRAPH_BS, dtype=torch.int64)
    ModelRunner._forward_form_a_worker(_runner(backend, seen), fb)
    assert backend.built == 1 and backend.forward_metadata.is_cuda_graph is False


def test_first_chunk_without_prefix_builds_nothing():
    backend, seen = _Backend(), []
    ModelRunner._forward_form_a_worker(_runner(backend, seen), _batch(prefix=0))
    assert backend.built == 0 and seen == [(GRAPH_BS, ROWS)]  # no rows path runs


def test_classic_form_a_without_the_cut_is_untouched():
    backend, seen = _Backend(form_a_dcp=None), []
    ModelRunner._forward_form_a_worker(_runner(backend, seen), _batch())
    assert backend.built == 0
