"""fnFA7 (20.09. 12:57Z): a Form A worker must not build a real attention
backend (flashinfer's should_use_tensor_core divided by its 0 kv heads).
It gets FormAWorkerAttnBackend: bookkeeping as no-ops, attention refused."""

import types

import pytest

from sglang.srt.form_a_construction import (
    FormAWorkerAttentionUsed,
    FormAWorkerAttnBackend,
)


def _runner(is_worker):
    # No name stamps here on purpose: the real runner gets them from
    # _get_attention_backend, which a worker never calls (fnFA9 died on
    # exactly that AttributeError because the stub used to carry them).
    r = types.SimpleNamespace(
        is_form_a_worker=is_worker,
        model_config=types.SimpleNamespace(context_len=262144),
        server_args=types.SimpleNamespace(enable_pdmux=False, enable_two_batch_overlap=False),
        is_draft_worker=False,
    )

    def _real(*a, **k):
        r.prefill_attention_backend_str = "flashinfer"
        r.decode_attention_backend_str = "flashinfer"
        return types.SimpleNamespace()

    r._real = _real
    return r


def test_bookkeeping_is_a_no_op_and_attention_refuses():
    b = FormAWorkerAttnBackend(_runner(True))
    fb = object()
    assert b.init_forward_metadata(fb) is None
    assert b.init_forward_metadata_out_graph(fb, in_capture=True) is None
    assert b.init_cuda_graph_state(4, 64) is None
    assert b.get_verify_buffers_to_fill_after_draft() == [None, None]
    assert b.max_context_len == 262144
    for name in ("forward", "forward_decode", "forward_extend", "forward_mixed"):
        with pytest.raises(FormAWorkerAttentionUsed, match=name):
            getattr(b, name)()


def test_runner_picks_the_worker_backend_only_on_a_worker(monkeypatch):
    from sglang.srt.model_executor import model_runner as mr

    picked = []
    r = _runner(True)
    r._get_attention_backend = lambda *a, **k: picked.append("real") or r._real()
    mr.ModelRunner.init_attention_backend(r)
    assert isinstance(r.attn_backend, FormAWorkerAttnBackend) and picked == []
    assert r.attn_backend.prefill_attention_backend_str == "form_a_worker"
    assert r.attn_backend.decode_attention_backend_str == "form_a_worker"
    r = _runner(False)
    r._get_attention_backend = lambda *a, **k: picked.append("real") or r._real()
    mr.ModelRunner.init_attention_backend(r)
    assert picked == ["real"] and not isinstance(r.attn_backend, FormAWorkerAttnBackend)
    assert r.attn_backend.prefill_attention_backend_str == "flashinfer"
