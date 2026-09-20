"""fnFA24 (20.09.): the adaptive runtime-state builder on a Form A expert
worker must take the worker's no-op attention backend, not construct a real
one -- the worker's kv-head share is 0 and flashinfer divides by it
(ZeroDivisionError in should_use_tensor_core, TP1/TP2 dead at boot)."""

import types

from sglang.srt.speculative import eagle_worker_v2 as ew


def test_a_form_a_worker_gets_its_own_worker_backend_per_state(monkeypatch):
    built = []

    class _Stub:
        def __init__(self, runner):
            built.append(runner)

    import sglang.srt.form_a_construction as fac

    monkeypatch.setattr(fac, "FormAWorkerAttnBackend", _Stub)
    calls = []
    runner = types.SimpleNamespace(
        is_form_a_worker=True,
        init_new_workspace=False,
        _get_attention_backend=lambda **kw: calls.append(kw) or object(),
    )
    a = ew.adaptive_target_attn_backend(runner)
    b = ew.adaptive_target_attn_backend(runner)
    assert isinstance(a, _Stub) and isinstance(b, _Stub) and a is not b
    assert calls == []  # never the real builder on a worker
    assert built == [runner, runner]


def test_every_other_rank_builds_a_private_workspace_backend_and_restores_the_flag():
    calls = []
    sentinel = object()

    def _get(**kw):
        calls.append((kw, runner.init_new_workspace))
        return sentinel

    runner = types.SimpleNamespace(
        is_form_a_worker=False, init_new_workspace=False, _get_attention_backend=_get
    )
    assert ew.adaptive_target_attn_backend(runner) is sentinel
    assert calls == [({"init_new_workspace": True}, False)]
    assert runner.init_new_workspace is False  # restored after the build


def test_the_builder_uses_the_helper():
    import inspect

    src = inspect.getsource(ew.EAGLEWorkerV2.build_adaptive_runtime_state)
    assert "target_attn_backend = adaptive_target_attn_backend(target_model_runner)" in src
    assert "_get_attention_backend(" not in src
