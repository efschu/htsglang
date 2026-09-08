# SPDX-License-Identifier: Apache-2.0
"""#1233 FIX 4 -- the weg2tr1 boot killer, at its two roots.

Boot weg2tr1 (tip 19f9f98faa) died 63 s in: group P's PP1 and PP2 both raised
inside DECODE cuda-graph capture, target-verify branch, at
``gdn_backend.py`` ``mixed_qkv.view(batch_size, draft_token_num, -1)``::

    RuntimeError: shape '[2, 3, -1]' is invalid for input of size 81920

The two roots, and what this file pins:

1. **Group P captured a target-verify graph at all.** It is Weg 2's PREFILL
   group -- it writes draft KV rows during prefill and never decodes,
   proposes or verifies. It only entered TARGET_VERIFY shape because
   ``--speculative-draft-kv-only`` must carry the decode group's full
   speculative flag set byte-for-byte (drafter identity). Pinned:
   ``ModelRunner.is_draft_kv_only_producer`` and the carve it drives.
2. **The floor division hid the mismatch.** ``seq_len // draft_token_num``
   turned 8 // 3 into 2 and the view then demanded 6 tokens' worth of a
   tensor holding 8. Pinned: ``target_verify_batch_size`` refuses the
   non-multiple BY NAME instead of flooring.

Pure arithmetic + pure predicate: no GPU, no model, no ForwardBatch.
"""

from __future__ import annotations

import types

import pytest

from sglang.srt.layers.attention.linear.gdn_backend import (
    GdnTargetVerifyRaggedTokens,
    target_verify_batch_size,
)


# ---------------------------------------------------------------- root 2 ----


def test_uniform_windows_still_divide():
    """The whole point of the branch: exact windows keep working, unchanged."""
    assert target_verify_batch_size(24, 3) == 8
    assert target_verify_batch_size(3, 3) == 1
    assert target_verify_batch_size(0, 3) == 0
    assert target_verify_batch_size(64, 4) == 16


def test_the_weg2tr1_numbers_refuse_instead_of_flooring():
    """8 tokens against a 3-token window: the exact shape that killed weg2tr1.

    Red-first: against the pre-fix ``seq_len // draft_token_num`` this returns
    2 and the caller's ``.view(2, 3, -1)`` is what raises, three frames later,
    naming neither 8 nor 3 nor the remainder.
    """
    with pytest.raises(GdnTargetVerifyRaggedTokens) as excinfo:
        target_verify_batch_size(8, 3)
    msg = str(excinfo.value)
    # Every number a reader needs is IN the message, not reconstructible from
    # an element count: seq_len, draft_token_num, the remainder, and what the
    # silent floor would have produced.
    assert "seq_len=8" in msg
    assert "draft_token_num=3" in msg
    assert "remainder=2" in msg
    assert "floor to 2" in msg


@pytest.mark.parametrize(
    "seq_len,draft_token_num,remainder",
    [(8, 3, 2), (7, 3, 1), (10, 4, 2), (1, 2, 1), (25, 5 + 1, 1)],
)
def test_every_non_multiple_refuses(seq_len, draft_token_num, remainder):
    with pytest.raises(GdnTargetVerifyRaggedTokens) as excinfo:
        target_verify_batch_size(seq_len, draft_token_num)
    assert f"remainder={remainder}" in str(excinfo.value)


@pytest.mark.parametrize("bad", [0, -1, None])
def test_non_positive_window_refuses_by_name(bad):
    """A zero window would raise ZeroDivisionError three frames from its cause."""
    with pytest.raises(GdnTargetVerifyRaggedTokens):
        target_verify_batch_size(8, bad)


def test_refusal_is_not_a_bare_runtimeerror_in_the_view():
    """The class must be catchable/greppable on its own name."""
    assert issubclass(GdnTargetVerifyRaggedTokens, RuntimeError)


def test_gdn_backend_calls_the_guard_not_the_bare_floor():
    """The forward_extend body must go through the guard.

    A revert that restores ``seq_len // forward_batch.spec_info.draft_token_num``
    would leave every test above green while the boot killer is back, so the
    call site itself is pinned (the desk-written-never-executed rule applied to
    a one-line arithmetic swap).
    """
    import inspect

    from sglang.srt.layers.attention.linear import gdn_backend

    src = inspect.getsource(gdn_backend.GDNAttnBackend.forward_extend)
    assert "target_verify_batch_size(seq_len, draft_token_num)" in src
    assert "seq_len // forward_batch.spec_info.draft_token_num" not in src


# ---------------------------------------------------------------- root 1 ----


def _runner(**kw):
    """A ModelRunner stand-in carrying only what the predicate reads."""
    from sglang.srt.model_executor.model_runner import ModelRunner

    obj = types.SimpleNamespace(
        server_args=types.SimpleNamespace(
            speculative_draft_kv_only=kw.pop("draft_kv_only", False)
        ),
        is_draft_worker=kw.pop("is_draft_worker", False),
        **kw,
    )
    # Bind the real property to the double: the predicate under test is the
    # production one, not a re-implementation.
    return ModelRunner.is_draft_kv_only_producer.fget(obj)


def test_producer_predicate_is_true_only_on_the_producer_target():
    assert _runner(draft_kv_only=True, is_draft_worker=False) is True
    # The draft runner is already eager via --disable-draft-cuda-graph and its
    # own pools must keep draft sizing.
    assert _runner(draft_kv_only=True, is_draft_worker=True) is False
    # Every ordinary speculative server is byte-inert.
    assert _runner(draft_kv_only=False, is_draft_worker=False) is False
    assert _runner(draft_kv_only=False, is_draft_worker=True) is False


def test_kv_mixin_predicate_survives_a_stub_without_the_property():
    """#624 stub-drift: the sizing path must not demand the property."""
    from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
        _is_draft_kv_only_producer,
    )

    bare = types.SimpleNamespace()
    assert _is_draft_kv_only_producer(bare) is False

    stub = types.SimpleNamespace(
        server_args=types.SimpleNamespace(speculative_draft_kv_only=True),
        is_draft_worker=False,
    )
    assert _is_draft_kv_only_producer(stub) is True

    real = types.SimpleNamespace(is_draft_kv_only_producer=True)
    assert _is_draft_kv_only_producer(real) is True


def test_init_cuda_graphs_carves_the_decode_capture_for_the_producer():
    """The carve is in the method, before any capture, and keeps prefill.

    Source-level because constructing a ModelRunner needs a GPU and weights;
    what has to be true is structural: the producer branch sets
    ``capture_decode_cuda_graph = False`` and does NOT touch
    ``init_prefill_cuda_graph``.
    """
    import inspect

    from sglang.srt.model_executor.model_runner import ModelRunner

    src = inspect.getsource(ModelRunner.init_cuda_graphs)
    assert "self.is_draft_kv_only_producer" in src
    assert "capture_decode_cuda_graph = False" in src
    # The carve must precede the capture call it disarms.
    assert src.index("capture_decode_cuda_graph = False") < src.index(
        "self.init_decode_cuda_graph()"
    )
    # ... and must not have taken the prefill graph with it.
    assert "self.init_prefill_cuda_graph()" in src
