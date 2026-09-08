# SPDX-License-Identifier: Apache-2.0
"""#1259 (b) -- the MTP head's own ``lm_head`` is never BUILT on the producer.

The finding this file pins, measured rather than argued. Boot ``weg2tr1``
(tip ``19f9f98faa``) printed, on ONE line of ``P.log:278``::

    WEG2 DRAFT-KV-PRODUCER armed stage=2/3 ... mtp_mib=405.3 embed_mib=1213.4
    resident_mib=1682.9 head_released_mib=2425.0 nvml_delta_mib=3998.0
    embed_dtype=torch.int8

``head_released_mib=2425.0`` says the head's own ``[248320, 5120]`` bf16
output table WAS deleted (``_drop_parameters``, the upstream
``del self.lm_head.weight`` form). ``nvml_delta_mib=3998.0`` on the same line
says the driver never got those bytes back: under ``--enable-memory-saver``
the build runs inside ``torch_memory_saver``'s single primary
``torch.cuda.MemPool``, and ``torch.cuda.empty_cache()`` does not hand a block
of a non-default pool to the driver. The KV budget is profiled from
``mem_get_info`` (``model_runner_kv_cache_mixin.py:847``), so the deleted table
was charged against PP2's ``max_total_num_tokens`` for the whole boot anyway.

**A table that is never built needs no release.** Nothing ever landed in it:
``Qwen3_5ForCausalLMMTP.load_weights`` skips every checkpoint name without
``mtp`` in it, and ``lm_head.weight`` has none -- the head has always run on
the co-located target's module (``set_lm_head_from_target``), which is
resident on this exact stage because the producer only ever runs on the
target's LAST one.

Pure construction + pure predicate: no GPU, no ForwardBatch, no model load.
``ParallelLMHead`` is monkeypatched to a COUNTER, so "the deferral did not
allocate" is asserted as *the constructor was not called*, which is the claim
-- not as a byte count after the fact.
"""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from sglang.srt.models import qwen3_5_mtp
from sglang.srt.models.qwen3_5_mtp import (
    Qwen3_5ForCausalLMMTP,
    Qwen3_5MtpLmHeadDeferred,
    Qwen3_5MtpLmHeadNotShared,
    build_mtp_lm_head,
    lm_head_from_target,
)
from sglang.srt.speculative.draft_kv_producer import (
    Weg2DraftHeadUnshareable,
    _drop_parameters,
    _refuse_unshareable_head,
)


VOCAB = 248320
HIDDEN = 5120


def _cfg(tie: bool = False):
    return types.SimpleNamespace(
        vocab_size=VOCAB, hidden_size=HIDDEN, tie_word_embeddings=tie
    )


@pytest.fixture
def counting_head(monkeypatch):
    """Replace ``ParallelLMHead`` in the module namespace with a counter.

    The real one allocates ``vocab x hidden`` on the current device; the point
    of the fix is that it is not REACHED, so a counter is the honest
    instrument and it also keeps this file hermetic.
    """
    calls = []

    class _FakeHead(nn.Module):
        def __init__(self, vocab_size, hidden_size, quant_config=None, prefix=""):
            super().__init__()
            calls.append((vocab_size, hidden_size, prefix))
            self.weight = nn.Parameter(torch.zeros(2, 2))

    monkeypatch.setattr(qwen3_5_mtp, "ParallelLMHead", _FakeHead)
    return calls


# ------------------------------------------------------- the context flag ----


def test_flag_is_off_by_default():
    """Every path that is not the draft-KV producer must be untouched."""
    assert qwen3_5_mtp._LM_HEAD_FROM_TARGET.get() is False


def test_flag_is_on_inside_and_restored_after():
    with lm_head_from_target():
        assert qwen3_5_mtp._LM_HEAD_FROM_TARGET.get() is True
    assert qwen3_5_mtp._LM_HEAD_FROM_TARGET.get() is False


def test_flag_is_restored_when_the_build_raises():
    """A draft build that dies must not leave the flag set for the next one."""
    with pytest.raises(ValueError):
        with lm_head_from_target():
            raise ValueError("draft build died")
    assert qwen3_5_mtp._LM_HEAD_FROM_TARGET.get() is False


def test_flag_nests_and_unwinds_to_the_outer_value():
    with lm_head_from_target():
        with lm_head_from_target():
            assert qwen3_5_mtp._LM_HEAD_FROM_TARGET.get() is True
        assert qwen3_5_mtp._LM_HEAD_FROM_TARGET.get() is True
    assert qwen3_5_mtp._LM_HEAD_FROM_TARGET.get() is False


# ------------------------------------------------------ the build decision ----


def test_deferred_build_never_calls_parallel_lm_head(counting_head):
    """THE LOAD-BEARING ASSERTION: not 'it was freed', but 'it was never
    allocated'. 2425.0 MiB on PP2 at this checkpoint's 248320 vocab."""
    with lm_head_from_target():
        head = build_mtp_lm_head(_cfg(), None, "")
    assert isinstance(head, Qwen3_5MtpLmHeadDeferred)
    assert counting_head == [], (
        "ParallelLMHead was constructed inside lm_head_from_target(): the "
        "vocab table was allocated and the KV budget is charged for it "
        "regardless of any later release."
    )


def test_normal_build_still_calls_parallel_lm_head(counting_head):
    """Group D's NEXTN drafter and every non-producer path are unchanged."""
    head = build_mtp_lm_head(_cfg(), None, "somewhere")
    assert not isinstance(head, Qwen3_5MtpLmHeadDeferred)
    assert counting_head == [(VOCAB, HIDDEN, "somewhere.lm_head")]


def test_tie_word_embeddings_is_never_deferred(counting_head):
    """Under a tie the head IS the resident embedding -- there is no second
    table to skip, and handing back a placeholder would replace a module the
    caller must keep."""
    with lm_head_from_target():
        head = build_mtp_lm_head(_cfg(tie=True), None, "")
    assert not isinstance(head, Qwen3_5MtpLmHeadDeferred)
    assert counting_head == [(VOCAB, HIDDEN, "lm_head")]


# ------------------------------------------------------- the placeholder ----


def test_placeholder_holds_no_parameters_and_no_buffers():
    head = Qwen3_5MtpLmHeadDeferred()
    assert list(head.parameters()) == []
    assert list(head.buffers()) == []


def test_placeholder_release_is_a_no_op():
    """``_drop_parameters`` is the release the producer used to pay for; with
    nothing built it must report 0.0 rather than a phantom saving."""
    assert _drop_parameters(Qwen3_5MtpLmHeadDeferred()) == 0.0


def test_placeholder_weight_refuses_by_name():
    head = Qwen3_5MtpLmHeadDeferred()
    with pytest.raises(Qwen3_5MtpLmHeadNotShared):
        _ = head.weight


def test_placeholder_forward_refuses_by_name():
    """The draft forward goes through ``LogitsProcessor`` before C21 returns,
    so an unshared head must stop there and say why."""
    head = Qwen3_5MtpLmHeadDeferred()
    with pytest.raises(Qwen3_5MtpLmHeadNotShared):
        head(torch.zeros(1, 4))


# ------------------------------------------------ the share, by data_ptr ----


class _StubMtp:
    """A stand-in carrying exactly the state the methods under test read --
    ``config.tie_word_embeddings`` and ``lm_head`` -- with the REAL methods and
    the REAL property descriptor bound onto it. Building a true
    ``Qwen3_5ForCausalLMMTP`` would need a process group and a checkpoint;
    borrowing the functions keeps the assertions about the shipped code.
    """

    lm_head_is_deferred = Qwen3_5ForCausalLMMTP.lm_head_is_deferred
    get_embed_and_head = Qwen3_5ForCausalLMMTP.get_embed_and_head
    set_lm_head_from_target = Qwen3_5ForCausalLMMTP.set_lm_head_from_target

    def __init__(self, lm_head, tie: bool = False):
        self.config = _cfg(tie=tie)
        self.lm_head = lm_head


def _stub_mtp(lm_head, tie: bool = False):
    return _StubMtp(lm_head, tie=tie)


def test_share_aliases_the_targets_table_not_a_copy():
    """WEIGHT ALIASING, asserted where it is decidable: same storage, so the
    share moved no bytes and the head runs on the table the target loaded."""
    target_head = nn.Linear(4, VOCAB, bias=False)
    stub = _stub_mtp(Qwen3_5MtpLmHeadDeferred())

    stub.set_lm_head_from_target(target_head)

    assert stub.lm_head is target_head
    assert stub.lm_head.weight.data_ptr() == target_head.weight.data_ptr()


def test_share_replaces_the_placeholder_entirely():
    target_head = nn.Linear(4, 8, bias=False)
    stub = _stub_mtp(Qwen3_5MtpLmHeadDeferred())
    stub.set_lm_head_from_target(target_head)
    assert not isinstance(stub.lm_head, Qwen3_5MtpLmHeadDeferred)


def test_share_is_skipped_under_tie_word_embeddings():
    """Unchanged upstream behaviour: the tie keeps the head pointing at the
    resident embedding, and the target's module must not overwrite it."""
    own = nn.Linear(4, 8, bias=False)
    stub = _stub_mtp(own, tie=True)
    stub.set_lm_head_from_target(nn.Linear(4, 8, bias=False))
    assert stub.lm_head is own


# ------------------------------------------------------- the predicate ------


def test_deferred_predicate_true_before_share_false_after():
    stub = _stub_mtp(Qwen3_5MtpLmHeadDeferred())
    assert stub.lm_head_is_deferred is True
    stub.set_lm_head_from_target(nn.Linear(4, 8, bias=False))
    assert stub.lm_head_is_deferred is False


def test_get_embed_and_head_refuses_while_deferred():
    stub = _stub_mtp(Qwen3_5MtpLmHeadDeferred())
    with pytest.raises(Qwen3_5MtpLmHeadNotShared):
        stub.get_embed_and_head()


# ------------------------------------------- the PRODUCER enters the scope ----
#
# The model-side tests above all pass with the producer's `with` clause
# deleted -- measured, mutant M2 of this file's own census SURVIVED the first
# round at 16/16 green. A helper nobody enters saves nothing, so the entry
# itself is asserted here, at the seam, by driving the real
# ``DraftKvProducer.__init__`` with the draft worker stubbed out.


class _FakePpGroup:
    is_last_rank = True
    rank_in_group = 2
    world_size = 3


class _FakeServerArgs:
    def __init__(self):
        self.speculative_draft_kv_only = True

    def override(self, _who, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeScheduler:
    def __init__(self):
        self.server_args = _FakeServerArgs()
        self.nccl_port = 0
        self.tp_worker = object()
        self.ps = types.SimpleNamespace(
            gpu_id=0, tp_rank=0, dp_rank=0, moe_ep_rank=0, attn_cp_rank=0, moe_dp_rank=0
        )


@pytest.fixture
def producer_build(monkeypatch):
    """Build a ``DraftKvProducer`` with the draft worker replaced by a probe
    that records ``_LM_HEAD_FROM_TARGET`` AT CONSTRUCTION TIME -- which is the
    only moment the flag can affect what gets allocated."""
    import contextlib

    from sglang.srt.distributed import parallel_state
    from sglang.srt.speculative import eagle_worker_v2

    seen = {}

    class _ProbeWorker:
        def __init__(self, **kw):
            seen["flag_at_build"] = qwen3_5_mtp._LM_HEAD_FROM_TARGET.get()
            self.draft_runner = object()

    monkeypatch.setattr(parallel_state, "get_pp_group", lambda: _FakePpGroup())
    monkeypatch.setattr(
        parallel_state, "draft_pp_scope", lambda *a, **k: contextlib.nullcontext()
    )
    monkeypatch.setattr(eagle_worker_v2, "EagleDraftWorker", _ProbeWorker)
    return seen


def test_producer_builds_the_draft_inside_lm_head_from_target(producer_build):
    """THE SEAM. If this reads False the head allocates its own 2425.0 MiB
    vocab table again and nothing else in this file notices."""
    from sglang.srt.speculative.draft_kv_producer import DraftKvProducer

    DraftKvProducer(_FakeScheduler(), None)

    assert producer_build["flag_at_build"] is True, (
        "the draft worker was constructed OUTSIDE lm_head_from_target(): "
        "Qwen3_5ForCausalLMMTP builds its own ParallelLMHead again."
    )


def test_producer_leaves_the_flag_off_afterwards(producer_build):
    """The scope must not leak into whatever this process builds next."""
    from sglang.srt.speculative.draft_kv_producer import DraftKvProducer

    DraftKvProducer(_FakeScheduler(), None)
    assert qwen3_5_mtp._LM_HEAD_FROM_TARGET.get() is False


# -------------------------------------------------- the boot-time refusal ----


def test_deferred_head_with_no_target_head_refuses_at_boot():
    """Without a built table there is no fallback: a producer on a stage that
    carries no target lm_head must stop here, not inside a per-chunk forward."""
    with pytest.raises(Weg2DraftHeadUnshareable):
        _refuse_unshareable_head(True, None)


def test_deferred_head_with_a_target_head_is_accepted():
    assert _refuse_unshareable_head(True, nn.Linear(4, 8, bias=False)) is None


def test_undeferred_head_with_no_target_head_is_not_this_guards_business():
    """Pre-#1259 shape: the head built its own table, so a missing target head
    is not fatal here and this guard must not invent a refusal for it."""
    assert _refuse_unshareable_head(False, None) is None
