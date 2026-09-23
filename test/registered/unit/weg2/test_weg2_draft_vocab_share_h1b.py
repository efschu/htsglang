# SPDX-License-Identifier: Apache-2.0
"""fnFL2 H1b -- group D's NEXTN draft builds NO vocabulary table of its own.

Measured, boot fnFL2x87 (5a96de48be), D TP0 (Form A host, solo draft)::

    WEG2-TAG-POOL occupancy tag=weights_draft when=after-load ... active_gib=3.86
    WEG2-XCHG-RESIDENT tag=weights_draft mib=3984.0
    [vram-census] pp0tp0-draft after load:  ... embed_tokens 1.18, lm_head 1.18
    [vram-census] pp0tp0-draft after pools: ... embed_tokens 0.60, lm_head 0.60
    WEG2-FLIP-TAG group=D rank=0 dir=d2h tag=weights_draft bytes=3984 MiB

The draft built two BF16 [248320, 2560] tables (2 x 1212.5 MiB) that nothing
ever loads (``load_weights`` keeps only ``mtp`` names), ``init_lm_head`` then
replaced both with the TARGET's INT8 modules -- and the released blocks stayed
in the weights_draft tag pool, which every flip moves.

Hermetic: stubs carry the real functions, the vocab constructors are counters.
"""

from __future__ import annotations

import types

import pytest
import torch
from torch import nn

from sglang.srt.models import qwen3_5, qwen4_exp
from sglang.srt.models import mtp_vocab_share as mvs
from sglang.srt.models.mtp_vocab_share import (
    MtpEmbedDeferred,
    MtpEmbedNotShared,
    embed_from_target_requested,
    mtp_builds_own_embed,
)
from sglang.srt.models.qwen3_5_mtp import (
    _LM_HEAD_FROM_TARGET,
    Qwen3_5ForCausalLMMTP,
    Qwen3_5MtpLmHeadDeferred,
    vocab_from_target,
)
from sglang.srt.speculative.eagle_worker_v2 import (
    EagleDraftWorker,
    draft_vocab_is_deferred,
    draft_vocab_shared_at_build,
)

# ------------------------------------------------ WHERE the build defers ----

_D_FORM_A_HOST = dict(
    enabled=True, is_eagle3=False, has_token_map=False, draft_kv_only=False,
    solo_active=True, solo_is_host=True, form_a_dense_unsharded=True,
)


@pytest.mark.parametrize(
    "change, expected, why",
    [
        ({}, True, "the D Form A solo host of fnFL2 -- the case this fixes"),
        ({"solo_active": False, "solo_is_host": True,
          "form_a_dense_unsharded": False}, True, "split placement shares modules"),
        ({"enabled": False}, False, "SGLANG_WEG2_DRAFT_SHARE_EMBED=0 = old form"),
        ({"is_eagle3": True}, False, "EAGLE3 may keep its own embeddings"),
        ({"has_token_map": True}, False, "hot-token head is a SLICED copy"),
        ({"draft_kv_only": True}, False, "P producer: placement A rebuilds its embed"),
        ({"solo_is_host": False}, False, "solo shadow never shares"),
        ({"form_a_dense_unsharded": False}, False,
         "classic solo host gathers FULL tensors (set_embed_and_head)"),
    ],
)
def test_the_build_defers_exactly_where_a_module_share_follows(change, expected, why):
    """A deferral where no MODULE share follows is a draft with placeholder
    vocab (refused at boot at best); a missing deferral is the 2.36 GiB."""
    assert draft_vocab_shared_at_build(**{**_D_FORM_A_HOST, **change}) is expected, why


def test_vocab_from_target_sets_both_halves_and_restores_them():
    """One switch for both tables: a head deferred without the embed (or the
    reverse) leaves one 1212.5 MiB table behind. Restored even if the build
    raises, so the next build (the target's, a second draft) is untouched."""
    with pytest.raises(ValueError):
        with vocab_from_target():
            assert _LM_HEAD_FROM_TARGET.get() is True
            assert embed_from_target_requested() is True
            raise ValueError("draft build died")
    assert _LM_HEAD_FROM_TARGET.get() is False
    assert embed_from_target_requested() is False


def test_tie_word_embeddings_is_never_deferred():
    """Under tie the draft's lm_head IS its embedding at construction; a
    placeholder there would hand the head a module without rows."""
    assert mtp_builds_own_embed(True, True) is True
    assert mtp_builds_own_embed(True, False) is False
    assert mtp_builds_own_embed(False, False) is True


# ------------------------------------------- the table is NEVER allocated ----


def _stub_backbone(cls, defer: bool):
    stub = cls.__new__(cls)
    nn.Module.__init__(stub)
    stub.pp_group = types.SimpleNamespace(is_first_rank=True)
    stub._defer_embed = defer
    stub._embed_quant_config = None
    return stub


@pytest.mark.parametrize(
    "cls, module", [(qwen3_5.Qwen3_5ForCausalLM, qwen3_5),
                    (qwen4_exp.Qwen4ExpModel, qwen4_exp)],
)
def test_deferred_backbone_never_constructs_the_embedding(monkeypatch, cls, module):
    """THE LOAD-BEARING ASSERTION, for BOTH builders (Qwen4-Exp overrides the
    base one wholesale): not 'it was freed' but 'it was never allocated' --
    a block freed into the weights_draft MemPool stays charged to the tag."""
    calls = []

    class _Counter(nn.Module):
        def __init__(self, *a, **k):
            super().__init__()
            calls.append(a)

    monkeypatch.setattr(module, "VocabParallelEmbedding", _Counter)
    cfg = types.SimpleNamespace(vocab_size=248320, hidden_size=2560)

    out = cls._build_embed_tokens(_stub_backbone(cls, True), cfg, None, "mtp")
    assert isinstance(out, MtpEmbedDeferred) and calls == []

    # Gegenprobe: without the deferral the same builder still builds.
    out = cls._build_embed_tokens(_stub_backbone(cls, False), cfg, None, "mtp")
    assert isinstance(out, _Counter) and len(calls) == 1


def test_the_decision_is_taken_before_the_builder_runs():
    """Order, not presence: `_defer_embed` read by `_build_embed_tokens` must
    be set BEFORE the base __init__ calls it, and only for is_nextn builds
    (the target itself is never deferred)."""
    import inspect

    src = inspect.getsource(qwen3_5.Qwen3_5ForCausalLM.__init__)
    assert src.index("self._defer_embed =") < src.index("self._build_embed_tokens(")
    assert "bool(is_nextn) and not mtp_builds_own_embed(" in src


# ------------------------------------------------- the share completes it ----


class _StubDraft(nn.Module):
    """A draft carrying the REAL share methods of Qwen3_5ForCausalLMMTP."""

    set_embed_and_head = Qwen3_5ForCausalLMMTP.set_embed_and_head
    set_embed_and_head_modules = Qwen3_5ForCausalLMMTP.set_embed_and_head_modules
    vocab_is_deferred = Qwen3_5ForCausalLMMTP.vocab_is_deferred
    embed_is_deferred = Qwen3_5ForCausalLMMTP.embed_is_deferred
    lm_head_is_deferred = Qwen3_5ForCausalLMMTP.lm_head_is_deferred

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(tie_word_embeddings=False)
        self.model = nn.Module()
        self.model.embed_tokens = MtpEmbedDeferred()
        self.lm_head = Qwen3_5MtpLmHeadDeferred()


def _worker(draft, target):
    w = EagleDraftWorker.__new__(EagleDraftWorker)
    w.target_worker = types.SimpleNamespace(
        model_runner=types.SimpleNamespace(model=target))
    w.draft_runner = types.SimpleNamespace(model=draft)
    w.speculative_algorithm = types.SimpleNamespace(is_eagle3=lambda: False)
    w.hot_token_id = None
    return w


def _target(embed=True):
    t = nn.Module()
    t.model = nn.Module()
    t.model.embed_tokens = nn.Embedding(16, 4) if embed else None
    t.lm_head = nn.Linear(4, 16, bias=False)
    return t


def test_deferred_draft_takes_the_module_share_even_for_a_tensor_target(monkeypatch):
    """Bug regression of the naive form: a target whose tables carry `.weight`
    goes down the TENSOR path (`set_embed_and_head`), which `del`s the draft's
    own `.weight` -- a placeholder has none. The deferred draft must take the
    module share, and afterwards IS the target's modules (same storage)."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    draft, target = _StubDraft(), _target()
    assert draft_vocab_is_deferred(draft)

    EagleDraftWorker.init_lm_head(_worker(draft, target))

    assert draft.model.embed_tokens is target.model.embed_tokens
    assert draft.lm_head is target.lm_head
    assert not draft_vocab_is_deferred(draft)


def test_a_share_that_cannot_complete_refuses_at_boot(monkeypatch):
    """No target module to hand in -> refused by name at init, never a
    placeholder reaching the first draft forward inside a graph capture."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: None)
    draft = _StubDraft()
    with pytest.raises(RuntimeError, match="SGLANG_WEG2_DRAFT_SHARE_EMBED=0"):
        EagleDraftWorker.init_lm_head(_worker(draft, _target(embed=False)))


def test_placeholders_refuse_by_name():
    with pytest.raises(MtpEmbedNotShared):
        MtpEmbedDeferred()(torch.zeros(1, dtype=torch.long))
    with pytest.raises(MtpEmbedNotShared):
        _StubDraft().set_embed_and_head(torch.zeros(1), torch.zeros(1))
    assert list(MtpEmbedDeferred().parameters()) == []
    assert mvs._EMBED_FROM_TARGET.get() is False
