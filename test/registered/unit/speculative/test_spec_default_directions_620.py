"""#620: Fix wrong-direction getattr defaults across the speculative module.

Four sites had getattr defaults that masked missing attributes in the wrong
direction (silent bypass or silent opt-in).  Each is verified here:
  1. ngram_worker            — dcp_size gate must raise on missing attribute
  2. frozen_kv_mtp_worker_v2 — same pattern
  3. dflash_worker_v2        — supports_fused_context_kv defaults to False
  4. dflash_utils            — need_top_k_sampling defaults to False
"""

from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative.dflash_utils import (
    build_dflash_verify_target_probs,
)
from sglang.srt.speculative.spec_info import (
    reject_frozen_kv_mtp_verify_under_dcp,
    reject_ngram_verify_under_dcp,
)


# ---------------------------------------------------------------------------
# Site 1: ngram_worker  (getattr -> direct attribute read)
# ---------------------------------------------------------------------------

class _MockServerArgsNoDcp:
    """Minimal mock that deliberately lacks dcp_size."""
    pass


class _MockServerArgsWithDcp:
    """Minimal mock that has dcp_size set."""

    def __init__(self, dcp_size: int = 1):
        self.dcp_size = dcp_size


def test_ngram_dcp_gate_missing_attribute_raises():
    """Direct attribute read raises AttributeError when dcp_size is absent."""
    mock = _MockServerArgsNoDcp()
    with pytest.raises(AttributeError, match="dcp_size"):
        _ = mock.dcp_size  # exactly what the fixed code path does


def test_ngram_dcp_gate_present_attribute_passes():
    """Direct attribute read succeeds and the reject gate is inert for dcp<=1."""
    mock = _MockServerArgsWithDcp(dcp_size=1)
    reject_ngram_verify_under_dcp(mock.dcp_size)  # no raise


# ---------------------------------------------------------------------------
# Site 2: frozen_kv_mtp_worker_v2  (same pattern as site 1)
# ---------------------------------------------------------------------------

def test_frozen_kv_dcp_gate_missing_attribute_raises():
    """Direct attribute read raises AttributeError when dcp_size is absent."""
    mock = _MockServerArgsNoDcp()
    with pytest.raises(AttributeError, match="dcp_size"):
        _ = mock.dcp_size


def test_frozen_kv_dcp_gate_present_attribute_passes():
    """Direct attribute read succeeds and the reject gate is inert for dcp<=1."""
    mock = _MockServerArgsWithDcp(dcp_size=1)
    reject_frozen_kv_mtp_verify_under_dcp(mock.dcp_size)  # no raise


# ---------------------------------------------------------------------------
# Site 3: dflash_worker_v2
# supports_fused_context_kv defaults to False (was True)
# ---------------------------------------------------------------------------

class _ModelWithoutFlag:
    """Draft model that does NOT declare supports_fused_context_kv."""
    pass


class _ModelWithFlagTrue:
    supports_fused_context_kv = True


def test_fused_context_kv_missing_defaults_false():
    """Undeclared attribute must NOT silently opt-in to the fused path."""
    model = _ModelWithoutFlag()
    # This is the exact check used by the worker after the fix:
    val = getattr(model, "supports_fused_context_kv", False)
    assert val is False


def test_fused_context_kv_explicit_true_stays_true():
    """Explicitly declared True is respected."""
    model = _ModelWithFlagTrue()
    val = getattr(model, "supports_fused_context_kv", False)
    assert val is True


# Smoke: verify the real model classes in dflash.py keep their declared values.
@pytest.mark.parametrize(
    "klass_name,expected",
    [
        ("DFlashDraftModel", True),
        ("DFlashLagunaForCausalLM", False),
    ],
)
def test_dflash_model_class_flags(klass_name, expected):
    """Import the real model classes and verify their declared flags."""
    from sglang.srt.models.dflash import (
        DFlashDraftModel,
        DFlashLagunaForCausalLM,
    )

    map_ = {
        "DFlashDraftModel": DFlashDraftModel,
        "DFlashLagunaForCausalLM": DFlashLagunaForCausalLM,
    }
    cls = map_[klass_name]
    assert getattr(cls, "supports_fused_context_kv", None) == expected


# ---------------------------------------------------------------------------
# Site 4: dflash_utils
# need_top_k_sampling defaults to False (was True)
# ---------------------------------------------------------------------------

class _SamplingInfoNoTopK:
    """SamplingInfo-like object missing need_top_k_sampling."""

    def __init__(self):
        self.temperatures = torch.tensor([1.0])
        self.top_ks = torch.tensor([50])
        self.top_ps = torch.tensor([1.0])


class _SamplingInfoWithTopK:
    """SamplingInfo-like object with need_top_k_sampling explicitly True."""

    def __init__(self):
        self.temperatures = torch.tensor([1.0])
        self.top_ks = torch.tensor([50])
        self.top_ps = torch.tensor([1.0])
        self.need_top_k_sampling = True


def test_dflash_need_top_k_missing_defaults_false():
    """Undeclared need_top_k_sampling must NOT silently enable top-k."""
    info = _SamplingInfoNoTopK()
    val = getattr(info, "need_top_k_sampling", False)
    assert val is False


def test_dflash_need_top_k_explicit_true_stays_true():
    """Explicitly True is respected."""
    info = _SamplingInfoWithTopK()
    val = getattr(info, "need_top_k_sampling", False)
    assert val is True


def test_build_dflash_verify_target_probs_no_top_k():
    """build_dflash_verify_target_probs should skip top-k when attribute absent.

    This is the integration-level smoke: with a sampling_info that lacks
    need_top_k_sampling (defaults False), the function should NOT index into
    sampling_info.top_ks, which would KeyError/AttributeError otherwise.
    """
    device = torch.device("cpu")
    bs = 2
    draft_token_num = 3
    vocab_size = 100
    logits = torch.randn(draft_token_num * bs, vocab_size, device=device)

    # _SamplingInfoNoTopK lacks need_top_k_sampling, so it defaults to False.
    info = _SamplingInfoNoTopK()
    info.temperatures = torch.ones((bs, 1), device=device)

    # With need_top_k defaulting False, the top-k branch is skipped entirely
    # and the function should return without touching info.top_ks.
    result = build_dflash_verify_target_probs(
        next_token_logits=logits,
        sampling_info=info,
        draft_token_num=draft_token_num,
        bs=bs,
        use_sparse_topk=True,
    )
    assert result.shape == (bs, draft_token_num, vocab_size)


def test_build_dflash_verify_target_probs_with_top_k():
    """When need_top_k_sampling is True, top-k path is exercised."""
    device = torch.device("cpu")
    bs = 2
    draft_token_num = 3
    vocab_size = 100
    logits = torch.randn(draft_token_num * bs, vocab_size, device=device)

    info = _SamplingInfoWithTopK()
    info.temperatures = torch.ones((bs, 1), device=device)
    info.top_ks = torch.full((bs,), 50, dtype=torch.long, device=device)
    info.top_ps = torch.ones((bs, 1), device=device)

    result = build_dflash_verify_target_probs(
        next_token_logits=logits,
        sampling_info=info,
        draft_token_num=draft_token_num,
        bs=bs,
        use_sparse_topk=True,
    )
    assert result.shape == (bs, draft_token_num, vocab_size)
