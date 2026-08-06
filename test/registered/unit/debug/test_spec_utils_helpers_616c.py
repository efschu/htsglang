"""Unit tests for pure helper functions in spec_utils.

Hermetic -- no CUDA required.  Uses CPU torch tensors exclusively.
Covers:
  _sample_simulated_acc_len
  fast_sample
  renorm_draft_probs
  sample_draft_proposal
  _select_top_k_tokens_first
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.speculative.spec_utils import (
    _sample_simulated_acc_len,
    _select_top_k_tokens_first,
    fast_sample,
    renorm_draft_probs,
    sample_draft_proposal,
)


# ---------------------------------------------------------------------------
# _sample_simulated_acc_len  (docstring: "Sample a simulated acceptance length in [1, max_len]")
# ---------------------------------------------------------------------------


class TestSampleSimulatedAccLenMultinomial:
    """multinomial method: draws from N(mean=simulate_acc_len, std=1),
    clamps to [1, max_len], rounds to int."""

    def test_returns_int(self):
        """Return type is int, never float."""
        result = _sample_simulated_acc_len(3.5, "multinomial", 5)
        assert isinstance(result, int)

    def test_clamped_to_max_len_one(self):
        """When max_len is 1, the result must be 1 regardless of mean."""
        # mean=10000, but max_len=1, so clamped to 1.0, rounded to 1.
        result = _sample_simulated_acc_len(10000.0, "multinomial", 1)
        assert result == 1

    def test_clamped_to_min_one(self):
        """When simulate_acc_len is negative, result is clamped to at least 1."""
        result = _sample_simulated_acc_len(-50.0, "multinomial", 10)
        assert result >= 1
        assert result <= 10

    def test_in_range(self):
        """Result always in [1, max_len]."""
        for _ in range(50):
            result = _sample_simulated_acc_len(5.0, "multinomial", 10)
            assert 1 <= result <= 10

    def test_mean_equals_max_len_stays_at_max(self):
        """When mean=max_len=3, draws from N(3,1) clamped to [1,3].
        Most rounds-to-3 but some round to 2 -- just verify in bounds."""
        for _ in range(50):
            result = _sample_simulated_acc_len(3.0, "multinomial", 3)
            assert 1 <= result <= 3


class TestSampleSimulatedAccLenMatchExpected:
    """match-expected method: deterministically rounds down/up with
    fractional-weighted probabilities."""

    def test_exact_integer_returns_same(self):
        """simulate_acc_len=4.0, max_len=10 -> 4 (lower=upper=4)."""
        result = _sample_simulated_acc_len(4.0, "match-expected", 10)
        assert result == 4

    def test_clamped_to_max_len(self):
        """simulate_acc_len=100.0, max_len=5 -> clamped to 5.0, lower=upper=5."""
        result = _sample_simulated_acc_len(100.0, "match-expected", 5)
        assert result == 5

    def test_clamped_to_min_one(self):
        """simulate_acc_len=0.3, max_len=10 -> clamped to 1.0, lower=upper=1."""
        result = _sample_simulated_acc_len(0.3, "match-expected", 10)
        assert result == 1

    def test_returns_int(self):
        """Return type is always int."""
        result = _sample_simulated_acc_len(3.7, "match-expected", 10)
        assert isinstance(result, int)
        # lower=3, upper=4, so result must be 3 or 4
        assert result in (3, 4)

    def test_fractional_always_rounds_down_or_up(self):
        """For 3.7: lower=3, upper=4. Result must be 3 or 4."""
        seen = set()
        for _ in range(100):
            result = _sample_simulated_acc_len(3.7, "match-expected", 10)
            seen.add(result)
        assert seen.issubset({3, 4})

    def test_max_len_boundary_lower_equals_max(self):
        """When simulate_acc_len=5.5, max_len=5: clamped to 5.0, lower=5, upper=5."""
        result = _sample_simulated_acc_len(5.5, "match-expected", 5)
        assert result == 5


class TestSampleSimulatedAccLenInvalidMethod:
    def test_raises_value_error(self):
        """Any method string other than the two known ones raises ValueError."""
        with pytest.raises(ValueError, match="Invalid simulate_acc_method"):
            _sample_simulated_acc_len(3.0, "banana", 10)


# ---------------------------------------------------------------------------
# fast_sample  (probs, num_samples) -> (sample_p, sample_index)
# ---------------------------------------------------------------------------


class TestFastSample:
    def test_returns_tensor_pair(self):
        """Returns two tensors, not a tuple of scalars."""
        probs = torch.tensor([[0.25, 0.25, 0.25, 0.25]], device="cpu")
        sp, si = fast_sample(probs, num_samples=1)
        assert isinstance(sp, torch.Tensor)
        assert isinstance(si, torch.Tensor)

    def test_single_sample_shape(self):
        """num_samples=1: sample_p is (batch, 1), sample_index is (batch, 1)."""
        probs = torch.tensor([[0.5, 0.3, 0.2]], device="cpu")
        sp, si = fast_sample(probs, num_samples=1)
        assert sp.shape == (1, 1)
        assert si.shape == (1, 1)

    def test_gather_matches_index(self):
        """sample_p equals probs.gather(1, sample_index) -- that is exactly
        what fast_sample does internally, so verify the contract."""
        probs = torch.tensor([[0.1, 0.7, 0.2]], device="cpu")
        sp, si = fast_sample(probs, num_samples=1)
        expected_p = probs.gather(1, si)
        assert torch.allclose(sp, expected_p)

    def test_batch_dimension_preserved(self):
        """Batch size 4 with 2 samples: output shapes (4, 2)."""
        probs = torch.tensor(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.25, 0.25, 0.25, 0.25],
                [0.0, 0.0, 0.5, 0.5],
            ],
            device="cpu",
        )
        sp, si = fast_sample(probs, num_samples=2)
        assert sp.shape == (4, 2)
        assert si.shape == (4, 2)

    def test_samples_only_from_valid_range(self):
        """All sampled indices are within the vocab size (dim 1)."""
        vocab = 7
        probs = torch.ones(3, vocab, device="cpu") / vocab
        _, si = fast_sample(probs, num_samples=1)
        assert (si >= 0).all() and (si < vocab).all()


# ---------------------------------------------------------------------------
# renorm_draft_probs  (next_token_logits, sampling_info, use_rejection_sampling)
# ---------------------------------------------------------------------------


class FakeSamplingInfo:
    def __init__(self, temperatures):
        self.temperatures = temperatures


class TestRenormDraftProbs:
    def test_plain_softmax_when_rejection_off(self):
        """use_rejection_sampling=False -> plain softmax regardless of temps."""
        logits = torch.tensor([[0.0, 1.0, 2.0]], device="cpu")
        info = FakeSamplingInfo(temperatures=torch.tensor([0.5]))
        result = renorm_draft_probs(logits, info, use_rejection_sampling=False)
        expected = torch.softmax(logits, dim=-1)
        assert torch.allclose(result, expected)

    def test_temperature_scaled_softmax(self):
        """use_rejection_sampling=True -> softmax(logits / temperature)."""
        logits = torch.tensor([[0.0, 0.0, 0.0]], device="cpu")
        info = FakeSamplingInfo(temperatures=torch.tensor([[2.0]]))
        result = renorm_draft_probs(logits, info, use_rejection_sampling=True)
        # softmax([0,0,0]/2.0) = softmax([0,0,0]) = [1/3, 1/3, 1/3]
        expected = torch.tensor([[1.0 / 3, 1.0 / 3, 1.0 / 3]], device="cpu")
        assert torch.allclose(result, expected, atol=1e-6)

    def test_batch_dimension_preserved(self):
        """Batch size 3, rejection off -> shape (3, vocab)."""
        logits = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], device="cpu")
        info = FakeSamplingInfo(temperatures=torch.tensor([[1.0], [1.0], [1.0]]))
        result = renorm_draft_probs(logits, info, False)
        assert result.shape == (3, 2)

    def test_empty_batch_returns_empty(self):
        """size(0)==0 with rejection on triggers early-return to plain softmax
        because the not next_token_logits.size(0) guard fires."""
        logits = torch.tensor([], device="cpu").reshape(0, 10)
        info = FakeSamplingInfo(temperatures=torch.tensor([]).reshape(0, 1))
        result = renorm_draft_probs(logits, info, use_rejection_sampling=True)
        assert result.shape == (0, 10)


# ---------------------------------------------------------------------------
# sample_draft_proposal  (next_token_logits, temperatures) -> (probs, topk_p, topk_index)
# ---------------------------------------------------------------------------


class TestSampleDraftProposal:
    def test_returns_three_tensors(self):
        """Returns (q, q(X), X) -- three tensors."""
        logits = torch.tensor([[0.0, 1.0]], device="cpu")
        temps = torch.tensor([[1.0]], device="cpu")
        q, qp, qi = sample_draft_proposal(logits, temps)
        assert isinstance(q, torch.Tensor)
        assert isinstance(qp, torch.Tensor)
        assert isinstance(qi, torch.Tensor)

    def test_q_is_softmax_of_scaled_logits(self):
        """q == softmax(logits / T)."""
        logits = torch.tensor([[0.0, 0.0, 0.0]], device="cpu")
        temps = torch.tensor([[1.0]], device="cpu")
        q, _, _ = sample_draft_proposal(logits, temps)
        expected = torch.tensor([[1.0 / 3, 1.0 / 3, 1.0 / 3]], device="cpu")
        assert torch.allclose(q, expected, atol=1e-6)

    def test_qxp_equals_q_gathered(self):
        """q(X) == q.gather(1, X) -- the sampled probability matches the
        entry at the sampled index in the distribution."""
        logits = torch.tensor([[0.0, 1.0, 2.0]], device="cpu")
        temps = torch.tensor([[1.0]], device="cpu")
        q, qp, qi = sample_draft_proposal(logits, temps)
        expected = q.gather(1, qi)
        assert torch.allclose(qp, expected)

    def test_batch_preserved(self):
        """Batch of 2, vocab 4 -> q is (2,4), qp is (2,1), qi is (2,1).
        temperatures must be (batch,1) to broadcast with logits (batch, vocab)."""
        logits = torch.randn(2, 4, device="cpu")
        temps = torch.tensor([[1.0], [1.0]], device="cpu")
        q, qp, qi = sample_draft_proposal(logits, temps)
        assert q.shape == (2, 4)
        assert qp.shape == (2, 1)
        assert qi.shape == (2, 1)


# ---------------------------------------------------------------------------
# _select_top_k_tokens_first  (topk_p, topk_index, hidden_states, topk)
# topk_p is (b, topk) -- 2D, from spec_info.topk_p
# ---------------------------------------------------------------------------


class TestSelectTopKTokensFirst:
    def test_basic_shapes(self):
        """topk_p (b, topk), topk_index (b, topk), hidden None
        -> input_ids (b*topk), hidden_states None, topk_p returned unchanged,
        tree_info: probs (b,1,topk), index (b,topk), arange (b,topk+1)."""
        b, topk = 2, 3
        topk_p = torch.rand(b, topk, device="cpu")  # 2D: (b, topk)
        topk_index = torch.randint(0, 10, (b, topk), device="cpu")
        input_ids, hidden, probs_out, tree_info = _select_top_k_tokens_first(
            topk_p, topk_index, None, topk
        )
        assert input_ids.shape == (b * topk,)
        assert hidden is None
        assert probs_out.shape == topk_p.shape  # (b, topk) unchanged
        assert tree_info[0].shape == (b, 1, topk)
        assert tree_info[1].shape == (b, topk)
        assert tree_info[2].shape == (b, topk + 1)

    def test_input_ids_is_flattened_topk_index(self):
        """input_ids == topk_index.flatten()."""
        topk_p = torch.rand(1, 2, device="cpu")
        topk_index = torch.tensor([[5, 7]], device="cpu")
        input_ids, _, _, _ = _select_top_k_tokens_first(topk_p, topk_index, None, 2)
        assert torch.equal(input_ids, topk_index.flatten())

    def test_hidden_states_repeated_interleave(self):
        """hidden_states = hidden.repeat_interleave(topk, dim=0)."""
        b, topk, dim = 2, 2, 4
        topk_p = torch.rand(b, topk, device="cpu")
        topk_index = torch.zeros(b, topk, dtype=torch.long, device="cpu")
        hidden_in = torch.arange(b * dim, dtype=torch.float, device="cpu").reshape(
            b, dim
        )
        _, hidden_out, _, _ = _select_top_k_tokens_first(
            topk_p, topk_index, hidden_in, topk
        )
        assert hidden_out is not None
        assert hidden_out.shape == (b * topk, dim)

    def test_hidden_states_expansion_correct(self):
        """Each row of hidden_states is repeated topk times consecutively."""
        b, topk = 1, 3
        topk_p = torch.rand(b, topk, device="cpu")
        topk_index = torch.zeros(b, topk, dtype=torch.long, device="cpu")
        # hidden_in row 0 = [10, 20], repeated 3x -> [10,20,10,20,10,20]
        hidden_in = torch.tensor([[10.0, 20.0]], device="cpu")
        _, hidden_out, _, _ = _select_top_k_tokens_first(
            topk_p, topk_index, hidden_in, topk
        )
        expected = torch.tensor(
            [[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]], device="cpu"
        )
        assert torch.equal(hidden_out, expected)

    def test_tree_info_arange_starts_at_minus_one(self):
        """The arange tensor in tree_info starts at -1 and has topk entries,
        then is expanded to (b, topk+1). The first column is -1 for all
        rows (parent pointer sentinel)."""
        b, topk = 3, 2
        topk_p = torch.rand(b, topk, device="cpu")
        topk_index = torch.zeros(b, topk, dtype=torch.long, device="cpu")
        _, _, _, tree_info = _select_top_k_tokens_first(topk_p, topk_index, None, topk)
        arange_tensor = tree_info[2]
        # First column is always -1 (sentinel for root parent)
        assert (arange_tensor[:, 0] == -1).all()
        # Shape is (b, topk+1)
        assert arange_tensor.shape == (b, topk + 1)

    def test_hidden_none_passes_through(self):
        """hidden_states=None -> returns None."""
        topk_p = torch.rand(1, 1, device="cpu")
        topk_index = torch.tensor([[0]], device="cpu")
        _, hidden_out, _, _ = _select_top_k_tokens_first(topk_p, topk_index, None, 1)
        assert hidden_out is None


# ---------------------------------------------------------------------------
# Single-element edge case for renorm_draft_probs
# ---------------------------------------------------------------------------


class TestRenormDraftProbsSingleVocab:
    def test_vocab_size_one_rejection_on(self):
        """Single-token vocab with rejection on: softmax over [0]/T is still [1]."""
        logits = torch.tensor([[0.0]], device="cpu")
        info = FakeSamplingInfo(temperatures=torch.tensor([[1.5]]))
        result = renorm_draft_probs(logits, info, use_rejection_sampling=True)
        assert torch.allclose(result, torch.tensor([[1.0]], device="cpu"))
