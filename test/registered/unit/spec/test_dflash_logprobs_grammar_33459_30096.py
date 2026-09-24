"""upstream #33459 (logprobs with DFlash) and #30096 (grammar with DFlash), on
the fork's DFLASH worker (group D of the 27B serving).

Before: group D aborted every request with ``return_logprob`` or a grammar
(json_schema / regex / ebnf / structural_tag -- e.g. an Anthropic
``tool_choice: any|tool`` becomes an OpenAI ``required``/named tool choice and
a structure constraint) with "DFLASH speculative decoding does not support
...". Hermetic, CPU. Pins:

* ``validate_dflash_request`` admits both for DFLASH and still refuses both for
  DSpark (no path there);
* ``_dflash_verify_logprobs`` puts log_softmax(row j)[out_tokens[:, j]] in the
  spec-v2 layout the result processor reads;
* ``_dflash_grammar_vocab_mask`` walks the LINEAR verify block with the fork's
  ``generate_token_bitmask``: constrained rows up to the first draft the
  grammar refuses, all-allowed rows past it and for grammar-less requests;
* the verify path is wired: mask built after the target forward, applied to
  the logits after the adjustments and before accept; logprobs computed when
  asked; the ValueError that refused logprobs is gone.
"""

import inspect
import math
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.speculative import dflash_worker_v2 as dfw
from sglang.srt.speculative.dflash_info import DFlashVerifyInput
from sglang.srt.speculative.dflash_utils import validate_dflash_request
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


def _req(*, logprob=False, hidden=False, json_schema=None):
    return SimpleNamespace(
        return_logprob=logprob,
        return_hidden_states=hidden,
        sampling_params=SimpleNamespace(
            json_schema=json_schema, regex=None, ebnf=None, structural_tag=None
        ),
    )


# ------------------------------------------------------------ admission


def test_dflash_admits_logprobs_and_grammar():
    dflash = SpeculativeAlgorithm.from_string("DFLASH")
    assert validate_dflash_request(_req(logprob=True), True) is None
    assert validate_dflash_request(_req(logprob=True), True, dflash) is None
    assert validate_dflash_request(_req(json_schema="{}"), True, dflash) is None


def test_dspark_still_refuses_both():
    dspark = SpeculativeAlgorithm.from_string("DSPARK")
    assert "return_logprob" in validate_dflash_request(_req(logprob=True), False, dspark)
    assert "grammar" in validate_dflash_request(_req(json_schema="{}"), False, dspark)


def test_hidden_states_under_overlap_still_refused():
    assert "return_hidden_states" in validate_dflash_request(_req(hidden=True), True)


# ------------------------------------------------------------ logprobs


def test_verify_logprobs_follow_the_committed_rows():
    torch.manual_seed(0)
    bs, block, vocab = 2, 4, 50
    logits = torch.randn(bs * block, vocab)
    out_tokens = torch.randint(0, vocab, (bs, block))
    batch = SimpleNamespace(
        seq_lens=torch.zeros(bs),
        sampling_info=SimpleNamespace(is_all_greedy=True),
        top_logprobs_nums=None,
        token_ids_logprobs=None,
    )
    lo = SimpleNamespace(next_token_logits=logits)
    dfw.DFlashWorkerV2._dflash_verify_logprobs(
        batch=batch, logits_output=lo, out_tokens=out_tokens, bs=bs, block_size=block
    )
    ref = torch.log_softmax(logits, dim=-1)
    assert lo.next_token_logprobs.shape == (bs, block)
    for b in range(bs):
        for j in range(block):
            assert math.isclose(
                float(lo.next_token_logprobs[b, j]),
                float(ref[b * block + j, out_tokens[b, j]]),
                rel_tol=1e-5,
                abs_tol=1e-6,
            )


# ------------------------------------------------------------ grammar


class _SeqGrammar:
    """Accepts exactly SEQ, one token per step (a toy FSM)."""

    SEQ = [5, 7, 9, 11]

    def __init__(self):
        self.state = 0

    @staticmethod
    def allocate_vocab_mask(vocab_size, batch_size, device):
        return torch.full(
            (batch_size, (vocab_size + 31) // 32), -1, dtype=torch.int32, device=device
        )

    def fill_vocab_mask(self, bitmask, idx):
        bitmask[idx].zero_()
        t = self.SEQ[self.state]
        bitmask[idx, t // 32] = 1 << (t % 32)

    def accept_token(self, t):
        assert t == self.SEQ[self.state]
        self.state += 1

    def rollback(self, n):
        self.state -= n

    def is_terminated(self):
        return self.state >= len(self.SEQ)

    @staticmethod
    def apply_vocab_mask(logits, vocab_mask):
        v = logits.shape[-1]
        bits = torch.arange(v)
        allowed = (vocab_mask[:, bits // 32] >> (bits % 32)) & 1
        logits.masked_fill_(allowed == 0, float("-inf"))


def _verify_input(bs, block):
    return DFlashVerifyInput(
        draft_token=torch.zeros(bs * block, dtype=torch.int64),
        positions=torch.zeros(bs * block, dtype=torch.int64),
        draft_token_num=block,
    )


def test_grammar_mask_walks_the_linear_block_and_stops_at_the_first_refusal():
    vocab, block = 64, 5
    g = _SeqGrammar()
    reqs = [SimpleNamespace(grammar=g), SimpleNamespace(grammar=None)]
    # req0: anchor 3, drafts 5, 7 (valid), 8 (refused: 9 expected), 11
    draft = torch.tensor([[3, 5, 7, 8, 11], [1, 2, 3, 4, 6]])
    sinfo = SimpleNamespace(vocab_size=vocab, vocab_mask="stale-extend-mask")
    batch = SimpleNamespace(reqs=reqs, sampling_info=sinfo)
    vi = _verify_input(2, block)
    mask = dfw.DFlashWorkerV2._dflash_grammar_vocab_mask(
        batch=batch, verify_input=vi, draft_tokens_cpu=draft, device="cpu"
    )
    assert vi.grammar is g and sinfo.vocab_mask is None
    assert g.state == 0  # every accept was rolled back
    logits = torch.zeros(2 * block, vocab)
    g.apply_vocab_mask(logits, mask)
    allowed = [(row != float("-inf")).nonzero().flatten().tolist() for row in logits]
    assert allowed[0] == [5] and allowed[1] == [7] and allowed[2] == [9]
    assert len(allowed[3]) == vocab and len(allowed[4]) == vocab  # past the refusal
    for r in range(block, 2 * block):  # grammar-less request: untouched
        assert len(allowed[r]) == vocab


def test_no_grammar_request_no_mask():
    batch = SimpleNamespace(
        reqs=[SimpleNamespace(grammar=None)],
        sampling_info=SimpleNamespace(vocab_size=64, vocab_mask=None),
    )
    assert (
        dfw.DFlashWorkerV2._dflash_grammar_vocab_mask(
            batch=batch,
            verify_input=_verify_input(1, 3),
            draft_tokens_cpu=torch.tensor([[1, 2, 3]]),
            device="cpu",
        )
        is None
    )


# ------------------------------------------------------------ wiring


def test_verify_path_is_wired():
    src = inspect.getsource(dfw.DFlashWorkerV2.forward_batch_generation)
    assert "does not support return_logprob" not in src
    # the verify body lives in the same class; order inside the module source
    mod = inspect.getsource(dfw.DFlashWorkerV2)
    i_cpu = mod.index("grammar_draft_tokens_cpu = (")
    i_fwd = mod.index("target_out = self.target_worker.forward_batch_generation(", i_cpu)
    i_mask = mod.index("self._dflash_grammar_vocab_mask(", i_fwd)
    i_adj = mod.index("apply_dflash_verify_logits_adjustments(", i_mask)
    i_apply = mod.index("verify_input.grammar.apply_vocab_mask(", i_adj)
    i_accept = mod.index("candidates = draft_tokens", i_apply)
    i_lp = mod.index("self._dflash_verify_logprobs(", i_accept)
    i_mamba = mod.index("if self._need_mamba_verify_commit:", i_lp)
    assert i_cpu < i_fwd < i_mask < i_adj < i_apply < i_accept < i_lp < i_mamba
