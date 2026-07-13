"""Replay-input contract of the EAGLE draft-extend cuda-graph runner.

#50 campaign Rest A: the captured graph reads the full bs*num_tokens_per_bs
token rows, while a replay only rewrites [:num_tokens] (accepted tokens,
variable per iteration). The pre-fix reset covered seq_lens/out_cache_loc/
positions/req_pool_indices/... but NOT input_ids and hidden_states, so their
tail rows kept the previous request's values and fed every subsequent
draft-extend forward (MoE routing of stale tail tokens perturbs the real
rows' logits via grouped-GEMM batch composition -> degenerate attractor
under cuda graphs).

These tests pin the contract: after reset_replay_tail_buffers, EVERY
graph-read input buffer is bit-identical to a request-independent state,
regardless of what an earlier replay left behind.
"""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative.eagle_draft_extend_cuda_graph_runner import (
    EagleDraftExtendInputBuffers,
    reset_replay_tail_buffers,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

MAX_BS = 3
NUM_TOKENS_PER_BS = 4
MAX_NUM_TOKEN = MAX_BS * NUM_TOKENS_PER_BS
HIDDEN = 8
SEQ_LEN_FILL = 1


def _make_buffers(fill: int) -> EagleDraftExtendInputBuffers:
    """Buffers pre-filled with junk, simulating a previous replay's residue."""
    return EagleDraftExtendInputBuffers(
        input_ids=torch.full((MAX_NUM_TOKEN,), fill, dtype=torch.int64),
        req_pool_indices=torch.full((MAX_BS,), fill, dtype=torch.int64),
        out_cache_loc=torch.full((MAX_NUM_TOKEN,), fill, dtype=torch.int64),
        positions=torch.full((MAX_NUM_TOKEN,), fill, dtype=torch.int64),
        mrope_positions=torch.full((3, MAX_NUM_TOKEN), fill, dtype=torch.int64),
        hidden_states=torch.full(
            (MAX_NUM_TOKEN, HIDDEN), float(fill), dtype=torch.float32
        ),
        seq_lens=torch.full((MAX_BS,), fill, dtype=torch.int64),
        seq_lens_cpu=torch.full((MAX_BS,), fill, dtype=torch.int64),
        extend_seq_lens=torch.full((MAX_BS,), fill, dtype=torch.int32),
        num_correct_drafts=torch.full((MAX_BS,), fill, dtype=torch.int32),
        num_accept_tokens=torch.full((MAX_BS,), fill, dtype=torch.int32),
        next_token_logits_buffer=torch.zeros((MAX_NUM_TOKEN, 16)),
        global_num_tokens_gpu=None,
        global_num_tokens_for_logprob_gpu=None,
    )


class TestDraftExtendTailReset(CustomTestCase):
    def test_reset_is_residue_independent(self):
        # Two buffer sets carrying DIFFERENT previous-request residue must be
        # bit-identical after the reset — otherwise request history leaks
        # into the next replayed forward.
        a = _make_buffers(fill=7)
        b = _make_buffers(fill=42)
        for buffers in (a, b):
            reset_replay_tail_buffers(buffers, SEQ_LEN_FILL, NUM_TOKENS_PER_BS)
        for name in (
            "input_ids",
            "req_pool_indices",
            "out_cache_loc",
            "positions",
            "hidden_states",
            "seq_lens",
            "extend_seq_lens",
            "num_correct_drafts",
            "num_accept_tokens",
        ):
            ta, tb = getattr(a, name), getattr(b, name)
            self.assertTrue(
                torch.equal(ta, tb),
                f"{name} still differs after reset (history leak)",
            )

    def test_token_and_hidden_tails_are_neutral(self):
        # The two buffers the pre-fix reset MISSED: input_ids and
        # hidden_states must be neutral (zero) everywhere after the reset.
        buffers = _make_buffers(fill=9)
        reset_replay_tail_buffers(buffers, SEQ_LEN_FILL, NUM_TOKENS_PER_BS)
        self.assertTrue(torch.all(buffers.input_ids == 0))
        self.assertTrue(torch.all(buffers.hidden_states == 0))
        self.assertTrue(torch.all(buffers.seq_lens == SEQ_LEN_FILL))
        self.assertTrue(torch.all(buffers.extend_seq_lens == NUM_TOKENS_PER_BS))

    def test_poison_falsifier_fills_loud_junk(self):
        buffers = _make_buffers(fill=9)
        with envs.SGLANG_POISON_GRAPH_PAD.override(True):
            reset_replay_tail_buffers(buffers, SEQ_LEN_FILL, NUM_TOKENS_PER_BS)
        self.assertTrue(torch.all(buffers.input_ids == 100))
        self.assertTrue(torch.all(buffers.hidden_states == 1024.0))
        # Poison must still be residue-independent (deterministic junk).
        other = _make_buffers(fill=3)
        with envs.SGLANG_POISON_GRAPH_PAD.override(True):
            reset_replay_tail_buffers(other, SEQ_LEN_FILL, NUM_TOKENS_PER_BS)
        self.assertTrue(torch.equal(buffers.input_ids, other.input_ids))
        self.assertTrue(torch.equal(buffers.hidden_states, other.hidden_states))

    def test_hidden_states_none_supported(self):
        buffers = _make_buffers(fill=5)
        buffers.hidden_states = None
        reset_replay_tail_buffers(buffers, SEQ_LEN_FILL, NUM_TOKENS_PER_BS)
        self.assertTrue(torch.all(buffers.input_ids == 0))


if __name__ == "__main__":
    unittest.main()
