"""Full prefill graph: stale mamba-track rows must not survive a smaller batch (#34184).

Ported from upstream sglang #34184 ("Fix stale track rows corrupting conv
checkpoints under the prefill graph"). Under the Full prefill CUDA-graph
backend the captured graph reads ``_capture_req_slots`` request rows. The
sentinel tail ``[bs:req_slots]`` of the static seq_lens / prefix_lens / ...
buffers was reset on every replay, but ``mamba_track_mask`` /
``mamba_track_indices`` were not: ``fill_from`` runs with ``padded_bs ==
raw_bs`` on this path, so it never touches ``[bs:req_slots]`` either. A
replay with fewer requests than the previous one therefore kept a live mask
row with a live destination slot, and the captured track scatter wrote this
replay's conv/GDN window into an EARLIER request's checkpoint.

Upstream's regression signal is an e2e deterministic model test (it removed
``--disable-prefill-cuda-graph`` from it). This file drives the production
``PrefillCudaGraphRunner.load_batch`` with the real prefill slot registry
(``build_prefill_registry`` with ``enable_mamba_track=True``) on CPU tensors:
a 3-request batch, then a 1-request batch. Red before the fix (rows 1..2 keep
mask True and their slot ids), green after.

Scope note for the fork: ``_is_mamba_track_enabled`` requires
``spec_algorithm.is_none()``; the 27B's P group (draft-KV producer) and D
group (DFLASH) both carry a speculative algorithm, so the track slots do not
exist on the 27B boot today -- the port keeps the non-spec path correct.
"""

import unittest

import torch

from sglang.srt.model_executor.cuda_graph_buffer_registry import (
    build_prefill_registry,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.runner import prefill_cuda_graph_runner as pcgr
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MAX_BS = 4
MAX_TOKENS = 16
REQ_SLOTS = 4


def _runner():
    runner = object.__new__(pcgr.PrefillCudaGraphRunner)
    runner.capture_num_tokens = [8, MAX_TOKENS]
    runner.buffer_registry = build_prefill_registry(
        device=torch.device("cpu"),
        max_bs=MAX_BS,
        max_num_token=MAX_TOKENS,
        cache_loc_dtype=torch.int64,
        enable_mamba_track=True,
        share_pool=False,
    )
    runner.prefill_backend_name = None
    runner.backend = object()  # neither Breakable nor a replay is exercised
    runner._prefill_static_buffers = {
        name: torch.zeros((MAX_BS,), dtype=torch.int64)
        for name in pcgr._PREFILL_STATIC_FIELDS
    }
    runner._is_full_backend = True
    runner._capture_req_slots = REQ_SLOTS
    runner.static_draft_hidden_states = None
    runner.capture_return_pooled_hidden_states = False
    # Metadata planning and the logits buffer need an attention backend and a
    # model; neither is what this test is about.
    runner._prepare_forward_metadata_for_replay = lambda *args, **kwargs: None
    runner._next_token_logits_buffer = lambda rows: None
    runner._prefill_logits_buffer_rows = lambda fb: fb.batch_size
    return runner


def _batch(track_slots):
    bs = len(track_slots)
    tokens = 2 * bs
    ext = torch.full((bs,), 2, dtype=torch.int64)
    return ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=bs,
        input_ids=torch.arange(tokens, dtype=torch.int64),
        req_pool_indices=torch.arange(bs, dtype=torch.int64) + 1,
        seq_lens=ext.clone(),
        out_cache_loc=torch.arange(tokens, dtype=torch.int64) + 100,
        seq_lens_sum=tokens,
        orig_seq_lens=ext.clone(),
        seq_lens_cpu=ext.clone(),
        positions=torch.arange(tokens, dtype=torch.int64),
        extend_num_tokens=tokens,
        extend_seq_lens=ext.clone(),
        extend_prefix_lens=torch.zeros((bs,), dtype=torch.int64),
        extend_start_loc=torch.arange(bs, dtype=torch.int64) * 2,
        extend_prefix_lens_cpu=[0] * bs,
        extend_seq_lens_cpu=[2] * bs,
        mamba_track_indices=torch.tensor(track_slots, dtype=torch.int64),
        mamba_track_mask=torch.ones((bs,), dtype=torch.bool),
        mamba_track_seqlens=ext.to(torch.int32),
        global_forward_mode=ForwardMode.EXTEND,
    )


class TestPrefillGraphStaleTrackRows(CustomTestCase):
    def test_smaller_batch_clears_the_previous_track_rows(self):
        runner = _runner()
        runner.load_batch(_batch([11, 12, 13]))
        mask = runner.buffer_registry.get_slot("mamba_track_mask").buffer
        idx = runner.buffer_registry.get_slot("mamba_track_indices").buffer
        # control: the first batch really armed three rows
        self.assertEqual(mask[:3].tolist(), [True, True, True])
        self.assertEqual(idx[:3].tolist(), [11, 12, 13])

        runner.load_batch(_batch([21]))
        self.assertEqual(mask[:1].tolist(), [True])
        self.assertEqual(idx[:1].tolist(), [21])
        # the captured scatter reads [bs:req_slots]; nothing may stay armed
        self.assertEqual(mask[1:REQ_SLOTS].tolist(), [False] * (REQ_SLOTS - 1))
        self.assertEqual(idx[1:REQ_SLOTS].tolist(), [0] * (REQ_SLOTS - 1))

    def test_sentinel_tail_of_static_buffers_still_reset(self):
        runner = _runner()
        runner.load_batch(_batch([11, 12, 13]))
        runner.load_batch(_batch([21]))
        s = runner._prefill_static_buffers
        self.assertEqual(s["seq_lens"][1:REQ_SLOTS].tolist(), [0] * (REQ_SLOTS - 1))
        self.assertEqual(
            s["req_pool_indices"][1:REQ_SLOTS].tolist(), [0] * (REQ_SLOTS - 1)
        )


if __name__ == "__main__":
    unittest.main()
