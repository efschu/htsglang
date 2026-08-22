"""Regression tests for the SWA chunked-req stash gate (#24252)."""

import unittest
from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import NextBatchPlan, Req
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.chunk_cache import ChunkCache
from sglang.srt.utils.common import Range

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _make_req(
    *,
    req_pool_idx: int,
    fill_ids: list,
    prefix_indices: torch.Tensor,
    extend_input_len: int,
    fill_len: int,
) -> Req:
    req = Req.__new__(Req)
    req.rid = "test-req"
    req.origin_input_ids = array("q", fill_ids)
    req.output_ids = array("q")
    req.full_untruncated_fill_ids = array("q", fill_ids)
    req.prefix_indices = prefix_indices
    req.req_pool_idx = req_pool_idx
    req.extend_range = Range(fill_len - extend_input_len, fill_len)
    req.inflight_middle_chunks = 0
    req.host_hit_length = 0
    req.cache_protected_len = 0
    req.skip_radix_cache_insert = False
    req.last_node = None
    req.swa_uuid_for_lock = None
    req.session = None
    req.return_logprob = False
    req.logprob_start_len = -1
    req.positional_embed_overrides = None
    req.extra_key = None
    req.mamba_pool_idx = None
    # #679 follow-up: this helper builds a Req with __new__, so every field the
    # class gained since it was written must be restated here or the first
    # accessor raises. Default taken from Req.__init__ (schedule_batch.py:759),
    # not invented -- a stub that guesses a default is a test that passes for
    # the wrong reason.
    req.kv_spill_state = None
    req.sampling_params = SimpleNamespace(max_new_tokens=128, ignore_eos=False)
    return req


def _make_req_to_token_pool(num_slots: int, max_context: int) -> SimpleNamespace:
    # Slot s contains a recognizable fingerprint [s*1000, s*1000+1, ...]
    # so we can tell a corrupted prefix_indices from a healthy one by content.
    pool = SimpleNamespace()
    pool.req_to_token = (
        torch.arange(max_context, dtype=torch.int32).unsqueeze(0).repeat(num_slots, 1)
        + torch.arange(num_slots, dtype=torch.int32).unsqueeze(1) * 1000
    )
    return pool


def _make_chunk_cache(req_to_token_pool) -> ChunkCache:
    return ChunkCache(
        SimpleNamespace(
            req_to_token_pool=req_to_token_pool,
            token_to_kv_pool_allocator=None,
            page_size=1,
        )
    )


def _scheduler_for_get_next_batch(*, tree_cache, chunked_req) -> Scheduler:
    s = Scheduler.__new__(Scheduler)
    s._abort_on_waiting_timeout = MagicMock()
    s._abort_on_running_timeout = MagicMock()
    s.dllm_config = None
    s.dllm_manager = None
    s.enable_hisparse = False
    s.enable_fpm = False
    s.last_batch = None
    s.require_mlp_sync = False
    # #797 (6f42d7f923) made the prologue consult the PP topology before
    # anything else: `if self.ps.pp_size > 1 and getattr(self,
    # "_pp_admission_pass_voided", False)` at scheduler.py:6154. Read on `self`
    # directly, not through getattr, so a `__new__`-built scheduler has to
    # carry it.
    #
    # A real namespace and not a MagicMock, for the reason the block below
    # already records about `server_args`: a bare MagicMock makes every flag
    # truthy, and here it would make `pp_size > 1` a comparison against a Mock
    # rather than the topology check it is. `pp_size=1` is this fixture's true
    # value -- these cases drive `get_next_batch_to_run` on a single-rank
    # scheduler -- and it is the value on which #797's void gate is scoped OUT,
    # which is what the gate under test assumes. The voided-pass behaviour
    # belongs to the PP event loop and is pinned in the #797 and #791 files.
    s.ps = SimpleNamespace(pp_size=1, pp_rank=0)
    s.spec_algorithm = MagicMock()
    # A bare MagicMock makes EVERY flag truthy, which arms the phase-boundary
    # actuators this gate test has nothing to do with (kv-reshard raised on
    # its own precondition). The flags the prologue reads are pinned off.
    s.server_args = MagicMock(
        speculative_skip_dp_mlp_sync=True,
        kv_reshard_vectors=None,
        kv_pressure_ladder=None,
        regime_controller="off",
        gdn_state_set_ladder=None,
        enable_vram_dial=False,
        # #679 follow-up: the SAME trap this block was written for, one
        # actuator further on. An unpinned flag is TRUTHY here, so the
        # prologue entered _phase_flip_on_round, which lazy-BUILDS a
        # PhaseFlipRuntime from a Scheduler that has none of the state a
        # build needs -- AttributeError on phase_flip_runtime, and three
        # tests in this file red ever since the flip gained that hook.
        # This gate has nothing to do with phase flips; pin it off with the
        # rest rather than teach the stub to build a runtime it never uses.
        enable_phase_flip=False,
    )
    s.running_batch = MagicMock()
    s.running_batch.is_empty.return_value = True
    s.running_batch.is_prefill_only = False
    s.running_batch.batch_is_full = False
    s.running_batch.reqs = []
    s.get_new_batch_prefill = MagicMock(
        return_value=NextBatchPlan(batch_to_run=None, running_batch=s.running_batch)
    )
    s.dp_attn_adapter = MagicMock()
    s.dp_attn_adapter.maybe_prepare_mlp_sync_batch = MagicMock(
        side_effect=lambda batch, **_: batch
    )
    s._maybe_prepare_ngram_embedding = MagicMock(side_effect=lambda batch: batch)
    s.update_running_batch = MagicMock(side_effect=lambda batch: batch)
    s.tree_cache = tree_cache
    s.chunked_req = chunked_req
    s._pending_chunked_abort_req = None
    # get_next_batch_to_run's prologue reads these before it reaches the
    # stash gate under test. A hand-built __new__ stub only carries what the
    # function needed on the day it was written, so it drifts red as the
    # function grows -- which is how it stood before #603 (AttributeError on
    # enable_hierarchical_cache, nothing to do with the gate).
    s.enable_hierarchical_cache = False
    s.enable_fpm = False
    # #603: the rank-uniform pool budget. None offload manager and no TP CPU
    # group means the helper takes its no-collective path and reads the local
    # allocator, so the stub only has to answer available_size().
    s.kv_session_offload = None
    s.tp_cpu_group = None
    s.token_to_kv_pool_allocator = MagicMock(available_size=lambda: 1 << 30)
    # The optional per-iteration runtimes. All of them are "None means the
    # feature is off, take no collective and change nothing", so None is the
    # configuration this gate test wants: it isolates the stash gate from
    # every phase-boundary actuator that shares the prologue.
    s.kv_reshard_runtime = None
    s.kv_pressure_runtime = None
    s.kv_capacity_runtime = None
    s.regime_observer = None
    s.regime_stage_table = None
    s._regime_observer_mode = None
    s.gdn_slot_executor = None
    s.waiting_queue = []
    s.req_to_token_pool = tree_cache.req_to_token_pool
    s.congruent_prefill_lane = None
    return s


class TestStashGatePreservesPrefixIndices(CustomTestCase):
    """Consumer side: real ChunkCache.cache_unfinished_req mutates
    req.prefix_indices iff stash actually runs, so prefix_indices content
    is the bug-detection signal. The stash gate is content-based:
    `fill_len > len(prefix_indices)` means there is freshly computed KV to
    cache; otherwise the chunk was parked and stashing must be skipped."""

    POOL_IDX = 4
    INITIAL_PREFIX_LEN = 8  # what was really cached last iter
    POST_RESET_FILL_LEN = 32  # length after init_next_round_input rebuilds
    NUM_SLOTS = 8
    MAX_CONTEXT = 64

    def _build(self, *, fill_len: int):
        pool = _make_req_to_token_pool(self.NUM_SLOTS, self.MAX_CONTEXT)
        cache = _make_chunk_cache(pool)
        initial_prefix = pool.req_to_token[self.POOL_IDX, : self.INITIAL_PREFIX_LEN].to(
            dtype=torch.int64, copy=True
        )
        req = _make_req(
            req_pool_idx=self.POOL_IDX,
            fill_ids=list(range(self.POST_RESET_FILL_LEN)),
            prefix_indices=initial_prefix,
            extend_input_len=fill_len - self.INITIAL_PREFIX_LEN,
            fill_len=fill_len,
        )
        s = _scheduler_for_get_next_batch(tree_cache=cache, chunked_req=req)
        return s, req, initial_prefix, pool

    def test_parked_chunked_req_keeps_real_prefix_indices(self):
        # A parked chunk has fill_len == len(prefix_indices): no new KV was
        # computed, so the gate must skip stash and leave prefix_indices intact.
        s, req, initial_prefix, _ = self._build(fill_len=self.INITIAL_PREFIX_LEN)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )

        self.assertEqual(req.prefix_indices.shape[0], self.INITIAL_PREFIX_LEN)
        self.assertTrue(torch.equal(req.prefix_indices, initial_prefix))

    def test_scheduled_chunked_req_advances_prefix_indices_via_real_stash(self):
        # Symmetric guard against over-gating: when fill_len has advanced past
        # the cached prefix, stash must run and advance prefix_indices.
        s, req, _, pool = self._build(fill_len=self.POST_RESET_FILL_LEN)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )

        expected = pool.req_to_token[self.POOL_IDX, : self.POST_RESET_FILL_LEN].to(
            dtype=torch.int64
        )
        self.assertEqual(req.prefix_indices.shape[0], self.POST_RESET_FILL_LEN)
        self.assertTrue(torch.equal(req.prefix_indices, expected))

    def test_no_chunked_req_never_mutates_state(self):
        # The outer `if chunked_req is not None` guard must hold on the retract
        # path that clears chunked_req.
        pool = _make_req_to_token_pool(self.NUM_SLOTS, self.MAX_CONTEXT)
        cache = _make_chunk_cache(pool)
        s = _scheduler_for_get_next_batch(tree_cache=cache, chunked_req=None)

        Scheduler.get_next_batch_to_run(
            s, running_batch=s.running_batch, last_batch=s.last_batch
        )
        self.assertIsNone(s.chunked_req)


if __name__ == "__main__":
    unittest.main()
