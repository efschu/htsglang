"""Unit tests for draft-solo placement (--speculative-draft-placement solo)
in the EAGLE-family v2 worker — hermetic, CPU-only, groups mocked.

Covered contracts:
* Split default: class-level solo flags default to OFF on EVERY draft
  worker / spec worker variant (#136a lesson: several variants bypass
  EagleDraftWorker.__init__, so base-class code must find the flags as
  class attributes).
* Broadcast contract: the solo host sends ONE [bs, num_steps] int64
  broadcast per round from the solo rank; a shadow allocates the matching
  buffer, consumes the broadcast, and never runs prepare_for_draft /
  draft_forward.
* Chain-tree metadata: the shadow's locally rebuilt (parent_list,
  top_scores_index) equal the host's preallocated topk=1 constants.
* Vocab gather: _solo_gather_full_vocab_rows reassembles the full table
  from (uneven) shards; only keep=True ranks retain it; replicated tables
  short-circuit without collectives.
* Init gating: shadow ranks build no draft attention backends, capture no
  draft CUDA graphs (incl. adaptive ladder rungs, which route through the
  same methods), and allocate no draft KV pool.
* ModelRunner role computation (compute_draft_solo_role).
"""

import contextlib
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.model_executor.model_runner import compute_draft_solo_role
from sglang.srt.speculative.base_spec_worker import (
    BaseSpecWorker,
    EagleDraftWorkerBase,
)
from sglang.srt.speculative.eagle_worker_v2 import (
    EagleDraftWorker,
    EAGLEWorkerV2,
    _broadcast_draft_picks,
    _solo_gather_full_vocab_rows,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

DEVICE = "cpu"


def _fake_server_args(**fields):
    ns = SimpleNamespace(**fields)

    def _override(source, **updates):
        for key, value in updates.items():
            setattr(ns, key, value)

    ns.override = _override
    return ns


def _make_worker(
    num_steps=3,
    solo_active=True,
    is_host=False,
    solo_rank=0,
    max_bs=8,
):
    worker = object.__new__(EagleDraftWorker)
    worker.topk = 1
    worker.device = DEVICE
    worker.speculative_num_steps = num_steps
    worker.speculative_num_draft_tokens = num_steps + 1
    worker.server_args = _fake_server_args(
        cuda_graph_config=SimpleNamespace(decode=SimpleNamespace(max_bs=max_bs)),
        max_running_requests=max_bs,
        speculative_use_rejection_sampling=False,
    )
    worker._spec_solo_active = solo_active
    worker._spec_solo_is_host = is_host
    worker._spec_solo_rank = solo_rank
    worker._rebuild_topk1_chain_buffers()
    return worker


class _RecordingGroup:
    """Fake GroupCoordinator: records broadcasts; recv side is fed from a
    payload table (src -> tensor)."""

    def __init__(self, world_size=3, rank_in_group=0, payloads=None):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.payloads = payloads or {}
        self.sent = []  # (tensor_clone, src)
        self.object_log = []

    def broadcast(self, tensor, src=0):
        if src == self.rank_in_group:
            self.sent.append((tensor.clone(), src))
        else:
            tensor.copy_(self.payloads[src].to(tensor.dtype))
        return tensor

    def broadcast_object(self, obj=None, src=0):
        if src == self.rank_in_group:
            self.object_log.append((obj, src))
            return obj
        return list(self.payloads[src].shape)


class TestClassDefaults(CustomTestCase):
    """#136a guard: solo flags must exist as CLASS attributes on every
    draft-worker / spec-worker variant, defaulting to the split path."""

    def _variant_classes(self):
        from sglang.srt.speculative.frozen_kv_mtp_worker_v2 import (
            FrozenKVMTPDraftWorker,
            FrozenKVMTPWorkerV2,
        )
        from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
            MultiLayerEagleDraftWorker,
            MultiLayerEagleWorkerV2,
        )
        from sglang.srt.speculative.standalone_worker_v2 import (
            StandaloneDraftWorker,
            StandaloneWorkerV2,
        )

        return [
            EagleDraftWorkerBase,
            BaseSpecWorker,
            EagleDraftWorker,
            EAGLEWorkerV2,
            StandaloneDraftWorker,
            StandaloneWorkerV2,
            FrozenKVMTPDraftWorker,
            FrozenKVMTPWorkerV2,
            MultiLayerEagleDraftWorker,
            MultiLayerEagleWorkerV2,
        ]

    def test_solo_flags_default_off_on_every_variant(self):
        for cls in self._variant_classes():
            self.assertFalse(
                getattr(cls, "_spec_solo_active"),
                f"{cls.__name__}._spec_solo_active must default to False",
            )
            self.assertTrue(
                getattr(cls, "_spec_solo_is_host"),
                f"{cls.__name__}._spec_solo_is_host must default to True",
            )
            self.assertEqual(getattr(cls, "_spec_solo_rank"), 0, cls.__name__)

    def test_split_worker_uses_nullcontext_build_ctx(self):
        worker = _make_worker(solo_active=False, is_host=True)
        self.assertIsInstance(worker._solo_build_ctx(), contextlib.nullcontext)


class TestBroadcastDraftPicksSoloGate(CustomTestCase):
    def test_solo_single_rank_is_noop(self):
        # Must return before touching any distributed state: exactly one
        # rank runs the draft forward, a broadcast here would hang.
        with patch(
            "sglang.srt.distributed.get_tp_group",
            side_effect=AssertionError("group must not be touched"),
        ):
            _broadcast_draft_picks(
                torch.zeros(2, 1), torch.ones(2, 1), None, solo_single_rank=True
            )

    def test_default_still_broadcasts(self):
        calls = []
        fake_group = SimpleNamespace(
            world_size=3, broadcast=lambda t, src: calls.append(src)
        )
        with (
            patch(
                "sglang.srt.distributed.get_tp_group", return_value=fake_group
            ),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
        ):
            _broadcast_draft_picks(torch.zeros(2, 1), None)
        self.assertEqual(calls, [0])


class TestTokenBroadcastContract(CustomTestCase):
    def test_host_sends_bs_by_steps_int64_from_solo_rank(self):
        worker = _make_worker(num_steps=3, is_host=True, solo_rank=1)
        group = _RecordingGroup(world_size=3, rank_in_group=1)
        worker._solo_tp_group = lambda: group
        # Host-native draft_tokens: flat int32 (graph-path shape).
        draft_tokens = torch.arange(2 * 3, dtype=torch.int32)
        worker._solo_send_draft_tokens(draft_tokens, bs=2)
        self.assertEqual(len(group.sent), 1)
        payload, src = group.sent[0]
        self.assertEqual(src, 1)
        self.assertEqual(payload.shape, (2, 3))
        self.assertEqual(payload.dtype, torch.int64)
        self.assertTrue(torch.equal(payload.flatten(), draft_tokens.long()))

    def test_host_send_noop_on_world_size_1(self):
        worker = _make_worker(num_steps=3, is_host=True)
        group = _RecordingGroup(world_size=1)
        worker._solo_tp_group = lambda: group
        worker._solo_send_draft_tokens(torch.zeros(3, dtype=torch.int64), bs=1)
        self.assertEqual(group.sent, [])

    def test_shadow_recv_matches_host_payload(self):
        host_tokens = torch.tensor([[5, 6, 7], [8, 9, 10]], dtype=torch.int64)
        worker = _make_worker(num_steps=3, is_host=False, solo_rank=0)
        group = _RecordingGroup(
            world_size=3, rank_in_group=2, payloads={0: host_tokens}
        )
        worker._solo_tp_group = lambda: group
        received = worker._solo_recv_draft_tokens(bs=2)
        self.assertEqual(received.dtype, torch.int64)
        self.assertTrue(torch.equal(received, host_tokens))


class TestChainTreeMeta(CustomTestCase):
    def test_prealloc_and_fresh_paths_agree(self):
        worker = _make_worker(num_steps=3, max_bs=4)
        small_parents, small_scores = worker._topk1_chain_meta(2)
        # bs beyond the prealloc triggers the fresh-build path.
        big_parents, big_scores = worker._topk1_chain_meta(6)
        self.assertTrue(torch.equal(big_parents[:2], small_parents))
        self.assertTrue(torch.equal(big_scores[:2], small_scores))
        self.assertEqual(big_parents.shape[0], 6)

    def test_matches_prealloc_constants(self):
        worker = _make_worker(num_steps=4, max_bs=8)
        parents, scores = worker._topk1_chain_meta(3)
        self.assertTrue(
            torch.equal(parents, worker._topk1_parents_prealloc[:3])
        )
        self.assertTrue(
            torch.equal(scores, worker._topk1_score_indices_prealloc[:3])
        )


class TestShadowDraftSkipsForward(CustomTestCase):
    def _batch(self, bs=2, idle=False):
        return SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: idle),
            seq_lens=torch.zeros(bs, dtype=torch.int64),
            spec_info=SimpleNamespace(bonus_tokens=torch.zeros(bs, dtype=torch.int32)),
        )

    def test_shadow_receives_and_never_runs_draft(self):
        worker = _make_worker(num_steps=3, is_host=False, solo_rank=0)
        worker.prepare_for_draft = MagicMock(
            side_effect=AssertionError("shadow must not prepare a draft batch")
        )
        worker.draft_forward = MagicMock(
            side_effect=AssertionError("shadow must not run the draft forward")
        )
        tokens = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int64)
        worker._solo_recv_draft_tokens = MagicMock(return_value=tokens)
        finish = MagicMock(return_value="verify-input")
        worker._finish_draft_tree = finish

        batch = self._batch(bs=2)
        result = worker.draft(batch)

        self.assertEqual(result, "verify-input")
        worker._solo_recv_draft_tokens.assert_called_once_with(2)
        worker.prepare_for_draft.assert_not_called()
        worker.draft_forward.assert_not_called()
        args = finish.call_args.args
        # (batch, draft_input, parent_list, top_scores_index, draft_tokens, draft_probs)
        self.assertIs(args[0], batch)
        self.assertTrue(torch.equal(args[2], worker._topk1_parents_prealloc[:2]))
        self.assertTrue(
            torch.equal(args[3], worker._topk1_score_indices_prealloc[:2])
        )
        self.assertTrue(torch.equal(args[4], tokens))
        self.assertIsNone(args[5])

    def test_shadow_idle_skips_broadcast(self):
        worker = _make_worker(num_steps=3, is_host=False)
        worker._solo_recv_draft_tokens = MagicMock(
            side_effect=AssertionError("idle round must not broadcast")
        )
        result = worker.draft(self._batch(bs=0, idle=True))
        self.assertTrue(result.is_verify_input())
        worker._solo_recv_draft_tokens.assert_not_called()

    def test_host_broadcasts_once_after_draft(self):
        worker = _make_worker(num_steps=1, is_host=True)
        worker.req_to_token_pool = None
        worker.cuda_graph_runner = None
        worker.draft_runner = SimpleNamespace(canary_manager=None)
        forward_batch = SimpleNamespace(
            forward_mode=SimpleNamespace(is_idle=lambda: False)
        )
        worker.prepare_for_draft = MagicMock(return_value=(forward_batch, False))
        tokens = torch.tensor([[42], [43]], dtype=torch.int64)
        worker.draft_forward = MagicMock(
            return_value=(
                worker._topk1_parents_prealloc[:2],
                worker._topk1_score_indices_prealloc[:2],
                tokens,
                None,
            )
        )
        worker._solo_send_draft_tokens = MagicMock()
        worker._finish_draft_tree = MagicMock(return_value="verify-input")

        batch = self._batch(bs=2)
        result = worker.draft(batch)

        self.assertEqual(result, "verify-input")
        worker.draft_forward.assert_called_once()
        worker._solo_send_draft_tokens.assert_called_once()
        sent_tokens, sent_bs = worker._solo_send_draft_tokens.call_args.args
        self.assertTrue(torch.equal(sent_tokens, tokens))
        self.assertEqual(sent_bs, 2)


class TestVocabGather(CustomTestCase):
    def _shards(self):
        torch.manual_seed(7)
        full = torch.randn(10, 4)
        # Uneven split 4/3/3 with one padding row appended to shard 0.
        shard0 = torch.cat([full[0:4], torch.zeros(1, 4)])
        return full, {0: full[0:4], 1: full[4:7], 2: full[7:10]}, [shard0, full[4:7], full[7:10]]


    def test_host_assembles_full_table(self):
        full, valid_payloads, shards = self._shards()
        group = _RecordingGroup(
            world_size=3, rank_in_group=0, payloads=valid_payloads
        )
        result = _solo_gather_full_vocab_rows(
            group, shards[0], valid_rows=4, out_rows=10, keep=True
        )
        self.assertEqual(result.shape, (10, 4))
        self.assertTrue(torch.equal(result, full))
        # The host broadcast its own VALID rows only (padding stripped).
        self.assertEqual(group.sent[0][0].shape, (4, 4))

    def test_shadow_contributes_and_keeps_nothing(self):
        full, valid_payloads, shards = self._shards()
        group = _RecordingGroup(
            world_size=3, rank_in_group=1, payloads=valid_payloads
        )
        result = _solo_gather_full_vocab_rows(
            group, shards[1], valid_rows=3, out_rows=10, keep=False
        )
        self.assertIsNone(result)
        self.assertEqual(len(group.sent), 1)  # its own shard broadcast

    def test_short_assembly_raises(self):
        _, valid_payloads, shards = self._shards()
        group = _RecordingGroup(
            world_size=3, rank_in_group=0, payloads=valid_payloads
        )
        with self.assertRaisesRegex(RuntimeError, "assembled only"):
            _solo_gather_full_vocab_rows(
                group, shards[0], valid_rows=4, out_rows=64, keep=True
            )

    def test_replicated_table_short_circuits(self):
        worker = _make_worker(is_host=True)
        table = torch.randn(12, 4)
        poisoned_group = SimpleNamespace(
            world_size=3,
            rank_in_group=0,
            broadcast=MagicMock(side_effect=AssertionError("no collective")),
            broadcast_object=MagicMock(side_effect=AssertionError("no collective")),
        )
        out = worker._solo_gather_or_local(
            poisoned_group, table, module=None, out_rows=10, keep=True
        )
        self.assertTrue(torch.equal(out, table[:10]))
        self.assertIsNone(
            worker._solo_gather_or_local(
                poisoned_group, table, module=None, out_rows=10, keep=False
            )
        )

    def test_added_vocab_layout_rejected(self):
        worker = _make_worker(is_host=True)
        shard = torch.randn(4, 4)
        module = SimpleNamespace(
            shard_indices=SimpleNamespace(num_added_elements=2, num_org_elements=2)
        )
        with self.assertRaises(NotImplementedError):
            worker._solo_gather_or_local(
                SimpleNamespace(world_size=3, rank_in_group=0),
                shard,
                module,
                out_rows=10,
                keep=True,
            )


class TestShadowInitGating(CustomTestCase):
    def test_shadow_captures_no_draft_graphs(self):
        worker = _make_worker(is_host=False)
        # No other attributes needed: the shadow gate must return before
        # touching server args / backends (this is also the adaptive-ladder
        # path — build_adaptive_runtime_state routes through this method).
        worker._capture_cuda_graphs()
        self.assertIsNone(worker.cuda_graph_runner)
        self.assertIsNone(worker.cuda_graph_runner_for_draft_extend)

    def test_shadow_builds_no_attention_backends(self):
        worker = _make_worker(is_host=False)
        worker.draft_runner = SimpleNamespace(draft_attn_backend="sentinel")
        worker.init_attention_backend()
        self.assertIsNone(worker.draft_attn_backend)
        self.assertIsNone(worker.draft_extend_attn_backend)
        self.assertIsNone(worker.draft_runner.draft_attn_backend)

    def test_shadow_init_attention_backends_skips_worker(self):
        worker = _make_worker(is_host=False)
        worker.draft_worker = MagicMock()
        worker.init_attention_backends()
        worker.draft_worker.init_attention_backends.assert_not_called()
        self.assertIsNone(worker.draft_attn_backend)

    def test_shadow_allocates_no_draft_kv_pool(self):
        worker = _make_worker(is_host=False)
        worker.draft_worker = MagicMock()
        worker._embed_head_shared_early = True
        worker.alloc_memory_pool(
            memory_pool_config="cfg",
            req_to_token_pool="pool",
            token_to_kv_pool_allocator="alloc",
        )
        worker.draft_worker.alloc_memory_pool.assert_not_called()
        # Shared pools are still referenced (scheduler-side bookkeeping).
        self.assertEqual(worker.req_to_token_pool, "pool")
        self.assertEqual(worker.token_to_kv_pool_allocator, "alloc")

    def test_shadow_init_cuda_graphs_sets_none_runners(self):
        worker = _make_worker(is_host=False)
        worker.draft_worker = MagicMock()
        worker.init_cuda_graphs()
        worker.draft_worker.init_cuda_graphs.assert_not_called()
        self.assertIsNone(worker.cuda_graph_runner)
        self.assertIsNone(worker.cuda_graph_runner_for_draft_extend)

    def test_host_build_ctx_overrides_to_tp1(self):
        worker = _make_worker(is_host=True)
        from sglang.srt.runtime_context import get_parallel

        with worker._solo_build_ctx():
            self.assertEqual(get_parallel().tp_size, 1)
            self.assertEqual(get_parallel().attn_tp_size, 1)
            self.assertEqual(get_parallel().tp_rank, 0)


class TestSoloStubDraftInput(CustomTestCase):
    def test_stub_shapes_and_bonus_passthrough(self):
        worker = object.__new__(EAGLEWorkerV2)
        worker.device = DEVICE
        worker.topk = 1
        worker._draft_worker = SimpleNamespace(
            draft_runner=SimpleNamespace(
                spec_algorithm=SimpleNamespace(is_standalone=lambda: False),
                model_config=SimpleNamespace(
                    spec_hidden_size=16, dtype=torch.float32
                ),
            )
        )
        batch = SimpleNamespace(seq_lens=torch.zeros(3, dtype=torch.int64))
        next_token_ids = torch.tensor([11, 12, 13], dtype=torch.int32)
        stub = worker._solo_stub_draft_input(batch, next_token_ids)
        self.assertIs(stub.bonus_tokens, next_token_ids)
        self.assertEqual(stub.topk_p.shape, (3, 1))
        self.assertEqual(stub.topk_index.shape, (3, 1))
        self.assertEqual(stub.topk_index.dtype, torch.int64)
        self.assertEqual(stub.hidden_states.shape, (3, 16))


class TestComputeDraftSoloRole(CustomTestCase):
    def _args(self, placement="solo", solo_rank=1):
        return SimpleNamespace(
            speculative_draft_placement=placement,
            speculative_draft_solo_rank=lambda: solo_rank,
        )

    def test_split_is_never_host_nor_shadow(self):
        self.assertEqual(
            compute_draft_solo_role(self._args("split"), True, 0), (False, False)
        )

    def test_target_runner_is_never_host_nor_shadow(self):
        self.assertEqual(
            compute_draft_solo_role(self._args(), False, 1), (False, False)
        )

    def test_draft_on_solo_rank_is_host(self):
        self.assertEqual(compute_draft_solo_role(self._args(), True, 1), (True, False))

    def test_draft_on_other_rank_is_shadow(self):
        self.assertEqual(compute_draft_solo_role(self._args(), True, 2), (False, True))

    def test_args_without_flag_default_split(self):
        self.assertEqual(
            compute_draft_solo_role(SimpleNamespace(), True, 0), (False, False)
        )


if __name__ == "__main__":
    unittest.main()
