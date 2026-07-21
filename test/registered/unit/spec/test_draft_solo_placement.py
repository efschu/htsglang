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
  short-circuit without collectives. Row counts travel via device-tensor
  all_gather — broadcast_object is forbidden (mq broadcaster is src=0
  only; the old per-source object broadcast crashed every solo boot).
* Capture / co-location safety: _broadcast_draft_picks routes through the
  pynccl communicator whenever it is available — warmup, capture, and
  eager alike (c10d's lazy NCCL comm init inside the capture phase is
  ncclInvalidUsage, and torch's bundled NCCL rejects co-located ranks
  outright); c10d is only the non-CUDA fallback.
* Shadow draft-runner surface: install_shadow_draft_runner_surface gives
  shadow ranks' runners explicit None attributes for everything a
  never-running draft lacks, and loud solo-mode AttributeErrors otherwise.
* Init gating: shadow ranks build no draft attention backends, capture no
  draft CUDA graphs (incl. adaptive ladder rungs, which route through the
  same methods), and allocate no draft KV pool.
* ModelRunner role computation (compute_draft_solo_role).
"""

import contextlib
import re
import unittest
from pathlib import Path
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
    """Fake GroupCoordinator: records collectives; recv side is fed from a
    payload table (src -> tensor). broadcast_object is POISONED: the real
    group's mq broadcaster only supports src=0, so the vocab gather must
    never touch the object path (boot regression: per-source
    broadcast_object crashed every multi-rank solo boot at init)."""

    def __init__(self, world_size=3, rank_in_group=0, payloads=None):
        self.world_size = world_size
        self.rank_in_group = rank_in_group
        self.payloads = payloads or {}
        self.sent = []  # (tensor_clone, src)
        self.gathered = []  # (tensor_clone, dim)

    def broadcast(self, tensor, src=0):
        if src == self.rank_in_group:
            self.sent.append((tensor.clone(), src))
        else:
            tensor.copy_(self.payloads[src].to(tensor.dtype))
        return tensor

    def all_gather(self, tensor, dim=0):
        """Device-tensor all_gather: each remote rank contributes its
        payload's row count (mirrors the real row-count announcement)."""
        self.gathered.append((tensor.clone(), dim))
        parts = []
        for r in range(self.world_size):
            if r == self.rank_in_group:
                parts.append(tensor)
            else:
                parts.append(
                    torch.tensor([self.payloads[r].shape[0]], dtype=tensor.dtype)
                )
        return torch.cat(parts, dim=dim)

    def broadcast_object(self, obj=None, src=0):
        raise AssertionError(
            "broadcast_object must not be used: the mq broadcaster only "
            "supports src=0 (parallel_state.GroupCoordinator.broadcast_object)"
        )


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


class _FakePyNccl:
    """Fake PyNcclCommunicator: records broadcasts and whether the comm was
    enabled (change_state) at call time."""

    def __init__(self, available=True):
        self.available = available
        self.disabled = True
        self.broadcasts = []  # (tensor_clone, src, disabled_at_call)
        self.change_state_enables = []

    @contextlib.contextmanager
    def change_state(self, enable=None):
        self.change_state_enables.append(enable)
        old_disabled = self.disabled
        self.disabled = not enable
        try:
            yield
        finally:
            self.disabled = old_disabled

    def broadcast(self, tensor, src):
        self.broadcasts.append((tensor.clone(), src, self.disabled))


class TestBroadcastDraftPicksCaptureSafe(CustomTestCase):
    """Boot regression (co-located tp>cards NEXTN): GroupCoordinator's hard
    c10d broadcast can never serve the pick sync — its NCCL communicator
    initializes lazily on first use, which lands in the draft graph-capture
    phase (warmup included -> ncclInvalidUsage), and torch's bundled NCCL
    rejects co-located ranks outright ("Duplicate GPU detected"). The pick
    broadcast must therefore use the pynccl communicator WHENEVER it is
    available — warmup, capture, and eager alike (the DFLASH pynccl-only
    pattern) — with c10d only as the non-CUDA fallback."""

    def _run(self, pynccl_comm):
        c10d_calls = []
        fake_group = SimpleNamespace(
            world_size=3,
            broadcast=lambda t, src: c10d_calls.append(src),
            pynccl_comm=pynccl_comm,
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
            _broadcast_draft_picks(torch.zeros(2, 1), None, torch.ones(2, 1))
        return c10d_calls

    def test_available_pynccl_is_always_used(self):
        pynccl = _FakePyNccl(available=True)
        c10d_calls = self._run(pynccl_comm=pynccl)
        self.assertEqual(c10d_calls, [])
        self.assertEqual(len(pynccl.broadcasts), 2)  # None arg skipped
        for _, src, disabled_at_call in pynccl.broadcasts:
            self.assertEqual(src, 0)
            self.assertFalse(disabled_at_call)  # enabled via change_state
        self.assertEqual(pynccl.change_state_enables, [True])

    def test_unavailable_pynccl_falls_back_to_c10d(self):
        pynccl = _FakePyNccl(available=False)
        c10d_calls = self._run(pynccl_comm=pynccl)
        self.assertEqual(c10d_calls, [0, 0])
        self.assertEqual(pynccl.broadcasts, [])

    def test_missing_pynccl_attr_falls_back_to_c10d(self):
        c10d_calls = self._run(pynccl_comm=None)
        self.assertEqual(c10d_calls, [0, 0])

    def test_helper_src_passthrough_and_none_skip(self):
        from sglang.srt.speculative.spec_utils import capture_safe_tp_broadcast

        pynccl = _FakePyNccl(available=True)
        group = SimpleNamespace(pynccl_comm=pynccl)
        payload = torch.arange(4).reshape(2, 2)
        capture_safe_tp_broadcast(group, (None, payload), src=2)
        self.assertEqual(len(pynccl.broadcasts), 1)
        sent, src, disabled_at_call = pynccl.broadcasts[0]
        self.assertTrue(torch.equal(sent, payload))
        self.assertEqual(src, 2)
        self.assertFalse(disabled_at_call)


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
        # Row counts announced via ONE device-tensor all_gather (int64),
        # never via broadcast_object (poisoned in _RecordingGroup).
        self.assertEqual(len(group.gathered), 1)
        rows_t, dim = group.gathered[0]
        self.assertEqual(rows_t.dtype, torch.int64)
        self.assertEqual(rows_t.tolist(), [4])
        self.assertEqual(dim, 0)
        # The assembled table owns its storage (no view into the shard —
        # the shard temporaries must be freeable afterwards).
        self.assertNotEqual(result.data_ptr(), shards[0].data_ptr())
        self.assertTrue(result.is_contiguous())

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
        self.assertEqual(len(group.gathered), 1)  # row-count announcement

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


class TestShadowDraftRunnerSurface(CustomTestCase):
    """Boot regressions (solo shadows): generic init/serve paths dereference
    draft_runner attributes that shadows never build — the adaptive
    _override_worker_state backup (draft_runner.draft_attn_backend) and the
    disaggregation KV builder (draft_runner.token_to_kv_pool, reached even
    with disaggregation_mode "null"). The shadow surface must expose these
    as None and make everything else fail with a loud solo-mode message."""

    class _FakeRunner:
        def __init__(self):
            self.attn_backend = None  # pre-set by ModelRunner.initialize()
            self.model_config = SimpleNamespace()

    def _installed(self):
        from sglang.srt.speculative.eagle_worker_v2 import (
            install_shadow_draft_runner_surface,
        )

        runner = self._FakeRunner()
        install_shadow_draft_runner_surface(runner)
        return runner

    def test_commonly_dereferenced_attrs_are_none(self):
        runner = self._installed()
        for attr in (
            "attn_backend",
            "draft_attn_backend",
            "token_to_kv_pool",
            "token_to_kv_pool_allocator",
            "req_to_token_pool",
            "decode_cuda_graph_runner",
        ):
            self.assertIsNone(getattr(runner, attr), attr)

    def test_preexisting_attrs_not_clobbered(self):
        from sglang.srt.speculative.eagle_worker_v2 import (
            install_shadow_draft_runner_surface,
        )

        runner = self._FakeRunner()
        sentinel = object()
        runner.token_to_kv_pool = sentinel
        install_shadow_draft_runner_surface(runner)
        self.assertIs(runner.token_to_kv_pool, sentinel)

    def test_missing_attr_raises_loud_solo_error(self):
        runner = self._installed()
        with self.assertRaisesRegex(
            AttributeError, "speculative-draft-placement"
        ):
            runner.definitely_not_an_attribute

    def test_hasattr_and_getattr_default_probes_unchanged(self):
        runner = self._installed()
        self.assertFalse(hasattr(runner, "nonexistent"))
        self.assertIsNone(getattr(runner, "nonexistent", None))
        self.assertTrue(isinstance(runner, self._FakeRunner))

    def test_idempotent_no_class_chain_growth(self):
        from sglang.srt.speculative.eagle_worker_v2 import (
            install_shadow_draft_runner_surface,
        )

        runner = self._installed()
        cls_after_first = type(runner)
        install_shadow_draft_runner_surface(runner)
        self.assertIs(type(runner), cls_after_first)

    def test_disagg_kv_builder_returns_none_for_shadow(self):
        from sglang.srt.mem_cache.kv_cache_builder import get_draft_kv_pool

        runner = self._installed()
        draft_worker = SimpleNamespace(
            draft_worker=SimpleNamespace(draft_runner=runner)
        )
        pool = get_draft_kv_pool(
            draft_worker=draft_worker,
            spec_algorithm=SimpleNamespace(is_ngram=lambda: False),
            server_args=SimpleNamespace(enable_multi_layer_eagle=False),
        )
        self.assertIsNone(pool)


class TestSoloHostRankLocalCaptureBarrier(CustomTestCase):
    """Boot deadlock regression: the solo HOST captures its draft graphs
    rank-locally (weight-TP=1, collective-free forward) while shadows skip
    draft capture — so the capture backends' per-warmup TP barrier waited
    on ranks that never enter it. Runners flagged spec_solo_rank_local_graphs
    must skip that barrier; unflagged runners keep it."""

    def _backend(self, flagged):
        from sglang.srt.model_executor.runner_backend.full_cuda_graph_backend import (
            FullCudaGraphBackend,
        )

        barrier_calls = []
        runner = SimpleNamespace(
            tp_group=SimpleNamespace(barrier=lambda: barrier_calls.append(1)),
        )
        if flagged:
            runner.spec_solo_rank_local_graphs = True
        fake_cgr = SimpleNamespace(
            model_runner=runner,
            device_module=SimpleNamespace(synchronize=lambda: None),
        )
        backend = object.__new__(FullCudaGraphBackend)
        # Mirror only the __init__ lines under test.
        backend._tp_group = fake_cgr.model_runner.tp_group
        backend._skip_warmup_barrier = getattr(
            fake_cgr.model_runner, "spec_solo_rank_local_graphs", False
        )
        return backend, barrier_calls

    def test_flagged_runner_skips_barrier(self):
        backend, calls = self._backend(flagged=True)
        self.assertTrue(backend._skip_warmup_barrier)
        if not backend._skip_warmup_barrier:
            backend._tp_group.barrier()
        self.assertEqual(calls, [])

    def test_unflagged_runner_keeps_barrier(self):
        backend, calls = self._backend(flagged=False)
        self.assertFalse(backend._skip_warmup_barrier)
        if not backend._skip_warmup_barrier:
            backend._tp_group.barrier()
        self.assertEqual(calls, [1])

    def test_all_capture_backends_gate_the_barrier(self):
        """Source-level check: every runner-backend warmup barrier is gated
        by _skip_warmup_barrier (a new backend copying the old unguarded
        pattern would reintroduce the solo deadlock)."""
        import sglang.srt.model_executor.runner_backend as rb_pkg

        rb_dir = Path(list(rb_pkg.__path__)[0])
        offenders = []
        for path in sorted(rb_dir.glob("*.py")):
            text = path.read_text(encoding="utf-8")
            for m in re.finditer(r"^(\s*)self\._tp_group\.barrier\(\)", text, re.M):
                # The barrier line must sit under an
                # `if not self._skip_warmup_barrier:` guard.
                before = text[: m.start()].rsplit("\n", 3)
                guarded = any(
                    "_skip_warmup_barrier" in line for line in before[-3:]
                )
                if not guarded:
                    line_no = text.count("\n", 0, m.start()) + 1
                    offenders.append(f"{path.name}:{line_no}")
        self.assertEqual(
            offenders,
            [],
            "Unguarded capture-warmup TP barrier(s) (solo-host draft capture "
            f"would deadlock): {offenders}",
        )


class TestNoNonzeroSrcBroadcastObject(CustomTestCase):
    """Source-level regression guard: GroupCoordinator.broadcast_object
    asserts src == 0 when the mq broadcaster is attached (the normal TP
    setup), so ANY broadcast_object call with a non-zero src crashes at
    runtime. The solo vocab gather used to do exactly that (src=1, src=2)
    and broke every multi-rank solo boot."""

    @staticmethod
    def _call_args_text(text, start):
        """Text of a call's argument list, from just past the open paren to
        its matching close paren (handles nested parens)."""
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
            i += 1
        return text[start : i - 1]

    def test_speculative_sources_use_only_src0_broadcast_object(self):
        import sglang.srt.speculative.eagle_worker_v2 as eagle_mod

        spec_dir = Path(eagle_mod.__file__).resolve().parent
        offenders = []
        for path in sorted(spec_dir.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for m in re.finditer(r"\.broadcast_object\s*\(", text):
                args = self._call_args_text(text, m.end())
                src_match = re.search(r"\bsrc\s*=\s*([^,)\s]+)", args)
                if src_match and src_match.group(1) != "0":
                    line = text.count("\n", 0, m.start()) + 1
                    offenders.append(
                        f"{path.relative_to(spec_dir)}:{line}: "
                        f"broadcast_object(..., src={src_match.group(1)})"
                    )
        self.assertEqual(
            offenders,
            [],
            "broadcast_object only supports src=0 under the mq broadcaster; "
            "use device-tensor collectives instead: " + "; ".join(offenders),
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
