"""Unit tests for the DFLASH draft-solo HIDDEN-STATE BROADCAST
(--speculative-draft-placement solo) — hermetic, CPU-only, groups faked.

Background. The first solo implementation moved the target's vocab tables to
the solo host: ``_solo_gather_full_vocab_rows`` assembled the FULL unsharded
embed + lm_head from every rank's shard so the host could embed and sample
without a collective the shadow ranks could not join. At a ~150k vocab in
bf16 that is roughly 5 GB pinned on the host for the process lifetime, which
crushed the KV pool on exactly the rank that also carries the unsharded
draft.

The broadcast scheme inverts the direction: the tables STAY sharded, and the
tiny per-round activations travel instead. Contracts covered here:

* Setup does NOT gather. ``_solo_setup_vocab_broadcast`` only records the
  rank-uniform geometry (hidden dim + dtype, both read off the target's
  lm_head shard) and installs the shadow draft-runner surface. No collective,
  no full table, on any rank.
* The staging buffer is grow-only, rank-uniform in shape/dtype, and reused.
* ``_solo_broadcast_draft_hidden`` publishes from the solo rank, short-circuits
  for world_size==1 and for an empty round, and hands shadows back the
  received buffer.
* THE CORRECTNESS CONTRACT: running the EXISTING vocab-parallel greedy
  reduction (``_greedy_sample_from_vocab_parallel_head``) on every rank over
  its own lm_head shard yields, on EVERY rank, exactly the token ids a
  full-table argmax over the concatenated weight would produce — including
  the lowest-index tie-break. This is what makes dropping the gather safe.
* Collective SYMMETRY: host and shadows issue the same ordered sequence of
  collectives, which is what keeps the round from deadlocking. The lockstep
  fake group asserts the ordering matches across ranks.
* The eagle-side gather helper is still exported: NEXTN/EAGLE solo applies
  embed and lm_head INSIDE its captured draft graph (per-step
  embed->decode->lm_head->argmax feedback) and its shadows hold no draft body
  at all, so the broadcast route is not available there and NEXTN keeps the
  gather.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

DEVICE = "cpu"
HIDDEN = 32
SOLO_RANK = 0


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ShardIndices(SimpleNamespace):
    """Minimal stand-in for VocabParallelEmbeddingShardIndices."""


def make_shard_indices(org_start: int, num_org: int) -> ShardIndices:
    return ShardIndices(
        num_org_elements=num_org,
        num_org_elements_padded=num_org,
        num_added_elements=0,
        org_vocab_start_index=org_start,
        added_vocab_start_index=0,
    )


class FakeLmHead:
    def __init__(self, weight: torch.Tensor, shard_indices=None):
        self.weight = weight
        if shard_indices is not None:
            self.shard_indices = shard_indices


class LockstepTPGroup:
    """Fake TP GroupCoordinator that replays collectives across ranks.

    Used in two passes. Pass 1 runs every rank with ``recording=True`` and
    captures each rank's contribution per call index. Pass 2 re-runs a chosen
    rank with the recorded contributions available, so its ``all_gather``
    returns what the real collective would have returned.

    Because each rank's LOCAL computation never depends on a gather result,
    the two-pass replay is exact. The group also asserts that every rank
    issued the same NUMBER of calls in the same order — a mismatch is
    precisely the deadlock signature this design has to avoid.
    """

    def __init__(self, world_size: int, rank: int, log=None, recording=True):
        self.world_size = world_size
        self.rank_in_group = rank
        self.pynccl_comm = None
        self._rank = rank
        self._log = log if log is not None else {}
        self._recording = recording
        self._call_index = 0
        self.broadcast_calls = []

    def all_gather_into_tensor(self, out: torch.Tensor, inp: torch.Tensor):
        idx = self._call_index
        self._call_index += 1
        if self._recording:
            self._log.setdefault(idx, {})[self._rank] = inp.clone()
            out.zero_()
            return out
        per_rank = self._log[idx]
        assert len(per_rank) == self.world_size, (
            f"call {idx}: only ranks {sorted(per_rank)} participated; "
            "host and shadows must issue identical collective sequences"
        )
        stacked = torch.cat(
            [per_rank[r].reshape(-1) for r in range(self.world_size)], dim=0
        )
        out.copy_(stacked)
        return out

    def broadcast(self, tensor: torch.Tensor, src: int = 0):
        self.broadcast_calls.append((tuple(tensor.shape), src))
        return tensor


def make_worker(
    *,
    tp_group,
    solo_active=True,
    is_host=True,
    hidden_dim=HIDDEN,
    dtype=torch.float32,
):
    """A DFlashWorkerV2 with only the attributes the solo helpers touch."""
    worker = object.__new__(DFlashWorkerV2)
    worker.device = DEVICE
    worker.tp_rank = tp_group.rank_in_group
    worker._spec_solo_active = solo_active
    worker._spec_solo_is_host = is_host
    worker._spec_solo_rank = SOLO_RANK
    worker._solo_hidden_dim = hidden_dim
    worker._solo_hs_dtype = dtype
    worker._solo_hs_buf = None
    worker._solo_hs_cap = 0
    # Buffers _greedy_sample_from_vocab_parallel_head lazily fills.
    worker._draft_greedy_gathered_max_buf = None
    worker._draft_greedy_gathered_ids_buf = None
    worker._draft_greedy_gather_cap = 0
    worker._draft_greedy_local_max_buf = None
    worker._draft_greedy_local_arg_buf = None
    worker._draft_greedy_local_cap = 0
    worker._draft_greedy_best_rank_buf = None
    worker._draft_greedy_rank_index_buf = None
    worker._draft_greedy_selected_ids_buf = None
    worker._draft_greedy_index_cap = 0
    return worker


def split_vocab(full_weight: torch.Tensor, world_size: int):
    """Contiguous ascending vocab shards, deliberately UNEVEN (the fork's
    --rank-vocab-ratio produces uneven shards on mismatched GPUs)."""
    vocab = full_weight.shape[0]
    base = vocab // world_size
    sizes = [base] * world_size
    sizes[0] += vocab - base * world_size  # remainder onto rank 0
    shards, start = [], 0
    for size in sizes:
        shards.append((start, full_weight[start : start + size].clone()))
        start += size
    return shards


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSoloVocabBroadcastSetup(CustomTestCase):
    def _target_worker(self, weight):
        lm_head = FakeLmHead(weight, make_shard_indices(0, weight.shape[0]))
        model = SimpleNamespace(lm_head=lm_head)
        return SimpleNamespace(
            model_runner=SimpleNamespace(model=model),
            model_config=SimpleNamespace(vocab_size=weight.shape[0] * 3),
        )

    def test_setup_records_geometry_and_never_gathers(self):
        weight = torch.randn(64, HIDDEN, dtype=torch.bfloat16)
        worker = object.__new__(DFlashWorkerV2)
        worker._spec_solo_is_host = True
        worker._target_worker = self._target_worker(weight)
        worker.draft_model_runner = MagicMock()
        # DFlashWorkerV2.__getattr__ delegates unknown names to the target
        # worker, so pre-seed what we want to observe as untouched.
        worker._solo_hs_buf = None

        with patch(
            "sglang.srt.speculative.eagle_worker_v2._solo_gather_full_vocab_rows"
        ) as gather, patch(
            "sglang.srt.speculative.eagle_worker_v2.install_shadow_draft_runner_surface"
        ) as install:
            worker._solo_setup_vocab_broadcast()

        # The whole point: no vocab rows move anywhere.
        gather.assert_not_called()
        install.assert_not_called()  # host keeps its real draft runner
        self.assertEqual(worker._solo_hidden_dim, HIDDEN)
        self.assertEqual(worker._solo_hs_dtype, torch.bfloat16)
        self.assertIsNone(worker._solo_hs_buf)

    def test_shadow_gets_stub_runner_surface(self):
        weight = torch.randn(64, HIDDEN, dtype=torch.float16)
        worker = object.__new__(DFlashWorkerV2)
        worker._spec_solo_is_host = False
        worker._target_worker = self._target_worker(weight)
        worker.draft_model_runner = MagicMock()

        with patch(
            "sglang.srt.speculative.eagle_worker_v2.install_shadow_draft_runner_surface"
        ) as install:
            worker._solo_setup_vocab_broadcast()

        install.assert_called_once_with(worker.draft_model_runner)
        # Shadows learn the SAME geometry from their own shard -> they can
        # size the receive buffer without hearing from the host first.
        self.assertEqual(worker._solo_hidden_dim, HIDDEN)
        self.assertEqual(worker._solo_hs_dtype, torch.float16)

    def test_quantized_lm_head_is_rejected_loudly(self):
        weight = torch.zeros(64, HIDDEN, dtype=torch.int8)
        worker = object.__new__(DFlashWorkerV2)
        worker._spec_solo_is_host = True
        worker._target_worker = self._target_worker(weight)
        worker.draft_model_runner = MagicMock()
        with self.assertRaises(NotImplementedError):
            worker._solo_setup_vocab_broadcast()

    def test_missing_lm_head_is_rejected_loudly(self):
        worker = object.__new__(DFlashWorkerV2)
        worker._spec_solo_is_host = True
        worker._target_worker = SimpleNamespace(
            model_runner=SimpleNamespace(model=SimpleNamespace(lm_head=None)),
            model_config=SimpleNamespace(vocab_size=1),
        )
        worker.draft_model_runner = MagicMock()
        with self.assertRaises(NotImplementedError):
            worker._solo_setup_vocab_broadcast()


class TestSoloHiddenBroadcast(CustomTestCase):
    def test_buffer_is_grow_only_and_rank_uniform(self):
        group = LockstepTPGroup(world_size=3, rank=0)
        worker = make_worker(tp_group=group, dtype=torch.float16)

        small = worker._solo_hidden_broadcast_buf(4)
        self.assertEqual(tuple(small.shape), (4, HIDDEN))
        self.assertEqual(small.dtype, torch.float16)
        storage_after_small = worker._solo_hs_buf

        # Shrinking reuses the same allocation ...
        again = worker._solo_hidden_broadcast_buf(2)
        self.assertEqual(tuple(again.shape), (2, HIDDEN))
        self.assertIs(worker._solo_hs_buf, storage_after_small)

        # ... growing reallocates once, and stays grown.
        big = worker._solo_hidden_broadcast_buf(16)
        self.assertEqual(tuple(big.shape), (16, HIDDEN))
        self.assertEqual(worker._solo_hs_cap, 16)
        worker._solo_hidden_broadcast_buf(16)
        self.assertEqual(worker._solo_hs_cap, 16)

    def test_host_publishes_from_solo_rank_and_shadow_receives(self):
        payload = torch.randn(6, HIDDEN)

        host_group = LockstepTPGroup(world_size=3, rank=SOLO_RANK)
        host = make_worker(tp_group=host_group, is_host=True)
        with patch(
            "sglang.srt.speculative.dflash_worker_v2.get_tp_group",
            return_value=host_group,
        ), patch(
            "sglang.srt.speculative.dflash_worker_v2.capture_safe_tp_broadcast"
        ) as bcast:
            out = host._solo_broadcast_draft_hidden(6, payload)

        # Staged into the rank-uniform buffer, broadcast FROM the solo rank.
        self.assertEqual(out.data_ptr(), host._solo_hs_buf.data_ptr())
        self.assertEqual(tuple(out.shape), (6, HIDDEN))
        torch.testing.assert_close(out, payload)
        bcast.assert_called_once()
        self.assertEqual(bcast.call_args.kwargs["src"], SOLO_RANK)

        shadow_group = LockstepTPGroup(world_size=3, rank=2)
        shadow = make_worker(tp_group=shadow_group, is_host=False)
        with patch(
            "sglang.srt.speculative.dflash_worker_v2.get_tp_group",
            return_value=shadow_group,
        ), patch(
            "sglang.srt.speculative.dflash_worker_v2.capture_safe_tp_broadcast"
        ) as bcast:
            recv = shadow._solo_broadcast_draft_hidden(6, None)

        self.assertEqual(tuple(recv.shape), (6, HIDDEN))
        bcast.assert_called_once()
        self.assertEqual(bcast.call_args.kwargs["src"], SOLO_RANK)

    def test_short_circuits_are_symmetric(self):
        payload = torch.randn(5, HIDDEN)

        # world_size == 1: nothing to publish to.
        solo_group = LockstepTPGroup(world_size=1, rank=0)
        worker = make_worker(tp_group=solo_group)
        with patch(
            "sglang.srt.speculative.dflash_worker_v2.get_tp_group",
            return_value=solo_group,
        ), patch(
            "sglang.srt.speculative.dflash_worker_v2.capture_safe_tp_broadcast"
        ) as bcast:
            out = worker._solo_broadcast_draft_hidden(5, payload)
        bcast.assert_not_called()
        self.assertIs(out, payload)

        # Empty round: num_tokens is rank-uniform, so skipping stays symmetric.
        group = LockstepTPGroup(world_size=3, rank=0)
        worker = make_worker(tp_group=group)
        with patch(
            "sglang.srt.speculative.dflash_worker_v2.get_tp_group",
            return_value=group,
        ), patch(
            "sglang.srt.speculative.dflash_worker_v2.capture_safe_tp_broadcast"
        ) as bcast:
            worker._solo_broadcast_draft_hidden(0, payload[:0])
        bcast.assert_not_called()


class TestVocabParallelGreedyMatchesFullTable(CustomTestCase):
    """The reduction contract that lets the gather go away."""

    def _run_all_ranks(self, full_weight, hidden, world_size, chunk_size):
        shards = split_vocab(full_weight, world_size)
        log = {}

        # Pass 1: record every rank's local contribution.
        for rank in range(world_size):
            org_start, shard_w = shards[rank]
            group = LockstepTPGroup(world_size, rank, log=log, recording=True)
            worker = make_worker(tp_group=group, dtype=full_weight.dtype)
            lm_head = FakeLmHead(
                shard_w, make_shard_indices(org_start, shard_w.shape[0])
            )
            with patch(
                "sglang.srt.speculative.dflash_worker_v2.get_tp_group",
                return_value=group,
            ):
                worker._greedy_sample_from_vocab_parallel_head(
                    hidden_states=hidden, lm_head=lm_head, chunk_size=chunk_size
                )

        # Every rank must have issued the same number of collectives, in the
        # same order — otherwise the real round deadlocks.
        for idx, per_rank in log.items():
            self.assertEqual(
                sorted(per_rank), list(range(world_size)), f"call {idx}"
            )

        # Pass 2: replay with the real gathered values, per rank.
        results = []
        for rank in range(world_size):
            org_start, shard_w = shards[rank]
            group = LockstepTPGroup(world_size, rank, log=log, recording=False)
            worker = make_worker(tp_group=group, dtype=full_weight.dtype)
            lm_head = FakeLmHead(
                shard_w, make_shard_indices(org_start, shard_w.shape[0])
            )
            with patch(
                "sglang.srt.speculative.dflash_worker_v2.get_tp_group",
                return_value=group,
            ):
                results.append(
                    worker._greedy_sample_from_vocab_parallel_head(
                        hidden_states=hidden, lm_head=lm_head, chunk_size=chunk_size
                    )
                )
        return results

    def test_matches_full_table_argmax_on_every_rank(self):
        torch.manual_seed(1481)
        full_weight = torch.randn(151, HIDDEN, dtype=torch.float32)
        hidden = torch.randn(15, HIDDEN, dtype=torch.float32)

        expected = torch.argmax(hidden @ full_weight.T, dim=-1).to(torch.long)

        for world_size in (2, 3, 4):
            for chunk_size in (256, 4):
                with self.subTest(world_size=world_size, chunk_size=chunk_size):
                    results = self._run_all_ranks(
                        full_weight, hidden, world_size, chunk_size
                    )
                    for rank, got in enumerate(results):
                        torch.testing.assert_close(
                            got, expected, msg=f"rank {rank} disagrees"
                        )

    def test_ties_break_to_the_lowest_global_id_like_full_argmax(self):
        """Duplicated rows across shard boundaries: torch.argmax takes the
        lowest index, and the rank reduction must agree (lowest rank, and
        within a rank the lowest local index — shards are contiguous and
        ascending, so that IS the lowest global id)."""
        row = torch.randn(1, HIDDEN, dtype=torch.float32)
        # Same winning row placed in several shards.
        full_weight = torch.full((12, HIDDEN), -5.0, dtype=torch.float32)
        for pos in (1, 5, 9):
            full_weight[pos] = row
        hidden = row.repeat(3, 1)

        expected = torch.argmax(hidden @ full_weight.T, dim=-1).to(torch.long)
        self.assertTrue(bool((expected == 1).all()))

        results = self._run_all_ranks(full_weight, hidden, world_size=3, chunk_size=256)
        for rank, got in enumerate(results):
            torch.testing.assert_close(got, expected, msg=f"rank {rank} disagrees")


class TestNextnKeepsTheGather(CustomTestCase):
    def test_dflash_no_longer_imports_the_gather_helper(self):
        import sglang.srt.speculative.dflash_worker_v2 as mod

        src = open(mod.__file__).read()
        self.assertNotIn("_solo_gather_full_vocab_rows", src)
        # And the removed local-full-table sampler is gone with it.
        self.assertNotIn("_solo_full_lm_head_weight", src)

    def test_eagle_still_exports_the_gather_helper(self):
        # NEXTN/EAGLE solo applies embed + lm_head inside its CAPTURED draft
        # graph (per-step embed->decode->lm_head->argmax feedback), and its
        # shadows hold no draft body, so it cannot use the broadcast route.
        from sglang.srt.speculative.eagle_worker_v2 import (
            _solo_gather_full_vocab_rows,
        )

        self.assertTrue(callable(_solo_gather_full_vocab_rows))


if __name__ == "__main__":
    unittest.main()
