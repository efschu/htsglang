"""Multi-layer EAGLE decode draft-extend: the eager (non-graph) multi-step path.

`MultiLayerEagleDraftWorker._draft_extend_for_decode` runs ONE MTP RUNNER PER
CHAIN RUNG. Each rung's pick is rotated into the next rung's input_ids, so the
rungs form a chain in which anything wrong at rung i is carried into rung i+1's
forward -- and into the KV that forward writes.

Two pre-existing defects of that path are pinned here, both found while
building #138 (see docs_new/multi_layer_eagle_adaptive_len.md, open points
O1/O2):

* #185 (#50 family) -- the picks were NOT rank-0-broadcast, unlike every pick
  in eagle_worker_v2. On heterogeneous GPUs per-rank logits differ in the last
  bits, so a near-tie flips on one rank and the TP group desynchronizes.
* #184 -- `prepare_for_draft_extend` plans attention metadata for
  `draft_runner_list[0]`'s backend only, then the batch is marked
  metadata-ready, so rungs >= 1 used to run on whatever metadata their backend
  happened to hold (the standing "may have correctness issue" warning). The
  per-rung graph path already re-plans per rung in
  `MultiLayerEagleDraftExtendCudaGraphRunner.replay`; the eager path now does
  the same.

Everything here runs on CPU with fake runners/backends: these tests pin
STRUCTURE (who is synced, in what order things are planned), not the numerics
of the MTP layers, which need a GPU and a multi-layer checkpoint.
"""

import ast
import contextlib
import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.eagle_info import EagleDraftExtendInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

BS = 2
STEPS = 3
VOCAB = 16
WINDOW = STEPS + 1  # speculative_num_draft_tokens


class _FakeBackend:
    """Per-rung attention backend; records when it is planned."""

    def __init__(self, step, trace):
        self.step = step
        self.trace = trace
        self.plans = 0

    def init_forward_metadata(self, forward_batch):
        self.plans += 1
        self.trace.append(("plan", self.step))


class _FakeRunner:
    """One MTP rung. Its logits depend on the CURRENT chain input_ids (so a
    divergent pick really propagates to the next rung) plus a per-rank
    perturbation on a near-tie -- the heterogeneous-GPU situation from #50."""

    def __init__(self, step, rank_eps, trace):
        self.step = step
        self.rank_eps = rank_eps
        self.trace = trace
        self.attn_backend = _FakeBackend(step, trace)

    def forward(self, forward_batch):
        self.trace.append(("forward", self.step))
        seed = int(forward_batch.input_ids.sum().item()) + self.step
        a = seed % VOCAB
        b = (a + 1) % VOCAB
        logits = torch.zeros(BS * WINDOW, VOCAB)
        logits[:, a] = 1.0
        logits[:, b] = 1.0 + self.rank_eps
        return SimpleNamespace(
            logits_output=SimpleNamespace(next_token_logits=logits, hidden_states=None)
        )


def _fake_rotate(input_ids, extend_start_loc, extend_seq_lens, new_ids, select_index=None):
    """Stand-in for the triton rotate_input_ids: shift each request's window
    left by one and append that request's freshly picked token."""
    flat = new_ids.reshape(-1)
    for i in range(BS):
        window = input_ids[i * WINDOW : (i + 1) * WINDOW]
        window[:-1] = window[1:].clone()
        window[-1] = flat[i]


def _make_worker(rank_eps, trace):
    from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
        MultiLayerEagleDraftWorker,
    )

    w = object.__new__(MultiLayerEagleDraftWorker)
    w.device = "cpu"
    w.topk = 1
    w.speculative_num_steps = STEPS
    w.speculative_num_draft_tokens = WINDOW
    w.use_rejection_sampling = False
    w.chain_mtp_hidden_states = False
    w.plan_stream = None
    w.plan_stream_ctx = contextlib.nullcontext()
    # Eager path: no draft-extend graph runner at all.
    w.cuda_graph_runner_for_draft_extend = None
    w.draft_runner_list = [_FakeRunner(s, rank_eps, trace) for s in range(STEPS)]
    w.draft_extend_attn_backend_list = [r.attn_backend for r in w.draft_runner_list]

    def _fake_prepare(
        draft_extend_input,
        batch,
        next_token_ids,
        num_draft_tokens,
        draft_model_runner,
        cuda_graph_runner,
    ):
        # Faithful stand-in for BaseSpecWorker.prepare_for_draft_extend's
        # metadata handling: it plans exactly the PASSED runner's backend
        # (draft_runner_list[0]) and then marks the batch ready, which makes
        # every subsequent forward skip its own metadata init.
        fb = SimpleNamespace(
            input_ids=torch.arange(BS * WINDOW, dtype=torch.int64) % VOCAB,
            extend_start_loc=torch.arange(BS, dtype=torch.int64) * WINDOW,
            extend_seq_lens=torch.full((BS,), WINDOW, dtype=torch.int64),
            forward_mode=SimpleNamespace(is_idle=lambda: False),
            spec_info=draft_extend_input,
            sampling_info=SimpleNamespace(temperatures=torch.ones(BS, 1)),
            return_hidden_states_before_norm=False,
        )
        fb.mark_forward_metadata_ready = lambda: trace.append(("mark_ready", 0))
        draft_model_runner.attn_backend.init_forward_metadata(fb)
        fb.mark_forward_metadata_ready()
        return fb

    w.prepare_for_draft_extend = _fake_prepare
    return w


def _make_batch_and_result():
    batch = SimpleNamespace(seq_lens=torch.full((BS,), 32, dtype=torch.int64))
    batch_result = SimpleNamespace(
        logits_output=SimpleNamespace(hidden_states=None),
        next_token_ids=torch.tensor([3, 4], dtype=torch.int64),
        accept_lens=torch.tensor([2, 1], dtype=torch.int64),
        next_draft_input=SimpleNamespace(
            topk_p=None, topk_index=None, hidden_states=None, draft_probs=None
        ),
    )
    return batch, batch_result


def _run_rank(rank_eps, broadcast_impl, trace):
    """Drive one rank through the eager decode draft-extend."""
    worker = _make_worker(rank_eps, trace)
    batch, batch_result = _make_batch_and_result()
    rotations = []

    def _recording_rotate(*args, **kwargs):
        rotations.append(args[3].clone())
        _fake_rotate(*args, **kwargs)

    fake_group = SimpleNamespace(world_size=2, broadcast=broadcast_impl)
    with (
        patch(
            "sglang.srt.speculative.multi_layer_eagle_worker_v2.rotate_input_ids",
            _recording_rotate,
        ),
        patch("sglang.srt.distributed.get_tp_group", return_value=fake_group),
        patch(
            "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
            return_value=False,
        ),
    ):
        worker._draft_extend_for_decode(batch, batch_result)
    return batch_result.next_draft_input, rotations, worker


class TestDecodeDraftExtendRankSync(CustomTestCase):
    """#185: two ranks whose per-rung logits differ only by a tie-breaking
    epsilon must end the round with identical chain state."""

    def _two_rank_run(self):
        payloads = []

        def rank0_broadcast(t, src):
            # Rank 0 IS the source: the broadcast leaves its tensors alone,
            # it only publishes them.
            payloads.append(t.clone())

        r0_out, r0_rot, _ = _run_rank(-1e-6, rank0_broadcast, [])

        replay = iter(payloads)

        def rank1_broadcast(t, src):
            t.copy_(next(replay))

        r1_out, r1_rot, _ = _run_rank(+1e-6, rank1_broadcast, [])
        return (r0_out, r0_rot), (r1_out, r1_rot), payloads

    def test_ranks_diverge_without_the_sync(self):
        """Guard on the fixture itself: with the broadcast disabled the two
        ranks really do pick different chains, otherwise the test below would
        pass vacuously."""
        noop = lambda t, src: None
        r0_out, r0_rot, _ = _run_rank(-1e-6, noop, [])
        r1_out, r1_rot, _ = _run_rank(+1e-6, noop, [])
        self.assertFalse(torch.equal(r0_out.topk_index, r1_out.topk_index))

    def test_every_rung_is_broadcast_and_ranks_converge(self):
        (r0_out, r0_rot), (r1_out, r1_rot), payloads = self._two_rank_run()

        # One sync per rung (topk_index + topk_p, draft_probs is None here).
        self.assertEqual(len(payloads), 2 * STEPS, f"payloads={len(payloads)}")

        self.assertTrue(
            torch.equal(r0_out.topk_index, r1_out.topk_index),
            "rank 1 ended the round on a different draft chain than rank 0",
        )
        self.assertTrue(torch.equal(r0_out.topk_p, r1_out.topk_p))

        # The chain rotation feeds the NEXT rung's forward, so it must already
        # carry rank 0's picks -- syncing only the returned tensors would leave
        # the intermediate KV writes divergent.
        # The eager branch rotates after every rung (no `step < k-1` guard,
        # unlike the graph branch), so there are k rotations.
        self.assertEqual(len(r0_rot), STEPS)
        self.assertEqual(len(r1_rot), STEPS)
        for i, (a, b) in enumerate(zip(r0_rot, r1_rot)):
            self.assertTrue(torch.equal(a, b), f"rotation {i} diverged")

    def test_single_rank_path_takes_no_collective(self):
        """world_size == 1 must not touch the group at all (byte-identical to
        the pre-fix behavior on a non-TP run)."""

        def boom(t, src):
            raise AssertionError("no collective may run at world_size 1")

        worker = _make_worker(0.0, [])
        batch, batch_result = _make_batch_and_result()
        fake_group = SimpleNamespace(world_size=1, broadcast=boom)
        with (
            patch(
                "sglang.srt.speculative.multi_layer_eagle_worker_v2.rotate_input_ids",
                _fake_rotate,
            ),
            patch("sglang.srt.distributed.get_tp_group", return_value=fake_group),
            patch(
                "sglang.srt.layers.dp_attention.is_dp_attention_enabled",
                return_value=False,
            ),
        ):
            worker._draft_extend_for_decode(batch, batch_result)
        self.assertEqual(batch_result.next_draft_input.topk_index.shape, (BS, STEPS))


class TestDecodeDraftExtendPerRungMetadata(CustomTestCase):
    """#184: every rung must have its attention metadata planned before its
    forward, mirroring the per-rung plan in the graph path's replay()."""

    def test_each_rung_is_planned_before_its_forward(self):
        trace = []
        _run_rank(0.0, lambda t, src: None, trace)

        plans_forwards = [e for e in trace if e[0] in ("plan", "forward")]
        self.assertEqual(
            plans_forwards,
            [x for s in range(STEPS) for x in (("plan", s), ("forward", s))],
            "eager multi-step draft extend did not plan every rung's attention "
            "metadata immediately before that rung's forward (#184)",
        )

    def test_graph_path_plans_per_rung(self):
        """The template: the composite graph runner re-plans the rung's
        backend inside replay(). Pinned so the eager fix cannot drift away
        from the path that is known to work."""
        import sglang.srt.speculative.multi_layer_eagle_draft_extend_cuda_graph_runner as gr

        tree = ast.parse(inspect.getsource(gr))
        found = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name != "replay":
                    continue
                for sub in ast.walk(node):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Attribute)
                        and sub.func.attr.startswith("init_forward_metadata")
                    ):
                        found.append(sub.func.attr)
        self.assertTrue(
            found,
            "graph replay() no longer plans its rung's attention metadata — "
            "the eager fix in _draft_extend_for_decode was modelled on it",
        )

    def test_eager_loop_plans_the_rung_backend_by_step(self):
        """Ratchet: the plan call must index draft_extend_attn_backend_list
        with the loop's step, inside the rung loop."""
        import sglang.srt.speculative.multi_layer_eagle_worker_v2 as ml

        tree = ast.parse(inspect.getsource(ml))
        hits = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "init_forward_metadata"
            ):
                continue
            target = node.func.value
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and target.value.attr == "draft_extend_attn_backend_list"
            ):
                hits.append(node.lineno)
        self.assertTrue(
            hits,
            "no per-rung draft_extend_attn_backend_list[...].init_forward_metadata "
            "call in the multi-layer worker — rungs >= 1 would run the eager "
            "draft extend on stale attention metadata (#184)",
        )


if __name__ == "__main__":
    unittest.main()
