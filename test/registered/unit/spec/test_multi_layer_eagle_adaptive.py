"""Adaptive draft length on multi-layer EAGLE (#138), CPU-only.

Multi-layer EAGLE runs ONE DRAFT MODEL PER CHAIN POSITION and does no model
forward at draft time: the chain columns consumed by round N+1 were produced at
the end of round N, at the k that was active then. The adaptive k switch lands
BETWEEN rounds, so the first round at a new k sees columns of the OLD width.

These tests pin the three parts of the retrofit that are testable without a GPU:
  1. config resolution   -- the multi-layer default has no step-0 rung, and the
                            algorithm key routes to it
  2. width adaptation    -- slice on downshift, repeat-last on upshift
  3. init-time guards    -- step-0 configs and a ladder ceiling above the draft
                            checkpoint's MTP layer count are hard errors
  4. group uniformity    -- the k decision is a pure function of rank-invariant
                            inputs, so replaying it per rank yields one k
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.adaptive_runtime_state import (
    SpecRuntimeState,
    assert_runtime_state_isolation,
)
from sglang.srt.speculative.adaptive_spec_params import (
    DEFAULT_ADAPTIVE_CONFIG,
    MULTI_LAYER_EAGLE_ALGO_KEY,
    MULTI_LAYER_EAGLE_DEFAULT_ADAPTIVE_CONFIG,
    AdaptiveSpeculativeParams,
    adaptive_algorithm_key,
    adaptive_unsupported_reason,
    default_adaptive_config_for,
    resolve_candidate_steps_from_config,
)
from sglang.srt.speculative.multi_layer_eagle_utils import (
    adapt_draft_columns,
    adapt_draft_state_width,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="base-a-test-cpu")


def _server_args(**kwargs):
    base = dict(
        speculative_algorithm="EAGLE",
        speculative_adaptive=True,
        speculative_adaptive_config=None,
        speculative_eagle_topk=1,
        speculative_num_steps=3,
        enable_multi_layer_eagle=True,
        enable_two_batch_overlap=False,
        enable_pdmux=False,
        enable_dp_attention=False,
    )
    base.update(kwargs)
    return SimpleNamespace(**base)


def _patch_resolved_view(server_args):
    """`resolved_view` needs the full override registry; the flags we read are
    plain attributes on the fake, so resolve to the fake itself."""
    return patch(
        "sglang.srt.arg_groups.overrides.resolved_view",
        side_effect=lambda sa: sa,
    )


class TestMultiLayerDefaultConfig(CustomTestCase):
    def test_algorithm_key_routes_multi_layer(self):
        sa = _server_args()
        with _patch_resolved_view(sa):
            self.assertEqual(adaptive_algorithm_key(sa), MULTI_LAYER_EAGLE_ALGO_KEY)

        sa_plain = _server_args(enable_multi_layer_eagle=False)
        with _patch_resolved_view(sa_plain):
            self.assertEqual(adaptive_algorithm_key(sa_plain), "EAGLE")

    def test_default_config_selection_by_key(self):
        self.assertIs(
            default_adaptive_config_for(MULTI_LAYER_EAGLE_ALGO_KEY),
            MULTI_LAYER_EAGLE_DEFAULT_ADAPTIVE_CONFIG,
        )
        # Plain EAGLE is untouched by the new key.
        self.assertIs(default_adaptive_config_for("EAGLE"), DEFAULT_ADAPTIVE_CONFIG)

    def test_multi_layer_default_has_no_step_below_one(self):
        steps = resolve_candidate_steps_from_config(
            algorithm=MULTI_LAYER_EAGLE_ALGO_KEY
        )
        self.assertEqual(steps, [1, 2, 3])
        # The generic EAGLE default DOES contain a step-0 rung, which is exactly
        # why multi-layer needs its own.
        self.assertIn(0, resolve_candidate_steps_from_config(algorithm="EAGLE"))

    def test_ceiling_matches_static_auto_shape(self):
        """The default ladder must not load more MTP layers than the default
        STATIC config does -- otherwise merely turning adaptive on would cost
        extra draft weights. _auto_choose_speculative_params picks steps=3 for
        the multi-layer architectures."""
        steps = resolve_candidate_steps_from_config(
            algorithm=MULTI_LAYER_EAGLE_ALGO_KEY
        )
        self.assertEqual(max(steps), 3)

    def test_gate_no_longer_rejects_multi_layer(self):
        sa = _server_args()
        with _patch_resolved_view(sa):
            self.assertIsNone(adaptive_unsupported_reason(sa))

    def test_gate_still_rejects_topk_gt_one_and_dp(self):
        sa = _server_args(speculative_eagle_topk=2)
        with _patch_resolved_view(sa):
            self.assertIn("speculative_eagle_topk", adaptive_unsupported_reason(sa))

        sa = _server_args(enable_dp_attention=True)
        with _patch_resolved_view(sa):
            self.assertIn("enable_dp_attention", adaptive_unsupported_reason(sa))


class TestAdaptDraftColumns(CustomTestCase):
    def test_width_match_is_identity_no_copy(self):
        t = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        self.assertIs(adapt_draft_columns(t, 3), t)

    def test_none_passthrough(self):
        self.assertIsNone(adapt_draft_columns(None, 3))

    def test_downshift_takes_the_prefix(self):
        t = torch.tensor([[10, 11, 12], [20, 21, 22]])
        out = adapt_draft_columns(t, 2)
        torch.testing.assert_close(out, torch.tensor([[10, 11], [20, 21]]))

    def test_upshift_repeats_the_last_column(self):
        t = torch.tensor([[10, 11], [20, 21]])
        out = adapt_draft_columns(t, 4)
        torch.testing.assert_close(
            out, torch.tensor([[10, 11, 11, 11], [20, 21, 21, 21]])
        )

    def test_upshift_on_three_dim_draft_probs(self):
        # draft_probs is [bs, num_steps, vocab]; the repeated step must carry a
        # full copy of the last step's distribution (Leviathan exactness).
        probs = torch.tensor([[[0.2, 0.8], [0.6, 0.4]]])
        out = adapt_draft_columns(probs, 3)
        self.assertEqual(tuple(out.shape), (1, 3, 2))
        torch.testing.assert_close(out[0, 2], torch.tensor([0.6, 0.4]))
        # ...and the padded rows are still valid distributions.
        torch.testing.assert_close(out.sum(dim=-1), torch.ones(1, 3))

    def test_zero_target_width_raises(self):
        with self.assertRaises(ValueError):
            adapt_draft_columns(torch.zeros(2, 3), 0)

    def test_zero_source_width_raises(self):
        with self.assertRaises(ValueError):
            adapt_draft_columns(torch.zeros(2, 0), 2)


class TestAdaptDraftStateWidth(CustomTestCase):
    def _draft_input(self, width, vocab=4, with_probs=False):
        return SimpleNamespace(
            topk_p=torch.rand(2, width),
            topk_index=torch.arange(2 * width).reshape(2, width),
            draft_probs=(
                torch.softmax(torch.rand(2, width, vocab), dim=-1)
                if with_probs
                else None
            ),
        )

    def test_noop_when_width_matches(self):
        di = self._draft_input(3)
        before = di.topk_p
        self.assertFalse(adapt_draft_state_width(di, speculative_num_steps=3, topk=1))
        self.assertIs(di.topk_p, before)

    def test_downshift_resizes_all_carried_tensors(self):
        di = self._draft_input(3, with_probs=True)
        self.assertTrue(adapt_draft_state_width(di, speculative_num_steps=1, topk=1))
        self.assertEqual(di.topk_p.shape[1], 1)
        self.assertEqual(di.topk_index.shape[1], 1)
        self.assertEqual(di.draft_probs.shape[1], 1)

    def test_upshift_resizes_all_carried_tensors(self):
        di = self._draft_input(1, with_probs=True)
        self.assertTrue(adapt_draft_state_width(di, speculative_num_steps=3, topk=1))
        self.assertEqual(di.topk_p.shape[1], 3)
        self.assertEqual(di.topk_index.shape[1], 3)
        self.assertEqual(di.draft_probs.shape[1], 3)

    def test_missing_draft_probs_stays_none(self):
        di = self._draft_input(2, with_probs=False)
        self.assertTrue(adapt_draft_state_width(di, speculative_num_steps=3, topk=1))
        self.assertIsNone(di.draft_probs)

    def test_absent_columns_is_a_noop(self):
        di = SimpleNamespace(topk_p=None, topk_index=None, draft_probs=None)
        self.assertFalse(adapt_draft_state_width(di, speculative_num_steps=3, topk=1))

    def test_result_width_always_equals_k_times_topk(self):
        for old in (1, 2, 3, 5):
            for new in (1, 2, 3, 5):
                di = self._draft_input(old)
                adapt_draft_state_width(di, speculative_num_steps=new, topk=1)
                self.assertEqual(di.topk_p.shape[1], new, f"{old}->{new}")
                self.assertEqual(di.topk_index.shape[1], new, f"{old}->{new}")


class TestPerStepBackendIsolation(CustomTestCase):
    """The M16/#50 guard must see the per-step draft-extend backends, not just
    the single-backend fields -- otherwise rungs could silently share them."""

    def _backend(self):
        return SimpleNamespace(init_forward_metadata=lambda *a, **k: None)

    def _state(self, steps, backend_list, target=None):
        return SpecRuntimeState(
            speculative_num_steps=steps,
            speculative_num_draft_tokens=steps + 1,
            draft_attn_backend=None,
            cuda_graph_runner=None,
            target_attn_backend=target or self._backend(),
            target_graph_runner=None,
            draft_extend_attn_backend=None,
            cuda_graph_runner_for_draft_extend=None,
            draft_extend_attn_backend_list=backend_list,
        )

    def test_distinct_per_step_backends_pass(self):
        states = {
            1: self._state(1, [self._backend()]),
            2: self._state(2, [self._backend(), self._backend()]),
        }
        assert_runtime_state_isolation(states)

    def test_shared_per_step_backend_across_rungs_raises(self):
        shared = self._backend()
        states = {
            1: self._state(1, [shared]),
            2: self._state(2, [shared, self._backend()]),
        }
        with self.assertRaises(RuntimeError) as ctx:
            assert_runtime_state_isolation(states)
        self.assertIn("draft_extend[0]", str(ctx.exception))

    def test_default_none_list_keeps_single_backend_workers_valid(self):
        # Construction without the new field must still work (EAGLE / frozen).
        state = SpecRuntimeState(
            speculative_num_steps=2,
            speculative_num_draft_tokens=3,
            draft_attn_backend=None,
            cuda_graph_runner=None,
            target_attn_backend=self._backend(),
            target_graph_runner=None,
            draft_extend_attn_backend=None,
            cuda_graph_runner_for_draft_extend=None,
        )
        self.assertIsNone(state.draft_extend_attn_backend_list)
        assert_runtime_state_isolation({2: state})


class TestInitGuards(CustomTestCase):
    """MultiLayerEagleWorkerV2._assert_adaptive_supported, run on a detached
    instance (a real __init__ needs a GPU and two loaded models)."""

    def _assert_supported(self, server_args):
        from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
            MultiLayerEagleWorkerV2,
        )

        worker = object.__new__(MultiLayerEagleWorkerV2)
        with _patch_resolved_view(server_args):
            MultiLayerEagleWorkerV2._assert_adaptive_supported(worker, server_args)

    def test_default_config_is_accepted(self):
        self._assert_supported(_server_args())

    def test_step_zero_config_is_rejected(self):
        sa = _server_args(speculative_adaptive_config=None)
        # Force the generic (step-0 bearing) EAGLE default through by pretending
        # multi-layer is off for the KEY only -- i.e. simulate a hand-written
        # config that contains a zero rung.
        import json
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"1": {"candidate_steps": [0, 1, 2]}}, f)
            path = f.name
        sa.speculative_adaptive_config = path
        with self.assertRaises(ValueError) as ctx:
            self._assert_supported(sa)
        self.assertIn("candidate steps >= 1", str(ctx.exception))

    def test_guard_runs_before_the_draft_worker_exists(self):
        """It must be callable without self._draft_worker: __init__ calls it
        BEFORE building the draft worker, because that constructor loads one
        MTP layer's weights per rung."""
        self._assert_supported(_server_args())

    def test_topk_gt_one_is_rejected(self):
        with self.assertRaises(ValueError):
            self._assert_supported(_server_args(speculative_eagle_topk=4))


class _FakeServerArgs(SimpleNamespace):
    def override(self, _tag, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _detached_worker(active_steps=2, num_runners=3):
    """A MultiLayerEagleWorkerV2 with only the attributes the state-swap and
    capture-override paths touch (a real __init__ needs two loaded models)."""
    from sglang.srt.speculative.multi_layer_eagle_worker_v2 import (
        MultiLayerEagleWorkerV2,
    )

    worker = object.__new__(MultiLayerEagleWorkerV2)
    worker.speculative_num_steps = active_steps
    worker.speculative_num_draft_tokens = active_steps + 1
    worker.server_args = _FakeServerArgs(
        speculative_num_steps=active_steps,
        speculative_num_draft_tokens=active_steps + 1,
        cuda_graph_bs_decode=[1, 8],
        disable_cuda_graph=False,
    )
    runners = [SimpleNamespace(attn_backend=f"base{i}") for i in range(num_runners)]
    worker._draft_worker = SimpleNamespace(
        speculative_num_steps=active_steps,
        speculative_num_draft_tokens=active_steps + 1,
        draft_runner_list=runners,
        draft_extend_attn_backend_list=[f"boot{i}" for i in range(active_steps)],
        cuda_graph_runner=None,
        cuda_graph_runner_for_draft_extend="boot_composite",
    )
    worker._target_worker = SimpleNamespace(
        model_runner=SimpleNamespace(
            attn_backend="boot_target", decode_cuda_graph_runner="boot_target_graph"
        )
    )
    return worker


class TestApplyRuntimeState(CustomTestCase):
    def _state(self, steps, backends, composite):
        return SpecRuntimeState(
            speculative_num_steps=steps,
            speculative_num_draft_tokens=steps + 1,
            draft_attn_backend=None,
            cuda_graph_runner=None,
            target_attn_backend=f"target{steps}",
            target_graph_runner=f"target_graph{steps}",
            draft_extend_attn_backend=None,
            cuda_graph_runner_for_draft_extend=composite,
            draft_extend_attn_backend_list=backends,
        )

    def test_swap_rebinds_every_per_step_backend(self):
        w = _detached_worker(active_steps=2, num_runners=3)
        w.apply_runtime_state(self._state(3, ["a0", "a1", "a2"], "c3"))

        dw = w._draft_worker
        self.assertEqual(w.speculative_num_steps, 3)
        self.assertEqual(w.speculative_num_draft_tokens, 4)
        self.assertEqual(dw.speculative_num_steps, 3)
        self.assertEqual(dw.draft_extend_attn_backend_list, ["a0", "a1", "a2"])
        self.assertEqual(dw.cuda_graph_runner_for_draft_extend, "c3")
        # The per-step draft forward reads draft_runner_list[step].attn_backend,
        # so the rebind must travel with the list.
        self.assertEqual(
            [r.attn_backend for r in dw.draft_runner_list], ["a0", "a1", "a2"]
        )
        self.assertEqual(w._target_worker.model_runner.attn_backend, "target3")
        self.assertEqual(
            w._target_worker.model_runner.decode_cuda_graph_runner, "target_graph3"
        )
        self.assertEqual(w.server_args.speculative_num_steps, 3)
        self.assertEqual(w.server_args.speculative_num_draft_tokens, 4)

    def test_downshift_leaves_runners_above_the_rung_untouched(self):
        w = _detached_worker(active_steps=3, num_runners=3)
        w._draft_worker.draft_extend_attn_backend_list = ["b0", "b1", "b2"]
        for r, b in zip(w._draft_worker.draft_runner_list, ["b0", "b1", "b2"]):
            r.attn_backend = b

        w.apply_runtime_state(self._state(1, ["a0"], "c1"))
        self.assertEqual(w.speculative_num_steps, 1)
        # Runner 0 follows the rung; 1 and 2 are simply not used at k=1.
        self.assertEqual(
            [r.attn_backend for r in w._draft_worker.draft_runner_list],
            ["a0", "b1", "b2"],
        )

    def test_same_k_is_a_noop(self):
        w = _detached_worker(active_steps=2)
        w.apply_runtime_state(self._state(2, ["x0", "x1"], "cx"))
        self.assertEqual(
            w._draft_worker.draft_extend_attn_backend_list, ["boot0", "boot1"]
        )

    def test_none_backend_list_does_not_strand_the_runners(self):
        w = _detached_worker(active_steps=2)
        w.apply_runtime_state(self._state(3, None, "c3"))
        self.assertEqual(
            w._draft_worker.draft_extend_attn_backend_list, ["boot0", "boot1"]
        )


class TestOverrideWorkerState(CustomTestCase):
    def test_capture_override_restores_everything(self):
        w = _detached_worker(active_steps=2, num_runners=3)
        dw = w._draft_worker
        before = (
            w.speculative_num_steps,
            dw.speculative_num_steps,
            list(dw.draft_extend_attn_backend_list),
            [r.attn_backend for r in dw.draft_runner_list],
            dw.cuda_graph_runner_for_draft_extend,
            w.server_args.speculative_num_steps,
            w.server_args.cuda_graph_bs_decode,
            w.server_args.disable_cuda_graph,
        )
        with w._override_worker_state(3, 4, cuda_graph_bs=[1]):
            # server_args is part of the contract: the per-step graph runner
            # reads the geometry off model_runner.server_args, not the worker.
            self.assertEqual(w.server_args.speculative_num_steps, 3)
            self.assertEqual(w.server_args.speculative_num_draft_tokens, 4)
            self.assertEqual(w.server_args.cuda_graph_bs_decode, [1])
            self.assertEqual(dw.speculative_num_steps, 3)
            dw.draft_extend_attn_backend_list = ["t0", "t1", "t2"]
            dw.draft_runner_list[0].attn_backend = "t0"
            dw.cuda_graph_runner_for_draft_extend = "tmp"

        after = (
            w.speculative_num_steps,
            dw.speculative_num_steps,
            list(dw.draft_extend_attn_backend_list),
            [r.attn_backend for r in dw.draft_runner_list],
            dw.cuda_graph_runner_for_draft_extend,
            w.server_args.speculative_num_steps,
            w.server_args.cuda_graph_bs_decode,
            w.server_args.disable_cuda_graph,
        )
        self.assertEqual(before, after)

    def test_empty_pruned_bs_disables_capture_for_that_rung(self):
        w = _detached_worker(active_steps=2)
        with w._override_worker_state(3, 4, cuda_graph_bs=[]):
            self.assertTrue(w.server_args.disable_cuda_graph)
        self.assertFalse(w.server_args.disable_cuda_graph)

    def test_restores_on_exception(self):
        w = _detached_worker(active_steps=2)
        with self.assertRaises(RuntimeError):
            with w._override_worker_state(3, 4):
                raise RuntimeError("capture blew up")
        self.assertEqual(w.speculative_num_steps, 2)
        self.assertEqual(w.server_args.speculative_num_steps, 2)


class TestDraftRunnerCount(CustomTestCase):
    """The MTP layer count is a WEIGHT budget fixed at load time: rung k needs
    layers 0..k-1 resident, so boot must load the ladder ceiling."""

    def _count(self, num_nextn_predict_layers=8, **kwargs):
        from sglang.srt.managers.tp_worker import TpModelWorker

        worker = object.__new__(TpModelWorker)
        worker.server_args = SimpleNamespace(**kwargs)
        worker.model_config = SimpleNamespace(
            hf_config=SimpleNamespace(
                num_nextn_predict_layers=num_nextn_predict_layers
            )
        )
        return TpModelWorker._num_multi_layer_eagle_draft_runners(worker)

    def test_static_path_uses_speculative_num_steps(self):
        self.assertEqual(
            self._count(speculative_num_steps=3, adaptive_max_candidate_steps=None), 3
        )

    def test_static_path_never_consults_the_checkpoint(self):
        """Default-path behaviour must be bit-for-bit unchanged: the layer-count
        guard is adaptive-only, so a static launch that over-asks still fails
        the way it always did (in the loader), not here."""
        self.assertEqual(
            self._count(
                speculative_num_steps=9,
                adaptive_max_candidate_steps=None,
                num_nextn_predict_layers=3,
            ),
            9,
        )

    def test_adaptive_path_uses_the_ladder_ceiling(self):
        self.assertEqual(
            self._count(speculative_num_steps=2, adaptive_max_candidate_steps=3), 3
        )

    def test_boot_k_above_the_ceiling_still_wins(self):
        self.assertEqual(
            self._count(speculative_num_steps=5, adaptive_max_candidate_steps=3), 5
        )

    def test_ladder_above_available_mtp_layers_is_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self._count(
                speculative_num_steps=2,
                adaptive_max_candidate_steps=3,
                num_nextn_predict_layers=2,
            )
        self.assertIn("num_nextn_predict_layers=2", str(ctx.exception))

    def test_unknown_mtp_layer_count_skips_the_guard(self):
        self.assertEqual(
            self._count(
                speculative_num_steps=2,
                adaptive_max_candidate_steps=3,
                num_nextn_predict_layers=None,
            ),
            3,
        )


class TestGroupUniformKDecision(CustomTestCase):
    """[[rank-lokaler-test-vor-kollektiv]]: a k switch is a GROUP decision. It
    is never broadcast -- it is a pure function of rank-invariant inputs
    (rank-0-broadcast accept counts + the batch size). Replaying the identical
    feed on N independent controller copies must therefore produce identical k
    on identical rounds; a divergence here would desync CUDA graphs and hang
    the group in NCCL."""

    def _params(self):
        return AdaptiveSpeculativeParams(
            initial_steps=3,
            algorithm=MULTI_LAYER_EAGLE_ALGO_KEY,
        )

    def test_replayed_feed_yields_identical_k_per_round(self):
        ranks = [self._params() for _ in range(4)]
        # Alternating high/low acceptance, enough rounds to cross the dwell.
        feed = []
        for i in range(600):
            counts = [3, 3] if (i // 100) % 2 == 0 else [0, 0]
            feed.append(counts)

        trajectories = []
        for params in ranks:
            traj = []
            for counts in feed:
                params.on_verify_complete(counts, batch_size=2)
                traj.append(params.get_steps_for_batch(2))
            trajectories.append(traj)

        for other in trajectories[1:]:
            self.assertEqual(trajectories[0], other)
        # The feed must actually have moved the ladder, else the test is vacuous.
        self.assertGreater(len(set(trajectories[0])), 1)

    def test_decision_never_reads_wall_clock(self):
        """RungMetrics.round_s is rank-local wall clock; the k decision must not
        depend on it. Feeding the same counts with wildly different timing must
        give the same k trajectory."""
        import time as _time

        fast = self._params()
        slow = self._params()
        traj_fast, traj_slow = [], []
        for i in range(400):
            counts = [3, 3] if i < 200 else [0, 0]
            fast.on_verify_complete(counts, batch_size=2)
            traj_fast.append(fast.get_steps_for_batch(2))
        with patch.object(_time, "monotonic", side_effect=lambda: 0.0):
            for i in range(400):
                counts = [3, 3] if i < 200 else [0, 0]
                slow.on_verify_complete(counts, batch_size=2)
                traj_slow.append(slow.get_steps_for_batch(2))
        self.assertEqual(traj_fast, traj_slow)


class TestGraphBucketSelection(CustomTestCase):
    """Only bucketed k values may be reached at runtime; a rung no BS slot can
    reach captures no graphs (the disable_cuda_graph branch of the override)."""

    def test_cuda_graph_bs_pruned_per_rung(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=3,
            algorithm=MULTI_LAYER_EAGLE_ALGO_KEY,
        )
        params.set_cuda_graph_bs([1, 8, 32])
        # bs32 slot only offers k=1, so k=3 is unreachable from bs 32.
        self.assertEqual(params.cuda_graph_bs_for_step(3), [1])
        self.assertEqual(params.cuda_graph_bs_for_step(2), [1, 8])
        self.assertEqual(params.cuda_graph_bs_for_step(1), [1, 8, 32])

    def test_every_reachable_k_is_a_candidate(self):
        params = AdaptiveSpeculativeParams(
            initial_steps=3,
            algorithm=MULTI_LAYER_EAGLE_ALGO_KEY,
        )
        for bs in (1, 4, 8, 16, 32, 64):
            self.assertIn(params.get_steps_for_batch(bs), params.candidate_steps)


if __name__ == "__main__":
    unittest.main()
