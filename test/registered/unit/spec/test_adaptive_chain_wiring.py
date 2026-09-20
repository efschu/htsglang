"""Wiring tests for the per-round adaptive chain length.

Two things are checked here that the pure-policy tests cannot:

* with ``SGLANG_SPEC_ADAPTIVE_CHAIN`` unset, nothing is allocated and no code
  path changes — the default run must stay byte-identical;
* when it is set, the probe/policy hand-off does what the hot path expects, and
  a not-yet-landed readout falls back instead of blocking.

Everything runs on CPU: ``SurvivalProbe`` and ``RoundCostProbe`` both have a
non-CUDA path precisely so this wiring is exercisable off-GPU.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.speculative.adaptive_chain import (
    AdaptiveChainPolicy,
    RoundCostProbe,
    SurvivalProbe,
)
from sglang.srt.speculative.eagle_worker_v2 import EagleDraftWorker, EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _draft_worker(topk=1, adaptive=True, num_steps=3, cfg=None):
    w = object.__new__(EagleDraftWorker)
    w.topk = topk
    w.device = "cpu"
    w.speculative_num_steps = num_steps
    w.survival_probe = None
    w.server_args = SimpleNamespace(
        speculative_adaptive=adaptive,
        speculative_adaptive_config=cfg,
        cuda_graph_config=None,
        max_running_requests=4,
    )
    return w


class TestProbeArming(unittest.TestCase):
    def test_disabled_by_default(self):
        w = _draft_worker()
        w._init_adaptive_chain_probe()
        self.assertIsNone(w.survival_probe)

    def test_armed_when_env_set(self):
        w = _draft_worker()
        with patch.dict("os.environ", {"SGLANG_SPEC_ADAPTIVE_CHAIN": "1"}):
            w._init_adaptive_chain_probe()
        self.assertIsNotNone(w.survival_probe)
        # Default config's candidate union is {0,1,3,7}; buffer must span the
        # longest chain the controller could ever activate.
        self.assertEqual(w.survival_probe.k_max, 7)
        self.assertEqual(w.survival_probe.buffer.shape, (4, 7))

    def test_not_armed_for_topk_above_one(self):
        w = _draft_worker(topk=4)
        with patch.dict("os.environ", {"SGLANG_SPEC_ADAPTIVE_CHAIN": "1"}):
            w._init_adaptive_chain_probe()
        self.assertIsNone(w.survival_probe)

    def test_not_armed_without_speculative_adaptive(self):
        # It selects between runtime states that only --speculative-adaptive builds.
        w = _draft_worker(adaptive=False)
        with patch.dict("os.environ", {"SGLANG_SPEC_ADAPTIVE_CHAIN": "1"}):
            w._init_adaptive_chain_probe()
        self.assertIsNone(w.survival_probe)


class TestSurvivalProbe(unittest.TestCase):
    def test_write_step_accumulates_the_product(self):
        p = SurvivalProbe(max_bs=2, k_max=3, device="cpu")
        p.write_step(0, torch.tensor([[0.5], [0.5]]), 2)
        p.write_step(1, torch.tensor([[0.4], [0.4]]), 2)
        p.write_step(2, torch.tensor([[0.5], [0.5]]), 2)
        self.assertAlmostEqual(p.buffer[0, 0].item(), 0.5, places=5)
        self.assertAlmostEqual(p.buffer[0, 1].item(), 0.2, places=5)
        self.assertAlmostEqual(p.buffer[0, 2].item(), 0.1, places=5)

    def test_readout_returns_batch_mean(self):
        p = SurvivalProbe(max_bs=2, k_max=2, device="cpu")
        p.write_step(0, torch.tensor([[1.0], [0.0]]), 2)
        p.start_readout(2, 2)
        curve = p.poll()
        self.assertEqual(len(curve), 2)
        self.assertAlmostEqual(curve[0], 0.5, places=5)

    def test_poll_without_readout_returns_none(self):
        p = SurvivalProbe(max_bs=2, k_max=2, device="cpu")
        self.assertIsNone(p.poll())

    def test_poll_is_single_shot(self):
        p = SurvivalProbe(max_bs=1, k_max=1, device="cpu")
        p.write_step(0, torch.tensor([[0.9]]), 1)
        p.start_readout(1, 1)
        self.assertIsNotNone(p.poll())
        self.assertIsNone(p.poll())

    def test_out_of_range_column_is_ignored(self):
        p = SurvivalProbe(max_bs=1, k_max=2, device="cpu")
        p.write_step(5, torch.tensor([[0.5]]), 1)
        p.write_step(-1, torch.tensor([[0.5]]), 1)
        self.assertEqual(p.buffer.sum().item(), 0.0)

    def test_batch_size_is_clamped_to_buffer(self):
        p = SurvivalProbe(max_bs=2, k_max=1, device="cpu")
        p.write_step(0, torch.tensor([[0.5], [0.5], [0.5], [0.5]]), 4)
        p.start_readout(9, 1)
        self.assertIsNotNone(p.poll())

    def test_zero_batch_readout_yields_nothing(self):
        p = SurvivalProbe(max_bs=2, k_max=2, device="cpu")
        p.start_readout(0, 2)
        self.assertIsNone(p.poll())


class TestRoundCostProbe(unittest.TestCase):
    def test_begin_end_drain_reports_once(self):
        seen = []
        p = RoundCostProbe(device="cpu")
        p.begin(3)
        p.end()
        self.assertEqual(p.drain(lambda k, ms: seen.append((k, ms))), 1)
        self.assertEqual(seen[0][0], 3)
        self.assertGreaterEqual(seen[0][1], 0.0)
        self.assertEqual(p.drain(lambda k, ms: seen.append((k, ms))), 1 - 1)

    def test_end_without_begin_is_a_noop(self):
        p = RoundCostProbe(device="cpu")
        p.end()
        self.assertEqual(p.drain(lambda k, ms: None), 0)

    def test_unclosed_round_is_dropped_not_mispaired(self):
        seen = []
        p = RoundCostProbe(device="cpu")
        p.begin(1)
        p.begin(2)  # previous round never closed
        p.end()
        p.drain(lambda k, ms: seen.append(k))
        self.assertEqual(seen, [2])

    def test_drain_feeds_the_policy_cost_model(self):
        policy = AdaptiveChainPolicy(k_max=3)
        p = RoundCostProbe(device="cpu")
        p.begin(2)
        p.end()
        self.assertEqual(p.drain(policy.record_duration), 1)


class TestArgmaxEquivalence(unittest.TestCase):
    """The armed path swaps argmax() for max()[1] to get the top logit for free.

    If the two disagreed on a tie, arming the probe would change which token the
    draft proposes — a behaviour change hiding inside an "instrumentation" flag.
    Both document "index of the first maximal value"; these pin it.
    """

    def _assert_same_index(self, scores):
        self.assertTrue(
            torch.equal(
                torch.argmax(scores, dim=-1, keepdim=True),
                scores.max(dim=-1, keepdim=True)[1],
            )
        )

    def test_same_index_on_random_scores(self):
        torch.manual_seed(0)
        self._assert_same_index(torch.randn(17, 257))

    def test_same_index_when_every_value_ties(self):
        self._assert_same_index(torch.zeros(4, 8))

    def test_same_index_with_duplicated_maxima(self):
        self._assert_same_index(
            torch.tensor([[1.0, 5.0, 5.0, 2.0], [7.0, 7.0, 0.0, 7.0]])
        )

    def test_confidence_matches_plain_softmax(self):
        # The probe computes exp(top1 - logsumexp) to avoid a second pass over
        # the vocabulary; it must equal the softmax top-1 probability.
        torch.manual_seed(1)
        scores = torch.randn(5, 64)
        top1, _ = scores.max(dim=-1, keepdim=True)
        via_lse = torch.exp(top1 - torch.logsumexp(scores, dim=-1, keepdim=True))
        via_softmax = torch.softmax(scores, dim=-1).max(dim=-1, keepdim=True)[0]
        self.assertTrue(torch.allclose(via_lse, via_softmax, atol=1e-6))


def _eagle_worker(chain_policy=None, probe=None, controller=None):
    w = object.__new__(EAGLEWorkerV2)
    w._draft_worker = SimpleNamespace(survival_probe=probe)
    w.chain_policy = chain_policy
    w.adaptive_controller = controller
    w.round_cost_probe = None
    return w


class _FakeController:
    def __init__(self):
        self.by_batch = []
        self.by_steps = []

    def activate_step_by_batch(self, bs):
        self.by_batch.append(bs)

    def activate_steps(self, k):
        self.by_steps.append(k)
        return True


class TestActivateStepByBatch(unittest.TestCase):
    def test_no_controller_is_inert(self):
        w = _eagle_worker()
        w.activate_step_by_batch(1)  # must not raise

    def test_without_chain_policy_the_ema_path_runs(self):
        """The default path: --speculative-adaptive alone behaves as before."""
        ctrl = _FakeController()
        w = _eagle_worker(controller=ctrl)
        w.activate_step_by_batch(4)
        self.assertEqual(ctrl.by_batch, [4])
        self.assertEqual(ctrl.by_steps, [])

    def test_chain_policy_takes_over_when_a_curve_landed(self):
        ctrl = _FakeController()
        probe = SurvivalProbe(max_bs=1, k_max=3, device="cpu")
        probe.write_step(0, torch.tensor([[0.0]]), 1)
        probe.start_readout(1, 3)
        policy = AdaptiveChainPolicy(k_max=3, candidates=[1, 3])
        w = _eagle_worker(chain_policy=policy, probe=probe, controller=ctrl)

        w.activate_step_by_batch(1)

        # Zero confidence -> shortest candidate, and the EMA path is skipped.
        self.assertEqual(ctrl.by_steps, [1])
        self.assertEqual(ctrl.by_batch, [])

    def test_falls_back_to_ema_when_readout_has_not_landed(self):
        ctrl = _FakeController()
        probe = SurvivalProbe(max_bs=1, k_max=3, device="cpu")  # no start_readout
        policy = AdaptiveChainPolicy(k_max=3, candidates=[1, 3])
        w = _eagle_worker(chain_policy=policy, probe=probe, controller=ctrl)

        w.activate_step_by_batch(2)

        self.assertEqual(ctrl.by_steps, [])
        self.assertEqual(ctrl.by_batch, [2])

    def test_policy_without_probe_falls_back(self):
        ctrl = _FakeController()
        policy = AdaptiveChainPolicy(k_max=3)
        w = _eagle_worker(chain_policy=policy, probe=None, controller=ctrl)
        w.activate_step_by_batch(1)
        self.assertEqual(ctrl.by_batch, [1])

    def test_round_cost_probe_is_drained_on_the_policy_path(self):
        ctrl = _FakeController()
        probe = SurvivalProbe(max_bs=1, k_max=2, device="cpu")
        probe.write_step(0, torch.tensor([[1.0]]), 1)
        probe.start_readout(1, 2)
        policy = AdaptiveChainPolicy(k_max=2, candidates=[1, 2])
        w = _eagle_worker(chain_policy=policy, probe=probe, controller=ctrl)
        w.round_cost_probe = RoundCostProbe(device="cpu")
        w.round_cost_probe.begin(2)
        w.round_cost_probe.end()

        w.activate_step_by_batch(1)

        self.assertEqual(policy.cost_model._counts[2], 1)


if __name__ == "__main__":
    unittest.main()
