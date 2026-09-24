"""fnFL2 H27: the per-round adaptive chain policy in the Next-Flash flip form.

Form A after the flip: TP0 (5090) is the attention host AND the solo MTP draft
host; TP1/TP2 (3080) are expert workers and draft SHADOWS. Every round the host
broadcasts ``(bs, k)`` draft tokens and every shadow receives into a buffer it
sizes from ITS OWN ``speculative_num_steps`` (eagle_worker_v2
``_solo_recv_draft_tokens``), and all three replay the verify graph for
``bs * (k + 1)`` rows. So k must be identical on all three ranks on every
round. These tests drive three policies (one decider, two shadows with
deliberately different local evidence) through a fake broadcast and check:

* k uniform on every round, including the switch lines and the fallback;
* the dwell: switches only on decision rounds, one consensus per decision;
* the throughput choice converges on the measured best length;
* the fallback to the fixed length (k=3) is collective and sticky;
* the controller's EMA no longer switches while the chain policy owns k;
* at most one tagged ladder state mapped (``reserve max(one state)``);
* the Weg-2 sleep park unmaps the ladder after activating the baseline.

All CPU, no torch.distributed: the broadcast is a fake bus with the same
semantics (one source, every rank receives the source's value).
"""

import logging
import math
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.speculative import adaptive_graph_memory as agm
from sglang.srt.speculative.adaptive_chain import (
    FALLBACK_SENTINEL,
    AdaptiveChainPolicy,
    ChainConsensusError,
    ChainCostModel,
)
from sglang.srt.speculative.adaptive_graph_memory import plan_residency
from sglang.srt.speculative.adaptive_runtime_state import (
    AdaptiveController,
    SpecRuntimeState,
)
from sglang.srt.speculative.adaptive_spec_params import (
    resolve_candidate_steps_from_config,
)
from sglang.srt.speculative.eagle_worker_v2 import EAGLEWorkerV2
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

LOGGER = "sglang.srt.speculative.adaptive_chain"


class _Bus:
    """One broadcast per decision round from the decider (src), in rank order.

    ``post`` is the src's side (it sends its own proposal and returns it,
    exactly what ``dist.broadcast_object_list`` leaves in the src's box);
    ``take`` is every other rank's side. A take without a matching post is a
    rank that entered a decision round alone -- the fn8s4 hang -- and fails.
    """

    def __init__(self):
        self.value = None
        self.posts = 0
        self.takes = 0
        self._taken = {}

    def post(self, proposal):
        self.value = int(proposal)
        self.posts += 1
        return self.value

    def taker(self, rank):
        self._taken[rank] = 0

        def take(_local):
            if self._taken[rank] >= self.posts:
                raise AssertionError(
                    f"rank {rank} entered a decision round the src did not"
                )
            self._taken[rank] += 1
            self.takes += 1
            return self.value

        return take


def _three_ranks(min_dwell=4, regret_pct=0.0, census_every=0, margin=0.0):
    bus = _Bus()
    policies = []
    for rank in range(3):
        policies.append(
            AdaptiveChainPolicy(
                k_max=3,
                candidates=[1, 2, 3],
                cost_model=ChainCostModel(k_max=3, min_samples=4),
                min_dwell=min_dwell,
                consensus=bus.post if rank == 0 else bus.taker(rank),
                switch_margin=margin,
                fallback_k=3,
                initial_k=3,
                census_every=census_every,
                decider=rank == 0,
                rank=rank,
                regret_pct=regret_pct,
            )
        )
    return bus, policies


#: Measured-looking round costs (ms) of the Form A host: verify + k drafts.
_COST = {1: 24.0, 2: 26.5, 3: 29.0}
#: Draft top-1 survival of a prose-like stream with a steep tail: k=2 is the
#: throughput optimum at these costs (E/C in tok/ms: k1 1.80/24.0=.0750,
#: k2 2.35/26.5=.0887, k3 2.47/29.0=.0852).
_SURVIVAL_K2 = [0.80, 0.55, 0.12]


def _round(policies, i, survival_host, cost=_COST, accept=None):
    """One lockstep round on three ranks: choose, 'run' k, feed measurements."""
    ks = []
    for rank, p in enumerate(policies):
        k = p.choose()
        ks.append(k)
    # Solo draft-token broadcast: host sends (bs, k_host), shadow receives
    # into (bs, k_shadow). Equal shapes or the collective misreads.
    host_payload = torch.zeros((1, ks[0]), dtype=torch.int64)
    for k in ks[1:]:
        shadow_buf = torch.empty((1, k), dtype=torch.int64)
        assert shadow_buf.shape == host_payload.shape, (i, ks)
    k = ks[0]
    for rank, p in enumerate(policies):
        if rank == 0:
            p.record_survival(survival_host)  # only the solo host drafts
        # Each rank times its own round (rank-local CUDA events): skew them.
        p.record_duration(k, cost[k] * (1.0 + 0.03 * rank) + 0.1 * (i % 3))
        acc = accept(k) if accept is not None else 1.0 + sum(survival_host[:k])
        p.record_accept(k, acc)  # rank-invariant (broadcast accept counts)
    return ks


class TestKUniformOverThreeRanks(unittest.TestCase):
    def test_every_round_every_rank_same_k_and_same_switch_lines(self):
        bus, policies = _three_ranks(min_dwell=4)
        with self.assertLogs(LOGGER, logging.INFO) as cm:
            for i in range(200):
                ks = _round(policies, i, _SURVIVAL_K2)
                self.assertEqual(len(set(ks)), 1, f"round {i}: ranks split {ks}")
        switch_lines = [r for r in cm.output if "SPEC-ADAPT k=" in r]
        by_rank = {r: [] for r in range(3)}
        for line in switch_lines:
            rank = int(line.rsplit("rank=", 1)[1])
            body = line.split("SPEC-ADAPT ", 1)[1]
            transition = body.split(" reason=")[0]
            rnd = int(body.split(" round=")[1].split()[0])
            by_rank[rank].append((transition, rnd))
        self.assertTrue(by_rank[0], "no switch was ever taken")
        self.assertEqual(by_rank[0], by_rank[1])
        self.assertEqual(by_rank[0], by_rank[2])
        # One post per decision round, taken by both shadows.
        self.assertEqual(bus.takes, 2 * bus.posts)

    def test_shadows_with_empty_survival_follow_the_host(self):
        """A shadow's survival probe is never written (it does not draft);
        left to itself it would always propose k_max. The host decides."""
        bus, policies = _three_ranks(min_dwell=2)
        for i in range(120):
            _round(policies, i, [0.05, 0.0, 0.0])  # host: drafts useless
        self.assertEqual(policies[0].current, 1)
        self.assertEqual(policies[1].current, 1)
        self.assertEqual(policies[2].current, 1)
        self.assertGreater(policies[1].switch_stats["consensus_overrides"], 0)

    def test_switch_line_names_reason_accept_and_rate(self):
        _bus, policies = _three_ranks(min_dwell=4)
        with self.assertLogs(LOGGER, logging.INFO) as cm:
            for i in range(200):
                _round(policies, i, _SURVIVAL_K2)
        host = [r for r in cm.output if "SPEC-ADAPT k=" in r and r.endswith("rank=0")]
        reasons = {r.split("reason=")[1].split()[0] for r in host}
        self.assertIn("dwell", reasons)  # warm-up blocks time each length
        self.assertIn("throughput", reasons)  # then the measured argmax
        last = host[-1]
        self.assertRegex(last, r"accept_ema=[0-9.na]+->[0-9.na]+")
        self.assertRegex(last, r"tok_s_ema=[0-9.na]+->[0-9.na]+")


class TestDwell(unittest.TestCase):
    def test_decisions_only_every_dwell_rounds(self):
        bus, policies = _three_ranks(min_dwell=8)
        for i in range(160):
            _round(policies, i, _SURVIVAL_K2)
        self.assertEqual(bus.posts, 160 // 8)

    def test_k_is_held_for_at_least_the_dwell(self):
        _bus, policies = _three_ranks(min_dwell=6)
        seq = [_round(policies, i, _SURVIVAL_K2)[0] for i in range(240)]
        run = 1
        runs = []
        for a, b in zip(seq, seq[1:]):
            if a == b:
                run += 1
            else:
                runs.append(run)
                run = 1
        self.assertTrue(runs)
        self.assertGreaterEqual(min(runs), 6)


class TestThroughputChoice(unittest.TestCase):
    def test_converges_on_the_measured_best_length(self):
        _bus, policies = _three_ranks(min_dwell=4)
        seq = [_round(policies, i, _SURVIVAL_K2)[0] for i in range(300)]
        tail = seq[-100:]
        self.assertGreater(tail.count(2), 90, tail)
        host = policies[0]
        # The census names what the choice was made on.
        line = host.census_line()
        self.assertIn("tok_s_ema={1:", line)
        self.assertIn("accept_ema={1:", line)
        self.assertFalse(host.warming_up)

    def test_code_like_stream_keeps_k3(self):
        """High survival (code): k=3 is best and the policy settles there."""
        _bus, policies = _three_ranks(min_dwell=4)
        seq = [_round(policies, i, [0.95, 0.9, 0.85])[0] for i in range(300)]
        self.assertGreater(seq[-100:].count(3), 90)


class TestFallbackToFixedK3(unittest.TestCase):
    def test_regret_guard_falls_back_on_every_rank_the_same_round(self):
        """The survival (draft confidence) says k=1, but the MEASURED accept
        (verify) makes k=3 far better -- a miscalibrated drafter. The regret
        guard compares realized tok/s against k=3's own measured rate and
        pulls every rank to the fixed length, together, for good."""
        bus, policies = _three_ranks(min_dwell=4, regret_pct=5.0, census_every=20)

        def accept(k):
            return {1: 1.3, 2: 2.2, 3: 3.4}[k]

        rounds_seen = []
        entered = {}
        for i in range(400):
            ks = _round(policies, i, [0.30, 0.02, 0.0], accept=accept)
            self.assertEqual(len(set(ks)), 1)
            rounds_seen.append(ks[0])
            for p in policies:
                if p.in_fallback and p.rank not in entered:
                    entered[p.rank] = i
        for p in policies:
            self.assertTrue(p.in_fallback, p.rank)
            self.assertEqual(p.current, 3)
        self.assertEqual(sorted(entered), [0, 1, 2])
        self.assertEqual(len(set(entered.values())), 1, entered)
        # Before the fallback the (miscalibrated) survival had it on k=1.
        self.assertIn(1, rounds_seen[: min(entered.values())])
        # Sticky and collective-free after entry.
        posts = bus.posts
        for i in range(50):
            _round(policies, 400 + i, [0.30, 0.02, 0.0], accept=accept)
        self.assertEqual(bus.posts, posts)
        self.assertTrue(all(k == 3 for k in rounds_seen[-50:]))

    def test_sentinel_from_the_decider_is_honoured_everywhere(self):
        bus, policies = _three_ranks(min_dwell=2)
        for i in range(10):
            _round(policies, i, _SURVIVAL_K2)
        policies[1].request_fallback("not-the-decider")  # no-op on a shadow
        policies[0].request_fallback("operator")
        for i in range(10, 16):
            ks = _round(policies, i, _SURVIVAL_K2)
            self.assertEqual(len(set(ks)), 1)
        for p in policies:
            self.assertTrue(p.in_fallback)
            self.assertEqual(p.current, 3)
        # The reason travels only to the decider's own census; the shadows
        # know that the decider asked, which is all they need.
        self.assertIn("fallback=operator", policies[0].census_line())
        self.assertIn("fallback=decider", policies[2].census_line())

    def test_not_armed_means_fixed_k(self):
        """Nothing to choose between -> the worker never arms the policy and
        the boot's static k=3 state is all that runs."""

        class _Ctrl:
            built_steps = [3]

        w = object.__new__(EAGLEWorkerV2)
        w._draft_worker = SimpleNamespace(survival_probe=object())
        w.adaptive_controller = _Ctrl()
        w.chain_policy = None
        w._arm_chain_policy()
        self.assertIsNone(w.chain_policy)
        self.assertIsNone(w._draft_worker.survival_probe)

    def test_fallback_k_must_be_a_built_candidate(self):
        p = AdaptiveChainPolicy(k_max=3, candidates=[1, 2], fallback_k=3)
        self.assertEqual(p.fallback_k, 2)


class TestConsensusSource(unittest.TestCase):
    def _worker(self, solo_active, solo_rank, tp_rank=0):
        w = object.__new__(EAGLEWorkerV2)
        w._spec_solo_active = solo_active
        w._spec_solo_rank = solo_rank
        w.tp_rank = tp_rank
        return w

    @contextmanager
    def _fake_dist(self, seen, value=None, fail=False):
        import torch.distributed as dist

        def bcast(box, src, group):
            seen.append(src)
            if fail:
                raise RuntimeError("gloo broken")
            if value is not None:
                box[0] = value

        with (
            mock.patch.object(dist, "is_initialized", lambda: True),
            mock.patch.object(dist, "get_world_size", lambda g=None: 3),
            mock.patch.object(dist, "get_global_rank", lambda g, r: 10 + r),
            mock.patch.object(dist, "broadcast_object_list", bcast),
            mock.patch(
                "sglang.srt.distributed.get_tp_group",
                lambda: SimpleNamespace(cpu_group=object()),
            ),
        ):
            yield

    def test_src_is_the_solo_draft_host(self):
        seen = []
        w = self._worker(solo_active=True, solo_rank=2)
        with self._fake_dist(seen, value=1):
            self.assertEqual(w._agree_chain_length(3), 1)
        self.assertEqual(seen, [12])

    def test_src_is_rank0_without_solo(self):
        seen = []
        w = self._worker(solo_active=False, solo_rank=0)
        with self._fake_dist(seen):
            self.assertEqual(w._agree_chain_length(2), 2)
        self.assertEqual(seen, [10])

    def test_broadcast_failure_is_crash_stop_not_keep_local(self):
        seen = []
        w = self._worker(solo_active=True, solo_rank=0)
        policy = AdaptiveChainPolicy(
            k_max=3, candidates=[1, 3], min_dwell=0, consensus=w._agree_chain_length
        )
        with self._fake_dist(seen, fail=True):
            with self.assertRaises(ChainConsensusError):
                policy.choose()

    def test_sentinel_value_is_not_a_candidate(self):
        self.assertNotIn(FALLBACK_SENTINEL, AdaptiveChainPolicy(k_max=3).candidates)


class _FakeWorker:
    speculative_num_steps = 3

    def __init__(self):
        self.applied = []

    def build_adaptive_runtime_state(
        self, speculative_num_steps, speculative_num_draft_tokens, cuda_graph_bs=None
    ):
        return _state(speculative_num_steps)

    def apply_runtime_state(self, state):
        self.applied.append(state.speculative_num_steps)
        self.speculative_num_steps = state.speculative_num_steps


def _state(steps):
    return SpecRuntimeState(
        speculative_num_steps=steps,
        speculative_num_draft_tokens=steps + 1,
        draft_attn_backend=SimpleNamespace(),
        cuda_graph_runner=None,
        target_attn_backend=SimpleNamespace(),
        target_graph_runner=None,
        draft_extend_attn_backend=None,
        cuda_graph_runner_for_draft_extend=None,
    )


def _controller():
    worker = _FakeWorker()
    ctrl = AdaptiveController(worker, config_path="chain-1-3")
    ctrl.register(_state(3))
    for k in (1, 2):
        ctrl._states[k] = _state(k)
    return worker, ctrl


class TestEmaDoesNotFightThePolicy(unittest.TestCase):
    def _controller(self):
        return _controller()

    def test_allow_switch_false_observes_but_never_switches(self):
        worker, ctrl = self._controller()
        with mock.patch.dict(
            "os.environ", {"SGLANG_ADAPTIVE_FORCE_SWAP_INTERVAL": "1"}
        ):
            for _ in range(5):
                ctrl.on_verify_complete([0, 0], batch_size=2, allow_switch=False)
            self.assertEqual(worker.applied, [])
            ctrl.on_verify_complete([0, 0], batch_size=2)  # default: may switch
        self.assertEqual(len(worker.applied), 1)

    def test_worker_routes_accept_to_policy_and_suppresses_ema_switch(self):
        calls = []

        class _Ctrl:
            def on_verify_complete(
                self, per_req, batch_size, result_steps, allow_switch
            ):
                calls.append((tuple(per_req), result_steps, allow_switch))

        w = object.__new__(EAGLEWorkerV2)
        w.adaptive_controller = _Ctrl()
        w.chain_policy = AdaptiveChainPolicy(k_max=3, candidates=[1, 2, 3])
        w.speculative_num_steps = 3
        w.on_verify_complete_cpu([2, 1], batch_size=2, steps=2)
        self.assertEqual(calls, [((2, 1), 2, False)])
        self.assertAlmostEqual(w.chain_policy.accept_ema(2), 2.5)
        # Without a chain policy the EMA keeps its switch (default path).
        w.chain_policy = None
        w.on_verify_complete_cpu([2], batch_size=1, steps=3)
        self.assertEqual(calls[-1][2], True)


class TestOneStateResident(unittest.TestCase):
    def test_fnFA25_case_budget_fits_both_but_cap_keeps_one(self):
        sizes = {"adaptive_state_k1": 490 << 20, "adaptive_state_k2": 516 << 20}
        budget = 4 << 30  # the boot-time reading said both fit
        # Budget alone: k1 stays mapped next to k2 (what OOM'd in round 17).
        self.assertEqual(
            plan_residency("adaptive_state_k2", ["adaptive_state_k1"], sizes, budget),
            [],
        )
        self.assertEqual(
            plan_residency(
                "adaptive_state_k2",
                ["adaptive_state_k1"],
                sizes,
                budget,
                max_resident=1,
            ),
            ["adaptive_state_k1"],
        )

    def test_cap_zero_is_the_old_budget_rule(self):
        sizes = {"a": 1, "b": 1, "c": 1}
        self.assertEqual(plan_residency("c", ["a", "b"], sizes, 10, max_resident=0), [])
        self.assertEqual(
            plan_residency("c", ["a", "b"], sizes, 10, max_resident=2), ["a"]
        )

    def test_env_is_read_by_the_manager(self):
        with mock.patch.dict(
            "os.environ", {"SGLANG_ADAPTIVE_GRAPH_MEMORY_MAX_RESIDENT": "1"}
        ):
            mgr = agm.AdaptiveGraphMemoryManager(mode="resident")
        self.assertEqual(mgr._max_resident, 1)


class _Adapter:
    def __init__(self):
        self.calls = []

    def pause(self, tag):
        self.calls.append(("pause", tag))

    def resume(self, tag):
        self.calls.append(("resume", tag))


class TestParkForSleep(unittest.TestCase):
    def _manager(self):
        mgr = object.__new__(agm.AdaptiveGraphMemoryManager)
        mgr.mode = "offload-scratch"
        mgr._finalized = True
        mgr._resident = ["adaptive_state_k2"]
        mgr._paused = {"adaptive_state_k1"}
        mgr._resumed_tag = "adaptive_state_k2"
        mgr._states = {
            "adaptive_state_k1": SimpleNamespace(footprint_bytes=416 << 20),
            "adaptive_state_k2": SimpleNamespace(footprint_bytes=416 << 20),
        }
        mgr._adapter = _Adapter()
        return mgr

    def test_park_activates_baseline_then_unmaps_the_ladder(self):
        worker, ctrl = _controller()
        ctrl._activate(2)
        mgr = self._manager()
        ctrl.graph_memory = mgr
        with (
            mock.patch.object(mgr, "ensure_active", lambda steps: None),
            mock.patch.object(mgr, "note_resident_activation", lambda steps: None),
            mock.patch.object(torch.cuda, "synchronize", lambda *a, **k: None),
        ):
            freed = ctrl.park(ctrl.baseline_steps)
        self.assertEqual(worker.speculative_num_steps, 3)  # baseline active first
        self.assertEqual(mgr._adapter.calls, [("pause", "adaptive_state_k2")])
        self.assertEqual(mgr._resident, [])
        self.assertEqual(freed, 416 << 20)

    def test_policy_park_is_rank_uniform(self):
        _bus, policies = _three_ranks(min_dwell=4)
        for i in range(37):
            _round(policies, i, _SURVIVAL_K2)
        for p in policies:
            p.park(3)
        ks = _round(policies, 37, _SURVIVAL_K2)
        self.assertEqual(ks, [3, 3, 3])
        for i in range(38, 80):
            ks = _round(policies, i, _SURVIVAL_K2)
            self.assertEqual(len(set(ks)), 1)


class TestChainProfile(unittest.TestCase):
    def test_chain_1_3_builds_no_step0_rung(self):
        self.assertEqual(resolve_candidate_steps_from_config("chain-1-3"), [1, 2, 3])


class TestCensus(unittest.TestCase):
    def test_census_every_n_rounds(self):
        _bus, policies = _three_ranks(min_dwell=4, census_every=50)
        with self.assertLogs(LOGGER, logging.INFO) as cm:
            for i in range(150):
                _round(policies, i, _SURVIVAL_K2)
        census = [r for r in cm.output if "SPEC-ADAPT-CENSUS" in r]
        self.assertEqual(len(census), 3 * 3)  # 3 per rank
        self.assertTrue(all("period_tok_s=" in r for r in census))

    def test_tok_s_is_na_until_measured(self):
        p = AdaptiveChainPolicy(k_max=3)
        self.assertTrue(math.isnan(p.tok_s_ema(2)))
        p.record_accept(2, 2.5)
        self.assertTrue(math.isnan(p.tok_s_ema(2)))  # cost still the prior
        for _ in range(8):
            p.record_duration(2, 25.0)
        self.assertAlmostEqual(p.tok_s_ema(2), 100.0)


if __name__ == "__main__":
    unittest.main()
