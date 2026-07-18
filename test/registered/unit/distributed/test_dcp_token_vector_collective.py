"""Unit tests for ModelRunnerKVCacheMixin._maybe_suggest_dcp_token_vector:

1. Collective uniformity: the function all_gathers each rank's LOCAL
   profiled token capacity P_r on the CPU group. Every gate BEFORE that
   collective must be rank-uniform; an early return keyed on the
   rank-local P_r (the pre-fix ``if local_p <= 0: return``) lets one rank
   skip the all_gather the other ranks entered -- a distributed hang. The
   tests simulate the collective with a recording fake and assert every
   rank reaches it even when its local capacity degenerates to 0.

2. --rank-kv-ratio capacity (task #88): with ``allow_install=True`` the
   measured optimal vector is INSTALLED (one-boot convergence) instead of
   logged as a restart hint -- but only in capacity mode, only when it
   strictly raises the context budget, never for the draft worker, and
   never over an explicit SGLANG_UNEVEN_TOKEN_VECTOR pin.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.distributed.utils import (
    get_cp_token_ratios,
    set_cp_token_ratios,
)
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _StubConfig:
    def __init__(self, tokens: int):
        self.max_total_num_tokens = tokens


class _StubConfigurator:
    def __init__(self, tokens: int):
        self._tokens = tokens

    def calculate_pool_sizes(self, budget_bytes, page_size):
        return _StubConfig(self._tokens)


def _capacity_server_args(capacity: bool):
    return SimpleNamespace(uneven_kv_capacity_mode=lambda: capacity)


def _run_ranks(
    per_rank_tokens,
    active,
    *,
    capacity_mode=False,
    allow_install=False,
    draft_worker=False,
):
    """Invoke the method once per simulated DCP rank. Returns
    (ranks_that_reached_the_collective, vector_installed_after_run)."""
    dcp_size = len(per_rank_tokens)
    collective_calls = []

    def fake_all_gather_object(gathered, payload, group=None):
        collective_calls.append(payload)
        full = [
            (r, max(int(per_rank_tokens[r]), 0)) for r in range(dcp_size)
        ]
        gathered[: len(full)] = full

    set_cp_token_ratios(list(active))
    try:
        for rank in range(dcp_size):
            stub = SimpleNamespace(
                dcp_size=dcp_size,
                page_size=1,
                tp_rank=rank,
                server_args=_capacity_server_args(capacity_mode),
                is_draft_worker=draft_worker,
            )
            world_group = mock.Mock(world_size=dcp_size, cpu_group=None)
            parallel = mock.Mock(attn_dcp_rank=rank)
            with mock.patch(
                "sglang.srt.model_executor.model_runner_kv_cache_mixin"
                ".get_world_group",
                return_value=world_group,
            ), mock.patch(
                "sglang.srt.model_executor.model_runner_kv_cache_mixin"
                ".get_parallel",
                return_value=parallel,
            ), mock.patch(
                "sglang.srt.model_executor.pool_configurator"
                ".create_memory_pool_configurator",
                return_value=_StubConfigurator(per_rank_tokens[rank]),
            ), mock.patch(
                "torch.distributed.all_gather_object",
                side_effect=fake_all_gather_object,
            ):
                ModelRunnerKVCacheMixin._maybe_suggest_dcp_token_vector(
                    stub, budget_bytes=1 << 30, allow_install=allow_install
                )
        installed = get_cp_token_ratios()
    finally:
        set_cp_token_ratios(None)
    return len(collective_calls), installed


class TestDcpTokenVectorCollective(CustomTestCase):
    """Rank-uniform collective guard (the [bugfix] commit)."""

    def test_degenerate_rank_still_reaches_collective(self):
        """One rank profiles to 0 tokens: pre-fix that rank returned before
        the all_gather (hang against the peer); post-fix both ranks enter
        the collective and bail uniformly afterwards."""
        reached, installed = _run_ranks([0, 1000], active=[2, 1])
        self.assertEqual(
            reached,
            2,
            "a rank skipped the all_gather based on its rank-local capacity "
            "(collective divergence -- distributed hang in real runs)",
        )
        self.assertEqual(installed, [2, 1])

    def test_negative_capacity_is_clamped_and_gathered(self):
        reached, _ = _run_ranks([-5, 1000], active=[2, 1])
        self.assertEqual(reached, 2)

    def test_happy_path_reaches_collective_on_all_ranks(self):
        reached, installed = _run_ranks([600, 300], active=[2, 1])
        self.assertEqual(reached, 2)
        # Hint-only mode never mutates the installed vector.
        self.assertEqual(installed, [2, 1])


# Mirrors the measured 27B-FP8 TP=3 boot: estimate [30,17,17], profiled
# P_r [301435, 117912, 158474] -> measured optimal [33,13,18] raising
# C from 443904 to ~563456.
MEASURED_P = [301435, 117912, 158474]
ESTIMATE = [30, 17, 17]
OPTIMAL = [33, 13, 18]


class TestCapacityInstall(CustomTestCase):
    """--rank-kv-ratio capacity: measured install semantics (task #88)."""

    def test_capacity_installs_measured_vector(self):
        reached, installed = _run_ranks(
            MEASURED_P, active=ESTIMATE, capacity_mode=True, allow_install=True
        )
        self.assertEqual(reached, 3)
        self.assertEqual(installed, OPTIMAL)

    def test_capacity_keeps_estimate_when_not_strictly_better(self):
        """Small P_r: the 64-unit integerized 'optimal' can predict a LOWER
        context than the active vector -- keep the active vector then."""
        reached, installed = _run_ranks(
            [600, 300], active=[2, 1], capacity_mode=True, allow_install=True
        )
        self.assertEqual(reached, 2)
        self.assertEqual(installed, [2, 1])

    def test_coupled_mode_never_installs(self):
        _, installed = _run_ranks(
            MEASURED_P, active=ESTIMATE, capacity_mode=False, allow_install=True
        )
        self.assertEqual(installed, ESTIMATE)

    def test_no_install_without_allow_install(self):
        """The post-capture pass (vector already snapshotted by backends and
        graphs) must stay hint-only even in capacity mode."""
        _, installed = _run_ranks(
            MEASURED_P, active=ESTIMATE, capacity_mode=True, allow_install=False
        )
        self.assertEqual(installed, ESTIMATE)

    def test_env_pin_suppresses_install(self):
        with mock.patch.dict(
            os.environ, {"SGLANG_UNEVEN_TOKEN_VECTOR": "30,17,17"}
        ):
            _, installed = _run_ranks(
                MEASURED_P,
                active=ESTIMATE,
                capacity_mode=True,
                allow_install=True,
            )
        self.assertEqual(installed, ESTIMATE)

    def test_draft_worker_never_installs(self):
        _, installed = _run_ranks(
            MEASURED_P,
            active=ESTIMATE,
            capacity_mode=True,
            allow_install=True,
            draft_worker=True,
        )
        self.assertEqual(installed, ESTIMATE)


if __name__ == "__main__":
    unittest.main()
