"""Unit test for the collective-uniformity fix in
ModelRunnerKVCacheMixin._maybe_suggest_dcp_token_vector.

The function all_gathers each rank's LOCAL profiled token capacity P_r on
the CPU group. Every gate BEFORE that collective must be rank-uniform:
an early return keyed on the rank-local P_r (the pre-fix `if local_p <= 0:
return`) lets one rank skip the all_gather that the other ranks entered --
a distributed hang instead of a clean bail-out.

The test simulates the collective with a recording fake (no real process
group): it invokes the method once per simulated rank, with one rank
profiling to 0 tokens, and asserts that EVERY rank still reaches the
collective (the degenerate value is rejected by the uniform post-gather
``any(p <= 0)`` check instead).
"""

import unittest
from unittest import mock

from sglang.srt.distributed.utils import set_cp_token_ratios
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

DCP_SIZE = 2


class _StubRunner:
    """The minimal ModelRunner surface the method touches."""

    dcp_size = DCP_SIZE
    page_size = 1
    tp_rank = 0


class _StubConfig:
    def __init__(self, tokens: int):
        self.max_total_num_tokens = tokens


class _StubConfigurator:
    def __init__(self, tokens: int):
        self._tokens = tokens

    def calculate_pool_sizes(self, budget_bytes, page_size):
        return _StubConfig(self._tokens)


class TestDcpTokenVectorCollective(CustomTestCase):
    def _run_ranks(self, per_rank_tokens):
        """Invoke the method once per simulated DCP rank; return the number
        of ranks that reached the all_gather collective."""
        collective_calls = []

        def fake_all_gather_object(gathered, payload, group=None):
            collective_calls.append(payload)
            # Deliver what a real gather would: every rank's payload.
            full = [
                (r, max(int(per_rank_tokens[r]), 0))
                for r in range(DCP_SIZE)
            ]
            gathered[: len(full)] = full

        set_cp_token_ratios([2, 1])  # non-uniform: weighted DCP active
        try:
            for rank in range(DCP_SIZE):
                world_group = mock.Mock(world_size=DCP_SIZE, cpu_group=None)
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
                        _StubRunner(), budget_bytes=1 << 30
                    )
        finally:
            set_cp_token_ratios(None)
        return len(collective_calls)

    def test_degenerate_rank_still_reaches_collective(self):
        """One rank profiles to 0 tokens: pre-fix that rank returned before
        the all_gather (hang against the peer); post-fix both ranks enter
        the collective and bail uniformly afterwards."""
        reached = self._run_ranks(per_rank_tokens=[0, 1000])
        self.assertEqual(
            reached,
            DCP_SIZE,
            "a rank skipped the all_gather based on its rank-local capacity "
            "(collective divergence -- distributed hang in real runs)",
        )

    def test_negative_capacity_is_clamped_and_gathered(self):
        reached = self._run_ranks(per_rank_tokens=[-5, 1000])
        self.assertEqual(reached, DCP_SIZE)

    def test_happy_path_reaches_collective_on_all_ranks(self):
        reached = self._run_ranks(per_rank_tokens=[600, 300])
        self.assertEqual(reached, DCP_SIZE)


if __name__ == "__main__":
    unittest.main()
