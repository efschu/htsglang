"""#602 corridor token vector: the collective / install half.

CPU only. The multi-rank behaviour is simulated the same way
``test_dcp_token_vector_collective.py`` does it: the unbound mixin method is
driven once per rank against a fresh stub, with
``torch.distributed.all_gather_object`` replaced by a fake that synthesizes
the full gathered list. No device, no process group, no model.

    CUDA_VISIBLE_DEVICES=99 PYTHONPATH=<worktree>/python \
        python -m pytest -q \
        test/registered/unit/distributed/test_corridor_token_vector_collective.py
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.distributed.corridor_vector import solve_token_vector
from sglang.srt.distributed.utils import get_cp_token_ratios, set_cp_token_ratios
from sglang.srt.model_executor.model_runner_kv_cache_mixin import (
    ModelRunnerKVCacheMixin,
    corridor_mode_active,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

MIB = 1 << 20

# Three cards, one rank each, in --rank-gpu-id order.
RANK_GPU_ID = [0, 1, 2]
# Profiled token capacity P_r (the budget model's answer).
PROFILED = [296710, 141494, 106011]
ACTIVE = [30, 17, 17]


class _StubConfig:
    def __init__(self, tokens):
        self.max_total_num_tokens = tokens


class _StubConfigurator:
    """Turns a byte budget into tokens at a fixed cell size, which is exactly
    what the real configurator does for the quantities under test."""

    def __init__(self, cell_bytes, profiled_tokens):
        self._cell = cell_bytes
        self._profiled = profiled_tokens

    def calculate_pool_sizes(self, budget_bytes, page_size):
        # The profiling path passes the model budget; the corridor path passes
        # the corridor budget. Distinguish them the way the caller does: by
        # the bytes handed in.
        if budget_bytes == _MODEL_BUDGET_SENTINEL:
            return _StubConfig(self._profiled)
        return _StubConfig(int(budget_bytes) // self._cell)


_MODEL_BUDGET_SENTINEL = 1 << 40
_CELL_BYTES = 4096


def _server_args(
    *,
    free_mib,
    reserve_mib=1024,
    post_sizing_mib=None,
    corridor=True,
    gpu_ids=None,
):
    gpu_ids = list(RANK_GPU_ID if gpu_ids is None else gpu_ids)
    post = (
        {g: 2000 for g in set(gpu_ids)}
        if post_sizing_mib is None
        else dict(post_sizing_mib)
    )
    sa = SimpleNamespace(
        rank_kv_ratio="corridor" if corridor else "capacity",
        rank_gpu_id=gpu_ids,
        corridor_post_sizing_mib=post,
        rank_kv_speed_weights=None,
        rank_perf_loose_ctx_percent=0.0,
    )
    sa.uneven_kv_corridor_mode = lambda: corridor
    sa.uneven_kv_capacity_mode = lambda: not corridor
    sa.uneven_kv_speed_mode = lambda: False
    sa.uneven_kv_derived_mode = lambda: True
    sa.speculative_draft_solo_active = lambda: False
    sa.speculative_draft_solo_rank = lambda: None
    sa.user_reserve_mib_per_gpu = lambda ids: {g: reserve_mib for g in set(ids)}
    sa._free_mib = dict(free_mib)
    return sa


def _run_ranks(server_args, *, allow_install=True, dcp_size=3):
    """Drive the vector path once per rank; return (installed-per-rank, calls)."""
    collective_calls = []
    installed_per_rank = []

    def fake_all_gather_object(gathered, payload, group=None):
        collective_calls.append(payload)
        full = [_payload_for(r) for r in range(dcp_size)]
        gathered[: len(full)] = full

    def _payload_for(rank):
        stub = _stub_for(rank)
        cfg = _StubConfigurator(_CELL_BYTES, PROFILED[rank])
        q = (
            ModelRunnerKVCacheMixin._corridor_local_capacity(stub, cfg)
            if corridor_mode_active(server_args)
            else None
        )
        return (rank, PROFILED[rank], None, q)

    def _stub_for(rank):
        stub = SimpleNamespace(
            dcp_size=dcp_size,
            page_size=1,
            tp_rank=rank,
            server_args=server_args,
            is_draft_worker=False,
            # #797 (model_runner.py:512) split the POOL-geometry flag off the
            # construction flag -- `is_draft_pool_worker = is_draft_worker and
            # not is_phase_flip_tp_stack` -- and the sizing sites this stub
            # drives read the pool flag, never `is_draft_worker` directly. A
            # SimpleNamespace mirroring a ModelRunner has to carry both.
            # Every stub in this file is a plain target-model rank, so both
            # flags are False and nothing under test moves.
            is_draft_pool_worker=False,
        )
        gpu = server_args.rank_gpu_id[rank]
        stub._corridor_card_free_bytes = server_args._free_mib[gpu] * MIB
        stub._rank_vector_index = lambda r=rank: r
        stub._corridor_card_key = lambda s=stub: (
            ModelRunnerKVCacheMixin._corridor_card_key(s)
        )
        stub._is_solo_draft_kv_host = lambda s=stub: (
            ModelRunnerKVCacheMixin._is_solo_draft_kv_host(s)
        )
        stub._solo_host_capacity_curve = lambda *a, s=stub: None
        stub._solo_fixed_point_capacity = (
            ModelRunnerKVCacheMixin._solo_fixed_point_capacity
        )
        stub._corridor_local_capacity = lambda cfg, s=stub: (
            ModelRunnerKVCacheMixin._corridor_local_capacity(s, cfg)
        )
        return stub

    set_cp_token_ratios(list(ACTIVE))
    try:
        for rank in range(dcp_size):
            stub = _stub_for(rank)
            world_group = mock.Mock(world_size=dcp_size, cpu_group=None)
            parallel = mock.Mock(attn_dcp_rank=rank)
            with (
                mock.patch(
                    "sglang.srt.model_executor.model_runner_kv_cache_mixin.get_world_group",
                    return_value=world_group,
                ),
                mock.patch(
                    "sglang.srt.model_executor.model_runner_kv_cache_mixin.get_parallel",
                    return_value=parallel,
                ),
                mock.patch(
                    "sglang.srt.model_executor.pool_configurator.create_memory_pool_configurator",
                    side_effect=lambda mr: _StubConfigurator(
                        _CELL_BYTES, PROFILED[mr.tp_rank]
                    ),
                ),
                mock.patch.object(
                    torch.distributed, "all_gather_object", fake_all_gather_object
                ),
            ):
                ModelRunnerKVCacheMixin._maybe_suggest_dcp_token_vector(
                    stub,
                    budget_bytes=_MODEL_BUDGET_SENTINEL,
                    allow_install=allow_install,
                )
            installed_per_rank.append(get_cp_token_ratios())
        return installed_per_rank, collective_calls
    finally:
        set_cp_token_ratios(None)


class TestCorridorCollective(CustomTestCase):
    def test_every_rank_installs_the_identical_vector(self):
        sa = _server_args(free_mib={0: 30000, 1: 16000, 2: 14000})
        installed, calls = _run_ranks(sa)
        self.assertEqual(len(calls), 3, "a rank skipped the collective")
        for got in installed:
            self.assertEqual(
                got, installed[0], "ranks disagreed on the installed vector"
            )
        self.assertNotEqual(installed[0], ACTIVE)

    def test_no_install_without_allow_install(self):
        sa = _server_args(free_mib={0: 30000, 1: 16000, 2: 14000})
        installed, _ = _run_ranks(sa, allow_install=False)
        for got in installed:
            self.assertEqual(got, ACTIVE)

    def test_the_floor_is_respected_by_the_installed_vector(self):
        """Every rank's share must fit under its corridor capacity."""
        free = {0: 30000, 1: 16000, 2: 14000}
        sa = _server_args(free_mib=free)
        installed, _ = _run_ranks(sa)
        vec = installed[0]
        q = [((free[g] - 1024 - 2000) * MIB) // _CELL_BYTES for g in RANK_GPU_ID]
        caps = [min(p, qq) for p, qq in zip(PROFILED, q)]
        unit = min(c // v for c, v in zip(caps, vec))
        for r, (held, cap) in enumerate(zip([unit * v for v in vec], caps)):
            self.assertLessEqual(held, cap, f"rank {r} exceeds its corridor")

    def test_corridor_shrinks_the_pool_when_the_floor_binds(self):
        """The mode must install even though the result is SMALLER -- that is
        the case it exists for, and an improvement gate would refuse it."""
        # 4000 MiB free - 1024 reserve - 2000 post-sizing = 976 MiB of pool on
        # card 0, i.e. ~250 k tokens against a profiled 296 k: the floor binds.
        tight = {0: 4000, 1: 16000, 2: 14000}
        sa = _server_args(free_mib=tight)
        installed, _ = _run_ranks(sa)
        vec = installed[0]
        self.assertNotEqual(vec, ACTIVE)
        q0 = ((tight[0] - 1024 - 2000) * MIB) // _CELL_BYTES
        self.assertLess(q0, PROFILED[0], "test setup: the floor must bind")
        caps = [
            min(PROFILED[0], q0),
            min(PROFILED[1], ((16000 - 1024 - 2000) * MIB) // _CELL_BYTES),
            min(PROFILED[2], ((14000 - 1024 - 2000) * MIB) // _CELL_BYTES),
        ]
        unit = min(c // v for c, v in zip(caps, vec))
        self.assertLessEqual(unit * vec[0], q0)

    def test_capacity_mode_is_unchanged_by_the_corridor_code(self):
        sa = _server_args(free_mib={0: 30000, 1: 16000, 2: 14000}, corridor=False)
        installed, calls = _run_ranks(sa)
        self.assertEqual(len(calls), 3)
        # capacity mode never consults a corridor capacity
        self.assertTrue(all(c[3] is None for c in calls))


class TestCorridorRefusals(CustomTestCase):
    def _capacity(self, stub_free_mib, post_sizing_mib=2000, reserve_mib=1024):
        sa = _server_args(
            free_mib={0: stub_free_mib, 1: 16000, 2: 14000},
            reserve_mib=reserve_mib,
            post_sizing_mib={0: post_sizing_mib, 1: 2000, 2: 2000},
        )
        stub = SimpleNamespace(
            dcp_size=3,
            page_size=1,
            tp_rank=0,
            server_args=sa,
            is_draft_worker=False,
            # #797, as in `_stub_for` above: the pool flag is read separately.
            is_draft_pool_worker=False,
        )
        stub._corridor_card_free_bytes = stub_free_mib * MIB
        stub._rank_vector_index = lambda: 0
        stub._corridor_card_key = lambda: ModelRunnerKVCacheMixin._corridor_card_key(
            stub
        )
        return stub, _StubConfigurator(_CELL_BYTES, PROFILED[0])

    def test_card_that_cannot_fund_its_floor_is_named(self):
        stub, cfg = self._capacity(2500)
        with self.assertRaises(ValueError) as ctx:
            ModelRunnerKVCacheMixin._corridor_local_capacity(stub, cfg)
        msg = str(ctx.exception)
        self.assertIn("GPU 0", msg)
        self.assertIn("1024", msg)
        self.assertIn("2000", msg)

    def test_missing_post_sizing_price_is_refused_not_assumed_zero(self):
        stub, cfg = self._capacity(30000)
        stub.server_args.corridor_post_sizing_mib = {1: 2000, 2: 2000}
        with self.assertRaises(ValueError) as ctx:
            ModelRunnerKVCacheMixin._corridor_local_capacity(stub, cfg)
        self.assertIn("no post-sizing demand", str(ctx.exception))

    def test_missing_free_reading_is_refused(self):
        stub, cfg = self._capacity(30000)
        stub._corridor_card_free_bytes = None
        with self.assertRaises(ValueError) as ctx:
            ModelRunnerKVCacheMixin._corridor_local_capacity(stub, cfg)
        self.assertIn("physical card", str(ctx.exception))

    def test_colocated_ranks_split_the_card(self):
        sa = _server_args(
            free_mib={0: 30000, 1: 16000},
            gpu_ids=[0, 0, 1],
            post_sizing_mib={0: 2000, 1: 2000},
        )
        stub = SimpleNamespace(
            dcp_size=3,
            page_size=1,
            tp_rank=0,
            server_args=sa,
            is_draft_worker=False,
            # #797, as in `_stub_for` above: the pool flag is read separately.
            is_draft_pool_worker=False,
        )
        stub._corridor_card_free_bytes = 30000 * MIB
        stub._rank_vector_index = lambda: 0
        stub._corridor_card_key = lambda: ModelRunnerKVCacheMixin._corridor_card_key(
            stub
        )
        cfg = _StubConfigurator(_CELL_BYTES, PROFILED[0])
        got = ModelRunnerKVCacheMixin._corridor_local_capacity(stub, cfg)
        expected = (((30000 - 1024 - 2000) * MIB) // 2) // _CELL_BYTES
        self.assertEqual(got, expected)


class TestModePredicate(CustomTestCase):
    def test_absent_predicate_is_the_default_path(self):
        self.assertFalse(corridor_mode_active(SimpleNamespace()))

    def test_present_predicate_is_honoured(self):
        self.assertTrue(
            corridor_mode_active(SimpleNamespace(uneven_kv_corridor_mode=lambda: True))
        )
        self.assertFalse(
            corridor_mode_active(SimpleNamespace(uneven_kv_corridor_mode=lambda: False))
        )


class TestSolverAgreesWithTheInstall(CustomTestCase):
    def test_installed_vector_is_the_solver_output(self):
        free = {0: 30000, 1: 16000, 2: 14000}
        sa = _server_args(free_mib=free)
        installed, _ = _run_ranks(sa)
        q = [((free[g] - 1024 - 2000) * MIB) // _CELL_BYTES for g in RANK_GPU_ID]
        caps = [min(p, qq) for p, qq in zip(PROFILED, q)]
        self.assertEqual(installed[0], solve_token_vector(caps).vector)


if __name__ == "__main__":
    unittest.main()
