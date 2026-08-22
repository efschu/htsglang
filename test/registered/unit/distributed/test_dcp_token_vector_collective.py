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


class _SoloHostConfigurator:
    """Draft-solo host: its capacity DEPENDS on the installed token vector,
    because the draft KV pool spans the GLOBAL context. Mirrors
    pool_configurator's cell = t_target + (S / ratio_host) * t_draft."""

    def __init__(self, avail_bytes: int, t_target: int, t_draft: int, dcp_rank: int):
        self._avail = avail_bytes
        self._t_target = t_target
        self._t_draft = t_draft
        self._dcp_rank = dcp_rank

    def calculate_pool_sizes(self, budget_bytes, page_size):
        ratios = get_cp_token_ratios()
        g = sum(ratios) / ratios[self._dcp_rank]
        cell = self._t_target + g * self._t_draft
        return _StubConfig(int(self._avail // cell))


def _capacity_server_args(capacity: bool, solo_rank=None):
    # The install gate reads `uneven_kv_derived_mode`, not the capacity flag
    # directly -- ServerArgs grew that wrapper ("either derived mode, i.e.
    # the vector is computed after profiling rather than pinned") after this
    # stub was written, and the stub kept answering only the old name. The
    # relation is reproduced here rather than hardcoded to `capacity`, so a
    # future speed-mode case cannot silently take the wrong branch.
    speed = False
    return SimpleNamespace(
        uneven_kv_capacity_mode=lambda: capacity,
        uneven_kv_speed_mode=lambda: speed,
        uneven_kv_derived_mode=lambda: capacity or speed,
        rank_kv_ratio=None,
        speculative_draft_solo_active=lambda: solo_rank is not None,
        speculative_draft_solo_rank=lambda: solo_rank,
    )


def _run_ranks(
    per_rank_tokens,
    active,
    *,
    capacity_mode=False,
    allow_install=False,
    draft_worker=False,
    solo_rank=None,
    solo_avail=None,
    solo_cells=(1000, 200),
):
    """Invoke the method once per simulated DCP rank. Returns
    (ranks_that_reached_the_collective, vector_installed_after_run).

    ``solo_rank``: simulate the draft-solo placement with that rank as the
    host (vector-dependent capacity, see _SoloHostConfigurator)."""
    dcp_size = len(per_rank_tokens)
    collective_calls = []
    curves = {}

    def _configurator_for(rank):
        if solo_rank is not None and rank == solo_rank:
            return _SoloHostConfigurator(
                solo_avail, solo_cells[0], solo_cells[1], rank
            )
        return _StubConfigurator(per_rank_tokens[rank])

    def _local_tokens(rank):
        return int(
            _configurator_for(rank)
            .calculate_pool_sizes(1 << 30, 1)
            .max_total_num_tokens
        )

    def fake_all_gather_object(gathered, payload, group=None):
        collective_calls.append(payload)
        curves[payload[0]] = payload[2]
        full = [
            (
                r,
                max(_local_tokens(r), 0),
                curves.get(r) if r == solo_rank else None,
            )
            for r in range(dcp_size)
        ]
        gathered[: len(full)] = full

    # The host is simulated first so the later ranks see the curve it
    # contributes to the collective (in a real run every rank receives it
    # from the same all_gather).
    order = [r for r in range(dcp_size) if r == solo_rank] + [
        r for r in range(dcp_size) if r != solo_rank
    ]
    per_rank_installed = []
    set_cp_token_ratios(list(active))
    try:
        for rank in order:
            stub = SimpleNamespace(
                dcp_size=dcp_size,
                page_size=1,
                tp_rank=rank,
                server_args=_capacity_server_args(capacity_mode, solo_rank),
                is_draft_worker=draft_worker,
                # #797 split the POOL-geometry flag off the construction flag:
                #
                #   model_runner.py:512
                #   self.is_draft_pool_worker = (
                #       is_draft_worker and not is_phase_flip_tp_stack)
                #
                # The pool-shape decision sites in
                # `model_runner_kv_cache_mixin` now consult the POOL flag and
                # "never is_draft_worker directly" (:504-511), so this stub --
                # which mirrors a ModelRunner rather than being one -- has to
                # carry it or the mixin helpers bound below raise
                # AttributeError before reaching the geometry under test.
                #
                # Written as the DERIVATION rather than a pinned literal: this
                # fixture builds no phase-flip TP stack, so the second term is
                # False and the pool flag simply equals the draft flag. If a
                # case here ever does build one, the value has to move with it,
                # and stating the rule is what makes that visible.
                is_draft_pool_worker=draft_worker,
            )
            # The stub is not a ModelRunner, so bind the mixin helpers the
            # method calls on ``self`` explicitly.
            stub._is_solo_draft_kv_host = (
                lambda s=stub: ModelRunnerKVCacheMixin._is_solo_draft_kv_host(s)
            )
            stub._solo_host_capacity_curve = (
                lambda *a, s=stub: ModelRunnerKVCacheMixin._solo_host_capacity_curve(
                    s, *a
                )
            )
            stub._solo_fixed_point_capacity = (
                ModelRunnerKVCacheMixin._solo_fixed_point_capacity
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
                side_effect=lambda mr: _configurator_for(mr.tp_rank),
            ), mock.patch(
                "torch.distributed.all_gather_object",
                side_effect=fake_all_gather_object,
            ):
                ModelRunnerKVCacheMixin._maybe_suggest_dcp_token_vector(
                    stub, budget_bytes=1 << 30, allow_install=allow_install
                )
            per_rank_installed.append(get_cp_token_ratios())
        installed = get_cp_token_ratios()
    finally:
        set_cp_token_ratios(None)
    assert all(
        v == per_rank_installed[0] for v in per_rank_installed
    ), f"ranks disagreed on the installed vector: {per_rank_installed}"
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


# Mirrors the measured 27B-FP8 TP=3 NEXTN draft-solo boot: rank 0 hosts the
# unsharded draft + a globally-sized draft KV pool, so its profiled capacity
# (108341 tokens under the seeded vector [16,23,25]) is a FUNCTION of its
# ownership share, while the shadows' is not. Pre-fix the boot stalled at
# max_total_num_tokens=433344 and only logged the un-actioned
# "restart with SGLANG_UNEVEN_TOKEN_VECTOR=5,13,14" hint.
SOLO_ACTIVE = [16, 23, 25]
SOLO_SHADOWS = [0, 290465, 312551]
SOLO_T_TARGET, SOLO_T_DRAFT = 1000, 200
SOLO_AVAIL = 108341 * (SOLO_T_TARGET + 4 * SOLO_T_DRAFT)
# What the optimizer would pick from the RAW measured capacities (the naive,
# vector-unaware install).
SOLO_NAIVE = [5, 13, 14]


def _solo_capacity(vector, rank):
    """Actual per-rank capacity once ``vector`` is installed."""
    if rank == 0:
        cell = SOLO_T_TARGET + (sum(vector) / vector[0]) * SOLO_T_DRAFT
        return int(SOLO_AVAIL // cell)
    return SOLO_SHADOWS[rank]


def _solo_achieved_context(vector):
    """max_total_num_tokens the owner rule delivers for ``vector``:
    min_r(P_r(vector) // ratio_r) * sum(ratios)."""
    return min(
        _solo_capacity(vector, r) // vector[r] for r in range(len(vector))
    ) * sum(vector)


class TestDraftSoloCapacityInstall(CustomTestCase):
    """--rank-kv-ratio capacity on the DRAFT-SOLO path (the bug): the
    post-profiling measured install must run there too, and it must use the
    host's self-consistent (vector-corrected) capacity."""

    def _install(self, **kwargs):
        return _run_ranks(
            list(SOLO_SHADOWS),
            active=SOLO_ACTIVE,
            capacity_mode=True,
            allow_install=True,
            solo_rank=0,
            solo_avail=SOLO_AVAIL,
            solo_cells=(SOLO_T_TARGET, SOLO_T_DRAFT),
            **kwargs,
        )

    def test_solo_installs_a_measured_vector(self):
        reached, installed = self._install()
        self.assertEqual(reached, 3)
        self.assertNotEqual(
            installed,
            SOLO_ACTIVE,
            "draft-solo stopped at the pre-boot prediction: the measured "
            "install never ran (ranks 1/2 stay idle)",
        )

    def test_solo_install_raises_the_pool(self):
        _, installed = self._install()
        self.assertGreater(
            _solo_achieved_context(installed),
            _solo_achieved_context(SOLO_ACTIVE),
        )

    def test_solo_beats_the_vector_unaware_optimum(self):
        """Installing partition_units over the RAW measured capacities hands
        the host a share it can no longer fund (its draft pool grows with the
        global context), so the min-rule claws the pool back. The corrected
        fixed point must do strictly better."""
        _, installed = self._install()
        self.assertGreater(
            _solo_achieved_context(installed),
            _solo_achieved_context(SOLO_NAIVE),
        )

    def test_solo_shrinks_the_hosts_share(self):
        _, installed = self._install()
        host_share = installed[0] / sum(installed)
        self.assertLess(host_share, SOLO_ACTIVE[0] / sum(SOLO_ACTIVE))

    def test_solo_hint_only_without_allow_install(self):
        _, installed = _run_ranks(
            list(SOLO_SHADOWS),
            active=SOLO_ACTIVE,
            capacity_mode=True,
            allow_install=False,
            solo_rank=0,
            solo_avail=SOLO_AVAIL,
            solo_cells=(SOLO_T_TARGET, SOLO_T_DRAFT),
        )
        self.assertEqual(installed, SOLO_ACTIVE)

    def test_solo_coupled_mode_never_installs(self):
        _, installed = _run_ranks(
            list(SOLO_SHADOWS),
            active=SOLO_ACTIVE,
            capacity_mode=False,
            allow_install=True,
            solo_rank=0,
            solo_avail=SOLO_AVAIL,
            solo_cells=(SOLO_T_TARGET, SOLO_T_DRAFT),
        )
        self.assertEqual(installed, SOLO_ACTIVE)

    def test_non_solo_install_is_unchanged(self):
        """Same capacities, no solo host: the plain measured optimum (no
        correction, no extra collective payload semantics)."""
        reached, installed = _run_ranks(
            MEASURED_P, active=ESTIMATE, capacity_mode=True, allow_install=True
        )
        self.assertEqual(reached, 3)
        self.assertEqual(installed, OPTIMAL)


if __name__ == "__main__":
    unittest.main()
