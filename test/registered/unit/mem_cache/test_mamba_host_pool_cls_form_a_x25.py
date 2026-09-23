"""fnFL2x25 (23.09.): the anchor host pool class follows the RANK, not the knob.

The 27B merge brought the arena anchor pool (#1427b): ``ArenaMambaPoolHost``
resolves blobs in the storage tier's arena. A Form A expert worker rides
``FormAWorkerNullStorage`` (no arena, no canonical mamba blob), so its
``ensure_bound`` answers False, ``build_hicache_transfers`` returns [] on
every PREFETCH ("host anchor pool exhausted" with 6 of 6 slots free), the
worker leaves the prefetch alone while the host registers it, and the next
prefetch collective dies with gloo ``8 vs 4`` (W17). x22, one commit before
the merge, ran the workers on the plain pool and all three ranks registered.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import os
import unittest
from unittest import mock

from sglang.srt import rank_role
from sglang.srt.mem_cache.memory_pool_host import (
    MambaPoolHost,
    mamba_host_pool_cls,
)
from sglang.srt.mem_cache.pool_host.arena_mamba_pool import ArenaMambaPoolHost
from sglang.test.test_utils import CustomTestCase


class _Plan:
    def __init__(self, workers):
        self._w = set(workers)

    def role_of(self, rank):
        return "worker" if rank in self._w else "host"

    def is_worker(self, rank):
        return rank in self._w


class TestAnchorPoolClassFollowsTheRank(CustomTestCase):
    def tearDown(self):
        rank_role.set_form_a_role_plan(None, 0)

    @mock.patch.dict(os.environ, {"SGLANG_HICACHE_ARENA_HOST": "1"})
    def test_a_form_a_worker_keeps_the_plain_pool(self):
        """RED ON 0f5294f51c: the worker got the arena class."""
        rank_role.set_form_a_role_plan(_Plan(workers={1, 2}), rank=1)
        self.assertIs(mamba_host_pool_cls(), MambaPoolHost)

    @mock.patch.dict(os.environ, {"SGLANG_HICACHE_ARENA_HOST": "1"})
    def test_the_form_a_host_and_a_classic_rank_get_the_arena(self):
        rank_role.set_form_a_role_plan(_Plan(workers={1, 2}), rank=0)
        self.assertIs(mamba_host_pool_cls(), ArenaMambaPoolHost)
        rank_role.set_form_a_role_plan(None, 0)
        self.assertIs(mamba_host_pool_cls(), ArenaMambaPoolHost)

    @mock.patch.dict(os.environ, {"SGLANG_HICACHE_ARENA_HOST": "0"})
    def test_without_the_knob_nobody_gets_the_arena(self):
        rank_role.set_form_a_role_plan(None, 0)
        self.assertIs(mamba_host_pool_cls(), MambaPoolHost)


if __name__ == "__main__":
    unittest.main()
