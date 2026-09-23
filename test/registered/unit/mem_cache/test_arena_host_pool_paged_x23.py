"""fnFL2x23 (2026-09-23) -> x59 (Task #107): which MHA host pool a rank gets.

x23 pinned the fallback: ``ArenaMHAHostPool`` knew one arena slot per TOKEN
and refused a paged device pool at ``bind()``, so page_size 64 (Next Flash)
kept the regular pinned pool -- and every rank of both groups then held its
own 4-GB L2 (11.7 GB pinned, nothing but pages in transit) next to the one
arena. x59 made the arena pool paged (one slot per page), so the KV anchor
takes the arena form at ANY page size; only a paged DRAFT pool, a per-key
sidecar addressed by the KV ids, stays the plain class.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.mem_cache.pool_host.mha import (
    MHATokenToKVPoolHost,
    get_mha_host_pool_cls,
)
from sglang.test.test_utils import CustomTestCase


def _pool(page_size):
    return SimpleNamespace(head_dim=4, v_head_dim=4, page_size=page_size)


class TestArenaHostPoolChooser(CustomTestCase):
    @mock.patch.dict(os.environ, {"SGLANG_HICACHE_ARENA_HOST": "1"})
    def test_a_paged_kv_pool_takes_the_arena(self):
        """x59: page_size 64 gets the (now paged) arena pool -- one L2."""
        from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool

        self.assertIs(get_mha_host_pool_cls(_pool(64)), ArenaMHAHostPool)

    @mock.patch.dict(os.environ, {"SGLANG_HICACHE_ARENA_HOST": "1"})
    def test_a_paged_draft_pool_keeps_the_regular_host_pool(self):
        """The paged draft page is a per-key sidecar addressed by the KV
        pool's ids; the arena pool's draft role is token-paged only."""
        self.assertIs(get_mha_host_pool_cls(_pool(64), role="draft"), MHATokenToKVPoolHost)

    @mock.patch.dict(os.environ, {"SGLANG_HICACHE_ARENA_HOST": "1"})
    def test_a_token_paged_pool_still_gets_the_arena(self):
        from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool

        self.assertIs(get_mha_host_pool_cls(_pool(1)), ArenaMHAHostPool)
        self.assertIs(get_mha_host_pool_cls(_pool(1), role="draft"), ArenaMHAHostPool)

    @mock.patch.dict(os.environ, {"SGLANG_HICACHE_ARENA_HOST": "0"})
    def test_arena_off_keeps_the_regular_host_pool(self):
        self.assertIs(get_mha_host_pool_cls(_pool(64)), MHATokenToKVPoolHost)


if __name__ == "__main__":
    unittest.main()
