"""fnFL2x23 (2026-09-23): the arena host pool is chosen only for page_size 1.

``ArenaMHAHostPool`` maps one arena slot per token and refuses a paged device
pool at ``bind()`` -- but by then it has been constructed with only its staging
ring (``SGLANG_HICACHE_ARENA_STAGING_GB``, 4096 tokens at 0.05 GB). On Next
Flash (page_size 64) every rank bound that ring, D's prefetch budget became
0.9 x 4096 = 3686 tokens and the launcher refused the boot:
``W45 Weg2CarrierCensusRefused (below_floor): the agreed bound 3686 is at or
below the floor 5120``.
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


class TestArenaHostPoolNeedsTokenPages(CustomTestCase):
    @mock.patch.dict(os.environ, {"SGLANG_HICACHE_ARENA_HOST": "1"})
    def test_a_paged_pool_keeps_the_regular_host_pool(self):
        """RED ON 8546c4e3d4: the arena class was chosen for page_size 64."""
        self.assertIs(get_mha_host_pool_cls(_pool(64)), MHATokenToKVPoolHost)

    @mock.patch.dict(os.environ, {"SGLANG_HICACHE_ARENA_HOST": "1"})
    def test_a_token_paged_pool_still_gets_the_arena(self):
        from sglang.srt.mem_cache.pool_host.arena_pool import ArenaMHAHostPool

        self.assertIs(get_mha_host_pool_cls(_pool(1)), ArenaMHAHostPool)


if __name__ == "__main__":
    unittest.main()
