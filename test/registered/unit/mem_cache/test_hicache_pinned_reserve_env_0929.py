# SPDX-License-Identifier: Apache-2.0
"""z30x2-yarn2 (2026-09-29 10:59:37Z): D TP0 died building its HiCache host pools.

    ValueError: Pinned host RAM over-committed: 0.67 GB requested across 3
    pool(s) [ArenaMHAHostPool 0.05 GB; HiCache Mamba anchor host pool 0.35 GB;
    V4 paged host pool qsa_indexer 0.27 GB] does not fit in 11.18 GB available
    minus a 10.74 GB OS reserve (caller) = 0.44 GB usable.

The container sets SGLANG_PINNED_HOST_RESERVE_GIB=2 (RC2.1), and
check_and_register_pinned_post reads it -- but only when the caller passes no
reserve. The six HiCache host-pool sites passed the fixed 10 GiB constant
HICACHE_HOST_MEMORY_RESERVE_BYTES, so the env never reached them ("(caller)"
in the refusal). They pass None now: one reserve, read at call time; unset
stays the native 10 GiB, byte for byte.
"""
from __future__ import annotations

import ast
import os
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.mem_cache import pinned_host_budget as phb
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

CAP = 84 * (1 << 30)                 # docker --memory 84g
AVAILABLE_AT_DEATH = int(11.18e9)    # the refusal's own figure
POOLS = int(0.67e9)                  # D TP0: 0.05 + 0.35 + 0.27 GB
SRC_ROOT = os.path.dirname(os.path.dirname(phb.__file__))
SITES = ("mem_cache/pool_host/base.py", "mem_cache/pool_host/mha.py",
         "mem_cache/memory_pool_host.py")


class TheHiCachePoolsReadTheConfiguredReserve(CustomTestCase):
    def test_no_site_passes_the_fixed_constant(self):
        for rel in SITES:
            tree = ast.parse(open(os.path.join(SRC_ROOT, rel)).read())
            for node in ast.walk(tree):
                if isinstance(node, ast.keyword) and node.arg == "reserve_bytes":
                    self.assertFalse(
                        isinstance(node.value, ast.Name)
                        and node.value.id == "HICACHE_HOST_MEMORY_RESERVE_BYTES",
                        f"{rel}:{node.value.lineno} passes the fixed 10 GiB reserve")

    def _register(self, env):
        phb.clear_registered_posts()
        saved = os.environ.pop(phb.PINNED_HOST_RESERVE_ENV, None)
        try:
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(phb, "pinned_host_memory_bytes",
                                      return_value=(CAP, AVAILABLE_AT_DEATH)):
                phb.check_and_register_pinned_post(
                    name="V4 paged host pool qsa_indexer", flag="--hicache-size",
                    requested_bytes=POOLS, reserve_bytes=None)
        finally:
            phb.clear_registered_posts()
            if saved is not None:
                os.environ[phb.PINNED_HOST_RESERVE_ENV] = saved

    def test_the_container_reserve_admits_the_yarn2_pools(self):
        self._register({phb.PINNED_HOST_RESERVE_ENV: "2"})  # 11.18 - 2.15 = 9.03 GB >= 0.67

    def test_the_v4_pool_site_admits_the_yarn2_request_under_the_container_reserve(self):
        """The site D TP0 died in (qsa_indexer), built for real up to its
        check: 7 layers x 23368 pages x 4096 B = 0.67 GB against 11.18 GB
        available, env reserve 2 GiB. Stops before the allocation."""
        import torch

        from sglang.srt.mem_cache import memory_pool_host as mph

        class Admitted(Exception):
            pass

        real = mph.check_and_register_pinned_post

        def admit_then_stop(**kw):
            real(**kw)
            raise Admitted()

        phb.clear_registered_posts()
        saved = os.environ.pop(phb.PINNED_HOST_RESERVE_ENV, None)
        try:
            with mock.patch.dict(os.environ, {phb.PINNED_HOST_RESERVE_ENV: "2"}), \
                    mock.patch.object(phb, "pinned_host_memory_bytes",
                                      return_value=(CAP, AVAILABLE_AT_DEATH)), \
                    mock.patch.object(mph, "check_and_register_pinned_post",
                                      side_effect=admit_then_stop):
                with self.assertRaises(Admitted):
                    mph.DeepSeekV4PagedHostPool(
                        pool_name="qsa_indexer",
                        device_buffers=[torch.empty(1, dtype=torch.uint8)] * 7,
                        item_bytes=4096, num_host_pages=23368, slot_page_size=64,
                        pin_memory=False)
        finally:
            phb.clear_registered_posts()
            if saved is not None:
                os.environ[phb.PINNED_HOST_RESERVE_ENV] = saved

    def test_unset_is_the_native_ten_gib_and_refuses_as_before(self):
        with self.assertRaises(ValueError) as cm:
            self._register({})
        self.assertIn("10.74 GB OS reserve", str(cm.exception))
        self.assertIn("default 10 GiB", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
