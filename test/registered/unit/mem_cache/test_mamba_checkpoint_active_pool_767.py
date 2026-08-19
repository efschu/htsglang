# Copyright 2025 SGLang Team
# Licensed under the Apache License, Version 2.0
"""#767 residual: checkpoint state copies must read the ACTIVE phase's pool.

THE SAME-PHASE POISON. The radix caches copy a request's mamba state into a
checkpoint slot through ``cache.req_to_token_pool.mamba_pool`` -- the tree's
bound pool, which under a phase-flip build is the PRIMARY PP pool forever.
A request computing in the TP phase has its real state bytes in the TP
stack's pool; the copy therefore duplicated whatever stale bytes the PP
pool still held at that slot (measured 2026-08-19: a probe about kites
answered with a foreign river essay -- the PP slot's previous occupant).
Bookkeeping (allocator, translation, mappings) stays single-authority on
the bound pool; only STATE BYTES move to the active pool's tensors.

Red-first against a tree with no resolver installed: the helper must fall
back to the bound pool, and the flip boot's installer must resolve by
``scheduler.phase_flip_active_stack``.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace


def _pool(tag):
    return SimpleNamespace(tag=tag)


class TestTheHelperResolvesTheActivePool(unittest.TestCase):
    def _helper(self):
        from sglang.srt.mem_cache.mamba_state_pool import active_mamba_state_pool

        return active_mamba_state_pool

    def test_without_a_resolver_the_bound_pool_is_used(self):
        cache = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(mamba_pool=_pool("bound"))
        )
        self.assertEqual(self._helper()(cache).tag, "bound")

    def test_an_installed_resolver_wins(self):
        active = _pool("active")
        cache = SimpleNamespace(
            req_to_token_pool=SimpleNamespace(mamba_pool=_pool("bound")),
            phase_active_mamba_pool=lambda: active,
        )
        self.assertIs(self._helper()(cache), active)


class TestTheFlipBootInstallsTheResolver(unittest.TestCase):
    def _scheduler(self):
        pp_pool = _pool("pp")
        tp_pool = _pool("tp")
        tree = SimpleNamespace()
        return SimpleNamespace(
            tree_cache=tree,
            req_to_token_pool=SimpleNamespace(mamba_pool=pp_pool),
            phase_flip_stacks=SimpleNamespace(
                tp_worker=SimpleNamespace(
                    model_runner=SimpleNamespace(
                        req_to_token_pool=SimpleNamespace(mamba_pool=tp_pool)
                    )
                )
            ),
            phase_flip_active_stack=None,
        ), pp_pool, tp_pool

    def _install(self, scheduler):
        from sglang.srt.managers.gdn_flip_mover import (
            install_phase_aware_mamba_state_pool,
        )

        install_phase_aware_mamba_state_pool(scheduler)

    def test_pp_standing_resolves_the_primary_pool(self):
        scheduler, pp_pool, _ = self._scheduler()
        self._install(scheduler)
        scheduler.phase_flip_active_stack = "pp"
        self.assertIs(scheduler.tree_cache.phase_active_mamba_pool(), pp_pool)

    def test_tp_standing_resolves_the_tp_stacks_pool(self):
        scheduler, _, tp_pool = self._scheduler()
        self._install(scheduler)
        scheduler.phase_flip_active_stack = "tp"
        self.assertIs(scheduler.tree_cache.phase_active_mamba_pool(), tp_pool)

    def test_before_the_first_cutover_the_primary_pool_is_active(self):
        # phase_flip_active_stack is unset until the first cutover; the
        # boot phase computes on the primary stack.
        scheduler, pp_pool, _ = self._scheduler()
        self._install(scheduler)
        self.assertIs(scheduler.tree_cache.phase_active_mamba_pool(), pp_pool)


class TestTheCheckpointCopySitesUseTheHelper(unittest.TestCase):
    """The two writer sites must route state bytes through the helper.

    Source-level pin rather than a full component harness: the sites are
    deep inside cache_unfinished flows whose surrounding machinery is
    exercised by the existing radix suites; what THIS defect needs pinned
    is that neither site reaches ``req_to_token_pool.mamba_pool.copy_from``
    directly again.
    """

    def _source(self, module):
        import importlib
        import inspect

        return inspect.getsource(importlib.import_module(module))

    def test_the_unified_component_routes_through_the_helper(self):
        src = self._source(
            "sglang.srt.mem_cache.unified_cache_components.mamba_component"
        )
        self.assertIn("active_mamba_state_pool", src)
        self.assertNotIn("req_to_token_pool.mamba_pool.copy_from", src)

    def test_the_mamba_radix_cache_routes_through_the_helper(self):
        src = self._source("sglang.srt.mem_cache.mamba_radix_cache")
        self.assertIn("active_mamba_state_pool", src)
        self.assertNotIn("req_to_token_pool.mamba_pool.copy_from", src)


if __name__ == "__main__":
    unittest.main()
