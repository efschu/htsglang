# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# ==============================================================================
"""nf-poolleak-1011, Commit B -- USER ORDER 11.10.2026: "er soll warnen und weitermachen".

NF int24e died twice on 11.10. because ``SchedulerInvariantChecker._report_leak`` raised
``ValueError`` under ``FLLIPER_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1`` (the default):
``RANK_EXCEPTION`` -> group death, over ONE double-owned 64-token page. The leak check
never kills the server any more: STRICT=1 writes ONE warning line with the full ledger
(marker ``PDFLIP-POOL-LEAK WARN``, ``overshoot_pages= free= evictable= total=``) and the idle
pass goes on. STRICT=0 is the pre-existing path, byte for byte.

Hermetic: real PagedTokenToKVPoolAllocator / SchedulerInvariantChecker over CPU tensors,
and the REAL ``Scheduler.on_idle`` driven over a mock ``self``.
"""

import logging
import os
import unittest
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from flliper.srt.environ import envs
from flliper.srt.managers.scheduler_components.invariant_checker import (
    LEAK_WARN_LOG_INTERVAL,
    SchedulerInvariantChecker,
)
from flliper.srt.managers.scheduler_components.pool_stats_observer import PoolStats
from flliper.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from flliper.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from flliper.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from flliper.test.ci.ci_register import register_cpu_ci
from flliper.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PS = 64
POOL = PS * 64  # 64 pages
MARKER = "PDFLIP-POOL-LEAK WARN"


class _Tree:
    def __init__(self, evictable):
        self._e = evictable

    def protected_size(self):
        return 0

    def full_protected_size(self):
        return 0

    def supports_mamba(self):
        return False

    def is_tree_cache(self):
        return True

    def evictable_size(self):
        return self._e

    def leak_census_str(self):
        return "TREE CENSUS (test)"


class _Observer:
    def session_held_tokens(self):
        return 0

    def session_held_full_tokens(self):
        return 0

    def session_held_req_count(self):
        return 0


class _Args:
    dcp_size = 1


def _checker(evictable_tokens, held_pages=40):
    """A pool whose tree 'holds' ``held_pages`` pages while the checker is told
    ``evictable_tokens``. ``evictable_tokens > held_pages*PS`` is a SURPLUS (a page counted
    twice -- the 11.10. NF shape), ``<`` a DEFICIT (rows nobody owns)."""
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy", page_size=PS))
    pool = ReqToTokenPool(size=2, max_context_len=64, device="cpu", enable_memory_saver=False)
    kv = MHATokenToKVPool(
        size=POOL, page_size=PS, dtype=torch.bfloat16, head_num=2, head_dim=64,
        layer_num=2, device="cpu", enable_memory_saver=False,
    )
    alloc = PagedTokenToKVPoolAllocator(
        size=POOL, page_size=PS, dtype=torch.bfloat16, device="cpu", kvcache=kv, need_sort=False
    )
    alloc.alloc(held_pages * PS)
    chk = SchedulerInvariantChecker(
        is_hybrid_swa=False, is_hybrid_ssm=False, disaggregation_mode=None, page_size=PS,
        full_tokens_per_layer=None, swa_tokens_per_layer=None, max_total_num_tokens=POOL,
        server_args=_Args(), tree_cache=_Tree(evictable_tokens), token_to_kv_pool_allocator=alloc,
        req_to_token_pool=pool, pool_stats_observer=_Observer(),
        get_last_batch=lambda: None, get_running_batch=lambda: None,
    )
    ps = PoolStats(
        full_num_used=0, full_token_usage=0.0,
        full_available_size=alloc.available_size(), full_evictable_size=evictable_tokens,
    )
    return chk, ps


class TestStrictLeakWarnsAndContinues1011(CustomTestCase):
    def test_strict_surplus_warns_with_the_full_ledger_and_does_not_raise(self):
        """RED WITH THE RAISE BACK: ValueError -> RANK_EXCEPTION on NF."""
        chk, ps = _checker(evictable_tokens=41 * PS)  # one page too many in the ledger
        leak, msgs = chk._check_all_pools(ps)
        self.assertTrue(leak, "the ledger must still DETECT the surplus")
        with envs.FLLIPER_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.override(True):
            with self.assertLogs(level=logging.WARNING) as cm:
                chk._report_leak("pool", "\n".join(msgs))  # must not raise
        text = "\n".join(cm.output)
        self.assertIn(MARKER, text)
        self.assertIn("overshoot_pages=+1", text)
        self.assertIn(f"overshoot_tokens=+{PS}", text)
        self.assertIn(f"free={ps.full_available_size}", text)
        self.assertIn(f"evictable={41 * PS}", text)
        self.assertIn(f"total={POOL}", text)
        self.assertIn("pool memory leak detected!", text, "the original message stays in the line")
        self.assertEqual(chk.count_memory_leak_warnings, 1)

    def test_strict_deficit_warns_with_a_negative_overshoot(self):
        chk, ps = _checker(evictable_tokens=39 * PS)  # one page nobody owns
        leak, msgs = chk._check_all_pools(ps)
        self.assertTrue(leak)
        with envs.FLLIPER_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.override(True):
            with self.assertLogs(level=logging.WARNING) as cm:
                chk._report_leak("pool", "\n".join(msgs))
        self.assertIn("overshoot_pages=-1", "\n".join(cm.output))

    def test_non_strict_is_the_unchanged_path(self):
        """STRICT=0: same text as before (no marker), same counter, still no raise."""
        chk, ps = _checker(evictable_tokens=41 * PS)
        leak, msgs = chk._check_all_pools(ps)
        with envs.FLLIPER_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.override(False):
            with self.assertLogs(level=logging.WARNING) as cm:
                chk._report_leak("pool", "\n".join(msgs))
            self.assertTrue(chk.leak_detail_due(), "STRICT=0: the census is always paid, as before")
        text = "\n".join(cm.output)
        self.assertNotIn(MARKER, text)
        self.assertIn("pool memory leak detected! [full] total=", text)
        self.assertEqual(chk.count_memory_leak_warnings, 1)

    def test_a_persistent_leak_is_one_line_per_interval_not_one_per_idle_pass(self):
        chk, ps = _checker(evictable_tokens=41 * PS)
        leak, msgs = chk._check_all_pools(ps)
        passes = 2 * LEAK_WARN_LOG_INTERVAL + 5
        due = 0
        with envs.FLLIPER_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.override(True):
            with self.assertLogs(level=logging.WARNING) as cm:
                for _ in range(passes):
                    due += bool(chk.leak_detail_due())
                    chk._report_leak("pool", "\n".join(msgs))
        lines = [o for o in cm.output if MARKER in o]
        self.assertEqual(len(lines), 3, lines)  # n=1, n=1001, n=2001
        self.assertEqual(due, 3, "the tree census is paid exactly when the line is written")
        self.assertEqual(chk.count_memory_leak_warnings, passes)
        self.assertIn("n=1 ", lines[0])
        self.assertIn(f"n={LEAK_WARN_LOG_INTERVAL + 1} ", lines[1])

    def test_the_log_interval_is_the_one_raise_error_or_warn_defaults_to(self):
        """The strict path leaves ``log_interval`` at raise_error_or_warn's default; the census
        gate (``leak_detail_due``) uses LEAK_WARN_LOG_INTERVAL. They must stay one number."""
        import inspect

        from flliper.srt.utils.common import raise_error_or_warn

        self.assertEqual(
            inspect.signature(raise_error_or_warn).parameters["log_interval"].default,
            LEAK_WARN_LOG_INTERVAL,
        )

    def test_a_balanced_ledger_reports_nothing(self):
        chk, ps = _checker(evictable_tokens=40 * PS)
        leak, _ = chk._check_all_pools(ps)
        self.assertFalse(leak)


class TestSchedulerLoopGoesOnAfterALeak1011(CustomTestCase):
    """The REAL ``Scheduler.on_idle`` over a mock ``self``: the leak pass is followed by the
    rest of the idle housekeeping, on every pass, and the loop is never left by an exception."""

    def _idle_self(self, chk, ps):
        sched = mock.MagicMock()
        sched.enable_hisparse = False
        # 27B line: Scheduler.on_idle also calls dual_p_kv_stage.on_idle(self); None = no P actor attached (a no-op)
        sched.tp_worker.model_runner.dual_p_kv = None
        sched.is_fully_idle.return_value = True
        sched.pdflip_dormant = False
        sched.weight_updater = None
        sched.idle_census = None
        sched.enable_unified_memory = False
        sched.token_to_kv_pool_allocator.reclaim_abandoned_free_group = None
        sched._idle_census_control_hold.return_value = False
        sched.pool_stats_observer.get_pool_stats.return_value = ps
        sched.tree_cache = chk.tree_cache
        sched.invariant_checker = chk
        return sched

    def test_on_idle_survives_a_leak_under_strict_and_finishes_every_pass(self):
        from flliper.srt.managers.scheduler import Scheduler

        chk, ps = _checker(evictable_tokens=39 * PS)  # a DEFICIT: decided on the O(1) ledger, every pass
        sched = self._idle_self(chk, ps)
        passes = 5
        with envs.FLLIPER_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.override(True):
            with self.assertLogs(level=logging.WARNING):
                for _ in range(passes):
                    Scheduler.on_idle(sched)  # RED with the raise back: ValueError on pass 1
        self.assertEqual(chk.count_memory_leak_warnings, passes)
        self.assertEqual(sched.maybe_sleep_on_idle.call_count, passes, "the pass ran to its end")
        self.assertEqual(sched.publish_load_snapshot.call_count, passes)

    def test_on_idle_pays_the_tree_census_only_when_the_line_is_written(self):
        from flliper.srt.managers.scheduler import Scheduler

        chk, ps = _checker(evictable_tokens=39 * PS)  # a DEFICIT: decided on the O(1) ledger, every pass
        sched = self._idle_self(chk, ps)
        census = mock.MagicMock(return_value="TREE CENSUS (test)")
        sched.tree_cache = mock.MagicMock(spec=["leak_census_str", "protected_size",
                                                 "full_protected_size", "supports_mamba",
                                                 "is_tree_cache", "evictable_size"])
        sched.tree_cache.leak_census_str = census
        sched.tree_cache.protected_size.return_value = 0
        sched.tree_cache.evictable_size.return_value = 39 * PS
        chk.tree_cache = sched.tree_cache
        with envs.FLLIPER_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.override(True):
            with self.assertLogs(level=logging.WARNING):
                for _ in range(7):
                    Scheduler.on_idle(sched)
        self.assertEqual(census.call_count, 1, "seven idle passes, one report, one tree walk")

    def test_on_idle_non_strict_walks_the_tree_on_every_pass_as_before(self):
        from flliper.srt.managers.scheduler import Scheduler

        chk, ps = _checker(evictable_tokens=39 * PS)  # a DEFICIT: decided on the O(1) ledger, every pass
        sched = self._idle_self(chk, ps)
        census = mock.MagicMock(return_value="TREE CENSUS (test)")
        sched.tree_cache = mock.MagicMock(spec=["leak_census_str", "protected_size",
                                                 "full_protected_size", "supports_mamba",
                                                 "is_tree_cache", "evictable_size"])
        sched.tree_cache.leak_census_str = census
        sched.tree_cache.protected_size.return_value = 0
        sched.tree_cache.evictable_size.return_value = 39 * PS
        chk.tree_cache = sched.tree_cache
        with envs.FLLIPER_ENABLE_STRICT_MEM_CHECK_DURING_IDLE.override(False):
            with self.assertLogs(level=logging.WARNING):
                for _ in range(7):
                    Scheduler.on_idle(sched)
        self.assertEqual(census.call_count, 7)


if __name__ == "__main__":
    unittest.main()
