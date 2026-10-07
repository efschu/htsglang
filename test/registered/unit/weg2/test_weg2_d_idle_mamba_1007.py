# SPDX-License-Identifier: Apache-2.0
"""D-IDLE-MAMBA-1007: group D runs the idle Mamba ledger on every K-th idle pass.

Voranalyse 07.10. (NF D): ``_check_mamba_pool`` costs 20-120 ms per idle pass
(allocator free list, a walk of the rank's radix tree); a request arriving
during it waits. ``SGLANG_WEG2_IDLE_MAMBA_CHECK_EVERY`` = K throttles it on
group D only. The ledger is rank-local -- no collective inside -- so skipping a
pass can only delay a leak report, never let ranks disagree.

Flip unchanged: the default (1) checks on every pass, and every group but D
ignores the knob.
"""
from __future__ import annotations

import inspect
import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.scheduler_components import invariant_checker as IC  # noqa: E402


def _checker(every: int) -> IC.SchedulerInvariantChecker:
    return IC.SchedulerInvariantChecker(
        is_hybrid_swa=False,
        is_hybrid_ssm=True,
        disaggregation_mode=None,
        page_size=1,
        full_tokens_per_layer=None,
        swa_tokens_per_layer=None,
        max_total_num_tokens=1,
        server_args=SimpleNamespace(dcp_size=1),
        tree_cache=SimpleNamespace(supports_mamba=lambda: True),
        token_to_kv_pool_allocator=None,
        req_to_token_pool=None,
        pool_stats_observer=None,
        get_last_batch=lambda: None,
        get_running_batch=lambda: None,
        idle_mamba_check_every=every,
    )


def _idle_passes(checker, n: int) -> list:
    """Run ``n`` idle passes the way ``Scheduler.on_idle`` asks; return the
    passes on which the Mamba ledger ran."""
    ran = []
    with mock.patch.object(IC.SchedulerInvariantChecker, "_check_full_pool",
                           return_value=(False, "full")), \
            mock.patch.object(IC.SchedulerInvariantChecker, "_check_mamba_pool",
                              return_value=(False, "mamba")) as mamba:
        for i in range(n):
            before = mamba.call_count
            checker._check_all_pools(None, check_mamba=checker.take_idle_mamba_turn())
            if mamba.call_count > before:
                ran.append(i)
    return ran


class Cadence(unittest.TestCase):
    def test_flip_unchanged_default_and_every_other_group_check_every_pass(self):
        self.assertEqual(envs.SGLANG_WEG2_IDLE_MAMBA_CHECK_EVERY.get(), 1)
        for group, every in (("D", 1), ("P", 8), ("", 8), ("p", 8), ("DUAL", 8)):
            with self.subTest(group=group, every=every):
                k = IC.idle_mamba_check_every(group=group, every=every)
                self.assertEqual(k, 1)
                self.assertEqual(_idle_passes(_checker(k), 6), list(range(6)))

    def test_group_d_checks_the_first_pass_then_every_kth(self):
        k = IC.idle_mamba_check_every(group=" d ", every=4)
        self.assertEqual(k, 4)
        self.assertEqual(_idle_passes(_checker(k), 10), [0, 4, 8])
        self.assertEqual(IC.idle_mamba_check_every(group="D", every=0), 1)

    def test_a_skipped_pass_still_runs_the_full_pool_ledger(self):
        checker = _checker(3)
        with mock.patch.object(IC.SchedulerInvariantChecker, "_check_full_pool",
                               return_value=(True, "full leak")) as full, \
                mock.patch.object(IC.SchedulerInvariantChecker, "_check_mamba_pool",
                                  return_value=(False, "mamba")):
            checker.take_idle_mamba_turn()
            leak, msgs = checker._check_all_pools(None, check_mamba=checker.take_idle_mamba_turn())
        self.assertTrue(leak)
        self.assertEqual(msgs, ["full leak"])
        self.assertEqual(full.call_count, 1)


class Wiring(unittest.TestCase):
    """Delivery, not presence: on_idle asks the checker for the turn, and the
    scheduler resolves K from its own group."""

    def test_on_idle_and_init_pass_the_cadence(self):
        from sglang.srt.managers.scheduler import Scheduler

        self.assertIn("check_mamba=self.invariant_checker.take_idle_mamba_turn()",
                      inspect.getsource(Scheduler.on_idle))
        init_src = inspect.getsource(Scheduler.init_invariant_checker)
        self.assertIn('group=os.environ.get("SGLANG_WEG2_GROUP", "")', init_src)
        self.assertIn("envs.SGLANG_WEG2_IDLE_MAMBA_CHECK_EVERY.get()", init_src)


if __name__ == "__main__":
    unittest.main()
