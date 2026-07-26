"""The token-vector honesty guard must be REACHABLE from the boot path.

THE DEFECT
----------
``resolve_cp_token_ratios`` carries a deliberate honesty guard: a
``SGLANG_UNEVEN_TOKEN_VECTOR`` with no engaged shard plan is REJECTED rather
than silently ignored (measured on Qwen3.5-2B TP=2/DCP=2 TOKVEC=2,1: boots
green, output token-identical to TP=1, zero uneven-machinery log lines -- a
configured-looking server doing nothing that was asked).

That guard was tested by calling the resolver directly, and it works there.
The BOOT PATH never called it in the very case it exists for:
``configure_scheduler_process`` gated the resolver on ``base_plan is not
None`` -- i.e. on the presence of the plan whose ABSENCE is what the guard
reports. Without a ``--rank-tp-ratio`` the resolver was never invoked, so on a
real server the vector-without-a-plan case fell through to a foreign guard
(the flashinfer no-op rejection, which speaks about the backend rather than
the vector) or, on the Triton backend, to branch (2) of
``reject_unsupported_dcp_geometry`` -- which is itself boot-dead, since it
keys on an INSTALLED token vector and nothing ever installed one without a
plan.

A guard that only fires in its own unit test is not a guard.

WHAT IS PINNED HERE
-------------------
 1. Boot path, vector set, no plan -> the honest ValueError from
    ``resolve_cp_token_ratios``, naming the vector and the missing plan.
 2. The same with the weighted-DCP switch off: the token vector is the
    token-vector machinery's OWN state, so its own resolver must run.
 3. The uneven lane (a non-uniform plan + the weighted switch) is UNCHANGED:
    same vector installed, no new rejection. The freshly validated G2-G6 /
    #169 boots run through this branch and must not move.
 4. The plainest default (no plan, no vector) is UNCHANGED: no rejection, no
    vector installed.
 5. Deliberately out of scope: a non-uniform plan + a vector while the
    weighted switch is off. That combination is inert today and stays inert
    -- changing it would change the plan lane, which this fix must not.

CPU only: the boot function is driven with a stub server_args and stopped at
``setproctitle``, which is the first statement after the block under test.
No device, no process group, no model.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# THE REAL BOOT FUNCTION -- not a re-implementation of its gate.
import sglang.srt.distributed.utils as du  # noqa: E402
from sglang.srt.managers import scheduler as sched_mod  # noqa: E402

_TOKVEC = "SGLANG_UNEVEN_TOKEN_VECTOR"
_WEIGHTED = "SGLANG_UNEVEN_DCP_WEIGHTED"


class _ReachedProcTitle(Exception):
    """Raised in place of setproctitle: the boot got past the block."""


def _server_args(*, plan, weighted, dcp_size=2, tp_size=2):
    sa = SimpleNamespace(
        rank_tp_ratio=plan,
        rank_mlp_ratio=None,
        rank_moe_ratio=None,
        rank_vocab_ratio=None,
        rank_kv_ratio=None,
        rank_kv_capacity_seed=None,
        rank_gpu_memory_mib=None,
        rank_gpu_id=None,
        model_path=None,
        dcp_size=dcp_size,
        tp_size=tp_size,
        pp_size=1,
        attn_cp_size=1,
        moe_dp_size=1,
        ep_size=1,
        nnodes=1,
        weightless_kv_fastlane=False,
    )
    sa.uneven_weighted_dcp_enabled = lambda: weighted
    sa.apply_rank_memory_budget = lambda tp_rank: None
    return sa


def _boot(sa):
    """Run the real ``configure_scheduler_process`` up to ``setproctitle``.

    Returns the token vector the boot installed (or None). Re-raises whatever
    the block under test raised.
    """

    def _boom(*a, **kw):
        raise _ReachedProcTitle

    with mock.patch.object(sched_mod, "kill_itself_when_parent_died", lambda: None):
        with mock.patch.object(sched_mod.setproctitle, "setproctitle", _boom):
            try:
                sched_mod.configure_scheduler_process(
                    sa,
                    gpu_id=0,
                    tp_rank=0,
                    attn_cp_rank=0,
                    moe_dp_rank=0,
                    moe_ep_rank=0,
                    pp_rank=0,
                    dp_rank=0,
                )
            except _ReachedProcTitle:
                pass
    return du.get_cp_token_ratios()


class BootTokenVectorGuardTest(CustomTestCase):
    def setUp(self):
        self._saved_vec = du.get_cp_token_ratios()
        self._saved_plan = du.get_tp_partition_ratios()
        du.set_cp_token_ratios(None)

    def tearDown(self):
        du.set_tp_partition_ratios(self._saved_plan)
        du.set_cp_token_ratios(self._saved_vec)

    def test_vector_without_a_plan_is_rejected_on_the_boot_path(self):
        env = {_TOKVEC: "2,1", _WEIGHTED: "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            with self.assertRaises(ValueError) as ei:
                _boot(_server_args(plan=None, weighted=True))
        msg = str(ei.exception)
        self.assertIn("SGLANG_UNEVEN_TOKEN_VECTOR", msg)
        self.assertIn("silently ignored", msg)
        self.assertIn("--rank-tp-ratio", msg)

    def test_the_rejection_does_not_need_the_weighted_switch(self):
        """The vector is the token-vector machinery's own state.

        Gating its resolver on a SECOND opt-in is how the guard became
        unreachable in the first place; a vector set with dcp_size > 1 must
        always be resolved, and may be refused.
        """
        env = {_TOKVEC: "3,1"}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop(_WEIGHTED, None)
            with self.assertRaises(ValueError) as ei:
                _boot(_server_args(plan=None, weighted=False))
        self.assertIn("silently ignored", str(ei.exception))

    def test_dcp_size_one_makes_the_vector_inert_not_an_error(self):
        env = {_TOKVEC: "2,1", _WEIGHTED: "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertIsNone(
                _boot(_server_args(plan=None, weighted=True, dcp_size=1, tp_size=1))
            )

    def test_the_uneven_lane_is_unchanged(self):
        """#169 / G2-G6 run here: plan + weighted switch, no env vector."""
        with mock.patch.dict(os.environ, {_WEIGHTED: "1"}, clear=False):
            os.environ.pop(_TOKVEC, None)
            self.assertEqual(_boot(_server_args(plan=[2, 1], weighted=True)), [2, 1])

    def test_the_uneven_lane_still_honours_an_explicit_env_vector(self):
        env = {_TOKVEC: "5,3", _WEIGHTED: "1"}
        with mock.patch.dict(os.environ, env, clear=False):
            self.assertEqual(_boot(_server_args(plan=[2, 1], weighted=True)), [5, 3])

    def test_the_plainest_default_installs_nothing_and_raises_nothing(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop(_TOKVEC, None)
            os.environ.pop(_WEIGHTED, None)
            self.assertIsNone(_boot(_server_args(plan=None, weighted=False)))
            self.assertIsNone(_boot(_server_args(plan=[2, 1], weighted=False)))

    def test_a_plan_with_the_weighted_switch_off_stays_inert(self):
        """Deliberately NOT changed: out of scope for this fix.

        With a plan installed but the weighted owner rule switched off, an env
        vector is inert today. Making that a rejection would alter the plan
        lane, which the fix must leave alone. Pinned so the decision is
        visible rather than accidental.
        """
        env = {_TOKVEC: "5,3"}
        with mock.patch.dict(os.environ, env, clear=False):
            os.environ.pop(_WEIGHTED, None)
            self.assertIsNone(_boot(_server_args(plan=[2, 1], weighted=False)))


if __name__ == "__main__":
    unittest.main()
