"""#881: the predicate names a FLAG and reads PROCESS STATE.

``uneven_dcp_kv_replicated`` decides whether the full-attention KV pool holds
REPLICATED kv-heads or a per-rank shard -- and via ``_pool_kv_head_num`` that
decision is a pool row stride, so getting it wrong is silent corruption, not a
sizing error (see #345: Llama-3.1-8B at ``--rank-tp-ratio 3,1``, first decode
step 57% wrong at L00.o_proj while prefill sat at the noise floor).

ITS DOCSTRING SAID: "a ``--rank-tp-ratio`` base plan is installed".

That names a FLAG. The function reads ``get_tp_partition_ratios()``, which is
PROCESS STATE (a context-local overlay, then a process global), and that state
has TWO installers:

    managers/scheduler.py:12223       -- the startup path, usually from the flag
    managers/phase_flip_boot.py:1428  -- ``set_tp_partition_ratios(list(vec))``
                                         where ``vec = parse_flip_vector(...)``
                                         reads ``--phase-flip-tp-vector``,
                                         a DIFFERENT FLAG on a DIFFERENT AXIS

So on a flip boot the predicate is True while ``--rank-tp-ratio`` is None. Two
separate readers concluded the opposite from that docstring on the same day, one
of them while CORRECTING the other. The error was not carelessness; the
documentation invited it.

MEASURED ON THIS RIG (boot_w40_857strict_0826_0516):
    rank_tp_ratio=None, phase_flip_tp_vector='32,16,16', dcp_size=3 in the TP
    phase -> predicate TRUE -> pools hold get_total_num_kv_heads()=4.
    PP phase: get_tp_partition_ratios() None, but tp_size=1 -> 4//1 = 4.
    BOTH phases hold 4 heads per layer, for different reasons. There is no
    factor of four on this rig.

WHAT THIS SUITE PINS, and only the first two can actually fail on prose:
* THE CONTRACT, behaviourally: the predicate follows PROCESS STATE installed
  with no flag anywhere in sight. If it ever starts reading ``server_args``,
  that test goes red.
* that the ordering which makes the flip installer matter still holds
  (installed BEFORE the TP worker is constructed, so it reaches the pool).
* that the docstring names both installers -- a positive marker only, because
  absence of prose is not testable where a correction must quote what it
  corrects (#871c, proved the hard way).

Hermetic: process-global state under a scope. No CUDA, no pools, no boot.
"""

import inspect
import unittest

from sglang.srt.distributed.utils import (
    get_tp_partition_ratios,
    scoped_tp_partition_ratios,
    uneven_dcp_kv_replicated,
)
from sglang.test.test_utils import CustomTestCase


class TestPredicateReadsProcessState(CustomTestCase):
    # -- the contract, behaviourally ----------------------------------------

    def test_the_predicate_follows_process_state_with_no_flag_involved(self):
        """THE TEST THE DOCSTRING MADE NOBODY WRITE.

        No ServerArgs, no ``--rank-tp-ratio``, nothing parsed. A plan is
        installed directly -- the way ``phase_flip_boot`` installs it -- and the
        predicate must turn True. "I checked the flag" cannot survive this.
        """
        self.assertIsNone(
            get_tp_partition_ratios(),
            "a plan is already installed in this process; the test cannot "
            "distinguish flag from state",
        )
        self.assertFalse(uneven_dcp_kv_replicated(3))
        with scoped_tp_partition_ratios([32, 16, 16]):
            self.assertEqual(get_tp_partition_ratios(), [32, 16, 16])
            self.assertTrue(
                uneven_dcp_kv_replicated(3),
                "the predicate did not follow the installed PROCESS plan -- if "
                "it now reads a flag, every flip boot silently reverts to "
                "head-sharded pools, which is #345's corruption",
            )
        self.assertFalse(
            uneven_dcp_kv_replicated(3),
            "the plan outlived its scope",
        )

    def test_both_conjuncts_are_required(self):
        """dcp_size alone is not enough, and a plan alone is not enough."""
        with scoped_tp_partition_ratios([32, 16, 16]):
            self.assertFalse(uneven_dcp_kv_replicated(1))
            self.assertTrue(uneven_dcp_kv_replicated(2))
        self.assertFalse(uneven_dcp_kv_replicated(3))

    # -- the ordering that makes the second installer matter -----------------

    def test_the_flip_installs_the_plan_BEFORE_it_builds_the_tp_worker(self):
        """Only an install that precedes pool construction can reach the pool.

        `_pool_kv_head_num` runs inside the TP worker's model runner. If the
        install ever moves after `TpModelWorker(...)`, the predicate is False
        at the moment the pool is shaped and the heads silently become a shard.
        """
        from sglang.srt.managers import phase_flip_boot as PFB

        src = inspect.getsource(PFB)
        install = src.find("set_tp_partition_ratios(list(vec)")
        build = src.find("tp_worker = TpModelWorker(")
        self.assertGreater(install, 0, "the flip's plan install moved or was renamed")
        self.assertGreater(build, 0, "the flip's TP worker construction moved")
        self.assertLess(
            install,
            build,
            "the flip installs the shard plan AFTER building the TP worker, so "
            "the KV pool is shaped while the predicate still reads None",
        )

    def test_the_flip_installer_uses_a_DIFFERENT_flag_than_the_docstring_named(self):
        """The two axes must not merge: weight shard vs token split."""
        from sglang.srt.managers import phase_flip_boot as PFB

        self.assertIn(
            "server_args.phase_flip_tp_vector",
            inspect.getsource(PFB.parse_flip_vector),
            "parse_flip_vector no longer reads --phase-flip-tp-vector; if it "
            "now reads --rank-tp-ratio the two axes have been merged",
        )

    # -- the prose, positive markers only ------------------------------------

    def test_the_docstring_names_process_state_and_BOTH_installers(self):
        """A reader must not be able to answer 'I checked the flag'.

        Positive markers only. Asserting the OLD sentence is absent would fail
        against the fixed file, because a correction has to quote what it
        corrects -- established in #871c after making exactly that mistake.
        """
        import sglang.srt.distributed.utils as U

        doc = U.uneven_dcp_kv_replicated.__doc__ or ""
        self.assertIn("PROCESS STATE", doc.upper())
        self.assertIn("phase_flip_boot", doc)
        self.assertIn("scheduler.py", doc)
        self.assertIn("phase-flip-tp-vector", doc)


if __name__ == "__main__":
    unittest.main()
