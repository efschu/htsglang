"""#701: the decode concurrency cap was a PREFILL-layout bound.

``pp_max_micro_batch_size`` auto-computes as ``max_running_requests //
pp_size``. That is right for classic PP, where the stages run micro-batches of
one batch and each may hold only its share. It is wrong under the phase flip,
because DECODE does not run in the PP layout -- it runs in the TP layout, which
has no pipeline to divide by.

MEASURED, which is why this is a bug and not a preference. On this deployment
(max_running_requests=4, pp_size=3) the default computed to max(4 // 3, 1) = 1.
Under a sustained depth-5 load the decode concurrency never exceeded 2:

    #running-req on 1,813 prefill rounds:  0 -> 973,  1 -> 792,  2 -> 48

against a configured ceiling of 4, with 0 BOTH-BLOCKED events, 66 decode
batches and 9 completions in 90 s. Nothing was deadlocked. The scheduler was
correctly obeying a cap of one.
"""

import unittest

from sglang.test.test_utils import CustomTestCase


class TestPPMicroBatchCap701(CustomTestCase):
    def test_the_scheduler_uses_the_helper(self):
        """Pin the wiring, or the fix is desk-written and never runs.

        The call site is ``init_model_worker``, NOT ``__init__`` -- the first
        version of this test asserted the latter and went red, which is the
        same placement trap that broke init_parked_decode_set earlier (#677).
        Pinning the wrong method is how a fix gets believed without running.
        """
        import inspect

        from sglang.srt.managers import scheduler as scheduler_mod

        src = inspect.getsource(scheduler_mod.Scheduler.init_model_worker)
        self.assertIn("default_pp_micro_batch_size", src)
        self.assertNotIn(
            "self.max_running_requests // self.ps.pp_size",
            src,
            "the old inline division must be gone, not shadowed",
        )

    def test_the_call_site_is_actually_reached_from_init(self):
        """And that method must still be called by __init__, or the wiring pin
        above is satisfied by dead code."""
        import inspect

        from sglang.srt.managers import scheduler as scheduler_mod

        self.assertIn(
            "init_model_worker",
            inspect.getsource(scheduler_mod.Scheduler.__init__),
        )


if __name__ == "__main__":
    unittest.main()
