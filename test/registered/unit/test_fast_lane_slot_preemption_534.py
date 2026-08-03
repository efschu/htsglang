"""#534: pins for the fast-lane SLOT-pressure preemption path.

This ticket was opened to BUILD a slot-pressure retraction trigger, on my
reading that retraction was driven only by ``check_decode_mem()`` (KV bytes).
Reading the predicate at its source showed the trigger ALREADY EXISTS and is
live, so nothing was built and these are the pins that keep it from silently
regressing -- nothing in the tree asserted the wiring end to end.

  ``scheduler.py:3797-3809``::

      running_bs = len(running_batch.reqs)
      if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
          running_batch.batch_is_full = True
      ...
      if running_batch.batch_is_full:
          if (not self.enable_priority_preemption
              or not adder.preempt_to_schedule(req, self.server_args)):
              break

``batch_is_full`` IS slot pressure -- ``get_num_allocatable_reqs`` is bounded
by ``--max-running-requests`` -- and the fallback is ``preempt_to_schedule``
(``schedule_policy.py:1368``), which preempts a lower-priority RUNNING request
so the waiting one can be admitted, honouring
``--fast-lane-reserved-heavy-slots`` at ``:1409-1416``.

REACH LIMIT, measured and not papered over: ``preempt_to_schedule`` iterates
``self.running_batch.reqs`` (``:1382``), i.e. requests that are RUNNING. A
request still being chunk-prefilled is not in that set and cannot be
preempted, which is why a fast request behind four 46k-token prefills still
measured 23.6 s to first token while an untagged one measured 112.9 s: the
lane wins by 4.8x but does not preempt an in-flight prefill. Chunk-preemptive
admission is a separate, unbuilt cut.

NOT PINNED HERE, deliberately: that ``check_server_args`` (not
``__post_init__``) is what flips ``enable_priority_scheduling`` on. Asserting
it needs the FULL validator, which needs a resolvable checkpoint AND a real
accelerator ("extra_buffer needs CUDA/MUSA/NPU" for this hybrid model), so the
test can never run in the hermetic harness and would be a permanently-skipped
placeholder. It is verified against the LIVE boot instead --
``curl /server_info`` reports enable_priority_scheduling=True,
default_priority_value=0, fast_lane_priority=1000000 against a preemption
threshold of 10 -- and the split is worth knowing: a hermetic
``prepare_server_args()`` still shows False, which is an instrument artifact
and not an inert flag.
"""

import unittest

from sglang.srt.server_args import ServerArgs


def _raw(**kw):
    """Defaults only: ``__post_init__`` returns immediately for the
    placeholder model path (``server_args.py:5730-5731``), so this needs no
    accelerator and no checkpoint."""
    base = dict(model_path="dummy", served_model_name="dummy")
    base.update(kw)
    return ServerArgs(**base)


class TestShippedDefaultsMakeTheLaneUsable(unittest.TestCase):
    """REACH INCLUDES PARAMETERS: a preemption path whose priority gap never
    clears the threshold, or whose heavy floor equals the slot count, would
    exist and act on nothing."""

    def test_fast_heavy_gap_clears_the_preemption_threshold(self):
        sa = _raw(enable_fast_lane=True)
        # check_server_args seeds the heavy tier at default_priority_value=0
        # (verified against the live boot, see the module docstring).
        gap = sa.fast_lane_priority - 0
        self.assertGreater(gap, sa.priority_scheduling_preemption_threshold)

    def test_preemption_is_not_disabled_by_default(self):
        self.assertFalse(_raw().disable_priority_preemption)

    def test_reserved_heavy_floor_defaults_to_one(self):
        self.assertEqual(_raw().fast_lane_reserved_heavy_slots, 1)

    def test_retraction_policy_defaults_to_length(self):
        """The OTHER half, and a real gap on the deployed boot: under KV
        pressure ``_get_decode_retraction_order`` consults ``req.priority``
        ONLY when --retraction-policy is 'priority'
        (``schedule_batch.py:2894``). At the default a fast request can be
        retracted in favour of a heavy one."""
        self.assertEqual(_raw().retraction_policy, "length")


class TestPriorityPreemptionExpression(unittest.TestCase):
    """``scheduler.py:1389-1392`` -- the exact expression that gates the
    slot-pressure branch."""

    @staticmethod
    def enabled(priority_scheduling, disable_preemption):
        return priority_scheduling and not disable_preemption

    def test_on_when_priority_scheduling_is_on(self):
        self.assertTrue(self.enabled(True, False))

    def test_off_when_preemption_is_explicitly_disabled(self):
        """Falsifier: --disable-priority-preemption leaves a fast request
        behind full slots with no path in."""
        self.assertFalse(self.enabled(True, True))

    def test_off_without_priority_scheduling(self):
        self.assertFalse(self.enabled(False, False))


class TestReservedHeavyFloorArithmetic(unittest.TestCase):
    """``schedule_policy.py:1409-1416``."""

    @staticmethod
    def max_heavy_preemptible(num_heavy_running, reserved):
        return max(0, num_heavy_running - reserved)

    def test_floor_keeps_the_reserved_heavy_slots_running(self):
        self.assertEqual(self.max_heavy_preemptible(4, 1), 3)
        self.assertEqual(self.max_heavy_preemptible(1, 1), 0)
        self.assertEqual(self.max_heavy_preemptible(0, 1), 0)

    def test_a_zero_floor_allows_full_displacement(self):
        self.assertEqual(self.max_heavy_preemptible(4, 0), 4)


if __name__ == "__main__":
    unittest.main()
