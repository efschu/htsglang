# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""#824 W4(b): a zero-iteration armed window must not move the slot.

THE MEASUREMENT (boot_827_review_0823_0910c), one line before the ring
went silent for 31 s:

    PHASE-FLIP PASS-CLOCK across the armed window: rank 0 ran 0 slot
    iteration(s) (armed at mb_id=0, disarmed at mb_id=1)

Zero iterations, and the slot still advanced.

WHY THAT IS A DEFECT AND NOT AN ACCOUNTING ODDITY. In steady state the
pass loop is paced by the request chain: one blocking chain receive per
slot iteration, so rank k's i-th iteration is rank k-1's i-th and the slot
indices cannot diverge. An ARMED rank admits nothing -- ``_pull_raw_reqs``
returns [] before touching the chain -- so its iterations consume no chain
message. If it advances the slot anyway, the slot moves while the thing
that paces it does not, and ranks that abandon on their own clock re-enter
the pipeline on different slots. From then on stage k's hidden states pair
with stage k+1's batch by an index the two no longer agree on: #631
defect Q, whose two earlier disposals (corpses R and S) each cost a boot.

``_pp_flip_hold_slot`` is the existing fix for exactly this, and it works
-- for windows long enough to reach it. It engages only once the armed
window has run the pipeline DRY, which takes ``pp_loop_size`` parked
iterations. A window that abandons sooner never gets there, and boot_827's
window was one iteration long.

scheduler_pp_mixin's own note predicted the remedy before it was built:
"an armed rank must not ADVANCE its slot loop while it is doing no
pipeline work."

CPU-only. No gloo, no CUDA, no scheduler construction.
"""

from types import SimpleNamespace

import pytest


def _tick_harness(armed=True, enabled=True, loop_size=3):
    """A holder carrying only what _pp_flip_pass_tick reads.

    Mirrors test_phase_flip_counters._tick_harness; counters are left None
    so the tick takes its counter-free path and this test measures the slot
    decision alone.
    """
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    class S:
        pass

    s = S()
    s.server_args = SimpleNamespace(enable_phase_flip=enabled)
    s.pp_flip_counters = None
    s.pp_loop_size = loop_size
    s._armed = armed
    s.pp_phase_flip_armed = lambda: s._armed
    return s, SchedulerPPMixin._pp_flip_pass_tick.__get__(s, S)


def test_zero_iteration_window_asks_for_the_arm_slot_back():
    """The boot_827 sequence, tick for tick."""
    s, tick = _tick_harness()

    tick(0)  # rising edge: armed at mb_id=0
    assert s._pp_flip_arm_mb_id == 0
    assert s._pp_flip_armed_passes == 0

    s._armed = False
    tick(1)  # falling edge one slot later, having run 0 iterations

    assert getattr(s, "_pp_flip_resume_slot", None) == 0, (
        "the window ran zero iterations and still let the slot advance "
        "0 -> 1; this is the boot_827 line, and the ranks resume on "
        "different slots from here"
    )


def test_a_window_that_ran_iterations_is_left_to_the_hold():
    """Not a blanket rewind. Longer windows reach _pp_flip_hold_slot, which
    already guarantees the arm slot, and second-guessing it here would move
    a slot the pipeline is still draining."""
    s, tick = _tick_harness()

    tick(0)  # rising edge
    tick(1)  # a real armed iteration -> passes = 1
    assert s._pp_flip_armed_passes == 1

    s._armed = False
    tick(2)

    assert getattr(s, "_pp_flip_resume_slot", None) is None, (
        "a window that ran iterations must be left to the hold, not "
        "rewound from here"
    )


def test_no_restore_when_the_slot_never_moved():
    """Abandoning on the slot it armed on needs no correction, and asking
    for one would make the loop restart a slot for nothing."""
    s, tick = _tick_harness()

    tick(2)
    s._armed = False
    tick(2)

    assert getattr(s, "_pp_flip_resume_slot", None) is None


def test_the_flip_being_off_costs_nothing():
    s, tick = _tick_harness(enabled=False)
    tick(0)
    s._armed = False
    tick(1)
    assert getattr(s, "_pp_flip_resume_slot", None) is None
    assert getattr(s, "_pp_flip_arm_mb_id", None) is None


def test_the_loop_returns_to_the_arm_slot_without_advancing():
    """The consumer half: the event loop must restart the body on the
    restored slot rather than carrying on and incrementing.

    Models _event_loop_pp_body's slot loop -- the while/continue shape it
    uses precisely so a slot can be held or restored -- and asserts the
    visited sequence.
    """
    s, tick = _tick_harness()
    visited = []

    mb_id = 0
    loop_size = 3
    guard = 0
    while mb_id < loop_size:
        guard += 1
        assert guard < 50, "slot loop failed to terminate"
        tick(mb_id)
        resume_slot = getattr(s, "_pp_flip_resume_slot", None)
        if resume_slot is not None:
            s._pp_flip_resume_slot = None
            if int(resume_slot) != mb_id:
                mb_id = int(resume_slot)
                continue
        visited.append(mb_id)
        if mb_id == 0 and s._armed:
            # boot_827's window: the flip abandons within the very
            # iteration it armed on, so the falling edge lands one slot
            # later having run ZERO armed iterations.
            s._armed = False
        mb_id += 1

    # Slot 0 is armed on; at slot 1 the abandon is noticed and the rank is
    # sent BACK to slot 0 -- the slot it armed on -- without that visit
    # having advanced anything. Then the pass loop runs on normally.
    assert visited == [0, 0, 1, 2], visited
