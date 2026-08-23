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
"""#829: an arm slot must not outlive the ring it names.

THE DEATH (boot_window2_0823_1554, integ/round2 @ f9d7637f04). The tree
came up, served, and flipped six times in 83 s. One second after the
sixth cutover COMPLETED, PP1 raised:

    RuntimeError: #631 PROXY LEFTOVER REFUSED: a proxy stamped mb_id=2
    seq=19 rows=4096 epoch=6 arrived while this rank is on mb_id=0 in
    flip epoch 6.

Same epoch on both sides, so this is not a stale message from before a
cutover. It is a slot-ring divergence WITHIN one epoch, and the line that
names it was printed by PP0 in the same second:

    PHASE-FLIP PASS-CLOCK across the armed window: rank 0 ran 0 slot
    iteration(s) (armed at mb_id=2, disarmed at mb_id=0); group passes
    [0, 0, 0], SPREAD 0; group RESUME SLOTS [2, 1, 1] -- DIVERGED

THE ROOT, and it is a lifetime and not a predicate. ``_pp_flip_arm_mb_id``
is written once on the rising edge (scheduler_pp_mixin.py:2562) and read
once on the falling edge (:2585). Nothing else in the tree touches it --
in particular ``init_pp_loop_state`` (:3630), which rebuilds ``mbs``,
``last_mbs``, ``running_mbs`` and ``mb_metadata``, i.e. THE ENTIRE SLOT
RING, does not clear it.

So a COMMITTED cutover carries it across the rebuild:

  1. ``_phase_flip_on_round`` (:1781) commits and raises PhaseFlipLoopExit
     while ``_pp_flip_armed_passes == 0``.
  2. The loop unwinds, the cutover swaps the topology, and
     ``init_pp_loop_state`` builds a NEW ring. The flip epoch advances.
  3. ``_event_loop_pp_body`` restarts at ``mb_id = 0`` (:1387).
  4. The first ``_pp_flip_pass_tick(0)`` (:1393) sees ``armed == False``
     with ``passes == 0`` -- THE FALLING EDGE -- and #824 W4b's restore
     (:2612) fires with ``arm_mb`` naming a slot in the RETIRED ring.
  5. :1405 jumps the rebuilt ring to that slot.

"disarmed at mb_id=0" is therefore not "one slot later". It is the first
slot of a ring that did not exist when mb_id=2 was recorded.

W4b IS NOT THE DEFECT AND MUST NOT BE WEAKENED. Its case -- an armed
window that ran zero iterations and ABANDONED -- returns to the same loop
with the same ring, where the arm slot still names the slot the rank is
parked on. Its guard `passes == 0 and arm_mb is not None and
int(arm_mb) != int(mb_id)` is simply missing the term that separates an
abandon from a commit, and this file supplies it. The tree already states
the law that term encodes, at scheduler_pp_mixin.py:2500:

    WHY A COMMIT IS SAFE AND AN ABANDON IS NOT [...] A committed flip
    re-forms the topology and rebuilds the loop state
    (``init_pp_loop_state``), so the phase is reset by construction. An
    abandon returns to the SAME loop with drifted counters.

and the #631 guard's own message names the hazard exactly:

    a pass from before a cutover that rebuilt this rank's whole slot
    ring, whose slot number therefore names nothing here however well it
    matches (#795).

WHY THE EARLIER WINDOW SURVIVED, which is the fact that makes this a
lifetime bug rather than an intermittent one. The epoch-4 cutover
(15:59:36) took the SAME path -- three ranks, zero iterations, a restore
each -- and logged ``RESUME SLOTS [1, 1, 1] -- AGREED``. It survived
because all three ranks happened to carry the same retired arm slot. The
epoch-6 cutover carried [2, 1, 1] and the group died. Agreement here is
luck, not a guarantee, and this suite asserts the guarantee.

CPU-only. No gloo, no CUDA, no scheduler construction, no GPU. Every arm
is a fixed number of direct calls, so this file cannot fail by hanging.
"""

from types import SimpleNamespace

import pytest


def _tick_harness(loop_size=3, epoch=5, enabled=True):
    """A holder carrying only what ``_pp_flip_pass_tick`` reads.

    Mirrors ``test_pp_zero_iteration_window_slot_824._tick_harness`` --
    counters are left None so the tick takes its counter-free path and
    this file measures the slot decision alone -- and adds the two things
    #829 turns on: a flip-epoch accessor (the ring generation, read the
    way ``pp_flip_epoch_of`` reads it) and a recording stand-in for
    ``pp_flip_drain_leftover_dicts``, so #757's drain can be asserted to
    still run rather than assumed.
    """
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    class S:
        pass

    s = S()
    s.server_args = SimpleNamespace(enable_phase_flip=enabled)
    s.pp_flip_counters = None
    s.pp_loop_size = loop_size
    s._armed = True
    s._epoch = epoch
    s.drained = []
    s.pp_phase_flip_armed = lambda: s._armed
    s._pp_flip_epoch = lambda: s._epoch
    s.pp_flip_drain_leftover_dicts = lambda mb_id: s.drained.append(mb_id)
    return s, SchedulerPPMixin._pp_flip_pass_tick.__get__(s, S)


def _commit(s, new_epoch):
    """What a committed cutover does to this rank, and nothing more.

    The flip is no longer pending and the ring generation has advanced.
    The slot ring itself is rebuilt by ``init_pp_loop_state``; that is not
    modelled here because the point of the arm is that it survives the
    rebuild -- the tick never reads the ring.
    """
    s._armed = False
    s._epoch = new_epoch


def _resume(s):
    return getattr(s, "_pp_flip_resume_slot", None)


# -- the root -----------------------------------------------------------


def test_a_committed_cutover_does_not_carry_the_arm_slot_into_the_new_ring():
    """PP0's sequence at 15:59:57, tick for tick."""
    s, tick = _tick_harness(epoch=5)

    tick(2)  # rising edge: armed at mb_id=2, in the ring of flip epoch 5
    assert s._pp_flip_arm_mb_id == 2
    assert s._pp_flip_armed_passes == 0

    _commit(s, 6)  # the cutover commits; init_pp_loop_state rebuilds the ring
    tick(0)  # first tick of the NEW ring -> falling edge

    assert _resume(s) is None, (
        "the arm slot was recorded against the ring of epoch 5 and that "
        "ring no longer exists; applying it to epoch 6's ring is the "
        "15:59:57 'armed at mb_id=2, disarmed at mb_id=0' line, and one "
        "second later PP1 refused a proxy stamped mb_id=2 while sitting "
        "on mb_id=0 of the same epoch"
    )


def test_two_ranks_crossing_one_cutover_resume_on_the_same_slot():
    """The group consequence, with the specimen's own numbers.

    WHAT THIS DOES AND DOES NOT ASSERT, because the loose version is
    false. It is NOT "all ranks always sit on the same mb_id": PP stages
    legitimately run at different microbatch offsets (#737), and the
    ring arithmetic in ``_event_loop_pp_body`` says so itself --
    ``next_first_rank_mb_id`` and ``next_mb_id`` are deliberately
    different slots.

    The assertion is the narrower one, and it holds at exactly one
    instant: the first tick after a rebuild, where ``_event_loop_pp_body``
    has just put EVERY rank on the ring's entry slot 0 unconditionally.
    What must not happen is that a rank is diverted off that entry slot by
    a number recorded against the ring the cutover just retired. Two ranks
    carrying different retired arm slots are diverted by different
    amounts, and the difference is the mispairing that follows.
    """
    pp0, tick0 = _tick_harness(epoch=5)
    pp1, tick1 = _tick_harness(epoch=5)

    # The two ranks armed on different slots of the retired ring. This is
    # the [2, 1, 1] the PASS-CLOCK line reported.
    tick0(2)
    tick1(1)

    # One cutover, both ranks commit it, both re-enter a fresh ring at 0.
    _commit(pp0, 6)
    _commit(pp1, 6)
    tick0(0)
    tick1(0)

    resumed0 = _resume(pp0) if _resume(pp0) is not None else 0
    resumed1 = _resume(pp1) if _resume(pp1) is not None else 0
    assert resumed0 == resumed1, (
        f"PP0 resumes epoch 6 on slot {resumed0} and PP1 on slot "
        f"{resumed1}: from here PP0 stamps its proxies with a slot PP1 is "
        f"not on, which is #631 PROXY LEFTOVER REFUSED and the death of "
        f"boot_window2_0823_1554"
    )


# -- what must NOT change: #824 W4b's own case --------------------------


def test_an_abandon_in_the_same_epoch_still_restores_the_arm_slot():
    """#824 W4b, unweakened.

    THIS ARM IS THE POINT OF THE FILE AS MUCH AS THE ONES ABOVE. The
    cheapest way to make the two tests above pass is to delete the
    restore, and that reopens boot_827: an armed window that ran zero
    iterations and abandoned advanced the slot while the request chain
    that paces it did not. Same ring, same epoch, so the arm slot still
    names the slot this rank is parked on and returning to it is right.
    """
    s, tick = _tick_harness(epoch=5)

    tick(0)  # armed at mb_id=0
    s._armed = False  # abandoned -- the epoch does NOT move
    tick(1)  # falling edge one slot later, having run 0 iterations

    assert _resume(s) == 0, (
        "an abandoned zero-iteration window returns to the SAME ring, "
        "where the arm slot is still the slot this rank is parked on; "
        "#824 W4b must survive #829 intact"
    )


def test_a_holder_that_names_no_epoch_keeps_its_pre_795_behaviour():
    """``pp_flip_epoch_of``'s documented convention, applied here too.

    "No accessor" reads as "no epoch", which is the slot-only behaviour
    that shipped before #795. A holder written before this file must not
    change meaning because #829 started asking a question it cannot
    answer.
    """
    s, tick = _tick_harness(epoch=5)
    del s._pp_flip_epoch

    tick(0)
    s._armed = False
    tick(1)

    assert _resume(s) == 0, (
        "with no epoch accessor there is no evidence the ring was "
        "rebuilt, so the restore must behave exactly as it did before"
    )


def test_the_leftover_drain_still_runs_on_a_committed_falling_edge():
    """#757 regression, asserted rather than assumed.

    The falling edge is the ONLY place every disarm passes through, and
    ``pp_flip_drain_leftover_dicts`` is there for that reason. A #829 fix
    that suppressed the falling edge after a commit -- for instance by
    clearing ``_pp_flip_armed_passes`` in ``init_pp_loop_state`` -- would
    silently take that drain away. #829 removes the SLOT decision on this
    path and nothing else.
    """
    s, tick = _tick_harness(epoch=5)

    tick(2)
    _commit(s, 6)
    tick(0)

    assert s.drained == [0], (
        "the commit path must still reach #757's leftover drain, on the "
        f"slot the rank is actually on; got {s.drained}"
    )


def test_a_disabled_flip_never_records_an_arm_at_all():
    """The default path is untouched.

    With ``enable_phase_flip`` off the tick returns on the server-args
    test before reading anything, so no arm is recorded and no restore
    can be requested. Booting without the flip must cost exactly nothing.
    """
    s, tick = _tick_harness(epoch=5, enabled=False)

    tick(2)
    _commit(s, 6)
    tick(0)

    assert getattr(s, "_pp_flip_arm_mb_id", None) is None
    assert _resume(s) is None


# -- the structural backstop, and proof that it is wired ----------------


def test_forgetting_ring_scoped_slots_clears_all_three_and_only_those():
    """The helper itself.

    ``_pp_flip_armed_passes`` surviving is not incidental: it is what
    makes the next tick take the falling edge, where #757's leftover
    drain lives.
    """
    from sglang.srt.managers.scheduler_pp_mixin import (
        pp_flip_forget_ring_scoped_slots,
    )

    s = SimpleNamespace(
        _pp_flip_arm_mb_id=2,
        _pp_flip_arm_epoch=5,
        _pp_flip_resume_slot=2,
        _pp_flip_armed_passes=0,
    )
    pp_flip_forget_ring_scoped_slots(s)

    assert s._pp_flip_arm_mb_id is None
    assert s._pp_flip_arm_epoch is None
    assert s._pp_flip_resume_slot is None
    assert s._pp_flip_armed_passes == 0, (
        "clearing this would take #757's leftover drain off the commit "
        "path, because the falling edge is the only place every disarm "
        "passes through"
    )


def test_a_holder_without_the_attributes_gains_them_as_none():
    """Plain assignment, not delattr -- every reader treats None as
    'nothing recorded', and an AttributeError here would break the boot
    path that calls init_pp_loop_state before any flip has ever armed."""
    from sglang.srt.managers.scheduler_pp_mixin import (
        pp_flip_forget_ring_scoped_slots,
    )

    s = SimpleNamespace()
    pp_flip_forget_ring_scoped_slots(s)
    assert s._pp_flip_arm_mb_id is None
    assert s._pp_flip_arm_epoch is None
    assert s._pp_flip_resume_slot is None


def test_the_ring_rebuild_actually_calls_the_helper():
    """WIRING, not intent.

    ``init_pp_loop_state`` is the one function every ring rebuild passes
    through -- boot, the cutover's topology swap, and event_loop_pp's own
    entry. A helper nobody calls is a fix that was written and never ran,
    and this suite cannot construct a Scheduler to find that out at
    runtime. Asserting on the name is stable: it is an identifier, not
    formatting, and it fails loudly if the call is ever dropped in a
    refactor.
    """
    import inspect

    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    src = inspect.getsource(SchedulerPPMixin.init_pp_loop_state)
    assert "pp_flip_forget_ring_scoped_slots(self)" in src, (
        "the ring rebuild must forget slot numbers recorded against the "
        "ring it is replacing; without this call #829's backstop is not "
        "wired to anything"
    )
    assert src.index("pp_flip_forget_ring_scoped_slots(self)") < src.index(
        "self.mbs = [None] * self.pp_loop_size"
    ), "forget the old ring's slot numbers before building the new ring"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
