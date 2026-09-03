"""#631 defect Q: the armed window must not move the microbatch slot index.

THE DEFECT, measured 2026-08-09 07:19:23Z on the boot that produced corpse
R (serving-30030.corpseR.log):

    rank 0 ran 44477 slot iteration(s) (armed at mb_id=2, disarmed at mb_id=2)
    rank 1 ran 33690 slot iteration(s) (armed at mb_id=2, disarmed at mb_id=0)
    rank 2 ran 38069 slot iteration(s) (armed at mb_id=2, disarmed at mb_id=2)
    SPREAD 10787

Every rank ARMS on the same slot and LEAVES on a different one. In steady
state the pass loop is paced by the request chain -- one blocking receive
per slot iteration -- so the indices cannot diverge. An armed rank admits
nothing and launches nothing, so its iterations become free-running spin
(~8 kHz here), it abandons on its own park deadline, and it re-enters the
pipeline wherever it happened to stop.

WHAT THAT BREAKS IS THE LABEL, NOT THE WIRE. A parked rank neither sends
nor receives a proxy, so the one-message-per-pass contract holds and the
counters stay balanced -- there is nothing stranded to dispose of. But
stage k then computes its slot-s batch while stage k+1 applies the result
to its slot-s' batch, permanently. Both earlier disposals died on that
misreading: corpse R took a second message against a debt of one and
wedged the instance in 6 s; corpse S drained a wire with no surplus on it
and ate an output that pre-arm work still owed.

WHAT IS PINNED HERE

* the hold predicate itself -- every clause, including the two that must
  REFUSE to hold (a half-written chunk, and any slot still occupied);
* the loop control, driven through the REAL ``_event_loop_pp_body`` rather
  than a transcription of it, because the fix IS the loop control;
* the property that matters: two ranks that spin a DIFFERENT number of
  armed iterations resume the pipeline on the SAME slot;
* the can-fail proof -- the same scenario with the hold disabled resumes
  on different slots, i.e. this suite would have caught the metal defect;
* the default path: with the flip off the index sequence is exactly
  0,1,2,0,1,2 as ``for mb_id in range(pp_loop_size)`` produced.

CPU-only, no CUDA, no distributed.
"""

import pytest

from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin


class _StopLoop(Exception):
    """Ends the (otherwise infinite) event loop at a known point."""


class _Batch:
    """Stands in for a ScheduleBatch. Truthy, which is what the loop reads."""


class _Plan:
    def __init__(self, batch_to_run, running_batch):
        self.batch_to_run = batch_to_run
        self.running_batch = running_batch


class _Event:
    def synchronize(self):
        pass


def assert_faithful_pp_roles(group, pp_size):
    """A rank of a pp_size>1 ring is not both the first and the last stage.

    #791, and the reason this check exists instead of two more attributes.
    The stub below used to carry ``is_last_rank = True`` and nothing else,
    so the loop's ``not self.pp_group.is_first_rank`` raised AttributeError
    the moment it asked -- the drift this suite was failing on.

    The cheap repair is to add ``is_first_rank = True`` beside it. That
    would make this rank the FIRST and the LAST stage of a three-stage
    pipeline simultaneously, which no rank of a pp_size=3 ring can be, and
    it would make every branch keyed on either role take the wrong arm
    while the suite stayed green. That is the #630 lesson exactly: an
    unfaithful stub does not merely fail to catch the defect, it ENCODES
    the defect's assumption and then certifies it.

    So the roles are derived from a real position in a real ring, and this
    guard makes the forbidden combination a loud failure rather than a
    silent one.
    """
    first = bool(group.is_first_rank)
    last = bool(group.is_last_rank)
    if pp_size > 1 and first and last:
        raise AssertionError(
            f"unfaithful PP group stub: is_first_rank and is_last_rank are "
            f"both True at pp_size={pp_size}. Only a degenerate one-stage "
            f"ring can be both. A stub in this shape silently takes the "
            f"wrong arm of every role-keyed branch in the loop under test."
        )
    return group


class _Group:
    """The PP group as ONE rank of a real pp_size=3 ring sees it.

    Rank 2 is the default because that is what the previous stub was
    reaching for with its lone ``is_last_rank = True``: the last stage,
    which skips the proxy-send block that is not under test here. The
    difference is that it is now last WITHOUT also claiming to be first,
    so ``_event_loop_pp_body``'s admission-decision branch takes the arm a
    real last rank takes.
    """

    def __init__(self, rank: int = 2, pp_size: int = 3):
        self.rank = int(rank)
        self.pp_size = int(pp_size)
        if not 0 <= self.rank < self.pp_size:
            raise AssertionError(
                f"rank {self.rank} is not a position in a {self.pp_size}-stage ring"
            )

    @property
    def is_first_rank(self):
        return self.rank == 0

    @property
    def is_last_rank(self):
        return self.rank == self.pp_size - 1


class _ServerArgs:
    def __init__(self, enable_phase_flip=True):
        self.enable_phase_flip = enable_phase_flip
        self.pp_async_batch_depth = 0


class _PS:
    pp_size = 3


class _Receiver:
    def recv_requests(self):
        return []


class _Rank:
    """A rank whose LOOP CONTROL is the shipping code.

    ``_event_loop_pp_body`` and ``_pp_flip_hold_slot`` are taken unbound off
    the mixin, so a regression in either is a failure here. Everything the
    body calls into is stubbed to the cheapest thing that keeps the control
    flow real.
    """

    _event_loop_pp_body = SchedulerPPMixin._event_loop_pp_body
    _pp_flip_hold_slot = SchedulerPPMixin._pp_flip_hold_slot
    # #631 defect R: the slot's last-batch bookkeeping is shipping code too,
    # so it is taken off the mixin like the rest rather than restubbed here.
    _pp_record_slot_last_batch = SchedulerPPMixin._pp_record_slot_last_batch

    def __init__(
        self,
        arm_at_iteration,
        armed_spins,
        enable_phase_flip=True,
        honour_hold=True,
        tail=4,
        pp_loop_size=3,
        pp_group=None,
    ):
        self.pp_loop_size = pp_loop_size
        self.ps = _PS()
        # #791: a REAL position in the ring, checked. ``pp_group`` is an
        # injection point so the can-fail test can plant an unfaithful one
        # and prove this suite rejects it.
        self.pp_group = (
            _Group(rank=2, pp_size=_PS.pp_size) if pp_group is None else pp_group
        )
        assert_faithful_pp_roles(self.pp_group, self.ps.pp_size)
        self.server_args = _ServerArgs(enable_phase_flip)
        self.request_receiver = _Receiver()
        self.mbs = [None] * pp_loop_size
        self.last_mbs = [None] * pp_loop_size
        self.running_mbs = [_Batch() for _ in range(pp_loop_size)]
        self.mb_metadata = [None] * pp_loop_size
        self.send_proxy_work = []
        self.pp_outputs = None
        self.last_rank_comm_queue = None
        self.chunked_req = None
        self.launch_event = None
        self.cur_batch_for_debug = None
        self.running_batch = None
        self.last_batch = None

        self._honour_hold = honour_hold
        self._arm_at = arm_at_iteration
        self._armed_spins = armed_spins
        self._tail = tail
        # A backstop so a broken predicate fails as a FAILED ASSERTION and
        # not as a hung suite. It is generous: the longest scenario here
        # spins 97 armed iterations.
        self._hard_stop = 400

        self.armed = False
        self.admission_recvs = 0
        self.iterations = 0
        self.spins = 0
        self.slots_seen = []
        self.slots_after_disarm = []
        self.disarmed = False

    # -- what the loop calls -------------------------------------------

    def _pp_flip_pass_tick(self, mb_id):
        self.slots_seen.append(mb_id)
        if self.disarmed:
            self.slots_after_disarm.append(mb_id)
        # The loop is infinite by design; end it here, at the TOP of an
        # iteration, so every recorded index belongs to a complete pass.
        if self.disarmed and len(self.slots_after_disarm) > self._tail:
            raise _StopLoop
        if len(self.slots_seen) > self._hard_stop:
            raise _StopLoop

    def _pp_forward_and_process_input_requests(self, recv_reqs):
        pass

    def get_next_batch_to_run(self, running_batch, last_batch):
        # THE PARK: a pending flip yields no batch at all. This single line
        # is why an armed rank's iterations are pure spin.
        if self.armed:
            return _Plan(None, running_batch)
        return _Plan(_Batch(), running_batch)

    # -- the admission-decision arm a non-first rank really takes ---------
    #
    # #791: reached only now that this rank reports a faithful position in
    # the ring. These are collaborators, not the subject: the loop CONTROL
    # is what this suite pins, so each is the cheapest thing that keeps the
    # control flow real and records nothing.

    def _pp_recv_admission_decision(self):
        # Counted, so the suite can prove the faithful role is LOAD-BEARING
        # and not decorative: a rank that reports "not first" must actually
        # take this arm.
        self.admission_recvs += 1
        return None

    def _pp_reconcile_incoming_admission(self, incoming):
        return None, None

    def _pp_void_retracted_pass(self, effective, amended):
        return effective, amended

    def _pp_forwarded_schedule_from(self, amended):
        return None

    def _pp_note_output_expectation(self, mb_id, expected, amended):
        pass

    _pp_output_expected_incoming = None
    #: #753's per-iteration lockstep barrier is for the GAPPED layout. This
    #: rank is an ordinary contiguous stage, so the barrier is correctly
    #: skipped -- and skipping it is what keeps this suite free of a real
    #: process group.
    _pp_gapped_wire = False

    def _pp_note_chunked_req_before_admission(self, mb_id):
        pass

    def _pp_void_pass_without_upstream_launch(self, mb_id):
        return False

    def _pp_void_own_batch(self, mb_id):
        pass

    def _pp_send_admission_decision(self, *args, **kwargs):
        pass

    def _pp_try_recv_admission_decision(self, *args, **kwargs):
        return None

    def _pp_commit_admission_send_work(self, *args, **kwargs):
        pass

    def _pp_commit_pending_req_work(self, *args, **kwargs):
        pass

    def _pp_drain_voided_proxy(self, *args, **kwargs):
        return None

    def _pp_proxy_stamp(self, *args, **kwargs):
        return None

    def _pp_send_dict_to_next_stage(self, *args, **kwargs):
        return None

    def _pp_recv_proxy_tensors(self, mb_id):
        return None

    #: #1173 D2b: the hold now asks whether a launched pass's proxy frame is
    #: waiting in the typed inbox, because a frame the arm cannot execute is a
    #: pass its launcher waits for for ever. This double carries the real mixin
    #: helper; it has no ``pp_group``, so the inbox read raises inside the
    #: helper's own guard and the answer is the honest "no frame is waiting" --
    #: which is exactly the state the predicate cases below describe.
    _pp_flip_stashed_frame_forces_advance = (
        SchedulerPPMixin._pp_flip_stashed_frame_forces_advance
    )
    #: #1173: every early return of the hold now forgets the stashed-frame
    #: window through this helper, so a double that binds the hold must bind
    #: this too -- otherwise the release path raises AttributeError and the
    #: predicate cases below never reach the answer they are asserting.
    _1173_forget_stashed_frame = SchedulerPPMixin._1173_forget_stashed_frame

    def _pp_commit_send_output_work_and_preprocess_output_tensors(self, a, b):
        return None, None, _Event()

    def _pp_commit_comm_work(self, work):
        pass

    def _pp_launch_batch(self, *args, **kwargs):
        return object(), None

    def _pp_process_batch_result(self, batch, result):
        pass

    def on_idle(self):
        pass

    def pp_phase_flip_armed(self):
        return self.armed

    def _phase_flip_on_round(self, require_armed_and_parked=False):
        """Arms, counts the armed spins, disarms, then ends the loop.

        Stands in for the runtime's bounded gate. The only behaviour that
        matters for this pin is WHEN the rank stops being armed -- which on
        metal is a rank-local park deadline, i.e. a different number of
        spins on every rank.
        """
        self.iterations += 1
        if not self.armed and not self.disarmed and self.iterations >= self._arm_at:
            self.armed = True
            return
        if self.armed:
            self.spins += 1
            if self.spins >= self._armed_spins:
                self.armed = False
                self.disarmed = True
            return

    def run(self):
        with pytest.raises(_StopLoop):
            self._event_loop_pp_body()
        return self

    # The can-fail lever: reproduces the pre-fix loop exactly.
    def hold_or_not(self):
        if not self._honour_hold:
            return False
        return SchedulerPPMixin._pp_flip_hold_slot(self)


def _rank(honour_hold=True, **kw):
    r = _Rank(honour_hold=honour_hold, **kw)
    if not honour_hold:
        # Pre-fix loop: the index advances every iteration, armed or not.
        r._pp_flip_hold_slot = lambda: False
    return r


# -- the predicate ------------------------------------------------------


def _predicate_rank(**over):
    r = _Rank(arm_at_iteration=1, armed_spins=1)
    r.armed = True
    r.mbs = [None, None, None]
    for k, v in over.items():
        setattr(r, k, v)
    return r


def test_holds_when_armed_and_every_slot_is_drained():
    """NARROWED by #1173 D2b, not withdrawn.

    The original claim was "armed plus every slot drained implies hold". That
    is still the answer here, but it is no longer the whole predicate: a
    drained ring whose typed inbox still holds a proxy frame belongs to a pass
    PP0 already launched, and the arm does not own it. This rank has no
    readable inbox, so no frame is waiting and the hold stands. The release
    case is pinned directly below.
    """
    assert _predicate_rank()._pp_flip_hold_slot() is True


def test_does_not_hold_while_a_launched_passs_frame_is_stashed(monkeypatch):
    """#1173 D2b: followers execute what PP0 launched, arm or not.

    A drained ring is NOT quiescent while a frame for a launched pass sits in
    the typed inbox. Holding there parks the follower on an order it cannot
    obey while the launcher waits for a frame that never comes -- the #1153
    starvation shape. So the hold releases and the loop walks the ring to the
    slot the frame names.
    """
    import sglang.srt.managers.scheduler_pp_mixin as ppm

    r = _predicate_rank()
    r.pp_group = object()
    monkeypatch.setattr(ppm, "resolve_src", lambda group, x: 2)
    monkeypatch.setattr(ppm, "typed_inbox", lambda group: {(2, "proxy"): [{}]})
    assert r._pp_flip_hold_slot() is False


def test_does_not_hold_when_no_flip_is_armed():
    r = _predicate_rank()
    r.armed = False
    assert r._pp_flip_hold_slot() is False


def test_does_not_hold_when_the_feature_is_off():
    r = _predicate_rank()
    r.server_args = _ServerArgs(enable_phase_flip=False)
    assert r._pp_flip_hold_slot() is False


def test_does_not_hold_while_a_slot_is_still_occupied():
    """The drain must finish first, or the pipeline stalls with work in it."""
    r = _predicate_rank()
    r.mbs = [None, _Batch(), None]
    assert r._pp_flip_hold_slot() is False


def test_does_not_hold_on_a_half_written_chunk():
    """``chunked_req`` is EXEMPT from the park, so those iterations launch
    real work and are chain-paced. Holding there would stop a rank the
    pipeline is still driving."""
    r = _predicate_rank()
    r.chunked_req = object()
    assert r._pp_flip_hold_slot() is False


def test_an_occupied_slot_beats_an_empty_looking_one():
    """``is None``, never ``is_empty()``.

    The stricter test is the one reached after a FIXED number of parked
    iterations on every rank, which is the whole reason the hold lands on
    the same slot group-wide. A truthy-but-empty batch must still block it.
    """

    class _EmptyLooking:
        def is_empty(self):
            return True

        def __bool__(self):
            return False

    r = _predicate_rank()
    r.mbs = [None, _EmptyLooking(), None]
    assert r._pp_flip_hold_slot() is False


def test_a_probe_failure_never_holds_the_loop():
    r = _predicate_rank()

    def _boom():
        raise RuntimeError("probe exploded")

    r.pp_phase_flip_armed = _boom
    assert r._pp_flip_hold_slot() is False


# -- the loop -----------------------------------------------------------


def test_the_default_path_walks_the_slots_exactly_as_the_for_loop_did():
    """With the flip disabled the while loop IS ``for mb_id in range(n)``."""
    r = _Rank(arm_at_iteration=10**9, armed_spins=1, enable_phase_flip=False, tail=8)
    r.disarmed = True  # so the stop condition can fire
    r.run()
    assert r.slots_seen[:6] == [0, 1, 2, 0, 1, 2]


def test_the_slot_does_not_advance_while_the_armed_window_spins():
    r = _Rank(arm_at_iteration=1, armed_spins=50).run()
    # Everything after the drain is one repeated index, not a walk.
    held = r.slots_seen[6:-6]
    assert len(set(held)) == 1, f"the slot walked while armed: {sorted(set(held))}"


@pytest.mark.parametrize("spins_a,spins_b", [(7, 50), (11, 12), (3, 97)])
def test_two_ranks_with_different_spin_counts_resume_on_the_same_slot(spins_a, spins_b):
    """THE PROPERTY THE FEATURE NEEDS, and the one metal falsified.

    The ranks arm together -- the arm rides the 1:1 ordered request chain,
    so it lands on the same ordinal iteration everywhere (measured: "armed
    at mb_id=2" on all three ranks). They then spin for their own park
    deadlines, which is a different number of iterations each. They must
    still come back on the same slot.
    """
    a = _Rank(arm_at_iteration=4, armed_spins=spins_a).run()
    b = _Rank(arm_at_iteration=4, armed_spins=spins_b).run()
    assert a.spins != b.spins
    assert a.slots_after_disarm[0] == b.slots_after_disarm[0]
    assert a.slots_after_disarm[:4] == b.slots_after_disarm[:4]


def test_can_fail_without_the_hold_the_ranks_resume_on_different_slots():
    """The can-fail proof: this is the metal defect, reproduced in a unit.

    Without the hold, the spin count leaks straight into the slot index and
    the two ranks re-enter the pipeline out of phase -- which is exactly
    what the 07:19:23Z PASS-CLOCK line reported.
    """
    a = _rank(honour_hold=False, arm_at_iteration=4, armed_spins=7).run()
    b = _rank(honour_hold=False, arm_at_iteration=4, armed_spins=9).run()
    assert a.slots_after_disarm[0] != b.slots_after_disarm[0]


def test_the_hold_is_released_the_moment_the_flip_disarms():
    """A held rank must not be stuck: disarming resumes the walk."""
    r = _Rank(arm_at_iteration=1, armed_spins=20).run()
    tail = r.slots_after_disarm[:4]
    assert len(set(tail)) > 1, f"the loop never resumed walking: {tail}"


# ---------------------------------------------------------------------------
# #791: the stub's FAITHFULNESS is itself under test.
# ---------------------------------------------------------------------------
#
# The red-first logic is inverted here, and deliberately. The seven tests
# above were RED for months with an AttributeError -- the loop asked this
# rank whether it was the first stage and the stub had no answer. So the
# repair is proven by those tests going GREEN with a faithful stub, and the
# CAN-FAIL is the other direction: the convenient repair must be REJECTED.


class _FirstAndLast:
    """The tempting stub: answer both questions with True and move on."""

    is_first_rank = True
    is_last_rank = True


def test_can_fail_a_stub_that_is_both_first_and_last_is_rejected():
    """THE #630 LESSON, enforced.

    Adding ``is_first_rank = True`` beside the existing
    ``is_last_rank = True`` makes every test above pass. It also makes this
    rank the first AND the last stage of a three-stage pipeline, which no
    rank of a pp_size=3 ring can be -- so every branch keyed on either role
    takes the wrong arm while the suite reports green. An unfaithful stub
    does not merely miss the defect; it encodes the defect's assumption and
    then certifies it.
    """
    with pytest.raises(AssertionError) as exc:
        _Rank(arm_at_iteration=2, armed_spins=3, pp_group=_FirstAndLast())

    assert "both True at pp_size=3" in str(exc.value)


def test_the_faithful_stub_carries_real_ring_positions():
    ring = [_Group(rank=r, pp_size=3) for r in range(3)]

    assert [g.is_first_rank for g in ring] == [True, False, False]
    assert [g.is_last_rank for g in ring] == [False, False, True]
    for g in ring:
        assert_faithful_pp_roles(g, 3)


def test_a_position_outside_the_ring_is_rejected():
    with pytest.raises(AssertionError):
        _Group(rank=3, pp_size=3)


def test_the_faithful_role_is_load_bearing_not_decorative():
    """A rank that reports "not the first stage" must actually take the
    admission-decision arm.

    Without this, the stubs added for that arm could be dead code and the
    word "faithful" would be doing no work: the suite would be green
    because the branch was never entered, which is the state it was in
    before -- only quieter.
    """
    rank = _Rank(arm_at_iteration=2, armed_spins=3).run()

    assert not rank.pp_group.is_first_rank
    assert rank.pp_group.is_last_rank
    assert rank.admission_recvs > 0, (
        "the non-first admission arm was never entered, so the faithful "
        "role changed nothing and this suite still does not exercise it"
    )
