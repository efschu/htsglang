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

from sglang.srt.managers.pp_admission_congruence import PPAdmissionDecision
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


class _Group:
    is_last_rank = True  # skips the proxy-send block; not what is under test
    # [#791] f31fd5e43e added a read of `is_first_rank` in the admission
    # block of `_event_loop_pp_body` (scheduler_pp_mixin.py:1297): PP0 never
    # receives an admission decision, every other rank does. Real
    # `is_first_rank`/`is_last_rank` are derived from the SAME rank number
    # (parallel_state.py:1051/1056, `self.rank == self.first_rank` /
    # `self.rank == self.last_rank`), so the two must not be set
    # independently here either. `_PS.pp_size = 3` below and `is_last_rank =
    # True` together fix this fixture as the LAST of 3 stages (rank 2 of
    # 0..2) -- a rank that is last in a >1-stage group cannot also be first,
    # so `is_first_rank` is False, not an independent stub.
    is_first_rank = False


class _ServerArgs:
    def __init__(self, enable_phase_flip=True):
        self.enable_phase_flip = enable_phase_flip
        self.pp_async_batch_depth = 0


class _PS:
    pp_size = 3
    # #791 (f31fd5e43e): `_pp_reconcile_incoming_admission` reads
    # `self.ps.pp_rank` directly (scheduler_pp_mixin.py:4510), no getattr
    # default. `_Group.is_last_rank = True` above already fixes this fixture
    # as the LAST of 3 stages -- ranks are 0..pp_size-1, so the last rank's
    # number is `pp_size - 1 = 2`. Not an independent stub: it is the same
    # rank identity `is_last_rank`/`is_first_rank` are derived from.
    pp_rank = 2


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

    # #795 (a996a653a7) put the admission decision on the wire, and
    # `_event_loop_pp_body` now issues a BLOCKING receive for it at
    # scheduler_pp_mixin.py:1299 on every rank that is not PP0. `_Group`
    # above fixes this fixture as the last of three stages, so it is such a
    # rank and the receive is genuinely on its path.
    #
    # STUBBED, not taken off the mixin, and that is the exception this class
    # states for itself: the loop CONTROL is shipping code, "everything the
    # body calls into is stubbed to the cheapest thing that keeps the control
    # flow real". The shipped receive blocks on a real PP wire
    # (`_pp_recv_typed_dict`), which this fixture has no peer for; binding it
    # would not test the receive, it would hang the suite.
    #
    # NOT a bare `None`. This fixture models an upstream that ALWAYS
    # launches (the #631 slot-hold scenario this file exists to pin, not the
    # #798 "upstream ran nothing" one -- that is covered separately by the
    # excluded test_pp_void_slot_advance_798.py). The shipped function sets
    # three flags straight off the wire message with `message.get(KEY,
    # False)` (scheduler_pp_mixin.py:4358-4366): `_pp_output_expected_
    # incoming`, `_pp_pass_voided_incoming`, and `_pp_upstream_launched_
    # incoming`. A real sender in this scenario always posts
    # `_PP_UPSTREAM_LAUNCHED_KEY=True` (`launched=self.mbs[mb_id] is not
    # None`, scheduler_pp_mixin.py:1530, true on every non-armed pass) and
    # never retracts or asks for output on this path, so the three flags a
    # real message would produce here are False/False/True -- set them
    # explicitly rather than leave the loop's own unconditional per-pass
    # reset (`False` at :1289/:1294/:1295) stand in for "received", because
    # leaving `_pp_upstream_launched_incoming` False would make `_pp_void_
    # pass_without_upstream_launch` below spuriously void every steady-state
    # pass (`self.mbs[mb_id]` is non-None whenever this rank is not armed),
    # corrupting the very slot state this suite pins. `PPAdmissionDecision
    # (mb_id=0, entries=())` is the shape `pp_admission_decision_from_wire`
    # produces for a message with no entries; `mb_id` on it is never read by
    # anything this fixture reaches (checked: `_pp_reconcile_incoming_
    # admission`, `_pp_void_retracted_pass`, `_pp_forwarded_schedule_from`,
    # `_pp_note_output_expectation` all key off `.entries`, not `.mb_id`).
    def _pp_recv_admission_decision(self):
        self._pp_output_expected_incoming = False
        self._pp_pass_voided_incoming = False
        self._pp_upstream_launched_incoming = True
        return PPAdmissionDecision(mb_id=0, entries=())

    # #791 (f31fd5e43e): the receive's consumer, one line later. BOUND, not
    # stubbed -- now that the line above answers with a real
    # `PPAdmissionDecision` instead of `None`, the shipped function no longer
    # dereferences anything this fixture lacks. With `entries=()` its `for
    # entry in decision.entries` loop body never runs, so it needs no
    # `self.tree_cache`; it does build `{req.rid: req for req in self.
    # waiting_queue}` unconditionally and read `self.ps.pp_rank` directly
    # (scheduler_pp_mixin.py:4489/4510), both supplied above. Reduces to
    # `reconcile_pp_admission_decision(decision, {}, rank=2, pp_size=3, ...)`,
    # which on an empty `decision.entries` returns `({}, decision)` unchanged
    # (pp_admission_congruence.py:584-628) -- the same answer this file used
    # to hand-stub, now produced by the delivered function itself.
    _pp_reconcile_incoming_admission = SchedulerPPMixin._pp_reconcile_incoming_admission

    # #797 (6f42d7f923/d2c5be2dee)'s prevention step, the third link of the
    # same chain. BOUND for the same reason as above: it reads `self.ps.
    # pp_rank` directly and calls `pp_pass_should_void(amended, rank,
    # _pp_pass_voided_incoming)` (scheduler_pp_mixin.py:4594-4598), which
    # with `incoming_voided=False` and an `amended` whose `entries=()` (so
    # `entries_retracted_by_rank` returns `()`) answers False -- "this rank
    # retracted nothing of its own and nothing upstream voided" -- so the
    # shipped path returns `(effective, amended)` UNCHANGED, exactly the
    # identity this file used to hand-stub, and additionally sets
    # `self._pp_admission_pass_voided = False` itself (this fixture no
    # longer relies solely on the loop's own per-pass reset at :1296 for it).
    _pp_void_retracted_pass = SchedulerPPMixin._pp_void_retracted_pass

    # #797b (scheduler_pp_mixin.py:4156). BOUND: "Tolerant of a stand-in that
    # never ran `init_pp_loop_state`... a holder without the array grows one
    # here rather than raising" is this file's own #787 convention stated in
    # the function's own docstring. Reads only `pp_loop_size` (set) and
    # `chunked_req` (set to `None` in `__init__`, the real pre-admission
    # value for a rank that has never started a chunk).
    _pp_note_chunked_req_before_admission = (
        SchedulerPPMixin._pp_note_chunked_req_before_admission
    )

    # #798 (07744abcb7). BOUND, not stubbed, and this is the safety-critical
    # one: `if self.ps.pp_size <= 1 or self.pp_group.is_first_rank: return
    # False` does not short-circuit here (pp_size=3, is_first_rank=False),
    # but the very next guard does -- `if getattr(self, "_pp_upstream_
    # launched_incoming", False): return False` (scheduler_pp_mixin.py:4904)
    # -- because `_pp_recv_admission_decision` above now sets that flag True
    # for every pass, matching the always-launching upstream this file
    # models. Without that flag this method would treat every non-armed
    # pass's `self.mbs[mb_id]` as "upstream voided, this rank must not run
    # it" and call the real `_pp_void_own_batch`, wiping the slot state the
    # #631 hold predicate depends on -- a #798 behaviour this file is not
    # scoped to exercise (that is test_pp_void_slot_advance_798.py).
    _pp_void_pass_without_upstream_launch = (
        SchedulerPPMixin._pp_void_pass_without_upstream_launch
    )

    # #797 (6f42d7f923/d2c5be2dee). BOUND: its very first line is `if not
    # getattr(self, "_pp_admission_pass_voided", False): return False`
    # (scheduler_pp_mixin.py:4649), and `_pp_admission_pass_voided` is False
    # throughout this fixture's runs (never voided, per the two bindings
    # above), so it returns immediately and never reaches the real
    # `_pp_recv_typed_dict` call that would need an actual wire.
    _pp_drain_voided_proxy = SchedulerPPMixin._pp_drain_voided_proxy

    # #796 (2323c9255d). BOUND: `pending = getattr(self, "_pp_admission_send_
    # work", None); if pending: ...` (scheduler_pp_mixin.py:4331-4333) -- this
    # fixture's `_pp_send_admission_decision` stub below never sets `_pp_
    # admission_send_work`, so `pending` stays `None`, `if pending` is False,
    # and the real `_pp_commit_comm_work` call it would otherwise make is
    # never reached.
    _pp_commit_admission_send_work = SchedulerPPMixin._pp_commit_admission_send_work

    # #791 (f31fd5e43e). STUBBED, like `_pp_recv_admission_decision` above
    # and for the same reason: the shipped function's job past its two early
    # returns is `self._pp_send_dict_to_next_stage(..., async_send=True,
    # msg_type=ADMISSION_DECISION_KIND)`, a real isend on a wire this fixture
    # has no peer for. No-op, following this class's own convention for
    # every other wire-touching call (`_pp_commit_comm_work`, `_pp_launch_
    # batch`).
    def _pp_send_admission_decision(self, decision, **kwargs):
        pass

    # #791: the point where forwarded chunk lengths enter this rank. BOUND,
    # unlike the three above -- it takes `Optional[PPAdmissionDecision]` and
    # is a single delegation to `forwarded_schedule`, so it answers for the
    # None this fixture carries without needing a peer. Its own docstring
    # calls it "the single point at which the forwarded chunk lengths enter
    # this rank, so it is the single point a test can neuter"; taking it off
    # the mixin means this fixture is NOT neutering it.
    _pp_forwarded_schedule_from = SchedulerPPMixin._pp_forwarded_schedule_from
    # #791b: the per-slot output-ring verdict, written on EVERY pass by every
    # rank. Bound: it reads only `pp_loop_size` (which this fixture sets) and
    # self-initialises its two per-slot lists through `getattr` defaults, so
    # it needs nothing a peer would have to supply.
    _pp_note_output_expectation = SchedulerPPMixin._pp_note_output_expectation

    def __init__(
        self,
        arm_at_iteration,
        armed_spins,
        enable_phase_flip=True,
        honour_hold=True,
        tail=4,
        pp_loop_size=3,
    ):
        self.pp_loop_size = pp_loop_size
        self.ps = _PS()
        self.pp_group = _Group()
        self.server_args = _ServerArgs(enable_phase_flip)
        self.request_receiver = _Receiver()
        # #791 (f31fd5e43e): `_pp_reconcile_incoming_admission` builds
        # `{req.rid: req for req in self.waiting_queue}` unconditionally, no
        # getattr default. Empty is the real shape: nothing in this fixture
        # ever appends to it, so a real object at this point in its life
        # (before any request admission) would have an empty queue too.
        self.waiting_queue = []
        # #753 (669f85eca9): `_event_loop_pp_body` reads `self._pp_gapped_wire`
        # directly at scheduler_pp_mixin.py:1669 (no getattr default), set for
        # real by `init_pp_loop_state` from `pp_gapped_ownership_active(pp_
        # size)` -- which this fixture never calls. `pp_gapped_ownership_
        # active` itself answers False whenever no `PP_LAYER_SET_ENV` is
        # configured (distributed/utils.py:2029-2030), which is the case in
        # this hermetic test process, so False is the value a REAL rank in
        # this run's configuration would carry, not a simplification: this
        # file's whole scope is the ungapped stage-boundary loop (#631), the
        # gapped crossing-wire lockstep is #753's separate concern.
        self._pp_gapped_wire = False
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

    def _pp_recv_proxy_tensors(self, mb_id):
        return None

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
    assert _predicate_rank()._pp_flip_hold_slot() is True


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
