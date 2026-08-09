"""#631 defect G: the armed service loop and its monotone send-counters.

THE DEFECT, measured on metal 2026-08-09 00:06-00:09Z. A rank that became
quiescent and spun at the flip's entry gate stopped issuing its per-pass
chain forward. Its downstream reached the hook ONLY by returning from the
blocking chain recv that this very forward satisfied -- so the first rank
to quiesce (rank 0, the intake rank, always) prevented every rank behind
it from becoming ready. Bounded, because the spinner abandoned at its
deadline, but NOT convergent: the same rank drained first every epoch, so
the starvation reproduced identically, epoch after epoch.

THE FIX under test. Do not resume sending -- an armed rank has nothing to
forward and unconsumed sends would pile up (that is the bounded-chain-recv
corpse). Instead stop the downstream from NEEDING the forward: while
armed, a rank services its channels and reaches the hook by its own poll.

The one hard part is consuming without ``is_completed()``, which never
fires on this transport in either direction (corpse F). The answer is a
POLLABLE SIDE CHANNEL: each sender publishes a monotone counter in
/dev/shm STRICTLY AFTER posting its isend, and a receiver makes the
BLOCKING recv only once that counter exceeds its own consumed count. The
message then provably exists, so the block is bounded by transfer time.

The ordering is the entire safety argument, so it is pinned by a can-fail
test that MUTATES it and requires the mutant to fail. Note the shape of
that mutant: the fakes here RAISE ``_WouldBlockForever`` instead of
actually blocking, because a pin that hangs under mutation tells you
nothing (it already cost this feature a debugging session on a frozen
fake clock).

CPU-only.
"""

import pytest

from sglang.srt.managers.phase_flip_counters import (
    CHAN_DICT,
    CHAN_REQ,
    PhaseFlipCounters,
)


def _counters(tmp_path, rank, n_ranks=3, instance="boot"):
    return PhaseFlipCounters(
        n_ranks=n_ranks, rank=rank, directory=str(tmp_path), instance=instance
    )


# -- the channel itself --------------------------------------------------------


def test_absent_counter_reads_as_zero(tmp_path):
    """Before the first publish, "nothing sent" is the truth.

    It is also the SAFE answer: a receiver that reads 0 makes no blocking
    call, which is counter-lags-send -- the only skew this design permits.
    """
    c = _counters(tmp_path, rank=1)
    assert c.sent(CHAN_REQ, 0) == 0
    assert c.consumed(CHAN_REQ, 0) == 0


def test_counts_are_monotone_and_visible_to_peers(tmp_path):
    sender = _counters(tmp_path, rank=0)
    reader = _counters(tmp_path, rank=1)
    seen = []
    for _ in range(5):
        sender.bump_sent(CHAN_REQ)
        seen.append(reader.sent(CHAN_REQ, 0))
    assert seen == [1, 2, 3, 4, 5]
    assert seen == sorted(seen), "a reader must never see a count go backwards"


def test_channels_are_independent(tmp_path):
    """proxy/output share ONE wire and so ONE counter; the request chain is
    a different wire and must not be conflated with it."""
    sender = _counters(tmp_path, rank=0)
    reader = _counters(tmp_path, rank=1)
    sender.bump_sent(CHAN_REQ)
    sender.bump_sent(CHAN_DICT)
    sender.bump_sent(CHAN_DICT)
    assert reader.sent(CHAN_REQ, 0) == 1
    assert reader.sent(CHAN_DICT, 0) == 2


def test_own_counts_come_from_memory_not_from_the_file(tmp_path):
    """A rank must never be able to FORGET what it already did.

    Reading its own count back off disk would make a failed publish (disk
    full, unlinked directory) look like "I have sent nothing", and this
    rank would then re-send or mis-flush. The file is the published view
    for peers; memory is the authority for the owner.
    """
    c = _counters(tmp_path, rank=0)
    c.bump_sent(CHAN_REQ)
    c.bump_sent(CHAN_REQ)
    import os

    os.unlink(str(tmp_path / "boot.ctr.req.s0"))
    assert c.local_sent(CHAN_REQ) == 2
    assert c.sent(CHAN_REQ, 0) == 2


def test_a_torn_or_empty_file_reads_as_zero_rather_than_raising(tmp_path):
    """One poll early beats an exception on a hot path -- and one poll
    early is exactly counter-lags-send, the safe direction."""
    c = _counters(tmp_path, rank=0)
    (tmp_path / "boot.ctr.req.s1").write_text("")
    assert c.sent(CHAN_REQ, 1) == 0
    (tmp_path / "boot.ctr.req.s1").write_text("not-a-number")
    assert c.sent(CHAN_REQ, 1) == 0


def test_sweep_touches_only_this_rank(tmp_path):
    """Ranks build their counters with no barrier between them, so a sweep
    that removed a peer's file could delete a count already published --
    and a count read as 0 when it is really 3 is the phantom-message
    hazard arriving by another door."""
    _counters(tmp_path, rank=0).bump_sent(CHAN_REQ)
    mine = _counters(tmp_path, rank=1)
    mine.bump_consumed(CHAN_REQ)
    mine.sweep()
    assert mine.sent(CHAN_REQ, 0) == 1, "a peer's published count was swept"


# -- the consume path, and THE ORDERING ----------------------------------------


class _WouldBlockForever(Exception):
    """Raised where the real transport would block with no peer coming.

    It RAISES rather than blocking on purpose. A can-fail test whose
    mutant hangs proves nothing and stalls the suite; this one terminates
    and fails.
    """


class _Wire:
    """A one-directional wire plus the counter published beside it.

    ``post`` puts a message on the wire; ``publish`` advertises it. Tests
    call them in either order, which is the whole point.
    """

    def __init__(self, counters):
        self.messages = []
        self.counters = counters

    def post(self, msg="m"):
        self.messages.append(msg)

    def publish(self):
        self.counters.bump_sent(CHAN_REQ)

    def send_correctly(self, msg="m"):
        # THE RULE: post, THEN publish.
        self.post(msg)
        self.publish()

    def send_with_the_ordering_mutated(self, msg="m"):
        # THE MUTANT: publish, THEN post. A receiver believing the counter
        # calls a blocking recv for a message that does not exist yet.
        self.publish()
        self.post(msg)


class _WireReceiver:
    """Mirrors PpChainReceiver's contract against ``_Wire``.

    ``consume_up_to`` is reproduced rather than imported because the point
    under test is the CONTRACT -- "block only once the counter says a
    message exists" -- and the real class's two-step framing is pinned
    separately in test_pp_chain_receiver.
    """

    def __init__(self, wire):
        self._wire = wire
        self.consumed = 0
        self.taken = []

    def _blocking_advance(self):
        if not self._wire.messages:
            raise _WouldBlockForever(
                "blocking recv on an empty wire: the counter advertised a "
                "message that was never posted"
            )
        self.taken.append(self._wire.messages.pop(0))
        self.consumed += 1

    def consume_up_to(self, sent_count):
        while self.consumed < int(sent_count):
            self._blocking_advance()


def test_the_service_loop_consumes_greedily_and_leaves_nothing_behind(tmp_path):
    """GREEDY IS THE DISTINCTION from the bounded-chain-recv corpse.

    That corpse completed iterations WITHOUT consuming while the upstream
    kept sending, so unmatched sends accumulated and the senders blocked.
    This loop takes every message the counter accounts for.
    """
    sender = _counters(tmp_path, rank=0)
    reader = _counters(tmp_path, rank=1)
    wire = _Wire(sender)
    rx = _WireReceiver(wire)
    for i in range(4):
        wire.send_correctly(f"m{i}")

    rx.consume_up_to(reader.sent(CHAN_REQ, 0))

    assert rx.taken == ["m0", "m1", "m2", "m3"]
    assert wire.messages == [], "a message was left on the wire"
    assert rx.consumed == 4


def test_a_receiver_never_blocks_when_the_sender_has_posted_nothing(tmp_path):
    """The quiescent case, and the one that runs at every idle boot: no
    counter movement means no blocking call at all."""
    sender = _counters(tmp_path, rank=0)
    reader = _counters(tmp_path, rank=1)
    rx = _WireReceiver(_Wire(sender))
    rx.consume_up_to(reader.sent(CHAN_REQ, 0))  # must not raise
    assert rx.consumed == 0


def test_can_fail_publishing_before_the_post_wedges_the_receiver(tmp_path):
    """CAN-FAIL FOR THE ORDERING CONSTRAINT -- the design's one axiom.

    Correct order (post, then publish): the only skew a peer can observe
    is counter-lags-send, i.e. a real message seen one poll late. Harmless.

    Mutated order (publish, then post): the skew inverts to
    send-lags-counter, and a receiver that believes the counter makes a
    BLOCKING call for a message nobody has posted. That is an unbounded
    block -- the exact wedge class this whole feature exists to remove.

    Both halves are asserted here, because a can-fail test that only shows
    the good case has not shown that the mutation matters.
    """
    sender = _counters(tmp_path, rank=0)
    reader = _counters(tmp_path, rank=1)

    good_wire = _Wire(sender)
    good = _WireReceiver(good_wire)
    good_wire.send_correctly("real")
    good.consume_up_to(reader.sent(CHAN_REQ, 0))
    assert good.taken == ["real"]

    bad_sender = _counters(tmp_path, rank=2)
    bad_reader = _counters(tmp_path, rank=1)
    bad_wire = _Wire(bad_sender)
    bad = _WireReceiver(bad_wire)
    # The mutant publishes first. Freeze the observation there -- exactly
    # what a peer polling at that instant sees.
    bad_wire.publish()
    advertised = bad_reader.sent(CHAN_REQ, 2)
    assert advertised == 1, "the mutated order does advertise the message"
    with pytest.raises(_WouldBlockForever):
        bad.consume_up_to(advertised)


def test_the_real_send_site_publishes_after_the_post(tmp_path):
    """THE ORDERING WHERE IT ACTUALLY LIVES, not only in a fake.

    The rule is only worth anything if the production call site obeys it,
    so this drives the real ``_pp_send_pyobj_to_next_stage`` and records
    the order of the two events.
    """
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    order = []

    class _Ps:
        attn_tp_rank = 0
        attn_cp_rank = 0
        attn_dp_rank = 0
        attn_cp_size = 1
        attn_tp_size = 1
        pp_rank = 0
        pp_size = 3
        tp_size = 1

    class _Counters:
        def bump_sent(self, chan):
            order.append(("publish", chan))

    class _Grp:
        cpu_group = None

    class S:
        ps = _Ps()
        world_group = _Grp()
        pp_flip_counters = _Counters()

    s = S()
    import sglang.srt.managers.scheduler_pp_mixin as mod

    def _fake_p2p(data, *a, **kw):
        order.append(("post", "req"))
        return ["work"]

    original = mod.point_to_point_pyobj
    mod.point_to_point_pyobj = _fake_p2p
    try:
        send = SchedulerPPMixin._pp_send_pyobj_to_next_stage.__get__(s, S)
        bump = SchedulerPPMixin._pp_flip_bump_sent.__get__(s, S)
        s._pp_flip_bump_sent = bump
        assert send(["req"], async_send=True) == ["work"]
    finally:
        mod.point_to_point_pyobj = original

    assert order == [("post", "req"), ("publish", CHAN_REQ)], (
        "the counter was published before the isend was posted; a peer "
        "polling in that window would block on a message that does not "
        "exist yet"
    )


# -- the flush half: reaping a send without blocking on a peer -----------------


class _CommitWire:
    """A downstream that has (or has not) taken the message off the wire.

    ``commit`` models ``_pp_commit_comm_work``: the measured wire fact is
    that the sender's ``wait()`` returns at once when the receiver has the
    message, and BLOCKS when the receiver has posted nothing. Blocking is
    raised here so a mutation terminates and fails rather than hanging.
    """

    def __init__(self):
        self.consumed_by_peer = False
        self.commits = 0

    def commit(self, work):
        if not self.consumed_by_peer:
            raise _WouldBlockForever(
                "blocking commit of a send the downstream has not taken"
            )
        self.commits += 1
        work.clear()


def _flush_harness(tmp_path, wire):
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin

    class _Ps:
        pp_rank = 0
        pp_size = 3

    class S:
        ps = _Ps()
        send_output_work = []
        send_proxy_work = []

    s = S()
    s.send_req_work = ["work"]
    s.pp_flip_counters = _counters(tmp_path, rank=0)
    s._pp_commit_comm_work = wire.commit
    for name in (
        "_pp_flip_downstream",
        "_pp_flip_upstream",
        "pp_flip_flush_drained_sends",
    ):
        setattr(s, name, getattr(SchedulerPPMixin, name).__get__(s, S))
    return s


def test_the_flush_holds_until_the_downstream_counter_covers_the_send(tmp_path):
    """The armed rank's own send is reaped by a BLOCKING commit -- but only
    once it is provably harmless.

    This is the working replacement for the send-side pump, which reaped
    nothing because ``is_completed()`` never fires for an isend here
    (corpse F). While the handle is unreaped this rank "owes a send",
    withholds presence, and the flip abandons at the presence deadline --
    so reaping is not tidiness, it is the difference between a flip that
    commits and a server that silently stops flipping.
    """
    wire = _CommitWire()
    s = _flush_harness(tmp_path, wire)
    s.pp_flip_counters.bump_sent(CHAN_REQ)

    # The downstream has not consumed yet: the flush must not touch it.
    s.pp_flip_flush_drained_sends()
    assert wire.commits == 0
    assert s.send_req_work == ["work"], "the send was reaped speculatively"

    # The downstream publishes its consumed count. Now the commit is
    # bounded -- the message is already off the wire.
    _counters(tmp_path, rank=1).bump_consumed(CHAN_REQ)
    wire.consumed_by_peer = True
    s.pp_flip_flush_drained_sends()
    assert wire.commits == 1
    assert s.send_req_work == [], "the send was not reaped once it was safe"


def test_can_fail_flushing_without_the_counter_gate_blocks_on_the_peer(tmp_path):
    """CAN-FAIL FOR THE FLUSH GATE.

    Drop the counter condition and the same call becomes the ordinary
    top-of-pass commit -- the exact ``work.wait()`` rank 1 was found
    blocked in while its peers sat in the reduction. The gate is what
    turns a speculative block into a bounded one.
    """
    wire = _CommitWire()
    s = _flush_harness(tmp_path, wire)
    s.pp_flip_counters.bump_sent(CHAN_REQ)
    assert s.pp_flip_counters.consumed(CHAN_REQ, 1) == 0

    with pytest.raises(_WouldBlockForever):
        # The mutant: commit regardless of what the downstream reports.
        s._pp_commit_comm_work(s.send_req_work)


# -- the armed intake: reaching the hook without a peer's traffic --------------


def _intake(armed, pp_rank, service_calls):
    from sglang.srt.managers.scheduler_components.request_receiver import (
        SchedulerRequestReceiver,
    )

    class _Ps:
        attn_tp_rank = 0
        attn_cp_rank = 0

    ps = _Ps()
    ps.pp_rank = pp_rank

    # A frozen dataclass: set the four fields this path reads directly.
    rx = SchedulerRequestReceiver.__new__(SchedulerRequestReceiver)
    for name, value in (
        ("ps", ps),
        ("chain_receiver", None),
        ("phase_flip_armed_hook", lambda: armed),
        ("phase_flip_service_hook", lambda: service_calls.append(pp_rank)),
    ):
        object.__setattr__(rx, name, value)
    return rx


@pytest.mark.parametrize("pp_rank", [0, 1, 2])
def test_the_armed_intake_services_and_never_blocks_on_any_rank(pp_rank):
    """CORPSE G, AT ITS ROOT.

    An armed rank must reach the flip's hook by its OWN poll. Blocking
    here for a message from an upstream that is itself armed -- and
    therefore issuing no forwards -- is the starvation that reproduced
    identically every epoch.

    Rank 0 is in the parametrisation deliberately. The armed rules used to
    be gated on the chain receiver, which exists only on ranks with an
    upstream, so they were off on exactly the rank that must stop
    admitting work for the group to reach a quiescent boundary at all.
    """
    calls = []
    rx = _intake(armed=True, pp_rank=pp_rank, service_calls=calls)
    assert rx._pull_raw_reqs() == [], (
        "an armed rank admitted work or returned None; empty means 'no new "
        "work this pass', which every later step already handles"
    )
    assert calls == [pp_rank], (
        "the armed intake skipped its service turn on this rank"
    )


def test_the_receiver_wiring_actually_publishes_the_consumed_count(tmp_path):
    """THE WIRING, not just the pieces. Metal boot 2026-08-09 01:00Z.

    Every unit above passed while the live system never published a single
    consumed count: the callback that ``Scheduler._build_pp_chain_receiver``
    hands to the receiver raised ``NameError`` on its first call, was
    caught as best-effort, and logged. The upstream could therefore never
    learn its send had been taken, withheld presence for ever, and all
    three epochs abandoned at the 60 s deadline.

    Nothing in the unit suite touched that lambda. This test builds the
    receiver THROUGH the real factory and drives one message through the
    real state machine, so the callback is executed rather than merely
    constructed.
    """
    from sglang.srt.managers import pp_chain_receiver as rxmod
    from sglang.srt.managers.scheduler import Scheduler

    class _Ps:
        pp_size = 3
        pp_rank = 1
        tp_size = 1
        attn_tp_rank = 0
        attn_cp_rank = 0
        attn_dp_rank = 0
        attn_cp_size = 1
        attn_tp_size = 1

    class _Args:
        enable_phase_flip = True

    class _Grp:
        cpu_group = None

    class S:
        ps = _Ps()
        server_args = _Args()
        world_group = _Grp()

    s = S()
    s.pp_flip_counters = _counters(tmp_path, rank=1)
    rx = Scheduler._build_pp_chain_receiver.__get__(s, S)()
    assert rx is not None, "the receiver must be built when the flip is on"

    # Drive one whole message through the real machine.
    monkey = _FakeSizeZeroDist()
    original = rxmod.dist
    rxmod.dist = monkey
    try:
        rx._advance(block=True)
    finally:
        rxmod.dist = original

    assert rx.publish_failures == 0, (
        "the consumed-counter callback raised; the upstream is then blind "
        "to this rank's progress and every flip abandons at the deadline"
    )
    assert rx.consumed == 1
    assert _counters(tmp_path, rank=0).consumed(CHAN_REQ, 1) == 1, (
        "the consumed count never reached /dev/shm, so no peer can read it"
    )


class _FakeSizeZeroDist:
    """Hands out one empty (size-0) chain message, which is a whole
    message: an empty forward still carries the pass."""

    def irecv(self, tensor, src=None, group=None):
        class _W:
            def wait(self_inner):
                tensor.zero_()

            def is_completed(self_inner):
                return True

        return _W()
