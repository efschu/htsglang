"""#800: a stashed message must not block the presence gate that feeds it.

THE METAL FACT THIS ENCODES. 2026-08-22, twice, both times on PP1 -- specimens
``/spinning/evidence-665-f1/wedge_1208_120909/boot.log`` (epoch 4) and
``/spinning/evidence-665-f1/boot_r8_0822_1210.log`` (epoch 2):

    PP1] #757 armed drain took a tensor dict off the wire and STASHED it:
         kind=admission_decision stamp=None
    PP1] PHASE-FLIP epoch 4 round 0: WITHHOLDING presence (57922 rounds so far)
         -- tensor-dict inbox holds 1 stashed message(s).
    PP*] PHASE-FLIP FLIP ABANDONED (no quorum): pp_to_tp waited 60.0s for
         epoch 4 and rank(s) [1] never reached the flip entry

repeated every 60 s for five minutes: abandon, re-arm, withhold, abandon. The
instance answered nothing while its port stayed open.

IT IS ONE DEFECT SEEN FROM TWO SIDES, not two. A rank is in the abandonment's
``missing`` list exactly when it did not ``announce``, and the only branch that
skips ``announce`` for a rank that HAS reached the gate is the withhold branch.
The abandonment then advised "look upstream, not at the flip" -- which the
withhold line contradicts in its own text.

THE CYCLE, and every link is individually right:
  1. An armed rank must keep servicing the wire, or its upstream blocks.
  2. #757 stashes what it cannot prove void, or it re-enters corpse S.
  3. #791/#795 put a third kind on that wire, sent every pass.
  4. The gate counts every stashed message, or an owed output crosses the
     cutover and a client loses a token.
The only consumer of kind #3 is at the top of a PP pass, which link 4 prevents
from running. The gate waits for a consumer the gate is blocking.

WHAT IS TESTED HERE, and each guard is proved in BOTH directions:
  * the shipped probe stops blocking on a PP-loop-only kind, and STILL blocks on
    an output and on an undeclared one,
  * the shipped GATE announces with the one and withholds with the other, wired
    through the real ``_channels_empty_fn`` call edge,
  * the undeclared escape retires on its deadline, does nothing before it, does
    nothing at all when switched off, and never touches a declared kind,
  * the shipped ``pp_flip_service`` really calls the escape,
  * the cutover retires the PP-loop-only stash and refuses to sweep a blocking
    one,
  * a channel probe that RAISES withholds instead of announcing -- one return
    value used to mean both "nothing to report" and "I could not tell".

CPU-only: every path under test is a pure function or a mixin method bound to a
holder, exactly as test_pp_flip_leftover_proxy_757 and test_pp_void_slot_advance
_798 bind theirs.
"""

import types
from collections import defaultdict, deque

import pytest

from sglang.srt.managers.pp_stash_disposition import (
    BLOCKS_FLIP,
    PP_LOOP_ONLY,
    UNDECLARED,
    census_stash,
    declared_stash_kinds,
    stash_flip_disposition,
)
from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20)

UPSTREAM = 0

#: The specimen's own kind and depth. Red-first means red on THESE.
STASHED_KIND = "admission_decision"


def _holder(inbox=None, clock=None):
    """A stand-in carrying only what the probe and the escape read.

    The methods below are the SHIPPED ones, bound here the way this suite's
    neighbours bind theirs. If the fix stops being wired into them, nothing in
    this rig papers over it.
    """
    h = types.SimpleNamespace(
        pp_flip_counters=types.SimpleNamespace(
            sent=lambda chan, rank: 0,
            local_consumed=lambda chan: 0,
        ),
        pp_chain_receiver=None,
        send_req_work=None,
        send_output_work=None,
        send_proxy_work=None,
        last_rank_comm_queue=None,
        pp_outputs=None,
    )
    h._pp_tensor_dict_inbox = inbox if inbox is not None else defaultdict(deque)
    h._pp_flip_upstream = lambda: UPSTREAM
    if clock is not None:
        h._pp_stash_clock = clock
    for name in (
        "pp_flip_channels_empty",
        "pp_flip_retire_undeclared_stash",
        "pp_flip_retire_pp_loop_stash",
    ):
        setattr(h, name, types.MethodType(getattr(SchedulerPPMixin, name), h))
    return h


def _inbox(kind, depth=1, src=UPSTREAM):
    box = defaultdict(deque)
    for _ in range(depth):
        box[(src, kind)].append({"__msg_type__": kind})
    return box


# ---------------------------------------------------------------- the contract


def test_the_contract_names_every_kind_that_travels_on_this_wire():
    """A kind with no declared disposition must be UNDECLARED, not silently ok.

    The three kinds that carry payload block; the admission decision does not;
    anything nobody classified lands in the named third state. A table that
    answered only "block / do not block" would make a future fourth kind
    indistinguishable from a declared one -- which is how this seam was built.
    """
    assert stash_flip_disposition("output") == BLOCKS_FLIP
    assert stash_flip_disposition("proxy") == BLOCKS_FLIP
    assert stash_flip_disposition("crossing") == BLOCKS_FLIP
    assert stash_flip_disposition(STASHED_KIND) == PP_LOOP_ONLY
    assert stash_flip_disposition("some_kind_added_next_month") == UNDECLARED
    assert stash_flip_disposition("default") == UNDECLARED
    assert stash_flip_disposition(None) == UNDECLARED
    assert set(declared_stash_kinds()) == {
        "output",
        "proxy",
        "crossing",
        STASHED_KIND,
    }


def test_the_census_never_answers_none_for_two_different_states():
    """Empty and gate-blind are different facts and must read differently."""
    empty = census_stash({})
    assert empty.block_reason() is None
    assert empty.blocking_total == 0 and empty.gate_blind_total == 0

    blind = census_stash(_inbox(STASHED_KIND))
    assert blind.block_reason() is None, "a gate-blind message must not block"
    assert blind.gate_blind_total == 1, (
        "the gate-blind message vanished from the census; 'does not block' must "
        "not be implemented as 'is not there'"
    )


# ------------------------------------------------------- the shipped probe


def test_an_admission_decision_no_longer_withholds_presence():
    """THE DEFECT, reproduced against the shipped probe. RED before the fix."""
    h = _holder(_inbox(STASHED_KIND))
    assert h.pp_flip_channels_empty() is None, (
        "the shipped hygiene probe still reports a stashed admission_decision "
        "as a non-empty channel, so the rank withholds presence and waits for "
        "a consumer that only runs at the top of a PP pass -- which this very "
        "withhold prevents. That is the 2026-08-22 wedge"
    )


def test_can_fail_an_output_still_withholds_presence():
    """THE GUARD MUST STILL FIRE. Corpse S is one over-broad exemption away.

    An output stashed in the inbox is a sampled token whose consumer looks for
    it after the cutover. If this ever stops blocking, a client loses a token
    per crossing (#631) and this file caused it.
    """
    h = _holder(_inbox("output"))
    why = h.pp_flip_channels_empty()
    assert why is not None, "an owed output stopped blocking the flip"
    assert "output" in why, f"the reason no longer names the kind: {why}"


def test_can_fail_an_undeclared_kind_still_withholds_and_says_so_by_name():
    """The conservative half of the contract, and it must be LOUD.

    A kind nobody classified is held -- it might be an owed payload -- but the
    reason has to say that nobody classified it, or the next reader repeats the
    log-dig this defect already cost once.
    """
    h = _holder(_inbox("a_kind_from_the_future"))
    why = h.pp_flip_channels_empty()
    assert why is not None, "an undeclared kind was waved through the gate"
    assert "UNDECLARED" in why and "a_kind_from_the_future" in why, (
        f"the undeclared reason does not name its state or its kind: {why}"
    )


def test_a_mixed_inbox_blocks_on_the_blocking_half_only():
    """Both kinds present: the output blocks, and the decision is not the cause."""
    box = _inbox("output")
    box[(UPSTREAM, STASHED_KIND)].append({"__msg_type__": STASHED_KIND})
    why = _holder(box).pp_flip_channels_empty()
    assert why is not None and "output" in why
    assert STASHED_KIND not in why, (
        f"the gate blamed the admission decision for a block the output caused: {why}"
    )


# --------------------------------------------------- the shipped presence gate


def _presence(tmpdir, n_ranks=3, rank=0):
    from sglang.srt.managers.phase_flip_presence import PhaseFlipPresence

    return PhaseFlipPresence(
        n_ranks=n_ranks, rank=rank, directory=str(tmpdir), instance="test"
    )


def _gate(presence, channels_empty_fn, deadline=60.0, clock=None):
    """The shipped gate, wired to a real probe -- the CALL EDGE under test.

    Mirrors test_phase_policy's `_runtime_stub`; the point of duplicating it is
    that `channels_empty_fn` here is the SHIPPED `pp_flip_channels_empty`, not a
    lambda. A fix that lives only in the probe and never reaches the gate fails
    here and nowhere else.
    """
    from sglang.srt.managers.phase_flip_runtime import PhaseFlipRuntime
    from sglang.srt.managers.phase_policy import PHASE_PP

    class R:
        pass

    r = R()
    r._presence = presence
    r._pump_fn = None
    r._drain_fn = None
    r._owes_send_fn = None
    r._service_fn = None
    r._channels_empty_fn = channels_empty_fn
    r.presence_withheld_rounds = 0
    r.presence_withheld_channels = 0
    r.entry_channel_violations = 0
    r._last_withhold_log = None
    r._last_not_ready_log = None
    r._log_not_ready = lambda: None
    r._entry_round = 0
    r._presence_wait_stamp = None
    r._presence_deadline_s = deadline
    r._presence_wait_started = None
    r._gate_open_epoch = None
    r._epoch = 1
    r._pending = "pp_to_tp"
    r._armed_at = 0.0
    r._last_hold_reason = None
    r._phase = PHASE_PP
    r.presence_timeouts = 0
    r._clock = clock or (lambda: 0.0)
    r._sleep = lambda _s: None
    r._presence_poll_interval_s = 0.0
    for name in ("_await_group_presence", "_abandon_no_quorum", "_commit_to_entering"):
        setattr(r, name, getattr(PhaseFlipRuntime, name).__get__(r, R))
    return r


def test_the_gate_announces_with_an_admission_decision_stashed(tmp_path):
    """THE WEDGE, through the real gate. RED before the fix.

    This is the specimen's exact state: rank at the entry, one
    admission_decision in the inbox, nothing else outstanding.
    """
    presence = _presence(tmp_path, rank=0)
    holder = _holder(_inbox(STASHED_KIND))
    gate = _gate(presence, holder.pp_flip_channels_empty)

    gate._await_group_presence()
    assert presence.observe(1, round_=0) == {0}, (
        "the rank withheld presence with only an admission_decision stashed. "
        "It will withhold for ever: the message's consumer runs at the top of "
        "a PP pass, and the withhold is what stops that pass"
    )
    assert gate.presence_withheld_rounds == 0


def test_can_fail_the_gate_still_withholds_with_an_output_stashed(tmp_path):
    """Same wiring, owed payload: the gate must still refuse to announce."""
    presence = _presence(tmp_path, rank=0)
    holder = _holder(_inbox("output"))
    gate = _gate(presence, holder.pp_flip_channels_empty)

    gate._await_group_presence()
    assert presence.observe(1, round_=0) == set(), (
        "a rank announced while an owed output sat in its inbox; the cutover "
        "would then discard a sampled token"
    )
    assert gate.presence_withheld_channels == 1


def test_the_gate_recovers_as_soon_as_the_output_is_taken(tmp_path):
    """Convergence, not merely refusal: the withhold must be releasable."""
    presence = _presence(tmp_path, rank=0)
    box = _inbox("output")
    holder = _holder(box)
    gate = _gate(presence, holder.pp_flip_channels_empty)

    gate._await_group_presence()
    assert presence.observe(1, round_=0) == set()
    box[(UPSTREAM, "output")].clear()
    gate._await_group_presence()
    assert presence.observe(1, round_=0) == {0}


# ------------------------------------------------------------- the escape hatch


def test_an_undeclared_kind_is_retired_once_the_escape_deadline_expires(monkeypatch):
    """THE ACTUATOR. Before this, an unplaceable message had no exit at all.

    Held before the deadline (it might be owed), retired after it -- and the
    probe must actually come up clean afterwards, or the escape moved a queue
    without unblocking anything.
    """
    monkeypatch.setenv("SGLANG_PP_STASH_ESCAPE_S", "20.0")
    now = {"t": 0.0}
    box = _inbox("a_kind_from_the_future")
    h = _holder(box, clock=lambda: now["t"])

    assert h.pp_flip_retire_undeclared_stash() == 0
    assert h.pp_flip_channels_empty() is not None, "retired before its deadline"

    now["t"] = 19.9
    assert h.pp_flip_retire_undeclared_stash() == 0, "retired one tick early"

    now["t"] = 20.1
    assert h.pp_flip_retire_undeclared_stash() == 1, (
        "the escape deadline expired and nothing was retired; an unplaceable "
        "message still has no exit"
    )
    assert h.pp_flip_channels_empty() is None, (
        "the entry was retired but the gate still blocks -- the escape has to "
        "unblock the rank, not merely empty a deque"
    )


def test_can_fail_the_escape_does_nothing_when_switched_off(monkeypatch):
    """The off-switch, which is what makes the guard provable in both senses.

    If this arm also retires, the retirement is being driven by something other
    than the deadline and the test above proves nothing.
    """
    monkeypatch.setenv("SGLANG_PP_STASH_ESCAPE_S", "0")
    now = {"t": 0.0}
    h = _holder(_inbox("a_kind_from_the_future"), clock=lambda: now["t"])
    # TWO turns with time between them. One turn can never retire anything --
    # the clock starts at the message's arrival -- so a single-call version of
    # this test would pass against a build with no deadline logic at all.
    assert h.pp_flip_retire_undeclared_stash() == 0
    now["t"] = 1e6
    assert h.pp_flip_retire_undeclared_stash() == 0
    assert h.pp_flip_channels_empty() is not None


@pytest.mark.parametrize("kind", ["output", "proxy", "crossing", STASHED_KIND])
def test_can_fail_the_escape_never_retires_a_declared_kind(monkeypatch, kind):
    """CORPSE S WITH A CLOCK ON IT is still corpse S.

    An owed output does not become void by waiting, and an admission decision
    must survive an abandon intact -- the resumed loop pops exactly one per
    pass, so dropping one puts every later receive off by one, permanently.
    """
    monkeypatch.setenv("SGLANG_PP_STASH_ESCAPE_S", "1.0")
    now = {"t": 0.0}
    box = _inbox(kind)
    h = _holder(box, clock=lambda: now["t"])
    # TWO turns, deadline crossed between them. With one turn this test is
    # vacuous: nothing is retired on arrival whatever the disposition, so it
    # would stay green against a build that retires every kind it sees.
    assert h.pp_flip_retire_undeclared_stash() == 0
    now["t"] = 1e6
    assert h.pp_flip_retire_undeclared_stash() == 0, (
        f"the escape retired a declared kind ({kind})"
    )
    assert len(box[(UPSTREAM, kind)]) == 1


def test_the_escape_clock_restarts_when_the_key_empties(monkeypatch):
    """A kind consumed and stashed again must be timed from its own arrival."""
    monkeypatch.setenv("SGLANG_PP_STASH_ESCAPE_S", "20.0")
    now = {"t": 0.0}
    box = _inbox("a_kind_from_the_future")
    h = _holder(box, clock=lambda: now["t"])

    h.pp_flip_retire_undeclared_stash()
    now["t"] = 10.0
    box[(UPSTREAM, "a_kind_from_the_future")].clear()
    h.pp_flip_retire_undeclared_stash()
    box[(UPSTREAM, "a_kind_from_the_future")].append({"__msg_type__": "x"})
    now["t"] = 25.0
    assert h.pp_flip_retire_undeclared_stash() == 0, (
        "a freshly stashed message inherited its predecessor's age and was "
        "retired 5s after arriving on a 20s deadline"
    )


def test_the_shipped_service_turn_runs_the_escape(monkeypatch):
    """THE CALL EDGE, not just the helper.

    `pp_flip_service` is the armed loop's one service turn and the only place
    the escape can run. Bound for real: if the call is removed or reordered
    after the probe, this fails and the helper tests do not.
    """
    monkeypatch.setenv("SGLANG_PP_STASH_ESCAPE_S", "1.0")
    now = {"t": 0.0}
    box = _inbox("a_kind_from_the_future")
    h = _holder(box, clock=lambda: now["t"])
    h.pp_flip_consume_inbound = lambda: None
    h.pp_flip_drain_tensor_dicts = lambda: 0
    h.pp_flip_flush_drained_sends = lambda: None
    h.pp_flip_service = types.MethodType(SchedulerPPMixin.pp_flip_service, h)

    # Two turns with the deadline between them: the escape times a message from
    # its ARRIVAL, so a single turn can never retire anything however late the
    # clock reads. That is the property, not an inconvenience of the rig.
    h.pp_flip_service()
    assert box[(UPSTREAM, "a_kind_from_the_future")], "retired on arrival"
    now["t"] = 5.0
    h.pp_flip_service()
    assert not box[(UPSTREAM, "a_kind_from_the_future")], (
        "the shipped service turn did not run the undeclared-stash escape, so "
        "the escape is a helper nothing calls"
    )


# ----------------------------------------------------------------- the cutover


def test_the_cutover_retires_the_pp_loop_stash():
    """A PP-loop-only message cannot survive the ring it names.

    `init_pp_loop_state` stopped clearing this inbox at #753 (it moved onto the
    pp_group), so without this sweep the message outlives the whole TP phase
    and is handed to the NEXT PP epoch's receive.
    """
    box = _inbox(STASHED_KIND, depth=2)
    h = _holder(box)
    assert h.pp_flip_retire_pp_loop_stash() == 2
    assert not box[(UPSTREAM, STASHED_KIND)]


def test_can_fail_the_cutover_does_not_sweep_a_blocking_stash(caplog):
    """A blocking stash at a cutover is a predicate bug, not litter.

    Sweeping it here would hide the very failure the presence gate exists to
    make impossible, so it is reported and LEFT.
    """
    box = _inbox("output")
    h = _holder(box)
    with caplog.at_level("ERROR"):
        assert h.pp_flip_retire_pp_loop_stash() == 0
    assert len(box[(UPSTREAM, "output")]) == 1, "the cutover swept an owed output"
    assert any("CUTOVER FOUND A BLOCKING STASH" in r.message for r in caplog.records), (
        "a blocking stash survived a cutover silently"
    )


def _cutover_bytecode():
    """The compiled body of the cutover closure, not its source text.

    A SOURCE pin is not good enough here and this suite learned it from its own
    mutation run: deleting the call while leaving the name in a comment kept a
    text search green. Bytecode carries no comments, so a name that appears
    here appears because the code loads it.
    """
    import dis

    from sglang.srt.managers import phase_flip_runtime

    outer = phase_flip_runtime.build_production_flip_cutover.__code__
    inner = [
        c for c in outer.co_consts if hasattr(c, "co_name") and c.co_name == "_cutover"
    ]
    assert inner, "build_production_flip_cutover no longer defines _cutover"
    return list(dis.get_instructions(inner[0]))


def test_the_cutover_call_site_runs_before_the_ring_is_rebuilt():
    """CALL-EDGE PIN for the one edge this suite cannot execute.

    The cutover closure needs a whole live scheduler, so the wiring is pinned
    in the compiled body instead: the retirement must be loaded, and it must be
    loaded BEFORE `init_pp_loop_state`, which is what destroys the ring the
    retired messages name. Pinning order matters as much as presence -- a sweep
    that ran after the rebuild would sweep a different ring's inbox.
    """
    instructions = _cutover_bytecode()

    def first_offset(name):
        for ins in instructions:
            if ins.argval == name:
                return ins.offset
        return None

    retire_at = first_offset("pp_flip_retire_pp_loop_stash")
    init_at = first_offset("init_pp_loop_state")
    assert retire_at is not None, (
        "the cutover does not retire the PP-loop-only stash. Since #753 nothing "
        "else clears this inbox, so the message survives into the next phase "
        "and is handed to a later epoch's receive"
    )
    assert init_at is not None, "the cutover no longer rebuilds the PP ring"
    assert retire_at < init_at, (
        "the retirement runs after the ring rebuild, so it sweeps the wrong "
        "ring's inbox"
    )


# ------------------------------------------------- a probe that cannot answer


def test_a_raising_channel_probe_withholds_instead_of_announcing(tmp_path):
    """ "I could not tell" is not "nothing to report". RED before the fix.

    The probe's result used to be None in both cases, so a probe that raised
    made the rank announce and enter the reduction with whatever live state the
    probe had been about to report.
    """
    presence = _presence(tmp_path, rank=0)

    def _boom():
        raise RuntimeError("the inbox blew up")

    gate = _gate(presence, _boom)
    gate._await_group_presence()
    assert presence.observe(1, round_=0) == set(), (
        "a rank announced on the strength of a channel probe that RAISED"
    )
    assert gate.presence_withheld_channels == 1


def test_a_raising_channel_probe_still_abandons_on_the_deadline(tmp_path):
    """And it must stay BOUNDED: loud abandon, never a new silent wait."""
    presence = _presence(tmp_path, rank=0)
    now = {"t": 0.0}

    def _boom():
        raise RuntimeError("the inbox blew up")

    gate = _gate(presence, _boom, deadline=30.0, clock=lambda: now["t"])
    for _ in range(4):
        gate._await_group_presence()
        now["t"] += 12.0
    assert gate._pending is None, "a rank with an unanswerable probe never left"
    assert gate.presence_timeouts == 1
    # AND it withheld on every round on the way there. Without this line the
    # case passes against a build that announces on a raising probe too -- a
    # lone rank abandons on its deadline either way, so the abandonment alone
    # separates nothing.
    assert gate.presence_withheld_channels == 4, (
        "the rank announced on a probe that could not answer; only the absent "
        "peers, not the fix, produced the abandonment above"
    )


def test_a_raising_entry_probe_abandons_before_entering(tmp_path):
    """The entry re-check has the same two states and the same old conflation."""
    presence = _presence(tmp_path, rank=0)
    for peer in (1, 2):
        _presence(tmp_path, rank=peer).announce(1, round_=0)

    calls = {"n": 0}

    def _clean_then_boom():
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        raise RuntimeError("the inbox blew up at the instant of entry")

    gate = _gate(presence, _clean_then_boom)
    assert gate._await_group_presence() is None, (
        "the gate ENTERED the reduction on an entry probe that raised"
    )
    assert gate.entry_channel_violations == 1
    assert presence.entering(1, round_=0) == set()


def test_the_abandonment_names_this_rank_s_own_withhold(tmp_path, caplog):
    """The indicator that read the same in two different states.

    The abandonment used to assert "a rank that never reaches the entry is
    blocked upstream of it", which is one of the TWO ways to be missing and was
    the wrong one in both 2026-08-22 wedges. A rank that withheld knows it and
    must say so.
    """
    presence = _presence(tmp_path, rank=0)
    now = {"t": 0.0}
    gate = _gate(
        presence,
        lambda: "send_output_work is not reaped",
        deadline=30.0,
        clock=lambda: now["t"],
    )
    with caplog.at_level("ERROR"):
        for _ in range(4):
            gate._await_group_presence()
            now["t"] += 12.0
    abandons = [r.message for r in caplog.records if "FLIP ABANDONED" in r.message]
    assert abandons, "no abandonment was logged"
    assert "THIS rank withheld its own presence" in abandons[-1], (
        f"the abandonment still sends every reader upstream: {abandons[-1]}"
    )
