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


def test_every_kind_the_wire_actually_carries_is_declared():
    """THE TABLE KEYS ON RAW STRINGS THAT LIVE SOMEWHERE ELSE.

    `stash_flip_disposition` matches literals, while the senders use constants
    defined in two other modules. Renaming `ADMISSION_DECISION_KIND` would not
    break a single import -- it would quietly move that kind into UNDECLARED,
    where it blocks presence again for 20 s per flip and is then RETIRED. That
    is the wedge coming back wearing the escape hatch as a hat, and no other
    test in this file would notice.

    So the constants are read from where the senders read them, and every kind
    the wire actually carries has to be declared.
    """
    from sglang.srt.distributed.pp_typed_channel import CROSSING_KIND
    from sglang.srt.managers.scheduler_pp_mixin import ADMISSION_DECISION_KIND

    for kind in (ADMISSION_DECISION_KIND, CROSSING_KIND, "output", "proxy"):
        assert stash_flip_disposition(kind) != UNDECLARED, (
            f"{kind!r} travels on this wire but has no declared flip "
            "disposition; it would block the presence gate and then be retired "
            "as unplaceable"
        )
    assert ADMISSION_DECISION_KIND in declared_stash_kinds()
    assert CROSSING_KIND in declared_stash_kinds()


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


# ------------------------------------------------------------- the escape hatch


# ----------------------------------------------------------------- the cutover


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


def test_the_cutover_instrument_splits_owed_from_pp_loop_only():
    """The instrument said one number for three different fates.

    Its heading calls the inbox figure "a sampled token that reaches no
    output_ids". That is true of an owed entry and false of a PP-loop-only one,
    which carries no token and is retired on purpose. Pinned in the compiled
    body, where a comment cannot stand in for a format string.
    """
    consts = [c for c in _cutover_bytecode() if isinstance(c.argval, str)]
    blob = "\n".join(c.argval for c in consts)
    assert "inbox_owed=" in blob, (
        "the cutover instrument still reports one undifferentiated inbox count "
        "under a heading that calls all of it lost tokens"
    )
    assert "inbox_pp_loop=" in blob, (
        "the routine PP-loop-only retirement is not reported, so a designed "
        "discard is indistinguishable from silence"
    )


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


