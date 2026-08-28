"""#955: the terminator is re-armed from state it never touches.

THE DEFECT, MEASURED BEFORE IT WAS READ. window-951-boot, pin 2897161bdb, two
boots, byte-identical counters: 87 ``#946 PREMISE RECOMPUTE`` lines, 85 of them
for ONE rid, each discarding 8192 tokens -- 696,320 tokens re-prefilled for a
single request in 14 seconds, against a standing law that allows at most ONE
chunk of loss. Each recompute was preceded by exactly three
``#946 REFETCH DECLINED ... reason=store_absent`` lines (streak 1/3, 2/3, 3/3),
so the loop has a period of four passes and it ran until the instance died.

THE MECHANISM, read at the code rather than inferred from the counters.
``pp_apply_dead_premise_at_chunk_boundary`` spends the terminator CORRECTLY:
it deletes ``_PREMISE_DEAD_STAMP`` and ``_REFETCH_DECLINE_STREAK`` from the
request (scheduler_pp_mixin.py:1562-1569), which is an operation in target
state and exactly what #949 established. It is re-armed from a DIFFERENT,
durable place it never touches: ``scheduler.py:9324-9326`` calls
``pp_mark_premise_dead`` unconditionally, every pass, for every rid the guard
reports as escalated -- and ``_escalated`` is discarded at exactly one site,
``pp_admission_congruence.py:607``, on the ``elif entry.admitted:`` branch of
``record_return_trip``, i.e. only after a completed PP ring round-trip.

THE CIRCLE, AND WHY IT IS THIS FAMILY'S NINTH INSTANCE. The escalation's own
consequence -- mark dead, decline the re-fetch three times, spend the
terminator, void the pass -- is what PREVENTS the round-trip that is the only
thing able to clear the flag. The one operation that can end the lifecycle is
unreachable from the path the lifecycle creates. #944b already learned this
lesson for the INCREMENT and made ``_offer_streak`` lap-free; the CLEAR was
left lap-gated, and pp_admission_congruence.py:608-613 states that asymmetry as
deliberate ("the CLEAR only has to work when the ring turns"). The premise of
that sentence is that a pass which serves the rid is reachable. When the
escalation itself is what stops the rid being served, it is not.

WHAT THIS FILE PINS.

1. THE LAW, DIRECTLY. Over a run of passes with no genuine premise change and
   no admitted lap, a single rid may collect AT MOST ONE terminator spend.
   That is the standing Kein-Doppel-Prefill rule stated as an executable
   invariant rather than as a comment: a named emergency exit, once, with its
   discard count -- never a loop.

2. THE RE-ARM ITSELF, at the guard's own surface: after the terminator has
   been spent, the guard must not still be reporting the rid as escalated,
   because that report is what re-writes the mark on the next pass.

3. BOTH MECHANISM BRANCHES, because the boot log leaves one question open and
   a test that pins only my favourite answer would pass for the wrong reason.
   The offer that reaches the wire is ``extend_range.start``
   (``_executed_extent``, pp_admission_congruence.py:651-659), written by
   ``set_extend_range(len(req.prefix_indices), ...)`` (schedule_policy.py:1763,
   :1806) -- i.e. AFTER the host load-back at schedule_policy.py:1733 has had
   its chance to re-grow the very prefix the clamp truncated. So the second
   ``note_offer`` of a pass may see either the clamped value or the restored
   one, and the specimen (``told=0`` exactly 3 times against 349 ``told=8192``)
   does not settle which. Both are driven here, and the invariant must hold in
   both.

4. THE COUNTER-DIRECTION, which is what stops this being a fix that simply
   switches the escape off: a rid whose offer genuinely MOVES is still allowed
   to escalate, an admitted lap still resets everything, a healthy rid is never
   marked at all, and the success path is untouched.

HERMETIC. No CUDA, no ranks, no collective: the guard is a pure object, and the
mark/actuator pair are module-level functions over a request. The only runtime
dependency is ``hicache_phase_binding.current_generation()``, which answers on
the desk.
"""

import importlib.util
import io
import tokenize
from pathlib import Path

import pytest

from sglang.srt.managers.pp_admission_congruence import (
    UNRESOLVED_DEFER_CAP,
    PPAdmissionCongruenceGuard,
    PPAdmissionDecision,
    PPAdmissionEntry,
)
from sglang.srt.managers.scheduler_pp_mixin import (
    REFETCH_DECLINE_CAP,
    _PREMISE_DEAD_STAMP,
    pp_apply_dead_premise_at_chunk_boundary,
    pp_mark_premise_dead,
)

RID = "371400de4bc045c7b5c6c7f9dc905686"  # the rid from the specimen
TOLD = 8192  # its frozen offer, on every one of 344 lines


class _Req:
    """The two fields the escape actually reads, and nothing else.

    ``prefix_indices`` is a LIST here on purpose. The production object holds a
    torch tensor, and ``phase_flip_draft_bootstrap.prefix_len`` -- which the
    terminator uses to name its discard count -- exists precisely because
    ``x or ()`` on a tensor raises. A list exercises the same call without
    importing torch into a desk test; the tensor half is already pinned by
    test_pp_admission_prefix_indices_tensor_796.py.
    """

    def __init__(self, rid=RID, prefix=TOLD):
        self.rid = rid
        self.prefix_indices = list(range(prefix))

    def truncate_prefix_to(self, told: int) -> None:
        told = int(told)
        if told < len(self.prefix_indices):
            self.prefix_indices = self.prefix_indices[:told]


class _Holder:
    """A scheduler stand-in whose re-fetch always declines the way metal did.

    ``store_absent`` is not an arbitrary choice of failure: it is the reason
    reported on all 261 declines of both boots, and it is the reason the cheap
    escape can never succeed, which is what forces the terminator to be the
    only exit.
    """

    def __init__(self, guard=None, verdict="declined:store_absent"):
        self._pp_admission_guard = guard
        self._verdict = verdict
        self.issued = 0

    def _prefetch_kvcache(self, req):
        if self._verdict == "issued":
            self.issued += 1
        return self._verdict


def _one_pass(guard, holder, req, candidate, *, executed_sees_clamp):
    """One PP0 prefill pass, in the order the scheduler runs it.

    The four steps are the production sequence, and each names its site:

      1. ``prefix_len_for``           scheduler.py:9074   (the clamping caller)
      2. ``note_offer`` on the built  pp_admission_congruence.py:783
         geometry                     (the EXECUTED branch: counts, never clamps)
      3. the re-mark                  scheduler.py:9324-9326
      4. the actuator                 scheduler_pp_mixin.py:1632

    ``executed_sees_clamp`` is the open question of #955 made into a parameter:
    True models a pass where the truncation survives to ``set_extend_range``,
    False models one where the host load-back (schedule_policy.py:1733) has
    re-grown the prefix first, so the executed geometry still reports the
    unclamped length. The invariant is asserted under both.
    """
    effective = guard.prefix_len_for(req.rid, candidate)
    guard.note_offer(req.rid, effective if executed_sees_clamp else candidate)
    if guard.is_escalated(req.rid):
        pp_mark_premise_dead(req)
    outcome = pp_apply_dead_premise_at_chunk_boundary(holder, req)
    return effective, outcome


def _run(passes, *, executed_sees_clamp, verdict="declined:store_absent"):
    guard = PPAdmissionCongruenceGuard()
    holder = _Holder(guard=guard, verdict=verdict)
    req = _Req()
    outcomes = []
    offers = []
    for _ in range(passes):
        effective, outcome = _one_pass(
            guard, holder, req, TOLD, executed_sees_clamp=executed_sees_clamp
        )
        offers.append(effective)
        outcomes.append(outcome)
    return guard, holder, req, offers, outcomes


# --------------------------------------------------------------------------
# 1. THE LAW: at most one terminator spend per rid without a premise change.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("executed_sees_clamp", [True, False])
def test_terminator_spends_at_most_once_without_new_evidence(executed_sees_clamp):
    """A rid that never changes its premise gets ONE recompute, not a stream.

    RED BEFORE #955: the loop re-marks from ``_escalated`` every pass, so the
    actuator reaches its terminator once per ``REFETCH_DECLINE_CAP + 1`` passes
    for as long as the instance lives. Forty passes is 14 seconds of the
    measured boot compressed into a desk test.
    """
    _, _, _, _, outcomes = _run(40, executed_sees_clamp=executed_sees_clamp)
    recomputes = outcomes.count("recompute")
    assert recomputes <= 1, (
        f"the terminator spent {recomputes} times for one rid with no premise "
        f"change and no admitted lap; the standing law allows at most one "
        f"named recompute per rid. Outcome sequence: {outcomes}"
    )


@pytest.mark.parametrize("executed_sees_clamp", [True, False])
def test_escalation_does_not_survive_its_own_terminator(executed_sees_clamp):
    """After the terminator is spent, the guard must stop reporting escalated.

    This is the re-arm at its source. ``is_escalated`` is read at
    scheduler.py:9325 and its True is what re-writes the mark, so a terminator
    that leaves it True has not ended anything -- it has only paid for the next
    iteration.
    """
    guard, _, _, _, outcomes = _run(
        4 * (REFETCH_DECLINE_CAP + 1) + UNRESOLVED_DEFER_CAP + 4,
        executed_sees_clamp=executed_sees_clamp,
    )
    assert "recompute" in outcomes, (
        "the run never reached the terminator, so this test would pass "
        f"vacuously; outcomes={outcomes}"
    )
    assert not guard.is_escalated(RID), (
        "the terminator was spent and the guard still reports the rid as "
        "escalated, so scheduler.py:9325 will re-write the mark on the next "
        "pass -- the circle #955 is about"
    )


# --------------------------------------------------------------------------
# 2. COUNTER-DIRECTION: the escape must survive the fix.
# --------------------------------------------------------------------------


def test_a_genuinely_moving_offer_can_still_escalate():
    """New evidence re-arms the escalation. Green in BOTH states, deliberately.

    A fix that pinned "escalated once, never again" would kill the #944 escape
    outright, and this arm is what makes the red above mean "the loop is dead"
    rather than "the feature is dead". The offer moving to a DIFFERENT positive
    length is a real measurement changing, which is the one thing that must
    still open the escape.
    """
    guard = PPAdmissionCongruenceGuard()
    for _ in range(UNRESOLVED_DEFER_CAP + 1):
        guard.note_offer(RID, TOLD)
    assert guard.is_escalated(RID)

    guard2 = PPAdmissionCongruenceGuard()
    for _ in range(UNRESOLVED_DEFER_CAP + 1):
        guard2.note_offer(RID, 4096)
    assert guard2.is_escalated(RID), (
        "a rid that froze at a different measured length must still be able to "
        "escalate; the bound is about a STUCK offer, not about one value"
    )


def test_an_escalation_ends_when_the_offer_starts_moving_again():
    """A rid that recovers on its own must stop being treated as escalated.

    THE SECOND HALF OF THE LAP-FREE CLEAR, and the one that matters BEFORE any
    terminator has been spent. The bound exists because an offer STOPPED
    moving; when it moves again the reason is gone. Without a lap-free clear
    the flag can only be dropped by an admitted round-trip, so a rid that
    recovered would keep being marked dead every pass (scheduler.py:9325) and
    be driven into a recompute it no longer needs -- paying the full
    double-prefill price for a request that had already healed.

    The moved offers here are LONGER, which is what a chunked prefill making
    progress actually looks like.
    """
    guard = PPAdmissionCongruenceGuard()
    holder = _Holder(guard=guard)
    req = _Req()
    for _ in range(UNRESOLVED_DEFER_CAP + 1):
        guard.note_offer(RID, TOLD)
    assert guard.is_escalated(RID), "setup: the rid must be escalated first"

    outcomes = []
    for chunk in range(9, 20):
        _, outcome = _one_pass(
            guard, holder, req, 1024 * chunk, executed_sees_clamp=True
        )
        outcomes.append(outcome)
    assert not guard.is_escalated(RID), (
        "the offer moved on every one of these passes, so the escalation's "
        "own reason is gone and it must not still be re-arming the mark"
    )
    assert "recompute" not in outcomes, (
        f"a recovered rid must never be driven into a recompute; {outcomes}"
    )
    assert guard.terminator_spent(RID) is None


def test_admitted_lap_still_resets_every_container():
    """The success path is untouched -- and it is still the full reset.

    ``record_return_trip``'s admitted branch is the designed lifecycle end for
    a rid that was actually served. #955 adds a lap-free end for the case where
    that lap cannot happen; it must not remove or weaken this one.
    """
    guard = PPAdmissionCongruenceGuard()
    for _ in range(UNRESOLVED_DEFER_CAP + 1):
        guard.note_offer(RID, TOLD)
    assert guard.is_escalated(RID)
    assert guard.offer_streak(RID) > UNRESOLVED_DEFER_CAP

    guard.record_return_trip(
        PPAdmissionDecision(
            mb_id=0,
            entries=(PPAdmissionEntry(rid=RID, prefix_len=0, extend_len=7, admitted=True),),
        )
    )
    assert not guard.is_escalated(RID)
    assert guard.offer_streak(RID) == 0
    assert guard.unresolved_rounds(RID) == 0
    assert guard.learned_floor(RID) is None


def test_a_served_rid_may_escalate_again_later():
    """One recompute per rid is per LIFE-EPISODE, not per process.

    After a genuine round-trip the rid is a different case entirely, and if it
    becomes unresolvable a second time the escape must be available a second
    time. pp_admission_congruence.py:604-605 states this requirement in the
    module itself; this pins it against the #955 change.
    """
    guard = PPAdmissionCongruenceGuard()
    holder = _Holder(guard=guard)
    req = _Req()
    for _ in range(4 * (REFETCH_DECLINE_CAP + 1) + UNRESOLVED_DEFER_CAP + 4):
        _one_pass(guard, holder, req, TOLD, executed_sees_clamp=False)

    guard.record_return_trip(
        PPAdmissionDecision(
            mb_id=0,
            entries=(PPAdmissionEntry(rid=RID, prefix_len=0, extend_len=7, admitted=True),),
        )
    )
    req2 = _Req()
    outcomes = []
    for _ in range(4 * (REFETCH_DECLINE_CAP + 1) + UNRESOLVED_DEFER_CAP + 4):
        _, outcome = _one_pass(guard, holder, req2, TOLD, executed_sees_clamp=False)
        outcomes.append(outcome)
    assert "recompute" in outcomes, (
        "after a completed round-trip the rid's escape must be available "
        f"again; outcomes={outcomes}"
    )


def test_a_progressing_rid_is_never_marked_and_never_acted_on():
    """The default path, which must stay byte-identical.

    A HEALTHY REQUEST IS ONE WHOSE OFFER MOVES, and that is the whole content
    of the bound: it fires on a STUCK offer, never on a working one. A chunked
    prefill advancing a chunk per pass re-offers a longer prefix every time, so
    ``_offer_streak`` resets at pp_admission_congruence.py:412 on every pass and
    the cap is never approached. Nothing escalates, nothing is marked, and the
    actuator walks away with "none".

    Written this way after the first draft of this arm pinned a rid whose offer
    was FROZEN and merely ran for few enough passes -- which is the defect case
    with a short clock, not the healthy case. Two ``note_offer`` calls happen
    per pass (the clamping caller at scheduler.py:9074 and the executed branch
    at pp_admission_congruence.py:783), so a frozen offer crosses the cap of 3
    inside two passes; the specimen's own "for 4 consecutive passes" line is
    that same arithmetic on metal.
    """
    guard = PPAdmissionCongruenceGuard()
    holder = _Holder(guard=guard)
    req = _Req()
    for chunk in range(1, 12):
        effective, outcome = _one_pass(
            guard, holder, req, 1024 * chunk, executed_sees_clamp=True
        )
        assert effective == 1024 * chunk
        assert outcome == "none"
    assert not guard.is_escalated(RID)
    assert getattr(req, _PREMISE_DEAD_STAMP, None) is None


def test_an_issued_refetch_still_spends_the_mark_and_skips_the_terminator():
    """The cheap escape still wins when the store can answer.

    #949's correction -- ISSUED clears the mark, a decline keeps it -- is the
    reason the terminator is a last resort at all. If #955 changed that, the
    fix would be paying for the loop with the feature.
    """
    guard = PPAdmissionCongruenceGuard()
    holder = _Holder(guard=guard, verdict="issued")
    req = _Req()
    outcomes = []
    for _ in range(UNRESOLVED_DEFER_CAP + 6):
        _, outcome = _one_pass(guard, holder, req, TOLD, executed_sees_clamp=False)
        outcomes.append(outcome)
    assert "recompute" not in outcomes, (
        f"an issuable re-fetch must never reach the terminator; {outcomes}"
    )
    assert holder.issued >= 1
    assert len(req.prefix_indices) == TOLD, (
        "an issued re-fetch keeps the premise, so nothing may be truncated"
    )


def test_told_zero_never_escalates():
    """A rid offering nothing is progressing by definition.

    pp_admission_congruence.py:414-416 states this and the whole terminator
    story rests on it: ``told=0`` is the offer a receiving rank admits
    unconditionally, so it is the forward exit, not a stuck state.
    """
    guard = PPAdmissionCongruenceGuard()
    for _ in range(UNRESOLVED_DEFER_CAP + 8):
        assert guard.note_offer(RID, 0) is False
    assert not guard.is_escalated(RID)


# --------------------------------------------------------------------------
# 3. THE WIRING, asserted over code rather than over prose.
# --------------------------------------------------------------------------


def _source_without_comments(path: Path) -> str:
    """The file's code with every comment and docstring removed.

    An assertion that reads the raw file passes or fails on the surrounding
    English, and this chain has already spent a boot window on a comment that
    described a guard the code did not have. Stripping first means the
    assertion can only be satisfied by the executable text.
    """
    src = path.read_text()
    out = []
    prev_type = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev_type in (
            tokenize.INDENT,
            tokenize.NEWLINE,
            tokenize.DEDENT,
            tokenize.NL,
        ):
            continue  # a bare string statement: a docstring
        out.append(tok.string)
        if tok.type not in (tokenize.NL, tokenize.NEWLINE):
            prev_type = tok.type
    return " ".join(out)


def _managers_dir() -> Path:
    spec = importlib.util.find_spec("sglang.srt.managers.scheduler")
    assert spec is not None and spec.origin is not None
    return Path(spec.origin).parent


def test_the_remark_site_still_reads_the_guard_every_pass():
    """The defect path this file drives is the one production still runs.

    If scheduler.py stops re-marking from ``is_escalated``, the loop above is
    no longer the production shape and these tests would be pinning history.
    Asserted on comment-stripped source so a rewritten comment cannot make it
    pass or fail.
    """
    code = _source_without_comments(_managers_dir() / "scheduler.py")
    assert "is_escalated" in code, (
        "scheduler.py no longer consults the guard's escalation set; the #955 "
        "re-mark path has moved and this test file must be re-aimed"
    )
    assert "_pp_mark_premise_dead" in code


def test_the_terminator_still_spends_its_own_marks():
    """#949's fix must remain intact under the #955 change.

    The terminator deleting its own two attributes is an operation in target
    state; #955 adds the guard-side end of the same lifecycle and must not
    replace this half with it.
    """
    code = _source_without_comments(_managers_dir() / "scheduler_pp_mixin.py")
    assert "_PREMISE_DEAD_STAMP" in code
    assert "_REFETCH_DECLINE_STREAK" in code
    assert "delattr" in code
