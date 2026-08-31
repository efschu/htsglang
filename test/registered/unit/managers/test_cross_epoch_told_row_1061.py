"""#1061: a told row decided under another cutover epoch is not applied — UNIFORMLY.

RED-FIRST AT THE BOOT-30 SPECIMEN. The numbers below are the ones the log
carries (`told prefix 12288 exceeds this rank's pinned span 0`), so the first
test is a direct replay of the death: without the epoch gate
`uniform_pass_geometry` raises and the group dies; with it the row is simply not
adopted and the rank runs its own geometry.

MANDATORY DANGER-DIRECTION MUTANT (`test_mutant_local_span_trigger_diverges`):
the plausible wrong fix is "void when the told value exceeds MY pinned span".
That reads a RANK-LOCAL quantity, and the guard's own message forbids it
("compensating locally here would move the batch and reappear as #631 on a
peer"). The test injects the boot-30 state with ranks that differ ONLY in their
local span and asserts that the local-span rule produces a SPLIT verdict while
the epoch rule produces an identical one on every rank. If a future edit swaps
the epoch trigger for a span trigger, this goes red.
"""

import pytest

from sglang.srt.managers.pp_uniform_width import (
    UniformWidthPromiseBroken,
    epoch_admits_row,
    uniform_pass_geometry,
)

TOLD_PREFIX = 12288  # boot 30, verbatim
TOLD_EXTEND = 4096


def test_boot30_replay_without_the_gate_still_raises():
    """The death is reproduced, so the gate is proven to be what prevents it."""
    with pytest.raises(UniformWidthPromiseBroken):
        uniform_pass_geometry(TOLD_PREFIX, TOLD_EXTEND, 0, pinned_prefix=0)


def test_boot30_row_is_refused_by_the_epoch_gate():
    """Row decided at epoch 1, applied at epoch 2 -> not admitted."""
    assert epoch_admits_row(1, 2) is False


def test_all_ranks_agree_regardless_of_their_local_state():
    """THE PROPERTY THE FIX EXISTS FOR: one verdict for the whole group.

    Three ranks with wildly different local spans -- the boot-30 shape, where
    the cutover left one rank at 0 and others elsewhere -- must reach the SAME
    adopt/not-adopt answer, because the verdict never reads the span.
    """
    row_epoch, now_epoch = 1, 2
    verdicts = set()
    for local_span in (0, 4096, 12288):
        admits = epoch_admits_row(row_epoch, now_epoch)
        if admits:
            geom = uniform_pass_geometry(
                TOLD_PREFIX, TOLD_EXTEND, local_span, pinned_prefix=local_span
            )
            verdicts.add(bool(geom.adopted))
        else:
            # The no-adopt path: the rank runs its own geometry and proceeds.
            geom = uniform_pass_geometry(
                None, TOLD_EXTEND, local_span, pinned_prefix=local_span
            )
            verdicts.add(bool(geom.adopted))
    assert verdicts == {False}, verdicts


def test_same_epoch_still_adopts():
    """The gate must not disable #1059 in its own generation."""
    assert epoch_admits_row(7, 7) is True
    geom = uniform_pass_geometry(4096, TOLD_EXTEND, 8192, pinned_prefix=8192)
    assert geom.adopted is True
    assert geom.prefix == 4096


def test_no_epoch_namespace_is_vacuous_not_refusing():
    """A boot without the phase flip has no cutovers, so the gate must not bite."""
    assert epoch_admits_row(None, None) is True


def test_half_absent_epoch_is_unverifiable_and_refused():
    """One side None: the row cannot be placed in this namespace -> absent fact."""
    assert epoch_admits_row(None, 3) is False
    assert epoch_admits_row(3, None) is False


def test_mutant_local_span_trigger_diverges():
    """DANGER DIRECTION: the local-span rule SPLITS the group; the epoch rule does not.

    This is the assertion that makes the difference between the two candidate
    fixes observable instead of a matter of taste.
    """
    row_epoch, now_epoch = 1, 2
    spans = (0, 4096, 12288)

    # The mutant rule: "void when told exceeds MY pinned span."
    mutant_verdicts = {TOLD_PREFIX <= span for span in spans}
    assert mutant_verdicts == {False, True}, (
        "the local-span rule must be shown to disagree across ranks; if this "
        "set is a singleton the fixture no longer exercises the divergence"
    )

    # The shipped rule reads only the two epochs.
    epoch_verdicts = {epoch_admits_row(row_epoch, now_epoch) for _ in spans}
    assert epoch_verdicts == {False}, epoch_verdicts


def test_epoch_source_is_read_through_not_cached():
    """The accessor must see a cutover that happens between two calls.

    Boot 30's apply ran INSIDE `_release_residents_for_cutover`; a value cached
    at the top of the pass would have been one generation stale exactly there.
    """
    from sglang.srt.managers import pp_uniform_width as puw

    box = {"e": 1}
    puw.set_epoch_source(lambda: box["e"])
    try:
        assert puw.current_epoch() == 1
        box["e"] = 2  # a cutover completes
        assert puw.current_epoch() == 2
    finally:
        puw.set_epoch_source(None)
    assert puw.current_epoch() is None


def test_wire_row_carries_the_epoch_round_trip():
    """The carrier half: a stamped epoch must survive encode -> decode."""
    from sglang.srt.managers.pp_admission_congruence import (
        PPAdmissionDecision,
        PPAdmissionEntry,
    )
    from sglang.srt.managers.scheduler_pp_mixin import (
        pp_admission_decision_from_wire,
        pp_admission_decision_to_wire,
    )

    dec = PPAdmissionDecision(
        mb_id=0,
        entries=(
            PPAdmissionEntry(
                rid="r1", prefix_len=TOLD_PREFIX, extend_len=TOLD_EXTEND,
                decided_epoch=5,
            ),
        ),
    )
    back = pp_admission_decision_from_wire(pp_admission_decision_to_wire(dec))
    assert back.entries[0].decided_epoch == 5
    assert back.entries[0].prefix_len == TOLD_PREFIX


def test_legacy_short_row_reads_epoch_as_none():
    """An older sender's row is unverifiable, never a silent match."""
    from sglang.srt.managers.scheduler_pp_mixin import _pp_admission_entry_from_row

    legacy = ("r1", 4096, 512, True, False, None, None, False)
    entry = _pp_admission_entry_from_row(legacy)
    assert entry.decided_epoch is None
    assert epoch_admits_row(entry.decided_epoch, 3) is False


def test_apply_site_actually_consults_the_gate():
    """WIRED, not merely present — the check the whole campaign exists to force.

    The pure-function tests above prove the RULE. They do not prove
    `apply_uniform_pass_geometry_1059` consults it, which is exactly the
    present-wired-never-populated shape that cost boot 29 a window. This drives
    the apply site itself with a live epoch source and asserts the boot-30 state
    does NOT raise and does NOT adopt — and that the same request adopts as soon
    as the epochs agree.
    """
    from sglang.srt.managers import pp_uniform_width as puw
    from sglang.srt.managers import schedule_batch as sb

    class _Req:
        rid = "a3fe52c0"
        prefix_indices = ()
        cache_protected_len = 0
        _1059_told_prefix = TOLD_PREFIX
        _1059_told_extend = TOLD_EXTEND
        _1059_told_epoch = 1

        truncate_prefix_to = sb.Req.truncate_prefix_to

    req = _Req()
    box = {"e": 2}  # a cutover completed between decide and apply
    puw.set_epoch_source(lambda: box["e"])
    sb._1061_STATE.clear()
    try:
        # Boot 30: this raised UniformWidthPromiseBroken and killed the group.
        adopted = sb.Req.apply_uniform_pass_geometry_1059(req)
        assert adopted is False
        assert sb._1061_STATE.get("refused_epoch") == 1, sb._1061_STATE
        assert sb._1061_STATE.get("adopted", 0) == 0, sb._1061_STATE

        # Same generation: the mechanism must still work.
        box["e"] = 1
        req2 = _Req()
        req2.prefix_indices = tuple(range(TOLD_PREFIX))
        req2.cache_protected_len = TOLD_PREFIX
        assert sb.Req.apply_uniform_pass_geometry_1059(req2) is True
        assert sb._1061_STATE.get("adopted") == 1, sb._1061_STATE
    finally:
        puw.set_epoch_source(None)
        sb._1061_STATE.clear()


def test_retract_clears_the_told_row_before_readmission():
    """#1061b: the RETRACT drops the told row — the epoch alone could not.

    RED-FIRST AT BOOT 31, which is the boot that refuted the epoch gate. There
    `refused_epoch=0` and `epoch_ok=1` on the raising rank: the gate ran, agreed,
    and the raise came after it, because `PhaseFlipRuntime._epoch` advances at
    cutover COMPLETION while `_release_residents_for_cutover` drops the tree and
    re-admits BEFORE completion — inside the SAME epoch. An epoch test cannot see
    a staleness window that lives inside one epoch.

    The retract is the destructive act itself, it is a GROUP event (boot 31:
    `#969AD RETRACT site=readmit_seam_residents` exactly once on each of PP0,
    PP1, PP2), and `_969ad_note_retract` is the chokepoint every retraction
    passes — called immediately before `_add_request_to_queue`, i.e. before the
    apply that raised.
    """
    from sglang.srt.managers import schedule_batch as sb
    from sglang.srt.managers.scheduler import Scheduler

    class _Req:
        rid = "a3fe52c0"
        prefix_indices = ()
        _1059_told_prefix = TOLD_PREFIX
        _1059_told_extend = TOLD_EXTEND
        _1059_told_epoch = 1

    class _Sched:
        class ps:
            pp_rank = 1

        forward_ct = 12
        _969ad_note_retract = Scheduler._969ad_note_retract

    req = _Req()
    sb._1061_STATE.clear()
    try:
        _Sched._969ad_note_retract(_Sched(), req, "readmit_seam_residents")
        assert req._1059_told_prefix is None
        assert req._1059_told_epoch is None
        assert sb._1061_STATE.get("told_cleared_at_retract") == 1, sb._1061_STATE
        assert sb._1061_STATE.get("retract_seen") == 1, sb._1061_STATE
        # And the apply is now a no-fact pass rather than a raise.
        assert sb.Req.apply_uniform_pass_geometry_1059(req) is False
        assert sb._1061_STATE.get("no_fact") == 1, sb._1061_STATE
    finally:
        sb._1061_STATE.clear()
