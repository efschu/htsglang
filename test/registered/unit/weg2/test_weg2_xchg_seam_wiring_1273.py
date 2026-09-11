# SPDX-License-Identifier: Apache-2.0
"""#1273 S6-BOUNCE step 5 -- WHO CARRIES THE WEIGHT BYTES ON THIS WAKE.

Design of record: WEG2_REUSE_SPEC_0908.md section 10.8 step 5, PLAN_S6_BOUNCE_0911
step B2 ("``--weg2-weight-source exchange`` wird die Autoritaet fuer diese
Bytes").  User law: memory/gewichtsaustausch-ziel-kein-dauer-hostram.md.

THE GAP THIS CLOSES, measured and named.  Under ``exchange`` the weights region
is opened ``enable_cpu_backup=False``, so ``resume(GPU_MEMORY_TYPE_WEIGHTS)`` is
a pure VMM recommit that moves NO BYTES -- the product says so itself at
``weight_updater.py`` (`"with the cpu backup OFF ... the entire 12-17 s
host->device refill is this call"`).  That call is
``update_weights_from_disk``, measured 12.073/14.143/16.749 s per wake on this
rig.  So today the ``exchange`` arm's bytes come FROM DISK, which is the
handover-over-disk cost (#1317/#1323/#1325) the exchange exists to remove.

THE SEAM IS A DECISION BEFORE IT IS A TRANSFER, and that is why this file tests
a decision.  Three carriers are possible on a wake and they are mutually
exclusive:

    tms-backup  --enable-weights-cpu-backup: the TMS restore already carried
                the bytes (measured 2.08 s / 27 GiB, campaign (a)); refilling
                would be a second writer.
    exchange    --weg2-weight-source exchange: the peer group's live VRAM is
                the source, through the bounded host bounce.  THE BYTES MUST
                NOT COME FROM DISK HERE.
    disk        neither: the upstream update_weights_from_disk path, unchanged.

    stock       --enable-memory-saver absent: pause() was a no-op, nothing was
                released, there is nothing to refill at all.

A FOURTH ANSWER IS NOT ALLOWED TO BE SILENCE.  The reason this is one named
function rather than an ``if`` inside the refill is the #1329 lesson in this
same file: a decision that lives inside a long method has no executing test,
and three boots were spent on exactly that shape.  It is also the
Lebenszyklus rule -- a new state field needs its writers, readers and the
event between them enumerated before the boot, not after.

RED ON 312ab2b169 by construction: ``_weg2_wake_weight_carrier`` does not
exist, so every test below fails at its first statement.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu  # noqa: E402
from sglang.srt.weg2 import weight_exchange as wx  # noqa: E402

CARRIER_STOCK = "stock"
CARRIER_TMS_BACKUP = "tms-backup"
CARRIER_EXCHANGE = "exchange"
CARRIER_DISK = "disk"


class _FakeServerArgs:
    """Only the four flags the decision reads, and nothing else.

    Written as an explicit class rather than a ``SimpleNamespace`` so a flag
    the decision starts reading later fails LOUDLY here instead of silently
    defaulting through ``getattr``.
    """

    def __init__(self, *, memory_saver: bool = True, weights_backup: bool = False,
                 draft_backup: bool = False,
                 speculative_draft_model_path=None,
                 model_path: str = "/models/main") -> None:
        self.enable_memory_saver = memory_saver
        self.enable_weights_cpu_backup = weights_backup
        self.enable_draft_weights_cpu_backup = draft_backup
        self.speculative_draft_model_path = speculative_draft_model_path
        self.model_path = model_path


class _FakeRunner:
    def __init__(self) -> None:
        self.model = None
        self.model_config = None


class _FakeWorker:
    def __init__(self) -> None:
        self.model_runner = _FakeRunner()


def _manager(server_args, monkeypatch, *, draft_worker=None):
    """The product object, with ONE seam faked: where server_args come from.

    ``_weg2_server_args`` is patched on the CLASS because the manager is a
    ``slots=True`` dataclass -- the #1329 defect was a write to an undeclared
    attribute on exactly this class, so a test may not invent instance
    attributes either.
    """
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_server_args",
                        lambda self: server_args, raising=True)
    return wu.SchedulerWeightUpdaterManager(
        tp_worker=_FakeWorker(), draft_worker=draft_worker, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )


@pytest.fixture()
def ring(monkeypatch):
    monkeypatch.delenv(wx.WEIGHT_SOURCE_ENV, raising=False)
    assert wx.exchange_armed() is False
    return "ring"


@pytest.fixture()
def exchange(monkeypatch):
    """The exchange arm in AUTHORITATIVE inject mode.

    Both are needed for the exchange to be the carrier (step 6c): the arm says
    the exchange is the weight SOURCE, the mode says its injection has
    REPLACED the refill.  Under the default `shadow` the refill stays the
    authority -- `test_under_shadow_mode_the_refill_stays_the_authority` is
    that half.
    """
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_AUTHORITATIVE)
    assert wx.exchange_armed() is True
    assert wx.inject_authoritative() is True
    return "exchange"


@pytest.fixture()
def exchange_shadow(monkeypatch):
    """The exchange arm in the DEFAULT inject mode: grading, not owning."""
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.delenv(wx.INJECT_ENV, raising=False)
    assert wx.exchange_armed() is True
    assert wx.inject_authoritative() is False
    return "shadow"


# ===========================================================================
# THE FOUR CARRIERS, each with the flag combination that selects it.
# ===========================================================================


class TheCarrierDecision:
    """Namespace only; the collected tests are the functions below."""


def test_without_the_memory_saver_nothing_was_released(monkeypatch, ring):
    """Stock boot: ``pause()`` was ``pass``, so there is nothing to refill.

    This must win over every other consideration, including the exchange arm:
    a stock resume has to be byte-for-byte the upstream path.
    """
    m = _manager(_FakeServerArgs(memory_saver=False), monkeypatch)
    assert m._weg2_wake_weight_carrier() == CARRIER_STOCK


def test_the_cpu_backup_carried_the_bytes(monkeypatch, ring):
    """``--enable-weights-cpu-backup``: the TMS restore already did it."""
    m = _manager(_FakeServerArgs(weights_backup=True), monkeypatch)
    assert m._weg2_wake_weight_carrier() == CARRIER_TMS_BACKUP


def test_the_default_backup_off_wake_still_refills_from_disk(monkeypatch, ring):
    """The unchanged upstream path, and it must STAY reachable.

    Backward compatibility is the point of this case: with no exchange armed,
    the answer may not change, or every ordinary RL boot changes behaviour.
    """
    m = _manager(_FakeServerArgs(), monkeypatch)
    assert m._weg2_wake_weight_carrier() == CARRIER_DISK


def test_under_exchange_the_bytes_do_not_come_from_disk(monkeypatch, exchange):
    """THE WHOLE POINT OF STEP 5, as one assertion.

    ``exchange`` + backup off + saver on must answer ``exchange``.  Answering
    ``disk`` here is what the tree does today, and it costs the measured
    12.073/14.143/16.749 s per wake AND puts the dormant image back on the
    disk path the exchange exists to remove.
    """
    m = _manager(_FakeServerArgs(), monkeypatch)
    assert m._weg2_wake_weight_carrier() == CARRIER_EXCHANGE


def test_under_shadow_mode_the_refill_stays_the_authority(monkeypatch,
                                                          exchange_shadow):
    """STEP 6c's DEFAULT, and the direction that must not drift.

    The exchange is armed, and it is NOT the carrier: the disk refill still
    owns the bytes and the injection grades itself against them.  A boot that
    armed the exchange and had never been graded must not become the authority
    BY OMISSION, which is exactly what S6I exists to close -- so the default
    answer here is `disk`, and `authoritative` has to be asked for.
    """
    m = _manager(_FakeServerArgs(), monkeypatch)
    assert m._weg2_wake_weight_carrier() == CARRIER_DISK


def test_the_authoritative_exchange_wins_over_the_cpu_backup(monkeypatch,
                                                             exchange):
    """SUPERSEDED PRECEDENCE, INVERTED ON PURPOSE (#1342 S1).

    THIS TEST USED TO ASSERT THE OPPOSITE and was named
    `test_the_cpu_backup_wins_over_the_exchange_arm`.  It is renamed and
    inverted rather than deleted, so the reversal shows up in the gate's name
    diff as one GONE and one NEW with this docstring attached, instead of
    vanishing.

    Its old reasoning: two writers for one payload is the ein-job-ein-mover
    defect, the backup is earlier and cheaper (2.08 s / 27 GiB), so the backup
    wins.  That holds only while NEITHER writer is the declared authority.
    Under `--weg2-xchg-inject authoritative` the exchange IS the authority, so
    this is one authority plus a redundant restore of the same bytes --
    wasteful, not incorrect -- and B7 closes it by taking the host image to
    0.00 GiB.

    What the old precedence actually cost, measured on boot weg2xsn17: because
    the launcher passes `--enable-weights-cpu-backup` unconditionally,
    `main_carried` was True on every boot, so this branch returned before the
    exchange was ever asked and `CARRIER_EXCHANGE` was unreachable at every
    argv.  An arm that cannot be selected cannot be graded, which is why four
    boots of S6I instruments produced nothing.
    """
    m = _manager(_FakeServerArgs(weights_backup=True), monkeypatch)
    assert m._weg2_wake_weight_carrier() == CARRIER_EXCHANGE


def test_the_stock_answer_wins_over_the_exchange_arm(monkeypatch, exchange):
    """No memory saver means no pause, so there is nothing for anyone to carry.

    An exchange that injected here would write into pages the model is already
    using, on a boot that never released them.
    """
    m = _manager(_FakeServerArgs(memory_saver=False), monkeypatch)
    assert m._weg2_wake_weight_carrier() == CARRIER_STOCK


# ===========================================================================
# THE REFUSALS THAT ALREADY EXIST MUST SURVIVE THE NEW BRANCH.
# ===========================================================================


class TheExistingRefusalsStillFire:
    """Namespace only; the collected tests are the functions below."""


def test_no_server_args_is_still_undecidable(monkeypatch, exchange):
    """W4: VRAM has already been mutated, so refuse rather than guess.

    Pinned under EXCHANGE deliberately: a new branch that answered its own
    question before this check would turn an undecidable wake into a silent
    injection.
    """
    m = _manager(None, monkeypatch)
    with pytest.raises(wu.Weg2WakeRefused) as e:
        m._weg2_wake_weight_carrier()
    assert "W4" in str(e.value)


def test_shards_that_disagree_about_the_backup_still_stop(monkeypatch, ring):
    """W4: the draft shard cpu-backed while the main shard is not.

    One ``update_weights_from_disk`` serves both shards with one model_path,
    so there is no arrangement that refills the main shard without pushing the
    main checkpoint through the draft runner.  Shards never disagree: STOP.
    """
    m = _manager(_FakeServerArgs(draft_backup=True), monkeypatch,
                 draft_worker=object())
    with pytest.raises(wu.Weg2WakeRefused) as e:
        m._weg2_wake_weight_carrier()
    assert "W4" in str(e.value)


def test_a_separate_draft_checkpoint_still_stops(monkeypatch, ring):
    """W4: the draft worker loads from a different path than the main shard."""
    m = _manager(_FakeServerArgs(
        speculative_draft_model_path="/models/draft"), monkeypatch,
        draft_worker=object())
    with pytest.raises(wu.Weg2WakeRefused) as e:
        m._weg2_wake_weight_carrier()
    assert "W4" in str(e.value)


# ===========================================================================
# THE CALL SITE USES THE DECISION -- not a second reading of the flags.
# ===========================================================================


class TheRefillObeysTheDecision:
    """Namespace only; the collected tests are the functions below."""


def test_the_refill_asks_the_decision_and_does_not_re_read_the_flags(
        monkeypatch, ring):
    """ONE producer of the verdict, pinned by substitution.

    ``_weg2_wake_reload_weights`` must route on
    ``_weg2_wake_weight_carrier()``.  The test forces the decision to say
    ``tms-backup`` on a configuration whose raw flags say ``disk``: a refill
    that re-read the flags itself would reload, and one that obeys the
    decision returns.  That is the difference between one authority and two
    readings, and it cannot be asserted by reading the code.
    """
    m = _manager(_FakeServerArgs(), monkeypatch)
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager,
                        "_weg2_wake_weight_carrier",
                        lambda self: CARRIER_TMS_BACKUP, raising=True)
    called = []
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager,
                        "_weg2_xchg_inject_weights",
                        lambda self, **kw: called.append("inject"),
                        raising=True)
    m._weg2_wake_reload_weights()
    assert called == []


def test_the_exchange_carrier_reaches_the_injector(monkeypatch, exchange):
    """And the exchange branch must actually CALL it.

    The #1329 shape again: a branch nothing reaches is indistinguishable from
    a branch that does not exist, and it was worth three boots last time.
    """
    m = _manager(_FakeServerArgs(), monkeypatch)
    called = []
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager,
                        "_weg2_xchg_inject_weights",
                        lambda self, **kw: called.append("inject"),
                        raising=True)
    m._weg2_wake_reload_weights()
    assert called == ["inject"]


# ===========================================================================
# THE PUBLISHED TERM.  Added because two mutants survived without it:
# "an unpublished term is silently sized to nothing" and "the published term
# is parsed leniently" both passed all 35 tests, i.e. the two refusals in
# `_weg2_xchg_inject_weights` / `read_published_terms` had no executing test.
# A refusal nothing exercises is indistinguishable from a refusal that is not
# there -- the #1329 shape, one level down.
# ===========================================================================


class ThePublishedTerm:
    """Namespace only; the collected tests are the functions below."""


def test_an_unpublished_term_refuses_instead_of_injecting(monkeypatch, exchange):
    """W4, and it may not become a zero-sized buffer.

    The resume has already remapped the weight pages and their content is
    undefined, so this wake has exactly two honest outcomes: inject against
    the size the launcher PRICED, or refuse.  Sizing a pinned host buffer
    locally would put bytes above the reap mark that no ledger carries, which
    `host-schwelle-nie-uebertreten` forbids; returning quietly would serve
    whatever the remap left behind.
    """
    from sglang.srt.weg2 import xchg_bounce as xb

    monkeypatch.delenv(xb.ENV_BOUNCE_TERMS, raising=False)
    m = _manager(_FakeServerArgs(), monkeypatch)
    with pytest.raises(wu.Weg2WakeRefused) as e:
        m._weg2_xchg_inject_weights()
    msg = str(e.value)
    assert "W4" in msg
    assert xb.ENV_BOUNCE_TERMS in msg


def test_a_published_term_round_trips_through_the_one_sizing_function(monkeypatch):
    """The launcher's object and the rank's object must be THE SAME term.

    Not "close": the rank rebuilds from the published INPUTS through
    `bounce_terms`, the single sizing function, so every derived field
    (buffer, staging, mean, coverage) is recomputed rather than trusted.  A
    published TOTAL would have been the thing that reads right while its
    factors drift -- measured twice in this slice already.
    """
    from sglang.srt.weg2 import xchg_bounce as xb

    original = xb.bounce_terms(
        bytes_per_direction=24_000_000, n_layers=8,
        widest_layer_bytes=4_000_000, pairs=3, depth=2,
        slot_bytes=128 * xb.MIB,
    )
    monkeypatch.setenv(xb.ENV_BOUNCE_TERMS, xb.publish_terms(original))
    got = xb.read_published_terms()
    assert got == original
    assert got.total_bytes == original.total_bytes
    assert got.staging_per_card == original.staging_per_card


def test_no_publication_reads_as_absent_and_never_as_zero(monkeypatch):
    """``None`` is a state, not a size."""
    from sglang.srt.weg2 import xchg_bounce as xb

    monkeypatch.delenv(xb.ENV_BOUNCE_TERMS, raising=False)
    assert xb.read_published_terms() is None
    assert xb.read_published_terms("   ") is None


def test_a_partially_published_term_raises_rather_than_defaulting(monkeypatch):
    """A missing field would be sized against `bounce_terms`' defaults.

    That is how a pinned constant re-enters through the back door: the term
    would look priced, and one of its factors would be this module's default
    rather than the launcher's measurement.
    """
    from sglang.srt.weg2 import xchg_bounce as xb

    with pytest.raises(ValueError) as e:
        xb.read_published_terms("n_layers=8,pairs=3")
    assert "missing" in str(e.value)


def test_an_unknown_published_field_raises(monkeypatch):
    """An env var that exists and cannot be parsed is a LAUNCHER defect.

    Swallowing it would inject against a size nobody priced, so it raises
    rather than degrading to ``None`` -- ``None`` means "not published", and
    conflating the two would turn a defect into a refusal with the wrong name.
    """
    from sglang.srt.weg2 import xchg_bounce as xb

    with pytest.raises(ValueError) as e:
        xb.read_published_terms("n_layers=8,slot_bytes=1,widest=2")
    assert "unknown field" in str(e.value)


def test_a_published_term_reaches_the_transfer_and_refuses_by_name(monkeypatch,
                                                                   exchange):
    """With the term present, the seam gets as far as the TRANSFER.

    UPDATED #1342 S2, and the update is the point rather than a repair: this
    test used to assert the refusal contained `NO_PLAN_REASON` or the phrase
    "plan provider", because the transfer refused UNCONDITIONALLY with a
    message naming an unregistered provider.  BOTH of that message's premises
    were stale -- `arm_coverage_at_load` (called from `model_runner.py:2564`)
    registers the provider via `install_default_plan_provider`, and boot
    weg2xsn17 emitted 18/21 real `WEG2-XCHG-PLAN card=` lines -- so the
    delegation was wired and the blanket refusal removed.

    WHAT STILL MUST HOLD, and is what this test now pins: with a valid term
    but NO plan reachable on this rank (no model on the fake runner, so the
    derivation yields nothing), the transfer still REFUSES BY NAME and still
    prints the priced size.  An operator reading it needs to know the term was
    fine and the plan was not, and the resume has already remapped the weight
    pages -- so a silent return here would serve whatever the remap left
    behind.  The refusal now quotes the DERIVATION'S OWN reason instead of a
    hard-coded sentence about a provider, which is strictly more informative
    and cannot go stale the same way.
    """
    from sglang.srt.weg2 import xchg_bounce as xb

    terms = xb.bounce_terms(
        bytes_per_direction=24_000_000, n_layers=8,
        widest_layer_bytes=4_000_000, pairs=3, depth=2,
        slot_bytes=128 * xb.MIB,
    )
    from sglang.srt.weg2 import weight_exchange_region as xr

    monkeypatch.setenv(xb.ENV_BOUNCE_TERMS, xb.publish_terms(terms))
    # #1342: the inject now runs an IDENTITY SWEEP before it asks for a plan,
    # so every earlier input has to be satisfied or the refusal this test is
    # about is not the one that fires.  Supplying them is not test-fitting --
    # it is what makes the assertion below actually about the PLAN, and on the
    # way it drives the sweep's success path, which nothing else does.
    monkeypatch.setenv(xr.ENV_REGION_BOOT, "1789122247")
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_group_name",
                        lambda self: "D", raising=True)
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_rank",
                        lambda self: 0, raising=True)
    monkeypatch.setattr(wu.SchedulerWeightUpdaterManager, "_weg2_device_index",
                        lambda self: 0, raising=True)
    m = _manager(_FakeServerArgs(), monkeypatch)
    with pytest.raises(wu.Weg2WakeRefused) as e:
        m._weg2_xchg_inject_weights()
    msg = str(e.value)
    assert "W4" in msg
    assert "PLAN" in msg, "the refusal must still name the plan as the gap"
    assert str(terms.total_bytes) in msg, "the priced size must still print"


# ===========================================================================
# #1342 S1: THE CARRIER PRECEDENCE, REORDERED SO THE EXCHANGE IS ASKED FIRST.
#
# WHY THIS BLOCK EXISTS, measured on boot weg2xsn17 (571a3da963, record
# BOOT_weg2xsn17_0911.md): the router returned `tms-backup` at what was then
# weight_updater.py:839-840 BEFORE it ever evaluated the exchange test at
# :852-853, and `main_carried` is `server_args.enable_weights_cpu_backup`,
# which the weg2 launcher passes UNCONDITIONALLY in `common_flags`.  So
# CARRIER_EXCHANGE was unreachable on EVERY arm at EVERY argv, and four boots
# of S6I instruments graded nothing.  Confirmed positively on that boot rather
# than inferred: `Weg2WakeRefused` bare 0 / genuine 0, which is what a
# never-entered raiser looks like and what an entered one could not be.
#
# THE ORDER SOUGHT (operator ruling): exchange (armed AND authoritative) ->
# main_carried -> disk.  The point is not that `shadow` changes answer -- it
# must NOT -- but that it declines for the RIGHT REASON (not authoritative)
# instead of because a backup flag returned earlier.
#
# HONESTY ABOUT WHICH OF THESE IS RED: only
# `test_the_exchange_is_asked_before_the_backup` is red on 571a3da963.  The
# two `shadow` pins below PASS on both sides of the fix by construction --
# they are OVERSHOOT GUARDS, not red tests, and they are labelled so rather
# than presented as proof of the fix.  Their job is to fail if the reorder
# ever slides `shadow` into `disk` or `exchange`.
# ===========================================================================


def test_the_exchange_is_asked_before_the_backup(monkeypatch, exchange):
    """THE BLOCKER, and the one genuinely RED case in this block.

    `exchange` + `authoritative` + `--enable-weights-cpu-backup` must answer
    `exchange`.  On 571a3da963 it answers `tms-backup`, because the backup
    branch returned before the exchange branch was reached.

    THIS INVERTS `test_the_cpu_backup_wins_over_the_exchange_arm`, which
    encoded the OLD precedence deliberately, and the inversion is argued
    rather than assumed: under `authoritative` the exchange IS the authority
    by definition, so the ein-job-ein-mover objection ("two writers for one
    payload") becomes "one authority plus a redundant restore of the same
    bytes" -- wasteful, not incorrect.  It is B7 that closes it properly by
    taking the host image to 0.00 GiB; until then the redundancy is the price
    of the arm being reachable at all, and an unreachable arm cannot be graded.
    """
    m = _manager(_FakeServerArgs(weights_backup=True), monkeypatch)
    assert m._weg2_wake_weight_carrier() == CARRIER_EXCHANGE


def test_shadow_with_a_backup_is_tms_backup_not_disk(monkeypatch,
                                                     exchange_shadow):
    """OVERSHOOT GUARD (passes before AND after the fix -- stated, not hidden).

    `shadow` + backup armed must be `tms-backup`.  It is NOT `disk`: the
    backup really did carry the bytes (`model_runner.py:2440-2442` computes
    `enable_cpu_backup` from server_args with NO arm predicate, so the weights
    region IS cpu-backed on the exchange arm -- the claim elsewhere in
    weight_updater.py that `exchange` opens it `enable_cpu_backup=False` is
    stale prose, corrected in this slice).

    This is the half that makes the reorder safe: moving the exchange test
    first must not strand `shadow` in `disk`, which would put the measured
    12.073/14.143/16.749 s checkpoint reload back on every wake.
    """
    m = _manager(_FakeServerArgs(weights_backup=True), monkeypatch)
    assert m._weg2_wake_weight_carrier() == CARRIER_TMS_BACKUP


def test_shadow_declines_for_the_right_reason_not_by_flag_order(monkeypatch,
                                                                exchange_shadow):
    """OVERSHOOT GUARD, and it pins the REASON rather than only the answer.

    With NO backup armed, `shadow` must still be `disk`.  Together with the
    test above this separates the two reasons a `shadow` wake is not the
    exchange: with a backup it is `tms-backup`, without one it is `disk`, and
    in NEITHER case is it `exchange`.  On the old order the first of those two
    was decided by the flag, not by the arm.
    """
    m = _manager(_FakeServerArgs(), monkeypatch)
    assert m._weg2_wake_weight_carrier() == CARRIER_DISK


def test_the_ring_arm_is_untouched_by_the_reorder(monkeypatch, ring):
    """THE DANGER DIRECTION of the reorder: the default boot must not move.

    `ring` + backup must stay `tms-backup`, and `ring` without it `disk`.  The
    exchange test now runs FIRST, so a predicate that answered True on `ring`
    would silently make every ordinary RL boot inject from a peer that is not
    there.  Asserted on both flag settings in one test so the pair cannot
    drift apart.
    """
    m = _manager(_FakeServerArgs(weights_backup=True), monkeypatch)
    assert m._weg2_wake_weight_carrier() == CARRIER_TMS_BACKUP
    m2 = _manager(_FakeServerArgs(), monkeypatch)
    assert m2._weg2_wake_weight_carrier() == CARRIER_DISK


def test_stock_still_outranks_the_exchange_after_the_reorder(monkeypatch,
                                                            exchange):
    """`stock` must remain the FIRST answer, ahead of the moved exchange test.

    No memory saver means `pause()` released nothing, so an exchange that
    injected here would write into pages the model is still using.  The
    reorder moves the exchange ABOVE the backup, and this pins that it did not
    also move above `stock`.
    """
    m = _manager(_FakeServerArgs(memory_saver=False, weights_backup=True),
                 monkeypatch)
    assert m._weg2_wake_weight_carrier() == CARRIER_STOCK
