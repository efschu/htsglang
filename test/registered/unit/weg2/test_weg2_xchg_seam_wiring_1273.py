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
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    assert wx.exchange_armed() is True
    return "exchange"


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


def test_the_cpu_backup_wins_over_the_exchange_arm(monkeypatch, exchange):
    """Two carriers armed at once is a CONFIGURATION to decide, not a race.

    If the TMS backup carried the bytes, the exchange must not also write them:
    two writers for one payload is the ein-job-ein-mover defect, and the
    exchange would be overwriting bytes that are already correct.  The backup
    is the earlier and cheaper carrier (2.08 s / 27 GiB), so it wins, and the
    decision says so once instead of two call sites each guessing.
    """
    m = _manager(_FakeServerArgs(weights_backup=True), monkeypatch)
    assert m._weg2_wake_weight_carrier() == CARRIER_TMS_BACKUP


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

    And the transfer refuses by name today, because the plan provider has no
    registrant (`weight_exchange.register_plan_provider`, TODO(S6)) and the
    on-card diagonal lane is still being fixed.  The refusal must name the
    missing producer and print the priced size -- an operator reading it needs
    to know the term was fine and the PLAN was not.
    """
    from sglang.srt.weg2 import xchg_bounce as xb

    terms = xb.bounce_terms(
        bytes_per_direction=24_000_000, n_layers=8,
        widest_layer_bytes=4_000_000, pairs=3, depth=2,
        slot_bytes=128 * xb.MIB,
    )
    monkeypatch.setenv(xb.ENV_BOUNCE_TERMS, xb.publish_terms(terms))
    m = _manager(_FakeServerArgs(), monkeypatch)
    with pytest.raises(wu.Weg2WakeRefused) as e:
        m._weg2_xchg_inject_weights()
    msg = str(e.value)
    assert "W4" in msg
    assert "NO_PLAN_REASON" in msg or "plan provider" in msg
    assert str(terms.total_bytes) in msg
