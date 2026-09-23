# SPDX-License-Identifier: Apache-2.0
"""#1391 (DESK10) -- Karte 0's collector never drains, and the wall it hides
behind.

THE FINDING (boot weg2xsn31, DEBUG_HOLD versuch 5 AND versuch 7,
byte-identical; record BOOT_weg2xsn31_0913_4.md, "VERSUCH 7 NACHTRAG 3"). P
PP0 (co-located with D TP0 on the 5090, boot tip d6c3d192f8) reaches
``resume_memory_occupation``'s per-tag loop for ``weights_0``/``weights_1``
with an EMPTY collect plan for its own card (``_weg2_xchg_inject_from_peer``
logs ``pieces=0`` and returns, silently) while D's three ranks -- diag,
``1-0``, ``2-0`` -- have ALREADY deposited real bands there: ``sem_getvalue``
read ``full=8/8/16 empty=0`` on all three, hand-probed with raw ``ctypes``
because the product itself had no accessor for the diagonal count (fixed here
as :meth:`weight_exchange_transport.SemSet.diagonal_getvalue`).

TWO GENUINELY SEPARATE CODE PATHS BLOCK 120 s EACH AND NAME THE WRONG THING:
D's ``wait_drained`` for ``weights_1`` never gets its ``drained`` token (W68
Weg2XchgPlanDisagree, raised as ``Weg2XchgBouncePhaseUnordered``), and P's
OWN, LATER credit wait for ``weights_3`` times out (W35
Weg2VramCreditRefused) -- Lock-Ordnungs-Inversion was checked and FALSIFIED at
the desk (two disjoint frame stacks, confirmed by DEBUG_HOLD dumps); the two
walls are independent symptoms of the SAME undrained lane.

DECISION (a) vs (b), settled by EXECUTING the real plan-derivation code
(``probe_plan_symmetry.py``, run at the desk against a P=PP3(22,21,21) /
D=TP3 geometry with 5 real ``weights_<k>`` tags): ``xchg_manifest.leg_plan_from_join``
and ``weight_exchange.build_plan``/``_emit`` build FULLY SYMMETRIC dst_rank==0
descriptors for BOTH the co-located diagonal (src==dst==0) and the two
cross-into-card-0 pairs, given consistent manifests -- (a) "the plan never
plans card 0" is FALSIFIED at the join/build_plan layer (weight_exchange.py /
xchg_manifest.py, both OUTSIDE this desk's file boundary). The wedge is (b):
the RUNTIME CALL comes back with an empty per-tag slice while the peer
independently deposits real bytes, and NOTHING inside this file's boundary
noticed. This file is the fix for that half: it cannot re-derive whether the
manifest asymmetry itself lives in xchg_manifest.py (outside the boundary,
flagged separately) -- it closes the part that is checkable and fixable HERE,
which is that an empty collect plan beside a real, undrained lane must never
be silent.

THE FIX, in two independent places (both inside weight_updater.py /
weg2_memory_saver.py / weight_exchange_transport.py, this desk's three
files):

  1. :meth:`SchedulerWeightUpdaterManager._weg2_xchg_undrained_lanes` -- reads
     every lane whose DESTINATION is this rank's own card (the diagonal PLUS
     every cross pair with ``dst == rank``) via the new
     :meth:`SemSet.diagonal_getvalue` and the existing
     :meth:`SemSet.getvalue`.
  2. Two call sites raise ``Weg2XchgLaneNeverDrainedRefused`` (W108 -- W100 at
     the time this was written, renumbered 2026-09-14 when TRAIN2's census
     against the merged tree found it colliding with host_ledger.py's
     pre-existing ``Weg2SleepLegCushionDeficit``; the later arrival moves,
     both freed by manual grep since the wcode census tool's own regex caps
     at 2 digits and cannot see either 3-digit code --
     ``test_weg2_wcode_uniqueness_1263.py`` stays green throughout, blind to
     both) the INSTANT they see a nonzero ``full`` beside work
     that should not exist: ``_weg2_xchg_inject_from_peer`` when its own
     filtered ``_cdescs`` is empty, and ``_weg2_await_vram_credit`` before it
     starts its own 120 s poll.

THE DANGER DIRECTION, named by the operator's order explicitly: a collector
that drains the WRONG card's lanes would compare and write foreign bytes and
report success -- silently wrong weights are worse than a refusal.
``test_M1_wrong_rank_misses_the_real_wedge`` below is that mutant: the same
undrained state, checked against the WRONG card identity, and the assertion
is that IT MISSES the wedge -- pinning why the real call sites must (and do)
pass this rank's own resolved identity rather than a fixed/derived stand-in.

ROUND 2 (file boundary extended to xchg_manifest.py/weight_exchange.py/
weight_exchange_region.py): the coordinator's strongest remaining candidate
was that a co-located DRAFT/MTP leg shares the SAME region-blind semaphore
names as the main leg (``sem_name``/``diagonal_sem_name``,
weight_exchange_region.py:1787-1843, neither carries a region/runner key) and
posts onto card 0's diagonal beside the main leg. REFUTED, on two
independent grounds, both verified below (and separately: round 3's own
metal read showed the diagonal's ``full=16`` is D's OWN TP-shard split
[17:7:8], one poster, not a second one -- see the module docstring further
down):

  1. ``model_runner.py:736-742`` -- "Only the LAST pipeline stage builds a
     producer" (``self.pp_rank == self.pp_size - 1``): the draft/MTP head
     lives on P's LAST stage. For P=PP3 that is rank 2, not rank 0 -- boot
     weg2xsn15 measured this directly (``weg2_memory_saver.py:2283-2296``,
     ``WEG2-XCHG-RESIDENT tag=weights_draft`` 1572 MiB on "P's LAST STAGE
     ONLY"). The card the wedge affects and the card the drafter resides on
     are DIFFERENT cards by construction.
  2. Even where it resided, no draft leg can currently REACH the shared
     semaphores at all: ``_weg2_shadow_plan`` (weight_updater.py) computes
     its region_tag EXCLUSIVELY from ``self.tp_worker.model_runner`` -- the
     MAIN runner -- on every call, from both
     ``_weg2_xchg_inject_from_peer`` and ``_weg2_xchg_deposit_before_sleep``,
     regardless of which ``tag`` string the per-tag loop is processing.
     ``test_the_draft_region_is_structurally_unreachable_from_either_call_site``
     below proves this by executing ``weight_exchange.weights_region_tag_for``
     against both a main and a draft ``RunnerShape``: only the main answer
     (``"weights"``) is EVER reachable from the real call sites, so a
     ``tag="weights_draft"`` resume/sleep step always narrows to the main
     region and finds zero descriptors on BOTH sides -- symmetric, silent,
     and a SEPARATE real defect (draft bytes never actually move through the
     bounce even though ``draft_tag_in_family()`` counts them as family
     members) but not a collision, and not card-0-specific. Flagged for a
     separate ticket rather than fixed here: fixing it means threading a
     per-tag region_tag through ``_weg2_shadow_plan``, which changes the
     signature both its callers use and is a wider diff than #1391's scope.

Since the semaphore names DO lack a region key (verified, ``sem_name``
concatenates only boot_nonce/src/dst/slot/kind, ``diagonal_sem_name`` only
boot_nonce/card/slot/kind) but this is NOT #1391's root, the region-key fix
the operator conditioned ("traegt der Name wirklich keinen Region-Schluessel
UND ist das die Wurzel") is NOT applied here -- the condition's second half
fails. The refusal (W108, see the renumber note above) stays regardless (operator: "der Refusal ist nicht der Notnagel,
er ist die Ratsche").

ALSO ADDED: ``WEG2-XCHG-LEG-TAGS`` (xchg_manifest.py, ``leg_plan_from_join``)
-- a boot log now prints, per leg, exactly which tags THIS rank's filtered
descriptor set (``mine``) carries beside the join's full family tag set. The
#1391 wedge (P PP0's ``mine`` missing ``weights_0``/``weights_1`` while D's
matching deposit line carries them) would show as two comparable log lines
instead of a DEBUG_HOLD dump plus a hand-rolled ``ctypes`` probe -- the
diagnostic BOOT7's instrument run can read directly.
"""

from __future__ import annotations

import inspect
import os
import sys
import tempfile
import threading
import time

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_weg2_xchg_transport_1273 import FakeDeviceOps, dev_ptr  # noqa: E402

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.managers.weg2_memory_saver import (
    Weg2VramCreditRefused,
    Weg2XchgLaneNeverDrainedRefused,
)
from sglang.srt.weg2 import weight_exchange as wx
from sglang.srt.weg2 import weight_exchange_bounce as bx
from sglang.srt.weg2 import weight_exchange_region as xr
from sglang.srt.weg2 import weight_exchange_transport as tp

Manager = wu.SchedulerWeightUpdaterManager


# ---------------------------------------------------------------------------
# Fakes -- the same shapes test_weg2_xchg_inject_wiring_1342.py drives the
# product with, kept local so this file stands alone.
# ---------------------------------------------------------------------------


class _FakeServerArgs:
    def __init__(self):
        self.enable_memory_saver = True
        self.enable_weights_cpu_backup = True
        self.enable_draft_weights_cpu_backup = False
        self.speculative_draft_model_path = None
        self.model_path = "/models/main"


class _FakeRunner:
    def __init__(self):
        self.model = None
        self.model_config = None


class _FakeWorker:
    def __init__(self):
        self.model_runner = _FakeRunner()


class _FakeDesc:
    """Only the fields the leg/filter read (mirrors test_..._1342.py)."""

    def __init__(self, name, tag="weights_3", nbytes=1024, src_rank=0, dst_rank=0):
        self.param_name = name
        self.tag = tag
        self.kind = "param"
        self.nbytes = int(nbytes)
        self.rows = 1
        self.run_bytes = int(nbytes)
        self.spitch = int(nbytes)
        self.src_rank = int(src_rank)
        self.dst_rank = int(dst_rank)


class _FakePlan:
    def __init__(self, descs):
        self.descs = tuple(descs)


class _FakeTerms:
    total_bytes = 402653184
    lane_slots = 8
    n_lanes = 8
    lanes_concurrent = 0

    def expression(self):
        return "(depth+1) x slot = 3 x 134217728"


def _manager(monkeypatch, *, group="P", rank=0, device=0, sems=None):
    monkeypatch.setattr(Manager, "_weg2_server_args",
                        lambda self: _FakeServerArgs(), raising=True)
    monkeypatch.setattr(Manager, "_weg2_group_name", lambda self: group,
                        raising=True)
    monkeypatch.setattr(Manager, "_weg2_rank", lambda self: rank, raising=True)
    monkeypatch.setattr(Manager, "_weg2_device_index", lambda self: device,
                        raising=True)
    monkeypatch.setattr(Manager, "_weg2_xchg_sems", lambda self: sems,
                        raising=True)
    m = Manager(
        tp_worker=_FakeWorker(), draft_worker=None, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )
    return m


@pytest.fixture()
def armed(monkeypatch):
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_AUTHORITATIVE)
    monkeypatch.setenv(xr.ENV_REGION_BOOT, "1391boot")
    assert wx.exchange_armed() is True
    assert wx.inject_authoritative() is True


@pytest.fixture()
def real_sems():
    """A REAL semaphore set, boot-scoped, cleaned up on the way out.

    Real POSIX semaphores rather than a stub: `getvalue`/`diagonal_getvalue`
    are ctypes calls into libc, and a test that stubbed them would prove
    nothing about the actual `sem_getvalue` plumbing boot weg2xsn31 needed a
    hand-rolled script to reach in the first place.
    """
    nonce = f"desk10-1391-{os.getpid()}"
    xr.unlink_semaphores(nonce)
    xr.create_semaphores(nonce)
    try:
        yield tp.SemSet(nonce)
    finally:
        xr.unlink_semaphores(nonce)


def _post_diagonal_bands(sems, card, n):
    """Prime the drain counter (empty 1->0, exactly `prime_drain` at tag 0)
    then post `n` bands -- the real boot's own sequence, so `empty` reads 0
    afterwards the same way weg2xsn31's hand-probe read it, not because
    nothing ever touched it."""
    sems.diagonal_timedwait(card, 0, "empty", 0.0)
    for _ in range(n):
        sems.diagonal_post(card, 0, "full")


def _post_cross_bands(sems, pair, n):
    sems.trywait(pair, 0, "empty")
    for _ in range(n):
        sems.post(pair, 0, "full")


# ===========================================================================
# 1. THE READ ITSELF -- diagonal_getvalue, and the detector built on it.
# ===========================================================================


def test_diagonal_getvalue_reads_what_was_posted(real_sems):
    assert real_sems.diagonal_getvalue(0, 0, "full") == 0
    assert real_sems.diagonal_getvalue(0, 0, "empty") == 1  # create_semaphores' arm
    _post_diagonal_bands(real_sems, 0, 16)
    assert real_sems.diagonal_getvalue(0, 0, "full") == 16
    assert real_sems.diagonal_getvalue(0, 0, "empty") == 0  # primed, never returned


def test_undrained_lanes_is_empty_on_a_clean_boot(real_sems):
    """THE #1233 CASE MUST STAY SILENT: a PP stage that legitimately owns no
    layers of a tag sees a clean (full=0) lane, and this check must not turn
    that legitimate silence into a false refusal."""
    assert Manager._weg2_xchg_undrained_lanes(None, 0, real_sems) == []


def test_undrained_lanes_finds_the_boot_weg2xsn31_shape(real_sems):
    """THE MEASURED NUMBERS: full=8 on both cross-into-0 lanes, full=16 on
    the diagonal, empty=0 on all three -- reproduced with the real semaphore
    set and read back through the product's own detector."""
    diag_ok = xr.CROSS_PAIRS.index((1, 0))
    cross_2_0 = xr.CROSS_PAIRS.index((2, 0))
    _post_diagonal_bands(real_sems, 0, 16)
    _post_cross_bands(real_sems, diag_ok, 8)
    _post_cross_bands(real_sems, cross_2_0, 8)

    found = Manager._weg2_xchg_undrained_lanes(None, 0, real_sems)
    by_lane = {lane: (full, empty) for lane, full, empty in found}
    assert by_lane["card0-0"] == (16, 0)
    assert by_lane["1-0-0"] == (8, 0)
    assert by_lane["2-0-0"] == (8, 0)
    assert len(found) == 3


def test_undrained_lanes_ignores_other_cards(real_sems):
    """A lane targeting card 1 must never count against card 0's check --
    otherwise the refusal would fire on every boot's normal traffic."""
    p01 = xr.CROSS_PAIRS.index((0, 1))
    _post_cross_bands(real_sems, p01, 3)
    assert Manager._weg2_xchg_undrained_lanes(None, 0, real_sems) == []
    assert Manager._weg2_xchg_undrained_lanes(None, 1, real_sems) != []


# ===========================================================================
# 2. WIRED -- the collect side. RED against the exact wedge, GREEN on the fix.
# ===========================================================================


def test_an_empty_collect_plan_beside_real_undrained_bands_refuses_by_name(
        monkeypatch, armed, real_sems):
    """RED-FIRST: this is boot weg2xsn31's own shape, driven through the real
    product method. Before this commit, `_cdescs=[]` logged one INFO line and
    returned -- the peer's `wait_drained` and this rank's own later credit
    wait were left to find out 120 s later."""
    _post_diagonal_bands(real_sems, 0, 16)
    m = _manager(monkeypatch, group="P", rank=0, sems=real_sems)
    # This rank's plan carries real work for weights_3 ONLY -- the #1233
    # legitimate shape -- so filtering by tag="weights_0" is empty, exactly
    # like P PP0's own resume loop measured on weg2xsn31. weg2xsn85: the
    # plan's one desc rides a CROSS lane (card 1 -> card 0), so the diagonal
    # card0-0 that holds the bands is covered by NO tag of the whole leg --
    # the shape W108 is for. (A diagonal desc of another tag would make the
    # bands that tag's own, see the lane-sparse test below.)
    monkeypatch.setattr(
        Manager, "_weg2_shadow_plan",
        lambda self, hook, g, r, agreed=None, require_agreement=None:
            (_FakePlan([_FakeDesc("a.w", tag="weights_3", src_rank=1,
                                  dst_rank=0)]), ""),
        raising=True)
    monkeypatch.setattr(Manager, "_weg2_xchg_device_ops", lambda self: object(),
                        raising=True)

    with pytest.raises(Weg2XchgLaneNeverDrainedRefused) as exc:
        m._weg2_xchg_inject_from_peer(terms=_FakeTerms(), tag="weights_0")
    msg = str(exc.value)
    # Pinned on the CLASS NAME, not the digit (#1265/#1306 collision class):
    # a digit pin breaks the instant a later census renumbers the code, and
    # this one already did once (W100 -> W108, 2026-09-14, TRAIN2's census
    # found it colliding with host_ledger.py's Weg2SleepLegCushionDeficit).
    assert "Weg2XchgLaneNeverDrainedRefused" in msg
    assert "card0-0" in msg
    assert "full=16" in msg
    assert "weights_0" in msg


def test_an_empty_collect_plan_on_a_clean_lane_stays_a_no_op(
        monkeypatch, armed, real_sems):
    """THE #1233 REGRESSION GUARD: the SAME empty plan, but nothing was ever
    deposited (the legitimate "this stage owns nothing of this tag" case) --
    must NOT raise. A detector that cannot tell this apart from the wedge
    would turn every normal boot into a refusal storm."""
    m = _manager(monkeypatch, group="P", rank=0, sems=real_sems)
    monkeypatch.setattr(
        Manager, "_weg2_shadow_plan",
        lambda self, hook, g, r, agreed=None, require_agreement=None:
            (_FakePlan([_FakeDesc("a.w", tag="weights_3")]), ""),
        raising=True)
    monkeypatch.setattr(Manager, "_weg2_xchg_device_ops", lambda self: object(),
                        raising=True)

    calls = {}
    monkeypatch.setattr(Manager, "_weg2_xchg_bounce_leg",
                        lambda self, **kw: calls.setdefault("kw", kw),
                        raising=True)
    m._weg2_xchg_inject_from_peer(terms=_FakeTerms(), tag="weights_0")
    assert calls["kw"]["descs"] == []


def test_xsn85_bands_of_a_later_tag_on_a_lane_the_whole_leg_covers_are_not_w108(
        monkeypatch, armed, real_sems):
    """weg2xsn85 (#1378): with P as the source, tags are lane-SPARSE and the
    source runs ahead past its no-op tags -- at D's weights_7 step lane
    1-0-0 held PP1's weights_4 bands and card0-0 PP0's weights_4 deposit in
    flight, both lanes the whole leg collects LATER. The check's own
    docstring said 'for ANY tag'; the code passed this tag's slice.
    MUTANT: `covered=set(_lanes.keys())` alone (the shipped form) -- the
    first assertion below is what it fails."""
    from sglang.srt.weg2 import weight_exchange_bounce as bx

    _post_diagonal_bands(real_sems, 0, 16)
    # The whole plan covers the diagonal via weights_3; this step is weights_0.
    plan = _FakePlan([_FakeDesc("a.w", tag="weights_3")])
    whole = set(bx.group_descs_by_pair(list(plan.descs)).keys())
    assert None in whole  # src_rank == dst_rank == 0: the on-card diagonal
    # 1. the check itself, with the whole-leg coverage: nothing is stuck.
    assert Manager._weg2_xchg_undrained_lanes(None, 0, real_sems,
                                              covered=whole) == []
    # ... and with the shipped per-tag (empty) coverage it WOULD refuse:
    assert Manager._weg2_xchg_undrained_lanes(None, 0, real_sems,
                                              covered=set()) != []
    # 2. the wiring: the collect call site hands the whole-leg lanes to the leg.
    m = _manager(monkeypatch, group="P", rank=0, sems=real_sems)
    monkeypatch.setattr(
        Manager, "_weg2_shadow_plan",
        lambda self, hook, g, r, agreed=None, require_agreement=None: (plan, ""),
        raising=True)
    monkeypatch.setattr(Manager, "_weg2_xchg_device_ops", lambda self: object(),
                        raising=True)
    calls = {}
    monkeypatch.setattr(Manager, "_weg2_xchg_bounce_leg",
                        lambda self, **kw: calls.setdefault("kw", kw),
                        raising=True)
    m._weg2_xchg_inject_from_peer(terms=_FakeTerms(), tag="weights_0")
    assert calls["kw"]["descs"] == []
    assert calls["kw"]["covered_lanes"] == whole


# ===========================================================================
# 3. WIRED -- the credit side. The SAME check, at weg2xsn31's own death site.
# ===========================================================================


class _CreditCapturesTheReader:
    """A stand-in `wait_for` that records the `stuck_lane_reader` callback
    and calls it once itself -- proves `_weg2_await_vram_credit` WIRES the
    callback (right rank, right sems) to the real `VramCredit.wait_for`
    call site, without needing a live 120s wait to prove it."""

    def __init__(self):
        self.kwargs = None
        self.stuck = None

    def wait_for(self, *a, **k):
        self.kwargs = k
        reader = k.get("stuck_lane_reader")
        assert reader is not None, "no stuck_lane_reader was passed at all"
        self.stuck = reader()
        return {"waited_s": 0.0, "claimed_bytes": 0}


def test_the_credit_wait_wires_a_stuck_lane_reader_not_a_one_shot_check(
        monkeypatch, real_sems):
    """#1391 ROUND 3: the coordinator's own finding -- a check taken ONCE at
    entry cannot see a condition that develops WHILE waiting (boot
    weg2xsn31's instrument run measured exactly that race: clean at entry,
    stuck 4s later). The check must be a CALLBACK `VramCredit.wait_for`
    polls on its own cadence, not code that runs before `wait_for` is even
    called. This pins the WIRING; the callback's own timing behaviour is
    pinned separately against the real `VramCredit.wait_for` below."""
    _post_diagonal_bands(real_sems, 0, 16)
    m = _manager(monkeypatch, group="P", rank=0, sems=real_sems)
    # weg2xsn257: the reader now applies the WHOLE-LEG lane coverage. A plan
    # that walks only the cross pair (1,0) leaves the diagonal uncovered, so
    # card0's 16 bands are still the xsn31 wedge this test pins.
    monkeypatch.setattr(Manager, "_weg2_xchg_whole_leg_lanes",
                        lambda self: {xr.CROSS_PAIRS.index((1, 0))},
                        raising=True)
    credit = _CreditCapturesTheReader()

    m._weg2_await_vram_credit(credit, "weights_3", 2916)

    assert credit.kwargs is not None, "credit.wait_for was never called"
    assert credit.kwargs["stuck_lane_reader"] is not None
    by_lane = {lane: (full, empty) for lane, full, empty in credit.stuck}
    assert by_lane["card0-0"] == (16, 0)


def test_xsn257_the_next_tags_bands_on_covered_lanes_are_not_a_wedge(
        monkeypatch, real_sems):
    """weg2xsn257 (17.09.): PP0 at tag weights_1 refused W108 `0.0s into
    the wait` because lanes 1-0-0 and 2-0-0 held 106 bands each -- the
    bands of weights_1 ITSELF, deposited by D-TP1/TP2 while PP0 still waited
    for D-TP0's credit. Every rank walks one tag order, so a full lane the
    whole-leg plan covers is the next tag's bands, never a wedge. The
    collect path applied that coverage since weg2xsn85; the credit wait
    did not. Pinned here: with both lanes covered the reader answers [] and
    the credit wait proceeds normally."""
    cross_1_0 = xr.CROSS_PAIRS.index((1, 0))
    cross_2_0 = xr.CROSS_PAIRS.index((2, 0))
    _post_cross_bands(real_sems, cross_1_0, 106)
    _post_cross_bands(real_sems, cross_2_0, 106)
    m = _manager(monkeypatch, group="P", rank=0, sems=real_sems)
    monkeypatch.setattr(Manager, "_weg2_xchg_whole_leg_lanes",
                        lambda self: {None, cross_1_0, cross_2_0},
                        raising=True)
    credit = _CreditCapturesTheReader()

    m._weg2_await_vram_credit(credit, "weights_1", 3820)

    assert credit.kwargs is not None, "credit.wait_for was never called"
    assert credit.stuck == [], (
        f"the xsn257 shape refused again: {credit.stuck}")
    # MUTANT (the shipped form before this fix): no coverage at all -- the
    # same lanes read as a wedge. This is the false refusal that killed the
    # first DFLASH flip on xsn256 and xsn257.
    assert len(Manager._weg2_xchg_undrained_lanes(None, 0, real_sems)) == 2
    assert Manager._weg2_xchg_undrained_lanes(
        None, 0, real_sems, covered={None, cross_1_0, cross_2_0}) == []


def test_no_whole_leg_plan_means_no_w108_from_the_credit_wait(
        monkeypatch, real_sems):
    """No plan, no proof: a rank that cannot derive its whole-leg coverage
    (a desk double, an unarmed boot) must not refuse on a full lane it
    cannot classify. The credit budget (W35) stays the detector, as for
    every caller that passes no reader at all."""
    _post_diagonal_bands(real_sems, 0, 16)
    m = _manager(monkeypatch, group="P", rank=0, sems=real_sems)
    monkeypatch.setattr(Manager, "_weg2_xchg_whole_leg_lanes",
                        lambda self: None, raising=True)
    credit = _CreditCapturesTheReader()

    m._weg2_await_vram_credit(credit, "weights_3", 2916)

    assert credit.stuck == []


def test_whole_leg_lanes_come_from_the_boot_cached_collect_plan(monkeypatch):
    """The coverage producer reads the SAME plan the collect path collects
    with (`_weg2_shadow_plan("authoritative", ...)`, boot-cached) and maps
    it through `group_descs_by_pair` -- one producer for both W108 sites.
    A plan that cannot be derived answers None, never an empty set (an
    empty set would mark EVERY full lane as a wedge)."""
    m = _manager(monkeypatch, group="P", rank=0, sems=None)
    seen = {}

    def _plan(self, hook, group, rank, *, agreed=None, require_agreement):
        seen["args"] = (hook, group, rank, agreed, require_agreement)
        return _FakePlan([
            _FakeDesc("a", tag="weights_0", src_rank=0, dst_rank=0),
            _FakeDesc("b", tag="weights_1", src_rank=1, dst_rank=0),
        ]), "ok"

    monkeypatch.setattr(Manager, "_weg2_shadow_plan", _plan, raising=True)
    lanes = m._weg2_xchg_whole_leg_lanes()
    assert seen["args"] == ("authoritative", "P", 0, None, False)
    assert lanes == {None, xr.CROSS_PAIRS.index((1, 0))}

    monkeypatch.setattr(Manager, "_weg2_shadow_plan",
                        lambda self, *a, **k: (None, "no manifest"),
                        raising=True)
    assert m._weg2_xchg_whole_leg_lanes() is None


def test_vram_credit_wait_for_catches_a_race_mid_wait_not_after_the_budget(
        tmp_path, monkeypatch):
    """THE REAL LOOP, driven end to end: `stuck_lane_reader` reports CLEAN
    for the first ~1.5s (the lane check's own 1s cadence gives it one or two
    clean looks) and then STUCK -- mirroring boot weg2xsn31's own timeline
    (clean at entry, saturating post lands 4s later). `wait_for` must raise
    `Weg2XchgLaneNeverDrainedRefused` well before its OWN 30s budget expires,
    proving the check lives INSIDE the poll loop and not only at the door.
    The persistence window (fnFL2x8) is shortened to 2 s here; the default
    20 s is pinned by its own test below.
    """
    from sglang.srt.managers.weg2_memory_saver import VramCredit

    monkeypatch.setenv("SGLANG_WEG2_W108_PERSIST_S", "2")
    credit = VramCredit("GPU-test-1391", credit_dir=str(tmp_path))
    t0 = time.monotonic()
    calls = {"n": 0}

    def _reader():
        calls["n"] += 1
        if calls["n"] <= 1:
            return []
        return [("card0-0", 16, 0)]

    with pytest.raises(Weg2XchgLaneNeverDrainedRefused) as exc:
        credit.wait_for(
            2916 * 1024 * 1024, budget_s=30.0, tag="weights_3",
            free_bytes_now=0, stuck_lane_reader=_reader)
    elapsed = time.monotonic() - t0
    assert elapsed < 10.0, (
        f"took {elapsed:.1f}s -- the lane check did not fire on its own "
        f"cadence, only (if at all) near the full budget")
    assert calls["n"] >= 2, "the reader was never polled more than once"
    msg = str(exc.value)
    # Class-name pin, not the digit -- see the comment at the first such
    # assertion above.
    assert "Weg2XchgLaneNeverDrainedRefused" in msg
    assert "card0-0" in msg


def test_vram_credit_wait_for_with_no_reader_is_byte_identical(tmp_path):
    """`stuck_lane_reader=None` (every existing caller before #1391) must
    behave exactly as before: refuse on the ORIGINAL W35 path, at the full
    budget, never on Weg2XchgLaneNeverDrainedRefused."""
    from sglang.srt.managers.weg2_memory_saver import VramCredit

    credit = VramCredit("GPU-test-1391b", credit_dir=str(tmp_path))
    with pytest.raises(Weg2VramCreditRefused) as exc:
        credit.wait_for(1024, budget_s=0.2, tag="weights_0", free_bytes_now=0)
    assert "W35" in str(exc.value)


def test_vram_credit_wait_for_rides_out_bands_of_its_own_tag(tmp_path,
                                                            monkeypatch):
    """fnFL2x8: PP2 waited for weights_14's credit (need 3048, free 2205);
    D's deposit of weights_14 ITSELF landed in lane 0-2-0 3.0 s into the wait
    (deposit -> pause -> credit: the bands always arrive before the credit)
    and W108 refused on the first full look, stranding the lane and starving
    PP0 90 s later. A reading that clears inside the persistence window is
    the lockstep working: the wait must go on (here to its own W35, since no
    peer funds this test), never W108."""
    from sglang.srt.managers.weg2_memory_saver import VramCredit

    monkeypatch.setenv("SGLANG_WEG2_W108_PERSIST_S", "3")
    credit = VramCredit("GPU-test-x8", credit_dir=str(tmp_path))
    calls = {"n": 0}

    def _reader():
        calls["n"] += 1
        return [("0-2-0", 117, 0)] if calls["n"] == 2 else []

    with pytest.raises(Weg2VramCreditRefused) as exc:
        credit.wait_for(
            3048 * 1024 * 1024, budget_s=4.0, tag="weights_14",
            free_bytes_now=0, stuck_lane_reader=_reader)
    assert "W35" in str(exc.value)
    assert "Weg2XchgLaneNeverDrainedRefused" not in str(exc.value)
    assert calls["n"] >= 3, "the full reading was never followed by a look"


def test_w108_persistence_default_survives_garbage(monkeypatch):
    """The default window is what keeps x8's false refusal gone on every arm
    that does not set the knob; a garbage value must not collapse it to 0."""
    from sglang.srt.managers.weg2_memory_saver import _w108_persist_s

    monkeypatch.delenv("SGLANG_WEG2_W108_PERSIST_S", raising=False)
    assert _w108_persist_s() == 20.0
    for bad in ("abc", ""):
        monkeypatch.setenv("SGLANG_WEG2_W108_PERSIST_S", bad)
        assert _w108_persist_s() == 20.0, bad


def test_the_credit_wait_polls_normally_on_a_clean_lane(monkeypatch, real_sems):
    """No false positive: a clean lane must let the real credit wait run."""
    m = _manager(monkeypatch, group="P", rank=0, sems=real_sems)
    called = {}

    class _Credit:
        def wait_for(self, *a, **k):
            called["yes"] = True
            return {"waited_s": 0.0, "claimed_bytes": 0}

    m._weg2_await_vram_credit(_Credit(), "weights_3", 2916)
    assert called.get("yes") is True


# ===========================================================================
# 4. THE MUTANT -- THE DANGER DIRECTION NAMED BY THE OPERATOR'S ORDER.
# ===========================================================================


def test_M1_wrong_rank_misses_the_real_wedge(real_sems):
    """MUTANT M1: the check computed for the WRONG card.

    This is the shape the order calls out explicitly: a collector reasoning
    about the wrong card drains (or here, fails to notice it must refuse for)
    the wrong lanes -- the silent-wrong-weights danger, not merely a missed
    refusal. Card 0 genuinely has 16 undrained bands; a check keyed to card 1
    (a plausible off-by-one -- the neighbouring card, or a stale identity
    read before this rank's own resolved rank was wired) finds NOTHING,
    because `_weg2_xchg_undrained_lanes` is -- correctly -- per-card, not a
    blanket "is anything stuck anywhere" scan (see
    `test_undrained_lanes_ignores_other_cards` above, which pins the same
    fact from the other side).  This is why both real call sites pass this
    rank's own resolved identity (`int(rank)` / `self._weg2_rank()`) and
    never a literal or a neighbour's: get that one integer wrong and the
    refusal this file adds goes silent on exactly the card that needed it.
    """
    _post_diagonal_bands(real_sems, 0, 16)
    diag_ok = xr.CROSS_PAIRS.index((1, 0))
    cross_2_0 = xr.CROSS_PAIRS.index((2, 0))
    _post_cross_bands(real_sems, diag_ok, 8)
    _post_cross_bands(real_sems, cross_2_0, 8)

    # The REAL rank (0) sees all three stuck lanes -- this is what the wired
    # call sites do and why they refuse.
    assert len(Manager._weg2_xchg_undrained_lanes(None, 0, real_sems)) == 3
    # The MUTANT rank (1, or 2) sees NOTHING on card 0's actual wedge --
    # exactly the silent miss the danger direction describes.
    assert Manager._weg2_xchg_undrained_lanes(None, 1, real_sems) == []
    assert Manager._weg2_xchg_undrained_lanes(None, 2, real_sems) == []


# ===========================================================================
# 5. THE DRAFT/MTP COLLISION HYPOTHESIS -- CHECKED AND REFUTED.
# ===========================================================================


def test_draft_residency_is_the_last_pp_stage_not_card_0():
    """model_runner.py:736-742: only ``pp_rank == pp_size - 1`` builds a draft
    KV producer. For P=PP3 that is rank 2 -- a different card than the one
    #1391's wedge affects (rank 0). This is the file:line the desk read to
    settle ground 1 of the refutation; it is asserted here as a STRING match
    against the live source so a future edit that moves the gate re-trips
    this test rather than silently invalidating the refutation."""
    import inspect

    from sglang.srt.model_executor import model_runner as mr

    src = inspect.getsource(mr.ModelRunner.__init__)
    assert "self.pp_rank == self.pp_size - 1" in src, (
        "the draft-KV-producer gate moved or was rephrased -- re-check "
        "whether the draft/MTP head still resides on the LAST PP stage "
        "before trusting the #1391 refutation that rests on it")


def test_the_draft_region_is_structurally_unreachable_from_either_call_site(
        monkeypatch):
    """GROUND 2 OF THE REFUTATION, executed: `_weg2_shadow_plan`'s region_tag
    is `wx.weights_region_tag_for(wx.RunnerShape.of(runner))` where `runner`
    is ALWAYS `self.tp_worker.model_runner` (weight_updater.py's own source,
    both call sites) -- never the draft runner. A main-shaped RunnerShape
    therefore NEVER classifies as SHAPE_DRAFT, so `tag="weights_draft"` can
    never select the draft region through either
    `_weg2_xchg_inject_from_peer` or `_weg2_xchg_deposit_before_sleep`,
    regardless of what the resume/sleep loop's own `tag` argument says. Not a
    collision (a collision needs two legs on the same name); a separate,
    symmetric, silent no-op -- flagged, not fixed here (#1391's scope is the
    card-0 wedge, not this)."""
    monkeypatch.setenv(wx.WEIGHT_SOURCE_ENV, wx.WEIGHT_SOURCE_EXCHANGE)
    monkeypatch.setenv(wx.INJECT_ENV, wx.INJECT_AUTHORITATIVE)
    assert wx.exchange_armed() is True

    class _MainRunner:
        is_draft_worker = False
        server_args = None

    class _DraftRunner:
        is_draft_worker = True

        class server_args:
            speculative_algorithm = "eagle"

    main_tag = wx.weights_region_tag_for(wx.RunnerShape.of(_MainRunner()))
    draft_tag = wx.weights_region_tag_for(wx.RunnerShape.of(_DraftRunner()))
    assert main_tag == wx.GPU_MEMORY_TYPE_WEIGHTS
    assert draft_tag == wx.GPU_MEMORY_TYPE_WEIGHTS_DRAFT
    assert main_tag != draft_tag, (
        "the two region tags collapsed -- if they ever match, the "
        "structural argument above (main runner can never resolve the draft "
        "tag) no longer holds and the refutation needs re-checking")
    # THE ACTUAL CALL SITES: `self.tp_worker.model_runner` is what both
    # `_weg2_xchg_inject_from_peer` and `_weg2_xchg_deposit_before_sleep`
    # read (grep-verified: no call site anywhere threads a per-tag or
    # per-region override into `_weg2_shadow_plan`), and that object is
    # always the MAIN runner -- never `_get_draft_model_runner(...)`.
    from sglang.srt.managers.scheduler_components import weight_updater as _wu

    src = inspect.getsource(_wu.SchedulerWeightUpdaterManager._weg2_shadow_plan_uncached)
    assert 'getattr(self.tp_worker, "model_runner", None)' in src
    assert "draft_worker" not in src, (
        "_weg2_shadow_plan now reads the draft runner -- the region-tag "
        "no-op this test documents may be fixed; re-verify #1391's scope "
        "note rather than assuming it still applies")


def test_the_leg_tags_line_names_this_ranks_own_tags(monkeypatch):
    """The new WEG2-XCHG-LEG-TAGS line (xchg_manifest.py::leg_plan_from_join):
    a boot log now carries, per leg, exactly which tags THIS rank's own
    filtered descriptors hold -- the #1391 wedge (P PP0 missing weights_0/1)
    would show here as a rank whose `tags=` omits what a peer's own line
    carries for the matching src/dst, readable without a DEBUG_HOLD dump."""
    from sglang.srt.weg2 import xchg_manifest as xm

    class _Piece:
        def __init__(self, name, tag, rows=4, cols=4, item=2):
            self.param_name = name
            self.tensor_class = "rows"
            self.rows_full = rows
            self.cols_full = cols
            self.itemsize = item
            self.tag = tag
            self.nbytes = rows * cols * item
            self.component_rows = ()

        @property
        def key(self):
            return (self.rows_full, self.cols_full, self.itemsize,
                    self.tensor_class)

    p_manifest = xm.RankManifest(
        group="P", rank=0, card=0, region_tag="weights", boot_token="b1",
        tp_rank=0, pp_rank=0,
        pieces=(_Piece("model.layers.0.w", "weights_0", rows=4, cols=4),))
    # A real ROW CUT (D's 3 ranks split the 4 rows 2/1/1 -- summing to the
    # PP side's whole 4), not three copies of a smaller tensor: the join's
    # own tiling check (`_check_tiles`) refuses anything else, correctly.
    d_rows = (2, 1, 1)
    d_manifests = [
        xm.RankManifest(
            group="D", rank=r, card=r, region_tag="weights", boot_token="b1",
            tp_rank=r, pp_rank=0,
            pieces=(_Piece("model.layers.0.w", "weights_0",
                           rows=d_rows[r], cols=4),))
        for r in range(3)
    ]
    lines = []
    leg, reason = xm.leg_plan_from_join(
        hook="authoritative", group="P", rank=0,
        manifests=[p_manifest] + d_manifests, pp_group="P", tp_group="D",
        model=None, region_tag="weights", log=lines.append)
    assert leg is not None, reason
    tag_lines = [l for l in lines if l.startswith("WEG2-XCHG-LEG-TAGS")]
    assert len(tag_lines) == 1, lines
    assert "tags=weights_0" in tag_lines[0]
    assert "family_tags=weights_0" in tag_lines[0]
    assert "rank=0" in tag_lines[0]
    assert "hook=authoritative" in tag_lines[0]


# ===========================================================================
# 6. THE DEADLOCK, HERMETIC -- ROUND 4. Two tags, one co-located lane,
# deposit and collect through the REAL `_weg2_xchg_bounce_leg` method: the
# OLD calling pattern (collect once, `tag=None`, after both tags would have
# resumed -- `_weg2_xchg_shadow_compare`'s shape before this round) hangs;
# the NEW pattern (collect once PER TAG, matching CARRIER_EXCHANGE's own
# loop and now `_weg2_xchg_shadow_armed_for`'s per-tag branch) drains.
# ===========================================================================


def _bare_manager():
    """No monkeypatching: `_weg2_xchg_bounce_leg` takes `sems`/`rank` as
    arguments and never reads `self._weg2_xchg_sems()`/`self._weg2_rank()`,
    so the plain product object is enough -- driving the REAL method, not a
    reimplementation of it."""
    return Manager(
        tp_worker=_FakeWorker(), draft_worker=None, tp_cpu_group=None,
        memory_saver_adapter=None, flush_cache=lambda *a, **k: True,
        is_fully_idle=lambda *a, **k: True,
    )


class _ShortBudgetRendezvous(bx.CrossSlotRendezvous):
    """The REAL rendezvous, budget shortened so a genuine deadlock costs
    seconds instead of the product's real 120s -- same trick
    test_weg2_lane_lockstep_1374.py uses at the module-function level; here
    it is applied one level up, at the MIXIN METHOD `_weg2_xchg_bounce_leg`
    actually constructs internally and gives no override for."""

    def __init__(self, *a, **k):
        k.setdefault("budget_s", 2.0)
        super().__init__(*a, **k)


def _diag_desc(tag_idx, *, ops_src, ops_dst, nbytes=64):
    return wx.XchgDesc(
        tag=f"weights_{tag_idx}", src_rank=0, dst_rank=0,
        param_name=f"model.layers.{tag_idx}.w",
        src_ptr=dev_ptr(0, tag_idx * 4096), dst_ptr=dev_ptr(0, tag_idx * 4096),
        kind=wx.FLAT, nbytes=nbytes, rows=1, run_bytes=nbytes,
        spitch=0, dpitch=0, src_off=0, dst_off=0,
    )


def _lane_descs_filter(descs):
    """2026-09-15: the sequential transport derives each lane's unit list
    from the manifests (`_weg2_seq_lane_descs`); this harness has no
    manifests, only explicit descs -- hand them to the leg filtered the way
    the derivation would (by tag, and by lane: the on-card diagonal is
    src == dst == card, a cross lane is one of CROSS_PAIRS)."""
    from sglang.srt.weg2 import weight_exchange_region as _xr

    def _f(self, *, hook, group, rank, pair, card, tag, log=None):
        out = []
        for d in descs:
            if tag is not None and str(getattr(d, "tag", "")) != str(tag):
                continue
            if pair is None:
                if int(d.src_rank) == int(d.dst_rank) == int(card):
                    out.append(d)
            elif (int(d.src_rank), int(d.dst_rank)) == tuple(
                    _xr.CROSS_PAIRS[int(pair)]):
                out.append(d)
        return out
    return _f


def _run_two_tag_leg(*, dest_per_tag: bool, tmp_path, timeout=8.0):
    """SOURCE deposits weights_0 then weights_1 through the REAL per-tag
    lockstep (exactly D's own shape, always per-tag regardless of carrier).
    DEST either mirrors it (``dest_per_tag=True``, the FIX) or collects both
    tags in ONE call with ``tag=None`` (``dest_per_tag=False``, shadow
    mode's shape BEFORE this round). Returns ``(source_result, dest_result)``.
    """
    from sglang.srt.weg2 import xchg_bounce as xb

    nonce = f"desk10-1391-r4-{os.getpid()}-{id(tmp_path)}"
    xr.unlink_semaphores(nonce)
    xr.create_semaphores(nonce)
    root = str(tmp_path)
    # A REAL BounceTerms, not the hand-rolled `_FakeTerms` -- this path runs
    # `run_bounce_leg` for real (real bytes, real slot record), which reads
    # more of the term than the plan-only tests above need.
    terms = xb.BounceTerms(
        bytes_per_direction=64, n_layers=1, widest_layer_bytes=64, depth=1,
        pairs=6, slot_bytes=4096, mean_layer_bytes=64, buffer_bytes=4096,
        staging_bytes=4096, n_lanes=1, max_tag_bytes=0, lanes_concurrent=0,
    )
    src_ops = FakeDeviceOps(root, 0)
    dst_ops = FakeDeviceOps(root, 0)
    src_ops.raw_malloc(0, 4096 * 4)
    dst_ops.raw_malloc(0, 4096 * 4)
    sems_src = tp.SemSet(nonce)
    sems_dst = tp.SemSet(nonce)
    results = {}

    def _source():
        mgr = _bare_manager()
        try:
            for t in range(2):
                mgr._weg2_xchg_bounce_leg(
                    descs=[_diag_desc(t, ops_src=src_ops, ops_dst=dst_ops)],
                    ops=src_ops, boot_nonce=nonce, terms=terms,
                    mode=wx.INJECT_AUTHORITATIVE, device=0, hook="source",
                    sems=sems_src, tag=f"weights_{t}", rank=0,
                )
            results["source"] = "deposited"
        except Exception as exc:  # noqa: BLE001
            results["source"] = f"{type(exc).__name__}: {exc}"

    def _dest():
        mgr = _bare_manager()
        try:
            if dest_per_tag:
                for t in range(2):
                    mgr._weg2_xchg_bounce_leg(
                        descs=[_diag_desc(t, ops_src=src_ops, ops_dst=dst_ops)],
                        ops=dst_ops, boot_nonce=nonce, terms=terms,
                        mode=wx.INJECT_AUTHORITATIVE, device=0,
                        hook="authoritative", sems=sems_dst,
                        tag=f"weights_{t}", rank=0,
                    )
            else:
                # THE OLD SHAPE: give resume() time to map both tags (as
                # `resume_memory_occupation` did before this round), THEN
                # collect the whole plan in ONE call with tag=None -- exactly
                # `_weg2_xchg_shadow_compare`'s call before this fix.
                time.sleep(0.3)
                mgr._weg2_xchg_bounce_leg(
                    descs=[_diag_desc(t, ops_src=src_ops, ops_dst=dst_ops)
                          for t in range(2)],
                    ops=dst_ops, boot_nonce=nonce, terms=terms,
                    mode=wx.INJECT_AUTHORITATIVE, device=0,
                    hook="authoritative", sems=sems_dst, tag=None, rank=0,
                )
            results["dest"] = "collected"
        except Exception as exc:  # noqa: BLE001
            results["dest"] = f"{type(exc).__name__}: {exc}"

    from unittest import mock
    _all_descs = [_diag_desc(t, ops_src=src_ops, ops_dst=dst_ops) for t in range(2)]
    threads = [threading.Thread(target=_source), threading.Thread(target=_dest)]
    with mock.patch.object(Manager, "_weg2_seq_lane_descs",
                           _lane_descs_filter(_all_descs)):
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=timeout)
    alive = [th for th in threads if th.is_alive()]
    try:
        return results, alive
    finally:
        xr.unlink_semaphores(nonce)


@pytest.fixture(autouse=True)
def _short_rendezvous_budget(monkeypatch, request):
    """Only for the deadlock-reproduction tests below: the REAL rendezvous,
    with its budget cut from 120s to 2s so a genuine deadlock is a few
    seconds of test time, not two minutes."""
    if "deadlock" in request.node.name:
        monkeypatch.setattr(bx, "CrossSlotRendezvous", _ShortBudgetRendezvous)
    yield


def test_M2_old_style_whole_plan_collect_deadlocks_RED_on_the_wall(tmp_path, monkeypatch):
    # 2026-09-15: this mutant models the per-tag drain law at buffer depth 1
    # (the source waits for tag t's drain before tag t+1). Under the shipped
    # depth 2 a TWO-tag leg never waits, so the deadlock it reproduces needs
    # the depth-1 form it was written for.
    monkeypatch.setenv(bx.SEQ_BUFFER_DEPTH_ENV, "1")
    """RED-FIRST: this IS boot weg2xsn31's wedge, hermetic. Before this
    round, shadow mode's only collect ran exactly this way -- once, after
    both tags, with `tag=None`. D's second deposit blocks in `wait_drained`
    forever (here: for the shortened 2s budget) because nobody ever posts
    `drained` per tag."""
    results, alive = _run_two_tag_leg(dest_per_tag=False, tmp_path=tmp_path)
    src_msg = str(results.get("source", ""))
    # 2026-09-15 (sequential per-unit transport): the old shape now wedges
    # one step EARLIER as well -- the whole-plan collector waits for tag 1's
    # units on the lane's own handshake while the source waits for tag 0's
    # drain; either side may be the one the harness timeout catches. What
    # the mutant proves is unchanged: the old-style collect never completes
    # and the source never gets its drain.
    assert results.get("dest") != "collected", results
    assert alive or "Weg2XchgPlanDisagree" in src_msg, (
        f"expected the old-style leg to wedge (a live thread or the source's "
        f"drain timeout); got results={results} alive={[t.name for t in alive]}")
    if "Weg2XchgPlanDisagree" in src_msg:
        assert "waited for the collector to drain" in src_msg, results


def test_the_fix_per_tag_collect_drains_the_same_lane(tmp_path):
    """GREEN: the SAME lane, the SAME two tags, DEST now calling per tag
    (matching the fix in `resume_memory_occupation` /
    `_weg2_xchg_shadow_armed_for`) -- both sides complete."""
    results, alive = _run_two_tag_leg(dest_per_tag=True, tmp_path=tmp_path)
    assert results.get("source") == "deposited", results
    assert results.get("dest") == "collected", results
    assert not alive, "a thread is still running -- the fix did not drain"


# ===========================================================================
# 7. THE DANGER-DIRECTION MUTANT (operator order): a per-tag grade that
# posts `drained` for a DIFFERENT tag than the one it just compared reports
# MATCH on bytes nobody checked -- still wrong, and worse than the deadlock.
# ===========================================================================


def test_M3_mismatched_tag_between_grade_and_drain_must_not_be_reachable(
        monkeypatch, real_sems):
    """THE MUTANT: `_weg2_xchg_bounce_leg` posts `drained(tag=...)` using
    WHATEVER STRING the caller passed as `tag`, and that same string is what
    filtered `descs` down to the bytes actually graded one frame up in
    `_weg2_xchg_inject_from_peer`/`_weg2_xchg_deposit_before_sleep` -- ONE
    variable, threaded through both the filter and the post. This test
    proves there is no SECOND place a tag could be substituted between the
    two: patch `_weg2_xchg_bounce_leg` to receive a genuinely different
    `tag` string than the one the wrapper filtered `_cdescs` with, and the
    filtered set (real bytes) must correspond to the SAME tag the drain
    names -- never a mismatch silently accepted.

    Concretely: call `_weg2_xchg_inject_from_peer` with `tag="weights_0"`
    (so `_cdescs` is filtered to weights_0's real descriptors) while
    recording, via a wrapped `_weg2_xchg_bounce_leg`, the `tag` value AND
    the descriptor tags it actually receives -- they must be the SAME set.
    A version that let a caller pass one tag to filter and another to drain
    (the mutant) would show `tag != {d.tag for d in descs}` here; the real
    code cannot, because both come from the ONE `tag` parameter of
    `_weg2_xchg_inject_from_peer` with no second read anywhere in between.
    """
    m = _bare_manager()
    monkeypatch.setenv(xr.ENV_REGION_BOOT, "1391boot-m3")
    monkeypatch.setattr(Manager, "_weg2_server_args",
                        lambda self: _FakeServerArgs(), raising=True)
    monkeypatch.setattr(Manager, "_weg2_group_name", lambda self: "P",
                        raising=True)
    monkeypatch.setattr(Manager, "_weg2_rank", lambda self: 0, raising=True)
    monkeypatch.setattr(Manager, "_weg2_device_index", lambda self: 0,
                        raising=True)
    monkeypatch.setattr(Manager, "_weg2_xchg_sems", lambda self: real_sems,
                        raising=True)
    monkeypatch.setattr(
        Manager, "_weg2_shadow_plan",
        lambda self, hook, g, r, agreed=None, require_agreement=None:
            (_FakePlan([_FakeDesc("a.w", tag="weights_0"),
                       _FakeDesc("b.w", tag="weights_1")]), ""),
        raising=True)
    monkeypatch.setattr(Manager, "_weg2_xchg_device_ops", lambda self: object(),
                        raising=True)

    # THE SPY DOES NOT CALL THROUGH: this test is about the WIRING (which
    # tag reaches the leg beside which descriptors), not about actually
    # moving bytes -- that is what the deadlock-reproduction tests above and
    # the byte-exact execution smoke (test_weg2_xchg_bounce_execution_smoke_1273.py)
    # already cover with a real device layer.
    seen = {}

    def _spy(self, *, descs, tag=None, **kw):
        seen["tag"] = tag
        seen["desc_tags"] = {d.tag for d in descs}

    monkeypatch.setattr(Manager, "_weg2_xchg_bounce_leg", _spy, raising=True)

    m._weg2_xchg_inject_from_peer(terms=_FakeTerms(), tag="weights_0")

    assert seen["tag"] == "weights_0"
    assert seen["desc_tags"] == {"weights_0"}, (
        "the descriptors the leg graded do not match the tag it will drain "
        "-- this is the mutant: MATCH would be reported on weights_1's "
        "bytes while weights_0's drain is the one posted")


if __name__ == "__main__":
    import unittest

    unittest.main()
