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
  2. Two call sites raise ``Weg2XchgLaneNeverDrainedRefused`` (W100, freed by
     the wcode census -- ``test_weg2_wcode_uniqueness_1263.py`` stays green
     with the new code) the INSTANT they see a nonzero ``full`` beside work
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
"""

from __future__ import annotations

import os
import tempfile

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers.scheduler_components import weight_updater as wu
from sglang.srt.managers.weg2_memory_saver import Weg2XchgLaneNeverDrainedRefused
from sglang.srt.weg2 import weight_exchange as wx
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
    # like P PP0's own resume loop measured on weg2xsn31.
    monkeypatch.setattr(
        Manager, "_weg2_shadow_plan",
        lambda self, hook, g, r, agreed=None, require_agreement=None:
            (_FakePlan([_FakeDesc("a.w", tag="weights_3")]), ""),
        raising=True)
    monkeypatch.setattr(Manager, "_weg2_xchg_device_ops", lambda self: object(),
                        raising=True)

    with pytest.raises(Weg2XchgLaneNeverDrainedRefused) as exc:
        m._weg2_xchg_inject_from_peer(terms=_FakeTerms(), tag="weights_0")
    msg = str(exc.value)
    assert "W100" in msg
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


# ===========================================================================
# 3. WIRED -- the credit side. The SAME check, at weg2xsn31's own death site.
# ===========================================================================


class _CreditMustNotBeAsked:
    """A credit object whose `wait_for` fails the test if it is ever reached
    -- the assertion that the new check runs BEFORE the 120 s poll, not
    beside or after it."""

    def wait_for(self, *a, **k):  # pragma: no cover -- must never run
        raise AssertionError(
            "credit.wait_for() was called -- the undrained-lane check did "
            "not short-circuit before the 120s poll")


def test_the_credit_wait_refuses_before_polling_when_a_lane_is_undrained(
        monkeypatch, real_sems):
    _post_diagonal_bands(real_sems, 0, 16)
    m = _manager(monkeypatch, group="P", rank=0, sems=real_sems)

    with pytest.raises(Weg2XchgLaneNeverDrainedRefused) as exc:
        m._weg2_await_vram_credit(_CreditMustNotBeAsked(), "weights_3", 2916)
    msg = str(exc.value)
    assert "W100" in msg
    assert "weights_3" in msg
    assert "card0-0" in msg


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


if __name__ == "__main__":
    import unittest

    unittest.main()
