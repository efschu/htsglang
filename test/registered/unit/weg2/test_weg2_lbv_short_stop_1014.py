# SPDX-License-Identifier: Apache-2.0
"""NF-RANG-AUDIT 1014, Fix 2 (S4): an ``LB-VERDICT SHORT`` is a named group stop, never a
retreat into the void pass.

853534a9ff states "a stage that cannot pay a RELEASED load-back stops by name", but on
the base ``scheduler.py``'s ``except AgreedRoomShort`` hands every AgreedRoomShort to
``prefill_oom_retreat.retreat_loadback``: on PP0 that is RETRY + a void frame, while the
followers have ``consume``d the verdict and loaded H->D in the same pass. After the void
PP0 still holds the rid released and the followers do not -- next pass PP0 admits alone,
the followers hold, #1233 one pass later. The verdict was distributed; its decider taking
it back is a detected divergence of the walls -> the group stops by name.

RANK-EQUALITY MUTANT (``test_rank_equality_mutant_the_retreat_split_the_stages``):
``SGLANG_WEG2_LBV_SHORT_STOP=0`` = the retreat of before takes the short (RETRY): PP0's
verdict stays released, the followers' is spent -> two states. On: ``None`` on every
stage, the exception propagates, every stage stops alike.
"""
from __future__ import annotations

import inspect
import types

import pytest

from sglang.srt.weg2 import pp_loadback_verdict as LBV
from sglang.srt.weg2 import prefill_oom_retreat as PR
from sglang.srt.weg2.pp_slot_fidelity import AgreedRoomShort

RID = "weg2-40-310"


def _short(flag: bool) -> AgreedRoomShort:
    return AgreedRoomShort(f"{LBV.MARK} SHORT rid={RID} kv_tokens=150720 floor=140416 avail_after=140416",
                           rid=RID, kv_tokens=150720, agreed=140416, avail=140416, avail_after=140416,
                           lb_verdict_short=flag)


def _sched(pp_rank: int):
    return types.SimpleNamespace(ps=types.SimpleNamespace(pp_rank=pp_rank, pp_size=3, tp_size=1),
                                 tree_cache=None, running_batch=None, max_total_num_tokens=0)


@pytest.fixture
def clean(monkeypatch):
    monkeypatch.delenv(PR.ENV_LBV_SHORT_STOP, raising=False)
    monkeypatch.delenv(PR.ENV, raising=False)


def test_the_exception_carries_the_mark():
    assert _short(True).lb_verdict_short is True
    assert _short(False).lb_verdict_short is False
    assert AgreedRoomShort("plain").lb_verdict_short is False


def test_a_verdict_short_is_not_taken_on_any_stage(clean, monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("applies() must not be consulted for an LB-VERDICT SHORT")

    monkeypatch.setattr(PR, "applies", _boom)
    req = types.SimpleNamespace(rid=RID)
    for rank in (0, 1, 2):
        assert PR.retreat_loadback(_sched(rank), _short(True), req, adder=None) is None


def test_a_plain_room_short_keeps_the_retreat_path(clean, monkeypatch):
    seen = []

    def _applies(sched, adder=None, can_run_list=()):
        seen.append(1)
        return False, "tp_group"

    monkeypatch.setattr(PR, "applies", _applies)
    assert PR.retreat_loadback(_sched(0), _short(False), types.SimpleNamespace(rid=RID), adder=None) is None
    assert seen == [1]  # the generic PREFILL-OOM short still asks applies()


def test_rank_equality_mutant_the_retreat_split_the_stages(clean, monkeypatch):
    """Old behaviour (switch off): the retreat TAKES the short on PP0 (RETRY) -- PP0's verdict
    stays released, the followers' is already spent: two states after the void. New: None
    on every stage, the stop is uniform."""
    monkeypatch.setattr(PR, "applies", lambda sched, adder=None, can_run_list=(): (True, ""))
    monkeypatch.setattr(PR, "frame_form", lambda sched: False)
    req = types.SimpleNamespace(rid=RID)
    # the stages' LB-VERDICT state at the moment of the short: PP0 released, the followers consumed
    pp0, pp1, pp2 = (types.SimpleNamespace() for _ in range(3))
    for t, is_pp0 in ((pp0, True), (pp1, False), (pp2, False)):
        setattr(t, LBV._STATE_ATTR, LBV._State(is_pp0))
        LBV.absorb(t, LBV.Weg2PpLoadbackVerdict(rid=RID))
    LBV.consume(pp1, RID)
    LBV.consume(pp2, RID)
    assert LBV.released(pp0, RID) and not LBV.released(pp1, RID) and not LBV.released(pp2, RID)

    monkeypatch.setenv(PR.ENV_LBV_SHORT_STOP, "0")
    old = PR.retreat_loadback(_sched(0), _short(True), req, adder=None)
    assert old == PR.RETRY  # the mutant: PP0 voids and keeps its verdict; the followers keep their rows
    # the detector: after the old retreat the stages disagree about the rid's verdict
    assert [LBV.released(t, RID) for t in (pp0, pp1, pp2)] == [True, False, False]

    monkeypatch.delenv(PR.ENV_LBV_SHORT_STOP, raising=False)
    assert [PR.retreat_loadback(_sched(r), _short(True), req, adder=None) for r in (0, 1, 2)] == [None, None, None]


def test_default_on(clean):
    assert PR.lbv_short_stop_enabled({}) is True
    assert PR.lbv_short_stop_enabled({PR.ENV_LBV_SHORT_STOP: "0"}) is False


def test_source_pins():
    """URC raises the short WITH the mark, and the scheduler re-raises on ``None``."""
    from sglang.srt.managers import scheduler as sched_mod
    from sglang.srt.mem_cache import unified_radix_cache as URC

    src = inspect.getsource(URC.UnifiedRadixCache.load_back)
    i = src.index("SHORT rid=")
    assert "lb_verdict_short=True" in src[i:i + 900]
    s2 = inspect.getsource(sched_mod.Scheduler.get_new_batch_prefill_raw if hasattr(sched_mod.Scheduler, "get_new_batch_prefill_raw")
                           else sched_mod.Scheduler._get_new_batch_prefill_raw)
    j = s2.index("except AgreedRoomShort")
    assert "retreat_loadback" in s2[j:j + 600] and "if _lb_verdict is None:\n" in s2[j:j + 700]
    assert "raise" in s2[j:j + 760]
