"""#1416f: PP0 admits a paced told without waiting out the window when
nothing is in pipeline flight.

Metal (NF z30e ca2a9706ec, boot ...stvsyncbar1dauer09282117, 21:25:50Z): the
first request after a P wake, weg2-8-29, 16384 tokens from the store. PP0's
read 831 ms (WEG2-LOAD-DEVICE read_ms=807 queue_ms=23), the followers' 49/71
ms; "#1416e PACED-ADMIT rid=weg2-8-2 told=16384 window=1.04s waited=1.04s"
with the whole pipeline empty -- ~1 s of three idle GPUs before the forward
(wake -> first PP0 forward 2.18 s). The ring double is the #1416e one; the
stages here carry the PP ring fields (``mbs``, ``running_mbs``,
``chunked_req``) the idle predicate reads.
"""

import os
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import test_weg2_store_told_paced_1416e as ring_mod  # noqa: E402

from sglang.srt.managers import weg2_store_told as m  # noqa: E402

DT = ring_mod.DT
RID = "weg2-8-29"
TOLD = 16384


def _empty_batch():
    return SimpleNamespace(is_empty=lambda: True)


def _busy_batch():
    return SimpleNamespace(is_empty=lambda: False)


def _ring(monkeypatch, pp0_s=0.831, follower_s=(0.071, 0.049)):
    ring = ring_mod._Ring(
        monkeypatch,
        {RID: TOLD},
        {0: {RID: pp0_s}, 1: {RID: follower_s[0]}, 2: {RID: follower_s[1]}},
    )
    for s in ring.stages:
        s.mbs = [None, None, None]
        s.running_mbs = [_empty_batch(), _empty_batch(), _empty_batch()]
        s.chunked_req = None
    return ring


def _admit_pass(ring):
    plans = ring.plans(RID)
    assert plans[0] == plans[1] == plans[2] and len(plans[0]) == 1, plans
    assert plans[0][0][2] == TOLD
    return plans[0][0][0]


def _told_pass(ring):
    return next(k for k in sorted(ring.wire) if ring.wire[k])


def test_idle_pipeline_admits_in_the_next_pass(monkeypatch):
    """The metal form: PP0's read-ahead goes out at ~0.83 s; with nothing in
    flight the Admit follows one pass later, not 1.04 s later. Ranks agree,
    and the followers' residual wait is bounded by their own read."""
    monkeypatch.delenv("SGLANG_WEG2_DISABLE_TOLD_PACE_IDLE_SKIP", raising=False)
    ring = _ring(monkeypatch)
    ring.arrive(RID)
    ring.run(80)
    told_k = _told_pass(ring)
    admit_k = _admit_pass(ring)
    assert admit_k == told_k + 1, (told_k, admit_k)
    assert (admit_k - told_k) * DT < 0.2  # base: >= the 1.04 s window
    assert ring.sleeps[0] == 0.0
    assert ring.sleeps[1] <= 0.071 + 1e-9 and ring.sleeps[2] <= 0.049 + 1e-9


def test_idle_admit_is_still_read_ahead_then_admit(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_DISABLE_TOLD_PACE_IDLE_SKIP", raising=False)
    ring = _ring(monkeypatch)
    ring.arrive(RID)
    ring.run(80)
    kinds = [(type(o).__name__, getattr(o, "paced", None)) for k in sorted(ring.wire) for o in ring.wire[k]]
    assert kinds == [("Weg2StoreTold", True), ("Weg2StoreAdmit", None)]


@pytest.mark.parametrize("busy", ["mbs", "running", "chunked"])
def test_work_in_flight_keeps_the_window(monkeypatch, busy):
    """Something in pipeline flight on PP0: the #1416e risk is real, the
    window runs to its end exactly as before."""
    monkeypatch.delenv("SGLANG_WEG2_DISABLE_TOLD_PACE_IDLE_SKIP", raising=False)
    ring = _ring(monkeypatch)
    pp0 = ring.stages[0]
    if busy == "mbs":
        pp0.mbs[1] = object()
    elif busy == "running":
        pp0.running_mbs[2] = _busy_batch()
    else:
        pp0.chunked_req = object()
    ring.arrive(RID)
    ring.run(120)
    told_k = _told_pass(ring)
    admit_k = _admit_pass(ring)
    assert (admit_k - told_k) * DT >= 1.0, (told_k, admit_k)


def test_switch_off_keeps_the_window(monkeypatch):
    monkeypatch.setenv("SGLANG_WEG2_DISABLE_TOLD_PACE_IDLE_SKIP", "1")
    ring = _ring(monkeypatch)
    ring.arrive(RID)
    ring.run(120)
    assert (_admit_pass(ring) - _told_pass(ring)) * DT >= 1.0


def test_pipeline_idle_reads_busy_when_the_ring_is_unknown():
    assert m.pipeline_idle(SimpleNamespace()) is False
    assert m.pipeline_idle(SimpleNamespace(mbs=[None], running_mbs=None)) is False
    idle = SimpleNamespace(mbs=[None, None], running_mbs=[None, _empty_batch()], chunked_req=None)
    assert m.pipeline_idle(idle) is True
    idle.running_mbs[0] = _busy_batch()
    assert m.pipeline_idle(idle) is False
