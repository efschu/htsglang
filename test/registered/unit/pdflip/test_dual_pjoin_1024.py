"""#DPJ (24.10.): two dual-layout P defects of 27B NVFP4 dual rc39a (b9cddabda1),
boot dkr27bnvfp4dualbar1drei10240856, P log ..._1024_085659.

A. TWIN BEHIND AN UNGRANTED SOURCE (p_twin_defer rule 7). 09:02:51 PP0 'P-KV PP0
   GRANT rid=pdflip-0-10 tokens=86016' (GRANT-BYPASS past the head pdflip-0-6, which
   waited for its own card grant), then '#TW TWIN-DEFER rid=pdflip-0-10 ...
   sources=['pdflip-0-6']'. The twin's follower pre-charge (held until its told
   leaves, Q-630) kept 0-6 short ('#1640 GRANT-INFEASIBLE ... waits for the return of
   the followers' pre-charge'); 0-6 waited for the twin, the twin for 0-6 -- until
   the Frist ('TWIN-RELEASE reason=deadline waited_s=120.00' 09:04:51) and 0-6's
   grant one second later ('after 5164 waits over 122.1 s'): ADMISSION-WEDGE
   P-KV-WAIT, 4 queued, 0 running.

B. #1180-W HOLD AGAINST PP0's pp0_admitted=False (scheduler._pdflip_defer_waiting_abort).
   09:09:36 PP0 'PDFLIP-PP-ABORT-VERDICT rid=pdflip-0-28 pp0_admitted=False ... every
   stage applies it at receipt (K14)'; PP2 'PDFLIP-PP-WAITING-ABORT held rid=pdflip-0-28'
   (a told of an earlier list stood there); the front's RESUME sent the next instance
   under the same rid; 09:09:46 PP2 '#996 SECOND CHUNKED MINT ... rid=pdflip-0-28' --
   group death.

Danger directions: (A) the twin held behind a source that computes nothing (the
specimen); a twin released while its source is granted and computing (lost share);
the flip form changed. (B) the stamped False held again (the specimen); an
unstamped / True abort applied at receipt (reopens #1180).
"""

from __future__ import annotations

import logging
import os
import types
import unittest.mock as mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.managers import scheduler as SC  # noqa: E402
from flliper.srt.pdflip import p_row_authority  # noqa: E402
from flliper.srt.pdflip import p_twin_defer as tw  # noqa: E402
from flliper.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

N = 8192
DUAL_ENV = {"FLLIPER_PDFLIP_DUAL_LAYOUT": "1", "FLLIPER_PDFLIP_GROUP": "P"}


# ---------------------------------------------------------------- A. twin rule 7


class _Req:
    def __init__(self, rid, ids, wait=False):
        self.rid = rid
        self.origin_input_ids = list(ids)
        self._dual_kv_wait = wait
        self.done = False

    def finished(self):
        return self.done


class _Sched:
    def __init__(self):
        self.ps = types.SimpleNamespace(pp_rank=0, pp_size=3, tp_size=1)
        self.waiting_queue = []
        self.chunked_req = None
        self.running_batch = None
        self.page_size = 1
        self.chunked_prefill_size = 1024


def _ids(shared, tail, salt):
    return list(range(shared)) + [10_000_000 + salt * 100_000 + i for i in range(tail)]


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()
    monkeypatch.setattr(tw, "_now", c)
    return c


def _arm(monkeypatch, dual):
    monkeypatch.setenv(tw.ENV, "1")
    monkeypatch.setenv(tw.ENV_MIN_TOKENS, str(N))
    monkeypatch.setenv(tw.ENV_WAIT_S, "120")
    monkeypatch.setenv(tw.ENV_SETTLE_MS, "500")
    monkeypatch.setenv("FLLIPER_PDFLIP_P_TWIN_CHAIN", "0")
    monkeypatch.setattr(tw, "_twin_anchor_armed", lambda: False)
    monkeypatch.setattr(tw, "_gain", lambda st, src, shared: True)
    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    if dual:
        monkeypatch.setenv("FLLIPER_PDFLIP_DUAL_LAYOUT", "1")
    else:
        monkeypatch.delenv("FLLIPER_PDFLIP_DUAL_LAYOUT", raising=False)


@pytest.fixture
def dual(monkeypatch):
    _arm(monkeypatch, True)


@pytest.fixture
def flip(monkeypatch):
    _arm(monkeypatch, False)


def _specimen(head_waits=True):
    """pdflip-0-6 (the head, 250854 tokens) and its fork twin pdflip-0-10 (82000,
    shared 18227) -- scaled: shared 9000 >= N."""
    s = _Sched()
    head = _Req("pdflip-0-6", _ids(9000, 4000, 6), wait=head_waits)
    twin = _Req("pdflip-0-10", _ids(9000, 2000, 10))
    s.waiting_queue = [head, twin]
    return s, head, twin


def test_specimen_twin_not_held_behind_a_grant_waiting_source(dual, clock, caplog):
    """RED on 1ecda02a58: TWIN-DEFER sources=['pdflip-0-6'] -> held to the Frist."""
    s, head, twin = _specimen(head_waits=True)
    with caplog.at_level(logging.INFO, logger=tw.logger.name):
        assert tw.intake_defer(s, twin) is False
    assert not tw.is_deferred(s, twin.rid)
    assert any("TWIN-SOURCE-UNGRANTED" in r.getMessage() and "at=intake" in r.getMessage()
               for r in caplog.records)


def test_held_twin_released_once_its_source_waits_for_its_grant(dual, clock, caplog):
    """A source granted at intake that goes back to a grant wait (P-PAUSE requeue,
    regrant) releases the held twin at once -- not after 120 s."""
    s, head, twin = _specimen(head_waits=False)
    assert tw.intake_defer(s, twin) is True
    head._dual_kv_wait = True
    clock.t += 1.0
    with caplog.at_level(logging.INFO, logger=tw.logger.name):
        out = tw.release_due(s, {twin.rid: twin})
    assert out == [(twin, False)]
    assert not tw.is_deferred(s, twin.rid)
    assert any(tw.REL_SOURCE_UNGRANTED in r.getMessage() for r in caplog.records)


def test_granted_source_still_holds_the_twin(dual, clock):
    """No lost prefix share: a granted, computing source keeps the hold."""
    s, head, twin = _specimen(head_waits=False)
    assert tw.intake_defer(s, twin) is True
    clock.t += 5.0
    assert tw.release_due(s, {twin.rid: twin}) == []
    assert tw.is_deferred(s, twin.rid)


def test_flip_form_unchanged(flip, clock):
    """No dual layout: _dual_kv_wait is never read, the hold as before."""
    s, head, twin = _specimen(head_waits=True)
    assert tw.intake_defer(s, twin) is True
    clock.t += 5.0
    assert tw.release_due(s, {twin.rid: twin}) == []


# ------------------------------------------------- B. follower #1180-W vs K14 stamp

RID = "pdflip-0-28"


class _Follower:
    def __init__(self):
        self.ps = types.SimpleNamespace(pp_rank=2, pp_size=3)
        self.waiting_queue = []
        self.chunked_req = None
        self._pdflip_store_told_armed = True
        self._pdflip_store_told = {}
        self.aborted = []

    def _abort_request_now(self, recv_req):
        if SC.Scheduler._pdflip_defer_waiting_abort(self, recv_req):
            return
        self.aborted.append(recv_req.rid)
        self.waiting_queue = [r for r in self.waiting_queue if not r.rid.startswith(recv_req.rid)]


def _freq(rid=RID):
    return types.SimpleNamespace(rid=rid, req_pool_idx=None, is_retracted=False)


@pytest.fixture
def dual_p_follower(monkeypatch):
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    with mock.patch.dict(os.environ, DUAL_ENV):
        yield


def _abort(stamp):
    a = SC.AbortReq(rid=RID)
    a.pdflip_pp0_admitted = stamp
    return a


def test_specimen_stamped_false_applied_at_receipt(dual_p_follower, caplog):
    """RED on 1ecda02a58: PP2 'PDFLIP-PP-WAITING-ABORT held rid=pdflip-0-28' although
    PP0 stamped pp0_admitted=False; a told of an earlier list stands here."""
    f = _Follower()
    f.waiting_queue = [_freq()]
    f._pdflip_store_told[RID] = 36352
    with caplog.at_level(logging.INFO, logger=SC.logger.name):
        f._abort_request_now(_abort(False))
    assert f.aborted == [RID] and f.waiting_queue == []
    assert not getattr(f, "_pdflip_pending_waiting_aborts", None)
    assert any("#DPJ-W" in r.getMessage() for r in caplog.records)


def test_no_second_instance_meets_a_zombie(dual_p_follower):
    """The death condition: after the RESUME instance arrives under the same rid,
    exactly ONE queued object carries the rid (no second chunked mint)."""
    f = _Follower()
    f.waiting_queue = [_freq()]
    f._pdflip_store_told[RID] = 36352
    f._abort_request_now(_abort(False))
    f.waiting_queue.append(_freq())
    assert sum(1 for r in f.waiting_queue if r.rid == RID) == 1


@pytest.mark.parametrize("stamp", [None, True])
def test_unstamped_or_admitted_keeps_the_hold(dual_p_follower, stamp):
    """PP0 admitted it (or no verdict rode the abort): #1180-W byte for byte."""
    f = _Follower()
    f.waiting_queue = [_freq()]
    f._pdflip_store_told[RID] = 36352
    f._abort_request_now(_abort(stamp))
    assert f.aborted == [] and sorted(f._pdflip_pending_waiting_aborts) == [RID]
