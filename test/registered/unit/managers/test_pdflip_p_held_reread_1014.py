"""P-HELD-REREAD-1014 (NF h17 38c6d40239, P 15:59:22-25, pdflip-1-11).

PP0: ``#1157 PREFETCH REAPED requested_pages=1475 hit_pages=1475
completed=52480 elapsed=2.49s budget=94.19s policy=timeout`` -- no timeout,
the read had ended (``PROBE-HOLD released reason=read-end``) short, because
sibling L3 fills recycled slots it was filling (``LATE-COMPLETE REFUSED
statuses=[3]``). ``#1416e PACED-TOLD told=52480`` went out at once, P
recomputed 41920 tokens the store held. Now PP0 re-reads before the told while
the record moves (cap FLLIPER_PDFLIP_P_HELD_SHORT_REREAD, default 2).
"""
from __future__ import annotations

import inspect
import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from flliper.srt.managers import pdflip_store_told as T  # noqa: E402
from flliper.srt.pdflip import short_read as SR  # noqa: E402


class _Tree:
    def __init__(self, have):
        self.have = have
        self.prefetch_loaded_tokens_by_reqid = {}

    def completed_prefetch_tokens(self, rid):
        return self.have


class _Sched:
    def __init__(self, verdict="issued"):
        self.calls = []
        self.verdict = verdict

    def _prefetch_kvcache(self, req, *a, **k):
        self.calls.append(req.rid)
        return self.verdict


def _req(rid="pdflip-1-11"):
    return types.SimpleNamespace(rid=rid)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(T.P_HELD_REREAD_ENV, raising=False)
    SR._HELD.clear()
    SR._MORE.clear()
    yield
    SR._HELD.clear()
    SR._MORE.clear()


def _reap_h17(rid="pdflip-1-11"):
    # the h17 reap: 1475 pages requested, all held, 52480 tokens landed
    SR.note_reap(rid, requested_pages=1475, hit_pages=1475, completed_tokens=52480, page_size=64)


def test_h17_short_read_with_held_span_is_reread_before_the_told():
    _reap_h17()
    s, req = _Sched(), _req()
    assert T._p_held_short_reread(s, _Tree(52480), req, req.rid) is True
    assert s.calls == ["pdflip-1-11"]


def test_no_growth_no_second_reread_then_the_told():
    _reap_h17()
    s, req, tree = _Sched(), _req(), _Tree(52480)
    assert T._p_held_short_reread(s, tree, req, req.rid)
    _reap_h17()   # the re-read ended short again, record unmoved
    assert T._p_held_short_reread(s, tree, req, req.rid) is False
    assert s.calls == ["pdflip-1-11"]


def test_growth_rereads_up_to_the_cap():
    s, req, tree = _Sched(), _req(), _Tree(52480)
    for have in (52480, 70000, 90000):
        _reap_h17()
        tree.have = have
        T._p_held_short_reread(s, tree, req, req.rid)
    assert len(s.calls) == 2, "default cap 2"


def test_full_read_or_store_short_is_not_reread():
    SR.note_reap("pdflip-1-11", requested_pages=1475, hit_pages=1475, completed_tokens=1475 * 64, page_size=64)
    s, req = _Sched(), _req()
    assert T._p_held_short_reread(s, _Tree(1475 * 64), req, req.rid) is False
    # the store holds no more than the read delivered: nothing to re-read
    SR.note_reap("pdflip-1-11", requested_pages=1475, hit_pages=820, completed_tokens=52480, page_size=64)
    assert T._p_held_short_reread(s, _Tree(52480), req, req.rid) is False
    assert s.calls == []


def test_switch_off(monkeypatch):
    monkeypatch.setenv(T.P_HELD_REREAD_ENV, "0")
    _reap_h17()
    s, req = _Sched(), _req()
    assert T._p_held_short_reread(s, _Tree(52480), req, req.rid) is False
    assert s.calls == []


def test_a_declined_reissue_publishes_the_told():
    _reap_h17()
    s, req = _Sched(verdict="declined:storage_disabled"), _req()
    assert T._p_held_short_reread(s, _Tree(52480), req, req.rid) is False
    assert "pdflip-1-11" not in SR._HELD


def test_both_pp0_publish_loops_reread_before_the_told():
    """PP0 decides alone: the re-read sits between the termination and the
    told in the plain and in the paced publish -- no follower path calls it."""
    for fn in (T.pp0_publish, T._pp0_publish_paced):
        src = inspect.getsource(fn)
        a = src.index("tree.check_prefetch_progress(rid)")
        b = src.index("_p_held_short_reread(scheduler, tree, req, rid)")
        c = src.index("_pp0_told_any(")
        assert a < b < c, fn.__name__
    for fn in (T._follower_absorb_impl, T._follower_admit):
        assert "_p_held_short_reread" not in inspect.getsource(fn)


def test_h18_store_short_of_the_span_but_more_than_the_read_is_reread():
    """h18 pdflip-0-2: hit 1440 of 1456 pages (the span is not whole), completed
    19008 -- the store still held 73152 tokens more: re-read."""
    SR.note_reap("pdflip-0-2", requested_pages=1456, hit_pages=1440, completed_tokens=19008, page_size=64)
    s, req = _Sched(), _req("pdflip-0-2")
    assert T._p_held_short_reread(s, _Tree(19008), req, req.rid) is True


def test_h18_pdflip_4_18_completed_zero_is_reread():
    SR.note_reap("pdflip-4-18", requested_pages=1061, hit_pages=1061, completed_tokens=0, page_size=64)
    s, req = _Sched(), _req("pdflip-4-18")
    assert T._p_held_short_reread(s, _Tree(0), req, req.rid) is True
