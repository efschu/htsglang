"""TW-WAKE (#294, NF y5a 30.09.): fork twins that reach group P while it SLEEPS.

Metal (y5a, P ...0110f8e132_0930_174039): the dmatrix stage pdflip-52-75..80, six
identical 65602-token prompts, reached P while it slept (``PLE-PREFETCH admit
rid=pdflip-52-76 queued source=hint dormant=1`` 17:56:08). The #1400 intake --
where TW asks -- ran for each, then ``_add_request_to_queue`` parked it in
``pdflip_dormant_hold``. TW's in-flight set did not include the dormant hold, so
no request had a sibling: not one ``#TW`` line, six told-0 registrations, six
full prefills in a row (``#cached-token: 0`` x 6, 17:56:10-17:57:18, 68 s of P).

Now: (b) the dormant hold is in flight for TW, a deferred twin is no source
(77..80 wait behind 75 only and are released with 76 in one pass), and (a) a
twin deferred in the hold stays deferred until the wake instead of being
forgotten by the publish (``release_due`` sees the dormant hold / settle as
queued). (c) Rank agreement is unchanged: only PP0 decides, the followers
follow the told on the request wire. Group P's dormant intake is the #1443
path; the #248 read-at-wake is group D only (``park_l3.defer_hold_read``).
"""
from __future__ import annotations

import importlib.util
import logging
import os

import pytest

from flliper.srt.managers import pdflip_store_told as m
from flliper.srt.pdflip import p_twin_defer as tw

_spec = importlib.util.spec_from_file_location(
    "_t_tw_base", os.path.join(os.path.dirname(__file__), "test_pdflip_p_twin_defer.py"))
B = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(B)

LEN = 65602
END = ((LEN - 1) // 64) * 64          # 65600: the source's END ANCHOR (y5a 'PDFLIP END-ANCHOR anchor=65600')
RIDS = ["pdflip-52-%d" % i for i in range(75, 81)]


@pytest.fixture
def clock(monkeypatch):
    c = B._Clock()
    monkeypatch.setattr(tw, "_now", c)
    return c


@pytest.fixture
def on(monkeypatch):
    monkeypatch.delenv(m.ENV_ARMED, raising=False)
    monkeypatch.setenv(tw.ENV, "1")
    monkeypatch.setenv(tw.ENV_MIN_TOKENS, "8192")
    monkeypatch.setenv(tw.ENV_WAIT_S, "120")
    monkeypatch.setenv(tw.ENV_SETTLE_MS, "500")
    yield


def _dormant(s):
    s.pdflip_dormant = True
    s.pdflip_dormant_hold = []
    return s


def _intake_dormant(s, req):
    """``_add_request_to_queue`` while P sleeps: the #1400 intake, then the
    dormant hold (not the waiting queue)."""
    v = m.intake(s, req, lambda g: None)
    s.pdflip_dormant_hold.append(req)
    return v


def _wake(s):
    """``_pdflip_release_dormant_hold``: the held requests join the queue in order."""
    s.waiting_queue.extend(s.pdflip_dormant_hold)
    s.pdflip_dormant_hold = []
    s.pdflip_dormant = False


def _burst(s):
    reqs = [B._Req(r, list(range(LEN))) for r in RIDS]
    for r in reqs:
        s.local.setdefault(r.rid, (0, 0))
    return reqs


def test_y5a_six_identical_prompts_while_p_sleeps(on, clock, caplog):
    caplog.set_level(logging.INFO)
    s = _dormant(B._Sched(0))
    assert m.armed(s)
    reqs = _burst(s)
    verdicts = [_intake_dormant(s, r) for r in reqs]
    assert verdicts[0] != tw.VERDICT_DEFERRED
    assert verdicts[1:] == [tw.VERDICT_DEFERRED] * 5, "y5a: every one after 75 is a twin"
    assert [rid for rid, _ in s.registered] == [RIDS[0]], "only 75 reads the store"
    st = getattr(s, tw._ATTR)
    assert all([tw._rid(x) for x in st.waits[r].sources] == [RIDS[0]] for r in RIDS[1:]), \
        "77..80 wait behind 75 only, not behind the deferred 76"
    line = [x for x in caplog.messages if "#TW TWIN-DEFER rid=pdflip-52-76" in x][0]
    assert "sources=['pdflip-52-75']" in line and "p_dormant=1" in line
    assert "sources_in_hold=['pdflip-52-75']" in line


def test_the_hold_keeps_them_deferred_the_wake_releases_them_together(on, clock):
    s = _dormant(B._Sched(0))
    assert m.armed(s)
    reqs = _burst(s)
    for r in reqs:
        _intake_dormant(s, r)
    B._run_pp0(s, clock, 5)                        # passes while P sleeps (idle loop)
    assert all(tw.is_deferred(s, r) for r in RIDS[1:]), "not forgotten while in the dormant hold"
    _wake(s)
    B._run_pp0(s, clock, 5)                        # 75 prefills (4 chunks)
    assert all(tw.is_deferred(s, r) for r in RIDS[1:])
    reqs[0].done = True                            # 75 finished, its END ANCHOR at 65600
    s.waiting_queue.remove(reqs[0])
    for r in RIDS[1:]:
        s.local[r] = (END, 0)
    wire = []
    for _ in range(8):
        clock.t += 0.2
        wire.append([w for w in m.pp0_publish(s, []) if isinstance(w, m.PdFlipStoreTold)])
    published = [(i, w.rid, type(w), w.told) for i, ws in enumerate(wire) for w in ws]
    assert sorted(r for _i, r, _t, _v in published) == RIDS[1:]
    assert len({i for i, *_ in published}) == 1, "the five twins go out in ONE pass"
    assert all(t is m.PdFlipStoreToldTwin and v == END for _i, _r, t, v in published)
    for r in reqs[1:]:
        assert m.admission(s, r, B._skip()[1]) == 0 and r._pdflip_prefix_cap == END


def test_followers_never_decide_and_admit_at_pp0s_told(on, clock):
    """(c) PP0 authoritative: the followers hold every request until PP0's told
    arrives on the wire (they never ask TW), then admit at the same absolute
    told -- no rank admits earlier, none at another depth."""
    ranks = [_dormant(B._Sched(r)) for r in range(3)]
    for r in ranks:
        assert m.armed(r)
    reqs = [_burst(r) for r in ranks]
    for i, r in enumerate(ranks):
        for q in reqs[i]:
            _intake_dormant(r, q)
    assert getattr(ranks[1], tw._ATTR, None) in (None, False) or not getattr(ranks[1], tw._ATTR).waits
    for r in ranks:
        _wake(r)
    wire_to = {1: [], 2: []}
    first = {}
    for p in range(16):
        if p == 4:
            for i, r in enumerate(ranks):
                reqs[i][0].done = True
                r.waiting_queue.remove(reqs[i][0])
                for q in RIDS[1:]:
                    r.local[q] = (END, 0)
        clock.t += 0.2
        out = [w for w in m.pp0_publish(ranks[0], []) if isinstance(w, m.PdFlipStoreTold)]
        m.follower_absorb(ranks[2], wire_to[2])
        wire_to[2] = list(out)
        m.follower_absorb(ranks[1], out)
        for i, r in enumerate(ranks):
            for q in reqs[i][1:]:
                if (i, q.rid) not in first and q.rid in r._pdflip_store_told:
                    if m.admission(r, q, B._skip()[1]) is not None:
                        first[(i, q.rid)] = p
    for rid in RIDS[1:]:
        f = [first[(i, rid)] for i in range(3)]
        assert f[0] <= f[1] <= f[2] and f[0] >= 4 + 2
    for i in range(3):
        for q in reqs[i][1:]:
            assert q._pdflip_prefix_cap == END


def test_group_p_dormant_intake_is_not_the_248_read_at_wake(monkeypatch):
    """#248 (read at the wake) is group D's; group P registers at the dormant
    intake -- the place TW asks."""
    from types import SimpleNamespace

    from flliper.srt.pdflip import park_l3

    monkeypatch.setenv("FLLIPER_PDFLIP_GROUP", "P")
    monkeypatch.setattr(park_l3, "enabled", lambda: True)
    assert park_l3.defer_hold_read(SimpleNamespace(), SimpleNamespace(rid="pdflip-52-76")) is False


def test_awake_path_unchanged_a_single_twin_still_waits(on, clock):
    s, a, b = B._pp0_with_sibling()
    assert B._intake(s, b) == tw.VERDICT_DEFERRED
