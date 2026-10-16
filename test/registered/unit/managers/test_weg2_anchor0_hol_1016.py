"""anchor0 review Auflage 1 (wf_a0c826e9-e01, B3): the held re-read (hint, wake
probe) must not hold OTHER rids (head of line) in either REAL PP0 publish loop
(pp0_publish, _pp0_publish_paced); the held rid stays held, the rid behind it
gets its told in the same pass. Taken over from the review test
test_rev_anchor0_hol_1015.py (its mutant MA1 'continue' -> 'break' goes red)."""
from __future__ import annotations

import os
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.managers import weg2_store_told as T  # noqa: E402
from sglang.srt.weg2 import short_read as SR  # noqa: E402

PAGE = 64


class _Tree:
    def __init__(self, have):
        self.have = have
        self.prefetch_loaded_tokens_by_reqid = {}

    def completed_prefetch_tokens(self, rid):
        return self.have[rid]

    def check_prefetch_progress(self, rid):
        return True


def _sched(paced, dormant=True):
    a = types.SimpleNamespace(rid="weg2-126-651")
    b = types.SimpleNamespace(rid="weg2-126-652")
    s = types.SimpleNamespace(
        _weg2_store_held={a.rid: a, b.rid: b}, _weg2_store_told={}, waiting_queue=[a, b],
        tree_cache=_Tree({a.rid: 38656, b.rid: 20000}), weg2_dormant=dormant, page_size=PAGE,
        _weg2_told_paced_on=paced, _weg2_told_pacing={}, calls=[])
    s._prefetch_kvcache = lambda req, *x, **k: (s.calls.append(req.rid), "issued")[1]
    return s, a, b


@pytest.fixture(autouse=True)
def _stubs(monkeypatch):
    for d in (SR._HELD, SR._MORE, SR._HINT):
        d.clear()
    tw = T._twin
    monkeypatch.setattr(tw, "tick", lambda s: None)
    monkeypatch.setattr(tw, "release_due", lambda s, q: [])
    monkeypatch.setattr(tw, "is_deferred", lambda s, r: False)
    monkeypatch.setattr(tw, "take_pp0_twin", lambda s, r: False)
    monkeypatch.setattr(tw, "registered_head", lambda req: 0, raising=False)
    monkeypatch.setattr(T, "_minifwd_hold", lambda s, q: None)
    monkeypatch.setattr(T, "_pp0_told_any", lambda s, t, req, rid, twin: (t.have[rid], False))
    monkeypatch.setattr(T, "_ple_admit_at_told", lambda *a, **k: None)
    monkeypatch.setattr(T, "_pp0_keys_digest", lambda req: "")
    monkeypatch.setattr(T, "_pp0_fork", lambda s, req, rid: -1)
    monkeypatch.setattr(T, "_pacing", lambda s: s._weg2_told_pacing, raising=False)
    monkeypatch.setattr(T, "_pace_intake_t", lambda s: {}, raising=False)
    monkeypatch.setattr(T, "_with_fork", lambda obj, fork: obj)
    yield
    for d in (SR._HELD, SR._MORE, SR._HINT):
        d.clear()


@pytest.mark.parametrize("paced", [False, True])
def test_hint_wait_holds_only_its_own_rid(paced):
    s, a, b = _sched(paced)
    SR.note_reap(a.rid, requested_pages=700, hit_pages=604, completed_tokens=38656, page_size=PAGE)
    SR.note_hint(a.rid, 45568)
    out = T.pp0_publish(s, [])
    rids = [getattr(o, "rid", None) for o in out]
    assert b.rid in rids, "the rid behind the hint-WAIT got no told (head of line)"
    assert a.rid not in rids and a.rid in s._weg2_store_held
    assert s.calls == []  # dormant: nothing issued


@pytest.mark.parametrize("paced", [False, True])
def test_awake_reread_holds_only_its_own_rid(paced):
    s, a, b = _sched(paced, dormant=False)
    SR.note_reap(a.rid, requested_pages=700, hit_pages=604, completed_tokens=38656, page_size=PAGE)
    SR.note_hint(a.rid, 45568)
    out = T.pp0_publish(s, [])
    rids = [getattr(o, "rid", None) for o in out]
    assert b.rid in rids and a.rid not in rids
    assert s.calls == [a.rid]


@pytest.mark.parametrize("paced", [False, True])
def test_wake_probe_reread_holds_only_its_own_rid(paced, monkeypatch):
    """weg2-5-52 form: a read issued during the flip, awake the store holds more."""
    s, a, b = _sched(paced, dormant=False)
    setattr(a, T.WAKE_PROBE_ATTR, True)
    SR.note_reap(a.rid, requested_pages=700, hit_pages=604, completed_tokens=38656, page_size=PAGE)
    monkeypatch.setattr(T, "_wake_store_depth", lambda sch, req, have: 45568)
    out = T.pp0_publish(s, [])
    rids = [getattr(o, "rid", None) for o in out]
    assert b.rid in rids and a.rid not in rids
    assert s.calls == [a.rid]
