"""#248 PARK-DEMOTE, anchor pool (29.09.): the pool keeps ANCHOR_TAIL_KEYS=2
candidate pages of which only the one holding the end anchor exists.
rc12z30j 27B: every anchor demote read 'pages=2 written=1 l3_pages=1 absent=1'
(242/242), NF 286/643 -- the other candidate has no slot, it is no lost page.
The span is complete once the existing page is on disk and nothing is busy;
before, it was never complete and every tick asked the store again."""

import types

from sglang.srt.weg2 import handoff_pending as hp
from sglang.srt.weg2 import park_demote


class _Backend:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = 0

    def arena_copy_to_disk(self, arena, stems):
        self.calls += 1
        return dict(self.answers.pop(0) if self.answers else {"on_disk": 1, "missing": 1, "absent": 1})


def _pool(role):
    p = types.SimpleNamespace(arena=object())
    setattr(p, hp.ROLE_ATTR, role)
    return p


def _spans(monkeypatch, stems):
    monkeypatch.setattr(hp, "rid_spans", lambda pool: [(hp.ROLE_PARK, "weg2-154-176", list(stems))])


def test_anchor_span_with_one_missing_candidate_is_complete(monkeypatch):
    _spans(monkeypatch, ["a", "b"])
    be = _Backend([{"written": 1, "absent": 1, "missing": 1, "bytes": 8}])
    state = {}
    recs = park_demote.demote_once(be, [_pool("anchor")], state=state)
    assert recs and recs[0]["missing"] == 1 and recs[0]["busy"] == 0
    assert any(state.values())
    park_demote.demote_once(be, [_pool("anchor")], state=state)
    park_demote.demote_once(be, [_pool("anchor")], state=state)
    # the pools are new objects each pass: the key follows id(pool), so pin one
    pool = _pool("anchor")
    state = {}
    be = _Backend([{"written": 1, "absent": 1, "missing": 1}])
    park_demote.demote_once(be, [pool], state=state)
    park_demote.demote_once(be, [pool], state=state)
    assert be.calls == 1  # complete after the first pass: not asked again


def test_anchor_span_with_a_busy_page_stays_open(monkeypatch):
    _spans(monkeypatch, ["a", "b"])
    pool = _pool("anchor")
    be = _Backend([{"written": 1, "absent": 1, "busy": 1}, {"on_disk": 1, "written": 1}])
    state = {}
    park_demote.demote_once(be, [pool], state=state)
    assert not any(state.values())  # a busy page may still have to be copied
    park_demote.demote_once(be, [pool], state=state)
    assert be.calls == 2 and any(state.values())


def test_kv_span_still_needs_every_page(monkeypatch):
    _spans(monkeypatch, ["a", "b"])
    pool = _pool("kv")
    be = _Backend([{"written": 1, "absent": 1, "missing": 1}])
    state = {}
    park_demote.demote_once(be, [pool], state=state)
    assert not any(state.values())
