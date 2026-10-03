"""Q-690 TWIN SOURCE GONE (weg2.p_twin_defer rule 6, dual layout only).

27B NVFP4 dual fs10031727 (bc2bd121c0): 17:33:21 PP0 '#TW TWIN-DEFER
rid=weg2-0-45 ... sources=['weg2-0-42']'; 17:33:22 the front took 0-42 back
('WEG2-INTAKE-STALL gate=seats', /abort_request, 'Q-580 TOLD-FORGET
why=intake_stall') -- 0-42 neither finished nor in flight on P, yet 0-45 was
held until the Frist ('TWIN-RELEASE reason=deadline waited_s=120.00
pending_sources=1'): P idle 100 s, ADMISSION-WEDGE.

Drives the shipped #1400 protocol (weg2_store_told intake -> pp0_publish ->
follower_absorb -> admission) on a three-stage double. Danger directions: the
twin stays held behind a source that left (the specimen), a twin is released
while its source is still on P (lost prefix share), the flip form changes
(no dual layout: Frist path byte-equal), and the ranks disagree on the told.
"""

import logging
from types import SimpleNamespace

import pytest

from sglang.srt.managers import weg2_store_told as m
from sglang.srt.weg2 import p_twin_defer as tw

N = 8192
DUAL_ENV = {"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "P"}


class _Tree:
    def __init__(self):
        self.progress_left = {}
        self.prefetch_loaded_tokens_by_reqid = {}
        self._prefetch_completed_tokens = {}
        self._pending = {}

    def register(self, rid, completed, flips=0):
        self.progress_left[rid] = flips
        self._pending[rid] = completed

    def check_prefetch_progress(self, rid):
        if rid not in self.progress_left:
            return True
        if self.progress_left[rid] > 0:
            self.progress_left[rid] -= 1
            return False
        del self.progress_left[rid]
        done = self._pending.pop(rid)
        self.prefetch_loaded_tokens_by_reqid[rid] = done
        self._prefetch_completed_tokens[rid] = done
        return True

    def completed_prefetch_tokens(self, rid):
        return self._prefetch_completed_tokens.get(rid)

    def pop_prefetch_loaded_tokens(self, rid):
        return self.prefetch_loaded_tokens_by_reqid.pop(rid, 0)


class _Sched:
    def __init__(self, pp_rank, pp_size=3):
        self.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size, tp_size=1)
        self.enable_hicache_storage = True
        self.tree_cache = _Tree()
        self.waiting_queue = []
        self.chunked_req = None
        self.running_batch = None
        self.pp_flip_counters = None
        self.page_size = 64
        self.chunked_prefill_size = 16384
        self.registered = []
        self.local = {}

    def _prefetch_kvcache(self, req, rematch=True, limit_tokens=None):
        self.registered.append((req.rid, limit_tokens))
        head, span = self.local.get(req.rid, (0, 0))
        if limit_tokens is not None:
            head = min(head, int(limit_tokens))
            span = max(0, min(span, int(limit_tokens) - head))
        req._prefetch_registered_prefix_len = head
        req.prefix_indices = [0] * head
        req.host_hit_length = 0
        if span <= 0 and head > 0:
            return "declined:too_short"
        self.tree_cache.register(req.rid, completed=span)
        return "issued"


class _Req:
    def __init__(self, rid, ids, extra_key=None):
        self.rid = rid
        self.origin_input_ids = list(ids)
        self.extra_key = extra_key
        self.prefetch_deferred = None
        self._dual_grant_untold = None
        self.done = False

    def finished(self):
        return self.done


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
    monkeypatch.delenv(m.ENV_ARMED, raising=False)
    monkeypatch.setenv(tw.ENV, "1")
    monkeypatch.setenv(tw.ENV_MIN_TOKENS, str(N))
    monkeypatch.setenv(tw.ENV_WAIT_S, "120")
    monkeypatch.setenv(tw.ENV_SETTLE_MS, "500")
    for k, v in DUAL_ENV.items():
        if dual:
            monkeypatch.setenv(k, v)
        else:
            monkeypatch.delenv(k, raising=False)


@pytest.fixture
def dual(monkeypatch):
    _arm(monkeypatch, True)
    yield


@pytest.fixture
def flip(monkeypatch):
    """The flip form: twin defer armed, no dual layout (group P set, as on P)."""
    _arm(monkeypatch, False)
    monkeypatch.setenv("SGLANG_WEG2_GROUP", "P")
    yield


def _note():
    return lambda kind, rid: None


def _intake(s, req):
    v = m.intake(s, req, lambda g: None)
    s.waiting_queue.append(req)
    return v


def _specimen(s=None):
    """weg2-0-42 (the source) queued on P, weg2-0-45 its twin, held."""
    s = s or _Sched(0)
    assert m.armed(s)
    a = _Req("weg2-0-42", _ids(60000, 300, 1))
    s.local[a.rid] = (0, 4095)
    _intake(s, a)
    b = _Req("weg2-0-45", _ids(60000, 450, 2))
    s.local[b.rid] = (0, 512)
    assert _intake(s, b) == tw.VERDICT_DEFERRED
    return s, a, b


def _run_pp0(s, clock, passes, dt=0.2):
    wire = []
    for _ in range(passes):
        clock.t += dt
        wire += [w for w in m.pp0_publish(s, []) if isinstance(w, m.Weg2StoreTold)]
    return wire


def _abort_queued(s, r):
    """The front takes the source back (WEG2-INTAKE-STALL, /abort_request): it
    leaves the waiting queue, NOT finished."""
    s.waiting_queue.remove(r)
    assert not r.finished()


def _abort_chunked(s, r):
    s.waiting_queue.remove(r)
    s.chunked_req = r
    return lambda: setattr(s, "chunked_req", None)


# --------------------------------------------------------------------------
# (a) dual: the source left unfinished -> released next pass, source_gone
# --------------------------------------------------------------------------


def test_dual_source_aborted_from_queue_releases_twin_next_pass(dual, clock, caplog):
    caplog.set_level(logging.INFO, logger=tw.__name__)
    s, a, b = _specimen()
    assert _run_pp0(s, clock, 3) and tw.is_deferred(s, b.rid)
    _abort_queued(s, a)
    wire = _run_pp0(s, clock, 1)
    assert not tw.is_deferred(s, b.rid)
    assert (b.rid, None) in s.registered
    told = [w for w in wire if w.rid == b.rid]
    # an ordinary request: plain told, the span record, no head added
    assert len(told) == 1 and type(told[0]) is m.Weg2StoreTold
    assert told[0].told == 512
    assert not tw.take_pp0_twin(s, b.rid)
    rel = [r.getMessage() for r in caplog.records if "TWIN-RELEASE" in r.getMessage()]
    assert len(rel) == 1 and "reason=source_gone" in rel[0]
    assert "gone=['weg2-0-42']" in rel[0] and "waited_s=0.80" in rel[0]
    assert getattr(s, tw._ATTR).n_source_gone == 1
    assert getattr(s, tw._ATTR).n_deadline == 0


def test_dual_source_admitted_then_aborted_releases_twin(dual, clock):
    s, a, b = _specimen()
    clear = _abort_chunked(s, a)  # admitted: the chunked one, in flight
    _run_pp0(s, clock, 3)
    assert tw.is_deferred(s, b.rid)  # (b) in flight -> held
    clear()  # aborted mid-prefill, never finished
    wire = _run_pp0(s, clock, 1)
    assert [w.rid for w in wire if w.rid == b.rid] == [b.rid]
    assert not tw.is_deferred(s, b.rid)


def test_dual_one_of_two_sources_gone_waits_for_the_other(dual, clock):
    s2, a, b2 = _specimen()
    s2.local[b2.rid] = (57344, 0)
    # a second source in flight (the intake rule makes a twin of a queued
    # twin wait itself; the double puts the second one in the wait directly)
    c = _Req("weg2-0-43", _ids(60000, 600, 3))
    s2.running_batch = SimpleNamespace(reqs=[c])
    w = getattr(s2, tw._ATTR).waits[b2.rid]
    w.sources.append(c)
    w.shared_by[c.rid] = 60000
    _abort_queued(s2, a)
    _run_pp0(s2, clock, 4)
    assert tw.is_deferred(s2, b2.rid)
    assert [x.rid for x in w.sources] == [c.rid]  # a dropped, c still held
    c.done = True
    wire = _run_pp0(s2, clock, 6)
    told = [x for x in wire if x.rid == b2.rid]
    # the remaining sibling published: released as a TWIN (absolute told)
    assert len(told) == 1 and type(told[0]) is m.Weg2StoreToldTwin
    assert told[0].told == 57344


def test_dual_source_gone_all_ranks_admit_the_same_plain_told(dual, clock):
    """RAENGE-NIE-UNEINS: PP0 decides, the followers absorb the plain told
    (no twin flag, as on the Frist path) and admit on the same number."""
    ranks = [_Sched(r) for r in range(3)]
    a, b = [], []
    for r in ranks:
        assert m.armed(r)
        ai = _Req("weg2-0-42", _ids(60000, 300, 1))
        bi = _Req("weg2-0-45", _ids(60000, 450, 2))
        r.local[ai.rid] = (0, 4095)
        r.local[bi.rid] = (0, 512)
        _intake(r, ai)
        _intake(r, bi)
        a.append(ai)
        b.append(bi)
    assert tw.is_deferred(ranks[0], b[0].rid)
    wire_to = {1: [], 2: []}
    first = [None, None, None]
    for p in range(8):
        if p == 2:
            for i in range(3):
                _abort_queued(ranks[i], a[i])
        clock.t += 0.2
        out = [w for w in m.pp0_publish(ranks[0], []) if isinstance(w, m.Weg2StoreTold)]
        for w in out:
            if w.rid == b[0].rid:
                assert type(w) is m.Weg2StoreTold and w.told == 512
        m.follower_absorb(ranks[2], wire_to[2])
        wire_to[2] = list(out)
        m.follower_absorb(ranks[1], out)
        for i, r in enumerate(ranks):
            if first[i] is None and b[i].rid in r._weg2_store_told:
                if m.admission(r, b[i], _note()) is not None:
                    first[i] = p
    assert None not in first
    assert first[0] == 2 and first[0] <= first[1] <= first[2] <= 3
    for i in (1, 2):
        assert not tw.take_follower_twin(ranks[i], b[i].rid)


# --------------------------------------------------------------------------
# (b) dual: the source still on P -> the twin stays held
# --------------------------------------------------------------------------


@pytest.mark.parametrize("where", ["queued", "chunked", "running", "post_wake_settle"])
def test_dual_source_still_on_p_keeps_twin_held(dual, clock, where):
    s, a, b = _specimen()
    if where != "queued":
        s.waiting_queue.remove(a)
        if where == "chunked":
            s.chunked_req = a
        elif where == "running":
            s.running_batch = SimpleNamespace(reqs=[a])
        else:
            s.weg2_post_wake_settle = [a]
    wire = _run_pp0(s, clock, 50)  # 10 s, well below the Frist
    assert all(w.rid != b.rid for w in wire)
    assert tw.is_deferred(s, b.rid)
    assert all(rid != b.rid for rid, _ in s.registered)
    assert getattr(getattr(s, tw._ATTR), "n_source_gone", 0) == 0


def test_dual_finished_source_is_not_gone_it_publishes(dual, clock):
    s, a, b = _specimen()
    s.local[b.rid] = (57344, 0)
    a.done = True
    s.waiting_queue.remove(a)
    wire = _run_pp0(s, clock, 6)
    told = [w for w in wire if w.rid == b.rid]
    assert len(told) == 1 and type(told[0]) is m.Weg2StoreToldTwin
    assert getattr(getattr(s, tw._ATTR), "n_source_gone", 0) == 0


# --------------------------------------------------------------------------
# (c) no dual layout: unchanged -- the Frist path
# --------------------------------------------------------------------------


def test_flip_form_aborted_source_holds_until_the_frist(flip, clock, caplog, monkeypatch):
    monkeypatch.setenv(tw.ENV_WAIT_S, "5")
    caplog.set_level(logging.INFO, logger=tw.__name__)
    s, a, b = _specimen()
    assert getattr(getattr(s, tw._ATTR), "dual", False) is False
    _abort_queued(s, a)
    wire = _run_pp0(s, clock, 20)  # 4 s: still held, as before Q-690
    assert all(w.rid != b.rid for w in wire)
    assert tw.is_deferred(s, b.rid)
    wire = _run_pp0(s, clock, 10)
    told = [w for w in wire if w.rid == b.rid]
    assert len(told) == 1 and type(told[0]) is m.Weg2StoreTold and told[0].told == 512
    rel = [r.getMessage() for r in caplog.records if "TWIN-RELEASE" in r.getMessage()]
    assert len(rel) == 1 and "reason=deadline" in rel[0] and "pending_sources=1" in rel[0]
    assert "source_gone" not in caplog.text


def test_dual_gate_needs_layout_and_group_p():
    assert tw._dual_p_env(DUAL_ENV)
    assert not tw._dual_p_env({"SGLANG_WEG2_GROUP": "P"})
    assert not tw._dual_p_env({"SGLANG_WEG2_DUAL_LAYOUT": "1", "SGLANG_WEG2_GROUP": "D"})
    assert not tw._dual_p_env({})
