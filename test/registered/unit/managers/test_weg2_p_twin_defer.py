"""TW: fork twins on group P wait for their sibling's prefix (weg2.p_twin_defer).

Drives the shipped #1400 protocol (weg2_store_told: intake -> pp0_publish ->
follower_absorb -> admission) on a three-stage double with the twin switch on
and off. Danger directions: a rank admitting the twin before PP0's told is on
the wire (rank split), a twin that never gets released (wedge), and a non-twin
that is delayed or changed (the switch must touch only twins).
"""

from types import SimpleNamespace

import pytest

from sglang.srt.managers import weg2_store_told as m
from sglang.srt.weg2 import p_twin_defer as tw

N = 8192


class _Tree:
    """Tree double: a registered read terminates after `flips` progress calls
    and leaves `completed` (span-relative, like the real insert) behind."""

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
        self.registered = []  # (rid, limit_tokens)
        #: what a registration finds: rid -> (local head, span completed)
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


@pytest.fixture
def on(monkeypatch):
    monkeypatch.delenv(m.ENV_ARMED, raising=False)
    monkeypatch.setenv(tw.ENV, "1")
    monkeypatch.setenv(tw.ENV_MIN_TOKENS, str(N))
    monkeypatch.setenv(tw.ENV_WAIT_S, "120")
    monkeypatch.setenv(tw.ENV_SETTLE_MS, "500")
    yield


@pytest.fixture
def off(monkeypatch):
    monkeypatch.delenv(m.ENV_ARMED, raising=False)
    monkeypatch.delenv(tw.ENV, raising=False)
    yield


def _skip():
    seen = []
    return seen, (lambda kind, rid: seen.append((kind, str(rid))))


def _intake(s, req):
    """The scheduler's `_add_request_to_queue` order: intake, then append."""
    v = m.intake(s, req, lambda g: None)
    s.waiting_queue.append(req)
    return v


def _pp0_with_sibling(shared=60000):
    s = _Sched(0)
    assert m.armed(s)
    a = _Req("weg2-22-63", _ids(shared, 300, 1))
    s.local[a.rid] = (0, 4095)
    _intake(s, a)
    b = _Req("weg2-23-64", _ids(shared, 450, 2))
    return s, a, b


def _run_pp0(s, clock, passes, dt=0.2):
    wire = []
    for _ in range(passes):
        clock.t += dt
        wire += [w for w in m.pp0_publish(s, []) if isinstance(w, m.Weg2StoreTold)]
    return wire


# --------------------------------------------------------------------------
# switch off = the pre-TW path
# --------------------------------------------------------------------------


def test_switch_off_registers_at_intake_and_publishes_plain_told(off, clock):
    s, a, b = _pp0_with_sibling()
    s.local[b.rid] = (20000, 512)
    assert _intake(s, b) == "issued"
    assert (b.rid, None) in s.registered
    wire = _run_pp0(s, clock, 1)
    told_b = [w for w in wire if w.rid == b.rid]
    assert len(told_b) == 1
    assert type(told_b[0]) is m.Weg2StoreTold
    # the pre-TW number: the span-relative record, no head added
    assert told_b[0].told == 512
    assert getattr(s, tw._ATTR, None) is None


def test_switch_off_never_touches_follower_admission(off):
    f = _Sched(1)
    m.armed(f)
    r = _Req("weg2-1-1", _ids(N + 10, 5, 3))
    f.waiting_queue.append(r)
    m.intake(f, r, lambda g: None)
    f.local[r.rid] = (100, 400)
    m.follower_absorb(f, [m.Weg2StoreTold(rid=r.rid, told=400)])
    _, note = _skip()
    # own = span record 300 (400 capped at limit minus head) != 400 -> the
    # unchanged #1400 refusal, no head is added without the twin flag.
    with pytest.raises(m.Weg2StoreToldMismatch):
        m.admission(f, r, note)


# --------------------------------------------------------------------------
# twin: deferred, released after the sibling finished, then hits
# --------------------------------------------------------------------------


def test_twin_is_held_without_store_read_while_sibling_in_flight(on, clock):
    s, a, b = _pp0_with_sibling()
    assert _intake(s, b) == tw.VERDICT_DEFERRED
    assert all(rid != b.rid for rid, _ in s.registered)
    assert b.rid in s._weg2_store_held
    wire = _run_pp0(s, clock, 20)
    assert [w.rid for w in wire] == [a.rid]  # only the sibling's told
    seen, note = _skip()
    assert m.admission(s, b, note) is None
    assert seen == [(m.SKIP_TOLD_PENDING, b.rid)]


def test_twin_released_after_sibling_finished_and_hits_the_prefix(on, clock):
    s, a, b = _pp0_with_sibling(shared=60000)
    _intake(s, b)
    _run_pp0(s, clock, 3)
    # the sibling finished on PP0: its rows are the twin's device head now
    a.done = True
    s.waiting_queue.remove(a)
    s.local[b.rid] = (57344, 0)  # deepest anchor <= shared, tail misses store
    # settle: pp_size passes AND 500 ms -- not before
    wire = _run_pp0(s, clock, 2, dt=0.1)
    assert wire == [] and all(rid != b.rid for rid, _ in s.registered)
    wire = _run_pp0(s, clock, 4, dt=0.2)
    assert (b.rid, None) in s.registered
    told = [w for w in wire if w.rid == b.rid]
    assert len(told) == 1 and type(told[0]) is m.Weg2StoreToldTwin
    assert told[0].told == 57344  # absolute: head + span
    # PP0 admits at the twin told; the #1419 cap now covers the sibling's rows
    assert m.admission(s, b, _skip()[1]) == 0
    assert b._weg2_prefix_cap == 57344


def test_twin_told_adds_head_to_a_read_span(on, clock):
    s, a, b = _pp0_with_sibling(shared=60000)
    _intake(s, b)
    a.done = True
    s.waiting_queue.remove(a)
    s.local[b.rid] = (57344, 2048)
    wire = _run_pp0(s, clock, 6)
    told = [w for w in wire if w.rid == b.rid]
    assert told and told[0].told == 57344 + 2048
    assert s.tree_cache.completed_prefetch_tokens(b.rid) == 57344 + 2048
    assert m.admission(s, b, _skip()[1]) == 2048  # credit = the loaded span


# --------------------------------------------------------------------------
# the Frist
# --------------------------------------------------------------------------


def test_frist_releases_as_ordinary_request(on, clock, monkeypatch):
    monkeypatch.setenv(tw.ENV_WAIT_S, "5")
    s, a, b = _pp0_with_sibling()
    _intake(s, b)
    s.local[b.rid] = (20000, 512)
    wire = _run_pp0(s, clock, 20, dt=0.2)  # 4 s: still held
    assert all(w.rid != b.rid for w in wire)
    wire = _run_pp0(s, clock, 10, dt=0.2)  # past 5 s, sibling never finished
    told = [w for w in wire if w.rid == b.rid]
    assert len(told) == 1 and type(told[0]) is m.Weg2StoreTold
    assert told[0].told == 512  # the pre-TW number, no head
    assert not tw.is_deferred(s, b.rid)


def test_dequeued_twin_is_forgotten(on, clock):
    s, a, b = _pp0_with_sibling()
    _intake(s, b)
    s.waiting_queue.remove(b)  # aborted
    _run_pp0(s, clock, 2)
    assert not tw.is_deferred(s, b.rid)
    assert b.rid not in s._weg2_store_held


# --------------------------------------------------------------------------
# non-twins are untouched
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ids,extra",
    [
        (_ids(N - 1, 5000, 7), None),  # shares less than N
        (_ids(60000, 450, 7), "salt"),  # same ids, other cache key
        (list(range(5000)), None),  # shorter than N
    ],
)
def test_non_twin_registers_at_intake(on, clock, ids, extra):
    s, a, _ = _pp0_with_sibling()
    c = _Req("weg2-9-9", ids, extra_key=extra)
    assert _intake(s, c) != tw.VERDICT_DEFERRED
    assert (c.rid, None) in s.registered


def test_finished_sibling_is_no_source(on, clock):
    s, a, b = _pp0_with_sibling()
    a.done = True
    assert _intake(s, b) != tw.VERDICT_DEFERRED


def test_non_twin_behind_a_held_twin_is_published_and_admitted(on, clock):
    s, a, b = _pp0_with_sibling()
    _intake(s, b)
    c = _Req("weg2-9-10", list(range(20_000_000, 20_003_000)))
    s.local[c.rid] = (0, 1024)
    _intake(s, c)
    wire = _run_pp0(s, clock, 1)
    assert [w.rid for w in wire if w.rid == c.rid] == [c.rid]
    assert type([w for w in wire if w.rid == c.rid][0]) is m.Weg2StoreTold
    seen, note = _skip()
    assert m.admission(s, b, note) is None  # the twin skips ...
    assert m.admission(s, c, note) == 1024  # ... the one behind it is admitted


# --------------------------------------------------------------------------
# rank agreement: one decision, carried by the told on the request wire
# --------------------------------------------------------------------------


def test_all_stages_skip_then_admit_the_twin_on_the_same_told(on, clock):
    ranks = [_Sched(r) for r in range(3)]
    for r in ranks:
        assert m.armed(r)
    shared = 60000
    a = [_Req("weg2-22-63", _ids(shared, 300, 1)) for _ in ranks]
    b = [_Req("weg2-23-64", _ids(shared, 450, 2)) for _ in ranks]
    for i, r in enumerate(ranks):
        r.local[a[i].rid] = (0, 4095)
        _intake(r, a[i])
        _intake(r, b[i])
    # the followers also hold the sibling's rows once it finished
    ranks[1].local[b[1].rid] = (57344, 0)  # satisfied locally
    ranks[2].local[b[2].rid] = (40960, 20000)  # evicted part, reads the rest
    ranks[0].local[b[0].rid] = (57344, 0)
    wire_to = {1: [], 2: []}
    decisions = []
    for p in range(14):
        if p == 5:
            for i in range(3):
                a[i].done = True
                ranks[i].waiting_queue.remove(a[i])
        clock.t += 0.2
        out = m.pp0_publish(ranks[0], [])
        told_objs = [w for w in out if isinstance(w, m.Weg2StoreTold)]
        # PP1 gets list m this pass, PP2 one pass later (the chain)
        m.follower_absorb(ranks[2], wire_to[2])
        wire_to[2] = list(told_objs)
        m.follower_absorb(ranks[1], told_objs)
        row = []
        for i, r in enumerate(ranks):
            row.append(m.admission(r, b[i], _skip()[1]) is not None
                       if b[i].rid in r._weg2_store_told else False)
        decisions.append(row)
    # no follower admits before PP0 has put the told on the wire, and once
    # a rank admits, it admits at PP0's absolute told
    first = [next(p for p, row in enumerate(decisions) if row[i]) for i in range(3)]
    assert first[0] <= first[1] <= first[2]
    assert first[0] >= 5 + 2  # never before the sibling finished + settle
    for i in range(3):
        assert b[i]._weg2_prefix_cap == 57344


def test_follower_twin_mismatch_still_refuses_by_name(on, clock):
    f = _Sched(1)
    m.armed(f)
    r = _Req("weg2-1-2", _ids(N + 10, 5, 3))
    f.waiting_queue.append(r)
    m.intake(f, r, lambda g: None)
    f.local[r.rid] = (1000, 100)  # can reproduce only 1100 of 5000
    m.follower_absorb(f, [m.Weg2StoreToldTwin(rid=r.rid, told=5000)])
    with pytest.raises(m.Weg2StoreToldMismatch):
        m.admission(f, r, _skip()[1])


def test_shared_prefix_len_and_is_twin():
    x = _ids(9000, 10, 1)
    y = _ids(9000, 20, 2)
    assert tw.shared_prefix_len(x, y) == 9000
    import array

    assert tw.shared_prefix_len(array.array("i", x), y) == 9000
    ra, rb = _Req("a", x), _Req("b", y)
    assert tw.is_twin(ra, rb, 8192) and not tw.is_twin(ra, rb, 9001)


# --------------------------------------------------------------------------
# #1416e paced form: the twin rides the read-ahead, the Admit follows
# --------------------------------------------------------------------------


def test_paced_twin_read_ahead_then_admit_on_every_rank(on, clock, monkeypatch):
    monkeypatch.setenv(m.ENV_PACED, "1")
    monkeypatch.setattr(m, "_clock", clock)
    ranks = [_Sched(0), _Sched(1)]
    for r in ranks:
        assert m.armed(r)
    shared = 60000
    a = [_Req("weg2-22-63", _ids(shared, 300, 1)) for _ in ranks]
    b = [_Req("weg2-23-64", _ids(shared, 450, 2)) for _ in ranks]
    for i, r in enumerate(ranks):
        _intake(r, a[i])  # sibling: nothing to read (told 0, single-phase)
        assert _intake(r, b[i]) == (tw.VERDICT_DEFERRED if i == 0 else "declined:weg2_held")
    for r in ranks:
        r.local[b[0].rid] = (57344, 0)
    kinds, admitted = [], {}
    for p in range(20):
        if p == 2:
            for i in range(2):
                a[i].done = True
                ranks[i].waiting_queue.remove(a[i])
        clock.t += 0.2
        out = [w for w in m.pp0_publish(ranks[0], [])
               if isinstance(w, (m.Weg2StoreTold, m.Weg2StoreAdmit)) and w.rid == b[0].rid]
        kinds += [(p, type(w).__name__, getattr(w, "paced", None), w.told) for w in out]
        m.follower_absorb(ranks[1], out)
        for i, r in enumerate(ranks):
            if i not in admitted and b[i].rid in r._weg2_store_told:
                assert m.admission(r, b[i], _skip()[1]) is not None
                admitted[i] = p
    assert [k[1:] for k in kinds] == [
        ("Weg2StoreToldTwin", True, 57344),
        ("Weg2StoreAdmit", None, 57344),
    ]
    assert kinds[1][0] > kinds[0][0]  # the Admit follows the read-ahead
    assert admitted[0] == admitted[1] == kinds[1][0]
    assert b[0]._weg2_prefix_cap == b[1]._weg2_prefix_cap == 57344
