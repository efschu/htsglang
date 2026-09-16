"""#1400: PP0's store verdict TOLD on the request wire (weg2_store_told).

Drives the protocol on a three-stage ring double with the shipped module:
intake -> publish -> absorb -> admission, and the two named refusals. The
danger direction is a rank ADMITTING without a told verdict (the W27 width
split): every admission assertion here fails if `admission` is bypassed.
"""

import types
import os
from types import SimpleNamespace

import pytest

from sglang.srt.managers import weg2_store_told as m


class _Tree:
    """Tree-cache double: a prefetch terminates after `flips` progress calls."""

    def __init__(self):
        self.progress_left = {}
        self.prefetch_loaded_tokens_by_reqid = {}
        self.progress_calls = 0

    def register(self, rid, loaded, flips=0, completed=None):
        """`completed` = the host-tree prefix after termination (matched +
        loaded); defaults to `loaded` (empty tree before the read)."""
        self.progress_left[rid] = flips
        self._loaded_when_done = getattr(self, "_loaded_when_done", {})
        self._loaded_when_done[rid] = loaded
        self._completed_when_done = getattr(self, "_completed_when_done", {})
        self._completed_when_done[rid] = loaded if completed is None else completed
        self._completed = getattr(self, "_completed", {})

    def check_prefetch_progress(self, rid):
        self.progress_calls += 1
        if rid not in self.progress_left:
            return True
        if self.progress_left[rid] > 0:
            self.progress_left[rid] -= 1
            return False
        del self.progress_left[rid]
        self.prefetch_loaded_tokens_by_reqid[rid] = self._loaded_when_done.pop(rid)
        self._completed[rid] = self._completed_when_done.pop(rid)
        return True

    def completed_prefetch_tokens(self, rid):
        return getattr(self, "_completed", {}).get(rid)

    def pop_prefetch_loaded_tokens(self, rid):
        return self.prefetch_loaded_tokens_by_reqid.pop(rid, 0)


class _Sched:
    def __init__(self, pp_rank, pp_size=3, tp_size=1, storage=True):
        self.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=pp_size, tp_size=tp_size)
        self.enable_hicache_storage = storage
        self.tree_cache = _Tree()
        self.waiting_queue = []
        self.pp_flip_counters = None
        self.registered = []  # (rid, limit_tokens)

    def _prefetch_kvcache(self, req, rematch=True, limit_tokens=None):
        self.registered.append((req.rid, limit_tokens))
        return "issued"


def _req(rid):
    return SimpleNamespace(rid=rid, prefetch_deferred=None)


def _skips():
    seen = []
    return seen, (lambda kind, rid: seen.append((kind, str(rid))))


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv(m.ENV_ARMED, raising=False)
    yield


def test_armed_only_on_carrierless_pp_tp1_with_storage(monkeypatch):
    assert m.armed(_Sched(0)) is True
    assert m.armed(_Sched(1)) is True
    assert m.armed(_Sched(0, pp_size=1)) is False
    assert m.armed(_Sched(0, tp_size=3)) is False
    assert m.armed(_Sched(0, storage=False)) is False
    monkeypatch.setenv(m.ENV_ARMED, "0")
    assert m.armed(_Sched(0)) is False


def test_armed_is_resolved_once_per_scheduler(monkeypatch):
    s = _Sched(0)
    assert m.armed(s) is True
    monkeypatch.setenv(m.ENV_ARMED, "0")
    assert m.armed(s) is True  # cached: a flip mid-boot would split the ranks


def test_pp0_registers_holds_and_admits_only_after_publish():
    s = _Sched(0)
    assert m.armed(s)
    r = _req("aaaa0001")
    s.waiting_queue.append(r)
    gates = []
    assert m.intake(s, r, gates.append) == "issued"
    assert s.registered == [("aaaa0001", None)]
    assert "aaaa0001" in s._weg2_store_held
    skips, note = _skips()
    # Prefetch still running: nothing published, admission skips by name.
    s.tree_cache.register("aaaa0001", loaded=4096, flips=1)
    assert m.pp0_publish(s, ["x"]) == ["x"]
    assert m.admission(s, r, note) is None
    assert skips == [(m.SKIP_TOLD_PENDING, "aaaa0001")]
    # Terminated: the told rides the wire and admission returns it once.
    wire = m.pp0_publish(s, ["x"])
    assert [type(w).__name__ for w in wire] == ["str", "Weg2StoreTold"]
    assert (wire[1].rid, wire[1].told) == ("aaaa0001", 4096)
    assert "aaaa0001" not in s._weg2_store_held
    assert m.admission(s, r, note) == 4096
    assert s.tree_cache.prefetch_loaded_tokens_by_reqid == {}


def test_pp0_publishes_zero_for_a_request_that_registered_nothing():
    s = _Sched(0)
    m.armed(s)
    r = _req("bbbb0001")
    s.waiting_queue.append(r)
    m.intake(s, r, lambda k: None)
    wire = m.pp0_publish(s, [])
    assert len(wire) == 1 and wire[0].told == 0
    assert m.admission(s, r, lambda k, rid: None) == 0


def test_pp0_does_not_publish_a_deferred_or_dequeued_request():
    s = _Sched(0)
    m.armed(s)
    deferred = _req("cccc0001")
    deferred.prefetch_deferred = "shortfall"
    gone = _req("cccc0002")
    s.waiting_queue.append(deferred)
    m.intake(s, deferred, lambda k: None)
    m.intake(s, gone, lambda k: None)
    assert m.pp0_publish(s, []) == []
    assert "cccc0001" in s._weg2_store_held  # still held: PP0 will re-issue
    assert "cccc0002" not in s._weg2_store_held  # left the queue: forgotten


def test_follower_holds_then_registers_exactly_the_told_span():
    s = _Sched(1)
    m.armed(s)
    r = _req("dddd0001")
    s.waiting_queue.append(r)
    gates = []
    assert m.intake(s, r, gates.append) == "declined:weg2_held"
    assert gates == [m.GATE_HELD]
    assert s.registered == []  # nothing registered at intake
    skips, note = _skips()
    assert m.admission(s, r, note) is None  # no verdict yet: skip by name
    assert skips == [(m.SKIP_TOLD_PENDING, "dddd0001")]
    rest = m.follower_absorb(s, ["req-obj", m.Weg2StoreTold("dddd0001", 2048)])
    assert rest == ["req-obj"]  # the told left the dispatch list
    assert s.registered == [("dddd0001", 2048)]
    # Own read completes inside the admission wait; the told is the credit.
    s.tree_cache.register("dddd0001", loaded=2048, flips=3)
    assert m.admission(s, r, note) == 2048
    assert s.tree_cache.progress_calls >= 4


def test_follower_told_zero_registers_nothing_and_admits_zero():
    s = _Sched(2)
    m.armed(s)
    r = _req("eeee0001")
    m.intake(s, r, lambda k: None)
    m.follower_absorb(s, [m.Weg2StoreTold("eeee0001", 0)])
    assert s.registered == []
    assert m.admission(s, r, lambda k, rid: None) == 0


def test_follower_refuses_by_name_when_own_read_differs_from_told():
    s = _Sched(1)
    m.armed(s)
    r = _req("ffff0001")
    m.intake(s, r, lambda k: None)
    m.follower_absorb(s, [m.Weg2StoreTold("ffff0001", 4096)])
    s.tree_cache.register("ffff0001", loaded=2048, flips=0)
    with pytest.raises(m.Weg2StoreToldMismatch, match="told=4096 own_prefix=2048"):
        m.admission(s, r, lambda k, rid: None)


def test_follower_refuses_when_own_read_never_terminates(monkeypatch):
    s = _Sched(1)
    m.armed(s)
    r = _req("0a0a0001")
    m.intake(s, r, lambda k: None)
    m.follower_absorb(s, [m.Weg2StoreTold("0a0a0001", 4096)])
    s.tree_cache.register("0a0a0001", loaded=4096, flips=10**9)
    monkeypatch.setattr(m, "WAIT_CAP_S", 0.01)
    with pytest.raises(m.Weg2StoreToldMismatch, match="WAIT EXCEEDED"):
        m.admission(s, r, lambda k, rid: None)


def test_told_arriving_before_the_request_registers_at_intake():
    s = _Sched(1)
    m.armed(s)
    m.follower_absorb(s, [m.Weg2StoreTold("1b1b0001", 1024)])
    r = _req("1b1b0001")
    assert m.intake(s, r, lambda k: None) == "issued"
    assert s.registered == [("1b1b0001", 1024)]


def test_ring_order_pp0_then_pp1_then_pp2_same_told():
    """The wire object forwarded unchanged by PP1 reaches PP2 with the same
    told; every stage ends with the same credit for the rid."""
    pp0, pp1, pp2 = _Sched(0), _Sched(1), _Sched(2)
    for s in (pp0, pp1, pp2):
        m.armed(s)
    rid = "2c2c0001"
    reqs = {s: _req(rid) for s in (pp0, pp1, pp2)}
    pp0.waiting_queue.append(reqs[pp0])
    m.intake(pp0, reqs[pp0], lambda k: None)
    m.intake(pp1, reqs[pp1], lambda k: None)
    m.intake(pp2, reqs[pp2], lambda k: None)
    pp0.tree_cache.register(rid, loaded=8192, flips=0)
    wire = m.pp0_publish(pp0, [])
    rest1 = m.follower_absorb(pp1, list(wire))  # PP1 forwards `wire` as-is
    rest2 = m.follower_absorb(pp2, list(wire))
    assert rest1 == [] and rest2 == []
    pp1.tree_cache.register(rid, loaded=8192, flips=1)
    pp2.tree_cache.register(rid, loaded=8192, flips=2)
    credits = [m.admission(s, reqs[s], lambda k, r: None) for s in (pp0, pp1, pp2)]
    assert credits == [8192, 8192, 8192]
    assert pp1.registered == [(rid, 8192)] and pp2.registered == [(rid, 8192)]


def test_intake_partition_carries_the_held_term():
    from sglang.srt.mem_cache.match_refusal_census import PREFETCH_INTAKE_PARTITION

    assert m.GATE_HELD in PREFETCH_INTAKE_PARTITION


def test_told_is_the_completed_prefix_not_the_loaded_increment():
    """xsn119: PP0 completed 4095 with 64 already in its host tree (loaded
    4031); the followers must register [0, 4095) -- anchor at 4094 inside."""
    pp0, pp1 = _Sched(0), _Sched(1)
    m.armed(pp0)
    m.armed(pp1)
    rid = "3d3d0001"
    r0, r1 = _req(rid), _req(rid)
    pp0.waiting_queue.append(r0)
    m.intake(pp0, r0, lambda k: None)
    m.intake(pp1, r1, lambda k: None)
    pp0.tree_cache.register(rid, loaded=4031, flips=0, completed=4095)
    wire = m.pp0_publish(pp0, [])
    assert wire[0].told == 4095
    m.follower_absorb(pp1, list(wire))
    assert pp1.registered == [(rid, 4095)]
    pp1.tree_cache.register(rid, loaded=4095, flips=0, completed=4095)
    # PP0's credit is its own increment; the prefix check passed on both.
    assert m.admission(pp0, r0, lambda k, x: None) == 4031
    assert m.admission(pp1, r1, lambda k, x: None) == 4095


def test_follower_with_short_prefix_refuses_even_if_it_loaded_something():
    s = _Sched(2)
    m.armed(s)
    r = _req("4e4e0001")
    m.intake(s, r, lambda k: None)
    m.follower_absorb(s, [m.Weg2StoreTold("4e4e0001", 4095)])
    s.tree_cache.register("4e4e0001", loaded=4031, flips=0, completed=4031)
    with pytest.raises(m.Weg2StoreToldMismatch, match="told=4095 own_prefix=4031"):
        m.admission(s, r, lambda k, rid: None)


def test_follower_limit_is_one_token_more_under_bigram_keys():
    """xsn120: told=4095 keys; a bigram tree (MTP draft) needs 4096 tokens to
    make 4095 keys so the anchor at key 4094 is inside the registered range."""
    plain = SimpleNamespace(is_eagle=False)
    bigram = SimpleNamespace(is_eagle=True)
    assert m.follower_limit_tokens(plain, 4095) == 4095
    assert m.follower_limit_tokens(bigram, 4095) == 4096
    s = _Sched(1)
    m.armed(s)
    s.tree_cache.is_eagle = True
    r = _req("5f5f0001")
    m.intake(s, r, lambda k: None)
    m.follower_absorb(s, [m.Weg2StoreTold("5f5f0001", 4095)])
    assert s.registered == [("5f5f0001", 4096)]


def test_follower_that_already_holds_the_told_span_is_satisfied_without_a_read():
    """Boot xsn141: PP0 read to told=3615, the followers held >= 3616 locally,
    their prefetch was 'declined:too_short' (nothing to fetch) and admission
    raised the mismatch on own_prefix=0. Holding the span IS satisfying it."""
    s = _Sched(1)
    m.armed(s)
    s._prefetch_kvcache = lambda req, rematch=True, limit_tokens=None: "declined:too_short"
    r = SimpleNamespace(rid="eeee0001", prefetch_deferred=None,
                        prefix_indices=list(range(3000)), host_hit_length=700)
    s.waiting_queue.append(r)
    gates = []
    assert m.intake(s, r, gates.append) == "declined:weg2_held"
    m.follower_absorb(s, [m.Weg2StoreTold("eeee0001", 3615)])
    skips, note = _skips()
    assert m.admission(s, r, note) == 0  # admitted at told, no credit, no wait
    assert skips == []
    # xsn155: the same with 'declined:store_absent' (probe started past told)
    s._prefetch_kvcache = lambda req, rematch=True, limit_tokens=None: "declined:store_absent"
    r3 = SimpleNamespace(rid="eeee0003", prefetch_deferred=None,
                         prefix_indices=list(range(94206)), host_hit_length=0)
    s.waiting_queue.append(r3)
    m.intake(s, r3, gates.append)
    m.follower_absorb(s, [m.Weg2StoreTold("eeee0003", 53246)])
    assert m.admission(s, r3, note) == 0
    # a follower that holds LESS than told still refuses by name
    r2 = SimpleNamespace(rid="eeee0002", prefetch_deferred=None,
                         prefix_indices=list(range(100)), host_hit_length=0)
    s.waiting_queue.append(r2)
    m.intake(s, r2, gates.append)
    m.follower_absorb(s, [m.Weg2StoreTold("eeee0002", 3615)])
    s.tree_cache.register("eeee0002", loaded=0, flips=0)
    with pytest.raises(m.Weg2StoreToldMismatch):
        m.admission(s, r2, note)


# ---- #1416: told is anchor-clamped like the followers' presence probe ------


def test_1416_told_is_clamped_to_the_anchored_presence():
    from sglang.srt.managers import weg2_store_told as wst

    class _CC:
        page_size = 1

        def store_presence_pages(self, ids, last_hash, prefix_keys=None):
            assert len(ids) == 53247
            return 4095

    class _Req:
        origin_input_ids = list(range(60000))
        rid = "abcdef0123456789"

    sched = types.SimpleNamespace(cache_controller=_CC())
    assert wst._anchor_clamp(sched, _Req(), 53247) == 4095
    sched.cache_controller.store_presence_pages = lambda ids, lh, prefix_keys=None: 99999
    assert wst._anchor_clamp(sched, _Req(), 53247) == 53247, "never above completed"
    assert wst._anchor_clamp(types.SimpleNamespace(), _Req(), 7) == 7, "no probe: no clamp"
    assert wst._anchor_clamp(sched, _Req(), 0) == 0


def test_1416b_pp0_record_follows_the_clamp(monkeypatch):
    """xsn169: a clamped told must also clamp PP0's own completed record, or
    PP0's admission raises the mismatch against itself."""
    from sglang.srt.managers import weg2_store_told as wst

    class _Tree:
        _prefetch_completed_tokens = {"r1": 4095}

        def check_prefetch_progress(self, rid):
            return True

        def completed_prefetch_tokens(self, rid):
            return self._prefetch_completed_tokens.get(rid)

    class _Req:
        rid = "r1"
        origin_input_ids = list(range(8000))
        prefetch_deferred = None

    class _CC:
        page_size = 1

        def store_presence_pages(self, ids, lh, prefix_keys=None):
            return 0

    req = _Req()
    sched = types.SimpleNamespace(
        tree_cache=_Tree(), cache_controller=_CC(), waiting_queue=[req],
        _weg2_store_held={"r1": req}, _weg2_store_told={},
    )
    out = wst.pp0_publish(sched, [])
    assert [o.told for o in out] == [0]
    assert sched.tree_cache._prefetch_completed_tokens["r1"] == 0
    assert sched._weg2_store_told["r1"] == 0


def test_1419_told_caps_the_radix_match_on_every_rank():
    from sglang.srt.managers import weg2_store_told as wst
    from sglang.srt.managers.schedule_batch import _weg2_cap_key_limit

    class _Tree:
        def check_prefetch_progress(self, rid):
            return True

        def completed_prefetch_tokens(self, rid):
            return 4095

        def pop_prefetch_loaded_tokens(self, rid):
            return 0

    req = types.SimpleNamespace(rid="r9")
    sched = types.SimpleNamespace(
        tree_cache=_Tree(), _weg2_store_told={"r9": 4095},
        _weg2_store_told_satisfied={}, ps=types.SimpleNamespace(pp_rank=1),
    )
    assert wst.admission(sched, req, lambda *a: None) == 0
    assert req._weg2_prefix_cap == 4095
    assert _weg2_cap_key_limit(req, None) == 4095
    assert _weg2_cap_key_limit(req, 100000) == 4095
    assert _weg2_cap_key_limit(req, 10) == 10
    assert _weg2_cap_key_limit(types.SimpleNamespace(), 77) == 77
