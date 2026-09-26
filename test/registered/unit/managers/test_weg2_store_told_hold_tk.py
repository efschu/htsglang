"""TK (26.09.), follow-up to path 4: the dormant hold's re-read (#1456) and
the post-wake settle (#1471) on the carrierless P form (tp_size 1, so no
group MIN -- every rank decides for itself) together with #1400's told.

Drives the SHIPPED scheduler methods (``_weg2_hold_refetch``,
``_weg2_refetch_one``, ``_weg2_release_dormant_hold``,
``_weg2_post_wake_settle_tick``) on a PP0 and a follower double, with
weg2_store_told's intake / publish / absorb / admission between them.

The question: does every follower register exactly once, on PP0's FINAL told
and keys_digest, or can it register on an older state (read-ahead, paced
admission, its own re-read)?

The tree double keeps the one geometry that matters: a registration reads
[head, limit) where head = device prefix + what earlier reads of this rid
already put on the host, and the record counts the span it retained (the
insert is rooted at last_host_node).
"""

from types import SimpleNamespace

import pytest

from sglang.srt.managers import weg2_store_told as m
from sglang.srt.managers.scheduler import Scheduler


class _Rec(int):
    """PrefetchOutcome-like: int = loaded, .materialized = matched + loaded."""

    def __new__(cls, loaded, materialized):
        o = int.__new__(cls, loaded)
        o.materialized = materialized
        return o


class _HoldTree:
    def __init__(self, extent):
        self.is_eagle = True
        self.extent = extent  # keys the shared store holds for this prompt NOW
        self.host = {}  # rid -> keys on the host from earlier reads
        self.ongoing_prefetch = {}
        self.prefetch_loaded_tokens_by_reqid = {}
        self._prefetch_completed_tokens = {}

    def register(self, rid, head, limit_raw):
        end = max(head, min(limit_raw - 1, self.extent))
        self.ongoing_prefetch[rid] = end - head

    def check_prefetch_progress(self, rid):
        if rid in self.ongoing_prefetch:
            span = self.ongoing_prefetch.pop(rid)
            self.host[rid] = self.host.get(rid, 0) + span
            self._prefetch_completed_tokens[rid] = span
            self.prefetch_loaded_tokens_by_reqid[rid] = _Rec(span, span)
        return True

    def completed_prefetch_tokens(self, rid):
        return self._prefetch_completed_tokens.get(rid)

    def pop_prefetch_loaded_tokens(self, rid):
        return int(self.prefetch_loaded_tokens_by_reqid.pop(rid, 0) or 0)


class _Rank:
    _weg2_hold_refetch = Scheduler._weg2_hold_refetch
    _weg2_refetch_one = Scheduler._weg2_refetch_one
    _weg2_release_dormant_hold = Scheduler._weg2_release_dormant_hold
    _weg2_post_wake_settle_tick = Scheduler._weg2_post_wake_settle_tick
    _weg2_group_min_flags = Scheduler._weg2_group_min_flags
    _weg2_note_store_shortfall = Scheduler._weg2_note_store_shortfall
    _weg2_drain_prefetch_revokes = Scheduler._weg2_drain_prefetch_revokes
    WEG2_POST_WAKE_SETTLE_S = Scheduler.WEG2_POST_WAKE_SETTLE_S
    WEG2_TAIL_RECOMPUTE_TOKENS = Scheduler.WEG2_TAIL_RECOMPUTE_TOKENS

    def __init__(self, pp_rank, extent, n_ids=100000):
        self.ps = SimpleNamespace(pp_rank=pp_rank, pp_size=3, tp_size=1)
        self.enable_hicache_storage = True
        self.tree_cache = _HoldTree(extent)
        self.waiting_queue = []
        self.pp_flip_counters = None
        self._weg2_store_told = {}
        self._weg2_store_held = {}
        self._weg2_store_told_armed = True
        self.weg2_dormant = True
        self.weg2_dormant_hold = []
        self.n_ids = n_ids
        self.registered = []  # (rid, limit_tokens, head)

    def _prefetch_kvcache(self, req, rematch=True, limit_tokens=None):
        from sglang.srt.weg2 import handoff
        from sglang.srt.weg2.handoff_keys import resolve_chain

        head = self.tree_cache.host.get(req.rid, 0)
        req._prefetch_registered_prefix_len = head
        req.prefix_indices, req.host_hit_length = [], head
        req.used_chain = resolve_chain(req, handoff.read)
        match_end = self.n_ids - 1
        if limit_tokens is not None:
            match_end = min(match_end, int(limit_tokens))
        self.registered.append((req.rid, limit_tokens, head))
        req._prefetch_span_tokens = max(0, match_end - head)
        if match_end - 1 <= head:
            return "declined:too_short"
        self.tree_cache.register(req.rid, head, match_end)
        return "issued"

    # the intake site's dormant branch (scheduler._add_request_to_queue)
    def add(self, req):
        m.intake(self, req, lambda g: None)
        if self.weg2_dormant:
            self.weg2_dormant_hold.append(req)
        else:
            self.waiting_queue.append(req)

    def wake(self):
        self.weg2_dormant = False
        return self._weg2_release_dormant_hold()


def _req(rid, n_ids=100000):
    return SimpleNamespace(rid=rid, prefetch_deferred=None, origin_input_ids=list(range(n_ids)),
                           full_untruncated_fill_ids=list(range(n_ids)), extra_key=None)


def _tick_2s(*reqs):
    for r in reqs:
        r._1456_last = 0.0  # the 2-s re-read timer has run out


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    from sglang.srt.weg2 import handoff

    for k in (m.ENV_ARMED, "SGLANG_WEG2_TOLD_ABSOLUTE", "SGLANG_WEG2_TOLD_PACED",
              "SGLANG_WEG2_P_TWIN_DEFER", "SGLANG_HICACHE_ARENA_DIR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv(m.ENV_TREE_KEY, "1")  # the agent-boot form: told > 0, absolute
    monkeypatch.setattr(m, "_anchor_clamp", lambda s, r, told: int(told))
    files = {}
    monkeypatch.setattr(handoff, "read", lambda rid: files.get(rid))
    yield files


def test_hold_reread_on_pp0_then_release_follower_registers_once_on_the_final_told(_env):
    """PP0's hold read answers zero (P's write-through has not landed),
    the #1456 re-read tops it up to 98304 and a hand-off file appears in
    between. Nothing is published in the hold; after the wake the follower
    registers ONCE, with PP0's final told and PP0's final key source."""
    files = _env
    p0, f = _Rank(0, extent=0, n_ids=98306), _Rank(1, extent=98304, n_ids=98306)
    r0, r1 = _req("weg2-11-1", 98306), _req("weg2-11-1", 98306)
    p0.add(r0)
    f.add(r1)
    assert f.registered == [], "a follower registers nothing before PP0's told"
    assert m.pp0_publish(p0, []) == []
    files["weg2-11-1"] = {"page_keys": ["P%d" % i for i in range(64)]}
    p0.tree_cache.extent = 98304
    _tick_2s(r0, r1)
    assert p0._weg2_hold_refetch() == 1
    assert f._weg2_hold_refetch() == 0 and f.registered == []
    assert m.pp0_publish(p0, []) == [], "still held: nothing on the wire"
    assert p0.wake() == 1 and f.wake() == 1
    wire = m.pp0_publish(p0, [])
    assert [(w.told, w.absolute) for w in wire] == [(98304, True)]
    assert wire[0].keys_digest.endswith(":64")
    m.follower_absorb(f, list(wire))
    assert f.registered == [("weg2-11-1", 98305, 0)]
    assert f.tree_cache.ongoing_prefetch  # the told read is in flight
    assert r1.used_chain == r0.used_chain == files["weg2-11-1"]["page_keys"]
    assert m.admission(p0, r0, lambda *a: None) is not None
    assert m.admission(f, r1, lambda *a: None) == 98304


def test_pp0_parked_at_the_wake_publishes_after_the_settle(_env):
    """No re-read fitted into the flip: PP0's zero read parks at the wake
    (#1471), the follower (nothing registered) releases. PP0's settle tick
    re-reads; the told goes out only when the settle released it."""
    p0, f = _Rank(0, extent=0, n_ids=98306), _Rank(1, extent=98304, n_ids=98306)
    r0, r1 = _req("weg2-11-2", 98306), _req("weg2-11-2", 98306)
    p0.add(r0)
    f.add(r1)
    p0.tree_cache.extent = 98304
    assert p0.wake() == 0 and p0.weg2_post_wake_settle == [r0]
    assert f.wake() == 1 and f.waiting_queue == [r1]
    assert m.pp0_publish(p0, []) == []
    _tick_2s(r0)
    p0._weg2_post_wake_settle_tick()  # re-reads
    assert m.pp0_publish(p0, []) == []
    p0._weg2_post_wake_settle_tick()  # read complete -> queued
    assert p0.waiting_queue == [r0]
    wire = m.pp0_publish(p0, [])
    assert [w.told for w in wire] == [98304]
    m.follower_absorb(f, list(wire))
    assert [x[1] for x in f.registered] == [98305]
    assert m.admission(f, r1, lambda *a: None) == 98304
    assert m.admission(p0, r0, lambda *a: None) is not None


def test_paced_read_ahead_leaves_only_after_the_release(_env, monkeypatch):
    p0, f = _Rank(0, extent=0, n_ids=98306), _Rank(1, extent=98304, n_ids=98306)
    p0._weg2_told_paced_on = f._weg2_told_paced_on = True
    r0, r1 = _req("weg2-11-3", 98306), _req("weg2-11-3", 98306)
    p0.add(r0)
    f.add(r1)
    p0.tree_cache.extent = 98304
    _tick_2s(r0)
    p0._weg2_hold_refetch()
    assert m.pp0_publish(p0, []) == [], "no read-ahead from the hold"
    p0.wake()
    f.wake()
    wire = m.pp0_publish(p0, [])
    assert [(w.told, w.paced) for w in wire] == [(98304, True)]
    m.follower_absorb(f, list(wire))
    assert [x[1] for x in f.registered] == [98305]


def test_follower_still_held_when_the_told_arrives_rereads_only_the_told_span(_env):
    """The ordering the wire normally prevents (the Resume rides ahead of the
    told) but a lagging or refused follower wake (W114 votes locally on
    tp_size 1) allows: the follower absorbs PP0's told while its request is
    still in the dormant hold, its read answers zero (revoked below the
    threshold, #1479), and its own #1456 re-read re-registers WITHOUT the told
    limit -- it reads past told (the store gained more since) and the
    admission raises Weg2StoreToldMismatch. A follower's re-read must be the
    told span, like its first read."""
    p0, f = _Rank(0, extent=98304), _Rank(1, extent=0)
    r0, r1 = _req("weg2-11-4"), _req("weg2-11-4")
    p0.add(r0)
    f.add(r1)
    p0.wake()
    wire = m.pp0_publish(p0, [])
    assert [w.told for w in wire] == [98304]
    m.follower_absorb(f, list(wire))  # follower still dormant, r1 in its hold
    assert f.tree_cache.check_prefetch_progress("weg2-11-4")  # zero answer
    f.tree_cache.extent = 120000
    _tick_2s(r1)
    f._weg2_hold_refetch()
    assert [x[1] for x in f.registered] == [98305, 98305], "re-read keeps the told limit"
    f.wake()
    assert m.admission(f, r1, lambda *a: None) is not None


def test_follower_without_a_told_never_rereads_on_its_own(_env):
    """A stale short record on a follower (an earlier leg of a re-routed rid)
    must not start a told-less read from the hold."""
    f = _Rank(1, extent=98304)
    r1 = _req("weg2-11-5")
    f.add(r1)
    f.tree_cache.prefetch_loaded_tokens_by_reqid["weg2-11-5"] = _Rec(0, 0)
    r1._prefetch_span_tokens = 5000
    _tick_2s(r1)
    f._weg2_hold_refetch()
    assert f.registered == []


def test_pp0_never_rereads_a_rid_whose_told_is_on_the_wire(_env):
    """PP0's record is what it published; a later re-read would move it under
    the followers' feet (PP0's own admission would refuse)."""
    p0 = _Rank(0, extent=4095)
    r0 = _req("weg2-11-6")
    p0.add(r0)
    p0.wake()
    wire = m.pp0_publish(p0, [])
    assert [w.told for w in wire] == [4095]
    p0.weg2_dormant, p0.weg2_dormant_hold = True, [r0]  # forced: back in a hold
    p0.tree_cache.prefetch_loaded_tokens_by_reqid["weg2-11-6"] = _Rec(0, 0)
    r0._prefetch_span_tokens = 5000
    p0.tree_cache.extent = 98304
    _tick_2s(r0)
    p0._weg2_hold_refetch()
    assert len(p0.registered) == 1
