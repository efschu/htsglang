"""NW (30.09.): a store read the HOST BUDGET refused is "not read yet", never
"read short, no writer".

Metal (NF y3u, 5bedac26f1, boot ...0930_002717, D TP0). Five requests parked
over the flip, three of them ~130k tokens: weg2-0-5 130048 + weg2-30-49 131136 +
weg2-30-50 131136 = 392320 registered tokens > the prefetch limit 373536. At the
wakes 00:41:52 / 00:42:08 / 00:42:21 the reads of weg2-30-52 (77695) and
weg2-31-53 (77854) were REFUSED before they ran::

    #915 PREFETCH REFUSED reason=vote_negative rid=weg2-31- need=77824
         occupied=392320 limit=373536
    #1471w SETTLE-WRITER rid=weg2-31-53 writer=p-published prev=none action=reread
    #1456 HOLD-REFETCH rid=weg2-31-53 n=2 reason=record-short verdict=declined:rate_limited
    #1471w SETTLE-NO-WRITER rid=weg2-31-53 remainder=? ... re_reads=2
    WEG2 X-GATE-TERMS rid=weg2-31-53 total=77854 head=0 store=0 ... verdict=W31

P had published the tails (``WEG2-TAIL-PUBLISH ... of=3``), the settle spent
P's ack on a re-read that never registered and decided "no writer" one tick
later; the X gate priced the whole prompt, the front re-routed via P twice and
the third refusal reached the client (``LEG2-TERMINAL-NAMED reason=W50``).

Red on 5bedac26f1: the request is released as ``no-writer`` while the budget is
still full. Green: it stays parked, is re-read the moment the budget has room,
and joins the queue ``complete``.

Hermetic: a bare Scheduler with the REAL ``_weg2_post_wake_settle_tick`` /
``_weg2_refetch_one`` / settle-writer path; the budget and the store are
stand-ins; the hand-off directory is a temp dir.
"""
import logging
import os
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler as sched_mod  # noqa: E402
from sglang.srt.weg2 import park_l3  # noqa: E402
from sglang.srt.weg2 import settle_writer as sw  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")

S = sched_mod.Scheduler
SPAN = 77854  # weg2-31-53
LIMIT = 373536


class _Budget:
    """The host prefetch budget and the store read: full until ``free()``."""

    def __init__(self, sched, span):
        self.full = True
        self.calls = []
        self.sched = sched
        self.span = span
        self.done = False

    def free(self):
        self.full = False

    def prefetch_kvcache(self, req, limit_tokens=None):
        self.calls.append(("full" if self.full else "room", req.rid))
        if self.full:
            return "declined:rate_limited"
        self.sched.tree_cache.ongoing_prefetch[req.rid] = object()
        return "issued"

    def land(self, rid):
        self.sched.tree_cache.ongoing_prefetch.pop(rid, None)
        self.sched.tree_cache.prefetch_loaded_tokens_by_reqid[rid] = self.span
        self.done = True


def _sched():
    s = S.__new__(S)
    s.weg2_dormant = False
    s.weg2_dormant_hold = []
    s.waiting_queue = []
    s.ps = types.SimpleNamespace(tp_size=1, pp_size=1)
    s.server_args = types.SimpleNamespace(tp_prefill_max_tokens=12288)
    s._weg2_wake_seq = 1
    s.tree_cache = types.SimpleNamespace(
        prefetch_loaded_tokens_by_reqid={},
        ongoing_prefetch={},
    )
    s.tree_cache.check_prefetch_progress = lambda rid: rid not in s.tree_cache.ongoing_prefetch
    b = _Budget(s, SPAN)
    s.tree_cache.cache_controller = types.SimpleNamespace(
        weg2_hold_rids=set(), prefetch_rate_limited=lambda: b.full
    )
    s._prefetch_kvcache = b.prefetch_kvcache
    s._clear_prefetch_deferral_fields = lambda req: None
    # the record is short of the span (the metal's "record-short") until the read lands
    s._weg2_note_store_shortfall = lambda req: None if b.done else "record-short"
    return s, b


def _req(rid="weg2-31-53"):
    r = types.SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(SPAN)),
                              _prefetch_span_tokens=SPAN, output_ids=[])
    r._1471_since = time.monotonic()
    r._1471_short = True
    r._1456_n = 1
    r._1456_last = time.monotonic()
    return r


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_WEG2_STORE_SHORT_TAIL", "0")
    (tmp_path / "handoff").mkdir()
    # P's tail parts are published (WEG2-TAIL-PUBLISH ... of=3 on the metal)
    monkeypatch.setattr(sw, "_tail_facts", lambda rid: ("complete", False))
    return tmp_path


def _tick(s, r):
    r._1471w_t = 0.0  # past the 100 ms listing throttle of the writer view
    r._1471b_t = 0.0 if getattr(r, "_1471b_t", None) is not None else None
    return s._weg2_post_wake_settle_tick()


def _msgs(caplog, tag):
    return [x.getMessage() for x in caplog.records if tag in x.getMessage()]


def test_the_metal_form_a_budget_refused_re_read_is_never_no_writer(arena, caplog):
    s, b = _sched()
    r = _req()
    s.weg2_post_wake_settle = [r]
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        for _ in range(4):  # the budget is held by the three 130k siblings
            assert _tick(s, r) == 0
        assert s.waiting_queue == [], "released on a read that never ran (the y3u W50 path)"
        assert not _msgs(caplog, "SETTLE-NO-WRITER")
        assert sw.budget_pending(r)
        assert r._1471w_ack is None, "the writer's ack must survive a refused re-read"
        # the siblings' reads land: the budget has room, the re-read registers at once
        b.free()
        assert _tick(s, r) == 0
        assert b.calls[-1] == ("room", "weg2-31-53")
        assert not sw.budget_pending(r) and s.waiting_queue == []  # in flight: still parked
        b.land("weg2-31-53")
        assert _tick(s, r) == 1
    assert s.waiting_queue == [r]
    assert any("state=complete" in m for m in _msgs(caplog, "SETTLE-RELEASE"))
    assert not _msgs(caplog, "SETTLE-NO-WRITER")
    assert _msgs(caplog, "#1471b BUDGET-REREAD")


def test_no_collective_is_attempted_while_the_budget_is_full(arena):
    """The retry is a group collective (#580 vote): with the budget full it is
    not re-issued on every tick."""
    s, b = _sched()
    r = _req()
    s.weg2_post_wake_settle = [r]
    _tick(s, r)  # the ack's re-read: refused
    n = len(b.calls)
    for _ in range(5):
        _tick(s, r)
    assert len(b.calls) == n


def test_the_wake_read_refused_by_the_budget_parks_unread(arena, caplog):
    """#248 WAKE-READ: the budget refuses the hold read at the wake -- marked
    unread, and named."""
    s, b = _sched()
    r = _req("weg2-30-52")
    setattr(r, park_l3.DEFER_ATTR, True)
    with caplog.at_level(logging.INFO, logger=park_l3.logger.name):
        park_l3.issue_deferred_reads(s, [r])
    assert sw.budget_pending(r)
    assert _msgs(caplog, "#1471b WAKE-READ BUDGET-REFUSED")
    # the release verdict of the wake: never "complete" on an unread request
    assert S._weg2_refetch_one(s, r, time.monotonic(), allow_reissue=False) == "wait"


def test_a_stale_ack_does_not_outlive_its_wake(arena):
    """y3u 00:42:21: the ack of the previous wake was still marked spent, so the
    settle decided at its first tick without any re-read. A new wake starts the
    writer view fresh."""
    r = _req()
    r._1471w_state, r._1471w_ack, r._1471w_t = sw.P_PUBLISHED, sw.P_PUBLISHED, 5.0
    sw.reset_for_wake(r)
    act, _ack = sw.step(r._1471w_state, sw.P_PUBLISHED, r._1471w_ack)
    assert act == "reread"
    src = open(sched_mod.__file__).read()
    j = src.index("def _weg2_release_dormant_hold")
    assert "reset_for_wake" in src[j:j + 9000]


def test_budget_terms():
    assert sw.budget_refused("declined:rate_limited")
    assert sw.budget_refused("declined:vote_negative")
    assert not sw.budget_refused("declined:too_short")
    assert not sw.budget_refused("declined:already_in_flight")
    assert not sw.budget_refused("issued")
    assert sw.gate_action("decide", types.SimpleNamespace(_1471b_budget=True)) == "poll"
    assert sw.gate_action("decide", types.SimpleNamespace()) == "decide"
