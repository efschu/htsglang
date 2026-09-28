"""P4b (28.09.): a short post-wake read with NO writer is decided at once; with a
writer the settle hold waits for its ack instead of re-reading every 2 s.

Metal (NF rc12z17-s0, 103bfdf29a, D log boot ...09281111): weg2-2-17 (41021
tokens, route=short straight to D) read 40512 of 40960 at the wake 11:18:00,
then ``#1456 HOLD-REFETCH zero-answer`` n=2..4 every 2 s beside ``#1442
HANDOFF-KEYS NONE registry=[]`` and ``#1472 READ-TRACE why=no-file`` -- nobody
was writing (D's own decode tail of the previous turn, never secured; P never
saw the request) -- until ``#1471 SETTLE-RELEASE lapsed=True
held_after_wake_s=20.0`` and ``X-GATE uncached=509 verdict=admit``: the
admission it would have had at 11:18:00. weg2-4-36 (1757) and weg2-5-37 (191)
the same.

Hermetic: the #1471 harness of test_weg2_store_short_tail_settle_rc2_0924 with a
refetch stand-in that obeys the 2 s timer, the hand-off directory a temp dir.
"""
import logging
import os
import time
import types

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler as sched_mod
from sglang.srt.weg2 import settle_writer as sw
from sglang.srt.weg2.handoff_keys import CHAIN_ATTR
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")

S = sched_mod.Scheduler
X = 12288  # NF-D's riegel on the metal (X-GATE ... X=12288)


class _Refetch:
    """The settle tick's view of _weg2_refetch_one: short, re-read due after 2 s."""

    def __init__(self):
        self.reissued = []

    def __call__(self, req, now, allow_reissue=False):
        req._1471_short = True
        if now - float(getattr(req, "_1456_last", 0.0) or 0.0) < 2.0:
            return "wait"
        if not allow_reissue:
            return "due"
        req._1456_last = now
        req._1456_n = int(getattr(req, "_1456_n", 0) or 0) + 1
        self.reissued.append(req.rid)
        return "reissued"


def _holder(x=X):
    h = types.SimpleNamespace()
    h.waiting_queue = []
    h.weg2_dormant_hold = []
    h.tree_cache = types.SimpleNamespace(cache_controller=types.SimpleNamespace(weg2_hold_rids=set()),
                                         prefetch_loaded_tokens_by_reqid={})
    h.WEG2_POST_WAKE_SETTLE_S = S.WEG2_POST_WAKE_SETTLE_S
    h.ps = types.SimpleNamespace(tp_size=1)
    h.server_args = types.SimpleNamespace(tp_prefill_max_tokens=x)
    for n in ("_weg2_post_wake_settle_tick", "_weg2_group_min_flags"):
        setattr(h, n, types.MethodType(getattr(S, n), h))
    h._weg2_refetch_one = _Refetch()
    return h


def _req(rid, n, delivered, *, seen_short=True, last=None):
    r = types.SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(n)),
                              _weg2_store_delivered=delivered, _prefetch_span_tokens=n)
    r._1471_since = time.monotonic()
    r._1456_last = time.monotonic() if last is None else last  # just re-read at the wake (n=1)
    r._1456_n = 1
    if seen_short:
        r._1471_short = True
    return r


@pytest.fixture
def arena(tmp_path, monkeypatch):
    monkeypatch.setenv("SGLANG_HICACHE_ARENA_DIR", str(tmp_path))
    monkeypatch.setenv("SGLANG_WEG2_STORE_SHORT_TAIL", "0")  # NF's default: the rc2 tail settle is off
    (tmp_path / "handoff").mkdir()
    return tmp_path


def _lines(caplog, tag):
    return [r.getMessage() for r in caplog.records if tag in r.getMessage()]


def test_the_metal_form_no_writer_is_released_at_once_without_a_re_read(arena, caplog):
    r = _req("weg2-2-17", 41021, 40512)
    h = _holder()
    h.weg2_post_wake_settle = [r]
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        assert h._weg2_post_wake_settle_tick() == 1
    assert h.waiting_queue == [r] and h.weg2_post_wake_settle == []
    assert h._weg2_refetch_one.reissued == [], "no writer: a re-read cannot see anything new"
    line = _lines(caplog, "SETTLE-NO-WRITER")
    assert len(line) == 1 and "remainder=509" in line[0] and "X=12288" in line[0] and "route=D" in line[0]
    assert any("state=no-writer lapsed=False" in m for m in _lines(caplog, "SETTLE-RELEASE"))
    assert any("writer=none prev=None action=decide" in m for m in _lines(caplog, "SETTLE-WRITER"))


def test_over_x_without_a_writer_goes_to_admission_now_for_the_p_route(arena, caplog):
    r = _req("weg2-9-9", 98210, 4095)  # remainder 94115 > X
    h = _holder()
    h.weg2_post_wake_settle = [r]
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        assert h._weg2_post_wake_settle_tick() == 1
    assert h.waiting_queue == [r]
    assert "route=P" in _lines(caplog, "SETTLE-NO-WRITER")[0]


def test_a_tail_part_under_write_is_waited_for_without_polling(arena, caplog):
    r = _req("weg2-3-3", 41021, 40512, last=0.0)  # the 2 s timer would allow a re-read
    (arena / "handoff" / "weg2-3-3.tail.pp2-1.pt.77.1.0.tmp").write_bytes(b"x")
    h = _holder()
    h.weg2_post_wake_settle = [r]
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        for _ in range(3):
            assert h._weg2_post_wake_settle_tick() == 0
    assert h.weg2_post_wake_settle == [r]
    assert h._weg2_refetch_one.reissued == [], "a writer at work: wait for its ack, no 2 s re-read"
    assert len([m for m in _lines(caplog, "SETTLE-WRITER") if "writer=p-writing" in m]) == 1


def test_the_publish_ack_re_reads_at_once_then_decides(arena, monkeypatch, caplog):
    r = _req("weg2-4-4", 41021, 40512)  # re-read 0 s ago: the timer says wait
    h = _holder()
    h.weg2_post_wake_settle = [r]
    monkeypatch.setattr(sw, "_tail_facts", lambda rid: ("partial", False))
    assert h._weg2_post_wake_settle_tick() == 0
    assert h._weg2_refetch_one.reissued == []
    monkeypatch.setattr(sw, "_tail_facts", lambda rid: ("complete", False))
    r._1471w_t = 0.0  # past the 100 ms listing throttle
    assert h._weg2_post_wake_settle_tick() == 0
    assert h._weg2_refetch_one.reissued == ["weg2-4-4"], "the ack is the clock, not the 2 s timer"
    r._1471w_t = 0.0
    with caplog.at_level(logging.INFO, logger=sched_mod.logger.name):
        assert h._weg2_post_wake_settle_tick() == 1  # the ack's re-read stayed short: no writer left
    assert h.waiting_queue == [r]
    assert _lines(caplog, "SETTLE-NO-WRITER")


def test_a_hand_off_without_a_visible_ack_keeps_the_2s_re_read(arena):
    r = _req("weg2-5-5", 41021, 40512)
    setattr(r, CHAIN_ATTR, ["k0", "k1"])
    h = _holder()
    h.weg2_post_wake_settle = [r]
    assert h._weg2_post_wake_settle_tick() == 0
    assert h._weg2_refetch_one.reissued == [] and h.weg2_post_wake_settle == [r]
    r._1456_last = time.monotonic() - 3.0
    assert h._weg2_post_wake_settle_tick() == 0
    assert h._weg2_refetch_one.reissued == ["weg2-5-5"], "unchanged: P's page write-through has no D-visible ack"


def test_without_the_shared_hand_off_directory_no_writer_is_provable(monkeypatch):
    monkeypatch.delenv("SGLANG_HICACHE_ARENA_DIR", raising=False)
    monkeypatch.setenv("SGLANG_WEG2_STORE_SHORT_TAIL", "0")
    r = _req("weg2-8-8", 41021, 40512)
    h = _holder()
    h.weg2_post_wake_settle = [r]
    assert h._weg2_post_wake_settle_tick() == 0
    assert h.weg2_post_wake_settle == [r], "unknown: the 2 s re-read stays, as shipped"
    assert sw.observe(r) == sw.UNKNOWN


def test_not_yet_seen_short_is_untouched(arena):
    r = _req("weg2-6-6", 41021, 40512, seen_short=False)
    h = _holder()
    h.weg2_post_wake_settle = [r]
    h._weg2_refetch_one = lambda req, now, allow_reissue=False: "reading"
    assert h._weg2_post_wake_settle_tick() == 0
    assert h.weg2_post_wake_settle == [r]


def test_a_read_in_flight_is_never_decided(arena):
    r = _req("weg2-7-7", 41021, 40512)
    h = _holder()
    h.weg2_post_wake_settle = [r]
    h._weg2_refetch_one = lambda req, now, allow_reissue=False: "reading"
    assert h._weg2_post_wake_settle_tick() == 0
    assert h.weg2_post_wake_settle == [r]


def test_classify_and_step():
    c = sw.classify
    assert c(chain=False, handoff_file=False, tail_state="none", tail_tmp=False) == sw.NONE
    assert c(chain=True, handoff_file=False, tail_state="none", tail_tmp=False) == sw.P_HANDOFF
    assert c(chain=False, handoff_file=True, tail_state="none", tail_tmp=False) == sw.P_HANDOFF
    assert c(chain=True, handoff_file=True, tail_state="partial", tail_tmp=False) == sw.P_WRITING
    assert c(chain=False, handoff_file=False, tail_state="none", tail_tmp=True) == sw.P_WRITING
    assert c(chain=True, handoff_file=False, tail_state="complete", tail_tmp=False) == sw.P_PUBLISHED
    assert sw.step(None, sw.NONE, None) == ("decide", None)
    assert sw.step(sw.P_WRITING, sw.P_WRITING, None) == ("wait", None)
    assert sw.step(sw.P_WRITING, sw.P_PUBLISHED, None) == ("reread", sw.P_PUBLISHED)
    assert sw.step(sw.P_PUBLISHED, sw.P_PUBLISHED, sw.P_PUBLISHED) == ("decide", sw.P_PUBLISHED)
    assert sw.step(sw.P_WRITING, sw.NONE, None) == ("reread", None)
    assert sw.step(None, sw.P_HANDOFF, None) == ("poll", None)
    assert sw.step(None, sw.UNKNOWN, None) == ("poll", None)
    assert sw.route(509, X) == "D" and sw.route(94115, X) == "P" and sw.route(None, X) == "D"
