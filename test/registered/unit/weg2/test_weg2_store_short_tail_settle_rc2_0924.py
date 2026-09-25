"""RC2 metal (boot weg2rc2, D log 23:21:05 -> 23:21:25): #1324 store-short tail x #1471 settle.

weg2-6-2 -- the in-flip GEN, 4316 tokens; the store delivered 4095, 221 short, the
xsn437 form -- was held across the P phase. At the wake #1471 parked it ("read
still short"), the settle tick re-read it every pass, and each pass the #1324 tail
recompute fired: 569 lines per rank of "STORE-SHORT TAIL RECOMPUTE ... admitted,
D prefills it" while the request stayed PARKED, until the 20 s bound lapsed
("#1471 SETTLE-RELEASE ... lapsed=True held_after_wake_s=20.1"). Only then did the
X gate admit it (uncached 221) and D prefill the tail.

A short read whose remainder fits in X is SETTLED at the wake: D prefills the
remainder (the recompute the tail path already decides), so waiting for a re-read
the store does not complete only costs the bound. Over X (weg2xsn229: 4095 of
98210) the request keeps waiting for its re-read, as #1471 was built for. The
recompute line is rate-limited per request.
"""
import logging
import os
import time
import types

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import scheduler as sched_mod
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-weg2-unit")

S = sched_mod.Scheduler
X = 4096


def _holder(state, x=X):
    """The #1471 harness (test_weg2_post_wake_settle_1471) plus server_args for X;
    ``state`` is what _weg2_refetch_one answers for every held request."""
    h = types.SimpleNamespace()
    h.waiting_queue = []
    h.weg2_dormant_hold = []
    h.tree_cache = types.SimpleNamespace(cache_controller=types.SimpleNamespace(weg2_hold_rids=set()))
    h.WEG2_POST_WAKE_SETTLE_S = S.WEG2_POST_WAKE_SETTLE_S
    h.ps = types.SimpleNamespace(tp_size=1)
    h.server_args = types.SimpleNamespace(tp_prefill_max_tokens=x)
    for n in ("_weg2_release_dormant_hold", "_weg2_post_wake_settle_tick", "_weg2_group_min_flags"):
        setattr(h, n, types.MethodType(getattr(S, n), h))
    h._weg2_refetch_one = lambda req, now, allow_reissue=False: state
    return h


def _req(rid, n, delivered):
    return types.SimpleNamespace(rid=rid, full_untruncated_fill_ids=list(range(n)),
                                 _weg2_store_delivered=delivered)


def test_the_rc2_metal_form_is_released_at_the_wake():
    r = _req("weg2-6-2", 4316, 4095)                      # remainder 221 <= X
    h = _holder("wait")
    h.weg2_dormant_hold = [r]
    h.tree_cache.cache_controller.weg2_hold_rids = {"weg2-6-2"}
    h._weg2_release_dormant_hold()
    assert h.waiting_queue == [r], "a remainder within X is D's to prefill -- no 20 s settle"
    assert not getattr(h, "weg2_post_wake_settle", None)
    assert h.tree_cache.cache_controller.weg2_hold_rids == set()


def test_over_x_the_request_waits_for_its_re_read():
    r = _req("weg2-5-1", 98210, 4095)                     # weg2xsn229: remainder 94115 > X
    h = _holder("wait")
    h.weg2_dormant_hold = [r]
    h._weg2_release_dormant_hold()
    assert h.waiting_queue == [] and h.weg2_post_wake_settle == [r]


def test_no_stamp_or_the_switch_off_keeps_todays_park(monkeypatch):
    r = types.SimpleNamespace(rid="a", full_untruncated_fill_ids=list(range(4316)))   # no short-read stamp
    h = _holder("wait")
    h.weg2_dormant_hold = [r]
    h._weg2_release_dormant_hold()
    assert h.weg2_post_wake_settle == [r]
    monkeypatch.setenv("SGLANG_WEG2_STORE_SHORT_TAIL", "0")
    r2 = _req("b", 4316, 4095)
    h2 = _holder("wait")
    h2.weg2_dormant_hold = [r2]
    h2._weg2_release_dormant_hold()
    assert h2.weg2_post_wake_settle == [r2]


def test_the_settle_tick_releases_a_tail_request_before_the_bound():
    r = _req("weg2-6-2", 4316, 4095)
    r._1471_since = time.monotonic()                        # fresh: the 20 s bound has not lapsed
    h = _holder("reading")
    h.weg2_post_wake_settle = [r]
    assert h._weg2_post_wake_settle_tick() == 1
    assert h.waiting_queue == [r] and h.weg2_post_wake_settle == []


def test_a_stand_in_without_server_args_answers_as_today():
    r = _req("a", 4316, 4095)
    h = _holder("wait")
    del h.server_args
    h.weg2_dormant_hold = [r]
    h._weg2_release_dormant_hold()
    assert h.weg2_post_wake_settle == [r]


def test_the_recompute_line_is_rate_limited_per_request(caplog):
    s = types.SimpleNamespace(server_args=types.SimpleNamespace(tp_prefill_max_tokens=X))
    s._clear_prefetch_deferral_fields = lambda req: True
    r = _req("weg2-6-2", 4316, 4095)
    with caplog.at_level(logging.WARNING, logger=sched_mod.logger.name):
        for _ in range(5):
            assert sched_mod._weg2_store_short_recompute(
                s, r, sched_mod._DEFER_REASON_STORE_SHORT, 220) == "expired"
    lines = [x.getMessage() for x in caplog.records if "STORE-SHORT TAIL RECOMPUTE" in x.getMessage()]
    assert len(lines) == 3, lines                           # the 1st, 2nd and 4th of this request
