# SPDX-License-Identifier: Apache-2.0
"""#1986 D-SEAT-LONG-FIRST (pt2 fs10051150): a request with a finished LONG P leg takes the next free D seat
before waiting requests without one. ENV switch SGLANG_WEG2_DUAL_D_SEAT_LONG_P_TOKENS, default 0 = off, dual
front only. One class per claim; FlipUnchanged is the "flip form / default dual path unchanged" test in the
style of test_dual_fixes_flip_unchanged_1003."""
from __future__ import annotations

import asyncio
import collections
import inspect
import os
import types
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import dual_seat_long_first as LF  # noqa: E402
from sglang.srt.weg2 import front as F  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

ENV = LF.ENV


def P(n, pt=0, ct=0):
    return types.SimpleNamespace(rid=f"weg2-0-{n}", leg1_prompt_tokens=pt, leg1_cached_tokens=ct)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in (ENV, "SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP"):
        monkeypatch.delenv(k, raising=False)


def _front(dual=True, items=()):
    f = object.__new__(F.Front)
    f.dual_layout = dual
    f.counters = collections.Counter()
    f._ready_for_d = collections.deque(items)
    return f


class TestOrder:
    def test_default_is_off(self):
        assert LF.threshold() == 0

    def test_p_work_is_prompt_minus_cached(self):
        # pt2 0-21: prompt 51122, cached 47360 = 6.3 s of P work, NOT a long leg
        assert LF.p_work_tokens(P(21, 51122, 47360)) == 3762
        assert LF.p_work_tokens(P(10, 249962, 0)) == 249962
        assert LF.p_work_tokens(types.SimpleNamespace(rid="x")) == 0   # a request without leg 1

    def test_long_ones_first_oldest_first_others_keep_order(self):
        items = [P(13, 9821), P(15, 12226), P(30, 90000), P(16, 12225), P(20, 50967), P(12, 40000)]
        out = [x.rid for x in LF.long_first(items, 32768)]
        assert out == ["weg2-0-12", "weg2-0-20", "weg2-0-30", "weg2-0-13", "weg2-0-15", "weg2-0-16"]

    def test_threshold_zero_or_no_long_leg_is_the_given_order(self):
        items = [P(5, 100), P(3, 200), P(4, 300)]
        assert LF.long_first(items, 0) == items
        assert LF.long_first(items, 32768) == items

    def test_cached_long_prompt_is_not_long(self):
        items = [P(7, 100), P(21, 51122, 47360)]
        assert LF.long_first(items, 32768) == items


class TestFrontReorder:
    def test_pt2_shape_long_leg_jumps_the_older_short_ones(self, monkeypatch):
        monkeypatch.setenv(ENV, "32768")
        short1, short2, longp = P(13, 9821), P(15, 12226), P(20, 50967)
        f = _front(True, [short1, short2, longp])
        assert f._d_seat_long_first() is True
        assert [q.rid for q in f._ready_for_d] == ["weg2-0-20", "weg2-0-13", "weg2-0-15"]
        assert f.counters["d_seat_long_first"] == 1
        # same pass again: nothing moves, edge trigger does not count twice
        assert f._d_seat_long_first() is False
        assert f.counters["d_seat_long_first"] == 1

    def test_head_already_long_is_not_an_overtake(self, monkeypatch):
        monkeypatch.setenv(ENV, "32768")
        f = _front(True, [P(10, 249962), P(13, 9821)])
        assert f._d_seat_long_first() is False
        assert f.counters["d_seat_long_first"] == 0

    def test_nothing_is_removed_or_added(self, monkeypatch):
        monkeypatch.setenv(ENV, "1000")
        items = [P(3, 10), P(5, 5000), P(4, 20), P(2, 9000)]
        f = _front(True, items)
        f._d_seat_long_first()
        assert sorted(id(q) for q in f._ready_for_d) == sorted(id(q) for q in items)


class TestFlipUnchanged:
    """The flip form (no dual layout) and the default dual path (switch 0) do not reorder, count or log."""

    def test_flip_form_with_switch_set_is_inert(self, monkeypatch):
        monkeypatch.setenv(ENV, "32768")
        items = [P(13, 9821), P(20, 50967)]
        f = _front(False, items)
        with mock.patch.object(F.logger, "info") as log:
            assert f._d_seat_long_first() is False
        assert list(f._ready_for_d) == items
        assert not f.counters
        log.assert_not_called()

    def test_dual_front_switch_off_is_inert(self):
        items = [P(13, 9821), P(20, 50967)]
        f = _front(True, items)
        assert f._d_seat_long_first() is False
        assert list(f._ready_for_d) == items and not f.counters

    def test_bad_switch_value_is_off(self, monkeypatch):
        monkeypatch.setenv(ENV, "banana")
        items = [P(13, 9821), P(20, 50967)]
        f = _front(True, items)
        assert f._d_seat_long_first() is False and list(f._ready_for_d) == items

    def test_admitter_calls_it_after_the_age_sort_and_before_the_head_is_read(self):
        src = inspect.getsource(F.Front.d_admitter)
        i_sa = src.index("_sa.by_age(self._ready_for_d)")
        i_lf = src.index("self._d_seat_long_first()")
        i_head = src.index("p = self._ready_for_d[0]")
        assert i_sa < i_lf < i_head

    def test_no_running_or_seat_state_is_touched(self):
        # the method sees only the deque and the counters: a front without seats/semaphore/groups runs it
        src = inspect.getsource(F.Front._d_seat_long_first)
        for forbidden in ("_d_seat.", "_d_seats_live", "groups[", "release(", "acquire(", "time.time", "monotonic"):
            assert forbidden not in src, forbidden


class TestAdmitterEndToEnd:
    """The real d_admitter with one seat: with the switch the long leg takes the seat first."""

    def _run(self, order_in, monkeypatch, switch):
        if switch:
            monkeypatch.setenv(ENV, "32768")
        monkeypatch.setattr(F, "POST_BARRIER_S", 0.01)
        loop = asyncio.new_event_loop()
        try:
            async def go():
                f = F.Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=1,
                            dual_layout=True)
                f.admit_d = True
                f._d_accepts_leg2 = lambda: True

                async def _no_reading():
                    return None
                f._d_reading_if_armed = _no_reading
                f._d_token_budget_blocks = lambda *a, **k: False
                f._hl_check = lambda *a, **k: False
                seated = []
                ps = []
                for rid, pt, ct in order_in:
                    p = types.SimpleNamespace(rid=rid, fut=loop.create_future(), est_prompt=pt, leg1_prompt_tokens=pt,
                                              leg1_cached_tokens=ct, t_arrive=0.0, short_kept=False, t_ready=0.0)
                    ps.append(p)
                    f._ready_for_d.append(p)
                task = asyncio.ensure_future(f.d_admitter())
                for _ in range(400):
                    await asyncio.sleep(0.02)
                    for p in ps:
                        if p.fut.done() and p.rid not in seated:
                            seated.append(p.rid)
                    if len(seated) == len(ps):
                        break
                task.cancel()
                return seated
            return loop.run_until_complete(go())
        finally:
            loop.close()

    def test_switch_on_long_leg_seated_first(self, monkeypatch):
        order = [("weg2-0-13", 9821, 0), ("weg2-0-15", 12226, 0), ("weg2-0-20", 50967, 0)]
        assert self._run(order, monkeypatch, True) == ["weg2-0-20", "weg2-0-13", "weg2-0-15"]

    def test_switch_off_age_order_as_before(self, monkeypatch):
        order = [("weg2-0-13", 9821, 0), ("weg2-0-15", 12226, 0), ("weg2-0-20", 50967, 0)]
        assert self._run(order, monkeypatch, False) == ["weg2-0-13", "weg2-0-15", "weg2-0-20"]

