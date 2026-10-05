# SPDX-License-Identifier: Apache-2.0
"""#1998 D-SEAT-LONG-FIRST RAW-PROMPT measure (pt4 finding of #1986): the long-leg measure counted only the tokens
the LAST P leg COMPUTED (prompt - cached). The 250k needle (weg2-0-61) was paused 3x and resumed from L2: its last
leg had prompt 249921, cached 241664 -> p_work 8257 < threshold, so the seat precedence did not protect it.
ENV switch SGLANG_WEG2_DUAL_D_SEAT_LONG_USE_RAW_PROMPT (default off = the old measure, byte for byte): the measure
becomes the raw prompt length of the request (leg1 prompt_tokens, est_prompt when leg 1 reported none).
Threshold SGLANG_WEG2_DUAL_D_SEAT_LONG_P_TOKENS unchanged. Dual front only; flip form unchanged."""
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
RAW = LF.ENV_RAW_PROMPT
THR = "131072"


def P(n, pt=0, ct=0, est=0):
    return types.SimpleNamespace(rid=f"weg2-0-{n}", leg1_prompt_tokens=pt, leg1_cached_tokens=ct, est_prompt=est)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in (ENV, RAW, "SGLANG_WEG2_DUAL_LAYOUT", "SGLANG_WEG2_GROUP"):
        monkeypatch.delenv(k, raising=False)


def _front(dual=True, items=()):
    f = object.__new__(F.Front)
    f.dual_layout = dual
    f.counters = collections.Counter()
    f._ready_for_d = collections.deque(items)
    return f


class TestMeasure:
    def test_switch_default_is_off(self):
        assert LF.use_raw_prompt() is False
        assert LF.measure() is LF.p_work_tokens

    def test_switch_on_selects_raw_prompt(self, monkeypatch):
        monkeypatch.setenv(RAW, "1")
        assert LF.use_raw_prompt() is True
        assert LF.measure() is LF.raw_prompt_tokens

    def test_raw_prompt_ignores_the_cache(self):
        needle = P(61, 249921, 241664)           # pt4 weg2-0-61: last leg computed only 8257
        assert LF.p_work_tokens(needle) == 8257
        assert LF.raw_prompt_tokens(needle) == 249921

    def test_raw_prompt_falls_back_to_est_prompt_without_leg1(self):
        assert LF.raw_prompt_tokens(P(5, 0, 0, est=77000)) == 77000
        assert LF.raw_prompt_tokens(types.SimpleNamespace(rid="x")) == 0


class TestFrontNeedle:
    """pt4: needle 249921 prompt / 8257 computed, threshold 131072; older 65k / 40k foreign requests are ahead."""

    def _items(self):
        return [P(50, 65000, 0), P(52, 40000, 0), P(61, 249921, 241664)]

    def test_switch_on_needle_goes_first(self, monkeypatch):
        monkeypatch.setenv(ENV, THR)
        monkeypatch.setenv(RAW, "1")
        f = _front(True, self._items())
        assert f._d_seat_long_first() is True
        assert [q.rid for q in f._ready_for_d] == ["weg2-0-61", "weg2-0-50", "weg2-0-52"]
        assert f.counters["d_seat_long_first"] == 1

    def test_switch_off_as_before_needle_not_long(self, monkeypatch):
        monkeypatch.setenv(ENV, THR)
        items = self._items()
        f = _front(True, items)
        assert f._d_seat_long_first() is False
        assert list(f._ready_for_d) == items and not f.counters

    def test_short_40k_stays_behind_with_switch_on(self, monkeypatch):
        monkeypatch.setenv(ENV, THR)
        monkeypatch.setenv(RAW, "1")
        items = [P(50, 65000, 0), P(52, 40000, 0), P(68, 52000, 47000)]
        f = _front(True, items)
        assert f._d_seat_long_first() is False
        assert list(f._ready_for_d) == items

    def test_raw_switch_without_threshold_is_off(self, monkeypatch):
        monkeypatch.setenv(RAW, "1")
        items = self._items()
        f = _front(True, items)
        assert f._d_seat_long_first() is False and list(f._ready_for_d) == items

    def test_long_ones_among_themselves_by_arrival(self, monkeypatch):
        monkeypatch.setenv(ENV, THR)
        monkeypatch.setenv(RAW, "1")
        f = _front(True, [P(70, 1000), P(75, 200000, 190000), P(61, 249921, 241664)])
        f._d_seat_long_first()
        assert [q.rid for q in f._ready_for_d] == ["weg2-0-61", "weg2-0-75", "weg2-0-70"]


class TestFlipUnchanged:
    def test_flip_form_with_both_switches_is_inert(self, monkeypatch):
        monkeypatch.setenv(ENV, THR)
        monkeypatch.setenv(RAW, "1")
        items = [P(64, 65000), P(61, 249921, 241664)]
        f = _front(False, items)
        with mock.patch.object(F.logger, "info") as log:
            assert f._d_seat_long_first() is False
        assert list(f._ready_for_d) == items and not f.counters
        log.assert_not_called()

    def test_default_off_log_line_unchanged(self, monkeypatch):
        monkeypatch.setenv(ENV, "32768")
        f = _front(True, [P(13, 9821), P(20, 50967)])
        with mock.patch.object(F.logger, "info") as log:
            f._d_seat_long_first()
        msg = log.call_args[0][0] % log.call_args[0][1:]
        assert msg.startswith("WEG2 D-SEAT-LONG-FIRST rid=weg2-0-20 p_work=50967 (leg 1 computed >= 32768 tokens)")

    def test_no_running_or_seat_state_is_touched(self):
        src = inspect.getsource(F.Front._d_seat_long_first) + inspect.getsource(LF)
        for forbidden in ("_d_seat.", "_d_seats_live", "groups[", "release(", "acquire(", "time.time", "monotonic"):
            assert forbidden not in src, forbidden


class TestAdmitterEndToEnd:
    def _run(self, order_in, monkeypatch, raw):
        monkeypatch.setenv(ENV, THR)
        if raw:
            monkeypatch.setenv(RAW, "1")
        monkeypatch.setattr(F, "POST_BARRIER_S", 0.01)
        loop = asyncio.new_event_loop()
        try:
            async def go():
                f = F.Front("http://p", "http://d", "D", "t", "", 0, 0, {}, 45.0, d_bs=1, dual_layout=True)
                f.admit_d = True
                f._d_accepts_leg2 = lambda: True

                async def _no_reading():
                    return None
                f._d_reading_if_armed = _no_reading
                f._d_token_budget_blocks = lambda *a, **k: False
                f._hl_check = lambda *a, **k: False
                seated, ps = [], []
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

    ORDER = [("weg2-0-50", 65000, 0), ("weg2-0-52", 40000, 0), ("weg2-0-61", 249921, 241664)]

    def test_raw_on_needle_seated_first(self, monkeypatch):
        assert self._run(self.ORDER, monkeypatch, True) == ["weg2-0-61", "weg2-0-50", "weg2-0-52"]

    def test_raw_off_age_order_as_before(self, monkeypatch):
        assert self._run(self.ORDER, monkeypatch, False) == ["weg2-0-50", "weg2-0-52", "weg2-0-61"]
