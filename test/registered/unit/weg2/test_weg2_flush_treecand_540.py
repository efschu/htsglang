"""540 (27B y8r 55c95a89c7): the L1.5 part of D's /flush_cache under L15-TREE-CAND.

The sleep flush's L15 block took 1.77-1.91 s on every rank at 151k held rows (09:14:02). The two
capped ranks (TP1/TP2, the critical path; TP0 waits for them at the SLEEP-AGREE POST gather) spent
755-770 ms of their ``retain_ms`` in L15-CHECK-SNAP -- a diagnostic that rebuilds the WHOLE wake
plan (``l15_restore.owned_l2_rows``) synchronously inside the flip, only to sample 64 rows. The
plan's warm thread built the same plan for free after the sleep.

* CHECK-SNAP is opt-in now (SGLANG_WEG2_L15_CHECK_SNAP=1); the wake's CHECK-WHO line still names
  a bad row without it (``snap=none``).
* The warm thread waits until D is dormant (kv paused) before its Python walk, so it never takes
  the GIL from the rest of the sleep leg; bounded by SGLANG_WEG2_L15_PLAN_WARM_DEFER_S.
* L15-SLEEP-TIMING names pre (agree + TREE-CAND), snap and post (POST gather incl. the wait).
"""
from __future__ import annotations

import inspect
import os
import threading
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest  # noqa: E402

from sglang.srt.weg2 import l15_check_snap as CS  # noqa: E402
from sglang.srt.weg2 import l15_restore as R  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


def test_check_snap_is_opt_in():
    assert CS.enabled({}) is False                                   # RED before 540 (default on)
    assert CS.enabled({"SGLANG_WEG2_L15_CHECK_SNAP": "0"}) is False
    assert CS.enabled({"SGLANG_WEG2_L15_CHECK_SNAP": ""}) is False
    assert CS.enabled({"SGLANG_WEG2_L15_CHECK_SNAP": "1"}) is True


def test_default_snapshot_never_builds_the_plan(monkeypatch):
    """The cost was the plan: with the default switch the sleep must not touch it."""
    monkeypatch.delenv("SGLANG_WEG2_L15_CHECK_SNAP", raising=False)

    def boom(*a, **k):
        raise AssertionError("owned_l2_rows built in the sleep flush")

    monkeypatch.setattr(R, "owned_l2_rows", boom)
    assert CS.snap_at_sleep(object(), 1, [0, 1, 2, 3], object(), object()) == 0


def test_check_who_still_names_the_row_without_the_snapshot(monkeypatch):
    monkeypatch.delenv("SGLANG_WEG2_L15_CHECK_SNAP", raising=False)
    seen = []
    monkeypatch.setattr(CS, "explain", lambda *a: seen.append(a) or ["L15-CHECK-WHO x snap=none"])
    CS.set_wake_context(0, object(), [0, 1], object())
    try:
        assert CS.explain_current([(3, None, None)]) == ["L15-CHECK-WHO x snap=none"]  # RED before
    finally:
        CS.set_wake_context(None, None, None, None)
    assert len(seen) == 1


def _counting_plan(monkeypatch):
    calls = []
    done = threading.Event()

    def plan(m, rank, prefix):
        calls.append((rank, tuple(prefix), time.monotonic()))
        done.set()
        return []

    monkeypatch.setattr(R, "owned_l2_rows", plan)
    return calls, done


def test_warm_waits_until_ready(monkeypatch):
    calls, done = _counting_plan(monkeypatch)
    flag = {"dormant": False}
    R.warm_plan_async(object(), 1, [0, 1, 2], ready=lambda: flag["dormant"], defer_s=5.0)
    time.sleep(0.25)
    assert calls == []                                   # RED before 540: built at once
    t_ready = time.monotonic()
    flag["dormant"] = True
    assert done.wait(2.0)
    assert len(calls) == 1 and calls[0][2] >= t_ready


def test_warm_builds_anyway_after_the_bound(monkeypatch):
    calls, done = _counting_plan(monkeypatch)
    t0 = time.monotonic()
    R.warm_plan_async(object(), 0, [0, 1], ready=lambda: False, defer_s=0.15)
    assert done.wait(2.0)
    assert calls[0][2] - t0 >= 0.12                      # a sleep that never goes dormant still warms


def test_warm_defer_zero_and_no_ready_are_the_old_form(monkeypatch):
    calls, done = _counting_plan(monkeypatch)
    R.warm_plan_async(object(), 0, [0, 1], ready=lambda: False, defer_s=0.0)
    assert done.wait(1.0)
    done.clear()
    R.warm_plan_async(object(), 0, [0, 1])               # the pre-540 call shape
    assert done.wait(1.0)
    assert len(calls) == 2


def test_warm_raising_ready_builds_at_once(monkeypatch):
    calls, done = _counting_plan(monkeypatch)

    def bad():
        raise RuntimeError("no scheduler")

    R.warm_plan_async(object(), 0, [0, 1], ready=bad, defer_s=5.0)
    assert done.wait(1.0)


def test_default_defer_bound(monkeypatch):
    from sglang.srt.environ import envs

    monkeypatch.delenv("SGLANG_WEG2_L15_PLAN_WARM_DEFER_S", raising=False)
    assert envs.SGLANG_WEG2_L15_PLAN_WARM_DEFER_S.get() == 5.0
    assert R.warm_defer_s() == 5.0
    monkeypatch.setenv("SGLANG_WEG2_L15_PLAN_WARM_DEFER_S", "0")
    assert R.warm_defer_s() == 0.0


def test_scheduler_wiring():
    from sglang.srt.managers import scheduler

    src = inspect.getsource(scheduler.Scheduler.flush_cache)
    assert 'ready=lambda: bool(getattr(self, "weg2_dormant", False))' in src
    assert "pre_ms=%.0f snap_ms=%.0f post_ms=%.0f" in src
    assert '_l15_tt["pre"] = _l15_tt["bind0"] - _l15_tt["t0"]' in src
    assert '_l15_tt["snap"] = time.perf_counter() - _l15_tt["snap0"]' in src
    assert '_l15_tt["post"] = time.perf_counter() - _l15_tt["post0"]' in src
    # the snapshot is still gated by its own switch at the call site
    assert "_l15_cs.enabled() and l15_shadow.own_cap_rows(self, os.environ) > 0" in src
