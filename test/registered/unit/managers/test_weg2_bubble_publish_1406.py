"""#1406: write_back + bubble publisher -- the publish sweep runs bounded in
the PP loop's bubbles, the backup thread's copies prefer those bubbles
through a soft gate that never blocks longer than its timeout."""

import os
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import pytest

from sglang.srt.managers import weg2_bubble_publish as bp


class _Tree:
    def __init__(self):
        self.calls = []
        self.cache_controller = SimpleNamespace()

    def publish_unbacked_sweep(self, max_issue=64):
        self.calls.append(max_issue)
        return {"unbacked": 3, "issued": 2, "refused": 0, "pending": 0, "skipped_pending": 1}


def _sched(storage=True):
    return SimpleNamespace(enable_hicache_storage=storage, tree_cache=_Tree())


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv(bp.ENV, "1")
    yield


def test_unarmed_without_env_or_storage(monkeypatch):
    monkeypatch.setenv(bp.ENV, "0")
    s = _sched()
    bp.bubble_begin(s)
    assert s.tree_cache.calls == []
    monkeypatch.setenv(bp.ENV, "1")
    s2 = _sched(storage=False)
    bp.bubble_begin(s2)
    assert s2.tree_cache.calls == []


def test_bubble_issues_a_bounded_sweep_at_most_every_interval():
    s = _sched()
    bp.bubble_begin(s)
    bp.bubble_begin(s)  # inside the interval: no second sweep
    assert s.tree_cache.calls == [bp.MAX_ISSUE]
    assert s._weg2_bubble_issued == 2
    s._weg2_bubble_last = time.monotonic() - bp.MIN_INTERVAL_S - 0.01
    bp.bubble_begin(s)
    assert len(s.tree_cache.calls) == 2


def test_gate_opens_in_the_bubble_and_closes_at_launch():
    s = _sched()
    bp.bubble_begin(s)
    ev = s.tree_cache.cache_controller._weg2_bubble_gate
    assert ev.is_set()
    assert bp.backup_wait_for_bubble(s.tree_cache.cache_controller, 0.01) is True
    bp.bubble_end(s)
    assert not ev.is_set()
    t = time.perf_counter()
    assert bp.backup_wait_for_bubble(s.tree_cache.cache_controller, 0.02) is False
    assert time.perf_counter() - t < 0.5, "soft gate: bounded wait, never a block"


def test_backup_without_a_gate_never_waits():
    t = time.perf_counter()
    assert bp.backup_wait_for_bubble(SimpleNamespace(), 1.0) is False
    assert time.perf_counter() - t < 0.1


def test_a_raising_sweep_does_not_take_the_loop_down():
    s = _sched()
    s.tree_cache.publish_unbacked_sweep = lambda max_issue=64: (_ for _ in ()).throw(RuntimeError("x"))
    bp.bubble_begin(s)  # no raise
