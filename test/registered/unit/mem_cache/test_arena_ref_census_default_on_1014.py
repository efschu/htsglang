"""NF arena fix 3 (14.10.2026): the arena reference census runs by default.

The boot of 14.10. 03:00-04:50Z had no ``ARENA-REF-CENSUS`` line in P.log or
D.log: ``SGLANG_HICACHE_ARENA_REF_CENSUS_S`` was only in ``profile_instr_env``
and that boot ran without instruments (nf-arena-voll-analyse-1014 section 1,
section 2 'Grenze des Zensus'), so every statement about WHO holds the arena at
the REFUSED moments was derived from the resets. Now the code default is 60 s
(user order 14.10.: new functions are ON); ``0`` still opts out, any other value
is the period. An instrument: no round and no decision reads it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(__file__)

import threading
import types

import pytest

from sglang.srt.mem_cache.storage.file import hicache_arena as ha


class _Event:
    """Records the period the census thread waits for, then says 'stop'."""

    waits = []

    def wait(self, every):
        _Event.waits.append(every)
        return True

    def set(self):
        pass


class _InlineThread:
    """Runs the census body AFTER ``_start_ref_census`` released its lock (the
    real thread takes the same lock for its registry exit; inline it would
    deadlock on the starter's own hold)."""

    pending = []

    def __init__(self, target=None, name=None, daemon=None):
        self.target = target

    def start(self):
        _InlineThread.pending.append(self.target)

    @classmethod
    def run_all(cls):
        todo, cls.pending = cls.pending, []
        for fn in todo:
            fn()


@pytest.fixture
def census(monkeypatch):
    _Event.waits = []
    _InlineThread.pending = []
    monkeypatch.setattr(ha.threading, "Event", _Event)
    monkeypatch.setattr(ha.threading, "Thread", _InlineThread)
    monkeypatch.setattr(ha, "_CENSUS_THREADS", {})
    return types.SimpleNamespace(path="/dev/shm/none-%d" % id(object()))


def test_unset_is_on_with_60_seconds(census, monkeypatch):
    monkeypatch.delenv(ha.ENV_REF_CENSUS_S, raising=False)
    ha._start_ref_census(census)
    _InlineThread.run_all()
    assert _Event.waits == [60.0], "RED on f4afb10f28: no thread (default 0)"


def test_zero_opts_out(census, monkeypatch):
    monkeypatch.setenv(ha.ENV_REF_CENSUS_S, "0")
    ha._start_ref_census(census)
    _InlineThread.run_all()
    assert _Event.waits == []


def test_a_value_is_the_period(census, monkeypatch):
    monkeypatch.setenv(ha.ENV_REF_CENSUS_S, "10")
    ha._start_ref_census(census)
    _InlineThread.run_all()
    assert _Event.waits == [10.0]


def test_a_closed_arena_ends_the_census(monkeypatch):
    """A tmp arena of a test, or a shutdown: the thread leaves the registry
    instead of logging 'failed' every period."""
    ticks = []

    class _TickEvent:
        def wait(self, every):
            ticks.append(every)
            return len(ticks) > 1       # one tick, then 'stop'

        def set(self):
            pass

    monkeypatch.setattr(ha.threading, "Event", _TickEvent)
    monkeypatch.setattr(ha.threading, "Thread", _InlineThread)
    monkeypatch.setattr(ha, "_CENSUS_THREADS", {})
    monkeypatch.delenv(ha.ENV_REF_CENSUS_S, raising=False)
    boom = []
    arena = types.SimpleNamespace(path="/dev/shm/closed-x", _mm=types.SimpleNamespace(closed=True),
                                  ref_census=lambda: boom.append(1))
    _InlineThread.pending = []
    ha._start_ref_census(arena)
    _InlineThread.run_all()
    assert boom == [], "a closed mapping is never read"
    assert ha._CENSUS_THREADS == {}, "the thread removed itself from the registry"
