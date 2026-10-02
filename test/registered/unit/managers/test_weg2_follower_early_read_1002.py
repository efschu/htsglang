"""DP-NACHLAUF 02.10.: a P follower starts its store read at intake, beside
PP0's (SGLANG_WEG2_FOLLOWER_EARLY_READ=1, absolute tolds only); PP0's told
stays the authority for the depth.

N5d (0c996cf05c 1002_124821, D->P epoch 25, weg2-24-93): PP0 queue 1333 ms +
read 206 ms, then -- only after PP0's told -- PP1/PP2 queue ~495 + read ~176 ms,
serialised before the first prefill forward.

Pinned (red before): default off; armed, the follower registers at intake and
holds; the told then settles against the early read instead of registering a
second one: equal -> admitted as #1400; over (absolute) -> SATISFIED at told,
credit popped; over (span-relative) -> refuse (the #1400 mismatch names it);
short -> the #1400 told-limited registration now.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from sglang.srt.managers import weg2_store_told as m  # noqa: E402


class _Tree:
    def __init__(self, completed=0, credit=0):
        self.prefetch_loaded_tokens_by_reqid = {}
        self._completed = completed
        self._credit = credit
        self.is_eagle = False

    def check_prefetch_progress(self, rid):
        return True

    def completed_prefetch_tokens(self, rid):
        return self._completed

    def pop_prefetch_loaded_tokens(self, rid):
        c, self._credit = self._credit, 0
        return c


def _sched(pp_rank=1, completed=0, credit=0):
    s = SimpleNamespace()
    s.ps = SimpleNamespace(pp_rank=pp_rank)
    s._weg2_store_held = {}
    s._weg2_store_told = {}
    s.tree_cache = _Tree(completed, credit)
    s.waiting_queue = []
    s.calls = []

    def _prefetch_kvcache(req, limit_tokens=None):
        s.calls.append((req.rid, limit_tokens))
        return "issued"

    s._prefetch_kvcache = _prefetch_kvcache
    return s


def _req(rid="weg2-24-93"):
    return SimpleNamespace(rid=rid, prefix_indices=None, host_hit_length=0)


def _arm(monkeypatch, on=True):
    monkeypatch.setenv(m.ENV_FOLLOWER_EARLY_READ, "1" if on else "0")
    monkeypatch.setattr(m, "_absolute_armed", lambda: True)
    monkeypatch.delenv("SGLANG_WEG2_DUAL_SHARE", raising=False)


def test_switch_default_on():
    # default ON since 02.10. (metal N5t..N6e); "0" restores the told-first order
    assert m.follower_early_read_on({})
    assert m.follower_early_read_on({m.ENV_FOLLOWER_EARLY_READ: "1"})
    assert not m.follower_early_read_on({m.ENV_FOLLOWER_EARLY_READ: "0"})


def test_armed_follower_registers_at_intake_and_holds(monkeypatch):
    _arm(monkeypatch)
    s, r, gates = _sched(), _req(), []
    v = m.intake(s, r, gates.append)
    assert v == "declined:" + m.GATE_HELD and gates == [m.GATE_HELD]
    assert s.calls == [(r.rid, None)]            # the whole read, now
    assert r.rid in s._weg2_store_held and r.rid in s._weg2_follower_early


def test_off_or_relative_tolds_keep_the_1400_order(monkeypatch):
    _arm(monkeypatch, on=False)
    s, r = _sched(), _req()
    m.intake(s, r, lambda g: None)
    assert s.calls == []
    _arm(monkeypatch, on=True)
    monkeypatch.setattr(m, "_absolute_armed", lambda: False)
    s2 = _sched()
    m.intake(s2, _req("x"), lambda g: None)
    assert s2.calls == []


def test_the_told_settles_against_the_early_read_instead_of_a_second_one(monkeypatch):
    _arm(monkeypatch)
    s, r = _sched(), _req()
    m.intake(s, r, lambda g: None)
    s.calls.clear()
    assert m._follower_register(s, r, 130496) == "early:own_read"
    assert r._weg2_early_told == 130496 and s.calls == []


def test_settle_outcomes(monkeypatch):
    _arm(monkeypatch)
    s, r = _sched(credit=7), _req()
    assert m.follower_early_settle(s, r, 100, 100) == "equal"
    assert m.follower_early_settle(s, r, 100, 140, absolute=False) == "refuse"
    assert m.follower_early_settle(s, r, 100, 140, absolute=True) == "over"
    assert s._weg2_store_told_satisfied[r.rid] == 100
    s2, r2 = _sched(), _req("y")
    assert m.follower_early_settle(s2, r2, 100, 60, absolute=True) == "short"
    assert s2.calls == [("y", 100)]              # the #1400 told-limited read


def test_admission_admits_an_equal_early_read(monkeypatch):
    _arm(monkeypatch)
    s, r = _sched(completed=4096, credit=4096), _req()
    s._weg2_store_told[r.rid] = 4096
    r._weg2_early_told = 4096
    assert m.admission(s, r, lambda *a: None) == 4096
    assert r._weg2_early_told is None
