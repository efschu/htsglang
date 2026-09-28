"""Load-back liveness (27B rc12z28 boot dkr27browauthoritybar1w109281851, 19:02:32-19:04:03).

weg2-24-49's #988 load-back was issued on PP1/PP2 in PP0's last pass; under the row
authority a frameless cycle skips the plan, so ``check_hicache_events`` (the only
caller of ``loading_check``) never ran again on the followers: ``ongoing_load_back``
kept 1 entry, every #1268 lap voted ``hicache_load_back(1)`` on PP1 and PP2 and the
P->D quiesce ended in ``WEG2 STOP W3``. The vote now polls the follower's own finished
acks (rank-local, #737) before it attaches its slot.
"""
from __future__ import annotations

import inspect
import types

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

from sglang.srt.managers import scheduler_pp_mixin as mix  # noqa: E402
from sglang.srt.weg2 import p_row_authority  # noqa: E402


class _Tree:
    def __init__(self, finished=True):
        self.ongoing_load_back = {173: object()}
        self.finished = finished
        self.polled = 0

    def loading_check(self):
        self.polled += 1
        if self.finished:
            self.ongoing_load_back.clear()


def _follower(pp_rank=1, drained=True, finished=True):
    s = types.SimpleNamespace()
    s.ps = types.SimpleNamespace(pp_rank=pp_rank)
    s.tree_cache = _Tree(finished)
    s._pp_microbatches_drained = lambda: drained
    return s


def test_a_drained_follower_polls_its_finished_load_back_at_the_vote(monkeypatch):
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    s = _follower()
    assert mix.weg2_loadback_drain_on_idle_vote(s) == 1
    assert s.tree_cache.ongoing_load_back == {}          # the blocker of 19:02:33-19:04:03 is gone


def test_an_unfinished_load_back_still_blocks(monkeypatch):
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    s = _follower(finished=False)
    assert mix.weg2_loadback_drain_on_idle_vote(s) == 0
    assert s.tree_cache.polled == 1 and len(s.tree_cache.ongoing_load_back) == 1


def test_pp0_undrained_or_no_row_authority_do_not_poll(monkeypatch):
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    for s in (_follower(pp_rank=0), _follower(drained=False)):
        assert mix.weg2_loadback_drain_on_idle_vote(s) == 0
        assert s.tree_cache.polled == 0
    monkeypatch.setattr(p_row_authority, "applies", lambda s: False)
    s = _follower()
    assert mix.weg2_loadback_drain_on_idle_vote(s) == 0 and s.tree_cache.polled == 0


def test_a_stand_in_without_a_tree_returns_at_once():
    assert mix.weg2_loadback_drain_on_idle_vote(types.SimpleNamespace()) == 0


def test_the_vote_drains_before_it_attaches_the_slot():
    src = inspect.getsource(mix.SchedulerPPMixin._weg2_vote_attach_own_slot)
    i = src.index("weg2_loadback_drain_on_idle_vote(self)")
    assert i < src.index("attach_slot(vote, rank, self.is_fully_idle()")
