"""Write-through liveness (27B rc12z30x2 boot dkr27browauthoritybar1fs09290956, 424346f693).

P.log 09:59:40: PP1 ``WEG2 RETAIN-PUBLISH rid=weg2-0-2 {... 'issued': 1 ... 'pending': 1}`` in the
last pass before the P->D quiesce; from then on every #1268 lap (epoch 2 .. 7374, 10:01:10) printed
``[PP1] WEG2-P-IDLE-VERDICT idle=False blocking_rank=1 blockers=[hicache_write_through(1)]``,
PP0 answered every /flush_cache ``Cache not flushed ... GROUP VERDICT PENDING`` and followers
``WEG2-FLUSH-VERDICT dropped ... PP0 refused this flush``; the front ended in ``WEG2 STOP W3
Weg2DrainWitnessDisagreement``. Only PP0 ever printed ``#1465 WRITE-BACK DRAIN`` after 09:58:07.

Under the row authority a frameless cycle skips the plan, so ``check_hicache_events`` (the
planned-pass caller of ``writing_check``) never ran again on PP1. Before 6218d74b35 the forwarded
flush itself ran ``flush_cache`` on every follower, whose #1470 block joined the write-throughs
(bb82fbcb68 boot ...09290020: #1465 on PP1 80x / PP2 80x, zero write-through blockers); since
6218d74b35 the follower parks the flush until PP0 passes it -- and PP0 never passes while PP1 votes
not-idle. The vote now polls the follower's finished write-through acks (rank-local, #737) before it
attaches the slot.

The ring below drives the real ``_weg2_vote_attach_own_slot`` and the real
``UnifiedRadixCache.writing_check`` on a follower stand-in: red on 424346f693 (every lap votes
``hicache_write_through(1)``), green with the drain.
"""
from __future__ import annotations

import inspect
import types

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

from sglang.srt.managers import scheduler_pp_mixin as mix  # noqa: E402
from sglang.srt.managers.weg2_idle_vote import Weg2IdleVoteReq, tally  # noqa: E402
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from sglang.srt.weg2 import p_row_authority  # noqa: E402


class _Event:
    def __init__(self, done=True):
        self.done = done

    def query(self):
        return self.done

    def synchronize(self):
        assert self.done, "a non-blocking drain must never synchronize an unfinished event"

    def elapsed_time(self, other):
        return 1.0


class _Tree:
    """The #737 tree's write-through state; ``writing_check`` and ``_count_ready_acks``
    are UnifiedRadixCache's own, bound to this stand-in."""

    writing_check = UnifiedRadixCache.writing_check
    _count_ready_acks = UnifiedRadixCache._count_ready_acks

    def __init__(self, finished=True, ack_id=10):
        # PP1 09:59:40: one RETAIN-PUBLISH write-through in flight (ack node=10)
        self.ongoing_write_through = {ack_id: (object(), None, [object()])}
        self.cache_controller = types.SimpleNamespace(
            ack_write_queue=[(_Event(True), _Event(finished), [ack_id])])
        self._drain_depth_every = 0
        self.pp_rank = 1
        self.acked = []

    def _finish_write_through_ack(self, ack_id):
        self.ongoing_write_through.pop(ack_id)
        self.acked.append(ack_id)


def _follower(pp_rank=1, drained=True, finished=True):
    s = types.SimpleNamespace()
    s.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=3, attn_tp_size=1)
    s.enable_hicache_storage = False
    s.tree_cache = _Tree(finished)
    s._pp_microbatches_drained = lambda: drained
    tc = s.tree_cache
    s.is_fully_idle = lambda: len(tc.ongoing_write_through) == 0
    s.idle_blockers = lambda: ([f"hicache_write_through({len(tc.ongoing_write_through)})"]
                               if tc.ongoing_write_through else [])
    return s


def _lap(s, epoch):
    vote = Weg2IdleVoteReq(epoch=epoch, origin=0, world=3, slots=[(0, 1, "none")])
    mix.SchedulerPPMixin._weg2_vote_attach_own_slot(s, vote)
    return vote


def test_the_z30x2_ring_a_parked_flush_follower_votes_idle_once_its_write_landed(monkeypatch):
    """The metal: PP1 holds one finished write-through, no planned pass and no flush
    runs on it any more; three laps of the #1268 vote. 424346f693: all three vote
    hicache_write_through(1) (the 14746 lines); fixed: the first lap drains it."""
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    s = _follower()
    slots = []
    for epoch in (2, 3, 4):
        vote = _lap(s, epoch)
        slots.append([sl for sl in vote.slots if sl[0] == 1][0])
    assert slots[0] == (1, 1, "none"), slots
    assert all(sl[1] == 1 for sl in slots), slots
    assert s.tree_cache.acked == [10]
    assert s.tree_cache.cache_controller.ack_write_queue == []


def test_an_unfinished_write_through_still_blocks_and_is_not_waited_on(monkeypatch):
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    s = _follower(finished=False)
    vote = _lap(s, 2)
    assert (1, 0, "hicache_write_through(1)") in vote.slots
    assert tally(vote).blocking_ranks == (1,)
    assert s.tree_cache.acked == []


def test_pp0_undrained_or_no_row_authority_do_not_poll(monkeypatch):
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    for s in (_follower(pp_rank=0), _follower(drained=False)):
        assert mix.weg2_writethrough_drain_on_idle_vote(s) == 0
        assert len(s.tree_cache.ongoing_write_through) == 1
    monkeypatch.setattr(p_row_authority, "applies", lambda s: False)
    s = _follower()
    assert mix.weg2_writethrough_drain_on_idle_vote(s) == 0
    assert len(s.tree_cache.ongoing_write_through) == 1


def test_a_legacy_tree_with_a_collective_writing_check_is_never_polled(monkeypatch):
    """HiRadixCache.writing_check carries a MIN all_reduce; a rank-local trigger of it
    is the #580 class. Only the #737 tree (``_count_ready_acks``) is polled."""
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    s = _follower()
    calls = []
    s.tree_cache = types.SimpleNamespace(ongoing_write_through={10: object()},
                                         writing_check=lambda: calls.append(1))
    assert mix.weg2_writethrough_drain_on_idle_vote(s) == 0 and calls == []


def test_a_stand_in_without_a_tree_returns_at_once():
    assert mix.weg2_writethrough_drain_on_idle_vote(types.SimpleNamespace()) == 0


def test_the_vote_drains_before_it_attaches_the_slot():
    src = inspect.getsource(mix.SchedulerPPMixin._weg2_vote_attach_own_slot)
    i = src.index("weg2_writethrough_drain_on_idle_vote(self)")
    assert i < src.index("attach_slot(vote, rank, self.is_fully_idle()")
