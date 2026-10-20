"""Storage-backup liveness (NF h16l 75e3ee8abd, boot
dkrnfint4h6ablxcw3spmlrhlvwbar1dauer10142044 ..._1014_204509, W3 at 20:50:22).

P.log 20:48:52, the P->D quiesce after pdflip-0-1 (the only request of the boot):
PP1/PP2 ``PDFLIP-WRITETHROUGH-DRAIN-ON-VOTE drained=3`` -> per ack ``#1427 DIRECT-WRITE
ACK`` + ``#106S SIDECAR-WRITE-ISSUED op=1..3 pools=['qsa_indexer']`` -> from then on
every #1268 lap (epoch 2 .. 6697) ``[PP1] PDFLIP-P-IDLE-VERDICT idle=False
blocking_rank=1 blockers=[hicache_backup(3)]``, PP0 ``Cache not flushed ... ranks that
voted NOT idle: [1, 2]``, ~100 x /flush_cache 400, front ``PDFLIP STOP W3
PdFlipDrainWitnessDisagreement -- front ledger [] (outstanding [])``. ``#1028
HICACHE-ROUND`` ran n=1 on PP1/PP2 in the whole boot: under the row authority
(h16j 31aa655fff turned it ON for nextflash) a frameless follower cycle skips the plan,
so the round drain that retires storage-backup acks (``check_hicache_events`` ->
``drain_storage_control_queues`` -> ``_drain_backup``) never ran again on a follower.

The ring below drives the real ``_pdflip_vote_attach_own_slot``, the real
``UnifiedRadixCache.writing_check`` and the real ``_drain_storage_control_queues_impl``
on a follower stand-in whose write-through ack issues the sidecar backup the way
``_pdflip_write_plain_sidecars`` does: red on 75e3ee8abd (every lap votes
``hicache_backup(3)``), green with ``pdflip_backup_ack_drain_on_idle_vote``.
"""
from __future__ import annotations

import inspect
import itertools
import types
from queue import Queue

from flliper.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")

from flliper.srt.managers import scheduler_pp_mixin as mix  # noqa: E402
from flliper.srt.managers.pdflip_idle_vote import PdFlipIdleVoteReq, tally  # noqa: E402
from flliper.srt.mem_cache.unified_radix_cache import UnifiedRadixCache  # noqa: E402
from flliper.srt.pdflip import p_row_authority  # noqa: E402


class _Event:
    def __init__(self, done=True):
        self.done = done

    def query(self):
        return self.done

    def synchronize(self):
        assert self.done

    def elapsed_time(self, other):
        return 1.0


class _Node:
    def __init__(self, nid):
        self.id = nid
        self.l3_present = False


class _Op:
    def __init__(self, oid):
        self.id = oid
        self.completed_tokens = 64


class _Tree:
    """A follower's #737 tree: real ``writing_check`` / ``_count_ready_acks`` /
    ``drain_backup_acks_uncoordinated`` / ``_drain_storage_control_queues_impl``.
    ``_finish_write_through_ack`` stands for ``_pdflip_direct_complete`` +
    ``_pdflip_write_plain_sidecars``: the arena slot completes and ONE store write
    of the QSA-index sidecar is issued into ``ongoing_backup``; ``store_acks``
    decides whether the backup thread has acked it by the time of the next poll
    (measured 20:48:52: within the same millisecond window)."""

    writing_check = UnifiedRadixCache.writing_check
    _count_ready_acks = UnifiedRadixCache._count_ready_acks
    # getattr: on 75e3ee8abd the method does not exist and the ring runs the
    # unfixed vote (red by behaviour, not by an import error)
    drain_backup_acks_uncoordinated = getattr(
        UnifiedRadixCache, "drain_backup_acks_uncoordinated", None)
    _drain_storage_control_queues_impl = UnifiedRadixCache._drain_storage_control_queues_impl

    def __init__(self, store_acks=True, collective=False, wt_ids=(19, 20, 21)):
        self.ongoing_write_through = {i: (object(), None, [object()]) for i in wt_ids}
        self.ongoing_backup = {}
        self.ongoing_prefetch = {}
        self.cache_controller = types.SimpleNamespace(
            ack_write_queue=[(_Event(True), _Event(True), [i]) for i in wt_ids],
            ack_backup_queue=Queue(),
            prefetch_revoke_queue=Queue(),
            host_mem_release_queue=Queue(),
            mem_pool_host=types.SimpleNamespace(),
            extra_host_mem_release_queues={},
        )
        self._drain_depth_every = 0
        self.pp_rank = 1
        self.enable_storage_metrics = False
        self.storage_metrics_collector = None
        self.staging_write_ring = None
        self._collective = collective
        self._store_acks = store_acks
        self._ops = itertools.count(1)
        self.unlocked = []
        self.sidecar_ops = []

    def _drain_agreement_is_collective(self):
        return self._collective

    def _finish_write_through_ack(self, ack_id):
        self.ongoing_write_through.pop(ack_id)
        node = _Node(ack_id)
        op = _Op(next(self._ops))  # #106S SIDECAR-WRITE-ISSUED op=n node=ack_id
        self.ongoing_backup[op.id] = (node, ("lock", ack_id))
        self.sidecar_ops.append(op)
        if self._store_acks:
            self.cache_controller.ack_backup_queue.put(op)

    def store_ack_all(self):
        for op in self.sidecar_ops:
            self.cache_controller.ack_backup_queue.put(op)

    def dec_host_lock_ref(self, node, params):
        self.unlocked.append(node.id)

    def _pdflip_rebind_host_to_arena(self, node):
        return False

    def _pdflip_host_is_transit(self):
        return False


def _blockers(tc):
    out = []
    if tc.ongoing_write_through:
        out.append(f"hicache_write_through({len(tc.ongoing_write_through)})")
    if tc.ongoing_backup:
        out.append(f"hicache_backup({len(tc.ongoing_backup)})")
    return out


def _follower(pp_rank=1, drained=True, **tree_kw):
    s = types.SimpleNamespace()
    s.ps = types.SimpleNamespace(pp_rank=pp_rank, pp_size=3, attn_tp_size=1)
    s.enable_hicache_storage = False
    s.tree_cache = _Tree(**tree_kw)
    s._pp_microbatches_drained = lambda: drained
    tc = s.tree_cache
    s.is_fully_idle = lambda: not _blockers(tc)
    s.idle_blockers = lambda: _blockers(tc)
    return s


def _lap(s, epoch):
    vote = PdFlipIdleVoteReq(epoch=epoch, origin=0, world=3, slots=[(0, 1, "none")])
    mix.SchedulerPPMixin._pdflip_vote_attach_own_slot(s, vote)
    return [sl for sl in vote.slots if sl[0] == 1][0], vote


def test_the_204509_ring_a_follower_retires_the_sidecar_backups_its_vote_issued(monkeypatch):
    """The metal: PP1 holds three finished direct write-throughs, plans no pass any
    more; the vote's write-through drain issues three #106S sidecar backups that the
    store acks at once. 75e3ee8abd: every lap votes hicache_backup(3) (the 90 s to
    W3); fixed: the first lap is idle, the acks were processed (lock released,
    node store-present), nothing dropped."""
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    s = _follower()
    slots = [_lap(s, epoch)[0] for epoch in (2, 3, 4)]
    assert slots[0] == (1, 1, "none"), slots
    assert all(sl[1] == 1 for sl in slots), slots
    tc = s.tree_cache
    assert tc.ongoing_backup == {}
    assert tc.cache_controller.ack_backup_queue.empty()
    assert sorted(tc.unlocked) == [19, 20, 21]


def test_a_backup_the_store_has_not_acked_still_blocks_and_is_not_waited_on(monkeypatch):
    """The drain never waits and never forces: an un-acked store write keeps the
    follower not-idle (the W3 witness stays intact); the lap after the ack drains it."""
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    s = _follower(store_acks=False)
    slot, vote = _lap(s, 2)
    assert slot == (1, 0, "hicache_backup(3)")
    assert tally(vote).blocking_ranks == (1,)
    assert len(s.tree_cache.ongoing_backup) == 3 and s.tree_cache.unlocked == []
    s.tree_cache.store_ack_all()
    slot, _ = _lap(s, 3)
    assert slot == (1, 1, "none")


def test_pp0_undrained_or_no_row_authority_do_not_drain(monkeypatch):
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    for s in (_follower(pp_rank=0), _follower(drained=False)):
        tc = s.tree_cache
        tc.writing_check()  # the sidecar backups exist and are acked
        assert len(tc.ongoing_backup) == 3
        assert mix.pdflip_backup_ack_drain_on_idle_vote(s) == 0
        assert len(tc.ongoing_backup) == 3
    monkeypatch.setattr(p_row_authority, "applies", lambda s: False)
    s = _follower()
    s.tree_cache.writing_check()
    assert mix.pdflip_backup_ack_drain_on_idle_vote(s) == 0
    assert len(s.tree_cache.ongoing_backup) == 3


def test_a_tree_whose_round_drain_is_a_group_agreement_is_never_drained_locally(monkeypatch):
    """An attention group > 1 drains by the group's MIN of the queue sizes; a
    rank-local drain from the vote would break that order (#580 class)."""
    monkeypatch.setattr(p_row_authority, "applies", lambda s: True)
    s = _follower(collective=True)
    tc = s.tree_cache
    tc.writing_check()
    assert UnifiedRadixCache.drain_backup_acks_uncoordinated(tc) == -1
    assert mix.pdflip_backup_ack_drain_on_idle_vote(s) == 0
    assert len(tc.ongoing_backup) == 3
    assert tc.cache_controller.ack_backup_queue.qsize() == 3


def test_only_the_backup_ack_queue_is_drained():
    """Revokes and host releases keep their round drain: the uncoordinated call
    asks the impl for backups only."""
    calls = []
    fake = types.SimpleNamespace(
        _drain_agreement_is_collective=lambda: False,
        ongoing_backup={},
        _drain_storage_control_queues_impl=lambda **kw: calls.append(kw),
    )
    assert UnifiedRadixCache.drain_backup_acks_uncoordinated(fake) == 0
    assert calls == [
        dict(n_revoke=0, n_backup=None, n_release=0,
             extra_release_counts=None, log_metrics=False)
    ]


def test_a_stand_in_without_a_tree_returns_at_once():
    assert mix.pdflip_backup_ack_drain_on_idle_vote(types.SimpleNamespace()) == 0


def test_the_vote_drains_backups_after_the_write_through_drain_and_before_the_slot():
    src = inspect.getsource(mix.SchedulerPPMixin._pdflip_vote_attach_own_slot)
    i_wt = src.index("pdflip_writethrough_drain_on_idle_vote(self)")
    i_bk = src.index("pdflip_backup_ack_drain_on_idle_vote(self)")
    assert i_wt < i_bk < src.index("attach_slot(vote, rank, self.is_fully_idle()")
