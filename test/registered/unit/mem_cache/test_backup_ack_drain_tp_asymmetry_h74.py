"""fnFL2 H74 (x172): a TP rank with one store write more than its peers still drains it.

Hermetic (no CUDA). Metal x172 (D, Form A TP3: TP0 host with the arena staging pool,
TP1/TP2 workers), 24.09. 23:37:33: D finished a 6144-token answer (rid weg2-8-13,
prompt 107) and the front, flipping to P, polled /flush_cache. The flush publish issued
7 write-throughs on TP0 and 6 on TP1/TP2 (``#1470 FLUSH-PUBLISH issued=7`` / ``issued=6``,
sweeps 3+3+1 against 3+2+1): TP0 alone split the answer's 6080-token node
(``PUBLISH-SPLIT n=1 node=105 pieces=2 window=4096 tail_tokens=1984``) -- the publish
window is a quarter of the rank's OWN host pool, 4096 on TP0 (arena staging pool, 4096
rows) and 86,016 on the workers (353,600-row host pool). Every write-through ack queued one
store write. The steady-state drain (``UnifiedRadixCache.drain_storage_control_queues``)
takes the MIN of the three ack-queue sizes over the TP group; TP0's seventh ack was never
drained, ``ongoing_backup`` kept it, ``not-idle because: hicache_backup(1)`` answered every
flush poll from 23:37:34 on and the front stopped with W3 at 23:39:03. (The #1421
``mamba_pin`` refusal at 23:37:33, TP0 only, moved one node to the next sweep and changed
no count; the #801 abort at 23:37:35 hit a request that had already finished.)

What these cases hold:

* the metal path: three ranks with the x172 host pools run the REAL publish window and
  the REAL drain over a real three-rank MIN reduce (one thread per rank, a barrier), sweep
  after sweep as on metal, then idle passes: every rank's ``ongoing_backup`` empties and
  ``Scheduler.idle_blockers`` names nothing -- RED on c281ce577a (TP0: hicache_backup(1));
* the window is the same on every rank of a TP group; a single-rank group (P's stages)
  keeps the quarter-of-the-pool window;
* the rank-local drain alone empties asymmetric counts (7/6/6 forced); switched off the
  x172 livelock comes back, named;
* revokes and host releases still drain the group MIN;
* x174 (tree dbf3db8a38, D arm SGLANG_HICACHE_DRAIN_AGREE_EVERY=8), the mirror image:
  TP1/TP2 carried two store writes TP0 did not -- a store-loaded span keeps its host
  rows on TP0 (arena rows, #1424) and releases them on the workers (plain host pool,
  #1408 transit release), so when the burst evicted two such nodes (``#1469 EVICT
  node=161/162`` on all three ranks) only the workers had to write them back first
  (``#969H BACKUP n=21/22`` + ``#1233 STORE-WRITE``). MIN(0, 2, 2) = 0 at every
  agreement: ``hicache_backup(2)`` on TP1/TP2 for 4944 flush polls, then W3. The
  cadence is not the cause -- the rounds kept advancing (``#1028 HICACHE-ROUND`` 9748
  -> 11865 on TP1 in that minute) and every eighth one agreed on 0.
"""

import itertools
import threading
from queue import Queue
from types import SimpleNamespace

import pytest
import torch

PAGE = 64
X172_POOLS = (4096, 353600, 353600)  # TP0 arena staging rows, TP1/TP2 host rows
ANSWER_NODE = 6080  # the answer's node: 6144 retained - the 64-token first page node
OTHER_NODES = 5  # the flush sweep's other un-backed nodes (TP1/TP2: unbacked=6 = 5 + 1)
_IDS = itertools.count(1000)


class _Group:
    """The TP group's MIN reduce, for real: every rank's thread contributes its
    tensor and gets the elementwise minimum of all three."""

    def __init__(self, world: int):
        self.world = world
        self.barrier = threading.Barrier(world)
        self.slots = [None] * world

    def min(self, idx: int, t: torch.Tensor) -> None:
        self.slots[idx] = t.clone()
        self.barrier.wait()
        m = torch.stack(self.slots).min(dim=0).values
        self.barrier.wait()
        t.copy_(m)


class _Rank:
    """One TP rank's tree cache: the REAL drain, window and world helpers on the
    state they touch (its queues, its ongoing_backup, its host pool)."""

    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache as _U

    drain_storage_control_queues = _U.drain_storage_control_queues
    _drain_storage_control_queues_impl = _U._drain_storage_control_queues_impl
    _weg2_publish_window = _U._weg2_publish_window
    _attn_reduce_world = _U._attn_reduce_world
    page_size = PAGE
    enable_storage_metrics = False
    storage_metrics_collector = None
    staging_write_ring = None

    def __init__(self, idx: int, group: _Group, host_rows: int):
        self.idx, self.group = idx, group
        self.attn_cp_group = self.attn_tp_group = None
        self.tp_world_size = group.world
        self.released = []
        self.cache_controller = SimpleNamespace(
            prefetch_revoke_queue=Queue(), ack_backup_queue=Queue(), host_mem_release_queue=Queue(),
            extra_host_mem_release_queues={},
            mem_pool_host=SimpleNamespace(size=host_rows, free=lambda idx: self.released.append(idx)),
        )
        self.ongoing_backup = {}
        self.ongoing_prefetch = {}

    def _all_reduce_attn_groups(self, t, op, label=""):
        assert op == torch.distributed.ReduceOp.MIN
        self.group.min(self.idx, t)

    def dec_host_lock_ref(self, node, params):
        pass

    def _weg2_rebind_host_to_arena(self, node):
        return False

    def _weg2_host_is_transit(self):
        return False

    def _weg2_release_chain_piece_host(self, node):
        pass

    def store_write_acked(self) -> None:
        """One node's store write, issued at its write-through ack
        (``_weg2_write_plain_sidecars`` / ``write_backup_storage`` register it
        in ``ongoing_backup``) and acked by the backup thread."""
        op = SimpleNamespace(id=next(_IDS), completed_tokens=0)
        self.ongoing_backup[op.id] = (SimpleNamespace(l3_present=False), None)
        self.cache_controller.ack_backup_queue.put(op)


def _x172():
    group = _Group(3)
    return [_Rank(i, group, rows) for i, rows in enumerate(X172_POOLS)]


def _pass(ranks) -> None:
    """One scheduler pass on every rank: check_hicache_events -> the drain."""
    threads = [threading.Thread(target=r.drain_storage_control_queues) for r in ranks]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
        assert not t.is_alive(), "the three-rank reduce did not complete"


def _blockers(rank) -> list:
    """The REAL Scheduler.idle_blockers of a rank that is otherwise idle."""
    from sglang.srt.managers.scheduler import Scheduler

    s = SimpleNamespace(
        running_batch=SimpleNamespace(is_empty=lambda: True), chunked_req=None, anchor_tails=None,
        dllm_manager=SimpleNamespace(any_staging_reqs=lambda: False), last_batch=None, enable_overlap=False,
        result_queue=[], kv_session_offload=None, waiting_queue=[],
        grammar_manager=SimpleNamespace(grammar_queue=[]), enable_hierarchical_cache=True,
        tree_cache=SimpleNamespace(ongoing_write_through={}, ongoing_load_back={}, enable_storage=True,
                                   ongoing_prefetch={}, ongoing_backup=rank.ongoing_backup),
        _pp_microbatches_drained=lambda: True,
    )
    return Scheduler.idle_blockers(s)


def _flush_publish(ranks) -> list:
    """x172's flush publish: the answer's node split by each rank's own window,
    issued over three sweeps (3, total-4, 1) with a drain pass after each, then
    the idle passes of the polling front. Returns the per-rank store writes."""
    totals = []
    for r in ranks:
        w = r._weg2_publish_window()
        totals.append(OTHER_NODES + (-(-ANSWER_NODE // w) if 0 < w < ANSWER_NODE else 1))
    for sweep in range(3):
        for r, total in zip(ranks, totals):
            for _ in range((3, total - 4, 1)[sweep]):
                r.store_write_acked()
        _pass(ranks)
    for _ in range(4):
        _pass(ranks)
    return totals


def test_d_flush_after_a_long_answer_drains_every_rank():
    """x172, RED on c281ce577a: TP0 keeps hicache_backup(1) for every pass."""
    ranks = _x172()
    _flush_publish(ranks)
    left = [len(r.ongoing_backup) for r in ranks]
    assert left == [0, 0, 0], (
        f"ongoing_backup per rank after the flush publish and 4 idle passes: {left} -- "
        f"TP0 {_blockers(ranks[0])} (x172: 'not-idle because: hicache_backup(1)' "
        f"from 23:37:34 to W3 at 23:39:03)")
    assert all(_blockers(r) == [] for r in ranks)


def test_the_publish_window_is_the_same_on_every_tp_rank():
    ranks = _x172()
    assert [r._weg2_publish_window() for r in ranks] == [4096, 4096, 4096]
    # ... so every rank splits the answer's node alike: 7 store writes each
    assert _flush_publish(_x172()) == [7, 7, 7]


def test_a_single_rank_group_keeps_its_pool_window():
    """Default path: P's stages (attn world 1) keep a quarter of their pool."""
    solo = _Group(1)
    assert _Rank(0, solo, 6976)._weg2_publish_window() == 4096  # P's arena staging rows
    assert _Rank(0, solo, 353600)._weg2_publish_window() == 86016
    assert _Rank(0, solo, 54254)._weg2_publish_window() == 12288  # the #1407 pool (xsn148)


def _asymmetric(ranks, counts) -> None:
    for r, n in zip(ranks, counts):
        for _ in range(n):
            r.store_write_acked()


def test_the_rank_local_drain_empties_asymmetric_counts():
    ranks = _x172()
    _asymmetric(ranks, (7, 6, 6))
    _pass(ranks)
    assert [len(r.ongoing_backup) for r in ranks] == [0, 0, 0]


def test_switched_off_the_x172_livelock_is_back(caplog):
    """SGLANG_WEG2_ENABLE_LOCAL_BACKUP_ACK_DRAIN=0 restores the MIN for the
    backup acks -- and with it the surplus that never drains."""
    from sglang.srt.environ import envs

    with envs.SGLANG_WEG2_ENABLE_LOCAL_BACKUP_ACK_DRAIN.override(False):
        ranks = _x172()
        _asymmetric(ranks, (7, 6, 6))
        for _ in range(5):
            _pass(ranks)
    assert [len(r.ongoing_backup) for r in ranks] == [1, 0, 0]
    assert _blockers(ranks[0]) == ["hicache_backup(1)"]


def test_the_surplus_is_named(caplog):
    import logging

    ranks = _x172()
    _asymmetric(ranks, (1, 0, 0))
    with caplog.at_level(logging.INFO, logger="sglang.srt.mem_cache.unified_radix_cache"):
        _pass(ranks)
    assert "H74 BACKUP-ACK DRAIN rank-local: acks=1 group_min=0 ongoing_backup=1" in caplog.text
    assert not ranks[0].ongoing_backup


def test_revokes_and_releases_keep_the_group_min():
    """The prefetch revokes and the host releases stay in lockstep: a rank
    with more of them drains only the group minimum per pass."""
    ranks = _x172()
    for r, n in zip(ranks, (3, 1, 1)):
        for i in range(n):
            r.cache_controller.host_mem_release_queue.put(torch.tensor([10 * r.idx + i]))
    _pass(ranks)
    assert [r.cache_controller.host_mem_release_queue.qsize() for r in ranks] == [2, 0, 0]


def _rounds(ranks, n: int, every: int) -> None:
    """``n`` scheduler rounds through the agreement cadence of the tree (x174's
    ``_gated_drain_storage_control_queues``) when it has one, else plain drains."""
    gated = getattr(_Rank._U, "_gated_drain_storage_control_queues", None)
    for _ in range(n):
        if gated is None:
            _pass(ranks)
            continue
        threads = [threading.Thread(target=gated, args=(r, every)) for r in ranks]
        for t in threads:
            t.start()
        for t in threads:
            t.join(10)
            assert not t.is_alive(), "the three-rank reduce did not complete"


def test_x174_the_workers_surplus_drains_on_the_agreement_cadence():
    """x174, RED on dbf3db8a38 (and c281ce577a): TP1/TP2 keep hicache_backup(2)."""
    ranks = _x172()
    _asymmetric(ranks, (0, 2, 2))
    _rounds(ranks, 24, every=8)  # three agreements at the x174 cadence
    left = [len(r.ongoing_backup) for r in ranks]
    assert left == [0, 0, 0], (
        f"ongoing_backup per rank after 24 rounds at DRAIN_AGREE_EVERY=8: {left} -- "
        f"TP1 {_blockers(ranks[1])} (x174: TP1/TP2 'hicache_backup(2)', TP0 'blockers=[none]', "
        f"4944 refused flush polls from 00:24:34 to W3)")


def test_x174_switched_off_the_workers_stay_stuck():
    from sglang.srt.environ import envs

    with envs.SGLANG_WEG2_ENABLE_LOCAL_BACKUP_ACK_DRAIN.override(False):
        ranks = _x172()
        _asymmetric(ranks, (0, 2, 2))
        _rounds(ranks, 24, every=8)
    assert [len(r.ongoing_backup) for r in ranks] == [0, 2, 2]
    assert _blockers(ranks[1]) == ["hicache_backup(2)"] and _blockers(ranks[0]) == []
