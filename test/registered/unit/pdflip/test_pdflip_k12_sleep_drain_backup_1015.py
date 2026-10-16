"""K12 SLEEP-DRAIN-W120 (NF h16p 226aad7105, D 15.10.2026 13:35:33Z -> 13:35:43Z).

THE DEATH, boot ..._226aad7105_1015_124245, D.log:

* 271593 TP0 ``not-idle because: hicache_prefetch(1: pdflip-68-289), hicache_backup(8)``
  (TP1/TP2: the prefetch only);
* 271625-271645 ``Exception in thread Thread-113 (backup_thread_func)`` ...
  ``hybrid_cache_controller.py:1528 _page_backup`` -> ``hicache_storage.py:5782
  batch_set_v2`` -> ``memory_pool_host.py:1600 get_data_page``:
  ``IndexError: index 72085 is out of bounds for dimension 0 with size 6549``
  (the ``qsa_indexer`` mirror has 6549 pages, D.log 5589);
* 271676 TP0 ``PDFLIP-FLUSH-NONBLOCK quiesced ... (this rank hicache_backup(2))``;
* 271707-271709 ``PDFLIP SLEEP-DRAIN: waited 10.01 s (891 group polls) ... TP0 at
  entry ['hicache_backup(2)'], now ['hicache_backup(2)']; group verdict
  SleepDrainVerdict(hicache_terms=1, other_terms=0, expired=True)``; W120 on 3/3,
  W29, GROUP-FENCE FAILED 0/3, RANK_EXCEPTION.

The thread that would have acknowledged the two backups was dead since
13:35:33; every earlier drain of the boot with the same blocker took <= 0.13 s.

RED on 226aad7105: the thread-survival case (the thread dies, no ack), the
group sleep case (W120 3/3), the failed-ack case (names absent). GREEN after
K12. The negatives (a live thread writing, a non-backup term, the switch off)
keep the W120 stop on both sides by design.
"""
from __future__ import annotations

import inspect
import os
import queue
import threading
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from flliper.test.test_utils import maybe_stub_sgl_kernel  # noqa: E402

maybe_stub_sgl_kernel()

from flliper.srt.disaggregation.utils import DisaggregationMode  # noqa: E402
from flliper.srt.managers import cache_controller as cc_mod  # noqa: E402
from flliper.srt.managers.scheduler import Scheduler  # noqa: E402
from flliper.srt.managers.scheduler_components.weight_updater import (  # noqa: E402
    SchedulerWeightUpdaterManager,
)

INDEX_ERROR = "index 72085 is out of bounds for dimension 0 with size 6549"
_COLLECTIVE_TIMEOUT_S = 5.0


# ------------------------------------------------------------ the backup thread
class _Ctl:
    """The fields ``backup_thread_func`` reads; the methods are the REAL ones."""

    backup_thread_func = cc_mod.HiCacheController.backup_thread_func

    def __init__(self, page_backup):
        self.storage_stop_event = threading.Event()
        self.backup_queue = queue.Queue()
        self.ack_backup_queue = queue.Queue()
        self.backup_skip = False
        self.enable_storage = True
        self.storage_backend = SimpleNamespace(check_disk_space=lambda: None)
        self._page_backup = page_backup



def _op(op_id):
    return SimpleNamespace(id=op_id, hash_value=["h"] * 2, request_id=f"pdflip-68-{op_id}",
                           completed_tokens=0)


def test_backup_thread_survives_the_index_error(monkeypatch):
    """RED on 226aad7105: the first operation's IndexError ends the thread, the
    second operation is never written and neither is acknowledged."""
    monkeypatch.setattr(cc_mod, "operation_is_stale", lambda *a, **k: False)
    written = []

    def page_backup(op):
        if op.id == 1:
            raise IndexError(INDEX_ERROR)
        written.append(op.id)
        op.completed_tokens = 128

    ctl = _Ctl(page_backup)
    t = threading.Thread(target=ctl.backup_thread_func, daemon=True)
    t.start()
    ctl.backup_queue.put(_op(1))
    ctl.backup_queue.put(_op(2))
    acks = []
    deadline = time.monotonic() + 5
    while len(acks) < 2 and time.monotonic() < deadline:
        try:
            acks.append(ctl.ack_backup_queue.get(timeout=0.1))
        except queue.Empty:
            pass
    ctl.storage_stop_event.set()
    assert [a.id for a in acks] == [1, 2], f"acks={[a.id for a in acks]} thread_alive={t.is_alive()}"
    assert written == [2]
    from flliper.srt.mem_cache import pdflip_backup_guard as g

    assert g.is_failed(acks[0]) and INDEX_ERROR in acks[0].pdflip_backup_failed
    assert not g.is_failed(acks[1])
    t.join(timeout=3)


def test_failed_ack_never_sets_l3_present():
    """#1317 C2: the ack drain releases a FAILED operation's host lock without
    ``l3_present`` and without the rebind / transit branches."""
    from flliper.srt.mem_cache import unified_radix_cache as urc

    src = inspect.getsource(urc)
    i = src.index("def _drain_backup():")
    blk = src[i:src.index("def _drain_release", i)]
    assert "_k12_guard.is_failed(operation)" in blk
    assert blk.index("_k12_guard.is_failed(operation)") < blk.index("node.l3_present = True")
    failed_arm = blk[blk.index("_k12_guard.is_failed(operation)"):blk.index("if entry is not None:\n", blk.index("_k12_guard.is_failed(operation)"))]
    assert "dec_host_lock_ref" in failed_arm and "entry = None" in failed_arm
    assert "l3_present" not in failed_arm.split("#")[0]


def test_abandoned_operation_is_acked_unwritten(monkeypatch):
    """A queued operation the group abandoned is acked FAILED at dequeue and
    never reaches ``_page_backup``."""
    from flliper.srt.mem_cache import pdflip_backup_guard as g

    monkeypatch.setattr(cc_mod, "operation_is_stale", lambda *a, **k: False)
    written = []
    ctl = _Ctl(lambda op: written.append(op.id))
    tree = SimpleNamespace(
        ongoing_backup={7: (SimpleNamespace(), None)},
        cache_controller=ctl,
        dec_host_lock_ref=lambda node, lp: None,
        staging_write_ring=None,
    )
    assert g.abandon(tree, why="test") == (1, 0)
    assert tree.ongoing_backup == {}
    t = threading.Thread(target=ctl.backup_thread_func, daemon=True)
    t.start()
    ctl.backup_queue.put(_op(7))
    ack = ctl.ack_backup_queue.get(timeout=5)
    ctl.storage_stop_event.set()
    assert ack.id == 7 and g.is_failed(ack) and written == []
    t.join(timeout=3)


# ----------------------------------------------------- the group sleep drain
class _Group:
    def __init__(self, world):
        self._slots = [None] * world
        self._out = None
        self._b1 = threading.Barrier(world, timeout=_COLLECTIVE_TIMEOUT_S)
        self._b2 = threading.Barrier(world, timeout=_COLLECTIVE_TIMEOUT_S)

    def all_reduce(self, rank, values, op):
        self._slots[rank] = list(values)
        if self._b1.wait() == 0:
            self._out = [op(col) for col in zip(*self._slots)]
        self._b2.wait()
        return list(self._out)


class _Thread:
    def __init__(self, alive):
        self._alive = alive

    def is_alive(self):
        return self._alive


class _RankTree:
    enable_storage = True

    def __init__(self, group, rank, *, backups=(), executing=None, alive=False, prefetch=()):
        self.group = group
        self.rank = rank
        self.ongoing_write_through = {}
        self.ongoing_load_back = {}
        self.ongoing_prefetch = {r: SimpleNamespace(rid=r) for r in prefetch}
        self.nodes = {i: SimpleNamespace(l3_present=False, host_lock=1) for i in backups}
        self.ongoing_backup = {i: (self.nodes[i], ("lp", i)) for i in backups}
        self.released = []
        self.staging_write_ring = None
        self.cache_controller = SimpleNamespace(backup_thread=_Thread(alive))
        if executing is not None:
            self.cache_controller._k12_backup_guard = {
                "lock": threading.Lock(), "executing": executing, "cancelled": set()}
        self.polls = 0

    def dec_host_lock_ref(self, node, lock_params):
        node.host_lock -= 1
        self.released.append(lock_params)

    def check_hicache_events(self):
        # the dead thread acknowledges nothing: polling cannot drain it
        self.polls += 1
        self.group.all_reduce(self.rank, [0], max)

    def hicache_group_max(self, values, *, label):
        return self.group.all_reduce(self.rank, values, max)


class _RankSched:
    idle_blockers = Scheduler.idle_blockers
    is_fully_idle = Scheduler.is_fully_idle

    def __init__(self, tree):
        self.running_batch = SimpleNamespace(reqs=[], is_empty=lambda: True)
        self.waiting_queue = []
        self.last_batch = None
        self.enable_overlap = False
        self.result_queue = []
        self.chunked_req = None
        self.anchor_tails = []
        self.server_args = SimpleNamespace()
        self.pdflip_dormant = False
        self.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        self.kv_session_offload = None
        self.grammar_manager = SimpleNamespace(grammar_queue=[])
        self.disaggregation_mode = DisaggregationMode.NULL
        self.enable_hisparse = False
        self.enable_hierarchical_cache = True
        self.tree_cache = tree

    def _pp_microbatches_drained(self):
        return True


def _run_group(trees, bound_s=0.05):
    out = [None] * len(trees)
    scheds = [_RankSched(t) for t in trees]

    def main(r):
        wu = SchedulerWeightUpdaterManager.__new__(SchedulerWeightUpdaterManager)
        wu.scheduler = scheds[r]
        wu.is_fully_idle = scheds[r].is_fully_idle
        try:
            wu._pdflip_drain_hicache_before_sleep(bound_s=bound_s)
            out[r] = "slept"
        except Exception as exc:  # noqa: BLE001 -- the outcome IS the finding
            out[r] = f"{type(exc).__name__}: {exc}"

    th = [threading.Thread(target=main, args=(r,)) for r in range(len(trees))]
    for t in th:
        t.start()
    for t in th:
        t.join(timeout=30)
        assert not t.is_alive(), "a rank never left the sleep drain"
    return out


def _h16p_group(**tp0):
    g = _Group(3)
    kw = dict(backups=(41, 42), alive=False)
    kw.update(tp0)
    return [_RankTree(g, 0, **kw), _RankTree(g, 1), _RankTree(g, 2)]


def test_h16p_dead_consumer_the_group_abandons_and_sleeps(monkeypatch):
    """RED on 226aad7105: 'W120 PdFlipSleepDrainRefused ... hicache_terms=1
    other_terms=0 expired=True ... blocks on [hicache_backup(2)]' on 3/3."""
    monkeypatch.delenv("FLLIPER_PDFLIP_SLEEP_ABANDON_BACKUP", raising=False)
    trees = _h16p_group()
    out = _run_group(trees)
    assert out == ["slept"] * 3, out
    tp0 = trees[0]
    assert tp0.ongoing_backup == {}
    assert sorted(tp0.released) == [("lp", 41), ("lp", 42)]
    assert all(not n.l3_present and n.host_lock == 0 for n in tp0.nodes.values())
    # every rank polled exactly as often (x105 law)
    assert len({t.polls for t in trees}) == 1


def test_live_thread_writing_keeps_the_stop(monkeypatch):
    """An operation EXECUTING on a live thread is never abandoned (its host-row
    reads would race the reset): W120 on 3/3, exactly as before K12."""
    monkeypatch.delenv("FLLIPER_PDFLIP_SLEEP_ABANDON_BACKUP", raising=False)
    trees = _h16p_group(executing=41, alive=True)
    out = _run_group(trees)
    assert all(o.startswith("PdFlipSleepDrainRefused: W120") for o in out), out
    assert set(trees[0].ongoing_backup) == {41, 42}


def test_a_non_backup_term_keeps_the_stop(monkeypatch):
    """A prefetch term left anywhere in the group is not a backup: no abandon."""
    monkeypatch.delenv("FLLIPER_PDFLIP_SLEEP_ABANDON_BACKUP", raising=False)
    g = _Group(3)
    trees = [_RankTree(g, 0, backups=(41, 42)), _RankTree(g, 1, prefetch=("pdflip-68-289",)),
             _RankTree(g, 2)]
    out = _run_group(trees)
    assert all(o.startswith("PdFlipSleepDrainRefused: W120") for o in out), out
    assert set(trees[0].ongoing_backup) == {41, 42}


def test_a_rank_without_a_controller_votes_executing(monkeypatch):
    """No controller to ask (a desk double, x105's shape): unknown is not
    'nobody is writing' -- the group keeps the W120 stop."""
    monkeypatch.delenv("FLLIPER_PDFLIP_SLEEP_ABANDON_BACKUP", raising=False)
    trees = _h16p_group()
    trees[0].cache_controller = None
    out = _run_group(trees)
    assert all(o.startswith("PdFlipSleepDrainRefused: W120") for o in out), out


def test_switch_off_is_the_old_stop(monkeypatch):
    monkeypatch.setenv("FLLIPER_PDFLIP_SLEEP_ABANDON_BACKUP", "0")
    trees = _h16p_group()
    out = _run_group(trees)
    assert all(o.startswith("PdFlipSleepDrainRefused: W120") for o in out), out


def test_verdict_pure():
    from flliper.srt.managers.pdflip_sleep_drain import (
        SleepDrainVerdict,
        local_drain_terms_k12,
        verdict_from_group,
    )

    # TP0 of the death: backups only, nothing executing, bound expired
    tp0 = local_drain_terms_k12(["hicache_backup(2)"], expired=True, backup_executing=0)
    tp1 = local_drain_terms_k12([], expired=True, backup_executing=0)
    v = verdict_from_group([max(c) for c in zip(tp0, tp1, tp1)])
    assert v == SleepDrainVerdict(hicache_terms=1, other_terms=0, expired=True,
                                  nonbackup_hicache_terms=0, backup_executing=0)
    assert v.abandonable
    # not expired yet -> keep polling, never abandon early
    assert not verdict_from_group([1, 0, 0, 0, 0]).abandonable
    # old 3-term reading never abandons
    assert not verdict_from_group([1, 0, 1]).abandonable
