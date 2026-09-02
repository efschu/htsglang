"""#1068 WEG 1 slice 2 fix 3 (WEG1_BUILD_SPEC_0901 A12.4, review finding (d)):
``HiCacheController.reset()`` terminates every in-flight prefetch operation
BEFORE it joins the storage threads, so the bounded join holds by
construction instead of by luck.

THE DEFECT (round 3 of the slice-2 review, verified on 6fb35a945c). Since
slice 2, ``reset()`` joins ``prefetch_io_aux_thread`` through
``_stop_storage_threads`` with ``join(timeout=10)`` and raises RuntimeError
afterwards. But the aux thread only leaves ``_page_transfer`` once the
operation it is transferring is COMPLETE or TERMINATED, and nothing
terminated it: the batch loop checked no stop event, and the tree's
``_reset_full`` cleared ``ongoing_prefetch`` without ``mark_terminate``. A
39365-token prefetch in flight at the cutover needs ~12.6 s at the spec's own
N5 rate (0.32 s/KiToken), which is longer than the 10 s bound, so
``reset()`` raised out of ``_reset_full`` and the cutover died on that rank.
Before slice 2 the aux thread was not joined at all (#1052, stray thread);
slice 2 turned a wedge into a crash. The prefetch thread had the same shape
on the other queue: it drained every queued operation through a full storage
hit query inside the same bound.

THE TEST IS IN TIMING FORM, deliberately: the transfer stub BLOCKS until it
observes ``mark_terminate`` on its operation (it never returns on its own),
and the storage presence stub of the second operation blocks until the stop
event is set. Only a controller that terminates the in-flight work before
joining can bring both loops home inside the bound. A stub that returned
instantly could not tell a terminating reset from one that just waits (that
is what made the bound-vs-transfer question structurally unanswerable in
test_cache_controller_reset_joins_io_aux_1068.py).

RED on the parent 6fb35a945c: RuntimeError('Failed to stop HiCache storage
threads cleanly.') after the 10 s join bound, because the aux thread never
sees its operation terminated. GREEN after the fix: both loops exit within
milliseconds, and the ONE '#1068 RESET JOIN' line names the bound, the
threads, the terminated and the drained operations and the seconds spent.

SLICE 2 FIX 4 (A12.4 AMENDMENT, review round 4 findings B1 and B1b, verified
on e5b7eb3b79): fix 3 landed on the BASE class only. The serving path never
runs it: ``UnifiedRadixCache`` is attached by ``hybrid_pool_assembler.py``
(six construction sites) to ``HybridCacheController``, whose ``prefetch()``
builds ``hybrid_cache_controller.PrefetchOperation`` -- a class that does
NOT subclass ``managers.cache_controller.PrefetchOperation`` (upstream twin
hierarchy, 0986bed8e2; not reparented). On e5b7eb3b79 the terminate pass
keyed on ``isinstance(op, PrefetchOperation)`` against the base class, so on
the serving controller ``terminated_ops`` was ALWAYS 0, the aux thread never
saw its transfer terminated and ``reset()`` raised after the bound exactly
as before fix 3 (reviewer probe hybrid_timing_probe.py: base returned in
0.00 s, hybrid raised RuntimeError after the 2.0 s bound with
terminated_ops=0). And ``HybridCacheController._storage_hit_query`` is a
full override without the ``is_terminated()`` guard, so a terminated probe
still walked the whole span on the store. The same scenario therefore runs
TWICE below: once on ``HiCacheController`` with the base operation class,
once on ``HybridCacheController`` with the hybrid one. RED on e5b7eb3b79 for
the hybrid half; GREEN after fix 4.

Hermetic: real ``prefetch_thread_func`` and real ``prefetch_io_aux_func`` on
real threads, real ``_storage_hit_query`` (the class's own override) against
a stub backend, only ``_page_transfer`` replaced by the blocking stub. The
restart half of ``reset()`` runs against a recording thread stand-in so no
fresh real loop is started.

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=python python -m pytest \\
        test/registered/unit/managers/test_cache_controller_reset_terminates_inflight_1068.py -q
"""

import inspect
import threading
import time
import types
import unittest
from queue import Queue
from unittest import mock

import torch

from sglang.srt.managers import cache_controller as cc_module
from sglang.srt.managers.cache_controller import HiCacheController, PrefetchOperation
from sglang.srt.mem_cache.hybrid_cache import hybrid_cache_controller as hyb_module
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

PAGE = 64
SPAN = 39365  # the prompt_max of the acceptance population (spec section 5)

HybridCacheController = hyb_module.HybridCacheController
HybridPrefetchOperation = hyb_module.PrefetchOperation


class _FakeThread:
    """Stands in for ``threading.Thread`` inside the controller module during
    the RESTART half of reset(), so no fresh real loop is started."""

    started = []

    def __init__(self, target=None, daemon=None):
        self.target = target
        self.daemon = daemon
        self.started_flag = False

    def start(self):
        self.started_flag = True
        _FakeThread.started.append(self)

    def is_alive(self):
        return self.started_flag

    def join(self, timeout=None):
        pass


def _op(rid: str, base: int, cls=PrefetchOperation):
    """One prefetch operation of the acceptance span; ``cls`` picks the
    hierarchy (base ``PrefetchOperation`` or the hybrid twin, which shares
    the positional signature up to ``prefix_keys``)."""
    tokens = list(range(base, base + SPAN))
    return cls(rid, torch.arange(SPAN, dtype=torch.int64), tokens, None, None)


def _hybrid_op(rid: str, base: int):
    return _op(rid, base, cls=HybridPrefetchOperation)


def _shell(stop_event: threading.Event, cls=HiCacheController):
    cc = cls.__new__(cls)
    cc.enable_storage = True
    cc.storage_stop_event = stop_event
    cc.page_size = PAGE
    cc.prefetch_threshold = 256
    cc.prefetch_sync_groups = []
    cc.write_queue = []
    cc.load_queue = []
    cc.ack_write_queue = []
    cc.ack_load_queue = []
    cc.prefetch_queue = Queue()
    cc.backup_queue = Queue()
    cc.prefetch_revoke_queue = Queue()
    cc.ack_backup_queue = Queue()
    cc.host_mem_release_queue = Queue()
    cc.prefetch_tokens_occupied = 4096
    cc._prefetch_current = None
    cc._prefetch_io_current = None
    cc._prefetch_drained_after_stop = 0
    cc._prefetch_io_drained_after_stop = 0
    cc.append_host_mem_release = lambda *a, **k: None
    # One content hash per page; the hash carries the operation's token base
    # so the storage stub can tell the operations apart.
    cc.get_hash_str = lambda tokens, last_hash, page_size: [
        f"h{tokens[0] // 1_000_000}-{i}" for i in range(len(tokens) // page_size)
    ]
    if cls is not HiCacheController:
        # HybridCacheController, the serving controller: its
        # _storage_hit_query override takes the v1 arm (plain batch_exists)
        # when no non-KV pool is registered and the operation carries no
        # pool transfers; its reset() and _start_storage_threads() touch the
        # extra release queues.
        cc.extra_host_mem_release_entries = []
        cc.extra_host_mem_release_queues = {}
        cc._init_extra_host_mem_release_queues = lambda: None
    return cc


def _hybrid_shell(stop_event: threading.Event):
    return _shell(stop_event, cls=HybridCacheController)


class TestResetTerminatesInFlightPrefetchBeforeJoining(CustomTestCase):
    def setUp(self):
        _FakeThread.started = []
        self.abort = threading.Event()  # tearDown escape for a stuck stub

    def tearDown(self):
        self.abort.set()

    def _drive_three_inflight_and_reset(self, cc, mk_op):
        """The A12.4 scenario on ``cc``: one operation in transfer (the aux
        thread's current op), one in the presence probe (the prefetch
        thread's current op), one still queued; then ``reset()`` under a 2 s
        bound. Every assertion of the contract lives here so the base and
        the serving controller are held to exactly the same bar."""
        stop_event = cc.storage_stop_event

        op_transfer = mk_op("in-transfer", 0)
        op_query = mk_op("in-query", 1_000_000)
        op_queued = mk_op("queued", 2_000_000)

        transfer_entered = threading.Event()
        observed = {}

        def blocking_transfer(operation):
            # The transfer of the acceptance-size span at the N5 rate takes
            # ~12.6 s; this stub takes exactly as long as the controller
            # takes to terminate the operation, and never returns otherwise.
            transfer_entered.set()
            while not operation.is_terminated():
                if self.abort.is_set():
                    return
                time.sleep(0.002)
            observed["terminated_at"] = time.monotonic()

        cc._page_transfer = blocking_transfer

        query_entered = threading.Event()
        asked = []

        def batch_exists(hashes, extra_info=None):
            asked.append(list(hashes))
            if hashes and hashes[0].startswith("h1-"):
                # The second operation's presence probe: parks the prefetch
                # thread INSIDE _storage_hit_query until the stop event.
                query_entered.set()
                while not stop_event.is_set():
                    if self.abort.is_set():
                        return 0
                    time.sleep(0.002)
            return len(hashes)  # full hit

        cc.storage_backend = types.SimpleNamespace(batch_exists=batch_exists)

        cc.prefetch_queue.put(op_transfer)
        cc.prefetch_queue.put(op_query)
        cc.prefetch_queue.put(op_queued)

        cc.backup_thread = threading.Thread(target=stop_event.wait, daemon=True)
        cc.backup_thread.start()
        cc.prefetch_thread = threading.Thread(
            target=cc.prefetch_thread_func, daemon=True
        )
        cc.prefetch_thread.start()
        # reset() replaces prefetch_thread/backup_thread with the restart
        # stand-ins; the OUTGOING real threads are what must be dead after.
        outgoing_prefetch, outgoing_backup = cc.prefetch_thread, cc.backup_thread

        # The real prefetch loop moves op_transfer into the aux thread, which
        # parks in the transfer stub; then parks itself in the storage probe
        # of op_query; op_queued is left in prefetch_queue.
        self.assertTrue(transfer_entered.wait(5.0), "aux thread never entered the transfer")
        self.assertTrue(query_entered.wait(5.0), "prefetch thread never entered the probe")
        self.assertEqual(cc.prefetch_queue.qsize(), 1)
        self.assertIs(cc.prefetch_queue.queue[0], op_queued)

        fake_threading = types.SimpleNamespace(Thread=_FakeThread, Event=threading.Event)
        t0 = time.monotonic()
        # create=True: on the parent the constant does not exist and reset()
        # keeps its hard-coded 10 s join, which is exactly the red form.
        with mock.patch.object(
            cc_module, "STORAGE_THREAD_JOIN_BOUND_S", 2.0, create=True
        ), mock.patch.object(cc_module, "threading", fake_threading), self.assertLogs(
            cc_module.logger, level="INFO"
        ) as logs:
            cc.reset()
        t_reset = time.monotonic() - t0

        # Every in-flight operation was terminated: the one in transfer, the
        # one in the presence probe, and the one still queued.
        self.assertTrue(op_transfer.is_terminated())
        self.assertTrue(op_query.is_terminated())
        self.assertTrue(op_queued.is_terminated())
        # Timing form: the transfer stub saw the termination BEFORE reset()
        # returned, i.e. the join was released by termination, not by luck.
        self.assertIn("terminated_at", observed)
        self.assertLessEqual(observed["terminated_at"], t0 + t_reset)
        self.assertLess(t_reset, 2.0, f"reset took {t_reset:.2f}s")
        # All three outgoing threads are gone (the aux thread is the real one
        # started by prefetch_thread_func; it is not replaced by the restart).
        cc.prefetch_io_aux_thread.join(timeout=1.0)
        self.assertFalse(cc.prefetch_io_aux_thread.is_alive())
        outgoing_prefetch.join(timeout=1.0)
        self.assertFalse(outgoing_prefetch.is_alive())
        outgoing_backup.join(timeout=1.0)
        self.assertFalse(outgoing_backup.is_alive())
        # The pipeline was restarted on fresh stand-ins and the event cleared.
        self.assertFalse(stop_event.is_set())
        self.assertEqual(len(_FakeThread.started), 2)
        self.assertEqual(cc.prefetch_tokens_occupied, 0)
        # The operation drained after the stop (op_queued, token base
        # 2_000_000, hashes 'h2-*') was terminated by the loop and asked the
        # store NOTHING: the probe guard holds on the class under test.
        self.assertFalse(
            any(h and h[0].startswith("h2-") for h in asked),
            "a terminated operation drained after the stop walked the store",
        )

        # ONE line, every term named: bound, threads, terminated, drained,
        # seconds. terminated_ops counts the three distinct operations;
        # drained_ops counts what the loops consumed after the stop event
        # (op_queued through the prefetch thread; the aux loop does not
        # drain prefetch_buffer on stop, upstream form).
        lines = [m for m in logs.output if "#1068 RESET JOIN" in m]
        self.assertEqual(len(lines), 1, logs.output)
        line = lines[0]
        self.assertIn("bound_s=2", line)
        self.assertIn("terminated_ops=3", line)
        self.assertIn("drained_ops=1", line)
        for name in ("prefetch", "backup", "prefetch_io_aux"):
            self.assertIn(name, line)
        self.assertRegex(line, r"joined_s=\d+\.\d\d")

    def test_reset_terminates_the_transfer_the_query_and_the_queue_inside_the_bound(self):
        self._drive_three_inflight_and_reset(_shell(threading.Event()), _op)

    def test_reset_terminates_in_flight_work_on_the_serving_controller(self):
        """Fix 4 (B1): the SAME contract on HybridCacheController with its own
        PrefetchOperation twin -- the shape UnifiedRadixCache actually runs.
        RED on e5b7eb3b79: terminated_ops=0, the aux thread never leaves the
        transfer, reset() raises RuntimeError after the bound."""
        self._drive_three_inflight_and_reset(
            _hybrid_shell(threading.Event()), _hybrid_op
        )

    def test_the_terminate_pass_is_class_agnostic(self):
        """Fix 4 (B1) pin on the helper itself, without threads: a hybrid
        operation sitting in prefetch_queue and one held as the aux loop's
        current operation are both terminated and both counted. RED on
        e5b7eb3b79 (isinstance against the base class: 0 terminated)."""
        cc = _hybrid_shell(threading.Event())
        op_queued = _hybrid_op("q", 0)
        op_io = _hybrid_op("io", 1_000_000)
        cc.prefetch_queue.put(op_queued)
        cc._prefetch_io_current = op_io
        self.assertEqual(cc._terminate_inflight_prefetch(), 2)
        self.assertTrue(op_queued.is_terminated())
        self.assertTrue(op_io.is_terminated())
        # A None / a stray non-operation in a queue is skipped, not counted.
        cc.prefetch_queue.put(None)
        cc.prefetch_queue.put(object())
        self.assertEqual(cc._terminate_inflight_prefetch(), 2)
        # The pointer reads tolerate a shell that never ran __init__ or
        # _start_storage_threads (attach/detach before the first start).
        del cc._prefetch_current
        del cc._prefetch_io_current
        self.assertEqual(cc._terminate_inflight_prefetch(), 1)

    def test_a_terminated_transfer_issues_no_page_reads(self):
        """The batch loop of _page_transfer aborts at the batch boundary of a
        terminated operation: zero storage reads, not one wasted 4 MiB batch."""
        cc = _shell(threading.Event())
        cc.draft_tier_armed = lambda site: False
        reads = []
        cc.page_get_func = lambda *a, **k: reads.append(a)
        op = _op("t", 0)
        op.hash_value = cc.get_hash_str(op.token_ids, None, page_size=PAGE)
        op.mark_terminate()
        cc._page_transfer(op)
        self.assertEqual(reads, [])

    def test_a_terminated_probe_issues_no_storage_queries(self):
        """_storage_hit_query of a terminated operation asks the store
        nothing and reports zero hits, so the prefetch thread revokes it in
        one pass instead of walking 308 batches of batch_exists.

        Fix 4 (B1b): held on BOTH classes. HybridCacheController overrides
        _storage_hit_query in full (one batch_exists / batch_exists_v2 call
        over the whole span, no batch loop), so the base guard does not
        reach it; the override carries its own. RED on e5b7eb3b79 for the
        hybrid subtest (one batch_exists call over all 615 page hashes)."""
        for label, mk_shell, mk_op in (
            ("HiCacheController", _shell, _op),
            ("HybridCacheController", _hybrid_shell, _hybrid_op),
        ):
            with self.subTest(controller=label):
                cc = mk_shell(threading.Event())
                asked = []
                cc.storage_backend = types.SimpleNamespace(
                    batch_exists=lambda hashes, extra_info=None: asked.append(
                        ("v1", list(hashes))
                    )
                    or len(hashes),
                    batch_exists_v2=lambda hashes, transfers, extra_info=None: asked.append(
                        ("v2", list(hashes))
                    )
                    or None,
                )
                op = mk_op("q", 0)
                op.mark_terminate()
                hash_value, hits = cc._storage_hit_query(op)
                self.assertEqual(asked, [])
                self.assertEqual((hash_value, hits), ([], 0))

    def test_the_bound_is_a_named_constant_and_the_helper_names_its_callers(self):
        self.assertIsInstance(cc_module.STORAGE_THREAD_JOIN_BOUND_S, float)
        src = inspect.getsource(HiCacheController._stop_storage_threads)
        self.assertIn("STORAGE_THREAD_JOIN_BOUND_S", src)
        self.assertNotIn("join(timeout=10)", src)
        doc = HiCacheController._stop_storage_threads.__doc__ or ""
        self.assertIn("reset", doc)
        self.assertIn("quiesce", doc)
        # Fix 4 (docstrings): the bound is claimed for BOTH controller
        # classes, and the pre-existing peer-skew term of the drain is named
        # rather than hidden behind "by construction".
        self.assertIn("HybridCacheController", doc)
        self.assertIn("peer", doc)


if __name__ == "__main__":
    unittest.main()
