# SPDX-License-Identifier: Apache-2.0
"""L3 background evictor: a fair lock handoff between batches (D-HEALTH 10.10.).

THE FINDING (boot dkr27bnvfp4dualvwweightsbar1fs10100032). The EVICT_OFFPATH thread released ``_lock``
between batches with ``time.sleep(0)`` and took it straight back. ``threading.Lock`` is not fair, so the
PARK-DEMOTE ``reserve()``/``commit()`` and the write-behind lost the race batch after batch: 16439 pages in
9672 ms instead of ~2.3 s.

THE CONTRACT. A ``reserve()`` or ``commit()`` that has to wait for the lock while the background run is busy
waits for at most ONE unlink (the victim in progress when it arrived): the batch ends at the next victim
boundary and ``handoff()`` lets the waiter in before the next batch. Counted in unlinks, not in
milliseconds, so the verdict does not depend on how fast the filesystem unlinks; ``os.remove`` is only
slowed so the run outlasts the demoter.

The unfair variant (``handoff`` = ``time.sleep(0)``, no early batch end -- the base) is the mutant: run this
module as a script with ``--mutant`` to print its figures (a waiter sits through the rest of a batch).
"""
import os
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

try:
    from sglang.test.ci.ci_register import register_cpu_ci
except ImportError:  # pragma: no cover

    def register_cpu_ci(*args, **kwargs):
        return None


from sglang.srt.mem_cache.storage.file import lru_file_evictor as lfe
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

PAGE = 1000
N_PAGES = 3000
UNLINK_S = 0.001
DEMOTE_PAGES = 40


def _measure(fair: bool):
    """Per demoter reserve()/commit(): how many background unlinks happened while it was inside the call."""
    with tempfile.TemporaryDirectory() as d:
        env = {"SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE": "", "SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE": "",
               "SGLANG_HICACHE_FILE_BACKEND_EVICT_OFFPATH": "1"}
        with mock.patch.dict(os.environ, env):
            ev = lfe.LRUFileEvictor(d, "_t", tp_rank=0, writes_shared_keys=False,
                                    extra_config={"max_size": str(10 ** 12)})

            def write(stem):
                assert ev.reserve(stem, PAGE, stem)
                with open(os.path.join(d, f"{stem}.bin"), "wb") as fh:
                    fh.write(b"x" * PAGE)
                ev.commit(stem)

            for i in range(N_PAGES):
                write(f"p{i:05d}_t")
            unlinks = [0]
            real_remove = os.remove

            def slow_remove(p):
                time.sleep(UNLINK_S)  # a GIL-free syscall, like the real unlink
                real_remove(p)
                unlinks[0] += 1

            patches = [mock.patch.object(lfe.os, "remove", slow_remove)]
            if not fair:
                patches.append(mock.patch.object(lfe._HandoffLock, "handoff",
                                                 lambda self, timeout=1.0: time.sleep(0)))
                patches.append(mock.patch.object(lfe._HandoffLock, "waiting", property(lambda self: 0)))
            for p in patches:
                p.start()
            try:
                with ev._lock:
                    dir_bytes = ev._directory_bytes_locked()
                    ev.max_size_bytes = dir_bytes + 64 * 4096  # room for the demoter: its reserve evicts nothing
                    ev.eviction_ratio = 0.3                    # the background run has ~2000 victims to go
                    ev._kick_bg_evictor_locked()
                t_end = time.monotonic() + 10
                while unlinks[0] < 2 and time.monotonic() < t_end:
                    time.sleep(0.001)
                per_call = []
                for i in range(DEMOTE_PAGES):
                    stem = f"demote{i:03d}_t"
                    u0 = unlinks[0]
                    assert ev.reserve(stem, PAGE, stem)
                    per_call.append(unlinks[0] - u0)
                    with open(os.path.join(d, f"{stem}.bin"), "wb") as fh:
                        fh.write(b"x" * PAGE)
                    u0 = unlinks[0]
                    ev.commit(stem)
                    per_call.append(unlinks[0] - u0)
                with ev._lock:  # the background run outlasted the demoter: every call met it
                    run_open = ev._directory_bytes_locked() > int(ev.max_size_bytes * ev.eviction_ratio)
                # park the evictor before the directory goes away
                ev.pause_background_eviction(timeout=10, reason="test")
                with ev._lock:
                    indexed = list(ev._lru)
            finally:
                for p in reversed(patches):
                    p.stop()
            missing = [s for s in indexed if not os.path.exists(os.path.join(d, f"{s}.bin"))]
            return per_call, missing, run_open


class L3EvictorFairHandoffTest(CustomTestCase):
    def test_reserve_and_commit_wait_at_most_one_unlink(self):
        per_call, missing, run_open = _measure(fair=True)
        self.assertTrue(run_open, "the background run ended before the demoter: nothing was contended")
        self.assertLessEqual(max(per_call), 1, f"unlinks during one reserve()/commit(): {per_call}")
        self.assertEqual(missing, [], "an index entry points to a deleted file")

    def test_handoff_lets_every_waiter_in_before_the_holder_returns(self):
        """The primitive: a holder that releases and hands off re-acquires only after the waiter had it."""
        lock = lfe._HandoffLock()
        for _ in range(50):
            order = []
            lock.acquire()

            def waiter():
                with lock:
                    order.append("W")

            t = threading.Thread(target=waiter)
            t.start()
            t_end = time.monotonic() + 5
            while lock.waiting == 0 and time.monotonic() < t_end:
                time.sleep(0.0005)
            self.assertEqual(lock.waiting, 1)
            lock.release()
            lock.handoff()
            with lock:
                order.append("H")
            t.join(5)
            self.assertEqual(order, ["W", "H"])

    def test_handoff_without_waiters_returns_at_once(self):
        lock = lfe._HandoffLock()
        t0 = time.monotonic()
        lock.handoff(timeout=5.0)
        self.assertLess(time.monotonic() - t0, 0.5)


if __name__ == "__main__":
    if "--mutant" in sys.argv:
        for fair in (True, False):
            per, missing, run_open = _measure(fair=fair)
            print(f"fair={fair}: max unlinks during one call={max(per)} sum={sum(per)} "
                  f"calls_waiting>1={sum(1 for x in per if x > 1)}/{len(per)} run_open={run_open} "
                  f"missing={len(missing)}")
    else:
        unittest.main()
