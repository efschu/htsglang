# SPDX-License-Identifier: Apache-2.0
"""L3 eviction off the RESET-JOIN path (27B S1 29.09., flip 13->14 7,05 s).

THE FINDING. ``LRUFileEvictor.reserve()`` ran the whole cap eviction -- cap down to cap x ratio, S1: 6.3 GB,
~170k unlinks, 4.8 s -- on the calling thread. That thread is HiCache's ``backup`` thread, and the sleep flush's
#1068 RESET JOIN (``cache_controller._stop_storage_threads``) waits for it, so the flip waited 4.9 s in quiesce.

THE REPRODUCTION. A store of many small committed pages sits just under its cap; a "backup" thread reserves one
more page, which crosses the cap; ``os.remove`` is slowed to the rig's per-file cost scaled up, so the run is long.
The "reset" joins that thread with a bound. Red on the base (the join waits for the whole run), green with
SGLANG_HICACHE_FILE_BACKEND_EVICT_OFFPATH=1 (reserve evicts only its own need, the "l3_evictor" thread finishes
the run). Invariants on both: every page still in the index has its file; every evicted page went through
on_evict; the directory ends at or below cap x ratio once the background run is done.
"""
import os
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

register_cpu_ci(est_time=15, suite="base-a-test-cpu")

PAGE = 1000          # bytes per page file
N_PAGES = 400        # pages committed before the crossing write
CAP = N_PAGES * 4096  # ZFS/ext4 allocate in blocks; the evictor charges allocated size (stat), so cap in blocks
UNLINK_S = 0.05      # slowed unlink: a run to cap x 0.9 = 40 of 400 pages x 50 ms ~ 2 s (S1 measured 4.8 s)
JOIN_BOUND_S = 1.0   # the flush may wait for one page's own need, never for the whole run


class _Harness:
    def __init__(self, tmpdir: str, offpath: bool):
        self.evicted = []
        env = {"SGLANG_HICACHE_FILE_BACKEND_MAX_SIZE": "", "SGLANG_HICACHE_FILE_BACKEND_MIN_FREE_SPACE": "",
               "SGLANG_HICACHE_FILE_BACKEND_EVICT_OFFPATH": "1" if offpath else "0"}
        self._env = mock.patch.dict(os.environ, env)
        self._env.start()
        self.ev = lfe.LRUFileEvictor(
            tmpdir, "_t", tp_rank=0, writes_shared_keys=False,
            extra_config={"max_size": str(10 ** 12)}, on_evict=self.evicted.append)
        self.dir = tmpdir

    def stop(self):
        self._env.stop()

    def write(self, stem: str) -> bool:
        if not self.ev.reserve(stem, PAGE, stem):
            return False
        with open(os.path.join(self.dir, f"{stem}.bin"), "wb") as fh:
            fh.write(b"x" * PAGE)
        self.ev.commit(stem)
        return True


class L3EvictOffResetJoinTest(CustomTestCase):
    def _run(self, offpath: bool):
        with tempfile.TemporaryDirectory() as d:
            h = _Harness(d, offpath)
            try:
                for i in range(N_PAGES):
                    self.assertTrue(h.write(f"p{i:05d}_t"))
                # the cap sits exactly at the committed directory: the next page crosses it
                with h.ev._lock:
                    h.ev.max_size_bytes = h.ev._directory_bytes_locked()
                    cap = h.ev.max_size_bytes
                real_remove = os.remove

                def slow_remove(p):
                    time.sleep(UNLINK_S)
                    real_remove(p)

                with mock.patch.object(lfe.os, "remove", slow_remove):
                    backup = threading.Thread(target=h.write, args=("crossing_t",), name="backup")
                    t0 = time.monotonic()
                    backup.start()
                    backup.join(JOIN_BOUND_S)          # the #1068 RESET JOIN, bounded
                    joined_in = time.monotonic() - t0
                    alive = backup.is_alive()
                    backup.join(30)
                    # let the background run finish (offpath) before the invariants
                    t_end = time.monotonic() + 30
                    while time.monotonic() < t_end:
                        with h.ev._lock:
                            done = h.ev._directory_bytes_locked() <= int(cap * h.ev.eviction_ratio)
                        if done:
                            break
                        time.sleep(0.05)
                with h.ev._lock:
                    indexed = list(h.ev._lru)
                    final = h.ev._directory_bytes_locked()
                missing = [s for s in indexed if not os.path.exists(os.path.join(d, f"{s}.bin"))]
                return alive, joined_in, missing, final, cap, list(h.evicted), h.ev
            finally:
                h.stop()

    def test_offpath_reset_join_does_not_wait_for_the_run(self):
        alive, joined_in, missing, final, cap, evicted, ev = self._run(offpath=True)
        self.assertFalse(alive, f"backup thread still in reserve() after {joined_in:.2f} s -- the join waits")
        self.assertLess(joined_in, JOIN_BOUND_S)
        self.assertEqual(missing, [], "an index entry points to a deleted file")
        self.assertLessEqual(final, int(cap * ev.eviction_ratio), "background run did not reach cap x ratio")
        self.assertGreater(len(evicted), 1)
        self.assertEqual(ev._bg_evict_thread.name, "l3_evictor")

    def test_off_is_the_old_path(self):
        """Off = the whole run in reserve(): the join waits for it (the 7,05 s flip). Documents the base."""
        alive, joined_in, missing, final, cap, evicted, ev = self._run(offpath=False)
        self.assertTrue(alive, "off must keep the historical in-reserve run")
        self.assertEqual(missing, [])
        self.assertLessEqual(final, int(cap * ev.eviction_ratio))
        self.assertIsNone(ev._bg_evict_thread)


class L3EvictOffpathPauseTest(CustomTestCase):
    """NF review 2 (5a8b44523c): (2a) rescan's walk runs WITHOUT _lock -- a background unlink between its
    snapshot and install leaves a PHANTOM entry (index says there, file gone); (2b) after sleep's pause the
    background evictor must not unlink anything (the sibling owns the store). Both red on 5a8b44523c."""

    def _store(self, d, n=N_PAGES):
        h = _Harness(d, offpath=True)
        for i in range(n):
            self.assertTrue(h.write(f"p{i:05d}_t"))
        with h.ev._lock:
            h.ev.max_size_bytes = h.ev._directory_bytes_locked()
        return h

    def test_rescan_walk_sees_no_background_unlink_phantom(self):
        with tempfile.TemporaryDirectory() as d:
            h = self._store(d)
            try:
                real_census = h.ev._census_existing_files
                real_remove = os.remove

                def slow_remove(p):
                    time.sleep(0.002)
                    real_remove(p)

                def census_then_evict():
                    entries = real_census()  # the walk's snapshot: every page still on disk
                    # background eviction wakes between snapshot and install (the review's interleaving)
                    with h.ev._lock:
                        h.ev._kick_bg_evictor_locked()
                    time.sleep(0.5)
                    return entries

                with mock.patch.object(lfe.os, "remove", slow_remove), \
                        mock.patch.object(h.ev, "_census_existing_files", census_then_evict), \
                        mock.patch.object(lfe._sj, "enabled", lambda: False):
                    h.ev.rescan()
                    with h.ev._lock:
                        indexed = list(h.ev._lru)
                    phantom = [s for s in indexed if not os.path.exists(os.path.join(d, f"{s}.bin"))]
                    self.assertEqual(phantom, [], f"{len(phantom)} phantom entries after rescan")
                    # after the install the evictor may run again and must still keep index == disk
                    t_end = time.monotonic() + 10
                    while time.monotonic() < t_end:
                        with h.ev._lock:
                            if h.ev._directory_bytes_locked() <= int(h.ev.max_size_bytes * h.ev.eviction_ratio):
                                break
                        time.sleep(0.05)
                    # the kick given during the walk is not lost: the run happens after the install ...
                    self.assertGreater(len(h.evicted), 0, "the background run never happened")
                    with h.ev._lock:
                        indexed = list(h.ev._lru)
                    # ... and still leaves no index entry without its file
                    self.assertEqual([s for s in indexed if not os.path.exists(os.path.join(d, f"{s}.bin"))], [])
            finally:
                h.stop()

    def test_after_sleep_pause_no_eviction(self):
        with tempfile.TemporaryDirectory() as d:
            h = self._store(d)
            try:
                real_remove = os.remove
                removed = []

                def slow_remove(p):
                    time.sleep(UNLINK_S)
                    removed.append(p)
                    real_remove(p)

                with mock.patch.object(lfe.os, "remove", slow_remove):
                    with h.ev._lock:
                        h.ev.max_size_bytes = int(h.ev._directory_bytes_locked() * 0.5)  # long background run
                        h.ev._kick_bg_evictor_locked()
                    time.sleep(0.3)  # it is running
                    self.assertTrue(h.ev.pause_background_eviction(timeout=5.0), "drain did not park the evictor")
                    n_at_pause = len(removed)
                    time.sleep(1.0)  # the sleeping group: nothing may be unlinked now
                    self.assertEqual(len(removed), n_at_pause, "the background evictor unlinked after sleep's pause")
                    h.ev.resume_background_eviction()
                    time.sleep(0.5)
                    self.assertGreater(len(removed), n_at_pause, "resume did not restart the background run")
                    h.ev.pause_background_eviction(timeout=10.0)
            finally:
                h.stop()


class L3EvictOffpathHoldsTest(CustomTestCase):
    """NF review 3 (619ed65def): (A) a clear()/rescan-walk hold must not lift the SLEEP hold; (B) a sleep park
    that cannot complete is a NAMED stop; (C) no background unlink while the L3-index seed walk runs."""

    def _running_store(self, d):
        h = _Harness(d, offpath=True)
        for i in range(N_PAGES):
            self.assertTrue(h.write(f"p{i:05d}_t"))
        return h

    def _start_long_run(self, h, removed, unlink_s=UNLINK_S):
        real_remove = os.remove

        def slow_remove(p):
            time.sleep(unlink_s)
            removed.append(p)
            real_remove(p)

        return mock.patch.object(lfe.os, "remove", slow_remove)

    def test_sleep_hold_survives_clear(self):
        with tempfile.TemporaryDirectory() as d:
            h = self._running_store(d)
            removed = []
            try:
                with self._start_long_run(h, removed):
                    with h.ev._lock:
                        h.ev.max_size_bytes = int(h.ev._directory_bytes_locked() * 0.5)
                        h.ev._kick_bg_evictor_locked()
                    time.sleep(0.3)
                    # the sleep entry (619ed65def had only the one bool pause -- the same call the sleep made there)
                    sleep = getattr(h.ev, "pause_for_sleep", None)
                    sleep(timeout=5.0) if sleep is not None else h.ev.pause_background_eviction(timeout=5.0)
                    h.ev.clear()                               # a clear during dormancy
                    with h.ev._lock:
                        h.ev.max_size_bytes = 10 ** 12
                    for i in range(100):                       # the sibling's new pages land in the index
                        self.assertTrue(h.write(f"q{i:05d}_t"))
                    n0 = len(removed)
                    with h.ev._lock:
                        h.ev.max_size_bytes = int(h.ev._directory_bytes_locked() * 0.5)
                        h.ev._kick_bg_evictor_locked()
                    time.sleep(1.0)
                    self.assertEqual(len(removed), n0, "clear() lifted the sleep hold -- the dormant owner unlinked")
            finally:
                h.stop()

    def test_sleep_park_timeout_is_a_named_stop(self):
        with tempfile.TemporaryDirectory() as d:
            h = self._running_store(d)
            removed = []
            try:
                with self._start_long_run(h, removed, unlink_s=2.0):  # one unlink outlasts the park bound
                    with h.ev._lock:
                        h.ev.max_size_bytes = int(h.ev._directory_bytes_locked() * 0.5)
                        h.ev._kick_bg_evictor_locked()
                    time.sleep(0.2)
                    with self.assertRaises(lfe.Weg2L3EvictorPauseRefused):
                        h.ev.pause_for_sleep(timeout=0.3)
                    h.ev.pause_background_eviction(timeout=10.0)
            finally:
                h.stop()

    def test_sleep_park_without_thread_is_a_noop_that_holds(self):
        with tempfile.TemporaryDirectory() as d:
            h = _Harness(d, offpath=True)
            try:
                h.ev.pause_for_sleep(timeout=0.01)  # no thread yet: no raise
                self.assertIsNone(h.ev._bg_evict_thread)
                self.assertTrue(h.ev._bg_paused)
            finally:
                h.stop()

    def test_seed_hold_blocks_the_first_kick(self):
        with tempfile.TemporaryDirectory() as d:
            h = self._running_store(d)
            removed = []
            try:
                with self._start_long_run(h, removed):
                    h.ev.pause_background_eviction(reason="seed")   # taken before the seed thread starts
                    with h.ev._lock:
                        h.ev.max_size_bytes = int(h.ev._directory_bytes_locked() * 0.5)
                        h.ev._kick_bg_evictor_locked()          # the first reserve's kick during the seed walk
                    time.sleep(0.8)
                    self.assertEqual(removed, [], "the evictor unlinked during the seed walk")
                    h.ev.resume_background_eviction(reason="seed")
                    time.sleep(0.5)
                    self.assertGreater(len(removed), 0, "the kick given during the seed was lost")
                    h.ev.pause_background_eviction(timeout=10.0)
            finally:
                h.stop()


class TestEvictOffpathDefault(unittest.TestCase):
    def test_default_is_on_after_metal_proof(self):
        # 29.09. l3cap10 boot proved the path; a proven performance switch is default on (user order).
        from sglang.srt.environ import envs

        old = os.environ.pop("SGLANG_HICACHE_FILE_BACKEND_EVICT_OFFPATH", None)
        try:
            self.assertTrue(envs.SGLANG_HICACHE_FILE_BACKEND_EVICT_OFFPATH.get())
        finally:
            if old is not None:
                os.environ["SGLANG_HICACHE_FILE_BACKEND_EVICT_OFFPATH"] = old


if __name__ == "__main__":
    unittest.main()

