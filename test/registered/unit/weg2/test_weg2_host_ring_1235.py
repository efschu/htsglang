"""T1 -- the host granule ring's state machine and the C7 entrypoints, hermetic.

No GPU, no CUDA context, no free card: ``host_ring_probe.cpp`` links the
SHIPPING ``host_ring.cpp`` against three driver stubs (``cuDeviceGetUuid``,
``cudaHostRegister``, ``cudaGetErrorString``) and drives the real state machine
over a real ``/dev/shm`` file with real forks.  What is faked is named; the code
under test is not touched.

Red-first, per case, all four checked by deleting the named line and re-running:

* ``acquire`` blocking on a peer, waking on its ``release``
  -- delete the ``pthread_cond_timedwait`` loop and the waiter returns 0 granules.
* the foreign-``owner_pid`` release refusal
  -- delete the ``owner_pid[i] != self`` guard and the steal is ACCEPTED.
* the stale-pid sweep freeing exactly the dead pid's granules
  -- delete ``sweep_stale_locked`` from the attach path and the sweep count is 0.
* a scatter list covering a size no contiguous run could
  -- make ``acquire`` demand a contiguous run and the odd-granule request fails.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
CSRC = os.path.join(TREE, "python", "sglang", "srt", "weg2", "tms_csrc")
VENV = os.environ.get("WEG2_VENV", "/spinning/htsglang-gpu/.venv")
CU13_INCLUDE = os.path.join(VENV, "lib", "python3.12", "site-packages", "nvidia", "cu13", "include")
GRANULE = 2 * 1024 * 1024
UUID = "GPU-10111213-1415-1617-1819-1a1b1c1d1e1f"


def _build(tmp: str) -> str:
    out = os.path.join(tmp, "probe")
    cmd = [
        "g++", "-std=c++17", "-O0", "-g", "-DUSE_CUDA=1",
        f"-I{CU13_INCLUDE}", f"-I{CSRC}",
        os.path.join(HERE, "host_ring_probe.cpp"),
        os.path.join(CSRC, "host_ring.cpp"),
        "-lpthread", "-o", out,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise unittest.SkipTest(f"cannot build the T1 probe: {r.stderr[-800:]}")
    return out


@unittest.skipUnless(
    os.path.isdir(CU13_INCLUDE), f"cu13 headers not at {CU13_INCLUDE}"
)
class HostRingStateMachineTest(unittest.TestCase):
    granules = 8

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="weg2ring-")
        cls.probe = _build(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _fresh_dir(self, granules: int = 0) -> str:
        granules = granules or self.granules
        d = tempfile.mkdtemp(prefix="ringdir-", dir=self.tmp)
        path = os.path.join(d, f"{UUID}.ring")
        with open(path, "wb") as f:
            f.truncate(GRANULE + granules * GRANULE)
        return d

    def _env(self, ring_dir, form="MAP_SHARED", granules: int = 0, **extra):
        granules = granules or self.granules
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = ""
        if ring_dir is not None:
            env["TMS_HOST_RING_DIR"] = ring_dir
            env["TMS_HOST_RING_MAP"] = f"{UUID}={granules * GRANULE}:{(granules // 2) * GRANULE}"
            env["TMS_HOST_RING_EPOCH"] = "4242"
            if form:
                env["TMS_HOST_RING_FORM"] = form
            else:
                env.pop("TMS_HOST_RING_FORM", None)
        else:
            for k in ("TMS_HOST_RING_DIR", "TMS_HOST_RING_MAP",
                      "TMS_HOST_RING_EPOCH", "TMS_HOST_RING_FORM"):
                env.pop(k, None)
        env.update(extra)
        return env

    def _run(self, case, env, timeout=180):
        return subprocess.run([self.probe, case], env=env, capture_output=True,
                              text=True, timeout=timeout)

    # ---------------------------------------------------------------- forms --

    def test_no_ring_published_is_the_stock_path(self):
        r = self._run("no_ring", self._env(None))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NO_RING", r.stdout)

    def test_map_shared_form_opens(self):
        r = self._run("uuid", self._env(self._fresh_dir()))
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn(f"CARD {UUID}", r.stdout)
        # The card is named by cuDeviceGetUuid, and the attach line proves the
        # file it resolved was keyed by that uuid, never by an ordinal.
        self.assertIn(f"{UUID}.ring", r.stderr)

    def test_memfd_form_opens_from_an_inherited_fd(self):
        d = self._fresh_dir()
        path = os.path.join(d, f"{UUID}.ring")
        fd = os.open(path, os.O_RDWR)
        try:
            os.set_inheritable(fd, True)
            env = self._env(d, form="MEMFD")
            env["TMS_HOST_RING_MAP"] = (
                f"{UUID}={self.granules * GRANULE}:{(self.granules // 2) * GRANULE}:fd={fd}"
            )
            r = subprocess.run([self.probe, "uuid"], env=env, capture_output=True,
                               text=True, pass_fds=(fd,), timeout=60)
        finally:
            os.close(fd)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn(f"CARD {UUID}", r.stdout)
        self.assertIn("form=MEMFD", r.stderr)

    def test_a_published_ring_without_a_proven_form_is_refused_by_name(self):
        r = self._run("uuid", self._env(self._fresh_dir(), form=""))
        self.assertEqual(r.returncode, 1)
        self.assertIn("W33 Weg2RingFormUnproven", r.stderr)

    def test_memfd_form_without_an_fd_is_refused_by_name(self):
        env = self._env(self._fresh_dir(), form="MEMFD")
        r = self._run("uuid", env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("W33 Weg2RingFormUnproven", r.stderr)
        self.assertIn("fd=", r.stderr)

    def test_a_register_failure_is_refused_by_name_not_worked_around(self):
        r = self._run("register", self._env(self._fresh_dir(),
                                            WEG2_TEST_REGISTER_FAILS="1"))
        self.assertEqual(r.returncode, 1)
        self.assertIn("W33 Weg2RingFormUnproven", r.stderr)
        self.assertIn("cudaHostRegister", r.stderr)

    def test_register_span_is_lazy_and_emits_l2(self):
        r = self._run("register", self._env(self._fresh_dir()))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("REGISTER spans=2", r.stdout)
        self.assertEqual(r.stderr.count("WEG2-RING-OPEN"), 2)
        self.assertIn("span=1/2", r.stderr)
        self.assertIn("span=2/2", r.stderr)

    # ----------------------------------------------------------- accounting --

    def test_granule_accounting_covers_a_non_multiple_size(self):
        r = self._run("granules", self._env(self._fresh_dir()))
        self.assertEqual(r.returncode, 0, r.stderr)
        m = re.search(
            r"GRANULES total=(\d+) got=(\d+) free0=(\d+) free1=(\d+) free2=(\d+) "
            r"peak=(\d+) acquires=(\d+)", r.stdout)
        self.assertIsNotNone(m, r.stdout)
        total, got, free0, free1, free2, peak, acq = (int(x) for x in m.groups())
        self.assertEqual(total, self.granules)
        # 3 granules + 1 byte must take FOUR granules, never three.
        self.assertEqual(got, 4)
        self.assertEqual(free0, self.granules)
        self.assertEqual(free1, self.granules - 4)
        self.assertEqual(free2, self.granules)
        self.assertEqual(peak, 4)
        self.assertEqual(acq, 1)

    def test_a_scatter_list_covers_what_no_contiguous_run_could(self):
        r = self._run("scatter", self._env(self._fresh_dir()))
        self.assertEqual(r.returncode, 0, r.stderr)
        m = re.search(r"SCATTER want=(\d+) got=(\d+) adjacent=(\d+)", r.stdout)
        self.assertIsNotNone(m, r.stdout)
        want, got, adjacent = (int(x) for x in m.groups())
        self.assertEqual(got, want)
        # Every free granule is isolated between two taken ones: a contiguous
        # allocator would have refused this request.
        self.assertEqual(adjacent, 0)

    def test_two_tag_families_share_the_one_region(self):
        r = self._run("family", self._env(self._fresh_dir()))
        self.assertEqual(r.returncode, 0, r.stderr)
        m = re.search(r"FAMILY a=(\d+) b=(\d+) same=(\d+) free=(\d+)", r.stdout)
        self.assertIsNotNone(m, r.stdout)
        a, b, same, free = (int(x) for x in m.groups())
        self.assertEqual((a, b), (1, 1))
        self.assertEqual(same, 0, "two families must never be handed the same granule")
        self.assertEqual(free, self.granules - 2)

    # ------------------------------------------------------------- blocking --

    def test_acquire_blocks_on_a_peer_and_wakes_on_its_release(self):
        r = self._run("blocking", self._env(self._fresh_dir()))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("BLOCKED_THEN_GOT 1", r.stdout)
        m = re.search(r"BLOCKING waiter_rc=(\d+) blocked_ms=(\d+)", r.stdout)
        self.assertIsNotNone(m, r.stdout)
        self.assertEqual(int(m.group(1)), 0)
        # It really waited: the child sleeps 400 ms before releasing.  A
        # non-blocking acquire would report ~0 and hand back nothing.
        self.assertGreater(int(m.group(2)), 100, r.stdout)

    # ------------------------------------------------------------- refusals --

    def test_a_foreign_release_is_refused_never_silently_accepted(self):
        r = self._run("foreign_release", self._env(self._fresh_dir()))
        self.assertIn("FOREIGN_RELEASE_ATTEMPT", r.stdout)
        self.assertNotIn("FOREIGN_RELEASE_ACCEPTED", r.stdout)
        self.assertEqual(r.returncode, 1)
        self.assertIn("REFUSES a foreign release", r.stderr)

    def test_exhaustion_is_w31_by_name_after_the_bounded_wait(self):
        # A request larger than the whole region can never be satisfied; the
        # bounded wait expires into W31 naming the waiter.  This is the
        # backstop, not a path (R15), which is why the test asserts on the
        # NAME rather than on a duration.
        env = self._env(self._fresh_dir())
        r = self._run("exhausted", env, timeout=400)
        self.assertIn("EXHAUST_ATTEMPT", r.stdout)
        self.assertNotIn("EXHAUST_ACCEPTED", r.stdout)
        self.assertEqual(r.returncode, 1)
        self.assertIn("W31 Weg2HostRingExhausted", r.stderr)
        for field in ("card=", "tag=too_big", "need=", "free=", "waiter=pid", "waited_ms="):
            self.assertIn(field, r.stderr)

    def test_the_stale_sweep_frees_exactly_the_dead_pids_granules(self):
        r = self._run("stale_sweep", self._env(self._fresh_dir()))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        m = re.search(r"SWEEP before_free=(\d+) got=(\d+) after_free=(\d+) "
                      r"swept=(\d+) mine=(\d+)", r.stdout)
        self.assertIsNotNone(m, r.stdout)
        before_free, got, after_free, swept, mine = (int(x) for x in m.groups())
        self.assertEqual(mine, 2)
        # The dead child held three; this process still holds its own two.
        self.assertEqual(swept, 3)
        self.assertEqual(got, self.granules - 2)
        self.assertEqual(after_free, 0)
        self.assertLess(before_free, self.granules - 2)

    def test_an_epoch_mismatch_zeroes_a_previous_boots_leftovers(self):
        d = self._fresh_dir()
        first = self._env(d)
        r1 = self._run("granules", first)
        self.assertEqual(r1.returncode, 0, r1.stderr)
        second = self._env(d)
        second["TMS_HOST_RING_EPOCH"] = "4243"
        r2 = self._run("granules", second)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertIn("fresh=1", r2.stderr)
        self.assertIn("acquires=1", r2.stdout, "the counters must restart with the epoch")


class EntrypointSurfaceTest(unittest.TestCase):
    """C7/C8: the two C entries exist and the adapter passes them through."""

    def test_entrypoint_cpp_exports_both_entries(self):
        src = open(os.path.join(CSRC, "entrypoint.cpp")).read()
        self.assertIn("uint64_t tms_tag_bytes(const char* tag)", src)
        self.assertIn("int tms_ring_stats(", src)
        # Both must sit inside the extern "C" block, or ctypes cannot find them.
        head = src.index('extern "C" {', src.index("entrypoints :: others"))
        self.assertGreater(src.index("tms_tag_bytes"), head)

    def test_adapter_passes_both_through_and_never_invents_a_value(self):
        from sglang.srt.utils.torch_memory_saver_adapter import (
            _TorchMemorySaverAdapterNoop,
            _weg2_ring_symbol,
        )

        noop = _TorchMemorySaverAdapterNoop()
        self.assertIsNone(noop.tag_bytes("weights_0"))
        self.assertIsNone(noop.ring_stats())
        # With the stock hook (no preload) the symbol is absent and the lookup
        # returns None rather than a zero that would read as "no bytes".
        self.assertIsNone(_weg2_ring_symbol("tms_tag_bytes_definitely_absent"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
