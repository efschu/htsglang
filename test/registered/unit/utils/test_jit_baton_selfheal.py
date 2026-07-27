"""An orphaned cpp_extension build lock must not stall every later process.

THE OBSERVED DEFECT
-------------------
A test run stood 18 minutes at 0 % CPU holding 682 MiB of VRAM and outlived
its own ``timeout 900``; a second run waited for the cards it would not
release. py-spy put every one of those minutes in one place::

    torch/utils/file_baton.py:51           wait
    torch/utils/cpp_extension.py:2286      _jit_compile
    mem_cache/cpp_utils/native_hash.py:33  _load_native_hash_module

``$TORCH_EXTENSIONS_DIR/hicache_hash_cpp/lock`` was a ZERO-BYTE file from the
previous day, 16:18, while ``hicache_hash_cpp.so`` in the SAME directory had
been finished at 16:15. The process that created the lock died between
finishing and cleaning up. ``FileBaton.wait()`` polls ``os.path.exists`` with
no bound and no evidence, so from that moment every process wanting that
module waits forever -- for a build that is already done. Deleting the lock by
hand let the waiter through instantly.

The same shape as the tvm-ffi cache (#172b) and the torch_extensions cache
(#181): an interrupted build turns a transient failure into a permanent one.
Only the callsite is new -- so what is pinned here is the LOCK, which those
two fixes do not look at.

WHAT IS PINNED
 1. Stock ``FileBaton.wait`` does not return on the orphan shape. Without this
    the rest of the file proves nothing about the real defect.
 2. The self-healing wait returns on that shape, and takes the lock with it.
 3. A build that is making progress is NOT stolen from -- a recently touched
    build directory keeps the waiter waiting.
 4. An artifact OLDER than its sources is a rebuild in flight, not a finished
    build: the waiter keeps waiting rather than importing a stale ``.so``.
 5. The wait is bounded in every case. Past the age limit the waiter either
    takes over (artifact present) or fails with a NAMED message (no artifact).
    It never waits silently forever.
 6. Exactly one of N concurrent waiters removes the lock, and a holder whose
    lock was removed can still call ``release()``.

CPU only: filesystem bookkeeping, no compiler, no GPU.
"""

import os
import threading
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

from torch.utils.file_baton import FileBaton  # noqa: E402

# THE REAL MODULE, imported -- not re-implemented here.
from sglang.jit_kernel.baton_health import (  # noqa: E402
    BATON_ORPHAN_MARKER,
    baton_verdict,
    claim_baton,
    install_baton_selfheal,
    register_sources,
)

#: The observed case: a lock from the previous day.
OLD = 48 * 3600.0
#: Quiescent (far past the 120 s quiet window) but still well inside the
#: 30 min age limit, so rules 2 and 4 can be told apart.
QUIET_BUT_FRESH = 600.0


def _backdate(path: Path, age_seconds: float) -> None:
    when = time.time() - age_seconds
    os.utime(path, (when, when))


def _build_dir(
    root: Path,
    name: str,
    *,
    artifact: bool = True,
    lock: bool = True,
    age: float = OLD,
    artifact_age: float | None = None,
) -> Path:
    """A torch cpp_extension build directory in a chosen state.

    ``age`` backdates the whole directory, which is what makes it quiescent;
    ``artifact_age`` overrides it for the ``.so`` alone.
    """
    d = root / name
    d.mkdir(parents=True)
    (d / "build.ninja").write_text("rule cc\n")
    (d / "main.o").write_bytes(b"\x00")
    if artifact:
        (d / f"{name}.so").write_bytes(b"\x7fELF-not-really")
    if lock:
        (d / "lock").write_bytes(b"")
    for child in sorted(d.iterdir()):
        _backdate(child, age)
    if artifact and artifact_age is not None:
        _backdate(d / f"{name}.so", artifact_age)
    _backdate(d, age)
    return d


def _source(root: Path, name: str, age: float) -> Path:
    src = root / f"{name}.cpp"
    src.write_text("int f(){return 0;}\n")
    _backdate(src, age)
    return src


def _wait_in_thread(fn, seconds: float) -> bool:
    """Run ``fn`` with a deadline. True when it returned in time."""
    done = threading.Event()
    box = {}

    def run():
        try:
            fn()
        except BaseException as exc:  # recorded, re-raised by the caller
            box["exc"] = exc
        finally:
            done.set()

    t = threading.Thread(target=run, daemon=True)
    t.start()
    finished = done.wait(seconds)
    if finished and "exc" in box:
        raise box["exc"]
    return finished


class TestOrphanedBatonDefect(CustomTestCase):
    """(1) The falsifier: stock torch does not come back from this."""

    def test_stock_wait_never_returns_on_orphaned_lock(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = _build_dir(root, "hicache_hash_cpp")
            # The exact observed shape: finished .so, zero-byte lock, nobody
            # left alive to release it.
            self.assertEqual((d / "lock").stat().st_size, 0)
            self.assertTrue((d / "hicache_hash_cpp.so").exists())

            baton = FileBaton(str(d / "lock"), wait_seconds=0.02)
            # install_baton_selfheal() replaces the bound method on the class;
            # __wrapped__ is the stock implementation, kept so this falsifier
            # keeps testing torch rather than the fix.
            install_baton_selfheal()
            stock = getattr(FileBaton.wait, "__wrapped__", FileBaton.wait)
            returned = _wait_in_thread(lambda: stock(baton), 2.0)
            self.assertFalse(
                returned,
                "stock FileBaton.wait returned on an orphaned lock -- the "
                "defect this file exists for is not reproduced, so nothing "
                "below proves anything",
            )
            self.assertTrue((d / "lock").exists())


class TestBatonSelfHeal(CustomTestCase):
    def setUp(self):
        install_baton_selfheal()

    def _wait(self, lock: Path, seconds: float = 5.0) -> bool:
        baton = FileBaton(str(lock), wait_seconds=0.02)
        return _wait_in_thread(baton.wait, seconds)

    def test_orphaned_lock_with_finished_artifact_is_reclaimed(self):
        """(2) The observed case, healed: return promptly, take the lock."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _source(root, "hicache_hash_cpp", OLD + 600)
            d = _build_dir(root, "hicache_hash_cpp")
            register_sources(d, [src])

            # Reclaimed on the evidence, not merely because the lock is old:
            # the verdict holds with the age limit pushed out of reach.
            verdict = baton_verdict(d / "lock", max_wait_seconds=10 * OLD)
            self.assertEqual(verdict.action, "reclaim")
            self.assertIn("newer than its sources", verdict.reason)

            self.assertTrue(self._wait(d / "lock", 5.0))
            self.assertFalse(
                (d / "lock").exists(), "the orphaned lock was left behind"
            )
            self.assertTrue((d / "hicache_hash_cpp.so").exists())

    def test_a_live_build_is_not_stolen_from(self):
        """(3) A directory touched just now belongs to somebody."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _source(root, "warm", OLD + 600)
            d = _build_dir(root, "warm", age=0.0)  # touched right now
            register_sources(d, [src])

            self.assertEqual(baton_verdict(d / "lock").action, "wait")
            self.assertFalse(self._wait(d / "lock", 1.0))
            self.assertTrue((d / "lock").exists())

    def test_artifact_older_than_sources_keeps_waiting(self):
        """(4) A stale .so is a rebuild in flight, not a finished build.

        Quiescent and orphan-shaped in every respect except the one that
        matters, and still inside the age limit -- so the only thing that can
        hold the waiter here is the source comparison.
        """
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = _build_dir(root, "rebuild", age=QUIET_BUT_FRESH)
            src = _source(root, "rebuild", 60.0)  # edited after the .so
            register_sources(d, [src])

            self.assertEqual(baton_verdict(d / "lock").action, "wait")
            self.assertFalse(self._wait(d / "lock", 1.0))
            self.assertTrue((d / "lock").exists())

    def test_age_limit_takes_over_when_an_artifact_exists(self):
        """(5a) Past the limit, an existing artifact is good enough.

        Same fixture as (4): only the limit changes, so what is measured is
        the limit and nothing else.
        """
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = _build_dir(root, "expired", age=QUIET_BUT_FRESH)
            src = _source(root, "expired", 60.0)  # newer than the .so
            register_sources(d, [src])

            self.assertEqual(baton_verdict(d / "lock").action, "wait")
            verdict = baton_verdict(d / "lock", max_wait_seconds=10.0)
            self.assertEqual(verdict.action, "reclaim")
            self.assertIn("limit", verdict.reason)

            os.environ["SGLANG_JIT_BATON_MAX_WAIT_SECONDS"] = "10"
            try:
                self.assertTrue(self._wait(d / "lock", 5.0))
            finally:
                del os.environ["SGLANG_JIT_BATON_MAX_WAIT_SECONDS"]
            self.assertFalse((d / "lock").exists())

    def test_age_limit_without_artifact_fails_by_name(self):
        """(5b) Nothing to import: a named error, never a silent hang."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = _build_dir(root, "nothing", artifact=False, age=QUIET_BUT_FRESH)

            self.assertEqual(baton_verdict(d / "lock").action, "wait")
            verdict = baton_verdict(d / "lock", max_wait_seconds=10.0)
            self.assertEqual(verdict.action, "fail")

            os.environ["SGLANG_JIT_BATON_MAX_WAIT_SECONDS"] = "10"
            try:
                with self.assertRaises(RuntimeError) as caught:
                    self._wait(d / "lock", 5.0)
            finally:
                del os.environ["SGLANG_JIT_BATON_MAX_WAIT_SECONDS"]
            message = str(caught.exception)
            self.assertIn(BATON_ORPHAN_MARKER, message)
            self.assertIn(str(d), message)

    def test_a_cold_build_that_produced_nothing_yet_is_left_alone(self):
        """A first build has no artifact and can be quiet for a long time.

        Quiescence must never be read as abandonment on its own -- a single
        nvcc edge produces no file until it finishes.
        """
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _source(root, "cold", QUIET_BUT_FRESH + 600)
            d = _build_dir(root, "cold", artifact=False, age=QUIET_BUT_FRESH)
            register_sources(d, [src])

            self.assertEqual(baton_verdict(d / "lock").action, "wait")
            self.assertFalse(self._wait(d / "lock", 1.0))

    def test_a_live_cache_health_marker_vetoes_reclaiming(self):
        """A co-located rank that says it is building is believed."""
        from sglang.jit_kernel.cache_health import MARKER_BUILDING, _hostname

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = _source(root, "marked", OLD + 600)
            d = _build_dir(root, "marked", age=QUIET_BUT_FRESH)
            register_sources(d, [src])
            self.assertEqual(baton_verdict(d / "lock").action, "reclaim")

            marker = d / MARKER_BUILDING
            marker.write_text(f"{_hostname()}\n{os.getpid()}\n{time.time()}\n")
            # Backdated with the rest of the directory: otherwise the write
            # alone would break quiescence and the marker would prove nothing.
            _backdate(marker, QUIET_BUT_FRESH)
            _backdate(d, QUIET_BUT_FRESH)

            self.assertEqual(baton_verdict(d / "lock").action, "wait")
            self.assertFalse(self._wait(d / "lock", 1.0))
            self.assertTrue((d / "lock").exists())

    def test_a_released_lock_ends_the_wait_normally(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = _build_dir(root, "released", age=0.0)
            lock = d / "lock"

            def release_soon():
                time.sleep(0.3)
                lock.unlink()

            threading.Thread(target=release_soon, daemon=True).start()
            self.assertTrue(self._wait(lock, 5.0))

    def test_exactly_one_claimant(self):
        """(6a) The claim is a rename, so concurrent waiters cannot both win."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = _build_dir(root, "contended", age=OLD)
            lock = d / "lock"

            start = threading.Barrier(8)
            results = []
            guard = threading.Lock()

            def claim():
                start.wait()
                won = claim_baton(lock)
                with guard:
                    results.append(won)

            threads = [threading.Thread(target=claim) for _ in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(10.0)

            self.assertEqual(sum(1 for r in results if r), 1, results)
            self.assertFalse(lock.exists())
            self.assertEqual(
                [p for p in d.iterdir() if "orphan" in p.name],
                [],
                "the renamed lock was not cleaned up",
            )

    def test_release_tolerates_a_reclaimed_lock(self):
        """(6b) Reclaiming must not turn into a crash for the lock's holder."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = _build_dir(root, "holder", lock=False, age=0.0)
            baton = FileBaton(str(d / "lock"))
            self.assertTrue(baton.try_acquire())
            claim_baton(d / "lock")
            baton.release()  # stock torch raises FileNotFoundError here

    def test_selfheal_can_be_switched_off(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = _build_dir(root, "disabled", age=OLD)
            os.environ["SGLANG_JIT_BATON_SELFHEAL"] = "0"
            try:
                self.assertFalse(self._wait(d / "lock", 1.0))
            finally:
                del os.environ["SGLANG_JIT_BATON_SELFHEAL"]
            self.assertTrue((d / "lock").exists())

    def test_install_is_idempotent(self):
        first = FileBaton.wait
        install_baton_selfheal()
        install_baton_selfheal()
        self.assertIs(FileBaton.wait, first)


if __name__ == "__main__":
    unittest.main()
