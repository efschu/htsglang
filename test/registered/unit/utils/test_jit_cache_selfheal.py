"""An incomplete JIT cache entry must be discarded, not reused forever.

Falsifier for the second half of the r3 cold-boot finding. A boot killed
mid-build (which the deadline collision of the sibling fix made routine)
leaves a tvm-ffi build directory holding `build.ninja`, `cuda.cu`,
`cuda_0.o.d` -- and no `.so`. Every later process that wants that module then
dies with

    Check failed: (lib_handle_ != nullptr) ... Failed to load

Four such directories had accumulated -- three from one afternoon's crashed
boots, one from ten days earlier -- and they had to be removed BY HAND. The
cache does not self-heal, so a single interrupted boot converts a transient
failure into a permanent one.

What is pinned here:

 1. A complete entry (its `.so` is present) is NEVER touched. That is the
    whole no-data-loss claim: warm caches take minutes to rebuild.
 2. An entry with build residue and no `.so` IS removed.
 3. An entry another process is actively building -- marked, live pid, fresh
    -- is NOT removed. Co-located ranks build into the same cache; a sweep
    that deleted a peer's in-flight directory would be a new bug, not a fix.
 4. Purging is idempotent: a second sweep finds nothing and removes nothing.
 5. `load_jit` heals the specific entry it is about to build, and a cached
    `.so` that fails to load takes its directory with it instead of being
    rebuilt on top of the wreckage.

CPU only: this is filesystem bookkeeping, no compiler and no GPU.
"""

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# THE REAL MODULE, imported -- not re-implemented here.
from sglang.jit_kernel.cache_health import (  # noqa: E402
    MARKER_BUILDING,
    building_marker,
    entry_state,
    heal_entry,
    sweep_cache_root,
)


def _make_complete(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "build.ninja").write_text("rule cc\n")
    (d / f"{name}.so").write_bytes(b"\x7fELF-not-really")
    return d


def _make_poisoned(root: Path, name: str) -> Path:
    """Exactly what a killed nvcc leaves behind: residue, no .so."""
    d = root / name
    d.mkdir(parents=True)
    (d / "build.ninja").write_text("rule cc\n")
    (d / "cuda.cu").write_text("// source\n")
    (d / "cuda_0.o.d").write_text("cuda_0.o: cuda.cu\n")
    return d


def _make_building(root: Path, name: str, pid: int) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "build.ninja").write_text("rule cc\n")
    (d / MARKER_BUILDING).write_text(f"{os.uname().nodename}\n{pid}\n{time.time()}\n")
    return d


class TestEntryState(CustomTestCase):
    def test_states_are_told_apart(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(entry_state(_make_complete(root, "ok")), "complete")
            self.assertEqual(entry_state(_make_poisoned(root, "dead")), "poisoned")
            self.assertEqual(
                entry_state(_make_building(root, "wip", os.getpid())), "building"
            )
            self.assertEqual(entry_state(root / "nope"), "absent")
            (root / "empty").mkdir()
            self.assertEqual(entry_state(root / "empty"), "absent")

    def test_a_marker_from_a_dead_pid_is_poison_not_progress(self):
        """A SIGKILLed builder cannot run its own cleanup.

        If a stale marker counted as 'building' the entry would be immortal --
        precisely the state the four hand-removed directories were in.
        """
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dead = _make_building(root, "killed", 0x7FFFFFFE)
            self.assertEqual(entry_state(dead), "poisoned")

    def test_a_marker_that_is_merely_old_is_poison(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = _make_building(root, "ancient", os.getpid())
            old = time.time() - 10_000
            os.utime(d / MARKER_BUILDING, (old, old))
            self.assertEqual(entry_state(d, stale_seconds=60), "poisoned")


class TestPurge(CustomTestCase):
    def test_only_the_poisoned_entry_is_removed(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            ok = _make_complete(root, "ok")
            dead = _make_poisoned(root, "dead")
            wip = _make_building(root, "wip", os.getpid())

            removed = sweep_cache_root(root)

            self.assertEqual([Path(p).name for p in removed], ["dead"])
            self.assertFalse(dead.exists())
            self.assertTrue(
                (ok / "ok.so").is_file(),
                "the sweep destroyed a VALID cache entry -- minutes of nvcc "
                "thrown away on every boot",
            )
            self.assertTrue(
                wip.is_dir(),
                "the sweep deleted a directory a peer rank was building into",
            )

    def test_sweep_is_idempotent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _make_complete(root, "ok")
            _make_poisoned(root, "dead")
            self.assertEqual(len(sweep_cache_root(root)), 1)
            self.assertEqual(sweep_cache_root(root), [])
            self.assertEqual(sweep_cache_root(root), [])

    def test_sweep_of_a_missing_root_is_a_no_op(self):
        with TemporaryDirectory() as tmp:
            self.assertEqual(sweep_cache_root(Path(tmp) / "never-created"), [])

    def test_heal_entry_targets_one_directory(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dead = _make_poisoned(root, "dead")
            ok = _make_complete(root, "ok")
            self.assertTrue(heal_entry(dead))
            self.assertFalse(dead.exists())
            self.assertFalse(heal_entry(ok))
            self.assertTrue(ok.is_dir())


class TestBuildMarker(CustomTestCase):
    def test_success_leaves_a_complete_entry_and_no_building_marker(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "mod"
            with building_marker(d):
                self.assertEqual(entry_state(d), "building")
                (d / "mod.so").write_bytes(b"\x7fELF")
            self.assertFalse((d / MARKER_BUILDING).exists())
            self.assertEqual(entry_state(d), "complete")

    def test_a_failed_build_is_left_recognizably_poisoned(self):
        """Not left as 'building'.

        An orderly failure knows it failed; it must not hand the next boot a
        directory that looks like someone else's work in progress.
        """
        with TemporaryDirectory() as tmp:
            d = Path(tmp) / "mod"
            with self.assertRaises(RuntimeError):
                with building_marker(d):
                    (d / "cuda.cu").write_text("// half-written\n")
                    raise RuntimeError("nvcc died")
            self.assertEqual(entry_state(d), "poisoned")


class TestLoadJitWiring(CustomTestCase):
    """The seam, checked against the real source of load_jit."""

    def test_load_jit_heals_before_building_and_after_a_failed_load(self):
        import inspect

        from sglang.jit_kernel import utils as jit_utils

        src = inspect.getsource(jit_utils.load_jit)
        self.assertIn("heal_entry", src)
        self.assertIn("building_marker", src)
        # The pre-existing "cached .so failed to load" branch must now take
        # the wreckage with it instead of rebuilding on top of it.
        head, _, tail = src.partition("failed to load; rebuilding")
        self.assertTrue(tail, "the cached-load failure branch disappeared")
        self.assertIn("purge_entry", src)

    def test_the_process_sweep_runs_once_and_is_env_gated(self):
        from unittest import mock

        from sglang.jit_kernel import utils as jit_utils

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            dead = _make_poisoned(root, "dead")

            def _sweep(env_value):
                jit_utils._jit_cache_swept = False
                patches = {"SGLANG_JIT_CACHE_SELFHEAL": env_value}
                with mock.patch.dict(os.environ, patches, clear=False):
                    with mock.patch.object(
                        jit_utils, "_jit_cache_root", lambda: root
                    ):
                        return jit_utils._selfheal_jit_cache_once()

            self.assertEqual(_sweep("0"), [])
            self.assertTrue(
                dead.is_dir(), "SGLANG_JIT_CACHE_SELFHEAL=0 did not disable the sweep"
            )
            self.assertEqual([Path(p).name for p in _sweep("1")], ["dead"])
            self.assertFalse(dead.exists())

            # Once per process: a second call is a no-op even with new poison.
            _make_poisoned(root, "dead2")
            with mock.patch.object(jit_utils, "_jit_cache_root", lambda: root):
                self.assertEqual(jit_utils._selfheal_jit_cache_once(), [])
            self.assertTrue((root / "dead2").is_dir())
            jit_utils._jit_cache_swept = False


class TestLoadJitEndToEnd(CustomTestCase):
    """Drive the real ``load_jit`` with a faked compiler.

    No nvcc and no GPU: ``tvm_ffi.cpp.load_inline`` is replaced by a stub that
    writes the ``.so`` a real build would have produced. Everything else --
    the cache path, the build-dir name, the healing seam -- is the shipping
    code.
    """

    @staticmethod
    def _drive(root, marker, built):
        """Run the real load_jit against `root` with a stubbed compiler."""
        from unittest import mock

        import tvm_ffi.cpp

        from sglang.jit_kernel import utils as jit_utils

        def fake_load_inline(module_name, *a, build_directory=None, **kw):
            built.append(module_name)
            Path(build_directory).mkdir(parents=True, exist_ok=True)
            (Path(build_directory) / f"{module_name}.so").write_bytes(b"\x7fELF")
            return f"module:{module_name}"

        jit_utils._jit_cache_swept = False
        try:
            with mock.patch.dict(
                os.environ, {"TVM_FFI_CACHE_DIR": str(root)}, clear=False
            ):
                with mock.patch.object(tvm_ffi.cpp, "load_inline", fake_load_inline):
                    return jit_utils.load_jit(marker)
        finally:
            jit_utils._jit_cache_swept = False

    def _entry_name(self, marker):
        """The directory load_jit will use, learned rather than predicted.

        The key is composed from the sources, the architecture, the toolchain
        and the flag set; a test that re-derives it would only pin its own copy
        of that composition. One throwaway build tells us the real answer.
        """
        with TemporaryDirectory() as scratch:
            self._drive(scratch, marker, [])
            entries = [p.name for p in Path(scratch).iterdir() if p.is_dir()]
            self.assertEqual(len(entries), 1, entries)
            return entries[0]

    def _run(self, marker, prepare):
        name = f"sgl_kernel_jit_{marker}"
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            entry = root / self._entry_name(marker)
            prepare(entry, name)

            built = []
            result = self._drive(root, marker, built)
            # Snapshot inside the TemporaryDirectory: it is gone after this.
            names = {p.name for p in entry.iterdir()} if entry.is_dir() else set()
            so_bytes = {
                p.name: p.read_bytes() for p in entry.iterdir() if p.suffix == ".so"
            }
            return names, so_bytes, built, result

    def test_a_poisoned_entry_is_discarded_and_rebuilt(self):
        def prepare(entry, name):
            entry.mkdir(parents=True)
            (entry / "build.ninja").write_text("rule cc\n")
            (entry / "cuda.cu").write_text("// source\n")
            (entry / "cuda_0.o.d").write_text("cuda_0.o: cuda.cu\n")

        names, _, built, result = self._run("selfheal_probe_a", prepare)
        self.assertEqual(len(built), 1, "the module was not rebuilt")
        self.assertIn(f"{built[0]}.so", names)
        self.assertNotIn(
            "cuda_0.o.d",
            names,
            "the incomplete build residue survived into the new build "
            "directory; ninja can still declare it up to date",
        )
        self.assertEqual(result, f"module:{built[0]}")

    def test_a_cached_so_that_cannot_be_loaded_takes_its_directory_with_it(self):
        def prepare(entry, name):
            entry.mkdir(parents=True)
            (entry / "build.ninja").write_text("rule cc\n")
            (entry / "cuda_0.o.d").write_text("cuda_0.o: cuda.cu\n")
            # Present, and not a loadable shared object.
            (entry / f"{name}.so").write_bytes(b"truncated")

        names, so_bytes, built, _ = self._run("selfheal_probe_b", prepare)
        self.assertEqual(len(built), 1, "the unusable .so was reused")
        self.assertNotIn(
            "cuda_0.o.d",
            names,
            "the directory of an unloadable .so was rebuilt on top of "
            "instead of replaced",
        )
        self.assertEqual(so_bytes[f"{built[0]}.so"], b"\x7fELF")


if __name__ == "__main__":
    unittest.main()
