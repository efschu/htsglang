"""The torch_extensions cache poisons boots exactly like the JIT cache did.

THE DEFECT (#181, same class as #172b, different writer)
-------------------------------------------------------
``htccl_device._load_ext`` builds the HTCCL device extension with
``torch.utils.cpp_extension.load_inline`` -- roughly 150 s of nvcc into
``$TORCH_EXTENSIONS_DIR/<name>/``. A boot killed inside that window leaves the
directory holding ``main.cpp``, ``cuda.cu``, ``build.ninja`` and ``*.o``, and
no ``.so``. torch hands that directory to every later boot, and ninja's mtime
check is happy to declare the wreck up to date, so one interrupted build turns
into a permanent failure -- the worst property a cache can have, and the same
one #172b removed from the tvm-ffi cache.

WHY THIS IS NOT THE SAME FIX TWICE
----------------------------------
The mechanics are reused from ``sglang.jit_kernel.cache_health`` (completeness
by artifact, host+pid+time build marker with a liveness check so a co-located
rank's live build is never mistaken for residue, rename-before-delete purge,
env kill switch, and never failing a boot on cache hygiene). Only the two
things that genuinely differ are passed in: what the finished artifact is, and
WHICH entries the sweep may judge.

That second one is the sharp edge here and has no counterpart in #172b: the
tvm-ffi root belongs to sglang alone, but torch's extensions root is SHARED
with every other cpp_extension on the machine. A half-built entry there may be
another extension's live business. So the sweep is scoped by name to
``htccl_device_ext*``, and the tests below pin that a foreign wreck survives
it. An unscoped sweep of that root would be a new bug, not a fix.

WHAT IS PINNED
--------------
 1. Residue under the name this rank is about to build is purged BEFORE
    load_inline sees the directory (the falsifier: pre-fix, load_inline was
    handed the wreck).
 2. A FOREIGN poisoned entry under the same root is never touched.
 3. A complete entry (its ``.so`` is present) is never touched -- 150 s of
    nvcc is not thrown away on every boot.
 4. A co-located peer's live build directory is never touched.
 5. The kill switch ``SGLANG_EXT_CACHE_SELFHEAL=0`` disables the sweep.
 6. Cache hygiene NEVER fails a boot: if the sweep itself raises, the build
    still runs.
 7. The build is wrapped in the build marker, so the next process can tell
    "a peer is compiling" from "somebody was killed compiling".

CPU only: no nvcc, no GPU, no process group. ``load_inline`` is replaced by a
stub that writes the ``.so`` a real build would have produced, and the arch
resolution (the only collective in ``_load_ext``) is stubbed out. Everything
else -- the cache path, the name, the healing seam -- is shipping code.
"""

import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

from sglang.jit_kernel.cache_health import (  # noqa: E402
    MARKER_BUILDING,
    sweep_cache_root,
)
from sglang.srt.distributed.device_communicators import htccl_device as hd  # noqa: E402

#: What the stubbed arch resolution reports, and the entry name it implies.
_ARCHES = {"cuda": ["8.6", "12.0"]}
_NAME = "htccl_device_ext_cuda_86_120"


def _residue(d: Path) -> Path:
    """Exactly what a killed nvcc leaves: sources, ninja, objects, no .so."""
    d.mkdir(parents=True, exist_ok=True)
    (d / "build.ninja").write_text("rule cuda_compile\n")
    (d / "main.cpp").write_text("// half-written\n")
    (d / "cuda.cu").write_text("// half-written\n")
    (d / "main.o").write_bytes(b"\x7fELF-object")
    return d


def _complete(d: Path, name: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    (d / "build.ninja").write_text("rule cuda_compile\n")
    (d / f"{name}.so").write_bytes(b"\x7fELF-not-really")
    return d


def _building(d: Path, pid: int) -> Path:
    _residue(d)
    (d / MARKER_BUILDING).write_text(f"{os.uname().nodename}\n{pid}\n{time.time()}\n")
    return d


class HtcclExtCacheSelfHealTest(CustomTestCase):
    def setUp(self):
        self._saved_ext = hd._ext
        hd._ext = None

    def tearDown(self):
        hd._ext = self._saved_ext

    def _load(self, root: Path, env=None):
        """Drive the real ``_load_ext`` with a faked compiler.

        Returns (seen, marked, built) where `seen` is the set of file names
        that were present in the build directory at the moment load_inline was
        called, and `marked` says whether the build ran under the build marker.
        """
        import torch.utils.cpp_extension as cppext

        seen: list = []
        marked: list = []

        def fake_load_inline(name, *a, build_directory=None, **kw):
            d = Path(build_directory) if build_directory else root / name
            d.mkdir(parents=True, exist_ok=True)
            seen.append({p.name for p in d.iterdir()})
            marked.append((d / MARKER_BUILDING).is_file())
            (d / f"{name}.so").write_bytes(b"\x7fELF")
            return f"module:{name}"

        environ = {"TORCH_EXTENSIONS_DIR": str(root)}
        environ.update(env or {})
        with mock.patch.dict(os.environ, environ, clear=False):
            with mock.patch.object(hd, "_local_vendor", lambda: "cuda"):
                with mock.patch.object(
                    hd, "_resolve_build_arches", lambda group: _ARCHES
                ):
                    with mock.patch.object(cppext, "load_inline", fake_load_inline):
                        built = hd._load_ext(None)
        return (seen[0] if seen else None), (marked[0] if marked else None), built

    # ---------------------------------------------------------------- 1, 7

    def test_residue_is_purged_before_the_build_and_the_build_is_marked(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _residue(root / _NAME)

            seen, marked, built = self._load(root)

            self.assertEqual(built, f"module:{_NAME}")
            self.assertNotIn(
                "main.o",
                seen,
                "the interrupted build's residue was handed to load_inline; "
                "ninja's mtime check can declare that wreck up to date and "
                "the boot fails forever",
            )
            self.assertNotIn("build.ninja", seen)
            self.assertTrue(
                marked,
                "the build did not run under the build marker, so a "
                "co-located rank's sweep cannot tell it from residue",
            )
            self.assertTrue((root / _NAME / f"{_NAME}.so").is_file())

    # ------------------------------------------------------------------- 2

    def test_a_foreign_poisoned_entry_is_never_touched(self):
        """The torch extensions root is shared. Scope, or cause a new bug."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _residue(root / _NAME)
            foreign = _residue(root / "flashinfer_jit_ext")
            other = _residue(root / "some_third_party_kernels")
            (root / "lock").write_text("")  # torch's own baton file

            self._load(root)

            self.assertTrue(
                (foreign / "main.o").is_file(),
                "the sweep deleted ANOTHER extension's build directory",
            )
            self.assertTrue((other / "main.o").is_file())
            self.assertTrue((root / "lock").is_file())

    # ------------------------------------------------------------------- 3

    def test_a_complete_entry_of_our_own_is_never_touched(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            warm = _complete(root / "htccl_device_ext_cuda_75", "htccl_device_ext_cuda_75")
            _residue(root / _NAME)

            self._load(root)

            self.assertTrue(
                (warm / "htccl_device_ext_cuda_75.so").is_file(),
                "a warm HTCCL extension was discarded -- 150 s of nvcc per boot",
            )

    # ------------------------------------------------------------------- 4

    def test_a_co_located_peers_live_build_is_never_touched(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            peer = _building(root / "htccl_device_ext_cuda_120", os.getpid())
            _residue(root / _NAME)

            self._load(root)

            self.assertTrue(
                (peer / "main.o").is_file(),
                "the sweep deleted a directory a co-located rank was "
                "building into",
            )

    # ------------------------------------------------------------------- 5

    def test_the_kill_switch_disables_the_sweep(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _residue(root / _NAME)

            seen, _, _ = self._load(root, env={"SGLANG_EXT_CACHE_SELFHEAL": "0"})

            self.assertIn(
                "main.o",
                seen,
                "SGLANG_EXT_CACHE_SELFHEAL=0 did not disable the sweep",
            )

    # ------------------------------------------------------------------- 6

    def test_cache_hygiene_never_fails_a_boot(self):
        from sglang.jit_kernel import cache_health

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _residue(root / _NAME)

            def _explode(*a, **kw):
                raise OSError("cache volume gone")

            with mock.patch.object(cache_health, "sweep_cache_root", _explode):
                _, _, built = self._load(root)
            self.assertEqual(
                built,
                f"module:{_NAME}",
                "a failure in cache hygiene took the boot down with it",
            )


class ScopedSweepTest(CustomTestCase):
    """The generalisation in cache_health, exercised directly."""

    def test_name_filter_confines_the_sweep(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mine = _residue(root / "htccl_device_ext_cuda_86")
            theirs = _residue(root / "vendor_ext")

            removed = sweep_cache_root(
                root,
                name_filter=lambda n: n.startswith("htccl_device_ext"),
                label="torch extension cache",
            )

            self.assertEqual([Path(p).name for p in removed], [mine.name])
            self.assertFalse(mine.exists())
            self.assertTrue((theirs / "main.o").is_file())

    def test_a_pyd_counts_as_a_finished_artifact(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            d = _residue(root / "htccl_device_ext_win")
            (d / "htccl_device_ext_win.pyd").write_bytes(b"MZ")
            self.assertEqual(
                sweep_cache_root(root, artifact_suffixes=(".so", ".pyd")), []
            )
            self.assertTrue(d.is_dir())
            # ...and with the default suffix set it is poison, which is why the
            # suffixes are the caller's to state.
            self.assertEqual(len(sweep_cache_root(root)), 1)

    def test_only_our_own_purge_debris_is_collected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mine = root / ".__sglpurge-1234-5678-htccl_device_ext_cuda_86"
            theirs = root / ".__sglpurge-1234-5678-vendor_ext"
            unparseable = root / ".__sglpurge-garbage"
            for d in (mine, theirs, unparseable):
                d.mkdir()
                (d / "main.o").write_bytes(b"x")

            sweep_cache_root(
                root, name_filter=lambda n: n.startswith("htccl_device_ext")
            )

            self.assertFalse(mine.exists())
            self.assertTrue(theirs.is_dir(), "foreign purge debris was collected")
            self.assertTrue(unparseable.is_dir())


if __name__ == "__main__":
    unittest.main()
