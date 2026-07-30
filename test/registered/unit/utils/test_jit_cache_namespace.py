"""A cached kernel from another tree, or another GPU, must not be reused blind.

Falsifier for the third member of the JIT cache-poisoning family (#172 the
cache does not heal itself, #181 the torch-extensions writer, #222 the stale
baton, #208 the architecture confusion). The new case, from the welle-2 card
probe:

    fp8 Marlin GEMM did not run: RuntimeError: Runtime check failed at
    /spinning/wt-merge-probe/python/sglang/jit_kernel/csrc/gemm/marlin/
    gptq_marlin_repack.cuh:355: CUDA error: no kernel image is available for
    execution on the device

Two independent defects produced that one line, and both are pinned here.

ARCHITECTURE. `get_jit_cuda_arch()` resolved `torch.cuda.current_device()`
ONCE per process and every JIT module getter memoized with `cache_once`, which
has no architecture in its key. The probe measures a whole rig from a single
process: it built the module against device 0 (an RTX 5090, sm_120), then
launched that same cubin on the two RTX 3080s. A JIT module is single-arch by
construction -- `sgl_kernel/utils.cuh` static_asserts `__CUDA_ARCH__ ==
SGL_CUDA_ARCH` -- so there is no fatbin to fall back on. The 5090 lane
measured 216 TFLOPS; both 3080 lanes recorded the error above and fell back to
the dequant lane. Loud and correct there, but the same reuse in a serving path
is a wrong kernel, not a message.

NAMESPACE. The cache root is one directory per host, shared by every checkout,
git worktree and wheel install. The entry name carries no tree identity -- by
design, so a warm cache survives a move -- but reuse across trees was assumed
rather than checked, and the entry name did not cover the flags, wrappers or
toolchain that also decide the binary. Hence a `__FILE__` from a worktree the
running process had never read.

CPU only: no compiler and no GPU. `tvm_ffi.cpp.load_inline` is replaced by a
stub that writes the `.so` a real build would have produced; the cache path,
the key composition, the provenance record and the healing seam are all the
shipping code.
"""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

# THE REAL MODULE, imported -- not re-implemented here.
from sglang.jit_kernel import utils as jit_utils  # noqa: E402
from sglang.jit_kernel.utils import (  # noqa: E402
    PROVENANCE_NAME,
    PROVENANCE_VERSION,
    ArchInfo,
    cache_once,
    cache_once_per_arch,
    check_provenance,
)

_SM86 = ArchInfo(8, 6, "")
_SM120 = ArchInfo(12, 0, "")


class _FakeDevices:
    """Pretend this process sees a heterogeneous rig, without a GPU.

    Patches the two seams `get_jit_cuda_arch` actually goes through, so the
    resolution logic under test is the shipping one.
    """

    def __init__(self, archs):
        self._archs = list(archs)
        self.current = 0
        self._patches = []

    def __enter__(self):
        self._saved = dict(jit_utils._DEVICE_ARCH)
        jit_utils._DEVICE_ARCH.clear()
        self._patches = [
            mock.patch.object(
                jit_utils, "_current_device_index", lambda: self.current
            ),
            mock.patch.object(
                jit_utils,
                "_resolve_device_arch",
                lambda index: self._archs[index],
            ),
        ]
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        jit_utils._DEVICE_ARCH.clear()
        jit_utils._DEVICE_ARCH.update(self._saved)
        return False


class TestArchFollowsTheDevice(CustomTestCase):
    def test_the_architecture_is_a_property_of_the_device_not_the_process(self):
        with _FakeDevices([_SM120, _SM86, _SM86]) as rig:
            self.assertEqual(jit_utils.get_jit_cuda_arch().target_name, "12.0")
            rig.current = 1
            self.assertEqual(
                jit_utils.get_jit_cuda_arch().target_name,
                "8.6",
                "the second card was described with the first card's "
                "capability -- this is the defect, not a symptom of it",
            )
            rig.current = 2
            self.assertEqual(jit_utils.get_jit_cuda_arch().target_name, "8.6")
            rig.current = 0
            self.assertEqual(jit_utils.get_jit_cuda_arch().target_name, "12.0")

    def test_an_explicit_override_still_wins_and_nests(self):
        with _FakeDevices([_SM120]):
            with jit_utils.override_jit_cuda_arch(9, 0, "a"):
                self.assertEqual(jit_utils.get_jit_cuda_arch().target_name, "9.0a")
                with jit_utils.override_jit_cuda_arch(10, 0):
                    self.assertEqual(jit_utils.get_jit_cuda_arch().target_name, "10.0")
                self.assertEqual(jit_utils.get_jit_cuda_arch().target_name, "9.0a")
            # Leaving the override returns to the DEVICE, not to whatever the
            # device happened to be when the override was entered.
            self.assertEqual(jit_utils.get_jit_cuda_arch().target_name, "12.0")

    def test_the_failure_sentinel_is_not_memoized(self):
        calls = []

        def flaky(index):
            calls.append(index)
            return ArchInfo(0, 0, "") if len(calls) == 1 else _SM86

        with _FakeDevices([_SM86]):
            with mock.patch.object(jit_utils, "_resolve_device_arch", flaky):
                self.assertEqual(jit_utils.get_jit_cuda_arch().target_name, "0.0")
                self.assertEqual(
                    jit_utils.get_jit_cuda_arch().target_name,
                    "8.6",
                    "a transient detection miss was pinned for the process",
                )

    def test_the_build_dir_name_separates_architectures(self):
        with _FakeDevices([_SM120, _SM86]) as rig:
            first = jit_utils._jit_build_dir_name("sgl_kernel_jit_probe", "abc")
            rig.current = 1
            second = jit_utils._jit_build_dir_name("sgl_kernel_jit_probe", "abc")
        self.assertNotEqual(first, second)
        self.assertIn("arch_12.0", first)
        self.assertIn("arch_8.6", second)
        self.assertTrue(first.endswith("__babc"))


class TestArchScopedMemo(CustomTestCase):
    def test_cache_once_hands_the_first_card_s_module_to_every_card(self):
        """The defect itself, reproduced against the shipping decorator."""
        built = []

        @cache_once
        def getter():
            built.append(jit_utils.get_jit_cuda_arch().target_name)
            return f"module_for_{built[-1]}"

        with _FakeDevices([_SM120, _SM86]) as rig:
            self.assertEqual(getter(), "module_for_12.0")
            rig.current = 1
            self.assertEqual(
                getter(),
                "module_for_12.0",
                "cache_once is expected to be arch-blind; if this ever "
                "changes, cache_once_per_arch is redundant",
            )
        self.assertEqual(built, ["12.0"])

    def test_cache_once_per_arch_builds_one_module_per_architecture(self):
        built = []

        @cache_once_per_arch
        def getter():
            built.append(jit_utils.get_jit_cuda_arch().target_name)
            return f"module_for_{built[-1]}"

        with _FakeDevices([_SM120, _SM86, _SM86]) as rig:
            self.assertEqual(getter(), "module_for_12.0")
            rig.current = 1
            self.assertEqual(getter(), "module_for_8.6")
            rig.current = 2
            # Same architecture as device 1: reused, not rebuilt.
            self.assertEqual(getter(), "module_for_8.6")
            rig.current = 0
            self.assertEqual(getter(), "module_for_12.0")
        self.assertEqual(built, ["12.0", "8.6"])

    def test_arguments_still_key_the_memo(self):
        @cache_once_per_arch
        def getter(dtype, flag=False):
            return (jit_utils.get_jit_cuda_arch().target_name, dtype, flag)

        with _FakeDevices([_SM86]):
            self.assertEqual(getter("bf16"), ("8.6", "bf16", False))
            self.assertEqual(getter("fp16"), ("8.6", "fp16", False))
            self.assertEqual(getter("bf16", flag=True), ("8.6", "bf16", True))
            self.assertEqual(len(getter.arch_scoped_cache), 3)

    def test_every_jit_module_getter_uses_the_arch_scoped_memo(self):
        """No module getter may be left on the arch-blind decorator.

        The audit, kept executable: a new kernel file that copies the old
        pattern is the way this defect comes back.
        """
        import ast

        root = Path(jit_utils.__file__).resolve().parents[1]
        offenders = []
        for path in sorted(root.rglob("*.py")):
            text = path.read_text()
            if "load_jit(" not in text or "cache_once" not in text:
                continue
            try:
                tree = ast.parse(text)
            except SyntaxError:  # pragma: no cover - not our source
                continue
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not any(
                    isinstance(d, ast.Name) and d.id == "cache_once"
                    for d in node.decorator_list
                ):
                    continue
                body = ast.get_source_segment(text, node) or ""
                if "load_jit(" in body:
                    offenders.append(f"{path.name}:{node.lineno} {node.name}")
        self.assertEqual(
            offenders,
            [],
            "these JIT module getters memoize across architectures; use "
            "@cache_once_per_arch",
        )


class TestBuildInputHash(CustomTestCase):
    """Everything that decides the binary has to reach the key."""

    BASE = dict(
        arch="8.6",
        vendor="cuda",
        backend="cuda",
        header_only=True,
        cpp_wrappers=[],
        cuda_wrappers=[("export", "kernel")],
        extra_cflags=["-std=c++20"],
        extra_cuda_cflags=["-DSGL_CUDA_ARCH=860"],
        extra_ldflags=[],
        dependencies=["cutlass"],
    )

    def _hash(self, **overrides):
        args = dict(self.BASE)
        args.update(overrides)
        return jit_utils._build_input_hash(**args)

    def test_identical_inputs_give_an_identical_hash(self):
        self.assertEqual(self._hash(), self._hash())

    def test_each_input_moves_the_hash(self):
        base = self._hash()
        for label, overrides in (
            ("arch", dict(arch="12.0")),
            ("vendor", dict(vendor="hip")),
            ("backend", dict(backend="hip")),
            ("header_only", dict(header_only=False)),
            ("cuda_wrappers", dict(cuda_wrappers=[("export", "other_kernel")])),
            ("extra_cflags", dict(extra_cflags=["-std=c++17"])),
            ("extra_cuda_cflags", dict(extra_cuda_cflags=["-DTILE=64"])),
            ("extra_ldflags", dict(extra_ldflags=["-lcuda"])),
            ("dependencies", dict(dependencies=["flashinfer"])),
        ):
            with self.subTest(input=label):
                self.assertNotEqual(
                    base,
                    self._hash(**overrides),
                    f"{label} does not reach the cache key",
                )

    def test_dependency_order_is_not_a_difference(self):
        a = self._hash(dependencies=["cutlass", "flashinfer"])
        b = self._hash(dependencies=["flashinfer", "cutlass"])
        self.assertEqual(a, b)

    def test_flags_that_only_differ_in_a_define_are_told_apart(self):
        """The real shape: template parameters arrive as -D flags."""
        a = self._hash(extra_cuda_cflags=["-DBLOCK_M=64"])
        b = self._hash(extra_cuda_cflags=["-DBLOCK_M=128"])
        self.assertNotEqual(a, b)


class TestCheckProvenance(CustomTestCase):
    def _record(self, **overrides):
        record = {
            "version": PROVENANCE_VERSION,
            "module": "sgl_kernel_jit_probe_deadbeefdeadbeef",
            "source_hash": "deadbeefdeadbeef",
            "build_hash": "0123456789ab",
            "target_archs": ["8.6"],
            "vendor": "cuda",
            "source_tree": "/spinning/wt-final/python/sglang/jit_kernel",
        }
        record.update(overrides)
        return record

    def _entry(self, root, stored):
        d = Path(root) / "entry"
        d.mkdir(parents=True, exist_ok=True)
        if stored is not None:
            (d / PROVENANCE_NAME).write_text(json.dumps(stored))
        return d

    def test_a_matching_record_is_accepted(self):
        with TemporaryDirectory() as tmp:
            d = self._entry(tmp, self._record())
            ok, why = check_provenance(d, self._record())
            self.assertTrue(ok, why)

    def test_the_same_kernel_from_another_worktree_is_accepted_and_named(self):
        """Content-addressed sharing is the point; the surprise was not.

        Identical inputs mean an identical binary, so reuse across trees is
        correct and worth keeping -- a cold rebuild of every kernel costs
        minutes of nvcc each. What must not happen is that reuse being
        invisible, which is how a `__FILE__` from an unrelated worktree ended
        up in a runtime error nobody could place.
        """
        with TemporaryDirectory() as tmp:
            foreign = self._record(
                source_tree="/spinning/wt-merge-probe/python/sglang/jit_kernel",
                host="other-host",
                pid=4242,
            )
            d = self._entry(tmp, foreign)
            ok, why = check_provenance(d, self._record())
            self.assertTrue(ok, why)
            self.assertIn("wt-merge-probe", why)

    def test_a_foreign_namespace_hit_is_refused(self):
        """Same name, different sources: the synthetic poisoning case."""
        with TemporaryDirectory() as tmp:
            d = self._entry(tmp, self._record(source_hash="ffffffffffffffff"))
            ok, why = check_provenance(d, self._record())
            self.assertFalse(ok)
            self.assertIn("source_hash mismatch", why)

    def test_a_different_flag_set_is_refused(self):
        with TemporaryDirectory() as tmp:
            d = self._entry(tmp, self._record(build_hash="ffffffffffff"))
            ok, why = check_provenance(d, self._record())
            self.assertFalse(ok)
            self.assertIn("build_hash mismatch", why)

    def test_an_artefact_for_another_architecture_is_refused(self):
        """The observed failure, at the point it should have been caught."""
        with TemporaryDirectory() as tmp:
            d = self._entry(tmp, self._record(target_archs=["12.0"]))
            ok, why = check_provenance(d, self._record(target_archs=["8.6"]))
            self.assertFalse(ok)
            self.assertIn("architecture mismatch", why)
            self.assertIn("12.0", why)
            self.assertIn("8.6", why)

    def test_an_artefact_covering_more_architectures_is_accepted(self):
        with TemporaryDirectory() as tmp:
            d = self._entry(tmp, self._record(target_archs=["8.6", "12.0"]))
            ok, why = check_provenance(d, self._record(target_archs=["8.6"]))
            self.assertTrue(ok, why)

    def test_another_vendor_is_refused(self):
        with TemporaryDirectory() as tmp:
            d = self._entry(tmp, self._record(vendor="hip"))
            ok, why = check_provenance(d, self._record(vendor="cuda"))
            self.assertFalse(ok)
            self.assertIn("vendor mismatch", why)

    def test_a_missing_record_is_refused_not_trusted(self):
        with TemporaryDirectory() as tmp:
            d = self._entry(tmp, None)
            ok, why = check_provenance(d, self._record())
            self.assertFalse(ok)
            self.assertIn("no provenance record", why)

    def test_an_unreadable_record_is_refused(self):
        with TemporaryDirectory() as tmp:
            d = self._entry(tmp, None)
            (d / PROVENANCE_NAME).write_text("{not json")
            ok, why = check_provenance(d, self._record())
            self.assertFalse(ok)
            self.assertIn("unreadable", why)

    def test_a_record_from_an_older_scheme_is_refused(self):
        with TemporaryDirectory() as tmp:
            d = self._entry(tmp, self._record(version=PROVENANCE_VERSION - 1))
            ok, why = check_provenance(d, self._record())
            self.assertFalse(ok)
            self.assertIn("provenance version", why)


class TestLoadJitNamespaceEndToEnd(CustomTestCase):
    """Drive the real `load_jit` with a faked compiler."""

    def _harness(self, root, built):
        import tvm_ffi
        import tvm_ffi.cpp

        def fake_load_inline(module_name, *a, build_directory=None, **kw):
            built.append(module_name)
            Path(build_directory).mkdir(parents=True, exist_ok=True)
            (Path(build_directory) / f"{module_name}.so").write_bytes(b"\x7fELF")
            return f"module:{module_name}"

        def fake_load_module(path):
            if Path(path).read_bytes() != b"\x7fELF":
                raise RuntimeError("not a shared object")
            return f"module:{Path(path).stem}"

        return (
            mock.patch.dict(os.environ, {"TVM_FFI_CACHE_DIR": str(root)}, clear=False),
            mock.patch.object(tvm_ffi.cpp, "load_inline", fake_load_inline),
            mock.patch.object(tvm_ffi, "load_module", fake_load_module),
        )

    def _run(self, root, marker, built):
        jit_utils._jit_cache_swept = False
        try:
            patches = self._harness(root, built)
            for p in patches:
                p.start()
            try:
                return jit_utils.load_jit(marker)
            finally:
                for p in reversed(patches):
                    p.stop()
        finally:
            jit_utils._jit_cache_swept = False

    @staticmethod
    def _only_entry(root):
        entries = [p for p in Path(root).iterdir() if p.is_dir()]
        assert len(entries) == 1, entries
        return entries[0]

    def test_a_build_records_its_provenance(self):
        built = []
        with TemporaryDirectory() as tmp:
            self._run(tmp, "namespace_probe_a", built)
            entry = self._only_entry(tmp)
            record = json.loads((entry / PROVENANCE_NAME).read_text())
        self.assertEqual(len(built), 1)
        self.assertEqual(record["version"], PROVENANCE_VERSION)
        self.assertEqual(record["module"], built[0])
        self.assertEqual(record["source_tree"], str(jit_utils.KERNEL_PATH))
        self.assertEqual(
            record["target_archs"], [jit_utils.get_jit_cuda_arch().target_name]
        )
        self.assertIn("build_hash", record)
        self.assertTrue(entry.name.endswith(f"__b{record['build_hash']}"))

    def test_a_verified_entry_is_reused_without_rebuilding(self):
        built = []
        with TemporaryDirectory() as tmp:
            first = self._run(tmp, "namespace_probe_b", built)
            second = self._run(tmp, "namespace_probe_b", built)
        self.assertEqual(len(built), 1, "a verified warm entry was rebuilt")
        self.assertEqual(first, second)

    def test_an_entry_stamped_by_another_worktree_is_still_reused(self):
        built = []
        with TemporaryDirectory() as tmp:
            self._run(tmp, "namespace_probe_c", built)
            entry = self._only_entry(tmp)
            path = entry / PROVENANCE_NAME
            record = json.loads(path.read_text())
            record["source_tree"] = "/spinning/wt-merge-probe/python/sglang/jit_kernel"
            record["host"] = "another-host"
            path.write_text(json.dumps(record))
            self._run(tmp, "namespace_probe_c", built)
        self.assertEqual(
            len(built), 1, "content-addressed sharing across trees was lost"
        )

    def test_an_entry_from_a_foreign_namespace_is_refused_and_rebuilt(self):
        """A .so under our name whose record says it is somebody else's."""
        built = []
        with TemporaryDirectory() as tmp:
            self._run(tmp, "namespace_probe_d", built)
            entry = self._only_entry(tmp)
            path = entry / PROVENANCE_NAME
            record = json.loads(path.read_text())
            record["source_hash"] = "ffffffffffffffff"
            record["source_tree"] = "/spinning/wt-merge-probe/python/sglang/jit_kernel"
            path.write_text(json.dumps(record))
            (entry / "cuda_0.o.d").write_text("cuda_0.o: cuda.cu\n")

            with self.assertLogs("sglang.jit_kernel.utils", level="WARNING") as logs:
                self._run(tmp, "namespace_probe_d", built)
            entry = self._only_entry(tmp)
            names = {p.name for p in entry.iterdir()}
        self.assertEqual(len(built), 2, "the foreign entry was reused")
        self.assertTrue(any("Discarding cached JIT module" in m for m in logs.output))
        self.assertTrue(
            any("source_hash mismatch" in m for m in logs.output),
            "the refusal did not say WHY, which is how a triage ends up "
            "chasing a foreign __FILE__ instead of a cache entry",
        )
        self.assertNotIn(
            "cuda_0.o.d",
            names,
            "the refused entry was rebuilt on top of instead of replaced",
        )

    def test_an_entry_for_another_architecture_is_refused_and_rebuilt(self):
        built = []
        with TemporaryDirectory() as tmp:
            self._run(tmp, "namespace_probe_e", built)
            entry = self._only_entry(tmp)
            path = entry / PROVENANCE_NAME
            record = json.loads(path.read_text())
            # Exactly the welle-2 artefact: an sm_120 cubin sitting where an
            # sm_86 rank looks for one.
            record["target_archs"] = ["12.0"]
            path.write_text(json.dumps(record))

            with self.assertLogs("sglang.jit_kernel.utils", level="WARNING") as logs:
                self._run(tmp, "namespace_probe_e", built)
        self.assertEqual(len(built), 2, "the cross-arch artefact was reused")
        self.assertTrue(any("architecture mismatch" in m for m in logs.output))

    def test_the_check_can_be_turned_off_in_place(self):
        """The operator lever, matching SGLANG_JIT_CACHE_SELFHEAL."""
        built = []
        with TemporaryDirectory() as tmp:
            self._run(tmp, "namespace_probe_h", built)
            entry = self._only_entry(tmp)
            (entry / PROVENANCE_NAME).unlink()
            with mock.patch.dict(
                os.environ, {"SGLANG_JIT_PROVENANCE_CHECK": "0"}, clear=False
            ):
                self._run(tmp, "namespace_probe_h", built)
        self.assertEqual(len(built), 1, "the kill switch did not disable the check")

    def test_an_entry_without_a_record_is_refused_and_rebuilt(self):
        built = []
        with TemporaryDirectory() as tmp:
            self._run(tmp, "namespace_probe_f", built)
            entry = self._only_entry(tmp)
            (entry / PROVENANCE_NAME).unlink()

            with self.assertLogs("sglang.jit_kernel.utils", level="WARNING") as logs:
                self._run(tmp, "namespace_probe_f", built)
        self.assertEqual(len(built), 2, "an unverifiable entry was reused")
        self.assertTrue(any("no provenance record" in m for m in logs.output))

    def test_two_architectures_in_one_process_get_two_entries(self):
        """The welle-2 shape, end to end: one process, a mixed rig."""
        built = []
        with TemporaryDirectory() as tmp:
            with _FakeDevices([_SM120, _SM86]) as rig:
                self._run(tmp, "namespace_probe_g", built)
                rig.current = 1
                self._run(tmp, "namespace_probe_g", built)
            entries = sorted(p.name for p in Path(tmp).iterdir() if p.is_dir())
            archs = sorted(
                json.loads((Path(tmp) / name / PROVENANCE_NAME).read_text())[
                    "target_archs"
                ][0]
                for name in entries
            )
        self.assertEqual(len(built), 2, "both cards shared one build directory")
        self.assertEqual(archs, ["12.0", "8.6"])
        self.assertTrue(any("arch_12.0" in n for n in entries))
        self.assertTrue(any("arch_8.6" in n for n in entries))


if __name__ == "__main__":
    unittest.main()
