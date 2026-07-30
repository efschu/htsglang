"""Environment vs. card-level failures in the stage-0 hardware probe
(task #310).

A venv without ``sgl_kernel`` is not evidence about any GPU: with sgl_kernel
not importable, ``sglang.srt.layers.quantization.utils.get_scalar_types()``
falls back to a ``MockScalarTypes`` stand-in, ``marlin_utils_fp8`` resolves
``scalar_types.float8_e4m3fn`` against it as a plain mock string instead of a
real ``ScalarType``, and the fp8 Marlin lane probe's real sgl_kernel call
(``b_q_type.id``) then raises ``AttributeError``. Before this task, that
``AttributeError`` was caught and stored as the lane's REASON --
``"fp8 Marlin GEMM did not run: AttributeError: ..."`` -- for every probed
card, including ones that run the lane natively. A missing dependency in the
probing interpreter silently became a permanent, wrong hardware verdict in
the cached profile.

What is asserted here, all on CPU, none of it touching a real GPU:

* ``_check_lane_probe_environment`` refuses an interpreter with no
  importable ``sgl_kernel``, or with ``sgl_kernel`` importable but
  ``get_scalar_types()`` still returning the ``MockScalarTypes`` fallback,
  and names the interpreter in the error;
* an interpreter with real scalar types passes the check without raising;
* ``_profile_note`` persists ``NOTE_CLASS_CARD`` reasons exactly as before
  this task, and logs-but-never-persists ``NOTE_CLASS_ENVIRONMENT`` ones;
* a lazy top-up whose subprocess failed for an environment reason (detected
  via ``PROBE_ENV_ERROR_PREFIX``) leaves the cached profile with NO note for
  the fields it could not measure; the same failure without the marker (a
  card-relevant top-up failure) still records its reason, unchanged;
* the actual ``python -m sglang.srt.uneven_perf --probe`` entry point aborts
  BEFORE writing anything and exits non-zero when ``sgl_kernel`` cannot be
  imported, and does not even attempt the guard when the requested probe
  groups do not include ``lanes``.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from sglang.srt import uneven_perf
from sglang.srt.layers.quantization import utils as quant_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_DRIVER = "595.58.03"
_UUIDS = ["GPU-aaa", "GPU-bbb"]
_GPUS = [
    {"cuda_index": 0, "uuid": "GPU-aaa", "name": "RTX 5090", "total_mib": 32607},
    {"cuda_index": 1, "uuid": "GPU-bbb", "name": "RTX 3080", "total_mib": 20480},
]


class MockScalarTypes:
    """Same shape (and, load-bearingly, the same CLASS NAME) as the real
    fallback in ``sglang.srt.layers.quantization.utils.get_scalar_types``:
    the check identifies the fallback by ``type(...).__name__``, not by
    identity, since it never imports the real utils fallback class."""

    def __getattr__(self, name):
        return f"mock_{name}"


class _RealLikeScalarTypes:
    """Any object whose class is NOT named MockScalarTypes -- stands in for
    the real ``sgl_kernel.scalar_type.scalar_types`` singleton."""


class TestCheckLaneProbeEnvironment(CustomTestCase):
    """The guard function in isolation, sgl_kernel state fully controlled."""

    def test_sgl_kernel_not_importable_raises_and_names_the_interpreter(self):
        # sys.modules[name] = None is the standard way to force `import name`
        # to raise ImportError, regardless of what is actually installed.
        with mock.patch.dict("sys.modules", {"sgl_kernel": None}):
            with self.assertRaises(uneven_perf.ProbeEnvironmentError) as cm:
                uneven_perf._check_lane_probe_environment()
        msg = str(cm.exception)
        self.assertIn(sys.executable, msg)
        self.assertIn("sgl_kernel", msg)
        self.assertIn("lanes", msg)

    def test_mock_scalar_types_raises_even_if_sgl_kernel_itself_imports(self):
        """The exact bug: sgl_kernel imports fine, but get_scalar_types()
        still hands back the mock (e.g. a broken sgl_kernel.scalar_type
        submodule)."""
        fake_sgl_kernel = mock.MagicMock()
        with mock.patch.dict(
            "sys.modules", {"sgl_kernel": fake_sgl_kernel}
        ), mock.patch.object(
            quant_utils,
            "get_scalar_types",
            return_value=(object, MockScalarTypes()),
        ):
            with self.assertRaises(uneven_perf.ProbeEnvironmentError) as cm:
                uneven_perf._check_lane_probe_environment()
        msg = str(cm.exception)
        self.assertIn(sys.executable, msg)
        self.assertIn("MockScalarTypes", msg)

    def test_real_scalar_types_pass_without_raising(self):
        fake_sgl_kernel = mock.MagicMock()
        with mock.patch.dict(
            "sys.modules", {"sgl_kernel": fake_sgl_kernel}
        ), mock.patch.object(
            quant_utils,
            "get_scalar_types",
            return_value=(object, _RealLikeScalarTypes()),
        ):
            uneven_perf._check_lane_probe_environment()  # must not raise


class TestProfileNoteClassification(CustomTestCase):
    """``_profile_note``: the class an existing card-level reason always had
    (``NOTE_CLASS_CARD``) is unaffected; the new ``NOTE_CLASS_ENVIRONMENT``
    class is logged and never written into the profile."""

    def test_card_note_is_persisted_exactly_as_before(self):
        profile = {}
        uneven_perf._profile_note(profile, "gemm_lanes", "a card fact")
        self.assertEqual(
            profile[uneven_perf.PROFILE_NOTES_KEY], {"gemm_lanes": "a card fact"}
        )

    def test_card_note_class_is_explicit_and_equivalent_to_the_default(self):
        profile = {}
        uneven_perf._profile_note(
            profile,
            "gemm_lanes",
            "a card fact",
            note_class=uneven_perf.NOTE_CLASS_CARD,
        )
        self.assertEqual(
            profile[uneven_perf.PROFILE_NOTES_KEY], {"gemm_lanes": "a card fact"}
        )

    def test_environment_error_is_logged_and_never_written(self):
        profile = {}
        with mock.patch.object(uneven_perf.logger, "warning") as warn:
            uneven_perf._profile_note(
                profile,
                "gemm_lanes",
                "sgl_kernel missing in this venv",
                note_class=uneven_perf.NOTE_CLASS_ENVIRONMENT,
            )
        self.assertNotIn(uneven_perf.PROFILE_NOTES_KEY, profile)
        warn.assert_called_once()
        self.assertIn("sgl_kernel missing in this venv", warn.call_args[0][0] % warn.call_args[0][1:])


class _CacheFixture:
    """A cache directory holding a v2 profile (no lanes) for a 2-card rig,
    self-contained here rather than imported from test_profile_migration.py
    (which defines its own equivalent fixture) -- same convention already
    used by test_gemm_lane_format.py."""

    def __init__(self, tmpdir):
        self.dir = tmpdir
        gpus = {}
        for i, g in enumerate(_GPUS):
            gpus[g["uuid"]] = {
                "name": g["name"],
                "cuda_index": i,
                "total_mib": g["total_mib"],
                "gemm_tflops": 232.97 if i == 0 else 62.72,
                "membw_gbs": 1664.2 if i == 0 else 717.8,
                "membw_read_gbs": 1664.2 if i == 0 else 717.8,
                "membw_copy_gbs": 1520.0 if i == 0 else 700.0,
                "membw_gemv_gbs": 1529.7 if i == 0 else 717.8,
            }
        self.profile = {
            "version": 2,
            "driver": _DRIVER,
            "uuids": sorted(_UUIDS),
            "gpus": gpus,
            "links": {"GPU-aaa|GPU-bbb": {"p2p_gbs": 5.1}},
            "probe_seconds": 8.1,
            "created": "2026-07-27 15:34:30",
        }
        self.old_path = uneven_perf.profile_cache_path(_UUIDS, _DRIVER, version=2)
        self.new_path = uneven_perf.profile_cache_path(_UUIDS, _DRIVER)
        with open(self.old_path, "w") as f:
            json.dump(self.profile, f)


class TestTopUpEnvironmentFailureIsNotPersisted(CustomTestCase):
    """The lazy top-up path (``_migrated_profile`` -> ``get_hardware_profile``):
    an environment-marked subprocess failure must not leave a note behind."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = mock.patch.object(uneven_perf, "PROFILE_CACHE_DIR", self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        inv = mock.patch.object(
            uneven_perf, "_nvml_gpu_inventory", return_value=(_GPUS, _DRIVER)
        )
        inv.start()
        self.addCleanup(inv.stop)
        self.cache = _CacheFixture(self._tmp.name)

    def test_environment_marked_failure_leaves_no_note(self):
        marked = (
            f"{uneven_perf.PROBE_ENV_ERROR_PREFIX}the 'lanes' probe group "
            f"needs sgl_kernel, which is not importable in /fake/venv/python"
        )
        with mock.patch.object(
            uneven_perf, "_run_probe_subprocess", return_value=marked
        ):
            profile, source, _ = uneven_perf.get_hardware_profile()
        self.assertIn("top-up failed", source)
        self.assertEqual(profile["version"], uneven_perf.PROFILE_VERSION)
        # The pre-existing v2 values are still carried over ...
        self.assertEqual(profile["gpus"]["GPU-aaa"]["membw_gemv_gbs"], 1529.7)
        self.assertNotIn("gemm_lanes", profile["gpus"]["GPU-aaa"])
        # ... but NO note was recorded for the fields the top-up could not
        # measure: an environment fact must never survive into the cache.
        notes = profile.get(uneven_perf.PROFILE_NOTES_KEY, {})
        self.assertNotIn("gemm_lanes", notes)
        self.assertNotIn("gemm_lane_notes", notes)
        # And the file on disk agrees with the in-memory profile returned.
        with open(self.cache.new_path) as f:
            written = json.load(f)
        written_notes = written.get(uneven_perf.PROFILE_NOTES_KEY, {})
        self.assertNotIn("gemm_lanes", written_notes)

    def test_an_unmarked_failure_still_records_a_card_note_unchanged(self):
        """Same failure shape as before this task (no PROBE_ENV_ERROR_PREFIX):
        the existing behaviour -- a persisted, named reason -- must not
        regress."""
        with mock.patch.object(
            uneven_perf,
            "_run_probe_subprocess",
            return_value="probe subprocess exited with rc=1",
        ):
            profile, source, _ = uneven_perf.get_hardware_profile()
        self.assertIn("top-up failed", source)
        note = profile[uneven_perf.PROFILE_NOTES_KEY]["gemm_lanes"]
        self.assertIn("rc=1", note)
        self.assertIn("SGLANG_PERF_REPROBE", note)


class TestRunProbeSubprocessMarkerDetection(CustomTestCase):
    """``_run_probe_subprocess`` recognizes ``PROBE_ENV_ERROR_PREFIX`` on the
    subprocess's stderr and keeps it in the returned failure string."""

    def _fake_run(self, returncode, stderr):
        result = mock.MagicMock()
        result.returncode = returncode
        result.stderr = stderr
        result.stdout = ""
        return result

    def test_the_marker_line_is_extracted_from_stderr(self):
        stderr = (
            "some unrelated traceback line\n"
            f"{uneven_perf.PROBE_ENV_ERROR_PREFIX}sgl_kernel missing in "
            "/fake/python\n"
        )
        with mock.patch(
            "subprocess.run", return_value=self._fake_run(1, stderr)
        ):
            failure = uneven_perf._run_probe_subprocess("/tmp/out.json", None)
        self.assertIsNotNone(failure)
        self.assertTrue(failure.startswith(uneven_perf.PROBE_ENV_ERROR_PREFIX))
        self.assertIn("sgl_kernel missing", failure)

    def test_no_marker_falls_back_to_the_returncode_reason(self):
        with mock.patch(
            "subprocess.run",
            return_value=self._fake_run(1, "a completely unrelated crash\n"),
        ):
            failure = uneven_perf._run_probe_subprocess("/tmp/out.json", None)
        self.assertEqual(failure, "probe subprocess exited with rc=1")


#: The python/ directory of THIS worktree, derived from the loaded module --
#: not hardcoded -- so the subprocess test below runs against the same code
#: this test process imported, wherever the worktree happens to be checked
#: out.
_REPO_PYTHON_DIR = os.path.abspath(
    os.path.join(os.path.dirname(uneven_perf.__file__), "..", "..")
)


def _write_broken_sgl_kernel_shadow(root: str) -> str:
    """A fake ``sgl_kernel`` package, importable as a package but raising
    ImportError from ``__init__.py`` -- deterministically reproduces "sgl_kernel
    not importable" for ANY interpreter, regardless of whether a real
    sgl_kernel build is actually broken or healthy on the machine running the
    test."""
    pkg_dir = os.path.join(root, "sgl_kernel")
    os.makedirs(pkg_dir)
    with open(os.path.join(pkg_dir, "__init__.py"), "w") as f:
        f.write(
            "raise ImportError("
            "'shadow sgl_kernel: deliberately broken for test_probe_"
            "environment_guard')\n"
        )
    return root


class TestProbeSubprocessEntryPoint(CustomTestCase):
    """The real ``python -m sglang.srt.uneven_perf --probe`` entry point, run
    as an actual subprocess (this is "am Sondeneinstieg" -- the entry point
    the launcher's ``_run_probe_subprocess`` spawns): the environment guard
    must abort before ``run_probe`` writes anything, and exit non-zero."""

    def test_broken_sgl_kernel_aborts_before_any_profile_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            shadow = _write_broken_sgl_kernel_shadow(os.path.join(tmp, "shadow"))
            out = os.path.join(tmp, "profile.json")
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([shadow, _REPO_PYTHON_DIR])
            env["CUDA_VISIBLE_DEVICES"] = "99"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sglang.srt.uneven_perf",
                    "--probe",
                    "--out",
                    out,
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=90,
            )
            self.assertNotEqual(proc.returncode, 0, proc.stderr[-2000:])
            self.assertIn(uneven_perf.PROBE_ENV_ERROR_PREFIX, proc.stderr)
            self.assertIn(sys.executable, proc.stderr)
            self.assertFalse(
                os.path.exists(out),
                "the environment guard must fire before any profile write",
            )

    def test_groups_without_lanes_never_invoke_the_guard(self):
        """A top-up that does not touch the 'lanes' group must not be
        refused just because sgl_kernel happens to be broken: the guard is
        scoped to the group it actually protects."""
        with tempfile.TemporaryDirectory() as tmp:
            shadow = _write_broken_sgl_kernel_shadow(os.path.join(tmp, "shadow"))
            out = os.path.join(tmp, "profile.json")
            base = {
                "version": 2,
                "driver": "0.0.0",
                "uuids": [],
                "gpus": {},
                "links": {},
                "probe_seconds": 0.0,
                "created": "2026-07-27 15:34:30",
            }
            with open(out, "w") as f:
                json.dump(base, f)
            env = dict(os.environ)
            env["PYTHONPATH"] = os.pathsep.join([shadow, _REPO_PYTHON_DIR])
            # No visible CUDA devices: the membw group probes zero cards and
            # completes immediately, keeping this test on CPU only.
            env["CUDA_VISIBLE_DEVICES"] = "99"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "sglang.srt.uneven_perf",
                    "--probe",
                    "--out",
                    out,
                    "--groups",
                    "membw",
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=90,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            self.assertNotIn(uneven_perf.PROBE_ENV_ERROR_PREFIX, proc.stderr)


if __name__ == "__main__":
    unittest.main()
