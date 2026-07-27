"""CPU unit tests for the co-location NCCL env handling (phase 4).

_configure_nccl_env_for_colocation() runs in the parent process right
before the scheduler processes are spawned, so the env vars are
inherited by every worker. It must:
- do nothing unless --rank-gpu-id contains duplicates (co-location),
- set NCCL_MULTI_RANK_GPU_ENABLE=1, NCCL_NVLS_ENABLE=0 (with warning)
  and a heuristic NCCL_MAX_CTAS cap,
- never override values the user has already exported.

No GPU, no process spawn: the helper only reads server_args.rank_gpu_id
and os.environ. `sgl_kernel` is stubbed before the sglang imports.
"""

import logging
import os
import importlib.util
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


def _install_sgl_kernel_stub():
    if importlib.util.find_spec("sgl_kernel") is not None:
        # The real package is importable; stubbing it here would leave a
        # process-wide empty-__path__ package that breaks every later
        # ``import sgl_kernel.<submodule>`` in the same pytest run.
        return
    def _make(name, pkg=False):
        mod = types.ModuleType(name)
        if pkg:
            mod.__path__ = []

        def _getattr(attr):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return lambda *a, **k: None

        mod.__getattr__ = _getattr
        sys.modules.setdefault(name, mod)

    _make("sgl_kernel", pkg=True)
    _make("sgl_kernel.quantization")
    _make("sgl_kernel.kvcacheio")


_install_sgl_kernel_stub()

from sglang.srt.entrypoints.engine import (  # noqa: E402
    _configure_nccl_env_for_colocation,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

NCCL_KEYS = ("NCCL_MULTI_RANK_GPU_ENABLE", "NCCL_NVLS_ENABLE", "NCCL_MAX_CTAS")

ENGINE_LOGGER = "sglang.srt.entrypoints.engine"


def make_args(rank_gpu_id):
    # The helper only reads .rank_gpu_id; a namespace keeps the test
    # independent of ServerArgs' __post_init__ NVML queries.
    return SimpleNamespace(rank_gpu_id=rank_gpu_id)


class ColocationNCCLEnvTest(CustomTestCase):
    def setUp(self):
        # patch.dict restores the FULL original environ on exit, so tests
        # may freely pop/set keys inside.
        self._env_patcher = patch.dict(os.environ, {}, clear=False)
        self._env_patcher.start()
        for key in NCCL_KEYS:
            os.environ.pop(key, None)
        # Point the MPS check at an existing dir by default so the
        # (separately tested) MPS warning does not fire in every test.
        self._mps_dir = tempfile.mkdtemp(prefix="fake-mps-")
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = self._mps_dir

    def tearDown(self):
        self._env_patcher.stop()
        os.rmdir(self._mps_dir)

    def test_default_path_no_rank_gpu_id(self):
        _configure_nccl_env_for_colocation(make_args(None))
        for key in NCCL_KEYS:
            self.assertNotIn(key, os.environ)

    def test_no_duplicates_is_noop(self):
        _configure_nccl_env_for_colocation(make_args([0, 1, 2]))
        for key in NCCL_KEYS:
            self.assertNotIn(key, os.environ)

    def test_duplicates_set_all_three(self):
        with self.assertLogs(ENGINE_LOGGER, level=logging.WARNING) as logs:
            _configure_nccl_env_for_colocation(make_args([0, 0, 1, 2]))
        self.assertEqual(os.environ["NCCL_MULTI_RANK_GPU_ENABLE"], "1")
        self.assertEqual(os.environ["NCCL_NVLS_ENABLE"], "0")
        # max_colocated=2 -> max(1, 8 // 2) = 4
        self.assertEqual(os.environ["NCCL_MAX_CTAS"], "4")
        self.assertTrue(any("NCCL_NVLS_ENABLE=0" in m for m in logs.output))

    def test_max_ctas_heuristic_scales_with_colocation_degree(self):
        _configure_nccl_env_for_colocation(make_args([0, 0, 0, 0]))
        self.assertEqual(os.environ["NCCL_MAX_CTAS"], "2")

    def test_max_ctas_heuristic_floor_is_one(self):
        _configure_nccl_env_for_colocation(make_args([0] * 16))
        self.assertEqual(os.environ["NCCL_MAX_CTAS"], "1")

    def test_user_values_win(self):
        os.environ["NCCL_MULTI_RANK_GPU_ENABLE"] = "0"
        os.environ["NCCL_NVLS_ENABLE"] = "2"
        os.environ["NCCL_MAX_CTAS"] = "16"
        _configure_nccl_env_for_colocation(make_args([0, 0]))
        self.assertEqual(os.environ["NCCL_MULTI_RANK_GPU_ENABLE"], "0")
        self.assertEqual(os.environ["NCCL_NVLS_ENABLE"], "2")
        self.assertEqual(os.environ["NCCL_MAX_CTAS"], "16")

    def test_partial_user_override(self):
        os.environ["NCCL_MAX_CTAS"] = "32"
        _configure_nccl_env_for_colocation(make_args([0, 0, 1]))
        self.assertEqual(os.environ["NCCL_MAX_CTAS"], "32")
        self.assertEqual(os.environ["NCCL_MULTI_RANK_GPU_ENABLE"], "1")
        self.assertEqual(os.environ["NCCL_NVLS_ENABLE"], "0")

    def test_mps_warning_when_pipe_dir_missing(self):
        os.environ["CUDA_MPS_PIPE_DIRECTORY"] = "/nonexistent-mps-pipe-dir"
        with self.assertLogs(ENGINE_LOGGER, level=logging.WARNING) as logs:
            _configure_nccl_env_for_colocation(make_args([0, 0]))
        self.assertTrue(any("MPS" in m for m in logs.output))

    def test_no_mps_warning_when_pipe_dir_exists(self):
        os.environ.pop("NCCL_NVLS_ENABLE", None)
        with self.assertLogs(ENGINE_LOGGER, level=logging.WARNING) as logs:
            _configure_nccl_env_for_colocation(make_args([0, 0]))
        self.assertFalse(any("MPS" in m for m in logs.output))


if __name__ == "__main__":
    unittest.main()
