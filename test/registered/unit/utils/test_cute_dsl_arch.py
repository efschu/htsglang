"""A CuTe kernel must be compiled and cached for the rank's OWN GPU (#208).

The observed failure: Llama-3.1-8B, TP=3, no ``--rank-gpu-id``, on a rig with
one 5090 (sm120) and two 3080s (sm86). Boot dies in

    layernorm.py forward_cuda -> sgl_kernel.rmsnorm
      -> flashinfer.norm.rmsnorm_cute -> cutlass tvm_ffi
      -> RuntimeError: CUDA Error: cudaErrorNoKernelImageForDevice

Root cause, in nvidia-cutlass-dsl, not in sglang:

    base_dsl/env_manager.py :: detect_gpu_arch()
      -> base_dsl/runtime/cuda.py :: get_compute_capability_major_minor(
             device_id=0)     # hardcoded 0, ignores the process's device
      -> CuTeDSL.envar.arch

and ``envar.arch`` is BOTH the codegen target (``compile_and_cache`` ->
``preprocess_pipeline``) AND part of the JIT cache key (``get_module_hash``
hashes every ``envar`` attribute, and that hash names the on-disk entry).
sglang TP ranks are separate processes, but only ``--rank-gpu-id`` gives each
one a single visible GPU; without it, driver device 0 is the same card in
every rank, so every rank compiles -- and files its cache entry -- under one
architecture. That is why the 27B boots (they all pass ``--rank-gpu-id``) and
the Llama TP=3 arms do not.

What is pinned here:

 1. RED, the defect modelled: a device-0-only arch lookup returns the SAME
    string for a rank on the 5090 and a rank on a 3080, so both the compile
    target and the cache key collide.
 2. GREEN: ``align_cute_dsl_arch`` gives each rank its own device's arch, in
    the exact string format cutlass itself produces.
 3. The cache keys separate afterwards -- modelled on cutlass's own
    ``get_module_hash``, which hashes the envar attributes.
 4. A singleton constructed BEFORE the call (which is the real situation:
    ``import flashinfer.norm`` builds it) is retargeted in place -- the env
    var alone would be too late.
 5. Homogeneous rigs are a no-op, and an explicit user ``CUTE_DSL_ARCH``
    is never overwritten.

CPU only: no CUDA, no cutlass, no compiler -- torch's capability lookup and
the DSL singleton are both mocked.
"""

import hashlib
import os
import sys
import types
import unittest
from unittest.mock import patch

from sglang.srt.utils.cute_dsl_arch import align_cute_dsl_arch, cute_dsl_arch_for_device
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

# The rig from the bug report, in CUDA device order: 0 = 5090 (sm120),
# 1 and 2 = 3080 (sm86).
FAKE_CAPABILITY = {0: (12, 0), 1: (8, 6), 2: (8, 6)}


def _fake_get_device_capability(device):
    return FAKE_CAPABILITY[int(device)]


def _detect_gpu_arch_device0_only():
    """cutlass's own detection, reduced to the part that is wrong.

    ``detect_gpu_arch()`` formats whatever
    ``get_compute_capability_major_minor()`` returns, and that helper defaults
    to ``device_id=0`` -- the process's actual device never enters.
    """
    major, minor = _fake_get_device_capability(0)
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


class _FakeEnvar:
    """The subset of cutlass's EnvironmentVarManager that keys the cache."""

    def __init__(self, arch):
        self.arch = arch
        self.jit_cache_max_elems = 128
        self.disable_file_caching = False


class _FakeDSL:
    def __init__(self, arch):
        self.name = "CUTE_DSL"
        self.envar = _FakeEnvar(arch)


def _module_hash(dsl, ir_bytes=b"<rmsnorm-ir>"):
    """Model of ``base_dsl/dsl.py :: get_module_hash``.

    The real one writes the module bytecode, then every non-None ``envar``
    attribute, then the compile options into one sha256. Only the envar part
    matters here: the IR for the same kernel is identical on both ranks.
    """
    h = hashlib.sha256()
    h.update(ir_bytes)
    for _, value in sorted(dsl.envar.__dict__.items()):
        if value is not None:
            h.update(str(value).encode())
    return h.hexdigest()


class _FakeSingletonMeta:
    _instances = {}


class CuteDslArchTest(CustomTestCase):
    def setUp(self):
        self._saved_env = os.environ.get("CUTE_DSL_ARCH")
        os.environ.pop("CUTE_DSL_ARCH", None)
        _FakeSingletonMeta._instances = {}

        fake_torch_cuda = types.SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=_fake_get_device_capability,
        )
        fake_torch = types.SimpleNamespace(
            cuda=fake_torch_cuda,
            version=types.SimpleNamespace(cuda="13.0"),
        )
        self._torch_patch = patch.dict(sys.modules, {"torch": fake_torch})
        self._torch_patch.start()

    def tearDown(self):
        self._torch_patch.stop()
        os.environ.pop("CUTE_DSL_ARCH", None)
        if self._saved_env is not None:
            os.environ["CUTE_DSL_ARCH"] = self._saved_env
        sys.modules.pop("cutlass.base_dsl.dsl", None)

    def _install_live_singleton(self, arch):
        """Stand in for the CuTeDSL instance that flashinfer builds at import."""
        dsl = _FakeDSL(arch)
        _FakeSingletonMeta._instances = {"CuTeDSL": dsl}
        module = types.ModuleType("cutlass.base_dsl.dsl")
        module.DSLSingletonMeta = _FakeSingletonMeta
        sys.modules["cutlass.base_dsl.dsl"] = module
        return dsl

    # ------------------------------------------------------------------
    # 1. The defect
    # ------------------------------------------------------------------
    def test_red_device0_only_lookup_collides_across_archs(self):
        """Both ranks get device 0's arch: wrong target AND one cache key."""
        rank0_arch = _detect_gpu_arch_device0_only()
        rank1_arch = _detect_gpu_arch_device0_only()
        self.assertEqual(rank0_arch, rank1_arch)
        self.assertEqual(rank0_arch, "sm_120a")
        # The rank on a 3080 would run an sm_120a image -> the reported
        # cudaErrorNoKernelImageForDevice.
        self.assertNotEqual(rank1_arch, cute_dsl_arch_for_device(1))

        collided = _module_hash(_FakeDSL(rank0_arch))
        self.assertEqual(collided, _module_hash(_FakeDSL(rank1_arch)))

    # ------------------------------------------------------------------
    # 2./3. The fix
    # ------------------------------------------------------------------
    def test_arch_string_follows_the_rank_device(self):
        self.assertEqual(cute_dsl_arch_for_device(0), "sm_120a")
        self.assertEqual(cute_dsl_arch_for_device(1), "sm_86")
        self.assertEqual(cute_dsl_arch_for_device(2), "sm_86")

    def test_arch_string_matches_cutlass_format_on_device0(self):
        """Same card, same string: the fix must not rename an existing key."""
        self.assertEqual(cute_dsl_arch_for_device(0), _detect_gpu_arch_device0_only())

    def test_green_cache_keys_separate_after_alignment(self):
        rank0 = self._install_live_singleton(_detect_gpu_arch_device0_only())
        align_cute_dsl_arch(0)
        key0 = _module_hash(rank0)

        _FakeSingletonMeta._instances = {}
        os.environ.pop("CUTE_DSL_ARCH", None)
        rank1 = self._install_live_singleton(_detect_gpu_arch_device0_only())
        align_cute_dsl_arch(1)
        key1 = _module_hash(rank1)

        self.assertEqual(rank0.envar.arch, "sm_120a")
        self.assertEqual(rank1.envar.arch, "sm_86")
        self.assertNotEqual(key0, key1)

    # ------------------------------------------------------------------
    # 4. Late arrival: the singleton already exists
    # ------------------------------------------------------------------
    def test_existing_singleton_is_retargeted_in_place(self):
        dsl = self._install_live_singleton("sm_120a")
        applied = align_cute_dsl_arch(1)
        self.assertEqual(applied, "sm_86")
        self.assertEqual(
            dsl.envar.arch,
            "sm_86",
            "the singleton built at flashinfer import kept device 0's arch; "
            "setting CUTE_DSL_ARCH alone is too late",
        )
        self.assertEqual(os.environ["CUTE_DSL_ARCH"], "sm_86")

    def test_works_before_cutlass_is_imported(self):
        self.assertNotIn("cutlass.base_dsl.dsl", sys.modules)
        self.assertEqual(align_cute_dsl_arch(1), "sm_86")
        self.assertEqual(os.environ["CUTE_DSL_ARCH"], "sm_86")

    # ------------------------------------------------------------------
    # 5. Do-no-harm
    # ------------------------------------------------------------------
    def test_homogeneous_rig_is_a_no_op(self):
        with patch.dict(FAKE_CAPABILITY, {0: (8, 6), 1: (8, 6)}, clear=True):
            dsl = self._install_live_singleton(_detect_gpu_arch_device0_only())
            before = _module_hash(dsl)
            align_cute_dsl_arch(1)
            self.assertEqual(dsl.envar.arch, "sm_86")
            self.assertEqual(_module_hash(dsl), before)

    def test_user_override_is_not_overwritten(self):
        os.environ["CUTE_DSL_ARCH"] = "sm_90a"
        dsl = self._install_live_singleton("sm_120a")
        self.assertEqual(align_cute_dsl_arch(1), "sm_90a")
        self.assertEqual(os.environ["CUTE_DSL_ARCH"], "sm_90a")
        self.assertEqual(dsl.envar.arch, "sm_90a")

    def test_no_cuda_is_tolerated(self):
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: False,
                get_device_capability=_fake_get_device_capability,
            ),
            version=types.SimpleNamespace(cuda=None),
        )
        with patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertIsNone(cute_dsl_arch_for_device(0))
            self.assertIsNone(align_cute_dsl_arch(0))
        self.assertNotIn("CUTE_DSL_ARCH", os.environ)


if __name__ == "__main__":
    unittest.main()
