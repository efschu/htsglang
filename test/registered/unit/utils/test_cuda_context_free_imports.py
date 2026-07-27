"""Task #237: GPU-passive processes must not create a CUDA context.

The launch_server parent (tokenizer manager + HTTP server) held a ~634 MiB
CUDA context because import-time code answered device-capability questions
through torch.cuda (get_device_capability / get_device_properties run
_lazy_init, which maps a full context). These tests pin the context-free
paths:

  * common.py no longer imports sgl_kernel at module scope (sgl_kernel's
    loader queries device properties at import),
  * capability questions are answered via NVML while torch.cuda is
    uninitialized (_nvml_cuda_device0 / get_device_capability_no_init),
  * the whole launch_server parent import closure performs ZERO first-party
    torch.cuda device queries (externals are reported but not gated here).

No GPU is required: the subprocess probes stub every torch.cuda entry point
that could initialize CUDA, and NVML is replaced by an injected fake pynvml,
so device enumeration is deterministic and no real device is touched even on
a machine with GPUs. Subprocesses therefore control CUDA_VISIBLE_DEVICES
themselves (the fake NVML path must parse a clean value).
"""

import os
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=200, suite="base-a-test-cpu")

REPO_PYTHON_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "python")
)


def make_fake_pynvml(devices):
    """devices: list of (major, minor, name, uuid, bus_id)."""

    def handle(i):
        return devices[i]

    return SimpleNamespace(
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetCount=lambda: len(devices),
        nvmlDeviceGetHandleByIndex=handle,
        nvmlDeviceGetCudaComputeCapability=lambda d: (d[0], d[1]),
        nvmlDeviceGetName=lambda d: d[2],
        nvmlDeviceGetUUID=lambda d: d[3],
        nvmlDeviceGetPciInfo=lambda d: SimpleNamespace(busId=d[4]),
    )


MIXED_RIG = [
    (8, 6, "NVIDIA GeForce RTX 3080", "GPU-aaaa", "0000:01:00.0"),
    (12, 0, "NVIDIA GeForce RTX 5090", "GPU-bbbb", "0000:02:00.0"),
    (8, 6, "NVIDIA GeForce RTX 3080", "GPU-cccc", "0000:03:00.0"),
]

# stubs shared by the subprocess probes: emulate an available GPU while making
# every context-creating torch.cuda entry point either record or fail loudly
SUBPROCESS_PREAMBLE = r"""
import os, sys, traceback
from types import SimpleNamespace

devices = [
    (8, 6, "NVIDIA GeForce RTX 3080", "GPU-aaaa", "0000:01:00.0"),
    (12, 0, "NVIDIA GeForce RTX 5090", "GPU-bbbb", "0000:02:00.0"),
    (8, 6, "NVIDIA GeForce RTX 3080", "GPU-cccc", "0000:03:00.0"),
]
def _handle(i):
    return devices[i]
sys.modules["pynvml"] = SimpleNamespace(
    nvmlInit=lambda: None,
    nvmlShutdown=lambda: None,
    nvmlDeviceGetCount=lambda: len(devices),
    nvmlDeviceGetHandleByIndex=_handle,
    nvmlDeviceGetCudaComputeCapability=lambda d: (d[0], d[1]),
    nvmlDeviceGetName=lambda d: d[2],
    nvmlDeviceGetUUID=lambda d: d[3],
    nvmlDeviceGetPciInfo=lambda d: SimpleNamespace(busId=d[4]),
)

os.environ.pop("CUDA_VISIBLE_DEVICES", None)
os.environ.pop("CUDA_DEVICE_ORDER", None)

import torch

touches = []
def _record(kind):
    frames = [
        f"{f.filename}:{f.lineno}"
        for f in traceback.extract_stack()[:-2]
        if "/torch/" not in f.filename and "importlib" not in f.filename
    ]
    touches.append((kind, frames[-3:] if frames else []))

torch.cuda.is_available = lambda: True
torch.cuda.device_count = lambda: 1
torch.cuda.current_device = lambda: 0
torch.cuda._lazy_init = lambda *a, **k: _record("_lazy_init")
torch.cuda.get_device_capability = lambda device=None: (_record("capability"), (12, 0))[1]
class _P:
    major, minor = 12, 0
    total_memory = 32 << 30
    multi_processor_count = 100
    name = "EMULATED"
torch.cuda.get_device_properties = lambda device=None: (_record("properties"), _P())[1]
torch.cuda.get_device_name = lambda device=None: (_record("name"), "EMULATED")[1]

def first_party_touches():
    out = []
    for kind, frames in touches:
        inner = frames[-1] if frames else "?"
        if "/site-packages/" not in inner and "<string>" not in inner:
            out.append((kind, inner))
    return out
"""


def run_probe(body: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_PYTHON_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-c", SUBPROCESS_PREAMBLE + body],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )


class TestCommonNoModuleScopeSglKernel(unittest.TestCase):
    def test_import_common_does_not_import_sgl_kernel(self):
        # sgl_kernel's loader queries device properties at import; common.py
        # must therefore not import it at module scope.
        proc = run_probe(
            "import sglang.srt.utils.common\n"
            'assert "sgl_kernel" not in sys.modules, "common.py imported sgl_kernel"\n'
            'print("OK")\n'
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
        self.assertIn("OK", proc.stdout)


class TestNvmlDevice0(unittest.TestCase):
    def setUp(self):
        import sglang.srt.utils.common as common

        self.common = common
        common._nvml_cuda_device0.cache_clear()

    def tearDown(self):
        self.common._nvml_cuda_device0.cache_clear()

    def _resolve(self, devices, env):
        fake = make_fake_pynvml(devices)
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
        }
        clean_env.update(env)
        with mock.patch.dict(sys.modules, {"pynvml": fake}):
            with mock.patch.dict(os.environ, clean_env, clear=True):
                self.common._nvml_cuda_device0.cache_clear()
                result = self.common._nvml_cuda_device0()
        self.common._nvml_cuda_device0.cache_clear()
        return result

    def test_fastest_first_picks_max_capability(self):
        info = self._resolve(MIXED_RIG, {})
        self.assertEqual(info.capability, (12, 0))
        self.assertIn("5090", info.name)

    def test_pci_order_uses_bus_id(self):
        info = self._resolve(MIXED_RIG, {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"})
        self.assertEqual(info.capability, (8, 6))

    def test_cvd_uuid_selects_exact_device(self):
        info = self._resolve(MIXED_RIG, {"CUDA_VISIBLE_DEVICES": "GPU-aaaa,GPU-bbbb"})
        self.assertEqual(info.capability, (8, 6))

    def test_cvd_nonzero_index_on_mixed_rig_is_refused(self):
        # FASTEST_FIRST specifies only position 0; an index into the
        # unspecified remainder of a mixed rig must fall back to torch.
        self.assertIsNone(self._resolve(MIXED_RIG, {"CUDA_VISIBLE_DEVICES": "1"}))

    def test_cvd_nonzero_index_on_uniform_rig_is_answered(self):
        uniform = [MIXED_RIG[0], MIXED_RIG[2]]
        info = self._resolve(uniform, {"CUDA_VISIBLE_DEVICES": "1"})
        self.assertEqual(info.capability, (8, 6))

    def test_cvd_out_of_range_is_refused(self):
        self.assertIsNone(self._resolve(MIXED_RIG, {"CUDA_VISIBLE_DEVICES": "99"}))

    def test_nvml_failure_is_refused(self):
        broken = SimpleNamespace(nvmlInit=mock.Mock(side_effect=RuntimeError))
        with mock.patch.dict(sys.modules, {"pynvml": broken}):
            self.common._nvml_cuda_device0.cache_clear()
            self.assertIsNone(self.common._nvml_cuda_device0())


class TestPredicatesContextFree(unittest.TestCase):
    """The capability predicates and get_device_sm must answer via NVML while
    torch.cuda is uninitialized -- touching torch.cuda.get_device_capability
    fails the test."""

    def _context(self):
        import sglang.srt.utils.common as common

        fake = make_fake_pynvml(MIXED_RIG)
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
        }

        def forbidden(device=None):
            raise AssertionError("torch.cuda.get_device_capability was called")

        import torch

        return common, [
            mock.patch.dict(sys.modules, {"pynvml": fake}),
            mock.patch.dict(os.environ, clean_env, clear=True),
            mock.patch.object(torch.cuda, "is_available", lambda: True),
            mock.patch.object(torch.cuda, "is_initialized", lambda: False),
            mock.patch.object(torch.cuda, "get_device_capability", forbidden),
            mock.patch.object(torch.version, "cuda", "12.8"),
        ]

    def _run(self, fn):
        common, patches = self._context()
        common._nvml_cuda_device0.cache_clear()
        common.is_cuda.cache_clear()
        try:
            for p in patches:
                p.start()
            return fn(common)
        finally:
            for p in reversed(patches):
                p.stop()
            common._nvml_cuda_device0.cache_clear()
            common.is_cuda.cache_clear()

    def test_check_cuda_device_version_uses_nvml(self):
        # fake rig's device 0 (FASTEST_FIRST) is the 5090 -> major 12
        self.assertTrue(
            self._run(
                lambda c: c._check_cuda_device_version(
                    device_capability_majors=[10, 11, 12], cuda_version=(12, 8)
                )
            )
        )
        self.assertFalse(
            self._run(
                lambda c: c._check_cuda_device_version(
                    device_capability_majors=[9], cuda_version=(12, 3)
                )
            )
        )

    def test_get_device_sm_uses_nvml(self):
        self.assertEqual(self._run(lambda c: c.get_device_sm()), 120)

    def test_get_device_capability_util_uses_nvml(self):
        self.assertEqual(
            self._run(lambda c: c.get_device_capability(0)), (12, 0)
        )


class TestSglKernelLoaderContextFree(unittest.TestCase):
    def test_loader_capability_via_nvml(self):
        # load the repo's sgl-kernel loader file standalone (the installed
        # wheel must not shadow it)
        import importlib.util

        path = os.path.join(
            REPO_PYTHON_ROOT,
            "..",
            "sgl-kernel",
            "python",
            "sgl_kernel",
            "load_utils.py",
        )
        spec = importlib.util.spec_from_file_location("_sgl_kernel_load_utils", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        import torch

        fake = make_fake_pynvml(MIXED_RIG)
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER")
        }

        def forbidden(device=None):
            raise AssertionError("torch.cuda.get_device_properties was called")

        with mock.patch.dict(sys.modules, {"pynvml": fake}):
            with mock.patch.dict(os.environ, clean_env, clear=True):
                with mock.patch.object(torch.cuda, "is_available", lambda: True):
                    with mock.patch.object(
                        torch.cuda, "is_initialized", lambda: False
                    ):
                        with mock.patch.object(
                            torch.cuda, "get_device_properties", forbidden
                        ):
                            self.assertEqual(module._get_compute_capability(), 120)


class TestParentImportClosureContextFree(unittest.TestCase):
    def test_http_server_closure_has_no_first_party_device_touch(self):
        # THE falsifier for task #237, as a regression test: importing the
        # launch_server parent's module must not run any first-party
        # import-time torch.cuda device query. External residuals
        # (torchao via transformers, the installed sgl_kernel wheel until it
        # is rebuilt with the NVML loader) are printed but not gated.
        proc = run_probe(
            "import sglang.srt.entrypoints.http_server\n"
            "fp = first_party_touches()\n"
            "for kind, site in fp:\n"
            "    print('FIRST-PARTY', kind, site)\n"
            "print('external touches:', len(touches) - len(fp))\n"
            'assert not fp, f"first-party device touches at import: {fp}"\n'
            'print("OK")\n'
        )
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-3000:])
        self.assertIn("OK", proc.stdout)

    def test_fla_utils_context_free_and_platform_correct(self):
        # fla/utils.py is in server_args' import closure; its module scope
        # answered platform/capability questions through triton->torch and
        # torch.cuda directly. Context-free now, with unchanged answers.
        proc = run_probe(
            "import sglang.srt.layers.attention.fla.utils as fla\n"
            'assert fla.get_available_device() == "cuda", fla.get_available_device()\n'
            "assert fla.is_nvidia, (fla.device_platform, fla.is_nvidia)\n"
            "assert fla.is_nvidia_hopper, 'cap (12,0) >= 9 expected'\n"
            "assert fla.is_tf32_supported\n"
            "fp = first_party_touches()\n"
            'assert not fp, f"fla touched the device at import: {fp}"\n'
            'print("OK")\n'
        )
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-3000:])
        self.assertIn("OK", proc.stdout)

    def test_moe_scheme_modules_import_flashinfer_lazily(self):
        # importing flashinfer's fused_moe/fp4 machinery evaluates modules
        # that query device properties -> the MoE quantization schemes must
        # not bind flashinfer symbols at module scope. (The installed
        # sgl_kernel wheel still triggers a flashinfer cascade of its own
        # until it is rebuilt; that shows up as external touches and is
        # deliberately not gated here.)
        proc = run_probe(
            "import sglang.srt.layers.quantization.compressed_tensors.schemes"
            ".compressed_tensors_w4a4_mxint4_moe as mxint4\n"
            "import sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe as mxfp4\n"
            "for mod, name in [\n"
            "    (mxint4, 'trtllm_mxint4_block_scale_moe'),\n"
            "    (mxint4, 'block_scale_interleave'),\n"
            "    (mxfp4, 'trtllm_fp4_block_scale_routed_moe'),\n"
            "    (mxfp4, 'mxfp8_quantize'),\n"
            "]:\n"
            "    assert not hasattr(mod, name), f'{mod.__name__} binds {name} at module scope'\n"
            "fp = first_party_touches()\n"
            'assert not fp, f"device touches at import: {fp}"\n'
            'print("OK")\n'
        )
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-3000:])
        self.assertIn("OK", proc.stdout)


class TestCudaInitTracer(unittest.TestCase):
    def test_emulate_mode_records_stack(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ)
            env["PYTHONPATH"] = REPO_PYTHON_ROOT + os.pathsep + env.get(
                "PYTHONPATH", ""
            )
            code = (
                "import importlib.util, os, sys\n"
                f"path = os.path.join({REPO_PYTHON_ROOT!r}, 'sglang', 'srt', 'utils', 'cuda_init_tracer.py')\n"
                "spec = importlib.util.spec_from_file_location('tracer', path)\n"
                "tracer = importlib.util.module_from_spec(spec)\n"
                "spec.loader.exec_module(tracer)\n"
                f"log = tracer.install(out_dir={tmp!r}, emulate=True)\n"
                "import torch\n"
                "cap = torch.cuda.get_device_capability()\n"
                "assert cap == (9, 0), cap\n"
                "assert torch.cuda.is_available()\n"
                "content = open(log).read()\n"
                "assert 'get_device_capability' in content, content\n"
                "print('OK')\n"
            )
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                env=env,
                timeout=300,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
            self.assertIn("OK", proc.stdout)


if __name__ == "__main__":
    unittest.main()
